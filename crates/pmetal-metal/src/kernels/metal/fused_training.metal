//  fused_training.metal
//  Fused AdamW optimizer step for Apple Silicon.
//
//  One dispatch updates every parameter of a model. The update is
//  bandwidth-bound (28 bytes of traffic per element), so the kernel's job is
//  to keep the memory system busy: the flattened parameter space is cut into
//  fixed-size chunks, one threadgroup per chunk, so a model with one huge
//  embedding and hundreds of small tensors launches the same number of
//  threadgroups per element as any other.

#include <metal_stdlib>
using namespace metal;

// Elements one threadgroup updates. `ADAMW_CHUNK` in fused_training.rs.
constant uint ADAMW_CHUNK = 4096;

/// One AdamW step's scalars, computed on the host as torch.optim.AdamW
/// computes them (in f64, each rounded to f32 once). `AdamWScalars` in
/// fused_training.rs.
struct AdamWScalars {
    float decay;                  // 1 - lr * weight_decay
    float one_minus_beta1;        // 1 - beta1
    float beta2;
    float one_minus_beta2;        // 1 - beta2
    float step_size;              // lr / (1 - beta1^t)
    float bias_correction2_sqrt;  // sqrt(1 - beta2^t)
    float epsilon;
};

/// Parameter metadata for batched processing. `ParamInfo` in fused_training.rs.
struct ParamInfo {
    uint offset;            // Offset into the flattened parameter buffer
    uint size;              // Number of elements in this parameter
    uint m_offset;          // Offset into first moment buffer
    uint v_offset;          // Offset into second moment buffer
    uint group;             // Index of this parameter's AdamWScalars
};

/// A run of up to ADAMW_CHUNK elements of one parameter. `AdamWChunk` in
/// fused_training.rs.
struct AdamWChunk {
    uint param;             // Index into param_info
    uint start;             // First element of the run, within the parameter
};

/// Fused AdamW update for ALL parameters in a single dispatch.
///
/// Memory layout:
/// - params: Flattened buffer containing all model parameters [total_elements]
/// - grads: Flattened buffer containing all gradients [total_elements]
/// - m: First moment estimates [total_elements]
/// - v: Second moment estimates [total_elements]
/// - param_info: Metadata for each parameter [num_params]
/// - groups: One step's scalars per parameter group; a parameter reads
///   groups[param_info.group], so groups train at their own learning rate
///   and weight decay, as torch.optim's param_groups do
/// - chunks: The runs of elements, one per threadgroup
///
/// The update is torch.optim.AdamW's, in its order: decoupled weight decay on
/// the old parameter, then the bias-corrected Adam step.
///   p = p * (1 - lr * wd)
///   m = m + (1 - beta1) * (g - m)
///   v = beta2 * v + (1 - beta2) * g^2
///   p = p - lr / (1 - beta1^t) * m / (sqrt(v) / sqrt(1 - beta2^t) + eps)
///
/// Grid: [num_chunks, 1, 1]
/// Threadgroup: [256, 1, 1]
kernel void fused_adamw_update(
    device float* params [[buffer(0)]],         // All parameters (in-place update)
    device const float* grads [[buffer(1)]],    // All gradients
    device float* m [[buffer(2)]],              // First moments (in-place update)
    device float* v [[buffer(3)]],              // Second moments (in-place update)
    device const ParamInfo* param_info [[buffer(4)]],  // Parameter metadata
    constant AdamWScalars* groups [[buffer(5)]],       // Scalars per group
    device const AdamWChunk* chunks [[buffer(6)]],     // The runs of elements
    uint chunk_id [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    const AdamWChunk chunk = chunks[chunk_id];
    const ParamInfo info = param_info[chunk.param];
    const AdamWScalars s = groups[info.group];
    const uint end = min(chunk.start + ADAMW_CHUNK, info.size);

    // Consecutive threads take consecutive elements, so a simdgroup reads and
    // writes whole cache lines.
    for (uint elem = chunk.start + lid; elem < end; elem += threads) {
        const uint p_idx = info.offset + elem;
        const uint m_idx = info.m_offset + elem;
        const uint v_idx = info.v_offset + elem;

        const float g = grads[p_idx];
        const float p = params[p_idx] * s.decay;
        const float m_old = m[m_idx];
        const float m_val = m_old + s.one_minus_beta1 * (g - m_old);
        const float v_val = s.beta2 * v[v_idx] + s.one_minus_beta2 * g * g;
        const float denom = sqrt(v_val) / s.bias_correction2_sqrt + s.epsilon;

        params[p_idx] = p - s.step_size * (m_val / denom);
        m[m_idx] = m_val;
        v[v_idx] = v_val;
    }
}
