//  fused_training.metal
//  Fused training kernels for Apple Silicon
//
//  These kernels enable single-command-buffer training steps by fusing:
//  1. AdamW optimizer update (all parameters in one dispatch)
//  2. Gradient clipping (global norm computation + scaling)
//
//  Key insight: MLX's mx.compile achieves ~2400 tok/s by batching operations
//  into a single Metal command buffer. We replicate this by:
//  - Processing all parameters in parallel within single kernels
//  - Using threadgroup memory for reductions
//  - Eliminating per-kernel GPU-CPU synchronization
//
//  Performance target: ~2400 tok/s (currently at ~1740)

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Configuration
// =============================================================================

constant uint WARP_SIZE = 32;
constant uint MAX_PARAMS_PER_DISPATCH = 1024;  // Handle batches of parameters

// =============================================================================
// AdamW Optimizer Structures
// =============================================================================

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

/// Parameter metadata for batched processing
struct ParamInfo {
    uint offset;            // Offset into the flattened parameter buffer
    uint size;              // Number of elements in this parameter
    uint m_offset;          // Offset into first moment buffer
    uint v_offset;          // Offset into second moment buffer
    uint group;             // Index of this parameter's AdamWScalars
};

// =============================================================================
// Fused AdamW Optimizer Kernel
// =============================================================================

/// Fused AdamW update for ALL parameters in a single dispatch.
///
/// Instead of launching N separate kernels for N parameters, this kernel
/// processes all parameters in parallel using a 2D grid:
/// - X dimension: elements within a parameter (parallel across elements)
/// - Y dimension: different parameters (parallel across params)
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
///
/// The update is torch.optim.AdamW's, in its order: decoupled weight decay on
/// the old parameter, then the bias-corrected Adam step.
///   p = p * (1 - lr * wd)
///   m = m + (1 - beta1) * (g - m)
///   v = beta2 * v + (1 - beta2) * g^2
///   p = p - lr / (1 - beta1^t) * m / (sqrt(v) / sqrt(1 - beta2^t) + eps)
///
/// Grid: [ceil(max_param_size / WARP_SIZE), num_params, 1]
/// Threadgroup: [WARP_SIZE, 1, 1]
kernel void fused_adamw_update(
    device float* params [[buffer(0)]],         // All parameters (in-place update)
    device const float* grads [[buffer(1)]],    // All gradients
    device float* m [[buffer(2)]],              // First moments (in-place update)
    device float* v [[buffer(3)]],              // Second moments (in-place update)
    device const ParamInfo* param_info [[buffer(4)]],  // Parameter metadata
    constant AdamWScalars* groups [[buffer(5)]],       // Scalars per group
    constant uint& num_params [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Which parameter are we processing?
    const uint param_idx = tgid.y;
    if (param_idx >= num_params) return;

    // Get parameter metadata
    const ParamInfo info = param_info[param_idx];

    // Which element within this parameter?
    const uint elem_idx = tid.x;
    if (elem_idx >= info.size) return;

    // Global indices
    const uint p_idx = info.offset + elem_idx;
    const uint m_idx = info.m_offset + elem_idx;
    const uint v_idx = info.v_offset + elem_idx;
    const AdamWScalars s = groups[info.group];

    const float g = grads[p_idx];
    const float p = params[p_idx] * s.decay;
    const float m_val = m[m_idx] + s.one_minus_beta1 * (g - m[m_idx]);
    const float v_val = s.beta2 * v[v_idx] + s.one_minus_beta2 * g * g;
    const float denom = sqrt(v_val) / s.bias_correction2_sqrt + s.epsilon;

    params[p_idx] = p - s.step_size * (m_val / denom);
    m[m_idx] = m_val;
    v[v_idx] = v_val;
}

// =============================================================================
// Gradient Clipping Kernels
// =============================================================================

/// Compute global gradient norm squared (partial reduction).
///
/// Each threadgroup computes partial sum of grad^2, stored to partial_sums.
/// Final reduction happens on CPU or with a follow-up kernel.
///
/// Grid: [num_threadgroups, 1, 1]
/// Threadgroup: [256, 1, 1]
kernel void gradient_norm_squared_partial(
    device const float* grads [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Threadgroup-local reduction buffer
    threadgroup float shared_sum[256];

    // Each thread accumulates 4 elements via float4 vectorized load + dot product
    float local_sum = 0.0f;
    uint base = tgid * tg_size * 4 + tid * 4;

    if (base + 3 < total_elements) {
        // Full float4 load — 128-bit aligned for maximum bandwidth
        float4 g4 = *(device const float4*)(grads + base);
        local_sum = dot(g4, g4);
    } else {
        // Scalar fallback for boundary
        for (uint i = 0; i < 4; i++) {
            uint idx = base + i;
            if (idx < total_elements) {
                float g = grads[idx];
                local_sum += g * g;
            }
        }
    }

    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within threadgroup
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread writes result
    if (tid == 0) {
        partial_sums[tgid] = shared_sum[0];
    }
}

/// Scale all gradients by a factor (for gradient clipping).
///
/// After computing total norm, if norm > max_norm:
///   scale = max_norm / norm
///   grads = grads * scale
///
/// Grid: [ceil(total_elements / WARP_SIZE), 1, 1]
kernel void scale_gradients(
    device float* grads [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Process 4 elements per thread via float4 for 4x bandwidth
    uint base = tid * 4;
    if (base + 3 < total_elements) {
        float4 g4 = *(device const float4*)(grads + base);
        *(device float4*)(grads + base) = g4 * scale;
    } else {
        // Scalar fallback for remainder
        for (uint i = base; i < min(base + 4u, total_elements); i++) {
            grads[i] *= scale;
        }
    }
}

/// Half-precision gradient scaling
kernel void scale_gradients_f16(
    device half* grads [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Process 4 elements per thread via half4 for 4x bandwidth
    uint base = tid * 4;
    if (base + 3 < total_elements) {
        half4 g4 = *(device const half4*)(grads + base);
        *(device half4*)(grads + base) = half4(float4(g4) * scale);
    } else {
        for (uint i = base; i < min(base + 4u, total_elements); i++) {
            grads[i] = half(float(grads[i]) * scale);
        }
    }
}

// =============================================================================
// Batched Command Buffer Support
// =============================================================================

/// Marker kernel for synchronization points (used for debugging/profiling).
/// This kernel does nothing but creates a synchronization point in the command buffer.
kernel void sync_marker(
    device uint* marker [[buffer(0)]],
    constant uint& value [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        marker[0] = value;
    }
}
