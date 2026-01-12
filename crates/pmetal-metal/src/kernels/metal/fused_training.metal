//  fused_training.metal
//  Fused training kernels for Apple Silicon
//
//  These kernels enable single-command-buffer training steps by fusing:
//  1. AdamW optimizer update (all parameters in one dispatch)
//  2. Gradient clipping (global norm computation + scaling)
//  3. Cross-entropy loss with backward pass
//
//  Key insight: MLX's mx.compile achieves ~2400 tok/s by batching operations
//  into a single Metal command buffer. We replicate this by:
//  - Processing all parameters in parallel within single kernels
//  - Using threadgroup memory for reductions
//  - Eliminating per-kernel GPU-CPU synchronization
//
//  Performance target: Match mlx_lm's ~2400 tok/s (currently at ~1740)

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

/// AdamW hyperparameters (shared across all parameters)
struct AdamWConfig {
    float learning_rate;    // Current learning rate (after scheduling)
    float beta1;            // First moment decay (default 0.9)
    float beta2;            // Second moment decay (default 0.999)
    float epsilon;          // Numerical stability (default 1e-8)
    float weight_decay;     // L2 regularization (default 0.01)
    uint step;              // Current optimization step (for bias correction)
};

/// Parameter metadata for batched processing
struct ParamInfo {
    uint offset;            // Offset into the flattened parameter buffer
    uint size;              // Number of elements in this parameter
    uint m_offset;          // Offset into first moment buffer
    uint v_offset;          // Offset into second moment buffer
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
///
/// The kernel computes (matching mlx-rs's AdamW without bias correction):
///   m = beta1 * m + (1 - beta1) * grad
///   v = beta2 * v + (1 - beta2) * grad^2
///   param = param * (1 - lr * weight_decay) - lr * m / (sqrt(v) + eps)
///
/// Note: Unlike PyTorch's AdamW, mlx-rs does NOT use bias correction.
/// This matches the original decoupled weight decay paper's formulation.
///
/// Grid: [ceil(max_param_size / WARP_SIZE), num_params, 1]
/// Threadgroup: [WARP_SIZE, 1, 1]
kernel void fused_adamw_update(
    device float* params [[buffer(0)]],         // All parameters (in-place update)
    device const float* grads [[buffer(1)]],    // All gradients
    device float* m [[buffer(2)]],              // First moments (in-place update)
    device float* v [[buffer(3)]],              // Second moments (in-place update)
    device const ParamInfo* param_info [[buffer(4)]],  // Parameter metadata
    constant AdamWConfig& config [[buffer(5)]],
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

    // Load values
    float param_val = params[p_idx];
    float grad_val = grads[p_idx];
    float m_val = m[m_idx];
    float v_val = v[v_idx];

    // AdamW update (mlx-rs style - NO bias correction)
    // m = beta1 * m + (1 - beta1) * grad
    m_val = config.beta1 * m_val + (1.0f - config.beta1) * grad_val;

    // v = beta2 * v + (1 - beta2) * grad^2
    v_val = config.beta2 * v_val + (1.0f - config.beta2) * grad_val * grad_val;

    // AdamW decoupled weight decay + Adam update (NO bias correction)
    // param = param * (1 - lr * weight_decay) - lr * m / (sqrt(v) + eps)
    float update = m_val / (sqrt(v_val) + config.epsilon);
    param_val = param_val * (1.0f - config.learning_rate * config.weight_decay)
                - config.learning_rate * update;

    // Write back
    params[p_idx] = param_val;
    m[m_idx] = m_val;
    v[v_idx] = v_val;
}

/// Half-precision version for memory efficiency
kernel void fused_adamw_update_f16(
    device half* params [[buffer(0)]],
    device const half* grads [[buffer(1)]],
    device float* m [[buffer(2)]],              // Keep moments in fp32
    device float* v [[buffer(3)]],
    device const ParamInfo* param_info [[buffer(4)]],
    constant AdamWConfig& config [[buffer(5)]],
    constant uint& num_params [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint param_idx = tgid.y;
    if (param_idx >= num_params) return;

    const ParamInfo info = param_info[param_idx];
    const uint elem_idx = tid.x;
    if (elem_idx >= info.size) return;

    const uint p_idx = info.offset + elem_idx;
    const uint m_idx = info.m_offset + elem_idx;
    const uint v_idx = info.v_offset + elem_idx;

    // Load and convert to fp32 for computation
    float param_val = float(params[p_idx]);
    float grad_val = float(grads[p_idx]);
    float m_val = m[m_idx];
    float v_val = v[v_idx];

    // AdamW update (mlx-rs style - NO bias correction)
    m_val = config.beta1 * m_val + (1.0f - config.beta1) * grad_val;
    v_val = config.beta2 * v_val + (1.0f - config.beta2) * grad_val * grad_val;

    // NO bias correction - matching mlx-rs
    float update = m_val / (sqrt(v_val) + config.epsilon);
    param_val = param_val * (1.0f - config.learning_rate * config.weight_decay)
                - config.learning_rate * update;

    // Write back (convert param to fp16)
    params[p_idx] = half(param_val);
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

    // Each thread accumulates multiple elements
    float local_sum = 0.0f;
    uint start = tgid * tg_size * 4 + tid;

    // Process 4 elements per thread for better throughput
    for (uint i = 0; i < 4; i++) {
        uint idx = start + i * tg_size;
        if (idx < total_elements) {
            float g = grads[idx];
            local_sum += g * g;
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
    if (tid < total_elements) {
        grads[tid] *= scale;
    }
}

/// Half-precision gradient scaling
kernel void scale_gradients_f16(
    device half* grads [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < total_elements) {
        grads[tid] = half(float(grads[tid]) * scale);
    }
}

// =============================================================================
// Fused Cross-Entropy Loss + Backward
// =============================================================================

/// Fused cross-entropy loss computation with backward pass.
///
/// Computes loss and gradients in a single pass:
/// - Forward: loss = -sum(label * log_softmax(logits))
/// - Backward: grad_logits = softmax(logits) - one_hot(labels)
///
/// This fuses softmax + log + nll_loss + backward into one kernel,
/// avoiding multiple passes over the logits tensor.
///
/// Memory layout:
/// - logits: [batch * seq, vocab_size] - model outputs (AFTER shifting)
/// - labels: [batch * seq] - target token IDs (AFTER shifting)
/// - grad_logits: [batch * seq, vocab_size] - output gradients
/// - loss: [1] - scalar loss (atomically accumulated)
///
/// Grid: [batch * seq, 1, 1]  (one threadgroup per position)
/// Threadgroup: [WARP_SIZE, 1, 1]
kernel void fused_cross_entropy_forward_backward(
    device const float* logits [[buffer(0)]],     // [N, vocab_size]
    device const int* labels [[buffer(1)]],        // [N]
    device float* grad_logits [[buffer(2)]],       // [N, vocab_size]
    device atomic_float* loss [[buffer(3)]],       // [1]
    constant uint& N [[buffer(4)]],                // batch * seq (after shift)
    constant uint& vocab_size [[buffer(5)]],
    constant int& ignore_index [[buffer(6)]],      // -100 typically
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tgid >= N) return;

    const int label = labels[tgid];

    // Skip ignored positions (padding)
    if (label == ignore_index) {
        // Zero out gradients for ignored positions
        for (uint v = tid; v < vocab_size; v += WARP_SIZE) {
            grad_logits[tgid * vocab_size + v] = 0.0f;
        }
        return;
    }

    // Threadgroup memory for reductions
    threadgroup float shared_max[32];
    threadgroup float shared_sum[32];

    // Phase 1: Find max logit (for numerical stability)
    float local_max = -INFINITY;
    for (uint v = tid; v < vocab_size; v += WARP_SIZE) {
        local_max = max(local_max, logits[tgid * vocab_size + v]);
    }

    // Warp reduction for max
    for (uint offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, simd_shuffle_xor(local_max, offset));
    }
    shared_max[0] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_logit = shared_max[0];

    // Phase 2: Compute exp(logits - max) and sum
    float local_sum = 0.0f;
    for (uint v = tid; v < vocab_size; v += WARP_SIZE) {
        float exp_val = exp(logits[tgid * vocab_size + v] - max_logit);
        local_sum += exp_val;
    }

    // Warp reduction for sum
    for (uint offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += simd_shuffle_xor(local_sum, offset);
    }
    shared_sum[0] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_exp = shared_sum[0];

    // Phase 3: Compute softmax and gradients
    // softmax[v] = exp(logits[v] - max) / sum_exp
    // grad[v] = softmax[v] - (v == label ? 1 : 0)
    float log_sum_exp = log(sum_exp);

    for (uint v = tid; v < vocab_size; v += WARP_SIZE) {
        float logit = logits[tgid * vocab_size + v];
        float softmax_val = exp(logit - max_logit - log_sum_exp);
        float target = (int(v) == label) ? 1.0f : 0.0f;
        grad_logits[tgid * vocab_size + v] = softmax_val - target;
    }

    // Phase 4: Compute and accumulate loss
    // loss = -log_softmax[label] = -(logits[label] - max - log_sum_exp)
    if (tid == 0) {
        float logit_label = logits[tgid * vocab_size + uint(label)];
        float token_loss = -(logit_label - max_logit - log_sum_exp);
        atomic_fetch_add_explicit(loss, token_loss, memory_order_relaxed);
    }
}

/// Half-precision version of fused cross-entropy
kernel void fused_cross_entropy_forward_backward_f16(
    device const half* logits [[buffer(0)]],
    device const int* labels [[buffer(1)]],
    device half* grad_logits [[buffer(2)]],
    device atomic_float* loss [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& vocab_size [[buffer(5)]],
    constant int& ignore_index [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tgid >= N) return;

    const int label = labels[tgid];

    if (label == ignore_index) {
        for (uint v = tid; v < vocab_size; v += WARP_SIZE) {
            grad_logits[tgid * vocab_size + v] = half(0.0f);
        }
        return;
    }

    threadgroup float shared_max[32];
    threadgroup float shared_sum[32];

    // Find max (compute in fp32)
    float local_max = -INFINITY;
    for (uint v = tid; v < vocab_size; v += WARP_SIZE) {
        local_max = max(local_max, float(logits[tgid * vocab_size + v]));
    }

    for (uint offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, simd_shuffle_xor(local_max, offset));
    }
    shared_max[0] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_logit = shared_max[0];

    // Compute sum of exp
    float local_sum = 0.0f;
    for (uint v = tid; v < vocab_size; v += WARP_SIZE) {
        float exp_val = exp(float(logits[tgid * vocab_size + v]) - max_logit);
        local_sum += exp_val;
    }

    for (uint offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += simd_shuffle_xor(local_sum, offset);
    }
    shared_sum[0] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_exp = shared_sum[0];
    float log_sum_exp = log(sum_exp);

    // Softmax and gradients
    for (uint v = tid; v < vocab_size; v += WARP_SIZE) {
        float logit = float(logits[tgid * vocab_size + v]);
        float softmax_val = exp(logit - max_logit - log_sum_exp);
        float target = (int(v) == label) ? 1.0f : 0.0f;
        grad_logits[tgid * vocab_size + v] = half(softmax_val - target);
    }

    // Accumulate loss
    if (tid == 0) {
        float logit_label = float(logits[tgid * vocab_size + uint(label)]);
        float token_loss = -(logit_label - max_logit - log_sum_exp);
        atomic_fetch_add_explicit(loss, token_loss, memory_order_relaxed);
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
