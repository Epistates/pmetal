//
// Fused Cross-Entropy Loss Kernels
//
// These kernels compute cross-entropy loss and gradients efficiently:
// - Forward: CE(x, y) = logsumexp(x) - x[y] without materializing softmax
// - Backward: dL/dx[i] = softmax(x)[i] - (1 if i == target else 0)
//
// Memory-efficient: O(1) per token instead of O(vocab)
// Uses online softmax algorithm (like FlashAttention)
//
// Reference: Flash-Attention Triton Fused Cross-Entropy
//

#include <metal_stdlib>
using namespace metal;

// SIMD configuration
#define SIMD_SIZE 32

/// Parameters for fused cross-entropy kernel
struct CrossEntropyParams {
    uint num_tokens;      // Total number of tokens
    uint vocab_size;      // Vocabulary size
    float label_smoothing; // Label smoothing factor (0.0 to disable)
    float softcap;        // Logit softcapping value (0.0 to disable)
    int ignore_index;     // Index to ignore in loss (-100 typically)
    uint block_size;      // Block size for chunked processing
};

/// Softcapping function: softcap * tanh(x / softcap)
inline float apply_softcap(float x, float softcap) {
    if (softcap == 0.0f) return x;
    return softcap * tanh(x / softcap);
}

/// Online logsumexp state for numerical stability
struct OnlineLogsumexp {
    float max_val;
    float sum_exp;

    OnlineLogsumexp() : max_val(-INFINITY), sum_exp(0.0f) {}

    void update(float val) {
        if (val > max_val) {
            sum_exp = sum_exp * exp(max_val - val) + 1.0f;
            max_val = val;
        } else {
            sum_exp += exp(val - max_val);
        }
    }

    float logsumexp() const {
        return max_val + log(sum_exp);
    }
};

/// Fused cross-entropy forward pass.
///
/// Computes: loss[i] = logsumexp(logits[i, :]) - logits[i, target[i]]
///
/// Uses online softmax to avoid materializing the full softmax distribution.
/// Handles ignored indices by setting their loss to 0.
/// Stores logsumexp for efficient backward pass (unsloth optimization).
kernel void fused_cross_entropy_forward(
    device const float* logits [[buffer(0)]],        // [num_tokens, vocab_size]
    device const int* targets [[buffer(1)]],         // [num_tokens]
    device float* losses [[buffer(2)]],              // [num_tokens]
    device float* logsumexp_out [[buffer(3)]],       // [num_tokens] for backward
    constant CrossEntropyParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    if (tid >= params.num_tokens) return;

    const int target = targets[tid];

    // Handle ignored indices
    if (target == params.ignore_index) {
        losses[tid] = 0.0f;
        logsumexp_out[tid] = 0.0f;
        return;
    }

    // Bounds check target
    if (target < 0 || uint(target) >= params.vocab_size) {
        losses[tid] = 0.0f;
        logsumexp_out[tid] = 0.0f;
        return;
    }

    device const float* row = logits + tid * params.vocab_size;

    // Online logsumexp computation (numerically stable)
    float m = -INFINITY;  // Running max
    float s = 0.0f;       // Running sum of exp(x - max)

    for (uint v = 0; v < params.vocab_size; v++) {
        float logit = row[v];

        // Apply softcapping if enabled
        if (params.softcap != 0.0f) {
            logit = apply_softcap(logit, params.softcap);
        }

        // Online softmax update
        if (logit > m) {
            s = s * exp(m - logit) + 1.0f;
            m = logit;
        } else {
            s += exp(logit - m);
        }
    }

    float lse = m + log(s);

    // Get target logit
    float target_logit = row[target];
    if (params.softcap != 0.0f) {
        target_logit = apply_softcap(target_logit, params.softcap);
    }

    // CE loss = logsumexp - target_logit
    float loss = lse - target_logit;

    // Apply label smoothing if enabled
    if (params.label_smoothing > 0.0f) {
        // Smoothed loss = (1 - eps) * CE + eps * (logsumexp - mean_logit)
        float smooth_loss = lse - (lse - log((float)params.vocab_size));
        loss = (1.0f - params.label_smoothing) * loss + params.label_smoothing * smooth_loss;
    }

    losses[tid] = loss;
    logsumexp_out[tid] = lse;  // Store logsumexp for efficient backward
}

/// Fused cross-entropy forward with SIMD parallelization.
///
/// Each SIMD group processes one token, parallelizing across vocabulary.
/// More efficient for larger vocabularies.
kernel void fused_cross_entropy_forward_simd(
    device const float* logits [[buffer(0)]],        // [num_tokens, vocab_size]
    device const int* targets [[buffer(1)]],         // [num_tokens]
    device float* losses [[buffer(2)]],              // [num_tokens]
    device float* logsumexp_out [[buffer(3)]],       // [num_tokens] for backward
    constant CrossEntropyParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint token_idx = tgid.x;

    if (token_idx >= params.num_tokens) return;

    const int target = targets[token_idx];

    // Handle ignored indices
    if (target == params.ignore_index) {
        if (lane_id == 0) {
            losses[token_idx] = 0.0f;
            logsumexp_out[token_idx] = 0.0f;
        }
        return;
    }

    device const float* row = logits + token_idx * params.vocab_size;

    // Each thread in SIMD group handles a subset of vocabulary
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (uint v = lane_id; v < params.vocab_size; v += SIMD_SIZE) {
        float logit = row[v];
        if (params.softcap != 0.0f) {
            logit = apply_softcap(logit, params.softcap);
        }

        if (logit > local_max) {
            local_sum = local_sum * exp(local_max - logit) + 1.0f;
            local_max = logit;
        } else {
            local_sum += exp(logit - local_max);
        }
    }

    // Reduce across SIMD group - find global max
    float global_max = simd_max(local_max);

    // Adjust local sums to global max
    local_sum = local_sum * exp(local_max - global_max);

    // Sum across SIMD group
    float global_sum = simd_sum(local_sum);

    // Compute final logsumexp
    float lse = global_max + log(global_sum);

    // Get target logit
    float target_logit = row[target];
    if (params.softcap != 0.0f) {
        target_logit = apply_softcap(target_logit, params.softcap);
    }

    // Compute loss
    float loss = lse - target_logit;

    if (lane_id == 0) {
        losses[token_idx] = loss;
        logsumexp_out[token_idx] = lse;  // Store logsumexp for efficient backward
    }
}

/// Fused cross-entropy backward pass (unsloth-style optimization).
///
/// Uses cached logsumexp from forward to compute gradients efficiently:
/// dL/dlogits[i, j] = exp(logits[i,j] - logsumexp[i]) - (1 if j == target[i] else 0)
///
/// Key insight: exp(x - logsumexp) = softmax(x), no need to recompute sum.
kernel void fused_cross_entropy_backward(
    device float* logits [[buffer(0)]],              // [num_tokens, vocab_size] - IN-PLACE gradient
    device const int* targets [[buffer(1)]],         // [num_tokens]
    device const float* logsumexp [[buffer(2)]],     // [num_tokens] from forward
    device const float* grad_loss [[buffer(3)]],     // [num_tokens] upstream gradient
    constant CrossEntropyParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint token_idx = gid.y;
    const uint vocab_idx = gid.x;

    if (token_idx >= params.num_tokens || vocab_idx >= params.vocab_size) return;

    const int target = targets[token_idx];

    // Handle ignored indices
    if (target == params.ignore_index) {
        logits[token_idx * params.vocab_size + vocab_idx] = 0.0f;
        return;
    }

    const float lse = logsumexp[token_idx];
    const float upstream = grad_loss[token_idx];
    device float* row = logits + token_idx * params.vocab_size;

    float x = row[vocab_idx];
    float orig_x = x;

    // Apply softcap if enabled
    float partial = x;
    if (params.softcap != 0.0f) {
        partial = tanh(x / params.softcap);
        x = params.softcap * partial;
    }

    // exp(x - logsumexp) = softmax(x)
    float grad = exp(x - lse);

    // Subtract 1 for target position
    if ((int)vocab_idx == target) {
        grad -= 1.0f;
    }

    // Handle softcap gradient: d/dx [t * tanh(x/t)] = 1 - tanh^2(x/t)
    if (params.softcap != 0.0f) {
        grad *= (1.0f - partial * partial);
    }

    // Scale by upstream gradient and write in-place
    row[vocab_idx] = upstream * grad;
}

/// Fused cross-entropy backward with SIMD parallelization.
///
/// Each SIMD group handles one token, computing gradients in parallel.
/// Uses cached logsumexp for efficient computation.
kernel void fused_cross_entropy_backward_simd(
    device float* logits [[buffer(0)]],              // [num_tokens, vocab_size] - IN-PLACE
    device const int* targets [[buffer(1)]],         // [num_tokens]
    device const float* logsumexp [[buffer(2)]],     // [num_tokens] from forward
    device const float* grad_loss [[buffer(3)]],     // [num_tokens]
    constant CrossEntropyParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    const uint token_idx = tgid.x;

    if (token_idx >= params.num_tokens) return;

    const int target = targets[token_idx];
    device float* row = logits + token_idx * params.vocab_size;

    // Handle ignored indices
    if (target == params.ignore_index) {
        for (uint v = lane_id; v < params.vocab_size; v += SIMD_SIZE) {
            row[v] = 0.0f;
        }
        return;
    }

    const float lse = logsumexp[token_idx];
    const float upstream = grad_loss[token_idx];

    // Compute gradients using cached logsumexp
    for (uint v = lane_id; v < params.vocab_size; v += SIMD_SIZE) {
        float x = row[v];
        float orig_x = x;

        // Apply softcap if enabled
        float partial = x;
        if (params.softcap != 0.0f) {
            partial = tanh(x / params.softcap);
            x = params.softcap * partial;
        }

        // exp(x - logsumexp) = softmax(x)
        float grad = exp(x - lse);

        // Subtract 1 for target position
        if ((int)v == target) {
            grad -= 1.0f;
        }

        // Softcap gradient: d/dx [t * tanh(x/t)] = 1 - tanh^2(x/t)
        if (params.softcap != 0.0f) {
            grad *= (1.0f - partial * partial);
        }

        // Scale by upstream and write in-place
        row[v] = upstream * grad;
    }
}

/// Compute mean loss over valid tokens.
///
/// Sums losses and counts valid (non-ignored) tokens, then divides.
kernel void cross_entropy_reduce_mean(
    device const float* losses [[buffer(0)]],       // [num_tokens]
    device const int* targets [[buffer(1)]],        // [num_tokens]
    device float* output [[buffer(2)]],             // [2] - [sum, count]
    constant uint& num_tokens [[buffer(3)]],
    constant int& ignore_index [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    // Each thread accumulates a partial sum
    float local_sum = 0.0f;
    float local_count = 0.0f;

    for (uint i = tid; i < num_tokens; i += threads_per_group * 256) {  // Assume max 256 threadgroups
        if (targets[i] != ignore_index) {
            local_sum += losses[i];
            local_count += 1.0f;
        }
    }

    // Reduce within SIMD group
    float simd_sum_val = simd_sum(local_sum);
    float simd_count_val = simd_sum(local_count);

    // First thread in each SIMD group adds to output atomically
    if (lane_id == 0) {
        atomic_fetch_add_explicit((device atomic_float*)&output[0], simd_sum_val, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&output[1], simd_count_val, memory_order_relaxed);
    }
}

/// Half-precision (fp16) fused cross-entropy forward.
///
/// For mixed precision training where logits are in fp16.
/// Accumulates in fp32 for numerical stability.
kernel void fused_cross_entropy_forward_f16(
    device const half* logits [[buffer(0)]],         // [num_tokens, vocab_size]
    device const int* targets [[buffer(1)]],         // [num_tokens]
    device float* losses [[buffer(2)]],              // [num_tokens] (fp32 for accuracy)
    device float* logsumexp_out [[buffer(3)]],       // [num_tokens] for backward
    constant CrossEntropyParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    const uint token_idx = tgid.x;

    if (token_idx >= params.num_tokens) return;

    const int target = targets[token_idx];

    if (target == params.ignore_index) {
        if (lane_id == 0) {
            losses[token_idx] = 0.0f;
            logsumexp_out[token_idx] = 0.0f;
        }
        return;
    }

    device const half* row = logits + token_idx * params.vocab_size;

    // Accumulate in fp32 for numerical stability
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (uint v = lane_id; v < params.vocab_size; v += SIMD_SIZE) {
        float logit = float(row[v]);
        if (params.softcap != 0.0f) {
            logit = apply_softcap(logit, params.softcap);
        }

        if (logit > local_max) {
            local_sum = local_sum * exp(local_max - logit) + 1.0f;
            local_max = logit;
        } else {
            local_sum += exp(logit - local_max);
        }
    }

    float global_max = simd_max(local_max);
    local_sum = local_sum * exp(local_max - global_max);
    float global_sum = simd_sum(local_sum);

    float lse = global_max + log(global_sum);

    float target_logit = float(row[target]);
    if (params.softcap != 0.0f) {
        target_logit = apply_softcap(target_logit, params.softcap);
    }

    float loss = lse - target_logit;

    if (lane_id == 0) {
        losses[token_idx] = loss;
        logsumexp_out[token_idx] = lse;  // Store logsumexp for backward
    }
}

// =============================================================================
// FUSED LINEAR + CROSS-ENTROPY (THE BIG WIN)
// =============================================================================
//
// This is unsloth's secret sauce: compute cross-entropy loss directly from
// hidden states without EVER materializing the full [batch, seq, vocab] logits.
//
// Memory savings:
//   - batch=4, seq=1024, vocab=150K, fp16 → logits would be 1.2GB
//   - With fusion: peak memory is only [chunk_size=4096] → 8MB
//
// Algorithm:
//   1. Chunk vocabulary into blocks of CHUNK_SIZE (e.g., 4096)
//   2. For each chunk: compute partial_logits = hidden @ weight[chunk].T
//   3. Track running logsumexp across chunks
//   4. Get logit at target index directly
//   5. Return loss = logsumexp - target_logit
//
// Reference: unsloth/unsloth/kernels/cross_entropy_loss.py
// =============================================================================

#define CE_CHUNK_SIZE 4096
#define CE_THREADS_PER_TOKEN 128  // Threads per token for parallel matmul

/// Parameters for fused linear cross-entropy
struct FusedLinearCEParams {
    uint num_tokens;      // Number of tokens
    uint hidden_size;     // Hidden dimension
    uint vocab_size;      // Vocabulary size
    uint chunk_size;      // Chunk size for vocabulary
    int ignore_index;     // Index to ignore (-100)
    float label_smoothing; // Label smoothing (0 to disable)
};

/// Fused linear + cross-entropy forward pass.
///
/// Computes CE loss directly from hidden states without materializing logits.
///
/// For each token i:
///   loss[i] = logsumexp(hidden[i] @ lm_head.T) - (hidden[i] @ lm_head[target[i]])
///
/// The computation is done in chunks to keep memory usage constant.
kernel void fused_linear_cross_entropy_forward(
    device const float* hidden_states [[buffer(0)]],  // [num_tokens, hidden_size]
    device const float* lm_head_weight [[buffer(1)]], // [vocab_size, hidden_size]
    device const int* targets [[buffer(2)]],          // [num_tokens]
    device float* losses [[buffer(3)]],               // [num_tokens]
    device float* logsumexp_out [[buffer(4)]],        // [num_tokens] for backward
    constant FusedLinearCEParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]     // Shared memory for partial logsumexp
) {
    const uint token_idx = tgid.x;

    if (token_idx >= params.num_tokens) return;

    const int target = targets[token_idx];

    // Handle ignored indices
    if (target == params.ignore_index) {
        if (lane_id == 0 && simd_group_id == 0) {
            losses[token_idx] = 0.0f;
            logsumexp_out[token_idx] = 0.0f;
        }
        return;
    }

    // Bounds check
    if (target < 0 || uint(target) >= params.vocab_size) {
        if (lane_id == 0 && simd_group_id == 0) {
            losses[token_idx] = 0.0f;
            logsumexp_out[token_idx] = 0.0f;
        }
        return;
    }

    // Pointer to this token's hidden state
    device const float* h = hidden_states + token_idx * params.hidden_size;

    // Online logsumexp across vocabulary chunks
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float target_logit = 0.0f;

    // Process vocabulary in chunks
    for (uint chunk_start = 0; chunk_start < params.vocab_size; chunk_start += params.chunk_size) {
        uint chunk_end = min(chunk_start + params.chunk_size, params.vocab_size);
        uint chunk_len = chunk_end - chunk_start;

        // Each thread computes dot products for a subset of vocab indices
        float local_max = -INFINITY;
        float local_sum = 0.0f;

        uint thread_idx = simd_group_id * SIMD_SIZE + lane_id;
        uint num_threads = CE_THREADS_PER_TOKEN;

        for (uint v = thread_idx; v < chunk_len; v += num_threads) {
            uint vocab_idx = chunk_start + v;
            device const float* w = lm_head_weight + vocab_idx * params.hidden_size;

            // Compute dot product: hidden[token] @ lm_head[vocab].T
            float logit = 0.0f;
            for (uint d = 0; d < params.hidden_size; d++) {
                logit += h[d] * w[d];
            }

            // Track target logit
            if (vocab_idx == uint(target)) {
                target_logit = logit;
            }

            // Online logsumexp update
            if (logit > local_max) {
                local_sum = local_sum * exp(local_max - logit) + 1.0f;
                local_max = logit;
            } else {
                local_sum += exp(logit - local_max);
            }
        }

        // Reduce within SIMD group
        float simd_max_val = simd_max(local_max);
        local_sum = local_sum * exp(local_max - simd_max_val);
        float simd_sum_val = simd_sum(local_sum);

        // Store to scratch for cross-SIMD-group reduction
        if (lane_id == 0) {
            scratch[simd_group_id * 2] = simd_max_val;
            scratch[simd_group_id * 2 + 1] = simd_sum_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Final reduction (first SIMD group only)
        if (simd_group_id == 0) {
            float chunk_max = -INFINITY;
            float chunk_sum = 0.0f;

            uint num_simd_groups = CE_THREADS_PER_TOKEN / SIMD_SIZE;
            for (uint g = lane_id; g < num_simd_groups; g += SIMD_SIZE) {
                float g_max = scratch[g * 2];
                float g_sum = scratch[g * 2 + 1];

                if (g_max > chunk_max) {
                    chunk_sum = chunk_sum * exp(chunk_max - g_max) + g_sum;
                    chunk_max = g_max;
                } else {
                    chunk_sum += g_sum * exp(g_max - chunk_max);
                }
            }

            // Reduce within this SIMD group
            chunk_max = simd_max(chunk_max);
            chunk_sum = simd_sum(chunk_sum * exp(scratch[lane_id * 2] - chunk_max));

            // Update running logsumexp across chunks
            if (lane_id == 0) {
                if (chunk_max > running_max) {
                    running_sum = running_sum * exp(running_max - chunk_max) + chunk_sum;
                    running_max = chunk_max;
                } else {
                    running_sum += chunk_sum * exp(chunk_max - running_max);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute final loss
    if (lane_id == 0 && simd_group_id == 0) {
        float lse = running_max + log(running_sum);
        float loss = lse - target_logit;

        // Apply label smoothing if enabled
        if (params.label_smoothing > 0.0f) {
            float smooth_loss = lse - (lse - log((float)params.vocab_size));
            loss = (1.0f - params.label_smoothing) * loss + params.label_smoothing * smooth_loss;
        }

        losses[token_idx] = loss;
        logsumexp_out[token_idx] = lse;
    }
}

/// Fused linear + cross-entropy forward (half precision).
///
/// Inputs in fp16, accumulation in fp32 for stability.
kernel void fused_linear_cross_entropy_forward_f16(
    device const half* hidden_states [[buffer(0)]],   // [num_tokens, hidden_size]
    device const half* lm_head_weight [[buffer(1)]],  // [vocab_size, hidden_size]
    device const int* targets [[buffer(2)]],          // [num_tokens]
    device float* losses [[buffer(3)]],               // [num_tokens] (fp32)
    device float* logsumexp_out [[buffer(4)]],        // [num_tokens] (fp32)
    constant FusedLinearCEParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint token_idx = tgid.x;

    if (token_idx >= params.num_tokens) return;

    const int target = targets[token_idx];

    if (target == params.ignore_index || target < 0 || uint(target) >= params.vocab_size) {
        if (lane_id == 0 && simd_group_id == 0) {
            losses[token_idx] = 0.0f;
            logsumexp_out[token_idx] = 0.0f;
        }
        return;
    }

    device const half* h = hidden_states + token_idx * params.hidden_size;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float target_logit = 0.0f;

    for (uint chunk_start = 0; chunk_start < params.vocab_size; chunk_start += params.chunk_size) {
        uint chunk_end = min(chunk_start + params.chunk_size, params.vocab_size);
        uint chunk_len = chunk_end - chunk_start;

        float local_max = -INFINITY;
        float local_sum = 0.0f;

        uint thread_idx = simd_group_id * SIMD_SIZE + lane_id;
        uint num_threads = CE_THREADS_PER_TOKEN;

        for (uint v = thread_idx; v < chunk_len; v += num_threads) {
            uint vocab_idx = chunk_start + v;
            device const half* w = lm_head_weight + vocab_idx * params.hidden_size;

            // fp32 accumulation for dot product
            float logit = 0.0f;
            for (uint d = 0; d < params.hidden_size; d++) {
                logit += float(h[d]) * float(w[d]);
            }

            if (vocab_idx == uint(target)) {
                target_logit = logit;
            }

            if (logit > local_max) {
                local_sum = local_sum * exp(local_max - logit) + 1.0f;
                local_max = logit;
            } else {
                local_sum += exp(logit - local_max);
            }
        }

        float simd_max_val = simd_max(local_max);
        local_sum = local_sum * exp(local_max - simd_max_val);
        float simd_sum_val = simd_sum(local_sum);

        if (lane_id == 0) {
            scratch[simd_group_id * 2] = simd_max_val;
            scratch[simd_group_id * 2 + 1] = simd_sum_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float chunk_max = -INFINITY;
            float chunk_sum = 0.0f;

            uint num_simd_groups = CE_THREADS_PER_TOKEN / SIMD_SIZE;
            for (uint g = lane_id; g < num_simd_groups; g += SIMD_SIZE) {
                float g_max = scratch[g * 2];
                float g_sum = scratch[g * 2 + 1];

                if (g_max > chunk_max) {
                    chunk_sum = chunk_sum * exp(chunk_max - g_max) + g_sum;
                    chunk_max = g_max;
                } else {
                    chunk_sum += g_sum * exp(g_max - chunk_max);
                }
            }

            chunk_max = simd_max(chunk_max);
            chunk_sum = simd_sum(chunk_sum * exp(scratch[lane_id * 2] - chunk_max));

            if (lane_id == 0) {
                if (chunk_max > running_max) {
                    running_sum = running_sum * exp(running_max - chunk_max) + chunk_sum;
                    running_max = chunk_max;
                } else {
                    running_sum += chunk_sum * exp(chunk_max - running_max);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane_id == 0 && simd_group_id == 0) {
        float lse = running_max + log(running_sum);
        float loss = lse - target_logit;

        if (params.label_smoothing > 0.0f) {
            float smooth_loss = lse - (lse - log((float)params.vocab_size));
            loss = (1.0f - params.label_smoothing) * loss + params.label_smoothing * smooth_loss;
        }

        losses[token_idx] = loss;
        logsumexp_out[token_idx] = lse;
    }
}

// =============================================================================
// END FUSED LINEAR + CROSS-ENTROPY
// =============================================================================

/// Half-precision (fp16) fused cross-entropy backward.
///
/// Writes gradients back to fp16 buffer.
kernel void fused_cross_entropy_backward_f16(
    device half* logits [[buffer(0)]],               // [num_tokens, vocab_size] - IN-PLACE
    device const int* targets [[buffer(1)]],         // [num_tokens]
    device const float* logsumexp [[buffer(2)]],     // [num_tokens] from forward
    device const float* grad_loss [[buffer(3)]],     // [num_tokens]
    constant CrossEntropyParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    const uint token_idx = tgid.x;

    if (token_idx >= params.num_tokens) return;

    const int target = targets[token_idx];
    device half* row = logits + token_idx * params.vocab_size;

    if (target == params.ignore_index) {
        for (uint v = lane_id; v < params.vocab_size; v += SIMD_SIZE) {
            row[v] = half(0.0f);
        }
        return;
    }

    const float lse = logsumexp[token_idx];
    const float upstream = grad_loss[token_idx];

    for (uint v = lane_id; v < params.vocab_size; v += SIMD_SIZE) {
        float x = float(row[v]);
        float orig_x = x;

        float partial = x;
        if (params.softcap != 0.0f) {
            partial = tanh(x / params.softcap);
            x = params.softcap * partial;
        }

        float grad = exp(x - lse);

        if ((int)v == target) {
            grad -= 1.0f;
        }

        if (params.softcap != 0.0f) {
            grad *= (1.0f - partial * partial);
        }

        row[v] = half(upstream * grad);
    }
}
