// mpp_fused_cross_entropy.metal
// Metal 4 Fused Cross-Entropy Loss using MPP cooperative reduction.
//
// Cross-entropy involves:
//   1. Log-softmax: max reduction → subtract max → exp → sum → log
//   2. NLL: gather target class, negate log-prob
//   3. Gradient: softmax(logits) - one_hot(label)
//
// MPP Guide Section 2.3.4 (Postfix Fusion): partial sums are reduced
// in register space using simd_sum() — no threadgroup memory round-trip
// for the reduction step, unlike the Metal 3 kernel.
//
// MPP Guide Section 2.3.1 (Single simdgroup): each threadgroup is exactly
// one SIMD group (32 lanes). Each lane handles a stride of the vocabulary,
// accumulating max/sum locally, then uses simd_max() / simd_sum() for the
// final cross-lane reduction.
//
// Grid: [num_tokens, 1, 1]  Threadgroup: [32, 1, 1]
//
// Mirrors the Metal 3 fused_training.metal math exactly — only the reduction
// mechanism changes (simd_max/simd_sum instead of threadgroup memory tree).

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// =============================================================================
// MPP Fused Cross-Entropy Forward + Backward (fp32)
// =============================================================================
//
// Per-token forward + backward in one pass per SIMD group.
// Reduction uses simd_max() / simd_sum() — no threadgroup staging.

kernel void mpp_fused_cross_entropy_fwd_bwd_f32(
    device const float*   logits       [[buffer(0)]],   // [N, vocab_size]
    device const int*     labels       [[buffer(1)]],   // [N]
    device float*         grad_logits  [[buffer(2)]],   // [N, vocab_size]
    device atomic_float*  loss         [[buffer(3)]],   // [1] accumulated
    constant uint&        N            [[buffer(4)]],
    constant uint&        vocab_size   [[buffer(5)]],
    constant int&         ignore_index [[buffer(6)]],
    uint token_idx [[threadgroup_position_in_grid]],    // one TG per token
    uint lane      [[thread_index_in_simdgroup]]         // 0..31
) {
    if (token_idx >= N) return;

    const int label = labels[token_idx];
    const device float* row = logits + token_idx * vocab_size;
    device float* grad_row  = grad_logits + token_idx * vocab_size;

    if (label == ignore_index) {
        for (uint v = lane; v < vocab_size; v += 32u) {
            grad_row[v] = 0.0f;
        }
        return;
    }

    // --- Phase 1: max reduction (numerical stability) ---
    float local_max = -INFINITY;
    for (uint v = lane; v < vocab_size; v += 32u) {
        local_max = max(local_max, row[v]);
    }
    // MPP cooperative reduction — no threadgroup barrier (MPP 2.3.1)
    float max_logit = simd_max(local_max);

    // --- Phase 2: exp sum ---
    float local_sum = 0.0f;
    for (uint v = lane; v < vocab_size; v += 32u) {
        local_sum += metal::fast::exp(row[v] - max_logit);
    }
    float sum_exp   = simd_sum(local_sum);
    float log_sum_e = metal::fast::log(sum_exp);

    // --- Phase 3: gradient (softmax - one_hot) ---
    for (uint v = lane; v < vocab_size; v += 32u) {
        float logit      = row[v];
        float softmax_v  = metal::fast::exp(logit - max_logit - log_sum_e);
        float target     = (int(v) == label) ? 1.0f : 0.0f;
        grad_row[v]      = softmax_v - target;
    }

    // --- Phase 4: accumulate token loss (lane 0 only) ---
    if (lane == 0u) {
        float logit_label = row[uint(label)];
        float token_loss  = -(logit_label - max_logit - log_sum_e);
        atomic_fetch_add_explicit(loss, token_loss, memory_order_relaxed);
    }
}

// =============================================================================
// MPP Fused Cross-Entropy Forward + Backward (fp16)
// =============================================================================

kernel void mpp_fused_cross_entropy_fwd_bwd_f16(
    device const half*    logits       [[buffer(0)]],
    device const int*     labels       [[buffer(1)]],
    device half*          grad_logits  [[buffer(2)]],
    device atomic_float*  loss         [[buffer(3)]],
    constant uint&        N            [[buffer(4)]],
    constant uint&        vocab_size   [[buffer(5)]],
    constant int&         ignore_index [[buffer(6)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]]
) {
    if (token_idx >= N) return;

    const int label = labels[token_idx];
    const device half* row  = logits + token_idx * vocab_size;
    device half* grad_row   = grad_logits + token_idx * vocab_size;

    if (label == ignore_index) {
        for (uint v = lane; v < vocab_size; v += 32u) {
            grad_row[v] = half(0.0f);
        }
        return;
    }

    // All reductions in fp32 for numerical stability
    float local_max = -INFINITY;
    for (uint v = lane; v < vocab_size; v += 32u) {
        local_max = max(local_max, float(row[v]));
    }
    float max_logit = simd_max(local_max);

    float local_sum = 0.0f;
    for (uint v = lane; v < vocab_size; v += 32u) {
        local_sum += metal::fast::exp(float(row[v]) - max_logit);
    }
    float sum_exp   = simd_sum(local_sum);
    float log_sum_e = metal::fast::log(sum_exp);

    for (uint v = lane; v < vocab_size; v += 32u) {
        float logit     = float(row[v]);
        float softmax_v = metal::fast::exp(logit - max_logit - log_sum_e);
        float target    = (int(v) == label) ? 1.0f : 0.0f;
        grad_row[v]     = half(softmax_v - target);
    }

    if (lane == 0u) {
        float logit_label = float(row[uint(label)]);
        float token_loss  = -(logit_label - max_logit - log_sum_e);
        atomic_fetch_add_explicit(loss, token_loss, memory_order_relaxed);
    }
}

// =============================================================================
// MPP Forward-only Cross-Entropy (fp32) — for inference / eval
// =============================================================================
//
// Only computes loss, no gradients. Cheaper when backward is not needed.
//
// NOTE: buffer(3) is intentionally absent from this forward-only signature.
// The fwd+bwd variants bind an `atomic_float* loss` accumulator at buffer(3).
// This kernel writes per-token losses to `per_token[buffer(2)]` instead of
// accumulating into a scalar, so there is no buffer(3) binding.
//
// IMPORTANT: The Rust dispatcher (`mpp_fused_cross_entropy.rs`) always uses
// `forward_only: false` (the default). The `mpp_cross_entropy_forward_f32`
// kernel is currently unreachable from the Rust side. If forward-only dispatch
// is ever needed, the Rust caller must NOT bind a loss buffer at index 3, and
// must instead pass a per_token output buffer. The buffer index gap is load-
// bearing for future callers — do not renumber without updating the dispatcher.

kernel void mpp_cross_entropy_forward_f32(
    device const float*   logits       [[buffer(0)]],
    device const int*     labels       [[buffer(1)]],
    device float*         per_token    [[buffer(2)]],   // [N] per-token loss
    // buffer(3) intentionally absent — see note above
    constant uint&        N            [[buffer(4)]],
    constant uint&        vocab_size   [[buffer(5)]],
    constant int&         ignore_index [[buffer(6)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]]
) {
    if (token_idx >= N) return;

    const int label = labels[token_idx];
    if (label == ignore_index) {
        if (lane == 0u) per_token[token_idx] = 0.0f;
        return;
    }

    const device float* row = logits + token_idx * vocab_size;

    float local_max = -INFINITY;
    for (uint v = lane; v < vocab_size; v += 32u) {
        local_max = max(local_max, row[v]);
    }
    float max_logit = simd_max(local_max);

    float local_sum = 0.0f;
    for (uint v = lane; v < vocab_size; v += 32u) {
        local_sum += metal::fast::exp(row[v] - max_logit);
    }
    float log_sum_e = metal::fast::log(simd_sum(local_sum));

    if (lane == 0u) {
        float logit_label    = row[uint(label)];
        per_token[token_idx] = -(logit_label - max_logit - log_sum_e);
    }
}
