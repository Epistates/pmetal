// mpp_fused_distill.metal
// Metal 4 Fused Distillation Loss using MPP cooperative reduction.
//
// Implements KL divergence and variants for knowledge distillation:
//   KL(P || Q)     = sum_i P(i) * (log P(i) - log Q(i))   — forward KL
//   KL(Q || P)     = sum_i Q(i) * (log Q(i) - log P(i))   — reverse KL
//   JS(P, Q)       = 0.5 * KL(P || M) + 0.5 * KL(Q || M)  — Jensen-Shannon
//                    where M = 0.5*(P+Q)
//   SoftCE(P, Q)   = -sum_i P(i) * log Q(i)               — soft cross-entropy
//
// MPP Guide Section 2.3.4 (Postfix Fusion): log-sum-exp and KL accumulation
// are computed directly via simd_max() / simd_sum() without threadgroup memory
// round-trips. Both teacher and student logsumexp are computed in a single
// combined pass over vocabulary, then KL is accumulated in a second pass.
//
// MPP Guide Section 2.3.1 (Single simdgroup): one SIMD group (32 lanes) per
// token. SIMD lanes stride across the vocabulary at intervals of 32.
//
// Grid: [num_tokens, 1, 1]  Threadgroup: [32, 1, 1]
//
// Mirrors Metal 3 fused_distill.metal math exactly.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

struct MppDistillParams {
    uint  num_tokens;
    uint  vocab_size;
    float temperature;
    float alpha;
    int   ignore_index;
};

// =============================================================================
// Forward KL Divergence: KL(teacher || student)  (fp32)
// =============================================================================
//
// Combined pass: compute logsumexp for both distributions simultaneously.
// Second pass: accumulate KL element-wise.
// Both reductions use simd_max/simd_sum (MPP Section 2.3.1).

kernel void mpp_fused_kl_divergence_f32(
    device const float*   teacher_logits [[buffer(0)]],   // [N, vocab]
    device const float*   student_logits [[buffer(1)]],   // [N, vocab]
    device float*         losses         [[buffer(2)]],   // [N]
    device float*         teacher_lse    [[buffer(3)]],   // [N] cached for backward
    device float*         student_lse    [[buffer(4)]],   // [N] cached for backward
    constant MppDistillParams& p         [[buffer(5)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]]
) {
    if (token_idx >= p.num_tokens) return;

    const float inv_T = 1.0f / p.temperature;
    const device float* t_row = teacher_logits + token_idx * p.vocab_size;
    const device float* s_row = student_logits + token_idx * p.vocab_size;

    // --- Pass 1: combined online logsumexp via simd reduction ---
    float t_max = -INFINITY;
    float s_max = -INFINITY;

    for (uint v = lane; v < p.vocab_size; v += 32u) {
        t_max = max(t_max, t_row[v] * inv_T);
        s_max = max(s_max, s_row[v] * inv_T);
    }
    t_max = simd_max(t_max);
    s_max = simd_max(s_max);

    float t_sum = 0.0f;
    float s_sum = 0.0f;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        t_sum += metal::fast::exp(t_row[v] * inv_T - t_max);
        s_sum += metal::fast::exp(s_row[v] * inv_T - s_max);
    }
    float t_lse = t_max + metal::fast::log(simd_sum(t_sum));
    float s_lse = s_max + metal::fast::log(simd_sum(s_sum));

    // Cache logsumexp for backward
    if (lane == 0u) {
        teacher_lse[token_idx] = t_lse;
        student_lse[token_idx] = s_lse;
    }

    // --- Pass 2: KL accumulation ---
    // KL = sum_i exp(t_i*invT - t_lse) * ((t_i*invT - t_lse) - (s_i*invT - s_lse))
    float local_kl = 0.0f;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        float log_p = t_row[v] * inv_T - t_lse;  // log teacher_prob
        float log_q = s_row[v] * inv_T - s_lse;  // log student_prob
        float p_v   = metal::fast::exp(log_p);
        local_kl   += p_v * (log_p - log_q);
    }
    float kl = simd_sum(local_kl);

    if (lane == 0u) {
        losses[token_idx] = kl * (p.temperature * p.temperature);
    }
}

// =============================================================================
// Reverse KL Divergence: KL(student || teacher)  (fp32)
// =============================================================================

kernel void mpp_fused_reverse_kl_f32(
    device const float*   teacher_logits [[buffer(0)]],
    device const float*   student_logits [[buffer(1)]],
    device float*         losses         [[buffer(2)]],
    device float*         teacher_lse    [[buffer(3)]],
    device float*         student_lse    [[buffer(4)]],
    constant MppDistillParams& p         [[buffer(5)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]]
) {
    if (token_idx >= p.num_tokens) return;

    const float inv_T = 1.0f / p.temperature;
    const device float* t_row = teacher_logits + token_idx * p.vocab_size;
    const device float* s_row = student_logits + token_idx * p.vocab_size;

    float t_max = -INFINITY, s_max = -INFINITY;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        t_max = max(t_max, t_row[v] * inv_T);
        s_max = max(s_max, s_row[v] * inv_T);
    }
    t_max = simd_max(t_max);
    s_max = simd_max(s_max);

    float t_sum = 0.0f, s_sum = 0.0f;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        t_sum += metal::fast::exp(t_row[v] * inv_T - t_max);
        s_sum += metal::fast::exp(s_row[v] * inv_T - s_max);
    }
    float t_lse = t_max + metal::fast::log(simd_sum(t_sum));
    float s_lse = s_max + metal::fast::log(simd_sum(s_sum));

    if (lane == 0u) {
        teacher_lse[token_idx] = t_lse;
        student_lse[token_idx] = s_lse;
    }

    // Reverse KL: sum_i Q(i) * (log Q(i) - log P(i))
    float local_kl = 0.0f;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        float log_p = t_row[v] * inv_T - t_lse;
        float log_q = s_row[v] * inv_T - s_lse;
        float q_v   = metal::fast::exp(log_q);
        local_kl   += q_v * (log_q - log_p);
    }
    float kl = simd_sum(local_kl);

    if (lane == 0u) {
        losses[token_idx] = kl * (p.temperature * p.temperature);
    }
}

// =============================================================================
// Jensen-Shannon Divergence  (fp32)
// =============================================================================
//
// JS(P, Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)  where M = 0.5*(P+Q)
//
// Computed without materializing M: using log-sum-exp of a mixture.
// log M(i) = log(0.5 * P(i) + 0.5 * Q(i))
//           = log(0.5) + log(exp(log_p) + exp(log_q))
//           = log(0.5) + softplus(log_p - log_q) + log_q  (stable form)

kernel void mpp_fused_js_divergence_f32(
    device const float*   teacher_logits [[buffer(0)]],
    device const float*   student_logits [[buffer(1)]],
    device float*         losses         [[buffer(2)]],
    device float*         teacher_lse    [[buffer(3)]],
    device float*         student_lse    [[buffer(4)]],
    constant MppDistillParams& p         [[buffer(5)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]]
) {
    if (token_idx >= p.num_tokens) return;

    const float inv_T = 1.0f / p.temperature;
    const device float* t_row = teacher_logits + token_idx * p.vocab_size;
    const device float* s_row = student_logits + token_idx * p.vocab_size;

    float t_max = -INFINITY, s_max = -INFINITY;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        t_max = max(t_max, t_row[v] * inv_T);
        s_max = max(s_max, s_row[v] * inv_T);
    }
    t_max = simd_max(t_max);
    s_max = simd_max(s_max);

    float t_sum = 0.0f, s_sum = 0.0f;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        t_sum += metal::fast::exp(t_row[v] * inv_T - t_max);
        s_sum += metal::fast::exp(s_row[v] * inv_T - s_max);
    }
    float t_lse = t_max + metal::fast::log(simd_sum(t_sum));
    float s_lse = s_max + metal::fast::log(simd_sum(s_sum));

    if (lane == 0u) {
        teacher_lse[token_idx] = t_lse;
        student_lse[token_idx] = s_lse;
    }

    // JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    //    = 0.5 * sum_i [P log(P/M) + Q log(Q/M)]
    //    = 0.5 * sum_i [P log_p + Q log_q - (P+Q) log_m]
    //      where log_m = log(0.5*(P+Q))
    float local_js = 0.0f;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        float log_p = t_row[v] * inv_T - t_lse;
        float log_q = s_row[v] * inv_T - s_lse;
        float p_v   = metal::fast::exp(log_p);
        float q_v   = metal::fast::exp(log_q);

        // log M(i) = log(0.5*p + 0.5*q) — numerically stable via log1p
        float max_pq  = max(log_p, log_q);
        float log_m   = max_pq + metal::fast::log(metal::fast::exp(log_p - max_pq)
                                                 + metal::fast::exp(log_q - max_pq))
                      - metal::fast::log(2.0f);

        local_js += 0.5f * (p_v * (log_p - log_m) + q_v * (log_q - log_m));
    }
    float js = simd_sum(local_js);

    if (lane == 0u) {
        losses[token_idx] = js * (p.temperature * p.temperature);
    }
}

// =============================================================================
// Soft Cross-Entropy: -sum_i P(i) * log Q(i)  (fp32)
// =============================================================================

kernel void mpp_fused_soft_cross_entropy_f32(
    device const float*   teacher_logits [[buffer(0)]],
    device const float*   student_logits [[buffer(1)]],
    device float*         losses         [[buffer(2)]],
    device float*         teacher_lse    [[buffer(3)]],
    device float*         student_lse    [[buffer(4)]],
    constant MppDistillParams& p         [[buffer(5)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]]
) {
    if (token_idx >= p.num_tokens) return;

    const float inv_T = 1.0f / p.temperature;
    const device float* t_row = teacher_logits + token_idx * p.vocab_size;
    const device float* s_row = student_logits + token_idx * p.vocab_size;

    float t_max = -INFINITY, s_max = -INFINITY;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        t_max = max(t_max, t_row[v] * inv_T);
        s_max = max(s_max, s_row[v] * inv_T);
    }
    t_max = simd_max(t_max);
    s_max = simd_max(s_max);

    float t_sum = 0.0f, s_sum = 0.0f;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        t_sum += metal::fast::exp(t_row[v] * inv_T - t_max);
        s_sum += metal::fast::exp(s_row[v] * inv_T - s_max);
    }
    float t_lse = t_max + metal::fast::log(simd_sum(t_sum));
    float s_lse = s_max + metal::fast::log(simd_sum(s_sum));

    if (lane == 0u) {
        teacher_lse[token_idx] = t_lse;
        student_lse[token_idx] = s_lse;
    }

    // SoftCE = -sum_i P(i) * log Q(i)
    float local_ce = 0.0f;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        float log_p = t_row[v] * inv_T - t_lse;
        float log_q = s_row[v] * inv_T - s_lse;
        float p_v   = metal::fast::exp(log_p);
        local_ce   -= p_v * log_q;
    }
    float ce = simd_sum(local_ce);

    if (lane == 0u) {
        losses[token_idx] = ce * (p.temperature * p.temperature);
    }
}

// =============================================================================
// Half-precision KL Divergence (fp16 logits, fp32 accumulation)
// =============================================================================

kernel void mpp_fused_kl_divergence_f16(
    device const half*    teacher_logits [[buffer(0)]],
    device const half*    student_logits [[buffer(1)]],
    device float*         losses         [[buffer(2)]],
    device float*         teacher_lse    [[buffer(3)]],
    device float*         student_lse    [[buffer(4)]],
    constant MppDistillParams& p         [[buffer(5)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]]
) {
    if (token_idx >= p.num_tokens) return;

    const float inv_T = 1.0f / p.temperature;
    const device half* t_row = teacher_logits + token_idx * p.vocab_size;
    const device half* s_row = student_logits + token_idx * p.vocab_size;

    float t_max = -INFINITY, s_max = -INFINITY;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        t_max = max(t_max, float(t_row[v]) * inv_T);
        s_max = max(s_max, float(s_row[v]) * inv_T);
    }
    t_max = simd_max(t_max);
    s_max = simd_max(s_max);

    float t_sum = 0.0f, s_sum = 0.0f;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        t_sum += metal::fast::exp(float(t_row[v]) * inv_T - t_max);
        s_sum += metal::fast::exp(float(s_row[v]) * inv_T - s_max);
    }
    float t_lse = t_max + metal::fast::log(simd_sum(t_sum));
    float s_lse = s_max + metal::fast::log(simd_sum(s_sum));

    if (lane == 0u) {
        teacher_lse[token_idx] = t_lse;
        student_lse[token_idx] = s_lse;
    }

    float local_kl = 0.0f;
    for (uint v = lane; v < p.vocab_size; v += 32u) {
        float log_p = float(t_row[v]) * inv_T - t_lse;
        float log_q = float(s_row[v]) * inv_T - s_lse;
        float p_v   = metal::fast::exp(log_p);
        local_kl   += p_v * (log_p - log_q);
    }
    float kl = simd_sum(local_kl);

    if (lane == 0u) {
        losses[token_idx] = kl * (p.temperature * p.temperature);
    }
}
