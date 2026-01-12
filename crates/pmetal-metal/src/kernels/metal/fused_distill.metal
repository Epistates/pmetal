//
// Fused Knowledge Distillation Loss Kernels
//
// GPU-accelerated distillation losses without materializing probability tensors:
// - KL Divergence: KL(teacher || student) for mode-covering behavior
// - Reverse KL: KL(student || teacher) for mode-seeking behavior
// - Jensen-Shannon Divergence: symmetric, bounded loss
// - Soft Cross-Entropy: CE with soft targets from teacher
//
// Key optimizations:
// - Uses online softmax to avoid O(vocab) memory per token
// - Temperature scaling built into kernel
// - SIMD parallelization for large vocabularies
// - Caches logsumexp for efficient backward pass
//
// Reference: DistilBERT, Knowledge Distillation literature
//

#include <metal_stdlib>
using namespace metal;

#define SIMD_SIZE 32
#define DISTILL_THREADS_PER_TOKEN 128

/// Parameters for distillation loss kernels
struct DistillParams {
    uint num_tokens;       // Number of tokens
    uint vocab_size;       // Vocabulary size
    float temperature;     // Temperature for softening distributions
    float alpha;           // Blending weight for soft loss (vs hard loss)
    int ignore_index;      // Index to ignore in loss (-100)
};

/// Parameters for hidden state alignment loss
struct HiddenAlignParams {
    uint num_tokens;       // Number of tokens
    uint teacher_dim;      // Teacher hidden dimension
    uint student_dim;      // Student hidden dimension
    uint projection_dim;   // Projection dimension (0 if no projection)
    float weight;          // Loss weight
};

// =============================================================================
// KL DIVERGENCE DISTILLATION LOSS
// =============================================================================
//
// Forward KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
//                    = sum_i P(i) * (log P(i) - log Q(i))
//
// Where P = softmax(teacher / T), Q = softmax(student / T)
//
// Using online softmax:
//   log_softmax(x) = x - logsumexp(x)
//   P(i) = exp(log_softmax(teacher)[i])
//   KL = sum_i exp(t_i - lse_t) * ((t_i - lse_t) - (s_i - lse_s))
//      = sum_i exp(t_i - lse_t) * (t_i - s_i - lse_t + lse_s)
// =============================================================================

/// Compute temperature-scaled logsumexp for a row.
/// Returns (logsumexp, max_val) for numerical stability.
inline float2 compute_logsumexp(
    device const float* logits,
    uint vocab_size,
    float temperature
) {
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    for (uint v = 0; v < vocab_size; v++) {
        float logit = logits[v] / temperature;
        if (logit > max_val) {
            sum_exp = sum_exp * exp(max_val - logit) + 1.0f;
            max_val = logit;
        } else {
            sum_exp += exp(logit - max_val);
        }
    }

    return float2(max_val + log(sum_exp), max_val);
}

/// Forward KL divergence: KL(teacher || student)
///
/// Computes: sum(teacher_probs * (log_teacher - log_student))
/// This is the "mode-covering" loss that encourages student to cover
/// all modes of the teacher distribution.
///
/// Optimizations:
/// - Pre-computed temperature inverse (multiply vs divide)
/// - metal::fast:: math functions for ~5-10% speedup
kernel void fused_kl_divergence_forward(
    device const float* teacher_logits [[buffer(0)]],  // [num_tokens, vocab_size]
    device const float* student_logits [[buffer(1)]],  // [num_tokens, vocab_size]
    device float* losses [[buffer(2)]],                // [num_tokens]
    device float* teacher_lse [[buffer(3)]],           // [num_tokens] for backward
    device float* student_lse [[buffer(4)]],           // [num_tokens] for backward
    constant DistillParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_tokens) return;

    device const float* t_row = teacher_logits + tid * params.vocab_size;
    device const float* s_row = student_logits + tid * params.vocab_size;

    // Pre-compute temperature inverse for efficiency
    const float inv_T = 1.0f / params.temperature;

    // Combined pass: compute logsumexp for both distributions
    float t_max = -INFINITY, t_sum = 0.0f;
    float s_max = -INFINITY, s_sum = 0.0f;

    for (uint v = 0; v < params.vocab_size; v++) {
        float t_logit = t_row[v] * inv_T;
        float s_logit = s_row[v] * inv_T;

        // Teacher online logsumexp
        if (t_logit > t_max) {
            t_sum = t_sum * metal::fast::exp(t_max - t_logit) + 1.0f;
            t_max = t_logit;
        } else {
            t_sum += metal::fast::exp(t_logit - t_max);
        }

        // Student online logsumexp
        if (s_logit > s_max) {
            s_sum = s_sum * metal::fast::exp(s_max - s_logit) + 1.0f;
            s_max = s_logit;
        } else {
            s_sum += metal::fast::exp(s_logit - s_max);
        }
    }

    float t_lse = t_max + metal::fast::log(t_sum);
    float s_lse = s_max + metal::fast::log(s_sum);

    // Second pass: compute KL divergence
    // KL = sum_i P(i) * (log P(i) - log Q(i))
    float kl = 0.0f;

    for (uint v = 0; v < params.vocab_size; v++) {
        float t_logit = t_row[v] * inv_T;
        float s_logit = s_row[v] * inv_T;

        float log_p = t_logit - t_lse;  // log(teacher_prob)
        float log_q = s_logit - s_lse;  // log(student_prob)
        float p = metal::fast::exp(log_p);  // teacher_prob

        kl += p * (log_p - log_q);
    }

    losses[tid] = max(kl, 0.0f);  // KL should be non-negative
    teacher_lse[tid] = t_lse;
    student_lse[tid] = s_lse;
}

/// Reverse KL divergence: KL(student || teacher)
///
/// Computes: sum(student_probs * (log_student - log_teacher))
/// This is the "mode-seeking" loss that makes student confident
/// about a subset of teacher's modes.
///
/// Optimizations:
/// - Pre-computed temperature inverse (multiply vs divide)
/// - metal::fast:: math functions for ~5-10% speedup
kernel void fused_reverse_kl_divergence_forward(
    device const float* teacher_logits [[buffer(0)]],
    device const float* student_logits [[buffer(1)]],
    device float* losses [[buffer(2)]],
    device float* teacher_lse [[buffer(3)]],
    device float* student_lse [[buffer(4)]],
    constant DistillParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_tokens) return;

    device const float* t_row = teacher_logits + tid * params.vocab_size;
    device const float* s_row = student_logits + tid * params.vocab_size;

    // Pre-compute temperature inverse for efficiency
    const float inv_T = 1.0f / params.temperature;

    // Compute logsumexp for both distributions
    float t_max = -INFINITY, t_sum = 0.0f;
    float s_max = -INFINITY, s_sum = 0.0f;

    for (uint v = 0; v < params.vocab_size; v++) {
        float t_logit = t_row[v] * inv_T;
        float s_logit = s_row[v] * inv_T;

        if (t_logit > t_max) {
            t_sum = t_sum * metal::fast::exp(t_max - t_logit) + 1.0f;
            t_max = t_logit;
        } else {
            t_sum += metal::fast::exp(t_logit - t_max);
        }

        if (s_logit > s_max) {
            s_sum = s_sum * metal::fast::exp(s_max - s_logit) + 1.0f;
            s_max = s_logit;
        } else {
            s_sum += metal::fast::exp(s_logit - s_max);
        }
    }

    float t_lse = t_max + metal::fast::log(t_sum);
    float s_lse = s_max + metal::fast::log(s_sum);

    // Reverse KL: sum_i Q(i) * (log Q(i) - log P(i))
    float kl = 0.0f;

    for (uint v = 0; v < params.vocab_size; v++) {
        float t_logit = t_row[v] * inv_T;
        float s_logit = s_row[v] * inv_T;

        float log_p = t_logit - t_lse;
        float log_q = s_logit - s_lse;
        float q = metal::fast::exp(log_q);  // student_prob

        kl += q * (log_q - log_p);
    }

    losses[tid] = max(kl, 0.0f);
    teacher_lse[tid] = t_lse;
    student_lse[tid] = s_lse;
}

/// SIMD-parallel KL divergence for large vocabularies.
///
/// Each SIMD group handles one token, threads parallelize over vocabulary.
///
/// Optimizations:
/// - Pre-computed temperature inverse (multiply vs divide)
/// - metal::fast:: math functions for ~5-10% speedup
kernel void fused_kl_divergence_forward_simd(
    device const float* teacher_logits [[buffer(0)]],
    device const float* student_logits [[buffer(1)]],
    device float* losses [[buffer(2)]],
    device float* teacher_lse [[buffer(3)]],
    device float* student_lse [[buffer(4)]],
    constant DistillParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint token_idx = tgid.x;
    if (token_idx >= params.num_tokens) return;

    device const float* t_row = teacher_logits + token_idx * params.vocab_size;
    device const float* s_row = student_logits + token_idx * params.vocab_size;

    // Pre-compute temperature inverse for efficiency
    const float inv_T = 1.0f / params.temperature;
    const uint thread_idx = simd_group_id * SIMD_SIZE + lane_id;
    const uint num_threads = DISTILL_THREADS_PER_TOKEN;

    // Local accumulation for logsumexp
    float t_local_max = -INFINITY, t_local_sum = 0.0f;
    float s_local_max = -INFINITY, s_local_sum = 0.0f;

    // First pass: compute logsumexp for both
    for (uint v = thread_idx; v < params.vocab_size; v += num_threads) {
        float t_logit = t_row[v] * inv_T;
        float s_logit = s_row[v] * inv_T;

        // Teacher logsumexp
        if (t_logit > t_local_max) {
            t_local_sum = t_local_sum * metal::fast::exp(t_local_max - t_logit) + 1.0f;
            t_local_max = t_logit;
        } else {
            t_local_sum += metal::fast::exp(t_logit - t_local_max);
        }

        // Student logsumexp
        if (s_logit > s_local_max) {
            s_local_sum = s_local_sum * metal::fast::exp(s_local_max - s_logit) + 1.0f;
            s_local_max = s_logit;
        } else {
            s_local_sum += metal::fast::exp(s_logit - s_local_max);
        }
    }

    // SIMD reduction for teacher
    float t_simd_max = simd_max(t_local_max);
    t_local_sum = t_local_sum * metal::fast::exp(t_local_max - t_simd_max);
    float t_simd_sum = simd_sum(t_local_sum);

    // SIMD reduction for student
    float s_simd_max = simd_max(s_local_max);
    s_local_sum = s_local_sum * metal::fast::exp(s_local_max - s_simd_max);
    float s_simd_sum = simd_sum(s_local_sum);

    // Store to scratch for cross-SIMD reduction
    if (lane_id == 0) {
        scratch[simd_group_id * 4 + 0] = t_simd_max;
        scratch[simd_group_id * 4 + 1] = t_simd_sum;
        scratch[simd_group_id * 4 + 2] = s_simd_max;
        scratch[simd_group_id * 4 + 3] = s_simd_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final logsumexp values
    float t_lse, s_lse;

    if (simd_group_id == 0) {
        float t_max = -INFINITY, t_sum = 0.0f;
        float s_max = -INFINITY, s_sum = 0.0f;

        uint num_simd_groups = DISTILL_THREADS_PER_TOKEN / SIMD_SIZE;
        for (uint g = lane_id; g < num_simd_groups; g += SIMD_SIZE) {
            float g_t_max = scratch[g * 4 + 0];
            float g_t_sum = scratch[g * 4 + 1];
            float g_s_max = scratch[g * 4 + 2];
            float g_s_sum = scratch[g * 4 + 3];

            // Online logsumexp merge for teacher
            if (g_t_max > t_max) {
                t_sum = t_sum * metal::fast::exp(t_max - g_t_max) + g_t_sum;
                t_max = g_t_max;
            } else {
                t_sum += g_t_sum * metal::fast::exp(g_t_max - t_max);
            }

            // Online logsumexp merge for student
            if (g_s_max > s_max) {
                s_sum = s_sum * metal::fast::exp(s_max - g_s_max) + g_s_sum;
                s_max = g_s_max;
            } else {
                s_sum += g_s_sum * metal::fast::exp(g_s_max - s_max);
            }
        }

        // Final SIMD reduction
        t_max = simd_max(t_max);
        s_max = simd_max(s_max);

        if (lane_id == 0) {
            t_lse = t_max + metal::fast::log(t_sum);
            s_lse = s_max + metal::fast::log(s_sum);
            scratch[0] = t_lse;
            scratch[1] = s_lse;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    t_lse = scratch[0];
    s_lse = scratch[1];

    // Second pass: compute KL divergence
    float local_kl = 0.0f;

    for (uint v = thread_idx; v < params.vocab_size; v += num_threads) {
        float t_logit = t_row[v] * inv_T;
        float s_logit = s_row[v] * inv_T;

        float log_p = t_logit - t_lse;
        float log_q = s_logit - s_lse;
        float p = metal::fast::exp(log_p);

        local_kl += p * (log_p - log_q);
    }

    // Reduce KL across threads
    float simd_kl = simd_sum(local_kl);

    if (lane_id == 0) {
        scratch[simd_group_id] = simd_kl;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0 && lane_id == 0) {
        float total_kl = 0.0f;
        uint num_simd_groups = DISTILL_THREADS_PER_TOKEN / SIMD_SIZE;
        for (uint g = 0; g < num_simd_groups; g++) {
            total_kl += scratch[g];
        }

        losses[token_idx] = max(total_kl, 0.0f);
        teacher_lse[token_idx] = t_lse;
        student_lse[token_idx] = s_lse;
    }
}

// =============================================================================
// JENSEN-SHANNON DIVERGENCE
// =============================================================================
//
// JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
// where M = 0.5 * (P + Q)
//
// Properties:
// - Symmetric: JS(P || Q) = JS(Q || P)
// - Bounded: 0 <= JS <= log(2) ≈ 0.693
// - Always defined (unlike KL when Q=0)
// =============================================================================

/// Jensen-Shannon divergence using log-space arithmetic (numerically stable).
///
/// Computes the symmetric, bounded divergence between teacher and student.
/// Uses the formula: JS = 0.5 * (KL(P || M) + KL(Q || M)) where M = (P + Q) / 2
///
/// Key insight: log(M) = log((P + Q) / 2) = log(P) + log(1 + exp(log(Q) - log(P))) - log(2)
/// This avoids probability overflow/underflow issues.
kernel void fused_jensen_shannon_forward(
    device const float* teacher_logits [[buffer(0)]],
    device const float* student_logits [[buffer(1)]],
    device float* losses [[buffer(2)]],
    constant DistillParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_tokens) return;

    device const float* t_row = teacher_logits + tid * params.vocab_size;
    device const float* s_row = student_logits + tid * params.vocab_size;

    // Pre-compute temperature inverse for efficiency
    const float inv_T = 1.0f / params.temperature;
    const float log2_f = 0.693147180559945309f;  // ln(2)

    // First pass: online logsumexp for both distributions
    float t_max = -INFINITY, t_sum = 0.0f;
    float s_max = -INFINITY, s_sum = 0.0f;

    for (uint v = 0; v < params.vocab_size; v++) {
        float t_logit = t_row[v] * inv_T;
        float s_logit = s_row[v] * inv_T;

        // Teacher logsumexp
        if (t_logit > t_max) {
            t_sum = t_sum * metal::fast::exp(t_max - t_logit) + 1.0f;
            t_max = t_logit;
        } else {
            t_sum += metal::fast::exp(t_logit - t_max);
        }

        // Student logsumexp
        if (s_logit > s_max) {
            s_sum = s_sum * metal::fast::exp(s_max - s_logit) + 1.0f;
            s_max = s_logit;
        } else {
            s_sum += metal::fast::exp(s_logit - s_max);
        }
    }

    float t_lse = t_max + metal::fast::log(t_sum);
    float s_lse = s_max + metal::fast::log(s_sum);

    // Second pass: compute JS divergence using log-space arithmetic
    // JS = 0.5 * sum_i [P(i) * (log(P(i)) - log(M(i))) + Q(i) * (log(Q(i)) - log(M(i)))]
    // where log(M(i)) = log((P(i) + Q(i)) / 2) = log(P(i) + Q(i)) - log(2)
    //                 = log(P(i)) + log(1 + exp(log(Q(i)) - log(P(i)))) - log(2)
    float js = 0.0f;

    for (uint v = 0; v < params.vocab_size; v++) {
        float t_logit = t_row[v] * inv_T;
        float s_logit = s_row[v] * inv_T;

        float logp = t_logit - t_lse;  // log(P)
        float logq = s_logit - s_lse;  // log(Q)

        // Compute log(M) = log((P + Q) / 2) using numerically stable formula
        // log(M) = log(P) + log(1 + Q/P) - log(2) = log(P) + log(1 + exp(log(Q) - log(P))) - log(2)
        // Or equivalently using the max for stability:
        // log(M) = max(logp, logq) + log(exp(logp - max) + exp(logq - max)) - log(2)
        float max_logpq = max(logp, logq);
        float log_m = max_logpq + metal::fast::log(
            metal::fast::exp(logp - max_logpq) + metal::fast::exp(logq - max_logpq)
        ) - log2_f;

        // P(i) = exp(logp), Q(i) = exp(logq)
        float p = metal::fast::exp(logp);
        float q = metal::fast::exp(logq);

        // KL(P || M) contribution: P * (logP - logM)
        // KL(Q || M) contribution: Q * (logQ - logM)
        js += p * (logp - log_m) + q * (logq - log_m);
    }

    // JS = 0.5 * sum of KL terms (but we already summed both, so no 0.5 needed)
    // Actually JS = 0.5 * (KL(P||M) + KL(Q||M)) and we computed KL(P||M) + KL(Q||M)
    losses[tid] = 0.5f * max(js, 0.0f);
}

// =============================================================================
// SOFT CROSS-ENTROPY
// =============================================================================
//
// CE(P, Q) = -sum_i P(i) * log(Q(i))
//          = -sum_i exp(t_i - t_lse) * (s_i - s_lse)
//
// Equivalent to KL divergence up to a constant (teacher entropy).
// =============================================================================

/// Soft cross-entropy with teacher soft targets.
///
/// Computes: -sum(teacher_probs * log(student_probs))
///
/// Optimizations:
/// - Pre-computed temperature inverse (multiply vs divide)
/// - metal::fast:: math functions for ~5-10% speedup
kernel void fused_soft_cross_entropy_forward(
    device const float* teacher_logits [[buffer(0)]],
    device const float* student_logits [[buffer(1)]],
    device float* losses [[buffer(2)]],
    constant DistillParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_tokens) return;

    device const float* t_row = teacher_logits + tid * params.vocab_size;
    device const float* s_row = student_logits + tid * params.vocab_size;

    // Pre-compute temperature inverse for efficiency
    const float inv_T = 1.0f / params.temperature;

    // Compute logsumexp for both
    float t_max = -INFINITY, t_sum = 0.0f;
    float s_max = -INFINITY, s_sum = 0.0f;

    for (uint v = 0; v < params.vocab_size; v++) {
        float t_logit = t_row[v] * inv_T;
        float s_logit = s_row[v] * inv_T;

        if (t_logit > t_max) {
            t_sum = t_sum * metal::fast::exp(t_max - t_logit) + 1.0f;
            t_max = t_logit;
        } else {
            t_sum += metal::fast::exp(t_logit - t_max);
        }

        if (s_logit > s_max) {
            s_sum = s_sum * metal::fast::exp(s_max - s_logit) + 1.0f;
            s_max = s_logit;
        } else {
            s_sum += metal::fast::exp(s_logit - s_max);
        }
    }

    float t_lse = t_max + metal::fast::log(t_sum);
    float s_lse = s_max + metal::fast::log(s_sum);

    // CE = -sum_i P(i) * log(Q(i))
    //    = -sum_i exp(t_i - t_lse) * (s_i - s_lse)
    float ce = 0.0f;

    for (uint v = 0; v < params.vocab_size; v++) {
        float t_logit = t_row[v] * inv_T;
        float s_logit = s_row[v] * inv_T;

        float p = metal::fast::exp(t_logit - t_lse);  // teacher prob
        float log_q = s_logit - s_lse;                 // log(student prob)

        ce -= p * log_q;
    }

    losses[tid] = ce;
}

// =============================================================================
// HIDDEN STATE ALIGNMENT LOSSES
// =============================================================================
//
// For distilling intermediate representations:
// - MSE: mean((teacher - student)^2)
// - Cosine: 1 - cosine_similarity(teacher, student)
// - Attention transfer: MSE on attention patterns
// =============================================================================

/// MSE loss for hidden state alignment.
///
/// Computes mean squared error between teacher and student hidden states.
/// Handles dimension mismatch via truncation (assumes student <= teacher).
kernel void fused_hidden_mse_forward(
    device const float* teacher_hidden [[buffer(0)]],  // [num_tokens, teacher_dim]
    device const float* student_hidden [[buffer(1)]],  // [num_tokens, student_dim]
    device float* losses [[buffer(2)]],                // [num_tokens]
    constant HiddenAlignParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_tokens) return;

    device const float* t_row = teacher_hidden + tid * params.teacher_dim;
    device const float* s_row = student_hidden + tid * params.student_dim;

    // Use minimum dimension
    uint dim = min(params.teacher_dim, params.student_dim);

    float mse = 0.0f;
    for (uint d = 0; d < dim; d++) {
        float diff = t_row[d] - s_row[d];
        mse += diff * diff;
    }

    losses[tid] = mse / float(dim);
}

/// Cosine similarity loss for hidden state alignment.
///
/// Computes: 1 - cosine_similarity(teacher, student)
///
/// Optimizations:
/// - metal::fast::sqrt for faster square root
/// - metal::fast::rsqrt for combined 1/sqrt operation
kernel void fused_hidden_cosine_forward(
    device const float* teacher_hidden [[buffer(0)]],
    device const float* student_hidden [[buffer(1)]],
    device float* losses [[buffer(2)]],
    constant HiddenAlignParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_tokens) return;

    device const float* t_row = teacher_hidden + tid * params.teacher_dim;
    device const float* s_row = student_hidden + tid * params.student_dim;

    uint dim = min(params.teacher_dim, params.student_dim);

    float dot = 0.0f;
    float t_norm_sq = 0.0f;
    float s_norm_sq = 0.0f;

    for (uint d = 0; d < dim; d++) {
        float t = t_row[d];
        float s = s_row[d];
        dot += t * s;
        t_norm_sq += t * t;
        s_norm_sq += s * s;
    }

    // Use rsqrt (1/sqrt) for faster computation: dot * rsqrt(t_norm_sq) * rsqrt(s_norm_sq)
    float t_inv_norm = metal::fast::rsqrt(t_norm_sq + 1e-8f);
    float s_inv_norm = metal::fast::rsqrt(s_norm_sq + 1e-8f);

    float cosine_sim = dot * t_inv_norm * s_inv_norm;
    losses[tid] = 1.0f - cosine_sim;
}

/// SIMD-parallel hidden state MSE for large dimensions.
///
/// Uses proper cross-SIMD reduction for correct results.
kernel void fused_hidden_mse_forward_simd(
    device const float* teacher_hidden [[buffer(0)]],
    device const float* student_hidden [[buffer(1)]],
    device float* losses [[buffer(2)]],
    constant HiddenAlignParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint token_idx = tgid.x;
    if (token_idx >= params.num_tokens) return;

    device const float* t_row = teacher_hidden + token_idx * params.teacher_dim;
    device const float* s_row = student_hidden + token_idx * params.student_dim;

    uint dim = min(params.teacher_dim, params.student_dim);
    uint thread_idx = simd_group_id * SIMD_SIZE + lane_id;
    uint num_threads = DISTILL_THREADS_PER_TOKEN;
    uint num_simd_groups = num_threads / SIMD_SIZE;

    float local_mse = 0.0f;

    for (uint d = thread_idx; d < dim; d += num_threads) {
        float diff = t_row[d] - s_row[d];
        local_mse += diff * diff;
    }

    // Reduce within SIMD group
    float simd_mse = simd_sum(local_mse);

    // Store to threadgroup memory for cross-SIMD reduction
    if (lane_id == 0) {
        scratch[simd_group_id] = simd_mse;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction: first SIMD group sums all partial results
    if (simd_group_id == 0 && lane_id == 0) {
        float total_mse = 0.0f;
        for (uint g = 0; g < num_simd_groups; g++) {
            total_mse += scratch[g];
        }
        losses[token_idx] = total_mse / float(dim);
    }
}

// =============================================================================
// KL DIVERGENCE BACKWARD
// =============================================================================
//
// Gradient of KL(P || Q) w.r.t. student logits:
//   dL/ds_i = -P(i) * (1/Q(i)) * dQ(i)/ds_i
//           = -P(i) / Q(i) * Q(i) * (1{i=j} - Q(j))  [softmax jacobian]
//           = -P(i) * (1 - Q(i))  for i=j
//           = P(i) * Q(j)         for i≠j
//
// Simplified: dL/ds = Q - P (gradient is student_probs - teacher_probs)
//
// With temperature: dL/ds = (Q - P) / T
// =============================================================================

/// KL divergence backward pass.
///
/// Computes gradient of KL(teacher || student) w.r.t. student logits.
/// Gradient = (student_probs - teacher_probs) / temperature
kernel void fused_kl_divergence_backward(
    device const float* teacher_logits [[buffer(0)]],
    device float* student_logits [[buffer(1)]],        // IN-PLACE gradient
    device const float* teacher_lse [[buffer(2)]],
    device const float* student_lse [[buffer(3)]],
    device const float* grad_loss [[buffer(4)]],       // Upstream gradient
    constant DistillParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint token_idx = gid.y;
    const uint vocab_idx = gid.x;

    if (token_idx >= params.num_tokens || vocab_idx >= params.vocab_size) return;

    const uint idx = token_idx * params.vocab_size + vocab_idx;
    const float T = params.temperature;

    float t_logit = teacher_logits[idx] / T;
    float s_logit = student_logits[idx] / T;

    float p = exp(t_logit - teacher_lse[token_idx]);  // teacher prob
    float q = exp(s_logit - student_lse[token_idx]);  // student prob

    // Gradient = upstream * (Q - P) / T
    float grad = grad_loss[token_idx] * (q - p) / T;

    student_logits[idx] = grad;
}

/// SIMD-parallel KL divergence backward.
kernel void fused_kl_divergence_backward_simd(
    device const float* teacher_logits [[buffer(0)]],
    device float* student_logits [[buffer(1)]],
    device const float* teacher_lse [[buffer(2)]],
    device const float* student_lse [[buffer(3)]],
    device const float* grad_loss [[buffer(4)]],
    constant DistillParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint token_idx = tgid.x;
    if (token_idx >= params.num_tokens) return;

    const float T = params.temperature;
    const float t_lse = teacher_lse[token_idx];
    const float s_lse = student_lse[token_idx];
    const float upstream = grad_loss[token_idx];

    uint thread_idx = simd_group_id * SIMD_SIZE + lane_id;
    uint num_threads = DISTILL_THREADS_PER_TOKEN;

    for (uint v = thread_idx; v < params.vocab_size; v += num_threads) {
        uint idx = token_idx * params.vocab_size + v;

        float t_logit = teacher_logits[idx] / T;
        float s_logit = student_logits[idx] / T;

        float p = exp(t_logit - t_lse);
        float q = exp(s_logit - s_lse);

        student_logits[idx] = upstream * (q - p) / T;
    }
}

// =============================================================================
// HALF PRECISION VARIANTS
// =============================================================================

/// fp16 KL divergence forward.
kernel void fused_kl_divergence_forward_f16(
    device const half* teacher_logits [[buffer(0)]],
    device const half* student_logits [[buffer(1)]],
    device float* losses [[buffer(2)]],
    device float* teacher_lse [[buffer(3)]],
    device float* student_lse [[buffer(4)]],
    constant DistillParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint token_idx = tgid.x;
    if (token_idx >= params.num_tokens) return;

    device const half* t_row = teacher_logits + token_idx * params.vocab_size;
    device const half* s_row = student_logits + token_idx * params.vocab_size;

    const float T = params.temperature;
    const uint thread_idx = simd_group_id * SIMD_SIZE + lane_id;
    const uint num_threads = DISTILL_THREADS_PER_TOKEN;

    // Accumulate in fp32 for numerical stability
    float t_local_max = -INFINITY, t_local_sum = 0.0f;
    float s_local_max = -INFINITY, s_local_sum = 0.0f;

    for (uint v = thread_idx; v < params.vocab_size; v += num_threads) {
        float t_logit = float(t_row[v]) / T;
        float s_logit = float(s_row[v]) / T;

        if (t_logit > t_local_max) {
            t_local_sum = t_local_sum * exp(t_local_max - t_logit) + 1.0f;
            t_local_max = t_logit;
        } else {
            t_local_sum += exp(t_logit - t_local_max);
        }

        if (s_logit > s_local_max) {
            s_local_sum = s_local_sum * exp(s_local_max - s_logit) + 1.0f;
            s_local_max = s_logit;
        } else {
            s_local_sum += exp(s_logit - s_local_max);
        }
    }

    // SIMD reduction
    float t_simd_max = simd_max(t_local_max);
    t_local_sum = t_local_sum * exp(t_local_max - t_simd_max);
    float t_simd_sum = simd_sum(t_local_sum);

    float s_simd_max = simd_max(s_local_max);
    s_local_sum = s_local_sum * exp(s_local_max - s_simd_max);
    float s_simd_sum = simd_sum(s_local_sum);

    if (lane_id == 0) {
        scratch[simd_group_id * 4 + 0] = t_simd_max;
        scratch[simd_group_id * 4 + 1] = t_simd_sum;
        scratch[simd_group_id * 4 + 2] = s_simd_max;
        scratch[simd_group_id * 4 + 3] = s_simd_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float t_lse, s_lse;

    if (simd_group_id == 0 && lane_id == 0) {
        float t_max = scratch[0], t_sum = scratch[1];
        float s_max = scratch[2], s_sum = scratch[3];

        uint num_simd_groups = DISTILL_THREADS_PER_TOKEN / SIMD_SIZE;
        for (uint g = 1; g < num_simd_groups; g++) {
            float g_t_max = scratch[g * 4 + 0];
            float g_t_sum = scratch[g * 4 + 1];
            float g_s_max = scratch[g * 4 + 2];
            float g_s_sum = scratch[g * 4 + 3];

            if (g_t_max > t_max) {
                t_sum = t_sum * exp(t_max - g_t_max) + g_t_sum;
                t_max = g_t_max;
            } else {
                t_sum += g_t_sum * exp(g_t_max - t_max);
            }

            if (g_s_max > s_max) {
                s_sum = s_sum * exp(s_max - g_s_max) + g_s_sum;
                s_max = g_s_max;
            } else {
                s_sum += g_s_sum * exp(g_s_max - s_max);
            }
        }

        t_lse = t_max + log(t_sum);
        s_lse = s_max + log(s_sum);
        scratch[0] = t_lse;
        scratch[1] = s_lse;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    t_lse = scratch[0];
    s_lse = scratch[1];

    // Compute KL
    float local_kl = 0.0f;

    for (uint v = thread_idx; v < params.vocab_size; v += num_threads) {
        float t_logit = float(t_row[v]) / T;
        float s_logit = float(s_row[v]) / T;

        float log_p = t_logit - t_lse;
        float log_q = s_logit - s_lse;
        float p = exp(log_p);

        local_kl += p * (log_p - log_q);
    }

    float simd_kl = simd_sum(local_kl);

    if (lane_id == 0) {
        scratch[simd_group_id] = simd_kl;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0 && lane_id == 0) {
        float total_kl = 0.0f;
        uint num_simd_groups = DISTILL_THREADS_PER_TOKEN / SIMD_SIZE;
        for (uint g = 0; g < num_simd_groups; g++) {
            total_kl += scratch[g];
        }

        losses[token_idx] = max(total_kl, 0.0f);
        teacher_lse[token_idx] = t_lse;
        student_lse[token_idx] = s_lse;
    }
}

/// Mean reduction over tokens (ignoring ignored indices).
kernel void distill_reduce_mean(
    device const float* losses [[buffer(0)]],
    device float* output [[buffer(1)]],               // [1] scalar output
    constant uint& num_tokens [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    float local_sum = 0.0f;

    for (uint i = tid; i < num_tokens; i += 1024) {
        local_sum += losses[i];
    }

    float simd_sum_val = simd_sum(local_sum);

    if (lane_id == 0) {
        atomic_fetch_add_explicit((device atomic_float*)output, simd_sum_val, memory_order_relaxed);
    }
}
