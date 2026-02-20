//! MoE (Mixture of Experts) routing Metal kernels.
//!
//! This kernel handles expert routing for MoE models:
//! 1. TopK selection from router logits
//! 2. Computing token counts per expert
//! 3. Computing gather/scatter indices for permutation

#include <metal_stdlib>
using namespace metal;

/// Parameters for MoE routing kernel.
struct MoeRoutingParams {
    uint num_tokens;      // Number of input tokens
    uint num_experts;     // Number of experts
    uint topk;            // Number of experts per token
    uint use_sigmoid;     // Use sigmoid (1) or softmax (0)
    uint renormalize;     // Renormalize weights after topk
};

/// TopK selection for MoE routing.
///
/// Computes softmax/sigmoid over router logits, then selects top-k experts.
///
/// Input:
///   router_logits: [num_tokens, num_experts]
/// Output:
///   topk_weights: [num_tokens, topk]
///   topk_ids: [num_tokens, topk]
kernel void moe_topk_selection(
    device const float* router_logits [[buffer(0)]],
    device float* topk_weights [[buffer(1)]],
    device uint* topk_ids [[buffer(2)]],
    constant MoeRoutingParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_tokens) return;

    uint num_experts = params.num_experts;
    // Clamp topk to num_experts to prevent out-of-bounds reads and garbage output
    // when the caller passes topk > num_experts.
    uint effective_topk = min(params.topk, num_experts);

    // Pointer to this token's logits
    device const float* logits = router_logits + tid * num_experts;

    // Thread-private score buffer (not threadgroup — each thread owns its own token's scores).
    // Declared at kernel function scope as required by MSL.
    float scores[256]; // Max experts

    if (params.use_sigmoid) {
        // Sigmoid activation
        for (uint e = 0; e < num_experts; e++) {
            scores[e] = 1.0f / (1.0f + exp(-logits[e]));
        }
    } else {
        // Softmax: find max for numerical stability
        float max_val = logits[0];
        for (uint e = 1; e < num_experts; e++) {
            max_val = max(max_val, logits[e]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (uint e = 0; e < num_experts; e++) {
            scores[e] = exp(logits[e] - max_val);
            sum += scores[e];
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (uint e = 0; e < num_experts; e++) {
            scores[e] *= inv_sum;
        }
    }

    // TopK selection using partial sort (bounded by effective_topk <= num_experts)
    device float* out_weights = topk_weights + tid * effective_topk;
    device uint* out_ids = topk_ids + tid * effective_topk;

    for (uint k = 0; k < effective_topk; k++) {
        float best_score = -1e10f;
        uint best_idx = 0;

        for (uint e = 0; e < num_experts; e++) {
            if (scores[e] > best_score) {
                best_score = scores[e];
                best_idx = e;
            }
        }

        out_weights[k] = best_score;
        out_ids[k] = best_idx;
        scores[best_idx] = -1e10f; // Mark as selected
    }

    // Renormalize if needed
    if (params.renormalize) {
        float sum = 0.0f;
        for (uint k = 0; k < effective_topk; k++) {
            sum += out_weights[k];
        }
        float inv_sum = 1.0f / sum;
        for (uint k = 0; k < effective_topk; k++) {
            out_weights[k] *= inv_sum;
        }
    }
}

/// Compute per-expert token counts and gather indices.
///
/// This is a two-phase algorithm:
/// 1. Histogram: Count tokens per expert
/// 2. Scatter: Compute gather indices for sorting tokens by expert
///
/// Input:
///   topk_ids: [num_tokens, topk]
/// Output:
///   token_counts: [num_experts]
///   gather_indices: [num_tokens * topk]
kernel void moe_compute_indices(
    device const uint* topk_ids [[buffer(0)]],
    device atomic_uint* token_counts [[buffer(1)]],
    device uint* gather_indices [[buffer(2)]],
    device uint* expert_offsets [[buffer(3)]],  // Prefix sum workspace
    constant MoeRoutingParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total_tokens = params.num_tokens * params.topk;

    if (tid >= total_tokens) return;

    // Get expert for this token-expert pair
    uint expert_id = topk_ids[tid];

    // Atomically increment the per-expert token count (pos is intentionally discarded;
    // the final sorted position is computed in moe_sort_indices using expert_offsets).
    atomic_fetch_add_explicit(&token_counts[expert_id], 1, memory_order_relaxed);

    // Store the expert assignment; moe_sort_indices will later scatter into sorted order.
    gather_indices[tid] = expert_id;
}

/// Parameters for grouped GEMM kernel.
struct GroupedGemmParams {
    uint total_tokens;    // Total token-expert pairs (num_tokens * topk)
    uint num_experts;     // Number of experts
    uint hidden_size;     // K dimension (input hidden size)
    uint intermediate;    // N dimension (intermediate size)
    uint topk;            // Number of experts per token
    uint permute_x;       // Permute input on load
    uint permute_y;       // Permute output on store
    uint fuse_mul;        // Fuse weight multiplication
};

/// Compute expert offset prefix sum.
///
/// Input:
///   token_counts: [num_experts]
/// Output:
///   expert_offsets: [num_experts + 1]
kernel void moe_compute_expert_offsets(
    device const uint* token_counts [[buffer(0)]],
    device uint* expert_offsets [[buffer(1)]],
    constant uint& num_experts [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Single-threaded prefix sum (num_experts is small)
    if (tid != 0) return;

    uint sum = 0;
    for (uint e = 0; e < num_experts; e++) {
        expert_offsets[e] = sum;
        sum += token_counts[e];
    }
    expert_offsets[num_experts] = sum;
}

/// Sort gather indices by expert.
///
/// Uses the precomputed expert offsets to scatter indices into sorted order.
///
/// Input:
///   topk_ids: [num_tokens, topk] - expert assignments
///   expert_offsets: [num_experts + 1] - prefix sums
/// Output:
///   sorted_indices: [num_tokens * topk] - indices sorted by expert
///   scatter_indices: [num_tokens * topk] - inverse permutation
kernel void moe_sort_indices(
    device const uint* topk_ids [[buffer(0)]],
    device const uint* expert_offsets [[buffer(1)]],
    device atomic_uint* expert_counters [[buffer(2)]],  // Temporary counters
    device uint* sorted_indices [[buffer(3)]],
    device uint* scatter_indices [[buffer(4)]],
    constant MoeRoutingParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total_tokens = params.num_tokens * params.topk;

    if (tid >= total_tokens) return;

    // Get expert for this token-expert pair
    uint expert_id = topk_ids[tid];

    // Get base offset for this expert
    uint base_offset = expert_offsets[expert_id];

    // Atomically claim a position within this expert's range
    uint local_pos = atomic_fetch_add_explicit(&expert_counters[expert_id], 1, memory_order_relaxed);

    uint sorted_pos = base_offset + local_pos;

    // Store gather index (sorted -> original)
    sorted_indices[sorted_pos] = tid;

    // Store scatter index (original -> sorted)
    scatter_indices[tid] = sorted_pos;
}
