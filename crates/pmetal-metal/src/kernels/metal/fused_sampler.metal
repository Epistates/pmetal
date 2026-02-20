//
//  fused_sampler.metal
//  Fused sampling kernel for LLM token generation.
//
//  This kernel fuses all sampling operations into a single GPU pass:
//  1. Argmax (greedy) or temperature scaling
//  2. Top-K filtering
//  3. Top-P (nucleus) filtering
//  4. Min-P filtering
//  5. Categorical sampling
//
//  Performance benefits:
//  - Single kernel launch vs 10+ separate launches
//  - No intermediate memory allocations
//  - Minimal CPU-GPU coordination overhead
//
//  Copyright 2024 PMetal Authors. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

// Sampling parameters passed from CPU
struct SamplingParams {
    uint vocab_size;           // Vocabulary size
    float temperature;         // Sampling temperature (0 = greedy)
    float top_p;              // Nucleus sampling threshold
    float min_p;              // Min-p threshold (relative to max)
    int top_k;                // Top-k value (0 = disabled)
    uint random_seed;         // Random seed for sampling
    bool do_sample;           // Whether to sample (false = greedy argmax)
};

// Threadgroup shared memory for parallel reduction
// Max supported vocab size: 256K tokens with 1024 threads
constant uint THREADGROUP_SIZE = 256;
constant uint MAX_TOP_K = 128;  // Maximum top-k we track in shared memory

// ============================================================================
// Greedy Argmax Kernel (Temperature = 0)
// ============================================================================
//
// Simple parallel reduction to find the maximum logit index.
// Uses two-phase reduction: within threadgroups, then across threadgroups.

kernel void fused_argmax(
    device const float* logits [[buffer(0)]],
    device uint* output_token [[buffer(1)]],
    constant SamplingParams& params [[buffer(2)]],
    threadgroup float* shared_max [[threadgroup(0)]],
    threadgroup uint* shared_idx [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint threads_per_tg [[threads_per_threadgroup]]
) {
    uint vocab_size = params.vocab_size;

    // Phase 1: Each thread finds max in its portion
    float local_max = -INFINITY;
    uint local_idx = 0;

    for (uint i = tid; i < vocab_size; i += threads_per_tg) {
        float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // Store in shared memory
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Parallel reduction in shared memory
    for (uint stride = threads_per_tg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_max[tid + stride] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the result
    if (tid == 0) {
        *output_token = shared_idx[0];
    }
}

// ============================================================================
// Fused Sampling Kernel (Temperature > 0)
// ============================================================================
//
// Full sampling pipeline with top-k, top-p, min-p filtering.
// Algorithm:
// 1. Find global max for numerical stability
// 2. Compute log-softmax probabilities
// 3. Apply top-k: keep only top K highest probability tokens
// 4. Apply min-p: filter tokens below min_p * max_prob threshold
// 5. Apply top-p: keep tokens until cumulative prob exceeds top_p
// 6. Sample from remaining distribution using inverse CDF
//
// For vocab sizes > threadgroup capacity, uses multi-pass approach.

// Simple xorshift random number generator
inline float xorshift_random(thread uint& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(state) / float(0xFFFFFFFFu);
}

// HOST ALLOCATION NOTE for fused_sample_small:
//   shared_logprobs [[threadgroup(0)]]: must be allocated as
//       threads_per_tg * LOCAL_TOP_L * sizeof(float)   (= threads_per_tg * 4 * 4 bytes)
//   shared_indices  [[threadgroup(1)]]: must be allocated as
//       threads_per_tg * LOCAL_TOP_L * sizeof(uint)    (= threads_per_tg * 4 * 4 bytes)
// where LOCAL_TOP_L = 4 (per-thread candidate count defined below).
// Example: with 256 threads -> 256 * 4 * 4 = 4096 bytes each.
kernel void fused_sample_small(
    device const float* logits [[buffer(0)]],
    device uint* output_token [[buffer(1)]],
    constant SamplingParams& params [[buffer(2)]],
    threadgroup float* shared_logprobs [[threadgroup(0)]],  // threads_per_tg * LOCAL_TOP_L floats
    threadgroup uint* shared_indices [[threadgroup(1)]],    // threads_per_tg * LOCAL_TOP_L uints
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]]
) {
    uint vocab_size = params.vocab_size;
    float temperature = params.temperature;
    float top_p = params.top_p;
    float min_p = params.min_p;
    int top_k = params.top_k;

    // Step 1: Find global max for numerical stability (parallel reduction)
    float local_max = -INFINITY;
    for (uint i = tid; i < vocab_size; i += threads_per_tg) {
        local_max = max(local_max, logits[i]);
    }

    shared_logprobs[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_tg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_logprobs[tid] = max(shared_logprobs[tid], shared_logprobs[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared_logprobs[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute exp(logits - max) / temperature and sum for normalization
    float local_sum = 0.0f;
    for (uint i = tid; i < vocab_size; i += threads_per_tg) {
        float scaled = (logits[i] - global_max) / temperature;
        local_sum += exp(scaled);
    }

    shared_logprobs[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_tg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_logprobs[tid] += shared_logprobs[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total_sum = shared_logprobs[0];
    float log_sum = log(total_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Find top-k candidates and their probabilities
    // Algorithm: Per-thread local top-L tracking + single-thread merge
    // This avoids race conditions by having each thread work on local data,
    // then merging in a single thread.

    // Phase 1: Each thread finds its LOCAL top-4 candidates (no synchronization needed)
    // L=4 is sufficient since vocab tokens are typically spread across many threads
    const uint LOCAL_TOP_L = 4;
    float local_probs[LOCAL_TOP_L];
    uint local_tokens[LOCAL_TOP_L];

    // Initialize thread-local arrays
    for (uint i = 0; i < LOCAL_TOP_L; i++) {
        local_probs[i] = -INFINITY;
        local_tokens[i] = 0;
    }

    // Each thread scans its portion of vocabulary
    for (uint i = tid; i < vocab_size; i += threads_per_tg) {
        float prob = exp((logits[i] - global_max) / temperature - log_sum);

        // Insert into thread-local top-L using sorted insertion (no race condition)
        if (prob > local_probs[LOCAL_TOP_L - 1]) {
            for (uint j = 0; j < LOCAL_TOP_L; j++) {
                if (prob > local_probs[j]) {
                    // Shift down and insert
                    for (uint k = LOCAL_TOP_L - 1; k > j; k--) {
                        local_probs[k] = local_probs[k - 1];
                        local_tokens[k] = local_tokens[k - 1];
                    }
                    local_probs[j] = prob;
                    local_tokens[j] = i;
                    break;
                }
            }
        }
    }

    // Phase 2: Write local candidates to shared memory
    // Each thread writes its LOCAL_TOP_L candidates to a dedicated slot
    // Shared memory layout: [thread0_L0, thread0_L1, ..., thread0_L3, thread1_L0, ...]
    uint base_idx = tid * LOCAL_TOP_L;
    for (uint i = 0; i < LOCAL_TOP_L; i++) {
        shared_logprobs[base_idx + i] = local_probs[i];
        shared_indices[base_idx + i] = local_tokens[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Thread 0 merges all candidates into final top-K (serialized - no race)
    // Threadgroup arrays must be declared at kernel function scope (MSL requirement),
    // not inside conditional blocks.
    threadgroup float top_probs[MAX_TOP_K];
    threadgroup uint top_tokens[MAX_TOP_K];

    uint effective_k = (top_k > 0 && uint(top_k) < MAX_TOP_K) ? uint(top_k) : MAX_TOP_K;

    if (tid == 0) {
        // Initialize top-K with -INFINITY
        for (uint i = 0; i < effective_k; i++) {
            top_probs[i] = -INFINITY;
            top_tokens[i] = 0;
        }

        // Process all candidates from all threads (threads_per_tg * LOCAL_TOP_L entries)
        uint total_candidates = threads_per_tg * LOCAL_TOP_L;
        for (uint c = 0; c < total_candidates; c++) {
            float prob = shared_logprobs[c];

            // Skip invalid candidates (still -INFINITY)
            if (prob <= -1e30f) continue;

            // Insert into top-K if qualifying
            if (prob > top_probs[effective_k - 1]) {
                for (uint j = 0; j < effective_k; j++) {
                    if (prob > top_probs[j]) {
                        // Shift down and insert
                        for (uint k = effective_k - 1; k > j; k--) {
                            top_probs[k] = top_probs[k - 1];
                            top_tokens[k] = top_tokens[k - 1];
                        }
                        top_probs[j] = prob;
                        top_tokens[j] = shared_indices[c];
                        break;
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Apply min-p filtering
    float max_prob = top_probs[0];
    float min_p_threshold = min_p * max_prob;

    // Step 5: Apply top-p filtering and compute cumulative sum
    float cumsum = 0.0f;
    uint final_count = 0;

    if (tid == 0) {
        for (uint i = 0; i < effective_k; i++) {
            if (top_probs[i] < min_p_threshold) break;  // min-p filter
            cumsum += top_probs[i];
            final_count = i + 1;
            if (cumsum >= top_p) break;  // top-p filter
        }

        // Ensure at least one token
        if (final_count == 0) final_count = 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 6: Sample from filtered distribution
    if (tid == 0) {
        // Renormalize the filtered probabilities
        float filtered_sum = 0.0f;
        for (uint i = 0; i < final_count; i++) {
            filtered_sum += top_probs[i];
        }

        // Generate random number using xorshift
        uint rng_state = params.random_seed;
        float r = xorshift_random(rng_state) * filtered_sum;

        // Sample using inverse CDF
        float running_sum = 0.0f;
        uint sampled_token = top_tokens[0];  // default

        for (uint i = 0; i < final_count; i++) {
            running_sum += top_probs[i];
            if (running_sum >= r) {
                sampled_token = top_tokens[i];
                break;
            }
        }

        *output_token = sampled_token;
    }
}

// ============================================================================
// Optimized Argmax with Top-K Tracking
// ============================================================================
//
// For greedy decoding, we just need argmax.
// This version is highly optimized using SIMD operations.

kernel void fused_argmax_simd(
    device const float* logits [[buffer(0)]],
    device uint* output_token [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]]
) {
    // Use SIMD operations for faster reduction
    float local_max = -INFINITY;
    uint local_idx = 0;

    // Coalesced memory access
    for (uint i = tid; i < vocab_size; i += threads_per_tg) {
        float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // SIMD reduction within simdgroup (32 threads on Apple Silicon)
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_max = simd_shuffle_down(local_max, offset);
        uint other_idx = simd_shuffle_down(local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    // Now lane 0 of each simdgroup has the local max
    threadgroup float simd_max[8];  // Max 8 simdgroups (256 threads)
    threadgroup uint simd_idx[8];

    if (simd_lane == 0) {
        simd_max[simd_group] = local_max;
        simd_idx[simd_group] = local_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by thread 0
    if (tid == 0) {
        float best_max = simd_max[0];
        uint best_idx = simd_idx[0];
        uint num_simdgroups = (threads_per_tg + 31) / 32;

        for (uint i = 1; i < num_simdgroups; i++) {
            if (simd_max[i] > best_max) {
                best_max = simd_max[i];
                best_idx = simd_idx[i];
            }
        }

        *output_token = best_idx;
    }
}
