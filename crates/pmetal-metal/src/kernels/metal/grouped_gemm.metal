//! Grouped GEMM Metal kernel for MoE models.
//!
//! Performs batched matrix multiplication where different groups (experts)
//! process different subsets of tokens. Supports:
//! - Permute on load (PERMUTE_X): gather input tokens by expert
//! - Permute on store (PERMUTE_Y): scatter output tokens back to original order
//! - Fused weight multiplication for MoE output merging

#include <metal_stdlib>
using namespace metal;

// Tile sizes for GEMM
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 16

/// Parameters for grouped GEMM.
struct GroupedGemmParams {
    uint total_tokens;     // M: Total token-expert pairs
    uint num_experts;      // E: Number of experts
    uint hidden_size;      // K: Input dimension
    uint intermediate;     // N: Output dimension
    uint topk;             // Number of experts per token
    uint permute_x;        // Permute input on load (token -> expert order)
    uint permute_y;        // Permute output on store (expert -> token order)
    uint fuse_mul;         // Fuse weight multiplication
};

/// Standard GEMM for a single tile.
///
/// Computes C[m, n] = A[m, k] @ B[k, n]
inline void gemm_tile(
    device const float* A,
    device const float* B,
    threadgroup float* C_tile,
    uint m_start, uint m_end,
    uint n_start, uint n_end,
    uint K,
    uint stride_a, uint stride_b,
    uint tid_x, uint tid_y,
    uint threads_x, uint threads_y
) {
    uint m_size = m_end - m_start;
    uint n_size = n_end - n_start;

    // Each thread handles a portion of the tile
    for (uint m = tid_y; m < min(m_size, (uint)BLOCK_M); m += threads_y) {
        for (uint n = tid_x; n < min(n_size, (uint)BLOCK_N); n += threads_x) {
            float acc = 0.0f;

            // K-dimension reduction
            for (uint k = 0; k < K; k++) {
                float a_val = A[(m_start + m) * stride_a + k];
                float b_val = B[(n_start + n) * stride_b + k];  // B is transposed: [N, K]
                acc += a_val * b_val;
            }

            C_tile[m * BLOCK_N + n] = acc;
        }
    }
}

/// Grouped GEMM forward pass.
///
/// Computes Y = X @ W^T for each expert where X is optionally permuted.
///
/// Input:
///   x: [M, K] - Input hidden states (may be permuted by expert)
///   w: [E, N, K] - Expert weights (E experts, each [N, K])
///   expert_offsets: [E+1] - Cumulative token counts per expert
///   gather_indices: [M] - Indices for loading X in permuted order
///   scatter_indices: [M] - Indices for storing Y in permuted order
///   topk_weights: [M] - Weights for fused multiplication
///
/// Output:
///   y: [M, N] - Output hidden states
kernel void grouped_gemm_forward(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const uint* expert_offsets [[buffer(3)]],
    device const uint* gather_indices [[buffer(4)]],
    device const uint* scatter_indices [[buffer(5)]],
    device const float* topk_weights [[buffer(6)]],
    constant GroupedGemmParams& params [[buffer(7)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    uint tile_idx = tgid.x;
    uint threads_x = tg_size.x;
    uint threads_y = tg_size.y;
    uint tid_x = tid.x;
    uint tid_y = tid.y;

    uint K = params.hidden_size;
    uint N = params.intermediate;
    uint E = params.num_experts;

    // Find which expert and tile within expert this threadgroup handles
    uint num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;

    uint processed_tiles = 0;
    uint expert_idx = 0;

    // Find expert for this tile
    for (expert_idx = 0; expert_idx < E; expert_idx++) {
        uint m_start = expert_offsets[expert_idx];
        uint m_end = expert_offsets[expert_idx + 1];
        uint m_size = m_end - m_start;

        if (m_size == 0) continue;

        uint num_m_tiles = (m_size + BLOCK_M - 1) / BLOCK_M;
        uint tiles_for_expert = num_m_tiles * num_n_tiles;

        if (tile_idx < processed_tiles + tiles_for_expert) {
            // This tile belongs to this expert
            uint local_tile = tile_idx - processed_tiles;
            uint tile_m_idx = local_tile % num_m_tiles;
            uint tile_n_idx = local_tile / num_m_tiles;

            // Compute tile bounds
            uint tile_m_start = m_start + tile_m_idx * BLOCK_M;
            uint tile_m_end = min(tile_m_start + BLOCK_M, m_end);
            uint tile_n_start = tile_n_idx * BLOCK_N;
            uint tile_n_end = min(tile_n_start + BLOCK_N, N);

            uint tile_m_size = tile_m_end - tile_m_start;
            uint tile_n_size = tile_n_end - tile_n_start;

            // Pointer to this expert's weights: w[expert_idx, :, :]
            device const float* w_expert = w + expert_idx * N * K;

            // Initialize accumulator in shared memory
            threadgroup float* C_tile = scratch;

            // Zero the tile
            for (uint i = tid_y * threads_x + tid_x; i < BLOCK_M * BLOCK_N; i += threads_x * threads_y) {
                C_tile[i] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute GEMM for this tile
            // Each thread handles a subset of (m, n) pairs
            for (uint m = tid_y; m < tile_m_size; m += threads_y) {
                uint global_m = tile_m_start + m;

                // Get input index (with optional permutation)
                uint input_idx;
                if (params.permute_x) {
                    uint sorted_idx = global_m;
                    uint original_idx = gather_indices[sorted_idx];
                    input_idx = original_idx / params.topk;  // Map back to original token
                } else {
                    input_idx = global_m;
                }

                device const float* x_row = x + input_idx * K;

                for (uint n = tid_x; n < tile_n_size; n += threads_x) {
                    uint global_n = tile_n_start + n;

                    // W is [N, K] (transposed)
                    device const float* w_row = w_expert + global_n * K;

                    // Dot product
                    float acc = 0.0f;
                    for (uint k = 0; k < K; k++) {
                        acc += x_row[k] * w_row[k];
                    }

                    C_tile[m * BLOCK_N + n] = acc;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Store output with optional permutation and weight multiplication
            for (uint m = tid_y; m < tile_m_size; m += threads_y) {
                uint global_m = tile_m_start + m;

                for (uint n = tid_x; n < tile_n_size; n += threads_x) {
                    uint global_n = tile_n_start + n;

                    float val = C_tile[m * BLOCK_N + n];

                    // Apply weight multiplication if fused
                    if (params.fuse_mul) {
                        uint weight_idx = params.permute_y ? gather_indices[global_m] : global_m;
                        val *= topk_weights[weight_idx];
                    }

                    // Store with optional permutation
                    uint output_m;
                    if (params.permute_y) {
                        output_m = scatter_indices[global_m];
                    } else {
                        output_m = global_m;
                    }

                    y[output_m * N + global_n] = val;
                }
            }

            return;
        }

        processed_tiles += tiles_for_expert;
    }
}

/// Grouped GEMM backward pass for input gradient (dX).
///
/// Computes dX = dY @ W (not transposed)
///
/// This is similar to forward but uses the untransposed weight.
kernel void grouped_gemm_backward_dx(
    device const float* dy [[buffer(0)]],     // [M, N] gradient w.r.t. output
    device const float* w [[buffer(1)]],      // [E, N, K] expert weights
    device float* dx [[buffer(2)]],           // [M, K] gradient w.r.t. input
    device const uint* expert_offsets [[buffer(3)]],
    device const uint* gather_indices [[buffer(4)]],
    device const uint* scatter_indices [[buffer(5)]],
    device const float* topk_weights [[buffer(6)]],
    constant GroupedGemmParams& params [[buffer(7)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    uint tile_idx = tgid.x;
    uint threads_x = tg_size.x;
    uint threads_y = tg_size.y;
    uint tid_x = tid.x;
    uint tid_y = tid.y;

    uint K = params.hidden_size;  // Output dimension for backward
    uint N = params.intermediate;  // Input dimension for backward
    uint E = params.num_experts;

    uint num_k_tiles = (K + BLOCK_N - 1) / BLOCK_N;  // Note: using BLOCK_N for K

    uint processed_tiles = 0;

    for (uint expert_idx = 0; expert_idx < E; expert_idx++) {
        uint m_start = expert_offsets[expert_idx];
        uint m_end = expert_offsets[expert_idx + 1];
        uint m_size = m_end - m_start;

        if (m_size == 0) continue;

        uint num_m_tiles = (m_size + BLOCK_M - 1) / BLOCK_M;
        uint tiles_for_expert = num_m_tiles * num_k_tiles;

        if (tile_idx < processed_tiles + tiles_for_expert) {
            uint local_tile = tile_idx - processed_tiles;
            uint tile_m_idx = local_tile % num_m_tiles;
            uint tile_k_idx = local_tile / num_m_tiles;

            uint tile_m_start = m_start + tile_m_idx * BLOCK_M;
            uint tile_m_end = min(tile_m_start + BLOCK_M, m_end);
            uint tile_k_start = tile_k_idx * BLOCK_N;
            uint tile_k_end = min(tile_k_start + BLOCK_N, K);

            uint tile_m_size = tile_m_end - tile_m_start;
            uint tile_k_size = tile_k_end - tile_k_start;

            device const float* w_expert = w + expert_idx * N * K;

            threadgroup float* C_tile = scratch;

            // Zero
            for (uint i = tid_y * threads_x + tid_x; i < BLOCK_M * BLOCK_N; i += threads_x * threads_y) {
                C_tile[i] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // dX[m, k] = sum_n dY[m, n] * W[n, k]
            for (uint m = tid_y; m < tile_m_size; m += threads_y) {
                uint global_m = tile_m_start + m;

                // Get dY index (with permutation if needed)
                uint dy_m = params.permute_y ? scatter_indices[global_m] : global_m;
                device const float* dy_row = dy + dy_m * N;

                // Weight for this token (for gradient scaling)
                float weight = 1.0f;
                if (params.fuse_mul) {
                    uint weight_idx = params.permute_x ? gather_indices[global_m] : global_m;
                    weight = topk_weights[weight_idx];
                }

                for (uint k = tid_x; k < tile_k_size; k += threads_x) {
                    uint global_k = tile_k_start + k;

                    float acc = 0.0f;
                    for (uint n = 0; n < N; n++) {
                        // W is [N, K]
                        acc += dy_row[n] * w_expert[n * K + global_k];
                    }

                    C_tile[m * BLOCK_N + k] = acc * weight;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Store dX with permutation
            for (uint m = tid_y; m < tile_m_size; m += threads_y) {
                uint global_m = tile_m_start + m;

                for (uint k = tid_x; k < tile_k_size; k += threads_x) {
                    uint global_k = tile_k_start + k;

                    float val = C_tile[m * BLOCK_N + k];

                    uint dx_m;
                    if (params.permute_x) {
                        uint original_idx = gather_indices[global_m];
                        dx_m = original_idx / params.topk;
                    } else {
                        dx_m = global_m;
                    }

                    // Atomic add since multiple experts may write to same token
                    atomic_fetch_add_explicit(
                        (device atomic_float*)(dx + dx_m * K + global_k),
                        val,
                        memory_order_relaxed
                    );
                }
            }

            return;
        }

        processed_tiles += tiles_for_expert;
    }
}

/// Grouped GEMM backward pass for weight gradient (dW).
///
/// Computes dW[e] = X^T @ dY for each expert
kernel void grouped_gemm_backward_dw(
    device const float* x [[buffer(0)]],      // [M, K] input
    device const float* dy [[buffer(1)]],     // [M, N] gradient w.r.t. output
    device float* dw [[buffer(2)]],           // [E, N, K] gradient w.r.t. weights
    device const uint* expert_offsets [[buffer(3)]],
    device const uint* gather_indices [[buffer(4)]],
    device const uint* scatter_indices [[buffer(5)]],
    constant GroupedGemmParams& params [[buffer(6)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    uint tile_idx = tgid.x;
    uint threads_x = tg_size.x;
    uint threads_y = tg_size.y;
    uint tid_x = tid.x;
    uint tid_y = tid.y;

    uint K = params.hidden_size;
    uint N = params.intermediate;
    uint E = params.num_experts;

    uint num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    uint num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;
    uint tiles_per_expert = num_n_tiles * num_k_tiles;

    // Each threadgroup handles one (n, k) tile for one expert
    uint expert_idx = tile_idx / tiles_per_expert;
    if (expert_idx >= E) return;

    uint local_tile = tile_idx % tiles_per_expert;
    uint tile_n_idx = local_tile / num_k_tiles;
    uint tile_k_idx = local_tile % num_k_tiles;

    uint m_start = expert_offsets[expert_idx];
    uint m_end = expert_offsets[expert_idx + 1];
    uint m_size = m_end - m_start;

    if (m_size == 0) return;

    uint tile_n_start = tile_n_idx * BLOCK_N;
    uint tile_n_end = min(tile_n_start + BLOCK_N, N);
    uint tile_k_start = tile_k_idx * BLOCK_K;
    uint tile_k_end = min(tile_k_start + BLOCK_K, K);

    uint tile_n_size = tile_n_end - tile_n_start;
    uint tile_k_size = tile_k_end - tile_k_start;

    threadgroup float* dW_tile = scratch;

    // Zero
    for (uint i = tid_y * threads_x + tid_x; i < BLOCK_N * BLOCK_K; i += threads_x * threads_y) {
        dW_tile[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dW[n, k] = sum_m X[m, k] * dY[m, n]
    // Reduction over M dimension
    for (uint m = m_start; m < m_end; m++) {
        // Get indices with permutation
        uint x_m, dy_m;
        if (params.permute_x) {
            uint original_idx = gather_indices[m];
            x_m = original_idx / params.topk;
        } else {
            x_m = m;
        }

        if (params.permute_y) {
            dy_m = scatter_indices[m];
        } else {
            dy_m = m;
        }

        device const float* x_row = x + x_m * K;
        device const float* dy_row = dy + dy_m * N;

        // Each thread handles a subset of (n, k) pairs
        for (uint n = tid_y; n < tile_n_size; n += threads_y) {
            uint global_n = tile_n_start + n;
            float dy_val = dy_row[global_n];

            for (uint k = tid_x; k < tile_k_size; k += threads_x) {
                uint global_k = tile_k_start + k;
                float x_val = x_row[global_k];

                dW_tile[n * BLOCK_K + k] += x_val * dy_val;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store to global memory
    device float* dw_expert = dw + expert_idx * N * K;

    for (uint n = tid_y; n < tile_n_size; n += threads_y) {
        uint global_n = tile_n_start + n;

        for (uint k = tid_x; k < tile_k_size; k += threads_x) {
            uint global_k = tile_k_start + k;

            dw_expert[global_n * K + global_k] = dW_tile[n * BLOCK_K + k];
        }
    }
}
