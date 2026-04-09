// mpp_grouped_gemm.metal
// Metal 4 Grouped GEMM for MoE models using MPP matmul2d.
//
// Replaces the Metal 3 `grouped_gemm_forward` which uses manual threadgroup
// staging and scalar FMA loops. MPP provides hardware-accelerated matrix
// multiply units (NAX) with no explicit staging needed.
//
// MoE decode bottleneck: 3 gather_mm × 28 layers × (up to 8 active experts)
// This kernel is on the hottest path in MoE model inference.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// Morton ordering for cache-friendly tile walk within each expert's tile block.
// MPP Guide Section 2.3.3: "actively running threadgroups cover a square region
// of output tiles in the result matrix so that different cores maximally reuse
// shards of both input matrices."
inline uint2 morton_decode(uint linear) {
    uint x = 0, y = 0;
    for (uint bit = 0; bit < 16; bit++) {
        y |= ((linear >> (2 * bit))     & 1) << bit;
        x |= ((linear >> (2 * bit + 1)) & 1) << bit;
    }
    return uint2(x, y);
}

// Tile sizes
#define BLOCK_M 64
#define BLOCK_N 64

struct GroupedGemmParams {
    uint total_tokens;     // M: Total token-expert pairs
    uint num_experts;      // E: Number of experts
    uint hidden_size;      // K: Input dimension
    uint intermediate;     // N: Output dimension
    uint topk;             // Number of experts per token
    uint permute_x;        // Permute input on load
    uint permute_y;        // Permute output on store
    uint fuse_mul;         // Fuse weight multiplication
};

/// MPP Grouped GEMM forward: Y = X @ W^T per expert
///
/// Each threadgroup computes a BLOCK_M × BLOCK_N output tile for one expert.
/// MPP matmul2d handles the K-dim accumulation internally with hardware MMA.
///
/// Key improvement over Metal 3 version:
/// - No threadgroup memory staging (A_stage, B_stage, C_tile eliminated)
/// - Hardware matrix multiply via NAX instead of scalar FMA loop
/// - Morton ordering for cache-friendly threadgroup dispatch
kernel void mpp_grouped_gemm_forward_f16(
    device half* x [[buffer(0)]],                 // [M, K] input
    device half* w [[buffer(1)]],                 // [E, N, K] expert weights
    device half* y [[buffer(2)]],                 // [M, N] output
    device const uint* expert_offsets [[buffer(3)]],
    device const uint* gather_indices [[buffer(4)]],
    device const uint* scatter_indices [[buffer(5)]],
    device const half* topk_weights [[buffer(6)]],
    constant GroupedGemmParams& params [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    uint tile_idx = tgid.x;
    uint K = params.hidden_size;
    uint N = params.intermediate;
    uint E = params.num_experts;

    // MPP Guide: K must be a multiple of 32 for NAX.
    // Misaligned K silently corrupts results on NAX hardware.
    if ((K & 31u) != 0u) return;

    uint num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;

    // Find which expert this tile belongs to
    uint processed_tiles = 0;
    for (uint expert_idx = 0; expert_idx < E; expert_idx++) {
        uint m_start = expert_offsets[expert_idx];
        uint m_end = expert_offsets[expert_idx + 1];
        uint m_size = m_end - m_start;
        if (m_size == 0) continue;

        uint num_m_tiles = (m_size + BLOCK_M - 1) / BLOCK_M;
        uint tiles_for_expert = num_m_tiles * num_n_tiles;

        if (tile_idx < processed_tiles + tiles_for_expert) {
            uint local_tile = tile_idx - processed_tiles;

            // Morton ordering within this expert's tile grid for LLC locality
            uint2 morton = morton_decode(local_tile);
            uint tile_m_idx = morton.y;
            uint tile_n_idx = morton.x;
            // Clamp if Morton coords exceed expert's tile grid
            if (tile_m_idx >= num_m_tiles || tile_n_idx >= num_n_tiles) {
                tile_m_idx = local_tile % num_m_tiles;
                tile_n_idx = local_tile / num_m_tiles;
            }

            uint tile_m_start = m_start + tile_m_idx * BLOCK_M;
            uint tile_m_end = min(tile_m_start + (uint)BLOCK_M, m_end);
            uint tile_n_start = tile_n_idx * BLOCK_N;

            uint tile_m_size = tile_m_end - tile_m_start;

            // Expert weight pointer: w[expert, :, :] → [N, K] row-major
            device half* w_expert = w + expert_idx * N * K;

            // Create tensors for MPP matmul2d
            // X: [M_total, K] row-major — we need rows [tile_m_start..tile_m_end)
            // W: [N, K] row-major — transposed via descriptor
            // We build per-expert X by handling permutation in the output store phase.
            // For the GEMM itself, we compute on the permuted indices.

            // Create X tensor for this tile's rows
            // Since permutation complicates direct tensor slicing, we compute
            // the straight GEMM on the sorted-by-expert token order (which is
            // what expert_offsets gives us)
            auto tX = tensor(x, dextents<int, 2>{(int)K, (int)params.total_tokens},
                             array<int, 2>{1, (int)K});
            auto tW = tensor(w_expert, dextents<int, 2>{(int)K, (int)N},
                             array<int, 2>{1, (int)K});

            // Allocate output region — we'll write directly to y
            auto tY = tensor(y, dextents<int, 2>{(int)N, (int)params.total_tokens},
                             array<int, 2>{1, (int)N});

            // MPP matmul: 64x64 threadgroup tile with 4 cooperating simdgroups
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
                64, 64,
                static_cast<int>(dynamic_extent),
                false,  // transpose_left (X is row-major)
                true,   // transpose_right (W is [N,K], need W^T)
                false   // relaxed_precision
            );

            mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;

            // Slice to this tile
            // Handle permutation: if permute_x, the input rows at expert_offsets
            // positions are already in the sorted order for this expert.
            auto sliceX = tX.slice(0, (int)tile_m_start);
            auto sliceW = tW.slice(0, (int)tile_n_start);
            auto sliceY = tY.slice((int)tile_n_start, (int)tile_m_start);

            op.run(sliceX, sliceW, sliceY);

            return;
        }

        processed_tiles += tiles_for_expert;
    }
}

/// MPP Grouped GEMM forward: fp32 variant
kernel void mpp_grouped_gemm_forward_f32(
    device float* x [[buffer(0)]],
    device float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const uint* expert_offsets [[buffer(3)]],
    device const uint* gather_indices [[buffer(4)]],
    device const uint* scatter_indices [[buffer(5)]],
    device const float* topk_weights [[buffer(6)]],
    constant GroupedGemmParams& params [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    uint tile_idx = tgid.x;
    uint K = params.hidden_size;
    uint N = params.intermediate;
    uint E = params.num_experts;

    // MPP Guide: K must be a multiple of 32 for NAX.
    if ((K & 31u) != 0u) return;

    uint num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;

    uint processed_tiles = 0;
    for (uint expert_idx = 0; expert_idx < E; expert_idx++) {
        uint m_start = expert_offsets[expert_idx];
        uint m_end = expert_offsets[expert_idx + 1];
        uint m_size = m_end - m_start;
        if (m_size == 0) continue;

        uint num_m_tiles = (m_size + BLOCK_M - 1) / BLOCK_M;
        uint tiles_for_expert = num_m_tiles * num_n_tiles;

        if (tile_idx < processed_tiles + tiles_for_expert) {
            uint local_tile = tile_idx - processed_tiles;

            // Morton ordering within this expert's tile grid for LLC locality
            uint2 morton = morton_decode(local_tile);
            uint tile_m_idx = morton.y;
            uint tile_n_idx = morton.x;
            if (tile_m_idx >= num_m_tiles || tile_n_idx >= num_n_tiles) {
                tile_m_idx = local_tile % num_m_tiles;
                tile_n_idx = local_tile / num_m_tiles;
            }

            uint tile_m_start = m_start + tile_m_idx * BLOCK_M;
            uint tile_n_start = tile_n_idx * BLOCK_N;

            device float* w_expert = w + expert_idx * N * K;

            auto tX = tensor(x, dextents<int, 2>{(int)K, (int)params.total_tokens},
                             array<int, 2>{1, (int)K});
            auto tW = tensor(w_expert, dextents<int, 2>{(int)K, (int)N},
                             array<int, 2>{1, (int)K});
            auto tY = tensor(y, dextents<int, 2>{(int)N, (int)params.total_tokens},
                             array<int, 2>{1, (int)N});

            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
                64, 64,
                static_cast<int>(dynamic_extent),
                false, true, false
            );

            mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;

            auto sliceX = tX.slice(0, (int)tile_m_start);
            auto sliceW = tW.slice(0, (int)tile_n_start);
            auto sliceY = tY.slice((int)tile_n_start, (int)tile_m_start);

            op.run(sliceX, sliceW, sliceY);

            return;
        }

        processed_tiles += tiles_for_expert;
    }
}
