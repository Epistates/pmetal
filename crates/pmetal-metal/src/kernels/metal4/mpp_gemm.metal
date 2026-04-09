// mpp_gemm.metal
// Metal 4 GEMM kernels using Metal Performance Primitives (MPP).
//
// Uses mpp::tensor_ops::matmul2d for hardware-accelerated matrix multiplication
// on M5 (Apple10) NAX cores.
//
// Key advantages over Metal 3 manual GEMM:
// - Hardware matrix multiply units (NAX)
// - No explicit threadgroup memory staging needed
// - Static tensor extents eliminate bounds checking for aligned tiles
//
// References:
// - Metal Performance Primitives Programming Guide (Apple, 2026)
// - MLX steel/gemm/nax.h (upstream reference implementation)

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// =============================================================================
// Morton ordering for threadgroup walk order
// =============================================================================

/// Decode a linearized index into 2D (x, y) coordinates via Morton Z-curve.
/// MPP Guide Section 2.3.3: Preserves spatial locality for LLC cache reuse.
inline uint2 morton_decode(uint linear) {
    uint x = 0, y = 0;
    for (uint bit = 0; bit < 16; bit++) {
        y |= ((linear >> (2 * bit))     & 1) << bit;
        x |= ((linear >> (2 * bit + 1)) & 1) << bit;
    }
    return uint2(x, y);
}

// =============================================================================
// Configuration
// =============================================================================

/// Function constants for compile-time specialization
constant bool FC_MORTON    [[function_constant(0)]];  // Use Morton ordering
constant bool FC_M_ALIGNED [[function_constant(1)]];  // M is a multiple of 64
constant bool FC_N_ALIGNED [[function_constant(2)]];  // N is a multiple of 64

struct MppGemmParams {
    uint M;           // Output rows
    uint N;           // Output columns
    uint K;           // Reduction dimension
    float alpha;      // Scalar multiplier
    float beta;       // Accumulate multiplier (0 = overwrite, 1 = accumulate)
    uint num_tiles_m; // Total M tiles (for Morton decode)
    uint num_tiles_n; // Total N tiles (for Morton decode)
};

// =============================================================================
// MPP GEMM helper variants.
//
// MPP Guide Section 2.3.1: "single simdgroup execution always resulted in
// significant performance drop" applies to naive multi-group configurations.
// All kernel entry points below use execution_simdgroup (single simdgroup).
// The function name suffix (sg1/sg2/sg4) encodes the *output tile size* chosen
// by the Rust auto-tuner, NOT the number of simdgroups.  A sg4_64x64 kernel
// still runs as one simdgroup over a 64x64 tile — the larger tile increases
// arithmetic intensity without the coordination overhead of multiple simdgroups.

inline uint2 decode_output_tile(uint linear, constant MppGemmParams& params) {
    uint2 tile;
    if (FC_MORTON) {
        tile = morton_decode(linear);
        if (tile.y >= params.num_tiles_m || tile.x >= params.num_tiles_n) {
            tile.y = linear / params.num_tiles_n;
            tile.x = linear % params.num_tiles_n;
        }
    } else {
        tile.y = linear / params.num_tiles_n;
        tile.x = linear % params.num_tiles_n;
    }
    return tile;
}

// MPP Guide: single simdgroup always outperforms multi-simdgroup configurations.
// NUM_GROUPS template parameter is retained only for source-level compatibility
// with instantiations below, but every instantiation passes NUM_GROUPS=1.
template <typename T, int SM, int SN, int NUM_GROUPS = 1>
inline void mpp_gemm_nn_impl(
    device T* A,
    device T* B,
    device T* D,
    constant MppGemmParams& params,
    uint3 tgid
) {
    static_assert(NUM_GROUPS == 1, "MPP Guide: single simdgroup required");

    const int M = (int)params.M;
    const int N = (int)params.N;
    const int K = (int)params.K;

    uint2 tile = decode_output_tile(tgid.x, params);
    const int tile_m = (int)(tile.y * SM);
    const int tile_n = (int)(tile.x * SN);
    if (tile_m >= M || tile_n >= N) return;

    const uint batch_idx = tgid.z;
    device T* A_batch = A + batch_idx * M * K;
    device T* B_batch = B + batch_idx * N * K;
    device T* D_batch = D + batch_idx * M * N;

    auto tA = tensor(A_batch, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    auto tB = tensor(B_batch, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    auto tD = tensor(D_batch, dextents<int, 2>{N, M}, array<int, 2>{1, N});

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        SM,
        SN,
        static_cast<int>(dynamic_extent),
        false,
        true,
        false
    );
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;

    const bool is_full_tile =
        (FC_M_ALIGNED && FC_N_ALIGNED) || (tile_m + SM <= M && tile_n + SN <= N);
    if (is_full_tile) {
        auto sliceA = tA.template slice<dynamic_extent, SM>(0, tile_m);
        auto sliceB = tB.template slice<dynamic_extent, SN>(0, tile_n);
        auto sliceD = tD.template slice<SN, SM>(tile_n, tile_m);
        op.run(sliceA, sliceB, sliceD);
    } else {
        auto sliceA = tA.slice(0, tile_m);
        auto sliceB = tB.slice(0, tile_n);
        auto sliceD = tD.slice(tile_n, tile_m);
        op.run(sliceA, sliceB, sliceD);
    }
}

template <int SM, int SN, int NUM_GROUPS = 1>
inline void mpp_gemm_accumulate_f16_impl(
    device half* A,
    device half* B,
    device half* C,
    device half* D,
    constant MppGemmParams& params,
    uint3 tgid
) {
    static_assert(NUM_GROUPS == 1, "MPP Guide: single simdgroup required");
    const int M = (int)params.M;
    const int N = (int)params.N;
    const int K = (int)params.K;

    uint2 tile = decode_output_tile(tgid.x, params);
    const int tile_m = (int)(tile.y * SM);
    const int tile_n = (int)(tile.x * SN);
    if (tile_m >= M || tile_n >= N) return;

    const uint batch_idx = tgid.z;
    device half* A_batch = A + batch_idx * M * K;
    device half* B_batch = B + batch_idx * N * K;
    device half* D_batch = D + batch_idx * M * N;

    auto tA = tensor(A_batch, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    auto tB = tensor(B_batch, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    auto tD = tensor(D_batch, dextents<int, 2>{N, M}, array<int, 2>{1, N});

    constexpr int BK = 128;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        SM,
        SN,
        BK,
        false,
        true,
        false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;

    const bool is_full_tile =
        (FC_M_ALIGNED && FC_N_ALIGNED) || (tile_m + SM <= M && tile_n + SN <= N);
    if (is_full_tile) {
        auto sliceA = tA.template slice<dynamic_extent, SM>(0, tile_m);
        auto sliceB = tB.template slice<dynamic_extent, SN>(0, tile_n);
        auto sliceD = tD.template slice<SN, SM>(tile_n, tile_m);

        auto rD =
            op.template get_destination_cooperative_tensor<decltype(sliceA), decltype(sliceB), float>();

        if (params.beta != 0.0f) {
            device half* C_batch = C + batch_idx * M * N;
            auto tC = tensor(C_batch, dextents<int, 2>{N, M}, array<int, 2>{1, N});
            auto sliceC = tC.template slice<SN, SM>(tile_n, tile_m);

            auto oC =
                op.template get_destination_cooperative_tensor<decltype(sliceA), decltype(sliceB), half>();
            oC.load(sliceC);

            for (int i = 0; i < rD.get_capacity(); i++) {
                rD[i] = float(oC[i]) * params.beta;
            }
        }

        const int num_k = (K + BK - 1) / BK;
        for (int kk = 0; kk < num_k; kk++) {
            threadgroup_barrier(mem_flags::mem_none);

            auto tkA = tA.template slice<dynamic_extent, SM>(kk * BK, tile_m);
            auto tkB = tB.template slice<dynamic_extent, SN>(kk * BK, tile_n);
            op.run(tkA, tkB, rD);
        }

        if (params.alpha != 1.0f) {
            for (int i = 0; i < rD.get_capacity(); i++) {
                rD[i] = rD[i] * params.alpha;
            }
        }

        auto oD =
            op.template get_destination_cooperative_tensor<decltype(sliceA), decltype(sliceB), half>();
        for (int i = 0; i < rD.get_capacity(); i++) {
            oD[i] = half(rD[i]);
        }
        oD.store(sliceD);
    } else {
        auto sliceA = tA.slice(0, tile_m);
        auto sliceB = tB.slice(0, tile_n);
        auto sliceD = tD.slice(tile_n, tile_m);

        auto rD =
            op.template get_destination_cooperative_tensor<decltype(sliceA), decltype(sliceB), float>();

        if (params.beta != 0.0f) {
            device half* C_batch = C + batch_idx * M * N;
            auto tC = tensor(C_batch, dextents<int, 2>{N, M}, array<int, 2>{1, N});
            auto sliceC = tC.slice(tile_n, tile_m);

            auto oC =
                op.template get_destination_cooperative_tensor<decltype(sliceA), decltype(sliceB), half>();
            oC.load(sliceC);

            for (int i = 0; i < rD.get_capacity(); i++) {
                rD[i] = float(oC[i]) * params.beta;
            }
        }

        const int num_k = (K + BK - 1) / BK;
        for (int kk = 0; kk < num_k; kk++) {
            threadgroup_barrier(mem_flags::mem_none);

            auto tkA = sliceA.slice(kk * BK, 0);
            auto tkB = sliceB.slice(kk * BK, 0);
            op.run(tkA, tkB, rD);
        }

        if (params.alpha != 1.0f) {
            for (int i = 0; i < rD.get_capacity(); i++) {
                rD[i] = rD[i] * params.alpha;
            }
        }

        auto oD =
            op.template get_destination_cooperative_tensor<decltype(sliceA), decltype(sliceB), half>();
        for (int i = 0; i < rD.get_capacity(); i++) {
            oD[i] = half(rD[i]);
        }
        oD.store(sliceD);
    }
}

// All kernels use single simdgroup (execution_simdgroup).
// The sg1/sg2/sg4 name suffixes reflect tile sizes chosen by the Rust
// auto-tuner, not simdgroup counts.

kernel void mpp_gemm_nn_f16_sg1_32x32(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant MppGemmParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_nn_impl<half, 32, 32, 1>(A, B, D, params, tgid);
}

kernel void mpp_gemm_nn_f16_sg2_64x32(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant MppGemmParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_nn_impl<half, 64, 32, 1>(A, B, D, params, tgid);
}

kernel void mpp_gemm_nn_f16_sg2_32x64(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant MppGemmParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_nn_impl<half, 32, 64, 1>(A, B, D, params, tgid);
}

kernel void mpp_gemm_nn_f16_sg4_64x64(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant MppGemmParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_nn_impl<half, 64, 64, 1>(A, B, D, params, tgid);
}

kernel void mpp_gemm_nn_f32_sg1_32x32(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* D [[buffer(2)]],
    constant MppGemmParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_nn_impl<float, 32, 32, 1>(A, B, D, params, tgid);
}

kernel void mpp_gemm_nn_f32_sg2_64x32(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* D [[buffer(2)]],
    constant MppGemmParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_nn_impl<float, 64, 32, 1>(A, B, D, params, tgid);
}

kernel void mpp_gemm_nn_f32_sg2_32x64(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* D [[buffer(2)]],
    constant MppGemmParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_nn_impl<float, 32, 64, 1>(A, B, D, params, tgid);
}

kernel void mpp_gemm_nn_f32_sg4_64x64(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* D [[buffer(2)]],
    constant MppGemmParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_nn_impl<float, 64, 64, 1>(A, B, D, params, tgid);
}

kernel void mpp_gemm_accumulate_f16_sg1_32x32(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device half* D [[buffer(3)]],
    constant MppGemmParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_accumulate_f16_impl<32, 32, 1>(A, B, C, D, params, tgid);
}

kernel void mpp_gemm_accumulate_f16_sg2_64x32(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device half* D [[buffer(3)]],
    constant MppGemmParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_accumulate_f16_impl<64, 32, 1>(A, B, C, D, params, tgid);
}

kernel void mpp_gemm_accumulate_f16_sg2_32x64(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device half* D [[buffer(3)]],
    constant MppGemmParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_accumulate_f16_impl<32, 64, 1>(A, B, C, D, params, tgid);
}

kernel void mpp_gemm_accumulate_f16_sg4_64x64(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device half* D [[buffer(3)]],
    constant MppGemmParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    mpp_gemm_accumulate_f16_impl<64, 64, 1>(A, B, C, D, params, tgid);
}
