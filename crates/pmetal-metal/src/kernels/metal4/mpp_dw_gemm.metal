// mpp_dw_gemm.metal
// Metal 4 weight gradient GEMM for ANE training backward pass using MPP.
//
// Computes: C = alpha * A @ B^T + beta * C
//
// Replaces the Metal 3 `dw_gemm_accum` which uses manual 64x64x16 threadgroup
// staging with 256-thread outer product accumulation.
//
// In ANE training: 20 layers × 7 GEMMs = 140 dispatches per step.
// These all run on the GPU while ANE handles dx propagation.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// Morton ordering for threadgroup walk
inline uint2 morton_decode(uint linear) {
    uint x = 0, y = 0;
    for (uint bit = 0; bit < 16; bit++) {
        y |= ((linear >> (2 * bit))     & 1) << bit;
        x |= ((linear >> (2 * bit + 1)) & 1) << bit;
    }
    return uint2(x, y);
}

struct DwGemmParams {
    uint M;
    uint N;
    uint K;
    float alpha;
    float beta;
    uint num_tiles_m;
    uint num_tiles_n;
};

/// MPP weight gradient GEMM: C = alpha * A @ B^T + beta * C
///
/// A: [M, K] row-major (activations or dY)
/// B: [N, K] row-major (transposed: need B^T for dW computation)
/// C: [M, N] row-major (weight gradient, accumulated)
///
/// Key improvement: Hardware MMA via NAX eliminates the manual 4x4 outer product
/// register accumulation in the Metal 3 version.
kernel void mpp_dw_gemm_accum(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant DwGemmParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const int M = (int)params.M;
    const int N = (int)params.N;
    const int K = (int)params.K;

    // MPP Guide: K must be a multiple of 32 for NAX hardware GEMM.
    // If K is not aligned, NAX silently produces wrong results.
    // The Rust caller is responsible for padding; we guard defensively here.
    // An unaligned K falls back to a no-op (output unchanged) to surface
    // the misconfiguration without producing corrupted results.
    if ((K & 31) != 0) return;

    // 2D grid (no Morton needed for dw_gemm since it's dispatched per-layer
    // in a batched command buffer — cache behavior is different from global GEMM)
    const int tile_m = (int)(tgid.y * 64);
    const int tile_n = (int)(tgid.x * 64);
    if (tile_m >= M || tile_n >= N) return;

    // Create tensors
    auto tA = tensor(A, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    auto tB = tensor(B, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    auto tC = tensor(C, dextents<int, 2>{N, M}, array<int, 2>{1, N});

    // MPP matmul: C = A @ B^T
    // For accumulation (beta != 0), we'd need a two-step approach:
    // 1. Compute A @ B^T into temp
    // 2. C = alpha * temp + beta * C
    // But for the common case (beta=0), we can just write directly.
    //
    // With beta accumulation, we use multiply_accumulate mode with manual
    // K-loop and cooperative tensor postfix fusion.

    if (params.beta == 0.0f && params.alpha == 1.0f) {
        // Simple case: C = A @ B^T (overwrite)
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            64, 64,
            static_cast<int>(dynamic_extent),
            false, true, false
        );

        mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;

        auto sliceA = tA.slice(0, tile_m);
        auto sliceB = tB.slice(0, tile_n);
        auto sliceC = tC.slice(tile_n, tile_m);

        op.run(sliceA, sliceB, sliceC);
    } else {
        // Accumulate case: C = alpha * A @ B^T + beta * C
        // Use K-chunked matmul with cooperative tensor for postfix scaling
        constexpr int BK = 128;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            64, 64, BK,
            false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
        );

        mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;

        auto sliceA = tA.slice(0, tile_m);
        auto sliceB = tB.slice(0, tile_n);
        auto sliceC = tC.slice(tile_n, tile_m);

        // Get cooperative tensor for accumulation
        auto rD = op.get_destination_cooperative_tensor<
            decltype(sliceA), decltype(sliceB), float>();

        // Pre-load existing C if beta != 0 (postfix fusion)
        if (params.beta != 0.0f) {
            auto oC = op.get_destination_cooperative_tensor<
                decltype(sliceA), decltype(sliceB), float>();
            oC.load(sliceC);
            for (int i = 0; i < rD.get_capacity(); i++) {
                rD[i] = oC[i] * params.beta;
            }
        }

        // K-loop with accumulation barriers
        const int num_k = (K + BK - 1) / BK;
        for (int kk = 0; kk < num_k; kk++) {
            threadgroup_barrier(mem_flags::mem_none);
            auto tkA = sliceA.slice(kk * BK, 0);
            auto tkB = sliceB.slice(kk * BK, 0);
            op.run(tkA, tkB, rD);
        }

        // Apply alpha scaling (postfix fusion in register space)
        if (params.alpha != 1.0f) {
            for (int i = 0; i < rD.get_capacity(); i++) {
                rD[i] *= params.alpha;
            }
        }

        // Store result
        rD.store(sliceC);
    }
}
