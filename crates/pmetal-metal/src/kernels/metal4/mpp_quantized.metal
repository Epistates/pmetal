// mpp_quantized.metal
// Metal 4 NAX-accelerated quantized inference.
//
// M5 NAX cores natively support FP4/FP8 quantized GEMM via matmul2d.
// This provides 2-4x compute density vs FP16 for quantized model inference.
//
// References:
// - MLX quantized_nax.metal
// - MLX fp_quantized_nax.metal

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

struct QuantGemmParams {
    uint M;           // Batch × seq_len (tokens)
    uint N;           // Output dimension
    uint K;           // Input dimension
    uint group_size;  // Quantization group size
    uint bits;        // Quantization bits (2, 4, 8)
    uint num_tiles_m;
    uint num_tiles_n;
};

// =============================================================================
// NAX Quantized MatVec (4-bit weights, fp16 activations)
// =============================================================================
//
// For decode (M=1), this is a matrix-vector multiply:
//   y = x @ W_dequant^T
//
// where W is stored as packed 4-bit with per-group scale+bias:
//   W_dequant[i] = scale * W_q[i] + bias
//
// For prefill (M>1), this becomes a full GEMM with on-the-fly dequantization.
//
// The MPP approach: dequantize W tiles into threadgroup fp16, then use
// matmul2d for the hardware GEMM. This amortizes dequant cost over the
// tile reuse in the M dimension.

kernel void mpp_qmm_4bit_f16(
    device half* x [[buffer(0)]],                    // [M, K] activations
    device const uint32_t* w_packed [[buffer(1)]],   // [N, K/8] packed 4-bit weights
    device const half* scales [[buffer(2)]],          // [N, K/group_size] per-group scales
    device const half* biases [[buffer(3)]],          // [N, K/group_size] per-group biases
    device half* y [[buffer(4)]],                     // [M, N] output
    constant QuantGemmParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const int M = (int)params.M;
    const int N = (int)params.N;
    const int K = (int)params.K;
    const uint gs = params.group_size;

    const int BM = 64;
    const int BN = 64;
    const int BK = 32;  // K-tile for dequant + matmul

    const int tile_m = (int)(tgid.y * BM);
    const int tile_n = (int)(tgid.x * BN);
    if (tile_m >= M || tile_n >= N) return;

    // Threadgroup memory for dequantized weight tile
    threadgroup half W_dequant[BN * BK];  // 64 × 32 × 2 = 4KB

    auto tX = tensor(x, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    auto tY = tensor(y, dextents<int, 2>{N, M}, array<int, 2>{1, N});

    // Accumulate output in cooperative tensor
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        64, 64, BK,
        false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;

    auto sliceX = tX.slice(0, tile_m);
    auto sliceY = tY.slice(tile_n, tile_m);

    auto rD = op.get_destination_cooperative_tensor<decltype(sliceX),
        decltype(tensor((threadgroup half*)W_dequant,
                        dextents<int, 2>{BK, BN},
                        array<int, 2>{1, BK})),
        float>();

    uint total_threads = 128;
    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint packed_cols = K / 8;  // 8 values per uint32

    // K-loop: dequantize BK columns of W at a time, then matmul
    for (int k_start = 0; k_start < K; k_start += BK) {
        int k_end = min(k_start + BK, K);
        int k_len = k_end - k_start;

        // Cooperatively dequantize W[tile_n..+BN, k_start..+BK] into W_dequant
        for (uint idx = linear_tid; idx < (uint)BN * (uint)BK; idx += total_threads) {
            uint n_local = idx / (uint)BK;
            uint k_local = idx % (uint)BK;
            uint global_n = (uint)tile_n + n_local;
            uint global_k = (uint)k_start + k_local;

            if (global_n < params.N && global_k < params.K) {
                // Unpack 4-bit value from packed word
                uint word_idx = global_k / 8;
                uint nibble_idx = global_k % 8;
                uint32_t packed = w_packed[global_n * packed_cols + word_idx];
                uint nibble = (packed >> (nibble_idx * 4)) & 0xFu;

                // Dequantize: val = scale * nibble + bias
                uint group_idx = global_k / gs;
                float scale = float(scales[global_n * (K / gs) + group_idx]);
                float bias = float(biases[global_n * (K / gs) + group_idx]);
                W_dequant[n_local * BK + k_local] = half(scale * float(nibble) + bias);
            } else {
                W_dequant[n_local * BK + k_local] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MPP matmul: tile_output += X_slice @ W_dequant^T
        auto tW = tensor((threadgroup half*)W_dequant,
                         dextents<int, 2>{BK, BN},
                         array<int, 2>{1, BK});

        auto tkX = sliceX.slice(k_start, 0);

        op.run(tkX, tW, rD);

        threadgroup_barrier(mem_flags::mem_none);
    }

    // Store accumulated result
    auto oD = op.get_destination_cooperative_tensor<decltype(sliceX),
        decltype(tensor((threadgroup half*)W_dequant,
                        dextents<int, 2>{BK, BN},
                        array<int, 2>{1, BK})),
        half>();
    for (int i = 0; i < rD.get_capacity(); i++) {
        oD[i] = half(rD[i]);
    }
    oD.store(sliceY);
}

// =============================================================================
// NAX Quantized MatMul (8-bit weights, fp16 activations)
// =============================================================================
//
// On-the-fly dequantization: int8 weights are dequantized into threadgroup
// fp16 tiles with per-group scale applied during the load, then matmul2d
// computes the GEMM on the dequantized tile.
//
// This follows the same pattern as the 4-bit kernel above:
//   For each K-chunk:
//     1. Cooperatively dequantize W[tile_n..+BN, k..+BK] → fp16 with scale
//     2. matmul2d: rD += X_slice @ W_dequant^T
//   Store accumulated result

kernel void mpp_qmm_8bit_f16(
    device half* x [[buffer(0)]],                    // [M, K] activations
    device const int8_t* w [[buffer(1)]],            // [N, K] int8 weights
    device const half* scales [[buffer(2)]],          // [N, K/group_size] per-group scales
    device half* y [[buffer(3)]],                     // [M, N] output
    constant QuantGemmParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const int M = (int)params.M;
    const int N = (int)params.N;
    const int K = (int)params.K;
    const uint gs = params.group_size;

    const int BM = 64;
    const int BN = 64;
    const int BK = 64;  // Larger than 4-bit since int8 dequant is cheaper

    const int tile_m = (int)(tgid.y * BM);
    const int tile_n = (int)(tgid.x * BN);
    if (tile_m >= M || tile_n >= N) return;

    // Threadgroup memory for dequantized + scaled weight tile
    threadgroup half W_dequant[BN * BK];  // 64 × 64 × 2 = 8KB

    auto tX = tensor(x, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    auto tY = tensor(y, dextents<int, 2>{N, M}, array<int, 2>{1, N});

    // Accumulate output in cooperative tensor
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        64, 64, BK,
        false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;

    auto sliceX = tX.slice(0, tile_m);
    auto sliceY = tY.slice(tile_n, tile_m);

    auto rD = op.get_destination_cooperative_tensor<decltype(sliceX),
        decltype(tensor((threadgroup half*)W_dequant,
                        dextents<int, 2>{BK, BN},
                        array<int, 2>{1, BK})),
        float>();

    uint total_threads = 128;
    uint linear_tid = simd_group_id * 32 + simd_lane_id;
    uint num_groups_k = K / gs;

    // K-loop: dequantize BK columns of W at a time with scale, then matmul
    for (int k_start = 0; k_start < K; k_start += BK) {
        // Cooperatively dequantize W[tile_n..+BN, k_start..+BK] with scale
        for (uint idx = linear_tid; idx < (uint)BN * (uint)BK; idx += total_threads) {
            uint n_local = idx / (uint)BK;
            uint k_local = idx % (uint)BK;
            uint global_n = (uint)tile_n + n_local;
            uint global_k = (uint)k_start + k_local;

            if (global_n < params.N && global_k < params.K) {
                // Dequantize: val = scale * int8_val
                int8_t raw = w[global_n * (uint)K + global_k];
                uint group_idx = global_k / gs;
                float scale = float(scales[global_n * num_groups_k + group_idx]);
                W_dequant[n_local * BK + k_local] = half(scale * float(raw));
            } else {
                W_dequant[n_local * BK + k_local] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MPP matmul: rD += X_slice @ W_dequant^T
        auto tW = tensor((threadgroup half*)W_dequant,
                         dextents<int, 2>{BK, BN},
                         array<int, 2>{1, BK});

        auto tkX = sliceX.slice(k_start, 0);

        op.run(tkX, tW, rD);

        threadgroup_barrier(mem_flags::mem_none);
    }

    // Store accumulated result
    auto oD = op.get_destination_cooperative_tensor<decltype(sliceX),
        decltype(tensor((threadgroup half*)W_dequant,
                        dextents<int, 2>{BK, BN},
                        array<int, 2>{1, BK})),
        half>();
    for (int i = 0; i < rD.get_capacity(); i++) {
        oD[i] = half(rD[i]);
    }
    oD.store(sliceY);
}
