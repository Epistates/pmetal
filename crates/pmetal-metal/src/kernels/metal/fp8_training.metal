//  FP8 Training Kernels for Apple Silicon
//
//  Implements block-wise FP8 quantization/dequantization and scaled GEMM
//  operations for memory-efficient training on M-series chips.
//
//  FP8 Format Support (OCP Standard):
//  - E4M3 (1-4-3): Range +-448, best for weights/activations.
//  - E5M2 (1-5-2): Range +-57344, best for gradients.
//
//  This implementation uses 'uchar' (8-bit) storage to maximize memory bandwidth,
//  unpacking to float/bfloat in registers during computation.

#include <metal_stdlib>
using namespace metal;

// Block size for FP8 quantization (must be power of 2)
constant int FP8_BLOCK_SIZE = 128;

// FP8 format max values
constant float FP8_E4M3_MAX = 448.0f;
constant float FP8_E5M2_MAX = 57344.0f;

// ============================================================================
// Helper Functions for FP8 Unpacking
// ============================================================================

// Unpack E5M2 (1-5-2) to half precision
// E5M2 layout matches the top 8 bits of IEEE 754 Binary16 (half)
// S EEEEE MM -> S EEEEE MM00000000
inline float unpack_fp8_e5m2(uchar val) {
    ushort bits = (ushort)val << 8;
    half h = as_type<half>(bits);
    return static_cast<float>(h);
}

// Unpack E4M3 (1-4-3) to float
// S EEEE MMM
// Bias = 7.
// We manually construct the float32 representation.
inline float unpack_fp8_e4m3(uchar val) {
    uint uval = (uint)val;
    uint sign = (uval & 0x80) << 24;
    uint exp_bits = (uval & 0x78) >> 3;
    uint mant_bits = (uval & 0x07);

    // Handle Zero
    if ((val & 0x7F) == 0) return 0.0f;

    // Check for Denormals (Exp = 0)
    // E4M3 Denorm: 0.MMM * 2^-6
    if (exp_bits == 0) {
         // Normalize it: find first set bit, shift, adjust exponent
         // Simplified: Just cast to float and multiply
         float f = (float)mant_bits * 0.125f; // 0.MMM as fraction
         f *= 0.015625f; // * 2^-6
         return (sign != 0) ? -f : f;
    }

    // Normalized: 1.MMM * 2^(exp - 7)
    // Float32 Exp: exp - 7 + 127 = exp + 120
    uint f32_exp = (exp_bits + 120) << 23;
    uint f32_mant = mant_bits << 20; // 3 bits -> top of 23 bits

    uint f32_bits = sign | f32_exp | f32_mant;
    return as_type<float>(f32_bits);
}

// Pack float to E4M3 (1-4-3) format
// Uses round-to-nearest-even for best accuracy
// E4M3 range: +-448, bias = 7
inline uchar pack_fp8_e4m3(float val) {
    // Handle special cases
    if (val == 0.0f) return 0;

    uint f32 = as_type<uint>(val);
    uint sign = (f32 >> 24) & 0x80;  // Extract sign to bit 7

    // Get absolute value for processing
    float abs_val = abs(val);

    // Clamp to E4M3 range (max = 448)
    abs_val = min(abs_val, FP8_E4M3_MAX);

    // Extract f32 components
    f32 = as_type<uint>(abs_val);
    int f32_exp = ((f32 >> 23) & 0xFF) - 127;  // Unbias f32 exponent
    uint f32_mant = f32 & 0x7FFFFF;            // 23-bit mantissa

    // Rebias to E4M3 (bias = 7)
    int e4m3_exp = f32_exp + 7;

    // Handle denormals (exp <= 0 in E4M3)
    if (e4m3_exp <= 0) {
        // Denormalize: shift mantissa right, add implicit 1
        int shift = 1 - e4m3_exp;
        if (shift > 4) return sign;  // Too small, round to zero

        // Add implicit 1 and shift
        uint mant_with_implicit = (1 << 23) | f32_mant;
        uint shifted_mant = mant_with_implicit >> (20 + shift);

        // Round-to-nearest-even
        uint round_bit = (mant_with_implicit >> (19 + shift)) & 1;
        uint sticky_bits = mant_with_implicit & ((1 << (19 + shift)) - 1);
        if (round_bit && (sticky_bits || (shifted_mant & 1))) {
            shifted_mant++;
        }

        return (uchar)(sign | (shifted_mant & 0x07));
    }

    // Handle overflow (exp >= 15 in E4M3, but max is 14 for finite values)
    if (e4m3_exp >= 15) {
        // Return max finite value (not NaN - E4M3 has no inf/nan)
        return (uchar)(sign | 0x7E);  // exp=14, mant=6 -> 448
    }

    // Normal case: extract top 3 bits of mantissa with rounding
    uint e4m3_mant = f32_mant >> 20;  // Top 3 bits

    // Round-to-nearest-even using bit 19 (round) and bits 0-18 (sticky)
    uint round_bit = (f32_mant >> 19) & 1;
    uint sticky_bits = f32_mant & 0x7FFFF;

    if (round_bit && (sticky_bits || (e4m3_mant & 1))) {
        e4m3_mant++;
        // Handle mantissa overflow
        if (e4m3_mant > 7) {
            e4m3_mant = 0;
            e4m3_exp++;
            // Check for overflow after rounding
            if (e4m3_exp >= 15) {
                return (uchar)(sign | 0x7E);  // Max value
            }
        }
    }

    return (uchar)(sign | ((uint)e4m3_exp << 3) | e4m3_mant);
}

// Pack float to E5M2 (1-5-2) format
// E5M2 range: +-57344, bias = 15
// Better dynamic range than E4M3, ideal for gradients
inline uchar pack_fp8_e5m2(float val) {
    // Handle special cases
    if (val == 0.0f) return 0;

    uint f32 = as_type<uint>(val);
    uint sign = (f32 >> 24) & 0x80;

    float abs_val = abs(val);

    // Clamp to E5M2 range
    abs_val = min(abs_val, FP8_E5M2_MAX);

    f32 = as_type<uint>(abs_val);
    int f32_exp = ((f32 >> 23) & 0xFF) - 127;
    uint f32_mant = f32 & 0x7FFFFF;

    // Rebias to E5M2 (bias = 15)
    int e5m2_exp = f32_exp + 15;

    // Handle denormals
    if (e5m2_exp <= 0) {
        int shift = 1 - e5m2_exp;
        if (shift > 3) return sign;

        uint mant_with_implicit = (1 << 23) | f32_mant;
        uint shifted_mant = mant_with_implicit >> (21 + shift);

        uint round_bit = (mant_with_implicit >> (20 + shift)) & 1;
        uint sticky_bits = mant_with_implicit & ((1 << (20 + shift)) - 1);
        if (round_bit && (sticky_bits || (shifted_mant & 1))) {
            shifted_mant++;
        }

        return (uchar)(sign | (shifted_mant & 0x03));
    }

    // Handle overflow (E5M2 uses exp=31 for inf/nan, max finite is exp=30)
    if (e5m2_exp >= 31) {
        return (uchar)(sign | 0x7B);  // exp=30, mant=3 -> max finite
    }

    // Normal case: extract top 2 bits with rounding
    uint e5m2_mant = f32_mant >> 21;

    uint round_bit = (f32_mant >> 20) & 1;
    uint sticky_bits = f32_mant & 0xFFFFF;

    if (round_bit && (sticky_bits || (e5m2_mant & 1))) {
        e5m2_mant++;
        if (e5m2_mant > 3) {
            e5m2_mant = 0;
            e5m2_exp++;
            if (e5m2_exp >= 31) {
                return (uchar)(sign | 0x7B);
            }
        }
    }

    return (uchar)(sign | ((uint)e5m2_exp << 2) | e5m2_mant);
}

// ============================================================================
// Block-wise Activation Quantization (act_quant)
// ============================================================================

// Quantize activations to FP8 E4M3 with per-block scales
// Input: X [M, K] (bf16/f32)
// Output: Y [M, K] (uchar packed FP8), scales [M, K/block_size]
kernel void fp8_act_quant_block(
    device const bfloat* x [[buffer(0)]],
    device uchar* y [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& block_size [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    uint row = tgid.y;
    uint block_idx = tgid.x;

    if (row >= M) return;

    uint block_start = block_idx * block_size;
    if (block_start >= K) return;

    // Shared memory for block max reduction
    threadgroup float shared_max[256];

    // Each thread processes elements in the block
    float local_max = 0.0f;
    uint threads_per_block = min(block_size, 256u);

    for (uint i = lid; i < block_size && (block_start + i) < K; i += threads_per_block) {
        uint idx = row * K + block_start + i;
        float val = abs(static_cast<float>(x[idx]));
        local_max = max(local_max, val);
    }

    // Store local max
    shared_max[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for block max
    for (uint stride = threads_per_block / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute scale from block max
    float amax = shared_max[0];
    float scale = (amax > 1e-12f) ? (amax / FP8_E4M3_MAX) : 1.0f;
    float scale_inv = (amax > 1e-12f) ? (FP8_E4M3_MAX / amax) : 1.0f;

    // First thread writes scale
    if (lid == 0) {
        // Use ceiling division so the scale index is correct when K is not
        // divisible by block_size (fixes MED-M1 off-by-one on the last block).
        uint scale_idx = row * ((K + block_size - 1) / block_size) + block_idx;
        scales[scale_idx] = scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Quantize using proper FP8 E4M3 packing with round-to-nearest-even
    for (uint i = lid; i < block_size && (block_start + i) < K; i += threads_per_block) {
        uint idx = row * K + block_start + i;
        float val = static_cast<float>(x[idx]) * scale_inv;

        // Pack to E4M3 format with proper IEEE-style rounding
        y[idx] = pack_fp8_e4m3(val);
    }
}

// ============================================================================
// Gradient Quantization (E5M2 for wider dynamic range)
// ============================================================================

// Quantize gradients to FP8 E5M2 with per-block scales
// E5M2 has wider dynamic range than E4M3, better suited for gradients
// Input: X [M, K] (bf16/f32 gradients)
// Output: Y [M, K] (uchar packed FP8 E5M2), scales [M, K/block_size]
kernel void fp8_grad_quant_block(
    device const bfloat* x [[buffer(0)]],
    device uchar* y [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& block_size [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    uint row = tgid.y;
    uint block_idx = tgid.x;

    if (row >= M) return;

    uint block_start = block_idx * block_size;
    if (block_start >= K) return;

    // Shared memory for block max reduction
    threadgroup float shared_max[256];

    // Each thread processes elements in the block
    float local_max = 0.0f;
    uint threads_per_block = min(block_size, 256u);

    for (uint i = lid; i < block_size && (block_start + i) < K; i += threads_per_block) {
        uint idx = row * K + block_start + i;
        float val = abs(static_cast<float>(x[idx]));
        local_max = max(local_max, val);
    }

    // Store local max
    shared_max[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for block max
    for (uint stride = threads_per_block / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute scale from block max using E5M2 range
    float amax = shared_max[0];
    float scale = (amax > 1e-12f) ? (amax / FP8_E5M2_MAX) : 1.0f;
    float scale_inv = (amax > 1e-12f) ? (FP8_E5M2_MAX / amax) : 1.0f;

    // First thread writes scale
    if (lid == 0) {
        // Use ceiling division so the scale index is correct when K is not
        // divisible by block_size (fixes MED-M1 off-by-one on the last block).
        uint scale_idx = row * ((K + block_size - 1) / block_size) + block_idx;
        scales[scale_idx] = scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Quantize using E5M2 format (better for gradients due to wider range)
    for (uint i = lid; i < block_size && (block_start + i) < K; i += threads_per_block) {
        uint idx = row * K + block_start + i;
        float val = static_cast<float>(x[idx]) * scale_inv;

        // Pack to E5M2 format with proper IEEE-style rounding
        y[idx] = pack_fp8_e5m2(val);
    }
}

// ============================================================================
// Weight Dequantization (weight_dequant)
// ============================================================================

// Dequantize FP8 weights back to BF16
// Input: X [M, N] (uchar packed FP8), scales [M/bs, N/bs]
// Output: Y [M, N] (bf16)
kernel void fp8_weight_dequant_block(
    device const uchar* x [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device bfloat* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& block_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    // Find which scale block this element belongs to
    uint scale_row = row / block_size;
    uint scale_col = col / block_size;
    uint n_scale_cols = (N + block_size - 1) / block_size;
    uint scale_idx = scale_row * n_scale_cols + scale_col;

    float scale = scales[scale_idx];
    uint idx = row * N + col;

    // Dequantize: y = unpack(x) * scale
    // Assume E4M3 for weights
    float val = unpack_fp8_e4m3(x[idx]) * scale;
    y[idx] = static_cast<bfloat>(val);
}

// ============================================================================
// Block FP8 GEMM (W8A8 with block scaling)
// ============================================================================

// Tile sizes for GEMM
constant int BLOCK_M = 64;
constant int BLOCK_N = 64;
constant int BLOCK_K = 64;

// Block-wise FP8 GEMM: C = A @ B with block-wise scales
// A: [M, K] (uchar), A_scales: [M, K/group_k]
// B: [N, K] (uchar), B_scales: [N/group_n, K/group_k]
// C: [M, N]
kernel void fp8_block_gemm(
    device const uchar* A [[buffer(0)]],        // Quantized activations
    device const uchar* B [[buffer(1)]],        // Quantized weights (N, K)
    device bfloat* C [[buffer(2)]],             // Output
    device const float* A_scales [[buffer(3)]], // [M, K/group_k]
    device const float* B_scales [[buffer(4)]], // [N/group_n, K/group_k]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& group_n [[buffer(8)]],
    constant uint& group_k [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    // Thread block computes BLOCK_M x BLOCK_N tile of C
    uint tile_m = tgid.y * BLOCK_M;
    uint tile_n = tgid.x * BLOCK_N;

    // Shared memory for tiles (stored as floats for compute).
    // Flattened to 1D: MSL does not support 2D threadgroup arrays (MED-M2 fix).
    // Access: A_tile[row * BLOCK_K + col], B_tile[row * BLOCK_N + col]
    threadgroup float A_tile[BLOCK_M * BLOCK_K];
    threadgroup float B_tile[BLOCK_K * BLOCK_N];

    // Accumulator for this thread
    float acc[4][4] = {{0.0f}};

    // Process K dimension in blocks
    uint n_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;

    for (uint kb = 0; kb < n_k_blocks; kb++) {
        uint k_start = kb * BLOCK_K;

        // Load A tile with scaling
        // Optimization: Use vector loads (uchar16 -> float16) if possible
        uint a_threads = BLOCK_M * BLOCK_K / 256; 
        for (uint t = lid; t < BLOCK_M * BLOCK_K; t += 256) {
            uint local_m = t / BLOCK_K;
            uint local_k = t % BLOCK_K;
            uint global_m = tile_m + local_m;
            uint global_k = k_start + local_k;

            if (global_m < M && global_k < K) {
                // Unpack Activation (E4M3 or E5M2?)
                // Usually Activations are E5M2, Weights are E4M3. Or both E4M3.
                // We'll assume E4M3 for this general kernel.
                float val = unpack_fp8_e4m3(A[global_m * K + global_k]);

                // Apply activation scale
                uint scale_k = global_k / group_k;
                float a_scale = A_scales[global_m * (K / group_k) + scale_k];
                A_tile[local_m * BLOCK_K + local_k] = val * a_scale;
            } else {
                A_tile[local_m * BLOCK_K + local_k] = 0.0f;
            }
        }

        // Load B tile with scaling (B is [N, K])
        for (uint t = lid; t < BLOCK_K * BLOCK_N; t += 256) {
            uint local_k = t / BLOCK_N;
            uint local_n = t % BLOCK_N;
            uint global_k = k_start + local_k;
            uint global_n = tile_n + local_n;

            if (global_n < N && global_k < K) {
                // Unpack Weight (E4M3)
                float val = unpack_fp8_e4m3(B[global_n * K + global_k]);

                // Apply weight scale
                uint scale_n = global_n / group_n;
                uint scale_k = global_k / group_k;
                float b_scale = B_scales[scale_n * (K / group_k) + scale_k];
                B_tile[local_k * BLOCK_N + local_n] = val * b_scale;
            } else {
                B_tile[local_k * BLOCK_N + local_n] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial products
        uint thread_m = (lid / 16) * 4;
        uint thread_n = (lid % 16) * 4;

        for (uint k = 0; k < BLOCK_K; k++) {
            float a_vals[4];
            float b_vals[4];

            for (uint i = 0; i < 4; i++) {
                a_vals[i] = A_tile[(thread_m + i) * BLOCK_K + k];
                b_vals[i] = B_tile[k * BLOCK_N + (thread_n + i)];
            }

            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    acc[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results to C
    uint thread_m_global = tile_m + (lid / 16) * 4;
    uint thread_n_global = tile_n + (lid % 16) * 4;

    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            uint global_m = thread_m_global + i;
            uint global_n = thread_n_global + j;
            if (global_m < M && global_n < N) {
                C[global_m * N + global_n] = static_cast<bfloat>(acc[i][j]);
            }
        }
    }
}