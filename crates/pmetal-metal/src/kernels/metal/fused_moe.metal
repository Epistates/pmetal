/// Fused MoE (Mixture of Experts) Metal compute kernels.
///
/// Provides optimized dequant kernels for MoE inference using the qdot technique
/// (pre-scaled activations, eliminating shifts from inner loop):
///
/// 1. `fused_gate_up_swiglu` — Fused gate+up+SwiGLU for a single quantized expert
/// 2. `dequant_matvec_4bit` — Optimized 4-bit dequant matrix-vector multiply
/// 3. `dequant_matvec_2bit` — 2-bit variant for further compressed experts
/// 4. `gather_qmm_swiglu` — Fused gather + quantized matmul + SwiGLU for resident mode
/// 5. `gather_dequant_matvec` — Down projection for resident mode
///
/// Key techniques:
///   - qdot: pre-scale activations to eliminate per-nibble shifts
///   - Thread-local x caching: load x once per column, reuse across all RESULTS_PER_SG rows
///   - Factored bias: result = scale * qdot_accum + bias * x_sum  (single outer FMA)
///   - Register-only: no threadgroup shared memory (no overflow, no barriers)
///
/// Thread model: 2 simdgroups × 32 threads = 64 threads per threadgroup
/// Each simdgroup computes RESULTS_PER_SG output rows

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BFloat16 helpers
// ============================================================================

inline float bf16_to_f32(uint16_t bf16) {
    return as_type<float>(uint(bf16) << 16);
}

// ============================================================================
// Threadgroup geometry
// ============================================================================

#define RESULTS_PER_SG 4
#define NUM_SIMDGROUPS  2
#define ROWS_PER_TG    (RESULTS_PER_SG * NUM_SIMDGROUPS)  // 8
#define TG_SIZE        (NUM_SIMDGROUPS * 32)                // 64

// ============================================================================
// Function constants for compile-time specialization (gather kernels)
// ============================================================================

constant uint FC_GROUP_SIZE     [[function_constant(0)]];
constant uint FC_BITS           [[function_constant(1)]]; // 2 or 4

// ============================================================================
// 4-bit qdot: pre-scaled dot product from thread-local x registers
// ============================================================================
//
// Takes x already loaded into thread-local registers (not device memory).
// Computes sum_i[ q_i * x_i ] via mask-only extraction with pre-scaled x.
// Returns (qdot_accum, x_sum).

inline float2 qdot4_local(thread const float* xl, uint32_t packed) {
    uint16_t lo = uint16_t(packed);
    uint16_t hi = uint16_t(packed >> 16);

    float accum = float(lo & 0x000fu) * xl[0]
                + float(lo & 0x00f0u) * xl[1]
                + float(lo & 0x0f00u) * xl[2]
                + float(lo & 0xf000u) * xl[3]
                + float(hi & 0x000fu) * xl[4]
                + float(hi & 0x00f0u) * xl[5]
                + float(hi & 0x0f00u) * xl[6]
                + float(hi & 0xf000u) * xl[7];

    return float2(accum, xl[8]); // xl[8] = pre-computed x_sum
}

// ============================================================================
// 2-bit qdot: pre-scaled dot product from thread-local x registers
// ============================================================================

inline float2 qdot2_local(thread const float* xl, uint32_t packed) {
    uint16_t lo = uint16_t(packed);
    uint16_t hi = uint16_t(packed >> 16);

    float accum = float(lo & 0x0003u) * xl[0]
                + float(lo & 0x000cu) * xl[1]
                + float(lo & 0x0030u) * xl[2]
                + float(lo & 0x00c0u) * xl[3]
                + float(lo & 0x0300u) * xl[4]
                + float(lo & 0x0c00u) * xl[5]
                + float(lo & 0x3000u) * xl[6]
                + float(lo & 0xc000u) * xl[7]
                + float(hi & 0x0003u) * xl[8]
                + float(hi & 0x000cu) * xl[9]
                + float(hi & 0x0030u) * xl[10]
                + float(hi & 0x00c0u) * xl[11]
                + float(hi & 0x0300u) * xl[12]
                + float(hi & 0x0c00u) * xl[13]
                + float(hi & 0x3000u) * xl[14]
                + float(hi & 0xc000u) * xl[15];

    return float2(accum, xl[16]); // xl[16] = pre-computed x_sum
}

// ============================================================================
// x-vector loaders: device → thread-local with pre-scaling
// ============================================================================
//
// Load x from device memory ONCE per column iteration, pre-scale for qdot,
// and compute x_sum. The results are stored in thread-local registers and
// reused across all RESULTS_PER_SG rows — eliminating redundant device reads.

// 4-bit: loads 8 floats, pre-scales, stores 9 values (8 pre-scaled + x_sum)
inline void load_x4(device const float* x_ptr, thread float* xl) {
    float x0 = x_ptr[0], x1 = x_ptr[1], x2 = x_ptr[2], x3 = x_ptr[3];
    float x4 = x_ptr[4], x5 = x_ptr[5], x6 = x_ptr[6], x7 = x_ptr[7];

    xl[0] = x0;                       // nibble 0: no pre-scale
    xl[1] = x1 * (1.0f / 16);         // nibble 1: /16
    xl[2] = x2 * (1.0f / 256);        // nibble 2: /256
    xl[3] = x3 * (1.0f / 4096);       // nibble 3: /4096
    xl[4] = x4;                        // nibble 4: no pre-scale (hi word)
    xl[5] = x5 * (1.0f / 16);         // nibble 5: /16
    xl[6] = x6 * (1.0f / 256);        // nibble 6: /256
    xl[7] = x7 * (1.0f / 4096);       // nibble 7: /4096
    xl[8] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; // x_sum for bias factoring
}

// 2-bit: loads 16 floats, pre-scales, stores 17 values (16 pre-scaled + x_sum)
inline void load_x2(device const float* x_ptr, thread float* xl) {
    float x0  = x_ptr[0],  x1  = x_ptr[1],  x2  = x_ptr[2],  x3  = x_ptr[3];
    float x4  = x_ptr[4],  x5  = x_ptr[5],  x6  = x_ptr[6],  x7  = x_ptr[7];
    float x8  = x_ptr[8],  x9  = x_ptr[9],  x10 = x_ptr[10], x11 = x_ptr[11];
    float x12 = x_ptr[12], x13 = x_ptr[13], x14 = x_ptr[14], x15 = x_ptr[15];

    // Pre-scale for 2-bit: shift factors 1, 4, 16, 64, 256, 1024, 4096, 16384
    xl[0]  = x0;                         xl[1]  = x1  * (1.0f / 4);
    xl[2]  = x2  * (1.0f / 16);          xl[3]  = x3  * (1.0f / 64);
    xl[4]  = x4  * (1.0f / 256);         xl[5]  = x5  * (1.0f / 1024);
    xl[6]  = x6  * (1.0f / 4096);        xl[7]  = x7  * (1.0f / 16384);
    xl[8]  = x8;                          xl[9]  = x9  * (1.0f / 4);
    xl[10] = x10 * (1.0f / 16);          xl[11] = x11 * (1.0f / 64);
    xl[12] = x12 * (1.0f / 256);         xl[13] = x13 * (1.0f / 1024);
    xl[14] = x14 * (1.0f / 4096);        xl[15] = x15 * (1.0f / 16384);

    // Balanced tree reduction for x_sum (better than sequential chain)
    float s0 = (x0 + x1) + (x2 + x3);
    float s1 = (x4 + x5) + (x6 + x7);
    float s2 = (x8 + x9) + (x10 + x11);
    float s3 = (x12 + x13) + (x14 + x15);
    xl[16] = (s0 + s1) + (s2 + s3);
}


// ============================================================================
// Kernel 1: Fused Gate+Up+SwiGLU (single expert, quantized 4-bit)
// ============================================================================
//
// x loaded once per column into thread-local registers, reused across
// RESULTS_PER_SG gate rows + RESULTS_PER_SG up rows = 8 qdot calls per col.

kernel void fused_gate_up_swiglu(
    device const uint32_t* gate_W      [[buffer(0)]],
    device const uint16_t* gate_scales [[buffer(1)]],
    device const uint16_t* gate_biases [[buffer(2)]],
    device const uint32_t* up_W        [[buffer(3)]],
    device const uint16_t* up_scales   [[buffer(4)]],
    device const uint16_t* up_biases   [[buffer(5)]],
    device const float*    x           [[buffer(6)]],
    device float*          out         [[buffer(7)]],
    constant uint&         out_dim     [[buffer(8)]],
    constant uint&         in_dim      [[buffer(9)]],
    constant uint&         group_size  [[buffer(10)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = tgid * ROWS_PER_TG + simd_group * RESULTS_PER_SG;
    if (row_base >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups = in_dim / group_size;
    uint pf_gs = group_size / 8;

    float ga[RESULTS_PER_SG] = {0, 0, 0, 0};
    float ua[RESULTS_PER_SG] = {0, 0, 0, 0};

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // Load x ONCE into thread-local registers — reused across all 8 qdot calls
        thread float xl[9]; // 8 pre-scaled + x_sum
        load_x4(x + col * 8, xl);

        uint g = col / pf_gs;

        for (uint r = 0; r < RESULTS_PER_SG; r++) {
            uint row = row_base + r;
            if (row >= out_dim) break;

            uint w_idx = row * packed_cols + col;
            uint sb_idx = row * num_groups + g;

            float2 gqd = qdot4_local(xl, gate_W[w_idx]);
            ga[r] += bf16_to_f32(gate_scales[sb_idx]) * gqd.x
                   + bf16_to_f32(gate_biases[sb_idx]) * gqd.y;

            float2 uqd = qdot4_local(xl, up_W[w_idx]);
            ua[r] += bf16_to_f32(up_scales[sb_idx]) * uqd.x
                   + bf16_to_f32(up_biases[sb_idx]) * uqd.y;
        }
    }

    for (uint r = 0; r < RESULTS_PER_SG; r++) {
        uint row = row_base + r;
        if (row >= out_dim) break;
        float rg = simd_sum(ga[r]);
        float ru = simd_sum(ua[r]);
        if (simd_lane == 0) {
            out[row] = (rg / (1.0f + exp(-rg))) * ru;
        }
    }
}


// ============================================================================
// Kernel 2: Optimized 4-bit dequant matrix-vector multiply (qdot)
// ============================================================================

kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = tgid * ROWS_PER_TG + simd_group * RESULTS_PER_SG;
    if (row_base >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups = in_dim / group_size;
    uint pf_gs = group_size / 8;

    float acc[RESULTS_PER_SG] = {0, 0, 0, 0};

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        thread float xl[9];
        load_x4(x + col * 8, xl);

        uint g = col / pf_gs;

        for (uint r = 0; r < RESULTS_PER_SG; r++) {
            uint row = row_base + r;
            if (row >= out_dim) break;

            float2 qd = qdot4_local(xl, W_packed[row * packed_cols + col]);
            acc[r] += bf16_to_f32(scales[row * num_groups + g]) * qd.x
                    + bf16_to_f32(biases[row * num_groups + g]) * qd.y;
        }
    }

    for (uint r = 0; r < RESULTS_PER_SG; r++) {
        uint row = row_base + r;
        if (row >= out_dim) break;
        float sum = simd_sum(acc[r]);
        if (simd_lane == 0) {
            out[row] = sum;
        }
    }
}


// ============================================================================
// Kernel 3: 2-bit dequant matrix-vector multiply (qdot)
// ============================================================================

kernel void dequant_matvec_2bit(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_base = tgid * ROWS_PER_TG + simd_group * RESULTS_PER_SG;
    if (row_base >= out_dim) return;

    uint packed_cols = in_dim / 16;
    uint num_groups = in_dim / group_size;
    uint pf_gs = group_size / 16;

    float acc[RESULTS_PER_SG] = {0, 0, 0, 0};

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        thread float xl[17]; // 16 pre-scaled + x_sum
        load_x2(x + col * 16, xl);

        uint g = col / pf_gs;

        for (uint r = 0; r < RESULTS_PER_SG; r++) {
            uint row = row_base + r;
            if (row >= out_dim) break;

            float2 qd = qdot2_local(xl, W_packed[row * packed_cols + col]);
            acc[r] += bf16_to_f32(scales[row * num_groups + g]) * qd.x
                    + bf16_to_f32(biases[row * num_groups + g]) * qd.y;
        }
    }

    for (uint r = 0; r < RESULTS_PER_SG; r++) {
        uint row = row_base + r;
        if (row >= out_dim) break;
        float sum = simd_sum(acc[r]);
        if (simd_lane == 0) {
            out[row] = sum;
        }
    }
}


// ============================================================================
// Kernel 4: Gather + Quantized MatMul + SwiGLU (resident mode, qdot)
// ============================================================================

struct GatherQmmSwigluParams {
    uint hidden_dim;
    uint intermediate_dim;
    uint num_tokens;
    uint topk;
};

kernel void gather_qmm_swiglu(
    device const uint32_t* gate_weights  [[buffer(0)]],
    device const uint16_t* gate_scales   [[buffer(1)]],
    device const uint16_t* gate_biases   [[buffer(2)]],
    device const uint32_t* up_weights    [[buffer(3)]],
    device const uint16_t* up_scales     [[buffer(4)]],
    device const uint16_t* up_biases     [[buffer(5)]],
    device const float*    input         [[buffer(6)]],
    device const uint*     expert_ids    [[buffer(7)]],
    device float*          output        [[buffer(8)]],
    constant GatherQmmSwigluParams& params [[buffer(9)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint D = params.hidden_dim;
    uint I = params.intermediate_dim;
    uint K = params.topk;

    uint tiles_per_token_expert = (I + ROWS_PER_TG - 1) / ROWS_PER_TG;
    uint token_expert_idx = tgid / tiles_per_token_expert;
    uint tile_idx = tgid % tiles_per_token_expert;

    uint n = token_expert_idx / K;
    uint k = token_expert_idx % K;
    uint row_base = tile_idx * ROWS_PER_TG + simd_group * RESULTS_PER_SG;

    if (n >= params.num_tokens || row_base >= I) return;

    uint expert_id = expert_ids[n * K + k];

    uint pack_factor = (FC_BITS == 2) ? 16 : 8;
    uint packed_cols = D / pack_factor;
    uint num_groups = D / FC_GROUP_SIZE;
    uint pf_gs = FC_GROUP_SIZE / pack_factor;

    uint expert_weight_offset = expert_id * I * packed_cols;
    uint expert_scale_offset = expert_id * I * num_groups;

    device const float* x_in = input + n * D;

    float ga[RESULTS_PER_SG] = {0, 0, 0, 0};
    float ua[RESULTS_PER_SG] = {0, 0, 0, 0};

    if (FC_BITS == 4) {
        for (uint col = simd_lane; col < packed_cols; col += 32) {
            thread float xl[9];
            load_x4(x_in + col * 8, xl);
            uint g = col / pf_gs;

            for (uint r = 0; r < RESULTS_PER_SG; r++) {
                uint row = row_base + r;
                if (row >= I) break;

                uint w_idx = expert_weight_offset + row * packed_cols + col;
                uint sb_idx = expert_scale_offset + row * num_groups + g;

                float2 gqd = qdot4_local(xl, gate_weights[w_idx]);
                ga[r] += bf16_to_f32(gate_scales[sb_idx]) * gqd.x
                       + bf16_to_f32(gate_biases[sb_idx]) * gqd.y;

                float2 uqd = qdot4_local(xl, up_weights[w_idx]);
                ua[r] += bf16_to_f32(up_scales[sb_idx]) * uqd.x
                       + bf16_to_f32(up_biases[sb_idx]) * uqd.y;
            }
        }
    } else {
        for (uint col = simd_lane; col < packed_cols; col += 32) {
            thread float xl[17];
            load_x2(x_in + col * pack_factor, xl);
            uint g = col / pf_gs;

            for (uint r = 0; r < RESULTS_PER_SG; r++) {
                uint row = row_base + r;
                if (row >= I) break;

                uint w_idx = expert_weight_offset + row * packed_cols + col;
                uint sb_idx = expert_scale_offset + row * num_groups + g;

                float2 gqd = qdot2_local(xl, gate_weights[w_idx]);
                ga[r] += bf16_to_f32(gate_scales[sb_idx]) * gqd.x
                       + bf16_to_f32(gate_biases[sb_idx]) * gqd.y;

                float2 uqd = qdot2_local(xl, up_weights[w_idx]);
                ua[r] += bf16_to_f32(up_scales[sb_idx]) * uqd.x
                       + bf16_to_f32(up_biases[sb_idx]) * uqd.y;
            }
        }
    }

    for (uint r = 0; r < RESULTS_PER_SG; r++) {
        uint row = row_base + r;
        if (row >= I) break;
        float rg = simd_sum(ga[r]);
        float ru = simd_sum(ua[r]);
        if (simd_lane == 0) {
            output[(n * K + k) * I + row] = (rg / (1.0f + exp(-rg))) * ru;
        }
    }
}


// ============================================================================
// Kernel 5: Gather + Dequant MatVec (down projection for resident mode, qdot)
// ============================================================================

struct GatherDequantMatvecParams {
    uint in_dim;
    uint out_dim;
    uint num_tokens;
    uint topk;
};

kernel void gather_dequant_matvec(
    device const uint32_t* down_weights [[buffer(0)]],
    device const uint16_t* down_scales  [[buffer(1)]],
    device const uint16_t* down_biases  [[buffer(2)]],
    device const float*    input        [[buffer(3)]],
    device const uint*     expert_ids   [[buffer(4)]],
    device float*          output       [[buffer(5)]],
    constant GatherDequantMatvecParams& params [[buffer(6)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint I = params.in_dim;
    uint D = params.out_dim;
    uint K = params.topk;

    uint tiles_per_token_expert = (D + ROWS_PER_TG - 1) / ROWS_PER_TG;
    uint token_expert_idx = tgid / tiles_per_token_expert;
    uint tile_idx = tgid % tiles_per_token_expert;

    uint n = token_expert_idx / K;
    uint k = token_expert_idx % K;
    uint row_base = tile_idx * ROWS_PER_TG + simd_group * RESULTS_PER_SG;

    if (n >= params.num_tokens || row_base >= D) return;

    uint expert_id = expert_ids[n * K + k];

    uint pack_factor = (FC_BITS == 2) ? 16 : 8;
    uint packed_cols = I / pack_factor;
    uint num_groups = I / FC_GROUP_SIZE;
    uint pf_gs = FC_GROUP_SIZE / pack_factor;

    uint expert_weight_offset = expert_id * D * packed_cols;
    uint expert_scale_offset = expert_id * D * num_groups;

    device const float* x_in = input + (n * K + k) * I;

    float acc[RESULTS_PER_SG] = {0, 0, 0, 0};

    if (FC_BITS == 4) {
        for (uint col = simd_lane; col < packed_cols; col += 32) {
            thread float xl[9];
            load_x4(x_in + col * 8, xl);
            uint g = col / pf_gs;

            for (uint r = 0; r < RESULTS_PER_SG; r++) {
                uint row = row_base + r;
                if (row >= D) break;

                uint w_idx = expert_weight_offset + row * packed_cols + col;
                uint sb_idx = expert_scale_offset + row * num_groups + g;

                float2 qd = qdot4_local(xl, down_weights[w_idx]);
                acc[r] += bf16_to_f32(down_scales[sb_idx]) * qd.x
                        + bf16_to_f32(down_biases[sb_idx]) * qd.y;
            }
        }
    } else {
        for (uint col = simd_lane; col < packed_cols; col += 32) {
            thread float xl[17];
            load_x2(x_in + col * pack_factor, xl);
            uint g = col / pf_gs;

            for (uint r = 0; r < RESULTS_PER_SG; r++) {
                uint row = row_base + r;
                if (row >= D) break;

                uint w_idx = expert_weight_offset + row * packed_cols + col;
                uint sb_idx = expert_scale_offset + row * num_groups + g;

                float2 qd = qdot2_local(xl, down_weights[w_idx]);
                acc[r] += bf16_to_f32(down_scales[sb_idx]) * qd.x
                        + bf16_to_f32(down_biases[sb_idx]) * qd.y;
            }
        }
    }

    for (uint r = 0; r < RESULTS_PER_SG; r++) {
        uint row = row_base + r;
        if (row >= D) break;
        float sum = simd_sum(acc[r]);
        if (simd_lane == 0) {
            output[(n * K + k) * D + row] = sum;
        }
    }
}
