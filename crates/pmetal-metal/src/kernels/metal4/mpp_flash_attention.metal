// mpp_flash_attention.metal
// Metal 4 FlashAttention using MPP matmul2d for BOTH block GEMMs.
//
// Key improvements over Metal 3 flash_attention.metal:
// - Block GEMM for S = Q @ K^T via matmul2d
// - Block GEMM for O += P @ V via matmul2d (4 chunks of 32 for D=128)
// - Online block softmax applied to score tile in threadgroup memory
// - No explicit threadgroup memory staging for GEMM (MPP handles cache hierarchy)
//
// Algorithm (FlashAttention-2 with block GEMM):
//   For each Q block (Bq rows):
//     Load Q tile into threadgroup memory
//     For each KV block (Bk rows):
//       S[Bq,Bk] = Q @ K^T                 ← MPP matmul2d (32×32)
//       Online softmax on S → P in half
//       Apply correction to O accumulator
//       O_partial[Bq,D] = P @ V             ← MPP matmul2d (32×32, 4 D-chunks)
//       o_local += O_partial
//     Normalize O by final l
//
// Threadgroup memory budget (30KB, under 32KB limit):
//   Q_tile: 32 × 128 × 2 =  8KB (loaded once, reused across KV blocks)
//   K_tile: 32 × 128 × 2 =  8KB (repurposed as O_partial after QK)
//   V_tile: 32 × 128 × 2 =  8KB
//   S_tile: 32 ×  32 × 4 =  4KB (score matrix, float for softmax precision)
//   P_half: 32 ×  32 × 2 =  2KB (softmax probs in half for PV matmul)
//
// References:
// - FlashAttention-2: https://arxiv.org/abs/2307.08691
// - MPP Guide Section 2.3.2: SM=SN=32 for fp16

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// =============================================================================
// Configuration
// =============================================================================

struct FlashAttentionParams {
    uint batch_size;
    uint num_heads;
    uint num_kv_heads;
    uint query_seq_len;
    uint kv_seq_len;
    uint head_dim;
    float scale;
    uint block_q;
    uint block_k;
    uint gqa_ratio;
    uint is_causal;
    uint sliding_window;
    float softcap;
};

constant uint SIMD_SIZE = 32;

// =============================================================================
// Utility functions
// =============================================================================

inline float simd_max_f32(float val) {
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, simd_shuffle_xor(val, offset));
    }
    return val;
}

inline float simd_sum_f32(float val) {
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        val += simd_shuffle_xor(val, offset);
    }
    return val;
}

// =============================================================================
// FlashAttention Forward with Block GEMM (D=64, causal)
// =============================================================================

kernel void mpp_flash_attention_fwd_d64_causal(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    constant FlashAttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D = 64;
    const uint ROWS_PER_GROUP = 8;

    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;
    const uint q_start = q_block_idx * Bq;

    const uint q_batch_stride = params.num_heads * params.query_seq_len * D;
    const uint q_head_stride = params.query_seq_len * D;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D;
    const uint kv_head_stride = params.kv_seq_len * D;

    device half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half V_tile[Bk * D];
    threadgroup float S_tile[Bq * Bk];
    threadgroup half P_half[Bq * Bk];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;
    const uint linear_tid = tid.y * SIMD_SIZE + tid.x;

    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][2];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = 0.0f;
    }

    for (uint i = linear_tid; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len)
                   ? Q_head[global_q_row * D + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_end = min(q_start + Bq, params.kv_seq_len);
    uint num_kv_blocks = (kv_end + Bk - 1) / Bk;

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, true, false
    );
    mpp::tensor_ops::matmul2d<qk_desc, execution_simdgroup> qk_op;

    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, false, false
    );
    mpp::tensor_ops::matmul2d<pv_desc, execution_simdgroup> pv_op;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        for (uint i = linear_tid; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual) ? K_head[global_row * D + col] : half(0.0f);
            V_tile[i] = (global_row < k_end_actual) ? V_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tQ = tensor((threadgroup half*)Q_tile,
                         dextents<int, 2>{(int)D, (int)Bq},
                         array<int, 2>{1, (int)D});
        auto tK = tensor((threadgroup half*)K_tile,
                         dextents<int, 2>{(int)D, (int)Bk},
                         array<int, 2>{1, (int)D});
        auto tS = tensor((threadgroup float*)S_tile,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});
        qk_op.run(tQ, tK, tS);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) {
                for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                    P_half[my_row * Bk + k] = half(0.0f);
                }
                continue;
            }

            float row_max = -INFINITY;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                uint global_k_pos = k_start + k;
                float s = S_tile[my_row * Bk + k] * params.scale;
                if (global_k_pos > global_q_pos || global_k_pos >= k_end_actual) {
                    s = -INFINITY;
                }
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) {
                    s = -INFINITY;
                }
                if (params.softcap > 0.0f && s > -INFINITY) {
                    s = params.softcap * tanh(s / params.softcap);
                }
                S_tile[my_row * Bk + k] = s;
                row_max = max(row_max, s);
            }
            row_max = simd_max_f32(row_max);

            float m_new = max(m_i[r], row_max);
            float correction = metal::exp(m_i[r] - m_new);

            float row_sum = 0.0f;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                float s = S_tile[my_row * Bk + k];
                float p = (s > -INFINITY) ? metal::exp(s - m_new) : 0.0f;
                P_half[my_row * Bk + k] = half(p);
                row_sum += p;
            }
            row_sum = simd_sum_f32(row_sum);

            l_i[r] = correction * l_i[r] + row_sum;

            #pragma unroll
            for (uint dd = 0; dd < 2; dd++) {
                o_local[r][dd] *= correction;
            }

            m_i[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half* O_partial = (threadgroup half*)K_tile;

        auto tP = tensor((threadgroup half*)P_half,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});
        auto tV_full = tensor((threadgroup half*)V_tile,
                              dextents<int, 2>{(int)D, (int)Bk},
                              array<int, 2>{1, (int)D});
        auto tO_full = tensor(O_partial,
                              dextents<int, 2>{(int)D, (int)Bq},
                              array<int, 2>{1, (int)D});

        #pragma unroll
        for (uint d_start = 0; d_start < D; d_start += 32) {
            auto tV_chunk = tV_full.slice((int)d_start, 0);
            auto tO_chunk = tO_full.slice((int)d_start, 0);
            pv_op.run(tP, tV_chunk, tO_chunk);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            if (my_row >= Bq) continue;

            #pragma unroll
            for (uint dd = 0; dd < 2; dd++) {
                uint d = simd_lane_id * 2 + dd;
                o_local[r][dd] += float(O_partial[my_row * D + d]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;
        #pragma unroll
        for (uint dd = 0; dd < 2; dd++) {
            uint d = simd_lane_id * 2 + dd;
            O_head[global_q_idx * D + d] = half(o_local[r][dd] * inv_l);
        }

        if (simd_lane_id == 0) {
            L_head[global_q_idx] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

// =============================================================================
// FlashAttention Forward (D=64, non-causal)
// =============================================================================

kernel void mpp_flash_attention_fwd_d64(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    constant FlashAttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D = 64;
    const uint ROWS_PER_GROUP = 8;

    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;
    const uint q_start = q_block_idx * Bq;

    const uint q_batch_stride = params.num_heads * params.query_seq_len * D;
    const uint q_head_stride = params.query_seq_len * D;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D;
    const uint kv_head_stride = params.kv_seq_len * D;

    device half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half V_tile[Bk * D];
    threadgroup float S_tile[Bq * Bk];
    threadgroup half P_half[Bq * Bk];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;
    const uint linear_tid = tid.y * SIMD_SIZE + tid.x;

    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][2];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = 0.0f;
    }

    for (uint i = linear_tid; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len)
                   ? Q_head[global_q_row * D + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_kv_blocks = (params.kv_seq_len + Bk - 1) / Bk;

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, true, false
    );
    mpp::tensor_ops::matmul2d<qk_desc, execution_simdgroup> qk_op;

    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, false, false
    );
    mpp::tensor_ops::matmul2d<pv_desc, execution_simdgroup> pv_op;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        for (uint i = linear_tid; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual) ? K_head[global_row * D + col] : half(0.0f);
            V_tile[i] = (global_row < k_end_actual) ? V_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tQ = tensor((threadgroup half*)Q_tile,
                         dextents<int, 2>{(int)D, (int)Bq}, array<int, 2>{1, (int)D});
        auto tK = tensor((threadgroup half*)K_tile,
                         dextents<int, 2>{(int)D, (int)Bk}, array<int, 2>{1, (int)D});
        auto tS = tensor((threadgroup float*)S_tile,
                         dextents<int, 2>{(int)Bk, (int)Bq}, array<int, 2>{1, (int)Bk});
        qk_op.run(tQ, tK, tS);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) {
                for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                    P_half[my_row * Bk + k] = half(0.0f);
                }
                continue;
            }

            float row_max = -INFINITY;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                uint global_k_pos = k_start + k;
                float s = S_tile[my_row * Bk + k] * params.scale;
                if (global_k_pos >= k_end_actual) s = -INFINITY;
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) s = -INFINITY;
                if (params.softcap > 0.0f && s > -INFINITY) s = params.softcap * tanh(s / params.softcap);
                S_tile[my_row * Bk + k] = s;
                row_max = max(row_max, s);
            }
            row_max = simd_max_f32(row_max);

            float m_new = max(m_i[r], row_max);
            float correction = metal::exp(m_i[r] - m_new);

            float row_sum = 0.0f;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                float s = S_tile[my_row * Bk + k];
                float p = (s > -INFINITY) ? metal::exp(s - m_new) : 0.0f;
                P_half[my_row * Bk + k] = half(p);
                row_sum += p;
            }
            row_sum = simd_sum_f32(row_sum);

            l_i[r] = correction * l_i[r] + row_sum;

            #pragma unroll
            for (uint dd = 0; dd < 2; dd++) {
                o_local[r][dd] *= correction;
            }

            m_i[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half* O_partial = (threadgroup half*)K_tile;

        auto tP = tensor((threadgroup half*)P_half,
                         dextents<int, 2>{(int)Bk, (int)Bq}, array<int, 2>{1, (int)Bk});
        auto tV_full = tensor((threadgroup half*)V_tile,
                              dextents<int, 2>{(int)D, (int)Bk}, array<int, 2>{1, (int)D});
        auto tO_full = tensor(O_partial,
                              dextents<int, 2>{(int)D, (int)Bq}, array<int, 2>{1, (int)D});

        #pragma unroll
        for (uint d_start = 0; d_start < D; d_start += 32) {
            auto tV_chunk = tV_full.slice((int)d_start, 0);
            auto tO_chunk = tO_full.slice((int)d_start, 0);
            pv_op.run(tP, tV_chunk, tO_chunk);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            if (my_row >= Bq) continue;

            #pragma unroll
            for (uint dd = 0; dd < 2; dd++) {
                uint d = simd_lane_id * 2 + dd;
                o_local[r][dd] += float(O_partial[my_row * D + d]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;
        #pragma unroll
        for (uint dd = 0; dd < 2; dd++) {
            uint d = simd_lane_id * 2 + dd;
            O_head[global_q_idx * D + d] = half(o_local[r][dd] * inv_l);
        }
        if (simd_lane_id == 0) {
            L_head[global_q_idx] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

// =============================================================================
// FlashAttention Forward with Block GEMM (D=80, causal via padded D=96 tiles)
// =============================================================================
//
// Uses an internal D=96 tile so the MPP 32x32 matmul contract stays identical to
// the existing D=96 kernel. Q/K/V loads zero-pad columns 80..95, and writeback
// only stores the first 80 output elements per row.

kernel void mpp_flash_attention_fwd_d80_causal(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    constant FlashAttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D_EXT = 80;
    const uint D_PAD = 96;
    const uint ROWS_PER_GROUP = 8;

    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;
    const uint q_start = q_block_idx * Bq;

    const uint q_batch_stride = params.num_heads * params.query_seq_len * D_EXT;
    const uint q_head_stride = params.query_seq_len * D_EXT;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D_EXT;
    const uint kv_head_stride = params.kv_seq_len * D_EXT;

    device half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    threadgroup half Q_tile[Bq * D_PAD];
    threadgroup half K_tile[Bk * D_PAD];
    threadgroup half V_tile[Bk * D_PAD];
    threadgroup float S_tile[Bq * Bk];
    threadgroup half P_half[Bq * Bk];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;
    const uint linear_tid = tid.y * SIMD_SIZE + tid.x;

    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][3];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = o_local[r][2] = 0.0f;
    }

    for (uint i = linear_tid; i < Bq * D_PAD; i += 128) {
        uint q_row = i / D_PAD;
        uint q_col = i % D_PAD;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len && q_col < D_EXT)
                   ? Q_head[global_q_row * D_EXT + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_end = min(q_start + Bq, params.kv_seq_len);
    uint num_kv_blocks = (kv_end + Bk - 1) / Bk;

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, true, false
    );
    mpp::tensor_ops::matmul2d<qk_desc, execution_simdgroup> qk_op;

    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, false, false
    );
    mpp::tensor_ops::matmul2d<pv_desc, execution_simdgroup> pv_op;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        for (uint i = linear_tid; i < Bk * D_PAD; i += 128) {
            uint row = i / D_PAD;
            uint col = i % D_PAD;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual && col < D_EXT)
                      ? K_head[global_row * D_EXT + col] : half(0.0f);
            V_tile[i] = (global_row < k_end_actual && col < D_EXT)
                      ? V_head[global_row * D_EXT + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tQ = tensor((threadgroup half*)Q_tile,
                         dextents<int, 2>{(int)D_PAD, (int)Bq},
                         array<int, 2>{1, (int)D_PAD});
        auto tK = tensor((threadgroup half*)K_tile,
                         dextents<int, 2>{(int)D_PAD, (int)Bk},
                         array<int, 2>{1, (int)D_PAD});
        auto tS = tensor((threadgroup float*)S_tile,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});
        qk_op.run(tQ, tK, tS);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) {
                for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                    P_half[my_row * Bk + k] = half(0.0f);
                }
                continue;
            }

            float row_max = -INFINITY;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                uint global_k_pos = k_start + k;
                float s = S_tile[my_row * Bk + k] * params.scale;

                if (global_k_pos > global_q_pos || global_k_pos >= k_end_actual) {
                    s = -INFINITY;
                }
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) {
                    s = -INFINITY;
                }
                if (params.softcap > 0.0f && s > -INFINITY) {
                    s = params.softcap * tanh(s / params.softcap);
                }
                S_tile[my_row * Bk + k] = s;
                row_max = max(row_max, s);
            }
            row_max = simd_max_f32(row_max);

            float m_new = max(m_i[r], row_max);
            float correction = metal::exp(m_i[r] - m_new);

            float row_sum = 0.0f;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                float s = S_tile[my_row * Bk + k];
                float p = (s > -INFINITY) ? metal::exp(s - m_new) : 0.0f;
                P_half[my_row * Bk + k] = half(p);
                row_sum += p;
            }
            row_sum = simd_sum_f32(row_sum);

            l_i[r] = correction * l_i[r] + row_sum;

            #pragma unroll
            for (uint dd = 0; dd < 3; dd++) {
                o_local[r][dd] *= correction;
            }

            m_i[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half* O_partial = (threadgroup half*)K_tile;

        auto tP = tensor((threadgroup half*)P_half,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});
        auto tV_full = tensor((threadgroup half*)V_tile,
                              dextents<int, 2>{(int)D_PAD, (int)Bk},
                              array<int, 2>{1, (int)D_PAD});
        auto tO_full = tensor(O_partial,
                              dextents<int, 2>{(int)D_PAD, (int)Bq},
                              array<int, 2>{1, (int)D_PAD});

        #pragma unroll
        for (uint d_start = 0; d_start < D_PAD; d_start += 32) {
            auto tV_chunk = tV_full.slice((int)d_start, 0);
            auto tO_chunk = tO_full.slice((int)d_start, 0);
            pv_op.run(tP, tV_chunk, tO_chunk);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            if (my_row >= Bq) continue;

            #pragma unroll
            for (uint dd = 0; dd < 3; dd++) {
                uint d = simd_lane_id * 3 + dd;
                o_local[r][dd] += float(O_partial[my_row * D_PAD + d]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;

        #pragma unroll
        for (uint dd = 0; dd < 3; dd++) {
            uint d = simd_lane_id * 3 + dd;
            if (d < D_EXT) {
                O_head[global_q_idx * D_EXT + d] = half(o_local[r][dd] * inv_l);
            }
        }

        if (simd_lane_id == 0) {
            L_head[global_q_idx] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

// =============================================================================
// FlashAttention Forward (D=80, non-causal via padded D=96 tiles)
// =============================================================================

kernel void mpp_flash_attention_fwd_d80(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    constant FlashAttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D_EXT = 80;
    const uint D_PAD = 96;
    const uint ROWS_PER_GROUP = 8;

    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;
    const uint q_start = q_block_idx * Bq;

    const uint q_batch_stride = params.num_heads * params.query_seq_len * D_EXT;
    const uint q_head_stride = params.query_seq_len * D_EXT;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D_EXT;
    const uint kv_head_stride = params.kv_seq_len * D_EXT;

    device half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    threadgroup half Q_tile[Bq * D_PAD];
    threadgroup half K_tile[Bk * D_PAD];
    threadgroup half V_tile[Bk * D_PAD];
    threadgroup float S_tile[Bq * Bk];
    threadgroup half P_half[Bq * Bk];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;
    const uint linear_tid = tid.y * SIMD_SIZE + tid.x;

    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][3];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = o_local[r][2] = 0.0f;
    }

    for (uint i = linear_tid; i < Bq * D_PAD; i += 128) {
        uint q_row = i / D_PAD;
        uint q_col = i % D_PAD;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len && q_col < D_EXT)
                   ? Q_head[global_q_row * D_EXT + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_kv_blocks = (params.kv_seq_len + Bk - 1) / Bk;

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, true, false
    );
    mpp::tensor_ops::matmul2d<qk_desc, execution_simdgroup> qk_op;

    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, false, false
    );
    mpp::tensor_ops::matmul2d<pv_desc, execution_simdgroup> pv_op;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        for (uint i = linear_tid; i < Bk * D_PAD; i += 128) {
            uint row = i / D_PAD;
            uint col = i % D_PAD;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual && col < D_EXT)
                      ? K_head[global_row * D_EXT + col] : half(0.0f);
            V_tile[i] = (global_row < k_end_actual && col < D_EXT)
                      ? V_head[global_row * D_EXT + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tQ = tensor((threadgroup half*)Q_tile,
                         dextents<int, 2>{(int)D_PAD, (int)Bq},
                         array<int, 2>{1, (int)D_PAD});
        auto tK = tensor((threadgroup half*)K_tile,
                         dextents<int, 2>{(int)D_PAD, (int)Bk},
                         array<int, 2>{1, (int)D_PAD});
        auto tS = tensor((threadgroup float*)S_tile,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});
        qk_op.run(tQ, tK, tS);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) {
                for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                    P_half[my_row * Bk + k] = half(0.0f);
                }
                continue;
            }

            float row_max = -INFINITY;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                uint global_k_pos = k_start + k;
                float s = S_tile[my_row * Bk + k] * params.scale;
                if (global_k_pos >= k_end_actual) s = -INFINITY;
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) s = -INFINITY;
                if (params.softcap > 0.0f && s > -INFINITY) s = params.softcap * tanh(s / params.softcap);
                S_tile[my_row * Bk + k] = s;
                row_max = max(row_max, s);
            }
            row_max = simd_max_f32(row_max);

            float m_new = max(m_i[r], row_max);
            float correction = metal::exp(m_i[r] - m_new);

            float row_sum = 0.0f;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                float s = S_tile[my_row * Bk + k];
                float p = (s > -INFINITY) ? metal::exp(s - m_new) : 0.0f;
                P_half[my_row * Bk + k] = half(p);
                row_sum += p;
            }
            row_sum = simd_sum_f32(row_sum);

            l_i[r] = correction * l_i[r] + row_sum;

            #pragma unroll
            for (uint dd = 0; dd < 3; dd++) {
                o_local[r][dd] *= correction;
            }

            m_i[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half* O_partial = (threadgroup half*)K_tile;

        auto tP = tensor((threadgroup half*)P_half,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});
        auto tV_full = tensor((threadgroup half*)V_tile,
                              dextents<int, 2>{(int)D_PAD, (int)Bk},
                              array<int, 2>{1, (int)D_PAD});
        auto tO_full = tensor(O_partial,
                              dextents<int, 2>{(int)D_PAD, (int)Bq},
                              array<int, 2>{1, (int)D_PAD});

        #pragma unroll
        for (uint d_start = 0; d_start < D_PAD; d_start += 32) {
            auto tV_chunk = tV_full.slice((int)d_start, 0);
            auto tO_chunk = tO_full.slice((int)d_start, 0);
            pv_op.run(tP, tV_chunk, tO_chunk);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            if (my_row >= Bq) continue;

            #pragma unroll
            for (uint dd = 0; dd < 3; dd++) {
                uint d = simd_lane_id * 3 + dd;
                o_local[r][dd] += float(O_partial[my_row * D_PAD + d]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;
        #pragma unroll
        for (uint dd = 0; dd < 3; dd++) {
            uint d = simd_lane_id * 3 + dd;
            if (d < D_EXT) {
                O_head[global_q_idx * D_EXT + d] = half(o_local[r][dd] * inv_l);
            }
        }
        if (simd_lane_id == 0) {
            L_head[global_q_idx] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

// =============================================================================
// FlashAttention Forward with Block GEMM (D=96, causal)
// =============================================================================
//
// Thread organization:
//   Grid: [num_q_blocks, num_heads, batch_size]
//   Threadgroup: [32, 4, 1] = 128 threads (4 SIMD groups)
//
// Block GEMM via matmul2d (all 32×32 tiles, 4 simdgroups):
//   S[32,32] = Q[32,96] @ K[32,96]^T  → desc(32,32,dyn,false,true,false)
//   O_chunk[32,32] = P[32,32] @ V_chunk[32,32]  → desc(32,32,dyn,false,false,false)
//   (3 chunks for D=96)

kernel void mpp_flash_attention_fwd_d96_causal(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    constant FlashAttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D = 96;
    const uint ROWS_PER_GROUP = 8;

    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;
    const uint q_start = q_block_idx * Bq;

    const uint q_batch_stride = params.num_heads * params.query_seq_len * D;
    const uint q_head_stride = params.query_seq_len * D;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D;
    const uint kv_head_stride = params.kv_seq_len * D;

    device half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    // Threadgroup memory (26KB total, under 32KB)
    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half V_tile[Bk * D];
    threadgroup float S_tile[Bq * Bk];
    threadgroup half P_half[Bq * Bk];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;
    const uint linear_tid = tid.y * SIMD_SIZE + tid.x;

    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][3];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = o_local[r][2] = 0.0f;
    }

    for (uint i = linear_tid; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len)
                   ? Q_head[global_q_row * D + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_end = min(q_start + Bq, params.kv_seq_len);
    uint num_kv_blocks = (kv_end + Bk - 1) / Bk;

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, true, false
    );
    mpp::tensor_ops::matmul2d<qk_desc, execution_simdgroup> qk_op;

    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, false, false
    );
    mpp::tensor_ops::matmul2d<pv_desc, execution_simdgroup> pv_op;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        for (uint i = linear_tid; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual) ? K_head[global_row * D + col] : half(0.0f);
            V_tile[i] = (global_row < k_end_actual) ? V_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tQ = tensor((threadgroup half*)Q_tile,
                         dextents<int, 2>{(int)D, (int)Bq},
                         array<int, 2>{1, (int)D});
        auto tK = tensor((threadgroup half*)K_tile,
                         dextents<int, 2>{(int)D, (int)Bk},
                         array<int, 2>{1, (int)D});
        auto tS = tensor((threadgroup float*)S_tile,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});
        qk_op.run(tQ, tK, tS);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) {
                for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                    P_half[my_row * Bk + k] = half(0.0f);
                }
                continue;
            }

            float row_max = -INFINITY;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                uint global_k_pos = k_start + k;
                float s = S_tile[my_row * Bk + k] * params.scale;

                if (global_k_pos > global_q_pos || global_k_pos >= k_end_actual) {
                    s = -INFINITY;
                }
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) {
                    s = -INFINITY;
                }
                if (params.softcap > 0.0f && s > -INFINITY) {
                    s = params.softcap * tanh(s / params.softcap);
                }
                S_tile[my_row * Bk + k] = s;
                row_max = max(row_max, s);
            }
            row_max = simd_max_f32(row_max);

            float m_new = max(m_i[r], row_max);
            float correction = metal::exp(m_i[r] - m_new);

            float row_sum = 0.0f;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                float s = S_tile[my_row * Bk + k];
                float p = (s > -INFINITY) ? metal::exp(s - m_new) : 0.0f;
                P_half[my_row * Bk + k] = half(p);
                row_sum += p;
            }
            row_sum = simd_sum_f32(row_sum);

            l_i[r] = correction * l_i[r] + row_sum;

            #pragma unroll
            for (uint dd = 0; dd < 3; dd++) {
                o_local[r][dd] *= correction;
            }

            m_i[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half* O_partial = (threadgroup half*)K_tile;

        auto tP = tensor((threadgroup half*)P_half,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});
        auto tV_full = tensor((threadgroup half*)V_tile,
                              dextents<int, 2>{(int)D, (int)Bk},
                              array<int, 2>{1, (int)D});
        auto tO_full = tensor(O_partial,
                              dextents<int, 2>{(int)D, (int)Bq},
                              array<int, 2>{1, (int)D});

        #pragma unroll
        for (uint d_start = 0; d_start < D; d_start += 32) {
            auto tV_chunk = tV_full.slice((int)d_start, 0);
            auto tO_chunk = tO_full.slice((int)d_start, 0);
            pv_op.run(tP, tV_chunk, tO_chunk);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            if (my_row >= Bq) continue;

            #pragma unroll
            for (uint dd = 0; dd < 3; dd++) {
                uint d = simd_lane_id * 3 + dd;
                o_local[r][dd] += float(O_partial[my_row * D + d]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;

        #pragma unroll
        for (uint dd = 0; dd < 3; dd++) {
            uint d = simd_lane_id * 3 + dd;
            O_head[global_q_idx * D + d] = half(o_local[r][dd] * inv_l);
        }

        if (simd_lane_id == 0) {
            L_head[global_q_idx] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

// =============================================================================
// FlashAttention Forward (D=96, non-causal)
// =============================================================================

kernel void mpp_flash_attention_fwd_d96(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    constant FlashAttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D = 96;
    const uint ROWS_PER_GROUP = 8;

    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;
    const uint q_start = q_block_idx * Bq;

    const uint q_batch_stride = params.num_heads * params.query_seq_len * D;
    const uint q_head_stride = params.query_seq_len * D;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D;
    const uint kv_head_stride = params.kv_seq_len * D;

    device half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half V_tile[Bk * D];
    threadgroup float S_tile[Bq * Bk];
    threadgroup half P_half[Bq * Bk];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;
    const uint linear_tid = tid.y * SIMD_SIZE + tid.x;

    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][3];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = o_local[r][2] = 0.0f;
    }

    for (uint i = linear_tid; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len)
                   ? Q_head[global_q_row * D + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_kv_blocks = (params.kv_seq_len + Bk - 1) / Bk;

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, true, false
    );
    mpp::tensor_ops::matmul2d<qk_desc, execution_simdgroup> qk_op;

    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, false, false
    );
    mpp::tensor_ops::matmul2d<pv_desc, execution_simdgroup> pv_op;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        for (uint i = linear_tid; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual) ? K_head[global_row * D + col] : half(0.0f);
            V_tile[i] = (global_row < k_end_actual) ? V_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tQ = tensor((threadgroup half*)Q_tile,
                         dextents<int, 2>{(int)D, (int)Bq}, array<int, 2>{1, (int)D});
        auto tK = tensor((threadgroup half*)K_tile,
                         dextents<int, 2>{(int)D, (int)Bk}, array<int, 2>{1, (int)D});
        auto tS = tensor((threadgroup float*)S_tile,
                         dextents<int, 2>{(int)Bk, (int)Bq}, array<int, 2>{1, (int)Bk});
        qk_op.run(tQ, tK, tS);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) {
                for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                    P_half[my_row * Bk + k] = half(0.0f);
                }
                continue;
            }

            float row_max = -INFINITY;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                uint global_k_pos = k_start + k;
                float s = S_tile[my_row * Bk + k] * params.scale;
                if (global_k_pos >= k_end_actual) s = -INFINITY;
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) s = -INFINITY;
                if (params.softcap > 0.0f && s > -INFINITY) s = params.softcap * tanh(s / params.softcap);
                S_tile[my_row * Bk + k] = s;
                row_max = max(row_max, s);
            }
            row_max = simd_max_f32(row_max);

            float m_new = max(m_i[r], row_max);
            float correction = metal::exp(m_i[r] - m_new);

            float row_sum = 0.0f;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                float s = S_tile[my_row * Bk + k];
                float p = (s > -INFINITY) ? metal::exp(s - m_new) : 0.0f;
                P_half[my_row * Bk + k] = half(p);
                row_sum += p;
            }
            row_sum = simd_sum_f32(row_sum);

            l_i[r] = correction * l_i[r] + row_sum;

            #pragma unroll
            for (uint dd = 0; dd < 3; dd++) {
                o_local[r][dd] *= correction;
            }

            m_i[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half* O_partial = (threadgroup half*)K_tile;

        auto tP = tensor((threadgroup half*)P_half,
                         dextents<int, 2>{(int)Bk, (int)Bq}, array<int, 2>{1, (int)Bk});
        auto tV_full = tensor((threadgroup half*)V_tile,
                              dextents<int, 2>{(int)D, (int)Bk}, array<int, 2>{1, (int)D});
        auto tO_full = tensor(O_partial,
                              dextents<int, 2>{(int)D, (int)Bq}, array<int, 2>{1, (int)D});

        #pragma unroll
        for (uint d_start = 0; d_start < D; d_start += 32) {
            auto tV_chunk = tV_full.slice((int)d_start, 0);
            auto tO_chunk = tO_full.slice((int)d_start, 0);
            pv_op.run(tP, tV_chunk, tO_chunk);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            if (my_row >= Bq) continue;

            #pragma unroll
            for (uint dd = 0; dd < 3; dd++) {
                uint d = simd_lane_id * 3 + dd;
                o_local[r][dd] += float(O_partial[my_row * D + d]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;
        #pragma unroll
        for (uint dd = 0; dd < 3; dd++) {
            uint d = simd_lane_id * 3 + dd;
            O_head[global_q_idx * D + d] = half(o_local[r][dd] * inv_l);
        }
        if (simd_lane_id == 0) {
            L_head[global_q_idx] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

// =============================================================================
// FlashAttention Forward with Block GEMM (D=128, causal)
// =============================================================================
//
// Thread organization:
//   Grid: [num_q_blocks, num_heads, batch_size]
//   Threadgroup: [32, 4, 1] = 128 threads (4 SIMD groups)
//
// Block GEMM via matmul2d (all 32×32 tiles, 4 simdgroups):
//   S[32,32] = Q[32,128] @ K[32,128]^T  → desc(32,32,dyn,false,true,false)
//   O_chunk[32,32] = P[32,32] @ V_chunk[32,32]  → desc(32,32,dyn,false,false,false)
//   (4 chunks for D=128)

kernel void mpp_flash_attention_fwd_d128_causal(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    constant FlashAttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D = 128;
    const uint ROWS_PER_GROUP = 8;

    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;
    const uint q_start = q_block_idx * Bq;

    const uint q_batch_stride = params.num_heads * params.query_seq_len * D;
    const uint q_head_stride = params.query_seq_len * D;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D;
    const uint kv_head_stride = params.kv_seq_len * D;

    device half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    // Threadgroup memory (30KB total, under 32KB)
    threadgroup half Q_tile[Bq * D];       // 8KB - Q block, loaded once
    threadgroup half K_tile[Bk * D];       // 8KB - K block, repurposed as O_partial after QK
    threadgroup half V_tile[Bk * D];       // 8KB - V block
    threadgroup float S_tile[Bq * Bk];     // 4KB - score matrix (float for softmax)
    threadgroup half P_half[Bq * Bk];      // 2KB - softmax probs in half for PV matmul

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;
    const uint linear_tid = tid.y * SIMD_SIZE + tid.x;

    // Per-row accumulators in registers
    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][4];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = o_local[r][2] = o_local[r][3] = 0.0f;
    }

    // Load Q tile cooperatively (stays resident across all KV blocks)
    for (uint i = linear_tid; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len)
                   ? Q_head[global_q_row * D + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Causal: only process KV positions up to end of this Q block
    uint kv_end = min(q_start + Bq, params.kv_seq_len);
    uint num_kv_blocks = (kv_end + Bk - 1) / Bk;

    // MPP matmul ops (compile-time descriptors)
    // QK: S[32,32] = Q[32,D] @ K[32,D]^T — transpose_right=true
    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, true, false
    );
    mpp::tensor_ops::matmul2d<qk_desc, execution_simdgroup> qk_op;

    // PV: O_chunk[32,32] = P[32,32] @ V_chunk[32,32] — no transpose
    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, false, false
    );
    mpp::tensor_ops::matmul2d<pv_desc, execution_simdgroup> pv_op;

    // Main loop over KV blocks
    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        // Load K and V tiles cooperatively
        for (uint i = linear_tid; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual) ? K_head[global_row * D + col] : half(0.0f);
            V_tile[i] = (global_row < k_end_actual) ? V_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =================================================================
        // Block GEMM: S = Q @ K^T  [Bq × Bk] via matmul2d
        // =================================================================
        auto tQ = tensor((threadgroup half*)Q_tile,
                         dextents<int, 2>{(int)D, (int)Bq},
                         array<int, 2>{1, (int)D});
        auto tK = tensor((threadgroup half*)K_tile,
                         dextents<int, 2>{(int)D, (int)Bk},
                         array<int, 2>{1, (int)D});
        auto tS = tensor((threadgroup float*)S_tile,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});
        qk_op.run(tQ, tK, tS);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =================================================================
        // Online softmax: scale, mask, exp, normalize
        // Write P values to P_half directly (saves a conversion pass)
        // =================================================================
        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) {
                // Zero out this row's P_half entries
                for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                    P_half[my_row * Bk + k] = half(0.0f);
                }
                continue;
            }

            // Pass 1: Scale + mask → find row max
            float row_max = -INFINITY;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                uint global_k_pos = k_start + k;
                float s = S_tile[my_row * Bk + k] * params.scale;

                // Causal mask
                if (global_k_pos > global_q_pos || global_k_pos >= k_end_actual) {
                    s = -INFINITY;
                }
                // Sliding window
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) {
                    s = -INFINITY;
                }
                // Softcap
                if (params.softcap > 0.0f && s > -INFINITY) {
                    s = params.softcap * tanh(s / params.softcap);
                }
                S_tile[my_row * Bk + k] = s;
                row_max = max(row_max, s);
            }
            row_max = simd_max_f32(row_max);

            // Online softmax correction
            float m_new = max(m_i[r], row_max);
            float correction = metal::exp(m_i[r] - m_new);

            // Pass 2: exp(s - m_new) → P values, write to P_half for PV matmul
            float row_sum = 0.0f;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                float s = S_tile[my_row * Bk + k];
                float p = (s > -INFINITY) ? metal::exp(s - m_new) : 0.0f;
                P_half[my_row * Bk + k] = half(p);
                row_sum += p;
            }
            row_sum = simd_sum_f32(row_sum);

            l_i[r] = correction * l_i[r] + row_sum;

            // Apply correction to existing O accumulator
            #pragma unroll
            for (uint dd = 0; dd < 4; dd++) {
                o_local[r][dd] *= correction;
            }

            m_i[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =================================================================
        // Block GEMM: O_partial = P @ V  [Bq × D] via matmul2d
        // =================================================================
        // Process D in 4 chunks of 32, each using a 32×32 matmul.
        // K_tile memory is repurposed as O_partial (same layout: [Bq, D] half).
        threadgroup half* O_partial = (threadgroup half*)K_tile;

        auto tP = tensor((threadgroup half*)P_half,
                         dextents<int, 2>{(int)Bk, (int)Bq},
                         array<int, 2>{1, (int)Bk});

        // V_tile: [Bk, D] row-major. Create full tensor for slicing.
        auto tV_full = tensor((threadgroup half*)V_tile,
                              dextents<int, 2>{(int)D, (int)Bk},
                              array<int, 2>{1, (int)D});

        // O_partial: [Bq, D] row-major. Create full tensor for slicing.
        auto tO_full = tensor(O_partial,
                              dextents<int, 2>{(int)D, (int)Bq},
                              array<int, 2>{1, (int)D});

        // 4 chunks of 32 columns each
        #pragma unroll
        for (uint d_start = 0; d_start < D; d_start += 32) {
            auto tV_chunk = tV_full.slice((int)d_start, 0);
            auto tO_chunk = tO_full.slice((int)d_start, 0);
            pv_op.run(tP, tV_chunk, tO_chunk);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =================================================================
        // Accumulate O_partial into per-thread registers
        // =================================================================
        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            if (my_row >= Bq) continue;

            #pragma unroll
            for (uint dd = 0; dd < 4; dd++) {
                uint d = simd_lane_id * 4 + dd;
                o_local[r][dd] += float(O_partial[my_row * D + d]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;

        #pragma unroll
        for (uint dd = 0; dd < 4; dd++) {
            uint d = simd_lane_id * 4 + dd;
            O_head[global_q_idx * D + d] = half(o_local[r][dd] * inv_l);
        }

        if (simd_lane_id == 0) {
            L_head[global_q_idx] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

// =============================================================================
// FlashAttention Forward (D=128, non-causal)
// =============================================================================

kernel void mpp_flash_attention_fwd_d128(
    device half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    constant FlashAttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D = 128;
    const uint ROWS_PER_GROUP = 8;

    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;
    const uint q_start = q_block_idx * Bq;

    const uint q_batch_stride = params.num_heads * params.query_seq_len * D;
    const uint q_head_stride = params.query_seq_len * D;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D;
    const uint kv_head_stride = params.kv_seq_len * D;

    device half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half V_tile[Bk * D];
    threadgroup float S_tile[Bq * Bk];
    threadgroup half P_half[Bq * Bk];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;
    const uint linear_tid = tid.y * SIMD_SIZE + tid.x;

    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][4];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = o_local[r][2] = o_local[r][3] = 0.0f;
    }

    for (uint i = linear_tid; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len)
                   ? Q_head[global_q_row * D + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_kv_blocks = (params.kv_seq_len + Bk - 1) / Bk;

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, true, false
    );
    mpp::tensor_ops::matmul2d<qk_desc, execution_simdgroup> qk_op;

    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, false, false
    );
    mpp::tensor_ops::matmul2d<pv_desc, execution_simdgroup> pv_op;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        for (uint i = linear_tid; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual) ? K_head[global_row * D + col] : half(0.0f);
            V_tile[i] = (global_row < k_end_actual) ? V_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QK matmul
        auto tQ = tensor((threadgroup half*)Q_tile,
                         dextents<int, 2>{(int)D, (int)Bq}, array<int, 2>{1, (int)D});
        auto tK = tensor((threadgroup half*)K_tile,
                         dextents<int, 2>{(int)D, (int)Bk}, array<int, 2>{1, (int)D});
        auto tS = tensor((threadgroup float*)S_tile,
                         dextents<int, 2>{(int)Bk, (int)Bq}, array<int, 2>{1, (int)Bk});
        qk_op.run(tQ, tK, tS);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax → write P to P_half
        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) {
                for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                    P_half[my_row * Bk + k] = half(0.0f);
                }
                continue;
            }

            float row_max = -INFINITY;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                uint global_k_pos = k_start + k;
                float s = S_tile[my_row * Bk + k] * params.scale;
                if (global_k_pos >= k_end_actual) s = -INFINITY;
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) s = -INFINITY;
                if (params.softcap > 0.0f && s > -INFINITY) s = params.softcap * tanh(s / params.softcap);
                S_tile[my_row * Bk + k] = s;
                row_max = max(row_max, s);
            }
            row_max = simd_max_f32(row_max);

            float m_new = max(m_i[r], row_max);
            float correction = metal::exp(m_i[r] - m_new);

            float row_sum = 0.0f;
            for (uint k = simd_lane_id; k < Bk; k += SIMD_SIZE) {
                float s = S_tile[my_row * Bk + k];
                float p = (s > -INFINITY) ? metal::exp(s - m_new) : 0.0f;
                P_half[my_row * Bk + k] = half(p);
                row_sum += p;
            }
            row_sum = simd_sum_f32(row_sum);

            l_i[r] = correction * l_i[r] + row_sum;

            #pragma unroll
            for (uint dd = 0; dd < 4; dd++) {
                o_local[r][dd] *= correction;
            }

            m_i[r] = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PV matmul: O_partial = P @ V in 4 chunks of 32
        threadgroup half* O_partial = (threadgroup half*)K_tile;

        auto tP = tensor((threadgroup half*)P_half,
                         dextents<int, 2>{(int)Bk, (int)Bq}, array<int, 2>{1, (int)Bk});
        auto tV_full = tensor((threadgroup half*)V_tile,
                              dextents<int, 2>{(int)D, (int)Bk}, array<int, 2>{1, (int)D});
        auto tO_full = tensor(O_partial,
                              dextents<int, 2>{(int)D, (int)Bq}, array<int, 2>{1, (int)D});

        #pragma unroll
        for (uint d_start = 0; d_start < D; d_start += 32) {
            auto tV_chunk = tV_full.slice((int)d_start, 0);
            auto tO_chunk = tO_full.slice((int)d_start, 0);
            pv_op.run(tP, tV_chunk, tO_chunk);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate O_partial into registers
        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            if (my_row >= Bq) continue;

            #pragma unroll
            for (uint dd = 0; dd < 4; dd++) {
                uint d = simd_lane_id * 4 + dd;
                o_local[r][dd] += float(O_partial[my_row * D + d]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;
        #pragma unroll
        for (uint dd = 0; dd < 4; dd++) {
            uint d = simd_lane_id * 4 + dd;
            O_head[global_q_idx * D + d] = half(o_local[r][dd] * inv_l);
        }
        if (simd_lane_id == 0) {
            L_head[global_q_idx] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}
