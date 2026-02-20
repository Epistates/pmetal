//  flash_attention.metal
//  FlashAttention implementation for Apple Silicon
//
//  Based on FlashAttention-2 algorithm with optimizations for Metal:
//  - Online softmax for numerical stability
//  - Block-wise computation for O(n) memory
//  - GQA/MQA support with efficient head repetition
//  - Causal masking with fused computation
//
//  References:
//  - FlashAttention-2: https://arxiv.org/abs/2307.08691
//  - Metal FlashAttention: https://github.com/philipturner/metal-flash-attention

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Configuration and Types
// =============================================================================

/// Kernel parameters passed from host.
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

/// Block dimensions for tiling.
/// These are compile-time constants for different head dimensions.
constant uint BLOCK_Q [[function_constant(0)]];
constant uint BLOCK_K [[function_constant(1)]];
constant uint HEAD_DIM [[function_constant(2)]];
constant bool IS_CAUSAL [[function_constant(3)]];

// SIMD group size on Apple GPUs
constant uint SIMD_SIZE = 32;

// =============================================================================
// Utility Functions
// =============================================================================

/// Fast exponential approximation (for performance-critical paths).
inline float fast_exp(float x) {
    // Use metal's native exp which is already optimized
    return metal::exp(x);
}

/// Warp-level max reduction.
inline float simd_max_f32(float val) {
    #pragma unroll
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, simd_shuffle_xor(val, offset));
    }
    return val;
}

/// Warp-level sum reduction.
inline float simd_sum_f32(float val) {
    #pragma unroll
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        val += simd_shuffle_xor(val, offset);
    }
    return val;
}

// =============================================================================
// FlashAttention Forward Kernel (D=128)
// =============================================================================

/// FlashAttention forward pass with online softmax.
///
/// Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
///
/// Uses block-wise computation to achieve O(n) memory complexity.
/// Per-element online softmax for correct numerical stability.
///
/// Thread organization:
/// - Grid: [num_q_blocks, num_heads, batch_size]
/// - Threadgroup: [32, 4, 1] = 128 threads (4 SIMD groups of 32)
/// - Each SIMD group processes 8 query rows (32 rows / 4 groups)
/// - Each lane handles 4 consecutive D-elements (128 / 32 = 4)
/// - Score computation uses SIMD-parallel dot products
kernel void flash_attention_forward_d128_causal(
    device const half* Q [[buffer(0)]],           // [B, H, N, D]
    device const half* K [[buffer(1)]],           // [B, H_kv, N, D]
    device const half* V [[buffer(2)]],           // [B, H_kv, N, D]
    device half* O [[buffer(3)]],                 // [B, H, N, D]
    device float* L [[buffer(4)]],                // [B, H, N] logsumexp
    constant FlashAttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D = 128;
    const uint ROWS_PER_GROUP = 8;  // Bq / 4 SIMD groups

    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;
    const uint q_start = q_block_idx * Bq;

    // Strides
    const uint q_batch_stride = params.num_heads * params.query_seq_len * D;
    const uint q_head_stride = params.query_seq_len * D;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D;
    const uint kv_head_stride = params.kv_seq_len * D;

    // Head pointers
    device const half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device const half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device const half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    // Shared memory: Q, K, V tiles only (no S_tile needed with online softmax)
    // Total: 3 * 32 * 128 * 2 = 24576 bytes (under 32KB)
    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half V_tile[Bk * D];

    // Each SIMD group handles ROWS_PER_GROUP consecutive query rows
    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;

    // Per-row accumulators: 8 rows * 4 D-elements per lane = 32 floats
    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][4];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = o_local[r][2] = o_local[r][3] = 0.0f;
    }

    // Collaborative load of Q tile (all 128 threads participate)
    for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len)
                   ? Q_head[global_q_row * D + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Causal: only process KV positions up to the end of this Q block
    uint kv_end = IS_CAUSAL ? min(q_start + Bq, params.kv_seq_len) : params.kv_seq_len;
    uint num_kv_blocks = (kv_end + Bk - 1) / Bk;

    // Main loop over KV blocks
    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        // Collaborative load of K and V tiles
        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual) ? K_head[global_row * D + col] : half(0.0f);
        }
        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            V_tile[i] = (global_row < k_end_actual) ? V_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process each query row assigned to this SIMD group
        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) continue;

            // Process each key with element-wise online softmax
            for (uint k = 0; k < Bk && k_start + k < k_end_actual; k++) {
                // SIMD-parallel dot product: each lane computes 4 multiply-adds
                float partial = 0.0f;
                #pragma unroll
                for (uint dd = 0; dd < 4; dd++) {
                    uint d = simd_lane_id * 4 + dd;
                    partial += float(Q_tile[my_row * D + d]) * float(K_tile[k * D + d]);
                }
                float score = simd_sum_f32(partial) * params.scale;

                // Causal mask
                uint global_k_pos = k_start + k;
                if (IS_CAUSAL && global_k_pos > global_q_pos) {
                    score = -INFINITY;
                }
                // Sliding window mask
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) {
                    score = -INFINITY;
                }
                // Softcap
                if (params.softcap > 0.0f && score > -INFINITY) {
                    score = params.softcap * tanh(score / params.softcap);
                }

                // Online softmax: update max, correct previous accumulations, add new
                float m_new = max(m_i[r], score);
                float correction = fast_exp(m_i[r] - m_new);
                float p = fast_exp(score - m_new);

                l_i[r] = correction * l_i[r] + p;

                #pragma unroll
                for (uint dd = 0; dd < 4; dd++) {
                    uint d = simd_lane_id * 4 + dd;
                    o_local[r][dd] = correction * o_local[r][dd]
                                   + p * float(V_tile[k * D + d]);
                }
                m_i[r] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize output and write results
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;

        // Each lane writes its 4 consecutive D-elements
        #pragma unroll
        for (uint dd = 0; dd < 4; dd++) {
            uint d = simd_lane_id * 4 + dd;
            O_head[global_q_idx * D + d] = half(o_local[r][dd] * inv_l);
        }

        // Lane 0 writes logsumexp
        if (simd_lane_id == 0) {
            L_head[global_q_idx] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

/// FlashAttention forward pass without causal masking.
/// Same algorithm as causal variant but processes all KV positions.
kernel void flash_attention_forward_d128(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
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

    device const half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device const half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device const half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half V_tile[Bk * D];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;

    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][4];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = o_local[r][2] = o_local[r][3] = 0.0f;
    }

    for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len)
                   ? Q_head[global_q_row * D + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Non-causal: process all KV positions
    uint num_kv_blocks = (params.kv_seq_len + Bk - 1) / Bk;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual) ? K_head[global_row * D + col] : half(0.0f);
        }
        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            V_tile[i] = (global_row < k_end_actual) ? V_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) continue;

            for (uint k = 0; k < Bk && k_start + k < k_end_actual; k++) {
                float partial = 0.0f;
                #pragma unroll
                for (uint dd = 0; dd < 4; dd++) {
                    uint d = simd_lane_id * 4 + dd;
                    partial += float(Q_tile[my_row * D + d]) * float(K_tile[k * D + d]);
                }
                float score = simd_sum_f32(partial) * params.scale;

                // Sliding window (no causal mask)
                uint global_k_pos = k_start + k;
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window) {
                    score = -INFINITY;
                }
                if (params.softcap > 0.0f && score > -INFINITY) {
                    score = params.softcap * tanh(score / params.softcap);
                }

                float m_new = max(m_i[r], score);
                float correction = fast_exp(m_i[r] - m_new);
                float p = fast_exp(score - m_new);
                l_i[r] = correction * l_i[r] + p;

                #pragma unroll
                for (uint dd = 0; dd < 4; dd++) {
                    uint d = simd_lane_id * 4 + dd;
                    o_local[r][dd] = correction * o_local[r][dd]
                                   + p * float(V_tile[k * D + d]);
                }
                m_i[r] = m_new;
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
// FlashAttention Backward dQ Kernel
// =============================================================================

/// Backward pass computing dQ gradient.
///
/// dQ = scale * sum_k[ P_ij * (dP_ij - D_i) * K_j ]
/// where D_i = rowsum(dO * O)
///
/// Threadgroup memory: Q_tile + K_tile + dO_tile = 3 * 8KB = 24KB (under 32KB)
/// V, O, L read from device memory to avoid exceeding threadgroup limit.
kernel void flash_attention_backward_dq_d128_causal(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const half* O [[buffer(3)]],
    device const half* dO [[buffer(4)]],
    device const float* L [[buffer(5)]],
    device half* dQ [[buffer(6)]],
    constant FlashAttentionParams& params [[buffer(7)]],
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

    device const half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device const half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device const half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device const half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device const half* dO_head = dO + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device const float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                                   + head_idx * params.query_seq_len;
    device half* dQ_head = dQ + batch_idx * q_batch_stride + head_idx * q_head_stride;

    // Shared memory: 3 tiles = 24576 bytes (under 32KB limit)
    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half dO_tile[Bq * D];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;

    // Per-row accumulators
    float dq_local[ROWS_PER_GROUP][4];
    float d_i[ROWS_PER_GROUP];     // D_i = rowsum(dO * O) per row
    float l_val[ROWS_PER_GROUP];   // Logsumexp per row

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        dq_local[r][0] = dq_local[r][1] = dq_local[r][2] = dq_local[r][3] = 0.0f;
        d_i[r] = 0.0f;
        l_val[r] = 0.0f;
    }

    // Load Q and dO tiles
    for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bq * D; i += 128) {
        uint row = i / D;
        uint col = i % D;
        uint global_row = q_start + row;
        if (global_row < params.query_seq_len) {
            Q_tile[i] = Q_head[global_row * D + col];
            dO_tile[i] = dO_head[global_row * D + col];
        } else {
            Q_tile[i] = half(0.0f);
            dO_tile[i] = half(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute D_i = rowsum(dO * O) from dO_tile and device O, and load L
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_row = q_start + my_row;
        if (my_row >= Bq || global_q_row >= params.query_seq_len) continue;

        // SIMD-parallel D_i computation
        float partial_di = 0.0f;
        #pragma unroll
        for (uint dd = 0; dd < 4; dd++) {
            uint d = simd_lane_id * 4 + dd;
            partial_di += float(dO_tile[my_row * D + d])
                        * float(O_head[global_q_row * D + d]);
        }
        d_i[r] = simd_sum_f32(partial_di);

        // Load logsumexp from device memory
        if (simd_lane_id == 0) {
            l_val[r] = L_head[global_q_row];
        }
        l_val[r] = simd_shuffle(l_val[r], 0);
    }

    // Causal KV range
    uint kv_end = min(q_start + Bq, params.kv_seq_len);
    uint num_kv_blocks = (kv_end + Bk - 1) / Bk;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;

        // Load K tile
        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < params.kv_seq_len)
                       ? K_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) continue;

            for (uint k_idx = 0; k_idx < Bk; k_idx++) {
                uint global_k_pos = k_start + k_idx;
                if (global_k_pos > global_q_pos || global_k_pos >= params.kv_seq_len) continue;

                // SIMD-parallel score: Q[my_row] · K[k_idx]
                float partial_s = 0.0f;
                #pragma unroll
                for (uint dd = 0; dd < 4; dd++) {
                    uint d = simd_lane_id * 4 + dd;
                    partial_s += float(Q_tile[my_row * D + d]) * float(K_tile[k_idx * D + d]);
                }
                float s = simd_sum_f32(partial_s) * params.scale;
                float p = fast_exp(s - l_val[r]);

                // SIMD-parallel dov: dO[my_row] · V[k_idx] (V from device memory)
                float partial_dov = 0.0f;
                #pragma unroll
                for (uint dd = 0; dd < 4; dd++) {
                    uint d = simd_lane_id * 4 + dd;
                    partial_dov += float(dO_tile[my_row * D + d])
                                 * float(V_head[(k_start + k_idx) * D + d]);
                }
                float dov = simd_sum_f32(partial_dov);

                // dS_ij = P_ij * (dP_ij - D_i)
                float ds = p * (dov - d_i[r]);

                // Accumulate dQ
                #pragma unroll
                for (uint dd = 0; dd < 4; dd++) {
                    uint d = simd_lane_id * 4 + dd;
                    dq_local[r][dd] += ds * float(K_tile[k_idx * D + d]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write dQ (scale applied once here)
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        #pragma unroll
        for (uint dd = 0; dd < 4; dd++) {
            uint d = simd_lane_id * 4 + dd;
            dQ_head[global_q_idx * D + d] = half(dq_local[r][dd] * params.scale);
        }
    }
}

// =============================================================================
// FlashAttention Backward dK/dV Kernel
// =============================================================================

/// Backward pass computing dK and dV gradients.
///
/// Threadgroup memory: K_tile + V_tile + Q_tile + dO_tile = 4 * 8KB = 32KB (at limit)
/// O and L read from device memory. D_i computed on the fly via SIMD reduction.
kernel void flash_attention_backward_dkv_d128_causal(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const half* O [[buffer(3)]],
    device const half* dO [[buffer(4)]],
    device const float* L [[buffer(5)]],
    device half* dK [[buffer(6)]],
    device half* dV [[buffer(7)]],
    constant FlashAttentionParams& params [[buffer(8)]],
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
    const uint kv_head_idx = tgid.y;
    const uint kv_block_idx = tgid.x;
    const uint k_start = kv_block_idx * Bk;

    const uint q_batch_stride = params.num_heads * params.query_seq_len * D;
    const uint q_head_stride = params.query_seq_len * D;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D;
    const uint kv_head_stride = params.kv_seq_len * D;

    device half* dK_head = dK + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* dV_head = dV + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device const half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device const half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;

    // Shared memory: exactly 32768 bytes
    threadgroup half K_tile[Bk * D];   // 8192
    threadgroup half V_tile[Bk * D];   // 8192
    threadgroup half Q_tile[Bq * D];   // 8192
    threadgroup half dO_tile[Bq * D];  // 8192

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;

    // Per-row accumulators for dK, dV
    float dk_local[ROWS_PER_GROUP][4];
    float dv_local[ROWS_PER_GROUP][4];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        dk_local[r][0] = dk_local[r][1] = dk_local[r][2] = dk_local[r][3] = 0.0f;
        dv_local[r][0] = dv_local[r][1] = dv_local[r][2] = dv_local[r][3] = 0.0f;
    }

    // Load K, V tiles for this KV block
    for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
        uint row = i / D;
        uint col = i % D;
        uint global_row = k_start + row;
        if (global_row < params.kv_seq_len) {
            K_tile[i] = K_head[global_row * D + col];
            V_tile[i] = V_head[global_row * D + col];
        } else {
            K_tile[i] = half(0.0f);
            V_tile[i] = half(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over all query heads mapping to this KV head (GQA)
    for (uint q_head_offset = 0; q_head_offset < params.gqa_ratio; q_head_offset++) {
        uint head_idx = kv_head_idx * params.gqa_ratio + q_head_offset;

        device const half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
        device const half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
        device const half* dO_head = dO + batch_idx * q_batch_stride + head_idx * q_head_stride;
        device const float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                                       + head_idx * params.query_seq_len;

        // Causal: only queries at positions >= k_start can attend to this KV block
        uint q_start_min = k_start;
        uint num_q_blocks = (params.query_seq_len > q_start_min)
                          ? (params.query_seq_len - q_start_min + Bq - 1) / Bq : 0;

        for (uint q_block = 0; q_block < num_q_blocks; q_block++) {
            uint q_start = q_start_min + q_block * Bq;

            // Load Q and dO tiles (reuse Q_tile and dO_tile memory)
            for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bq * D; i += 128) {
                uint row = i / D;
                uint col = i % D;
                uint global_row = q_start + row;
                if (global_row < params.query_seq_len) {
                    Q_tile[i] = Q_head[global_row * D + col];
                    dO_tile[i] = dO_head[global_row * D + col];
                } else {
                    Q_tile[i] = half(0.0f);
                    dO_tile[i] = half(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Process each KV row assigned to this SIMD group
            for (uint r = 0; r < ROWS_PER_GROUP; r++) {
                uint my_kv_row = group_row_start + r;
                uint global_k_pos = k_start + my_kv_row;
                if (my_kv_row >= Bk || global_k_pos >= params.kv_seq_len) continue;

                for (uint q_idx = 0; q_idx < Bq; q_idx++) {
                    uint global_q_pos = q_start + q_idx;
                    if (global_q_pos >= params.query_seq_len || global_k_pos > global_q_pos) continue;

                    // Read logsumexp from device memory
                    float l_q;
                    if (simd_lane_id == 0) {
                        l_q = L_head[global_q_pos];
                    }
                    l_q = simd_shuffle(l_q, 0);

                    // SIMD-parallel score: Q[q_idx] · K[my_kv_row]
                    float partial_s = 0.0f;
                    #pragma unroll
                    for (uint dd = 0; dd < 4; dd++) {
                        uint d = simd_lane_id * 4 + dd;
                        partial_s += float(Q_tile[q_idx * D + d])
                                   * float(K_tile[my_kv_row * D + d]);
                    }
                    float s = simd_sum_f32(partial_s) * params.scale;
                    float p = fast_exp(s - l_q);

                    // dV += P^T @ dO
                    #pragma unroll
                    for (uint dd = 0; dd < 4; dd++) {
                        uint d = simd_lane_id * 4 + dd;
                        dv_local[r][dd] += p * float(dO_tile[q_idx * D + d]);
                    }

                    // Compute D_i from device O and threadgroup dO via SIMD reduction
                    float partial_di = 0.0f;
                    #pragma unroll
                    for (uint dd = 0; dd < 4; dd++) {
                        uint d = simd_lane_id * 4 + dd;
                        partial_di += float(dO_tile[q_idx * D + d])
                                    * float(O_head[global_q_pos * D + d]);
                    }
                    float d_i = simd_sum_f32(partial_di);

                    // dov = dO · V (SIMD-parallel)
                    float partial_dov = 0.0f;
                    #pragma unroll
                    for (uint dd = 0; dd < 4; dd++) {
                        uint d = simd_lane_id * 4 + dd;
                        partial_dov += float(dO_tile[q_idx * D + d])
                                     * float(V_tile[my_kv_row * D + d]);
                    }
                    float dov = simd_sum_f32(partial_dov);

                    float ds = p * (dov - d_i);

                    // dK += dS^T @ Q
                    #pragma unroll
                    for (uint dd = 0; dd < 4; dd++) {
                        uint d = simd_lane_id * 4 + dd;
                        dk_local[r][dd] += ds * float(Q_tile[q_idx * D + d]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write dK, dV (scale applied to dK only)
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_kv_row = group_row_start + r;
        uint global_k_idx = k_start + my_kv_row;
        if (my_kv_row >= Bk || global_k_idx >= params.kv_seq_len) continue;

        #pragma unroll
        for (uint dd = 0; dd < 4; dd++) {
            uint d = simd_lane_id * 4 + dd;
            dK_head[global_k_idx * D + d] = half(dk_local[r][dd] * params.scale);
            dV_head[global_k_idx * D + d] = half(dv_local[r][dd]);
        }
    }
}

// =============================================================================
// Additional Head Dimension Variants
// =============================================================================

// D=64 variant: each lane handles 2 D-elements (64/32), 8 rows per group
kernel void flash_attention_forward_d64_causal(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
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

    device const half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device const half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device const half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                             + head_idx * params.query_seq_len;

    // 3 tiles * 32 * 64 * 2 = 12288 bytes
    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half V_tile[Bk * D];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;

    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][2];  // 2 elements per lane for D=64

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = 0.0f;
    }

    for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        uint global_q_row = q_start + q_row;
        Q_tile[i] = (global_q_row < params.query_seq_len)
                   ? Q_head[global_q_row * D + q_col] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_end = IS_CAUSAL ? min(q_start + Bq, params.kv_seq_len) : params.kv_seq_len;
    uint num_kv_blocks = (kv_end + Bk - 1) / Bk;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end_actual = min(k_start + Bk, params.kv_seq_len);

        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < k_end_actual) ? K_head[global_row * D + col] : half(0.0f);
        }
        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            V_tile[i] = (global_row < k_end_actual) ? V_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) continue;

            for (uint k = 0; k < Bk && k_start + k < k_end_actual; k++) {
                // SIMD-parallel dot: each lane does 2 mults for D=64
                float partial = 0.0f;
                #pragma unroll
                for (uint dd = 0; dd < 2; dd++) {
                    uint d = simd_lane_id * 2 + dd;
                    partial += float(Q_tile[my_row * D + d]) * float(K_tile[k * D + d]);
                }
                float score = simd_sum_f32(partial) * params.scale;

                uint global_k_pos = k_start + k;
                if (IS_CAUSAL && global_k_pos > global_q_pos) score = -INFINITY;
                if (params.sliding_window > 0 && global_q_pos > global_k_pos + params.sliding_window)
                    score = -INFINITY;
                if (params.softcap > 0.0f && score > -INFINITY)
                    score = params.softcap * tanh(score / params.softcap);

                float m_new = max(m_i[r], score);
                float correction = fast_exp(m_i[r] - m_new);
                float p = fast_exp(score - m_new);
                l_i[r] = correction * l_i[r] + p;

                #pragma unroll
                for (uint dd = 0; dd < 2; dd++) {
                    uint d = simd_lane_id * 2 + dd;
                    o_local[r][dd] = correction * o_local[r][dd]
                                   + p * float(V_tile[k * D + d]);
                }
                m_i[r] = m_new;
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

kernel void flash_attention_backward_dq_d64_causal(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const half* O [[buffer(3)]],
    device const half* dO [[buffer(4)]],
    device const float* L [[buffer(5)]],
    device half* dQ [[buffer(6)]],
    constant FlashAttentionParams& params [[buffer(7)]],
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

    device const half* Q_head = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device const half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device const half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device const half* O_head = O + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device const half* dO_head = dO + batch_idx * q_batch_stride + head_idx * q_head_stride;
    device const float* L_head = L + batch_idx * params.num_heads * params.query_seq_len
                                   + head_idx * params.query_seq_len;
    device half* dQ_head = dQ + batch_idx * q_batch_stride + head_idx * q_head_stride;

    // Q_tile + K_tile + dO_tile = 3 * 4KB = 12KB
    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half dO_tile[Bq * D];

    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;

    float dq_local[ROWS_PER_GROUP][2];
    float d_i[ROWS_PER_GROUP];
    float l_val[ROWS_PER_GROUP];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        dq_local[r][0] = dq_local[r][1] = 0.0f;
        d_i[r] = 0.0f;
        l_val[r] = 0.0f;
    }

    // Load Q and dO tiles
    for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bq * D; i += 128) {
        uint row = i / D;
        uint col = i % D;
        uint global_row = q_start + row;
        if (global_row < params.query_seq_len) {
            Q_tile[i] = Q_head[global_row * D + col];
            dO_tile[i] = dO_head[global_row * D + col];
        } else {
            Q_tile[i] = half(0.0f);
            dO_tile[i] = half(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute D_i and load L (per-group, from device O)
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_row = q_start + my_row;
        if (my_row >= Bq || global_q_row >= params.query_seq_len) continue;

        float partial_di = 0.0f;
        #pragma unroll
        for (uint dd = 0; dd < 2; dd++) {
            uint d = simd_lane_id * 2 + dd;
            partial_di += float(dO_tile[my_row * D + d])
                        * float(O_head[global_q_row * D + d]);
        }
        d_i[r] = simd_sum_f32(partial_di);

        if (simd_lane_id == 0) l_val[r] = L_head[global_q_row];
        l_val[r] = simd_shuffle(l_val[r], 0);
    }

    uint kv_end = min(q_start + Bq, params.kv_seq_len);
    uint num_kv_blocks = (kv_end + Bk - 1) / Bk;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;

        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint row = i / D;
            uint col = i % D;
            uint global_row = k_start + row;
            K_tile[i] = (global_row < params.kv_seq_len)
                       ? K_head[global_row * D + col] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q_pos = q_start + my_row;
            if (my_row >= Bq || global_q_pos >= params.query_seq_len) continue;

            for (uint k_idx = 0; k_idx < Bk; k_idx++) {
                uint global_k_pos = k_start + k_idx;
                if (global_k_pos > global_q_pos || global_k_pos >= params.kv_seq_len) continue;

                float partial_s = 0.0f;
                #pragma unroll
                for (uint dd = 0; dd < 2; dd++) {
                    uint d = simd_lane_id * 2 + dd;
                    partial_s += float(Q_tile[my_row * D + d]) * float(K_tile[k_idx * D + d]);
                }
                float s = simd_sum_f32(partial_s) * params.scale;
                float p = fast_exp(s - l_val[r]);

                float partial_dov = 0.0f;
                #pragma unroll
                for (uint dd = 0; dd < 2; dd++) {
                    uint d = simd_lane_id * 2 + dd;
                    partial_dov += float(dO_tile[my_row * D + d])
                                 * float(V_head[(k_start + k_idx) * D + d]);
                }
                float dov = simd_sum_f32(partial_dov);

                float ds = p * (dov - d_i[r]);

                #pragma unroll
                for (uint dd = 0; dd < 2; dd++) {
                    uint d = simd_lane_id * 2 + dd;
                    dq_local[r][dd] += ds * float(K_tile[k_idx * D + d]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= params.query_seq_len) continue;

        #pragma unroll
        for (uint dd = 0; dd < 2; dd++) {
            uint d = simd_lane_id * 2 + dd;
            dQ_head[global_q_idx * D + d] = half(dq_local[r][dd] * params.scale);
        }
    }
}

// =============================================================================
// Variable-Length Sequence Support (Packed Sequences)
// =============================================================================

/// Parameters for variable-length attention (packed sequences)
struct FlashAttentionVarlenParams {
    uint total_tokens;       // Total tokens in packed batch
    uint num_heads;          // Number of query heads
    uint num_kv_heads;       // Number of KV heads
    uint head_dim;           // Head dimension
    uint num_seqs;           // Number of sequences in pack
    float scale;             // Softmax scaling
    uint gqa_ratio;          // GQA ratio (num_heads / num_kv_heads)
    uint max_seqlen;         // Maximum sequence length in batch
    uint is_causal;          // Whether to apply causal mask
    float softcap;           // Softcapping value (0.0 = disabled)
    uint sliding_window;     // Sliding window size (0 = disabled)
};

/// FlashAttention forward for variable-length sequences.
///
/// Each sequence in the packed batch is identified by cu_seqlens:
/// - cu_seqlens[i] = start position of sequence i
/// - cu_seqlens[i+1] - cu_seqlens[i] = length of sequence i
///
/// Block-diagonal attention: each sequence only attends to itself.
kernel void flash_attention_varlen_forward_d128(
    device const half* Q [[buffer(0)]],              // [total_tokens, num_heads, head_dim]
    device const half* K [[buffer(1)]],              // [total_tokens, num_kv_heads, head_dim]
    device const half* V [[buffer(2)]],              // [total_tokens, num_kv_heads, head_dim]
    device half* O [[buffer(3)]],                    // [total_tokens, num_heads, head_dim]
    device float* L [[buffer(4)]],                   // [total_tokens, num_heads] logsumexp
    device const int* cu_seqlens [[buffer(5)]],      // [num_seqs + 1] cumulative lengths
    constant FlashAttentionVarlenParams& params [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint D = 128;
    const uint Bq = 32;
    const uint Bk = 32;

    // Grid: [num_q_blocks_total, num_heads, 1]
    // We process query blocks across all sequences
    const uint head_idx = tgid.y;
    const uint q_block_global = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;

    // Find which sequence this query block belongs to
    // Binary search through cu_seqlens
    uint seq_idx = 0;
    uint q_block_offset = q_block_global;
    uint seq_start = 0;
    uint seq_len = 0;

    for (uint s = 0; s < params.num_seqs; s++) {
        uint start = cu_seqlens[s];
        uint end = cu_seqlens[s + 1];
        uint len = end - start;
        uint num_blocks = (len + Bq - 1) / Bq;

        if (q_block_offset < num_blocks) {
            seq_idx = s;
            seq_start = start;
            seq_len = len;
            break;
        }
        q_block_offset -= num_blocks;
    }

    // Bounds check
    if (seq_len == 0) return;

    const uint q_start = q_block_offset * Bq;
    const uint q_end = min(q_start + Bq, seq_len);

    // Strides for NHD layout
    const uint head_stride = D;
    const uint token_stride = params.num_heads * D;
    const uint kv_token_stride = params.num_kv_heads * D;

    // Pointers for this sequence
    device const half* Q_seq = Q + seq_start * token_stride + head_idx * head_stride;
    device const half* K_seq = K + seq_start * kv_token_stride + kv_head_idx * head_stride;
    device const half* V_seq = V + seq_start * kv_token_stride + kv_head_idx * head_stride;
    device half* O_seq = O + seq_start * token_stride + head_idx * head_stride;
    device float* L_seq = L + seq_start * params.num_heads + head_idx;

    // Threadgroup memory: Q_tile + K_tile + V_tile = 3 * 8KB = 24KB
    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half V_tile[Bk * D];

    // 8 rows per SIMD group (4 groups × 8 rows = 32 = Bq)
    const uint ROWS_PER_GROUP = 8;
    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;

    // Per-row accumulators
    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][4]; // 4 D-elements per lane (128/32)

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = o_local[r][2] = o_local[r][3] = 0.0f;
    }

    // Load Q tile (NHD layout, stride by num_heads * D between tokens)
    for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        if (q_start + q_row < seq_len) {
            Q_tile[i] = Q_seq[(q_start + q_row) * token_stride + q_col];
        } else {
            Q_tile[i] = half(0.0f);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Causal: only attend to keys <= query position within sequence
    uint kv_end = params.is_causal ? min(q_start + Bq, seq_len) : seq_len;
    uint num_kv_blocks = (kv_end + Bk - 1) / Bk;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end = min(k_start + Bk, seq_len);

        // Load K, V tiles (NHD layout)
        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint k_row = i / D;
            uint k_col = i % D;
            if (k_start + k_row < k_end) {
                K_tile[i] = K_seq[(k_start + k_row) * kv_token_stride + k_col];
                V_tile[i] = V_seq[(k_start + k_row) * kv_token_stride + k_col];
            } else {
                K_tile[i] = half(0.0f);
                V_tile[i] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row, per-key online softmax with SIMD-parallel dot products
        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q = q_start + my_row;
            if (my_row >= Bq || global_q >= seq_len) continue;

            for (uint k_idx = 0; k_idx < Bk; k_idx++) {
                uint global_k = k_start + k_idx;

                // Masking: sequence boundary + causal + sliding window
                bool masked = false;
                if (global_k >= seq_len) masked = true;
                if (params.is_causal && global_k > global_q) masked = true;
                if (params.sliding_window > 0 && global_q > global_k + params.sliding_window) masked = true;

                if (masked) continue;

                // SIMD-parallel Q·K dot product
                float partial = 0.0f;
                #pragma unroll
                for (uint dd = 0; dd < 4; dd++) {
                    uint d = simd_lane_id * 4 + dd;
                    partial += float(Q_tile[my_row * D + d]) * float(K_tile[k_idx * D + d]);
                }
                float score = simd_sum_f32(partial) * params.scale;

                // Softcapping
                if (params.softcap > 0.0f) {
                    score = params.softcap * tanh(score / params.softcap);
                }

                // Online softmax update (per-element, correct)
                float m_new = max(m_i[r], score);
                float correction = fast_exp(m_i[r] - m_new);
                float p = fast_exp(score - m_new);
                l_i[r] = correction * l_i[r] + p;

                #pragma unroll
                for (uint dd = 0; dd < 4; dd++) {
                    uint d = simd_lane_id * 4 + dd;
                    o_local[r][dd] = correction * o_local[r][dd]
                                     + p * float(V_tile[k_idx * D + d]);
                }
                m_i[r] = m_new;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output (NHD layout)
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;
        #pragma unroll
        for (uint dd = 0; dd < 4; dd++) {
            uint d = simd_lane_id * 4 + dd;
            O_seq[global_q_idx * token_stride + d] = half(o_local[r][dd] * inv_l);
        }
        if (simd_lane_id == 0) {
            L_seq[global_q_idx * params.num_heads] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

/// Variable-length forward for D=64
kernel void flash_attention_varlen_forward_d64(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    device float* L [[buffer(4)]],
    device const int* cu_seqlens [[buffer(5)]],
    constant FlashAttentionVarlenParams& params [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Same algorithm as D=128 but with D=64 and adjusted block sizes
    const uint D = 64;
    const uint Bq = 64;  // Larger block for smaller D
    const uint Bk = 64;

    const uint head_idx = tgid.y;
    const uint q_block_global = tgid.x;
    const uint kv_head_idx = head_idx / params.gqa_ratio;

    // Find sequence
    uint seq_start = 0;
    uint seq_len = 0;
    uint q_block_offset = q_block_global;

    for (uint s = 0; s < params.num_seqs; s++) {
        uint start = cu_seqlens[s];
        uint end = cu_seqlens[s + 1];
        uint len = end - start;
        uint num_blocks = (len + Bq - 1) / Bq;

        if (q_block_offset < num_blocks) {
            seq_start = start;
            seq_len = len;
            break;
        }
        q_block_offset -= num_blocks;
    }

    if (seq_len == 0) return;

    const uint q_start = q_block_offset * Bq;
    const uint head_stride = D;
    const uint token_stride = params.num_heads * D;
    const uint kv_token_stride = params.num_kv_heads * D;

    device const half* Q_seq = Q + seq_start * token_stride + head_idx * head_stride;
    device const half* K_seq = K + seq_start * kv_token_stride + kv_head_idx * head_stride;
    device const half* V_seq = V + seq_start * kv_token_stride + kv_head_idx * head_stride;
    device half* O_seq = O + seq_start * token_stride + head_idx * head_stride;
    device float* L_seq = L + seq_start * params.num_heads + head_idx;

    // Threadgroup memory: Q_tile + K_tile + V_tile = 3 * 8KB = 24KB
    threadgroup half Q_tile[Bq * D];
    threadgroup half K_tile[Bk * D];
    threadgroup half V_tile[Bk * D];

    // 16 rows per SIMD group (4 groups × 16 rows = 64 = Bq)
    const uint ROWS_PER_GROUP = 16;
    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;

    // Per-row accumulators
    float m_i[ROWS_PER_GROUP];
    float l_i[ROWS_PER_GROUP];
    float o_local[ROWS_PER_GROUP][2]; // 2 D-elements per lane (64/32)

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        o_local[r][0] = o_local[r][1] = 0.0f;
    }

    // Load Q tile (NHD layout)
    for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bq * D; i += 128) {
        uint q_row = i / D;
        uint q_col = i % D;
        if (q_start + q_row < seq_len) {
            Q_tile[i] = Q_seq[(q_start + q_row) * token_stride + q_col];
        } else {
            Q_tile[i] = half(0.0f);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_end = params.is_causal ? min(q_start + Bq, seq_len) : seq_len;
    uint num_kv_blocks = (kv_end + Bk - 1) / Bk;

    for (uint kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        uint k_start = kv_block * Bk;
        uint k_end = min(k_start + Bk, seq_len);

        // Load K, V tiles (NHD layout)
        for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
            uint k_row = i / D;
            uint k_col = i % D;
            if (k_start + k_row < k_end) {
                K_tile[i] = K_seq[(k_start + k_row) * kv_token_stride + k_col];
                V_tile[i] = V_seq[(k_start + k_row) * kv_token_stride + k_col];
            } else {
                K_tile[i] = half(0.0f);
                V_tile[i] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row, per-key online softmax with SIMD-parallel dot products
        for (uint r = 0; r < ROWS_PER_GROUP; r++) {
            uint my_row = group_row_start + r;
            uint global_q = q_start + my_row;
            if (my_row >= Bq || global_q >= seq_len) continue;

            for (uint k_idx = 0; k_idx < Bk && k_start + k_idx < k_end; k_idx++) {
                uint global_k = k_start + k_idx;
                if (params.is_causal && global_k > global_q) continue;

                // SIMD-parallel Q·K dot product
                float partial = 0.0f;
                #pragma unroll
                for (uint dd = 0; dd < 2; dd++) {
                    uint d = simd_lane_id * 2 + dd;
                    partial += float(Q_tile[my_row * D + d]) * float(K_tile[k_idx * D + d]);
                }
                float score = simd_sum_f32(partial) * params.scale;

                // Online softmax update
                float m_new = max(m_i[r], score);
                float correction = fast_exp(m_i[r] - m_new);
                float p = fast_exp(score - m_new);
                l_i[r] = correction * l_i[r] + p;

                #pragma unroll
                for (uint dd = 0; dd < 2; dd++) {
                    uint d = simd_lane_id * 2 + dd;
                    o_local[r][dd] = correction * o_local[r][dd]
                                     + p * float(V_tile[k_idx * D + d]);
                }
                m_i[r] = m_new;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output (NHD layout)
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        uint global_q_idx = q_start + my_row;
        if (my_row >= Bq || global_q_idx >= seq_len) continue;

        float inv_l = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;
        #pragma unroll
        for (uint dd = 0; dd < 2; dd++) {
            uint d = simd_lane_id * 2 + dd;
            O_seq[global_q_idx * token_stride + d] = half(o_local[r][dd] * inv_l);
        }
        if (simd_lane_id == 0) {
            L_seq[global_q_idx * params.num_heads] = m_i[r] + log(max(l_i[r], 1e-10f));
        }
    }
}

/// D=64 variant with O buffer for exact gradients
kernel void flash_attention_backward_dkv_d64_causal(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device const half* O [[buffer(3)]],          // Forward output for D_i computation
    device const half* dO [[buffer(4)]],
    device const float* L [[buffer(5)]],
    device half* dK [[buffer(6)]],
    device half* dV [[buffer(7)]],
    constant FlashAttentionParams& params [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Configuration for D=64 (reduced block sizes to fit threadgroup memory)
    const uint Bq = 32;
    const uint Bk = 32;
    const uint D = 64;
    const uint SIMD_SIZE = 32;

    // Grid position - parallelize over KV blocks
    const uint batch_idx = tgid.z;
    const uint kv_head_idx = tgid.y;
    const uint kv_block_idx = tgid.x;

    const uint k_start = kv_block_idx * Bk;

    // Strides
    const uint q_batch_stride = params.num_heads * params.query_seq_len * D;
    const uint q_head_stride = params.query_seq_len * D;
    const uint kv_batch_stride = params.num_kv_heads * params.kv_seq_len * D;
    const uint kv_head_stride = params.kv_seq_len * D;
    const uint l_batch_stride = params.num_heads * params.query_seq_len;
    const uint l_head_stride = params.query_seq_len;

    // Get pointers to this batch/head
    device const half* K_head = K + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device const half* V_head = V + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* dK_head = dK + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;
    device half* dV_head = dV + batch_idx * kv_batch_stride + kv_head_idx * kv_head_stride;

    // 8 rows per SIMD group (4 groups × 8 rows = 32 = Bk)
    const uint ROWS_PER_GROUP = 8;
    const uint group_row_start = simd_group_id * ROWS_PER_GROUP;

    // Threadgroup memory: K_tile + V_tile + Q_tile + dO_tile = 4 * 4KB = 16KB
    threadgroup half K_tile[Bk * D];     // 32 * 64 * 2 = 4KB
    threadgroup half V_tile[Bk * D];     // 4KB
    threadgroup half Q_tile[Bq * D];     // 4KB
    threadgroup half dO_tile[Bq * D];    // 4KB

    // Per-row gradient accumulators
    float dk_local[ROWS_PER_GROUP][2];
    float dv_local[ROWS_PER_GROUP][2];

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        dk_local[r][0] = dk_local[r][1] = 0.0f;
        dv_local[r][0] = dv_local[r][1] = 0.0f;
    }

    // Load K and V tiles for this block
    for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bk * D; i += 128) {
        uint row = i / D;
        uint col = i % D;
        uint global_row = k_start + row;
        if (global_row < params.kv_seq_len) {
            K_tile[i] = K_head[global_row * D + col];
            V_tile[i] = V_head[global_row * D + col];
        } else {
            K_tile[i] = half(0.0f);
            V_tile[i] = half(0.0f);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over Q heads that use this KV head (for GQA)
    for (uint head_offset = 0; head_offset < params.gqa_ratio; head_offset++) {
        uint q_head_idx = kv_head_idx * params.gqa_ratio + head_offset;
        if (q_head_idx >= params.num_heads) continue;

        device const half* Q_head = Q + batch_idx * q_batch_stride + q_head_idx * q_head_stride;
        device const half* O_head = O + batch_idx * q_batch_stride + q_head_idx * q_head_stride;
        device const half* dO_head = dO + batch_idx * q_batch_stride + q_head_idx * q_head_stride;
        device const float* L_head = L + batch_idx * l_batch_stride + q_head_idx * l_head_stride;

        uint num_q_blocks = (params.query_seq_len + Bq - 1) / Bq;
        for (uint q_block = 0; q_block < num_q_blocks; q_block++) {
            uint q_start = q_block * Bq;

            // Load Q and dO tiles
            for (uint i = tid.y * SIMD_SIZE + tid.x; i < Bq * D; i += 128) {
                uint row = i / D;
                uint col = i % D;
                uint global_row = q_start + row;
                if (global_row < params.query_seq_len) {
                    Q_tile[i] = Q_head[global_row * D + col];
                    dO_tile[i] = dO_head[global_row * D + col];
                } else {
                    Q_tile[i] = half(0.0f);
                    dO_tile[i] = half(0.0f);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute gradients for each KV row
            for (uint r = 0; r < ROWS_PER_GROUP; r++) {
                uint my_row = group_row_start + r;
                if (my_row >= Bk || k_start + my_row >= params.kv_seq_len) continue;
                uint global_k_pos = k_start + my_row;

                for (uint q_idx = 0; q_idx < Bq; q_idx++) {
                    uint global_q_pos = q_start + q_idx;
                    if (global_q_pos >= params.query_seq_len) continue;
                    if (global_k_pos > global_q_pos) continue; // causal mask

                    // Compute D_i = dot(dO_i, O_i) via SIMD reduction
                    float partial_di = 0.0f;
                    #pragma unroll
                    for (uint dd = 0; dd < 2; dd++) {
                        uint d = simd_lane_id * 2 + dd;
                        partial_di += float(dO_tile[q_idx * D + d])
                                    * float(O_head[global_q_pos * D + d]);
                    }
                    float d_i = simd_sum_f32(partial_di);

                    // Read L from device, broadcast via SIMD shuffle
                    float l_val = 0.0f;
                    if (simd_lane_id == 0) l_val = L_head[global_q_pos];
                    l_val = simd_shuffle(l_val, 0);

                    // SIMD-parallel attention score recomputation
                    float partial_s = 0.0f;
                    #pragma unroll
                    for (uint dd = 0; dd < 2; dd++) {
                        uint d = simd_lane_id * 2 + dd;
                        partial_s += float(Q_tile[q_idx * D + d]) * float(K_tile[my_row * D + d]);
                    }
                    float s = simd_sum_f32(partial_s) * params.scale;
                    float p = fast_exp(s - l_val);

                    // Compute dov = dot(dO_i, V_j) via SIMD reduction
                    float partial_dov = 0.0f;
                    #pragma unroll
                    for (uint dd = 0; dd < 2; dd++) {
                        uint d = simd_lane_id * 2 + dd;
                        partial_dov += float(dO_tile[q_idx * D + d])
                                     * float(V_tile[my_row * D + d]);
                    }
                    float dov = simd_sum_f32(partial_dov);

                    float ds = p * (dov - d_i);

                    // Accumulate dK and dV
                    #pragma unroll
                    for (uint dd = 0; dd < 2; dd++) {
                        uint d = simd_lane_id * 2 + dd;
                        dv_local[r][dd] += p * float(dO_tile[q_idx * D + d]);
                        dk_local[r][dd] += ds * float(Q_tile[q_idx * D + d]) * params.scale;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write dK and dV output
    for (uint r = 0; r < ROWS_PER_GROUP; r++) {
        uint my_row = group_row_start + r;
        if (my_row >= Bk || k_start + my_row >= params.kv_seq_len) continue;
        uint global_k_idx = k_start + my_row;

        #pragma unroll
        for (uint dd = 0; dd < 2; dd++) {
            uint d = simd_lane_id * 2 + dd;
            dK_head[global_k_idx * D + d] = half(dk_local[r][dd]);
            dV_head[global_k_idx * D + d] = half(dv_local[r][dd]);
        }
    }
}
