// TurboQuant long-context q8 attention kernels for D=256 key/value dim.
// Includes all 2-pass variants (pass 1 emits partial accumulators +
// log-sum-exp stats; pass 2 merges them into the rotated output) plus
// the shared pass-2 merge kernel used by every D256 family.

#include "bridge_turboquant_internal.h"

static const char* TURBOQUANT_ATTENTION_Q8_D256_2PASS_1_SOURCE = R"(
    constexpr uint kDim = 256u;
    constexpr uint kVec = 8u;
    constexpr uint kQjlWords = 8u;
    constexpr uint kKeyCentroids = 128u;
    constexpr uint kValueCentroids = 256u;
    constexpr float kQjlConst = 1.2533141373155003f / 256.0f;
    threadgroup float shared_k_codebook[kKeyCentroids];
    threadgroup float shared_v_codebook[kValueCentroids];

    uint kv_head = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint block = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint groups = q_heads / kv_heads;
    uint row = batch * q_heads + kv_head * groups + simd_gid;
    if (row >= n_rows || block >= blocks) return;

    if (simd_gid == 0u) {
        for (uint c = lane; c < kKeyCentroids; c += 32u) {
            shared_k_codebook[c] = key_codebook[c];
        }
        for (uint c = lane; c < kValueCentroids; c += 32u) {
            shared_v_codebook[c] = value_codebook[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_row = batch * kv_heads + kv_head;
    uint d0 = lane * kVec;
    uint query_base = row * kDim + d0;

    float qrot0 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 0u];
    float qrot1 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 1u];
    float qrot2 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 2u];
    float qrot3 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 3u];
    float qrot4 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 4u];
    float qrot5 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 5u];
    float qrot6 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 6u];
    float qrot7 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 7u];
    float qproj0 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 0u];
    float qproj1 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 1u];
    float qproj2 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 2u];
    float qproj3 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 3u];
    float qproj4 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 4u];
    float qproj5 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 5u];
    float qproj6 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 6u];
    float qproj7 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 7u];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint seq = block; seq < n_seq; seq += blocks) {
        uint scalar_idx = kv_row * cache_seq_capacity + seq;
        float key_norm = key_norms[scalar_idx];
        float residual_scale = key_residual_norms[scalar_idx] * kQjlConst;
        float slot_scale = key_slot_scale[scalar_idx];
        uint key_base = (kv_row * kDim + d0) * cache_seq_capacity + seq;
        uint value_base = (kv_row * kDim + d0) * cache_seq_capacity + seq;
        uint sign_word = key_qjl_signs[(kv_row * kQjlWords + (lane >> 2u)) * cache_seq_capacity + seq];
        uint bit_base = (lane & 3u) * 8u;

        float score_part = 0.0f;
        score_part += qrot0 * shared_k_codebook[(uint)key_indices[key_base + 0u * cache_seq_capacity]];
        score_part += qrot1 * shared_k_codebook[(uint)key_indices[key_base + 1u * cache_seq_capacity]];
        score_part += qrot2 * shared_k_codebook[(uint)key_indices[key_base + 2u * cache_seq_capacity]];
        score_part += qrot3 * shared_k_codebook[(uint)key_indices[key_base + 3u * cache_seq_capacity]];
        score_part += qrot4 * shared_k_codebook[(uint)key_indices[key_base + 4u * cache_seq_capacity]];
        score_part += qrot5 * shared_k_codebook[(uint)key_indices[key_base + 5u * cache_seq_capacity]];
        score_part += qrot6 * shared_k_codebook[(uint)key_indices[key_base + 6u * cache_seq_capacity]];
        score_part += qrot7 * shared_k_codebook[(uint)key_indices[key_base + 7u * cache_seq_capacity]];
        // Recover original-magnitude codebook contribution; QJL term added
        // below stays in correct units (residual_norms captures the rescaled
        // codebook residual).
        score_part *= slot_scale;
        score_part += residual_scale * qproj0 * ((((sign_word >> (bit_base + 0u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj1 * ((((sign_word >> (bit_base + 1u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj2 * ((((sign_word >> (bit_base + 2u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj3 * ((((sign_word >> (bit_base + 3u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj4 * ((((sign_word >> (bit_base + 4u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj5 * ((((sign_word >> (bit_base + 5u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj6 * ((((sign_word >> (bit_base + 6u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj7 * ((((sign_word >> (bit_base + 7u)) & 1u) == 0u) ? -1.0f : 1.0f);
        float score = key_norm * simd_sum(score_part);

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        float value_scale = exp_score * value_norms[scalar_idx];
        acc0 = acc0 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 0u * cache_seq_capacity]];
        acc1 = acc1 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 1u * cache_seq_capacity]];
        acc2 = acc2 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 2u * cache_seq_capacity]];
        acc3 = acc3 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 3u * cache_seq_capacity]];
        acc4 = acc4 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 4u * cache_seq_capacity]];
        acc5 = acc5 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 5u * cache_seq_capacity]];
        acc6 = acc6 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 6u * cache_seq_capacity]];
        acc7 = acc7 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 7u * cache_seq_capacity]];
    }

    if (lane == 0u) {
        sums[row * blocks + block] = sum_exp_score;
        maxs[row * blocks + block] = max_score;
    }
    uint out_base = (row * blocks + block) * kDim + d0;
    partials[out_base + 0u] = acc0;
    partials[out_base + 1u] = acc1;
    partials[out_base + 2u] = acc2;
    partials[out_base + 3u] = acc3;
    partials[out_base + 4u] = acc4;
    partials[out_base + 5u] = acc5;
    partials[out_base + 6u] = acc6;
    partials[out_base + 7u] = acc7;
)";

static const char* TURBOQUANT_ATTENTION_Q8_D256_PACKED_KEYS_2PASS_1_SOURCE = R"(
    constexpr uint kDim = 256u;
    constexpr uint kVec = 8u;
    constexpr uint kKeyCentroids = 128u;
    constexpr uint kValueCentroids = 256u;
    constexpr float kQjlConst = 1.2533141373155003f / 256.0f;
    threadgroup float shared_k_codebook[kKeyCentroids];
    threadgroup float shared_v_codebook[kValueCentroids];

    uint kv_head = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint block = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint groups = q_heads / kv_heads;
    uint row = batch * q_heads + kv_head * groups + simd_gid;
    if (row >= n_rows || block >= blocks) return;

    if (simd_gid == 0u) {
        for (uint c = lane; c < kKeyCentroids; c += 32u) {
            shared_k_codebook[c] = key_codebook[c];
        }
        for (uint c = lane; c < kValueCentroids; c += 32u) {
            shared_v_codebook[c] = value_codebook[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_row = batch * kv_heads + kv_head;
    uint d0 = lane * kVec;
    uint query_base = row * kDim + d0;

    float qrot0 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 0u];
    float qrot1 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 1u];
    float qrot2 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 2u];
    float qrot3 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 3u];
    float qrot4 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 4u];
    float qrot5 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 5u];
    float qrot6 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 6u];
    float qrot7 = as_type<float>((uint)attn_scale_bits) * query_rot[query_base + 7u];
    float qproj0 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 0u];
    float qproj1 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 1u];
    float qproj2 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 2u];
    float qproj3 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 3u];
    float qproj4 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 4u];
    float qproj5 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 5u];
    float qproj6 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 6u];
    float qproj7 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 7u];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint seq = block; seq < n_seq; seq += blocks) {
        uint scalar_idx = kv_row * cache_seq_capacity + seq;
        uint scale_base = scalar_idx * 4u;
        float key_norm = slot_scales[scale_base + 0u];
        float residual_scale = slot_scales[scale_base + 1u] * kQjlConst;
        float key_slot_scale = slot_scales[scale_base + 3u];
        float value_norm = slot_scales[scale_base + 2u];
        uint key_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;
        uint value_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;

        uchar key_byte0 = key_bytes[key_base + 0u];
        uchar key_byte1 = key_bytes[key_base + 1u];
        uchar key_byte2 = key_bytes[key_base + 2u];
        uchar key_byte3 = key_bytes[key_base + 3u];
        uchar key_byte4 = key_bytes[key_base + 4u];
        uchar key_byte5 = key_bytes[key_base + 5u];
        uchar key_byte6 = key_bytes[key_base + 6u];
        uchar key_byte7 = key_bytes[key_base + 7u];

        float sign0 = (key_byte0 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign1 = (key_byte1 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign2 = (key_byte2 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign3 = (key_byte3 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign4 = (key_byte4 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign5 = (key_byte5 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign6 = (key_byte6 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign7 = (key_byte7 & 0x80u) == 0u ? -1.0f : 1.0f;

        float score_part = 0.0f;
        score_part += qrot0 * shared_k_codebook[(uint)(key_byte0 & 0x7fu)];
        score_part += qrot1 * shared_k_codebook[(uint)(key_byte1 & 0x7fu)];
        score_part += qrot2 * shared_k_codebook[(uint)(key_byte2 & 0x7fu)];
        score_part += qrot3 * shared_k_codebook[(uint)(key_byte3 & 0x7fu)];
        score_part += qrot4 * shared_k_codebook[(uint)(key_byte4 & 0x7fu)];
        score_part += qrot5 * shared_k_codebook[(uint)(key_byte5 & 0x7fu)];
        score_part += qrot6 * shared_k_codebook[(uint)(key_byte6 & 0x7fu)];
        score_part += qrot7 * shared_k_codebook[(uint)(key_byte7 & 0x7fu)];
        // Codebook indices were quantised against rotated values divided by
        // key_slot_scale; recover the original-magnitude codebook contribution.
        // QJL term below is in correct units already (residual_norm captures
        // the rescaled-codebook residual), so it's added AFTER this multiply.
        score_part *= key_slot_scale;
        score_part += residual_scale * qproj0 * sign0;
        score_part += residual_scale * qproj1 * sign1;
        score_part += residual_scale * qproj2 * sign2;
        score_part += residual_scale * qproj3 * sign3;
        score_part += residual_scale * qproj4 * sign4;
        score_part += residual_scale * qproj5 * sign5;
        score_part += residual_scale * qproj6 * sign6;
        score_part += residual_scale * qproj7 * sign7;
        float score = key_norm * simd_sum(score_part);

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        float value_scale = exp_score * value_norm;
        acc0 = acc0 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 0u]];
        acc1 = acc1 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 1u]];
        acc2 = acc2 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 2u]];
        acc3 = acc3 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 3u]];
        acc4 = acc4 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 4u]];
        acc5 = acc5 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 5u]];
        acc6 = acc6 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 6u]];
        acc7 = acc7 * factor + value_scale * shared_v_codebook[(uint)value_indices[value_base + 7u]];
    }

    if (lane == 0u) {
        sums[row * blocks + block] = sum_exp_score;
        maxs[row * blocks + block] = max_score;
    }
    uint out_base = (row * blocks + block) * kDim + d0;
    partials[out_base + 0u] = acc0;
    partials[out_base + 1u] = acc1;
    partials[out_base + 2u] = acc2;
    partials[out_base + 3u] = acc3;
    partials[out_base + 4u] = acc4;
    partials[out_base + 5u] = acc5;
    partials[out_base + 6u] = acc6;
    partials[out_base + 7u] = acc7;
)";

static const char* TURBOQUANT_ATTENTION_Q8_D256_PACKED_KEYS_DENSE_VALUES_2PASS_1_SOURCE = R"(
    constexpr uint kDim = 256u;
    constexpr uint kVec = 8u;
    constexpr uint kKeyCentroids = 128u;
    constexpr float kQjlConst = 1.2533141373155003f / 256.0f;
    threadgroup float shared_k_codebook[kKeyCentroids];

    uint kv_head = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint block = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint groups = q_heads / kv_heads;
    uint row = batch * q_heads + kv_head * groups + simd_gid;
    if (row >= n_rows || block >= blocks) return;

    if (simd_gid == 0u) {
        for (uint c = lane; c < kKeyCentroids; c += 32u) {
            shared_k_codebook[c] = key_codebook[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_row = batch * kv_heads + kv_head;
    uint d0 = lane * kVec;
    uint query_base = row * kDim + d0;

    float attn_scale = as_type<float>((uint)attn_scale_bits);
    float qrot0 = attn_scale * query_rot[query_base + 0u];
    float qrot1 = attn_scale * query_rot[query_base + 1u];
    float qrot2 = attn_scale * query_rot[query_base + 2u];
    float qrot3 = attn_scale * query_rot[query_base + 3u];
    float qrot4 = attn_scale * query_rot[query_base + 4u];
    float qrot5 = attn_scale * query_rot[query_base + 5u];
    float qrot6 = attn_scale * query_rot[query_base + 6u];
    float qrot7 = attn_scale * query_rot[query_base + 7u];
    float qproj0 = attn_scale * query_proj[query_base + 0u];
    float qproj1 = attn_scale * query_proj[query_base + 1u];
    float qproj2 = attn_scale * query_proj[query_base + 2u];
    float qproj3 = attn_scale * query_proj[query_base + 3u];
    float qproj4 = attn_scale * query_proj[query_base + 4u];
    float qproj5 = attn_scale * query_proj[query_base + 5u];
    float qproj6 = attn_scale * query_proj[query_base + 6u];
    float qproj7 = attn_scale * query_proj[query_base + 7u];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint seq = block; seq < n_seq; seq += blocks) {
        uint scalar_idx = kv_row * cache_seq_capacity + seq;
        uint scale_base = scalar_idx * 4u;
        float key_norm = slot_scales[scale_base + 0u];
        float residual_scale = slot_scales[scale_base + 1u] * kQjlConst;
        float key_slot_scale = slot_scales[scale_base + 3u];
        uint key_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;
        uint value_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;

        uchar key_byte0 = key_bytes[key_base + 0u];
        uchar key_byte1 = key_bytes[key_base + 1u];
        uchar key_byte2 = key_bytes[key_base + 2u];
        uchar key_byte3 = key_bytes[key_base + 3u];
        uchar key_byte4 = key_bytes[key_base + 4u];
        uchar key_byte5 = key_bytes[key_base + 5u];
        uchar key_byte6 = key_bytes[key_base + 6u];
        uchar key_byte7 = key_bytes[key_base + 7u];

        float sign0 = (key_byte0 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign1 = (key_byte1 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign2 = (key_byte2 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign3 = (key_byte3 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign4 = (key_byte4 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign5 = (key_byte5 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign6 = (key_byte6 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign7 = (key_byte7 & 0x80u) == 0u ? -1.0f : 1.0f;

        float score_part = 0.0f;
        score_part += qrot0 * shared_k_codebook[(uint)(key_byte0 & 0x7fu)];
        score_part += qrot1 * shared_k_codebook[(uint)(key_byte1 & 0x7fu)];
        score_part += qrot2 * shared_k_codebook[(uint)(key_byte2 & 0x7fu)];
        score_part += qrot3 * shared_k_codebook[(uint)(key_byte3 & 0x7fu)];
        score_part += qrot4 * shared_k_codebook[(uint)(key_byte4 & 0x7fu)];
        score_part += qrot5 * shared_k_codebook[(uint)(key_byte5 & 0x7fu)];
        score_part += qrot6 * shared_k_codebook[(uint)(key_byte6 & 0x7fu)];
        score_part += qrot7 * shared_k_codebook[(uint)(key_byte7 & 0x7fu)];
        // Codebook indices were quantised against rotated values divided by
        // key_slot_scale; recover the original-magnitude codebook contribution.
        // QJL term below is in correct units already (residual_norm captures
        // the rescaled-codebook residual), so it's added AFTER this multiply.
        score_part *= key_slot_scale;
        score_part += residual_scale * qproj0 * sign0;
        score_part += residual_scale * qproj1 * sign1;
        score_part += residual_scale * qproj2 * sign2;
        score_part += residual_scale * qproj3 * sign3;
        score_part += residual_scale * qproj4 * sign4;
        score_part += residual_scale * qproj5 * sign5;
        score_part += residual_scale * qproj6 * sign6;
        score_part += residual_scale * qproj7 * sign7;
        float score = key_norm * simd_sum(score_part);

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        acc0 = acc0 * factor + exp_score * float(value_dense[value_base + 0u]);
        acc1 = acc1 * factor + exp_score * float(value_dense[value_base + 1u]);
        acc2 = acc2 * factor + exp_score * float(value_dense[value_base + 2u]);
        acc3 = acc3 * factor + exp_score * float(value_dense[value_base + 3u]);
        acc4 = acc4 * factor + exp_score * float(value_dense[value_base + 4u]);
        acc5 = acc5 * factor + exp_score * float(value_dense[value_base + 5u]);
        acc6 = acc6 * factor + exp_score * float(value_dense[value_base + 6u]);
        acc7 = acc7 * factor + exp_score * float(value_dense[value_base + 7u]);
    }

    if (lane == 0u) {
        sums[row * blocks + block] = sum_exp_score;
        maxs[row * blocks + block] = max_score;
    }
    uint out_base = (row * blocks + block) * kDim + d0;
    partials[out_base + 0u] = acc0;
    partials[out_base + 1u] = acc1;
    partials[out_base + 2u] = acc2;
    partials[out_base + 3u] = acc3;
    partials[out_base + 4u] = acc4;
    partials[out_base + 5u] = acc5;
    partials[out_base + 6u] = acc6;
    partials[out_base + 7u] = acc7;
)";

static const char* TURBOQUANT_ATTENTION_Q8_D256_FULLBYTE_DENSE_VALUES_2PASS_1_SOURCE = R"(
    constexpr uint kDim = 256u;
    constexpr uint kVec = 8u;
    constexpr uint kKeyCentroids = 256u;
    threadgroup float shared_k_codebook[kKeyCentroids];

    uint kv_head = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint block = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint groups = q_heads / kv_heads;
    uint row = batch * q_heads + kv_head * groups + simd_gid;
    if (row >= n_rows || block >= blocks) return;

    if (simd_gid == 0u) {
        for (uint c = lane; c < kKeyCentroids; c += 32u) {
            shared_k_codebook[c] = key_codebook[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_row = batch * kv_heads + kv_head;
    uint d0 = lane * kVec;
    uint query_base = row * kDim + d0;

    float attn_scale = as_type<float>((uint)attn_scale_bits);
    float qrot0 = attn_scale * query_rot[query_base + 0u];
    float qrot1 = attn_scale * query_rot[query_base + 1u];
    float qrot2 = attn_scale * query_rot[query_base + 2u];
    float qrot3 = attn_scale * query_rot[query_base + 3u];
    float qrot4 = attn_scale * query_rot[query_base + 4u];
    float qrot5 = attn_scale * query_rot[query_base + 5u];
    float qrot6 = attn_scale * query_rot[query_base + 6u];
    float qrot7 = attn_scale * query_rot[query_base + 7u];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint seq = block; seq < n_seq; seq += blocks) {
        uint scalar_idx = kv_row * cache_seq_capacity + seq;
        float key_norm = slot_scales[scalar_idx * 4u + 0u];
        float key_slot_scale = slot_scales[scalar_idx * 4u + 3u];
        uint key_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;
        uint value_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;

        uchar key_idx0 = key_indices[key_base + 0u];
        uchar key_idx1 = key_indices[key_base + 1u];
        uchar key_idx2 = key_indices[key_base + 2u];
        uchar key_idx3 = key_indices[key_base + 3u];
        uchar key_idx4 = key_indices[key_base + 4u];
        uchar key_idx5 = key_indices[key_base + 5u];
        uchar key_idx6 = key_indices[key_base + 6u];
        uchar key_idx7 = key_indices[key_base + 7u];

        float score_part = 0.0f;
        score_part += qrot0 * shared_k_codebook[(uint)key_idx0];
        score_part += qrot1 * shared_k_codebook[(uint)key_idx1];
        score_part += qrot2 * shared_k_codebook[(uint)key_idx2];
        score_part += qrot3 * shared_k_codebook[(uint)key_idx3];
        score_part += qrot4 * shared_k_codebook[(uint)key_idx4];
        score_part += qrot5 * shared_k_codebook[(uint)key_idx5];
        score_part += qrot6 * shared_k_codebook[(uint)key_idx6];
        score_part += qrot7 * shared_k_codebook[(uint)key_idx7];
        // Recover original-magnitude codebook contribution; see PACKED_KEYS
        // kernel commentary above. Fullbyte kernels have no QJL residual term,
        // so slot_scale folds cleanly into the codebook accumulator.
        score_part *= key_slot_scale;
        float score = key_norm * simd_sum(score_part);

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        acc0 = acc0 * factor + exp_score * float(value_dense[value_base + 0u]);
        acc1 = acc1 * factor + exp_score * float(value_dense[value_base + 1u]);
        acc2 = acc2 * factor + exp_score * float(value_dense[value_base + 2u]);
        acc3 = acc3 * factor + exp_score * float(value_dense[value_base + 3u]);
        acc4 = acc4 * factor + exp_score * float(value_dense[value_base + 4u]);
        acc5 = acc5 * factor + exp_score * float(value_dense[value_base + 5u]);
        acc6 = acc6 * factor + exp_score * float(value_dense[value_base + 6u]);
        acc7 = acc7 * factor + exp_score * float(value_dense[value_base + 7u]);
    }

    if (lane == 0u) {
        sums[row * blocks + block] = sum_exp_score;
        maxs[row * blocks + block] = max_score;
    }
    uint out_base = (row * blocks + block) * kDim + d0;
    partials[out_base + 0u] = acc0;
    partials[out_base + 1u] = acc1;
    partials[out_base + 2u] = acc2;
    partials[out_base + 3u] = acc3;
    partials[out_base + 4u] = acc4;
    partials[out_base + 5u] = acc5;
    partials[out_base + 6u] = acc6;
    partials[out_base + 7u] = acc7;
)";

static const char* TURBOQUANT_ATTENTION_Q8_D256_FULLBYTE_DENSE_VALUES_2PASS_1_LOCALSOFTMAX_SOURCE = R"(
    constexpr uint kDim = 256u;
    constexpr uint kVec = 8u;
    constexpr uint kKeyCentroids = 256u;
    threadgroup float shared_k_codebook[kKeyCentroids];

    uint kv_head = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint block = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint groups = q_heads / kv_heads;
    uint row = batch * q_heads + kv_head * groups + simd_gid;
    if (row >= n_rows || block >= blocks) return;

    if (simd_gid == 0u) {
        for (uint c = lane; c < kKeyCentroids; c += 32u) {
            shared_k_codebook[c] = key_codebook[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_row = batch * kv_heads + kv_head;
    uint d0 = lane * kVec;
    uint query_base = row * kDim + d0;

    float attn_scale = as_type<float>((uint)attn_scale_bits);
    float qrot0 = attn_scale * query_rot[query_base + 0u];
    float qrot1 = attn_scale * query_rot[query_base + 1u];
    float qrot2 = attn_scale * query_rot[query_base + 2u];
    float qrot3 = attn_scale * query_rot[query_base + 3u];
    float qrot4 = attn_scale * query_rot[query_base + 4u];
    float qrot5 = attn_scale * query_rot[query_base + 5u];
    float qrot6 = attn_scale * query_rot[query_base + 6u];
    float qrot7 = attn_scale * query_rot[query_base + 7u];

    float local_max = -INFINITY;
    for (uint seq = block; seq < n_seq; seq += blocks) {
        uint scalar_idx = kv_row * cache_seq_capacity + seq;
        float key_norm = slot_scales[scalar_idx * 4u + 0u];
        float key_slot_scale = slot_scales[scalar_idx * 4u + 3u];
        uint key_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;

        uchar key_idx0 = key_indices[key_base + 0u];
        uchar key_idx1 = key_indices[key_base + 1u];
        uchar key_idx2 = key_indices[key_base + 2u];
        uchar key_idx3 = key_indices[key_base + 3u];
        uchar key_idx4 = key_indices[key_base + 4u];
        uchar key_idx5 = key_indices[key_base + 5u];
        uchar key_idx6 = key_indices[key_base + 6u];
        uchar key_idx7 = key_indices[key_base + 7u];

        float score_part = 0.0f;
        score_part += qrot0 * shared_k_codebook[(uint)key_idx0];
        score_part += qrot1 * shared_k_codebook[(uint)key_idx1];
        score_part += qrot2 * shared_k_codebook[(uint)key_idx2];
        score_part += qrot3 * shared_k_codebook[(uint)key_idx3];
        score_part += qrot4 * shared_k_codebook[(uint)key_idx4];
        score_part += qrot5 * shared_k_codebook[(uint)key_idx5];
        score_part += qrot6 * shared_k_codebook[(uint)key_idx6];
        score_part += qrot7 * shared_k_codebook[(uint)key_idx7];
        // Recover original-magnitude codebook contribution; see PACKED_KEYS
        // kernel commentary above. Fullbyte kernels have no QJL residual term,
        // so slot_scale folds cleanly into the codebook accumulator.
        score_part *= key_slot_scale;
        float score = key_norm * simd_sum(score_part);
        local_max = max(local_max, score);
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float sum_exp_score = 0.0f;

    for (uint seq = block; seq < n_seq; seq += blocks) {
        uint scalar_idx = kv_row * cache_seq_capacity + seq;
        float key_norm = slot_scales[scalar_idx * 4u + 0u];
        float key_slot_scale = slot_scales[scalar_idx * 4u + 3u];
        uint key_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;
        uint value_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;

        uchar key_idx0 = key_indices[key_base + 0u];
        uchar key_idx1 = key_indices[key_base + 1u];
        uchar key_idx2 = key_indices[key_base + 2u];
        uchar key_idx3 = key_indices[key_base + 3u];
        uchar key_idx4 = key_indices[key_base + 4u];
        uchar key_idx5 = key_indices[key_base + 5u];
        uchar key_idx6 = key_indices[key_base + 6u];
        uchar key_idx7 = key_indices[key_base + 7u];

        float score_part = 0.0f;
        score_part += qrot0 * shared_k_codebook[(uint)key_idx0];
        score_part += qrot1 * shared_k_codebook[(uint)key_idx1];
        score_part += qrot2 * shared_k_codebook[(uint)key_idx2];
        score_part += qrot3 * shared_k_codebook[(uint)key_idx3];
        score_part += qrot4 * shared_k_codebook[(uint)key_idx4];
        score_part += qrot5 * shared_k_codebook[(uint)key_idx5];
        score_part += qrot6 * shared_k_codebook[(uint)key_idx6];
        score_part += qrot7 * shared_k_codebook[(uint)key_idx7];
        // Recover original-magnitude codebook contribution; see PACKED_KEYS
        // kernel commentary above. Fullbyte kernels have no QJL residual term,
        // so slot_scale folds cleanly into the codebook accumulator.
        score_part *= key_slot_scale;
        float score = key_norm * simd_sum(score_part);
        float exp_score = fast::exp(score - local_max);
        sum_exp_score += exp_score;

        acc0 += exp_score * float(value_dense[value_base + 0u]);
        acc1 += exp_score * float(value_dense[value_base + 1u]);
        acc2 += exp_score * float(value_dense[value_base + 2u]);
        acc3 += exp_score * float(value_dense[value_base + 3u]);
        acc4 += exp_score * float(value_dense[value_base + 4u]);
        acc5 += exp_score * float(value_dense[value_base + 5u]);
        acc6 += exp_score * float(value_dense[value_base + 6u]);
        acc7 += exp_score * float(value_dense[value_base + 7u]);
    }

    if (lane == 0u) {
        sums[row * blocks + block] = sum_exp_score;
        maxs[row * blocks + block] = local_max;
    }
    uint out_base = (row * blocks + block) * kDim + d0;
    partials[out_base + 0u] = acc0;
    partials[out_base + 1u] = acc1;
    partials[out_base + 2u] = acc2;
    partials[out_base + 3u] = acc3;
    partials[out_base + 4u] = acc4;
    partials[out_base + 5u] = acc5;
    partials[out_base + 6u] = acc6;
    partials[out_base + 7u] = acc7;
)";

static const char* TURBOQUANT_ATTENTION_Q8_D256_PACKED_KV_2PASS_1_SOURCE = R"(
    constexpr uint kDim = 256u;
    constexpr uint kVec = 8u;
    constexpr uint kKeyCentroids = 128u;
    constexpr uint kValueCentroids = 256u;
    constexpr float kQjlConst = 1.2533141373155003f / 256.0f;
    threadgroup float shared_k_codebook[kKeyCentroids];
    threadgroup float shared_v_codebook[kValueCentroids];

    uint kv_head = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint block = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint groups = q_heads / kv_heads;
    uint row = batch * q_heads + kv_head * groups + simd_gid;
    if (row >= n_rows || block >= blocks) return;

    if (simd_gid == 0u) {
        for (uint c = lane; c < kKeyCentroids; c += 32u) {
            shared_k_codebook[c] = key_codebook[c];
        }
        for (uint c = lane; c < kValueCentroids; c += 32u) {
            shared_v_codebook[c] = value_codebook[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_row = batch * kv_heads + kv_head;
    uint d0 = lane * kVec;
    uint query_base = row * kDim + d0;

    float attn_scale = as_type<float>((uint)attn_scale_bits);
    float qrot0 = attn_scale * query_rot[query_base + 0u];
    float qrot1 = attn_scale * query_rot[query_base + 1u];
    float qrot2 = attn_scale * query_rot[query_base + 2u];
    float qrot3 = attn_scale * query_rot[query_base + 3u];
    float qrot4 = attn_scale * query_rot[query_base + 4u];
    float qrot5 = attn_scale * query_rot[query_base + 5u];
    float qrot6 = attn_scale * query_rot[query_base + 6u];
    float qrot7 = attn_scale * query_rot[query_base + 7u];
    float qproj0 = attn_scale * query_proj[query_base + 0u];
    float qproj1 = attn_scale * query_proj[query_base + 1u];
    float qproj2 = attn_scale * query_proj[query_base + 2u];
    float qproj3 = attn_scale * query_proj[query_base + 3u];
    float qproj4 = attn_scale * query_proj[query_base + 4u];
    float qproj5 = attn_scale * query_proj[query_base + 5u];
    float qproj6 = attn_scale * query_proj[query_base + 6u];
    float qproj7 = attn_scale * query_proj[query_base + 7u];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint seq = block; seq < n_seq; seq += blocks) {
        uint scalar_idx = kv_row * cache_seq_capacity + seq;
        uint scale_base = scalar_idx * 4u;
        float key_norm = slot_scales[scale_base + 0u];
        float residual_scale = slot_scales[scale_base + 1u] * kQjlConst;
        float key_slot_scale = slot_scales[scale_base + 3u];
        float value_norm = slot_scales[scale_base + 2u];
        uint kv_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;

        ushort kv_pair0 = kv_bytes[kv_base + 0u];
        ushort kv_pair1 = kv_bytes[kv_base + 1u];
        ushort kv_pair2 = kv_bytes[kv_base + 2u];
        ushort kv_pair3 = kv_bytes[kv_base + 3u];
        ushort kv_pair4 = kv_bytes[kv_base + 4u];
        ushort kv_pair5 = kv_bytes[kv_base + 5u];
        ushort kv_pair6 = kv_bytes[kv_base + 6u];
        ushort kv_pair7 = kv_bytes[kv_base + 7u];

        uchar key_byte0 = (uchar)(kv_pair0 & 0xffu);
        uchar key_byte1 = (uchar)(kv_pair1 & 0xffu);
        uchar key_byte2 = (uchar)(kv_pair2 & 0xffu);
        uchar key_byte3 = (uchar)(kv_pair3 & 0xffu);
        uchar key_byte4 = (uchar)(kv_pair4 & 0xffu);
        uchar key_byte5 = (uchar)(kv_pair5 & 0xffu);
        uchar key_byte6 = (uchar)(kv_pair6 & 0xffu);
        uchar key_byte7 = (uchar)(kv_pair7 & 0xffu);

        float sign0 = (key_byte0 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign1 = (key_byte1 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign2 = (key_byte2 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign3 = (key_byte3 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign4 = (key_byte4 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign5 = (key_byte5 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign6 = (key_byte6 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign7 = (key_byte7 & 0x80u) == 0u ? -1.0f : 1.0f;

        float score_part = 0.0f;
        score_part += qrot0 * shared_k_codebook[(uint)(key_byte0 & 0x7fu)];
        score_part += qrot1 * shared_k_codebook[(uint)(key_byte1 & 0x7fu)];
        score_part += qrot2 * shared_k_codebook[(uint)(key_byte2 & 0x7fu)];
        score_part += qrot3 * shared_k_codebook[(uint)(key_byte3 & 0x7fu)];
        score_part += qrot4 * shared_k_codebook[(uint)(key_byte4 & 0x7fu)];
        score_part += qrot5 * shared_k_codebook[(uint)(key_byte5 & 0x7fu)];
        score_part += qrot6 * shared_k_codebook[(uint)(key_byte6 & 0x7fu)];
        score_part += qrot7 * shared_k_codebook[(uint)(key_byte7 & 0x7fu)];
        // Codebook indices were quantised against rotated values divided by
        // key_slot_scale; recover the original-magnitude codebook contribution.
        // QJL term below is in correct units already (residual_norm captures
        // the rescaled-codebook residual), so it's added AFTER this multiply.
        score_part *= key_slot_scale;
        score_part += residual_scale * qproj0 * sign0;
        score_part += residual_scale * qproj1 * sign1;
        score_part += residual_scale * qproj2 * sign2;
        score_part += residual_scale * qproj3 * sign3;
        score_part += residual_scale * qproj4 * sign4;
        score_part += residual_scale * qproj5 * sign5;
        score_part += residual_scale * qproj6 * sign6;
        score_part += residual_scale * qproj7 * sign7;
        float score = key_norm * simd_sum(score_part);

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        float value_scale = exp_score * value_norm;
        acc0 = acc0 * factor + value_scale * shared_v_codebook[(uint)(kv_pair0 >> 8)];
        acc1 = acc1 * factor + value_scale * shared_v_codebook[(uint)(kv_pair1 >> 8)];
        acc2 = acc2 * factor + value_scale * shared_v_codebook[(uint)(kv_pair2 >> 8)];
        acc3 = acc3 * factor + value_scale * shared_v_codebook[(uint)(kv_pair3 >> 8)];
        acc4 = acc4 * factor + value_scale * shared_v_codebook[(uint)(kv_pair4 >> 8)];
        acc5 = acc5 * factor + value_scale * shared_v_codebook[(uint)(kv_pair5 >> 8)];
        acc6 = acc6 * factor + value_scale * shared_v_codebook[(uint)(kv_pair6 >> 8)];
        acc7 = acc7 * factor + value_scale * shared_v_codebook[(uint)(kv_pair7 >> 8)];
    }

    if (lane == 0u) {
        sums[row * blocks + block] = sum_exp_score;
        maxs[row * blocks + block] = max_score;
    }
    uint out_base = (row * blocks + block) * kDim + d0;
    partials[out_base + 0u] = acc0;
    partials[out_base + 1u] = acc1;
    partials[out_base + 2u] = acc2;
    partials[out_base + 3u] = acc3;
    partials[out_base + 4u] = acc4;
    partials[out_base + 5u] = acc5;
    partials[out_base + 6u] = acc6;
    partials[out_base + 7u] = acc7;
)";

static const char* TURBOQUANT_ATTENTION_Q8_D256_PACKED_KV_DENSE_VALUES_2PASS_1_SOURCE = R"(
    constexpr uint kDim = 256u;
    constexpr uint kVec = 8u;
    constexpr uint kKeyCentroids = 128u;
    constexpr float kQjlConst = 1.2533141373155003f / 256.0f;
    threadgroup float shared_k_codebook[kKeyCentroids];

    uint kv_head = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint block = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint groups = q_heads / kv_heads;
    uint row = batch * q_heads + kv_head * groups + simd_gid;
    if (row >= n_rows || block >= blocks) return;

    if (simd_gid == 0u) {
        for (uint c = lane; c < kKeyCentroids; c += 32u) {
            shared_k_codebook[c] = key_codebook[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_row = batch * kv_heads + kv_head;
    uint d0 = lane * kVec;
    uint query_base = row * kDim + d0;

    float attn_scale = as_type<float>((uint)attn_scale_bits);
    float qrot0 = attn_scale * query_rot[query_base + 0u];
    float qrot1 = attn_scale * query_rot[query_base + 1u];
    float qrot2 = attn_scale * query_rot[query_base + 2u];
    float qrot3 = attn_scale * query_rot[query_base + 3u];
    float qrot4 = attn_scale * query_rot[query_base + 4u];
    float qrot5 = attn_scale * query_rot[query_base + 5u];
    float qrot6 = attn_scale * query_rot[query_base + 6u];
    float qrot7 = attn_scale * query_rot[query_base + 7u];
    float qproj0 = attn_scale * query_proj[query_base + 0u];
    float qproj1 = attn_scale * query_proj[query_base + 1u];
    float qproj2 = attn_scale * query_proj[query_base + 2u];
    float qproj3 = attn_scale * query_proj[query_base + 3u];
    float qproj4 = attn_scale * query_proj[query_base + 4u];
    float qproj5 = attn_scale * query_proj[query_base + 5u];
    float qproj6 = attn_scale * query_proj[query_base + 6u];
    float qproj7 = attn_scale * query_proj[query_base + 7u];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint seq = block; seq < n_seq; seq += blocks) {
        uint scalar_idx = kv_row * cache_seq_capacity + seq;
        uint scale_base = scalar_idx * 4u;
        float key_norm = slot_scales[scale_base + 0u];
        float residual_scale = slot_scales[scale_base + 1u] * kQjlConst;
        float key_slot_scale = slot_scales[scale_base + 3u];
        uint kv_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;
        uint value_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;

        ushort kv_pair0 = kv_bytes[kv_base + 0u];
        ushort kv_pair1 = kv_bytes[kv_base + 1u];
        ushort kv_pair2 = kv_bytes[kv_base + 2u];
        ushort kv_pair3 = kv_bytes[kv_base + 3u];
        ushort kv_pair4 = kv_bytes[kv_base + 4u];
        ushort kv_pair5 = kv_bytes[kv_base + 5u];
        ushort kv_pair6 = kv_bytes[kv_base + 6u];
        ushort kv_pair7 = kv_bytes[kv_base + 7u];

        uchar key_byte0 = (uchar)(kv_pair0 & 0xffu);
        uchar key_byte1 = (uchar)(kv_pair1 & 0xffu);
        uchar key_byte2 = (uchar)(kv_pair2 & 0xffu);
        uchar key_byte3 = (uchar)(kv_pair3 & 0xffu);
        uchar key_byte4 = (uchar)(kv_pair4 & 0xffu);
        uchar key_byte5 = (uchar)(kv_pair5 & 0xffu);
        uchar key_byte6 = (uchar)(kv_pair6 & 0xffu);
        uchar key_byte7 = (uchar)(kv_pair7 & 0xffu);

        float sign0 = (key_byte0 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign1 = (key_byte1 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign2 = (key_byte2 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign3 = (key_byte3 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign4 = (key_byte4 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign5 = (key_byte5 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign6 = (key_byte6 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign7 = (key_byte7 & 0x80u) == 0u ? -1.0f : 1.0f;

        float score_part = 0.0f;
        score_part += qrot0 * shared_k_codebook[(uint)(key_byte0 & 0x7fu)];
        score_part += qrot1 * shared_k_codebook[(uint)(key_byte1 & 0x7fu)];
        score_part += qrot2 * shared_k_codebook[(uint)(key_byte2 & 0x7fu)];
        score_part += qrot3 * shared_k_codebook[(uint)(key_byte3 & 0x7fu)];
        score_part += qrot4 * shared_k_codebook[(uint)(key_byte4 & 0x7fu)];
        score_part += qrot5 * shared_k_codebook[(uint)(key_byte5 & 0x7fu)];
        score_part += qrot6 * shared_k_codebook[(uint)(key_byte6 & 0x7fu)];
        score_part += qrot7 * shared_k_codebook[(uint)(key_byte7 & 0x7fu)];
        // Codebook indices were quantised against rotated values divided by
        // key_slot_scale; recover the original-magnitude codebook contribution.
        // QJL term below is in correct units already (residual_norm captures
        // the rescaled-codebook residual), so it's added AFTER this multiply.
        score_part *= key_slot_scale;
        score_part += residual_scale * qproj0 * sign0;
        score_part += residual_scale * qproj1 * sign1;
        score_part += residual_scale * qproj2 * sign2;
        score_part += residual_scale * qproj3 * sign3;
        score_part += residual_scale * qproj4 * sign4;
        score_part += residual_scale * qproj5 * sign5;
        score_part += residual_scale * qproj6 * sign6;
        score_part += residual_scale * qproj7 * sign7;
        float score = key_norm * simd_sum(score_part);

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        acc0 = acc0 * factor + exp_score * float(value_dense[value_base + 0u]);
        acc1 = acc1 * factor + exp_score * float(value_dense[value_base + 1u]);
        acc2 = acc2 * factor + exp_score * float(value_dense[value_base + 2u]);
        acc3 = acc3 * factor + exp_score * float(value_dense[value_base + 3u]);
        acc4 = acc4 * factor + exp_score * float(value_dense[value_base + 4u]);
        acc5 = acc5 * factor + exp_score * float(value_dense[value_base + 5u]);
        acc6 = acc6 * factor + exp_score * float(value_dense[value_base + 6u]);
        acc7 = acc7 * factor + exp_score * float(value_dense[value_base + 7u]);
    }

    if (lane == 0u) {
        sums[row * blocks + block] = sum_exp_score;
        maxs[row * blocks + block] = max_score;
    }
    uint out_base = (row * blocks + block) * kDim + d0;
    partials[out_base + 0u] = acc0;
    partials[out_base + 1u] = acc1;
    partials[out_base + 2u] = acc2;
    partials[out_base + 3u] = acc3;
    partials[out_base + 4u] = acc4;
    partials[out_base + 5u] = acc5;
    partials[out_base + 6u] = acc6;
    partials[out_base + 7u] = acc7;
)";

static const char* TURBOQUANT_ATTENTION_Q8_D256_2PASS_2_SOURCE = R"(
    constexpr uint kBlocksPerSimd = 32u;
    constexpr uint kSimds = 32u;
    constexpr uint kVec = 8u;
    constexpr uint kDim = 256u;
    threadgroup float outputs[kSimds * kSimds];

    uint row = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    if (row >= n_rows || blocks == 0u) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    const device float* row_partials = partials + row * blocks * kDim + simd_gid * kDim + simd_lid * kVec;
    const device float* row_sums = sums + row * blocks;
    const device float* row_maxs = maxs + row * blocks;
    device float* row_out = output + row * kDim + simd_lid * kVec;

    float max_score = -INFINITY;
    for (uint b = 0u; b < blocks / kBlocksPerSimd; ++b) {
        max_score = max(max_score, row_maxs[simd_lid + kBlocksPerSimd * b]);
    }
    max_score = simd_max(max_score);

    float sum_exp_score = 0.0f;
    for (uint b = 0u; b < blocks / kBlocksPerSimd; ++b) {
        float factor = fast::exp(row_maxs[simd_lid + kBlocksPerSimd * b] - max_score);
        sum_exp_score += factor * row_sums[simd_lid + kBlocksPerSimd * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    const device float* partial_ptr = row_partials;
    const device float* max_ptr = row_maxs;
    for (uint b = 0u; b < blocks / kBlocksPerSimd; ++b) {
        float factor = fast::exp(max_ptr[simd_gid] - max_score);
        acc0 += factor * partial_ptr[0u];
        acc1 += factor * partial_ptr[1u];
        acc2 += factor * partial_ptr[2u];
        acc3 += factor * partial_ptr[3u];
        acc4 += factor * partial_ptr[4u];
        acc5 += factor * partial_ptr[5u];
        acc6 += factor * partial_ptr[6u];
        acc7 += factor * partial_ptr[7u];
        partial_ptr += kBlocksPerSimd * kDim;
        max_ptr += kBlocksPerSimd;
    }

    outputs[simd_gid * kSimds + simd_lid] = acc0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0u) {
        float sum = 0.0f;
        for (uint g = 0u; g < kSimds; ++g) sum += outputs[g * kSimds + simd_lid];
        acc0 = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outputs[simd_gid * kSimds + simd_lid] = acc1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0u) {
        float sum = 0.0f;
        for (uint g = 0u; g < kSimds; ++g) sum += outputs[g * kSimds + simd_lid];
        acc1 = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outputs[simd_gid * kSimds + simd_lid] = acc2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0u) {
        float sum = 0.0f;
        for (uint g = 0u; g < kSimds; ++g) sum += outputs[g * kSimds + simd_lid];
        acc2 = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outputs[simd_gid * kSimds + simd_lid] = acc3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0u) {
        float sum = 0.0f;
        for (uint g = 0u; g < kSimds; ++g) sum += outputs[g * kSimds + simd_lid];
        acc3 = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outputs[simd_gid * kSimds + simd_lid] = acc4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0u) {
        float sum = 0.0f;
        for (uint g = 0u; g < kSimds; ++g) sum += outputs[g * kSimds + simd_lid];
        acc4 = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outputs[simd_gid * kSimds + simd_lid] = acc5;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0u) {
        float sum = 0.0f;
        for (uint g = 0u; g < kSimds; ++g) sum += outputs[g * kSimds + simd_lid];
        acc5 = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outputs[simd_gid * kSimds + simd_lid] = acc6;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0u) {
        float sum = 0.0f;
        for (uint g = 0u; g < kSimds; ++g) sum += outputs[g * kSimds + simd_lid];
        acc6 = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outputs[simd_gid * kSimds + simd_lid] = acc7;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0u) {
        float sum = 0.0f;
        for (uint g = 0u; g < kSimds; ++g) sum += outputs[g * kSimds + simd_lid];
        acc7 = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = sum_exp_score > 0.0f ? 1.0f / sum_exp_score : 0.0f;
    if (simd_gid == 0u) {
        row_out[0u] = acc0 * inv_sum;
        row_out[1u] = acc1 * inv_sum;
        row_out[2u] = acc2 * inv_sum;
        row_out[3u] = acc3 * inv_sum;
        row_out[4u] = acc4 * inv_sum;
        row_out[5u] = acc5 * inv_sum;
        row_out[6u] = acc6 * inv_sum;
        row_out[7u] = acc7 * inv_sum;
    }
)";

// Specialized q8 long-context decode primitive for D=128/V=128.
// Pass 1 performs block-strided online softmax and accumulates unnormalized
// partial value outputs directly from compressed K/V. Pass 2 merges the block
// partials using log-sum-exp arithmetic, following MLX's sdpa_vector_2pass

static mlx::core::fast::CustomKernelFunction& get_turboquant_attention_q8_d256_2pass_1_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_q8_d256_2pass_1",
        {
            "query_rot",
            "query_proj",
            "key_indices",
            "key_qjl_signs",
            "key_norms",
            "key_residual_norms",
            "key_slot_scale",
            "key_codebook",
            "value_indices",
            "value_norms",
            "value_codebook",
        },
        {"partials", "sums", "maxs"},
        TURBOQUANT_ATTENTION_Q8_D256_2PASS_1_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_attention_q8_d256_packed_keys_2pass_1_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_q8_d256_packed_keys_2pass_1",
        {
            "query_rot",
            "query_proj",
            "key_bytes",
            "slot_scales",
            "key_codebook",
            "value_indices",
            "value_codebook",
        },
        {"partials", "sums", "maxs"},
        TURBOQUANT_ATTENTION_Q8_D256_PACKED_KEYS_2PASS_1_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_attention_q8_d256_packed_keys_dense_values_2pass_1_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_q8_d256_packed_keys_dense_values_2pass_1",
        {
            "query_rot",
            "query_proj",
            "key_bytes",
            "slot_scales",
            "key_codebook",
            "value_dense",
        },
        {"partials", "sums", "maxs"},
        TURBOQUANT_ATTENTION_Q8_D256_PACKED_KEYS_DENSE_VALUES_2PASS_1_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_1_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_q8_d256_fullbyte_dense_values_2pass_1",
        {
            "query_rot",
            "key_indices",
            "slot_scales",
            "key_codebook",
            "value_dense",
        },
        {"partials", "sums", "maxs"},
        TURBOQUANT_ATTENTION_Q8_D256_FULLBYTE_DENSE_VALUES_2PASS_1_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_1_localsoftmax_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_q8_d256_fullbyte_dense_values_2pass_1_localsoftmax",
        {
            "query_rot",
            "key_indices",
            "slot_scales",
            "key_codebook",
            "value_dense",
        },
        {"partials", "sums", "maxs"},
        TURBOQUANT_ATTENTION_Q8_D256_FULLBYTE_DENSE_VALUES_2PASS_1_LOCALSOFTMAX_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}


static mlx::core::fast::CustomKernelFunction& get_turboquant_attention_q8_d256_packed_kv_2pass_1_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_q8_d256_packed_kv_2pass_1",
        {
            "query_rot",
            "query_proj",
            "kv_bytes",
            "slot_scales",
            "key_codebook",
            "value_codebook",
        },
        {"partials", "sums", "maxs"},
        TURBOQUANT_ATTENTION_Q8_D256_PACKED_KV_2PASS_1_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_attention_q8_d256_packed_kv_dense_values_2pass_1_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_q8_d256_packed_kv_dense_values_2pass_1",
        {
            "query_rot",
            "query_proj",
            "kv_bytes",
            "slot_scales",
            "key_codebook",
            "value_dense",
        },
        {"partials", "sums", "maxs"},
        TURBOQUANT_ATTENTION_Q8_D256_PACKED_KV_DENSE_VALUES_2PASS_1_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_attention_q8_d256_2pass_2_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_q8_d256_2pass_2",
        {"partials", "sums", "maxs"},
        {"output"},
        TURBOQUANT_ATTENTION_Q8_D256_2PASS_2_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

extern "C" {

int mlx_inline_turboquant_attention_q8_d256_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* key_qjl_signs,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_residual_norms,
    const mlx_inline_array* key_slot_scale,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_norms,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;
    const uint32_t blocks = turboquant_q8_2pass_blocks_override_or(32u);

    if (n_rows == 0 || n_seq < 1024 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& query_proj_arr = as_arr(query_proj);
        const array& key_indices_arr = as_arr(key_indices);
        const array& key_qjl_signs_arr = as_arr(key_qjl_signs);
        const array& key_norms_arr = as_arr(key_norms);
        const array& key_residual_norms_arr = as_arr(key_residual_norms);
        const array& key_slot_scale_arr = as_arr(key_slot_scale);
        const array& key_codebook_arr = as_arr(key_codebook);
        const array& value_indices_arr = as_arr(value_indices);
        const array& value_norms_arr = as_arr(value_norms);
        const array& value_codebook_arr = as_arr(value_codebook);

        if (query_rot_arr.shape(-1) != dim || query_proj_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 128 || value_codebook_arr.shape(0) != 256) return 1;

        auto& pass1 = get_turboquant_attention_q8_d256_2pass_1_kernel();
        auto pass1_outputs = pass1(
            {
                query_rot_arr,
                query_proj_arr,
                key_indices_arr,
                key_qjl_signs_arr,
                key_norms_arr,
                key_residual_norms_arr,
                key_slot_scale_arr,
                key_codebook_arr,
                value_indices_arr,
                value_norms_arr,
                value_codebook_arr,
            },
            {{(int)n_rows, (int)blocks, (int)dim}, {(int)n_rows, (int)blocks}, {(int)n_rows, (int)blocks}},
            {float32, float32, float32},
            {32 * (int)kv_heads, (int)((q_heads / kv_heads) * (n_rows / q_heads)), (int)blocks},
            {32, (int)(q_heads / kv_heads), 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"blocks", (int)blocks},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
                {"attn_scale_bits", (int)attn_scale_bits},
            },
            std::nullopt, false, {}
        );

        auto& pass2 = get_turboquant_attention_q8_d256_2pass_2_kernel();
        auto pass2_outputs = pass2(
            {pass1_outputs[0], pass1_outputs[1], pass1_outputs[2]},
            {{(int)n_rows, (int)dim}},
            {float32},
            {1024, (int)n_rows, 1},
            {1024, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"blocks", (int)blocks},
            },
            std::nullopt, false, {}
        );

        new (out->buf) array(pass2_outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_2pass", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_2pass", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_attention_q8_d256_packed_keys_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_bytes,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;
    const uint32_t blocks = turboquant_q8_2pass_blocks_override_or(32u);

    if (n_rows == 0 || n_seq < 1024 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& query_proj_arr = as_arr(query_proj);
        const array& key_bytes_arr = as_arr(key_bytes);
        const array& slot_scales_arr = as_arr(slot_scales);
        const array& key_codebook_arr = as_arr(key_codebook);
        const array& value_indices_arr = as_arr(value_indices);
        const array& value_codebook_arr = as_arr(value_codebook);

        if (query_rot_arr.shape(-1) != dim || query_proj_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 128 || value_codebook_arr.shape(0) != 256) return 1;

        auto& pass1 = get_turboquant_attention_q8_d256_packed_keys_2pass_1_kernel();
        auto pass1_outputs = pass1(
            {
                query_rot_arr,
                query_proj_arr,
                key_bytes_arr,
                slot_scales_arr,
                key_codebook_arr,
                value_indices_arr,
                value_codebook_arr,
            },
            {{(int)n_rows, (int)blocks, (int)dim}, {(int)n_rows, (int)blocks}, {(int)n_rows, (int)blocks}},
            {float32, float32, float32},
            {32 * (int)kv_heads, (int)((q_heads / kv_heads) * (n_rows / q_heads)), (int)blocks},
            {32, (int)(q_heads / kv_heads), 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"blocks", (int)blocks},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
                {"attn_scale_bits", (int)attn_scale_bits},
            },
            std::nullopt, false, {}
        );

        auto& pass2 = get_turboquant_attention_q8_d256_2pass_2_kernel();
        auto pass2_outputs = pass2(
            {pass1_outputs[0], pass1_outputs[1], pass1_outputs[2]},
            {{(int)n_rows, (int)dim}},
            {float32},
            {1024, (int)n_rows, 1},
            {1024, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"blocks", (int)blocks},
            },
            std::nullopt, false, {}
        );

        new (out->buf) array(pass2_outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_packed_keys_2pass", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_packed_keys_2pass", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_attention_q8_d256_packed_keys_dense_values_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_bytes,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;
    const uint32_t blocks = turboquant_q8_2pass_blocks_override_or(32u);

    if (n_rows == 0 || n_seq < 1024 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& query_proj_arr = as_arr(query_proj);
        const array& key_bytes_arr = as_arr(key_bytes);
        const array& slot_scales_arr = as_arr(slot_scales);
        const array& key_codebook_arr = as_arr(key_codebook);
        const array& value_dense_arr = as_arr(value_dense);

        if (query_rot_arr.shape(-1) != dim || query_proj_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 128) return 1;

        auto& pass1 = get_turboquant_attention_q8_d256_packed_keys_dense_values_2pass_1_kernel();
        auto pass1_outputs = pass1(
            {
                query_rot_arr,
                query_proj_arr,
                key_bytes_arr,
                slot_scales_arr,
                key_codebook_arr,
                value_dense_arr,
            },
            {{(int)n_rows, (int)blocks, (int)dim}, {(int)n_rows, (int)blocks}, {(int)n_rows, (int)blocks}},
            {float32, float32, float32},
            {32 * (int)kv_heads, (int)((q_heads / kv_heads) * (n_rows / q_heads)), (int)blocks},
            {32, (int)(q_heads / kv_heads), 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"blocks", (int)blocks},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
                {"attn_scale_bits", (int)attn_scale_bits},
            },
            std::nullopt, false, {}
        );

        auto& pass2 = get_turboquant_attention_q8_d256_2pass_2_kernel();
        auto pass2_outputs = pass2(
            {pass1_outputs[0], pass1_outputs[1], pass1_outputs[2]},
            {{(int)n_rows, (int)dim}},
            {float32},
            {1024, (int)n_rows, 1},
            {1024, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"blocks", (int)blocks},
            },
            std::nullopt, false, {}
        );

        new (out->buf) array(pass2_outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_packed_keys_dense_values_2pass", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_packed_keys_dense_values_2pass", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;
    const uint32_t blocks = turboquant_q8_2pass_blocks_override_or(32u);

    if (n_rows == 0 || n_seq < 1024 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& key_indices_arr = as_arr(key_indices);
        const array& slot_scales_arr = as_arr(slot_scales);
        const array& key_codebook_arr = as_arr(key_codebook);
        const array& value_dense_arr = as_arr(value_dense);

        if (query_rot_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 256) return 1;

        auto& pass1 = get_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_1_kernel();
        auto pass1_outputs = pass1(
            {
                query_rot_arr,
                key_indices_arr,
                slot_scales_arr,
                key_codebook_arr,
                value_dense_arr,
            },
            {{(int)n_rows, (int)blocks, (int)dim}, {(int)n_rows, (int)blocks}, {(int)n_rows, (int)blocks}},
            {float32, float32, float32},
            {32 * (int)kv_heads, (int)((q_heads / kv_heads) * (n_rows / q_heads)), (int)blocks},
            {32, (int)(q_heads / kv_heads), 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"blocks", (int)blocks},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
                {"attn_scale_bits", (int)attn_scale_bits},
            },
            std::nullopt, false, {}
        );

        auto& pass2 = get_turboquant_attention_q8_d256_2pass_2_kernel();
        auto pass2_outputs = pass2(
            {pass1_outputs[0], pass1_outputs[1], pass1_outputs[2]},
            {{(int)n_rows, (int)dim}},
            {float32},
            {1024, (int)n_rows, 1},
            {1024, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"blocks", (int)blocks},
            },
            std::nullopt, false, {}
        );

        new (out->buf) array(pass2_outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_fullbyte_dense_values_2pass", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_fullbyte_dense_values_2pass", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_state(
    mlx_inline_array*       out_partials,
    mlx_inline_array*       out_sums,
    mlx_inline_array*       out_maxs,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;
    const uint32_t blocks = turboquant_q8_2pass_blocks_override_or(32u);

    if (n_rows == 0 || n_seq < 1024 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& key_indices_arr = as_arr(key_indices);
        const array& slot_scales_arr = as_arr(slot_scales);
        const array& key_codebook_arr = as_arr(key_codebook);
        const array& value_dense_arr = as_arr(value_dense);

        if (query_rot_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 256) return 1;

        auto& pass1 = get_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_1_kernel();
        auto pass1_outputs = pass1(
            {query_rot_arr, key_indices_arr, slot_scales_arr, key_codebook_arr, value_dense_arr},
            {{(int)n_rows, (int)blocks, (int)dim}, {(int)n_rows, (int)blocks}, {(int)n_rows, (int)blocks}},
            {float32, float32, float32},
            {32 * (int)kv_heads, (int)((q_heads / kv_heads) * (n_rows / q_heads)), (int)blocks},
            {32, (int)(q_heads / kv_heads), 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"blocks", (int)blocks},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
                {"attn_scale_bits", (int)attn_scale_bits},
            },
            std::nullopt, false, {}
        );

        new (out_partials->buf) array(pass1_outputs[0]);
        new (out_sums->buf) array(pass1_outputs[1]);
        new (out_maxs->buf) array(pass1_outputs[2]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_fullbyte_dense_values_2pass_state", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_fullbyte_dense_values_2pass_state", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_pass1(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;
    const uint32_t blocks = turboquant_q8_2pass_blocks_override_or(32u);

    if (n_rows == 0 || n_seq < 1024 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& key_indices_arr = as_arr(key_indices);
        const array& slot_scales_arr = as_arr(slot_scales);
        const array& key_codebook_arr = as_arr(key_codebook);
        const array& value_dense_arr = as_arr(value_dense);

        if (query_rot_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 256) return 1;

        auto& pass1 = get_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_1_kernel();
        auto pass1_outputs = pass1(
            {
                query_rot_arr,
                key_indices_arr,
                slot_scales_arr,
                key_codebook_arr,
                value_dense_arr,
            },
            {{(int)n_rows, (int)blocks, (int)dim}, {(int)n_rows, (int)blocks}, {(int)n_rows, (int)blocks}},
            {float32, float32, float32},
            {32 * (int)kv_heads, (int)((q_heads / kv_heads) * (n_rows / q_heads)), (int)blocks},
            {32, (int)(q_heads / kv_heads), 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"blocks", (int)blocks},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
                {"attn_scale_bits", (int)attn_scale_bits},
            },
            std::nullopt, false, {}
        );

        new (out->buf) array(pass1_outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_fullbyte_dense_values_2pass_pass1", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_fullbyte_dense_values_2pass_pass1", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_attention_q8_d256_pass2_merge(
    mlx_inline_array*       out,
    const mlx_inline_array* partials,
    const mlx_inline_array* sums,
    const mlx_inline_array* maxs,
    uint32_t                n_rows,
    uint32_t                blocks)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;
    if (n_rows == 0 || blocks == 0 || (blocks % 32u) != 0u) return 1;

    try {
        const array& partials_arr = as_arr(partials);
        const array& sums_arr = as_arr(sums);
        const array& maxs_arr = as_arr(maxs);
        if (partials_arr.shape(-1) != dim) return 1;

        auto& pass2 = get_turboquant_attention_q8_d256_2pass_2_kernel();
        auto pass2_outputs = pass2(
            {partials_arr, sums_arr, maxs_arr},
            {{(int)n_rows, (int)dim}},
            {float32},
            {1024, (int)n_rows, 1},
            {1024, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"blocks", (int)blocks},
            },
            std::nullopt, false, {}
        );
        new (out->buf) array(pass2_outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_pass2_merge", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_pass2_merge", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_localsoftmax(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;
    const uint32_t blocks = turboquant_q8_2pass_blocks_override_or(32u);

    if (n_rows == 0 || n_seq < 1024 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& key_indices_arr = as_arr(key_indices);
        const array& slot_scales_arr = as_arr(slot_scales);
        const array& key_codebook_arr = as_arr(key_codebook);
        const array& value_dense_arr = as_arr(value_dense);

        if (query_rot_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 256) return 1;

        auto& pass1 = get_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_1_localsoftmax_kernel();
        auto pass1_outputs = pass1(
            {query_rot_arr, key_indices_arr, slot_scales_arr, key_codebook_arr, value_dense_arr},
            {{(int)n_rows, (int)blocks, (int)dim}, {(int)n_rows, (int)blocks}, {(int)n_rows, (int)blocks}},
            {float32, float32, float32},
            {32 * (int)kv_heads, (int)((q_heads / kv_heads) * (n_rows / q_heads)), (int)blocks},
            {32, (int)(q_heads / kv_heads), 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"blocks", (int)blocks},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
                {"attn_scale_bits", (int)attn_scale_bits},
            },
            std::nullopt, false, {}
        );

        auto& pass2 = get_turboquant_attention_q8_d256_2pass_2_kernel();
        auto pass2_outputs = pass2(
            {pass1_outputs[0], pass1_outputs[1], pass1_outputs[2]},
            {{(int)n_rows, (int)dim}},
            {float32},
            {1024, (int)n_rows, 1},
            {1024, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"blocks", (int)blocks},
            },
            std::nullopt, false, {}
        );

        new (out->buf) array(pass2_outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_fullbyte_dense_values_2pass_localsoftmax", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_fullbyte_dense_values_2pass_localsoftmax", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_attention_q8_d256_packed_kv_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* kv_bytes,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;
    const uint32_t blocks = turboquant_q8_2pass_blocks_override_or(32u);

    if (n_rows == 0 || n_seq < 1024 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& query_proj_arr = as_arr(query_proj);
        const array& kv_bytes_arr = as_arr(kv_bytes);
        const array& slot_scales_arr = as_arr(slot_scales);
        const array& key_codebook_arr = as_arr(key_codebook);
        const array& value_codebook_arr = as_arr(value_codebook);

        if (query_rot_arr.shape(-1) != dim || query_proj_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 128 || value_codebook_arr.shape(0) != 256) return 1;

        auto& pass1 = get_turboquant_attention_q8_d256_packed_kv_2pass_1_kernel();
        auto pass1_outputs = pass1(
            {
                query_rot_arr,
                query_proj_arr,
                kv_bytes_arr,
                slot_scales_arr,
                key_codebook_arr,
                value_codebook_arr,
            },
            {{(int)n_rows, (int)blocks, (int)dim}, {(int)n_rows, (int)blocks}, {(int)n_rows, (int)blocks}},
            {float32, float32, float32},
            {32 * (int)kv_heads, (int)((q_heads / kv_heads) * (n_rows / q_heads)), (int)blocks},
            {32, (int)(q_heads / kv_heads), 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"blocks", (int)blocks},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
                {"attn_scale_bits", (int)attn_scale_bits},
            },
            std::nullopt, false, {}
        );

        auto& pass2 = get_turboquant_attention_q8_d256_2pass_2_kernel();
        auto pass2_outputs = pass2(
            {pass1_outputs[0], pass1_outputs[1], pass1_outputs[2]},
            {{(int)n_rows, (int)dim}},
            {float32},
            {1024, (int)n_rows, 1},
            {1024, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"blocks", (int)blocks},
            },
            std::nullopt, false, {}
        );

        new (out->buf) array(pass2_outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_packed_kv_2pass", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_packed_kv_2pass", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* kv_bytes,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;
    const uint32_t blocks = turboquant_q8_2pass_blocks_override_or(32u);

    if (n_rows == 0 || n_seq < 1024 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& query_proj_arr = as_arr(query_proj);
        const array& kv_bytes_arr = as_arr(kv_bytes);
        const array& slot_scales_arr = as_arr(slot_scales);
        const array& key_codebook_arr = as_arr(key_codebook);
        const array& value_dense_arr = as_arr(value_dense);

        if (query_rot_arr.shape(-1) != dim || query_proj_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 128) return 1;

        auto& pass1 = get_turboquant_attention_q8_d256_packed_kv_dense_values_2pass_1_kernel();
        auto pass1_outputs = pass1(
            {
                query_rot_arr,
                query_proj_arr,
                kv_bytes_arr,
                slot_scales_arr,
                key_codebook_arr,
                value_dense_arr,
            },
            {{(int)n_rows, (int)blocks, (int)dim}, {(int)n_rows, (int)blocks}, {(int)n_rows, (int)blocks}},
            {float32, float32, float32},
            {32 * (int)kv_heads, (int)((q_heads / kv_heads) * (n_rows / q_heads)), (int)blocks},
            {32, (int)(q_heads / kv_heads), 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"blocks", (int)blocks},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
                {"attn_scale_bits", (int)attn_scale_bits},
            },
            std::nullopt, false, {}
        );

        auto& pass2 = get_turboquant_attention_q8_d256_2pass_2_kernel();
        auto pass2_outputs = pass2(
            {pass1_outputs[0], pass1_outputs[1], pass1_outputs[2]},
            {{(int)n_rows, (int)dim}},
            {float32},
            {1024, (int)n_rows, 1},
            {1024, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"blocks", (int)blocks},
            },
            std::nullopt, false, {}
        );

        new (out->buf) array(pass2_outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_packed_kv_dense_values_2pass", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d256_packed_kv_dense_values_2pass", "unknown C++ exception"); return 1; }
}

} // extern "C"
