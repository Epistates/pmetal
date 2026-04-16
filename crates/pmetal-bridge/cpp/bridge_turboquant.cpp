// TurboQuant fused Metal kernels and bridge functions.
// Extracted from bridge.cpp for maintainability.

#include "bridge_internal.h"
#include <cstdlib>

static inline uint32_t turboquant_q8_2pass_blocks_override_or(uint32_t fallback) {
    const char* env = std::getenv("PMETAL_TQ_Q8_2PASS_BLOCKS");
    if (!env || !*env) return fallback;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(env, &end, 10);
    if (end == env || *end != '\0') return fallback;
    if (parsed < 32ul) parsed = 32ul;
    if (parsed > 1024ul) parsed = 1024ul;
    parsed = (parsed / 32ul) * 32ul;
    return parsed ? static_cast<uint32_t>(parsed) : fallback;
}

// ── TurboQuant fused Metal kernel sources ───────────────────────────────────
//
// ENCODE: for each (row, dim) pair, find the nearest centroid in the codebook.
// The input is already normalised onto the unit sphere AND rotated.
// Input dtype is f32 (post-normalise + matmul path ensures f32).
//
// Grid: (D, N)  — x = dim index, y = row index.
// Threadgroup: (min(D,256), 1).
//
// n_centroids is a runtime constant, so the same kernel handles both low-bit
// and q8 codebooks. Smaller codebooks still unroll well, while larger ones
// avoid falling back to an ops graph.
//
// This replaces the ops chain:
//   expand_dims(rotated, -1) → subtract(codebook) → square → argmin
// which allocates a [N, D, C] intermediate (409 KB for typical inference step).
static const char* TURBOQUANT_ENCODE_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint d   = thread_position_in_grid.x;
    if (d >= dim || row >= n_rows) return;

    float x = input[row * dim + d];

    // Nearest-centroid search over the codebook.  This stays on-GPU even for
    // larger q8 codebooks, which is still preferable to materializing the
    // expand_dims/subtract/square/argmin fallback graph.
    float best_dist = (x - codebook[0]) * (x - codebook[0]);
    uint  best_idx  = 0u;
    for (uint c = 1u; c < n_centroids; ++c) {
        float dist = (x - codebook[c]) * (x - codebook[c]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx  = c;
        }
    }

    indices[row * dim + d] = best_idx;
)";

// DECODE: for each (row, dim) pair, look up the centroid.
// Grid: (D, N), threadgroup: (min(D,256), 1).
// Output is f32 in the rotated domain.  The caller is responsible for the
// subsequent matmul(rotation) to get back to the original input space.
// Norm rescaling is done OUTSIDE the kernel (by the matmul + multiply in Rust)
// to keep the kernel minimal and avoid a dependency on the norms shape layout.
static const char* TURBOQUANT_DECODE_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint d   = thread_position_in_grid.x;
    if (d >= dim || row >= n_rows) return;

    output[row * dim + d] = codebook[indices[row * dim + d]];
)";

// SCORE: for each (row, seq) pair, compute the TurboQuant key score directly
// from compressed centroids + QJL signs without materializing a [row, seq, dim]
// decoded tensor.
//
// The key cache is stored in score-friendly transposed views:
//   indices:   [KvRows, D, S_cap]
//   qjl_signs: [KvRows, ceil(D/32), S_cap]
// so threads that walk `seq` read contiguous memory instead of striding by D.
static const char* TURBOQUANT_SCORE_SOURCE = R"(
    constexpr uint kTileSeq = 64u;
    constexpr uint kMaxDim = 512u;
    threadgroup float shared_query_rot[kMaxDim];
    threadgroup float shared_query_proj[kMaxDim];

    uint row = thread_position_in_grid.y;
    if (row >= n_rows || dim > kMaxDim) return;

    uint lane = thread_position_in_threadgroup.x;
    uint query_base = row * dim;
    for (uint d = lane; d < dim; d += kTileSeq) {
        shared_query_rot[d] = query_rot[query_base + d];
        shared_query_proj[d] = query_proj[query_base + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint seq = thread_position_in_grid.x;
    if (seq >= n_seq) return;

    uint groups = q_heads / kv_heads;
    uint batch = row / q_heads;
    uint q_head = row % q_heads;
    uint kv_row = batch * kv_heads + (q_head / groups);

    uint scalar_idx = row * n_seq + seq;
    uint kv_scalar_idx = kv_row * cache_seq_capacity + seq;

    float mse = 0.0f;
    float qjl = 0.0f;
    for (uint d = 0u; d < dim; ++d) {
        float q_rot = shared_query_rot[d];
        float q_proj_val = shared_query_proj[d];
        uint idx = (uint)indices[(kv_row * dim + d) * cache_seq_capacity + seq];
        uint sign_word = qjl_signs[(kv_row * qjl_words + (d >> 5)) * cache_seq_capacity + seq];
        float q_sign = ((sign_word >> (d & 31u)) & 1u) == 0 ? -1.0f : 1.0f;
        mse += q_rot * codebook[idx];
        qjl += q_proj_val * q_sign;
    }
    float residual = residual_norms[kv_scalar_idx];
    float score = mse;
    if (residual > 0.0f) {
        score += qjl * residual * (1.2533141373155003f / float(dim));
    }
    float attn_scale = as_type<float>((uint)attn_scale_bits);
    output[scalar_idx] = norms[kv_scalar_idx] * score * attn_scale;
)";

// Specialized q8 score kernel for D=256 on the seq-major transposed cache layout.
// This keeps the generic score kernel's row-and-seq tiled execution model so one
// threadgroup reuses the query row across a 64-token seq tile, but it specializes
// the inner loop for q8 keys (128-key codebook, 8 QJL sign words, 256 dims).
static const char* TURBOQUANT_SCORE_Q8_D256_SOURCE = R"(
    constexpr uint kDim = 256u;
    constexpr uint kTileSeq = 64u;
    constexpr float kQjlConst = 1.2533141373155003f / 256.0f;
    threadgroup float shared_query_rot[kDim];
    threadgroup float shared_query_proj[kDim];
    threadgroup float shared_k_codebook[128];

    uint row = thread_position_in_grid.y;
    uint seq = thread_position_in_grid.x;
    if (row >= n_rows || seq >= n_seq) return;

    uint lane = thread_position_in_threadgroup.x;
    uint query_base = row * kDim;
    for (uint d = lane; d < kDim; d += kTileSeq) {
        shared_query_rot[d] = query_rot[query_base + d];
        shared_query_proj[d] = query_proj[query_base + d];
    }
    for (uint c = lane; c < 128u; c += kTileSeq) {
        shared_k_codebook[c] = codebook[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint groups = q_heads / kv_heads;
    uint batch = row / q_heads;
    uint q_head = row % q_heads;
    uint kv_head = q_head / groups;
    uint kv_row = batch * kv_heads + kv_head;
    float attn_scale = as_type<float>((uint)attn_scale_bits);

    uint scalar_idx = kv_row * cache_seq_capacity + seq;
    float key_norm = norms[scalar_idx];
    float residual_scale = residual_norms[scalar_idx] * kQjlConst;

    float score_part = 0.0f;
    for (uint d0 = 0u; d0 < kDim; d0 += 8u) {
        uint key_base = (kv_row * kDim + d0) * cache_seq_capacity + seq;
        uint sign_word = qjl_signs[(kv_row * 8u + (d0 >> 5u)) * cache_seq_capacity + seq];
        uint bit_base = d0 & 31u;
        float qrot0 = attn_scale * shared_query_rot[d0 + 0u];
        float qrot1 = attn_scale * shared_query_rot[d0 + 1u];
        float qrot2 = attn_scale * shared_query_rot[d0 + 2u];
        float qrot3 = attn_scale * shared_query_rot[d0 + 3u];
        float qrot4 = attn_scale * shared_query_rot[d0 + 4u];
        float qrot5 = attn_scale * shared_query_rot[d0 + 5u];
        float qrot6 = attn_scale * shared_query_rot[d0 + 6u];
        float qrot7 = attn_scale * shared_query_rot[d0 + 7u];
        float qproj0 = attn_scale * shared_query_proj[d0 + 0u];
        float qproj1 = attn_scale * shared_query_proj[d0 + 1u];
        float qproj2 = attn_scale * shared_query_proj[d0 + 2u];
        float qproj3 = attn_scale * shared_query_proj[d0 + 3u];
        float qproj4 = attn_scale * shared_query_proj[d0 + 4u];
        float qproj5 = attn_scale * shared_query_proj[d0 + 5u];
        float qproj6 = attn_scale * shared_query_proj[d0 + 6u];
        float qproj7 = attn_scale * shared_query_proj[d0 + 7u];
        score_part += qrot0 * shared_k_codebook[(uint)indices[key_base + 0u * cache_seq_capacity]];
        score_part += qrot1 * shared_k_codebook[(uint)indices[key_base + 1u * cache_seq_capacity]];
        score_part += qrot2 * shared_k_codebook[(uint)indices[key_base + 2u * cache_seq_capacity]];
        score_part += qrot3 * shared_k_codebook[(uint)indices[key_base + 3u * cache_seq_capacity]];
        score_part += qrot4 * shared_k_codebook[(uint)indices[key_base + 4u * cache_seq_capacity]];
        score_part += qrot5 * shared_k_codebook[(uint)indices[key_base + 5u * cache_seq_capacity]];
        score_part += qrot6 * shared_k_codebook[(uint)indices[key_base + 6u * cache_seq_capacity]];
        score_part += qrot7 * shared_k_codebook[(uint)indices[key_base + 7u * cache_seq_capacity]];
        score_part += residual_scale * qproj0 * ((((sign_word >> (bit_base + 0u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj1 * ((((sign_word >> (bit_base + 1u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj2 * ((((sign_word >> (bit_base + 2u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj3 * ((((sign_word >> (bit_base + 3u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj4 * ((((sign_word >> (bit_base + 4u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj5 * ((((sign_word >> (bit_base + 5u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj6 * ((((sign_word >> (bit_base + 6u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part += residual_scale * qproj7 * ((((sign_word >> (bit_base + 7u)) & 1u) == 0u) ? -1.0f : 1.0f);
    }

    output[row * n_seq + seq] = key_norm * score_part;
)";

// MIXED_SCORE: combine regular and outlier TurboQuant key contributions in one
// launch instead of scoring the two subspaces independently and adding later.
static const char* TURBOQUANT_MIXED_SCORE_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint seq = thread_position_in_grid.x;
    if (seq >= n_seq || row >= n_rows) return;

    uint groups = q_heads / kv_heads;
    uint batch = row / q_heads;
    uint q_head = row % q_heads;
    uint kv_row = batch * kv_heads + (q_head / groups);
    uint scalar_idx = row * n_seq + seq;
    uint kv_scalar_idx = kv_row * cache_seq_capacity + seq;

    float regular_mse = 0.0f;
    float regular_qjl = 0.0f;
    uint regular_query_base = row * regular_dim;
    uint regular_cache_base = (kv_row * cache_seq_capacity + seq) * regular_dim;
    uint regular_qjl_base = (kv_row * cache_seq_capacity + seq) * regular_qjl_words;
    for (uint d = 0u; d < regular_dim; ++d) {
        float q_rot = regular_query_rot[regular_query_base + d];
        float q_proj_val = regular_query_proj[regular_query_base + d];
        uint idx = (uint)regular_indices[regular_cache_base + d];
        uint sign_word = regular_qjl_signs[regular_qjl_base + (d >> 5)];
        float q_sign = ((sign_word >> (d & 31u)) & 1u) == 0 ? -1.0f : 1.0f;
        regular_mse += q_rot * regular_codebook[idx];
        regular_qjl += q_proj_val * q_sign;
    }

    float outlier_mse = 0.0f;
    float outlier_qjl = 0.0f;
    uint outlier_query_base = row * outlier_dim;
    uint outlier_cache_base = (kv_row * cache_seq_capacity + seq) * outlier_dim;
    uint outlier_qjl_base = (kv_row * cache_seq_capacity + seq) * outlier_qjl_words;
    for (uint d = 0u; d < outlier_dim; ++d) {
        float q_rot = outlier_query_rot[outlier_query_base + d];
        float q_proj_val = outlier_query_proj[outlier_query_base + d];
        uint idx = (uint)outlier_indices[outlier_cache_base + d];
        uint sign_word = outlier_qjl_signs[outlier_qjl_base + (d >> 5)];
        float q_sign = ((sign_word >> (d & 31u)) & 1u) == 0 ? -1.0f : 1.0f;
        outlier_mse += q_rot * outlier_codebook[idx];
        outlier_qjl += q_proj_val * q_sign;
    }

    float regular_score = regular_mse;
    float regular_residual = regular_residual_norms[kv_scalar_idx];
    if (regular_residual > 0.0f) {
        regular_score += regular_qjl * regular_residual * (1.2533141373155003f / float(regular_dim));
    }
    regular_score *= regular_norms[kv_scalar_idx];

    float outlier_score = outlier_mse;
    float outlier_residual = outlier_residual_norms[kv_scalar_idx];
    if (outlier_residual > 0.0f) {
        outlier_score += outlier_qjl * outlier_residual * (1.2533141373155003f / float(outlier_dim));
    }
    outlier_score *= outlier_norms[kv_scalar_idx];

    float attn_scale = as_type<float>((uint)attn_scale_bits);
    output[scalar_idx] = (regular_score + outlier_score) * attn_scale;
)";

static const char* TURBOQUANT_PACK_SIGN_BITS_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint packed_idx = thread_position_in_grid.x;
    if (row >= n_rows || packed_idx >= packed_dim) return;

    uint base = row * dim;
    uint start = packed_idx * 32u;
    uint word = 0u;
    for (uint bit = 0u; bit < 32u; ++bit) {
        uint d = start + bit;
        if (d < dim && projected[base + d] >= 0.0f) {
            word |= (1u << bit);
        }
    }
    output[row * packed_dim + packed_idx] = word;
)";

static const char* TURBOQUANT_UNPACK_SIGN_BITS_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint d = thread_position_in_grid.x;
    if (row >= n_rows || d >= dim) return;

    uint word = packed[row * packed_dim + (d >> 5)];
    output[row * dim + d] = ((word >> (d & 31u)) & 1u) == 0 ? -1.0f : 1.0f;
)";

static const char* TURBOQUANT_PACK_Q8_KEYBYTES_SOURCE = R"(
    uint seq = thread_position_in_grid.x;
    uint d = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;
    if (row >= n_rows || d >= dim || seq >= cache_seq_capacity) return;

    uint index_base = ((row * dim) + d) * cache_seq_capacity + seq;
    uint sign_word_idx = ((row * packed_dim) + (d >> 5)) * cache_seq_capacity + seq;
    uchar index = indices[index_base] & 0x7fu;
    uint sign_word = qjl_signs[sign_word_idx];
    uchar sign_bit = ((sign_word >> (d & 31u)) & 1u) != 0u ? 0x80u : 0x00u;
    output[index_base] = index | sign_bit;
)";

static const char* TURBOQUANT_PACK_Q8_KEYBYTES_SEQ_SOURCE = R"(
    uint seq = thread_position_in_grid.x;
    uint d = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;
    if (row >= n_rows || d >= dim || seq >= cache_seq_capacity) return;

    uint index_base = ((row * dim) + d) * cache_seq_capacity + seq;
    uint sign_word_idx = ((row * packed_dim) + (d >> 5)) * cache_seq_capacity + seq;
    uchar index = indices[index_base] & 0x7fu;
    uint sign_word = qjl_signs[sign_word_idx];
    uchar sign_bit = ((sign_word >> (d & 31u)) & 1u) != 0u ? 0x80u : 0x00u;
    uint output_idx = ((row * cache_seq_capacity) + seq) * dim + d;
    output[output_idx] = index | sign_bit;
)";

static const char* TURBOQUANT_PACK_Q8_KVBYTES_SEQ_SOURCE = R"(
    uint seq = thread_position_in_grid.x;
    uint d = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;
    if (row >= n_rows || d >= dim || seq >= cache_seq_capacity) return;

    uint index_base = ((row * dim) + d) * cache_seq_capacity + seq;
    uint sign_word_idx = ((row * packed_dim) + (d >> 5)) * cache_seq_capacity + seq;
    uchar key_index = indices[index_base] & 0x7fu;
    uint sign_word = qjl_signs[sign_word_idx];
    uchar sign_bit = ((sign_word >> (d & 31u)) & 1u) != 0u ? 0x80u : 0x00u;
    uchar key_byte = key_index | sign_bit;
    uint output_idx = ((row * cache_seq_capacity) + seq) * dim + d;
    ushort value_idx = (ushort)value_indices[output_idx];
    output[output_idx] = ((value_idx & 0xffu) << 8) | (ushort)key_byte;
)";

// WEIGHTED_DECODE: aggregate value centroids directly in the rotated domain.
// This avoids materializing a [row, seq, dim] decoded value tensor before the
// attention reduction.
//
// The value cache is stored in a decode-friendly transposed view:
//   indices: [KvRows, D, S_cap]
// so threads that walk `seq` read contiguous memory instead of striding by D.
static const char* TURBOQUANT_WEIGHTED_DECODE_SOURCE = R"(
    constexpr uint kTileDims = 8u;
    constexpr uint kMaxCentroids = 512u;
    threadgroup float shared_codebook[kMaxCentroids];

    if (n_centroids > kMaxCentroids) return;
    uint lane = thread_index_in_simdgroup;
    for (uint c = lane; c < n_centroids; c += 32u) {
        shared_codebook[c] = codebook[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile = thread_position_in_grid.y;
    uint row = tile / ((dim + kTileDims - 1u) / kTileDims);
    if (row >= n_rows) return;
    uint tile_idx = tile % ((dim + kTileDims - 1u) / kTileDims);
    uint d0 = tile_idx * kTileDims;

    uint groups = q_heads / kv_heads;
    uint batch = row / q_heads;
    uint q_head = row % q_heads;
    uint kv_row = batch * kv_heads + (q_head / groups);

    uint scalar_base = row * n_seq;
    uint kv_scalar_base = kv_row * cache_seq_capacity;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    for (uint seq = lane; seq < n_seq; seq += 32u) {
        uint scalar_idx = scalar_base + seq;
        uint kv_scalar_idx = kv_scalar_base + seq;
        float scalar = weights[scalar_idx] * norms[kv_scalar_idx];
        uint cache_base = (kv_row * dim + d0) * cache_seq_capacity + seq;
        if (d0 + 0u < dim) {
            acc0 += scalar * shared_codebook[(uint)indices[cache_base + 0u * cache_seq_capacity]];
        }
        if (d0 + 1u < dim) {
            acc1 += scalar * shared_codebook[(uint)indices[cache_base + 1u * cache_seq_capacity]];
        }
        if (d0 + 2u < dim) {
            acc2 += scalar * shared_codebook[(uint)indices[cache_base + 2u * cache_seq_capacity]];
        }
        if (d0 + 3u < dim) {
            acc3 += scalar * shared_codebook[(uint)indices[cache_base + 3u * cache_seq_capacity]];
        }
        if (d0 + 4u < dim) {
            acc4 += scalar * shared_codebook[(uint)indices[cache_base + 4u * cache_seq_capacity]];
        }
        if (d0 + 5u < dim) {
            acc5 += scalar * shared_codebook[(uint)indices[cache_base + 5u * cache_seq_capacity]];
        }
        if (d0 + 6u < dim) {
            acc6 += scalar * shared_codebook[(uint)indices[cache_base + 6u * cache_seq_capacity]];
        }
        if (d0 + 7u < dim) {
            acc7 += scalar * shared_codebook[(uint)indices[cache_base + 7u * cache_seq_capacity]];
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);
    if (thread_index_in_simdgroup == 0) {
        uint out_base = row * dim + d0;
        if (d0 + 0u < dim) output[out_base + 0u] = acc0;
        if (d0 + 1u < dim) output[out_base + 1u] = acc1;
        if (d0 + 2u < dim) output[out_base + 2u] = acc2;
        if (d0 + 3u < dim) output[out_base + 3u] = acc3;
        if (d0 + 4u < dim) output[out_base + 4u] = acc4;
        if (d0 + 5u < dim) output[out_base + 5u] = acc5;
        if (d0 + 6u < dim) output[out_base + 6u] = acc6;
        if (d0 + 7u < dim) output[out_base + 7u] = acc7;
    }
)";

// Specialized q8 long-context decode primitive for D=256/V=256.
// This follows the same 2-pass structure as the D128 primitive, but widens the
// lane-local vector width to cover the real 35B Qwen3.5 full-attention shape.
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
        uint scale_base = scalar_idx * 3u;
        float key_norm = slot_scales[scale_base + 0u];
        float residual_scale = slot_scales[scale_base + 1u] * kQjlConst;
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
        uint scale_base = scalar_idx * 3u;
        float key_norm = slot_scales[scale_base + 0u];
        float residual_scale = slot_scales[scale_base + 1u] * kQjlConst;
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
        float key_norm = slot_scales[scalar_idx * 3u + 0u];
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
        float key_norm = slot_scales[scalar_idx * 3u + 0u];
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
        float key_norm = slot_scales[scalar_idx * 3u + 0u];
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

static const char* TURBOQUANT_SCORE_Q8_D256_FULLBYTE_SOURCE = R"(
    constexpr uint kDim = 256u;
    constexpr uint kKeyCentroids = 256u;
    constexpr uint kTileSeq = 64u;
    threadgroup float shared_query_rot[kDim];
    threadgroup float shared_k_codebook[kKeyCentroids];

    uint row = threadgroup_position_in_grid.y;
    uint seq = thread_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= n_rows || seq >= n_seq) return;

    uint query_base = row * kDim;
    for (uint d = lane; d < kDim; d += kTileSeq) {
        shared_query_rot[d] = query_rot[query_base + d];
    }
    for (uint c = lane; c < kKeyCentroids; c += kTileSeq) {
        shared_k_codebook[c] = key_codebook[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint groups = q_heads / kv_heads;
    uint batch = row / q_heads;
    uint q_head = row - batch * q_heads;
    uint kv_head = q_head / groups;
    uint kv_row = batch * kv_heads + kv_head;

    float attn_scale = as_type<float>((uint)attn_scale_bits);
    uint scalar_idx = kv_row * cache_seq_capacity + seq;
    float key_norm = slot_scales[scalar_idx * 3u + 0u];
    float score_part = 0.0f;
    uint key_base = (kv_row * cache_seq_capacity + seq) * kDim;
    for (uint d0 = 0u; d0 < kDim; d0 += 8u) {
        float qrot0 = attn_scale * shared_query_rot[d0 + 0u];
        float qrot1 = attn_scale * shared_query_rot[d0 + 1u];
        float qrot2 = attn_scale * shared_query_rot[d0 + 2u];
        float qrot3 = attn_scale * shared_query_rot[d0 + 3u];
        float qrot4 = attn_scale * shared_query_rot[d0 + 4u];
        float qrot5 = attn_scale * shared_query_rot[d0 + 5u];
        float qrot6 = attn_scale * shared_query_rot[d0 + 6u];
        float qrot7 = attn_scale * shared_query_rot[d0 + 7u];
        score_part += qrot0 * shared_k_codebook[(uint)key_indices[key_base + d0 + 0u]];
        score_part += qrot1 * shared_k_codebook[(uint)key_indices[key_base + d0 + 1u]];
        score_part += qrot2 * shared_k_codebook[(uint)key_indices[key_base + d0 + 2u]];
        score_part += qrot3 * shared_k_codebook[(uint)key_indices[key_base + d0 + 3u]];
        score_part += qrot4 * shared_k_codebook[(uint)key_indices[key_base + d0 + 4u]];
        score_part += qrot5 * shared_k_codebook[(uint)key_indices[key_base + d0 + 5u]];
        score_part += qrot6 * shared_k_codebook[(uint)key_indices[key_base + d0 + 6u]];
        score_part += qrot7 * shared_k_codebook[(uint)key_indices[key_base + d0 + 7u]];
    }
    output[row * n_seq + seq] = key_norm * score_part;
)";

static const char* TURBOQUANT_WEIGHTED_SUM_D256_DENSE_VALUES_SOURCE = R"(
    constexpr uint kDim = 256u;
    constexpr uint kVec = 8u;

    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    if (row >= n_rows) return;

    uint groups = q_heads / kv_heads;
    uint batch = row / q_heads;
    uint q_head = row - batch * q_heads;
    uint kv_head = q_head / groups;
    uint kv_row = batch * kv_heads + kv_head;
    uint d0 = lane * kVec;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    const device float* row_weights = weights + row * n_seq;
    for (uint seq = 0u; seq < n_seq; ++seq) {
        float w = row_weights[seq];
        uint value_base = (kv_row * cache_seq_capacity + seq) * kDim + d0;
        acc0 += w * float(value_dense[value_base + 0u]);
        acc1 += w * float(value_dense[value_base + 1u]);
        acc2 += w * float(value_dense[value_base + 2u]);
        acc3 += w * float(value_dense[value_base + 3u]);
        acc4 += w * float(value_dense[value_base + 4u]);
        acc5 += w * float(value_dense[value_base + 5u]);
        acc6 += w * float(value_dense[value_base + 6u]);
        acc7 += w * float(value_dense[value_base + 7u]);
    }

    uint out_base = row * kDim + d0;
    output[out_base + 0u] = acc0;
    output[out_base + 1u] = acc1;
    output[out_base + 2u] = acc2;
    output[out_base + 3u] = acc3;
    output[out_base + 4u] = acc4;
    output[out_base + 5u] = acc5;
    output[out_base + 6u] = acc6;
    output[out_base + 7u] = acc7;
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
        uint scale_base = scalar_idx * 3u;
        float key_norm = slot_scales[scale_base + 0u];
        float residual_scale = slot_scales[scale_base + 1u] * kQjlConst;
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
        uint scale_base = scalar_idx * 3u;
        float key_norm = slot_scales[scale_base + 0u];
        float residual_scale = slot_scales[scale_base + 1u] * kQjlConst;
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
// structure for decode-time long contexts.
static const char* TURBOQUANT_ATTENTION_Q8_D128_2PASS_1_SOURCE = R"(
    constexpr uint kDim = 128u;
    constexpr uint kVec = 4u;
    constexpr uint kQjlWords = 4u;
    constexpr float kQjlConst = 1.2533141373155003f / 128.0f;
    threadgroup float shared_k_codebook[256];
    threadgroup float shared_v_codebook[256];

    uint kv_head = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint block = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint groups = q_heads / kv_heads;
    uint row = batch * q_heads + kv_head * groups + simd_gid;
    if (row >= n_rows || block >= blocks) return;

    if (simd_gid == 0u) {
        for (uint c = lane; c < 256u; c += 32u) {
            shared_k_codebook[c] = key_codebook[c];
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
    float qproj0 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 0u];
    float qproj1 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 1u];
    float qproj2 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 2u];
    float qproj3 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 3u];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint seq = block; seq < n_seq; seq += blocks) {
        uint scalar_idx = kv_row * cache_seq_capacity + seq;
        float key_norm = key_norms[scalar_idx];
        float residual_scale = key_residual_norms[scalar_idx] * kQjlConst;
        uint key_base = (kv_row * kDim + d0) * cache_seq_capacity + seq;
        uint value_base = (kv_row * kDim + d0) * cache_seq_capacity + seq;
        uint sign_word = key_qjl_signs[(kv_row * kQjlWords + (lane >> 3u)) * cache_seq_capacity + seq];
        uint bit_base = (lane & 7u) * 4u;
        float sign0 = ((sign_word >> (bit_base + 0u)) & 1u) == 0u ? -1.0f : 1.0f;
        float sign1 = ((sign_word >> (bit_base + 1u)) & 1u) == 0u ? -1.0f : 1.0f;
        float sign2 = ((sign_word >> (bit_base + 2u)) & 1u) == 0u ? -1.0f : 1.0f;
        float sign3 = ((sign_word >> (bit_base + 3u)) & 1u) == 0u ? -1.0f : 1.0f;

        float score_part = 0.0f;
        score_part += qrot0 * shared_k_codebook[(uint)key_indices[key_base + 0u * cache_seq_capacity]];
        score_part += qrot1 * shared_k_codebook[(uint)key_indices[key_base + 1u * cache_seq_capacity]];
        score_part += qrot2 * shared_k_codebook[(uint)key_indices[key_base + 2u * cache_seq_capacity]];
        score_part += qrot3 * shared_k_codebook[(uint)key_indices[key_base + 3u * cache_seq_capacity]];
        score_part += residual_scale * qproj0 * sign0;
        score_part += residual_scale * qproj1 * sign1;
        score_part += residual_scale * qproj2 * sign2;
        score_part += residual_scale * qproj3 * sign3;
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
)";

// Variant of the q8 2-pass primitive that consumes a q8-specific seq-major key
// byte view where low 7 bits are the centroid index and the high bit is the
// QJL sign. This reduces key-side decode bandwidth for larger-head decode.
static const char* TURBOQUANT_ATTENTION_Q8_D128_PACKED_KEYS_2PASS_1_SOURCE = R"(
    constexpr uint kDim = 128u;
    constexpr uint kVec = 4u;
    constexpr float kQjlConst = 1.2533141373155003f / 128.0f;
    threadgroup float shared_k_codebook[128];
    threadgroup float shared_v_codebook[256];

    uint kv_head = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint block = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint groups = q_heads / kv_heads;
    uint row = batch * q_heads + kv_head * groups + simd_gid;
    if (row >= n_rows || block >= blocks) return;

    if (simd_gid == 0u) {
        for (uint c = lane; c < 128u; c += 32u) {
            shared_k_codebook[c] = key_codebook[c];
        }
        for (uint c = lane; c < 256u; c += 32u) {
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
    float qproj0 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 0u];
    float qproj1 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 1u];
    float qproj2 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 2u];
    float qproj3 = as_type<float>((uint)attn_scale_bits) * query_proj[query_base + 3u];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint seq = block; seq < n_seq; seq += blocks) {
        uint scalar_idx = kv_row * cache_seq_capacity + seq;
        float key_norm = key_norms[scalar_idx];
        float residual_scale = key_residual_norms[scalar_idx] * kQjlConst;
        uint key_base = (kv_row * kDim + d0) * cache_seq_capacity + seq;
        uint value_base = (kv_row * kDim + d0) * cache_seq_capacity + seq;

        uchar key_byte0 = key_bytes[key_base + 0u * cache_seq_capacity];
        uchar key_byte1 = key_bytes[key_base + 1u * cache_seq_capacity];
        uchar key_byte2 = key_bytes[key_base + 2u * cache_seq_capacity];
        uchar key_byte3 = key_bytes[key_base + 3u * cache_seq_capacity];

        float sign0 = (key_byte0 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign1 = (key_byte1 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign2 = (key_byte2 & 0x80u) == 0u ? -1.0f : 1.0f;
        float sign3 = (key_byte3 & 0x80u) == 0u ? -1.0f : 1.0f;

        float score_part = 0.0f;
        score_part += qrot0 * shared_k_codebook[(uint)(key_byte0 & 0x7fu)];
        score_part += qrot1 * shared_k_codebook[(uint)(key_byte1 & 0x7fu)];
        score_part += qrot2 * shared_k_codebook[(uint)(key_byte2 & 0x7fu)];
        score_part += qrot3 * shared_k_codebook[(uint)(key_byte3 & 0x7fu)];
        score_part += residual_scale * qproj0 * sign0;
        score_part += residual_scale * qproj1 * sign1;
        score_part += residual_scale * qproj2 * sign2;
        score_part += residual_scale * qproj3 * sign3;
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
)";

static const char* TURBOQUANT_ATTENTION_Q8_D128_2PASS_2_SOURCE = R"(
    constexpr uint kBlocksPerSimd = 32u;
    constexpr uint kSimds = 32u;
    constexpr uint kVec = 4u;
    constexpr uint kDim = 128u;
    threadgroup float outputs[kBlocksPerSimd * kSimds];

    uint row = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    if (row >= n_rows || blocks == 0u) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

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
        partial_ptr += kBlocksPerSimd * kDim;
        max_ptr += kBlocksPerSimd;
    }

    outputs[simd_lid * kSimds + simd_gid] = acc0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc0 = simd_sum(outputs[simd_gid * kSimds + simd_lid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outputs[simd_lid * kSimds + simd_gid] = acc1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc1 = simd_sum(outputs[simd_gid * kSimds + simd_lid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outputs[simd_lid * kSimds + simd_gid] = acc2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc2 = simd_sum(outputs[simd_gid * kSimds + simd_lid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outputs[simd_lid * kSimds + simd_gid] = acc3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc3 = simd_sum(outputs[simd_gid * kSimds + simd_lid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = sum_exp_score > 0.0f ? 1.0f / sum_exp_score : 0.0f;
    if (simd_gid == 0u) {
        row_out[0u] = acc0 * inv_sum;
        row_out[1u] = acc1 * inv_sum;
        row_out[2u] = acc2 * inv_sum;
        row_out[3u] = acc3 * inv_sum;
    }
)";

// GATHER_LAST_DIM: gather selected coordinates from a flattened [row, dim]
// tensor using a fixed position list.
static const char* TURBOQUANT_GATHER_LAST_DIM_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint out_d = thread_position_in_grid.x;
    if (out_d >= out_dim || row >= n_rows) return;

    int src_d = positions[out_d];
    output[row * out_dim + out_d] = input[row * full_dim + (uint)src_d];
)";

// SCATTER_LAST_DIM: reassemble mixed regular/outlier sub-vectors into the
// original full-dimensional row layout.
static const char* TURBOQUANT_SCATTER_LAST_DIM_SOURCE = R"(
    uint row = thread_position_in_grid.y;
    uint d = thread_position_in_grid.x;
    if (d >= full_dim || row >= n_rows) return;

    float value = 0.0f;
    bool found = false;
    for (uint i = 0u; i < regular_dim; ++i) {
        if ((uint)regular_positions[i] == d) {
            value = regular[row * regular_dim + i];
            found = true;
            break;
        }
    }
    if (!found) {
        for (uint i = 0u; i < outlier_dim; ++i) {
            if ((uint)outlier_positions[i] == d) {
                value = outlier[row * outlier_dim + i];
                break;
            }
        }
    }
    output[row * full_dim + d] = value;
)";

// ── TurboQuant kernel getters ────────────────────────────────────────────────

static mlx::core::fast::CustomKernelFunction& get_turboquant_encode_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_encode",
        {"input", "codebook"},
        {"indices"},
        TURBOQUANT_ENCODE_SOURCE,
        "",    // no header
        true,  // ensure_row_contiguous
        false  // atomic_outputs
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_decode_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_decode",
        {"indices", "codebook"},
        {"output"},
        TURBOQUANT_DECODE_SOURCE,
        "",    // no header
        true,  // ensure_row_contiguous
        false  // atomic_outputs
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_score_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_score",
        {"query_rot", "query_proj", "indices", "qjl_signs", "norms", "residual_norms", "codebook"},
        {"output"},
        TURBOQUANT_SCORE_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_score_q8_d256_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_score_q8_d256",
        {"query_rot", "query_proj", "indices", "qjl_signs", "norms", "residual_norms", "codebook"},
        {"output"},
        TURBOQUANT_SCORE_Q8_D256_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_mixed_score_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_mixed_score",
        {
            "regular_query_rot",
            "regular_query_proj",
            "regular_indices",
            "regular_qjl_signs",
            "regular_norms",
            "regular_residual_norms",
            "regular_codebook",
            "outlier_query_rot",
            "outlier_query_proj",
            "outlier_indices",
            "outlier_qjl_signs",
            "outlier_norms",
            "outlier_residual_norms",
            "outlier_codebook",
        },
        {"output"},
        TURBOQUANT_MIXED_SCORE_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_weighted_decode_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_weighted_decode",
        {"weights", "indices", "norms", "codebook"},
        {"output"},
        TURBOQUANT_WEIGHTED_DECODE_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_attention_q8_d128_2pass_1_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_q8_d128_2pass_1",
        {
            "query_rot",
            "query_proj",
            "key_indices",
            "key_qjl_signs",
            "key_norms",
            "key_residual_norms",
            "key_codebook",
            "value_indices",
            "value_norms",
            "value_codebook",
        },
        {"partials", "sums", "maxs"},
        TURBOQUANT_ATTENTION_Q8_D128_2PASS_1_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_attention_q8_d128_2pass_2_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_q8_d128_2pass_2",
        {"partials", "sums", "maxs"},
        {"output"},
        TURBOQUANT_ATTENTION_Q8_D128_2PASS_2_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_pack_sign_bits_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_pack_sign_bits",
        {"projected"},
        {"output"},
        TURBOQUANT_PACK_SIGN_BITS_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_pack_q8_keybytes_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_pack_q8_keybytes",
        {"indices", "qjl_signs"},
        {"output"},
        TURBOQUANT_PACK_Q8_KEYBYTES_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_pack_q8_keybytes_seq_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_pack_q8_keybytes_seq",
        {"indices", "qjl_signs"},
        {"output"},
        TURBOQUANT_PACK_Q8_KEYBYTES_SEQ_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_pack_q8_kvbytes_seq_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_pack_q8_kvbytes_seq",
        {"indices", "qjl_signs", "value_indices"},
        {"output"},
        TURBOQUANT_PACK_Q8_KVBYTES_SEQ_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_unpack_sign_bits_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_unpack_sign_bits",
        {"packed"},
        {"output"},
        TURBOQUANT_UNPACK_SIGN_BITS_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

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

static mlx::core::fast::CustomKernelFunction& get_turboquant_score_q8_d256_fullbyte_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_score_q8_d256_fullbyte",
        {
            "query_rot",
            "key_indices",
            "slot_scales",
            "key_codebook",
        },
        {"output"},
        TURBOQUANT_SCORE_Q8_D256_FULLBYTE_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_weighted_sum_d256_dense_values_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_weighted_sum_d256_dense_values",
        {"weights", "value_dense"},
        {"output"},
        TURBOQUANT_WEIGHTED_SUM_D256_DENSE_VALUES_SOURCE,
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

static mlx::core::fast::CustomKernelFunction& get_turboquant_attention_q8_d128_packed_keys_2pass_1_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_attention_q8_d128_packed_keys_2pass_1",
        {
            "query_rot",
            "query_proj",
            "key_bytes",
            "key_norms",
            "key_residual_norms",
            "key_codebook",
            "value_indices",
            "value_norms",
            "value_codebook",
        },
        {"partials", "sums", "maxs"},
        TURBOQUANT_ATTENTION_Q8_D128_PACKED_KEYS_2PASS_1_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_gather_last_dim_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_gather_last_dim",
        {"input", "positions"},
        {"output"},
        TURBOQUANT_GATHER_LAST_DIM_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}

static mlx::core::fast::CustomKernelFunction& get_turboquant_scatter_last_dim_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_scatter_last_dim",
        {"regular", "outlier", "regular_positions", "outlier_positions"},
        {"output"},
        TURBOQUANT_SCATTER_LAST_DIM_SOURCE,
        "",
        true,
        false
    );
    return kernel;
}


extern "C" {

// ── TurboQuant C bridge functions ───────────────────────────────────────────
//
// encode: input is already normalised+rotated f32 [N, D].
//         codebook is f32 [C], C <= 16.
//         Output: indices [N, D] uint32.
//
// decode: indices [N, D] uint32.  codebook [C] f32.
//         Output: centroid values [N, D] f32 (in rotated domain, un-scaled).
//         The caller multiplies by the original L2 norms.

int mlx_inline_turboquant_encode(
    mlx_inline_array*       out_indices,
    mlx_inline_array*       /*out_norms — unused, kept for ABI symmetry*/,
    const mlx_inline_array* input,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (n_centroids == 0 || dim == 0 || n_rows == 0) return 1;

    const array& inp = as_arr(input);
    const array& cb  = as_arr(codebook);

    // Input must be f32 (normalised+rotated via matmul produces f32).
    // Codebook is always f32.
    try {
        auto& kernel = get_turboquant_encode_kernel();
        auto outputs = kernel(
            {inp, cb},
            {{(int)n_rows, (int)dim}},   // output shape: [N, D]
            {uint8},                      // output dtype: uint8
            {(int)dim, (int)n_rows, 1},  // grid (x=D, y=N, z=1)
            {(int)std::min(dim, 256u), 1, 1},
            {{"dim",         (int)dim},
             {"n_centroids", (int)n_centroids},
             {"n_rows",      (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out_indices->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_encode", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_encode", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_decode(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* /*norms — unused, kept for ABI symmetry*/,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (n_centroids == 0 || dim == 0 || n_rows == 0) return 1;

    const array& idx = as_arr(indices);
    const array& cb  = as_arr(codebook);

    try {
        auto& kernel = get_turboquant_decode_kernel();
        auto outputs = kernel(
            {idx, cb},
            {{(int)n_rows, (int)dim}},
            {float32},
            {(int)dim, (int)n_rows, 1},
            {(int)std::min(dim, 256u), 1, 1},
            {{"dim",         (int)dim},
             {"n_centroids", (int)n_centroids},
             {"n_rows",      (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_decode", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_decode", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_score(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    const mlx_inline_array* norms,
    const mlx_inline_array* residual_norms,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                qjl_words,
    uint32_t                n_centroids,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    if (n_centroids == 0 || dim == 0 || n_rows == 0 || n_seq == 0 || kv_heads == 0 || q_heads == 0) return 1;
    if (cache_seq_capacity < n_seq) return 1;
    if (q_heads % kv_heads != 0) return 1;

    try {
        auto& kernel = get_turboquant_score_kernel();
        constexpr uint32_t tg_threads = 64u;
        auto outputs = kernel(
            {as_arr(query_rot), as_arr(query_proj), as_arr(indices), as_arr(qjl_signs), as_arr(norms), as_arr(residual_norms), as_arr(codebook)},
            {{(int)n_rows, (int)n_seq}},
            {float32},
            {(int)(((n_seq + tg_threads - 1) / tg_threads) * tg_threads), (int)n_rows, 1},
            {(int)tg_threads, 1, 1},
            {{"dim",         (int)dim},
             {"qjl_words",   (int)qjl_words},
             {"n_centroids", (int)n_centroids},
             {"n_rows",      (int)n_rows},
             {"n_seq",       (int)n_seq},
             {"cache_seq_capacity", (int)cache_seq_capacity},
             {"q_heads",     (int)q_heads},
             {"kv_heads",    (int)kv_heads},
             {"attn_scale_bits",  (int)attn_scale_bits}},
            std::nullopt, false, {}
        );
        new (out_scores->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_score", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_score", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_score_q8_d256(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    const mlx_inline_array* norms,
    const mlx_inline_array* residual_norms,
    const mlx_inline_array* codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;
    constexpr uint32_t qjl_words = 8u;
    constexpr uint32_t n_centroids = 128u;

    if (n_rows == 0 || n_seq == 0 || kv_heads == 0 || q_heads == 0) return 1;
    if (cache_seq_capacity < n_seq) return 1;
    if (q_heads % kv_heads != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& query_proj_arr = as_arr(query_proj);
        const array& indices_arr = as_arr(indices);
        const array& qjl_signs_arr = as_arr(qjl_signs);
        const array& norms_arr = as_arr(norms);
        const array& residual_norms_arr = as_arr(residual_norms);
        const array& codebook_arr = as_arr(codebook);

        if (query_rot_arr.shape(-1) != dim || query_proj_arr.shape(-1) != dim) return 1;
        if (codebook_arr.shape(0) != n_centroids) return 1;

        auto& kernel = get_turboquant_score_q8_d256_kernel();
        auto outputs = kernel(
            {query_rot_arr, query_proj_arr, indices_arr, qjl_signs_arr, norms_arr, residual_norms_arr, codebook_arr},
            {{(int)n_rows, (int)n_seq}},
            {float32},
            {((int)n_seq + 63) & ~63, (int)n_rows, 1},
            {64, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
                {"attn_scale_bits", (int)attn_scale_bits},
            },
            std::nullopt, false, {}
        );
        new (out_scores->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_score_q8_d256", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_score_q8_d256", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_mixed_score(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* regular_query_rot,
    const mlx_inline_array* regular_query_proj,
    const mlx_inline_array* regular_indices,
    const mlx_inline_array* regular_qjl_signs,
    const mlx_inline_array* regular_norms,
    const mlx_inline_array* regular_residual_norms,
    const mlx_inline_array* regular_codebook,
    const mlx_inline_array* outlier_query_rot,
    const mlx_inline_array* outlier_query_proj,
    const mlx_inline_array* outlier_indices,
    const mlx_inline_array* outlier_qjl_signs,
    const mlx_inline_array* outlier_norms,
    const mlx_inline_array* outlier_residual_norms,
    const mlx_inline_array* outlier_codebook,
    uint32_t                regular_dim,
    uint32_t                regular_qjl_words,
    uint32_t                regular_n_centroids,
    uint32_t                outlier_dim,
    uint32_t                outlier_qjl_words,
    uint32_t                outlier_n_centroids,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    if (regular_n_centroids == 0 || outlier_n_centroids == 0) return 1;
    if (regular_dim == 0 || outlier_dim == 0 || n_rows == 0 || n_seq == 0) return 1;
    if (cache_seq_capacity < n_seq) return 1;
    if (kv_heads == 0 || q_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        auto& kernel = get_turboquant_mixed_score_kernel();
        auto outputs = kernel(
            {
                as_arr(regular_query_rot),
                as_arr(regular_query_proj),
                as_arr(regular_indices),
                as_arr(regular_qjl_signs),
                as_arr(regular_norms),
                as_arr(regular_residual_norms),
                as_arr(regular_codebook),
                as_arr(outlier_query_rot),
                as_arr(outlier_query_proj),
                as_arr(outlier_indices),
                as_arr(outlier_qjl_signs),
                as_arr(outlier_norms),
                as_arr(outlier_residual_norms),
                as_arr(outlier_codebook),
            },
            {{(int)n_rows, (int)n_seq}},
            {float32},
            {(int)n_seq, (int)n_rows, 1},
            {(int)std::min(n_seq, 256u), 1, 1},
            {{"regular_dim", (int)regular_dim},
             {"regular_qjl_words", (int)regular_qjl_words},
             {"outlier_dim", (int)outlier_dim},
             {"outlier_qjl_words", (int)outlier_qjl_words},
             {"n_rows",      (int)n_rows},
             {"n_seq",       (int)n_seq},
             {"cache_seq_capacity", (int)cache_seq_capacity},
             {"q_heads",     (int)q_heads},
             {"kv_heads",    (int)kv_heads},
             {"attn_scale_bits",  (int)attn_scale_bits}},
            std::nullopt, false, {}
        );
        new (out_scores->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_mixed_score", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_mixed_score", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_pack_sign_bits(
    mlx_inline_array*       out,
    const mlx_inline_array* projected,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (dim == 0 || packed_dim == 0 || n_rows == 0) return 1;

    try {
        auto& kernel = get_turboquant_pack_sign_bits_kernel();
        auto outputs = kernel(
            {as_arr(projected)},
            {{(int)n_rows, (int)packed_dim}},
            {uint32},
            {(int)packed_dim, (int)n_rows, 1},
            {(int)std::min(packed_dim, 256u), 1, 1},
            {{"dim", (int)dim},
             {"packed_dim", (int)packed_dim},
             {"n_rows", (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_pack_sign_bits", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_pack_sign_bits", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_pack_q8_keybytes(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity)
{
    using namespace mlx::core;

    if (dim == 0 || packed_dim == 0 || n_rows == 0 || cache_seq_capacity == 0) return 1;

    try {
        auto& kernel = get_turboquant_pack_q8_keybytes_kernel();
        auto outputs = kernel(
            {as_arr(indices), as_arr(qjl_signs)},
            {{(int)n_rows, (int)dim, (int)cache_seq_capacity}},
            {uint8},
            {(int)cache_seq_capacity, (int)dim, (int)n_rows},
            {(int)std::min(cache_seq_capacity, 256u), 1, 1},
            {{"dim", (int)dim},
             {"packed_dim", (int)packed_dim},
             {"n_rows", (int)n_rows},
             {"cache_seq_capacity", (int)cache_seq_capacity}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_pack_q8_keybytes", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_pack_q8_keybytes", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_pack_q8_keybytes_seq(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity)
{
    using namespace mlx::core;

    if (dim == 0 || packed_dim == 0 || n_rows == 0 || cache_seq_capacity == 0) return 1;

    try {
        auto& kernel = get_turboquant_pack_q8_keybytes_seq_kernel();
        auto outputs = kernel(
            {as_arr(indices), as_arr(qjl_signs)},
            {{(int)n_rows, (int)cache_seq_capacity, (int)dim}},
            {uint8},
            {(int)cache_seq_capacity, (int)dim, (int)n_rows},
            {(int)std::min(cache_seq_capacity, 256u), 1, 1},
            {{"dim", (int)dim},
             {"packed_dim", (int)packed_dim},
             {"n_rows", (int)n_rows},
             {"cache_seq_capacity", (int)cache_seq_capacity}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_pack_q8_keybytes_seq", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_pack_q8_keybytes_seq", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_pack_q8_kvbytes_seq(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    const mlx_inline_array* value_indices,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity)
{
    using namespace mlx::core;

    if (dim == 0 || packed_dim == 0 || n_rows == 0 || cache_seq_capacity == 0) return 1;

    try {
        auto& kernel = get_turboquant_pack_q8_kvbytes_seq_kernel();
        auto outputs = kernel(
            {as_arr(indices), as_arr(qjl_signs), as_arr(value_indices)},
            {{(int)n_rows, (int)cache_seq_capacity, (int)dim}},
            {uint16},
            {(int)cache_seq_capacity, (int)dim, (int)n_rows},
            {(int)std::min(cache_seq_capacity, 256u), 1, 1},
            {{"dim", (int)dim},
             {"packed_dim", (int)packed_dim},
             {"n_rows", (int)n_rows},
             {"cache_seq_capacity", (int)cache_seq_capacity}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_pack_q8_kvbytes_seq", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_pack_q8_kvbytes_seq", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_unpack_sign_bits(
    mlx_inline_array*       out,
    const mlx_inline_array* packed,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (dim == 0 || packed_dim == 0 || n_rows == 0) return 1;

    try {
        auto& kernel = get_turboquant_unpack_sign_bits_kernel();
        auto outputs = kernel(
            {as_arr(packed)},
            {{(int)n_rows, (int)dim}},
            {float32},
            {(int)dim, (int)n_rows, 1},
            {(int)std::min(dim, 256u), 1, 1},
            {{"dim", (int)dim},
             {"packed_dim", (int)packed_dim},
             {"n_rows", (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_unpack_sign_bits", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_unpack_sign_bits", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_signed_fwht_256_rows(
    mlx_inline_array*       out,
    const mlx_inline_array* input,
    const mlx_inline_array* left_signs,
    const mlx_inline_array* right_signs,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (n_rows == 0) return 1;

    try {
        constexpr float kScale = 0.0625f;
        auto input_f32 = mlx::core::astype(as_arr(input), float32);
        auto input_contig = mlx::core::contiguous(input_f32);
        auto signed_input = mlx::core::multiply(input_contig, as_arr(left_signs));
        auto signed_input_contig = mlx::core::contiguous(signed_input);
        auto transformed = mlx::core::hadamard_transform(signed_input_contig, kScale);
        auto output_arr = mlx::core::multiply(transformed, as_arr(right_signs));
        new (out->buf) array(output_arr);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_signed_fwht_256_rows", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_signed_fwht_256_rows", "unknown C++ exception"); return 1; }
}
int mlx_inline_turboquant_attention_q8_d256_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* key_qjl_signs,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_residual_norms,
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

int mlx_inline_turboquant_score_q8_d256_fullbyte(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;

    if (n_rows == 0 || n_seq == 0 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& key_indices_arr = as_arr(key_indices);
        const array& slot_scales_arr = as_arr(slot_scales);
        const array& key_codebook_arr = as_arr(key_codebook);

        if (query_rot_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 256) return 1;

        auto& kernel = get_turboquant_score_q8_d256_fullbyte_kernel();
        auto outputs = kernel(
            {query_rot_arr, key_indices_arr, slot_scales_arr, key_codebook_arr},
            {{(int)n_rows, (int)n_seq}},
            {float32},
            {((int)n_seq + 63) & ~63, (int)n_rows, 1},
            {64, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
                {"attn_scale_bits", (int)attn_scale_bits},
            },
            std::nullopt, false, {}
        );
        new (out_scores->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_score_q8_d256_fullbyte", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_score_q8_d256_fullbyte", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_weighted_sum_d256_dense_values(
    mlx_inline_array*       out,
    const mlx_inline_array* weights,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads)
{
    using namespace mlx::core;

    constexpr uint32_t dim = 256u;

    if (n_rows == 0 || n_seq == 0 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& weights_arr = as_arr(weights);
        const array& value_dense_arr = as_arr(value_dense);

        if (weights_arr.shape(-1) != n_seq) return 1;
        if (value_dense_arr.shape(-1) != dim) return 1;

        auto& kernel = get_turboquant_weighted_sum_d256_dense_values_kernel();
        auto outputs = kernel(
            {weights_arr, value_dense_arr},
            {{(int)n_rows, (int)dim}},
            {float32},
            {(int)n_rows, 1, 1},
            {32, 1, 1},
            {
                {"n_rows", (int)n_rows},
                {"n_seq", (int)n_seq},
                {"cache_seq_capacity", (int)cache_seq_capacity},
                {"q_heads", (int)q_heads},
                {"kv_heads", (int)kv_heads},
            },
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_weighted_sum_d256_dense_values", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_weighted_sum_d256_dense_values", "unknown C++ exception"); return 1; }
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

int mlx_inline_turboquant_attention_q8_d128_packed_keys_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_bytes,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_residual_norms,
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

    constexpr uint32_t dim = 128u;
    const uint32_t blocks = turboquant_q8_2pass_blocks_override_or(32u);

    if (n_rows == 0 || n_seq < 1024 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& query_proj_arr = as_arr(query_proj);
        const array& key_bytes_arr = as_arr(key_bytes);
        const array& key_norms_arr = as_arr(key_norms);
        const array& key_residual_norms_arr = as_arr(key_residual_norms);
        const array& key_codebook_arr = as_arr(key_codebook);
        const array& value_indices_arr = as_arr(value_indices);
        const array& value_norms_arr = as_arr(value_norms);
        const array& value_codebook_arr = as_arr(value_codebook);

        if (query_rot_arr.shape(-1) != dim || query_proj_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 128 || value_codebook_arr.shape(0) != 256) return 1;

        auto& pass1 = get_turboquant_attention_q8_d128_packed_keys_2pass_1_kernel();
        auto pass1_outputs = pass1(
            {
                query_rot_arr,
                query_proj_arr,
                key_bytes_arr,
                key_norms_arr,
                key_residual_norms_arr,
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

        auto& pass2 = get_turboquant_attention_q8_d128_2pass_2_kernel();
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
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d128_packed_keys_2pass", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d128_packed_keys_2pass", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_weighted_decode(
    mlx_inline_array*       out,
    const mlx_inline_array* weights,
    const mlx_inline_array* indices,
    const mlx_inline_array* norms,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads)
{
    using namespace mlx::core;

    if (n_centroids == 0 || dim == 0 || n_rows == 0 || n_seq == 0 || kv_heads == 0 || q_heads == 0) return 1;
    if (cache_seq_capacity < n_seq) return 1;
    if (q_heads % kv_heads != 0) return 1;

    try {
        auto& kernel = get_turboquant_weighted_decode_kernel();
        constexpr uint32_t tg_threads = 32u;
        constexpr uint32_t tile_dims = 8u;
        uint32_t dim_tiles = (dim + tile_dims - 1u) / tile_dims;
        auto outputs = kernel(
            {as_arr(weights), as_arr(indices), as_arr(norms), as_arr(codebook)},
            {{(int)n_rows, (int)dim}},
            {float32},
            {(int)tg_threads, (int)(n_rows * dim_tiles), 1},
            {(int)tg_threads, 1, 1},
            {{"dim",         (int)dim},
             {"n_centroids", (int)n_centroids},
             {"n_rows",      (int)n_rows},
             {"n_seq",       (int)n_seq},
             {"cache_seq_capacity", (int)cache_seq_capacity},
             {"q_heads",     (int)q_heads},
             {"kv_heads",    (int)kv_heads}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_weighted_decode", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_weighted_decode", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_attention_q8_d128_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* key_qjl_signs,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_residual_norms,
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

    constexpr uint32_t dim = 128u;
    const uint32_t blocks = turboquant_q8_2pass_blocks_override_or(64u);

    if (n_rows == 0 || n_seq < 1024 || cache_seq_capacity < n_seq) return 1;
    if (q_heads == 0 || kv_heads == 0 || (q_heads % kv_heads) != 0) return 1;

    try {
        const array& query_rot_arr = as_arr(query_rot);
        const array& query_proj_arr = as_arr(query_proj);
        const array& key_indices_arr = as_arr(key_indices);
        const array& key_qjl_signs_arr = as_arr(key_qjl_signs);
        const array& key_norms_arr = as_arr(key_norms);
        const array& key_residual_norms_arr = as_arr(key_residual_norms);
        const array& key_codebook_arr = as_arr(key_codebook);
        const array& value_indices_arr = as_arr(value_indices);
        const array& value_norms_arr = as_arr(value_norms);
        const array& value_codebook_arr = as_arr(value_codebook);

        if (query_rot_arr.shape(-1) != dim || query_proj_arr.shape(-1) != dim) return 1;
        if (key_codebook_arr.shape(0) != 256 || value_codebook_arr.shape(0) != 256) return 1;

        auto& pass1 = get_turboquant_attention_q8_d128_2pass_1_kernel();
        auto pass1_outputs = pass1(
            {
                query_rot_arr,
                query_proj_arr,
                key_indices_arr,
                key_qjl_signs_arr,
                key_norms_arr,
                key_residual_norms_arr,
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

        auto& pass2 = get_turboquant_attention_q8_d128_2pass_2_kernel();
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
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_attention_q8_d128_2pass", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_attention_q8_d128_2pass", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_gather_last_dim(
    mlx_inline_array*       out,
    const mlx_inline_array* input,
    const mlx_inline_array* positions,
    uint32_t                full_dim,
    uint32_t                out_dim,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (full_dim == 0 || out_dim == 0 || n_rows == 0) return 1;

    try {
        auto& kernel = get_turboquant_gather_last_dim_kernel();
        auto outputs = kernel(
            {as_arr(input), as_arr(positions)},
            {{(int)n_rows, (int)out_dim}},
            {float32},
            {(int)out_dim, (int)n_rows, 1},
            {(int)std::min(out_dim, 256u), 1, 1},
            {{"full_dim", (int)full_dim},
             {"out_dim",  (int)out_dim},
             {"n_rows",   (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_gather_last_dim", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_gather_last_dim", "unknown C++ exception"); return 1; }
}

int mlx_inline_turboquant_scatter_last_dim(
    mlx_inline_array*       out,
    const mlx_inline_array* regular,
    const mlx_inline_array* outlier,
    const mlx_inline_array* regular_positions,
    const mlx_inline_array* outlier_positions,
    uint32_t                full_dim,
    uint32_t                regular_dim,
    uint32_t                outlier_dim,
    uint32_t                n_rows)
{
    using namespace mlx::core;

    if (full_dim == 0 || n_rows == 0) return 1;

    try {
        auto& kernel = get_turboquant_scatter_last_dim_kernel();
        auto outputs = kernel(
            {
                as_arr(regular),
                as_arr(outlier),
                as_arr(regular_positions),
                as_arr(outlier_positions),
            },
            {{(int)n_rows, (int)full_dim}},
            {float32},
            {(int)full_dim, (int)n_rows, 1},
            {(int)std::min(full_dim, 256u), 1, 1},
            {{"full_dim",    (int)full_dim},
             {"regular_dim", (int)regular_dim},
             {"outlier_dim", (int)outlier_dim},
             {"n_rows",      (int)n_rows}},
            std::nullopt, false, {}
        );
        new (out->buf) array(outputs[0]);
        return 0;
    } catch (const std::exception& e) { pmetal_bridge_set_last_error("turboquant_scatter_last_dim", e.what()); return 1; } catch (...) { pmetal_bridge_set_last_error("turboquant_scatter_last_dim", "unknown C++ exception"); return 1; }
}

int mlx_inline_gdn_update(
    mlx_inline_array* dst_y,
    mlx_inline_array* dst_state,
    const mlx_inline_array* q,
    const mlx_inline_array* k,
    const mlx_inline_array* v,
    const mlx_inline_array* a,
    const mlx_inline_array* b,
    const mlx_inline_array* a_log,
    const mlx_inline_array* dt_bias,
    const mlx_inline_array* state_in,
    bool training) {
    try {
        auto& q_ref = as_arr(q);
        auto& k_ref = as_arr(k);
        auto& v_ref = as_arr(v);
        auto& a_ref = as_arr(a);
        auto& b_ref = as_arr(b);
        auto& a_log_ref = as_arr(a_log);
        auto& dt_bias_ref = as_arr(dt_bias);
        auto& state_ref = as_arr(state_in);

        using namespace mlx::core;

        // Compute beta and g
        auto beta = sigmoid(b_ref);
        auto a_log_f32 = astype(a_log_ref, float32);
        auto decay_rate = exp(a_log_f32);
        auto sp = log1p(exp(add(a_ref, dt_bias_ref)));
        auto g = exp(negative(multiply(decay_rate, sp)));

        int B = q_ref.shape(0);
        int T = q_ref.shape(1);
        int Hk = q_ref.shape(2);
        int Dk = q_ref.shape(3);
        int Hv = v_ref.shape(2);
        int Dv = v_ref.shape(3);

        // Try Metal kernel: requires Dk%32==0, Dk<=256, scalar gating (g is 3D)
        bool use_metal = !training && (Dk % 32 == 0) && (Dk <= 256) && (Dk > 0) && (g.ndim() == 3);

        if (use_metal) {
            auto input_dtype = q_ref.dtype();
            auto state_dtype = state_ref.dtype();
            auto t_arr = array(T);

            auto& kernel = get_gdn_kernel();
            auto outputs = kernel(
                {q_ref, k_ref, v_ref, g, beta, state_ref, t_arr},
                {{B, T, Hv, Dv}, state_ref.shape()},       // output shapes
                {input_dtype, state_dtype},                   // output dtypes
                {32, Dv, B * Hv},                            // grid
                {32, 4, 1},                                   // threadgroup
                {{"InT", input_dtype}, {"StT", state_dtype},
                 {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}},
                std::nullopt,                                 // init_value
                false,                                        // verbose
                {}                                            // default stream
            );

            new (dst_y->buf) array(outputs[0]);
            new (dst_state->buf) array(outputs[1]);
            return 0;
        }

        // Fallback: ops-based recurrence
        int repeat_factor = Hv / Hk;
        auto q_expanded = (repeat_factor > 1) ? repeat(q_ref, repeat_factor, 2) : q_ref;
        auto k_expanded = (repeat_factor > 1) ? repeat(k_ref, repeat_factor, 2) : k_ref;

        auto state = state_ref;
        std::vector<array> ys;
        ys.reserve(T);

        for (int t = 0; t < T; ++t) {
            auto q_t = squeeze(slice(q_expanded, {0, t, 0, 0}, {B, t+1, Hv, Dk}), 1);
            auto k_t = squeeze(slice(k_expanded, {0, t, 0, 0}, {B, t+1, Hv, Dk}), 1);
            auto v_t = squeeze(slice(v_ref, {0, t, 0, 0}, {B, t+1, Hv, Dv}), 1);
            auto g_t = squeeze(slice(g, {0, t, 0}, {B, t+1, Hv}), 1);
            auto beta_t = squeeze(slice(beta, {0, t, 0}, {B, t+1, Hv}), 1);

            auto g_exp = reshape(g_t, {B, Hv, 1, 1});
            auto decayed = multiply(state, g_exp);
            auto k_4d = reshape(k_t, {B, Hv, 1, Dk});
            auto kv_mem = sum(multiply(decayed, k_4d), {-1}, false);
            auto beta_exp = reshape(beta_t, {B, Hv, 1});
            auto delta = multiply(subtract(v_t, kv_mem), beta_exp);
            auto delta_4d = reshape(delta, {B, Hv, Dv, 1});
            state = add(decayed, multiply(k_4d, delta_4d));
            auto q_4d = reshape(q_t, {B, Hv, 1, Dk});
            auto y_t = sum(multiply(state, q_4d), {-1}, false);
            ys.push_back(astype(y_t, q_ref.dtype()));
        }

        auto y = (T == 1) ? reshape(ys[0], {B, 1, Hv, Dv}) : stack(ys, 1);
        new (dst_y->buf) array(y);
        new (dst_state->buf) array(state);
        pmetal_bridge_clear_error_internal();
        return 0;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("gdn_update", e.what());
        return -1;
    } catch (...) {
        pmetal_bridge_set_last_error("gdn_update", "unknown C++ exception");
        return -1;
    }
}


} // extern "C"
