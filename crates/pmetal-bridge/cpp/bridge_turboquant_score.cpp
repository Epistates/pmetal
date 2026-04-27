// TurboQuant scoring kernels: generic score, specialised q8_d256
// (transposed and fullbyte layouts), and the mixed regular+outlier
// score kernel.

#include "bridge_turboquant_internal.h"

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
    // Codebook indices were quantised against rotated values divided by
    // key_slot_scale; recover the original-magnitude codebook contribution.
    float score = mse * key_slot_scale[kv_scalar_idx];
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
    float key_slot_scale_val = key_slot_scale[scalar_idx];

    float score_part_codebook = 0.0f;
    float score_part_qjl = 0.0f;
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
        score_part_codebook += qrot0 * shared_k_codebook[(uint)indices[key_base + 0u * cache_seq_capacity]];
        score_part_codebook += qrot1 * shared_k_codebook[(uint)indices[key_base + 1u * cache_seq_capacity]];
        score_part_codebook += qrot2 * shared_k_codebook[(uint)indices[key_base + 2u * cache_seq_capacity]];
        score_part_codebook += qrot3 * shared_k_codebook[(uint)indices[key_base + 3u * cache_seq_capacity]];
        score_part_codebook += qrot4 * shared_k_codebook[(uint)indices[key_base + 4u * cache_seq_capacity]];
        score_part_codebook += qrot5 * shared_k_codebook[(uint)indices[key_base + 5u * cache_seq_capacity]];
        score_part_codebook += qrot6 * shared_k_codebook[(uint)indices[key_base + 6u * cache_seq_capacity]];
        score_part_codebook += qrot7 * shared_k_codebook[(uint)indices[key_base + 7u * cache_seq_capacity]];
        score_part_qjl += residual_scale * qproj0 * ((((sign_word >> (bit_base + 0u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part_qjl += residual_scale * qproj1 * ((((sign_word >> (bit_base + 1u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part_qjl += residual_scale * qproj2 * ((((sign_word >> (bit_base + 2u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part_qjl += residual_scale * qproj3 * ((((sign_word >> (bit_base + 3u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part_qjl += residual_scale * qproj4 * ((((sign_word >> (bit_base + 4u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part_qjl += residual_scale * qproj5 * ((((sign_word >> (bit_base + 5u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part_qjl += residual_scale * qproj6 * ((((sign_word >> (bit_base + 6u)) & 1u) == 0u) ? -1.0f : 1.0f);
        score_part_qjl += residual_scale * qproj7 * ((((sign_word >> (bit_base + 7u)) & 1u) == 0u) ? -1.0f : 1.0f);
    }

    // Codebook indices were quantised against rotated values divided by
    // key_slot_scale; recover the original-magnitude codebook contribution.
    output[row * n_seq + seq] = key_norm * (key_slot_scale_val * score_part_codebook + score_part_qjl);
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

    // Codebook indices were quantised against rotated values divided by
    // *_slot_scale; recover the original-magnitude codebook contribution.
    float regular_score = regular_mse * regular_slot_scale[kv_scalar_idx];
    float regular_residual = regular_residual_norms[kv_scalar_idx];
    if (regular_residual > 0.0f) {
        regular_score += regular_qjl * regular_residual * (1.2533141373155003f / float(regular_dim));
    }
    regular_score *= regular_norms[kv_scalar_idx];

    float outlier_score = outlier_mse * outlier_slot_scale[kv_scalar_idx];
    float outlier_residual = outlier_residual_norms[kv_scalar_idx];
    if (outlier_residual > 0.0f) {
        outlier_score += outlier_qjl * outlier_residual * (1.2533141373155003f / float(outlier_dim));
    }
    outlier_score *= outlier_norms[kv_scalar_idx];

    float attn_scale = as_type<float>((uint)attn_scale_bits);
    output[scalar_idx] = (regular_score + outlier_score) * attn_scale;
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
    float key_norm = slot_scales[scalar_idx * 4u + 0u];
    float key_slot_scale = slot_scales[scalar_idx * 4u + 3u];
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
    // Codebook indices were quantised against rotated values divided by
    // key_slot_scale; recover the original-magnitude codebook contribution.
    output[row * n_seq + seq] = key_norm * key_slot_scale * score_part;
)";

static mlx::core::fast::CustomKernelFunction& get_turboquant_score_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "turboquant_score",
        {"query_rot", "query_proj", "indices", "qjl_signs", "norms", "residual_norms", "key_slot_scale", "codebook"},
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
        {"query_rot", "query_proj", "indices", "qjl_signs", "norms", "residual_norms", "key_slot_scale", "codebook"},
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
            "regular_slot_scale",
            "regular_codebook",
            "outlier_query_rot",
            "outlier_query_proj",
            "outlier_indices",
            "outlier_qjl_signs",
            "outlier_norms",
            "outlier_residual_norms",
            "outlier_slot_scale",
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

extern "C" {

int mlx_inline_turboquant_score(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    const mlx_inline_array* norms,
    const mlx_inline_array* residual_norms,
    const mlx_inline_array* key_slot_scale,
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
            {as_arr(query_rot), as_arr(query_proj), as_arr(indices), as_arr(qjl_signs), as_arr(norms), as_arr(residual_norms), as_arr(key_slot_scale), as_arr(codebook)},
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
    const mlx_inline_array* key_slot_scale,
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
        const array& key_slot_scale_arr = as_arr(key_slot_scale);
        const array& codebook_arr = as_arr(codebook);

        if (query_rot_arr.shape(-1) != dim || query_proj_arr.shape(-1) != dim) return 1;
        if (codebook_arr.shape(0) != n_centroids) return 1;

        auto& kernel = get_turboquant_score_q8_d256_kernel();
        auto outputs = kernel(
            {query_rot_arr, query_proj_arr, indices_arr, qjl_signs_arr, norms_arr, residual_norms_arr, key_slot_scale_arr, codebook_arr},
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
    const mlx_inline_array* regular_slot_scale,
    const mlx_inline_array* regular_codebook,
    const mlx_inline_array* outlier_query_rot,
    const mlx_inline_array* outlier_query_proj,
    const mlx_inline_array* outlier_indices,
    const mlx_inline_array* outlier_qjl_signs,
    const mlx_inline_array* outlier_norms,
    const mlx_inline_array* outlier_residual_norms,
    const mlx_inline_array* outlier_slot_scale,
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
                as_arr(regular_slot_scale),
                as_arr(regular_codebook),
                as_arr(outlier_query_rot),
                as_arr(outlier_query_proj),
                as_arr(outlier_indices),
                as_arr(outlier_qjl_signs),
                as_arr(outlier_norms),
                as_arr(outlier_residual_norms),
                as_arr(outlier_slot_scale),
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

} // extern "C"
