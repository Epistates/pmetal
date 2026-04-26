// TurboQuant long-context q8 attention kernels for D=128 key/value dim.
// Single 2-pass family (base indices + packed-keys variant) sharing
// the D=128 pass-2 merge kernel.
//
// ─────────────────────────────────────────────────────────────────────────
// Templating note (audit revisit 2026-04-26)
// ─────────────────────────────────────────────────────────────────────────
// The April 2026 audit considered consolidating this file and
// bridge_turboquant_attn_d256.cpp behind a `MAKE_TQ_ATTN_KERNEL(NAME, DIM,
// VEC, ...)` macro that would emit kernel sources with constants substituted.
// After deeper exploration the conclusion is "do not pursue":
//
//   1. Pass-1 unroll counts genuinely differ (4 vs 8 accumulators per lane).
//      A constants-only macro can't change this; consolidation would need
//      either a token-pasting unroll macro (`ACC_LANE(0); ACC_LANE(1); …`)
//      or a Metal `for` loop. Both have downsides — the former obscures
//      the code; the latter risks a perf regression we'd need to re-tune.
//
//   2. Pass-2 merge reductions diverge on purpose. d128 uses `simd_sum`;
//      d256 uses an explicit `for (uint g = 0; g < kSimds; ++g) sum += …`
//      because the larger d256 footprint benefits from the loop form. This
//      is a perf-tuning choice baked into the kernel body, not a constant.
//
//   3. d256 has four kernels with NO d128 equivalent: packed_keys_dense_values,
//      fullbyte_dense_values, fullbyte_localsoftmax, packed_kv_dense_values.
//      The fullbyte family is q8-MSE-only with no QJL residual term — the
//      score-side math is structurally distinct, not a constant swap.
//
//   4. MLX upstream solves this with `template <typename T, int D, int V>`
//      at the .metal source level (see mlx/backend/metal/kernels/sdpa_vector.h)
//      plus Metal function constants for runtime branches. The public
//      `mlx::core::fast::metal_kernel` API we use here does NOT expose
//      function constants, and exposing them would mean linking against
//      MLX internals — out of scope for a duplication cleanup.
//
// Net: keeping d128 and d256 as separate maintainable units is the right
// call. The actual large-file problem is the 1930-LOC d256 file's growth
// across five variant families; that's tracked as a navigability concern
// (split by family) rather than a templating one.

#include "bridge_turboquant_internal.h"

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

extern "C" {

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


} // extern "C"
