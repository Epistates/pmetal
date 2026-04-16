// Native forward pass: GDN, Attention, MoE layers + Qwen3.5 decode step.
// Extracted from bridge.cpp for maintainability.

#include "bridge_internal.h"

static const char* GDN_METAL_SOURCE = R"(
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;
    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
)";

mlx::core::fast::CustomKernelFunction& get_gdn_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "gated_delta_step",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"},
        GDN_METAL_SOURCE,
        "",    // no header
        true,  // ensure_row_contiguous
        false  // atomic_outputs
    );
    return kernel;
}

// State-only variant: advances [B, Hv, Dv, Dk] GDN state through T steps
// WITHOUT computing the attention output `y`. Used by speculative-decoding
// rollback replay, where we only need the final state — skipping the
// output projection saves ~50% of the per-step work compared to reusing
// the full `gated_delta_step` kernel and discarding `y`.
//
// Ported from dflash-mlx `adapters.py:34-81` (z-lab, 2026) and adapted to
// the pmetal convention where keys stay at Hk heads (GQA expansion handled
// via `hk_idx`).
static const char* GDN_STATE_UPDATE_METAL_SOURCE = R"(
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;
    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
      }
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
)";

mlx::core::fast::CustomKernelFunction& get_gdn_state_update_kernel() {
    static auto kernel = mlx::core::fast::metal_kernel(
        "gated_delta_state_update",
        {"k", "v", "g", "beta", "state_in", "T"},
        {"state_out"},
        GDN_STATE_UPDATE_METAL_SOURCE,
        "",    // no header
        true,  // ensure_row_contiguous
        false  // atomic_outputs
    );
    return kernel;
}

extern "C" {

// Helper: layer i is GDN when (i+1) % interval != 0
static inline bool layer_is_gdn(int i, int interval) {
    return ((i + 1) % interval) != 0;
}

// Return the array at a flat weight pointer slot.
// weight_ptrs[idx] must not be NULL (caller guarantees for required weights).
static inline const array& W(const mlx_inline_array* const* wp, int idx) {
    return *reinterpret_cast<const array*>(wp[idx]->buf);
}

// Access a cache slot (in/out array, always initialised before entry).
static inline array& C_arr(mlx_inline_array* cp) {
    return *reinterpret_cast<array*>(cp->buf);
}

// ── GDN layer forward ─────────────────────────────────────────────────────
//
// Handles both T=1 (compiled tape-replay) and T>1 (direct ops).
// Returns the layer output. Cache slots are updated in-place.
static array run_gdn_layer(
    const array&             normed,
    const array& qkv_w, const array& z_w,   const array& b_w,  const array& a_w,
    const array& conv_w,
    const array& q_nw,  const array& k_nw,
    const array& a_log, const array& dt_bias,
    const array& norm_w, const array& out_w,
    int nv, int nk, int dk, int dv, int cd, int ck, int kd, float norm_eps,
    mlx_inline_array* cache_conv,   // in/out conv_state
    mlx_inline_array* cache_ssm,    // in/out ssm_state
    int model_dtype
) {
    using namespace mlx::core;

    int B = normed.shape(0);
    int S = normed.shape(1);
    auto dtype = dtype_from_int(model_dtype);

    if (S == 1) {
        // ── T=1: compiled tape-replay path ──────────────────────────────────
        // Build temporary mlx_inline_array wrappers (stack-only, no heap).
        mlx_inline_array w_normed, w_qkv, w_z, w_b, w_a, w_conv;
        mlx_inline_array w_qn, w_kn, w_al, w_dt, w_nm, w_out, w_cs, w_ss;
        new (w_normed.buf) array(normed);
        new (w_qkv.buf)   array(qkv_w);  new (w_z.buf) array(z_w);
        new (w_b.buf)     array(b_w);    new (w_a.buf) array(a_w);
        new (w_conv.buf)  array(conv_w);
        new (w_qn.buf)    array(q_nw);   new (w_kn.buf) array(k_nw);
        new (w_al.buf)    array(a_log);  new (w_dt.buf) array(dt_bias);
        new (w_nm.buf)    array(norm_w); new (w_out.buf) array(out_w);
        new (w_cs.buf)    array(C_arr(cache_conv));
        new (w_ss.buf)    array(C_arr(cache_ssm));

        mlx_inline_array dst_out, dst_conv, dst_ssm;
        mlx_inline_compiled_gdn_layer_fixed(
            &dst_out, &dst_conv, &dst_ssm,
            &w_normed,
            &w_qkv, &w_z, &w_b, &w_a, &w_conv,
            &w_qn, &w_kn, &w_al, &w_dt, &w_nm, &w_out,
            &w_cs, &w_ss,
            nv, nk, dk, dv, cd, ck, kd, norm_eps);

        // Destroy temp wrappers
        as_arr(&w_normed).~array(); as_arr(&w_qkv).~array();
        as_arr(&w_z).~array();     as_arr(&w_b).~array();
        as_arr(&w_a).~array();     as_arr(&w_conv).~array();
        as_arr(&w_qn).~array();    as_arr(&w_kn).~array();
        as_arr(&w_al).~array();    as_arr(&w_dt).~array();
        as_arr(&w_nm).~array();    as_arr(&w_out).~array();
        as_arr(&w_cs).~array();    as_arr(&w_ss).~array();

        // Write back cache state (destroy old + placement-new new)
        C_arr(cache_conv).~array();
        new (cache_conv->buf) array(std::move(as_arr(&dst_conv)));
        C_arr(cache_ssm).~array();
        new (cache_ssm->buf) array(std::move(as_arr(&dst_ssm)));

        array gdn_out = as_arr(&dst_out);
        as_arr(&dst_out).~array();
        return gdn_out;
    }

    // ── Direct ops path (T>=1) ─────────────────────────────────────────────
    auto qkv   = matmul(normed, qkv_w);
    auto z     = reshape(matmul(normed, z_w), {B, S, nv, dv});
    auto b_val = matmul(normed, b_w);
    auto a_val = matmul(normed, a_w);

    auto conv_in  = concatenate({C_arr(cache_conv), qkv}, 1);
    auto new_conv = slice(conv_in, {0, 1, 0}, {B, ck, cd});
    auto conv_out = mlx::core::conv1d(conv_in, conv_w, 1, 0, 1, cd);
    // fused silu: x * sigmoid(x)
    auto conv_act = multiply(conv_out, sigmoid(conv_out));

    // Single split → 3 siblings sharing one Split primitive (matches Python's mx.split).
    // This replaces 3 Slice nodes with 1 Split node, saving 2 nodes per GDN prefill layer.
    auto conv_parts = split(conv_act, Shape{kd, kd * 2}, -1);
    auto q = fast::rms_norm(reshape(conv_parts[0], {B,S,nk,dk}), q_nw, norm_eps);
    auto k = fast::rms_norm(reshape(conv_parts[1], {B,S,nk,dk}), k_nw, norm_eps);
    auto v =                reshape(conv_parts[2], {B,S,nv,dv});

    // compute_g: exp(-exp(a_log.f32) * softplus(a_val + dt_bias))
    auto g    = exp(negative(multiply(exp(astype(a_log, float32)),
                               log1p(exp(add(a_val, dt_bias))))));
    auto beta = sigmoid(b_val);

    // GDN Metal kernel recurrence — wrap in mlx_inline_array for the existing C fn
    mlx_inline_array wq, wk, wv, wg, wb, wsi, tmp_y, tmp_state;
    new (wq.buf) array(q);     new (wk.buf) array(k);     new (wv.buf) array(v);
    new (wg.buf) array(g);     new (wb.buf) array(beta);
    new (wsi.buf) array(C_arr(cache_ssm));
    mlx_inline_gdn_metal_step(&tmp_y, &tmp_state, &wq, &wk, &wv, &wg, &wb, &wsi, S);
    as_arr(&wq).~array(); as_arr(&wk).~array(); as_arr(&wv).~array();
    as_arr(&wg).~array(); as_arr(&wb).~array(); as_arr(&wsi).~array();

    array gdn_y     = std::move(as_arr(&tmp_y));
    array new_state = std::move(as_arr(&tmp_state));
    as_arr(&tmp_y).~array(); as_arr(&tmp_state).~array();

    // Write back cache
    C_arr(cache_conv).~array();
    new (cache_conv->buf) array(std::move(new_conv));
    C_arr(cache_ssm).~array();
    new (cache_ssm->buf) array(std::move(new_state));

    // Output: rms_norm → precise_swiglu → reshape → out_proj
    auto out_n = fast::rms_norm(gdn_y, norm_w, norm_eps);
    auto g32   = multiply(astype(z, float32), sigmoid(astype(z, float32)));
    auto gated = astype(multiply(g32, astype(out_n, float32)), dtype);
    return matmul(reshape(gated, {B, S, nv * dv}), out_w);
}

// ── Attention layer forward ───────────────────────────────────────────────
// Returns the layer output. Cache slots and kv_offset are updated in-place.
static array run_attn_layer(
    const array&         normed,
    int B, int S,
    const array& q_w,  const array& k_w, const array& v_w, const array& o_w,
    const array& q_nw, const array& k_nw,
    float q_norm_eps, float k_norm_eps,
    int n_heads, int n_kv, int head_dim,
    float scale, int rope_dims, float rope_base, float rope_scale,
    mlx_inline_array* cache_keys,
    mlx_inline_array* cache_vals,
    int& kv_offset,
    int  rope_offset,
    int  model_dtype
) {
    using namespace mlx::core;
    auto dtype = dtype_from_int(model_dtype);

    // Q projection — width = n_heads * head_dim * 2 (queries + gate)
    auto q_proj = matmul(normed, q_w);
    auto qg     = reshape(q_proj, {B, S, n_heads, head_dim * 2});
    auto qg_split = split(qg, Shape{head_dim}, -1);
    auto queries  = qg_split[0];                                        // [B,S,H,D]
    auto gate     = reshape(qg_split[1], {B, S, n_heads * head_dim});  // [B,S,H*D]

    // K, V projections
    auto new_k = matmul(normed, k_w);
    auto new_v = matmul(normed, v_w);

    // Q/K norms
    queries       = fast::rms_norm(queries,                                    q_nw, q_norm_eps);
    auto keys     = fast::rms_norm(reshape(new_k, {B, S, n_kv, head_dim}),    k_nw, k_norm_eps);
    auto values   = reshape(new_v, {B, S, n_kv, head_dim});

    // Transpose to [B, H, S, D]
    queries = transpose(queries, {0, 2, 1, 3});
    keys    = transpose(keys,    {0, 2, 1, 3});
    values  = transpose(values,  {0, 2, 1, 3});

    // Partial RoPE
    queries = fast::rope(queries, rope_dims, false, rope_base, rope_scale, rope_offset);
    keys    = fast::rope(keys,    rope_dims, false, rope_base, rope_scale, rope_offset);

    // KV cache: grow if needed
    int prev = kv_offset;
    int next = prev + S;
    {
        const array& ck = C_arr(cache_keys);
        bool is_empty   = (ck.ndim() == 0 || ck.size() == 0);
        int  allocated  = is_empty ? 0 : ck.shape(2);

        if (is_empty || next > allocated) {
            int new_alloc = ((next + 255) / 256) * 256;
            if (is_empty) {
                auto nb_k = zeros({B, n_kv, new_alloc, head_dim}, dtype);
                auto nb_v = zeros({B, n_kv, new_alloc, head_dim}, dtype);
                C_arr(cache_keys).~array();
                new (cache_keys->buf) array(std::move(nb_k));
                C_arr(cache_vals).~array();
                new (cache_vals->buf) array(std::move(nb_v));
            } else {
                int extend = new_alloc - allocated;
                auto ext_k = zeros({B, n_kv, extend, head_dim}, dtype);
                auto ext_v = zeros({B, n_kv, extend, head_dim}, dtype);
                auto grown_k = concatenate({C_arr(cache_keys), ext_k}, 2);
                auto grown_v = concatenate({C_arr(cache_vals), ext_v}, 2);
                C_arr(cache_keys).~array();
                new (cache_keys->buf) array(std::move(grown_k));
                C_arr(cache_vals).~array();
                new (cache_vals->buf) array(std::move(grown_v));
            }
        }
    }

    if (S == 1) {
        mlx_inline_array w_normed, w_q, w_k, w_v, w_o, w_qn, w_kn, w_ck, w_cv;
        new (w_normed.buf) array(normed);
        new (w_q.buf) array(q_w);
        new (w_k.buf) array(k_w);
        new (w_v.buf) array(v_w);
        new (w_o.buf) array(o_w);
        new (w_qn.buf) array(q_nw);
        new (w_kn.buf) array(k_nw);
        new (w_ck.buf) array(C_arr(cache_keys));
        new (w_cv.buf) array(C_arr(cache_vals));

        mlx_inline_array dst_out, dst_k, dst_v;
        mlx_inline_compiled_attn_layer_fixed(
            &dst_out,
            &dst_k,
            &dst_v,
            &w_normed,
            &w_q,
            &w_k,
            &w_v,
            &w_o,
            &w_qn,
            &w_kn,
            &w_ck,
            &w_cv,
            prev,
            rope_offset,
            n_heads,
            n_kv,
            head_dim,
            scale,
            rope_dims,
            rope_base,
            rope_scale,
            q_norm_eps,
            k_norm_eps,
            /*gated=*/true);

        as_arr(&w_normed).~array();
        as_arr(&w_q).~array();
        as_arr(&w_k).~array();
        as_arr(&w_v).~array();
        as_arr(&w_o).~array();
        as_arr(&w_qn).~array();
        as_arr(&w_kn).~array();
        as_arr(&w_ck).~array();
        as_arr(&w_cv).~array();

        C_arr(cache_keys).~array();
        new (cache_keys->buf) array(std::move(as_arr(&dst_k)));
        C_arr(cache_vals).~array();
        new (cache_vals->buf) array(std::move(as_arr(&dst_v)));
        kv_offset = next;

        array out = as_arr(&dst_out);
        as_arr(&dst_out).~array();
        return out;
    }

    // In-place slice_set: cache[..., prev:next, :] = new_kv
    {
        auto upd_k = slice_update(C_arr(cache_keys), keys,   {0,0,prev,0}, {B,n_kv,next,head_dim});
        auto upd_v = slice_update(C_arr(cache_vals), values, {0,0,prev,0}, {B,n_kv,next,head_dim});
        C_arr(cache_keys).~array();
        new (cache_keys->buf) array(std::move(upd_k));
        C_arr(cache_vals).~array();
        new (cache_vals->buf) array(std::move(upd_v));
    }
    kv_offset = next;

    // SDPA over valid portion
    auto valid_k = slice(C_arr(cache_keys), {0,0,0,0}, {B,n_kv,next,head_dim});
    auto valid_v = slice(C_arr(cache_vals), {0,0,0,0}, {B,n_kv,next,head_dim});
    auto output  = fast::scaled_dot_product_attention(queries, valid_k, valid_v, scale, "causal");

    // Gated output + projection
    output         = transpose(output, {0, 2, 1, 3});
    output         = reshape(output, {B, S, n_heads * head_dim});
    auto gated_out = multiply(output, sigmoid(gate));
    return matmul(gated_out, o_w);
}

// ── MoE block forward ─────────────────────────────────────────────────────
// Returns the post-attention MoE output for either decode (S=1) or prefill.
static array run_moe_layer(
    const array& mlp_in,
    const array& router_w,
    const array& moe_gate_w,
    const array& moe_up_w,
    const array& moe_down_w,
    const array& shared_gate_w,
    const array& shared_up_w,
    const array& shared_down_w,
    const array& shared_expert_gate_w,
    int top_k,
    bool norm_topk_prob
) {
    using namespace mlx::core;

    int B = mlp_in.shape(0);
    int S = mlp_in.shape(1);
    int H = mlp_in.shape(2);

    if (S == 1) {
        mlx_inline_array w_x, w_router, w_moe_gate, w_moe_up, w_moe_down;
        mlx_inline_array w_shared_gate, w_shared_up, w_shared_down, w_shared_expert_gate;
        new (w_x.buf) array(mlp_in);
        new (w_router.buf) array(router_w);
        new (w_moe_gate.buf) array(moe_gate_w);
        new (w_moe_up.buf) array(moe_up_w);
        new (w_moe_down.buf) array(moe_down_w);
        new (w_shared_gate.buf) array(shared_gate_w);
        new (w_shared_up.buf) array(shared_up_w);
        new (w_shared_down.buf) array(shared_down_w);
        new (w_shared_expert_gate.buf) array(shared_expert_gate_w);

        mlx_inline_array dst_out;
        mlx_inline_compiled_moe_layer_fixed(
            &dst_out,
            &w_x,
            &w_router,
            &w_moe_gate,
            &w_moe_up,
            &w_moe_down,
            &w_shared_gate,
            &w_shared_up,
            &w_shared_down,
            &w_shared_expert_gate,
            top_k,
            norm_topk_prob);

        as_arr(&w_x).~array();
        as_arr(&w_router).~array();
        as_arr(&w_moe_gate).~array();
        as_arr(&w_moe_up).~array();
        as_arr(&w_moe_down).~array();
        as_arr(&w_shared_gate).~array();
        as_arr(&w_shared_up).~array();
        as_arr(&w_shared_down).~array();
        as_arr(&w_shared_expert_gate).~array();

        array out = as_arr(&dst_out);
        as_arr(&dst_out).~array();
        return out;
    }

    auto x_flat = reshape(mlp_in, {B * S, H});
    auto gates = softmax(matmul(x_flat, router_w), -1, /*precise=*/true);
    int expert_count = router_w.shape(1);
    auto all_inds = argpartition(gates, -top_k, -1);
    auto inds = slice(all_inds, {0, expert_count - top_k}, {B * S, expert_count});
    auto scores = take_along_axis(gates, inds, -1);
    if (norm_topk_prob) {
        scores = divide(scores, sum(scores, {-1}, true));
    }

    auto switch_in = expand_dims(expand_dims(x_flat, 1), 2);
    auto rhs_indices = std::optional<array>(inds);
    auto x_gate_exp = gather_mm(switch_in, moe_gate_w, std::nullopt, rhs_indices, false);
    auto x_up_exp = gather_mm(switch_in, moe_up_w, std::nullopt, rhs_indices, false);
    auto x_act = multiply(multiply(x_gate_exp, sigmoid(x_gate_exp)), x_up_exp);
    auto y_exp = squeeze(gather_mm(x_act, moe_down_w, std::nullopt, rhs_indices, false), 2);
    auto scores_exp = reshape(scores, {B * S, top_k, 1});
    auto y_routed = sum(multiply(y_exp, scores_exp), {-2}, false);

    auto sh_gate = matmul(x_flat, shared_gate_w);
    auto sh_up = matmul(x_flat, shared_up_w);
    auto sh_act = multiply(multiply(sh_gate, sigmoid(sh_gate)), sh_up);
    auto sh_out = matmul(sh_act, shared_down_w);
    auto sh_scale = sigmoid(matmul(x_flat, shared_expert_gate_w));
    auto y_shared = multiply(sh_out, sh_scale);

    return reshape(add(y_routed, y_shared), {B, S, H});
}

// ── Main entry point ──────────────────────────────────────────────────────
void mlx_inline_qwen35_decode_step(
    mlx_inline_array*              dst_logits,
    const mlx_inline_array*        token_ids,
    const mlx_inline_array* const* weight_ptrs,
    int                            num_weights,
    mlx_inline_array**             cache_ptrs,
    int                            num_cache,
    int*                           attn_kv_offsets,
    int*                           rope_offset,
    const int*                     config_ints,
    int                            num_config_ints,
    const float*                   config_floats,
    int                            num_config_floats
) {
    using namespace mlx::core;
    (void)num_weights; (void)num_cache; (void)num_config_ints; (void)num_config_floats;

  try {
    // ── Unpack config ints ─────────────────────────────────────────────────
    const int  num_layers         = config_ints[0];
    const int  model_dtype        = config_ints[2];
    const int  n_gdn              = config_ints[3];
    const int  gdn_nv             = config_ints[5];
    const int  gdn_nk             = config_ints[6];
    const int  gdn_dk             = config_ints[7];
    const int  gdn_dv             = config_ints[8];
    const int  gdn_cd             = config_ints[9];
    const int  gdn_ck             = config_ints[10];
    const int  gdn_kd             = config_ints[11];
    const int  attn_n_heads       = config_ints[12];
    const int  attn_n_kv          = config_ints[13];
    const int  attn_head_dim      = config_ints[14];
    const int  attn_rope_dims     = config_ints[15];
    const int  full_attn_interval = config_ints[16];
    const bool tie_embeddings     = (config_ints[17] != 0);
    const int  moe_top_k          = config_ints[18];
    const bool moe_norm_topk_prob = (config_ints[19] != 0);
    const int* layer_is_moe       = config_ints + 20;

    // ── Unpack config floats ───────────────────────────────────────────────
    const float final_norm_eps   = config_floats[0];
    const float attn_scale       = config_floats[1];
    const float attn_rope_base   = config_floats[2];
    const float attn_rope_scale  = config_floats[3];

    // ── Dimensions ────────────────────────────────────────────────────────
    const array& tok = as_arr(token_ids);
    int B = tok.shape(0);
    int S = tok.shape(1);

    // ── Embedding lookup — take(embed_w, token_ids, axis=0) ────────────────
    // embed_w: [vocab, hidden], token_ids: [B, T] → flatten to [B*T] for take
    auto flat_ids = reshape(tok, {B * S});
    auto emb_flat = take(W(weight_ptrs, 0), flat_ids, 0);             // [B*T, hidden]
    auto hidden   = reshape(emb_flat, {B, S, emb_flat.shape(1)});     // [B, T, hidden]

    // ── Layer loop ─────────────────────────────────────────────────────────
    int gdn_slot  = 0;
    int attn_slot = 0;

    for (int li = 0; li < num_layers; ++li) {
        bool is_gdn = layer_is_gdn(li, full_attn_interval);
        int  base   = 3 + li * QWEN35_WEIGHTS_PER_LAYER;

        float input_eps = config_floats[4 + li * 2];
        float post_eps  = config_floats[4 + li * 2 + 1];

        // Input LayerNorm
        auto normed = fast::rms_norm(hidden, W(weight_ptrs, base + 0), input_eps);

        array layer_out(0.0f);  // placeholder; overwritten by run_gdn_layer/run_attn_layer

        if (is_gdn) {
            float gdn_norm_eps = config_floats[4 + num_layers * 2 + gdn_slot];

            layer_out = run_gdn_layer(
                normed,
                W(weight_ptrs, base + 5),   // gdn_qkv_w
                W(weight_ptrs, base + 6),   // gdn_z_w
                W(weight_ptrs, base + 7),   // gdn_b_w
                W(weight_ptrs, base + 8),   // gdn_a_w
                W(weight_ptrs, base + 9),   // gdn_conv_w
                W(weight_ptrs, base + 10),  // gdn_q_nw
                W(weight_ptrs, base + 11),  // gdn_k_nw
                W(weight_ptrs, base + 12),  // gdn_a_log
                W(weight_ptrs, base + 13),  // gdn_dt_bias
                W(weight_ptrs, base + 14),  // gdn_norm_w
                W(weight_ptrs, base + 15),  // gdn_out_w
                gdn_nv, gdn_nk, gdn_dk, gdn_dv, gdn_cd, gdn_ck, gdn_kd, gdn_norm_eps,
                cache_ptrs[gdn_slot * 2 + 0],       // conv_state
                cache_ptrs[gdn_slot * 2 + 1],       // ssm_state
                model_dtype
            );
            gdn_slot++;
        } else {
            float q_norm_eps = config_floats[4 + num_layers * 2 + n_gdn + attn_slot * 2];
            float k_norm_eps = config_floats[4 + num_layers * 2 + n_gdn + attn_slot * 2 + 1];

            layer_out = run_attn_layer(
                normed,
                B, S,
                W(weight_ptrs, base + 5),   // attn_q_w
                W(weight_ptrs, base + 6),   // attn_k_w
                W(weight_ptrs, base + 7),   // attn_v_w
                W(weight_ptrs, base + 8),   // attn_o_w
                W(weight_ptrs, base + 9),   // attn_q_norm_w
                W(weight_ptrs, base + 10),  // attn_k_norm_w
                q_norm_eps, k_norm_eps,
                attn_n_heads, attn_n_kv, attn_head_dim,
                attn_scale, attn_rope_dims, attn_rope_base, attn_rope_scale,
                cache_ptrs[n_gdn * 2 + attn_slot * 4 + 0],  // kv_keys
                cache_ptrs[n_gdn * 2 + attn_slot * 4 + 1],  // kv_vals
                attn_kv_offsets[attn_slot],
                *rope_offset,
                model_dtype
            );
            attn_slot++;
        }

        // Residual: h = hidden + layer_out
        auto h = add(hidden, layer_out);

        // Post-attention LayerNorm + dense / MoE MLP.
        auto mlp_in  = fast::rms_norm(h, W(weight_ptrs, base + 1), post_eps);
        array mlp_out(0.0f);
        if (layer_is_moe[li] != 0) {
            mlp_out = run_moe_layer(
                mlp_in,
                W(weight_ptrs, base + 2),   // moe_router_w
                W(weight_ptrs, base + 3),   // moe_gate_w
                W(weight_ptrs, base + 4),   // moe_up_w
                W(weight_ptrs, base + 16),  // moe_down_w
                W(weight_ptrs, base + 17),  // shared_gate_w
                W(weight_ptrs, base + 18),  // shared_up_w
                W(weight_ptrs, base + 19),  // shared_down_w
                W(weight_ptrs, base + 20),  // shared_expert_gate_w
                moe_top_k,
                moe_norm_topk_prob
            );
        } else {
            auto gate_v = matmul(mlp_in, W(weight_ptrs, base + 2));  // gate_w
            auto up_v = matmul(mlp_in, W(weight_ptrs, base + 3));    // up_w
            auto swiglu = multiply(multiply(gate_v, sigmoid(gate_v)), up_v);
            mlp_out = matmul(swiglu, W(weight_ptrs, base + 4));      // down_w
        }

        // Residual: hidden = h + mlp_out
        hidden = add(h, mlp_out);
    }

    // Advance rope_offset by S tokens
    *rope_offset += S;

    // ── Final norm + LM head ───────────────────────────────────────────────
    auto normed_final = fast::rms_norm(hidden, W(weight_ptrs, 1), final_norm_eps);
    array logits = tie_embeddings
        ? matmul(normed_final, transpose(W(weight_ptrs, 0)))
        : matmul(normed_final, W(weight_ptrs, 2));

    new (dst_logits->buf) array(std::move(logits));

    pmetal_bridge_clear_error_internal();
  } catch (const std::exception& e) {
    pmetal_bridge_set_last_error("qwen35_decode_step", e.what());
    new (dst_logits->buf) array(0.0f);
  } catch (...) {
    pmetal_bridge_set_last_error("qwen35_decode_step", "unknown C++ exception");
    new (dst_logits->buf) array(0.0f);
  }
}

// State-only variant for speculative-decoding rollback replay.
// Inputs: (k, v, g, beta, state_in, T)
// Output: state_out (no `y`).
// Falls back to the ops-based recurrence when the Metal kernel's
// function-constant preconditions are not met.
void mlx_inline_gdn_metal_state_update(
    mlx_inline_array* dst_state,
    const mlx_inline_array* k,
    const mlx_inline_array* v,
    const mlx_inline_array* g,
    const mlx_inline_array* beta,
    const mlx_inline_array* state_in,
    int T) {
    try {
        using namespace mlx::core;
        auto& k_ref = as_arr(k);
        auto& v_ref = as_arr(v);
        auto& g_ref = as_arr(g);
        auto& beta_ref = as_arr(beta);
        auto& state_ref = as_arr(state_in);

        int B = k_ref.shape(0);
        int Hk = k_ref.shape(2);
        int Dk = k_ref.shape(3);
        int Hv = v_ref.shape(2);
        int Dv = v_ref.shape(3);

        bool use_metal = (Dk % 32 == 0) && (Dk <= 256) && (Dk > 0) && (g_ref.ndim() == 3);

        if (use_metal) {
            auto input_dtype = k_ref.dtype();
            auto state_dtype = state_ref.dtype();
            auto t_arr = array(T);
            auto& kernel = get_gdn_state_update_kernel();
            auto outputs = kernel(
                {k_ref, v_ref, g_ref, beta_ref, state_ref, t_arr},
                {state_ref.shape()},
                {state_dtype},
                {32, Dv, B * Hv},
                {32, 4, 1},
                {{"InT", input_dtype}, {"StT", state_dtype},
                 {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}},
                std::nullopt, false, {});
            new (dst_state->buf) array(outputs[0]);
            return;
        }

        // Ops fallback: sequential recurrence without the `y` projection.
        int repeat_factor = Hv / Hk;
        auto k_exp = (repeat_factor > 1) ? repeat(k_ref, repeat_factor, 2) : k_ref;
        auto state = state_ref;
        for (int t = 0; t < T; ++t) {
            auto k_t = squeeze(slice(k_exp, {0, t, 0, 0}, {B, t + 1, Hv, Dk}), 1);
            auto v_t = squeeze(slice(v_ref, {0, t, 0, 0}, {B, t + 1, Hv, Dv}), 1);
            auto g_t = squeeze(slice(g_ref, {0, t, 0}, {B, t + 1, Hv}), 1);
            auto beta_t = squeeze(slice(beta_ref, {0, t, 0}, {B, t + 1, Hv}), 1);
            auto g_4d = reshape(g_t, {B, Hv, 1, 1});
            auto decayed = multiply(state, g_4d);
            auto k_4d = reshape(k_t, {B, Hv, 1, Dk});
            auto kv_mem = sum(multiply(decayed, k_4d), {-1}, false);
            auto beta_3d = reshape(beta_t, {B, Hv, 1});
            auto delta = multiply(subtract(v_t, kv_mem), beta_3d);
            auto delta_4d = reshape(delta, {B, Hv, Dv, 1});
            state = add(decayed, multiply(k_4d, delta_4d));
        }
        new (dst_state->buf) array(state);
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("gdn_metal_state_update", e.what());
        new (dst_state->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("gdn_metal_state_update", "unknown C++ exception");
        new (dst_state->buf) array(0.0f);
    }
}

void mlx_inline_gdn_metal_step(
    mlx_inline_array* dst_y,
    mlx_inline_array* dst_state,
    const mlx_inline_array* q,
    const mlx_inline_array* k,
    const mlx_inline_array* v,
    const mlx_inline_array* g,
    const mlx_inline_array* beta,
    const mlx_inline_array* state_in,
    int T) {
    try {
        using namespace mlx::core;
        auto& q_ref = as_arr(q);
        auto& k_ref = as_arr(k);
        auto& v_ref = as_arr(v);
        auto& g_ref = as_arr(g);
        auto& beta_ref = as_arr(beta);
        auto& state_ref = as_arr(state_in);

        int B = q_ref.shape(0);
        int Hk = q_ref.shape(2);
        int Dk = q_ref.shape(3);
        int Hv = v_ref.shape(2);
        int Dv = v_ref.shape(3);

        bool use_metal = (Dk % 32 == 0) && (Dk <= 256) && (Dk > 0) && (g_ref.ndim() == 3);

        if (use_metal) {
            auto input_dtype = q_ref.dtype();
            auto state_dtype = state_ref.dtype();
            auto t_arr = array(T);
            auto& kernel = get_gdn_kernel();
            auto outputs = kernel(
                {q_ref, k_ref, v_ref, g_ref, beta_ref, state_ref, t_arr},
                {{B, T, Hv, Dv}, state_ref.shape()},
                {input_dtype, state_dtype},
                {32, Dv, B * Hv},
                {32, 4, 1},
                {{"InT", input_dtype}, {"StT", state_dtype},
                 {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}},
                std::nullopt, false, {});
            new (dst_y->buf) array(outputs[0]);
            new (dst_state->buf) array(outputs[1]);
            return;
        }

        // Fallback: ops-based recurrence
        int repeat_factor = Hv / Hk;
        auto q_exp = (repeat_factor > 1) ? repeat(q_ref, repeat_factor, 2) : q_ref;
        auto k_exp = (repeat_factor > 1) ? repeat(k_ref, repeat_factor, 2) : k_ref;
        auto state = state_ref;
        std::vector<array> ys;
        ys.reserve(T);
        for (int t = 0; t < T; ++t) {
            auto q_t = squeeze(slice(q_exp, {0,t,0,0}, {B,t+1,Hv,Dk}), 1);
            auto k_t = squeeze(slice(k_exp, {0,t,0,0}, {B,t+1,Hv,Dk}), 1);
            auto v_t = squeeze(slice(v_ref, {0,t,0,0}, {B,t+1,Hv,Dv}), 1);
            auto g_t = squeeze(slice(g_ref, {0,t,0}, {B,t+1,Hv}), 1);
            auto beta_t = squeeze(slice(beta_ref, {0,t,0}, {B,t+1,Hv}), 1);
            auto g_4d = reshape(g_t, {B,Hv,1,1});
            auto decayed = multiply(state, g_4d);
            auto k_4d = reshape(k_t, {B,Hv,1,Dk});
            auto kv_mem = sum(multiply(decayed, k_4d), {-1}, false);
            auto beta_3d = reshape(beta_t, {B,Hv,1});
            auto delta = multiply(subtract(v_t, kv_mem), beta_3d);
            auto delta_4d = reshape(delta, {B,Hv,Dv,1});
            state = add(decayed, multiply(k_4d, delta_4d));
            auto q_4d = reshape(q_t, {B,Hv,1,Dk});
            ys.push_back(astype(sum(multiply(state, q_4d), {-1}, false), q_ref.dtype()));
        }
        auto y = (T == 1) ? reshape(ys[0], {B,1,Hv,Dv}) : stack(ys, 1);
        new (dst_y->buf) array(y);
        new (dst_state->buf) array(state);
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("gdn_metal_step", e.what());
        new (dst_y->buf) array(0.0f);
        new (dst_state->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("gdn_metal_step", "unknown C++ exception");
        new (dst_y->buf) array(0.0f);
        new (dst_state->buf) array(0.0f);
    }
}

} // extern "C"
