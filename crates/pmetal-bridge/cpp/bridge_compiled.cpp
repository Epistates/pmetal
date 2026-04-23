// Fused compiled operations — @mx.compile(shapeless=True) equivalents.
// Extracted from bridge.cpp for maintainability.

#include "bridge_internal.h"

// ============================================================================
// Fused compiled ops — matching Python's @mx.compile(shapeless=True)
// Each creates a compiled closure on first call, caches it, and replays.
// This produces a single Compiled graph node instead of N separate nodes.
// Must be outside extern "C" for C++ template/lambda support.
// ============================================================================

using namespace mlx::core;
using CompiledFn = std::function<std::vector<array>(const std::vector<array>&)>;

// Heap-allocate and intentionally leak compiled functions.  Function-local
// static CompiledFn objects destruct at program exit via __cxa_atexit, calling
// compile_erase() into libmlx.dylib's CompilerCache.  Cross-DSO atexit
// ordering is unspecified — the CompilerCache may already be torn down,
// causing SIGSEGV.  Heap-leaked objects are never destroyed, avoiding the
// problem entirely (the OS reclaims all memory at process exit).
static CompiledFn* make_compiled(CompiledFn fn) {
    return new CompiledFn(mlx::core::compile(std::move(fn), /*shapeless=*/true));
}

// shapeless=false: works with ALL primitives (Split, CustomKernel, etc.)
// but only replays the tape when input shapes match the first trace.
// Perfect for T=1 decode where shapes are always the same.
static CompiledFn* make_compiled_fixed(CompiledFn fn) {
    return new CompiledFn(mlx::core::compile(std::move(fn), /*shapeless=*/false));
}

extern "C" {

void mlx_inline_fused_swiglu(mlx_inline_array* dst,
    const mlx_inline_array* gate, const mlx_inline_array* up) {
    BRIDGE_TRY_DST("fused_swiglu", dst, {
        static auto* compiled = make_compiled(
            [](const std::vector<array>& inputs) -> std::vector<array> {
                auto& g = inputs[0];
                auto& u = inputs[1];
                return {multiply(multiply(g, sigmoid(g)), u)};
            });
        auto result = (*compiled)({as_arr(gate), as_arr(up)});
        new (dst->buf) array(result[0]);
    });
}

// Tanh-approximation GELU gating matching mlx-lm's `nn.gelu_approx(gate) * up`.
// Structure: 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3))) * u
// All scalar constants are astype'd to gate.dtype() inside the compiled
// lambda so bf16 inputs stay bf16 (no silent f32 promotion of the whole
// MLP result). Shapeless compile — one trace reused across all layers.
void mlx_inline_fused_geglu_tanh(mlx_inline_array* dst,
    const mlx_inline_array* gate, const mlx_inline_array* up) {
    BRIDGE_TRY_DST("fused_geglu_tanh", dst, {
        static auto* compiled = make_compiled(
            [](const std::vector<array>& inputs) -> std::vector<array> {
                auto& g = inputs[0];
                auto& u = inputs[1];
                auto dt = g.dtype();
                auto half     = astype(array(0.5f),         dt);
                auto one      = astype(array(1.0f),         dt);
                auto sqrt2_pi = astype(array(0.7978845608f), dt);
                auto coef     = astype(array(0.044715f),    dt);
                auto g3       = multiply(multiply(g, g), g);
                auto inner    = add(g, multiply(coef, g3));
                auto t        = tanh(multiply(sqrt2_pi, inner));
                auto gelu_g   = multiply(half, multiply(g, add(one, t)));
                return {multiply(gelu_g, u)};
            });
        auto result = (*compiled)({as_arr(gate), as_arr(up)});
        new (dst->buf) array(result[0]);
    });
}

void mlx_inline_fused_silu(mlx_inline_array* dst, const mlx_inline_array* x) {
    BRIDGE_TRY_DST("fused_silu", dst, {
        static auto* compiled = make_compiled(
            [](const std::vector<array>& inputs) -> std::vector<array> {
                auto& x = inputs[0];
                return {multiply(x, sigmoid(x))};
            });
        auto result = (*compiled)({as_arr(x)});
        new (dst->buf) array(result[0]);
    });
}

void mlx_inline_fused_compute_g(mlx_inline_array* dst,
    const mlx_inline_array* a_log, const mlx_inline_array* a, const mlx_inline_array* dt_bias) {
    BRIDGE_TRY_DST("fused_compute_g", dst, {
        static auto* compiled = make_compiled(
            [](const std::vector<array>& inputs) -> std::vector<array> {
                auto decay = exp(astype(inputs[0], float32));
                auto sp = log1p(exp(add(inputs[1], inputs[2])));
                return {exp(negative(multiply(decay, sp)))};
            });
        auto result = (*compiled)({as_arr(a_log), as_arr(a), as_arr(dt_bias)});
        new (dst->buf) array(result[0]);
    });
}

void mlx_inline_fused_precise_swiglu(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* gate) {
    BRIDGE_TRY_DST("fused_precise_swiglu", dst, {
        static auto* compiled = make_compiled(
            [](const std::vector<array>& inputs) -> std::vector<array> {
                auto& x = inputs[0];
                auto& g = inputs[1];
                auto g32 = multiply(astype(g, float32), sigmoid(astype(g, float32)));
                auto x32 = astype(x, float32);
                return {astype(multiply(g32, x32), x.dtype())};
            });
        auto result = (*compiled)({as_arr(x), as_arr(gate)});
        new (dst->buf) array(result[0]);
    });
}

// shapeless=false version — fixed shapes, works with ALL primitives.
void mlx_inline_compiled_gdn_layer_fixed(
    mlx_inline_array* dst_out,
    mlx_inline_array* dst_conv_state,
    mlx_inline_array* dst_ssm_state,
    const mlx_inline_array* normed,
    const mlx_inline_array* qkv_w, const mlx_inline_array* z_w,
    const mlx_inline_array* b_w, const mlx_inline_array* a_w,
    const mlx_inline_array* conv_w,
    const mlx_inline_array* q_nw, const mlx_inline_array* k_nw,
    const mlx_inline_array* a_log, const mlx_inline_array* dt_bias,
    const mlx_inline_array* norm_w, const mlx_inline_array* out_w,
    const mlx_inline_array* conv_state_in, const mlx_inline_array* ssm_state_in,
    int nv, int nk, int dk, int dv, int cd, int ck, int kd, float norm_eps
) {
    // Heap-leaked to avoid cross-DSO static destructor ordering crash.
    static CompiledFn* compiled = nullptr;
    try {
        if (!compiled) {
            int NV=nv, NK=nk, DK=dk, DV=dv, CD=cd, CK=ck, KD=kd;
            float EPS=norm_eps;
            compiled = make_compiled_fixed(
                [NV, NK, DK, DV, CD, CK, KD, EPS](const std::vector<array>& ins) -> std::vector<array> {
                    auto& normed = ins[0];
                    auto& qkv_w = ins[1]; auto& z_w = ins[2];
                    auto& b_w = ins[3]; auto& a_w = ins[4]; auto& conv_w = ins[5];
                    auto& q_nw = ins[6]; auto& k_nw = ins[7];
                    auto& a_log_arr = ins[8]; auto& dt_bias_arr = ins[9];
                    auto& norm_w_arr = ins[10]; auto& out_w = ins[11];
                    auto& conv_state = ins[12]; auto& ssm_state = ins[13];
                    int B = normed.shape(0); int S = normed.shape(1);

                    auto qkv = matmul(normed, qkv_w);
                    auto z = reshape(matmul(normed, z_w), {B, S, NV, DV});
                    auto b_val = matmul(normed, b_w);
                    auto a_val = matmul(normed, a_w);

                    auto conv_in = concatenate({conv_state, qkv}, 1);
                    auto new_conv = slice(conv_in, {0, 1, 0}, {B, CK, CD});
                    auto conv_out = mlx::core::conv1d(conv_in, conv_w, 1, 0, 1, CD);
                    auto conv_act = multiply(conv_out, sigmoid(conv_out));

                    // Single split → 3 siblings sharing one Split primitive (matches Python).
                    auto conv_parts = split(conv_act, Shape{KD, KD * 2}, -1);
                    auto q = fast::rms_norm(reshape(conv_parts[0], {B, S, NK, DK}), q_nw, EPS);
                    auto k = fast::rms_norm(reshape(conv_parts[1], {B, S, NK, DK}), k_nw, EPS);
                    auto v = reshape(conv_parts[2], {B, S, NV, DV});

                    auto g = exp(negative(multiply(exp(astype(a_log_arr, float32)),
                                 log1p(exp(add(a_val, dt_bias_arr))))));
                    auto beta = sigmoid(b_val);

                    auto& kernel = get_gdn_kernel();
                    auto kout = kernel(
                        {q, k, v, g, beta, ssm_state, array(S)},
                        {{B, S, NV, DV}, ssm_state.shape()},
                        {q.dtype(), ssm_state.dtype()},
                        {32, DV, B * NV}, {32, 4, 1},
                        {{"InT", q.dtype()}, {"StT", ssm_state.dtype()},
                         {"Dk", DK}, {"Dv", DV}, {"Hk", NK}, {"Hv", NV}},
                        std::nullopt, false, {});

                    auto out_n = fast::rms_norm(kout[0], norm_w_arr, EPS);
                    auto g32 = multiply(astype(z, float32), sigmoid(astype(z, float32)));
                    auto output = matmul(
                        reshape(astype(multiply(g32, astype(out_n, float32)), q.dtype()),
                                {B, S, NV * DV}), out_w);
                    return {output, new_conv, kout[1]};
                });
        }

        auto result = (*compiled)({
            as_arr(normed),
            as_arr(qkv_w), as_arr(z_w), as_arr(b_w), as_arr(a_w), as_arr(conv_w),
            as_arr(q_nw), as_arr(k_nw), as_arr(a_log), as_arr(dt_bias),
            as_arr(norm_w), as_arr(out_w), as_arr(conv_state_in), as_arr(ssm_state_in)
        });
        new (dst_out->buf) array(result[0]);
        new (dst_conv_state->buf) array(result[1]);
        new (dst_ssm_state->buf) array(result[2]);
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("compiled_gdn_layer_fixed", e.what());
        new (dst_out->buf) array(0.0f);
        new (dst_conv_state->buf) array(0.0f);
        new (dst_ssm_state->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("compiled_gdn_layer_fixed", "unknown C++ exception");
        new (dst_out->buf) array(0.0f);
        new (dst_conv_state->buf) array(0.0f);
        new (dst_ssm_state->buf) array(0.0f);
    }
}

void mlx_inline_compiled_attn_layer_fixed(
    mlx_inline_array* dst_out,
    mlx_inline_array* dst_cache_keys,
    mlx_inline_array* dst_cache_vals,
    const mlx_inline_array* normed,
    const mlx_inline_array* q_w,
    const mlx_inline_array* k_w,
    const mlx_inline_array* v_w,
    const mlx_inline_array* o_w,
    const mlx_inline_array* q_nw,
    const mlx_inline_array* k_nw,
    const mlx_inline_array* cache_keys_in,
    const mlx_inline_array* cache_vals_in,
    int kv_offset,
    int rope_offset,
    int n_heads,
    int n_kv,
    int head_dim,
    float scale,
    int rope_dims,
    float rope_base,
    float rope_scale,
    float q_norm_eps,
    float k_norm_eps,
    bool gated
) {
    struct Entry {
        int batch;
        int cache_len;
        int n_heads;
        int n_kv;
        int head_dim;
        int rope_dims;
        int gated;
        CompiledFn compiled;
    };
    static auto* entries = new std::vector<Entry>();

    try {
        int batch = as_arr(normed).shape(0);
        int cache_len = as_arr(cache_keys_in).shape(2);

        CompiledFn* compiled = nullptr;
        for (auto& entry : *entries) {
            if (entry.batch == batch
                && entry.cache_len == cache_len
                && entry.n_heads == n_heads
                && entry.n_kv == n_kv
                && entry.head_dim == head_dim
                && entry.rope_dims == rope_dims
                && entry.gated == static_cast<int>(gated)) {
                compiled = &entry.compiled;
                break;
            }
        }

        if (compiled == nullptr) {
            int NH = n_heads;
            int NKV = n_kv;
            int HD = head_dim;
            int RD = rope_dims;
            int L = cache_len;
            bool GATED = gated;
            float SCALE = scale;
            float RBASE = rope_base;
            float RSCALE = rope_scale;
            float QEPS = q_norm_eps;
            float KEPS = k_norm_eps;

            entries->push_back(Entry{
                batch,
                cache_len,
                n_heads,
                n_kv,
                head_dim,
                rope_dims,
                static_cast<int>(gated),
                *make_compiled_fixed(
                    [NH, NKV, HD, RD, L, GATED, SCALE, RBASE, RSCALE, QEPS, KEPS]
                    (const std::vector<array>& ins) -> std::vector<array> {
                        using namespace mlx::core;

                        auto& normed = ins[0];
                        auto& q_w = ins[1];
                        auto& k_w = ins[2];
                        auto& v_w = ins[3];
                        auto& o_w = ins[4];
                        auto& q_nw = ins[5];
                        auto& k_nw = ins[6];
                        auto& cache_keys = ins[7];
                        auto& cache_vals = ins[8];
                        auto& kv_offset_arr = ins[9];
                        auto& rope_offset_arr = ins[10];

                        int B = normed.shape(0);
                        int S = normed.shape(1);

                        auto q_proj = matmul(normed, q_w);
                        array queries(0.0f);
                        array gate(0.0f);
                        if (GATED) {
                            auto qg = reshape(q_proj, {B, S, NH, HD * 2});
                            auto qg_parts = split(qg, Shape{HD}, -1);
                            queries = qg_parts[0];
                            gate = reshape(qg_parts[1], {B, S, NH * HD});
                        } else {
                            queries = reshape(q_proj, {B, S, NH, HD});
                        }

                        auto new_k = matmul(normed, k_w);
                        auto new_v = matmul(normed, v_w);

                        queries = fast::rms_norm(queries, q_nw, QEPS);
                        auto keys = fast::rms_norm(reshape(new_k, {B, S, NKV, HD}), k_nw, KEPS);
                        auto values = reshape(new_v, {B, S, NKV, HD});

                        queries = transpose(queries, {0, 2, 1, 3});
                        keys = transpose(keys, {0, 2, 1, 3});
                        values = transpose(values, {0, 2, 1, 3});

                        queries = fast::rope(queries, RD, false, RBASE, RSCALE, rope_offset_arr);
                        keys = fast::rope(keys, RD, false, RBASE, RSCALE, rope_offset_arr);

                        auto kv_indices = broadcast_to(
                            reshape(kv_offset_arr, {1, 1, 1, 1}),
                            {B, NKV, S, HD});
                        auto updated_keys = put_along_axis(cache_keys, kv_indices, keys, 2);
                        auto updated_vals = put_along_axis(cache_vals, kv_indices, values, 2);

                        auto next_offset = add(kv_offset_arr, array(S));
                        auto positions = reshape(arange(L, int32), {1, 1, 1, L});
                        auto valid_mask = less(positions, reshape(next_offset, {1, 1, 1, 1}));

                        auto output = fast::scaled_dot_product_attention(
                            queries, updated_keys, updated_vals, SCALE, "", valid_mask);
                        output = transpose(output, {0, 2, 1, 3});
                        output = reshape(output, {B, S, NH * HD});
                        if (GATED) {
                            output = multiply(output, sigmoid(gate));
                        }
                        auto result = matmul(output, o_w);
                        return {result, updated_keys, updated_vals};
                    })
            });
            compiled = &entries->back().compiled;
        }

        auto result = (*compiled)({
            as_arr(normed),
            as_arr(q_w),
            as_arr(k_w),
            as_arr(v_w),
            as_arr(o_w),
            as_arr(q_nw),
            as_arr(k_nw),
            as_arr(cache_keys_in),
            as_arr(cache_vals_in),
            array(kv_offset),
            array(rope_offset),
        });
        new (dst_out->buf) array(result[0]);
        new (dst_cache_keys->buf) array(result[1]);
        new (dst_cache_vals->buf) array(result[2]);
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("compiled_attn_layer_fixed", e.what());
        new (dst_out->buf) array(0.0f);
        new (dst_cache_keys->buf) array(0.0f);
        new (dst_cache_vals->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("compiled_attn_layer_fixed", "unknown C++ exception");
        new (dst_out->buf) array(0.0f);
        new (dst_cache_keys->buf) array(0.0f);
        new (dst_cache_vals->buf) array(0.0f);
    }
}

void mlx_inline_compiled_moe_layer_fixed(
    mlx_inline_array* dst_out,
    const mlx_inline_array* x,
    const mlx_inline_array* router_w,
    const mlx_inline_array* moe_gate_w,
    const mlx_inline_array* moe_up_w,
    const mlx_inline_array* moe_down_w,
    const mlx_inline_array* shared_gate_w,
    const mlx_inline_array* shared_up_w,
    const mlx_inline_array* shared_down_w,
    const mlx_inline_array* shared_expert_gate_w,
    int top_k,
    bool norm_topk_prob
) {
    struct Entry {
        int batch;
        int hidden;
        int num_experts;
        int routed_hidden;
        int shared_hidden;
        int top_k;
        int dtype;
        int norm_topk_prob;
        CompiledFn compiled;
    };
    static auto* entries = new std::vector<Entry>();

    try {
        int batch = as_arr(x).shape(0);
        int hidden = as_arr(x).shape(2);
        int num_experts = as_arr(router_w).shape(1);
        int routed_hidden = as_arr(moe_down_w).shape(1);
        int shared_hidden = as_arr(shared_down_w).shape(0);
        int dtype = static_cast<int>(as_arr(x).dtype().val());

        CompiledFn* compiled = nullptr;
        for (auto& entry : *entries) {
            if (entry.batch == batch
                && entry.hidden == hidden
                && entry.num_experts == num_experts
                && entry.routed_hidden == routed_hidden
                && entry.shared_hidden == shared_hidden
                && entry.top_k == top_k
                && entry.dtype == dtype
                && entry.norm_topk_prob == static_cast<int>(norm_topk_prob)) {
                compiled = &entry.compiled;
                break;
            }
        }

        if (compiled == nullptr) {
            int TOPK = top_k;
            bool NORM_TOPK = norm_topk_prob;
            entries->push_back(Entry{
                batch,
                hidden,
                num_experts,
                routed_hidden,
                shared_hidden,
                top_k,
                dtype,
                static_cast<int>(norm_topk_prob),
                *make_compiled_fixed(
                    [TOPK, NORM_TOPK](const std::vector<array>& ins) -> std::vector<array> {
                        using namespace mlx::core;

                        auto& x = ins[0];
                        auto& router_w = ins[1];
                        auto& moe_gate_w = ins[2];
                        auto& moe_up_w = ins[3];
                        auto& moe_down_w = ins[4];
                        auto& shared_gate_w = ins[5];
                        auto& shared_up_w = ins[6];
                        auto& shared_down_w = ins[7];
                        auto& shared_expert_gate_w = ins[8];

                        int B = x.shape(0);
                        int S = x.shape(1);
                        int H = x.shape(2);
                        int expert_count = router_w.shape(1);

                        auto x_flat = reshape(x, {B * S, H});
                        auto gates = softmax(matmul(x_flat, router_w), -1, /*precise=*/true);
                        auto all_inds = argpartition(gates, -TOPK, -1);
                        auto inds = slice(all_inds, {0, expert_count - TOPK}, {B * S, expert_count});
                        auto scores = take_along_axis(gates, inds, -1);
                        if (NORM_TOPK) {
                            scores = divide(scores, sum(scores, {-1}, true));
                        }

                        auto switch_in = expand_dims(expand_dims(x_flat, 1), 2);
                        auto rhs_indices = std::optional<array>(inds);
                        auto x_gate_exp =
                            gather_mm(switch_in, moe_gate_w, std::nullopt, rhs_indices, false);
                        auto x_up_exp =
                            gather_mm(switch_in, moe_up_w, std::nullopt, rhs_indices, false);
                        auto x_act = multiply(multiply(x_gate_exp, sigmoid(x_gate_exp)), x_up_exp);
                        auto y_exp =
                            squeeze(gather_mm(x_act, moe_down_w, std::nullopt, rhs_indices, false), 2);
                        auto scores_exp = reshape(scores, {B * S, TOPK, 1});
                        auto y_routed = sum(multiply(y_exp, scores_exp), {-2}, false);

                        auto sh_gate = matmul(x_flat, shared_gate_w);
                        auto sh_up = matmul(x_flat, shared_up_w);
                        auto sh_act = multiply(multiply(sh_gate, sigmoid(sh_gate)), sh_up);
                        auto sh_out = matmul(sh_act, shared_down_w);
                        auto sh_scale = sigmoid(matmul(x_flat, shared_expert_gate_w));
                        auto y_shared = multiply(sh_out, sh_scale);

                        return {reshape(add(y_routed, y_shared), {B, S, H})};
                    })
            });
            compiled = &entries->back().compiled;
        }

        auto result = (*compiled)({
            as_arr(x),
            as_arr(router_w),
            as_arr(moe_gate_w),
            as_arr(moe_up_w),
            as_arr(moe_down_w),
            as_arr(shared_gate_w),
            as_arr(shared_up_w),
            as_arr(shared_down_w),
            as_arr(shared_expert_gate_w),
        });
        new (dst_out->buf) array(result[0]);
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("compiled_moe_layer_fixed", e.what());
        new (dst_out->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("compiled_moe_layer_fixed", "unknown C++ exception");
        new (dst_out->buf) array(0.0f);
    }
}

// ----------------------------------------------------------------------------
// Gemma 4 fused decoder layer halves
// ----------------------------------------------------------------------------
//
// Two compiled functions cover the entire Gemma 4 layer except the residual
// adds and the per-layer scalar multiply (which are 3 cheap element-wise
// ops left outside to keep the FFI surface narrow):
//
//   * `mlx_inline_compiled_gemma4_attn_block`
//        x → input_layernorm → q/k/v projections (k_eq_v variant supported)
//          → q_norm / k_norm / v_norm-no-scale
//          → transpose to [B,H,L,D] → partial RoPE via custom `freqs`
//          → KV cache update (put_along_axis) → SDPA → o_proj
//          → post_attention_layernorm
//        (out, cache_keys', cache_vals')
//
//   * `mlx_inline_compiled_gemma4_mlp_block`
//        x → pre_feedforward_layernorm → gate_proj / up_proj
//          → tanh-approx GELU → multiply → down_proj → post_feedforward_layernorm
//        (out)
//
// Each call site keeps a list of `Entry` records keyed by shape signature
// (batch, seq_len, cache_len, n_heads, n_kv, head_dim, k_eq_v, has_freqs)
// so prefill-vs-decode and sliding-vs-full layers each get their own
// shapeless=false compiled trace.

void mlx_inline_compiled_gemma4_attn_block(
    mlx_inline_array* dst_out,
    mlx_inline_array* dst_cache_keys,
    mlx_inline_array* dst_cache_vals,
    // Wider-than-Qwen3 compiled graph: includes input_layernorm at the
    // top and post_attention_layernorm at the bottom. Empirically, for
    // Gemma 4 31B this gives ~15-20% better decode time than the
    // narrower Qwen3-style variant — the per-op cost of launching two
    // extra `fast::rms_norm` kernels outside the compile outweighs the
    // savings from a smaller compile trace.
    const mlx_inline_array* x,
    const mlx_inline_array* in_norm_w,
    const mlx_inline_array* q_w,
    const mlx_inline_array* k_w,
    const mlx_inline_array* v_w,           // may be null when use_k_eq_v
    const mlx_inline_array* o_w,
    const mlx_inline_array* q_norm_w,
    const mlx_inline_array* k_norm_w,
    const mlx_inline_array* post_norm_w,
    const mlx_inline_array* rope_freqs,    // may be null for full rotation
    const mlx_inline_array* cache_keys_in,
    const mlx_inline_array* cache_vals_in,
    int kv_offset,
    int n_heads,
    int n_kv,
    int head_dim,
    float in_norm_eps,
    float qk_norm_eps,
    float post_norm_eps,
    int sliding_window,                    // 0 = causal, >0 = sliding window
    bool use_k_eq_v,
    float rope_base,                       // ignored when rope_freqs != null
    int rope_dims                          // = head_dim for partial freqs path
) {
    struct Entry {
        int batch;
        int seq_len;
        int cache_len;
        int n_heads;
        int n_kv;
        int head_dim;
        int k_eq_v;
        int has_freqs;
        int sliding_window;
        int dtype;
        CompiledFn compiled;
    };
    static auto* entries = new std::vector<Entry>();

    try {
        int batch = as_arr(x).shape(0);
        int seq_len = as_arr(x).shape(1);
        int cache_len = as_arr(cache_keys_in).shape(2);
        int dtype = static_cast<int>(as_arr(x).dtype().val());
        int has_freqs = (rope_freqs != nullptr) ? 1 : 0;

        CompiledFn* compiled = nullptr;
        for (auto& entry : *entries) {
            if (entry.batch == batch
                && entry.seq_len == seq_len
                && entry.cache_len == cache_len
                && entry.n_heads == n_heads
                && entry.n_kv == n_kv
                && entry.head_dim == head_dim
                && entry.k_eq_v == static_cast<int>(use_k_eq_v)
                && entry.has_freqs == has_freqs
                && entry.sliding_window == sliding_window
                && entry.dtype == dtype) {
                compiled = &entry.compiled;
                break;
            }
        }

        if (compiled == nullptr) {
            int NH = n_heads;
            int NKV = n_kv;
            int HD = head_dim;
            int RD = rope_dims;
            int L = cache_len;
            int SW = sliding_window;
            bool KEV = use_k_eq_v;
            bool HAS_FREQS = has_freqs == 1;
            float INE = in_norm_eps;
            float QKE = qk_norm_eps;
            float PNE = post_norm_eps;
            float RBASE = rope_base;

            entries->push_back(Entry{
                batch,
                seq_len,
                cache_len,
                n_heads,
                n_kv,
                head_dim,
                static_cast<int>(use_k_eq_v),
                has_freqs,
                sliding_window,
                dtype,
                *make_compiled_fixed(
                    [NH, NKV, HD, RD, L, SW, KEV, HAS_FREQS, INE, QKE, PNE, RBASE]
                    (const std::vector<array>& ins) -> std::vector<array> {
                        using namespace mlx::core;

                        std::size_t idx = 0;
                        const array& x = ins[idx++];
                        const array& in_norm_w = ins[idx++];
                        const array& q_w = ins[idx++];
                        const array& k_w = ins[idx++];
                        // v_w is only present when !KEV, but we always pass
                        // a placeholder to keep the input vector shape stable.
                        const array& v_w = ins[idx++];
                        const array& o_w = ins[idx++];
                        const array& q_norm_w = ins[idx++];
                        const array& k_norm_w = ins[idx++];
                        const array& post_norm_w = ins[idx++];
                        const array& rope_freqs_arr = ins[idx++];
                        const array& cache_keys = ins[idx++];
                        const array& cache_vals = ins[idx++];
                        const array& kv_offset_arr = ins[idx++];

                        int B = x.shape(0);
                        int S = x.shape(1);

                        // 1. Input layernorm.
                        auto normed = fast::rms_norm(x, in_norm_w, INE);

                        // 2. Q/K/V projections. Weights are expected in
                        // `[in, out]` form (pre-transposed by the caller
                        // at load time — matches the qwen3_native
                        // pattern, avoids per-step strided matmul).
                        auto q_proj = matmul(normed, q_w);
                        auto k_proj = matmul(normed, k_w);
                        // For attention_k_eq_v full layers, values are taken
                        // from the *raw* k_proj output BEFORE k_norm — we keep
                        // a copy here and skip the v_proj matmul.
                        array v_pre = k_proj;
                        if (!KEV) {
                            v_pre = matmul(normed, v_w);
                        }

                        auto q4 = reshape(q_proj, {B, S, NH, HD});
                        auto k4 = reshape(k_proj, {B, S, NKV, HD});
                        auto v4 = reshape(v_pre, {B, S, NKV, HD});

                        // 2. Per-head norms. v uses the no-scale variant.
                        auto q = fast::rms_norm(q4, q_norm_w, QKE);
                        auto k = fast::rms_norm(k4, k_norm_w, QKE);
                        auto v = fast::rms_norm(v4, std::nullopt, QKE);

                        // 3. Transpose to [B, H, L, D].
                        q = transpose(q, {0, 2, 1, 3});
                        k = transpose(k, {0, 2, 1, 3});
                        v = transpose(v, {0, 2, 1, 3});

                        // 4. RoPE — partial via custom freqs, or full base.
                        if (HAS_FREQS) {
                            q = fast::rope(q, HD, false, std::nullopt, 1.0f,
                                           kv_offset_arr,
                                           std::optional<array>(rope_freqs_arr));
                            k = fast::rope(k, HD, false, std::nullopt, 1.0f,
                                           kv_offset_arr,
                                           std::optional<array>(rope_freqs_arr));
                        } else {
                            q = fast::rope(q, RD, false,
                                           std::optional<float>(RBASE), 1.0f,
                                           kv_offset_arr);
                            k = fast::rope(k, RD, false,
                                           std::optional<float>(RBASE), 1.0f,
                                           kv_offset_arr);
                        }

                        // 5. KV cache write. For prefill `S > 1` the indices
                        // must vary per token: position `kv_offset + s` for
                        // each `s in 0..S`. For decode `S == 1` this
                        // collapses to a single scalar. Without the
                        // `arange(S)` term every prefill token would be
                        // scatter-written to the same slot, clobbering
                        // earlier positions.
                        auto seq_range = reshape(arange(S, int32), {1, 1, S, 1});
                        auto kv_indices = broadcast_to(
                            add(seq_range, reshape(kv_offset_arr, {1, 1, 1, 1})),
                            {B, NKV, S, HD});
                        auto updated_keys = put_along_axis(cache_keys, kv_indices, k, 2);
                        auto updated_vals = put_along_axis(cache_vals, kv_indices, v, 2);

                        // 6. Build the attention validity mask. For Gemma 4
                        //    we only need to mask out trailing junk in the
                        //    pre-allocated cache (positions >= next_offset)
                        //    AND, for sliding layers, keys outside the window
                        //    behind the current decode position.
                        auto next_offset = add(kv_offset_arr, array(S));
                        auto positions = reshape(arange(L, int32), {1, 1, 1, L});
                        auto valid_mask = less(positions, reshape(next_offset, {1, 1, 1, 1}));
                        if (SW > 0) {
                            auto window_start = subtract(next_offset, array(SW));
                            auto in_window = greater_equal(
                                positions,
                                reshape(window_start, {1, 1, 1, 1}));
                            valid_mask = logical_and(valid_mask, in_window);
                        }

                        // 7. SDPA (scale=1.0, Gemma 4 bakes scale into weights).
                        auto output = fast::scaled_dot_product_attention(
                            q, updated_keys, updated_vals, 1.0f, "", valid_mask);
                        output = transpose(output, {0, 2, 1, 3});
                        output = reshape(output, {B, S, NH * HD});

                        // 8. Output projection + post_attention_layernorm.
                        auto attn_out = matmul(output, o_w);
                        auto post = fast::rms_norm(attn_out, post_norm_w, PNE);
                        return {post, updated_keys, updated_vals};
                    })
            });
            compiled = &entries->back().compiled;
        }

        // Build the input vector. We always emit a v_w slot — when
        // use_k_eq_v=true the caller passes q_w as a placeholder (the
        // compiled lambda branches on KEV and ignores that slot). Same
        // for rope_freqs: when null the caller passes a 1-element scalar
        // dummy. This keeps the input vector shape stable across calls.
        const mlx_inline_array* v_input = use_k_eq_v ? q_w : v_w;
        static array dummy_freqs(0.0f);
        const array& freqs_input = (rope_freqs != nullptr)
            ? as_arr(rope_freqs)
            : dummy_freqs;

        auto result = (*compiled)({
            as_arr(x),
            as_arr(in_norm_w),
            as_arr(q_w),
            as_arr(k_w),
            as_arr(v_input),
            as_arr(o_w),
            as_arr(q_norm_w),
            as_arr(k_norm_w),
            as_arr(post_norm_w),
            freqs_input,
            as_arr(cache_keys_in),
            as_arr(cache_vals_in),
            array(kv_offset),
        });
        new (dst_out->buf) array(result[0]);
        new (dst_cache_keys->buf) array(result[1]);
        new (dst_cache_vals->buf) array(result[2]);
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("compiled_gemma4_attn_block", e.what());
        new (dst_out->buf) array(0.0f);
        new (dst_cache_keys->buf) array(0.0f);
        new (dst_cache_vals->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("compiled_gemma4_attn_block", "unknown C++ exception");
        new (dst_out->buf) array(0.0f);
        new (dst_cache_keys->buf) array(0.0f);
        new (dst_cache_vals->buf) array(0.0f);
    }
}

void mlx_inline_compiled_gemma4_shared_attn_decode(
    mlx_inline_array* dst_out,
    const mlx_inline_array* x,
    const mlx_inline_array* in_norm_w,
    const mlx_inline_array* q_w,
    const mlx_inline_array* o_w,
    const mlx_inline_array* q_norm_w,
    const mlx_inline_array* post_norm_w,
    const mlx_inline_array* rope_freqs,
    const mlx_inline_array* cache_keys_in,
    const mlx_inline_array* cache_vals_in,
    int valid_kv_len,
    int rope_offset,
    int n_heads,
    int n_kv,
    int head_dim,
    float in_norm_eps,
    float q_norm_eps,
    float post_norm_eps,
    int sliding_window,
    float rope_base,
    int rope_dims
) {
    struct Entry {
        int batch;
        int seq_len;
        int cache_len;
        int n_heads;
        int n_kv;
        int head_dim;
        int has_freqs;
        int sliding_window;
        int dtype;
        CompiledFn compiled;
    };
    static auto* entries = new std::vector<Entry>();

    try {
        int batch = as_arr(x).shape(0);
        int seq_len = as_arr(x).shape(1);
        int cache_len = as_arr(cache_keys_in).shape(2);
        int dtype = static_cast<int>(as_arr(x).dtype().val());
        int has_freqs = (rope_freqs != nullptr) ? 1 : 0;

        CompiledFn* compiled = nullptr;
        for (auto& entry : *entries) {
            if (entry.batch == batch
                && entry.seq_len == seq_len
                && entry.cache_len == cache_len
                && entry.n_heads == n_heads
                && entry.n_kv == n_kv
                && entry.head_dim == head_dim
                && entry.has_freqs == has_freqs
                && entry.sliding_window == sliding_window
                && entry.dtype == dtype) {
                compiled = &entry.compiled;
                break;
            }
        }

        if (compiled == nullptr) {
            int NH = n_heads;
            int NKV = n_kv;
            int HD = head_dim;
            int RD = rope_dims;
            int L = cache_len;
            int SW = sliding_window;
            bool HAS_FREQS = has_freqs == 1;
            float INE = in_norm_eps;
            float QNE = q_norm_eps;
            float PNE = post_norm_eps;
            float RBASE = rope_base;

            entries->push_back(Entry{
                batch,
                seq_len,
                cache_len,
                n_heads,
                n_kv,
                head_dim,
                has_freqs,
                sliding_window,
                dtype,
                *make_compiled_fixed(
                    [NH, NKV, HD, RD, L, SW, HAS_FREQS, INE, QNE, PNE, RBASE]
                    (const std::vector<array>& ins) -> std::vector<array> {
                        using namespace mlx::core;

                        std::size_t idx = 0;
                        const array& x = ins[idx++];
                        const array& in_norm_w = ins[idx++];
                        const array& q_w = ins[idx++];
                        const array& o_w = ins[idx++];
                        const array& q_norm_w = ins[idx++];
                        const array& post_norm_w = ins[idx++];
                        const array& rope_freqs_arr = ins[idx++];
                        const array& cache_keys = ins[idx++];
                        const array& cache_vals = ins[idx++];
                        const array& valid_kv_len_arr = ins[idx++];
                        const array& rope_offset_arr = ins[idx++];

                        int B = x.shape(0);
                        int S = x.shape(1);

                        auto normed = fast::rms_norm(x, in_norm_w, INE);
                        auto q_proj = matmul(normed, q_w);
                        auto q = reshape(q_proj, {B, S, NH, HD});
                        q = fast::rms_norm(q, q_norm_w, QNE);
                        q = transpose(q, {0, 2, 1, 3});

                        if (HAS_FREQS) {
                            q = fast::rope(q, HD, false, std::nullopt, 1.0f,
                                           rope_offset_arr,
                                           std::optional<array>(rope_freqs_arr));
                        } else {
                            q = fast::rope(q, RD, false,
                                           std::optional<float>(RBASE), 1.0f,
                                           rope_offset_arr);
                        }

                        auto positions = reshape(arange(L, int32), {1, 1, 1, L});
                        auto valid_mask = less(
                            positions, reshape(valid_kv_len_arr, {1, 1, 1, 1}));
                        if (SW > 0) {
                            auto window_start = subtract(valid_kv_len_arr, array(SW));
                            auto in_window = greater_equal(
                                positions,
                                reshape(window_start, {1, 1, 1, 1}));
                            valid_mask = logical_and(valid_mask, in_window);
                        }

                        auto output = fast::scaled_dot_product_attention(
                            q, cache_keys, cache_vals, 1.0f, "", valid_mask);
                        output = transpose(output, {0, 2, 1, 3});
                        output = reshape(output, {B, S, NH * HD});

                        auto attn_out = matmul(output, o_w);
                        auto post = fast::rms_norm(attn_out, post_norm_w, PNE);
                        return {post};
                    })
            });
            compiled = &entries->back().compiled;
        }

        static array dummy_freqs(0.0f);
        const array& freqs_input = (rope_freqs != nullptr)
            ? as_arr(rope_freqs)
            : dummy_freqs;

        auto result = (*compiled)({
            as_arr(x),
            as_arr(in_norm_w),
            as_arr(q_w),
            as_arr(o_w),
            as_arr(q_norm_w),
            as_arr(post_norm_w),
            freqs_input,
            as_arr(cache_keys_in),
            as_arr(cache_vals_in),
            array(valid_kv_len),
            array(rope_offset),
        });
        new (dst_out->buf) array(result[0]);
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("compiled_gemma4_shared_attn_decode", e.what());
        new (dst_out->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("compiled_gemma4_shared_attn_decode", "unknown C++ exception");
        new (dst_out->buf) array(0.0f);
    }
}

void mlx_inline_compiled_gemma4_mlp_block(
    mlx_inline_array* dst_out,
    const mlx_inline_array* x,
    const mlx_inline_array* pre_norm_w,
    const mlx_inline_array* gate_w,
    const mlx_inline_array* up_w,
    const mlx_inline_array* down_w,
    const mlx_inline_array* post_norm_w,
    float pre_norm_eps,
    float post_norm_eps
) {
    struct Entry {
        int batch;
        int seq_len;
        int hidden;
        int intermediate;
        int dtype;
        CompiledFn compiled;
    };
    static auto* entries = new std::vector<Entry>();

    try {
        int batch = as_arr(x).shape(0);
        int seq_len = as_arr(x).shape(1);
        int hidden = as_arr(x).shape(2);
        int intermediate = as_arr(gate_w).shape(1);
        int dtype = static_cast<int>(as_arr(x).dtype().val());

        CompiledFn* compiled = nullptr;
        for (auto& entry : *entries) {
            if (entry.batch == batch
                && entry.seq_len == seq_len
                && entry.hidden == hidden
                && entry.intermediate == intermediate
                && entry.dtype == dtype) {
                compiled = &entry.compiled;
                break;
            }
        }

        if (compiled == nullptr) {
            float PRE = pre_norm_eps;
            float POST = post_norm_eps;
            entries->push_back(Entry{
                batch,
                seq_len,
                hidden,
                intermediate,
                dtype,
                *make_compiled_fixed(
                    [PRE, POST](const std::vector<array>& ins) -> std::vector<array> {
                        using namespace mlx::core;
                        const array& x = ins[0];
                        const array& pre_w = ins[1];
                        const array& gate_w = ins[2];
                        const array& up_w = ins[3];
                        const array& down_w = ins[4];
                        const array& post_w = ins[5];

                        // pre_feedforward_layernorm.
                        auto h = fast::rms_norm(x, pre_w, PRE);

                        // Tanh-approx GELU on gate_proj output, multiply by
                        // up_proj output. Matches mlx-lm's `geglu(gate, x) =
                        // nn.gelu_approx(gate) * x` (sqrt(2/pi)·(g + 0.044715·g^3)).
                        // Weights are expected in `[in, out]` form (caller
                        // pre-transposed at load time).
                        auto gate = matmul(h, gate_w);
                        auto up = matmul(h, up_w);

                        // Scalars MUST be cast to gate.dtype() or bf16
                        // inputs silently promote to f32, forcing every
                        // downstream matmul to rematerialize its weights
                        // in f32 per-op. Same fix as fused_geglu_tanh.
                        auto dt = gate.dtype();
                        auto half = astype(array(0.5f), dt);
                        auto one = astype(array(1.0f), dt);
                        auto sqrt2_pi = astype(array(0.7978845608f), dt);
                        auto coef = astype(array(0.044715f), dt);
                        auto gate3 = multiply(multiply(gate, gate), gate);
                        auto inner = add(gate, multiply(coef, gate3));
                        auto t = tanh(multiply(sqrt2_pi, inner));
                        auto gelu_g = multiply(half, multiply(gate, add(one, t)));

                        auto activated = multiply(gelu_g, up);
                        auto down = matmul(activated, down_w);
                        // post_feedforward_layernorm.
                        auto post = fast::rms_norm(down, post_w, POST);
                        return {post};
                    })
            });
            compiled = &entries->back().compiled;
        }

        auto result = (*compiled)({
            as_arr(x),
            as_arr(pre_norm_w),
            as_arr(gate_w),
            as_arr(up_w),
            as_arr(down_w),
            as_arr(post_norm_w),
        });
        new (dst_out->buf) array(result[0]);
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("compiled_gemma4_mlp_block", e.what());
        new (dst_out->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("compiled_gemma4_mlp_block", "unknown C++ exception");
        new (dst_out->buf) array(0.0f);
    }
}

void mlx_inline_compiled_gemma4_per_layer_input_block(
    mlx_inline_array* dst_out,
    const mlx_inline_array* x,
    const mlx_inline_array* layer_input,
    const mlx_inline_array* gate_w,
    const mlx_inline_array* projection_w,
    const mlx_inline_array* post_norm_w,
    float post_norm_eps
) {
    struct Entry {
        int batch;
        int seq_len;
        int hidden;
        int per_layer_hidden;
        int dtype;
        CompiledFn compiled;
    };
    static auto* entries = new std::vector<Entry>();

    try {
        int batch = as_arr(x).shape(0);
        int seq_len = as_arr(x).shape(1);
        int hidden = as_arr(x).shape(2);
        int per_layer_hidden = as_arr(layer_input).shape(2);
        int dtype = static_cast<int>(as_arr(x).dtype().val());

        CompiledFn* compiled = nullptr;
        for (auto& entry : *entries) {
            if (entry.batch == batch
                && entry.seq_len == seq_len
                && entry.hidden == hidden
                && entry.per_layer_hidden == per_layer_hidden
                && entry.dtype == dtype) {
                compiled = &entry.compiled;
                break;
            }
        }

        if (compiled == nullptr) {
            float POST = post_norm_eps;
            entries->push_back(Entry{
                batch,
                seq_len,
                hidden,
                per_layer_hidden,
                dtype,
                *make_compiled_fixed(
                    [POST](const std::vector<array>& ins) -> std::vector<array> {
                        using namespace mlx::core;
                        const array& x = ins[0];
                        const array& layer_input = ins[1];
                        const array& gate_w = ins[2];
                        const array& projection_w = ins[3];
                        const array& post_norm_w = ins[4];

                        auto gate = matmul(x, gate_w);

                        auto dt = gate.dtype();
                        auto half = astype(array(0.5f), dt);
                        auto one = astype(array(1.0f), dt);
                        auto sqrt2_pi = astype(array(0.7978845608f), dt);
                        auto coef = astype(array(0.044715f), dt);
                        auto gate3 = multiply(multiply(gate, gate), gate);
                        auto inner = add(gate, multiply(coef, gate3));
                        auto t = tanh(multiply(sqrt2_pi, inner));
                        auto gelu_g = multiply(half, multiply(gate, add(one, t)));

                        auto mixed = multiply(gelu_g, layer_input);
                        auto projected = matmul(mixed, projection_w);
                        auto post = fast::rms_norm(projected, post_norm_w, POST);
                        return {add(x, post)};
                    })
            });
            compiled = &entries->back().compiled;
        }

        auto result = (*compiled)({
            as_arr(x),
            as_arr(layer_input),
            as_arr(gate_w),
            as_arr(projection_w),
            as_arr(post_norm_w),
        });
        new (dst_out->buf) array(result[0]);
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("compiled_gemma4_per_layer_input_block", e.what());
        new (dst_out->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("compiled_gemma4_per_layer_input_block", "unknown C++ exception");
        new (dst_out->buf) array(0.0f);
    }
}

} // extern "C"

// ============================================================================
// Generic mlx::core::compile() wrapper — see bridge.h for the ABI docs.
// ============================================================================
//
// Each `BridgeCompiledHandle` owns one compiled closure; the `CompileCtx`
// member holds the Rust callback + ctx + max output count, and the
// lambda captured by `mlx::core::compile()` dereferences through a
// stable pointer into that member. Because the handle is heap-allocated
// and never moved, the captured pointer stays valid for the life of the
// handle.

struct CompileCtx {
    mlx_rust_compile_forward_fn fn;
    void* ctx;
    int n_outputs_max;
};

struct BridgeCompiledHandle {
    CompiledFn fn;
    CompileCtx cctx;
};

// Trampoline: MLX gives us a std::vector<array>; we wrap each as an
// inline buffer, hand control to Rust, then collect the Rust-populated
// outputs back into a std::vector<array>.
//
// The `mlx_inline_init_copy` on each input is the same pattern used by
// `value_and_grad`'s trampoline; it gives Rust a borrowed InlineArray
// whose placement-new'd underlying array is destroyed on the way out.
static std::vector<array> rust_compile_trampoline(
    const std::vector<array>& inputs,
    const CompileCtx* cctx
) {
    // Wrap inputs as mlx_inline_array buffers.
    std::vector<mlx_inline_array> input_bufs(inputs.size());
    std::vector<const mlx_inline_array*> input_ptrs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        new (input_bufs[i].buf) array(inputs[i]);
        input_ptrs[i] = &input_bufs[i];
    }

    // Allocate output buffers — Rust writes into these and updates n_out.
    std::vector<mlx_inline_array> output_bufs(cctx->n_outputs_max);
    for (int i = 0; i < cctx->n_outputs_max; ++i) {
        mlx_inline_init_empty(&output_bufs[i]);
    }

    int n_out = 0;
    cctx->fn(
        input_ptrs.data(),
        static_cast<int>(inputs.size()),
        output_bufs.data(),
        &n_out,
        cctx->ctx
    );

    // Collect results before destroying the output buffers.
    std::vector<array> results;
    results.reserve(static_cast<size_t>(std::max(0, n_out)));
    int n_to_collect = std::max(0, std::min(n_out, cctx->n_outputs_max));
    for (int i = 0; i < n_to_collect; ++i) {
        results.push_back(as_arr(&output_bufs[i]));
    }

    // Destroy all output buffers (including any Rust didn't fill — those
    // still hold the default placement-new from mlx_inline_init_empty).
    for (int i = 0; i < cctx->n_outputs_max; ++i) {
        as_arr(&output_bufs[i]).~array();
    }
    // Destroy input wrappers.
    for (auto& b : input_bufs) {
        as_arr(&b).~array();
    }

    return results;
}

extern "C" {

void* mlx_inline_compile_make(
    mlx_rust_compile_forward_fn fn,
    void* ctx,
    int n_outputs_max,
    bool shapeless
) {
    if (n_outputs_max <= 0) {
        pmetal_bridge_set_last_error("compile_make", "n_outputs_max must be > 0");
        return nullptr;
    }
    try {
        auto* handle = new BridgeCompiledHandle;
        handle->cctx = CompileCtx{fn, ctx, n_outputs_max};
        // Capture a pointer to the handle's own cctx member; it stays
        // alive for as long as the handle (heap-allocated, never moved).
        CompileCtx* cctx_ptr = &handle->cctx;
        auto lambda = [cctx_ptr](const std::vector<array>& inputs) -> std::vector<array> {
            return rust_compile_trampoline(inputs, cctx_ptr);
        };
        handle->fn = mlx::core::compile(std::move(lambda), shapeless);
        pmetal_bridge_clear_error_internal();
        return handle;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("compile_make", e.what());
        return nullptr;
    } catch (...) {
        pmetal_bridge_set_last_error("compile_make", "unknown C++ exception");
        return nullptr;
    }
}

int mlx_inline_compile_call(
    void* compiled_handle,
    const mlx_inline_array* const* inputs,
    int n_inputs,
    mlx_inline_array* outputs,
    int n_outputs_max,
    int* n_outputs_written
) {
    if (!compiled_handle) {
        pmetal_bridge_set_last_error("compile_call", "null handle");
        if (n_outputs_written) *n_outputs_written = 0;
        return -1;
    }
    try {
        auto* h = static_cast<BridgeCompiledHandle*>(compiled_handle);

        std::vector<array> input_arrs;
        input_arrs.reserve(static_cast<size_t>(std::max(0, n_inputs)));
        for (int i = 0; i < n_inputs; ++i) {
            input_arrs.push_back(as_arr(inputs[i]));
        }

        auto results = h->fn(input_arrs);

        int n = std::min(static_cast<int>(results.size()), n_outputs_max);
        for (int i = 0; i < n; ++i) {
            new (outputs[i].buf) array(std::move(results[i]));
        }
        if (n_outputs_written) *n_outputs_written = n;
        pmetal_bridge_clear_error_internal();
        return 0;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("compile_call", e.what());
        if (n_outputs_written) *n_outputs_written = 0;
        return -1;
    } catch (...) {
        pmetal_bridge_set_last_error("compile_call", "unknown C++ exception");
        if (n_outputs_written) *n_outputs_written = 0;
        return -1;
    }
}

void mlx_inline_compile_free(void* compiled_handle) {
    if (compiled_handle) {
        delete static_cast<BridgeCompiledHandle*>(compiled_handle);
    }
}

} // extern "C"

// ============================================================================
// Full Qwen3.5 forward pass — single C++ function, low FFI overhead.
//
// The entire forward pass (embedding, all N layers, final norm, lm_head) runs
// here as pure C++ MLX ops.  No Rust stack frame is entered between ops;
// intermediate arrays are C++ locals, never placement-new'd through the FFI
// bridge.  This eliminates ~1800 Rust<->C++ FFI round trips per decode step.
//
// Important: this path still rebuilds the MLX graph on every token. It is not
// a traced or tape-replayed whole-model decode session.
// ============================================================================
