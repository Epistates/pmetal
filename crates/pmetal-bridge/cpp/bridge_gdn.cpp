// Gated Delta Network (GDN) recurrence bridge.
// `mlx_inline_gdn_update` was previously co-located inside
// bridge_turboquant.cpp; it has no turboquant dependency and
// belongs with the GDN Metal step bridge ops in bridge_native.cpp.
// Extracted here to keep the turboquant family self-contained.

#include "bridge_internal.h"

extern "C" {

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
            pmetal_bridge_clear_error_internal();
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
        new (dst_y->buf) array(0.0f);
        new (dst_state->buf) array(0.0f);
        return -1;
    } catch (...) {
        pmetal_bridge_set_last_error("gdn_update", "unknown C++ exception");
        new (dst_y->buf) array(0.0f);
        new (dst_state->buf) array(0.0f);
        return -1;
    }
}

} // extern "C"
