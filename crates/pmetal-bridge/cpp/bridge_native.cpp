// Gated-delta-net Metal kernels: the recurrence step and the state-only update.
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
// Follows the DFlash reference implementation (z-lab, 2026), adapted to
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
            pmetal_bridge_clear_error_internal();
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
        bridge_placeholder(dst_state);
    } catch (...) {
        pmetal_bridge_set_last_error("gdn_metal_state_update", "unknown C++ exception");
        bridge_placeholder(dst_state);
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
            pmetal_bridge_clear_error_internal();
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
        bridge_placeholder(dst_y);
        bridge_placeholder(dst_state);
    } catch (...) {
        pmetal_bridge_set_last_error("gdn_metal_step", "unknown C++ exception");
        bridge_placeholder(dst_y);
        bridge_placeholder(dst_state);
    }
}

} // extern "C"
