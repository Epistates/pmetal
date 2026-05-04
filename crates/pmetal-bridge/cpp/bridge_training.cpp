// Training operations, autograd, and model-migration helper ops.
// Extracted from bridge.cpp for maintainability.

#include "bridge_internal.h"
#include <numeric>

extern "C" {

// ── Training ops: random ─────────────────────────────────────────────────────

void mlx_inline_random_normal(mlx_inline_array* dst, const int* shape, int ndim, int dtype) {
    BRIDGE_TRY_DST("random_normal", dst,
        new (dst->buf) array(mlx::core::random::normal(
        {shape, shape + ndim}, dtype_from_int(dtype))));
}

void mlx_inline_random_uniform(mlx_inline_array* dst, const int* shape, int ndim, int dtype) {
    BRIDGE_TRY_DST("random_uniform", dst,
        new (dst->buf) array(mlx::core::random::uniform(
        {shape, shape + ndim}, dtype_from_int(dtype))));
}

void mlx_inline_random_bernoulli(mlx_inline_array* dst, const mlx_inline_array* p, const int* shape, int ndim) {
    BRIDGE_TRY_DST("random_bernoulli", dst,
        new (dst->buf) array(mlx::core::random::bernoulli(
        as_arr(p), {shape, shape + ndim})));
}

void mlx_inline_random_seed(uint64_t seed) {
    mlx::core::random::seed(seed);
}

void mlx_inline_random_randint(mlx_inline_array* dst, int low, int high, const int* shape, int ndim, int dtype) {
    BRIDGE_TRY_DST("random_randint", dst,
        new (dst->buf) array(mlx::core::random::randint(
        low, high, {shape, shape + ndim}, dtype_from_int(dtype))));
}

// ── Training ops: math ───────────────────────────────────────────────────────

void mlx_inline_mean_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    BRIDGE_TRY_DST("mean_axis", dst,
        new (dst->buf) array(mlx::core::mean(as_arr(a), axis, keepdims)));
}

void mlx_inline_mean_all(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("mean_all", dst,
        new (dst->buf) array(mlx::core::mean(as_arr(a))));
}

void mlx_inline_pow(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("pow", dst,
        new (dst->buf) array(mlx::core::power(as_arr(a), as_arr(b))));
}

void mlx_inline_reciprocal(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("reciprocal", dst,
        new (dst->buf) array(mlx::core::reciprocal(as_arr(a))));
}

void mlx_inline_sin(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("sin", dst,
        new (dst->buf) array(mlx::core::sin(as_arr(a))));
}

void mlx_inline_cos(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("cos", dst,
        new (dst->buf) array(mlx::core::cos(as_arr(a))));
}

void mlx_inline_clip(mlx_inline_array* dst, const mlx_inline_array* a,
                     const mlx_inline_array* lo, const mlx_inline_array* hi) {
    BRIDGE_TRY_DST("clip", dst, {
        auto low_opt  = lo ? std::optional<array>(as_arr(lo)) : std::nullopt;
        auto high_opt = hi ? std::optional<array>(as_arr(hi)) : std::nullopt;
        new (dst->buf) array(mlx::core::clip(as_arr(a), low_opt, high_opt));
    });
}

void mlx_inline_log_softmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    // log_softmax(x) = x - logsumexp(x, axis, keepdims=true)
    BRIDGE_TRY_DST("log_softmax", dst, {
        const auto& x = as_arr(a);
        auto lse = mlx::core::logsumexp(x, axis, true);
        new (dst->buf) array(mlx::core::subtract(x, lse));
    });
}

void mlx_inline_cross_entropy(mlx_inline_array* dst, const mlx_inline_array* logits,
                               const mlx_inline_array* targets, int axis) {
    BRIDGE_TRY_DST("cross_entropy", dst, {
        // cross_entropy = -sum(targets * log_softmax(logits), axis=axis)
                const auto& l = as_arr(logits);
                auto lse      = mlx::core::logsumexp(l, axis, true);
                auto log_probs = mlx::core::subtract(l, lse);
                new (dst->buf) array(mlx::core::negative(
                    mlx::core::sum(mlx::core::multiply(as_arr(targets), log_probs), axis, false)));
    });
}

void mlx_inline_cross_entropy_sparse(mlx_inline_array* dst, const mlx_inline_array* logits,
                                      const mlx_inline_array* indices, int axis) {
    BRIDGE_TRY_DST("cross_entropy_sparse", dst, {
        // NLL = logsumexp(logits, axis) - take_along_axis(logits, indices, axis)
        // Output shape == logits.shape with `axis` removed.
        const auto& l = as_arr(logits);
        const auto& idx = as_arr(indices);
        auto idx_exp = mlx::core::expand_dims(idx, axis);
        auto gathered = mlx::core::take_along_axis(l, idx_exp, axis);
        auto score = mlx::core::squeeze(gathered, axis);
        auto lse = mlx::core::logsumexp(l, axis, false);
        new (dst->buf) array(mlx::core::subtract(lse, score));
    });
}

void mlx_inline_square(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("square", dst,
        new (dst->buf) array(mlx::core::square(as_arr(a))));
}

// ── Training ops: creation ───────────────────────────────────────────────────

void mlx_inline_full(mlx_inline_array* dst, const int* shape, int ndim, float val, int dtype) {
    BRIDGE_TRY_DST("full", dst, {
        new (dst->buf) array(mlx::core::full(
                    {shape, shape + ndim}, val, dtype_from_int(dtype)));
    });
}

void mlx_inline_eye(mlx_inline_array* dst, int n, int dtype) {
    BRIDGE_TRY_DST("eye", dst, {
        // eye(n, m=n, k=0, dtype)
                new (dst->buf) array(mlx::core::eye(n, n, 0, dtype_from_int(dtype)));
    });
}

void mlx_inline_tri(mlx_inline_array* dst, int n, int m, int k, int dtype) {
    BRIDGE_TRY_DST("tri", dst,
        new (dst->buf) array(mlx::core::tri(n, m, k, dtype_from_int(dtype))));
}

// ── Training ops: shape ──────────────────────────────────────────────────────

void mlx_inline_broadcast_to(mlx_inline_array* dst, const mlx_inline_array* a,
                              const int* shape, int ndim) {
    BRIDGE_TRY_DST("broadcast_to", dst, {
        new (dst->buf) array(mlx::core::broadcast_to(
                    as_arr(a), {shape, shape + ndim}));
    });
}

void mlx_inline_flatten(mlx_inline_array* dst, const mlx_inline_array* a,
                        int start_axis, int end_axis) {
    BRIDGE_TRY_DST("flatten", dst,
        new (dst->buf) array(mlx::core::flatten(as_arr(a), start_axis, end_axis)));
}

// ── Training ops: sort/reduction ─────────────────────────────────────────────

void mlx_inline_argsort(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    BRIDGE_TRY_DST("argsort", dst,
        new (dst->buf) array(mlx::core::argsort(as_arr(a), axis)));
}

void mlx_inline_sum_all(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("sum_all", dst,
        new (dst->buf) array(mlx::core::sum(as_arr(a))));
}

void mlx_inline_max_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    BRIDGE_TRY_DST("max_axis", dst,
        new (dst->buf) array(mlx::core::max(as_arr(a), axis, keepdims)));
}

void mlx_inline_min_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    BRIDGE_TRY_DST("min_axis", dst,
        new (dst->buf) array(mlx::core::min(as_arr(a), axis, keepdims)));
}

void mlx_inline_minimum(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("minimum", dst,
        new (dst->buf) array(mlx::core::minimum(as_arr(a), as_arr(b))));
}

// ── Training ops: activation ─────────────────────────────────────────────────

void mlx_inline_relu(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("relu", dst,
        new (dst->buf) array(mlx::core::maximum(as_arr(a), array(0.0f))));
}

void mlx_inline_gelu(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("gelu", dst, {
        // GELU fast approx: x * sigmoid(1.702 * x)
                const auto& x = as_arr(a);
                new (dst->buf) array(mlx::core::multiply(
                    x, mlx::core::sigmoid(mlx::core::multiply(array(1.702f), x))));
    });
}

// ── Training ops: comparison ─────────────────────────────────────────────────

void mlx_inline_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("equal", dst,
        new (dst->buf) array(mlx::core::equal(as_arr(a), as_arr(b))));
}

void mlx_inline_not_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("not_equal", dst,
        new (dst->buf) array(mlx::core::not_equal(as_arr(a), as_arr(b))));
}

void mlx_inline_greater(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("greater", dst,
        new (dst->buf) array(mlx::core::greater(as_arr(a), as_arr(b))));
}

void mlx_inline_less(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("less", dst,
        new (dst->buf) array(mlx::core::less(as_arr(a), as_arr(b))));
}

void mlx_inline_greater_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("greater_equal", dst,
        new (dst->buf) array(mlx::core::greater_equal(as_arr(a), as_arr(b))));
}

void mlx_inline_less_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("less_equal", dst,
        new (dst->buf) array(mlx::core::less_equal(as_arr(a), as_arr(b))));
}

// ── Training ops: serialization ──────────────────────────────────────────────

void mlx_inline_save_safetensors(const char* path, const char** keys,
                                  const mlx_inline_array* arrays, int count) {
    BRIDGE_TRY_VOID("save_safetensors", {
        std::unordered_map<std::string, array> map;
        map.reserve(count);
        for (int i = 0; i < count; i++) {
            map.emplace(std::string(keys[i]), as_arr(&arrays[i]));
        }
        mlx::core::save_safetensors(std::string(path), std::move(map));
    });
}

// ── Training ops: quantize ───────────────────────────────────────────────────

void mlx_inline_quantize(mlx_inline_array* dst_w, mlx_inline_array* dst_scales,
                          mlx_inline_array* dst_biases,
                          const mlx_inline_array* a, int group_size, int bits) {
    // Multi-output: BRIDGE_TRY_DST targets a single dst, so use manual try/catch
    // and placement-new scalar-zero sentinels into all three slots on failure
    // so Rust drops never call ~array() on uninit memory.
    try {
        auto result = mlx::core::quantize(as_arr(a), group_size, bits);
        new (dst_w->buf)      array(std::move(result[0]));
        new (dst_scales->buf) array(std::move(result[1]));
        new (dst_biases->buf) array(std::move(result[2]));
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("quantize", e.what());
        new (dst_w->buf)      array(0.0f);
        new (dst_scales->buf) array(0.0f);
        new (dst_biases->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("quantize", "unknown C++ exception");
        new (dst_w->buf)      array(0.0f);
        new (dst_scales->buf) array(0.0f);
        new (dst_biases->buf) array(0.0f);
    }
}

void mlx_inline_quantize_mode(mlx_inline_array* dst_w, mlx_inline_array* dst_scales,
                               const mlx_inline_array* a, int group_size, int bits, int mode) {
    // Multi-output op for MLX floating-point quantization modes such as mxfp8.
    try {
        auto result = mlx::core::quantize(as_arr(a), group_size, bits, quant_mode_from_int(mode));
        new (dst_w->buf)      array(std::move(result[0]));
        new (dst_scales->buf) array(std::move(result[1]));
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("quantize_mode", e.what());
        new (dst_w->buf)      array(0.0f);
        new (dst_scales->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("quantize_mode", "unknown C++ exception");
        new (dst_w->buf)      array(0.0f);
        new (dst_scales->buf) array(0.0f);
    }
}

// ── Training ops: multi-axis sum/mean ────────────────────────────────────────

void mlx_inline_sum_axes(mlx_inline_array* dst, const mlx_inline_array* a,
                          const int* axes, int num_axes, bool keepdims) {
    BRIDGE_TRY_DST("sum_axes", dst,
        new (dst->buf) array(mlx::core::sum(
        as_arr(a), {axes, axes + num_axes}, keepdims)));
}

void mlx_inline_mean_axes(mlx_inline_array* dst, const mlx_inline_array* a,
                           const int* axes, int num_axes, bool keepdims) {
    BRIDGE_TRY_DST("mean_axes", dst,
        new (dst->buf) array(mlx::core::mean(
        as_arr(a), {axes, axes + num_axes}, keepdims)));
}

// ── Training ops: misc ───────────────────────────────────────────────────────

size_t mlx_inline_size(const mlx_inline_array* a) {
    return as_arr(a).size();
}

size_t mlx_inline_nbytes(const mlx_inline_array* a) {
    return as_arr(a).nbytes();
}

int mlx_inline_data_ptr(const mlx_inline_array* a, const void** out_ptr) {
    *out_ptr = as_arr(a).data<void>();
    return 0;
}

uintptr_t mlx_inline_array_id(const mlx_inline_array* a) {
    // array::id() returns `uintptr_t(array_desc_.get())` — stable for the
    // lifetime of the underlying ArrayDesc, valid on lazy (unevaluated)
    // arrays. Used as a cheap identity for change-detection caches.
    return as_arr(a).id();
}

void mlx_inline_stop_gradient(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("stop_gradient", dst,
        new (dst->buf) array(mlx::core::stop_gradient(as_arr(a))));
}

void mlx_inline_tri_inv(mlx_inline_array* dst, const mlx_inline_array* a, bool upper, bool use_cpu) {
    // tri_inv has no VJP in MLX — used in WY factorization as a fixed preconditioner.
    // use_cpu=true routes execution to the CPU device (matching mlx-lm's StreamOrDevice::cpu()).
    BRIDGE_TRY_DST("tri_inv", dst, {
        mlx::core::StreamOrDevice stream = use_cpu
            ? mlx::core::StreamOrDevice{mlx::core::Device(mlx::core::Device::cpu)}
            : mlx::core::StreamOrDevice{};
        new (dst->buf) array(mlx::core::linalg::tri_inv(as_arr(a), upper, stream));
    });
}

void mlx_inline_svd(
    mlx_inline_array* dst_u,
    mlx_inline_array* dst_s,
    mlx_inline_array* dst_vt,
    const mlx_inline_array* a)
{
    // SVD always runs on CPU (GPU SVD not available in MLX).
    // Returns economy / thin SVD: U[m,k], S[k], Vt[k,n] where k=min(m,n).
    // Multi-output: manual try/catch with sentinel-zero on failure.
    try {
        mlx::core::StreamOrDevice cpu_stream{mlx::core::Device(mlx::core::Device::cpu)};
        auto result = mlx::core::linalg::svd(as_arr(a), /* compute_uv */ true, cpu_stream);
        new (dst_u->buf)  array(result[0]);
        new (dst_s->buf)  array(result[1]);
        new (dst_vt->buf) array(result[2]);
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("svd", e.what());
        new (dst_u->buf)  array(0.0f);
        new (dst_s->buf)  array(0.0f);
        new (dst_vt->buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("svd", "unknown C++ exception");
        new (dst_u->buf)  array(0.0f);
        new (dst_s->buf)  array(0.0f);
        new (dst_vt->buf) array(0.0f);
    }
}

// ── Autograd: value_and_grad ─────────────────────────────────────────────────
//
// Callback-based autograd bridge. The Rust caller provides a function pointer
// that builds the MLX computation graph through InlineArray ops (the forward
// pass). MLX traces this graph and differentiates it w.r.t. the first
// n_params inputs.
//
// forward_fn signature (called from C++):
//   - all_arrays[0..n_params-1]   — parameters being differentiated
//   - all_arrays[n_params..n_total-1] — additional non-differentiated inputs
//   - loss_out                    — write the scalar loss here
//
// Returns: loss → loss_out, gradients → grads_out[0..n_params-1]
// (mlx_rust_forward_fn typedef is declared in bridge.h)

void mlx_inline_value_and_grad(
    mlx_rust_forward_fn forward_fn,
    void* ctx,
    const mlx_inline_array* const* all_arrays,
    int n_params,
    int n_total,
    mlx_inline_array* loss_out,
    mlx_inline_array** grads_out
) {
    // Snapshot input arrays (they will be moved into the lambda capture).
    std::vector<array> inputs;
    inputs.reserve(n_total);
    for (int i = 0; i < n_total; i++) {
        inputs.push_back(as_arr(all_arrays[i]));
    }

    // C++ closure that calls back into Rust to build the forward graph.
    // The lambda takes a std::vector<array> (MLX convention) and returns one.
    auto cpp_forward = [&](std::vector<array> args) -> std::vector<array> {
        // Wrap each array in a temporary InlineArray so Rust can read them.
        std::vector<mlx_inline_array> bufs(args.size());
        std::vector<const mlx_inline_array*> ptrs(args.size());
        for (size_t i = 0; i < args.size(); i++) {
            new (bufs[i].buf) array(args[i]);
            ptrs[i] = &bufs[i];
        }

        mlx_inline_array loss_buf;
        mlx_inline_init_empty(&loss_buf);
        forward_fn(ptrs.data(), (int)ptrs.size(), &loss_buf, ctx);

        array result = as_arr(&loss_buf);

        // Destroy temporaries (arrays, not loss_buf which we've already copied).
        for (auto& b : bufs) {
            as_arr(&b).~array();
        }
        as_arr(&loss_buf).~array();

        return {result};
    };

    // argnums: differentiate w.r.t. the first n_params inputs.
    std::vector<int> argnums(n_params);
    std::iota(argnums.begin(), argnums.end(), 0);

    // mlx::core::value_and_grad returns a function that produces
    // pair<vector<array>, vector<array>> = (values, grads).
    try {
        auto vg_fn = mlx::core::value_and_grad(
            std::function<std::vector<array>(std::vector<array>)>(cpp_forward),
            argnums);
        auto result = vg_fn(inputs);
        auto& values = result.first;
        auto& grads = result.second;

        new (loss_out->buf) array(values[0]);
        for (int i = 0; i < n_params; i++) {
            new (grads_out[i]->buf) array(grads[i]);
        }
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("value_and_grad", e.what());
        // Return scalar NaN loss and zero gradients so the training loop can detect failure.
        new (loss_out->buf) array(std::numeric_limits<float>::quiet_NaN());
        for (int i = 0; i < n_params; i++) {
            new (grads_out[i]->buf) array(0.0f);
        }
    } catch (...) {
        pmetal_bridge_set_last_error("value_and_grad", "unknown C++ exception");
        new (loss_out->buf) array(std::numeric_limits<float>::quiet_NaN());
        for (int i = 0; i < n_params; i++) {
            new (grads_out[i]->buf) array(0.0f);
        }
    }
}

// ── FFT ops ──────────────────────────────────────────────────────────────────

void mlx_inline_rfft(mlx_inline_array* dst, const mlx_inline_array* a, int n_fft, int axis) {
    BRIDGE_TRY_DST("rfft", dst, {
        const auto& x = as_arr(a);
                if (n_fft < 0) {
                    new (dst->buf) array(mlx::core::fft::rfft(x, axis));
                } else {
                    new (dst->buf) array(mlx::core::fft::rfft(x, n_fft, axis));
                };
    });
}

void mlx_inline_irfft(mlx_inline_array* dst, const mlx_inline_array* a, int n_fft, int axis) {
    BRIDGE_TRY_DST("irfft", dst, {
        const auto& x = as_arr(a);
                if (n_fft < 0) {
                    new (dst->buf) array(mlx::core::fft::irfft(x, axis));
                } else {
                    new (dst->buf) array(mlx::core::fft::irfft(x, n_fft, axis));
                };
    });
}

// ── leaky_relu ────────────────────────────────────────────────────────────────

void mlx_inline_leaky_relu(mlx_inline_array* dst, const mlx_inline_array* a, float neg_slope) {
    BRIDGE_TRY_DST("leaky_relu", dst, {
        const auto& x = as_arr(a);
                new (dst->buf) array(mlx::core::maximum(
                    mlx::core::multiply(x, array(neg_slope)),
                    x));
    });
}

// ── squeeze all size-1 axes ────────────────────────────────────────────────────

void mlx_inline_squeeze_all(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("squeeze_all", dst, {
        const auto& x = as_arr(a);
                std::vector<int> axes;
                for (int i = 0; i < (int)x.ndim(); ++i) {
                    if (x.shape(i) == 1) axes.push_back(i);
                }
                if (axes.empty()) {
                    new (dst->buf) array(x);
                } else {
                    new (dst->buf) array(mlx::core::squeeze(x, axes));
                };
    });
}

// ── pad ───────────────────────────────────────────────────────────────────────

void mlx_inline_pad(mlx_inline_array* dst, const mlx_inline_array* a,
                    const int* pad_widths, int ndim, float fill_value) {
    BRIDGE_TRY_DST("pad", dst, {
        const auto& x = as_arr(a);
                std::vector<std::pair<int,int>> pw(ndim);
                for (int i = 0; i < ndim; ++i) {
                    pw[i] = { pad_widths[2*i], pad_widths[2*i+1] };
                }
                new (dst->buf) array(mlx::core::pad(x, pw, array(fill_value)));
    });
}

// ── Missing ops for pmetal-models migration ───────────────────────────────────

void mlx_inline_rsqrt(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("rsqrt", dst,
        new (dst->buf) array(mlx::core::rsqrt(as_arr(a))));
}

void mlx_inline_zeros_like(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("zeros_like", dst,
        new (dst->buf) array(mlx::core::zeros_like(as_arr(a))));
}

void mlx_inline_ones_like(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("ones_like", dst,
        new (dst->buf) array(mlx::core::ones_like(as_arr(a))));
}

void mlx_inline_tile(mlx_inline_array* dst, const mlx_inline_array* a, const int* reps, int ndim) {
    BRIDGE_TRY_DST("tile", dst, {
        std::vector<int> r(reps, reps + ndim);
                new (dst->buf) array(mlx::core::tile(as_arr(a), r));
    });
}

void mlx_inline_linspace(mlx_inline_array* dst, float start, float stop, int n, int dtype) {
    BRIDGE_TRY_DST("linspace", dst,
        new (dst->buf) array(mlx::core::linspace(start, stop, n, dtype_from_int(dtype))));
}

// Multi-output (writes up to `sections` slots into dst_arr). Uses a manual
// try/catch because BRIDGE_TRY_DST targets a single dst buffer.
void mlx_inline_split_sections(mlx_inline_array* dst_arr, const mlx_inline_array* a,
                                int sections, int axis, int* out_count) {
    try {
        auto parts = mlx::core::split(as_arr(a), sections, axis);
        *out_count = (int)parts.size();
        for (int i = 0; i < (int)parts.size(); i++) {
            new (dst_arr[i].buf) array(parts[i]);
        }
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("split_sections", e.what());
        *out_count = 0;
    } catch (...) {
        pmetal_bridge_set_last_error("split_sections", "unknown C++ exception");
        *out_count = 0;
    }
}

void mlx_inline_scatter_add(mlx_inline_array* dst, const mlx_inline_array* a,
                             const mlx_inline_array* indices, const mlx_inline_array* updates,
                             int axis) {
    BRIDGE_TRY_DST("scatter_add", dst,
        new (dst->buf) array(mlx::core::scatter_add(as_arr(a), as_arr(indices), as_arr(updates), axis)));
}

void mlx_inline_topk(mlx_inline_array* dst, const mlx_inline_array* a, int k, int axis) {
    BRIDGE_TRY_DST("topk", dst,
        new (dst->buf) array(mlx::core::topk(as_arr(a), k, axis)));
}

void mlx_inline_put_along_axis(mlx_inline_array* dst, const mlx_inline_array* a,
                                const mlx_inline_array* indices, const mlx_inline_array* values,
                                int axis) {
    BRIDGE_TRY_DST("put_along_axis", dst,
        new (dst->buf) array(mlx::core::put_along_axis(as_arr(a), as_arr(indices), as_arr(values), axis)));
}

void mlx_inline_layer_norm(mlx_inline_array* dst, const mlx_inline_array* x,
                            const mlx_inline_array* weight, const mlx_inline_array* bias,
                            float eps) {
    BRIDGE_TRY_DST("layer_norm", dst, {
        auto w_opt = weight ? std::optional<array>(as_arr(weight)) : std::nullopt;
                auto b_opt = bias   ? std::optional<array>(as_arr(bias))   : std::nullopt;
                new (dst->buf) array(mlx::core::fast::layer_norm(as_arr(x), w_opt, b_opt, eps));
    });
}

void mlx_inline_addmm(mlx_inline_array* dst, const mlx_inline_array* c,
                       const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("addmm", dst,
        new (dst->buf) array(mlx::core::addmm(as_arr(c), as_arr(a), as_arr(b))));
}

void mlx_inline_conv2d(mlx_inline_array* dst, const mlx_inline_array* input,
                       const mlx_inline_array* weight,
                       int stride_h, int stride_w, int pad_h, int pad_w,
                       int dil_h, int dil_w, int groups) {
    BRIDGE_TRY_DST("conv2d", dst, {
        new (dst->buf) array(mlx::core::conv2d(
                    as_arr(input), as_arr(weight),
                    {stride_h, stride_w}, {pad_h, pad_w},
                    {dil_h, dil_w}, groups));
    });
}

// ── Gradient checkpointing ───────────────────────────────────────────────────
//
// Wraps a Rust forward function with mlx::core::checkpoint() so that
// intermediate activations are discarded after the forward pass and
// recomputed on-demand during the backward pass.  This reduces peak
// activation memory from O(layers × batch × seq × hidden) to O(1 layer)
// at the cost of one extra forward pass per backward pass.
//
// Signature mirrors mlx_rust_forward_fn but with a vector of outputs:
//
//   forward_fn(all_arrays, n_total, outputs_out, n_outputs_out, ctx)
//
// where `outputs_out` is a caller-allocated array of mlx_inline_array
// with capacity n_outputs_max, and `*n_outputs_out` is set by the callback
// to the actual number of output arrays it produced.
//
// The bridge:
//   1. Snapshots inputs into a std::vector<array>.
//   2. Builds a cpp_forward lambda that invokes forward_fn via InlineArray bufs.
//   3. Wraps that lambda with checkpoint().
//   4. Calls the wrapped function with the inputs.
//   5. Writes outputs via placement-new into dst_outputs[0..n_outputs-1].
//
// n_outputs_max must equal the number of outputs the forward_fn will produce.

typedef void (*mlx_rust_checkpoint_fn)(
    const mlx_inline_array* const* all_arrays,
    int n_total,
    mlx_inline_array* outputs_out,
    int* n_outputs_out,
    void* ctx
);

void mlx_inline_checkpoint(
    mlx_rust_checkpoint_fn forward_fn,
    void* ctx,
    const mlx_inline_array* const* all_arrays,
    int n_total,
    int n_outputs_max,
    mlx_inline_array* dst_outputs,
    int* n_outputs_written
) {
    // Snapshot inputs so the lambda can capture by value.
    std::vector<array> inputs;
    inputs.reserve(n_total);
    for (int i = 0; i < n_total; i++) {
        inputs.push_back(as_arr(all_arrays[i]));
    }

    // Lambda that calls back into Rust to build the forward graph.
    // Returns a std::vector<array> matching the outputs the callback emits.
    auto cpp_forward = [&](const std::vector<array>& args) -> std::vector<array> {
        // Wrap each array as a temporary InlineArray for the Rust callback.
        std::vector<mlx_inline_array> bufs(args.size());
        std::vector<const mlx_inline_array*> ptrs(args.size());
        for (size_t i = 0; i < args.size(); i++) {
            new (bufs[i].buf) array(args[i]);
            ptrs[i] = &bufs[i];
        }

        // Allocate output buffer for the Rust callback.
        std::vector<mlx_inline_array> out_bufs(n_outputs_max);
        for (int i = 0; i < n_outputs_max; i++) {
            mlx_inline_init_empty(&out_bufs[i]);
        }
        int n_out = 0;
        forward_fn(ptrs.data(), (int)ptrs.size(), out_bufs.data(), &n_out, ctx);

        // Collect outputs before destroying the bufs.
        std::vector<array> results;
        results.reserve(n_out);
        for (int i = 0; i < n_out; i++) {
            results.push_back(as_arr(&out_bufs[i]));
            as_arr(&out_bufs[i]).~array();
        }
        // Remaining output slots that were never initialised are still
        // default-initialised (mlx_inline_init_empty) — destroy them too.
        for (int i = n_out; i < n_outputs_max; i++) {
            as_arr(&out_bufs[i]).~array();
        }

        // Destroy input wrappers.
        for (auto& b : bufs) {
            as_arr(&b).~array();
        }

        return results;
    };

    try {
        auto checkpointed = mlx::core::checkpoint(
            std::function<std::vector<array>(const std::vector<array>&)>(cpp_forward));
        auto results = checkpointed(inputs);

        int n = (int)results.size();
        for (int i = 0; i < n && i < n_outputs_max; i++) {
            new (dst_outputs[i].buf) array(std::move(results[i]));
        }
        *n_outputs_written = n;
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("checkpoint", e.what());
        *n_outputs_written = 0;
    } catch (...) {
        pmetal_bridge_set_last_error("checkpoint", "unknown C++ exception");
        *n_outputs_written = 0;
    }
}

} // extern "C"
