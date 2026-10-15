// Generic binding of `mlx::core::fast::metal_kernel`.
//
// A caller supplies the body of a Metal kernel as source text plus its input
// and output names; the first call with a given kernel name builds MLX's
// kernel function and every later call reuses it (MLX in turn caches the
// compiled library per template instantiation). The kernel runs on the GPU
// stream like any other op, in the same lazy graph, so its inputs and outputs
// are ordinary arrays.
//
// A custom kernel has no gradient: MLX cannot differentiate through it.

#include "bridge_internal.h"

#include <map>
#include <memory>
#include <tuple>
#include <variant>

namespace {

namespace mx = mlx::core;

struct CachedKernel {
    std::string source;
    std::string header;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    bool ensure_row_contiguous;
    std::unique_ptr<mx::fast::CustomKernelFunction> fn;
};

std::mutex& kernel_mutex() {
    static auto* m = new std::mutex;  // leaked: no destructor to order against libmlx's
    return *m;
}

std::map<std::string, CachedKernel>& kernel_cache() {
    static auto* cache = new std::map<std::string, CachedKernel>;
    return *cache;
}

std::vector<std::string> to_strings(const char* const* names, int n) {
    std::vector<std::string> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) out.emplace_back(names[i]);
    return out;
}

}  // namespace

extern "C" {

// Run the kernel `name`. `outputs` points at `n_out` uninitialised slots, all
// of which are initialised on every path (placeholders on failure).
//
// Template arguments: `tmpl_kinds[i]` is 0 for an int (`tmpl_vals[i]`) and 1
// for a dtype code.
void pmetal_runtime_metal_kernel(
    mlx_inline_array* outputs,
    int n_out,
    const char* name,
    const char* source,
    const char* header,
    const char* const* in_names,
    int n_in,
    const char* const* out_names,
    const mlx_inline_array* const* inputs,
    const int* out_shapes,
    const int* out_ndims,
    const int* out_dtypes,
    const int* grid,
    const int* threadgroup,
    const char* const* tmpl_names,
    const int* tmpl_kinds,
    const int* tmpl_vals,
    int n_tmpl,
    bool ensure_row_contiguous) {
    try {
        mx::fast::CustomKernelFunction* fn = nullptr;
        {
            std::lock_guard<std::mutex> lock(kernel_mutex());
            auto& cache = kernel_cache();
            auto it = cache.find(name);
            if (it == cache.end()) {
                CachedKernel k;
                k.source = source;
                k.header = header;
                k.input_names = to_strings(in_names, n_in);
                k.output_names = to_strings(out_names, n_out);
                k.ensure_row_contiguous = ensure_row_contiguous;
                k.fn = std::make_unique<mx::fast::CustomKernelFunction>(mx::fast::metal_kernel(
                    name, k.input_names, k.output_names, k.source, k.header,
                    ensure_row_contiguous, /*atomic_outputs=*/false));
                it = cache.emplace(name, std::move(k)).first;
            } else if (it->second.source != source) {
                throw std::invalid_argument(
                    std::string("a different Metal kernel is already registered as `") + name + "`");
            }
            fn = it->second.fn.get();
        }

        std::vector<array> args;
        args.reserve(n_in);
        for (int i = 0; i < n_in; ++i) args.push_back(as_arr(inputs[i]));

        std::vector<mx::Shape> shapes;
        std::vector<mx::Dtype> dtypes;
        const int* shape_at = out_shapes;
        for (int i = 0; i < n_out; ++i) {
            shapes.emplace_back(shape_at, shape_at + out_ndims[i]);
            shape_at += out_ndims[i];
            dtypes.push_back(dtype_from_int(out_dtypes[i]));
        }

        std::vector<std::pair<std::string, mx::fast::TemplateArg>> template_args;
        for (int i = 0; i < n_tmpl; ++i) {
            if (tmpl_kinds[i] == 0) {
                template_args.emplace_back(tmpl_names[i], tmpl_vals[i]);
            } else {
                template_args.emplace_back(tmpl_names[i], dtype_from_int(tmpl_vals[i]));
            }
        }

        auto results = (*fn)(
            args, shapes, dtypes,
            std::make_tuple(grid[0], grid[1], grid[2]),
            std::make_tuple(threadgroup[0], threadgroup[1], threadgroup[2]),
            template_args, std::nullopt, false, {});
        if (static_cast<int>(results.size()) != n_out) {
            throw std::runtime_error("the Metal kernel returned an unexpected number of outputs");
        }
        for (int i = 0; i < n_out; ++i) {
            new (outputs[i].buf) array(std::move(results[i]));
        }
        pmetal_runtime_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_runtime_set_last_error("metal_kernel", e.what());
        for (int i = 0; i < n_out; ++i) bridge_placeholder(&outputs[i]);
    } catch (...) {
        pmetal_runtime_set_last_error("metal_kernel", "unknown C++ exception");
        for (int i = 0; i < n_out; ++i) bridge_placeholder(&outputs[i]);
    }
}

}  // extern "C"
