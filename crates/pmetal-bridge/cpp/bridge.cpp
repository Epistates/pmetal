// Inline array bridge — stores mlx::core::array on the Rust stack.
// Zero heap allocation per op. Direct C++ calls.

#include "bridge.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"  // for typeid on Primitive subclasses
#include <typeinfo>
#include <cstring>
#include <cstdlib>
#include <unordered_set>
#include <sys/sysctl.h>

using mlx::core::array;

static_assert(sizeof(array) <= MLX_ARRAY_SIZE, "MLX_ARRAY_SIZE too small");
static_assert(alignof(array) <= MLX_ARRAY_ALIGN, "MLX_ARRAY_ALIGN too small");

// Placement new/delete helpers
static inline array& as_arr(mlx_inline_array* a) {
    return *reinterpret_cast<array*>(a->buf);
}
static inline const array& as_arr(const mlx_inline_array* a) {
    return *reinterpret_cast<const array*>(a->buf);
}

extern "C" {

void mlx_inline_init_empty(mlx_inline_array* dst) {
    new (dst->buf) array(0.0f);  // MLX array default = scalar 0
}

void mlx_inline_init_copy(mlx_inline_array* dst, const mlx_inline_array* src) {
    new (dst->buf) array(as_arr(src));
}

void mlx_inline_init_move(mlx_inline_array* dst, mlx_inline_array* src) {
    new (dst->buf) array(std::move(as_arr(src)));
}

void mlx_inline_destroy(mlx_inline_array* a) {
    as_arr(a).~array();
}

// Convert from legacy mlx_array handle (for interop with existing mlx-rs code)
void mlx_inline_from_handle(mlx_inline_array* dst, void* handle_ctx) {
    if (handle_ctx) {
        new (dst->buf) array(*static_cast<array*>(handle_ctx));
    } else {
        new (dst->buf) array(0.0f);
    }
}

// Convert TO legacy mlx_array handle
void* mlx_inline_to_handle(const mlx_inline_array* src) {
    return new array(as_arr(src));
}

// ── Core ops — write result directly into caller's stack buffer ──

void mlx_inline_matmul(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::matmul(as_arr(a), as_arr(b)));
}

void mlx_inline_add(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::add(as_arr(a), as_arr(b)));
}

void mlx_inline_multiply(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::multiply(as_arr(a), as_arr(b)));
}

void mlx_inline_subtract(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::subtract(as_arr(a), as_arr(b)));
}

void mlx_inline_divide(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::divide(as_arr(a), as_arr(b)));
}

void mlx_inline_negative(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::negative(as_arr(a)));
}

void mlx_inline_exp(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::exp(as_arr(a)));
}

void mlx_inline_sigmoid(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::sigmoid(as_arr(a)));
}

void mlx_inline_silu(mlx_inline_array* dst, const mlx_inline_array* a) {
    auto& x = as_arr(a);
    new (dst->buf) array(mlx::core::multiply(x, mlx::core::sigmoid(x)));
}

void mlx_inline_softmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::softmax(as_arr(a), axis));
}

void mlx_inline_sqrt(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::sqrt(as_arr(a)));
}

void mlx_inline_transpose(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::transpose(as_arr(a)));
}

void mlx_inline_reshape(mlx_inline_array* dst, const mlx_inline_array* a, const int* shape, int ndim) {
    new (dst->buf) array(mlx::core::reshape(as_arr(a), {shape, shape + ndim}));
}

void mlx_inline_sum_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    new (dst->buf) array(mlx::core::sum(as_arr(a), axis, keepdims));
}

void mlx_inline_astype(mlx_inline_array* dst, const mlx_inline_array* a, int dtype) {
    // Map int dtype codes to MLX Dtype constants
    static const mlx::core::Dtype dtypes[] = {
        mlx::core::bool_,    // 0
        mlx::core::uint8,    // 1
        mlx::core::uint16,   // 2
        mlx::core::uint32,   // 3
        mlx::core::uint64,   // 4
        mlx::core::int8,     // 5
        mlx::core::int16,    // 6
        mlx::core::int32,    // 7
        mlx::core::int64,    // 8
        mlx::core::float16,  // 9
        mlx::core::float32,  // 10
        mlx::core::bfloat16, // 11
        mlx::core::complex64 // 12
    };
    auto dt = (dtype >= 0 && dtype <= 12) ? dtypes[dtype] : mlx::core::float32;
    new (dst->buf) array(mlx::core::astype(as_arr(a), dt));
}

// Gather MM
void mlx_inline_gather_mm(
    mlx_inline_array* dst,
    const mlx_inline_array* a, const mlx_inline_array* b,
    const mlx_inline_array* lhs, const mlx_inline_array* rhs, bool sorted) {
    auto lhs_opt = lhs ? std::optional<array>(as_arr(lhs)) : std::nullopt;
    auto rhs_opt = rhs ? std::optional<array>(as_arr(rhs)) : std::nullopt;
    new (dst->buf) array(mlx::core::gather_mm(as_arr(a), as_arr(b), lhs_opt, rhs_opt, sorted));
}

// Fast ops
void mlx_inline_rms_norm(mlx_inline_array* dst, const mlx_inline_array* x,
                          const mlx_inline_array* weight, float eps) {
    auto w = weight ? std::optional<array>(as_arr(weight)) : std::nullopt;
    new (dst->buf) array(mlx::core::fast::rms_norm(as_arr(x), w, eps));
}

void mlx_inline_rope(mlx_inline_array* dst, const mlx_inline_array* x,
                      int dims, bool traditional, float base, float scale, int offset) {
    new (dst->buf) array(mlx::core::fast::rope(
        as_arr(x), dims, traditional, base, scale, offset));
}

void mlx_inline_sdpa(mlx_inline_array* dst,
                      const mlx_inline_array* q, const mlx_inline_array* k,
                      const mlx_inline_array* v, float scale, const char* mask_mode) {
    std::string mode = mask_mode ? mask_mode : "";
    new (dst->buf) array(mlx::core::fast::scaled_dot_product_attention(
        as_arr(q), as_arr(k), as_arr(v), scale, mode));
}

// Split (writes N+1 arrays into pre-allocated output slots)
void mlx_inline_split(const mlx_inline_array* input, const int* indices, int num_indices,
                       int axis, mlx_inline_array* outputs) {
    auto results = mlx::core::split(as_arr(input), {indices, indices + num_indices}, axis);
    for (size_t i = 0; i < results.size(); i++) {
        new (outputs[i].buf) array(std::move(results[i]));
    }
}

void mlx_inline_concatenate(mlx_inline_array* dst, const mlx_inline_array* arrays,
                              int num, int axis) {
    std::vector<array> arrs;
    arrs.reserve(num);
    for (int i = 0; i < num; i++) arrs.push_back(as_arr(&arrays[i]));
    new (dst->buf) array(mlx::core::concatenate(arrs, axis));
}

void mlx_inline_argpartition(mlx_inline_array* dst, const mlx_inline_array* a, int kth, int axis) {
    new (dst->buf) array(mlx::core::argpartition(as_arr(a), kth, axis));
}

void mlx_inline_take_along_axis(mlx_inline_array* dst, const mlx_inline_array* a,
                                  const mlx_inline_array* indices, int axis) {
    new (dst->buf) array(mlx::core::take_along_axis(as_arr(a), as_arr(indices), axis));
}

// Eval
void mlx_inline_eval(mlx_inline_array* a) { as_arr(a).eval(); }
void mlx_inline_async_eval(mlx_inline_array* a) { mlx::core::async_eval(as_arr(a)); }

// Factory
void mlx_inline_from_f32(mlx_inline_array* dst, float val) { new (dst->buf) array(val); }
void mlx_inline_from_i32(mlx_inline_array* dst, int val) { new (dst->buf) array(val); }

// Query — operate directly on the inline buffer
int mlx_inline_ndim(const mlx_inline_array* a) { return as_arr(a).ndim(); }
int mlx_inline_dim(const mlx_inline_array* a, int axis) {
    int ndim = as_arr(a).ndim();
    int idx = axis < 0 ? axis + ndim : axis;
    return as_arr(a).shape(idx);
}
const int* mlx_inline_shape(const mlx_inline_array* a) { return as_arr(a).shape().data(); }
int mlx_inline_dtype(const mlx_inline_array* a) {
    auto dt = as_arr(a).dtype();
    if (dt == mlx::core::float16) return 9;
    if (dt == mlx::core::float32) return 10;
    if (dt == mlx::core::bfloat16) return 11;
    if (dt == mlx::core::int32) return 7;
    if (dt == mlx::core::uint32) return 3;
    if (dt == mlx::core::bool_) return 0;
    return 10;
}

// Item extraction
float mlx_inline_item_f32(mlx_inline_array* a) { as_arr(a).eval(); return as_arr(a).item<float>(); }
uint32_t mlx_inline_item_u32(mlx_inline_array* a) { as_arr(a).eval(); return as_arr(a).item<uint32_t>(); }

void mlx_inline_sign(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::sign(as_arr(a)));
}

void mlx_inline_from_f32_slice(mlx_inline_array* dst, const float* data, const int* shape, int ndim) {
    mlx::core::Shape s(shape, shape + ndim);
    size_t n = 1;
    for (int i = 0; i < ndim; ++i) n *= shape[i];
    new (dst->buf) array(data, s, mlx::core::float32);
}

// Copy the evaluated f32 data of an array into a caller-provided buffer.
// The array is cast to float32 and eval'd first.  `n` must equal the total
// element count (product of all dimensions).  Returns 0 on success, -1 on
// dtype error (non-finite cast or wrong count).
int mlx_inline_to_f32_slice(mlx_inline_array* a, float* out, size_t n) {
    array& src = as_arr(a);
    // Cast to f32 if needed, then eval to materialise on CPU.
    array f32_arr = src.dtype() == mlx::core::float32
        ? src
        : mlx::core::astype(src, mlx::core::float32);
    f32_arr.eval();
    if ((size_t)f32_arr.size() != n) return -1;
    std::memcpy(out, f32_arr.data<float>(), n * sizeof(float));
    return 0;
}

void mlx_inline_stack(mlx_inline_array* dst, const mlx_inline_array* arrays, int num, int axis) {
    std::vector<array> arrs;
    arrs.reserve(num);
    for (int i = 0; i < num; ++i) {
        arrs.push_back(*reinterpret_cast<const array*>(arrays[i].buf));
    }
    new (dst->buf) array(mlx::core::stack(arrs, axis));
}

void mlx_inline_norm_l2(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    new (dst->buf) array(mlx::core::sqrt(mlx::core::sum(
        mlx::core::square(as_arr(a)), axis, keepdims)));
}

// Conv1d
void mlx_inline_conv1d(mlx_inline_array* dst, const mlx_inline_array* input,
                         const mlx_inline_array* weight, int stride, int padding,
                         int dilation, int groups) {
    new (dst->buf) array(mlx::core::conv1d(
        as_arr(input), as_arr(weight), stride, padding, dilation, groups));
}

// Print size for Rust to use
size_t mlx_inline_array_size(void) { return sizeof(array); }
size_t mlx_inline_array_align(void) { return alignof(array); }

void mlx_inline_enable_compile(void) { mlx::core::enable_compile(); }
void mlx_inline_disable_compile(void) { mlx::core::disable_compile(); }
void mlx_inline_clear_cache(void) { mlx::core::clear_cache(); }
size_t mlx_inline_set_cache_limit(size_t limit) { return mlx::core::set_cache_limit(limit); }

static mlx::core::Stream* generation_stream_ = nullptr;

size_t mlx_inline_set_wired_limit(size_t limit) {
    return mlx::core::set_wired_limit(limit);
}

size_t mlx_inline_get_max_recommended_size(void) {
    // Use system memory as a proxy — Metal's recommendedMaxWorkingSetSize
    // is typically 75% of total RAM on Apple Silicon.
    // For M4 Max with 128GB: 96GB. For M3 with 36GB: 27GB.
    size_t total_ram = 0;
    size_t len = sizeof(total_ram);
    if (sysctlbyname("hw.memsize", &total_ram, &len, nullptr, 0) == 0) {
        return (total_ram * 3) / 4; // 75% of total RAM
    }
    return (size_t)8 * 1024 * 1024 * 1024ULL; // 8 GiB fallback
}

int mlx_inline_new_stream(void) {
    // Heap-allocate the stream and intentionally leak it (never delete).
    // A static local Stream destructs at program exit in unpredictable order
    // relative to Metal device teardown, causing a SIGSEGV. A leaked heap
    // object outlives all destructors, letting the OS reclaim it cleanly.
    if (!generation_stream_) {
        generation_stream_ = new mlx::core::Stream(
            mlx::core::new_stream(mlx::core::default_device()));
    }
    return 0;
}

void mlx_inline_set_default_stream(int /*index*/) {
    if (generation_stream_) {
        mlx::core::set_default_stream(*generation_stream_);
    }
}

void mlx_inline_synchronize(void) {
    if (generation_stream_) {
        mlx::core::synchronize(*generation_stream_);
    } else {
        mlx::core::synchronize();
    }
}

int mlx_inline_metal_start_capture(const char* path) {
    mlx::core::metal::start_capture(path);
    return 0;
}
void mlx_inline_metal_stop_capture(void) {
    mlx::core::metal::stop_capture();
}

// Traverse computation graph and count unique nodes
static void count_nodes(const array& a, std::unordered_set<uintptr_t>& visited) {
    auto id = reinterpret_cast<uintptr_t>(&a);
    if (visited.count(id)) return;
    visited.insert(id);
    for (auto& inp : a.inputs()) {
        count_nodes(inp, visited);
    }
}
size_t mlx_inline_graph_node_count(const mlx_inline_array* a) {
    std::unordered_set<uintptr_t> visited;
    count_nodes(as_arr(a), visited);
    return visited.size();
}

// Debug: count graph nodes using ArrayDesc ID (shared_ptr target) instead of array address
static void count_descs(const array& a, std::unordered_set<uintptr_t>& visited) {
    // Use the ArrayDesc pointer as the unique ID (the actual graph node)
    auto id = a.id();
    if (visited.count(id)) return;
    visited.insert(id);
    for (auto& inp : a.inputs()) {
        count_descs(inp, visited);
    }
}
size_t mlx_inline_graph_desc_count(const mlx_inline_array* a) {
    std::unordered_set<uintptr_t> visited;
    count_descs(as_arr(a), visited);
    return visited.size();
}

// Helper: return a short dtype name for display.
static const char* dtype_name(mlx::core::Dtype dt) {
    if (dt == mlx::core::float32)  return "f32";
    if (dt == mlx::core::float16)  return "f16";
    if (dt == mlx::core::bfloat16) return "bf16";
    if (dt == mlx::core::int32)    return "i32";
    if (dt == mlx::core::uint32)   return "u32";
    return "?";
}

// Helper to demangle an MLX primitive type name.
static std::string demangle_prim(const mlx::core::Primitive& prim) {
    std::string name = typeid(prim).name();
    auto pos = name.rfind("E");
    if (pos != std::string::npos) {
        auto start = name.find_last_of("0123456789", pos - 1);
        if (start != std::string::npos) {
            return name.substr(start + 1, pos - start - 1);
        }
    }
    return name;
}

void mlx_inline_graph_dump(const mlx_inline_array* a) {
    using namespace mlx::core;
    std::unordered_set<uintptr_t> visited;
    std::unordered_map<std::string, int> prim_counts;
    // AsType: track (src_dtype→dst_dtype) signature frequencies for debugging.
    std::unordered_map<std::string, int> astype_sigs;
    // AsType parent: track the CHILD primitive (the one that has AsType as input).
    std::unordered_map<std::string, int> astype_parents;
    int total_dispatches = 0;
    size_t total_nodes = 0;

    // Stack stores (array*, parent_prim_name)
    std::vector<std::pair<const array*, std::string>> stack = {{&as_arr(a), "root"}};
    while (!stack.empty()) {
        auto [arr, parent_name] = stack.back(); stack.pop_back();
        auto id = arr->id();
        if (visited.count(id)) continue;
        visited.insert(id);
        total_nodes++;

        bool is_available = arr->is_available();
        std::string prim_name;

        if (arr->has_primitive()) {
            prim_name = demangle_prim(arr->primitive());
            prim_counts[prim_name]++;
            if (!is_available) total_dispatches++;

            // For AsType: record src→dst dtype breakdown AND the parent.
            if (prim_name == "AsType" && !arr->inputs().empty()) {
                const auto& src = arr->inputs()[0];
                char sig[64];
                snprintf(sig, sizeof(sig), "    AsType %s→%s",
                    dtype_name(src.dtype()), dtype_name(arr->dtype()));
                astype_sigs[std::string(sig)]++;
                // Record child_prim → count (who has this AsType as input)
                char parent_sig[128];
                snprintf(parent_sig, sizeof(parent_sig), "  parent=%-20s AsType %s→%s",
                    parent_name.c_str(), dtype_name(src.dtype()), dtype_name(arr->dtype()));
                astype_parents[std::string(parent_sig)]++;
            }
        } else {
            prim_name = is_available ? "(evaluated)" : "(detached)";
            prim_counts[prim_name]++;
        }

        // Only recurse into inputs for UNEVALUATED nodes.
        // Evaluated nodes are historical computation that's already been executed;
        // recursing into them would traverse the entire prefill history.
        if (!is_available) {
            for (auto& inp : arr->inputs()) {
                stack.push_back({&inp, prim_name});
            }
        }
    }

    fprintf(stderr, "=== Graph Dump: %zu unique nodes, %d unevaluated dispatches ===\n",
        total_nodes, total_dispatches);
    // Print primitive type summary (sorted by count descending).
    std::vector<std::pair<std::string, int>> sorted_prims(prim_counts.begin(), prim_counts.end());
    std::sort(sorted_prims.begin(), sorted_prims.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& [name, count] : sorted_prims) {
        fprintf(stderr, "  %4d  %s\n", count, name.c_str());
    }
    // Print AsType src→dst breakdown if any AsType nodes exist.
    if (!astype_sigs.empty()) {
        fprintf(stderr, "  --- AsType dtype breakdown ---\n");
        std::vector<std::pair<std::string, int>> sorted_at(astype_sigs.begin(), astype_sigs.end());
        std::sort(sorted_at.begin(), sorted_at.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        for (const auto& [sig, count] : sorted_at) {
            fprintf(stderr, "  %4d  %s\n", count, sig.c_str());
        }
        // Print parent breakdown (limit to first 20 unique)
        if (astype_parents.size() <= 20) {
            fprintf(stderr, "  --- AsType parent breakdown (child prim that has this AsType as input) ---\n");
            std::vector<std::pair<std::string, int>> sorted_p(astype_parents.begin(), astype_parents.end());
            std::sort(sorted_p.begin(), sorted_p.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            for (const auto& [sig, count] : sorted_p) {
                fprintf(stderr, "  %4d  %s\n", count, sig.c_str());
            }
        }
    }
}

// ── Additional ops for complete model inference ──

// Helper: map integer dtype code (same table as mlx_inline_astype) to MLX Dtype.
static inline mlx::core::Dtype dtype_from_int(int dtype) {
    static const mlx::core::Dtype dtypes[] = {
        mlx::core::bool_,    // 0
        mlx::core::uint8,    // 1
        mlx::core::uint16,   // 2
        mlx::core::uint32,   // 3
        mlx::core::uint64,   // 4
        mlx::core::int8,     // 5
        mlx::core::int16,    // 6
        mlx::core::int32,    // 7
        mlx::core::int64,    // 8
        mlx::core::float16,  // 9
        mlx::core::float32,  // 10
        mlx::core::bfloat16, // 11
        mlx::core::complex64 // 12
    };
    return (dtype >= 0 && dtype <= 12) ? dtypes[dtype] : mlx::core::float32;
}

void mlx_inline_concatenate_2(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b, int axis) {
    new (dst->buf) array(mlx::core::concatenate({as_arr(a), as_arr(b)}, axis));
}

void mlx_inline_softplus(mlx_inline_array* dst, const mlx_inline_array* a) {
    // softplus(x) = log(1 + exp(x)) = log1p(exp(x))
    auto& x = as_arr(a);
    new (dst->buf) array(mlx::core::log1p(mlx::core::exp(x)));
}

void mlx_inline_where(mlx_inline_array* dst, const mlx_inline_array* condition,
                       const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::where(as_arr(condition), as_arr(a), as_arr(b)));
}

void mlx_inline_maximum(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::maximum(as_arr(a), as_arr(b)));
}

void mlx_inline_zeros(mlx_inline_array* dst, const int* shape, int ndim, int dtype) {
    new (dst->buf) array(mlx::core::zeros(
        mlx::core::Shape(shape, shape + ndim), dtype_from_int(dtype)));
}

void mlx_inline_ones(mlx_inline_array* dst, const int* shape, int ndim, int dtype) {
    new (dst->buf) array(mlx::core::ones(
        mlx::core::Shape(shape, shape + ndim), dtype_from_int(dtype)));
}

void mlx_inline_slice(mlx_inline_array* dst, const mlx_inline_array* a,
                       const int* start, const int* stop, int ndim) {
    new (dst->buf) array(mlx::core::slice(
        as_arr(a),
        mlx::core::Shape(start, start + ndim),
        mlx::core::Shape(stop, stop + ndim)));
}

void mlx_inline_slice_set(mlx_inline_array* dst, const mlx_inline_array* a,
                            const mlx_inline_array* value,
                            const int* start, const int* stop, int ndim) {
    new (dst->buf) array(mlx::core::slice_update(
        as_arr(a), as_arr(value),
        mlx::core::Shape(start, start + ndim),
        mlx::core::Shape(stop, stop + ndim)));
}

void mlx_inline_repeat(mlx_inline_array* dst, const mlx_inline_array* a, int repeats, int axis) {
    new (dst->buf) array(mlx::core::repeat(as_arr(a), repeats, axis));
}

void mlx_inline_squeeze(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::squeeze(as_arr(a), axis));
}

void mlx_inline_expand_dims(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::expand_dims(as_arr(a), axis));
}

void mlx_inline_transpose_axes(mlx_inline_array* dst, const mlx_inline_array* a,
                                 const int* axes, int ndim) {
    new (dst->buf) array(mlx::core::transpose(
        as_arr(a), std::vector<int>(axes, axes + ndim)));
}

void mlx_inline_cumsum(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::cumsum(as_arr(a), axis));
}

void mlx_inline_log(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::log(as_arr(a)));
}

void mlx_inline_tril(mlx_inline_array* dst, const mlx_inline_array* a, int k) {
    new (dst->buf) array(mlx::core::tril(as_arr(a), k));
}

void mlx_inline_index(mlx_inline_array* dst, const mlx_inline_array* a,
                       const mlx_inline_array* indices) {
    // take(a, indices) — flat gather over all elements (no axis specified)
    new (dst->buf) array(mlx::core::take(as_arr(a), as_arr(indices)));
}

void mlx_inline_softmax_precise(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::softmax(as_arr(a), axis, /*precise=*/true));
}

void mlx_inline_sdpa_with_mask(mlx_inline_array* dst,
                                 const mlx_inline_array* q, const mlx_inline_array* k,
                                 const mlx_inline_array* v, float scale,
                                 const mlx_inline_array* mask) {
    auto mask_opt = mask
        ? std::optional<array>(as_arr(mask))
        : std::optional<array>(std::nullopt);
    new (dst->buf) array(mlx::core::fast::scaled_dot_product_attention(
        as_arr(q), as_arr(k), as_arr(v), scale, /*mask_mode=*/"", mask_opt));
}

void mlx_inline_eval_2(mlx_inline_array* a, mlx_inline_array* b) {
    mlx::core::eval({as_arr(a), as_arr(b)});
}

void mlx_inline_eval_many(mlx_inline_array** arrays, int count) {
    std::vector<array> arrs;
    arrs.reserve(count);
    for (int i = 0; i < count; ++i) {
        arrs.push_back(as_arr(arrays[i]));
    }
    mlx::core::eval(std::move(arrs));
}

void mlx_inline_async_eval_many(mlx_inline_array** arrays, int count) {
    std::vector<array> arrs;
    arrs.reserve(count);
    for (int i = 0; i < count; ++i) {
        arrs.push_back(as_arr(arrays[i]));
    }
    mlx::core::async_eval(std::move(arrs));
}

void mlx_inline_quantized_matmul(mlx_inline_array* dst,
                                   const mlx_inline_array* x, const mlx_inline_array* w,
                                   const mlx_inline_array* scales, const mlx_inline_array* biases,
                                   bool transpose, int group_size, int bits) {
    auto biases_opt = biases
        ? std::optional<array>(as_arr(biases))
        : std::optional<array>(std::nullopt);
    new (dst->buf) array(mlx::core::quantized_matmul(
        as_arr(x), as_arr(w), as_arr(scales), biases_opt,
        transpose, group_size, bits));
}

void mlx_inline_gather_qmm(mlx_inline_array* dst,
                              const mlx_inline_array* x, const mlx_inline_array* w,
                              const mlx_inline_array* scales, const mlx_inline_array* biases,
                              const mlx_inline_array* lhs_indices, const mlx_inline_array* rhs_indices,
                              bool transpose, int group_size, int bits, bool sorted) {
    auto biases_opt = biases
        ? std::optional<array>(as_arr(biases))
        : std::optional<array>(std::nullopt);
    auto lhs_opt = lhs_indices
        ? std::optional<array>(as_arr(lhs_indices))
        : std::optional<array>(std::nullopt);
    auto rhs_opt = rhs_indices
        ? std::optional<array>(as_arr(rhs_indices))
        : std::optional<array>(std::nullopt);
    new (dst->buf) array(mlx::core::gather_qmm(
        as_arr(x), as_arr(w), as_arr(scales), biases_opt,
        lhs_opt, rhs_opt,
        transpose, group_size, bits,
        /*mode=*/"affine", sorted));
}

// GDN Metal kernel source — fuses the entire recurrence into one Metal dispatch.
// Matches the Rust GDN_KERNEL_SOURCE from pmetal-mlx/src/kernels/gated_delta.rs.
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

// Cached Metal kernel function (created once, reused)
static mlx::core::fast::CustomKernelFunction& get_gdn_kernel() {
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
    return 0;
}

void mlx_inline_take_axis(mlx_inline_array* dst, const mlx_inline_array* a,
    const mlx_inline_array* indices, int axis) {
    new (dst->buf) array(mlx::core::take(as_arr(a), as_arr(indices), axis));
}

void mlx_inline_kv_cache_append(mlx_inline_array* dst,
    const mlx_inline_array* cached, const mlx_inline_array* new_kv, int axis) {
    new (dst->buf) array(mlx::core::concatenate({as_arr(cached), as_arr(new_kv)}, axis));
}

void mlx_inline_async_eval_arr(const mlx_inline_array* a) {
    mlx::core::async_eval({as_arr(a)});
}

// ── Sampling ops ──

void mlx_inline_argmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::argmax(as_arr(a), axis));
}

void mlx_inline_argmin(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::argmin(as_arr(a), axis));
}

void mlx_inline_abs(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::abs(as_arr(a)));
}

void mlx_inline_logsumexp(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    new (dst->buf) array(mlx::core::logsumexp(as_arr(a), axis, keepdims));
}

void mlx_inline_categorical(mlx_inline_array* dst, const mlx_inline_array* logits) {
    new (dst->buf) array(mlx::core::random::categorical(as_arr(logits)));
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
}

void mlx_inline_arange(mlx_inline_array* dst, int n, int dtype) {
    new (dst->buf) array(mlx::core::arange(0, n, dtype_from_int(dtype)));
}

int mlx_inline_load_safetensors_key(mlx_inline_array* dst, const char* path, const char* key) {
    try {
        auto [arrays, metadata] = mlx::core::load_safetensors(std::string(path));
        auto it = arrays.find(std::string(key));
        if (it == arrays.end()) return 1;
        new (dst->buf) array(it->second);
        return 0;
    } catch (...) {
        return 1;
    }
}

// Load ALL tensors from a safetensors file in a single parse.
// Each entry gets a strdup'd key and a placement-new'd array in the caller buffers.
// Returns the number of entries written, or -1 on error.
int mlx_inline_load_safetensors_all(
        const char* path,
        char** key_buf,
        mlx_inline_array* arr_buf,
        int max_entries) {
    try {
        auto [arrays, metadata] = mlx::core::load_safetensors(std::string(path));
        int count = 0;
        for (auto& [key, arr] : arrays) {
            if (count >= max_entries) break;
            key_buf[count] = strdup(key.c_str());
            new (arr_buf[count].buf) array(arr);
            count++;
        }
        return count;
    } catch (...) {
        return -1;
    }
}

void mlx_inline_free_key_strings(char** keys, int count) {
    for (int i = 0; i < count; ++i) {
        free(keys[i]);
    }
}

void mlx_inline_from_i32_slice(mlx_inline_array* dst, const int32_t* data, int len) {
    new (dst->buf) array(data, {len}, mlx::core::int32);
}

void mlx_inline_detach(mlx_inline_array* a) {
    as_arr(a).detach();
}

// ── Metal memory instrumentation ──

size_t mlx_inline_get_active_memory(void) {
    return mlx::core::get_active_memory();
}

size_t mlx_inline_get_cache_memory(void) {
    return mlx::core::get_cache_memory();
}

size_t mlx_inline_get_peak_memory(void) {
    return mlx::core::get_peak_memory();
}

void mlx_inline_reset_peak_memory(void) {
    mlx::core::reset_peak_memory();
}

} // extern "C"

// ============================================================================
// Fused compiled ops — matching Python's @mx.compile(shapeless=True)
// Each creates a compiled closure on first call, caches it, and replays.
// This produces a single Compiled graph node instead of N separate nodes.
// Must be outside extern "C" for C++ template/lambda support.
// ============================================================================

using namespace mlx::core;
using CompiledFn = std::function<std::vector<array>(const std::vector<array>&)>;

static CompiledFn make_compiled(CompiledFn fn) {
    return mlx::core::compile(std::move(fn), /*shapeless=*/true);
}

// shapeless=false: works with ALL primitives (Split, CustomKernel, etc.)
// but only replays the tape when input shapes match the first trace.
// Perfect for T=1 decode where shapes are always the same.
static CompiledFn make_compiled_fixed(CompiledFn fn) {
    return mlx::core::compile(std::move(fn), /*shapeless=*/false);
}

extern "C" {

void mlx_inline_fused_swiglu(mlx_inline_array* dst,
    const mlx_inline_array* gate, const mlx_inline_array* up) {
    static auto compiled = make_compiled(
        [](const std::vector<array>& inputs) -> std::vector<array> {
            auto& g = inputs[0];
            auto& u = inputs[1];
            return {multiply(multiply(g, sigmoid(g)), u)};
        });
    auto result = compiled({as_arr(gate), as_arr(up)});
    new (dst->buf) array(result[0]);
}

void mlx_inline_fused_silu(mlx_inline_array* dst, const mlx_inline_array* x) {
    static auto compiled = make_compiled(
        [](const std::vector<array>& inputs) -> std::vector<array> {
            auto& x = inputs[0];
            return {multiply(x, sigmoid(x))};
        });
    auto result = compiled({as_arr(x)});
    new (dst->buf) array(result[0]);
}

void mlx_inline_fused_compute_g(mlx_inline_array* dst,
    const mlx_inline_array* a_log, const mlx_inline_array* a, const mlx_inline_array* dt_bias) {
    static auto compiled = make_compiled(
        [](const std::vector<array>& inputs) -> std::vector<array> {
            auto decay = exp(astype(inputs[0], float32));
            auto sp = log1p(exp(add(inputs[1], inputs[2])));
            return {exp(negative(multiply(decay, sp)))};
        });
    auto result = compiled({as_arr(a_log), as_arr(a), as_arr(dt_bias)});
    new (dst->buf) array(result[0]);
}

void mlx_inline_fused_precise_swiglu(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* gate) {
    static auto compiled = make_compiled(
        [](const std::vector<array>& inputs) -> std::vector<array> {
            auto& x = inputs[0];
            auto& g = inputs[1];
            auto g32 = multiply(astype(g, float32), sigmoid(astype(g, float32)));
            auto x32 = astype(x, float32);
            return {astype(multiply(g32, x32), x.dtype())};
        });
    auto result = compiled({as_arr(x), as_arr(gate)});
    new (dst->buf) array(result[0]);
}

// Compiled entire GDN layer forward.
// Uses 4 separate projection weights matching Python's in_proj_qkv/z/b/a.
// Scalar params are captured in the closure (all GDN layers share the same dims).
void mlx_inline_compiled_gdn_layer(
    mlx_inline_array* dst_out,
    mlx_inline_array* dst_conv_state,
    mlx_inline_array* dst_ssm_state,
    const mlx_inline_array* normed,
    const mlx_inline_array* qkv_w,
    const mlx_inline_array* z_w,
    const mlx_inline_array* b_w,
    const mlx_inline_array* a_w,
    const mlx_inline_array* conv_w,
    const mlx_inline_array* q_nw,
    const mlx_inline_array* k_nw,
    const mlx_inline_array* a_log,
    const mlx_inline_array* dt_bias,
    const mlx_inline_array* norm_w,
    const mlx_inline_array* out_w,
    const mlx_inline_array* conv_state_in,
    const mlx_inline_array* ssm_state_in,
    int nv, int nk, int dk, int dv, int cd, int ck, int kd, float norm_eps
) {
    // Lazy-init compiled function with captured scalar params.
    // All GDN layers have the same dimensions, so this is compiled once.
    static CompiledFn compiled;
    static bool initialized = false;
    if (!initialized) {
        int NV=nv, NK=nk, DK=dk, DV=dv, CD=cd, CK=ck, KD=kd;
        float EPS=norm_eps;
        compiled = make_compiled(
            [NV, NK, DK, DV, CD, CK, KD, EPS](const std::vector<array>& ins) -> std::vector<array> {
                auto& normed      = ins[0];
                auto& qkv_w       = ins[1];
                auto& z_w         = ins[2];
                auto& b_w         = ins[3];
                auto& a_w         = ins[4];
                auto& conv_w      = ins[5];
                auto& q_nw        = ins[6];
                auto& k_nw        = ins[7];
                auto& a_log_arr   = ins[8];
                auto& dt_bias_arr = ins[9];
                auto& norm_w_arr  = ins[10];
                auto& out_w       = ins[11];
                auto& conv_state  = ins[12];
                auto& ssm_state   = ins[13];
                int B = normed.shape(0); int S = normed.shape(1);

                // 4 separate matmuls — no splitting needed, matches Python exactly
                auto qkv   = matmul(normed, qkv_w);
                auto z     = reshape(matmul(normed, z_w), {B, S, NV, DV});
                auto b_val = matmul(normed, b_w);
                auto a_val = matmul(normed, a_w);

                auto conv_in = concatenate({conv_state, qkv}, 1);
                auto new_conv = slice(conv_in, {0, 1, 0}, {B, CK, CD});
                auto conv_out = mlx::core::conv1d(conv_in, conv_w, 1, 0, 1, CD);
                auto conv_act = multiply(conv_out, sigmoid(conv_out));

                // shapeless=true does NOT support Split — keep slices here.
                // (compiled_gdn_layer is for variable-length; fixed-shape decode uses
                //  compiled_gdn_layer_fixed which uses shapeless=false and supports split.)
                auto q = fast::rms_norm(reshape(slice(conv_act, {0, 0, 0}, {B, S, KD}), {B, S, NK, DK}), q_nw, EPS);
                auto k = fast::rms_norm(reshape(slice(conv_act, {0, 0, KD}, {B, S, KD * 2}), {B, S, NK, DK}), k_nw, EPS);
                auto v = reshape(slice(conv_act, {0, 0, KD * 2}, {B, S, CD}), {B, S, NV, DV});

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
                            {B, S, NV * DV}),
                    out_w);
                return {output, new_conv, kout[1]};
            });
        initialized = true;
    }

    auto result = compiled({
        as_arr(normed),
        as_arr(qkv_w), as_arr(z_w), as_arr(b_w), as_arr(a_w),
        as_arr(conv_w),
        as_arr(q_nw), as_arr(k_nw), as_arr(a_log), as_arr(dt_bias),
        as_arr(norm_w), as_arr(out_w), as_arr(conv_state_in), as_arr(ssm_state_in)
    });
    new (dst_out->buf) array(result[0]);
    new (dst_conv_state->buf) array(result[1]);
    new (dst_ssm_state->buf) array(result[2]);
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
    static CompiledFn compiled;
    static bool initialized = false;
    if (!initialized) {
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
        initialized = true;
    }

    auto result = compiled({
        as_arr(normed),
        as_arr(qkv_w), as_arr(z_w), as_arr(b_w), as_arr(a_w), as_arr(conv_w),
        as_arr(q_nw), as_arr(k_nw), as_arr(a_log), as_arr(dt_bias),
        as_arr(norm_w), as_arr(out_w), as_arr(conv_state_in), as_arr(ssm_state_in)
    });
    new (dst_out->buf) array(result[0]);
    new (dst_conv_state->buf) array(result[1]);
    new (dst_ssm_state->buf) array(result[2]);
}

} // extern "C"

// ============================================================================
// Full Qwen3.5 forward pass — single C++ function, zero per-op FFI overhead.
//
// The entire forward pass (embedding, all N layers, final norm, lm_head) runs
// here as pure C++ MLX ops.  No Rust stack frame is entered between ops;
// intermediate arrays are C++ locals, never placement-new'd through the FFI
// bridge.  This eliminates ~1800 FFI round trips per decode step.
// ============================================================================

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

        // Post-attention LayerNorm + SwiGLU MLP (fused inline, no FFI)
        auto mlp_in  = fast::rms_norm(h, W(weight_ptrs, base + 1), post_eps);
        auto gate_v  = matmul(mlp_in, W(weight_ptrs, base + 2));  // gate_w
        auto up_v    = matmul(mlp_in, W(weight_ptrs, base + 3));  // up_w
        // silu(gate) * up — inlined (matches fused_swiglu exactly)
        auto swiglu  = multiply(multiply(gate_v, sigmoid(gate_v)), up_v);
        auto mlp_out = matmul(swiglu, W(weight_ptrs, base + 4));  // down_w

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

  } catch (const std::exception& e) {
    fprintf(stderr, "[C++ EXCEPTION] qwen35_decode_step: %s\n", e.what());
    new (dst_logits->buf) array(0.0f);  // safe fallback
  }
}

} // extern "C" (qwen35 full forward)
