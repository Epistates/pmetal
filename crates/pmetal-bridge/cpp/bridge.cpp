// Inline array bridge — stores mlx::core::array on the Rust stack.
// Zero heap allocation per op. Direct C++ calls.

#include "bridge.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"  // for typeid on Primitive subclasses
#include <typeinfo>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <unordered_set>
#include <numeric>
#include <sys/sysctl.h>

using mlx::core::array;

static inline mlx::core::Dtype dtype_from_int(int dtype);

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
void mlx_inline_eval(mlx_inline_array* a) {
    try { as_arr(a).eval(); }
    catch (const std::exception& e) { fprintf(stderr, "[C++ EVAL EXCEPTION] %s\n", e.what()); }
    catch (...) { fprintf(stderr, "[C++ EVAL EXCEPTION] unknown exception\n"); }
}
void mlx_inline_async_eval(mlx_inline_array* a) {
    try { mlx::core::async_eval(as_arr(a)); }
    catch (const std::exception& e) { fprintf(stderr, "[C++ ASYNC_EVAL EXCEPTION] %s\n", e.what()); }
}

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
float mlx_inline_item_f32(mlx_inline_array* a) {
    try { as_arr(a).eval(); return as_arr(a).item<float>(); }
    catch (const std::exception& e) { fprintf(stderr, "[C++ ITEM_F32 EXCEPTION] %s\n", e.what()); return 0.0f; }
}
uint32_t mlx_inline_item_u32(mlx_inline_array* a) {
    try { as_arr(a).eval(); return as_arr(a).item<uint32_t>(); }
    catch (const std::exception& e) { fprintf(stderr, "[C++ ITEM_U32 EXCEPTION] %s\n", e.what()); return 0; }
}

void mlx_inline_sign(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::sign(as_arr(a)));
}

void mlx_inline_dequantize(mlx_inline_array* dst, const mlx_inline_array* w,
    const mlx_inline_array* scales, const mlx_inline_array* biases,
    int group_size, int bits) {
    try {
        new (dst->buf) array(mlx::core::dequantize(
            as_arr(w), as_arr(scales), as_arr(biases), group_size, bits));
    } catch (const std::exception& e) {
        fprintf(stderr, "[C++ EXCEPTION] dequantize: %s\n", e.what());
        new (dst->buf) array(0.0f);
    }
}

void mlx_inline_from_f32_slice(mlx_inline_array* dst, const float* data, const int* shape, int ndim) {
    mlx::core::Shape s(shape, shape + ndim);
    new (dst->buf) array(data, s, mlx::core::float32);
}

void mlx_inline_from_u32_slice(
    mlx_inline_array* dst,
    const uint32_t* data,
    const int* shape,
    int ndim) {
    mlx::core::Shape s(shape, shape + ndim);
    new (dst->buf) array(data, s, mlx::core::uint32);
}

void mlx_inline_from_u8_slice(
    mlx_inline_array* dst,
    const uint8_t* data,
    const int* shape,
    int ndim) {
    mlx::core::Shape s(shape, shape + ndim);
    new (dst->buf) array(data, s, mlx::core::uint8);
}

void mlx_inline_from_u16_bits_slice(
    mlx_inline_array* dst,
    const uint16_t* data,
    const int* shape,
    int ndim,
    int dtype) {
    mlx::core::Shape s(shape, shape + ndim);
    auto dt = dtype_from_int(dtype);
    switch (dt) {
      case mlx::core::bfloat16:
        new (dst->buf) array(reinterpret_cast<const mlx::core::bfloat16_t*>(data), s, dt);
        break;
      case mlx::core::float16:
        new (dst->buf) array(reinterpret_cast<const mlx::core::float16_t*>(data), s, dt);
        break;
      case mlx::core::uint16:
        new (dst->buf) array(data, s, dt);
        break;
      default:
        throw std::invalid_argument("mlx_inline_from_u16_bits_slice requires float16, bfloat16, or uint16 dtype");
    }
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
    try {
        auto biases_opt = biases
            ? std::optional<array>(as_arr(biases))
            : std::optional<array>(std::nullopt);
        new (dst->buf) array(mlx::core::quantized_matmul(
            as_arr(x), as_arr(w), as_arr(scales), biases_opt,
            transpose, group_size, bits));
    } catch (const std::exception& e) {
        fprintf(stderr, "[C++ EXCEPTION] quantized_matmul: %s\n", e.what());
        new (dst->buf) array(0.0f);
    }
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

// Cached Metal kernel functions (created once per process)
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (...) {
        return 1;
    }
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
    } catch (const std::exception& e) {
        fprintf(stderr, "[C++ EXCEPTION] load_safetensors_key(%s, %s): %s\n", path, key, e.what());
        return 1;
    } catch (...) {
        fprintf(stderr, "[C++ EXCEPTION] load_safetensors_key(%s, %s): unknown exception\n", path, key);
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
    } catch (const std::exception& e) {
        fprintf(stderr, "[C++ EXCEPTION] load_safetensors_all(%s): %s\n", path, e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "[C++ EXCEPTION] load_safetensors_all(%s): unknown exception\n", path);
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
            make_compiled_fixed(
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
            make_compiled_fixed(
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

  } catch (const std::exception& e) {
    fprintf(stderr, "[C++ EXCEPTION] qwen35_decode_step: %s\n", e.what());
    new (dst_logits->buf) array(0.0f);  // safe fallback
  }
}

// ── Training ops: random ─────────────────────────────────────────────────────

void mlx_inline_random_normal(mlx_inline_array* dst, const int* shape, int ndim, int dtype) {
    new (dst->buf) array(mlx::core::random::normal(
        {shape, shape + ndim}, dtype_from_int(dtype)));
}

void mlx_inline_random_uniform(mlx_inline_array* dst, const int* shape, int ndim, int dtype) {
    new (dst->buf) array(mlx::core::random::uniform(
        {shape, shape + ndim}, dtype_from_int(dtype)));
}

void mlx_inline_random_bernoulli(mlx_inline_array* dst, const mlx_inline_array* p, const int* shape, int ndim) {
    new (dst->buf) array(mlx::core::random::bernoulli(
        as_arr(p), {shape, shape + ndim}));
}

void mlx_inline_random_seed(uint64_t seed) {
    mlx::core::random::seed(seed);
}

void mlx_inline_random_randint(mlx_inline_array* dst, int low, int high, const int* shape, int ndim, int dtype) {
    new (dst->buf) array(mlx::core::random::randint(
        low, high, {shape, shape + ndim}, dtype_from_int(dtype)));
}

// ── Training ops: math ───────────────────────────────────────────────────────

void mlx_inline_mean_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    new (dst->buf) array(mlx::core::mean(as_arr(a), axis, keepdims));
}

void mlx_inline_mean_all(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::mean(as_arr(a)));
}

void mlx_inline_var_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    new (dst->buf) array(mlx::core::var(as_arr(a), axis, keepdims));
}

void mlx_inline_pow(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::power(as_arr(a), as_arr(b)));
}

void mlx_inline_reciprocal(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::reciprocal(as_arr(a)));
}

void mlx_inline_sin(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::sin(as_arr(a)));
}

void mlx_inline_cos(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::cos(as_arr(a)));
}

void mlx_inline_clip(mlx_inline_array* dst, const mlx_inline_array* a,
                     const mlx_inline_array* lo, const mlx_inline_array* hi) {
    auto low_opt  = lo ? std::optional<array>(as_arr(lo)) : std::nullopt;
    auto high_opt = hi ? std::optional<array>(as_arr(hi)) : std::nullopt;
    new (dst->buf) array(mlx::core::clip(as_arr(a), low_opt, high_opt));
}

void mlx_inline_log_softmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    // log_softmax(x) = x - logsumexp(x, axis, keepdims=true)
    const auto& x = as_arr(a);
    auto lse = mlx::core::logsumexp(x, axis, true);
    new (dst->buf) array(mlx::core::subtract(x, lse));
}

void mlx_inline_cross_entropy(mlx_inline_array* dst, const mlx_inline_array* logits,
                               const mlx_inline_array* targets, int axis) {
    // cross_entropy = -sum(targets * log_softmax(logits), axis=axis)
    const auto& l = as_arr(logits);
    auto lse      = mlx::core::logsumexp(l, axis, true);
    auto log_probs = mlx::core::subtract(l, lse);
    new (dst->buf) array(mlx::core::negative(
        mlx::core::sum(mlx::core::multiply(as_arr(targets), log_probs), axis, false)));
}

void mlx_inline_square(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::square(as_arr(a)));
}

// ── Training ops: creation ───────────────────────────────────────────────────

void mlx_inline_full(mlx_inline_array* dst, const int* shape, int ndim, float val, int dtype) {
    new (dst->buf) array(mlx::core::full(
        {shape, shape + ndim}, val, dtype_from_int(dtype)));
}

void mlx_inline_eye(mlx_inline_array* dst, int n, int dtype) {
    // eye(n, m=n, k=0, dtype)
    new (dst->buf) array(mlx::core::eye(n, n, 0, dtype_from_int(dtype)));
}

void mlx_inline_tri(mlx_inline_array* dst, int n, int m, int k, int dtype) {
    new (dst->buf) array(mlx::core::tri(n, m, k, dtype_from_int(dtype)));
}

// ── Training ops: shape ──────────────────────────────────────────────────────

void mlx_inline_broadcast_to(mlx_inline_array* dst, const mlx_inline_array* a,
                              const int* shape, int ndim) {
    new (dst->buf) array(mlx::core::broadcast_to(
        as_arr(a), {shape, shape + ndim}));
}

void mlx_inline_flatten(mlx_inline_array* dst, const mlx_inline_array* a,
                        int start_axis, int end_axis) {
    new (dst->buf) array(mlx::core::flatten(as_arr(a), start_axis, end_axis));
}

// ── Training ops: sort/reduction ─────────────────────────────────────────────

void mlx_inline_argsort(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    new (dst->buf) array(mlx::core::argsort(as_arr(a), axis));
}

void mlx_inline_sum_all(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::sum(as_arr(a)));
}

void mlx_inline_max_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    new (dst->buf) array(mlx::core::max(as_arr(a), axis, keepdims));
}

void mlx_inline_min_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    new (dst->buf) array(mlx::core::min(as_arr(a), axis, keepdims));
}

void mlx_inline_minimum(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::minimum(as_arr(a), as_arr(b)));
}

// ── Training ops: activation ─────────────────────────────────────────────────

void mlx_inline_relu(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::maximum(as_arr(a), array(0.0f)));
}

void mlx_inline_gelu(mlx_inline_array* dst, const mlx_inline_array* a) {
    // GELU fast approx: x * sigmoid(1.702 * x)
    const auto& x = as_arr(a);
    new (dst->buf) array(mlx::core::multiply(
        x, mlx::core::sigmoid(mlx::core::multiply(array(1.702f), x))));
}

// ── Training ops: comparison ─────────────────────────────────────────────────

void mlx_inline_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::equal(as_arr(a), as_arr(b)));
}

void mlx_inline_not_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::not_equal(as_arr(a), as_arr(b)));
}

void mlx_inline_greater(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::greater(as_arr(a), as_arr(b)));
}

void mlx_inline_less(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::less(as_arr(a), as_arr(b)));
}

void mlx_inline_greater_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::greater_equal(as_arr(a), as_arr(b)));
}

void mlx_inline_less_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    new (dst->buf) array(mlx::core::less_equal(as_arr(a), as_arr(b)));
}

// ── Training ops: serialization ──────────────────────────────────────────────

void mlx_inline_save_safetensors(const char* path, const char** keys,
                                  const mlx_inline_array* arrays, int count) {
    std::unordered_map<std::string, array> map;
    map.reserve(count);
    for (int i = 0; i < count; i++) {
        map.emplace(std::string(keys[i]), as_arr(&arrays[i]));
    }
    mlx::core::save_safetensors(std::string(path), std::move(map));
}

// ── Training ops: quantize ───────────────────────────────────────────────────

void mlx_inline_quantize(mlx_inline_array* dst_w, mlx_inline_array* dst_scales,
                          mlx_inline_array* dst_biases,
                          const mlx_inline_array* a, int group_size, int bits) {
    auto result = mlx::core::quantize(as_arr(a), group_size, bits);
    new (dst_w->buf)      array(std::move(result[0]));
    new (dst_scales->buf) array(std::move(result[1]));
    new (dst_biases->buf) array(std::move(result[2]));
}

// ── Training ops: multi-axis sum/mean ────────────────────────────────────────

void mlx_inline_sum_axes(mlx_inline_array* dst, const mlx_inline_array* a,
                          const int* axes, int num_axes, bool keepdims) {
    new (dst->buf) array(mlx::core::sum(
        as_arr(a), {axes, axes + num_axes}, keepdims));
}

void mlx_inline_mean_axes(mlx_inline_array* dst, const mlx_inline_array* a,
                           const int* axes, int num_axes, bool keepdims) {
    new (dst->buf) array(mlx::core::mean(
        as_arr(a), {axes, axes + num_axes}, keepdims));
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

void mlx_inline_stop_gradient(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::stop_gradient(as_arr(a)));
}

void mlx_inline_tri_inv(mlx_inline_array* dst, const mlx_inline_array* a, bool upper, bool use_cpu) {
    // tri_inv has no VJP in MLX — used in WY factorization as a fixed preconditioner.
    // use_cpu=true routes execution to the CPU device (matching mlx-lm's StreamOrDevice::cpu()).
    mlx::core::StreamOrDevice stream = use_cpu
        ? mlx::core::StreamOrDevice{mlx::core::Device(mlx::core::Device::cpu)}
        : mlx::core::StreamOrDevice{};
    new (dst->buf) array(mlx::core::linalg::tri_inv(as_arr(a), upper, stream));
}

void mlx_inline_svd(
    mlx_inline_array* dst_u,
    mlx_inline_array* dst_s,
    mlx_inline_array* dst_vt,
    const mlx_inline_array* a)
{
    // SVD always runs on CPU (GPU SVD not available in MLX).
    // Returns economy / thin SVD: U[m,k], S[k], Vt[k,n] where k=min(m,n).
    mlx::core::StreamOrDevice cpu_stream{mlx::core::Device(mlx::core::Device::cpu)};
    auto result = mlx::core::linalg::svd(as_arr(a), /* compute_uv */ true, cpu_stream);
    // result = {U, S, Vt}
    new (dst_u->buf)  array(result[0]);
    new (dst_s->buf)  array(result[1]);
    new (dst_vt->buf) array(result[2]);
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
}

// ── FFT ops ──────────────────────────────────────────────────────────────────

void mlx_inline_rfft(mlx_inline_array* dst, const mlx_inline_array* a, int n_fft, int axis) {
    const auto& x = as_arr(a);
    // n_fft < 0 means "use full axis size" — use the no-n overload
    if (n_fft < 0) {
        new (dst->buf) array(mlx::core::fft::rfft(x, axis));
    } else {
        new (dst->buf) array(mlx::core::fft::rfft(x, n_fft, axis));
    }
}

void mlx_inline_irfft(mlx_inline_array* dst, const mlx_inline_array* a, int n_fft, int axis) {
    const auto& x = as_arr(a);
    if (n_fft < 0) {
        new (dst->buf) array(mlx::core::fft::irfft(x, axis));
    } else {
        new (dst->buf) array(mlx::core::fft::irfft(x, n_fft, axis));
    }
}

// ── leaky_relu ────────────────────────────────────────────────────────────────

void mlx_inline_leaky_relu(mlx_inline_array* dst, const mlx_inline_array* a, float neg_slope) {
    const auto& x = as_arr(a);
    // leaky_relu(x) = where(x >= 0, x, neg_slope * x)
    new (dst->buf) array(mlx::core::maximum(
        mlx::core::multiply(x, array(neg_slope)),
        x));
}

// ── squeeze all size-1 axes ────────────────────────────────────────────────────

void mlx_inline_squeeze_all(mlx_inline_array* dst, const mlx_inline_array* a) {
    const auto& x = as_arr(a);
    std::vector<int> axes;
    for (int i = 0; i < (int)x.ndim(); ++i) {
        if (x.shape(i) == 1) axes.push_back(i);
    }
    if (axes.empty()) {
        new (dst->buf) array(x);
    } else {
        new (dst->buf) array(mlx::core::squeeze(x, axes));
    }
}

// ── pad ───────────────────────────────────────────────────────────────────────

void mlx_inline_pad(mlx_inline_array* dst, const mlx_inline_array* a,
                    const int* pad_widths, int ndim, float fill_value) {
    const auto& x = as_arr(a);
    std::vector<std::pair<int,int>> pw(ndim);
    for (int i = 0; i < ndim; ++i) {
        pw[i] = { pad_widths[2*i], pad_widths[2*i+1] };
    }
    new (dst->buf) array(mlx::core::pad(x, pw, array(fill_value)));
}

// ── Missing ops for pmetal-models migration ───────────────────────────────────

void mlx_inline_rsqrt(mlx_inline_array* dst, const mlx_inline_array* a) {
    new (dst->buf) array(mlx::core::rsqrt(as_arr(a)));
}

void mlx_inline_zeros_like(mlx_inline_array* dst, const mlx_inline_array* a) {
    const auto& x = as_arr(a);
    new (dst->buf) array(mlx::core::zeros_like(x));
}

void mlx_inline_ones_like(mlx_inline_array* dst, const mlx_inline_array* a) {
    const auto& x = as_arr(a);
    new (dst->buf) array(mlx::core::ones_like(x));
}

void mlx_inline_tile(mlx_inline_array* dst, const mlx_inline_array* a, const int* reps, int ndim) {
    std::vector<int> r(reps, reps + ndim);
    new (dst->buf) array(mlx::core::tile(as_arr(a), r));
}

void mlx_inline_linspace(mlx_inline_array* dst, float start, float stop, int n, int dtype) {
    new (dst->buf) array(mlx::core::linspace(start, stop, n, dtype_from_int(dtype)));
}

void mlx_inline_split_sections(mlx_inline_array* dst_arr, const mlx_inline_array* a,
                                int sections, int axis, int* out_count) {
    auto parts = mlx::core::split(as_arr(a), sections, axis);
    *out_count = (int)parts.size();
    for (int i = 0; i < (int)parts.size(); i++) {
        new (dst_arr[i].buf) array(parts[i]);
    }
}

void mlx_inline_scatter_add(mlx_inline_array* dst, const mlx_inline_array* a,
                             const mlx_inline_array* indices, const mlx_inline_array* updates,
                             int axis) {
    new (dst->buf) array(mlx::core::scatter_add(as_arr(a), as_arr(indices), as_arr(updates), axis));
}

void mlx_inline_topk(mlx_inline_array* dst, const mlx_inline_array* a, int k, int axis) {
    new (dst->buf) array(mlx::core::topk(as_arr(a), k, axis));
}

void mlx_inline_put_along_axis(mlx_inline_array* dst, const mlx_inline_array* a,
                                const mlx_inline_array* indices, const mlx_inline_array* values,
                                int axis) {
    // MLX scatter can be used to implement put_along_axis
    // scatter(a, indices, values, axis) where indices shape matches values shape
    new (dst->buf) array(mlx::core::scatter(as_arr(a), {as_arr(indices)}, as_arr(values), axis));
}

void mlx_inline_layer_norm(mlx_inline_array* dst, const mlx_inline_array* x,
                            const mlx_inline_array* weight, const mlx_inline_array* bias,
                            float eps) {
    auto w_opt = weight ? std::optional<array>(as_arr(weight)) : std::nullopt;
    auto b_opt = bias   ? std::optional<array>(as_arr(bias))   : std::nullopt;
    new (dst->buf) array(mlx::core::fast::layer_norm(as_arr(x), w_opt, b_opt, eps));
}

void mlx_inline_addmm(mlx_inline_array* dst, const mlx_inline_array* c,
                       const mlx_inline_array* a, const mlx_inline_array* b) {
    // addmm(c, a, b) = c + a @ b
    new (dst->buf) array(mlx::core::addmm(as_arr(c), as_arr(a), as_arr(b)));
}

void mlx_inline_conv2d(mlx_inline_array* dst, const mlx_inline_array* input,
                       const mlx_inline_array* weight,
                       int stride_h, int stride_w, int pad_h, int pad_w,
                       int dil_h, int dil_w, int groups) {
    new (dst->buf) array(mlx::core::conv2d(
        as_arr(input), as_arr(weight),
        {stride_h, stride_w}, {pad_h, pad_w},
        {dil_h, dil_w}, groups));
}

} // extern "C" (qwen35 full forward)
