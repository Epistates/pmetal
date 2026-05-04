// Inline array bridge — stores mlx::core::array on the Rust stack.
// Zero heap allocation per op. Direct C++ calls.
//
// Split into multiple files for maintainability:
//   bridge.cpp            — Core array lifecycle + fundamental math ops
//   bridge_inference.cpp  — Additional inference ops, sampling, memory
//   bridge_turboquant.cpp — TurboQuant Metal kernels + bridge functions
//   bridge_compiled.cpp   — Fused compiled ops (@mx.compile equivalents)
//   bridge_native.cpp     — GDN/Attention/MoE native forward pass
//   bridge_training.cpp   — Training ops, autograd, FFT

#include "bridge_internal.h"
#include "mlx/primitives.h"  // for typeid on Primitive subclasses
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <typeinfo>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <string>
#include <sys/sysctl.h>

static_assert(sizeof(array) <= MLX_ARRAY_SIZE, "MLX_ARRAY_SIZE too small");
static_assert(alignof(array) <= MLX_ARRAY_ALIGN, "MLX_ARRAY_ALIGN too small");

// ---------------------------------------------------------------------------
// Thread-local error channel
// ---------------------------------------------------------------------------
//
// Single translation unit owns the storage. Internal setters are exposed
// via bridge_internal.h so other .cpp files in the bridge can reuse the
// BRIDGE_TRY_{DST,VOID} macros; public query functions in bridge.h let
// Rust read and clear the slot.

namespace {
    thread_local int32_t g_bridge_error_code = 0;
    thread_local std::string g_bridge_error_message;

    // Process-wide toggle for stderr emission when an exception is caught
    // inside a BRIDGE_TRY_{DST,VOID} wrapper. Writing to stderr on every
    // failure makes the *first* exception visible to the user even when
    // they haven't peppered their code with check_last_error() calls —
    // critical for bring-up work, since the silent scalar-zero sentinel
    // tensor would otherwise flow 3–4 ops downstream before a cryptic
    // shape panic shows up in an unrelated place.
    //
    // Default: on in debug builds, off in release. Override at runtime via
    // the PMETAL_BRIDGE_LOG_ERRORS env var ("1"/"0"/"true"/"false") or
    // pmetal_bridge_set_error_log_mode() from Rust.
    std::atomic<bool>& bridge_error_log_flag() {
        static std::atomic<bool> enabled{[]() -> bool {
            const char* env = std::getenv("PMETAL_BRIDGE_LOG_ERRORS");
            if (env && env[0] != '\0') {
                return env[0] == '1' || env[0] == 't' || env[0] == 'T';
            }
#ifdef NDEBUG
            return false;
#else
            return true;
#endif
        }()};
        return enabled;
    }
}

void pmetal_bridge_set_last_error(const char* op, const char* what) noexcept {
    // code=1 for std::exception, code=2 for unknown (the BRIDGE_TRY macros
    // tag the "unknown" variant by passing a fixed sentinel message).
    g_bridge_error_code =
        (what && std::strcmp(what, "unknown C++ exception") == 0) ? 2 : 1;
    try {
        g_bridge_error_message.clear();
        if (op)   { g_bridge_error_message.append("[").append(op).append("] "); }
        if (what) { g_bridge_error_message.append(what); }
    } catch (...) {
        // Even message formatting can OOM; keep the code set so Rust
        // still detects failure, just with an empty message.
        g_bridge_error_message.clear();
    }

    if (bridge_error_log_flag().load(std::memory_order_relaxed)) {
        // fprintf is signal-safe enough for this use and avoids pulling in
        // iostream on the bridge's hot path. Single call to keep the line
        // atomic with respect to other threads' stderr writes.
        std::fprintf(stderr, "[pmetal-bridge] exception in [%s]: %s\n",
                     op ? op : "(unknown op)",
                     what ? what : "(no message)");
    }
}

void pmetal_bridge_clear_error_internal() noexcept {
    g_bridge_error_code = 0;
    g_bridge_error_message.clear();
}

extern "C" {

int32_t pmetal_bridge_last_error_code(void) {
    return g_bridge_error_code;
}

const char* pmetal_bridge_last_error_message(void) {
    return g_bridge_error_message.c_str();
}

void pmetal_bridge_clear_error(void) {
    pmetal_bridge_clear_error_internal();
}

void pmetal_bridge_set_error_log_mode(int32_t enabled) {
    bridge_error_log_flag().store(enabled != 0, std::memory_order_relaxed);
}

int32_t pmetal_bridge_get_error_log_mode(void) {
    return bridge_error_log_flag().load(std::memory_order_relaxed) ? 1 : 0;
}

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
    BRIDGE_TRY_DST("matmul", dst,
        new (dst->buf) array(mlx::core::matmul(as_arr(a), as_arr(b))));
}

void mlx_inline_add(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("add", dst,
        new (dst->buf) array(mlx::core::add(as_arr(a), as_arr(b))));
}

void mlx_inline_multiply(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("multiply", dst,
        new (dst->buf) array(mlx::core::multiply(as_arr(a), as_arr(b))));
}

void mlx_inline_subtract(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("subtract", dst,
        new (dst->buf) array(mlx::core::subtract(as_arr(a), as_arr(b))));
}

void mlx_inline_divide(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b) {
    BRIDGE_TRY_DST("divide", dst,
        new (dst->buf) array(mlx::core::divide(as_arr(a), as_arr(b))));
}

void mlx_inline_negative(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("negative", dst,
        new (dst->buf) array(mlx::core::negative(as_arr(a))));
}

void mlx_inline_exp(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("exp", dst,
        new (dst->buf) array(mlx::core::exp(as_arr(a))));
}

void mlx_inline_sigmoid(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("sigmoid", dst,
        new (dst->buf) array(mlx::core::sigmoid(as_arr(a))));
}

void mlx_inline_silu(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("silu", dst, {
        auto& x = as_arr(a);
        new (dst->buf) array(mlx::core::multiply(x, mlx::core::sigmoid(x)));
    });
}

void mlx_inline_softmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis) {
    BRIDGE_TRY_DST("softmax", dst,
        new (dst->buf) array(mlx::core::softmax(as_arr(a), axis)));
}

void mlx_inline_sqrt(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("sqrt", dst,
        new (dst->buf) array(mlx::core::sqrt(as_arr(a))));
}

void mlx_inline_transpose(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("transpose", dst,
        new (dst->buf) array(mlx::core::transpose(as_arr(a))));
}

void mlx_inline_reshape(mlx_inline_array* dst, const mlx_inline_array* a, const int* shape, int ndim) {
    BRIDGE_TRY_DST("reshape", dst,
        new (dst->buf) array(mlx::core::reshape(as_arr(a), {shape, shape + ndim})));
}

void mlx_inline_sum_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    BRIDGE_TRY_DST("sum_axis", dst,
        new (dst->buf) array(mlx::core::sum(as_arr(a), axis, keepdims)));
}

void mlx_inline_astype(mlx_inline_array* dst, const mlx_inline_array* a, int dtype) {
    BRIDGE_TRY_DST("astype", dst,
        new (dst->buf) array(mlx::core::astype(as_arr(a), dtype_from_int(dtype))));
}

void mlx_inline_to_fp8(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("to_fp8", dst,
        new (dst->buf) array(mlx::core::to_fp8(as_arr(a))));
}

void mlx_inline_from_fp8(mlx_inline_array* dst, const mlx_inline_array* a, int dtype) {
    BRIDGE_TRY_DST("from_fp8", dst,
        new (dst->buf) array(mlx::core::from_fp8(as_arr(a), dtype_from_int(dtype))));
}

// Gather MM
void mlx_inline_gather_mm(
    mlx_inline_array* dst,
    const mlx_inline_array* a, const mlx_inline_array* b,
    const mlx_inline_array* lhs, const mlx_inline_array* rhs, bool sorted) {
    BRIDGE_TRY_DST("gather_mm", dst, {
        auto lhs_opt = lhs ? std::optional<array>(as_arr(lhs)) : std::nullopt;
        auto rhs_opt = rhs ? std::optional<array>(as_arr(rhs)) : std::nullopt;
        new (dst->buf) array(mlx::core::gather_mm(as_arr(a), as_arr(b), lhs_opt, rhs_opt, sorted));
    });
}

// Fast ops
void mlx_inline_rms_norm(mlx_inline_array* dst, const mlx_inline_array* x,
                          const mlx_inline_array* weight, float eps) {
    BRIDGE_TRY_DST("rms_norm", dst, {
        auto w = weight ? std::optional<array>(as_arr(weight)) : std::nullopt;
        new (dst->buf) array(mlx::core::fast::rms_norm(as_arr(x), w, eps));
    });
}

void mlx_inline_rope(mlx_inline_array* dst, const mlx_inline_array* x,
                      int dims, bool traditional, float base, float scale, int offset) {
    BRIDGE_TRY_DST("rope", dst,
        new (dst->buf) array(mlx::core::fast::rope(
            as_arr(x), dims, traditional, std::optional<float>(base), scale, offset)));
}

// RoPE with an explicit inverse-frequency array. Used by templates like
// Gemma 4's ProportionalRoPE where only a subset of the head dim rotates,
// but the rotation pairing still uses the full head dim — passing freqs
// with `inf` in the non-rotated slots tells mlx to skip those
// dimensions without any slicing on the caller side.
void mlx_inline_rope_with_freqs(mlx_inline_array* dst, const mlx_inline_array* x,
                                 int dims, bool traditional, float scale, int offset,
                                 const mlx_inline_array* freqs) {
    BRIDGE_TRY_DST("rope_with_freqs", dst,
        new (dst->buf) array(mlx::core::fast::rope(
            as_arr(x), dims, traditional, std::nullopt, scale, offset,
            std::optional<array>(as_arr(freqs)))));
}

void mlx_inline_sdpa(mlx_inline_array* dst,
                      const mlx_inline_array* q, const mlx_inline_array* k,
                      const mlx_inline_array* v, float scale, const char* mask_mode) {
    BRIDGE_TRY_DST("sdpa", dst, {
        std::string mode = mask_mode ? mask_mode : "";
        new (dst->buf) array(mlx::core::fast::scaled_dot_product_attention(
            as_arr(q), as_arr(k), as_arr(v), scale, mode));
    });
}

// Per-position RoPE: applies an array of int offsets (one per token) to x.
// Used by DDTree-style tree verify where each tree node has its own depth
// rather than a single sequential offset. `offset_arr` must be a 1-D int32
// InlineArray of length `seq_len` (the `q_len` dim of x).
void mlx_inline_rope_with_pos_ids(mlx_inline_array* dst,
                                    const mlx_inline_array* x,
                                    int dims, bool traditional, float base, float scale,
                                    const mlx_inline_array* offset_arr) {
    BRIDGE_TRY_DST("rope_with_pos_ids", dst,
        new (dst->buf) array(mlx::core::fast::rope(
            as_arr(x), dims, traditional, std::optional<float>(base), scale,
            as_arr(offset_arr))));
}

// `mlx_inline_sdpa_with_mask` is defined in bridge_inference.cpp.

// Split (writes N+1 arrays into pre-allocated output slots). If MLX throws
// before split produces its outputs, we must still placement-new every
// output slot so Rust's Drop doesn't `~array()` uninitialised memory.
void mlx_inline_split(const mlx_inline_array* input, const int* indices, int num_indices,
                       int axis, mlx_inline_array* outputs) {
    try {
        auto results = mlx::core::split(as_arr(input), {indices, indices + num_indices}, axis);
        for (size_t i = 0; i < results.size(); i++) {
            new (outputs[i].buf) array(std::move(results[i]));
        }
        pmetal_bridge_clear_error_internal();
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("split", e.what());
        // Caller's `outputs` slice length is num_indices + 1.
        int expected = num_indices + 1;
        for (int i = 0; i < expected; i++) new (outputs[i].buf) array(0.0f);
    } catch (...) {
        pmetal_bridge_set_last_error("split", "unknown C++ exception");
        int expected = num_indices + 1;
        for (int i = 0; i < expected; i++) new (outputs[i].buf) array(0.0f);
    }
}

void mlx_inline_concatenate(mlx_inline_array* dst, const mlx_inline_array* arrays,
                              int num, int axis) {
    BRIDGE_TRY_DST("concatenate", dst, {
        std::vector<array> arrs;
        arrs.reserve(num);
        for (int i = 0; i < num; i++) arrs.push_back(as_arr(&arrays[i]));
        new (dst->buf) array(mlx::core::concatenate(arrs, axis));
    });
}

void mlx_inline_argpartition(mlx_inline_array* dst, const mlx_inline_array* a, int kth, int axis) {
    BRIDGE_TRY_DST("argpartition", dst,
        new (dst->buf) array(mlx::core::argpartition(as_arr(a), kth, axis)));
}

void mlx_inline_take_along_axis(mlx_inline_array* dst, const mlx_inline_array* a,
                                  const mlx_inline_array* indices, int axis) {
    BRIDGE_TRY_DST("take_along_axis", dst,
        new (dst->buf) array(mlx::core::take_along_axis(as_arr(a), as_arr(indices), axis)));
}

// Eval — no dst to re-initialise; just routes any thrown exception into the
// thread-local error channel.
void mlx_inline_eval(mlx_inline_array* a) {
    BRIDGE_TRY_VOID("eval", as_arr(a).eval());
}
void mlx_inline_async_eval(mlx_inline_array* a) {
    BRIDGE_TRY_VOID("async_eval", mlx::core::async_eval(as_arr(a)));
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
    // Reverse of dtype_from_int in bridge_internal.h. Returns -1 for any
    // dtype not in the enumeration so callers can detect "unknown" instead
    // of silently treating it as Float32.
    auto dt = as_arr(a).dtype();
    if (dt == mlx::core::bool_)     return 0;
    if (dt == mlx::core::uint8)     return 1;
    if (dt == mlx::core::uint16)    return 2;
    if (dt == mlx::core::uint32)    return 3;
    if (dt == mlx::core::uint64)    return 4;
    if (dt == mlx::core::int8)      return 5;
    if (dt == mlx::core::int16)     return 6;
    if (dt == mlx::core::int32)     return 7;
    if (dt == mlx::core::int64)     return 8;
    if (dt == mlx::core::float16)   return 9;
    if (dt == mlx::core::float32)   return 10;
    if (dt == mlx::core::bfloat16)  return 11;
    if (dt == mlx::core::complex64) return 12;
    return -1; // sentinel: unknown dtype
}

// Item extraction. Scalar returns mean we can't use BRIDGE_TRY_DST; use a
// manual try/catch that sets the thread-local error and returns a sentinel.
float mlx_inline_item_f32(mlx_inline_array* a) {
    try {
        as_arr(a).eval();
        float v = as_arr(a).item<float>();
        pmetal_bridge_clear_error_internal();
        return v;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("item_f32", e.what());
        return 0.0f;
    } catch (...) {
        pmetal_bridge_set_last_error("item_f32", "unknown C++ exception");
        return 0.0f;
    }
}
uint32_t mlx_inline_item_u32(mlx_inline_array* a) {
    try {
        as_arr(a).eval();
        uint32_t v = as_arr(a).item<uint32_t>();
        pmetal_bridge_clear_error_internal();
        return v;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("item_u32", e.what());
        return 0;
    } catch (...) {
        pmetal_bridge_set_last_error("item_u32", "unknown C++ exception");
        return 0;
    }
}

void mlx_inline_sign(mlx_inline_array* dst, const mlx_inline_array* a) {
    BRIDGE_TRY_DST("sign", dst,
        new (dst->buf) array(mlx::core::sign(as_arr(a))));
}

void mlx_inline_dequantize(mlx_inline_array* dst, const mlx_inline_array* w,
    const mlx_inline_array* scales, const mlx_inline_array* biases,
    int group_size, int bits) {
    BRIDGE_TRY_DST("dequantize", dst,
        new (dst->buf) array(mlx::core::dequantize(
            as_arr(w), as_arr(scales), as_arr(biases), group_size, bits)));
}

void mlx_inline_from_f32_slice(mlx_inline_array* dst, const float* data, const int* shape, int ndim) {
    BRIDGE_TRY_DST("from_f32_slice", dst, {
        mlx::core::Shape s(shape, shape + ndim);
        new (dst->buf) array(data, s, mlx::core::float32);
    });
}

void mlx_inline_from_u32_slice(
    mlx_inline_array* dst,
    const uint32_t* data,
    const int* shape,
    int ndim) {
    BRIDGE_TRY_DST("from_u32_slice", dst, {
        mlx::core::Shape s(shape, shape + ndim);
        new (dst->buf) array(data, s, mlx::core::uint32);
    });
}

void mlx_inline_from_u8_slice(
    mlx_inline_array* dst,
    const uint8_t* data,
    const int* shape,
    int ndim) {
    BRIDGE_TRY_DST("from_u8_slice", dst, {
        mlx::core::Shape s(shape, shape + ndim);
        new (dst->buf) array(data, s, mlx::core::uint8);
    });
}

void mlx_inline_from_u16_bits_slice(
    mlx_inline_array* dst,
    const uint16_t* data,
    const int* shape,
    int ndim,
    int dtype) {
    BRIDGE_TRY_DST("from_u16_bits_slice", dst, {
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
    });
}

// Copy the evaluated f32 data of an array into a caller-provided buffer.
// The array is cast to float32 and eval'd first. `n` must equal the total
// element count (product of all dimensions). Returns 0 on success, -1 on a
// count mismatch or caught C++ exception. Callers can disambiguate via
// `pmetal_bridge_last_error_code()`.
int mlx_inline_to_f32_slice(mlx_inline_array* a, float* out, size_t n) {
    try {
        array& src = as_arr(a);
        array f32_arr = src.dtype() == mlx::core::float32
            ? src
            : mlx::core::astype(src, mlx::core::float32);
        f32_arr.eval();
        if ((size_t)f32_arr.size() != n) {
            pmetal_bridge_set_last_error("to_f32_slice", "size mismatch");
            return -1;
        }
        std::memcpy(out, f32_arr.data<float>(), n * sizeof(float));
        pmetal_bridge_clear_error_internal();
        return 0;
    } catch (const std::exception& e) {
        pmetal_bridge_set_last_error("to_f32_slice", e.what());
        return -1;
    } catch (...) {
        pmetal_bridge_set_last_error("to_f32_slice", "unknown C++ exception");
        return -1;
    }
}

void mlx_inline_stack(mlx_inline_array* dst, const mlx_inline_array* arrays, int num, int axis) {
    BRIDGE_TRY_DST("stack", dst, {
        std::vector<array> arrs;
        arrs.reserve(num);
        for (int i = 0; i < num; ++i) {
            arrs.push_back(*reinterpret_cast<const array*>(arrays[i].buf));
        }
        new (dst->buf) array(mlx::core::stack(arrs, axis));
    });
}

void mlx_inline_norm_l2(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims) {
    BRIDGE_TRY_DST("norm_l2", dst,
        new (dst->buf) array(mlx::core::sqrt(mlx::core::sum(
            mlx::core::square(as_arr(a)), axis, keepdims))));
}

// Conv1d
void mlx_inline_conv1d(mlx_inline_array* dst, const mlx_inline_array* input,
                         const mlx_inline_array* weight, int stride, int padding,
                         int dilation, int groups) {
    BRIDGE_TRY_DST("conv1d", dst,
        new (dst->buf) array(mlx::core::conv1d(
            as_arr(input), as_arr(weight), stride, padding, dilation, groups)));
}

// Print size for Rust to use
size_t mlx_inline_array_size(void) { return sizeof(array); }
size_t mlx_inline_array_align(void) { return alignof(array); }

void mlx_inline_enable_compile(void) { mlx::core::enable_compile(); }
void mlx_inline_disable_compile(void) { mlx::core::disable_compile(); }
void mlx_inline_clear_cache(void) { mlx::core::clear_cache(); }

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

void mlx_inline_reset_default_stream(void) {
    // Restore MLX's original default stream (GPU stream on the default device).
    // Must be called after generation completes and before InlineArray drops,
    // otherwise array destructors execute on the generation stream which can
    // race with Metal teardown and cause SIGSEGV.
    mlx::core::set_default_stream(
        mlx::core::default_stream(mlx::core::default_device()));
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

} // extern "C"
