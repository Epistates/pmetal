// Shared internal helpers for bridge C++ source files.
// Not part of the public C interface (bridge.h).
#pragma once

#include "bridge.h"
#include "mlx/mlx.h"

#include <cstring>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using mlx::core::array;

static inline array& as_arr(mlx_inline_array* a) {
    return *reinterpret_cast<array*>(a->buf);
}
static inline const array& as_arr(const mlx_inline_array* a) {
    return *reinterpret_cast<const array*>(a->buf);
}

// ---------------------------------------------------------------------------
// Thread-local error channel (definition in bridge.cpp)
// ---------------------------------------------------------------------------
//
// Bridge entry points historically caught C++ exceptions, printed them to
// stderr, and returned silently-zeroed arrays — which looked like success
// to Rust and produced NaN propagation or silent wrong answers. The two
// helpers below, paired with the public query API in bridge.h, let Rust
// detect whether the last bridge call on this thread threw.
//
// Ownership: both functions act on a `thread_local` string + int32 code
// owned by bridge.cpp. Successful ops clear the state; failures set it.
// Rust reads `pmetal_bridge_last_error_code` / `pmetal_bridge_last_error_message`
// and MUST copy the message before issuing the next bridge call on the
// same thread.

// Internal setters — used by BRIDGE_TRY_DST / BRIDGE_TRY_VOID macros below.
void pmetal_bridge_set_last_error(const char* op, const char* what) noexcept;
void pmetal_bridge_clear_error_internal() noexcept;

// Construct a placeholder array into `dst` without letting an exception
// escape. Every path that must leave `dst` holding a valid array uses this:
// error paths (so Rust's drop never runs ~array() on uninit memory) and
// empty initialisation.
//
// The placeholder is normally a scalar zero, which allocates through MLX's
// Metal allocator. When the failure being reported is the Metal device itself
// (an unloadable metallib, say), that allocation throws the same error again,
// and a C++ exception unwinding into Rust aborts the process. The fallback is
// an empty array over a null buffer with a no-op deleter: it touches no GPU
// state, and with zero elements nothing can read through the null pointer.
static inline void bridge_placeholder(mlx_inline_array* dst) noexcept {
    try {
        new (dst->buf) array(0.0f);
    } catch (...) {
        new (dst->buf) array(
            mlx::core::allocator::Data(
                mlx::core::allocator::Buffer(nullptr),
                [](mlx::core::allocator::Buffer) {}),
            mlx::core::Shape{0},
            mlx::core::float32);
    }
}

// How many autograd traces the bridge has open on this thread
// (`value_and_grad`, and `checkpoint`'s forward). MLX keeps the same count
// (`detail::in_grad_tracing`), but its counter is a thread-local that
// libmlx.dylib doesn't export, so a shared-MLX build can't link against it.
// `inline`, not `static`: one counter for every translation unit.
inline int& pmetal_bridge_grad_depth() noexcept {
    thread_local int depth = 0;
    return depth;
}

struct BridgeGradScope {
    BridgeGradScope() noexcept { ++pmetal_bridge_grad_depth(); }
    ~BridgeGradScope() { --pmetal_bridge_grad_depth(); }
    BridgeGradScope(const BridgeGradScope&) = delete;
    BridgeGradScope& operator=(const BridgeGradScope&) = delete;
};

// `fast::scaled_dot_product_attention`, as every bridge entry point runs it.
//
// Under autograd with no array mask, MLX (since #4563) runs a fused forward
// that also outputs the logsumexp its fused Metal backward reads. Two cases
// can't take that path, and MLX sends them down it anyway:
//   - 8 queries or fewer: the forward picks the vector kernel, which outputs
//     no logsumexp, and evaluation throws;
//   - attention sinks: the fused backward has none ("NYI: attention sinks").
// For those, the mask goes in as an array (causal, or all-true), which keeps
// the forward and its gradient on MLX's unfused ops. The result is the same.
// Drop this once MLX routes these cases to the unfused path itself.
static inline array bridge_sdpa(const array& q, const array& k, const array& v, float scale,
                                const std::string& mask_mode,
                                const std::optional<array>& mask,
                                const std::optional<array>& sinks) {
    namespace mx = mlx::core;
    if (!mask && pmetal_bridge_grad_depth() > 0 && (sinks || q.shape(2) <= 8)) {
        const int q_len = q.shape(2);
        const int k_len = k.shape(2);
        // MLX's causal mask: query i sees keys up to i + (k_len - q_len).
        array explicit_mask = mask_mode == "causal"
            ? mx::greater_equal(
                  mx::expand_dims(mx::arange(k_len - q_len, k_len, mx::int32), 1),
                  mx::expand_dims(mx::arange(k_len, mx::int32), 0))
            : mx::ones({q_len, k_len}, mx::bool_);
        return mx::fast::scaled_dot_product_attention(
            q, k, v, scale, "", std::optional<array>(explicit_mask), sinks);
    }
    return mx::fast::scaled_dot_product_attention(q, k, v, scale, mask_mode, mask, sinks);
}

// ---------------------------------------------------------------------------
// MLX's compile cache
// ---------------------------------------------------------------------------
//
// `mlx::core::compile` keeps its traces in one process-wide table (a map of
// vectors of entries) that MLX does not lock: finding or adding an entry from
// two threads at once corrupts it, and an entry is marked filled before its
// tape is built, so a second thread that finds it in between replays an empty
// tape. The bridge's own per-block caches are vectors of the same kind. Every
// entry point that calls a compiled function (building the graph, never
// evaluating it) holds this for the call. Recursive, since a Rust closure
// MLX traces may call another compiled op on the same thread. Leaked, so it
// has no destructor to order against libmlx's at exit.
inline std::recursive_mutex& compile_cache_mutex() {
    static auto* mutex = new std::recursive_mutex;
    return *mutex;
}

class CompileCacheLock {
public:
    CompileCacheLock() : guard_(compile_cache_mutex()) {}

private:
    std::lock_guard<std::recursive_mutex> guard_;
};

// ---------------------------------------------------------------------------
// Callbacks into Rust
// ---------------------------------------------------------------------------
//
// `value_and_grad`, `checkpoint` and the generic compile wrapper call a Rust
// closure from inside MLX. The closure's Rust side catches a panic (one can
// not unwind through this frame) and reports it as a non-zero return; a
// non-zero return is turned into this exception, which MLX unwinds through
// like any other and the bridge entry point catches. The panic itself is
// re-raised by the Rust caller once the entry point has returned.
struct RustCallbackPanicked : std::runtime_error {
    RustCallbackPanicked() : std::runtime_error("a Rust callback panicked") {}
};

// The arrays handed to a callback: copies of `args` in raw buffers, destroyed
// with this object, on every path out including an exception.
class CallbackInputs {
public:
    explicit CallbackInputs(const std::vector<array>& args)
        : bufs_(args.size()), ptrs_(args.size()) {
        for (size_t i = 0; i < args.size(); ++i) {
            new (bufs_[i].buf) array(args[i]);
            ptrs_[i] = &bufs_[i];
        }
    }
    ~CallbackInputs() {
        for (auto& b : bufs_) {
            as_arr(&b).~array();
        }
    }
    CallbackInputs(const CallbackInputs&) = delete;
    CallbackInputs& operator=(const CallbackInputs&) = delete;

    const mlx_inline_array* const* data() const { return ptrs_.data(); }
    int size() const { return static_cast<int>(ptrs_.size()); }

private:
    std::vector<mlx_inline_array> bufs_;
    std::vector<const mlx_inline_array*> ptrs_;
};

// Output slots a callback writes into. They start uninitialised: the callback
// placement-news the slots it uses and says how many (`written`), and only
// those are read back and destroyed. Pre-filling them would leak one array per
// slot per call, since placement-new does not destroy what it overwrites.
class CallbackOutputs {
public:
    explicit CallbackOutputs(int capacity) : bufs_(capacity > 0 ? capacity : 0) {}
    ~CallbackOutputs() {
        for (int i = 0; i < written_; ++i) {
            as_arr(&bufs_[i]).~array();
        }
    }
    CallbackOutputs(const CallbackOutputs&) = delete;
    CallbackOutputs& operator=(const CallbackOutputs&) = delete;

    mlx_inline_array* data() { return bufs_.data(); }
    int capacity() const { return static_cast<int>(bufs_.size()); }
    // Take ownership of the first `n` slots (the count the callback reported).
    void adopt(int n) { written_ = n; }
    const array& operator[](int i) const { return as_arr(&bufs_[i]); }

private:
    std::vector<mlx_inline_array> bufs_;
    int written_ = 0;
};

// Standard try/catch wrapper for ops that construct a single output via
// placement-new into `dst->buf`. On success, clears any prior error and
// runs `body` (which is expected to placement-new into dst->buf). On
// failure, sets the thread-local error state AND fills dst with
// `bridge_placeholder` so Rust's drop never calls `~array()` on uninit memory.
//
// `body` is variadic so call sites can pass expressions containing
// unparenthesised commas (e.g. templates like `std::pair<int,int>`) — a
// fixed 3-arg macro would split on those commas.
#define BRIDGE_TRY_DST(op_name, dst, ...) \
    do { \
        try { \
            __VA_ARGS__; \
            pmetal_bridge_clear_error_internal(); \
        } catch (const std::exception& e) { \
            pmetal_bridge_set_last_error((op_name), e.what()); \
            bridge_placeholder(dst); \
        } catch (...) { \
            pmetal_bridge_set_last_error((op_name), "unknown C++ exception"); \
            bridge_placeholder(dst); \
        } \
    } while (0)

// Variant for ops with no single dst buffer (void-returning, in-place,
// multi-output, or query functions with a scalar return handled at the
// callsite). Body must NOT throw past the macro — it sets thread-local
// error state on any exception and otherwise runs to completion.
#define BRIDGE_TRY_VOID(op_name, ...) \
    do { \
        try { \
            __VA_ARGS__; \
            pmetal_bridge_clear_error_internal(); \
        } catch (const std::exception& e) { \
            pmetal_bridge_set_last_error((op_name), e.what()); \
        } catch (...) { \
            pmetal_bridge_set_last_error((op_name), "unknown C++ exception"); \
        } \
    } while (0)

// Guard for calls that are not ops: memory queries, cache and stream
// management, synchronize. These reach the Metal device (and synchronize
// rethrows a failed command buffer), so an exception must not escape into
// Rust. A failure is reported through the error channel, but unlike the
// BRIDGE_TRY_* macros a success leaves the channel alone: these run between
// an op and its error check (memory logging, syncs) and must not erase it.
#define BRIDGE_GUARD(op_name, ...) \
    do { \
        try { \
            __VA_ARGS__; \
        } catch (const std::exception& e) { \
            pmetal_bridge_set_last_error((op_name), e.what()); \
        } catch (...) { \
            pmetal_bridge_set_last_error((op_name), "unknown C++ exception"); \
        } \
    } while (0)

// Make `x` safe to read as `x.size()` contiguous elements starting at
// `x.data<T>()`, in row-major order.
//
// MLX arrays are strided. A transpose, a slice of an inner axis or a step,
// and a broadcast all share their source's buffer and only change strides
// (a broadcast also has `data_size() < size()`), so walking `size()` elements
// from the data pointer reads the wrong elements, or runs past the end of
// the buffer. This evaluates `x` (waiting on an in-flight async eval) and,
// only when it is not row-contiguous, swaps a row-contiguous copy into `x`
// with `copy_shared_buffer`, the same move MLX's `Contiguous` primitive makes
// on its output. The logical values are unchanged, so every handle sharing
// `x` still sees the same array; views of the same source keep their own
// buffer. A row-contiguous array is left alone, so the common case stays
// zero-copy.
static inline void make_row_contiguous_for_read(array& x) {
    x.eval();
    if (x.size() == 0 || x.flags().row_contiguous) {
        return;
    }
    array packed = mlx::core::contiguous(x);
    packed.eval();
    x.copy_shared_buffer(packed);
}

// The three GELUs, evaluated in f32 (f64 stays f64). A bf16/f16 input comes
// back as f32, which is what these activations have always returned: their
// f32 constants promoted the input. Rounding the result back to the input's
// dtype is what PyTorch does, and it is a one-line change here, but it is not
// a free one. Every op after the activation then runs in bf16 too, and on
// Gemma 3 that exposes MLX's bf16 attention, which is less precise than
// PyTorch's: 640 tokens of the 1B model drift to 204x the bf16 noise floor
// against transformers, where the f32 stream this keeps stays inside it.
template <typename F>
static inline array gelu_family(const array& in, F formula) {
    using namespace mlx::core;
    const auto dt = in.dtype();
    const bool exact_float = dt == float32 || dt == float64;
    const array x = exact_float ? in : astype(in, float32);
    const auto c = [&](float v) { return array(v, x.dtype()); };
    return formula(x, c);
}

// GDN Metal kernel getter — defined in bridge_native.cpp, used across files.
mlx::core::fast::CustomKernelFunction& get_gdn_kernel();

// Map integer dtype code to MLX Dtype.
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

// Map Rust-side quantization mode discriminants to MLX quantized-matmul modes.
static inline std::string quant_mode_from_int(int mode) {
    switch (mode) {
        case 1:
            return "mxfp8";
        case 2:
            return "mxfp4";
        case 3:
            return "nvfp4";
        case 0:
        default:
            return "affine";
    }
}

// ---------------------------------------------------------------------------
// Polymorphic projections inside a compiled block
// ---------------------------------------------------------------------------
//
// A compiled trace bakes in which kernel each projection uses, so a block that
// serves both dense and packed weights needs two things: the quantization
// signature in its cache key, and a fixed input arity so the traced lambda can
// index its inputs by position. Every projection therefore contributes exactly
// `QPROJ_SLOTS` slots, with dummies standing in for the tensors a dense weight
// does not have.

// The part of a projection that selects a kernel. The arrays themselves arrive
// through the compiled function's input vector; this is what gets captured.
struct QProjSig {
    int quantized = 0;
    int has_biases = 0;
    int has_tensor_scale = 0;
    int group_size = 0;
    int bits = 0;
    int mode = 0;

    bool operator==(const QProjSig& other) const {
        return quantized == other.quantized && has_biases == other.has_biases
            && has_tensor_scale == other.has_tensor_scale
            && group_size == other.group_size && bits == other.bits
            && mode == other.mode;
    }
};

static inline QProjSig qproj_sig(const mlx_inline_qweight* w) {
    if (w == nullptr || w->scales == nullptr) {
        return QProjSig{};
    }
    return QProjSig{
        1,
        w->biases != nullptr ? 1 : 0,
        w->tensor_scale != nullptr ? 1 : 0,
        w->group_size, w->bits, w->mode};
}

// A stand-in for a slot the projection does not use. Shared across every
// block: a compiled trace keys on shape and dtype, so one scalar serves all.
static inline const array& qproj_dummy() {
    static array dummy(0.0f);
    return dummy;
}

// Append one projection's four slots: weight, scales, biases, tensor scale.
static inline void push_qproj(std::vector<array>& ins, const mlx_inline_qweight* w) {
    ins.push_back(as_arr(w->weight));
    ins.push_back(w->scales != nullptr ? as_arr(w->scales) : qproj_dummy());
    ins.push_back(w->biases != nullptr ? as_arr(w->biases) : qproj_dummy());
    ins.push_back(w->tensor_scale != nullptr ? as_arr(w->tensor_scale) : qproj_dummy());
}

// How many input slots one projection occupies.
static constexpr std::size_t QPROJ_SLOTS = 4;

// `x @ w`, dense or packed. The dense arm relies on the caller having
// pre-transposed to `[in, out]`; the packed arm asks MLX to transpose, since a
// packed tensor cannot be transposed without unpacking it first. Mirrors
// `native_weight::LayerWeight::matmul_from`.
static inline array qproj_matmul(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    const array& tensor_scale,
    const QProjSig& sig
) {
    if (sig.quantized == 0) {
        return mlx::core::matmul(x, weight);
    }
    auto y = mlx::core::quantized_matmul(
        x, weight, scales,
        sig.has_biases ? std::optional<array>(biases) : std::nullopt,
        /* transpose */ true, sig.group_size, sig.bits,
        quant_mode_from_int(sig.mode));
    if (!sig.has_tensor_scale) {
        return y;
    }
    // Constant across the tensor, so it commutes with the matmul and the
    // weight stays packed. Applied in fp32 like MLX's own kernels, then cast
    // back. Mirrors `scale_product` in native_weight.rs.
    auto dtype = y.dtype();
    return mlx::core::astype(
        mlx::core::multiply(mlx::core::astype(y, mlx::core::float32), tensor_scale),
        dtype);
}

// The output width of a projection. A packed weight keeps its `[out, in]`
// orientation, so the axis to read depends on the variant.
static inline int qproj_out_dim(const mlx_inline_qweight* w) {
    const array& weight = as_arr(w->weight);
    return w->scales != nullptr ? weight.shape(0) : weight.shape(1);
}
