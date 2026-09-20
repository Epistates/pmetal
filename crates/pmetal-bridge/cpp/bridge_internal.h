// Shared internal helpers for bridge C++ source files.
// Not part of the public C interface (bridge.h).
#pragma once

#include "bridge.h"
#include "mlx/mlx.h"

#include <cstring>
#include <cstdlib>
#include <exception>
#include <optional>
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

// Standard try/catch wrapper for ops that construct a single output via
// placement-new into `dst->buf`. On success, clears any prior error and
// runs `body` (which is expected to placement-new into dst->buf). On
// failure, sets the thread-local error state AND placement-news a scalar
// zero into dst->buf so Rust's drop never calls `~array()` on uninit memory.
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
            new ((dst)->buf) array(0.0f); \
        } catch (...) { \
            pmetal_bridge_set_last_error((op_name), "unknown C++ exception"); \
            new ((dst)->buf) array(0.0f); \
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
// three slots, with dummies standing in for the tensors a dense weight does
// not have.

// The part of a projection that selects a kernel. The arrays themselves arrive
// through the compiled function's input vector; this is what gets captured.
struct QProjSig {
    int quantized = 0;
    int has_biases = 0;
    int has_global = 0;
    int group_size = 0;
    int bits = 0;
    int mode = 0;

    bool operator==(const QProjSig& other) const {
        return quantized == other.quantized && has_biases == other.has_biases
            && has_global == other.has_global
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
        w->global_scale != nullptr ? 1 : 0,
        w->group_size, w->bits, w->mode};
}

// A stand-in for a slot the projection does not use. Shared across every
// block: a compiled trace keys on shape and dtype, so one scalar serves all.
static inline const array& qproj_dummy() {
    static array dummy(0.0f);
    return dummy;
}

// Append one projection's four slots: weight, scales, biases, global scale.
static inline void push_qproj(std::vector<array>& ins, const mlx_inline_qweight* w) {
    ins.push_back(as_arr(w->weight));
    ins.push_back(w->scales != nullptr ? as_arr(w->scales) : qproj_dummy());
    ins.push_back(w->biases != nullptr ? as_arr(w->biases) : qproj_dummy());
    ins.push_back(w->global_scale != nullptr ? as_arr(w->global_scale) : qproj_dummy());
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
    const array& global_scale,
    const QProjSig& sig
) {
    if (sig.quantized == 0) {
        return mlx::core::matmul(x, weight);
    }
    // ⚠️ A two-level weight can use neither kernel here. `quantized_matmul`
    // has no global-scale argument and would drop the per-tensor factor in
    // silence; `qqmm` requires both global scales for nvfp4 or neither, since
    // it is written for a quantized activation and this one is bf16. Unpack
    // and multiply densely: correct, at the cost of a materialisation.
    if (sig.has_global) {
        auto dense = mlx::core::dequantize(
            weight, scales,
            sig.has_biases ? std::optional<array>(biases) : std::nullopt,
            sig.group_size, sig.bits, quant_mode_from_int(sig.mode),
            /* global_scale */ std::optional<array>(global_scale));
        return mlx::core::matmul(x, mlx::core::transpose(dense));
    }
    return mlx::core::quantized_matmul(
        x, weight, scales,
        sig.has_biases ? std::optional<array>(biases) : std::nullopt,
        /* transpose */ true, sig.group_size, sig.bits,
        quant_mode_from_int(sig.mode));
}

// The output width of a projection. A packed weight keeps its `[out, in]`
// orientation, so the axis to read depends on the variant.
static inline int qproj_out_dim(const mlx_inline_qweight* w) {
    const array& weight = as_arr(w->weight);
    return w->scales != nullptr ? weight.shape(0) : weight.shape(1);
}
