// Shared internal helpers for bridge C++ source files.
// Not part of the public C interface (bridge.h).
#pragma once

#include "bridge.h"
#include "mlx/mlx.h"

#include <cstring>
#include <cstdlib>
#include <exception>
#include <string>

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
