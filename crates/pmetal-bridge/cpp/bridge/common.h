// Shared typedefs, lifecycle, interop, error reporting.
// Every other bridge/*.h pulls this in for the `mlx_inline_array` typedef.

#ifndef MLX_INLINE_BRIDGE_COMMON_H
#define MLX_INLINE_BRIDGE_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Size/alignment of mlx::core::array — queried at build time via mlx_inline_array_size()
// Conservative upper bound; static_assert in .cpp verifies at compile time.
#define MLX_ARRAY_SIZE 128
#define MLX_ARRAY_ALIGN 8

// Stack-allocated array — NO heap allocation per op.
// Rust creates this on the stack, C++ placement-news into buf.
typedef struct {
    _Alignas(MLX_ARRAY_ALIGN) unsigned char buf[MLX_ARRAY_SIZE];
} mlx_inline_array;

// One projection weight, dense or packed. Mirrors `native_weight::LayerWeight`.
//
// `scales == NULL` means dense: `weight` is already pre-transposed to
// `[in, out]` and the op is a plain matmul. Otherwise `weight` is packed as
// the checkpoint stores it, `[out, in / (32 / bits)]`, and the op is a
// quantized_matmul with transpose = true. `biases` is NULL for the
// floating-point modes (mxfp4, nvfp4, mxfp8), which carry scales only.
//
// Passing this rather than a bare array is what lets one compiled Gemma 4
// graph serve both a bf16 and a 4-bit checkpoint. `group_size` / `bits` /
// `mode` are part of every compiled block's cache key, so a mixed-precision
// checkpoint gets one trace per distinct combination.
// `tensor_scale` is one fp32 factor for the whole tensor, so the weight is
// `dequantize(weight, scales) * tensor_scale`. NVIDIA ModelOpt ships it as
// `weight_scale_2` (nvfp4) or `weight_scale` (per-tensor fp8). It commutes with
// the matmul, so the block scales the packed product. NULL means none.
typedef struct {
    const mlx_inline_array* weight;
    const mlx_inline_array* scales;
    const mlx_inline_array* biases;
    const mlx_inline_array* tensor_scale;
    int group_size;
    int bits;
    int mode;
} mlx_inline_qweight;

// ── Lifecycle ─────────────────────────────────────────────────────────────
void mlx_inline_init_empty(mlx_inline_array* dst);
void mlx_inline_init_copy(mlx_inline_array* dst, const mlx_inline_array* src);
void mlx_inline_init_move(mlx_inline_array* dst, mlx_inline_array* src);
void mlx_inline_destroy(mlx_inline_array* a);

// ── Interop with legacy mlx_array handles ─────────────────────────────────
void mlx_inline_from_handle(mlx_inline_array* dst, void* handle_ctx);
void* mlx_inline_to_handle(const mlx_inline_array* src);

// ── Size query (for Rust build-time verification) ─────────────────────────
size_t mlx_inline_array_size(void);
size_t mlx_inline_array_align(void);

// ── Error reporting ───────────────────────────────────────────────────────
//
// Every `mlx_inline_*` entry point that can throw a C++ exception writes
// its failure into a thread-local error slot before returning. Rust callers
// can read the slot via these three functions and must copy the message
// string before issuing another bridge call on the same thread.

// Returns 0 on no error, 1 on a caught std::exception, 2 on an unknown
// (non-std) C++ exception. Thread-local.
int32_t pmetal_bridge_last_error_code(void);

// Returns a NUL-terminated message describing the most recent failure on
// this thread. Pointer is valid until the next bridge call on the same
// thread. Always returns a non-NULL pointer (empty string when no error).
const char* pmetal_bridge_last_error_message(void);

// Manually clears the thread-local error slot. Normally unnecessary —
// every successful bridge op clears it automatically.
void pmetal_bridge_clear_error(void);

// Process-wide toggle for stderr logging on caught C++ exceptions.
// When enabled, every exception caught inside a BRIDGE_TRY_{DST,VOID}
// wrapper prints a `[pmetal-bridge] exception in [op]: what()` line to
// stderr in addition to setting the thread-local error slot. Makes the
// first failure visible without requiring a check_last_error() call at
// every op site — critical for debugging, since the silent scalar-zero
// sentinel tensor otherwise propagates several ops downstream before
// showing up as an unrelated shape panic.
//
// Default: on in debug builds, off in release. Overridable at process
// start via the PMETAL_BRIDGE_LOG_ERRORS env var ("1"/"0"/"true"/"false").
void pmetal_bridge_set_error_log_mode(int32_t enabled);
int32_t pmetal_bridge_get_error_log_mode(void);

#ifdef __cplusplus
}
#endif

#endif
