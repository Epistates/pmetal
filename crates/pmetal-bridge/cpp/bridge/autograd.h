// value_and_grad + checkpoint trampolines. Matches inline_array/autograd.rs.

#ifndef MLX_INLINE_BRIDGE_AUTOGRAD_H
#define MLX_INLINE_BRIDGE_AUTOGRAD_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Autograd: value_and_grad ─────────────────────────────────────────────
// Callback type for Rust forward function.
typedef void (*mlx_rust_forward_fn)(
    const mlx_inline_array* const* all_arrays,
    int n_total,
    mlx_inline_array* loss_out,
    void* ctx
);

// Compute loss + gradients via callback-based autograd.
// forward_fn is called ONCE with traced arrays to build the computation graph.
// Gradients are computed w.r.t. the first n_params arrays.
void mlx_inline_value_and_grad(
    mlx_rust_forward_fn forward_fn,
    void* ctx,
    const mlx_inline_array* const* all_arrays,
    int n_params,
    int n_total,
    mlx_inline_array* loss_out,
    mlx_inline_array** grads_out
);

// ── Gradient checkpointing ───────────────────────────────────────────────
// Callback type for the checkpointed inner function.
// Writes output arrays into outputs_out[0..*n_outputs_out-1].
typedef void (*mlx_rust_checkpoint_fn)(
    const mlx_inline_array* const* all_arrays,
    int n_total,
    mlx_inline_array* outputs_out,
    int* n_outputs_out,
    void* ctx
);

// Called once, when C++ releases `ctx`. Frees the boxed Rust closure.
typedef void (*mlx_rust_drop_fn)(void* ctx);

// Apply gradient checkpointing to a forward function over the given inputs.
// The inner function may produce multiple output arrays (n_outputs_max capacity).
// On return, dst_outputs[0..*n_outputs_written-1] hold the output arrays.
//
// Takes ownership of `ctx`: checkpoint() re-invokes `forward_fn` during the
// backward pass, long after this call returns, so `ctx` must outlive the graph
// rather than the call. `drop_fn` is invoked when the last copy of the internal
// closure is destroyed.
void mlx_inline_checkpoint(
    mlx_rust_checkpoint_fn forward_fn,
    void* ctx,
    mlx_rust_drop_fn drop_fn,
    const mlx_inline_array* const* all_arrays,
    int n_total,
    int n_outputs_max,
    mlx_inline_array* dst_outputs,
    int* n_outputs_written
);

#ifdef __cplusplus
}
#endif

#endif
