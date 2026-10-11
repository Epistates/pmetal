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

// True when `a` is being traced by a function transform: it is (or was
// computed from) an input that value_and_grad / vjp / jvp differentiates,
// including inside a checkpoint's recomputation during the backward pass.
bool mlx_inline_is_tracer(const mlx_inline_array* a);

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

// ── Cut cross-entropy, fused (bridge_cce.cpp) ────────────────────────────
// Each token's logsumexp over the vocabulary and its target logit, both [n]
// f32, of the logits softcap(logit_scale * (hidden · weightᵀ + bias)), from
// hidden [n, k] and weight [v, k] without the [n, v] logits. A custom VJP:
// gradients reach hidden, weight and bias (`bias` may be NULL). A target
// outside [0, v) has a target logit of zero. The backward leaves out each
// 8-token group's 64-row vocabulary tiles where its softmax stays under
// `filter_eps` (0 leaves out nothing) and holds one `vocab_chunk`-column chunk
// of logit gradients at a time. `want_dx` /
// `want_dw` say which gradients autograd will ask for; they only shape the
// backward's memory schedule.
//
// Returns 0 on success; 1 when the fused kernels can't take these inputs
// (`*reason` says why, outputs are placeholders); 2 on an exception (in the
// error channel).
int mlx_inline_cce_fused(mlx_inline_array* dst_lse, mlx_inline_array* dst_target,
                         const mlx_inline_array* hidden, const mlx_inline_array* weight,
                         const mlx_inline_array* targets, const mlx_inline_array* bias,
                         float logit_scale, float softcap, float filter_eps,
                         int vocab_chunk, bool want_dx, bool want_dw,
                         const char** reason);

#ifdef __cplusplus
}
#endif

#endif
