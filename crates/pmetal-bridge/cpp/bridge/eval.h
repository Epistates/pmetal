// Evaluation, detach, item extraction, layout queries.
// Matches inline_array/eval.rs (with a couple of state-query decls
// from the diagnostics-adjacent region).

#ifndef MLX_INLINE_BRIDGE_EVAL_H
#define MLX_INLINE_BRIDGE_EVAL_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Eval ─────────────────────────────────────────────────────────────────
void mlx_inline_eval(mlx_inline_array* a);
void mlx_inline_async_eval(mlx_inline_array* a);
void mlx_inline_eval_2(mlx_inline_array* a, mlx_inline_array* b);
// Eval many arrays in one call (single GPU submission, no per-array sync)
void mlx_inline_eval_many(mlx_inline_array** arrays, int count);
// Async eval for a single borrowed InlineArray
void mlx_inline_async_eval_arr(const mlx_inline_array* a);

// Detach: sever the computation graph, freeing all input references.
// Critical for caches: without this, cache updates chain across steps,
// keeping ALL previous steps' Metal buffers alive (memory leak).
void mlx_inline_detach(mlx_inline_array* a);

// ── Item extraction ──────────────────────────────────────────────────────
float mlx_inline_item_f32(mlx_inline_array* a);
uint32_t mlx_inline_item_u32(mlx_inline_array* a);

// ── Layout / buffer queries ──────────────────────────────────────────────
size_t mlx_inline_size(const mlx_inline_array* a);
size_t mlx_inline_nbytes(const mlx_inline_array* a);
int mlx_inline_data_ptr(const mlx_inline_array* a, const void** out_ptr);
// Stable identity for the underlying lazy array desc. Safe on unevaluated
// arrays — returns `uintptr_t(array_desc_.get())` without materialization.
uintptr_t mlx_inline_array_id(const mlx_inline_array* a);

#ifdef __cplusplus
}
#endif

#endif
