// gather_mm, take_* variants, kv_cache_append, index, scatter_add,
// put_along_axis, argpartition. Matches inline_array/gather.rs + bits of
// shape_ops.rs that live in the gather-ish space.

#ifndef MLX_INLINE_BRIDGE_GATHER_H
#define MLX_INLINE_BRIDGE_GATHER_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Gather matmul ────────────────────────────────────────────────────────
void mlx_inline_gather_mm(mlx_inline_array* dst,
    const mlx_inline_array* a, const mlx_inline_array* b,
    const mlx_inline_array* lhs, const mlx_inline_array* rhs, bool sorted);

// ── Partition / take / index ─────────────────────────────────────────────
void mlx_inline_argpartition(mlx_inline_array* dst, const mlx_inline_array* a, int kth, int axis);
void mlx_inline_take_along_axis(mlx_inline_array* dst, const mlx_inline_array* a,
    const mlx_inline_array* indices, int axis);
// Take rows along axis (for embedding lookup: take(weight, indices, axis=0))
void mlx_inline_take_axis(mlx_inline_array* dst, const mlx_inline_array* a,
    const mlx_inline_array* indices, int axis);
// Index / embedding lookup: take(a, indices) — flat gather over all elements
void mlx_inline_index(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* indices);

// ── KV cache helper ──────────────────────────────────────────────────────
// Equivalent to concatenate([cached, new], axis=2) for [B, H, T, D] format.
void mlx_inline_kv_cache_append(mlx_inline_array* dst,
    const mlx_inline_array* cached, const mlx_inline_array* new_kv, int axis);

// ── Scatter / put ────────────────────────────────────────────────────────
void mlx_inline_scatter_add(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* indices, const mlx_inline_array* updates, int axis);
void mlx_inline_put_along_axis(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* indices, const mlx_inline_array* values, int axis);

#ifdef __cplusplus
}
#endif

#endif
