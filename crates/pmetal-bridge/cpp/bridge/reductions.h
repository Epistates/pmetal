// Arg-reductions, value reductions (sum/mean/max/min/logsumexp),
// top-k, abs. Matches inline_array/reductions.rs.

#ifndef MLX_INLINE_BRIDGE_REDUCTIONS_H
#define MLX_INLINE_BRIDGE_REDUCTIONS_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Arg reductions / sampling ────────────────────────────────────────────
void mlx_inline_argmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_argmin(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_logsumexp(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);
void mlx_inline_categorical(mlx_inline_array* dst, const mlx_inline_array* logits);

// ── Element-wise absolute value ──────────────────────────────────────────
void mlx_inline_abs(mlx_inline_array* dst, const mlx_inline_array* a);

// ── Mean reductions ──────────────────────────────────────────────────────
void mlx_inline_mean_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);
void mlx_inline_mean_all(mlx_inline_array* dst, const mlx_inline_array* a);

// ── Sort / sum / max / min along axes ────────────────────────────────────
void mlx_inline_argsort(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_sum_all(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_max_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);
void mlx_inline_min_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);

// ── Multi-axis sum/mean ──────────────────────────────────────────────────
void mlx_inline_sum_axes(mlx_inline_array* dst, const mlx_inline_array* a, const int* axes, int num_axes, bool keepdims);
void mlx_inline_mean_axes(mlx_inline_array* dst, const mlx_inline_array* a, const int* axes, int num_axes, bool keepdims);
// Logical-OR reduction to bool. `axes == NULL` reduces every axis.
void mlx_inline_any(mlx_inline_array* dst, const mlx_inline_array* a, const int* axes, int num_axes, bool keepdims);

// ── Top-k ────────────────────────────────────────────────────────────────
void mlx_inline_topk(mlx_inline_array* dst, const mlx_inline_array* a, int k, int axis);

#ifdef __cplusplus
}
#endif

#endif
