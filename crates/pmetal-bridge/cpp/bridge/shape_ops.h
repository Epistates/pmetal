// Shape queries, concat/slice/squeeze/expand/transpose, cumsum/tril,
// broadcast/flatten/tile. Matches inline_array/shape_ops.rs.

#ifndef MLX_INLINE_BRIDGE_SHAPE_OPS_H
#define MLX_INLINE_BRIDGE_SHAPE_OPS_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Shape / dtype queries ────────────────────────────────────────────────
int mlx_inline_ndim(const mlx_inline_array* a);
int mlx_inline_dim(const mlx_inline_array* a, int axis);
const int* mlx_inline_shape(const mlx_inline_array* a);
int mlx_inline_dtype(const mlx_inline_array* a);

// ── Stack / concatenate ──────────────────────────────────────────────────
void mlx_inline_stack(mlx_inline_array* dst, const mlx_inline_array* arrays, int num, int axis);
void mlx_inline_concatenate(mlx_inline_array* dst, const mlx_inline_array* arrays, int num, int axis);
// Concatenate exactly two arrays along axis (avoids heap vector)
void mlx_inline_concatenate_2(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b, int axis);

// ── Slice / slice_set ────────────────────────────────────────────────────
// Slice: a[start:stop] with stride 1 along every axis
void mlx_inline_slice(mlx_inline_array* dst, const mlx_inline_array* a, const int* start, const int* stop, int ndim);
// Slice-set (update): returns copy of a with value written into [start:stop]
void mlx_inline_slice_set(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* value, const int* start, const int* stop, int ndim);

// ── Repeat / squeeze / expand / transpose ────────────────────────────────
void mlx_inline_repeat(mlx_inline_array* dst, const mlx_inline_array* a, int repeats, int axis);
void mlx_inline_squeeze(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
// Remove all size-1 dimensions
void mlx_inline_squeeze_all(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_expand_dims(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
// Multi-axis squeeze / expand_dims with NumPy's axes: squeeze's name axes of
// the input, expand_dims' positions in the output.
void mlx_inline_squeeze_axes(mlx_inline_array* dst, const mlx_inline_array* a, const int* axes, int num_axes);
void mlx_inline_expand_dims_axes(mlx_inline_array* dst, const mlx_inline_array* a, const int* axes, int num_axes);
// Transpose with explicit axis permutation
void mlx_inline_transpose_axes(mlx_inline_array* dst, const mlx_inline_array* a, const int* axes, int ndim);

// ── Cumulative sum / lower-triangular mask ───────────────────────────────
void mlx_inline_cumsum(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
// Lower-triangular mask (k=0 includes main diagonal; negative k excludes more)
void mlx_inline_tril(mlx_inline_array* dst, const mlx_inline_array* a, int k);

// ── Broadcast / flatten / tile / split ───────────────────────────────────
void mlx_inline_broadcast_to(mlx_inline_array* dst, const mlx_inline_array* a, const int* shape, int ndim);
void mlx_inline_flatten(mlx_inline_array* dst, const mlx_inline_array* a, int start_axis, int end_axis);
void mlx_inline_tile(mlx_inline_array* dst, const mlx_inline_array* a, const int* reps, int ndim);
void mlx_inline_split_sections(mlx_inline_array* dst_arr, const mlx_inline_array* a, int sections, int axis, int* out_count);

#ifdef __cplusplus
}
#endif

#endif
