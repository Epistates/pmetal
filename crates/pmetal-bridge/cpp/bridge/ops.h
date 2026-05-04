// Element-wise binary/unary ops, comparisons, softmax/norm/reshape, FFT,
// leaky_relu. Matches the Rust inline_array/ops.rs module.

#ifndef MLX_INLINE_BRIDGE_OPS_H
#define MLX_INLINE_BRIDGE_OPS_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Binary ops — result written to dst via placement new ─────────────────
void mlx_inline_matmul(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_add(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_multiply(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_subtract(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_divide(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_maximum(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_minimum(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_pow(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);

// ── Unary ops ────────────────────────────────────────────────────────────
void mlx_inline_negative(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_exp(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_sigmoid(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_silu(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_sqrt(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_transpose(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_softplus(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_log(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_sign(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_reciprocal(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_sin(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_cos(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_rsqrt(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_square(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_relu(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_gelu(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_stop_gradient(mlx_inline_array* dst, const mlx_inline_array* a);

// ── Element-wise comparisons ─────────────────────────────────────────────
void mlx_inline_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_not_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_greater(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_less(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_greater_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);
void mlx_inline_less_equal(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* b);

// ── Where: condition ? a : b ─────────────────────────────────────────────
void mlx_inline_where(mlx_inline_array* dst, const mlx_inline_array* condition, const mlx_inline_array* a, const mlx_inline_array* b);

// ── Softmax / reshape / astype / sum_axis / norm_l2 ──────────────────────
void mlx_inline_softmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_softmax_precise(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_reshape(mlx_inline_array* dst, const mlx_inline_array* a, const int* shape, int ndim);
void mlx_inline_sum_axis(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);
void mlx_inline_astype(mlx_inline_array* dst, const mlx_inline_array* a, int dtype);
void mlx_inline_to_fp8(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_from_fp8(mlx_inline_array* dst, const mlx_inline_array* a, int dtype);
void mlx_inline_norm_l2(mlx_inline_array* dst, const mlx_inline_array* a, int axis, bool keepdims);

// ── FFT ──────────────────────────────────────────────────────────────────
// rfft: real-valued FFT along the given axis. n_fft=-1 means use full axis size.
void mlx_inline_rfft(mlx_inline_array* dst, const mlx_inline_array* a, int n_fft, int axis);
// irfft: inverse rfft. n_fft=-1 means infer from input size (n = 2*(freq-1)).
void mlx_inline_irfft(mlx_inline_array* dst, const mlx_inline_array* a, int n_fft, int axis);

// ── leaky_relu ───────────────────────────────────────────────────────────
void mlx_inline_leaky_relu(mlx_inline_array* dst, const mlx_inline_array* a, float neg_slope);

#ifdef __cplusplus
}
#endif

#endif
