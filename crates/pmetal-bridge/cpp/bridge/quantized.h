// Dequantize / quantized matmul / gather_qmm / quantize.
// Matches inline_array/quantized.rs.

#ifndef MLX_INLINE_BRIDGE_QUANTIZED_H
#define MLX_INLINE_BRIDGE_QUANTIZED_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dequantize: reconstruct float from packed int + scales + biases
//
// `biases` may be null: only affine mode has them.
//
// `global_scale` may be null. nvfp4 is two-level: an fp8 scale per group *and*
// one fp32 scale for the whole tensor. NVIDIA ModelOpt checkpoints ship that
// second scale as `weight_scale_2`, and dropping it does not fail, it returns
// every weight off by a constant factor.
void mlx_inline_dequantize(mlx_inline_array* dst, const mlx_inline_array* w,
    const mlx_inline_array* scales, const mlx_inline_array* biases,
    int group_size, int bits, int mode,
    const mlx_inline_array* global_scale);

// Quantized matmul: x @ dequantize(w, scales, biases)
void mlx_inline_quantized_matmul(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* w,
    const mlx_inline_array* scales, const mlx_inline_array* biases,
    bool transpose, int group_size, int bits, int mode);

// Two-level quantized matmul (`mlx::core::qqmm`), for nvfp4 and mxfp4.
//
// Separate from `mlx_inline_quantized_matmul` because MLX's own
// `quantized_matmul` takes no global scale: a two-level weight has to go
// through this entry point or its per-tensor scale is silently dropped.
// `global_scale_x` is for a quantized *activation* and stays null on the
// weight-only path; `global_scale_w` is the weight's `weight_scale_2`.
void mlx_inline_qqmm(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* w,
    const mlx_inline_array* w_scales,
    int group_size, int bits, int mode,
    const mlx_inline_array* global_scale_x,
    const mlx_inline_array* global_scale_w);

// Two-level gather matmul (`mlx::core::gather_qqmm`) for MoE expert dispatch.
void mlx_inline_gather_qqmm(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* w,
    const mlx_inline_array* w_scales,
    const mlx_inline_array* lhs_indices, const mlx_inline_array* rhs_indices,
    int group_size, int bits, int mode,
    const mlx_inline_array* global_scale_x,
    const mlx_inline_array* global_scale_w,
    bool sorted_indices);

// Gather quantized matmul (gathers rows of w before dequantize + matmul)
void mlx_inline_gather_qmm(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* w,
    const mlx_inline_array* scales, const mlx_inline_array* biases,
    const mlx_inline_array* lhs_indices, const mlx_inline_array* rhs_indices,
    bool transpose, int group_size, int bits, bool sorted, int mode);

// Quantize weights — inverse of dequantize.
void mlx_inline_quantize(mlx_inline_array* dst_w, mlx_inline_array* dst_scales, mlx_inline_array* dst_biases,
    const mlx_inline_array* a, int group_size, int bits);

// Quantize weights in a non-affine MLX mode. Modes with no bias term (mxfp8,
// mxfp4, nvfp4) return only packed weights + scales.
void mlx_inline_quantize_mode(mlx_inline_array* dst_w, mlx_inline_array* dst_scales,
    const mlx_inline_array* a, int group_size, int bits, int mode);

#ifdef __cplusplus
}
#endif

#endif
