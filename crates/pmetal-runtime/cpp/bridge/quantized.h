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
// `global_scale` may be null. It is MLX's nvfp4 second level in MLX's own
// convention: the tensor's amax, which the kernel divides by 448 * 6. NVIDIA
// ModelOpt's `weight_scale_2` is the same quantity already divided, so passing
// it here unconverted returns every weight 2688 times too small.
void mlx_inline_dequantize(mlx_inline_array* dst, const mlx_inline_array* w,
    const mlx_inline_array* scales, const mlx_inline_array* biases,
    int group_size, int bits, int mode,
    const mlx_inline_array* global_scale);

// Quantized matmul: x @ dequantize(w, scales, biases)
void mlx_inline_quantized_matmul(mlx_inline_array* dst,
    const mlx_inline_array* x, const mlx_inline_array* w,
    const mlx_inline_array* scales, const mlx_inline_array* biases,
    bool transpose, int group_size, int bits, int mode);

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
