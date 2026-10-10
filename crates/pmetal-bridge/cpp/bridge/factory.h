// Constructors: scalars, shape-only, slice loaders, random samplers,
// shape-shifted helpers (zeros_like/ones_like). Matches inline_array/factory.rs.

#ifndef MLX_INLINE_BRIDGE_FACTORY_H
#define MLX_INLINE_BRIDGE_FACTORY_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Scalar constructors ──────────────────────────────────────────────────
void mlx_inline_from_f32(mlx_inline_array* dst, float val);
void mlx_inline_from_i32(mlx_inline_array* dst, int val);

// ── Shape-only constructors ──────────────────────────────────────────────
// dtype codes match mlx_inline_astype.
void mlx_inline_zeros(mlx_inline_array* dst, const int* shape, int ndim, int dtype);
void mlx_inline_ones(mlx_inline_array* dst, const int* shape, int ndim, int dtype);
void mlx_inline_full(mlx_inline_array* dst, const int* shape, int ndim, float val, int dtype);
void mlx_inline_eye(mlx_inline_array* dst, int n, int dtype);
void mlx_inline_tri(mlx_inline_array* dst, int n, int m, int k, int dtype);

// Arange: create [0, 1, 2, ..., n-1] — forces full Metal buffer allocation (no broadcast)
void mlx_inline_arange(mlx_inline_array* dst, int n, int dtype);
void mlx_inline_linspace(mlx_inline_array* dst, float start, float stop, int n, int dtype);

// ── "*_like" shaped constructors ─────────────────────────────────────────
void mlx_inline_zeros_like(mlx_inline_array* dst, const mlx_inline_array* a);
void mlx_inline_ones_like(mlx_inline_array* dst, const mlx_inline_array* a);

// ── Slice loaders ────────────────────────────────────────────────────────
void mlx_inline_from_f32_slice(mlx_inline_array* dst, const float* data, const int* shape, int ndim);
void mlx_inline_from_u32_slice(mlx_inline_array* dst, const uint32_t* data, const int* shape, int ndim);
void mlx_inline_from_u8_slice(mlx_inline_array* dst, const uint8_t* data, const int* shape, int ndim);
void mlx_inline_from_u16_bits_slice(mlx_inline_array* dst, const uint16_t* data, const int* shape, int ndim, int dtype);
void mlx_inline_from_i32_slice(mlx_inline_array* dst, const int32_t* data, int len);

// Copy evaluated f32 data out of an array into a caller-provided buffer.
// Array is cast to float32 and eval'd. n must equal the total element count.
// Returns 0 on success, -1 on size mismatch.
int mlx_inline_to_f32_slice(mlx_inline_array* a, float* out, size_t n);

// ── Random samplers ──────────────────────────────────────────────────────
void mlx_inline_random_normal(mlx_inline_array* dst, const int* shape, int ndim, int dtype);
void mlx_inline_random_uniform(mlx_inline_array* dst, const int* shape, int ndim, int dtype);
void mlx_inline_random_bernoulli(mlx_inline_array* dst, const mlx_inline_array* p, const int* shape, int ndim);
void mlx_inline_random_seed(uint64_t seed);
void mlx_inline_random_key(mlx_inline_array* dst, uint64_t seed);
void mlx_inline_random_randint(mlx_inline_array* dst, int low, int high, const int* shape, int ndim, int dtype);

#ifdef __cplusplus
}
#endif

#endif
