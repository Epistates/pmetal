// TurboQuant KV-cache kernels: quantize-and-pack, dequantize, and decode
// attention. The Rust side is `src/turboquant/`; each function returns 0 on
// success and 1 on failure, with the reason in the bridge's error channel.
//
// Packed layout (the contract `src/turboquant/quantizer.rs` documents and the
// tests spell out independently): a vector of `dim` coordinates is `dim`
// centroid indices of `bits` bits, coordinate `i` in bits
// [i * bits, (i + 1) * bits) of a little-endian bitstream, in
// ceil(dim * bits / 32) uint32 words; an index may straddle two words. Rows
// are stored with a separate f32 scale (the norm times the gain of the reconstruction).

#ifndef MLX_INLINE_BRIDGE_TURBOQUANT_H
#define MLX_INLINE_BRIDGE_TURBOQUANT_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Normalise, quantize and pack rows.
//   rotated:    [N, dim] f32, rows already multiplied by the rotation
//   thresholds: [2^bits - 1] f32, midpoints between neighbouring centroids
//   codebook:   [2^bits] f32, the centroids
//   out_packed: [N, ceil(dim * bits / 32)] uint32
//   out_scales:  [N] f32, |x| times the least-squares gain <y, y_hat> / |y_hat|^2
//               of the reconstruction (0 for a zero or non-finite row, which
//               is stored as zeros)
int mlx_inline_turboquant_quantize_pack(
    mlx_inline_array*       out_packed,
    mlx_inline_array*       out_scales,
    const mlx_inline_array* rotated,
    const mlx_inline_array* thresholds,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                bits,
    uint32_t                n_rows);

// scale * codebook[index] for every coordinate: [N, dim] f32, rotated basis.
int mlx_inline_turboquant_dequantize(
    mlx_inline_array*       out,
    const mlx_inline_array* packed,
    const mlx_inline_array* scales,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                bits,
    uint32_t                n_rows);

// Decode attention, pass 1: per (query row, block) the unnormalised
// accumulator, running max and running sum. See bridge_turboquant_attention.cpp
// for the `params` fields.
//   q:        [R, dim]        queries, original basis, any float dtype
//   q_rot:    [R, dim]        the same rotated with the key rotation, f32
//   k_packed: [KV, cold_cap, words_k] uint32, k_scale: [KV, cold_cap] f32
//   v_packed: [KV, cold_cap, words_v] uint32, v_scale: [KV, cold_cap] f32
//   cb_k, cb_v: key and value codebooks, f32
//   dense_k:  [KV, dense_cap, dim], dense_v: [KV, dense_cap, value_dim]
//   params:   int32[15]
//   center:   [KV, dim] f32 key centre, read only when params[14] is set
//   out_acc:  [R, n_blocks, value_dim] f32, out_m / out_l: [R, n_blocks] f32
//   out_row_shift: [R] f32, the query's dot with the key centre in the
//             base-2 logit domain (zero without a centre)
// R = batch * q_heads, KV = batch * kv_heads, groups = q_heads / kv_heads.
int mlx_inline_turboquant_attention_partials(
    mlx_inline_array*       out_acc,
    mlx_inline_array*       out_m,
    mlx_inline_array*       out_l,
    mlx_inline_array*       out_row_shift,
    const mlx_inline_array* q,
    const mlx_inline_array* q_rot,
    const mlx_inline_array* k_packed,
    const mlx_inline_array* k_scale,
    const mlx_inline_array* v_packed,
    const mlx_inline_array* v_scale,
    const mlx_inline_array* cb_k,
    const mlx_inline_array* cb_v,
    const mlx_inline_array* dense_k,
    const mlx_inline_array* dense_v,
    const mlx_inline_array* params,
    const mlx_inline_array* center,
    uint32_t                dim,
    uint32_t                value_dim,
    uint32_t                key_bits,
    uint32_t                value_bits,
    uint32_t                kv_heads,
    uint32_t                groups,
    uint32_t                n_rows,
    uint32_t                n_kv_rows,
    uint32_t                n_blocks);

// Decode attention, pass 2: merge the blocks of every row, rotate the cold
// part of the accumulator back with `inv_rot` ([value_dim, value_dim] f32),
// add the dense part, normalise (with optional per-head `sinks` logits, f32
// [q_heads]) and write [R, value_dim] in `out_dtype`.
int mlx_inline_turboquant_attention_merge(
    mlx_inline_array*       out,
    const mlx_inline_array* part_acc,
    const mlx_inline_array* part_m,
    const mlx_inline_array* part_l,
    const mlx_inline_array* inv_rot,
    const mlx_inline_array* sinks,
    const mlx_inline_array* row_shift,
    const mlx_inline_array* params,
    uint32_t                value_dim,
    uint32_t                q_heads,
    uint32_t                n_rows,
    uint32_t                n_blocks,
    int                     out_dtype);

#ifdef __cplusplus
}
#endif

#endif
