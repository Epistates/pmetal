// Gated Delta Network (GDN) recurrence: ops fallback + Metal kernel variants.
// Matches inline_array/gdn_methods.rs.

#ifndef MLX_INLINE_BRIDGE_GDN_H
#define MLX_INLINE_BRIDGE_GDN_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// GDN Metal kernel step with pre-computed g and beta.
// q, k: [B, T, Hk, Dk], v: [B, T, Hv, Dv], g: [B, T, Hv], beta: [B, T, Hv]
// state_in: [B, Hv, Dv, Dk], T: sequence length (usually 1 for decode)
// Returns y → dst_y, new_state → dst_state. Falls back to ops if Metal unavailable.
void mlx_inline_gdn_metal_step(
    mlx_inline_array* dst_y,
    mlx_inline_array* dst_state,
    const mlx_inline_array* q,
    const mlx_inline_array* k,
    const mlx_inline_array* v,
    const mlx_inline_array* g,
    const mlx_inline_array* beta,
    const mlx_inline_array* state_in,
    int T);

// GDN state-only advance for speculative-decoding rollback replay.
// Same shapes as `mlx_inline_gdn_metal_step` but WITHOUT the `q` / `y`
// channels: the caller only needs the post-replay state. Saves roughly
// half the per-step work versus dispatching the full step kernel and
// discarding `y`.
void mlx_inline_gdn_metal_state_update(
    mlx_inline_array* dst_state,
    const mlx_inline_array* k,
    const mlx_inline_array* v,
    const mlx_inline_array* g,
    const mlx_inline_array* beta,
    const mlx_inline_array* state_in,
    int T);

#ifdef __cplusplus
}
#endif

#endif
