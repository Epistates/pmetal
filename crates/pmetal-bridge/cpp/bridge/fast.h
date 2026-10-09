// Fast fused ops: rms_norm, rope, sdpa, conv, tri_inv, svd,
// clip, log_softmax, cross_entropy, layer_norm, addmm, pad, split.
// Matches inline_array/fast.rs.

#ifndef MLX_INLINE_BRIDGE_FAST_H
#define MLX_INLINE_BRIDGE_FAST_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── RMS norm / LayerNorm ─────────────────────────────────────────────────
void mlx_inline_rms_norm(mlx_inline_array* dst, const mlx_inline_array* x,
    const mlx_inline_array* weight, float eps);
void mlx_inline_layer_norm(mlx_inline_array* dst, const mlx_inline_array* x, const mlx_inline_array* weight, const mlx_inline_array* bias, float eps);

// ── Rotary position embedding ────────────────────────────────────────────
void mlx_inline_rope(mlx_inline_array* dst, const mlx_inline_array* x,
    int dims, bool traditional, float base, float scale, int offset);
void mlx_inline_rope_with_freqs(mlx_inline_array* dst, const mlx_inline_array* x,
    int dims, bool traditional, float scale, int offset,
    const mlx_inline_array* freqs);
// Per-position RoPE: applies an array of int32 offsets (one per token).
// Used by DDTree-style tree verify where each tree node has its own depth.
void mlx_inline_rope_with_pos_ids(mlx_inline_array* dst, const mlx_inline_array* x,
    int dims, bool traditional, float base, float scale,
    const mlx_inline_array* offset_arr);

// ── Scaled-dot-product attention ─────────────────────────────────────────
void mlx_inline_sdpa(mlx_inline_array* dst,
    const mlx_inline_array* q, const mlx_inline_array* k,
    const mlx_inline_array* v, float scale, const char* mask_mode);
void mlx_inline_sdpa_with_mask(mlx_inline_array* dst,
    const mlx_inline_array* q, const mlx_inline_array* k,
    const mlx_inline_array* v, float scale,
    const mlx_inline_array* mask);
// SDPA with per-head attention sinks (`sinks [n_heads]`, one extra logit in
// each row's softmax that contributes no value). `mask_mode` is "" or
// "causal"; `mask` (nullable) is an additive or boolean array mask.
void mlx_inline_sdpa_sinks(mlx_inline_array* dst,
    const mlx_inline_array* q, const mlx_inline_array* k,
    const mlx_inline_array* v, float scale, const char* mask_mode,
    const mlx_inline_array* mask, const mlx_inline_array* sinks);

// ── Split ────────────────────────────────────────────────────────────────
void mlx_inline_split(const mlx_inline_array* input, const int* indices, int num_indices,
    int axis, mlx_inline_array* outputs);

// ── Convolution ──────────────────────────────────────────────────────────
void mlx_inline_conv1d(mlx_inline_array* dst, const mlx_inline_array* input,
    const mlx_inline_array* weight, int stride, int padding, int dilation, int groups);
void mlx_inline_conv2d(mlx_inline_array* dst, const mlx_inline_array* input, const mlx_inline_array* weight, int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups);

// ── Triangular inverse (CPU stream for WY factorization) ────────────────
// upper=false -> lower triangular (default).
// Runs on the CPU stream: tri_inv(StreamOrDevice::cpu()).
void mlx_inline_tri_inv(mlx_inline_array* dst, const mlx_inline_array* a, bool upper, bool use_cpu);

// ── SVD ──────────────────────────────────────────────────────────────────
// Singular Value Decomposition (economy/thin SVD).
// Writes U -> dst_u, S -> dst_s, Vt -> dst_vt.
// Always runs on the CPU stream (GPU SVD not yet in MLX).
void mlx_inline_svd(
    mlx_inline_array* dst_u,
    mlx_inline_array* dst_s,
    mlx_inline_array* dst_vt,
    const mlx_inline_array* a);

// ── Clip / log_softmax / cross_entropy / addmm ───────────────────────────
void mlx_inline_clip(mlx_inline_array* dst, const mlx_inline_array* a, const mlx_inline_array* lo, const mlx_inline_array* hi);
void mlx_inline_log_softmax(mlx_inline_array* dst, const mlx_inline_array* a, int axis);
void mlx_inline_cross_entropy(mlx_inline_array* dst, const mlx_inline_array* logits, const mlx_inline_array* targets, int axis);
// Sparse cross-entropy: `targets` is an integer array of class indices whose
// shape equals `logits.shape` with `axis` removed. Output is per-position NLL
// in nats with that same reduced shape. Composes `logsumexp(logits, axis) -
// take_along_axis(logits, targets_expanded, axis).squeeze(axis)` inside a
// single bridge call so the `[..., V]` log-softmax tensor never materializes.
void mlx_inline_cross_entropy_sparse(mlx_inline_array* dst, const mlx_inline_array* logits, const mlx_inline_array* indices, int axis);
void mlx_inline_addmm(mlx_inline_array* dst, const mlx_inline_array* c, const mlx_inline_array* a, const mlx_inline_array* b);

// ── Pad ──────────────────────────────────────────────────────────────────
// pad_widths: flat array of [before_0, after_0, before_1, after_1, ...] length 2*ndim.
// mode: an MLX pad mode ("constant", "edge", "reflect", "symmetric"); MLX
// throws on any other. fill_value is used by "constant" only.
void mlx_inline_pad(mlx_inline_array* dst, const mlx_inline_array* a,
                    const int* pad_widths, int ndim, float fill_value,
                    const char* mode);

#ifdef __cplusplus
}
#endif

#endif
