// TurboQuant fused Metal kernels.
//
// These replace the expand_dims+subtract+square+argmin chain (which allocates
// a [N, D, C] intermediate tensor) with single-dispatch Metal kernels that
// keep everything in thread registers.
//
// Pipeline split:
//   The Rust caller handles norm computation (keys.norm_l2) and rotation
//   (keys.matmul(rot_t)) — both are standard MLX ops with no intermediates.
//   These kernels handle ONLY the innermost bottleneck.
//
// turboquant_encode:
//   input [N, D] f32  — already normalised + rotated
//   codebook [C] f32  — sorted Lloyd-Max centroids (C = 2^bits, max 16)
//   → indices [N, D] uint32 — nearest centroid index per coordinate
//   (out_norms is reserved for ABI symmetry but not written by the kernel)
//
// turboquant_decode:
//   indices [N, D] uint32   — centroid indices
//   norms [N] f32           — reserved for ABI symmetry (not read by kernel)
//   codebook [C] f32        — same centroids used at encode time
//   → output [N, D] f32     — centroid values in the rotated domain
//   (caller multiply-by-norms and matmul-with-rotation happen outside)
//
// Both return 0 on success, 1 if Metal kernel is unavailable.
//
// ─────────────────────────────────────────────────────────────────────────
// Storage layout invariants (audit-pinned 2026-04-25)
// ─────────────────────────────────────────────────────────────────────────
// Pack/encode/decode/score MUST agree on these axis orders. Drift here
// historically produced silently-wrong attention scores rather than a clean
// crash, so the invariants are documented here and mirrored as Rust static
// asserts in `src/turboquant.rs::layout_invariants`.
//
// Where:
//   N            = kv_rows (batch * kv_heads)
//   D            = head_dim (key_dim or value_dim — must match for the same
//                  tensor)
//   S_cap        = cache_seq_capacity (rounded-up token slots)
//   ceil(D/32)   = qjl_words (sign-bit packing along D, 32 bits per word)
//   C            = n_centroids (2^bits, ≤ 16 for non-fullbyte; up to 256
//                  for the q8 d256 fullbyte path)
//
// Live-tensor cache layouts (the "shadow" layouts the score kernels read):
//   indices       — [N, D, S_cap]               uint8  (7-bit centroid id)
//   qjl_signs     — [N, ceil(D/32), S_cap]      uint32 (packed sign bits)
//   norms         — [N, S_cap]                  f32    (slot l2 norm)
//   residual_norms — [N, S_cap]                 f32    (residual l2 norm)
//   key_slot_scale — [N, S_cap]                 f32    (per-row codebook
//                                                       scaling factor:
//                                                       max(|rotated|)/centroid_max)
//   codebook      — [C]                         f32    (sorted centroids)
//
// Scratch / one-shot encode-decode buffers:
//   indices_2d    — [N, D]                      uint32
//   sign_bits     — [N, ceil(D/32)]             uint32
//   keys_packed_2d — [N, D]                     uint32
//
// q8 fullbyte (D=256, n_seq>=1024) uses a seq-major transposition + a packed
// per-slot scalar buffer that the long-context score kernels offset into:
//   q8_fullbyte_seq    — [N, S_cap, D]          uint8
//   q8_slot_scales_seq — [N, S_cap, 4]          f32  packed scalars
//                          [0]=key_norm, [1]=residual_norm,
//                          [2]=value_norm, [3]=key_slot_scale
//   d256_rot_values_seq — [N, S_cap, D]         bf16
//
// Mixed-precision split layouts (regular + outlier sub-vectors):
//   regular_indices   — [KvRows, S_cap, D_reg]
//   outlier_indices   — [KvRows, S_cap, D_out]
//   regular_qjl_signs — [KvRows, S_cap, ceil(D_reg/32)]
//   outlier_qjl_signs — [KvRows, S_cap, ceil(D_out/32)]
//
// GPU-attention coverage (head_dim, config):
//   * Uniform + head_dim == 128  →  fused d128 attention kernel
//   * Uniform + head_dim == 256  →  fused d256 attention kernel
//                                    (+ q8 fullbyte specialisation when
//                                     bits==8 and n_seq >= 1024)
//   * Uniform + other head_dim  →  dequantize + SDPA fallback
//   * Mixed precision           →  dequantize + SDPA fallback (no kernel)

#ifndef MLX_INLINE_BRIDGE_TURBOQUANT_H
#define MLX_INLINE_BRIDGE_TURBOQUANT_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Encode / decode primitives ───────────────────────────────────────────
// Fused encode: nearest-centroid search over a small codebook.
// input: [N, D] f32 (normalised+rotated).  codebook: [C] f32 (C <= 16).
// out_indices: [N, D] uint32.  out_norms: reserved (may be NULL).
int mlx_inline_turboquant_encode(
    mlx_inline_array*       out_indices,   // [N, D] uint32  (written)
    mlx_inline_array*       out_norms,     // reserved — may be NULL
    const mlx_inline_array* input,         // [N, D] f32
    const mlx_inline_array* codebook,      // [C]    f32
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows);

// Fused decode: codebook lookup → [N, D] f32 centroid values.
// indices: [N, D] uint32.  norms: reserved (may be NULL).  codebook: [C] f32.
// out: [N, D] f32 centroid values in the rotated domain (un-scaled by norm).
int mlx_inline_turboquant_decode(
    mlx_inline_array*       out,           // [N, D] f32      (written)
    const mlx_inline_array* indices,       // [N, D] uint32
    const mlx_inline_array* norms,         // reserved — may be NULL
    const mlx_inline_array* codebook,      // [C]    f32
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows);

// ── Score kernels ────────────────────────────────────────────────────────
// Fused TurboQuant key scoring.
// query_rot/query_proj: [N, D] f32
// indices:              [N, D, S_cap] uint8 transposed for score-friendly seq access
// qjl_signs:            [N, ceil(D/32), S_cap] packed uint32 sign words
// norms/residual_norms/key_slot_scale: [N, S_cap] f32
// codebook:             [C] f32
// out_scores:           [N, S] f32
int mlx_inline_turboquant_score(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    const mlx_inline_array* norms,
    const mlx_inline_array* residual_norms,
    const mlx_inline_array* key_slot_scale,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                qjl_words,
    uint32_t                n_centroids,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized q8 key scoring for D=256 on the seq-major transposed cache layout.
int mlx_inline_turboquant_score_q8_d256(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    const mlx_inline_array* norms,
    const mlx_inline_array* residual_norms,
    const mlx_inline_array* key_slot_scale,
    const mlx_inline_array* codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Fused mixed TurboQuant key scoring.
// regular/outlier query tensors: [N, D_reg]/[N, D_out] f32
// regular/outlier indices: [KvRows, S_cap, D_*]
// regular/outlier qjl_signs: [KvRows, S_cap, ceil(D_*/32)] packed uint32 words
// regular/outlier norms/residual_norms/slot_scale: [KvRows, S_cap] f32
// out_scores: [N, S] f32
int mlx_inline_turboquant_mixed_score(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* regular_query_rot,
    const mlx_inline_array* regular_query_proj,
    const mlx_inline_array* regular_indices,
    const mlx_inline_array* regular_qjl_signs,
    const mlx_inline_array* regular_norms,
    const mlx_inline_array* regular_residual_norms,
    const mlx_inline_array* regular_slot_scale,
    const mlx_inline_array* regular_codebook,
    const mlx_inline_array* outlier_query_rot,
    const mlx_inline_array* outlier_query_proj,
    const mlx_inline_array* outlier_indices,
    const mlx_inline_array* outlier_qjl_signs,
    const mlx_inline_array* outlier_norms,
    const mlx_inline_array* outlier_residual_norms,
    const mlx_inline_array* outlier_slot_scale,
    const mlx_inline_array* outlier_codebook,
    uint32_t                regular_dim,
    uint32_t                regular_qjl_words,
    uint32_t                regular_n_centroids,
    uint32_t                outlier_dim,
    uint32_t                outlier_qjl_words,
    uint32_t                outlier_n_centroids,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// ── Pack / unpack helpers ────────────────────────────────────────────────
// Pack sign(projected >= 0) along the last dimension into uint32 words.
// projected: [N, D] f32
// out:       [N, ceil(D/32)] uint32
int mlx_inline_turboquant_pack_sign_bits(
    mlx_inline_array*       out,
    const mlx_inline_array* projected,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows);

// Pack q8 TurboQuant key bytes from centroid indices plus QJL signs.
// indices:   [N, D, S_cap] uint8   (7-bit centroid indices)
// qjl_signs: [N, ceil(D/32), S_cap] uint32 packed sign words
// out:       [N, D, S_cap] uint8   (low 7 bits index, high bit sign)
int mlx_inline_turboquant_pack_q8_keybytes(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity);

// Pack q8 TurboQuant key bytes into a seq-major shadow layout.
// indices:   [N, D, S_cap] uint8
// qjl_signs: [N, ceil(D/32), S_cap] uint32
// out:       [N, S_cap, D] uint8
int mlx_inline_turboquant_pack_q8_keybytes_seq(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity);

// Pack q8 TurboQuant key bytes and value indices into a seq-major shadow.
// indices:       [N, D, S_cap] uint8
// qjl_signs:     [N, ceil(D/32), S_cap] uint32
// value_indices: [N, S_cap, D] uint8
// out:           [N, S_cap, D] uint16
//                low byte = key byte, high byte = value centroid index
int mlx_inline_turboquant_pack_q8_kvbytes_seq(
    mlx_inline_array*       out,
    const mlx_inline_array* indices,
    const mlx_inline_array* qjl_signs,
    const mlx_inline_array* value_indices,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                cache_seq_capacity);

// Phase F (Hamming skip-list): XOR + popcount Hamming distance between a
// per-q-row query sign hash and a [kv_rows, S, packed_dim] cache of key
// sign hashes.
//   query_signs:    [N, packed_dim] uint32                (N = q_rows)
//   key_signs:      [kv_rows, S, packed_dim] uint32       (kv_rows = N / groups)
//   out_distances:  [N, S] uint32  (Hamming distance, 0..D)
//   groups:         q_heads / kv_heads — kernel maps row → kv_row = row / groups.
//                   Pass groups=1 for non-GQA models (recovers original 1:1).
int mlx_inline_turboquant_hamming_distances(
    mlx_inline_array*       out,
    const mlx_inline_array* query_signs,
    const mlx_inline_array* key_signs,
    uint32_t                packed_dim,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                groups);

// Unpack uint32 sign words back to {-1,+1} f32 signs.
// packed: [N, ceil(D/32)] uint32
// out:    [N, D] f32
int mlx_inline_turboquant_unpack_sign_bits(
    mlx_inline_array*       out,
    const mlx_inline_array* packed,
    uint32_t                dim,
    uint32_t                packed_dim,
    uint32_t                n_rows);

// input:       [N, dim] f32 (dim must be a power of two)
// left_signs:  [dim] f32 (Rademacher ±1)
// right_signs: [dim] f32 (Rademacher ±1)
// out:         [N, dim] f32  (Walsh-Hadamard transform with 1/sqrt(dim) scale)
int mlx_inline_turboquant_signed_fwht_pow2_rows(
    mlx_inline_array*       out,
    const mlx_inline_array* input,
    const mlx_inline_array* left_signs,
    const mlx_inline_array* right_signs,
    uint32_t                dim,
    uint32_t                n_rows);

// ── Weighted-decode / attention fused kernels ────────────────────────────
// Fused TurboQuant value aggregation in the rotated domain.
// weights:  [N, S] f32
// indices:  [N, D, S_cap] uint8
// norms:    [N, S_cap] f32
// codebook: [C] f32
// out:      [N, D] f32 aggregated rotated vectors
int mlx_inline_turboquant_weighted_decode(
    mlx_inline_array*       out,
    const mlx_inline_array* weights,
    const mlx_inline_array* indices,
    const mlx_inline_array* norms,
    const mlx_inline_array* codebook,
    uint32_t                dim,
    uint32_t                n_centroids,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads);

// Specialized long-context q8 decode primitive for D=256/V=256.
// Computes TurboQuant attention directly from compressed K/V in two passes:
// pass 1 emits per-block partial accumulators + log-sum-exp stats,
// pass 2 merges those partials into the final rotated output.
int mlx_inline_turboquant_attention_q8_d256_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* key_qjl_signs,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_residual_norms,
    const mlx_inline_array* key_slot_scale,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_norms,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Variant F (NoQjl) of the q8 D=256 2-pass primitive. Codebook gets the full
// 8 bits, no QJL residual term — drops `query_proj`, `key_qjl_signs`, and
// `key_residual_norms` from the parameter list. Score formula:
//   s = key_norm * slot_scale * (q_rot · codebook[idx])
int mlx_inline_turboquant_attention_q8_d256_no_qjl_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_slot_scale,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_norms,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized long-context q8 decode primitive for D=256/V=256 over packed
// key bytes stored as [N, S_cap, D] (low 7 bits centroid index, high bit QJL sign).
int mlx_inline_turboquant_attention_q8_d256_packed_keys_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_bytes,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized long-context q8 decode primitive for D=256/V=256 over a
// seq-major packed key shadow plus dense rotated values:
// - `key_bytes`: [N, S_cap, D] uint8
//   low 7 bits = key centroid index, high bit = QJL sign
// - `value_dense`: [N, S_cap, D] bf16/f32 rotated dense values
int mlx_inline_turboquant_attention_q8_d256_packed_keys_dense_values_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_bytes,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized long-context q8 decode primitive for D=256/V=256 over a
// seq-major pure-q8 key shadow plus dense rotated values:
// - `key_indices`: [N, S_cap, D] uint8, full 8-bit centroid index
// - `value_dense`: [N, S_cap, D] bf16/f32 rotated dense values
int mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Full-byte D256 pass-1 state output only.
// Returns:
// - partials: [N, blocks, 256]
// - sums:     [N, blocks]
// - maxs:     [N, blocks]
int mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_state(
    mlx_inline_array*       out_partials,
    mlx_inline_array*       out_sums,
    mlx_inline_array*       out_maxs,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Full-byte D256 pass-1 output only.
// Returns unmerged partial outputs [N, blocks, 256].
int mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_pass1(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Merge precomputed D256 2-pass partials/maxs/sums.
int mlx_inline_turboquant_attention_q8_d256_pass2_merge(
    mlx_inline_array*       out,
    const mlx_inline_array* partials,
    const mlx_inline_array* sums,
    const mlx_inline_array* maxs,
    uint32_t                n_rows,
    uint32_t                blocks);

// Full-byte D256 2-pass variant that uses a block-local 2-loop softmax
// in pass 1 instead of online renormalization.
int mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_localsoftmax(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Full-byte D256 score-only long-context kernel.
int mlx_inline_turboquant_score_q8_d256_fullbyte(
    mlx_inline_array*       out_scores,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// D256 dense-value weighted sum over resident rotated values.
int mlx_inline_turboquant_weighted_sum_d256_dense_values(
    mlx_inline_array*       out,
    const mlx_inline_array* weights,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads);

// Specialized long-context q8 decode primitive for D=256/V=256 over a
// seq-major packed `{key,value}` shadow:
// - `kv_bytes`: [N, S_cap, D] uint16
//   low byte = key byte, high byte = value centroid index
int mlx_inline_turboquant_attention_q8_d256_packed_kv_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* kv_bytes,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized long-context q8 decode primitive for D=256/V=256 over a
// seq-major packed key shadow plus dense rotated values:
// - `kv_bytes`: [N, S_cap, D] uint16
//   low byte = key byte, high byte unused by this path
// - `value_dense`: [N, S_cap, D] bf16/f32 rotated dense values
int mlx_inline_turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* kv_bytes,
    const mlx_inline_array* slot_scales,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_dense,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized long-context q8 decode primitive for D=128/V=128.
// Computes TurboQuant attention directly from compressed K/V in two passes:
// pass 1 emits per-block partial accumulators + log-sum-exp stats,
// pass 2 merges those partials into the final rotated output.
int mlx_inline_turboquant_attention_q8_d128_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* key_qjl_signs,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_residual_norms,
    const mlx_inline_array* key_slot_scale,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_norms,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Variant F (NoQjl) of the q8 D=128 2-pass primitive. Codebook gets the full
// 8 bits, no QJL residual term — drops `query_proj`, `key_qjl_signs`, and
// `key_residual_norms` from the parameter list. Score formula:
//   s = key_norm * slot_scale * (q_rot · codebook[idx])
int mlx_inline_turboquant_attention_q8_d128_no_qjl_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* key_indices,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_slot_scale,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_norms,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// Specialized long-context q8 decode primitive for D=128/V=128 over packed
// key bytes stored as [N, D, S_cap] (low 7 bits centroid index, high bit QJL sign).
int mlx_inline_turboquant_attention_q8_d128_packed_keys_2pass(
    mlx_inline_array*       out,
    const mlx_inline_array* query_rot,
    const mlx_inline_array* query_proj,
    const mlx_inline_array* key_bytes,
    const mlx_inline_array* key_norms,
    const mlx_inline_array* key_residual_norms,
    const mlx_inline_array* key_slot_scale,
    const mlx_inline_array* key_codebook,
    const mlx_inline_array* value_indices,
    const mlx_inline_array* value_norms,
    const mlx_inline_array* value_codebook,
    uint32_t                n_rows,
    uint32_t                n_seq,
    uint32_t                cache_seq_capacity,
    uint32_t                q_heads,
    uint32_t                kv_heads,
    uint32_t                attn_scale_bits);

// ── Gather/scatter helpers for mixed TurboQuant component layouts ────────
// input: [N, D] f32, positions: [P] int32, out: [N, P] f32
int mlx_inline_turboquant_gather_last_dim(
    mlx_inline_array*       out,
    const mlx_inline_array* input,
    const mlx_inline_array* positions,
    uint32_t                full_dim,
    uint32_t                out_dim,
    uint32_t                n_rows);

// regular/outlier: [N, R]/[N, O] f32, positions: [R]/[O] int32, out: [N, D] f32
int mlx_inline_turboquant_scatter_last_dim(
    mlx_inline_array*       out,
    const mlx_inline_array* regular,
    const mlx_inline_array* outlier,
    const mlx_inline_array* regular_positions,
    const mlx_inline_array* outlier_positions,
    uint32_t                full_dim,
    uint32_t                regular_dim,
    uint32_t                outlier_dim,
    uint32_t                n_rows);

#ifdef __cplusplus
}
#endif

#endif
