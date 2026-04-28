//! TurboQuant fused Metal kernels surfaced as methods on [`InlineArray`].
//!
//! These wrap the GPU kernels that power TurboQuant KV-cache quantization:
//! encode/decode, key/value scoring, bit packing, attention pass-1/2 variants,
//! and gather/scatter helpers. All kernels are Metal-only and return `None`
//! when the active device cannot dispatch.

use std::mem::MaybeUninit;

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;

impl InlineArray {
    // ── TurboQuant fused Metal kernels ──────────────────────────────────

    /// Fused TurboQuant encode: nearest-centroid search over a tiny codebook.
    ///
    /// Replaces the expand_dims+subtract+square+argmin chain that allocates a
    /// huge `[N, D, C]` intermediate tensor.  For D=128, C=8 (3-bit MSE), N=100
    /// the old intermediate is 409 600 f32 elements per call; this kernel uses
    /// only registers (n_centroids ≤ 16).
    ///
    /// - `input`: `[N, D]` f32 — already normalised onto the unit sphere AND
    ///   rotated by the orthogonal projection matrix.
    /// - `codebook`: `[C]` f32, C ≤ 16.
    /// - Returns `indices [N, D]` uint32 on success, `None` if Metal unavailable.
    ///
    /// Norm computation (`keys.norm_l2(-1, true)`) and the rotation matmul are
    /// handled by the caller before calling this function.
    pub fn turboquant_encode(
        input: &Self,
        codebook: &Self,
        dim: u32,
        n_centroids: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out_indices = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_encode(
                out_indices.as_mut_ptr(),
                std::ptr::null_mut(), // norms: reserved
                &input.raw,
                &codebook.raw,
                dim,
                n_centroids,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out_indices.assume_init() },
            })
        } else {
            None
        }
    }

    /// Fused TurboQuant decode: codebook lookup producing `[N, D]` f32 centroid
    /// values in the rotated domain.
    ///
    /// Replaces: `take(codebook, flat_indices, 0).reshape(orig_shape)`.
    /// The result is **un-scaled** (no norm multiplication) and in the *rotated*
    /// domain.  The caller multiplies by norms and matmuls with the rotation
    /// matrix to recover the original input-space vectors.
    ///
    /// - `indices`: `[N, D]` uint32.
    /// - `codebook`: `[C]` f32, C ≤ 16.
    /// - Returns `output [N, D]` f32 on success, `None` if Metal unavailable.
    pub fn turboquant_decode(
        indices: &Self,
        codebook: &Self,
        dim: u32,
        n_centroids: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_decode(
                out.as_mut_ptr(),
                &indices.raw,
                std::ptr::null_mut(), // norms: reserved
                &codebook.raw,
                dim,
                n_centroids,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Fused TurboQuant key scoring directly from compressed indices/signs.
    ///
    /// Inputs:
    /// - `query_rot` / `query_proj`: `[N, D]` f32
    /// - `indices`: `[N, D, S]` transposed uint8 key indices
    /// - `qjl_signs`: `[N, ceil(D/32), S]` packed uint32 sign words
    /// - `norms` / `residual_norms` / `key_slot_scale`: `[N, S]` f32
    /// - `codebook`: `[C]` f32
    ///
    /// Returns `scores [N, S]` f32 on success.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_score(
        query_rot: &Self,
        query_proj: &Self,
        indices: &Self,
        qjl_signs: &Self,
        norms: &Self,
        residual_norms: &Self,
        key_slot_scale: &Self,
        codebook: &Self,
        dim: u32,
        qjl_words: u32,
        n_centroids: u32,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_score(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &indices.raw,
                &qjl_signs.raw,
                &norms.raw,
                &residual_norms.raw,
                &key_slot_scale.raw,
                &codebook.raw,
                dim,
                qjl_words,
                n_centroids,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized q8 key scoring for D=256 on the seq-major transposed cache layout.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_score_q8_d256(
        query_rot: &Self,
        query_proj: &Self,
        indices: &Self,
        qjl_signs: &Self,
        norms: &Self,
        residual_norms: &Self,
        key_slot_scale: &Self,
        codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_score_q8_d256(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &indices.raw,
                &qjl_signs.raw,
                &norms.raw,
                &residual_norms.raw,
                &key_slot_scale.raw,
                &codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Fused mixed TurboQuant key scoring directly from regular/outlier
    /// compressed subspaces.
    #[allow(clippy::too_many_arguments)]
    // TODO(turboquant): staged for the TurboQuant KV-cache rollout
    // (see project_turboquant_impl.md). Not yet wired into attention path.
    #[allow(dead_code)]
    pub fn turboquant_mixed_score(
        regular_query_rot: &Self,
        regular_query_proj: &Self,
        regular_indices: &Self,
        regular_qjl_signs: &Self,
        regular_norms: &Self,
        regular_residual_norms: &Self,
        regular_slot_scale: &Self,
        regular_codebook: &Self,
        outlier_query_rot: &Self,
        outlier_query_proj: &Self,
        outlier_indices: &Self,
        outlier_qjl_signs: &Self,
        outlier_norms: &Self,
        outlier_residual_norms: &Self,
        outlier_slot_scale: &Self,
        outlier_codebook: &Self,
        regular_dim: u32,
        regular_qjl_words: u32,
        regular_n_centroids: u32,
        outlier_dim: u32,
        outlier_qjl_words: u32,
        outlier_n_centroids: u32,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_mixed_score(
                out.as_mut_ptr(),
                &regular_query_rot.raw,
                &regular_query_proj.raw,
                &regular_indices.raw,
                &regular_qjl_signs.raw,
                &regular_norms.raw,
                &regular_residual_norms.raw,
                &regular_slot_scale.raw,
                &regular_codebook.raw,
                &outlier_query_rot.raw,
                &outlier_query_proj.raw,
                &outlier_indices.raw,
                &outlier_qjl_signs.raw,
                &outlier_norms.raw,
                &outlier_residual_norms.raw,
                &outlier_slot_scale.raw,
                &outlier_codebook.raw,
                regular_dim,
                regular_qjl_words,
                regular_n_centroids,
                outlier_dim,
                outlier_qjl_words,
                outlier_n_centroids,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Phase F: XOR + popcount Hamming distance from a per-q-row query sign
    /// hash to a `[kv_rows, S, packed_dim]` cache of key sign hashes. Output:
    /// `[N, S]` u32 — Hamming distance per (q_row, slot), in range `[0, D]`.
    /// Pre-filter caller picks the M slots with the smallest distances and
    /// runs exact attention on those M only (sparse score path).
    ///
    /// - `query_signs`: `[N, packed_dim]` uint32                    (N = q_rows)
    /// - `key_signs`:   `[kv_rows, S, packed_dim]` uint32           (kv_rows = N / groups)
    /// - `groups`: `q_heads / kv_heads`. Pass `1` for non-GQA models.
    pub fn turboquant_hamming_distances(
        query_signs: &Self,
        key_signs: &Self,
        packed_dim: u32,
        n_rows: u32,
        n_seq: u32,
        groups: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_hamming_distances(
                out.as_mut_ptr(),
                &query_signs.raw,
                &key_signs.raw,
                packed_dim,
                n_rows,
                n_seq,
                groups,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Pack `sign(projected >= 0)` along the last dimension into uint32 words.
    ///
    /// - `projected`: `[N, D]` f32
    /// - Returns packed `[N, ceil(D/32)]` uint32 on success.
    pub fn turboquant_pack_sign_bits(
        projected: &Self,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_pack_sign_bits(
                out.as_mut_ptr(),
                &projected.raw,
                dim,
                packed_dim,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Pack q8 key bytes from centroid indices and packed QJL signs.
    ///
    /// - `indices`: `[N, D, S_cap]` uint8
    /// - `qjl_signs`: `[N, ceil(D/32), S_cap]` uint32
    /// - Returns `[N, D, S_cap]` uint8 where low 7 bits are the centroid index
    ///   and the high bit is the QJL sign.
    pub fn turboquant_pack_q8_keybytes(
        indices: &Self,
        qjl_signs: &Self,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_pack_q8_keybytes(
                out.as_mut_ptr(),
                &indices.raw,
                &qjl_signs.raw,
                dim,
                packed_dim,
                n_rows,
                cache_seq_capacity,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Pack q8 key bytes directly into a seq-major shadow layout.
    ///
    /// - `indices`: `[N, D, S_cap]` uint8
    /// - `qjl_signs`: `[N, ceil(D/32), S_cap]` uint32
    /// - Returns `[N, S_cap, D]` uint8 where low 7 bits are the centroid index
    ///   and the high bit is the QJL sign.
    pub fn turboquant_pack_q8_keybytes_seq(
        indices: &Self,
        qjl_signs: &Self,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_pack_q8_keybytes_seq(
                out.as_mut_ptr(),
                &indices.raw,
                &qjl_signs.raw,
                dim,
                packed_dim,
                n_rows,
                cache_seq_capacity,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Pack q8 key bytes and q8 value indices into one seq-major shadow.
    ///
    /// - `indices`: `[N, D, S_cap]` uint8
    /// - `qjl_signs`: `[N, ceil(D/32), S_cap]` uint32
    /// - `value_indices`: `[N, S_cap, D]` uint8
    /// - Returns `[N, S_cap, D]` uint16 where:
    ///   low byte = key byte (low 7 bits centroid index, high bit QJL sign)
    ///   high byte = value centroid index
    // TODO(turboquant): staged for TurboQuant KV-cache rollout
    // (see project_turboquant_impl.md).
    #[allow(dead_code)]
    pub fn turboquant_pack_q8_kvbytes_seq(
        indices: &Self,
        qjl_signs: &Self,
        value_indices: &Self,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_pack_q8_kvbytes_seq(
                out.as_mut_ptr(),
                &indices.raw,
                &qjl_signs.raw,
                &value_indices.raw,
                dim,
                packed_dim,
                n_rows,
                cache_seq_capacity,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Unpack uint32 sign words back into `{-1,+1}` float32 signs.
    ///
    /// - `packed`: `[N, ceil(D/32)]` uint32
    /// - Returns unpacked `[N, D]` f32 on success.
    pub fn turboquant_unpack_sign_bits(
        packed: &Self,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_unpack_sign_bits(
                out.as_mut_ptr(),
                &packed.raw,
                dim,
                packed_dim,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Signed, normalized FWHT-256 transform applied row-wise:
    /// `out[row] = left_signs * FWHT(right_signs * input[row]) / sqrt(256)`.
    ///
    /// - `input`: `[N, dim]` f32 (`dim` must be a power of two)
    /// - `left_signs`: `[dim]` f32 (Rademacher ±1)
    /// - `right_signs`: `[dim]` f32 (Rademacher ±1)
    /// - Returns `[N, dim]` f32 on success, with the unitary 1/sqrt(dim) scale baked in.
    pub fn turboquant_signed_fwht_pow2_rows(
        input: &Self,
        left_signs: &Self,
        right_signs: &Self,
        dim: u32,
        n_rows: u32,
    ) -> Option<Self> {
        if dim == 0 || (dim & (dim - 1)) != 0 {
            return None;
        }
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_signed_fwht_pow2_rows(
                out.as_mut_ptr(),
                &input.raw,
                &left_signs.raw,
                &right_signs.raw,
                dim,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Fused TurboQuant value aggregation in the rotated domain.
    ///
    /// Inputs:
    /// - `weights`: `[N, S]` f32
    /// - `indices`: `[N, D, S_cap]` uint8
    /// - `norms`: `[N, S]` f32
    /// - `codebook`: `[C]` f32
    ///
    /// Returns `output [N, D]` f32 on success.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_weighted_decode(
        weights: &Self,
        indices: &Self,
        norms: &Self,
        codebook: &Self,
        dim: u32,
        n_centroids: u32,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_weighted_decode(
                out.as_mut_ptr(),
                &weights.raw,
                &indices.raw,
                &norms.raw,
                &codebook.raw,
                dim,
                n_centroids,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=256/V=256.
    ///
    /// Returns the rotated aggregated values `[N, 256]` on success.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d256_2pass(
        query_rot: &Self,
        query_proj: &Self,
        key_indices: &Self,
        key_qjl_signs: &Self,
        key_norms: &Self,
        key_residual_norms: &Self,
        key_slot_scale: &Self,
        key_codebook: &Self,
        value_indices: &Self,
        value_norms: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &key_indices.raw,
                &key_qjl_signs.raw,
                &key_norms.raw,
                &key_residual_norms.raw,
                &key_slot_scale.raw,
                &key_codebook.raw,
                &value_indices.raw,
                &value_norms.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=256/V=256 over
    /// combined slot-major storage:
    /// - packed key bytes `[N, S_cap, D]`
    /// - value indices `[N, S_cap, D]`
    ///
    /// Returns the rotated aggregated values `[N, 256]` on success.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d256_packed_keys_2pass(
        query_rot: &Self,
        query_proj: &Self,
        key_bytes: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        value_indices: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_packed_keys_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &key_bytes.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                &value_indices.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=256/V=256 over
    /// a seq-major packed key shadow plus dense rotated values:
    /// - `key_bytes`: `[N, S_cap, D]` uint8
    /// - `value_dense`: `[N, S_cap, D]` bf16/f32 rotated dense values
    ///
    /// Returns the rotated aggregated values `[N, 256]` on success.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d256_packed_keys_dense_values_2pass(
        query_rot: &Self,
        query_proj: &Self,
        key_bytes: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        value_dense: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_packed_keys_dense_values_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &key_bytes.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                &value_dense.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 decode for D=256/V=256 over
    /// a seq-major pure-q8 key shadow plus dense rotated values:
    /// - `key_indices`: `[N, S_cap, D]` uint8, full 8-bit centroid index
    /// - `value_dense`: `[N, S_cap, D]` bf16/f32 rotated dense values
    ///
    /// Returns the rotated aggregated values `[N, 256]` on success.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
        query_rot: &Self,
        key_indices: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        value_dense: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &key_indices.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                &value_dense.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Full-byte D256 long-context pass-1 state output.
    /// Returns `(partials, sums, maxs)`.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d256_fullbyte_dense_values_2pass_state(
        query_rot: &Self,
        key_indices: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        value_dense: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<(Self, Self, Self)> {
        let mut partials = MaybeUninit::<RawBuf>::uninit();
        let mut sums = MaybeUninit::<RawBuf>::uninit();
        let mut maxs = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_state(
                partials.as_mut_ptr(),
                sums.as_mut_ptr(),
                maxs.as_mut_ptr(),
                &query_rot.raw,
                &key_indices.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                &value_dense.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some((
                Self {
                    raw: unsafe { partials.assume_init() },
                },
                Self {
                    raw: unsafe { sums.assume_init() },
                },
                Self {
                    raw: unsafe { maxs.assume_init() },
                },
            ))
        } else {
            None
        }
    }

    /// Full-byte D256 long-context pass-1 output only.
    /// Returns the unmerged partial outputs `[N, blocks, 256]`.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d256_fullbyte_dense_values_2pass_pass1(
        query_rot: &Self,
        key_indices: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        value_dense: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_pass1(
                out.as_mut_ptr(),
                &query_rot.raw,
                &key_indices.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                &value_dense.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Merge precomputed D256 2-pass partials/maxs/sums.
    pub fn turboquant_attention_q8_d256_pass2_merge(
        partials: &Self,
        sums: &Self,
        maxs: &Self,
        n_rows: u32,
        blocks: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_pass2_merge(
                out.as_mut_ptr(),
                &partials.raw,
                &sums.raw,
                &maxs.raw,
                n_rows,
                blocks,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Full-byte D256 long-context 2-pass variant with block-local 2-loop softmax.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d256_fullbyte_dense_values_2pass_localsoftmax(
        query_rot: &Self,
        key_indices: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        value_dense: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_localsoftmax(
                out.as_mut_ptr(),
                &query_rot.raw,
                &key_indices.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                &value_dense.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Full-byte D256 score-only long-context kernel.
    /// Returns scores `[N, S]`.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_score_q8_d256_fullbyte(
        query_rot: &Self,
        key_indices: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_score_q8_d256_fullbyte(
                out.as_mut_ptr(),
                &query_rot.raw,
                &key_indices.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// D256 dense-value weighted sum over resident rotated values.
    /// Returns rotated outputs `[N, 256]`.
    pub fn turboquant_weighted_sum_d256_dense_values(
        weights: &Self,
        value_dense: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_weighted_sum_d256_dense_values(
                out.as_mut_ptr(),
                &weights.raw,
                &value_dense.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=256/V=256 over
    /// a seq-major packed `{key,value}` shadow:
    /// - `kv_bytes`: `[N, S_cap, D]` uint16
    ///   low byte = key byte
    ///   high byte = value centroid index
    /// - `slot_scales`: `[N, S_cap, 4]` f32
    ///
    /// Returns the rotated aggregated values `[N, 256]` on success.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d256_packed_kv_2pass(
        query_rot: &Self,
        query_proj: &Self,
        kv_bytes: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_packed_kv_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &kv_bytes.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=256/V=256 over
    /// a seq-major packed key shadow plus dense rotated values:
    /// - `kv_bytes`: `[N, S_cap, D]` uint16, low byte = key byte
    /// - `value_dense`: `[N, S_cap, D]` bf16/f32 rotated dense values
    ///
    /// Returns the rotated aggregated values `[N, 256]` on success.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
        query_rot: &Self,
        query_proj: &Self,
        kv_bytes: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        value_dense: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &kv_bytes.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                &value_dense.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=128/V=128.
    ///
    /// Returns the rotated aggregated values `[N, 128]` on success.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d128_2pass(
        query_rot: &Self,
        query_proj: &Self,
        key_indices: &Self,
        key_qjl_signs: &Self,
        key_norms: &Self,
        key_residual_norms: &Self,
        key_slot_scale: &Self,
        key_codebook: &Self,
        value_indices: &Self,
        value_norms: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d128_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &key_indices.raw,
                &key_qjl_signs.raw,
                &key_norms.raw,
                &key_residual_norms.raw,
                &key_slot_scale.raw,
                &key_codebook.raw,
                &value_indices.raw,
                &value_norms.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Variant F (NoQjl) D=128/V=128 q8 2-pass attention. Codebook gets the
    /// full 8 bits, no QJL residual term — fewer inputs and fewer multiply-adds
    /// per slot than the Standard kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d128_no_qjl_2pass(
        query_rot: &Self,
        key_indices: &Self,
        key_norms: &Self,
        key_slot_scale: &Self,
        key_codebook: &Self,
        value_indices: &Self,
        value_norms: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d128_no_qjl_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &key_indices.raw,
                &key_norms.raw,
                &key_slot_scale.raw,
                &key_codebook.raw,
                &value_indices.raw,
                &value_norms.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Variant F (NoQjl) D=256/V=256 q8 2-pass attention. Same shape as the
    /// d128 no_qjl kernel: full 8-bit codebook for keys, no QJL residual,
    /// no `query_proj`. Pass 2 reuses the standard d256 merge kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d256_no_qjl_2pass(
        query_rot: &Self,
        key_indices: &Self,
        key_norms: &Self,
        key_slot_scale: &Self,
        key_codebook: &Self,
        value_indices: &Self,
        value_norms: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_no_qjl_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &key_indices.raw,
                &key_norms.raw,
                &key_slot_scale.raw,
                &key_codebook.raw,
                &value_indices.raw,
                &value_norms.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=128/V=128 over
    /// packed key bytes stored as `[N, D, S_cap]`.
    ///
    /// Returns the rotated aggregated values `[N, 128]` on success.
    #[allow(clippy::too_many_arguments)]
    pub fn turboquant_attention_q8_d128_packed_keys_2pass(
        query_rot: &Self,
        query_proj: &Self,
        key_bytes: &Self,
        key_norms: &Self,
        key_residual_norms: &Self,
        key_slot_scale: &Self,
        key_codebook: &Self,
        value_indices: &Self,
        value_norms: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d128_packed_keys_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &key_bytes.raw,
                &key_norms.raw,
                &key_residual_norms.raw,
                &key_slot_scale.raw,
                &key_codebook.raw,
                &value_indices.raw,
                &value_norms.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Gather selected coordinates from a `[N, D]` f32 tensor.
    // TODO(turboquant): staged for TurboQuant KV-cache rollout.
    #[allow(dead_code)]
    pub fn turboquant_gather_last_dim(
        input: &Self,
        positions: &Self,
        full_dim: u32,
        out_dim: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_gather_last_dim(
                out.as_mut_ptr(),
                &input.raw,
                &positions.raw,
                full_dim,
                out_dim,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Scatter regular/outlier component rows back into `[N, D]` f32 rows.
    #[allow(clippy::too_many_arguments)]
    // TODO(turboquant): staged for TurboQuant KV-cache rollout.
    #[allow(dead_code)]
    pub fn turboquant_scatter_last_dim(
        regular: &Self,
        outlier: &Self,
        regular_positions: &Self,
        outlier_positions: &Self,
        full_dim: u32,
        regular_dim: u32,
        outlier_dim: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_scatter_last_dim(
                out.as_mut_ptr(),
                &regular.raw,
                &outlier.raw,
                &regular_positions.raw,
                &outlier_positions.raw,
                full_dim,
                regular_dim,
                outlier_dim,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }
}
