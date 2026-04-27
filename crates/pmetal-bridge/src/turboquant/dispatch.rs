//! GPU-side TurboQuant kernel dispatch glue.
//!
//! Drives the MLX/Metal kernels that quantise + dequantise the K/V tensors
//! end-to-end on device. The Uniform path
//! ([`gpu_quantize_kv`] / [`gpu_dequantize_keys`] / [`gpu_dequantize_values`])
//! covers single-bit-width configs; the Mixed path
//! ([`gpu_quantize_kv_mixed`] / [`gpu_dequantize_keys_mixed`] /
//! [`gpu_dequantize_values_mixed`]) splits each row into a regular sub-vector
//! and an outlier sub-vector with independent codebooks.
//!
//! All functions in this module return `None` rather than panicking when the
//! GPU path is unavailable for a given config (e.g. unsupported head_dim,
//! missing codebook); callers fall back to the host encode/decode path in
//! [`super::encode`].

use crate::InlineArray;
use crate::compat::Dtype;

use super::bits::packed_qjl_words;
use super::config::{TurboQuantConfig, TurboQuantTensorConfig};
use super::core::TurboQuantCore;
use super::encode::decode_value_component_rows_raw;
use super::gpu_keystore::{GpuKeyStore, GpuMixedKeyStore, GpuMixedValueStore, GpuValueStore};
use super::math::inline_array_to_f32_vec;
use super::state::{TensorRuntime, TurboQuantState};
use super::{MAX_RESIDUAL_NORM, ZERO_EPSILON, dim_uses_fwht, turboquant_q8_fullbyte_enabled};


/// Quantise keys and values entirely on GPU.
///
/// Returns `None` if the GPU path is unavailable (e.g. missing codebook arr).
/// On success returns `(GpuKeyStore, GpuValueStore)` — both shapes `[B, H, S, *]`.
///
/// Algorithm:
///   1. Normalise onto unit sphere; keep per-vector L2 norm.
///   2. Rotate: x_norm @ rotation.T  → rotated   [B, H, S, D]
///   3. Nearest-centroid: argmin over squared diffs to codebook [C] → indices [B, H, S, D]
///   4. (Keys) Reconstruct MSE approx, compute residual, project via J, take sign.
pub(super) fn gpu_quantize_kv(
    state: &TurboQuantState,
    keys: &InlineArray,   // [B, H, S, Dk]  f32
    values: &InlineArray, // [B, H, S, Dv]  f32
    config: TurboQuantConfig,
) -> Option<(GpuKeyStore, GpuValueStore)> {
    let TurboQuantTensorConfig::Uniform { bits: key_bits } = config.keys else {
        return None;
    };
    let TurboQuantTensorConfig::Uniform { bits: val_bits } = config.values else {
        return None;
    };
    let key_mse_bits = key_bits.saturating_sub(1);

    let k_core = match &state.keys {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };
    let v_core = match &state.values {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };

    // ── Keys ─────────────────────────────────────────────────────────────
    // 1. L2 norm along D axis, keepdims → [B, H, S, 1]
    let key_norms = keys.norm_l2(-1, true);
    // 2. Normalise: x / max(norm, eps)
    let eps = InlineArray::from_f32(ZERO_EPSILON);
    let safe_norms_k = key_norms.maximum(&eps);
    let k_norm = keys.divide(&safe_norms_k);

    // 3. Rotate: k_norm @ rotation.T  (CPU: matmul_rows(rotation, dim, input) = input @ rotation.T)
    //    rotation.T == inverse_rotation, so we matmul with inverse_rotation_arr.
    let k_rot = k_core.rotate_array(&k_norm)?;

    // 4. GPU nearest-centroid → [B, H, S, D] uint32
    let k_indices = k_core.gpu_quantize_mse(&k_rot, key_mse_bits)?;

    // 5. Reconstruct MSE approximation in the rotated space.
    let k_mse_recon_rot = k_core.gpu_reconstruct_mse(&k_indices, key_mse_bits)?;

    // 6. Residual norms: rotation-invariant, so compute directly in rotated space.
    //    residual_rot = k_rot - k_mse_recon_rot  →  norm_l2  [B, H, S, 1]
    //    Defensive clip to [0, MAX_RESIDUAL_NORM]: IEEE fmax/fmin sanitize NaN
    //    to 0 (via maximum with 0) and cap Inf to the bound. Real residual
    //    norms sit well below this range.
    let k_residual_rot = k_rot.subtract(&k_mse_recon_rot);
    let k_residual_norms_raw = k_residual_rot.norm_l2(-1, true);
    let zero_bound = InlineArray::from_f32(0.0f32);
    let upper_bound = InlineArray::from_f32(MAX_RESIDUAL_NORM);
    let k_residual_norms = k_residual_norms_raw
        .maximum(&zero_bound)
        .minimum(&upper_bound);
    // QJL ablation: when the tq-ablation feature is enabled and the runtime
    // flag is set, zero the residual norms so the score kernel's residual
    // term collapses to 0 — measurement-only short-circuit, no kernel change.
    let k_residual_norms = if super::should_zero_qjl() {
        k_residual_norms.multiply(&zero_bound)
    } else {
        k_residual_norms
    };

    // 7. QJL: project the residual in the **unrotated** space.
    //    residual_unrot = k_mse_recon_rot @ rotation_arr  (inverse-rotate the rotated reconstruction)
    //    then: residual_unrot = k_norm - inv_rotate(k_mse_recon_rot)
    //    QJL: residual_unrot @ inverse_qjl_arr  (= residual @ qjl.T)
    let k_mse_recon_unrot = k_core.inverse_rotate_array(&k_mse_recon_rot)?;
    let k_residual_unrot = k_norm.subtract(&k_mse_recon_unrot);
    let k_qjl_proj = k_core.project_array(&k_residual_unrot)?;
    let qjl_shape = k_qjl_proj.shape();
    let qjl_ndim = qjl_shape.len();
    let qjl_rows: i32 = qjl_shape[..qjl_ndim - 1].iter().product();
    let packed_dim = packed_qjl_words(k_core.dim) as i32;
    let k_qjl_proj_2d = if qjl_ndim == 2 {
        k_qjl_proj.clone()
    } else {
        k_qjl_proj.reshape(&[qjl_rows, k_core.dim as i32])
    };
    let k_qjl_signs = InlineArray::turboquant_pack_sign_bits(
        &k_qjl_proj_2d,
        k_core.dim as u32,
        packed_dim as u32,
        qjl_rows as u32,
    )?;
    let k_qjl_signs = if qjl_ndim == 2 {
        k_qjl_signs
    } else {
        let mut packed_shape: Vec<i32> = qjl_shape[..qjl_ndim - 1].to_vec();
        packed_shape.push(packed_dim);
        k_qjl_signs.reshape(&packed_shape)
    };
    let use_q8_seq_shadow = key_bits == 8 && k_core.dim == 256 && v_core.dim == 256;
    let k_indices_t = (!use_q8_seq_shadow).then(|| k_indices.transpose_axes(&[0, 1, 3, 2]));
    let k_qjl_signs_t = (!use_q8_seq_shadow).then(|| k_qjl_signs.transpose_axes(&[0, 1, 3, 2]));
    let q8_pack_inputs = if key_bits == 8 {
        let kv_rows = (keys.dim(0) * keys.dim(1)) as u32;
        let seq = keys.dim(2) as u32;
        let indices_t_3d = if let Some(indices_t) = k_indices_t.as_ref() {
            indices_t.reshape(&[kv_rows as i32, k_core.dim as i32, seq as i32])
        } else {
            k_indices.transpose_axes(&[0, 1, 3, 2]).reshape(&[
                kv_rows as i32,
                k_core.dim as i32,
                seq as i32,
            ])
        };
        let qjl_signs_t_3d = if let Some(qjl_signs_t) = k_qjl_signs_t.as_ref() {
            qjl_signs_t.reshape(&[kv_rows as i32, packed_dim, seq as i32])
        } else {
            k_qjl_signs.transpose_axes(&[0, 1, 3, 2]).reshape(&[
                kv_rows as i32,
                packed_dim,
                seq as i32,
            ])
        };
        Some((kv_rows, seq, indices_t_3d, qjl_signs_t_3d))
    } else {
        None
    };
    let q8_keybytes_t = if use_q8_seq_shadow {
        None
    } else if let Some((kv_rows, seq, indices_t_3d, qjl_signs_t_3d)) = q8_pack_inputs.as_ref() {
        InlineArray::turboquant_pack_q8_keybytes(
            indices_t_3d,
            qjl_signs_t_3d,
            k_core.dim as u32,
            packed_dim as u32,
            *kv_rows,
            *seq,
        )
        .map(|packed| packed.reshape(&[keys.dim(0), keys.dim(1), k_core.dim as i32, keys.dim(2)]))
    } else {
        None
    };
    let q8_keybytes_seq = if use_q8_seq_shadow {
        q8_pack_inputs
            .as_ref()
            .and_then(|(kv_rows, seq, indices_t_3d, qjl_signs_t_3d)| {
                InlineArray::turboquant_pack_q8_keybytes_seq(
                    indices_t_3d,
                    qjl_signs_t_3d,
                    k_core.dim as u32,
                    packed_dim as u32,
                    *kv_rows,
                    *seq,
                )
                .map(|packed| {
                    packed.reshape(&[keys.dim(0), keys.dim(1), keys.dim(2), k_core.dim as i32])
                })
            })
    } else {
        None
    };
    let q8_fullbyte_seq = if use_q8_seq_shadow && turboquant_q8_fullbyte_enabled() {
        k_core
            .gpu_quantize_mse(&k_rot, 8)
            .map(|indices| indices.as_dtype(Dtype::Uint8.as_i32()))
    } else {
        None
    };

    // ── Values ────────────────────────────────────────────────────────────
    let (v_indices, v_indices_t, val_norms, d256_rot_values_seq) = if use_q8_seq_shadow {
        (
            None,
            None,
            None,
            Some(
                v_core
                    .rotate_array(values)?
                    .as_dtype(Dtype::Bfloat16.as_i32()),
            ),
        )
    } else {
        let val_norms = values.norm_l2(-1, true);
        let safe_norms_v = val_norms.maximum(&eps);
        let v_norm = values.divide(&safe_norms_v);

        // v_norm @ rotation.T = v_norm @ inverse_rotation_arr
        let v_rot = v_core.rotate_array(&v_norm)?;
        let v_indices = v_core.gpu_quantize_mse(&v_rot, val_bits)?;
        let v_indices_t = Some(v_indices.transpose_axes(&[0, 1, 3, 2]));
        (Some(v_indices), v_indices_t, Some(val_norms), None)
    };
    let q8_kvbytes_seq = None;
    let q8_slot_scales_seq = if use_q8_seq_shadow {
        let key_scales = key_norms.concatenate_2(&k_residual_norms, 3);
        let value_scales = InlineArray::ones(
            &[values.dim(0), values.dim(1), values.dim(2), 1],
            Dtype::Float32.as_i32(),
        );
        Some(key_scales.concatenate_2(&value_scales, 3))
    } else {
        None
    };

    Some((
        GpuKeyStore {
            indices: k_indices,
            indices_t: k_indices_t,
            q8_keybytes_t,
            q8_keybytes_seq,
            q8_fullbyte_seq,
            q8_kvbytes_seq,
            q8_slot_scales_seq,
            norms: (!use_q8_seq_shadow).then_some(key_norms),
            qjl_signs: k_qjl_signs,
            qjl_signs_t: k_qjl_signs_t,
            residual_norms: (!use_q8_seq_shadow).then_some(k_residual_norms),
        },
        GpuValueStore {
            indices: v_indices,
            indices_t: v_indices_t,
            norms: val_norms,
            d256_rot_values_seq,
        },
    ))
}

/// Encode a single-tensor sub-vector (regular *or* outlier slice) into the
/// rotated-MSE + QJL representation that the Mixed-precision attention
/// kernels will read. Mirrors the keys half of `gpu_quantize_kv` (lines
/// 2571-2630) but parameterised by an arbitrary `TurboQuantCore` so it
/// works for the regular_core (D_reg, dense rotation) and outlier_core
/// (D_out, FWHT rotation when D_out is pow2) without duplication.
pub(super) fn gpu_encode_key_subvector(
    rows: &InlineArray, // [B, H, S, sub_dim] f32
    core: &TurboQuantCore,
    key_bits: u8,
) -> Option<MixedKeySubvectorEncoding> {
    let mse_bits = key_bits.saturating_sub(1);
    let eps = InlineArray::from_f32(ZERO_EPSILON);

    let norms = rows.norm_l2(-1, true);
    let safe_norms = norms.maximum(&eps);
    let normalized = rows.divide(&safe_norms);

    let rotated = core.rotate_array(&normalized)?;
    let indices_u32 = core.gpu_quantize_mse(&rotated, mse_bits)?;
    // Indices kept as u32 for round-trip clarity. A u8 packed shadow (matching
    // Uniform's q8_keybytes) would let a fused Mixed score kernel skip the
    // u32→u8 cast each step; deferred until the fused path lands.
    let indices = indices_u32.clone();

    let recon_rot = core.gpu_reconstruct_mse(&indices_u32, mse_bits)?;
    let residual_rot = rotated.subtract(&recon_rot);
    let residual_norms_raw = residual_rot.norm_l2(-1, true);
    let zero_bound = InlineArray::from_f32(0.0f32);
    let upper_bound = InlineArray::from_f32(MAX_RESIDUAL_NORM);
    let residual_norms = residual_norms_raw
        .maximum(&zero_bound)
        .minimum(&upper_bound);
    // QJL ablation: see the `gpu_quantize_kv` site for rationale.
    let residual_norms = if super::should_zero_qjl() {
        residual_norms.multiply(&zero_bound)
    } else {
        residual_norms
    };

    let recon_unrot = core.inverse_rotate_array(&recon_rot)?;
    let residual_unrot = normalized.subtract(&recon_unrot);
    let qjl_proj = core.project_array(&residual_unrot)?;
    let qjl_shape = qjl_proj.shape();
    let qjl_ndim = qjl_shape.len();
    let qjl_rows: i32 = qjl_shape[..qjl_ndim - 1].iter().product();
    let packed_dim = packed_qjl_words(core.dim) as i32;
    let qjl_proj_2d = if qjl_ndim == 2 {
        qjl_proj.clone()
    } else {
        qjl_proj.reshape(&[qjl_rows, core.dim as i32])
    };
    let qjl_signs = InlineArray::turboquant_pack_sign_bits(
        &qjl_proj_2d,
        core.dim as u32,
        packed_dim as u32,
        qjl_rows as u32,
    )?;
    let qjl_signs = if qjl_ndim == 2 {
        qjl_signs
    } else {
        let mut packed_shape: Vec<i32> = qjl_shape[..qjl_ndim - 1].to_vec();
        packed_shape.push(packed_dim);
        qjl_signs.reshape(&packed_shape)
    };

    let indices_t = indices.transpose_axes(&[0, 1, 3, 2]);
    let qjl_signs_t = qjl_signs.transpose_axes(&[0, 1, 3, 2]);

    Some(MixedKeySubvectorEncoding {
        indices,
        indices_t: Some(indices_t),
        qjl_signs,
        qjl_signs_t: Some(qjl_signs_t),
        norms,
        residual_norms,
    })
}

pub(super) struct MixedKeySubvectorEncoding {
    pub(super) indices: InlineArray,
    pub(super) indices_t: Option<InlineArray>,
    pub(super) qjl_signs: InlineArray,
    pub(super) qjl_signs_t: Option<InlineArray>,
    pub(super) norms: InlineArray,
    pub(super) residual_norms: InlineArray,
}

/// Encode a single-tensor sub-vector for the *value* path (norm + rotated
/// MSE indices, no QJL residual term — mirrors the values half of
/// `gpu_quantize_kv` at lines 2701-2723).
pub(super) fn gpu_encode_value_subvector(
    rows: &InlineArray, // [B, H, S, sub_dim] f32
    core: &TurboQuantCore,
    val_bits: u8,
) -> Option<MixedValueSubvectorEncoding> {
    let eps = InlineArray::from_f32(ZERO_EPSILON);

    let norms = rows.norm_l2(-1, true);
    let safe_norms = norms.maximum(&eps);
    let normalized = rows.divide(&safe_norms);

    let rotated = core.rotate_array(&normalized)?;
    let indices_u32 = core.gpu_quantize_mse(&rotated, val_bits)?;
    let indices = indices_u32.clone();
    let indices_t = indices.transpose_axes(&[0, 1, 3, 2]);

    Some(MixedValueSubvectorEncoding {
        indices,
        indices_t: Some(indices_t),
        norms,
    })
}

pub(super) struct MixedValueSubvectorEncoding {
    pub(super) indices: InlineArray,
    pub(super) indices_t: Option<InlineArray>,
    pub(super) norms: InlineArray,
}

/// Compute the Mixed-precision outlier partition on the GPU.
///
/// Returns `(regular_src_dim, outlier_src_dim)` of shape `[B, H, S, D_reg]`
/// and `[B, H, S, D_out]` respectively, both `int32`. Each row's entries
/// hold the original-D positions of the corresponding sub-vector slot,
/// **sorted ascending**. The kernel reads these tables to scatter regular
/// and outlier contributions back into the `[B, H, D_total]` output.
///
/// Pipeline:
///   1. `argpartition(-|x|, outlier_count, axis=-1)` → `[B,H,S,D]` with the
///      first `outlier_count` entries being outlier positions (unsorted).
///   2. Slice into `[outlier_idxs_unsorted, regular_idxs_unsorted]`.
///   3. Stable-sort each subset ascending by re-running argsort and
///      `take_along_axis` (positions are unique ints, no tie-break needed).
pub(super) fn gpu_compute_outlier_partition(
    rows: &InlineArray, // [B, H, S, D_total] f32
    outlier_count: usize,
    total_dim: usize,
) -> Option<(InlineArray, InlineArray)> {
    if outlier_count == 0 || outlier_count >= total_dim {
        return None;
    }
    let abs = rows.abs();
    let neg_abs = abs.negative();
    let part = neg_abs.argpartition(outlier_count as i32, -1);

    let b = part.dim(0);
    let h = part.dim(1);
    let s = part.dim(2);
    let outlier_count_i32 = outlier_count as i32;
    let total_dim_i32 = total_dim as i32;

    let outlier_unsorted = part.slice(&[0, 0, 0, 0], &[b, h, s, outlier_count_i32]);
    let regular_unsorted = part.slice(&[0, 0, 0, outlier_count_i32], &[b, h, s, total_dim_i32]);

    let outlier_perm = outlier_unsorted.argsort(-1);
    let outlier_src_dim = outlier_unsorted.take_along_axis(&outlier_perm, -1);

    let regular_perm = regular_unsorted.argsort(-1);
    let regular_src_dim = regular_unsorted.take_along_axis(&regular_perm, -1);

    Some((regular_src_dim, outlier_src_dim))
}

/// Build a `GpuMixedKeyStore` for one append step from `[B, H, S, D]` f32
/// keys + values. Mirrors `gpu_quantize_kv` for the Mixed branch.
pub(super) fn gpu_quantize_kv_mixed(
    state: &TurboQuantState,
    keys: &InlineArray,   // [B, H, S, Dk] f32
    values: &InlineArray, // [B, H, S, Dv] f32
    config: TurboQuantConfig,
) -> Option<(GpuMixedKeyStore, GpuMixedValueStore)> {
    let TurboQuantTensorConfig::Mixed {
        regular_bits: kr_bits,
        outlier_bits: ko_bits,
        outlier_count: k_oc,
    } = config.keys
    else {
        return None;
    };
    let TurboQuantTensorConfig::Mixed {
        regular_bits: vr_bits,
        outlier_bits: vo_bits,
        outlier_count: v_oc,
    } = config.values
    else {
        return None;
    };

    let (k_reg_core, k_out_core) = match &state.keys {
        TensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => (regular_core, outlier_core),
        _ => return None,
    };
    let (v_reg_core, v_out_core) = match &state.values {
        TensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => (regular_core, outlier_core),
        _ => return None,
    };

    let k_total_dim = keys.dim(3) as usize;
    let v_total_dim = values.dim(3) as usize;

    // ── Keys ─────────────────────────────────────────────────────────────
    let (k_reg_src, k_out_src) = gpu_compute_outlier_partition(keys, k_oc, k_total_dim)?;
    let k_reg_rows = keys.take_along_axis(&k_reg_src, -1);
    let k_out_rows = keys.take_along_axis(&k_out_src, -1);

    let k_reg_enc = gpu_encode_key_subvector(&k_reg_rows, k_reg_core, kr_bits)?;
    let k_out_enc = gpu_encode_key_subvector(&k_out_rows, k_out_core, ko_bits)?;

    // Cast scatter tables to u8 for storage (D_total ≤ 256 fits comfortably).
    let k_reg_src_u8 = k_reg_src.as_dtype(Dtype::Uint8.as_i32());
    let k_out_src_u8 = k_out_src.as_dtype(Dtype::Uint8.as_i32());

    // ── Values ───────────────────────────────────────────────────────────
    let (v_reg_src, v_out_src) = gpu_compute_outlier_partition(values, v_oc, v_total_dim)?;
    let v_reg_rows = values.take_along_axis(&v_reg_src, -1);
    let v_out_rows = values.take_along_axis(&v_out_src, -1);

    let v_reg_enc = gpu_encode_value_subvector(&v_reg_rows, v_reg_core, vr_bits)?;
    let v_out_enc = gpu_encode_value_subvector(&v_out_rows, v_out_core, vo_bits)?;

    let v_reg_src_u8 = v_reg_src.as_dtype(Dtype::Uint8.as_i32());
    let v_out_src_u8 = v_out_src.as_dtype(Dtype::Uint8.as_i32());

    let mut kstore = GpuMixedKeyStore {
        regular_indices: k_reg_enc.indices,
        regular_indices_t: k_reg_enc.indices_t,
        regular_qjl_signs: k_reg_enc.qjl_signs,
        regular_qjl_signs_t: k_reg_enc.qjl_signs_t,
        regular_norms: k_reg_enc.norms,
        regular_residual_norms: k_reg_enc.residual_norms,
        regular_src_dim: k_reg_src_u8,
        outlier_indices: k_out_enc.indices,
        outlier_indices_t: k_out_enc.indices_t,
        outlier_qjl_signs: k_out_enc.qjl_signs,
        outlier_qjl_signs_t: k_out_enc.qjl_signs_t,
        outlier_norms: k_out_enc.norms,
        outlier_residual_norms: k_out_enc.residual_norms,
        outlier_src_dim: k_out_src_u8,
    };
    let mut vstore = GpuMixedValueStore {
        regular_indices: v_reg_enc.indices,
        regular_indices_t: v_reg_enc.indices_t,
        regular_norms: v_reg_enc.norms,
        regular_src_dim: v_reg_src_u8,
        outlier_indices: v_out_enc.indices,
        outlier_indices_t: v_out_enc.indices_t,
        outlier_norms: v_out_enc.norms,
        outlier_src_dim: v_out_src_u8,
    };

    // The Mixed encode chain branches at two non-deterministic-on-re-eval
    // points: (1) `argpartition` returns an unspecified permutation within
    // each bucket, so the gather (rows = take_along_axis(keys, src_dim)) and
    // the stored `src_dim` table need to commit to the *same* permutation;
    // (2) `gpu_quantize_mse`'s argmin is sensitive to MLX scheduler-driven
    // re-evals on the codebook-distance argmin under f32 noise. Both are
    // correct in isolation but compose into a hidden invariant — the stored
    // `indices` and the residual_norms derived from `recon = take(codebook,
    // indices)` must come from the same materialisation. A single eval+detach
    // here freezes the whole chain at one consistent point. ~1ms per encoded
    // chunk; matches the cost of the existing `eval_and_detach_gpu_state`
    // barrier on the Uniform path.
    let mut to_eval: Vec<&mut InlineArray> = Vec::new();
    kstore.collect_for_detach(&mut to_eval);
    vstore.collect_for_detach(&mut to_eval);
    crate::inline_array::eval_and_detach_many(&mut to_eval);
    Some((kstore, vstore))
}

/// Score `queries` against a Mixed-precision GPU key store using the
/// `mlx_inline_turboquant_mixed_score` kernel.
///
/// **Layout-oracle contract.** This helper validates the Mixed-precision
/// GPU storage layout against the C++ score kernel — a regression gate that
/// catches stride/dtype/qjl-word-count drift before it reaches a fused
/// attention path. It is **not** a production scoring path: the kernel takes
/// a single `[N, D_sub]` query slice per sub-vector and reuses it for every
/// cache slot, so the result is only correct when every slot's outlier mask
/// matches `kstore.regular_src_dim[..,0,..]` / `outlier_src_dim[..,0,..]`.
/// A production fused mixed-score kernel would gather Q per-slot from the
/// full `[N, D_total]` query instead.
///
/// Inputs:
///   - `queries`: `[B, q_heads, 1, D_total]` f32 — the un-rotated, un-projected
///     query (matches the shape produced by attention dispatch).
///   - `kstore`: GPU-resident encoded keys.
///   - `n_seq`: how many of the `T` cached slots to score against (≤ T).
///
/// Returns scores `[N, n_seq]` (N = B · q_heads).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_gpu_mixed_score(
    state: &TurboQuantState,
    config: &TurboQuantConfig,
    kstore: &GpuMixedKeyStore,
    queries: &InlineArray,
    q_heads: i32,
    kv_heads: i32,
    n_seq: i32,
    scale: f32,
) -> Option<InlineArray> {
    let (reg_core, out_core) = match &state.keys {
        TensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => (regular_core.as_ref(), outlier_core.as_ref()),
        _ => return None,
    };
    let TurboQuantTensorConfig::Mixed {
        regular_bits,
        outlier_bits,
        outlier_count: _,
    } = config.keys
    else {
        return None;
    };
    if q_heads <= 0 || kv_heads <= 0 || (q_heads % kv_heads) != 0 {
        return None;
    }

    let reg_codebook = reg_core.codebook_arr(regular_bits.saturating_sub(1))?;
    let out_codebook = out_core.codebook_arr(outlier_bits.saturating_sub(1))?;

    let b = queries.dim(0);
    let d_reg = kstore.regular_indices.dim(3);
    let d_out = kstore.outlier_indices.dim(3);
    let t = kstore.regular_indices.dim(2);
    if n_seq <= 0 || n_seq > t {
        return None;
    }
    let n_rows = b * q_heads;
    let kv_rows = b * kv_heads;
    let groups = q_heads / kv_heads;

    // Slice the slot-0 mask (shape [B, kv_heads, 1, D_*]) and broadcast across
    // q_heads via the GQA grouping. `repeat(groups, axis=1)` produces the
    // q_head-sized mask in the same kv_head→q_head order the kernel expects
    // (q_head = kv_head * groups + g).
    let reg_src_slot0 = kstore
        .regular_src_dim
        .slice(&[0, 0, 0, 0], &[b, kv_heads, 1, d_reg])
        .as_dtype(Dtype::Int32.as_i32());
    let out_src_slot0 = kstore
        .outlier_src_dim
        .slice(&[0, 0, 0, 0], &[b, kv_heads, 1, d_out])
        .as_dtype(Dtype::Int32.as_i32());
    let reg_src_q = reg_src_slot0.repeat(groups, 1);
    let out_src_q = out_src_slot0.repeat(groups, 1);

    let q_reg = queries.take_along_axis(&reg_src_q, -1);
    let q_out = queries.take_along_axis(&out_src_q, -1);

    let q_reg_2d = q_reg.reshape(&[n_rows, d_reg]);
    let q_out_2d = q_out.reshape(&[n_rows, d_out]);

    let q_reg_rot = reg_core.rotate_array(&q_reg_2d)?;
    let q_reg_proj = reg_core.project_array(&q_reg_2d)?;
    let q_out_rot = out_core.rotate_array(&q_out_2d)?;
    let q_out_proj = out_core.project_array(&q_out_2d)?;

    let reg_norms_flat = kstore.regular_norms.reshape(&[kv_rows, t]);
    let reg_residual_flat = kstore.regular_residual_norms.reshape(&[kv_rows, t]);
    let out_norms_flat = kstore.outlier_norms.reshape(&[kv_rows, t]);
    let out_residual_flat = kstore.outlier_residual_norms.reshape(&[kv_rows, t]);

    let reg_qjl_words = kstore.regular_qjl_signs.dim(3);
    let out_qjl_words = kstore.outlier_qjl_signs.dim(3);

    InlineArray::turboquant_mixed_score(
        &q_reg_rot,
        &q_reg_proj,
        &kstore.regular_indices,
        &kstore.regular_qjl_signs,
        &reg_norms_flat,
        &reg_residual_flat,
        reg_codebook,
        &q_out_rot,
        &q_out_proj,
        &kstore.outlier_indices,
        &kstore.outlier_qjl_signs,
        &out_norms_flat,
        &out_residual_flat,
        out_codebook,
        d_reg as u32,
        reg_qjl_words as u32,
        reg_codebook.dim(0) as u32,
        d_out as u32,
        out_qjl_words as u32,
        out_codebook.dim(0) as u32,
        n_rows as u32,
        n_seq as u32,
        t as u32,
        q_heads as u32,
        kv_heads as u32,
        scale,
    )
}

/// Dequantise GPU-stored keys back to `[B, H, T, Dk]` f32.
///
/// Formula (per coordinate):
///   k̃ = (codebook[idx] + (√(π/2)/D) · (J^T · sign) · residual_norm) · norm  [inv-rotated]
pub(super) fn gpu_dequantize_keys(
    store: &GpuKeyStore,
    runtime: &TensorRuntime,
    key_bits: u8,
) -> Option<InlineArray> {
    let key_mse_bits = key_bits.saturating_sub(1);
    let core = match runtime {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };

    // 1. Reconstruct MSE centroids in the rotated domain: take(codebook, indices) → [B,H,T,D].
    let mse_recon_rot = core.gpu_reconstruct_mse(&store.indices, key_mse_bits)?;

    // 2. Inverse-rotate back to input space.
    //    CPU: inverse_rotate_rows = matmul_rows(inverse_rotation, dim, input) = input @ inverse_rotation.T = input @ rotation.
    //    So GPU: recon_rot @ rotation_arr.
    let mse_base = core.inverse_rotate_array(&mse_recon_rot)?;

    // 3. QJL correction.
    //    CPU: inverse_project_rows(signs) = matmul_rows(inverse_qjl, dim, signs) = signs @ inverse_qjl.T = signs @ qjl.
    //    The GPU store keeps packed uint32 sign words, so unpack to {-1,+1}
    //    before the matmul with qjl_arr.
    let packed_shape = store.qjl_signs.shape();
    let packed_ndim = packed_shape.len();
    let packed_rows: i32 = packed_shape[..packed_ndim - 1].iter().product();
    let packed_words = packed_shape[packed_ndim - 1];
    let packed_signs = if packed_ndim == 2 {
        store.qjl_signs.clone()
    } else {
        store.qjl_signs.reshape(&[packed_rows, packed_words])
    };
    let unpacked_qjl_2d = InlineArray::turboquant_unpack_sign_bits(
        &packed_signs,
        core.dim as u32,
        packed_words as u32,
        packed_rows as u32,
    )?;
    let unpacked_qjl = if packed_ndim == 2 {
        unpacked_qjl_2d
    } else {
        let mut unpacked_shape: Vec<i32> = packed_shape[..packed_ndim - 1].to_vec();
        unpacked_shape.push(core.dim as i32);
        unpacked_qjl_2d.reshape(&unpacked_shape)
    };
    let qjl_correction = core.inverse_project_array(&unpacked_qjl)?;
    let dim = core.dim as f32;
    let qjl_scale_factor = InlineArray::from_f32((std::f32::consts::PI / 2.0).sqrt() / dim);
    // residual_norms: [B,H,T,1] keepdims — broadcast along D.
    let residual_norms = store.residual_norms_array()?;
    let scale = residual_norms.multiply(&qjl_scale_factor);
    let correction = qjl_correction.multiply(&scale);

    // 4. Base + QJL correction, rescale by original L2 norm.
    // norms: [B,H,T,1] keepdims — broadcast along D.
    let combined = mse_base.add(&correction);
    Some(combined.multiply(&store.key_norms_array()?))
}

/// Dequantise GPU-stored values back to `[B, H, T, Dv]` f32.
pub(super) fn gpu_dequantize_values(
    store: &GpuValueStore,
    runtime: &TensorRuntime,
    val_bits: u8,
) -> Option<InlineArray> {
    if let Some(d256_rot_values_seq) = store.d256_rot_values_seq.as_ref() {
        let core = match runtime {
            TensorRuntime::Uniform { core, .. } => core,
            TensorRuntime::Mixed { .. } => return None,
        };
        let dense_rot = d256_rot_values_seq.as_dtype(Dtype::Float32.as_i32());
        return core.inverse_rotate_array(&dense_rot);
    }

    let core = match runtime {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };

    if dim_uses_fwht(core.dim) {
        let indices_arr = store.indices.as_ref()?;
        let norms_arr = store.norms.as_ref()?;
        let shape = indices_arr.shape();
        let total = shape.iter().product::<i32>() as usize;
        let rows = shape[..shape.len() - 1].iter().product::<i32>() as usize;
        let indices: Vec<u16> = inline_array_to_f32_vec(indices_arr, total)?
            .into_iter()
            .map(|v| v as u16)
            .collect();
        let norms = inline_array_to_f32_vec(norms_arr, rows)?;
        let reconstructed = decode_value_component_rows_raw(core, &indices, &norms, val_bits);
        return Some(InlineArray::from_f32_slice(&reconstructed, shape));
    }

    // 1. Reconstruct MSE centroids in rotated space.
    let mse_recon_rot = core.gpu_reconstruct_mse(store.indices.as_ref()?, val_bits)?;

    // 2. Inverse-rotate: recon_rot @ rotation_arr (same derivation as keys).
    let mse_base = core.inverse_rotate_array(&mse_recon_rot)?;

    // 3. Rescale by stored L2 norms [B,H,T,1].
    Some(mse_base.multiply(store.norms.as_ref()?))
}

/// Dequantise a single key sub-vector (regular *or* outlier) for the
/// Mixed-precision path. Mirrors `gpu_dequantize_keys` body but
/// parameterised by an arbitrary `TurboQuantCore` and stored arrays.
pub(super) fn gpu_dequantize_key_subvector(
    indices_u8: &InlineArray,        // [B, H, T, D_sub] u8
    qjl_signs: &InlineArray,         // [B, H, T, ceil(D_sub/32)] u32
    norms: &InlineArray,             // [B, H, T, 1] f32
    residual_norms: &InlineArray,    // [B, H, T, 1] f32
    core: &TurboQuantCore,
    key_bits: u8,
) -> Option<InlineArray> {
    let mse_bits = key_bits.saturating_sub(1);

    // Phase 3a: indices stored as u32; cast no-op when already u32.
    let indices_u32 = indices_u8.as_dtype(Dtype::Uint32.as_i32());
    let mse_recon_rot = core.gpu_reconstruct_mse(&indices_u32, mse_bits)?;
    let mse_base = core.inverse_rotate_array(&mse_recon_rot)?;

    let packed_shape = qjl_signs.shape();
    let packed_ndim = packed_shape.len();
    let packed_rows: i32 = packed_shape[..packed_ndim - 1].iter().product();
    let packed_words = packed_shape[packed_ndim - 1];
    let packed_signs = if packed_ndim == 2 {
        qjl_signs.clone()
    } else {
        qjl_signs.reshape(&[packed_rows, packed_words])
    };
    let unpacked_2d = InlineArray::turboquant_unpack_sign_bits(
        &packed_signs,
        core.dim as u32,
        packed_words as u32,
        packed_rows as u32,
    )?;
    let unpacked = if packed_ndim == 2 {
        unpacked_2d
    } else {
        let mut shape: Vec<i32> = packed_shape[..packed_ndim - 1].to_vec();
        shape.push(core.dim as i32);
        unpacked_2d.reshape(&shape)
    };
    let qjl_correction = core.inverse_project_array(&unpacked)?;
    let dim_f = core.dim as f32;
    let qjl_scale_factor = InlineArray::from_f32((std::f32::consts::PI / 2.0).sqrt() / dim_f);
    let scale = residual_norms.multiply(&qjl_scale_factor);
    let correction = qjl_correction.multiply(&scale);
    let combined = mse_base.add(&correction);
    Some(combined.multiply(norms))
}

/// Dequantise a single value sub-vector for the Mixed-precision path.
pub(super) fn gpu_dequantize_value_subvector(
    indices_u8: &InlineArray,
    norms: &InlineArray,
    core: &TurboQuantCore,
    val_bits: u8,
) -> Option<InlineArray> {
    let indices_u32 = indices_u8.as_dtype(Dtype::Uint32.as_i32());
    let mse_recon_rot = core.gpu_reconstruct_mse(&indices_u32, val_bits)?;
    let mse_base = core.inverse_rotate_array(&mse_recon_rot)?;
    Some(mse_base.multiply(norms))
}

/// Dequantise a `GpuMixedKeyStore` back to `[B, H, T, D_total]` f32 by
/// dequantising each sub-vector and scattering through the per-row
/// scatter tables.
pub(super) fn gpu_dequantize_keys_mixed(
    store: &GpuMixedKeyStore,
    runtime: &TensorRuntime,
    config: &TurboQuantConfig,
) -> Option<InlineArray> {
    let TurboQuantTensorConfig::Mixed {
        regular_bits,
        outlier_bits,
        outlier_count: _,
    } = config.keys
    else {
        return None;
    };
    let (reg_core, out_core) = match runtime {
        TensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => (regular_core.as_ref(), outlier_core.as_ref()),
        _ => return None,
    };

    let regular_recon = gpu_dequantize_key_subvector(
        &store.regular_indices,
        &store.regular_qjl_signs,
        &store.regular_norms,
        &store.regular_residual_norms,
        reg_core,
        regular_bits,
    )?;
    let outlier_recon = gpu_dequantize_key_subvector(
        &store.outlier_indices,
        &store.outlier_qjl_signs,
        &store.outlier_norms,
        &store.outlier_residual_norms,
        out_core,
        outlier_bits,
    )?;

    let b = store.regular_indices.dim(0);
    let h = store.regular_indices.dim(1);
    let t = store.regular_indices.dim(2);
    let d_reg = store.regular_indices.dim(3);
    let d_out = store.outlier_indices.dim(3);
    let d_total = d_reg + d_out;

    let regular_src_i32 = store.regular_src_dim.as_dtype(Dtype::Int32.as_i32());
    let outlier_src_i32 = store.outlier_src_dim.as_dtype(Dtype::Int32.as_i32());

    let zero = InlineArray::zeros(&[b, h, t, d_total], Dtype::Float32.as_i32());
    let with_regular = zero.put_along_axis_op(&regular_src_i32, &regular_recon, -1);
    let merged = with_regular.put_along_axis_op(&outlier_src_i32, &outlier_recon, -1);
    Some(merged)
}

/// Dequantise a `GpuMixedValueStore` back to `[B, H, T, D_total]` f32.
pub(super) fn gpu_dequantize_values_mixed(
    store: &GpuMixedValueStore,
    runtime: &TensorRuntime,
    config: &TurboQuantConfig,
) -> Option<InlineArray> {
    let TurboQuantTensorConfig::Mixed {
        regular_bits,
        outlier_bits,
        outlier_count: _,
    } = config.values
    else {
        return None;
    };
    let (reg_core, out_core) = match runtime {
        TensorRuntime::Mixed {
            regular_core,
            outlier_core,
            ..
        } => (regular_core.as_ref(), outlier_core.as_ref()),
        _ => return None,
    };

    let regular_recon =
        gpu_dequantize_value_subvector(&store.regular_indices, &store.regular_norms, reg_core, regular_bits)?;
    let outlier_recon =
        gpu_dequantize_value_subvector(&store.outlier_indices, &store.outlier_norms, out_core, outlier_bits)?;

    let b = store.regular_indices.dim(0);
    let h = store.regular_indices.dim(1);
    let t = store.regular_indices.dim(2);
    let d_reg = store.regular_indices.dim(3);
    let d_out = store.outlier_indices.dim(3);
    let d_total = d_reg + d_out;

    let regular_src_i32 = store.regular_src_dim.as_dtype(Dtype::Int32.as_i32());
    let outlier_src_i32 = store.outlier_src_dim.as_dtype(Dtype::Int32.as_i32());

    let zero = InlineArray::zeros(&[b, h, t, d_total], Dtype::Float32.as_i32());
    let with_regular = zero.put_along_axis_op(&regular_src_i32, &regular_recon, -1);
    let merged = with_regular.put_along_axis_op(&outlier_src_i32, &outlier_recon, -1);
    Some(merged)
}

