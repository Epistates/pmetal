//! CPU-side encode/decode helpers for the TurboQuant K/V cache.
//!
//! These functions are the host-fallback path when no GPU kernel is available
//! for a (head_dim, config) combination. They produce/consume the same
//! [`PackedBits`](super::PackedBits) layout the score kernels read, so cold
//! reconstruction goes through the same Beta-codebook + QJL residual stack
//! as the GPU encode.
//!
//! The pure-CPU pipeline:
//!   1. Normalise rows to the unit sphere; record the L2 norm.
//!   2. Rotate (signed-FWHT or dense matmul) and quantise via Lloyd-Max.
//!   3. For keys: project the residual through a Gaussian J and pack signs.
//!   4. For Mixed configs: split each row by per-row top-K outlier mask.

use std::f32::consts::PI;

use super::config::TurboQuantTensorConfig;
use super::core::TurboQuantCore;
use super::math::l2_norm;
use super::state::TensorRuntime;
use super::host_keystore::{QuantizedKeyStore, QuantizedValueStore};
use super::bits::unpack_all;
use super::{MAX_RESIDUAL_NORM, ZERO_EPSILON};


pub(super) struct EncodedKeyRows {
    pub(super) mse_indices: Vec<u16>,
    pub(super) qjl_signs: Vec<u16>,
    pub(super) norms: Vec<f32>,
    pub(super) residual_norms: Vec<f32>,
    /// Per-row codebook scaling factor: `max(|rotated_row|) / centroid_max`.
    /// At decode time the codebook lookup is multiplied by this scalar before
    /// inverse-rotation, so a fixed Beta codebook adapts to each slot's
    /// rotated range. Length == norms.len() (one entry per row).
    pub(super) slot_scale: Vec<f32>,
}

pub(super) struct EncodedValueRows {
    pub(super) indices: Vec<u16>,
    pub(super) norms: Vec<f32>,
}

pub(super) struct BatchedKeyRows {
    pub(super) regular: EncodedKeyRows,
    pub(super) outlier_mask: Option<Vec<u16>>,
    pub(super) outlier: Option<EncodedKeyRows>,
}

pub(super) struct BatchedValueRows {
    pub(super) regular: EncodedValueRows,
    pub(super) outlier_mask: Option<Vec<u16>>,
    pub(super) outlier: Option<EncodedValueRows>,
}

pub(super) fn encode_key_rows(runtime: &TensorRuntime, total_dim: usize, rows: &[f32]) -> BatchedKeyRows {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            BatchedKeyRows {
                regular: encode_key_component_rows(core, rows, *bits),
                outlier_mask: None,
                outlier: None,
            }
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let (mask, regular_rows, outlier_rows) =
                split_rows_by_outliers(rows, total_dim, *outlier_count);
            BatchedKeyRows {
                regular: encode_key_component_rows(regular_core, &regular_rows, *regular_bits),
                outlier_mask: Some(mask),
                outlier: Some(encode_key_component_rows(
                    outlier_core,
                    &outlier_rows,
                    *outlier_bits,
                )),
            }
        }
    }
}

pub(super) fn encode_value_rows(runtime: &TensorRuntime, total_dim: usize, rows: &[f32]) -> BatchedValueRows {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            BatchedValueRows {
                regular: encode_value_component_rows(core, rows, *bits),
                outlier_mask: None,
                outlier: None,
            }
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let (mask, regular_rows, outlier_rows) =
                split_rows_by_outliers(rows, total_dim, *outlier_count);
            BatchedValueRows {
                regular: encode_value_component_rows(regular_core, &regular_rows, *regular_bits),
                outlier_mask: Some(mask),
                outlier: Some(encode_value_component_rows(
                    outlier_core,
                    &outlier_rows,
                    *outlier_bits,
                )),
            }
        }
    }
}

/// Two-stage key encoder: MSE at (bits-1) with per-row slot_scale + QJL on residual.
///
/// `slot_scale` adapts the fixed Beta codebook to each row's rotated range so
/// values closer to ±1 are quantized at full codebook resolution rather than
/// crowding the centre. Reconstruction multiplies the codebook lookup by
/// `slot_scale` before inverse-rotation; the score kernel multiplies by it
/// once per slot in the inner loop.
#[allow(clippy::needless_range_loop)]
pub(super) fn encode_key_component_rows(core: &TurboQuantCore, rows: &[f32], key_bits: u8) -> EncodedKeyRows {
    let num_rows = rows.len() / core.dim;
    let mut norms = vec![0.0f32; num_rows];
    let mut normalized = vec![0.0f32; rows.len()];

    // Step 1: Normalise onto unit sphere. Non-finite inputs (NaN/Inf) or
    // degenerate zero rows are zeroed out — the MSE quantizer's binary
    // search does not tolerate NaN.
    for (row_idx, row) in rows.chunks(core.dim).enumerate() {
        let norm = l2_norm(row);
        if !norm.is_finite() || norm <= ZERO_EPSILON {
            norms[row_idx] = 0.0;
            continue;
        }
        norms[row_idx] = norm;
        let dst = &mut normalized[row_idx * core.dim..(row_idx + 1) * core.dim];
        for (dst, &src) in dst.iter_mut().zip(row.iter()) {
            let n = src / norm;
            *dst = if n.is_finite() { n } else { 0.0 };
        }
    }

    let mse_bits = key_bits.saturating_sub(1);
    let mut mse_indices = vec![0u16; rows.len()];
    let mut slot_scale = vec![0.0f32; num_rows];
    let mut decoded_mse = vec![0.0f32; rows.len()];

    if mse_bits > 0 {
        // Step 2: Rotate once; per-row max gives the codebook adaptation factor.
        let rotated = core.rotate_rows(&normalized);
        let codebook = core.codebook(mse_bits);
        // Beta codebook is symmetric around 0; centroid_max is the rightmost
        // (positive) value. Floor at ZERO_EPSILON so degenerate rows don't
        // produce inf when we divide.
        let centroid_max = codebook
            .last()
            .copied()
            .unwrap_or(1.0)
            .abs()
            .max(ZERO_EPSILON);

        let mut decoded_rot = vec![0.0f32; rows.len()];
        for row_idx in 0..num_rows {
            let start = row_idx * core.dim;
            let end = start + core.dim;
            if norms[row_idx] <= ZERO_EPSILON {
                continue;
            }
            // Step 3: per-row max defines slot_scale. We divide rotated by
            // slot_scale before quantize so values use the full codebook range.
            let row_max = rotated[start..end]
                .iter()
                .fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let s = (row_max / centroid_max).max(ZERO_EPSILON);
            slot_scale[row_idx] = s;
            let inv_s = 1.0 / s;
            for i in start..end {
                let scaled = rotated[i] * inv_s;
                let idx = nearest_centroid_index(scaled, codebook);
                mse_indices[i] = idx as u16;
                // Decoded value in rotated domain: codebook[idx] * slot_scale.
                decoded_rot[i] = codebook[idx] * s;
            }
        }
        // Step 4: inverse-rotate the rescaled codebook recon to get decoded_mse.
        decoded_mse = core.inverse_rotate_rows(&decoded_rot);
    }

    // Step 5: residual = normalized - decoded_mse.
    let mut residual = vec![0.0f32; rows.len()];
    let mut residual_norms = vec![0.0f32; num_rows];
    for row_idx in 0..num_rows {
        let start = row_idx * core.dim;
        let end = start + core.dim;
        if norms[row_idx] <= ZERO_EPSILON {
            mse_indices[start..end].fill(0);
            continue;
        }
        let res_row = &mut residual[start..end];
        for ((dst, &lhs), &rhs) in res_row
            .iter_mut()
            .zip(normalized[start..end].iter())
            .zip(decoded_mse[start..end].iter())
        {
            *dst = lhs - rhs;
        }
        let raw_norm = l2_norm(res_row);
        residual_norms[row_idx] = if raw_norm.is_finite() {
            raw_norm.clamp(0.0, MAX_RESIDUAL_NORM)
        } else {
            0.0
        };
    }

    // Step 6: QJL — project residual and take signs.
    let projected = core.project_rows(&residual);
    let mut qjl_signs: Vec<u16> = projected
        .iter()
        .map(|&v| if v >= 0.0 { 1 } else { 0 })
        .collect();

    // Zero-vector rows get all-zero signs.
    for row_idx in 0..num_rows {
        if norms[row_idx] <= ZERO_EPSILON {
            let start = row_idx * core.dim;
            let end = start + core.dim;
            qjl_signs[start..end].fill(0);
        }
    }

    // QJL ablation: zero residual norms when the tq-ablation knob is on.
    // The decode path's QJL correction block is gated on
    // `residual_norms.iter().any(|rn| rn > ZERO_EPSILON)`, so all-zero
    // residual_norms make the entire QJL stage a no-op without touching
    // the decoder.
    if super::should_zero_qjl() {
        residual_norms.fill(0.0);
    }

    EncodedKeyRows {
        mse_indices,
        qjl_signs,
        norms,
        residual_norms,
        slot_scale,
    }
}

/// MSE-only value encoder.
#[allow(clippy::needless_range_loop)]
pub(super) fn encode_value_component_rows(
    core: &TurboQuantCore,
    rows: &[f32],
    value_bits: u8,
) -> EncodedValueRows {
    let num_rows = rows.len() / core.dim;
    let mut norms = vec![0.0f32; num_rows];
    let mut normalized = vec![0.0f32; rows.len()];

    for (row_idx, row) in rows.chunks(core.dim).enumerate() {
        let norm = l2_norm(row);
        if !norm.is_finite() || norm <= ZERO_EPSILON {
            norms[row_idx] = 0.0;
            continue;
        }
        norms[row_idx] = norm;
        let dst = &mut normalized[row_idx * core.dim..(row_idx + 1) * core.dim];
        for (dst, &src) in dst.iter_mut().zip(row.iter()) {
            let n = src / norm;
            *dst = if n.is_finite() { n } else { 0.0 };
        }
    }

    let mut indices = quantize_mse_rows(core, &normalized, value_bits);
    for row_idx in 0..num_rows {
        if norms[row_idx] <= ZERO_EPSILON {
            let start = row_idx * core.dim;
            let end = start + core.dim;
            indices[start..end].fill(0);
        }
    }

    EncodedValueRows { indices, norms }
}


pub(super) fn decode_key_rows(
    runtime: &TensorRuntime,
    total_dim: usize,
    store: &QuantizedKeyStore,
) -> Vec<f32> {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            decode_key_component_rows_raw(
                core,
                &unpack_all(&store.regular_indices),
                &unpack_all(&store.regular_qjl_signs),
                &store.regular_norms,
                &store.regular_residual_norms,
                &store.regular_slot_scale,
                *bits,
            )
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let regular = decode_key_component_rows_raw(
                regular_core,
                &unpack_all(&store.regular_indices),
                &unpack_all(&store.regular_qjl_signs),
                &store.regular_norms,
                &store.regular_residual_norms,
                &store.regular_slot_scale,
                *regular_bits,
            );
            let outlier = decode_key_component_rows_raw(
                outlier_core,
                &unpack_all(
                    store
                        .outlier_indices
                        .as_ref()
                        .expect("TurboQuant key outlier indices missing"),
                ),
                &unpack_all(
                    store
                        .outlier_qjl_signs
                        .as_ref()
                        .expect("TurboQuant key outlier QJL signs missing"),
                ),
                store
                    .outlier_norms
                    .as_ref()
                    .expect("TurboQuant key outlier norms missing"),
                store
                    .outlier_residual_norms
                    .as_ref()
                    .expect("TurboQuant key outlier residual_norms missing"),
                store
                    .outlier_slot_scale
                    .as_ref()
                    .expect("TurboQuant key outlier slot_scale missing"),
                *outlier_bits,
            );
            let mask = unpack_all(
                store
                    .outlier_mask
                    .as_ref()
                    .expect("TurboQuant key outlier mask missing"),
            );
            scatter_mixed_rows(&mask, total_dim, *outlier_count, &regular, &outlier)
        }
    }
}

pub(super) fn decode_value_rows(
    runtime: &TensorRuntime,
    total_dim: usize,
    store: &QuantizedValueStore,
) -> Vec<f32> {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            decode_value_component_rows_raw(
                core,
                &unpack_all(&store.regular_indices),
                &store.regular_norms,
                *bits,
            )
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let regular = decode_value_component_rows_raw(
                regular_core,
                &unpack_all(&store.regular_indices),
                &store.regular_norms,
                *regular_bits,
            );
            let outlier = decode_value_component_rows_raw(
                outlier_core,
                &unpack_all(
                    store
                        .outlier_indices
                        .as_ref()
                        .expect("TurboQuant value outlier indices missing"),
                ),
                store
                    .outlier_norms
                    .as_ref()
                    .expect("TurboQuant value outlier norms missing"),
                *outlier_bits,
            );
            let mask = unpack_all(
                store
                    .outlier_mask
                    .as_ref()
                    .expect("TurboQuant value outlier mask missing"),
            );
            scatter_mixed_rows(&mask, total_dim, *outlier_count, &regular, &outlier)
        }
    }
}

/// Reconstruct key rows from MSE indices + QJL signs + norms + slot_scale.
///
/// Formula (per row):
///   k̃ = Π^T · (codebook[idx] · slot_scale) · norm
///       + (√(π/2)/D) · Π^T · J^T · sign · residual_norm · norm
#[allow(clippy::needless_range_loop)]
pub(super) fn decode_key_component_rows_raw(
    core: &TurboQuantCore,
    indices: &[u16],
    qjl_signs: &[u16],
    norms: &[f32],
    residual_norms: &[f32],
    slot_scale: &[f32],
    key_bits: u8,
) -> Vec<f32> {
    let total_rows = norms.len();
    let mse_bits = key_bits.saturating_sub(1);

    // MSE base reconstruction: codebook[idx] * slot_scale, then inverse-rotate.
    let mut reconstructed = if mse_bits == 0 {
        vec![0.0; total_rows * core.dim]
    } else {
        let codebook = core.codebook(mse_bits);
        let mut decoded_rot = vec![0.0f32; total_rows * core.dim];
        for row_idx in 0..total_rows {
            let s = slot_scale[row_idx];
            let start = row_idx * core.dim;
            let end = start + core.dim;
            for i in start..end {
                decoded_rot[i] = codebook[usize::from(indices[i])] * s;
            }
        }
        core.inverse_rotate_rows(&decoded_rot)
    };

    // QJL correction term — only if any residual is non-trivial.
    if residual_norms.iter().any(|&rn| rn > ZERO_EPSILON) {
        let qjl_signs_f32: Vec<f32> = qjl_signs
            .iter()
            .map(|&v| if v == 0 { -1.0 } else { 1.0 })
            .collect();
        let qjl_correction = core.inverse_project_rows(&qjl_signs_f32);

        for row_idx in 0..total_rows {
            let residual_norm = residual_norms[row_idx];
            if residual_norm <= ZERO_EPSILON {
                continue;
            }
            let scale = ((PI / 2.0).sqrt() * residual_norm) / (core.dim as f32);
            let start = row_idx * core.dim;
            let end = start + core.dim;
            for (val, &correction) in reconstructed[start..end]
                .iter_mut()
                .zip(qjl_correction[start..end].iter())
            {
                *val += scale * correction;
            }
        }
    }

    // Rescale by stored norm.
    for row_idx in 0..total_rows {
        let start = row_idx * core.dim;
        let end = start + core.dim;
        let norm = norms[row_idx];
        if norm <= ZERO_EPSILON {
            reconstructed[start..end].fill(0.0);
        } else {
            for v in &mut reconstructed[start..end] {
                *v *= norm;
            }
        }
    }

    reconstructed
}

/// Reconstruct value rows from MSE indices + norms.
#[allow(clippy::needless_range_loop)]
pub(super) fn decode_value_component_rows_raw(
    core: &TurboQuantCore,
    indices: &[u16],
    norms: &[f32],
    value_bits: u8,
) -> Vec<f32> {
    let total_rows = norms.len();
    let mut reconstructed = reconstruct_mse_rows(core, indices, value_bits);

    for row_idx in 0..total_rows {
        let start = row_idx * core.dim;
        let end = start + core.dim;
        let norm = norms[row_idx];
        if norm <= ZERO_EPSILON {
            reconstructed[start..end].fill(0.0);
        } else {
            for v in &mut reconstructed[start..end] {
                *v *= norm;
            }
        }
    }

    reconstructed
}


/// Rotate then nearest-centroid quantise: returns a per-coordinate index.
pub(super) fn quantize_mse_rows(core: &TurboQuantCore, normalized: &[f32], bits: u8) -> Vec<u16> {
    if bits == 0 {
        return vec![0; normalized.len()];
    }
    let rotated = core.rotate_rows(normalized);
    let codebook = core.codebook(bits);
    rotated
        .iter()
        .map(|&v| nearest_centroid_index(v, codebook) as u16)
        .collect()
}

/// Look up centroids then inverse-rotate to reconstruct approximate vectors.
pub(super) fn reconstruct_mse_rows(core: &TurboQuantCore, indices: &[u16], bits: u8) -> Vec<f32> {
    if bits == 0 {
        return vec![0.0; indices.len()];
    }
    let codebook = core.codebook(bits);
    let rotated: Vec<f32> = indices.iter().map(|&i| codebook[usize::from(i)]).collect();
    core.inverse_rotate_rows(&rotated)
}

/// Binary search for the nearest centroid (codebook is sorted ascending).
pub(super) fn nearest_centroid_index(value: f32, codebook: &[f32]) -> usize {
    match codebook.binary_search_by(|probe| probe.partial_cmp(&value).unwrap()) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) if i >= codebook.len() => codebook.len() - 1,
        Err(i) => {
            let left = codebook[i - 1];
            let right = codebook[i];
            if (value - left).abs() <= (right - value).abs() {
                i - 1
            } else {
                i
            }
        }
    }
}


/// Identify the top-k highest-magnitude coordinates as outliers.
pub(super) fn select_outlier_mask(row: &[f32], outlier_count: usize) -> Vec<u16> {
    let mut ranked: Vec<usize> = (0..row.len()).collect();
    ranked.sort_unstable_by(|&lhs, &rhs| {
        row[rhs]
            .abs()
            .total_cmp(&row[lhs].abs())
            .then_with(|| lhs.cmp(&rhs))
    });
    let mut mask = vec![0u16; row.len()];
    for dim_idx in ranked.into_iter().take(outlier_count) {
        mask[dim_idx] = 1;
    }
    mask
}

/// Partition rows into regular and outlier sub-vectors.
///
/// Returns `(outlier_mask, regular_rows, outlier_rows)`.
pub(super) fn split_rows_by_outliers(
    rows: &[f32],
    total_dim: usize,
    outlier_count: usize,
) -> (Vec<u16>, Vec<f32>, Vec<f32>) {
    let num_rows = rows.len() / total_dim;
    let regular_dim = total_dim - outlier_count;
    let mut mask_all = Vec::with_capacity(rows.len());
    let mut regular_rows = Vec::with_capacity(num_rows * regular_dim);
    let mut outlier_rows = Vec::with_capacity(num_rows * outlier_count);

    for row in rows.chunks(total_dim) {
        let mask = select_outlier_mask(row, outlier_count);
        for (&v, &is_outlier) in row.iter().zip(mask.iter()) {
            if is_outlier == 1 {
                outlier_rows.push(v);
            } else {
                regular_rows.push(v);
            }
        }
        mask_all.extend_from_slice(&mask);
    }

    (mask_all, regular_rows, outlier_rows)
}

/// Re-interleave regular and outlier sub-vectors using the stored mask.
pub(super) fn scatter_mixed_rows(
    outlier_mask: &[u16],
    total_dim: usize,
    outlier_count: usize,
    regular_rows: &[f32],
    outlier_rows: &[f32],
) -> Vec<f32> {
    let num_rows = outlier_mask.len() / total_dim;
    let regular_dim = total_dim - outlier_count;
    let mut merged = vec![0.0f32; outlier_mask.len()];

    for row_idx in 0..num_rows {
        let mask_row = &outlier_mask[row_idx * total_dim..(row_idx + 1) * total_dim];
        let mut reg_cur = 0usize;
        let mut out_cur = 0usize;
        for dim_idx in 0..total_dim {
            let dst = &mut merged[row_idx * total_dim + dim_idx];
            if mask_row[dim_idx] == 1 {
                *dst = outlier_rows[row_idx * outlier_count + out_cur];
                out_cur += 1;
            } else {
                *dst = regular_rows[row_idx * regular_dim + reg_cur];
                reg_cur += 1;
            }
        }
    }

    merged
}
