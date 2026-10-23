//! Quantization of `f32` tensors to GGUF tensor types.
//!
//! [`quantize_tensor`] is the entry point: it takes a tensor's values in
//! row-major order together with the length of its rows and, optionally, an
//! importance vector, and returns the bytes of the quantized tensor.
//!
//! # Rows, blocks and importance
//!
//! Quantized formats store whole blocks (32, 64, 128 or 256 weights) and a
//! block never straddles a row, so `n_per_row` must be a multiple of the
//! type's block size. The optional importance vector has one weight per input
//! channel (per column of the weight matrix), typically the mean squared
//! activation from a calibration run (see [`crate::imatrix::IMatrix`]). Where
//! a format supports it the quantizer minimises the *importance-weighted*
//! reconstruction error instead of the plain one, which spends the format's
//! few levels where the activations are large. A tensor made of several
//! slices stacked along its outer dimension (the experts of a mixture-of-
//! experts layer) takes one importance vector per slice, concatenated.
//!
//! # Algorithms
//!
//! The block layouts are those of the ggml reference and the algorithms its
//! reference quantizers use: per-sub-block scale (and minimum) search by
//! weighted least squares over a grid of candidate scales, a joint
//! re-quantization of the sub-block scales for the K-quants, and the
//! non-linear nearest-value search for IQ4. The output is deterministic and
//! independent of the number of threads (rows and blocks are independent and
//! processed in parallel).
//!
//! `x * (1/d)` rather than `x / d` and round-half-to-even `nearest_int` are
//! used exactly where the reference uses them, so legacy formats match its
//! output bit for bit.

// The quantizers follow the reference algorithms index for index; rewriting their
// loops as iterator chains over several parallel arrays would obscure that.
#![allow(clippy::needless_range_loop)]

use crate::fp4::{best_index_mxfp4, e8m0_to_fp32_half, fp32_to_ue4m3, ue4m3_to_fp32};
use crate::k_quants as kq;
use crate::types::GgmlType;
use half::f16;
use pmetal_core::{PMetalError, Result};
use rayon::prelude::*;

/// Smallest magnitude treated as non-zero when scaling a group.
pub(crate) const GROUP_MAX_EPS: f32 = 1e-15;

/// Blocks a parallel work item handles at minimum.
const BLOCKS_PER_TASK: usize = 64;

/// Signature of a per-row quantizer: `x` holds whole blocks, `qw` the
/// importance of the same columns, `sigma2` the row-wide mean square of the
/// *whole* row (some formats weight by it), and `out` the bytes for `x`.
type RowFn = fn(x: &[f32], qw: Option<&[f32]>, sigma2: f32, out: &mut [u8]);

/// Quantize a float slice to the specified GGUF type.
///
/// The slice is treated as one row, so its length must be a multiple of the
/// type's block size. Use [`quantize_tensor`] to quantize a matrix with an
/// importance vector.
pub fn quantize(data: &[f32], dtype: GgmlType) -> Result<Vec<u8>> {
    quantize_tensor(data, dtype, data.len().max(1), None)
}

/// Whether [`quantize_tensor`] can produce `dtype`.
pub fn can_quantize(dtype: GgmlType) -> bool {
    matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16)
        || row_quantizer(dtype).is_some()
}

/// The type to store rows of `n_per_row` values in when `dtype` was wanted.
///
/// A block never straddles a row, so a type whose block size does not divide
/// the row cannot hold the tensor. The reference falls back to a 32-element
/// type of similar size (Q2_K, Q3_K, IQ4_XS and the IQ1/2/3 types to IQ4_NL,
/// Q4_K to Q5_0, Q5_K to Q5_1, Q6_K and Q8_K to Q8_0, the ternary types to
/// Q4_0), and to F16 when even 32 does not divide the row. Returns `dtype`
/// itself when it fits.
pub fn fit_to_row_length(dtype: GgmlType, n_per_row: usize) -> GgmlType {
    if n_per_row % dtype.block_size() == 0 {
        return dtype;
    }
    let fallback = match dtype {
        GgmlType::Tq1_0 | GgmlType::Tq2_0 => GgmlType::Q4_0,
        GgmlType::Iq2Xxs
        | GgmlType::Iq2Xs
        | GgmlType::Iq2S
        | GgmlType::Iq3Xxs
        | GgmlType::Iq3S
        | GgmlType::Iq1S
        | GgmlType::Iq1M
        | GgmlType::Q2K
        | GgmlType::Q3K
        | GgmlType::Iq4Xs => GgmlType::Iq4Nl,
        GgmlType::Q4K => GgmlType::Q5_0,
        GgmlType::Q5K => GgmlType::Q5_1,
        GgmlType::Q6K | GgmlType::Q8K => GgmlType::Q8_0,
        _ => GgmlType::F16,
    };
    if n_per_row % fallback.block_size() == 0 {
        fallback
    } else {
        GgmlType::F16
    }
}

/// Whether `dtype` uses the importance vector. The others ignore it.
pub fn uses_importance(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Iq4Nl
            | GgmlType::Iq4Xs
    )
}

/// Quantize a tensor given as `data` (row-major) with rows of `n_per_row`
/// elements.
///
/// `importance`, when given, holds one non-negative weight per input channel:
/// `n_per_row` values, or a multiple of that for a tensor of equally sized
/// slices stacked along its outer dimension (one vector per slice, in order).
/// Types that cannot use it ignore it ([`uses_importance`]).
///
/// # Errors
///
/// - the type has no quantizer (the repacked layouts and the integer types);
/// - `data.len()` is not a multiple of `n_per_row`, or `n_per_row` is not a
///   multiple of the type's block size;
/// - the importance vector has an incompatible length.
pub fn quantize_tensor(
    data: &[f32],
    dtype: GgmlType,
    n_per_row: usize,
    importance: Option<&[f32]>,
) -> Result<Vec<u8>> {
    match dtype {
        GgmlType::F32 => return Ok(data.iter().flat_map(|f| f.to_le_bytes()).collect()),
        GgmlType::F16 => {
            return Ok(data
                .iter()
                .flat_map(|&f| f16::from_f32(f).to_le_bytes())
                .collect());
        }
        GgmlType::Bf16 => {
            return Ok(data
                .iter()
                .flat_map(|&f| half::bf16::from_f32(f).to_le_bytes())
                .collect());
        }
        _ => {}
    }
    let (block_size, type_size, quantize_row) = row_quantizer(dtype).ok_or_else(|| {
        PMetalError::Quantization(format!("Quantization to {dtype:?} is not supported"))
    })?;

    if data.is_empty() {
        return Ok(Vec::new());
    }
    if n_per_row == 0 || data.len() % n_per_row != 0 {
        return Err(PMetalError::Quantization(format!(
            "{} values do not divide into rows of {n_per_row}",
            data.len()
        )));
    }
    if n_per_row % block_size != 0 {
        return Err(PMetalError::Quantization(format!(
            "rows of {n_per_row} values are not a whole number of {dtype:?} blocks ({block_size} values)"
        )));
    }
    let n_rows = data.len() / n_per_row;

    // Which slice of the importance vector a row uses.
    let (slices, rows_per_slice) = match importance {
        Some(_) if !uses_importance(dtype) => (0, 0),
        Some(w) => {
            if w.is_empty() || w.len() % n_per_row != 0 {
                return Err(PMetalError::Quantization(format!(
                    "importance has {} values, not a multiple of the row length {n_per_row}",
                    w.len()
                )));
            }
            let slices = w.len() / n_per_row;
            if n_rows % slices != 0 {
                return Err(PMetalError::Quantization(format!(
                    "{n_rows} rows do not divide into the {slices} importance slices"
                )));
            }
            (slices, n_rows / slices)
        }
        None => (0, 0),
    };
    let importance = importance.filter(|_| slices > 0);
    if let Some(w) = importance {
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(PMetalError::Quantization(
                "importance weights must be finite and non-negative".into(),
            ));
        }
    }

    let blocks_per_row = n_per_row / block_size;
    let row_bytes = blocks_per_row * type_size;
    let task_blocks = blocks_per_row.clamp(1, BLOCKS_PER_TASK);
    let mut out = vec![0u8; n_rows * row_bytes];

    out.par_chunks_mut(row_bytes)
        .zip(data.par_chunks(n_per_row))
        .enumerate()
        .for_each(|(row, (out_row, x_row))| {
            let qw_row = importance.map(|w| {
                let slice = row / rows_per_slice;
                &w[slice * n_per_row..(slice + 1) * n_per_row]
            });
            let sigma2 = if qw_row.is_some() {
                x_row.iter().map(|v| v * v).sum::<f32>() / n_per_row as f32
            } else {
                0.0
            };
            out_row
                .par_chunks_mut(task_blocks * type_size)
                .zip(x_row.par_chunks(task_blocks * block_size))
                .enumerate()
                .for_each(|(task, (o, x))| {
                    let qw = qw_row.map(|w| {
                        let start = task * task_blocks * block_size;
                        &w[start..start + x.len()]
                    });
                    quantize_row(x, qw, sigma2, o);
                });
        });
    Ok(out)
}

/// `(block size in elements, bytes per block, per-row function)`.
fn row_quantizer(dtype: GgmlType) -> Option<(usize, usize, RowFn)> {
    let f: RowFn = match dtype {
        GgmlType::Q4_0 => quantize_q4_0,
        GgmlType::Q4_1 => quantize_q4_1,
        GgmlType::Q5_0 => quantize_q5_0,
        GgmlType::Q5_1 => quantize_q5_1,
        GgmlType::Q8_0 => quantize_q8_0,
        GgmlType::Q8_1 => quantize_q8_1,
        GgmlType::Q1_0 => quantize_q1_0,
        GgmlType::Tq1_0 => quantize_tq1_0,
        GgmlType::Tq2_0 => quantize_tq2_0,
        GgmlType::Mxfp4 => quantize_mxfp4,
        GgmlType::Nvfp4 => quantize_nvfp4,
        GgmlType::Q2K => kq::quantize_q2_k,
        GgmlType::Q3K => kq::quantize_q3_k,
        GgmlType::Q4K => kq::quantize_q4_k,
        GgmlType::Q5K => kq::quantize_q5_k,
        GgmlType::Q6K => kq::quantize_q6_k,
        GgmlType::Q8K => kq::quantize_q8_k,
        GgmlType::Iq4Nl => crate::iq_quants::quantize_iq4_nl,
        GgmlType::Iq4Xs => crate::iq_quants::quantize_iq4_xs,
        _ => return None,
    };
    Some((dtype.block_size(), dtype.type_size(), f))
}

// =============================================================================
// Shared numeric helpers (the reference quantizers' building blocks)
// =============================================================================

/// Round to nearest, ties to even (the reference's `nearest_int`).
#[inline(always)]
pub(crate) fn nearest_int(v: f32) -> i32 {
    v.round_ties_even() as i32
}

/// Round half away from zero (C's `roundf`).
#[inline(always)]
fn roundf(v: f32) -> i32 {
    v.round() as i32
}

/// Scale for a symmetric quantization with levels `-nmax..nmax-1` (stored
/// biased by `nmax` in `l`), found by weighted least squares over a small
/// neighbourhood of candidate scales.
///
/// `rmse_type` selects the weights when `qw` is `None`: 1 is `x^2`, 2 is
/// uniform, 3 is `|x|`, anything else `sqrt(|x|)`; a negative value returns
/// early with a blend of the two scale estimates. 0 skips the search.
pub(crate) fn make_qx_quants(
    nmax: i32,
    x: &[f32],
    l: &mut [i8],
    rmse_type: i32,
    qw: Option<&[f32]>,
) -> f32 {
    let n = x.len();
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l[..n].fill(0);
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let q = nearest_int(iscale * x[i]);
            l[i] = (nmax + q.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let (rmse_type, return_early) = if rmse_type < 0 {
        (-rmse_type, true)
    } else {
        (rmse_type, false)
    };
    let weight = |i: usize| -> f32 {
        if let Some(qw) = qw {
            qw[i]
        } else {
            match rmse_type {
                1 => x[i] * x[i],
                2 => 1.0,
                3 => x[i].abs(),
                _ => x[i].abs().sqrt(),
            }
        }
    };
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = (q + nmax) as i8;
        let w = weight(i);
        sumlx += w * x[i] * q as f32;
        suml2 += w * (q * q) as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            let w = weight(i);
            sumlx += w * x[i] * q as f32;
            suml2 += w * (q * q) as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let q = nearest_int(iscale * x[i]);
                l[i] = (nmax + q.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// Scale for Q3_K's 3-bit sub-blocks (`l` is stored biased by `nmax`).
pub(crate) fn make_q3_quants(nmax: i32, x: &[f32], l: &mut [i8], do_rmse: bool) -> f32 {
    let n = x.len();
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l[..n].fill(0);
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    if do_rmse {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0.0f32;
        for i in 0..n {
            let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            l[i] = q as i8;
            let w = x[i] * x[i];
            sumlx += w * x[i] * q as f32;
            suml2 += w * (q * q) as f32;
        }
        for _ in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let li = i32::from(l[i]);
                let mut slx = sumlx - w * x[i] * li as f32;
                if slx > 0.0 {
                    let mut sl2 = suml2 - w * (li * li) as f32;
                    let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                    if new_l != li {
                        slx += w * x[i] * new_l as f32;
                        sl2 += w * (new_l * new_l) as f32;
                        if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            l[i] = new_l as i8;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        for v in l[..n].iter_mut() {
            *v += nmax as i8;
        }
        return sumlx / suml2;
    }
    for i in 0..n {
        let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = (q + nmax) as i8;
    }
    1.0 / iscale
}

/// How [`make_qkx2_quants`] searches for the best scale: `nstep + 1`
/// candidates `(rmin + rdelta * k + nmax) / range`, scored by squared error
/// or, with `use_mad`, absolute error.
#[derive(Clone, Copy)]
pub(crate) struct ScaleSearch {
    pub rmin: f32,
    pub rdelta: f32,
    pub nstep: i32,
    pub use_mad: bool,
}

impl ScaleSearch {
    /// The finer search the reference uses when an importance vector is given.
    pub const IMPORTANCE: Self = Self {
        rmin: -0.9,
        rdelta: 0.05,
        nstep: 36,
        use_mad: false,
    };
}

/// Scale and minimum for an unsigned `0..=nmax` quantization with an offset:
/// `x ~ scale * l - min_out`. Searches `nstep + 1` candidate scales from
/// `rmin` in steps of `rdelta` (relative to `nmax`) and keeps the one with the
/// least weighted error (squared, or absolute with `use_mad`).
///
/// Returns `(scale, min_out)`; `l` receives the quantized values and `laux`
/// is scratch of the same length.
pub(crate) fn make_qkx2_quants(
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    laux: &mut [u8],
    search: ScaleSearch,
) -> (f32, f32) {
    let ScaleSearch {
        rmin,
        rdelta,
        nstep,
        use_mad,
    } = search;
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        l[..n].fill(0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let err = |diff: f32| if use_mad { diff.abs() } else { diff * diff };
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = q as u8;
        best_mad += weights[i] * err(scale * q as f32 + min - x[i]);
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = q as u8;
            let w = weights[i];
            sum_l += w * q as f32;
            sum_l2 += w * (q * q) as f32;
            sum_xl += w * q as f32 * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                mad += weights[i] * err(this_scale * f32::from(laux[i]) + this_min - x[i]);
            }
            if mad < best_mad {
                l[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// Quantize non-negative values `x` (the sub-block scales or minimums of a
/// K-quant super-block) to `0..=nmax` with a shared scale, weighting each
/// by `quant_weights`; returns the scale.
pub(crate) fn make_qp_quants(nmax: i32, x: &[f32], l: &mut [u8], quant_weights: &[f32]) -> f32 {
    let n = x.len();
    let max = x.iter().fold(0.0f32, |m, &v| m.max(v));
    if max == 0.0 {
        l[..n].fill(0);
        return 0.0;
    }
    let mut iscale = nmax as f32 / max;
    for i in 0..n {
        l[i] = nearest_int(iscale * x[i]) as u8;
    }
    let mut scale = 1.0 / iscale;
    let mut best_mse = 0.0f32;
    for i in 0..n {
        let diff = x[i] - scale * f32::from(l[i]);
        best_mse += quant_weights[i] * diff * diff;
    }
    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0.0f32;
        for i in 0..n {
            let q = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = x[i] - scale_is * q as f32;
            mse += quant_weights[i] * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * x[i]).min(nmax);
        l[i] = q as u8;
        let w = quant_weights[i];
        sumlx += w * x[i] * q as f32;
        suml2 += w * (q * q) as f32;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = quant_weights[i];
            let li = i32::from(l[i]);
            let mut slx = sumlx - w * x[i] * li as f32;
            let mut sl2 = suml2 - w * (li * li) as f32;
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx).min(nmax);
                if new_l != li {
                    slx += w * x[i] * new_l as f32;
                    sl2 += w * (new_l * new_l) as f32;
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    scale = if suml2 > 0.0 { sumlx / suml2 } else { scale };
    scale
}

/// Index of the value in the ascending table `values` nearest to `x`.
pub(crate) fn best_index_int8(values: &[i8], x: f32) -> usize {
    let n = values.len();
    if x <= f32::from(values[0]) {
        return 0;
    }
    if x >= f32::from(values[n - 1]) {
        return n - 1;
    }
    let (mut ml, mut mu) = (0, n - 1);
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < f32::from(values[mav]) {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if x - f32::from(values[mu - 1]) < f32::from(values[mu]) - x {
        mu - 1
    } else {
        mu
    }
}

#[inline]
fn put_f16(out: &mut [u8], at: usize, v: f32) {
    out[at..at + 2].copy_from_slice(&f16::from_f32(v).to_le_bytes());
}

/// The weights an importance-aware legacy quantizer minimises: the column's
/// importance times `sqrt(sigma2 + x^2)`.
#[inline]
fn weighted(qw: &[f32], x: &[f32], sigma2: f32, weight: &mut [f32]) {
    for ((w, &q), &v) in weight.iter_mut().zip(qw).zip(x) {
        *w = q * (sigma2 + v * v).sqrt();
    }
}

// =============================================================================
// Legacy 32-element formats
// =============================================================================

/// Q4_0: `d: f16`, 16 nibble bytes.
fn quantize_q4_0(x: &[f32], qw: Option<&[f32]>, sigma2: f32, out: &mut [u8]) {
    for (b, (xb, ob)) in x
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<18>().0.iter_mut())
        .enumerate()
    {
        if let Some(qw) = qw {
            let mut weight = [0.0f32; 32];
            weighted(&qw[32 * b..32 * b + 32], xb, sigma2, &mut weight);
            let mut l = [0i8; 32];
            let d = make_qx_quants(8, xb, &mut l, 1, Some(&weight));
            put_f16(ob, 0, d);
            for j in 0..16 {
                ob[2 + j] = (l[j] as u8) | ((l[j + 16] as u8) << 4);
            }
            continue;
        }
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in xb {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        put_f16(ob, 0, d);
        for j in 0..16 {
            let x0 = xb[j] * id;
            let x1 = xb[j + 16] * id;
            let xi0 = ((x0 + 8.5) as i32).clamp(0, 15) as u8;
            let xi1 = ((x1 + 8.5) as i32).clamp(0, 15) as u8;
            ob[2 + j] = xi0 | (xi1 << 4);
        }
    }
}

/// Q4_1: `d: f16`, `m: f16`, 16 nibble bytes.
fn quantize_q4_1(x: &[f32], qw: Option<&[f32]>, sigma2: f32, out: &mut [u8]) {
    for (b, (xb, ob)) in x
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<20>().0.iter_mut())
        .enumerate()
    {
        if let Some(qw) = qw {
            let mut weight = [0.0f32; 32];
            weighted(&qw[32 * b..32 * b + 32], xb, sigma2, &mut weight);
            let (mut l, mut laux) = ([0u8; 32], [0u8; 32]);
            let (d, m) =
                make_qkx2_quants(15, xb, &weight, &mut l, &mut laux, ScaleSearch::IMPORTANCE);
            put_f16(ob, 0, d);
            put_f16(ob, 2, -m);
            for j in 0..16 {
                ob[4 + j] = l[j] | (l[j + 16] << 4);
            }
            continue;
        }
        let min = xb.iter().fold(f32::MAX, |m, &v| m.min(v));
        let max = xb.iter().fold(f32::MIN, |m, &v| m.max(v));
        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        put_f16(ob, 0, d);
        put_f16(ob, 2, min);
        for j in 0..16 {
            let x0 = (xb[j] - min) * id;
            let x1 = (xb[j + 16] - min) * id;
            let xi0 = ((x0 + 0.5) as i32).clamp(0, 15) as u8;
            let xi1 = ((x1 + 0.5) as i32).clamp(0, 15) as u8;
            ob[4 + j] = xi0 | (xi1 << 4);
        }
    }
}

/// Pack 32 five-bit values: the low nibbles into 16 bytes (element `j` low,
/// `j + 16` high) and the fifth bits into a `u32`.
#[inline]
fn pack_q5(q: &[u8; 32], qh_out: &mut [u8], qs_out: &mut [u8]) {
    let mut qh = 0u32;
    for j in 0..16 {
        qs_out[j] = (q[j] & 0x0f) | ((q[j + 16] & 0x0f) << 4);
        qh |= u32::from((q[j] & 0x10) >> 4) << j;
        qh |= u32::from((q[j + 16] & 0x10) >> 4) << (j + 16);
    }
    qh_out.copy_from_slice(&qh.to_le_bytes());
}

/// Q5_0: `d: f16`, `qh: u32`, 16 nibble bytes.
fn quantize_q5_0(x: &[f32], qw: Option<&[f32]>, sigma2: f32, out: &mut [u8]) {
    for (b, (xb, ob)) in x
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<22>().0.iter_mut())
        .enumerate()
    {
        let mut q = [0u8; 32];
        if let Some(qw) = qw {
            let mut weight = [0.0f32; 32];
            weighted(&qw[32 * b..32 * b + 32], xb, sigma2, &mut weight);
            let mut l = [0i8; 32];
            let d = make_qx_quants(16, xb, &mut l, 1, Some(&weight));
            put_f16(ob, 0, d);
            for j in 0..32 {
                q[j] = l[j] as u8;
            }
        } else {
            let mut amax = 0.0f32;
            let mut max = 0.0f32;
            for &v in xb {
                if amax < v.abs() {
                    amax = v.abs();
                    max = v;
                }
            }
            let d = max / -16.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            put_f16(ob, 0, d);
            for j in 0..32 {
                q[j] = ((xb[j] * id + 16.5) as i32).clamp(0, 31) as u8;
            }
        }
        let (head, rest) = ob.split_at_mut(6);
        pack_q5(&q, &mut head[2..6], rest);
    }
}

/// Q5_1: `d: f16`, `m: f16`, `qh: u32`, 16 nibble bytes.
fn quantize_q5_1(x: &[f32], qw: Option<&[f32]>, sigma2: f32, out: &mut [u8]) {
    for (b, (xb, ob)) in x
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<24>().0.iter_mut())
        .enumerate()
    {
        let mut q = [0u8; 32];
        if let Some(qw) = qw {
            let mut weight = [0.0f32; 32];
            weighted(&qw[32 * b..32 * b + 32], xb, sigma2, &mut weight);
            let mut laux = [0u8; 32];
            let (d, m) =
                make_qkx2_quants(31, xb, &weight, &mut q, &mut laux, ScaleSearch::IMPORTANCE);
            put_f16(ob, 0, d);
            put_f16(ob, 2, -m);
        } else {
            let min = xb.iter().fold(f32::MAX, |m, &v| m.min(v));
            let max = xb.iter().fold(f32::MIN, |m, &v| m.max(v));
            let d = (max - min) / 31.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            put_f16(ob, 0, d);
            put_f16(ob, 2, min);
            for j in 0..32 {
                q[j] = (((xb[j] - min) * id + 0.5) as i32).clamp(0, 31) as u8;
            }
        }
        let (head, rest) = ob.split_at_mut(8);
        pack_q5(&q, &mut head[4..8], rest);
    }
}

/// Q8_0: `d: f16`, 32 signed bytes.
fn quantize_q8_0(x: &[f32], _qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (xb, ob) in x
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<34>().0.iter_mut())
    {
        let amax = xb.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        put_f16(ob, 0, d);
        for j in 0..32 {
            ob[2 + j] = roundf(xb[j] * id).clamp(-127, 127) as i8 as u8;
        }
    }
}

/// Q8_1: `d: f16`, `s: f16` (`d` times the sum of the quants), 32 signed
/// bytes.
fn quantize_q8_1(x: &[f32], _qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (xb, ob) in x
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<36>().0.iter_mut())
    {
        let amax = xb.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        put_f16(ob, 0, d);
        let mut sum = 0i32;
        for j in 0..32 {
            let q = roundf(xb[j] * id).clamp(-127, 127);
            ob[4 + j] = q as i8 as u8;
            sum += q;
        }
        put_f16(ob, 2, sum as f32 * d);
    }
}

// =============================================================================
// 1-bit, ternary and 4-bit float formats
// =============================================================================

/// Q1_0: `d: f16` (mean magnitude), 128 sign bits.
fn quantize_q1_0(x: &[f32], _qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (xb, ob) in x
        .as_chunks::<128>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<18>().0.iter_mut())
    {
        let sum_abs: f32 = xb.iter().filter(|v| v.is_finite()).map(|v| v.abs()).sum();
        put_f16(ob, 0, sum_abs / 128.0);
        ob[2..].fill(0);
        for (i, &v) in xb.iter().enumerate() {
            if v >= 0.0 {
                ob[2 + i / 8] |= 1 << (i % 8);
            }
        }
    }
}

#[inline]
fn amax_finite(x: &[f32]) -> f32 {
    x.iter()
        .filter(|v| v.is_finite())
        .fold(0.0f32, |m, v| m.max(v.abs()))
}

/// A ternary digit in `0..=2` for `v * id` rounded to `-1, 0, 1`.
#[inline]
fn trit(v: f32, id: f32) -> u8 {
    (roundf(v * id).clamp(-1, 1) + 1) as u8
}

/// TQ1_0: five trits per byte in `qs[48]`, four per byte in `qh[4]`, `d`.
fn quantize_tq1_0(x: &[f32], _qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (xb, ob) in x
        .as_chunks::<256>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<54>().0.iter_mut())
    {
        let d = amax_finite(xb);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut pos = 0;
        let mut j = 0;
        // 32-byte groups, then the 16-byte remainder of qs.
        for (width, groups) in [(32usize, 1usize), (16, 1)] {
            for _ in 0..groups {
                for m in 0..width {
                    let mut q = 0u16;
                    for n in 0..5 {
                        q = q * 3 + u16::from(trit(xb[pos + m + n * width], id));
                    }
                    ob[j + m] = (q * 256).div_ceil(243) as u8;
                }
                pos += 5 * width;
                j += width;
            }
        }
        for m in 0..4 {
            let mut q = 0u16;
            for k in 0..4 {
                q = q * 3 + u16::from(trit(xb[pos + m + k * 4], id));
            }
            q *= 3;
            ob[48 + m] = (q * 256).div_ceil(243) as u8;
        }
        put_f16(ob, 52, d);
    }
}

/// TQ2_0: four 2-bit trits per byte in `qs[64]`, `d`.
fn quantize_tq2_0(x: &[f32], _qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (xb, ob) in x
        .as_chunks::<256>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<66>().0.iter_mut())
    {
        let d = amax_finite(xb);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for j in 0..2 {
            for m in 0..32 {
                let mut q = 0u8;
                for n in 0..4 {
                    q |= (trit(xb[128 * j + m + n * 32], id) & 3) << (2 * n);
                }
                ob[32 * j + m] = q;
            }
        }
        put_f16(ob, 64, d);
    }
}

/// MXFP4: one E8M0 exponent, 16 bytes of E2M1 nibbles.
fn quantize_mxfp4(x: &[f32], _qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (xb, ob) in x
        .as_chunks::<32>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<17>().0.iter_mut())
    {
        let amax = amax_finite(xb);
        let e = if amax > 0.0 {
            (amax.log2().floor() - 2.0 + 127.0).clamp(0.0, 255.0) as u8
        } else {
            0
        };
        let d = e8m0_to_fp32_half(e);
        ob[0] = e;
        for j in 0..16 {
            let q0 = best_index_mxfp4(xb[j], d);
            let q1 = best_index_mxfp4(xb[j + 16], d);
            ob[1 + j] = q0 | (q1 << 4);
        }
    }
}

/// NVFP4: four UE4M3 scales (one per 16 values), 32 bytes of E2M1 nibbles.
fn quantize_nvfp4(x: &[f32], _qw: Option<&[f32]>, _sigma2: f32, out: &mut [u8]) {
    for (xb, ob) in x
        .as_chunks::<64>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<36>().0.iter_mut())
    {
        for s in 0..4 {
            let sub = &xb[16 * s..16 * s + 16];
            let ue = fp32_to_ue4m3(amax_finite(sub) / 6.0);
            let d = ue4m3_to_fp32(ue);
            ob[s] = ue;
            for j in 0..8 {
                let q0 = best_index_mxfp4(sub[j], d);
                let q1 = best_index_mxfp4(sub[j + 8], d);
                ob[4 + 8 * s + j] = q0 | (q1 << 4);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::dequantize;

    /// The deterministic test signal of the reference quantization tests:
    /// `0.1 + 2 cos(i)`.
    fn reference_signal(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.1 + 2.0 * (i as f32).cos()).collect()
    }

    /// Error measure of the reference quantization tests:
    /// `sqrt(sum (a - b)^2) / n`.
    fn total_error(a: &[f32], b: &[f32]) -> f32 {
        let sum: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
        (sum.sqrt() / a.len() as f64) as f32
    }

    fn round_trip_error(data: &[f32], dtype: GgmlType) -> f32 {
        let bytes = quantize(data, dtype).unwrap();
        assert_eq!(bytes.len(), dtype.tensor_size(data.len()));
        let back = dequantize(&bytes, dtype, &[data.len() as i32]).unwrap();
        total_error(data, &back)
    }

    /// The error bounds the reference implementation's own quantization tests
    /// hold each format to on this signal (4096 values).
    #[test]
    fn round_trip_error_is_within_the_reference_bound_for_every_format() {
        let data = reference_signal(4096);
        let bounds: &[(GgmlType, f32)] = &[
            (GgmlType::Q4_0, 0.002),
            (GgmlType::Q4_1, 0.002),
            (GgmlType::Q5_0, 0.002),
            (GgmlType::Q5_1, 0.002),
            (GgmlType::Q8_0, 0.002),
            (GgmlType::Q8_1, 0.002),
            (GgmlType::Q2K, 0.0075),
            (GgmlType::Q3K, 0.0040),
            (GgmlType::Q4K, 0.002),
            (GgmlType::Q5K, 0.002),
            (GgmlType::Q6K, 0.002),
            (GgmlType::Q8K, 0.002),
            (GgmlType::Iq4Nl, 0.002),
            (GgmlType::Iq4Xs, 0.002),
            (GgmlType::Tq1_0, 0.01),
            (GgmlType::Tq2_0, 0.01),
            (GgmlType::Mxfp4, 0.0030),
            (GgmlType::Nvfp4, 0.0030),
        ];
        for &(dtype, bound) in bounds {
            let err = round_trip_error(&data, dtype);
            assert!(err < bound, "{dtype:?}: error {err} exceeds {bound}");
        }
    }

    /// A weight matrix as trained ones look: rows of groups of 16 whose scales
    /// vary by a factor of a few, and about one value in 200 an outlier. The
    /// scale search of a K-quant has something to do on this and none of it
    /// is a cosine.
    fn heavy_tailed(rows: usize, cols: usize) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut uniform = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut gauss = move || {
            let (u, v) = (uniform(), uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        };
        let mut out = Vec::with_capacity(rows * cols);
        for _ in 0..rows {
            let mut group_scale = 0.0;
            for c in 0..cols {
                if c % 16 == 0 {
                    group_scale = (0.7 * gauss()).exp();
                }
                let mut x = 0.02 * group_scale * gauss();
                if gauss().abs() > 2.8 {
                    x *= 6.0;
                }
                out.push(x as f32);
            }
        }
        out
    }

    /// The K-quants find as small an error as the reference quantizer does.
    ///
    /// The reference values are the relative RMS error (`|x - x'| / |x|`) of
    /// this exact tensor quantized by the reference quantizer without an
    /// importance matrix. They agree to about a thousandth because the scale
    /// searches are the same computation; a weaker search (one that settles for
    /// the first scale it tries, say) is worse by percents, and a file made
    /// with it is worse under the same name.
    #[test]
    fn k_quants_lose_no_more_than_the_reference_quantizer_on_a_heavy_tailed_matrix() {
        let (rows, cols) = (256, 1024);
        let data = heavy_tailed(rows, cols);
        let energy: f64 = data.iter().map(|v| f64::from(*v).powi(2)).sum();
        for (dtype, reference) in [
            (GgmlType::Q2K, 0.325_816),
            (GgmlType::Q3K, 0.172_322),
            (GgmlType::Q4K, 0.085_718),
            (GgmlType::Q5K, 0.043_839),
            (GgmlType::Q6K, 0.021_481),
        ] {
            let bytes = quantize_tensor(&data, dtype, cols, None).unwrap();
            let back = dequantize(&bytes, dtype, &[rows as i32, cols as i32]).unwrap();
            let error: f64 = data
                .iter()
                .zip(&back)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum();
            let relative = (error / energy).sqrt();
            // Not worse (a rounding that falls the other way in a handful of
            // blocks can make it a hair better), and not so different that the
            // search is another one.
            assert!(
                relative <= reference * 1.001 && relative >= reference * 0.99,
                "{dtype:?}: relative error {relative:.6} against the reference's {reference}"
            );
        }
    }

    #[test]
    fn bits_buy_accuracy() {
        // Within a family, more bits must not be worse.
        let data = reference_signal(4096);
        let e = |t| round_trip_error(&data, t);
        assert!(e(GgmlType::Q8_0) < e(GgmlType::Q6K));
        assert!(e(GgmlType::Q6K) < e(GgmlType::Q5K));
        assert!(e(GgmlType::Q5K) < e(GgmlType::Q4K));
        assert!(e(GgmlType::Q4K) < e(GgmlType::Q3K));
        assert!(e(GgmlType::Q3K) < e(GgmlType::Q2K));
        assert!(e(GgmlType::Q5_0) < e(GgmlType::Q4_0));
        assert!(e(GgmlType::Q5_1) < e(GgmlType::Q4_1));
    }

    #[test]
    fn unaligned_lengths_are_errors_not_padding() {
        assert!(quantize(&[1.0; 33], GgmlType::Q4_0).is_err());
        assert!(quantize(&[1.0; 255], GgmlType::Q4K).is_err());
        // Rows must be whole blocks even when the total is.
        assert!(quantize_tensor(&[1.0; 128], GgmlType::Q4_0, 48, None).is_err());
        assert!(quantize_tensor(&[1.0; 96], GgmlType::Q4_0, 48, None).is_err());
        assert!(quantize_tensor(&[1.0; 128], GgmlType::Q4_0, 64, None).is_ok());
        assert_eq!(quantize(&[], GgmlType::Q4_0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn incompatible_row_lengths_fall_back_like_the_reference() {
        // A row that fits keeps its type.
        assert_eq!(fit_to_row_length(GgmlType::Q4K, 4096), GgmlType::Q4K);
        // 4,160 = 130 x 32 but not a multiple of 256.
        assert_eq!(fit_to_row_length(GgmlType::Q4K, 4160), GgmlType::Q5_0);
        assert_eq!(fit_to_row_length(GgmlType::Q5K, 4160), GgmlType::Q5_1);
        assert_eq!(fit_to_row_length(GgmlType::Q6K, 4160), GgmlType::Q8_0);
        assert_eq!(fit_to_row_length(GgmlType::Q3K, 4160), GgmlType::Iq4Nl);
        assert_eq!(fit_to_row_length(GgmlType::Iq2Xxs, 4160), GgmlType::Iq4Nl);
        assert_eq!(fit_to_row_length(GgmlType::Tq2_0, 4160), GgmlType::Q4_0);
        // Not even 32 divides it.
        assert_eq!(fit_to_row_length(GgmlType::Q4K, 100), GgmlType::F16);
        assert_eq!(fit_to_row_length(GgmlType::Q8_0, 100), GgmlType::F16);
        // The fallback is always something the rows can be stored in.
        for t in [GgmlType::Q4K, GgmlType::Q2K, GgmlType::Q6K, GgmlType::Mxfp4] {
            for n in [32usize, 96, 100, 4160] {
                let f = fit_to_row_length(t, n);
                assert_eq!(n % f.block_size(), 0, "{t:?} {n} -> {f:?}");
                assert!(can_quantize(f));
            }
        }
    }

    #[test]
    fn unsupported_types_are_errors() {
        for t in [
            GgmlType::Iq2Xxs,
            GgmlType::Iq1S,
            GgmlType::Q4_0_4_4,
            GgmlType::I8,
        ] {
            assert!(quantize(&[0.0; 256], t).is_err(), "{t:?}");
            assert!(!can_quantize(t), "{t:?}");
        }
        assert!(can_quantize(GgmlType::Q4K));
        assert!(can_quantize(GgmlType::F16));
    }

    #[test]
    fn all_zero_and_constant_blocks_are_stable() {
        for dtype in [
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::Q2K,
            GgmlType::Q3K,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Q8K,
            GgmlType::Iq4Nl,
            GgmlType::Iq4Xs,
            GgmlType::Tq2_0,
            GgmlType::Mxfp4,
            GgmlType::Nvfp4,
        ] {
            let n = 512;
            for value in [0.0f32, 1.5, -1.5] {
                let data = vec![value; n];
                let bytes = quantize(&data, dtype).unwrap();
                let back = dequantize(&bytes, dtype, &[n as i32]).unwrap();
                assert!(back.iter().all(|v| v.is_finite()), "{dtype:?} {value}");
                if value == 0.0 {
                    assert!(back.iter().all(|v| *v == 0.0), "{dtype:?} zeros");
                }
            }
        }
    }

    #[test]
    fn output_is_deterministic_and_independent_of_threading() {
        let data = reference_signal(256 * 64);
        for dtype in [
            GgmlType::Q4K,
            GgmlType::Q6K,
            GgmlType::Iq4Xs,
            GgmlType::Q4_0,
        ] {
            let a = quantize_tensor(&data, dtype, 256 * 4, None).unwrap();
            let b = quantize_tensor(&data, dtype, 256 * 4, None).unwrap();
            assert_eq!(a, b, "{dtype:?}");
            // One row quantized whole equals the same data split into rows:
            // blocks are independent.
            let c = quantize_tensor(&data, dtype, 256, None).unwrap();
            assert_eq!(a, c, "{dtype:?}");
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let single = pool.install(|| quantize(&data, GgmlType::Q5K).unwrap());
        assert_eq!(single, quantize(&data, GgmlType::Q5K).unwrap());
    }

    #[test]
    fn importance_steers_the_error_to_the_unimportant_columns() {
        // A row where a quarter of the columns matter 1000x more: the weighted error
        // on the important half must drop, at the expense of the other half.
        let n = 256 * 4;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * 2.0)
            .collect();
        let importance: Vec<f32> = (0..n)
            .map(|i| {
                if (i * 2654435761usize) >> 7 & 3 == 0 {
                    1000.0
                } else {
                    1.0
                }
            })
            .collect();
        for dtype in [
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q2K,
            GgmlType::Q3K,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Iq4Nl,
            GgmlType::Iq4Xs,
        ] {
            let plain = quantize_tensor(&data, dtype, n, None).unwrap();
            let weighted = quantize_tensor(&data, dtype, n, Some(&importance)).unwrap();
            let decode = |b: &[u8]| dequantize(b, dtype, &[n as i32]).unwrap();
            let werr = |back: &[f32]| -> f64 {
                data.iter()
                    .zip(back)
                    .zip(&importance)
                    .map(|((a, b), w)| f64::from(*w) * f64::from(a - b).powi(2))
                    .sum()
            };
            let (e_plain, e_weighted) = (werr(&decode(&plain)), werr(&decode(&weighted)));
            assert!(
                e_weighted <= e_plain * 1.001,
                "{dtype:?}: importance-weighted error {e_weighted} > unweighted {e_plain}"
            );
        }
    }

    #[test]
    fn importance_that_cannot_apply_is_ignored_or_rejected() {
        let data = reference_signal(512);
        // Formats without an importance mode produce the same bytes.
        let w = vec![1.0f32; 512];
        assert_eq!(
            quantize_tensor(&data, GgmlType::Q8_0, 512, Some(&w)).unwrap(),
            quantize_tensor(&data, GgmlType::Q8_0, 512, None).unwrap()
        );
        // Wrong length, negative or non-finite weights are errors.
        assert!(quantize_tensor(&data, GgmlType::Q4K, 512, Some(&w[..100])).is_err());
        let mut bad = w.clone();
        bad[3] = -1.0;
        assert!(quantize_tensor(&data, GgmlType::Q4K, 512, Some(&bad)).is_err());
        bad[3] = f32::NAN;
        assert!(quantize_tensor(&data, GgmlType::Q4K, 512, Some(&bad)).is_err());
        // Stacked slices: two slices of one row each.
        let two = vec![1.0f32; 512];
        assert!(quantize_tensor(&data, GgmlType::Q4K, 256, Some(&two)).is_ok());
        assert!(quantize_tensor(&data, GgmlType::Q4K, 256, Some(&vec![1.0; 768])).is_err());
    }

    #[test]
    fn non_finite_input_does_not_panic() {
        let mut data = reference_signal(512);
        data[3] = f32::NAN;
        data[100] = f32::INFINITY;
        data[200] = f32::NEG_INFINITY;
        for dtype in [
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::Q8_1,
            GgmlType::Q2K,
            GgmlType::Q3K,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Q8K,
            GgmlType::Iq4Nl,
            GgmlType::Iq4Xs,
            GgmlType::Q1_0,
            GgmlType::Tq1_0,
            GgmlType::Tq2_0,
            GgmlType::Mxfp4,
            GgmlType::Nvfp4,
        ] {
            let _ = quantize(&data, dtype).unwrap();
        }
    }
}
