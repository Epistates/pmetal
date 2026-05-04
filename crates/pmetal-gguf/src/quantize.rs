//! K-Quant quantization utilities.
//!
//! Helper functions for quantizing f32 data to K-quant block formats.
//! Based on llama.cpp and Candle implementations.

use crate::fp4::{best_index_mxfp4, e8m0_to_fp32_half, fp32_to_ue4m3, ue4m3_to_fp32};
use crate::k_quants::{
    quantize_q2k, quantize_q3k, quantize_q4k, quantize_q5k, quantize_q6k, quantize_q8k,
};
use crate::types::GgmlType;
use pmetal_core::{PMetalError, Result};
use zerocopy::IntoBytes;

/// Quantize a float slice to the specified GGUF type.
///
/// This is the main entry point for quantization.
pub fn quantize(data: &[f32], dtype: GgmlType) -> Result<Vec<u8>> {
    match dtype {
        GgmlType::F32 => {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            Ok(bytes)
        }
        GgmlType::F16 => {
            let bytes: Vec<u8> = data
                .iter()
                .map(|&f| half::f16::from_f32(f))
                .flat_map(|f| f.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgmlType::Bf16 => {
            let bytes: Vec<u8> = data
                .iter()
                .map(|&f| half::bf16::from_f32(f))
                .flat_map(|f| f.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgmlType::Q4_0 => {
            let bytes = quantize_q4_0(data);
            Ok(bytes)
        }
        GgmlType::Q4_1 => Ok(quantize_q4_1(data)),
        GgmlType::Q5_0 => Ok(quantize_q5_0(data)),
        GgmlType::Q5_1 => Ok(quantize_q5_1(data)),
        GgmlType::Q8_0 => {
            // Q8_0: 32-element blocks with f16 scale + 32 i8 values (34 bytes/block).
            // Must NOT reuse Q8K which has 256-element blocks with different layout.
            let blocks = quantize_q8_0(data);
            Ok(blocks_to_bytes(&blocks))
        }
        GgmlType::Q8_1 => Ok(quantize_q8_1(data)),
        GgmlType::Q1_0 => Ok(quantize_q1_0(data)),
        GgmlType::Tq1_0 => Ok(quantize_tq1_0(data)),
        GgmlType::Tq2_0 => Ok(quantize_tq2_0(data)),
        GgmlType::Mxfp4 => Ok(quantize_mxfp4(data)),
        GgmlType::Nvfp4 => Ok(quantize_nvfp4(data)),
        GgmlType::Q8K => {
            let blocks = quantize_q8k(data);
            Ok(blocks_to_bytes(&blocks))
        }
        GgmlType::Q4K => {
            let blocks = quantize_q4k(data);
            Ok(blocks_to_bytes(&blocks))
        }
        GgmlType::Q5K => {
            let blocks = quantize_q5k(data);
            Ok(blocks_to_bytes(&blocks))
        }
        GgmlType::Q6K => {
            let blocks = quantize_q6k(data);
            Ok(blocks_to_bytes(&blocks))
        }
        GgmlType::Q3K => {
            let blocks = quantize_q3k(data);
            Ok(blocks_to_bytes(&blocks))
        }
        GgmlType::Q2K => {
            let blocks = quantize_q2k(data);
            Ok(blocks_to_bytes(&blocks))
        }
        _ => Err(PMetalError::Quantization(format!(
            "Quantization to {:?} not yet implemented",
            dtype
        ))),
    }
}

/// Quantize f32 data to Q4_0 format.
///
/// Q4_0 stores one f16 scale and 32 signed 4-bit values per block. GGML packs
/// elements 0..16 in the low nibbles and elements 16..32 in the high nibbles.
fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    const QK4_0: usize = 32;
    let n_blocks = data.len().div_ceil(QK4_0);
    let mut output = Vec::with_capacity(n_blocks * 18);

    for block_idx in 0..n_blocks {
        let start = block_idx * QK4_0;
        let end = (start + QK4_0).min(data.len());
        let block_data = &data[start..end];

        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &x in block_data {
            let ax = x.abs();
            if ax > amax {
                amax = ax;
                max = x;
            }
        }

        let scale = max / -8.0;
        let iscale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        output.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());

        for i in 0..16 {
            let x0 = block_data.get(i).copied().unwrap_or(0.0);
            let x1 = block_data.get(i + 16).copied().unwrap_or(0.0);
            let q0 = nearest_int(x0 * iscale + 8.5).clamp(0, 15) as u8;
            let q1 = nearest_int(x1 * iscale + 8.5).clamp(0, 15) as u8;
            output.push(q0 | (q1 << 4));
        }
    }

    output
}

/// Quantize f32 data to Q4_1 format.
fn quantize_q4_1(data: &[f32]) -> Vec<u8> {
    const QK4_1: usize = 32;
    let n_blocks = data.len().div_ceil(QK4_1);
    let mut output = Vec::with_capacity(n_blocks * 20);

    for block_idx in 0..n_blocks {
        let start = block_idx * QK4_1;
        let end = (start + QK4_1).min(data.len());
        let block_data = &data[start..end];

        let (min, max) = block_min_max(block_data);
        let scale = (max - min) / 15.0;
        let iscale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        output.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        output.extend_from_slice(&half::f16::from_f32(min).to_le_bytes());

        for i in 0..16 {
            let x0 = block_data.get(i).copied().unwrap_or(0.0);
            let x1 = block_data.get(i + 16).copied().unwrap_or(0.0);
            let q0 = nearest_int((x0 - min) * iscale).clamp(0, 15) as u8;
            let q1 = nearest_int((x1 - min) * iscale).clamp(0, 15) as u8;
            output.push(q0 | (q1 << 4));
        }
    }

    output
}

fn quantize_q5_0(data: &[f32]) -> Vec<u8> {
    const QK5_0: usize = 32;
    let n_blocks = data.len().div_ceil(QK5_0);
    let mut output = Vec::with_capacity(n_blocks * 22);

    for block_idx in 0..n_blocks {
        let start = block_idx * QK5_0;
        let end = (start + QK5_0).min(data.len());
        let block_data = &data[start..end];

        let (max, _) = signed_absmax(block_data);
        let scale = max / -16.0;
        let iscale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let mut qh = 0u32;
        let mut qs = [0u8; QK5_0 / 2];

        for i in 0..16 {
            let x0 = block_data.get(i).copied().unwrap_or(0.0);
            let x1 = block_data.get(i + 16).copied().unwrap_or(0.0);
            let q0 = nearest_int(x0 * iscale + 16.5).clamp(0, 31) as u8;
            let q1 = nearest_int(x1 * iscale + 16.5).clamp(0, 31) as u8;
            qs[i] = (q0 & 0x0f) | ((q1 & 0x0f) << 4);
            qh |= u32::from((q0 & 0x10) >> 4) << i;
            qh |= u32::from((q1 & 0x10) >> 4) << (i + 16);
        }

        output.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        output.extend_from_slice(&qh.to_le_bytes());
        output.extend_from_slice(&qs);
    }

    output
}

fn quantize_q5_1(data: &[f32]) -> Vec<u8> {
    const QK5_1: usize = 32;
    let n_blocks = data.len().div_ceil(QK5_1);
    let mut output = Vec::with_capacity(n_blocks * 24);

    for block_idx in 0..n_blocks {
        let start = block_idx * QK5_1;
        let end = (start + QK5_1).min(data.len());
        let block_data = &data[start..end];

        let (min, max) = block_min_max(block_data);
        let scale = (max - min) / 31.0;
        let iscale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let mut qh = 0u32;
        let mut qs = [0u8; QK5_1 / 2];

        for i in 0..16 {
            let x0 = block_data.get(i).copied().unwrap_or(0.0);
            let x1 = block_data.get(i + 16).copied().unwrap_or(0.0);
            let q0 = nearest_int((x0 - min) * iscale).clamp(0, 31) as u8;
            let q1 = nearest_int((x1 - min) * iscale).clamp(0, 31) as u8;
            qs[i] = (q0 & 0x0f) | ((q1 & 0x0f) << 4);
            qh |= u32::from((q0 & 0x10) >> 4) << i;
            qh |= u32::from((q1 & 0x10) >> 4) << (i + 16);
        }

        output.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        output.extend_from_slice(&half::f16::from_f32(min).to_le_bytes());
        output.extend_from_slice(&qh.to_le_bytes());
        output.extend_from_slice(&qs);
    }

    output
}

/// Quantize f32 data to Q1_0 format.
///
/// Q1_0 stores one f16 mean-absolute scale and 128 sign bits per block.
fn quantize_q1_0(data: &[f32]) -> Vec<u8> {
    const QK1_0: usize = 128;
    let n_blocks = data.len().div_ceil(QK1_0);
    let mut output = Vec::with_capacity(n_blocks * 18);

    for block_idx in 0..n_blocks {
        let start = block_idx * QK1_0;
        let end = (start + QK1_0).min(data.len());
        let block_data = &data[start..end];

        let sum_abs = block_data.iter().fold(0.0f32, |sum, value| {
            if value.is_finite() {
                sum + value.abs()
            } else {
                sum
            }
        });
        let scale = sum_abs / QK1_0 as f32;
        output.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());

        let mut qs = [0u8; QK1_0 / 8];
        for i in 0..QK1_0 {
            let value = block_data.get(i).copied().unwrap_or(0.0);
            if value >= 0.0 {
                qs[i / 8] |= 1 << (i % 8);
            }
        }
        output.extend_from_slice(&qs);
    }

    output
}

/// Quantize f32 data to Q8_0 format.
///
/// Q8_0: 32-element blocks, each with an f16 scale and 32 i8 quantized values.
/// Block size: 34 bytes (2 bytes scale + 32 bytes data).
fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
    const QK8_0: usize = 32;
    let n_blocks = data.len().div_ceil(QK8_0);
    let mut output = Vec::with_capacity(n_blocks * 34);

    for block_idx in 0..n_blocks {
        let start = block_idx * QK8_0;
        let end = (start + QK8_0).min(data.len());
        let block_data = &data[start..end];

        // Find max absolute value for scale
        let amax = block_data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

        let scale = amax / 127.0;
        let iscale = if amax == 0.0 { 0.0 } else { 127.0 / amax };

        // Write f16 scale
        let scale_f16 = half::f16::from_f32(scale);
        output.extend_from_slice(&scale_f16.to_le_bytes());

        // Quantize and write i8 values
        for i in 0..QK8_0 {
            let val = if i < block_data.len() {
                nearest_int(block_data[i] * iscale).clamp(-127, 127) as i8
            } else {
                0i8 // zero-pad partial last block
            };
            output.push(val as u8);
        }
    }

    output
}

fn quantize_q8_1(data: &[f32]) -> Vec<u8> {
    const QK8_1: usize = 32;
    let n_blocks = data.len().div_ceil(QK8_1);
    let mut output = Vec::with_capacity(n_blocks * 36);

    for block_idx in 0..n_blocks {
        let start = block_idx * QK8_1;
        let end = (start + QK8_1).min(data.len());
        let block_data = &data[start..end];

        let amax = block_data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let scale = amax / 127.0;
        let iscale = if amax == 0.0 { 0.0 } else { 127.0 / amax };
        let mut qs = [0i8; QK8_1];
        let mut sum = 0i32;

        for (i, quant) in qs.iter_mut().enumerate() {
            let value = block_data.get(i).copied().unwrap_or(0.0);
            *quant = nearest_int(value * iscale).clamp(-127, 127) as i8;
            sum += i32::from(*quant);
        }

        output.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        output.extend_from_slice(&half::f16::from_f32(sum as f32 * scale).to_le_bytes());
        output.extend(qs.iter().map(|value| *value as u8));
    }

    output
}

fn signed_absmax(data: &[f32]) -> (f32, f32) {
    let mut signed = 0.0f32;
    let mut absmax = 0.0f32;
    for &value in data {
        if value.is_finite() && value.abs() > absmax {
            absmax = value.abs();
            signed = value;
        }
    }
    (signed, absmax)
}

fn block_min_max(data: &[f32]) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let (min, max) = data
        .iter()
        .filter(|value| value.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &value| {
            (min.min(value), max.max(value))
        });
    if min.is_finite() && max.is_finite() {
        (min, max)
    } else {
        (0.0, 0.0)
    }
}

fn quantize_mxfp4(data: &[f32]) -> Vec<u8> {
    const QK_MXFP4: usize = 32;
    let n_blocks = data.len().div_ceil(QK_MXFP4);
    let mut output = Vec::with_capacity(n_blocks * 17);

    for block_idx in 0..n_blocks {
        let start = block_idx * QK_MXFP4;
        let end = (start + QK_MXFP4).min(data.len());
        let block_data = &data[start..end];
        let amax = block_data.iter().fold(0.0f32, |max, value| {
            if value.is_finite() {
                max.max(value.abs())
            } else {
                max
            }
        });
        let scale_exp = if amax > 0.0 {
            (amax.log2().floor() - 2.0 + 127.0).clamp(0.0, 255.0) as u8
        } else {
            0
        };
        let scale = e8m0_to_fp32_half(scale_exp);
        output.push(scale_exp);

        for i in 0..16 {
            let x0 = block_data.get(i).copied().unwrap_or(0.0);
            let x1 = block_data.get(i + 16).copied().unwrap_or(0.0);
            let q0 = best_index_mxfp4(x0, scale);
            let q1 = best_index_mxfp4(x1, scale);
            output.push(q0 | (q1 << 4));
        }
    }

    output
}

fn quantize_nvfp4(data: &[f32]) -> Vec<u8> {
    const QK_NVFP4: usize = 64;
    const QK_NVFP4_SUB: usize = 16;
    const N_SUB_BLOCKS: usize = QK_NVFP4 / QK_NVFP4_SUB;
    let n_blocks = data.len().div_ceil(QK_NVFP4);
    let mut output = Vec::with_capacity(n_blocks * 36);

    for block_idx in 0..n_blocks {
        let start = block_idx * QK_NVFP4;
        let block_start = output.len();
        output.extend_from_slice(&[0u8; 36]);

        for sub in 0..N_SUB_BLOCKS {
            let sub_start = start + sub * QK_NVFP4_SUB;
            let sub_end = (sub_start + QK_NVFP4_SUB).min(data.len());
            let sub_data = if sub_start < data.len() {
                &data[sub_start..sub_end]
            } else {
                &[]
            };
            let amax = sub_data.iter().fold(0.0f32, |max, value| {
                if value.is_finite() {
                    max.max(value.abs())
                } else {
                    max
                }
            });
            let scale_byte = fp32_to_ue4m3(amax / 6.0);
            let scale = ue4m3_to_fp32(scale_byte);
            output[block_start + sub] = scale_byte;

            for i in 0..QK_NVFP4_SUB / 2 {
                let x0 = sub_data.get(i).copied().unwrap_or(0.0);
                let x1 = sub_data.get(i + QK_NVFP4_SUB / 2).copied().unwrap_or(0.0);
                let q0 = best_index_mxfp4(x0, scale);
                let q1 = best_index_mxfp4(x1, scale);
                output[block_start + 4 + sub * (QK_NVFP4_SUB / 2) + i] = q0 | (q1 << 4);
            }
        }
    }

    output
}

#[inline]
fn ternary_quant(value: f32, inverse_scale: f32) -> u8 {
    let quant = (value * inverse_scale).round().clamp(-1.0, 1.0) as i32;
    (quant + 1) as u8
}

#[inline]
fn encode_five_trits(mut q: u8) -> u8 {
    q = q.saturating_mul(3);
    ((u16::from(q) * 256 + 242) / 243) as u8
}

fn quantize_tq1_0(data: &[f32]) -> Vec<u8> {
    const QK_K: usize = 256;
    const QS_BYTES: usize = 48;
    const QH_BYTES: usize = 4;
    let n_blocks = data.len().div_ceil(QK_K);
    let mut output = Vec::with_capacity(n_blocks * 54);

    for block_idx in 0..n_blocks {
        let start = block_idx * QK_K;
        let end = (start + QK_K).min(data.len());
        let block_data = &data[start..end];
        let amax = block_data.iter().fold(0.0f32, |max, value| {
            if value.is_finite() {
                max.max(value.abs())
            } else {
                max
            }
        });
        let inverse_scale = if amax > 0.0 { 1.0 / amax } else { 0.0 };
        let mut qs = [0u8; QS_BYTES];
        let mut qh = [0u8; QH_BYTES];
        let mut base = 0usize;

        for j in (0..QS_BYTES - QS_BYTES % 32).step_by(32) {
            for m in 0..32 {
                let mut q = 0u8;
                for n in 0..5 {
                    let value = block_data.get(base + m + n * 32).copied().unwrap_or(0.0);
                    q = q * 3 + ternary_quant(value, inverse_scale);
                }
                qs[j + m] = ((u16::from(q) * 256 + 242) / 243) as u8;
            }
            base += 5 * 32;
        }

        for j in (QS_BYTES - QS_BYTES % 32..QS_BYTES).step_by(16) {
            for m in 0..16 {
                let mut q = 0u8;
                for n in 0..5 {
                    let value = block_data.get(base + m + n * 16).copied().unwrap_or(0.0);
                    q = q * 3 + ternary_quant(value, inverse_scale);
                }
                qs[j + m] = ((u16::from(q) * 256 + 242) / 243) as u8;
            }
            base += 5 * 16;
        }

        for (j, qh_byte) in qh.iter_mut().enumerate() {
            let mut q = 0u8;
            for m in 0..4 {
                let value = block_data
                    .get(base + j + m * QH_BYTES)
                    .copied()
                    .unwrap_or(0.0);
                q = q * 3 + ternary_quant(value, inverse_scale);
            }
            *qh_byte = encode_five_trits(q);
        }

        output.extend_from_slice(&qs);
        output.extend_from_slice(&qh);
        output.extend_from_slice(&half::f16::from_f32(amax).to_le_bytes());
    }

    output
}

fn quantize_tq2_0(data: &[f32]) -> Vec<u8> {
    const QK_K: usize = 256;
    const QS_BYTES: usize = 64;
    let n_blocks = data.len().div_ceil(QK_K);
    let mut output = Vec::with_capacity(n_blocks * 66);

    for block_idx in 0..n_blocks {
        let start = block_idx * QK_K;
        let end = (start + QK_K).min(data.len());
        let block_data = &data[start..end];
        let amax = block_data.iter().fold(0.0f32, |max, value| {
            if value.is_finite() {
                max.max(value.abs())
            } else {
                max
            }
        });
        let inverse_scale = if amax > 0.0 { 1.0 / amax } else { 0.0 };
        let mut qs = [0u8; QS_BYTES];
        let mut base = 0usize;

        for j in (0..QS_BYTES).step_by(32) {
            for m in 0..32 {
                let mut q = 0u8;
                for n in 0..4 {
                    let value = block_data.get(base + m + n * 32).copied().unwrap_or(0.0);
                    let xi = ternary_quant(value, inverse_scale);
                    q |= (xi & 3) << (2 * n);
                }
                qs[j + m] = q;
            }
            base += 4 * 32;
        }

        output.extend_from_slice(&qs);
        output.extend_from_slice(&half::f16::from_f32(amax).to_le_bytes());
    }

    output
}

/// Convert a slice of any block type to raw bytes.
fn blocks_to_bytes<T: zerocopy::IntoBytes + zerocopy::Immutable>(blocks: &[T]) -> Vec<u8> {
    blocks.as_bytes().to_vec()
}

/// Round to nearest integer.
#[inline]
pub fn nearest_int(v: f32) -> i32 {
    v.round() as i32
}

/// Compute scale and minimum for quantization with range optimization.
///
/// Used by Q2K, Q4K, Q5K quantization.
///
/// # Arguments
/// * `nmax` - Maximum quantized value (e.g., 3 for Q2K, 15 for Q4K, 31 for Q5K)
/// * `ntry` - Number of optimization iterations
/// * `x` - Input float slice to quantize
///
/// # Returns
/// Tuple of (scale, min) for dequantization: `output = scale * q - min`
pub fn make_qkx1_quants(nmax: i32, ntry: usize, x: &[f32]) -> (f32, f32) {
    let n = x.len();
    let mut l = vec![0u8; n];

    // Get min/max
    let min = *x
        .iter()
        .take(n)
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(&x[0]);
    let max = *x.iter().max_by(|a, b| a.total_cmp(b)).unwrap_or(&x[0]);

    // If min == max, all values are the same
    if max == min {
        return (0.0, 0.0);
    }

    // Ensure min <= 0.0
    let mut min = min.min(0.0);

    // Compute scale and inverse scale
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;

    for _ in 0..ntry {
        let mut sumlx = 0.0;
        let mut suml2 = 0i32;
        let mut did_change = false;

        for (i, value) in x.iter().enumerate().take(n) {
            let li = nearest_int(iscale * (value - min)).clamp(0, nmax);
            let clamped_li = li as u8;
            if clamped_li != l[i] {
                l[i] = clamped_li;
                did_change = true;
            }
            sumlx += (value - min) * li as f32;
            suml2 += li * li;
        }
        if suml2 == 0 {
            break; // All quantized values are zero; no further refinement possible
        }
        scale = sumlx / suml2 as f32;

        let sum: f32 = x
            .iter()
            .take(n)
            .zip(l.iter().take(n))
            .map(|(xi, &li)| xi - scale * li as f32)
            .sum();

        min = sum / n as f32;
        if min > 0.0 {
            min = 0.0;
        }
        iscale = 1.0 / scale;
        if !did_change {
            break;
        }
    }
    (scale, -min)
}

/// Compute scale for Q3K-style quantization with RMSE optimization.
///
/// # Arguments
/// * `x` - Input float slice to quantize
/// * `nmax` - Maximum absolute quantized value (typically 4)
/// * `do_rmse` - Whether to use RMSE-based optimization
///
/// # Returns
/// Scale factor for dequantization
pub fn make_q3_quants(x: &[f32], nmax: i32, do_rmse: bool) -> f32 {
    let n = x.len();
    let mut l = vec![0i8; n];

    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &xi in x.iter().take(n) {
        let ax = xi.abs();
        if ax > amax {
            amax = ax;
            max = xi;
        }
    }

    if amax == 0.0 {
        return 0.0;
    }

    let iscale = -(nmax as f32) / max;

    if do_rmse {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0.0f32;
        for i in 0..n {
            let li = (iscale * x[i]).round() as i32;
            let li = li.clamp(-nmax, nmax - 1);
            l[i] = li as i8;
            let w = x[i] * x[i];
            sumlx += w * x[i] * li as f32;
            suml2 += w * (li * li) as f32;
        }

        // RMSE optimization iterations
        for _ in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let mut slx = sumlx - w * x[i] * l[i] as f32;
                if slx > 0.0 {
                    let mut sl2 = suml2 - w * (l[i] as i32 * l[i] as i32) as f32;
                    let mut new_l = (x[i] * sl2 / slx).round() as i32;
                    new_l = new_l.clamp(-nmax, nmax - 1);
                    if new_l != l[i] as i32 {
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

        for li in l.iter_mut() {
            *li += nmax as i8;
        }
        return sumlx / suml2;
    }

    // Simple quantization without RMSE optimization
    for i in 0..n {
        let li = (iscale * x[i]).round() as i32;
        l[i] = (li.clamp(-nmax, nmax - 1) + nmax) as i8;
    }
    1.0 / iscale
}

/// Compute scale for Q6K-style symmetric quantization.
///
/// # Arguments
/// * `n` - Number of elements
/// * `nmax` - Maximum quantized value (typically 32)
/// * `x` - Input float slice
/// * `rmse_type` - RMSE optimization type (0 = none, 1 = weight by x^2)
///
/// # Returns
/// Scale factor for dequantization
pub fn make_qx_quants(n: usize, nmax: i32, x: &[f32], rmse_type: i32) -> (f32, Vec<i8>) {
    let mut ls = vec![0i8; n];

    // Find max absolute value
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &xi in x.iter().take(n) {
        let ax = xi.abs();
        if ax > amax {
            amax = ax;
            max = xi;
        }
    }

    if amax == 0.0 {
        return (0.0, ls);
    }

    let iscale = -(nmax as f32) / max;

    if rmse_type == 0 {
        for i in 0..n {
            let l = nearest_int(iscale * x[i]);
            ls[i] = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        return (1.0 / iscale, ls);
    }

    let weight_type = rmse_type % 2;
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;

    for i in 0..n {
        let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        ls[i] = (l + nmax) as i8;
        let w = if weight_type == 1 { x[i] * x[i] } else { 1.0 };
        sumlx += w * x[i] * l as f32;
        suml2 += w * (l * l) as f32;
    }

    let mut scale = sumlx / suml2;
    let mut best = scale * sumlx;

    // Optimization iterations
    for _ in 0..3 {
        let iscale = 1.0 / scale;
        let mut slx = 0.0f32;
        let mut sl2 = 0.0f32;
        let mut changed = false;

        for i in 0..n {
            let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            if l + nmax != ls[i] as i32 {
                changed = true;
            }
            let w = if weight_type == 1 { x[i] * x[i] } else { 1.0 };
            slx += w * x[i] * l as f32;
            sl2 += w * (l * l) as f32;
        }

        if !changed || sl2 == 0.0 || slx * slx <= best * sl2 {
            break;
        }

        for i in 0..n {
            let l = nearest_int(iscale * x[i]);
            ls[i] = (nmax + l.clamp(-nmax, nmax - 1)) as i8;
        }
        sumlx = slx;
        suml2 = sl2;
        scale = sumlx / suml2;
        best = scale * sumlx;
    }

    // Fine-tuning iterations
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = if weight_type == 1 { x[i] * x[i] } else { 1.0 };
            let l = ls[i] as i32 - nmax;
            let mut slx = sumlx - w * x[i] * l as f32;
            if slx > 0.0 {
                let mut sl2 = suml2 - w * (l * l) as f32;
                let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                if new_l != l {
                    slx += w * x[i] * new_l as f32;
                    sl2 += w * (new_l * new_l) as f32;
                    if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        ls[i] = (nmax + new_l) as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        scale = sumlx / suml2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }

    (scale, ls)
}

/// Extract scale and min from packed Q4K/Q5K scales array.
#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_int() {
        assert_eq!(nearest_int(0.5), 1);
        assert_eq!(nearest_int(0.4), 0);
        assert_eq!(nearest_int(-0.5), -1); // Rust rounds .5 away from zero
        assert_eq!(nearest_int(-0.6), -1);
        assert_eq!(nearest_int(1.5), 2);
    }

    #[test]
    fn test_quantize_q4_0_round_trip_layout() {
        let input: Vec<f32> = (-16..16).map(|i| i as f32 / 2.0).collect();
        let bytes = quantize(&input, GgmlType::Q4_0).unwrap();
        assert_eq!(bytes.len(), 18);

        let restored = crate::dequant::dequantize(&bytes, GgmlType::Q4_0, &[32]).unwrap();
        assert_eq!(restored.len(), input.len());
        for (actual, expected) in restored.iter().zip(input.iter()) {
            assert!(
                (actual - expected).abs() <= 1.1,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_quantize_legacy_formats_round_trip_layouts() {
        let input: Vec<f32> = (-16..16).map(|i| i as f32 / 4.0).collect();

        let q4_1 = quantize(&input, GgmlType::Q4_1).unwrap();
        assert_eq!(q4_1.len(), 20);
        let restored = crate::dequant::dequantize(&q4_1, GgmlType::Q4_1, &[32]).unwrap();
        for (actual, expected) in restored.iter().zip(input.iter()) {
            assert!((actual - expected).abs() <= 0.3);
        }

        let q5_0 = quantize(&input, GgmlType::Q5_0).unwrap();
        assert_eq!(q5_0.len(), 22);
        let restored = crate::dequant::dequantize(&q5_0, GgmlType::Q5_0, &[32]).unwrap();
        for (actual, expected) in restored.iter().zip(input.iter()) {
            assert!((actual - expected).abs() <= 0.25);
        }

        let q5_1 = quantize(&input, GgmlType::Q5_1).unwrap();
        assert_eq!(q5_1.len(), 24);
        let restored = crate::dequant::dequantize(&q5_1, GgmlType::Q5_1, &[32]).unwrap();
        for (actual, expected) in restored.iter().zip(input.iter()) {
            assert!((actual - expected).abs() <= 0.25);
        }

        let q8_1 = quantize(&input, GgmlType::Q8_1).unwrap();
        assert_eq!(q8_1.len(), 36);
        let restored = crate::dequant::dequantize(&q8_1, GgmlType::Q8_1, &[32]).unwrap();
        for (actual, expected) in restored.iter().zip(input.iter()) {
            assert!((actual - expected).abs() <= 0.04);
        }
    }

    #[test]
    fn test_quantize_bf16_round_trip_layout() {
        let input = [1.0f32, -2.5, 3.25];
        let bytes = quantize(&input, GgmlType::Bf16).unwrap();
        assert_eq!(bytes.len(), 6);

        let restored = crate::dequant::dequantize(&bytes, GgmlType::Bf16, &[3]).unwrap();
        for (actual, expected) in restored.iter().zip(input.iter()) {
            assert!((actual - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantize_current_low_bit_formats_round_trip_layouts() {
        let q1_input: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { -0.25 } else { 0.25 })
            .collect();
        let q1_bytes = quantize(&q1_input, GgmlType::Q1_0).unwrap();
        assert_eq!(q1_bytes.len(), 18);
        let q1_restored = crate::dequant::dequantize(&q1_bytes, GgmlType::Q1_0, &[128]).unwrap();
        assert!(q1_restored[0] < 0.0);
        assert!(q1_restored[1] > 0.0);

        let ternary_input: Vec<f32> = (0..256)
            .map(|i| match i % 3 {
                0 => -3.0,
                1 => 0.0,
                _ => 3.0,
            })
            .collect();
        let tq1_bytes = quantize(&ternary_input, GgmlType::Tq1_0).unwrap();
        assert_eq!(tq1_bytes.len(), 54);
        let tq1_restored = crate::dequant::dequantize(&tq1_bytes, GgmlType::Tq1_0, &[256]).unwrap();
        assert!((tq1_restored[0] + 3.0).abs() < 0.01);
        assert!(tq1_restored[1].abs() < 0.01);
        assert!((tq1_restored[2] - 3.0).abs() < 0.01);

        let tq2_bytes = quantize(&ternary_input, GgmlType::Tq2_0).unwrap();
        assert_eq!(tq2_bytes.len(), 66);
        let tq2_restored = crate::dequant::dequantize(&tq2_bytes, GgmlType::Tq2_0, &[256]).unwrap();
        assert!((tq2_restored[0] + 3.0).abs() < 0.01);
        assert!(tq2_restored[1].abs() < 0.01);
        assert!((tq2_restored[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_quantize_fp4_formats_round_trip_layouts() {
        let mxfp4_input: Vec<f32> = (0..32)
            .map(|i| match i % 4 {
                0 => -12.0,
                1 => -1.0,
                2 => 1.0,
                _ => 12.0,
            })
            .collect();
        let mxfp4_bytes = quantize(&mxfp4_input, GgmlType::Mxfp4).unwrap();
        assert_eq!(mxfp4_bytes.len(), 17);
        let mxfp4_restored =
            crate::dequant::dequantize(&mxfp4_bytes, GgmlType::Mxfp4, &[32]).unwrap();
        assert!((mxfp4_restored[0] + 12.0).abs() < 0.01);
        assert!((mxfp4_restored[3] - 12.0).abs() < 0.01);

        let nvfp4_input: Vec<f32> = (0..64)
            .map(|i| match i % 4 {
                0 => -6.0,
                1 => -1.0,
                2 => 1.0,
                _ => 6.0,
            })
            .collect();
        let nvfp4_bytes = quantize(&nvfp4_input, GgmlType::Nvfp4).unwrap();
        assert_eq!(nvfp4_bytes.len(), 36);
        let nvfp4_restored =
            crate::dequant::dequantize(&nvfp4_bytes, GgmlType::Nvfp4, &[64]).unwrap();
        assert!((nvfp4_restored[0] + 6.0).abs() < 0.01);
        assert!((nvfp4_restored[3] - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_make_qkx1_quants_uniform() {
        // Test with uniform values
        let x = [1.0f32; 32];
        let (scale, min) = make_qkx1_quants(15, 5, &x);
        // All values same, so scale should be 0
        assert_eq!(scale, 0.0);
        assert_eq!(min, 0.0);
    }

    #[test]
    fn test_make_qkx1_quants_range() {
        // Test with range [0, 15]
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let (scale, min) = make_qkx1_quants(15, 5, &x);

        // Scale should be approximately 1.0
        assert!(scale > 0.5 && scale < 2.0, "scale: {}", scale);
        // Min should be approximately 0
        assert!(min.abs() < 1.0, "min: {}", min);
    }

    #[test]
    fn test_make_q3_quants_zero() {
        let x = [0.0f32; 16];
        let scale = make_q3_quants(&x, 4, true);
        assert_eq!(scale, 0.0);
    }

    #[test]
    fn test_make_q3_quants_symmetric() {
        // Symmetric range around 0
        let x: Vec<f32> = (-4..4).map(|i| i as f32).collect();
        let scale = make_q3_quants(&x, 4, true);
        assert!(scale.abs() > 0.0, "scale should be non-zero");
    }

    #[test]
    fn test_get_scale_min_k4() {
        // Test first 4 scales (simple extraction)
        let scales = [63u8, 63, 63, 63, 63, 63, 63, 63, 0, 0, 0, 0];
        let (s0, m0) = get_scale_min_k4(0, &scales);
        assert_eq!(s0, 63);
        assert_eq!(m0, 63);

        // Test with upper bits
        let scales2 = [0xC0u8, 0, 0, 0, 0, 0, 0, 0, 0x03, 0, 0, 0];
        let (s4, _m4) = get_scale_min_k4(4, &scales2);
        // Upper 2 bits from scales[0] = 0xC0 >> 6 = 3, combined with lower 4 from scales[8] = 0x03
        assert!(s4 > 0);
    }
}
