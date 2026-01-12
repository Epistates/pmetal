//! K-Quant quantization utilities.
//!
//! Helper functions for quantizing f32 data to K-quant block formats.
//! Based on llama.cpp and Candle implementations.

use crate::k_quants::{quantize_q2k, quantize_q3k, quantize_q4k, quantize_q5k, quantize_q6k, quantize_q8k};
use crate::types::GgmlType;
use pmetal_core::{PMetalError, Result};

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
        GgmlType::Q8_0 | GgmlType::Q8K => {
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

/// Convert a slice of any block type to raw bytes.
fn blocks_to_bytes<T: Copy>(blocks: &[T]) -> Vec<u8> {
    let byte_len = blocks.len() * std::mem::size_of::<T>();
    let mut bytes = vec![0u8; byte_len];
    // SAFETY: We're reinterpreting the blocks as raw bytes
    // The blocks are repr(C) packed structs
    unsafe {
        std::ptr::copy_nonoverlapping(
            blocks.as_ptr() as *const u8,
            bytes.as_mut_ptr(),
            byte_len,
        );
    }
    bytes
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