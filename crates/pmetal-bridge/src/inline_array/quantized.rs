//! Quantization primitives on [`InlineArray`].
//!
//! - `quantize_weights` / `dequantize`: per-group int4 / int8 quantization.
//! - `quantized_matmul` / `gather_qmm`: dense and MoE-routed quantized matmul.
//! - `save_safetensors`: save-side of the safetensors codec (the load side
//!   lives in [`super::safetensors`]).

use std::mem::MaybeUninit;

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;

/// Quantized matmul/quantize mode understood by MLX.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizedMode {
    Affine = 0,
    Mxfp8 = 1,
    Mxfp4 = 2,
    Nvfp4 = 3,
}

impl QuantizedMode {
    #[inline]
    pub(crate) fn as_i32(self) -> i32 {
        self as i32
    }

    /// The spelling a checkpoint's `quantization` block uses, as accepted by
    /// `mx.quantize(..., mode=)`. Unknown names return `None` so a caller can
    /// reject the file rather than silently decode it as affine.
    pub fn from_config_name(name: &str) -> Option<Self> {
        match name {
            "affine" => Some(Self::Affine),
            "mxfp8" => Some(Self::Mxfp8),
            "mxfp4" => Some(Self::Mxfp4),
            "nvfp4" => Some(Self::Nvfp4),
            _ => None,
        }
    }
}

impl InlineArray {
    // ── Dequantize ──────────────────────────────────────────────────────

    /// Dequantize packed integer weights using per-group scales and biases.
    pub fn dequantize(&self, scales: &Self, biases: &Self, group_size: i32, bits: i32) -> Self {
        self.dequantize_mode(
            scales,
            Some(biases),
            group_size,
            bits,
            QuantizedMode::Affine,
        )
    }

    /// Dequantize in a specific MLX quantization mode.
    ///
    /// `biases` is optional because only affine mode has them: mxfp4, nvfp4
    /// and mxfp8 store a scale per group and nothing else. Passing a bias array
    /// for one of those is an error rather than a no-op, so the `Option` is
    /// load-bearing.
    #[inline]
    pub fn dequantize_mode(
        &self,
        scales: &Self,
        biases: Option<&Self>,
        group_size: i32,
        bits: i32,
        mode: QuantizedMode,
    ) -> Self {
        self.dequantize_two_level(scales, biases, None, group_size, bits, mode)
    }

    /// Dequantize a tensor that may carry a per-tensor scale beside its
    /// per-group ones.
    ///
    /// nvfp4 is two-level: an fp8 scale per group of 16, and one fp32 scale for
    /// the whole tensor. NVIDIA ModelOpt checkpoints ship the second as
    /// `weight_scale_2`. Dropping it does not fail or change a shape, it
    /// returns every weight off by a constant factor.
    #[inline]
    pub fn dequantize_two_level(
        &self,
        scales: &Self,
        biases: Option<&Self>,
        global_scale: Option<&Self>,
        group_size: i32,
        bits: i32,
        mode: QuantizedMode,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        let b_ptr = biases
            .map(|b| &b.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        let g_ptr = global_scale
            .map(|g| &g.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        unsafe {
            mlx_inline_dequantize(
                dst.as_mut_ptr(),
                &self.raw,
                &scales.raw,
                b_ptr,
                group_size,
                bits,
                mode.as_i32(),
                g_ptr,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Two-level quantized matmul: `x @ dequantize(self)`.
    ///
    /// MLX's `quantized_matmul` takes no global scale, so an nvfp4 weight has
    /// to come through here or its per-tensor scale is silently dropped.
    /// `global_scale_x` is for a quantized activation and stays `None` on the
    /// weight-only path.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn qqmm_from(
        &self,
        x: &Self,
        w_scales: Option<&Self>,
        global_scale_x: Option<&Self>,
        global_scale_w: Option<&Self>,
        group_size: i32,
        bits: i32,
        mode: QuantizedMode,
    ) -> Self {
        let ptr = |a: Option<&Self>| {
            a.map(|v| &v.raw as *const RawBuf)
                .unwrap_or(std::ptr::null())
        };
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_qqmm(
                dst.as_mut_ptr(),
                &x.raw,
                &self.raw,
                ptr(w_scales),
                group_size,
                bits,
                mode.as_i32(),
                ptr(global_scale_x),
                ptr(global_scale_w),
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Two-level gather matmul for MoE expert dispatch.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn gather_qqmm_from(
        &self,
        x: &Self,
        w_scales: Option<&Self>,
        lhs_indices: Option<&Self>,
        rhs_indices: Option<&Self>,
        global_scale_x: Option<&Self>,
        global_scale_w: Option<&Self>,
        group_size: i32,
        bits: i32,
        mode: QuantizedMode,
        sorted_indices: bool,
    ) -> Self {
        let ptr = |a: Option<&Self>| {
            a.map(|v| &v.raw as *const RawBuf)
                .unwrap_or(std::ptr::null())
        };
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_gather_qqmm(
                dst.as_mut_ptr(),
                &x.raw,
                &self.raw,
                ptr(w_scales),
                ptr(lhs_indices),
                ptr(rhs_indices),
                group_size,
                bits,
                mode.as_i32(),
                ptr(global_scale_x),
                ptr(global_scale_w),
                sorted_indices,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Quantized matmul ──────────────────────────────────────────────────

    /// Quantized matmul: `x @ dequantize(w, scales, biases)`.
    #[inline]
    pub fn quantized_matmul(
        &self,
        w: &Self,
        scales: &Self,
        biases: Option<&Self>,
        transpose: bool,
        group_size: i32,
        bits: i32,
    ) -> Self {
        self.quantized_matmul_mode(
            w,
            scales,
            biases,
            transpose,
            group_size,
            bits,
            QuantizedMode::Affine,
        )
    }

    /// Quantized matmul in a specific MLX quantization mode.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn quantized_matmul_mode(
        &self,
        w: &Self,
        scales: &Self,
        biases: Option<&Self>,
        transpose: bool,
        group_size: i32,
        bits: i32,
        mode: QuantizedMode,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        let b_ptr = biases
            .map(|b| &b.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        unsafe {
            mlx_inline_quantized_matmul(
                dst.as_mut_ptr(),
                &self.raw,
                &w.raw,
                &scales.raw,
                b_ptr,
                transpose,
                group_size,
                bits,
                mode.as_i32(),
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Gather quantized matmul (MoE expert routing).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn gather_qmm(
        &self,
        w: &Self,
        scales: &Self,
        biases: Option<&Self>,
        lhs_indices: Option<&Self>,
        rhs_indices: Option<&Self>,
        transpose: bool,
        group_size: i32,
        bits: i32,
        sorted: bool,
    ) -> Self {
        self.gather_qmm_mode(
            w,
            scales,
            biases,
            lhs_indices,
            rhs_indices,
            transpose,
            group_size,
            bits,
            sorted,
            QuantizedMode::Affine,
        )
    }

    /// Gather quantized matmul in a specific MLX quantization mode.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn gather_qmm_mode(
        &self,
        w: &Self,
        scales: &Self,
        biases: Option<&Self>,
        lhs_indices: Option<&Self>,
        rhs_indices: Option<&Self>,
        transpose: bool,
        group_size: i32,
        bits: i32,
        sorted: bool,
        mode: QuantizedMode,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        let b_ptr = biases
            .map(|b| &b.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        let l_ptr = lhs_indices
            .map(|l| &l.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        let r_ptr = rhs_indices
            .map(|r| &r.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        unsafe {
            mlx_inline_gather_qmm(
                dst.as_mut_ptr(),
                &self.raw,
                &w.raw,
                &scales.raw,
                b_ptr,
                l_ptr,
                r_ptr,
                transpose,
                group_size,
                bits,
                sorted,
                mode.as_i32(),
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Quantize ──────────────────────────────────────────────────────

    /// Quantize: returns (packed_weights, scales, biases).
    pub fn quantize_weights(&self, group_size: i32, bits: i32) -> (Self, Self, Self) {
        let mut w = MaybeUninit::<RawBuf>::uninit();
        let mut s = MaybeUninit::<RawBuf>::uninit();
        let mut b = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_quantize(
                w.as_mut_ptr(),
                s.as_mut_ptr(),
                b.as_mut_ptr(),
                &self.raw,
                group_size,
                bits,
            );
            (
                Self {
                    raw: w.assume_init(),
                },
                Self {
                    raw: s.assume_init(),
                },
                Self {
                    raw: b.assume_init(),
                },
            )
        }
    }

    /// Quantize in an MLX floating-point quantization mode such as mxfp8.
    ///
    /// These modes do not have affine biases, so the return value is only
    /// `(packed_weights, scales)`.
    pub fn quantize_weights_mode(
        &self,
        group_size: i32,
        bits: i32,
        mode: QuantizedMode,
    ) -> (Self, Self) {
        let mut w = MaybeUninit::<RawBuf>::uninit();
        let mut s = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_quantize_mode(
                w.as_mut_ptr(),
                s.as_mut_ptr(),
                &self.raw,
                group_size,
                bits,
                mode.as_i32(),
            );
            (
                Self {
                    raw: w.assume_init(),
                },
                Self {
                    raw: s.assume_init(),
                },
            )
        }
    }

    // ── save-side of the safetensors codec ──────────────────────────────

    /// Save arrays to safetensors format.
    pub fn save_safetensors(path: &str, entries: &[(&str, &InlineArray)]) {
        let c_path = std::ffi::CString::new(path).expect("null byte in path");
        let c_keys: Vec<std::ffi::CString> = entries
            .iter()
            .map(|(k, _)| std::ffi::CString::new(*k).expect("null byte in key"))
            .collect();
        let key_ptrs: Vec<*const std::ffi::c_char> = c_keys.iter().map(|k| k.as_ptr()).collect();
        // Build a contiguous array of RawBufs (copy refs, not move)
        let raw_arrays: Vec<RawBuf> = entries.iter().map(|(_, a)| a.raw).collect();
        unsafe {
            mlx_inline_save_safetensors(
                c_path.as_ptr(),
                key_ptrs.as_ptr(),
                raw_arrays.as_ptr(),
                entries.len() as i32,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;

    /// nvfp4 is two-level: an fp8 scale per group of 16 and one scale for the
    /// whole tensor. NVIDIA ModelOpt checkpoints ship the second as
    /// `weight_scale_2`, and the bridge had no way to pass it, so it went to
    /// MLX's `std::nullopt` default.
    ///
    /// ⚠️ Dropping it does not fail and does not change a shape. It returns
    /// every weight off by a constant factor, which is exactly the class of
    /// bug that loads, runs at the right speed, and decodes noise.
    #[test]
    fn the_nvfp4_global_scale_reaches_the_kernel() {
        let rows = 4;
        let cols = 64;
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect();
        let w = InlineArray::from_f32_slice(&values, &[rows, cols]);

        let (packed, scales) = w.quantize_weights_mode(16, 4, QuantizedMode::Nvfp4);
        crate::check_last_error().expect("nvfp4 quantize");

        let plain = packed.dequantize_two_level(&scales, None, None, 16, 4, QuantizedMode::Nvfp4);
        crate::check_last_error().expect("dequantize without a global scale");

        let half = InlineArray::from_f32_slice(&[0.5], &[1]).as_dtype(dtype::F32);
        let scaled =
            packed.dequantize_two_level(&scales, None, Some(&half), 16, 4, QuantizedMode::Nvfp4);
        crate::check_last_error().expect("dequantize with a global scale");

        let n = (rows * cols) as usize;
        let mut plain = plain;
        let mut scaled = scaled;
        let a = plain.to_f32_vec(n).expect("read plain");
        let b = scaled.to_f32_vec(n).expect("read scaled");

        // A global scale that is not 1.0 has to change the result. If the
        // argument were dropped on the floor these would be identical.
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "global scale had no effect: max|diff| = {max_diff}, so it never reached the kernel"
        );
    }
}
