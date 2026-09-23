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

    /// Dequantize with MLX's nvfp4 `global_scale`.
    ///
    /// ⚠️ MLX's convention: `global_scale` is the tensor's amax, and the kernel
    /// multiplies by `global_scale / (448 × 6)`. NVIDIA ModelOpt's
    /// `weight_scale_2` is that quotient already, so it goes in multiplied by
    /// 2688 or every weight comes out 2688 times too small. The native engines
    /// avoid the question by scaling the product instead: see `tensor_scale`
    /// on `LayerWeight::Quantized`.
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

    /// Pins MLX's convention for the nvfp4 global scale: the kernel multiplies
    /// by `global_scale / (448 × 6)`, so it is the tensor's amax and NVIDIA's
    /// `weight_scale_2` is that divided by 2688.
    ///
    /// ⚠️ An earlier version of this test only checked that the scale changed
    /// the answer. Treating `weight_scale_2` as MLX's global scale passes that
    /// check and leaves every weight 2688 times too small, which is the bug
    /// this exists to catch.
    #[test]
    fn the_nvfp4_global_scale_is_divided_by_2688() {
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

        // What ModelOpt would store as `weight_scale_2`.
        let weight_scale_2 = 0.37f32;
        let global =
            InlineArray::from_f32_slice(&[weight_scale_2 * 448.0 * 6.0], &[1]).as_dtype(dtype::F32);
        let scaled =
            packed.dequantize_two_level(&scales, None, Some(&global), 16, 4, QuantizedMode::Nvfp4);
        crate::check_last_error().expect("dequantize with a global scale");

        let n = (rows * cols) as usize;
        let mut plain = plain;
        let mut scaled = scaled;
        let a = plain.to_f32_vec(n).expect("read plain");
        let b = scaled.to_f32_vec(n).expect("read scaled");

        // Relative to one bfloat16 ulp, the dtype fp-mode `dequantize` returns.
        // A wrong convention is off by 2688x, not by a rounding step.
        let worst = a
            .iter()
            .zip(b.iter())
            .map(|(p, q)| (p * weight_scale_2 - q).abs() / (q.abs() + f32::EPSILON))
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 2f32.powi(-8),
            "dequantize(global = s × 2688) should equal dequantize() × s: worst relative error {worst}"
        );
    }
}
