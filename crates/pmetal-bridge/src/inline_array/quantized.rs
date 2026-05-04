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
}

impl InlineArray {
    // ── Dequantize ──────────────────────────────────────────────────────

    /// Dequantize packed integer weights using per-group scales and biases.
    pub fn dequantize(&self, scales: &Self, biases: &Self, group_size: i32, bits: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_dequantize(
                dst.as_mut_ptr(),
                &self.raw,
                &scales.raw,
                &biases.raw,
                group_size,
                bits,
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
