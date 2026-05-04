//! Binary, unary, and element-wise math on [`InlineArray`].
//!
//! Includes all `binop!` / `unop!` macro expansions plus a grab-bag of
//! simple element-wise / scalar methods that don't naturally belong in the
//! fast, gather, or shape modules: softmax, norm_l2, FFT, leaky_relu,
//! aliases (`eq`/`ne`/`gt`/`lt`), layout queries (`size`/`nbytes`/`id`).

use std::mem::MaybeUninit;

use super::dtype::AsDtype;
use super::ffi::*;
use super::{InlineArray, RawBuf, binop, unop};

impl InlineArray {
    // ── Binary ops (macros) ──────────────────────────────────────────────

    binop!(matmul, mlx_inline_matmul);
    binop!(add, mlx_inline_add);
    binop!(multiply, mlx_inline_multiply);
    binop!(subtract, mlx_inline_subtract);
    binop!(divide, mlx_inline_divide);
    binop!(maximum, mlx_inline_maximum);
    binop!(minimum, mlx_inline_minimum);
    binop!(pow, mlx_inline_pow);
    binop!(equal, mlx_inline_equal);
    binop!(not_equal, mlx_inline_not_equal);
    binop!(greater, mlx_inline_greater);
    binop!(less, mlx_inline_less);
    binop!(greater_equal, mlx_inline_greater_equal);
    binop!(less_equal, mlx_inline_less_equal);

    // ── Unary ops (macros) ───────────────────────────────────────────────

    unop!(negative, mlx_inline_negative);
    unop!(exp, mlx_inline_exp);
    unop!(sigmoid, mlx_inline_sigmoid);
    unop!(silu, mlx_inline_silu);
    unop!(sqrt, mlx_inline_sqrt);
    unop!(t, mlx_inline_transpose);
    unop!(softplus, mlx_inline_softplus);
    unop!(log, mlx_inline_log);
    unop!(sign, mlx_inline_sign);
    unop!(reciprocal, mlx_inline_reciprocal);
    unop!(sin, mlx_inline_sin);
    unop!(cos, mlx_inline_cos);
    unop!(rsqrt, mlx_inline_rsqrt);
    unop!(zeros_like, mlx_inline_zeros_like);
    unop!(ones_like, mlx_inline_ones_like);
    unop!(square, mlx_inline_square);
    unop!(relu, mlx_inline_relu);
    unop!(gelu, mlx_inline_gelu);
    unop!(stop_gradient, mlx_inline_stop_gradient);

    // ── Softmax / norm / reshape / sum_axis ──────────────────────────────

    pub fn norm_l2(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_norm_l2(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn softmax(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_softmax(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn softmax_precise(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_softmax_precise(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn reshape(&self, shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_reshape(
                dst.as_mut_ptr(),
                &self.raw,
                shape.as_ptr(),
                shape.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn sum_axis(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_sum_axis(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn as_dtype(&self, dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_astype(dst.as_mut_ptr(), &self.raw, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Convert a floating-point matrix to MLX's native E4M3 FP8 representation.
    ///
    /// MLX stores FP8 payloads in `uint8` arrays and expects callers to use
    /// `from_fp8` before arithmetic that does not consume FP8 directly.
    pub fn to_fp8(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_to_fp8(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Dequantize an MLX E4M3 FP8 payload into a floating-point dtype.
    pub fn from_fp8(&self, dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_fp8(dst.as_mut_ptr(), &self.raw, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Cast to a Rust primitive type `T` — compatible with mlx-rs `as_type::<T>()`.
    ///
    /// Uses the [`AsDtype`] sealed trait to map Rust types to MLX dtypes.
    #[inline]
    pub fn as_type<T: AsDtype>(&self) -> Self {
        self.as_dtype(T::DTYPE_ID)
    }

    // ── Misc element-wise aliases and layout queries ────────────────────

    pub fn power(&self, other: &Self) -> Self {
        self.pow(other)
    }

    /// gt/lt/ge/le aliases for compat with mlx-rs naming.
    pub fn eq(&self, other: &Self) -> Self {
        self.equal(other)
    }
    pub fn ne(&self, other: &Self) -> Self {
        self.not_equal(other)
    }
    pub fn gt(&self, other: &Self) -> Self {
        self.greater(other)
    }
    pub fn lt(&self, other: &Self) -> Self {
        self.less(other)
    }
    /// Swap two axes.
    pub fn swap_axes(&self, a: i32, b: i32) -> Self {
        let ndim = self.ndim();
        let mut perm: Vec<i32> = (0..ndim).collect();
        let a_idx = if a < 0 { ndim + a } else { a } as usize;
        let b_idx = if b < 0 { ndim + b } else { b } as usize;
        perm.swap(a_idx, b_idx);
        self.transpose_axes(&perm)
    }

    /// Total element count.
    pub fn size(&self) -> usize {
        unsafe { mlx_inline_size(&self.raw) }
    }

    /// Total byte count.
    pub fn nbytes(&self) -> usize {
        unsafe { mlx_inline_nbytes(&self.raw) }
    }

    /// Get a raw const pointer to the evaluated data.
    /// Array must be evaluated first.
    pub fn data_ptr(&self) -> *const std::ffi::c_void {
        let mut ptr: *const std::ffi::c_void = std::ptr::null();
        unsafe { mlx_inline_data_ptr(&self.raw, &mut ptr) };
        ptr
    }

    /// Stable identity of the underlying MLX array desc.
    ///
    /// Returns `uintptr_t(array_desc_.get())` from MLX — unique per array
    /// over its lifetime, cheap, and — critically — valid on unevaluated
    /// (lazy) arrays. Use this as a change-detection handle for caches
    /// keyed on weight tensors; **do not** dereference it.
    #[inline]
    pub fn id(&self) -> usize {
        unsafe { mlx_inline_array_id(&self.raw) }
    }

    // ── FFT ──────────────────────────────────────────────────────────────

    pub fn rfft(&self, n_fft: i32, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_rfft(dst.as_mut_ptr(), &self.raw, n_fft, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Inverse real-valued FFT along `axis`. Pass `n_fft = -1` to infer from input.
    #[inline]
    pub fn irfft(&self, n_fft: i32, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_irfft(dst.as_mut_ptr(), &self.raw, n_fft, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Activations / trivial reshape ───────────────────────────────────

    pub fn leaky_relu(&self, neg_slope: f32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_leaky_relu(dst.as_mut_ptr(), &self.raw, neg_slope);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── squeeze all ───────────────────────────────────────────────────────

    /// Remove all size-1 dimensions.
    #[inline]
    pub fn squeeze_all(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_squeeze_all(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }
}
