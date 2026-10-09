//! Constructors for [`InlineArray`].
//!
//! Covers scalar constructors (`from_f32`, `from_i32`), shaped-slice loaders
//! (`from_{f32,u32,u8,u16_bits,i32}_slice`, `from_slice<T>`), shape-only
//! constructors (`zeros`, `ones`, `full`, `eye`, `tri`, `arange`, `linspace`),
//! random samplers, and typed convenience constructors.

use std::mem::MaybeUninit;

use super::dtype::ArrayElement;
use super::ffi::*;
use super::{InlineArray, RawBuf};

impl InlineArray {
    // ── Scalar / identity / iterator constructors ────────────────────────

    /// Identity constructor — clone an existing array.
    ///
    /// Since `Array = InlineArray` in this bridge, this is just `.clone()`.
    #[inline]
    pub fn from_array(other: &Self) -> Self {
        other.clone()
    }

    /// Scalar integer array constructor; alias for [`Self::from_i32`].
    #[inline]
    pub fn from_int(val: i32) -> Self {
        Self::from_i32(val)
    }

    /// Construct an array from an iterator of integers with an explicit shape.
    ///
    /// The iterator is collected into a `Vec<i32>` and shaped.
    pub fn from_iter(iter: impl IntoIterator<Item = i32>, shape: &[i32]) -> Self {
        let v: Vec<i32> = iter.into_iter().collect();
        Self::from_i32_slice_shaped(&v, shape)
    }

    pub fn from_f32(val: f32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_f32(dst.as_mut_ptr(), val);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn from_i32(val: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_i32(dst.as_mut_ptr(), val);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn zeros(shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_zeros(dst.as_mut_ptr(), shape.as_ptr(), shape.len() as i32, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn ones(shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_ones(dst.as_mut_ptr(), shape.as_ptr(), shape.len() as i32, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Shaped slice loaders ─────────────────────────────────────────────

    pub fn from_f32_slice(data: &[f32], shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_f32_slice(
                dst.as_mut_ptr(),
                data.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn from_u32_slice(data: &[u32], shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_u32_slice(
                dst.as_mut_ptr(),
                data.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn from_u8_slice(data: &[u8], shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_u8_slice(
                dst.as_mut_ptr(),
                data.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn from_u16_bits_slice(data: &[u16], shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_u16_bits_slice(
                dst.as_mut_ptr(),
                data.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
                dtype,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Copy all f32 values out of this array into a `Vec<f32>`.
    ///
    /// The array is cast to f32 and evaluated (GPU → CPU sync) before copying.
    /// Returns `None` when the element count doesn't match `n` or on dtype error.
    pub fn to_f32_vec(&mut self, n: usize) -> Option<Vec<f32>> {
        let mut out = vec![0.0f32; n];
        let rc = unsafe { mlx_inline_to_f32_slice(&mut self.raw, out.as_mut_ptr(), n) };
        if rc == 0 { Some(out) } else { None }
    }

    /// Create a 1-D int32 array from a Rust slice — zero copy for token IDs.
    ///
    /// Typical use: prefill token IDs for `embedding.take_axis(ids, 0)`.
    pub fn from_i32_slice(data: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_i32_slice(dst.as_mut_ptr(), data.as_ptr(), data.len() as i32);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Create a shaped int32 array from a Rust slice.
    ///
    /// Shape must satisfy `shape.iter().product::<i32>() == data.len() as i32`.
    pub fn from_i32_slice_shaped(data: &[i32], shape: &[i32]) -> Self {
        Self::from_i32_slice(data).reshape(shape)
    }

    /// Generic `from_slice::<T>(data, shape)`.
    ///
    /// Supports `i32`, `f32`, and `u32` element types via the [`ArrayElement`] trait.
    /// Typical usage:
    /// ```ignore
    /// let arr = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
    /// let arr = Array::from_slice(&[0.1f32, 0.2], &[2]);
    /// ```
    pub fn from_slice<T: ArrayElement>(data: &[T], shape: &[i32]) -> Self {
        T::into_array(data, shape)
    }

    /// Create a range [0, 1, ..., n-1] with full Metal buffer (no broadcast).
    /// Useful for benchmarks — ensures matmuls read real data from GPU memory.
    pub fn arange(n: i32, dtype: i32) -> Self {
        let mut dst = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_arange(dst.as_mut_ptr(), n, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Random samplers ──────────────────────────────────────────────────

    pub fn random_normal(shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_random_normal(dst.as_mut_ptr(), shape.as_ptr(), shape.len() as i32, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Sample from U(0,1) with given shape and dtype.
    pub fn random_uniform(shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_random_uniform(dst.as_mut_ptr(), shape.as_ptr(), shape.len() as i32, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Sample Bernoulli with given probability and shape.
    pub fn random_bernoulli(p: &Self, shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_random_bernoulli(
                dst.as_mut_ptr(),
                &p.raw,
                shape.as_ptr(),
                shape.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Random integers in [low, high) with given shape and dtype.
    pub fn random_randint(low: i32, high: i32, shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_random_randint(
                dst.as_mut_ptr(),
                low,
                high,
                shape.as_ptr(),
                shape.len() as i32,
                dtype,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Creation helpers: full, eye, tri ─────────────────────────────────

    pub fn full(shape: &[i32], val: f32, dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_full(
                dst.as_mut_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
                val,
                dtype,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Identity matrix [n, n].
    pub fn eye(n: i32, dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_eye(dst.as_mut_ptr(), n, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Triangular matrix [n, m] with diagonal offset k.
    pub fn tri(n: i32, m: i32, k: i32, dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_tri(dst.as_mut_ptr(), n, m, k, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── linspace ─────────────────────────────────────────────────────────

    pub fn linspace(start: f32, stop: f32, n: i32, dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_linspace(dst.as_mut_ptr(), start, stop, n, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Typed convenience constructors ──────────────────────────────────

    /// Convenience constructor: zeros with float32 dtype.
    pub fn zeros_f32(shape: &[i32]) -> Self {
        Self::zeros(shape, crate::compat::Dtype::Float32.as_i32())
    }

    /// Convenience constructor: ones with float32 dtype.
    #[inline]
    pub fn ones_f32(shape: &[i32]) -> Self {
        Self::ones(shape, crate::compat::Dtype::Float32.as_i32())
    }

    /// Convenience constructor: zeros with int32 dtype.
    #[inline]
    pub fn zeros_i32(shape: &[i32]) -> Self {
        Self::zeros(shape, crate::compat::Dtype::Int32.as_i32())
    }

    /// Cast to the specified dtype enum value; [`Self::as_dtype`] takes the
    /// raw `i32` code.
    #[inline]
    pub fn cast(&self, dtype: crate::compat::Dtype) -> Self {
        self.as_dtype(dtype.as_i32())
    }

    #[inline]
    pub fn unwrap(self) -> Self {
        self
    }

    #[inline]
    pub fn expect(self, _msg: &str) -> Self {
        self
    }
}
