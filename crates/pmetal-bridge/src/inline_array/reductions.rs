//! Reductions and sampling on [`InlineArray`].
//!
//! Covers arg*-returning reductions (argmax, argmin, argsort, topk),
//! value-returning reductions (sum, mean, max, min, logsumexp — per-axis
//! and global), and sampling helpers (categorical, abs).

use std::mem::MaybeUninit;

use super::ffi::*;
use super::{InlineArray, RawBuf};

impl InlineArray {
    // ── Sampling / arg-reductions / element-wise abs ─────────────────────

    #[inline]
    pub fn argmax(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_argmax(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn argmin(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_argmin(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Element-wise absolute value.
    #[inline]
    pub fn abs(&self) -> Self {
        self.abs_val()
    }

    /// Element-wise absolute value (alias to avoid f32::abs conflict in some contexts).
    #[inline]
    pub fn abs_val(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_abs(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn logsumexp(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_logsumexp(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn categorical(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_categorical(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Mean reductions ─────────────────────────────────────────────────

    pub fn mean_axis(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_mean_axis(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn mean_all(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_mean_all(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Sort / sum / max / min along axes ───────────────────────────────

    pub fn argsort(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_argsort(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Sum all elements to a scalar.
    pub fn sum_all(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_sum_all(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn max_axis(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_max_axis(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn min_axis(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_min_axis(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Sum over multiple axes.
    pub fn sum_axes(&self, axes: &[i32], keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_sum_axes(
                dst.as_mut_ptr(),
                &self.raw,
                axes.as_ptr(),
                axes.len() as i32,
                keepdims,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Mean over multiple axes.
    pub fn mean_axes(&self, axes: &[i32], keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_mean_axes(
                dst.as_mut_ptr(),
                &self.raw,
                axes.as_ptr(),
                axes.len() as i32,
                keepdims,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Top-k ───────────────────────────────────────────────────────────

    /// Top-k values along `axis`.
    pub fn topk(&self, k: i32, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_topk(dst.as_mut_ptr(), &self.raw, k, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Scalar reductions (mlx-rs compat) ───────────────────────────────
    //
    // The optional `axis` argument is now honored to match mlx-rs semantics:
    //   * `Some(ax)`: reduce along that axis (no keepdims).
    //   * `None`:     global reduction (flatten then reduce).

    /// Reduce to the maximum (along `axis` if provided, else global scalar).
    pub fn max(&self, axis: Option<i32>) -> Self {
        match axis {
            Some(ax) => self.max_axis(ax, false),
            None => self.flatten(0, -1).max_axis(0, false),
        }
    }

    /// Reduce to the minimum (along `axis` if provided, else global scalar).
    pub fn min(&self, axis: Option<i32>) -> Self {
        match axis {
            Some(ax) => self.min_axis(ax, false),
            None => self.flatten(0, -1).min_axis(0, false),
        }
    }

    /// Reduce to a sum (along `axis` if provided, else global scalar).
    pub fn sum(&self, axis: Option<i32>) -> Self {
        match axis {
            Some(ax) => self.sum_axes(&[ax], false),
            None => self.sum_all(),
        }
    }

    /// Reduce to a mean (along `axis` if provided, else global scalar).
    pub fn mean(&self, axis: Option<i32>) -> Self {
        match axis {
            Some(ax) => self.mean_axis(ax, false),
            None => self.mean_all(),
        }
    }

    /// mlx-rs compat alias.
    pub fn logsumexp_axis(&self, axis: i32, keepdims: bool) -> Self {
        self.logsumexp(axis, keepdims)
    }
}
