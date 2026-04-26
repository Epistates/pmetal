//! Shape queries and shape-changing operations on [`InlineArray`].
//!
//! Covers layout queries (ndim, dim, shape, dtype), indexing/slicing
//! (concatenate_2, slice, slice_set, squeeze, expand_dims, transpose_axes,
//! cumsum, tril, index), broadcast / flatten, and scatter/put/tile helpers.

use std::mem::MaybeUninit;

use super::ffi::*;
use super::{InlineArray, RawBuf};

impl InlineArray {
    // ── Shape / dtype query ─────────────────────────────────────────────

    pub fn ndim(&self) -> i32 {
        unsafe { mlx_inline_ndim(&self.raw) }
    }
    pub fn dim(&self, axis: i32) -> i32 {
        unsafe { mlx_inline_dim(&self.raw, axis) }
    }
    pub fn shape(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(mlx_inline_shape(&self.raw), self.ndim() as usize) }
    }
    pub fn dtype_raw(&self) -> i32 {
        unsafe { mlx_inline_dtype(&self.raw) }
    }

    /// Returns the dtype as a [`crate::compat::Dtype`] enum.
    ///
    /// Equivalent to mlx-rs `Array::dtype()`.
    #[inline]
    pub fn dtype(&self) -> crate::compat::Dtype {
        crate::compat::Dtype::from_raw(self.dtype_raw())
    }

    // ── Indexing / slicing ──────────────────────────────────────────────

    #[inline]
    pub fn concatenate_2(&self, other: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_concatenate_2(dst.as_mut_ptr(), &self.raw, &other.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn where_cond(&self, a: &Self, b: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_where(dst.as_mut_ptr(), &self.raw, &a.raw, &b.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn slice(&self, start: &[i32], stop: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_slice(
                dst.as_mut_ptr(),
                &self.raw,
                start.as_ptr(),
                stop.as_ptr(),
                start.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// In-place slice update. Consumes self so MLX sees refcount=1 and can
    /// mutate the buffer directly (zero allocation). Matches Python's
    /// `self.keys[..., prev:offset, :] = keys` pattern.
    #[inline]
    pub fn slice_set(&self, value: &Self, start: &[i32], stop: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_slice_set(
                dst.as_mut_ptr(),
                &self.raw,
                &value.raw,
                start.as_ptr(),
                stop.as_ptr(),
                start.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn repeat(&self, repeats: i32, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_repeat(dst.as_mut_ptr(), &self.raw, repeats, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn squeeze(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_squeeze(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Squeeze all size-1 dimensions (multi-axis compat alias).
    ///
    /// Process axes in descending order so each squeeze leaves the remaining
    /// axis indices valid. Avoids an initial `self.clone()` by performing the
    /// first squeeze directly against `self`.
    #[inline]
    pub fn squeeze_axes(&self, axes: &[i32]) -> Self {
        if axes.is_empty() {
            return self.clone();
        }
        let mut sorted = axes.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let mut result = self.squeeze(sorted[0]);
        for &ax in &sorted[1..] {
            result = result.squeeze(ax);
        }
        result
    }

    #[inline]
    pub fn expand_dims(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_expand_dims(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Multi-axis expand_dims — insert a new size-1 axis at each position.
    ///
    /// Compatible with mlx-rs `expand_dims_axes(&[ax1, ax2, ...])`. Insert in
    /// ascending order so each prior insertion remains at its requested axis.
    /// Avoids an initial `self.clone()` by performing the first insertion
    /// directly against `self`.
    #[inline]
    pub fn expand_dims_axes(&self, axes: &[i32]) -> Self {
        if axes.is_empty() {
            return self.clone();
        }
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        let mut result = self.expand_dims(sorted[0]);
        for &ax in &sorted[1..] {
            result = result.expand_dims(ax);
        }
        result
    }

    #[inline]
    pub fn transpose_axes(&self, axes: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_transpose_axes(
                dst.as_mut_ptr(),
                &self.raw,
                axes.as_ptr(),
                axes.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn cumsum(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_cumsum(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn tril(&self, k: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_tril(dst.as_mut_ptr(), &self.raw, k);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub(crate) fn index_array(&self, indices: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_index(dst.as_mut_ptr(), &self.raw, &indices.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Index or slice this array using the compatibility bridge.
    ///
    /// Supports gather indexing with an index array as well as mlx-rs style
    /// integer and tuple/range slicing via `compat::indexing::IndexOp`.
    #[inline]
    pub fn index<Idx>(&self, idx: Idx) -> Self
    where
        Self: crate::compat::indexing::IndexOp<Idx>,
    {
        <Self as crate::compat::indexing::IndexOp<Idx>>::index(self, idx)
    }

    // ── Broadcast / flatten ─────────────────────────────────────────────

    pub fn broadcast_to(&self, shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_broadcast_to(
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

    pub fn flatten(&self, start_axis: i32, end_axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_flatten(dst.as_mut_ptr(), &self.raw, start_axis, end_axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Tile / split / scatter / put ────────────────────────────────────

    pub fn tile(&self, reps: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_tile(
                dst.as_mut_ptr(),
                &self.raw,
                reps.as_ptr(),
                reps.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }
    pub fn split_sections(&self, sections: i32, axis: i32) -> Vec<Self> {
        // Allocate enough output slots.
        let max = sections as usize;
        let mut buf: Vec<MaybeUninit<RawBuf>> = (0..max).map(|_| MaybeUninit::uninit()).collect();
        let mut out_count: i32 = 0;
        unsafe {
            mlx_inline_split_sections(
                buf[0].as_mut_ptr(),
                &self.raw,
                sections,
                axis,
                &mut out_count,
            );
        }
        (0..out_count as usize)
            .map(|i| unsafe {
                Self {
                    raw: buf[i].assume_init(),
                }
            })
            .collect()
    }

    /// Scatter-add: `self[indices] += updates` along `axis`.
    pub fn scatter_add_axis(&self, indices: &Self, updates: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_scatter_add(
                dst.as_mut_ptr(),
                &self.raw,
                &indices.raw,
                &updates.raw,
                axis,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }
    /// Put values at `indices` along `axis` (in-place scatter).
    pub fn put_along_axis_op(&self, indices: &Self, values: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_put_along_axis(dst.as_mut_ptr(), &self.raw, &indices.raw, &values.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }
}
