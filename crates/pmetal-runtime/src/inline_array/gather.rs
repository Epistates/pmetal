//! Indexing and embedding lookups: gather_mm, argpartition, take_along_axis,
//! take_axis (embedding lookup), and kv_cache_append (sequence-axis concat).

use std::mem::MaybeUninit;

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;

impl InlineArray {
    // ── Gather / MoE ─────────────────────────────────────────────────────

    pub fn gather_mm(
        &self,
        b: &Self,
        lhs: Option<&Self>,
        rhs: Option<&Self>,
        sorted: bool,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_gather_mm(
                dst.as_mut_ptr(),
                &self.raw,
                &b.raw,
                lhs.map_or(std::ptr::null(), |a| &a.raw),
                rhs.map_or(std::ptr::null(), |a| &a.raw),
                sorted,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn argpartition(&self, kth: i32, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_argpartition(dst.as_mut_ptr(), &self.raw, kth, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn take_along_axis(&self, indices: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_take_along_axis(dst.as_mut_ptr(), &self.raw, &indices.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Embedding / KV cache ────────────────────────────────────────────

    /// Take rows along axis (embedding lookup: `take(weight, indices, axis=0)`).
    #[inline]
    pub fn take_axis(&self, indices: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_take_axis(dst.as_mut_ptr(), &self.raw, &indices.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Concatenate cached and new K/V along the sequence axis.
    #[inline]
    pub fn kv_cache_append(&self, new_kv: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_kv_cache_append(dst.as_mut_ptr(), &self.raw, &new_kv.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }
}
