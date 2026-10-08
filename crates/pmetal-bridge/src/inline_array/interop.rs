//! Interop with mlx-rs and crate-internal raw pointer accessors.
//!
//! The `from_raw_ctx` / `to_raw_ctx` pair bridges `InlineArray` and mlx-rs
//! `Array` during the migration period. `as_raw_ptr` / `as_raw_ptr_mut` expose
//! the inline buffer to other modules in this crate that dispatch to the C++
//! bridge directly.

use std::mem::MaybeUninit;

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;

impl InlineArray {
    // ── Interop with mlx-rs (transition period) ──────────────────────────

    /// Create from an opaque mlx-rs array context pointer.
    ///
    /// # Safety
    /// `ctx` must be a valid `mlx::core::array*` as returned by
    /// `mlx_array { ctx }` from the mlx-c / mlx-rs layer.  The C++ side
    /// copies (ref-counts) the array, so the caller retains ownership of the
    /// original handle.
    ///
    /// Typical usage during migration:
    /// ```ignore
    /// let inline = InlineArray::from_raw_ctx(arr.as_ptr().ctx);
    /// ```
    pub unsafe fn from_raw_ctx(ctx: *mut std::ffi::c_void) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_handle(dst.as_mut_ptr(), ctx);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Export as an opaque heap-allocated `mlx::core::array*` context pointer.
    ///
    /// The caller is responsible for freeing the returned pointer via the
    /// mlx-c `mlx_array_free` mechanism (or by wrapping in an `mlx_array`
    /// handle and passing to `mlx_array_free`).
    ///
    /// Typical usage during migration:
    /// ```ignore
    /// let ctx = inline.to_raw_ctx();
    /// let handle = mlx_sys::mlx_array { ctx };
    /// let arr = unsafe { mlx_rs::Array::from_ptr(handle) };
    /// ```
    pub fn to_raw_ctx(&self) -> *mut std::ffi::c_void {
        unsafe { mlx_inline_to_handle(&self.raw) }
    }

    // ── Raw pointer access (crate-internal, for C++ bridge) ──────────────

    /// Return a const raw pointer to the inline buffer (for C++ bridge calls).
    #[inline]
    pub(crate) fn as_raw_ptr(&self) -> *const RawBuf {
        &self.raw
    }
}
