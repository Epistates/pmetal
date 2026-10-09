//! Crate-internal raw pointer access.
//!
//! `as_raw_ptr` exposes the inline buffer to other modules in this crate that
//! dispatch to the C++ bridge directly.

use super::InlineArray;
use super::RawBuf;

impl InlineArray {
    /// Return a const raw pointer to the inline buffer (for C++ bridge calls).
    #[inline]
    pub(crate) fn as_raw_ptr(&self) -> *const RawBuf {
        &self.raw
    }
}
