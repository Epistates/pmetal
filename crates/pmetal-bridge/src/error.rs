//! Error-propagation helpers for the C++ bridge.
//!
//! Every `mlx_inline_*` entry point that can throw a C++ exception writes
//! its failure into a thread-local error slot on the C++ side instead of
//! calling `std::terminate` (or, worse, returning a silently-zeroed array
//! that looks like success). This module exposes that channel to Rust.
//!
//! ## Usage
//!
//! Most bridge ops still have void return types for ABI stability with the
//! existing Rust surface. To detect failure, call [`check_last_error`]
//! immediately after the op:
//!
//! ```ignore
//! let mut out = InlineArray::new_empty();
//! unsafe { mlx_inline_matmul(&mut out.raw, &a.raw, &b.raw) };
//! pmetal_bridge::check_last_error()?;   // ← propagate any C++ exception
//! ```
//!
//! The thread-local slot is cleared by every successful op, so a stale
//! error from an earlier call can't shadow a later success. The pointer
//! returned by the C-side `pmetal_bridge_last_error_message` is valid
//! only until the next bridge call on the same thread — this module copies
//! it into an owned `String` before handing it out.
//!
//! ## Roadmap
//!
//! This module intentionally keeps the existing (infallible-looking) ABI
//! unchanged so downstream crates don't have to migrate in a single
//! explosion. Later work introduces `try_*` variants on `InlineArray` that
//! wrap the existing ops + `check_last_error` in a `BridgeResult<Self>`.

use std::ffi::CStr;
use std::fmt;

unsafe extern "C" {
    fn pmetal_bridge_last_error_code() -> i32;
    fn pmetal_bridge_last_error_message() -> *const std::os::raw::c_char;
    fn pmetal_bridge_clear_error();
}

/// Error type for faults reported by the bridge's C++ layer.
///
/// The [`CxxException`] variant carries the `what()` text from the caught
/// `std::exception`; [`Unknown`] is reserved for `catch (...)` matches on
/// non-std exception types (rare but possible for third-party code).
///
/// [`CxxException`]: Self::CxxException
/// [`Unknown`]: Self::Unknown
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// Caught a `std::exception` subclass.
    CxxException(String),
    /// Caught `...` — unknown C++ exception type.
    Unknown(String),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CxxException(msg) => write!(f, "bridge C++ exception: {msg}"),
            Self::Unknown(msg) => write!(f, "bridge unknown C++ exception: {msg}"),
        }
    }
}

impl std::error::Error for BridgeError {}

/// Alias for `Result<T, BridgeError>`.
pub type BridgeResult<T> = std::result::Result<T, BridgeError>;

/// Reads and clears the thread-local error slot set by the most recent
/// bridge call on this thread.
///
/// Returns `Ok(())` when no error is pending. When an error is present,
/// the slot is cleared before returning so repeated reads don't alias.
pub fn check_last_error() -> BridgeResult<()> {
    // SAFETY: pmetal_bridge_last_error_code is a thread-local read with no
    // pointer arithmetic; pmetal_bridge_last_error_message returns a pointer
    // into a thread-local std::string whose contents are stable until the
    // next bridge call on this thread. We copy the bytes before
    // pmetal_bridge_clear_error invalidates them.
    unsafe {
        let code = pmetal_bridge_last_error_code();
        if code == 0 {
            return Ok(());
        }
        let raw = pmetal_bridge_last_error_message();
        let msg = if raw.is_null() {
            String::new()
        } else {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        };
        pmetal_bridge_clear_error();
        match code {
            2 => Err(BridgeError::Unknown(msg)),
            _ => Err(BridgeError::CxxException(msg)),
        }
    }
}

/// Clears any pending thread-local error without returning it.
///
/// Use sparingly — most callers should [`check_last_error`] and propagate.
/// Handy when entering a retry loop where a prior failure is expected and
/// must not leak into the next call's success check.
pub fn clear_last_error() {
    unsafe { pmetal_bridge_clear_error() };
}

#[cfg(test)]
mod tests {
    use super::*;

    // The C++ side is linked at crate-load time; these tests just make sure
    // the symbols exist and that the "no error" path is wired correctly.
    #[test]
    fn no_error_returns_ok() {
        clear_last_error();
        assert_eq!(check_last_error(), Ok(()));
    }

    #[test]
    fn clear_is_idempotent() {
        clear_last_error();
        clear_last_error();
        assert_eq!(check_last_error(), Ok(()));
    }

    // End-to-end: trigger a real C++ exception and confirm the channel
    // delivers it as a `BridgeError::CxxException` on this thread. We use
    // `item_f32` on a 2-element array — MLX's `array::item<T>()` throws
    // when the array isn't a scalar, so this is a reliable throw site.
    #[test]
    fn catches_real_cxx_exception() {
        use crate::InlineArray;

        clear_last_error();
        let two_elem = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
        // The C++ side catches the std::runtime_error("item can only be called on a scalar.")
        // into the thread-local slot; Rust sees the 0.0f sentinel.
        let v = two_elem.item_f32();
        assert_eq!(v, 0.0);

        let err = check_last_error().expect_err("item on non-scalar should surface an error");
        match err {
            BridgeError::CxxException(msg) => {
                assert!(msg.contains("item_f32"), "op tag missing: {msg}");
            }
            BridgeError::Unknown(msg) => {
                panic!("expected CxxException, got Unknown: {msg}");
            }
        }

        // Slot must have been cleared by check_last_error.
        assert_eq!(check_last_error(), Ok(()));
    }
}
