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
    fn pmetal_bridge_unobserved_error_code() -> i32;
    fn pmetal_bridge_unobserved_error_message() -> *const std::os::raw::c_char;
    fn pmetal_bridge_set_error_log_mode(enabled: i32);
    fn pmetal_bridge_get_error_log_mode() -> i32;
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

/// Reports the first error a bridge op recorded on this thread since one was
/// last observed, and clears it.
///
/// [`check_last_error`] reports the most recent op only: every op that
/// succeeds clears its slot. MLX builds its graph lazily, so a computation
/// carries on past an op that threw, on the placeholder the bridge put in its
/// place, and by the time it is checked the slot is clean again. A training
/// step is such a computation: one op failing inside `value_and_grad` leaves
/// a step that finishes, reports a loss, and moved nothing. Call this at the
/// end of one instead; any error [`check_last_error`] has already reported,
/// or [`clear_last_error`] discarded, doesn't count.
pub fn check_unobserved_error() -> BridgeResult<()> {
    // SAFETY: as in `check_last_error`: a thread-local read, and the message
    // is copied before `pmetal_bridge_clear_error` invalidates it.
    unsafe {
        let code = pmetal_bridge_unobserved_error_code();
        if code == 0 {
            return Ok(());
        }
        let raw = pmetal_bridge_unobserved_error_message();
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

/// Toggle the bridge's `stderr` emission on caught C++ exceptions.
///
/// When enabled, every exception caught inside the bridge's BRIDGE_TRY
/// wrappers prints a `[pmetal-bridge] exception in [op]: what()` line
/// alongside setting the thread-local error slot. This makes the *first*
/// failure visible even in code paths that don't call [`check_last_error`]
/// after every op — the bridge's exception path replaces the failed op's
/// output with a scalar-zero sentinel tensor, so without a log line the
/// error first surfaces as a shape panic three or four ops later in
/// unrelated code.
///
/// Default: enabled in debug builds, disabled in release. Can also be
/// overridden at process start via the `PMETAL_BRIDGE_LOG_ERRORS` env var
/// (`1`/`0`/`true`/`false`).
pub fn set_error_log_mode(enabled: bool) {
    unsafe { pmetal_bridge_set_error_log_mode(enabled as i32) };
}

/// Returns the current `stderr` emission mode.
///
/// See [`set_error_log_mode`] for the semantics.
pub fn error_log_mode() -> bool {
    unsafe { pmetal_bridge_get_error_log_mode() != 0 }
}

/// Runs `f` as code that expects a bridge op to fail.
///
/// Test builds turn on the `strict-errors` feature, under which a bridge op
/// whose C++ side throws panics the calling thread at that op. A test that
/// exercises an error path on purpose, or code that tries an op and handles
/// its failure, runs that op inside this scope: errors raised in it stay in
/// the error channel for [`check_last_error`] and [`check_unobserved_error`]
/// to report, as they do in a build without the feature, and none of them
/// panics, then or at a later op. Scopes nest. Without `strict-errors` this
/// just calls `f`.
pub fn expect_bridge_errors<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(feature = "strict-errors")]
    let _expected = strict::Expected::enter();
    f()
}

/// The `strict-errors` runtime: what each bridge entry point runs around its
/// C call (see `bridge_extern!`).
///
/// The panic is raised on the Rust side, after the C call has returned, so
/// it never unwinds through a C++ frame. A Rust callback that MLX calls from
/// inside a bridge call (a traced loss, a compiled closure) runs in an
/// `extern "C"` frame, where a panic would abort the process, so a nested
/// call never panics: the outermost call reports its error when MLX returns.
/// An error that Rust code drained before then (a callback that tried an op
/// and fell back) is not reported.
#[cfg(feature = "strict-errors")]
pub(crate) mod strict {
    use std::cell::Cell;

    unsafe extern "C" {
        fn pmetal_bridge_error_seq() -> u64;
    }

    thread_local! {
        /// Bridge calls in progress on this thread (above one: a callback).
        static DEPTH: Cell<u32> = const { Cell::new(0) };
        /// Open `expect_bridge_errors` scopes.
        static EXPECTED: Cell<u32> = const { Cell::new(0) };
        /// The error count the last outermost call ended on.
        static SEEN: Cell<u64> = const { Cell::new(0) };
    }

    fn error_seq() -> u64 {
        // SAFETY: a thread-local read on the C side.
        unsafe { pmetal_bridge_error_seq() }
    }

    /// One bridge call: [`Call::begin`] before the C call, [`Call::end`]
    /// after it.
    pub(crate) struct Call {
        outermost: bool,
    }

    impl Call {
        #[inline]
        pub(crate) fn begin() -> Self {
            let depth = DEPTH.get();
            DEPTH.set(depth + 1);
            Self {
                outermost: depth == 0,
            }
        }

        /// Panics if the call raised an error that nothing has observed.
        #[inline]
        #[track_caller]
        pub(crate) fn end(self, op: &'static str) {
            DEPTH.set(DEPTH.get() - 1);
            if !self.outermost || std::thread::panicking() {
                return;
            }
            let seq = error_seq();
            if seq == SEEN.get() {
                return;
            }
            SEEN.set(seq);
            if EXPECTED.get() > 0 {
                return;
            }
            // The first error since one was last observed, else the call's
            // own; neither when a callback drained it.
            let pending = super::check_unobserved_error().and_then(|()| super::check_last_error());
            if let Err(err) = pending {
                panic!(
                    "bridge op `{op}` failed: {err}\n\
                     (`strict-errors` fails a test at every bridge error; run an op that \
                     is meant to fail inside `pmetal_bridge::expect_bridge_errors`)"
                );
            }
        }
    }

    /// An open [`super::expect_bridge_errors`] scope.
    pub(super) struct Expected(());

    impl Expected {
        pub(super) fn enter() -> Self {
            EXPECTED.set(EXPECTED.get() + 1);
            Self(())
        }
    }

    impl Drop for Expected {
        fn drop(&mut self) {
            EXPECTED.set(EXPECTED.get() - 1);
            // Whichever call raised them, errors from inside the scope were
            // expected.
            SEEN.set(error_seq());
        }
    }
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
        let v = expect_bridge_errors(|| two_elem.item_f32());
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

    /// An op that threw and was followed by ops that succeeded leaves the
    /// last-op slot clean, and the unobserved one holding the failure.
    #[test]
    fn an_error_followed_by_successful_ops_stays_unobserved() {
        use crate::InlineArray;

        clear_last_error();
        let two_elem = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
        let _ = expect_bridge_errors(|| two_elem.item_f32());
        let later = two_elem.add(&two_elem);
        let _ = later.sum_all();

        assert_eq!(check_last_error(), Ok(()), "a later op cleared the slot");
        let err = check_unobserved_error().expect_err("the failure is still there");
        assert!(err.to_string().contains("item_f32"), "{err}");
        assert_eq!(check_unobserved_error(), Ok(()), "reading it cleared it");

        // An error the caller observed doesn't count again.
        let _ = expect_bridge_errors(|| two_elem.item_f32());
        assert!(check_last_error().is_err());
        assert_eq!(check_unobserved_error(), Ok(()));
    }

    /// Calls that are not ops (memory queries, syncs, scalar constructors) run
    /// between an op and its error check, so a success there must not erase
    /// the op's error.
    #[test]
    fn non_op_calls_leave_a_pending_error_in_place() {
        use crate::InlineArray;
        use crate::inline_array::{clear_cache, get_active_memory, synchronize};

        clear_last_error();
        let two_elem = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
        let _ = expect_bridge_errors(|| two_elem.item_f32());

        let _ = get_active_memory();
        synchronize();
        clear_cache();
        let _ = InlineArray::from_f32(1.0);

        let err = check_last_error().expect_err("the item_f32 error is still pending");
        assert!(
            format!("{err:?}").contains("item_f32"),
            "wrong error: {err:?}"
        );
    }

    /// Under `strict-errors`, the op that throws panics, naming itself, and
    /// leaves the channel clean.
    #[cfg(feature = "strict-errors")]
    #[test]
    fn strict_mode_panics_at_the_op_that_throws() {
        use crate::InlineArray;

        let two_elem = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
        let panic = std::panic::catch_unwind(|| two_elem.item_f32())
            .expect_err("a throwing op must panic in a strict build");
        let msg = panic
            .downcast_ref::<String>()
            .expect("a formatted panic message");
        assert!(msg.contains("mlx_inline_item_f32"), "{msg}");
        assert!(msg.contains("[item_f32]"), "{msg}");
        assert_eq!(check_last_error(), Ok(()));
        assert_eq!(check_unobserved_error(), Ok(()));
    }

    /// An op that throws inside a callback MLX runs (here a traced loss)
    /// can't panic there, inside an `extern "C"` frame; the bridge call that
    /// ran the callback panics once it returns.
    #[cfg(feature = "strict-errors")]
    #[test]
    fn strict_mode_reports_an_error_from_a_callback_at_the_outer_call() {
        use crate::InlineArray;
        use crate::inline_array::value_and_grad;

        let w = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
        let bad = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0], &[3]);
        let panic = std::panic::catch_unwind(|| {
            value_and_grad(
                |p: &[InlineArray]| {
                    // [2] x [3]: a shape error, raised while tracing.
                    let _ = p[0].matmul(&bad);
                    p[0].sum_all()
                },
                std::slice::from_ref(&w),
                &[],
            )
        })
        .expect_err("the traced op's error must fail the outer call");
        let msg = panic.downcast_ref::<String>().expect("a panic message");
        assert!(msg.contains("mlx_inline_value_and_grad"), "{msg}");
        assert!(msg.contains("[matmul]"), "{msg}");
    }

    /// A callback that tries an op and handles its failure has observed the
    /// error, so the outer call doesn't report it.
    #[cfg(feature = "strict-errors")]
    #[test]
    fn strict_mode_ignores_an_error_a_callback_handled() {
        use crate::InlineArray;
        use crate::inline_array::value_and_grad;

        let w = InlineArray::from_f32_slice(&[1.0, 2.0], &[2]);
        let bad = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0], &[3]);
        // No asserting inside the callback: a panic there aborts.
        let handled = std::cell::Cell::new(false);
        let (loss, mut grads) = value_and_grad(
            |p: &[InlineArray]| {
                let _ = p[0].matmul(&bad);
                handled.set(check_last_error().is_err());
                p[0].sum_all()
            },
            std::slice::from_ref(&w),
            &[],
        );
        assert!(handled.get(), "the callback saw the matmul fail");
        assert_eq!(loss.item_f32(), 3.0);
        assert_eq!(grads[0].to_f32_vec(2), Some(vec![1.0, 1.0]));
    }
}
