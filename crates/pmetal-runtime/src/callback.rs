//! Rust closures that MLX calls back into.
//!
//! `value_and_grad`, `checkpoint_apply` and [`CompiledFn`](crate::CompiledFn)
//! hand MLX a Rust closure, and MLX calls it from inside a bridge call, in an
//! `extern "C"` frame. A panic that unwound out of that frame would abort the
//! process, so every trampoline runs its closure through [`guard`]: a panic is
//! caught, its payload parked in a thread-local, and the trampoline reports
//! failure to C++ by return value. C++ turns that into an exception, which
//! MLX unwinds through like any other and the bridge entry point catches;
//! once the entry point has returned, the Rust wrapper calls
//! [`resume_pending`] and the panic continues from there, with its original
//! payload, as if the closure had been called directly.

use std::any::Any;
use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

type Payload = Box<dyn Any + Send>;

thread_local! {
    /// The panic a callback on this thread has not yet re-raised.
    static PENDING: RefCell<Option<Payload>> = const { RefCell::new(None) };
}

/// A callback finished.
pub(crate) const OK: i32 = 0;
/// A callback's Rust closure panicked; the payload is parked.
pub(crate) const PANICKED: i32 = 1;

/// Run a callback's closure; [`OK`] if it returned, [`PANICKED`] if it
/// panicked (the panic is parked for [`resume_pending`]).
pub(crate) fn guard(f: impl FnOnce()) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(()) => OK,
        Err(payload) => {
            PENDING.with_borrow_mut(|slot| *slot = Some(payload));
            PANICKED
        }
    }
}

/// Whether a callback on this thread panicked and the panic is still parked.
#[cfg(feature = "strict-errors")]
pub(crate) fn is_pending() -> bool {
    PENDING.with_borrow(Option::is_some)
}

/// Re-raise a parked panic, after the bridge call that ran the callback has
/// returned. The error the C++ side recorded for the abandoned call is
/// dropped: the panic is the failure.
pub(crate) fn resume_pending() {
    if let Some(payload) = PENDING.with_borrow_mut(Option::take) {
        crate::error::clear_last_error();
        resume_unwind(payload);
    }
}
