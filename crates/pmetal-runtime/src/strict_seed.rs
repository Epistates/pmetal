//! A test's random draws are the same on every run.
//!
//! An unkeyed random draw takes the calling thread's global MLX key, and a
//! thread that never seeded it starts from the clock. A test that takes its
//! inputs from such a draw (a tiny model's weights, a batch) sees different
//! numbers on every run, so it passes or fails by chance and no run says which.
//!
//! Under `strict-errors`, the feature every workspace crate turns on for its
//! tests, the first unkeyed draw on a thread that has not been seeded seeds it
//! from the thread's name first. Cargo names a test's thread after the test, so
//! a test that never seeds draws the same numbers on every run and on every
//! machine, and no two tests share a stream by accident. A test that wants
//! other numbers, or a result that does not depend on its own name, calls
//! [`seed`](crate::compat::random::seed) itself, and that wins. Without the
//! feature this does nothing: library code draws from the clock as before.

#[cfg(feature = "strict-errors")]
mod imp {
    use std::cell::Cell;

    thread_local! {
        /// Whether this thread has seeded its global key.
        static SEEDED: Cell<bool> = const { Cell::new(false) };
    }

    /// The thread's global key was seeded.
    pub(crate) fn mark_seeded() {
        SEEDED.set(true);
    }

    /// FNV-1a: the same 64 bits for the same name, in every build.
    fn seed_of(name: &str) -> u64 {
        name.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Seeds the thread's global key from its name if nothing has seeded it.
    pub(crate) fn ensure_seeded() {
        if SEEDED.get() {
            return;
        }
        let thread = std::thread::current();
        crate::compat::random::seed(seed_of(thread.name().unwrap_or("<unnamed>")));
    }
}

#[cfg(not(feature = "strict-errors"))]
mod imp {
    #[inline(always)]
    pub(crate) fn mark_seeded() {}

    #[inline(always)]
    pub(crate) fn ensure_seeded() {}
}

pub(crate) use imp::{ensure_seeded, mark_seeded};
