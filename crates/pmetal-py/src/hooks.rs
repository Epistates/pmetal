//! Hooks for the tests (the `test-hooks` feature; never in a published wheel).

use pyo3::prelude::*;

use crate::runtime::block_on_interruptible;

/// Panic on the calling thread.
#[pyfunction]
pub fn _panic(message: &str) {
    panic!("{message}");
}

/// Panic in a task on the shared runtime, as a Hub call's future could.
#[pyfunction]
pub fn _panic_in_task(py: Python<'_>, message: String) -> PyResult<()> {
    block_on_interruptible(py, async move {
        if !message.is_empty() {
            panic!("{message}");
        }
    })
}

/// Whether a panic unwinds in this build (it does not under `panic = "abort"`).
#[pyfunction]
pub fn _panic_unwinds() -> bool {
    cfg!(panic = "unwind")
}
