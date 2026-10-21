//! The tokio runtime the bindings share, and how a Python thread waits on it.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, mpsc};
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::panic::PanicException;
use pyo3::prelude::*;

use crate::error::Exceptions;

/// How often a waiting thread looks for a signal (Ctrl-C).
const SIGNAL_POLL: Duration = Duration::from_millis(50);

/// The shared multi-thread runtime, created on first use.
pub fn runtime() -> PyResult<&'static tokio::runtime::Runtime> {
    static RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("pmetal-py-runtime")
                .build()
                .map_err(|e| format!("cannot start the async runtime: {e}"))
        })
        .as_ref()
        .map_err(|e| PyRuntimeError::new_err(e.clone()))
}

/// Run `future` on the runtime and wait for it with the GIL released.
///
/// The calling thread looks for signals every [`SIGNAL_POLL`]: on Ctrl-C the
/// task is aborted at its next await point (a download stops, its partial
/// files stay for the next call to resume) and `KeyboardInterrupt` is raised.
/// A panic inside the task is raised as `PanicException`, as one on the
/// calling thread would be.
pub fn block_on_interruptible<T, F>(py: Python<'_>, future: F) -> PyResult<T>
where
    T: Send + 'static,
    F: Future<Output = T> + Send + 'static,
{
    block_on_interruptible_with(py, future, || None)
}

/// [`block_on_interruptible`] that also asks `poll` on every tick whether the
/// task should be abandoned, with the error to raise: how a progress callback
/// that raised ends the transfer it was reporting on.
pub fn block_on_interruptible_with<T, F>(
    py: Python<'_>,
    future: F,
    mut poll: impl FnMut() -> Option<PyErr>,
) -> PyResult<T>
where
    T: Send + 'static,
    F: Future<Output = T> + Send + 'static,
{
    let rt = runtime()?;
    let mut task = rt.spawn(future);
    loop {
        let waited =
            py.detach(|| rt.block_on(async { tokio::time::timeout(SIGNAL_POLL, &mut task).await }));
        match waited {
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(join)) if join.is_panic() => {
                return Err(PanicException::new_err(panic_message(join.into_panic())));
            }
            Ok(Err(_)) => {
                return Err(PyErr::from_type(
                    Exceptions::get(py).cancelled_error.bind(py).clone(),
                    "the operation was cancelled",
                ));
            }
            Err(_elapsed) => {
                if let Err(signal) = py.check_signals() {
                    task.abort();
                    return Err(signal);
                }
                if let Some(err) = poll() {
                    task.abort();
                    return Err(err);
                }
            }
        }
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map_or_else(|| "a task panicked".to_string(), |s| (*s).to_string()),
    }
}

/// The receiving end of a channel from a worker thread. A `Receiver` is not
/// `Sync`, so a closure borrowing one cannot run with the GIL released; behind
/// a lock it can.
pub struct Inbox<T>(Mutex<mpsc::Receiver<T>>);

/// A channel from a worker thread holding at most `bound` unread messages: the
/// worker waits when the reader is behind, so it cannot run ahead of it.
pub fn inbox<T>(bound: usize) -> (mpsc::SyncSender<T>, Inbox<T>) {
    let (tx, rx) = mpsc::sync_channel(bound);
    (tx, Inbox(Mutex::new(rx)))
}

impl<T> Inbox<T> {
    fn recv_timeout(&self, wait: Duration) -> Result<T, mpsc::RecvTimeoutError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .recv_timeout(wait)
    }
}

/// Wait for the next message from a worker thread with the GIL released.
///
/// Signals are looked for on every call, and every [`SIGNAL_POLL`] of a long
/// wait: a worker that produces messages faster than they are read would
/// otherwise never leave a wait that times out. On Ctrl-C `cancel` is set (the
/// worker ends its job at the next token or step) and the next message, or the
/// worker's end, is answered with the interrupt.
pub fn recv_interruptible<T: Send>(
    py: Python<'_>,
    rx: &Inbox<T>,
    cancel: &AtomicBool,
) -> PyResult<Option<T>> {
    let mut interrupt = None;
    loop {
        if interrupt.is_none() {
            if let Err(e) = py.check_signals() {
                cancel.store(true, Ordering::SeqCst);
                interrupt = Some(e);
            }
        }
        match py.detach(|| rx.recv_timeout(SIGNAL_POLL)) {
            Ok(message) => {
                return match interrupt {
                    Some(e) => Err(e),
                    None => Ok(Some(message)),
                };
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return match interrupt {
                    Some(e) => Err(e),
                    None => Ok(None),
                };
            }
        }
    }
}
