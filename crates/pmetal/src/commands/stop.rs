//! Stopping a training command on request.
//!
//! A supervisor stops a job with SIGTERM (the terminal UI, the MCP server and
//! the Python bindings all do), and a person at a terminal presses Ctrl-C. In
//! both cases a training run should finish the step it is on, save what it has
//! and exit with the interrupted status, not die with its progress. Every
//! training command takes the request from the one [`StopFlag`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pmetal_core::TrainingCallback;

/// Whether the process has been asked to stop.
///
/// Cheap to clone: every clone shares the flag. It is a
/// [`TrainingCallback`] (the trainers that take callbacks ask it after each
/// step) and hands out the bare flag ([`handle`](Self::handle)) to the loops
/// that take one.
#[derive(Clone, Debug, Default)]
pub(crate) struct StopFlag(Arc<AtomicBool>);

impl StopFlag {
    /// A flag set by the first Ctrl-C (SIGINT) or SIGTERM. A second one,
    /// because a step or a model load can take long, exits at once.
    ///
    /// Must run inside the Tokio runtime.
    pub(crate) fn watch() -> Self {
        let flag = Self::default();
        let watcher = flag.clone();
        tokio::spawn(async move {
            if stop_requested().await {
                watcher.request();
                eprintln!("Interrupted: stopping after this step (Ctrl-C again to quit now).");
                if stop_requested().await {
                    std::process::exit(130);
                }
            }
        });
        flag
    }

    /// Ask for the stop, as the signal does.
    pub(crate) fn request(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether a stop has been asked for.
    pub(crate) fn requested(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// The shared flag, for a loop that checks one itself.
    pub(crate) fn handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0)
    }

    /// The outcome of a command that has returned: its `result`, and, if it
    /// stopped because it was asked to, the interrupted status. The command
    /// has saved what it had by then; the status says the run did not finish.
    pub(crate) fn finish(&self, result: anyhow::Result<()>) -> anyhow::Result<()> {
        result?;
        if self.requested() {
            return Err(crate::Interrupted.into());
        }
        Ok(())
    }
}

impl TrainingCallback for StopFlag {
    fn should_stop(&self) -> bool {
        self.requested()
    }
}

/// What a trainer returns for a run that a stop request ended early.
///
/// The trainers that take callbacks end a run a callback asked to stop with
/// their `Cancelled` error, once the step in flight has finished. For the
/// command that is not a failure: the student, adapter or policy holds what
/// the steps so far trained and is saved like a finished run's. `None` is a
/// run that was stopped; any other error is the caller's.
pub(crate) fn unless_stopped<T, E>(
    result: Result<T, E>,
    is_stop: impl FnOnce(&E) -> bool,
) -> Result<Option<T>, E> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(e) if is_stop(&e) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Resolves when the process is asked to stop: Ctrl-C (SIGINT), or SIGTERM,
/// which is how the job supervisor stops a job.
async fn stop_requested() -> bool {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let Ok(mut term) = signal(SignalKind::terminate()) else {
            return tokio::signal::ctrl_c().await.is_ok();
        };
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.is_ok(),
            _ = term.recv() => true,
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_requested_stop_is_seen_by_every_clone_and_as_a_callback() {
        let flag = StopFlag::default();
        let callback: Box<dyn TrainingCallback> = Box::new(flag.clone());
        let handle = flag.handle();
        assert!(!flag.requested() && !callback.should_stop());

        flag.request();

        assert!(flag.requested());
        assert!(callback.should_stop());
        assert!(handle.load(Ordering::SeqCst));
    }

    #[test]
    fn a_run_a_stop_ended_is_kept_and_any_other_error_is_not() {
        let is_stop = |e: &&str| *e == "cancelled";
        assert_eq!(unless_stopped(Ok::<_, &str>(7), is_stop), Ok(Some(7)));
        assert_eq!(
            unless_stopped(Err::<i32, _>("cancelled"), is_stop),
            Ok(None)
        );
        assert_eq!(
            unless_stopped(Err::<i32, _>("out of memory"), is_stop),
            Err("out of memory")
        );
    }

    #[test]
    fn a_stopped_command_exits_as_interrupted_and_a_failed_one_as_failed() {
        let flag = StopFlag::default();
        assert!(flag.finish(Ok(())).is_ok());

        flag.request();
        let interrupted = flag.finish(Ok(())).unwrap_err();
        assert!(interrupted.is::<crate::Interrupted>());

        // A real failure is not hidden behind the interrupt.
        let failed = flag
            .finish(Err(anyhow::anyhow!("out of memory")))
            .unwrap_err();
        assert!(!failed.is::<crate::Interrupted>());
    }
}
