//! `--log-events`: the run's `JobEvent` stream, one JSON line each, in a file.
//!
//! The TUI's subprocess runner and the MCP server tail the file; a script can
//! too. A write that fails (a full disk, a closed pipe) cannot stop a training
//! run that is hours in, so the first one is kept and reported when the run
//! ends, as an error: a consumer that was promised the stream must not read a
//! silently truncated one as a complete log.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use pmetal_core::{JsonlSink, TrainingCallback, TrainingCallbackToSink};

/// An open event log.
pub(crate) struct EventLog<W: Write + Send + Sync + 'static = File> {
    sink: Arc<JsonlSink<W>>,
    path: PathBuf,
}

impl EventLog<File> {
    /// Create (or truncate) the file, so a bad path fails before the run
    /// starts rather than partway through it and a reader can tail it at once.
    pub(crate) fn create(path: &Path) -> anyhow::Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("--log-events: could not create '{}'", path.display()))?;
        Ok(Self::new(file, path))
    }
}

impl<W: Write + Send + Sync + 'static> EventLog<W> {
    fn new(writer: W, path: &Path) -> Self {
        Self {
            sink: Arc::new(JsonlSink::new(writer)),
            path: path.to_path_buf(),
        }
    }

    /// The trainer callback that writes the run's events. `job_id` names the
    /// run in the stream.
    pub(crate) fn callback(&self, job_id: &str) -> Box<dyn TrainingCallback> {
        Box::new(TrainingCallbackToSink::new(job_id, Arc::clone(&self.sink)))
    }

    /// The run's result, with a failed log write reported when the run itself
    /// succeeded. A run that failed keeps its own error: the log is a
    /// by-product of it, and the write error is only noted.
    pub(crate) fn finish<T>(self, result: anyhow::Result<T>) -> anyhow::Result<T> {
        let Some(write_error) = self.sink.take_error() else {
            return result;
        };
        match result {
            Ok(_) => Err(anyhow::anyhow!(
                "--log-events: writing '{}' failed: {write_error}. The run itself finished, \
                 but the event log is incomplete.",
                self.path.display()
            )),
            Err(run_error) => {
                tracing::warn!(
                    path = %self.path.display(),
                    "the event log is incomplete: {write_error}"
                );
                Err(run_error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_core::{JobEvent, JobEventSink, JobKind};

    /// A writer that fails every write after the first.
    struct FullDisk(usize);

    impl Write for FullDisk {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0 += 1;
            if self.0 > 1 {
                return Err(std::io::Error::other("no space left on device"));
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn emit_two(log: &EventLog<FullDisk>) {
        log.sink.emit(JobEvent::started("j", JobKind::Train));
        log.sink.emit(JobEvent::failed("j", "boom"));
    }

    #[test]
    fn a_failed_write_fails_a_run_that_otherwise_finished() {
        let log = EventLog::new(FullDisk(0), Path::new("events.jsonl"));
        emit_two(&log);
        let err = log.finish(Ok(())).unwrap_err().to_string();
        assert!(err.contains("events.jsonl"), "{err}");
        assert!(err.contains("no space left"), "{err}");
        assert!(err.contains("incomplete"), "{err}");
    }

    #[test]
    fn a_failed_run_keeps_its_own_error() {
        let log = EventLog::new(FullDisk(0), Path::new("events.jsonl"));
        emit_two(&log);
        let err = log
            .finish::<()>(Err(anyhow::anyhow!("out of memory")))
            .unwrap_err()
            .to_string();
        assert_eq!(err, "out of memory");
    }

    #[test]
    fn a_clean_log_changes_nothing() {
        let log = EventLog::new(Vec::new(), Path::new("events.jsonl"));
        log.sink.emit(JobEvent::started("j", JobKind::Train));
        assert_eq!(log.finish(Ok(7)).unwrap(), 7);
    }

    #[test]
    fn an_unwritable_path_is_refused_before_the_run() {
        let err = match EventLog::create(Path::new("/nonexistent-dir/events.jsonl")) {
            Ok(_) => panic!("created a log in a directory that does not exist"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("--log-events"), "{err}");
        assert!(err.contains("/nonexistent-dir/events.jsonl"), "{err}");
    }
}
