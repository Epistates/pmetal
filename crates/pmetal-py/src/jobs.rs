//! `pmetal` commands as supervised child processes.
//!
//! The terminal UI, the MCP server and the desktop app run their jobs under
//! `pmetal-supervisor`, and so do these bindings: the job leads a process group
//! of its own (a stop reaches what it spawned), its output is redacted and
//! bounded, its progress is read from the files it writes, and a stop is
//! SIGTERM, then SIGKILL after a grace period. This module is the glue between
//! a supervisor and Python: the `_Job` handle the package's `Job` wraps (see
//! `py_src/pmetal/_jobs.py`).
//!
//! Each job has a supervisor of its own, so its event channel carries nothing
//! else and a busy neighbour cannot make it lag.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use pmetal_core::sync::locked;
use pmetal_supervisor::{
    Event, EventKind, JobId, JobRequest, Limits, ProcessStatus, Stream, Supervisor,
};
use pyo3::prelude::*;
use serde_json::Value;
use tokio::sync::broadcast::error::RecvError;

use crate::error::config_err;
use crate::json::{to_py, value_of};
use crate::runtime::runtime;

/// The most of a job's standard output kept whole (for `JobResult.json()`).
const MAX_STDOUT: usize = 64 * 1024 * 1024;
/// The most of its standard error returned.
const MAX_STDERR: usize = 64 * 1024;
/// How often a waiting thread looks for Ctrl-C.
const SIGNAL_POLL: Duration = Duration::from_millis(50);

/// The jobs running, for the interpreter's exit. A job a program started and
/// let go of keeps running; one still running when the program ends is stopped.
static LIVE: Mutex<Option<HashMap<JobId, Supervisor>>> = Mutex::new(None);

fn live<R>(f: impl FnOnce(&mut HashMap<JobId, Supervisor>) -> R) -> R {
    f(locked(&LIVE).get_or_insert_with(HashMap::new))
}

#[derive(Default)]
struct Stdout {
    bytes: Vec<u8>,
    /// Everything the job wrote is in `bytes` (no event was missed, nothing
    /// was cut at the cap).
    complete: bool,
}

/// What a job's collector task and its Python handle share.
struct Core {
    supervisor: Supervisor,
    id: JobId,
    binary: PathBuf,
    events: Mutex<Vec<Value>>,
    stdout: Mutex<Stdout>,
    on_event: Option<Py<PyAny>>,
    failure: Mutex<Option<PyErr>>,
    finished: (Mutex<bool>, Condvar),
    /// Holds the `--log-events` file; removed when the job is gone.
    _events_dir: Option<Scratch>,
}

/// A directory removed when this is dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl Core {
    fn push_stdout(&self, text: &str) {
        let mut out = locked(&self.stdout);
        if out.bytes.len() + text.len() > MAX_STDOUT {
            out.complete = false;
            return;
        }
        out.bytes.extend_from_slice(text.as_bytes());
    }

    fn job_event(&self, event: &pmetal_core::events::JobEvent) {
        let Ok(value) = value_of(event) else {
            return;
        };
        locked(&self.events).push(value.clone());
        let Some(callback) = &self.on_event else {
            return;
        };
        if locked(&self.failure).is_some() {
            return;
        }
        Python::attach(|py| {
            let result = to_py(py, &value).and_then(|dict| callback.bind(py).call1((dict,)));
            if let Err(e) = result {
                *locked(&self.failure) = Some(e);
            }
        });
    }

    fn finish(&self) {
        live(|jobs| jobs.remove(&self.id));
        *locked(&self.finished.0) = true;
        self.finished.1.notify_all();
    }

    fn is_finished(&self) -> bool {
        *locked(&self.finished.0)
    }
}

/// Follow a job's events until it exits.
async fn collect(core: Arc<Core>, mut events: tokio::sync::broadcast::Receiver<Event>) {
    loop {
        match events.recv().await {
            Ok(Event { job, kind }) if job == core.id => match kind {
                EventKind::Delta(text) => core.push_stdout(&text),
                EventKind::Job(event) => core.job_event(&event),
                EventKind::Exited(_) => break,
                EventKind::Started | EventKind::Output(_) => {}
            },
            Ok(_) => {}
            // Deltas were lost: the whole standard output is no longer in hand.
            Err(RecvError::Lagged(_)) => locked(&core.stdout).complete = false,
            Err(RecvError::Closed) => break,
        }
    }
    core.finish();
}

/// A running or finished job.
#[pyclass(name = "_Job", module = "pmetal", frozen)]
pub struct PyJob {
    core: Arc<Core>,
}

fn status_parts(status: &ProcessStatus) -> (&'static str, Option<i32>, Option<String>) {
    match status {
        ProcessStatus::Running => ("running", None, None),
        ProcessStatus::Stopping => ("stopping", None, None),
        ProcessStatus::Completed { exit_code } => ("completed", Some(*exit_code), None),
        ProcessStatus::Failed { exit_code, error } => {
            ("failed", Some(*exit_code), Some(error.clone()))
        }
        ProcessStatus::Stopped => ("stopped", None, None),
    }
}

#[pymethods]
impl PyJob {
    /// The supervisor's id for the job.
    #[getter]
    fn id(&self) -> &str {
        &self.core.id
    }

    /// The command that runs, with credentials hidden.
    #[getter]
    fn command(&self) -> PyResult<Vec<String>> {
        let summary = self
            .core
            .supervisor
            .summary(&self.core.id)
            .map_err(config_err)?;
        let mut command = vec![self.core.binary.display().to_string()];
        command.extend(summary.args);
        Ok(command)
    }

    /// `(state, exit_code, error)`: `state` is `running`, `stopping`,
    /// `completed`, `failed` or `stopped`; the exit code is -1 when a signal
    /// ended the process, and None while it runs or when it was stopped.
    fn status(&self) -> PyResult<(&'static str, Option<i32>, Option<String>)> {
        let summary = self
            .core
            .supervisor
            .summary(&self.core.id)
            .map_err(config_err)?;
        Ok(status_parts(&summary.status))
    }

    /// Wait for the job to end and its events to be in, with the GIL released.
    /// True when it has; False when `timeout` seconds passed first. Ctrl-C
    /// raises `KeyboardInterrupt` (the job keeps running).
    #[pyo3(signature = (timeout=None))]
    fn wait(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<bool> {
        let deadline = timeout.map(|t| Instant::now() + Duration::from_secs_f64(t.max(0.0)));
        loop {
            py.check_signals()?;
            let core = &self.core;
            let done = py.detach(|| {
                let guard = locked(&core.finished.0);
                let (guard, _) = core
                    .finished
                    .1
                    .wait_timeout_while(guard, SIGNAL_POLL, |finished| !*finished)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                *guard
            });
            if done {
                return Ok(true);
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(false);
            }
        }
    }

    /// Whether the job has ended and its events are in.
    fn finished(&self) -> bool {
        self.core.is_finished()
    }

    /// Stop the job: SIGTERM to its process group, SIGKILL after the grace
    /// period. A job that has ended is left alone.
    fn stop(&self) -> PyResult<()> {
        // The supervisor times the kill on the runtime.
        let _inside = runtime()?.enter();
        let _ = self.core.supervisor.stop(&self.core.id);
        Ok(())
    }

    /// Seconds after a stop by which the job is certain to be gone.
    fn stop_deadline(&self) -> f64 {
        self.core.supervisor.stop_deadline().as_secs_f64()
    }

    /// The events from the `since`-th on, as dicts.
    #[pyo3(signature = (since=0))]
    fn events<'py>(&self, py: Python<'py>, since: usize) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let events = locked(&self.core.events);
        events
            .iter()
            .skip(since)
            .map(|value| to_py(py, value))
            .collect()
    }

    /// The job's standard output, whole when it fits, else what the
    /// supervisor kept of it.
    fn stdout(&self) -> String {
        {
            let out = locked(&self.core.stdout);
            if out.complete {
                return String::from_utf8_lossy(&out.bytes).into_owned();
            }
        }
        self.lines(Stream::Stdout, usize::MAX)
    }

    /// The end of the job's standard error.
    fn stderr(&self) -> String {
        self.lines(Stream::Stderr, MAX_STDERR)
    }

    /// The last of a job's output lines, as `(stream, text)` pairs from
    /// sequence `since`.
    #[pyo3(signature = (tail=200, since=0))]
    fn logs(&self, tail: usize, since: u64) -> PyResult<Vec<(&'static str, String)>> {
        let page = self
            .core
            .supervisor
            .logs(&self.core.id, tail, Some(since))
            .map_err(config_err)?;
        Ok(page
            .entries
            .into_iter()
            .map(|line| {
                let stream = match line.stream {
                    Stream::Stdout => "stdout",
                    Stream::Stderr => "stderr",
                };
                (stream, line.text)
            })
            .collect())
    }

    /// What the job reported about its progress (last step, loss, ...), as a dict.
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let summary = self
            .core
            .supervisor
            .summary(&self.core.id)
            .map_err(config_err)?;
        to_py(py, &value_of(&summary.metrics).map_err(config_err)?)
    }

    /// What the job said when it finished successfully, if it said anything.
    fn completion<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let completion = self
            .core
            .supervisor
            .completion(&self.core.id)
            .map_err(config_err)?;
        completion
            .map(|c| to_py(py, &value_of(&c).map_err(config_err)?))
            .transpose()
    }

    /// Raise the exception an `on_event` callback raised, if one did (once).
    fn raise_callback_error(&self) -> PyResult<()> {
        match locked(&self.core.failure).take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl PyJob {
    fn lines(&self, stream: Stream, limit: usize) -> String {
        let Ok(page) = self
            .core
            .supervisor
            .logs(&self.core.id, usize::MAX, Some(0))
        else {
            return String::new();
        };
        let mut text = page
            .entries
            .into_iter()
            .filter(|line| line.stream == stream)
            .map(|line| line.text)
            .collect::<Vec<_>>()
            .join("\n");
        if text.len() > limit {
            let mut cut = text.len() - limit;
            while !text.is_char_boundary(cut) {
                cut += 1;
            }
            text.drain(..cut);
        }
        text
    }
}

/// Start `pmetal <subcommand> <argv>` under a supervisor and return its handle.
///
/// `argv` and `events_file` are what a spec's `_launch` gave. With an
/// `events_file` (a file of the `JobEvent` stream the command was told to write,
/// `--log-events`) the supervisor follows it and the events reach `on_event` as
/// they are written; `events_dir`, the directory made for it, is removed when
/// the job is gone. `stop_grace` is the seconds a stopped job has to exit
/// before it is killed.
#[pyfunction]
#[pyo3(signature = (binary, subcommand, argv, events_file=None, events_dir=None, stop_grace=30.0, on_event=None))]
pub fn _start_job(
    binary: PathBuf,
    subcommand: String,
    argv: Vec<String>,
    events_file: Option<PathBuf>,
    events_dir: Option<PathBuf>,
    stop_grace: f64,
    on_event: Option<Py<PyAny>>,
) -> PyResult<PyJob> {
    let rt = runtime()?;
    let events_dir = events_dir.map(Scratch);

    let supervisor = Supervisor::new(
        binary.clone(),
        Limits {
            max_running: 1,
            max_finished: 1,
            stop_grace: Duration::from_secs_f64(stop_grace.max(0.0)),
        },
    );
    // Subscribed first, so nothing the job does is missed.
    let receiver = supervisor.subscribe();
    let id = {
        let _inside = rt.enter();
        supervisor
            .spawn(JobRequest {
                subcommand,
                argv,
                events_file,
                // Standard output arrives whole as it is written, for a job whose
                // output is its result (`pmetal infer --json`).
                deltas: true,
                ..JobRequest::default()
            })
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?
    };
    let core = Arc::new(Core {
        supervisor: supervisor.clone(),
        id: id.clone(),
        binary,
        events: Mutex::new(Vec::new()),
        stdout: Mutex::new(Stdout {
            bytes: Vec::new(),
            complete: true,
        }),
        on_event,
        failure: Mutex::new(None),
        finished: (Mutex::new(false), Condvar::new()),
        _events_dir: events_dir,
    });
    live(|jobs| jobs.insert(id, supervisor));
    rt.spawn(collect(core.clone(), receiver));
    Ok(PyJob { core })
}

/// Stop every job still running, for the interpreter's exit: none outlives the
/// program that started it.
#[pyfunction]
pub fn _stop_all_jobs() {
    let Ok(rt) = runtime() else {
        return;
    };
    let supervisors: Vec<Supervisor> = live(|jobs| jobs.values().cloned().collect());
    for supervisor in supervisors {
        rt.block_on(supervisor.shutdown());
    }
}
