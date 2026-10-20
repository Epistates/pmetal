//! Python callbacks for a training run.
//!
//! A callback is any object with some of these methods (all optional):
//!
//! - `on_train_start()`, `on_train_end()`
//! - `on_epoch_start(epoch)`, `on_epoch_end(epoch, loss, perplexity)`
//! - `on_step_start(step)`, `on_step_end(step, loss)`
//! - `on_step_metrics(metrics: dict)`: the step's loss, learning rate, token
//!   rate, gradient norm and timings
//! - `on_save(path)`, `on_lr_event(event)`
//! - `should_stop() -> bool`: return true to end the run cleanly after the
//!   current step
//!
//! An exception in a callback ends the run the way `should_stop` does (the
//! current step finishes and the best weights are kept) and is raised from
//! `train()` once it has. So is Ctrl-C: it is looked for after every step.
//!
//! The built-in `ProgressCallback`, `LoggingCallback` and `MetricsJsonCallback`
//! are the trainer's own Rust callbacks and take no part in this.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pmetal_core::{StepMetrics, TrainingCallback};
use pyo3::exceptions::PyAttributeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// What the callbacks and the thread that started the run share.
#[derive(Default)]
pub struct RunState {
    /// The first exception a callback raised, or the interrupt.
    error: Mutex<Option<PyErr>>,
    /// The run should end after the current step.
    stop: AtomicBool,
}

impl RunState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record why the run is ending: the first reason wins.
    fn fail(&self, err: PyErr) {
        let mut slot = self.error.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some(err);
        }
        self.stop.store(true, Ordering::SeqCst);
    }

    fn failed(&self) -> bool {
        self.error
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_some()
    }

    /// The exception that ended the run, if one did.
    pub fn take_error(&self) -> Option<PyErr> {
        self.error.lock().unwrap_or_else(|p| p.into_inner()).take()
    }
}

/// Calls the Python callbacks, watches for Ctrl-C, and asks the loop to stop
/// when either says so. Always part of a run, even with no callbacks, for the
/// interrupt.
pub struct PythonCallbackBridge {
    callbacks: Vec<Py<PyAny>>,
    state: Arc<RunState>,
}

impl PythonCallbackBridge {
    pub fn new(callbacks: Vec<Py<PyAny>>, state: Arc<RunState>) -> Self {
        Self { callbacks, state }
    }

    /// Call `name` on each callback that has it, with the GIL held for the
    /// whole sweep. A callback that raises ends the run; the rest are skipped.
    fn dispatch(
        &self,
        name: &str,
        args: impl for<'py> Fn(Python<'py>) -> PyResult<Bound<'py, PyTuple>>,
    ) {
        if self.callbacks.is_empty() || self.state.failed() {
            return;
        }
        Python::attach(|py| {
            for callback in &self.callbacks {
                let result = (|| {
                    let method = match callback.bind(py).getattr(name) {
                        Ok(method) => method,
                        Err(e) if e.is_instance_of::<PyAttributeError>(py) => return Ok(()),
                        Err(e) => return Err(e),
                    };
                    method.call1(args(py)?).map(|_| ())
                })();
                if let Err(e) = result {
                    self.state.fail(e);
                    return;
                }
            }
        });
    }
}

/// The arguments of a callback call.
fn tuple<'py, T>(py: Python<'py>, args: T) -> PyResult<Bound<'py, PyTuple>>
where
    T: IntoPyObject<'py, Target = PyTuple, Output = Bound<'py, PyTuple>>,
{
    args.into_pyobject(py).map_err(Into::into)
}

fn metrics_dict<'py>(py: Python<'py>, m: &StepMetrics) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("step", m.step)?;
    d.set_item("epoch", m.epoch)?;
    d.set_item("total_epochs", m.total_epochs)?;
    d.set_item("total_steps", m.total_steps)?;
    d.set_item("loss", m.loss)?;
    d.set_item("lr", m.lr)?;
    d.set_item("tok_sec", m.tok_sec)?;
    d.set_item("tokens", m.tokens)?;
    d.set_item("grad_norm", m.grad_norm)?;
    d.set_item("gpu_fwd_bwd_ms", m.gpu_fwd_bwd_ms)?;
    d.set_item("optimizer_ms", m.optimizer_ms)?;
    d.set_item("total_ms", m.total_ms)?;
    Ok(d)
}

impl TrainingCallback for PythonCallbackBridge {
    fn on_train_start(&mut self) {
        self.dispatch("on_train_start", |py| tuple(py, ()));
    }

    fn on_train_end(&mut self) {
        self.dispatch("on_train_end", |py| tuple(py, ()));
    }

    fn on_epoch_start(&mut self, epoch: usize) {
        self.dispatch("on_epoch_start", |py| tuple(py, (epoch,)));
    }

    fn on_epoch_end(&mut self, epoch: usize, metrics: &pmetal_core::EvalMetrics) {
        let (loss, perplexity) = (metrics.loss, metrics.perplexity);
        self.dispatch("on_epoch_end", |py| tuple(py, (epoch, loss, perplexity)));
    }

    fn on_step_start(&mut self, step: usize) {
        self.dispatch("on_step_start", |py| tuple(py, (step,)));
    }

    fn on_step_end(&mut self, step: usize, loss: f64) {
        self.dispatch("on_step_end", |py| tuple(py, (step, loss)));
    }

    fn on_step_end_with_metrics(&mut self, metrics: &StepMetrics) {
        self.on_step_end(metrics.step, metrics.loss);
        if self.callbacks.is_empty() || self.state.failed() {
            return;
        }
        Python::attach(|py| {
            let dict = match metrics_dict(py, metrics) {
                Ok(d) => d,
                Err(e) => return self.state.fail(e),
            };
            for callback in &self.callbacks {
                let bound = callback.bind(py);
                let Ok(method) = bound.getattr("on_step_metrics") else {
                    continue;
                };
                if let Err(e) = method.call1((&dict,)) {
                    return self.state.fail(e);
                }
            }
        });
    }

    fn on_save(&mut self, path: &std::path::Path) {
        let path = path.to_string_lossy().into_owned();
        self.dispatch("on_save", |py| tuple(py, (path.as_str(),)));
    }

    fn on_lr_event(&mut self, event: &str) {
        let event = event.to_string();
        self.dispatch("on_lr_event", |py| tuple(py, (event.as_str(),)));
    }

    fn should_stop(&self) -> bool {
        if self.state.stop.load(Ordering::SeqCst) {
            return true;
        }
        Python::attach(|py| {
            // Ctrl-C. Only the main thread sees it; elsewhere this is Ok.
            if let Err(interrupt) = py.check_signals() {
                self.state.fail(interrupt);
                return true;
            }
            for callback in &self.callbacks {
                let bound = callback.bind(py);
                let Ok(method) = bound.getattr("should_stop") else {
                    continue;
                };
                match method.call0().and_then(|v| v.is_truthy()) {
                    Ok(true) => {
                        self.state.stop.store(true, Ordering::SeqCst);
                        return true;
                    }
                    Ok(false) => {}
                    Err(e) => {
                        self.state.fail(e);
                        return true;
                    }
                }
            }
            false
        })
    }
}

// ---------------------------------------------------------------------------
// The trainer's own callbacks
// ---------------------------------------------------------------------------

/// Progress bar over a run of `total_steps` steps.
#[pyclass(name = "ProgressCallback", module = "pmetal", frozen)]
pub struct PyProgressCallback {
    pub(crate) total_steps: usize,
}

#[pymethods]
impl PyProgressCallback {
    #[new]
    fn new(total_steps: usize) -> Self {
        Self { total_steps }
    }

    #[getter]
    fn total_steps(&self) -> usize {
        self.total_steps
    }

    fn __repr__(&self) -> String {
        format!("ProgressCallback(total_steps={})", self.total_steps)
    }
}

/// Logs the loss every `log_every` steps.
#[pyclass(name = "LoggingCallback", module = "pmetal", frozen)]
pub struct PyLoggingCallback {
    pub(crate) log_every: usize,
}

#[pymethods]
impl PyLoggingCallback {
    #[new]
    #[pyo3(signature = (log_every=10))]
    fn new(log_every: usize) -> Self {
        Self { log_every }
    }

    #[getter]
    fn log_every(&self) -> usize {
        self.log_every
    }

    fn __repr__(&self) -> String {
        format!("LoggingCallback(log_every={})", self.log_every)
    }
}

/// Appends each step's metrics to a JSONL file.
#[pyclass(name = "MetricsJsonCallback", module = "pmetal", frozen)]
pub struct PyMetricsJsonCallback {
    pub(crate) path: String,
}

#[pymethods]
impl PyMetricsJsonCallback {
    #[new]
    fn new(path: std::path::PathBuf) -> Self {
        Self {
            path: path.to_string_lossy().into_owned(),
        }
    }

    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    fn __repr__(&self) -> String {
        format!("MetricsJsonCallback(path={:?})", self.path)
    }
}
