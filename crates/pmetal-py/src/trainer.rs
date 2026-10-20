//! `Trainer`: a `TrainSpec` run in this process.
//!
//! This is the same run `pmetal train` does: the spec becomes a
//! `TrainingJobConfig` (`TrainingJobConfig::from_spec`, the conversion the
//! command line, the TUI and the GUI use) and `orchestrator::run_training`
//! runs it. The bindings add only what a library caller needs: Python
//! callbacks, Ctrl-C, and the GIL released for the whole run.

use pmetal_core::TrainingCallback;
use pmetal_trainer::orchestrator::{self, TrainingJobConfig, TrainingResult};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::callbacks::{
    PyLoggingCallback, PyMetricsJsonCallback, PyProgressCallback, PythonCallbackBridge, RunState,
};
use crate::error::{anyhow_to_pyerr, config_err};
use crate::json::to_py;
use crate::runtime::runtime;
use crate::spec;
use crate::specs::PyTrainSpec;

/// What a finished training run produced.
#[pyclass(name = "TrainResult", module = "pmetal", frozen, skip_from_py_object)]
pub struct PyTrainResult {
    /// The loss of the last step.
    #[pyo3(get)]
    final_loss: f64,
    /// Optimizer steps taken.
    #[pyo3(get)]
    total_steps: usize,
    /// Tokens trained on.
    #[pyo3(get)]
    total_tokens: usize,
    /// The run's output directory.
    #[pyo3(get)]
    output_dir: String,
    /// The trained weights: the LoRA adapter, or for full-parameter Neural Engine training the model's own weights.
    #[pyo3(get)]
    weights_path: String,
}

impl From<TrainingResult> for PyTrainResult {
    fn from(r: TrainingResult) -> Self {
        Self {
            final_loss: r.final_loss,
            total_steps: r.total_steps,
            total_tokens: r.total_tokens,
            output_dir: r.output_dir.to_string_lossy().into_owned(),
            weights_path: r.lora_weights_path.to_string_lossy().into_owned(),
        }
    }
}

#[pymethods]
impl PyTrainResult {
    /// The result as a dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("final_loss", self.final_loss)?;
        d.set_item("total_steps", self.total_steps)?;
        d.set_item("total_tokens", self.total_tokens)?;
        d.set_item("output_dir", &self.output_dir)?;
        d.set_item("weights_path", &self.weights_path)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainResult(final_loss={:.4}, total_steps={}, total_tokens={}, output_dir={:?}, weights_path={:?})",
            self.final_loss,
            self.total_steps,
            self.total_tokens,
            self.output_dir,
            self.weights_path
        )
    }
}

/// The callbacks for a run: the built-in ones become the trainer's own, every
/// other object goes through the bridge, which also watches for Ctrl-C.
fn build_callbacks(
    py: Python<'_>,
    objects: &[Py<PyAny>],
    state: std::sync::Arc<RunState>,
) -> PyResult<Vec<Box<dyn TrainingCallback>>> {
    let mut native: Vec<Box<dyn TrainingCallback>> = Vec::new();
    let mut bridged = Vec::new();
    for object in objects {
        let bound = object.bind(py);
        if let Ok(p) = bound.extract::<PyRef<'_, PyProgressCallback>>() {
            native.push(Box::new(pmetal_trainer::ProgressCallback::new(
                p.total_steps,
            )));
        } else if let Ok(l) = bound.extract::<PyRef<'_, PyLoggingCallback>>() {
            native.push(Box::new(pmetal_trainer::LoggingCallback::new(l.log_every)));
        } else if let Ok(m) = bound.extract::<PyRef<'_, PyMetricsJsonCallback>>() {
            let callback = pmetal_trainer::MetricsJsonCallback::new(&m.path)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
            native.push(Box::new(callback));
        } else {
            bridged.push(object.clone_ref(py));
        }
    }
    native.push(Box::new(PythonCallbackBridge::new(bridged, state)));
    Ok(native)
}

/// Run `job` to completion on this thread with the GIL released, then raise
/// what a callback or Ctrl-C ended it with, in preference to its result.
pub(crate) fn run_job(
    py: Python<'_>,
    job: TrainingJobConfig,
    objects: &[Py<PyAny>],
) -> PyResult<PyTrainResult> {
    let state = RunState::new();
    let callbacks = build_callbacks(py, objects, state.clone())?;
    let rt = runtime()?;
    let outcome = py.detach(move || rt.block_on(orchestrator::run_training(job, None, callbacks)));
    if let Some(err) = state.take_error() {
        return Err(err);
    }
    outcome.map(PyTrainResult::from).map_err(anyhow_to_pyerr)
}

/// Supervised fine-tuning of one `TrainSpec`, in this process.
///
/// `train()` runs `pmetal train` for the spec with the GIL released, so other
/// Python threads keep running, and stops cleanly on Ctrl-C (the current step
/// finishes, the weights are saved, and `KeyboardInterrupt` is raised).
#[pyclass(name = "Trainer", module = "pmetal")]
pub struct PyTrainer {
    spec: pmetal_core::jobs::TrainSpec,
    callbacks: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyTrainer {
    /// Args:
    ///     spec: What to train. Validated here: `SpecValidationError` lists every problem.
    ///     callbacks: Objects with any of the callback methods (see `pmetal.ProgressCallback`).
    #[new]
    #[pyo3(signature = (spec, callbacks=None))]
    fn new(spec: &PyTrainSpec, callbacks: Option<Vec<Py<PyAny>>>) -> PyResult<Self> {
        let mut spec = spec.inner.clone();
        spec::normalize(&mut spec)?;
        // The conversion the run itself makes, so a name it rejects (an
        // optimizer, a schedule, a quantization scheme) fails here.
        TrainingJobConfig::from_spec(&spec).map_err(config_err)?;
        Ok(Self {
            spec,
            callbacks: callbacks.unwrap_or_default(),
        })
    }

    /// A copy of the spec the trainer runs.
    #[getter]
    fn spec(&self) -> PyTrainSpec {
        PyTrainSpec {
            inner: self.spec.clone(),
        }
    }

    /// Add a callback object.
    fn add_callback(&mut self, callback: Py<PyAny>) {
        self.callbacks.push(callback);
    }

    /// Run the training. Raises `TrainingError` (or the error's own type) if it fails.
    fn train(&self, py: Python<'_>) -> PyResult<PyTrainResult> {
        let mut job = TrainingJobConfig::from_spec(&self.spec).map_err(config_err)?;
        // A library call prints nothing the caller did not ask for.
        job.emit_console_output = false;
        run_job(py, job, &self.callbacks)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Trainer({})",
            spec::repr::<pmetal_core::jobs::TrainSpec>(py, &self.spec)?
        ))
    }
}

/// The fields of a `TrainSpec` as a Python dict, for the `TrainSpec.lora_config()`
/// and `TrainSpec.training_config()` views.
pub(crate) fn config_dict<'py, T: serde::Serialize>(
    py: Python<'py>,
    config: &T,
) -> PyResult<Bound<'py, PyAny>> {
    let value = crate::json::value_of(config).map_err(config_err)?;
    to_py(py, &value)
}
