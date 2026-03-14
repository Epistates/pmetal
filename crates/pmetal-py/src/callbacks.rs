//! Python callback bridge for training events.

use pmetal_core::TrainingCallback;
use pyo3::prelude::*;

/// Bridge that dispatches Rust training callbacks to Python objects.
///
/// Python callback objects should implement any of these methods:
/// - `on_train_start()`
/// - `on_train_end()`
/// - `on_epoch_start(epoch: int)`
/// - `on_epoch_end(epoch: int, loss: float, perplexity: float)`
/// - `on_step_start(step: int)`
/// - `on_step_end(step: int, loss: float)`
/// - `on_save(path: str)`
#[allow(dead_code)]
pub struct PythonCallbackBridge {
    pub(crate) py_callbacks: Vec<Py<PyAny>>,
}

impl TrainingCallback for PythonCallbackBridge {
    fn on_train_start(&mut self) {
        Python::attach(|py| {
            for cb in &self.py_callbacks {
                let _ = cb.call_method0(py, "on_train_start");
            }
        });
    }

    fn on_train_end(&mut self) {
        Python::attach(|py| {
            for cb in &self.py_callbacks {
                let _ = cb.call_method0(py, "on_train_end");
            }
        });
    }

    fn on_epoch_start(&mut self, epoch: usize) {
        Python::attach(|py| {
            for cb in &self.py_callbacks {
                let _ = cb.call_method1(py, "on_epoch_start", (epoch,));
            }
        });
    }

    fn on_epoch_end(&mut self, epoch: usize, metrics: &pmetal_core::EvalMetrics) {
        Python::attach(|py| {
            for cb in &self.py_callbacks {
                let _ = cb.call_method1(
                    py,
                    "on_epoch_end",
                    (epoch, metrics.loss, metrics.perplexity),
                );
            }
        });
    }

    fn on_step_start(&mut self, step: usize) {
        Python::attach(|py| {
            for cb in &self.py_callbacks {
                let _ = cb.call_method1(py, "on_step_start", (step,));
            }
        });
    }

    fn on_step_end(&mut self, step: usize, loss: f64) {
        Python::attach(|py| {
            for cb in &self.py_callbacks {
                let _ = cb.call_method1(py, "on_step_end", (step, loss));
            }
        });
    }

    fn on_save(&mut self, path: &std::path::Path) {
        Python::attach(|py| {
            for cb in &self.py_callbacks {
                let _ = cb.call_method1(py, "on_save", (path.to_string_lossy().to_string(),));
            }
        });
    }
}

// Expose Rust built-in callbacks to Python

/// Progress bar callback that shows training progress.
#[pyclass(name = "ProgressCallback")]
pub struct PyProgressCallback {
    pub(crate) total_steps: usize,
}

#[pymethods]
impl PyProgressCallback {
    #[new]
    fn new(total_steps: usize) -> Self {
        Self { total_steps }
    }

    fn __repr__(&self) -> String {
        format!("ProgressCallback(total_steps={})", self.total_steps)
    }
}

/// Logging callback that prints metrics every N steps.
#[pyclass(name = "LoggingCallback")]
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

    fn __repr__(&self) -> String {
        format!("LoggingCallback(log_every={})", self.log_every)
    }
}

/// Metrics JSON callback that writes training metrics to a JSONL file.
#[pyclass(name = "MetricsJsonCallback")]
pub struct PyMetricsJsonCallback {
    pub(crate) path: String,
}

#[pymethods]
impl PyMetricsJsonCallback {
    #[new]
    fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("MetricsJsonCallback(path='{}')", self.path)
    }
}
