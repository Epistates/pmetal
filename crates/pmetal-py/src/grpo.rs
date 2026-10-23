//! `GrpoTrainer`: a `GrpoSpec` run in this process, with Python reward
//! functions.
//!
//! The run is the one `pmetal grpo` makes (`pmetal_trainer::grpo::run_grpo_job`).
//! The bindings add Python callables as rewards, in the usual shape for one:
//! `fn(prompts, completions, **columns) -> list[float]`,
//! one entry per completion, where `columns` are the dataset's other columns
//! (a list each, one entry per completion). They are summed, weighted, with the
//! rewards the spec asks for (`reward_cmd`, `reward_regex`, ...).
//!
//! The run holds the GIL only while a reward function runs: the thread that
//! trains has released it (`py.detach`), and a function is called with the GIL
//! reacquired for exactly the call, on whichever thread scores (the training
//! thread, or the scorer thread with `async_rewards`). A function that raises
//! ends the run after the batch with the exception attached as the cause of a
//! `RewardError`.

use std::sync::{Arc, Mutex};

use pmetal_core::jobs::GrpoSpec;
use pmetal_trainer::grpo::{
    GrpoError, GrpoJobOptions, GrpoJobResult, GrpoResult, RewardBatch, RewardFunction, run_grpo_job,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::callbacks::RunState;
use crate::error::{Exceptions, anyhow_to_pyerr, config_err};
use crate::json::to_py;
use crate::runtime::runtime;
use crate::spec;
use crate::specs::PyGrpoSpec;
use crate::trainer::build_callbacks;

/// What a finished GRPO run produced.
#[pyclass(name = "GrpoResult", module = "pmetal", frozen)]
pub struct PyGrpoResult {
    /// Optimizer steps taken.
    #[pyo3(get)]
    total_steps: usize,
    /// A callback asked the run to stop before it finished.
    #[pyo3(get)]
    stopped: bool,
    /// The run's output directory.
    #[pyo3(get)]
    output_dir: String,
}

#[pymethods]
impl PyGrpoResult {
    /// The result as a dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("total_steps", self.total_steps)?;
        d.set_item("stopped", self.stopped)?;
        d.set_item("output_dir", &self.output_dir)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "GrpoResult(total_steps={}, stopped={}, output_dir={:?})",
            self.total_steps,
            if self.stopped { "True" } else { "False" },
            self.output_dir
        )
    }
}

/// The first exception a reward function raised, and which function.
#[derive(Default)]
struct RewardFailure(Mutex<Option<(String, PyErr)>>);

impl RewardFailure {
    fn record(&self, name: &str, err: PyErr) {
        let mut slot = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some((name.to_string(), err));
        }
    }

    fn take(&self) -> Option<(String, PyErr)> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).take()
    }
}

/// A Python callable as a reward: `fn(prompts, completions, **columns)`.
struct PyCallableReward {
    function: Py<PyAny>,
    name: String,
    failure: Arc<RewardFailure>,
}

impl PyCallableReward {
    /// Call the function for `batch` with the GIL held, and read what it
    /// returned: one number per completion.
    fn call(&self, py: Python<'_>, batch: &RewardBatch<'_>) -> PyResult<Vec<f64>> {
        let n = batch.completions.len();
        let kwargs = PyDict::new(py);
        // One list per column of the dataset, one entry per completion.
        let mut keys: Vec<&String> = batch.columns.iter().flat_map(|row| row.keys()).collect();
        keys.sort_unstable();
        keys.dedup();
        for key in keys {
            if key == "prompts" || key == "completions" {
                continue;
            }
            let column = PyList::empty(py);
            for row in batch.columns {
                match row.get(key) {
                    Some(value) => column.append(to_py(py, value)?)?,
                    None => column.append(py.None())?,
                }
            }
            kwargs.set_item(key, column)?;
        }
        let returned = self.function.bind(py).call(
            (
                PyList::new(py, batch.prompts)?,
                PyList::new(py, batch.completions)?,
            ),
            Some(&kwargs),
        )?;

        let mut rewards = Vec::with_capacity(n);
        for (i, item) in returned.try_iter()?.enumerate() {
            let item = item?;
            if item.is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "reward function {:?} returned None for completion {i}; a completion it \
                     cannot score is an error, not a 0 (return a number)",
                    self.name
                )));
            }
            rewards.push(item.extract::<f64>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "reward function {:?} returned {item} for completion {i}, which is not a number",
                    self.name
                ))
            })?);
        }
        if rewards.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "reward function {:?} returned {} rewards for {n} completions",
                self.name,
                rewards.len()
            )));
        }
        Ok(rewards)
    }
}

impl RewardFunction for PyCallableReward {
    fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<pmetal_runtime::compat::Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        let references = vec![None; completions.len()];
        self.compute_with_references(prompts, completions, &references, images)
    }

    fn compute_with_references(
        &self,
        prompts: &[String],
        completions: &[String],
        references: &[Option<String>],
        images: Option<&[Vec<pmetal_runtime::compat::Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        self.score(&RewardBatch {
            prompts,
            completions,
            references,
            columns: &[],
            images,
        })
    }

    fn score(&self, batch: &RewardBatch<'_>) -> GrpoResult<Vec<f64>> {
        Python::attach(|py| match self.call(py, batch) {
            Ok(rewards) => Ok(rewards),
            Err(err) => {
                let message = format!("reward function {:?} failed: {}", self.name, err.value(py));
                self.failure.record(&self.name, err);
                Err(GrpoError::Reward(message))
            }
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// The name a callable goes by in logs and errors.
fn callable_name(function: &Bound<'_, PyAny>) -> String {
    function
        .getattr("__qualname__")
        .or_else(|_| function.getattr("__name__"))
        .and_then(|n| n.extract::<String>())
        .unwrap_or_else(|_| "reward".to_string())
}

/// GRPO reinforcement learning of one `GrpoSpec`, in this process, with
/// Python reward functions.
///
/// `train()` runs `pmetal grpo` for the spec with the GIL released except
/// while a reward function runs, and stops cleanly on Ctrl-C.
#[pyclass(name = "GrpoTrainer", module = "pmetal")]
pub struct PyGrpoTrainer {
    spec: GrpoSpec,
    reward_funcs: Vec<Py<PyAny>>,
    reward_weights: Vec<f64>,
    callbacks: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyGrpoTrainer {
    /// Args:
    ///     spec: What to train. Validated here: `SpecValidationError` lists every problem.
    ///     reward_funcs: Callables `fn(prompts, completions, **columns) -> list[float]`.
    ///     reward_weights: One weight per function (1 each when omitted).
    ///     callbacks: Objects with any of the callback methods.
    #[new]
    #[pyo3(signature = (spec, reward_funcs=None, reward_weights=None, callbacks=None))]
    fn new(
        py: Python<'_>,
        spec: &PyGrpoSpec,
        reward_funcs: Option<Vec<Py<PyAny>>>,
        reward_weights: Option<Vec<f64>>,
        callbacks: Option<Vec<Py<PyAny>>>,
    ) -> PyResult<Self> {
        let mut spec = spec.inner.clone();
        spec::normalize(&mut spec)?;
        let reward_funcs = reward_funcs.unwrap_or_default();
        for function in &reward_funcs {
            if !function.bind(py).is_callable() {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "a reward function must be callable, got {}",
                    function.bind(py).get_type().name()?
                )));
            }
        }
        let reward_weights = match reward_weights {
            Some(weights) => {
                if weights.len() != reward_funcs.len() {
                    return Err(config_err(format!(
                        "{} reward_weights for {} reward_funcs; give one weight per function",
                        weights.len(),
                        reward_funcs.len()
                    )));
                }
                if let Some(bad) = weights.iter().find(|w| !w.is_finite()) {
                    return Err(config_err(format!("reward weight {bad} is not finite")));
                }
                weights
            }
            None => vec![1.0; reward_funcs.len()],
        };
        Ok(Self {
            spec,
            reward_funcs,
            reward_weights,
            callbacks: callbacks.unwrap_or_default(),
        })
    }

    /// A copy of the spec the trainer runs.
    #[getter]
    fn spec(&self) -> PyGrpoSpec {
        PyGrpoSpec {
            inner: self.spec.clone(),
        }
    }

    /// Add a callback object.
    fn add_callback(&mut self, callback: Py<PyAny>) {
        self.callbacks.push(callback);
    }

    /// Run the training. Raises `RewardError` if a reward function fails (the
    /// function's own exception is its `__cause__`), or `TrainingError` (or
    /// the error's own type) for any other failure.
    fn train(&self, py: Python<'_>) -> PyResult<PyGrpoResult> {
        let failure = Arc::new(RewardFailure::default());
        let extra_rewards: Vec<(Box<dyn RewardFunction>, f64)> = self
            .reward_funcs
            .iter()
            .zip(&self.reward_weights)
            .map(|(function, &weight)| {
                let reward = PyCallableReward {
                    function: function.clone_ref(py),
                    name: callable_name(function.bind(py)),
                    failure: Arc::clone(&failure),
                };
                (Box::new(reward) as Box<dyn RewardFunction>, weight)
            })
            .collect();

        let state = RunState::new();
        let (callbacks, bridge) = build_callbacks(py, &self.callbacks, state.clone())?;
        let spec = self.spec.clone();
        let rt = runtime()?;
        let outcome = py.detach(move || {
            rt.block_on(run_grpo_job(
                spec,
                GrpoJobOptions {
                    // A library call prints nothing the caller did not ask for.
                    emit_console_output: false,
                    callbacks,
                    extra_rewards,
                },
            ))
        });
        bridge.train_end();

        if let Some((name, cause)) = failure.take() {
            return Err(reward_error(py, &name, cause));
        }
        if let Some(err) = state.take_error() {
            return Err(err);
        }
        let GrpoJobResult { steps, stopped } = outcome.map_err(anyhow_to_pyerr)?;
        Ok(PyGrpoResult {
            total_steps: steps,
            stopped,
            output_dir: self.spec.output_dir.clone(),
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "GrpoTrainer({}, reward_funcs={})",
            spec::repr::<GrpoSpec>(py, &self.spec)?,
            self.reward_funcs.len()
        ))
    }
}

/// `RewardError` for a reward function that raised `cause`.
fn reward_error(py: Python<'_>, function: &str, cause: PyErr) -> PyErr {
    // Ctrl-C that landed while a function ran is the user's stop, not its failure.
    if cause.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
        return cause;
    }
    let message = format!(
        "reward function {function:?} raised {}: {}",
        cause
            .get_type(py)
            .name()
            .map_or_else(|_| "an exception".to_string(), |n| n.to_string()),
        cause.value(py)
    );
    let err = PyErr::from_type(Exceptions::get(py).reward_error.bind(py).clone(), message);
    err.set_cause(py, Some(cause));
    err
}
