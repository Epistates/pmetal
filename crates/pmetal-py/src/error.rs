//! The exception hierarchy, and the mapping from Rust errors onto it.
//!
//! Every error pmetal raises is a [`PMetalError`](Exceptions::pmetal_error)
//! (`pmetal.PMetalError`), so one `except pmetal.PMetalError` catches them
//! all. The subclasses say what failed, and the ones that stand for a built-in
//! failure also derive from it (`ConfigError` is a `ValueError`,
//! `OutOfMemoryError` a `MemoryError`), so code written against the built-ins
//! keeps working.
//!
//! ```text
//! PMetalError
//! ├── ConfigError (ValueError)
//! │   └── SpecValidationError      .errors: list[tuple[field, message]]
//! ├── SerializationError (ValueError)
//! ├── HubError
//! ├── ModelLoadError
//! │   └── UnsupportedArchitectureError
//! ├── TokenizerError
//! ├── TrainingError
//! │   └── RewardError              a GRPO reward function failed; `__cause__` is its exception
//! ├── QuantizationError
//! ├── BackendError                 MLX / Metal / ANE
//! ├── OutOfMemoryError (MemoryError)
//! ├── CancelledError
//! └── JobError                     (a `pmetal` subprocess job; defined in Python)
//! ```
//!
//! A Rust panic is not one of these: PyO3 turns it into
//! `pyo3_runtime.PanicException` at the boundary, which is why the wheel is
//! built with unwinding (`[profile.wheel]`).

use pmetal_core::{FieldError, PMetalError};
use pyo3::exceptions::{
    PyBrokenPipeError, PyException, PyFileExistsError, PyFileNotFoundError, PyInterruptedError,
    PyMemoryError, PyNotImplementedError, PyOSError, PyPermissionError, PyTimeoutError,
    PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

/// The exception classes, created once when the module is imported.
pub struct Exceptions {
    pub pmetal_error: Py<PyType>,
    pub config_error: Py<PyType>,
    pub spec_validation_error: Py<PyType>,
    pub serialization_error: Py<PyType>,
    pub hub_error: Py<PyType>,
    pub model_load_error: Py<PyType>,
    pub unsupported_architecture_error: Py<PyType>,
    pub tokenizer_error: Py<PyType>,
    pub training_error: Py<PyType>,
    pub reward_error: Py<PyType>,
    pub quantization_error: Py<PyType>,
    pub backend_error: Py<PyType>,
    pub out_of_memory_error: Py<PyType>,
    pub cancelled_error: Py<PyType>,
}

static EXCEPTIONS: PyOnceLock<Exceptions> = PyOnceLock::new();

/// A new exception class, `type(name, bases, {"__module__": "pmetal", ...})`.
/// `create_exception!` takes one base; several are needed here
/// (`ConfigError` is both a `PMetalError` and a `ValueError`).
fn new_class<'py>(
    py: Python<'py>,
    name: &str,
    bases: &[&Bound<'py, PyType>],
    doc: &str,
) -> PyResult<Py<PyType>> {
    let namespace = PyDict::new(py);
    namespace.set_item("__module__", "pmetal")?;
    namespace.set_item("__doc__", doc)?;
    let class = py
        .get_type::<PyType>()
        .call1((name, PyTuple::new(py, bases)?, namespace))?;
    Ok(class.cast_into::<PyType>()?.unbind())
}

impl Exceptions {
    fn create(py: Python<'_>) -> PyResult<Self> {
        let exception = py.get_type::<PyException>();
        let value_error = py.get_type::<PyValueError>();
        let memory_error = py.get_type::<PyMemoryError>();

        let pmetal_error = new_class(
            py,
            "PMetalError",
            &[&exception],
            "Base class of every error pmetal raises.",
        )?;
        let base = pmetal_error.bind(py).clone();
        let config_error = new_class(
            py,
            "ConfigError",
            &[&base, &value_error],
            "A configuration, argument or tensor description pmetal cannot use.",
        )?;
        let config = config_error.bind(py).clone();
        let spec_validation_error = new_class(
            py,
            "SpecValidationError",
            &[&config],
            "A job spec failed validation. `errors` lists every problem as `(field, message)`.",
        )?;
        let serialization_error = new_class(
            py,
            "SerializationError",
            &[&base, &value_error],
            "Data pmetal could not serialize or deserialize.",
        )?;
        let hub_error = new_class(
            py,
            "HubError",
            &[&base],
            "A HuggingFace Hub operation failed.",
        )?;
        let model_load_error = new_class(
            py,
            "ModelLoadError",
            &[&base],
            "A model could not be loaded.",
        )?;
        let load = model_load_error.bind(py).clone();
        let unsupported_architecture_error = new_class(
            py,
            "UnsupportedArchitectureError",
            &[&load],
            "The model's architecture is not one pmetal implements.",
        )?;
        let tokenizer_error = new_class(
            py,
            "TokenizerError",
            &[&base],
            "A tokenizer failed to load or run.",
        )?;
        let training_error = new_class(py, "TrainingError", &[&base], "A training run failed.")?;
        let reward_error = new_class(
            py,
            "RewardError",
            &[&training_error.bind(py).clone()],
            "A reward function of a GRPO run failed: it raised, or returned something that is not one number per completion. The function's own exception is the `__cause__`.",
        )?;
        let quantization_error =
            new_class(py, "QuantizationError", &[&base], "Quantization failed.")?;
        let backend_error = new_class(
            py,
            "BackendError",
            &[&base],
            "The compute backend (MLX, Metal, the Neural Engine) reported an error.",
        )?;
        let out_of_memory_error = new_class(
            py,
            "OutOfMemoryError",
            &[&base, &memory_error],
            "The operation needs more memory than the device can give it.",
        )?;
        let cancelled_error = new_class(
            py,
            "CancelledError",
            &[&base],
            "The operation was cancelled.",
        )?;

        Ok(Self {
            pmetal_error,
            config_error,
            spec_validation_error,
            serialization_error,
            hub_error,
            model_load_error,
            unsupported_architecture_error,
            tokenizer_error,
            training_error,
            reward_error,
            quantization_error,
            backend_error,
            out_of_memory_error,
            cancelled_error,
        })
    }

    /// The classes, creating them on first use.
    pub fn get(py: Python<'_>) -> &'static Exceptions {
        EXCEPTIONS.get_or_init(py, || {
            Exceptions::create(py).expect("creating the pmetal exception classes cannot fail")
        })
    }

    fn all(&self) -> [(&'static str, &Py<PyType>); 14] {
        [
            ("PMetalError", &self.pmetal_error),
            ("ConfigError", &self.config_error),
            ("SpecValidationError", &self.spec_validation_error),
            ("SerializationError", &self.serialization_error),
            ("HubError", &self.hub_error),
            ("ModelLoadError", &self.model_load_error),
            (
                "UnsupportedArchitectureError",
                &self.unsupported_architecture_error,
            ),
            ("TokenizerError", &self.tokenizer_error),
            ("TrainingError", &self.training_error),
            ("RewardError", &self.reward_error),
            ("QuantizationError", &self.quantization_error),
            ("BackendError", &self.backend_error),
            ("OutOfMemoryError", &self.out_of_memory_error),
            ("CancelledError", &self.cancelled_error),
        ]
    }
}

/// Add every exception class to the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for (name, class) in Exceptions::get(m.py()).all() {
        m.add(name, class.bind(m.py()))?;
    }
    Ok(())
}

fn raise(class: &Py<PyType>, message: String) -> PyErr {
    Python::attach(|py| PyErr::from_type(class.bind(py).clone(), message))
}

/// The Python exception for a `PMetalError`.
pub fn pmetal_to_pyerr(err: PMetalError) -> PyErr {
    let message = err.to_string();
    Python::attach(|py| {
        let e = Exceptions::get(py);
        match &err {
            PMetalError::InvalidArgument(_)
            | PMetalError::Config(_)
            | PMetalError::ShapeMismatch { .. }
            | PMetalError::DtypeMismatch { .. } => raise(&e.config_error, message),
            PMetalError::Io(io) => io_to_pyerr(io, message),
            PMetalError::OutOfMemory { .. } => raise(&e.out_of_memory_error, message),
            PMetalError::NotImplemented(_) => PyNotImplementedError::new_err(message),
            PMetalError::ModelLoad(_) => raise(&e.model_load_error, message),
            PMetalError::UnsupportedArchitecture(_) => {
                raise(&e.unsupported_architecture_error, message)
            }
            PMetalError::Quantization(_) => raise(&e.quantization_error, message),
            PMetalError::Training(_) => raise(&e.training_error, message),
            PMetalError::Serialization(_) => raise(&e.serialization_error, message),
            PMetalError::Hub(_) => raise(&e.hub_error, message),
            PMetalError::Tokenizer(_) => raise(&e.tokenizer_error, message),
            PMetalError::Cancelled => raise(&e.cancelled_error, message),
            // MLX errors, and the feature-gated variants (ANE).
            _ => raise(&e.backend_error, message),
        }
    })
}

/// The `OSError` for an I/O error: the subclass its errno names
/// (`FileNotFoundError`, `PermissionError`, ...), or when it carries none (a
/// library's own message), the one its kind names.
fn io_to_pyerr(io: &std::io::Error, message: String) -> PyErr {
    use std::io::ErrorKind;
    if let Some(errno) = io.raw_os_error() {
        return PyOSError::new_err((errno, message));
    }
    match io.kind() {
        ErrorKind::NotFound => PyFileNotFoundError::new_err(message),
        ErrorKind::PermissionDenied => PyPermissionError::new_err(message),
        ErrorKind::AlreadyExists => PyFileExistsError::new_err(message),
        ErrorKind::TimedOut => PyTimeoutError::new_err(message),
        ErrorKind::BrokenPipe => PyBrokenPipeError::new_err(message),
        ErrorKind::Interrupted => PyInterruptedError::new_err(message),
        _ => PyOSError::new_err(message),
    }
}

/// The Python exception for an `anyhow::Error`: a `PMetalError` or an
/// `io::Error` inside it keeps its type; anything else is a `TrainingError`.
pub fn anyhow_to_pyerr(err: anyhow::Error) -> PyErr {
    match err.downcast::<PMetalError>() {
        Ok(err) => pmetal_to_pyerr(err),
        Err(err) => match err.downcast::<std::io::Error>() {
            Ok(io) => pmetal_to_pyerr(PMetalError::Io(io)),
            Err(err) => {
                Python::attach(|py| raise(&Exceptions::get(py).training_error, format!("{err:#}")))
            }
        },
    }
}

/// An MLX or runtime exception: the backend's.
pub fn backend_err(e: impl std::fmt::Display) -> PyErr {
    Python::attach(|py| raise(&Exceptions::get(py).backend_error, e.to_string()))
}

/// A model that would not load.
pub fn model_load_err(e: impl std::fmt::Display) -> PyErr {
    Python::attach(|py| raise(&Exceptions::get(py).model_load_error, e.to_string()))
}

/// A tokenizer failure.
pub fn tokenizer_err(e: impl std::fmt::Display) -> PyErr {
    Python::attach(|py| raise(&Exceptions::get(py).tokenizer_error, e.to_string()))
}

/// An argument the call cannot use.
pub fn config_err(e: impl std::fmt::Display) -> PyErr {
    Python::attach(|py| raise(&Exceptions::get(py).config_error, e.to_string()))
}

/// A spec that failed validation: `SpecValidationError` carrying every
/// problem in `.errors`.
pub fn validation_err(spec: &str, errors: &[FieldError]) -> PyErr {
    Python::attach(|py| {
        let lines: Vec<String> = errors
            .iter()
            .map(|e| format!("  {}: {}", e.field, e.message))
            .collect();
        let message = format!("invalid {spec}:\n{}", lines.join("\n"));
        let err = PyErr::from_type(
            Exceptions::get(py).spec_validation_error.bind(py).clone(),
            message,
        );
        let pairs = errors
            .iter()
            .map(|e| PyTuple::new(py, [e.field.as_str(), e.message.as_str()]))
            .collect::<PyResult<Vec<_>>>()
            .and_then(|pairs| PyList::new(py, pairs));
        if let Ok(pairs) = pairs {
            let _ = err.value(py).setattr("errors", pairs);
        }
        err
    })
}
