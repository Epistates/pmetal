//! Error mapping from Rust PMetal errors to Python exceptions.

use pmetal_core::PMetalError;
use pyo3::PyErr;
use pyo3::exceptions::{
    PyIOError, PyMemoryError, PyNotImplementedError, PyRuntimeError, PyValueError,
};

/// Convert a `PMetalError` into the appropriate Python exception.
pub fn pmetal_to_pyerr(err: PMetalError) -> PyErr {
    match &err {
        PMetalError::InvalidArgument(_) => PyValueError::new_err(err.to_string()),
        PMetalError::Config(_) => PyValueError::new_err(err.to_string()),
        PMetalError::ShapeMismatch { .. } => PyValueError::new_err(err.to_string()),
        PMetalError::DtypeMismatch { .. } => PyValueError::new_err(err.to_string()),
        PMetalError::Io(_) => PyIOError::new_err(err.to_string()),
        PMetalError::OutOfMemory { .. } => PyMemoryError::new_err(err.to_string()),
        PMetalError::NotImplemented(_) => PyNotImplementedError::new_err(err.to_string()),
        PMetalError::ModelLoad(_)
        | PMetalError::UnsupportedArchitecture(_)
        | PMetalError::Quantization(_)
        | PMetalError::Training(_)
        | PMetalError::Serialization(_)
        | PMetalError::Hub(_)
        | PMetalError::Tokenizer(_)
        | PMetalError::Mlx(_) => PyRuntimeError::new_err(err.to_string()),
        // Catch-all for feature-gated variants (e.g., ANE)
        #[allow(unreachable_patterns)]
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}

/// Extension trait for converting `pmetal_core::Result<T>` to `PyResult<T>`.
pub trait IntoPyResult<T> {
    fn into_pyresult(self) -> pyo3::PyResult<T>;
}

impl<T> IntoPyResult<T> for pmetal_core::Result<T> {
    fn into_pyresult(self) -> pyo3::PyResult<T> {
        self.map_err(pmetal_to_pyerr)
    }
}

/// Convert a Display error into a Python RuntimeError.
pub fn runtime_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}
