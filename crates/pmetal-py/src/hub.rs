//! Python wrappers for HuggingFace Hub operations.

use pyo3::prelude::*;
use std::sync::OnceLock;

use crate::error::{IntoPyResult, pmetal_to_pyerr};

/// Shared tokio runtime for async operations.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"))
}

/// Download a model from HuggingFace Hub.
///
/// Args:
///     model_id: HuggingFace model identifier (e.g., "Qwen/Qwen3-0.6B")
///     revision: Optional git revision/branch
///
/// Returns:
///     Local path to the downloaded model directory
#[pyfunction]
#[pyo3(signature = (model_id, revision=None))]
pub fn download_model(py: Python<'_>, model_id: &str, revision: Option<&str>) -> PyResult<String> {
    py.allow_threads(|| {
        runtime()
            .block_on(pmetal_hub::download_model(model_id, revision, None))
            .map(|p| p.to_string_lossy().to_string())
            .map_err(pmetal_to_pyerr)
    })
}

/// Download a specific file from a HuggingFace Hub repository.
///
/// Args:
///     model_id: HuggingFace model identifier
///     filename: Name of the file to download
///     revision: Optional git revision/branch
///
/// Returns:
///     Local path to the downloaded file
#[pyfunction]
#[pyo3(signature = (model_id, filename, revision=None))]
pub fn download_file(
    py: Python<'_>,
    model_id: &str,
    filename: &str,
    revision: Option<&str>,
) -> PyResult<String> {
    py.allow_threads(|| {
        runtime()
            .block_on(pmetal_hub::download_file(
                model_id, filename, revision, None,
            ))
            .map(|p| p.to_string_lossy().to_string())
            .into_pyresult()
    })
}

/// Access the shared tokio runtime (for use by other modules).
pub fn shared_runtime() -> &'static tokio::runtime::Runtime {
    runtime()
}

/// Check if a string looks like a HuggingFace model ID (e.g., "org/model")
/// rather than a local filesystem path.
pub fn is_hf_model_id(s: &str) -> bool {
    !s.starts_with('/') && !s.starts_with('.') && s.contains('/')
}
