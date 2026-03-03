//! Python wrapper for the PMetal tokenizer.

use pyo3::prelude::*;

use crate::error::IntoPyResult;

#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    pub(crate) inner: pmetal_data::Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load a tokenizer from a local file path.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = pmetal_data::Tokenizer::from_file(path).into_pyresult()?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from a HuggingFace model (downloads tokenizer.json).
    ///
    /// Releases the GIL during the network download.
    #[staticmethod]
    fn from_pretrained(py: Python<'_>, model_id: &str) -> PyResult<Self> {
        let id = model_id.to_string();
        // Release GIL during download
        let tokenizer_path = py.allow_threads(move || {
            crate::hub::shared_runtime()
                .block_on(pmetal_hub::download_file(&id, "tokenizer.json", None, None))
                .map_err(crate::error::runtime_err)
        })?;
        let inner = pmetal_data::Tokenizer::from_file(tokenizer_path.to_string_lossy().as_ref())
            .into_pyresult()?;
        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner.encode(text).into_pyresult()
    }

    /// Decode token IDs to text.
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&ids).into_pyresult()
    }

    /// Get the vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Get the pad token ID, if available.
    #[getter]
    fn pad_token_id(&self) -> Option<u32> {
        self.inner.pad_token_id()
    }

    /// Get the EOS token ID, if available.
    #[getter]
    fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id()
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }
}
