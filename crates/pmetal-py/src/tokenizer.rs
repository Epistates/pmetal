//! `Tokenizer`.

use std::sync::Arc;

use pmetal_data::Tokenizer;
use pyo3::prelude::*;

use crate::error::tokenizer_err;
use crate::hub::{HubArgs, resolve_model};

/// A model's tokenizer.
///
/// Immutable and shareable between threads; encoding and decoding release the
/// GIL.
#[pyclass(name = "Tokenizer", module = "pmetal", frozen)]
pub struct PyTokenizer {
    pub(crate) inner: Arc<Tokenizer>,
}

#[pymethods]
impl PyTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    #[staticmethod]
    fn from_file(path: std::path::PathBuf) -> PyResult<Self> {
        let inner = Tokenizer::from_file(path).map_err(tokenizer_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Load the tokenizer of a model: a local directory, or a Hub id (its
    /// files are downloaded, and Ctrl-C interrupts the download).
    ///
    /// Special tokens (beginning, end, padding) are read from the model's own
    /// configuration, not guessed from their names.
    #[staticmethod]
    #[pyo3(signature = (model_id, *, revision=None, token=None))]
    fn from_pretrained(
        py: Python<'_>,
        model_id: &str,
        revision: Option<String>,
        token: Option<String>,
    ) -> PyResult<Self> {
        let hub = HubArgs {
            revision,
            token,
            ..HubArgs::default()
        };
        let dir = resolve_model(py, model_id, &hub)?;
        let inner = Tokenizer::from_model_dir(&dir).map_err(tokenizer_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Token ids for `text`.
    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        let tokenizer = &self.inner;
        py.detach(|| tokenizer.encode(text)).map_err(tokenizer_err)
    }

    /// The text for token ids.
    fn decode(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<String> {
        let tokenizer = &self.inner;
        py.detach(|| tokenizer.decode(&ids)).map_err(tokenizer_err)
    }

    /// The vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// The padding token's id, if the tokenizer has one.
    #[getter]
    fn pad_token_id(&self) -> Option<u32> {
        self.inner.pad_token_id()
    }

    /// The end-of-sequence token's id, if the tokenizer has one.
    #[getter]
    fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id()
    }

    /// The beginning-of-sequence token's id, if the tokenizer has one.
    #[getter]
    fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id()
    }

    /// The unknown token's id, if the tokenizer has one.
    #[getter]
    fn unk_token_id(&self) -> Option<u32> {
        self.inner.unk_token_id()
    }

    fn __len__(&self) -> usize {
        self.inner.vocab_size()
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }
}
