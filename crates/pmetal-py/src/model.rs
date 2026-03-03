//! Python wrapper for model loading and inference.

use std::path::PathBuf;

use pyo3::prelude::*;

use crate::config::PyModelArchitecture;
use crate::error::runtime_err;
use crate::hub::is_hf_model_id;

/// `DynamicModel` contains types that are not Send+Sync (dyn trait objects).
/// We mark the pyclass as `unsendable` so it can only be used from the
/// thread that created it, which is fine for single-threaded Python access.
///
/// Note: The GIL cannot be released during model loading or generation because
/// DynamicModel is !Send. The GIL *is* released during model downloads.
#[pyclass(name = "Model", unsendable)]
pub struct PyModel {
    inner: pmetal_models::DynamicModel,
    tokenizer: Option<pmetal_data::Tokenizer>,
    model_path: PathBuf,
}

#[pymethods]
impl PyModel {
    /// Load a model from a local path or HuggingFace Hub.
    ///
    /// Args:
    ///     path_or_id: Local model directory or HuggingFace model ID
    ///     fp8: Whether to quantize to FP8 for memory savings
    #[staticmethod]
    #[pyo3(signature = (path_or_id, fp8=false))]
    fn load(py: Python<'_>, path_or_id: &str, fp8: bool) -> PyResult<Self> {
        // Release GIL during download (network I/O is Send-safe)
        let model_path = if is_hf_model_id(path_or_id) {
            let id = path_or_id.to_string();
            py.allow_threads(move || {
                crate::hub::shared_runtime()
                    .block_on(async {
                        let path = pmetal_hub::download_model(&id, None, None).await?;
                        let _ = pmetal_hub::download_file(&id, "tokenizer.json", None, None).await;
                        let _ = pmetal_hub::download_file(&id, "tokenizer_config.json", None, None)
                            .await;
                        Ok::<_, pmetal_core::PMetalError>(path)
                    })
                    .map_err(runtime_err)
            })?
        } else {
            PathBuf::from(path_or_id)
        };

        // Model loading must hold the GIL (DynamicModel is !Send)
        let mut model = pmetal_models::DynamicModel::load(&model_path).map_err(runtime_err)?;

        if fp8 {
            model.quantize_fp8().map_err(runtime_err)?;
        }

        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            pmetal_data::Tokenizer::from_file(&tokenizer_path).ok()
        } else {
            None
        };

        Ok(Self {
            inner: model,
            tokenizer,
            model_path,
        })
    }

    /// Generate text from a prompt.
    ///
    /// Note: The GIL is held during generation because DynamicModel is !Send.
    /// For non-blocking inference in async Python, run this in a thread executor.
    ///
    /// Args:
    ///     prompt: Input text
    ///     max_tokens: Maximum tokens to generate (default 256)
    ///     temperature: Sampling temperature (default 0.7)
    ///     top_k: Top-k sampling (default 50)
    ///     top_p: Nucleus sampling threshold (default 0.9)
    ///     seed: Random seed for reproducibility
    ///
    /// Returns:
    ///     Generated text string
    #[pyo3(signature = (prompt, max_tokens=256, temperature=0.7, top_k=50, top_p=0.9, seed=None))]
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "No tokenizer loaded. Load model from a directory that contains tokenizer.json",
            )
        })?;

        let input_ids = tokenizer.encode(prompt).map_err(runtime_err)?;

        let mut gen_config = if temperature < 1e-6 {
            pmetal_models::GenerationConfig::greedy(max_tokens)
        } else {
            pmetal_models::GenerationConfig::sampling(max_tokens, temperature)
                .with_top_k(top_k)
                .with_top_p(top_p)
        };

        if let Some(s) = seed {
            gen_config = gen_config.with_seed(s);
        }

        // Use actual EOS token, don't hardcode a fallback
        if let Some(eos) = tokenizer.eos_token_id() {
            gen_config = gen_config.with_stop_tokens(vec![eos]);
        }

        let max_seq_len = input_ids.len() + max_tokens + 64;
        let mut cache = self.inner.create_cache(max_seq_len);

        let output = pmetal_models::generate_cached_async(
            |input, cache| {
                self.inner
                    .forward_with_hybrid_cache(input, None, Some(cache), None)
            },
            &input_ids,
            gen_config,
            &mut cache,
        )
        .map_err(runtime_err)?;

        let prompt_len = input_ids.len();
        let generated = &output.token_ids[prompt_len..];
        tokenizer.decode(generated).map_err(runtime_err)
    }

    /// Get the model architecture.
    fn architecture(&self) -> PyModelArchitecture {
        self.inner.architecture().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(path='{}', architecture={:?})",
            self.model_path.display(),
            self.inner.architecture(),
        )
    }
}
