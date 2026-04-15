//! Python bindings for DFlash block-diffusion speculative decoding.
//!
//! Exposes a thin `DFlashGenerator` class that matches the upstream
//! `dflash_mlx.DFlashGenerator` Python API. The target must currently be a
//! Qwen3 checkpoint — Qwen3.5 support arrives once the qwen3_next GDN
//! verify-input capture lands.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyList;

use pmetal_mlx::Array;
use pmetal_models::DynamicModel;
use pmetal_models::dflash_decoder::{
    DFlashConfig, DFlashDecoder, DFlashOutput, load_dflash_draft_from_dir,
};

use crate::error::runtime_err;
use crate::hub::is_hf_model_id;

/// DFlash block-diffusion speculative decoder.
///
/// Loads a target model (Qwen3) and a DFlash draft, then runs the
/// draft→verify→accept→rollback loop on each `generate` call.
///
/// Example:
///
/// ```python
/// from pmetal import DFlashGenerator
///
/// gen = DFlashGenerator(
///     target_model="mlx-community/Qwen3-4B-bf16",
///     draft_model="z-lab/Qwen3-4B-DFlash-b16",
/// )
/// tokens = gen.generate([1, 2, 3], max_new_tokens=128)
/// ```
///
/// Because the underlying `Qwen3ForCausalLM` is `!Send`, the class is
/// marked `unsendable` — use it from the thread that constructed it.
#[pyclass(name = "DFlashGenerator", unsendable)]
pub struct PyDFlashGenerator {
    decoder: DFlashDecoder<DynamicModel>,
    target_path: PathBuf,
    draft_path: PathBuf,
}

#[pymethods]
impl PyDFlashGenerator {
    /// Load a target Qwen3 model + DFlash draft model.
    ///
    /// Both arguments accept a local directory path or a HuggingFace Hub
    /// model id (downloaded on demand, network I/O releases the GIL).
    #[new]
    #[pyo3(signature = (target_model, draft_model))]
    fn new(py: Python<'_>, target_model: &str, draft_model: &str) -> PyResult<Self> {
        let target_path = resolve_path(py, target_model)?;
        let draft_path = resolve_path(py, draft_model)?;

        // Load target via DynamicModel — DFlashTarget is implemented for
        // the enum, so any architecture with a `forward_with_capture` tap
        // just works. Architectures that don't yet have capture return an
        // explicit runtime error on the first forward.
        let target = DynamicModel::load(&target_path).map_err(runtime_err)?;

        // Load DFlash draft.
        let (draft, report) = load_dflash_draft_from_dir(&draft_path).map_err(runtime_err)?;
        if !report.skipped.is_empty() {
            tracing::info!(
                "DFlashGenerator: loaded {} draft params ({} unused keys)",
                report.loaded,
                report.skipped.len()
            );
        }
        if target.vocab_size() != draft.config.vocab_size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "DFlashGenerator: target vocab_size {} does not match draft vocab_size {}",
                target.vocab_size(),
                draft.config.vocab_size
            )));
        }

        let decoder = DFlashDecoder::new(target, draft);
        Ok(Self {
            decoder,
            target_path,
            draft_path,
        })
    }

    /// Run speculative generation.
    ///
    /// Args:
    ///     prompt_ids: Python list of i32 token ids.
    ///     max_new_tokens: Upper bound on tokens to emit after the prompt.
    ///     temperature: 0.0 (greedy) is bit-identical to baseline decoding.
    ///     stop_tokens: Optional list of token ids that terminate generation.
    ///     speculative_tokens: Override the draft block size.
    ///
    /// Returns a `(tokens, metrics)` tuple where `tokens` is a Python list
    /// of i32 ids (prompt + generated) and `metrics` is a dict with
    /// acceptance statistics.
    #[pyo3(signature = (
        prompt_ids,
        max_new_tokens = 128,
        temperature = 0.0,
        stop_tokens = None,
        speculative_tokens = None,
    ))]
    fn generate<'py>(
        &mut self,
        py: Python<'py>,
        prompt_ids: Vec<i32>,
        max_new_tokens: usize,
        temperature: f32,
        stop_tokens: Option<Vec<i32>>,
        speculative_tokens: Option<usize>,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, pyo3::types::PyDict>)> {
        if prompt_ids.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "prompt_ids must contain at least one token",
            ));
        }

        let prompt_len = prompt_ids.len();
        let prompt = Array::from_slice(prompt_ids.as_slice(), &[1, prompt_len as i32]);

        let config = DFlashConfig {
            max_new_tokens,
            temperature,
            stop_tokens: stop_tokens.unwrap_or_default(),
            speculative_tokens,
            ..Default::default()
        };

        let output: DFlashOutput = self
            .decoder
            .generate(&prompt, &config)
            .map_err(runtime_err)?;

        let token_list = PyList::new(py, output.tokens.iter())?;
        let metrics = metrics_to_py(py, &output)?;
        Ok((token_list, metrics))
    }

    fn __repr__(&self) -> String {
        format!(
            "DFlashGenerator(target='{}', draft='{}', block_size={})",
            self.target_path.display(),
            self.draft_path.display(),
            self.decoder.draft().block_size(),
        )
    }
}

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

fn resolve_path(py: Python<'_>, path_or_id: &str) -> PyResult<PathBuf> {
    if is_hf_model_id(path_or_id) {
        let id = path_or_id.to_string();
        py.detach(move || {
            crate::hub::shared_runtime()
                .block_on(async {
                    let path = pmetal_hub::download_model(&id, None, None).await?;
                    // Pull tokenizer alongside so callers can tokenize outside.
                    let _ = pmetal_hub::download_file(&id, "tokenizer.json", None, None).await;
                    let _ =
                        pmetal_hub::download_file(&id, "tokenizer_config.json", None, None).await;
                    Ok::<_, pmetal_core::PMetalError>(path)
                })
                .map_err(runtime_err)
        })
    } else {
        Ok(PathBuf::from(path_or_id))
    }
}

fn metrics_to_py<'py>(
    py: Python<'py>,
    output: &DFlashOutput,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("num_generated", output.metrics.num_generated)?;
    dict.set_item("total_drafted", output.metrics.total_drafted)?;
    dict.set_item("total_accepted", output.metrics.total_accepted)?;
    dict.set_item(
        "avg_acceptance_length",
        output.metrics.avg_acceptance_length(),
    )?;
    dict.set_item("acceptance_rate", output.metrics.acceptance_rate())?;
    dict.set_item("acceptance_lengths", output.metrics.acceptance_lengths.clone())?;
    Ok(dict)
}
