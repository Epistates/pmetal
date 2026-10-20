//! `DFlashGenerator`: block-diffusion speculative decoding.
//!
//! The target and the draft are loaded on a thread of their own (see
//! `model.rs`), so the handle is `Send + Sync`, the GIL is released while a
//! generation runs, and Ctrl-C ends it after the step in flight.

use crate::runtime::inbox;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use pmetal_core::jobs::DflashSpec;
use pmetal_models::dflash_decoder::{DFlashConfig, DFlashDecoder, DFlashOutput};
use pmetal_models::dflash_drafts::{DFlashDraftQuant, load_dflash_draft};
use pmetal_models::model_thread::{ModelThread, ModelThreadStartError};
use pmetal_models::{DynamicModel, GenerationConfig};
use pyo3::exceptions::PyValueError;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::{backend_err, model_load_err};
use crate::hub::{HubArgs, resolve_model};
use crate::runtime::recv_interruptible;

/// `pmetal dflash` draws from the target's own distribution, untruncated: no
/// top-k, no top-p, no min-p, no repetition penalty. These are those
/// settings, for a caller that names none of them.
const DEFAULT_TOP_K: usize = 0;
const DEFAULT_TOP_P: f32 = 1.0;
const DEFAULT_MIN_P: f32 = 0.0;

/// DFlash block-diffusion speculative decoder.
///
/// Loads a target model and a DFlash draft, then runs the
/// draft, verify, accept, roll-back loop on each `generate` call. The draft
/// only changes the speed: the tokens are the target's own, as plain decoding
/// draws them.
///
/// ```python
/// gen = pmetal.DFlashGenerator(
///     target_model="mlx-community/Qwen3-4B-bf16",
///     draft_model="z-lab/Qwen3-4B-DFlash-b16",
/// )
/// tokens, metrics = gen.generate(prompt_ids, max_new_tokens=128)
/// ```
#[pyclass(name = "DFlashGenerator", module = "pmetal", frozen)]
pub struct PyDFlashGenerator {
    thread: ModelThread<DFlashDecoder<DynamicModel>>,
    target_path: PathBuf,
    draft_path: PathBuf,
    block_size: usize,
}

#[pymethods]
impl PyDFlashGenerator {
    /// Load a target model and its DFlash draft.
    ///
    /// Both arguments take a local directory or a Hub id (downloaded with the
    /// GIL released; Ctrl-C interrupts it).
    ///
    /// Raises:
    ///     ModelLoadError: A checkpoint did not load.
    ///     ValueError: The target and the draft disagree on the vocabulary.
    #[new]
    #[pyo3(signature = (target_model, draft_model, *, revision=None, token=None))]
    fn new(
        py: Python<'_>,
        target_model: PathBuf,
        draft_model: PathBuf,
        revision: Option<String>,
        token: Option<String>,
    ) -> PyResult<Self> {
        let hub = HubArgs {
            revision,
            token,
            ..HubArgs::default()
        };
        let target_path = resolve_model(py, &target_model.to_string_lossy(), &hub)?;
        let draft_path = resolve_model(py, &draft_model.to_string_lossy(), &hub)?;

        let (target, draft) = (target_path.clone(), draft_path.clone());
        let started = py.detach(move || {
            ModelThread::spawn("pmetal-dflash", move || {
                // `DFlashTarget` is implemented for the dispatcher, so any
                // architecture with a hidden-state tap works as a target.
                let target = DynamicModel::load(&target).map_err(|e| e.to_string())?;
                let draft =
                    load_dflash_draft(&draft, DFlashDraftQuant::None).map_err(|e| e.to_string())?;
                if target.vocab_size() != draft.config().vocab_size {
                    return Err(format!(
                        "the target's vocab_size {} does not match the draft's {}",
                        target.vocab_size(),
                        draft.config().vocab_size
                    ));
                }
                Ok(DFlashDecoder::new(target, draft))
            })
        });
        let thread = started.map_err(|e| match e {
            ModelThreadStartError::Init(msg) if msg.contains("does not match the draft") => {
                PyValueError::new_err(format!("DFlashGenerator: {msg}"))
            }
            ModelThreadStartError::Init(msg) => model_load_err(msg),
            other => backend_err(other),
        })?;
        let block_size = thread
            .call(|decoder| decoder.draft().block_size())
            .map_err(|_| thread_gone())?;
        Ok(Self {
            thread,
            target_path,
            draft_path,
            block_size,
        })
    }

    /// The draft's block size: the most tokens one verify step can accept.
    #[getter]
    fn block_size(&self) -> usize {
        self.block_size
    }

    /// Run speculative generation from token ids.
    ///
    /// Args:
    ///     prompt_ids: The prompt's token ids (at least one).
    ///     max_new_tokens: The most tokens to emit after the prompt (`DflashSpec`'s default when None).
    ///     temperature: 0 is greedy; above it each token is drawn at this temperature, after top_k, top_p and min_p (`DflashSpec`'s default when None).
    ///     stop_tokens: Token ids that end the generation (they are not in the output).
    ///     speculative_tokens: Override the draft's block size.
    ///     top_k: Keep the k likeliest tokens (0 keeps all).
    ///     top_p: Keep the smallest set of likeliest tokens whose probabilities reach top_p (1.0 keeps all).
    ///     min_p: Drop tokens less than min_p times as likely as the likeliest (0.0 keeps all).
    ///     seed: Seeds the draws: the same seed gives the same tokens.
    ///
    /// Returns:
    ///     `(tokens, metrics)`: the prompt followed by the generated token
    ///     ids, and a dict of acceptance statistics.
    ///
    /// Raises:
    ///     KeyboardInterrupt: On Ctrl-C, after the step in flight.
    #[pyo3(signature = (prompt_ids, max_new_tokens=None, temperature=None, stop_tokens=None,
        speculative_tokens=None, top_k=None, top_p=None, min_p=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn generate<'py>(
        &self,
        py: Python<'py>,
        prompt_ids: Vec<u32>,
        max_new_tokens: Option<usize>,
        temperature: Option<f32>,
        stop_tokens: Option<Vec<u32>>,
        speculative_tokens: Option<usize>,
        top_k: Option<usize>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        seed: Option<u64>,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyDict>)> {
        if prompt_ids.is_empty() {
            return Err(PyValueError::new_err(
                "prompt_ids must contain at least one token",
            ));
        }
        let defaults = DflashSpec::default();
        let max_new_tokens = max_new_tokens.unwrap_or(defaults.max_new_tokens);
        let temperature = temperature.unwrap_or(defaults.temperature);

        let mut generation = if temperature > 0.0 {
            GenerationConfig::sampling(max_new_tokens, temperature)
                .with_top_k(top_k.unwrap_or(DEFAULT_TOP_K))
                .with_top_p(top_p.unwrap_or(DEFAULT_TOP_P))
                .with_min_p(min_p.unwrap_or(DEFAULT_MIN_P))
                .with_repetition_penalty(1.0)
        } else {
            GenerationConfig::greedy(max_new_tokens)
        }
        .with_stop_tokens(stop_tokens.unwrap_or_default());
        if let Some(seed) = seed {
            generation = generation.with_seed(seed);
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let config = DFlashConfig {
            speculative_tokens,
            cancel: Some(cancel.clone()),
            ..DFlashConfig::new(generation)
        };

        let (tx, rx) = inbox(1);
        self.thread
            .submit(move |decoder| {
                let output = decoder
                    .generate(&prompt_ids, &config)
                    .map_err(|e| e.to_string());
                let _ = tx.send(output);
            })
            .map_err(|_| thread_gone())?;

        let output: DFlashOutput = recv_interruptible(py, &rx, &cancel)?
            .ok_or_else(thread_gone)?
            .map_err(backend_err)?;
        Ok((
            PyList::new(py, output.tokens.iter())?,
            metrics_to_py(py, &output)?,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "DFlashGenerator(target={:?}, draft={:?}, block_size={})",
            self.target_path.display().to_string(),
            self.draft_path.display().to_string(),
            self.block_size
        )
    }
}

fn thread_gone() -> PyErr {
    PanicException::new_err(
        "the DFlash thread failed while generating (see the log); reload the generator",
    )
}

fn metrics_to_py<'py>(py: Python<'py>, output: &DFlashOutput) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("num_generated", output.metrics.num_generated)?;
    dict.set_item("total_drafted", output.metrics.total_drafted)?;
    dict.set_item("total_accepted", output.metrics.total_accepted)?;
    dict.set_item(
        "avg_acceptance_length",
        output.metrics.avg_acceptance_length(),
    )?;
    dict.set_item("acceptance_rate", output.metrics.acceptance_rate())?;
    dict.set_item(
        "acceptance_lengths",
        output.metrics.acceptance_lengths.clone(),
    )?;
    Ok(dict)
}
