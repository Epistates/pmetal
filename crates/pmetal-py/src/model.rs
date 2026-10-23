//! `Model`: a model kept loaded for repeated generation.
//!
//! A model's arrays belong to the thread that built them (MLX streams are
//! per-thread) and `DynamicModel` is not `Send`. So the model lives on a
//! thread of its own (`ModelThread`): it is built there, every generation runs
//! there, and it is dropped there. A `Model` handle is the channel to that
//! thread, which makes it `Send + Sync`: any Python thread may call it (one
//! generation at a time runs), and the GIL is released while it works, so a
//! generation does not stop the rest of the program.
//!
//! Ctrl-C ends a generation at its next token and raises `KeyboardInterrupt`.

use crate::runtime::{Inbox, inbox};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pmetal_data::Tokenizer;
use pmetal_models::model_thread::{ModelThread, ModelThreadStartError};
use pmetal_models::{DynamicModel, GenerationConfig};
use pyo3::exceptions::PyValueError;
use pyo3::panic::PanicException;
use pyo3::prelude::*;

use crate::config::{PyGenerationConfig, PyModelArchitecture};
use crate::error::{backend_err, model_load_err, tokenizer_err};
use crate::hub::{HubArgs, resolve_model};
use crate::runtime::recv_interruptible;
use crate::tokenizer::PyTokenizer;

/// What the model thread sends back from a generation.
pub(crate) enum Event {
    /// One token, as it is generated (streaming only).
    Token(u32),
    /// The end, with all the tokens when they were not streamed.
    Done(Result<Vec<u32>, String>),
}

/// Unread tokens a stream lets the model thread get ahead by.
const STREAM_LEAD: usize = 64;

fn thread_gone() -> PyErr {
    PanicException::new_err(
        "the model's thread failed while generating (see the log); reload the model",
    )
}

/// A model, loaded and ready to generate.
///
/// Example:
///
/// ```python
/// model = pmetal.Model.load("Qwen/Qwen3-0.6B")
/// text = model.generate("The capital of France is", max_tokens=16, temperature=0.0)
/// for piece in model.stream("Once upon a time", max_tokens=64):
///     print(piece, end="", flush=True)
/// ```
#[pyclass(name = "Model", module = "pmetal", frozen)]
pub struct PyModel {
    thread: ModelThread<DynamicModel>,
    tokenizer: Arc<Tokenizer>,
    model_path: PathBuf,
    architecture: PyModelArchitecture,
    /// The model's own sampling defaults (its `generation_config.json`).
    defaults: GenerationConfig,
}

impl PyModel {
    /// The prompt's tokens and the generation they ask for.
    fn request(
        &self,
        prompt: &str,
        config: Option<&PyGenerationConfig>,
    ) -> PyResult<(Vec<u32>, GenerationConfig)> {
        let ids = self.tokenizer.encode(prompt).map_err(tokenizer_err)?;
        if ids.is_empty() {
            return Err(PyValueError::new_err("the prompt has no tokens"));
        }
        let mut resolved = config.map_or_else(
            || self.defaults.clone(),
            |c| c.resolve(self.defaults.clone()),
        );
        if let Some(eos) = self.tokenizer.eos_token_id() {
            if !resolved.stop_tokens.contains(&eos) {
                resolved.stop_tokens.push(eos);
            }
        }
        Ok((ids, resolved))
    }

    /// Start a generation on the model thread; its tokens arrive on the
    /// returned channel, ending with `Done`.
    fn start(
        &self,
        ids: Vec<u32>,
        config: GenerationConfig,
        cancel: Arc<AtomicBool>,
        stream: bool,
    ) -> PyResult<Inbox<Event>> {
        let (tx, rx) = inbox(if stream { STREAM_LEAD } else { 1 });
        self.thread
            .submit(move |model: &mut DynamicModel| {
                let sent = tx.clone();
                let mut tokens = Vec::new();
                let result = decode(model, &ids, config, |token| {
                    if stream {
                        // Blocks while the reader is `STREAM_LEAD` behind; fails
                        // once it is gone.
                        if sent.send(Event::Token(token)).is_err() {
                            return false;
                        }
                    } else {
                        tokens.push(token);
                    }
                    !cancel.load(Ordering::SeqCst)
                })
                .map(|()| tokens);
                let _ = tx.send(Event::Done(result));
            })
            .map_err(|_| thread_gone())?;
        Ok(rx)
    }
}

/// Generate from `ids`, calling `on_token` with each token (false stops).
fn decode(
    model: &mut DynamicModel,
    ids: &[u32],
    config: GenerationConfig,
    on_token: impl FnMut(u32) -> bool,
) -> Result<(), String> {
    model
        .generate_streaming(ids, config, on_token)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[pymethods]
impl PyModel {
    /// Load a model from a local directory or a Hub id.
    ///
    /// The download (Ctrl-C interrupts it) and the load both run with the GIL
    /// released.
    ///
    /// Args:
    ///     path_or_id: A model directory, or a Hub id (`org/name`, or `org/name:Q4_K_M` for one GGUF quantization).
    ///     fp8: Quantize the weights to FP8 as they load (about half the memory).
    ///     revision: Branch, tag or commit of a Hub model.
    ///     token: Hub access token; `HF_TOKEN` or `hf auth login` when None.
    #[staticmethod]
    #[pyo3(signature = (path_or_id, fp8=false, *, revision=None, token=None))]
    fn load(
        py: Python<'_>,
        path_or_id: PathBuf,
        fp8: bool,
        revision: Option<String>,
        token: Option<String>,
    ) -> PyResult<Self> {
        let hub = HubArgs {
            revision,
            token,
            ..HubArgs::default()
        };
        let model_path = resolve_model(py, &path_or_id.to_string_lossy(), &hub)?;
        let tokenizer = Arc::new(crate::tokenizer::load_from_dir(&model_path)?);
        let defaults = GenerationConfig::from_model_dir(&model_path);

        let path = model_path.clone();
        let thread = py.detach(move || {
            ModelThread::spawn("pmetal-model", move || {
                let mut model = DynamicModel::load(&path).map_err(|e| e.to_string())?;
                if fp8 {
                    model.quantize_fp8().map_err(|e| e.to_string())?;
                }
                Ok::<_, String>(model)
            })
        });
        let thread = thread.map_err(|e| match e {
            ModelThreadStartError::Init(msg) => model_load_err(msg),
            other => backend_err(other),
        })?;
        let architecture = thread
            .call(|model| model.architecture())
            .map_err(|_| thread_gone())?;
        Ok(Self {
            thread,
            tokenizer,
            model_path,
            architecture: architecture.into(),
            defaults,
        })
    }

    /// Generate text continuing `prompt`.
    ///
    /// The prompt is tokenized as given: no chat template is applied. Sampling
    /// starts from the model's own defaults (its `generation_config.json`);
    /// `config` and the keyword arguments override them.
    ///
    /// Args:
    ///     prompt: The text to continue.
    ///     config: A `GenerationConfig`; every field it leaves unset keeps the model's default.
    ///     max_tokens: At most this many new tokens.
    ///     temperature: Sampling temperature; 0 or less is greedy.
    ///     top_k: Keep the k likeliest tokens (0 keeps all).
    ///     top_p: Keep the smallest set of likeliest tokens whose probabilities reach top_p.
    ///     min_p: Drop tokens less than min_p times as likely as the likeliest.
    ///     repetition_penalty: Above 1 discourages repeats.
    ///     frequency_penalty: Subtracted per previous occurrence of a token.
    ///     presence_penalty: Subtracted once from every token that occurred.
    ///     seed: Seeds the draws: the same seed gives the same text.
    ///     stop_tokens: Token ids that end the generation, besides the end-of-sequence token.
    ///     logit_bias: Added to token logits before sampling, `{token_id: bias}`.
    ///
    /// Returns:
    ///     The generated text (the prompt is not included).
    #[pyo3(signature = (prompt, config=None, *, max_tokens=None, temperature=None, top_k=None,
        top_p=None, min_p=None, repetition_penalty=None, frequency_penalty=None,
        presence_penalty=None, seed=None, stop_tokens=None, logit_bias=None))]
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: &str,
        config: Option<&PyGenerationConfig>,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        seed: Option<u64>,
        stop_tokens: Option<Vec<u32>>,
        logit_bias: Option<std::collections::HashMap<u32, f32>>,
    ) -> PyResult<String> {
        let overrides = PyGenerationConfig::from_parts(
            config,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed,
            stop_tokens,
            logit_bias,
        );
        let (ids, resolved) = self.request(prompt, Some(&overrides))?;
        let cancel = Arc::new(AtomicBool::new(false));
        let rx = self.start(ids, resolved, cancel.clone(), false)?;

        match recv_interruptible(py, &rx, &cancel)? {
            Some(Event::Done(Ok(generated))) => {
                let tokenizer = &self.tokenizer;
                py.detach(|| tokenizer.decode(&generated))
                    .map_err(tokenizer_err)
            }
            Some(Event::Done(Err(message))) => Err(backend_err(message)),
            Some(Event::Token(_)) | None => Err(thread_gone()),
        }
    }

    /// Like `generate`, but yields the text piece by piece as it is produced.
    ///
    /// Closing the iterator (or dropping it) ends the generation.
    #[pyo3(signature = (prompt, config=None, *, max_tokens=None, temperature=None, top_k=None,
        top_p=None, min_p=None, repetition_penalty=None, frequency_penalty=None,
        presence_penalty=None, seed=None, stop_tokens=None, logit_bias=None))]
    #[allow(clippy::too_many_arguments)]
    fn stream(
        &self,
        prompt: &str,
        config: Option<&PyGenerationConfig>,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        seed: Option<u64>,
        stop_tokens: Option<Vec<u32>>,
        logit_bias: Option<std::collections::HashMap<u32, f32>>,
    ) -> PyResult<PyTextStream> {
        let overrides = PyGenerationConfig::from_parts(
            config,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed,
            stop_tokens,
            logit_bias,
        );
        let (ids, resolved) = self.request(prompt, Some(&overrides))?;
        let cancel = Arc::new(AtomicBool::new(false));
        let rx = self.start(ids, resolved, cancel.clone(), true)?;
        Ok(PyTextStream {
            tokenizer: self.tokenizer.clone(),
            cancel,
            state: Mutex::new(StreamState {
                rx: Some(rx),
                tokens: Vec::new(),
                emitted: 0,
            }),
        })
    }

    /// The model's tokenizer.
    #[getter]
    fn tokenizer(&self) -> PyTokenizer {
        PyTokenizer {
            inner: self.tokenizer.clone(),
        }
    }

    /// The model's architecture.
    #[getter]
    fn architecture(&self) -> PyModelArchitecture {
        self.architecture.clone()
    }

    /// Where the model's files are.
    #[getter]
    fn path(&self) -> String {
        self.model_path.to_string_lossy().into_owned()
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(path={:?}, architecture={:?})",
            self.model_path.display().to_string(),
            self.architecture
        )
    }
}

struct StreamState {
    rx: Option<Inbox<Event>>,
    tokens: Vec<u32>,
    /// Bytes of the decoded text already yielded.
    emitted: usize,
}

/// The pieces of one streamed generation.
#[pyclass(name = "TextStream", module = "pmetal", frozen)]
pub struct PyTextStream {
    tokenizer: Arc<Tokenizer>,
    cancel: Arc<AtomicBool>,
    state: Mutex<StreamState>,
}

impl Drop for PyTextStream {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

impl PyTextStream {
    fn lock(&self) -> std::sync::MutexGuard<'_, StreamState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
}

#[pymethods]
impl PyTextStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<String>> {
        // The receiver is taken out for the wait, so the lock is never held
        // across it; a second reader meanwhile finds the stream ended.
        let Some(rx) = self.lock().rx.take() else {
            return Ok(None);
        };
        loop {
            match recv_interruptible(py, &rx, &self.cancel)? {
                Some(Event::Token(token)) => {
                    let mut state = self.lock();
                    state.tokens.push(token);
                    let text = self
                        .tokenizer
                        .decode(&state.tokens)
                        .map_err(tokenizer_err)?;
                    // A token can end mid-character; wait for the rest of it.
                    if text.ends_with('\u{FFFD}') {
                        continue;
                    }
                    if let Some(piece) = text.get(state.emitted..).filter(|p| !p.is_empty()) {
                        let piece = piece.to_string();
                        state.emitted = text.len();
                        state.rx = Some(rx);
                        return Ok(Some(piece));
                    }
                }
                Some(Event::Done(Ok(_))) => {
                    let state = self.lock();
                    let text = self
                        .tokenizer
                        .decode(&state.tokens)
                        .map_err(tokenizer_err)?;
                    return Ok(text
                        .get(state.emitted..)
                        .filter(|p| !p.is_empty())
                        .map(str::to_string));
                }
                Some(Event::Done(Err(message))) => return Err(backend_err(message)),
                None => return Err(thread_gone()),
            }
        }
    }

    /// End the generation; the iterator then yields nothing more.
    fn close(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        self.lock().rx = None;
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, pyo3::types::PyTuple>) {
        self.close();
    }
}
