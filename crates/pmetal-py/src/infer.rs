//! `infer` and `infer_with_metrics`: one prompt through `pmetal infer`'s
//! inference runner, in this process.
//!
//! The options are the fields of `InferSpec`, with its defaults and checks. A
//! field `None` here is the same `None` as there (for the sampling fields: the
//! model's own default). The model is loaded for the call; for many prompts
//! against one model, keep a `Model`.
//!
//! Some `InferSpec` fields are things only the command line does (images and
//! video, the Neural Engine, benchmarks, profiles, JSON output, tools). Setting
//! one raises `ConfigError` naming it, rather than ignoring it; run the spec
//! with `pmetal.run` to use them.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pmetal_core::jobs::InferSpec;
use pmetal_data::chat_templates::ChatTemplateKwargs;
use pmetal_data::inference_config::SamplingMode;
use pmetal_lib::inference_runner::{InferenceRunner, InferenceRunnerConfig, TurboQuantPreset};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{backend_err, config_err};
use crate::hub::{HubArgs, resolve_model};
use crate::spec;

/// The `InferSpec` fields only the command line honours.
pub(crate) const COMMAND_LINE_ONLY: &[&str] = &[
    "image",
    "video",
    "video_fps",
    "json",
    "hide_thinking",
    "backend",
    "tools",
    "ane",
    "ane_max_seq_len",
    "benchmark",
    "benchmark_iters",
    "benchmark_prompt_tokens",
    "profile_layers",
    "profile_output",
];

/// The `InferSpec` fields `infer` maps onto the runner (the test below holds
/// the two lists to the spec).
#[cfg(test)]
pub(crate) const MAPPED: &[&str] = &[
    "model",
    "lora",
    "prompt",
    "max_tokens",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "chat",
    "system",
    "no_thinking",
    "reasoning_effort",
    "no_preserve_thinking",
    "chat_template_kwargs",
    "mode",
    "draft_model",
    "mtp",
    "mtp_model",
    "mtp_draft_tokens",
    "fp8",
    "experts_dir",
    "kv_quant",
    "kv_k_bits",
    "kv_v_bits",
    "kv_group_size",
    "kv_turboquant",
    "kv_turboquant_preset",
    "kv_quant_preset",
    "no_kv_quant",
    "kv_qjl",
    "detect_repetition",
];

/// What a generation produced.
#[pyclass(name = "InferResult", module = "pmetal", frozen, skip_from_py_object)]
pub struct PyInferResult {
    /// The generated text (the prompt is not included).
    #[pyo3(get)]
    text: String,
    /// Tokens generated.
    #[pyo3(get)]
    num_generated: usize,
    /// A stop token ended the generation.
    #[pyo3(get)]
    stopped_by_token: bool,
    /// The token limit ended the generation.
    #[pyo3(get)]
    stopped_by_length: bool,
    /// Seconds from the first token to the last.
    #[pyo3(get)]
    generation_seconds: f64,
    /// Speculative-decoding statistics (drafted, accepted, acceptance rate, ...), or None for plain decoding.
    #[pyo3(get)]
    speculative: Option<Py<PyDict>>,
}

#[pymethods]
impl PyInferResult {
    /// The result as a dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("text", &self.text)?;
        d.set_item("num_generated", self.num_generated)?;
        d.set_item("stopped_by_token", self.stopped_by_token)?;
        d.set_item("stopped_by_length", self.stopped_by_length)?;
        d.set_item("generation_seconds", self.generation_seconds)?;
        d.set_item("speculative", self.speculative.as_ref().map(|s| s.bind(py)))?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "InferResult(num_generated={}, stopped_by_token={}, text={:?})",
            self.num_generated, self.stopped_by_token, self.text
        )
    }
}

/// The spec for a call's `model`, `prompt` and keyword options, validated.
fn spec_for(
    py: Python<'_>,
    model: &Path,
    prompt: &str,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<InferSpec> {
    let fields = match options {
        Some(options) => options.copy()?,
        None => PyDict::new(py),
    };
    fields.set_item("model", model.to_string_lossy())?;
    fields.set_item("prompt", prompt)?;
    let mut spec = spec::new::<InferSpec>(Some(&fields))?;

    let reject = command_line_only_fields_set(&spec)?;
    if !reject.is_empty() {
        return Err(config_err(format!(
            "infer() does not support {}: the command line does. Run the spec with \
             pmetal.run(pmetal.InferSpec(...)) to use {}.",
            reject.join(", "),
            if reject.len() == 1 { "it" } else { "them" }
        )));
    }
    spec::normalize(&mut spec)?;
    Ok(spec)
}

/// The command-line-only fields `spec` sets away from their defaults.
fn command_line_only_fields_set(spec: &InferSpec) -> PyResult<Vec<&'static str>> {
    let set = serde_json::to_value(spec).map_err(config_err)?;
    let default = serde_json::to_value(InferSpec::default()).map_err(config_err)?;
    Ok(COMMAND_LINE_ONLY
        .iter()
        .copied()
        .filter(|name| set.get(name) != default.get(name))
        .collect())
}

fn turboquant_preset(name: Option<&str>) -> PyResult<Option<TurboQuantPreset>> {
    match name {
        None => Ok(None),
        Some("q2_5") => Ok(Some(TurboQuantPreset::Q2_5)),
        Some("q3_5") => Ok(Some(TurboQuantPreset::Q3_5)),
        Some(other) => Err(config_err(format!(
            "kv_turboquant_preset: unknown preset '{other}'; valid: q2_5, q3_5"
        ))),
    }
}

/// The runner's configuration for `spec`, with the model paths resolved.
fn runner_config(py: Python<'_>, spec: &InferSpec) -> PyResult<InferenceRunnerConfig> {
    if spec.mtp && spec.draft_model.is_some() {
        return Err(config_err(
            "use either mtp (bundled Qwen MTP) or draft_model (Gemma 4 MTP), not both",
        ));
    }
    let hub = HubArgs::default();
    let model_path = resolve_model(py, &spec.model, &hub)?;
    let draft_path = spec
        .draft_model
        .as_deref()
        .map(|draft| resolve_model(py, draft, &hub))
        .transpose()?;
    if draft_path
        .as_deref()
        .is_some_and(pmetal_models::dflash_drafter::is_dflash_draft)
    {
        return Err(config_err(
            "draft_model is a DFlash draft, which drafts for the Neural Engine on the command \
             line and for a target held by pmetal.DFlashGenerator here",
        ));
    }
    let qwen_mtp_path = spec
        .mtp_model
        .as_deref()
        .map(|path| resolve_model(py, path, &hub))
        .transpose()?;

    let chat_template_kwargs = ChatTemplateKwargs::from_controls(
        spec.chat_template_kwargs.as_deref(),
        spec.no_thinking,
        spec.reasoning_effort.as_deref(),
        spec.no_preserve_thinking,
    )
    .map_err(config_err)?;
    // `chat_template_kwargs='{"enable_thinking": false}'` is `no_thinking`.
    let no_thinking = spec.no_thinking || chat_template_kwargs.enable_thinking() == Some(false);
    let mode: SamplingMode = spec.mode.parse().map_err(config_err)?;

    Ok(InferenceRunnerConfig {
        model_path,
        lora_path: spec.lora.clone(),
        mtp_assistant_path: draft_path,
        qwen_mtp_path,
        qwen_mtp: spec.mtp,
        qwen_mtp_draft_tokens: spec.mtp_draft_tokens.max(1),
        experts_dir: spec.experts_dir.clone(),
        fp8: spec.fp8,
        prompt: spec.prompt.clone(),
        chat_messages: None,
        system_message: spec.system.clone(),
        chat: spec.chat,
        no_thinking,
        chat_template_kwargs,
        tools: None,
        images: Vec::new(),
        videos: Vec::new(),
        temperature: spec.temperature,
        top_k: spec.top_k,
        top_p: spec.top_p,
        min_p: spec.min_p,
        max_tokens: spec.max_tokens,
        repetition_penalty: spec.repetition_penalty,
        frequency_penalty: spec.frequency_penalty,
        presence_penalty: spec.presence_penalty,
        seed: spec.seed,
        kv_quant: spec.kv_quant,
        kv_k_bits: spec.kv_k_bits,
        kv_v_bits: spec.kv_v_bits,
        kv_group_size: spec.kv_group_size,
        kv_turboquant: spec.kv_turboquant,
        kv_turboquant_preset: turboquant_preset(spec.kv_turboquant_preset.as_deref())?,
        kv_quant_preset: spec.kv_quant_preset.clone(),
        no_kv_quant: spec.no_kv_quant,
        mode,
        detect_repetition: spec.detect_repetition,
        kv_qjl: spec.kv_qjl,
    })
}

/// How often a generating thread looks for Ctrl-C.
const SIGNAL_POLL: Duration = Duration::from_millis(25);

fn run(py: Python<'_>, spec: &InferSpec) -> PyResult<PyInferResult> {
    let config = runner_config(py, spec)?;
    let interrupt: Arc<Mutex<Option<PyErr>>> = Arc::default();
    let seen = interrupt.clone();

    let outcome = py.detach(move || -> Result<_, String> {
        let mut runner = InferenceRunner::prepare(config).map_err(|e| e.to_string())?;
        let mut generated = Vec::new();
        let mut last_check = Instant::now();
        let started = Instant::now();
        let output = runner
            .state
            .generate_streaming(|token| {
                generated.push(token);
                if last_check.elapsed() < SIGNAL_POLL {
                    return true;
                }
                last_check = Instant::now();
                // Only the main thread sees Ctrl-C; elsewhere this is Ok.
                match Python::attach(|py| py.check_signals()) {
                    Ok(()) => true,
                    Err(e) => {
                        *seen.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
                        false
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        let text = runner
            .tokenizer
            .decode(&generated)
            .map_err(|e| e.to_string())?;
        Ok((text, output, started.elapsed()))
    });

    if let Some(e) = interrupt.lock().unwrap_or_else(|p| p.into_inner()).take() {
        return Err(e);
    }
    let (text, output, elapsed) = outcome.map_err(backend_err)?;
    let speculative = output
        .speculative_metrics
        .map(|m| -> PyResult<Py<PyDict>> {
            let d = PyDict::new(py);
            d.set_item("drafted_tokens", m.drafted_tokens)?;
            d.set_item("accepted_draft_tokens", m.accepted_draft_tokens)?;
            d.set_item("emitted_draft_tokens", m.emitted_draft_tokens)?;
            d.set_item("bonus_tokens", m.bonus_tokens)?;
            d.set_item("correction_tokens", m.correction_tokens)?;
            d.set_item("verify_steps", m.verify_steps)?;
            d.set_item("full_accept_steps", m.full_accept_steps)?;
            d.set_item("partial_accept_steps", m.partial_accept_steps)?;
            d.set_item("zero_accept_steps", m.zero_accept_steps)?;
            d.set_item("max_draft_tokens", m.max_draft_tokens)?;
            d.set_item("acceptance_rate", m.acceptance_rate())?;
            d.set_item("avg_accepted_per_verify", m.avg_accepted_per_verify())?;
            Ok(d.unbind())
        })
        .transpose()?;
    Ok(PyInferResult {
        text,
        num_generated: output.num_generated,
        stopped_by_token: output.stopped_by_token,
        stopped_by_length: output.stopped_by_length,
        generation_seconds: elapsed.as_secs_f64(),
        speculative,
    })
}

/// Generate text for one prompt and return it.
///
/// The model is loaded for the call. Options are the fields of
/// `pmetal.InferSpec` (see `InferSpec.fields()`), with the same defaults: a
/// sampling option left out takes the model's own value.
///
/// Raises:
///     ConfigError: An option is invalid, or is one only the command line honours.
///     KeyboardInterrupt: On Ctrl-C, after the token in flight.
#[pyfunction]
#[pyo3(signature = (model, prompt, **options))]
pub fn infer(
    py: Python<'_>,
    model: PathBuf,
    prompt: &str,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let spec = spec_for(py, &model, prompt, options)?;
    run(py, &spec).map(|r| r.text)
}

/// Like `infer`, with the generation's statistics: token counts, why it
/// stopped, its time, and the acceptance statistics of speculative decoding.
#[pyfunction]
#[pyo3(signature = (model, prompt, **options))]
pub fn infer_with_metrics(
    py: Python<'_>,
    model: PathBuf,
    prompt: &str,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyInferResult> {
    let spec = spec_for(py, &model, prompt, options)?;
    run(py, &spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `InferSpec` field is either mapped onto the runner or named as
    /// command-line-only, so a field added to the spec cannot be silently
    /// dropped by `infer`.
    #[test]
    fn every_infer_spec_field_is_accounted_for() {
        let value = serde_json::to_value(InferSpec::default()).unwrap();
        let fields: Vec<&String> = value.as_object().unwrap().keys().collect();
        for field in &fields {
            let mapped = MAPPED.contains(&field.as_str());
            let cli_only = COMMAND_LINE_ONLY.contains(&field.as_str());
            assert!(
                mapped ^ cli_only,
                "InferSpec.{field} must be in exactly one of MAPPED and COMMAND_LINE_ONLY"
            );
        }
        for name in MAPPED.iter().chain(COMMAND_LINE_ONLY) {
            assert!(
                fields.iter().any(|f| f.as_str() == *name),
                "{name} is not an InferSpec field"
            );
        }
    }
}
