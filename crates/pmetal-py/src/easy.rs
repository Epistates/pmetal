//! Top-level easy API functions for Python.

use std::path::{Path, PathBuf};

use pmetal_core::LrSchedulerType;
use pmetal_data::streaming::{StreamConfig, StreamingShardReader};
use pmetal_models::DynamicModel;
use pmetal_models::architectures::{
    DFlashDraftConfig, DFlashDraftModel, Gemma4AssistantForCausalLM, Qwen3NextMtpForCausalLM,
    load_gemma4_assistant_from_dir, load_qwen3_next_mtp_from_dir,
};
use pmetal_models::dflash_decoder::load_dflash_draft_from_dir;
use pmetal_trainer::{
    MtpTrainingConfig, MtpTrainingResult, batch_from_tokens, initialize_qwen3_next_mtp_from_target,
    make_gemma4_assistant_config, make_qwen3_next_mtp_config, run_dflash_draft_training,
    run_gemma4_assistant_mtp_training, run_qwen3_next_mtp_training,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::runtime_err;

struct InferRunResult {
    text: String,
    num_generated: usize,
    stopped_by_token: bool,
    stopped_by_length: bool,
    speculative_metrics: Option<pmetal_models::SpeculativeDecodeMetrics>,
}

#[allow(clippy::struct_excessive_bools)]
struct InferOptions {
    model_id: String,
    prompt: String,
    lora: Option<String>,
    max_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    fp8: bool,
    experts_dir: Option<String>,
    draft_model: Option<String>,
    mtp: bool,
    mtp_model: Option<String>,
    mtp_draft_tokens: usize,
    chat: bool,
    system_message: Option<String>,
    no_thinking: bool,
    kv_quant: Option<u8>,
    no_kv_quant: bool,
    detect_repetition: bool,
}

/// Fine-tune a model with sensible defaults.
///
/// Args:
///     model_id: HuggingFace model ID or local path
///     dataset_path: Path to JSONL training dataset
///     lora_r: LoRA rank (default 16)
///     lora_alpha: LoRA alpha (default 32.0)
///     epochs: Number of training epochs (default 3)
///     learning_rate: Learning rate (default 2e-4)
///     batch_size: Batch size (default 4)
///     max_seq_len: Maximum sequence length (default 2048)
///     output: Output directory (default "./output")
///
/// Returns:
///     dict with keys: final_loss, total_steps, total_tokens, output_dir, lora_weights_path
#[pyfunction]
#[pyo3(signature = (
    model_id,
    dataset_path,
    lora_r=16,
    lora_alpha=32.0,
    epochs=3,
    learning_rate=2e-4,
    batch_size=4,
    max_seq_len=2048,
    output="./output",
))]
#[allow(clippy::too_many_arguments)]
pub fn finetune<'py>(
    py: Python<'py>,
    model_id: &str,
    dataset_path: &str,
    lora_r: usize,
    lora_alpha: f32,
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
    max_seq_len: usize,
    output: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let model_id = model_id.to_string();
    let dataset_path = dataset_path.to_string();
    let output = output.to_string();

    let result = py
        .detach(move || {
            crate::hub::shared_runtime().block_on(async {
                use pmetal_trainer::orchestrator;

                let job_config = orchestrator::TrainingJobConfig {
                    model_id: model_id.clone(),
                    dataset: dataset_path,
                    eval_dataset: None,
                    output_dir: output,
                    lora: pmetal_core::LoraConfig {
                        r: lora_r,
                        alpha: lora_alpha,
                        ..Default::default()
                    },
                    qlora: None,
                    training: pmetal_core::TrainingConfig {
                        learning_rate,
                        batch_size,
                        num_epochs: epochs,
                        max_seq_len,
                        ..Default::default()
                    },
                    columns: None,
                    dispatch: orchestrator::DispatchConfig::default(),
                    config_path: None,
                    log_metrics: None,
                    resume: false,
                    seed: 42,
                    emit_console_output: false,
                };

                let result = orchestrator::run_training(job_config, None, vec![])
                    .await
                    .map_err(|e| e.to_string())?;

                Ok::<_, String>((
                    result.final_loss,
                    result.total_steps,
                    result.total_tokens,
                    result.output_dir.to_string_lossy().to_string(),
                    result.lora_weights_path.to_string_lossy().to_string(),
                ))
            })
        })
        .map_err(runtime_err)?;

    let dict = PyDict::new(py);
    dict.set_item("final_loss", result.0)?;
    dict.set_item("total_steps", result.1)?;
    dict.set_item("total_tokens", result.2)?;
    dict.set_item("output_dir", result.3)?;
    dict.set_item("lora_weights_path", result.4)?;
    Ok(dict)
}

/// Run inference with a model.
///
/// Args:
///     model_id: HuggingFace model ID or local path
///     prompt: Text prompt
///     lora: Optional path to LoRA weights
///     max_tokens: Maximum tokens to generate (default 256)
///     temperature: Sampling temperature (default 0.7)
///     seed: Random seed for reproducibility
///     draft_model: Optional Gemma 4 MTP assistant model
///     mtp: Enable bundled Qwen3Next/Qwen3.6 MTP weights
///     mtp_model: Optional Qwen MTP checkpoint
///     chat: Force chat-template formatting
///     no_thinking: Disable thinking mode for models/templates that support it
///     kv_quant: KV cache quantization bits (8, 4, or 0)
///     detect_repetition: Stop on repeated n-gram loops
///
/// Returns:
///     Generated text string
#[pyfunction]
#[pyo3(signature = (
    model_id,
    prompt,
    lora=None,
    max_tokens=256,
    temperature=0.7,
    seed=None,
    top_k=None,
    top_p=None,
    min_p=None,
    repetition_penalty=None,
    frequency_penalty=None,
    presence_penalty=None,
    fp8=false,
    experts_dir=None,
    draft_model=None,
    mtp=false,
    mtp_model=None,
    mtp_draft_tokens=3,
    chat=false,
    system_message=None,
    no_thinking=false,
    kv_quant=None,
    no_kv_quant=false,
    detect_repetition=false,
))]
#[allow(clippy::too_many_arguments)]
pub fn infer(
    py: Python<'_>,
    model_id: &str,
    prompt: &str,
    lora: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    fp8: bool,
    experts_dir: Option<&str>,
    draft_model: Option<&str>,
    mtp: bool,
    mtp_model: Option<&str>,
    mtp_draft_tokens: usize,
    chat: bool,
    system_message: Option<&str>,
    no_thinking: bool,
    kv_quant: Option<u8>,
    no_kv_quant: bool,
    detect_repetition: bool,
) -> PyResult<String> {
    run_infer(
        py,
        InferOptions::new(
            model_id,
            prompt,
            lora,
            max_tokens,
            temperature,
            seed,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            fp8,
            experts_dir,
            draft_model,
            mtp,
            mtp_model,
            mtp_draft_tokens,
            chat,
            system_message,
            no_thinking,
            kv_quant,
            no_kv_quant,
            detect_repetition,
        ),
    )
    .map(|result| result.text)
}

/// Run inference and return generation/speculative decoding metrics.
#[pyfunction]
#[pyo3(signature = (
    model_id,
    prompt,
    lora=None,
    max_tokens=256,
    temperature=0.7,
    seed=None,
    top_k=None,
    top_p=None,
    min_p=None,
    repetition_penalty=None,
    frequency_penalty=None,
    presence_penalty=None,
    fp8=false,
    experts_dir=None,
    draft_model=None,
    mtp=false,
    mtp_model=None,
    mtp_draft_tokens=3,
    chat=false,
    system_message=None,
    no_thinking=false,
    kv_quant=None,
    no_kv_quant=false,
    detect_repetition=false,
))]
#[allow(clippy::too_many_arguments)]
pub fn infer_with_metrics<'py>(
    py: Python<'py>,
    model_id: &str,
    prompt: &str,
    lora: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    fp8: bool,
    experts_dir: Option<&str>,
    draft_model: Option<&str>,
    mtp: bool,
    mtp_model: Option<&str>,
    mtp_draft_tokens: usize,
    chat: bool,
    system_message: Option<&str>,
    no_thinking: bool,
    kv_quant: Option<u8>,
    no_kv_quant: bool,
    detect_repetition: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let result = run_infer(
        py,
        InferOptions::new(
            model_id,
            prompt,
            lora,
            max_tokens,
            temperature,
            seed,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            fp8,
            experts_dir,
            draft_model,
            mtp,
            mtp_model,
            mtp_draft_tokens,
            chat,
            system_message,
            no_thinking,
            kv_quant,
            no_kv_quant,
            detect_repetition,
        ),
    )?;

    infer_result_to_dict(py, result)
}

/// Train a Qwen3Next/Qwen3.6 MTP predictor or Gemma 4 assistant.
#[pyfunction]
#[pyo3(signature = (
    model,
    shards,
    output="./mtp-output",
    family="auto",
    init_mtp=None,
    seq_len=512,
    batch_size=1,
    steps=1000,
    learning_rate=2e-4,
    min_lr=1e-5,
    warmup_steps=100,
    lr_schedule="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    checkpoint_every=500,
    log_every=10,
    eos_token_id=0,
    mtp_layers=1,
    assistant_layers=2,
    num_assistant_tokens=6,
    seed=42,
))]
#[allow(clippy::too_many_arguments)]
pub fn train_mtp<'py>(
    py: Python<'py>,
    model: &str,
    shards: Vec<String>,
    output: &str,
    family: &str,
    init_mtp: Option<&str>,
    seq_len: usize,
    batch_size: usize,
    steps: usize,
    learning_rate: f32,
    min_lr: f32,
    warmup_steps: usize,
    lr_schedule: &str,
    weight_decay: f32,
    max_grad_norm: f32,
    checkpoint_every: usize,
    log_every: usize,
    eos_token_id: u32,
    mtp_layers: usize,
    assistant_layers: usize,
    num_assistant_tokens: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let family = parse_mtp_family(family)?;
    let options = MtpTrainOptions {
        model: model.to_string(),
        shards,
        output: output.to_string(),
        family,
        init_mtp: init_mtp.map(String::from),
        seq_len,
        batch_size,
        steps,
        learning_rate,
        min_lr,
        warmup_steps,
        lr_schedule: lr_schedule.to_string(),
        weight_decay,
        max_grad_norm,
        checkpoint_every,
        log_every,
        eos_token_id,
        mtp_layers,
        assistant_layers,
        num_assistant_tokens,
        seed,
    };

    let result = py
        .detach(move || run_train_mtp_blocking(options))
        .map_err(runtime_err)?;
    mtp_result_to_dict(py, result)
}

/// Train a DFlash draft model for a Qwen3 target.
#[pyfunction]
#[pyo3(signature = (
    target,
    shards,
    output="./draft-output",
    draft=None,
    draft_config=None,
    seq_len=512,
    batch_size=1,
    steps=1000,
    learning_rate=2e-4,
    min_lr=1e-5,
    warmup_steps=100,
    lr_schedule="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    checkpoint_every=500,
    log_every=10,
    eos_token_id=0,
    seed=42,
))]
#[allow(clippy::too_many_arguments)]
pub fn train_draft<'py>(
    py: Python<'py>,
    target: &str,
    shards: Vec<String>,
    output: &str,
    draft: Option<&str>,
    draft_config: Option<&str>,
    seq_len: usize,
    batch_size: usize,
    steps: usize,
    learning_rate: f32,
    min_lr: f32,
    warmup_steps: usize,
    lr_schedule: &str,
    weight_decay: f32,
    max_grad_norm: f32,
    checkpoint_every: usize,
    log_every: usize,
    eos_token_id: u32,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    if draft.is_none() && draft_config.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "train_draft requires draft or draft_config",
        ));
    }
    if draft.is_some() && draft_config.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "train_draft accepts draft or draft_config, not both",
        ));
    }

    let options = DraftTrainOptions {
        target: target.to_string(),
        shards,
        output: output.to_string(),
        draft: draft.map(String::from),
        draft_config: draft_config.map(String::from),
        seq_len,
        batch_size,
        steps,
        learning_rate,
        min_lr,
        warmup_steps,
        lr_schedule: lr_schedule.to_string(),
        weight_decay,
        max_grad_norm,
        checkpoint_every,
        log_every,
        eos_token_id,
        seed,
    };

    let result = py
        .detach(move || run_train_draft_blocking(options))
        .map_err(runtime_err)?;
    mtp_result_to_dict(py, result)
}

impl InferOptions {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_id: &str,
        prompt: &str,
        lora: Option<&str>,
        max_tokens: usize,
        temperature: f32,
        seed: Option<u64>,
        top_k: Option<usize>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        fp8: bool,
        experts_dir: Option<&str>,
        draft_model: Option<&str>,
        mtp: bool,
        mtp_model: Option<&str>,
        mtp_draft_tokens: usize,
        chat: bool,
        system_message: Option<&str>,
        no_thinking: bool,
        kv_quant: Option<u8>,
        no_kv_quant: bool,
        detect_repetition: bool,
    ) -> Self {
        Self {
            model_id: model_id.to_string(),
            prompt: prompt.to_string(),
            lora: lora.map(String::from),
            max_tokens,
            temperature,
            seed,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            fp8,
            experts_dir: experts_dir.map(String::from),
            draft_model: draft_model.map(String::from),
            mtp,
            mtp_model: mtp_model.map(String::from),
            mtp_draft_tokens,
            chat,
            system_message: system_message.map(String::from),
            no_thinking,
            kv_quant,
            no_kv_quant,
            detect_repetition,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MtpFamily {
    Auto,
    Qwen3Next,
    Gemma4,
}

struct MtpTrainOptions {
    model: String,
    shards: Vec<String>,
    output: String,
    family: MtpFamily,
    init_mtp: Option<String>,
    seq_len: usize,
    batch_size: usize,
    steps: usize,
    learning_rate: f32,
    min_lr: f32,
    warmup_steps: usize,
    lr_schedule: String,
    weight_decay: f32,
    max_grad_norm: f32,
    checkpoint_every: usize,
    log_every: usize,
    eos_token_id: u32,
    mtp_layers: usize,
    assistant_layers: usize,
    num_assistant_tokens: usize,
    seed: u64,
}

struct DraftTrainOptions {
    target: String,
    shards: Vec<String>,
    output: String,
    draft: Option<String>,
    draft_config: Option<String>,
    seq_len: usize,
    batch_size: usize,
    steps: usize,
    learning_rate: f32,
    min_lr: f32,
    warmup_steps: usize,
    lr_schedule: String,
    weight_decay: f32,
    max_grad_norm: f32,
    checkpoint_every: usize,
    log_every: usize,
    eos_token_id: u32,
    seed: u64,
}

fn run_infer(py: Python<'_>, opts: InferOptions) -> PyResult<InferRunResult> {
    py.detach(move || {
        crate::hub::shared_runtime().block_on(async move {
            let model_path = pmetal_hub::resolve_model_path(&opts.model_id, None, None)
                .await
                .map_err(|e| e.to_string())?;
            let mtp_assistant_path = if let Some(draft) = opts.draft_model {
                Some(
                    pmetal_hub::resolve_model_path(&draft, None, None)
                        .await
                        .map_err(|e| e.to_string())?,
                )
            } else {
                None
            };
            let qwen_mtp_path = if let Some(mtp_model) = opts.mtp_model {
                Some(
                    pmetal_hub::resolve_model_path(&mtp_model, None, None)
                        .await
                        .map_err(|e| e.to_string())?,
                )
            } else {
                None
            };

            let config = pmetal_lib::inference_runner::InferenceRunnerConfig {
                model_path,
                lora_path: opts.lora,
                mtp_assistant_path,
                qwen_mtp_path,
                qwen_mtp: opts.mtp,
                qwen_mtp_draft_tokens: opts.mtp_draft_tokens.max(1),
                experts_dir: opts.experts_dir,
                fp8: opts.fp8,
                prompt: opts.prompt,
                chat_messages: None,
                system_message: opts.system_message,
                chat: opts.chat,
                no_thinking: opts.no_thinking,
                tools: None,
                temperature: Some(opts.temperature),
                top_k: opts.top_k,
                top_p: opts.top_p,
                min_p: opts.min_p,
                max_tokens: opts.max_tokens,
                repetition_penalty: opts.repetition_penalty,
                frequency_penalty: opts.frequency_penalty,
                presence_penalty: opts.presence_penalty,
                seed: opts.seed,
                kv_quant: opts.kv_quant,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: None,
                kv_quant_preset: None,
                no_kv_quant: opts.no_kv_quant,
                mode: pmetal_data::inference_config::SamplingMode::Auto,
                detect_repetition: opts.detect_repetition,
                kv_qjl: false,
            };

            let mut runner = pmetal_lib::inference_runner::InferenceRunner::prepare(config)
                .map_err(|e| e.to_string())?;
            let mut generated = Vec::new();
            let output = runner
                .state
                .generate_streaming(|token| {
                    generated.push(token);
                    true
                })
                .map_err(|e| e.to_string())?;
            let text = runner
                .tokenizer
                .decode(&generated)
                .map_err(|e| e.to_string())?;

            Ok::<_, String>(InferRunResult {
                text,
                num_generated: output.num_generated,
                stopped_by_token: output.stopped_by_token,
                stopped_by_length: output.stopped_by_length,
                speculative_metrics: output.speculative_metrics,
            })
        })
    })
    .map_err(runtime_err)
}

fn infer_result_to_dict<'py>(
    py: Python<'py>,
    result: InferRunResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("text", result.text)?;
    dict.set_item("num_generated", result.num_generated)?;
    dict.set_item("stopped_by_token", result.stopped_by_token)?;
    dict.set_item("stopped_by_length", result.stopped_by_length)?;
    if let Some(metrics) = result.speculative_metrics {
        let speculative = PyDict::new(py);
        speculative.set_item("drafted_tokens", metrics.drafted_tokens)?;
        speculative.set_item("accepted_draft_tokens", metrics.accepted_draft_tokens)?;
        speculative.set_item("emitted_draft_tokens", metrics.emitted_draft_tokens)?;
        speculative.set_item("bonus_tokens", metrics.bonus_tokens)?;
        speculative.set_item("correction_tokens", metrics.correction_tokens)?;
        speculative.set_item("verify_steps", metrics.verify_steps)?;
        speculative.set_item("full_accept_steps", metrics.full_accept_steps)?;
        speculative.set_item("partial_accept_steps", metrics.partial_accept_steps)?;
        speculative.set_item("zero_accept_steps", metrics.zero_accept_steps)?;
        speculative.set_item("max_draft_tokens", metrics.max_draft_tokens)?;
        speculative.set_item("acceptance_rate", metrics.acceptance_rate())?;
        speculative.set_item("avg_accepted_per_verify", metrics.avg_accepted_per_verify())?;
        dict.set_item("speculative", speculative)?;
    } else {
        dict.set_item("speculative", py.None())?;
    }
    Ok(dict)
}

fn run_train_mtp_blocking(opts: MtpTrainOptions) -> Result<MtpTrainingResult, String> {
    crate::hub::shared_runtime().block_on(async move {
        let config = build_mtp_config(
            opts.steps,
            opts.learning_rate,
            opts.min_lr,
            opts.warmup_steps,
            &opts.lr_schedule,
            opts.weight_decay,
            opts.max_grad_norm,
            opts.checkpoint_every,
            opts.output.clone(),
            opts.log_every,
            opts.num_assistant_tokens,
        );
        let batches = shard_batches(
            &opts.shards,
            opts.seq_len,
            opts.batch_size,
            opts.eos_token_id,
        )?;
        let target_path = pmetal_hub::resolve_model_path(&opts.model, None, None)
            .await
            .map_err(|e| format!("resolve target model {}: {e}", opts.model))?;
        let arch = pmetal_models::ModelArchitecture::detect(&target_path)
            .map_err(|e| format!("detect target architecture: {e}"))?;
        let family = resolve_mtp_family(opts.family, arch)?;

        match family {
            MtpFamily::Qwen3Next => {
                let mut target = match DynamicModel::load(&target_path)
                    .map_err(|e| format!("load target model: {e}"))?
                {
                    DynamicModel::Qwen3Next(model) => model,
                    other => {
                        return Err(format!(
                            "train_mtp family='qwen3-next' requires Qwen3Next target, got {:?}",
                            other.architecture()
                        ));
                    }
                };
                pmetal_bridge::compat::random::seed(opts.seed);
                let mut mtp = if let Some(init) = opts.init_mtp.as_deref() {
                    let init_path = pmetal_hub::resolve_model_path(init, None, None)
                        .await
                        .map_err(|e| format!("resolve init MTP {init}: {e}"))?;
                    load_qwen3_next_mtp_from_dir(&init_path, &target.config)
                        .map_err(|e| format!("load init Qwen MTP: {e}"))?
                } else {
                    let mtp_config = make_qwen3_next_mtp_config(&target.config, opts.mtp_layers);
                    let mut mtp = Qwen3NextMtpForCausalLM::new(mtp_config)
                        .map_err(|e| format!("create Qwen MTP model: {e}"))?;
                    initialize_qwen3_next_mtp_from_target(&target, &mut mtp);
                    mtp
                };
                run_qwen3_next_mtp_training(&mut target, &mut mtp, &config, batches)
                    .map_err(|e| format!("train Qwen MTP: {e}"))
            }
            MtpFamily::Gemma4 => {
                let mut target = match DynamicModel::load(&target_path)
                    .map_err(|e| format!("load target model: {e}"))?
                {
                    DynamicModel::Gemma4(model) => model,
                    other => {
                        return Err(format!(
                            "train_mtp family='gemma4' requires Gemma4 target, got {:?}",
                            other.architecture()
                        ));
                    }
                };
                pmetal_bridge::compat::random::seed(opts.seed);
                let mut assistant = if let Some(init) = opts.init_mtp.as_deref() {
                    let init_path = pmetal_hub::resolve_model_path(init, None, None)
                        .await
                        .map_err(|e| format!("resolve init assistant {init}: {e}"))?;
                    let (assistant, _) = load_gemma4_assistant_from_dir(&init_path)
                        .map_err(|e| format!("load init Gemma assistant: {e}"))?;
                    assistant
                } else {
                    let assistant_config =
                        make_gemma4_assistant_config(&target.config, opts.assistant_layers)
                            .map_err(|e| format!("create Gemma assistant config: {e}"))?;
                    Gemma4AssistantForCausalLM::new(assistant_config)
                        .map_err(|e| format!("create Gemma assistant: {e}"))?
                };
                run_gemma4_assistant_mtp_training(&mut target, &mut assistant, &config, batches)
                    .map_err(|e| format!("train Gemma assistant MTP: {e}"))
            }
            MtpFamily::Auto => unreachable!("auto is resolved before dispatch"),
        }
    })
}

fn run_train_draft_blocking(opts: DraftTrainOptions) -> Result<MtpTrainingResult, String> {
    crate::hub::shared_runtime().block_on(async move {
        let config = build_mtp_config(
            opts.steps,
            opts.learning_rate,
            opts.min_lr,
            opts.warmup_steps,
            &opts.lr_schedule,
            opts.weight_decay,
            opts.max_grad_norm,
            opts.checkpoint_every,
            opts.output,
            opts.log_every,
            1,
        );
        let batches = shard_batches(&opts.shards, opts.seq_len, opts.batch_size, opts.eos_token_id)?;
        let target_path = pmetal_hub::resolve_model_path(&opts.target, None, None)
            .await
            .map_err(|e| format!("resolve target model {}: {e}", opts.target))?;
        let mut target = match DynamicModel::load(&target_path)
            .map_err(|e| format!("load target model: {e}"))?
        {
            DynamicModel::Qwen3(model) => model,
            other => {
                return Err(format!(
                    "train_draft requires a Qwen3 target for the current DFlash objective, got {:?}",
                    other.architecture()
                ));
            }
        };

        let mut draft = if let Some(draft) = opts.draft.as_deref() {
            let draft_path = pmetal_hub::resolve_model_path(draft, None, None)
                .await
                .map_err(|e| format!("resolve draft checkpoint {draft}: {e}"))?;
            let (draft, report) = load_dflash_draft_from_dir(&draft_path)
                .map_err(|e| format!("load DFlash draft: {e}"))?;
            if !report.skipped.is_empty() {
                eprintln!(
                    "DFlash draft loaded with {} unused tensor key(s)",
                    report.skipped.len()
                );
            }
            draft
        } else {
            let config_path = opts
                .draft_config
                .as_deref()
                .ok_or_else(|| "train_draft requires draft or draft_config".to_string())?;
            let config_bytes = std::fs::read(config_path)
                .map_err(|e| format!("read DFlash config {config_path}: {e}"))?;
            let config: DFlashDraftConfig = serde_json::from_slice(&config_bytes)
                .map_err(|e| format!("parse DFlash config {config_path}: {e}"))?;
            pmetal_bridge::compat::random::seed(opts.seed);
            DFlashDraftModel::new(config).map_err(|e| format!("create DFlash draft: {e}"))?
        };

        run_dflash_draft_training(&mut target, &mut draft, &config, batches)
            .map_err(|e| format!("train DFlash draft: {e}"))
    })
}

#[allow(clippy::too_many_arguments)]
fn build_mtp_config(
    steps: usize,
    learning_rate: f32,
    min_lr: f32,
    warmup_steps: usize,
    lr_schedule: &str,
    weight_decay: f32,
    max_grad_norm: f32,
    checkpoint_every: usize,
    output: String,
    log_every: usize,
    num_assistant_tokens: usize,
) -> MtpTrainingConfig {
    MtpTrainingConfig {
        num_steps: steps,
        learning_rate,
        min_lr,
        warmup_steps,
        lr_schedule: parse_lr_schedule(lr_schedule),
        weight_decay,
        max_grad_norm: if max_grad_norm > 0.0 {
            Some(max_grad_norm)
        } else {
            None
        },
        checkpoint_every: if checkpoint_every > 0 {
            Some(checkpoint_every)
        } else {
            None
        },
        checkpoint_dir: PathBuf::from(output),
        log_every,
        num_assistant_tokens: num_assistant_tokens.max(1),
        ..Default::default()
    }
}

fn parse_lr_schedule(value: &str) -> LrSchedulerType {
    match value.to_ascii_lowercase().as_str() {
        "constant" => LrSchedulerType::Constant,
        "linear" => LrSchedulerType::Linear,
        _ => LrSchedulerType::Cosine,
    }
}

fn shard_batches(
    shards: &[String],
    seq_len: usize,
    batch_size: usize,
    eos_token_id: u32,
) -> Result<impl Iterator<Item = pmetal_bridge::compat::Array>, String> {
    if shards.is_empty() {
        return Err("shards must not be empty".to_string());
    }
    if seq_len < 2 {
        return Err("seq_len must be >= 2".to_string());
    }
    let reader = StreamingShardReader::new(StreamConfig {
        shard_paths: shards
            .iter()
            .map(Path::new)
            .map(Path::to_path_buf)
            .collect(),
        seq_len,
        batch_size: batch_size.max(1),
        eos_token_id,
        resume_from: None,
    })
    .map_err(|e| format!("open tokenized shards: {e}"))?;
    Ok(reader.map(|(tokens, _)| {
        batch_from_tokens(&tokens).expect("streaming shard reader must yield rectangular batches")
    }))
}

fn mtp_result_to_dict<'py>(
    py: Python<'py>,
    result: MtpTrainingResult,
) -> PyResult<Bound<'py, PyDict>> {
    let total_steps = result.losses.len();
    let dict = PyDict::new(py);
    dict.set_item("final_loss", result.final_loss)?;
    dict.set_item("losses", result.losses)?;
    dict.set_item("total_steps", total_steps)?;
    dict.set_item(
        "output_dir",
        result.output_dir.to_string_lossy().to_string(),
    )?;
    Ok(dict)
}

fn parse_mtp_family(value: &str) -> PyResult<MtpFamily> {
    match value.to_ascii_lowercase().replace(['_', '.'], "-").as_str() {
        "auto" => Ok(MtpFamily::Auto),
        "qwen3-next" | "qwen3next" | "qwen3-5" | "qwen35" | "qwen3-6" | "qwen36" => {
            Ok(MtpFamily::Qwen3Next)
        }
        "gemma4" | "gemma-4" => Ok(MtpFamily::Gemma4),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown MTP family {other:?}; expected 'auto', 'qwen3-next', or 'gemma4'"
        ))),
    }
}

fn resolve_mtp_family(
    requested: MtpFamily,
    arch: pmetal_models::ModelArchitecture,
) -> Result<MtpFamily, String> {
    match requested {
        MtpFamily::Auto => match arch {
            pmetal_models::ModelArchitecture::Qwen3Next => Ok(MtpFamily::Qwen3Next),
            pmetal_models::ModelArchitecture::Gemma4 => Ok(MtpFamily::Gemma4),
            other => Err(format!(
                "train_mtp family='auto' supports Qwen3Next/Qwen3.6 and Gemma4; got {other:?}"
            )),
        },
        MtpFamily::Qwen3Next => Ok(MtpFamily::Qwen3Next),
        MtpFamily::Gemma4 => Ok(MtpFamily::Gemma4),
    }
}
