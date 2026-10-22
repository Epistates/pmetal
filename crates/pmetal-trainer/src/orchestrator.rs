//! Shared training orchestrator: the one pipeline behind `pmetal train`, the
//! TUI, the GUI, the MCP server and the Python bindings.
//!
//! [`run_training`] is the single entry point. It handles:
//! - Model/dataset resolution (HF download or local path)
//! - Tokenization with chat template detection
//! - ANE training path (with GPU fallback)
//! - QLoRA and standard LoRA paths
//! - All dispatch modes: packed, compiled, metal-fused, standard
//! - Adaptive LR, checkpointing, metrics callbacks

use std::path::{Path, PathBuf};

/// The sections of a training config file, loaded and checked by `pmetal-core`.
pub use pmetal_core::FullTrainingConfig;
use pmetal_core::{ConfigValidator, LoraConfig, TrainingCallback, TrainingConfig};
use pmetal_data::{
    DataLoaderConfig, DatasetColumnConfig, DatasetFormat, DatasetSource, Tokenizer,
    TrainingDataset, resolve_dataset_source,
};
use pmetal_lora::{DynamicLoraModel, QLoraConfig, QLoraScheme, TrainableModel};
use pmetal_models::WeightFormat;
use pmetal_runtime::compat::ModuleParametersExt;

use crate::full_finetune::{self, FullFineTuneConfig};
use crate::{
    AdaptiveLrConfig, CheckpointManager, MetricsJsonCallback, TrainingLoop, TrainingLoopConfig,
};

#[cfg(feature = "distributed")]
type DistributedSyncOption = Option<crate::distributed_bridge::DistributedGradientSync>;
#[cfg(not(feature = "distributed"))]
type DistributedSyncOption = Option<()>;

#[cfg(feature = "distributed")]
fn take_distributed_sync(sync: &mut DistributedSyncOption) -> DistributedSyncOption {
    sync.take()
}

#[cfg(not(feature = "distributed"))]
fn take_distributed_sync(_: &mut DistributedSyncOption) -> DistributedSyncOption {
    None
}

#[cfg(feature = "distributed")]
fn attach_distributed_sync(training_loop: &mut TrainingLoop, sync: DistributedSyncOption) {
    if let Some(sync) = sync {
        training_loop.set_distributed(sync);
    }
}

#[cfg(not(feature = "distributed"))]
fn attach_distributed_sync(_: &mut TrainingLoop, _: DistributedSyncOption) {}

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Quantization scheme for QLoRA.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QuantizationScheme {
    #[default]
    None,
    Nf4,
    Fp4,
    Int8,
}

/// QLoRA-specific configuration.
#[derive(Debug, Clone)]
pub struct QLoraOrchConfig {
    pub scheme: QuantizationScheme,
    pub block_size: usize,
    pub double_quant: bool,
}

impl Default for QLoraOrchConfig {
    fn default() -> Self {
        Self {
            scheme: QuantizationScheme::None,
            block_size: 64,
            double_quant: false,
        }
    }
}

/// Dispatch / optimization flags.
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    pub sequence_packing: bool,
    /// Explicit override for the packing sequence length.
    ///
    /// When `None` (default), packing auto-computes the length from the p99 of
    /// sample lengths, rounded up to the next power of two.
    /// When `Some(n)`, `n` is used directly, bypassing the adaptive heuristic.
    pub pack_max_seq_len: Option<usize>,
    pub gradient_checkpointing: bool,
    pub gradient_checkpointing_layers: usize,
    pub cut_cross_entropy: bool,
    pub ane: bool,
    /// Loss scaling factor for ANE training. Multiplies dlogits before backward
    /// to prevent fp32 gradient underflow at >350M params. Default: 1.0.
    pub loss_scale: f32,
    /// Disable automatic adaptive LR (spike/plateau/divergence detection).
    /// Control file polling stays active for manual LR control via MCP/TUI.
    pub no_adaptive_lr: bool,
    #[cfg(feature = "distributed")]
    pub distributed: Option<pmetal_core::DistributedTrainingConfig>,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            sequence_packing: true,
            pack_max_seq_len: None,
            gradient_checkpointing: true,
            gradient_checkpointing_layers: 4,
            cut_cross_entropy: false,
            ane: false,
            loss_scale: 1.0,
            no_adaptive_lr: false,
            #[cfg(feature = "distributed")]
            distributed: None,
        }
    }
}

/// Complete training job configuration — replaces 38 positional parameters.
#[derive(Debug, Clone)]
pub struct TrainingJobConfig {
    /// Model identifier (HF repo ID or local path).
    pub model_id: String,
    /// Dataset path or HF dataset ID.
    pub dataset: String,
    /// Optional evaluation dataset.
    pub eval_dataset: Option<String>,
    /// Output directory for weights and checkpoints.
    pub output_dir: String,
    /// LoRA configuration.
    pub lora: LoraConfig,
    /// Optional QLoRA quantization.
    pub qlora: Option<QLoraOrchConfig>,
    /// Train every weight of the model instead of LoRA adapters: a
    /// full-parameter fine-tune, which saves a whole model (see
    /// [`crate::full_finetune`]).
    pub full: Option<FullFineTuneConfig>,
    /// Core training hyperparameters.
    pub training: TrainingConfig,
    /// Optional dataset column overrides.
    pub columns: Option<DatasetColumnConfig>,
    /// Dispatch / optimization flags.
    pub dispatch: DispatchConfig,
    /// Optional YAML config file to load defaults from.
    pub config_path: Option<String>,
    /// Path for JSONL metrics output.
    pub log_metrics: Option<String>,
    /// Resume from latest checkpoint.
    pub resume: bool,
    /// Random seed.
    pub seed: u64,
    /// Whether to print the final summary to stdout.
    pub emit_console_output: bool,
}

impl Default for TrainingJobConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            dataset: String::new(),
            eval_dataset: None,
            output_dir: "./output".to_string(),
            lora: LoraConfig::default(),
            qlora: None,
            full: None,
            training: TrainingConfig::default(),
            columns: None,
            dispatch: DispatchConfig::default(),
            config_path: None,
            log_metrics: None,
            resume: false,
            seed: 42,
            emit_console_output: true,
        }
    }
}

impl std::str::FromStr for QuantizationScheme {
    type Err = String;

    /// The names `--quantization` and a spec's `quantization` field use.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "nf4" => Ok(Self::Nf4),
            "fp4" => Ok(Self::Fp4),
            "int8" => Ok(Self::Int8),
            other => Err(format!(
                "unknown quantization '{other}'; valid: none, nf4, fp4, int8"
            )),
        }
    }
}

impl TrainingJobConfig {
    /// The job a [`TrainSpec`](pmetal_core::jobs::TrainSpec) asks for.
    ///
    /// Every surface that runs a train spec (the command line, the GUI, the
    /// TUI) builds its job here, so a field the spec carries cannot reach the
    /// trainer from one surface and be dropped by another. The caller adjusts
    /// what is its own: the resolved dataset path, where metrics go, whether
    /// the run prints a summary.
    ///
    /// # Errors
    ///
    /// A name the spec holds that the trainer does not know (an optimizer, a
    /// schedule, a quantization scheme, a compression strategy), and a request
    /// for distributed training from a build without it.
    pub fn from_spec(spec: &pmetal_core::jobs::TrainSpec) -> Result<Self, String> {
        let scheme: QuantizationScheme = spec
            .quantization
            .as_deref()
            .map_or(Ok(QuantizationScheme::None), str::parse)?;
        let qlora = (scheme != QuantizationScheme::None).then_some(QLoraOrchConfig {
            scheme,
            block_size: spec.quant_block_size,
            double_quant: spec.double_quant,
        });
        #[cfg(not(feature = "distributed"))]
        if spec.distributed_auto || spec.distributed_peers.is_some() {
            return Err("this build has no distributed training support".to_string());
        }
        Ok(Self {
            model_id: spec.model.clone(),
            dataset: spec.dataset.clone(),
            eval_dataset: spec.eval_dataset.clone(),
            output_dir: spec.output_dir.clone(),
            lora: spec.lora_config(),
            qlora,
            full: spec.full_finetune.then_some(FullFineTuneConfig {
                freeze_embeddings: spec.freeze_embeddings,
                skip_memory_check: spec.no_memory_check,
            }),
            training: spec.training_config()?,
            columns: DatasetColumnConfig::from_selectors(
                spec.text_column.clone(),
                spec.text_columns.as_deref(),
                spec.column_separator.clone(),
                spec.prompt_column.clone(),
                spec.response_column.clone(),
            ),
            dispatch: DispatchConfig {
                sequence_packing: !spec.no_sequence_packing,
                pack_max_seq_len: spec.pack_max_seq_len,
                gradient_checkpointing: !spec.no_gradient_checkpointing,
                cut_cross_entropy: spec.cut_cross_entropy,
                ane: spec.ane,
                loss_scale: spec.loss_scale,
                no_adaptive_lr: spec.no_adaptive_lr,
                #[cfg(feature = "distributed")]
                distributed: spec.distributed_config()?,
                ..DispatchConfig::default()
            },
            config_path: spec.config_path.clone(),
            log_metrics: spec.log_metrics.clone(),
            resume: spec.resume,
            seed: spec.seed,
            emit_console_output: true,
        })
    }
}

fn cli_training_defaults() -> TrainingConfig {
    TrainingConfig {
        num_epochs: 1,
        max_seq_len: 0,
        warmup_steps: 0,
        ..TrainingConfig::default()
    }
}

/// Copy into `target` every field of `overrides` that differs from `defaults`.
///
/// A field `overrides` leaves at its default is the caller saying nothing
/// about it, so the file's value stands; a field it sets, even to the value
/// the file has, wins. The fields are found by serializing, so a field added
/// to the config is merged without anyone remembering to add a line here (the
/// hand-copied version of this silently dropped every field added after it
/// was written).
fn merge_overrides<T>(target: &mut T, overrides: &T, defaults: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let to_object = |value: &T| match serde_json::to_value(value) {
        Ok(serde_json::Value::Object(map)) => map,
        other => panic!("a config section serializes to an object, got {other:?}"),
    };
    let (mut merged, overrides, defaults) =
        (to_object(target), to_object(overrides), to_object(defaults));
    for (key, value) in overrides {
        if defaults.get(&key) != Some(&value) {
            merged.insert(key, value);
        }
    }
    *target = serde_json::from_value(serde_json::Value::Object(merged))
        .expect("a merge of two valid config sections is a valid config section");
}

fn merge_lora_overrides(target: &mut LoraConfig, overrides: &LoraConfig) {
    merge_overrides(target, overrides, &LoraConfig::default());
}

fn merge_training_overrides(target: &mut TrainingConfig, overrides: &TrainingConfig) {
    merge_overrides(target, overrides, &cli_training_defaults());
}

/// The training loop's settings for a run: the config's training and LoRA
/// sections, and the dispatch choices.
fn build_training_loop_config(
    full_config: &FullTrainingConfig,
    dispatch: &DispatchConfig,
    dataloader: &DataLoaderConfig,
    has_eval_dataset: bool,
) -> TrainingLoopConfig {
    TrainingLoopConfig {
        training: full_config.training.clone(),
        dataloader: dataloader.clone(),
        log_every: full_config.training.logging_steps.max(1),
        checkpoint_every: full_config.training.save_steps.unwrap_or(500),
        eval_every: if has_eval_dataset {
            full_config.training.eval_steps.unwrap_or(100)
        } else {
            0
        },
        use_sequence_packing: dispatch.sequence_packing,
        pack_max_seq_len: dispatch.pack_max_seq_len,
        gradient_checkpointing: dispatch.gradient_checkpointing,
        gradient_checkpointing_layers: dispatch.gradient_checkpointing_layers,
        embedding_lr: full_config
            .training
            .embedding_learning_rate
            .map(|v| v as f32),
        loraplus_lr_ratio: full_config.lora.loraplus_lr_ratio,
        use_cut_cross_entropy: dispatch.cut_cross_entropy,
        #[cfg(feature = "distributed")]
        distributed: dispatch.distributed.clone(),
    }
}

fn merge_job_overrides_into_full_config(full: &mut FullTrainingConfig, job: &TrainingJobConfig) {
    if !job.model_id.is_empty() {
        full.model.model_id = job.model_id.clone();
    }
    if !job.dataset.is_empty() {
        full.dataset.dataset_id = job.dataset.clone();
    }
    merge_lora_overrides(&mut full.lora, &job.lora);
    merge_training_overrides(&mut full.training, &job.training);
}

// ---------------------------------------------------------------------------
// Phase reporting
// ---------------------------------------------------------------------------

/// Training pipeline phase for status reporting.
#[derive(Debug, Clone)]
pub enum TrainingPhase {
    ResolvingModel,
    ResolvingDataset,
    LoadingTokenizer,
    TokenizingDataset,
    LoadingModel,
    CompilingAneKernels,
    /// ANE was attempted but fell back to GPU. Contains the reason.
    AneFallback(String),
    Training,
    SavingWeights,
    Complete,
    Failed(String),
}

impl TrainingPhase {
    /// Human-readable status string.
    pub fn message(&self) -> &str {
        match self {
            Self::ResolvingModel => "Resolving model…",
            Self::ResolvingDataset => "Resolving dataset…",
            Self::LoadingTokenizer => "Loading tokenizer…",
            Self::TokenizingDataset => "Tokenizing dataset…",
            Self::LoadingModel => "Loading model…",
            Self::CompilingAneKernels => "Compiling ANE kernels…",
            Self::AneFallback(_) => "ANE unavailable, using GPU…",
            Self::Training => "Training…",
            Self::SavingWeights => "Saving weights…",
            Self::Complete => "Complete",
            Self::Failed(_) => "Failed",
        }
    }
}

/// Callback trait for receiving phase updates.
pub trait PhaseCallback: Send + Sync {
    fn on_phase(&self, phase: &TrainingPhase);
}

/// Blanket impl for closures.
impl<F: Fn(&TrainingPhase) + Send + Sync> PhaseCallback for F {
    fn on_phase(&self, phase: &TrainingPhase) {
        self(phase);
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of a completed training run.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub final_loss: f64,
    pub total_steps: usize,
    pub total_tokens: usize,
    pub output_dir: PathBuf,
    /// The trained weights: a LoRA adapter, or for full-parameter ANE
    /// training the model's own weights (`model.safetensors`).
    pub lora_weights_path: PathBuf,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Emit a phase update if a callback is set.
fn emit_phase(phase_cb: Option<&dyn PhaseCallback>, p: TrainingPhase) {
    if let Some(cb) = phase_cb {
        cb.on_phase(&p);
    }
}

/// Run a complete training pipeline.
///
/// All async work (model/dataset download) completes before any runtime
/// arrays are created, avoiding `!Send` issues with `Rc`-based array handles.
pub async fn run_training(
    mut config: TrainingJobConfig,
    phase_cb: Option<&dyn PhaseCallback>,
    callbacks: Vec<Box<dyn TrainingCallback>>,
) -> anyhow::Result<TrainingResult> {
    // Log MLX's default memory configuration for diagnostics.
    // We do NOT override MLX's defaults — it auto-configures based on
    // the Metal device's recommendedMaxWorkingSetSize.
    pmetal_runtime::memory::log_memory_stats();

    // -----------------------------------------------------------------------
    // Phase 1: Load or create YAML configuration and merge with job config
    // -----------------------------------------------------------------------
    let has_config_file = config.config_path.is_some();
    let mut full_config = if let Some(ref path) = config.config_path {
        let loaded = FullTrainingConfig::from_yaml_file(path)?;
        for note in &loaded.notes {
            tracing::warn!("{path}: {note}");
        }
        loaded.config
    } else {
        FullTrainingConfig::default()
    };

    if has_config_file {
        merge_job_overrides_into_full_config(&mut full_config, &config);
    } else {
        full_config.model.model_id = config.model_id.clone();
        full_config.dataset.dataset_id = config.dataset.clone();
        full_config.lora = config.lora.clone();
        full_config.training = config.training.clone();
    }

    // The merged config is what runs: a file that was valid alone can still
    // contradict the command line (a warmup ratio against `--warmup-steps`).
    full_config.validate()?;

    // Validate required fields after the YAML file has had a chance to provide them.
    if full_config.model.model_id.is_empty() {
        anyhow::bail!("A model is required. Specify --model <model_id> or set model_id in config.");
    }
    if full_config.dataset.dataset_id.is_empty() {
        anyhow::bail!("A dataset is required. Specify --dataset <path> or set dataset in config.");
    }

    let output_candidate =
        if has_config_file && config.output_dir == TrainingJobConfig::default().output_dir {
            full_config.training.output_dir.clone()
        } else {
            config.output_dir.clone()
        };
    let validated_output = pmetal_core::paths::output_path(&output_candidate, "output directory")?;
    let output_dir = validated_output.to_string_lossy().to_string();
    full_config.training.output_dir = output_dir.clone();

    if !has_config_file || config.seed != TrainingJobConfig::default().seed {
        full_config.training.seed = config.seed;
    }

    config.model_id = full_config.model.model_id.clone();
    config.dataset = full_config.dataset.dataset_id.clone();
    config.output_dir = output_dir.clone();
    config.lora = full_config.lora.clone();
    config.training = full_config.training.clone();
    config.seed = full_config.training.seed;

    // MLX draws LoRA's A matrices (and dropout masks) from its own random key,
    // separate from the data pipeline's seed. Seed it before anything builds
    // the model, or runs with the same seed start from different weights.
    pmetal_runtime::compat::random::seed(config.seed);

    if has_config_file {
        let dispatch_defaults = DispatchConfig::default();
        if config.dispatch.sequence_packing == dispatch_defaults.sequence_packing {
            config.dispatch.sequence_packing = full_config.training.use_packing;
        }
    }

    // -----------------------------------------------------------------------
    // Phase 2: Resolve model (async — may download from HF)
    // -----------------------------------------------------------------------
    emit_phase(phase_cb, TrainingPhase::ResolvingModel);
    tokio::task::yield_now().await;
    tracing::info!("Loading model: {}", full_config.model.model_id);
    let model_path = resolve_model_path(&full_config.model.model_id).await?;

    // -----------------------------------------------------------------------
    // Phase 3: Load model config
    // -----------------------------------------------------------------------
    // Parse config.json as a generic JSON value for architecture-agnostic field extraction.
    // Previous versions parsed as LlamaConfig which crashed for non-Llama models.
    let model_config_path = model_path.join("config.json");
    let model_config_json: Option<serde_json::Value> = if model_config_path.exists() {
        let content = std::fs::read_to_string(&model_config_path)?;
        Some(serde_json::from_str(&content)?)
    } else {
        if WeightFormat::detect(&model_path) != Some(WeightFormat::Gguf) {
            anyhow::bail!(
                "Model config.json not found at {:?}. If using GGUF, pass the .gguf file directly.",
                model_config_path
            );
        }
        None
    };

    if config.full.is_some() {
        if config.qlora.is_some() {
            anyhow::bail!(
                "--full trains the weights themselves, which a quantized base cannot take; \
                 drop --quantization (QLoRA is the quantized way to fine-tune)"
            );
        }
        if config.dispatch.ane {
            anyhow::bail!(
                "--ane already trains every weight of the model on the Neural Engine; use \
                 either --ane or --full"
            );
        }
        if WeightFormat::detect(&model_path) == Some(WeightFormat::Gguf) {
            anyhow::bail!(
                "--full fine-tunes a safetensors checkpoint and saves one; a GGUF file is \
                 quantized. Point at the safetensors checkpoint."
            );
        }
    }

    // Auto-detect max_seq_len if requested (0)
    if full_config.training.max_seq_len == 0 {
        if let Some(ref cfg) = model_config_json {
            // Try common field names across architectures
            let model_max = cfg
                .get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(8192) as usize;
            full_config.training.max_seq_len = model_max.min(8192);
            tracing::info!(
                "Auto-detected max_seq_len: {} (model supports {}, capped at 8192)",
                full_config.training.max_seq_len,
                model_max
            );
        } else {
            full_config.training.max_seq_len = 8192;
            tracing::info!(
                "Defaulting max_seq_len to {} (GGUF or unknown config)",
                full_config.training.max_seq_len
            );
        }
    }

    if let Some(ref cfg) = model_config_json {
        let hidden = cfg.get("hidden_size").and_then(|v| v.as_i64()).unwrap_or(0);
        let layers = cfg
            .get("num_hidden_layers")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let heads = cfg
            .get("num_attention_heads")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        tracing::info!(
            "Model: {} hidden, {} layers, {} heads",
            hidden,
            layers,
            heads
        );
    } else {
        tracing::info!("Model config will be extracted from GGUF metadata");
    }

    // -----------------------------------------------------------------------
    // Phase 5: Load tokenizer + detect chat template
    // -----------------------------------------------------------------------
    emit_phase(phase_cb, TrainingPhase::LoadingTokenizer);
    tokio::task::yield_now().await;
    tracing::info!("Loading tokenizer...");
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Tokenizer::from_model_dir(&model_path)?
    } else {
        anyhow::bail!(
            "Tokenizer not found at {:?}. GGUF models don't bundle a tokenizer — \
             download the source model first with: pmetal download {}",
            tokenizer_path,
            full_config.model.model_id
        );
    };

    let chat_template =
        pmetal_data::chat_templates::detect_chat_template(&model_path, &full_config.model.model_id);

    // -----------------------------------------------------------------------
    // Phase 6: Resolve + tokenize dataset (async for HF download, then sync)
    // -----------------------------------------------------------------------
    emit_phase(phase_cb, TrainingPhase::ResolvingDataset);
    tokio::task::yield_now().await;
    let dataset_path_resolved = resolve_dataset_path(&full_config.dataset.dataset_id).await?;

    emit_phase(phase_cb, TrainingPhase::TokenizingDataset);
    tokio::task::yield_now().await;
    tracing::info!(
        "Loading training dataset: {}",
        dataset_path_resolved.display()
    );
    let train_dataset = load_dataset(
        &dataset_path_resolved,
        &tokenizer,
        full_config.training.max_seq_len,
        &chat_template,
        config.columns.as_ref(),
    )?;
    tracing::info!("Training dataset loaded: {} samples", train_dataset.len());

    // Log sequence-length statistics
    {
        let stats = train_dataset.compute_statistics(full_config.training.max_seq_len);
        tracing::info!(
            "Dataset: {} samples | lengths: mean={:.0}, median={}, p95={}, p99={}, max={} | truncated: {:.1}%",
            stats.total_samples,
            stats.mean_length,
            stats.median_length,
            stats.p95_length,
            stats.p99_length,
            stats.max_length,
            stats.truncated_pct,
        );
        let warnings = train_dataset.validate_seq_len(full_config.training.max_seq_len);
        for w in &warnings {
            tracing::warn!("{}", w);
        }
    }

    // Load evaluation dataset if provided
    let eval_dataset = if let Some(ref eval_path) = config.eval_dataset {
        tracing::info!("Loading evaluation dataset: {}", eval_path);
        let ds = TrainingDataset::from_jsonl_tokenized(
            eval_path,
            &tokenizer,
            DatasetFormat::Auto,
            full_config.training.max_seq_len,
            Some(&chat_template),
            config.columns.as_ref(),
        )?;
        tracing::info!("Evaluation dataset loaded: {} samples", ds.len());
        Some(ds)
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Phase 7: ANE full-parameter training (if requested and compatible)
    //
    // ANE training uses full model weights (not LoRA adapters). When ANE is
    // requested and the model is compatible (dense transformer, no MoE), we
    // attempt the ANE path first. On failure, fall back to GPU LoRA training.
    // -----------------------------------------------------------------------
    let mut extra_callbacks = Some(callbacks);

    #[cfg(feature = "ane")]
    if config.dispatch.ane {
        use crate::DynamicAneTrainerConfig;

        // Check ANE compatibility from config.json
        let config_text = std::fs::read_to_string(model_path.join("config.json"))?;
        let config_json: serde_json::Value = serde_json::from_str(&config_text)?;

        match DynamicAneTrainerConfig::is_ane_trainable(&config_json) {
            Ok(()) => {
                tracing::info!("Model is ANE-compatible — attempting ANE full-parameter training");
                emit_phase(phase_cb, TrainingPhase::CompilingAneKernels);

                match attempt_ane_training(
                    &config,
                    &full_config,
                    &model_path,
                    &output_dir,
                    phase_cb,
                    &mut extra_callbacks,
                    &train_dataset,
                )
                .await
                {
                    Ok(result) => {
                        // ANE training succeeded — save and return
                        emit_phase(phase_cb, TrainingPhase::SavingWeights);

                        tracing::info!(
                            loss = format!("{:.4}", result.final_loss),
                            steps = result.total_steps,
                            tokens = result.total_tokens,
                            "ANE training complete"
                        );

                        emit_phase(phase_cb, TrainingPhase::Complete);
                        return Ok(result);
                    }
                    Err(e) => {
                        // ANE failed — fall back to GPU LoRA
                        let reason = format!("{e}");
                        tracing::warn!("ANE training failed, falling back to GPU: {reason}");
                        emit_phase(phase_cb, TrainingPhase::AneFallback(reason));
                    }
                }
            }
            Err(reason) => {
                tracing::info!("Model not ANE-compatible ({reason}), using GPU training");
                emit_phase(phase_cb, TrainingPhase::AneFallback(reason));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 7: Resolve metrics path (absolute, using canonicalized output_dir)
    // Actual MetricsJsonCallback creation is deferred to AFTER ANE attempt
    // to avoid truncating the metrics file if ANE writes to it first.
    // -----------------------------------------------------------------------
    let metrics_path_resolved = config.log_metrics.as_ref().map(|metrics_path| {
        // If the path is just a filename (e.g. "metrics.jsonl"), join it to the
        // canonicalized output_dir so all consumers agree on the absolute path.
        let p = PathBuf::from(metrics_path);
        if p.is_absolute() {
            p
        } else if metrics_path.contains('/') || metrics_path.contains('\\') {
            // Relative path with directories — resolve against output_dir
            PathBuf::from(&output_dir).join(p)
        } else {
            PathBuf::from(&output_dir).join(metrics_path)
        }
    });
    let has_metrics_cb = metrics_path_resolved.is_some();
    let mut metrics_callback: Option<Box<dyn TrainingCallback>> = if let Some(ref path) =
        metrics_path_resolved
    {
        // Create the metrics callback NOW (after ANE attempt).
        // This truncates the file — safe because ANE either succeeded
        // (and we returned early) or failed (and its partial metrics are stale).
        let mut callback = MetricsJsonCallback::new(path)?
                .with_run_name(format!(
                    "{}-{}",
                    full_config.model.model_id.replace('/', "-"),
                    chrono::Utc::now().format("%Y%m%d-%H%M%S")
                ))
                .with_config(serde_json::json!({
                    "model": full_config.model.model_id,
                    "lora_r": config.lora.r,
                    "learning_rate": full_config.training.learning_rate,
                    "batch_size": full_config.training.batch_size,
                    "epochs": full_config.training.num_epochs,
                    "max_seq_len": full_config.training.max_seq_len,
                    "gradient_accumulation_steps": full_config.training.gradient_accumulation_steps,
                    "gradient_checkpointing": config.dispatch.gradient_checkpointing,
                    "quantization": format!("{:?}", config.qlora.as_ref().map(|q| q.scheme).unwrap_or_default()),
                }));
        callback.on_train_start();
        Some(Box::new(callback) as Box<dyn TrainingCallback>)
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Phase 8: Set up training infrastructure
    // -----------------------------------------------------------------------
    // A full fine-tune's checkpoint is the whole model and its optimizer's
    // state, so only the latest, the one a resume reads, is kept.
    let checkpoint_dir = PathBuf::from(&output_dir).join("checkpoints");
    let checkpoint_manager = CheckpointManager::new(&checkpoint_dir)?
        .with_max_checkpoints(if config.full.is_some() { 1 } else { 3 });

    if let Some(full) = &config.full {
        full_finetune::check_memory(
            full,
            &model_path,
            &full_finetune::RunShape {
                optimizer: full_config.training.optimizer,
                batch_size: full_config.training.batch_size,
                seq_len: full_config.training.max_seq_len,
                gradient_checkpointing: config.dispatch.gradient_checkpointing,
                cut_cross_entropy: config.dispatch.cut_cross_entropy,
            },
        )?;
    }

    let dataloader_config = DataLoaderConfig {
        batch_size: full_config.training.batch_size,
        max_seq_len: full_config.training.max_seq_len,
        shuffle: full_config.dataset.shuffle,
        seed: config.seed,
        pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
        drop_last: false,
        ..Default::default()
    };

    #[cfg(feature = "distributed")]
    let mut distributed_sync: DistributedSyncOption = if let Some(ref distributed_config) =
        config.dispatch.distributed
    {
        let ctx = crate::distributed_bridge::create_distributed_context(distributed_config)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initialize distributed training: {e}"))?;
        tracing::info!(
            "Distributed training initialized: rank={}/{}",
            ctx.rank(),
            ctx.world_size()
        );
        let sync = crate::distributed_bridge::DistributedGradientSync::new(ctx, distributed_config);
        Some(sync)
    } else {
        // A process MLX launched into a distributed group trains data-parallel
        // over MLX's collectives, with no peers to name.
        let sync = crate::distributed_bridge::DistributedGradientSync::from_mlx_group(
            &pmetal_core::DistributedTrainingConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to join the MLX distributed group: {e}"))?;
        if let Some(sync) = &sync {
            tracing::info!(
                "Distributed training over MLX collectives: rank={}/{}",
                sync.rank(),
                sync.world_size()
            );
        }
        sync
    };

    #[cfg(not(feature = "distributed"))]
    let mut distributed_sync: DistributedSyncOption = None;

    let training_loop_config = build_training_loop_config(
        &full_config,
        &config.dispatch,
        &dataloader_config,
        eval_dataset.is_some(),
    );

    // -----------------------------------------------------------------------
    // Phase 9: Load model + run training
    // -----------------------------------------------------------------------
    emit_phase(phase_cb, TrainingPhase::LoadingModel);
    tokio::task::yield_now().await;
    // NOTE: After this point, MLX types (Rc-based) will be created.
    // No more yields — the future becomes !Send.

    // Catch panics from MLX/Metal so all frontends (CLI, TUI, GUI) get a
    // proper error instead of a silent process crash.
    let training_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_lora_path(
            &config,
            &full_config,
            &model_path,
            training_loop_config,
            train_dataset,
            eval_dataset,
            &checkpoint_manager,
            &mut metrics_callback,
            &mut extra_callbacks,
            &output_dir,
            phase_cb,
            has_metrics_cb,
            take_distributed_sync(&mut distributed_sync),
        )
    }));

    // Drop the training result's captured state (model, optimizer, etc.)
    // BEFORE clearing the cache. The catch_unwind closure captures references
    // to the model — those Arrays must be dropped first so their Metal buffers
    // move from "active" to "cache", then clear_cache() returns them to the OS.
    let training_outcome = match training_result {
        Ok(result) => result,
        Err(panic_payload) => {
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                format!("Training crashed: {s}")
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                format!("Training crashed: {s}")
            } else {
                "Training crashed (internal panic)".to_string()
            };
            // Model + optimizer already dropped (catch_unwind closure ended).
            pmetal_runtime::inline_array::clear_cache();
            pmetal_runtime::memory::log_memory_stats();
            anyhow::bail!(msg);
        }
    };

    // For success/error paths, clear cache after extracting the result
    // (model dropped when catch_unwind closure returned).
    pmetal_runtime::inline_array::clear_cache();
    pmetal_runtime::memory::log_memory_stats();

    let (final_loss, final_step, total_tokens) = training_outcome?;

    // -----------------------------------------------------------------------
    // Phase 10: Finalize
    // -----------------------------------------------------------------------
    emit_phase(phase_cb, TrainingPhase::SavingWeights);

    // Finalize metrics callback
    if let Some(ref mut callback) = metrics_callback {
        let mut custom = std::collections::HashMap::new();
        custom.insert("total_tokens".to_string(), total_tokens as f64);
        custom.insert("total_steps".to_string(), final_step as f64);
        callback.on_epoch_end(
            full_config.training.num_epochs.saturating_sub(1),
            &pmetal_core::EvalMetrics {
                loss: final_loss,
                perplexity: final_loss.exp(),
                accuracy: None,
                custom,
            },
        );
        callback.on_train_end();
        if let Some(ref path) = metrics_path_resolved {
            tracing::info!("Metrics saved to {:?}", path);
        }
    }

    if config.emit_console_output {
        println!("\n========================================");
        println!("  Training Complete!");
        println!("========================================");
        println!("Final Loss:   {:.4}", final_loss);
        println!("Total Steps:  {}", final_step);
        println!("Total Tokens: {}", total_tokens);
        println!("Output:       {}", output_dir);
        println!("========================================");

        println!("\nNext steps:");
        if config.full.is_some() {
            println!("  Inference:  pmetal infer -m {output_dir} -p \"Your prompt\"");
            println!("  Serve:      pmetal serve -m {output_dir}");
        } else {
            println!(
                "  Inference:  pmetal infer -m {} --lora {}/{} -p \"Your prompt\"",
                full_config.model.model_id,
                output_dir,
                pmetal_lora::peft::ADAPTER_WEIGHTS
            );
            println!(
                "  Quantize:   pmetal quantize -m {} --lora {}/{} -o model.gguf",
                full_config.model.model_id,
                output_dir,
                pmetal_lora::peft::ADAPTER_WEIGHTS
            );
        }
    }

    // The trained weights: an adapter, or for a full fine-tune the model itself
    // (a single file, or the index of its shards).
    let lora_weights_path = if config.full.is_some() {
        let single = PathBuf::from(&output_dir).join("model.safetensors");
        if single.exists() {
            single
        } else {
            PathBuf::from(&output_dir).join("model.safetensors.index.json")
        }
    } else {
        PathBuf::from(&output_dir).join(pmetal_lora::peft::ADAPTER_WEIGHTS)
    };

    emit_phase(phase_cb, TrainingPhase::Complete);

    Ok(TrainingResult {
        final_loss,
        total_steps: final_step,
        total_tokens,
        output_dir: PathBuf::from(&output_dir),
        lora_weights_path,
    })
}

// ---------------------------------------------------------------------------
// LoRA training path, with the base packed for QLoRA when asked
// ---------------------------------------------------------------------------

/// The base-packing config `--quantization` asks for, if any.
fn qlora_config(config: &TrainingJobConfig) -> Option<QLoraConfig> {
    let qlora = config.qlora.as_ref()?;
    let scheme = match qlora.scheme {
        QuantizationScheme::None => return None,
        QuantizationScheme::Nf4 => QLoraScheme::Nf4,
        QuantizationScheme::Fp4 => QLoraScheme::Fp4,
        QuantizationScheme::Int8 => QLoraScheme::Int8,
    };
    Some(QLoraConfig {
        scheme,
        group_size: qlora.block_size as i32,
        double_quant: qlora.double_quant,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_lora_path(
    config: &TrainingJobConfig,
    full_config: &FullTrainingConfig,
    model_path: &Path,
    mut training_loop_config: TrainingLoopConfig,
    train_dataset: TrainingDataset,
    eval_dataset: Option<TrainingDataset>,
    checkpoint_manager: &CheckpointManager,
    metrics_callback: &mut Option<Box<dyn TrainingCallback>>,
    extra_callbacks: &mut Option<Vec<Box<dyn TrainingCallback>>>,
    output_dir: &str,
    phase_cb: Option<&dyn PhaseCallback>,
    has_metrics_cb: bool,
    distributed_sync: DistributedSyncOption,
) -> anyhow::Result<(f64, usize, usize)> {
    let full = config.full.as_ref();
    tracing::info!(
        "Initializing {} model with auto-detected architecture...",
        if full.is_some() {
            "full fine-tune"
        } else {
            "LoRA"
        }
    );

    let qlora = qlora_config(config);
    let mut model = match (WeightFormat::detect(model_path), qlora) {
        _ if full.is_some() => DynamicLoraModel::from_pretrained_full(
            model_path,
            pmetal_lora::FullFineTune {
                freeze_embeddings: full.is_some_and(|f| f.freeze_embeddings),
            },
        )?,
        (Some(WeightFormat::Gguf), Some(_)) => anyhow::bail!(
            "--quantization packs a safetensors checkpoint for QLoRA; a GGUF file is \
             already quantized in its own format. Train LoRA on it without --quantization, \
             or point at the safetensors checkpoint."
        ),
        (Some(WeightFormat::Gguf), None) => {
            tracing::info!("Detected GGUF format, loading with dequantization...");
            DynamicLoraModel::from_gguf(model_path, config.lora.clone())?
        }
        (_, Some(qlora)) => {
            if qlora.scheme == QLoraScheme::Fp4 && qlora.group_size != 64 {
                tracing::warn!(
                    "--quant-block-size {} is ignored for fp4: NVFP4's block is 16",
                    qlora.group_size
                );
            }
            let (model, packed) = DynamicLoraModel::from_pretrained_quantized(
                model_path,
                config.lora.clone(),
                &qlora,
            )?;
            let mb = |bytes: usize| bytes as f64 / 1_000_000.0;
            tracing::info!(
                "QLoRA: {} projections packed as {}{}, {:.1} MB -> {:.1} MB",
                packed.packed,
                qlora.scheme,
                if qlora.double_quant {
                    " with double quantization"
                } else {
                    ""
                },
                mb(packed.dense_bytes),
                mb(packed.packed_bytes),
            );
            for (path, reason) in &packed.kept {
                tracing::debug!("QLoRA kept {path} as loaded: {reason}");
            }
            model
        }
        (_, None) => DynamicLoraModel::from_pretrained(model_path, config.lora.clone())?,
    };
    tracing::info!(
        "Loaded {} model {}",
        model.architecture_name(),
        if full.is_some() {
            "for a full fine-tune"
        } else {
            "with LoRA adapters"
        }
    );

    // A full fine-tune's result is a whole model written in the checkpoint's own
    // layout. Whether it can be is known now, not after the run.
    let save_plan = match full {
        Some(_) => Some(full_finetune::plan_for_model(
            model_path,
            &model.flatten_params(),
        )?),
        None => None,
    };

    // GDN resource guard
    if model.architecture_name() == "Qwen3Next" && full_config.training.max_seq_len > 512 {
        tracing::warn!(
            "Qwen3.5 (GDN) training with max_seq_len={} may exceed Metal resource limits (499K buffers). \
             Recommended: --max-seq-len 512 or lower.",
            full_config.training.max_seq_len
        );
    }

    tracing::info!(
        "Trainable parameters: {}",
        format_param_count(model.num_trainable_params())
    );
    pmetal_runtime::memory::log_memory_stats();

    if config.dispatch.gradient_checkpointing {
        if model.supports_gradient_checkpointing() {
            model.enable_gradient_checkpointing(config.dispatch.gradient_checkpointing_layers);
            tracing::info!(
                "Gradient checkpointing enabled ({} layers per block)",
                config.dispatch.gradient_checkpointing_layers
            );
        } else {
            tracing::warn!(
                "Gradient checkpointing requested but not supported by {} architecture.",
                model.architecture_name()
            );
        }
        // The model has already handled the capability check above. Do not let
        // the TrainingLoop re-apply or re-log this setting.
        training_loop_config.gradient_checkpointing = false;
    }

    let mut training_loop = TrainingLoop::new(training_loop_config)?;
    attach_distributed_sync(&mut training_loop, distributed_sync);

    if let Some(cb) = metrics_callback.take() {
        training_loop.add_callback(cb);
    }
    if let Some(callbacks) = extra_callbacks.take() {
        for callback in callbacks {
            training_loop.add_callback(callback);
        }
    }

    {
        let mut adaptive_config = AdaptiveLrConfig::for_lora();
        // Rolling back from a divergence means holding a copy of the best
        // weights, which for a full fine-tune is a second copy of the model.
        if config.dispatch.no_adaptive_lr || full.is_some() {
            adaptive_config.enabled = false;
        }
        let control_file = pmetal_core::lr_control_path(std::path::Path::new(output_dir));
        training_loop.enable_adaptive_lr_with_control(adaptive_config, control_file);
        if full.is_none() {
            training_loop
                .set_snapshot_persist_dir(checkpoint_manager.checkpoint_dir().to_path_buf());
        }
    }

    if config.resume && !training_loop.resume_from(&mut model, checkpoint_manager)? {
        tracing::info!("No checkpoint found, starting fresh");
    }

    tracing::info!(
        "Starting {} training...",
        if full.is_some() {
            "full fine-tune"
        } else {
            "LoRA"
        }
    );
    emit_phase(phase_cb, TrainingPhase::Training);

    // Notify all callbacks that training is starting (TrainingLoop doesn't do this)
    for cb in &mut training_loop.callbacks {
        cb.on_train_start();
    }

    training_loop.run(
        &mut model,
        train_dataset,
        eval_dataset,
        Some(checkpoint_manager),
    )?;

    match &save_plan {
        Some(plan) => {
            let weights = full_finetune::save_full_model(
                &model.flatten_params(),
                plan,
                model_path,
                Path::new(output_dir),
            )?;
            tracing::info!("Saved the fine-tuned model to {}", weights.display());
        }
        None => {
            save_adapter(
                &model,
                Path::new(output_dir),
                &config.lora,
                Some(&full_config.model.model_id),
            )?;
        }
    }

    // Recover the metrics callback (always inserted first) for finalization.
    // Only recover if we actually created one — don't grab a user callback by mistake.
    if has_metrics_cb && metrics_callback.is_none() {
        let mut cbs = training_loop.take_callbacks();
        if !cbs.is_empty() {
            *metrics_callback = Some(cbs.remove(0));
        }
    }

    Ok((
        training_loop.current_loss(),
        training_loop.current_step(),
        training_loop.total_tokens(),
    ))
}

// ---------------------------------------------------------------------------
// ANE training path
// ---------------------------------------------------------------------------

#[cfg(feature = "ane")]
async fn attempt_ane_training(
    config: &TrainingJobConfig,
    full_config: &FullTrainingConfig,
    model_path: &Path,
    output_dir: &str,
    phase_cb: Option<&dyn PhaseCallback>,
    extra_callbacks: &mut Option<Vec<Box<dyn TrainingCallback>>>,
    train_dataset: &TrainingDataset,
) -> anyhow::Result<TrainingResult> {
    use crate::{AneTrainingLoop, AneTrainingLoopConfig, DynamicAneTrainerConfig, VocabMap};

    tracing::info!("Attempting ANE dynamic weight pipeline");

    // -----------------------------------------------------------------------
    // Validate ANE compatibility from config.json
    // -----------------------------------------------------------------------
    let config_text = std::fs::read_to_string(model_path.join("config.json"))?;
    let config_json: serde_json::Value = serde_json::from_str(&config_text)?;

    // The model's shape, RMSNorm epsilon and RoPE base come from the engine's
    // own reading of config.json, which also refuses what it can't train.
    let model_config =
        DynamicAneTrainerConfig::from_model_config(&config_json).map_err(|e| anyhow::anyhow!(e))?;
    let vocab_size = model_config.vocab_size;

    // ANE projection kernels scale linearly with seq_len (IOSurface dimensions).
    // The O(seq²) attention is decomposed to CPU BLAS, so there's no hard cap,
    // but attention backward cost is quadratic — use a tight seq_len that fits
    // the actual data rather than the model's max_position_embeddings.
    //
    // Strategy: use dataset p95 length (rounded up to next power of 2, capped
    // at model max and 4096) to minimize padding waste while covering most samples.
    let model_max_seq = config_json
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(2048);

    let max_seq_len = {
        // Compute p95 from dataset sample lengths
        let mut lengths: Vec<usize> = train_dataset
            .samples()
            .iter()
            .map(|s| s.input_ids.len())
            .collect();
        lengths.sort_unstable();
        let p95_idx = (lengths.len() as f64 * 0.95) as usize;
        let p95 = lengths.get(p95_idx).copied().unwrap_or(512);

        // Round up to next power of 2 for IOSurface alignment
        let rounded = p95.next_power_of_two().max(64);

        // Apply caps: user-specified max_seq_len, model max, and 4096 upper bound
        let user_max = if full_config.training.max_seq_len > 0 {
            full_config.training.max_seq_len
        } else {
            4096
        };
        let capped = rounded.min(user_max).min(model_max_seq).min(4096);

        tracing::info!(
            dataset_p95 = p95,
            rounded = rounded,
            model_max = model_max_seq,
            selected = capped,
            "ANE seq_len: auto-selected from dataset p95"
        );
        capped
    };

    // -----------------------------------------------------------------------
    // Convert the already-tokenized dataset to ANE batch format.
    // The dataset was loaded and tokenized once in the main pipeline —
    // no duplicate resolution, download, or tokenization.
    // Step 1: collect all u32 token IDs for VocabMap construction.
    // Step 2: build VocabMap (maps full vocab → compact u16 range).
    // Step 3: remap tokens and build u16 batches.
    // This correctly handles vocab > 65536 (e.g. Qwen3 @ 151936).
    let gradient_accumulation_steps = full_config.training.gradient_accumulation_steps;
    let num_epochs = full_config.training.num_epochs;

    // Collect all u32 token IDs for VocabMap
    let mut all_token_ids: Vec<u32> = Vec::new();
    // Also collect the (input, target) pairs as u32 for later remapping, with
    // which targets count toward the loss.
    let mut pairs_u32: Vec<(Vec<u32>, Vec<u32>, Vec<bool>)> = Vec::new();
    // ANE kernels require all sequences to be EXACTLY max_seq_len tokens
    // (baked into IOSurface dimensions). Pad short sequences, truncate long ones.
    let pad_token = 0u32; // Pad token ID (will be compacted by VocabMap)
    for sample in train_dataset.samples() {
        let ids = &sample.input_ids;
        if ids.len() < 2 {
            continue;
        }
        // Build input/target of exactly max_seq_len
        let mut input = Vec::with_capacity(max_seq_len);
        let mut target = Vec::with_capacity(max_seq_len);
        let usable = ids.len().min(max_seq_len + 1);
        input.extend_from_slice(&ids[..usable - 1]);
        target.extend_from_slice(&ids[1..usable]);
        // Target t predicts ids[t + 1]. It counts unless the dataset masks
        // that token (label -100, a prompt token) or it's padding. Padding
        // used to count, so most of a short example's loss was predicting
        // the pad token after the pad token.
        let mut counted: Vec<bool> = (1..usable)
            .map(|i| {
                sample
                    .labels
                    .as_ref()
                    .is_none_or(|labels| labels.get(i).is_none_or(|&l| l != -100))
            })
            .collect();
        if !counted.contains(&true) {
            continue;
        }
        // Pad to exact seq_len
        while input.len() < max_seq_len {
            input.push(pad_token);
            target.push(pad_token);
            counted.push(false);
        }
        all_token_ids.extend_from_slice(&input);
        all_token_ids.extend_from_slice(&target);
        pairs_u32.push((input, target, counted));
    }

    if pairs_u32.is_empty() {
        anyhow::bail!("No training examples after tokenization");
    }

    // Build VocabMap from u32 IDs (correct for large vocabs)
    let vocab_map = VocabMap::from_token_ids(&all_token_ids, vocab_size);
    drop(all_token_ids); // free memory
    tracing::info!(
        compact_vocab = vocab_map.compact_vocab,
        full_vocab = vocab_size,
        "Vocab compaction ready"
    );

    // Remap to compact u16 and build batches
    let mut batches: Vec<Vec<(Vec<u16>, Vec<u16>)>> = Vec::new();
    let mut current_batch: Vec<(Vec<u16>, Vec<u16>)> = Vec::new();
    for (input_u32, target_u32, counted) in &pairs_u32 {
        let input = vocab_map.remap_u32(input_u32);
        let mut target = vocab_map.remap_u32(target_u32);
        for (t, &counts) in target.iter_mut().zip(counted) {
            if !counts {
                *t = crate::IGNORE_TARGET;
            }
        }
        current_batch.push((input, target));

        if current_batch.len() >= gradient_accumulation_steps.max(1) {
            batches.push(std::mem::take(&mut current_batch));
        }
    }
    if !current_batch.is_empty() {
        batches.push(current_batch);
    }
    drop(pairs_u32);

    let total_batches = batches.len() * num_epochs;
    tracing::info!(
        examples = train_dataset.len(),
        batches = batches.len(),
        epochs = num_epochs,
        total_steps = total_batches,
        "Prepared ANE training data"
    );

    // -----------------------------------------------------------------------
    // Build ANE trainer config — wire all applicable training hyperparams
    // -----------------------------------------------------------------------
    let trainer_config = DynamicAneTrainerConfig {
        seq_len: max_seq_len,
        // Full-parameter training wants about a tenth of LoRA's learning
        // rate: the usual recipe is 2e-5 for full fine-tuning against 2e-4
        // for LoRA, and published full SFT of models this size runs at 1e-5
        // to 2e-5. A LoRA rate here would diverge, so the rate is capped at
        // 2e-5 (the GUI's ANE default); a lower one is kept as given.
        learning_rate: {
            let user_lr = full_config.training.learning_rate as f32;
            let max_full_param_lr = 2e-5;
            if user_lr > max_full_param_lr {
                tracing::info!(
                    user_lr = format!("{:.1e}", user_lr),
                    capped = format!("{:.1e}", max_full_param_lr),
                    "LR capped for full-parameter ANE training (LoRA defaults are too high)"
                );
                max_full_param_lr
            } else {
                user_lr
            }
        },
        accum_steps: gradient_accumulation_steps.max(1),
        warmup_steps: full_config.training.warmup_steps.max(50),
        gradient_clip_norm: full_config.training.max_grad_norm as f32,
        loss_scale: config.dispatch.loss_scale,
        embedding_lr: full_config
            .training
            .embedding_learning_rate
            .map(|v| v as f32),
        weight_decay: full_config.training.weight_decay as f32,
        ..model_config
    };

    // Ensure output directory exists
    std::fs::create_dir_all(output_dir)?;

    // Log every step for ANE training — each step is expensive (seconds, not ms)
    // so per-step visibility is critical for monitoring.
    let log_every = 1;
    let save_every = full_config.training.save_steps;

    let loop_config = AneTrainingLoopConfig {
        trainer: trainer_config,
        num_batches: total_batches,
        max_steps: total_batches,
        log_every,
        save_every,
        output_dir: PathBuf::from(output_dir),
    };

    let mut training_loop = AneTrainingLoop::new(loop_config);

    // -----------------------------------------------------------------------
    // Load model weights
    // -----------------------------------------------------------------------
    emit_phase(phase_cb, TrainingPhase::LoadingModel);
    tokio::task::yield_now().await;
    tracing::info!("Loading model weights for ANE training...");
    training_loop.load_weights_safetensors(model_path)?;

    // Install the VocabMap built during batch construction (already logged above)
    training_loop.install_vocab_map(vocab_map);

    // -----------------------------------------------------------------------
    // Compile ANE kernels
    // -----------------------------------------------------------------------
    emit_phase(phase_cb, TrainingPhase::CompilingAneKernels);
    tokio::task::yield_now().await;
    tracing::info!("Compiling dynamic ANE kernels (one-time)...");
    training_loop.compile_kernels()?;

    // -----------------------------------------------------------------------
    // Wire callbacks: metrics JSONL + user-provided callbacks
    // NOTE: Don't call on_train_start() — AneTrainingLoop::train() calls it
    // on all callbacks internally.
    // -----------------------------------------------------------------------
    if let Some(ref metrics_str) = config.log_metrics {
        // Resolve to absolute path using canonicalized output_dir
        let p = PathBuf::from(metrics_str);
        let metrics_path = if p.is_absolute() {
            p
        } else {
            PathBuf::from(output_dir).join(
                p.file_name()
                    .unwrap_or(std::ffi::OsStr::new("metrics.jsonl")),
            )
        };
        if let Ok(cb) = crate::MetricsJsonCallback::new(&metrics_path) {
            let cb = cb
                .with_run_name(format!("ane-{}", config.model_id.replace('/', "-")))
                .with_config(serde_json::json!({
                    "model": config.model_id,
                    "backend": "ane",
                    "learning_rate": full_config.training.learning_rate,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "epochs": num_epochs,
                    "max_seq_len": max_seq_len,
                }));
            training_loop.add_callback(Box::new(cb));
        }
    }

    // Wire user-provided callbacks (cancel, GUI metrics) into the ANE loop.
    // These are consumed here — if ANE fails, the fallback path creates fresh
    // callbacks or the orchestrator returns early.
    if let Some(cbs) = extra_callbacks.take() {
        for cb in cbs {
            training_loop.add_callback(cb);
        }
    }

    // -----------------------------------------------------------------------
    // Train
    // -----------------------------------------------------------------------
    emit_phase(phase_cb, TrainingPhase::Training);
    tokio::task::yield_now().await;

    let mut final_loss = 0.0f64;
    let mut total_tokens_processed = 0usize;
    for epoch in 0..num_epochs {
        tracing::info!(epoch = epoch + 1, total = num_epochs, "Starting epoch");
        let state = training_loop.train(&batches)?;
        final_loss = state.loss;
        total_tokens_processed += state.tokens_processed;
        tracing::info!(
            loss = format!("{:.4}", state.loss),
            tokens = state.tokens_processed,
            tok_per_sec = format!("{:.1}", state.tokens_per_sec()),
            "Epoch complete"
        );
    }

    tracing::info!("ANE training complete");

    // The ANE trains every weight, so the result is a whole model: the
    // trained weights next to the base model's config and tokenizer. It used
    // to be thrown away.
    emit_phase(phase_cb, TrainingPhase::SavingWeights);
    let output = PathBuf::from(output_dir);
    let source_dir = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or(Path::new("."))
    };
    pmetal_merge::copy_side_files(source_dir, &output)?;
    pmetal_merge::patch_config_dtype(&output, safetensors::Dtype::F32)?;
    let weights_path = output.join("model.safetensors");
    training_loop.save_model(&weights_path)?;
    tracing::info!(path = %weights_path.display(), "Saved ANE-trained model");

    if config.emit_console_output {
        println!("\n========================================");
        println!("  Training Complete!");
        println!("========================================");
        println!("Final Loss:   {:.4}", final_loss);
        println!("Total Steps:  {}", total_batches);
        println!("Total Tokens: {}", total_tokens_processed);
        println!("Output:       {}", output_dir);
        println!("========================================");
        println!("\nNext steps:");
        println!(
            "  Inference:  pmetal infer -m {} -p \"Your prompt\"",
            output_dir
        );
    }

    Ok(TrainingResult {
        final_loss,
        total_steps: total_batches,
        total_tokens: total_tokens_processed,
        output_dir: output,
        lora_weights_path: weights_path,
    })
}

// ---------------------------------------------------------------------------
// Helper: dataset loading
// ---------------------------------------------------------------------------

fn load_dataset(
    path: &Path,
    tokenizer: &Tokenizer,
    max_seq_len: usize,
    chat_template: &pmetal_data::chat_templates::ChatTemplate,
    columns: Option<&DatasetColumnConfig>,
) -> anyhow::Result<TrainingDataset> {
    // One loader for every container (JSONL, JSON, Parquet, Arrow, CSV) and
    // for a directory of shards.
    Ok(TrainingDataset::from_jsonl_tokenized(
        path,
        tokenizer,
        DatasetFormat::Auto,
        max_seq_len,
        Some(chat_template),
        columns,
    )?)
}

// ---------------------------------------------------------------------------
// Helper: resolve model path (async, may download from HF)
// ---------------------------------------------------------------------------

pub(crate) async fn resolve_model_path(model_id: &str) -> anyhow::Result<PathBuf> {
    #[cfg(feature = "hub")]
    {
        pmetal_hub::resolve_model_path(model_id, None, None)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
    #[cfg(not(feature = "hub"))]
    {
        if model_id.contains('/') && !PathBuf::from(model_id).exists() {
            anyhow::bail!(
                "Model '{}' looks like a HuggingFace ID but the 'hub' feature is not enabled",
                model_id
            );
        }
        Ok(PathBuf::from(model_id))
    }
}

// ---------------------------------------------------------------------------
// Helper: resolve dataset path (async, may download from HF)
// ---------------------------------------------------------------------------

/// Resolve a dataset identifier to a local file path, downloading from
/// HuggingFace Hub if it looks like a dataset ID.
pub async fn resolve_dataset_path(dataset_id: &str) -> anyhow::Result<PathBuf> {
    match resolve_dataset_source(dataset_id) {
        DatasetSource::Local(p) => Ok(TrainingDataset::resolve_dataset_path_pub(&p)?),
        DatasetSource::HuggingFace(id) => {
            #[cfg(feature = "hub")]
            {
                tracing::info!("Downloading dataset from HuggingFace: {}", id);
                let dir = pmetal_hub::download_dataset(&id, None, None, None).await?;
                // The cached repository: its train split, whatever the
                // container (JSONL, JSON, Parquet, Arrow, CSV), every shard.
                if let Ok(p) = TrainingDataset::resolve_dataset_path_pub(&dir) {
                    return Ok(p);
                }

                // Last resort: download parquet from HF API (network call)
                let parquet_paths =
                    pmetal_hub::download_dataset_parquet(&id, "train", None, None).await?;
                match parquet_paths.as_slice() {
                    [] => anyhow::bail!("No JSONL or Parquet files found for dataset {}", id),
                    [only] => Ok(only.clone()),
                    [first, rest @ ..] => {
                        // Shards of one split: read together, from their directory.
                        match first.parent() {
                            Some(parent) if rest.iter().all(|p| p.parent() == Some(parent)) => {
                                Ok(parent.to_path_buf())
                            }
                            _ => Ok(first.clone()),
                        }
                    }
                }
            }
            #[cfg(not(feature = "hub"))]
            {
                anyhow::bail!(
                    "Dataset '{}' looks like a HuggingFace ID but the 'hub' feature is not enabled",
                    id
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: save adapter config
// ---------------------------------------------------------------------------

/// Write a trained adapter into `dir` as PEFT reads it: `adapter_model.safetensors`
/// and an `adapter_config.json` that names the base model it was trained on when
/// known and lists the modules that actually carry an adapter (an empty
/// `target_modules` in a training config means every linear layer, which a
/// PEFT config can only say by listing them). Returns the weights file.
pub fn save_adapter<M: pmetal_lora::TrainableModel>(
    model: &M,
    dir: &Path,
    lora: &LoraConfig,
    base_model: Option<&str>,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let weights = dir.join(pmetal_lora::peft::ADAPTER_WEIGHTS);
    model.save_lora_weights(&weights)?;
    pmetal_lora::peft::write_adapter_config(
        dir,
        lora,
        &model.adapter_target_modules(),
        base_model,
    )?;
    tracing::info!("Saved adapter to {:?}", dir);
    Ok(weights)
}

// ---------------------------------------------------------------------------
// Helper: format param count
// ---------------------------------------------------------------------------

pub fn format_param_count(count: usize) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1_000.0)
    } else {
        format!("{}", count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_loop_gets_the_lora_plus_ratio_and_embedding_rate_the_config_sets() {
        // `lora.loraplus_lr_ratio` used to be read from the file and then
        // handed to the training loop as `None`, so LoRA+ never ran.
        let mut full = FullTrainingConfig::default();
        full.lora.loraplus_lr_ratio = Some(16.0);
        full.training.embedding_learning_rate = Some(5e-5);
        let loop_config = build_training_loop_config(
            &full,
            &DispatchConfig::default(),
            &DataLoaderConfig::default(),
            false,
        );
        assert_eq!(loop_config.loraplus_lr_ratio, Some(16.0));
        assert_eq!(loop_config.embedding_lr, Some(5e-5));
        assert_eq!(loop_config.eval_every, 0, "no eval dataset, no evaluation");
    }

    #[test]
    fn the_adapter_config_names_its_base_model() {
        let dir = tempfile::tempdir().unwrap();
        let lora = LoraConfig {
            r: 8,
            alpha: 16.0,
            use_rslora: true,
            ..LoraConfig::default()
        };
        let targets = vec!["q_proj".to_string(), "v_proj".to_string()];
        pmetal_lora::peft::write_adapter_config(
            dir.path(),
            &lora,
            &targets,
            Some("Qwen/Qwen3-0.6B"),
        )
        .unwrap();
        let written: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("adapter_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(written["peft_type"], "LORA");
        assert_eq!(written["r"], 8);
        assert_eq!(written["lora_alpha"], 16.0);
        assert_eq!(written["use_rslora"], true);
        assert_eq!(
            written["target_modules"],
            serde_json::json!(["q_proj", "v_proj"])
        );
        assert_eq!(written["base_model_name_or_path"], "Qwen/Qwen3-0.6B");
    }

    /// Every field a train spec carries reaches the job the trainer runs.
    #[test]
    fn a_train_spec_becomes_the_job_it_asks_for() {
        use pmetal_core::jobs::TrainSpec;
        let spec = TrainSpec {
            model: "m".into(),
            dataset: "d".into(),
            eval_dataset: Some("e".into()),
            output_dir: "out".into(),
            max_steps: Some(9),
            warmup_ratio: Some(0.1),
            min_lr: Some(1e-6),
            save_steps: Some(3),
            eval_steps: Some(2),
            logging_steps: 4,
            lora_targets: vec!["q_proj".into()],
            quantization: Some("fp4".into()),
            quant_block_size: 16,
            text_columns: Some("a,b".into()),
            prompt_column: Some("p".into()),
            no_sequence_packing: true,
            cut_cross_entropy: true,
            loss_scale: 2.0,
            resume: true,
            log_metrics: Some("m.jsonl".into()),
            config_path: Some("c.yaml".into()),
            ..TrainSpec::default()
        };
        let job = TrainingJobConfig::from_spec(&spec).unwrap();
        assert_eq!((job.model_id.as_str(), job.dataset.as_str()), ("m", "d"));
        assert_eq!(job.eval_dataset.as_deref(), Some("e"));
        assert_eq!(job.output_dir, "out");
        assert_eq!(job.training.max_steps, Some(9));
        assert_eq!(job.training.warmup_ratio, Some(0.1));
        assert_eq!(job.training.min_lr, Some(1e-6));
        assert_eq!(job.training.save_steps, Some(3));
        assert_eq!(job.training.eval_steps, Some(2));
        assert_eq!(job.training.logging_steps, 4);
        assert_eq!(job.lora.target_modules, ["q_proj"]);
        let qlora = job.qlora.unwrap();
        assert_eq!(
            (qlora.scheme, qlora.block_size),
            (QuantizationScheme::Fp4, 16)
        );
        assert_eq!(job.columns.unwrap().text_columns.unwrap(), ["a", "b"]);
        assert!(!job.dispatch.sequence_packing && job.dispatch.cut_cross_entropy);
        assert_eq!(job.dispatch.loss_scale, 2.0);
        assert!(job.resume);
        assert_eq!(job.log_metrics.as_deref(), Some("m.jsonl"));
        assert_eq!(job.config_path.as_deref(), Some("c.yaml"));

        let plain = TrainingJobConfig::from_spec(&TrainSpec::default()).unwrap();
        assert!(plain.qlora.is_none() && plain.columns.is_none());
        let bad = TrainSpec {
            quantization: Some("nf8".into()),
            ..TrainSpec::default()
        };
        assert!(
            TrainingJobConfig::from_spec(&bad)
                .unwrap_err()
                .contains("nf8")
        );
    }

    #[test]
    fn config_file_merge_preserves_yaml_training_defaults() {
        let mut full = FullTrainingConfig::default();
        full.model.model_id = "yaml/model".to_string();
        full.dataset.dataset_id = "yaml.jsonl".to_string();
        full.training.num_epochs = 7;
        full.training.max_steps = Some(123);
        full.training.save_steps = Some(11);
        full.training.eval_steps = Some(13);
        full.training.min_lr = Some(1e-6);
        full.training.use_packing = false;

        let job = TrainingJobConfig {
            model_id: String::new(),
            dataset: String::new(),
            training: cli_training_defaults(),
            ..TrainingJobConfig::default()
        };

        merge_job_overrides_into_full_config(&mut full, &job);

        assert_eq!(full.model.model_id, "yaml/model");
        assert_eq!(full.dataset.dataset_id, "yaml.jsonl");
        assert_eq!(full.training.num_epochs, 7);
        assert_eq!(full.training.max_steps, Some(123));
        assert_eq!(full.training.save_steps, Some(11));
        assert_eq!(full.training.eval_steps, Some(13));
        assert_eq!(full.training.min_lr, Some(1e-6));
        assert!(!full.training.use_packing);
    }

    #[test]
    fn config_file_merge_applies_explicit_job_overrides() {
        let mut full = FullTrainingConfig::default();
        full.model.model_id = "yaml/model".to_string();
        full.dataset.dataset_id = "yaml.jsonl".to_string();
        full.training.num_epochs = 7;

        let mut overrides = cli_training_defaults();
        overrides.num_epochs = 2;
        overrides.max_steps = Some(5);
        let job = TrainingJobConfig {
            model_id: "cli/model".to_string(),
            dataset: "cli.jsonl".to_string(),
            lora: LoraConfig {
                r: 8,
                ..LoraConfig::default()
            },
            training: overrides,
            ..TrainingJobConfig::default()
        };

        merge_job_overrides_into_full_config(&mut full, &job);

        assert_eq!(full.model.model_id, "cli/model");
        assert_eq!(full.dataset.dataset_id, "cli.jsonl");
        assert_eq!(full.lora.r, 8);
        assert_eq!(full.training.num_epochs, 2);
        assert_eq!(full.training.max_steps, Some(5));
    }

    /// Every field a job sets reaches the merged config, the ones added since
    /// the merge was written included, and the ones it leaves alone keep the
    /// file's value.
    #[test]
    fn the_override_merge_carries_every_field_a_job_sets() {
        let mut full = FullTrainingConfig::default();
        full.training.weight_decay = 0.3;
        let overrides = TrainingConfig {
            label_smoothing: 0.1,
            z_loss: 1e-4,
            neftune_noise_alpha: Some(5.0),
            optimizer: pmetal_core::OptimizerType::Muon,
            ..cli_training_defaults()
        };
        let job = TrainingJobConfig {
            training: overrides,
            ..TrainingJobConfig::default()
        };
        merge_job_overrides_into_full_config(&mut full, &job);
        assert_eq!(full.training.label_smoothing, 0.1);
        assert_eq!(full.training.z_loss, 1e-4);
        assert_eq!(full.training.neftune_noise_alpha, Some(5.0));
        assert_eq!(full.training.optimizer, pmetal_core::OptimizerType::Muon);
        assert_eq!(
            full.training.weight_decay, 0.3,
            "untouched fields keep the file's value"
        );
    }
}
