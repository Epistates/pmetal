//! Shared training orchestrator consumed by CLI, TUI, GUI, and easy API.
//!
//! Replaces four separate training pipeline implementations with one canonical
//! `run_training()` entry point that handles:
//! - Model/dataset resolution (HF download or local path)
//! - Tokenization with chat template detection
//! - ANE training path (with GPU fallback)
//! - QLoRA and standard LoRA paths
//! - All dispatch modes: packed, compiled, metal-fused, standard
//! - Adaptive LR, checkpointing, metrics callbacks

use std::path::{Component, Path, PathBuf};

use pmetal_core::{DatasetConfig, LoraConfig, ModelConfig, TrainingCallback, TrainingConfig};
use pmetal_data::{
    DataLoaderConfig, DatasetColumnConfig, DatasetFormat, DatasetSource, Tokenizer,
    TrainingDataset, resolve_dataset_source,
};
use pmetal_lora::{DynamicLoraModel, DynamicQloraModel, QLoraConfig, TrainableModel};
use pmetal_mlx::quantization::QuantScheme;
use pmetal_models::WeightFormat;
// LlamaConfig import removed — config.json is now parsed as generic serde_json::Value
// to support all architectures (QLoRA uses DynamicQloraModel for arch-specific parsing).

use crate::{
    AdaptiveLrConfig, CheckpointManager, MetricsJsonCallback, TrainingLoop, TrainingLoopConfig,
};

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
    pub flash_attention: bool,
    pub sequence_packing: bool,
    /// Explicit override for the packing sequence length.
    ///
    /// When `None` (default), `run_packed` auto-computes the packing length from
    /// the p99 of sample lengths, rounded up to the next power of two.
    /// When `Some(n)`, `n` is used directly, bypassing the adaptive heuristic.
    pub pack_max_seq_len: Option<usize>,
    pub jit_compilation: bool,
    pub fused: bool,
    pub metal_fused_optimizer: bool,
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
            flash_attention: true,
            sequence_packing: true,
            pack_max_seq_len: None,
            jit_compilation: true,
            fused: true,
            metal_fused_optimizer: true,
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
    pub lora_weights_path: PathBuf,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run a complete training pipeline.
///
/// All async work (model/dataset download) completes before any MLX types
/// are created, avoiding `!Send` issues with `Rc`-based array handles.
/// Emit a phase update if a callback is set.
fn emit_phase(phase_cb: Option<&dyn PhaseCallback>, p: TrainingPhase) {
    if let Some(cb) = phase_cb {
        cb.on_phase(&p);
    }
}

pub async fn run_training(
    config: TrainingJobConfig,
    phase_cb: Option<&dyn PhaseCallback>,
    callbacks: Vec<Box<dyn TrainingCallback>>,
) -> anyhow::Result<TrainingResult> {
    // Log MLX's default memory configuration for diagnostics.
    // We do NOT override MLX's defaults — it auto-configures based on
    // the Metal device's recommendedMaxWorkingSetSize.
    pmetal_mlx::memory::log_memory_stats();

    // Validate required fields
    if config.model_id.is_empty() {
        anyhow::bail!("A model is required. Specify --model <model_id> or set model_id in config.");
    }
    if config.dataset.is_empty() {
        anyhow::bail!("A dataset is required. Specify --dataset <path> or set dataset in config.");
    }

    let use_qlora = config
        .qlora
        .as_ref()
        .is_some_and(|q| !matches!(q.scheme, QuantizationScheme::None));

    // Validate output directory
    let validated_output = validate_output_path(&config.output_dir, "output directory")?;
    let output_dir = validated_output.to_string_lossy().to_string();

    // -----------------------------------------------------------------------
    // Phase 1: Load or create YAML configuration and merge with job config
    // -----------------------------------------------------------------------
    let mut full_config = if let Some(ref path) = config.config_path {
        let content = std::fs::read_to_string(path)?;
        serde_yaml::from_str(&content)?
    } else {
        FullTrainingConfig::default()
    };

    // Override with job config values
    full_config.model.model_id = config.model_id.clone();
    full_config.dataset.dataset_id = config.dataset.clone();
    full_config.lora = config.lora.clone();
    full_config.training = config.training.clone();
    full_config.training.output_dir = output_dir.clone();
    full_config.training.seed = config.seed;

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
    let is_parquet = dataset_path_resolved
        .extension()
        .is_some_and(|ext| ext == "parquet");
    let train_dataset = load_dataset(
        &dataset_path_resolved,
        is_parquet,
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

        match DynamicAneTrainerConfig::is_ane_compatible(&config_json) {
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
            PathBuf::from(&output_dir).join(
                p.file_name()
                    .unwrap_or(std::ffi::OsStr::new("metrics.jsonl")),
            )
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
    let checkpoint_dir = PathBuf::from(&output_dir).join("checkpoints");
    let checkpoint_manager = CheckpointManager::new(&checkpoint_dir)?.with_max_checkpoints(3);

    let dataloader_config = DataLoaderConfig {
        batch_size: full_config.training.batch_size,
        max_seq_len: full_config.training.max_seq_len,
        shuffle: full_config.dataset.shuffle,
        seed: config.seed,
        pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
        drop_last: false,
        ..Default::default()
    };

    let training_loop_config = TrainingLoopConfig {
        training: full_config.training.clone(),
        dataloader: dataloader_config.clone(),
        use_metal_flash_attention: config.dispatch.flash_attention,
        log_every: full_config.training.logging_steps.max(1),
        checkpoint_every: full_config.training.save_steps.unwrap_or(500),
        eval_every: if eval_dataset.is_some() {
            full_config.training.eval_steps.unwrap_or(100)
        } else {
            0
        },
        use_jit_compilation: config.dispatch.jit_compilation,
        use_sequence_packing: config.dispatch.sequence_packing,
        pack_max_seq_len: config.dispatch.pack_max_seq_len,
        gradient_checkpointing: config.dispatch.gradient_checkpointing,
        gradient_checkpointing_layers: config.dispatch.gradient_checkpointing_layers,
        embedding_lr: full_config
            .training
            .embedding_learning_rate
            .map(|v| v as f32),
        eager_evaluation: true,
        use_metal_fused_optimizer: config.dispatch.metal_fused_optimizer,
        loraplus_lr_ratio: None,
        neftune_noise_alpha: None,
        use_cut_cross_entropy: config.dispatch.cut_cross_entropy,
        #[cfg(feature = "distributed")]
        distributed: config.dispatch.distributed.clone(),
    };

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
        if use_qlora {
            run_qlora_path(
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
            )
        } else {
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
            )
        }
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
            pmetal_mlx::memory::clear_cache();
            pmetal_mlx::memory::log_memory_stats();
            anyhow::bail!(msg);
        }
    };

    // For success/error paths, clear cache after extracting the result
    // (model dropped when catch_unwind closure returned).
    pmetal_mlx::memory::clear_cache();
    pmetal_mlx::memory::log_memory_stats();

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
        println!(
            "  Inference:  pmetal infer -m {} --lora {}/lora_weights.safetensors -p \"Your prompt\"",
            full_config.model.model_id, output_dir
        );
        println!(
            "  Quantize:   pmetal quantize -m {} --lora {}/lora_weights.safetensors -o model.gguf",
            full_config.model.model_id, output_dir
        );
    }

    let lora_weights_path = PathBuf::from(&output_dir).join("lora_weights.safetensors");

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
// QLoRA training path
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_qlora_path(
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
) -> anyhow::Result<(f64, usize, usize)> {
    let qlora_cfg = config.qlora.as_ref().unwrap();
    let quant_scheme = match qlora_cfg.scheme {
        QuantizationScheme::Nf4 => QuantScheme::NF4,
        QuantizationScheme::Fp4 => QuantScheme::FP4,
        QuantizationScheme::Int8 => QuantScheme::Int8,
        QuantizationScheme::None => unreachable!(),
    };

    let qlora_config = QLoraConfig {
        lora: config.lora.clone(),
        quant_scheme,
        block_size: qlora_cfg.block_size,
        double_quant: qlora_cfg.double_quant,
        compute_in_half: true,
    };

    tracing::info!(
        "Initializing QLoRA model with {:?} quantization...",
        qlora_cfg.scheme
    );

    // Detect architecture and construct the correct QLoRA model.
    // Errors early with a clear message for unsupported architectures.
    let mut model = DynamicQloraModel::from_model_dir(model_path, qlora_config)?;

    tracing::info!(
        "Loading and quantizing {} base model weights from {:?}...",
        model.arch_name(),
        model_path
    );
    model.load_and_quantize_from_dir(model_path)?;

    let savings = model.memory_savings();
    let (quant_bytes, lora_bytes, total_bytes) = model.memory_usage();
    tracing::info!(
        "Memory usage: {:.2} MB (quantized: {:.2} MB, LoRA: {:.2} MB) - {:.1}% of full precision",
        total_bytes as f64 / 1_000_000.0,
        quant_bytes as f64 / 1_000_000.0,
        lora_bytes as f64 / 1_000_000.0,
        savings * 100.0
    );
    tracing::info!(
        "Trainable parameters: {}",
        format_param_count(model.num_trainable_params())
    );
    pmetal_mlx::memory::log_memory_stats();

    if config.dispatch.gradient_checkpointing {
        if model.supports_gradient_checkpointing() {
            model.enable_gradient_checkpointing(config.dispatch.gradient_checkpointing_layers);
            tracing::info!(
                "Gradient checkpointing enabled ({} layers per block)",
                config.dispatch.gradient_checkpointing_layers
            );
        } else {
            tracing::warn!(
                "Gradient checkpointing requested but not supported by this QLoRA model ({}).",
                model.arch_name()
            );
        }
        // The model has already handled the capability check above. Do not let
        // the TrainingLoop re-apply or re-log this setting.
        training_loop_config.gradient_checkpointing = false;
    }

    #[cfg(feature = "distributed")]
    if config.dispatch.distributed.is_some() {
        tracing::warn!(
            "Distributed training is not yet supported through the orchestrator. \
             Falling back to single-device training."
        );
    }

    let mut training_loop = TrainingLoop::new(training_loop_config);

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
        if config.dispatch.no_adaptive_lr {
            adaptive_config.enabled = false;
        }
        let control_file = PathBuf::from(output_dir).join(".lr_control.json");
        training_loop.enable_adaptive_lr_with_control(adaptive_config, control_file);
        training_loop.set_snapshot_persist_dir(checkpoint_manager.checkpoint_dir().to_path_buf());
    }

    if config.resume {
        if let Some((lora_params, metadata)) = checkpoint_manager.load_latest()? {
            tracing::info!("Resuming from checkpoint at step {}", metadata.step);
            model.set_lora_parameters(&lora_params);
            training_loop.set_step(metadata.step);
            training_loop.set_epoch(metadata.epoch);
            // Restore persisted best-loss snapshot so rollback works after resume.
            if let Some(snapshot) =
                crate::checkpoint::load_best_snapshot(checkpoint_manager.checkpoint_dir())
            {
                training_loop.best_lora_snapshot = Some(snapshot);
                // Notify adaptive LR controller that a snapshot is available for rollback.
                if let Some(ref mut ctrl) = training_loop.adaptive_lr {
                    ctrl.set_has_snapshot(metadata.running_loss, metadata.step);
                }
                tracing::info!("Restored best snapshot from disk for rollback-on-resume");
            }
        } else {
            tracing::info!("No checkpoint found, starting fresh");
        }
    }

    tracing::info!("Starting QLoRA training...");
    pmetal_mlx::memory::log_memory_stats();
    emit_phase(phase_cb, TrainingPhase::Training);

    for cb in &mut training_loop.callbacks {
        cb.on_train_start();
    }

    if config.dispatch.fused {
        tracing::warn!("Fused training is not yet supported for QLoRA, using standard training");
    }

    tracing::info!("Entering training_loop.run() for QLoRA...");
    training_loop.run(
        &mut model,
        train_dataset,
        eval_dataset,
        Some(checkpoint_manager),
    )?;

    let final_path = PathBuf::from(output_dir).join("lora_weights.safetensors");
    model.save_lora_weights(&final_path)?;
    tracing::info!("Saved LoRA weights to {:?}", final_path);
    save_adapter_config(
        &final_path,
        config.lora.r,
        config.lora.alpha,
        &config.lora.target_modules,
        config.lora.use_rslora,
    )?;

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
// Standard LoRA training path
// ---------------------------------------------------------------------------

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
) -> anyhow::Result<(f64, usize, usize)> {
    tracing::info!("Initializing LoRA model with auto-detected architecture...");

    let mut model = match WeightFormat::detect(model_path) {
        Some(WeightFormat::Gguf) => {
            tracing::info!("Detected GGUF format, loading with dequantization...");
            DynamicLoraModel::from_gguf(model_path, config.lora.clone())?
        }
        _ => DynamicLoraModel::from_pretrained(model_path, config.lora.clone())?,
    };
    tracing::info!(
        "Loaded {} model with LoRA adapters",
        model.architecture_name()
    );

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
    pmetal_mlx::memory::log_memory_stats();

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

    #[cfg(feature = "distributed")]
    if config.dispatch.distributed.is_some() {
        tracing::warn!(
            "Distributed training is not yet supported through the orchestrator. \
             Falling back to single-device training."
        );
    }

    let mut training_loop = TrainingLoop::new(training_loop_config);

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
        if config.dispatch.no_adaptive_lr {
            adaptive_config.enabled = false;
        }
        let control_file = PathBuf::from(output_dir).join(".lr_control.json");
        training_loop.enable_adaptive_lr_with_control(adaptive_config, control_file);
        training_loop.set_snapshot_persist_dir(checkpoint_manager.checkpoint_dir().to_path_buf());
    }

    if config.resume {
        if let Some((lora_params, metadata)) = checkpoint_manager.load_latest()? {
            tracing::info!("Resuming from checkpoint at step {}", metadata.step);
            model.set_lora_parameters(&lora_params);
            training_loop.set_step(metadata.step);
            training_loop.set_epoch(metadata.epoch);
            // Restore persisted best-loss snapshot so rollback works after resume.
            if let Some(snapshot) =
                crate::checkpoint::load_best_snapshot(checkpoint_manager.checkpoint_dir())
            {
                training_loop.best_lora_snapshot = Some(snapshot);
                // Notify adaptive LR controller that a snapshot is available for rollback.
                if let Some(ref mut ctrl) = training_loop.adaptive_lr {
                    ctrl.set_has_snapshot(metadata.running_loss, metadata.step);
                }
                tracing::info!("Restored best snapshot from disk for rollback-on-resume");
            }
        } else {
            tracing::info!("No checkpoint found, starting fresh");
        }
    }

    tracing::info!("Starting LoRA training...");
    emit_phase(phase_cb, TrainingPhase::Training);

    // Notify all callbacks that training is starting (TrainingLoop doesn't do this)
    for cb in &mut training_loop.callbacks {
        cb.on_train_start();
    }

    // Dispatch: packed > compiled > metal-fused > standard
    if config.dispatch.sequence_packing {
        let model = training_loop.run_packed(
            model,
            train_dataset.clone(),
            eval_dataset.clone(),
            Some(checkpoint_manager),
        )?;

        let final_path = PathBuf::from(output_dir).join("lora_weights.safetensors");
        model.save_lora_weights(&final_path)?;
        tracing::info!("Saved LoRA weights to {:?}", final_path);
        save_adapter_config(
            &final_path,
            config.lora.r,
            config.lora.alpha,
            &config.lora.target_modules,
            config.lora.use_rslora,
        )?;
    } else if (config.dispatch.fused || config.dispatch.jit_compilation)
        && full_config.training.gradient_accumulation_steps == 1
    {
        let model = training_loop.run_compiled(
            model,
            train_dataset,
            eval_dataset,
            Some(checkpoint_manager),
        )?;

        let final_path = PathBuf::from(output_dir).join("lora_weights.safetensors");
        model.save_lora_weights(&final_path)?;
        tracing::info!("Saved LoRA weights to {:?}", final_path);
        save_adapter_config(
            &final_path,
            config.lora.r,
            config.lora.alpha,
            &config.lora.target_modules,
            config.lora.use_rslora,
        )?;
    } else if config.dispatch.metal_fused_optimizer {
        tracing::info!("Using Metal fused optimizer for training");
        training_loop.run_metal_fused(
            &mut model,
            train_dataset,
            eval_dataset,
            Some(checkpoint_manager),
        )?;

        let final_path = PathBuf::from(output_dir).join("lora_weights.safetensors");
        model.save_lora_weights(&final_path)?;
        tracing::info!("Saved LoRA weights to {:?}", final_path);
        save_adapter_config(
            &final_path,
            config.lora.r,
            config.lora.alpha,
            &config.lora.target_modules,
            config.lora.use_rslora,
        )?;
    } else {
        if (config.dispatch.fused || config.dispatch.jit_compilation)
            && full_config.training.gradient_accumulation_steps != 1
        {
            tracing::warn!(
                "Fused/JIT training requires gradient_accumulation_steps=1, falling back to standard training"
            );
        }
        training_loop.run(
            &mut model,
            train_dataset,
            eval_dataset,
            Some(checkpoint_manager),
        )?;

        let final_path = PathBuf::from(output_dir).join("lora_weights.safetensors");
        model.save_lora_weights(&final_path)?;
        tracing::info!("Saved LoRA weights to {:?}", final_path);
        save_adapter_config(
            &final_path,
            config.lora.r,
            config.lora.alpha,
            &config.lora.target_modules,
            config.lora.use_rslora,
        )?;
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

    if let Err(reason) = DynamicAneTrainerConfig::is_ane_compatible(&config_json) {
        anyhow::bail!("{}", reason);
    }

    let get_usize = |key: &str| -> anyhow::Result<usize> {
        config_json
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| anyhow::anyhow!("config.json missing '{key}'"))
    };
    let dim = get_usize("hidden_size")?;
    let hidden_dim = get_usize("intermediate_size")?;
    let n_heads = get_usize("num_attention_heads")?;
    let n_kv_heads = config_json
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(n_heads);
    let n_layers = get_usize("num_hidden_layers")?;
    let vocab_size = get_usize("vocab_size")?;

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
    // Also collect the (input, target) pairs as u32 for later remapping
    let mut pairs_u32: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
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
        // Pad to exact seq_len
        while input.len() < max_seq_len {
            input.push(pad_token);
            target.push(pad_token);
        }
        all_token_ids.extend_from_slice(&input);
        all_token_ids.extend_from_slice(&target);
        pairs_u32.push((input, target));
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
    for (input_u32, target_u32) in &pairs_u32 {
        let input = vocab_map.remap_u32(input_u32);
        let target = vocab_map.remap_u32(target_u32);
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
    let head_dim = config_json
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    let rms_norm_eps = config_json
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(1e-6);

    let trainer_config = DynamicAneTrainerConfig {
        dim,
        hidden_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        n_layers,
        vocab_size,
        seq_len: max_seq_len,
        // Full-parameter training needs much lower LR than LoRA.
        // LoRA default (1e-4) is 10-100x too high for all-param updates.
        // Cap at 2e-5 unless the user explicitly set a lower value.
        learning_rate: {
            let user_lr = full_config.training.learning_rate as f32;
            let max_full_param_lr = 5e-6;
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
        rms_norm_eps,
        loss_scale: config.dispatch.loss_scale,
        embedding_lr: full_config
            .training
            .embedding_learning_rate
            .map(|v| v as f32),
        ..Default::default()
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

    let lora_weights_path = PathBuf::from(output_dir).join("lora_weights.safetensors");

    Ok(TrainingResult {
        final_loss,
        total_steps: total_batches,
        total_tokens: total_tokens_processed,
        output_dir: PathBuf::from(output_dir),
        lora_weights_path,
    })
}

// ---------------------------------------------------------------------------
// Helper: dataset loading
// ---------------------------------------------------------------------------

fn load_dataset(
    path: &Path,
    is_parquet: bool,
    tokenizer: &Tokenizer,
    max_seq_len: usize,
    chat_template: &pmetal_data::chat_templates::ChatTemplate,
    columns: Option<&DatasetColumnConfig>,
) -> anyhow::Result<TrainingDataset> {
    if is_parquet {
        tracing::info!("Detected Parquet format");
        let explicit_col = columns.and_then(|c| c.text_column.as_deref());
        let prompt_col = columns.and_then(|c| c.prompt_column.as_deref());
        if let Some(col) = explicit_col {
            Ok(TrainingDataset::from_parquet_tokenized(
                path,
                tokenizer,
                col,
                max_seq_len,
                prompt_col,
            )?)
        } else {
            // Try well-known column names, then fall back to converting
            // parquet → temporary JSONL for chat-format datasets (e.g. "messages" column).
            let result =
                TrainingDataset::from_parquet_tokenized(path, tokenizer, "text", max_seq_len, None);
            if let Ok(ds) = result {
                return Ok(ds);
            }
            let result = TrainingDataset::from_parquet_tokenized(
                path,
                tokenizer,
                "content",
                max_seq_len,
                None,
            );
            if let Ok(ds) = result {
                return Ok(ds);
            }
            // Parquet with chat format (e.g. "messages" column): convert to
            // temporary JSONL and use the full-featured JSONL pipeline which
            // handles OpenAI/ShareGPT/Alpaca formats with chat templates.
            tracing::info!(
                "Parquet has no text/content column — converting to JSONL for chat format detection"
            );
            let tmp_dir = std::env::temp_dir().join("pmetal-parquet-convert");
            std::fs::create_dir_all(&tmp_dir)?;
            let tmp_jsonl = tmp_dir.join("converted.jsonl");
            {
                use std::io::Write;
                let file = std::fs::File::open(path)?;
                let builder =
                    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?;
                let reader = builder.build()?;
                let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp_jsonl)?);
                let mut buf = Vec::new();
                for batch in reader {
                    let batch = batch?;
                    buf.clear();
                    let mut json_writer = arrow::json::LineDelimitedWriter::new(&mut buf);
                    json_writer.write(&batch)?;
                    json_writer.finish()?;
                    out.write_all(&buf)?;
                }
                out.flush()?;
            }
            Ok(TrainingDataset::from_jsonl_tokenized(
                &tmp_jsonl,
                tokenizer,
                DatasetFormat::Auto,
                max_seq_len,
                Some(chat_template),
                columns,
            )?)
        }
    } else {
        Ok(TrainingDataset::from_jsonl_tokenized(
            path,
            tokenizer,
            DatasetFormat::Auto,
            max_seq_len,
            Some(chat_template),
            columns,
        )?)
    }
}

// ---------------------------------------------------------------------------
// Helper: resolve model path (async, may download from HF)
// ---------------------------------------------------------------------------

async fn resolve_model_path(model_id: &str) -> anyhow::Result<PathBuf> {
    #[cfg(feature = "hub")]
    {
        pmetal_hub::resolve_model_path(model_id)
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
                // Try JSONL/JSON/CSV first
                let resolved = TrainingDataset::resolve_dataset_path_pub(&dir);
                if let Ok(p) = resolved {
                    return Ok(p);
                }
                // Check for parquet files in the cached directory (no network call)
                let mut parquet_files: Vec<_> = std::fs::read_dir(&dir)?
                    .filter_map(|e| e.ok())
                    .flat_map(|e| {
                        let p = e.path();
                        if p.is_dir() {
                            // Recurse one level (e.g. default/train/)
                            std::fs::read_dir(&p)
                                .ok()
                                .into_iter()
                                .flatten()
                                .filter_map(|e2| e2.ok())
                                .flat_map(|e2| {
                                    let p2 = e2.path();
                                    if p2.is_dir() {
                                        std::fs::read_dir(&p2)
                                            .ok()
                                            .into_iter()
                                            .flatten()
                                            .filter_map(|e3| e3.ok())
                                            .map(|e3| e3.path())
                                            .collect::<Vec<_>>()
                                    } else {
                                        vec![p2]
                                    }
                                })
                                .collect::<Vec<_>>()
                        } else {
                            vec![p]
                        }
                    })
                    .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
                    .collect();
                parquet_files.sort();

                if let Some(pf) = parquet_files.into_iter().next() {
                    tracing::info!("Found parquet in cache: {}", pf.display());
                    return Ok(pf);
                }

                // Last resort: download parquet from HF API (network call)
                let parquet_paths =
                    pmetal_hub::download_dataset_parquet(&id, "train", None, None).await?;
                if parquet_paths.is_empty() {
                    anyhow::bail!("No JSONL or Parquet files found for dataset {}", id);
                }
                Ok(parquet_paths[0].clone())
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

pub fn save_adapter_config(
    lora_weights_path: &Path,
    r: usize,
    alpha: f32,
    target_modules: &[String],
    use_rslora: bool,
) -> anyhow::Result<()> {
    save_adapter_config_with_base(
        lora_weights_path,
        r,
        alpha,
        target_modules,
        use_rslora,
        None,
    )
}

/// Save adapter_config.json with optional base_model metadata.
pub fn save_adapter_config_with_base(
    lora_weights_path: &Path,
    r: usize,
    alpha: f32,
    target_modules: &[String],
    use_rslora: bool,
    base_model: Option<&str>,
) -> anyhow::Result<()> {
    let mut adapter_config = serde_json::json!({
        "r": r,
        "alpha": alpha,
        "target_modules": target_modules,
        "use_rslora": use_rslora,
    });
    if let Some(bm) = base_model {
        adapter_config["base_model"] = serde_json::Value::String(bm.to_string());
    }
    let config_path = lora_weights_path
        .parent()
        .unwrap_or(Path::new("."))
        .join("adapter_config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&adapter_config)?)?;
    tracing::info!("Saved adapter config to {:?}", config_path);
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: validate output path (security)
// ---------------------------------------------------------------------------

pub fn validate_output_path(path: &str, context: &str) -> anyhow::Result<PathBuf> {
    let path = PathBuf::from(path);

    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            anyhow::bail!(
                "Path traversal detected in {}: '{}' contains '..' component. \
                 Please use a path within the current directory.",
                context,
                path.display()
            );
        }
    }

    let cwd = std::env::current_dir()?;
    let resolved = if path.is_absolute() {
        path.clone()
    } else {
        cwd.join(&path)
    };

    if let Some(parent) = resolved.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let canonical = if resolved.exists() {
        resolved.canonicalize()?
    } else if let Some(parent) = resolved.parent() {
        let canonical_parent = parent.canonicalize()?;
        if let Some(filename) = resolved.file_name() {
            canonical_parent.join(filename)
        } else {
            canonical_parent
        }
    } else {
        resolved
    };

    let cwd_canonical = cwd.canonicalize()?;
    let home_dir = dirs::home_dir();
    let temp_dir = std::env::temp_dir().canonicalize().ok();

    let is_safe = canonical.starts_with(&cwd_canonical)
        || home_dir
            .as_ref()
            .map(|h| canonical.starts_with(h))
            .unwrap_or(false)
        || temp_dir
            .as_ref()
            .map(|t| canonical.starts_with(t))
            .unwrap_or(false)
        || canonical.starts_with("/tmp")
        || canonical.starts_with("/private/tmp");

    if !is_safe {
        anyhow::bail!(
            "Unsafe output path for {}: '{}' resolves to '{}' which is outside \
             the current directory, home directory, and temp directory.",
            context,
            path.display(),
            canonical.display()
        );
    }

    Ok(canonical)
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

// ---------------------------------------------------------------------------
// Internal: YAML config wrapper (moved from CLI)
// ---------------------------------------------------------------------------

/// Full training config loaded from YAML.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FullTrainingConfig {
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub lora: LoraConfig,
    #[serde(default)]
    pub training: TrainingConfig,
    #[serde(default)]
    pub dataset: DatasetConfig,
}

impl Default for FullTrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            lora: LoraConfig::default(),
            training: TrainingConfig::default(),
            dataset: DatasetConfig::default(),
        }
    }
}
