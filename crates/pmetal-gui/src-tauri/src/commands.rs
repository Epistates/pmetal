use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use futures::FutureExt;
use pmetal::prelude::TrainingCallback;
use pmetal_bridge::compat::ops;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use crate::state::{
    AppConfig, AppEvent, AppState, CachedModel, DistillationRun, DistillationStatus, GrpoRun,
    GrpoStatus, TrainingConfigSummary, TrainingRun, TrainingStatus, format_downloads, format_size,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct AppError(String);

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<reqwest::Error> for AppError {
    fn from(e: reqwest::Error) -> Self {
        AppError(e.to_string())
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError(e.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError(e.to_string())
    }
}

type Result<T> = std::result::Result<T, AppError>;

struct CancelOnFlag {
    cancelled: Arc<std::sync::atomic::AtomicBool>,
}

impl TrainingCallback for CancelOnFlag {
    fn should_stop(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Training metrics callback that streams updates to the frontend via
/// `tauri::ipc::Channel` — the Tauri-recommended mechanism for streaming
/// data from backend to frontend. Channel::send() is Send + Sync and
/// safe to call from any thread, including blocked tokio worker threads.
struct ChannelMetricsCallback {
    channel: tauri::ipc::Channel<serde_json::Value>,
    started_at: chrono::DateTime<chrono::Utc>,
    best_loss: Option<f64>,
}

impl TrainingCallback for ChannelMetricsCallback {
    fn on_train_start(&mut self) {
        let _ = self.channel.send(serde_json::json!({
            "event": "train_start",
            "status_message": "Training started, waiting for first step...",
        }));
    }

    fn on_step_end_with_metrics(&mut self, metrics: &pmetal::core::StepMetrics) {
        if self.best_loss.map_or(true, |b| metrics.loss < b) {
            self.best_loss = Some(metrics.loss);
        }
        let elapsed = (chrono::Utc::now() - self.started_at).num_seconds().max(1) as f64;
        let eta = if metrics.total_steps > 0 && metrics.step > 0 {
            let remaining = metrics.total_steps.saturating_sub(metrics.step) as f64;
            Some(((elapsed / metrics.step as f64) * remaining) as u64)
        } else {
            None
        };
        let _ = self.channel.send(serde_json::json!({
            "event": "step",
            "step": metrics.step,
            "total_steps": metrics.total_steps,
            "total_epochs": metrics.total_epochs,
            "epoch": metrics.epoch,
            "loss": metrics.loss,
            "best_loss": self.best_loss,
            "lr": metrics.lr,
            "tok_sec": metrics.tok_sec,
            "grad_norm": metrics.grad_norm,
            "eta_seconds": eta,
        }));
    }

    fn on_step_end(&mut self, step: usize, loss: f64) {
        if self.best_loss.map_or(true, |b| loss < b) {
            self.best_loss = Some(loss);
        }
        let _ = self.channel.send(serde_json::json!({
            "event": "step",
            "step": step,
            "loss": loss,
            "best_loss": self.best_loss,
        }));
    }

    fn on_epoch_end(&mut self, epoch: usize, metrics: &pmetal::core::EvalMetrics) {
        let _ = self.channel.send(serde_json::json!({
            "event": "epoch_end",
            "epoch": epoch,
            "loss": metrics.loss,
        }));
    }
}

// ---------------------------------------------------------------------------
// Response DTOs — names match api.ts interfaces exactly
// ---------------------------------------------------------------------------

/// Matches TS `SystemInfo` and `DeviceInfo` (same fields).
#[derive(Debug, Serialize)]
pub struct SystemInfo {
    pub version: String,
    pub platform: String,
    pub arch: String,
    pub is_apple_silicon: bool,
    pub gpu_name: String,
    pub chip_tier: Option<String>,
    pub total_memory: u64,
    pub available_memory: u64,
    pub total_memory_formatted: String,
    pub available_memory_formatted: String,
    pub gpu_cores: Option<u32>,
    pub ane_cores: Option<u32>,
    pub memory_bandwidth_gbps: Option<f64>,
    pub has_ane: bool,
    pub has_nax: bool,
}

/// Matches TS `DashboardStats`.
#[derive(Debug, Serialize)]
pub struct DashboardStats {
    pub models_count: usize,
    pub total_model_size: String,
    pub active_training_runs: usize,
    pub completed_training_runs: usize,
    pub total_training_runs: usize,
    pub active_grpo_runs: usize,
    pub active_distillation_runs: usize,
}

/// Matches TS `ModelInfo`.
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub path: String,
    pub size: u64,
    pub size_formatted: String,
    pub model_type: Option<String>,
    pub hidden_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub vocab_size: Option<usize>,
    pub context_length: Option<usize>,
}

/// Matches TS `ModelFitInfo`.
#[derive(Debug, Serialize)]
pub struct ModelFitInfo {
    pub inference_fit: String,
    pub training_fit: String,
    pub weights_gb: f64,
    pub inference_memory_gb: f64,
    pub training_memory_gb: f64,
    pub available_memory_gb: f64,
    pub estimated_tps: Option<f64>,
    pub recommended_batch_size: u32,
}

/// Matches TS `HubSearchResult`.
#[derive(Debug, Serialize)]
pub struct HubSearchResult {
    pub id: String,
    pub author: Option<String>,
    pub downloads: u64,
    pub downloads_formatted: String,
    pub likes: u64,
    pub pipeline_tag: Option<String>,
    pub is_gated: bool,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
}

/// Matches TS `DatasetSearchResult`.
#[derive(Debug, Serialize)]
pub struct DatasetSearchResult {
    pub id: String,
    pub author: Option<String>,
    pub downloads: u64,
    pub downloads_formatted: String,
    pub likes: u64,
    pub tags: Vec<String>,
    pub description: Option<String>,
}

/// Matches TS `CachedDatasetInfo`.
#[derive(Debug, Serialize)]
pub struct CachedDatasetInfo {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub size_formatted: String,
}

/// Matches TS `MergeStrategy`.
#[derive(Debug, Serialize)]
pub struct MergeStrategy {
    pub name: String,
    pub description: String,
    pub supports_weights: bool,
}

/// Matches TS `FuseResult`.
#[derive(Debug, Serialize)]
pub struct FuseResult {
    pub output_dir: String,
    pub model_size_bytes: u64,
}

/// A discovered LoRA adapter on disk. Matches TS `TrainedAdapter`.
#[derive(Debug, Clone, Serialize)]
pub struct TrainedAdapter {
    /// Absolute path to the adapter directory.
    pub path: String,
    /// Human-readable name (directory basename).
    pub name: String,
    /// Base model this adapter was trained on (if known).
    pub base_model: Option<String>,
    /// LoRA rank from adapter_config.json.
    pub rank: Option<u32>,
    /// LoRA alpha from adapter_config.json.
    pub alpha: Option<f32>,
    /// Size of the weights file in bytes.
    pub size_bytes: u64,
}

/// Model generation/training defaults read from generation_config.json + config.json.
/// Matches TS `ModelDefaults`.
#[derive(Debug, Default, Serialize)]
pub struct ModelDefaults {
    // Generation params (from generation_config.json)
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub max_new_tokens: Option<u32>,
    pub repetition_penalty: Option<f32>,
    // Model arch info (from config.json)
    pub max_position_embeddings: Option<u32>,
    pub hidden_size: Option<u32>,
    pub num_hidden_layers: Option<u32>,
    pub vocab_size: Option<u32>,
}

// ---------------------------------------------------------------------------
// Request DTOs — match api.ts invoke calls
// ---------------------------------------------------------------------------

/// Full training config matching TS `TrainingConfig`.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    pub model: String,
    pub dataset: Option<String>,
    pub method: String,
    pub epochs: Option<u32>,
    pub learning_rate: Option<f64>,
    pub batch_size: Option<u32>,
    pub lora_rank: Option<u32>,
    pub lora_alpha: Option<u32>,
    pub lora_dropout: Option<f64>,
    pub use_rslora: Option<bool>,
    pub use_dora: Option<bool>,
    pub output_dir: Option<String>,
    pub load_in_4bit: Option<bool>,
    pub gradient_accumulation_steps: Option<u32>,
    pub max_seq_len: Option<u32>,
    pub text_column: Option<String>,
    pub dataset_format: Option<String>,
    pub embedding_lr: Option<f64>,
    pub jit_compilation: Option<bool>,
    pub gradient_checkpointing: Option<bool>,
    pub gradient_checkpointing_layers: Option<u32>,
    pub flash_attention: Option<bool>,
    pub fused_optimizer: Option<bool>,
    pub warmup_steps: Option<u32>,
    pub weight_decay: Option<f64>,
    pub max_grad_norm: Option<f64>,
    pub save_steps: Option<u32>,
    pub logging_steps: Option<u32>,
    pub lr_scheduler: Option<String>,
    pub sequence_packing: Option<bool>,
    pub resume_from: Option<String>,
    pub prompt_column: Option<String>,
    pub response_column: Option<String>,
    // DPO-specific
    pub dpo_beta: Option<f64>,
    pub dpo_loss_type: Option<String>,
    pub ref_model: Option<String>,
    // SimPO-specific
    pub simpo_beta: Option<f64>,
    pub simpo_gamma: Option<f64>,
    // ORPO-specific
    pub orpo_lambda: Option<f64>,
    // KTO-specific
    pub kto_desirable_weight: Option<f64>,
    pub kto_undesirable_weight: Option<f64>,
}

/// Full distillation config matching TS `DistillationConfig`.
#[derive(Debug, Deserialize)]
pub struct DistillationConfig {
    pub student_model: String,
    pub teacher_model: String,
    pub dataset: Option<String>,
    pub loss_type: Option<String>,
    pub temperature: Option<f32>,
    pub alpha: Option<f32>,
    pub epochs: Option<u32>,
    pub learning_rate: Option<f64>,
    pub batch_size: Option<u32>,
    pub lora_rank: Option<u32>,
    pub lora_alpha: Option<u32>,
    pub max_seq_len: Option<u32>,
    pub output_dir: Option<String>,
    pub text_column: Option<String>,
}

/// Full GRPO config matching TS `GrpoConfig`.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct GrpoConfig {
    pub model: String,
    pub dataset: Option<String>,
    pub epochs: Option<u32>,
    pub learning_rate: Option<f64>,
    pub batch_size: Option<u32>,
    pub group_size: Option<u32>,
    pub beta: Option<f64>,
    pub lora_rank: Option<u32>,
    pub lora_alpha: Option<u32>,
    pub max_seq_len: Option<u32>,
    pub output_dir: Option<String>,
    pub use_reasoning_rewards: Option<bool>,
    pub text_column: Option<String>,
    /// KV cache quantization bits for rollout generation (2, 4, or 8).
    pub kv_cache_bits: Option<u8>,
}

/// Inference config matching TS `InferenceConfig`.
///
/// All sampling fields are `Option` — `None` means "use model's
/// `generation_config.json` default" (same behavior as CLI).
#[derive(Debug, Deserialize, Clone)]
pub struct InferenceMessage {
    pub role: String,
    pub content: String,
}

fn chat_message_from_inference_message(
    message: &InferenceMessage,
) -> std::result::Result<pmetal::data::chat_templates::Message, String> {
    match message.role.as_str() {
        "user" => Ok(pmetal::data::chat_templates::Message::user(
            message.content.clone(),
        )),
        "assistant" => Ok(pmetal::data::chat_templates::Message::assistant(
            message.content.clone(),
        )),
        "system" => Ok(pmetal::data::chat_templates::Message::system(
            message.content.clone(),
        )),
        other => Err(format!(
            "unsupported chat role '{other}' in GUI inference history"
        )),
    }
}

#[derive(Debug, Deserialize)]
pub struct InferenceConfig {
    pub model: String,
    pub lora_path: Option<String>,
    pub prompt: String,
    pub messages: Option<Vec<InferenceMessage>>,
    pub system_message: Option<String>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    /// Quantize weights to FP8 E4M3 for ~2x memory savings.
    pub fp8: Option<bool>,
    /// Disable thinking mode for models that support it.
    pub no_thinking: Option<bool>,
    /// Path to packed expert weights directory for SSD-offloaded MoE inference.
    pub experts_dir: Option<String>,
    /// KV cache quantization bits (8=q8_0, 4=q4_0, 0=fp16). None = auto.
    pub kv_quant: Option<u8>,
    /// Override key bits for asymmetric K/V quantization.
    pub kv_k_bits: Option<u8>,
    /// Override value bits for asymmetric K/V quantization.
    pub kv_v_bits: Option<u8>,
    /// KV cache quantization group size.
    pub kv_group_size: Option<usize>,
    /// Disable KV cache quantization entirely (force fp16).
    pub no_kv_quant: Option<bool>,
    /// Use TurboQuant KV cache instead of MLX affine quantization.
    pub kv_turboquant: Option<bool>,
    /// Mixed-bit TurboQuant preset (`q2_5` or `q3_5`).
    pub kv_turboquant_preset: Option<String>,
    /// TurboQuant v2 affine mixed-bit preset ("q2_5" or "q3_5").
    pub kv_quant_preset: Option<String>,
    /// Enable QJL residual correction for Q2-Q3 uniform KV cache.
    pub kv_qjl: Option<bool>,
}

/// Merge config matching TS `MergeConfig`.
#[derive(Debug, Deserialize)]
pub struct MergeConfig {
    pub base_model: String,
    pub models: Vec<MergeModelEntry>,
    pub strategy: String,
    pub output: String,
}

#[derive(Debug, Deserialize)]
pub struct MergeModelEntry {
    pub model: String,
    pub weight: f64,
}

// ---------------------------------------------------------------------------
// System commands
// ---------------------------------------------------------------------------

/// Returns combined `SystemInfo` / `DeviceInfo` from the library runtime.
async fn get_system_info_inner() -> SystemInfo {
    let version = pmetal::version::VERSION.to_string();
    let platform = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    let is_apple_silicon = arch == "aarch64" && platform == "macos";

    let device_info = pmetal::version::device_info();
    let total_memory = (device_info.memory_total_gb * 1024.0_f64.powi(3)) as u64;
    let available_memory = (device_info.memory_available_gb * 1024.0_f64.powi(3)) as u64;

    let mut gpu_name = "Apple GPU".to_string();
    let mut chip_tier: Option<String> = None;
    let mut gpu_cores: Option<u32> = None;
    let mut ane_cores: Option<u32> = None;
    let mut memory_bandwidth_gbps: Option<f64> = None;
    let mut has_ane = is_apple_silicon;
    let mut has_nax = false;

    if let Ok(ctx) = pmetal::metal::MetalContext::global() {
        let props = ctx.properties();
        gpu_name = props.name.clone();
        gpu_cores = Some(props.gpu_core_count);
        ane_cores = Some(props.ane_core_count);
        memory_bandwidth_gbps = Some(props.memory_bandwidth_gbps);
        has_ane = props.ane_core_count > 0;
        has_nax = props.has_nax;
        chip_tier = Some(format!("{:?}", props.device_tier).to_lowercase());
    }

    SystemInfo {
        version,
        platform,
        arch,
        is_apple_silicon,
        gpu_name,
        chip_tier,
        total_memory,
        available_memory,
        total_memory_formatted: format_size(total_memory),
        available_memory_formatted: format_size(available_memory),
        gpu_cores,
        ane_cores,
        memory_bandwidth_gbps,
        has_ane,
        has_nax,
    }
}

#[tauri::command]
pub async fn get_system_info() -> Result<SystemInfo> {
    Ok(get_system_info_inner().await)
}

/// `get_device_info` returns the same struct as `get_system_info` — both map to SystemInfo.
#[tauri::command]
pub async fn get_device_info() -> Result<SystemInfo> {
    Ok(get_system_info_inner().await)
}

#[tauri::command]
pub async fn get_config(state: State<'_, AppState>) -> Result<AppConfig> {
    Ok(state.config.read().await.clone())
}

#[tauri::command]
pub async fn set_config(state: State<'_, AppState>, config: AppConfig) -> Result<()> {
    *state.config.write().await = config;
    state.save_config().await;
    Ok(())
}

#[tauri::command]
pub async fn get_dashboard_stats(state: State<'_, AppState>) -> Result<DashboardStats> {
    let models = state.cached_models.read().await;
    let training = state.training_runs.read().await;
    let distillation = state.distillation_runs.read().await;
    let grpo = state.grpo_runs.read().await;

    let total_size: u64 = models.iter().map(|m| m.size).sum();

    let active_training = training
        .iter()
        .filter(|r| r.status == TrainingStatus::Running)
        .count();
    let completed_training = training
        .iter()
        .filter(|r| r.status == TrainingStatus::Completed)
        .count();
    let active_grpo = grpo
        .iter()
        .filter(|r| r.status == GrpoStatus::Running)
        .count();
    let active_distillation = distillation
        .iter()
        .filter(|r| {
            r.status == DistillationStatus::Training
                || r.status == DistillationStatus::LoadingModels
                || r.status == DistillationStatus::GeneratingSignals
        })
        .count();

    Ok(DashboardStats {
        models_count: models.len(),
        total_model_size: format_size(total_size),
        active_training_runs: active_training,
        completed_training_runs: completed_training,
        total_training_runs: training.len(),
        active_grpo_runs: active_grpo,
        active_distillation_runs: active_distillation,
    })
}

// ---------------------------------------------------------------------------
// Model commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn list_models(state: State<'_, AppState>) -> Result<Vec<CachedModel>> {
    Ok(state.cached_models.read().await.clone())
}

/// Returns a `ModelInfo` by reading `config.json` from the model snapshot directory.
#[tauri::command]
pub async fn get_model_info(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<Option<ModelInfo>> {
    let cached = {
        let models = state.cached_models.read().await;
        models.iter().find(|m| m.id == model_id).cloned()
    };

    let Some(cached) = cached else {
        return Ok(None);
    };

    // Attempt to read config.json from the snapshots directory
    let config_json = read_model_config_json(&cached.path).await;

    let model_type = config_json
        .as_ref()
        .and_then(|v| v["model_type"].as_str().map(str::to_string))
        .or_else(|| cached.model_type.clone());

    let hidden_size = config_json
        .as_ref()
        .and_then(|v| v["hidden_size"].as_u64().map(|n| n as usize));

    let num_layers = config_json.as_ref().and_then(|v| {
        v["num_hidden_layers"]
            .as_u64()
            .or_else(|| v["n_layer"].as_u64())
            .map(|n| n as usize)
    });

    let vocab_size = config_json
        .as_ref()
        .and_then(|v| v["vocab_size"].as_u64().map(|n| n as usize));

    let context_length = config_json.as_ref().and_then(|v| {
        v["max_position_embeddings"]
            .as_u64()
            .or_else(|| v["max_seq_len"].as_u64())
            .or_else(|| v["n_ctx"].as_u64())
            .map(|n| n as usize)
    });

    Ok(Some(ModelInfo {
        id: cached.id,
        path: cached.path,
        size: cached.size,
        size_formatted: cached.size_formatted,
        model_type,
        hidden_size,
        num_layers,
        vocab_size,
        context_length,
    }))
}

/// Read generation_config.json + config.json defaults for a model.
///
/// Returns recommended sampling parameters and model architecture info
/// so the GUI can auto-fill params when a model is selected.
#[tauri::command]
pub async fn get_model_defaults(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<ModelDefaults> {
    let model_path = {
        let models = state.cached_models.read().await;
        models
            .iter()
            .find(|m| m.id == model_id)
            .map(|m| m.path.clone())
            .unwrap_or_default()
    };

    if model_path.is_empty() {
        return Ok(ModelDefaults::default());
    }
    let model_path = PathBuf::from(&model_path);

    let mut defaults = ModelDefaults::default();

    // Read generation_config.json for sampling params
    let gen_config_path = model_path.join("generation_config.json");
    if let Ok(content) = tokio::fs::read_to_string(&gen_config_path).await {
        if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&content) {
            defaults.temperature = cfg["temperature"].as_f64().map(|v| v as f32);
            defaults.top_k = cfg["top_k"].as_u64().map(|v| v as u32);
            defaults.top_p = cfg["top_p"].as_f64().map(|v| v as f32);
            defaults.repetition_penalty = cfg["repetition_penalty"].as_f64().map(|v| v as f32);
            defaults.max_new_tokens = cfg["max_new_tokens"].as_u64().map(|v| v as u32);
        }
    }

    // Read config.json for model arch info
    if let Some(cfg) = read_model_config_json(&model_path.to_string_lossy()).await {
        defaults.max_position_embeddings = cfg["max_position_embeddings"]
            .as_u64()
            .or_else(|| cfg["max_seq_len"].as_u64())
            .map(|v| v as u32);
        defaults.hidden_size = cfg["hidden_size"].as_u64().map(|v| v as u32);
        defaults.num_hidden_layers = cfg["num_hidden_layers"].as_u64().map(|v| v as u32);
        defaults.vocab_size = cfg["vocab_size"].as_u64().map(|v| v as u32);
    }

    Ok(defaults)
}

#[tauri::command]
pub async fn delete_model(state: State<'_, AppState>, model_id: String) -> Result<()> {
    let path = {
        let models = state.cached_models.read().await;
        models
            .iter()
            .find(|m| m.id == model_id)
            .map(|m| m.path.clone())
    };

    if let Some(path) = path {
        tokio::fs::remove_dir_all(&path)
            .await
            .map_err(|e| AppError(format!("Failed to delete model directory: {}", e)))?;

        state
            .cached_models
            .write()
            .await
            .retain(|m| m.id != model_id);
        let _ = state.event_tx.send(AppEvent::ModelRemoved { model_id });
    }

    Ok(())
}

/// Add a custom directory to scan for models (e.g. LM Studio path).
#[tauri::command]
pub async fn add_model_directory(
    state: State<'_, AppState>,
    path: String,
) -> Result<Vec<CachedModel>> {
    let dir = PathBuf::from(&path);
    if !dir.exists() {
        return Err(AppError(format!("Directory does not exist: {path}")));
    }
    if !dir.is_dir() {
        return Err(AppError(format!("Not a directory: {path}")));
    }

    // Add to config if not already present
    {
        let mut cfg = state.config.write().await;
        if !cfg.custom_model_dirs.contains(&path) {
            cfg.custom_model_dirs.push(path);
        }
    }
    state.save_config().await;

    // Rescan all models
    state.refresh_cached_models().await;
    Ok(state.cached_models.read().await.clone())
}

/// Remove a custom model directory.
#[tauri::command]
pub async fn remove_model_directory(
    state: State<'_, AppState>,
    path: String,
) -> Result<Vec<CachedModel>> {
    {
        let mut cfg = state.config.write().await;
        cfg.custom_model_dirs.retain(|d| d != &path);
    }
    state.save_config().await;

    // Rescan all models
    state.refresh_cached_models().await;
    Ok(state.cached_models.read().await.clone())
}

/// List configured custom model directories.
#[tauri::command]
pub async fn list_model_directories(state: State<'_, AppState>) -> Result<Vec<String>> {
    Ok(state.config.read().await.custom_model_dirs.clone())
}

#[tauri::command]
pub async fn download_model(
    state: State<'_, AppState>,
    app_handle: AppHandle,
    model_id: String,
    revision: Option<String>,
) -> Result<String> {
    let run_id = uuid::Uuid::new_v4().to_string();
    let run_id_task = run_id.clone();
    let model_id_task = model_id.clone();

    let hf_token = state.config.read().await.hf_token.clone();
    let cached_models = state.cached_models.clone();
    let cache_dir = state.config.read().await.cache_dir.clone();

    tokio::spawn(async move {
        let _ = app_handle.emit("download-started", &run_id_task);

        let _ = app_handle.emit(
            "download-progress",
            serde_json::json!({
                "run_id": run_id_task,
                "line": format!("Resolving {model_id_task}"),
            }),
        );

        let token = hf_token
            .as_ref()
            .map(|s| pmetal::core::SecretString::from(s.clone()));
        match pmetal::hub::download_model(&model_id_task, revision.as_deref(), token.as_ref()).await
        {
            Ok(path) => {
                let _ = app_handle.emit(
                    "download-progress",
                    serde_json::json!({
                        "run_id": run_id_task,
                        "line": format!("Downloaded to {}", path.display()),
                    }),
                );

                let hub_dir = PathBuf::from(&cache_dir).join("hub");
                let models = crate::state::scan_hub_cache_pub(&hub_dir).await;
                *cached_models.write().await = models;

                let _ = app_handle.emit("download-completed", &run_id_task);
            }
            Err(e) => {
                let _ = app_handle.emit(
                    "download-error",
                    serde_json::json!({ "run_id": run_id_task, "error": e.to_string() }),
                );
            }
        }
    });

    Ok(run_id)
}

#[tauri::command]
pub async fn search_hub_models(
    state: State<'_, AppState>,
    query: String,
    limit: Option<u32>,
) -> Result<Vec<HubSearchResult>> {
    let token = state.config.read().await.hf_token.clone();
    search_hf_models_inner(query, limit.unwrap_or(20), token).await
}

#[tauri::command]
pub async fn get_trending_models(
    state: State<'_, AppState>,
    limit: Option<u32>,
) -> Result<Vec<HubSearchResult>> {
    let token = state.config.read().await.hf_token.clone();
    search_hf_models_inner(String::new(), limit.unwrap_or(20), token).await
}

#[tauri::command]
pub async fn get_model_fit(state: State<'_, AppState>, model_id: String) -> Result<ModelFitInfo> {
    let available_memory = get_available_memory_bytes()
        .await
        .unwrap_or_else(|| get_total_memory_bytes_sync() / 2);
    let available_memory_gb = available_memory as f64 / (1024.0_f64.powi(3));

    let param_b = estimate_params_b(&model_id);

    // Attempt to get actual size from cached models
    let size_from_cache = {
        let models = state.cached_models.read().await;
        models.iter().find(|m| m.id == model_id).map(|m| m.size)
    };

    let weights_gb = if let Some(bytes) = size_from_cache {
        bytes as f64 / (1024.0_f64.powi(3))
    } else {
        // Estimate: fp16 ~= 2 bytes/param
        param_b * 2.0
    };

    // Inference: weights + KV cache overhead (~10%)
    let inference_memory_gb = weights_gb * 1.1;
    // Training with LoRA: weights + optimizer states + activations (~4x)
    let training_memory_gb = weights_gb * 4.0;

    // Bandwidth estimate for tok/s
    let bandwidth_gbps = get_bandwidth_gbps().await.unwrap_or(400.0);
    let estimated_tps = if weights_gb > 0.01 {
        Some((bandwidth_gbps / weights_gb) * 0.55)
    } else {
        None
    };

    let inference_fit = if inference_memory_gb <= available_memory_gb * 0.85 {
        "fits".to_string()
    } else if inference_memory_gb <= available_memory_gb {
        "tight".to_string()
    } else {
        "too_large".to_string()
    };

    let training_fit = if training_memory_gb <= available_memory_gb * 0.85 {
        "fits".to_string()
    } else if training_memory_gb <= available_memory_gb {
        "tight".to_string()
    } else {
        "too_large".to_string()
    };

    // Recommended batch size: scale with available headroom
    let headroom_gb = (available_memory_gb - inference_memory_gb).max(0.0);
    // Rough estimate: ~0.5 GB per batch slot for typical models
    let recommended_batch_size = ((headroom_gb / (weights_gb * 0.05 + 0.5)) as u32).clamp(1, 64);

    Ok(ModelFitInfo {
        inference_fit,
        training_fit,
        weights_gb,
        inference_memory_gb,
        training_memory_gb,
        available_memory_gb,
        estimated_tps,
        recommended_batch_size,
    })
}

// ---------------------------------------------------------------------------
// Dataset commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn search_hub_datasets(
    state: State<'_, AppState>,
    query: String,
    limit: Option<u32>,
) -> Result<Vec<DatasetSearchResult>> {
    let token = state.config.read().await.hf_token.clone();
    search_hf_datasets_inner(query, limit.unwrap_or(20), token).await
}

#[tauri::command]
pub async fn get_trending_datasets(
    state: State<'_, AppState>,
    limit: Option<u32>,
) -> Result<Vec<DatasetSearchResult>> {
    let token = state.config.read().await.hf_token.clone();
    search_hf_datasets_inner(String::new(), limit.unwrap_or(20), token).await
}

#[tauri::command]
pub async fn list_cached_datasets(state: State<'_, AppState>) -> Result<Vec<CachedDatasetInfo>> {
    let cache_dir = state.config.read().await.cache_dir.clone();
    let hub_dir = PathBuf::from(&cache_dir).join("hub");
    let mut datasets = Vec::new();

    let mut rd = match tokio::fs::read_dir(&hub_dir).await {
        Ok(rd) => rd,
        Err(_) => return Ok(datasets),
    };

    while let Ok(Some(entry)) = rd.next_entry().await {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("datasets--") {
            let id = name_str
                .strip_prefix("datasets--")
                .unwrap_or(&name_str)
                .replace("--", "/");
            let path = entry.path();
            let size_bytes = dir_size_simple(&path).await;
            datasets.push(CachedDatasetInfo {
                name: id,
                path: path.to_string_lossy().into_owned(),
                size_bytes,
                size_formatted: format_size(size_bytes),
            });
        }
    }

    Ok(datasets)
}

/// Download a dataset from HuggingFace Hub (or validate a local path).
///
/// Returns the resolved path to the dataset file on disk.
#[tauri::command]
pub async fn download_dataset(dataset_id: String) -> Result<String> {
    let path = resolve_dataset_path(&dataset_id).await?;
    Ok(path.to_string_lossy().into_owned())
}

/// Peek at the first JSONL record in a file and return the available field names.
///
/// Peek at a dataset file: returns column names and rough length statistics
/// (sampled from the first 100 records) so the frontend can show seq len warnings.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DatasetPeek {
    pub columns: Vec<String>,
    /// Estimated average token count (chars / 4 heuristic, sampled from first 100 rows).
    pub avg_tokens_estimate: usize,
    /// Estimated max token count in the sample.
    pub max_tokens_estimate: usize,
    /// Suggested max_seq_len (next power of two covering the p95 estimate).
    pub suggested_seq_len: usize,
    /// Number of rows sampled.
    pub rows_sampled: usize,
}

#[tauri::command]
pub async fn peek_dataset_columns(path: String, limit: Option<usize>) -> Result<DatasetPeek> {
    use std::io::{BufRead, BufReader};
    let sample_limit = limit.unwrap_or(100);

    let p = std::path::PathBuf::from(&path);
    let resolved = pmetal::data::TrainingDataset::resolve_dataset_path_pub(&p).unwrap_or(p);

    let columns = pmetal::data::peek_columns(&resolved).map_err(|e| AppError(e.to_string()))?;

    // Sample first 100 rows for length estimates
    let mut char_lengths: Vec<usize> = Vec::new();
    if let Ok(file) = std::fs::File::open(&resolved) {
        let reader = BufReader::new(file);
        let iter: Box<dyn Iterator<Item = _>> = if sample_limit > 0 {
            Box::new(reader.lines().take(sample_limit))
        } else {
            Box::new(reader.lines()) // scan all
        };
        for line in iter {
            if let Ok(line) = line {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                // Sum all string-valued fields as a rough content length
                if let Ok(obj) =
                    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(trimmed)
                {
                    let total_chars: usize = obj
                        .values()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.len())
                        .sum();
                    char_lengths.push(total_chars);
                }
            }
        }
    }

    let rows_sampled = char_lengths.len();
    // Rough estimate: 1 token ≈ 4 characters for English text
    let token_estimates: Vec<usize> = char_lengths.iter().map(|&c| c / 4).collect();
    let avg = if token_estimates.is_empty() {
        0
    } else {
        token_estimates.iter().sum::<usize>() / token_estimates.len()
    };
    let max = token_estimates.iter().copied().max().unwrap_or(0);

    // p95 estimate for suggested seq len
    let mut sorted = token_estimates;
    sorted.sort();
    let p95 = sorted
        .get((sorted.len() as f64 * 0.95) as usize)
        .copied()
        .unwrap_or(avg);
    // Round up to next multiple of 64 (practical for GPU alignment)
    let suggested = if p95 > 0 { (p95 + 63) / 64 * 64 } else { 2048 };

    Ok(DatasetPeek {
        columns,
        avg_tokens_estimate: avg,
        max_tokens_estimate: max,
        suggested_seq_len: suggested,
        rows_sampled,
    })
}

// ---------------------------------------------------------------------------
// Training commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn start_training(
    state: State<'_, AppState>,
    _app_handle: AppHandle,
    config: TrainingConfig,
    on_metrics: tauri::ipc::Channel<serde_json::Value>,
) -> Result<String> {
    if !matches!(config.method.as_str(), "sft" | "lora" | "qlora" | "ane") {
        return Err(AppError(format!(
            "Unsupported training method '{}'. Expected: ane, lora, qlora, or sft.",
            config.method
        )));
    }

    if config.resume_from.is_some() {
        return Err(AppError(
            "Resume from checkpoint is not yet supported in GUI mode. \
             Use the CLI instead: pmetal train --resume"
                .to_string(),
        ));
    }

    // -----------------------------------------------------------------------
    // Pre-flight validation: resolve dataset before creating the run so that
    // HF download errors surface immediately rather than silently after spawn.
    // -----------------------------------------------------------------------
    let dataset_id = config
        .dataset
        .clone()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| AppError("A dataset is required for training".to_string()))?;

    // Resolve HF dataset IDs to a local path (downloads if necessary).
    // This runs synchronously relative to the caller so errors are returned
    // to the frontend before any run record is created.
    let resolved_dataset = resolve_dataset_path(&dataset_id)
        .await
        .map_err(|e| AppError(format!("Dataset not found: {e}")))?;

    let total_epochs = config.epochs.unwrap_or(3);
    // Resolve output_dir to an absolute path under the user's home directory
    // so it doesn't depend on the GUI process's working directory.
    //
    // Default naming: "{model_short}-{method}-{YYYYMMDD-HHMM}" so trained
    // adapters are easily identifiable in the fuse/inference dropdowns.
    let output_dir = {
        let raw = config.output_dir.as_deref().unwrap_or("");
        let p = PathBuf::from(raw);
        let base = dirs::home_dir()
            .map(|h| h.join("pmetal-output"))
            .unwrap_or_else(|| PathBuf::from("./pmetal-output"));
        let _ = std::fs::create_dir_all(&base);

        if !raw.is_empty() && p.is_absolute() {
            p.to_string_lossy().to_string()
        } else if !raw.is_empty() && raw != "./output" {
            // User provided a custom relative name — use it under ~/pmetal-output/
            base.join(p.file_name().unwrap_or(std::ffi::OsStr::new("output")))
                .to_string_lossy()
                .to_string()
        } else {
            // Auto-generate a descriptive name: "Qwen3-0.6B-lora-20260318-2145"
            let model_short = config.model.rsplit('/').next().unwrap_or(&config.model);
            let method = &config.method;
            let ts = chrono::Local::now().format("%Y%m%d-%H%M");
            base.join(format!("{model_short}-{method}-{ts}"))
                .to_string_lossy()
                .to_string()
        }
    };
    let metrics_path = PathBuf::from(&output_dir).join("metrics.jsonl");

    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Build a config summary for display in the UI.
    let config_summary = TrainingConfigSummary {
        learning_rate: config.learning_rate.unwrap_or(1e-4),
        batch_size: config.batch_size.unwrap_or(1) as usize,
        max_seq_len: config.max_seq_len.unwrap_or(2048) as usize,
        lora_rank: config.lora_rank.map(|r| r as usize),
        lora_alpha: config.lora_alpha.map(|a| a as f32),
        sequence_packing: config.sequence_packing.unwrap_or(true),
        flash_attention: config.flash_attention.unwrap_or(true),
        jit_compilation: config.jit_compilation.unwrap_or(true),
        gradient_checkpointing: config.gradient_checkpointing.unwrap_or(true),
    };

    let mut run = TrainingRun::new(
        &config.model,
        &config.method,
        Some(&dataset_id),
        Some(&output_dir),
        total_epochs,
    );
    run.status = TrainingStatus::Running;
    run.status_message = Some("Starting…".to_string());
    run.config_summary = Some(config_summary);
    let run_id = run.id.clone();

    // Register cancellation flag before creating the run so the flag is always
    // present when the run record is visible to the frontend.
    state
        .cancel_flags
        .write()
        .await
        .insert(run_id.clone(), cancel_flag.clone());

    state.create_training_run(run).await;

    let run_id_task = run_id.clone();
    let state_arc = state.training_runs.clone();
    let event_tx = state.event_tx.clone();
    let cancel_flags = state.cancel_flags.clone();

    // Write training_info.json before spawning so the adapter scanner can
    // find base_model + dataset even if training is still running.
    let _ = std::fs::create_dir_all(&output_dir);
    {
        let info = serde_json::json!({
            "base_model": config.model,
            "dataset": dataset_id,
            "method": config.method,
            "created": chrono::Local::now().to_rfc3339(),
        });
        let _ = std::fs::write(
            PathBuf::from(&output_dir).join("training_info.json"),
            serde_json::to_string_pretty(&info).unwrap_or_default(),
        );
    }

    tokio::spawn(async move {
        let _ = tokio::fs::create_dir_all(&output_dir).await;
        let _ = tokio::fs::write(&metrics_path, "").await;

        // Emit a "loading model" status before the expensive model load.
        {
            let mut runs = state_arc.write().await;
            if let Some(run) = runs.iter_mut().find(|r| r.id == run_id_task) {
                run.status_message = Some("Loading model…".to_string());
                let _ = event_tx.send(AppEvent::TrainingUpdate { run: run.clone() });
            }
        }

        // Build orchestrator config from GUI DTO
        let qlora = if config.method == "qlora" || config.load_in_4bit == Some(true) {
            Some(pmetal::trainer::QLoraOrchConfig {
                scheme: pmetal::trainer::QuantizationScheme::Nf4,
                block_size: 64,
                double_quant: false,
            })
        } else {
            None
        };

        // Build column config from GUI
        let columns = {
            let mut text_column = None;
            let mut text_columns = None;
            let prompt_column = config.prompt_column.clone();
            let response_column = config.response_column.clone();
            if let Some(ref tc) = config.text_column {
                if tc.contains('+') {
                    text_columns = Some(tc.split('+').map(str::to_string).collect::<Vec<_>>());
                    text_column = text_columns.as_ref().and_then(|c| c.first().cloned());
                } else {
                    text_column = Some(tc.clone());
                }
            }
            if text_column.is_some()
                || text_columns.is_some()
                || prompt_column.is_some()
                || response_column.is_some()
            {
                Some(pmetal::data::DatasetColumnConfig {
                    text_column,
                    text_columns,
                    column_separator: Some("\n\n".to_string()),
                    prompt_column,
                    response_column,
                })
            } else {
                None
            }
        };

        let job_config = pmetal::trainer::TrainingJobConfig {
            model_id: config.model.clone(),
            dataset: resolved_dataset.to_string_lossy().into_owned(),
            eval_dataset: None,
            output_dir: output_dir.clone(),
            lora: pmetal::core::LoraConfig {
                r: config.lora_rank.unwrap_or(16) as usize,
                alpha: config.lora_alpha.unwrap_or(32) as f32,
                dropout: config.lora_dropout.unwrap_or(0.0) as f32,
                use_rslora: config.use_rslora.unwrap_or(false),
                use_dora: config.use_dora.unwrap_or(false),
                ..Default::default()
            },
            qlora,
            training: pmetal::core::TrainingConfig {
                learning_rate: config.learning_rate.unwrap_or(1e-4),
                batch_size: config.batch_size.unwrap_or(1) as usize,
                num_epochs: config.epochs.unwrap_or(3) as usize,
                max_seq_len: config.max_seq_len.unwrap_or(2048) as usize,
                gradient_accumulation_steps: config.gradient_accumulation_steps.unwrap_or(4)
                    as usize,
                weight_decay: config.weight_decay.unwrap_or(0.01),
                max_grad_norm: config.max_grad_norm.unwrap_or(1.0),
                warmup_steps: config.warmup_steps.unwrap_or(100) as usize,
                logging_steps: config.logging_steps.unwrap_or(10) as usize,
                save_steps: config.save_steps.map(|v| v as usize),
                lr_scheduler: match config.lr_scheduler.as_deref() {
                    Some("constant") => pmetal::core::LrSchedulerType::Constant,
                    Some("linear") => pmetal::core::LrSchedulerType::Linear,
                    Some("cosine_with_restarts") => {
                        pmetal::core::LrSchedulerType::CosineWithRestarts
                    }
                    Some("polynomial") => pmetal::core::LrSchedulerType::Polynomial,
                    Some("wsd") => pmetal::core::LrSchedulerType::Wsd,
                    _ => pmetal::core::LrSchedulerType::Cosine,
                },
                output_dir: output_dir.clone(),
                embedding_learning_rate: config.embedding_lr.map(|v| v as f64),
                ..Default::default()
            },
            columns,
            dispatch: pmetal::trainer::DispatchConfig {
                flash_attention: config.flash_attention.unwrap_or(true),
                sequence_packing: config.sequence_packing.unwrap_or(true),
                jit_compilation: config.jit_compilation.unwrap_or(true),
                fused: true,
                metal_fused_optimizer: config.fused_optimizer.unwrap_or(true),
                gradient_checkpointing: config.gradient_checkpointing.unwrap_or(true),
                gradient_checkpointing_layers: config.gradient_checkpointing_layers.unwrap_or(4)
                    as usize,
                cut_cross_entropy: false,
                ane: config.method == "ane",
                no_adaptive_lr: false,
                loss_scale: 1.0,
                distributed: None,
            },
            config_path: None,
            log_metrics: Some(metrics_path.display().to_string()),
            resume: false,
            seed: 42,
            emit_console_output: false,
        };

        // Wire PhaseCallback to update the GUI run status
        let status_state = state_arc.clone();
        let status_tx = event_tx.clone();
        let status_run_id = run_id_task.clone();
        let phase_cb = move |phase: &pmetal::trainer::TrainingPhase| {
            if let Ok(mut runs) = status_state.try_write() {
                if let Some(run) = runs.iter_mut().find(|r| r.id == status_run_id) {
                    run.status_message = Some(phase.message().to_string());
                    let _ = status_tx.send(AppEvent::TrainingUpdate { run: run.clone() });
                }
            }
        };

        let callbacks: Vec<Box<dyn pmetal::core::TrainingCallback>> = vec![
            Box::new(CancelOnFlag {
                cancelled: cancel_flag.clone(),
            }),
            Box::new(ChannelMetricsCallback {
                channel: on_metrics,
                started_at: Utc::now(),
                best_loss: None,
            }),
        ];

        let result =
            pmetal::trainer::orchestrator::run_training(job_config, Some(&phase_cb), callbacks)
                .await
                .map(|_| ())
                .map_err(|e| AppError(e.to_string()));

        // Clear the status message on success so the UI can show step progress.
        if result.is_ok() {
            let mut runs = state_arc.write().await;
            if let Some(run) = runs.iter_mut().find(|r| r.id == run_id_task) {
                run.status_message = None;
                let _ = event_tx.send(AppEvent::TrainingUpdate { run: run.clone() });
            }
        }

        let success = result.is_ok();
        let error = result.err().map(|e| e.to_string());
        finalize_training_run(
            &state_arc,
            &event_tx,
            &run_id_task,
            &cancel_flag,
            success,
            error,
        )
        .await;

        // Release MLX Metal buffer cache after training ends (success, error, or cancel).
        // Without this, freed buffers stay in MLX's cache indefinitely.
        pmetal::mlx::memory::clear_cache();
        tracing::info!("Training cleanup: MLX cache cleared");

        cancel_flags.write().await.remove(&run_id_task);
    });

    Ok(run_id)
}

#[tauri::command]
pub async fn get_training_status(
    state: State<'_, AppState>,
    run_id: String,
) -> Result<Option<TrainingRun>> {
    Ok(state.get_training_run(&run_id).await)
}

#[tauri::command]
pub async fn list_training_runs(state: State<'_, AppState>) -> Result<Vec<TrainingRun>> {
    Ok(state.list_training_runs().await)
}

#[tauri::command]
pub async fn stop_training(state: State<'_, AppState>, run_id: String) -> Result<()> {
    state.cancel_training_run(&run_id).await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Distillation commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn start_distillation(
    state: State<'_, AppState>,
    _app_handle: AppHandle,
    config: DistillationConfig,
) -> Result<String> {
    let temperature = config.temperature.unwrap_or(2.0) as f64;
    let loss_type = config.loss_type.clone().unwrap_or_else(|| "kl".to_string());
    let total_epochs = config.epochs.unwrap_or(3) as u64;
    let output_dir = config
        .output_dir
        .as_deref()
        .unwrap_or("./output")
        .to_string();
    let metrics_path = PathBuf::from(&output_dir).join("metrics.jsonl");

    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let mut run = DistillationRun::new(
        &config.student_model,
        &config.teacher_model,
        config.dataset.as_deref(),
        &loss_type,
        temperature,
        total_epochs,
        Some(&output_dir),
    );
    run.status = DistillationStatus::Training;
    let run_id = run.id.clone();

    state
        .cancel_flags
        .write()
        .await
        .insert(run_id.clone(), cancel_flag.clone());
    state.create_distillation_run(run).await;

    let run_id_task = run_id.clone();
    let state_arc = state.distillation_runs.clone();
    let event_tx = state.event_tx.clone();
    let cancel_flags = state.cancel_flags.clone();

    tokio::spawn(async move {
        let _ = tokio::fs::create_dir_all(&output_dir).await;
        let _ = tokio::fs::write(&metrics_path, "").await;

        let watcher_state = state_arc.clone();
        let watcher_event_tx = event_tx.clone();
        let watcher_run_id = run_id_task.clone();
        let watcher_metrics = metrics_path.clone();
        let watcher_cancel = cancel_flag.clone();
        tokio::spawn(async move {
            watch_distillation_metrics_file(
                watcher_metrics,
                watcher_run_id,
                watcher_state,
                watcher_event_tx,
                watcher_cancel,
            )
            .await;
        });

        let result = std::panic::AssertUnwindSafe(run_distillation_in_process(
            &config,
            &metrics_path,
            cancel_flag.clone(),
        ))
        .catch_unwind()
        .await;

        let (success, error) = match result {
            Ok(Ok(())) => (true, None),
            Ok(Err(e)) => (false, Some(e.to_string())),
            Err(panic_payload) => {
                let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    format!("Distillation crashed: {s}")
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    format!("Distillation crashed: {s}")
                } else {
                    "Distillation crashed (internal panic)".to_string()
                };
                (false, Some(msg))
            }
        };
        finalize_distillation_run(
            &state_arc,
            &event_tx,
            &run_id_task,
            &cancel_flag,
            success,
            error,
        )
        .await;

        cancel_flags.write().await.remove(&run_id_task);
    });

    Ok(run_id)
}

#[tauri::command]
pub async fn get_distillation_status(
    state: State<'_, AppState>,
    run_id: String,
) -> Result<Option<DistillationRun>> {
    Ok(state.get_distillation_run(&run_id).await)
}

#[tauri::command]
pub async fn list_distillation_runs(state: State<'_, AppState>) -> Result<Vec<DistillationRun>> {
    Ok(state.list_distillation_runs().await)
}

#[tauri::command]
pub async fn stop_distillation(state: State<'_, AppState>, run_id: String) -> Result<()> {
    state.cancel_distillation_run(&run_id).await;
    Ok(())
}

// ---------------------------------------------------------------------------
// GRPO commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn start_grpo(
    state: State<'_, AppState>,
    _app_handle: AppHandle,
    config: GrpoConfig,
) -> Result<String> {
    let group_size = config.group_size.unwrap_or(8);
    let beta = config.beta.unwrap_or(0.04);
    let output_dir = config
        .output_dir
        .as_deref()
        .unwrap_or("./output")
        .to_string();
    let metrics_path = PathBuf::from(&output_dir).join("metrics.jsonl");

    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let mut run = GrpoRun::new(
        &config.model,
        config.dataset.as_deref(),
        group_size,
        beta,
        Some(&output_dir),
    );
    run.status = GrpoStatus::Running;
    let run_id = run.id.clone();

    state
        .cancel_flags
        .write()
        .await
        .insert(run_id.clone(), cancel_flag.clone());
    state.create_grpo_run(run).await;

    let run_id_task = run_id.clone();
    let state_arc = state.grpo_runs.clone();
    let event_tx = state.event_tx.clone();
    let cancel_flags = state.cancel_flags.clone();

    tokio::spawn(async move {
        let _ = tokio::fs::create_dir_all(&output_dir).await;
        let _ = tokio::fs::write(&metrics_path, "").await;

        let watcher_state = state_arc.clone();
        let watcher_event_tx = event_tx.clone();
        let watcher_run_id = run_id_task.clone();
        let watcher_metrics = metrics_path.clone();
        let watcher_cancel = cancel_flag.clone();
        tokio::spawn(async move {
            watch_grpo_metrics_file(
                watcher_metrics,
                watcher_run_id,
                watcher_state,
                watcher_event_tx,
                watcher_cancel,
            )
            .await;
        });

        let result = std::panic::AssertUnwindSafe(run_grpo_in_process(
            &config,
            &metrics_path,
            cancel_flag.clone(),
        ))
        .catch_unwind()
        .await;

        let (success, error) = match result {
            Ok(Ok(())) => (true, None),
            Ok(Err(e)) => (false, Some(e.to_string())),
            Err(panic_payload) => {
                let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    format!("GRPO crashed: {s}")
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    format!("GRPO crashed: {s}")
                } else {
                    "GRPO crashed (internal panic)".to_string()
                };
                (false, Some(msg))
            }
        };
        finalize_grpo_run(
            &state_arc,
            &event_tx,
            &run_id_task,
            &cancel_flag,
            success,
            error,
        )
        .await;

        cancel_flags.write().await.remove(&run_id_task);
    });

    Ok(run_id)
}

#[tauri::command]
pub async fn get_grpo_status(
    state: State<'_, AppState>,
    run_id: String,
) -> Result<Option<GrpoRun>> {
    Ok(state.get_grpo_run(&run_id).await)
}

#[tauri::command]
pub async fn list_grpo_runs(state: State<'_, AppState>) -> Result<Vec<GrpoRun>> {
    Ok(state.list_grpo_runs().await)
}

#[tauri::command]
pub async fn stop_grpo(state: State<'_, AppState>, run_id: String) -> Result<()> {
    state.cancel_grpo_run(&run_id).await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Inference helpers
// ---------------------------------------------------------------------------

/// Run streaming inference — mirrors the logic from the former `easy::infer().generate_streaming()`.
///
/// Run inference using the shared InferenceRunner pipeline.
///
/// This ensures the GUI gets identical behavior to `pmetal infer`:
/// same chat template handling, sampling defaults, stop tokens,
/// LoRA loading, FP8, expert offloading, KV cache quantization, etc.
async fn run_inference_streaming(
    config: &InferenceConfig,
    cancel_flag: &std::sync::atomic::AtomicBool,
    app_handle: &AppHandle,
) -> std::result::Result<serde_json::Value, String> {
    use pmetal::hub::resolve_model_path;
    use pmetal::inference_runner::{InferenceRunner, InferenceRunnerConfig, TurboQuantPreset};

    let kv_turboquant_preset = match config.kv_turboquant_preset.as_deref() {
        Some("q2_5") => Some(TurboQuantPreset::Q2_5),
        Some("q3_5") => Some(TurboQuantPreset::Q3_5),
        Some(other) => {
            return Err(format!("unsupported TurboQuant preset: {other}"));
        }
        None => None,
    };

    let model_path = resolve_model_path(&config.model)
        .await
        .map_err(|e| e.to_string())?;

    let chat_messages = config
        .messages
        .as_ref()
        .map(|messages| {
            messages
                .iter()
                .map(chat_message_from_inference_message)
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .transpose()?;

    let runner_config = InferenceRunnerConfig {
        model_path,
        lora_path: config.lora_path.clone(),
        experts_dir: config.experts_dir.clone(),
        fp8: config.fp8.unwrap_or(false),
        prompt: config.prompt.clone(),
        chat_messages,
        system_message: config.system_message.clone(),
        chat: false, // let the shared runner auto-detect chat-capable models
        no_thinking: config.no_thinking.unwrap_or(false),
        tools: None,
        temperature: config.temperature,
        top_k: config.top_k.map(|k| k as usize),
        top_p: config.top_p,
        min_p: config.min_p,
        max_tokens: config.max_tokens.unwrap_or(1024) as usize,
        repetition_penalty: config.repetition_penalty,
        frequency_penalty: config.frequency_penalty,
        presence_penalty: config.presence_penalty,
        seed: config.seed,
        kv_quant: config.kv_quant,
        kv_k_bits: config.kv_k_bits,
        kv_v_bits: config.kv_v_bits,
        kv_group_size: config.kv_group_size.unwrap_or(64),
        kv_turboquant: config.kv_turboquant.unwrap_or(false),
        kv_turboquant_preset,
        kv_quant_preset: config.kv_quant_preset.clone(),
        no_kv_quant: config.no_kv_quant.unwrap_or(false),
        kv_qjl: config.kv_qjl.unwrap_or(false),
    };

    let prompt_tokens = runner_config.prompt.len(); // rough pre-tokenize hint
    let mut runner = InferenceRunner::prepare(runner_config).map_err(|e| e.to_string())?;

    let mut token_buf: Vec<u32> = Vec::new();
    let mut streamed_text = String::new();
    let start = std::time::Instant::now();
    let mut first_token_time: Option<std::time::Instant> = None;
    let mut generated_tokens: usize = 0;

    // Split borrow: &runner.tokenizer captured by closure, runner.gen borrows the rest.
    let tokenizer = &runner.tokenizer;
    runner
        .state
        .generate_streaming(|token_id| {
            generated_tokens += 1;
            if first_token_time.is_none() {
                first_token_time = Some(std::time::Instant::now());
            }
            token_buf.push(token_id);
            if let Ok(text) = tokenizer.decode(&token_buf) {
                if text.len() > streamed_text.len() {
                    let delta = &text[streamed_text.len()..];
                    if !delta.is_empty() {
                        if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                            return false;
                        }
                        let _ = app_handle.emit("inference-token", delta);
                    }
                }
                streamed_text = text;
            }
            true
        })
        .map_err(|e| e.to_string())?;

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    // TTFT is measured from when generate_streaming starts (post model-load), so it correctly
    // captures only prefill + first-token latency.  decode_ms = total - ttft covers decode only.
    let ttft_ms = first_token_time.map(|t| t.duration_since(start).as_secs_f64() * 1000.0);
    let decode_ms = ttft_ms.map(|ttft| total_ms - ttft);

    // For native paths, prefer the bridge's own decode metrics (measured inside the decode loop,
    // skipping the first 10 steps and excluding stop-token overhead) over wall-clock estimation.
    // Fall back to wall-clock calculation for non-native paths.
    let (tok_per_sec, avg_step_ms, p50_step_ms) =
        if let Some(m) = runner.state.last_decode_metrics {
            (Some(m.tok_per_sec), Some(m.avg_step_ms), Some(m.p50_step_ms))
        } else {
            // Token 1 is the prefill output (counted in TTFT); tokens 2..N are decode steps.
            let tps = if let Some(dm) = decode_ms {
                if dm > 0.0 && generated_tokens > 1 {
                    Some((generated_tokens - 1) as f64 / (dm / 1000.0))
                } else {
                    None
                }
            } else {
                None
            };
            (tps, None, None)
        };

    let parsed_response = pmetal::response_parser::parse_assistant_response(&streamed_text);

    Ok(serde_json::json!({
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "tok_per_sec": tok_per_sec,
        "avg_step_ms": avg_step_ms,
        "p50_step_ms": p50_step_ms,
        "response_text": parsed_response.response,
        "thinking": parsed_response.thinking,
        "truncated_thinking": parsed_response.truncated_thinking,
    }))
}

// ---------------------------------------------------------------------------
// Inference commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn start_inference(
    state: State<'_, AppState>,
    app_handle: AppHandle,
    config: InferenceConfig,
) -> Result<()> {
    let session_id = uuid::Uuid::new_v4().to_string();
    let session_id_task = session_id.clone();
    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let inference_flags = state.inference_cancel_flags.clone();
    state
        .inference_cancel_flags
        .write()
        .await
        .insert(session_id.clone(), cancel_flag.clone());

    tokio::spawn(async move {
        let result = run_inference_streaming(&config, &cancel_flag, &app_handle).await;

        match result {
            Ok(metrics) => {
                let _ = app_handle.emit("inference-done", metrics);
            }
            Err(e) => {
                let _ = app_handle.emit(
                    "inference-error",
                    serde_json::json!({ "session_id": session_id_task, "error": e }),
                );
            }
        }

        inference_flags.write().await.remove(&session_id_task);
    });

    Ok(())
}

#[tauri::command]
pub async fn stop_inference(state: State<'_, AppState>, session_id: Option<String>) -> Result<()> {
    if let Some(id) = session_id {
        let flags = state.inference_cancel_flags.read().await;
        if let Some(flag) = flags.get(&id) {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    } else {
        let flags = state.inference_cancel_flags.read().await;
        for flag in flags.values() {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Merge / Fuse / Quantize commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_merge_strategies() -> Result<Vec<MergeStrategy>> {
    Ok(vec![
        MergeStrategy {
            name: "linear".to_string(),
            description: "Simple weighted average of model weights".to_string(),
            supports_weights: true,
        },
        MergeStrategy {
            name: "slerp".to_string(),
            description: "Spherical linear interpolation for smooth weight blending".to_string(),
            supports_weights: true,
        },
        MergeStrategy {
            name: "ties".to_string(),
            description: "Trim, elect sign, merge — conflict-free parameter merging".to_string(),
            supports_weights: true,
        },
        MergeStrategy {
            name: "dare".to_string(),
            description: "Delta weight pruning before merge to reduce interference".to_string(),
            supports_weights: true,
        },
        MergeStrategy {
            name: "model_stock".to_string(),
            description: "Geometric mean based merging using model weights as anchors".to_string(),
            supports_weights: false,
        },
    ])
}

#[tauri::command]
pub async fn merge_models(_app_handle: AppHandle, config: MergeConfig) -> Result<String> {
    let merge_method = match config.strategy.as_str() {
        "linear" => pmetal::merge::MergeMethodConfig::Linear,
        "slerp" => pmetal::merge::MergeMethodConfig::Slerp,
        "ties" => pmetal::merge::MergeMethodConfig::Ties,
        "dare" => pmetal::merge::MergeMethodConfig::DareTies,
        "model_stock" => pmetal::merge::MergeMethodConfig::ModelStock,
        other => return Err(AppError(format!("Unsupported merge strategy: {other}"))),
    };

    let merge_config = pmetal::merge::MergeConfig {
        merge_method,
        base_model: Some(config.base_model.clone()),
        models: config
            .models
            .into_iter()
            .map(|model| pmetal::merge::ModelConfig {
                model: model.model,
                parameters: pmetal::merge::MergeParameters {
                    weight: Some(pmetal::merge::ParameterSetting::Scalar(model.weight as f32)),
                    ..Default::default()
                },
            })
            .collect(),
        output_path: Some(PathBuf::from(&config.output)),
        parameters: pmetal::merge::MergeParameters {
            normalize: Some(true),
            ..Default::default()
        },
        ..Default::default()
    };

    let output = tokio::task::spawn_blocking(move || pmetal::merge::run_merge(&merge_config))
        .await
        .map_err(|e| AppError(format!("merge task failed: {e}")))?
        .map_err(|e| AppError(e.to_string()))?;

    Ok(output.display().to_string())
}

#[tauri::command]
pub async fn fuse_lora(
    _app_handle: AppHandle,
    base_model: String,
    lora_path: String,
    output_dir: String,
) -> Result<FuseResult> {
    // Resolve remote model IDs (e.g. "Qwen/Qwen3-0.6B-Base") before the
    // blocking fuse step, since `run_fuse_in_process` cannot be async.
    let resolved_base = resolve_model_path(&base_model).await?;
    let base_model_task = resolved_base.to_string_lossy().into_owned();
    let lora_path_task = lora_path.clone();
    let output_dir_task = output_dir.clone();

    tokio::task::spawn_blocking(move || {
        run_fuse_in_process(&base_model_task, &lora_path_task, &output_dir_task)
    })
    .await
    .map_err(|e| AppError(format!("fuse task failed: {e}")))?
    .map_err(|e| AppError(e.to_string()))?;

    let output_path = PathBuf::from(&output_dir);
    let model_size_bytes = dir_size_simple(&output_path).await;

    Ok(FuseResult {
        output_dir,
        model_size_bytes,
    })
}

/// Scan for trained LoRA adapters on disk.
///
/// Checks `~/pmetal-output/` (GUI default) for directories containing
/// `adapter_config.json` or `lora_weights.safetensors`.
#[tauri::command]
pub async fn list_trained_adapters() -> Result<Vec<TrainedAdapter>> {
    tokio::task::spawn_blocking(scan_trained_adapters)
        .await
        .map_err(|e| AppError(format!("scan failed: {e}")))
}

fn scan_trained_adapters() -> Vec<TrainedAdapter> {
    let mut adapters = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Scan roots
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(home) = dirs::home_dir() {
        roots.push(home.join("pmetal-output"));
    }

    for root in &roots {
        if !root.is_dir() {
            continue;
        }
        // Scan up to 2 levels deep
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    try_add_adapter(&path, &mut adapters, &mut seen);
                    // One more level
                    if let Ok(sub_entries) = std::fs::read_dir(&path) {
                        for sub in sub_entries.flatten() {
                            let sub_path = sub.path();
                            if sub_path.is_dir() {
                                try_add_adapter(&sub_path, &mut adapters, &mut seen);
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort by modification time (newest first)
    adapters.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    adapters
}

fn try_add_adapter(
    dir: &Path,
    adapters: &mut Vec<TrainedAdapter>,
    seen: &mut std::collections::HashSet<PathBuf>,
) {
    let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    if !seen.insert(canonical) {
        return;
    }

    let has_config = dir.join("adapter_config.json").exists();
    let weights_file = if dir.join("lora_weights.safetensors").exists() {
        Some(dir.join("lora_weights.safetensors"))
    } else if dir.join("adapter_model.safetensors").exists() {
        Some(dir.join("adapter_model.safetensors"))
    } else {
        None
    };

    if !has_config && weights_file.is_none() {
        return;
    }

    let dir_name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| dir.display().to_string());

    let size_bytes = weights_file
        .as_ref()
        .and_then(|f| f.metadata().ok())
        .map(|m| m.len())
        .unwrap_or(0);

    // Read adapter_config.json for rank/alpha
    let (rank, alpha, base_model) = if has_config {
        let cfg = std::fs::read_to_string(dir.join("adapter_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        let r = cfg.as_ref().and_then(|c| c["r"].as_u64()).map(|v| v as u32);
        let a = cfg
            .as_ref()
            .and_then(|c| c["alpha"].as_f64().or_else(|| c["lora_alpha"].as_f64()))
            .map(|v| v as f32);
        let bm = cfg
            .as_ref()
            .and_then(|c| c["base_model"].as_str().map(String::from));
        (r, a, bm)
    } else {
        (None, None, None)
    };

    // Also check training_info.json for base_model + dataset (written by TUI/GUI)
    let training_info = std::fs::read_to_string(dir.join("training_info.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
    let base_model = base_model.or_else(|| {
        training_info
            .as_ref()
            .and_then(|c| c["base_model"].as_str().map(String::from))
    });
    let dataset = training_info
        .as_ref()
        .and_then(|c| c["dataset"].as_str().map(String::from));

    // Build a descriptive display name like TUI:
    // - "Qwen3-0.6B-Base + my-dataset — dir_name"
    // - "Qwen3-0.6B-Base — dir_name"
    // - Just dir_name if no metadata
    let display_name = match (&base_model, &dataset) {
        (Some(bm), Some(ds)) => {
            let model_short = bm.rsplit('/').next().unwrap_or(bm);
            let ds_short = ds.rsplit('/').next().unwrap_or(ds);
            format!("{model_short} + {ds_short}")
        }
        (Some(bm), None) => {
            let model_short = bm.rsplit('/').next().unwrap_or(bm);
            format!("{model_short} — {dir_name}")
        }
        _ => dir_name,
    };

    adapters.push(TrainedAdapter {
        path: dir.display().to_string(),
        name: display_name,
        base_model,
        rank,
        alpha,
        size_bytes,
    });
}

#[tauri::command]
pub async fn quantize_model(
    _app_handle: AppHandle,
    model_id: String,
    quant_type: String,
    output_dir: String,
) -> Result<String> {
    // Resolve remote model IDs (e.g. "Qwen/Qwen3-0.6B-Base") to local cache paths
    let resolved_path = resolve_model_path(&model_id).await?;
    let model_path = resolved_path.to_string_lossy().into_owned();
    let quant_type_task = quant_type.clone();
    let output_dir_task = output_dir.clone();

    tokio::task::spawn_blocking(move || {
        run_quantize_in_process(&model_path, &quant_type_task, &output_dir_task)
    })
    .await
    .map_err(|e| AppError(format!("quantize task failed: {e}")))?
    .map_err(|e| AppError(e.to_string()))?;

    Ok(output_dir)
}

async fn finalize_training_run(
    state_arc: &Arc<tokio::sync::RwLock<Vec<TrainingRun>>>,
    event_tx: &tokio::sync::broadcast::Sender<AppEvent>,
    run_id: &str,
    cancel_flag: &Arc<std::sync::atomic::AtomicBool>,
    success: bool,
    error: Option<String>,
) {
    // Signal the metrics watcher to exit before updating state.
    cancel_flag.store(true, std::sync::atomic::Ordering::SeqCst);

    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        // Preserve Cancelled if already set by cancel_training_run(); only
        // override status when the run ended naturally (success or failure).
        if run.status != TrainingStatus::Cancelled {
            run.status = if success {
                TrainingStatus::Completed
            } else {
                TrainingStatus::Failed
            };
        }
        run.ended_at = Some(Utc::now());
        run.error_message = error;
        run.status_message = None;
        let _ = event_tx.send(AppEvent::TrainingUpdate { run: run.clone() });
        let _ = event_tx.send(AppEvent::TrainingStopped {
            run_id: run_id.to_string(),
        });
    }
}

async fn watch_distillation_metrics_file(
    metrics_path: PathBuf,
    run_id: String,
    state_arc: Arc<tokio::sync::RwLock<Vec<DistillationRun>>>,
    event_tx: tokio::sync::broadcast::Sender<AppEvent>,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
) {
    let mut last_pos: u64 = 0;
    let started_at = Utc::now();
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));

    loop {
        interval.tick().await;

        if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }

        let Ok(file) = tokio::fs::File::open(&metrics_path).await else {
            continue;
        };
        let Ok(metadata) = file.metadata().await else {
            continue;
        };
        let file_len = metadata.len();
        if file_len < last_pos {
            // File was truncated (e.g., ANE→GPU fallback recreates the metrics file).
            last_pos = 0;
        }
        if file_len <= last_pos {
            continue;
        }

        let path = metrics_path.clone();
        let read_result = tokio::task::spawn_blocking(move || {
            use std::io::{BufRead, Seek};

            let Ok(file) = std::fs::File::open(&path) else {
                return (last_pos, Vec::<serde_json::Value>::new());
            };
            let mut reader = std::io::BufReader::new(file);
            if reader.seek(std::io::SeekFrom::Start(last_pos)).is_err() {
                return (last_pos, Vec::new());
            }

            let mut line = String::new();
            let mut rows = Vec::new();
            while reader.read_line(&mut line).unwrap_or(0) > 0 {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                    rows.push(json);
                }
                line.clear();
            }

            let pos = reader.stream_position().unwrap_or(last_pos);
            (pos, rows)
        })
        .await;

        let Ok((new_pos, rows)) = read_result else {
            continue;
        };
        last_pos = new_pos;

        if rows.is_empty() {
            continue;
        }

        let mut runs = state_arc.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
            for row in rows {
                apply_metrics_to_distillation(run, &row, started_at);
            }
            let _ = event_tx.send(AppEvent::DistillationUpdate { run: run.clone() });
        }
    }
}

fn apply_metrics_to_distillation(
    run: &mut DistillationRun,
    metrics: &serde_json::Value,
    started_at: chrono::DateTime<Utc>,
) {
    if let Some(v) = metrics["step"].as_u64() {
        run.step = v;
    }
    if let Some(v) = metrics["total_steps"].as_u64() {
        run.total_steps = Some(v);
    }
    if let Some(v) = metrics["epoch"].as_u64() {
        run.epoch = v;
    }
    if let Some(v) = metrics["loss"].as_f64() {
        run.loss = Some(v);
        if run.best_loss.map_or(true, |best| v < best) {
            run.best_loss = Some(v);
        }
    }
    if let Some(v) = metrics["lr"]
        .as_f64()
        .or_else(|| metrics["learning_rate"].as_f64())
    {
        run.learning_rate = Some(v);
    }
    if let Some(v) = metrics["tok_sec"]
        .as_f64()
        .or_else(|| metrics["tokens_per_second"].as_f64())
    {
        run.tokens_per_second = Some(v);
    }
    if let (Some(total_steps), step) = (run.total_steps, run.step) {
        if step > 0 {
            let elapsed = (Utc::now() - started_at).num_seconds().max(1) as f64;
            let remaining = total_steps.saturating_sub(step) as f64;
            run.eta_seconds = Some(((elapsed / step as f64) * remaining) as u64);
        }
    }
}

async fn finalize_distillation_run(
    state_arc: &Arc<tokio::sync::RwLock<Vec<DistillationRun>>>,
    event_tx: &tokio::sync::broadcast::Sender<AppEvent>,
    run_id: &str,
    cancel_flag: &Arc<std::sync::atomic::AtomicBool>,
    success: bool,
    error: Option<String>,
) {
    // Signal the metrics watcher to exit before updating state.
    cancel_flag.store(true, std::sync::atomic::Ordering::SeqCst);

    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        if run.status != DistillationStatus::Cancelled {
            run.status = if success {
                DistillationStatus::Completed
            } else {
                DistillationStatus::Failed
            };
        }
        run.ended_at = Some(Utc::now());
        run.error_message = error;
        let _ = event_tx.send(AppEvent::DistillationUpdate { run: run.clone() });
        let _ = event_tx.send(AppEvent::DistillationStopped {
            run_id: run_id.to_string(),
        });
    }
}

async fn watch_grpo_metrics_file(
    metrics_path: PathBuf,
    run_id: String,
    state_arc: Arc<tokio::sync::RwLock<Vec<GrpoRun>>>,
    event_tx: tokio::sync::broadcast::Sender<AppEvent>,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
) {
    let mut last_pos: u64 = 0;
    let started_at = Utc::now();
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));

    loop {
        interval.tick().await;

        if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }

        let Ok(file) = tokio::fs::File::open(&metrics_path).await else {
            continue;
        };
        let Ok(metadata) = file.metadata().await else {
            continue;
        };
        let file_len = metadata.len();
        if file_len < last_pos {
            // File was truncated (e.g., ANE→GPU fallback recreates the metrics file).
            last_pos = 0;
        }
        if file_len <= last_pos {
            continue;
        }

        let path = metrics_path.clone();
        let read_result = tokio::task::spawn_blocking(move || {
            use std::io::{BufRead, Seek};

            let Ok(file) = std::fs::File::open(&path) else {
                return (last_pos, Vec::<serde_json::Value>::new());
            };
            let mut reader = std::io::BufReader::new(file);
            if reader.seek(std::io::SeekFrom::Start(last_pos)).is_err() {
                return (last_pos, Vec::new());
            }

            let mut line = String::new();
            let mut rows = Vec::new();
            while reader.read_line(&mut line).unwrap_or(0) > 0 {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                    rows.push(json);
                }
                line.clear();
            }

            let pos = reader.stream_position().unwrap_or(last_pos);
            (pos, rows)
        })
        .await;

        let Ok((new_pos, rows)) = read_result else {
            continue;
        };
        last_pos = new_pos;

        if rows.is_empty() {
            continue;
        }

        let mut runs = state_arc.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
            for row in rows {
                apply_metrics_to_grpo(run, &row, started_at);
            }
            let _ = event_tx.send(AppEvent::GrpoUpdate { run: run.clone() });
        }
    }
}

fn apply_metrics_to_grpo(
    run: &mut GrpoRun,
    metrics: &serde_json::Value,
    started_at: chrono::DateTime<Utc>,
) {
    if let Some(v) = metrics["step"].as_u64() {
        run.step = v;
    }
    if let Some(v) = metrics["total_steps"].as_u64() {
        run.total_steps = Some(v);
    }
    if let Some(v) = metrics["loss"].as_f64() {
        run.loss = Some(v);
        if run.best_loss.map_or(true, |best| v < best) {
            run.best_loss = Some(v);
        }
    }
    if let Some(v) = metrics["reward_mean"].as_f64() {
        run.reward_mean = Some(v);
    }
    if let Some(v) = metrics["reward_std"].as_f64() {
        run.reward_std = Some(v);
    }
    if let Some(v) = metrics["kl_div"].as_f64() {
        run.kl_div = Some(v);
    }
    if let Some(v) = metrics["lr"]
        .as_f64()
        .or_else(|| metrics["learning_rate"].as_f64())
    {
        run.learning_rate = Some(v);
    }
    if let Some(v) = metrics["tok_sec"]
        .as_f64()
        .or_else(|| metrics["tokens_per_second"].as_f64())
    {
        run.tokens_per_second = Some(v);
    }
    if let (Some(total_steps), step) = (run.total_steps, run.step) {
        if step > 0 {
            let elapsed = (Utc::now() - started_at).num_seconds().max(1) as f64;
            let remaining = total_steps.saturating_sub(step) as f64;
            run.eta_seconds = Some(((elapsed / step as f64) * remaining) as u64);
        }
    }
}

async fn finalize_grpo_run(
    state_arc: &Arc<tokio::sync::RwLock<Vec<GrpoRun>>>,
    event_tx: &tokio::sync::broadcast::Sender<AppEvent>,
    run_id: &str,
    cancel_flag: &Arc<std::sync::atomic::AtomicBool>,
    success: bool,
    error: Option<String>,
) {
    // Signal the metrics watcher to exit before updating state.
    cancel_flag.store(true, std::sync::atomic::Ordering::SeqCst);

    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        if run.status != GrpoStatus::Cancelled {
            run.status = if success {
                GrpoStatus::Completed
            } else {
                GrpoStatus::Failed
            };
        }
        run.ended_at = Some(Utc::now());
        run.error_message = error;
        let _ = event_tx.send(AppEvent::GrpoUpdate { run: run.clone() });
        let _ = event_tx.send(AppEvent::GrpoStopped {
            run_id: run_id.to_string(),
        });
    }
}

/// Scan a directory (up to 3 levels deep) for .parquet files.
fn find_parquet_in_dir(dir: &Path) -> Option<PathBuf> {
    fn scan(dir: &Path, depth: usize) -> Option<PathBuf> {
        if depth > 3 {
            return None;
        }
        let entries = std::fs::read_dir(dir).ok()?;
        let mut dirs = Vec::new();
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().is_some_and(|e| e == "parquet") {
                return Some(p);
            }
            if p.is_dir() {
                dirs.push(p);
            }
        }
        for d in dirs {
            if let Some(p) = scan(&d, depth + 1) {
                return Some(p);
            }
        }
        None
    }
    scan(dir, 0)
}

async fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    if model_id.contains('/') && !PathBuf::from(model_id).exists() {
        pmetal::hub::download_model(model_id, None, None)
            .await
            .map_err(|e| AppError(e.to_string()))
    } else {
        Ok(PathBuf::from(model_id))
    }
}

async fn resolve_dataset_path(dataset_id: &str) -> Result<PathBuf> {
    match pmetal::data::resolve_dataset_source(dataset_id) {
        pmetal::data::DatasetSource::Local(path) => {
            // Resolve directories to a data file within (handles HF cache structure)
            pmetal::data::TrainingDataset::resolve_dataset_path_pub(&path)
                .map_err(|e| AppError(e.to_string()))
        }
        pmetal::data::DatasetSource::HuggingFace(id) => {
            let dir = pmetal::hub::download_dataset(&id, None, None, None)
                .await
                .map_err(|e| AppError(e.to_string()))?;
            if let Ok(path) = pmetal::data::TrainingDataset::resolve_dataset_path_pub(&dir) {
                return Ok(path);
            }
            // Scan cached directory for parquet files (no network call)
            if let Some(pf) = find_parquet_in_dir(&dir) {
                return Ok(pf);
            }
            // Last resort: HF API (slow, network call)
            let parquet_paths = pmetal::hub::download_dataset_parquet(&id, "train", None, None)
                .await
                .map_err(|e| AppError(e.to_string()))?;
            parquet_paths
                .into_iter()
                .next()
                .ok_or_else(|| AppError(format!("No dataset files found for {id}")))
        }
    }
}

async fn run_distillation_in_process(
    config: &DistillationConfig,
    metrics_path: &PathBuf,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    use pmetal::lora::TrainableModel;

    let teacher_path = resolve_model_path(&config.teacher_model).await?;
    let student_path = resolve_model_path(&config.student_model).await?;
    let dataset_id = config
        .dataset
        .as_deref()
        .ok_or_else(|| AppError("Dataset is required for distillation".to_string()))?;
    let dataset_path = resolve_dataset_path(dataset_id).await?;

    let tokenizer = pmetal::data::Tokenizer::from_model_dir(&student_path)
        .map_err(|e| AppError(e.to_string()))?;
    let chat_template =
        pmetal::data::chat_templates::detect_chat_template(&student_path, &config.student_model);

    let max_seq_len = config.max_seq_len.unwrap_or(1024) as usize;

    // Build column config from the `+`-separated multi-column string the GUI sends.
    let col_cfg = config.text_column.as_deref().map(|tc| {
        if tc.contains('+') {
            pmetal::data::DatasetColumnConfig {
                text_columns: Some(tc.split('+').map(str::to_string).collect()),
                ..Default::default()
            }
        } else {
            pmetal::data::DatasetColumnConfig {
                text_column: Some(tc.to_string()),
                ..Default::default()
            }
        }
    });

    let train_dataset = if dataset_path.extension().is_some_and(|ext| ext == "parquet") {
        pmetal::data::TrainingDataset::from_parquet_tokenized(
            &dataset_path,
            &tokenizer,
            "text",
            max_seq_len,
            None,
        )
        .or_else(|_| {
            pmetal::data::TrainingDataset::from_parquet_tokenized(
                &dataset_path,
                &tokenizer,
                "content",
                max_seq_len,
                None,
            )
        })
        .map_err(|e| AppError(e.to_string()))?
    } else {
        pmetal::data::TrainingDataset::from_jsonl_tokenized(
            &dataset_path,
            &tokenizer,
            pmetal::data::DatasetFormat::Auto,
            max_seq_len,
            Some(&chat_template),
            col_cfg.as_ref(),
        )
        .map_err(|e| AppError(e.to_string()))?
    };

    let teacher_lora_config = pmetal::core::LoraConfig {
        r: 0,
        ..Default::default()
    };
    let mut teacher_model =
        pmetal::lora::DynamicLoraModel::from_pretrained(&teacher_path, teacher_lora_config)
            .map_err(|e| AppError(e.to_string()))?;

    let student_lora_config = pmetal::core::LoraConfig {
        r: config.lora_rank.unwrap_or(16) as usize,
        alpha: config.lora_alpha.unwrap_or(32) as f32,
        ..Default::default()
    };
    let mut student_model =
        pmetal::lora::DynamicLoraModel::from_pretrained(&student_path, student_lora_config.clone())
            .map_err(|e| AppError(e.to_string()))?;

    let loss_type = match config
        .loss_type
        .clone()
        .unwrap_or_else(|| "kl_divergence".to_string())
        .to_lowercase()
        .as_str()
    {
        "kl" | "kl_divergence" => pmetal::distill::LossType::KlDivergence,
        "js" | "jensen_shannon" => pmetal::distill::LossType::JensenShannon,
        "soft_cross_entropy" => pmetal::distill::LossType::SoftCrossEntropy,
        "mse" | "mse_loss" => pmetal::distill::LossType::MseLoss,
        other => {
            return Err(AppError(format!(
                "Unsupported distillation loss type: {other}"
            )));
        }
    };

    let distill_config = pmetal::distill::DistillConfig {
        teacher: config.teacher_model.clone(),
        student: config.student_model.clone(),
        method: pmetal::distill::DistillMethod::Online,
        loss: pmetal::distill::LossConfig {
            loss_type,
            temperature: config.temperature.unwrap_or(2.0),
            alpha: config.alpha.unwrap_or(0.5),
            ..Default::default()
        },
        offline: None,
        output_path: config.output_dir.as_ref().map(PathBuf::from),
        training: pmetal::distill::TrainingConfig {
            batch_size: config.batch_size.unwrap_or(1) as usize,
            learning_rate: config.learning_rate.unwrap_or(2e-5) as f32,
            epochs: config.epochs.unwrap_or(3) as usize,
            max_seq_len,
            ..Default::default()
        },
    };

    let distiller =
        pmetal::distill::Distiller::new(distill_config).map_err(|e| AppError(e.to_string()))?;

    let training_loop_config = pmetal::trainer::TrainingLoopConfig {
        training: pmetal::core::TrainingConfig {
            learning_rate: config.learning_rate.unwrap_or(2e-5),
            batch_size: config.batch_size.unwrap_or(1) as usize,
            num_epochs: config.epochs.unwrap_or(3) as usize,
            max_seq_len,
            output_dir: config
                .output_dir
                .clone()
                .unwrap_or_else(|| "./output".to_string()),
            ..Default::default()
        },
        dataloader: pmetal::data::DataLoaderConfig {
            batch_size: config.batch_size.unwrap_or(1) as usize,
            max_seq_len,
            shuffle: true,
            seed: 42,
            pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
            drop_last: false,
            ..Default::default()
        },
        use_metal_flash_attention: true,
        log_every: 1,
        checkpoint_every: 100,
        eval_every: 0,
        use_jit_compilation: true,
        use_sequence_packing: true,
        gradient_checkpointing: true,
        gradient_checkpointing_layers: 4,
        embedding_lr: None,
        eager_evaluation: false,
        use_metal_fused_optimizer: true,
        loraplus_lr_ratio: None,
        neftune_noise_alpha: None,
        ..Default::default()
    };

    let mut trainer = pmetal::trainer::DistillationTrainer::new(distiller, training_loop_config);
    let adaptive_config = pmetal::trainer::AdaptiveLrConfig::for_distillation();
    let control_file = PathBuf::from(
        config
            .output_dir
            .clone()
            .unwrap_or_else(|| "./output".to_string()),
    )
    .join(".lr_control.json");
    trainer.enable_adaptive_lr_with_control(adaptive_config, control_file);

    let callback = pmetal::trainer::MetricsJsonCallback::new(metrics_path)
        .map_err(|e| AppError(e.to_string()))?
        .with_run_name(format!(
            "distill-{}",
            config.student_model.replace('/', "-")
        ));
    trainer.add_callback(Box::new(callback));
    trainer.add_callback(Box::new(CancelOnFlag {
        cancelled: cancel_flag,
    }));

    trainer
        .run(
            &mut student_model,
            &mut teacher_model,
            train_dataset,
            None,
            None,
        )
        .map_err(|e| AppError(e.to_string()))?;

    let output_dir = PathBuf::from(
        config
            .output_dir
            .clone()
            .unwrap_or_else(|| "./output".to_string()),
    );
    std::fs::create_dir_all(&output_dir).map_err(|e| AppError(e.to_string()))?;
    let lora_output = output_dir.join("lora_weights.safetensors");
    student_model
        .save_lora_weights(&lora_output)
        .map_err(|e| AppError(e.to_string()))?;
    let adapter_config = serde_json::json!({
        "r": student_lora_config.r,
        "alpha": student_lora_config.alpha,
        "target_modules": student_lora_config.target_modules,
        "use_rslora": student_lora_config.use_rslora,
        "base_model": config.student_model,
    });
    std::fs::write(
        output_dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&adapter_config).map_err(|e| AppError(e.to_string()))?,
    )
    .map_err(|e| AppError(e.to_string()))?;

    Ok(())
}

async fn run_grpo_in_process(
    config: &GrpoConfig,
    metrics_path: &PathBuf,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    use pmetal::lora::TrainableModel;

    let model_path = resolve_model_path(&config.model).await?;
    let dataset_id = config
        .dataset
        .as_deref()
        .ok_or_else(|| AppError("Dataset is required for GRPO".to_string()))?;
    let dataset_path = resolve_dataset_path(dataset_id).await?;
    let tokenizer = pmetal::data::Tokenizer::from_model_dir(&model_path)
        .map_err(|e| AppError(e.to_string()))?;
    let chat_template =
        pmetal::data::chat_templates::detect_chat_template(&model_path, &config.model);

    let max_seq_len = config.max_seq_len.unwrap_or(512) as usize;

    // Build column config from the `+`-separated multi-column string the GUI sends.
    let col_cfg = config.text_column.as_deref().map(|tc| {
        if tc.contains('+') {
            pmetal::data::DatasetColumnConfig {
                text_columns: Some(tc.split('+').map(str::to_string).collect()),
                ..Default::default()
            }
        } else {
            pmetal::data::DatasetColumnConfig {
                text_column: Some(tc.to_string()),
                ..Default::default()
            }
        }
    });

    let dataset = if dataset_path.extension().is_some_and(|ext| ext == "parquet") {
        pmetal::data::TrainingDataset::from_parquet_tokenized(
            &dataset_path,
            &tokenizer,
            "text",
            max_seq_len,
            None,
        )
        .or_else(|_| {
            pmetal::data::TrainingDataset::from_parquet_tokenized(
                &dataset_path,
                &tokenizer,
                "content",
                max_seq_len,
                None,
            )
        })
        .map_err(|e| AppError(e.to_string()))?
    } else {
        pmetal::data::TrainingDataset::from_jsonl_tokenized(
            &dataset_path,
            &tokenizer,
            pmetal::data::DatasetFormat::Auto,
            max_seq_len,
            Some(&chat_template),
            col_cfg.as_ref(),
        )
        .map_err(|e| AppError(e.to_string()))?
    };

    let lora_config = pmetal::core::LoraConfig {
        r: config.lora_rank.unwrap_or(16) as usize,
        alpha: config.lora_alpha.unwrap_or(32) as f32,
        ..Default::default()
    };
    let mut model =
        pmetal::lora::DynamicLoraModel::from_pretrained(&model_path, lora_config.clone())
            .map_err(|e| AppError(e.to_string()))?;

    let mut grpo_config = pmetal::trainer::GrpoConfig::new(config.group_size.unwrap_or(8) as usize)
        .with_beta(config.beta.unwrap_or(0.04));
    grpo_config.max_prompt_length = max_seq_len;
    grpo_config.max_completion_length = 512;
    grpo_config.kv_cache_bits = config.kv_cache_bits;

    let mut rewards = pmetal::trainer::CombinedReward::new();
    if config.use_reasoning_rewards.unwrap_or(false) {
        rewards = rewards.add(
            Box::new(pmetal::trainer::XmlFormatReward::default_reasoning()),
            1.0,
        );
    } else {
        return Err(AppError(
            "GRPO requires a reward function. Enable 'Use Reasoning Rewards' or use the CLI \
             with a custom reward configuration."
                .to_string(),
        ));
    }

    let output_dir = config
        .output_dir
        .clone()
        .unwrap_or_else(|| "./output".to_string());
    let training_config = pmetal::core::TrainingConfig {
        learning_rate: config.learning_rate.unwrap_or(5e-6),
        batch_size: 1,
        num_epochs: config.epochs.unwrap_or(1) as usize,
        max_seq_len,
        output_dir: output_dir.clone(),
        ..Default::default()
    };

    let mut trainer = pmetal::trainer::GrpoTrainer::new(grpo_config, training_config)
        .map_err(|e| AppError(e.to_string()))?;
    let callback = pmetal::trainer::MetricsJsonCallback::new(metrics_path)
        .map_err(|e| AppError(e.to_string()))?
        .with_run_name(format!("grpo-{}", config.model.replace('/', "-")));
    trainer.add_callback(Box::new(callback));
    trainer.add_callback(Box::new(CancelOnFlag {
        cancelled: cancel_flag,
    }));

    let adaptive_config = pmetal::trainer::AdaptiveLrConfig::default();
    let control_file = PathBuf::from(&output_dir).join(".lr_control.json");
    trainer.enable_adaptive_lr_with_control(adaptive_config, control_file);

    let mut optimizer = pmetal_bridge::compat::optimizers::AdamWBuilder::new(
        config.learning_rate.unwrap_or(5e-6) as f32,
    )
    .build()
    .map_err(|e| AppError(e.to_string()))?;
    let mut ref_model =
        pmetal::models::DynamicModel::load(&model_path).map_err(|e| AppError(e.to_string()))?;

    trainer
        .run(
            &mut model,
            Some(&mut ref_model),
            &tokenizer,
            &dataset,
            &rewards,
            &mut optimizer,
            |opt, lr| {
                opt.lr = pmetal_bridge::array!(lr);
            },
        )
        .map_err(|e| AppError(e.to_string()))?;

    let output_dir = PathBuf::from(output_dir);
    std::fs::create_dir_all(&output_dir).map_err(|e| AppError(e.to_string()))?;
    let lora_output = output_dir.join("lora_weights.safetensors");
    model
        .save_lora_weights(&lora_output)
        .map_err(|e| AppError(e.to_string()))?;
    let adapter_config = serde_json::json!({
        "r": lora_config.r,
        "alpha": lora_config.alpha,
        "target_modules": lora_config.target_modules,
        "use_rslora": lora_config.use_rslora,
        "base_model": config.model,
    });
    std::fs::write(
        output_dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&adapter_config).map_err(|e| AppError(e.to_string()))?,
    )
    .map_err(|e| AppError(e.to_string()))?;

    Ok(())
}

fn run_fuse_in_process(
    model_path: &str,
    lora_path: &str,
    output_path: &str,
) -> std::result::Result<(), AppError> {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    // The caller (`fuse_lora`) has already resolved remote model IDs to local
    // cache paths, so model_path is always a local directory at this point.
    let model_dir: PathBuf = PathBuf::from(model_path);

    let lora_file = if Path::new(lora_path).is_dir() {
        let f = Path::new(lora_path).join("lora_weights.safetensors");
        if !f.exists() {
            return Err(AppError(format!(
                "No lora_weights.safetensors found in {lora_path}"
            )));
        }
        f
    } else {
        PathBuf::from(lora_path)
    };

    let mut base_weights =
        pmetal::models::loader::load_weights(&model_dir).map_err(|e| AppError(e.to_string()))?;
    let lora_file_str = lora_file
        .to_str()
        .ok_or_else(|| AppError(format!("non-UTF-8 LoRA path: {}", lora_file.display())))?;
    let lora_weights: HashMap<String, pmetal::mlx::Array> =
        pmetal_bridge::inline_array::load_safetensors_shard(lora_file_str)
            .map(|pairs| pairs.into_iter().collect())
            .ok_or_else(|| {
                AppError(format!(
                    "failed to load safetensors: {}",
                    lora_file.display()
                ))
            })?;

    let lora_dir = if Path::new(lora_path).is_dir() {
        PathBuf::from(lora_path)
    } else {
        Path::new(lora_path)
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf()
    };
    let adapter_config = std::fs::read_to_string(lora_dir.join("adapter_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());

    let rank = adapter_config
        .as_ref()
        .and_then(|cfg| cfg["r"].as_u64())
        .map(|r| r as usize)
        .or_else(|| {
            lora_weights
                .iter()
                .filter(|(k, _)| k.contains("self_attn") && k.contains("lora_a"))
                .map(|(_, v)| *v.shape().iter().min().unwrap_or(&16) as usize)
                .next()
        })
        .unwrap_or(16);
    let alpha = adapter_config
        .as_ref()
        .and_then(|cfg| cfg["alpha"].as_f64())
        .map(|a| a as f32)
        .unwrap_or(rank as f32);
    let scale = alpha / rank as f32;

    let mut lora_a_map: HashMap<String, &pmetal::mlx::Array> = HashMap::new();
    let mut lora_b_map: HashMap<String, &pmetal::mlx::Array> = HashMap::new();
    for (name, array) in &lora_weights {
        if let Some(base_name) = name.strip_suffix(".lora_a") {
            lora_a_map.insert(base_name.to_string(), array);
        } else if let Some(base_name) = name.strip_suffix(".lora_b") {
            lora_b_map.insert(base_name.to_string(), array);
        }
    }

    for (layer_name, &lora_a) in &lora_a_map {
        let Some(&lora_b) = lora_b_map.get(layer_name) else {
            continue;
        };
        let base_key = if layer_name.starts_with("model.") {
            format!("{layer_name}.weight")
        } else {
            format!("model.{layer_name}.weight")
        };
        let Some(base_weight) = base_weights.get(&base_key) else {
            continue;
        };

        let base_dtype = base_weight.dtype_raw();
        let delta = ops::matmul(lora_b, lora_a);
        let scaled_delta = ops::multiply(&delta, &pmetal::mlx::Array::from_f32(scale));
        let fused = ops::add(base_weight, &scaled_delta);
        // Cast back to base dtype (LoRA is f32, base is typically bf16/f16 —
        // without this cast the fused model is 2x larger than necessary)
        let fused = if fused.dtype_raw() != base_dtype {
            fused.as_dtype(base_dtype)
        } else {
            fused
        };
        base_weights.insert(base_key, fused);
    }

    let output_dir = Path::new(output_path);
    std::fs::create_dir_all(output_dir)?;
    for entry in std::fs::read_dir(&model_dir)
        .map_err(|e| AppError(e.to_string()))?
        .flatten()
    {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.ends_with(".safetensors")
            || name_str == "model.safetensors.index.json"
            || name_str.starts_with('.')
        {
            continue;
        }
        let dest = output_dir.join(&name);
        if entry.path().is_file() {
            std::fs::copy(entry.path(), &dest).map_err(|e| AppError(e.to_string()))?;
        }
    }

    let output_file = output_dir.join("model.safetensors");
    let output_file_str = output_file
        .to_str()
        .ok_or_else(|| AppError(format!("non-UTF-8 output path: {}", output_file.display())))?;
    let entries: Vec<(&str, &pmetal::mlx::Array)> = base_weights
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect();
    pmetal::mlx::Array::save_safetensors(output_file_str, &entries);

    // Generate model.safetensors.index.json (required by LM Studio and other tools)
    let mut weight_map = serde_json::Map::new();
    let mut total_size: u64 = 0;
    for (key, arr) in &base_weights {
        weight_map.insert(
            key.clone(),
            serde_json::Value::String("model.safetensors".to_string()),
        );
        total_size += arr.nbytes() as u64;
    }
    let index = serde_json::json!({
        "metadata": { "total_size": total_size },
        "weight_map": weight_map,
    });
    std::fs::write(
        output_dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index).map_err(|e| AppError(e.to_string()))?,
    )?;
    Ok(())
}

fn run_quantize_in_process(
    model_path: &str,
    method: &str,
    output_path: &str,
) -> std::result::Result<(), AppError> {
    use pmetal::gguf::{
        GgmlType, GgufBuilder,
        dynamic::{DynamicQuantizationConfig, DynamicQuantizer},
        quantize::quantize,
    };

    let resolved_model_path = PathBuf::from(model_path);

    let quantizer = if method == "dynamic" {
        DynamicQuantizer::new(DynamicQuantizationConfig::default(), None)
    } else {
        let base_type = match method {
            "q8_0" => GgmlType::Q8_0,
            "q4_k_m" => GgmlType::Q4K,
            other => {
                return Err(AppError(format!(
                    "Unsupported quantization method: {other}"
                )));
            }
        };
        DynamicQuantizer::new(
            DynamicQuantizationConfig {
                base_type,
                high_precision_type: base_type,
                fallback_type: base_type,
                ..Default::default()
            },
            None,
        )
    };

    let weights = pmetal::models::loader::load_weights(&resolved_model_path)
        .map_err(|e| AppError(e.to_string()))?;

    // Use canonical architecture detection instead of hardcoded string matching.
    let architecture = pmetal::models::ModelArchitecture::detect(&resolved_model_path)
        .map(|arch| arch.to_string().to_lowercase())
        .unwrap_or_else(|_| "llama".to_string());

    let mut builder = GgufBuilder::with_model(&architecture, "quantized-model");
    let mut keys: Vec<_> = weights.keys().collect();
    keys.sort();

    for name in keys {
        let tensor = weights
            .get(name)
            .ok_or_else(|| AppError(format!("Tensor {name} not found")))?;
        let shape_u64: Vec<u64> = tensor.shape().iter().map(|&d| d as u64).collect();
        let target_type = quantizer.get_tensor_type(name, &shape_u64);
        let tensor = tensor.clone();
        tensor.eval();

        let data_f32: Vec<f32> = match tensor.dtype() {
            pmetal::mlx::Dtype::Float32 => tensor.as_slice::<f32>().to_vec(),
            pmetal::mlx::Dtype::Float16 | pmetal::mlx::Dtype::Bfloat16 => {
                let t_f32 = tensor.as_dtype(pmetal::mlx::Dtype::Float32.as_i32());
                t_f32.eval();
                t_f32.as_slice::<f32>().to_vec()
            }
            _ => continue,
        };

        let quantized_data =
            quantize(&data_f32, target_type).map_err(|e| AppError(format!("{e:?}")))?;
        builder.add_raw_tensor(name, shape_u64, target_type, quantized_data);
    }

    let mut file = std::fs::File::create(output_path).map_err(|e| AppError(e.to_string()))?;
    builder
        .write(&mut file)
        .map_err(|e| AppError(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Event forwarder
// ---------------------------------------------------------------------------

/// Subscribes to the broadcast channel and re-emits events as Tauri events.
pub fn start_event_forwarder(app_handle: AppHandle, state: &AppState) {
    let mut rx = state.subscribe();
    tauri::async_runtime::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let (event_name, payload) = match &event {
                        AppEvent::TrainingStarted { run } => (
                            "training-started",
                            serde_json::to_value(run).unwrap_or_default(),
                        ),
                        AppEvent::TrainingStopped { run_id } => (
                            "training-stopped",
                            serde_json::Value::String(run_id.clone()),
                        ),
                        AppEvent::TrainingUpdate { run } => (
                            "training-update",
                            serde_json::to_value(run).unwrap_or_default(),
                        ),
                        AppEvent::DistillationStarted { run } => (
                            "distillation-started",
                            serde_json::to_value(run).unwrap_or_default(),
                        ),
                        AppEvent::DistillationStopped { run_id } => (
                            "distillation-stopped",
                            serde_json::Value::String(run_id.clone()),
                        ),
                        AppEvent::DistillationUpdate { run } => (
                            "distillation-update",
                            serde_json::to_value(run).unwrap_or_default(),
                        ),
                        AppEvent::GrpoStarted { run } => (
                            "grpo-started",
                            serde_json::to_value(run).unwrap_or_default(),
                        ),
                        AppEvent::GrpoStopped { run_id } => {
                            ("grpo-stopped", serde_json::Value::String(run_id.clone()))
                        }
                        AppEvent::GrpoUpdate { run } => {
                            ("grpo-update", serde_json::to_value(run).unwrap_or_default())
                        }
                        AppEvent::ModelCached { model } => (
                            "model-cached",
                            serde_json::to_value(model).unwrap_or_default(),
                        ),
                        AppEvent::ModelRemoved { model_id } => {
                            ("model-removed", serde_json::json!({ "model_id": model_id }))
                        }
                        AppEvent::ProcessLog { run_id, line } => (
                            "process-log",
                            serde_json::json!({ "run_id": run_id, "line": line }),
                        ),
                    };
                    if let Err(e) = app_handle.emit(event_name, payload) {
                        tracing::debug!("Event emit error (no listeners): {}", e);
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("Event forwarder lagged, dropped {} events", n);
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// HF API helpers
// ---------------------------------------------------------------------------

async fn search_hf_models_inner(
    query: String,
    limit: u32,
    token: Option<String>,
) -> Result<Vec<HubSearchResult>> {
    let client = reqwest::Client::new();

    // When query is empty (trending), sort by trending score; otherwise sort by downloads for search
    let sort = if query.is_empty() {
        "trending"
    } else {
        "downloads"
    };
    let mut url = format!(
        "https://huggingface.co/api/models?filter=text-generation&sort={sort}&limit={}",
        limit
    );
    if !query.is_empty() {
        url.push_str(&format!("&search={}", url_encode(&query)));
    }

    let mut req = client.get(&url).header(
        "User-Agent",
        concat!("pmetal-gui/", env!("CARGO_PKG_VERSION")),
    );

    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {}", t));
    }

    let resp = req.send().await?;
    let items: Vec<serde_json::Value> = resp.json().await?;

    let results = items
        .into_iter()
        .map(|v| {
            let id = v["id"].as_str().unwrap_or("").to_string();
            let author = id.split('/').next().map(str::to_string);
            let downloads = v["downloads"].as_u64().unwrap_or(0);
            let tags: Vec<String> = v["tags"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|t| t.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let is_gated =
                v["gated"].as_bool().unwrap_or(false) || tags.iter().any(|t| t == "gated");
            HubSearchResult {
                author,
                downloads_formatted: format_downloads(downloads),
                downloads,
                id,
                likes: v["likes"].as_u64().unwrap_or(0),
                pipeline_tag: v["pipeline_tag"].as_str().map(str::to_string),
                is_gated,
                library_name: v["library_name"].as_str().map(str::to_string),
                tags,
            }
        })
        .collect();

    Ok(results)
}

async fn search_hf_datasets_inner(
    query: String,
    limit: u32,
    token: Option<String>,
) -> Result<Vec<DatasetSearchResult>> {
    let client = reqwest::Client::new();

    let sort = if query.is_empty() {
        "trending"
    } else {
        "downloads"
    };
    let mut url = format!(
        "https://huggingface.co/api/datasets?sort={sort}&limit={}",
        limit
    );
    if !query.is_empty() {
        url.push_str(&format!("&search={}", url_encode(&query)));
    }

    let mut req = client.get(&url).header(
        "User-Agent",
        concat!("pmetal-gui/", env!("CARGO_PKG_VERSION")),
    );

    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {}", t));
    }

    let resp = req.send().await?;
    let items: Vec<serde_json::Value> = resp.json().await?;

    let results = items
        .into_iter()
        .map(|v| {
            let id = v["id"].as_str().unwrap_or("").to_string();
            let author = id.split('/').next().map(str::to_string);
            let downloads = v["downloads"].as_u64().unwrap_or(0);
            DatasetSearchResult {
                author,
                downloads_formatted: format_downloads(downloads),
                downloads,
                id,
                likes: v["likes"].as_u64().unwrap_or(0),
                tags: v["tags"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|t| t.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default(),
                description: v["description"].as_str().map(str::to_string),
            }
        })
        .collect();

    Ok(results)
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

fn get_total_memory_bytes_sync() -> u64 {
    let device_info = pmetal::version::device_info();
    (device_info.memory_total_gb * 1024.0_f64.powi(3)) as u64
}

async fn get_available_memory_bytes() -> Option<u64> {
    let device_info = pmetal::version::device_info();
    Some((device_info.memory_available_gb * 1024.0_f64.powi(3)) as u64)
}

async fn get_bandwidth_gbps() -> Option<f64> {
    pmetal::metal::MetalContext::global()
        .ok()
        .map(|ctx| ctx.properties().memory_bandwidth_gbps)
}

/// Percent-encode a string for use in a URL query parameter value.
fn url_encode(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                vec![c.to_string()]
            }
            ' ' => vec!["+".to_string()],
            c => {
                // Encode each UTF-8 byte
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                s.bytes().map(|b| format!("%{:02X}", b)).collect()
            }
        })
        .collect()
}

fn estimate_params_b(model_id: &str) -> f64 {
    let lower = model_id.to_lowercase();
    let patterns: &[(&str, f64)] = &[
        ("0.5b", 0.5),
        ("1b", 1.0),
        ("1.5b", 1.5),
        ("1.8b", 1.8),
        ("2b", 2.0),
        ("3b", 3.0),
        ("3.8b", 3.8),
        ("4b", 4.0),
        ("7b", 7.0),
        ("8b", 8.0),
        ("9b", 9.0),
        ("11b", 11.0),
        ("13b", 13.0),
        ("14b", 14.0),
        ("20b", 20.0),
        ("27b", 27.0),
        ("32b", 32.0),
        ("34b", 34.0),
        ("40b", 40.0),
        ("70b", 70.0),
        ("72b", 72.0),
        ("110b", 110.0),
        ("235b", 235.0),
    ];
    patterns
        .iter()
        .filter(|(pat, _)| lower.contains(pat))
        .max_by_key(|(pat, _)| pat.len())
        .map(|(_, b)| *b)
        .unwrap_or(7.0)
}

/// Simple non-symlink-aware dir size (for output directories we just created).
async fn dir_size_simple(path: &PathBuf) -> u64 {
    let mut total: u64 = 0;
    let mut stack = vec![path.clone()];
    while let Some(dir) = stack.pop() {
        let mut rd = match tokio::fs::read_dir(&dir).await {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let ft = match entry.file_type().await {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_file() {
                if let Ok(meta) = entry.metadata().await {
                    total += meta.len();
                }
            } else if ft.is_dir() {
                stack.push(entry.path());
            }
        }
    }
    total
}

/// Read `config.json` from a model path.
///
/// Checks (in order):
/// 1. Direct `config.json` at path root (custom dirs, GGUF models with config)
/// 2. HF hub cache layout: `snapshots/{hash}/config.json`
async fn read_model_config_json(repo_path: &str) -> Option<serde_json::Value> {
    let base = PathBuf::from(repo_path);

    // 1. Check root config.json (custom dirs, non-HF layouts)
    let root_config = base.join("config.json");
    if let Ok(data) = tokio::fs::read_to_string(&root_config).await {
        if let Ok(cfg) = serde_json::from_str(&data) {
            return Some(cfg);
        }
    }

    // 2. Check HF hub cache layout: snapshots/{hash}/config.json
    let snapshots = base.join("snapshots");
    if let Ok(mut rd) = tokio::fs::read_dir(&snapshots).await {
        while let Ok(Some(entry)) = rd.next_entry().await {
            if entry.file_type().await.ok().is_some_and(|ft| ft.is_dir()) {
                let config_path = entry.path().join("config.json");
                if let Ok(data) = tokio::fs::read_to_string(&config_path).await {
                    if let Ok(cfg) = serde_json::from_str(&data) {
                        return Some(cfg);
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{InferenceMessage, chat_message_from_inference_message};

    #[test]
    fn inference_message_mapping_preserves_supported_roles() {
        let user = chat_message_from_inference_message(&InferenceMessage {
            role: "user".into(),
            content: "hello".into(),
        })
        .unwrap();
        assert_eq!(user.role, "user");
        assert_eq!(user.content, "hello");

        let assistant = chat_message_from_inference_message(&InferenceMessage {
            role: "assistant".into(),
            content: "hi".into(),
        })
        .unwrap();
        assert_eq!(assistant.role, "assistant");
        assert_eq!(assistant.content, "hi");

        let system = chat_message_from_inference_message(&InferenceMessage {
            role: "system".into(),
            content: "stay concise".into(),
        })
        .unwrap();
        assert_eq!(system.role, "system");
        assert_eq!(system.content, "stay concise");
    }

    #[test]
    fn inference_message_mapping_rejects_unknown_roles() {
        let err = chat_message_from_inference_message(&InferenceMessage {
            role: "tool".into(),
            content: "{}".into(),
        })
        .unwrap_err();

        assert!(err.contains("unsupported chat role 'tool'"));
    }
}
