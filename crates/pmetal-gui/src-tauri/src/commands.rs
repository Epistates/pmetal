use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use mlx_rs::builder::Builder as _;
use mlx_rs::ops;
use pmetal::easy;
use pmetal::prelude::TrainingCallback;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::state::{
    AppConfig, AppEvent, AppState, CachedModel, DistillationRun, DistillationStatus, GrpoRun,
    GrpoStatus, TrainingRun, TrainingStatus, format_downloads, format_size,
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

// ---------------------------------------------------------------------------
// Request DTOs — match api.ts invoke calls
// ---------------------------------------------------------------------------

/// Full training config matching TS `TrainingConfig`.
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
}

/// Full GRPO config matching TS `GrpoConfig`.
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
}

/// Inference config matching TS `InferenceConfig`.
#[derive(Debug, Deserialize)]
pub struct InferenceConfig {
    pub model: String,
    pub lora_path: Option<String>,
    pub prompt: String,
    pub system_message: Option<String>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub repetition_penalty: Option<f32>,
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

    let active_training = training.iter().filter(|r| r.status == TrainingStatus::Running).count();
    let completed_training = training.iter().filter(|r| r.status == TrainingStatus::Completed).count();
    let active_grpo = grpo.iter().filter(|r| r.status == GrpoStatus::Running).count();
    let active_distillation = distillation.iter().filter(|r| {
        r.status == DistillationStatus::Training
            || r.status == DistillationStatus::LoadingModels
            || r.status == DistillationStatus::GeneratingSignals
    }).count();

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

    let Some(cached) = cached else { return Ok(None) };

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

#[tauri::command]
pub async fn delete_model(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<()> {
    let path = {
        let models = state.cached_models.read().await;
        models.iter().find(|m| m.id == model_id).map(|m| m.path.clone())
    };

    if let Some(path) = path {
        tokio::fs::remove_dir_all(&path)
            .await
            .map_err(|e| AppError(format!("Failed to delete model directory: {}", e)))?;

        state.cached_models.write().await.retain(|m| m.id != model_id);
        let _ = state.event_tx.send(AppEvent::ModelRemoved { model_id });
    }

    Ok(())
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

        let token = hf_token.as_ref().map(|s| pmetal::core::SecretString::from(s.clone()));
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
pub async fn get_model_fit(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<ModelFitInfo> {
    let available_memory = get_available_memory_bytes().await
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
pub async fn list_cached_datasets(
    state: State<'_, AppState>,
) -> Result<Vec<CachedDatasetInfo>> {
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

// ---------------------------------------------------------------------------
// Training commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn start_training(
    state: State<'_, AppState>,
    _app_handle: AppHandle,
    config: TrainingConfig,
) -> Result<String> {
    if !matches!(config.method.as_str(), "sft" | "lora" | "qlora") {
        return Err(AppError(format!(
            "GUI training currently supports direct-library SFT/LoRA/QLoRA only. \
Selected method '{}' is not library-backed here yet.",
            config.method
        )));
    }

    if config.resume_from.is_some() {
        return Err(AppError(
            "Resume from checkpoint is not yet supported in GUI direct mode. \
             Use the CLI instead: pmetal train --resume-from <path>".to_string(),
        ));
    }

    let total_epochs = config.epochs.unwrap_or(3);
    let output_dir = config.output_dir.as_deref()
        .unwrap_or("./output")
        .to_string();
    let metrics_path = PathBuf::from(&output_dir).join("metrics.jsonl");

    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let mut run = TrainingRun::new(
        &config.model,
        &config.method,
        config.dataset.as_deref(),
        Some(&output_dir),
        total_epochs,
    );
    run.status = TrainingStatus::Running;
    let run_id = run.id.clone();

    // Register cancellation flag
    state.cancel_flags.write().await.insert(run_id.clone(), cancel_flag.clone());

    state.create_training_run(run).await;

    let run_id_task = run_id.clone();
    let state_arc = state.training_runs.clone();
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
            watch_training_metrics_file(
                watcher_metrics,
                watcher_run_id,
                watcher_state,
                watcher_event_tx,
                watcher_cancel,
            )
            .await;
        });

        let result = if config.method == "qlora" || config.load_in_4bit == Some(true) {
            run_qlora_training_in_process(
                &config,
                &metrics_path,
                cancel_flag.clone(),
            )
            .await
        } else {
            let mut builder = easy::finetune(
                config.model.clone(),
                config.dataset.clone().unwrap_or_default(),
            )
            .epochs(config.epochs.unwrap_or(3) as usize)
            .learning_rate(config.learning_rate.unwrap_or(2e-4))
            .batch_size(config.batch_size.unwrap_or(4) as usize)
            .max_seq_len(config.max_seq_len.unwrap_or(2048) as usize)
            .output(output_dir.clone())
            .lora_dropout(config.lora_dropout.unwrap_or(0.0) as f32)
            .use_rslora(config.use_rslora.unwrap_or(false))
            .use_dora(config.use_dora.unwrap_or(false))
            .flash_attention(config.flash_attention.unwrap_or(true))
            .sequence_packing(config.sequence_packing.unwrap_or(true))
            .gradient_checkpointing(config.gradient_checkpointing.unwrap_or(false))
            .gradient_checkpointing_layers(config.gradient_checkpointing_layers.unwrap_or(4) as usize)
            .metal_fused_optimizer(config.fused_optimizer.unwrap_or(false))
            .metrics_path(metrics_path.clone())
            .callback(Box::new(CancelOnFlag {
                cancelled: cancel_flag.clone(),
            }));

            if let Some(rank) = config.lora_rank {
                builder = builder.lora(rank as usize, config.lora_alpha.unwrap_or(32) as f32);
            }
            if let Some(eval) = config.embedding_lr {
                builder = builder.embedding_lr(eval as f32);
            }

            builder
                .run()
                .await
                .map(|_| ())
                .map_err(|e| AppError(e.to_string()))
        };

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
    let output_dir = config.output_dir.as_deref().unwrap_or("./output").to_string();
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

    state.cancel_flags.write().await.insert(run_id.clone(), cancel_flag.clone());
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

        let result = run_distillation_in_process(
            &config,
            &metrics_path,
            cancel_flag.clone(),
        )
        .await;

        let success = result.is_ok();
        let error = result.err().map(|e| e.to_string());
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
pub async fn list_distillation_runs(
    state: State<'_, AppState>,
) -> Result<Vec<DistillationRun>> {
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
    let output_dir = config.output_dir.as_deref().unwrap_or("./output").to_string();
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

    state.cancel_flags.write().await.insert(run_id.clone(), cancel_flag.clone());
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

        let result = run_grpo_in_process(
            &config,
            &metrics_path,
            cancel_flag.clone(),
        )
        .await;

        let success = result.is_ok();
        let error = result.err().map(|e| e.to_string());
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
        let mut builder = easy::infer(config.model.clone())
            .temperature(config.temperature.unwrap_or(0.7))
            .max_tokens(config.max_tokens.unwrap_or(1024) as usize)
            .top_k(config.top_k.unwrap_or(50) as usize)
            .top_p(config.top_p.unwrap_or(0.9))
            .repetition_penalty(config.repetition_penalty.unwrap_or(1.0));

        if let Some(ref lora) = config.lora_path {
            builder = builder.lora(lora.clone());
        }

        let prompt = match &config.system_message {
            Some(system) if !system.is_empty() => format!("{system}\n\n{}", config.prompt),
            _ => config.prompt.clone(),
        };

        let result = builder.generate_streaming(&prompt, |delta| {
            if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                return false;
            }
            let _ = app_handle.emit("inference-token", delta);
            true
        }).await;

        match result {
            Ok(_) => {
                let _ = app_handle.emit("inference-done", ());
            }
            Err(e) => {
                let _ = app_handle.emit(
                    "inference-error",
                    serde_json::json!({ "session_id": session_id_task, "error": e.to_string() }),
                );
            }
        }

        inference_flags.write().await.remove(&session_id_task);
    });

    Ok(())
}

#[tauri::command]
pub async fn stop_inference(
    state: State<'_, AppState>,
    session_id: Option<String>,
) -> Result<()> {
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
pub async fn merge_models(
    _app_handle: AppHandle,
    config: MergeConfig,
) -> Result<String> {
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
                    weight: Some(model.weight as f32),
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
    let base_model_task = base_model.clone();
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

#[tauri::command]
pub async fn quantize_model(
    _app_handle: AppHandle,
    model_id: String,
    quant_type: String,
    output_dir: String,
) -> Result<String> {
    let model_id_task = model_id.clone();
    let quant_type_task = quant_type.clone();
    let output_dir_task = output_dir.clone();

    tokio::task::spawn_blocking(move || {
        run_quantize_in_process(&model_id_task, &quant_type_task, &output_dir_task)
    })
    .await
    .map_err(|e| AppError(format!("quantize task failed: {e}")))?
    .map_err(|e| AppError(e.to_string()))?;

    Ok(output_dir)
}

async fn watch_training_metrics_file(
    metrics_path: PathBuf,
    run_id: String,
    state_arc: Arc<tokio::sync::RwLock<Vec<TrainingRun>>>,
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
        if file_len <= last_pos {
            continue;
        }

        let path = metrics_path.clone();
        let run_id_inner = run_id.clone();
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
        if let Some(run) = runs.iter_mut().find(|r| r.id == run_id_inner) {
            for row in rows {
                apply_metrics_to_training(run, &row, started_at);
            }
            let _ = event_tx.send(AppEvent::TrainingUpdate { run: run.clone() });
        }
    }
}

async fn finalize_training_run(
    state_arc: &Arc<tokio::sync::RwLock<Vec<TrainingRun>>>,
    event_tx: &tokio::sync::broadcast::Sender<AppEvent>,
    run_id: &str,
    cancel_flag: &Arc<std::sync::atomic::AtomicBool>,
    success: bool,
    error: Option<String>,
) {
    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        run.status = if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
            TrainingStatus::Cancelled
        } else if success {
            TrainingStatus::Completed
        } else {
            TrainingStatus::Failed
        };
        run.ended_at = Some(Utc::now());
        run.error_message = error;
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
    if let Some(v) = metrics["lr"].as_f64().or_else(|| metrics["learning_rate"].as_f64()) {
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
    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        run.status = if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
            DistillationStatus::Cancelled
        } else if success {
            DistillationStatus::Completed
        } else {
            DistillationStatus::Failed
        };
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
    if let Some(v) = metrics["lr"].as_f64().or_else(|| metrics["learning_rate"].as_f64()) {
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
    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        run.status = if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
            GrpoStatus::Cancelled
        } else if success {
            GrpoStatus::Completed
        } else {
            GrpoStatus::Failed
        };
        run.ended_at = Some(Utc::now());
        run.error_message = error;
        let _ = event_tx.send(AppEvent::GrpoUpdate { run: run.clone() });
        let _ = event_tx.send(AppEvent::GrpoStopped {
            run_id: run_id.to_string(),
        });
    }
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
        pmetal::data::DatasetSource::Local(path) => Ok(path),
        pmetal::data::DatasetSource::HuggingFace(id) => {
            let dir = pmetal::hub::download_dataset(&id, None, None, None)
                .await
                .map_err(|e| AppError(e.to_string()))?;
            if let Ok(path) = pmetal::data::TrainingDataset::resolve_dataset_path_pub(&dir) {
                return Ok(path);
            }
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

async fn run_qlora_training_in_process(
    config: &TrainingConfig,
    metrics_path: &PathBuf,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    use pmetal::lora::TrainableModel;

    let model_path = resolve_model_path(&config.model).await?;
    let dataset_id = config
        .dataset
        .as_deref()
        .ok_or_else(|| AppError("Dataset is required for training".to_string()))?;
    let dataset_path = resolve_dataset_path(dataset_id).await?;
    let output_dir = config
        .output_dir
        .clone()
        .unwrap_or_else(|| "./output".to_string());

    let config_path = model_path.join("config.json");
    let config_text = std::fs::read_to_string(&config_path).map_err(|e| {
        AppError(format!(
            "QLoRA requires config.json; failed to read {}: {e}",
            config_path.display()
        ))
    })?;

    let config_json: serde_json::Value = serde_json::from_str(&config_text)
        .map_err(|e| AppError(format!("Failed to parse config.json: {e}")))?;
    let model_type = config_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama")
        .to_lowercase();

    let llama_config: pmetal::models::architectures::llama::LlamaConfig = match model_type.as_str() {
        "llama" | "mistral" => {
            serde_json::from_str(&config_text).map_err(|e| AppError(e.to_string()))?
        }
        other => {
            return Err(AppError(format!(
                "QLoRA in GUI currently supports Llama/Mistral architectures only. \
                 Detected model_type '{}'. Use LoRA instead, or the CLI for other architectures.",
                other
            )));
        }
    };

    let tokenizer = pmetal::data::Tokenizer::from_model_dir(&model_path)
        .map_err(|e| AppError(e.to_string()))?;
    let chat_template =
        pmetal::data::chat_templates::detect_chat_template(&model_path, &config.model);

    let max_seq_len = config.max_seq_len.unwrap_or(2048) as usize;
    let train_dataset = if dataset_path
        .extension()
        .is_some_and(|ext| ext == "parquet")
    {
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
        )
        .map_err(|e| AppError(e.to_string()))?
    };

    let quant_scheme = pmetal::mlx::quantization::QuantScheme::NF4;
    let qlora_config = pmetal::lora::QLoraConfig {
        lora: pmetal::core::LoraConfig {
            r: config.lora_rank.unwrap_or(16) as usize,
            alpha: config.lora_alpha.unwrap_or(32) as f32,
            dropout: config.lora_dropout.unwrap_or(0.0) as f32,
            use_rslora: config.use_rslora.unwrap_or(false),
            use_dora: config.use_dora.unwrap_or(false),
            ..Default::default()
        },
        quant_scheme,
        block_size: 64,
        double_quant: false,
        compute_in_half: true,
    };

    let mut model =
        pmetal::lora::LlamaQloraForCausalLM::with_qlora_config(llama_config, qlora_config.clone())
            .map_err(|e| AppError(e.to_string()))?;
    model
        .load_and_quantize_from_dir(&model_path)
        .map_err(|e| AppError(e.to_string()))?;

    if config.gradient_checkpointing.unwrap_or(false) && model.supports_gradient_checkpointing() {
        model.enable_gradient_checkpointing(
            config.gradient_checkpointing_layers.unwrap_or(4) as usize,
        );
    }

    let checkpoint_dir = PathBuf::from(&output_dir).join("checkpoints");
    let checkpoint_manager = pmetal::trainer::CheckpointManager::new(&checkpoint_dir)
        .map_err(|e| AppError(e.to_string()))?
        .with_max_checkpoints(3);

    let training_loop_config = pmetal::trainer::TrainingLoopConfig {
        training: pmetal::core::TrainingConfig {
            learning_rate: config.learning_rate.unwrap_or(2e-4),
            batch_size: config.batch_size.unwrap_or(4) as usize,
            num_epochs: config.epochs.unwrap_or(3) as usize,
            max_seq_len,
            gradient_accumulation_steps: config.gradient_accumulation_steps.unwrap_or(1) as usize,
            weight_decay: config.weight_decay.unwrap_or(0.0),
            max_grad_norm: config.max_grad_norm.unwrap_or(1.0),
            output_dir: output_dir.clone(),
            ..Default::default()
        },
        dataloader: pmetal::data::DataLoaderConfig {
            batch_size: config.batch_size.unwrap_or(4) as usize,
            max_seq_len,
            shuffle: true,
            seed: 42,
            pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
            drop_last: false,
        },
        use_metal_flash_attention: config.flash_attention.unwrap_or(true),
        log_every: config.logging_steps.unwrap_or(10) as usize,
        checkpoint_every: config.save_steps.unwrap_or(500) as usize,
        eval_every: 0,
        use_jit_compilation: config.jit_compilation.unwrap_or(true),
        use_sequence_packing: config.sequence_packing.unwrap_or(true),
        gradient_checkpointing: config.gradient_checkpointing.unwrap_or(false),
        gradient_checkpointing_layers: config.gradient_checkpointing_layers.unwrap_or(4) as usize,
        embedding_lr: config.embedding_lr.map(|v| v as f32),
        eager_evaluation: true,
        use_metal_fused_optimizer: config.fused_optimizer.unwrap_or(false),
    };

    let mut training_loop = pmetal::trainer::TrainingLoop::new(training_loop_config);
    let callback = pmetal::trainer::MetricsJsonCallback::new(metrics_path)
        .map_err(|e| AppError(e.to_string()))?
        .with_run_name(format!("train-{}", config.model.replace('/', "-")));
    training_loop.add_callback(Box::new(callback));
    training_loop.add_callback(Box::new(CancelOnFlag {
        cancelled: cancel_flag,
    }));

    let adaptive_config = pmetal::trainer::AdaptiveLrConfig::default();
    let control_file = PathBuf::from(&output_dir).join(".lr_control.json");
    training_loop.enable_adaptive_lr_with_control(adaptive_config, control_file);

    training_loop
        .run(&mut model, train_dataset, None, Some(&checkpoint_manager))
        .map_err(|e| AppError(e.to_string()))?;

    let output_dir_path = PathBuf::from(&output_dir);
    std::fs::create_dir_all(&output_dir_path).map_err(|e| AppError(e.to_string()))?;
    let final_path = output_dir_path.join("lora_weights.safetensors");
    model
        .save_lora_weights(&final_path)
        .map_err(|e| AppError(e.to_string()))?;
    let adapter_config = serde_json::json!({
        "r": qlora_config.lora.r,
        "alpha": qlora_config.lora.alpha,
        "target_modules": qlora_config.lora.target_modules,
        "use_rslora": qlora_config.lora.use_rslora,
    });
    std::fs::write(
        output_dir_path.join("adapter_config.json"),
        serde_json::to_string_pretty(&adapter_config).map_err(|e| AppError(e.to_string()))?,
    )
    .map_err(|e| AppError(e.to_string()))?;

    Ok(())
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
    let train_dataset = if dataset_path
        .extension()
        .is_some_and(|ext| ext == "parquet")
    {
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
        )
        .map_err(|e| AppError(e.to_string()))?
    };

    let teacher_lora_config = pmetal::core::LoraConfig {
        r: 0,
        ..Default::default()
    };
    let mut teacher_model = pmetal::lora::DynamicLoraModel::from_pretrained(
        &teacher_path,
        teacher_lora_config,
    )
    .map_err(|e| AppError(e.to_string()))?;

    let student_lora_config = pmetal::core::LoraConfig {
        r: config.lora_rank.unwrap_or(16) as usize,
        alpha: config.lora_alpha.unwrap_or(32) as f32,
        ..Default::default()
    };
    let mut student_model = pmetal::lora::DynamicLoraModel::from_pretrained(
        &student_path,
        student_lora_config.clone(),
    )
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
        other => return Err(AppError(format!("Unsupported distillation loss type: {other}"))),
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

    let distiller = pmetal::distill::Distiller::new(distill_config)
        .map_err(|e| AppError(e.to_string()))?;

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
        .with_run_name(format!("distill-{}", config.student_model.replace('/', "-")));
    trainer.add_callback(Box::new(callback));
    trainer.add_callback(Box::new(CancelOnFlag {
        cancelled: cancel_flag,
    }));

    trainer
        .run(&mut student_model, &mut teacher_model, train_dataset, None, None)
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
    let dataset = if dataset_path
        .extension()
        .is_some_and(|ext| ext == "parquet")
    {
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

    let mut grpo_config =
        pmetal::trainer::GrpoConfig::new(config.group_size.unwrap_or(8) as usize)
            .with_beta(config.beta.unwrap_or(0.04));
    grpo_config.max_prompt_length = max_seq_len;
    grpo_config.max_completion_length = 512;

    let mut rewards = pmetal::trainer::CombinedReward::new();
    if config.use_reasoning_rewards.unwrap_or(false) {
        rewards = rewards.add(
            Box::new(pmetal::trainer::XmlFormatReward::default_reasoning()),
            1.0,
        );
    } else {
        return Err(AppError(
            "GRPO requires a reward function. Enable 'Use Reasoning Rewards' or use the CLI \
             with a custom reward configuration.".to_string(),
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

    let mut optimizer = mlx_rs::optimizers::AdamWBuilder::new(
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
                opt.lr = mlx_rs::array!(lr);
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

    let model_dir: PathBuf = if model_path.contains('/') && !PathBuf::from(model_path).exists() {
        return Err(AppError(
            "Fuse with remote base models is not supported in the GUI yet.".to_string(),
        ));
    } else {
        PathBuf::from(model_path)
    };

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
    let lora_weights = pmetal::mlx::Array::load_safetensors(&lora_file)
        .map_err(|e| AppError(e.to_string()))?;

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

    for (layer_name, lora_a) in &lora_a_map {
        let Some(lora_b) = lora_b_map.get(layer_name) else {
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

        let delta = ops::matmul(lora_b, lora_a)
            .map_err(|e| AppError(e.to_string()))?;
        let scaled_delta = ops::multiply(&delta, pmetal::mlx::Array::from_f32(scale))
            .map_err(|e| AppError(e.to_string()))?;
        let fused = ops::add(base_weight, &scaled_delta)
            .map_err(|e| AppError(e.to_string()))?;
        base_weights.insert(base_key, fused);
    }

    let output_dir = Path::new(output_path);
    std::fs::create_dir_all(output_dir)?;
    for entry in std::fs::read_dir(&model_dir).map_err(|e| AppError(e.to_string()))?.flatten() {
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
    pmetal::mlx::Array::save_safetensors(&base_weights, None, &output_file)
        .map_err(|e| AppError(e.to_string()))?;
    Ok(())
}

fn run_quantize_in_process(
    model_path: &str,
    method: &str,
    output_path: &str,
) -> std::result::Result<(), AppError> {
    use pmetal::gguf::{
        GgmlType,
        GgufBuilder,
        dynamic::{DynamicQuantizationConfig, DynamicQuantizer},
        quantize::quantize,
    };

    let resolved_model_path = if model_path.contains('/') && !PathBuf::from(model_path).exists() {
        return Err(AppError(
            "Quantizing remote models is not supported in the GUI yet.".to_string(),
        ));
    } else {
        PathBuf::from(model_path)
    };

    let quantizer = if method == "dynamic" {
        DynamicQuantizer::new(DynamicQuantizationConfig::default(), None)
    } else {
        let base_type = match method {
            "q8_0" => GgmlType::Q8_0,
            "q4_k_m" => GgmlType::Q4K,
            other => return Err(AppError(format!("Unsupported quantization method: {other}"))),
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

    let config_path = resolved_model_path.join("config.json");
    let mut architecture = "llama".to_string();
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(archs) = json.get("architectures").and_then(|v| v.as_array()) {
                    if let Some(arch_str) = archs.first().and_then(|v| v.as_str()) {
                        architecture = match arch_str {
                            "LlamaForCausalLM" => "llama".to_string(),
                            "MistralForCausalLM" => "mistral".to_string(),
                            "Qwen2ForCausalLM" => "qwen2".to_string(),
                            "GemmaForCausalLM" | "Gemma2ForCausalLM" => "gemma".to_string(),
                            "PhiForCausalLM" | "Phi3ForCausalLM" => "phi".to_string(),
                            _ => "llama".to_string(),
                        };
                    }
                }
            }
        }
    }

    let mut builder = GgufBuilder::with_model(&architecture, "quantized-model");
    let mut keys: Vec<_> = weights.keys().collect();
    keys.sort();

    for name in keys {
        let tensor = weights
            .get(name)
            .ok_or_else(|| AppError(format!("Tensor {name} not found")))?;
        let shape_u64: Vec<u64> = tensor.shape().iter().map(|&d| d as u64).collect();
        let target_type = quantizer.get_tensor_type(name, &shape_u64);
        tensor.eval().map_err(|e| AppError(e.to_string()))?;

        let data_f32: Vec<f32> = match tensor.dtype() {
            pmetal::mlx::Dtype::Float32 => tensor.as_slice::<f32>().to_vec(),
            pmetal::mlx::Dtype::Float16 | pmetal::mlx::Dtype::Bfloat16 => {
                let t_f32 = tensor
                    .as_dtype(pmetal::mlx::Dtype::Float32)
                    .map_err(|e| AppError(e.to_string()))?;
                t_f32.eval().map_err(|e| AppError(e.to_string()))?;
                t_f32.as_slice::<f32>().to_vec()
            }
            _ => continue,
        };

        let quantized_data =
            quantize(&data_f32, target_type).map_err(|e| AppError(format!("{e:?}")))?;
        builder.add_raw_tensor(name, shape_u64, target_type, quantized_data);
    }

    let mut file = std::fs::File::create(output_path).map_err(|e| AppError(e.to_string()))?;
    builder.write(&mut file).map_err(|e| AppError(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Event forwarder
// ---------------------------------------------------------------------------

/// Subscribes to the broadcast channel and re-emits events as Tauri events.
pub fn start_event_forwarder(app_handle: AppHandle, state: &AppState) {
    let mut rx = state.subscribe();
    tokio::spawn(async move {
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
                        AppEvent::GrpoStopped { run_id } => (
                            "grpo-stopped",
                            serde_json::Value::String(run_id.clone()),
                        ),
                        AppEvent::GrpoUpdate { run } => (
                            "grpo-update",
                            serde_json::to_value(run).unwrap_or_default(),
                        ),
                        AppEvent::ModelCached { model } => (
                            "model-cached",
                            serde_json::to_value(model).unwrap_or_default(),
                        ),
                        AppEvent::ModelRemoved { model_id } => (
                            "model-removed",
                            serde_json::json!({ "model_id": model_id }),
                        ),
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
// Internal subprocess monitoring helpers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn spawn_and_monitor_training(
    mut child: tokio::process::Child,
    run_id: String,
    state_arc: Arc<tokio::sync::RwLock<Vec<TrainingRun>>>,
    event_tx: tokio::sync::broadcast::Sender<AppEvent>,
    procs_arc: Arc<tokio::sync::RwLock<HashMap<String, tokio::process::Child>>>,
    cancel_flags: Arc<tokio::sync::RwLock<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>>,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
    app_handle: AppHandle,
) {
    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => {
            mark_training_failed(&state_arc, &event_tx, &run_id, "No stdout").await;
            return;
        }
    };

    // Also capture stderr and forward as process-log lines
    if let Some(stderr) = child.stderr.take() {
        let ah = app_handle.clone();
        let rid = run_id.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = ah.emit(
                    "process-log",
                    serde_json::json!({ "run_id": rid, "line": line }),
                );
            }
        });
    }

    {
        let mut procs = procs_arc.write().await;
        procs.insert(run_id.clone(), child);
    }

    let mut lines = BufReader::new(stdout).lines();
    let start = Utc::now();

    while let Ok(Some(line)) = lines.next_line().await {
        if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }

        let _ = app_handle.emit(
            "process-log",
            serde_json::json!({ "run_id": run_id, "line": line }),
        );

        if let Ok(metrics) = serde_json::from_str::<serde_json::Value>(&line) {
            let mut runs = state_arc.write().await;
            if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
                apply_metrics_to_training(run, &metrics, start);
                let _ = event_tx.send(AppEvent::TrainingUpdate { run: run.clone() });
            }
        }
    }

    // Reap process
    let exit_ok = {
        let mut procs = procs_arc.write().await;
        if let Some(mut child) = procs.remove(&run_id) {
            child.wait().await.map(|s| s.success()).unwrap_or(false)
        } else {
            true
        }
    };

    // Remove cancellation flag
    cancel_flags.write().await.remove(&run_id);

    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        if run.status == TrainingStatus::Running {
            run.status = if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                TrainingStatus::Cancelled
            } else if exit_ok {
                TrainingStatus::Completed
            } else {
                TrainingStatus::Failed
            };
            run.ended_at = Some(Utc::now());
            let _ = event_tx.send(AppEvent::TrainingUpdate { run: run.clone() });
            let _ = event_tx.send(AppEvent::TrainingStopped { run_id: run_id.clone() });
        }
    }
}

fn apply_metrics_to_training(
    run: &mut TrainingRun,
    m: &serde_json::Value,
    started_at: chrono::DateTime<Utc>,
) {
    if let Some(v) = m["step"].as_u64() { run.step = v; }
    if let Some(v) = m["total_steps"].as_u64() { run.total_steps = v; }
    if let Some(v) = m["epoch"].as_f64() { run.epoch = v as f32; }
    if let Some(v) = m["loss"].as_f64() {
        run.loss = Some(v);
        if run.best_loss.map_or(true, |b| v < b) {
            run.best_loss = Some(v);
        }
    }
    if let Some(v) = m["learning_rate"].as_f64() { run.learning_rate = Some(v); }
    if let Some(v) = m["grad_norm"].as_f64() { run.grad_norm = Some(v); }
    if let Some(v) = m["tokens_per_second"].as_f64() { run.tokens_per_second = Some(v); }
    if run.total_steps > 0 && run.step > 0 {
        let elapsed = (Utc::now() - started_at).num_seconds().max(1) as f64;
        let remaining = run.total_steps.saturating_sub(run.step) as f64;
        run.eta_seconds = Some(((elapsed / run.step as f64) * remaining) as u64);
    }
}

#[allow(clippy::too_many_arguments)]
async fn spawn_and_monitor_distillation(
    mut child: tokio::process::Child,
    run_id: String,
    state_arc: Arc<tokio::sync::RwLock<Vec<DistillationRun>>>,
    event_tx: tokio::sync::broadcast::Sender<AppEvent>,
    procs_arc: Arc<tokio::sync::RwLock<HashMap<String, tokio::process::Child>>>,
    cancel_flags: Arc<tokio::sync::RwLock<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>>,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
    app_handle: AppHandle,
) {
    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => {
            mark_distillation_failed(&state_arc, &event_tx, &run_id, "No stdout").await;
            return;
        }
    };

    if let Some(stderr) = child.stderr.take() {
        let ah = app_handle.clone();
        let rid = run_id.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = ah.emit("process-log", serde_json::json!({ "run_id": rid, "line": line }));
            }
        });
    }

    {
        let mut procs = procs_arc.write().await;
        procs.insert(run_id.clone(), child);
    }

    let mut lines = BufReader::new(stdout).lines();
    let start = Utc::now();

    while let Ok(Some(line)) = lines.next_line().await {
        if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        let _ = app_handle.emit("process-log", serde_json::json!({ "run_id": run_id, "line": line }));

        if let Ok(m) = serde_json::from_str::<serde_json::Value>(&line) {
            let mut runs = state_arc.write().await;
            if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
                if let Some(v) = m["step"].as_u64() { run.step = v; }
                if let Some(v) = m["total_steps"].as_u64() { run.total_steps = Some(v); }
                if let Some(v) = m["epoch"].as_u64() { run.epoch = v; }
                if let Some(v) = m["loss"].as_f64() {
                    run.loss = Some(v);
                    if run.best_loss.map_or(true, |b| v < b) { run.best_loss = Some(v); }
                }
                if let Some(v) = m["learning_rate"].as_f64() { run.learning_rate = Some(v); }
                if let Some(v) = m["tokens_per_second"].as_f64() { run.tokens_per_second = Some(v); }
                if let (Some(total), step) = (run.total_steps, run.step) {
                    if step > 0 {
                        let elapsed = (Utc::now() - start).num_seconds().max(1) as f64;
                        run.eta_seconds = Some(
                            ((elapsed / step as f64) * total.saturating_sub(step) as f64) as u64,
                        );
                    }
                }
                let _ = event_tx.send(AppEvent::DistillationUpdate { run: run.clone() });
            }
        }
    }

    let exit_ok = {
        let mut procs = procs_arc.write().await;
        if let Some(mut child) = procs.remove(&run_id) {
            child.wait().await.map(|s| s.success()).unwrap_or(false)
        } else {
            true
        }
    };

    cancel_flags.write().await.remove(&run_id);

    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        if run.status == DistillationStatus::Training
            || run.status == DistillationStatus::LoadingModels
            || run.status == DistillationStatus::GeneratingSignals
        {
            run.status = if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                DistillationStatus::Cancelled
            } else if exit_ok {
                DistillationStatus::Completed
            } else {
                DistillationStatus::Failed
            };
            run.ended_at = Some(Utc::now());
            let _ = event_tx.send(AppEvent::DistillationUpdate { run: run.clone() });
            let _ = event_tx.send(AppEvent::DistillationStopped { run_id: run_id.clone() });
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn spawn_and_monitor_grpo(
    mut child: tokio::process::Child,
    run_id: String,
    state_arc: Arc<tokio::sync::RwLock<Vec<GrpoRun>>>,
    event_tx: tokio::sync::broadcast::Sender<AppEvent>,
    procs_arc: Arc<tokio::sync::RwLock<HashMap<String, tokio::process::Child>>>,
    cancel_flags: Arc<tokio::sync::RwLock<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>>,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
    app_handle: AppHandle,
) {
    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => {
            mark_grpo_failed(&state_arc, &event_tx, &run_id, "No stdout").await;
            return;
        }
    };

    if let Some(stderr) = child.stderr.take() {
        let ah = app_handle.clone();
        let rid = run_id.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = ah.emit("process-log", serde_json::json!({ "run_id": rid, "line": line }));
            }
        });
    }

    {
        let mut procs = procs_arc.write().await;
        procs.insert(run_id.clone(), child);
    }

    let mut lines = BufReader::new(stdout).lines();
    let start = Utc::now();

    while let Ok(Some(line)) = lines.next_line().await {
        if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        let _ = app_handle.emit("process-log", serde_json::json!({ "run_id": run_id, "line": line }));

        if let Ok(m) = serde_json::from_str::<serde_json::Value>(&line) {
            let mut runs = state_arc.write().await;
            if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
                if let Some(v) = m["step"].as_u64() { run.step = v; }
                if let Some(v) = m["total_steps"].as_u64() { run.total_steps = Some(v); }
                if let Some(v) = m["loss"].as_f64() {
                    run.loss = Some(v);
                    if run.best_loss.map_or(true, |b| v < b) { run.best_loss = Some(v); }
                }
                if let Some(v) = m["reward_mean"].as_f64() { run.reward_mean = Some(v); }
                if let Some(v) = m["reward_std"].as_f64() { run.reward_std = Some(v); }
                if let Some(v) = m["kl_div"].as_f64() { run.kl_div = Some(v); }
                if let Some(v) = m["learning_rate"].as_f64() { run.learning_rate = Some(v); }
                if let Some(v) = m["tokens_per_second"].as_f64() { run.tokens_per_second = Some(v); }
                if let (Some(total), step) = (run.total_steps, run.step) {
                    if step > 0 {
                        let elapsed = (Utc::now() - start).num_seconds().max(1) as f64;
                        run.eta_seconds = Some(
                            ((elapsed / step as f64) * total.saturating_sub(step) as f64) as u64,
                        );
                    }
                }
                let _ = event_tx.send(AppEvent::GrpoUpdate { run: run.clone() });
            }
        }
    }

    let exit_ok = {
        let mut procs = procs_arc.write().await;
        if let Some(mut child) = procs.remove(&run_id) {
            child.wait().await.map(|s| s.success()).unwrap_or(false)
        } else {
            true
        }
    };

    cancel_flags.write().await.remove(&run_id);

    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        if run.status == GrpoStatus::Running {
            run.status = if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                GrpoStatus::Cancelled
            } else if exit_ok {
                GrpoStatus::Completed
            } else {
                GrpoStatus::Failed
            };
            run.ended_at = Some(Utc::now());
            let _ = event_tx.send(AppEvent::GrpoUpdate { run: run.clone() });
            let _ = event_tx.send(AppEvent::GrpoStopped { run_id: run_id.clone() });
        }
    }
}

// ---------------------------------------------------------------------------
// Failure markers
// ---------------------------------------------------------------------------

async fn mark_training_failed(
    state_arc: &Arc<tokio::sync::RwLock<Vec<TrainingRun>>>,
    event_tx: &tokio::sync::broadcast::Sender<AppEvent>,
    run_id: &str,
    msg: &str,
) {
    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        run.status = TrainingStatus::Failed;
        run.error_message = Some(msg.to_string());
        run.ended_at = Some(Utc::now());
        let _ = event_tx.send(AppEvent::TrainingUpdate { run: run.clone() });
        let _ = event_tx.send(AppEvent::TrainingStopped { run_id: run_id.to_string() });
    }
}

async fn mark_distillation_failed(
    state_arc: &Arc<tokio::sync::RwLock<Vec<DistillationRun>>>,
    event_tx: &tokio::sync::broadcast::Sender<AppEvent>,
    run_id: &str,
    msg: &str,
) {
    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        run.status = DistillationStatus::Failed;
        run.error_message = Some(msg.to_string());
        run.ended_at = Some(Utc::now());
        let _ = event_tx.send(AppEvent::DistillationUpdate { run: run.clone() });
        let _ = event_tx.send(AppEvent::DistillationStopped { run_id: run_id.to_string() });
    }
}

async fn mark_grpo_failed(
    state_arc: &Arc<tokio::sync::RwLock<Vec<GrpoRun>>>,
    event_tx: &tokio::sync::broadcast::Sender<AppEvent>,
    run_id: &str,
    msg: &str,
) {
    let mut runs = state_arc.write().await;
    if let Some(run) = runs.iter_mut().find(|r| r.id == run_id) {
        run.status = GrpoStatus::Failed;
        run.error_message = Some(msg.to_string());
        run.ended_at = Some(Utc::now());
        let _ = event_tx.send(AppEvent::GrpoUpdate { run: run.clone() });
        let _ = event_tx.send(AppEvent::GrpoStopped { run_id: run_id.to_string() });
    }
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

    let mut url = format!(
        "https://huggingface.co/api/models?filter=text-generation&sort=downloads&limit={}",
        limit
    );
    if !query.is_empty() {
        url.push_str(&format!("&search={}", url_encode(&query)));
    }

    let mut req = client
        .get(&url)
        .header("User-Agent", "pmetal-gui/0.3.6");

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
                .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            let is_gated = v["gated"].as_bool().unwrap_or(false)
                || tags.iter().any(|t| t == "gated");
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

    let mut url = format!(
        "https://huggingface.co/api/datasets?sort=downloads&limit={}",
        limit
    );
    if !query.is_empty() {
        url.push_str(&format!("&search={}", url_encode(&query)));
    }

    let mut req = client
        .get(&url)
        .header("User-Agent", "pmetal-gui/0.3.6");

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
                    .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
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

async fn get_pmetal_version() -> String {
    pmetal::version::VERSION.to_string()
}

async fn get_total_memory_bytes() -> u64 {
    let out = tokio::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .await
        .ok();
    out.and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(16 * 1024 * 1024 * 1024)
}

fn get_total_memory_bytes_sync() -> u64 {
    // Synchronous fallback using sysctl directly — just use a constant estimate
    16 * 1024 * 1024 * 1024
}

/// Attempts to read available (free + inactive) memory via `vm_stat` on macOS.
async fn get_available_memory_bytes() -> Option<u64> {
    let out = tokio::process::Command::new("vm_stat")
        .output()
        .await
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;

    let page_size: u64 = 16384; // macOS default page size

    let mut free_pages: u64 = 0;
    let mut inactive_pages: u64 = 0;

    for line in text.lines() {
        if line.contains("Pages free:") {
            let val = line.split(':').nth(1)?
                .trim().trim_end_matches('.').parse::<u64>().ok()?;
            free_pages = val;
        } else if line.contains("Pages inactive:") {
            let val = line.split(':').nth(1)?
                .trim().trim_end_matches('.').parse::<u64>().ok()?;
            inactive_pages = val;
        }
    }

    Some((free_pages + inactive_pages) * page_size)
}

/// Try to get memory bandwidth via `pmetal memory` output, or return None.
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
        ("0.5b", 0.5), ("1b", 1.0), ("1.5b", 1.5), ("1.8b", 1.8),
        ("2b", 2.0), ("3b", 3.0), ("3.8b", 3.8), ("4b", 4.0),
        ("7b", 7.0), ("8b", 8.0), ("9b", 9.0), ("11b", 11.0),
        ("13b", 13.0), ("14b", 14.0), ("20b", 20.0), ("27b", 27.0),
        ("32b", 32.0), ("34b", 34.0), ("40b", 40.0), ("70b", 70.0),
        ("72b", 72.0), ("110b", 110.0), ("235b", 235.0),
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

/// Read `config.json` from the first snapshot directory found under a HF hub model repo.
async fn read_model_config_json(repo_path: &str) -> Option<serde_json::Value> {
    let snapshots = PathBuf::from(repo_path).join("snapshots");
    let mut rd = tokio::fs::read_dir(&snapshots).await.ok()?;

    // Take the first (usually only) snapshot hash directory
    while let Ok(Some(entry)) = rd.next_entry().await {
        if entry.file_type().await.ok()?.is_dir() {
            let config_path = entry.path().join("config.json");
            if let Ok(data) = tokio::fs::read_to_string(&config_path).await {
                return serde_json::from_str(&data).ok();
            }
        }
    }
    None
}
