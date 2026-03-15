use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use chrono::Utc;
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

/// Returns combined `SystemInfo` / `DeviceInfo` by parsing `pmetal memory` stdout.
/// Falls back gracefully if the binary is not found.
async fn get_system_info_inner() -> SystemInfo {
    let version = get_pmetal_version().await;
    let platform = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    let is_apple_silicon = arch == "aarch64" && platform == "macos";

    // Try to get memory via sysctl for a baseline
    let total_memory = get_total_memory_bytes().await;
    // available memory: use vm_stat parsing on macOS
    let available_memory = get_available_memory_bytes().await.unwrap_or(total_memory / 4);

    let mut gpu_name = "Apple GPU".to_string();
    let mut chip_tier: Option<String> = None;
    let mut gpu_cores: Option<u32> = None;
    let mut ane_cores: Option<u32> = None;
    let mut memory_bandwidth_gbps: Option<f64> = None;
    let mut has_ane = is_apple_silicon;
    let mut has_nax = false;

    // Parse `pmetal memory` text output
    if let Ok(pmetal_bin) = which::which("pmetal") {
        if let Ok(out) = tokio::process::Command::new(&pmetal_bin)
            .arg("memory")
            .output()
            .await
        {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                parse_pmetal_memory_output(
                    &text,
                    &mut gpu_name,
                    &mut chip_tier,
                    &mut gpu_cores,
                    &mut ane_cores,
                    &mut memory_bandwidth_gbps,
                    &mut has_ane,
                    &mut has_nax,
                );
            }
        }
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

/// Parse lines such as:
///   Device: Apple M4 Max
///   GPU Cores: 40
///   Memory: 128.00 GB (125.47 GB available)
///   Bandwidth: 546 GB/s
///   ANE: 16 cores
///   NAX: true
fn parse_pmetal_memory_output(
    text: &str,
    gpu_name: &mut String,
    chip_tier: &mut Option<String>,
    gpu_cores: &mut Option<u32>,
    ane_cores: &mut Option<u32>,
    memory_bandwidth_gbps: &mut Option<f64>,
    has_ane: &mut bool,
    has_nax: &mut bool,
) {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Device:") {
            let name = rest.trim().to_string();
            // Infer chip tier from device name
            let lower = name.to_lowercase();
            *chip_tier = if lower.contains("ultra") {
                Some("ultra".to_string())
            } else if lower.contains("max") {
                Some("max".to_string())
            } else if lower.contains("pro") {
                Some("pro".to_string())
            } else {
                None
            };
            *gpu_name = name;
        } else if let Some(rest) = line.strip_prefix("GPU Cores:") {
            if let Ok(n) = rest.trim().parse::<u32>() {
                *gpu_cores = Some(n);
            }
        } else if let Some(rest) = line.strip_prefix("Bandwidth:") {
            // e.g. "546 GB/s"
            let val = rest.trim().split_whitespace().next().unwrap_or("0");
            if let Ok(v) = val.parse::<f64>() {
                *memory_bandwidth_gbps = Some(v);
            }
        } else if let Some(rest) = line.strip_prefix("ANE:") {
            // e.g. "16 cores" or "none"
            let val = rest.trim();
            if val == "none" || val == "false" {
                *has_ane = false;
            } else {
                *has_ane = true;
                // parse core count
                if let Ok(n) = val.split_whitespace().next().unwrap_or("0").parse::<u32>() {
                    *ane_cores = Some(n);
                }
            }
        } else if let Some(rest) = line.strip_prefix("NAX:") {
            *has_nax = rest.trim().eq_ignore_ascii_case("true");
        }
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
    let pmetal_bin = which::which("pmetal")
        .map_err(|_| AppError("pmetal binary not found in PATH".to_string()))?;

    let run_id = uuid::Uuid::new_v4().to_string();
    let run_id_task = run_id.clone();
    let model_id_task = model_id.clone();

    let hf_token = state.config.read().await.hf_token.clone();
    let cached_models = state.cached_models.clone();
    let cache_dir = state.config.read().await.cache_dir.clone();

    tokio::spawn(async move {
        let mut cmd = tokio::process::Command::new(&pmetal_bin);
        cmd.arg("download").arg(&model_id_task);
        if let Some(rev) = revision {
            cmd.args(["--revision", &rev]);
        }
        if let Some(token) = hf_token {
            cmd.args(["--hf-token", &token]);
        }
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let _ = app_handle.emit("download-started", &run_id_task);

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take().unwrap();
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = app_handle.emit(
                        "download-progress",
                        serde_json::json!({ "run_id": run_id_task, "line": line }),
                    );
                }
                let status = child.wait().await.ok();
                let success = status.map(|s| s.success()).unwrap_or(false);

                if success {
                    // Refresh the model cache so the new model appears immediately
                    let hub_dir = PathBuf::from(&cache_dir).join("hub");
                    let models = crate::state::scan_hub_cache_pub(&hub_dir).await;
                    *cached_models.write().await = models;

                    let _ = app_handle.emit("download-completed", &run_id_task);
                } else {
                    let _ = app_handle.emit(
                        "download-error",
                        serde_json::json!({ "run_id": run_id_task, "error": "Download failed" }),
                    );
                }
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
    app_handle: AppHandle,
    config: TrainingConfig,
) -> Result<String> {
    let pmetal_bin = which::which("pmetal")
        .map_err(|_| AppError("pmetal binary not found in PATH".to_string()))?;

    let total_epochs = config.epochs.unwrap_or(3);
    let output_dir = config.output_dir.as_deref()
        .unwrap_or("./output")
        .to_string();

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
    let procs_arc = state.active_processes.clone();
    let cancel_flags = state.cancel_flags.clone();

    tokio::spawn(async move {
        let mut cmd = tokio::process::Command::new(&pmetal_bin);
        cmd.arg("train");
        cmd.args(["--model", &config.model]);
        cmd.args(["--method", &config.method]);
        cmd.args(["--output", &output_dir]);

        if let Some(ref ds) = config.dataset {
            cmd.args(["--dataset", ds]);
        }
        if let Some(v) = config.epochs {
            cmd.args(["--epochs", &v.to_string()]);
        }
        if let Some(v) = config.learning_rate {
            cmd.args(["--learning-rate", &v.to_string()]);
        }
        if let Some(v) = config.batch_size {
            cmd.args(["--batch-size", &v.to_string()]);
        }
        if let Some(v) = config.max_seq_len {
            cmd.args(["--max-seq-len", &v.to_string()]);
        }
        if let Some(v) = config.lora_rank {
            cmd.args(["--lora-r", &v.to_string()]);
        }
        if let Some(v) = config.lora_alpha {
            cmd.args(["--lora-alpha", &v.to_string()]);
        }
        if let Some(v) = config.lora_dropout {
            cmd.args(["--lora-dropout", &v.to_string()]);
        }
        if config.use_rslora == Some(true) {
            cmd.arg("--rslora");
        }
        if config.use_dora == Some(true) {
            cmd.arg("--dora");
        }
        if let Some(v) = config.gradient_accumulation_steps {
            cmd.args(["--gradient-accumulation-steps", &v.to_string()]);
        }
        if let Some(v) = config.gradient_checkpointing_layers {
            cmd.args(["--gradient-checkpointing-layers", &v.to_string()]);
        }
        if let Some(v) = config.weight_decay {
            cmd.args(["--weight-decay", &v.to_string()]);
        }
        if let Some(v) = config.max_grad_norm {
            cmd.args(["--max-grad-norm", &v.to_string()]);
        }
        if let Some(ref s) = config.lr_scheduler {
            cmd.args(["--lr-scheduler", s]);
        }
        if let Some(ref s) = config.text_column {
            cmd.args(["--text-column", s]);
        }
        if let Some(ref s) = config.dataset_format {
            cmd.args(["--dataset-format", s]);
        }
        if let Some(v) = config.embedding_lr {
            cmd.args(["--embedding-lr", &v.to_string()]);
        }
        if let Some(ref s) = config.resume_from {
            cmd.args(["--resume-from", s]);
        }
        // load_in_4bit → qlora style quantization
        if config.load_in_4bit == Some(true) {
            cmd.args(["--quantization", "nf4"]);
        }
        // Boolean negation flags
        if config.sequence_packing == Some(false) {
            cmd.arg("--no-sequence-packing");
        }
        if config.jit_compilation == Some(false) {
            cmd.arg("--no-jit-compilation");
        }
        if config.gradient_checkpointing == Some(false) {
            cmd.arg("--no-gradient-checkpointing");
        }
        if config.flash_attention == Some(false) {
            cmd.arg("--no-flash-attention");
        }
        if config.fused_optimizer == Some(false) {
            cmd.arg("--no-metal-fused-optimizer");
        }
        // Method-specific flags
        if let Some(v) = config.dpo_beta {
            cmd.args(["--beta", &v.to_string()]);
        }
        if let Some(ref s) = config.dpo_loss_type {
            cmd.args(["--dpo-loss-type", s]);
        }
        if let Some(ref s) = config.ref_model {
            cmd.args(["--ref-model", s]);
        }
        if let Some(v) = config.simpo_beta {
            cmd.args(["--beta", &v.to_string()]);
        }
        if let Some(v) = config.simpo_gamma {
            cmd.args(["--gamma", &v.to_string()]);
        }
        if let Some(v) = config.orpo_lambda {
            cmd.args(["--lambda", &v.to_string()]);
        }
        if let Some(v) = config.kto_desirable_weight {
            cmd.args(["--desirable-weight", &v.to_string()]);
        }
        if let Some(v) = config.kto_undesirable_weight {
            cmd.args(["--undesirable-weight", &v.to_string()]);
        }

        // Use a temp file for metrics instead of --metrics-format jsonl
        let metrics_path = std::env::temp_dir().join(format!("pmetal_train_{}.jsonl", run_id_task));
        cmd.args(["--log-metrics", &metrics_path.to_string_lossy()]);

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        match cmd.spawn() {
            Ok(child) => {
                spawn_and_monitor_training(
                    child,
                    run_id_task,
                    state_arc,
                    event_tx,
                    procs_arc,
                    cancel_flags,
                    cancel_flag,
                    app_handle,
                )
                .await;
            }
            Err(e) => {
                mark_training_failed(&state_arc, &event_tx, &run_id_task, &e.to_string()).await;
            }
        }
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
    app_handle: AppHandle,
    config: DistillationConfig,
) -> Result<String> {
    let pmetal_bin = which::which("pmetal")
        .map_err(|_| AppError("pmetal binary not found in PATH".to_string()))?;

    let temperature = config.temperature.unwrap_or(2.0) as f64;
    let loss_type = config.loss_type.clone().unwrap_or_else(|| "kl".to_string());
    let total_epochs = config.epochs.unwrap_or(3) as u64;
    let output_dir = config.output_dir.as_deref().unwrap_or("./output").to_string();

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
    let procs_arc = state.active_processes.clone();
    let cancel_flags = state.cancel_flags.clone();

    tokio::spawn(async move {
        let mut cmd = tokio::process::Command::new(&pmetal_bin);
        cmd.arg("distill");
        cmd.args(["--student", &config.student_model]);
        cmd.args(["--teacher", &config.teacher_model]);
        cmd.args(["--output", &output_dir]);
        cmd.args(["--temperature", &temperature.to_string()]);
        cmd.args(["--loss-type", &loss_type]);

        if let Some(ref ds) = config.dataset {
            cmd.args(["--dataset", ds]);
        }
        if let Some(v) = config.alpha {
            cmd.args(["--alpha", &v.to_string()]);
        }
        if let Some(v) = config.epochs {
            cmd.args(["--epochs", &v.to_string()]);
        }
        if let Some(v) = config.learning_rate {
            cmd.args(["--learning-rate", &v.to_string()]);
        }
        if let Some(v) = config.batch_size {
            cmd.args(["--batch-size", &v.to_string()]);
        }
        if let Some(v) = config.max_seq_len {
            cmd.args(["--max-seq-len", &v.to_string()]);
        }
        if let Some(v) = config.lora_rank {
            cmd.args(["--lora-r", &v.to_string()]);
        }
        if let Some(v) = config.lora_alpha {
            cmd.args(["--lora-alpha", &v.to_string()]);
        }

        let metrics_path = std::env::temp_dir()
            .join(format!("pmetal_distill_{}.jsonl", run_id_task));
        cmd.args(["--log-metrics", &metrics_path.to_string_lossy()]);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        match cmd.spawn() {
            Ok(child) => {
                spawn_and_monitor_distillation(
                    child,
                    run_id_task,
                    state_arc,
                    event_tx,
                    procs_arc,
                    cancel_flags,
                    cancel_flag,
                    app_handle,
                )
                .await;
            }
            Err(e) => {
                mark_distillation_failed(&state_arc, &event_tx, &run_id_task, &e.to_string())
                    .await;
            }
        }
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
    app_handle: AppHandle,
    config: GrpoConfig,
) -> Result<String> {
    let pmetal_bin = which::which("pmetal")
        .map_err(|_| AppError("pmetal binary not found in PATH".to_string()))?;

    let group_size = config.group_size.unwrap_or(8);
    let beta = config.beta.unwrap_or(0.04);
    let output_dir = config.output_dir.as_deref().unwrap_or("./output").to_string();

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
    let procs_arc = state.active_processes.clone();
    let cancel_flags = state.cancel_flags.clone();

    tokio::spawn(async move {
        let mut cmd = tokio::process::Command::new(&pmetal_bin);
        cmd.arg("grpo");
        cmd.args(["--model", &config.model]);
        cmd.args(["--output", &output_dir]);
        cmd.args(["--num-generations", &group_size.to_string()]);
        cmd.args(["--beta", &beta.to_string()]);

        if let Some(ref ds) = config.dataset {
            cmd.args(["--dataset", ds]);
        }
        if let Some(v) = config.epochs {
            cmd.args(["--epochs", &v.to_string()]);
        }
        if let Some(v) = config.learning_rate {
            cmd.args(["--learning-rate", &v.to_string()]);
        }
        if let Some(v) = config.lora_rank {
            cmd.args(["--lora-r", &v.to_string()]);
        }
        if let Some(v) = config.lora_alpha {
            cmd.args(["--lora-alpha", &v.to_string()]);
        }
        if let Some(v) = config.max_seq_len {
            cmd.args(["--max-seq-len", &v.to_string()]);
        }
        if config.use_reasoning_rewards == Some(true) {
            cmd.arg("--reasoning-rewards");
        }

        let metrics_path = std::env::temp_dir()
            .join(format!("pmetal_grpo_{}.jsonl", run_id_task));
        cmd.args(["--log-metrics", &metrics_path.to_string_lossy()]);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        match cmd.spawn() {
            Ok(child) => {
                spawn_and_monitor_grpo(
                    child,
                    run_id_task,
                    state_arc,
                    event_tx,
                    procs_arc,
                    cancel_flags,
                    cancel_flag,
                    app_handle,
                )
                .await;
            }
            Err(e) => {
                mark_grpo_failed(&state_arc, &event_tx, &run_id_task, &e.to_string()).await;
            }
        }
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
    let pmetal_bin = which::which("pmetal")
        .map_err(|_| AppError("pmetal binary not found in PATH".to_string()))?;

    let session_id = uuid::Uuid::new_v4().to_string();
    let session_id_task = session_id.clone();
    let procs_arc = state.active_processes.clone();

    tokio::spawn(async move {
        let mut cmd = tokio::process::Command::new(&pmetal_bin);
        cmd.arg("infer");
        cmd.args(["--model", &config.model]);
        cmd.args(["--prompt", &config.prompt]);
        cmd.arg("--chat");
        cmd.arg("--stream");

        if let Some(ref lora) = config.lora_path {
            cmd.args(["--lora", lora]);
        }
        if let Some(ref sys) = config.system_message {
            cmd.args(["--system", sys]);
        }
        if let Some(v) = config.temperature {
            cmd.args(["--temperature", &v.to_string()]);
        }
        if let Some(v) = config.top_k {
            cmd.args(["--top-k", &v.to_string()]);
        }
        if let Some(v) = config.top_p {
            cmd.args(["--top-p", &v.to_string()]);
        }
        if let Some(v) = config.max_tokens {
            cmd.args(["--max-tokens", &v.to_string()]);
        }
        if let Some(v) = config.repetition_penalty {
            cmd.args(["--repetition-penalty", &v.to_string()]);
        }

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take().unwrap();
                let ah = app_handle.clone();

                // Forward stderr as logs
                if let Some(stderr) = child.stderr.take() {
                    let ah2 = app_handle.clone();
                    let sid2 = session_id_task.clone();
                    tokio::spawn(async move {
                        let mut lines = BufReader::new(stderr).lines();
                        while let Ok(Some(line)) = lines.next_line().await {
                            let _ = ah2.emit(
                                "process-log",
                                serde_json::json!({ "run_id": sid2, "line": line }),
                            );
                        }
                    });
                }

                {
                    let mut procs = procs_arc.write().await;
                    procs.insert(session_id_task.clone(), child);
                }

                // Stream stdout tokens
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = ah.emit("inference-token", &line);
                }

                // Signal completion
                let _ = ah.emit("inference-done", ());

                // Remove from active processes
                procs_arc.write().await.remove(&session_id_task);
            }
            Err(e) => {
                let _ = app_handle.emit(
                    "inference-error",
                    serde_json::json!({ "session_id": session_id_task, "error": e.to_string() }),
                );
            }
        }
    });

    Ok(())
}

#[tauri::command]
pub async fn stop_inference(
    state: State<'_, AppState>,
    session_id: Option<String>,
) -> Result<()> {
    let mut procs = state.active_processes.write().await;
    if let Some(id) = session_id {
        if let Some(mut child) = procs.remove(&id) {
            let _ = child.kill().await;
        }
    } else {
        // Kill all inference sessions (no session_id provided)
        for (_, mut child) in procs.drain() {
            let _ = child.kill().await;
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
    // pmetal merge CLI is not yet fully implemented.
    // Log and return the output path so the frontend can track the operation.
    tracing::warn!(
        "merge_models called with strategy={} base={} output={} — pmetal merge pipeline not yet implemented",
        config.strategy,
        config.base_model,
        config.output
    );
    Ok(config.output)
}

#[tauri::command]
pub async fn fuse_lora(
    app_handle: AppHandle,
    base_model: String,
    lora_path: String,
    output_dir: String,
) -> Result<FuseResult> {
    let pmetal_bin = which::which("pmetal")
        .map_err(|_| AppError("pmetal binary not found in PATH".to_string()))?;

    let op_id = uuid::Uuid::new_v4().to_string();
    let op_id_task = op_id.clone();
    let output_dir_task = output_dir.clone();

    // Run synchronously so we can return FuseResult
    let mut cmd = tokio::process::Command::new(&pmetal_bin);
    cmd.arg("fuse");
    cmd.args(["--model", &base_model]);
    cmd.args(["--lora", &lora_path]);
    cmd.args(["--output", &output_dir]);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn()?;

    // Stream progress events
    if let Some(stdout) = child.stdout.take() {
        let ah = app_handle.clone();
        let oid = op_id_task.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = ah.emit(
                    "fuse-progress",
                    serde_json::json!({ "op_id": oid, "line": line }),
                );
            }
        });
    }

    let status = child.wait().await?;
    if !status.success() {
        return Err(AppError(format!(
            "fuse exited with status {}",
            status
        )));
    }

    // Compute output model size
    let output_path = PathBuf::from(&output_dir_task);
    let model_size_bytes = dir_size_simple(&output_path).await;

    Ok(FuseResult {
        output_dir: output_dir_task,
        model_size_bytes,
    })
}

#[tauri::command]
pub async fn quantize_model(
    app_handle: AppHandle,
    model_id: String,
    quant_type: String,
    output_dir: String,
) -> Result<String> {
    let pmetal_bin = which::which("pmetal")
        .map_err(|_| AppError("pmetal binary not found in PATH".to_string()))?;

    let op_id = uuid::Uuid::new_v4().to_string();
    let op_id_task = op_id.clone();
    let output_dir_task = output_dir.clone();

    let mut cmd = tokio::process::Command::new(&pmetal_bin);
    cmd.arg("quantize");
    cmd.args(["--model", &model_id]);
    cmd.args(["--method", &quant_type]);
    cmd.args(["--output", &output_dir]);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn()?;

    if let Some(stdout) = child.stdout.take() {
        let ah = app_handle.clone();
        let oid = op_id_task.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = ah.emit(
                    "quantize-progress",
                    serde_json::json!({ "op_id": oid, "line": line }),
                );
            }
        });
    }

    let status = child.wait().await?;
    if !status.success() {
        return Err(AppError(format!(
            "quantize exited with status {}",
            status
        )));
    }

    Ok(output_dir_task)
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
    if let Ok(pmetal_bin) = which::which("pmetal") {
        let out = tokio::process::Command::new(&pmetal_bin)
            .arg("--version")
            .output()
            .await
            .ok();
        if let Some(o) = out {
            if let Ok(s) = String::from_utf8(o.stdout) {
                return s.trim().to_string();
            }
        }
    }
    "unknown".to_string()
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
    let pmetal_bin = which::which("pmetal").ok()?;
    let out = tokio::process::Command::new(&pmetal_bin)
        .arg("memory")
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("Bandwidth:") {
            let val = rest.trim().split_whitespace().next()?;
            return val.parse::<f64>().ok();
        }
    }
    None
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
