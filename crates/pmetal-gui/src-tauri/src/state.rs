use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub cache_dir: String,
    pub hf_token: Option<String>,
    pub default_model: Option<String>,
    pub theme: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_hf_cache_dir().to_string_lossy().into_owned(),
            hf_token: None,
            default_model: None,
            theme: "dark".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cached model — field names match TS CachedModel interface
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModel {
    pub id: String,
    pub path: String,
    /// Raw byte count (serialised as `size`)
    pub size: u64,
    /// Human-readable string (serialised as `size_formatted`)
    pub size_formatted: String,
    pub downloaded_at: Option<DateTime<Utc>>,
    /// e.g. "text-generation", "embedding", "audio", "image"
    pub model_type: Option<String>,
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    pub id: String,
    pub status: TrainingStatus,
    pub model: String,
    pub method: String,
    pub dataset: Option<String>,
    pub epoch: f32,
    pub total_epochs: u32,
    pub step: u64,
    pub total_steps: u64,
    pub loss: Option<f64>,
    pub best_loss: Option<f64>,
    pub learning_rate: Option<f64>,
    pub grad_norm: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub eta_seconds: Option<u64>,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub output_dir: Option<String>,
    pub error_message: Option<String>,
    pub log_lines: Vec<String>,
}

impl TrainingRun {
    pub fn new(
        model: &str,
        method: &str,
        dataset: Option<&str>,
        output_dir: Option<&str>,
        total_epochs: u32,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            status: TrainingStatus::Pending,
            model: model.to_string(),
            method: method.to_string(),
            dataset: dataset.map(str::to_string),
            epoch: 0.0,
            total_epochs,
            step: 0,
            total_steps: 0,
            loss: None,
            best_loss: None,
            learning_rate: None,
            grad_norm: None,
            tokens_per_second: None,
            eta_seconds: None,
            started_at: Utc::now(),
            ended_at: None,
            output_dir: output_dir.map(str::to_string),
            error_message: None,
            log_lines: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Distillation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DistillationStatus {
    Pending,
    LoadingModels,
    GeneratingSignals,
    Training,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossBreakdown {
    pub ce_loss: Option<f64>,
    pub kl_loss: Option<f64>,
    pub cosine_loss: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationRun {
    pub id: String,
    pub status: DistillationStatus,
    pub student_model: String,
    pub teacher_model: String,
    pub dataset: Option<String>,
    pub loss_type: String,
    pub temperature: f64,
    pub epoch: u64,
    pub total_epochs: u64,
    pub step: u64,
    pub total_steps: Option<u64>,
    pub loss: Option<f64>,
    pub best_loss: Option<f64>,
    pub loss_breakdown: Option<LossBreakdown>,
    pub learning_rate: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub eta_seconds: Option<u64>,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub output_dir: Option<String>,
    pub error_message: Option<String>,
    pub log_lines: Vec<String>,
}

impl DistillationRun {
    pub fn new(
        student_model: &str,
        teacher_model: &str,
        dataset: Option<&str>,
        loss_type: &str,
        temperature: f64,
        total_epochs: u64,
        output_dir: Option<&str>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            status: DistillationStatus::Pending,
            student_model: student_model.to_string(),
            teacher_model: teacher_model.to_string(),
            dataset: dataset.map(str::to_string),
            loss_type: loss_type.to_string(),
            temperature,
            epoch: 0,
            total_epochs,
            step: 0,
            total_steps: None,
            loss: None,
            best_loss: None,
            loss_breakdown: None,
            learning_rate: None,
            tokens_per_second: None,
            eta_seconds: None,
            started_at: Utc::now(),
            ended_at: None,
            output_dir: output_dir.map(str::to_string),
            error_message: None,
            log_lines: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// GRPO
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GrpoStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoRun {
    pub id: String,
    pub status: GrpoStatus,
    pub model: String,
    pub dataset: Option<String>,
    pub group_size: u32,
    pub beta: f64,
    pub step: u64,
    pub total_steps: Option<u64>,
    pub reward_mean: Option<f64>,
    pub reward_std: Option<f64>,
    pub kl_div: Option<f64>,
    pub loss: Option<f64>,
    pub best_loss: Option<f64>,
    pub learning_rate: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub eta_seconds: Option<u64>,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub output_dir: Option<String>,
    pub error_message: Option<String>,
    pub log_lines: Vec<String>,
}

impl GrpoRun {
    pub fn new(
        model: &str,
        dataset: Option<&str>,
        group_size: u32,
        beta: f64,
        output_dir: Option<&str>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            status: GrpoStatus::Pending,
            model: model.to_string(),
            dataset: dataset.map(str::to_string),
            group_size,
            beta,
            step: 0,
            total_steps: None,
            reward_mean: None,
            reward_std: None,
            kl_div: None,
            loss: None,
            best_loss: None,
            learning_rate: None,
            tokens_per_second: None,
            eta_seconds: None,
            started_at: Utc::now(),
            ended_at: None,
            output_dir: output_dir.map(str::to_string),
            error_message: None,
            log_lines: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InferenceStatus {
    Idle,
    Running,
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSession {
    pub id: String,
    pub model: String,
    pub status: InferenceStatus,
    pub tokens_per_second: Option<f64>,
    pub started_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppEvent {
    TrainingStarted {
        run: TrainingRun,
    },
    TrainingStopped {
        run_id: String,
    },
    TrainingUpdate {
        run: TrainingRun,
    },
    DistillationStarted {
        run: DistillationRun,
    },
    DistillationStopped {
        run_id: String,
    },
    DistillationUpdate {
        run: DistillationRun,
    },
    GrpoStarted {
        run: GrpoRun,
    },
    GrpoStopped {
        run_id: String,
    },
    GrpoUpdate {
        run: GrpoRun,
    },
    ModelCached {
        model: CachedModel,
    },
    ModelRemoved {
        model_id: String,
    },
    ProcessLog {
        run_id: String,
        line: String,
    },
}

// ---------------------------------------------------------------------------
// AppState
// ---------------------------------------------------------------------------

pub struct AppState {
    pub config: Arc<RwLock<AppConfig>>,
    pub training_runs: Arc<RwLock<Vec<TrainingRun>>>,
    pub distillation_runs: Arc<RwLock<Vec<DistillationRun>>>,
    pub grpo_runs: Arc<RwLock<Vec<GrpoRun>>>,
    pub cached_models: Arc<RwLock<Vec<CachedModel>>>,
    pub event_tx: broadcast::Sender<AppEvent>,
    pub active_processes: Arc<RwLock<HashMap<String, tokio::process::Child>>>,
    /// Per-run cancellation flags (run_id → cancelled).
    pub cancel_flags: Arc<RwLock<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>>,
    /// Active inference sessions (session_id → cancelled).
    pub inference_cancel_flags:
        Arc<RwLock<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>>,
}

impl AppState {
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(512);
        Self {
            config: Arc::new(RwLock::new(AppConfig::default())),
            training_runs: Arc::new(RwLock::new(Vec::new())),
            distillation_runs: Arc::new(RwLock::new(Vec::new())),
            grpo_runs: Arc::new(RwLock::new(Vec::new())),
            cached_models: Arc::new(RwLock::new(Vec::new())),
            event_tx,
            active_processes: Arc::new(RwLock::new(HashMap::new())),
            cancel_flags: Arc::new(RwLock::new(HashMap::new())),
            inference_cancel_flags: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<AppEvent> {
        self.event_tx.subscribe()
    }

    // -----------------------------------------------------------------------
    // Config persistence
    // -----------------------------------------------------------------------

    fn config_path() -> PathBuf {
        Self::config_path_pub()
    }

    /// Public accessor so lib.rs initialisation tasks can use it without a full `AppState`.
    pub fn config_path_pub() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("pmetal")
            .join("config.json")
    }

    pub async fn load_config(&self) {
        let path = Self::config_path();
        if let Ok(data) = tokio::fs::read_to_string(&path).await {
            if let Ok(cfg) = serde_json::from_str::<AppConfig>(&data) {
                *self.config.write().await = cfg;
                tracing::info!("Loaded config from {}", path.display());
            }
        }
    }

    pub async fn save_config(&self) {
        let path = Self::config_path();
        let cfg = self.config.read().await.clone();
        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        match serde_json::to_string_pretty(&cfg) {
            Ok(data) => {
                if let Err(e) = tokio::fs::write(&path, data).await {
                    tracing::error!("Failed to save config: {}", e);
                }
            }
            Err(e) => tracing::error!("Failed to serialize config: {}", e),
        }
    }

    // -----------------------------------------------------------------------
    // Cache scanning
    // -----------------------------------------------------------------------

    pub async fn refresh_cached_models(&self) {
        let cache_root = {
            let cfg = self.config.read().await;
            PathBuf::from(&cfg.cache_dir)
        };

        let hub_models_dir = cache_root.join("hub");
        let models = scan_hub_cache(&hub_models_dir).await;
        *self.cached_models.write().await = models;
    }

    // -----------------------------------------------------------------------
    // Training CRUD
    // -----------------------------------------------------------------------

    pub async fn create_training_run(&self, run: TrainingRun) {
        let _ = self.event_tx.send(AppEvent::TrainingStarted { run: run.clone() });
        self.training_runs.write().await.push(run);
    }

    pub async fn update_training_run<F>(&self, id: &str, f: F)
    where
        F: FnOnce(&mut TrainingRun),
    {
        let mut runs = self.training_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            f(run);
            let _ = self.event_tx.send(AppEvent::TrainingUpdate { run: run.clone() });
        }
    }

    pub async fn get_training_run(&self, id: &str) -> Option<TrainingRun> {
        self.training_runs.read().await.iter().find(|r| r.id == id).cloned()
    }

    pub async fn list_training_runs(&self) -> Vec<TrainingRun> {
        self.training_runs.read().await.clone()
    }

    pub async fn cancel_training_run(&self, id: &str) -> bool {
        // Set cancellation flag first (avoids race with monitor task)
        {
            let flags = self.cancel_flags.read().await;
            if let Some(flag) = flags.get(id) {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        // Kill the process if still running
        {
            let mut procs = self.active_processes.write().await;
            if let Some(mut child) = procs.remove(id) {
                let _ = child.kill().await;
            }
        }
        // Mark as cancelled
        let mut found = false;
        let mut runs = self.training_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            if run.status == TrainingStatus::Running || run.status == TrainingStatus::Pending {
                run.status = TrainingStatus::Cancelled;
                run.ended_at = Some(Utc::now());
                let _ = self.event_tx.send(AppEvent::TrainingUpdate { run: run.clone() });
                let _ = self.event_tx.send(AppEvent::TrainingStopped { run_id: id.to_string() });
            }
            found = true;
        }
        found
    }

    // -----------------------------------------------------------------------
    // Distillation CRUD
    // -----------------------------------------------------------------------

    pub async fn create_distillation_run(&self, run: DistillationRun) {
        let _ = self.event_tx.send(AppEvent::DistillationStarted { run: run.clone() });
        self.distillation_runs.write().await.push(run);
    }

    pub async fn update_distillation_run<F>(&self, id: &str, f: F)
    where
        F: FnOnce(&mut DistillationRun),
    {
        let mut runs = self.distillation_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            f(run);
            let _ = self.event_tx.send(AppEvent::DistillationUpdate { run: run.clone() });
        }
    }

    pub async fn get_distillation_run(&self, id: &str) -> Option<DistillationRun> {
        self.distillation_runs.read().await.iter().find(|r| r.id == id).cloned()
    }

    pub async fn list_distillation_runs(&self) -> Vec<DistillationRun> {
        self.distillation_runs.read().await.clone()
    }

    pub async fn cancel_distillation_run(&self, id: &str) -> bool {
        {
            let flags = self.cancel_flags.read().await;
            if let Some(flag) = flags.get(id) {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        {
            let mut procs = self.active_processes.write().await;
            if let Some(mut child) = procs.remove(id) {
                let _ = child.kill().await;
            }
        }
        let mut found = false;
        let mut runs = self.distillation_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            if run.status != DistillationStatus::Completed
                && run.status != DistillationStatus::Failed
            {
                run.status = DistillationStatus::Cancelled;
                run.ended_at = Some(Utc::now());
                let _ = self.event_tx.send(AppEvent::DistillationUpdate { run: run.clone() });
                let _ = self.event_tx.send(AppEvent::DistillationStopped { run_id: id.to_string() });
            }
            found = true;
        }
        found
    }

    // -----------------------------------------------------------------------
    // GRPO CRUD
    // -----------------------------------------------------------------------

    pub async fn create_grpo_run(&self, run: GrpoRun) {
        let _ = self.event_tx.send(AppEvent::GrpoStarted { run: run.clone() });
        self.grpo_runs.write().await.push(run);
    }

    pub async fn update_grpo_run<F>(&self, id: &str, f: F)
    where
        F: FnOnce(&mut GrpoRun),
    {
        let mut runs = self.grpo_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            f(run);
            let _ = self.event_tx.send(AppEvent::GrpoUpdate { run: run.clone() });
        }
    }

    pub async fn get_grpo_run(&self, id: &str) -> Option<GrpoRun> {
        self.grpo_runs.read().await.iter().find(|r| r.id == id).cloned()
    }

    pub async fn list_grpo_runs(&self) -> Vec<GrpoRun> {
        self.grpo_runs.read().await.clone()
    }

    pub async fn cancel_grpo_run(&self, id: &str) -> bool {
        {
            let flags = self.cancel_flags.read().await;
            if let Some(flag) = flags.get(id) {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        {
            let mut procs = self.active_processes.write().await;
            if let Some(mut child) = procs.remove(id) {
                let _ = child.kill().await;
            }
        }
        let mut found = false;
        let mut runs = self.grpo_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            if run.status == GrpoStatus::Running || run.status == GrpoStatus::Pending {
                run.status = GrpoStatus::Cancelled;
                run.ended_at = Some(Utc::now());
                let _ = self.event_tx.send(AppEvent::GrpoUpdate { run: run.clone() });
                let _ = self.event_tx.send(AppEvent::GrpoStopped { run_id: id.to_string() });
            }
            found = true;
        }
        found
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the HuggingFace hub cache root, honouring the standard env vars in
/// priority order: HF_HOME > HUGGINGFACE_HUB_CACHE > HF_HUB_CACHE > ~/.cache/huggingface
pub fn default_hf_cache_dir() -> PathBuf {
    if let Ok(v) = std::env::var("HF_HOME") {
        return PathBuf::from(v);
    }
    if let Ok(v) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(v);
    }
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(v);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("huggingface")
}

/// Public alias used by `lib.rs` init tasks.
pub async fn scan_hub_cache_pub(hub_dir: &PathBuf) -> Vec<CachedModel> {
    scan_hub_cache(hub_dir).await
}

/// Walks `~/.cache/huggingface/hub` and returns a `CachedModel` entry for each
/// repo directory that looks like a downloaded model.
///
/// The HF hub cache layout is:
///   hub/models--{org}--{name}/
///     snapshots/{hash}/    ← these are symlinks into blobs/
async fn scan_hub_cache(hub_dir: &PathBuf) -> Vec<CachedModel> {
    let mut models = Vec::new();

    let mut read_dir = match tokio::fs::read_dir(hub_dir).await {
        Ok(rd) => rd,
        Err(_) => return models,
    };

    while let Ok(Some(entry)) = read_dir.next_entry().await {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Only look at model repos (not datasets/spaces)
        if !name_str.starts_with("models--") {
            continue;
        }

        let repo_path = entry.path();
        let model_id = name_str
            .strip_prefix("models--")
            .unwrap_or(&name_str)
            .replace("--", "/");

        let size = dir_size_follow_symlinks(&repo_path).await;
        let downloaded_at = entry
            .metadata()
            .await
            .ok()
            .and_then(|m| m.modified().ok())
            .map(DateTime::<Utc>::from);

        let model_type = infer_model_type(&model_id);

        models.push(CachedModel {
            id: model_id,
            path: repo_path.to_string_lossy().into_owned(),
            size,
            size_formatted: format_size(size),
            downloaded_at,
            model_type: Some(model_type),
        });
    }

    models.sort_by(|a, b| b.size.cmp(&a.size));
    models
}

/// Recursively compute directory size, following symlinks so that HF hub
/// snapshot symlinks resolve to the actual blob sizes.
async fn dir_size_follow_symlinks(path: &PathBuf) -> u64 {
    let mut total: u64 = 0;
    let mut stack = vec![path.clone()];

    while let Some(dir) = stack.pop() {
        let mut rd = match tokio::fs::read_dir(&dir).await {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let entry_path = entry.path();

            // Use symlink_metadata to detect symlinks, then resolve for size
            let symlink_meta = match tokio::fs::symlink_metadata(&entry_path).await {
                Ok(m) => m,
                Err(_) => continue,
            };

            if symlink_meta.file_type().is_symlink() {
                // Resolve symlink and use real file metadata
                if let Ok(real_meta) = tokio::fs::metadata(&entry_path).await {
                    if real_meta.is_file() {
                        total += real_meta.len();
                    } else if real_meta.is_dir() {
                        stack.push(entry_path);
                    }
                }
            } else if symlink_meta.is_file() {
                total += symlink_meta.len();
            } else if symlink_meta.is_dir() {
                stack.push(entry_path);
            }
        }
    }

    total
}

fn infer_model_type(model_id: &str) -> String {
    let lower = model_id.to_lowercase();
    if lower.contains("embed") {
        "embedding".to_string()
    } else if lower.contains("whisper") || lower.contains("wav2vec") || lower.contains("parakeet") {
        "audio".to_string()
    } else if lower.contains("flux") || lower.contains("stable-diffusion") || lower.contains("sdxl") {
        "image".to_string()
    } else {
        "text-generation".to_string()
    }
}

/// Format a byte count using 1024-based IEC units.
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1_024;
    const MB: u64 = KB * 1_024;
    const GB: u64 = MB * 1_024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format download counts for display: 1234 → "1.2K", 1234567 → "1.2M", etc.
pub fn format_downloads(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
