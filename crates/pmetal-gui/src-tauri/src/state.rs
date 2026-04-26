use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
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
    /// User-configured directories to scan for models (LM Studio, custom paths, etc.)
    #[serde(default)]
    pub custom_model_dirs: Vec<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_hf_cache_dir().to_string_lossy().into_owned(),
            hf_token: None,
            default_model: None,
            theme: "dark".to_string(),
            custom_model_dirs: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cached model — field names match TS CachedModel interface
// ---------------------------------------------------------------------------

/// Where a model was discovered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    /// Standard HuggingFace hub cache
    HfCache,
    /// Fine-tuned output directory
    Trained,
    /// User-added custom directory (LM Studio, manual download, etc.)
    Custom,
}

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
    /// Where this model was found
    #[serde(default = "default_source")]
    pub source: ModelSource,
}

fn default_source() -> ModelSource {
    ModelSource::HfCache
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

/// A condensed snapshot of the hyperparameters used for a training run, embedded in
/// `TrainingRun` so the frontend can display them without re-fetching the full config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfigSummary {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub lora_rank: Option<usize>,
    pub lora_alpha: Option<f32>,
    pub sequence_packing: bool,
    pub flash_attention: bool,
    pub jit_compilation: bool,
    pub gradient_checkpointing: bool,
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
    /// Human-readable description of the current setup phase (e.g. "Resolving dataset…").
    /// Cleared (set to `None`) once training steps start arriving.
    pub status_message: Option<String>,
    /// Snapshot of the training hyperparameters for display in the UI.
    pub config_summary: Option<TrainingConfigSummary>,
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
            status_message: None,
            config_summary: None,
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
        }
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InferenceStatus {
    Idle,
    Running,
    Stopped,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSession {
    pub id: String,
    pub model: String,
    pub status: InferenceStatus,
    pub tokens_per_second: Option<f64>,
    pub started_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Bench / Eval — one-shot measurement jobs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Single trial row parsed from `pmetal bench` stdout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchTrial {
    pub index: usize,
    pub prompt_tps: f64,
    pub generation_tps: f64,
    pub peak_memory_gb: f64,
}

/// A bench run tracked by the GUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRun {
    pub id: String,
    pub status: JobStatus,
    pub mode: String,
    pub model: String,
    pub preset: Option<String>,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub trials: Vec<BenchTrial>,
    pub error_message: Option<String>,
    #[serde(default)]
    pub log_tail: Vec<String>,
}

impl BenchRun {
    pub fn new(mode: &str, model: &str, preset: Option<&str>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            status: JobStatus::Running,
            mode: mode.to_string(),
            model: model.to_string(),
            preset: preset.map(str::to_string),
            started_at: Utc::now(),
            ended_at: None,
            trials: Vec::new(),
            error_message: None,
            log_tail: Vec::new(),
        }
    }

    pub fn append_log(&mut self, line: &str) {
        if let Some(trial) = parse_bench_trial_line(line) {
            if let Some(existing) = self.trials.iter_mut().find(|t| t.index == trial.index) {
                *existing = trial;
            } else {
                self.trials.push(trial);
            }
        }
        self.log_tail.push(line.to_string());
        if self.log_tail.len() > 300 {
            let drop = self.log_tail.len() - 300;
            self.log_tail.drain(..drop);
        }
    }
}

/// Snapshot of per-sample progress + final metrics from `pmetal eval`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvalMetrics {
    pub samples_done: usize,
    pub samples_total: usize,
    pub perplexity: Option<f64>,
    pub accuracy: Option<f64>,
    pub loss: Option<f64>,
}

/// An eval run tracked by the GUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRun {
    pub id: String,
    pub status: JobStatus,
    pub model: String,
    pub dataset: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub metrics: EvalMetrics,
    pub error_message: Option<String>,
    #[serde(default)]
    pub log_tail: Vec<String>,
}

impl EvalRun {
    pub fn new(model: &str, dataset: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            status: JobStatus::Running,
            model: model.to_string(),
            dataset: dataset.to_string(),
            started_at: Utc::now(),
            ended_at: None,
            metrics: EvalMetrics::default(),
            error_message: None,
            log_tail: Vec::new(),
        }
    }

    pub fn append_log(&mut self, line: &str) {
        if let Some((done, total)) = parse_eval_sample_progress(line) {
            self.metrics.samples_done = done;
            self.metrics.samples_total = total;
        }
        if let Some(v) = parse_eval_metric(line, "perplexity") {
            self.metrics.perplexity = Some(v);
        }
        if let Some(v) = parse_eval_metric(line, "accuracy") {
            self.metrics.accuracy = Some(v);
        }
        if let Some(v) = parse_eval_metric(line, "loss") {
            self.metrics.loss = Some(v);
        }
        self.log_tail.push(line.to_string());
        if self.log_tail.len() > 300 {
            let drop = self.log_tail.len() - 300;
            self.log_tail.drain(..drop);
        }
    }
}

/// `Trial 3: prompt_tps=512.4, generation_tps=102.1, peak_memory=9.23`
fn parse_bench_trial_line(line: &str) -> Option<BenchTrial> {
    let rest = line.trim_start().strip_prefix("Trial ")?;
    let (idx_part, rest) = rest.split_once(':')?;
    let index: usize = idx_part.trim().parse().ok()?;
    let prompt_tps = extract_kv_f64(rest, "prompt_tps")?;
    let generation_tps = extract_kv_f64(rest, "generation_tps")?;
    let peak_memory_gb = extract_kv_f64(rest, "peak_memory")?;
    Some(BenchTrial {
        index,
        prompt_tps,
        generation_tps,
        peak_memory_gb,
    })
}

fn extract_kv_f64(hay: &str, key: &str) -> Option<f64> {
    let pos = hay.find(key)?;
    let after = &hay[pos + key.len()..];
    let after = after.trim_start().strip_prefix('=')?.trim_start();
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e'))
        .unwrap_or(after.len());
    after[..end].parse::<f64>().ok()
}

fn parse_eval_sample_progress(line: &str) -> Option<(usize, usize)> {
    if let Some(rest) = line.trim_start().strip_prefix('[') {
        let end = rest.find(']')?;
        let slice = &rest[..end];
        let (a, b) = slice.split_once('/')?;
        return Some((a.trim().parse().ok()?, b.trim().parse().ok()?));
    }
    for (i, ch) in line.char_indices() {
        if ch.is_ascii_digit() {
            let tail = &line[i..];
            let end = tail.find([' ', ',']).unwrap_or(tail.len());
            let slice = &tail[..end];
            if let Some((a, b)) = slice.split_once('/') {
                if let (Ok(done), Ok(total)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    if total > 0 && done <= total {
                        return Some((done, total));
                    }
                }
            }
            break;
        }
    }
    None
}

fn parse_eval_metric(line: &str, key: &str) -> Option<f64> {
    let lower = line.to_lowercase();
    let idx = lower.find(key)?;
    let tail = &line[idx + key.len()..];
    let tail = tail.trim_start_matches([' ', '=', ':']);
    let end = tail
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e'))
        .unwrap_or(tail.len());
    tail[..end].parse::<f64>().ok()
}

// ---------------------------------------------------------------------------
// Pretrain — long-running subprocess spawned by `pmetal pretrain`
// ---------------------------------------------------------------------------

/// Step-level metrics parsed from `pmetal pretrain` stdout.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PretrainMetrics {
    pub step: usize,
    pub total_steps: usize,
    pub loss: Option<f64>,
    pub best_loss: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub learning_rate: Option<f64>,
    pub eta_seconds: Option<u64>,
}

/// A pretrain run tracked by the GUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainRun {
    pub id: String,
    pub status: JobStatus,
    pub arch: String,
    pub output_dir: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub metrics: PretrainMetrics,
    pub error_message: Option<String>,
    #[serde(default)]
    pub log_tail: Vec<String>,
}

impl PretrainRun {
    pub fn new(arch: &str, output_dir: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            status: JobStatus::Running,
            arch: arch.to_string(),
            output_dir: output_dir.to_string(),
            started_at: Utc::now(),
            ended_at: None,
            metrics: PretrainMetrics::default(),
            error_message: None,
            log_tail: Vec::new(),
        }
    }

    pub fn append_log(&mut self, line: &str) {
        // Parse step-level metrics from pretrain stdout lines.
        // Expected format: `step=N/Total loss=X.XXXX lr=X.Xe-X tok/s=XXX eta=Xs`
        if let Some(step) = extract_kv_usize(line, "step") {
            self.metrics.step = step;
        }
        if let Some(total) = parse_pretrain_total_steps(line) {
            self.metrics.total_steps = total;
        }
        if let Some(v) = extract_kv_f64(line, "loss") {
            if self.metrics.best_loss.is_none_or(|b| v < b) {
                self.metrics.best_loss = Some(v);
            }
            self.metrics.loss = Some(v);
        }
        if let Some(v) = extract_kv_f64(line, "lr") {
            self.metrics.learning_rate = Some(v);
        }
        if let Some(v) = extract_kv_f64(line, "tok/s") {
            self.metrics.tokens_per_second = Some(v);
        }
        if let Some(v) = extract_kv_usize(line, "eta") {
            self.metrics.eta_seconds = Some(v as u64);
        }
        self.log_tail.push(line.to_string());
        if self.log_tail.len() > 300 {
            let drop = self.log_tail.len() - 300;
            self.log_tail.drain(..drop);
        }
    }
}

fn extract_kv_usize(hay: &str, key: &str) -> Option<usize> {
    let pos = hay.find(key)?;
    let after = &hay[pos + key.len()..];
    let after = after.trim_start().strip_prefix('=')?.trim_start();
    let end = after
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after.len());
    after[..end].parse::<usize>().ok()
}

/// Parse `step=N/Total` — the total steps embedded after `/`.
fn parse_pretrain_total_steps(line: &str) -> Option<usize> {
    let pos = line.find("step=")?;
    let after = &line[pos + 5..];
    let slash = after.find('/')?;
    let rest = &after[slash + 1..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse::<usize>().ok()
}

// ---------------------------------------------------------------------------
// Serve
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ServeStatus {
    /// Binary spawned but HTTP listener hasn't come up yet.
    Starting,
    /// Server is accepting requests on `bind_url`.
    Running,
    /// Server exited cleanly (user stopped it or exit 0).
    Stopped,
    /// Process failed to start / died unexpectedly.
    Failed,
}

/// A running `pmetal serve` instance tracked by the GUI.
///
/// Frontend displays the active instance (if any) in the Serve page.
/// Stdout/stderr tails are pushed into `log_tail` (capped ring buffer)
/// so the operator gets the same live-log experience as the TUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServeInstance {
    pub id: String,
    pub status: ServeStatus,
    pub model: String,
    pub host: String,
    pub port: u16,
    pub bind_url: String,
    pub max_seq_len: usize,
    pub fp8: bool,
    pub kv_cache: String,
    pub started_at: DateTime<Utc>,
    pub ready_at: Option<DateTime<Utc>>,
    pub stopped_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub status_message: Option<String>,
    #[serde(default)]
    pub log_tail: Vec<String>,
}

impl ServeInstance {
    pub fn new(
        model: &str,
        host: &str,
        port: u16,
        max_seq_len: usize,
        fp8: bool,
        kv_cache: &str,
    ) -> Self {
        let display_host = if host == "0.0.0.0" { "localhost" } else { host };
        Self {
            id: Uuid::new_v4().to_string(),
            status: ServeStatus::Starting,
            model: model.to_string(),
            host: host.to_string(),
            port,
            bind_url: format!("http://{display_host}:{port}"),
            max_seq_len,
            fp8,
            kv_cache: kv_cache.to_string(),
            started_at: Utc::now(),
            ready_at: None,
            stopped_at: None,
            error_message: None,
            status_message: Some("Starting…".to_string()),
            log_tail: Vec::new(),
        }
    }

    /// Push a stdout/stderr line into the capped log tail. Detects the
    /// "Listening on …" banner so the status flips Starting → Running
    /// without a separate round-trip.
    pub fn append_log(&mut self, line: &str) {
        if matches!(self.status, ServeStatus::Starting) {
            let lower = line.to_lowercase();
            if lower.contains("listening") || lower.contains("serving") || lower.contains(" ready")
            {
                self.status = ServeStatus::Running;
                self.ready_at = Some(Utc::now());
                self.status_message = None;
            }
        }
        self.log_tail.push(line.to_string());
        if self.log_tail.len() > 200 {
            let drop = self.log_tail.len() - 200;
            self.log_tail.drain(..drop);
        }
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppEvent {
    TrainingStarted { run: TrainingRun },
    TrainingStopped { run_id: String },
    TrainingUpdate { run: TrainingRun },
    DistillationStarted { run: DistillationRun },
    DistillationStopped { run_id: String },
    DistillationUpdate { run: DistillationRun },
    GrpoStarted { run: GrpoRun },
    GrpoStopped { run_id: String },
    GrpoUpdate { run: GrpoRun },
    ServeStarted { instance: ServeInstance },
    ServeStopped { instance_id: String },
    ServeUpdate { instance: ServeInstance },
    BenchStarted { run: BenchRun },
    BenchStopped { run_id: String },
    BenchUpdate { run: BenchRun },
    EvalStarted { run: EvalRun },
    EvalStopped { run_id: String },
    EvalUpdate { run: EvalRun },
    PretrainStarted { run: PretrainRun },
    PretrainStopped { run_id: String },
    PretrainUpdate { run: PretrainRun },
    ModelCached { model: CachedModel },
    ModelRemoved { model_id: String },
    ProcessLog { run_id: String, line: String },
}

// ---------------------------------------------------------------------------
// AppState
// ---------------------------------------------------------------------------

pub struct AppState {
    pub config: Arc<RwLock<AppConfig>>,
    pub training_runs: Arc<RwLock<Vec<TrainingRun>>>,
    pub distillation_runs: Arc<RwLock<Vec<DistillationRun>>>,
    pub grpo_runs: Arc<RwLock<Vec<GrpoRun>>>,
    pub serve_instances: Arc<RwLock<Vec<ServeInstance>>>,
    pub bench_runs: Arc<RwLock<Vec<BenchRun>>>,
    pub eval_runs: Arc<RwLock<Vec<EvalRun>>>,
    pub pretrain_runs: Arc<RwLock<Vec<PretrainRun>>>,
    pub cached_models: Arc<RwLock<Vec<CachedModel>>>,
    pub event_tx: broadcast::Sender<AppEvent>,
    pub active_processes: Arc<RwLock<HashMap<String, tokio::process::Child>>>,
    /// Per-run cancellation flags (run_id → cancelled).
    pub cancel_flags: Arc<RwLock<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>>,
    /// Active inference sessions (session_id → cancelled).
    pub inference_cancel_flags: Arc<RwLock<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>>,
    /// Limits concurrent HF model downloads to 3 to avoid saturating the
    /// network connection and exhausting disk write bandwidth.
    pub download_semaphore: Arc<tokio::sync::Semaphore>,
}

#[allow(dead_code)]
impl AppState {
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(512);
        Self {
            config: Arc::new(RwLock::new(AppConfig::default())),
            training_runs: Arc::new(RwLock::new(Vec::new())),
            distillation_runs: Arc::new(RwLock::new(Vec::new())),
            grpo_runs: Arc::new(RwLock::new(Vec::new())),
            serve_instances: Arc::new(RwLock::new(Vec::new())),
            bench_runs: Arc::new(RwLock::new(Vec::new())),
            eval_runs: Arc::new(RwLock::new(Vec::new())),
            pretrain_runs: Arc::new(RwLock::new(Vec::new())),
            cached_models: Arc::new(RwLock::new(Vec::new())),
            event_tx,
            active_processes: Arc::new(RwLock::new(HashMap::new())),
            cancel_flags: Arc::new(RwLock::new(HashMap::new())),
            inference_cancel_flags: Arc::new(RwLock::new(HashMap::new())),
            download_semaphore: Arc::new(tokio::sync::Semaphore::new(3)),
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
        let (cache_root, custom_dirs) = {
            let cfg = self.config.read().await;
            (PathBuf::from(&cfg.cache_dir), cfg.custom_model_dirs.clone())
        };

        let mut models = Vec::new();

        // 1. Scan trained model outputs (./output/)
        scan_trained_outputs(&mut models).await;

        // 2. Scan HuggingFace hub cache
        let hub_models_dir = cache_root.join("hub");
        scan_hub_cache(&hub_models_dir, &mut models).await;

        // 3. Scan well-known third-party model directories
        scan_well_known_dirs(&mut models).await;

        // 4. Scan user-configured custom directories
        for dir in &custom_dirs {
            scan_custom_dir(&PathBuf::from(dir), &mut models).await;
        }

        models.sort_by(|a, b| {
            // Trained first, then by size descending
            let a_trained = a.source == ModelSource::Trained;
            let b_trained = b.source == ModelSource::Trained;
            b_trained.cmp(&a_trained).then(b.size.cmp(&a.size))
        });

        *self.cached_models.write().await = models;
    }

    // -----------------------------------------------------------------------
    // Training CRUD
    // -----------------------------------------------------------------------

    pub async fn create_training_run(&self, run: TrainingRun) {
        let _ = self
            .event_tx
            .send(AppEvent::TrainingStarted { run: run.clone() });
        self.training_runs.write().await.push(run);
    }

    pub async fn update_training_run<F>(&self, id: &str, f: F)
    where
        F: FnOnce(&mut TrainingRun),
    {
        let mut runs = self.training_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            f(run);
            let _ = self
                .event_tx
                .send(AppEvent::TrainingUpdate { run: run.clone() });
        }
    }

    pub async fn get_training_run(&self, id: &str) -> Option<TrainingRun> {
        self.training_runs
            .read()
            .await
            .iter()
            .find(|r| r.id == id)
            .cloned()
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
                let _ = self
                    .event_tx
                    .send(AppEvent::TrainingUpdate { run: run.clone() });
                let _ = self.event_tx.send(AppEvent::TrainingStopped {
                    run_id: id.to_string(),
                });
            }
            found = true;
        }
        found
    }

    // -----------------------------------------------------------------------
    // Distillation CRUD
    // -----------------------------------------------------------------------

    pub async fn create_distillation_run(&self, run: DistillationRun) {
        let _ = self
            .event_tx
            .send(AppEvent::DistillationStarted { run: run.clone() });
        self.distillation_runs.write().await.push(run);
    }

    pub async fn update_distillation_run<F>(&self, id: &str, f: F)
    where
        F: FnOnce(&mut DistillationRun),
    {
        let mut runs = self.distillation_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            f(run);
            let _ = self
                .event_tx
                .send(AppEvent::DistillationUpdate { run: run.clone() });
        }
    }

    pub async fn get_distillation_run(&self, id: &str) -> Option<DistillationRun> {
        self.distillation_runs
            .read()
            .await
            .iter()
            .find(|r| r.id == id)
            .cloned()
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
                let _ = self
                    .event_tx
                    .send(AppEvent::DistillationUpdate { run: run.clone() });
                let _ = self.event_tx.send(AppEvent::DistillationStopped {
                    run_id: id.to_string(),
                });
            }
            found = true;
        }
        found
    }

    // -----------------------------------------------------------------------
    // GRPO CRUD
    // -----------------------------------------------------------------------

    pub async fn create_grpo_run(&self, run: GrpoRun) {
        let _ = self
            .event_tx
            .send(AppEvent::GrpoStarted { run: run.clone() });
        self.grpo_runs.write().await.push(run);
    }

    pub async fn update_grpo_run<F>(&self, id: &str, f: F)
    where
        F: FnOnce(&mut GrpoRun),
    {
        let mut runs = self.grpo_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            f(run);
            let _ = self
                .event_tx
                .send(AppEvent::GrpoUpdate { run: run.clone() });
        }
    }

    pub async fn get_grpo_run(&self, id: &str) -> Option<GrpoRun> {
        self.grpo_runs
            .read()
            .await
            .iter()
            .find(|r| r.id == id)
            .cloned()
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
                let _ = self
                    .event_tx
                    .send(AppEvent::GrpoUpdate { run: run.clone() });
                let _ = self.event_tx.send(AppEvent::GrpoStopped {
                    run_id: id.to_string(),
                });
            }
            found = true;
        }
        found
    }

    // -----------------------------------------------------------------------
    // Serve CRUD
    // -----------------------------------------------------------------------

    pub async fn create_serve_instance(&self, instance: ServeInstance) {
        let _ = self.event_tx.send(AppEvent::ServeStarted {
            instance: instance.clone(),
        });
        self.serve_instances.write().await.push(instance);
    }

    pub async fn update_serve_instance<F>(&self, id: &str, f: F)
    where
        F: FnOnce(&mut ServeInstance),
    {
        let mut instances = self.serve_instances.write().await;
        if let Some(inst) = instances.iter_mut().find(|i| i.id == id) {
            f(inst);
            let _ = self.event_tx.send(AppEvent::ServeUpdate {
                instance: inst.clone(),
            });
        }
    }

    pub async fn list_serve_instances(&self) -> Vec<ServeInstance> {
        self.serve_instances.read().await.clone()
    }

    // -----------------------------------------------------------------------
    // Bench / Eval CRUD
    // -----------------------------------------------------------------------

    pub async fn create_bench_run(&self, run: BenchRun) {
        let _ = self
            .event_tx
            .send(AppEvent::BenchStarted { run: run.clone() });
        self.bench_runs.write().await.push(run);
    }

    pub async fn update_bench_run<F>(&self, id: &str, f: F)
    where
        F: FnOnce(&mut BenchRun),
    {
        let mut runs = self.bench_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            f(run);
            let _ = self
                .event_tx
                .send(AppEvent::BenchUpdate { run: run.clone() });
        }
    }

    pub async fn list_bench_runs(&self) -> Vec<BenchRun> {
        self.bench_runs.read().await.clone()
    }

    pub async fn cancel_bench_run(&self, id: &str) -> bool {
        {
            let mut procs = self.active_processes.write().await;
            if let Some(mut child) = procs.remove(id) {
                let _ = child.kill().await;
            }
        }
        let mut found = false;
        let mut runs = self.bench_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            if run.status == JobStatus::Running || run.status == JobStatus::Pending {
                run.status = JobStatus::Cancelled;
                run.ended_at = Some(Utc::now());
                let _ = self
                    .event_tx
                    .send(AppEvent::BenchUpdate { run: run.clone() });
                let _ = self.event_tx.send(AppEvent::BenchStopped {
                    run_id: id.to_string(),
                });
            }
            found = true;
        }
        found
    }

    pub async fn create_eval_run(&self, run: EvalRun) {
        let _ = self
            .event_tx
            .send(AppEvent::EvalStarted { run: run.clone() });
        self.eval_runs.write().await.push(run);
    }

    pub async fn update_eval_run<F>(&self, id: &str, f: F)
    where
        F: FnOnce(&mut EvalRun),
    {
        let mut runs = self.eval_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            f(run);
            let _ = self
                .event_tx
                .send(AppEvent::EvalUpdate { run: run.clone() });
        }
    }

    pub async fn list_eval_runs(&self) -> Vec<EvalRun> {
        self.eval_runs.read().await.clone()
    }

    pub async fn cancel_eval_run(&self, id: &str) -> bool {
        {
            let mut procs = self.active_processes.write().await;
            if let Some(mut child) = procs.remove(id) {
                let _ = child.kill().await;
            }
        }
        let mut found = false;
        let mut runs = self.eval_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            if run.status == JobStatus::Running || run.status == JobStatus::Pending {
                run.status = JobStatus::Cancelled;
                run.ended_at = Some(Utc::now());
                let _ = self
                    .event_tx
                    .send(AppEvent::EvalUpdate { run: run.clone() });
                let _ = self.event_tx.send(AppEvent::EvalStopped {
                    run_id: id.to_string(),
                });
            }
            found = true;
        }
        found
    }

    pub async fn create_pretrain_run(&self, run: PretrainRun) {
        let _ = self
            .event_tx
            .send(AppEvent::PretrainStarted { run: run.clone() });
        self.pretrain_runs.write().await.push(run);
    }

    pub async fn list_pretrain_runs(&self) -> Vec<PretrainRun> {
        self.pretrain_runs.read().await.clone()
    }

    pub async fn cancel_pretrain_run(&self, id: &str) -> bool {
        {
            let mut procs = self.active_processes.write().await;
            if let Some(mut child) = procs.remove(id) {
                let _ = child.kill().await;
            }
        }
        let mut found = false;
        let mut runs = self.pretrain_runs.write().await;
        if let Some(run) = runs.iter_mut().find(|r| r.id == id) {
            if run.status == JobStatus::Running || run.status == JobStatus::Pending {
                run.status = JobStatus::Cancelled;
                run.ended_at = Some(Utc::now());
                let _ = self
                    .event_tx
                    .send(AppEvent::PretrainUpdate { run: run.clone() });
                let _ = self.event_tx.send(AppEvent::PretrainStopped {
                    run_id: id.to_string(),
                });
            }
            found = true;
        }
        found
    }

    /// Stop a running serve instance: kill the child process, mark the
    /// instance `Stopped`, and broadcast the transition.
    pub async fn stop_serve_instance(&self, id: &str) -> bool {
        {
            let mut procs = self.active_processes.write().await;
            if let Some(mut child) = procs.remove(id) {
                let _ = child.kill().await;
            }
        }
        let mut found = false;
        let mut instances = self.serve_instances.write().await;
        if let Some(inst) = instances.iter_mut().find(|i| i.id == id) {
            if matches!(inst.status, ServeStatus::Starting | ServeStatus::Running) {
                inst.status = ServeStatus::Stopped;
                inst.stopped_at = Some(Utc::now());
                inst.status_message = Some("Stopped by user".to_string());
                let _ = self.event_tx.send(AppEvent::ServeUpdate {
                    instance: inst.clone(),
                });
                let _ = self.event_tx.send(AppEvent::ServeStopped {
                    instance_id: id.to_string(),
                });
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
    let mut models = Vec::new();
    scan_hub_cache(hub_dir, &mut models).await;
    models
}

/// Walks `~/.cache/huggingface/hub` and appends a `CachedModel` entry for each
/// repo directory that looks like a downloaded model.
///
/// The HF hub cache layout is:
///   hub/models--{org}--{name}/
///     snapshots/{hash}/    ← these are symlinks into blobs/
async fn scan_hub_cache(hub_dir: &PathBuf, models: &mut Vec<CachedModel>) {
    let mut read_dir = match tokio::fs::read_dir(hub_dir).await {
        Ok(rd) => rd,
        Err(_) => return,
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
            source: ModelSource::HfCache,
        });
    }
}

/// Scan `./output/` for fine-tuned model outputs.
async fn scan_trained_outputs(models: &mut Vec<CachedModel>) {
    let output_dir = PathBuf::from("./output");
    scan_model_subdir(&output_dir, ModelSource::Trained, models, 2).await;
}

/// Scan well-known third-party model directories (LM Studio, etc.)
async fn scan_well_known_dirs(models: &mut Vec<CachedModel>) {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));

    let well_known = [
        home.join(".lmstudio").join("models"),
        home.join(".cache").join("lm-studio").join("models"),
    ];

    for dir in &well_known {
        if dir.exists() {
            tracing::debug!(path = %dir.display(), "Scanning well-known model directory");
            scan_custom_dir(dir, models).await;
        }
    }
}

/// Scan a user-configured custom directory for model directories.
///
/// A directory is considered a model if it contains `config.json` or
/// `*.safetensors` files. Recursively scans one level of subdirectories.
async fn scan_custom_dir(dir: &PathBuf, models: &mut Vec<CachedModel>) {
    if !dir.exists() {
        return;
    }

    let seen: std::collections::HashSet<String> = models.iter().map(|m| m.path.clone()).collect();

    // Check if the directory itself is a model
    if is_model_dir(dir).await {
        let path_str = dir.to_string_lossy().into_owned();
        if !seen.contains(&path_str) {
            let source = if is_trained_dir(dir).await {
                ModelSource::Trained
            } else {
                ModelSource::Custom
            };
            if let Some(model) = build_model_from_dir(dir, source).await {
                models.push(model);
            }
        }
        return;
    }

    // Scan subdirectories (1 level deep)
    scan_model_subdir(dir, ModelSource::Custom, models, 1).await;
}

/// Recursively scan subdirectories for model dirs up to `max_depth` levels.
async fn scan_model_subdir(
    dir: &PathBuf,
    default_source: ModelSource,
    models: &mut Vec<CachedModel>,
    max_depth: usize,
) {
    if max_depth == 0 || !dir.exists() {
        return;
    }

    let seen: std::collections::HashSet<String> = models.iter().map(|m| m.path.clone()).collect();

    let mut read_dir = match tokio::fs::read_dir(dir).await {
        Ok(rd) => rd,
        Err(_) => return,
    };

    while let Ok(Some(entry)) = read_dir.next_entry().await {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let path_str = path.to_string_lossy().into_owned();
        if seen.contains(&path_str) {
            continue;
        }

        if is_model_dir(&path).await {
            let source = if is_trained_dir(&path).await {
                ModelSource::Trained
            } else {
                default_source.clone()
            };
            if let Some(model) = build_model_from_dir(&path, source).await {
                models.push(model);
            }
        } else if max_depth > 1 {
            // Recurse into subdirectories
            Box::pin(scan_model_subdir(
                &path,
                default_source.clone(),
                models,
                max_depth - 1,
            ))
            .await;
        }
    }
}

/// Check if a directory looks like a model directory.
async fn is_model_dir(dir: &PathBuf) -> bool {
    // Has config.json?
    if dir.join("config.json").exists() {
        return true;
    }
    // Has adapter_config.json (LoRA adapter)?
    if dir.join("adapter_config.json").exists() {
        return true;
    }
    // Has any .safetensors or .gguf files?
    if let Ok(mut rd) = tokio::fs::read_dir(dir).await {
        while let Ok(Some(entry)) = rd.next_entry().await {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".safetensors") || name_str.ends_with(".gguf") {
                return true;
            }
        }
    }
    false
}

/// Check if a directory is a trained/fine-tuned model output.
async fn is_trained_dir(dir: &Path) -> bool {
    dir.join("adapter_config.json").exists()
        || dir.join("lora_weights.safetensors").exists()
        || dir.join("training_state.json").exists()
}

/// Build a CachedModel from a directory path.
async fn build_model_from_dir(dir: &PathBuf, source: ModelSource) -> Option<CachedModel> {
    let dir_name = dir.file_name()?.to_string_lossy().to_string();

    // Try to get a better model ID from config.json
    let mut model_id = if let Ok(content) = tokio::fs::read_to_string(dir.join("config.json")).await
    {
        if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&content) {
            cfg["_name_or_path"]
                .as_str()
                .filter(|s| !s.is_empty() && !s.starts_with('/'))
                .map(str::to_string)
                .unwrap_or_else(|| dir_name.clone())
        } else {
            dir_name.clone()
        }
    } else {
        dir_name.clone()
    };

    // For GGUF-only directories without config.json: try extracting model name
    // from GGUF metadata and generate config.json for downstream consumers.
    if !dir.join("config.json").exists() {
        if let Some(gguf_path) = find_first_gguf(dir).await {
            if let Ok(content) = pmetal::gguf::GgufContent::from_file(&gguf_path) {
                // Use general.name as model ID if available
                if let Some(pmetal::gguf::MetadataValue::String(name)) =
                    content.get_metadata("general.name")
                {
                    if !name.is_empty() {
                        model_id = name.clone();
                    }
                }
                // Generate config.json from GGUF metadata
                pmetal::gguf::config::write_config_from_gguf(&content, dir);
            }
        }
    }

    let size = dir_size_follow_symlinks(dir).await;
    let downloaded_at = tokio::fs::metadata(dir)
        .await
        .ok()
        .and_then(|m| m.modified().ok())
        .map(DateTime::<Utc>::from);

    let model_type = infer_model_type(&model_id);

    Some(CachedModel {
        id: model_id,
        path: dir.to_string_lossy().into_owned(),
        size,
        size_formatted: format_size(size),
        downloaded_at,
        model_type: Some(model_type),
        source,
    })
}

/// Find the first .gguf file in a directory.
async fn find_first_gguf(dir: &PathBuf) -> Option<PathBuf> {
    let mut rd = tokio::fs::read_dir(dir).await.ok()?;
    while let Ok(Some(entry)) = rd.next_entry().await {
        let name = entry.file_name();
        if name.to_string_lossy().ends_with(".gguf") {
            return Some(entry.path());
        }
    }
    None
}

/// Recursively compute directory size, following symlinks so that HF hub
/// snapshot symlinks resolve to the actual blob sizes.
async fn dir_size_follow_symlinks(path: &Path) -> u64 {
    let mut total: u64 = 0;
    let mut stack = vec![path.to_path_buf()];

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
    } else if lower.contains("flux") || lower.contains("stable-diffusion") || lower.contains("sdxl")
    {
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
