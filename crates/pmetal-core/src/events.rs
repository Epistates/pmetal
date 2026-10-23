//! Job events — the canonical streaming protocol used across all pmetal
//! user-facing surfaces (CLI, TUI, GUI, MCP).
//!
//! Every long-running pmetal operation (training, distillation, GRPO,
//! inference, serving, quantization, …) produces a sequence of [`JobEvent`]s.
//! The same events flow through:
//!
//! - In-process callbacks via [`JobEventSink`] (TUI, GUI direct path).
//! - Tokio mpsc / broadcast channels (each surface defines its own sink).
//! - JSONL on stdout for subprocess fallback (CLI piped to MCP/TUI).
//! - `tauri::ipc::Channel<JobEvent>` for the GUI.
//!
//! Aggregated state is tracked by [`JobStatus<R>`]; per-surface result types
//! plug in via the `R` parameter.
//!
//! `pmetal-core` keeps no async runtime or tauri dependency; concrete sinks
//! that need tokio/tauri live in their respective surface crates and are wired
//! up through the trait below.

use crate::types::lenient_f64;
use crate::{EvalMetrics, StepMetrics};
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// JobKind — what kind of job this event stream describes
// ---------------------------------------------------------------------------

/// Identifies the high-level pmetal job type producing an event stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    /// Supervised fine-tuning / LoRA / QLoRA.
    Train,
    /// Gemma 4 assistant or Qwen3-Next MTP predictor training.
    TrainMtp,
    /// DFlash block-diffusion draft model training.
    TrainDraft,
    /// DiffusionGemma block-diffusion LoRA fine-tuning.
    TrainDiffusion,
    /// Knowledge distillation.
    Distill,
    /// Group Relative Policy Optimization.
    Grpo,
    /// Reinforcement Learning with Knowledge Distillation.
    Rlkd,
    /// Offline preference optimization (DPO, IPO, hinge, SimPO, ORPO, KTO).
    Preference,
    /// Embedding model training.
    EmbedTrain,
    /// Full-parameter pretraining.
    Pretrain,
    /// OpenAI-compatible inference server.
    Serve,
    /// One-shot inference / generation.
    Infer,
    /// Typed decisions from a decision model (`/v1/systemone`).
    Decide,
    /// Benchmarking sweep.
    Bench,
    /// Evaluation / perplexity / accuracy.
    Eval,
    /// Weight quantization (GGUF / MLX).
    Quantize,
    /// Importance-matrix measurement on calibration data.
    Imatrix,
    /// LoRA / model merging.
    Merge,
    /// LoRA fusion into base weights.
    Fuse,
    /// Speculative decoding setup.
    Dflash,
    /// Pack routed MoE experts into a single shard file.
    PackExperts,
    /// Tokenize text against a model's tokenizer.
    Tokenize,
    /// Synthesize a waveform from a mel spectrogram.
    Vocode,
}

// ---------------------------------------------------------------------------
// Phase — coarse pipeline phase (superset of TrainingPhase)
// ---------------------------------------------------------------------------

/// Coarse pipeline phase reported via [`JobEvent::Phase`].
///
/// Superset of `pmetal_trainer::TrainingPhase` plus phases unique to other
/// jobs (quantization calibration, merge shard writes, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum Phase {
    /// Resolving the model identifier (lookup or download).
    ResolvingModel,
    /// Resolving the dataset identifier.
    ResolvingDataset,
    /// Loading the tokenizer.
    LoadingTokenizer,
    /// Tokenizing the dataset.
    TokenizingDataset,
    /// Loading the model into memory.
    LoadingModel,
    /// Compiling Apple Neural Engine kernels.
    CompilingAneKernels,
    /// ANE attempted but fell back to GPU; carries the reason.
    AneFallback {
        /// Why ANE was unavailable (e.g. an unsupported architecture).
        reason: String,
    },
    /// Training loop is running.
    Training,
    /// Distilling teacher logits.
    Distilling,
    /// Computing offline teacher logits cache.
    GeneratingTeacherLogits,
    /// Calibrating quantization (importance matrix, KL).
    Calibrating,
    /// Quantizing weights.
    Quantizing,
    /// Merging shards.
    MergingShards,
    /// Compiling JIT graphs / fused kernels.
    Compiling,
    /// Saving weights / checkpoint.
    SavingWeights,
    /// Server bound and ready for requests.
    ServerReady {
        /// Bound socket address (e.g. `0.0.0.0:8080`).
        address: String,
    },
    /// Job is running but in a brief idle period (waiting for input).
    Idle,
    /// Job finished successfully.
    Complete,
    /// Job failed; mirrors [`JobEvent::Failed`] but kept for finer status.
    Failed {
        /// Failure reason.
        reason: String,
    },
    /// Surface-specific phase escape hatch.
    Custom {
        /// Snake-case phase identifier.
        name: String,
    },
}

impl Phase {
    /// Human-readable status string suitable for status bars.
    pub fn message(&self) -> String {
        match self {
            Self::ResolvingModel => "Resolving model…".into(),
            Self::ResolvingDataset => "Resolving dataset…".into(),
            Self::LoadingTokenizer => "Loading tokenizer…".into(),
            Self::TokenizingDataset => "Tokenizing dataset…".into(),
            Self::LoadingModel => "Loading model…".into(),
            Self::CompilingAneKernels => "Compiling ANE kernels…".into(),
            Self::AneFallback { .. } => "ANE unavailable, using GPU…".into(),
            Self::Training => "Training…".into(),
            Self::Distilling => "Distilling…".into(),
            Self::GeneratingTeacherLogits => "Generating teacher logits…".into(),
            Self::Calibrating => "Calibrating…".into(),
            Self::Quantizing => "Quantizing…".into(),
            Self::MergingShards => "Merging shards…".into(),
            Self::Compiling => "Compiling…".into(),
            Self::SavingWeights => "Saving weights…".into(),
            Self::ServerReady { address } => format!("Listening on {address}"),
            Self::Idle => "Idle".into(),
            Self::Complete => "Complete".into(),
            Self::Failed { reason } => format!("Failed: {reason}"),
            Self::Custom { name } => name.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// MetricPayload — per-step / per-trial / per-request typed metric
// ---------------------------------------------------------------------------

/// Per-event metric payload, typed per job-family.
///
/// Strongly typed (no `serde_json::Value` blobs) so dashboard renderers can
/// pattern-match the variant they care about. Add a new variant when adding
/// a new metric category; legacy consumers keep working because the enum is
/// `#[non_exhaustive]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MetricPayload {
    /// Per-step training metrics (training, distill, GRPO, RLKD, embed, pretrain).
    Step(StepMetrics),
    /// Periodic evaluation metrics.
    Eval(EvalMetrics),
    /// One bench trial result.
    BenchTrial(BenchTrialMetrics),
    /// One serve request completed.
    ServeRequest(ServeRequestMetrics),
    /// Inference run completed.
    InferenceDone {
        /// Number of generated tokens.
        tokens: u32,
        /// Tokens per second over the full run.
        #[serde(deserialize_with = "lenient_f64::deserialize")]
        tok_per_sec: f64,
        /// Time-to-first-token in milliseconds.
        #[serde(deserialize_with = "lenient_f64::deserialize")]
        ttft_ms: f64,
    },
}

/// One bench-trial measurement.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchTrialMetrics {
    /// Trial identifier (e.g. `"prefill_1024"`).
    pub label: String,
    /// Total tokens processed in the trial.
    pub tokens: u64,
    /// Throughput in tokens/sec.
    #[serde(deserialize_with = "lenient_f64::deserialize")]
    pub tok_per_sec: f64,
    /// Median latency in ms (per-token decode, per-batch prefill).
    #[serde(deserialize_with = "lenient_f64::deserialize")]
    pub p50_ms: f64,
    /// 99th-percentile latency in ms.
    #[serde(deserialize_with = "lenient_f64::deserialize")]
    pub p99_ms: f64,
}

/// One serve-request observation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServeRequestMetrics {
    /// HTTP route (e.g. `/v1/chat/completions`).
    pub route: String,
    /// HTTP status code.
    pub status: u16,
    /// Total request latency in ms.
    #[serde(deserialize_with = "lenient_f64::deserialize")]
    pub latency_ms: f64,
    /// Number of generated tokens, if applicable.
    pub tokens: Option<u32>,
}

// ---------------------------------------------------------------------------
// Progress, LogLevel, CompletionSummary
// ---------------------------------------------------------------------------

/// Discrete progress update (e.g. one of N tensors quantized).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Progress {
    /// Current count.
    pub current: u64,
    /// Total expected count, if known.
    pub total: Option<u64>,
    /// Free-form label for the units (e.g. `"tensors"`, `"trials"`).
    pub label: String,
}

/// Log level for [`JobEvent::Log`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Diagnostic information.
    Trace,
    /// Verbose information.
    Debug,
    /// Routine information.
    Info,
    /// Recoverable issue.
    Warn,
    /// Failure.
    Error,
}

/// Summary attached to [`JobEvent::Completed`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompletionSummary {
    /// Final loss for training-like jobs.
    pub final_loss: Option<f64>,
    /// Total steps executed (training-like).
    pub total_steps: Option<u64>,
    /// Total tokens processed (training/inference).
    pub total_tokens: Option<u64>,
    /// Output artifact path (model dir, GGUF file, merged adapter, …).
    pub output_path: Option<String>,
    /// Free-form key/value pairs the surface may want to display.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub extra: std::collections::BTreeMap<String, String>,
}

// ---------------------------------------------------------------------------
// JobEvent — the canonical event type
// ---------------------------------------------------------------------------

/// Unix-seconds timestamp (UTC). Defaults to `SystemTime::now()` at construction.
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Canonical event emitted by every pmetal long-running operation.
///
/// Tagged via `#[serde(tag = "event")]`, so JSONL on the wire looks like:
///
/// ```json
/// {"event":"started","job_id":"abc","kind":"train","ts":1700000000}
/// {"event":"phase","job_id":"abc","phase":{"phase":"training"}}
/// {"event":"metric","job_id":"abc","payload":{"kind":"step","step":1,"loss":2.31, ...}}
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum JobEvent {
    /// Job started.
    Started {
        /// Stable job identifier.
        job_id: String,
        /// What kind of job this is.
        kind: JobKind,
        /// Unix timestamp (seconds) when the job started.
        #[serde(default = "now_unix")]
        ts: u64,
    },
    /// Job entered a new pipeline phase.
    Phase {
        /// Stable job identifier.
        job_id: String,
        /// New phase.
        phase: Phase,
    },
    /// Job emitted a metric measurement.
    Metric {
        /// Stable job identifier.
        job_id: String,
        /// The metric payload.
        payload: MetricPayload,
    },
    /// Job streamed a single token (inference, GRPO rollouts).
    Token {
        /// Stable job identifier.
        job_id: String,
        /// The decoded token text (UTF-8 safe boundary).
        token: String,
        /// Absolute index of this token within the response.
        index: u32,
    },
    /// Job emitted a log line.
    Log {
        /// Stable job identifier.
        job_id: String,
        /// Log level.
        level: LogLevel,
        /// Free-form line (no trailing newline).
        line: String,
    },
    /// Job reported discrete progress.
    Progress {
        /// Stable job identifier.
        job_id: String,
        /// Progress observation.
        progress: Progress,
    },
    /// Job completed successfully.
    Completed {
        /// Stable job identifier.
        job_id: String,
        /// Completion summary.
        summary: CompletionSummary,
        /// Unix timestamp (seconds) when the job finished.
        #[serde(default = "now_unix")]
        ts: u64,
    },
    /// Job failed.
    Failed {
        /// Stable job identifier.
        job_id: String,
        /// Human-readable failure message.
        error: String,
        /// Unix timestamp (seconds) when the job finished.
        #[serde(default = "now_unix")]
        ts: u64,
    },
    /// Job was cancelled (by user or callback).
    Cancelled {
        /// Stable job identifier.
        job_id: String,
        /// Unix timestamp (seconds) when the job was cancelled.
        #[serde(default = "now_unix")]
        ts: u64,
    },
}

impl JobEvent {
    /// Convenience constructor for the `Started` variant.
    pub fn started(job_id: impl Into<String>, kind: JobKind) -> Self {
        Self::Started {
            job_id: job_id.into(),
            kind,
            ts: now_unix(),
        }
    }

    /// Convenience constructor for the `Failed` variant.
    pub fn failed(job_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self::Failed {
            job_id: job_id.into(),
            error: error.into(),
            ts: now_unix(),
        }
    }

    /// Job identifier this event belongs to.
    pub fn job_id(&self) -> &str {
        match self {
            Self::Started { job_id, .. }
            | Self::Phase { job_id, .. }
            | Self::Metric { job_id, .. }
            | Self::Token { job_id, .. }
            | Self::Log { job_id, .. }
            | Self::Progress { job_id, .. }
            | Self::Completed { job_id, .. }
            | Self::Failed { job_id, .. }
            | Self::Cancelled { job_id, .. } => job_id,
        }
    }
}

// ---------------------------------------------------------------------------
// JobStatus<R> — aggregated state derived from JobEvent stream
// ---------------------------------------------------------------------------

/// Aggregated job state, parameterized over the per-surface result type.
///
/// Drive transitions via [`JobStatus::apply_event`] so every surface advances
/// state the same way. Replaces all per-tab `*Status` enums.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JobStatus<R = ()> {
    /// Job has not started.
    #[default]
    Idle,
    /// Job is starting up; carries the current [`Phase`].
    Starting {
        /// Current phase.
        phase: Phase,
    },
    /// Job is running; carries the latest [`Progress`] and [`MetricPayload`].
    Running {
        /// Latest progress observation.
        progress: Option<Progress>,
        /// Latest metric payload.
        last_metric: Option<MetricPayload>,
    },
    /// Job completed successfully.
    Completed {
        /// Surface-specific result.
        result: R,
    },
    /// Job failed.
    Failed {
        /// Failure message.
        error: String,
    },
    /// Job was cancelled.
    Cancelled,
}

impl<R> JobStatus<R> {
    /// Apply an event to advance the status state machine.
    ///
    /// Caller is responsible for materialising `R` on `Completed` (this method
    /// only flips to `Failed` / `Cancelled` automatically; success transitions
    /// to a typed result are surface-specific).
    pub fn apply_event(&mut self, event: &JobEvent) {
        match event {
            JobEvent::Started { .. } => {
                *self = Self::Starting {
                    phase: Phase::ResolvingModel,
                };
            }
            JobEvent::Phase { phase, .. } => {
                *self = Self::Starting {
                    phase: phase.clone(),
                };
            }
            JobEvent::Metric { payload, .. } => match self {
                Self::Running { last_metric, .. } => {
                    *last_metric = Some(payload.clone());
                }
                _ => {
                    *self = Self::Running {
                        progress: None,
                        last_metric: Some(payload.clone()),
                    };
                }
            },
            JobEvent::Progress { progress, .. } => match self {
                Self::Running { progress: p, .. } => {
                    *p = Some(progress.clone());
                }
                _ => {
                    *self = Self::Running {
                        progress: Some(progress.clone()),
                        last_metric: None,
                    };
                }
            },
            JobEvent::Failed { error, .. } => {
                *self = Self::Failed {
                    error: error.clone(),
                };
            }
            JobEvent::Cancelled { .. } => {
                *self = Self::Cancelled;
            }
            // Token, Log, and Completed (which carries no `R` here) are
            // surface-driven; callers materialise `R` themselves.
            JobEvent::Token { .. } | JobEvent::Log { .. } | JobEvent::Completed { .. } => {}
        }
    }

    /// [`apply_event`](Self::apply_event), and on [`JobEvent::Completed`] flip
    /// to `Completed` with the result `on_completed` builds from its summary,
    /// so a surface does not have to watch for the event itself.
    pub fn apply_event_with(
        &mut self,
        event: &JobEvent,
        on_completed: impl FnOnce(&CompletionSummary) -> R,
    ) {
        match event {
            JobEvent::Completed { summary, .. } => self.complete(on_completed(summary)),
            other => self.apply_event(other),
        }
    }

    /// Flip to `Completed { result }`. Use after observing [`JobEvent::Completed`]
    /// when the surface has constructed its `R`.
    pub fn complete(&mut self, result: R) {
        *self = Self::Completed { result };
    }
}

// ---------------------------------------------------------------------------
// JobEventSink — the trait every transport implements
// ---------------------------------------------------------------------------

/// Receiver of [`JobEvent`]s. Implementors handle delivery (channel send,
/// JSONL write, Tauri ipc emit, …).
pub trait JobEventSink: Send + Sync {
    /// Emit one event. Implementations must not block on this hot path.
    fn emit(&self, event: JobEvent);

    /// Cooperative cancellation hook. Producers (training callbacks, inference
    /// loops) check this between steps and stop cleanly when it returns `true`.
    fn is_cancelled(&self) -> bool {
        false
    }
}

// Trivial blanket so `Arc<dyn JobEventSink>` and `Box<dyn JobEventSink>` work
// transparently.
impl<S: JobEventSink + ?Sized> JobEventSink for std::sync::Arc<S> {
    fn emit(&self, event: JobEvent) {
        (**self).emit(event);
    }
    fn is_cancelled(&self) -> bool {
        (**self).is_cancelled()
    }
}

impl<S: JobEventSink + ?Sized> JobEventSink for Box<S> {
    fn emit(&self, event: JobEvent) {
        (**self).emit(event);
    }
    fn is_cancelled(&self) -> bool {
        (**self).is_cancelled()
    }
}

// ---------------------------------------------------------------------------
// Concrete sinks bundled with pmetal-core (no async-runtime dependency)
// ---------------------------------------------------------------------------

/// Sink that writes JSONL to any [`Write`] target (file, stdout, pipe).
///
/// Used by the CLI's `--log-events <path>` flag and by subprocess fallbacks
/// in TUI/MCP. The matching reader is [`parse_event`].
///
/// [`JobEventSink::emit`] cannot fail, so a write that does (a full disk, a
/// closed pipe) is kept instead of lost: [`take_error`](Self::take_error)
/// returns the first one, for the caller to report when the job ends.
pub struct JsonlSink<W: Write + Send + Sync> {
    inner: std::sync::Mutex<W>,
    first_error: std::sync::Mutex<Option<io::Error>>,
}

impl<W: Write + Send + Sync> JsonlSink<W> {
    /// Wrap a writer.
    pub fn new(writer: W) -> Self {
        Self {
            inner: std::sync::Mutex::new(writer),
            first_error: std::sync::Mutex::new(None),
        }
    }

    /// The first error a write returned, if any, leaving none behind.
    pub fn take_error(&self) -> Option<io::Error> {
        self.first_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }
}

impl<W: Write + Send + Sync> JobEventSink for JsonlSink<W> {
    fn emit(&self, event: JobEvent) {
        // A writer that panicked mid-event leaves a usable writer behind.
        let mut w = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(e) = write_event(&mut *w, &event) {
            self.first_error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get_or_insert(e);
        }
    }
}

/// Sink that discards every event. Useful as a default in tests and
/// in surfaces that haven't wired up streaming yet.
pub struct NullSink;

impl JobEventSink for NullSink {
    fn emit(&self, _: JobEvent) {}
}

/// Sink that fans out to multiple inner sinks.
///
/// Used by surfaces that need both a live channel (for the UI) and a JSONL log
/// file. Cancellation is `true` if **any** inner sink reports cancelled.
pub struct BroadcastSink {
    sinks: Vec<std::sync::Arc<dyn JobEventSink>>,
}

impl BroadcastSink {
    /// Create a fan-out sink.
    pub fn new(sinks: Vec<std::sync::Arc<dyn JobEventSink>>) -> Self {
        Self { sinks }
    }
}

impl JobEventSink for BroadcastSink {
    fn emit(&self, event: JobEvent) {
        for sink in &self.sinks {
            sink.emit(event.clone());
        }
    }
    fn is_cancelled(&self) -> bool {
        self.sinks.iter().any(|s| s.is_cancelled())
    }
}

// ---------------------------------------------------------------------------
// JSONL codec
// ---------------------------------------------------------------------------

/// Write one event as a single-line JSON record terminated by `\n`.
///
/// The record is assembled first and handed to the writer in one `write_all`,
/// so an unbuffered file or pipe sees one system call per event rather than
/// one per token of JSON, and a reader tailing the stream never sees half a
/// line.
pub fn write_event(w: &mut impl Write, event: &JobEvent) -> io::Result<()> {
    let mut record =
        serde_json::to_vec(event).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    record.push(b'\n');
    w.write_all(&record)
}

/// Parse one event from a single line of JSON.
pub fn parse_event(line: &str) -> Result<JobEvent, ParseError> {
    serde_json::from_str(line.trim()).map_err(ParseError)
}

/// Error returned by [`parse_event`].
#[derive(Debug, thiserror::Error)]
#[error("failed to parse JobEvent JSONL line: {0}")]
pub struct ParseError(#[source] serde_json::Error);

// ---------------------------------------------------------------------------
// Adapter: existing TrainingCallback → JobEventSink
// ---------------------------------------------------------------------------

/// Adapter that lets any [`JobEventSink`] satisfy the legacy
/// [`crate::TrainingCallback`] trait.
///
/// Trainers that already accept a `&mut dyn TrainingCallback` keep working; new
/// surfaces wire them to a `JobEventSink` with this wrapper. Eventually
/// `TrainingCallback` will be deprecated in favor of pure event emission.
pub struct TrainingCallbackToSink<S: JobEventSink> {
    sink: S,
    job_id: String,
}

impl<S: JobEventSink> TrainingCallbackToSink<S> {
    /// Wrap a sink with a training-callback adapter.
    pub fn new(job_id: impl Into<String>, sink: S) -> Self {
        Self {
            sink,
            job_id: job_id.into(),
        }
    }

    /// Borrow the underlying sink.
    pub fn sink(&self) -> &S {
        &self.sink
    }
}

impl<S: JobEventSink> crate::TrainingCallback for TrainingCallbackToSink<S> {
    fn on_train_start(&mut self) {
        self.sink
            .emit(JobEvent::started(&self.job_id, JobKind::Train));
    }

    fn on_train_end(&mut self) {
        self.sink.emit(JobEvent::Phase {
            job_id: self.job_id.clone(),
            phase: Phase::Complete,
        });
    }

    fn on_step_end_with_metrics(&mut self, metrics: &StepMetrics) {
        self.sink.emit(JobEvent::Metric {
            job_id: self.job_id.clone(),
            payload: MetricPayload::Step(metrics.clone()),
        });
    }

    fn on_epoch_end(&mut self, _epoch: usize, metrics: &EvalMetrics) {
        self.sink.emit(JobEvent::Metric {
            job_id: self.job_id.clone(),
            payload: MetricPayload::Eval(metrics.clone()),
        });
    }

    fn on_save(&mut self, path: &std::path::Path) {
        self.sink.emit(JobEvent::Phase {
            job_id: self.job_id.clone(),
            phase: Phase::SavingWeights,
        });
        self.sink.emit(JobEvent::Log {
            job_id: self.job_id.clone(),
            level: LogLevel::Info,
            line: format!("checkpoint saved: {}", path.display()),
        });
    }

    fn on_lr_event(&mut self, event: &str) {
        self.sink.emit(JobEvent::Log {
            job_id: self.job_id.clone(),
            level: LogLevel::Info,
            line: format!("lr: {event}"),
        });
    }

    fn should_stop(&self) -> bool {
        self.sink.is_cancelled()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(event: JobEvent) -> JobEvent {
        let json = serde_json::to_string(&event).expect("serialize");
        serde_json::from_str(&json).expect("deserialize")
    }

    #[test]
    fn started_round_trip() {
        let ev = JobEvent::started("job1", JobKind::Train);
        let back = round_trip(ev.clone());
        assert!(matches!(back, JobEvent::Started { .. }));
        assert_eq!(back.job_id(), "job1");
    }

    #[test]
    fn phase_round_trip() {
        let ev = JobEvent::Phase {
            job_id: "j".into(),
            phase: Phase::ServerReady {
                address: "0.0.0.0:8080".into(),
            },
        };
        let back = round_trip(ev);
        match back {
            JobEvent::Phase { phase, .. } => match phase {
                Phase::ServerReady { address } => assert_eq!(address, "0.0.0.0:8080"),
                other => panic!("wrong phase: {other:?}"),
            },
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn step_metric_round_trip() {
        let step = StepMetrics {
            step: 7,
            loss: 2.31,
            lr: 1e-4,
            tok_sec: 1234.5,
            ..Default::default()
        };
        let ev = JobEvent::Metric {
            job_id: "j".into(),
            payload: MetricPayload::Step(step),
        };
        let back = round_trip(ev);
        match back {
            JobEvent::Metric {
                payload: MetricPayload::Step(s),
                ..
            } => {
                assert_eq!(s.step, 7);
                assert!((s.loss - 2.31).abs() < 1e-9);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn diverged_metrics_survive_the_wire() {
        // JSON has no NaN or infinity: serde_json writes them as `null`. A
        // run that diverges is exactly when a consumer needs the event, so
        // reading `null` back must give a non-finite number, not an error.
        let step = StepMetrics {
            step: 3,
            loss: f64::NAN,
            lr: 1e-4,
            tok_sec: f64::INFINITY,
            ..Default::default()
        };
        let eval = EvalMetrics {
            loss: 800.0,
            perplexity: f64::INFINITY,
            ..Default::default()
        };
        for payload in [MetricPayload::Step(step), MetricPayload::Eval(eval)] {
            let mut buf = Vec::new();
            let ev = JobEvent::Metric {
                job_id: "j".into(),
                payload,
            };
            write_event(&mut buf, &ev).unwrap();
            let back = parse_event(std::str::from_utf8(&buf).unwrap()).unwrap();
            match back {
                JobEvent::Metric {
                    payload: MetricPayload::Step(s),
                    ..
                } => {
                    assert_eq!(s.step, 3);
                    assert!(s.loss.is_nan() && s.tok_sec.is_nan());
                    assert_eq!(s.lr, 1e-4);
                }
                JobEvent::Metric {
                    payload: MetricPayload::Eval(e),
                    ..
                } => {
                    assert_eq!(e.loss, 800.0);
                    assert!(e.perplexity.is_nan());
                }
                other => panic!("wrong variant: {other:?}"),
            }
        }
    }

    /// Counts the `write` calls it receives.
    struct CountingWriter {
        writes: Vec<Vec<u8>>,
    }

    impl Write for CountingWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.writes.push(buf.to_vec());
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn an_event_is_one_write() {
        // Consumers tail the stream, and several processes can share a pipe:
        // a record written in pieces can be cut or interleaved, one written
        // in a single call (under the pipe's atomic size) cannot.
        let mut out = CountingWriter { writes: Vec::new() };
        write_event(
            &mut out,
            &JobEvent::Log {
                job_id: "j".into(),
                level: LogLevel::Info,
                line: "step 1".into(),
            },
        )
        .unwrap();
        assert_eq!(out.writes.len(), 1, "{:?}", out.writes);
        let line = std::str::from_utf8(&out.writes[0]).unwrap();
        assert!(line.ends_with('\n') && line.matches('\n').count() == 1);
        assert_eq!(parse_event(line).unwrap().job_id(), "j");
    }

    #[test]
    fn jsonl_codec() {
        let ev = JobEvent::Token {
            job_id: "j".into(),
            token: "hello".into(),
            index: 0,
        };
        let mut buf = Vec::new();
        write_event(&mut buf, &ev).unwrap();
        let line = std::str::from_utf8(&buf).unwrap();
        assert!(line.ends_with('\n'));
        let back = parse_event(line).unwrap();
        match back {
            JobEvent::Token { token, index, .. } => {
                assert_eq!(token, "hello");
                assert_eq!(index, 0);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn jsonl_codec_multiline() {
        let mut buf = Vec::new();
        let events = vec![
            JobEvent::started("j", JobKind::Train),
            JobEvent::Phase {
                job_id: "j".into(),
                phase: Phase::Training,
            },
            JobEvent::failed("j", "boom"),
        ];
        for e in &events {
            write_event(&mut buf, e).unwrap();
        }
        let s = std::str::from_utf8(&buf).unwrap();
        let parsed: Vec<_> = s.lines().map(|l| parse_event(l).expect("parse")).collect();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].job_id(), "j");
    }

    #[test]
    fn job_status_state_machine() {
        let mut status: JobStatus<&'static str> = JobStatus::Idle;
        status.apply_event(&JobEvent::started("j", JobKind::Train));
        assert!(matches!(status, JobStatus::Starting { .. }));

        status.apply_event(&JobEvent::Phase {
            job_id: "j".into(),
            phase: Phase::Training,
        });
        assert!(matches!(status, JobStatus::Starting { .. }));

        let step = StepMetrics {
            step: 1,
            ..Default::default()
        };
        status.apply_event(&JobEvent::Metric {
            job_id: "j".into(),
            payload: MetricPayload::Step(step),
        });
        assert!(matches!(status, JobStatus::Running { .. }));

        status.complete("done");
        assert!(matches!(status, JobStatus::Completed { result: "done" }));
    }

    #[test]
    fn a_completed_event_completes_the_status() {
        let mut status: JobStatus<Option<f64>> = JobStatus::Idle;
        let events = [
            JobEvent::started("j", JobKind::Train),
            JobEvent::Metric {
                job_id: "j".into(),
                payload: MetricPayload::Step(StepMetrics::default()),
            },
            JobEvent::Completed {
                job_id: "j".into(),
                summary: CompletionSummary {
                    final_loss: Some(0.5),
                    ..Default::default()
                },
                ts: 0,
            },
        ];
        for event in &events {
            status.apply_event_with(event, |summary| summary.final_loss);
        }
        assert!(matches!(
            status,
            JobStatus::Completed { result: Some(loss) } if loss == 0.5
        ));
    }

    #[test]
    fn null_sink_drops() {
        let sink = NullSink;
        sink.emit(JobEvent::started("j", JobKind::Train));
        assert!(!sink.is_cancelled());
    }

    #[test]
    fn jsonl_sink_keeps_the_first_write_error() {
        struct Failing(u32);
        impl Write for Failing {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                self.0 += 1;
                Err(io::Error::other(format!("write {}", self.0)))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let sink = JsonlSink::new(Failing(0));
        assert!(sink.take_error().is_none());
        sink.emit(JobEvent::started("j", JobKind::Train));
        sink.emit(JobEvent::failed("j", "boom"));
        let first = sink.take_error().expect("the failure is kept");
        assert_eq!(first.to_string(), "write 1");
        assert!(sink.take_error().is_none(), "taking it clears it");
    }

    #[test]
    fn jsonl_sink_writes_to_buffer() {
        let buf: Vec<u8> = Vec::new();
        let cursor = std::io::Cursor::new(buf);
        let sink = JsonlSink::new(cursor);
        sink.emit(JobEvent::started("j", JobKind::Train));
        let inner = sink.inner.into_inner().unwrap().into_inner();
        let line = std::str::from_utf8(&inner).unwrap();
        let back = parse_event(line).unwrap();
        assert_eq!(back.job_id(), "j");
    }
}
