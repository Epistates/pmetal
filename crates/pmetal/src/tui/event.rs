//! Event handling for the PMetal TUI.
//!
//! Uses crossterm's `EventStream` for async event polling, combined with
//! a tokio mpsc channel for messages from background processes (training,
//! inference, downloads, etc.).

use std::path::PathBuf;

use crossterm::event::{Event as CrosstermEvent, EventStream, KeyEvent, MouseEvent};
use futures::StreamExt;
use tokio::sync::mpsc;

/// Application-level events.
#[derive(Debug)]
pub enum Event {
    /// A key was pressed.
    Key(KeyEvent),
    /// A mouse event occurred.
    Mouse(MouseEvent),
    /// The terminal was resized.
    #[allow(dead_code)]
    Resize(u16, u16),
    /// A tick event for periodic updates.
    Tick,
    /// A message from a background process.
    App(AppMsg),
}

/// Messages sent from background tasks (training, inference, downloads) to the TUI.
#[derive(Debug)]
#[allow(dead_code)]
pub enum AppMsg {
    /// A background job has started.
    JobStarted { job_id: String, job_type: JobType },
    /// Real-time metrics from a running training job.
    JobMetrics {
        job_id: String,
        step: usize,
        epoch: usize,
        total_epochs: usize,
        total_steps: usize,
        loss: f64,
        lr: f64,
        tok_sec: f64,
        ane_fwd_ms: f64,
        ane_bwd_ms: f64,
        rmsnorm_ms: f64,
        cblas_ms: f64,
        adam_ms: f64,
        total_ms: f64,
    },
    /// Status phase update from a running job (e.g. "Loading model...", "Tokenizing...").
    JobPhase { job_id: String, phase: String },
    /// A line of stdout/stderr from a running job.
    JobOutput { job_id: String, line: String },
    /// A background job has finished.
    JobFinished {
        job_id: String,
        success: bool,
        message: String,
    },
    /// Progress update for a model download (0.0..1.0).
    DownloadProgress { model_id: String, progress: f64 },
    /// A model download completed.
    DownloadComplete {
        model_id: String,
        success: bool,
        message: String,
    },
    /// A single token from streaming inference.
    InferenceToken { token: String },
    /// Inference generation completed.
    InferenceDone { tok_sec: f64, total_tokens: usize },
    /// Inference encountered an error.
    InferenceError { message: String },
    /// HuggingFace Hub search results arrived.
    HfSearchResults {
        results: Vec<pmetal_hub::HfSearchResult>,
    },
    /// HuggingFace Hub search failed.
    HfSearchError { message: String },
}

/// The type of background job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobType {
    Train,
    Infer,
    Distill,
    Grpo,
    Download,
    Convert,
    /// Long-running HTTP server (`pmetal serve`). No metrics file, pure
    /// stdout tailing, lives until the user cancels the job.
    Serve,
    /// One-shot quantize job (`pmetal quantize`). Progress tracked by
    /// parsing `[N/M]` tensor lines from stdout.
    Quantize,
    /// One-shot benchmark job (`pmetal bench` / `bench-workload`).
    /// Trial rows are parsed out of stdout.
    Bench,
    /// One-shot evaluation job (`pmetal eval`). Metrics are parsed out
    /// of stdout (perplexity / accuracy / loss).
    Eval,
    /// One-shot model-merge job (`pmetal merge`).
    Merge,
    /// Full-parameter pretraining job (`pmetal pretrain`).
    Pretrain,
}

impl std::fmt::Display for JobType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobType::Train => write!(f, "Train"),
            JobType::Infer => write!(f, "Infer"),
            JobType::Distill => write!(f, "Distill"),
            JobType::Grpo => write!(f, "GRPO"),
            JobType::Download => write!(f, "Download"),
            JobType::Convert => write!(f, "Convert"),
            JobType::Serve => write!(f, "Serve"),
            JobType::Quantize => write!(f, "Quantize"),
            JobType::Bench => write!(f, "Bench"),
            JobType::Eval => write!(f, "Eval"),
            JobType::Merge => write!(f, "Merge"),
            JobType::Pretrain => write!(f, "Pretrain"),
        }
    }
}

/// Async event handler that merges crossterm events, tick intervals,
/// and application messages into a single stream.
pub struct EventHandler {
    /// Receive events here.
    rx: mpsc::Receiver<Event>,
    /// Send app messages on this channel (clone it for background tasks).
    app_tx: mpsc::Sender<AppMsg>,
}

impl EventHandler {
    /// Create a new async event handler.
    ///
    /// Spawns a background task that polls crossterm events and tick intervals,
    /// forwarding them as `Event` variants. Returns the handler (which owns the
    /// receiver) and exposes `app_tx()` for sending `AppMsg` from background jobs.
    pub fn new(tick_rate: std::time::Duration) -> Self {
        // Bounded channels prevent unbounded memory growth when the TUI is slower than
        // background tasks. Messages exceeding the buffer are dropped with a warning.
        let (event_tx, event_rx) = mpsc::channel::<Event>(256);
        let (app_tx, mut app_rx) = mpsc::channel::<AppMsg>(256);

        // Forward app messages into the unified event channel
        let fwd_tx = event_tx.clone();
        tokio::spawn(async move {
            while let Some(msg) = app_rx.recv().await {
                if fwd_tx.try_send(Event::App(msg)).is_err() {
                    // Channel full or closed — drop the message to avoid blocking
                    tracing::debug!("Event channel full, dropping AppMsg");
                }
            }
        });

        // Crossterm event stream + tick interval
        let ct_tx = event_tx;
        tokio::spawn(async move {
            let mut event_stream = EventStream::new();
            let mut tick_interval = tokio::time::interval(tick_rate);

            loop {
                tokio::select! {
                    _ = tick_interval.tick() => {
                        // Tick events are low-priority; drop silently if channel is full
                        let _ = ct_tx.try_send(Event::Tick);
                    }
                    maybe_event = event_stream.next() => {
                        match maybe_event {
                            Some(Ok(ct_event)) => {
                                let event = match ct_event {
                                    CrosstermEvent::Key(key) => {
                                        if key.kind == crossterm::event::KeyEventKind::Press {
                                            Some(Event::Key(key))
                                        } else {
                                            None
                                        }
                                    }
                                    CrosstermEvent::Mouse(mouse) => Some(Event::Mouse(mouse)),
                                    CrosstermEvent::Resize(w, h) => Some(Event::Resize(w, h)),
                                    _ => None,
                                };
                                if let Some(event) = event {
                                    // Key/mouse events are higher priority — use blocking send
                                    if ct_tx.send(event).await.is_err() {
                                        break;
                                    }
                                }
                            }
                            Some(Err(_)) | None => break,
                        }
                    }
                }
            }
        });

        Self {
            rx: event_rx,
            app_tx,
        }
    }

    /// Receive the next event (async).
    pub async fn next(&mut self) -> Option<Event> {
        self.rx.recv().await
    }

    /// Get a sender for application messages. Clone this for background tasks.
    pub fn app_tx(&self) -> mpsc::Sender<AppMsg> {
        self.app_tx.clone()
    }
}

/// Specification for a command to run in the background.
#[derive(Debug, Clone)]
pub struct CommandSpec {
    pub job_type: JobType,
    pub args: Vec<String>,
    pub metrics_file: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
}
