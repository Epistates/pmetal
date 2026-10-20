//! Event handling for the PMetal TUI.
//!
//! One bounded channel carries everything the interface reacts to: keys,
//! mouse, paste, resize, the tick, termination signals, and messages from
//! background work (jobs, scans, searches). The terminal reader, the ticker
//! and the signal listener run as tasks of their own, so nothing the
//! interface does for a key press waits on I/O.

use std::time::Duration;

use crossterm::event::{Event as CrosstermEvent, EventStream, KeyEvent, KeyEventKind, MouseEvent};
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::tui::tabs::{DatasetEntry, DatasetPeek, JobEntry, ModelEntry};

/// Events the interface handles.
#[derive(Debug)]
pub enum Event {
    /// A key was pressed.
    Key(KeyEvent),
    /// A mouse event occurred.
    Mouse(MouseEvent),
    /// Text was pasted (bracketed paste).
    Paste(String),
    /// The terminal was resized.
    Resize,
    /// A tick event for periodic updates.
    Tick,
    /// The process was asked to end (SIGTERM, SIGHUP) or its terminal went away.
    Terminate,
    /// A message from background work.
    App(AppMsg),
}

/// Messages from background work to the interface.
#[derive(Debug)]
pub enum AppMsg {
    /// Something happened to a job.
    Job(pmetal_supervisor::Event),
    /// The job event channel overflowed: events were lost, so the interface
    /// must read the jobs' state instead of trusting what it saw.
    JobsLagged,
    /// The model directories were scanned (`seq` numbers the scans, so a slow
    /// one that finishes after a newer one is ignored).
    ModelsScanned { seq: u64, models: Vec<ModelEntry> },
    /// The dataset directories were scanned.
    DatasetsScanned {
        seq: u64,
        datasets: Vec<DatasetEntry>,
    },
    /// A dataset was sampled for its columns and lengths.
    DatasetPeeked { path: String, peek: DatasetPeek },
    /// Finished runs were scanned.
    JobHistoryScanned { seq: u64, runs: Vec<JobEntry> },
    /// The kernel was asked how much memory is in use.
    MemorySampled(Option<u64>),
    /// Steps read from the metrics file `pmetal tui --metrics-file` follows.
    MetricsFile(Vec<pmetal_core::StepMetrics>),
    /// Something answers (or does not) at the address a job announced.
    Reachable { job: String, reachable: bool },
    /// Hugging Face Hub search results arrived.
    HfSearchResults {
        results: Vec<pmetal_hub::HfSearchResult>,
    },
    /// Hugging Face Hub search failed.
    HfSearchError { message: String },
}

/// Merges terminal input, ticks, signals and background messages into one
/// stream.
pub struct EventHandler {
    rx: mpsc::Receiver<Event>,
    app_tx: mpsc::Sender<AppMsg>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

/// Events that can wait in the channel. A burst of output from a job queues
/// here, not in memory without limit; the producers wait their turn.
const CHANNEL_CAPACITY: usize = 1024;

impl EventHandler {
    /// Start the reader, ticker, signal listener and forwarder tasks. Must be
    /// called from within a tokio runtime.
    pub fn new(tick_rate: Duration) -> Self {
        let (event_tx, event_rx) = mpsc::channel::<Event>(CHANNEL_CAPACITY);
        let (app_tx, mut app_rx) = mpsc::channel::<AppMsg>(CHANNEL_CAPACITY);

        // Background messages are never dropped: when the interface is slow
        // the sender waits (a job's reader, a scan), which is what bounds the
        // memory. Only ticks, which carry nothing, are skipped when it is busy.
        let mut tasks = Vec::new();
        let fwd_tx = event_tx.clone();
        tasks.push(tokio::spawn(async move {
            while let Some(msg) = app_rx.recv().await {
                if fwd_tx.send(Event::App(msg)).await.is_err() {
                    break;
                }
            }
        }));

        let tick_tx = event_tx.clone();
        tasks.push(tokio::spawn(async move {
            let mut interval = tokio::time::interval(tick_rate);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                match tick_tx.try_send(Event::Tick) {
                    Err(mpsc::error::TrySendError::Closed(_)) => break,
                    Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => {}
                }
            }
        }));

        let signal_tx = event_tx.clone();
        tasks.push(tokio::spawn(async move {
            termination_signal().await;
            let _ = signal_tx.send(Event::Terminate).await;
        }));

        let input_tx = event_tx;
        tasks.push(tokio::spawn(async move {
            let mut stream = EventStream::new();
            while let Some(next) = stream.next().await {
                let event = match next {
                    Ok(CrosstermEvent::Key(key)) if key.kind == KeyEventKind::Press => {
                        Some(Event::Key(key))
                    }
                    Ok(CrosstermEvent::Mouse(mouse)) => Some(Event::Mouse(mouse)),
                    Ok(CrosstermEvent::Paste(text)) => Some(Event::Paste(text)),
                    Ok(CrosstermEvent::Resize(..)) => Some(Event::Resize),
                    Ok(_) => None,
                    // The terminal is gone (stdin closed): there is nobody to
                    // draw for, so end rather than spin.
                    Err(_) => {
                        let _ = input_tx.send(Event::Terminate).await;
                        return;
                    }
                };
                if let Some(event) = event {
                    if input_tx.send(event).await.is_err() {
                        return;
                    }
                }
            }
            let _ = input_tx.send(Event::Terminate).await;
        }));

        Self {
            rx: event_rx,
            app_tx,
            tasks,
        }
    }

    /// Wait for the next event.
    pub async fn next(&mut self) -> Option<Event> {
        self.rx.recv().await
    }

    /// The next event if one is already waiting.
    pub fn try_next(&mut self) -> Option<Event> {
        self.rx.try_recv().ok()
    }

    /// A sender for background work. Clone it for each task.
    pub fn app_tx(&self) -> mpsc::Sender<AppMsg> {
        self.app_tx.clone()
    }
}

impl Drop for EventHandler {
    fn drop(&mut self) {
        // The reader must stop competing for the terminal once the interface
        // is gone.
        for task in &self.tasks {
            task.abort();
        }
    }
}

/// Resolves when the process is sent SIGTERM or SIGHUP. (Ctrl+C arrives as a
/// key press while the terminal is in raw mode.)
async fn termination_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let (Ok(mut term), Ok(mut hup)) = (
        signal(SignalKind::terminate()),
        signal(SignalKind::hangup()),
    ) else {
        // Without handlers the default action ends the process.
        std::future::pending::<()>().await;
        return;
    };
    tokio::select! {
        _ = term.recv() => {}
        _ = hup.recv() => {}
    }
}
