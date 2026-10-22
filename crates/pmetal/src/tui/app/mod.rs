//! Main application state, the event loop and drawing.
//!
//! The pieces live beside this file: `input` turns keys into actions, `runs`
//! starts and follows jobs, `modals` handles what dialogs answer, `forms` holds
//! the tabs that run a job from a form.

mod forms;
mod input;
mod modals;
mod runs;
#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::tui::event::{AppMsg, Event, EventHandler};
use crate::tui::jobs::JobHub;
use crate::tui::modal::Modal;
use crate::tui::tabs::{
    DashboardTab, DatasetsTab, DeviceTab, InferenceTab, JobsTab, ModelsTab, Tab,
};
use crate::tui::terminal::{TerminalSession, guard_jobs};
use crate::tui::theme::THEME;
use crate::tui::widgets::{Footer, Header};

pub use forms::Forms;

/// Below this size the interface asks for a bigger terminal instead of
/// drawing into a space its layouts do not fit.
pub const MIN_SIZE: (u16, u16) = (60, 14);
/// The fastest the screen is redrawn, however much is happening.
const FRAME: Duration = Duration::from_millis(33);
/// How long a note in the footer stays.
const NOTICE_FOR: Duration = Duration::from_secs(5);
/// Events handled between two draws at most, so a flood of output cannot
/// keep the screen from updating.
const EVENTS_PER_FRAME: usize = 512;

/// What the interface needs from the machine and the command line.
pub struct Environment {
    /// The Metal device's report, if there is a device.
    pub device: Option<crate::tui::tabs::DeviceInfo>,
    /// What the GPU can address, for the fit estimates.
    pub device_spec: Option<pmetal_hub::DeviceSpec>,
    /// Installed memory in bytes.
    pub installed_memory: u64,
    /// The `pmetal` binary jobs run.
    pub binary: PathBuf,
    /// A metrics file for the monitor to follow.
    pub metrics_file: Option<PathBuf>,
    /// The tab to open on.
    pub start_tab: Tab,
}

/// What a dialog is about, so its answer reaches the right place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Pending {
    /// Choosing a model for a field of a form.
    PickModel {
        tab: Tab,
        field: String,
    },
    /// Choosing a dataset for a field of a form.
    PickDataset {
        tab: Tab,
        field: String,
    },
    /// "Start this job?" for a form tab.
    Start(Tab),
    /// Choosing the model to chat with.
    InferenceModel,
    /// Choosing the base model of a LoRA adapter to chat with.
    InferenceBase(String),
    /// Choosing the base model of a LoRA adapter to fuse.
    FuseBase(String),
    /// "Fuse this adapter?"
    FuseStart,
    /// "Chat with what the run wrote to this directory?"
    TestInference(PathBuf),
    DownloadModel,
    HfSearch,
    AddModelDir,
    AddDatasetDir,
    ConvertDataset,
    /// Changing the learning rate of the job writing to this directory.
    SetLearningRate(PathBuf),
    /// "Quit and stop the running jobs?"
    Quit,
}

/// Top-level application state.
pub struct App {
    /// The event loop should end.
    pub(crate) should_quit: bool,
    /// Jobs are being stopped on the way out.
    pub(crate) stopping: bool,
    pub(crate) active_tab: Tab,
    /// Whether mouse capture is wanted (F2 toggles it).
    pub(crate) mouse_wanted: bool,

    pub(crate) dashboard: DashboardTab,
    pub(crate) device: DeviceTab,
    pub(crate) models: ModelsTab,
    pub(crate) datasets: DatasetsTab,
    pub(crate) inference: InferenceTab,
    pub(crate) jobs: JobsTab,
    pub(crate) forms: Forms,

    /// Modal dialog stack (topmost is rendered last / receives keys first).
    pub(crate) modal_stack: Vec<Modal>,
    /// What the topmost dialog is about.
    pub(crate) pending: Option<Pending>,

    /// The jobs; present once the event loop is running (or a test attaches).
    pub(crate) hub: Option<JobHub>,
    pub(crate) tx: Option<tokio::sync::mpsc::Sender<AppMsg>>,
    binary: PathBuf,

    /// Cancels the Hugging Face search in flight.
    pub(crate) hf_search_cancel: Option<tokio_util::sync::CancellationToken>,
    /// A note for the footer and when it was made.
    pub(crate) notice: Option<(String, Instant)>,
    /// The numbers of the newest scans started, so a late one is ignored.
    scan_seq: ScanSeq,
    pub(crate) tick_count: u64,
    /// The screen needs drawing.
    dirty: bool,
    /// The width the header was last drawn at, for mouse hit-testing.
    pub(crate) header_width: u16,
    metrics_file: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub(crate) struct ScanSeq {
    models: u64,
    datasets: u64,
    history: u64,
}

impl App {
    /// An app for this machine.
    pub fn new(metrics_file: Option<PathBuf>, start_tab: Tab) -> Self {
        let device = crate::tui::tabs::DeviceInfo::query();
        Self::with_environment(Environment {
            device,
            device_spec: crate::tui::tabs::query_device_spec(),
            installed_memory: pmetal_runtime::memory::get_system_memory().unwrap_or(0),
            binary: pmetal_supervisor::resolve_pmetal_binary(),
            metrics_file,
            start_tab,
        })
    }

    /// An app for the machine `env` describes.
    pub fn with_environment(env: Environment) -> Self {
        Self {
            should_quit: false,
            stopping: false,
            active_tab: env.start_tab,
            mouse_wanted: true,
            dashboard: DashboardTab::new(),
            device: DeviceTab::with(env.device, env.installed_memory),
            models: ModelsTab::with_device(env.device_spec),
            datasets: DatasetsTab::new(),
            inference: InferenceTab::new(),
            jobs: JobsTab::new(),
            forms: Forms::new(),
            modal_stack: Vec::new(),
            pending: None,
            hub: None,
            tx: None,
            binary: env.binary,
            hf_search_cancel: None,
            notice: None,
            scan_seq: ScanSeq::default(),
            tick_count: 0,
            dirty: true,
            header_width: 0,
            metrics_file: env.metrics_file,
        }
    }

    /// Give the app its jobs: they report to `tx`.
    pub(crate) fn attach(&mut self, tx: tokio::sync::mpsc::Sender<AppMsg>) {
        self.hub = Some(JobHub::new(self.binary.clone(), tx.clone()));
        self.tx = Some(tx);
    }

    /// Run the interface until the person quits. Must be called from within a
    /// tokio runtime.
    ///
    /// The terminal is given back however this ends: [`TerminalSession`]
    /// restores it when dropped, on a normal return, an error from the loop or
    /// an unwinding panic.
    pub async fn run(mut self) -> anyhow::Result<()> {
        let mut session = TerminalSession::start(self.mouse_wanted)?;
        let mut events = EventHandler::new(Duration::from_millis(200));
        self.attach(events.app_tx());
        guard_jobs(self.hub.as_ref().map(|hub| hub.supervisor().clone()));
        self.start_background_work();

        let mut last_draw = Instant::now() - FRAME;
        let outcome = loop {
            if self.dirty {
                // Hold the frame rate: events that arrive meanwhile are
                // handled together on the next one.
                let since = last_draw.elapsed();
                if since < FRAME {
                    tokio::time::sleep(FRAME - since).await;
                }
                if let Err(e) = session.terminal().draw(|frame| self.draw(frame)) {
                    break Err(e.into());
                }
                last_draw = Instant::now();
                self.dirty = false;
            }
            if self.should_quit {
                break Ok(());
            }
            let Some(event) = events.next().await else {
                break Ok(());
            };
            self.handle_event(event);
            let mut handled = 1;
            while handled < EVENTS_PER_FRAME {
                match events.try_next() {
                    Some(event) => self.handle_event(event),
                    None => break,
                }
                handled += 1;
            }
            session.set_mouse(self.mouse_wanted);
        };

        // Nothing started here outlives the interface: stop the jobs, with the
        // screen saying so, then give the terminal back.
        if self.running_jobs() > 0 {
            self.stopping = true;
            self.dirty = true;
            let _ = session.terminal().draw(|frame| self.draw(frame));
        }
        if let Some(hub) = &self.hub {
            hub.supervisor().shutdown().await;
        }
        guard_jobs(None);
        drop(events);
        drop(session);
        outcome
    }

    /// Kick off the work the first screen needs, off the interface's thread.
    pub(crate) fn start_background_work(&mut self) {
        self.rescan_models();
        self.rescan_datasets();
        self.rescan_history();
        self.follow_metrics_file();
    }

    /// Handle one event.
    pub(crate) fn handle_event(&mut self, event: Event) {
        match event {
            Event::Key(key) => self.handle_key(key),
            Event::Mouse(mouse) => self.handle_mouse(mouse),
            Event::Paste(text) => self.handle_paste(&text),
            Event::Resize => {}
            Event::Tick => self.on_tick(),
            Event::Terminate => self.begin_quit(),
            Event::App(msg) => self.handle_app_msg(msg),
        }
        self.dirty = true;
    }

    /// Called on each tick (200 ms).
    fn on_tick(&mut self) {
        self.tick_count += 1;
        if self
            .notice
            .as_ref()
            .is_some_and(|(_, since)| since.elapsed() > NOTICE_FOR)
        {
            self.notice = None;
        }
        // A job's last event may have been lost; its state is not.
        self.reconcile_jobs();
        // Memory use is read every two seconds while the System tab is open.
        if self.active_tab == Tab::Device && self.tick_count % 10 == 0 {
            self.sample_memory();
        }
    }

    /// Show a note in the footer for a few seconds.
    pub(crate) fn notify(&mut self, text: impl Into<String>) {
        self.notice = Some((text.into(), Instant::now()));
    }

    pub(crate) fn next_scan(&mut self, which: fn(&mut ScanSeq) -> &mut u64) -> u64 {
        let n = which(&mut self.scan_seq);
        *n += 1;
        *n
    }

    pub(crate) fn scan_is_current(
        &mut self,
        which: fn(&mut ScanSeq) -> &mut u64,
        seq: u64,
    ) -> bool {
        *which(&mut self.scan_seq) == seq
    }
}

pub(crate) fn models_seq(s: &mut ScanSeq) -> &mut u64 {
    &mut s.models
}
pub(crate) fn datasets_seq(s: &mut ScanSeq) -> &mut u64 {
    &mut s.datasets
}
pub(crate) fn history_seq(s: &mut ScanSeq) -> &mut u64 {
    &mut s.history
}

impl App {
    /// Draw the full UI to a frame.
    pub(crate) fn draw(&mut self, frame: &mut ratatui::Frame<'_>) {
        let area = frame.area();
        let buf = frame.buffer_mut();
        buf.set_style(area, THEME.root);

        if area.width < MIN_SIZE.0 || area.height < MIN_SIZE.1 {
            let message = format!(
                "Terminal too small: {}x{} (need {}x{})",
                area.width, area.height, MIN_SIZE.0, MIN_SIZE.1
            );
            let y = area.y + area.height / 2;
            if area.height > 0 {
                Paragraph::new(Line::from(Span::styled(message, THEME.text_warning)))
                    .centered()
                    .render(Rect::new(area.x, y, area.width, 1), buf);
            }
            return;
        }

        // Layout: header (2) | content (fill) | footer (1)
        let [header_area, content_area, footer_area] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(area);
        self.header_width = header_area.width;

        Header {
            active_tab: self.active_tab,
            tabs: Tab::ALL,
        }
        .render(header_area, buf);

        let telemetry = self.dashboard.telemetry();
        match self.active_tab {
            Tab::Dashboard => self.dashboard.render(content_area, buf),
            Tab::Device => (&self.device).render(content_area, buf),
            Tab::Models => (&mut self.models).render(content_area, buf),
            Tab::Datasets => (&mut self.datasets).render(content_area, buf),
            Tab::Inference => (&mut self.inference).render(content_area, buf),
            Tab::Jobs => (&mut self.jobs).render(content_area, buf),
            form => {
                if let Some(tab) = self.forms.get_mut(form) {
                    tab.render(content_area, buf, &telemetry);
                }
            }
        }

        Footer {
            tab: self.active_tab,
            active_jobs: self.running_jobs(),
            notice: self.notice.as_ref().map(|(text, _)| text.clone()),
        }
        .render(footer_area, buf);

        if self.stopping {
            let stopping = Modal::progress(
                "Stopping",
                "Stopping the jobs that are running before the interface closes...",
            );
            let mut stopping = stopping;
            stopping.render(area, buf);
        } else if let Some(modal) = self.modal_stack.last_mut() {
            modal.render(area, buf);
        }
    }
}
