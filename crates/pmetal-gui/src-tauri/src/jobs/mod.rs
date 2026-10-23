//! The app's jobs: what is running, what ran, and how the webview hears of it.
//!
//! The processes, their process groups, bounded output, progress and shutdown
//! belong to [`pmetal_supervisor::Supervisor`], the one job runner the
//! terminal interface, the MCP server and the Python bindings use too. This
//! module adds what is the app's own: the kinds of job it offers
//! ([`kinds`]), the view of a job the webview draws ([`JobView`]), and the
//! pump that turns the supervisor's events into at most a few updates a second.

pub mod form;
pub mod kinds;
mod local;
#[cfg(test)]
mod typegen;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use pmetal::core::SecretString;
use pmetal::core::events::{JobEvent, MetricPayload};
use pmetal::core::sync::locked;
use pmetal_supervisor::{
    Error as SupervisorError, Event, EventKind, JobId, Limits, ProcessStatus, Stream, Supervisor,
};
use serde::Serialize;
use tokio::sync::broadcast::error::RecvError;

use self::kinds::{FieldIssue, Launch, LaunchContext};
use crate::error::AppError;

/// Jobs that may run at once. Training, a server and an evaluation side by side
/// are normal on a large Mac; the machine's memory is the real limit.
const MAX_RUNNING: usize = 6;
/// Finished jobs remembered, with their output.
const MAX_FINISHED: usize = 100;
/// How often changed jobs and their new output are sent to the webview.
const FLUSH_EVERY: Duration = Duration::from_millis(200);
/// The line a server prints once it has built its router and is about to bind.
const SERVE_ANNOUNCEMENT: &str = "OpenAI-compatible API available at http://";
/// How long a server that announced itself is waited for to answer.
const SERVE_PROBE_FOR: Duration = Duration::from_secs(120);

/// How a job stands, for the webview.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Running,
    Stopping,
    Completed,
    Failed,
    Stopped,
}

impl State {
    #[cfg(test)]
    pub fn is_active(self) -> bool {
        matches!(self, Self::Running | Self::Stopping)
    }
}

/// A server job's address, and whether anything answers there.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ServeView {
    pub url: String,
    pub reachable: bool,
}

/// What a finished job said it produced.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResultView {
    pub output_path: Option<String>,
    pub final_loss: Option<f64>,
    pub total_steps: Option<u64>,
    pub extra: BTreeMap<String, String>,
}

/// A job as the webview draws it. Everything in it is derived from the
/// supervisor's state and the job's own events; nothing is kept twice.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct JobView {
    pub id: String,
    pub kind: String,
    pub label: String,
    /// The model the job works on.
    pub title: String,
    /// The command, as a person would type it, with credentials hidden.
    pub command: String,
    pub state: State,
    pub error: Option<String>,
    pub exit_code: Option<i32>,
    /// What stage the job says it is in: "Loading model…".
    pub phase: Option<String>,
    pub step: Option<u64>,
    pub total_steps: Option<u64>,
    pub epoch: Option<f64>,
    pub total_epochs: Option<u64>,
    pub loss: Option<f64>,
    pub best_loss: Option<f64>,
    pub eval_loss: Option<f64>,
    pub learning_rate: Option<f64>,
    pub grad_norm: Option<f64>,
    pub tokens_per_second: Option<f64>,
    /// 0 to 1, when the job knows its total.
    pub progress: Option<f64>,
    pub eta_seconds: Option<u64>,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub elapsed_seconds: f64,
    /// Where the job writes its results.
    pub output: Option<String>,
    pub serve: Option<ServeView>,
    pub result: Option<ResultView>,
}

/// One line of a job's output.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutputLine {
    pub seq: u64,
    pub stream: &'static str,
    pub text: String,
}

impl OutputLine {
    fn of(line: &pmetal_supervisor::LogLine) -> Self {
        Self {
            seq: line.seq,
            stream: match line.stream {
                Stream::Stdout => "stdout",
                Stream::Stderr => "stderr",
            },
            text: line.text.clone(),
        }
    }
}

/// A page of a job's output, for a reader that joined late.
#[derive(Debug, Clone, Serialize)]
pub struct OutputPage {
    pub lines: Vec<OutputLine>,
    pub next_since: u64,
    pub dropped_lines: u64,
}

/// A step a job reported, for a chart.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StepPoint {
    pub step: u64,
    pub loss: f64,
    pub learning_rate: f64,
    pub tokens_per_second: f64,
}

/// Steps after a position, and the position to ask from next.
#[derive(Debug, Clone, Serialize)]
pub struct StepPage {
    pub steps: Vec<StepPoint>,
    pub next_since: u64,
}

/// Where updates for the webview go.
pub trait EventSink: Send + Sync + 'static {
    /// A job changed. Sent at most every [`FLUSH_EVERY`], except that a job
    /// starting or ending is sent at once.
    fn update(&self, job: &JobView);
    /// New output lines of a job.
    fn output(&self, id: &str, lines: &[OutputLine]);
    /// Updates were missed; read the jobs again.
    fn resync(&self);
}

#[derive(Default)]
struct Meta {
    kind: String,
    label: String,
    title: String,
    output: Option<PathBuf>,
    exclusive: Option<String>,
    /// When the first step was seen and which it was: the pace is measured
    /// from there, not from the start, which includes loading the model.
    first_step: Option<(Instant, u64)>,
    reachable: bool,
}

struct Shared {
    supervisor: Supervisor,
    meta: Mutex<HashMap<JobId, Meta>>,
    /// Jobs that changed without an event of the supervisor's (a server that
    /// began answering): the pump sends them on.
    touch: tokio::sync::mpsc::UnboundedSender<JobId>,
    touched: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<JobId>>>,
}

/// The app's jobs.
#[derive(Clone)]
pub struct JobHub {
    shared: Arc<Shared>,
    /// `Err` when no `pmetal` could be found: the reason, shown on start.
    binary: Result<PathBuf, String>,
    output_root: PathBuf,
}

impl JobHub {
    /// A hub running `binary` (or failing every start with the reason it is
    /// missing), writing generated outputs under `output_root`.
    /// Must be created inside a tokio runtime.
    pub fn new(binary: Result<PathBuf, String>, output_root: PathBuf) -> Self {
        let supervisor = Supervisor::new(
            binary.clone().unwrap_or_else(|_| PathBuf::from("pmetal")),
            Limits {
                max_running: MAX_RUNNING,
                max_finished: MAX_FINISHED,
                ..Limits::default()
            },
        );
        let (touch, touched) = tokio::sync::mpsc::unbounded_channel();
        Self {
            shared: Arc::new(Shared {
                supervisor,
                meta: Mutex::new(HashMap::new()),
                touch,
                touched: Mutex::new(Some(touched)),
            }),
            binary,
            output_root,
        }
    }

    pub fn supervisor(&self) -> &Supervisor {
        &self.shared.supervisor
    }

    /// Start `kind` from the text entered in its form. `token` is given to the
    /// job as `HF_TOKEN`.
    pub fn start(
        &self,
        kind: &str,
        values: &BTreeMap<String, String>,
        token: Option<&SecretString>,
    ) -> Result<JobView, StartError> {
        let job_kind = kinds::find(kind).ok_or_else(|| {
            StartError::Message(AppError::new(format!(
                "There is no job called '{kind}'. This is a bug in PMetal's window; \
                 restart the app."
            )))
        })?;
        if let Err(reason) = &self.binary {
            return Err(StartError::Message(AppError::new(reason.clone())));
        }
        let ctx = LaunchContext {
            output_root: self.output_root.clone(),
            now: chrono::Local::now(),
        };
        let launch = job_kind
            .prepare(values, &ctx)
            .map_err(StartError::Invalid)?;
        self.spawn(kind, job_kind.form().label, launch, token)
    }

    fn spawn(
        &self,
        kind: &str,
        label: String,
        launch: Launch,
        token: Option<&SecretString>,
    ) -> Result<JobView, StartError> {
        // Two jobs writing one folder would truncate each other's progress
        // files; two servers on one port, one of them fails to bind.
        {
            let meta = locked(&self.shared.meta);
            for (id, other) in meta.iter() {
                let running = self
                    .shared
                    .supervisor
                    .summary(id)
                    .is_ok_and(|s| s.status.is_active());
                if !running {
                    continue;
                }
                if let (Some(mine), Some(theirs)) = (&launch.plan.output, &other.output)
                    && crate::paths::overlaps(mine, theirs)
                {
                    return Err(StartError::Message(AppError::new(format!(
                        "A {} job is already writing to {}. Stop it, or choose a different \
                         output location.",
                        other.label.to_lowercase(),
                        theirs.display()
                    ))));
                }
                if launch.exclusive.is_some() && launch.exclusive == other.exclusive {
                    return Err(StartError::Message(AppError::new(format!(
                        "A {} is already running with the same {}. Stop it first, or use \
                         another value.",
                        other.label.to_lowercase(),
                        launch.exclusive.as_deref().unwrap_or("").replace(':', " ")
                    ))));
                }
            }
        }

        let subcommand = launch.plan.subcommand.clone();
        let output = launch.plan.output.clone();
        let env = token
            .map(|t| vec![("HF_TOKEN".to_string(), t.expose_secret().to_string())])
            .unwrap_or_default();
        let id = self
            .shared
            .supervisor
            .spawn(launch.plan.into_request_with_env(env))
            .map_err(|e| match e {
                SupervisorError::Io(_) => StartError::Message(AppError::new(format!(
                    "Could not start `pmetal {subcommand}`: {e}"
                ))),
                other => StartError::Message(AppError::new(other.to_string())),
            })?;
        locked(&self.shared.meta).insert(
            id.clone(),
            Meta {
                kind: kind.to_string(),
                label,
                title: launch.title,
                output,
                exclusive: launch.exclusive,
                ..Meta::default()
            },
        );
        self.prune();
        self.view(&id)
            .ok_or_else(|| StartError::Message(AppError::new("The job ended as it started.")))
    }

    /// Forget the metadata of jobs the supervisor has dropped.
    fn prune(&self) {
        let supervisor = &self.shared.supervisor;
        locked(&self.shared.meta).retain(|id, _| supervisor.summary(id).is_ok());
    }

    /// Ask a job to stop: SIGTERM to its process group, SIGKILL after a grace
    /// period. A job already stopping, or stopped, is fine.
    pub fn stop(&self, id: &str) -> Result<(), AppError> {
        self.shared.supervisor.stop(id).map_err(|e| match e {
            SupervisorError::NotFound(_) => AppError::new("That job is no longer in the list."),
            SupervisorError::NotRunning => AppError::new("That job has already finished."),
            other => AppError::new(other.to_string()),
        })
    }

    /// Remove a finished job and its output from the list.
    pub fn forget(&self, id: &str) -> Result<(), AppError> {
        self.shared.supervisor.forget(id).map_err(|e| match e {
            SupervisorError::NotRunning => {
                AppError::new("Stop the job before removing it from the list.")
            }
            other => AppError::new(other.to_string()),
        })?;
        locked(&self.shared.meta).remove(id);
        Ok(())
    }

    /// Jobs running, or being stopped.
    pub fn running(&self) -> usize {
        self.shared.supervisor.running()
    }

    /// Whether a running job works on this model (by the reference it was
    /// started with), so it is not deleted from under it.
    pub fn model_in_use(&self, matches: impl Fn(&str) -> bool) -> Option<String> {
        let meta = locked(&self.shared.meta);
        meta.iter().find_map(|(id, m)| {
            let running = self
                .shared
                .supervisor
                .summary(id)
                .is_ok_and(|s| s.status.is_active());
            (running && !m.title.is_empty() && matches(&m.title)).then(|| m.label.clone())
        })
    }

    /// Stop every running job and wait for them to leave.
    pub async fn shutdown(&self) {
        self.shared.supervisor.shutdown().await;
    }

    /// A job as the webview draws it.
    pub fn view(&self, id: &str) -> Option<JobView> {
        let supervisor = &self.shared.supervisor;
        let summary = supervisor.summary(id).ok()?;
        let meta = locked(&self.shared.meta);
        let meta = meta.get(id)?;
        let (state, error, exit_code) = match &summary.status {
            ProcessStatus::Running => (State::Running, None, None),
            ProcessStatus::Stopping => (State::Stopping, None, None),
            ProcessStatus::Completed { exit_code } => (State::Completed, None, Some(*exit_code)),
            ProcessStatus::Failed { exit_code, error } => (
                State::Failed,
                Some(explain_failure(error)),
                Some(*exit_code),
            ),
            ProcessStatus::Stopped => (State::Stopped, None, None),
        };
        let m = &summary.metrics;
        let best_loss = if m.last_loss.is_some() {
            supervisor.steps(id, 0).ok().and_then(|page| {
                page.steps
                    .iter()
                    .map(|s| s.loss)
                    .filter(|l| l.is_finite())
                    .reduce(f64::min)
            })
        } else {
            None
        };
        let progress = match (m.last_step, m.total_steps) {
            (Some(step), Some(total)) if total > 0 => {
                Some((step as f64 / total as f64).clamp(0.0, 1.0))
            }
            _ => None,
        };
        let eta_seconds = match (&meta.first_step, m.last_step, m.total_steps) {
            (Some((t0, step0)), Some(step), Some(total))
                if state == State::Running && step > *step0 =>
            {
                let per_step = t0.elapsed().as_secs_f64() / (step - step0) as f64;
                Some((per_step * total.saturating_sub(step) as f64).round() as u64)
            }
            _ => None,
        };
        let serve = (meta.kind == "serve").then(|| {
            let url = server_url(&supervisor.logs(id, 2000, None).ok());
            url.map(|url| ServeView {
                url,
                reachable: meta.reachable && state == State::Running,
            })
        });
        let result = supervisor
            .completion(id)
            .ok()
            .flatten()
            .map(|c| ResultView {
                output_path: c.output_path,
                final_loss: c.final_loss,
                total_steps: c.total_steps,
                extra: c.extra,
            });
        Some(JobView {
            id: summary.id,
            kind: meta.kind.clone(),
            label: meta.label.clone(),
            title: meta.title.clone(),
            command: format!("pmetal {}", summary.args.join(" ")),
            state,
            error,
            exit_code,
            phase: summary.phase,
            step: m.last_step,
            total_steps: m.total_steps,
            epoch: m.epoch,
            total_epochs: m.total_epochs,
            loss: m.last_loss,
            best_loss,
            eval_loss: m.eval_loss,
            learning_rate: m.last_lr,
            grad_norm: m.grad_norm,
            tokens_per_second: m.tokens_per_second,
            progress,
            eta_seconds,
            started_at: summary.started_at,
            finished_at: summary.finished_at,
            elapsed_seconds: summary.elapsed_seconds,
            output: summary.output,
            serve: serve.flatten(),
            result,
        })
    }

    /// Every job, newest first.
    pub fn list(&self) -> Vec<JobView> {
        self.shared
            .supervisor
            .list()
            .into_iter()
            .filter_map(|s| self.view(&s.id))
            .collect()
    }

    /// A job's output after position `since`, the last `tail` lines.
    pub fn output(
        &self,
        id: &str,
        tail: usize,
        since: Option<u64>,
    ) -> Result<OutputPage, AppError> {
        let page = self
            .shared
            .supervisor
            .logs(id, tail, since)
            .map_err(|_| AppError::new("That job is no longer in the list."))?;
        Ok(OutputPage {
            lines: page.entries.iter().map(OutputLine::of).collect(),
            next_since: page.next_since,
            dropped_lines: page.dropped_lines,
        })
    }

    /// The steps a job reported after position `since`.
    pub fn steps(&self, id: &str, since: u64) -> Result<StepPage, AppError> {
        let page = self
            .shared
            .supervisor
            .steps(id, since)
            .map_err(|_| AppError::new("That job is no longer in the list."))?;
        Ok(StepPage {
            steps: page
                .steps
                .iter()
                .map(|s| StepPoint {
                    step: s.step as u64,
                    loss: s.loss,
                    learning_rate: s.lr,
                    tokens_per_second: s.tok_sec,
                })
                .collect(),
            next_since: page.next_since,
        })
    }

    /// Send the webview what happens to jobs, until the supervisor is gone.
    /// Must be called inside a tokio runtime.
    pub fn pump(&self, sink: Arc<dyn EventSink>) -> tokio::task::JoinHandle<()> {
        let hub = self.clone();
        let mut events = self.shared.supervisor.subscribe();
        let mut touched = locked(&self.shared.touched).take();
        tokio::spawn(async move {
            let mut dirty: HashSet<JobId> = HashSet::new();
            let mut lines: HashMap<JobId, Vec<OutputLine>> = HashMap::new();
            let mut tick = tokio::time::interval(FLUSH_EVERY);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    received = events.recv() => match received {
                        Ok(event) => hub.on_event(event, &*sink, &mut dirty, &mut lines),
                        Err(RecvError::Lagged(_)) => {
                            // Some events are gone. Everything is read again,
                            // from the supervisor's own state.
                            sink.resync();
                            dirty.clear();
                            lines.clear();
                        }
                        Err(RecvError::Closed) => break,
                    },
                    Some(id) = async {
                        match touched.as_mut() {
                            Some(rx) => rx.recv().await,
                            None => std::future::pending().await,
                        }
                    } => {
                        dirty.insert(id);
                    }
                    _ = tick.tick() => hub.flush(&*sink, &mut dirty, &mut lines),
                }
            }
        })
    }

    fn on_event(
        &self,
        event: Event,
        sink: &dyn EventSink,
        dirty: &mut HashSet<JobId>,
        lines: &mut HashMap<JobId, Vec<OutputLine>>,
    ) {
        let Event { job, kind } = event;
        match kind {
            EventKind::Started => {
                self.send_now(&job, sink);
            }
            EventKind::Output(line) => {
                if let Some(address) = announced_address(&line.text) {
                    self.probe_server(&job, address);
                }
                lines.entry(job).or_default().push(OutputLine::of(&line));
            }
            EventKind::Delta(_) => {}
            EventKind::Job(event) => {
                if let JobEvent::Metric {
                    payload: MetricPayload::Step(step),
                    ..
                } = &event
                {
                    let mut meta = locked(&self.shared.meta);
                    if let Some(meta) = meta.get_mut(&job)
                        && meta.first_step.is_none()
                    {
                        meta.first_step = Some((Instant::now(), step.step as u64));
                    }
                }
                dirty.insert(job);
            }
            EventKind::Exited(_) => {
                // The output first, then the end, so the last lines are not
                // shown after the job is marked finished.
                if let Some(pending) = lines.remove(&job) {
                    sink.output(&job, &pending);
                }
                dirty.remove(&job);
                self.send_now(&job, sink);
            }
        }
    }

    fn send_now(&self, id: &str, sink: &dyn EventSink) {
        if let Some(view) = self.view(id) {
            sink.update(&view);
        }
    }

    fn flush(
        &self,
        sink: &dyn EventSink,
        dirty: &mut HashSet<JobId>,
        lines: &mut HashMap<JobId, Vec<OutputLine>>,
    ) {
        for (id, pending) in lines.drain() {
            sink.output(&id, &pending);
        }
        for id in dirty.drain() {
            self.send_now(&id, sink);
        }
    }

    /// A server printed where it listens. It is "serving" once something
    /// answers there, not when the process started: a server that cannot bind
    /// (the port is taken) is never shown as running.
    fn probe_server(&self, id: &str, address: String) {
        let hub = self.clone();
        let id = id.to_string();
        tokio::spawn(async move {
            let port = address.rsplit(':').next().unwrap_or("").to_string();
            let target = format!("127.0.0.1:{port}");
            let deadline = Instant::now() + SERVE_PROBE_FOR;
            while Instant::now() < deadline {
                let alive = hub
                    .shared
                    .supervisor
                    .summary(&id)
                    .is_ok_and(|s| s.status.is_active());
                if !alive {
                    return;
                }
                if tokio::net::TcpStream::connect(&target).await.is_ok() {
                    if let Some(meta) = locked(&hub.shared.meta).get_mut(&id) {
                        meta.reachable = true;
                    }
                    let _ = hub.shared.touch.send(id);
                    return;
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        });
    }
}

/// Why a start was refused.
#[derive(Debug)]
pub enum StartError {
    /// Fields of the form that are wrong, each with what is wrong.
    Invalid(Vec<FieldIssue>),
    Message(AppError),
}

/// The address a server announced: `127.0.0.1:8080` from "OpenAI-compatible API
/// available at http://127.0.0.1:8080/v1".
pub fn announced_address(line: &str) -> Option<String> {
    let at = line.find(SERVE_ANNOUNCEMENT)?;
    let rest = &line[at + SERVE_ANNOUNCEMENT.len()..];
    let address = rest.split(['/', ' ']).next().unwrap_or("");
    (!address.is_empty()).then(|| address.to_string())
}

/// The URL to reach a server at, from its output so far.
fn server_url(page: &Option<pmetal_supervisor::LogPage>) -> Option<String> {
    let address = page
        .as_ref()?
        .entries
        .iter()
        .find_map(|l| announced_address(&l.text))?;
    // A server bound to every interface is reached on this Mac as localhost.
    Some(format!(
        "http://{}",
        address.replace("0.0.0.0", "localhost")
    ))
}

/// What a failure means to the person who sees it, when the job's last error
/// line is something a person cannot act on as it stands.
fn explain_failure(error: &str) -> String {
    let lower = error.to_lowercase();
    if lower.contains("killed by signal 9") || lower.contains("out of memory") {
        return format!(
            "{error}. The system may have run out of memory: close other apps, use a smaller \
             model or batch size, or turn on gradient checkpointing."
        );
    }
    if lower.contains("401") || lower.contains("authentication required") || lower.contains("gated")
    {
        return format!(
            "{error}. If the model or dataset is gated or private, accept its terms on \
             huggingface.co and add an access token in Settings."
        );
    }
    error.to_string()
}

#[cfg(test)]
mod tests;
