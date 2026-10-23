//! The supervisor: starts `pmetal` subcommands, owns them until they end.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use pmetal_core::StepMetrics;
use pmetal_core::events::{CompletionSummary, JobEvent, MetricPayload};
use pmetal_core::sync::locked;
use serde::Serialize;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::broadcast;

use crate::group::{kill_group, terminate_group};
use crate::log::{LineAssembler, LogBuffer, LogLine, LogPage, MAX_LINE_BYTES, Stream, Utf8Stream};
use crate::metrics::{JobMetrics, parse_record};
use crate::tail::Tail;
use crate::text::{redact_argv, redact_secrets, sanitize_line};

/// A supervised job's identifier.
pub type JobId = String;

/// Longest a stalled pipe is waited for after its process has exited.
const DRAIN_GRACE: Duration = Duration::from_secs(2);
/// How often the metrics and event files are read.
const TAIL_POLL: Duration = Duration::from_millis(100);
/// Steps kept per job for [`Supervisor::steps`].
const MAX_STEPS_KEPT: usize = 20_000;
/// Events a subscriber can fall behind by before it is told it missed some.
const EVENT_BACKLOG: usize = 2048;

/// Limits on what the supervisor runs and remembers.
#[derive(Debug, Clone)]
pub struct Limits {
    /// Jobs that may run at once; starting one more is refused.
    pub max_running: usize,
    /// Finished jobs kept for inspection; the oldest are forgotten first.
    pub max_finished: usize,
    /// How long a stopped job has to exit after SIGTERM before SIGKILL.
    pub stop_grace: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_running: 4,
            max_finished: 50,
            stop_grace: Duration::from_secs(5),
        }
    }
}

/// What can go wrong asking the supervisor for something.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// No job has this id (it was never started, or it was forgotten).
    #[error("job not found: {0}")]
    NotFound(String),
    /// Starting the job would exceed [`Limits::max_running`].
    #[error(
        "{running} jobs are already running (the limit is {limit}). Stop one, or wait for one \
         to finish."
    )]
    TooManyRunning {
        /// Jobs running now.
        running: usize,
        /// The limit.
        limit: usize,
    },
    /// The job has already ended.
    #[error("job is not running")]
    NotRunning,
    /// The job has no output location.
    #[error("`{0}` jobs have no output directory")]
    NoOutput(String),
    /// The process could not be started, or a file it needs could not be made.
    #[error("{0}")]
    Io(String),
    /// A spec could not be turned into a command (a bug in the spec).
    #[error("the job's settings could not be turned into a command: {0}")]
    InvalidSpec(String),
}

/// How a job's process stands.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ProcessStatus {
    /// Alive.
    Running,
    /// Told to stop, not yet gone.
    Stopping,
    /// Exited with status 0.
    Completed {
        /// The exit code.
        exit_code: i32,
    },
    /// Exited with a failure, or was killed by something other than a stop.
    Failed {
        /// The exit code, or -1 when it was a signal.
        exit_code: i32,
        /// Why: the last thing it wrote to standard error, or how it ended.
        error: String,
    },
    /// Ended because it was stopped.
    Stopped,
}

impl ProcessStatus {
    /// Whether the process is still alive (or being told to leave).
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Running | Self::Stopping)
    }
}

/// Serializable summary of a job (without internal handles).
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct JobSummary {
    /// The job's id.
    pub id: JobId,
    /// The subcommand.
    pub command: String,
    /// The subcommand and its arguments, with credentials hidden.
    pub args: Vec<String>,
    /// How the process stands.
    pub status: ProcessStatus,
    /// When it was started.
    pub started_at: DateTime<Utc>,
    /// When it ended.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    /// Where the job writes its results, when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// What stage the job says it is in ("Loading model…").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    /// Progress the job has reported.
    pub metrics: JobMetrics,
    /// Seconds since it started (until it ended).
    pub elapsed_seconds: f64,
}

/// What to run.
#[derive(Debug, Clone, Default)]
pub struct JobRequest {
    /// The `pmetal` subcommand.
    pub subcommand: String,
    /// Its arguments.
    pub argv: Vec<String>,
    /// Where the job writes its results (a directory, or a file for a command
    /// that makes one), if it says.
    pub output: Option<PathBuf>,
    /// A JSONL file of flat step records (`--log-metrics`) the job writes,
    /// read while it runs. The argv must tell the job to write it.
    pub metrics_file: Option<PathBuf>,
    /// A JSONL file of `JobEvent`s (`--log-events`) the job writes, read while
    /// it runs. The argv must tell the job to write it.
    pub events_file: Option<PathBuf>,
    /// Also publish standard output as it arrives, in pieces, not only as
    /// whole lines ([`EventKind::Delta`]). For a job whose output is a reply
    /// being typed out.
    pub deltas: bool,
    /// Environment variables set for the job, on top of what the supervisor's
    /// own process has. For credentials: unlike arguments they are not visible
    /// to other users in the process list, and they are never recorded in
    /// [`JobSummary::args`].
    pub env: Vec<(String, String)>,
}

/// Something that happened to a job.
#[derive(Debug, Clone)]
pub struct Event {
    /// The job it happened to.
    pub job: JobId,
    /// What happened.
    pub kind: EventKind,
}

/// What happened to a job.
#[derive(Debug, Clone)]
pub enum EventKind {
    /// The process started.
    Started,
    /// A whole line of output.
    Output(LogLine),
    /// Standard output as it arrived, when the job asked for it. May end in
    /// the middle of a line; the whole lines follow as [`Output`](Self::Output).
    Delta(String),
    /// An event from the job's own stream (`--log-events`, `--log-metrics` or
    /// `JobEvent` lines on standard output): a phase, a step, a result.
    Job(JobEvent),
    /// The process ended.
    Exited(ProcessStatus),
}

/// Steps the job reported, for a reader that joined late or fell behind.
#[derive(Debug, Clone)]
pub struct StepPage {
    /// Steps after the position asked for, oldest first.
    pub steps: Vec<StepMetrics>,
    /// Pass as `since` to get only what comes after this page.
    pub next_since: u64,
}

struct JobState {
    status: ProcessStatus,
    finished_at: Option<DateTime<Utc>>,
    metrics: JobMetrics,
    phase: Option<String>,
    completion: Option<CompletionSummary>,
    steps: VecDeque<StepMetrics>,
    steps_seen: u64,
}

/// Internal representation of a tracked job.
struct Job {
    id: JobId,
    command: String,
    args: Vec<String>,
    output: Option<PathBuf>,
    started_at: DateTime<Utc>,
    pid: u32,
    deltas: bool,
    state: Mutex<JobState>,
    logs: Mutex<LogBuffer>,
    stop_requested: AtomicBool,
    events: broadcast::Sender<Event>,
}

impl Job {
    fn status(&self) -> ProcessStatus {
        locked(&self.state).status.clone()
    }

    fn emit(&self, kind: EventKind) {
        // No subscriber is not an error.
        let _ = self.events.send(Event {
            job: self.id.clone(),
            kind,
        });
    }

    fn summary(&self) -> JobSummary {
        let state = locked(&self.state);
        let elapsed = state
            .finished_at
            .unwrap_or_else(Utc::now)
            .signed_duration_since(self.started_at)
            .num_milliseconds() as f64
            / 1000.0;
        JobSummary {
            id: self.id.clone(),
            command: self.command.clone(),
            args: self.args.clone(),
            status: state.status.clone(),
            started_at: self.started_at,
            finished_at: state.finished_at,
            output: self.output.as_ref().map(|p| p.display().to_string()),
            phase: state.phase.clone(),
            metrics: state.metrics.clone(),
            elapsed_seconds: elapsed,
        }
    }

    /// Record an event from the job's own stream and publish it.
    fn apply(&self, event: JobEvent) {
        {
            let mut state = locked(&self.state);
            state.metrics.apply(&event);
            match &event {
                JobEvent::Phase { phase, .. } => state.phase = Some(phase.message()),
                JobEvent::Metric {
                    payload: MetricPayload::Step(step),
                    ..
                } => {
                    // The job is past its set-up once it reports a step.
                    state.phase = None;
                    state.steps_seen += 1;
                    state.steps.push_back(step.clone());
                    if state.steps.len() > MAX_STEPS_KEPT {
                        state.steps.pop_front();
                    }
                }
                JobEvent::Completed { summary, .. } => state.completion = Some(summary.clone()),
                _ => {}
            }
        }
        self.emit(EventKind::Job(event));
    }

    /// A line of the job's output has arrived.
    fn line(&self, stream: Stream, raw: &str) {
        let text = redact_secrets(&sanitize_line(raw)).into_owned();
        if stream == Stream::Stdout {
            if let Some(event) = parse_record(&text, &self.id) {
                self.apply(event);
            }
        }
        let seq = locked(&self.logs).push(stream, text.clone());
        self.emit(EventKind::Output(LogLine { seq, stream, text }));
    }
}

struct Inner {
    jobs: Mutex<HashMap<JobId, Arc<Job>>>,
    binary: PathBuf,
    limits: Limits,
    events: broadcast::Sender<Event>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // The last handle is gone: no child outlives it.
        for job in locked(&self.jobs).values() {
            if job.status().is_active() {
                kill_group(job.pid);
            }
        }
    }
}

/// Starts `pmetal` subcommands as child processes and owns them until they end.
///
/// A job is the leader of its own process group, so stopping it reaches
/// everything it spawned. Its output is held in a bounded ring, redacted and
/// stripped of terminal control sequences; its progress is read from the files
/// it writes and from its standard output. Everything that happens is also
/// published to [`subscribe`](Self::subscribe)rs on a bounded channel, so a
/// slow reader loses events rather than slowing a job, and can catch up from
/// the supervisor's own state.
///
/// Cloning gives another handle to the same supervisor. When the last handle is
/// dropped the jobs still running are killed.
#[derive(Clone)]
pub struct Supervisor {
    inner: Arc<Inner>,
}

impl Supervisor {
    /// A supervisor that runs `binary` and enforces `limits`.
    pub fn new(binary: impl Into<PathBuf>, limits: Limits) -> Self {
        let (events, _) = broadcast::channel(EVENT_BACKLOG);
        Self {
            inner: Arc::new(Inner {
                jobs: Mutex::new(HashMap::new()),
                binary: binary.into(),
                limits,
                events,
            }),
        }
    }

    /// The `pmetal` binary jobs run.
    pub fn binary(&self) -> &Path {
        &self.inner.binary
    }

    /// The limits this supervisor enforces.
    pub fn limits(&self) -> &Limits {
        &self.inner.limits
    }

    /// Everything that happens to any job from now on. A receiver that falls
    /// more than a couple of thousand events behind is told how many it
    /// missed (`RecvError::Lagged`) and should read the state it cares about
    /// from [`summary`](Self::summary), [`logs`](Self::logs) and
    /// [`steps`](Self::steps).
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.inner.events.subscribe()
    }

    fn get(&self, id: &str) -> Result<Arc<Job>, Error> {
        locked(&self.inner.jobs)
            .get(id)
            .cloned()
            .ok_or_else(|| Error::NotFound(id.to_string()))
    }

    /// Forget the oldest finished jobs beyond the cap.
    fn prune(&self, jobs: &mut HashMap<JobId, Arc<Job>>) {
        let mut finished: Vec<(DateTime<Utc>, JobId)> = jobs
            .values()
            .filter(|job| !job.status().is_active())
            .map(|job| {
                let ended = locked(&job.state).finished_at.unwrap_or(job.started_at);
                (ended, job.id.clone())
            })
            .collect();
        if finished.len() <= self.inner.limits.max_finished {
            return;
        }
        finished.sort();
        let excess = finished.len() - self.inner.limits.max_finished;
        for (_, id) in finished.into_iter().take(excess) {
            jobs.remove(&id);
        }
    }

    /// Start a job. Must be called from within a tokio runtime. Returns its id.
    pub fn spawn(&self, request: JobRequest) -> Result<JobId, Error> {
        let JobRequest {
            subcommand,
            argv,
            output,
            metrics_file,
            events_file,
            deltas,
            env,
        } = request;
        let id = uuid::Uuid::new_v4().to_string();

        let mut cmd = Command::new(&self.inner.binary);
        cmd.arg(&subcommand)
            .args(&argv)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Output goes to a pipe, not a terminal: colour codes in it are noise.
            .env("NO_COLOR", "1")
            .envs(env)
            // A handle dropped without a stop must not leave a run behind.
            .kill_on_drop(true);
        // Its own process group: a stop signals the group, so helpers the job
        // spawned go with it.
        #[cfg(unix)]
        cmd.process_group(0);

        // The slot is counted and taken under one lock, so two starts cannot
        // both see room for one more.
        let mut guard = locked(&self.inner.jobs);
        let running = guard.values().filter(|j| j.status().is_active()).count();
        if running >= self.inner.limits.max_running {
            return Err(Error::TooManyRunning {
                running,
                limit: self.inner.limits.max_running,
            });
        }

        // Start both files empty so a rerun into the same directory is not
        // read as this run's steps.
        for file in [&metrics_file, &events_file].into_iter().flatten() {
            if let Some(parent) = file.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::Io(format!("cannot create {}: {e}", parent.display())))?;
            }
            std::fs::write(file, b"")
                .map_err(|e| Error::Io(format!("cannot create {}: {e}", file.display())))?;
        }

        let mut child = cmd.spawn().map_err(|e| {
            Error::Io(format!(
                "failed to spawn {} {subcommand}: {e}",
                self.inner.binary.display()
            ))
        })?;
        let pid = child
            .id()
            .ok_or_else(|| Error::Io("the job exited before it could be tracked".to_string()))?;

        let mut full_args = vec![subcommand.clone()];
        full_args.extend(argv);
        let job = Arc::new(Job {
            id: id.clone(),
            command: subcommand,
            args: redact_argv(&full_args),
            output,
            started_at: Utc::now(),
            pid,
            deltas,
            state: Mutex::new(JobState {
                status: ProcessStatus::Running,
                finished_at: None,
                metrics: JobMetrics::default(),
                phase: None,
                completion: None,
                steps: VecDeque::new(),
                steps_seen: 0,
            }),
            logs: Mutex::new(LogBuffer::default()),
            stop_requested: AtomicBool::new(false),
            events: self.inner.events.clone(),
        });
        guard.insert(id.clone(), job.clone());
        self.prune(&mut guard);
        drop(guard);

        job.emit(EventKind::Started);
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let files = Files {
            metrics: metrics_file,
            events: events_file,
        };
        tokio::spawn(supervise(job, child, stdout, stderr, files));
        Ok(id)
    }

    /// Stop a running job: SIGTERM to its process group, SIGKILL if it is still
    /// there after the grace period. Stopping a job that is already stopping,
    /// or that has already been stopped, does nothing; one that ended on its
    /// own cannot be stopped.
    pub fn stop(&self, id: &str) -> Result<(), Error> {
        let job = self.get(id)?;
        {
            let mut state = locked(&job.state);
            match state.status {
                ProcessStatus::Running => state.status = ProcessStatus::Stopping,
                ProcessStatus::Stopping | ProcessStatus::Stopped => return Ok(()),
                _ => return Err(Error::NotRunning),
            }
        }
        job.stop_requested.store(true, Ordering::SeqCst);
        terminate(job, self.inner.limits.stop_grace);
        Ok(())
    }

    /// Wait for a job to end, for at most `limit`. Returns how it stands then:
    /// still active if the limit passed first.
    pub async fn wait(&self, id: &str, limit: Duration) -> Result<ProcessStatus, Error> {
        let job = self.get(id)?;
        let deadline = tokio::time::Instant::now() + limit;
        loop {
            let status = job.status();
            if !status.is_active() || tokio::time::Instant::now() >= deadline {
                return Ok(status);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// How long after a stop a job is certain to be gone: the grace period
    /// before SIGKILL, and the time allowed for its pipes to drain.
    pub fn stop_deadline(&self) -> Duration {
        self.inner.limits.stop_grace + DRAIN_GRACE * 2
    }

    /// Stop every running job and wait for them to leave. Called when the
    /// program is going away so no child outlives it.
    pub async fn shutdown(&self) {
        let active: Vec<Arc<Job>> = locked(&self.inner.jobs)
            .values()
            .filter(|j| j.status().is_active())
            .cloned()
            .collect();
        for job in &active {
            {
                let mut state = locked(&job.state);
                if state.status == ProcessStatus::Running {
                    state.status = ProcessStatus::Stopping;
                }
            }
            job.stop_requested.store(true, Ordering::SeqCst);
            terminate(job.clone(), self.inner.limits.stop_grace);
        }
        let deadline = tokio::time::Instant::now() + self.stop_deadline();
        while active.iter().any(|j| j.status().is_active())
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// Kill every running job's process group now, without waiting. For the
    /// last moments of a program that is going down (a panic hook); a normal
    /// exit should call [`shutdown`](Self::shutdown), which lets jobs finish
    /// what they are writing.
    pub fn kill_all_now(&self) {
        for job in locked(&self.inner.jobs).values() {
            if job.status().is_active() {
                job.stop_requested.store(true, Ordering::SeqCst);
                kill_group(job.pid);
            }
        }
    }

    /// Forget a finished job and its output.
    pub fn forget(&self, id: &str) -> Result<(), Error> {
        let mut jobs = locked(&self.inner.jobs);
        match jobs.get(id) {
            None => Err(Error::NotFound(id.to_string())),
            Some(job) if job.status().is_active() => Err(Error::NotRunning),
            Some(_) => {
                jobs.remove(id);
                Ok(())
            }
        }
    }

    /// The subcommand and output location of a running job.
    pub fn running_output(&self, id: &str) -> Result<(String, PathBuf), Error> {
        let job = self.get(id)?;
        if job.status() != ProcessStatus::Running {
            return Err(Error::NotRunning);
        }
        let dir = job
            .output
            .clone()
            .ok_or_else(|| Error::NoOutput(job.command.clone()))?;
        Ok((job.command.clone(), dir))
    }

    /// A summary of a job.
    pub fn summary(&self, id: &str) -> Result<JobSummary, Error> {
        Ok(self.get(id)?.summary())
    }

    /// Summaries of all jobs, newest first.
    pub fn list(&self) -> Vec<JobSummary> {
        let mut summaries: Vec<JobSummary> = locked(&self.inner.jobs)
            .values()
            .map(|j| j.summary())
            .collect();
        summaries.sort_by_key(|s| std::cmp::Reverse(s.started_at));
        summaries
    }

    /// The number of jobs running (or being stopped).
    pub fn running(&self) -> usize {
        locked(&self.inner.jobs)
            .values()
            .filter(|j| j.status().is_active())
            .count()
    }

    /// The last `tail` lines of a job's output at or after sequence `since`,
    /// both streams in the order they arrived.
    pub fn logs(&self, id: &str, tail: usize, since: Option<u64>) -> Result<LogPage, Error> {
        let job = self.get(id)?;
        Ok(locked(&job.logs).page(tail, since.unwrap_or(0)))
    }

    /// The steps a job reported after position `since` (the number it had
    /// reported when the last page was read; 0 for all that are kept).
    pub fn steps(&self, id: &str, since: u64) -> Result<StepPage, Error> {
        let job = self.get(id)?;
        let state = locked(&job.state);
        let first_kept = state.steps_seen - state.steps.len() as u64;
        let skip = since.saturating_sub(first_kept) as usize;
        Ok(StepPage {
            steps: state.steps.iter().skip(skip).cloned().collect(),
            next_since: state.steps_seen,
        })
    }

    /// What a job said when it finished successfully, if it said anything.
    pub fn completion(&self, id: &str) -> Result<Option<CompletionSummary>, Error> {
        Ok(locked(&self.get(id)?.state).completion.clone())
    }
}

/// SIGTERM the job's process group, then SIGKILL it after `grace` if it is
/// still running.
fn terminate(job: Arc<Job>, grace: Duration) {
    terminate_group(job.pid);
    tokio::spawn(async move {
        tokio::time::sleep(grace).await;
        if job.status().is_active() {
            kill_group(job.pid);
        }
    });
}

struct Files {
    metrics: Option<PathBuf>,
    events: Option<PathBuf>,
}

/// Own a job until it exits: read its output and metrics, record how it ended.
async fn supervise(
    job: Arc<Job>,
    mut child: tokio::process::Child,
    stdout: Option<tokio::process::ChildStdout>,
    stderr: Option<tokio::process::ChildStderr>,
    files: Files,
) {
    let out_task = stdout.map(|s| tokio::spawn(read_stream(job.clone(), s, Stream::Stdout)));
    let err_task = stderr.map(|s| tokio::spawn(read_stream(job.clone(), s, Stream::Stderr)));
    let tail_done = Arc::new(AtomicBool::new(false));
    let tail_tasks: Vec<_> = [files.metrics, files.events]
        .into_iter()
        .flatten()
        .map(|file| tokio::spawn(tail_records(job.clone(), file, tail_done.clone())))
        .collect();

    let exit = child.wait().await;

    // The pipes close when every process holding them is gone. A helper that
    // outlived the job keeps them open, so the wait for them is bounded, and
    // what is left of the group is killed.
    let mut leftovers = false;
    for task in [out_task, err_task].into_iter().flatten() {
        let abort = task.abort_handle();
        if tokio::time::timeout(DRAIN_GRACE, task).await.is_err() {
            abort.abort();
            leftovers = true;
        }
    }
    if leftovers {
        kill_group(job.pid);
    }
    tail_done.store(true, Ordering::SeqCst);
    for task in tail_tasks {
        let _ = task.await;
    }

    let stopped = job.stop_requested.load(Ordering::SeqCst);
    let status = match exit {
        Ok(_) if stopped => ProcessStatus::Stopped,
        Ok(es) if es.success() => ProcessStatus::Completed {
            exit_code: es.code().unwrap_or(0),
        },
        Ok(es) => {
            let code = es.code().unwrap_or(-1);
            let error = locked(&job.logs)
                .last_nonblank(Stream::Stderr)
                .map(|l| l.text.clone())
                .unwrap_or_else(|| describe_exit(&es));
            ProcessStatus::Failed {
                exit_code: code,
                error,
            }
        }
        Err(e) => ProcessStatus::Failed {
            exit_code: -1,
            error: e.to_string(),
        },
    };
    {
        let mut state = locked(&job.state);
        state.status = status.clone();
        state.finished_at = Some(Utc::now());
    }
    job.emit(EventKind::Exited(status));
}

fn describe_exit(status: &std::process::ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;
    match (status.code(), status.signal()) {
        (Some(code), _) => format!("exited with code {code}"),
        (None, Some(signal)) => format!("killed by signal {signal}"),
        (None, None) => "exited abnormally".to_string(),
    }
}

/// Read one of the job's output streams to its end: whole lines into the ring,
/// and, for a job that asked, the text as it arrives.
async fn read_stream<R: tokio::io::AsyncRead + Unpin>(job: Arc<Job>, mut stream: R, which: Stream) {
    let mut lines = LineAssembler::new(MAX_LINE_BYTES);
    let mut text = Utf8Stream::default();
    let mut chunk = vec![0u8; 8 * 1024];
    loop {
        let n = match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        if job.deltas && which == Stream::Stdout {
            let delta = text.push(&chunk[..n]);
            if !delta.is_empty() {
                job.emit(EventKind::Delta(redact_secrets(&delta).into_owned()));
            }
        }
        lines.feed(&chunk[..n], |line| job.line(which, &line));
    }
    lines.finish(|line| job.line(which, &line));
}

/// Follow a JSONL file the job writes, applying each record as it appears
/// until told to stop; then read what is left.
async fn tail_records(job: Arc<Job>, file: PathBuf, done: Arc<AtomicBool>) {
    let mut tail = Tail::new(file);
    loop {
        let finished = done.load(Ordering::SeqCst);
        for line in tail.poll(finished).await {
            if let Some(event) = parse_record(&line, &job.id) {
                job.apply(event);
            }
        }
        if finished {
            return;
        }
        tokio::time::sleep(TAIL_POLL).await;
    }
}
