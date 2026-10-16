//! Background `pmetal` subprocesses started by the MCP tools.
//!
//! A job is one `pmetal <subcommand> <argv…>` child in its own process group, so
//! stopping it reaches everything it spawned. Its output is held in a bounded
//! ring (lines are cut, the ring is capped by count and by bytes), so a job that
//! prints without end costs a fixed amount of memory. Metrics come from the
//! `--log-metrics` JSONL file the manager asks the trainer to write, and from
//! `JobEvent` lines on stdout when a command emits them.
//!
//! The manager also owns the limits a long-lived server needs: a cap on jobs
//! running at once, a cap on finished jobs it remembers, and a shutdown that
//! stops what is still running so no child outlives the server.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncSeekExt, BufReader};
use tokio::process::Command;
use turbomcp::prelude::*;

use crate::util;

/// Most lines kept per job (both streams together).
const MAX_LOG_LINES: usize = 10_000;
/// Most bytes kept per job (both streams together).
const MAX_LOG_BYTES: usize = 4 * 1024 * 1024;
/// Longest line kept whole; the rest of a longer line is dropped.
const MAX_LINE_BYTES: usize = 16 * 1024;
/// Longest a stalled pipe is waited for after its process has exited.
const DRAIN_GRACE: Duration = Duration::from_secs(2);
/// How often the metrics file is read.
const METRICS_POLL: Duration = Duration::from_millis(250);

/// Limits on what the manager runs and remembers.
#[derive(Debug, Clone)]
pub struct JobLimits {
    /// Jobs that may run at once; starting one more is refused.
    pub max_running: usize,
    /// Finished jobs kept for inspection; the oldest are forgotten first.
    pub max_finished: usize,
    /// How long a stopped job has to exit after SIGTERM before SIGKILL.
    pub stop_grace: Duration,
    /// Commands that run to completion (generation, evaluation, the dataset
    /// tools) at once; the rest wait their turn.
    pub max_blocking: usize,
}

impl JobLimits {
    /// The default for `PMETAL_MCP_MAX_JOBS` when it is unset.
    pub const DEFAULT_MAX_RUNNING: usize = 4;
    /// The default for `PMETAL_MCP_MAX_BLOCKING` when it is unset.
    pub const DEFAULT_MAX_BLOCKING: usize = 2;

    /// Limits from the environment: `PMETAL_MCP_MAX_JOBS` sets the running cap
    /// and `PMETAL_MCP_MAX_BLOCKING` the commands that run to completion at once.
    pub fn from_env() -> Self {
        let positive = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(default)
        };
        Self {
            max_running: positive("PMETAL_MCP_MAX_JOBS", Self::DEFAULT_MAX_RUNNING),
            max_blocking: positive("PMETAL_MCP_MAX_BLOCKING", Self::DEFAULT_MAX_BLOCKING),
            ..Self::default()
        }
    }
}

impl Default for JobLimits {
    fn default() -> Self {
        Self {
            max_running: Self::DEFAULT_MAX_RUNNING,
            max_finished: 50,
            stop_grace: Duration::from_secs(5),
            max_blocking: Self::DEFAULT_MAX_BLOCKING,
        }
    }
}

/// Status of a background job.
#[derive(Debug, Clone, PartialEq, Serialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JobStatus {
    Running,
    Stopping,
    Completed { exit_code: i32 },
    Failed { exit_code: i32, error: String },
    Stopped,
}

impl JobStatus {
    /// Whether the process is still alive (or being told to leave).
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Running | Self::Stopping)
    }
}

/// Training metrics extracted from a job's output.
#[derive(Debug, Clone, Default, Serialize, schemars::JsonSchema)]
pub struct JobMetrics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_step: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_steps: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_lr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grad_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
    /// Current epoch, counted from 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_epochs: Option<u64>,
}

/// Serializable summary of a job (without internal handles).
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct JobSummary {
    pub id: String,
    pub command: String,
    pub args: Vec<String>,
    pub status: JobStatus,
    pub started_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    /// Where the job writes its results, when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    pub metrics: JobMetrics,
    pub elapsed_seconds: f64,
}

/// Which stream a log line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stream {
    Stdout,
    Stderr,
}

#[derive(Debug)]
struct LogLine {
    seq: u64,
    stream: Stream,
    text: String,
}

/// Lines of a job's output in arrival order, bounded by count and bytes.
#[derive(Default)]
struct LogBuffer {
    lines: VecDeque<LogLine>,
    bytes: usize,
    next_seq: u64,
}

impl LogBuffer {
    fn push(&mut self, stream: Stream, text: String) {
        self.bytes += text.len();
        self.lines.push_back(LogLine {
            seq: self.next_seq,
            stream,
            text,
        });
        self.next_seq += 1;
        while self.lines.len() > MAX_LOG_LINES || self.bytes > MAX_LOG_BYTES {
            match self.lines.pop_front() {
                Some(dropped) => self.bytes -= dropped.text.len(),
                None => break,
            }
        }
    }
}

/// A page of a job's output.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LogPage {
    /// The lines, oldest first; stderr lines are prefixed `[stderr] `.
    pub lines: Vec<String>,
    /// Lines the job has printed in all, including ones no longer kept.
    pub total_lines: u64,
    /// Pass as `since` to get only what comes after this page.
    pub next_since: u64,
    /// Lines that were printed after `since` but have already been dropped.
    pub dropped_lines: u64,
}

struct JobState {
    status: JobStatus,
    finished_at: Option<DateTime<Utc>>,
    metrics: JobMetrics,
}

/// Internal representation of a tracked job.
struct Job {
    id: String,
    command: String,
    args: Vec<String>,
    output: Option<PathBuf>,
    started_at: DateTime<Utc>,
    pid: u32,
    state: Mutex<JobState>,
    logs: Mutex<LogBuffer>,
    stop_requested: AtomicBool,
}

fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panicking reader task leaves the data usable; a poisoned lock must not
    // take the whole server's job view down with it.
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Job {
    fn status(&self) -> JobStatus {
        locked(&self.state).status.clone()
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
            metrics: state.metrics.clone(),
            elapsed_seconds: elapsed,
        }
    }
}

/// What to run.
#[derive(Debug, Clone)]
pub struct JobRequest {
    /// The `pmetal` subcommand.
    pub subcommand: String,
    /// Its arguments.
    pub argv: Vec<String>,
    /// Where the job writes its results (a directory, or a file for a command
    /// that makes one), if it says.
    pub output: Option<PathBuf>,
    /// A JSONL file the job writes metrics to, read while it runs.
    pub metrics_file: Option<PathBuf>,
}

/// Manages the lifecycle of background pmetal subprocesses.
pub struct JobManager {
    jobs: Mutex<HashMap<String, Arc<Job>>>,
    binary: PathBuf,
    limits: JobLimits,
}

impl Default for JobManager {
    fn default() -> Self {
        Self::new()
    }
}

impl JobManager {
    pub fn new() -> Self {
        Self::with_binary(util::resolve_pmetal_binary(), JobLimits::from_env())
    }

    /// A manager that runs `binary` and enforces `limits`.
    pub fn with_binary(binary: impl Into<PathBuf>, limits: JobLimits) -> Self {
        Self {
            jobs: Mutex::new(HashMap::new()),
            binary: binary.into(),
            limits,
        }
    }

    /// The `pmetal` binary jobs and commands run.
    pub fn binary(&self) -> &Path {
        &self.binary
    }

    /// The limits this manager enforces.
    pub fn limits(&self) -> &JobLimits {
        &self.limits
    }

    fn get(&self, job_id: &str) -> McpResult<Arc<Job>> {
        locked(&self.jobs)
            .get(job_id)
            .cloned()
            .ok_or_else(|| McpError::invalid_params(format!("job not found: {job_id}")))
    }

    /// Forget the oldest finished jobs beyond the cap.
    fn prune(&self, jobs: &mut HashMap<String, Arc<Job>>) {
        let mut finished: Vec<(DateTime<Utc>, String)> = jobs
            .values()
            .filter(|job| !job.status().is_active())
            .map(|job| {
                let ended = locked(&job.state).finished_at.unwrap_or(job.started_at);
                (ended, job.id.clone())
            })
            .collect();
        if finished.len() <= self.limits.max_finished {
            return;
        }
        finished.sort();
        let excess = finished.len() - self.limits.max_finished;
        for (_, id) in finished.into_iter().take(excess) {
            jobs.remove(&id);
        }
    }

    /// Start a job. Returns its ID.
    pub async fn spawn(&self, request: JobRequest) -> McpResult<String> {
        let JobRequest {
            subcommand,
            argv,
            output,
            metrics_file,
        } = request;
        let id = uuid::Uuid::new_v4().to_string();

        let mut cmd = Command::new(&self.binary);
        cmd.arg(&subcommand)
            .args(&argv)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Neither the server's death nor a dropped handle may leave a
            // training run behind.
            .kill_on_drop(true);
        // Its own process group: a stop signals the group, so helpers the job
        // spawned go with it.
        #[cfg(unix)]
        cmd.process_group(0);

        // The slot is counted and taken under one lock, so two starts cannot both
        // see room for one more.
        let mut guard = locked(&self.jobs);
        let running = guard.values().filter(|j| j.status().is_active()).count();
        if running >= self.limits.max_running {
            return Err(McpError::invalid_request(format!(
                "{running} jobs are already running (the limit is {}). Stop one with stop_job, \
                 wait for one to finish, or raise PMETAL_MCP_MAX_JOBS",
                self.limits.max_running
            )));
        }

        if let Some(file) = &metrics_file {
            if let Some(parent) = file.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    McpError::internal(format!("cannot create {}: {e}", parent.display()))
                })?;
            }
            // Start empty so a rerun into the same directory is not read as
            // this run's steps.
            std::fs::write(file, b"").map_err(|e| {
                McpError::internal(format!("cannot create {}: {e}", file.display()))
            })?;
        }

        let mut child = cmd.spawn().map_err(|e| {
            McpError::internal(format!(
                "failed to spawn {} {subcommand}: {e}",
                self.binary.display()
            ))
        })?;
        let pid = child
            .id()
            .ok_or_else(|| McpError::internal("the job exited before it could be tracked"))?;

        let mut full_args = vec![subcommand.clone()];
        full_args.extend(argv);
        let job = Arc::new(Job {
            id: id.clone(),
            command: subcommand,
            args: full_args,
            output,
            started_at: Utc::now(),
            pid,
            state: Mutex::new(JobState {
                status: JobStatus::Running,
                finished_at: None,
                metrics: JobMetrics::default(),
            }),
            logs: Mutex::new(LogBuffer::default()),
            stop_requested: AtomicBool::new(false),
        });
        guard.insert(id.clone(), job.clone());
        self.prune(&mut guard);
        drop(guard);

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        tokio::spawn(supervise(job, child, stdout, stderr, metrics_file));
        Ok(id)
    }

    /// Stop a running job: SIGTERM to its process group, SIGKILL if it is still
    /// there after the grace period. Stopping a job that is already stopping
    /// does nothing.
    pub async fn stop(&self, job_id: &str) -> McpResult<()> {
        let job = self.get(job_id)?;
        {
            let mut state = locked(&job.state);
            match state.status {
                JobStatus::Running => state.status = JobStatus::Stopping,
                JobStatus::Stopping => return Ok(()),
                _ => return Err(McpError::invalid_request("job is not running")),
            }
        }
        job.stop_requested.store(true, Ordering::SeqCst);
        terminate(job, self.limits.stop_grace).await;
        Ok(())
    }

    /// Stop every running job and wait for them to leave. Called when the
    /// server is going away so no child outlives it.
    pub async fn shutdown(&self) {
        let active: Vec<Arc<Job>> = locked(&self.jobs)
            .values()
            .filter(|j| j.status().is_active())
            .cloned()
            .collect();
        for job in &active {
            {
                let mut state = locked(&job.state);
                if state.status == JobStatus::Running {
                    state.status = JobStatus::Stopping;
                }
            }
            job.stop_requested.store(true, Ordering::SeqCst);
            terminate(job.clone(), self.limits.stop_grace).await;
        }
        let deadline = tokio::time::Instant::now() + self.limits.stop_grace + DRAIN_GRACE * 2;
        while active.iter().any(|j| j.status().is_active())
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// The output directory of a running job, and its subcommand.
    pub fn running_output(&self, job_id: &str) -> McpResult<(String, PathBuf)> {
        let job = self.get(job_id)?;
        if job.status() != JobStatus::Running {
            return Err(McpError::invalid_request("job is not running"));
        }
        let dir = job.output.clone().ok_or_else(|| {
            McpError::invalid_request(format!("`{}` jobs have no output directory", job.command))
        })?;
        Ok((job.command.clone(), dir))
    }

    /// Get a summary of a specific job.
    pub fn get_summary(&self, job_id: &str) -> McpResult<JobSummary> {
        Ok(self.get(job_id)?.summary())
    }

    /// Get all job summaries, newest first.
    pub fn list_summaries(&self) -> Vec<JobSummary> {
        let mut summaries: Vec<JobSummary> =
            locked(&self.jobs).values().map(|j| j.summary()).collect();
        summaries.sort_by_key(|s| std::cmp::Reverse(s.started_at));
        summaries
    }

    /// The last `tail` lines of a job's output after sequence `since`, both
    /// streams in the order they arrived.
    pub fn get_logs(&self, job_id: &str, tail: usize, since: Option<u64>) -> McpResult<LogPage> {
        let job = self.get(job_id)?;
        let logs = locked(&job.logs);
        let since = since.unwrap_or(0);
        let first_kept = logs.lines.front().map_or(logs.next_seq, |l| l.seq);
        let wanted: Vec<&LogLine> = logs.lines.iter().filter(|l| l.seq >= since).collect();
        let start = wanted.len().saturating_sub(tail);
        let lines = wanted[start..]
            .iter()
            .map(|l| match l.stream {
                Stream::Stdout => l.text.clone(),
                Stream::Stderr => format!("[stderr] {}", l.text),
            })
            .collect();
        Ok(LogPage {
            lines,
            total_lines: logs.next_seq,
            next_since: logs.next_seq,
            dropped_lines: first_kept.saturating_sub(since),
        })
    }
}

/// SIGTERM the job's process group, then SIGKILL it after `grace` if it is
/// still running.
async fn terminate(job: Arc<Job>, grace: Duration) {
    signal_group(job.pid, "TERM").await;
    tokio::spawn(async move {
        tokio::time::sleep(grace).await;
        if job.status().is_active() {
            signal_group(job.pid, "KILL").await;
        }
    });
}

/// Send `signal` to the process group led by `pid`.
///
/// The workspace forbids `unsafe`, so this goes through `kill(1)` rather than
/// `killpg(2)`; a negative PID names the group.
async fn signal_group(pid: u32, signal: &str) {
    let group = format!("-{pid}");
    let sent = Command::new("kill")
        .args(["-s", signal, "--", &group])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .is_ok_and(|s| s.success());
    if !sent {
        // The group is gone or never formed; the leader may still be there.
        let _ = Command::new("kill")
            .args(["-s", signal, &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }
}

/// Own a job until it exits: read its output and metrics, record how it ended.
async fn supervise(
    job: Arc<Job>,
    mut child: tokio::process::Child,
    stdout: Option<tokio::process::ChildStdout>,
    stderr: Option<tokio::process::ChildStderr>,
    metrics_file: Option<PathBuf>,
) {
    let out_task = stdout.map(|s| tokio::spawn(read_stream(job.clone(), s, Stream::Stdout)));
    let err_task = stderr.map(|s| tokio::spawn(read_stream(job.clone(), s, Stream::Stderr)));
    let tail_done = Arc::new(AtomicBool::new(false));
    let metrics_task =
        metrics_file.map(|file| tokio::spawn(tail_metrics(job.clone(), file, tail_done.clone())));

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
        signal_group(job.pid, "KILL").await;
    }
    tail_done.store(true, Ordering::SeqCst);
    if let Some(task) = metrics_task {
        let _ = task.await;
    }

    let stopped = job.stop_requested.load(Ordering::SeqCst);
    let status = match exit {
        Ok(_) if stopped => JobStatus::Stopped,
        Ok(es) if es.success() => JobStatus::Completed {
            exit_code: es.code().unwrap_or(0),
        },
        Ok(es) => {
            let code = es.code().unwrap_or(-1);
            let error = locked(&job.logs)
                .lines
                .iter()
                .rev()
                .find(|l| l.stream == Stream::Stderr && !l.text.trim().is_empty())
                .map(|l| l.text.clone())
                .unwrap_or_else(|| format!("exited with code {code}"));
            JobStatus::Failed {
                exit_code: code,
                error,
            }
        }
        Err(e) => JobStatus::Failed {
            exit_code: -1,
            error: e.to_string(),
        },
    };
    let mut state = locked(&job.state);
    state.status = status;
    state.finished_at = Some(Utc::now());
}

/// Read one line of at most `max` bytes. What a longer line has beyond `max` is
/// read and dropped, so a stream with no newline cannot grow memory.
async fn read_bounded_line<R: AsyncBufReadExt + Unpin>(
    reader: &mut R,
    max: usize,
) -> std::io::Result<Option<String>> {
    let mut line: Vec<u8> = Vec::new();
    let mut dropped = 0usize;
    let mut saw_any = false;
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            break;
        }
        saw_any = true;
        let (chunk, found_newline) = match available.iter().position(|&b| b == b'\n') {
            Some(i) => (&available[..i], true),
            None => (available, false),
        };
        let room = max.saturating_sub(line.len());
        let keep = chunk.len().min(room);
        line.extend_from_slice(&chunk[..keep]);
        dropped += chunk.len() - keep;
        let consumed = chunk.len() + usize::from(found_newline);
        reader.consume(consumed);
        if found_newline {
            break;
        }
    }
    if !saw_any {
        return Ok(None);
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    let mut text = String::from_utf8_lossy(&line).into_owned();
    if dropped > 0 {
        text.push_str(&format!(" … [{dropped} more bytes dropped]"));
    }
    Ok(Some(text))
}

async fn read_stream<R: AsyncRead + Unpin>(job: Arc<Job>, stream: R, which: Stream) {
    let mut reader = BufReader::new(stream);
    while let Ok(Some(line)) = read_bounded_line(&mut reader, MAX_LINE_BYTES).await {
        if which == Stream::Stdout {
            apply_metrics_line(&line, &mut locked(&job.state).metrics);
        }
        locked(&job.logs).push(which, util::redact_secrets(&line).into_owned());
    }
}

/// Follow a metrics JSONL file, applying each complete line as it appears, until
/// told to stop; then read what is left.
async fn tail_metrics(job: Arc<Job>, file: PathBuf, done: Arc<AtomicBool>) {
    let mut offset: u64 = 0;
    let mut pending: Vec<u8> = Vec::new();
    loop {
        let finished = done.load(Ordering::SeqCst);
        // The file may not exist yet, or briefly; that is not an error.
        if let Ok(bytes) = read_new_bytes(&file, &mut offset).await {
            pending.extend_from_slice(&bytes);
            while let Some(end) = pending.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = pending.drain(..=end).collect();
                let line = String::from_utf8_lossy(&line[..line.len() - 1]).into_owned();
                apply_metrics_line(&line, &mut locked(&job.state).metrics);
            }
            // A line the writer never ended with a newline still counts once
            // the job is over.
            if finished && !pending.is_empty() {
                let line = String::from_utf8_lossy(&pending).into_owned();
                apply_metrics_line(&line, &mut locked(&job.state).metrics);
                pending.clear();
            }
            // A runaway "line" with no newline is not metrics.
            if pending.len() > MAX_LINE_BYTES {
                pending.clear();
            }
        }
        if finished {
            return;
        }
        tokio::time::sleep(METRICS_POLL).await;
    }
}

/// The bytes appended to `path` since `offset`, advancing it. A file that
/// shrank was rewritten and is read from its start.
async fn read_new_bytes(path: &Path, offset: &mut u64) -> std::io::Result<Vec<u8>> {
    const CHUNK: u64 = 256 * 1024;
    let mut file = tokio::fs::File::open(path).await?;
    let len = file.metadata().await?.len();
    if len < *offset {
        *offset = 0;
    }
    if len == *offset {
        return Ok(Vec::new());
    }
    file.seek(std::io::SeekFrom::Start(*offset)).await?;
    let mut buf = Vec::new();
    let read = file.take(CHUNK).read_to_end(&mut buf).await?;
    *offset += read as u64;
    Ok(buf)
}

/// Update `metrics` from one line: a `JobEvent` (what a command emits with
/// `--log-events`) or a flat metrics record (what `--log-metrics` writes).
/// Lines that are neither, and records about something other than a step, are
/// ignored.
fn apply_metrics_line(line: &str, metrics: &mut JobMetrics) {
    let line = line.trim();
    if !line.starts_with('{') {
        return;
    }
    if let Ok(event) = pmetal_core::events::parse_event(line) {
        apply_job_event(&event, metrics);
        return;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
        return;
    };
    let Some(record) = value.as_object() else {
        return;
    };
    if let Some(event) = record.get("event").and_then(|e| e.as_str()) {
        // `epoch_end` carries the evaluation loss; the other events carry no step.
        if event == "epoch_end" {
            if let Some(loss) = record.get("loss").and_then(|v| v.as_f64()) {
                metrics.eval_loss = Some(loss);
            }
        }
        return;
    }
    if let Some(step) = record.get("step").and_then(|v| v.as_u64()) {
        metrics.last_step = Some(step);
    }
    if let Some(total) = record
        .get("total_steps")
        .and_then(|v| v.as_u64())
        .filter(|&t| t > 0)
    {
        metrics.total_steps = Some(total);
    }
    if let Some(loss) = record.get("loss").and_then(|v| v.as_f64()) {
        metrics.last_loss = Some(loss);
    }
    if let Some(lr) = record.get("lr").and_then(|v| v.as_f64()) {
        metrics.last_lr = Some(lr);
    }
    if let Some(norm) = record.get("grad_norm").and_then(|v| v.as_f64()) {
        metrics.grad_norm = Some(norm);
    }
    if let Some(tps) = ["tok_sec", "tok_s", "tokens_per_second"]
        .iter()
        .find_map(|k| record.get(*k).and_then(|v| v.as_f64()))
        .filter(|&t| t > 0.0)
    {
        metrics.tokens_per_second = Some(tps);
    }
    if let Some(epoch) = record.get("epoch").and_then(|v| v.as_f64()) {
        // The trainer counts epochs from 0.
        metrics.epoch = Some(epoch + 1.0);
    }
    if let Some(total) = record
        .get("total_epochs")
        .and_then(|v| v.as_u64())
        .filter(|&t| t > 0)
    {
        metrics.total_epochs = Some(total);
    }
}

fn apply_job_event(event: &pmetal_core::JobEvent, metrics: &mut JobMetrics) {
    use pmetal_core::events::{JobEvent, MetricPayload};
    let JobEvent::Metric { payload, .. } = event else {
        return;
    };
    match payload {
        MetricPayload::Step(step) => {
            metrics.last_step = Some(step.step as u64);
            if step.total_steps > 0 {
                metrics.total_steps = Some(step.total_steps as u64);
            }
            metrics.last_loss = Some(step.loss);
            metrics.last_lr = Some(step.lr);
            metrics.grad_norm = step.grad_norm.or(metrics.grad_norm);
            if step.tok_sec > 0.0 {
                metrics.tokens_per_second = Some(step.tok_sec);
            }
            metrics.epoch = Some((step.epoch + 1) as f64);
            if step.total_epochs > 0 {
                metrics.total_epochs = Some(step.total_epochs as u64);
            }
        }
        MetricPayload::Eval(eval) => metrics.eval_loss = Some(eval.loss),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_step_record_updates_the_metrics() {
        let mut m = JobMetrics::default();
        apply_metrics_line(
            r#"{"step":7,"epoch":0,"total_epochs":3,"total_steps":90,"loss":2.5,"lr":0.0002,"tok_sec":1234.5,"grad_norm":0.8,"timestamp":"t"}"#,
            &mut m,
        );
        assert_eq!(m.last_step, Some(7));
        assert_eq!(m.total_steps, Some(90));
        assert_eq!(m.last_loss, Some(2.5));
        assert_eq!(m.last_lr, Some(0.0002));
        assert_eq!(m.tokens_per_second, Some(1234.5));
        assert_eq!(m.grad_norm, Some(0.8));
        // Counted from 1, as the JobEvent path counts it.
        assert_eq!(m.epoch, Some(1.0));
        assert_eq!(m.total_epochs, Some(3));
    }

    #[test]
    fn an_epoch_record_is_the_eval_loss_not_the_training_loss() {
        let mut m = JobMetrics::default();
        apply_metrics_line(r#"{"step":3,"loss":2.0}"#, &mut m);
        apply_metrics_line(
            r#"{"event":"epoch_end","epoch":0,"loss":1.5,"perplexity":4.4}"#,
            &mut m,
        );
        assert_eq!(m.last_loss, Some(2.0));
        assert_eq!(m.eval_loss, Some(1.5));
        // Lifecycle records are not steps.
        apply_metrics_line(r#"{"event":"train_start","timestamp":"t"}"#, &mut m);
        assert_eq!(m.last_step, Some(3));
    }

    #[test]
    fn text_that_is_not_json_is_ignored() {
        let mut m = JobMetrics::default();
        apply_metrics_line("Epoch 1/3 loss=2.0", &mut m);
        apply_metrics_line("{not json", &mut m);
        apply_metrics_line("[1,2,3]", &mut m);
        assert!(m.last_step.is_none() && m.last_loss.is_none());
    }

    #[test]
    fn the_log_ring_is_bounded_by_lines_and_by_bytes() {
        let mut buf = LogBuffer::default();
        for i in 0..(MAX_LOG_LINES + 50) {
            buf.push(Stream::Stdout, format!("line {i}"));
        }
        assert_eq!(buf.lines.len(), MAX_LOG_LINES);
        assert_eq!(buf.lines.front().unwrap().seq, 50);
        assert_eq!(buf.next_seq, (MAX_LOG_LINES + 50) as u64);

        let mut buf = LogBuffer::default();
        let big = "x".repeat(MAX_LINE_BYTES);
        for _ in 0..(MAX_LOG_BYTES / MAX_LINE_BYTES + 10) {
            buf.push(Stream::Stderr, big.clone());
        }
        assert!(buf.bytes <= MAX_LOG_BYTES, "{}", buf.bytes);
    }

    #[tokio::test]
    async fn a_line_without_a_newline_is_cut_not_buffered() {
        let data = vec![b'a'; 1_000_000];
        let mut reader = BufReader::new(&data[..]);
        let line = read_bounded_line(&mut reader, 1024).await.unwrap().unwrap();
        assert!(line.len() < 1024 + 64, "{}", line.len());
        assert!(line.contains("more bytes dropped"), "{line}");
        assert!(
            read_bounded_line(&mut reader, 1024)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn lines_split_on_newlines_and_crlf() {
        let mut reader = BufReader::new(&b"one\r\ntwo\n\nlast"[..]);
        let mut got = Vec::new();
        while let Some(line) = read_bounded_line(&mut reader, 1024).await.unwrap() {
            got.push(line);
        }
        assert_eq!(got, ["one", "two", "", "last"]);
    }
}
