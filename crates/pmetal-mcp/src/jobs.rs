//! Background `pmetal` subprocesses started by the MCP tools.
//!
//! The processes, their bounded output and their metrics are owned by
//! [`pmetal_supervisor::Supervisor`], the same supervisor the terminal UI runs
//! its jobs under. This module adds what is the MCP server's own: the limit on
//! commands that run to completion, how a call to one is answered (progress,
//! cancellation, a time limit, bounded output), and the mapping of the
//! supervisor's errors onto MCP's.
//!
//! A command that runs to completion has a supervisor of its own, apart from
//! the background jobs a client follows: it is not a job to list, and it must
//! neither take a job's slot nor be stopped with `stop_job`.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Duration;

use pmetal_supervisor::{
    Error as SupervisorError, EventKind, Limits, ProcessStatus, Stream, Supervisor, redact_secrets,
};
use tokio::sync::broadcast::error::RecvError;
use turbomcp::prelude::*;

pub use pmetal_supervisor::{
    JobMetrics, JobRequest, JobSummary, LogPage, ProcessStatus as JobStatus,
};

use crate::util;

/// Most standard output a command that runs to completion may produce before
/// the rest is dropped.
const MAX_STDOUT_BYTES: usize = 8 * 1024 * 1024;
/// How much of the end of standard error is kept to explain a failure.
const MAX_STDERR_LINES: usize = 200;
/// How often a command that runs to completion is checked for cancellation.
const POLL: Duration = Duration::from_millis(250);
/// How often a client that asked for progress is told the command still runs.
const HEARTBEAT: Duration = Duration::from_secs(5);
/// How long a command that runs to completion runs before it is stopped,
/// unless `PMETAL_MCP_TIMEOUT_SECS` says otherwise.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60 * 60);
/// Finished commands the supervisor of those keeps; each is forgotten when read.
const KEPT_COMMANDS: usize = 8;

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
        let supervisor = Limits::default();
        Self {
            max_running: Self::DEFAULT_MAX_RUNNING,
            max_finished: supervisor.max_finished,
            stop_grace: supervisor.stop_grace,
            max_blocking: Self::DEFAULT_MAX_BLOCKING,
        }
    }
}

/// Manages the lifecycle of background pmetal subprocesses.
pub struct JobManager {
    supervisor: Supervisor,
    /// Runs the commands that go to completion (see the module docs).
    commands: Supervisor,
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
        let binary = binary.into();
        let supervisor = Supervisor::new(
            binary.clone(),
            Limits {
                max_running: limits.max_running,
                max_finished: limits.max_finished,
                stop_grace: limits.stop_grace,
            },
        );
        // The semaphore in front of `run` is what limits these; the supervisor
        // only has to allow them.
        let commands = Supervisor::new(
            binary,
            Limits {
                max_running: limits.max_blocking.max(1) * 4,
                max_finished: KEPT_COMMANDS,
                stop_grace: limits.stop_grace,
            },
        );
        Self {
            supervisor,
            commands,
            limits,
        }
    }

    /// The `pmetal` binary jobs and commands run.
    pub fn binary(&self) -> &Path {
        self.supervisor.binary()
    }

    /// The limits this manager enforces.
    pub fn limits(&self) -> &JobLimits {
        &self.limits
    }

    /// Start a job. Returns its ID.
    pub async fn spawn(&self, request: JobRequest) -> McpResult<String> {
        self.supervisor.spawn(request).map_err(|e| match e {
            SupervisorError::TooManyRunning { .. } => McpError::invalid_request(format!(
                "{e} Stop one with stop_job, or raise the limit with PMETAL_MCP_MAX_JOBS."
            )),
            other => mcp_error(other),
        })
    }

    /// Stop a running job: SIGTERM to its process group, SIGKILL if it is still
    /// there after the grace period, and return once it is gone. Stopping a job
    /// that is already stopping does nothing.
    pub async fn stop(&self, job_id: &str) -> McpResult<()> {
        self.supervisor.stop(job_id).map_err(mcp_error)?;
        // A job being stopped still holds a slot; a client that starts another
        // right after this returns must find it free.
        self.supervisor
            .wait(job_id, self.supervisor.stop_deadline())
            .await
            .map_err(mcp_error)?;
        Ok(())
    }

    /// Stop every running job and command and wait for them to leave. Called
    /// when the server is going away so no child outlives it.
    pub async fn shutdown(&self) {
        tokio::join!(self.supervisor.shutdown(), self.commands.shutdown());
    }

    /// Run `pmetal <argv>` to completion and return its standard output.
    ///
    /// `argv` is everything after the binary name. While it runs, a client that
    /// asked for progress gets a heartbeat (so it does not give up on a long
    /// command), a cancelled request stops the command, and one that runs past
    /// the timeout (`PMETAL_MCP_TIMEOUT_SECS`) is stopped. Output is bounded:
    /// standard output keeps its first 8 MiB, and a failure reports the end of
    /// standard error.
    pub async fn run(&self, argv: &[String], ctx: &RequestContext) -> McpResult<String> {
        self.run_within(argv, ctx, timeout(), HEARTBEAT).await
    }

    /// [`run`](Self::run) with the time limit and the progress interval given.
    pub async fn run_within(
        &self,
        argv: &[String],
        ctx: &RequestContext,
        limit: Duration,
        heartbeat: Duration,
    ) -> McpResult<String> {
        let (command, args) = argv
            .split_first()
            .ok_or_else(|| McpError::internal("no command to run"))?;
        // Listening before starting, so no output is missed.
        let mut events = self.commands.subscribe();
        let id = self
            .commands
            .spawn(JobRequest {
                subcommand: command.clone(),
                argv: args.to_vec(),
                deltas: true,
                ..JobRequest::default()
            })
            .map_err(|e| McpError::internal(format!("failed to run pmetal: {e}")))?;
        // A request that is dropped (the client went away) takes its command
        // with it.
        let mut guard = Stopping {
            supervisor: &self.commands,
            id: &id,
            armed: true,
        };

        let mut stdout = Vec::<u8>::new();
        let mut dropped = 0usize;
        let mut lost = false;
        let mut stderr = VecDeque::<String>::new();
        let started = tokio::time::Instant::now();
        let mut last_heartbeat = started;
        let mut tick = tokio::time::interval(POLL);
        let outcome = loop {
            tokio::select! {
                event = events.recv() => match event {
                    Ok(event) if event.job == id => match event.kind {
                        EventKind::Delta(text) => {
                            let room = MAX_STDOUT_BYTES.saturating_sub(stdout.len());
                            let take = text.len().min(room);
                            stdout.extend_from_slice(&text.as_bytes()[..take]);
                            dropped += text.len() - take;
                        }
                        EventKind::Output(line) if line.stream == Stream::Stderr => {
                            if stderr.len() == MAX_STDERR_LINES {
                                stderr.pop_front();
                            }
                            stderr.push_back(line.text);
                        }
                        EventKind::Exited(status) => break Outcome::Exited(status),
                        _ => {}
                    },
                    Ok(_) => {}
                    Err(RecvError::Lagged(_)) => lost = true,
                    Err(RecvError::Closed) => {
                        return Err(McpError::internal("the server is shutting down"));
                    }
                },
                _ = tick.tick() => {
                    if ctx.is_cancelled() {
                        break Outcome::Cancelled;
                    }
                    if started.elapsed() >= limit {
                        break Outcome::TimedOut;
                    }
                    if ctx.wants_progress() && last_heartbeat.elapsed() >= heartbeat {
                        last_heartbeat = tokio::time::Instant::now();
                        let _ = ctx
                            .report_progress(
                                started.elapsed().as_secs_f64(),
                                None,
                                Some(&format!(
                                    "pmetal {command} running ({}s)",
                                    started.elapsed().as_secs()
                                )),
                            )
                            .await;
                    }
                }
            }
        };

        if !matches!(outcome, Outcome::Exited(_)) {
            let _ = self.commands.stop(&id);
            let _ = self.commands.wait(&id, self.commands.stop_deadline()).await;
        }
        guard.armed = false;
        let _ = self.commands.forget(&id);

        match outcome {
            Outcome::Cancelled => Err(McpError::cancelled("the request was cancelled")),
            Outcome::TimedOut => Err(McpError::timeout(format!(
                "pmetal {command} did not finish in {}s and was stopped. \
                 Raise PMETAL_MCP_TIMEOUT_SECS, or run it as a job",
                limit.as_secs()
            ))),
            Outcome::Exited(ProcessStatus::Completed { .. }) => {
                let mut text = redact_secrets(&String::from_utf8_lossy(&stdout)).into_owned();
                if dropped > 0 {
                    text.push_str(&format!("\n… [{dropped} more bytes of output dropped]"));
                }
                if lost {
                    text.push_str("\n… [some output was lost: the server could not keep up]");
                }
                Ok(text)
            }
            Outcome::Exited(ProcessStatus::Failed { exit_code, error }) => {
                let detail = if stderr.is_empty() {
                    // Nothing on standard error: what it printed explains it,
                    // else how it ended.
                    let printed = String::from_utf8_lossy(&stdout);
                    if printed.trim().is_empty() {
                        error
                    } else {
                        printed.into_owned()
                    }
                } else {
                    stderr.into_iter().collect::<Vec<_>>().join("\n")
                };
                Err(McpError::internal(format!(
                    "pmetal {command} exited with code {exit_code}: {}",
                    tail_chars(redact_secrets(detail.trim()).as_ref(), 4000)
                )))
            }
            // Stopped by something else (a job stop, the server going away).
            Outcome::Exited(_) => Err(McpError::cancelled(format!(
                "pmetal {command} was stopped before it finished"
            ))),
        }
    }

    /// The output directory of a running job, and its subcommand.
    pub fn running_output(&self, job_id: &str) -> McpResult<(String, PathBuf)> {
        self.supervisor.running_output(job_id).map_err(mcp_error)
    }

    /// Get a summary of a specific job.
    pub fn get_summary(&self, job_id: &str) -> McpResult<JobSummary> {
        self.supervisor.summary(job_id).map_err(mcp_error)
    }

    /// Get all job summaries, newest first.
    pub fn list_summaries(&self) -> Vec<JobSummary> {
        self.supervisor.list()
    }

    /// The last `tail` lines of a job's output after sequence `since`, both
    /// streams in the order they arrived.
    pub fn get_logs(&self, job_id: &str, tail: usize, since: Option<u64>) -> McpResult<LogPage> {
        self.supervisor.logs(job_id, tail, since).map_err(mcp_error)
    }
}

/// How a command that runs to completion ended.
enum Outcome {
    Exited(ProcessStatus),
    Cancelled,
    TimedOut,
}

/// Stops a command when the request that started it is dropped.
struct Stopping<'a> {
    supervisor: &'a Supervisor,
    id: &'a str,
    armed: bool,
}

impl Drop for Stopping<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.supervisor.stop(self.id);
        }
    }
}

/// How long a command that runs to completion may run.
fn timeout() -> Duration {
    std::env::var("PMETAL_MCP_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .map_or(DEFAULT_TIMEOUT, Duration::from_secs)
}

/// The last `max` characters of `text`.
fn tail_chars(text: &str, max: usize) -> String {
    let count = text.chars().count();
    if count <= max {
        return text.to_string();
    }
    format!("…{}", text.chars().skip(count - max).collect::<String>())
}

fn mcp_error(e: SupervisorError) -> McpError {
    match e {
        SupervisorError::NotFound(_) => McpError::invalid_params(e.to_string()),
        SupervisorError::NotRunning
        | SupervisorError::NoOutput(_)
        | SupervisorError::TooManyRunning { .. } => McpError::invalid_request(e.to_string()),
        SupervisorError::Io(_) | SupervisorError::InvalidSpec(_) => {
            McpError::internal(e.to_string())
        }
    }
}
