//! Background `pmetal` subprocesses started by the MCP tools.
//!
//! The processes, their bounded output and their metrics are owned by
//! [`pmetal_supervisor::Supervisor`], the same supervisor the terminal UI runs
//! its jobs under. This module adds what is the MCP server's own: the limit on
//! commands that run to completion, and the mapping of the supervisor's errors
//! onto MCP's.

use std::path::{Path, PathBuf};
use std::time::Duration;

use pmetal_supervisor::{Error as SupervisorError, Limits, Supervisor};
use turbomcp::prelude::*;

pub use pmetal_supervisor::{
    JobMetrics, JobRequest, JobSummary, LogPage, ProcessStatus as JobStatus,
};

use crate::util;

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
        let supervisor = Supervisor::new(
            binary,
            Limits {
                max_running: limits.max_running,
                max_finished: limits.max_finished,
                stop_grace: limits.stop_grace,
            },
        );
        Self { supervisor, limits }
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
                "{e}. Stop one with stop_job, wait for one to finish, or raise \
                 PMETAL_MCP_MAX_JOBS"
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

    /// Stop every running job and wait for them to leave. Called when the
    /// server is going away so no child outlives it.
    pub async fn shutdown(&self) {
        self.supervisor.shutdown().await;
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

fn mcp_error(e: SupervisorError) -> McpError {
    match e {
        SupervisorError::NotFound(_) => McpError::invalid_params(e.to_string()),
        SupervisorError::NotRunning
        | SupervisorError::NoOutput(_)
        | SupervisorError::TooManyRunning { .. } => McpError::invalid_request(e.to_string()),
        SupervisorError::Io(_) => McpError::internal(e.to_string()),
    }
}
