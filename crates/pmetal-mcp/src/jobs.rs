use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use turbomcp::prelude::*;

use crate::util;

/// Maximum lines to buffer per stream (stdout/stderr) per job.
const MAX_BUFFER_LINES: usize = 10_000;

/// Status of a background job.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JobStatus {
    Running,
    Stopping,
    Completed { exit_code: i32 },
    Failed { exit_code: i32, error: String },
    Stopped,
}

/// Training metrics extracted from pmetal's output.
#[derive(Debug, Clone, Default, Serialize)]
pub struct JobMetrics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_step: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_lr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<f64>,
}

/// Serializable summary of a job (without internal handles).
#[derive(Debug, Serialize)]
pub struct JobSummary {
    pub id: String,
    pub command: String,
    pub args: Vec<String>,
    pub status: JobStatus,
    pub started_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    pub metrics: JobMetrics,
    pub elapsed_seconds: f64,
}

/// Internal representation of a tracked job.
struct Job {
    id: String,
    command: String,
    args: Vec<String>,
    status: Arc<RwLock<JobStatus>>,
    started_at: DateTime<Utc>,
    finished_at: Arc<RwLock<Option<DateTime<Utc>>>>,
    stdout_lines: Arc<RwLock<Vec<String>>>,
    stderr_lines: Arc<RwLock<Vec<String>>>,
    metrics: Arc<RwLock<JobMetrics>>,
    child_pid: Option<u32>,
    stop_requested: Arc<AtomicBool>,
    #[allow(dead_code)]
    handle: JoinHandle<()>,
}

/// Manages the lifecycle of background pmetal subprocesses.
pub struct JobManager {
    jobs: HashMap<String, Job>,
    pmetal_binary: String,
}

impl Default for JobManager {
    fn default() -> Self {
        Self::new()
    }
}

impl JobManager {
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
            pmetal_binary: util::resolve_pmetal_binary(),
        }
    }

    /// Spawn a pmetal subcommand as a background job.
    ///
    /// Returns the job ID.
    pub async fn spawn(&mut self, subcommand: &str, args: Vec<String>) -> McpResult<String> {
        let id = uuid::Uuid::new_v4().to_string();

        let mut cmd = Command::new(&self.pmetal_binary);
        cmd.arg(subcommand);
        for arg in &args {
            cmd.arg(arg);
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::internal(format!("failed to spawn pmetal {subcommand}: {e}")))?;

        let child_pid = child.id();

        let stdout_buf: Arc<RwLock<Vec<String>>> = Arc::new(RwLock::new(Vec::new()));
        let stderr_buf: Arc<RwLock<Vec<String>>> = Arc::new(RwLock::new(Vec::new()));
        let status: Arc<RwLock<JobStatus>> = Arc::new(RwLock::new(JobStatus::Running));
        let finished_at: Arc<RwLock<Option<DateTime<Utc>>>> = Arc::new(RwLock::new(None));
        let metrics: Arc<RwLock<JobMetrics>> = Arc::new(RwLock::new(JobMetrics::default()));
        let stop_requested = Arc::new(AtomicBool::new(false));

        let stdout_handle = child.stdout.take();
        let stderr_handle = child.stderr.take();

        // Clone Arcs for the background task.
        let s_stdout = stdout_buf.clone();
        let s_stderr = stderr_buf.clone();
        let s_status = status.clone();
        let s_finished = finished_at.clone();
        let s_metrics = metrics.clone();
        let s_stop_requested = stop_requested.clone();

        let handle = tokio::spawn(async move {
            // Read stdout in a separate task
            let stdout_task = {
                let buf = s_stdout.clone();
                let met = s_metrics.clone();
                tokio::spawn(async move {
                    if let Some(stdout) = stdout_handle {
                        let reader = BufReader::new(stdout);
                        let mut lines = reader.lines();
                        while let Ok(Some(line)) = lines.next_line().await {
                            // Try to parse as metrics JSONL
                            try_parse_metrics(&line, &met).await;
                            let mut b = buf.write().await;
                            if b.len() >= MAX_BUFFER_LINES {
                                let drain_end = b.len() / 4;
                                b.drain(..drain_end);
                            }
                            b.push(line);
                        }
                    }
                })
            };

            // Read stderr in a separate task
            let stderr_task = {
                let buf = s_stderr.clone();
                tokio::spawn(async move {
                    if let Some(stderr) = stderr_handle {
                        let reader = BufReader::new(stderr);
                        let mut lines = reader.lines();
                        while let Ok(Some(line)) = lines.next_line().await {
                            let mut b = buf.write().await;
                            if b.len() >= MAX_BUFFER_LINES {
                                let drain_end = b.len() / 4;
                                b.drain(..drain_end);
                            }
                            b.push(line);
                        }
                    }
                })
            };

            // Wait for the child process to exit
            let exit_status = child.wait().await;

            // Wait for readers to finish draining
            let _ = stdout_task.await;
            let _ = stderr_task.await;

            // Update status
            let mut s = s_status.write().await;
            match exit_status {
                Ok(es) => {
                    if s_stop_requested.load(Ordering::Relaxed) {
                        *s = JobStatus::Stopped;
                    } else {
                        let code = es.code().unwrap_or(-1);
                        if es.success() {
                            *s = JobStatus::Completed { exit_code: code };
                        } else {
                            let stderr_lines = s_stderr.read().await;
                            let error = stderr_lines
                                .last()
                                .cloned()
                                .unwrap_or_else(|| format!("exited with code {code}"));
                            *s = JobStatus::Failed {
                                exit_code: code,
                                error,
                            };
                        }
                    }
                }
                Err(e) => {
                    *s = JobStatus::Failed {
                        exit_code: -1,
                        error: e.to_string(),
                    };
                }
            }
            *s_finished.write().await = Some(Utc::now());
        });

        let full_args = {
            let mut a = vec![subcommand.to_string()];
            a.extend(args.clone());
            a
        };

        let job = Job {
            id: id.clone(),
            command: subcommand.to_string(),
            args: full_args,
            status,
            started_at: Utc::now(),
            finished_at,
            stdout_lines: stdout_buf,
            stderr_lines: stderr_buf,
            metrics,
            child_pid,
            stop_requested,
            handle,
        };

        self.jobs.insert(id.clone(), job);
        Ok(id)
    }

    /// Stop a running job by sending SIGTERM, then SIGKILL after 5 seconds.
    pub async fn stop(&self, job_id: &str) -> McpResult<()> {
        let job = self
            .jobs
            .get(job_id)
            .ok_or_else(|| McpError::invalid_params(format!("job not found: {job_id}")))?;

        let pid = job
            .child_pid
            .ok_or_else(|| McpError::internal("job has no PID (already exited?)"))?;

        {
            let mut status = job.status.write().await;
            if !matches!(*status, JobStatus::Running) {
                return Err(McpError::invalid_request("job is not running"));
            }
            *status = JobStatus::Stopping;
        }
        job.stop_requested.store(true, Ordering::Relaxed);

        // Send SIGTERM via the kill command (no unsafe needed)
        let _ = std::process::Command::new("kill")
            .args(["-s", "TERM", &pid.to_string()])
            .status();

        // Wait up to 5 seconds, then SIGKILL
        let status_ref = job.status.clone();
        let pid_str = pid.to_string();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let s = status_ref.read().await;
            if status_allows_force_kill(&s) {
                let _ = std::process::Command::new("kill")
                    .args(["-s", "KILL", &pid_str])
                    .status();
            }
        });

        Ok(())
    }

    /// Extract the output directory from a job's CLI args.
    ///
    /// Looks for `--output` followed by a path in the stored args.
    /// Falls back to the subcommand's CLI default output directory.
    pub fn get_output_dir(&self, job_id: &str) -> McpResult<String> {
        let job = self
            .jobs
            .get(job_id)
            .ok_or_else(|| McpError::invalid_params(format!("job not found: {job_id}")))?;

        // Search for --output in args
        let args = &job.args;
        for (i, arg) in args.iter().enumerate() {
            if arg == "--output" || arg == "-o" {
                if let Some(val) = args.get(i + 1) {
                    return Ok(val.clone());
                }
            }
        }
        Ok(default_output_dir_for_command(&job.command).to_string())
    }

    /// Write a control command to a job's control file.
    pub fn write_control(&self, job_id: &str, json: &str) -> McpResult<()> {
        let output_dir = self.get_output_dir(job_id)?;
        let control_path = std::path::Path::new(&output_dir).join(".lr_control.json");
        std::fs::write(&control_path, json).map_err(|e| {
            McpError::internal(format!(
                "failed to write control file {}: {e}",
                control_path.display()
            ))
        })
    }

    /// Get a summary of a specific job.
    pub async fn get_summary(&self, job_id: &str) -> McpResult<JobSummary> {
        let job = self
            .jobs
            .get(job_id)
            .ok_or_else(|| McpError::invalid_params(format!("job not found: {job_id}")))?;

        let status = job.status.read().await.clone();
        let finished = *job.finished_at.read().await;
        let metrics = job.metrics.read().await.clone();

        let elapsed = finished
            .unwrap_or_else(Utc::now)
            .signed_duration_since(job.started_at)
            .num_milliseconds() as f64
            / 1000.0;

        Ok(JobSummary {
            id: job.id.clone(),
            command: job.command.clone(),
            args: job.args.clone(),
            status,
            started_at: job.started_at,
            finished_at: finished,
            metrics,
            elapsed_seconds: elapsed,
        })
    }

    /// Get all job summaries.
    pub async fn list_summaries(&self) -> Vec<JobSummary> {
        let mut summaries = Vec::with_capacity(self.jobs.len());
        for job in self.jobs.values() {
            let status = job.status.read().await.clone();
            let finished = *job.finished_at.read().await;
            let metrics = job.metrics.read().await.clone();

            let elapsed = finished
                .unwrap_or_else(Utc::now)
                .signed_duration_since(job.started_at)
                .num_milliseconds() as f64
                / 1000.0;

            summaries.push(JobSummary {
                id: job.id.clone(),
                command: job.command.clone(),
                args: job.args.clone(),
                status,
                started_at: job.started_at,
                finished_at: finished,
                metrics,
                elapsed_seconds: elapsed,
            });
        }
        summaries.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        summaries
    }

    /// Get recent log lines from a job.
    pub async fn get_logs(&self, job_id: &str, tail: usize) -> McpResult<(Vec<String>, usize)> {
        let job = self
            .jobs
            .get(job_id)
            .ok_or_else(|| McpError::invalid_params(format!("job not found: {job_id}")))?;

        let stdout = job.stdout_lines.read().await;
        let stderr = job.stderr_lines.read().await;

        let total = stdout.len() + stderr.len();

        // Interleave stdout and stderr, taking last `tail` lines.
        // Since we don't have timestamps per line, we just concat stdout then stderr.
        let mut all_lines: Vec<String> = Vec::with_capacity(total);
        for line in stdout.iter() {
            all_lines.push(line.clone());
        }
        for line in stderr.iter() {
            all_lines.push(format!("[stderr] {line}"));
        }

        let start = all_lines.len().saturating_sub(tail);
        Ok((all_lines[start..].to_vec(), total))
    }
}

fn status_allows_force_kill(status: &JobStatus) -> bool {
    matches!(status, JobStatus::Running | JobStatus::Stopping)
}

fn default_output_dir_for_command(command: &str) -> &'static str {
    match command {
        "train" => "./output",
        "distill" => "./output/distilled",
        "grpo" => "./output/grpo",
        "rlkd" => "./output/rlkd",
        "embed-train" => "./output-embed",
        _ => "./output",
    }
}

/// Try to parse a line as JSONL training metrics and update the metrics state.
async fn try_parse_metrics(line: &str, metrics: &Arc<RwLock<JobMetrics>>) {
    // pmetal training output emits JSON lines like:
    // {"step":100,"loss":2.345,"lr":0.0002,"tok_s":1234.5,"epoch":0.5}
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
        let mut m = metrics.write().await;
        if let Some(step) = v.get("step").and_then(|s| s.as_u64()) {
            m.last_step = Some(step);
        }
        if let Some(loss) = v.get("loss").and_then(|s| s.as_f64()) {
            m.last_loss = Some(loss);
        }
        if let Some(lr) = v.get("lr").and_then(|s| s.as_f64()) {
            m.last_lr = Some(lr);
        }
        if let Some(tok_s) = v
            .get("tok_s")
            .or_else(|| v.get("tokens_per_second"))
            .and_then(|s| s.as_f64())
        {
            m.tokens_per_second = Some(tok_s);
        }
        if let Some(epoch) = v.get("epoch").and_then(|s| s.as_f64()) {
            m.epoch = Some(epoch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{JobStatus, default_output_dir_for_command, status_allows_force_kill};

    #[test]
    fn default_output_dirs_match_cli_defaults() {
        assert_eq!(default_output_dir_for_command("train"), "./output");
        assert_eq!(
            default_output_dir_for_command("distill"),
            "./output/distilled"
        );
        assert_eq!(default_output_dir_for_command("grpo"), "./output/grpo");
        assert_eq!(default_output_dir_for_command("rlkd"), "./output/rlkd");
        assert_eq!(
            default_output_dir_for_command("embed-train"),
            "./output-embed"
        );
    }

    #[test]
    fn force_kill_still_applies_while_stopping() {
        assert!(status_allows_force_kill(&JobStatus::Running));
        assert!(status_allows_force_kill(&JobStatus::Stopping));
        assert!(!status_allows_force_kill(&JobStatus::Stopped));
    }
}
