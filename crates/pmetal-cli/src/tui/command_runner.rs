//! Background command runner for spawning pmetal subcommands.
//!
//! Manages child processes for training, inference, distillation, GRPO,
//! and model downloads. Streams output and metrics back to the TUI via
//! `AppMsg` on the event channel.

use std::collections::HashMap;
use std::path::PathBuf;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::tui::event::{AppMsg, CommandSpec, JobType};

/// A currently running background job.
#[allow(dead_code)]
pub struct RunningJob {
    pub id: String,
    pub job_type: JobType,
    pub cancel: CancellationToken,
    pub metrics_file: Option<PathBuf>,
}

/// Manages background pmetal child processes.
pub struct CommandRunner {
    app_tx: mpsc::UnboundedSender<AppMsg>,
    jobs: HashMap<String, RunningJob>,
    next_id: u32,
}

impl CommandRunner {
    pub fn new(app_tx: mpsc::UnboundedSender<AppMsg>) -> Self {
        Self {
            app_tx,
            jobs: HashMap::new(),
            next_id: 0,
        }
    }

    /// Spawn a background job from a command spec. Returns the job ID.
    pub fn spawn(&mut self, spec: CommandSpec) -> String {
        let job_id = format!("job-{}", self.next_id);
        self.next_id += 1;

        let cancel = CancellationToken::new();
        let cancel_child = cancel.clone();
        let tx = self.app_tx.clone();
        let job_id_clone = job_id.clone();
        let job_type = spec.job_type;
        let metrics_file = spec.metrics_file.clone();

        // Notify TUI that job started
        let _ = tx.send(AppMsg::JobStarted {
            job_id: job_id.clone(),
            job_type: spec.job_type,
        });

        tokio::spawn(async move {
            let result = run_command(spec, &job_id_clone, tx.clone(), cancel_child).await;

            let (success, message) = match result {
                Ok(()) => (true, "Completed successfully".to_string()),
                Err(e) => (false, e.to_string()),
            };

            let _ = tx.send(AppMsg::JobFinished {
                job_id: job_id_clone,
                success,
                message,
            });
        });

        self.jobs.insert(
            job_id.clone(),
            RunningJob {
                id: job_id.clone(),
                job_type,
                cancel,
                metrics_file,
            },
        );

        job_id
    }

    /// Cancel a running job by ID.
    pub fn cancel(&mut self, job_id: &str) {
        if let Some(job) = self.jobs.get(job_id) {
            job.cancel.cancel();
        }
    }

    /// Remove a finished job from tracking.
    pub fn remove(&mut self, job_id: &str) {
        self.jobs.remove(job_id);
    }

    /// Check if a job is being tracked.
    #[allow(dead_code)]
    pub fn has_job(&self, job_id: &str) -> bool {
        self.jobs.contains_key(job_id)
    }

    /// Get all running job IDs.
    #[allow(dead_code)]
    pub fn running_jobs(&self) -> Vec<&str> {
        self.jobs.keys().map(|s| s.as_str()).collect()
    }

    /// Get the metrics file for a job (if any).
    #[allow(dead_code)]
    pub fn metrics_file(&self, job_id: &str) -> Option<&PathBuf> {
        self.jobs.get(job_id).and_then(|j| j.metrics_file.as_ref())
    }
}

/// Find the pmetal binary path.
fn pmetal_binary() -> PathBuf {
    // First try the current executable (works when installed)
    if let Ok(exe) = std::env::current_exe() {
        if exe.file_name().map(|n| n == "pmetal").unwrap_or(false) {
            return exe;
        }
        // If running via cargo, the exe is the CLI binary itself
        return exe;
    }
    // Fallback: assume pmetal is on PATH
    PathBuf::from("pmetal")
}

/// Run a command spec as a child process, streaming output back to the TUI.
async fn run_command(
    spec: CommandSpec,
    job_id: &str,
    tx: mpsc::UnboundedSender<AppMsg>,
    cancel: CancellationToken,
) -> Result<(), anyhow::Error> {
    let binary = pmetal_binary();
    let mut cmd = Command::new(&binary);

    // Build the command arguments
    cmd.args(&spec.args);

    // If this is a training-type job, add metrics logging
    if let Some(ref metrics_path) = spec.metrics_file {
        if matches!(
            spec.job_type,
            JobType::Train | JobType::Distill | JobType::Grpo
        ) {
            cmd.args(["--log-metrics", &metrics_path.display().to_string()]);
        }
    }

    // If this is a training-type job, write a sentinel file
    if let Some(ref output_dir) = spec.output_dir {
        if matches!(
            spec.job_type,
            JobType::Train | JobType::Distill | JobType::Grpo
        ) {
            let running_file = output_dir.join(".running");
            let _ = tokio::fs::create_dir_all(output_dir).await;
            let _ = tokio::fs::write(&running_file, job_id).await;
        }
    }

    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn()?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let job_id_owned = job_id.to_string();
    let is_inference = spec.job_type == JobType::Infer;

    // Stream stdout — for inference jobs, parse output and route as InferenceToken/InferenceDone
    if let Some(stdout) = stdout {
        let tx_out = tx.clone();
        let jid = job_id_owned.clone();
        tokio::spawn(async move {
            if is_inference {
                // For inference: two-phase reading.
                // Phase 1: Read lines until "Generating..." header is found.
                // Phase 2: Read byte-by-byte for true token streaming, until "---" stats footer.
                let mut reader = BufReader::new(stdout);
                let mut line_buf = String::new();

                // Phase 1: Skip header lines
                loop {
                    line_buf.clear();
                    match reader.read_line(&mut line_buf).await {
                        Ok(0) => break, // EOF
                        Ok(_) => {
                            if line_buf.trim_end().starts_with("Generating") {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }

                // Phase 2: Stream bytes as tokens for real-time display.
                // Read byte chunks and forward partial text immediately.
                // Detect "---\n" footer to parse stats and signal completion.
                let mut byte_buf = [0u8; 512];
                let mut accum = String::new();
                let mut got_stats = false;

                'stream: loop {
                    let n = match reader.read(&mut byte_buf).await {
                        Ok(0) => break,
                        Ok(n) => n,
                        Err(_) => break,
                    };

                    let chunk = String::from_utf8_lossy(&byte_buf[..n]);
                    accum.push_str(&chunk);

                    // Check if we have the stats footer in the accumulated text
                    // Format: ...\n---\nGenerated X tokens in Y.YYs (Z.Z tok/s)\n...
                    if let Some(footer_pos) = accum.find("\n---\n") {
                        // Send everything before the footer as content
                        let content = &accum[..footer_pos];
                        if !content.is_empty() {
                            let _ = tx_out.send(AppMsg::InferenceToken {
                                token: content.to_string(),
                            });
                        }
                        // Parse stats from after "---\n"
                        let after_footer = &accum[footer_pos + "\n---\n".len()..];
                        if let Some(stats_line) = after_footer.lines().next() {
                            let (total_tokens, tok_sec) = parse_inference_stats(stats_line);
                            let _ = tx_out.send(AppMsg::InferenceDone {
                                tok_sec,
                                total_tokens,
                            });
                            got_stats = true;
                        }
                        break 'stream;
                    }

                    // Also check "---\n" at the very start (no leading newline)
                    if let Some(after) = accum.strip_prefix("---\n") {
                        if let Some(stats_line) = after.lines().next() {
                            let (total_tokens, tok_sec) = parse_inference_stats(stats_line);
                            let _ = tx_out.send(AppMsg::InferenceDone {
                                tok_sec,
                                total_tokens,
                            });
                            got_stats = true;
                        }
                        break 'stream;
                    }

                    // Send accumulated content but keep the last line (might be partial "---")
                    // Safe to send everything before the last newline
                    if let Some(last_nl) = accum.rfind('\n') {
                        let to_send = &accum[..last_nl];
                        if !to_send.is_empty() {
                            let _ = tx_out.send(AppMsg::InferenceToken {
                                token: to_send.to_string(),
                            });
                        }
                        accum = accum[last_nl + 1..].to_string();
                    }
                }

                // Send any remaining content
                if !accum.is_empty() && !accum.starts_with("---") {
                    let _ = tx_out.send(AppMsg::InferenceToken { token: accum });
                }

                if !got_stats {
                    let _ = tx_out.send(AppMsg::InferenceDone {
                        tok_sec: 0.0,
                        total_tokens: 0,
                    });
                }
            } else {
                // Non-inference: route all output to job log
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if tx_out
                        .send(AppMsg::JobOutput {
                            job_id: jid.clone(),
                            line,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            }
        });
    }

    // Stream stderr
    if let Some(stderr) = stderr {
        let tx_err = tx.clone();
        let jid = job_id_owned.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                // For inference, send errors as InferenceError
                if is_inference && !line.is_empty() {
                    // Skip tracing/debug lines (they start with timestamp or level)
                    let is_tracing = line.contains("INFO")
                        || line.contains("DEBUG")
                        || line.contains("TRACE")
                        || line.contains("WARN");
                    if !is_tracing {
                        let _ = tx_err.send(AppMsg::InferenceError { message: line });
                        continue;
                    }
                }
                if tx_err
                    .send(AppMsg::JobOutput {
                        job_id: jid.clone(),
                        line,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
    }

    // Poll metrics file for training jobs
    if let Some(ref metrics_path) = spec.metrics_file {
        let tx_metrics = tx.clone();
        let jid = job_id_owned.clone();
        let path = metrics_path.clone();
        let cancel_metrics = cancel.clone();
        tokio::spawn(async move {
            poll_metrics_file(&path, &jid, tx_metrics, cancel_metrics).await;
        });
    }

    // Wait for the child to exit, or cancel
    tokio::select! {
        status = child.wait() => {
            let status = status?;
            // Clean up sentinel file
            if let Some(ref output_dir) = spec.output_dir {
                let running_file = output_dir.join(".running");
                let _ = tokio::fs::remove_file(&running_file).await;
            }
            if status.success() {
                Ok(())
            } else {
                Err(anyhow::anyhow!(
                    "Process exited with code {}",
                    status.code().unwrap_or(-1)
                ))
            }
        }
        _ = cancel.cancelled() => {
            let _ = child.kill().await;
            // Clean up sentinel file
            if let Some(ref output_dir) = spec.output_dir {
                let running_file = output_dir.join(".running");
                let _ = tokio::fs::remove_file(&running_file).await;
            }
            Err(anyhow::anyhow!("Cancelled by user"))
        }
    }
}

/// Parse inference stats from a line like "Generated 42 tokens in 1.23s (34.1 tok/s)"
fn parse_inference_stats(line: &str) -> (usize, f64) {
    let mut total_tokens = 0usize;
    let mut tok_sec = 0.0f64;

    // Extract token count: "Generated X tokens"
    if let Some(after_gen) = line.strip_prefix("Generated ") {
        if let Some(tok_str) = after_gen.split_whitespace().next() {
            total_tokens = tok_str.parse().unwrap_or(0);
        }
    }

    // Extract tok/s: "(Z.Z tok/s)"
    if let Some(paren_start) = line.rfind('(') {
        let inside = &line[paren_start + 1..];
        if let Some(tok_s_str) = inside.split_whitespace().next() {
            tok_sec = tok_s_str.parse().unwrap_or(0.0);
        }
    }

    (total_tokens, tok_sec)
}

/// Poll a JSONL metrics file and send updates to the TUI.
async fn poll_metrics_file(
    path: &std::path::Path,
    job_id: &str,
    tx: mpsc::UnboundedSender<AppMsg>,
    cancel: CancellationToken,
) {
    let mut last_pos: u64 = 0;
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));

    loop {
        tokio::select! {
            _ = interval.tick() => {}
            _ = cancel.cancelled() => break,
        }

        let Ok(file) = tokio::fs::File::open(path).await else {
            continue;
        };
        let metadata = match file.metadata().await {
            Ok(m) => m,
            Err(_) => continue,
        };

        let file_len = metadata.len();
        if file_len <= last_pos {
            if file_len < last_pos {
                // File was truncated/rotated
                last_pos = 0;
            }
            continue;
        }

        // Read new lines using sync I/O (small reads, fine on tokio)
        let read_result = tokio::task::spawn_blocking({
            let path = path.to_owned();
            let job_id = job_id.to_string();
            let tx = tx.clone();
            move || {
                use std::io::{BufRead, Seek};
                let Ok(file) = std::fs::File::open(&path) else {
                    return last_pos;
                };
                let mut reader = std::io::BufReader::new(file);
                if reader.seek(std::io::SeekFrom::Start(last_pos)).is_err() {
                    return last_pos;
                }

                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                        let _ = tx.send(AppMsg::JobMetrics {
                            job_id: job_id.clone(),
                            step: json["step"].as_u64().unwrap_or(0) as usize,
                            epoch: json["epoch"].as_u64().unwrap_or(0) as usize,
                            total_epochs: json["total_epochs"].as_u64().unwrap_or(0) as usize,
                            total_steps: json["total_steps"].as_u64().unwrap_or(0) as usize,
                            loss: json["loss"].as_f64().unwrap_or(0.0),
                            lr: json["lr"].as_f64().unwrap_or(0.0),
                            tok_sec: json["tok_sec"].as_f64().unwrap_or(0.0),
                            ane_fwd_ms: json["ane_fwd_ms"].as_f64().unwrap_or(0.0),
                            ane_bwd_ms: json["ane_bwd_ms"].as_f64().unwrap_or(0.0),
                            rmsnorm_ms: json["rmsnorm_ms"].as_f64().unwrap_or(0.0),
                            cblas_ms: json["cblas_ms"].as_f64().unwrap_or(0.0),
                            adam_ms: json["adam_ms"].as_f64().unwrap_or(0.0),
                            total_ms: json["total_ms"].as_f64().unwrap_or(0.0),
                        });
                    }
                    line.clear();
                }
                reader.stream_position().unwrap_or(last_pos)
            }
        })
        .await;

        if let Ok(new_pos) = read_result {
            last_pos = new_pos;
        }
    }
}
