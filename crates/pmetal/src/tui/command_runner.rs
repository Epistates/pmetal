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

/// Manages background pmetal child processes.
pub struct CommandRunner {
    app_tx: mpsc::Sender<AppMsg>,
    /// Cancellation token of each running job, by job id.
    jobs: HashMap<String, CancellationToken>,
    next_id: u32,
}

impl CommandRunner {
    pub fn new(app_tx: mpsc::Sender<AppMsg>) -> Self {
        Self {
            app_tx,
            jobs: HashMap::new(),
            next_id: 0,
        }
    }

    /// Get a clone of the app message sender (for background tasks).
    pub fn app_tx(&self) -> mpsc::Sender<AppMsg> {
        self.app_tx.clone()
    }

    /// Spawn a background job from a command spec. Returns the job ID.
    pub fn spawn(&mut self, spec: CommandSpec) -> String {
        let job_id = format!("job-{}", self.next_id);
        self.next_id += 1;

        let cancel = CancellationToken::new();
        let cancel_child = cancel.clone();
        let tx = self.app_tx.clone();
        let job_id_clone = job_id.clone();

        // Notify TUI that job started (sync context — use try_send, drop on full)
        let _ = tx.try_send(AppMsg::JobStarted {
            job_id: job_id.clone(),
            job_type: spec.job_type,
        });

        tokio::spawn(async move {
            let result = run_command(spec, &job_id_clone, tx.clone(), cancel_child).await;

            let (success, message) = match result {
                Ok(()) => (true, "Completed successfully".to_string()),
                Err(e) => (false, e.to_string()),
            };

            let _ = tx
                .send(AppMsg::JobFinished {
                    job_id: job_id_clone,
                    success,
                    message,
                })
                .await;
        });

        self.jobs.insert(job_id.clone(), cancel);

        job_id
    }

    /// Cancel a running job by ID.
    pub fn cancel(&mut self, job_id: &str) {
        if let Some(cancel) = self.jobs.get(job_id) {
            cancel.cancel();
        }
    }

    /// Remove a finished job from tracking, cancelling its token so that any
    /// background tasks (e.g. the metrics-file poller) stop promptly.
    pub fn remove(&mut self, job_id: &str) {
        if let Some(cancel) = self.jobs.remove(job_id) {
            cancel.cancel();
        }
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

struct CancelOnToken {
    token: CancellationToken,
}

impl pmetal_core::TrainingCallback for CancelOnToken {
    fn should_stop(&self) -> bool {
        self.token.is_cancelled()
    }
}

/// In-process training metrics callback.
///
/// When a training job runs in the same process as the TUI (via
/// `run_direct_command`), the trainer's `on_step_end_with_metrics` hook is
/// called directly on each step — we forward the `StepMetrics` to the UI
/// event loop through `AppMsg::JobMetrics`, eliminating the 500ms JSONL
/// polling latency used by the subprocess fallback.
struct ChannelMetricsCallback {
    tx: mpsc::Sender<AppMsg>,
    job_id: String,
}

impl pmetal_core::TrainingCallback for ChannelMetricsCallback {
    fn on_step_end_with_metrics(&mut self, metrics: &pmetal_core::StepMetrics) {
        // try_send so a slow UI thread can't back-pressure the trainer —
        // the oldest metric will be dropped rather than stalling training.
        let _ = self.tx.try_send(AppMsg::JobMetrics {
            job_id: self.job_id.clone(),
            step: metrics.step,
            epoch: metrics.epoch,
            total_epochs: metrics.total_epochs,
            total_steps: metrics.total_steps,
            loss: metrics.loss,
            lr: metrics.lr,
            tok_sec: metrics.tok_sec,
            ane_fwd_ms: metrics.ane_fwd_ms,
            ane_bwd_ms: metrics.ane_bwd_ms,
            rmsnorm_ms: metrics.rmsnorm_ms,
            cblas_ms: metrics.cblas_ms,
            adam_ms: metrics.adam_ms,
            total_ms: metrics.total_ms,
        });
    }
}

/// `JobEventSink` that fans events into the TUI's `AppMsg` mpsc channel.
///
/// Phase 2 wiring — runs alongside [`ChannelMetricsCallback`] until Phase 4
/// collapses the legacy `AppMsg::Job*` variants. `try_send` keeps the trainer
/// from back-pressuring on a slow UI thread.
struct MpscAppMsgSink {
    tx: mpsc::Sender<AppMsg>,
    cancel: CancellationToken,
}

impl pmetal_core::JobEventSink for MpscAppMsgSink {
    fn emit(&self, event: pmetal_core::JobEvent) {
        let _ = self.tx.try_send(AppMsg::Job(event));
    }
    fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

/// Run a command spec as a child process, streaming output back to the TUI.
#[allow(unsafe_code)]
async fn run_command(
    spec: CommandSpec,
    job_id: &str,
    tx: mpsc::Sender<AppMsg>,
    cancel: CancellationToken,
) -> Result<(), anyhow::Error> {
    // If this is a training-type job, write a sentinel file
    if let Some(ref output_dir) = spec.output_dir {
        if matches!(
            spec.job_type,
            JobType::Train | JobType::Distill | JobType::Grpo | JobType::Pretrain
        ) {
            let running_file = output_dir.join(".running");
            let _ = tokio::fs::create_dir_all(output_dir).await;
            let _ = tokio::fs::write(&running_file, job_id).await;
        }
    }

    // Log the full command to the job output for debugging
    let cmd_display = format!("pmetal {}", spec.args.join(" "));
    let _ = tx
        .send(AppMsg::JobOutput {
            job_id: job_id.to_string(),
            line: format!("$ {cmd_display}"),
        })
        .await;

    // Emit phase status for training-type jobs so the dashboard shows setup progress.
    if matches!(
        spec.job_type,
        JobType::Train | JobType::Distill | JobType::Grpo | JobType::Pretrain
    ) {
        let _ = tx
            .send(AppMsg::JobPhase {
                job_id: job_id.to_string(),
                phase: "Loading model and preparing dataset...".to_string(),
            })
            .await;
    }

    // Direct in-process path: for train/distill/grpo the callbacks stream
    // metrics through ChannelMetricsCallback, so no JSONL polling is needed.
    if let Some(result) = run_direct_command(&spec, job_id, tx.clone(), cancel.clone()).await {
        cleanup_running_file(&spec).await;
        return result;
    }

    // Subprocess fallback path: the child writes a JSONL metrics file and we
    // tail it from here. The 500ms polling latency is the price for being
    // out-of-process; direct-command jobs above don't pay it.
    if let Some(ref metrics_path) = spec.metrics_file {
        let tx_metrics = tx.clone();
        let jid = job_id.to_string();
        let path = metrics_path.clone();
        let cancel_metrics = cancel.clone();
        tokio::spawn(async move {
            poll_metrics_file(&path, &jid, tx_metrics, cancel_metrics).await;
        });
    }

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

    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    // Place child in its own process group so cancellation kills the entire
    // process tree (child + any grandchildren it spawns), not just the direct child.
    #[cfg(unix)]
    cmd.process_group(0);

    let mut child = cmd.spawn()?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let job_id_owned = job_id.to_string();
    let is_inference = spec.job_type == JobType::Infer;

    // Shared ring buffer to capture last stderr lines for error reporting
    let last_stderr = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    const MAX_STDERR_LINES: usize = 10;

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
                            let _ = tx_out
                                .send(AppMsg::InferenceToken {
                                    token: content.to_string(),
                                })
                                .await;
                        }
                        // Parse stats from after "---\n"
                        let after_footer = &accum[footer_pos + "\n---\n".len()..];
                        if let Some(stats_line) = after_footer.lines().next() {
                            let (total_tokens, tok_sec) = parse_inference_stats(stats_line);
                            let _ = tx_out
                                .send(AppMsg::InferenceDone {
                                    tok_sec,
                                    total_tokens,
                                })
                                .await;
                            got_stats = true;
                        }
                        break 'stream;
                    }

                    // Also check "---\n" at the very start (no leading newline)
                    if let Some(after) = accum.strip_prefix("---\n") {
                        if let Some(stats_line) = after.lines().next() {
                            let (total_tokens, tok_sec) = parse_inference_stats(stats_line);
                            let _ = tx_out
                                .send(AppMsg::InferenceDone {
                                    tok_sec,
                                    total_tokens,
                                })
                                .await;
                            got_stats = true;
                        }
                        break 'stream;
                    }

                    // Send accumulated content but keep the last line (might be partial "---")
                    // Safe to send everything before the last newline
                    if let Some(last_nl) = accum.rfind('\n') {
                        let to_send = &accum[..last_nl];
                        if !to_send.is_empty() {
                            let _ = tx_out
                                .send(AppMsg::InferenceToken {
                                    token: to_send.to_string(),
                                })
                                .await;
                        }
                        accum = accum[last_nl + 1..].to_string();
                    }
                }

                // Send any remaining content
                if !accum.is_empty() && !accum.starts_with("---") {
                    let _ = tx_out.send(AppMsg::InferenceToken { token: accum }).await;
                }

                if !got_stats {
                    let _ = tx_out
                        .send(AppMsg::InferenceDone {
                            tok_sec: 0.0,
                            total_tokens: 0,
                        })
                        .await;
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
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
        });
    }

    // Stream stderr
    let stderr_capture = last_stderr.clone();
    if let Some(stderr) = stderr {
        let tx_err = tx.clone();
        let jid = job_id_owned.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                // Capture recent stderr lines for error reporting.
                // Use `unwrap_or_else` so a poisoned mutex (from a panicking
                // sibling task) never kills the metrics-streaming task.
                {
                    let mut buf = stderr_capture.lock().unwrap_or_else(|e| e.into_inner());
                    buf.push(line.clone());
                    if buf.len() > MAX_STDERR_LINES {
                        buf.remove(0);
                    }
                }

                // For inference, send errors as InferenceError
                if is_inference && !line.is_empty() {
                    // Skip tracing/debug lines (they start with timestamp or level)
                    let is_tracing = line.contains("INFO")
                        || line.contains("DEBUG")
                        || line.contains("TRACE")
                        || line.contains("WARN");
                    if !is_tracing {
                        let _ = tx_err.send(AppMsg::InferenceError { message: line }).await;
                        continue;
                    }
                }
                if tx_err
                    .send(AppMsg::JobOutput {
                        job_id: jid.clone(),
                        line,
                    })
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });
    }

    // Wait for the child to exit, or cancel
    tokio::select! {
        status = child.wait() => {
            let status = status?;
            cleanup_running_file(&spec).await;
            if status.success() {
                Ok(())
            } else {
                // Give stderr reader a moment to flush remaining lines
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;

                // Include last stderr lines in error for better diagnostics
                let code = status.code().unwrap_or(-1);
                // Same poison guard — a panic in the stderr-capture task must
                // not prevent the error from being propagated to the TUI.
                let stderr_lines = last_stderr.lock().unwrap_or_else(|e| e.into_inner());
                let context: Vec<&str> = stderr_lines
                    .iter()
                    .rev()
                    .filter(|l| {
                        let l = l.trim();
                        !l.is_empty() && !l.contains("INFO") && !l.contains("DEBUG") && !l.contains("TRACE")
                    })
                    .take(5)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .map(|s| s.as_str())
                    .collect();

                if context.is_empty() {
                    Err(anyhow::anyhow!("Process exited with code {code}"))
                } else {
                    Err(anyhow::anyhow!(
                        "Process exited with code {code}:\n{}",
                        context.join("\n")
                    ))
                }
            }
        }
        _ = cancel.cancelled() => {
            // On Unix, kill the entire process group (child + all grandchildren).
            // Fall back to killing just the child process on other platforms.
            #[cfg(unix)]
            {
                if let Some(pid) = child.id() {
                    // SAFETY: kill(2) is async-signal-safe; pid comes from a live child.
                    unsafe { libc::killpg(pid as i32, libc::SIGTERM); }
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    unsafe { libc::killpg(pid as i32, libc::SIGKILL); }
                }
            }
            let _ = child.kill().await;
            cleanup_running_file(&spec).await;
            Err(anyhow::anyhow!("Cancelled by user"))
        }
    }
}

async fn cleanup_running_file(spec: &CommandSpec) {
    if let Some(ref output_dir) = spec.output_dir {
        let running_file = output_dir.join(".running");
        let _ = tokio::fs::remove_file(&running_file).await;
    }
}

async fn run_direct_command(
    spec: &CommandSpec,
    job_id: &str,
    tx: mpsc::Sender<AppMsg>,
    cancel: CancellationToken,
) -> Option<Result<(), anyhow::Error>> {
    let subcommand = spec.args.first()?.as_str();

    match subcommand {
        "train" => Some(run_training_direct(spec, job_id, tx, cancel).await),
        "distill" => Some(run_distillation_direct(spec, job_id, tx, cancel).await),
        "grpo" => Some(run_grpo_direct(spec, job_id, tx, cancel).await),
        _ => None,
    }
}

/// Build the callback vec used by every direct-path training entry point:
/// cancellation + legacy metrics + new `JobEvent` sink. Centralised so the
/// three trainers (train / distill / grpo) stay in lockstep when new
/// callbacks land.
///
/// Phase 2: emits both the legacy `AppMsg::Job*` variants (via
/// `ChannelMetricsCallback`) and the canonical `AppMsg::Job(JobEvent)` (via
/// `MpscAppMsgSink` wrapped in `TrainingCallbackToSink`). Phase 4 collapses
/// to the canonical path only.
fn direct_training_callbacks(
    job_id: &str,
    tx: mpsc::Sender<AppMsg>,
    cancel: CancellationToken,
) -> Vec<Box<dyn pmetal_core::TrainingCallback>> {
    let sink = MpscAppMsgSink {
        tx: tx.clone(),
        cancel: cancel.clone(),
    };
    vec![
        Box::new(CancelOnToken { token: cancel }),
        Box::new(ChannelMetricsCallback {
            tx,
            job_id: job_id.to_string(),
        }),
        Box::new(pmetal_core::TrainingCallbackToSink::new(job_id, sink)),
    ]
}

async fn run_training_direct(
    spec: &CommandSpec,
    job_id: &str,
    tx: mpsc::Sender<AppMsg>,
    cancel: CancellationToken,
) -> Result<(), anyhow::Error> {
    use clap::Parser;
    use pmetal_core::jobs::TrainSpec;
    use pmetal_trainer::orchestrator;

    // The command line the subprocess runner would spawn, read by the same
    // parser, so a flag cannot be understood by one runner and dropped by the
    // other.
    let argv = ["pmetal", "train"]
        .into_iter()
        .map(str::to_string)
        .chain(spec.args.iter().cloned());
    let crate::cli::Commands::Train(args) = crate::cli::Cli::try_parse_from(argv)?.command else {
        anyhow::bail!("the train command line parsed as another command");
    };
    let mut train_spec = TrainSpec::from(args);
    train_spec
        .normalize()
        .map_err(|errors| crate::commands::invalid_spec::<TrainSpec>(&errors))?;
    let mut job_config =
        orchestrator::TrainingJobConfig::from_spec(&train_spec).map_err(anyhow::Error::msg)?;
    // In-process: metrics stream through ChannelMetricsCallback, not a JSONL file.
    job_config.log_metrics = None;
    job_config.emit_console_output = false;

    let callbacks = direct_training_callbacks(job_id, tx, cancel);
    orchestrator::run_training(job_config, None, callbacks).await?;
    Ok(())
}

async fn run_distillation_direct(
    spec: &CommandSpec,
    job_id: &str,
    tx: mpsc::Sender<AppMsg>,
    cancel: CancellationToken,
) -> Result<(), anyhow::Error> {
    use pmetal_core::jobs::DistillSpec;

    let crate::cli::Commands::Distill(args) = parse_command("distill", spec)? else {
        anyhow::bail!("the distill command line parsed as another command");
    };
    let mut distill_spec = DistillSpec::from(args);
    // In-process: metrics stream through the callbacks, and the dashboard's
    // own metrics file is the one the run writes.
    distill_spec.log_metrics = spec.metrics_file.as_ref().map(|p| p.display().to_string());
    distill_spec
        .normalize()
        .map_err(|errors| crate::commands::invalid_spec::<DistillSpec>(&errors))?;
    crate::commands::distill::run_distillation(
        distill_spec,
        crate::commands::RunContext {
            emit_console_output: false,
            callbacks: direct_training_callbacks(job_id, tx, cancel),
        },
    )
    .await
}

/// The command a TUI job's arguments spell, read by the parser the
/// subprocess runner's command line goes through.
fn parse_command(
    subcommand: &str,
    spec: &CommandSpec,
) -> Result<crate::cli::Commands, anyhow::Error> {
    use clap::Parser;
    let argv = ["pmetal", subcommand]
        .into_iter()
        .map(str::to_string)
        .chain(spec.args.iter().cloned());
    Ok(crate::cli::Cli::try_parse_from(argv)?.command)
}

async fn run_grpo_direct(
    spec: &CommandSpec,
    job_id: &str,
    tx: mpsc::Sender<AppMsg>,
    cancel: CancellationToken,
) -> Result<(), anyhow::Error> {
    use pmetal_core::jobs::GrpoSpec;

    let crate::cli::Commands::Grpo(args) = parse_command("grpo", spec)? else {
        anyhow::bail!("the grpo command line parsed as another command");
    };
    let mut grpo_spec = GrpoSpec::from(args);
    grpo_spec.log_metrics = spec.metrics_file.as_ref().map(|p| p.display().to_string());
    grpo_spec
        .normalize()
        .map_err(|errors| crate::commands::invalid_spec::<GrpoSpec>(&errors))?;
    crate::commands::grpo::run_grpo(
        grpo_spec,
        crate::commands::RunContext {
            emit_console_output: false,
            callbacks: direct_training_callbacks(job_id, tx, cancel),
        },
    )
    .await
}

fn optional_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].clone())
}

fn required_arg(args: &[String], flag: &str) -> Result<String, anyhow::Error> {
    optional_arg(args, flag).ok_or_else(|| anyhow::anyhow!("Missing required argument: {flag}"))
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

/// Parse an optional typed argument — returns `None` when the flag is absent.
fn parse_opt_arg<T>(args: &[String], flag: &str) -> Option<T>
where
    T: std::str::FromStr,
{
    optional_arg(args, flag).and_then(|v| v.parse::<T>().ok())
}

fn parse_arg<T>(args: &[String], flag: &str, default: T) -> Result<T, anyhow::Error>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match optional_arg(args, flag) {
        Some(value) => value
            .parse::<T>()
            .map_err(|e| anyhow::anyhow!("Invalid value for {flag}: {e}")),
        None => Ok(default),
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
    tx: mpsc::Sender<AppMsg>,
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
                // Collect metrics lines then send outside the sync read loop
                let mut metrics_to_send = Vec::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                        metrics_to_send.push(AppMsg::JobMetrics {
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
                for msg in metrics_to_send {
                    let _ = tx.try_send(msg);
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
