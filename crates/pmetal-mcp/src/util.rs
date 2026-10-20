use std::borrow::Cow;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use turbomcp::prelude::*;

/// Most standard output a blocking command may produce before the rest is
/// dropped.
const MAX_STDOUT_BYTES: usize = 8 * 1024 * 1024;
/// How much of the end of standard error is kept to explain a failure.
const MAX_STDERR_BYTES: usize = 64 * 1024;
/// How often a blocking command is checked for cancellation, and progress is
/// reported.
const POLL: Duration = Duration::from_millis(250);
const HEARTBEAT: Duration = Duration::from_secs(5);
/// How long a blocking command runs before it is stopped, unless
/// `PMETAL_MCP_TIMEOUT_SECS` says otherwise.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60 * 60);

/// Query total system memory via sysctl (macOS).
pub fn get_system_memory() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let mem_str = String::from_utf8(output.stdout).ok()?;
    mem_str.trim().parse().ok()
}

pub use pmetal_supervisor::{kill_group, redact_secrets, resolve_pmetal_binary};

/// A `Command` for the `pmetal` binary `binary` with stdin closed, output piped, in its
/// own process group, and killed if the handle is dropped.
pub fn pmetal_command(binary: &std::path::Path) -> Command {
    let mut cmd = Command::new(binary);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);
    cmd
}

/// How long a blocking command may run.
fn timeout() -> Duration {
    std::env::var("PMETAL_MCP_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .map_or(DEFAULT_TIMEOUT, Duration::from_secs)
}

/// Run a pmetal subcommand to completion and return its standard output.
///
/// `argv` is everything after the binary name. While it runs, a client that
/// asked for progress gets a heartbeat (so it does not give up on a long
/// command), a cancelled request stops the command, and a command that runs
/// past the timeout is stopped. Output is bounded: standard output keeps its
/// first 8 MiB, and a failure reports the end of standard error.
pub async fn run_pmetal(
    binary: &std::path::Path,
    argv: &[String],
    ctx: &RequestContext,
) -> McpResult<String> {
    run_command(pmetal_command(binary), argv, ctx, timeout(), HEARTBEAT).await
}

/// [`run_pmetal`] with the command, the time limit and the progress interval given.
pub async fn run_command(
    mut cmd: Command,
    argv: &[String],
    ctx: &RequestContext,
    limit: Duration,
    heartbeat: Duration,
) -> McpResult<String> {
    cmd.args(argv);
    let mut child = cmd
        .spawn()
        .map_err(|e| McpError::internal(format!("failed to run pmetal: {e}")))?;
    let pid = child.id();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_task = tokio::spawn(capture(stdout, MAX_STDOUT_BYTES, Keep::Head));
    let err_task = tokio::spawn(capture(stderr, MAX_STDERR_BYTES, Keep::Tail));

    let started = tokio::time::Instant::now();
    let mut last_heartbeat = started;
    let command = argv.first().map_or("pmetal", String::as_str).to_string();
    let outcome = loop {
        tokio::select! {
            status = child.wait() => break Outcome::Exited(status),
            () = tokio::time::sleep(POLL) => {}
        }
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
    };

    if !matches!(outcome, Outcome::Exited(_)) {
        if let Some(pid) = pid {
            kill_group(pid);
        }
        let _ = child.wait().await;
    }
    let (stdout, dropped) = out_task.await.unwrap_or_default();
    let (stderr, _) = err_task.await.unwrap_or_default();

    match outcome {
        Outcome::Cancelled => Err(McpError::cancelled("the request was cancelled")),
        Outcome::TimedOut => Err(McpError::timeout(format!(
            "pmetal {command} did not finish in {}s and was stopped. \
             Raise PMETAL_MCP_TIMEOUT_SECS, or run it as a job",
            limit.as_secs()
        ))),
        Outcome::Exited(Err(e)) => Err(McpError::internal(format!("failed to run pmetal: {e}"))),
        Outcome::Exited(Ok(status)) if !status.success() => {
            let stderr = String::from_utf8_lossy(&stderr);
            let stdout = String::from_utf8_lossy(&stdout);
            let detail = if stderr.trim().is_empty() {
                tail_chars(&stdout, 4000)
            } else {
                tail_chars(&stderr, 4000)
            };
            Err(McpError::internal(format!(
                "pmetal {command} exited with {status}: {}",
                redact_secrets(detail.trim())
            )))
        }
        Outcome::Exited(Ok(_)) => {
            let mut text = redact_secrets(&String::from_utf8_lossy(&stdout)).into_owned();
            if dropped > 0 {
                text.push_str(&format!("\n… [{dropped} more bytes of output dropped]"));
            }
            Ok(text)
        }
    }
}

enum Outcome {
    Exited(std::io::Result<std::process::ExitStatus>),
    Cancelled,
    TimedOut,
}

enum Keep {
    Head,
    Tail,
}

/// Read `reader` to the end keeping at most `max` bytes (its start, or its
/// end), and say how many were dropped.
async fn capture<R: AsyncRead + Unpin>(
    reader: Option<R>,
    max: usize,
    keep: Keep,
) -> (Vec<u8>, u64) {
    let Some(mut reader) = reader else {
        return (Vec::new(), 0);
    };
    let mut kept: Vec<u8> = Vec::new();
    let mut dropped = 0u64;
    let mut chunk = vec![0u8; 16 * 1024];
    loop {
        let n = match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        match keep {
            Keep::Head => {
                let room = max.saturating_sub(kept.len());
                let take = n.min(room);
                kept.extend_from_slice(&chunk[..take]);
                dropped += (n - take) as u64;
            }
            Keep::Tail => {
                kept.extend_from_slice(&chunk[..n]);
                if kept.len() > max {
                    let excess = kept.len() - max;
                    kept.drain(..excess);
                    dropped += excess as u64;
                }
            }
        }
    }
    (kept, dropped)
}

/// The last `max` characters of `text`.
fn tail_chars(text: &str, max: usize) -> Cow<'_, str> {
    let count = text.chars().count();
    if count <= max {
        return Cow::Borrowed(text);
    }
    let skip = count - max;
    Cow::Owned(format!("…{}", text.chars().skip(skip).collect::<String>()))
}

/// Build device info JSON by calling MetalContext directly.
pub fn build_device_info_json() -> McpResult<serde_json::Value> {
    let ctx = pmetal_metal::context::MetalContext::global()
        .map_err(|e| McpError::internal(format!("Metal not available: {e}")))?;

    let props = ctx.properties();
    let system_memory = get_system_memory();

    const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

    Ok(serde_json::json!({
        "device_name": props.name,
        "gpu_family": format!("{:?}", props.gpu_family),
        "architecture_gen": props.architecture_gen,
        "has_nax": props.has_nax,
        "gpu_cores": props.gpu_core_count,
        "ane_cores": props.ane_core_count,
        "memory_total_gb": system_memory.map(|b| b as f64 / BYTES_PER_GB),
        "recommended_working_set_gb": props.recommended_working_set_size as f64 / BYTES_PER_GB,
        "memory_bandwidth_gbps": props.memory_bandwidth_gbps,
        "memory_bandwidth_source": format!("{:?}", props.memory_bandwidth_source),
        "has_unified_memory": props.has_unified_memory,
        "metal_available": true,
    }))
}

/// Build a DeviceSpec for fit estimation.
pub fn build_device_spec() -> McpResult<pmetal_hub::DeviceSpec> {
    let ctx = pmetal_metal::context::MetalContext::global()
        .map_err(|e| McpError::internal(format!("Metal not available: {e}")))?;
    let props = ctx.properties();

    Ok(pmetal_hub::DeviceSpec::from_device(
        props.recommended_working_set_size,
        props.memory_bandwidth_gbps,
        props.has_unified_memory,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn capture_keeps_the_head_or_the_tail() {
        let data: Vec<u8> = (0..100u8).collect();
        let (head, dropped) = capture(Some(&data[..]), 10, Keep::Head).await;
        assert_eq!(head, (0..10u8).collect::<Vec<_>>());
        assert_eq!(dropped, 90);
        let (tail, dropped) = capture(Some(&data[..]), 10, Keep::Tail).await;
        assert_eq!(tail, (90..100u8).collect::<Vec<_>>());
        assert_eq!(dropped, 90);
    }
}
