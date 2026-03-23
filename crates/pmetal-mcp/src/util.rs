use std::path::Path;
use std::process::Stdio;

use tokio::process::Command;
use turbomcp::prelude::*;

/// Query total system memory via sysctl (macOS).
pub fn get_system_memory() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let mem_str = String::from_utf8(output.stdout).ok()?;
    mem_str.trim().parse().ok()
}

/// Resolve the path to the `pmetal` binary.
///
/// Search order:
/// 1. `PMETAL_BIN` environment variable (explicit override)
/// 2. Co-located binary (same directory as this executable)
/// 3. `pmetal` on PATH (fallback)
pub fn resolve_pmetal_binary() -> String {
    if let Ok(bin) = std::env::var("PMETAL_BIN") {
        return bin;
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("pmetal");
            if candidate.exists() {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }

    "pmetal".to_string()
}

/// Create a `tokio::process::Command` pre-configured with the pmetal binary
/// and piped stdout/stderr.
pub fn pmetal_command() -> Command {
    let bin = resolve_pmetal_binary();
    let mut cmd = Command::new(bin);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

/// Run a pmetal subcommand synchronously and return its stdout on success.
pub async fn run_pmetal_blocking(args: &[&str]) -> McpResult<String> {
    let mut cmd = pmetal_command();
    for arg in args {
        cmd.arg(arg);
    }

    let output = cmd
        .output()
        .await
        .map_err(|e| McpError::internal(format!("failed to run pmetal: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let msg = if stderr.is_empty() {
            stdout.into_owned()
        } else {
            stderr.into_owned()
        };
        return Err(McpError::internal(
            format!("pmetal exited with {}: {msg}", output.status),
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Build device info JSON by calling MetalContext directly.
pub fn build_device_info_json() -> McpResult<serde_json::Value> {
    let ctx = pmetal_metal::context::MetalContext::global().map_err(|e| {
        McpError::internal(format!("Metal not available: {e}"))
    })?;

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
    let ctx = pmetal_metal::context::MetalContext::global().map_err(|e| {
        McpError::internal(format!("Metal not available: {e}"))
    })?;
    let props = ctx.properties();

    const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;
    Ok(pmetal_hub::DeviceSpec {
        memory_gb: props.recommended_working_set_size as f64 / BYTES_PER_GB,
        bandwidth_gbps: props.memory_bandwidth_gbps,
        unified_memory: props.has_unified_memory,
    })
}

/// Conditionally push an `--option value` pair onto an args vec.
pub fn push_opt(args: &mut Vec<String>, flag: &str, value: &Option<impl ToString>) {
    if let Some(v) = value {
        args.push(flag.to_string());
        args.push(v.to_string());
    }
}

/// Conditionally push a `--flag` (boolean) onto an args vec.
pub fn push_bool_flag(args: &mut Vec<String>, flag: &str, value: &Option<bool>) {
    if let Some(true) = value {
        args.push(flag.to_string());
    }
}

/// Find the HuggingFace cache directory.
pub fn hf_cache_dir() -> std::path::PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return Path::new(&cache).join("hub");
    }
    if let Ok(cache) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return cache.into();
    }
    dirs_fallback().join(".cache").join("huggingface").join("hub")
}

fn dirs_fallback() -> std::path::PathBuf {
    std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
}
