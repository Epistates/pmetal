//! Memory management utilities for Apple Silicon.
//!
//! Re-exports MLX Metal memory management via pmetal-bridge and adds
//! pmetal-specific helpers (RSS tracking, system memory queries, estimation
//! functions, and training-oriented diagnostics).

#![allow(unsafe_code)]

#[cfg(target_os = "macos")]
use std::sync::OnceLock;

use pmetal_core::MemoryStats;

// ---------------------------------------------------------------------------
// Re-export MLX Metal memory management via pmetal-bridge
// ---------------------------------------------------------------------------

pub use pmetal_bridge::inline_array::{
    clear_cache, get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory,
    set_wired_limit,
};

/// Get the current memory limit (returns maximum recommended working set size).
pub fn get_memory_limit() -> usize {
    pmetal_bridge::inline_array::get_max_recommended_size()
}

/// Set the memory limit. Maps to set_wired_limit in the bridge.
pub fn set_memory_limit(limit: usize) -> usize {
    pmetal_bridge::inline_array::set_wired_limit(limit)
}

#[cfg(target_os = "macos")]
static SYSTEM_MEMORY_BYTES: OnceLock<Option<u64>> = OnceLock::new();

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Log current MLX memory state at INFO level.
pub fn log_memory_stats() {
    let active = get_active_memory();
    let cache = get_cache_memory();
    let peak = get_peak_memory();
    let limit = get_memory_limit();
    tracing::info!(
        "MLX memory: active={}, cache={}, peak={}, limit={}",
        format_bytes(active as u64),
        format_bytes(cache as u64),
        format_bytes(peak as u64),
        format_bytes(limit as u64),
    );
}

// ---------------------------------------------------------------------------
// System memory queries (RSS, total RAM)
// ---------------------------------------------------------------------------

#[cfg(unix)]
#[allow(non_camel_case_types)]
mod sys {
    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct timeval {
        pub tv_sec: i64,
        pub tv_usec: i32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct rusage {
        pub ru_utime: timeval,
        pub ru_stime: timeval,
        pub ru_maxrss: i64,
        pub ru_ixrss: i64,
        pub ru_idrss: i64,
        pub ru_isrss: i64,
        pub ru_minflt: i64,
        pub ru_majflt: i64,
        pub ru_nswap: i64,
        pub ru_inblock: i64,
        pub ru_oublock: i64,
        pub ru_msgsnd: i64,
        pub ru_msgrcv: i64,
        pub ru_nsignals: i64,
        pub ru_nvcsw: i64,
        pub ru_nivcsw: i64,
    }

    pub const RUSAGE_SELF: i32 = 0;

    unsafe extern "C" {
        pub fn getrusage(who: i32, usage: *mut rusage) -> i32;
    }
}

/// Get the current Resident Set Size (RSS) in bytes.
#[cfg(unix)]
fn get_rss() -> u64 {
    // SAFETY: getrusage is a POSIX system call; zeroed rusage is valid.
    unsafe {
        let mut usage: sys::rusage = std::mem::zeroed();
        if sys::getrusage(sys::RUSAGE_SELF, &mut usage) == 0 {
            #[cfg(target_os = "macos")]
            return usage.ru_maxrss as u64;

            #[cfg(not(target_os = "macos"))]
            return (usage.ru_maxrss * 1024) as u64;
        }
    }
    0
}

#[cfg(not(unix))]
fn get_rss() -> u64 {
    0
}

/// Get current memory statistics.
///
/// Combines MLX allocator stats with process-level RSS.
pub fn get_memory_stats() -> MemoryStats {
    let total_bytes = get_system_memory().unwrap_or(0);
    let active = get_active_memory() as u64;
    let peak = get_peak_memory() as u64;
    let rss = get_rss();

    let used_bytes = if active > 0 { active } else { rss };

    MemoryStats {
        total_bytes,
        used_bytes,
        peak_bytes: peak.max(used_bytes),
    }
}

/// Get total system memory in bytes.
pub fn get_system_memory() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        *SYSTEM_MEMORY_BYTES.get_or_init(query_system_memory)
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn query_system_memory() -> Option<u64> {
    use std::process::Command;

    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let mem_str = String::from_utf8(output.stdout).ok()?;
    mem_str.trim().parse().ok()
}

/// Get GPU memory limit (recommended working set size).
///
/// Falls back to the device's own recommendation rather than a fraction of
/// installed RAM: that ratio is not a constant, and guessing it is what made
/// `set_wired_limit` abort on a 16 GB Mac mini.
pub fn get_gpu_memory_limit() -> Option<u64> {
    let limit = get_memory_limit();
    if limit > 0 {
        return Some(limit as u64);
    }
    match pmetal_bridge::inline_array::get_max_recommended_size() {
        0 => None,
        recommended => Some(recommended as u64),
    }
}

// ---------------------------------------------------------------------------
// Memory-efficient context
// ---------------------------------------------------------------------------

/// RAII guard that clears the MLX buffer cache on drop.
#[derive(Debug)]
pub struct MemoryEfficientContext {
    _private: (),
}

impl MemoryEfficientContext {
    /// Create a new memory-efficient context.
    /// Resets peak memory tracking for profiling.
    pub fn new() -> Self {
        reset_peak_memory();
        Self { _private: () }
    }
}

impl Default for MemoryEfficientContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MemoryEfficientContext {
    fn drop(&mut self) {
        clear_cache();
    }
}

// ---------------------------------------------------------------------------
// Estimation helpers
// ---------------------------------------------------------------------------

/// Estimate memory required for a tensor.
pub fn estimate_tensor_memory(shape: &[usize], dtype_size: usize) -> u64 {
    let num_elements: usize = shape.iter().product();
    (num_elements * dtype_size) as u64
}

/// Estimate memory required for a model with given parameters.
pub fn estimate_model_memory(
    num_params: u64,
    dtype_size: usize,
    include_gradients: bool,
    include_optimizer: bool,
) -> u64 {
    let base = num_params * dtype_size as u64;
    let gradient_factor = if include_gradients { 1 } else { 0 };
    let optimizer_factor = if include_optimizer { 2 } else { 0 };
    base * (1 + gradient_factor + optimizer_factor)
}

/// Estimate memory required for training a model.
pub fn estimate_training_memory(
    num_params: u64,
    dtype_size: usize,
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    num_layers: usize,
) -> u64 {
    let model_memory = estimate_model_memory(num_params, dtype_size, true, true);
    let activation_per_layer = (batch_size * seq_len * hidden_size * dtype_size) as u64 * 4;
    let activation_memory = activation_per_layer * num_layers as u64;
    model_memory + activation_memory
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Format bytes as human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Print memory statistics to stdout.
pub fn print_memory_stats() {
    let stats = get_memory_stats();
    let cache = get_cache_memory();

    println!("Memory Statistics:");
    println!("  System Total:    {}", format_bytes(stats.total_bytes));
    println!("  MLX Active:      {}", format_bytes(stats.used_bytes));
    println!("  MLX Cache:       {}", format_bytes(cache as u64));
    println!("  MLX Peak:        {}", format_bytes(stats.peak_bytes));
    println!(
        "  MLX Limit:       {}",
        format_bytes(get_memory_limit() as u64)
    );
}

/// Check if memory usage is approaching limit.
pub fn memory_warning(estimated_bytes: u64, warning_threshold: f64) -> bool {
    if let Some(limit) = get_gpu_memory_limit() {
        let threshold = (limit as f64 * warning_threshold) as u64;
        estimated_bytes > threshold
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let stats = get_memory_stats();
        #[cfg(target_os = "macos")]
        {
            assert!(stats.total_bytes > 0);
        }
        let _ = stats;
    }

    #[test]
    fn test_mlx_memory_queries() {
        let active = get_active_memory();
        let peak = get_peak_memory();
        let cache = get_cache_memory();
        let limit = get_memory_limit();
        let _ = (active, peak, cache, limit);
    }

    #[test]
    fn test_clear_cache() {
        clear_cache();
    }

    #[test]
    fn test_reset_peak_memory() {
        reset_peak_memory();
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }

    #[test]
    fn test_estimate_tensor_memory() {
        let mem = estimate_tensor_memory(&[2, 3, 4], 4);
        assert_eq!(mem, 2 * 3 * 4 * 4);
    }

    #[test]
    fn test_estimate_model_memory() {
        let num_params = 1_000_000u64;
        let fp16_size = 2;

        let weights_only = estimate_model_memory(num_params, fp16_size, false, false);
        assert_eq!(weights_only, 2_000_000);

        let with_grads = estimate_model_memory(num_params, fp16_size, true, false);
        assert_eq!(with_grads, 4_000_000);

        let with_all = estimate_model_memory(num_params, fp16_size, true, true);
        assert_eq!(with_all, 8_000_000);
    }

    #[test]
    fn test_estimate_training_memory() {
        let mem = estimate_training_memory(7_000_000_000, 2, 4, 2048, 4096, 32);
        assert!(mem > 50_000_000_000);
    }

    #[test]
    fn test_memory_efficient_context() {
        {
            let _ctx = MemoryEfficientContext::new();
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_system_memory() {
        let mem = get_system_memory();
        assert!(mem.is_some());
        assert!(mem.unwrap() > 0);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_system_memory_is_cached() {
        let first = get_system_memory();
        let second = get_system_memory();

        assert_eq!(first, second);
        assert_eq!(first, query_system_memory());
    }
}
