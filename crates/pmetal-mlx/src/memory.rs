//! Memory statistics for Apple Silicon.
//!
//! The allocator's own counters come from `pmetal-runtime` (re-exported
//! below for the callers that already reach them here); this module adds the
//! process-level figures, the installed RAM as the device reports it, and the
//! estimates the trainers use to size a run.

#![allow(unsafe_code)]

#[cfg(target_os = "macos")]
use std::sync::OnceLock;

use pmetal_core::MemoryStats;

pub use pmetal_runtime::inline_array::{
    clear_cache, get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory,
};

#[cfg(target_os = "macos")]
static SYSTEM_MEMORY_BYTES: OnceLock<Option<u64>> = OnceLock::new();

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Log current MLX memory state at INFO level.
pub fn log_memory_stats() {
    tracing::info!(
        "MLX memory: active={}, cache={}, peak={}, limit={}, recommended working set={}",
        format_bytes(get_active_memory() as u64),
        format_bytes(get_cache_memory() as u64),
        format_bytes(get_peak_memory() as u64),
        format_bytes(pmetal_runtime::inline_array::get_memory_limit() as u64),
        format_bytes(pmetal_runtime::inline_array::get_max_recommended_size() as u64),
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

/// The process's peak resident set size in bytes (`ru_maxrss`).
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
/// The allocator's active and peak bytes; before anything is allocated, the
/// process's peak resident size stands in for the active figure.
pub fn get_memory_stats() -> MemoryStats {
    let total_bytes = get_system_memory().unwrap_or(0);
    let active = get_active_memory() as u64;
    let peak = get_peak_memory() as u64;

    let used_bytes = if active > 0 { active } else { get_rss() };

    MemoryStats {
        total_bytes,
        used_bytes,
        peak_bytes: peak.max(used_bytes),
    }
}

/// Installed RAM in bytes, as the Metal device reports it (the GPU shares it).
pub fn get_system_memory() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        *SYSTEM_MEMORY_BYTES.get_or_init(|| {
            pmetal_runtime::inline_array::device_info()
                .ok()
                .map(|device| device.memory_size)
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// Estimation and formatting
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The installed RAM is what `sysctl hw.memsize` reports; the device's
    /// figure must agree with the machine's.
    #[test]
    #[cfg(target_os = "macos")]
    fn the_system_memory_is_the_installed_ram() {
        let from_sysctl: u64 = String::from_utf8(
            std::process::Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .parse()
        .unwrap();
        assert_eq!(get_system_memory(), Some(from_sysctl));
        assert_eq!(get_system_memory(), get_system_memory(), "cached");
    }

    /// The stats read the allocator's counters: while a 4 MiB array is alive,
    /// both the active and the peak figure cover it. (Other tests allocate
    /// and free concurrently, so only a lower bound is stable.)
    #[test]
    fn the_stats_follow_the_allocator() {
        use pmetal_runtime::compat::{Dtype, ops};
        let x = ops::ones(&[1024, 1024], Dtype::Float32);
        x.eval();
        let stats = get_memory_stats();
        assert!(stats.used_bytes >= 4 << 20, "{stats:?}");
        assert!(stats.peak_bytes >= 4 << 20, "{stats:?}");
        assert!(stats.total_bytes >= stats.used_bytes, "{stats:?}");
        drop(x);
    }
}
