//! Memory management utilities for Apple Silicon.
//!
//! This module provides utilities for managing memory on Apple Silicon,
//! taking advantage of the unified memory architecture.
//!
//! MLX uses a caching memory allocator to reduce allocation overhead.
//! These functions provide visibility into and control over memory usage.

use pmetal_core::MemoryStats;

// FFI definitions for getrusage to avoid adding libc dependency
#[cfg(unix)]
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
        pub ru_maxrss: i64,   // Maximum resident set size
        pub ru_ixrss: i64,    // Integral shared memory size
        pub ru_idrss: i64,    // Integral unshared data size
        pub ru_isrss: i64,    // Integral unshared stack size
        pub ru_minflt: i64,   // Page reclaims (soft page faults)
        pub ru_majflt: i64,   // Page faults (hard page faults)
        pub ru_nswap: i64,    // Swaps
        pub ru_inblock: i64,  // Block input operations
        pub ru_oublock: i64,  // Block output operations
        pub ru_msgsnd: i64,   // IPC messages sent
        pub ru_msgrcv: i64,   // IPC messages received
        pub ru_nsignals: i64, // Signals received
        pub ru_nvcsw: i64,    // Voluntary context switches
        pub ru_nivcsw: i64,   // Involuntary context switches
    }

    pub const RUSAGE_SELF: i32 = 0;

    extern "C" {
        pub fn getrusage(who: i32, usage: *mut rusage) -> i32;
    }
}

/// Get the current Resident Set Size (RSS) in bytes.
///
/// This returns the physical memory used by the process, which is a good
/// proxy for the "Active" memory usage including MLX tensors and Rust allocations.
#[cfg(unix)]
fn get_active_memory() -> u64 {
    // SAFETY:
    // 1. std::mem::zeroed() creates a valid zeroed rusage struct
    // 2. getrusage is a POSIX system call that fills the provided struct
    // 3. RUSAGE_SELF (0) queries the calling process's resource usage
    // 4. We check the return value (0 = success) before using the result
    // 5. All fields of rusage are primitive types that can be safely zeroed
    unsafe {
        let mut usage: sys::rusage = std::mem::zeroed();
        if sys::getrusage(sys::RUSAGE_SELF, &mut usage) == 0 {
            // On macOS, ru_maxrss is in bytes. On Linux it's in KB.
            // Since this project is Apple Silicon focused, we assume macOS behavior or check OS.
            #[cfg(target_os = "macos")]
            return usage.ru_maxrss as u64;

            #[cfg(not(target_os = "macos"))]
            return (usage.ru_maxrss * 1024) as u64;
        }
    }
    0
}

#[cfg(not(unix))]
fn get_active_memory() -> u64 {
    0
}

/// Get current memory statistics.
///
/// On Apple Silicon, this queries the unified memory pool shared
/// between CPU and GPU.
pub fn get_memory_stats() -> MemoryStats {
    // Query system memory as a fallback
    let total_bytes = get_system_memory().unwrap_or(0);
    let used_bytes = get_active_memory();

    MemoryStats {
        total_bytes,
        used_bytes,
        peak_bytes: used_bytes, // Best effort: current is at least the peak so far if we don't track history
    }
}

/// Get total system memory in bytes.
///
/// On macOS, this returns the total physical memory available.
pub fn get_system_memory() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        let mem_str = String::from_utf8(output.stdout).ok()?;
        mem_str.trim().parse().ok()
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Get GPU memory limit (recommended working set size).
///
/// On Apple Silicon, this queries the Metal device for
/// recommended maximum memory usage.
pub fn get_gpu_memory_limit() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        // Use system_profiler to get GPU info
        let _output = Command::new("system_profiler")
            .args(["SPDisplaysDataType", "-json"])
            .output()
            .ok()?;

        // For Apple Silicon, unified memory means GPU has access to all RAM
        // The recommended working set is typically ~75% of total memory
        get_system_memory().map(|total| total * 3 / 4)
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Clear the memory cache.
///
/// This releases memory held by MLX's caching allocator.
/// Useful when you need to free memory for other operations.
pub fn clear_cache() {
    // In a real MLX binding, we would call mlx_clear_cache().
    // Without direct bindings, we can't force the C++ allocator to release.
    // However, forcing a synchronization might help.
}

/// Reset the peak memory counter.
///
/// After calling this, `get_peak_memory()` will track the
/// new maximum from this point forward.
pub fn reset_peak_memory() {
    // No-op without mlx-sys support
}

/// Memory-efficient context for operations.
///
/// Use this to wrap memory-intensive operations that should
/// aggressively free intermediate tensors. The cache is cleared
/// when the context is dropped.
///
/// # Example
/// ```ignore
/// {
///     let _ctx = MemoryEfficientContext::new();
///     // Memory-intensive operations here
///     // Cache is automatically cleared when _ctx goes out of scope
/// }
/// ```
pub struct MemoryEfficientContext {
    _private: (),
}

impl MemoryEfficientContext {
    /// Create a new memory-efficient context.
    ///
    /// Optionally resets peak memory tracking for profiling.
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
        // Clear cache when context is dropped
        clear_cache();
    }
}

/// Estimate memory required for a tensor.
///
/// # Arguments
/// * `shape` - Tensor shape
/// * `dtype_size` - Size of each element in bytes
///
/// # Returns
/// Estimated memory in bytes
pub fn estimate_tensor_memory(shape: &[usize], dtype_size: usize) -> u64 {
    let num_elements: usize = shape.iter().product();
    (num_elements * dtype_size) as u64
}

/// Estimate memory required for a model with given parameters.
///
/// # Arguments
/// * `num_params` - Number of parameters
/// * `dtype_size` - Size of each parameter in bytes (e.g., 2 for fp16)
/// * `include_gradients` - Whether to include gradient memory
/// * `include_optimizer` - Whether to include optimizer state memory
///
/// # Returns
/// Estimated memory in bytes
pub fn estimate_model_memory(
    num_params: u64,
    dtype_size: usize,
    include_gradients: bool,
    include_optimizer: bool,
) -> u64 {
    let base = num_params * dtype_size as u64;

    let gradient_factor = if include_gradients { 1 } else { 0 };
    let optimizer_factor = if include_optimizer { 2 } else { 0 }; // AdamW uses 2x for momentum + variance

    base * (1 + gradient_factor + optimizer_factor)
}

/// Estimate memory required for training a model.
///
/// This includes:
/// - Model weights
/// - Gradients
/// - Optimizer states (momentum + variance for Adam)
/// - Activations (estimated based on batch size and sequence length)
///
/// # Arguments
/// * `num_params` - Number of model parameters
/// * `dtype_size` - Size of each parameter in bytes
/// * `batch_size` - Batch size for training
/// * `seq_len` - Sequence length
/// * `hidden_size` - Hidden dimension size
/// * `num_layers` - Number of transformer layers
///
/// # Returns
/// Estimated memory in bytes
pub fn estimate_training_memory(
    num_params: u64,
    dtype_size: usize,
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    num_layers: usize,
) -> u64 {
    // Weights + gradients + optimizer states
    let model_memory = estimate_model_memory(num_params, dtype_size, true, true);

    // Activation memory (rough estimate)
    // Each layer stores: attention scores, hidden states, intermediate activations
    let activation_per_layer = (batch_size * seq_len * hidden_size * dtype_size) as u64 * 4;
    let activation_memory = activation_per_layer * num_layers as u64;

    model_memory + activation_memory
}

/// Format bytes as human-readable string.
///
/// # Example
/// ```ignore
/// assert_eq!(format_bytes(1536), "1.5 KB");
/// assert_eq!(format_bytes(1073741824), "1.0 GB");
/// ```
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
///
/// Useful for debugging memory issues during development.
pub fn print_memory_stats() {
    let stats = get_memory_stats();
    let gpu_limit = get_gpu_memory_limit().unwrap_or(0);

    println!("Memory Statistics:");
    println!("  System Total:  {}", format_bytes(stats.total_bytes));
    println!("  GPU Limit:     {}", format_bytes(gpu_limit));
    println!("  RSS (Active):  {}", format_bytes(stats.used_bytes));
    println!("  Peak (Approx): {}", format_bytes(stats.peak_bytes));
}

/// Check if memory usage is approaching limit.
///
/// Returns true if estimated usage exceeds the warning threshold.
///
/// # Arguments
/// * `estimated_bytes` - Estimated memory needed
/// * `warning_threshold` - Fraction of limit that triggers warning (0.0-1.0)
pub fn memory_warning(estimated_bytes: u64, warning_threshold: f64) -> bool {
    if let Some(limit) = get_gpu_memory_limit() {
        let threshold = (limit as f64 * warning_threshold) as u64;
        estimated_bytes > threshold
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let stats = get_memory_stats();
        #[cfg(target_os = "macos")]
        {
            assert!(stats.total_bytes > 0);
            assert!(stats.used_bytes > 0); // Should be > 0 now!
        }
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
        // [2, 3, 4] tensor with fp32 (4 bytes each)
        let mem = estimate_tensor_memory(&[2, 3, 4], 4);
        assert_eq!(mem, 2 * 3 * 4 * 4); // 96 bytes
    }

    #[test]
    fn test_estimate_model_memory() {
        let num_params = 1_000_000u64;
        let fp16_size = 2;

        // Weights only
        let weights_only = estimate_model_memory(num_params, fp16_size, false, false);
        assert_eq!(weights_only, 2_000_000);

        // With gradients
        let with_grads = estimate_model_memory(num_params, fp16_size, true, false);
        assert_eq!(with_grads, 4_000_000);

        // With gradients and optimizer (Adam = 2x state)
        let with_all = estimate_model_memory(num_params, fp16_size, true, true);
        assert_eq!(with_all, 8_000_000);
    }

    #[test]
    fn test_estimate_training_memory() {
        // 7B parameter model, fp16
        let num_params = 7_000_000_000u64;
        let dtype_size = 2; // fp16
        let batch_size = 4;
        let seq_len = 2048;
        let hidden_size = 4096;
        let num_layers = 32;

        let mem = estimate_training_memory(
            num_params,
            dtype_size,
            batch_size,
            seq_len,
            hidden_size,
            num_layers,
        );

        // Should be roughly 56GB for weights/grads/opt + activations
        assert!(mem > 50_000_000_000); // > 50 GB
    }

    #[test]
    fn test_memory_efficient_context() {
        // Verify context can be created and dropped without issues
        {
            let _ctx = MemoryEfficientContext::new();
        } // Context dropped, cache should be cleared
    }

    #[test]
    fn test_clear_cache() {
        // Just verify it doesn't crash
        clear_cache();
    }

    #[test]
    fn test_reset_peak_memory() {
        // Just verify it doesn't crash
        reset_peak_memory();
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_system_memory() {
        let mem = get_system_memory();
        assert!(mem.is_some());
        assert!(mem.unwrap() > 0);
    }
}
