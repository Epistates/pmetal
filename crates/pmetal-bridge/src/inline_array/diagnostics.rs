//! Diagnostic and resource-management entry points.
//!
//! These free functions are thin wrappers around global MLX runtime state:
//! graph introspection, Metal GPU capture, memory tracking, stream management,
//! global compile toggles, and buffer-layout verification.

use super::InlineArray;
use super::RawBuf;
use super::ffi::*;
use super::{ARRAY_BUF_ALIGN, ARRAY_BUF_SIZE};

// ── Graph / compile helpers ───────────────────────────────────────────────

/// Count the number of unique nodes in the computation graph rooted at this
/// array.  Useful for diagnosing performance — each node becomes a Metal
/// kernel dispatch during eval.
pub fn graph_node_count(arr: &InlineArray) -> usize {
    unsafe { mlx_inline_graph_node_count(&arr.raw) }
}

/// Count unique ArrayDesc nodes (the REAL graph nodes that map to dispatches).
pub fn graph_desc_count(arr: &InlineArray) -> usize {
    unsafe { mlx_inline_graph_desc_count(&arr.raw) }
}

/// Dump the graph topology to stderr: print every node's primitive type and shape.
pub fn graph_dump(arr: &InlineArray) {
    unsafe { mlx_inline_graph_dump(&arr.raw) }
}

/// Start a Metal GPU capture to the given .gputrace path.
/// Must run with MTL_CAPTURE_ENABLED=1 environment variable.
///
/// Diagnostic entry point for Xcode GPU traces — retained for ad-hoc profiling.
#[allow(dead_code)]
pub fn metal_start_capture(path: &str) -> bool {
    let c_path = std::ffi::CString::new(path).unwrap();
    unsafe { mlx_inline_metal_start_capture(c_path.as_ptr()) == 0 }
}

/// Point MLX at a specific `mlx.metallib` (upstream `set_metallib_path`,
/// MLX >= 0.32). Lazy: MLX consults the stored path when the Metal device
/// first loads its kernel library, so call this before the first GPU op.
/// Replaces the pre-0.32 source patch that read PMETAL_METALLIB_PATH.
pub fn set_metallib_path(path: &str) {
    let c_path = std::ffi::CString::new(path).unwrap();
    unsafe { mlx_inline_set_metallib_path(c_path.as_ptr()) }
}

/// Stop the Metal GPU capture.
///
/// Diagnostic entry point — retained for ad-hoc profiling.
#[allow(dead_code)]
pub fn metal_stop_capture() {
    unsafe { mlx_inline_metal_stop_capture() }
}

// ── Memory limits ────────────────────────────────────────────────────────

/// Set wired memory limit to maximum recommended — CRITICAL for GPU performance.
/// Without this, Metal buffers may be paged out causing massive overhead.
/// Returns previous limit.
pub fn set_wired_limit_max() -> usize {
    let max_size = unsafe { mlx_inline_get_max_recommended_size() };
    if max_size > 0 {
        unsafe { mlx_inline_set_wired_limit(max_size) }
    } else {
        0
    }
}

/// Set the wired memory limit, clamped to what the device allows.
///
/// Returns the previous limit. Asking for more than
/// [`get_max_recommended_size`] is not an error: the bridge clamps, because
/// MLX's own `set_wired_limit` throws on an over-large request and this runs
/// on the decode path where that would terminate the process.
pub fn set_wired_limit(limit: usize) -> usize {
    unsafe { mlx_inline_set_wired_limit(limit) }
}

/// The device's maximum recommended working set size.
///
/// This is the number MLX validates `set_wired_limit` against, read from the
/// device rather than estimated from installed RAM. The ratio to `hw.memsize`
/// is not a constant: 84% on a 128 GB M4 Max, 74% on a 16 GB M4 mini.
pub fn get_max_recommended_size() -> usize {
    unsafe { mlx_inline_get_max_recommended_size() }
}

// ── Stream management ────────────────────────────────────────────────────

/// Create a new GPU stream and set it as default for all subsequent ops.
/// Matches Python's `generation_stream = mx.new_stream(mx.default_device())`.
pub fn new_generation_stream() {
    unsafe {
        mlx_inline_new_stream();
    }
}

/// Set the generation stream as the default stream for all ops.
pub fn set_generation_stream() {
    unsafe {
        mlx_inline_set_default_stream(0);
    }
}

/// Restore MLX's original default stream (GPU stream on the default device).
///
/// Must be called after generation completes and before returning from the
/// inference function, so that InlineArray drops execute on the main stream
/// instead of the generation stream. Without this, array destructors race
/// with Metal teardown and cause SIGSEGV at program exit.
pub fn reset_default_stream() {
    unsafe {
        mlx_inline_reset_default_stream();
    }
}

/// Synchronize the generation stream (wait for all pending GPU work).
pub fn synchronize() {
    unsafe {
        mlx_inline_synchronize();
    }
}

// ── Cache / compile toggles ──────────────────────────────────────────────

/// Clear the Metal buffer cache — frees unused GPU memory.
/// Call periodically during generation to prevent memory accumulation.
pub fn clear_cache() {
    unsafe { mlx_inline_clear_cache() }
}

/// Enable MLX global compilation — fuses ops across the entire computation
/// graph.
///
/// Diagnostic toggle — retained for A/B perf experiments; not wired into
/// production paths (compile is managed per-fn via `mlx::core::compile`).
#[allow(dead_code)]
pub fn enable_compile() {
    unsafe { mlx_inline_enable_compile() }
}

/// Disable MLX global compilation.
///
/// Diagnostic toggle — retained for A/B perf experiments.
#[allow(dead_code)]
pub fn disable_compile() {
    unsafe { mlx_inline_disable_compile() }
}

// ── Batched eval ─────────────────────────────────────────────────────────

/// Eval a batch of arrays in a SINGLE GPU submission, then detach each one.
/// This is critical for cache arrays: eval+detach severs the computation
/// graph chain across decode steps without per-array sync barriers.
pub fn eval_and_detach_many(arrays: &mut [&mut InlineArray]) {
    if arrays.is_empty() {
        return;
    }
    let mut ptrs: Vec<*mut RawBuf> = arrays
        .iter_mut()
        .map(|a| &mut a.raw as *mut RawBuf)
        .collect();
    unsafe {
        mlx_inline_eval_many(ptrs.as_mut_ptr(), ptrs.len() as i32);
    }
    for a in arrays.iter_mut() {
        unsafe {
            mlx_inline_detach(&mut a.raw);
        }
    }
}

// ── Memory tracking ──────────────────────────────────────────────────────

/// Metal memory: bytes currently in use by live arrays.
pub fn get_active_memory() -> usize {
    unsafe { mlx_inline_get_active_memory() }
}

/// Metal memory: bytes freed but held in buffer cache for reuse.
pub fn get_cache_memory() -> usize {
    unsafe { mlx_inline_get_cache_memory() }
}

/// Metal memory: high-water mark of active memory.
pub fn get_peak_memory() -> usize {
    unsafe { mlx_inline_get_peak_memory() }
}

/// Reset the peak memory tracker.
pub fn reset_peak_memory() {
    unsafe { mlx_inline_reset_peak_memory() }
}

// ── Buffer-layout verification ───────────────────────────────────────────

/// Panic at runtime if the Rust buffer constants don't match the C++ values.
/// Call once at program startup (or in a test).
#[allow(dead_code)] // Diagnostic — only invoked by the layout sanity test.
pub fn verify_buffer_layout() {
    let sz = unsafe { mlx_inline_array_size() };
    let al = unsafe { mlx_inline_array_align() };
    assert!(
        sz <= ARRAY_BUF_SIZE,
        "mlx::core::array is {sz} bytes but ARRAY_BUF_SIZE={ARRAY_BUF_SIZE}"
    );
    assert!(
        al <= ARRAY_BUF_ALIGN,
        "mlx::core::array alignment is {al} but ARRAY_BUF_ALIGN={ARRAY_BUF_ALIGN}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⚠️ The regression this exists for: `get_max_recommended_size` returned
    /// `hw.memsize * 3 / 4` on the theory that Metal recommends 75% of RAM.
    /// The real ratio is not a constant (84% on a 128 GB M4 Max, 74% on a
    /// 16 GB M4 mini), so on the mini the estimate came out 160 MB *above* the
    /// device maximum, `mlx::core::set_wired_limit` threw, and the uncaught
    /// exception terminated the process before a single token was decoded.
    /// Both engines call this on every generate.
    #[test]
    fn the_recommended_size_is_always_a_legal_wired_limit() {
        let max = get_max_recommended_size();
        assert!(max > 0, "device reported no working set size");

        // Would have aborted the test binary, not failed the assertion.
        let previous = set_wired_limit(max);
        crate::check_last_error().expect("setting the reported maximum is legal");
        set_wired_limit(previous);
    }

    /// Over-asking is clamped rather than thrown, so a caller that does its own
    /// sizing arithmetic cannot take the process down.
    #[test]
    fn an_oversized_request_is_clamped_not_fatal() {
        let previous = set_wired_limit(usize::MAX);
        crate::check_last_error().expect("an oversized request is clamped");
        set_wired_limit(previous);
    }
}
