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

/// Build MLX's Metal device now, loading the metallib given to
/// [`set_metallib_path`].
///
/// MLX otherwise builds it on the first allocation. If the library can't be
/// loaded, that throws from whichever bridge call allocates first, and several
/// (memory queries, `synchronize`, stream setup) have no error guard, so the
/// process aborts; the guarded ones leave placeholder arrays that a caller
/// that doesn't check each op keeps computing on. Calling this first turns
/// that into one error, reported before any work starts.
///
/// [`validate_metallib`] catches a wrong or truncated file without the GPU;
/// this also catches one whose header is intact but whose contents MLX
/// rejects, such as a metallib format it no longer supports.
pub fn init_device() -> crate::BridgeResult<()> {
    unsafe { mlx_inline_init_device() };
    crate::check_last_error()
}

/// Check that `path` looks like a whole Metal library before handing it to
/// [`set_metallib_path`], without touching the GPU.
///
/// MLX builds its Metal device on the first allocation, and a library it
/// can't load makes that throw from whichever bridge call allocates first,
/// often outside any error guard, which aborts the process. Rejecting a bad
/// file here lets the caller fall back or report it instead. The header starts
/// with the magic `MTLB` and records the file's total size as a little-endian
/// u64 at offset 16 (true of MLX's metallib and of Apple's own), so this
/// catches a wrong file, an empty one, and a truncated copy.
pub fn validate_metallib(path: &std::path::Path) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind, Read};

    let invalid = |why: String| Error::new(ErrorKind::InvalidData, why);
    let mut file = std::fs::File::open(path)?;
    let actual = file.metadata()?.len();
    let mut header = [0u8; 24];
    file.read_exact(&mut header)
        .map_err(|_| invalid(format!("{actual} bytes is too short for a Metal library")))?;
    if &header[..4] != b"MTLB" {
        return Err(invalid("not a Metal library (no MTLB header)".into()));
    }
    let declared = u64::from_le_bytes(header[16..24].try_into().unwrap());
    if declared != actual {
        return Err(invalid(format!(
            "header declares {declared} bytes but the file has {actual}; it is truncated or corrupt"
        )));
    }
    Ok(())
}

/// Stop the Metal GPU capture.
///
/// Diagnostic entry point — retained for ad-hoc profiling.
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

/// Create this thread's generation stream, once per thread; make it the
/// default with [`set_generation_stream`]. Matches Python's
/// `generation_stream = mx.new_stream(mx.default_device())`. Each thread has
/// its own, since an MLX stream works only on the thread that created it.
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
pub fn enable_compile() {
    unsafe { mlx_inline_enable_compile() }
}

/// Disable MLX global compilation.
///
/// Diagnostic toggle — retained for A/B perf experiments.
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

    /// The metallib this build uses and hands out, which the header check must
    /// accept. build.rs exports its path, since with a reused MLX prefix it is
    /// not under OUT_DIR; it exports none when there is no metallib (no
    /// `metal` feature), and then there is nothing to check.
    #[test]
    fn the_built_metallib_validates() {
        let Some(built) = option_env!("PMETAL_BRIDGE_MLX_METALLIB") else {
            return;
        };
        validate_metallib(std::path::Path::new(built)).expect("MLX's own metallib is valid");
    }

    #[test]
    fn a_wrong_empty_or_truncated_file_is_rejected() {
        let dir = std::env::temp_dir().join(format!("pmetal-metallib-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let p = dir.join(name);
            std::fs::write(&p, bytes).unwrap();
            p
        };
        let header = |size: u64| {
            let mut h = b"MTLB".to_vec();
            h.resize(16, 0);
            h.extend_from_slice(&size.to_le_bytes());
            h
        };

        let mut whole = header(64);
        whole.resize(64, 0);
        assert!(validate_metallib(&write("whole", &whole)).is_ok());

        let mut truncated = header(64);
        truncated.resize(40, 0);
        assert!(validate_metallib(&write("truncated", &truncated)).is_err());
        assert!(validate_metallib(&write("empty", b"")).is_err());
        assert!(validate_metallib(&write("text", b"not a metallib, just some text")).is_err());
        assert!(validate_metallib(&dir.join("missing")).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

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
