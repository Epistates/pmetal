#![allow(unsafe_code)]

//! Shared dispatch helper for single-pass Metal compute kernels.
//!
//! The pmetal-metal crate was flagged by the April 2026 audit as having
//! "34 kernels × 7-step boilerplate" — every kernel dispatcher hand-rolls
//! the same (a) pipeline lookup, (b) command buffer creation, (c) encoder
//! setup, (d) setComputePipelineState, (e) setBuffer loop, (f) dispatch,
//! (g) commit + wait sequence. This module captures the most common shape
//! of that pattern as [`dispatch_simple_kernel`].
//!
//! ## When to use the helper
//!
//! This helper fits kernels that:
//! * Look up a single pipeline by function name (no function constants).
//! * Bind a fixed list of `MTLBuffer`s at index 0, 1, 2, … (no `setBytes`,
//!   no textures).
//! * Issue one `dispatchThreads` call with a grid size and an optional
//!   threadgroup size (auto-computed from the pipeline's max threads
//!   when omitted).
//! * Want the blocking behaviour `commit` + `waitUntilCompleted` provides.
//!
//! Kernels that need any of: function-constant specialisation, `setBytes`
//! for inline parameter structs, multi-dispatch sequences inside one
//! command buffer, or asynchronous completion callbacks, should continue
//! to use the explicit path — the dequant/sampler style of hand-rolling
//! the 7 steps manually. Wrapping those behind a more elaborate helper
//! would just hide complexity rather than remove it.
//!
//! ## Coverage
//!
//! As of this commit, `DequantKernels` in `dequant.rs` is migrated. Other
//! simple single-pass kernels can adopt the helper incrementally — no
//! atomic rollout required.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use crate::context::{CommandQueueExt, MetalContext};
use crate::error::{MetalError, Result};

/// Alias for the widely-used Metal buffer protocol object — every kernel
/// dispatcher in the crate types its buffers this way.
type MetalBufferRef = Retained<ProtocolObject<dyn MTLBuffer>>;

/// Dispatches a single-pass Metal compute kernel and blocks until the GPU
/// finishes. See [module docs](self) for when to prefer this helper vs. a
/// hand-rolled path.
///
/// # Parameters
///
/// * `ctx` — the global [`MetalContext`] (owns device, command queue,
///   pipeline cache).
/// * `kernel_name` — entry-point function name in the compiled Metal
///   library. Used as the pipeline cache key.
/// * `buffers` — ordered slice of `MTLBuffer`s bound at indices
///   `0..buffers.len()`. Each binding uses `offset = 0`.
/// * `grid_size` — total thread count per dimension (`dispatchThreads`
///   semantics, not threadgroup count).
/// * `thread_group_size` — optional explicit threadgroup size. When
///   `None`, the helper picks `min(grid_size.width, pipeline_max_total)`
///   for `width` and `1` for `height`/`depth` — matching the idiom used
///   across the crate.
///
/// # Errors
///
/// Returns `Err` when the pipeline lookup fails or when the Metal
/// runtime declines to create a command buffer / encoder. The latter is
/// typically a sign that the process already holds one too many active
/// encoders — the helper converts that into a typed error instead of
/// the `None.unwrap()` panic the hand-rolled kernels still use today.
pub fn dispatch_simple_kernel(
    ctx: &MetalContext,
    kernel_name: &str,
    buffers: &[&MetalBufferRef],
    grid_size: MTLSize,
    thread_group_size: Option<MTLSize>,
) -> Result<()> {
    // Metal hands back the command buffer and encoder autoreleased; a thread
    // with no pool of its own would hold them until it exits.
    objc2::rc::autoreleasepool(|_| {
        dispatch_in_pool(ctx, kernel_name, buffers, grid_size, thread_group_size)
    })
}

fn dispatch_in_pool(
    ctx: &MetalContext,
    kernel_name: &str,
    buffers: &[&MetalBufferRef],
    grid_size: MTLSize,
    thread_group_size: Option<MTLSize>,
) -> Result<()> {
    // (a) Pipeline
    let pipeline = {
        let mut cache = ctx.pipeline_cache_mut();
        cache.get_or_create_pipeline(ctx.device(), kernel_name, None)?
    };

    // (b) Command buffer + (c) encoder — use the typed Metal errors so
    // callers get a structured failure instead of the legacy `None.unwrap()`.
    let command_buffer = ctx.command_queue().new_command_buffer()?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or(MetalError::EncoderCreation)?;

    // (d) Pipeline state
    encoder.setComputePipelineState(&pipeline);

    // (e) Buffer bindings at indices 0..N
    // SAFETY: `setBuffer_offset_atIndex` is Metal's standard buffer-binding
    // entry point. Each buffer is a retained `ProtocolObject<dyn MTLBuffer>`
    // with a valid lifetime through the encoder's scope.
    unsafe {
        for (i, buf) in buffers.iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buf.as_ref()), 0, i);
        }
    }

    // (f) Dispatch — auto-derive threadgroup size from the pipeline if the
    // caller didn't supply one, matching the legacy dequant kernel style.
    let tg_size = thread_group_size.unwrap_or_else(|| MTLSize {
        width: pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(grid_size.width.max(1)),
        height: 1,
        depth: 1,
    });
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
    encoder.endEncoding();

    // (g) Commit + wait (blocking semantics matching the legacy kernels).
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // A command buffer that the GPU failed is not a result.
    if let Some(error) = command_buffer.error() {
        return Err(MetalError::ExecutionFailed(error.to_string()));
    }

    Ok(())
}

/// Convenience wrapper around [`dispatch_simple_kernel`] for kernels that
/// take a 1-D grid keyed on an element count. Handles the common
/// "N elements, width = N" pattern the dequant kernels use.
///
/// Equivalent to calling `dispatch_simple_kernel` with
/// `MTLSize { width: n_elements, height: 1, depth: 1 }`.
pub fn dispatch_linear_kernel(
    ctx: &MetalContext,
    kernel_name: &str,
    buffers: &[&MetalBufferRef],
    n_elements: usize,
) -> Result<()> {
    let grid_size = MTLSize {
        width: n_elements,
        height: 1,
        depth: 1,
    };
    dispatch_simple_kernel(ctx, kernel_name, buffers, grid_size, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{BufferUsage, MetalBuffer};

    /// Resident memory of this process, in KiB.
    fn resident_kib() -> i64 {
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .expect("ps");
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .expect("ps prints a number")
    }

    /// A dispatch from a thread with no autorelease pool of its own (every
    /// Rust thread) must not accumulate the command buffer and encoder Metal
    /// hands back autoreleased: without a pool around the dispatch this grew
    /// by about 1.8 KiB per call, forever.
    #[test]
    fn repeated_dispatches_do_not_accumulate_autoreleased_objects() {
        let ctx = MetalContext::new().unwrap();
        let n = 4096usize;
        let input = MetalBuffer::<u8>::zeros(&ctx, n / 32 * 18, BufferUsage::Shared).unwrap();
        let output = MetalBuffer::<f32>::zeros(&ctx, n, BufferUsage::Shared).unwrap();
        let (input, output) = (input.as_retained(), output.as_retained());
        let dispatch = || {
            dispatch_linear_kernel(&ctx, "dequantize_q4_0", &[&input, &output], n).unwrap();
        };

        for _ in 0..200 {
            dispatch();
        }
        let before = resident_kib();
        for _ in 0..10_000 {
            dispatch();
        }
        let growth_kib = resident_kib() - before;
        assert!(
            growth_kib < 6 * 1024,
            "10k dispatches grew resident memory by {growth_kib} KiB"
        );
    }

    #[test]
    fn a_dispatch_of_an_unknown_kernel_is_an_error() {
        let ctx = MetalContext::new().unwrap();
        let buffer = MetalBuffer::<f32>::zeros(&ctx, 4, BufferUsage::Shared).unwrap();
        let buffer = buffer.as_retained();
        let result = dispatch_linear_kernel(&ctx, "no_such_kernel", &[&buffer], 4);
        assert!(matches!(result, Err(MetalError::FunctionNotFound(_))));
    }
}
