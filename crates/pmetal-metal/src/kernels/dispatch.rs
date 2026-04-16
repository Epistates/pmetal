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

use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use crate::context::MetalContext;
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
    // (a) Pipeline
    let pipeline = {
        let mut cache = ctx.pipeline_cache_mut();
        cache.get_or_create_pipeline(ctx.device(), kernel_name, None)?
    };

    // (b) Command buffer + (c) encoder — use the typed Metal errors so
    // callers get a structured failure instead of the legacy `None.unwrap()`.
    let command_buffer = ctx
        .command_queue()
        .commandBuffer()
        .ok_or(MetalError::CommandBufferCreation)?;
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

/// Like [`dispatch_simple_kernel`] but additionally binds an inline
/// parameters struct at `buffers.len()` via `setBytes_length_atIndex`.
///
/// The Metal side sees:
/// * buffer 0 … N-1 — the elements of `buffers`
/// * buffer N (index = `buffers.len()`) — a `constant T&` parameter struct
///
/// That mirrors the convention used throughout the pmetal kernels (see
/// e.g. `fused_swiglu`, `fused_rope`, `fused_moe::FusedMoeExpert`): all
/// tensor inputs come first, the small `#[repr(C)]` config struct last.
///
/// # Safety invariants imposed on `T`
///
/// `T` must be a `#[repr(C)]` / plain-old-data type whose in-memory
/// representation exactly matches the Metal shader's `constant ParamsT&`.
/// Rust can't verify this; callers uphold it — typically by deriving
/// `zerocopy::{IntoBytes, Immutable, KnownLayout}` alongside `#[repr(C)]`.
/// `std::mem::size_of::<T>()` is passed to the Metal API, so the struct
/// must have a stable, well-defined ABI (no references, no unions of
/// varying-size variants, no `usize` / `*const T` fields — use fixed
/// widths).
///
/// The helper records + immediately dispatches + commits + waits, so the
/// caller's `params: &T` only needs to outlive this function call.
pub fn dispatch_kernel_with_params<T>(
    ctx: &MetalContext,
    kernel_name: &str,
    buffers: &[&MetalBufferRef],
    params: &T,
    grid_size: MTLSize,
    thread_group_size: Option<MTLSize>,
) -> Result<()> {
    // (a) Pipeline
    let pipeline = {
        let mut cache = ctx.pipeline_cache_mut();
        cache.get_or_create_pipeline(ctx.device(), kernel_name, None)?
    };

    // (b) Command buffer + (c) encoder
    let command_buffer = ctx
        .command_queue()
        .commandBuffer()
        .ok_or(MetalError::CommandBufferCreation)?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or(MetalError::EncoderCreation)?;

    // (d) Pipeline state
    encoder.setComputePipelineState(&pipeline);

    // (e) Buffers at 0..N, then the params struct at index N.
    // SAFETY:
    // * `setBuffer_offset_atIndex` is Metal's standard buffer-binding
    //   entry point; the retained `ProtocolObject<dyn MTLBuffer>`s
    //   outlive the encoder (held by the caller).
    // * `setBytes_length_atIndex` takes a pointer to `params` + byte
    //   length. The pointer is derived from a valid `&T` reference, so
    //   it is non-null and properly aligned. `size_of::<T>()` is the
    //   authoritative byte size for a `#[repr(C)]` struct; the caller's
    //   docs-level contract guarantees the shader's `constant T&` matches.
    unsafe {
        for (i, buf) in buffers.iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buf.as_ref()), 0, i);
        }
        let params_ptr = NonNull::from(params).cast();
        encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of::<T>(), buffers.len());
    }

    // (f) Dispatch.
    let tg_size = thread_group_size.unwrap_or_else(|| MTLSize {
        width: pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(grid_size.width.max(1)),
        height: 1,
        depth: 1,
    });
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
    encoder.endEncoding();

    // (g) Commit + wait.
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    Ok(())
}

// Re-export Arc so downstream kernel modules can name it through the
// dispatch module rather than rediscovering the std::sync import. Kept
// unexported for now — revisit if helper variants end up stashing
// command-queue state.
#[allow(dead_code)]
type _ArcMetalContext = Arc<MetalContext>;

#[cfg(test)]
mod tests {
    //! These tests intentionally avoid spawning a real Metal context — the
    //! dispatch pipeline is tested end-to-end via the downstream kernels
    //! (`dequant`, etc.) that have integration coverage. The assertions
    //! here are purely compile-time: they verify the helper signatures
    //! expose the types downstream kernels expect.
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct _NoopParams {
        n: u32,
        scale: f32,
    }

    #[allow(dead_code)]
    fn _signature_check(
        ctx: &crate::context::MetalContext,
        bufs: &[&super::MetalBufferRef],
    ) -> crate::error::Result<()> {
        super::dispatch_linear_kernel(ctx, "noop", bufs, 1024)?;
        let params = _NoopParams {
            n: 1024,
            scale: 1.0,
        };
        super::dispatch_kernel_with_params(
            ctx,
            "noop_with_params",
            bufs,
            &params,
            objc2_metal::MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
            None,
        )?;
        Ok(())
    }
}
