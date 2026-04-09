#![allow(unsafe_code)]

//! Shared dispatch helper for Metal 4 / MPP kernels.
//!
//! Eliminates the ~15-line boilerplate that was previously repeated in every
//! `mpp_*.rs` dispatch file:
//!
//! 1. Look up (or create) the pipeline from the Metal 4 library.
//! 2. Create a command buffer from the command queue.
//! 3. Create a compute encoder.
//! 4. Set the pipeline state.
//! 5. Invoke the caller-supplied buffer-binding closure.
//! 6. Dispatch threadgroups.
//! 7. End encoding.
//! 8. Commit the command buffer.
//!
//! # Metal 4 lifecycle migration note
//!
//! All callers currently use the Metal 3 command-buffer path (step 2 above).
//! When Metal 4 command buffers are runtime-tested on M5, replace the creation
//! block marked `TODO(metal4-lifecycle)` with:
//!
//! ```text
//! // TODO(metal4-lifecycle): When Metal4CommandBuffer is runtime-tested on M5,
//! // replace the command buffer creation below with:
//! //   let mut cb = Metal4CommandBuffer::new(device, pool)?;
//! //   cb.begin()?;
//! //   let encoder = cb.encoder()?;
//! //   ... bind + dispatch ...
//! //   cb.end_and_commit(queue)?;
//! ```
//!
//! That is a one-line change per call site because the binding closure and grid
//! arguments are unchanged.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::{
    context::MetalContext,
    error::{MetalError, Result},
};

/// Encode and commit a single Metal 4 / MPP kernel dispatch.
///
/// This helper handles the full lifecycle of one GPU compute dispatch:
/// pipeline lookup, command buffer and encoder creation, buffer binding
/// (via the `bind_buffers` closure), threadgroup dispatch, and commit.
///
/// The returned command buffer has been committed but **not** waited on.
/// Call `cb.waitUntilCompleted()` for synchronous execution.
///
/// # Arguments
///
/// - `ctx` — active Metal context.
/// - `kernel_name` — name of the Metal 4 function to dispatch.
/// - `grid` — threadgroup grid dimensions passed to
///   `dispatchThreadgroups_threadsPerThreadgroup`.
/// - `threadgroup_size` — threads per threadgroup.
/// - `bind_buffers` — closure that sets all `buffer(N)` / `bytes(N)` bindings
///   on the encoder before dispatch.
///
/// # Availability
///
/// Callers are responsible for checking `is_available()` (NAX + Metal 4 library
/// present) before calling this function. Calling without the Metal 4 library
/// loaded will return `Err(MetalError::LibraryLoad(...))`.
pub(crate) fn encode_mpp_kernel(
    ctx: &MetalContext,
    kernel_name: &str,
    grid: objc2_metal::MTLSize,
    threadgroup_size: objc2_metal::MTLSize,
    bind_buffers: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
    // Empty constants — all MPP kernels that use this helper rely on
    // function constants baked into the Metal 4 library at compile time
    // rather than runtime specialisation.
    let constants: HashMap<u64, crate::pipeline::FunctionConstant> = HashMap::new();

    let pipeline = {
        let mut cache = ctx.pipeline_cache_mut();
        cache.get_or_create_metal4_pipeline(ctx.device(), kernel_name, &constants)?
    };

    // TODO(metal4-lifecycle): When Metal4CommandBuffer is runtime-tested on M5,
    // replace the three lines below with:
    //   let mut cb = Metal4CommandBuffer::new(ctx.device(), pool)?;
    //   cb.begin()?;
    //   let encoder = cb.encoder()?;
    // and replace `command_buffer.commit()` at the end with:
    //   cb.end_and_commit(ctx.command_queue())?;
    let command_buffer = ctx
        .command_queue()
        .commandBuffer()
        .ok_or(MetalError::CommandBufferCreation)?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or(MetalError::EncoderCreation)?;

    encoder.setComputePipelineState(&pipeline);

    bind_buffers(&encoder);

    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();

    Ok(command_buffer)
}
