#![allow(unsafe_code)]

//! Batched Metal compute-kernel dispatch.
//!
//! Every kernel in `pmetal-metal` currently records a single dispatch into
//! its own fresh command buffer and then blocks on
//! `waitUntilCompleted()`. For chains of small kernels that's two sources
//! of waste — a CPU↔GPU round-trip between each kernel, and lost
//! opportunity for the GPU scheduler to interleave work.
//!
//! [`BatchEncoder`] fixes both by owning one command buffer + one compute
//! encoder for the life of a call chain: the caller issues N
//! [`BatchEncoder::dispatch`] calls (each records pipeline state + buffer
//! bindings + `dispatchThreads`), then calls [`BatchEncoder::finish`]
//! once at the end, which closes the encoder, commits the buffer, and
//! blocks until the whole batch finishes.
//!
//! ```ignore
//! use pmetal_metal::kernels::batch_encoder::BatchEncoder;
//! use objc2_metal::MTLSize;
//!
//! let mut batch = BatchEncoder::new(ctx)?;
//! batch.dispatch_linear("rms_norm",       &[&x, &w, &out1], 1024)?;
//! batch.dispatch_linear("silu",           &[&out1, &out2],  1024)?;
//! batch.dispatch_linear("matmul_small",   &[&out2, &wt, &final_out], 4096)?;
//! batch.finish()?;   // one commit + one wait for all three
//! ```
//!
//! ## Semantics
//!
//! * Kernels run in the order they're dispatched. Metal's default
//!   execution model ensures sequential ordering within one encoder,
//!   so a later dispatch sees the previous one's memory effects —
//!   equivalent to the legacy commit-per-dispatch style but without the
//!   flush.
//! * All dispatches use `offset = 0` on every bound buffer and share the
//!   same pipeline cache as [`super::dispatch`].
//! * `BatchEncoder` is deliberately `!Send` / `!Sync` — command encoders
//!   must stay on the thread that created them. The Metal runtime will
//!   crash at dispatch time on misuse; the lifetime tie to the
//!   `MetalContext` makes the common case safe.
//! * Dropping a [`BatchEncoder`] without calling [`BatchEncoder::finish`]
//!   leaves the encoder un-ended — a programming error that would
//!   normally wedge Metal's state machine. The Drop impl defensively
//!   ends the encoder and discards the buffer (no commit) so the GPU
//!   stays healthy even when the caller panics mid-batch.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use crate::context::MetalContext;
use crate::error::{MetalError, Result};

/// Re-type the widely-used buffer protocol object to match
/// [`super::dispatch::dispatch_simple_kernel`].
type MetalBufferRef = Retained<ProtocolObject<dyn MTLBuffer>>;

/// Accumulates multiple compute dispatches into one command buffer.
///
/// Borrowed from [`MetalContext`] so the encoder can't outlive the
/// context that owns the underlying device + command queue. See the
/// [module docs](self) for usage.
pub struct BatchEncoder<'ctx> {
    ctx: &'ctx MetalContext,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    /// Tripped by `finish()`; the Drop impl uses this to decide whether
    /// to end the encoder / commit the buffer defensively.
    finished: bool,
}

impl<'ctx> BatchEncoder<'ctx> {
    /// Starts a new batch. Allocates one command buffer and one compute
    /// encoder upfront; both are released in either [`finish`] (success)
    /// or [`Drop`] (panic path).
    ///
    /// [`finish`]: Self::finish
    pub fn new(ctx: &'ctx MetalContext) -> Result<Self> {
        let command_buffer = ctx
            .command_queue()
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;
        Ok(Self {
            ctx,
            command_buffer,
            encoder,
            finished: false,
        })
    }

    /// Records a single compute-kernel dispatch with the supplied grid /
    /// threadgroup shape. No commit happens here — work is held in the
    /// command buffer until [`finish`] is called.
    ///
    /// When `thread_group_size` is `None`, the helper picks
    /// `min(grid_size.width, pipeline.maxTotalThreadsPerThreadgroup)` for
    /// the width, matching the idiom used elsewhere in the crate.
    ///
    /// [`finish`]: Self::finish
    pub fn dispatch(
        &self,
        kernel_name: &str,
        buffers: &[&MetalBufferRef],
        grid_size: MTLSize,
        thread_group_size: Option<MTLSize>,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), kernel_name, None)?
        };

        self.encoder.setComputePipelineState(&pipeline);

        // SAFETY: `setBuffer_offset_atIndex` is Metal's standard buffer
        // binding entry point. Each buffer is a retained
        // `ProtocolObject<dyn MTLBuffer>` whose lifetime outlives this
        // encoder (the caller holds the `Retained<…>` clones).
        unsafe {
            for (i, buf) in buffers.iter().enumerate() {
                self.encoder
                    .setBuffer_offset_atIndex(Some(buf.as_ref()), 0, i);
            }
        }

        let tg_size = thread_group_size.unwrap_or_else(|| MTLSize {
            width: pipeline
                .maxTotalThreadsPerThreadgroup()
                .min(grid_size.width.max(1)),
            height: 1,
            depth: 1,
        });
        self.encoder
            .dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
        Ok(())
    }

    /// Convenience wrapper for the very common "1-D grid of N elements"
    /// pattern (matches [`super::dispatch::dispatch_linear_kernel`]).
    pub fn dispatch_linear(
        &self,
        kernel_name: &str,
        buffers: &[&MetalBufferRef],
        n_elements: usize,
    ) -> Result<()> {
        let grid_size = MTLSize {
            width: n_elements,
            height: 1,
            depth: 1,
        };
        self.dispatch(kernel_name, buffers, grid_size, None)
    }

    /// Closes the encoder, commits the command buffer, and blocks until
    /// the GPU has executed every queued dispatch.
    ///
    /// This is the one place ownership of the encoder is consumed, so
    /// callers who have dispatched N kernels must call `finish` exactly
    /// once. The method takes `mut self` by value to statically prevent
    /// double-finishing.
    pub fn finish(mut self) -> Result<()> {
        self.encoder.endEncoding();
        self.command_buffer.commit();
        self.command_buffer.waitUntilCompleted();
        self.finished = true;
        Ok(())
    }
}

impl Drop for BatchEncoder<'_> {
    fn drop(&mut self) {
        // The caller skipped `finish()` — either because they panicked
        // mid-batch or because of an early `?` return. End the encoder so
        // Metal doesn't wedge; deliberately skip `commit()` because the
        // batch is semantically abandoned at this point.
        if !self.finished {
            self.encoder.endEncoding();
        }
    }
}

#[cfg(test)]
mod tests {
    //! Real Metal context integration is covered by downstream kernel
    //! tests. These assertions just verify the API shape (compile-time
    //! contracts) — no GPU work happens here.

    #[allow(dead_code)]
    fn _signature_check(ctx: &crate::context::MetalContext) -> crate::error::Result<()> {
        use super::BatchEncoder;
        let batch = BatchEncoder::new(ctx)?;
        batch.dispatch_linear("noop", &[], 512)?;
        batch.finish()
    }
}
