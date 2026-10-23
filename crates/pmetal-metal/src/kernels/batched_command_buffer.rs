#![allow(unsafe_code)]

//! Batched command buffer: many dispatches, one wait.
//!
//! Without batching, every kernel gets its own command buffer:
//! ```text
//! for each kernel:
//!     create_command_buffer()
//!     encode_kernel()
//!     commit()
//!     waitUntilCompleted()  // GPU-CPU sync, ~0.1ms overhead
//! ```
//!
//! With many dispatches per training step, that adds up.
//! [`BatchedCommandBuffer`] encodes every dispatch into one command buffer and
//! waits once, at the end.

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use crate::context::{CommandQueueExt, MetalContext};
use crate::error::{MetalError, Result};

/// A command buffer that batches multiple kernel dispatches.
///
/// Instead of creating a new command buffer for each kernel and waiting
/// after each one, this struct accumulates dispatches and only waits
/// at the end.
///
/// # Usage
///
/// ```ignore
/// let mut batch = BatchedCommandBuffer::new(ctx)?;
///
/// // Encode multiple dispatches (no waiting)
/// encode_first(batch.encoder().unwrap());
/// batch.add_dispatch();
/// encode_second(batch.encoder().unwrap());
/// batch.add_dispatch();
///
/// // Execute all at once
/// batch.execute()?;  // Single GPU-CPU sync
/// ```
pub struct BatchedCommandBuffer {
    ctx: Arc<MetalContext>,
    command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    dispatch_count: usize,
}

impl BatchedCommandBuffer {
    /// Create a new batched command buffer.
    pub fn new(ctx: Arc<MetalContext>) -> Result<Self> {
        let command_buffer = ctx.command_queue().new_command_buffer()?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        Ok(Self {
            ctx,
            command_buffer: Some(command_buffer),
            encoder: Some(encoder),
            dispatch_count: 0,
        })
    }

    /// Get the context.
    pub fn context(&self) -> &Arc<MetalContext> {
        &self.ctx
    }

    /// Get the current encoder for adding dispatches.
    ///
    /// Returns None if the buffer has already been executed.
    pub fn encoder(&self) -> Option<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.encoder.as_ref().map(|e| e.as_ref())
    }

    /// Get the encoder, or an error once the buffer has been executed.
    pub(crate) fn encoder_mut(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.encoder
            .as_ref()
            .map(|e| e.as_ref())
            .ok_or(MetalError::CommandBufferCreation)
    }

    /// Get the number of dispatches queued.
    pub fn dispatch_count(&self) -> usize {
        self.dispatch_count
    }

    /// Increment dispatch count.
    pub fn add_dispatch(&mut self) {
        self.dispatch_count += 1;
    }

    /// Execute all queued operations.
    ///
    /// This ends encoding, commits the command buffer, and waits for completion.
    /// After this call, the buffer cannot be reused.
    pub fn execute(mut self) -> Result<()> {
        // End encoding
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }

        // Commit and wait
        if let Some(command_buffer) = self.command_buffer.take() {
            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            if let Some(error) = command_buffer.error() {
                return Err(MetalError::ExecutionFailed(error.to_string()));
            }
        }

        tracing::trace!(
            "BatchedCommandBuffer executed {} dispatches",
            self.dispatch_count
        );
        Ok(())
    }

    /// Execute without waiting (for async patterns).
    ///
    /// Returns a handle that can be used to wait later.
    pub fn execute_async(mut self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }

        let command_buffer = self
            .command_buffer
            .take()
            .ok_or(MetalError::CommandBufferCreation)?;

        command_buffer.commit();
        Ok(command_buffer)
    }
}

impl Drop for BatchedCommandBuffer {
    fn drop(&mut self) {
        // Ensure encoder is properly ended even if execute() wasn't called.
        // Metal requires endEncoding() before the encoder is deallocated.
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        // Note: We don't commit or wait - just clean up the encoder.
        // The command buffer will be dropped automatically without execution.
    }
}
