//! Fused training kernels for maximum throughput.
//!
//! This module provides Metal kernels that eliminate GPU-CPU synchronization
//! overhead by batching all operations into a single command buffer per training step.
//!
//! # The Problem
//!
//! Standard execution pattern (what we had):
//! ```text
//! for each kernel:
//!     create_command_buffer()
//!     encode_kernel()
//!     commit()
//!     waitUntilCompleted()  // ← GPU-CPU sync, ~0.1ms overhead
//! ```
//!
//! With 100+ kernel dispatches per training step, this adds ~10-15ms overhead.
//!
//! # The Solution
//!
//! Batched execution pattern (what MLX's mx.compile does):
//! ```text
//! create_command_buffer()
//! for each kernel:
//!     encode_kernel()
//! commit()
//! waitUntilCompleted()  // ← Single sync at end
//! ```
//!
//! This module provides:
//! - [`FusedAdamW`]: All-parameter optimizer update in one dispatch
//! - [`FusedGradientClipping`]: Global norm + scaling in two dispatches
//! - [`FusedCrossEntropy`]: Loss + backward fused
//! - [`BatchedCommandBuffer`]: Helper for batched execution
//!
//! # Performance Target
//!
//! - Current: ~1740 tok/s (per-kernel sync)
//! - Target: ~2400 tok/s (match mlx_lm)
//! - Expected gain: ~40% from eliminating sync overhead

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLSize,
};

use crate::async_scheduler::GpuCompletionToken;
use crate::buffer::{BufferUsage, MetalBuffer};
use crate::context::MetalContext;
use crate::error::{MetalError, Result};

// =============================================================================
// Batched Command Buffer
// =============================================================================

/// A command buffer that batches multiple kernel dispatches.
///
/// Instead of creating a new command buffer for each kernel and waiting
/// after each one, this struct accumulates dispatches and only waits
/// at the end.
///
/// # Usage
///
/// ```ignore
/// let mut batch = BatchedCommandBuffer::new(&ctx)?;
///
/// // Queue multiple operations (no waiting)
/// batch.queue_adamw_update(...)?;
/// batch.queue_gradient_clip(...)?;
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
        let command_buffer = ctx
            .command_queue()
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

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

    /// Get mutable encoder access (internal use).
    fn encoder_mut(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
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

    /// Execute without waiting and return a completion token.
    ///
    /// This provides a richer async experience with completion tracking.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut batch = BatchedCommandBuffer::new(&ctx)?;
    /// // ... queue operations ...
    ///
    /// let token = batch.execute_with_token()?;
    ///
    /// // Do other work while GPU executes
    /// prepare_next_batch();
    ///
    /// // Wait when results are needed
    /// token.wait();
    /// ```
    pub fn execute_with_token(mut self) -> Result<BatchCompletionToken> {
        static OPERATION_COUNTER: AtomicU64 = AtomicU64::new(0);

        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }

        let command_buffer = self
            .command_buffer
            .take()
            .ok_or(MetalError::CommandBufferCreation)?;

        let operation_id = OPERATION_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dispatch_count = self.dispatch_count;

        command_buffer.commit();

        tracing::trace!(
            "BatchedCommandBuffer async submitted {} dispatches (op_id={})",
            dispatch_count,
            operation_id
        );

        Ok(BatchCompletionToken {
            command_buffer,
            operation_id,
            dispatch_count,
        })
    }
}

/// Token for tracking async batched command buffer completion.
///
/// Provides methods to check and wait for GPU work completion.
/// Implements [`GpuCompletionToken`] for use in generic contexts.
pub struct BatchCompletionToken {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    operation_id: u64,
    dispatch_count: usize,
}

impl BatchCompletionToken {
    /// Get the number of dispatches in this batch.
    ///
    /// This is specific to BatchCompletionToken (not in the trait).
    #[inline]
    pub fn dispatch_count(&self) -> usize {
        self.dispatch_count
    }

    /// Wait and return an error if execution failed.
    ///
    /// Alias for `wait_checked()` for backwards compatibility.
    #[deprecated(since = "0.2.0", note = "Use wait_checked() instead")]
    pub fn wait_and_check(&self) -> Result<()> {
        self.wait_checked()
    }
}

impl GpuCompletionToken for BatchCompletionToken {
    fn wait(&self) {
        self.command_buffer.waitUntilCompleted();
    }

    fn wait_checked(&self) -> Result<()> {
        self.command_buffer.waitUntilCompleted();
        if let Some(err) = self.command_buffer.error() {
            return Err(MetalError::ExecutionFailed(err.to_string()));
        }
        Ok(())
    }

    #[inline]
    fn is_complete(&self) -> bool {
        let status = self.command_buffer.status();
        status == MTLCommandBufferStatus::Completed || status == MTLCommandBufferStatus::Error
    }

    fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        use std::time::Instant;

        const POLL_INTERVAL: Duration = Duration::from_micros(100);
        let deadline = Instant::now() + timeout;

        while !self.is_complete() {
            let now = Instant::now();
            if now >= deadline {
                return Ok(false);
            }
            // Sleep for poll interval or remaining time, whichever is smaller
            let remaining = deadline - now;
            std::thread::sleep(POLL_INTERVAL.min(remaining));
        }

        // Check for GPU errors after completion
        if let Some(err) = self.command_buffer.error() {
            return Err(MetalError::ExecutionFailed(err.to_string()));
        }
        Ok(true)
    }

    #[inline]
    fn operation_id(&self) -> u64 {
        self.operation_id
    }

    fn error(&self) -> Option<String> {
        self.command_buffer.error().map(|e| e.to_string())
    }
}

// SAFETY: BatchCompletionToken wraps Metal objects which are thread-safe.
unsafe impl Send for BatchCompletionToken {}
unsafe impl Sync for BatchCompletionToken {}

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

// =============================================================================
// AdamW Configuration
// =============================================================================

/// AdamW optimizer hyperparameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AdamWConfig {
    /// Learning rate (after scheduling).
    pub learning_rate: f32,
    /// First moment decay (default 0.9).
    pub beta1: f32,
    /// Second moment decay (default 0.999).
    pub beta2: f32,
    /// Numerical stability constant (default 1e-8).
    pub epsilon: f32,
    /// L2 regularization strength (default 0.01).
    pub weight_decay: f32,
    /// Current optimization step (for bias correction).
    pub step: u32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            step: 1,
        }
    }
}

/// Parameter metadata for batched processing.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ParamInfo {
    /// Offset into the flattened parameter buffer.
    pub offset: u32,
    /// Number of elements in this parameter.
    pub size: u32,
    /// Offset into first moment buffer.
    pub m_offset: u32,
    /// Offset into second moment buffer.
    pub v_offset: u32,
}

// =============================================================================
// Fused AdamW Optimizer
// =============================================================================

/// Fused AdamW optimizer that updates all parameters in a single dispatch.
///
/// Instead of launching N kernel dispatches for N parameters, this
/// processes all parameters in parallel within a single Metal dispatch.
///
/// # Memory Layout
///
/// Parameters, gradients, and optimizer state are stored in flattened buffers:
/// - `params`: All model parameters concatenated
/// - `grads`: All gradients concatenated (same layout as params)
/// - `m`: First moment estimates (same layout)
/// - `v`: Second moment estimates (same layout)
/// - `param_info`: Metadata describing offset and size of each parameter
pub struct FusedAdamW {
    ctx: Arc<MetalContext>,
    /// Total number of parameters.
    num_params: usize,
    /// Maximum parameter size (for grid calculation).
    max_param_size: usize,
    /// Total elements across all parameters.
    total_elements: usize,
}

impl FusedAdamW {
    /// Create a new fused AdamW optimizer.
    ///
    /// # Arguments
    /// * `ctx` - Metal context
    /// * `param_sizes` - Sizes of each parameter tensor
    pub fn new(ctx: Arc<MetalContext>, param_sizes: &[usize]) -> Self {
        let num_params = param_sizes.len();
        let max_param_size = param_sizes.iter().copied().max().unwrap_or(0);
        let total_elements: usize = param_sizes.iter().sum();

        Self {
            ctx,
            num_params,
            max_param_size,
            total_elements,
        }
    }

    /// Build parameter info metadata from sizes.
    pub fn build_param_info(param_sizes: &[usize]) -> Vec<ParamInfo> {
        let mut offset = 0u32;
        param_sizes
            .iter()
            .map(|&size| {
                let info = ParamInfo {
                    offset,
                    size: size as u32,
                    m_offset: offset,
                    v_offset: offset,
                };
                offset += size as u32;
                info
            })
            .collect()
    }

    /// Queue an AdamW update into a batched command buffer.
    ///
    /// This does NOT execute immediately - call `batch.execute()` after
    /// queueing all operations.
    pub fn queue_update(
        &self,
        batch: &mut BatchedCommandBuffer,
        params: &MetalBuffer<f32>,
        grads: &MetalBuffer<f32>,
        m: &MetalBuffer<f32>,
        v: &MetalBuffer<f32>,
        param_info: &MetalBuffer<ParamInfo>,
        config: &AdamWConfig,
    ) -> Result<()> {
        let function_name = "fused_adamw_update";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch.encoder_mut()?;
        encoder.setComputePipelineState(&pipeline);

        // Set buffers
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(params.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(grads.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(m.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(param_info.metal_buffer()), 0, 4);

            let config_ptr = NonNull::from(config).cast();
            encoder.setBytes_length_atIndex(config_ptr, std::mem::size_of::<AdamWConfig>(), 5);

            let num_params = self.num_params as u32;
            let num_params_ptr = NonNull::from(&num_params).cast();
            encoder.setBytes_length_atIndex(num_params_ptr, std::mem::size_of::<u32>(), 6);
        }

        // Grid: [ceil(max_param_size / 32), num_params, 1]
        let grid_size = MTLSize {
            width: (self.max_param_size + 31) / 32,
            height: self.num_params,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Execute AdamW update immediately (for compatibility).
    ///
    /// This creates a dedicated command buffer and waits for completion.
    /// Prefer using `queue_update` with `BatchedCommandBuffer` for better performance.
    pub fn execute_update(
        &self,
        params: &MetalBuffer<f32>,
        grads: &MetalBuffer<f32>,
        m: &MetalBuffer<f32>,
        v: &MetalBuffer<f32>,
        param_info: &MetalBuffer<ParamInfo>,
        config: &AdamWConfig,
    ) -> Result<()> {
        let mut batch = BatchedCommandBuffer::new(self.ctx.clone())?;
        self.queue_update(&mut batch, params, grads, m, v, param_info, config)?;
        batch.execute()
    }

    /// Get total number of elements.
    pub fn total_elements(&self) -> usize {
        self.total_elements
    }
}

// =============================================================================
// Fused Gradient Clipping
// =============================================================================

/// Fused gradient clipping that computes global norm and scales in two passes.
pub struct FusedGradientClipping {
    ctx: Arc<MetalContext>,
    /// Total elements in gradient buffer.
    total_elements: usize,
    /// Number of threadgroups for partial reduction.
    num_threadgroups: usize,
}

impl FusedGradientClipping {
    /// Create a new gradient clipper.
    pub fn new(ctx: Arc<MetalContext>, total_elements: usize) -> Self {
        // Each threadgroup processes 4 * 256 = 1024 elements
        let elements_per_tg = 1024;
        let num_threadgroups = (total_elements + elements_per_tg - 1) / elements_per_tg;

        Self {
            ctx,
            total_elements,
            num_threadgroups,
        }
    }

    /// Compute gradient norm squared (returns buffer with partial sums).
    pub fn compute_norm_squared_partial(
        &self,
        batch: &mut BatchedCommandBuffer,
        grads: &MetalBuffer<f32>,
        partial_sums: &MetalBuffer<f32>,
    ) -> Result<()> {
        let function_name = "gradient_norm_squared_partial";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch.encoder_mut()?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(grads.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(partial_sums.metal_buffer()), 0, 1);

            let total = self.total_elements as u32;
            let total_ptr = NonNull::from(&total).cast();
            encoder.setBytes_length_atIndex(total_ptr, std::mem::size_of::<u32>(), 2);
        }

        let grid_size = MTLSize {
            width: self.num_threadgroups,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Scale gradients by a factor.
    pub fn scale_gradients(
        &self,
        batch: &mut BatchedCommandBuffer,
        grads: &MetalBuffer<f32>,
        scale: f32,
    ) -> Result<()> {
        let function_name = "scale_gradients";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch.encoder_mut()?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(grads.metal_buffer()), 0, 0);

            let scale_ptr = NonNull::from(&scale).cast();
            encoder.setBytes_length_atIndex(scale_ptr, std::mem::size_of::<f32>(), 1);

            let total = self.total_elements as u32;
            let total_ptr = NonNull::from(&total).cast();
            encoder.setBytes_length_atIndex(total_ptr, std::mem::size_of::<u32>(), 2);
        }

        let grid_size = MTLSize {
            width: (self.total_elements + 31) / 32,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Get number of threadgroups for partial reduction.
    pub fn num_threadgroups(&self) -> usize {
        self.num_threadgroups
    }
}

// =============================================================================
// Fused Cross-Entropy
// =============================================================================

/// Fused cross-entropy loss with backward pass.
///
/// Computes both loss and gradients in a single kernel, avoiding
/// multiple passes over the logits tensor.
pub struct FusedCrossEntropyTraining {
    ctx: Arc<MetalContext>,
}

impl FusedCrossEntropyTraining {
    /// Create a new fused cross-entropy module.
    pub fn new(ctx: Arc<MetalContext>) -> Self {
        Self { ctx }
    }

    /// Queue fused forward + backward pass.
    ///
    /// # Arguments
    /// * `batch` - Command buffer to queue into
    /// * `logits` - Model outputs [N, vocab_size]
    /// * `labels` - Target labels [N]
    /// * `grad_logits` - Output gradients [N, vocab_size]
    /// * `loss` - Scalar loss (atomically accumulated)
    /// * `n` - Number of positions (batch * seq after shift)
    /// * `vocab_size` - Vocabulary size
    /// * `ignore_index` - Label to ignore (-100 typically)
    pub fn queue_forward_backward(
        &self,
        batch: &mut BatchedCommandBuffer,
        logits: &MetalBuffer<f32>,
        labels: &MetalBuffer<i32>,
        grad_logits: &MetalBuffer<f32>,
        loss: &MetalBuffer<f32>,
        n: usize,
        vocab_size: usize,
        ignore_index: i32,
    ) -> Result<()> {
        let function_name = "fused_cross_entropy_forward_backward";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch.encoder_mut()?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(labels.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(grad_logits.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(loss.metal_buffer()), 0, 3);

            let n_u32 = n as u32;
            let n_ptr = NonNull::from(&n_u32).cast();
            encoder.setBytes_length_atIndex(n_ptr, std::mem::size_of::<u32>(), 4);

            let vocab_u32 = vocab_size as u32;
            let vocab_ptr = NonNull::from(&vocab_u32).cast();
            encoder.setBytes_length_atIndex(vocab_ptr, std::mem::size_of::<u32>(), 5);

            let ignore_ptr = NonNull::from(&ignore_index).cast();
            encoder.setBytes_length_atIndex(ignore_ptr, std::mem::size_of::<i32>(), 6);
        }

        // One threadgroup per position
        let grid_size = MTLSize {
            width: n,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }
}

// =============================================================================
// Training Step Coordinator
// =============================================================================

/// Coordinates a complete training step using batched Metal execution.
///
/// This struct manages the command buffer batching for an entire training step:
/// 1. Forward pass (via MLX or custom kernels)
/// 2. Loss computation (fused cross-entropy)
/// 3. Backward pass (via MLX autodiff or custom kernels)
/// 4. Gradient clipping (fused)
/// 5. Optimizer update (fused AdamW)
///
/// The key insight is that steps 2, 4, and 5 can be batched into a single
/// command buffer, eliminating 3+ GPU-CPU synchronization points.
pub struct FusedTrainingCoordinator {
    ctx: Arc<MetalContext>,
    adamw: FusedAdamW,
    grad_clip: Option<FusedGradientClipping>,
    cross_entropy: FusedCrossEntropyTraining,
}

impl FusedTrainingCoordinator {
    /// Create a new training coordinator.
    ///
    /// # Arguments
    /// * `ctx` - Metal context
    /// * `param_sizes` - Sizes of each parameter tensor
    /// * `max_grad_norm` - Optional gradient clipping threshold
    pub fn new(
        ctx: Arc<MetalContext>,
        param_sizes: &[usize],
        max_grad_norm: Option<f32>,
    ) -> Self {
        let total_elements: usize = param_sizes.iter().sum();

        let adamw = FusedAdamW::new(ctx.clone(), param_sizes);

        let grad_clip = max_grad_norm.map(|_| FusedGradientClipping::new(ctx.clone(), total_elements));

        let cross_entropy = FusedCrossEntropyTraining::new(ctx.clone());

        Self {
            ctx,
            adamw,
            grad_clip,
            cross_entropy,
        }
    }

    /// Get the context.
    pub fn context(&self) -> &Arc<MetalContext> {
        &self.ctx
    }

    /// Get the fused AdamW optimizer.
    pub fn adamw(&self) -> &FusedAdamW {
        &self.adamw
    }

    /// Get the gradient clipper (if configured).
    pub fn grad_clipper(&self) -> Option<&FusedGradientClipping> {
        self.grad_clip.as_ref()
    }

    /// Get the fused cross-entropy module.
    pub fn cross_entropy(&self) -> &FusedCrossEntropyTraining {
        &self.cross_entropy
    }

    /// Start a new batched training step.
    pub fn begin_step(&self) -> Result<BatchedCommandBuffer> {
        BatchedCommandBuffer::new(self.ctx.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> Arc<MetalContext> {
        Arc::new(MetalContext::new().expect("Failed to create Metal context"))
    }

    #[test]
    fn test_adamw_config_default() {
        let config = AdamWConfig::default();
        assert!((config.beta1 - 0.9).abs() < 1e-6);
        assert!((config.beta2 - 0.999).abs() < 1e-6);
        assert!((config.epsilon - 1e-8).abs() < 1e-12);
    }

    #[test]
    fn test_param_info_building() {
        let sizes = vec![100, 200, 50];
        let info = FusedAdamW::build_param_info(&sizes);

        assert_eq!(info.len(), 3);
        assert_eq!(info[0].offset, 0);
        assert_eq!(info[0].size, 100);
        assert_eq!(info[1].offset, 100);
        assert_eq!(info[1].size, 200);
        assert_eq!(info[2].offset, 300);
        assert_eq!(info[2].size, 50);
    }

    #[test]
    fn test_fused_adamw_creation() {
        let ctx = create_test_context();
        let param_sizes = vec![1024, 2048, 512];
        let adamw = FusedAdamW::new(ctx, &param_sizes);

        assert_eq!(adamw.num_params, 3);
        assert_eq!(adamw.max_param_size, 2048);
        assert_eq!(adamw.total_elements, 1024 + 2048 + 512);
    }

    #[test]
    fn test_batched_command_buffer_creation() {
        let ctx = create_test_context();
        let batch = BatchedCommandBuffer::new(ctx);
        assert!(batch.is_ok());

        let batch = batch.unwrap();
        assert_eq!(batch.dispatch_count(), 0);
        assert!(batch.encoder().is_some());
    }

    #[test]
    fn test_gradient_clipping_creation() {
        let ctx = create_test_context();
        let clipper = FusedGradientClipping::new(ctx, 10000);

        // 10000 elements / 1024 per threadgroup = 10 threadgroups
        assert_eq!(clipper.num_threadgroups(), 10);
    }

    #[test]
    fn test_fused_adamw_execution() {
        use crate::buffer::BufferUsage;

        let ctx = create_test_context();

        // Simulate 3 parameters with sizes matching typical LoRA
        let param_sizes = vec![1024, 2048, 512];
        let total_elements: usize = param_sizes.iter().sum();

        let adamw = FusedAdamW::new(ctx.clone(), &param_sizes);

        // Create buffers
        // Initialize params to 1.0, grads to 0.1
        let params_data: Vec<f32> = vec![1.0; total_elements];
        let grads_data: Vec<f32> = vec![0.1; total_elements];

        let params = MetalBuffer::from_slice(&ctx, &params_data, BufferUsage::Shared)
            .expect("Failed to create params buffer");
        let grads = MetalBuffer::from_slice(&ctx, &grads_data, BufferUsage::Shared)
            .expect("Failed to create grads buffer");
        let m = MetalBuffer::zeros(&ctx, total_elements, BufferUsage::Shared)
            .expect("Failed to create m buffer");
        let v = MetalBuffer::zeros(&ctx, total_elements, BufferUsage::Shared)
            .expect("Failed to create v buffer");

        // Build param info
        let param_info_vec = FusedAdamW::build_param_info(&param_sizes);
        let param_info = MetalBuffer::from_slice(&ctx, &param_info_vec, BufferUsage::Shared)
            .expect("Failed to create param_info buffer");

        // Execute AdamW update
        let config = AdamWConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            step: 1,
        };

        adamw
            .execute_update(&params, &grads, &m, &v, &param_info, &config)
            .expect("AdamW update failed");

        // Verify params were updated
        let updated_params = params.as_slice();

        // After one step with grad=0.1, lr=0.001:
        // m = 0.1 * 0.1 = 0.01
        // v = 0.001 * 0.01 = 0.00001
        // m_hat = 0.01 / 0.1 = 0.1
        // v_hat = 0.00001 / 0.001 = 0.01
        // update = 0.1 / (sqrt(0.01) + 1e-8) = 0.1 / 0.1 = 1.0
        // param = 1.0 * (1 - 0.001 * 0.01) - 0.001 * 1.0 = 0.99999 - 0.001 ≈ 0.999
        //
        // The exact value depends on bias correction, but params should be < 1.0
        assert!(
            updated_params[0] < 1.0,
            "Params should decrease after AdamW update, got {}",
            updated_params[0]
        );
        assert!(
            updated_params[0] > 0.9,
            "Params should not decrease too much, got {}",
            updated_params[0]
        );
    }

    #[test]
    fn test_fused_adamw_batched() {
        use crate::buffer::BufferUsage;
        use std::time::Instant;

        let ctx = create_test_context();

        // Larger test for performance measurement
        // 10M parameters (typical LoRA model)
        let param_sizes = vec![1_000_000; 10]; // 10 params of 1M each
        let total_elements: usize = param_sizes.iter().sum();

        let adamw = FusedAdamW::new(ctx.clone(), &param_sizes);

        // Create buffers
        let params = MetalBuffer::zeros(&ctx, total_elements, BufferUsage::Shared)
            .expect("Failed to create params buffer");
        let grads = MetalBuffer::zeros(&ctx, total_elements, BufferUsage::Shared)
            .expect("Failed to create grads buffer");
        let m = MetalBuffer::zeros(&ctx, total_elements, BufferUsage::Shared)
            .expect("Failed to create m buffer");
        let v = MetalBuffer::zeros(&ctx, total_elements, BufferUsage::Shared)
            .expect("Failed to create v buffer");

        // Build param info
        let param_info_vec = FusedAdamW::build_param_info(&param_sizes);
        let param_info = MetalBuffer::from_slice(&ctx, &param_info_vec, BufferUsage::Shared)
            .expect("Failed to create param_info buffer");

        let config = AdamWConfig {
            learning_rate: 0.001,
            ..Default::default()
        };

        // Warmup
        adamw
            .execute_update(&params, &grads, &m, &v, &param_info, &config)
            .expect("Warmup failed");

        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let mut batch = BatchedCommandBuffer::new(ctx.clone()).unwrap();
            adamw
                .queue_update(&mut batch, &params, &grads, &m, &v, &param_info, &config)
                .unwrap();
            batch.execute().unwrap();
        }
        let elapsed = start.elapsed();

        let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let params_per_sec = (total_elements as f64 / ms_per_iter) * 1000.0;

        println!(
            "Fused AdamW: {:.2}ms per iter, {:.0} params/sec ({} total params)",
            ms_per_iter,
            params_per_sec,
            total_elements
        );

        // Should be fast - less than 1ms for 10M params
        assert!(
            ms_per_iter < 5.0,
            "AdamW update too slow: {:.2}ms",
            ms_per_iter
        );
    }
}
