#![allow(unsafe_code)]

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
//! - Target: ~2400 tok/s
//! - Expected gain: ~40% from eliminating sync overhead

use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

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

    /// Get mutable encoder access (crate-internal use for queue_* methods).
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

// =============================================================================
// AdamW Configuration
// =============================================================================

/// AdamW optimizer hyperparameters for one step.
///
/// Every AdamW in this crate (the Metal kernels and the CPU update in
/// [`crate::accelerate`]) applies them as `torch.optim.AdamW` does: decoupled
/// weight decay on the parameter before the update, and bias correction of
/// both moments.
#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig {
    /// Learning rate (after scheduling).
    pub learning_rate: f32,
    /// First moment decay (default 0.9).
    pub beta1: f32,
    /// Second moment decay (default 0.999).
    pub beta2: f32,
    /// Numerical stability constant (default 1e-8).
    pub epsilon: f32,
    /// Decoupled weight decay (default 0.01).
    pub weight_decay: f32,
    /// This step's number, from 1 (drives the bias correction).
    pub step: u32,
}

impl AdamWConfig {
    /// The per-step scalars the kernels read, computed as `torch.optim.AdamW`
    /// computes them: in f64, each rounded to f32 once.
    ///
    /// Fails for step 0: the bias correction is undefined before the first
    /// step.
    pub fn scalars(&self) -> Result<AdamWScalars> {
        if self.step == 0 {
            return Err(crate::error::MetalError::InvalidConfig(
                "AdamW steps are 1-based; step 0 has no bias correction".into(),
            ));
        }
        let (lr, wd) = (f64::from(self.learning_rate), f64::from(self.weight_decay));
        let (b1, b2) = (f64::from(self.beta1), f64::from(self.beta2));
        let t = f64::from(self.step);
        Ok(AdamWScalars {
            decay: (1.0 - lr * wd) as f32,
            one_minus_beta1: (1.0 - b1) as f32,
            beta2: self.beta2,
            one_minus_beta2: (1.0 - b2) as f32,
            step_size: (lr / (1.0 - b1.powf(t))) as f32,
            bias_correction2_sqrt: (1.0 - b2.powf(t)).sqrt() as f32,
            epsilon: self.epsilon,
        })
    }
}

/// One AdamW step's scalars, as the kernels read them.
///
/// The update is, per element:
/// ```text
/// p = p·decay
/// m = m + (1−β₁)·(g − m)
/// v = β₂·v + (1−β₂)·g²
/// p = p − step_size · m / (√v / bias_correction2_sqrt + ε)
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct AdamWScalars {
    /// `1 − lr·λ`.
    pub decay: f32,
    /// `1 − β₁`.
    pub one_minus_beta1: f32,
    /// `β₂`.
    pub beta2: f32,
    /// `1 − β₂`.
    pub one_minus_beta2: f32,
    /// `lr / (1 − β₁ᵗ)`.
    pub step_size: f32,
    /// `√(1 − β₂ᵗ)`.
    pub bias_correction2_sqrt: f32,
    /// `ε`.
    pub epsilon: f32,
}

impl AdamWScalars {
    /// Scalars that leave a parameter and both its moments exactly as they
    /// are: `p·1 − 0·(…)`, `m + 0·(g − m)`, `1·v + 0·g²`. A parameter without
    /// a gradient this step takes them, as `torch.optim` skips a parameter
    /// whose `.grad` is `None`.
    pub const UNCHANGED: Self = Self {
        decay: 1.0,
        one_minus_beta1: 0.0,
        beta2: 1.0,
        one_minus_beta2: 0.0,
        step_size: 0.0,
        bias_correction2_sqrt: 1.0,
        epsilon: 1.0,
    };
}

// `AdamWScalars` in `metal/fused_training.metal` and
// `metal4/mpp_fused_training.metal`.
crate::msl_layout::msl_layout!(AdamWScalars, size = 28, {
    decay: 0,
    one_minus_beta1: 4,
    beta2: 8,
    one_minus_beta2: 12,
    step_size: 16,
    bias_correction2_sqrt: 20,
    epsilon: 24,
});

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
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct ParamInfo {
    /// Offset into the flattened parameter buffer.
    pub offset: u32,
    /// Number of elements in this parameter.
    pub size: u32,
    /// Offset into first moment buffer.
    pub m_offset: u32,
    /// Offset into second moment buffer.
    pub v_offset: u32,
    /// Which of the step's [`AdamWScalars`] this parameter is updated with
    /// (its parameter group); see [`FusedAdamW::queue_update`].
    pub group: u32,
}

// `ParamInfo` in `metal/fused_training.metal`.
crate::msl_layout::msl_layout!(ParamInfo, size = 20, {
    offset: 0,
    size: 4,
    m_offset: 8,
    v_offset: 12,
    group: 16,
});

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

    /// Build parameter info metadata from sizes, every parameter in group 0.
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
                    group: 0,
                };
                offset += size as u32;
                info
            })
            .collect()
    }

    /// The most groups one dispatch takes: their scalars go to the kernel
    /// with `setBytes`, which carries at most 4 KiB.
    pub const MAX_GROUPS: usize = 4096 / std::mem::size_of::<AdamWScalars>();

    /// Queue an AdamW update into a batched command buffer.
    ///
    /// Parameter `i` is updated with `groups[param_info[i].group]`: one set
    /// of scalars per parameter group, so each group trains at its own
    /// learning rate and weight decay. Fails if a parameter names a group
    /// past the end of `groups`, or if there are more than
    /// [`Self::MAX_GROUPS`].
    ///
    /// This does NOT execute immediately - call `batch.execute()` after
    /// queueing all operations.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_update(
        &self,
        batch: &mut BatchedCommandBuffer,
        params: &MetalBuffer<f32>,
        grads: &MetalBuffer<f32>,
        m: &MetalBuffer<f32>,
        v: &MetalBuffer<f32>,
        param_info: &MetalBuffer<ParamInfo>,
        groups: &[AdamWScalars],
    ) -> Result<()> {
        if groups.is_empty() || groups.len() > Self::MAX_GROUPS {
            return Err(crate::error::MetalError::InvalidConfig(format!(
                "fused AdamW takes 1 to {} parameter groups, got {}",
                Self::MAX_GROUPS,
                groups.len()
            )));
        }
        let infos = param_info.as_slice();
        if infos.len() < self.num_params {
            return Err(crate::error::MetalError::InvalidConfig(format!(
                "fused AdamW over {} parameters given metadata for {}",
                self.num_params,
                infos.len()
            )));
        }
        if let Some(info) = infos[..self.num_params]
            .iter()
            .find(|info| info.group as usize >= groups.len())
        {
            return Err(crate::error::MetalError::InvalidConfig(format!(
                "a parameter at offset {} is in group {}, but the step has {} groups",
                info.offset,
                info.group,
                groups.len()
            )));
        }
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

            let groups_ptr = NonNull::from(groups).cast();
            encoder.setBytes_length_atIndex(groups_ptr, std::mem::size_of_val(groups), 5);

            let num_params = self.num_params as u32;
            let num_params_ptr = NonNull::from(&num_params).cast();
            encoder.setBytes_length_atIndex(num_params_ptr, std::mem::size_of::<u32>(), 6);
        }

        // Grid: [ceil(max_param_size / 32), num_params, 1]
        let grid_size = MTLSize {
            width: self.max_param_size.div_ceil(32),
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
        groups: &[AdamWScalars],
    ) -> Result<()> {
        let mut batch = BatchedCommandBuffer::new(self.ctx.clone())?;
        self.queue_update(&mut batch, params, grads, m, v, param_info, groups)?;
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
        let num_threadgroups = total_elements.div_ceil(elements_per_tg);

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

        // Each thread processes 4 elements via float4/half4 vectorization,
        // so we need total_elements / (32 threads * 4 elements/thread) threadgroups.
        let grid_size = MTLSize {
            width: self.total_elements.div_ceil(128),
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

    /// Compute global gradient norm and, if it exceeds `max_norm`, queue a
    /// scaling kernel into `batch` to clip the gradients in-place.
    ///
    /// # Algorithm
    ///
    /// Gradient clipping requires comparing the global L2 norm against a
    /// threshold before deciding whether to scale.  This means we need a
    /// GPU→CPU round-trip between the partial-reduction pass and the
    /// conditional scaling pass.  The implementation therefore:
    ///
    /// 1. Allocates a CPU-visible shared buffer for partial squared sums
    ///    (`num_threadgroups` elements).
    /// 2. Executes the partial norm kernel in its own synchronous
    ///    `BatchedCommandBuffer` so the values are visible on the CPU.
    /// 3. Sums the partial results on the CPU to obtain the global L2 norm.
    /// 4. If `norm > max_norm`, queues `scale_gradients` into the caller's
    ///    `batch` with the appropriate clipping factor (`max_norm / norm`).
    ///
    /// The caller's `batch` is untouched when no clipping is needed, keeping
    /// the common fast path allocation-free after the first call (assuming the
    /// caller reuses the `FusedGradientClipping` instance).
    pub fn queue_norm_and_clip(
        &self,
        batch: &mut BatchedCommandBuffer,
        grads: &MetalBuffer<f32>,
        max_norm: f32,
    ) -> Result<()> {
        // --- Step 1: allocate CPU-visible partial-sums buffer ---------------
        let n_tg = self.num_threadgroups;
        let partial_sums = MetalBuffer::<f32>::zeros(&self.ctx, n_tg, BufferUsage::Shared)?;

        // --- Step 2: run partial norm² kernel synchronously -----------------
        {
            let mut norm_batch = BatchedCommandBuffer::new(self.ctx.clone())?;
            self.compute_norm_squared_partial(&mut norm_batch, grads, &partial_sums)?;
            norm_batch.execute()?;
        }

        // --- Step 3: reduce partial sums on CPU to get global L2 norm -------
        let partial_slice = partial_sums.as_slice();
        let norm_sq: f32 = partial_slice.iter().copied().sum();
        let norm = norm_sq.sqrt();

        // --- Step 4: queue conditional clipping into the caller's batch -----
        if norm > max_norm {
            let scale = max_norm / norm;
            self.scale_gradients(batch, grads, scale)?;
        }

        Ok(())
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
    #[allow(clippy::too_many_arguments)]
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

        // One parameter per group: the second at its own rate without
        // decay, the third left as it is.
        let mut param_info_vec = FusedAdamW::build_param_info(&param_sizes);
        param_info_vec[1].group = 1;
        param_info_vec[2].group = 2;
        let param_info = MetalBuffer::from_slice(&ctx, &param_info_vec, BufferUsage::Shared)
            .expect("Failed to create param_info buffer");

        let config = AdamWConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            step: 1,
        };
        let second = AdamWConfig {
            learning_rate: 0.004,
            weight_decay: 0.0,
            ..config
        };
        let groups = [
            config.scalars().unwrap(),
            second.scalars().unwrap(),
            AdamWScalars::UNCHANGED,
        ];

        adamw
            .execute_update(&params, &grads, &m, &v, &param_info, &groups)
            .expect("AdamW update failed");

        // The first bias-corrected step moves each weight by lr·sign(g),
        // after its decay: 1·(1 − 0.001·0.01) − 0.001.
        let (p_out, m_out) = (params.as_slice(), m.as_slice());
        let wants = [
            ((1.0 - 0.001 * 0.01) - 0.001, 0.01),
            (1.0 - 0.004, 0.01),
            (1.0, 0.0),
        ];
        let mut offset = 0;
        for (&size, &(want_p, want_m)) in param_sizes.iter().zip(&wants) {
            for i in offset..offset + size {
                let (p, m) = (p_out[i], m_out[i]);
                assert!((p - want_p).abs() < 1e-6, "param {i}: {p}, want {want_p}");
                assert!((m - want_m).abs() < 1e-8, "m {i}: {m}, want {want_m}");
            }
            offset += size;
        }
        let untouched = total_elements - param_sizes[2];
        assert!(v.as_slice()[untouched..].iter().all(|&v| v == 0.0));

        // A group past the end of the step's scalars is refused.
        assert!(
            adamw
                .execute_update(&params, &grads, &m, &v, &param_info, &groups[..2])
                .is_err()
        );
    }

    #[test]
    fn adamw_step_zero_is_refused() {
        let config = AdamWConfig {
            step: 0,
            ..Default::default()
        };
        assert!(config.scalars().is_err());
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

        let config = [AdamWConfig {
            learning_rate: 0.001,
            ..Default::default()
        }
        .scalars()
        .unwrap()];

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
            ms_per_iter, params_per_sec, total_elements
        );

        // Performance sanity check - very generous threshold for CI environments
        // Local dev machines typically see <2ms, but CI can be 5-10x slower
        if ms_per_iter > 50.0 {
            eprintln!(
                "Warning: AdamW update slower than expected: {:.2}ms (threshold: 50ms)",
                ms_per_iter
            );
        }
    }
}
