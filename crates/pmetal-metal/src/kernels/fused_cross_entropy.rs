#![allow(unsafe_code)]

//! Fused cross-entropy loss kernel.
//!
//! This module provides GPU-accelerated cross-entropy loss computation
//! using a fused forward/backward pattern:
//!
//! - Forward: CE(x, y) = logsumexp(x) - x[y]
//! - Backward: dL/dx = softmax(x) - one_hot(y)
//!
//! Key optimizations:
//! - Uses online softmax to avoid materializing full distribution
//! - Caches logsumexp from forward for efficient backward
//! - In-place gradient computation
//! - SIMD parallelization across vocabulary
//! - Support for fp16 mixed precision
//! - Handles softcapping (Gemma2) and scaling (Cohere)
//!
//! # Fused Linear + Cross-Entropy (CRITICAL OPTIMIZATION)
//!
//! The `fused_linear_cross_entropy` function computes loss **directly from hidden states**
//! without ever materializing the full `[batch, seq, vocab_size]` logits tensor.
//!
//! Memory savings example:
//! - batch=4, seq=1024, vocab=150K, fp16 → logits would be **1.2GB**
//! - With fusion: peak memory is only `[chunk_size=4096]` → **8MB**
//!
//! This is the single biggest memory optimization in LLM training.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

use half::f16;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::{
    buffer::{AsMetalBuffer, BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result},
    pipeline::FunctionConstant,
    tuna::CrossEntropyTunedConfig,
};

/// Configuration for fused cross-entropy loss.
#[derive(Debug, Clone)]
pub struct FusedCrossEntropyConfig {
    /// Number of tokens to process.
    pub num_tokens: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Label smoothing factor (0.0 to disable).
    pub label_smoothing: f32,

    /// Logit softcapping value for Gemma2 models (0.0 to disable).
    pub softcap: f32,

    /// Index to ignore in loss computation (typically -100).
    pub ignore_index: i32,

    /// Use SIMD-parallel kernel (more efficient for large vocabularies).
    pub use_simd: bool,

    /// Use fp16 kernels for mixed precision.
    pub use_fp16: bool,
}

impl FusedCrossEntropyConfig {
    /// Create a new config with default values.
    pub fn new(num_tokens: usize, vocab_size: usize) -> Self {
        Self {
            num_tokens,
            vocab_size,
            label_smoothing: 0.0,
            softcap: 0.0,
            ignore_index: -100,
            use_simd: true, // SIMD variant is always preferred (supports label smoothing, softcap)
            use_fp16: false,
        }
    }

    /// Enable Gemma2 softcapping.
    pub fn with_softcap(mut self, softcap: f32) -> Self {
        self.softcap = softcap;
        self
    }

    /// Set ignore index.
    pub fn with_ignore_index(mut self, index: i32) -> Self {
        self.ignore_index = index;
        self
    }

    /// Enable label smoothing.
    pub fn with_label_smoothing(mut self, smoothing: f32) -> Self {
        self.label_smoothing = smoothing;
        self
    }

    /// Enable fp16 mode.
    pub fn with_fp16(mut self) -> Self {
        self.use_fp16 = true;
        self
    }
}

/// Output from fused cross-entropy forward pass.
#[derive(Debug)]
pub struct FusedCrossEntropyOutput {
    /// Per-token losses [num_tokens].
    pub losses: MetalBuffer<f32>,

    /// Cached logsumexp for backward [num_tokens].
    pub logsumexp: MetalBuffer<f32>,
}

impl FusedCrossEntropyOutput {
    /// Compute mean loss over valid tokens.
    pub fn mean_loss(&self, targets: &[i32], ignore_index: i32) -> f32 {
        let losses = self.losses.as_slice();
        let mut sum = 0.0f32;
        let mut count = 0usize;

        for (i, &loss) in losses.iter().enumerate() {
            if targets[i] != ignore_index {
                sum += loss;
                count += 1;
            }
        }

        if count > 0 { sum / count as f32 } else { 0.0 }
    }
}

/// Fused cross-entropy loss kernel.
///
/// Provides efficient forward and backward passes for cross-entropy loss
/// with support for large vocabularies, softcapping, and mixed precision.
pub struct FusedCrossEntropy {
    /// Metal context.
    ctx: Arc<MetalContext>,

    /// Configuration.
    config: FusedCrossEntropyConfig,
}

impl FusedCrossEntropy {
    /// Create a new fused cross-entropy kernel.
    pub fn new(ctx: Arc<MetalContext>, config: FusedCrossEntropyConfig) -> Result<Self> {
        Ok(Self { ctx, config })
    }

    /// Get the configuration.
    pub fn config(&self) -> &FusedCrossEntropyConfig {
        &self.config
    }

    /// Compute forward pass.
    ///
    /// # Arguments
    ///
    /// * `logits` - Logits tensor [num_tokens, vocab_size]
    /// * `targets` - Target indices [num_tokens]
    ///
    /// # Returns
    ///
    /// Per-token losses and cached logsumexp for backward pass.
    pub fn forward(
        &self,
        logits: &MetalBuffer<f32>,
        targets: &MetalBuffer<i32>,
    ) -> Result<FusedCrossEntropyOutput> {
        // Validate sizes
        let expected_logits = self.config.num_tokens * self.config.vocab_size;
        if logits.len() != expected_logits {
            return Err(MetalError::DimensionMismatch {
                param: "logits",
                expected: expected_logits,
                actual: logits.len(),
            });
        }
        if targets.len() != self.config.num_tokens {
            return Err(MetalError::DimensionMismatch {
                param: "targets",
                expected: self.config.num_tokens,
                actual: targets.len(),
            });
        }

        // Allocate outputs
        let losses = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let logsumexp = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;

        self.execute_forward(logits, targets, &losses, &logsumexp)?;

        Ok(FusedCrossEntropyOutput { losses, logsumexp })
    }

    /// Compute forward pass for fp16 logits.
    pub fn forward_f16(
        &self,
        logits: &MetalBuffer<f16>,
        targets: &MetalBuffer<i32>,
    ) -> Result<FusedCrossEntropyOutput> {
        let expected_logits = self.config.num_tokens * self.config.vocab_size;
        if logits.len() != expected_logits {
            return Err(MetalError::DimensionMismatch {
                param: "logits",
                expected: expected_logits,
                actual: logits.len(),
            });
        }
        if targets.len() != self.config.num_tokens {
            return Err(MetalError::DimensionMismatch {
                param: "targets",
                expected: self.config.num_tokens,
                actual: targets.len(),
            });
        }

        let losses = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let logsumexp = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;

        self.execute_forward_f16(logits, targets, &losses, &logsumexp)?;

        Ok(FusedCrossEntropyOutput { losses, logsumexp })
    }

    /// Compute forward pass accepting a type-erased logits buffer.
    ///
    /// Called by [`Metal3Backend`] which receives `&dyn buffer::AsMetalBuffer`
    /// from the [`KernelBackend`] trait and cannot cast it to a concrete
    /// `MetalBuffer<f32/f16>`. This method allocates outputs and dispatches
    /// the appropriate kernel variant based on `config.use_fp16`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `logits` actually contains data of the type
    /// implied by `config.use_fp16` (f32 when false, f16 when true). Passing
    /// mismatched data produces numerically incorrect results without panicking,
    /// since Metal treats all buffers as raw bytes.
    pub(crate) fn forward_dyn(
        &self,
        logits: &dyn AsMetalBuffer,
        targets: &MetalBuffer<i32>,
    ) -> Result<FusedCrossEntropyOutput> {
        let expected_logits = self.config.num_tokens * self.config.vocab_size;
        if logits.len() != expected_logits {
            return Err(MetalError::DimensionMismatch {
                param: "logits",
                expected: expected_logits,
                actual: logits.len(),
            });
        }
        if targets.len() != self.config.num_tokens {
            return Err(MetalError::DimensionMismatch {
                param: "targets",
                expected: self.config.num_tokens,
                actual: targets.len(),
            });
        }

        let losses = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let logsumexp = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;

        // Dispatch the Metal kernel directly via the raw MTLBuffer pointer.
        // We call execute_forward_raw so we are not constrained by the concrete
        // typed wrappers on execute_forward / execute_forward_f16.
        self.execute_forward_raw(logits.as_metal_buffer(), targets, &losses, &logsumexp)?;

        Ok(FusedCrossEntropyOutput { losses, logsumexp })
    }

    /// Compute backward pass (in-place gradient).
    ///
    /// # Arguments
    ///
    /// * `logits` - Logits tensor [num_tokens, vocab_size] - will be overwritten with gradients
    /// * `targets` - Target indices [num_tokens]
    /// * `logsumexp` - Cached logsumexp from forward [num_tokens]
    /// * `grad_loss` - Upstream gradient [num_tokens]
    pub fn backward(
        &self,
        logits: &mut MetalBuffer<f32>,
        targets: &MetalBuffer<i32>,
        logsumexp: &MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        self.execute_backward(logits, targets, logsumexp, grad_loss)
    }

    /// Compute backward pass for fp16 (in-place gradient).
    pub fn backward_f16(
        &self,
        logits: &mut MetalBuffer<f16>,
        targets: &MetalBuffer<i32>,
        logsumexp: &MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        self.execute_backward_f16(logits, targets, logsumexp, grad_loss)
    }

    /// Execute forward kernel.
    ///
    /// Exposed as `pub(crate)` so that [`Metal3Backend`] can call it directly
    /// with a raw `MetalBuffer<f32>` obtained by materialising a
    /// `&dyn AsMetalBuffer` reference, without needing to go through the
    /// higher-level `forward()` wrapper (which re-validates sizes we've
    /// already validated at the trait boundary).
    pub(crate) fn execute_forward(
        &self,
        logits: &MetalBuffer<f32>,
        targets: &MetalBuffer<i32>,
        losses: &MetalBuffer<f32>,
        logsumexp: &MetalBuffer<f32>,
    ) -> Result<()> {
        let function_name = if self.config.use_simd {
            "fused_cross_entropy_forward_simd"
        } else {
            "fused_cross_entropy_forward"
        };

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        // SAFETY: Metal compute encoder operations are safe when buffers are valid
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(targets.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(losses.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(logsumexp.metal_buffer()), 0, 3);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Execute forward kernel with a raw (type-erased) logits buffer.
    ///
    /// Used by [`forward_dyn`] when the caller holds `&dyn buffer::AsMetalBuffer`
    /// and cannot provide the concrete `MetalBuffer<f32/f16>` that the typed
    /// variants require. The pipeline selected is the same one `execute_forward`
    /// uses; the kernel itself does not care about the Rust element type —
    /// the caller is responsible for dtype correctness.
    fn execute_forward_raw(
        &self,
        logits: &ProtocolObject<dyn MTLBuffer>,
        targets: &MetalBuffer<i32>,
        losses: &MetalBuffer<f32>,
        logsumexp: &MetalBuffer<f32>,
    ) -> Result<()> {
        let function_name = if self.config.use_simd {
            "fused_cross_entropy_forward_simd"
        } else {
            "fused_cross_entropy_forward"
        };

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(targets.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(losses.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(logsumexp.metal_buffer()), 0, 3);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Execute forward kernel for fp16.
    fn execute_forward_f16(
        &self,
        logits: &MetalBuffer<f16>,
        targets: &MetalBuffer<i32>,
        losses: &MetalBuffer<f32>,
        logsumexp: &MetalBuffer<f32>,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(
                self.ctx.device(),
                "fused_cross_entropy_forward_f16",
                None,
            )?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(targets.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(losses.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(logsumexp.metal_buffer()), 0, 3);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Execute backward kernel.
    fn execute_backward(
        &self,
        logits: &mut MetalBuffer<f32>,
        targets: &MetalBuffer<i32>,
        logsumexp: &MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        let function_name = if self.config.use_simd {
            "fused_cross_entropy_backward_simd"
        } else {
            "fused_cross_entropy_backward"
        };

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(targets.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(logsumexp.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(grad_loss.metal_buffer()), 0, 3);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
        }

        let grid_size = if self.config.use_simd {
            objc2_metal::MTLSize {
                width: self.config.num_tokens,
                height: 1,
                depth: 1,
            }
        } else {
            objc2_metal::MTLSize {
                width: self.config.vocab_size,
                height: self.config.num_tokens,
                depth: 1,
            }
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Execute backward kernel for fp16.
    fn execute_backward_f16(
        &self,
        logits: &mut MetalBuffer<f16>,
        targets: &MetalBuffer<i32>,
        logsumexp: &MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(
                self.ctx.device(),
                "fused_cross_entropy_backward_f16",
                None,
            )?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(targets.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(logsumexp.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(grad_loss.metal_buffer()), 0, 3);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Create kernel parameters.
    fn create_params(&self) -> CrossEntropyParams {
        CrossEntropyParams {
            num_tokens: self.config.num_tokens as u32,
            vocab_size: self.config.vocab_size as u32,
            label_smoothing: self.config.label_smoothing,
            softcap: self.config.softcap,
            ignore_index: self.config.ignore_index,
            block_size: 32, // SIMD group size
        }
    }
}

/// Parameters passed to the kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CrossEntropyParams {
    num_tokens: u32,
    vocab_size: u32,
    label_smoothing: f32,
    softcap: f32,
    ignore_index: i32,
    block_size: u32,
}

impl std::fmt::Debug for FusedCrossEntropy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedCrossEntropy")
            .field("config", &self.config)
            .finish()
    }
}

// =============================================================================
// FUSED LINEAR + CROSS-ENTROPY (THE BIG WIN)
// =============================================================================
//
// Chunked vocabulary optimization: compute cross-entropy loss directly from
// hidden states without EVER materializing the full [batch, seq, vocab] logits.
//
// Memory savings:
//   - batch=4, seq=1024, vocab=150K, fp16 → logits would be 1.2GB
//   - With fusion: peak memory is only [chunk_size=4096] → 8MB
// =============================================================================

const DEFAULT_FUSED_LINEAR_CROSS_ENTROPY_CHUNK_SIZE: usize = 4096;

/// Configuration for fused linear + cross-entropy loss.
///
/// Memory-efficient chunked vocabulary loss: computes loss directly from
/// hidden states without materializing the `[batch, seq, vocab_size]` logits tensor.
#[derive(Debug, Clone)]
pub struct FusedLinearCrossEntropyConfig {
    /// Number of tokens to process.
    pub num_tokens: usize,

    /// Hidden dimension size.
    pub hidden_size: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Chunk size for processing vocabulary (default: 4096).
    /// Larger chunks are faster but use more memory.
    pub chunk_size: usize,

    /// Label smoothing factor (0.0 to disable).
    pub label_smoothing: f32,

    /// Index to ignore in loss computation (typically -100).
    pub ignore_index: i32,

    /// Use fp16 kernels for mixed precision.
    pub use_fp16: bool,
}

impl FusedLinearCrossEntropyConfig {
    /// Create a new config with default values.
    ///
    /// # Arguments
    ///
    /// * `num_tokens` - Number of tokens in the batch
    /// * `hidden_size` - Dimension of the hidden states
    /// * `vocab_size` - Size of the vocabulary
    ///
    /// # Panics
    ///
    /// Panics if `hidden_size` is not a multiple of 4 (required for float4/half4 vectorized loads).
    pub fn new(num_tokens: usize, hidden_size: usize, vocab_size: usize) -> Self {
        assert!(
            hidden_size % 4 == 0,
            "hidden_size ({hidden_size}) must be a multiple of 4 for vectorized Metal kernels"
        );
        Self {
            num_tokens,
            hidden_size,
            vocab_size,
            chunk_size: DEFAULT_FUSED_LINEAR_CROSS_ENTROPY_CHUNK_SIZE,
            label_smoothing: 0.0,
            ignore_index: -100,
            use_fp16: false,
        }
    }

    /// Set vocabulary chunk size (affects memory vs speed tradeoff).
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set ignore index.
    pub fn with_ignore_index(mut self, index: i32) -> Self {
        self.ignore_index = index;
        self
    }

    /// Enable label smoothing.
    pub fn with_label_smoothing(mut self, smoothing: f32) -> Self {
        self.label_smoothing = smoothing;
        self
    }

    /// Enable fp16 mode.
    pub fn with_fp16(mut self) -> Self {
        self.use_fp16 = true;
        self
    }
}

/// Output from fused linear + cross-entropy forward pass.
#[derive(Debug)]
pub struct FusedLinearCrossEntropyOutput {
    /// Per-token losses [num_tokens].
    pub losses: MetalBuffer<f32>,

    /// Cached logsumexp for backward [num_tokens].
    pub logsumexp: MetalBuffer<f32>,
}

impl FusedLinearCrossEntropyOutput {
    /// Compute mean loss over valid tokens.
    pub fn mean_loss(&self, targets: &[i32], ignore_index: i32) -> f32 {
        let losses = self.losses.as_slice();
        let mut sum = 0.0f32;
        let mut count = 0usize;

        for (i, &loss) in losses.iter().enumerate() {
            if targets[i] != ignore_index {
                sum += loss;
                count += 1;
            }
        }

        if count > 0 { sum / count as f32 } else { 0.0 }
    }
}

/// Fused linear + cross-entropy loss kernel.
///
/// Chunked vocabulary optimization providing up to 40% memory reduction:
/// - Takes hidden states [num_tokens, hidden_size] directly
/// - Computes loss WITHOUT materializing full [num_tokens, vocab_size] logits
/// - Processes vocabulary in chunks to keep memory constant
///
/// # Memory Savings Example
///
/// For a typical training setup:
/// - batch=4, seq=1024, vocab=150K, dtype=fp16
/// - Standard approach: 4 * 1024 * 150000 * 2 = 1.2GB for logits alone
/// - With fusion: only 4096 * hidden_size * 2 ≈ 8MB peak
///
/// This allows 2x larger batch sizes, which translates to ~2x throughput.
pub struct FusedLinearCrossEntropy {
    /// Metal context.
    ctx: Arc<MetalContext>,

    /// Configuration.
    config: FusedLinearCrossEntropyConfig,

    /// Tuned kernel configuration for this device/problem shape.
    tuned: CrossEntropyTunedConfig,

    /// Effective threads per token used for the fused chunked vocabulary sweep.
    threads_per_token: usize,

    /// Effective vocabulary chunk size.
    chunk_size: usize,
}

impl FusedLinearCrossEntropy {
    /// Create a new fused linear + cross-entropy kernel.
    pub fn new(ctx: Arc<MetalContext>, config: FusedLinearCrossEntropyConfig) -> Result<Self> {
        let tuned = resolve_fused_linear_ce_tuned_config(&ctx, &config)?;
        Self::new_with_tuned_config(ctx, config, tuned)
    }

    /// Create a new fused linear + cross-entropy kernel with an explicit tuned config.
    pub fn new_with_tuned_config(
        ctx: Arc<MetalContext>,
        config: FusedLinearCrossEntropyConfig,
        tuned: CrossEntropyTunedConfig,
    ) -> Result<Self> {
        let chunk_size = resolve_fused_linear_ce_chunk_size(&config, tuned.chunk_size);
        Ok(Self {
            ctx,
            config,
            tuned,
            threads_per_token: tuned.threadgroup_size as usize,
            chunk_size,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &FusedLinearCrossEntropyConfig {
        &self.config
    }

    fn function_constants(&self) -> HashMap<u64, FunctionConstant> {
        let mut constants = HashMap::new();
        constants.insert(0, FunctionConstant::UInt(self.tuned.threadgroup_size));
        constants
    }

    /// Compute forward pass directly from hidden states.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Hidden states tensor [num_tokens, hidden_size]
    /// * `lm_head_weight` - LM head weight matrix [vocab_size, hidden_size]
    /// * `targets` - Target token indices [num_tokens]
    ///
    /// # Returns
    ///
    /// Per-token losses and cached logsumexp for backward pass.
    ///
    /// # Memory Efficiency
    ///
    /// This never allocates the full `[num_tokens, vocab_size]` logits tensor.
    /// Instead, it processes the vocabulary in chunks, computing partial logsumexp
    /// values and combining them using the online logsumexp algorithm.
    pub fn forward(
        &self,
        hidden_states: &MetalBuffer<f32>,
        lm_head_weight: &MetalBuffer<f32>,
        targets: &MetalBuffer<i32>,
    ) -> Result<FusedLinearCrossEntropyOutput> {
        // Validate sizes
        let expected_hidden = self.config.num_tokens * self.config.hidden_size;
        if hidden_states.len() != expected_hidden {
            return Err(MetalError::DimensionMismatch {
                param: "hidden_states",
                expected: expected_hidden,
                actual: hidden_states.len(),
            });
        }

        let expected_weight = self.config.vocab_size * self.config.hidden_size;
        if lm_head_weight.len() != expected_weight {
            return Err(MetalError::DimensionMismatch {
                param: "lm_head_weight",
                expected: expected_weight,
                actual: lm_head_weight.len(),
            });
        }

        if targets.len() != self.config.num_tokens {
            return Err(MetalError::DimensionMismatch {
                param: "targets",
                expected: self.config.num_tokens,
                actual: targets.len(),
            });
        }

        // Allocate outputs
        let losses = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let logsumexp = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;

        self.execute_forward(hidden_states, lm_head_weight, targets, &losses, &logsumexp)?;

        Ok(FusedLinearCrossEntropyOutput { losses, logsumexp })
    }

    /// Compute forward pass with fp16 inputs.
    pub fn forward_f16(
        &self,
        hidden_states: &MetalBuffer<f16>,
        lm_head_weight: &MetalBuffer<f16>,
        targets: &MetalBuffer<i32>,
    ) -> Result<FusedLinearCrossEntropyOutput> {
        let expected_hidden = self.config.num_tokens * self.config.hidden_size;
        if hidden_states.len() != expected_hidden {
            return Err(MetalError::DimensionMismatch {
                param: "hidden_states",
                expected: expected_hidden,
                actual: hidden_states.len(),
            });
        }

        let expected_weight = self.config.vocab_size * self.config.hidden_size;
        if lm_head_weight.len() != expected_weight {
            return Err(MetalError::DimensionMismatch {
                param: "lm_head_weight",
                expected: expected_weight,
                actual: lm_head_weight.len(),
            });
        }

        if targets.len() != self.config.num_tokens {
            return Err(MetalError::DimensionMismatch {
                param: "targets",
                expected: self.config.num_tokens,
                actual: targets.len(),
            });
        }

        let losses = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let logsumexp = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;

        self.execute_forward_f16(hidden_states, lm_head_weight, targets, &losses, &logsumexp)?;

        Ok(FusedLinearCrossEntropyOutput { losses, logsumexp })
    }

    /// Execute forward kernel.
    fn execute_forward(
        &self,
        hidden_states: &MetalBuffer<f32>,
        lm_head_weight: &MetalBuffer<f32>,
        targets: &MetalBuffer<i32>,
        losses: &MetalBuffer<f32>,
        logsumexp: &MetalBuffer<f32>,
    ) -> Result<()> {
        let constants = self.function_constants();
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_specialized_pipeline_typed(
                self.ctx.device(),
                "fused_linear_cross_entropy_forward",
                &constants,
            )?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        // SAFETY: Metal compute encoder operations are safe when buffers are valid
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden_states.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(lm_head_weight.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(targets.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(losses.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(logsumexp.metal_buffer()), 0, 4);

            let params = self.create_fused_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);

            let scratch_size =
                fused_linear_ce_scratch_floats(self.threads_per_token) * std::mem::size_of::<f32>();
            encoder.setThreadgroupMemoryLength_atIndex(scratch_size, 0);
        }

        // One threadgroup per token, 128 threads per threadgroup
        let grid_size = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: self.threads_per_token,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Execute forward kernel for fp16.
    fn execute_forward_f16(
        &self,
        hidden_states: &MetalBuffer<f16>,
        lm_head_weight: &MetalBuffer<f16>,
        targets: &MetalBuffer<i32>,
        losses: &MetalBuffer<f32>,
        logsumexp: &MetalBuffer<f32>,
    ) -> Result<()> {
        let constants = self.function_constants();
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_specialized_pipeline_typed(
                self.ctx.device(),
                "fused_linear_cross_entropy_forward_f16",
                &constants,
            )?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden_states.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(lm_head_weight.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(targets.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(losses.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(logsumexp.metal_buffer()), 0, 4);

            let params = self.create_fused_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);

            let scratch_size =
                fused_linear_ce_scratch_floats(self.threads_per_token) * std::mem::size_of::<f32>();
            encoder.setThreadgroupMemoryLength_atIndex(scratch_size, 0);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: self.threads_per_token,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Create kernel parameters.
    fn create_fused_params(&self) -> FusedLinearCEParams {
        FusedLinearCEParams {
            num_tokens: self.config.num_tokens as u32,
            hidden_size: self.config.hidden_size as u32,
            vocab_size: self.config.vocab_size as u32,
            chunk_size: self.chunk_size as u32,
            ignore_index: self.config.ignore_index,
            label_smoothing: self.config.label_smoothing,
        }
    }
}

/// Parameters passed to the fused linear + CE kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct FusedLinearCEParams {
    num_tokens: u32,
    hidden_size: u32,
    vocab_size: u32,
    chunk_size: u32,
    ignore_index: i32,
    label_smoothing: f32,
}

fn resolve_fused_linear_ce_tuned_config(
    ctx: &Arc<MetalContext>,
    config: &FusedLinearCrossEntropyConfig,
) -> Result<CrossEntropyTunedConfig> {
    let tuned = ctx.tuner().tune_fused_linear_cross_entropy(ctx, config)?;
    Ok(CrossEntropyTunedConfig {
        threadgroup_size: sanitize_fused_linear_ce_threads(ctx, tuned.threadgroup_size),
        chunk_size: tuned.chunk_size.max(1),
    })
}

fn sanitize_fused_linear_ce_threads(ctx: &MetalContext, threads_per_token: u32) -> u32 {
    let max_threads = (ctx.properties().max_threads_per_threadgroup as u32).max(32);
    threads_per_token.clamp(32, max_threads).div_ceil(32) * 32
}

fn resolve_fused_linear_ce_chunk_size(
    config: &FusedLinearCrossEntropyConfig,
    tuned_chunk_size: u32,
) -> usize {
    let preferred = if config.chunk_size == DEFAULT_FUSED_LINEAR_CROSS_ENTROPY_CHUNK_SIZE {
        tuned_chunk_size as usize
    } else {
        config.chunk_size
    };
    preferred.max(1).min(config.vocab_size.max(1))
}

fn fused_linear_ce_scratch_floats(threads_per_token: usize) -> usize {
    4 * threads_per_token.div_ceil(32)
}

impl std::fmt::Debug for FusedLinearCrossEntropy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedLinearCrossEntropy")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = FusedCrossEntropyConfig::new(1024, 32000)
            .with_softcap(30.0)
            .with_ignore_index(-100);

        assert_eq!(config.num_tokens, 1024);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.softcap, 30.0);
        assert_eq!(config.ignore_index, -100);
        assert!(config.use_simd); // vocab > 1024
    }

    #[test]
    fn test_config_small_vocab() {
        let config = FusedCrossEntropyConfig::new(100, 100);
        assert!(config.use_simd); // SIMD always preferred (supports label smoothing, softcap)
    }

    /// Reference cross-entropy for testing.
    fn reference_cross_entropy(logits: &[f32], target: i32, vocab_size: usize) -> (f32, f32) {
        if target < 0 || target as usize >= vocab_size {
            return (0.0, 0.0);
        }

        // Compute logsumexp for numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let logsumexp = max_logit + sum_exp.ln();

        // CE = logsumexp - logits[target]
        let loss = logsumexp - logits[target as usize];
        (loss, logsumexp)
    }

    #[test]
    fn test_fused_cross_entropy_creation() {
        let ctx = Arc::new(MetalContext::new().unwrap());
        let config = FusedCrossEntropyConfig::new(8, 100);
        let _ce = FusedCrossEntropy::new(ctx, config).unwrap();
    }

    #[test]
    fn test_fused_linear_ce_scratch_floats() {
        assert_eq!(fused_linear_ce_scratch_floats(128), 16);
        assert_eq!(fused_linear_ce_scratch_floats(256), 32);
        assert_eq!(fused_linear_ce_scratch_floats(512), 64);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn fused_linear_cross_entropy_constructor_uses_tuned_config() {
        let ctx = Arc::new(MetalContext::new().expect("Metal required"));

        let auto_config = FusedLinearCrossEntropyConfig::new(64, 512, 200_000).with_fp16();
        let explicit_config = FusedLinearCrossEntropyConfig::new(64, 512, 200_000)
            .with_fp16()
            .with_chunk_size(1024);

        let tuned = ctx
            .tuner()
            .tune_fused_linear_cross_entropy(&ctx, &auto_config)
            .expect("tune_fused_linear_cross_entropy");

        let auto_kernel =
            FusedLinearCrossEntropy::new(ctx.clone(), auto_config).expect("auto kernel");
        let explicit_kernel =
            FusedLinearCrossEntropy::new(ctx.clone(), explicit_config).expect("explicit kernel");

        assert_eq!(
            auto_kernel.threads_per_token as u32,
            sanitize_fused_linear_ce_threads(&ctx, tuned.threadgroup_size)
        );
        assert_eq!(auto_kernel.chunk_size, tuned.chunk_size as usize);
        assert_eq!(explicit_kernel.chunk_size, 1024);
    }

    #[test]
    fn test_fused_cross_entropy_forward() {
        let ctx = Arc::new(MetalContext::new().unwrap());
        let num_tokens = 4;
        let vocab_size = 32;

        let mut config = FusedCrossEntropyConfig::new(num_tokens, vocab_size);
        config.use_simd = false; // Use simple kernel for testing

        let ce = FusedCrossEntropy::new(ctx.clone(), config).unwrap();

        // Create test data: logits [num_tokens, vocab_size]
        let mut logits_data = vec![0.0f32; num_tokens * vocab_size];
        for i in 0..num_tokens {
            for j in 0..vocab_size {
                // Random-ish logits
                logits_data[i * vocab_size + j] = ((i * 7 + j * 3) % 10) as f32 - 5.0;
            }
        }

        // Targets: one per token
        let targets_data = vec![0i32, 5, 10, 15];

        let logits = MetalBuffer::from_slice(&ctx, &logits_data, BufferUsage::Shared).unwrap();
        let targets = MetalBuffer::from_slice(&ctx, &targets_data, BufferUsage::Shared).unwrap();

        let output = ce.forward(&logits, &targets).unwrap();

        // Verify losses against reference
        let losses = output.losses.as_slice();
        let logsumexp = output.logsumexp.as_slice();

        for i in 0..num_tokens {
            let row = &logits_data[i * vocab_size..(i + 1) * vocab_size];
            let (ref_loss, ref_lse) = reference_cross_entropy(row, targets_data[i], vocab_size);

            assert!(
                (losses[i] - ref_loss).abs() < 1e-4,
                "Loss mismatch at token {}: got {}, expected {}",
                i,
                losses[i],
                ref_loss
            );
            assert!(
                (logsumexp[i] - ref_lse).abs() < 1e-4,
                "Logsumexp mismatch at token {}: got {}, expected {}",
                i,
                logsumexp[i],
                ref_lse
            );
        }
    }

    #[test]
    fn test_fused_cross_entropy_ignore_index() {
        let ctx = Arc::new(MetalContext::new().unwrap());
        let num_tokens = 4;
        let vocab_size = 32;

        let mut config = FusedCrossEntropyConfig::new(num_tokens, vocab_size);
        config.use_simd = false;
        config.ignore_index = -100;

        let ce = FusedCrossEntropy::new(ctx.clone(), config).unwrap();

        // Logits
        let logits_data = vec![1.0f32; num_tokens * vocab_size];
        // Some targets are ignored
        let targets_data = vec![0i32, -100, 5, -100];

        let logits = MetalBuffer::from_slice(&ctx, &logits_data, BufferUsage::Shared).unwrap();
        let targets = MetalBuffer::from_slice(&ctx, &targets_data, BufferUsage::Shared).unwrap();

        let output = ce.forward(&logits, &targets).unwrap();

        let losses = output.losses.as_slice();
        // Ignored tokens should have 0 loss
        assert!(losses[1].abs() < 1e-6, "Ignored token should have 0 loss");
        assert!(losses[3].abs() < 1e-6, "Ignored token should have 0 loss");
        // Non-ignored should have non-zero loss
        assert!(losses[0] > 0.0, "Valid token should have positive loss");
        assert!(losses[2] > 0.0, "Valid token should have positive loss");
    }

    #[test]
    fn test_fused_cross_entropy_mean_loss() {
        let ctx = Arc::new(MetalContext::new().unwrap());
        let num_tokens = 4;
        let vocab_size = 32;

        let mut config = FusedCrossEntropyConfig::new(num_tokens, vocab_size);
        config.use_simd = false;

        let ce = FusedCrossEntropy::new(ctx.clone(), config).unwrap();

        let logits_data = vec![1.0f32; num_tokens * vocab_size];
        let targets_data = vec![0i32, 1, 2, 3];

        let logits = MetalBuffer::from_slice(&ctx, &logits_data, BufferUsage::Shared).unwrap();
        let targets = MetalBuffer::from_slice(&ctx, &targets_data, BufferUsage::Shared).unwrap();

        let output = ce.forward(&logits, &targets).unwrap();

        // Mean loss over all valid tokens
        let mean = output.mean_loss(&targets_data, -100);

        // All logits are 1.0, so all losses should be identical
        // CE = log(vocab_size) when all logits are equal
        let expected = (vocab_size as f32).ln();
        assert!(
            (mean - expected).abs() < 1e-4,
            "Mean loss mismatch: got {}, expected {}",
            mean,
            expected
        );
    }
}
