//! Fused knowledge distillation loss kernels.
//!
//! GPU-accelerated distillation losses without materializing probability tensors:
//! - KL Divergence (forward and reverse)
//! - Jensen-Shannon Divergence
//! - Soft Cross-Entropy
//! - Hidden state alignment (MSE, Cosine)
//!
//! Key optimizations:
//! - Uses online softmax to avoid O(vocab) memory per token
//! - Temperature scaling built into kernel
//! - SIMD parallelization for large vocabularies
//! - Caches logsumexp for efficient backward pass

use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState,
};

use crate::{
    buffer::{AsMetalBuffer, BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result},
};

/// Configuration for fused distillation loss kernels.
#[derive(Debug, Clone)]
pub struct FusedDistillConfig {
    /// Number of tokens to process.
    pub num_tokens: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Temperature for softening distributions.
    pub temperature: f32,

    /// Blending weight for soft loss (vs hard loss).
    pub alpha: f32,

    /// Index to ignore in loss computation (typically -100).
    pub ignore_index: i32,

    /// Use SIMD-parallel kernel (more efficient for large vocabularies).
    pub use_simd: bool,

    /// Use fp16 kernels for mixed precision.
    pub use_fp16: bool,
}

impl FusedDistillConfig {
    /// Create a new config with default values.
    pub fn new(num_tokens: usize, vocab_size: usize) -> Self {
        Self {
            num_tokens,
            vocab_size,
            temperature: 2.0,
            alpha: 0.5,
            ignore_index: -100,
            use_simd: vocab_size > 1024,
            use_fp16: false,
        }
    }

    /// Set temperature for soft targets.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set alpha for blending soft and hard losses.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set ignore index.
    pub fn with_ignore_index(mut self, index: i32) -> Self {
        self.ignore_index = index;
        self
    }

    /// Enable fp16 mode.
    pub fn with_fp16(mut self) -> Self {
        self.use_fp16 = true;
        self
    }
}

/// Output from fused distillation loss forward pass.
#[derive(Debug)]
pub struct FusedDistillOutput {
    /// Per-token losses [num_tokens].
    pub losses: MetalBuffer<f32>,

    /// Cached teacher logsumexp for backward [num_tokens].
    pub teacher_lse: MetalBuffer<f32>,

    /// Cached student logsumexp for backward [num_tokens].
    pub student_lse: MetalBuffer<f32>,
}

impl FusedDistillOutput {
    /// Compute mean loss over all tokens.
    pub fn mean_loss(&self) -> f32 {
        let losses = self.losses.as_slice();
        if losses.is_empty() {
            return 0.0;
        }
        losses.iter().sum::<f32>() / losses.len() as f32
    }
}

/// Type of distillation loss to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistillLossType {
    /// Forward KL divergence: KL(teacher || student)
    /// Mode-covering behavior - student tries to cover all teacher modes.
    KlDivergence,

    /// Reverse KL divergence: KL(student || teacher)
    /// Mode-seeking behavior - student focuses on main teacher modes.
    ReverseKlDivergence,

    /// Jensen-Shannon divergence: symmetric, bounded loss.
    JensenShannon,

    /// Soft cross-entropy with teacher soft targets.
    SoftCrossEntropy,
}

/// Fused distillation loss kernel.
///
/// Provides efficient forward and backward passes for knowledge distillation
/// loss functions with support for large vocabularies and mixed precision.
pub struct FusedDistill {
    /// Metal context.
    ctx: Arc<MetalContext>,

    /// Configuration.
    config: FusedDistillConfig,
}

impl FusedDistill {
    /// Create a new fused distillation kernel.
    pub fn new(ctx: Arc<MetalContext>, config: FusedDistillConfig) -> Result<Self> {
        Ok(Self { ctx, config })
    }

    /// Get the configuration.
    pub fn config(&self) -> &FusedDistillConfig {
        &self.config
    }

    /// Compute forward pass for KL divergence.
    ///
    /// # Arguments
    ///
    /// * `teacher_logits` - Teacher logits [num_tokens, vocab_size]
    /// * `student_logits` - Student logits [num_tokens, vocab_size]
    /// * `loss_type` - Type of distillation loss to compute
    ///
    /// # Returns
    ///
    /// Per-token losses and cached values for backward pass.
    pub fn forward(
        &self,
        teacher_logits: &impl AsMetalBuffer,
        student_logits: &impl AsMetalBuffer,
        loss_type: DistillLossType,
    ) -> Result<FusedDistillOutput> {
        // Validate sizes not possible easily with generic traits without size method
        // Assume caller ensures sizes or add len() to AsMetalBuffer

        // Allocate outputs
        let losses = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let teacher_lse = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let student_lse = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;

        self.execute_forward(
            teacher_logits,
            student_logits,
            &losses,
            &teacher_lse,
            &student_lse,
            loss_type,
        )?;

        Ok(FusedDistillOutput {
            losses,
            teacher_lse,
            student_lse,
        })
    }

    /// Compute forward pass with fp16 logits.
    pub fn forward_f16(
        &self,
        teacher_logits: &impl AsMetalBuffer,
        student_logits: &impl AsMetalBuffer,
        loss_type: DistillLossType,
    ) -> Result<FusedDistillOutput> {
        let losses = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let teacher_lse = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let student_lse = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;

        self.execute_forward_f16(
            teacher_logits,
            student_logits,
            &losses,
            &teacher_lse,
            &student_lse,
            loss_type,
        )?;

        Ok(FusedDistillOutput {
            losses,
            teacher_lse,
            student_lse,
        })
    }

    /// Compute backward pass (in-place gradient).
    ///
    /// # Arguments
    ///
    /// * `teacher_logits` - Teacher logits [num_tokens, vocab_size]
    /// * `student_logits` - Student logits [num_tokens, vocab_size] - will be overwritten with gradients
    /// * `teacher_lse` - Cached teacher logsumexp from forward [num_tokens]
    /// * `student_lse` - Cached student logsumexp from forward [num_tokens]
    /// * `grad_loss` - Upstream gradient [num_tokens]
    pub fn backward(
        &self,
        teacher_logits: &MetalBuffer<f32>,
        student_logits: &mut MetalBuffer<f32>,
        teacher_lse: &MetalBuffer<f32>,
        student_lse: &MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        self.execute_backward(
            teacher_logits,
            student_logits,
            teacher_lse,
            student_lse,
            grad_loss,
        )
    }

    /// Execute forward kernel.
    fn execute_forward(
        &self,
        teacher_logits: &impl AsMetalBuffer,
        student_logits: &impl AsMetalBuffer,
        losses: &MetalBuffer<f32>,
        teacher_lse: &MetalBuffer<f32>,
        student_lse: &MetalBuffer<f32>,
        loss_type: DistillLossType,
    ) -> Result<()> {
        let function_name = match (loss_type, self.config.use_simd) {
            (DistillLossType::KlDivergence, true) => "fused_kl_divergence_forward_simd",
            (DistillLossType::KlDivergence, false) => "fused_kl_divergence_forward",
            (DistillLossType::ReverseKlDivergence, _) => "fused_reverse_kl_divergence_forward",
            (DistillLossType::JensenShannon, _) => "fused_jensen_shannon_forward",
            (DistillLossType::SoftCrossEntropy, _) => "fused_soft_cross_entropy_forward",
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

        // Set buffers based on loss type
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(teacher_logits.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(student_logits.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(losses.as_metal_buffer()), 0, 2);

            match loss_type {
                DistillLossType::KlDivergence | DistillLossType::ReverseKlDivergence => {
                    encoder.setBuffer_offset_atIndex(Some(teacher_lse.as_metal_buffer()), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(student_lse.as_metal_buffer()), 0, 4);

                    let params = self.create_params();
                    let params_ptr = NonNull::from(&params).cast();
                    encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);

                    if self.config.use_simd {
                        // Threadgroup memory for reduction
                        let scratch_size = 16 * std::mem::size_of::<f32>(); // 4 floats per SIMD group, 4 groups
                        encoder.setThreadgroupMemoryLength_atIndex(scratch_size, 0);
                    }
                }
                DistillLossType::JensenShannon | DistillLossType::SoftCrossEntropy => {
                    let params = self.create_params();
                    let params_ptr = NonNull::from(&params).cast();
                    encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 3);
                }
            }
        }

        let (grid_size, threadgroup_size) =
            if self.config.use_simd && matches!(loss_type, DistillLossType::KlDivergence) {
                (
                    objc2_metal::MTLSize {
                        width: self.config.num_tokens,
                        height: 1,
                        depth: 1,
                    },
                    objc2_metal::MTLSize {
                        width: 128, // DISTILL_THREADS_PER_TOKEN
                        height: 1,
                        depth: 1,
                    },
                )
            } else {
                (
                    objc2_metal::MTLSize {
                        width: self.config.num_tokens,
                        height: 1,
                        depth: 1,
                    },
                    objc2_metal::MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                )
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
        teacher_logits: &impl AsMetalBuffer,
        student_logits: &impl AsMetalBuffer,
        losses: &MetalBuffer<f32>,
        teacher_lse: &MetalBuffer<f32>,
        student_lse: &MetalBuffer<f32>,
        loss_type: DistillLossType,
    ) -> Result<()> {
        // Only KL divergence has fp16 variant
        if loss_type != DistillLossType::KlDivergence {
            return Err(MetalError::ExecutionFailed(
                "fp16 only supported for KL divergence".to_string(),
            ));
        }

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(
                self.ctx.device(),
                "fused_kl_divergence_forward_f16",
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
            encoder.setBuffer_offset_atIndex(Some(teacher_logits.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(student_logits.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(losses.as_metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(teacher_lse.as_metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(student_lse.as_metal_buffer()), 0, 4);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);

            let scratch_size = 16 * std::mem::size_of::<f32>();
            encoder.setThreadgroupMemoryLength_atIndex(scratch_size, 0);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: 128,
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
        teacher_logits: &MetalBuffer<f32>,
        student_logits: &mut MetalBuffer<f32>,
        teacher_lse: &MetalBuffer<f32>,
        student_lse: &MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        let function_name = if self.config.use_simd {
            "fused_kl_divergence_backward_simd"
        } else {
            "fused_kl_divergence_backward"
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
            encoder.setBuffer_offset_atIndex(Some(teacher_logits.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(student_logits.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(teacher_lse.as_metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(student_lse.as_metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(grad_loss.as_metal_buffer()), 0, 4);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);
        }

        let (grid_size, threadgroup_size) = if self.config.use_simd {
            (
                objc2_metal::MTLSize {
                    width: self.config.num_tokens,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            )
        } else {
            (
                objc2_metal::MTLSize {
                    width: self.config.vocab_size,
                    height: self.config.num_tokens,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            )
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
    fn create_params(&self) -> DistillParams {
        DistillParams {
            num_tokens: self.config.num_tokens as u32,
            vocab_size: self.config.vocab_size as u32,
            temperature: self.config.temperature,
            alpha: self.config.alpha,
            ignore_index: self.config.ignore_index,
        }
    }

    /// Compute backward pass for soft cross-entropy loss.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher logits [num_tokens, vocab_size]
    /// * `student_logits` - Student logits [num_tokens, vocab_size]
    /// * `grad_student` - Output gradient buffer [num_tokens, vocab_size]
    /// * `grad_loss` - Upstream gradient [num_tokens]
    pub fn backward_soft_ce(
        &self,
        teacher_logits: &MetalBuffer<f32>,
        student_logits: &MetalBuffer<f32>,
        grad_student: &mut MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        self.execute_backward_generic(
            teacher_logits,
            student_logits,
            grad_student,
            grad_loss,
            "fused_soft_cross_entropy_backward",
        )
    }

    /// Compute backward pass for Jensen-Shannon divergence.
    pub fn backward_jensen_shannon(
        &self,
        teacher_logits: &MetalBuffer<f32>,
        student_logits: &MetalBuffer<f32>,
        grad_student: &mut MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        self.execute_backward_generic(
            teacher_logits,
            student_logits,
            grad_student,
            grad_loss,
            "fused_jensen_shannon_backward",
        )
    }

    /// Execute a generic backward kernel.
    fn execute_backward_generic(
        &self,
        teacher_logits: &MetalBuffer<f32>,
        student_logits: &MetalBuffer<f32>,
        grad_student: &mut MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
        function_name: &str,
    ) -> Result<()> {
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
            encoder.setBuffer_offset_atIndex(Some(teacher_logits.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(student_logits.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(grad_student.as_metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(grad_loss.as_metal_buffer()), 0, 3);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.vocab_size,
            height: self.config.num_tokens,
            depth: 1,
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: 32.min(self.config.vocab_size),
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

    /// Fused forward+backward pass for KL divergence.
    ///
    /// Computes loss and gradient in a single kernel dispatch, avoiding
    /// the need for separate forward and backward passes.
    ///
    /// # Returns
    /// Tuple of (losses, grad_student) buffers.
    pub fn forward_backward_kl(
        &self,
        teacher_logits: &impl AsMetalBuffer,
        student_logits: &impl AsMetalBuffer,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<(MetalBuffer<f32>, MetalBuffer<f32>)> {
        let losses = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let grad_student = MetalBuffer::new(
            &self.ctx,
            self.config.num_tokens * self.config.vocab_size,
            BufferUsage::Shared,
        )?;

        self.execute_forward_backward_kl(
            teacher_logits,
            student_logits,
            &losses,
            &grad_student,
            grad_loss,
        )?;

        Ok((losses, grad_student))
    }

    /// Execute fused KL forward+backward kernel.
    fn execute_forward_backward_kl(
        &self,
        teacher_logits: &impl AsMetalBuffer,
        student_logits: &impl AsMetalBuffer,
        losses: &MetalBuffer<f32>,
        grad_student: &MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        // Only one fused forward+backward kernel exists - it uses multi-SIMD internally
        let function_name = "fused_kl_divergence_forward_backward";

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
            encoder.setBuffer_offset_atIndex(Some(teacher_logits.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(student_logits.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(losses.as_metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(grad_student.as_metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(grad_loss.as_metal_buffer()), 0, 4);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);

            // Threadgroup memory for reduction
            // 4 values per SIMD group (t_max, t_sum, s_max, s_sum), 4 SIMD groups
            let scratch_size = 16 * std::mem::size_of::<f32>();
            encoder.setThreadgroupMemoryLength_atIndex(scratch_size, 0);
        }

        // Dispatch one threadgroup per token, clamped to pipeline max
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        let threads_per_tg = max_threads.min(128); // DISTILL_THREADS_PER_TOKEN

        let grid_size = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: threads_per_tg,
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

    /// Fused combined loss forward+backward (hard CE + soft KL).
    ///
    /// Computes both hard cross-entropy and soft KL losses with gradients
    /// in a single kernel dispatch.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher logits [num_tokens, vocab_size]
    /// * `student_logits` - Student logits [num_tokens, vocab_size]
    /// * `labels` - Ground truth labels [num_tokens]
    /// * `grad_loss` - Upstream gradient [num_tokens]
    ///
    /// # Returns
    /// Tuple of (hard_loss, soft_loss, grad_student) buffers.
    pub fn forward_backward_combined(
        &self,
        teacher_logits: &impl AsMetalBuffer,
        student_logits: &impl AsMetalBuffer,
        labels: &MetalBuffer<i32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<(MetalBuffer<f32>, MetalBuffer<f32>, MetalBuffer<f32>)> {
        let hard_loss = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let soft_loss = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;
        let grad_student = MetalBuffer::new(
            &self.ctx,
            self.config.num_tokens * self.config.vocab_size,
            BufferUsage::Shared,
        )?;

        self.execute_forward_backward_combined(
            teacher_logits,
            student_logits,
            labels,
            &hard_loss,
            &soft_loss,
            &grad_student,
            grad_loss,
        )?;

        Ok((hard_loss, soft_loss, grad_student))
    }

    /// Execute fused combined loss forward+backward kernel.
    #[allow(clippy::too_many_arguments)]
    fn execute_forward_backward_combined(
        &self,
        teacher_logits: &impl AsMetalBuffer,
        student_logits: &impl AsMetalBuffer,
        labels: &MetalBuffer<i32>,
        hard_loss: &MetalBuffer<f32>,
        soft_loss: &MetalBuffer<f32>,
        grad_student: &MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(
                self.ctx.device(),
                "fused_combined_loss_forward_backward",
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
            encoder.setBuffer_offset_atIndex(Some(teacher_logits.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(student_logits.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(labels.as_metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(hard_loss.as_metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(soft_loss.as_metal_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(grad_student.as_metal_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(grad_loss.as_metal_buffer()), 0, 6);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 7);
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
}

/// Parameters passed to the kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DistillParams {
    num_tokens: u32,
    vocab_size: u32,
    temperature: f32,
    alpha: f32,
    ignore_index: i32,
}

impl std::fmt::Debug for FusedDistill {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedDistill")
            .field("config", &self.config)
            .finish()
    }
}

// =============================================================================
// HIDDEN STATE ALIGNMENT
// =============================================================================

/// Configuration for hidden state alignment loss.
#[derive(Debug, Clone)]
pub struct HiddenAlignConfig {
    /// Number of tokens.
    pub num_tokens: usize,

    /// Teacher hidden dimension.
    pub teacher_dim: usize,

    /// Student hidden dimension.
    pub student_dim: usize,

    /// Loss weight.
    pub weight: f32,

    /// Use SIMD-parallel kernel.
    pub use_simd: bool,
}

impl HiddenAlignConfig {
    /// Create a new config.
    pub fn new(num_tokens: usize, teacher_dim: usize, student_dim: usize) -> Self {
        Self {
            num_tokens,
            teacher_dim,
            student_dim,
            weight: 1.0,
            use_simd: teacher_dim > 256,
        }
    }

    /// Set loss weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

/// Type of hidden state alignment loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiddenAlignLossType {
    /// Mean squared error.
    Mse,
    /// Cosine similarity loss (1 - cosine_sim).
    Cosine,
}

/// Fused hidden state alignment loss kernel.
pub struct FusedHiddenAlign {
    /// Metal context.
    ctx: Arc<MetalContext>,

    /// Configuration.
    config: HiddenAlignConfig,
}

impl FusedHiddenAlign {
    /// Create a new hidden alignment kernel.
    pub fn new(ctx: Arc<MetalContext>, config: HiddenAlignConfig) -> Result<Self> {
        Ok(Self { ctx, config })
    }

    /// Compute forward pass.
    ///
    /// Accepts any buffer type implementing `AsMetalBuffer` trait for zero-copy support.
    pub fn forward(
        &self,
        teacher_hidden: &impl AsMetalBuffer,
        student_hidden: &impl AsMetalBuffer,
        loss_type: HiddenAlignLossType,
    ) -> Result<MetalBuffer<f32>> {
        // Note: Size validation is skipped for generic buffers since AsMetalBuffer
        // doesn't require a len() method. Caller must ensure correct sizes.

        let losses = MetalBuffer::new(&self.ctx, self.config.num_tokens, BufferUsage::Shared)?;

        self.execute_forward(teacher_hidden, student_hidden, &losses, loss_type)?;

        Ok(losses)
    }

    /// Execute forward kernel.
    fn execute_forward(
        &self,
        teacher_hidden: &impl AsMetalBuffer,
        student_hidden: &impl AsMetalBuffer,
        losses: &MetalBuffer<f32>,
        loss_type: HiddenAlignLossType,
    ) -> Result<()> {
        let function_name = match (loss_type, self.config.use_simd) {
            (HiddenAlignLossType::Mse, true) => "fused_hidden_mse_forward_simd",
            (HiddenAlignLossType::Mse, false) => "fused_hidden_mse_forward",
            (HiddenAlignLossType::Cosine, _) => "fused_hidden_cosine_forward",
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
            encoder.setBuffer_offset_atIndex(Some(teacher_hidden.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(student_hidden.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(losses.metal_buffer()), 0, 2);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 3);
        }

        let (grid_size, threadgroup_size) =
            if self.config.use_simd && loss_type == HiddenAlignLossType::Mse {
                (
                    objc2_metal::MTLSize {
                        width: self.config.num_tokens,
                        height: 1,
                        depth: 1,
                    },
                    objc2_metal::MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                )
            } else {
                (
                    objc2_metal::MTLSize {
                        width: self.config.num_tokens,
                        height: 1,
                        depth: 1,
                    },
                    objc2_metal::MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                )
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
    fn create_params(&self) -> HiddenAlignParams {
        HiddenAlignParams {
            num_tokens: self.config.num_tokens as u32,
            teacher_dim: self.config.teacher_dim as u32,
            student_dim: self.config.student_dim as u32,
            projection_dim: 0, // Not used yet
            weight: self.config.weight,
        }
    }

    /// Compute backward pass for hidden MSE loss.
    ///
    /// # Arguments
    /// * `teacher_hidden` - Teacher hidden states [num_tokens, teacher_dim]
    /// * `student_hidden` - Student hidden states [num_tokens, student_dim]
    /// * `grad_student` - Output gradient buffer [num_tokens, student_dim]
    /// * `grad_loss` - Upstream gradient [num_tokens]
    pub fn backward_mse(
        &self,
        teacher_hidden: &MetalBuffer<f32>,
        student_hidden: &MetalBuffer<f32>,
        grad_student: &mut MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        self.execute_backward(
            teacher_hidden,
            student_hidden,
            grad_student,
            grad_loss,
            "fused_hidden_mse_backward",
        )
    }

    /// Compute backward pass for hidden cosine similarity loss.
    pub fn backward_cosine(
        &self,
        teacher_hidden: &MetalBuffer<f32>,
        student_hidden: &MetalBuffer<f32>,
        grad_student: &mut MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
    ) -> Result<()> {
        self.execute_backward(
            teacher_hidden,
            student_hidden,
            grad_student,
            grad_loss,
            "fused_hidden_cosine_backward",
        )
    }

    /// Execute backward kernel.
    fn execute_backward(
        &self,
        teacher_hidden: &MetalBuffer<f32>,
        student_hidden: &MetalBuffer<f32>,
        grad_student: &mut MetalBuffer<f32>,
        grad_loss: &MetalBuffer<f32>,
        function_name: &str,
    ) -> Result<()> {
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
            encoder.setBuffer_offset_atIndex(Some(teacher_hidden.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(student_hidden.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(grad_student.as_metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(grad_loss.as_metal_buffer()), 0, 3);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.student_dim,
            height: self.config.num_tokens,
            depth: 1,
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: 32.min(self.config.student_dim),
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
}

/// Parameters for hidden state alignment kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HiddenAlignParams {
    num_tokens: u32,
    teacher_dim: u32,
    student_dim: u32,
    projection_dim: u32,
    weight: f32,
}

impl std::fmt::Debug for FusedHiddenAlign {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedHiddenAlign")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distill_config() {
        let config = FusedDistillConfig::new(1024, 32000)
            .with_temperature(4.0)
            .with_alpha(0.7);

        assert_eq!(config.num_tokens, 1024);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.temperature, 4.0);
        assert_eq!(config.alpha, 0.7);
        assert!(config.use_simd); // vocab > 1024
    }

    #[test]
    fn test_distill_config_small_vocab() {
        let config = FusedDistillConfig::new(100, 100);
        assert!(!config.use_simd); // vocab <= 1024
    }

    #[test]
    fn test_fused_distill_creation() {
        let ctx = Arc::new(MetalContext::new().unwrap());
        let config = FusedDistillConfig::new(8, 100);
        let _distill = FusedDistill::new(ctx, config).unwrap();
    }

    /// Reference KL divergence for testing.
    fn reference_kl_divergence(
        teacher_logits: &[f32],
        student_logits: &[f32],
        vocab_size: usize,
        temperature: f32,
    ) -> f32 {
        // Compute softmax for teacher
        let t_max = teacher_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let t_exp: Vec<f32> = teacher_logits
            .iter()
            .map(|&x| ((x / temperature) - t_max / temperature).exp())
            .collect();
        let t_sum: f32 = t_exp.iter().sum();
        let t_probs: Vec<f32> = t_exp.iter().map(|&x| x / t_sum).collect();

        // Compute softmax for student
        let s_max = student_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let s_exp: Vec<f32> = student_logits
            .iter()
            .map(|&x| ((x / temperature) - s_max / temperature).exp())
            .collect();
        let s_sum: f32 = s_exp.iter().sum();
        let s_probs: Vec<f32> = s_exp.iter().map(|&x| x / s_sum).collect();

        // KL divergence
        let mut kl = 0.0f32;
        for i in 0..vocab_size {
            if t_probs[i] > 1e-10 {
                kl += t_probs[i] * (t_probs[i].ln() - s_probs[i].max(1e-10).ln());
            }
        }

        kl.max(0.0)
    }

    #[test]
    fn test_fused_kl_divergence() {
        let ctx = Arc::new(MetalContext::new().unwrap());
        let num_tokens = 4;
        let vocab_size = 32;
        let temperature = 2.0;

        let mut config = FusedDistillConfig::new(num_tokens, vocab_size);
        config.temperature = temperature;
        config.use_simd = false;

        let distill = FusedDistill::new(ctx.clone(), config).unwrap();

        // Create test data
        let mut teacher_data = vec![0.0f32; num_tokens * vocab_size];
        let mut student_data = vec![0.0f32; num_tokens * vocab_size];

        for i in 0..num_tokens {
            for j in 0..vocab_size {
                teacher_data[i * vocab_size + j] = ((i * 7 + j * 3) % 10) as f32 - 5.0;
                student_data[i * vocab_size + j] = ((i * 5 + j * 2) % 10) as f32 - 4.0;
            }
        }

        let teacher = MetalBuffer::from_slice(&ctx, &teacher_data, BufferUsage::Shared).unwrap();
        let student = MetalBuffer::from_slice(&ctx, &student_data, BufferUsage::Shared).unwrap();

        let output = distill
            .forward(&teacher, &student, DistillLossType::KlDivergence)
            .unwrap();

        // Verify against reference
        let losses = output.losses.as_slice();

        for i in 0..num_tokens {
            let t_row = &teacher_data[i * vocab_size..(i + 1) * vocab_size];
            let s_row = &student_data[i * vocab_size..(i + 1) * vocab_size];
            let ref_kl = reference_kl_divergence(t_row, s_row, vocab_size, temperature);

            assert!(
                (losses[i] - ref_kl).abs() < 1e-3,
                "KL mismatch at token {}: got {}, expected {}",
                i,
                losses[i],
                ref_kl
            );
        }
    }

    #[test]
    fn test_kl_identical_distributions() {
        let ctx = Arc::new(MetalContext::new().unwrap());
        let num_tokens = 4;
        let vocab_size = 32;

        let mut config = FusedDistillConfig::new(num_tokens, vocab_size);
        config.use_simd = false;

        let distill = FusedDistill::new(ctx.clone(), config).unwrap();

        // Same logits for teacher and student
        let logits_data: Vec<f32> = (0..(num_tokens * vocab_size))
            .map(|i| (i % 10) as f32 - 5.0)
            .collect();

        let teacher = MetalBuffer::from_slice(&ctx, &logits_data, BufferUsage::Shared).unwrap();
        let student = MetalBuffer::from_slice(&ctx, &logits_data, BufferUsage::Shared).unwrap();

        let output = distill
            .forward(&teacher, &student, DistillLossType::KlDivergence)
            .unwrap();

        // KL of identical distributions should be 0
        let losses = output.losses.as_slice();
        for (i, &loss) in losses.iter().enumerate() {
            assert!(
                loss.abs() < 1e-5,
                "KL of identical distributions should be 0, got {} at token {}",
                loss,
                i
            );
        }
    }

    #[test]
    fn test_hidden_align_config() {
        let config = HiddenAlignConfig::new(1024, 4096, 2048).with_weight(0.5);

        assert_eq!(config.num_tokens, 1024);
        assert_eq!(config.teacher_dim, 4096);
        assert_eq!(config.student_dim, 2048);
        assert_eq!(config.weight, 0.5);
        assert!(config.use_simd); // dim > 256
    }

    #[test]
    fn test_fused_hidden_mse() {
        let ctx = Arc::new(MetalContext::new().unwrap());
        let num_tokens = 4;
        let dim = 64;

        let mut config = HiddenAlignConfig::new(num_tokens, dim, dim);
        config.use_simd = false;

        let align = FusedHiddenAlign::new(ctx.clone(), config).unwrap();

        // Create test data
        let teacher_data: Vec<f32> = (0..(num_tokens * dim))
            .map(|i| (i % 10) as f32 / 10.0)
            .collect();
        let student_data: Vec<f32> = (0..(num_tokens * dim))
            .map(|i| ((i + 3) % 10) as f32 / 10.0)
            .collect();

        let teacher = MetalBuffer::from_slice(&ctx, &teacher_data, BufferUsage::Shared).unwrap();
        let student = MetalBuffer::from_slice(&ctx, &student_data, BufferUsage::Shared).unwrap();

        let losses = align
            .forward(&teacher, &student, HiddenAlignLossType::Mse)
            .unwrap();

        // Verify MSE is reasonable
        let loss_data = losses.as_slice();
        for (i, &loss) in loss_data.iter().enumerate() {
            assert!(loss >= 0.0, "MSE should be non-negative at token {}", i);
            assert!(
                loss < 1.0,
                "MSE should be reasonable at token {}: got {}",
                i,
                loss
            );
        }
    }

    #[test]
    fn test_hidden_mse_identical() {
        let ctx = Arc::new(MetalContext::new().unwrap());
        let num_tokens = 4;
        let dim = 64;

        let mut config = HiddenAlignConfig::new(num_tokens, dim, dim);
        config.use_simd = false;

        let align = FusedHiddenAlign::new(ctx.clone(), config).unwrap();

        let data: Vec<f32> = (0..(num_tokens * dim))
            .map(|i| (i % 10) as f32 / 10.0)
            .collect();

        let teacher = MetalBuffer::from_slice(&ctx, &data, BufferUsage::Shared).unwrap();
        let student = MetalBuffer::from_slice(&ctx, &data, BufferUsage::Shared).unwrap();

        let losses = align
            .forward(&teacher, &student, HiddenAlignLossType::Mse)
            .unwrap();

        // MSE of identical vectors should be 0
        let loss_data = losses.as_slice();
        for (i, &loss) in loss_data.iter().enumerate() {
            assert!(
                loss.abs() < 1e-6,
                "MSE of identical vectors should be 0, got {} at token {}",
                loss,
                i
            );
        }
    }

    #[test]
    fn test_fused_forward_backward_kl() {
        let ctx = Arc::new(MetalContext::new().unwrap());
        let num_tokens = 4;
        let vocab_size = 32;
        let temperature = 2.0;

        let mut config = FusedDistillConfig::new(num_tokens, vocab_size);
        config.temperature = temperature;
        config.use_simd = false;

        let distill = FusedDistill::new(ctx.clone(), config).unwrap();

        // Create test data
        let mut teacher_data = vec![0.0f32; num_tokens * vocab_size];
        let mut student_data = vec![0.0f32; num_tokens * vocab_size];

        for i in 0..num_tokens {
            for j in 0..vocab_size {
                teacher_data[i * vocab_size + j] = ((i * 7 + j * 3) % 10) as f32 - 5.0;
                student_data[i * vocab_size + j] = ((i * 5 + j * 2) % 10) as f32 - 4.0;
            }
        }

        let teacher = MetalBuffer::from_slice(&ctx, &teacher_data, BufferUsage::Shared).unwrap();
        let student = MetalBuffer::from_slice(&ctx, &student_data, BufferUsage::Shared).unwrap();
        let grad_loss =
            MetalBuffer::from_slice(&ctx, &vec![1.0f32; num_tokens], BufferUsage::Shared).unwrap();

        // Compute fused forward+backward
        let (fused_losses, grad_student) = distill
            .forward_backward_kl(&teacher, &student, &grad_loss)
            .unwrap();

        // Check if kernel executed by verifying gradients are non-zero
        let grad_data = grad_student.as_slice();
        assert_eq!(grad_data.len(), num_tokens * vocab_size);

        let any_nonzero = grad_data.iter().any(|&x| x.abs() > 1e-10);
        assert!(
            any_nonzero,
            "Gradients are all zero - kernel may not have executed"
        );

        // Verify fused losses against CPU reference (T^2 scaled KL)
        let fused_loss_data = fused_losses.as_slice();
        let t2 = temperature * temperature;

        for i in 0..num_tokens {
            let t_row = &teacher_data[i * vocab_size..(i + 1) * vocab_size];
            let s_row = &student_data[i * vocab_size..(i + 1) * vocab_size];
            let ref_kl = reference_kl_divergence(t_row, s_row, vocab_size, temperature);
            let expected_fused = ref_kl * t2;

            assert!(
                (expected_fused - fused_loss_data[i]).abs() < 0.1,
                "Fused loss mismatch at token {}: expected {} (ref_kl {} * T^2 {}), got {}",
                i,
                expected_fused,
                ref_kl,
                t2,
                fused_loss_data[i]
            );
        }

        // Sum of gradients per token should be close to 0 (softmax derivative property)
        for i in 0..num_tokens {
            let grad_sum: f32 = grad_data[i * vocab_size..(i + 1) * vocab_size].iter().sum();
            assert!(
                grad_sum.abs() < 1e-3,
                "Gradient sum should be ~0 at token {}: got {}",
                i,
                grad_sum
            );
        }
    }
}
