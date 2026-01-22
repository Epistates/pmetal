//! Fused LoRA kernels for efficient training on Apple Silicon.
//!
//! This module provides Metal kernels that fuse LoRA forward and backward passes,
//! eliminating intermediate tensor allocations and kernel launch overhead.
//!
//! # Algorithm
//!
//! LoRA (Low-Rank Adaptation) decomposes a weight update as:
//! ```text
//! W' = W + scale * B @ A
//! y = x @ W'.T = x @ W.T + scale * (x @ A.T) @ B.T
//! ```
//!
//! The fused kernels compute this in a single pass, keeping intermediates
//! (like `x @ A.T`) in threadgroup memory for gradient computation.
//!
//! # Performance
//!
//! Compared to naive implementation with separate matmul calls:
//! - Forward: ~2x faster (single kernel vs 3 kernels)
//! - Backward: ~2x faster (fused gradient computation)
//! - Memory: ~50% less (no intermediate tensor allocations)
//!
//! # References
//!
//! - [LoRA Paper](https://arxiv.org/abs/2106.09685)
//! - [Unsloth Fused Kernels](https://github.com/unslothai/unsloth)

use half::f16;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

use crate::buffer::{BufferUsage, MetalBuffer};
use crate::context::MetalContext;
use crate::error::{MetalError, Result};
use crate::kernels::fused_sampler::AsMetalBuffer;

/// Configuration for fused LoRA operations.
#[derive(Debug, Clone)]
pub struct FusedLoraConfig {
    /// Batch size (flattened: batch * seq_len).
    pub batch_size: usize,

    /// Input features dimension.
    pub in_features: usize,

    /// Output features dimension.
    pub out_features: usize,

    /// LoRA rank.
    pub rank: usize,

    /// LoRA scaling factor (typically alpha / rank).
    pub scale: f32,
}

impl FusedLoraConfig {
    /// Create a new configuration.
    pub fn new(
        batch_size: usize,
        in_features: usize,
        out_features: usize,
        rank: usize,
        scale: f32,
    ) -> Self {
        Self {
            batch_size,
            in_features,
            out_features,
            rank,
            scale,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(MetalError::InvalidConfig("batch_size must be > 0".into()));
        }
        if self.in_features == 0 {
            return Err(MetalError::InvalidConfig("in_features must be > 0".into()));
        }
        if self.out_features == 0 {
            return Err(MetalError::InvalidConfig("out_features must be > 0".into()));
        }
        if self.rank == 0 {
            return Err(MetalError::InvalidConfig("rank must be > 0".into()));
        }
        if self.rank > 64 {
            return Err(MetalError::InvalidConfig(
                "rank must be <= 64 (kernel limitation)".into(),
            ));
        }
        Ok(())
    }

    /// Get the expected input size.
    pub fn input_size(&self) -> usize {
        self.batch_size * self.in_features
    }

    /// Get the expected output size.
    pub fn output_size(&self) -> usize {
        self.batch_size * self.out_features
    }

    /// Get the expected A matrix size.
    pub fn a_size(&self) -> usize {
        self.rank * self.in_features
    }

    /// Get the expected B matrix size.
    pub fn b_size(&self) -> usize {
        self.out_features * self.rank
    }

    /// Get the expected intermediate (xA) size.
    pub fn intermediate_size(&self) -> usize {
        self.batch_size * self.rank
    }

    /// Get the expected base weight size.
    pub fn weight_size(&self) -> usize {
        self.out_features * self.in_features
    }
}

/// Parameters passed to the Metal kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct FusedLoraParams {
    batch_size: u32,
    in_features: u32,
    out_features: u32,
    rank: u32,
    scale: f32,
}

impl From<&FusedLoraConfig> for FusedLoraParams {
    fn from(config: &FusedLoraConfig) -> Self {
        Self {
            batch_size: config.batch_size as u32,
            in_features: config.in_features as u32,
            out_features: config.out_features as u32,
            rank: config.rank as u32,
            scale: config.scale,
        }
    }
}

/// Output from fused LoRA forward pass.
#[derive(Debug)]
pub struct FusedLoraOutput {
    /// Output tensor [batch_size, out_features].
    pub output: MetalBuffer<f16>,

    /// Intermediate x @ A.T [batch_size, rank] for backward pass.
    /// Only present during training.
    pub intermediate: Option<MetalBuffer<f16>>,
}

/// Fused LoRA kernel executor.
///
/// This struct manages Metal pipelines and executes fused LoRA operations
/// efficiently on the GPU.
pub struct FusedLora {
    /// Metal context.
    ctx: Arc<MetalContext>,

    /// Configuration.
    config: FusedLoraConfig,
}

impl FusedLora {
    /// Create a new fused LoRA executor.
    pub fn new(ctx: Arc<MetalContext>, config: FusedLoraConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { ctx, config })
    }

    /// Get the configuration.
    pub fn config(&self) -> &FusedLoraConfig {
        &self.config
    }

    /// Returns
    ///
    /// Output tensor and intermediate for backward.
    pub fn forward<B: AsMetalBuffer>(
        &self,
        x: &B,
        weight: &B,
        lora_a: &B,
        lora_b: &B,
    ) -> Result<FusedLoraOutput> {
        // Validate input sizes
        self.validate_forward_inputs(x, weight, lora_a, lora_b)?;

        // Allocate output and intermediate buffers
        let output = MetalBuffer::new(&self.ctx, self.config.output_size(), BufferUsage::Shared)?;

        let intermediate = MetalBuffer::new(
            &self.ctx,
            self.config.intermediate_size(),
            BufferUsage::Shared,
        )?;

        // Execute kernel
        self.execute_forward(x, weight, lora_a, lora_b, &output, &intermediate)?;

        Ok(FusedLoraOutput {
            output,
            intermediate: Some(intermediate),
        })
    }

    /// Execute fused LoRA forward pass (inference mode).
    ///
    /// Same as forward but doesn't save intermediates.
    pub fn forward_inference(
        &self,
        x: &MetalBuffer<f16>,
        weight: &MetalBuffer<f16>,
        lora_a: &MetalBuffer<f16>,
        lora_b: &MetalBuffer<f16>,
    ) -> Result<MetalBuffer<f16>> {
        self.validate_forward_inputs(x, weight, lora_a, lora_b)?;

        let output = MetalBuffer::new(&self.ctx, self.config.output_size(), BufferUsage::Shared)?;

        self.execute_forward_inference(x, weight, lora_a, lora_b, &output)?;

        Ok(output)
    }

    /// Returns
    ///
    /// Tuple of (grad_a, grad_b).
    pub fn backward_ab<B: AsMetalBuffer>(
        &self,
        grad_output: &B,
        x: &B,
        intermediate: &B,
        lora_b: &B,
    ) -> Result<(MetalBuffer<f16>, MetalBuffer<f16>)> {
        // Validate sizes
        if grad_output.len() != self.config.output_size() {
            return Err(MetalError::DimensionMismatch {
                param: "grad_output",
                expected: self.config.output_size(),
                actual: grad_output.len(),
            });
        }
        if x.len() != self.config.input_size() {
            return Err(MetalError::DimensionMismatch {
                param: "x",
                expected: self.config.input_size(),
                actual: x.len(),
            });
        }
        if intermediate.len() != self.config.intermediate_size() {
            return Err(MetalError::DimensionMismatch {
                param: "intermediate",
                expected: self.config.intermediate_size(),
                actual: intermediate.len(),
            });
        }

        // Allocate gradient buffers
        let grad_a = MetalBuffer::zeros(&self.ctx, self.config.a_size(), BufferUsage::Shared)?;

        let grad_b = MetalBuffer::zeros(&self.ctx, self.config.b_size(), BufferUsage::Shared)?;

        // Execute backward kernels
        self.execute_backward_ab(grad_output, x, intermediate, lora_b, &grad_a, &grad_b)?;

        Ok((grad_a, grad_b))
    }

    /// Returns
    ///
    /// Input gradient [batch_size, in_features].
    pub fn backward_x<B: AsMetalBuffer>(
        &self,
        grad_output: &B,
        weight: &B,
        lora_a: &B,
        lora_b: &B,
    ) -> Result<MetalBuffer<f16>> {
        // Validate sizes
        if grad_output.len() != self.config.output_size() {
            return Err(MetalError::DimensionMismatch {
                param: "grad_output",
                expected: self.config.output_size(),
                actual: grad_output.len(),
            });
        }

        let grad_x = MetalBuffer::zeros(&self.ctx, self.config.input_size(), BufferUsage::Shared)?;

        self.execute_backward_x(grad_output, weight, lora_a, lora_b, &grad_x)?;

        Ok(grad_x)
    }

    /// Validate input sizes for forward pass.
    fn validate_forward_inputs<B: AsMetalBuffer>(
        &self,
        x: &B,
        weight: &B,
        lora_a: &B,
        lora_b: &B,
    ) -> Result<()> {
        if x.len() != self.config.input_size() {
            return Err(MetalError::DimensionMismatch {
                param: "x",
                expected: self.config.input_size(),
                actual: x.len(),
            });
        }
        if weight.len() != self.config.weight_size() {
            return Err(MetalError::DimensionMismatch {
                param: "weight",
                expected: self.config.weight_size(),
                actual: weight.len(),
            });
        }
        if lora_a.len() != self.config.a_size() {
            return Err(MetalError::DimensionMismatch {
                param: "lora_a",
                expected: self.config.a_size(),
                actual: lora_a.len(),
            });
        }
        if lora_b.len() != self.config.b_size() {
            return Err(MetalError::DimensionMismatch {
                param: "lora_b",
                expected: self.config.b_size(),
                actual: lora_b.len(),
            });
        }
        Ok(())
    }

    /// Execute the forward kernel.
    fn execute_forward<B: AsMetalBuffer>(
        &self,
        x: &B,
        weight: &B,
        lora_a: &B,
        lora_b: &B,
        output: &MetalBuffer<f16>,
        intermediate: &MetalBuffer<f16>,
    ) -> Result<()> {
        let function_name = "fused_lora_forward";

        // Auto-tune parameters
        let tuned_config = self.ctx.tuner().tune_lora_forward(
            &self.ctx,
            self.config.batch_size,
            self.config.in_features,
            self.config.out_features,
            self.config.rank,
        )?;

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            let mut constants = std::collections::HashMap::new();
            constants.insert(0, tuned_config.tile_m);
            constants.insert(1, tuned_config.tile_n);
            constants.insert(2, tuned_config.tile_k);
            cache.get_or_create_specialized_pipeline(
                self.ctx.device(),
                function_name,
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

        // Set buffers
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(lora_a.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(lora_b.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(intermediate.metal_buffer()), 0, 5);

            let params = FusedLoraParams::from(&self.config);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 6);
        }

        // Calculate grid size using tuned parameters
        let grid_size = MTLSize {
            width: self
                .config
                .batch_size
                .div_ceil(tuned_config.tile_m as usize),
            height: self
                .config
                .out_features
                .div_ceil(tuned_config.tile_n as usize),
            depth: 1,
        };

        // Threadgroup size must match kernel logic: [TILE_N, TILE_M/SIMD_SIZE, 1]
        let threadgroup_size = MTLSize {
            width: tuned_config.tile_n as usize,
            height: (tuned_config.tile_m as usize) / 32,
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

    /// Execute the inference forward kernel.
    fn execute_forward_inference(
        &self,
        x: &MetalBuffer<f16>,
        weight: &MetalBuffer<f16>,
        lora_a: &MetalBuffer<f16>,
        lora_b: &MetalBuffer<f16>,
        output: &MetalBuffer<f16>,
    ) -> Result<()> {
        let function_name = "lora_forward_inference";
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
            encoder.setBuffer_offset_atIndex(Some(x.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(lora_a.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(lora_b.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 4);

            let params = FusedLoraParams::from(&self.config);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);
        }

        let grid_size = MTLSize {
            width: self.config.batch_size,
            height: self.config.out_features.div_ceil(32),
            depth: 1,
        };

        let threadgroup_size = MTLSize {
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

    /// Execute the backward AB kernel.
    fn execute_backward_ab<B: AsMetalBuffer>(
        &self,
        grad_output: &B,
        x: &B,
        intermediate: &B,
        lora_b: &B,
        grad_a: &MetalBuffer<f16>,
        grad_b: &MetalBuffer<f16>,
    ) -> Result<()> {
        // First kernel: compute dB
        {
            let function_name = "fused_lora_backward_ab";
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
                encoder.setBuffer_offset_atIndex(Some(grad_output.metal_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(x.metal_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(intermediate.metal_buffer()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(lora_b.metal_buffer()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(grad_a.metal_buffer()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(grad_b.metal_buffer()), 0, 5);

                let params = FusedLoraParams::from(&self.config);
                let params_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 6);
            }

            const TILE_N: usize = 32;
            const TILE_K: usize = 32;

            let grid_size = MTLSize {
                width: self.config.out_features.div_ceil(TILE_N),
                height: self.config.rank.div_ceil(TILE_K),
                depth: 1,
            };

            let threadgroup_size = MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            };

            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            if let Some(error) = command_buffer.error() {
                return Err(MetalError::ExecutionFailed(error.to_string()));
            }
        }

        // Second kernel: compute dA
        {
            let function_name = "fused_lora_backward_a";
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
                encoder.setBuffer_offset_atIndex(Some(grad_output.metal_buffer()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(x.metal_buffer()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(lora_b.metal_buffer()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(grad_a.metal_buffer()), 0, 3);

                let params = FusedLoraParams::from(&self.config);
                let params_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
            }

            const TILE_K: usize = 32;
            const TILE_M: usize = 32;

            let grid_size = MTLSize {
                width: self.config.in_features.div_ceil(TILE_K),
                height: self.config.rank.div_ceil(TILE_M),
                depth: 1,
            };

            let threadgroup_size = MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            };

            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            if let Some(error) = command_buffer.error() {
                return Err(MetalError::ExecutionFailed(error.to_string()));
            }
        }

        Ok(())
    }

    /// Execute the backward X kernel.
    fn execute_backward_x<B: AsMetalBuffer>(
        &self,
        grad_output: &B,
        weight: &B,
        lora_a: &B,
        lora_b: &B,
        grad_x: &MetalBuffer<f16>,
    ) -> Result<()> {
        let function_name = "fused_lora_backward_x";
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
            encoder.setBuffer_offset_atIndex(Some(grad_output.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(lora_a.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(lora_b.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(grad_x.metal_buffer()), 0, 4);

            let params = FusedLoraParams::from(&self.config);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);
        }

        const TILE_M: usize = 32;
        const TILE_K: usize = 32;

        let grid_size = MTLSize {
            width: self.config.batch_size.div_ceil(TILE_M),
            height: self.config.in_features.div_ceil(TILE_K),
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: 32,
            height: 4,
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

impl std::fmt::Debug for FusedLora {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedLora")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> Arc<MetalContext> {
        Arc::new(MetalContext::new().expect("Failed to create Metal context"))
    }

    #[test]
    fn test_config_validation() {
        let valid = FusedLoraConfig::new(128, 512, 1024, 8, 2.0);
        assert!(valid.validate().is_ok());

        let invalid_rank = FusedLoraConfig::new(128, 512, 1024, 128, 2.0);
        assert!(invalid_rank.validate().is_err());

        let invalid_batch = FusedLoraConfig::new(0, 512, 1024, 8, 2.0);
        assert!(invalid_batch.validate().is_err());
    }

    #[test]
    fn test_fused_lora_creation() {
        let ctx = create_test_context();
        let config = FusedLoraConfig::new(128, 512, 1024, 8, 2.0);
        let lora = FusedLora::new(ctx, config);
        assert!(lora.is_ok());
    }

    #[test]
    fn test_config_sizes() {
        let config = FusedLoraConfig::new(128, 512, 1024, 8, 2.0);

        assert_eq!(config.input_size(), 128 * 512);
        assert_eq!(config.output_size(), 128 * 1024);
        assert_eq!(config.a_size(), 8 * 512);
        assert_eq!(config.b_size(), 1024 * 8);
        assert_eq!(config.intermediate_size(), 128 * 8);
        assert_eq!(config.weight_size(), 1024 * 512);
    }
}
