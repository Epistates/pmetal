//! Fused model merging kernels for maximum throughput.
//!
//! This module provides Metal kernels for batched model merging operations,
//! eliminating GPU-CPU synchronization overhead by processing multiple tensors
//! in a single command buffer.
//!
//! # Optimizations
//!
//! 1. **Batched Processing**: Process multiple tensors in single GPU dispatch
//! 2. **Online Thresholding**: O(n) streaming algorithm instead of O(n log n) sort
//! 3. **Fused Operations**: Combine magnitude + sparsify + consensus in one kernel
//!
//! # Example
//!
//! ```ignore
//! let merger = FusedMergeMetal::new(ctx.clone())?;
//! let mut batch = BatchedCommandBuffer::new(ctx)?;
//!
//! // Queue multiple tensor operations
//! for (tensor, density) in tensors.iter().zip(densities.iter()) {
//!     merger.queue_sparsify(&mut batch, tensor, output, *density)?;
//! }
//!
//! // Execute all at once (single GPU sync)
//! batch.execute()?;
//! ```

use std::ptr::NonNull;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::{MetalError, Result};
use crate::kernels::fused_training::BatchedCommandBuffer;
use crate::tuna::{MergeTunedConfig, Tuner};

// =============================================================================
// Tensor Info for Batched Processing
// =============================================================================

/// Metadata for batched tensor processing.
///
/// Enables processing multiple tensors with different sizes in a single dispatch.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TensorInfo {
    /// Offset into the flattened buffer.
    pub offset: u32,
    /// Number of elements in this tensor.
    pub size: u32,
    /// Density parameter for this tensor (0.0-1.0).
    pub density: f32,
    /// Computed threshold (filled by threshold kernel).
    pub threshold: f32,
}

/// Configuration for merge operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MergeConfig {
    /// Number of tensors in batch.
    pub num_tensors: u32,
    /// Total elements across all tensors.
    pub total_elements: u32,
    /// Epsilon for numerical stability.
    pub epsilon: f32,
    /// Padding for alignment.
    pub _pad: u32,
}

/// Configuration for TIES merge operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TiesConfig {
    /// Number of models being merged.
    pub num_models: u32,
    /// Number of elements per model tensor.
    pub elements_per_model: u32,
    /// Global scaling factor (lambda).
    pub lambda: f32,
    /// Epsilon for numerical stability.
    pub epsilon: f32,
}

/// Maximum number of models supported by fused TIES kernel.
/// Limited by threadgroup memory for sparse_task_vectors array in Metal shader.
pub const MAX_TIES_MODELS: usize = 16;

impl TiesConfig {
    /// Create a new TIES configuration.
    ///
    /// # Panics
    /// Panics if `num_models` exceeds `MAX_TIES_MODELS` (16).
    pub fn new(num_models: usize, elements_per_model: usize, lambda: f32) -> Self {
        assert!(
            num_models <= MAX_TIES_MODELS,
            "TIES merge supports at most {} models, got {}",
            MAX_TIES_MODELS,
            num_models
        );
        assert!(num_models > 0, "TIES merge requires at least 1 model");
        assert!(
            elements_per_model > 0,
            "elements_per_model must be greater than 0"
        );

        Self {
            num_models: num_models as u32,
            elements_per_model: elements_per_model as u32,
            lambda,
            epsilon: 1e-8,
        }
    }

    /// Try to create a new TIES configuration, returning None if invalid.
    pub fn try_new(num_models: usize, elements_per_model: usize, lambda: f32) -> Option<Self> {
        if num_models == 0 || num_models > MAX_TIES_MODELS || elements_per_model == 0 {
            return None;
        }

        Some(Self {
            num_models: num_models as u32,
            elements_per_model: elements_per_model as u32,
            lambda,
            epsilon: 1e-8,
        })
    }
}

// =============================================================================
// Fused Merge Operations
// =============================================================================

/// Fused model merging kernels.
///
/// Provides batched Metal kernels for:
/// - Magnitude computation
/// - Online threshold calculation
/// - Sparsification
/// - Sign consensus
pub struct FusedMergeMetal {
    ctx: Arc<MetalContext>,
    tuner: Tuner,
    tuned_config: Option<MergeTunedConfig>,
}

impl FusedMergeMetal {
    /// Create a new fused merge kernel handler.
    pub fn new(ctx: Arc<MetalContext>) -> Self {
        Self {
            ctx,
            tuner: Tuner::new(),
            tuned_config: None,
        }
    }

    /// Get the Metal context.
    pub fn context(&self) -> &Arc<MetalContext> {
        &self.ctx
    }

    /// Get the current tuned configuration.
    pub fn tuned_config(&self) -> Option<MergeTunedConfig> {
        self.tuned_config
    }

    /// Auto-tune for a specific problem size.
    ///
    /// Benchmarks different configurations and caches the optimal settings
    /// for the given problem size on the current hardware.
    ///
    /// # Arguments
    /// * `num_elements` - Total number of elements to process
    /// * `num_models` - Number of models being merged (for TIES operations)
    ///
    /// # Returns
    /// The optimal configuration for this problem size.
    pub fn tune_for_problem_size(
        &mut self,
        num_elements: usize,
        num_models: usize,
    ) -> Result<MergeTunedConfig> {
        let config = self.tuner.tune_merge(&self.ctx, num_elements, num_models)?;
        self.tuned_config = Some(config);
        Ok(config)
    }

    /// Set the tuned configuration manually.
    ///
    /// Use this when you want to use a specific configuration without auto-tuning.
    pub fn set_tuned_config(&mut self, config: MergeTunedConfig) {
        self.tuned_config = Some(config);
    }

    /// Get the effective threadgroup size for dispatches.
    pub(crate) fn effective_threads_per_group(&self) -> usize {
        self.tuned_config
            .map(|c| c.threads_per_group as usize)
            .unwrap_or(256)
    }

    /// Get the effective elements per thread.
    pub(crate) fn effective_elements_per_thread(&self) -> usize {
        self.tuned_config
            .map(|c| c.elements_per_thread as usize)
            .unwrap_or(1)
    }

    /// Queue magnitude computation for multiple tensors.
    ///
    /// Computes `|x|` for each element in the input tensors and stores
    /// the result in the output buffer.
    ///
    /// # Arguments
    /// * `batch` - Command buffer to queue into
    /// * `input` - Flattened input tensors (concatenated)
    /// * `output` - Output buffer for magnitudes (same layout as input)
    /// * `tensor_info` - Metadata for each tensor
    /// * `config` - Merge configuration
    pub fn queue_compute_magnitudes(
        &self,
        batch: &mut BatchedCommandBuffer,
        input: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        tensor_info: &MetalBuffer<TensorInfo>,
        config: &MergeConfig,
    ) -> Result<()> {
        let function_name = "fused_compute_magnitudes";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch
            .encoder()
            .ok_or(MetalError::CommandBufferCreation)?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(tensor_info.metal_buffer()), 0, 2);

            let config_ptr = NonNull::from(config).cast();
            encoder.setBytes_length_atIndex(config_ptr, std::mem::size_of::<MergeConfig>(), 3);
        }

        // Use tuned configuration for dispatch sizes
        let threads = self.effective_threads_per_group();
        let elements_per_thread = self.effective_elements_per_thread();
        let elements_per_group = threads * elements_per_thread;

        let grid_size = MTLSize {
            width: (config.total_elements as usize).div_ceil(elements_per_group),
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Queue partial sum computation for threshold estimation.
    ///
    /// Each threadgroup computes a partial histogram/sample of magnitudes
    /// that can be used to estimate the k-th percentile threshold.
    ///
    /// # Algorithm
    /// Uses reservoir sampling to maintain approximate distribution of magnitudes,
    /// enabling O(n) threshold computation instead of O(n log n) sorting.
    pub fn queue_partial_threshold(
        &self,
        batch: &mut BatchedCommandBuffer,
        magnitudes: &MetalBuffer<f32>,
        partial_samples: &MetalBuffer<f32>,
        tensor_info: &MetalBuffer<TensorInfo>,
        config: &MergeConfig,
        samples_per_tensor: usize,
    ) -> Result<()> {
        let function_name = "fused_partial_threshold";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch
            .encoder()
            .ok_or(MetalError::CommandBufferCreation)?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(magnitudes.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(partial_samples.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(tensor_info.metal_buffer()), 0, 2);

            let config_ptr = NonNull::from(config).cast();
            encoder.setBytes_length_atIndex(config_ptr, std::mem::size_of::<MergeConfig>(), 3);

            let samples = samples_per_tensor as u32;
            let samples_ptr = NonNull::from(&samples).cast();
            encoder.setBytes_length_atIndex(samples_ptr, std::mem::size_of::<u32>(), 4);
        }

        // One threadgroup per tensor for sampling
        let grid_size = MTLSize {
            width: config.num_tensors as usize,
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

    /// Queue sparsification with pre-computed thresholds.
    ///
    /// Applies mask: `output[i] = |input[i]| >= threshold ? input[i] : 0`
    ///
    /// # Arguments
    /// * `batch` - Command buffer to queue into
    /// * `input` - Input tensors (concatenated)
    /// * `output` - Output buffer (same layout)
    /// * `tensor_info` - Metadata with thresholds filled in
    /// * `config` - Merge configuration
    pub fn queue_apply_sparsification(
        &self,
        batch: &mut BatchedCommandBuffer,
        input: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        tensor_info: &MetalBuffer<TensorInfo>,
        config: &MergeConfig,
    ) -> Result<()> {
        let function_name = "fused_apply_sparsification";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch
            .encoder()
            .ok_or(MetalError::CommandBufferCreation)?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(tensor_info.metal_buffer()), 0, 2);

            let config_ptr = NonNull::from(config).cast();
            encoder.setBytes_length_atIndex(config_ptr, std::mem::size_of::<MergeConfig>(), 3);
        }

        // Use tuned configuration for dispatch sizes
        let threads = self.effective_threads_per_group();
        let elements_per_thread = self.effective_elements_per_thread();
        let elements_per_group = threads * elements_per_thread;

        let grid_size = MTLSize {
            width: (config.total_elements as usize).div_ceil(elements_per_group),
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Queue sign consensus computation for TIES merge.
    ///
    /// Computes weighted sign majority:
    /// `sign_consensus[i] = sign(sum(weight[j] * sign(tensor[j][i])))`
    ///
    /// # Arguments
    /// * `batch` - Command buffer to queue into
    /// * `tensors` - Stacked input tensors [num_models, total_elements]
    /// * `weights` - Weight per model [num_models]
    /// * `consensus` - Output consensus signs [total_elements]
    /// * `num_models` - Number of models being merged
    /// * `total_elements` - Elements per model
    pub fn queue_sign_consensus(
        &self,
        batch: &mut BatchedCommandBuffer,
        tensors: &MetalBuffer<f32>,
        weights: &MetalBuffer<f32>,
        consensus: &MetalBuffer<f32>,
        num_models: usize,
        total_elements: usize,
    ) -> Result<()> {
        let function_name = "fused_sign_consensus";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch
            .encoder()
            .ok_or(MetalError::CommandBufferCreation)?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(tensors.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(consensus.metal_buffer()), 0, 2);

            let num_models_u32 = num_models as u32;
            let num_models_ptr = NonNull::from(&num_models_u32).cast();
            encoder.setBytes_length_atIndex(num_models_ptr, std::mem::size_of::<u32>(), 3);

            let total_u32 = total_elements as u32;
            let total_ptr = NonNull::from(&total_u32).cast();
            encoder.setBytes_length_atIndex(total_ptr, std::mem::size_of::<u32>(), 4);
        }

        // Use tuned configuration for dispatch sizes
        let threads = self.effective_threads_per_group();
        let elements_per_thread = self.effective_elements_per_thread();
        let elements_per_group = threads * elements_per_thread;

        let grid_size = MTLSize {
            width: total_elements.div_ceil(elements_per_group),
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Queue task vector computation.
    ///
    /// Computes `task_vector = fine_tuned - base` for each tensor.
    pub fn queue_task_vectors(
        &self,
        batch: &mut BatchedCommandBuffer,
        fine_tuned: &MetalBuffer<f32>,
        base: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        total_elements: usize,
    ) -> Result<()> {
        let function_name = "fused_task_vectors";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch
            .encoder()
            .ok_or(MetalError::CommandBufferCreation)?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(fine_tuned.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(base.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 2);

            let total_u32 = total_elements as u32;
            let total_ptr = NonNull::from(&total_u32).cast();
            encoder.setBytes_length_atIndex(total_ptr, std::mem::size_of::<u32>(), 3);
        }

        // Use tuned configuration for dispatch sizes
        let threads = self.effective_threads_per_group();
        let elements_per_thread = self.effective_elements_per_thread();
        let elements_per_group = threads * elements_per_thread;

        let grid_size = MTLSize {
            width: total_elements.div_ceil(elements_per_group),
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Queue weighted sum for final merge.
    ///
    /// Computes `output = base + lambda * sum(weight[i] * tensor[i])`
    #[allow(clippy::too_many_arguments)]
    pub fn queue_weighted_sum(
        &self,
        batch: &mut BatchedCommandBuffer,
        tensors: &MetalBuffer<f32>,
        weights: &MetalBuffer<f32>,
        base: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        num_models: usize,
        total_elements: usize,
        lambda: f32,
    ) -> Result<()> {
        let function_name = "fused_weighted_sum";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch
            .encoder()
            .ok_or(MetalError::CommandBufferCreation)?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(tensors.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(base.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 3);

            let num_models_u32 = num_models as u32;
            let num_models_ptr = NonNull::from(&num_models_u32).cast();
            encoder.setBytes_length_atIndex(num_models_ptr, std::mem::size_of::<u32>(), 4);

            let total_u32 = total_elements as u32;
            let total_ptr = NonNull::from(&total_u32).cast();
            encoder.setBytes_length_atIndex(total_ptr, std::mem::size_of::<u32>(), 5);

            let lambda_ptr = NonNull::from(&lambda).cast();
            encoder.setBytes_length_atIndex(lambda_ptr, std::mem::size_of::<f32>(), 6);
        }

        // Use tuned configuration for dispatch sizes
        let threads = self.effective_threads_per_group();
        let elements_per_thread = self.effective_elements_per_thread();
        let elements_per_group = threads * elements_per_thread;

        let grid_size = MTLSize {
            width: total_elements.div_ceil(elements_per_group),
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Queue fused TIES merge kernel.
    ///
    /// Performs complete TIES merge in a single kernel dispatch:
    /// 1. Compute task vectors (tensor - base)
    /// 2. Apply magnitude threshold (sparsification)
    /// 3. Compute sign consensus
    /// 4. Apply consensus mask and weighted sum
    /// 5. Scale by lambda and add to base
    ///
    /// # Arguments
    /// * `batch` - Command buffer to queue into
    /// * `tensors` - Stacked fine-tuned models [num_models, elements]
    /// * `base` - Base model [elements]
    /// * `weights` - Per-model weights [num_models]
    /// * `thresholds` - Pre-computed magnitude thresholds [num_models]
    /// * `output` - Output buffer [elements]
    /// * `config` - TIES configuration
    #[allow(clippy::too_many_arguments)]
    pub fn queue_fused_ties_merge(
        &self,
        batch: &mut BatchedCommandBuffer,
        tensors: &MetalBuffer<f32>,
        base: &MetalBuffer<f32>,
        weights: &MetalBuffer<f32>,
        thresholds: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        config: &TiesConfig,
    ) -> Result<()> {
        let function_name = "fused_ties_merge";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch
            .encoder()
            .ok_or(MetalError::CommandBufferCreation)?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(tensors.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(base.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weights.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(thresholds.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 4);

            let config_ptr = NonNull::from(config).cast();
            encoder.setBytes_length_atIndex(config_ptr, std::mem::size_of::<TiesConfig>(), 5);
        }

        // Use tuned configuration for dispatch sizes
        let threads = self.effective_threads_per_group();
        let elements_per_thread = self.effective_elements_per_thread();
        let elements_per_group = threads * elements_per_thread;

        let grid_size = MTLSize {
            width: (config.elements_per_model as usize).div_ceil(elements_per_group),
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Queue linear merge kernel.
    ///
    /// Computes simple weighted average: `output = sum(weight[i] * tensor[i])`
    ///
    /// # Arguments
    /// * `batch` - Command buffer to queue into
    /// * `tensors` - Stacked models [num_models, elements]
    /// * `weights` - Per-model weights [num_models]
    /// * `output` - Output buffer [elements]
    /// * `num_models` - Number of models
    /// * `total_elements` - Elements per model
    pub fn queue_linear_merge(
        &self,
        batch: &mut BatchedCommandBuffer,
        tensors: &MetalBuffer<f32>,
        weights: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        num_models: usize,
        total_elements: usize,
    ) -> Result<()> {
        let function_name = "fused_linear_merge";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch
            .encoder()
            .ok_or(MetalError::CommandBufferCreation)?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(tensors.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 2);

            let num_models_u32 = num_models as u32;
            let num_models_ptr = NonNull::from(&num_models_u32).cast();
            encoder.setBytes_length_atIndex(num_models_ptr, std::mem::size_of::<u32>(), 3);

            let total_u32 = total_elements as u32;
            let total_ptr = NonNull::from(&total_u32).cast();
            encoder.setBytes_length_atIndex(total_ptr, std::mem::size_of::<u32>(), 4);
        }

        // Use tuned configuration for dispatch sizes
        let threads = self.effective_threads_per_group();
        let elements_per_thread = self.effective_elements_per_thread();
        let elements_per_group = threads * elements_per_thread;

        let grid_size = MTLSize {
            width: total_elements.div_ceil(elements_per_group),
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Queue SLERP merge kernel.
    ///
    /// Performs spherical linear interpolation between two tensors:
    /// `slerp(a, b, t) = sin((1-t)*omega)/sin(omega) * a + sin(t*omega)/sin(omega) * b`
    ///
    /// # Arguments
    /// * `batch` - Command buffer to queue into
    /// * `tensor_a` - First tensor [elements]
    /// * `tensor_b` - Second tensor [elements]
    /// * `output` - Output buffer [elements]
    /// * `t` - Interpolation factor (0.0 = a, 1.0 = b)
    /// * `omega` - Pre-computed angle between vectors
    /// * `sin_omega` - Pre-computed sin(omega)
    /// * `total_elements` - Number of elements
    #[allow(clippy::too_many_arguments)]
    pub fn queue_slerp_merge(
        &self,
        batch: &mut BatchedCommandBuffer,
        tensor_a: &MetalBuffer<f32>,
        tensor_b: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        t: f32,
        omega: f32,
        sin_omega: f32,
        total_elements: usize,
    ) -> Result<()> {
        let function_name = "fused_slerp_merge";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch
            .encoder()
            .ok_or(MetalError::CommandBufferCreation)?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(tensor_a.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(tensor_b.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 2);

            let t_ptr = NonNull::from(&t).cast();
            encoder.setBytes_length_atIndex(t_ptr, std::mem::size_of::<f32>(), 3);

            let omega_ptr = NonNull::from(&omega).cast();
            encoder.setBytes_length_atIndex(omega_ptr, std::mem::size_of::<f32>(), 4);

            let sin_omega_ptr = NonNull::from(&sin_omega).cast();
            encoder.setBytes_length_atIndex(sin_omega_ptr, std::mem::size_of::<f32>(), 5);

            let total_u32 = total_elements as u32;
            let total_ptr = NonNull::from(&total_u32).cast();
            encoder.setBytes_length_atIndex(total_ptr, std::mem::size_of::<u32>(), 6);
        }

        // Use tuned configuration for dispatch sizes
        let threads = self.effective_threads_per_group();
        let elements_per_thread = self.effective_elements_per_thread();
        let elements_per_group = threads * elements_per_thread;

        let grid_size = MTLSize {
            width: total_elements.div_ceil(elements_per_group),
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Queue DARE sparsification kernel.
    ///
    /// Randomly drops elements with probability (1-density) and rescales remaining
    /// elements by 1/density for unbiased estimation.
    ///
    /// # Arguments
    /// * `batch` - Command buffer to queue into
    /// * `input` - Input tensor [elements]
    /// * `output` - Output buffer [elements]
    /// * `density` - Fraction of elements to keep (0.0-1.0)
    /// * `total_elements` - Number of elements
    /// * `seed` - Random seed for reproducibility
    #[allow(clippy::too_many_arguments)]
    pub fn queue_dare_sparsify(
        &self,
        batch: &mut BatchedCommandBuffer,
        input: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        density: f32,
        total_elements: usize,
        seed: u32,
    ) -> Result<()> {
        let function_name = "fused_dare_sparsify";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch
            .encoder()
            .ok_or(MetalError::CommandBufferCreation)?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 1);

            let density_ptr = NonNull::from(&density).cast();
            encoder.setBytes_length_atIndex(density_ptr, std::mem::size_of::<f32>(), 2);

            let total_u32 = total_elements as u32;
            let total_ptr = NonNull::from(&total_u32).cast();
            encoder.setBytes_length_atIndex(total_ptr, std::mem::size_of::<u32>(), 3);

            let seed_ptr = NonNull::from(&seed).cast();
            encoder.setBytes_length_atIndex(seed_ptr, std::mem::size_of::<u32>(), 4);
        }

        // Use tuned configuration for dispatch sizes
        let threads = self.effective_threads_per_group();
        let elements_per_thread = self.effective_elements_per_thread();
        let elements_per_group = threads * elements_per_thread;

        let grid_size = MTLSize {
            width: total_elements.div_ceil(elements_per_group),
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }

    /// Queue SLERP dot product and norm computation (partial reduction).
    ///
    /// Computes partial sums for dot(a, b), ||a||^2, and ||b||^2 that can be
    /// reduced on CPU to compute omega for SLERP.
    ///
    /// # Arguments
    /// * `batch` - Command buffer to queue into
    /// * `tensor_a` - First tensor [elements]
    /// * `tensor_b` - Second tensor [elements]
    /// * `partial_results` - Output for partial sums [3 * num_threadgroups]
    /// * `total_elements` - Number of elements
    pub fn queue_slerp_dot_norm(
        &self,
        batch: &mut BatchedCommandBuffer,
        tensor_a: &MetalBuffer<f32>,
        tensor_b: &MetalBuffer<f32>,
        partial_results: &MetalBuffer<f32>,
        total_elements: usize,
    ) -> Result<()> {
        let function_name = "fused_slerp_dot_norm";
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), function_name, None)?
        };

        let encoder = batch
            .encoder()
            .ok_or(MetalError::CommandBufferCreation)?;
        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(tensor_a.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(tensor_b.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(partial_results.metal_buffer()), 0, 2);

            let total_u32 = total_elements as u32;
            let total_ptr = NonNull::from(&total_u32).cast();
            encoder.setBytes_length_atIndex(total_ptr, std::mem::size_of::<u32>(), 3);
        }

        // Use tuned configuration for dispatch sizes
        let threads = self.effective_threads_per_group();
        let elements_per_thread = self.effective_elements_per_thread();
        let elements_per_group = threads * elements_per_thread;
        let num_threadgroups = total_elements.div_ceil(elements_per_group);

        let grid_size = MTLSize {
            width: num_threadgroups,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        batch.add_dispatch();

        Ok(())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Build tensor info for batched processing.
pub fn build_tensor_info(sizes: &[usize], densities: &[f32]) -> Vec<TensorInfo> {
    let mut offset = 0u32;
    sizes
        .iter()
        .zip(densities.iter())
        .map(|(&size, &density)| {
            let info = TensorInfo {
                offset,
                size: size as u32,
                density,
                threshold: 0.0, // Filled by threshold kernel
            };
            offset += size as u32;
            info
        })
        .collect()
}

/// Build merge config from tensor info.
pub fn build_merge_config(tensor_info: &[TensorInfo], epsilon: f32) -> MergeConfig {
    let total_elements: u32 = tensor_info.iter().map(|t| t.size).sum();
    MergeConfig {
        num_tensors: tensor_info.len() as u32,
        total_elements,
        epsilon,
        _pad: 0,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_info_building() {
        let sizes = vec![100, 200, 50];
        let densities = vec![0.5, 0.3, 0.7];
        let info = build_tensor_info(&sizes, &densities);

        assert_eq!(info.len(), 3);
        assert_eq!(info[0].offset, 0);
        assert_eq!(info[0].size, 100);
        assert!((info[0].density - 0.5).abs() < 1e-6);

        assert_eq!(info[1].offset, 100);
        assert_eq!(info[1].size, 200);
        assert!((info[1].density - 0.3).abs() < 1e-6);

        assert_eq!(info[2].offset, 300);
        assert_eq!(info[2].size, 50);
        assert!((info[2].density - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_merge_config_building() {
        let sizes = vec![100, 200, 50];
        let densities = vec![0.5, 0.3, 0.7];
        let tensor_info = build_tensor_info(&sizes, &densities);
        let config = build_merge_config(&tensor_info, 1e-8);

        assert_eq!(config.num_tensors, 3);
        assert_eq!(config.total_elements, 350);
        assert!((config.epsilon - 1e-8).abs() < 1e-12);
    }

    #[test]
    fn test_fused_merge_creation() {
        let ctx = Arc::new(MetalContext::new().expect("Failed to create Metal context"));
        let merger = FusedMergeMetal::new(ctx.clone());
        assert!(Arc::ptr_eq(merger.context(), &ctx));
    }

    #[test]
    fn test_merge_auto_tuning() {
        let ctx = Arc::new(MetalContext::new().expect("Failed to create Metal context"));
        let mut merger = FusedMergeMetal::new(ctx);

        // Initially no config
        assert!(merger.tuned_config().is_none());

        // Tune for a specific problem size
        let config = merger
            .tune_for_problem_size(1_000_000, 4)
            .expect("Tuning failed");

        // Config should now be set
        assert!(merger.tuned_config().is_some());
        assert_eq!(merger.tuned_config().unwrap(), config);

        // Verify config values are reasonable
        assert!(config.threads_per_group >= 128);
        assert!(config.threads_per_group <= 1024);
        assert!(config.elements_per_thread >= 1);
        assert!(config.elements_per_thread <= 16);

        // Test manual config setting
        let manual_config = MergeTunedConfig {
            threads_per_group: 512,
            elements_per_thread: 8,
            use_simd: true,
        };
        merger.set_tuned_config(manual_config);
        assert_eq!(merger.tuned_config().unwrap(), manual_config);
        assert_eq!(merger.effective_threads_per_group(), 512);
        assert_eq!(merger.effective_elements_per_thread(), 8);
    }

    #[test]
    fn test_tuned_dispatch_sizes() {
        let ctx = Arc::new(MetalContext::new().expect("Failed to create Metal context"));
        let mut merger = FusedMergeMetal::new(ctx);

        // Default values
        assert_eq!(merger.effective_threads_per_group(), 256);
        assert_eq!(merger.effective_elements_per_thread(), 1);

        // After tuning
        merger.set_tuned_config(MergeTunedConfig {
            threads_per_group: 128,
            elements_per_thread: 4,
            use_simd: true,
        });
        assert_eq!(merger.effective_threads_per_group(), 128);
        assert_eq!(merger.effective_elements_per_thread(), 4);
    }
}
