//! MoE (Mixture of Experts) Metal kernels.
//!
//! This module provides GPU-accelerated operations for Mixture of Experts models:
//!
//! - **Routing**: TopK expert selection from router logits
//! - **Grouped GEMM**: Batched matrix multiplication with expert-specific weights
//! - **Permutation**: Token reordering by expert assignment
//!
//! # Architecture
//!
//! MoE models route each token to a subset of experts (topk). The efficient
//! implementation requires:
//!
//! 1. **Routing Phase**: Compute softmax/sigmoid over router logits, select top-k experts
//! 2. **Permutation Phase**: Group tokens by assigned expert for memory locality
//! 3. **GEMM Phase**: Batched matrix multiplication with gather/scatter
//! 4. **Merge Phase**: Combine expert outputs weighted by routing scores
//!
//! # Example
//!
//! ```ignore
//! let config = MoeConfig::new(num_tokens, num_experts, topk, hidden_size, intermediate_size);
//! let moe = MoeKernel::new(ctx, config)?;
//!
//! // Route tokens to experts
//! let routing = moe.route(&router_logits)?;
//!
//! // Forward through experts (gate_up projection)
//! let hidden = moe.forward_gate_up(&input, &expert_weights, &routing)?;
//!
//! // Apply SwiGLU activation (external)
//!
//! // Forward through experts (down projection)
//! let output = moe.forward_down(&hidden_activated, &down_weights, &routing)?;
//!
//! // Merge expert outputs
//! let merged = moe.merge(&output, &routing.topk_weights)?;
//! ```

use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::{
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result},
};

/// Configuration for MoE kernel.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Number of input tokens.
    pub num_tokens: usize,

    /// Number of experts.
    pub num_experts: usize,

    /// Number of experts per token (typically 2 or 8).
    pub topk: usize,

    /// Input hidden dimension.
    pub hidden_size: usize,

    /// Intermediate dimension (MLP hidden size).
    pub intermediate_size: usize,

    /// Use sigmoid activation for routing (vs softmax).
    pub use_sigmoid: bool,

    /// Renormalize topk weights to sum to 1.
    pub renormalize: bool,
}

impl MoeConfig {
    /// Create a new MoE configuration.
    pub fn new(
        num_tokens: usize,
        num_experts: usize,
        topk: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Self {
        Self {
            num_tokens,
            num_experts,
            topk,
            hidden_size,
            intermediate_size,
            use_sigmoid: false,
            renormalize: true,
        }
    }

    /// Use sigmoid activation for routing.
    pub fn with_sigmoid(mut self) -> Self {
        self.use_sigmoid = true;
        self
    }

    /// Disable weight renormalization.
    pub fn without_renormalize(mut self) -> Self {
        self.renormalize = false;
        self
    }

    /// Total number of token-expert pairs.
    pub fn total_tokens(&self) -> usize {
        self.num_tokens * self.topk
    }
}

/// Routing result from expert selection.
#[derive(Debug)]
pub struct MoeRouting {
    /// TopK weights [num_tokens, topk].
    pub topk_weights: MetalBuffer<f32>,

    /// TopK expert IDs [num_tokens, topk].
    pub topk_ids: MetalBuffer<u32>,

    /// Token counts per expert [num_experts].
    pub token_counts: MetalBuffer<u32>,

    /// Cumulative offsets per expert [num_experts + 1].
    pub expert_offsets: MetalBuffer<u32>,

    /// Gather indices: sorted -> original [total_tokens].
    pub gather_indices: MetalBuffer<u32>,

    /// Scatter indices: original -> sorted [total_tokens].
    pub scatter_indices: MetalBuffer<u32>,
}

/// Output from grouped GEMM.
#[derive(Debug)]
pub struct MoeGemmOutput {
    /// Output tensor [total_tokens, output_dim].
    pub output: MetalBuffer<f32>,
}

/// MoE kernel for expert routing and grouped GEMM.
pub struct MoeKernel {
    ctx: Arc<MetalContext>,
    config: MoeConfig,
}

impl MoeKernel {
    /// Create a new MoE kernel.
    pub fn new(ctx: Arc<MetalContext>, config: MoeConfig) -> Result<Self> {
        Ok(Self { ctx, config })
    }

    /// Get the configuration.
    pub fn config(&self) -> &MoeConfig {
        &self.config
    }

    /// Perform expert routing.
    ///
    /// # Arguments
    ///
    /// * `router_logits` - Router output [num_tokens, num_experts]
    ///
    /// # Returns
    ///
    /// Routing information for subsequent GEMM operations.
    pub fn route(&self, router_logits: &MetalBuffer<f32>) -> Result<MoeRouting> {
        // Validate input
        let expected_size = self.config.num_tokens * self.config.num_experts;
        if router_logits.len() != expected_size {
            return Err(MetalError::DimensionMismatch {
                param: "router_logits",
                expected: expected_size,
                actual: router_logits.len(),
            });
        }

        let total_tokens = self.config.total_tokens();

        // Allocate outputs
        let topk_weights = MetalBuffer::new(
            &self.ctx,
            self.config.num_tokens * self.config.topk,
            BufferUsage::Shared,
        )?;
        let topk_ids = MetalBuffer::new(
            &self.ctx,
            self.config.num_tokens * self.config.topk,
            BufferUsage::Shared,
        )?;
        let token_counts =
            MetalBuffer::zeros(&self.ctx, self.config.num_experts, BufferUsage::Shared)?;
        let expert_offsets =
            MetalBuffer::zeros(&self.ctx, self.config.num_experts + 1, BufferUsage::Shared)?;
        let gather_indices = MetalBuffer::new(&self.ctx, total_tokens, BufferUsage::Shared)?;
        let scatter_indices = MetalBuffer::new(&self.ctx, total_tokens, BufferUsage::Shared)?;

        // Execute routing kernels
        self.execute_topk_selection(router_logits, &topk_weights, &topk_ids)?;
        self.execute_compute_indices(&topk_ids, &token_counts, &expert_offsets)?;
        self.execute_sort_indices(
            &topk_ids,
            &expert_offsets,
            &gather_indices,
            &scatter_indices,
        )?;

        Ok(MoeRouting {
            topk_weights,
            topk_ids,
            token_counts,
            expert_offsets,
            gather_indices,
            scatter_indices,
        })
    }

    /// Forward pass through experts (gate+up projection).
    ///
    /// # Arguments
    ///
    /// * `input` - Input hidden states [num_tokens, hidden_size]
    /// * `weights` - Expert weights [num_experts, intermediate_size, hidden_size]
    /// * `routing` - Routing information from `route()`
    /// * `fuse_mul` - Fuse weight multiplication
    ///
    /// # Returns
    ///
    /// Output tensor [total_tokens, intermediate_size]
    pub fn forward(
        &self,
        input: &MetalBuffer<f32>,
        weights: &MetalBuffer<f32>,
        routing: &MoeRouting,
        fuse_mul: bool,
    ) -> Result<MoeGemmOutput> {
        // Validate input
        let expected_input = self.config.num_tokens * self.config.hidden_size;
        if input.len() != expected_input {
            return Err(MetalError::DimensionMismatch {
                param: "input",
                expected: expected_input,
                actual: input.len(),
            });
        }

        let expected_weights =
            self.config.num_experts * self.config.intermediate_size * self.config.hidden_size;
        if weights.len() != expected_weights {
            return Err(MetalError::DimensionMismatch {
                param: "weights",
                expected: expected_weights,
                actual: weights.len(),
            });
        }

        // Allocate output
        let output_size = self.config.total_tokens() * self.config.intermediate_size;
        let output = MetalBuffer::new(&self.ctx, output_size, BufferUsage::Shared)?;

        self.execute_grouped_gemm_forward(
            input,
            weights,
            &output,
            routing,
            self.config.intermediate_size,
            true,  // permute_x: gather tokens by expert
            false, // permute_y: keep in expert order
            fuse_mul,
        )?;

        Ok(MoeGemmOutput { output })
    }

    /// Forward pass through experts (down projection) with output permutation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input hidden states [total_tokens, intermediate_size]
    /// * `weights` - Expert weights [num_experts, hidden_size, intermediate_size]
    /// * `routing` - Routing information from `route()`
    /// * `fuse_mul` - Fuse weight multiplication
    ///
    /// # Returns
    ///
    /// Output tensor [total_tokens, hidden_size] permuted back to token order
    pub fn forward_down(
        &self,
        input: &MetalBuffer<f32>,
        weights: &MetalBuffer<f32>,
        routing: &MoeRouting,
        fuse_mul: bool,
    ) -> Result<MoeGemmOutput> {
        // Validate input
        let expected_input = self.config.total_tokens() * self.config.intermediate_size;
        if input.len() != expected_input {
            return Err(MetalError::DimensionMismatch {
                param: "input",
                expected: expected_input,
                actual: input.len(),
            });
        }

        let expected_weights =
            self.config.num_experts * self.config.hidden_size * self.config.intermediate_size;
        if weights.len() != expected_weights {
            return Err(MetalError::DimensionMismatch {
                param: "weights",
                expected: expected_weights,
                actual: weights.len(),
            });
        }

        // Allocate output
        let output_size = self.config.total_tokens() * self.config.hidden_size;
        let output = MetalBuffer::new(&self.ctx, output_size, BufferUsage::Shared)?;

        self.execute_grouped_gemm_forward(
            input,
            weights,
            &output,
            routing,
            self.config.hidden_size,
            false, // permute_x: input already in expert order
            true,  // permute_y: scatter output to token order
            fuse_mul,
        )?;

        Ok(MoeGemmOutput { output })
    }

    fn execute_topk_selection(
        &self,
        router_logits: &MetalBuffer<f32>,
        topk_weights: &MetalBuffer<f32>,
        topk_ids: &MetalBuffer<u32>,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "moe_topk_selection", None)?
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
            encoder.setBuffer_offset_atIndex(Some(router_logits.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(topk_weights.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(topk_ids.metal_buffer()), 0, 2);

            let params = MoeRoutingParams {
                num_tokens: self.config.num_tokens as u32,
                num_experts: self.config.num_experts as u32,
                topk: self.config.topk as u32,
                use_sigmoid: self.config.use_sigmoid as u32,
                renormalize: self.config.renormalize as u32,
            };
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 3);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: 1,
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

    fn execute_compute_indices(
        &self,
        topk_ids: &MetalBuffer<u32>,
        token_counts: &MetalBuffer<u32>,
        expert_offsets: &MetalBuffer<u32>,
    ) -> Result<()> {
        // First pass: compute token counts per expert
        let pipeline_count = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "moe_compute_indices", None)?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline_count);

        let total_tokens = self.config.total_tokens();

        // Temporary gather indices (unused in this phase)
        let temp_gather = MetalBuffer::<u32>::new(&self.ctx, total_tokens, BufferUsage::Shared)?;

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(topk_ids.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(token_counts.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(temp_gather.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(expert_offsets.metal_buffer()), 0, 3);

            let params = MoeRoutingParams {
                num_tokens: self.config.num_tokens as u32,
                num_experts: self.config.num_experts as u32,
                topk: self.config.topk as u32,
                use_sigmoid: 0,
                renormalize: 0,
            };
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
        }

        let grid_size = objc2_metal::MTLSize {
            width: total_tokens,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: 256,
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

        // Second pass: compute prefix sum
        let pipeline_prefix = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "moe_compute_expert_offsets", None)?
        };

        let command_buffer2 = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder2 = command_buffer2
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder2.setComputePipelineState(&pipeline_prefix);

        unsafe {
            encoder2.setBuffer_offset_atIndex(Some(token_counts.metal_buffer()), 0, 0);
            encoder2.setBuffer_offset_atIndex(Some(expert_offsets.metal_buffer()), 0, 1);

            let num_experts = self.config.num_experts as u32;
            let num_experts_ptr = NonNull::from(&num_experts).cast();
            encoder2.setBytes_length_atIndex(
                num_experts_ptr,
                std::mem::size_of_val(&num_experts),
                2,
            );
        }

        let grid_size = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        encoder2.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder2.endEncoding();
        command_buffer2.commit();
        command_buffer2.waitUntilCompleted();

        if let Some(error) = command_buffer2.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    fn execute_sort_indices(
        &self,
        topk_ids: &MetalBuffer<u32>,
        expert_offsets: &MetalBuffer<u32>,
        gather_indices: &MetalBuffer<u32>,
        scatter_indices: &MetalBuffer<u32>,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "moe_sort_indices", None)?
        };

        let total_tokens = self.config.total_tokens();

        // Temporary counters for atomic claiming
        let expert_counters =
            MetalBuffer::<u32>::zeros(&self.ctx, self.config.num_experts, BufferUsage::Shared)?;

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(topk_ids.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(expert_offsets.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(expert_counters.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(gather_indices.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(scatter_indices.metal_buffer()), 0, 4);

            let params = MoeRoutingParams {
                num_tokens: self.config.num_tokens as u32,
                num_experts: self.config.num_experts as u32,
                topk: self.config.topk as u32,
                use_sigmoid: 0,
                renormalize: 0,
            };
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);
        }

        let grid_size = objc2_metal::MTLSize {
            width: total_tokens,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: 256,
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

    #[allow(clippy::too_many_arguments)]
    fn execute_grouped_gemm_forward(
        &self,
        input: &MetalBuffer<f32>,
        weights: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
        routing: &MoeRouting,
        output_dim: usize,
        permute_x: bool,
        permute_y: bool,
        fuse_mul: bool,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "grouped_gemm_forward", None)?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        // Compute total tiles needed
        let block_m = 64;
        let block_n = 64;

        // Estimate total tiles across all experts
        let total_tokens = self.config.total_tokens();
        let num_m_tiles = (total_tokens + block_m - 1) / block_m;
        let num_n_tiles = (output_dim + block_n - 1) / block_n;
        let total_tiles = num_m_tiles * num_n_tiles;

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(routing.expert_offsets.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(routing.gather_indices.metal_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(routing.scatter_indices.metal_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(routing.topk_weights.metal_buffer()), 0, 6);

            let input_dim = if permute_x {
                self.config.hidden_size
            } else {
                self.config.intermediate_size
            };

            let params = GroupedGemmParams {
                total_tokens: total_tokens as u32,
                num_experts: self.config.num_experts as u32,
                hidden_size: input_dim as u32,
                intermediate: output_dim as u32,
                topk: self.config.topk as u32,
                permute_x: permute_x as u32,
                permute_y: permute_y as u32,
                fuse_mul: fuse_mul as u32,
            };
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 7);

            // Threadgroup memory for tile accumulation
            let scratch_size = block_m * block_n * std::mem::size_of::<f32>();
            encoder.setThreadgroupMemoryLength_atIndex(scratch_size, 0);
        }

        let grid_size = objc2_metal::MTLSize {
            width: total_tiles,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: 16,
            height: 16,
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

/// Parameters passed to MoE routing kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MoeRoutingParams {
    num_tokens: u32,
    num_experts: u32,
    topk: u32,
    use_sigmoid: u32,
    renormalize: u32,
}

/// Parameters passed to grouped GEMM kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GroupedGemmParams {
    total_tokens: u32,
    num_experts: u32,
    hidden_size: u32,
    intermediate: u32,
    topk: u32,
    permute_x: u32,
    permute_y: u32,
    fuse_mul: u32,
}

impl std::fmt::Debug for MoeKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeKernel")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_config() {
        let config = MoeConfig::new(128, 8, 2, 512, 2048);

        assert_eq!(config.num_tokens, 128);
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.topk, 2);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.intermediate_size, 2048);
        assert_eq!(config.total_tokens(), 256);
        assert!(!config.use_sigmoid);
        assert!(config.renormalize);
    }

    #[test]
    fn test_moe_config_with_sigmoid() {
        let config = MoeConfig::new(128, 8, 2, 512, 2048).with_sigmoid();

        assert!(config.use_sigmoid);
    }
}
