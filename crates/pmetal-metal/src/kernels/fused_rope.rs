//! Fused RoPE (Rotary Position Embedding) Metal kernel.
//!
//! Provides high-performance rotary position embeddings with:
//!
//! - **In-place rotation**: Modifies input directly without allocation
//! - **Custom position IDs**: Essential for sequence packing
//! - **Fused QK RoPE**: Apply RoPE to both Q and K in single pass
//! - **Precomputed cache**: For efficient batched inference
//!
//! # Performance Benefits
//!
//! - No intermediate tensor allocations
//! - Single kernel launch for QK RoPE
//! - Amortized sin/cos computation across heads
//! - Optimized for Apple Silicon
//!
//! # Example
//!
//! ```ignore
//! // In-place RoPE with sequential positions
//! let rope = FusedRoPE::new(ctx, config)?;
//! rope.apply_inplace(&mut x)?;
//!
//! // With custom position IDs (sequence packing)
//! rope.apply_with_positions(&mut x, &position_ids)?;
//!
//! // Fused QK RoPE
//! rope.apply_qk_inplace(&mut q, &mut k)?;
//! ```

use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::{
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result},
};

/// Configuration for fused RoPE kernel.
#[derive(Debug, Clone)]
pub struct FusedRoPEConfig {
    /// Batch size.
    pub batch_size: usize,

    /// Number of attention heads.
    pub num_heads: usize,

    /// Number of KV heads (for GQA).
    pub num_kv_heads: usize,

    /// Sequence length.
    pub seq_len: usize,

    /// Head dimension (must be even).
    pub head_dim: usize,

    /// Base frequency for RoPE (default 10000).
    pub base: f32,

    /// Position scale factor (default 1.0).
    pub scale: f32,

    /// Use fp16 kernel.
    pub use_fp16: bool,
}

impl FusedRoPEConfig {
    /// Create a new RoPE config.
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            batch_size,
            num_heads,
            num_kv_heads: num_heads,
            seq_len,
            head_dim,
            base: 10000.0,
            scale: 1.0,
            use_fp16: false,
        }
    }

    /// Create config for GQA (different Q and KV head counts).
    pub fn with_gqa(
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            batch_size,
            num_heads,
            num_kv_heads,
            seq_len,
            head_dim,
            base: 10000.0,
            scale: 1.0,
            use_fp16: false,
        }
    }

    /// Set the base frequency.
    pub fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    /// Set the position scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Enable fp16 mode.
    pub fn with_fp16(mut self) -> Self {
        self.use_fp16 = true;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.head_dim % 2 != 0 {
            return Err(MetalError::InvalidConfig(
                "head_dim must be even for RoPE".to_string(),
            ));
        }
        Ok(())
    }
}

/// Precomputed RoPE cache for efficient batched inference.
#[derive(Debug)]
pub struct RoPECache {
    /// Cosine values [max_seq_len, head_dim/2].
    pub cos_cache: MetalBuffer<f32>,
    /// Sine values [max_seq_len, head_dim/2].
    pub sin_cache: MetalBuffer<f32>,
    /// Maximum sequence length cached.
    pub max_seq_len: usize,
    /// Head dimension.
    pub head_dim: usize,
}

/// Fused RoPE kernel.
///
/// Provides in-place rotary position embeddings optimized for Apple Silicon.
pub struct FusedRoPE {
    ctx: Arc<MetalContext>,
    config: FusedRoPEConfig,
}

impl FusedRoPE {
    /// Create a new fused RoPE kernel.
    pub fn new(ctx: Arc<MetalContext>, config: FusedRoPEConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { ctx, config })
    }

    /// Get the configuration.
    pub fn config(&self) -> &FusedRoPEConfig {
        &self.config
    }

    /// Apply RoPE in-place with sequential positions [0, 1, 2, ...].
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, heads, seq_len, head_dim] (modified in-place)
    pub fn apply_inplace(&self, x: &MetalBuffer<f32>) -> Result<()> {
        let expected_size = self.config.batch_size
            * self.config.num_heads
            * self.config.seq_len
            * self.config.head_dim;

        if x.len() != expected_size {
            return Err(MetalError::DimensionMismatch {
                param: "x",
                expected: expected_size,
                actual: x.len(),
            });
        }

        self.execute_inplace(x)
    }

    /// Apply RoPE in-place with custom position IDs.
    ///
    /// Essential for sequence packing where positions reset for each sequence.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, heads, seq_len, head_dim] (modified in-place)
    /// * `position_ids` - Position indices [seq_len]
    pub fn apply_with_positions(
        &self,
        x: &MetalBuffer<f32>,
        position_ids: &MetalBuffer<i32>,
    ) -> Result<()> {
        let expected_size = self.config.batch_size
            * self.config.num_heads
            * self.config.seq_len
            * self.config.head_dim;

        if x.len() != expected_size {
            return Err(MetalError::DimensionMismatch {
                param: "x",
                expected: expected_size,
                actual: x.len(),
            });
        }

        if position_ids.len() != self.config.seq_len {
            return Err(MetalError::DimensionMismatch {
                param: "position_ids",
                expected: self.config.seq_len,
                actual: position_ids.len(),
            });
        }

        self.execute_with_positions(x, position_ids)
    }

    /// Apply RoPE to both Q and K tensors in a single kernel launch.
    ///
    /// More efficient than two separate calls as it:
    /// 1. Computes sin/cos once per position
    /// 2. Amortizes kernel launch overhead
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor [batch, q_heads, seq_len, head_dim] (modified in-place)
    /// * `k` - Key tensor [batch, kv_heads, seq_len, head_dim] (modified in-place)
    pub fn apply_qk_inplace(
        &self,
        q: &MetalBuffer<f32>,
        k: &MetalBuffer<f32>,
    ) -> Result<()> {
        let expected_q = self.config.batch_size
            * self.config.num_heads
            * self.config.seq_len
            * self.config.head_dim;

        let expected_k = self.config.batch_size
            * self.config.num_kv_heads
            * self.config.seq_len
            * self.config.head_dim;

        if q.len() != expected_q {
            return Err(MetalError::DimensionMismatch {
                param: "q",
                expected: expected_q,
                actual: q.len(),
            });
        }

        if k.len() != expected_k {
            return Err(MetalError::DimensionMismatch {
                param: "k",
                expected: expected_k,
                actual: k.len(),
            });
        }

        self.execute_qk_inplace(q, k)
    }

    /// Apply RoPE to Q and K with custom position IDs.
    pub fn apply_qk_with_positions(
        &self,
        q: &MetalBuffer<f32>,
        k: &MetalBuffer<f32>,
        position_ids: &MetalBuffer<i32>,
    ) -> Result<()> {
        let expected_q = self.config.batch_size
            * self.config.num_heads
            * self.config.seq_len
            * self.config.head_dim;

        let expected_k = self.config.batch_size
            * self.config.num_kv_heads
            * self.config.seq_len
            * self.config.head_dim;

        if q.len() != expected_q {
            return Err(MetalError::DimensionMismatch {
                param: "q",
                expected: expected_q,
                actual: q.len(),
            });
        }

        if k.len() != expected_k {
            return Err(MetalError::DimensionMismatch {
                param: "k",
                expected: expected_k,
                actual: k.len(),
            });
        }

        if position_ids.len() != self.config.seq_len {
            return Err(MetalError::DimensionMismatch {
                param: "position_ids",
                expected: self.config.seq_len,
                actual: position_ids.len(),
            });
        }

        self.execute_qk_with_positions(q, k, position_ids)
    }

    /// Precompute a RoPE cache for efficient batched inference.
    ///
    /// The cache stores precomputed sin/cos values for all positions,
    /// eliminating redundant computation during inference.
    pub fn compute_cache(&self, max_seq_len: usize) -> Result<RoPECache> {
        let half_dim = self.config.head_dim / 2;
        let cache_size = max_seq_len * half_dim;

        let cos_cache = MetalBuffer::new(&self.ctx, cache_size, BufferUsage::Shared)?;
        let sin_cache = MetalBuffer::new(&self.ctx, cache_size, BufferUsage::Shared)?;

        self.execute_compute_cache(&cos_cache, &sin_cache, max_seq_len)?;

        Ok(RoPECache {
            cos_cache,
            sin_cache,
            max_seq_len,
            head_dim: self.config.head_dim,
        })
    }

    /// Apply RoPE using a precomputed cache.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [batch, heads, seq_len, head_dim] (modified in-place)
    /// * `cache` - Precomputed RoPE cache
    /// * `offset` - Position offset (for KV cache during generation)
    pub fn apply_with_cache(
        &self,
        x: &MetalBuffer<f32>,
        cache: &RoPECache,
        offset: usize,
    ) -> Result<()> {
        let expected_size = self.config.batch_size
            * self.config.num_heads
            * self.config.seq_len
            * self.config.head_dim;

        if x.len() != expected_size {
            return Err(MetalError::DimensionMismatch {
                param: "x",
                expected: expected_size,
                actual: x.len(),
            });
        }

        if self.config.seq_len + offset > cache.max_seq_len {
            return Err(MetalError::InvalidConfig(format!(
                "seq_len + offset ({}) exceeds cache max_seq_len ({})",
                self.config.seq_len + offset,
                cache.max_seq_len
            )));
        }

        self.execute_with_cache(x, cache, offset)
    }

    fn execute_inplace(&self, x: &MetalBuffer<f32>) -> Result<()> {
        let kernel_name = if self.config.use_fp16 {
            "rope_inplace_f16"
        } else {
            "rope_inplace"
        };

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), kernel_name, None)?
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

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 1);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.batch_size,
            height: self.config.num_heads,
            depth: self.config.seq_len,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: 64,
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

    fn execute_with_positions(
        &self,
        x: &MetalBuffer<f32>,
        position_ids: &MetalBuffer<i32>,
    ) -> Result<()> {
        let kernel_name = if self.config.use_fp16 {
            "rope_with_positions_f16"
        } else {
            "rope_with_positions"
        };

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), kernel_name, None)?
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
            encoder.setBuffer_offset_atIndex(Some(position_ids.metal_buffer()), 0, 1);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 2);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.batch_size,
            height: self.config.num_heads,
            depth: self.config.seq_len,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: 64,
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

    fn execute_qk_inplace(
        &self,
        q: &MetalBuffer<f32>,
        k: &MetalBuffer<f32>,
    ) -> Result<()> {
        let kernel_name = "rope_qk_inplace";

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), kernel_name, None)?
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
            encoder.setBuffer_offset_atIndex(Some(q.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.metal_buffer()), 0, 1);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 2);

            let kv_heads = self.config.num_kv_heads as u32;
            let kv_heads_ptr = NonNull::from(&kv_heads).cast();
            encoder.setBytes_length_atIndex(kv_heads_ptr, std::mem::size_of::<u32>(), 3);

            // Threadgroup memory for sin/cos cache
            let half_dim = self.config.head_dim / 2;
            let cache_size = 2 * half_dim * std::mem::size_of::<f32>();
            encoder.setThreadgroupMemoryLength_atIndex(cache_size, 0);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.batch_size,
            height: self.config.seq_len,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: 64,
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

    fn execute_qk_with_positions(
        &self,
        q: &MetalBuffer<f32>,
        k: &MetalBuffer<f32>,
        position_ids: &MetalBuffer<i32>,
    ) -> Result<()> {
        let kernel_name = "rope_qk_with_positions";

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), kernel_name, None)?
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
            encoder.setBuffer_offset_atIndex(Some(q.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(position_ids.metal_buffer()), 0, 2);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 3);

            let kv_heads = self.config.num_kv_heads as u32;
            let kv_heads_ptr = NonNull::from(&kv_heads).cast();
            encoder.setBytes_length_atIndex(kv_heads_ptr, std::mem::size_of::<u32>(), 4);

            let half_dim = self.config.head_dim / 2;
            let cache_size = 2 * half_dim * std::mem::size_of::<f32>();
            encoder.setThreadgroupMemoryLength_atIndex(cache_size, 0);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.batch_size,
            height: self.config.seq_len,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: 64,
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

    fn execute_compute_cache(
        &self,
        cos_cache: &MetalBuffer<f32>,
        sin_cache: &MetalBuffer<f32>,
        max_seq_len: usize,
    ) -> Result<()> {
        let kernel_name = "compute_rope_cache";

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), kernel_name, None)?
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
            encoder.setBuffer_offset_atIndex(Some(cos_cache.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(sin_cache.metal_buffer()), 0, 1);

            let max_seq = max_seq_len as u32;
            let head_dim = self.config.head_dim as u32;
            let base = self.config.base;
            let scale = self.config.scale;

            let max_seq_ptr = NonNull::from(&max_seq).cast();
            encoder.setBytes_length_atIndex(max_seq_ptr, std::mem::size_of::<u32>(), 2);

            let head_dim_ptr = NonNull::from(&head_dim).cast();
            encoder.setBytes_length_atIndex(head_dim_ptr, std::mem::size_of::<u32>(), 3);

            let base_ptr = NonNull::from(&base).cast();
            encoder.setBytes_length_atIndex(base_ptr, std::mem::size_of::<f32>(), 4);

            let scale_ptr = NonNull::from(&scale).cast();
            encoder.setBytes_length_atIndex(scale_ptr, std::mem::size_of::<f32>(), 5);
        }

        let grid_size = objc2_metal::MTLSize {
            width: max_seq_len,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: 64,
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

    fn execute_with_cache(
        &self,
        x: &MetalBuffer<f32>,
        cache: &RoPECache,
        offset: usize,
    ) -> Result<()> {
        let kernel_name = "rope_with_cache";

        let pipeline = {
            let mut pipe_cache = self.ctx.pipeline_cache_mut();
            pipe_cache.get_or_create_pipeline(self.ctx.device(), kernel_name, None)?
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
            encoder.setBuffer_offset_atIndex(Some(cache.cos_cache.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(cache.sin_cache.metal_buffer()), 0, 2);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 3);

            let offset_u32 = offset as u32;
            let offset_ptr = NonNull::from(&offset_u32).cast();
            encoder.setBytes_length_atIndex(offset_ptr, std::mem::size_of::<u32>(), 4);
        }

        let grid_size = objc2_metal::MTLSize {
            width: self.config.batch_size,
            height: self.config.num_heads,
            depth: self.config.seq_len,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: 64,
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

    fn create_params(&self) -> RoPEParams {
        RoPEParams {
            batch_size: self.config.batch_size as u32,
            num_heads: self.config.num_heads as u32,
            seq_len: self.config.seq_len as u32,
            head_dim: self.config.head_dim as u32,
            base: self.config.base,
            scale: self.config.scale,
        }
    }
}

/// Parameters passed to the Metal kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct RoPEParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    base: f32,
    scale: f32,
}

impl std::fmt::Debug for FusedRoPE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedRoPE")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_rope_config() {
        let config = FusedRoPEConfig::new(4, 8, 128, 64);

        assert_eq!(config.batch_size, 4);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.seq_len, 128);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.base, 10000.0);
        assert_eq!(config.scale, 1.0);
    }

    #[test]
    fn test_fused_rope_config_gqa() {
        let config = FusedRoPEConfig::with_gqa(4, 32, 8, 128, 128);

        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
    }

    #[test]
    fn test_fused_rope_config_validation() {
        let config = FusedRoPEConfig::new(4, 8, 128, 64);
        assert!(config.validate().is_ok());

        let bad_config = FusedRoPEConfig::new(4, 8, 128, 63); // Odd head_dim
        assert!(bad_config.validate().is_err());
    }
}
