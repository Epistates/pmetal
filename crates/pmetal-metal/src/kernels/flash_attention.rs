//! FlashAttention implementation for Metal.
//!
//! This module provides a memory-efficient attention implementation based on
//! the FlashAttention algorithm, optimized for Apple Silicon.
//!
//! # Algorithm Overview
//!
//! FlashAttention achieves O(n) memory complexity (vs O(n²) for naive attention)
//! by computing attention in blocks without materializing the full attention matrix:
//!
//! ```text
//! For each query block Q_i:
//!     m_i = -∞, l_i = 0, O_i = 0
//!     For each key-value block (K_j, V_j):
//!         S_ij = Q_i @ K_j^T / √d
//!         Apply causal mask if needed
//!         m_new = max(m_i, rowmax(S_ij))
//!         P_ij = exp(S_ij - m_new)
//!         l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
//!         O_i = exp(m_i - m_new) * O_i + P_ij @ V_j
//!         m_i = m_new, l_i = l_new
//!     O_i = O_i / l_i
//! ```
//!
//! # References
//!
//! - [FlashAttention-2](https://arxiv.org/abs/2307.08691)
//! - [Metal FlashAttention](https://github.com/philipturner/metal-flash-attention)

use half::f16;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::buffer::{BufferUsage, MetalBuffer};
use crate::context::MetalContext;
use crate::error::{MetalError, Result};

/// Configuration for FlashAttention.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Batch size.
    pub batch_size: usize,

    /// Number of query heads.
    pub num_heads: usize,

    /// Number of key-value heads (for GQA/MQA).
    /// Set equal to `num_heads` for standard MHA.
    pub num_kv_heads: usize,

    /// Query sequence length.
    pub query_seq_len: usize,

    /// Key-value sequence length (can differ for cross-attention).
    pub kv_seq_len: usize,

    /// Head dimension.
    pub head_dim: usize,

    /// Softmax scaling factor (default: 1/√head_dim).
    pub scale: Option<f32>,

    /// Use causal attention mask.
    pub is_causal: bool,

    /// Sliding window size (None = no window, full attention).
    pub sliding_window: Option<usize>,

    /// Logit softcapping value (None = disabled).
    pub softcap: Option<f32>,

    /// Whether this is for training (stores logsumexp for backward).
    pub is_training: bool,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_heads: 32,
            num_kv_heads: 8,
            query_seq_len: 512,
            kv_seq_len: 512,
            head_dim: 128,
            scale: None,
            is_causal: true,
            sliding_window: None,
            softcap: None,
            is_training: false,
        }
    }
}

impl FlashAttentionConfig {
    /// Create a new config for inference.
    pub fn inference(
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
            query_seq_len: seq_len,
            kv_seq_len: seq_len,
            head_dim,
            is_training: false,
            ..Default::default()
        }
    }

    /// Create a new config for training.
    pub fn training(
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
            query_seq_len: seq_len,
            kv_seq_len: seq_len,
            head_dim,
            is_training: true,
            ..Default::default()
        }
    }

    /// Get the softmax scaling factor.
    #[inline]
    pub fn scaling_factor(&self) -> f32 {
        self.scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt())
    }

    /// Check if using grouped-query attention.
    #[inline]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }

    /// Get the GQA ratio (queries per KV head).
    #[inline]
    pub fn gqa_ratio(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(MetalError::InvalidConfig("batch_size must be > 0".into()));
        }
        if self.num_heads == 0 {
            return Err(MetalError::InvalidConfig("num_heads must be > 0".into()));
        }
        if self.num_kv_heads == 0 {
            return Err(MetalError::InvalidConfig("num_kv_heads must be > 0".into()));
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(MetalError::InvalidConfig(
                "num_heads must be divisible by num_kv_heads".into(),
            ));
        }
        if self.head_dim == 0 {
            return Err(MetalError::InvalidConfig("head_dim must be > 0".into()));
        }
        if !matches!(self.head_dim, 64 | 80 | 96 | 128 | 256) {
            return Err(MetalError::InvalidConfig(
                "head_dim must be one of: 64, 80, 96, 128, 256".into(),
            ));
        }
        if self.query_seq_len == 0 {
            return Err(MetalError::InvalidConfig(
                "query_seq_len must be > 0".into(),
            ));
        }
        if self.kv_seq_len == 0 {
            return Err(MetalError::InvalidConfig("kv_seq_len must be > 0".into()));
        }

        Ok(())
    }

    /// Get the expected size of query tensor.
    pub fn query_size(&self) -> usize {
        self.batch_size * self.num_heads * self.query_seq_len * self.head_dim
    }

    /// Get the expected size of key/value tensor.
    pub fn kv_size(&self) -> usize {
        self.batch_size * self.num_kv_heads * self.kv_seq_len * self.head_dim
    }

    /// Get the expected size of output tensor.
    pub fn output_size(&self) -> usize {
        self.batch_size * self.num_heads * self.query_seq_len * self.head_dim
    }

    /// Get the expected size of logsumexp tensor (for training).
    pub fn logsumexp_size(&self) -> usize {
        self.batch_size * self.num_heads * self.query_seq_len
    }
}

/// Output from FlashAttention forward pass.
#[derive(Debug)]
pub struct FlashAttentionOutput {
    /// Attention output [batch, num_heads, seq_len, head_dim].
    pub output: MetalBuffer<f16>,

    /// Log-sum-exp values for backward pass [batch, num_heads, seq_len].
    /// Only present if config.is_training is true.
    pub logsumexp: Option<MetalBuffer<f32>>,
}

/// FlashAttention kernel executor.
///
/// This struct manages the Metal pipelines and executes FlashAttention
/// operations efficiently on the GPU.
pub struct FlashAttention {
    /// Metal context.
    ctx: Arc<MetalContext>,

    /// Configuration.
    config: FlashAttentionConfig,

    /// Block size for queries (Bq).
    block_q: usize,

    /// Block size for keys (Bk).
    block_k: usize,
}

impl FlashAttention {
    /// Create a new FlashAttention executor.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Metal context
    /// * `config` - Attention configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(ctx: Arc<MetalContext>, config: FlashAttentionConfig) -> Result<Self> {
        config.validate()?;

        // Determine block sizes based on head dimension AND device tier
        // M4 Max/Ultra can use larger blocks due to higher memory bandwidth
        let device_tier = ctx.properties().device_tier;
        let (block_q, block_k) = Self::select_block_sizes(config.head_dim, device_tier);

        Ok(Self {
            ctx,
            config,
            block_q,
            block_k,
        })
    }

    /// Select optimal block sizes based on head dimension and device tier.
    ///
    /// Higher-tier devices (M4 Max/Ultra) benefit from larger tile sizes
    /// due to increased memory bandwidth and shader core count.
    fn select_block_sizes(
        head_dim: usize,
        device_tier: crate::context::DeviceTier,
    ) -> (usize, usize) {
        use crate::context::DeviceTier;

        match (head_dim, device_tier) {
            // D=64: Can use larger blocks on high-end devices
            (64, DeviceTier::Ultra | DeviceTier::Max) => (64, 64),
            (64, DeviceTier::Pro) => (64, 64),
            (64, DeviceTier::Base) => (64, 32),

            // D=80: Awkward dimension, use asymmetric blocks
            (80, DeviceTier::Ultra | DeviceTier::Max) => (64, 32),
            (80, DeviceTier::Pro) => (64, 32),
            (80, DeviceTier::Base) => (32, 32),

            // D=96: Similar to D=80
            (96, DeviceTier::Ultra | DeviceTier::Max) => (64, 32),
            (96, DeviceTier::Pro) => (64, 32),
            (96, DeviceTier::Base) => (32, 32),

            // D=128: Most common, optimize carefully
            (128, DeviceTier::Ultra | DeviceTier::Max) => (64, 32),
            (128, DeviceTier::Pro) => (32, 32),
            (128, DeviceTier::Base) => (32, 32),

            // D=256: Large head dim needs smaller blocks
            (256, DeviceTier::Ultra | DeviceTier::Max) => (32, 16),
            (256, DeviceTier::Pro) => (16, 16),
            (256, DeviceTier::Base) => (16, 16),

            // Default fallback
            (_, DeviceTier::Ultra | DeviceTier::Max) => (32, 32),
            (_, _) => (32, 32),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &FlashAttentionConfig {
        &self.config
    }

    /// Compute attention forward pass.
    ///
    /// # Arguments
    ///
    /// * `queries` - Query tensor [batch, num_heads, query_seq_len, head_dim]
    /// * `keys` - Key tensor [batch, num_kv_heads, kv_seq_len, head_dim]
    /// * `values` - Value tensor [batch, num_kv_heads, kv_seq_len, head_dim]
    ///
    /// # Returns
    ///
    /// - `output` - Attention output [batch, num_heads, query_seq_len, head_dim]
    /// - `logsumexp` - Log-sum-exp for backward (if training)
    pub fn forward(
        &self,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
    ) -> Result<FlashAttentionOutput> {
        // Validate input sizes
        self.validate_input_sizes(queries, keys, values)?;

        // Allocate output buffer
        let output = MetalBuffer::new(&self.ctx, self.config.output_size(), BufferUsage::Shared)?;

        // Allocate logsumexp if training
        let logsumexp = if self.config.is_training {
            Some(MetalBuffer::new(
                &self.ctx,
                self.config.logsumexp_size(),
                BufferUsage::Shared,
            )?)
        } else {
            None
        };

        // Execute the kernel
        self.execute_forward(queries, keys, values, &output, logsumexp.as_ref())?;

        Ok(FlashAttentionOutput { output, logsumexp })
    }

    /// Compute attention backward pass.
    ///
    /// # Arguments
    ///
    /// * `queries` - Query tensor from forward
    /// * `keys` - Key tensor from forward
    /// * `values` - Value tensor from forward
    /// * `output` - Output from forward pass
    /// * `d_output` - Gradient of loss w.r.t. output
    /// * `logsumexp` - Logsumexp from forward pass
    ///
    /// # Returns
    ///
    /// Gradients (dQ, dK, dV)
    pub fn backward(
        &self,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
        output: &MetalBuffer<f16>,
        d_output: &MetalBuffer<f16>,
        logsumexp: &MetalBuffer<f32>,
    ) -> Result<(MetalBuffer<f16>, MetalBuffer<f16>, MetalBuffer<f16>)> {
        // Validate sizes
        self.validate_input_sizes(queries, keys, values)?;

        if output.len() != self.config.output_size() {
            return Err(MetalError::DimensionMismatch {
                param: "output",
                expected: self.config.output_size(),
                actual: output.len(),
            });
        }
        if d_output.len() != self.config.output_size() {
            return Err(MetalError::DimensionMismatch {
                param: "d_output",
                expected: self.config.output_size(),
                actual: d_output.len(),
            });
        }
        if logsumexp.len() != self.config.logsumexp_size() {
            return Err(MetalError::DimensionMismatch {
                param: "logsumexp",
                expected: self.config.logsumexp_size(),
                actual: logsumexp.len(),
            });
        }

        // Allocate gradient buffers
        let d_queries =
            MetalBuffer::zeros(&self.ctx, self.config.query_size(), BufferUsage::Shared)?;
        let d_keys = MetalBuffer::zeros(&self.ctx, self.config.kv_size(), BufferUsage::Shared)?;
        let d_values = MetalBuffer::zeros(&self.ctx, self.config.kv_size(), BufferUsage::Shared)?;

        // Execute backward kernels
        self.execute_backward_dq(
            queries, keys, values, output, d_output, logsumexp, &d_queries,
        )?;
        self.execute_backward_dkv(
            queries, keys, values, output, d_output, logsumexp, &d_keys, &d_values,
        )?;

        Ok((d_queries, d_keys, d_values))
    }

    /// Validate input tensor sizes.
    fn validate_input_sizes(
        &self,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
    ) -> Result<()> {
        let expected_q = self.config.query_size();
        let expected_kv = self.config.kv_size();

        if queries.len() != expected_q {
            return Err(MetalError::DimensionMismatch {
                param: "queries",
                expected: expected_q,
                actual: queries.len(),
            });
        }
        if keys.len() != expected_kv {
            return Err(MetalError::DimensionMismatch {
                param: "keys",
                expected: expected_kv,
                actual: keys.len(),
            });
        }
        if values.len() != expected_kv {
            return Err(MetalError::DimensionMismatch {
                param: "values",
                expected: expected_kv,
                actual: values.len(),
            });
        }

        Ok(())
    }

    /// Create typed function constants for Metal shader specialization.
    ///
    /// The FlashAttention Metal shader uses these constants:
    /// - 0: BLOCK_Q (query block size) - UInt
    /// - 1: BLOCK_K (key block size) - UInt
    /// - 2: HEAD_DIM - UInt
    /// - 3: IS_CAUSAL - Bool
    fn create_function_constants(
        &self,
    ) -> std::collections::HashMap<u64, crate::pipeline::FunctionConstant> {
        use crate::pipeline::FunctionConstant;
        let mut constants = std::collections::HashMap::new();
        constants.insert(0, FunctionConstant::UInt(self.block_q as u32));
        constants.insert(1, FunctionConstant::UInt(self.block_k as u32));
        constants.insert(2, FunctionConstant::UInt(self.config.head_dim as u32));
        constants.insert(3, FunctionConstant::Bool(self.config.is_causal));
        constants
    }

    /// Execute the forward kernel.
    fn execute_forward(
        &self,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
        output: &MetalBuffer<f16>,
        logsumexp: Option<&MetalBuffer<f32>>,
    ) -> Result<()> {
        // Get or create the SPECIALIZED pipeline with typed function constants
        // This avoids leaving encoder in bad state on error
        let function_name = self.forward_kernel_name();
        let constants = self.create_function_constants();
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_specialized_pipeline_typed(
                self.ctx.device(),
                &function_name,
                &constants,
            )?
        };

        // Get command queue and create command buffer
        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        // Create compute encoder
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        // Set pipeline state
        encoder.setComputePipelineState(&pipeline);

        // Set buffers
        // SAFETY: Metal compute encoder operations are safe when buffers are valid
        // and encoder is in the correct state (between creation and endEncoding).
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(queries.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(keys.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(values.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 3);

            if let Some(lse) = logsumexp {
                encoder.setBuffer_offset_atIndex(Some(lse.metal_buffer()), 0, 4);
            }

            // Set parameters as bytes
            let params = self.create_kernel_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);
        }

        // Calculate grid and threadgroup sizes
        let num_q_blocks = (self.config.query_seq_len + self.block_q - 1) / self.block_q;
        let grid_size = objc2_metal::MTLSize {
            width: num_q_blocks,
            height: self.config.num_heads,
            depth: self.config.batch_size,
        };

        let threadgroup_size = objc2_metal::MTLSize {
            width: 32, // SIMD width on Apple GPUs
            height: 4, // Warp-level parallelism
            depth: 1,
        };

        // Dispatch
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);

        // End encoding and commit
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Check for errors
        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Execute the backward dQ kernel.
    fn execute_backward_dq(
        &self,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
        output: &MetalBuffer<f16>,
        d_output: &MetalBuffer<f16>,
        logsumexp: &MetalBuffer<f32>,
        d_queries: &MetalBuffer<f16>,
    ) -> Result<()> {
        // Get or create the SPECIALIZED pipeline with typed function constants
        let function_name = self.backward_dq_kernel_name();
        let constants = self.create_function_constants();
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_specialized_pipeline_typed(
                self.ctx.device(),
                &function_name,
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
        // and encoder is in the correct state.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(queries.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(keys.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(values.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(d_output.metal_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(logsumexp.metal_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(d_queries.metal_buffer()), 0, 6);

            let params = self.create_kernel_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 7);
        }

        // Calculate grid size (parallelize over query blocks)
        let num_q_blocks = (self.config.query_seq_len + self.block_q - 1) / self.block_q;
        let grid_size = objc2_metal::MTLSize {
            width: num_q_blocks,
            height: self.config.num_heads,
            depth: self.config.batch_size,
        };

        let threadgroup_size = objc2_metal::MTLSize {
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

    /// Execute the backward dK/dV kernel.
    ///
    /// Note: The `output` buffer is required for exact gradient computation via D_i = rowsum(dO * O)
    fn execute_backward_dkv(
        &self,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
        output: &MetalBuffer<f16>,
        d_output: &MetalBuffer<f16>,
        logsumexp: &MetalBuffer<f32>,
        d_keys: &MetalBuffer<f16>,
        d_values: &MetalBuffer<f16>,
    ) -> Result<()> {
        // Get or create the SPECIALIZED pipeline with typed function constants
        let function_name = self.backward_dkv_kernel_name();
        let constants = self.create_function_constants();
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_specialized_pipeline_typed(
                self.ctx.device(),
                &function_name,
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
        // and encoder is in the correct state.
        // Buffer layout: Q(0), K(1), V(2), O(3), dO(4), L(5), dK(6), dV(7), params(8)
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(queries.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(keys.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(values.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(d_output.metal_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(logsumexp.metal_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(d_keys.metal_buffer()), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(d_values.metal_buffer()), 0, 7);

            let params = self.create_kernel_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 8);
        }

        // Calculate grid size (parallelize over KV blocks)
        let num_kv_blocks = (self.config.kv_seq_len + self.block_k - 1) / self.block_k;
        let grid_size = objc2_metal::MTLSize {
            width: num_kv_blocks,
            height: self.config.num_kv_heads,
            depth: self.config.batch_size,
        };

        let threadgroup_size = objc2_metal::MTLSize {
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

    /// Get the forward kernel function name.
    fn forward_kernel_name(&self) -> String {
        format!(
            "flash_attention_forward_d{}{}",
            self.config.head_dim,
            if self.config.is_causal { "_causal" } else { "" }
        )
    }

    /// Get the backward dQ kernel function name.
    fn backward_dq_kernel_name(&self) -> String {
        format!(
            "flash_attention_backward_dq_d{}{}",
            self.config.head_dim,
            if self.config.is_causal { "_causal" } else { "" }
        )
    }

    /// Get the backward dKV kernel function name.
    fn backward_dkv_kernel_name(&self) -> String {
        format!(
            "flash_attention_backward_dkv_d{}{}",
            self.config.head_dim,
            if self.config.is_causal { "_causal" } else { "" }
        )
    }

    /// Get a configuration key for pipeline caching.
    fn config_key(&self) -> String {
        format!(
            "bq{}_bk{}_gqa{}",
            self.block_q,
            self.block_k,
            self.config.gqa_ratio()
        )
    }

    /// Create kernel parameters struct.
    fn create_kernel_params(&self) -> FlashAttentionParams {
        FlashAttentionParams {
            batch_size: self.config.batch_size as u32,
            num_heads: self.config.num_heads as u32,
            num_kv_heads: self.config.num_kv_heads as u32,
            query_seq_len: self.config.query_seq_len as u32,
            kv_seq_len: self.config.kv_seq_len as u32,
            head_dim: self.config.head_dim as u32,
            scale: self.config.scaling_factor(),
            block_q: self.block_q as u32,
            block_k: self.block_k as u32,
            gqa_ratio: self.config.gqa_ratio() as u32,
            is_causal: self.config.is_causal as u32,
            sliding_window: self.config.sliding_window.unwrap_or(0) as u32,
            softcap: self.config.softcap.unwrap_or(0.0),
        }
    }
}

/// Parameters passed to the kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct FlashAttentionParams {
    batch_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    query_seq_len: u32,
    kv_seq_len: u32,
    head_dim: u32,
    scale: f32,
    block_q: u32,
    block_k: u32,
    gqa_ratio: u32,
    is_causal: u32,
    sliding_window: u32,
    softcap: f32,
}

impl std::fmt::Debug for FlashAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlashAttention")
            .field("config", &self.config)
            .field("block_q", &self.block_q)
            .field("block_k", &self.block_k)
            .finish()
    }
}

// =============================================================================
// Variable-Length Sequence Support (Packed Sequences)
// =============================================================================

/// Configuration for variable-length FlashAttention (packed sequences).
#[derive(Debug, Clone)]
pub struct FlashAttentionVarlenConfig {
    /// Total number of tokens across all sequences.
    pub total_tokens: usize,

    /// Number of query heads.
    pub num_heads: usize,

    /// Number of key-value heads (for GQA/MQA).
    pub num_kv_heads: usize,

    /// Head dimension.
    pub head_dim: usize,

    /// Number of sequences in the packed batch.
    pub num_seqs: usize,

    /// Maximum sequence length in the batch.
    pub max_seqlen: usize,

    /// Softmax scaling factor (default: 1/√head_dim).
    pub scale: Option<f32>,

    /// Use causal attention mask.
    pub is_causal: bool,

    /// Logit softcapping value (None = disabled).
    pub softcap: Option<f32>,

    /// Sliding window size (None = no window).
    pub sliding_window: Option<usize>,

    /// Whether this is for training (stores logsumexp for backward).
    pub is_training: bool,
}

impl FlashAttentionVarlenConfig {
    /// Create a new config for packed training.
    pub fn training(
        total_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_seqs: usize,
        max_seqlen: usize,
    ) -> Self {
        Self {
            total_tokens,
            num_heads,
            num_kv_heads,
            head_dim,
            num_seqs,
            max_seqlen,
            scale: None,
            is_causal: true,
            softcap: None,
            sliding_window: None,
            is_training: true,
        }
    }

    /// Get the softmax scaling factor.
    #[inline]
    pub fn scaling_factor(&self) -> f32 {
        self.scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt())
    }

    /// Get the GQA ratio.
    #[inline]
    pub fn gqa_ratio(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.total_tokens == 0 {
            return Err(MetalError::InvalidConfig("total_tokens must be > 0".into()));
        }
        if self.num_heads == 0 {
            return Err(MetalError::InvalidConfig("num_heads must be > 0".into()));
        }
        if self.num_kv_heads == 0 {
            return Err(MetalError::InvalidConfig("num_kv_heads must be > 0".into()));
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(MetalError::InvalidConfig(
                "num_heads must be divisible by num_kv_heads".into(),
            ));
        }
        if !matches!(self.head_dim, 64 | 80 | 96 | 128 | 256) {
            return Err(MetalError::InvalidConfig(
                "head_dim must be one of: 64, 80, 96, 128, 256".into(),
            ));
        }
        if self.num_seqs == 0 {
            return Err(MetalError::InvalidConfig("num_seqs must be > 0".into()));
        }
        Ok(())
    }

    /// Get expected Q/O size.
    pub fn query_size(&self) -> usize {
        self.total_tokens * self.num_heads * self.head_dim
    }

    /// Get expected K/V size.
    pub fn kv_size(&self) -> usize {
        self.total_tokens * self.num_kv_heads * self.head_dim
    }

    /// Get expected logsumexp size.
    pub fn logsumexp_size(&self) -> usize {
        self.total_tokens * self.num_heads
    }
}

/// Output from variable-length FlashAttention.
#[derive(Debug)]
pub struct FlashAttentionVarlenOutput {
    /// Attention output [total_tokens, num_heads, head_dim].
    pub output: MetalBuffer<f16>,

    /// Log-sum-exp for backward pass [total_tokens, num_heads].
    pub logsumexp: Option<MetalBuffer<f32>>,
}

/// Variable-length FlashAttention for packed sequences.
///
/// This kernel handles packed batches where multiple sequences are
/// concatenated together. Uses cu_seqlens to determine sequence
/// boundaries and implements block-diagonal attention.
pub struct FlashAttentionVarlen {
    /// Metal context.
    ctx: Arc<MetalContext>,

    /// Configuration.
    config: FlashAttentionVarlenConfig,

    /// Block size for queries.
    block_q: usize,

    /// Block size for keys.
    block_k: usize,
}

impl FlashAttentionVarlen {
    /// Create a new variable-length FlashAttention executor.
    pub fn new(ctx: Arc<MetalContext>, config: FlashAttentionVarlenConfig) -> Result<Self> {
        config.validate()?;

        let (block_q, block_k) = match config.head_dim {
            64 => (64, 64),
            80 => (64, 32),
            96 => (64, 32),
            128 => (32, 32),
            256 => (16, 16),
            _ => (32, 32),
        };

        Ok(Self {
            ctx,
            config,
            block_q,
            block_k,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &FlashAttentionVarlenConfig {
        &self.config
    }

    /// Compute attention forward pass for packed sequences.
    ///
    /// # Arguments
    ///
    /// * `queries` - Query tensor [total_tokens, num_heads, head_dim]
    /// * `keys` - Key tensor [total_tokens, num_kv_heads, head_dim]
    /// * `values` - Value tensor [total_tokens, num_kv_heads, head_dim]
    /// * `cu_seqlens` - Cumulative sequence lengths [num_seqs + 1]
    pub fn forward(
        &self,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
        cu_seqlens: &MetalBuffer<i32>,
    ) -> Result<FlashAttentionVarlenOutput> {
        // Validate sizes
        if queries.len() != self.config.query_size() {
            return Err(MetalError::DimensionMismatch {
                param: "queries",
                expected: self.config.query_size(),
                actual: queries.len(),
            });
        }
        if keys.len() != self.config.kv_size() {
            return Err(MetalError::DimensionMismatch {
                param: "keys",
                expected: self.config.kv_size(),
                actual: keys.len(),
            });
        }
        if cu_seqlens.len() != self.config.num_seqs + 1 {
            return Err(MetalError::DimensionMismatch {
                param: "cu_seqlens",
                expected: self.config.num_seqs + 1,
                actual: cu_seqlens.len(),
            });
        }

        // Allocate output
        let output = MetalBuffer::new(&self.ctx, self.config.query_size(), BufferUsage::Shared)?;

        let logsumexp = if self.config.is_training {
            Some(MetalBuffer::new(
                &self.ctx,
                self.config.logsumexp_size(),
                BufferUsage::Shared,
            )?)
        } else {
            None
        };

        self.execute_forward(
            queries,
            keys,
            values,
            cu_seqlens,
            &output,
            logsumexp.as_ref(),
        )?;

        Ok(FlashAttentionVarlenOutput { output, logsumexp })
    }

    /// Execute the forward kernel.
    fn execute_forward(
        &self,
        queries: &MetalBuffer<f16>,
        keys: &MetalBuffer<f16>,
        values: &MetalBuffer<f16>,
        cu_seqlens: &MetalBuffer<i32>,
        output: &MetalBuffer<f16>,
        logsumexp: Option<&MetalBuffer<f32>>,
    ) -> Result<()> {
        let function_name = self.kernel_name();
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), &function_name, None)?
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
            encoder.setBuffer_offset_atIndex(Some(queries.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(keys.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(values.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 3);

            if let Some(lse) = logsumexp {
                encoder.setBuffer_offset_atIndex(Some(lse.metal_buffer()), 0, 4);
            }

            encoder.setBuffer_offset_atIndex(Some(cu_seqlens.metal_buffer()), 0, 5);

            let params = self.create_params();
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 6);
        }

        // Calculate grid size: total query blocks across all sequences
        let num_q_blocks = self.compute_total_q_blocks();
        let grid_size = objc2_metal::MTLSize {
            width: num_q_blocks,
            height: self.config.num_heads,
            depth: 1,
        };

        let threadgroup_size = objc2_metal::MTLSize {
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

    /// Compute total number of query blocks across all sequences.
    fn compute_total_q_blocks(&self) -> usize {
        // Estimate: assuming average sequence length
        let avg_len = self.config.total_tokens / self.config.num_seqs.max(1);
        let blocks_per_seq = (avg_len + self.block_q - 1) / self.block_q;
        blocks_per_seq * self.config.num_seqs
    }

    /// Get kernel function name.
    fn kernel_name(&self) -> String {
        format!("flash_attention_varlen_forward_d{}", self.config.head_dim)
    }

    /// Create kernel parameters.
    fn create_params(&self) -> FlashAttentionVarlenParams {
        FlashAttentionVarlenParams {
            total_tokens: self.config.total_tokens as u32,
            num_heads: self.config.num_heads as u32,
            num_kv_heads: self.config.num_kv_heads as u32,
            head_dim: self.config.head_dim as u32,
            num_seqs: self.config.num_seqs as u32,
            scale: self.config.scaling_factor(),
            gqa_ratio: self.config.gqa_ratio() as u32,
            max_seqlen: self.config.max_seqlen as u32,
            is_causal: self.config.is_causal as u32,
            softcap: self.config.softcap.unwrap_or(0.0),
            sliding_window: self.config.sliding_window.unwrap_or(0) as u32,
        }
    }
}

/// Parameters for variable-length attention kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct FlashAttentionVarlenParams {
    total_tokens: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    num_seqs: u32,
    scale: f32,
    gqa_ratio: u32,
    max_seqlen: u32,
    is_causal: u32,
    softcap: f32,
    sliding_window: u32,
}

impl std::fmt::Debug for FlashAttentionVarlen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlashAttentionVarlen")
            .field("config", &self.config)
            .field("block_q", &self.block_q)
            .field("block_k", &self.block_k)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let valid_config = FlashAttentionConfig {
            batch_size: 2,
            num_heads: 32,
            num_kv_heads: 8,
            query_seq_len: 512,
            kv_seq_len: 512,
            head_dim: 128,
            ..Default::default()
        };
        assert!(valid_config.validate().is_ok());

        let invalid_config = FlashAttentionConfig {
            num_heads: 32,
            num_kv_heads: 7, // Not divisible
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_config_sizes() {
        let config = FlashAttentionConfig {
            batch_size: 2,
            num_heads: 32,
            num_kv_heads: 8,
            query_seq_len: 512,
            kv_seq_len: 1024,
            head_dim: 128,
            ..Default::default()
        };

        assert_eq!(config.query_size(), 2 * 32 * 512 * 128);
        assert_eq!(config.kv_size(), 2 * 8 * 1024 * 128);
        assert_eq!(config.output_size(), 2 * 32 * 512 * 128);
        assert_eq!(config.logsumexp_size(), 2 * 32 * 512);
    }

    #[test]
    fn test_gqa_ratio() {
        let config = FlashAttentionConfig {
            num_heads: 32,
            num_kv_heads: 8,
            ..Default::default()
        };
        assert!(config.is_gqa());
        assert_eq!(config.gqa_ratio(), 4);

        let mha_config = FlashAttentionConfig {
            num_heads: 32,
            num_kv_heads: 32,
            ..Default::default()
        };
        assert!(!mha_config.is_gqa());
        assert_eq!(mha_config.gqa_ratio(), 1);
    }
}
