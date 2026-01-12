//! Fused sampling kernel for high-performance token generation.
//!
//! This module bypasses mlx-rs for the sampling hot path, executing all
//! sampling operations in a single Metal kernel:
//!
//! - Argmax (greedy decoding)
//! - Temperature scaling
//! - Top-K filtering
//! - Top-P (nucleus) filtering
//! - Min-P filtering
//! - Categorical sampling
//!
//! # Performance Benefits
//!
//! - **Single kernel launch**: vs 10+ separate launches with mlx-rs
//! - **Zero intermediate allocations**: all work in threadgroup memory
//! - **Minimal CPU overhead**: critical for battery mode performance
//!
//! # Usage
//!
//! ```rust,ignore
//! use pmetal_metal::kernels::FusedSampler;
//!
//! let sampler = FusedSampler::new(vocab_size)?;
//!
//! // For greedy decoding (temp=0)
//! let token = sampler.argmax(&logits_buffer)?;
//!
//! // For temperature sampling
//! let token = sampler.sample(&logits_buffer, temperature, top_k, top_p, min_p)?;
//! ```

use std::ptr::NonNull;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::bridge::MetalBufferView;
use crate::buffer::{BufferUsage, MetalBuffer};
use crate::context::MetalContext;
use crate::error::{MetalError, Result};

/// Trait for types that can provide a Metal buffer reference.
///
/// Implemented by both `MetalBuffer` (owned) and `MetalBufferView` (borrowed).
pub trait AsMetalBuffer {
    /// Get a reference to the underlying Metal buffer.
    fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer>;

    /// Get the number of elements.
    fn len(&self) -> usize;

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Pod + Zeroable> AsMetalBuffer for MetalBuffer<T> {
    fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        MetalBuffer::metal_buffer(self)
    }

    fn len(&self) -> usize {
        MetalBuffer::len(self)
    }
}

impl<T: Pod + Zeroable> AsMetalBuffer for MetalBufferView<T> {
    fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        MetalBufferView::metal_buffer(self)
    }

    fn len(&self) -> usize {
        MetalBufferView::len(self)
    }
}

/// Sampling parameters matching the Metal kernel's SamplingParams struct.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Sampling temperature (0 = greedy).
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Min-p threshold (relative to max probability).
    pub min_p: f32,
    /// Top-k value (0 = disabled).
    pub top_k: i32,
    /// Random seed for sampling.
    pub random_seed: u32,
    /// Whether to sample (false = greedy argmax).
    pub do_sample: bool,
    // Padding to match Metal struct alignment
    _padding: [u8; 3],
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            temperature: 1.0,
            top_p: 1.0,
            min_p: 0.0,
            top_k: 0,
            random_seed: 42,
            do_sample: true,
            _padding: [0; 3],
        }
    }
}

/// Configuration for the fused sampler.
#[derive(Debug, Clone)]
pub struct FusedSamplerConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of threads per threadgroup (default: 256).
    pub threadgroup_size: usize,
}

impl Default for FusedSamplerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            threadgroup_size: 256,
        }
    }
}

/// Fused sampler that executes all sampling operations in a single Metal kernel.
///
/// This is the high-performance path for token generation, bypassing mlx-rs
/// to eliminate CPU overhead.
pub struct FusedSampler {
    /// Metal context.
    ctx: Arc<MetalContext>,
    /// Configuration.
    config: FusedSamplerConfig,
    /// Pre-allocated output buffer (single u32).
    output_buffer: MetalBuffer<u32>,
    /// Random state for seeding.
    rng_counter: std::sync::atomic::AtomicU64,
    /// Base seed for reproducibility (None = use system time for randomness).
    base_seed: Option<u64>,
}

impl FusedSampler {
    /// Create a new fused sampler.
    ///
    /// # Arguments
    /// * `vocab_size` - Vocabulary size of the model.
    ///
    /// # Errors
    /// Returns an error if Metal initialization fails.
    pub fn new(vocab_size: usize) -> Result<Self> {
        let ctx = MetalContext::global()?;
        Self::with_context(ctx, vocab_size)
    }

    /// Create a new fused sampler with a specific seed for reproducibility.
    ///
    /// # Arguments
    /// * `vocab_size` - Vocabulary size of the model.
    /// * `seed` - Random seed for reproducible sampling.
    pub fn with_seed(vocab_size: usize, seed: u64) -> Result<Self> {
        let ctx = MetalContext::global()?;
        Self::with_context_and_seed(ctx, vocab_size, Some(seed))
    }

    /// Create a new fused sampler with a specific Metal context.
    pub fn with_context(ctx: Arc<MetalContext>, vocab_size: usize) -> Result<Self> {
        Self::with_context_and_seed(ctx, vocab_size, None)
    }

    /// Create a new fused sampler with a specific Metal context and optional seed.
    pub fn with_context_and_seed(
        ctx: Arc<MetalContext>,
        vocab_size: usize,
        seed: Option<u64>,
    ) -> Result<Self> {
        let config = FusedSamplerConfig {
            vocab_size,
            ..Default::default()
        };

        // Pre-allocate output buffer (Shared for CPU read-back)
        let output_buffer = MetalBuffer::new(&ctx, 1, BufferUsage::Shared)?;

        Ok(Self {
            ctx,
            config,
            output_buffer,
            rng_counter: std::sync::atomic::AtomicU64::new(0),
            base_seed: seed,
        })
    }

    /// Set the seed for reproducible sampling.
    pub fn set_seed(&mut self, seed: u64) {
        self.base_seed = Some(seed);
        self.rng_counter
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get a unique random seed for this sample.
    fn next_seed(&self) -> u32 {
        let counter = self
            .rng_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if let Some(base) = self.base_seed {
            // Deterministic: combine base seed with counter using SplitMix64
            let mut state = base.wrapping_add(counter.wrapping_mul(0x9E3779B97F4A7C15));
            state = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            state = (state ^ (state >> 27)).wrapping_mul(0x94D049BB133111EB);
            state = state ^ (state >> 31);
            (state & 0xFFFFFFFF) as u32
        } else {
            // Non-deterministic: mix with system time for better randomness
            let time_bits = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            ((counter ^ time_bits) & 0xFFFFFFFF) as u32
        }
    }

    /// Perform greedy argmax sampling (temperature = 0).
    ///
    /// This is the fastest path for deterministic decoding.
    ///
    /// # Arguments
    /// * `logits` - Logits buffer of shape [vocab_size] as f32.
    ///
    /// # Returns
    /// The token ID with the highest logit value.
    pub fn argmax(&self, logits: &impl AsMetalBuffer) -> Result<u32> {
        // Get or create the pipeline
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "fused_argmax_simd", None)?
        };

        // Create command buffer and encoder
        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        // Set buffers
        // SAFETY:
        // 1. encoder is a valid MTLComputeCommandEncoder from the command buffer
        // 2. logits.metal_buffer() returns a valid MTLBuffer reference
        // 3. output_buffer.metal_buffer() returns a valid MTLBuffer reference
        // 4. Buffer indices 0, 1, 2 match the kernel's buffer arguments
        // 5. params_ptr is stack-allocated and valid for the duration of encoding
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(self.output_buffer.metal_buffer()), 0, 1);

            // Set vocab_size as constant
            let vocab_size = self.config.vocab_size as u32;
            let params_ptr = NonNull::from(&vocab_size).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of::<u32>(), 2);
        }

        // Allocate threadgroup memory for SIMD reduction
        let simd_groups = (self.config.threadgroup_size + 31) / 32;
        // SAFETY:
        // 1. encoder is still valid (encoding not yet ended)
        // 2. Memory sizes are computed correctly based on simd_groups and type sizes
        // 3. Indices 0 and 1 match the kernel's threadgroup memory declarations
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(
                simd_groups * std::mem::size_of::<f32>(),
                0,
            );
            encoder.setThreadgroupMemoryLength_atIndex(
                simd_groups * std::mem::size_of::<u32>(),
                1,
            );
        }

        // Dispatch - single threadgroup is enough for argmax
        let grid_size = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: self.config.threadgroup_size,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Check for errors
        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        // Read result
        Ok(self.output_buffer.as_slice()[0])
    }

    /// Perform temperature sampling with filtering.
    ///
    /// # Arguments
    /// * `logits` - Logits buffer of shape [vocab_size] as f32.
    /// * `temperature` - Sampling temperature (higher = more random).
    /// * `top_k` - Top-K filtering (0 = disabled).
    /// * `top_p` - Top-P nucleus sampling threshold.
    /// * `min_p` - Min-P threshold relative to max probability.
    ///
    /// # Returns
    /// A sampled token ID.
    pub fn sample(
        &self,
        logits: &impl AsMetalBuffer,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
    ) -> Result<u32> {
        // For temperature = 0, use argmax
        if temperature == 0.0 {
            return self.argmax(logits);
        }

        // Get or create the pipeline
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "fused_sample_small", None)?
        };

        // Create command buffer and encoder
        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        // Create sampling params
        let params = SamplingParams {
            vocab_size: self.config.vocab_size as u32,
            temperature,
            top_p,
            min_p,
            top_k,
            random_seed: self.next_seed(),
            do_sample: true,
            _padding: [0; 3],
        };

        // Set buffers
        // SAFETY:
        // 1. encoder is a valid MTLComputeCommandEncoder from the command buffer
        // 2. logits.metal_buffer() returns a valid MTLBuffer reference
        // 3. output_buffer.metal_buffer() returns a valid MTLBuffer reference
        // 4. Buffer indices 0, 1, 2 match the kernel's buffer arguments
        // 5. params is stack-allocated and repr(C) for stable ABI
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(self.output_buffer.metal_buffer()), 0, 1);

            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of::<SamplingParams>(), 2);
        }

        // Allocate threadgroup memory
        // Each thread stores 4 local candidates, so we need tg_size * 4 entries
        const LOCAL_TOP_L: usize = 4;
        let tg_size = self.config.threadgroup_size;
        // SAFETY:
        // 1. encoder is still valid (encoding not yet ended)
        // 2. Memory sizes are computed correctly based on tg_size, LOCAL_TOP_L, and type sizes
        // 3. Indices 0 and 1 match the kernel's threadgroup memory declarations
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(
                tg_size * LOCAL_TOP_L * std::mem::size_of::<f32>(),
                0,
            );
            encoder.setThreadgroupMemoryLength_atIndex(
                tg_size * LOCAL_TOP_L * std::mem::size_of::<u32>(),
                1,
            );
        }

        // Dispatch
        let grid_size = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Check for errors
        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        // Read result
        Ok(self.output_buffer.as_slice()[0])
    }

    /// Dispatch argmax kernel asynchronously without waiting.
    ///
    /// Returns a command buffer handle that can be used to wait for completion.
    /// Call `read_result()` after the command buffer completes to get the token.
    ///
    /// # Arguments
    /// * `logits` - Logits buffer of shape [vocab_size] as f32.
    ///
    /// # Returns
    /// A command buffer handle for synchronization.
    pub fn argmax_async(
        &self,
        logits: &impl AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>> {
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "fused_argmax_simd", None)?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        // SAFETY: Same invariants as argmax() - see comments there
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(self.output_buffer.metal_buffer()), 0, 1);

            let vocab_size = self.config.vocab_size as u32;
            let params_ptr = NonNull::from(&vocab_size).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of::<u32>(), 2);
        }

        let simd_groups = (self.config.threadgroup_size + 31) / 32;
        // SAFETY: Same invariants as argmax() - see comments there
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(
                simd_groups * std::mem::size_of::<f32>(),
                0,
            );
            encoder.setThreadgroupMemoryLength_atIndex(
                simd_groups * std::mem::size_of::<u32>(),
                1,
            );
        }

        let grid_size = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: self.config.threadgroup_size,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();

        command_buffer.commit();
        // Note: Do NOT call waitUntilCompleted() - caller handles sync

        Ok(command_buffer)
    }

    /// Dispatch sampling kernel asynchronously without waiting.
    ///
    /// Returns a command buffer handle that can be used to wait for completion.
    /// Call `read_result()` after the command buffer completes to get the token.
    pub fn sample_async(
        &self,
        logits: &impl AsMetalBuffer,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
    ) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>> {
        // For temperature = 0, use argmax
        if temperature == 0.0 {
            return self.argmax_async(logits);
        }

        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "fused_sample_small", None)?
        };

        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        let params = SamplingParams {
            vocab_size: self.config.vocab_size as u32,
            temperature,
            top_p,
            min_p,
            top_k,
            random_seed: self.next_seed(),
            do_sample: true,
            _padding: [0; 3],
        };

        // SAFETY: Same invariants as sample() - see comments there
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(self.output_buffer.metal_buffer()), 0, 1);

            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of::<SamplingParams>(), 2);
        }

        // Each thread stores 4 local candidates, so we need tg_size * 4 entries
        const LOCAL_TOP_L: usize = 4;
        let tg_size = self.config.threadgroup_size;
        // SAFETY: Same invariants as sample() - see comments there
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(
                tg_size * LOCAL_TOP_L * std::mem::size_of::<f32>(),
                0,
            );
            encoder.setThreadgroupMemoryLength_atIndex(
                tg_size * LOCAL_TOP_L * std::mem::size_of::<u32>(),
                1,
            );
        }

        let grid_size = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();

        command_buffer.commit();
        // Note: Do NOT call waitUntilCompleted() - caller handles sync

        Ok(command_buffer)
    }

    /// Read the result from the output buffer.
    ///
    /// Call this after ensuring the command buffer has completed.
    #[inline]
    pub fn read_result(&self) -> u32 {
        self.output_buffer.as_slice()[0]
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

impl std::fmt::Debug for FusedSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedSampler")
            .field("vocab_size", &self.config.vocab_size)
            .field("threadgroup_size", &self.config.threadgroup_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_params_size() {
        // Ensure the struct is properly aligned for Metal
        assert_eq!(std::mem::size_of::<SamplingParams>(), 28);
    }

    #[test]
    fn test_fused_sampler_creation() {
        let sampler = FusedSampler::new(32000);
        assert!(sampler.is_ok(), "Should create FusedSampler on macOS");
    }
}
