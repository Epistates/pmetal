#![allow(unsafe_code)]

//! Metal 4 / MPP Fused RoPE dispatch.
//!
//! Provides hardware-accelerated Rotary Position Embedding via Metal Performance
//! Primitives on M5+ (Apple10) GPUs with NAX cores.
//!
//! Replaces the Metal 3 `rope_inplace` / `rope_with_positions` / `rope_qk_inplace`
//! kernels, which use threadgroup caches and THREADS_PER_HEAD=64 thread groups.
//! The MPP version uses a single SIMD group (32 lanes) per token-head, with
//! cross-lane stride eliminating threadgroup memory.
//!
//! Kernel families:
//! - `mpp_rope_inplace_{f32,f16}` — sequential positions
//! - `mpp_rope_with_positions_{f32,f16}` — custom position IDs (seq packing)
//! - `mpp_rope_qk_inplace_f16` — fused QK RoPE in one dispatch
//! - `mpp_rope_qk_with_positions_f16` — QK RoPE with position IDs
//!
//! Grid layout for standalone: `[batch, heads, seq_len]`
//! Each threadgroup = one SIMD group (32 lanes) covering half_dim pairs.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::{
    buffer::AsMetalBuffer,
    context::MetalContext,
    error::{MetalError, Result},
};

// =============================================================================
// Config
// =============================================================================

/// Configuration for the MPP Fused RoPE kernel.
#[derive(Debug, Clone)]
pub struct MppFusedRoPEConfig {
    /// Batch size.
    pub batch_size: usize,
    /// Number of Q attention heads.
    pub num_heads: usize,
    /// Number of KV heads (for GQA; equals `num_heads` for MHA).
    pub num_kv_heads: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Head dimension (must be even).
    pub head_dim: usize,
    /// RoPE base frequency (default 10000.0).
    pub base: f32,
    /// Position scale factor (default 1.0).
    pub scale: f32,
    /// Use fp16 (true) or fp32 (false) kernel.
    pub use_fp16: bool,
}

impl MppFusedRoPEConfig {
    /// Create a standard RoPE config (MHA, sequential positions, fp32).
    pub fn new(batch_size: usize, num_heads: usize, seq_len: usize, head_dim: usize) -> Self {
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

    /// Configure for GQA with different Q and KV head counts.
    pub fn with_gqa(mut self, num_kv_heads: usize) -> Self {
        self.num_kv_heads = num_kv_heads;
        self
    }

    /// Enable fp16 kernel.
    pub fn with_fp16(mut self) -> Self {
        self.use_fp16 = true;
        self
    }

    /// Set custom RoPE base frequency.
    pub fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    /// Set position scale factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

// =============================================================================
// Metal-side parameter block (must match MppRoPEParams in Metal)
// =============================================================================

#[repr(C)]
struct MppRoPEParamsMetal {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    base: f32,
    scale: f32,
}

impl MppRoPEParamsMetal {
    fn from_config(config: &MppFusedRoPEConfig) -> Self {
        Self {
            batch_size: config.batch_size as u32,
            num_heads: config.num_heads as u32,
            seq_len: config.seq_len as u32,
            head_dim: config.head_dim as u32,
            base: config.base,
            scale: config.scale,
        }
    }
}

// =============================================================================
// Dispatcher
// =============================================================================

/// MPP Fused RoPE dispatcher.
///
/// Dispatches in-place RoPE to `mpp_rope_inplace_{f32,f16}` on M5+ hardware.
pub struct MppFusedRoPE {
    ctx: Arc<MetalContext>,
    config: MppFusedRoPEConfig,
}

impl MppFusedRoPE {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppFusedRoPEConfig) -> Self {
        Self { ctx, config }
    }

    /// Returns true when MPP RoPE is available (requires M5+ NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Apply RoPE in-place with sequential positions `[0, 1, 2, ...]`.
    ///
    /// `x`: `[batch, heads, seq_len, head_dim]` (in-place).
    pub fn apply_inplace(&self, x: &dyn AsMetalBuffer) -> Result<()> {
        let cb = self.apply_inplace_async(x)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Apply RoPE in-place asynchronously.
    pub fn apply_inplace_async(
        &self,
        x: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let kernel_name = if self.config.use_fp16 {
            "mpp_rope_inplace_f16"
        } else {
            "mpp_rope_inplace_f32"
        };
        self.dispatch_1buf(x, None, kernel_name, self.config.num_heads, false)
    }

    /// Apply RoPE with custom position IDs (for sequence packing).
    ///
    /// `x`: `[batch, heads, seq_len, head_dim]` (in-place).
    /// `position_ids`: `[seq_len]` (i32).
    pub fn apply_with_positions(
        &self,
        x: &dyn AsMetalBuffer,
        position_ids: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.apply_with_positions_async(x, position_ids)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Apply RoPE with custom position IDs asynchronously.
    pub fn apply_with_positions_async(
        &self,
        x: &dyn AsMetalBuffer,
        position_ids: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let kernel_name = if self.config.use_fp16 {
            "mpp_rope_with_positions_f16"
        } else {
            "mpp_rope_with_positions_f32"
        };
        self.dispatch_1buf(x, Some(position_ids), kernel_name, self.config.num_heads, true)
    }

    /// Apply RoPE to both Q and K in a single dispatch (fp16).
    ///
    /// `q`: `[batch, num_heads, seq_len, head_dim]` (in-place).
    /// `k`: `[batch, num_kv_heads, seq_len, head_dim]` (in-place).
    pub fn apply_qk_inplace(&self, q: &dyn AsMetalBuffer, k: &dyn AsMetalBuffer) -> Result<()> {
        let cb = self.apply_qk_inplace_async(q, k, None)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Apply QK RoPE asynchronously, with optional position IDs.
    pub fn apply_qk_inplace_async(
        &self,
        q: &dyn AsMetalBuffer,
        k: &dyn AsMetalBuffer,
        position_ids: Option<&dyn AsMetalBuffer>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused RoPE not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let kernel_name = if position_ids.is_some() {
            "mpp_rope_qk_with_positions_f16"
        } else {
            "mpp_rope_qk_inplace_f16"
        };

        let constants: HashMap<u64, crate::pipeline::FunctionConstant> = HashMap::new();
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_metal4_pipeline(self.ctx.device(), kernel_name, &constants)?
        };

        let command_buffer = self
            .ctx
            .command_queue()
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        let params = MppRoPEParamsMetal::from_config(&self.config);
        let kv_heads = self.config.num_kv_heads as u32;

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.as_metal_buffer()), 0, 1);

            if let Some(pos_ids) = position_ids {
                // mpp_rope_qk_with_positions_f16: pos_ids at buffer 2, params at 3, kv_heads at 4
                encoder.setBuffer_offset_atIndex(Some(pos_ids.as_metal_buffer()), 0, 2);
                let p_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 3);
                let kv_ptr = NonNull::from(&kv_heads).cast();
                encoder.setBytes_length_atIndex(kv_ptr, std::mem::size_of_val(&kv_heads), 4);
            } else {
                // mpp_rope_qk_inplace_f16: params at buffer 2, kv_heads at 3
                let p_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 2);
                let kv_ptr = NonNull::from(&kv_heads).cast();
                encoder.setBytes_length_atIndex(kv_ptr, std::mem::size_of_val(&kv_heads), 3);
            }
        }

        // Grid: [batch_size, seq_len, 1]  Threadgroup: [32, 1, 1]
        let threadgroup_size = objc2_metal::MTLSize { width: 32, height: 1, depth: 1 };
        let grid_size = objc2_metal::MTLSize {
            width: self.config.batch_size,
            height: self.config.seq_len,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();

        Ok(command_buffer)
    }

    // Internal helper for single-buffer kernels (inplace / with_positions).
    fn dispatch_1buf(
        &self,
        x: &dyn AsMetalBuffer,
        position_ids: Option<&dyn AsMetalBuffer>,
        kernel_name: &str,
        _heads: usize,
        has_positions: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused RoPE not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let constants: HashMap<u64, crate::pipeline::FunctionConstant> = HashMap::new();
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_metal4_pipeline(self.ctx.device(), kernel_name, &constants)?
        };

        let command_buffer = self
            .ctx
            .command_queue()
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        let params = MppRoPEParamsMetal::from_config(&self.config);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.as_metal_buffer()), 0, 0);
            if has_positions {
                if let Some(pos) = position_ids {
                    encoder.setBuffer_offset_atIndex(Some(pos.as_metal_buffer()), 0, 1);
                }
                let p_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 2);
            } else {
                let p_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 1);
            }
        }

        // Grid: [batch, heads, seq_len]  Threadgroup: [32, 1, 1]
        let threadgroup_size = objc2_metal::MTLSize { width: 32, height: 1, depth: 1 };
        let grid_size = objc2_metal::MTLSize {
            width: self.config.batch_size,
            height: self.config.num_heads,
            depth: self.config.seq_len,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();

        Ok(command_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = MppFusedRoPEConfig::new(2, 8, 128, 64);
        assert_eq!(cfg.batch_size, 2);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_kv_heads, 8); // matches num_heads by default
        assert_eq!(cfg.seq_len, 128);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.base, 10000.0);
        assert_eq!(cfg.scale, 1.0);
        assert!(!cfg.use_fp16);
    }

    #[test]
    fn test_gqa_config() {
        let cfg = MppFusedRoPEConfig::new(1, 32, 64, 128).with_gqa(8);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
    }

    #[test]
    fn test_grid_size() {
        let cfg = MppFusedRoPEConfig::new(2, 4, 16, 64);
        // grid should be [2, 4, 16] for standalone
        assert_eq!(cfg.batch_size, 2);
        assert_eq!(cfg.num_heads, 4);
        assert_eq!(cfg.seq_len, 16);
    }
}
