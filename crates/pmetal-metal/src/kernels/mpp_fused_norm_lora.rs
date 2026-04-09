#![allow(unsafe_code)]

//! Metal 4 / MPP Fused RMSNorm + Linear + LoRA dispatch.
//!
//! Fuses RMSNorm with the subsequent linear projection and LoRA overlay using
//! Metal Performance Primitives on M5+ (Apple10) GPUs with NAX cores.
//!
//! Single kernel launch:
//!   Phase 1: RMSNorm(x) — cooperative SIMD reduction per token
//!   Phase 2: base = norm_x @ W^T — via SIMD dot products per output element
//!   Phase 3: lora_out = scale * (norm_x @ A^T) @ B^T — small-rank LoRA
//!   Phase 4: output = base + lora_out

use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLComputeCommandEncoder};

use crate::{
    buffer::AsMetalBuffer,
    context::MetalContext,
    error::{MetalError, Result},
    kernels::mpp_dispatch::encode_mpp_kernel,
};

/// Configuration for MPP Fused RMSNorm + Linear + LoRA.
#[derive(Debug, Clone)]
pub struct MppFusedNormLoraConfig {
    /// Batch size (token count).
    pub batch_size: usize,
    /// Input hidden dimension.
    pub hidden_size: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// LoRA rank (0 to disable LoRA path).
    pub lora_rank: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// LoRA scaling factor (alpha / rank).
    pub lora_scale: f32,
}

impl MppFusedNormLoraConfig {
    /// Create a config for the fused norm-lora kernel.
    pub fn new(batch_size: usize, hidden_size: usize, out_features: usize) -> Self {
        Self {
            batch_size,
            hidden_size,
            out_features,
            lora_rank: 0,
            eps: 1e-6,
            lora_scale: 1.0,
        }
    }

    /// Output buffer element count.
    pub fn output_size(&self) -> usize {
        self.batch_size * self.out_features
    }
}

/// Metal-side parameter block (must match `FusedNormLoraParams` in Metal).
#[repr(C)]
struct FusedNormLoraParams {
    batch_size: u32,
    hidden_size: u32,
    out_features: u32,
    lora_rank: u32,
    eps: f32,
    lora_scale: f32,
}

#[derive(Debug, Clone, Copy)]
struct DispatchGeometry {
    /// Number of output tiles per token row.
    num_out_tiles: usize,
    /// Threads per threadgroup: 4 simdgroups × 32 = 128.
    threads_per_threadgroup: usize,
}

fn dispatch_geometry(config: &MppFusedNormLoraConfig) -> DispatchGeometry {
    const BN: usize = 64;
    DispatchGeometry {
        num_out_tiles: config.out_features.div_ceil(BN),
        threads_per_threadgroup: 4 * 32,
    }
}

/// MPP Fused RMSNorm + Linear + LoRA dispatcher.
///
/// Dispatches to `mpp_fused_norm_lora_forward_f16` on M5+ hardware.
pub struct MppFusedNormLora {
    ctx: Arc<MetalContext>,
    config: MppFusedNormLoraConfig,
}

impl MppFusedNormLora {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppFusedNormLoraConfig) -> Self {
        Self { ctx, config }
    }

    /// Check if MPP Fused NormLora is available (requires M5+ with NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute synchronously.
    ///
    /// `input`: `[batch, hidden]` fp16, `gamma`: `[hidden]` fp16 RMSNorm weights,
    /// `weight`: `[out_features, hidden]` fp16, `lora_a`: `[rank, hidden]` fp16,
    /// `lora_b`: `[out_features, rank]` fp16, `output`: `[batch, out_features]` fp16.
    ///
    /// When `config.lora_rank == 0`, `lora_a` and `lora_b` are not accessed by
    /// the kernel but the Metal API still requires valid buffer objects at their
    /// indices. Pass any non-null zero-length placeholder buffer.
    pub fn execute(
        &self,
        input: &dyn AsMetalBuffer,
        gamma: &dyn AsMetalBuffer,
        weight: &dyn AsMetalBuffer,
        lora_a: &dyn AsMetalBuffer,
        lora_b: &dyn AsMetalBuffer,
        output: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let command_buffer =
            self.execute_async(input, gamma, weight, lora_a, lora_b, output)?;
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Execute asynchronously and return the submitted command buffer.
    pub fn execute_async(
        &self,
        input: &dyn AsMetalBuffer,
        gamma: &dyn AsMetalBuffer,
        weight: &dyn AsMetalBuffer,
        lora_a: &dyn AsMetalBuffer,
        lora_b: &dyn AsMetalBuffer,
        output: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused NormLora not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let geometry = dispatch_geometry(&self.config);

        let params = FusedNormLoraParams {
            batch_size: self.config.batch_size as u32,
            hidden_size: self.config.hidden_size as u32,
            out_features: self.config.out_features as u32,
            lora_rank: self.config.lora_rank as u32,
            eps: self.config.eps,
            lora_scale: self.config.lora_scale,
        };

        // Grid: [num_out_tiles, batch_size, 1] — one threadgroup per token per output tile
        let grid = objc2_metal::MTLSize {
            width: geometry.num_out_tiles,
            height: self.config.batch_size,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: geometry.threads_per_threadgroup,
            height: 1,
            depth: 1,
        };

        let input_buf = input.as_metal_buffer();
        let gamma_buf = gamma.as_metal_buffer();
        let weight_buf = weight.as_metal_buffer();
        let lora_a_buf = lora_a.as_metal_buffer();
        let lora_b_buf = lora_b.as_metal_buffer();
        let output_buf = output.as_metal_buffer();

        encode_mpp_kernel(&self.ctx, "mpp_fused_norm_lora_forward_f16", grid, tg_size, |encoder| unsafe {
            // buffer(0): input, buffer(1): gamma, buffer(2): weight,
            // buffer(3): lora_a, buffer(4): lora_b, buffer(5): output,
            // buffer(6): params
            encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gamma_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weight_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(lora_a_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(lora_b_buf), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 5);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 6);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_output_size() {
        let config = MppFusedNormLoraConfig::new(8, 4096, 4096);
        assert_eq!(config.output_size(), 8 * 4096);
    }

    #[test]
    fn test_dispatch_geometry_tile_counts() {
        let config = MppFusedNormLoraConfig::new(4, 4096, 4096);
        let geom = dispatch_geometry(&config);
        // 4096 / 64 = 64 output tiles
        assert_eq!(geom.num_out_tiles, 64);
        assert_eq!(geom.threads_per_threadgroup, 128);
    }

    #[test]
    fn test_dispatch_geometry_non_aligned_out_features() {
        let config = MppFusedNormLoraConfig::new(1, 2048, 100);
        let geom = dispatch_geometry(&config);
        // ceil(100 / 64) = 2
        assert_eq!(geom.num_out_tiles, 2);
    }

    #[test]
    fn test_config_defaults() {
        let config = MppFusedNormLoraConfig::new(1, 2048, 4096);
        assert_eq!(config.lora_rank, 0);
        assert!((config.eps - 1e-6).abs() < 1e-10);
        assert!((config.lora_scale - 1.0).abs() < 1e-6);
    }
}
