#![allow(unsafe_code)]

//! Metal 4 / MPP Fused LoRA dispatch.
//!
//! Provides hardware-accelerated fused LoRA forward pass via Metal Performance
//! Primitives on M5+ (Apple10) GPUs with NAX cores.
//!
//! All three phases use hardware matrix multiply:
//!   Phase 1: y = x @ W^T              via MPP matmul2d
//!   Phase 2: xA = x @ A^T             via SIMD dot products
//!   Phase 3: y += scale * xA @ B^T    via SIMD dot products
//!
//! Two kernel variants:
//! - Training: saves xA intermediate for backward pass (`mpp_fused_lora_forward_f16`)
//! - Inference: skips xA save (`mpp_lora_forward_inference_f16`)

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

/// Whether to use the training variant (saves xA for backward) or inference variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MppFusedLoraMode {
    /// Training: saves `xA = x @ A^T` to `xA_out` for the backward pass.
    Training,
    /// Inference: skips saving `xA`, no `xA_out` buffer required.
    Inference,
}

/// Configuration for MPP Fused LoRA.
#[derive(Debug, Clone)]
pub struct MppFusedLoraConfig {
    /// Batch size (token count).
    pub batch_size: usize,
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// LoRA rank.
    pub rank: usize,
    /// LoRA scaling factor (alpha / rank).
    pub scale: f32,
    /// Learning rate scale for A matrix (used in some LoRA variants).
    pub lr_scale_a: f32,
    /// Learning rate scale for B matrix.
    pub lr_scale_b: f32,
    /// Training or inference mode.
    pub mode: MppFusedLoraMode,
}

impl MppFusedLoraConfig {
    /// Create a training config.
    pub fn new_training(
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
            lr_scale_a: 1.0,
            lr_scale_b: 1.0,
            mode: MppFusedLoraMode::Training,
        }
    }

    /// Create an inference config (no xA_out buffer needed).
    pub fn new_inference(
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
            lr_scale_a: 1.0,
            lr_scale_b: 1.0,
            mode: MppFusedLoraMode::Inference,
        }
    }

    /// Output buffer element count.
    pub fn output_size(&self) -> usize {
        self.batch_size * self.out_features
    }

    /// xA intermediate buffer element count (only needed for training mode).
    pub fn xa_size(&self) -> usize {
        self.batch_size * self.rank
    }
}

/// Metal-side parameter block (must match `FusedLoraParams` in Metal).
#[repr(C)]
struct FusedLoraParams {
    batch_size: u32,
    in_features: u32,
    out_features: u32,
    rank: u32,
    scale: f32,
    lr_scale_a: f32,
    lr_scale_b: f32,
}

#[derive(Debug, Clone, Copy)]
struct DispatchGeometry {
    num_batch_tiles: usize,
    num_out_tiles: usize,
    /// Threads per threadgroup: 4 simdgroups × 32 = 128.
    threads_per_threadgroup: usize,
}

fn dispatch_geometry(config: &MppFusedLoraConfig) -> DispatchGeometry {
    const BM: usize = 64;
    const BN: usize = 64;
    DispatchGeometry {
        num_batch_tiles: config.batch_size.div_ceil(BM),
        num_out_tiles: config.out_features.div_ceil(BN),
        threads_per_threadgroup: 4 * 32,
    }
}

fn kernel_name(config: &MppFusedLoraConfig) -> &'static str {
    match config.mode {
        MppFusedLoraMode::Training => "mpp_fused_lora_forward_f16",
        MppFusedLoraMode::Inference => "mpp_lora_forward_inference_f16",
    }
}

/// MPP Fused LoRA dispatcher.
///
/// Dispatches to the appropriate training or inference kernel on M5+ hardware.
pub struct MppFusedLora {
    ctx: Arc<MetalContext>,
    config: MppFusedLoraConfig,
}

impl MppFusedLora {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppFusedLoraConfig) -> Self {
        Self { ctx, config }
    }

    /// Check if MPP Fused LoRA is available (requires M5+ with NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute the training variant synchronously.
    ///
    /// `x`: `[batch, in_features]` fp16, `W`: `[out_features, in_features]` fp16,
    /// `A`: `[rank, in_features]` fp16, `B`: `[out_features, rank]` fp16,
    /// `y`: `[batch, out_features]` fp16 (output), `xa_out`: `[batch, rank]` fp16
    /// (saved xA intermediate for backward pass).
    pub fn execute_training(
        &self,
        x: &dyn AsMetalBuffer,
        w: &dyn AsMetalBuffer,
        a: &dyn AsMetalBuffer,
        b: &dyn AsMetalBuffer,
        y: &dyn AsMetalBuffer,
        xa_out: &dyn AsMetalBuffer,
    ) -> Result<()> {
        if self.config.mode != MppFusedLoraMode::Training {
            return Err(MetalError::InvalidConfig(
                "execute_training called on inference-mode MppFusedLora".to_string(),
            ));
        }
        let command_buffer = self.execute_training_async(x, w, a, b, y, xa_out)?;
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Execute the inference variant synchronously.
    ///
    /// `x`: `[batch, in_features]` fp16, `W`: `[out_features, in_features]` fp16,
    /// `A`: `[rank, in_features]` fp16, `B`: `[out_features, rank]` fp16,
    /// `y`: `[batch, out_features]` fp16 (output).
    pub fn execute_inference(
        &self,
        x: &dyn AsMetalBuffer,
        w: &dyn AsMetalBuffer,
        a: &dyn AsMetalBuffer,
        b: &dyn AsMetalBuffer,
        y: &dyn AsMetalBuffer,
    ) -> Result<()> {
        if self.config.mode != MppFusedLoraMode::Inference {
            return Err(MetalError::InvalidConfig(
                "execute_inference called on training-mode MppFusedLora".to_string(),
            ));
        }
        let command_buffer = self.execute_inference_async(x, w, a, b, y)?;
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }

    /// Asynchronous training dispatch.
    pub fn execute_training_async(
        &self,
        x: &dyn AsMetalBuffer,
        w: &dyn AsMetalBuffer,
        a: &dyn AsMetalBuffer,
        b: &dyn AsMetalBuffer,
        y: &dyn AsMetalBuffer,
        xa_out: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused LoRA not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let geometry = dispatch_geometry(&self.config);

        let params = FusedLoraParams {
            batch_size: self.config.batch_size as u32,
            in_features: self.config.in_features as u32,
            out_features: self.config.out_features as u32,
            rank: self.config.rank as u32,
            scale: self.config.scale,
            lr_scale_a: self.config.lr_scale_a,
            lr_scale_b: self.config.lr_scale_b,
        };

        // Grid: [num_out_tiles, num_batch_tiles, 1]
        let grid = objc2_metal::MTLSize {
            width: geometry.num_out_tiles,
            height: geometry.num_batch_tiles,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: geometry.threads_per_threadgroup,
            height: 1,
            depth: 1,
        };

        let x_buf = x.as_metal_buffer();
        let w_buf = w.as_metal_buffer();
        let a_buf = a.as_metal_buffer();
        let b_buf = b.as_metal_buffer();
        let y_buf = y.as_metal_buffer();
        let xa_buf = xa_out.as_metal_buffer();

        encode_mpp_kernel(&self.ctx, "mpp_fused_lora_forward_f16", grid, tg_size, |encoder| unsafe {
            // Training: buffer(0): x, buffer(1): W, buffer(2): A, buffer(3): B,
            //           buffer(4): y, buffer(5): xA_out, buffer(6): params
            encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y_buf), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(xa_buf), 0, 5);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 6);
        })
    }

    /// Asynchronous inference dispatch.
    pub fn execute_inference_async(
        &self,
        x: &dyn AsMetalBuffer,
        w: &dyn AsMetalBuffer,
        a: &dyn AsMetalBuffer,
        b: &dyn AsMetalBuffer,
        y: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused LoRA not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let geometry = dispatch_geometry(&self.config);

        let params = FusedLoraParams {
            batch_size: self.config.batch_size as u32,
            in_features: self.config.in_features as u32,
            out_features: self.config.out_features as u32,
            rank: self.config.rank as u32,
            scale: self.config.scale,
            lr_scale_a: self.config.lr_scale_a,
            lr_scale_b: self.config.lr_scale_b,
        };

        // Grid: [num_out_tiles, num_batch_tiles, 1]
        let grid = objc2_metal::MTLSize {
            width: geometry.num_out_tiles,
            height: geometry.num_batch_tiles,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: geometry.threads_per_threadgroup,
            height: 1,
            depth: 1,
        };

        let x_buf = x.as_metal_buffer();
        let w_buf = w.as_metal_buffer();
        let a_buf = a.as_metal_buffer();
        let b_buf = b.as_metal_buffer();
        let y_buf = y.as_metal_buffer();

        encode_mpp_kernel(&self.ctx, "mpp_lora_forward_inference_f16", grid, tg_size, |encoder| unsafe {
            // Inference: buffer(0): x, buffer(1): W, buffer(2): A, buffer(3): B,
            //            buffer(4): y, buffer(5): params
            encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y_buf), 0, 4);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 5);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_sizes() {
        let config = MppFusedLoraConfig::new_training(8, 2048, 4096, 16, 1.0);
        assert_eq!(config.output_size(), 8 * 4096);
        assert_eq!(config.xa_size(), 8 * 16);
    }

    #[test]
    fn test_dispatch_geometry_tile_counts() {
        let config = MppFusedLoraConfig::new_training(128, 2048, 4096, 16, 1.0);
        let geom = dispatch_geometry(&config);
        assert_eq!(geom.num_batch_tiles, 2);  // 128 / 64
        assert_eq!(geom.num_out_tiles, 64);   // 4096 / 64
        assert_eq!(geom.threads_per_threadgroup, 128);
    }

    #[test]
    fn test_dispatch_geometry_non_aligned() {
        let config = MppFusedLoraConfig::new_inference(65, 1024, 1024, 8, 0.5);
        let geom = dispatch_geometry(&config);
        assert_eq!(geom.num_batch_tiles, 2);  // ceil(65/64)
        assert_eq!(geom.num_out_tiles, 16);   // 1024 / 64
    }

    #[test]
    fn test_kernel_name_selects_mode() {
        let training = MppFusedLoraConfig::new_training(1, 2048, 4096, 16, 1.0);
        assert_eq!(kernel_name(&training), "mpp_fused_lora_forward_f16");

        let inference = MppFusedLoraConfig::new_inference(1, 2048, 4096, 16, 1.0);
        assert_eq!(kernel_name(&inference), "mpp_lora_forward_inference_f16");
    }

    #[test]
    fn test_training_defaults() {
        let config = MppFusedLoraConfig::new_training(4, 2048, 4096, 16, 0.5);
        assert_eq!(config.mode, MppFusedLoraMode::Training);
        assert!((config.lr_scale_a - 1.0).abs() < 1e-6);
        assert!((config.lr_scale_b - 1.0).abs() < 1e-6);
    }
}
