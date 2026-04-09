#![allow(unsafe_code)]

//! Metal 4 / MPP Fused SwiGLU dispatch.
//!
//! Provides hardware-accelerated fused SwiGLU MLP via Metal Performance Primitives
//! on M5+ (Apple10) GPUs with NAX cores.
//!
//! Computes: output = silu(x @ gate_W^T) * (x @ up_W^T)
//!
//! Single kernel launch combines both projections and the activation, eliminating
//! intermediate buffer round-trips that would stall the memory bus.

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

/// Configuration for MPP Fused SwiGLU.
#[derive(Debug, Clone)]
pub struct MppFusedSwiGLUConfig {
    /// Batch size (token count).
    pub batch_size: usize,
    /// Input hidden dimension.
    pub hidden_size: usize,
    /// Intermediate (output) dimension.
    pub intermediate_size: usize,
    /// Use fp16 (true) or fp32 (false).
    pub use_fp16: bool,
}

impl MppFusedSwiGLUConfig {
    /// Create a new config for `output[batch, intermediate] = silu(x @ gate_W^T) * (x @ up_W^T)`.
    pub fn new(batch_size: usize, hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            batch_size,
            hidden_size,
            intermediate_size,
            use_fp16: true,
        }
    }

    /// Output buffer element count.
    pub fn output_size(&self) -> usize {
        self.batch_size * self.intermediate_size
    }
}

/// Metal-side parameter block (must match `FusedSwiGLUParams` in Metal).
#[repr(C)]
struct FusedSwiGLUParams {
    batch_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    lora_rank: u32,
    lora_scale: f32,
}

#[derive(Debug, Clone, Copy)]
struct DispatchGeometry {
    /// Threadgroups in the intermediate (x) dimension.
    num_tiles_intermediate: usize,
    /// Threadgroups in the batch (y) dimension.
    num_tiles_batch: usize,
    /// Threads per threadgroup: 32 (single SIMD group).
    ///
    /// The shader uses `execution_simdgroup` (MPP Guide §2.3.1 single-simdgroup
    /// pattern). Only one simdgroup per threadgroup is active; dispatching 128
    /// would waste 96 threads per threadgroup with zero benefit.
    threads_per_threadgroup: usize,
}

fn dispatch_geometry(config: &MppFusedSwiGLUConfig) -> DispatchGeometry {
    // The shader tile sizes are BM=32, BN=32 (single-simdgroup 32×32 tile).
    // The dispatch tile sizes must match to avoid under-covering the output.
    const BM: usize = 32;
    const BN: usize = 32;
    DispatchGeometry {
        num_tiles_intermediate: config.intermediate_size.div_ceil(BN),
        num_tiles_batch: config.batch_size.div_ceil(BM),
        threads_per_threadgroup: 32,
    }
}

fn kernel_name(config: &MppFusedSwiGLUConfig) -> &'static str {
    if config.use_fp16 {
        "mpp_fused_swiglu_forward_f16"
    } else {
        "mpp_fused_swiglu_forward_f32"
    }
}

/// MPP Fused SwiGLU dispatcher.
///
/// Dispatches to `mpp_fused_swiglu_forward_{f16,f32}` on M5+ hardware.
pub struct MppFusedSwiGLU {
    ctx: Arc<MetalContext>,
    config: MppFusedSwiGLUConfig,
}

impl MppFusedSwiGLU {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppFusedSwiGLUConfig) -> Self {
        Self { ctx, config }
    }

    /// Check if MPP Fused SwiGLU is available (requires M5+ with NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute synchronously.
    ///
    /// `input`: `[batch, hidden]`, `gate_weight`: `[intermediate, hidden]`,
    /// `up_weight`: `[intermediate, hidden]`, `output`: `[batch, intermediate]`.
    pub fn execute(
        &self,
        input: &dyn AsMetalBuffer,
        gate_weight: &dyn AsMetalBuffer,
        up_weight: &dyn AsMetalBuffer,
        output: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let command_buffer = self.execute_async(input, gate_weight, up_weight, output)?;
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
        gate_weight: &dyn AsMetalBuffer,
        up_weight: &dyn AsMetalBuffer,
        output: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused SwiGLU not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let geometry = dispatch_geometry(&self.config);
        let kernel_name = kernel_name(&self.config);

        let params = FusedSwiGLUParams {
            batch_size: self.config.batch_size as u32,
            hidden_size: self.config.hidden_size as u32,
            intermediate_size: self.config.intermediate_size as u32,
            // No LoRA for this basic dispatch variant.
            lora_rank: 0,
            lora_scale: 0.0,
        };

        // Grid: [num_intermediate_tiles, num_batch_tiles, 1]
        let grid = objc2_metal::MTLSize {
            width: geometry.num_tiles_intermediate,
            height: geometry.num_tiles_batch,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: geometry.threads_per_threadgroup,
            height: 1,
            depth: 1,
        };

        let input_buf = input.as_metal_buffer();
        let gate_buf = gate_weight.as_metal_buffer();
        let up_buf = up_weight.as_metal_buffer();
        let output_buf = output.as_metal_buffer();

        encode_mpp_kernel(&self.ctx, kernel_name, grid, tg_size, |encoder| unsafe {
            // buffer(0): input, buffer(1): gate_weight, buffer(2): up_weight,
            // buffer(3): output, buffer(4): params
            encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(up_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 3);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_output_size() {
        let config = MppFusedSwiGLUConfig::new(4, 2048, 8192);
        assert_eq!(config.output_size(), 4 * 8192);
    }

    #[test]
    fn test_dispatch_geometry_tile_counts() {
        let config = MppFusedSwiGLUConfig::new(1, 2048, 8192);
        let geom = dispatch_geometry(&config);
        // Shader tile size is BN=32 (single-simdgroup), so 8192/32 = 256 tiles.
        assert_eq!(geom.num_tiles_intermediate, 8192 / 32);
        assert_eq!(geom.num_tiles_batch, 1);
        assert_eq!(geom.threads_per_threadgroup, 32);
    }

    #[test]
    fn test_dispatch_geometry_non_aligned_batch() {
        let config = MppFusedSwiGLUConfig::new(65, 2048, 128);
        let geom = dispatch_geometry(&config);
        // ceil(65/32) = 3, ceil(128/32) = 4
        assert_eq!(geom.num_tiles_batch, 3);
        assert_eq!(geom.num_tiles_intermediate, 4);
    }

    #[test]
    fn test_kernel_name_selects_dtype() {
        let mut config = MppFusedSwiGLUConfig::new(1, 2048, 8192);
        assert_eq!(kernel_name(&config), "mpp_fused_swiglu_forward_f16");

        config.use_fp16 = false;
        assert_eq!(kernel_name(&config), "mpp_fused_swiglu_forward_f32");
    }
}
