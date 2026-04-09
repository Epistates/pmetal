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
    /// LoRA rank (0 = no LoRA).
    pub lora_rank: usize,
    /// LoRA scaling factor (alpha / rank).
    pub lora_scale: f32,
}

impl MppFusedSwiGLUConfig {
    /// Create a new config for `output[batch, intermediate] = silu(x @ gate_W^T) * (x @ up_W^T)`.
    pub fn new(batch_size: usize, hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            batch_size,
            hidden_size,
            intermediate_size,
            use_fp16: true,
            lora_rank: 0,
            lora_scale: 0.0,
        }
    }

    /// Builder: enable LoRA on this config.
    pub fn with_lora(mut self, rank: usize, scale: f32) -> Self {
        self.lora_rank = rank;
        self.lora_scale = scale;
        self
    }

    /// Whether this config includes LoRA adapters.
    pub fn has_lora(&self) -> bool {
        self.lora_rank > 0
    }

    /// Output buffer element count.
    pub fn output_size(&self) -> usize {
        self.batch_size * self.intermediate_size
    }

    /// Threadgroup scratch size in bytes for the LoRA intermediate buffers
    /// (`x_gate_a` + `x_up_a`, each `batch_tile * lora_rank` floats).
    ///
    /// The Metal LoRA kernels use a fixed batch tile of BM=32.
    pub fn lora_scratch_bytes(&self) -> usize {
        const BM: usize = 32;
        2 * BM * self.lora_rank * std::mem::size_of::<f32>()
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
    match (config.has_lora(), config.use_fp16) {
        (false, true)  => "mpp_fused_swiglu_forward_f16",
        (false, false) => "mpp_fused_swiglu_forward_f32",
        (true,  true)  => "mpp_fused_swiglu_lora_forward_f16",
        (true,  false) => "mpp_fused_swiglu_lora_forward_f32",
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

    /// Execute the base (no-LoRA) kernel asynchronously.
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
        let kname = kernel_name(&self.config);

        let params = FusedSwiGLUParams {
            batch_size: self.config.batch_size as u32,
            hidden_size: self.config.hidden_size as u32,
            intermediate_size: self.config.intermediate_size as u32,
            lora_rank: self.config.lora_rank as u32,
            lora_scale: self.config.lora_scale,
        };

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

        let input_buf  = input.as_metal_buffer();
        let gate_buf   = gate_weight.as_metal_buffer();
        let up_buf     = up_weight.as_metal_buffer();
        let output_buf = output.as_metal_buffer();

        encode_mpp_kernel(&self.ctx, kname, grid, tg_size, |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(input_buf),  0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate_buf),   0, 1);
            encoder.setBuffer_offset_atIndex(Some(up_buf),     0, 2);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 3);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 4);
        })
    }

    /// Execute the LoRA kernel synchronously.
    ///
    /// `gate_lora_a`: `[lora_rank, hidden]`, `gate_lora_b`: `[intermediate, lora_rank]`
    /// (and similarly for the up projection). The config must have `lora_rank > 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_lora(
        &self,
        input: &dyn AsMetalBuffer,
        gate_weight: &dyn AsMetalBuffer,
        up_weight: &dyn AsMetalBuffer,
        gate_lora_a: &dyn AsMetalBuffer,
        gate_lora_b: &dyn AsMetalBuffer,
        up_lora_a: &dyn AsMetalBuffer,
        up_lora_b: &dyn AsMetalBuffer,
        output: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cmd = self.execute_lora_async(
            input,
            gate_weight,
            up_weight,
            gate_lora_a,
            gate_lora_b,
            up_lora_a,
            up_lora_b,
            output,
        )?;
        cmd.waitUntilCompleted();
        if let Some(err) = cmd.error() {
            return Err(MetalError::ExecutionFailed(err.to_string()));
        }
        Ok(())
    }

    /// Execute the LoRA kernel asynchronously.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_lora_async(
        &self,
        input: &dyn AsMetalBuffer,
        gate_weight: &dyn AsMetalBuffer,
        up_weight: &dyn AsMetalBuffer,
        gate_lora_a: &dyn AsMetalBuffer,
        gate_lora_b: &dyn AsMetalBuffer,
        up_lora_a: &dyn AsMetalBuffer,
        up_lora_b: &dyn AsMetalBuffer,
        output: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused SwiGLU not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }
        if self.config.lora_rank == 0 {
            return Err(MetalError::InvalidConfig(
                "execute_lora called but lora_rank is 0; use execute() for the base path".to_string(),
            ));
        }

        let geometry = dispatch_geometry(&self.config);
        let kname = kernel_name(&self.config);

        let params = FusedSwiGLUParams {
            batch_size: self.config.batch_size as u32,
            hidden_size: self.config.hidden_size as u32,
            intermediate_size: self.config.intermediate_size as u32,
            lora_rank: self.config.lora_rank as u32,
            lora_scale: self.config.lora_scale,
        };

        // Threadgroup scratch: 2 * BM * lora_rank floats (BM=32).
        let scratch_bytes = self.config.lora_scratch_bytes();

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

        let input_buf  = input.as_metal_buffer();
        let gate_buf   = gate_weight.as_metal_buffer();
        let up_buf     = up_weight.as_metal_buffer();
        let gla_buf    = gate_lora_a.as_metal_buffer();
        let glb_buf    = gate_lora_b.as_metal_buffer();
        let ula_buf    = up_lora_a.as_metal_buffer();
        let ulb_buf    = up_lora_b.as_metal_buffer();
        let output_buf = output.as_metal_buffer();

        encode_mpp_kernel(&self.ctx, kname, grid, tg_size, |encoder| unsafe {
            // buffer(0..7): inputs; buffer(8): params; threadgroup(0): scratch
            encoder.setBuffer_offset_atIndex(Some(input_buf),  0, 0);
            encoder.setBuffer_offset_atIndex(Some(gate_buf),   0, 1);
            encoder.setBuffer_offset_atIndex(Some(up_buf),     0, 2);
            encoder.setBuffer_offset_atIndex(Some(gla_buf),    0, 3);
            encoder.setBuffer_offset_atIndex(Some(glb_buf),    0, 4);
            encoder.setBuffer_offset_atIndex(Some(ula_buf),    0, 5);
            encoder.setBuffer_offset_atIndex(Some(ulb_buf),    0, 6);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 7);
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 8);
            encoder.setThreadgroupMemoryLength_atIndex(scratch_bytes, 0);
        })
    }
}

// =============================================================================
// MPP Fused MLP (gate + up + down combined)
// =============================================================================

/// Configuration for MPP Fused MLP (full SwiGLU MLP with down projection).
#[derive(Debug, Clone)]
pub struct MppFusedMLPConfig {
    /// Batch size (token count).
    pub batch_size: usize,
    /// Input/output hidden dimension.
    pub hidden_size: usize,
    /// Intermediate dimension (gate/up output).
    pub intermediate_size: usize,
}

impl MppFusedMLPConfig {
    /// Create a new config for the full MLP.
    pub fn new(batch_size: usize, hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            batch_size,
            hidden_size,
            intermediate_size,
        }
    }

    /// Output element count.
    pub fn output_size(&self) -> usize {
        self.batch_size * self.hidden_size
    }
}

/// Metal-side parameter block (must match `FusedMLPParams` in Metal).
#[repr(C)]
struct FusedMLPParams {
    batch_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
}

/// MPP Fused MLP dispatcher.
///
/// Dispatches `mpp_fused_mlp_forward_f16` — combines gate, up (SwiGLU) and
/// down projections in a single kernel with register-level fusion on M5+ NAX.
pub struct MppFusedMLP {
    ctx: Arc<MetalContext>,
    config: MppFusedMLPConfig,
}

impl MppFusedMLP {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppFusedMLPConfig) -> Self {
        Self { ctx, config }
    }

    /// Check if MPP Fused MLP is available (requires M5+ with NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute synchronously.
    ///
    /// `input`: `[batch, hidden]`, `gate_weight`: `[intermediate, hidden]`,
    /// `up_weight`: `[intermediate, hidden]`, `down_weight`: `[hidden, intermediate]`,
    /// `output`: `[batch, hidden]`.
    pub fn execute(
        &self,
        input: &dyn AsMetalBuffer,
        gate_weight: &dyn AsMetalBuffer,
        up_weight: &dyn AsMetalBuffer,
        down_weight: &dyn AsMetalBuffer,
        output: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.execute_async(input, gate_weight, up_weight, down_weight, output)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Execute asynchronously and return the submitted command buffer.
    pub fn execute_async(
        &self,
        input: &dyn AsMetalBuffer,
        gate_weight: &dyn AsMetalBuffer,
        up_weight: &dyn AsMetalBuffer,
        down_weight: &dyn AsMetalBuffer,
        output: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused MLP not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let params = FusedMLPParams {
            batch_size: self.config.batch_size as u32,
            hidden_size: self.config.hidden_size as u32,
            intermediate_size: self.config.intermediate_size as u32,
        };

        // Grid: [ceil(H/32), ceil(B/32), 1]  — output covers [B, H] space.
        const BM: usize = 32;
        const BN: usize = 32;
        let grid = objc2_metal::MTLSize {
            width: self.config.hidden_size.div_ceil(BN),
            height: self.config.batch_size.div_ceil(BM),
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: 32, // single SIMD group
            height: 1,
            depth: 1,
        };

        let input_buf = input.as_metal_buffer();
        let gate_buf = gate_weight.as_metal_buffer();
        let up_buf = up_weight.as_metal_buffer();
        let down_buf = down_weight.as_metal_buffer();
        let output_buf = output.as_metal_buffer();

        encode_mpp_kernel(
            &self.ctx,
            "mpp_fused_mlp_forward_f16",
            grid,
            tg_size,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(gate_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(up_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(down_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 4);
                let p_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 5);
            },
        )
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

    #[test]
    fn test_kernel_name_selects_lora_variant() {
        let config = MppFusedSwiGLUConfig::new(1, 2048, 8192).with_lora(16, 0.5);
        assert!(config.has_lora());
        assert_eq!(kernel_name(&config), "mpp_fused_swiglu_lora_forward_f16");

        let mut config_f32 = config.clone();
        config_f32.use_fp16 = false;
        assert_eq!(kernel_name(&config_f32), "mpp_fused_swiglu_lora_forward_f32");
    }

    #[test]
    fn test_lora_scratch_bytes() {
        // BM=32, lora_rank=16, 2 buffers, f32 = 4 bytes
        let config = MppFusedSwiGLUConfig::new(4, 2048, 8192).with_lora(16, 1.0);
        assert_eq!(config.lora_scratch_bytes(), 2 * 32 * 16 * 4);

        let config_r32 = MppFusedSwiGLUConfig::new(4, 2048, 8192).with_lora(32, 1.0);
        assert_eq!(config_r32.lora_scratch_bytes(), 2 * 32 * 32 * 4);
    }

    // MppFusedMLP config tests
    #[test]
    fn test_mlp_config_output_size() {
        let config = MppFusedMLPConfig::new(4, 2048, 8192);
        assert_eq!(config.output_size(), 4 * 2048);
    }

    #[test]
    fn test_mlp_dispatch_geometry() {
        let config = MppFusedMLPConfig::new(1, 2048, 8192);
        // Grid: [ceil(H/32), ceil(B/32), 1]
        let tiles_h = config.hidden_size.div_ceil(32);
        let tiles_b = config.batch_size.div_ceil(32);
        assert_eq!(tiles_h, 2048 / 32); // 64
        assert_eq!(tiles_b, 1);
    }

    #[test]
    fn test_mlp_dispatch_geometry_non_aligned() {
        let config = MppFusedMLPConfig::new(65, 2048, 4096);
        let tiles_b = config.batch_size.div_ceil(32);
        let tiles_h = config.hidden_size.div_ceil(32);
        assert_eq!(tiles_b, 3);  // ceil(65/32)
        assert_eq!(tiles_h, 64); // 2048/32
    }
}
