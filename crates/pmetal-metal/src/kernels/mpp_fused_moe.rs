#![allow(unsafe_code)]

//! Metal 4 / MPP Fused MoE expert forward dispatch.
//!
//! Provides hardware-accelerated MoE expert forward passes via Metal Performance
//! Primitives on M5+ (Apple10) GPUs with NAX cores.
//!
//! Replaces the Metal 3 quantized-expert kernels for dense (fp16) MoE models.
//! Uses `matmul2d` with single simdgroup (NAX) for the GEMM steps, and MPP
//! postfix fusion to keep SwiGLU in register space between gate/up and down
//! projections.
//!
//! Kernel families:
//! - `mpp_fused_moe_gate_up_f16` — gate + up GEMMs + SwiGLU postfix
//! - `mpp_fused_moe_down_f16` — down projection GEMM
//! - `mpp_moe_weighted_scatter_f16` — weighted residual accumulation (combine)
//!
//! Grid for gate/up: `[ceil(intermediate/32), ceil(tokens/32), 1]`
//! Grid for down:    `[ceil(hidden/32), ceil(tokens/32), 1]`

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

// =============================================================================
// Config
// =============================================================================

/// Configuration for one MPP MoE expert forward pass.
#[derive(Debug, Clone)]
pub struct MppFusedMoEConfig {
    /// Number of tokens dispatched to this expert.
    pub batch_size: usize,
    /// Hidden dimension (input and output of the expert).
    pub hidden_dim: usize,
    /// Expert intermediate dimension (gate/up output).
    pub intermediate_dim: usize,
}

impl MppFusedMoEConfig {
    /// Create a new expert config.
    pub fn new(batch_size: usize, hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            batch_size,
            hidden_dim,
            intermediate_dim,
        }
    }
}

/// Configuration for the weighted scatter (combine) kernel.
#[derive(Debug, Clone)]
pub struct MppMoEScatterConfig {
    /// Number of tokens being accumulated.
    pub num_tokens: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
}

// =============================================================================
// Metal-side parameter blocks
// =============================================================================

/// Must match `MppMoEParams` in mpp_fused_moe.metal.
#[repr(C)]
struct MppMoEParamsMetal {
    batch_size: u32,
    hidden_dim: u32,
    intermediate_dim: u32,
}

// `MppMoEParams` in `metal4/mpp_fused_moe.metal`.
crate::msl_layout::msl_layout!(MppMoEParamsMetal, size = 12, {
    batch_size: 0,
    hidden_dim: 4,
    intermediate_dim: 8,
});

/// Must match `MppMoEScatterParams` in mpp_fused_moe.metal.
#[repr(C)]
struct MppMoEScatterParamsMetal {
    num_tokens: u32,
    hidden_dim: u32,
}

// `MppMoEScatterParams` in `metal4/mpp_fused_moe.metal`.
crate::msl_layout::msl_layout!(MppMoEScatterParamsMetal, size = 8, {
    num_tokens: 0,
    hidden_dim: 4,
});

// =============================================================================
// MoE Expert Forward Dispatcher
// =============================================================================

/// MPP Fused MoE expert forward dispatcher.
///
/// Dispatches a full expert forward pass (gate+up SwiGLU + down projection)
/// using MPP matmul2d on M5+ hardware.
pub struct MppFusedMoE {
    ctx: Arc<MetalContext>,
    config: MppFusedMoEConfig,
}

impl MppFusedMoE {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppFusedMoEConfig) -> Self {
        Self { ctx, config }
    }

    /// Returns true when MPP MoE is available (requires M5+ NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute the gate + up SwiGLU projection synchronously.
    ///
    /// - `input`: `[batch, hidden]`
    /// - `gate_weight`: `[intermediate, hidden]`
    /// - `up_weight`: `[intermediate, hidden]`
    /// - `act_out`: `[batch, intermediate]` — SwiGLU output
    pub fn execute_gate_up(
        &self,
        input: &dyn AsMetalBuffer,
        gate_weight: &dyn AsMetalBuffer,
        up_weight: &dyn AsMetalBuffer,
        act_out: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.execute_gate_up_async(input, gate_weight, up_weight, act_out)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Execute gate + up SwiGLU asynchronously.
    pub fn execute_gate_up_async(
        &self,
        input: &dyn AsMetalBuffer,
        gate_weight: &dyn AsMetalBuffer,
        up_weight: &dyn AsMetalBuffer,
        act_out: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused MoE not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let params = MppMoEParamsMetal {
            batch_size: self.config.batch_size as u32,
            hidden_dim: self.config.hidden_dim as u32,
            intermediate_dim: self.config.intermediate_dim as u32,
        };

        // Grid: [ceil(intermediate/32), ceil(batch/32), 1]
        let grid = objc2_metal::MTLSize {
            width: self.config.intermediate_dim.div_ceil(32),
            height: self.config.batch_size.div_ceil(32),
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        let input_buf = input.as_metal_buffer();
        let gate_buf = gate_weight.as_metal_buffer();
        let up_buf = up_weight.as_metal_buffer();
        let act_buf = act_out.as_metal_buffer();

        encode_mpp_kernel(
            &self.ctx,
            "mpp_fused_moe_gate_up_f16",
            grid,
            tg_size,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(gate_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(up_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(act_buf), 0, 3);
                let p_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 4);
            },
        )
    }

    /// Execute the down projection synchronously.
    ///
    /// - `act_in`: `[batch, intermediate]` — SwiGLU output from gate_up
    /// - `down_weight`: `[hidden, intermediate]`
    /// - `out`: `[batch, hidden]`
    pub fn execute_down(
        &self,
        act_in: &dyn AsMetalBuffer,
        down_weight: &dyn AsMetalBuffer,
        out: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.execute_down_async(act_in, down_weight, out)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Execute down projection asynchronously.
    pub fn execute_down_async(
        &self,
        act_in: &dyn AsMetalBuffer,
        down_weight: &dyn AsMetalBuffer,
        out: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused MoE not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let params = MppMoEParamsMetal {
            batch_size: self.config.batch_size as u32,
            hidden_dim: self.config.hidden_dim as u32,
            intermediate_dim: self.config.intermediate_dim as u32,
        };

        // Grid: [ceil(hidden/32), ceil(batch/32), 1]
        let grid = objc2_metal::MTLSize {
            width: self.config.hidden_dim.div_ceil(32),
            height: self.config.batch_size.div_ceil(32),
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        let act_buf = act_in.as_metal_buffer();
        let down_buf = down_weight.as_metal_buffer();
        let out_buf = out.as_metal_buffer();

        encode_mpp_kernel(
            &self.ctx,
            "mpp_fused_moe_down_f16",
            grid,
            tg_size,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(act_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(down_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                let p_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 3);
            },
        )
    }
}

// =============================================================================
// Weighted Scatter Dispatcher (MoE combine)
// =============================================================================

/// MPP MoE weighted scatter dispatcher.
///
/// Accumulates expert contributions into an output buffer with router weighting:
/// `out[token] += weight[token] * expert_out[token]`.
pub struct MppMoEScatter {
    ctx: Arc<MetalContext>,
    config: MppMoEScatterConfig,
}

impl MppMoEScatter {
    /// Create a new scatter dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppMoEScatterConfig) -> Self {
        Self { ctx, config }
    }

    /// Returns true when MPP scatter is available.
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute weighted scatter synchronously.
    ///
    /// `expert_out`: `[num_tokens, hidden]`, `weights`: `[num_tokens]` (fp32),
    /// `accum`: `[num_tokens, hidden]` read-modify-write.
    pub fn execute(
        &self,
        expert_out: &dyn AsMetalBuffer,
        weights: &dyn AsMetalBuffer,
        accum: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.execute_async(expert_out, weights, accum)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Execute asynchronously.
    pub fn execute_async(
        &self,
        expert_out: &dyn AsMetalBuffer,
        weights: &dyn AsMetalBuffer,
        accum: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP MoE scatter not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let params = MppMoEScatterParamsMetal {
            num_tokens: self.config.num_tokens as u32,
            hidden_dim: self.config.hidden_dim as u32,
        };

        // Grid: [num_tokens, 1, 1]  Threadgroup: [32, 1, 1]
        let grid = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        let expert_buf = expert_out.as_metal_buffer();
        let weights_buf = weights.as_metal_buffer();
        let accum_buf = accum.as_metal_buffer();

        encode_mpp_kernel(
            &self.ctx,
            "mpp_moe_weighted_scatter_f16",
            grid,
            tg_size,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(expert_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(weights_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(accum_buf), 0, 2);
                let p_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 3);
            },
        )
    }
}

// =============================================================================
// Quantized MoE Expert Dispatchers
// =============================================================================

/// Configuration for quantized MPP MoE expert forward pass.
#[derive(Debug, Clone)]
pub struct MppFusedMoEQuantConfig {
    /// Number of tokens dispatched to this expert.
    pub batch_size: usize,
    /// Hidden dimension (input and output of the expert).
    pub hidden_dim: usize,
    /// Expert intermediate dimension.
    pub intermediate_dim: usize,
    /// Quantization group size (e.g., 64).
    pub group_size: usize,
    /// Quantization bits (must be 4).
    pub bits: u32,
}

impl MppFusedMoEQuantConfig {
    /// Create a new quantized expert config.
    pub fn new(
        batch_size: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        group_size: usize,
        bits: u32,
    ) -> Self {
        Self {
            batch_size,
            hidden_dim,
            intermediate_dim,
            group_size,
            bits,
        }
    }
}

/// Metal-side parameter block (must match `MppMoEQuantParams` in Metal).
#[repr(C)]
struct MppMoEQuantParamsMetal {
    batch_size: u32,
    hidden_dim: u32,
    intermediate_dim: u32,
    group_size: u32,
    bits: u32,
}

// `MppMoEQuantParams` in `metal4/mpp_fused_moe.metal`.
crate::msl_layout::msl_layout!(MppMoEQuantParamsMetal, size = 20, {
    batch_size: 0,
    hidden_dim: 4,
    intermediate_dim: 8,
    group_size: 12,
    bits: 16,
});

/// MPP Fused MoE expert forward dispatcher for quantized (4-bit) weights.
///
/// Accepts packed `u32` gate/up/down weights and fp16 scales/biases as carried
/// by `MoeExpertDescriptor`. Dequantizes cooperatively on-GPU, then uses MPP
/// matmul2d for the GEMM steps.
pub struct MppFusedMoEQuant {
    ctx: Arc<MetalContext>,
    config: MppFusedMoEQuantConfig,
}

impl MppFusedMoEQuant {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppFusedMoEQuantConfig) -> Self {
        Self { ctx, config }
    }

    /// Returns true when quantized MPP MoE is available (requires M5+ NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax()
            && self.ctx.pipeline_cache().metal4_library().is_some()
            && self.config.bits == 4
    }

    fn make_params(&self) -> MppMoEQuantParamsMetal {
        MppMoEQuantParamsMetal {
            batch_size: self.config.batch_size as u32,
            hidden_dim: self.config.hidden_dim as u32,
            intermediate_dim: self.config.intermediate_dim as u32,
            group_size: self.config.group_size as u32,
            bits: self.config.bits,
        }
    }

    /// Execute gate + up SwiGLU (quantized) synchronously.
    ///
    /// - `input`: `[batch, hidden]` fp16
    /// - `gate_w_q`: `[intermediate, hidden/8]` packed 4-bit
    /// - `gate_scales`, `gate_biases`: `[intermediate, hidden/group_size]` fp16
    /// - `up_w_q`, `up_scales`, `up_biases`: same layout
    /// - `act_out`: `[batch, intermediate]` fp16 — SwiGLU output
    #[allow(clippy::too_many_arguments)]
    pub fn execute_gate_up(
        &self,
        input: &dyn AsMetalBuffer,
        gate_w_q: &dyn AsMetalBuffer,
        gate_scales: &dyn AsMetalBuffer,
        gate_biases: &dyn AsMetalBuffer,
        up_w_q: &dyn AsMetalBuffer,
        up_scales: &dyn AsMetalBuffer,
        up_biases: &dyn AsMetalBuffer,
        act_out: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.execute_gate_up_async(
            input,
            gate_w_q,
            gate_scales,
            gate_biases,
            up_w_q,
            up_scales,
            up_biases,
            act_out,
        )?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Execute gate + up SwiGLU (quantized) asynchronously.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_gate_up_async(
        &self,
        input: &dyn AsMetalBuffer,
        gate_w_q: &dyn AsMetalBuffer,
        gate_scales: &dyn AsMetalBuffer,
        gate_biases: &dyn AsMetalBuffer,
        up_w_q: &dyn AsMetalBuffer,
        up_scales: &dyn AsMetalBuffer,
        up_biases: &dyn AsMetalBuffer,
        act_out: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP quantized MoE gate_up not available (requires M5+ NAX, bits=4)".to_string(),
            ));
        }

        let params = self.make_params();

        // Grid: [ceil(I/32), ceil(B/32), 1]  Threadgroup: [128, 1, 1]
        let grid = objc2_metal::MTLSize {
            width: self.config.intermediate_dim.div_ceil(32),
            height: self.config.batch_size.div_ceil(32),
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: 128, // 4 simdgroups for cooperative dequant
            height: 1,
            depth: 1,
        };

        let input_buf = input.as_metal_buffer();
        let gw_buf = gate_w_q.as_metal_buffer();
        let gs_buf = gate_scales.as_metal_buffer();
        let gb_buf = gate_biases.as_metal_buffer();
        let uw_buf = up_w_q.as_metal_buffer();
        let us_buf = up_scales.as_metal_buffer();
        let ub_buf = up_biases.as_metal_buffer();
        let act_buf = act_out.as_metal_buffer();

        encode_mpp_kernel(
            &self.ctx,
            "mpp_fused_moe_gate_up_quant_f16",
            grid,
            tg_size,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(gw_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(gs_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(gb_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(uw_buf), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(us_buf), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(ub_buf), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(act_buf), 0, 7);
                let p_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 8);
            },
        )
    }

    /// Execute down projection (quantized) synchronously.
    ///
    /// - `act_in`: `[batch, intermediate]` fp16 — SwiGLU output
    /// - `down_w_q`: `[hidden, intermediate/8]` packed 4-bit
    /// - `down_scales`, `down_biases`: `[hidden, intermediate/group_size]` fp16
    /// - `out`: `[batch, hidden]` fp16
    pub fn execute_down(
        &self,
        act_in: &dyn AsMetalBuffer,
        down_w_q: &dyn AsMetalBuffer,
        down_scales: &dyn AsMetalBuffer,
        down_biases: &dyn AsMetalBuffer,
        out: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.execute_down_async(act_in, down_w_q, down_scales, down_biases, out)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Execute down projection (quantized) asynchronously.
    pub fn execute_down_async(
        &self,
        act_in: &dyn AsMetalBuffer,
        down_w_q: &dyn AsMetalBuffer,
        down_scales: &dyn AsMetalBuffer,
        down_biases: &dyn AsMetalBuffer,
        out: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP quantized MoE down not available (requires M5+ NAX, bits=4)".to_string(),
            ));
        }

        let params = self.make_params();

        // Grid: [ceil(H/32), ceil(B/32), 1]  Threadgroup: [128, 1, 1]
        let grid = objc2_metal::MTLSize {
            width: self.config.hidden_dim.div_ceil(32),
            height: self.config.batch_size.div_ceil(32),
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };

        let act_buf = act_in.as_metal_buffer();
        let dw_buf = down_w_q.as_metal_buffer();
        let ds_buf = down_scales.as_metal_buffer();
        let db_buf = down_biases.as_metal_buffer();
        let out_buf = out.as_metal_buffer();

        encode_mpp_kernel(
            &self.ctx,
            "mpp_fused_moe_down_quant_f16",
            grid,
            tg_size,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(act_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(dw_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(ds_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(db_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 4);
                let p_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 5);
            },
        )
    }
}

// =============================================================================
// GPU Tile Count for Grouped GEMM
// =============================================================================

/// Metal-side parameter block (must match `TileCountParams` in Metal).
#[repr(C)]
struct TileCountParams {
    num_experts: u32,
    intermediate: u32,
    block_m: u32,
    block_n: u32,
}

// `TileCountParams` in `metal4/mpp_fused_moe.metal`.
crate::msl_layout::msl_layout!(TileCountParams, size = 16, {
    num_experts: 0,
    intermediate: 4,
    block_m: 8,
    block_n: 12,
});

/// Computes the grouped GEMM tile count from `expert_offsets` entirely on GPU.
///
/// Eliminates the CPU round-trip in `Metal4Backend::grouped_gemm`: instead of
/// copying `expert_offsets` to host, reading each expert's token count, and
/// computing the tile sum, this dispatches a single tiny kernel that atomically
/// accumulates the tile count into a 1-element output buffer.
///
/// The result is a `MetalBuffer<u32>` containing `total_tiles`. The caller
/// reads this single u32 (4 bytes) to construct `GroupedGemmDispatch`.
pub struct MppGroupedGemmTileCount {
    ctx: Arc<MetalContext>,
}

impl MppGroupedGemmTileCount {
    /// Create a new tile count dispatcher.
    pub fn new(ctx: Arc<MetalContext>) -> Self {
        Self { ctx }
    }

    /// Returns true when the tile count kernel is available (requires M5+ NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Compute the total grouped GEMM tile count on GPU and return it as a u32.
    ///
    /// `expert_offsets`: GPU buffer `[num_experts + 1]` u32 (prefix sum),
    /// `num_experts`: number of experts E,
    /// `intermediate`: output dimension N (used to compute num_n_tiles),
    /// `block_m`, `block_n`: tile dimensions (64 for grouped GEMM).
    ///
    /// Returns the `total_tiles` value needed for `GroupedGemmDispatch`.
    pub fn compute(
        &self,
        expert_offsets: &dyn AsMetalBuffer,
        num_experts: usize,
        intermediate: usize,
        block_m: usize,
        block_n: usize,
    ) -> Result<usize> {
        use crate::buffer::BufferUsage;

        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP tile count kernel not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        // Clamp num_experts to Metal's max threadgroup size.
        // A single threadgroup handles all experts — E <= 1024 is safe on M5.
        let e = num_experts.min(1024);
        if e == 0 {
            return Ok(0);
        }

        let params = TileCountParams {
            num_experts: e as u32,
            intermediate: intermediate as u32,
            block_m: block_m as u32,
            block_n: block_n as u32,
        };

        // 1-element output buffer (atomic u32, zero-initialized via Shared).
        let tile_count_buf =
            crate::buffer::MetalBuffer::<u32>::new(&self.ctx, 1, BufferUsage::Shared)?;

        let eo_buf = expert_offsets.as_metal_buffer();
        let tc_buf = tile_count_buf.as_metal_buffer();

        // Grid: [1, 1, 1]  Threadgroup: [E, 1, 1]
        let grid = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize {
            width: e,
            height: 1,
            depth: 1,
        };

        let cb = encode_mpp_kernel(
            &self.ctx,
            "mpp_grouped_gemm_tile_count",
            grid,
            tg_size,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(eo_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(tc_buf), 0, 1);
                let p_ptr = NonNull::from(&params).cast();
                encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 2);
            },
        )?;

        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }

        // Read the single u32 result — 4 bytes, trivial CPU cost.
        let contents = tile_count_buf.as_slice();
        let total_tiles = contents[0] as usize;
        Ok(total_tiles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let cfg = MppFusedMoEConfig::new(4, 2048, 4096);
        assert_eq!(cfg.batch_size, 4);
        assert_eq!(cfg.hidden_dim, 2048);
        assert_eq!(cfg.intermediate_dim, 4096);
    }

    #[test]
    fn test_gate_up_tiles() {
        let cfg = MppFusedMoEConfig::new(1, 2048, 4096);
        let tiles_i = cfg.intermediate_dim.div_ceil(32);
        let tiles_b = cfg.batch_size.div_ceil(32);
        assert_eq!(tiles_i, 128);
        assert_eq!(tiles_b, 1);
    }

    #[test]
    fn test_down_tiles() {
        let cfg = MppFusedMoEConfig::new(32, 2048, 4096);
        let tiles_h = cfg.hidden_dim.div_ceil(32);
        let tiles_b = cfg.batch_size.div_ceil(32);
        assert_eq!(tiles_h, 64);
        assert_eq!(tiles_b, 1);
    }

    #[test]
    fn test_scatter_config() {
        let cfg = MppMoEScatterConfig {
            num_tokens: 8,
            hidden_dim: 2048,
        };
        assert_eq!(cfg.num_tokens, 8);
    }

    // MppFusedMoEQuant config tests
    #[test]
    fn test_quant_config() {
        let cfg = MppFusedMoEQuantConfig::new(4, 2048, 4096, 64, 4);
        assert_eq!(cfg.batch_size, 4);
        assert_eq!(cfg.hidden_dim, 2048);
        assert_eq!(cfg.intermediate_dim, 4096);
        assert_eq!(cfg.group_size, 64);
        assert_eq!(cfg.bits, 4);
    }

    #[test]
    fn test_quant_gate_up_tiles() {
        let cfg = MppFusedMoEQuantConfig::new(1, 2048, 4096, 64, 4);
        // Grid: [ceil(I/32), ceil(B/32), 1]
        let tiles_i = cfg.intermediate_dim.div_ceil(32);
        let tiles_b = cfg.batch_size.div_ceil(32);
        assert_eq!(tiles_i, 128);
        assert_eq!(tiles_b, 1);
    }

    #[test]
    fn test_quant_down_tiles() {
        let cfg = MppFusedMoEQuantConfig::new(32, 2048, 4096, 64, 4);
        // Grid: [ceil(H/32), ceil(B/32), 1]
        let tiles_h = cfg.hidden_dim.div_ceil(32);
        let tiles_b = cfg.batch_size.div_ceil(32);
        assert_eq!(tiles_h, 64);
        assert_eq!(tiles_b, 1);
    }
}
