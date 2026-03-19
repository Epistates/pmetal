#![allow(unsafe_code)]

//! Fused MoE (Mixture of Experts) Metal kernels.
//!
//! Provides GPU-accelerated fused operations for MoE inference:
//!
//! - [`FusedMoeExpert`]: Fused gate+up+SwiGLU + down projection for quantized experts
//! - [`GatherQmmSwiglu`]: Fused gather + quantized matmul + SwiGLU for resident mode
//!
//! All dequant kernels use the qdot technique (pre-scaled activations) from MLX's
//! `quantized.h`, eliminating per-nibble shifts from the inner loop for 30-40%
//! compute reduction. Thread-private registers replace threadgroup shared memory
//! to avoid the x_shared[4096] overflow bug and reduce barrier overhead.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::{
    buffer::MetalBuffer,
    context::MetalContext,
    error::{MetalError, Result},
    pipeline::FunctionConstant,
};

// ============================================================================
// Fused Quantized Expert Forward
// ============================================================================

/// Quantization bit width for expert weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertBits {
    /// 4-bit affine quantization (8 values per uint32)
    Four,
    /// 2-bit affine quantization (16 values per uint32)
    Two,
}

impl ExpertBits {
    /// Number of values packed per uint32.
    pub fn pack_factor(self) -> u32 {
        match self {
            Self::Four => 8,
            Self::Two => 16,
        }
    }

    /// Metal kernel suffix for the appropriate bit width.
    fn matvec_kernel_name(self) -> &'static str {
        match self {
            Self::Four => "dequant_matvec_4bit",
            Self::Two => "dequant_matvec_2bit",
        }
    }
}

/// Configuration for a single quantized expert.
#[derive(Debug, Clone)]
pub struct FusedMoeExpertConfig {
    /// Hidden dimension (input to gate/up, output of down).
    pub hidden_dim: u32,
    /// Intermediate dimension (output of gate/up, input to down).
    pub intermediate_dim: u32,
    /// Quantization group size (typically 64).
    pub group_size: u32,
    /// Quantization bit width (2 or 4).
    pub bits: ExpertBits,
}

impl FusedMoeExpertConfig {
    fn validate(&self) -> Result<()> {
        if self.hidden_dim == 0 {
            return Err(MetalError::InvalidConfig(
                "hidden_dim must be positive".into(),
            ));
        }
        if self.intermediate_dim == 0 {
            return Err(MetalError::InvalidConfig(
                "intermediate_dim must be positive".into(),
            ));
        }
        let pf = self.bits.pack_factor();
        if self.hidden_dim % pf != 0 {
            return Err(MetalError::InvalidConfig(format!(
                "hidden_dim ({}) must be divisible by pack_factor ({})",
                self.hidden_dim, pf
            )));
        }
        // Down projection uses intermediate_dim as input — must also be aligned
        if self.intermediate_dim % pf != 0 {
            return Err(MetalError::InvalidConfig(format!(
                "intermediate_dim ({}) must be divisible by pack_factor ({})",
                self.intermediate_dim, pf
            )));
        }
        if self.group_size == 0 || self.hidden_dim % self.group_size != 0 {
            return Err(MetalError::InvalidConfig(format!(
                "hidden_dim ({}) must be divisible by group_size ({})",
                self.hidden_dim, self.group_size
            )));
        }
        if self.intermediate_dim % self.group_size != 0 {
            return Err(MetalError::InvalidConfig(format!(
                "intermediate_dim ({}) must be divisible by group_size ({})",
                self.intermediate_dim, self.group_size
            )));
        }
        // group_size must be >= pack_factor to prevent divide-by-zero
        // in group index calculation (col / (group_size / pack_factor))
        if self.group_size < pf {
            return Err(MetalError::InvalidConfig(format!(
                "group_size ({}) must be >= pack_factor ({}) for {:?}-bit quantization",
                self.group_size, pf, self.bits
            )));
        }
        Ok(())
    }
}

/// Fused quantized expert forward kernel.
///
/// For offloaded/single-expert inference (T=1 decode):
/// 1. Phase A: `fused_gate_up_swiglu` — dequant gate+up weights, compute SwiGLU
/// 2. Phase B: `dequant_matvec_*bit` — dequant down weights, compute projection
///
/// Both phases are dispatched in a single command buffer for minimal overhead.
pub struct FusedMoeExpert {
    ctx: Arc<MetalContext>,
    config: FusedMoeExpertConfig,
}

/// Packed expert weight buffers for a single expert.
///
/// All weights are in quantized form (uint32 packed, bf16 scales/biases).
pub struct ExpertWeightBuffers {
    /// Gate projection packed weights `[intermediate, hidden/pack_factor]`
    pub gate_weights: MetalBuffer<u32>,
    /// Gate projection scales `[intermediate, hidden/group_size]`
    pub gate_scales: MetalBuffer<u16>,
    /// Gate projection biases `[intermediate, hidden/group_size]`
    pub gate_biases: MetalBuffer<u16>,
    /// Up projection packed weights `[intermediate, hidden/pack_factor]`
    pub up_weights: MetalBuffer<u32>,
    /// Up projection scales `[intermediate, hidden/group_size]`
    pub up_scales: MetalBuffer<u16>,
    /// Up projection biases `[intermediate, hidden/group_size]`
    pub up_biases: MetalBuffer<u16>,
    /// Down projection packed weights `[hidden, intermediate/pack_factor]`
    pub down_weights: MetalBuffer<u32>,
    /// Down projection scales `[hidden, intermediate/group_size]`
    pub down_scales: MetalBuffer<u16>,
    /// Down projection biases `[hidden, intermediate/group_size]`
    pub down_biases: MetalBuffer<u16>,
}

impl FusedMoeExpert {
    /// Create a new fused expert forward kernel instance.
    pub fn new(ctx: Arc<MetalContext>, config: FusedMoeExpertConfig) -> Result<Self> {
        config.validate()?;

        // Verify both pipelines can be created
        {
            let mut cache = ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(ctx.device(), "fused_gate_up_swiglu", None)?;
            cache.get_or_create_pipeline(
                ctx.device(),
                config.bits.matvec_kernel_name(),
                None,
            )?;
        }

        Ok(Self { ctx, config })
    }

    /// Run a single expert's full forward pass (blocking).
    ///
    /// Convenience wrapper that creates a command buffer, encodes both phases,
    /// commits, and waits. Use `encode_into` for non-blocking dispatch.
    pub fn forward_single_expert(
        &self,
        input: &MetalBuffer<f32>,
        weights: &ExpertWeightBuffers,
        output: &MetalBuffer<f32>,
        intermediate: &MetalBuffer<f32>,
    ) -> Result<()> {
        let command_queue = self.ctx.command_queue();
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        self.encode_into(&command_buffer, input, weights, output, intermediate)?;

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }
        Ok(())
    }

    /// Encode both phases into an external command buffer (non-blocking).
    ///
    /// The caller manages command buffer lifecycle: create, accumulate multiple
    /// expert dispatches, commit once, defer wait. This is the pipelined path
    /// — encode all K experts into one CMD buffer, submit, defer sync to next layer.
    pub fn encode_into(
        &self,
        command_buffer: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        input: &MetalBuffer<f32>,
        weights: &ExpertWeightBuffers,
        output: &MetalBuffer<f32>,
        intermediate: &MetalBuffer<f32>,
    ) -> Result<()> {
        // Phase A: fused gate+up+SwiGLU
        self.encode_gate_up_swiglu(command_buffer, input, weights, intermediate)?;
        // Phase B: down projection
        self.encode_down_projection(command_buffer, intermediate, weights, output)?;
        Ok(())
    }

    /// Create a new command buffer from this context's queue.
    pub fn create_command_buffer(
        &self,
    ) -> Result<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>>> {
        let command_queue = self.ctx.command_queue();
        command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)
    }

    /// Submit a command buffer and wait for GPU completion.
    pub fn submit_and_wait(
        &self,
        command_buffer: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<()> {
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }
        Ok(())
    }

    /// Encode Phase A: fused gate+up+SwiGLU into a command buffer.
    fn encode_gate_up_swiglu(
        &self,
        command_buffer: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        input: &MetalBuffer<f32>,
        weights: &ExpertWeightBuffers,
        output: &MetalBuffer<f32>,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), "fused_gate_up_swiglu", None)?
        };

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.gate_weights.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights.gate_scales.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weights.gate_biases.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(weights.up_weights.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(weights.up_scales.metal_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(weights.up_biases.metal_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(input.metal_buffer()), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 7);

            let out_dim = self.config.intermediate_dim;
            let in_dim = self.config.hidden_dim;
            let group_size = self.config.group_size;

            let out_dim_ptr = NonNull::from(&out_dim).cast();
            encoder.setBytes_length_atIndex(out_dim_ptr, 4, 8);
            let in_dim_ptr = NonNull::from(&in_dim).cast();
            encoder.setBytes_length_atIndex(in_dim_ptr, 4, 9);
            let gs_ptr = NonNull::from(&group_size).cast();
            encoder.setBytes_length_atIndex(gs_ptr, 4, 10);
        }

        // ROWS_PER_TG = 8 (RESULTS_PER_SG=4 × NUM_SIMDGROUPS=2), TG_SIZE = 64
        let rows_per_tg: u32 = 8;
        let grid_size = objc2_metal::MTLSize {
            width: self.config.intermediate_dim.div_ceil(rows_per_tg) as usize,
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

        Ok(())
    }

    /// Encode Phase B: down projection into a command buffer.
    fn encode_down_projection(
        &self,
        command_buffer: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        input: &MetalBuffer<f32>,
        weights: &ExpertWeightBuffers,
        output: &MetalBuffer<f32>,
    ) -> Result<()> {
        let kernel_name = self.config.bits.matvec_kernel_name();
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(self.ctx.device(), kernel_name, None)?
        };

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.down_weights.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights.down_scales.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weights.down_biases.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(input.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 4);

            let out_dim = self.config.hidden_dim;
            let in_dim = self.config.intermediate_dim;
            let group_size = self.config.group_size;

            let out_dim_ptr = NonNull::from(&out_dim).cast();
            encoder.setBytes_length_atIndex(out_dim_ptr, 4, 5);
            let in_dim_ptr = NonNull::from(&in_dim).cast();
            encoder.setBytes_length_atIndex(in_dim_ptr, 4, 6);
            let gs_ptr = NonNull::from(&group_size).cast();
            encoder.setBytes_length_atIndex(gs_ptr, 4, 7);
        }

        let rows_per_tg: u32 = 8;
        let grid_size = objc2_metal::MTLSize {
            width: self.config.hidden_dim.div_ceil(rows_per_tg) as usize,
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

        Ok(())
    }

    /// Encode a single expert forward using an aligned buffer with byte offsets.
    ///
    /// Zero-copy path: the `AlignedBuffer` from pread is used directly as the
    /// Metal buffer, with component offsets from `ExpertRecord` passed via
    /// `setBuffer:offset:atIndex:`. No intermediate copies or allocations.
    ///
    /// Both Phase A (gate+up+SwiGLU) and Phase B (down projection) are encoded
    /// into the provided command buffer. The caller is responsible for submitting
    /// and synchronizing the command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_expert_aligned(
        &self,
        command_buffer: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        input: &MetalBuffer<f32>,
        expert_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        gate_w_off: usize,
        gate_s_off: usize,
        gate_b_off: usize,
        up_w_off: usize,
        up_s_off: usize,
        up_b_off: usize,
        down_w_off: usize,
        down_s_off: usize,
        down_b_off: usize,
        output: &MetalBuffer<f32>,
        intermediate: &MetalBuffer<f32>,
    ) -> Result<()> {
        // Phase A: gate+up+SwiGLU
        {
            let pipeline = {
                let mut cache = self.ctx.pipeline_cache_mut();
                cache.get_or_create_pipeline(self.ctx.device(), "fused_gate_up_swiglu", None)?
            };

            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or(MetalError::EncoderCreation)?;

            encoder.setComputePipelineState(&pipeline);

            unsafe {
                // All 6 weight components point to the same AlignedBuffer at different offsets
                encoder.setBuffer_offset_atIndex(Some(expert_buf), gate_w_off, 0);
                encoder.setBuffer_offset_atIndex(Some(expert_buf), gate_s_off, 1);
                encoder.setBuffer_offset_atIndex(Some(expert_buf), gate_b_off, 2);
                encoder.setBuffer_offset_atIndex(Some(expert_buf), up_w_off, 3);
                encoder.setBuffer_offset_atIndex(Some(expert_buf), up_s_off, 4);
                encoder.setBuffer_offset_atIndex(Some(expert_buf), up_b_off, 5);
                encoder.setBuffer_offset_atIndex(Some(input.metal_buffer()), 0, 6);
                encoder.setBuffer_offset_atIndex(Some(intermediate.metal_buffer()), 0, 7);

                let out_dim = self.config.intermediate_dim;
                let in_dim = self.config.hidden_dim;
                let group_size = self.config.group_size;
                let out_dim_ptr = NonNull::from(&out_dim).cast();
                encoder.setBytes_length_atIndex(out_dim_ptr, 4, 8);
                let in_dim_ptr = NonNull::from(&in_dim).cast();
                encoder.setBytes_length_atIndex(in_dim_ptr, 4, 9);
                let gs_ptr = NonNull::from(&group_size).cast();
                encoder.setBytes_length_atIndex(gs_ptr, 4, 10);
            }

            let rows_per_tg: u32 = 8;
            let grid_size = objc2_metal::MTLSize {
                width: self.config.intermediate_dim.div_ceil(rows_per_tg) as usize,
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
        }

        // Phase B: down projection
        {
            let kernel_name = self.config.bits.matvec_kernel_name();
            let pipeline = {
                let mut cache = self.ctx.pipeline_cache_mut();
                cache.get_or_create_pipeline(self.ctx.device(), kernel_name, None)?
            };

            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or(MetalError::EncoderCreation)?;

            encoder.setComputePipelineState(&pipeline);

            unsafe {
                encoder.setBuffer_offset_atIndex(Some(expert_buf), down_w_off, 0);
                encoder.setBuffer_offset_atIndex(Some(expert_buf), down_s_off, 1);
                encoder.setBuffer_offset_atIndex(Some(expert_buf), down_b_off, 2);
                encoder.setBuffer_offset_atIndex(Some(intermediate.metal_buffer()), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 4);

                let out_dim = self.config.hidden_dim;
                let in_dim = self.config.intermediate_dim;
                let group_size = self.config.group_size;
                let out_dim_ptr = NonNull::from(&out_dim).cast();
                encoder.setBytes_length_atIndex(out_dim_ptr, 4, 5);
                let in_dim_ptr = NonNull::from(&in_dim).cast();
                encoder.setBytes_length_atIndex(in_dim_ptr, 4, 6);
                let gs_ptr = NonNull::from(&group_size).cast();
                encoder.setBytes_length_atIndex(gs_ptr, 4, 7);
            }

            let rows_per_tg: u32 = 8;
            let grid_size = objc2_metal::MTLSize {
                width: self.config.hidden_dim.div_ceil(rows_per_tg) as usize,
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
        }

        Ok(())
    }
}

// ============================================================================
// Gather + QMM + SwiGLU (Resident Mode)
// ============================================================================

/// Parameters matching the Metal `GatherQmmSwigluParams` struct.
/// group_size and bits are now function constants (FC_GROUP_SIZE, FC_BITS).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GatherQmmSwigluParams {
    hidden_dim: u32,
    intermediate_dim: u32,
    num_tokens: u32,
    topk: u32,
}

/// Parameters matching the Metal `GatherDequantMatvecParams` struct.
/// group_size and bits are now function constants (FC_GROUP_SIZE, FC_BITS).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GatherDequantMatvecParams {
    in_dim: u32,
    out_dim: u32,
    num_tokens: u32,
    topk: u32,
}

/// Configuration for the resident-mode gather+QMM+SwiGLU kernel.
#[derive(Debug, Clone)]
pub struct GatherQmmSwigluConfig {
    /// Hidden dimension (D).
    pub hidden_dim: u32,
    /// Intermediate dimension (I).
    pub intermediate_dim: u32,
    /// Quantization group size.
    pub group_size: u32,
    /// Quantization bit width.
    pub bits: ExpertBits,
}

impl GatherQmmSwigluConfig {
    fn validate(&self) -> Result<()> {
        let pf = self.bits.pack_factor();
        if self.hidden_dim == 0 || self.hidden_dim % pf != 0 {
            return Err(MetalError::InvalidConfig(format!(
                "hidden_dim ({}) must be positive and divisible by pack_factor ({})",
                self.hidden_dim, pf
            )));
        }
        if self.intermediate_dim == 0 || self.intermediate_dim % pf != 0 {
            return Err(MetalError::InvalidConfig(format!(
                "intermediate_dim ({}) must be positive and divisible by pack_factor ({})",
                self.intermediate_dim, pf
            )));
        }
        if self.group_size < pf {
            return Err(MetalError::InvalidConfig(format!(
                "group_size ({}) must be >= pack_factor ({}) for {:?}-bit quantization",
                self.group_size, pf, self.bits
            )));
        }
        if self.hidden_dim % self.group_size != 0 || self.intermediate_dim % self.group_size != 0 {
            return Err(MetalError::InvalidConfig(format!(
                "both hidden_dim ({}) and intermediate_dim ({}) must be divisible by group_size ({})",
                self.hidden_dim, self.intermediate_dim, self.group_size
            )));
        }
        Ok(())
    }
}

/// Stacked quantized expert weight buffers for all experts (resident mode).
///
/// Weights are stacked along the expert axis: `[num_experts, ...]`
pub struct StackedExpertWeights {
    /// Gate projection packed weights `[E, I, D/pack_factor]`
    pub gate_weights: MetalBuffer<u32>,
    /// Gate projection scales `[E, I, D/group_size]`
    pub gate_scales: MetalBuffer<u16>,
    /// Gate projection biases `[E, I, D/group_size]`
    pub gate_biases: MetalBuffer<u16>,
    /// Up projection packed weights `[E, I, D/pack_factor]`
    pub up_weights: MetalBuffer<u32>,
    /// Up projection scales `[E, I, D/group_size]`
    pub up_scales: MetalBuffer<u16>,
    /// Up projection biases `[E, I, D/group_size]`
    pub up_biases: MetalBuffer<u16>,
    /// Down projection packed weights `[E, D, I/pack_factor]`
    pub down_weights: MetalBuffer<u32>,
    /// Down projection scales `[E, D, I/group_size]`
    pub down_scales: MetalBuffer<u16>,
    /// Down projection biases `[E, D, I/group_size]`
    pub down_biases: MetalBuffer<u16>,
}

/// Fused gather + quantized matmul + SwiGLU kernel for resident-mode inference.
///
/// When all expert weights fit in GPU memory, this replaces 3x `gather_mm` + `silu`
/// + `multiply` with two fused kernel dispatches:
/// 1. `gather_qmm_swiglu`: gate+up+SwiGLU with gathered expert weights
/// 2. `gather_dequant_matvec`: down projection with gathered expert weights
pub struct GatherQmmSwiglu {
    ctx: Arc<MetalContext>,
    config: GatherQmmSwigluConfig,
}

impl GatherQmmSwiglu {
    /// Create a new resident-mode fused kernel instance.
    pub fn new(ctx: Arc<MetalContext>, config: GatherQmmSwigluConfig) -> Result<Self> {
        config.validate()?;

        // Pre-create specialized pipelines with function constants
        let fc = Self::function_constants(&config);
        {
            let mut cache = ctx.pipeline_cache_mut();
            cache.get_or_create_specialized_pipeline_typed(
                ctx.device(),
                "gather_qmm_swiglu",
                &fc,
            )?;
            cache.get_or_create_specialized_pipeline_typed(
                ctx.device(),
                "gather_dequant_matvec",
                &fc,
            )?;
        }

        Ok(Self { ctx, config })
    }

    /// Build function constants for FC_GROUP_SIZE (index 0) and FC_BITS (index 1).
    fn function_constants(config: &GatherQmmSwigluConfig) -> HashMap<u64, FunctionConstant> {
        let bits_u32 = match config.bits {
            ExpertBits::Four => 4u32,
            ExpertBits::Two => 2u32,
        };
        let mut fc = HashMap::new();
        fc.insert(0, FunctionConstant::UInt(config.group_size));
        fc.insert(1, FunctionConstant::UInt(bits_u32));
        fc
    }

    /// Run the full gather+QMM+SwiGLU+down pipeline (blocking).
    ///
    /// Convenience wrapper. Use `encode_into` for non-blocking dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        num_tokens: u32,
        topk: u32,
        input: &MetalBuffer<f32>,
        expert_ids: &MetalBuffer<u32>,
        weights: &StackedExpertWeights,
        intermediate: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
    ) -> Result<()> {
        let command_buffer = self.create_command_buffer()?;
        self.encode_into(&command_buffer, num_tokens, topk, input, expert_ids, weights, intermediate, output)?;

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }
        Ok(())
    }

    /// Encode both phases into an external command buffer (non-blocking).
    ///
    /// The caller manages command buffer lifecycle. Encode multiple expert
    /// dispatches + combine into one CMD buffer, submit once, defer wait.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_into(
        &self,
        command_buffer: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        num_tokens: u32,
        topk: u32,
        input: &MetalBuffer<f32>,
        expert_ids: &MetalBuffer<u32>,
        weights: &StackedExpertWeights,
        intermediate: &MetalBuffer<f32>,
        output: &MetalBuffer<f32>,
    ) -> Result<()> {
        let fc = Self::function_constants(&self.config);

        self.encode_gate_up_swiglu(
            command_buffer, num_tokens, topk, input, expert_ids, weights, intermediate, &fc,
        )?;

        self.encode_down(
            command_buffer, num_tokens, topk, intermediate, expert_ids, weights, output, &fc,
        )?;

        Ok(())
    }

    /// Create a new command buffer from this context's queue.
    pub fn create_command_buffer(
        &self,
    ) -> Result<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>>> {
        let command_queue = self.ctx.command_queue();
        command_queue
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_gate_up_swiglu(
        &self,
        command_buffer: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        num_tokens: u32,
        topk: u32,
        input: &MetalBuffer<f32>,
        expert_ids: &MetalBuffer<u32>,
        weights: &StackedExpertWeights,
        output: &MetalBuffer<f32>,
        fc: &HashMap<u64, FunctionConstant>,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_specialized_pipeline_typed(
                self.ctx.device(),
                "gather_qmm_swiglu",
                fc,
            )?
        };

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.gate_weights.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights.gate_scales.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weights.gate_biases.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(weights.up_weights.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(weights.up_scales.metal_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(weights.up_biases.metal_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(input.metal_buffer()), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(expert_ids.metal_buffer()), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 8);

            let params = GatherQmmSwigluParams {
                hidden_dim: self.config.hidden_dim,
                intermediate_dim: self.config.intermediate_dim,
                num_tokens,
                topk,
            };
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 9);
        }

        let rows_per_tg: u32 = 8;
        let tiles_per_token_expert =
            self.config.intermediate_dim.div_ceil(rows_per_tg);
        let total_threadgroups = num_tokens * topk * tiles_per_token_expert;

        let grid_size = objc2_metal::MTLSize {
            width: total_threadgroups as usize,
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

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_down(
        &self,
        command_buffer: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
        num_tokens: u32,
        topk: u32,
        input: &MetalBuffer<f32>,
        expert_ids: &MetalBuffer<u32>,
        weights: &StackedExpertWeights,
        output: &MetalBuffer<f32>,
        fc: &HashMap<u64, FunctionConstant>,
    ) -> Result<()> {
        let pipeline = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_specialized_pipeline_typed(
                self.ctx.device(),
                "gather_dequant_matvec",
                fc,
            )?
        };

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        encoder.setComputePipelineState(&pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights.down_weights.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weights.down_scales.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(weights.down_biases.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(input.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(expert_ids.metal_buffer()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(output.metal_buffer()), 0, 5);

            let params = GatherDequantMatvecParams {
                in_dim: self.config.intermediate_dim,
                out_dim: self.config.hidden_dim,
                num_tokens,
                topk,
            };
            let params_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of_val(&params), 6);
        }

        let rows_per_tg: u32 = 8;
        let tiles_per_token_expert = self.config.hidden_dim.div_ceil(rows_per_tg);
        let total_threadgroups = num_tokens * topk * tiles_per_token_expert;

        let grid_size = objc2_metal::MTLSize {
            width: total_threadgroups as usize,
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

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_config_validation() {
        let valid = FusedMoeExpertConfig {
            hidden_dim: 4096,
            intermediate_dim: 1024,
            group_size: 64,
            bits: ExpertBits::Four,
        };
        assert!(valid.validate().is_ok());

        // hidden_dim not divisible by pack_factor (8 for 4-bit)
        let invalid_bits = FusedMoeExpertConfig {
            hidden_dim: 4097,
            intermediate_dim: 1024,
            group_size: 64,
            bits: ExpertBits::Four,
        };
        assert!(invalid_bits.validate().is_err());

        let valid_2bit = FusedMoeExpertConfig {
            hidden_dim: 4096,
            intermediate_dim: 1024,
            group_size: 64,
            bits: ExpertBits::Two,
        };
        assert!(valid_2bit.validate().is_ok());

        // BUG-5: group_size < pack_factor should fail
        let small_gs = FusedMoeExpertConfig {
            hidden_dim: 4096,
            intermediate_dim: 1024,
            group_size: 4, // < 8 (pack_factor for 4-bit)
            bits: ExpertBits::Four,
        };
        assert!(small_gs.validate().is_err());

        // Large hidden_dim (was capped at 8192, now unrestricted)
        let large_hidden = FusedMoeExpertConfig {
            hidden_dim: 16384,
            intermediate_dim: 4096,
            group_size: 64,
            bits: ExpertBits::Four,
        };
        assert!(large_hidden.validate().is_ok());
    }
}
