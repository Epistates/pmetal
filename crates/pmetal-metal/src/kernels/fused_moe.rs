#![allow(unsafe_code)]

//! Fused MoE (Mixture of Experts) Metal kernels.
//!
//! Provides GPU-accelerated fused operations for MoE inference:
//!
//! - [`FusedMoeExpert`]: Fused gate+up+SwiGLU + down projection for quantized experts
//!
//! All dequant kernels use the qdot technique (pre-scaled activations) from MLX's
//! `quantized.h`, eliminating per-nibble shifts from the inner loop for 30-40%
//! compute reduction. Thread-private registers replace threadgroup shared memory
//! to avoid the `x_shared[4096]` overflow bug and reduce barrier overhead.

use std::ptr::NonNull;
use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};

use crate::{
    buffer::MetalBuffer,
    context::{CommandQueueExt, MetalContext},
    error::{MetalError, Result},
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

    /// The kernel that computes the gate and up rows with the SwiGLU.
    fn gate_up_kernel_name(self) -> &'static str {
        match self {
            Self::Four => "fused_gate_up_swiglu_4bit",
            Self::Two => "fused_gate_up_swiglu_2bit",
        }
    }

    /// The kernel that computes the down projection.
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
        // A group must be a whole number of packed words: the kernels find a
        // word's group as `word / (group_size / pack_factor)`.
        if self.group_size < pf || self.group_size % pf != 0 {
            return Err(MetalError::InvalidConfig(format!(
                "group_size ({}) must be a multiple of pack_factor ({}) for {:?}-bit quantization",
                self.group_size, pf, self.bits
            )));
        }
        Ok(())
    }
}

/// Fused quantized expert forward kernel.
///
/// For offloaded/single-expert inference (T=1 decode):
/// 1. Phase A: `fused_gate_up_swiglu_*bit` — dequant gate+up weights, compute SwiGLU
/// 2. Phase B: `dequant_matvec_*bit` — dequant down weights, compute projection
///
/// Both phases are dispatched in a single command buffer for minimal overhead.
pub struct FusedMoeExpert {
    ctx: Arc<MetalContext>,
    config: FusedMoeExpertConfig,
}

impl std::fmt::Debug for FusedMoeExpert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedMoeExpert")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
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
            cache.get_or_create_pipeline(ctx.device(), config.bits.gate_up_kernel_name(), None)?;
            cache.get_or_create_pipeline(ctx.device(), config.bits.matvec_kernel_name(), None)?;
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
        let command_buffer = self.create_command_buffer()?;
        self.encode_into(&command_buffer, input, weights, output, intermediate)?;
        self.submit_and_wait(&command_buffer)
    }

    /// Encode both phases into an external command buffer (non-blocking).
    ///
    /// The caller manages command buffer lifecycle: create, accumulate multiple
    /// expert dispatches, commit once, defer wait. This is the pipelined path
    /// — encode all K experts into one CMD buffer, submit, defer sync to next layer.
    pub fn encode_into(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &MetalBuffer<f32>,
        weights: &ExpertWeightBuffers,
        output: &MetalBuffer<f32>,
        intermediate: &MetalBuffer<f32>,
    ) -> Result<()> {
        self.encode_into_with_offsets(command_buffer, input, 0, weights, output, 0, intermediate)
    }

    /// Encode both phases into an external command buffer (non-blocking),
    /// reading and writing from element offsets inside the provided buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_into_with_offsets(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &MetalBuffer<f32>,
        input_offset_elems: usize,
        weights: &ExpertWeightBuffers,
        output: &MetalBuffer<f32>,
        output_offset_elems: usize,
        intermediate: &MetalBuffer<f32>,
    ) -> Result<()> {
        let part = Part::whole;
        let parts = [
            part(weights.gate_weights.metal_buffer()),
            part(weights.gate_scales.metal_buffer()),
            part(weights.gate_biases.metal_buffer()),
            part(weights.up_weights.metal_buffer()),
            part(weights.up_scales.metal_buffer()),
            part(weights.up_biases.metal_buffer()),
            part(weights.down_weights.metal_buffer()),
            part(weights.down_scales.metal_buffer()),
            part(weights.down_biases.metal_buffer()),
        ];
        self.encode_parts(
            command_buffer,
            input,
            input_offset_elems,
            &parts,
            output,
            output_offset_elems,
            intermediate,
        )
    }

    /// Create a new command buffer from this context's queue.
    pub fn create_command_buffer(
        &self,
    ) -> Result<objc2::rc::Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        self.ctx.command_queue().new_command_buffer()
    }

    /// Submit a command buffer and wait for GPU completion.
    pub fn submit_and_wait(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<()> {
        self.submit(command_buffer)?;
        self.wait_for_completion(command_buffer)
    }

    /// Submit a command buffer without waiting.
    pub fn submit(&self, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
        command_buffer.commit();
        Ok(())
    }

    /// Wait for a previously submitted command buffer.
    pub fn wait_for_completion(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<()> {
        command_buffer.waitUntilCompleted();
        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }
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
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &MetalBuffer<f32>,
        expert_buf: &ProtocolObject<dyn MTLBuffer>,
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
        self.encode_expert_aligned_with_offsets(
            command_buffer,
            input,
            0,
            expert_buf,
            gate_w_off,
            gate_s_off,
            gate_b_off,
            up_w_off,
            up_s_off,
            up_b_off,
            down_w_off,
            down_s_off,
            down_b_off,
            output,
            0,
            intermediate,
        )
    }

    /// Offset-capable aligned expert encode.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_expert_aligned_with_offsets(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &MetalBuffer<f32>,
        input_offset_elems: usize,
        expert_buf: &ProtocolObject<dyn MTLBuffer>,
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
        output_offset_elems: usize,
        intermediate: &MetalBuffer<f32>,
    ) -> Result<()> {
        let part = |offset| Part {
            buffer: expert_buf,
            offset,
        };
        let parts = [
            part(gate_w_off),
            part(gate_s_off),
            part(gate_b_off),
            part(up_w_off),
            part(up_s_off),
            part(up_b_off),
            part(down_w_off),
            part(down_s_off),
            part(down_b_off),
        ];
        self.encode_parts(
            command_buffer,
            input,
            input_offset_elems,
            &parts,
            output,
            output_offset_elems,
            intermediate,
        )
    }

    /// The bytes each of the nine weight parts must hold, in
    /// [`Part`] order: gate, up and down, each as words, scales, biases.
    fn part_bytes(&self) -> [usize; 9] {
        let (hidden, inter) = (
            self.config.hidden_dim as usize,
            self.config.intermediate_dim as usize,
        );
        let (pack, group) = (
            self.config.bits.pack_factor() as usize,
            self.config.group_size as usize,
        );
        let words = |rows: usize, cols: usize| rows * (cols / pack) * std::mem::size_of::<u32>();
        let halves = |rows: usize, cols: usize| rows * (cols / group) * std::mem::size_of::<u16>();
        [
            words(inter, hidden),
            halves(inter, hidden),
            halves(inter, hidden),
            words(inter, hidden),
            halves(inter, hidden),
            halves(inter, hidden),
            words(hidden, inter),
            halves(hidden, inter),
            halves(hidden, inter),
        ]
    }

    /// Both phases of one expert, in one compute encoder: the dispatches of a
    /// serial encoder run in order, so the down projection sees the whole
    /// intermediate vector.
    #[allow(clippy::too_many_arguments)]
    fn encode_parts(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        input: &MetalBuffer<f32>,
        input_offset_elems: usize,
        parts: &[Part<'_>; 9],
        output: &MetalBuffer<f32>,
        output_offset_elems: usize,
        intermediate: &MetalBuffer<f32>,
    ) -> Result<()> {
        let (hidden, inter) = (
            self.config.hidden_dim as usize,
            self.config.intermediate_dim as usize,
        );
        validate_offset_span(input.len(), input_offset_elems, hidden, "input")?;
        validate_offset_span(output.len(), output_offset_elems, hidden, "output")?;
        validate_offset_span(intermediate.len(), 0, inter, "intermediate")?;
        const NAMES: [&str; 9] = [
            "gate weights",
            "gate scales",
            "gate biases",
            "up weights",
            "up scales",
            "up biases",
            "down weights",
            "down scales",
            "down biases",
        ];
        for ((part, bytes), name) in parts.iter().zip(self.part_bytes()).zip(NAMES) {
            part.validate(bytes, name)?;
        }

        let gate_up = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(
                self.ctx.device(),
                self.config.bits.gate_up_kernel_name(),
                None,
            )?
        };
        let down = {
            let mut cache = self.ctx.pipeline_cache_mut();
            cache.get_or_create_pipeline(
                self.ctx.device(),
                self.config.bits.matvec_kernel_name(),
                None,
            )?
        };

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;
        let threadgroup_size = MTLSize {
            width: TG_SIZE,
            height: 1,
            depth: 1,
        };
        let group_size = self.config.group_size;

        // Phase A: gate and up rows, with the SwiGLU, into `intermediate`.
        encoder.setComputePipelineState(&gate_up);
        // SAFETY: every buffer outlives the command buffer's execution, the
        // part spans were validated above, and the scalars are copied by
        // `setBytes`.
        unsafe {
            for (index, part) in parts[..6].iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(part.buffer), part.offset, index);
            }
            encoder.setBuffer_offset_atIndex(
                Some(input.metal_buffer()),
                byte_offset::<f32>(input_offset_elems)?,
                6,
            );
            encoder.setBuffer_offset_atIndex(Some(intermediate.metal_buffer()), 0, 7);
            set_u32(&encoder, inter as u32, 8);
            set_u32(&encoder, hidden as u32, 9);
            set_u32(&encoder, group_size, 10);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: inter.div_ceil(ROWS_PER_TG),
                height: 1,
                depth: 1,
            },
            threadgroup_size,
        );

        // Phase B: the down projection of the intermediate vector.
        encoder.setComputePipelineState(&down);
        // SAFETY: as above.
        unsafe {
            for (index, part) in parts[6..].iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(part.buffer), part.offset, index);
            }
            encoder.setBuffer_offset_atIndex(Some(intermediate.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(
                Some(output.metal_buffer()),
                byte_offset::<f32>(output_offset_elems)?,
                4,
            );
            set_u32(&encoder, hidden as u32, 5);
            set_u32(&encoder, inter as u32, 6);
            set_u32(&encoder, group_size, 7);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: hidden.div_ceil(ROWS_PER_TG),
                height: 1,
                depth: 1,
            },
            threadgroup_size,
        );
        encoder.endEncoding();
        Ok(())
    }
}

/// Output rows one threadgroup computes (`ROWS_PER_TG` in the shader:
/// 2 simdgroups of 4 rows) and its thread count.
const ROWS_PER_TG: usize = 8;
const TG_SIZE: usize = 64;

/// One weight component: a Metal buffer and the byte offset it starts at.
struct Part<'a> {
    buffer: &'a ProtocolObject<dyn MTLBuffer>,
    offset: usize,
}

impl<'a> Part<'a> {
    /// A whole buffer, from its first byte.
    fn whole(buffer: &'a ProtocolObject<dyn MTLBuffer>) -> Self {
        Self { buffer, offset: 0 }
    }

    /// Fail unless `bytes` bytes from the offset lie inside the buffer, and
    /// the offset is a multiple of 4 (the kernels read 32-bit words, and a
    /// scale or bias table may start mid-record).
    fn validate(&self, bytes: usize, name: &'static str) -> Result<()> {
        if self.offset % 4 != 0 {
            return Err(MetalError::InvalidConfig(format!(
                "expert {name} offset {} is not a multiple of 4",
                self.offset
            )));
        }
        let end = self
            .offset
            .checked_add(bytes)
            .ok_or_else(|| MetalError::InvalidConfig(format!("expert {name} offset overflow")))?;
        if end > self.buffer.length() {
            return Err(MetalError::InvalidConfig(format!(
                "expert {name} span [{}..{end}) exceeds its buffer of {} bytes",
                self.offset,
                self.buffer.length()
            )));
        }
        Ok(())
    }
}

/// Bind a 32-bit scalar as a kernel constant.
///
/// # Safety
///
/// `index` must be a `constant uint&` argument of the bound pipeline.
unsafe fn set_u32(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: u32,
    index: usize,
) {
    // SAFETY: `setBytes` copies the four bytes before returning.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::from(&value).cast(),
            std::mem::size_of::<u32>(),
            index,
        );
    }
}

fn validate_offset_span(
    buffer_len: usize,
    offset_elems: usize,
    span_elems: usize,
    label: &'static str,
) -> Result<()> {
    let end = offset_elems.checked_add(span_elems).ok_or_else(|| {
        MetalError::InvalidConfig(format!(
            "{label} offset overflow: offset={offset_elems}, span={span_elems}"
        ))
    })?;
    if end > buffer_len {
        return Err(MetalError::InvalidConfig(format!(
            "{label} offset range [{offset_elems}..{end}) exceeds buffer length {buffer_len}"
        )));
    }
    Ok(())
}

fn byte_offset<T>(offset_elems: usize) -> Result<usize> {
    offset_elems
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| {
            MetalError::InvalidConfig(format!(
                "buffer byte offset overflow: offset={offset_elems}, elem_size={}",
                std::mem::size_of::<T>()
            ))
        })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_offset_span_accepts_in_bounds_range() {
        validate_offset_span(1024, 128, 256, "input").unwrap();
        validate_offset_span(256, 0, 256, "output").unwrap();
    }

    #[test]
    fn test_validate_offset_span_rejects_out_of_bounds_range() {
        let error = validate_offset_span(256, 128, 256, "input").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("input offset range"));
        assert!(message.contains("exceeds buffer length 256"));
    }

    #[test]
    fn test_byte_offset_overflow_is_reported() {
        let error = byte_offset::<f32>(usize::MAX).unwrap_err();
        assert!(error.to_string().contains("buffer byte offset overflow"));
    }

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

    // ------------------------------------------------------------------
    // Numerical checks against a float64 reference.
    // ------------------------------------------------------------------

    use crate::buffer::BufferUsage;

    /// Deterministic uniform values in [-1, 1).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    fn bf16_bits(value: f64) -> u16 {
        half::bf16::from_f64(value).to_bits()
    }

    fn bf16_value(bits: u16) -> f64 {
        f64::from(half::bf16::from_bits(bits).to_f32())
    }

    /// An affine-quantized `[rows, cols]` matrix in MLX's layout (element
    /// `j` of a word at bits `j * bits`, bf16 scales and biases per group),
    /// with the f64 values it dequantizes to.
    struct QuantMatrix {
        words: Vec<u32>,
        scales: Vec<u16>,
        biases: Vec<u16>,
        dequantized: Vec<f64>,
    }

    fn quantize(rng: &mut Lcg, rows: usize, cols: usize, group: usize, bits: usize) -> QuantMatrix {
        let levels = (1u32 << bits) - 1;
        let per_word = 32 / bits;
        let mut words = vec![0u32; rows * cols / per_word];
        let mut scales = vec![0u16; rows * cols / group];
        let mut biases = vec![0u16; rows * cols / group];
        let mut dequantized = vec![0.0; rows * cols];
        for row in 0..rows {
            for g in 0..cols / group {
                let weights: Vec<f64> = (0..group).map(|_| rng.next() * 0.5).collect();
                let lo = weights.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let scale = bf16_bits((hi - lo) / f64::from(levels));
                let bias = bf16_bits(lo);
                scales[row * (cols / group) + g] = scale;
                biases[row * (cols / group) + g] = bias;
                for (k, w) in weights.iter().enumerate() {
                    let q = ((w - bf16_value(bias)) / bf16_value(scale))
                        .round()
                        .clamp(0.0, f64::from(levels)) as u32;
                    let col = g * group + k;
                    words[(row * cols + col) / per_word] |= q << (bits * (col % per_word));
                    dequantized[row * cols + col] =
                        bf16_value(scale) * f64::from(q) + bf16_value(bias);
                }
            }
        }
        QuantMatrix {
            words,
            scales,
            biases,
            dequantized,
        }
    }

    fn matvec(w: &[f64], x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        (0..rows)
            .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
            .collect()
    }

    /// `down(silu(gate(x)) * up(x))` in f64, with the L1 mass of each output's
    /// terms as the error scale (the kernel accumulates in f32).
    fn reference_expert(
        gate: &QuantMatrix,
        up: &QuantMatrix,
        down: &QuantMatrix,
        x: &[f64],
        hidden: usize,
        inter: usize,
    ) -> Vec<(f64, f64)> {
        let g = matvec(&gate.dequantized, x, inter, hidden);
        let u = matvec(&up.dequantized, x, inter, hidden);
        let act: Vec<f64> = g
            .iter()
            .zip(&u)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let out = matvec(&down.dequantized, &act, hidden, inter);
        (0..hidden)
            .map(|r| {
                let mass: f64 = (0..inter)
                    .map(|c| (down.dequantized[r * inter + c] * act[c]).abs())
                    .sum();
                (out[r], mass)
            })
            .collect()
    }

    fn weight_buffers(
        ctx: &MetalContext,
        gate: &QuantMatrix,
        up: &QuantMatrix,
        down: &QuantMatrix,
    ) -> ExpertWeightBuffers {
        let usage = BufferUsage::Shared;
        ExpertWeightBuffers {
            gate_weights: MetalBuffer::from_slice(ctx, &gate.words, usage).unwrap(),
            gate_scales: MetalBuffer::from_slice(ctx, &gate.scales, usage).unwrap(),
            gate_biases: MetalBuffer::from_slice(ctx, &gate.biases, usage).unwrap(),
            up_weights: MetalBuffer::from_slice(ctx, &up.words, usage).unwrap(),
            up_scales: MetalBuffer::from_slice(ctx, &up.scales, usage).unwrap(),
            up_biases: MetalBuffer::from_slice(ctx, &up.biases, usage).unwrap(),
            down_weights: MetalBuffer::from_slice(ctx, &down.words, usage).unwrap(),
            down_scales: MetalBuffer::from_slice(ctx, &down.scales, usage).unwrap(),
            down_biases: MetalBuffer::from_slice(ctx, &down.biases, usage).unwrap(),
        }
    }

    fn check_expert(hidden: usize, inter: usize, group: usize, bits: ExpertBits, seed: u64) {
        let ctx = MetalContext::global().expect("Metal context");
        let b = match bits {
            ExpertBits::Four => 4,
            ExpertBits::Two => 2,
        };
        let mut rng = Lcg(seed);
        let gate = quantize(&mut rng, inter, hidden, group, b);
        let up = quantize(&mut rng, inter, hidden, group, b);
        let down = quantize(&mut rng, hidden, inter, group, b);
        let x32: Vec<f32> = (0..hidden).map(|_| rng.next() as f32).collect();
        let x: Vec<f64> = x32.iter().map(|&v| f64::from(v)).collect();

        let expert = FusedMoeExpert::new(
            ctx.clone(),
            FusedMoeExpertConfig {
                hidden_dim: hidden as u32,
                intermediate_dim: inter as u32,
                group_size: group as u32,
                bits,
            },
        )
        .unwrap();
        let weights = weight_buffers(&ctx, &gate, &up, &down);
        let input = MetalBuffer::from_slice(&ctx, &x32, BufferUsage::Shared).unwrap();
        let output = MetalBuffer::<f32>::zeros(&ctx, hidden, BufferUsage::Shared).unwrap();
        let scratch = MetalBuffer::<f32>::zeros(&ctx, inter, BufferUsage::Shared).unwrap();
        expert
            .forward_single_expert(&input, &weights, &output, &scratch)
            .unwrap();

        let want = reference_expert(&gate, &up, &down, &x, hidden, inter);
        for (row, (&got, &(want, mass))) in output.as_slice().iter().zip(&want).enumerate() {
            let err = (f64::from(got) - want).abs();
            assert!(
                err <= 2e-5 * mass.max(1e-3),
                "{bits:?} hidden {hidden} inter {inter} group {group} row {row}: \
                 got {got}, want {want} (err {err:e}, mass {mass:e})"
            );
        }
    }

    #[test]
    fn four_bit_expert_matches_float64_reference() {
        check_expert(256, 128, 64, ExpertBits::Four, 1);
        check_expert(2048, 512, 64, ExpertBits::Four, 2);
        // A group no wider than the 8 values a word holds.
        check_expert(64, 40, 8, ExpertBits::Four, 3);
    }

    #[test]
    fn two_bit_expert_matches_float64_reference() {
        check_expert(256, 128, 64, ExpertBits::Two, 4);
        check_expert(2048, 512, 64, ExpertBits::Two, 5);
        check_expert(128, 48, 16, ExpertBits::Two, 6);
    }

    /// The zero-copy path hands the kernel one buffer with every component at
    /// a byte offset, as the expert reader fills it.
    #[test]
    fn an_expert_read_from_one_buffer_matches_its_separate_buffers() {
        let ctx = MetalContext::global().expect("Metal context");
        let (hidden, inter, group) = (256usize, 128usize, 64usize);
        let mut rng = Lcg(7);
        let gate = quantize(&mut rng, inter, hidden, group, 4);
        let up = quantize(&mut rng, inter, hidden, group, 4);
        let down = quantize(&mut rng, hidden, inter, group, 4);
        let x32: Vec<f32> = (0..hidden).map(|_| rng.next() as f32).collect();

        // One record: [gate w, s, b][up w, s, b][down w, s, b], each part
        // starting on a 16-byte boundary.
        let mut record: Vec<u8> = Vec::new();
        let mut place = |bytes: &[u8]| {
            while record.len() % 16 != 0 {
                record.push(0);
            }
            let at = record.len();
            record.extend_from_slice(bytes);
            at
        };
        let words = |w: &[u32]| w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        let halves = |h: &[u16]| h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        let offsets = [
            place(&words(&gate.words)),
            place(&halves(&gate.scales)),
            place(&halves(&gate.biases)),
            place(&words(&up.words)),
            place(&halves(&up.scales)),
            place(&halves(&up.biases)),
            place(&words(&down.words)),
            place(&halves(&down.scales)),
            place(&halves(&down.biases)),
        ];
        let buffer = MetalBuffer::from_slice(&ctx, &record, BufferUsage::Shared).unwrap();

        let expert = FusedMoeExpert::new(
            ctx.clone(),
            FusedMoeExpertConfig {
                hidden_dim: hidden as u32,
                intermediate_dim: inter as u32,
                group_size: group as u32,
                bits: ExpertBits::Four,
            },
        )
        .unwrap();
        let input = MetalBuffer::from_slice(&ctx, &x32, BufferUsage::Shared).unwrap();
        let scratch = MetalBuffer::<f32>::zeros(&ctx, inter, BufferUsage::Shared).unwrap();
        let aligned = MetalBuffer::<f32>::zeros(&ctx, hidden, BufferUsage::Shared).unwrap();
        let command_buffer = expert.create_command_buffer().unwrap();
        expert
            .encode_expert_aligned(
                &command_buffer,
                &input,
                buffer.metal_buffer(),
                offsets[0],
                offsets[1],
                offsets[2],
                offsets[3],
                offsets[4],
                offsets[5],
                offsets[6],
                offsets[7],
                offsets[8],
                &aligned,
                &scratch,
            )
            .unwrap();
        expert.submit_and_wait(&command_buffer).unwrap();

        let separate = MetalBuffer::<f32>::zeros(&ctx, hidden, BufferUsage::Shared).unwrap();
        let weights = weight_buffers(&ctx, &gate, &up, &down);
        expert
            .forward_single_expert(&input, &weights, &separate, &scratch)
            .unwrap();
        assert_eq!(aligned.as_slice(), separate.as_slice());
    }

    /// A weight table that runs off the end of its buffer would be read past
    /// the allocation by the GPU, silently; the encode refuses it.
    #[test]
    fn a_weight_table_outside_its_buffer_is_refused() {
        let ctx = MetalContext::global().expect("Metal context");
        let (hidden, inter, group) = (64usize, 32usize, 32usize);
        let mut rng = Lcg(8);
        let gate = quantize(&mut rng, inter, hidden, group, 4);
        let up = quantize(&mut rng, inter, hidden, group, 4);
        let down = quantize(&mut rng, hidden, inter, group, 4);
        let expert = FusedMoeExpert::new(
            ctx.clone(),
            FusedMoeExpertConfig {
                hidden_dim: hidden as u32,
                intermediate_dim: inter as u32,
                group_size: group as u32,
                bits: ExpertBits::Four,
            },
        )
        .unwrap();
        let usage = BufferUsage::Shared;
        let input = MetalBuffer::<f32>::zeros(&ctx, hidden, usage).unwrap();
        let output = MetalBuffer::<f32>::zeros(&ctx, hidden, usage).unwrap();
        let scratch = MetalBuffer::<f32>::zeros(&ctx, inter, usage).unwrap();

        let mut weights = weight_buffers(&ctx, &gate, &up, &down);
        weights.up_scales =
            MetalBuffer::from_slice(&ctx, &up.scales[..up.scales.len() - 1], usage).unwrap();
        let err = expert
            .forward_single_expert(&input, &weights, &output, &scratch)
            .unwrap_err();
        assert!(err.to_string().contains("up scales"), "{err}");

        // An offset that is not word aligned.
        let record = MetalBuffer::<u8>::zeros(&ctx, 1 << 16, usage).unwrap();
        let command_buffer = expert.create_command_buffer().unwrap();
        let err = expert
            .encode_expert_aligned(
                &command_buffer,
                &input,
                record.metal_buffer(),
                0,
                8192,
                8194,
                16384,
                24576,
                24580,
                32768,
                40960,
                40964,
                &output,
                &scratch,
            )
            .unwrap_err();
        assert!(err.to_string().contains("multiple of 4"), "{err}");
    }

    /// The kernel finds a word's group by `column / (group_size / pack_factor)`,
    /// which is only right when a group is a whole number of words.
    #[test]
    fn a_group_that_is_not_a_whole_number_of_words_is_refused() {
        let config = FusedMoeExpertConfig {
            hidden_dim: 48,
            intermediate_dim: 48,
            group_size: 12,
            bits: ExpertBits::Four,
        };
        assert!(config.validate().is_err());
    }
}
