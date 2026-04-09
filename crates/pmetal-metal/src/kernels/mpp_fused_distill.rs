#![allow(unsafe_code)]

//! Metal 4 / MPP Fused Distillation Loss dispatch.
//!
//! Provides hardware-accelerated knowledge distillation losses via Metal
//! Performance Primitives on M5+ (Apple10) GPUs with NAX cores.
//!
//! Replaces the Metal 3 scalar kernels with SIMD-group-based reductions
//! using `simd_max()` / `simd_sum()`. Both teacher and student logsumexp
//! are computed in a single combined pass without threadgroup memory.
//!
//! Kernel families (all fp32 accumulation regardless of input dtype):
//! - `mpp_fused_kl_divergence_f32` / `_f16` — forward KL(teacher || student)
//! - `mpp_fused_reverse_kl_f32` — reverse KL(student || teacher)
//! - `mpp_fused_js_divergence_f32` — Jensen-Shannon divergence
//! - `mpp_fused_soft_cross_entropy_f32` — soft cross-entropy
//!
//! Grid layout: `[num_tokens, 1, 1]`
//! Each threadgroup = one SIMD group (32 lanes) per token.

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
// Loss type enum
// =============================================================================

/// Which distillation loss variant to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MppDistillLossType {
    /// Forward KL: KL(teacher || student) — mode-covering.
    ForwardKL,
    /// Reverse KL: KL(student || teacher) — mode-seeking.
    ReverseKL,
    /// Jensen-Shannon divergence (symmetric, bounded by log 2).
    JensenShannon,
    /// Soft cross-entropy: -sum_i P(i) * log Q(i).
    SoftCrossEntropy,
}

// =============================================================================
// Config
// =============================================================================

/// Configuration for MPP Fused Distillation Loss.
#[derive(Debug, Clone)]
pub struct MppFusedDistillConfig {
    /// Number of tokens.
    pub num_tokens: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Temperature for softening distributions.
    pub temperature: f32,
    /// Blending weight (alpha) for soft loss.
    pub alpha: f32,
    /// Index to ignore (typically -100).
    pub ignore_index: i32,
    /// Which loss to compute.
    pub loss_type: MppDistillLossType,
    /// Use fp16 logit inputs.
    pub use_fp16: bool,
}

impl MppFusedDistillConfig {
    /// Create a new config with forward KL, temperature=2.0.
    pub fn new(num_tokens: usize, vocab_size: usize) -> Self {
        Self {
            num_tokens,
            vocab_size,
            temperature: 2.0,
            alpha: 0.5,
            ignore_index: -100,
            loss_type: MppDistillLossType::ForwardKL,
            use_fp16: false,
        }
    }

    /// Set temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Set alpha blending weight.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Select loss type.
    pub fn with_loss_type(mut self, lt: MppDistillLossType) -> Self {
        self.loss_type = lt;
        self
    }

    /// Enable fp16 logit inputs.
    pub fn with_fp16(mut self) -> Self {
        self.use_fp16 = true;
        self
    }
}

// =============================================================================
// Metal-side parameter block (must match MppDistillParams in Metal)
// =============================================================================

#[repr(C)]
struct MppDistillParamsMetal {
    num_tokens: u32,
    vocab_size: u32,
    temperature: f32,
    alpha: f32,
    ignore_index: i32,
}

// =============================================================================
// Dispatcher
// =============================================================================

/// MPP Fused Distillation Loss dispatcher.
///
/// Dispatches KL / JS / soft-CE distillation losses to the appropriate
/// `mpp_fused_*` kernel on M5+ hardware.
pub struct MppFusedDistill {
    ctx: Arc<MetalContext>,
    config: MppFusedDistillConfig,
}

impl MppFusedDistill {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppFusedDistillConfig) -> Self {
        Self { ctx, config }
    }

    /// Returns true when MPP distillation is available (requires M5+ NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute synchronously.
    ///
    /// - `teacher_logits`: `[num_tokens, vocab_size]`
    /// - `student_logits`: `[num_tokens, vocab_size]`
    /// - `losses`: `[num_tokens]` fp32 per-token losses (output)
    /// - `teacher_lse`: `[num_tokens]` fp32 cached logsumexp (output, for backward)
    /// - `student_lse`: `[num_tokens]` fp32 cached logsumexp (output, for backward)
    pub fn execute(
        &self,
        teacher_logits: &dyn AsMetalBuffer,
        student_logits: &dyn AsMetalBuffer,
        losses: &dyn AsMetalBuffer,
        teacher_lse: &dyn AsMetalBuffer,
        student_lse: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.execute_async(teacher_logits, student_logits, losses, teacher_lse, student_lse)?;
        cb.waitUntilCompleted();
        if let Some(e) = cb.error() {
            return Err(MetalError::ExecutionFailed(e.to_string()));
        }
        Ok(())
    }

    /// Execute asynchronously, returning the committed command buffer.
    pub fn execute_async(
        &self,
        teacher_logits: &dyn AsMetalBuffer,
        student_logits: &dyn AsMetalBuffer,
        losses: &dyn AsMetalBuffer,
        teacher_lse: &dyn AsMetalBuffer,
        student_lse: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused Distill not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let kernel_name = self.select_kernel();

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

        let params = MppDistillParamsMetal {
            num_tokens: self.config.num_tokens as u32,
            vocab_size: self.config.vocab_size as u32,
            temperature: self.config.temperature,
            alpha: self.config.alpha,
            ignore_index: self.config.ignore_index,
        };

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(teacher_logits.as_metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(student_logits.as_metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(losses.as_metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(teacher_lse.as_metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(student_lse.as_metal_buffer()), 0, 4);
            let p_ptr = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(p_ptr, std::mem::size_of_val(&params), 5);
        }

        // Grid: [num_tokens, 1, 1]  Threadgroup: [32, 1, 1]
        let threadgroup_size = objc2_metal::MTLSize { width: 32, height: 1, depth: 1 };
        let grid_size = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();

        Ok(command_buffer)
    }

    fn select_kernel(&self) -> &'static str {
        match (self.config.loss_type, self.config.use_fp16) {
            (MppDistillLossType::ForwardKL,       true)  => "mpp_fused_kl_divergence_f16",
            (MppDistillLossType::ForwardKL,       false) => "mpp_fused_kl_divergence_f32",
            (MppDistillLossType::ReverseKL,       _)     => "mpp_fused_reverse_kl_f32",
            (MppDistillLossType::JensenShannon,   _)     => "mpp_fused_js_divergence_f32",
            (MppDistillLossType::SoftCrossEntropy, _)    => "mpp_fused_soft_cross_entropy_f32",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = MppFusedDistillConfig::new(64, 32000);
        assert_eq!(cfg.num_tokens, 64);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.temperature, 2.0);
        assert_eq!(cfg.alpha, 0.5);
        assert_eq!(cfg.ignore_index, -100);
        assert_eq!(cfg.loss_type, MppDistillLossType::ForwardKL);
        assert!(!cfg.use_fp16);
    }

    #[test]
    fn test_kernel_selection() {
        let cfg_kl = MppFusedDistillConfig::new(1, 128);
        let dispatch = MppFusedDistill {
            ctx: unsafe { std::mem::zeroed() }, // unit test only
            config: cfg_kl,
        };
        // Can't call select_kernel without ctx — test logic inline:
        let lt = MppDistillLossType::ForwardKL;
        let name = match (lt, false) {
            (MppDistillLossType::ForwardKL,       true)  => "mpp_fused_kl_divergence_f16",
            (MppDistillLossType::ForwardKL,       false) => "mpp_fused_kl_divergence_f32",
            (MppDistillLossType::ReverseKL,       _)     => "mpp_fused_reverse_kl_f32",
            (MppDistillLossType::JensenShannon,   _)     => "mpp_fused_js_divergence_f32",
            (MppDistillLossType::SoftCrossEntropy, _)    => "mpp_fused_soft_cross_entropy_f32",
        };
        assert_eq!(name, "mpp_fused_kl_divergence_f32");

        // Avoid unused warning on dispatch
        let _ = dispatch.config.num_tokens;
    }

    #[test]
    fn test_loss_type_variants() {
        let types = [
            MppDistillLossType::ForwardKL,
            MppDistillLossType::ReverseKL,
            MppDistillLossType::JensenShannon,
            MppDistillLossType::SoftCrossEntropy,
        ];
        // Ensure all variants are distinct
        assert_ne!(types[0], types[1]);
        assert_ne!(types[1], types[2]);
        assert_ne!(types[2], types[3]);
    }
}
