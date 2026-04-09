#![allow(unsafe_code)]

//! Metal 4 / MPP Fused Cross-Entropy dispatch.
//!
//! Provides hardware-accelerated cross-entropy loss computation via Metal
//! Performance Primitives on M5+ (Apple10) GPUs with NAX cores.
//!
//! The MPP variant replaces threadgroup-memory tree reductions with
//! `simd_max()` / `simd_sum()` reductions, keeping the entire forward +
//! backward computation within a single SIMD group (32 lanes) per token.
//!
//! Kernel families:
//! - `mpp_fused_cross_entropy_fwd_bwd_f32` / `_f16` — forward + backward
//! - `mpp_cross_entropy_forward_f32` — forward only (eval / inference)
//!
//! Grid layout: `[num_tokens, 1, 1]`
//! Each threadgroup is exactly one SIMD group (32 lanes).

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

/// Configuration for the MPP Fused Cross-Entropy kernel.
#[derive(Debug, Clone)]
pub struct MppFusedCrossEntropyConfig {
    /// Number of tokens (batch * seq, after label shift).
    pub num_tokens: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Index to ignore in loss computation (typically -100).
    pub ignore_index: i32,
    /// Use fp16 logits / gradients.
    pub use_fp16: bool,
    /// Forward-only mode — skip gradient computation.
    pub forward_only: bool,
}

impl MppFusedCrossEntropyConfig {
    /// Create a new config with default settings.
    pub fn new(num_tokens: usize, vocab_size: usize) -> Self {
        Self {
            num_tokens,
            vocab_size,
            ignore_index: -100,
            use_fp16: false,
            forward_only: false,
        }
    }

    /// Enable fp16 mode.
    pub fn with_fp16(mut self) -> Self {
        self.use_fp16 = true;
        self
    }

    /// Forward-only mode (no gradient output).
    pub fn forward_only(mut self) -> Self {
        self.forward_only = true;
        self
    }
}

// =============================================================================
// Dispatcher
// =============================================================================

/// MPP Fused Cross-Entropy dispatcher.
///
/// Dispatches to `mpp_fused_cross_entropy_fwd_bwd_{f32,f16}` on M5+ hardware.
pub struct MppFusedCrossEntropy {
    ctx: Arc<MetalContext>,
    config: MppFusedCrossEntropyConfig,
}

impl MppFusedCrossEntropy {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppFusedCrossEntropyConfig) -> Self {
        Self { ctx, config }
    }

    /// Returns true when MPP cross-entropy is available (requires M5+ NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute synchronously.
    ///
    /// - `logits`: `[N, vocab_size]`
    /// - `labels`: `[N]` (i32)
    /// - `grad_logits`: `[N, vocab_size]` output gradients (ignored in forward-only mode)
    /// - `loss`: scalar accumulator buffer (atomic float, must be zeroed before call)
    pub fn execute(
        &self,
        logits: &dyn AsMetalBuffer,
        labels: &dyn AsMetalBuffer,
        grad_logits: &dyn AsMetalBuffer,
        loss: &dyn AsMetalBuffer,
    ) -> Result<()> {
        let cb = self.execute_async(logits, labels, grad_logits, loss)?;
        cb.waitUntilCompleted();
        if let Some(error) = cb.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }
        Ok(())
    }

    /// Execute asynchronously, returning the committed command buffer.
    pub fn execute_async(
        &self,
        logits: &dyn AsMetalBuffer,
        labels: &dyn AsMetalBuffer,
        grad_logits: &dyn AsMetalBuffer,
        loss: &dyn AsMetalBuffer,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused Cross-Entropy not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        // NOTE: `mpp_cross_entropy_forward_f32` is currently unreachable —
        // `forward_only` defaults to false and no caller sets it to true.
        // If forward-only dispatch is ever wired up, the buffer binding below
        // must be updated: the forward-only kernel has no buffer(3) (no loss
        // accumulator); it writes per-token losses to `per_token[buffer(2)]`
        // and expects constants at buffer(4..6). Binding `loss` at index 3
        // as this function currently does would corrupt the kernel's constant
        // parameters. See the shader comment for the full explanation.
        let kernel_name = if self.config.forward_only {
            "mpp_cross_entropy_forward_f32"
        } else if self.config.use_fp16 {
            "mpp_fused_cross_entropy_fwd_bwd_f16"
        } else {
            "mpp_fused_cross_entropy_fwd_bwd_f32"
        };

        let n = self.config.num_tokens as u32;
        let vocab = self.config.vocab_size as u32;
        let ignore = self.config.ignore_index;

        // Grid: [num_tokens, 1, 1]  Threadgroup: [32, 1, 1]
        let grid = objc2_metal::MTLSize {
            width: self.config.num_tokens,
            height: 1,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize { width: 32, height: 1, depth: 1 };

        let logits_buf = logits.as_metal_buffer();
        let labels_buf = labels.as_metal_buffer();
        let grad_buf = grad_logits.as_metal_buffer();
        let loss_buf = loss.as_metal_buffer();

        encode_mpp_kernel(&self.ctx, kernel_name, grid, tg_size, |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(labels_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(grad_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(loss_buf), 0, 3);
            let n_ptr = NonNull::from(&n).cast();
            encoder.setBytes_length_atIndex(n_ptr, std::mem::size_of_val(&n), 4);
            let v_ptr = NonNull::from(&vocab).cast();
            encoder.setBytes_length_atIndex(v_ptr, std::mem::size_of_val(&vocab), 5);
            let ig_ptr = NonNull::from(&ignore).cast();
            encoder.setBytes_length_atIndex(ig_ptr, std::mem::size_of_val(&ignore), 6);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = MppFusedCrossEntropyConfig::new(128, 32000);
        assert_eq!(cfg.num_tokens, 128);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.ignore_index, -100);
        assert!(!cfg.use_fp16);
        assert!(!cfg.forward_only);
    }

    #[test]
    fn test_kernel_name_fp16() {
        let cfg = MppFusedCrossEntropyConfig::new(1, 128).with_fp16();
        assert!(cfg.use_fp16);
    }

    #[test]
    fn test_kernel_name_forward_only() {
        let cfg = MppFusedCrossEntropyConfig::new(1, 128).forward_only();
        assert!(cfg.forward_only);
    }
}
