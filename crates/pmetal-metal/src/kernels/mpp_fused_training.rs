#![allow(unsafe_code)]

//! Metal 4 / MPP Fused AdamW optimizer dispatch.
//!
//! Provides hardware-accelerated AdamW optimizer and gradient clipping via
//! Metal Performance Primitives on M5+ (Apple10) GPUs with NAX cores.
//!
//! Dispatches element-wise AdamW over all model parameters in a single GPU
//! command buffer, eliminating per-kernel synchronization overhead that
//! plagues the Metal 3 path.
//!
//! Kernel families:
//! - `mpp_fused_adamw_f32` / `mpp_fused_adamw_f16` — optimizer update
//! - `mpp_gradient_norm_partial` + `mpp_scale_gradients` — gradient clipping
//!
//! Grid layout: `[ceil(max_param_elements / 32), num_params, 1]`
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

/// Configuration for the MPP Fused AdamW optimizer.
#[derive(Debug, Clone)]
pub struct MppFusedAdamWConfig {
    /// Total number of parameter tensors.
    pub num_params: usize,
    /// Maximum number of elements in any single parameter (sets grid width).
    pub max_param_elements: usize,
    /// Use fp16 params / fp32 moments (true) or all fp32 (false).
    pub use_fp16: bool,
    /// Current optimizer step (0 = disable bias correction, mlx-rs style).
    pub step: u32,
}

impl MppFusedAdamWConfig {
    /// Create a new AdamW config.
    pub fn new(num_params: usize, max_param_elements: usize) -> Self {
        Self {
            num_params,
            max_param_elements,
            use_fp16: false,
            step: 0,
        }
    }

    /// Enable bias-corrected AdamW (PyTorch-compatible).
    pub fn with_bias_correction(mut self, step: u32) -> Self {
        self.step = step;
        self
    }
}

/// Configuration for gradient clipping.
#[derive(Debug, Clone)]
pub struct MppGradClipConfig {
    /// Total number of gradient elements across all parameters.
    pub total_elements: usize,
    /// Use fp16 gradients.
    pub use_fp16: bool,
}

// =============================================================================
// Metal-side parameter blocks (must match Metal struct layout)
// =============================================================================

/// Must match `MppAdamWConfig` in mpp_fused_training.metal.
#[repr(C)]
struct MppAdamWConfigMetal {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step: u32,
}

/// Must match `MppParamInfo` in mpp_fused_training.metal.
#[repr(C)]
pub struct MppParamInfo {
    pub offset: u32,
    pub size: u32,
    pub m_offset: u32,
    pub v_offset: u32,
}

// =============================================================================
// AdamW dispatcher
// =============================================================================

/// MPP Fused AdamW optimizer dispatcher.
///
/// Dispatches all-parameter AdamW update to `mpp_fused_adamw_{f16,f32}` on
/// M5+ hardware, processing every parameter in a single command buffer.
pub struct MppFusedAdamW {
    ctx: Arc<MetalContext>,
    config: MppFusedAdamWConfig,
}

impl MppFusedAdamW {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppFusedAdamWConfig) -> Self {
        Self { ctx, config }
    }

    /// Returns true when MPP AdamW is available (requires M5+ NAX).
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Execute one AdamW step synchronously.
    ///
    /// - `params`: flattened parameter buffer
    /// - `grads`: flattened gradient buffer
    /// - `m`: first-moment buffer (fp32)
    /// - `v`: second-moment buffer (fp32)
    /// - `param_infos`: per-parameter metadata slice
    /// - `lr`, `beta1`, `beta2`, `eps`, `wd`: optimizer hyperparameters
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        params: &dyn AsMetalBuffer,
        grads: &dyn AsMetalBuffer,
        m: &dyn AsMetalBuffer,
        v: &dyn AsMetalBuffer,
        param_infos: &[MppParamInfo],
        param_info_buf: &dyn AsMetalBuffer,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
    ) -> Result<()> {
        let cb = self.execute_async(params, grads, m, v, param_infos, param_info_buf, lr, beta1, beta2, eps, wd)?;
        cb.waitUntilCompleted();
        if let Some(error) = cb.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }
        Ok(())
    }

    /// Execute asynchronously, returning the committed command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_async(
        &self,
        params: &dyn AsMetalBuffer,
        grads: &dyn AsMetalBuffer,
        m: &dyn AsMetalBuffer,
        v: &dyn AsMetalBuffer,
        param_infos: &[MppParamInfo],
        param_info_buf: &dyn AsMetalBuffer,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP Fused AdamW not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        assert_eq!(
            param_infos.len(),
            self.config.num_params,
            "param_infos length ({}) must match config.num_params ({})",
            param_infos.len(),
            self.config.num_params,
        );

        let kernel_name = if self.config.use_fp16 {
            "mpp_fused_adamw_f16"
        } else {
            "mpp_fused_adamw_f32"
        };

        let metal_config = MppAdamWConfigMetal {
            learning_rate: lr,
            beta1,
            beta2,
            epsilon: eps,
            weight_decay: wd,
            step: self.config.step,
        };
        let num_params = self.config.num_params as u32;

        // Grid: [ceil(max_param_elements / 32), num_params, 1]
        // Threadgroup: [32, 1, 1] — single SIMD group
        let tiles_x = self.config.max_param_elements.div_ceil(32);
        let grid = objc2_metal::MTLSize {
            width: tiles_x,
            height: self.config.num_params,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize { width: 32, height: 1, depth: 1 };

        let params_buf = params.as_metal_buffer();
        let grads_buf = grads.as_metal_buffer();
        let m_buf = m.as_metal_buffer();
        let v_buf = v.as_metal_buffer();
        let info_buf = param_info_buf.as_metal_buffer();

        encode_mpp_kernel(&self.ctx, kernel_name, grid, tg_size, |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(params_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(grads_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(m_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(v_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(info_buf), 0, 4);
            let cfg_ptr = NonNull::from(&metal_config).cast();
            encoder.setBytes_length_atIndex(cfg_ptr, std::mem::size_of_val(&metal_config), 5);
            let np_ptr = NonNull::from(&num_params).cast();
            encoder.setBytes_length_atIndex(np_ptr, std::mem::size_of_val(&num_params), 6);
        })
    }
}

// =============================================================================
// Gradient scaling dispatcher
// =============================================================================

/// MPP gradient scaling dispatcher (used after norm reduction for clipping).
pub struct MppGradScale {
    ctx: Arc<MetalContext>,
    config: MppGradClipConfig,
}

impl MppGradScale {
    /// Create a new dispatcher.
    pub fn new(ctx: Arc<MetalContext>, config: MppGradClipConfig) -> Self {
        Self { ctx, config }
    }

    /// Returns true when MPP grad scaling is available.
    pub fn is_available(&self) -> bool {
        self.ctx.properties().has_nax() && self.ctx.pipeline_cache().metal4_library().is_some()
    }

    /// Scale all gradients by `scale` in-place.
    pub fn execute(&self, grads: &dyn AsMetalBuffer, scale: f32) -> Result<()> {
        let cb = self.execute_async(grads, scale)?;
        cb.waitUntilCompleted();
        if let Some(error) = cb.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }
        Ok(())
    }

    /// Execute asynchronously.
    pub fn execute_async(
        &self,
        grads: &dyn AsMetalBuffer,
        scale: f32,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if !self.is_available() {
            return Err(MetalError::ExecutionFailed(
                "MPP grad scaling not available (requires M5+ GPU with NAX)".to_string(),
            ));
        }

        let kernel_name = if self.config.use_fp16 {
            "mpp_scale_gradients_f16"
        } else {
            "mpp_scale_gradients"
        };

        let total = self.config.total_elements as u32;

        // Each thread handles 4 elements → ceil(total/4) threads, groups of 32
        let num_threads = self.config.total_elements.div_ceil(4);
        let grid = objc2_metal::MTLSize {
            width: num_threads.div_ceil(32),
            height: 1,
            depth: 1,
        };
        let tg_size = objc2_metal::MTLSize { width: 32, height: 1, depth: 1 };

        let grads_buf = grads.as_metal_buffer();

        encode_mpp_kernel(&self.ctx, kernel_name, grid, tg_size, |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(grads_buf), 0, 0);
            let scale_ptr = NonNull::from(&scale).cast();
            encoder.setBytes_length_atIndex(scale_ptr, std::mem::size_of_val(&scale), 1);
            let total_ptr = NonNull::from(&total).cast();
            encoder.setBytes_length_atIndex(total_ptr, std::mem::size_of_val(&total), 2);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_config_defaults() {
        let cfg = MppFusedAdamWConfig::new(10, 4096);
        assert_eq!(cfg.num_params, 10);
        assert_eq!(cfg.max_param_elements, 4096);
        assert!(!cfg.use_fp16);
        assert_eq!(cfg.step, 0);
    }

    #[test]
    fn test_bias_correction_step() {
        let cfg = MppFusedAdamWConfig::new(1, 512).with_bias_correction(42);
        assert_eq!(cfg.step, 42);
    }

    #[test]
    fn test_grid_tiles_aligned() {
        // max_param_elements = 64 → 64/32 = 2 tiles
        let cfg = MppFusedAdamWConfig::new(4, 64);
        let tiles_x = cfg.max_param_elements.div_ceil(32);
        assert_eq!(tiles_x, 2);
    }

    #[test]
    fn test_grid_tiles_unaligned() {
        // max_param_elements = 33 → ceil(33/32) = 2 tiles
        let cfg = MppFusedAdamWConfig::new(1, 33);
        let tiles_x = cfg.max_param_elements.div_ceil(32);
        assert_eq!(tiles_x, 2);
    }
}
