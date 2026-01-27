//! Fused LoRA training with Metal acceleration.
//!
//! This module provides accelerated LoRA training using fused Metal kernels
//! when available. The fused kernels compute forward and backward passes
//! in a single kernel launch, providing ~2x speedup over separate operations.
//!
//! # Feature Flag
//!
//! Enable the `metal-fused` feature to use Metal acceleration:
//! ```toml
//! [dependencies]
//! pmetal-lora = { version = "*", features = ["metal-fused"] }
//! ```
//!
//! # Fallback Behavior
//!
//! When Metal is not available (or the feature is disabled), the module
//! provides a pure MLX fallback that maintains API compatibility.

use mlx_rs::{error::Exception, Array};
use tracing::{debug, warn};

#[cfg(feature = "metal-fused")]
use half::f16;
#[cfg(feature = "metal-fused")]
use std::sync::Arc;

#[cfg(feature = "metal-fused")]
use pmetal_metal::{
    bridge::metal_buffer_from_ptr, BufferUsage, FusedLora, FusedLoraConfig, FusedLoraOutput,
    MetalBuffer, MetalContext,
};

/// Error type for fused training operations.
#[derive(Debug, thiserror::Error)]
pub enum FusedTrainingError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),

    /// Metal error (when metal-fused feature is enabled).
    #[cfg(feature = "metal-fused")]
    #[error("Metal error: {0}")]
    Metal(#[from] pmetal_metal::MetalError),

    /// Shape mismatch error.
    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },
}

/// Configuration for fused LoRA training.
#[derive(Debug, Clone)]
pub struct FusedTrainingConfig {
    /// Input features dimension.
    pub in_features: usize,
    /// Output features dimension.
    pub out_features: usize,
    /// LoRA rank.
    pub rank: usize,
    /// LoRA scaling factor (alpha / rank).
    pub scale: f32,
    /// Whether to use Metal fused kernels (when available).
    pub use_metal: bool,
}

impl FusedTrainingConfig {
    /// Create a new configuration.
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32) -> Self {
        Self {
            in_features,
            out_features,
            rank,
            scale: alpha / rank as f32,
            use_metal: true, // Prefer Metal when available
        }
    }

    /// Disable Metal acceleration (use pure MLX).
    pub fn with_mlx_only(mut self) -> Self {
        self.use_metal = false;
        self
    }
}

/// Output from fused LoRA forward pass during training.
#[derive(Debug)]
pub struct FusedForwardOutput {
    /// Output tensor [batch_size, out_features].
    pub output: Array,
    /// Intermediate x @ A.T [batch_size, rank] for backward pass.
    pub intermediate: Option<Array>,
}

/// Fused LoRA trainer that handles accelerated forward/backward passes.
///
/// This struct manages the Metal context and buffers for efficient training.
/// It automatically falls back to pure MLX when Metal is not available.
pub struct FusedLoraTrainer {
    config: FusedTrainingConfig,

    #[cfg(feature = "metal-fused")]
    metal_ctx: Option<Arc<MetalContext>>,

    #[cfg(feature = "metal-fused")]
    fused_lora: Option<FusedLora>,
}

impl FusedLoraTrainer {
    /// Create a new fused LoRA trainer.
    pub fn new(config: FusedTrainingConfig) -> Result<Self, FusedTrainingError> {
        #[cfg(feature = "metal-fused")]
        {
            if config.use_metal {
                match MetalContext::global() {
                    Ok(ctx) => {
                        debug!(
                            "Initialized Metal fused LoRA trainer for {}x{} with rank {}",
                            config.in_features, config.out_features, config.rank
                        );
                        return Ok(Self {
                            config,
                            metal_ctx: Some(ctx),
                            fused_lora: None, // Created lazily per batch size
                        });
                    }
                    Err(e) => {
                        warn!("Metal context creation failed, falling back to MLX: {}", e);
                    }
                }
            }
        }

        debug!(
            "Using MLX-only fused LoRA trainer for {}x{}",
            config.in_features, config.out_features
        );

        Ok(Self {
            config,
            #[cfg(feature = "metal-fused")]
            metal_ctx: None,
            #[cfg(feature = "metal-fused")]
            fused_lora: None,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &FusedTrainingConfig {
        &self.config
    }

    /// Check if Metal acceleration is available and enabled.
    pub fn is_metal_enabled(&self) -> bool {
        #[cfg(feature = "metal-fused")]
        {
            self.metal_ctx.is_some()
        }
        #[cfg(not(feature = "metal-fused"))]
        {
            false
        }
    }

    /// Forward pass through the LoRA linear layer.
    ///
    /// Computes: `y = x @ W.T + scale * (x @ A.T) @ B.T`
    ///
    /// Also saves the intermediate `x @ A.T` for efficient backward pass.
    #[allow(unsafe_code)]
    pub fn forward(
        &self,
        x: &Array,
        weight: &Array,
        lora_a: &Array,
        lora_b: &Array,
    ) -> Result<FusedForwardOutput, FusedTrainingError> {
        #[cfg(feature = "metal-fused")]
        if let Some(ctx) = &self.metal_ctx {
            // Get shapes
            let batch_size = x.dim(0) as usize;
            // Support flattened input [batch * seq, hidden]

            // Create config for this batch size
            let config = FusedLoraConfig::new(
                batch_size,
                self.config.in_features,
                self.config.out_features,
                self.config.rank,
                self.config.scale,
            );

            // Create executor (pipelines are cached in ctx)
            let fused =
                FusedLora::new(ctx.clone(), config).map_err(pmetal_metal::MetalError::from)?;

            // Create views (unsafe - assuming Array memory is valid/unified)
            // Note: We need to ensure arrays are evaluated and contiguous
            x.eval()?;
            weight.eval()?;
            lora_a.eval()?;
            lora_b.eval()?;

            // Get pointers (via as_slice().as_ptr() - safe as long as Array lives)
            // MLX arrays are usually contiguous.
            unsafe {
                let x_view =
                    metal_buffer_from_ptr(ctx, x.as_slice::<f16>().as_ptr() as *mut f16, x.size())?;
                let w_view = metal_buffer_from_ptr(
                    ctx,
                    weight.as_slice::<f16>().as_ptr() as *mut f16,
                    weight.size(),
                )?;
                let a_view = metal_buffer_from_ptr(
                    ctx,
                    lora_a.as_slice::<f16>().as_ptr() as *mut f16,
                    lora_a.size(),
                )?;
                let b_view = metal_buffer_from_ptr(
                    ctx,
                    lora_b.as_slice::<f16>().as_ptr() as *mut f16,
                    lora_b.size(),
                )?;

                let output = fused
                    .forward(&x_view, &w_view, &a_view, &b_view)
                    .map_err(pmetal_metal::MetalError::from)?;

                // Convert back to Array (copying for now for safety)
                let out_vec = output.output.to_vec()?;
                let out_arr = Array::from_slice(
                    &out_vec,
                    &[batch_size as i32, self.config.out_features as i32],
                );

                let inter_vec = output.intermediate.as_ref().unwrap().to_vec()?;
                let inter_arr =
                    Array::from_slice(&inter_vec, &[batch_size as i32, self.config.rank as i32]);

                return Ok(FusedForwardOutput {
                    output: out_arr,
                    intermediate: Some(inter_arr),
                });
            }
        }

        self.forward_mlx(x, weight, lora_a, lora_b)
    }

    /// Pure MLX forward pass.
    fn forward_mlx(
        &self,
        x: &Array,
        weight: &Array,
        lora_a: &Array,
        lora_b: &Array,
    ) -> Result<FusedForwardOutput, FusedTrainingError> {
        // Base forward: y_base = x @ W.T
        let y_base = x.matmul(&weight.t())?;

        // LoRA forward: y_lora = scale * (x @ A.T) @ B.T
        let xa = x.matmul(&lora_a.t())?;
        let xab = xa.matmul(&lora_b.t())?;
        let scale_arr = Array::from_f32(self.config.scale);
        let y_lora = xab.multiply(&scale_arr)?;

        // Combined output
        let output = y_base.add(&y_lora)?;

        Ok(FusedForwardOutput {
            output,
            intermediate: Some(xa), // Save for backward
        })
    }

    /// Backward pass to compute gradients for LoRA parameters.
    ///
    /// Computes:
    /// - `dA = scale * (dY @ B).T @ x` -> [rank, in_features]
    /// - `dB = scale * dY.T @ (x @ A.T)` -> [out_features, rank]
    ///
    /// Uses the saved intermediate from forward pass for efficiency.
    ///
    /// # Shape derivation
    ///
    /// For y = scale * (x @ A.T) @ B.T:
    /// - x: [batch, in_features]
    /// - A: [rank, in_features] -> A.T: [in_features, rank]
    /// - B: [out_features, rank] -> B.T: [rank, out_features]
    /// - intermediate = x @ A.T: [batch, rank]
    /// - y: [batch, out_features]
    ///
    /// Gradients (using matrix calculus):
    /// - dB = dY.T @ intermediate: [out_features, batch] @ [batch, rank] = [out_features, rank]
    /// - dA = (dY @ B).T @ x: [rank, batch] @ [batch, in_features] = [rank, in_features]
    #[allow(unsafe_code)]
    pub fn backward_lora(
        &self,
        grad_output: &Array,
        x: &Array,
        intermediate: &Array,
        lora_b: &Array,
    ) -> Result<(Array, Array), FusedTrainingError> {
        #[cfg(feature = "metal-fused")]
        if let Some(ctx) = &self.metal_ctx {
            let batch_size = x.dim(0) as usize;
            let config = FusedLoraConfig::new(
                batch_size,
                self.config.in_features,
                self.config.out_features,
                self.config.rank,
                self.config.scale,
            );
            let fused =
                FusedLora::new(ctx.clone(), config).map_err(pmetal_metal::MetalError::from)?;

            grad_output.eval()?;
            x.eval()?;
            intermediate.eval()?;
            lora_b.eval()?;

            unsafe {
                let dy_view = metal_buffer_from_ptr(
                    ctx,
                    grad_output.as_slice::<f16>().as_ptr() as *mut f16,
                    grad_output.size(),
                )?;
                let x_view =
                    metal_buffer_from_ptr(ctx, x.as_slice::<f16>().as_ptr() as *mut f16, x.size())?;
                let inter_view = metal_buffer_from_ptr(
                    ctx,
                    intermediate.as_slice::<f16>().as_ptr() as *mut f16,
                    intermediate.size(),
                )?;
                let b_view = metal_buffer_from_ptr(
                    ctx,
                    lora_b.as_slice::<f16>().as_ptr() as *mut f16,
                    lora_b.size(),
                )?;

                let (grad_a_buf, grad_b_buf) = fused
                    .backward_ab(&dy_view, &x_view, &inter_view, &b_view)
                    .map_err(pmetal_metal::MetalError::from)?;

                let grad_a = Array::from_slice(
                    &grad_a_buf.to_vec()?,
                    &[self.config.rank as i32, self.config.in_features as i32],
                );
                let grad_b = Array::from_slice(
                    &grad_b_buf.to_vec()?,
                    &[self.config.out_features as i32, self.config.rank as i32],
                );

                return Ok((grad_a, grad_b));
            }
        }

        let scale_arr = Array::from_f32(self.config.scale);

        // dB = scale * dY.T @ xA
        // dY: [batch, out_features], xA (intermediate): [batch, rank]
        // dY.T @ xA: [out_features, batch] @ [batch, rank] = [out_features, rank]
        let grad_b = grad_output.t().matmul(intermediate)?;
        let grad_b = grad_b.multiply(&scale_arr)?;

        // dA = scale * (dY @ B).T @ x
        // dY @ B: [batch, out_features] @ [out_features, rank] = [batch, rank]
        // (dY @ B).T @ x: [rank, batch] @ [batch, in_features] = [rank, in_features]
        let dy_b = grad_output.matmul(lora_b)?;
        let grad_a = dy_b.t().matmul(x)?;
        let grad_a = grad_a.multiply(&scale_arr)?;

        Ok((grad_a, grad_b))
    }

    /// Backward pass to compute input gradient.
    ///
    /// Computes: `dX = dY @ W + scale * (dY @ B) @ A`
    #[allow(unsafe_code)]
    pub fn backward_input(
        &self,
        grad_output: &Array,
        weight: &Array,
        lora_a: &Array,
        lora_b: &Array,
    ) -> Result<Array, FusedTrainingError> {
        #[cfg(feature = "metal-fused")]
        if let Some(ctx) = &self.metal_ctx {
            let batch_size = grad_output.dim(0) as usize;
            let config = FusedLoraConfig::new(
                batch_size,
                self.config.in_features,
                self.config.out_features,
                self.config.rank,
                self.config.scale,
            );
            let fused =
                FusedLora::new(ctx.clone(), config).map_err(pmetal_metal::MetalError::from)?;

            grad_output.eval()?;
            weight.eval()?;
            lora_a.eval()?;
            lora_b.eval()?;

            unsafe {
                let dy_view = metal_buffer_from_ptr(
                    ctx,
                    grad_output.as_slice::<f16>().as_ptr() as *mut f16,
                    grad_output.size(),
                )?;
                let w_view = metal_buffer_from_ptr(
                    ctx,
                    weight.as_slice::<f16>().as_ptr() as *mut f16,
                    weight.size(),
                )?;
                let a_view = metal_buffer_from_ptr(
                    ctx,
                    lora_a.as_slice::<f16>().as_ptr() as *mut f16,
                    lora_a.size(),
                )?;
                let b_view = metal_buffer_from_ptr(
                    ctx,
                    lora_b.as_slice::<f16>().as_ptr() as *mut f16,
                    lora_b.size(),
                )?;

                let grad_x_buf = fused
                    .backward_x(&dy_view, &w_view, &a_view, &b_view)
                    .map_err(pmetal_metal::MetalError::from)?;

                let grad_x = Array::from_slice(
                    &grad_x_buf.to_vec()?,
                    &[batch_size as i32, self.config.in_features as i32],
                );
                return Ok(grad_x);
            }
        }

        // dX from base weights
        let dx_base = grad_output.matmul(weight)?;

        // dX from LoRA
        let dy_b = grad_output.matmul(lora_b)?;
        let dy_ba = dy_b.matmul(lora_a)?;
        let scale_arr = Array::from_f32(self.config.scale);
        let dx_lora = dy_ba.multiply(&scale_arr)?;

        // Combined gradient
        Ok(dx_base.add(&dx_lora)?)
    }
}

impl std::fmt::Debug for FusedLoraTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedLoraTrainer")
            .field("config", &self.config)
            .field("metal_enabled", &self.is_metal_enabled())
            .finish()
    }
}

// =============================================================================
// FUSED LINEAR + CROSS-ENTROPY (UNSLOTH'S SECRET SAUCE)
// =============================================================================
//
// This is the key memory optimization from unsloth: compute cross-entropy loss
// directly from hidden states, without materializing the full logits tensor.
//
// Memory savings example:
// - batch=4, seq=1024, vocab=150K, fp16 → logits would be 1.2GB
// - With fusion: peak memory is only chunk_size * hidden_size ≈ 8MB
//
// Implementation options:
//
// 1. **Metal kernel** (fastest, for inference/non-gradient use):
//    `pmetal_metal::FusedLinearCrossEntropy`
//    - Single GPU kernel, maximum efficiency
//    - ~37x memory reduction
//
// 2. **MLX-based** (for training with autodiff):
//    Use standard `model.forward()` + `cross_entropy_loss()`, which already
//    uses chunked logsumexp for large vocabularies (>65K tokens).
//    See: `pmetal_mlx::kernels::cross_entropy::cross_entropy_loss`
//
// The chunked approach in cross_entropy.rs provides:
// - Stable logsumexp computation
// - Chunked processing for vocab > 65536
// - Proper ignore_index handling
// - Label smoothing support
// =============================================================================

/// Configuration for fused linear + cross-entropy loss.
///
/// This config can be used with `pmetal_metal::FusedLinearCrossEntropy`
/// for maximum performance when gradients are not needed.
#[derive(Debug, Clone)]
pub struct FusedLinearCrossEntropyConfig {
    /// Vocabulary chunk size (default: 4096).
    /// Larger = faster but more memory. Smaller = less memory but slower.
    pub chunk_size: usize,

    /// Index to ignore in loss computation (typically -100).
    pub ignore_index: i32,

    /// Label smoothing factor (0.0 to disable).
    pub label_smoothing: f32,
}

impl Default for FusedLinearCrossEntropyConfig {
    fn default() -> Self {
        Self {
            chunk_size: 4096,
            ignore_index: -100,
            label_smoothing: 0.0,
        }
    }
}

impl FusedLinearCrossEntropyConfig {
    /// Create config with custom chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set ignore index.
    pub fn with_ignore_index(mut self, index: i32) -> Self {
        self.ignore_index = index;
        self
    }

    /// Enable label smoothing.
    pub fn with_label_smoothing(mut self, smoothing: f32) -> Self {
        self.label_smoothing = smoothing;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_training_config() {
        let config = FusedTrainingConfig::new(512, 1024, 8, 16.0);

        assert_eq!(config.in_features, 512);
        assert_eq!(config.out_features, 1024);
        assert_eq!(config.rank, 8);
        assert!((config.scale - 2.0).abs() < 1e-6); // 16 / 8 = 2
        assert!(config.use_metal);

        let config_mlx = config.with_mlx_only();
        assert!(!config_mlx.use_metal);
    }

    #[test]
    fn test_fused_trainer_creation() {
        let config = FusedTrainingConfig::new(128, 256, 4, 8.0);
        let trainer = FusedLoraTrainer::new(config).unwrap();

        assert_eq!(trainer.config().in_features, 128);
        assert_eq!(trainer.config().out_features, 256);
    }

    #[test]
    fn test_fused_forward_pass() {
        let config = FusedTrainingConfig::new(32, 64, 4, 8.0);
        let trainer = FusedLoraTrainer::new(config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[2, 8, 32], None, None, None).unwrap();
        let weight = mlx_rs::random::normal::<f32>(&[64, 32], None, None, None).unwrap();
        let lora_a = mlx_rs::random::normal::<f32>(&[4, 32], None, None, None).unwrap();
        let lora_b = mlx_rs::ops::zeros::<f32>(&[64, 4]).unwrap();

        let output = trainer.forward(&x, &weight, &lora_a, &lora_b).unwrap();

        assert_eq!(output.output.shape(), &[2, 8, 64]);
        assert!(output.intermediate.is_some());
        assert_eq!(output.intermediate.unwrap().shape(), &[2, 8, 4]);
    }

    #[test]
    fn test_fused_backward_lora() {
        let config = FusedTrainingConfig::new(32, 64, 4, 8.0);
        let trainer = FusedLoraTrainer::new(config).unwrap();

        let batch_size = 16;
        let x = mlx_rs::random::normal::<f32>(&[batch_size, 32], None, None, None).unwrap();
        let weight = mlx_rs::random::normal::<f32>(&[64, 32], None, None, None).unwrap();
        let lora_a = mlx_rs::random::normal::<f32>(&[4, 32], None, None, None).unwrap();
        let lora_b = mlx_rs::random::normal::<f32>(&[64, 4], None, None, None).unwrap();

        // Forward
        let output = trainer.forward(&x, &weight, &lora_a, &lora_b).unwrap();

        // Fake gradient
        let grad_output =
            mlx_rs::random::normal::<f32>(&[batch_size, 64], None, None, None).unwrap();

        // Backward for LoRA params
        let (grad_a, grad_b) = trainer
            .backward_lora(
                &grad_output,
                &x,
                output.intermediate.as_ref().unwrap(),
                &lora_b,
            )
            .unwrap();

        assert_eq!(grad_a.shape(), lora_a.shape());
        assert_eq!(grad_b.shape(), lora_b.shape());
    }

    #[test]
    fn test_fused_backward_input() {
        let config = FusedTrainingConfig::new(32, 64, 4, 8.0);
        let trainer = FusedLoraTrainer::new(config).unwrap();

        let batch_size = 16;
        let weight = mlx_rs::random::normal::<f32>(&[64, 32], None, None, None).unwrap();
        let lora_a = mlx_rs::random::normal::<f32>(&[4, 32], None, None, None).unwrap();
        let lora_b = mlx_rs::random::normal::<f32>(&[64, 4], None, None, None).unwrap();
        let grad_output =
            mlx_rs::random::normal::<f32>(&[batch_size, 64], None, None, None).unwrap();

        let grad_x = trainer
            .backward_input(&grad_output, &weight, &lora_a, &lora_b)
            .unwrap();

        assert_eq!(grad_x.shape(), &[batch_size, 32]);
    }
}
