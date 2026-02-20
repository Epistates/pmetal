//! Standard LoRA (Low-Rank Adaptation) implementation.
//!
//! LoRA adds low-rank trainable matrices to frozen pretrained weights:
//! `y = x @ W.T + scale * (x @ A.T) @ B.T`
//!
//! Where:
//! - `W` is the frozen base weight matrix
//! - `A` is the LoRA down-projection matrix (rank x in_features)
//! - `B` is the LoRA up-projection matrix (out_features x rank)
//! - `scale = alpha / rank` (or `alpha / sqrt(rank)` for RSLoRA)

use mlx_rs::{error::Exception, nn, Array};

use pmetal_core::LoraConfig;

/// Error type for LoRA operations.
#[derive(Debug, thiserror::Error)]
pub enum LoraError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] mlx_rs::error::IoError),
    /// Shape mismatch error.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    /// Invalid state error.
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// LoRA Linear layer that wraps a base Linear layer with low-rank adaptation.
///
/// Implements: `y = x @ W.T + scale * (x @ A.T) @ B.T`
#[derive(Debug)]
pub struct LoraLinear {
    /// Input features dimension.
    pub in_features: i32,
    /// Output features dimension.
    pub out_features: i32,
    /// LoRA rank.
    pub rank: i32,
    /// LoRA scaling factor (alpha / rank).
    pub scale: f32,
    /// Whether the layer is merged.
    pub merged: bool,
    /// Whether to use bias.
    pub use_bias: bool,

    /// Frozen base weight matrix [out_features, in_features].
    pub(crate) weight: Array,
    /// Optional bias [out_features].
    pub(crate) bias: Option<Array>,
    /// LoRA A matrix (rank x in_features) - trainable.
    pub(crate) lora_a: Array,
    /// LoRA B matrix (out_features x rank) - trainable.
    pub(crate) lora_b: Array,
}

impl LoraLinear {
    /// Create a new LoRA linear layer from a base Linear layer.
    pub fn from_linear(
        linear: &nn::Linear,
        rank: i32,
        alpha: f32,
        use_rslora: bool,
    ) -> Result<Self, LoraError> {
        let weight = linear.weight.value.as_ref();
        let in_features = weight.dim(-1);
        let out_features = weight.dim(-2);

        // Compute scaling factor
        let scale = if rank > 0 {
            if use_rslora {
                alpha / (rank as f32).sqrt()
            } else {
                alpha / rank as f32
            }
        } else {
            0.0
        };

        // Initialize LoRA A.
        // - Standard LoRA: Kaiming uniform with bound = sqrt(3 / in_features).
        // - rsLoRA: uses bound = sqrt(1 / rank) so that the initial A norm is
        //   rank-independent, matching the rsLoRA paper (Kalajdzievski 2023).
        //   The scale factor (alpha / sqrt(rank)) then keeps the effective
        //   contribution stable across different ranks.
        let lora_a = if rank > 0 {
            let bound = if use_rslora {
                (1.0_f32 / rank as f32).sqrt()
            } else {
                (3.0_f32 / in_features as f32).sqrt()
            };
            mlx_rs::random::uniform::<_, f32>(-bound, bound, &[rank, in_features], None)?
        } else {
            mlx_rs::ops::zeros::<f32>(&[1, in_features])? // Dummy small array
        };

        // Initialize LoRA B with zeros
        let lora_b = if rank > 0 {
            mlx_rs::ops::zeros::<f32>(&[out_features, rank])?
        } else {
            mlx_rs::ops::zeros::<f32>(&[out_features, 1])? // Dummy
        };

        // Clone bias if present
        let bias = linear.bias.value.as_ref().cloned();

        Ok(Self {
            in_features,
            out_features,
            rank,
            scale,
            merged: false,
            use_bias: bias.is_some(),
            weight: weight.clone(),
            bias,
            lora_a,
            lora_b,
        })
    }

    /// Create a new LoRA linear layer with given dimensions.
    pub fn new(
        in_features: i32,
        out_features: i32,
        rank: i32,
        alpha: f32,
        use_rslora: bool,
        use_bias: bool,
    ) -> Result<Self, LoraError> {
        // Compute scaling factor
        let scale = if rank > 0 {
            if use_rslora {
                alpha / (rank as f32).sqrt()
            } else {
                alpha / rank as f32
            }
        } else {
            0.0
        };

        // Initialize base weight with Kaiming uniform
        let bound = (3.0_f32 / in_features as f32).sqrt();
        let weight =
            mlx_rs::random::uniform::<_, f32>(-bound, bound, &[out_features, in_features], None)?;

        // Initialize bias if needed
        let bias = if use_bias {
            Some(mlx_rs::ops::zeros::<f32>(&[out_features])?)
        } else {
            None
        };

        // Initialize LoRA A.
        // - Standard LoRA: reuse the same Kaiming bound as the base weight.
        // - rsLoRA: use bound = sqrt(1 / rank) (rank-dependent), matching the
        //   rsLoRA paper (Kalajdzievski 2023) so the norm is rank-independent.
        let lora_a_bound = if use_rslora {
            (1.0_f32 / rank as f32).sqrt()
        } else {
            bound
        };
        let lora_a = if rank > 0 {
            mlx_rs::random::uniform::<_, f32>(-lora_a_bound, lora_a_bound, &[rank, in_features], None)?
        } else {
            mlx_rs::ops::zeros::<f32>(&[1, in_features])?
        };

        // Initialize LoRA B with zeros
        let lora_b = if rank > 0 {
            mlx_rs::ops::zeros::<f32>(&[out_features, rank])?
        } else {
            mlx_rs::ops::zeros::<f32>(&[out_features, 1])?
        };

        Ok(Self {
            in_features,
            out_features,
            rank,
            scale,
            merged: false,
            use_bias,
            weight,
            bias,
            lora_a,
            lora_b,
        })
    }

    /// Create from LoraConfig.
    pub fn from_config(
        in_features: i32,
        out_features: i32,
        config: &LoraConfig,
        use_bias: bool,
    ) -> Result<Self, LoraError> {
        Self::new(
            in_features,
            out_features,
            config.r as i32,
            config.alpha,
            config.use_rslora,
            use_bias,
        )
    }

    /// Forward pass through the LoRA linear layer.
    ///
    /// If merged, uses merged weights. Otherwise computes:
    /// `y = x @ W.T + scale * (x @ A.T) @ B.T`
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        if self.merged || self.rank == 0 {
            // Use base weight directly if merged or rank=0 (frozen)
            let y = x.matmul(&self.weight.t())?;
            if let Some(ref bias) = self.bias {
                Ok(y.add(bias)?)
            } else {
                Ok(y)
            }
        } else {
            // Standard forward: y_base = x @ W.T
            let y_base = x.matmul(&self.weight.t())?;

            // LoRA forward: y_lora = scale * (x @ A.T) @ B.T
            let xa = x.matmul(&self.lora_a.t())?;
            let xab = xa.matmul(&self.lora_b.t())?;
            let scale_arr = Array::from_f32(self.scale);
            let y_lora = xab.multiply(&scale_arr)?;

            // Combined output
            let y = y_base.add(&y_lora)?;

            // Add bias if present
            if let Some(ref bias) = self.bias {
                Ok(y.add(bias)?)
            } else {
                Ok(y)
            }
        }
    }

    /// Forward pass with gradient context for custom autograd.
    ///
    /// This is the unsloth-style custom autograd forward that saves minimal state:
    /// - `x`: Input tensor (for dA computation)
    /// - `x @ A^T`: Intermediate (for dB computation)
    ///
    /// Use this with `backward_with_saved()` for ~50% memory reduction vs standard autodiff.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, ..., in_features]
    /// * `ctx` - Gradient context controlling which gradients to compute
    ///
    /// # Returns
    /// Tuple of (output, saved state for backward)
    pub fn forward_with_grad(
        &self,
        x: &Array,
        ctx: &crate::autograd::LoraGradContext,
    ) -> Result<(Array, crate::autograd::LoraForwardSaved), LoraError> {
        crate::autograd::lora_forward_with_grad(
            x,
            &self.weight,
            &self.lora_a,
            &self.lora_b,
            self.scale,
            ctx,
        )
        .map_err(LoraError::from)
    }

    /// Backward pass using saved state from forward_with_grad.
    ///
    /// Computes gradients for LoRA parameters (and optionally input) using
    /// explicit gradient formulas:
    /// - `dB = scale * (x @ A)^T @ dY`
    /// - `dA = scale * x^T @ (dY @ B^T)`
    /// - `dX = dY @ W + scale * (dY @ B) @ A`
    ///
    /// # Arguments
    /// * `d_output` - Upstream gradient [batch, ..., out_features]
    /// * `saved` - Saved state from forward_with_grad
    ///
    /// # Returns
    /// LoRA gradients (dA, dB, and optionally dX for chain rule)
    pub fn backward_with_saved(
        &self,
        d_output: &Array,
        saved: &crate::autograd::LoraForwardSaved,
    ) -> Result<crate::autograd::LoraGrads, LoraError> {
        crate::autograd::lora_backward(d_output, saved).map_err(LoraError::from)
    }

    /// Apply computed gradients to LoRA parameters using SGD.
    ///
    /// This is a simple gradient descent update:
    /// - `lora_a -= lr * d_lora_a`
    /// - `lora_b -= lr * d_lora_b`
    ///
    /// For more sophisticated optimizers (AdamW, etc.), use the optimizer's update method
    /// with the gradient HashMap from AccumulatedLoraGrads.
    pub fn apply_grads_sgd(
        &mut self,
        grads: &crate::autograd::LoraGrads,
        learning_rate: f32,
    ) -> Result<(), LoraError> {
        let lr = Array::from_f32(learning_rate);
        self.lora_a = self.lora_a.subtract(&grads.d_lora_a.multiply(&lr)?)?;
        self.lora_b = self.lora_b.subtract(&grads.d_lora_b.multiply(&lr)?)?;
        Ok(())
    }

    /// Merge LoRA weights into base weights.
    ///
    /// After merging: `W_merged = W + scale * B @ A`
    ///
    /// Note: this operation is irreversible. The original base weight is lost.
    /// To restore the base weights, reload the model checkpoint.
    pub fn merge(&mut self) -> Result<(), LoraError> {
        if self.merged {
            return Ok(());
        }

        // W_merged = W + scale * B @ A
        let ba = self.lora_b.matmul(&self.lora_a)?;
        let scale_arr = Array::from_f32(self.scale);
        let delta = ba.multiply(&scale_arr)?;
        let merged_weight = self.weight.add(&delta)?;

        self.weight = merged_weight;
        self.merged = true;
        Ok(())
    }

    /// Unmerge is not supported.
    ///
    /// LoRA weights cannot be reliably unmerged once merged because the original
    /// base weight is overwritten. To "unmerge", reload the base model weights
    /// and re-apply the LoRA adapter.
    // NOTE: unmerge() is intentionally omitted. The original base weight is
    // lost after merge(); callers must reload base weights to undo a merge.


    /// Get the LoRA A parameters (for gradient computation).
    pub fn lora_a_params(&self) -> &Array {
        &self.lora_a
    }

    /// Get the LoRA B parameters (for gradient computation).
    pub fn lora_b_params(&self) -> &Array {
        &self.lora_b
    }

    /// Set the LoRA A parameters.
    pub fn set_lora_a(&mut self, a: Array) {
        self.lora_a = a;
    }

    /// Set the LoRA B parameters.
    pub fn set_lora_b(&mut self, b: Array) {
        self.lora_b = b;
    }

    /// Get the number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        let lora_a_params = (self.rank * self.in_features) as usize;
        let lora_b_params = (self.out_features * self.rank) as usize;
        lora_a_params + lora_b_params
    }

    /// Get the number of frozen parameters.
    pub fn num_frozen_params(&self) -> usize {
        let weight_params = (self.out_features * self.in_features) as usize;
        let bias_params = if self.use_bias {
            self.out_features as usize
        } else {
            0
        };
        weight_params + bias_params
    }

    /// Get the compression ratio (trainable / frozen).
    pub fn compression_ratio(&self) -> f32 {
        let trainable = self.num_trainable_params() as f32;
        let frozen = self.num_frozen_params() as f32;
        trainable / frozen
    }
}

/// Per-layer LoRA configuration derived from `LoraConfig`.
///
/// Wraps `LoraConfig` rather than duplicating its fields, ensuring
/// that model patching code always references a single source of truth.
#[derive(Debug, Clone)]
pub struct LoraLayerConfig {
    /// Underlying core config this layer config is derived from.
    config: LoraConfig,
}

impl LoraLayerConfig {
    /// Create from core `LoraConfig`.
    pub fn from_core(config: &LoraConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// LoRA rank.
    pub fn rank(&self) -> i32 {
        self.config.r as i32
    }

    /// LoRA alpha.
    pub fn alpha(&self) -> f32 {
        self.config.alpha
    }

    /// Whether to use RSLoRA scaling.
    pub fn use_rslora(&self) -> bool {
        self.config.use_rslora
    }

    /// Dropout rate.
    pub fn dropout(&self) -> f32 {
        self.config.dropout
    }

    /// Compute the LoRA scaling factor (delegates to `LoraConfig::scaling()`).
    pub fn scale(&self) -> f32 {
        self.config.scaling()
    }

    /// Access the underlying `LoraConfig`.
    pub fn as_core(&self) -> &LoraConfig {
        &self.config
    }
}

impl Default for LoraLayerConfig {
    fn default() -> Self {
        Self {
            config: LoraConfig::default(),
        }
    }
}

impl From<LoraConfig> for LoraLayerConfig {
    fn from(config: LoraConfig) -> Self {
        Self { config }
    }
}

impl From<&LoraConfig> for LoraLayerConfig {
    fn from(config: &LoraConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

/// Compute fused LoRA forward pass (functional version).
///
/// Implements: y = x @ W.T + scale * (x @ A.T) @ B.T
///
/// # Arguments
/// * `x` - Input tensor of shape [..., in_features]
/// * `weight` - Base weight matrix of shape [out_features, in_features]
/// * `lora_a` - LoRA A matrix of shape [rank, in_features]
/// * `lora_b` - LoRA B matrix of shape [out_features, rank]
/// * `scale` - LoRA scaling factor (typically alpha / rank)
///
/// # Returns
/// Output tensor of shape [..., out_features]
pub fn fused_lora_forward(
    x: &Array,
    weight: &Array,
    lora_a: &Array,
    lora_b: &Array,
    scale: f32,
) -> Result<Array, LoraError> {
    // Base forward: y_base = x @ W.T
    let y_base = x.matmul(&weight.t())?;

    // LoRA forward: y_lora = scale * (x @ A.T) @ B.T
    let xa = x.matmul(&lora_a.t())?;
    let xab = xa.matmul(&lora_b.t())?;
    let scale_arr = Array::from_f32(scale);
    let y_lora = xab.multiply(&scale_arr)?;

    // Combined output
    Ok(y_base.add(&y_lora)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_linear_new() {
        let lora = LoraLinear::new(64, 128, 8, 16.0, false, false).unwrap();

        assert_eq!(lora.in_features, 64);
        assert_eq!(lora.out_features, 128);
        assert_eq!(lora.rank, 8);
        assert!((lora.scale - 2.0).abs() < 1e-6); // alpha / rank = 16 / 8 = 2
        assert!(!lora.merged);
    }

    #[test]
    fn test_lora_linear_forward() {
        let mut lora = LoraLinear::new(32, 64, 4, 8.0, false, false).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[2, 4, 32], None, None, None).unwrap();
        let output = lora.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 4, 64]);
    }

    #[test]
    fn test_lora_linear_with_bias() {
        let mut lora = LoraLinear::new(32, 64, 4, 8.0, false, true).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[2, 4, 32], None, None, None).unwrap();
        let output = lora.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 4, 64]);
        assert!(lora.bias.is_some());
    }

    #[test]
    fn test_lora_zero_contribution_initial() {
        // With B initialized to zeros, LoRA should have minimal effect initially
        let mut lora = LoraLinear::new(32, 64, 8, 16.0, false, false).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 32], None, None, None).unwrap();
        let output = lora.forward(&x).unwrap();

        // Check base forward without LoRA
        let base_output = x.matmul(&lora.weight.t()).unwrap();

        output.eval().unwrap();
        base_output.eval().unwrap();

        // Outputs should be close since B is zeros
        let diff = output.subtract(&base_output).unwrap();
        let max_diff = diff.abs().unwrap().max(None).unwrap();
        max_diff.eval().unwrap();
        assert!(max_diff.item::<f32>() < 1e-5);
    }

    #[test]
    fn test_lora_merge() {
        let mut lora = LoraLinear::new(32, 64, 4, 8.0, false, false).unwrap();

        // Initialize B to non-zero for merge to have effect
        lora.lora_b = mlx_rs::random::normal::<f32>(&[64, 4], None, None, None).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 32], None, None, None).unwrap();

        // Get output before merge
        let output_before = lora.forward(&x).unwrap();
        output_before.eval().unwrap();

        // Merge
        lora.merge().unwrap();
        assert!(lora.merged);

        // Get output after merge
        let output_after = lora.forward(&x).unwrap();
        output_after.eval().unwrap();

        // Outputs should be close (merge is numerically equivalent to unmerged forward)
        let diff = output_before.subtract(&output_after).unwrap();
        let max_diff = diff.abs().unwrap().max(None).unwrap();
        max_diff.eval().unwrap();
        assert!(max_diff.item::<f32>() < 1e-4);
    }

    #[test]
    fn test_lora_rslora_scaling() {
        let lora_regular = LoraLinear::new(64, 128, 16, 32.0, false, false).unwrap();
        let lora_rs = LoraLinear::new(64, 128, 16, 32.0, true, false).unwrap();

        // Regular: scale = alpha / rank = 32 / 16 = 2.0
        assert!((lora_regular.scale - 2.0).abs() < 1e-6);

        // RSLoRA: scale = alpha / sqrt(rank) = 32 / 4 = 8.0
        assert!((lora_rs.scale - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_lora_param_count() {
        let lora = LoraLinear::new(512, 1024, 16, 32.0, false, false).unwrap();

        // Trainable: A (16 * 512) + B (1024 * 16) = 8192 + 16384 = 24576
        assert_eq!(lora.num_trainable_params(), 24576);

        // Frozen: W (1024 * 512) = 524288
        assert_eq!(lora.num_frozen_params(), 524288);

        // Compression: 24576 / 524288 ≈ 0.0469
        assert!((lora.compression_ratio() - 0.046875).abs() < 1e-6);
    }

    #[test]
    fn test_fused_lora_forward() {
        let in_features = 32;
        let out_features = 64;
        let rank = 8;
        let scale = 2.0;

        // Create random weights
        let x = mlx_rs::random::normal::<f32>(&[2, 4, in_features], None, None, None).unwrap();
        let weight =
            mlx_rs::random::normal::<f32>(&[out_features, in_features], None, None, None).unwrap();
        let lora_a = mlx_rs::random::normal::<f32>(&[rank, in_features], None, None, None).unwrap();
        let lora_b = mlx_rs::ops::zeros::<f32>(&[out_features, rank]).unwrap();

        let output = fused_lora_forward(&x, &weight, &lora_a, &lora_b, scale).unwrap();

        assert_eq!(output.shape(), &[2, 4, out_features]);
    }
}
