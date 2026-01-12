//! DoRA (Weight-Decomposed Low-Rank Adaptation) implementation.
//!
//! DoRA decomposes weight updates into magnitude and direction:
//! `W_final = m * (W + scale * B @ A) / ||W + scale * B @ A||`
//!
//! Where:
//! - `W` is the frozen base weight matrix
//! - `m` is the learnable magnitude vector (initialized to ||W||)
//! - `A`, `B` are LoRA low-rank matrices
//! - `scale` is the LoRA scaling factor
//!
//! DoRA typically achieves better performance than LoRA by allowing the magnitude
//! and direction of updates to be learned separately, closer to full fine-tuning.

use mlx_rs::{error::Exception, nn, Array};
use pmetal_core::LoraConfig;
use crate::lora::LoraError;

/// DoRA Linear layer implementing weight decomposition.
#[derive(Debug)]
pub struct DoraLinear {
    /// Input features dimension.
    pub in_features: i32,
    /// Output features dimension.
    pub out_features: i32,
    /// LoRA rank.
    pub rank: i32,
    /// LoRA scaling factor.
    pub scale: f32,
    /// Whether the layer is merged.
    pub merged: bool,
    /// Whether to use bias.
    pub use_bias: bool,

    /// Frozen base weight matrix [out_features, in_features].
    pub weight: Array,
    /// Optional bias [out_features].
    pub bias: Option<Array>,
    /// LoRA A matrix (rank x in_features) - trainable.
    pub lora_a: Array,
    /// LoRA B matrix (out_features x rank) - trainable.
    pub lora_b: Array,
    /// Magnitude vector [out_features, 1] - trainable.
    pub magnitude: Array,
}

impl DoraLinear {
    /// Create a new DoRA linear layer from a base Linear layer.
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
        let scale = if use_rslora {
            alpha / (rank as f32).sqrt()
        } else {
            alpha / rank as f32
        };

        // Initialize LoRA A with Kaiming uniform
        let bound = (3.0_f32 / in_features as f32).sqrt();
        let lora_a = mlx_rs::random::uniform::<_, f32>(-bound, bound, &[rank, in_features], None)?;

        // Initialize LoRA B with zeros
        let lora_b = mlx_rs::ops::zeros::<f32>(&[out_features, rank])?;

        // Initialize magnitude with column norms of base weight
        // weight is [out, in], so we want norms along dim 1 to get [out, 1]
        // Use sum_axis(axis, keep_dims)
        let magnitude = weight.square()?.sum_axis(1, true)?.sqrt()?;

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
            magnitude,
        })
    }

    /// Create a new DoRA linear layer with given dimensions.
    pub fn new(
        in_features: i32,
        out_features: i32,
        rank: i32,
        alpha: f32,
        use_rslora: bool,
        use_bias: bool,
    ) -> Result<Self, LoraError> {
        let scale = if use_rslora {
            alpha / (rank as f32).sqrt()
        } else {
            alpha / rank as f32
        };

        // Initialize base weight with Kaiming uniform
        let bound = (3.0_f32 / in_features as f32).sqrt();
        let weight =
            mlx_rs::random::uniform::<_, f32>(-bound, bound, &[out_features, in_features], None)?;

        let bias = if use_bias {
            Some(mlx_rs::ops::zeros::<f32>(&[out_features])?)
        } else {
            None
        };

        let lora_a = mlx_rs::random::uniform::<_, f32>(-bound, bound, &[rank, in_features], None)?;
        let lora_b = mlx_rs::ops::zeros::<f32>(&[out_features, rank])?;

        // Initialize magnitude from weight norms
        let magnitude = weight.square()?.sum_axis(1, true)?.sqrt()?;

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
            magnitude,
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

    /// Forward pass through the DoRA linear layer.
    ///
    /// Computes:
    /// 1. V = W + scale * B @ A
    /// 2. V_norm = V / ||V||_c
    /// 3. W_final = m * V_norm
    /// 4. y = x @ W_final.T
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        if self.merged {
            let y = x.matmul(&self.weight.t())?;
            if let Some(ref bias) = self.bias {
                Ok(y.add(bias)?)
            } else {
                Ok(y)
            }
        } else {
            // 1. Calculate effective weight direction: V = W + scale * B @ A
            // We need full matrix reconstruction here, which is expensive but necessary for DoRA
            let ba = self.lora_b.matmul(&self.lora_a)?;
            let scale_arr = Array::from_f32(self.scale);
            let update = ba.multiply(&scale_arr)?;
            let v = self.weight.add(&update)?;

            // 2. Normalize V: V_norm = V / ||V||
            let v_norm = v.square()?.sum_axis(1, true)?.sqrt()?;
            // Add epsilon to avoid div by zero? MLX handles this reasonably well usually
            let normalized_v = v.divide(&v_norm.add(&Array::from_f32(1e-6))?)?;

            // 3. Scale by magnitude: W_final = m * V_norm
            // magnitude is [out, 1], normalized_v is [out, in] -> broadcasting works
            let w_final = normalized_v.multiply(&self.magnitude)?;

            // 4. Linear projection: y = x @ W_final.T
            let y = x.matmul(&w_final.t())?;

            if let Some(ref bias) = self.bias {
                Ok(y.add(bias)?)
            } else {
                Ok(y)
            }
        }
    }

    /// Merge DoRA weights into base weights.
    ///
    /// The merged weight is simply the final computed W used in forward pass.
    pub fn merge(&mut self) -> Result<(), LoraError> {
        if self.merged {
            return Ok(());
        }

        // Reconstruct full weight
        let ba = self.lora_b.matmul(&self.lora_a)?;
        let scale_arr = Array::from_f32(self.scale);
        let update = ba.multiply(&scale_arr)?;
        let v = self.weight.add(&update)?;

        let v_norm = v.square()?.sum_axis(1, true)?.sqrt()?;
        let normalized_v = v.divide(&v_norm.add(&Array::from_f32(1e-6))?)?;
        let w_final = normalized_v.multiply(&self.magnitude)?;

        self.weight = w_final;
        self.merged = true;
        Ok(())
    }

    /// Unmerge DoRA weights (restore base weight and params).
    ///
    /// WARNING: DoRA unmerge is non-trivial/approximate because the decomposition
    /// fuses magnitude and direction. This implementation simply resets the merged flag
    /// if the original weights were preserved or if we accept the merged state as new base.
    ///
    /// For this implementation, we follow the pattern of assuming `self.weight` holds
    /// the *base* weight when unmerged, and *final* weight when merged.
    /// HOWEVER, calculating the original W from W_final, m, A, B is complex.
    ///
    /// Current strategy: DoRA merging usually implies "baking in" for inference.
    /// Unmerging for continued training would require saving the original W.
    /// As `LoraLinear` modifies `self.weight` in place, we cannot perfectly unmerge
    /// without extra storage.
    ///
    /// Throw error for now to be safe.
    pub fn unmerge(&mut self) -> Result<(), LoraError> {
        if !self.merged {
            return Ok(());
        }
        // TODO: Store original weight for unmerging
        Err(LoraError::Mlx(Exception::custom("DoRA unmerge not yet supported without original weight storage")))
    }

    /// Get trainable parameters: LoRA A, LoRA B, and Magnitude.
    pub fn trainable_params(&self) -> Vec<&Array> {
        vec![&self.lora_a, &self.lora_b, &self.magnitude]
    }

    /// Get the number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        let lora_a_params = (self.rank * self.in_features) as usize;
        let lora_b_params = (self.out_features * self.rank) as usize;
        let magnitude_params = self.out_features as usize;
        lora_a_params + lora_b_params + magnitude_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dora_linear_initialization() {
        let dora = DoraLinear::new(64, 128, 8, 16.0, false, false).unwrap();
        
        assert_eq!(dora.in_features, 64);
        assert_eq!(dora.out_features, 128);
        assert_eq!(dora.magnitude.shape(), &[128, 1]);
        
        // Magnitude should be close to norm of initialized weights
        let w_norm = dora.weight.square().unwrap().sum_axis(1, true).unwrap().sqrt().unwrap();
        let diff = dora.magnitude.subtract(&w_norm).unwrap().abs().unwrap().sum(None).unwrap();
        assert!(diff.item::<f32>() < 1e-4);
    }

    #[test]
    fn test_dora_forward_pass() {
        let mut dora = DoraLinear::new(32, 64, 4, 8.0, false, false).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[2, 4, 32], None, None, None).unwrap();
        
        let output = dora.forward(&x).unwrap();
        assert_eq!(output.shape(), &[2, 4, 64]);
    }
}
