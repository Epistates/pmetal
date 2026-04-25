//! Rotary Position Embedding (RoPE) with extended context support.
//!
//! Provides additional utilities for rotary embeddings, including:
//!
//! - **Linear Scaling**: Simple position scaling for 2-4x context extension
//! - **Dynamic NTK**: Neural Tangent Kernel-aware scaling for better quality
//! - **YaRN**: Yet another RoPE extensioN for 8-128x context extension
//!
//! ## Context Extension Methods
//!
//! | Method | Extension | Quality | Use Case |
//! |--------|-----------|---------|----------|
//! | Linear | 2-4x | Good | Simple extension |
//! | NTK | 2-8x | Better | Balance of quality/extension |
//! | YaRN | 8-128x | Best | Long context (Code Llama, etc.) |
//!
//! ## YaRN Theory
//!
//! YaRN divides RoPE dimensions into three groups:
//! 1. **Low frequency** (λ > original_max_pos): No interpolation needed
//! 2. **Medium frequency**: Smooth interpolation via ramp function
//! 3. **High frequency** (λ < original_max_pos / factor): Full interpolation
//!
//! This preserves high-frequency positional information while extending
//! the effective context length.

use pmetal_bridge::compat::{Array, Dtype, Exception, fast, ops};

/// RoPE scaling type for extended context.
#[derive(Debug, Clone)]
pub enum RopeScaling {
    /// No scaling.
    None,
    /// Linear scaling with given factor.
    /// Positions are divided by factor: pos' = pos / factor
    Linear {
        /// Scaling factor (e.g., 2.0 for 2x context extension).
        factor: f32,
    },
    /// Dynamic NTK scaling.
    /// Base frequency is modified: base' = base * factor^(d/(d-2))
    DynamicNtk {
        /// Scaling factor for NTK base modification.
        factor: f32,
    },
    /// NTK-aware interpolation.
    /// Combines NTK base modification with position scaling.
    NtkAware {
        /// Scaling factor for context extension.
        factor: f32,
        /// Alpha for NTK-aware scaling (typically 1.0).
        alpha: f32,
    },
    /// YaRN (Yet another RoPE extensioN).
    /// Advanced scaling with attention factor and dimension-aware interpolation.
    Yarn(YarnConfig),
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self::None
    }
}

impl RopeScaling {
    /// Get the position scale factor for RoPE.
    pub fn scale(&self) -> f32 {
        match self {
            RopeScaling::None => 1.0,
            RopeScaling::Linear { factor } => 1.0 / factor,
            RopeScaling::DynamicNtk { .. } => 1.0, // NTK modifies base, not scale
            RopeScaling::NtkAware { factor, .. } => 1.0 / factor.sqrt(),
            RopeScaling::Yarn(config) => 1.0 / config.factor,
        }
    }

    /// Get the modified base frequency.
    pub fn effective_base(&self, base: f32, dims: i32) -> f32 {
        match self {
            RopeScaling::None | RopeScaling::Linear { .. } => base,
            RopeScaling::DynamicNtk { factor } => {
                base * factor.powf(dims as f32 / (dims - 2) as f32)
            }
            RopeScaling::NtkAware { factor, alpha } => {
                base * (alpha * factor - alpha + 1.0).powf(dims as f32 / (dims - 2) as f32)
            }
            RopeScaling::Yarn(config) => config.compute_base(base, dims),
        }
    }
}

impl RopeScaling {
    /// Parse rope_scaling from a HuggingFace config HashMap.
    ///
    /// Expected keys:
    /// - "type": "linear", "dynamic", "yarn" (String)
    /// - "factor": scaling factor (Float)
    /// - "original_max_position_embeddings": for YaRN (Float)
    /// - "attention_factor": optional YaRN attention factor (Float)
    pub fn from_config_map(map: &std::collections::HashMap<String, serde_json::Value>) -> Self {
        let rope_type = map
            .get("type")
            .or_else(|| map.get("rope_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        let factor = map.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

        match rope_type {
            "linear" => RopeScaling::Linear { factor },
            "dynamic" => RopeScaling::DynamicNtk { factor },
            "yarn" => {
                let original_max_pos = map
                    .get("original_max_position_embeddings")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(4096) as i32;
                let mut config = YarnConfig::new(factor, original_max_pos);
                if let Some(attn) = map.get("attention_factor").and_then(|v| v.as_f64()) {
                    config = config.with_attention_factor(attn as f32);
                }
                if let Some(beta_fast) = map.get("beta_fast").and_then(|v| v.as_f64()) {
                    if let Some(beta_slow) = map.get("beta_slow").and_then(|v| v.as_f64()) {
                        config = config.with_betas(beta_fast as f32, beta_slow as f32);
                    }
                }
                RopeScaling::Yarn(config)
            }
            _ => RopeScaling::None,
        }
    }
}

/// Configuration for YaRN (Yet another RoPE extensioN).
///
/// YaRN provides high-quality context extension by applying different
/// interpolation strategies to different frequency bands of RoPE.
#[derive(Debug, Clone)]
pub struct YarnConfig {
    /// Extension factor (e.g., 8.0 for 8x extension).
    pub factor: f32,
    /// Original maximum position embeddings the model was trained on.
    pub original_max_position: i32,
    /// Beta for fast wavelength (default: 32).
    pub beta_fast: f32,
    /// Beta for slow wavelength (default: 1).
    pub beta_slow: f32,
    /// Attention scaling factor (default: computed from factor).
    pub attention_factor: Option<f32>,
    /// Whether to use extrapolation for positions beyond training.
    pub extrapolation_factor: f32,
}

impl YarnConfig {
    /// Create a new YaRN configuration.
    pub fn new(factor: f32, original_max_position: i32) -> Self {
        Self {
            factor,
            original_max_position,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
            extrapolation_factor: 1.0,
        }
    }

    /// Set beta values for wavelength boundaries.
    pub fn with_betas(mut self, beta_fast: f32, beta_slow: f32) -> Self {
        self.beta_fast = beta_fast;
        self.beta_slow = beta_slow;
        self
    }

    /// Set attention scaling factor.
    pub fn with_attention_factor(mut self, factor: f32) -> Self {
        self.attention_factor = Some(factor);
        self
    }

    /// Get the attention scaling factor.
    ///
    /// If not explicitly set, computed as: 0.1 * ln(factor) + 1.0
    pub fn get_attention_factor(&self) -> f32 {
        self.attention_factor
            .unwrap_or_else(|| 0.1 * self.factor.ln() + 1.0)
    }

    /// Compute the modified base for YaRN.
    ///
    /// Uses the extension factor (not attention factor) to modify the base frequency,
    /// matching the NTK-aware base modification: base * factor^(dims / (dims - 2))
    fn compute_base(&self, base: f32, dims: i32) -> f32 {
        base * self.factor.powf(dims as f32 / (dims - 2) as f32)
    }

    /// Compute the interpolation factor for each dimension.
    ///
    /// Returns a tensor of shape [dims/2] with interpolation weights.
    ///
    /// # Arguments
    /// * `dims` - Number of RoPE dimensions
    /// * `base` - RoPE base frequency (rope_theta from config, e.g. 10000.0)
    pub fn compute_mscale(&self, dims: i32, base: f32) -> Vec<f32> {
        let mut mscale = Vec::with_capacity((dims / 2) as usize);

        for i in 0..(dims / 2) {
            let dim = 2 * i;
            // Compute wavelength for this dimension using the configured base
            let wavelength = 2.0 * std::f32::consts::PI * base.powf(dim as f32 / dims as f32);

            // Compute bounds
            let low = self.original_max_position as f32 / self.beta_fast;
            let high = self.original_max_position as f32 / self.beta_slow;

            // Ramp function
            let ramp = if wavelength < low {
                0.0 // Full interpolation
            } else if wavelength > high {
                1.0 // No interpolation
            } else {
                // Linear ramp between bounds
                (wavelength - low) / (high - low)
            };

            // Final scale: blend between interpolated (1/factor) and original (1)
            let scale = (1.0 - ramp) / self.factor + ramp;
            mscale.push(scale);
        }

        mscale
    }
}

/// Apply RoPE to a tensor (functional version).
///
/// # Arguments
/// * `x` - Input tensor of shape [..., seq_len, head_dim]
/// * `dims` - Number of dimensions to apply RoPE to
/// * `traditional` - If true, use traditional RoPE implementation
/// * `base` - Base frequency for the embeddings
/// * `scale` - Scale for the positions
/// * `offset` - Position offset
///
/// # Returns
/// Tensor with rotary embeddings applied.
pub fn apply_rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: i32,
) -> Result<Array, Exception> {
    Ok(fast::rope(x, dims, traditional, base, scale, offset))
}

/// Apply RoPE with explicit position IDs.
///
/// This is essential for packed sequence training where multiple sequences
/// are concatenated and position IDs need to reset for each sequence.
///
/// Uses the non-traditional (efficient) RoPE implementation where dimensions
/// are split in half rather than interleaved.
///
/// # Arguments
/// * `x` - Input tensor of shape [batch, heads, seq_len, head_dim]
/// * `position_ids` - Position indices of shape [seq_len]
/// * `dims` - Number of dimensions to apply RoPE to (usually head_dim)
/// * `traditional` - If true, use traditional (interleaved) RoPE
/// * `base` - Base frequency for the embeddings (default 10000.0)
/// * `scale` - Scale factor for positions (default 1.0)
///
/// # Returns
/// Tensor with rotary embeddings applied according to position_ids.
///
/// # Example
/// ```ignore
/// // Packed sequences: [seq1_tok1, seq1_tok2, seq2_tok1, seq2_tok2, seq2_tok3]
/// // Position IDs:     [0,         1,         0,         1,         2]
/// let x = Array::zeros::<f32>(&[1, 4, 5, 64]); // batch=1, heads=4, seq=5, dim=64
/// let position_ids = Array::from_i32_slice(&[0_i32, 1, 0, 1, 2], &[5]);
/// let output = apply_rope_with_positions(&x, &position_ids, 64, false, 10000.0, 1.0)?;
/// ```
pub fn apply_rope_with_positions(
    x: &Array,
    position_ids: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
) -> Result<Array, Exception> {
    // x shape: [batch, heads, seq_len, head_dim]
    let shape = x.shape();
    let head_dim = shape[3];
    let half_dims = dims / 2;

    // Compute inverse frequencies: inv_freq[i] = 1.0 / (base^(2i/dims))
    // indices: [0, 1, ..., half_dims-1] as float
    let indices = ops::arange_range(0, half_dims); // [half_dims] float32
    let neg_two_over_dims = Array::from_f32(-2.0 / dims as f32);
    let exponents = indices.multiply(&neg_two_over_dims);
    let base_arr = Array::from_f32(base);
    let inv_freq = base_arr.pow(&exponents); // [half_dims]

    // position_ids: [seq_len] as i32 → float and scale
    let pos_float = position_ids.as_dtype(Dtype::Float32.as_i32());
    let scale_arr = Array::from_f32(scale);
    let scaled_pos = pos_float.multiply(&scale_arr); // [seq_len]

    // Compute angles: [seq_len, half_dims]
    let pos_expanded = scaled_pos.expand_dims(-1); // [seq_len, 1]
    let inv_freq_expanded = inv_freq.expand_dims(0); // [1, half_dims]
    let angles = pos_expanded.multiply(&inv_freq_expanded); // [seq_len, half_dims]

    // Compute cos and sin
    let cos_theta = angles.cos(); // [seq_len, half_dims]
    let sin_theta = angles.sin(); // [seq_len, half_dims]

    // Reshape for broadcasting with x: [1, 1, seq_len, half_dims]
    let cos_theta = cos_theta.reshape(&[1, 1, -1, half_dims]);
    let sin_theta = sin_theta.reshape(&[1, 1, -1, half_dims]);

    if traditional {
        // Traditional (interleaved) RoPE: pairs are (x[0], x[1]), (x[2], x[3]), ...
        // x shape: [batch, heads, seq_len, head_dim]

        let x_rope = if dims < head_dim {
            // Split off only the RoPE portion
            let parts = x.split(&[dims], -1);
            parts[0].clone()
        } else {
            x.clone()
        };

        // Reshape to [..., half_dims, 2] for interleaved pairs
        let rope_shape = x_rope.shape();
        let batch = rope_shape[0];
        let heads = rope_shape[1];
        let seq_len = rope_shape[2];
        let x_pairs = x_rope.reshape(&[batch, heads, seq_len, half_dims, 2]);

        // Extract even (index 0) and odd (index 1) elements along last dim
        // Use slice: [batch, heads, seq_len, half_dims, 0..1] then squeeze(-1)
        let x_even = x_pairs
            .slice(&[0, 0, 0, 0, 0], &[batch, heads, seq_len, half_dims, 1])
            .squeeze(-1); // [batch, heads, seq_len, half_dims]
        let x_odd = x_pairs
            .slice(&[0, 0, 0, 0, 1], &[batch, heads, seq_len, half_dims, 2])
            .squeeze(-1); // [batch, heads, seq_len, half_dims]

        // Apply rotation
        let r_even = x_even
            .multiply(&cos_theta)
            .subtract(&x_odd.multiply(&sin_theta));
        let r_odd = x_even.multiply(&sin_theta).add(&x_odd.multiply(&cos_theta));

        // Interleave back: stack along last dim then reshape
        let stacked = ops::stack_axis(vec![r_even, r_odd].as_slice(), -1); // [..., half_dims, 2]
        let x_rotated = stacked.reshape(&[batch, heads, seq_len, dims]);

        if dims < head_dim {
            let parts = x.split(&[dims], -1);
            Ok(ops::concatenate_axis(&[&x_rotated, &parts[1]], -1))
        } else {
            Ok(x_rotated)
        }
    } else {
        // Non-traditional (split-half) RoPE: first half and second half
        let parts = if dims == head_dim {
            // Split into 2 equal halves: split at [half_dims]
            x.split(&[half_dims], -1)
        } else {
            // Split at [half_dims, dims]: produces 3 parts
            x.split(&[half_dims, dims], -1)
        };

        let x1 = &parts[0]; // [batch, heads, seq_len, half_dims]
        let x2 = &parts[1];

        // Apply rotation:
        // rx1 = x1 * cos - x2 * sin
        // rx2 = x1 * sin + x2 * cos
        let rx1 = x1.multiply(&cos_theta).subtract(&x2.multiply(&sin_theta));
        let rx2 = x1.multiply(&sin_theta).add(&x2.multiply(&cos_theta));

        let x_rotated = ops::concatenate_axis(&[&rx1, &rx2], -1);

        if dims < head_dim && parts.len() > 2 {
            let x_pass = &parts[2];
            Ok(ops::concatenate_axis(&[&x_rotated, x_pass], -1))
        } else {
            Ok(x_rotated)
        }
    }
}

/// Apply RoPE with per-batch-row position IDs.
///
/// Sibling of [`apply_rope_with_positions`] for the fused continuous-batching
/// decode path. Each batch row carries its own absolute offsets, so the
/// computed cos/sin are broadcast as `[batch, 1, seq_len, half_dims]` rather
/// than `[1, 1, seq_len, half_dims]`.
///
/// # Arguments
/// * `x` - Input tensor of shape `[batch, heads, seq_len, head_dim]`
/// * `position_ids` - Position indices of shape `[batch, seq_len]` (int32)
/// * `dims` - Number of dimensions to apply RoPE to
/// * `traditional` - If true, use traditional (interleaved) RoPE
/// * `base` - Base frequency for the embeddings
/// * `scale` - Scale factor for positions
pub fn apply_rope_with_per_batch_positions(
    x: &Array,
    position_ids: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
) -> Result<Array, Exception> {
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(Exception::custom(format!(
            "apply_rope_with_per_batch_positions: expected rank-4 input, got {shape:?}"
        )));
    }
    let batch = shape[0];
    let head_dim = shape[3];
    let half_dims = dims / 2;

    let pos_shape = position_ids.shape();
    if pos_shape.len() != 2 || pos_shape[0] != batch {
        return Err(Exception::custom(format!(
            "apply_rope_with_per_batch_positions: position_ids must be [batch={batch}, seq_len], got {pos_shape:?}"
        )));
    }

    // inv_freq[i] = base ^ (-2i / dims)
    let indices = ops::arange_range(0, half_dims);
    let neg_two_over_dims = Array::from_f32(-2.0 / dims as f32);
    let exponents = indices.multiply(&neg_two_over_dims);
    let base_arr = Array::from_f32(base);
    let inv_freq = base_arr.pow(&exponents); // [half_dims]

    // scaled positions: [batch, seq_len]
    let pos_float = position_ids.as_dtype(Dtype::Float32.as_i32());
    let scale_arr = Array::from_f32(scale);
    let scaled_pos = pos_float.multiply(&scale_arr);

    // angles: [batch, seq_len, half_dims]
    let pos_expanded = scaled_pos.expand_dims(-1); // [batch, seq_len, 1]
    let inv_freq_expanded = inv_freq.reshape(&[1, 1, half_dims]); // [1, 1, half_dims]
    let angles = pos_expanded.multiply(&inv_freq_expanded);

    let cos_theta = angles.cos();
    let sin_theta = angles.sin();

    // Reshape to [batch, 1, seq_len, half_dims] for broadcasting with
    // x shaped [batch, heads, seq_len, head_dim].
    let cos_theta = cos_theta.reshape(&[batch, 1, -1, half_dims]);
    let sin_theta = sin_theta.reshape(&[batch, 1, -1, half_dims]);

    if traditional {
        let x_rope = if dims < head_dim {
            x.split(&[dims], -1).remove(0)
        } else {
            x.clone()
        };
        let rope_shape = x_rope.shape();
        let batch_r = rope_shape[0];
        let heads = rope_shape[1];
        let seq_len = rope_shape[2];
        let x_pairs = x_rope.reshape(&[batch_r, heads, seq_len, half_dims, 2]);
        let x_even = x_pairs
            .slice(&[0, 0, 0, 0, 0], &[batch_r, heads, seq_len, half_dims, 1])
            .squeeze(-1);
        let x_odd = x_pairs
            .slice(&[0, 0, 0, 0, 1], &[batch_r, heads, seq_len, half_dims, 2])
            .squeeze(-1);

        let r_even = x_even
            .multiply(&cos_theta)
            .subtract(&x_odd.multiply(&sin_theta));
        let r_odd = x_even.multiply(&sin_theta).add(&x_odd.multiply(&cos_theta));

        let stacked = ops::stack_axis(vec![r_even, r_odd].as_slice(), -1);
        let x_rotated = stacked.reshape(&[batch_r, heads, seq_len, dims]);

        if dims < head_dim {
            let parts = x.split(&[dims], -1);
            Ok(ops::concatenate_axis(&[&x_rotated, &parts[1]], -1))
        } else {
            Ok(x_rotated)
        }
    } else {
        let parts = if dims == head_dim {
            x.split(&[half_dims], -1)
        } else {
            x.split(&[half_dims, dims], -1)
        };
        let x1 = &parts[0];
        let x2 = &parts[1];
        let rx1 = x1.multiply(&cos_theta).subtract(&x2.multiply(&sin_theta));
        let rx2 = x1.multiply(&sin_theta).add(&x2.multiply(&cos_theta));
        let x_rotated = ops::concatenate_axis(&[&rx1, &rx2], -1);
        if dims < head_dim && parts.len() > 2 {
            Ok(ops::concatenate_axis(&[&x_rotated, &parts[2]], -1))
        } else {
            Ok(x_rotated)
        }
    }
}

/// Apply RoPE with extended context scaling.
///
/// # Arguments
/// * `x` - Input tensor of shape [..., seq_len, head_dim]
/// * `dims` - Number of dimensions to apply RoPE to
/// * `traditional` - If true, use traditional RoPE implementation
/// * `base` - Base frequency for the embeddings
/// * `offset` - Position offset
/// * `scaling` - Scaling configuration for extended context
///
/// # Returns
/// Tensor with scaled rotary embeddings applied.
pub fn apply_rope_scaled(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    offset: i32,
    scaling: &RopeScaling,
) -> Result<Array, Exception> {
    let effective_base = scaling.effective_base(base, dims);
    let scale = scaling.scale();
    Ok(fast::rope(
        x,
        dims,
        traditional,
        effective_base,
        scale,
        offset,
    ))
}

/// Compute the effective maximum context length after scaling.
///
/// This is approximate - actual performance may vary based on model and task.
pub fn effective_context_length(original: i32, scaling: &RopeScaling) -> i32 {
    match scaling {
        RopeScaling::None => original,
        RopeScaling::Linear { factor } => (original as f32 * factor) as i32,
        RopeScaling::DynamicNtk { factor } => (original as f32 * factor) as i32,
        RopeScaling::NtkAware { factor, .. } => (original as f32 * factor) as i32,
        RopeScaling::Yarn(config) => (original as f32 * config.factor) as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::{Array, Dtype, random};

    #[test]
    fn test_rope_functional() {
        let x = random::normal(&[8, 4, 64], Dtype::Float32);
        let output = apply_rope(&x, 64, false, 10000.0, 1.0, 0).unwrap();
        assert_eq!(output.shape()[1], 4); // seq_len
        assert_eq!(output.shape()[2], 64); // head_dim
    }

    #[test]
    fn test_rope_scaling_linear() {
        let scaling = RopeScaling::Linear { factor: 2.0 };
        assert_eq!(scaling.scale(), 0.5);
        assert_eq!(scaling.effective_base(10000.0, 64), 10000.0);
    }

    #[test]
    fn test_rope_scaling_ntk() {
        let scaling = RopeScaling::DynamicNtk { factor: 2.0 };
        assert_eq!(scaling.scale(), 1.0); // NTK doesn't modify scale

        // NTK modifies base: base * factor^(d/(d-2))
        let effective_base = scaling.effective_base(10000.0, 64);
        assert!(effective_base > 10000.0);
    }

    #[test]
    fn test_rope_scaling_ntk_aware() {
        let scaling = RopeScaling::NtkAware {
            factor: 4.0,
            alpha: 1.0,
        };

        // NTK-aware uses sqrt of factor for scale
        assert!((scaling.scale() - 0.5).abs() < 0.01);

        // Should modify base
        let effective_base = scaling.effective_base(10000.0, 64);
        assert!(effective_base > 10000.0);
    }

    #[test]
    fn test_yarn_config() {
        let config = YarnConfig::new(8.0, 4096);
        assert_eq!(config.factor, 8.0);
        assert_eq!(config.original_max_position, 4096);
        assert_eq!(config.beta_fast, 32.0);
        assert_eq!(config.beta_slow, 1.0);

        // Attention factor: 0.1 * ln(8) + 1.0 ≈ 1.208
        let attn = config.get_attention_factor();
        assert!((attn - 1.208).abs() < 0.01);
    }

    #[test]
    fn test_yarn_config_builder() {
        let config = YarnConfig::new(16.0, 4096)
            .with_betas(64.0, 2.0)
            .with_attention_factor(1.5);

        assert_eq!(config.beta_fast, 64.0);
        assert_eq!(config.beta_slow, 2.0);
        assert_eq!(config.get_attention_factor(), 1.5);
    }

    #[test]
    fn test_yarn_mscale() {
        let config = YarnConfig::new(8.0, 4096);
        let mscale = config.compute_mscale(64, 10000.0);

        assert_eq!(mscale.len(), 32); // dims / 2

        // Lower dimensions (high frequency) should have smaller scale (more interpolation)
        // Higher dimensions (low frequency) should have larger scale (less interpolation)
        assert!(mscale[0] > 0.0);
        assert!(mscale[0] <= 1.0);
    }

    #[test]
    fn test_rope_scaling_yarn() {
        let config = YarnConfig::new(8.0, 4096);
        let scaling = RopeScaling::Yarn(config);

        assert_eq!(scaling.scale(), 1.0 / 8.0);

        // YaRN should modify base
        let effective_base = scaling.effective_base(10000.0, 64);
        assert!(effective_base > 10000.0);
    }

    #[test]
    fn test_apply_rope_scaled() {
        let x = random::normal(&[8, 4, 64], Dtype::Float32);
        let scaling = RopeScaling::Linear { factor: 2.0 };

        let output = apply_rope_scaled(&x, 64, false, 10000.0, 0, &scaling).unwrap();
        assert_eq!(output.shape(), x.shape());
    }

    #[test]
    fn test_effective_context_length() {
        let original = 4096;

        assert_eq!(effective_context_length(original, &RopeScaling::None), 4096);

        let linear = RopeScaling::Linear { factor: 2.0 };
        assert_eq!(effective_context_length(original, &linear), 8192);

        let yarn = RopeScaling::Yarn(YarnConfig::new(8.0, 4096));
        assert_eq!(effective_context_length(original, &yarn), 32768);
    }

    #[test]
    fn test_rope_default() {
        let scaling = RopeScaling::default();
        assert!(matches!(scaling, RopeScaling::None));
    }
}
