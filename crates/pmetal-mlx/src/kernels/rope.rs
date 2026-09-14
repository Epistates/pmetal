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
    /// Llama 3 frequency-band scaling (`"rope_type": "llama3"`).
    ///
    /// Unlike every variant above, this has no scalar `(base, scale)`
    /// equivalent — see [`Llama3Config`]. Callers must read
    /// [`RopeScaling::rope_periods`].
    Llama3(Llama3Config),
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self::None
    }
}

impl RopeScaling {
    /// Get the position scale factor for RoPE.
    ///
    /// Identity for [`RopeScaling::Llama3`], which rescales frequencies
    /// rather than positions — see [`Self::rope_periods`].
    pub fn scale(&self) -> f32 {
        match self {
            RopeScaling::None => 1.0,
            RopeScaling::Linear { factor } => 1.0 / factor,
            RopeScaling::DynamicNtk { .. } => 1.0, // NTK modifies base, not scale
            RopeScaling::NtkAware { factor, .. } => 1.0 / factor.sqrt(),
            RopeScaling::Yarn(config) => 1.0 / config.factor,
            RopeScaling::Llama3(_) => 1.0,
        }
    }

    /// Get the modified base frequency.
    ///
    /// Identity for [`RopeScaling::Llama3`] — see [`Self::rope_periods`].
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
            RopeScaling::Llama3(_) => base,
        }
    }

    /// Per-dimension RoPE periods for schemes no `(base, scale)` pair can express.
    ///
    /// Returns `None` for the scalar schemes, whose behaviour
    /// [`Self::effective_base`] and [`Self::scale`] fully describe.
    ///
    /// **A caller that ignores a `Some` runs the model unscaled.** Llama 3
    /// rescales each frequency band independently, so there is no single base
    /// or position scale that reproduces it; reading only `effective_base` and
    /// `scale` silently yields plain RoPE.
    ///
    /// The returned layout is what `mx.fast.rope`'s `freqs` argument wants:
    /// *periods* (`base^(2i/dims)`, the reciprocal of an inverse frequency),
    /// length `dims / 2`.
    pub fn rope_periods(&self, dims: i32, base: f32) -> Option<Vec<f32>> {
        match self {
            RopeScaling::Llama3(config) => Some(config.rope_periods(dims, base)),
            _ => None,
        }
    }
}

impl RopeScaling {
    /// Parse rope_scaling from a HuggingFace config HashMap.
    ///
    /// Expected keys:
    /// - "type": "linear", "dynamic", "yarn", "llama3" (String)
    /// - "factor": scaling factor (Float)
    /// - "original_max_position_embeddings": for YaRN and Llama 3 (Float)
    /// - "attention_factor": optional YaRN attention factor (Float)
    /// - "low_freq_factor" / "high_freq_factor": Llama 3 band boundaries (Float)
    pub fn from_config_map(map: &std::collections::HashMap<String, serde_json::Value>) -> Self {
        // `as_f64` rather than `as_i64` even for the integer-valued keys: some
        // callers round-trip the map through an f32 before handing it over, so
        // `original_max_position_embeddings` arrives as `8192.0`, and `as_i64`
        // returns `None` for a JSON float.
        let number = |key: &str| map.get(key).and_then(|v| v.as_f64()).map(|v| v as f32);

        let rope_type = map
            .get("type")
            .or_else(|| map.get("rope_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        let factor = number("factor").unwrap_or(1.0);

        match rope_type {
            "linear" => RopeScaling::Linear { factor },
            "dynamic" => RopeScaling::DynamicNtk { factor },
            "yarn" => {
                let original_max_pos =
                    number("original_max_position_embeddings").unwrap_or(4096.0) as i32;
                let mut config = YarnConfig::new(factor, original_max_pos);
                if let Some(attn) = number("attention_factor") {
                    config = config.with_attention_factor(attn);
                }
                if let Some(beta_fast) = number("beta_fast") {
                    if let Some(beta_slow) = number("beta_slow") {
                        config = config.with_betas(beta_fast, beta_slow);
                    }
                }
                RopeScaling::Yarn(config)
            }
            "llama3" => RopeScaling::Llama3(Llama3Config {
                factor,
                low_freq_factor: number("low_freq_factor").unwrap_or(1.0),
                high_freq_factor: number("high_freq_factor").unwrap_or(4.0),
                original_max_position: number("original_max_position_embeddings").unwrap_or(8192.0)
                    as i32,
            }),
            _ => RopeScaling::None,
        }
    }
}

/// Configuration for Llama 3 frequency-band RoPE scaling.
///
/// Llama 3.1 onwards extends context by rescaling RoPE frequencies according
/// to each dimension's wavelength, rather than by scaling positions or the
/// base. Three bands, split at `original_max_position / {low,high}_freq_factor`:
///
/// - **High frequency** (short wavelength): untouched, so local ordering is
///   preserved exactly as trained.
/// - **Low frequency** (long wavelength): divided by `factor`, the plain
///   linear interpolation that buys the extra context.
/// - **Medium**: a linear ramp between the two, which is what stops the seam
///   between the bands from showing up as a discontinuity.
///
/// This is why the scheme cannot ride the scalar `(base, scale)` path: the
/// three bands need three different treatments of the same base.
///
/// Mirrors `transformers`' `_compute_llama3_parameters`.
#[derive(Debug, Clone)]
pub struct Llama3Config {
    /// Context extension factor (32 for Llama 3.2, 8 for Llama 3.1).
    pub factor: f32,
    /// Divides `original_max_position` to give the low-frequency boundary.
    pub low_freq_factor: f32,
    /// Divides `original_max_position` to give the high-frequency boundary.
    pub high_freq_factor: f32,
    /// Context length the model was originally trained on.
    pub original_max_position: i32,
}

impl Llama3Config {
    /// Compute the `[dims / 2]` period table for this configuration.
    ///
    /// Periods, not inverse frequencies: that is the layout `mx.fast.rope`
    /// takes via its `freqs` argument, so the result feeds the fused kernel
    /// directly.
    pub fn rope_periods(&self, dims: i32, base: f32) -> Vec<f32> {
        let half = (dims / 2).max(0) as usize;
        let original = self.original_max_position as f32;
        let low_wavelen = original / self.low_freq_factor;
        let high_wavelen = original / self.high_freq_factor;
        // Degenerate config: no medium band to ramp across, and the ramp's
        // denominator would be zero. Fall back to the two-band split.
        let band_width = self.high_freq_factor - self.low_freq_factor;

        (0..half)
            .map(|i| {
                let period = base.powf((2 * i) as f32 / dims as f32);
                let inv_freq = 1.0 / period;
                let wavelen = 2.0 * std::f32::consts::PI * period;

                let scaled = if wavelen > low_wavelen {
                    inv_freq / self.factor
                } else if wavelen < high_wavelen || band_width == 0.0 {
                    inv_freq
                } else {
                    let smooth = (original / wavelen - self.low_freq_factor) / band_width;
                    (1.0 - smooth) * inv_freq / self.factor + smooth * inv_freq
                };
                1.0 / scaled
            })
            .collect()
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

/// Where a rotary embedding takes its positions from.
///
/// [`Offset`] is the contiguous run `offset, offset + 1, …`: every cached
/// decode, and every forward over one unbroken sequence. [`Explicit`] is one
/// position per token, which is what a packed batch needs so the second
/// sequence in a row restarts at 0 instead of continuing the first.
///
/// Architectures thread `Option<&Array>` down from their own forward and call
/// [`RopePositions::resolve`] once against the cache offset, rather than
/// branching separately at each `q` and `k` rotation.
///
/// [`Offset`]: RopePositions::Offset
/// [`Explicit`]: RopePositions::Explicit
#[derive(Debug, Clone, Copy)]
pub enum RopePositions<'a> {
    /// Positions `offset, offset + 1, …, offset + seq_len - 1`.
    Offset(i32),
    /// One position per token, shape `[seq_len]`.
    Explicit(&'a Array),
}

impl<'a> RopePositions<'a> {
    /// The caller's explicit positions when it has them, the contiguous run
    /// from `offset` otherwise.
    pub fn resolve(positions: Option<&'a Array>, offset: i32) -> Self {
        match positions {
            Some(ids) => Self::Explicit(ids),
            None => Self::Offset(offset),
        }
    }
}

/// Apply RoPE at `positions` using a scalar base frequency.
///
/// The contiguous arm is the fused `mx.fast.rope` kernel; the explicit arm
/// builds the cos/sin tables from the position vector. They compute the same
/// rotation, which this module's tests pin down for both `traditional`
/// settings, at a non-zero offset, and for partial RoPE.
pub fn rope(
    x: &Array,
    positions: RopePositions<'_>,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
) -> Result<Array, Exception> {
    match positions {
        RopePositions::Offset(offset) => apply_rope(x, dims, traditional, base, scale, offset),
        RopePositions::Explicit(ids) => {
            apply_rope_with_positions(x, ids, dims, traditional, base, scale)
        }
    }
}

/// [`rope`] with an explicit `[dims / 2]` inverse-frequency table.
///
/// Needed wherever no single `base` describes the rotation: Phi-3 LongRoPE
/// scales each band by its own `long_factor`, and YaRN blends per-band ramps.
pub fn rope_with_inv_freq(
    x: &Array,
    positions: RopePositions<'_>,
    inv_freq: &Array,
    dims: i32,
    traditional: bool,
) -> Result<Array, Exception> {
    match positions {
        RopePositions::Offset(offset) => {
            apply_rope_with_freqs(x, inv_freq, dims, traditional, offset)
        }
        RopePositions::Explicit(ids) => {
            rope_with_positions_and_inv_freq(x, ids, inv_freq, dims, traditional, 1.0)
        }
    }
}

/// [`rope`] with an explicit `[dims / 2]` *period* table, the form
/// `mx.fast.rope` takes through its `freqs=` argument.
///
/// Llama 3's frequency-band scaling is published this way, so the contiguous
/// arm stays on the fused kernel instead of rebuilding the tables per call.
pub fn rope_with_periods(
    x: &Array,
    positions: RopePositions<'_>,
    periods: &Array,
    dims: i32,
    traditional: bool,
    scale: f32,
) -> Result<Array, Exception> {
    match positions {
        RopePositions::Offset(offset) => Ok(fast::rope_with_freqs(
            x,
            dims,
            traditional,
            scale,
            offset,
            periods,
        )),
        RopePositions::Explicit(ids) => {
            apply_rope_with_positions_and_periods(x, ids, periods, dims, traditional, scale)
        }
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
    // Compute inverse frequencies: inv_freq[i] = 1.0 / (base^(2i/dims))
    // indices: [0, 1, ..., half_dims-1] as float
    let indices = ops::arange_range(0, dims / 2); // [half_dims] float32
    let neg_two_over_dims = Array::from_f32(-2.0 / dims as f32);
    let exponents = indices.multiply(&neg_two_over_dims);
    let base_arr = Array::from_f32(base);
    let inv_freq = base_arr.pow(&exponents); // [half_dims]

    rope_with_positions_and_inv_freq(x, position_ids, &inv_freq, dims, traditional, scale)
}

/// Apply RoPE with explicit position IDs and an explicit period table.
///
/// Sibling of [`apply_rope_with_positions`] for scalings that rescale each
/// frequency band separately, where no single `base` describes the rotation
/// (Llama 3). `periods` is the same `[dims / 2]` table `mx.fast.rope` takes —
/// periods, not inverse frequencies — so one table serves both the fused
/// contiguous-offset path and this packed-sequence one.
pub fn apply_rope_with_positions_and_periods(
    x: &Array,
    position_ids: &Array,
    periods: &Array,
    dims: i32,
    traditional: bool,
    scale: f32,
) -> Result<Array, Exception> {
    let inv_freq = Array::from_f32(1.0).divide(periods);
    rope_with_positions_and_inv_freq(x, position_ids, &inv_freq, dims, traditional, scale)
}

/// Shared body: rotate `x` at the given per-token positions using a
/// precomputed `[dims / 2]` inverse-frequency table.
fn rope_with_positions_and_inv_freq(
    x: &Array,
    position_ids: &Array,
    inv_freq: &Array,
    dims: i32,
    traditional: bool,
    scale: f32,
) -> Result<Array, Exception> {
    // x shape: [batch, heads, seq_len, head_dim]
    let half_dims = dims / 2;

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

    Ok(rope_rotate_with_cos_sin(
        x,
        &cos_theta,
        &sin_theta,
        dims,
        traditional,
    ))
}

/// Core RoPE rotation given precomputed `cos`/`sin` tables (broadcastable to
/// `[batch, heads, seq_len, half_dims]`). Shared by the position-ID and
/// custom-frequency entry points so the interleaved/split-half rotation lives
/// in exactly one place.
fn rope_rotate_with_cos_sin(
    x: &Array,
    cos_theta: &Array,
    sin_theta: &Array,
    dims: i32,
    traditional: bool,
) -> Array {
    let head_dim = x.shape()[3];
    let half_dims = dims / 2;

    if traditional {
        // Traditional (interleaved) RoPE: pairs are (x[0], x[1]), (x[2], x[3]), ...
        let x_rope = if dims < head_dim {
            let parts = x.split(&[dims], -1);
            parts[0].clone()
        } else {
            x.clone()
        };

        let rope_shape = x_rope.shape();
        let batch = rope_shape[0];
        let heads = rope_shape[1];
        let seq_len = rope_shape[2];
        let x_pairs = x_rope.reshape(&[batch, heads, seq_len, half_dims, 2]);

        let x_even = x_pairs
            .slice(&[0, 0, 0, 0, 0], &[batch, heads, seq_len, half_dims, 1])
            .squeeze(-1);
        let x_odd = x_pairs
            .slice(&[0, 0, 0, 0, 1], &[batch, heads, seq_len, half_dims, 2])
            .squeeze(-1);

        let r_even = x_even
            .multiply(cos_theta)
            .subtract(&x_odd.multiply(sin_theta));
        let r_odd = x_even.multiply(sin_theta).add(&x_odd.multiply(cos_theta));

        let stacked = ops::stack_axis(vec![r_even, r_odd].as_slice(), -1);
        let x_rotated = stacked.reshape(&[batch, heads, seq_len, dims]);

        if dims < head_dim {
            let parts = x.split(&[dims], -1);
            ops::concatenate_axis(&[&x_rotated, &parts[1]], -1)
        } else {
            x_rotated
        }
    } else {
        // Non-traditional (split-half) RoPE: first half and second half
        let parts = if dims == head_dim {
            x.split(&[half_dims], -1)
        } else {
            x.split(&[half_dims, dims], -1)
        };

        let x1 = &parts[0];
        let x2 = &parts[1];

        let rx1 = x1.multiply(cos_theta).subtract(&x2.multiply(sin_theta));
        let rx2 = x1.multiply(sin_theta).add(&x2.multiply(cos_theta));

        let x_rotated = ops::concatenate_axis(&[&rx1, &rx2], -1);

        if dims < head_dim && parts.len() > 2 {
            let x_pass = &parts[2];
            ops::concatenate_axis(&[&x_rotated, x_pass], -1)
        } else {
            x_rotated
        }
    }
}

/// Apply RoPE using explicit per-dimension inverse frequencies.
///
/// Unlike [`apply_rope`] (which derives `inv_freq[i] = base^(-2i/dims)` from a
/// single scalar `base`), this takes a precomputed `inv_freq` table of length
/// `dims/2`. This is required for Phi-3 LongRoPE / SuRoPE, where each
/// frequency is independently scaled by a per-dimension `long_factor`:
/// `inv_freq[i] = 1 / (long_factor[i] * base^(2i/dims))`.
///
/// Positions are the contiguous range `[offset, offset + seq_len)`. Any
/// magnitude (mscale) rescaling of the activations must be applied by the
/// caller *before* this call — this function only rotates.
///
/// # Arguments
/// * `x` - `[batch, heads, seq_len, head_dim]`
/// * `inv_freq` - `[dims/2]` angular frequencies
/// * `dims` - rotary dimension (may be < head_dim for partial RoPE)
/// * `traditional` - interleaved (true) vs split-half (false)
/// * `offset` - absolute position of the first token (KV-cache aware)
pub fn apply_rope_with_freqs(
    x: &Array,
    inv_freq: &Array,
    dims: i32,
    traditional: bool,
    offset: i32,
) -> Result<Array, Exception> {
    let seq_len = x.shape()[2];
    let half_dims = dims / 2;

    // positions: [offset, offset+1, ..., offset+seq_len-1] as float32
    let positions = ops::arange_range(offset, offset + seq_len);
    let angles = positions
        .expand_dims(-1) // [seq_len, 1]
        .multiply(&inv_freq.expand_dims(0)); // [1, half_dims] → [seq_len, half_dims]

    let cos_theta = angles.cos().reshape(&[1, 1, -1, half_dims]);
    let sin_theta = angles.sin().reshape(&[1, 1, -1, half_dims]);

    Ok(rope_rotate_with_cos_sin(
        x,
        &cos_theta,
        &sin_theta,
        dims,
        traditional,
    ))
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
        RopeScaling::Llama3(config) => (original as f32 * config.factor) as i32,
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

    /// Llama-3.2-1B's shipped `rope_scaling` block.
    fn llama3_2_config_map() -> std::collections::HashMap<String, serde_json::Value> {
        [
            ("rope_type", serde_json::Value::from("llama3")),
            ("factor", serde_json::Value::from(32.0)),
            ("low_freq_factor", serde_json::Value::from(1.0)),
            ("high_freq_factor", serde_json::Value::from(4.0)),
            // Deliberately a float: `LlamaConfig` round-trips rope_scaling
            // values through f32, so this is the shape the parser really sees.
            (
                "original_max_position_embeddings",
                serde_json::Value::from(8192.0),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    #[test]
    fn llama3_rope_type_parses_instead_of_falling_through_to_none() {
        let scaling = RopeScaling::from_config_map(&llama3_2_config_map());
        let RopeScaling::Llama3(config) = &scaling else {
            panic!("expected Llama3, got {scaling:?}");
        };
        assert_eq!(config.factor, 32.0);
        assert_eq!(config.low_freq_factor, 1.0);
        assert_eq!(config.high_freq_factor, 4.0);
        // The float spelling has to survive; `as_i64` would drop it to the
        // 8192 default by luck here and to the wrong value elsewhere.
        assert_eq!(config.original_max_position, 8192);

        // Neither scalar knob carries the scaling, so an unaware caller gets
        // plain RoPE. That is the failure this variant exists to make visible.
        assert_eq!(scaling.scale(), 1.0);
        assert_eq!(scaling.effective_base(500000.0, 64), 500000.0);
    }

    #[test]
    fn llama3_periods_match_transformers_compute_llama3_parameters() {
        let scaling = RopeScaling::from_config_map(&llama3_2_config_map());
        let periods = scaling
            .rope_periods(64, 500000.0)
            .expect("llama3 must publish per-dimension periods");
        assert_eq!(periods.len(), 32);

        // Reference: `1 / _compute_llama3_parameters(...)[0]` from
        // transformers, for Llama-3.2-1B (head_dim 64, rope_theta 5e5).
        // The three sampled bands are, in order: untouched high frequency,
        // ramped medium, and low frequency divided by `factor`.
        for (index, expected) in [
            (0_usize, 1.0_f32),
            (1, 1.506929),
            (10, 60.384865),
            (15, 774.864624),
            (20, 116682.632812),
            (24, 601696.5),
            (28, 3102764.0),
            (31, 10617620.0),
        ] {
            let got = periods[index];
            let rel = (got - expected).abs() / expected;
            assert!(
                rel < 1e-5,
                "period[{index}]: got {got}, transformers says {expected} (rel {rel:e})"
            );
        }

        // The high band is left alone, which is the whole point of banding
        // rather than scaling everything. (Not bit-exact: the untouched branch
        // still round-trips through `1 / (1 / period)`.)
        let untouched_high = 500000.0_f32.powf(20.0 / 64.0);
        assert!((periods[10] / untouched_high - 1.0).abs() < 1e-6);
        // ...and the low band is a clean factor-32 stretch.
        let unscaled_low = 500000.0_f32.powf(48.0 / 64.0);
        assert!((periods[24] / unscaled_low - 32.0).abs() < 1e-3);
    }

    #[test]
    fn llama3_degenerate_band_split_does_not_produce_nan() {
        // low == high leaves no medium band to ramp across; the ramp's
        // denominator is zero, so guard rather than emit NaN frequencies.
        let config = Llama3Config {
            factor: 8.0,
            low_freq_factor: 4.0,
            high_freq_factor: 4.0,
            original_max_position: 8192,
        };
        assert!(
            config
                .rope_periods(64, 500000.0)
                .iter()
                .all(|p| p.is_finite())
        );
    }

    #[test]
    fn scalar_scalings_publish_no_period_table() {
        assert!(RopeScaling::None.rope_periods(64, 10000.0).is_none());
        assert!(
            RopeScaling::Linear { factor: 2.0 }
                .rope_periods(64, 10000.0)
                .is_none()
        );
        assert!(
            RopeScaling::Yarn(YarnConfig::new(8.0, 4096))
                .rope_periods(64, 10000.0)
                .is_none()
        );
    }

    // ------------------------------------------------------------------
    // RopePositions
    //
    // The two arms take different code paths — a fused Metal kernel versus
    // cos/sin tables built here — so "same rotation" is an assertion, not a
    // definition. Everything downstream of `RopePositions` assumes it.
    // ------------------------------------------------------------------

    fn ramp(shape: &[i32]) -> Array {
        let n: i32 = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin()).collect();
        Array::from_slice(&values, shape)
    }

    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let n = a.shape().iter().product::<i32>() as usize;
        let a = a.subtract(b).abs();
        a.try_eval().expect("eval");
        a.as_slice::<f32>()[..n]
            .iter()
            .fold(0.0f32, |worst, v| worst.max(v.abs()))
    }

    fn contiguous_positions(offset: i32, seq_len: i32) -> Array {
        let values: Vec<i32> = (offset..offset + seq_len).collect();
        Array::from_i32_slice_shaped(&values, &[seq_len])
    }

    #[test]
    fn contiguous_and_explicit_positions_agree() {
        let (seq_len, dims) = (6, 16);
        let x = ramp(&[1, 2, seq_len, dims]);

        for traditional in [false, true] {
            for offset in [0, 5] {
                let contiguous = rope(
                    &x,
                    RopePositions::Offset(offset),
                    dims,
                    traditional,
                    10000.0,
                    1.0,
                )
                .unwrap();
                let ids = contiguous_positions(offset, seq_len);
                let explicit = rope(
                    &x,
                    RopePositions::Explicit(&ids),
                    dims,
                    traditional,
                    10000.0,
                    1.0,
                )
                .unwrap();

                let diff = max_abs_diff(&contiguous, &explicit);
                assert!(
                    diff < 1e-4,
                    "traditional={traditional} offset={offset}: max |Δ| = {diff:e}"
                );
            }
        }
    }

    /// Partial RoPE (`dims < head_dim`) has to leave the tail untouched on
    /// both arms, not just the fused one.
    #[test]
    fn contiguous_and_explicit_positions_agree_on_partial_rope() {
        let (seq_len, head_dim, dims) = (4, 16, 8);
        let x = ramp(&[1, 2, seq_len, head_dim]);
        let ids = contiguous_positions(3, seq_len);

        for traditional in [false, true] {
            let contiguous = rope(
                &x,
                RopePositions::Offset(3),
                dims,
                traditional,
                10000.0,
                1.0,
            )
            .unwrap();
            let explicit = rope(
                &x,
                RopePositions::Explicit(&ids),
                dims,
                traditional,
                10000.0,
                1.0,
            )
            .unwrap();

            let diff = max_abs_diff(&contiguous, &explicit);
            assert!(diff < 1e-4, "traditional={traditional}: max |Δ| = {diff:e}");
        }
    }

    #[test]
    fn contiguous_and_explicit_positions_agree_with_an_inv_freq_table() {
        let (seq_len, dims) = (5, 16);
        let x = ramp(&[1, 2, seq_len, dims]);
        // A table no scalar base produces: every other band stretched.
        let table: Vec<f32> = (0..dims / 2)
            .map(|i| 1.0 / (10000.0f32.powf(2.0 * i as f32 / dims as f32) * (1.0 + i as f32)))
            .collect();
        let inv_freq = Array::from_slice(&table, &[dims / 2]);
        let ids = contiguous_positions(2, seq_len);

        let contiguous =
            rope_with_inv_freq(&x, RopePositions::Offset(2), &inv_freq, dims, false).unwrap();
        let explicit =
            rope_with_inv_freq(&x, RopePositions::Explicit(&ids), &inv_freq, dims, false).unwrap();

        let diff = max_abs_diff(&contiguous, &explicit);
        assert!(diff < 1e-4, "max |Δ| = {diff:e}");
    }

    #[test]
    fn contiguous_and_explicit_positions_agree_with_a_period_table() {
        let (seq_len, dims) = (5, 64);
        let x = ramp(&[1, 2, seq_len, dims]);
        let periods = Llama3Config {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position: 8192,
        }
        .rope_periods(dims, 500000.0);
        let periods = Array::from_slice(&periods, &[dims / 2]);
        let ids = contiguous_positions(7, seq_len);

        let contiguous =
            rope_with_periods(&x, RopePositions::Offset(7), &periods, dims, false, 1.0).unwrap();
        let explicit = rope_with_periods(
            &x,
            RopePositions::Explicit(&ids),
            &periods,
            dims,
            false,
            1.0,
        )
        .unwrap();

        let diff = max_abs_diff(&contiguous, &explicit);
        assert!(diff < 1e-4, "max |Δ| = {diff:e}");
    }

    /// The reason the type exists: two sequences packed into one row must
    /// rotate as if each had been run on its own.
    #[test]
    fn packed_positions_restart_the_rotation_at_a_sequence_boundary() {
        let dims = 16;
        let first = ramp(&[1, 1, 2, dims]);
        let second = ramp(&[1, 1, 3, dims]);
        let packed = ops::concatenate_axis(&[&first, &second], 2);

        let separate = ops::concatenate_axis(
            &[
                &rope(&first, RopePositions::Offset(0), dims, false, 10000.0, 1.0).unwrap(),
                &rope(&second, RopePositions::Offset(0), dims, false, 10000.0, 1.0).unwrap(),
            ],
            2,
        );

        let ids = Array::from_i32_slice_shaped(&[0, 1, 0, 1, 2], &[5]);
        let together = rope(
            &packed,
            RopePositions::Explicit(&ids),
            dims,
            false,
            10000.0,
            1.0,
        )
        .unwrap();
        assert!(max_abs_diff(&separate, &together) < 1e-4);

        // And that the run-through positions a packed forward gets today are
        // genuinely different, so the test above is not vacuous.
        let run_through =
            rope(&packed, RopePositions::Offset(0), dims, false, 10000.0, 1.0).unwrap();
        assert!(max_abs_diff(&separate, &run_through) > 1e-2);
    }

    #[test]
    fn resolve_prefers_explicit_positions_over_the_cache_offset() {
        let ids = Array::from_i32_slice_shaped(&[0, 1, 0], &[3]);
        assert!(matches!(
            RopePositions::resolve(Some(&ids), 17),
            RopePositions::Explicit(_)
        ));
        assert!(matches!(
            RopePositions::resolve(None, 17),
            RopePositions::Offset(17)
        ));
    }
}
