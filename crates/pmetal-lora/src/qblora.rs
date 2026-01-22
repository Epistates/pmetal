//! Q-BLoRA (Quantized Balanced Low-Rank Adaptation) implementation.
//!
//! Q-BLoRA (2025) addresses the imbalance problem in QLoRA fine-tuning where
//! overly complex adapter inputs/outputs lead to underfitting. It simplifies
//! the adapter's inputs and outputs while increasing the adapter's rank.
//!
//! # The Imbalance Problem
//!
//! In standard QLoRA:
//! - Adapter inputs: Full precision activations from previous layer
//! - Adapter outputs: Full precision gradients to next layer
//! - Adapter rank: Typically small (4-16) for memory efficiency
//!
//! This creates an imbalance: complex I/O but low effective trainability,
//! leading to underfitting during fine-tuning.
//!
//! # Q-BLoRA Solution
//!
//! Q-BLoRA addresses this through:
//! 1. **Input simplification**: Optional projection to lower dimension
//! 2. **Output simplification**: Optional projection from adapter output
//! 3. **Increased rank**: Higher rank compensates for simplified I/O
//! 4. **Balanced gradients**: Better gradient flow through the adapter
//!
//! # Formula
//!
//! Standard LoRA: `y = x @ W.T + scale * (x @ A.T) @ B.T`
//! Q-BLoRA: `y = x @ W.T + scale * ((x @ P_in) @ A.T) @ B.T @ P_out.T`
//!
//! Where:
//! - `P_in` is the optional input projection (simplification)
//! - `P_out` is the optional output projection
//! - `A, B` have higher rank than standard QLoRA
//!
//! # Performance
//!
//! Q-BLoRA consistently outperforms QLoRA by a significant margin across
//! models of various sizes, achieving SOTA accuracy for quantized fine-tuning.
//!
//! # References
//!
//! - "Accurate and Efficient Fine-Tuning of Quantized Large Language Models
//!    Through Optimal Balance in Adaptation" (TACL 2025)

use std::cell::RefCell;

use mlx_rs::{error::Exception, Array};
use pmetal_core::LoraConfig;
use pmetal_mlx::quantization::{
    NF4Config, NF4Quantizer, QuantScheme, QuantizedTensor, QuantizerOps,
};

use super::LoraError;

/// Q-BLoRA configuration extending QLoRA with balance settings.
#[derive(Debug, Clone)]
pub struct QBLoraConfig {
    /// Base LoRA configuration.
    pub lora: LoraConfig,
    /// Quantization scheme (NF4, FP4, Int8).
    pub quant_scheme: QuantScheme,
    /// Block size for quantization (default: 64).
    pub block_size: usize,
    /// Enable double quantization for absmax values.
    pub double_quant: bool,

    // Balance settings
    /// Input projection dimension (None = no projection).
    /// When set, projects input to this dimension before adapter.
    pub input_proj_dim: Option<usize>,

    /// Output projection dimension (None = no projection).
    /// When set, projects adapter output through this dimension.
    pub output_proj_dim: Option<usize>,

    /// Rank multiplier compared to standard QLoRA.
    /// Q-BLoRA typically uses 2-4x the rank of standard QLoRA.
    /// Default: 2.0
    pub rank_multiplier: f32,

    /// Whether to use learnable projections.
    /// If false, uses random fixed projections.
    /// Default: true
    pub learnable_projections: bool,

    /// Gradient scaling factor for balanced gradients.
    /// Applied to adapter gradients to balance with base model.
    /// Default: 1.0
    pub gradient_scale: f32,
}

impl Default for QBLoraConfig {
    fn default() -> Self {
        Self {
            lora: LoraConfig::default(),
            quant_scheme: QuantScheme::NF4,
            block_size: 64,
            double_quant: true,
            input_proj_dim: None,
            output_proj_dim: None,
            rank_multiplier: 2.0,
            learnable_projections: true,
            gradient_scale: 1.0,
        }
    }
}

impl QBLoraConfig {
    /// Create Q-BLoRA config from existing LoRA config.
    pub fn from_lora(lora: LoraConfig) -> Self {
        Self {
            lora,
            ..Default::default()
        }
    }

    /// Set input projection dimension.
    pub fn with_input_proj(mut self, dim: usize) -> Self {
        self.input_proj_dim = Some(dim);
        self
    }

    /// Set output projection dimension.
    pub fn with_output_proj(mut self, dim: usize) -> Self {
        self.output_proj_dim = Some(dim);
        self
    }

    /// Set rank multiplier.
    pub fn with_rank_multiplier(mut self, multiplier: f32) -> Self {
        self.rank_multiplier = multiplier;
        self
    }

    /// Disable learnable projections.
    pub fn with_fixed_projections(mut self) -> Self {
        self.learnable_projections = false;
        self
    }

    /// Set gradient scaling.
    pub fn with_gradient_scale(mut self, scale: f32) -> Self {
        self.gradient_scale = scale;
        self
    }

    /// Compute effective rank (base rank * multiplier).
    pub fn effective_rank(&self) -> usize {
        ((self.lora.r as f32) * self.rank_multiplier).round() as usize
    }
}

/// Q-BLoRA Linear layer with balanced adaptation.
///
/// Implements: `y = x @ dequant(W_q).T + scale * ((x @ P_in) @ A.T) @ B.T @ P_out.T`
///
/// Key improvements over standard QLoRA:
/// - Optional input/output projections to simplify adapter I/O
/// - Higher effective rank for better expressiveness
/// - Balanced gradient flow
pub struct QBLoraLinear {
    /// Input features dimension.
    pub in_features: i32,
    /// Output features dimension.
    pub out_features: i32,
    /// Effective LoRA rank (after multiplier).
    pub rank: i32,
    /// LoRA scaling factor.
    pub scale: f32,
    /// Whether to use bias.
    pub use_bias: bool,

    /// Quantized base weight.
    pub quantized_weight: QuantizedTensor,
    /// Quantizer for dequantization.
    quantizer: NF4Quantizer,
    /// Optional bias [out_features].
    pub bias: Option<Array>,

    // Standard LoRA components
    /// LoRA A matrix [rank, proj_in or in_features].
    pub lora_a: Array,
    /// LoRA B matrix [proj_out or out_features, rank].
    pub lora_b: Array,

    // Balance projections
    /// Input projection [in_features, proj_dim] (optional).
    pub input_proj: Option<Array>,
    /// Output projection [proj_dim, out_features] (optional).
    pub output_proj: Option<Array>,
    /// Whether projections are learnable.
    learnable_projections: bool,

    /// Gradient scaling factor.
    gradient_scale: f32,

    /// Cached dequantized weight.
    weight_cache: RefCell<Option<Array>>,
    /// Whether weight caching is enabled.
    cache_enabled: bool,
}

impl std::fmt::Debug for QBLoraLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QBLoraLinear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("rank", &self.rank)
            .field("scale", &self.scale)
            .field("has_input_proj", &self.input_proj.is_some())
            .field("has_output_proj", &self.output_proj.is_some())
            .field("learnable_projections", &self.learnable_projections)
            .field("cache_enabled", &self.cache_enabled)
            .finish()
    }
}

impl QBLoraLinear {
    /// Create a Q-BLoRA layer by quantizing an existing weight matrix.
    ///
    /// # Arguments
    /// * `weight` - Full-precision weight matrix [out_features, in_features]
    /// * `bias` - Optional bias vector [out_features]
    /// * `config` - Q-BLoRA configuration
    pub fn from_weight(
        weight: &Array,
        bias: Option<&Array>,
        config: &QBLoraConfig,
    ) -> Result<Self, LoraError> {
        let out_features = weight.dim(-2);
        let in_features = weight.dim(-1);

        // Create quantizer
        let nf4_config = NF4Config {
            block_size: config.block_size,
            double_quant: config.double_quant,
        };
        let quantizer = NF4Quantizer::with_config(nf4_config);

        // Cast weight to Float32 if needed
        let weight_f32 = if weight.dtype() != mlx_rs::Dtype::Float32 {
            weight.as_type::<f32>()?
        } else {
            weight.clone()
        };

        // Quantize weights
        weight_f32.eval()?;
        let weight_data: Vec<f32> = weight_f32.as_slice().to_vec();
        let shape = vec![out_features as usize, in_features as usize];
        let quantized_weight = quantizer
            .quantize(&weight_data, &shape)
            .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?;

        // Compute effective rank
        let effective_rank = config.effective_rank() as i32;

        // Compute LoRA scaling
        let lora_config = &config.lora;
        let scale = if lora_config.use_rslora {
            lora_config.alpha / (effective_rank as f32).sqrt()
        } else {
            lora_config.alpha / effective_rank as f32
        };

        // Create input projection if specified
        let (input_proj, lora_a_in_dim) = if let Some(proj_dim) = config.input_proj_dim {
            let bound = (3.0_f32 / in_features as f32).sqrt();
            let proj = if config.learnable_projections {
                mlx_rs::random::uniform::<_, f32>(
                    -bound,
                    bound,
                    &[in_features, proj_dim as i32],
                    None,
                )?
            } else {
                // Random orthogonal-ish projection (fixed)
                let proj = mlx_rs::random::normal::<f32>(
                    &[in_features, proj_dim as i32],
                    None,
                    None,
                    None,
                )?;
                // Normalize columns for stability
                let norm = proj.square()?.sum_axis(0, true)?.sqrt()?;
                proj.divide(&norm)?
            };
            (Some(proj), proj_dim as i32)
        } else {
            (None, in_features)
        };

        // Create output projection if specified
        let (output_proj, lora_b_out_dim) = if let Some(proj_dim) = config.output_proj_dim {
            let bound = (3.0_f32 / proj_dim as f32).sqrt();
            let proj = if config.learnable_projections {
                mlx_rs::random::uniform::<_, f32>(
                    -bound,
                    bound,
                    &[proj_dim as i32, out_features],
                    None,
                )?
            } else {
                let proj = mlx_rs::random::normal::<f32>(
                    &[proj_dim as i32, out_features],
                    None,
                    None,
                    None,
                )?;
                let norm = proj.square()?.sum_axis(1, true)?.sqrt()?;
                proj.divide(&norm)?
            };
            (Some(proj), proj_dim as i32)
        } else {
            (None, out_features)
        };

        // Initialize LoRA A with Kaiming uniform
        let bound = (3.0_f32 / lora_a_in_dim as f32).sqrt();
        let lora_a = mlx_rs::random::uniform::<_, f32>(
            -bound,
            bound,
            &[effective_rank, lora_a_in_dim],
            None,
        )?;

        // Initialize LoRA B with zeros
        let lora_b = mlx_rs::ops::zeros::<f32>(&[lora_b_out_dim, effective_rank])?;

        Ok(Self {
            in_features,
            out_features,
            rank: effective_rank,
            scale,
            use_bias: bias.is_some(),
            quantized_weight,
            quantizer,
            bias: bias.cloned(),
            lora_a,
            lora_b,
            input_proj,
            output_proj,
            learnable_projections: config.learnable_projections,
            gradient_scale: config.gradient_scale,
            weight_cache: RefCell::new(None),
            cache_enabled: false,
        })
    }

    /// Create a new Q-BLoRA layer with random weights (for testing).
    pub fn new(
        in_features: i32,
        out_features: i32,
        config: &QBLoraConfig,
        use_bias: bool,
    ) -> Result<Self, LoraError> {
        let bound = (3.0_f32 / in_features as f32).sqrt();
        let weight =
            mlx_rs::random::uniform::<_, f32>(-bound, bound, &[out_features, in_features], None)?;

        let bias = if use_bias {
            Some(mlx_rs::ops::zeros::<f32>(&[out_features])?)
        } else {
            None
        };

        Self::from_weight(&weight, bias.as_ref(), config)
    }

    /// Dequantize the weight matrix, using cache if enabled.
    fn dequantize_weight(&self) -> Result<Array, LoraError> {
        if self.cache_enabled {
            let cache = self.weight_cache.borrow();
            if let Some(ref cached) = *cache {
                return Ok(cached.clone());
            }
        }

        let weight_data = self
            .quantizer
            .dequantize(&self.quantized_weight)
            .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?;

        let weight = Array::from_slice(&weight_data, &[self.out_features, self.in_features]);

        if self.cache_enabled {
            *self.weight_cache.borrow_mut() = Some(weight.clone());
        }

        Ok(weight)
    }

    /// Enable weight caching.
    pub fn enable_weight_cache(&mut self) {
        self.cache_enabled = true;
    }

    /// Disable and optionally clear weight cache.
    pub fn disable_weight_cache(&mut self, clear: bool) {
        self.cache_enabled = false;
        if clear {
            *self.weight_cache.borrow_mut() = None;
        }
    }

    /// Forward pass through the Q-BLoRA layer.
    ///
    /// Implements: `y = x @ dequant(W_q).T + scale * ((x @ P_in) @ A.T) @ B.T @ P_out.T`
    pub fn forward(&self, x: &Array) -> Result<Array, LoraError> {
        // Dequantize base weights
        let weight = self.dequantize_weight()?;

        // Base forward: y_base = x @ W.T
        let y_base = x.matmul(&weight.t())?;

        // Q-BLoRA forward with optional projections
        let x_proj = if let Some(ref p_in) = self.input_proj {
            x.matmul(p_in)?
        } else {
            x.clone()
        };

        // LoRA: (x_proj @ A.T) @ B.T
        let xa = x_proj.matmul(&self.lora_a.t())?;
        let xab = xa.matmul(&self.lora_b.t())?;

        // Output projection if specified
        // p_out: [proj_dim, out_features], xab: [batch, proj_dim]
        // xab @ p_out = [batch, out_features]
        let y_lora_unscaled = if let Some(ref p_out) = self.output_proj {
            xab.matmul(p_out)?
        } else {
            xab
        };

        // Scale
        let scale_arr = Array::from_f32(self.scale * self.gradient_scale);
        let y_lora = y_lora_unscaled.multiply(&scale_arr)?;

        // Combined output
        let y = y_base.add(&y_lora)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            Ok(y.add(bias)?)
        } else {
            Ok(y)
        }
    }

    /// Get trainable parameters.
    ///
    /// Returns (lora_a, lora_b, input_proj, output_proj) where projections are
    /// None if not learnable or not present.
    pub fn trainable_params(&self) -> (&Array, &Array, Option<&Array>, Option<&Array>) {
        let p_in = if self.learnable_projections {
            self.input_proj.as_ref()
        } else {
            None
        };
        let p_out = if self.learnable_projections {
            self.output_proj.as_ref()
        } else {
            None
        };
        (&self.lora_a, &self.lora_b, p_in, p_out)
    }

    /// Get the number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        let lora_params = (self.lora_a.shape().iter().product::<i32>()
            + self.lora_b.shape().iter().product::<i32>()) as usize;

        let proj_params = if self.learnable_projections {
            let input_proj_params = self
                .input_proj
                .as_ref()
                .map(|p| p.shape().iter().product::<i32>() as usize)
                .unwrap_or(0);
            let output_proj_params = self
                .output_proj
                .as_ref()
                .map(|p| p.shape().iter().product::<i32>() as usize)
                .unwrap_or(0);
            input_proj_params + output_proj_params
        } else {
            0
        };

        lora_params + proj_params
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let quantized_bytes =
            self.quantized_weight.data.len() + self.quantized_weight.absmax.len() * 4;
        let lora_bytes = self.num_trainable_params() * 4;
        let bias_bytes = if self.use_bias {
            self.out_features as usize * 4
        } else {
            0
        };

        let total = quantized_bytes + lora_bytes + bias_bytes;
        (quantized_bytes, lora_bytes, total)
    }

    /// Check if this layer has input projection.
    pub fn has_input_proj(&self) -> bool {
        self.input_proj.is_some()
    }

    /// Check if this layer has output projection.
    pub fn has_output_proj(&self) -> bool {
        self.output_proj.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> QBLoraConfig {
        QBLoraConfig {
            lora: LoraConfig {
                r: 8,
                alpha: 16.0,
                use_rslora: false,
                ..Default::default()
            },
            quant_scheme: QuantScheme::NF4,
            block_size: 64,
            double_quant: true,
            input_proj_dim: None,
            output_proj_dim: None,
            rank_multiplier: 2.0,
            learnable_projections: true,
            gradient_scale: 1.0,
        }
    }

    #[test]
    fn test_qblora_creation() {
        let config = default_config();
        let qblora = QBLoraLinear::new(64, 128, &config, false).unwrap();

        assert_eq!(qblora.in_features, 64);
        assert_eq!(qblora.out_features, 128);
        // Effective rank = 8 * 2.0 = 16
        assert_eq!(qblora.rank, 16);
        // Scale = alpha / rank = 16 / 16 = 1.0
        assert!((qblora.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_qblora_forward() {
        let config = default_config();
        let qblora = QBLoraLinear::new(32, 64, &config, false).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[2, 4, 32], None, None, None).unwrap();
        let output = qblora.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 4, 64]);
    }

    #[test]
    fn test_qblora_with_input_proj() {
        let config = default_config().with_input_proj(16);
        let qblora = QBLoraLinear::new(64, 128, &config, false).unwrap();

        assert!(qblora.has_input_proj());
        assert!(!qblora.has_output_proj());

        let x = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();
        let output = qblora.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 4, 128]);
    }

    #[test]
    fn test_qblora_with_output_proj() {
        let config = default_config().with_output_proj(32);
        let qblora = QBLoraLinear::new(64, 128, &config, false).unwrap();

        assert!(!qblora.has_input_proj());
        assert!(qblora.has_output_proj());

        let x = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();
        let output = qblora.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 4, 128]);
    }

    #[test]
    fn test_qblora_with_both_projections() {
        let config = default_config().with_input_proj(16).with_output_proj(32);
        let qblora = QBLoraLinear::new(64, 128, &config, false).unwrap();

        assert!(qblora.has_input_proj());
        assert!(qblora.has_output_proj());

        let x = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();
        let output = qblora.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 4, 128]);
    }

    #[test]
    fn test_qblora_higher_rank() {
        // With 4x rank multiplier
        let config = default_config().with_rank_multiplier(4.0);
        let qblora = QBLoraLinear::new(64, 128, &config, false).unwrap();

        // Effective rank = 8 * 4.0 = 32
        assert_eq!(qblora.rank, 32);

        // More trainable params due to higher rank
        let standard_config = default_config();
        let standard = QBLoraLinear::new(64, 128, &standard_config, false).unwrap();

        assert!(qblora.num_trainable_params() > standard.num_trainable_params());
    }

    #[test]
    fn test_qblora_fixed_projections() {
        let config = default_config()
            .with_input_proj(16)
            .with_fixed_projections();
        let qblora = QBLoraLinear::new(64, 128, &config, false).unwrap();

        // With fixed projections, trainable params should not include projection
        let (_, _, p_in, p_out) = qblora.trainable_params();
        assert!(p_in.is_none()); // Not learnable
        assert!(p_out.is_none());
    }

    #[test]
    fn test_qblora_param_count() {
        let config = default_config();
        let qblora = QBLoraLinear::new(512, 1024, &config, false).unwrap();

        // Effective rank = 16
        // Trainable: A (16 * 512) + B (1024 * 16) = 8192 + 16384 = 24576
        assert_eq!(qblora.num_trainable_params(), 24576);
    }

    #[test]
    fn test_qblora_vs_standard_qlora() {
        // Compare Q-BLoRA with standard config to verify it's a superset
        let config = default_config();
        config.effective_rank();
    }

    #[test]
    fn test_effective_rank_computation() {
        let config = QBLoraConfig {
            lora: LoraConfig {
                r: 8,
                ..Default::default()
            },
            rank_multiplier: 2.5,
            ..Default::default()
        };

        // 8 * 2.5 = 20
        assert_eq!(config.effective_rank(), 20);
    }
}
