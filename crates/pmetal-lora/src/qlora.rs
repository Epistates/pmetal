//! QLoRA (Quantized LoRA) implementation.
//!
//! QLoRA enables memory-efficient fine-tuning by:
//! - Storing base weights in 4-bit NF4 (the default), 4-bit FP4 (E2M1) or 8-bit int8,
//!   each with a per-block absmax scale
//! - Keeping LoRA adapters A and B in full precision (trainable)
//! - Dequantizing base weights on-the-fly during forward pass
//! - Optional dequantization caching for frozen weights (avoids redundant computation per forward pass)
//!
//! Memory savings for a 7B model:
//! - Full precision: 28 GB
//! - QLoRA (NF4): ~4 GB
//!
//! # Performance Optimizations
//!
//! When caching is enabled via `enable_weight_cache()`, dequantized weights are cached
//! in memory. This trades memory for speed during training when base weights are frozen.
//! The cache can be cleared with `clear_weight_cache()` after training.
//!
//! Reference: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)

use std::cell::RefCell;

use pmetal_bridge::compat::{Array, Exception};
use pmetal_core::LoraConfig;
use pmetal_mlx::quantization::{
    FP4Config, FP4Quantizer, Int8Config, Int8Quantizer, NF4Config, NF4Quantizer, QuantScheme,
    QuantizedTensor, QuantizerOps,
};

use super::LoraError;

/// QLoRA configuration extending standard LoRA config with quantization settings.
#[derive(Debug, Clone)]
pub struct QLoraConfig {
    /// Base LoRA configuration.
    pub lora: LoraConfig,
    /// Quantization scheme (NF4, FP4, Int8).
    pub quant_scheme: QuantScheme,
    /// Block size for quantization (default: 64).
    pub block_size: usize,
    /// Enable double quantization for absmax values.
    pub double_quant: bool,
    /// Compute dtype for dequantized weights during forward pass.
    /// Note: MLX uses f32/f16 automatically based on array dtype.
    pub compute_in_half: bool,
}

impl Default for QLoraConfig {
    fn default() -> Self {
        Self {
            lora: LoraConfig::default(),
            quant_scheme: QuantScheme::NF4,
            block_size: 64,
            double_quant: true,
            compute_in_half: true,
        }
    }
}

impl QLoraConfig {
    /// Create QLoRA config from existing LoRA config.
    pub fn from_lora(lora: LoraConfig) -> Self {
        Self {
            lora,
            ..Default::default()
        }
    }

    /// Set quantization scheme.
    pub fn with_scheme(mut self, scheme: QuantScheme) -> Self {
        self.quant_scheme = scheme;
        self
    }

    /// Set block size for quantization.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Enable or disable double quantization.
    pub fn with_double_quant(mut self, double_quant: bool) -> Self {
        self.double_quant = double_quant;
        self
    }
}

/// The quantizer that stores a QLoRA layer's frozen base weight.
#[derive(Debug, Clone)]
enum BaseQuantizer {
    Nf4(NF4Quantizer),
    Fp4(FP4Quantizer),
    Int8(Int8Quantizer),
}

impl BaseQuantizer {
    /// The quantizer `config.quant_scheme` names, or an error naming a scheme
    /// that can't hold a base weight.
    fn for_config(config: &QLoraConfig) -> Result<Self, LoraError> {
        let block_size = config.block_size;
        match config.quant_scheme {
            QuantScheme::NF4 => Ok(Self::Nf4(NF4Quantizer::with_config(NF4Config {
                block_size,
                double_quant: config.double_quant,
            }))),
            QuantScheme::FP4 => Ok(Self::Fp4(FP4Quantizer::with_config(FP4Config {
                block_size,
            }))),
            QuantScheme::Int8 => Ok(Self::Int8(Int8Quantizer::with_config(Int8Config {
                block_size,
                symmetric: true,
            }))),
            other => Err(LoraError::InvalidState(format!(
                "QLoRA can store base weights as nf4, fp4 or int8, not {other:?}"
            ))),
        }
    }

    fn ops(&self) -> &dyn QuantizerOps {
        match self {
            Self::Nf4(q) => q,
            Self::Fp4(q) => q,
            Self::Int8(q) => q,
        }
    }
}

/// QLoRA Linear layer with quantized base weights and full-precision LoRA adapters.
///
/// Implements: `y = x @ dequant(W_q).T + scale * (x @ A.T) @ B.T`
///
/// Where:
/// - `W_q` is the quantized base weight (NF4, FP4 or int8, per `QLoraConfig::quant_scheme`)
/// - `dequant(W_q)` dequantizes to full precision on-the-fly (or cached)
/// - `A` is the LoRA down-projection (trainable, full precision)
/// - `B` is the LoRA up-projection (trainable, full precision)
///
/// # Weight Caching
///
/// Since base weights are frozen during LoRA training, dequantization can be cached
/// to avoid redundant computation on each forward pass. Enable with `enable_weight_cache()`.
pub struct QLoraLinear {
    /// Input features dimension.
    pub in_features: i32,
    /// Output features dimension.
    pub out_features: i32,
    /// LoRA rank.
    pub rank: i32,
    /// LoRA scaling factor (alpha / rank).
    pub scale: f32,
    /// Whether to use bias.
    pub use_bias: bool,

    /// Quantized base weight.
    pub quantized_weight: QuantizedTensor,
    /// Quantizer for dequantization, the one `QLoraConfig::quant_scheme` names.
    quantizer: BaseQuantizer,
    /// Optional bias [out_features] - kept in full precision.
    pub bias: Option<Array>,
    /// LoRA A matrix [rank, in_features] - trainable, full precision.
    pub lora_a: Array,
    /// LoRA B matrix [out_features, rank] - trainable, full precision.
    pub lora_b: Array,

    /// Cached dequantized weight (optional, for training performance).
    /// Uses RefCell for interior mutability to allow caching in immutable forward().
    ///
    // SAFETY: RefCell<Option<Array>> is not Sync. This type must only be used from a
    // single thread. MLX training runs on a single thread with GPU dispatch, so this
    // is safe in practice. If multi-threaded access is ever needed, replace with
    // Mutex<Option<Array>> or use thread-local storage.
    weight_cache: RefCell<Option<Array>>,
    /// Whether weight caching is enabled.
    cache_enabled: bool,
    /// Whether in training mode (controls dropout).
    pub training: bool,
    /// LoRA dropout probability.
    pub lora_dropout: f32,
}

impl std::fmt::Debug for QLoraLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QLoraLinear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("rank", &self.rank)
            .field("scale", &self.scale)
            .field("use_bias", &self.use_bias)
            .field("cache_enabled", &self.cache_enabled)
            .field("weight_cached", &self.weight_cache.borrow().is_some())
            .finish()
    }
}

impl QLoraLinear {
    /// Create a QLoRA layer by quantizing an existing weight matrix.
    ///
    /// # Arguments
    /// * `weight` - Full-precision weight matrix [out_features, in_features].
    ///   Supports Float32, Float16, and BFloat16 input (will be cast to Float32)
    /// * `bias` - Optional bias vector [out_features]
    /// * `config` - QLoRA configuration
    pub fn from_weight(
        weight: &Array,
        bias: Option<&Array>,
        config: &QLoraConfig,
    ) -> Result<Self, LoraError> {
        let out_features = weight.dim(-2);
        let in_features = weight.dim(-1);

        let quantizer = BaseQuantizer::for_config(config)?;

        // Cast weight to Float32 if needed (handles BFloat16, Float16, etc.)
        // This is necessary because models like Qwen3 use BFloat16 weights
        let weight_f32 = if weight.dtype() != pmetal_bridge::compat::Dtype::Float32 {
            weight.as_type::<f32>()
        } else {
            weight.clone()
        };

        // Quantize weights
        weight_f32.eval();
        let weight_data: Vec<f32> = weight_f32.as_slice().to_vec();
        let shape = vec![out_features as usize, in_features as usize];
        let quantized_weight = quantizer
            .ops()
            .quantize(&weight_data, &shape)
            .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?;

        // Compute LoRA scaling
        let lora_config = &config.lora;
        let rank = lora_config.r as i32;
        let scale = if rank > 0 {
            if lora_config.use_rslora {
                lora_config.alpha / (rank as f32).sqrt()
            } else {
                lora_config.alpha / rank as f32
            }
        } else {
            0.0
        };

        // Initialize LoRA A with Kaiming uniform (dummy when rank=0)
        let lora_a = if rank > 0 {
            let bound = (3.0_f32 / in_features as f32).sqrt();
            pmetal_bridge::compat::random::uniform_range(
                -bound,
                bound,
                &[rank, in_features],
                pmetal_bridge::compat::Dtype::Float32,
            )
        } else {
            pmetal_bridge::compat::ops::zeros(
                &[1, in_features],
                pmetal_bridge::compat::Dtype::Float32,
            )
        };

        // Initialize LoRA B with zeros (dummy when rank=0)
        let lora_b = if rank > 0 {
            pmetal_bridge::compat::ops::zeros(
                &[out_features, rank],
                pmetal_bridge::compat::Dtype::Float32,
            )
        } else {
            pmetal_bridge::compat::ops::zeros(
                &[out_features, 1],
                pmetal_bridge::compat::Dtype::Float32,
            )
        };

        Ok(Self {
            in_features,
            out_features,
            rank,
            scale,
            use_bias: bias.is_some(),
            quantized_weight,
            quantizer,
            bias: bias.cloned(),
            lora_a,
            lora_b,
            weight_cache: RefCell::new(None),
            cache_enabled: false,
            training: false,
            lora_dropout: 0.0,
        })
    }

    /// Create a new QLoRA layer with random weights (for testing).
    pub fn new(
        in_features: i32,
        out_features: i32,
        config: &QLoraConfig,
        use_bias: bool,
    ) -> Result<Self, LoraError> {
        // Create random weight
        let bound = (3.0_f32 / in_features as f32).sqrt();
        let weight = pmetal_bridge::compat::random::uniform_range(
            -bound,
            bound,
            &[out_features, in_features],
            pmetal_bridge::compat::Dtype::Float32,
        );

        // Create random bias if needed
        let bias = if use_bias {
            Some(pmetal_bridge::compat::ops::zeros(
                &[out_features],
                pmetal_bridge::compat::Dtype::Float32,
            ))
        } else {
            None
        };

        Self::from_weight(&weight, bias.as_ref(), config)
    }

    /// Dequantize the weight matrix, using cache if enabled.
    ///
    /// When caching is enabled, the dequantized weight is stored and reused
    /// on subsequent forward passes. This is beneficial during training when
    /// base weights are frozen.
    fn dequantize_weight(&self) -> Result<Array, LoraError> {
        // Check cache first if enabled
        if self.cache_enabled {
            let cache = self.weight_cache.borrow();
            if let Some(ref cached) = *cache {
                return Ok(cached.clone());
            }
        }

        // Dequantize
        let weight_data = self
            .quantizer
            .ops()
            .dequantize(&self.quantized_weight)
            .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?;

        let weight = Array::from_slice(&weight_data, &[self.out_features, self.in_features]);

        // Store in cache if enabled
        if self.cache_enabled {
            *self.weight_cache.borrow_mut() = Some(weight.clone());
        }

        Ok(weight)
    }

    /// Enable weight caching for faster forward passes.
    ///
    /// When enabled, dequantized base weights are cached in memory.
    /// This trades memory for speed during training when weights are frozen.
    ///
    /// # Memory Impact
    /// Adds `out_features * in_features * 4` bytes (fp32) per layer.
    pub fn enable_weight_cache(&mut self) {
        self.cache_enabled = true;
    }

    /// Disable weight caching and optionally clear cached weights.
    pub fn disable_weight_cache(&mut self, clear: bool) {
        self.cache_enabled = false;
        if clear {
            self.clear_weight_cache();
        }
    }

    /// Clear cached dequantized weights to free memory.
    ///
    /// Call this after training to reclaim memory used by the cache.
    pub fn clear_weight_cache(&mut self) {
        *self.weight_cache.borrow_mut() = None;
    }

    /// Check if weight caching is enabled.
    pub fn is_cache_enabled(&self) -> bool {
        self.cache_enabled
    }

    /// Check if weights are currently cached.
    pub fn is_weight_cached(&self) -> bool {
        self.weight_cache.borrow().is_some()
    }

    /// Pre-warm the cache by dequantizing weights.
    ///
    /// Call this before training to ensure dequantized weights are cached
    /// before the first forward pass. Requires caching to be enabled.
    pub fn warm_cache(&self) -> Result<(), LoraError> {
        if self.cache_enabled && self.weight_cache.borrow().is_none() {
            let _ = self.dequantize_weight()?;
        }
        Ok(())
    }

    /// Forward pass through the QLoRA layer.
    ///
    /// Implements: `y = x @ dequant(W_q).T + scale * (x @ A.T) @ B.T`
    ///
    /// The base weights are dequantized on-the-fly and immediately discarded,
    /// keeping peak memory usage low.
    ///
    /// When `rank == 0` the layer acts as a frozen quantized linear — only the
    /// base weight is applied, skipping the LoRA path entirely.
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        // Dequantize base weights on-the-fly
        let weight = self.dequantize_weight()?;

        // Base forward: y_base = x @ W.T
        let y_base = x.matmul(&weight.t());

        // Skip LoRA path when rank=0 (frozen module not in target_modules)
        let y = if self.rank == 0 {
            y_base
        } else {
            // Apply dropout to LoRA input when training
            let x_lora = if self.training && self.lora_dropout > 0.0 {
                crate::lora::apply_dropout(x, self.lora_dropout)?
            } else {
                x.clone()
            };

            // LoRA forward: y_lora = scale * (x_lora @ A.T) @ B.T
            let xa = x_lora.matmul(&self.lora_a.t());
            let xab = xa.matmul(&self.lora_b.t());
            let scale_arr = Array::from_f32(self.scale);
            let y_lora = xab.multiply(&scale_arr);

            // Combined output
            y_base.add(&y_lora)
        };

        // Add bias if present
        if let Some(ref bias) = self.bias {
            Ok(y.add(bias))
        } else {
            Ok(y)
        }
    }

    /// Merge LoRA weights into the base weight.
    ///
    /// Returns `dequant(W) + scale * B @ A` as a full-precision array.
    pub fn merge(&self) -> Result<Array, LoraError> {
        let weight = self.dequantize_weight()?;
        let ba = self.lora_b.matmul(&self.lora_a);
        let scale_arr = Array::from_f32(self.scale);
        let update = ba.multiply(&scale_arr);
        Ok(weight.add(&update))
    }

    /// Set training mode (controls dropout).
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

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

    /// Get the number of trainable parameters (LoRA A + B).
    pub fn num_trainable_params(&self) -> usize {
        let lora_a_params = (self.rank * self.in_features) as usize;
        let lora_b_params = (self.out_features * self.rank) as usize;
        lora_a_params + lora_b_params
    }

    /// Get the number of frozen parameters (quantized weight + bias).
    pub fn num_frozen_params(&self) -> usize {
        let weight_params = (self.out_features * self.in_features) as usize;
        let bias_params = if self.use_bias {
            self.out_features as usize
        } else {
            0
        };
        weight_params + bias_params
    }

    /// Get memory usage in bytes.
    ///
    /// Returns (quantized_bytes, lora_bytes, total_bytes)
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        // Quantized weight: one byte per int8 value, two per byte for the 4-bit schemes
        let absmax_bytes = if self.quantized_weight.absmax_quant.is_some() {
            // Double quantization: 1 byte per absmax + 8 bytes overhead (offset + scale)
            self.quantized_weight.absmax.len() + 8
        } else {
            // Standard: f32 per absmax
            self.quantized_weight.absmax.len() * 4
        };
        let quantized_bytes = self.quantized_weight.data.len() + absmax_bytes;

        // LoRA params in f32
        let lora_bytes = self.num_trainable_params() * 4;

        // Bias if present
        let bias_bytes = if self.use_bias {
            self.out_features as usize * 4
        } else {
            0
        };

        let total = quantized_bytes + lora_bytes + bias_bytes;
        (quantized_bytes, lora_bytes, total)
    }

    /// Get memory savings compared to full-precision LoRA.
    ///
    /// Returns the ratio of QLoRA memory to full-precision memory.
    pub fn memory_savings(&self) -> f32 {
        let (quantized_bytes, lora_bytes, _) = self.memory_usage();
        let full_precision_bytes = self.num_frozen_params() * 4 + lora_bytes;

        (quantized_bytes + lora_bytes) as f32 / full_precision_bytes as f32
    }

    /// Get the quantization scheme used.
    pub fn quant_scheme(&self) -> QuantScheme {
        self.quantized_weight.scheme
    }
}

/// Create a QLoRA layer from a standard LoRA layer by quantizing the base weights.
pub fn quantize_lora_layer(
    lora: &super::LoraLinear,
    config: &QLoraConfig,
) -> Result<QLoraLinear, LoraError> {
    QLoraLinear::from_weight(&lora.weight, lora.bias.as_ref(), config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> QLoraConfig {
        QLoraConfig {
            lora: LoraConfig {
                r: 8,
                alpha: 16.0,
                use_rslora: false,
                ..Default::default()
            },
            quant_scheme: QuantScheme::NF4,
            block_size: 64,
            double_quant: true,
            compute_in_half: false,
        }
    }

    #[test]
    fn test_qlora_linear_creation() {
        let config = default_config();
        let qlora = QLoraLinear::new(64, 128, &config, false).unwrap();

        assert_eq!(qlora.in_features, 64);
        assert_eq!(qlora.out_features, 128);
        assert_eq!(qlora.rank, 8);
        assert!((qlora.scale - 2.0).abs() < 1e-6); // alpha / rank = 16 / 8 = 2
    }

    #[test]
    fn test_qlora_forward() {
        let config = default_config();
        let mut qlora = QLoraLinear::new(32, 64, &config, false).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[2, 4, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = qlora.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 4, 64]);
    }

    #[test]
    fn test_qlora_with_bias() {
        let config = default_config();
        let mut qlora = QLoraLinear::new(32, 64, &config, true).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[2, 4, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = qlora.forward(&x).unwrap();

        assert_eq!(output.shape(), &[2, 4, 64]);
        assert!(qlora.bias.is_some());
    }

    #[test]
    fn test_qlora_zero_lora_contribution() {
        // With B initialized to zeros, LoRA should have minimal effect
        let config = default_config();
        let mut qlora = QLoraLinear::new(32, 64, &config, false).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = qlora.forward(&x).unwrap();

        // Get base output (dequantized weight only)
        let weight = qlora.dequantize_weight().unwrap();
        let base_output = x.matmul(&weight.t()).unwrap();

        output.eval().unwrap();
        base_output.eval().unwrap();

        // Outputs should be close since B is zeros
        let diff = output.subtract(&base_output).unwrap();
        let max_diff = diff.abs().unwrap().max(None).unwrap();
        max_diff.eval().unwrap();
        assert!(max_diff.item::<f32>() < 1e-5);
    }

    #[test]
    fn test_qlora_memory_savings() {
        let config = default_config();
        let qlora = QLoraLinear::new(512, 1024, &config, false).unwrap();

        let savings = qlora.memory_savings();

        // NF4 should give roughly 8x compression for weights
        // But LoRA params are in full precision, so overall savings is less
        // Expected: (quantized_weight + lora) / (full_weight + lora)
        // quantized ~= full / 8, lora stays same
        assert!(
            savings < 0.3,
            "Expected significant memory savings, got {}",
            savings
        );
    }

    #[test]
    fn test_qlora_param_count() {
        let config = default_config();
        let qlora = QLoraLinear::new(512, 1024, &config, false).unwrap();

        // Trainable: A (8 * 512) + B (1024 * 8) = 4096 + 8192 = 12288
        assert_eq!(qlora.num_trainable_params(), 12288);

        // Frozen: W (1024 * 512) = 524288
        assert_eq!(qlora.num_frozen_params(), 524288);
    }

    #[test]
    fn test_qlora_quantization_accuracy() {
        // Test that quantization doesn't introduce too much error
        let config = default_config();

        // Create a layer with known weights
        let in_f = 64;
        let out_f = 128;
        let weight = pmetal_bridge::compat::random::normal(
            &[out_f, in_f],
            pmetal_bridge::compat::Dtype::Float32,
        );
        weight.eval().unwrap();

        let qlora = QLoraLinear::from_weight(&weight, None, &config).unwrap();

        // Dequantize and compare
        let dequantized = qlora.dequantize_weight().unwrap();
        dequantized.eval().unwrap();

        let diff = weight.subtract(&dequantized).unwrap();
        let mean_error = diff.abs().unwrap().mean(None).unwrap();
        mean_error.eval().unwrap();

        // NF4 should have small quantization error for normally distributed weights
        assert!(
            mean_error.item::<f32>() < 0.1,
            "Mean quantization error too high: {}",
            mean_error.item::<f32>()
        );
    }

    #[test]
    fn test_qlora_weight_caching() {
        let config = default_config();
        let mut qlora = QLoraLinear::new(64, 128, &config, false).unwrap();

        // Initially cache should be disabled
        assert!(!qlora.is_cache_enabled());
        assert!(!qlora.is_weight_cached());

        // Enable caching
        qlora.enable_weight_cache();
        assert!(qlora.is_cache_enabled());

        // First forward should populate cache
        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output1 = qlora.forward(&x).unwrap();
        output1.eval().unwrap();

        assert!(qlora.is_weight_cached());

        // Second forward should use cache (same result)
        let output2 = qlora.forward(&x).unwrap();
        output2.eval().unwrap();

        // Results should be identical
        let diff = output1
            .subtract(&output2)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        diff.eval().unwrap();
        assert!(diff.item::<f32>() < 1e-10);

        // Clear cache
        qlora.clear_weight_cache();
        assert!(!qlora.is_weight_cached());
        assert!(qlora.is_cache_enabled()); // Still enabled, just cleared

        // Disable caching
        qlora.disable_weight_cache(true);
        assert!(!qlora.is_cache_enabled());
    }

    #[test]
    fn test_qlora_warm_cache() {
        let config = default_config();
        let mut qlora = QLoraLinear::new(64, 128, &config, false).unwrap();

        qlora.enable_weight_cache();
        assert!(!qlora.is_weight_cached());

        // Warm the cache
        qlora.warm_cache().unwrap();
        assert!(qlora.is_weight_cached());
    }

    /// Largest |dequant(W_q) − W| over a fixed weight stored with `scheme`.
    fn base_weight_error(scheme: QuantScheme) -> (QuantScheme, usize, f32) {
        let (out_f, in_f) = (32, 64);
        let values: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i as f32) * 0.618).sin() * 0.05)
            .collect();
        let weight = Array::from_slice(&values, &[out_f as i32, in_f as i32]);
        let config = QLoraConfig {
            quant_scheme: scheme,
            double_quant: false,
            ..default_config()
        };
        let qlora = QLoraLinear::from_weight(&weight, None, &config).unwrap();
        let back = qlora.dequantize_weight().unwrap();
        back.eval();
        pmetal_bridge::check_last_error().unwrap();
        let max_err = back
            .as_slice::<f32>()
            .iter()
            .zip(values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        (
            qlora.quant_scheme(),
            qlora.quantized_weight.data.len(),
            max_err,
        )
    }

    /// `--quantization fp4` and `int8` reach `QLoraConfig::quant_scheme`; the
    /// layer used to quantize with NF4 whatever it said.
    #[test]
    fn base_weight_is_stored_in_the_configured_scheme() {
        let n = 32 * 64;

        let (scheme, bytes, err) = base_weight_error(QuantScheme::NF4);
        assert_eq!((scheme, bytes), (QuantScheme::NF4, n / 2));
        let nf4_err = err;

        let (scheme, bytes, err) = base_weight_error(QuantScheme::FP4);
        assert_eq!((scheme, bytes), (QuantScheme::FP4, n / 2));
        // E2M1's widest gap is between 4 and 6 of the block's 6: a third of absmax.
        assert!(err <= 0.05 / 6.0 + 1e-6, "fp4 error {err}");

        let (scheme, bytes, err) = base_weight_error(QuantScheme::Int8);
        assert_eq!((scheme, bytes), (QuantScheme::Int8, n));
        // Half of an int8 step at absmax 0.05.
        assert!(err <= 0.05 / 254.0 + 1e-6, "int8 error {err}");
        assert!(err < nf4_err, "int8 ({err}) should beat nf4 ({nf4_err})");
    }

    #[test]
    fn schemes_that_cannot_hold_a_base_weight_are_refused_by_name() {
        let config = QLoraConfig {
            quant_scheme: QuantScheme::FP8,
            ..default_config()
        };
        let err = QLoraLinear::new(16, 16, &config, false).unwrap_err();
        assert!(err.to_string().contains("FP8"), "{err}");
    }
}
