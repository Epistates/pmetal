//! FP8 Quantization support for memory-efficient training and inference.
//!
//! FP8 (8-bit floating point) provides ~2x memory reduction compared to FP16/BF16
//! with minimal accuracy loss for inference. MLX uses the E4M3 format:
//!
//! - **E4M3**: 4-bit exponent, 3-bit mantissa - range ~±240
//!
//! This module provides:
//! - Native FP8 weight quantization via MLX's `to_fp8`/`from_fp8` operations
//! - FP8 linear layers for inference
//! - Dynamic scaling for FP8 training
//!
//! # Example
//!
//! ```ignore
//! use pmetal_mlx::fp8_quantization::{Fp8Linear, Fp8Config};
//!
//! // Quantize weights for inference
//! let fp8_linear = Fp8Linear::from_weights(&weight, None, Fp8Config::default())?;
//!
//! // Run inference (uses native FP8 operations)
//! let output = fp8_linear.forward(&input)?;
//! ```

use mlx_rs::error::Exception;
use mlx_rs::ops::{from_fp8, to_fp8};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

/// FP8 format type.
///
/// MLX currently supports E4M3 format natively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fp8Format {
    /// E4M3 format: 4-bit exponent, 3-bit mantissa.
    /// Range: ~±240, Best for weights.
    E4M3,
    /// E5M2 format: 5-bit exponent, 2-bit mantissa.
    /// Range: ~±57344, Best for activations.
    /// Note: Currently uses E4M3 internally as MLX only supports E4M3.
    E5M2,
}

impl Default for Fp8Format {
    fn default() -> Self {
        Self::E4M3
    }
}

impl Fp8Format {
    /// Maximum representable value for this format.
    pub fn max_value(&self) -> f32 {
        match self {
            Self::E4M3 => 240.0,
            Self::E5M2 => 57344.0,
        }
    }

    /// Epsilon value for numerical stability.
    pub fn epsilon(&self) -> f32 {
        match self {
            Self::E4M3 => 0.0625,       // 2^-4
            Self::E5M2 => 0.0009765625, // 2^-10
        }
    }
}

/// Configuration for FP8 quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp8Config {
    /// Format for weight quantization.
    pub weight_format: Fp8Format,
    /// Format for activation quantization.
    pub activation_format: Fp8Format,
    /// Whether to use per-tensor (false) or per-channel (true) scaling.
    pub per_channel: bool,
    /// Whether to use dynamic scaling during inference.
    pub dynamic_scaling: bool,
    /// Margin for amax computation (for stability).
    pub amax_margin: f32,
}

impl Default for Fp8Config {
    fn default() -> Self {
        Self {
            weight_format: Fp8Format::E4M3,
            activation_format: Fp8Format::E5M2,
            per_channel: false,
            dynamic_scaling: true,
            amax_margin: 0.01,
        }
    }
}

/// Quantized FP8 tensor using native MLX FP8 operations.
///
/// Data is stored as uint8 in E4M3 format using MLX's native `to_fp8` operation.
#[derive(Debug, Clone)]
pub struct Fp8Tensor {
    /// The quantized data (stored as uint8 in E4M3 format).
    pub data: Array,
    /// Original shape for reference.
    pub shape: Vec<i32>,
    /// Format used for quantization.
    pub format: Fp8Format,
}

impl Fp8Tensor {
    /// Quantize a tensor to FP8 using native MLX operations.
    ///
    /// Uses MLX's `to_fp8` which converts to E4M3 format stored as uint8.
    pub fn quantize(x: &Array, format: Fp8Format, _per_channel: bool) -> Result<Self, Exception> {
        // Use native MLX FP8 conversion
        let data = to_fp8(x)?;

        Ok(Self {
            data,
            shape: x.shape().to_vec(),
            format,
        })
    }

    /// Dequantize the FP8 tensor back to full precision.
    ///
    /// Uses MLX's `from_fp8` for native conversion.
    pub fn dequantize(&self) -> Result<Array, Exception> {
        from_fp8(&self.data, mlx_rs::Dtype::Float32)
    }

    /// Dequantize to bfloat16 for efficient computation.
    pub fn dequantize_bf16(&self) -> Result<Array, Exception> {
        from_fp8(&self.data, mlx_rs::Dtype::Bfloat16)
    }

    /// Get the quantized data.
    pub fn data(&self) -> &Array {
        &self.data
    }

    /// Memory size in bytes (1 byte per element for FP8).
    pub fn memory_bytes(&self) -> usize {
        self.data.size()
    }
}

/// FP8 Linear layer for inference using native MLX FP8 operations.
///
/// Weights are stored in native FP8 E4M3 format using MLX's `to_fp8`.
/// Computation dequantizes to BF16 for matmul, providing ~2x memory savings.
#[derive(Debug, Clone)]
pub struct Fp8Linear {
    /// Quantized weights in FP8 format.
    pub weight: Fp8Tensor,
    /// Optional bias (kept in full precision).
    pub bias: Option<Array>,
    /// Configuration.
    pub config: Fp8Config,
}

impl Fp8Linear {
    /// Create from a standard linear layer's weights.
    ///
    /// Weights are quantized to FP8 E4M3 format using native MLX operations.
    pub fn from_weights(
        weight: &Array,
        bias: Option<&Array>,
        config: Fp8Config,
    ) -> Result<Self, Exception> {
        let quantized_weight =
            Fp8Tensor::quantize(weight, config.weight_format, config.per_channel)?;

        Ok(Self {
            weight: quantized_weight,
            bias: bias.cloned(),
            config,
        })
    }

    /// Forward pass with FP8 weights.
    ///
    /// Dequantizes weights to BF16 for the matmul operation.
    /// This provides memory savings while maintaining computation precision.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // Dequantize weights to BF16 for computation
        let weight = self.weight.dequantize_bf16()?;
        let x_bf16 = x.as_dtype(mlx_rs::Dtype::Bfloat16)?;

        // matmul: x @ weight.T
        let weight_t = weight.t();
        let mut output = x_bf16.matmul(&weight_t)?;

        if let Some(ref bias) = self.bias {
            output = output.add(bias)?;
        }

        Ok(output)
    }

    /// Memory size in bytes.
    pub fn memory_bytes(&self) -> usize {
        let weight_bytes = self.weight.memory_bytes();
        let bias_bytes = self.bias.as_ref().map(|b| b.size() * 4).unwrap_or(0);
        weight_bytes + bias_bytes
    }

    /// Calculate memory savings compared to FP16 weights.
    pub fn memory_savings(&self) -> f32 {
        // FP8 = 1 byte, FP16 = 2 bytes, so 50% savings
        0.5
    }
}

/// Dynamic scaling context for FP8 training.
///
/// Tracks activation/gradient statistics for determining optimal scale factors.
#[derive(Debug, Clone)]
pub struct Fp8DynamicScaling {
    /// Window size for amax history.
    pub window_size: usize,
    /// History of amax values for activations.
    amax_history_activation: Vec<f32>,
    /// History of amax values for gradients.
    amax_history_gradient: Vec<f32>,
    /// Current scale for activations.
    pub activation_scale: f32,
    /// Current scale for gradients.
    pub gradient_scale: f32,
}

impl Default for Fp8DynamicScaling {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl Fp8DynamicScaling {
    /// Create a new dynamic scaling context.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            amax_history_activation: Vec::with_capacity(window_size),
            amax_history_gradient: Vec::with_capacity(window_size),
            activation_scale: 1.0,
            gradient_scale: 1.0,
        }
    }

    /// Update activation scale with new amax.
    pub fn update_activation(&mut self, amax: f32, format: Fp8Format) {
        self.amax_history_activation.push(amax);
        if self.amax_history_activation.len() > self.window_size {
            self.amax_history_activation.remove(0);
        }

        // Compute scale from max of history
        let max_amax = self
            .amax_history_activation
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        self.activation_scale = format.max_value() / max_amax.max(1e-12);
    }

    /// Update gradient scale with new amax.
    pub fn update_gradient(&mut self, amax: f32, format: Fp8Format) {
        self.amax_history_gradient.push(amax);
        if self.amax_history_gradient.len() > self.window_size {
            self.amax_history_gradient.remove(0);
        }

        let max_amax = self
            .amax_history_gradient
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        self.gradient_scale = format.max_value() / max_amax.max(1e-12);
    }
}

/// Quantize a model's weights to FP8 for inference.
///
/// This function takes a model's parameter map and returns a map of FP8 tensors.
pub fn quantize_weights_fp8(
    weights: &std::collections::HashMap<std::rc::Rc<str>, Array>,
    config: &Fp8Config,
) -> Result<std::collections::HashMap<std::rc::Rc<str>, Fp8Tensor>, Exception> {
    weights
        .iter()
        .map(|(name, tensor)| {
            let quantized = Fp8Tensor::quantize(tensor, config.weight_format, config.per_channel)?;
            Ok((name.clone(), quantized))
        })
        .collect()
}

/// Calculate memory savings from FP8 quantization.
pub fn calculate_fp8_savings(original_size_bytes: usize, dtype_bits: usize) -> (usize, f32) {
    // Native FP8 = exactly 8 bits per element
    let fp8_bits = 8;
    let fp8_size = (original_size_bytes * fp8_bits) / dtype_bits;
    let savings = 1.0 - (fp8_size as f32 / original_size_bytes as f32);
    (fp8_size, savings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_format() {
        assert_eq!(Fp8Format::E4M3.max_value(), 240.0);
        assert_eq!(Fp8Format::E5M2.max_value(), 57344.0);
    }

    #[test]
    fn test_native_fp8_quantize_dequantize() {
        let x = mlx_rs::random::normal::<f32>(&[4, 4], None, None, None).unwrap();

        let quantized = Fp8Tensor::quantize(&x, Fp8Format::E4M3, false).unwrap();

        // FP8 data should be uint8
        assert_eq!(quantized.data.dtype(), mlx_rs::Dtype::Uint8);

        let dequantized = quantized.dequantize().unwrap();

        // Check shape preserved
        assert_eq!(x.shape(), dequantized.shape());

        // Evaluate to check no errors
        x.eval().unwrap();
        dequantized.eval().unwrap();
    }

    #[test]
    fn test_fp8_linear() {
        let weight = mlx_rs::random::normal::<f32>(&[16, 8], None, None, None).unwrap();
        let config = Fp8Config::default();

        let fp8_linear = Fp8Linear::from_weights(&weight, None, config).unwrap();

        // Verify weight is in FP8 format
        assert_eq!(fp8_linear.weight.data.dtype(), mlx_rs::Dtype::Uint8);

        let x = mlx_rs::random::normal::<f32>(&[2, 8], None, None, None).unwrap();
        let output = fp8_linear.forward(&x).unwrap();
        output.eval().unwrap();

        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_fp8_memory_savings() {
        let weight = mlx_rs::random::normal::<f32>(&[1024, 1024], None, None, None).unwrap();
        let config = Fp8Config::default();

        let fp8_linear = Fp8Linear::from_weights(&weight, None, config).unwrap();

        // FP8 should use 1 byte per element vs 4 bytes for f32
        let fp8_bytes = fp8_linear.memory_bytes();
        let f32_bytes = 1024 * 1024 * 4;

        // Should be ~75% savings (1 byte vs 4 bytes)
        assert!(fp8_bytes < f32_bytes / 2);
    }

    #[test]
    fn test_dynamic_scaling() {
        let mut scaling = Fp8DynamicScaling::new(10);

        for i in 1..20 {
            scaling.update_activation(i as f32, Fp8Format::E5M2);
        }

        // Scale should be based on recent max (19)
        let expected_scale = Fp8Format::E5M2.max_value() / 19.0;
        assert!((scaling.activation_scale - expected_scale).abs() < 1.0);
    }

    #[test]
    fn test_memory_savings_calculation() {
        // F32 to FP8: 4 bytes to 1 byte = 75% savings
        let (fp8_size, savings) = calculate_fp8_savings(1000, 32);
        assert_eq!(fp8_size, 250); // 1000 * 8 / 32 = 250
        assert!((savings - 0.75).abs() < 0.01);

        // F16 to FP8: 2 bytes to 1 byte = 50% savings
        let (fp8_size, savings) = calculate_fp8_savings(1000, 16);
        assert_eq!(fp8_size, 500); // 1000 * 8 / 16 = 500
        assert!((savings - 0.5).abs() < 0.01);
    }
}
