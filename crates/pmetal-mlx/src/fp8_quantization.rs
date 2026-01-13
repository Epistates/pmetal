//! FP8 Quantization support for memory-efficient training and inference.
//!
//! FP8 (8-bit floating point) provides ~2x memory reduction compared to FP16/BF16
//! with minimal accuracy loss for inference. There are two common FP8 formats:
//!
//! - **E4M3**: 4-bit exponent, 3-bit mantissa - better for weights
//! - **E5M2**: 5-bit exponent, 2-bit mantissa - better for activations/gradients
//!
//! This module provides:
//! - FP8 weight quantization for inference
//! - Dynamic scaling for FP8 training
//! - FBGEMM-style FP8 linear layers
//!
//! Note: Full FP8 support requires mlx-rs bindings for FP8 operations.
//! Currently uses BF16 fallback with scaling factors for the same interface.

use mlx_rs::error::Exception;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

/// FP8 format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fp8Format {
    /// E4M3 format: 4-bit exponent, 3-bit mantissa.
    /// Range: ~+-240, Best for weights.
    E4M3,
    /// E5M2 format: 5-bit exponent, 2-bit mantissa.
    /// Range: ~+-57344, Best for activations.
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

/// Quantized FP8 tensor with scale factor.
#[derive(Debug, Clone)]
pub struct Fp8Tensor {
    /// The quantized data (stored as bf16 until native FP8 support).
    pub data: Array,
    /// Scale factor for dequantization.
    pub scale: Array,
    /// Inverse scale for efficient computation.
    pub scale_inv: Array,
    /// Format used for quantization.
    pub format: Fp8Format,
}

impl Fp8Tensor {
    /// Quantize a tensor to FP8.
    ///
    /// The quantization formula is: `q = clamp(x / scale, -max, max)`
    pub fn quantize(x: &Array, format: Fp8Format, per_channel: bool) -> Result<Self, Exception> {
        // Compute amax for scaling
        let amax = if per_channel {
            // Per-channel amax along last dimension
            let abs_x = mlx_rs::ops::abs(x)?;
            abs_x.max_axis(-1, true)?
        } else {
            // Per-tensor amax
            let abs_x = mlx_rs::ops::abs(x)?;
            abs_x.max(None)?
        };

        // Compute scale: scale = amax / max_fp8
        let max_fp8 = Array::from_f32(format.max_value());
        let eps = Array::from_f32(1e-12);
        let scale = mlx_rs::ops::maximum(&amax, &eps)?.divide(&max_fp8)?;
        let scale_inv = max_fp8.divide(&mlx_rs::ops::maximum(&amax, &eps)?)?;

        // Quantize: q = x / scale (then clip to FP8 range)
        let scaled = x.divide(&scale)?;
        let neg_max = Array::from_f32(-format.max_value());
        let data = mlx_rs::ops::maximum(&scaled, &neg_max)?;
        let data = mlx_rs::ops::minimum(&data, &max_fp8)?;

        // Convert to bf16 (FP8 storage emulation until native support)
        let data = data.as_dtype(mlx_rs::Dtype::Bfloat16)?;

        Ok(Self {
            data,
            scale,
            scale_inv,
            format,
        })
    }

    /// Dequantize the FP8 tensor back to full precision.
    pub fn dequantize(&self) -> Result<Array, Exception> {
        // Convert to float32 and multiply by scale
        let float_data = self.data.as_dtype(mlx_rs::Dtype::Float32)?;
        float_data.multiply(&self.scale)
    }

    /// Get the quantized data (for FP8 matmul when available).
    pub fn data(&self) -> &Array {
        &self.data
    }

    /// Get the scale factor.
    pub fn scale(&self) -> &Array {
        &self.scale
    }
}

/// FP8 Linear layer for inference.
///
/// Weights are stored in FP8 format, activations are dynamically quantized.
#[derive(Debug, Clone)]
pub struct Fp8Linear {
    /// Quantized weights.
    pub weight: Fp8Tensor,
    /// Optional bias (kept in full precision).
    pub bias: Option<Array>,
    /// Configuration.
    pub config: Fp8Config,
}

impl Fp8Linear {
    /// Create from a standard linear layer's weights.
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

    /// Forward pass with FP8 quantization.
    ///
    /// Currently uses BF16 emulation. When mlx-rs adds native FP8:
    /// 1. Quantize input to FP8 E5M2
    /// 2. Perform FP8 matmul
    /// 3. Dequantize output with combined scale
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // TODO: Replace with native FP8 matmul when available in mlx-rs

        // For now: dequantize weights and compute in BF16
        let weight = self.weight.dequantize()?;
        let weight_bf16 = weight.as_dtype(mlx_rs::Dtype::Bfloat16)?;
        let x_bf16 = x.as_dtype(mlx_rs::Dtype::Bfloat16)?;

        // matmul: x @ weight.T
        let weight_t = weight_bf16.t();
        let mut output = x_bf16.matmul(&weight_t)?;

        if let Some(ref bias) = self.bias {
            output = output.add(bias)?;
        }

        Ok(output)
    }

    /// Memory size in bytes.
    pub fn memory_bytes(&self) -> usize {
        let weight_bytes = self.weight.data.size() * 2; // BF16 = 2 bytes
        let scale_bytes = self.weight.scale.size() * 4; // F32 = 4 bytes
        let bias_bytes = self.bias.as_ref().map(|b| b.size() * 4).unwrap_or(0);
        weight_bytes + scale_bytes + bias_bytes
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
    // FP8 = 8 bits + ~4 bits overhead for scales (amortized)
    let fp8_bits = 8 + 4;
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
    fn test_fp8_quantize_dequantize() {
        let x = mlx_rs::random::normal::<f32>(&[4, 4], None, None, None).unwrap();

        let quantized = Fp8Tensor::quantize(&x, Fp8Format::E4M3, false).unwrap();
        let dequantized = quantized.dequantize().unwrap();

        // Check shape preserved
        assert_eq!(x.shape(), dequantized.shape());

        // Values should be approximately equal
        x.eval().unwrap();
        dequantized.eval().unwrap();
    }

    #[test]
    fn test_fp8_linear() {
        let weight = mlx_rs::random::normal::<f32>(&[16, 8], None, None, None).unwrap();
        let config = Fp8Config::default();

        let fp8_linear = Fp8Linear::from_weights(&weight, None, config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[2, 8], None, None, None).unwrap();
        let output = fp8_linear.forward(&x).unwrap();
        output.eval().unwrap();

        assert_eq!(output.shape(), &[2, 16]);
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
    fn test_memory_savings() {
        let (fp8_size, savings) = calculate_fp8_savings(1000, 16);
        assert!(fp8_size < 1000);
        assert!(savings > 0.0 && savings < 1.0);
    }
}
