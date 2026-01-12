//! Group-wise quantization for GPTQ/AWQ format loading.
//!
//! This module provides support for loading and working with models quantized
//! using GPTQ (GPT Quantization) or AWQ (Activation-aware Weight Quantization).
//!
//! Both formats use the same underlying storage:
//! - Packed quantized weights (uint32, multiple values per element)
//! - Per-group scales for dequantization
//! - Per-group zeros/biases
//! - Configurable bits (2, 3, 4, 8) and group_size (typically 64 or 128)
//!
//! # GPTQ Reference
//! - Paper: <https://arxiv.org/abs/2210.17323>
//! - Uses Hessian-based calibration to minimize quantization error
//!
//! # AWQ Reference
//! - Paper: <https://arxiv.org/abs/2306.00978>
//! - Uses activation-aware scaling to preserve salient weights
//!
//! # Usage
//!
//! ```ignore
//! use pmetal_mlx::quantization::group::{GroupQuantizer, GroupQuantConfig};
//!
//! // Create quantizer for 4-bit with group_size=64
//! let config = GroupQuantConfig::new(4, 64);
//! let quantizer = GroupQuantizer::new(config);
//!
//! // Quantize weights
//! let quantized = quantizer.quantize(weights)?;
//!
//! // Dequantize for inference
//! let weights = quantizer.dequantize(&quantized)?;
//! ```

use super::{QuantScheme, QuantizerOps, QuantizedTensor};
use pmetal_core::{PMetalError, Result};

/// Configuration for group-wise quantization.
#[derive(Debug, Clone, Copy)]
pub struct GroupQuantConfig {
    /// Number of bits per weight (2, 3, 4, or 8).
    pub bits: u8,
    /// Group size for quantization (typically 64 or 128).
    pub group_size: usize,
    /// Whether to use symmetric quantization.
    pub symmetric: bool,
}

impl GroupQuantConfig {
    /// Create a new group quantization config.
    pub fn new(bits: u8, group_size: usize) -> Self {
        assert!(
            bits == 2 || bits == 3 || bits == 4 || bits == 8,
            "bits must be 2, 3, 4, or 8"
        );
        Self {
            bits,
            group_size,
            symmetric: false, // GPTQ/AWQ typically use asymmetric
        }
    }

    /// Use symmetric quantization.
    pub fn symmetric(mut self) -> Self {
        self.symmetric = true;
        self
    }

    /// Calculate number of quantization levels.
    pub fn n_levels(&self) -> usize {
        1 << self.bits
    }

    /// Calculate elements packed per u32.
    pub fn elements_per_u32(&self) -> usize {
        32 / self.bits as usize
    }
}

impl Default for GroupQuantConfig {
    fn default() -> Self {
        Self::new(4, 64)
    }
}

/// Group-wise quantized tensor.
#[derive(Debug, Clone)]
pub struct GroupQuantizedTensor {
    /// Packed quantized weights [out_features, in_features / el_per_int].
    pub qweight: Vec<u32>,
    /// Per-group scales [out_features, in_features / group_size].
    pub scales: Vec<f32>,
    /// Per-group zeros (for asymmetric quantization).
    /// [out_features, in_features / group_size]
    pub zeros: Option<Vec<f32>>,
    /// Original weight shape [out_features, in_features].
    pub shape: Vec<usize>,
    /// Quantization config.
    pub config: GroupQuantConfig,
}

impl GroupQuantizedTensor {
    /// Get the output features (rows).
    pub fn out_features(&self) -> usize {
        self.shape[0]
    }

    /// Get the input features (columns).
    pub fn in_features(&self) -> usize {
        self.shape[1]
    }

    /// Calculate packed weight dimension.
    pub fn packed_dim(&self) -> usize {
        (self.in_features() + self.config.elements_per_u32() - 1) / self.config.elements_per_u32()
    }

    /// Calculate number of groups.
    pub fn num_groups(&self) -> usize {
        (self.in_features() + self.config.group_size - 1) / self.config.group_size
    }

    /// Estimated memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let qweight_bytes = self.qweight.len() * 4;
        let scales_bytes = self.scales.len() * 4;
        let zeros_bytes = self.zeros.as_ref().map(|z| z.len() * 4).unwrap_or(0);
        qweight_bytes + scales_bytes + zeros_bytes
    }

    /// Calculate compression ratio vs f32.
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.out_features() * self.in_features() * 4; // f32
        original_bytes as f32 / self.memory_usage() as f32
    }
}

/// Group-wise quantizer for GPTQ/AWQ format.
#[derive(Debug, Clone)]
pub struct GroupQuantizer {
    config: GroupQuantConfig,
}

impl GroupQuantizer {
    /// Create a new group quantizer.
    pub fn new(config: GroupQuantConfig) -> Self {
        Self { config }
    }

    /// Create a 4-bit quantizer with default group size.
    pub fn q4(group_size: usize) -> Self {
        Self::new(GroupQuantConfig::new(4, group_size))
    }

    /// Create an 8-bit quantizer with default group size.
    pub fn q8(group_size: usize) -> Self {
        Self::new(GroupQuantConfig::new(8, group_size))
    }

    /// Get the config.
    pub fn config(&self) -> &GroupQuantConfig {
        &self.config
    }

    /// Quantize a weight matrix.
    ///
    /// Input shape: [out_features, in_features]
    pub fn quantize_weights(&self, weights: &[f32], shape: &[usize]) -> Result<GroupQuantizedTensor> {
        if shape.len() != 2 {
            return Err(PMetalError::Quantization(
                "Group quantization requires 2D weights".to_string(),
            ));
        }

        let out_features = shape[0];
        let in_features = shape[1];
        let group_size = self.config.group_size;
        let bits = self.config.bits;
        let n_levels = self.config.n_levels();
        let el_per_int = self.config.elements_per_u32();

        let num_groups = (in_features + group_size - 1) / group_size;
        let packed_dim = (in_features + el_per_int - 1) / el_per_int;

        let mut qweight = vec![0u32; out_features * packed_dim];
        let mut scales = vec![0.0f32; out_features * num_groups];
        let mut zeros = if self.config.symmetric {
            None
        } else {
            Some(vec![0.0f32; out_features * num_groups])
        };

        // Quantize each row
        for row in 0..out_features {
            let row_start = row * in_features;

            // Process each group
            for g in 0..num_groups {
                let group_start = g * group_size;
                let group_end = (group_start + group_size).min(in_features);

                // Find min/max for this group
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;
                for i in group_start..group_end {
                    let val = weights[row_start + i];
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }

                // Calculate scale and zero
                let (scale, zero) = if self.config.symmetric {
                    let amax = max_val.abs().max(min_val.abs());
                    let scale = if amax > 0.0 {
                        amax / ((n_levels / 2 - 1) as f32)
                    } else {
                        1.0
                    };
                    (scale, 0.0)
                } else {
                    let range = max_val - min_val;
                    let scale = if range > 0.0 {
                        range / (n_levels - 1) as f32
                    } else {
                        1.0
                    };
                    (scale, min_val)
                };

                scales[row * num_groups + g] = scale;
                if let Some(ref mut z) = zeros {
                    z[row * num_groups + g] = zero;
                }

                // Quantize values in this group
                for i in group_start..group_end {
                    let val = weights[row_start + i];
                    let q = if self.config.symmetric {
                        let half = (n_levels / 2) as f32;
                        ((val / scale).round() + half)
                            .clamp(0.0, (n_levels - 1) as f32) as u32
                    } else {
                        ((val - zero) / scale)
                            .round()
                            .clamp(0.0, (n_levels - 1) as f32) as u32
                    };

                    // Pack into qweight
                    let pack_idx = i / el_per_int;
                    let bit_offset = (i % el_per_int) * bits as usize;
                    qweight[row * packed_dim + pack_idx] |= q << bit_offset;
                }
            }
        }

        Ok(GroupQuantizedTensor {
            qweight,
            scales,
            zeros,
            shape: shape.to_vec(),
            config: self.config,
        })
    }

    /// Dequantize weights back to f32.
    pub fn dequantize_weights(&self, quantized: &GroupQuantizedTensor) -> Result<Vec<f32>> {
        let out_features = quantized.out_features();
        let in_features = quantized.in_features();
        let group_size = quantized.config.group_size;
        let bits = quantized.config.bits;
        let n_levels = quantized.config.n_levels();
        let el_per_int = quantized.config.elements_per_u32();
        let num_groups = quantized.num_groups();
        let packed_dim = quantized.packed_dim();
        let mask = (1u32 << bits) - 1;

        let mut weights = vec![0.0f32; out_features * in_features];

        for row in 0..out_features {
            for i in 0..in_features {
                let g = i / group_size;
                let scale = quantized.scales[row * num_groups + g];
                let zero = quantized
                    .zeros
                    .as_ref()
                    .map(|z| z[row * num_groups + g])
                    .unwrap_or(0.0);

                // Unpack quantized value
                let pack_idx = i / el_per_int;
                let bit_offset = (i % el_per_int) * bits as usize;
                let q = (quantized.qweight[row * packed_dim + pack_idx] >> bit_offset) & mask;

                // Dequantize
                let val = if quantized.config.symmetric {
                    let half = (n_levels / 2) as f32;
                    (q as f32 - half) * scale
                } else {
                    q as f32 * scale + zero
                };

                weights[row * in_features + i] = val;
            }
        }

        Ok(weights)
    }
}

impl QuantizerOps for GroupQuantizer {
    fn quantize(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedTensor> {
        let group_quantized = self.quantize_weights(data, shape)?;

        // Convert to generic QuantizedTensor format
        let data: Vec<u8> = group_quantized
            .qweight
            .iter()
            .flat_map(|&w| w.to_le_bytes())
            .collect();

        Ok(QuantizedTensor {
            data,
            absmax: group_quantized.scales,
            absmax_quant: None,
            shape: shape.to_vec(),
            block_size: self.config.group_size,
            scheme: self.scheme(),
        })
    }

    fn dequantize(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>> {
        if quantized.scheme != self.scheme() {
            return Err(PMetalError::Quantization(format!(
                "Scheme mismatch: expected {:?}, got {:?}",
                self.scheme(),
                quantized.scheme
            )));
        }

        // Reconstruct GroupQuantizedTensor
        let qweight: Vec<u32> = quantized
            .data
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        let group_quantized = GroupQuantizedTensor {
            qweight,
            scales: quantized.absmax.clone(),
            zeros: None, // Not stored in generic format
            shape: quantized.shape.clone(),
            config: GroupQuantConfig::new(self.config.bits, quantized.block_size),
        };

        self.dequantize_weights(&group_quantized)
    }

    fn scheme(&self) -> QuantScheme {
        match self.config.bits {
            2 => QuantScheme::Group2,
            3 => QuantScheme::Group3,
            4 => QuantScheme::Group4,
            8 => QuantScheme::Group8,
            _ => unreachable!(),
        }
    }
}

/// Load GPTQ-quantized weights from raw components.
///
/// This is useful for loading pre-quantized models from HuggingFace.
pub fn load_gptq_weights(
    qweight: Vec<u32>,
    scales: Vec<f32>,
    zeros: Option<Vec<f32>>,
    shape: &[usize],
    bits: u8,
    group_size: usize,
) -> GroupQuantizedTensor {
    GroupQuantizedTensor {
        qweight,
        scales,
        zeros,
        shape: shape.to_vec(),
        config: GroupQuantConfig::new(bits, group_size),
    }
}

/// Load AWQ-quantized weights (same format as GPTQ).
pub fn load_awq_weights(
    qweight: Vec<u32>,
    scales: Vec<f32>,
    zeros: Option<Vec<f32>>,
    shape: &[usize],
    bits: u8,
    group_size: usize,
) -> GroupQuantizedTensor {
    // AWQ uses the same storage format as GPTQ
    load_gptq_weights(qweight, scales, zeros, shape, bits, group_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_quant_config() {
        let config = GroupQuantConfig::new(4, 64);
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 64);
        assert_eq!(config.n_levels(), 16);
        assert_eq!(config.elements_per_u32(), 8);
    }

    #[test]
    fn test_group_quant_config_default() {
        let config = GroupQuantConfig::default();
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 64);
    }

    #[test]
    #[should_panic(expected = "bits must be 2, 3, 4, or 8")]
    fn test_invalid_bits() {
        let _ = GroupQuantConfig::new(5, 64);
    }

    #[test]
    fn test_group_quantizer_4bit() {
        let quantizer = GroupQuantizer::q4(64);

        // Create simple test weights
        let weights: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let shape = vec![4, 64];

        let quantized = quantizer.quantize_weights(&weights, &shape).unwrap();

        assert_eq!(quantized.shape, shape);
        assert_eq!(quantized.out_features(), 4);
        assert_eq!(quantized.in_features(), 64);
        assert!(quantized.compression_ratio() > 4.0); // Should be ~8x for 4-bit

        // Dequantize and check error
        let dequantized = quantizer.dequantize_weights(&quantized).unwrap();
        assert_eq!(dequantized.len(), weights.len());

        // Check error is reasonable
        let mse: f32 = weights
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / weights.len() as f32;
        assert!(mse < 0.01, "MSE too high: {}", mse);
    }

    #[test]
    fn test_group_quantizer_8bit() {
        let quantizer = GroupQuantizer::q8(64);

        let weights: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let shape = vec![2, 64];

        let quantized = quantizer.quantize_weights(&weights, &shape).unwrap();
        let dequantized = quantizer.dequantize_weights(&quantized).unwrap();

        // 8-bit should have very low error
        let mse: f32 = weights
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / weights.len() as f32;
        assert!(mse < 0.0001, "MSE too high for 8-bit: {}", mse);
    }

    #[test]
    fn test_group_quantizer_symmetric() {
        let config = GroupQuantConfig::new(4, 64).symmetric();
        let quantizer = GroupQuantizer::new(config);

        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let shape = vec![1, 64];

        let quantized = quantizer.quantize_weights(&weights, &shape).unwrap();
        assert!(quantized.zeros.is_none());

        let dequantized = quantizer.dequantize_weights(&quantized).unwrap();
        assert_eq!(dequantized.len(), weights.len());
    }

    #[test]
    fn test_load_gptq_weights() {
        // Simulate loading pre-quantized weights
        let qweight = vec![0u32; 8]; // 1 row, 64 cols packed into 8 u32s
        let scales = vec![1.0f32; 1]; // 1 group
        let shape = vec![1, 64];

        let quantized = load_gptq_weights(qweight, scales, None, &shape, 4, 64);

        assert_eq!(quantized.out_features(), 1);
        assert_eq!(quantized.in_features(), 64);
        assert_eq!(quantized.config.bits, 4);
    }

    #[test]
    fn test_quantizer_ops_trait() {
        let quantizer = GroupQuantizer::q4(64);

        let weights: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let shape = vec![4, 64];

        // Use the trait methods
        let quantized = quantizer.quantize(&weights, &shape).unwrap();
        assert_eq!(quantized.scheme, QuantScheme::Group4);

        let dequantized = quantizer.dequantize(&quantized).unwrap();
        assert_eq!(dequantized.len(), weights.len());
    }

    #[test]
    fn test_memory_usage() {
        let quantizer = GroupQuantizer::q4(64);

        let weights: Vec<f32> = vec![0.0; 4096 * 4096]; // 16M weights
        let shape = vec![4096, 4096];

        let quantized = quantizer.quantize_weights(&weights, &shape).unwrap();

        // Original: 4096 * 4096 * 4 = 64MB
        // Quantized: ~8MB (qweight) + ~1MB (scales) = ~9MB
        let compression = quantized.compression_ratio();
        assert!(compression > 6.0, "Compression ratio: {}", compression);
    }
}
