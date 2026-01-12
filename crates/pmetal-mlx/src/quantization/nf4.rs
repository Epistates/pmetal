//! NF4 (Normal Float 4-bit) quantization.
//!
//! NF4 uses 16 quantization bins based on the normal distribution N(0,1),
//! optimized for neural network weight distributions.
//!
//! Key properties:
//! - 16 bins with equal area under normal distribution
//! - Block-wise quantization (typically block size 64)
//! - Optional double quantization of absmax values

use super::{QuantScheme, QuantizedAbsmax, QuantizedTensor, QuantizerOps};
use pmetal_core::Result;

/// NF4 quantization bins derived from normal distribution.
///
/// These values are chosen such that each bin has equal probability
/// under a standard normal distribution N(0, 1).
pub const NF4_BINS: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Decision tree thresholds for fast NF4 quantization.
/// These are midpoints between adjacent NF4 bins.
const NF4_THRESHOLDS: [f32; 15] = [
    -0.8480964004993439,  // between bins 0-1
    -0.6106329262256622,  // between bins 1-2
    -0.4599952697753906,  // between bins 2-3
    -0.33967943489551544, // between bins 3-4
    -0.23460740596055984, // between bins 4-5
    -0.13791173324,       // between bins 5-6
    -0.04552501812577248, // between bins 6-7
    0.03979014977812767,  // between bins 7-8
    0.1202552503,         // between bins 8-9
    0.20352125167846680,  // between bins 9-10
    0.29201376438140869,  // between bins 10-11
    0.38931253552436829,  // between bins 11-12
    0.50166341662406921,  // between bins 12-13
    0.64278691411018372,  // between bins 13-14
    0.86147841811180115,  // between bins 14-15
];

/// NF4 quantizer configuration.
#[derive(Debug, Clone)]
pub struct NF4Config {
    /// Block size for blockwise quantization.
    pub block_size: usize,
    /// Enable double quantization of absmax values.
    pub double_quant: bool,
}

impl Default for NF4Config {
    fn default() -> Self {
        Self {
            block_size: 64,
            double_quant: true,
        }
    }
}

/// NF4 quantizer.
#[derive(Debug, Clone)]
pub struct NF4Quantizer {
    /// Configuration.
    pub config: NF4Config,
}

impl NF4Quantizer {
    /// Create a new NF4 quantizer with default configuration.
    pub fn new() -> Self {
        Self::with_config(NF4Config::default())
    }

    /// Create a new NF4 quantizer with custom configuration.
    pub fn with_config(config: NF4Config) -> Self {
        Self { config }
    }

    /// Quantize a single value to NF4 index using binary search decision tree.
    ///
    /// This is an efficient O(log n) implementation matching bitsandbytes.
    #[inline]
    pub fn quantize_value(&self, value: f32) -> u8 {
        // Binary search decision tree
        if value < NF4_THRESHOLDS[7] {
            if value < NF4_THRESHOLDS[3] {
                if value < NF4_THRESHOLDS[1] {
                    if value < NF4_THRESHOLDS[0] {
                        0
                    } else {
                        1
                    }
                } else if value < NF4_THRESHOLDS[2] {
                    2
                } else {
                    3
                }
            } else if value < NF4_THRESHOLDS[5] {
                if value < NF4_THRESHOLDS[4] {
                    4
                } else {
                    5
                }
            } else if value < NF4_THRESHOLDS[6] {
                6
            } else {
                7
            }
        } else if value < NF4_THRESHOLDS[11] {
            if value < NF4_THRESHOLDS[9] {
                if value < NF4_THRESHOLDS[8] {
                    8
                } else {
                    9
                }
            } else if value < NF4_THRESHOLDS[10] {
                10
            } else {
                11
            }
        } else if value < NF4_THRESHOLDS[13] {
            if value < NF4_THRESHOLDS[12] {
                12
            } else {
                13
            }
        } else if value < NF4_THRESHOLDS[14] {
            14
        } else {
            15
        }
    }

    /// Dequantize an NF4 index to a float value.
    #[inline]
    pub fn dequantize_value(&self, index: u8) -> f32 {
        NF4_BINS[index as usize]
    }

    /// Quantize absmax values for double quantization.
    fn quantize_absmax(&self, absmax: &[f32]) -> QuantizedAbsmax {
        // Find offset (min value)
        let offset = absmax.iter().cloned().fold(f32::INFINITY, f32::min);

        // Compute scale
        let max = absmax.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max - offset) / 255.0;

        // Quantize to 8-bit
        let data: Vec<u8> = absmax
            .iter()
            .map(|&v| {
                if scale > 0.0 {
                    ((v - offset) / scale).round().clamp(0.0, 255.0) as u8
                } else {
                    0
                }
            })
            .collect();

        QuantizedAbsmax {
            data,
            offset,
            scale,
        }
    }

    /// Dequantize absmax values from double quantization.
    fn dequantize_absmax(&self, quant: &QuantizedAbsmax) -> Vec<f32> {
        quant
            .data
            .iter()
            .map(|&v| v as f32 * quant.scale + quant.offset)
            .collect()
    }
}

impl Default for NF4Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizerOps for NF4Quantizer {
    fn quantize(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedTensor> {
        let block_size = self.config.block_size;
        let n_elements = data.len();
        let n_blocks = (n_elements + block_size - 1) / block_size;

        let mut quantized = Vec::with_capacity((n_elements + 1) / 2);
        let mut absmax = Vec::with_capacity(n_blocks);

        for block_start in (0..n_elements).step_by(block_size) {
            let block_end = (block_start + block_size).min(n_elements);
            let block = &data[block_start..block_end];

            // Compute absolute maximum for this block
            let block_absmax = block
                .iter()
                .map(|&v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-10); // Prevent division by zero
            absmax.push(block_absmax);

            // Normalize and quantize
            let mut indices: Vec<u8> = block
                .iter()
                .map(|&v| {
                    let normalized = (v / block_absmax).clamp(-1.0, 1.0);
                    self.quantize_value(normalized)
                })
                .collect();

            // Pad to even length if needed
            if indices.len() % 2 != 0 {
                indices.push(0);
            }

            // Pack two 4-bit values into one byte
            for pair in indices.chunks(2) {
                quantized.push((pair[0] << 4) | pair[1]);
            }
        }

        // Optional double quantization
        let absmax_quant = if self.config.double_quant {
            Some(self.quantize_absmax(&absmax))
        } else {
            None
        };

        Ok(QuantizedTensor {
            data: quantized,
            absmax,
            absmax_quant,
            shape: shape.to_vec(),
            block_size,
            scheme: QuantScheme::NF4,
        })
    }

    fn dequantize(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>> {
        // Get absmax values (dequantize if double-quantized)
        let absmax = match &quantized.absmax_quant {
            Some(quant) => self.dequantize_absmax(quant),
            None => quantized.absmax.clone(),
        };

        let block_size = quantized.block_size;
        let total_elements: usize = quantized.shape.iter().product();
        let mut result = Vec::with_capacity(total_elements);

        let mut byte_idx = 0;
        for (block_idx, &block_absmax) in absmax.iter().enumerate() {
            let block_start = block_idx * block_size;
            let block_end = (block_start + block_size).min(total_elements);
            let block_len = block_end - block_start;

            for i in 0..block_len {
                // Unpack 4-bit value
                let packed = quantized.data[byte_idx + i / 2];
                let index = if i % 2 == 0 {
                    (packed >> 4) & 0x0F
                } else {
                    packed & 0x0F
                };

                // Dequantize
                let normalized = self.dequantize_value(index);
                result.push(normalized * block_absmax);
            }

            byte_idx += (block_len + 1) / 2;
        }

        Ok(result)
    }

    fn scheme(&self) -> QuantScheme {
        QuantScheme::NF4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_bins_properties() {
        // Bins should be sorted
        for i in 1..NF4_BINS.len() {
            assert!(NF4_BINS[i] > NF4_BINS[i - 1]);
        }

        // First and last should be -1 and 1
        assert_eq!(NF4_BINS[0], -1.0);
        assert_eq!(NF4_BINS[15], 1.0);

        // Middle value should be 0
        assert_eq!(NF4_BINS[7], 0.0);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let quantizer = NF4Quantizer::new();

        // Test values spread across the range
        let values = vec![
            -0.8, -0.5, -0.2, 0.0, 0.1, 0.3, 0.6, 0.9, -0.3, 0.5, 0.7, -0.1, 0.2, 0.4, -0.6, 0.8,
        ];
        let shape = vec![4, 4];

        let quantized = quantizer.quantize(&values, &shape).unwrap();
        let dequantized = quantizer.dequantize(&quantized).unwrap();

        // Values should be approximately preserved
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            // NF4 has ~0.1 quantization error on average
            assert!(
                (orig - deq).abs() < 0.2,
                "orig: {}, deq: {}",
                orig,
                deq
            );
        }
    }
}
