//! Int8 (8-bit integer) quantization.
//!
//! Symmetric quantization to 8-bit integers, primarily used for activations.

use super::{QuantScheme, QuantizedTensor, QuantizerOps};
use pmetal_core::Result;

/// Int8 quantizer configuration.
#[derive(Debug, Clone)]
pub struct Int8Config {
    /// Block size for blockwise quantization.
    pub block_size: usize,
    /// Use symmetric quantization (vs asymmetric).
    pub symmetric: bool,
}

impl Default for Int8Config {
    fn default() -> Self {
        Self {
            block_size: 64,
            symmetric: true,
        }
    }
}

/// Int8 quantizer.
#[derive(Debug, Clone)]
pub struct Int8Quantizer {
    /// Configuration.
    pub config: Int8Config,
}

impl Int8Quantizer {
    /// Create a new Int8 quantizer.
    pub fn new() -> Self {
        Self::with_config(Int8Config::default())
    }

    /// Create a new Int8 quantizer with custom configuration.
    pub fn with_config(config: Int8Config) -> Self {
        Self { config }
    }
}

impl Default for Int8Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizerOps for Int8Quantizer {
    fn quantize(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedTensor> {
        let block_size = self.config.block_size;
        let n_elements = data.len();
        let n_blocks = (n_elements + block_size - 1) / block_size;

        let mut quantized = Vec::with_capacity(n_elements);
        let mut absmax = Vec::with_capacity(n_blocks);

        for block_start in (0..n_elements).step_by(block_size) {
            let block_end = (block_start + block_size).min(n_elements);
            let block = &data[block_start..block_end];

            let block_absmax = block
                .iter()
                .map(|&v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-10);
            absmax.push(block_absmax);

            // Symmetric quantization to [-127, 127]
            for &v in block {
                let normalized = v / block_absmax;
                let quantized_val = (normalized * 127.0).round().clamp(-127.0, 127.0) as i8;
                quantized.push(quantized_val as u8);
            }
        }

        Ok(QuantizedTensor {
            data: quantized,
            absmax,
            absmax_quant: None,
            shape: shape.to_vec(),
            block_size,
            scheme: QuantScheme::Int8,
        })
    }

    fn dequantize(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>> {
        let block_size = quantized.block_size;
        let total_elements: usize = quantized.shape.iter().product();
        let mut result = Vec::with_capacity(total_elements);

        let mut idx = 0;
        for (block_idx, &block_absmax) in quantized.absmax.iter().enumerate() {
            let block_start = block_idx * block_size;
            let block_end = (block_start + block_size).min(total_elements);
            let block_len = block_end - block_start;

            for _ in 0..block_len {
                let quantized_val = quantized.data[idx] as i8;
                let dequantized = (quantized_val as f32 / 127.0) * block_absmax;
                result.push(dequantized);
                idx += 1;
            }
        }

        Ok(result)
    }

    fn scheme(&self) -> QuantScheme {
        QuantScheme::Int8
    }
}
