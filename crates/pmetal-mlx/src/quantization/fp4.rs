//! FP4 (Floating Point 4-bit) quantization.
//!
//! FP4 uses a simpler quantization scheme with 8 magnitude levels + sign bit.

use super::{QuantScheme, QuantizedTensor, QuantizerOps};
use pmetal_core::Result;

/// FP4 quantization bins (positive values only, sign is separate).
pub const FP4_BINS: [f32; 8] = [
    0.0, 0.0625, // 2^-4
    0.125,  // 2^-3
    0.25,   // 2^-2
    0.5,    // 2^-1
    1.0,    // 2^0
    2.0,    // 2^1
    4.0,    // 2^2 (scaled)
];

/// FP4 quantizer configuration.
#[derive(Debug, Clone)]
pub struct FP4Config {
    /// Block size for blockwise quantization.
    pub block_size: usize,
}

impl Default for FP4Config {
    fn default() -> Self {
        Self { block_size: 64 }
    }
}

/// FP4 quantizer.
#[derive(Debug, Clone)]
pub struct FP4Quantizer {
    /// Configuration.
    pub config: FP4Config,
}

impl FP4Quantizer {
    /// Create a new FP4 quantizer.
    pub fn new() -> Self {
        Self::with_config(FP4Config::default())
    }

    /// Create a new FP4 quantizer with custom configuration.
    pub fn with_config(config: FP4Config) -> Self {
        Self { config }
    }
}

impl Default for FP4Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizerOps for FP4Quantizer {
    fn quantize(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedTensor> {
        // Similar implementation to NF4 but using FP4 bins
        let block_size = self.config.block_size;
        let n_elements = data.len();
        let n_blocks = (n_elements + block_size - 1) / block_size;

        let mut quantized = Vec::with_capacity((n_elements + 1) / 2);
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

            let mut indices: Vec<u8> = block
                .iter()
                .map(|&v| {
                    let normalized = v.abs() / block_absmax;
                    let sign = if v < 0.0 { 8u8 } else { 0u8 };

                    // Find closest FP4 bin
                    let mut best_idx = 0u8;
                    let mut best_dist = f32::MAX;
                    for (i, &bin) in FP4_BINS.iter().enumerate() {
                        let dist = (normalized - bin).abs();
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = i as u8;
                        }
                    }
                    sign | best_idx
                })
                .collect();

            if indices.len() % 2 != 0 {
                indices.push(0);
            }

            for pair in indices.chunks(2) {
                quantized.push((pair[0] << 4) | pair[1]);
            }
        }

        Ok(QuantizedTensor {
            data: quantized,
            absmax,
            absmax_quant: None,
            shape: shape.to_vec(),
            block_size,
            scheme: QuantScheme::FP4,
        })
    }

    fn dequantize(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>> {
        let block_size = quantized.block_size;
        let total_elements: usize = quantized.shape.iter().product();
        let mut result = Vec::with_capacity(total_elements);

        let mut byte_idx = 0;
        for (block_idx, &block_absmax) in quantized.absmax.iter().enumerate() {
            let block_start = block_idx * block_size;
            let block_end = (block_start + block_size).min(total_elements);
            let block_len = block_end - block_start;

            for i in 0..block_len {
                let packed = quantized.data[byte_idx + i / 2];
                let index = if i % 2 == 0 {
                    (packed >> 4) & 0x0F
                } else {
                    packed & 0x0F
                };

                let sign = if index & 8 != 0 { -1.0 } else { 1.0 };
                let magnitude_idx = (index & 7) as usize;
                let normalized = FP4_BINS[magnitude_idx];
                result.push(sign * normalized * block_absmax);
            }

            byte_idx += (block_len + 1) / 2;
        }

        Ok(result)
    }

    fn scheme(&self) -> QuantScheme {
        QuantScheme::FP4
    }
}
