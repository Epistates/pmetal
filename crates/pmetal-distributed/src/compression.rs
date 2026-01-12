//! Gradient compression for bandwidth optimization.
//!
//! Provides several compression strategies:
//! - TopK: Keep only the k largest gradients
//! - Random sparsification: Randomly sample gradients
//! - Quantization: Reduce precision (FP16, BF16, INT8)
//! - Error feedback: Accumulate compression errors for future updates
//!
//! References:
//! - Deep Gradient Compression (Lin et al., 2018)
//! - 1-Bit SGD (Seide et al., 2014)
//! - PowerSGD (Vogels et al., 2019)

use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use tracing::debug;

/// Compression strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// No compression.
    None,
    /// Keep only top-k% gradients by magnitude.
    TopK { ratio: f32 },
    /// Random sparsification with given probability.
    Random { probability: f32 },
    /// Quantize to lower precision.
    Quantize(QuantizationType),
    /// PowerSGD low-rank approximation.
    PowerSGD { rank: usize },
}

impl Default for CompressionStrategy {
    fn default() -> Self {
        Self::None
    }
}

/// Quantization type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QuantizationType {
    /// FP16 (IEEE half precision).
    FP16,
    /// BF16 (Brain float).
    BF16,
    /// 8-bit integer with scale.
    INT8,
    /// 1-bit sign only (with scale).
    OneBit,
}

/// Compressed gradient representation.
#[derive(Debug, Clone)]
pub struct CompressedGradient {
    /// Original size.
    pub original_size: usize,
    /// Compression strategy used.
    pub strategy: CompressionStrategy,
    /// Compressed data.
    pub data: CompressedData,
}

/// Compressed data variants.
#[derive(Debug, Clone)]
pub enum CompressedData {
    /// Full precision (no compression).
    Full(Vec<f32>),
    /// Sparse representation (indices + values).
    Sparse {
        indices: Vec<u32>,
        values: Vec<f32>,
    },
    /// FP16 quantized.
    FP16(Vec<u16>),
    /// BF16 quantized.
    BF16(Vec<u16>),
    /// INT8 quantized with scale.
    INT8 { data: Vec<i8>, scale: f32 },
    /// 1-bit with scale.
    OneBit { signs: Vec<u8>, scale: f32 },
}

impl CompressedGradient {
    /// Get the compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.original_size * 4;
        let compressed_bytes = self.compressed_bytes();
        original_bytes as f32 / compressed_bytes as f32
    }

    /// Get compressed size in bytes.
    pub fn compressed_bytes(&self) -> usize {
        match &self.data {
            CompressedData::Full(v) => v.len() * 4,
            CompressedData::Sparse { indices, values } => indices.len() * 4 + values.len() * 4,
            CompressedData::FP16(v) => v.len() * 2,
            CompressedData::BF16(v) => v.len() * 2,
            CompressedData::INT8 { data, .. } => data.len() + 4,
            CompressedData::OneBit { signs, .. } => signs.len() + 4,
        }
    }
}

/// Gradient compressor with error feedback.
pub struct GradientCompressor {
    /// Compression strategy.
    strategy: CompressionStrategy,
    /// Error feedback buffer (accumulated residuals).
    error_feedback: Option<Vec<f32>>,
    /// Whether to use error feedback.
    use_error_feedback: bool,
    /// Random seed for reproducibility.
    rng_seed: u64,
}

impl GradientCompressor {
    /// Create a new compressor.
    pub fn new(strategy: CompressionStrategy, use_error_feedback: bool) -> Self {
        Self {
            strategy,
            error_feedback: None,
            use_error_feedback,
            rng_seed: 42,
        }
    }

    /// Compress gradients.
    pub fn compress(&mut self, gradients: &[f32]) -> CompressedGradient {
        let original_size = gradients.len();

        // Apply error feedback if enabled
        let working_grads = if self.use_error_feedback {
            if let Some(ref error) = self.error_feedback {
                gradients
                    .iter()
                    .zip(error.iter())
                    .map(|(g, e)| g + e)
                    .collect()
            } else {
                gradients.to_vec()
            }
        } else {
            gradients.to_vec()
        };

        let (data, residual) = match &self.strategy {
            CompressionStrategy::None => (CompressedData::Full(working_grads.clone()), None),
            CompressionStrategy::TopK { ratio } => self.compress_topk(&working_grads, *ratio),
            CompressionStrategy::Random { probability } => {
                self.compress_random(&working_grads, *probability)
            }
            CompressionStrategy::Quantize(qtype) => {
                (self.quantize(&working_grads, *qtype), None)
            }
            CompressionStrategy::PowerSGD { rank } => {
                // PowerSGD requires state across iterations, simplified here
                (CompressedData::Full(working_grads.clone()), None)
            }
        };

        // Store residual for error feedback
        if self.use_error_feedback {
            self.error_feedback = residual;
        }

        let result = CompressedGradient {
            original_size,
            strategy: self.strategy.clone(),
            data,
        };

        debug!(
            "Compressed {} floats, ratio={:.2}x",
            original_size,
            result.compression_ratio()
        );

        result
    }

    /// Decompress gradients.
    pub fn decompress(&self, compressed: &CompressedGradient) -> Vec<f32> {
        match &compressed.data {
            CompressedData::Full(v) => v.clone(),
            CompressedData::Sparse { indices, values } => {
                let mut result = vec![0.0f32; compressed.original_size];
                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    result[idx as usize] = val;
                }
                result
            }
            CompressedData::FP16(v) => v
                .iter()
                .map(|&x| f16::from_bits(x).to_f32())
                .collect(),
            CompressedData::BF16(v) => v
                .iter()
                .map(|&x| bf16::from_bits(x).to_f32())
                .collect(),
            CompressedData::INT8 { data, scale } => {
                data.iter().map(|&x| x as f32 * scale).collect()
            }
            CompressedData::OneBit { signs, scale } => {
                let mut result = Vec::with_capacity(compressed.original_size);
                for byte in signs {
                    for bit in 0..8 {
                        if result.len() >= compressed.original_size {
                            break;
                        }
                        let sign = if (byte >> bit) & 1 == 1 { 1.0 } else { -1.0 };
                        result.push(sign * scale);
                    }
                }
                result
            }
        }
    }

    /// Top-K sparsification.
    fn compress_topk(&self, gradients: &[f32], ratio: f32) -> (CompressedData, Option<Vec<f32>>) {
        let k = ((gradients.len() as f32 * ratio) as usize).max(1);

        // Find top-k by magnitude using a min-heap
        let mut heap: BinaryHeap<std::cmp::Reverse<(ordered_float::OrderedFloat<f32>, u32)>> =
            BinaryHeap::with_capacity(k + 1);

        for (i, &val) in gradients.iter().enumerate() {
            let abs_val = ordered_float::OrderedFloat(val.abs());
            heap.push(std::cmp::Reverse((abs_val, i as u32)));
            if heap.len() > k {
                heap.pop();
            }
        }

        // Extract indices and values
        let mut indices: Vec<u32> = heap.iter().map(|x| x.0 .1).collect();
        indices.sort_unstable();

        let values: Vec<f32> = indices.iter().map(|&i| gradients[i as usize]).collect();

        // Compute residual (unselected values)
        let mut residual = gradients.to_vec();
        for &idx in &indices {
            residual[idx as usize] = 0.0;
        }

        (CompressedData::Sparse { indices, values }, Some(residual))
    }

    /// Random sparsification.
    fn compress_random(
        &mut self,
        gradients: &[f32],
        probability: f32,
    ) -> (CompressedData, Option<Vec<f32>>) {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut residual = gradients.to_vec();

        // Simple PRNG
        let mut rng = self.rng_seed;

        for (i, &val) in gradients.iter().enumerate() {
            // LCG random number generator
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rand_val = (rng >> 33) as f32 / (u32::MAX >> 1) as f32;

            if rand_val < probability {
                indices.push(i as u32);
                values.push(val / probability); // Scale to maintain expectation
                residual[i] = 0.0;
            }
        }

        self.rng_seed = rng;
        (CompressedData::Sparse { indices, values }, Some(residual))
    }

    /// Quantize gradients.
    fn quantize(&self, gradients: &[f32], qtype: QuantizationType) -> CompressedData {
        match qtype {
            QuantizationType::FP16 => {
                let data: Vec<u16> = gradients.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
                CompressedData::FP16(data)
            }
            QuantizationType::BF16 => {
                let data: Vec<u16> = gradients.iter().map(|&x| bf16::from_f32(x).to_bits()).collect();
                CompressedData::BF16(data)
            }
            QuantizationType::INT8 => {
                let max_abs = gradients
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f32, |a, b| a.max(b));
                let scale = max_abs / 127.0;

                let data: Vec<i8> = gradients
                    .iter()
                    .map(|&x| (x / scale).clamp(-127.0, 127.0) as i8)
                    .collect();

                CompressedData::INT8 { data, scale }
            }
            QuantizationType::OneBit => {
                let mean_abs =
                    gradients.iter().map(|x| x.abs()).sum::<f32>() / gradients.len() as f32;

                let num_bytes = (gradients.len() + 7) / 8;
                let mut signs = vec![0u8; num_bytes];

                for (i, &val) in gradients.iter().enumerate() {
                    if val > 0.0 {
                        signs[i / 8] |= 1 << (i % 8);
                    }
                }

                CompressedData::OneBit {
                    signs,
                    scale: mean_abs,
                }
            }
        }
    }

    /// Reset error feedback.
    pub fn reset_error_feedback(&mut self) {
        self.error_feedback = None;
    }
}

/// Serialize compressed gradient to bytes.
pub fn serialize_compressed(compressed: &CompressedGradient) -> Vec<u8> {
    let mut result = Vec::new();

    // Header: original_size (4 bytes) + strategy_id (1 byte)
    result.extend_from_slice(&(compressed.original_size as u32).to_le_bytes());

    match &compressed.data {
        CompressedData::Full(v) => {
            result.push(0u8);
            for f in v {
                result.extend_from_slice(&f.to_le_bytes());
            }
        }
        CompressedData::Sparse { indices, values } => {
            result.push(1u8);
            result.extend_from_slice(&(indices.len() as u32).to_le_bytes());
            for &idx in indices {
                result.extend_from_slice(&idx.to_le_bytes());
            }
            for &val in values {
                result.extend_from_slice(&val.to_le_bytes());
            }
        }
        CompressedData::FP16(v) => {
            result.push(2u8);
            for &x in v {
                result.extend_from_slice(&x.to_le_bytes());
            }
        }
        CompressedData::BF16(v) => {
            result.push(3u8);
            for &x in v {
                result.extend_from_slice(&x.to_le_bytes());
            }
        }
        CompressedData::INT8 { data, scale } => {
            result.push(4u8);
            result.extend_from_slice(&scale.to_le_bytes());
            result.extend_from_slice(data.iter().map(|&x| x as u8).collect::<Vec<_>>().as_slice());
        }
        CompressedData::OneBit { signs, scale } => {
            result.push(5u8);
            result.extend_from_slice(&scale.to_le_bytes());
            result.extend_from_slice(signs);
        }
    }

    result
}

/// Deserialize compressed gradient from bytes.
pub fn deserialize_compressed(bytes: &[u8]) -> Option<CompressedGradient> {
    if bytes.len() < 5 {
        return None;
    }

    let original_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let strategy_id = bytes[4];

    let data = match strategy_id {
        0 => {
            // Full
            let floats: Vec<f32> = bytes[5..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            CompressedData::Full(floats)
        }
        1 => {
            // Sparse
            let num_indices =
                u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
            let indices_end = 9 + num_indices * 4;
            let indices: Vec<u32> = bytes[9..indices_end]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let values: Vec<f32> = bytes[indices_end..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            CompressedData::Sparse { indices, values }
        }
        2 => {
            // FP16
            let data: Vec<u16> = bytes[5..]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            CompressedData::FP16(data)
        }
        3 => {
            // BF16
            let data: Vec<u16> = bytes[5..]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            CompressedData::BF16(data)
        }
        4 => {
            // INT8
            let scale = f32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
            let data: Vec<i8> = bytes[9..].iter().map(|&x| x as i8).collect();
            CompressedData::INT8 { data, scale }
        }
        5 => {
            // OneBit
            let scale = f32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
            let signs = bytes[9..].to_vec();
            CompressedData::OneBit { signs, scale }
        }
        _ => return None,
    };

    Some(CompressedGradient {
        original_size,
        strategy: CompressionStrategy::None, // Not tracked in serialization
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_compression() {
        let mut compressor = GradientCompressor::new(CompressionStrategy::None, false);
        let grads = vec![1.0, 2.0, 3.0, 4.0];

        let compressed = compressor.compress(&grads);
        let decompressed = compressor.decompress(&compressed);

        assert_eq!(grads, decompressed);
        assert!((compressed.compression_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_topk() {
        let mut compressor =
            GradientCompressor::new(CompressionStrategy::TopK { ratio: 0.5 }, false);
        let grads = vec![1.0, 4.0, 2.0, 3.0];

        let compressed = compressor.compress(&grads);
        let decompressed = compressor.decompress(&compressed);

        // Top 50% should be 4.0 and 3.0
        assert!(decompressed[1] == 4.0);
        assert!(decompressed[3] == 3.0);
        assert!(decompressed[0] == 0.0);
        assert!(decompressed[2] == 0.0);

        // With 50% sparsity on 4 elements: 2 indices + 2 values = same as original
        // Compression becomes effective with larger tensors
        assert!(compressed.compression_ratio() >= 1.0);
    }

    #[test]
    fn test_fp16_quantization() {
        let mut compressor =
            GradientCompressor::new(CompressionStrategy::Quantize(QuantizationType::FP16), false);
        let grads = vec![1.0, 2.5, 3.14159, 4.0];

        let compressed = compressor.compress(&grads);
        let decompressed = compressor.decompress(&compressed);

        // FP16 should be approximately equal
        for (orig, decomp) in grads.iter().zip(decompressed.iter()) {
            assert!((orig - decomp).abs() < 0.01);
        }

        // 2x compression ratio
        assert!((compressed.compression_ratio() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_int8_quantization() {
        let mut compressor =
            GradientCompressor::new(CompressionStrategy::Quantize(QuantizationType::INT8), false);
        let grads = vec![1.0, 2.0, 3.0, 4.0];

        let compressed = compressor.compress(&grads);
        let decompressed = compressor.decompress(&compressed);

        // INT8 should be approximately equal
        for (orig, decomp) in grads.iter().zip(decompressed.iter()) {
            assert!((orig - decomp).abs() < 0.1);
        }

        // INT8: 4 bytes data + 4 bytes scale = 8 bytes vs 16 bytes original = 2x ratio
        assert!(compressed.compression_ratio() >= 2.0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut compressor =
            GradientCompressor::new(CompressionStrategy::TopK { ratio: 0.5 }, false);
        let grads = vec![1.0, 4.0, 2.0, 3.0];

        let compressed = compressor.compress(&grads);
        let bytes = serialize_compressed(&compressed);
        let restored = deserialize_compressed(&bytes).unwrap();

        let decompressed = compressor.decompress(&restored);

        // Should match original sparse decompression
        assert!(decompressed[1] == 4.0);
        assert!(decompressed[3] == 3.0);
    }

    #[test]
    fn test_error_feedback() {
        let mut compressor =
            GradientCompressor::new(CompressionStrategy::TopK { ratio: 0.5 }, true);

        // First compression - will accumulate residuals
        let grads1 = vec![1.0, 4.0, 2.0, 3.0];
        let _compressed1 = compressor.compress(&grads1);

        // Second compression - should include accumulated error
        let grads2 = vec![0.1, 0.1, 0.1, 0.1];
        let compressed2 = compressor.compress(&grads2);
        let decompressed2 = compressor.decompress(&compressed2);

        // Residual from first (1.0 and 2.0) should be added to second
        // Top-k should now pick the accumulated values
        assert!(decompressed2.iter().any(|&x| x > 1.0));
    }
}
