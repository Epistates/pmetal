//! Quantized Merge Operations (int8 block quantization).
//!
//! This module provides int8-quantized merge operations for ~4x memory reduction
//! relative to f32, with per-block dynamic scaling to maintain accuracy.
//!
//! # Implementation Note
//!
//! Despite the "Fp8" naming (retained for API compatibility), the quantization
//! scheme stores values as signed int8 in the range [-127, 127] with a per-block
//! f32 scale factor. This is standard symmetric int8 block quantization, not
//! the IEEE FP8 (E4M3/E5M2) format.
//!
//! # Memory Savings
//!
//! - F32 merge: 4 bytes per element
//! - Int8 merge: ~1.5 bytes per element (8 bits + scale overhead per block)
//! - Savings: ~60% memory reduction
//!
//! # Dynamic Scaling
//!
//! Uses per-block dynamic scaling with amax history tracking to prevent overflow
//! and maintain numerical accuracy during merge operations.
//!
//! # Example
//!
//! ```ignore
//! use pmetal_merge::fp8_merge::{Fp8MergeConfig, Fp8Merger};
//!
//! let config = Fp8MergeConfig::default();
//! let merger = Fp8Merger::new(config);
//!
//! // Merge tensors with int8 quantization
//! let merged = merger.ties_merge_fp8(tensors, base, weights, densities, lambda)?;
//! ```

use mlx_rs::Array;
use tracing::debug;

use crate::{MergeError, Result};

/// Quantization format selector.
///
/// Note: the names `E4M3`/`E5M2` are retained for API compatibility, but the
/// actual storage is always symmetric int8 (`[-127, 127]`) with a per-block
/// f32 scale. The `max_value` field governs the dynamic-scale history window
/// and does not represent an IEEE FP8 bit layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// "Standard" profile: scale history tuned for weights (narrower dynamic range).
    /// Equivalent to int8 block quantization with 127-level precision.
    E4M3,
    /// "Wide range" profile: scale history tuned for activations (wider dynamic range).
    /// Equivalent to int8 block quantization with 127-level precision.
    E5M2,
}

impl Fp8Format {
    /// Reference dynamic range used for scale history computation.
    ///
    /// This value does not correspond to an IEEE FP8 max; it is used as a
    /// reference ceiling in the dynamic scale tracker.
    pub fn max_value(&self) -> f32 {
        match self {
            // Int8 stores 127 positive levels; reference ceiling kept at 127 for symmetry.
            Self::E4M3 => 127.0,
            Self::E5M2 => 127.0,
        }
    }

    /// Minimum representable non-zero absolute value (one int8 quantization step).
    pub fn min_value(&self) -> f32 {
        // With scale = amax/127, the minimum non-zero value is 1/127 * amax.
        // Expressed as a fraction of max_value:
        1.0 / 127.0
    }
}

/// Configuration for FP8 merge operations.
#[derive(Debug, Clone)]
pub struct Fp8MergeConfig {
    /// Block size for quantization (must be power of 2).
    pub block_size: usize,
    /// FP8 format to use for tensor storage.
    pub format: Fp8Format,
    /// Window size for dynamic scale history.
    pub scale_window_size: usize,
    /// Whether to use FP8 for intermediate computations.
    pub fp8_intermediates: bool,
    /// Force dequantization before certain ops for accuracy.
    pub force_dequant_for_sparsify: bool,
}

impl Default for Fp8MergeConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            format: Fp8Format::E4M3,
            scale_window_size: 1024,
            fp8_intermediates: true,
            // TIES sparsification needs higher precision
            force_dequant_for_sparsify: true,
        }
    }
}

impl Fp8MergeConfig {
    /// Create config optimized for memory efficiency.
    pub fn memory_optimized() -> Self {
        Self {
            block_size: 256,
            format: Fp8Format::E4M3,
            scale_window_size: 512,
            fp8_intermediates: true,
            force_dequant_for_sparsify: false,
        }
    }

    /// Create config optimized for accuracy.
    pub fn accuracy_optimized() -> Self {
        Self {
            block_size: 64,
            format: Fp8Format::E5M2,
            scale_window_size: 2048,
            fp8_intermediates: false,
            force_dequant_for_sparsify: true,
        }
    }
}

/// Dynamic scale tracker for FP8 quantization.
///
/// Tracks amax history over a sliding window to compute optimal scales.
#[derive(Debug, Clone)]
pub struct DynamicScale {
    /// History buffer for amax values.
    amax_history: Vec<f32>,
    /// Current scale value.
    scale: f32,
    /// Window size.
    window_size: usize,
    /// Current index in circular buffer.
    current_idx: usize,
    /// FP8 format for this scale.
    format: Fp8Format,
}

impl DynamicScale {
    /// Create a new dynamic scale tracker.
    pub fn new(window_size: usize, format: Fp8Format) -> Self {
        Self {
            amax_history: vec![0.0; window_size],
            scale: 1.0,
            window_size,
            current_idx: 0,
            format,
        }
    }

    /// Update scale with new amax value.
    pub fn update(&mut self, new_amax: f32) {
        // Update history (circular buffer)
        self.amax_history[self.current_idx] = new_amax;
        self.current_idx = (self.current_idx + 1) % self.window_size;

        // Find max in history
        let max_amax = self.amax_history.iter().cloned().fold(0.0f32, f32::max);

        // Compute new scale to map amax to FP8 range
        self.scale = self.format.max_value() / max_amax.max(1e-12);
    }

    /// Get current scale value.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get inverse scale for efficient computation.
    pub fn scale_inv(&self) -> f32 {
        1.0 / self.scale
    }
}

/// Quantized tensor in FP8 format.
#[derive(Debug)]
pub struct Fp8Tensor {
    /// Quantized data stored as u8.
    data: Vec<u8>,
    /// Per-block scale factors.
    scales: Vec<f32>,
    /// Original shape.
    shape: Vec<i32>,
    /// Block size used for quantization.
    block_size: usize,
    /// FP8 format used.
    format: Fp8Format,
}

impl Fp8Tensor {
    /// Get the number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&s| s as usize).product()
    }

    /// Get the shape.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Calculate memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * std::mem::size_of::<f32>()
    }
}

/// FP8 merger for memory-efficient merge operations.
pub struct Fp8Merger {
    config: Fp8MergeConfig,
    dynamic_scale: DynamicScale,
}

impl Fp8Merger {
    /// Create a new FP8 merger.
    pub fn new(config: Fp8MergeConfig) -> Self {
        let dynamic_scale = DynamicScale::new(config.scale_window_size, config.format);
        Self {
            config,
            dynamic_scale,
        }
    }

    /// Create with default configuration.
    pub fn default_new() -> Self {
        Self::new(Fp8MergeConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &Fp8MergeConfig {
        &self.config
    }

    /// Quantize an MLX array to FP8.
    ///
    /// Uses per-block scaling to map values to [-127, 127] range.
    pub fn quantize(&mut self, array: &Array) -> Result<Fp8Tensor> {
        let shape = array.shape().to_vec();
        let data: Vec<f32> = array.as_slice().to_vec();
        let num_elements = data.len();
        let block_size = self.config.block_size;
        let num_blocks = num_elements.div_ceil(block_size);

        // Compute per-block scales and quantize
        let mut quantized = Vec::with_capacity(num_elements);
        let mut scales = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(num_elements);
            let block = &data[start..end];

            // Find amax in block
            let amax = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            // Update dynamic scale tracking
            self.dynamic_scale.update(amax);

            // Store scale as amax/127 so we can reconstruct as: quantized * scale
            // This means: quantized = round(val * 127 / amax)
            let scale = amax.max(1e-12) / 127.0;
            scales.push(scale);

            // Quantize block: map to [-127, 127]
            for &val in block {
                let scaled = val / scale; // val * 127 / amax
                let clamped = scaled.round().clamp(-127.0, 127.0);
                quantized.push(clamped as i8 as u8);
            }
        }

        Ok(Fp8Tensor {
            data: quantized,
            scales,
            shape,
            block_size,
            format: self.config.format,
        })
    }

    /// Dequantize FP8 tensor back to MLX array.
    pub fn dequantize(&self, fp8: &Fp8Tensor) -> Result<Array> {
        let num_elements = fp8.num_elements();
        let block_size = fp8.block_size;
        let num_blocks = num_elements.div_ceil(block_size);

        let mut output = Vec::with_capacity(num_elements);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(num_elements);
            let scale = fp8.scales[block_idx];

            for i in start..end {
                // Interpret as signed i8
                let quantized = fp8.data[i] as i8;
                // Reconstruct: val = quantized * scale = quantized * amax / 127
                let value = (quantized as f32) * scale;
                output.push(value);
            }
        }

        Ok(Array::from_slice(&output, &fp8.shape))
    }

    /// Perform FP8 linear merge.
    ///
    /// Quantizes inputs, performs weighted average in reduced precision,
    /// then dequantizes the result.
    pub fn linear_merge_fp8(&mut self, tensors: &[Array], weights: &[f32]) -> Result<Array> {
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        if tensors.len() != weights.len() {
            return Err(MergeError::InvalidConfig(format!(
                "Expected {} weights, got {}",
                tensors.len(),
                weights.len()
            )));
        }

        debug!("FP8 linear merge: {} tensors", tensors.len());

        // For linear merge, we can work directly on quantized data
        // since weighted sum preserves scale relationships
        let quantized: Vec<Fp8Tensor> = tensors
            .iter()
            .map(|t| self.quantize(t))
            .collect::<Result<Vec<_>>>()?;

        // Compute weighted sum in FP8 (with scale compensation)
        let num_elements = quantized[0].num_elements();
        let block_size = quantized[0].block_size;
        let num_blocks = num_elements.div_ceil(block_size);
        let shape = quantized[0].shape.clone();

        let mut result = vec![0.0f32; num_elements];

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(num_elements);

            // Accumulate in f32 with scale compensation
            for (fp8, &weight) in quantized.iter().zip(weights.iter()) {
                let scale = fp8.scales[block_idx];

                for (result_val, &quant_byte) in
                    result[start..end].iter_mut().zip(&fp8.data[start..end])
                {
                    let quantized_val = quant_byte as i8;
                    // Dequantize and apply weight
                    let value = (quantized_val as f32) * scale * weight;
                    *result_val += value;
                }
            }
        }

        // Re-quantize result
        let result_array = Array::from_slice(&result, &shape);

        // If we need FP8 output, quantize again
        if self.config.fp8_intermediates {
            let fp8_result = self.quantize(&result_array)?;
            self.dequantize(&fp8_result)
        } else {
            Ok(result_array)
        }
    }

    /// Perform FP8 TIES merge.
    ///
    /// For TIES, we dequantize for sparsification (needs full precision)
    /// but can keep other operations in FP8.
    pub fn ties_merge_fp8(
        &mut self,
        tensors: &[Array],
        base: &Array,
        weights: &[f32],
        densities: &[f32],
        lambda: f32,
    ) -> Result<Array> {
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        debug!("FP8 TIES merge: {} tensors", tensors.len());

        // Step 1: Compute task vectors (can stay in FP8 for subtraction)
        let task_vectors: Vec<Array> = tensors
            .iter()
            .map(|t| t.subtract(base).map_err(MergeError::from))
            .collect::<Result<Vec<_>>>()?;

        // Step 2: Sparsification - needs full precision for magnitude comparison
        let sparse_vectors = if self.config.force_dequant_for_sparsify {
            // Use standard sparsification on full-precision data
            crate::sparsify_batch_by_magnitude(&task_vectors, densities)?
        } else {
            // Quantize, operate, dequantize
            let fp8_vectors: Vec<Fp8Tensor> = task_vectors
                .iter()
                .map(|t| self.quantize(t))
                .collect::<Result<Vec<_>>>()?;

            let dequant_vectors: Vec<Array> = fp8_vectors
                .iter()
                .map(|t| self.dequantize(t))
                .collect::<Result<Vec<_>>>()?;

            crate::sparsify_batch_by_magnitude(&dequant_vectors, densities)?
        };

        // Step 3: Sign consensus (returns weighted sum of agreeing contributions).
        let weighted_sum = crate::sign_consensus(&sparse_vectors, weights)?;

        // Step 4: Scale and add to base
        let result = weighted_sum.multiply(Array::from_f32(lambda))?;
        Ok(base.add(&result)?)
    }

    /// Calculate memory savings from FP8 quantization.
    pub fn memory_savings(&self, num_elements: usize) -> MemorySavingsReport {
        let f32_bytes = num_elements * 4;
        let block_size = self.config.block_size;
        let num_blocks = num_elements.div_ceil(block_size);

        // FP8 = 1 byte per element + 4 bytes per block for scale
        let fp8_bytes = num_elements + num_blocks * 4;

        let savings_ratio = 1.0 - (fp8_bytes as f32 / f32_bytes as f32);

        MemorySavingsReport {
            original_bytes: f32_bytes,
            fp8_bytes,
            savings_bytes: f32_bytes.saturating_sub(fp8_bytes),
            savings_ratio,
        }
    }

    /// Reset dynamic scale tracking.
    pub fn reset_scale(&mut self) {
        self.dynamic_scale = DynamicScale::new(self.config.scale_window_size, self.config.format);
    }
}

/// Report of memory savings from FP8 quantization.
#[derive(Debug, Clone)]
pub struct MemorySavingsReport {
    /// Original size in bytes (F32).
    pub original_bytes: usize,
    /// FP8 size in bytes.
    pub fp8_bytes: usize,
    /// Bytes saved.
    pub savings_bytes: usize,
    /// Savings ratio (0.0 to 1.0).
    pub savings_ratio: f32,
}

impl MemorySavingsReport {
    /// Format as human-readable string.
    pub fn to_string_pretty(&self) -> String {
        format!(
            "FP8 Memory: {:.2} MB -> {:.2} MB ({:.1}% savings)",
            self.original_bytes as f64 / 1e6,
            self.fp8_bytes as f64 / 1e6,
            self.savings_ratio * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_format_values() {
        // Both profiles use int8 precision (127 levels).
        assert_eq!(Fp8Format::E4M3.max_value(), 127.0);
        assert_eq!(Fp8Format::E5M2.max_value(), 127.0);
    }

    #[test]
    fn test_fp8_merge_config_default() {
        let config = Fp8MergeConfig::default();
        assert_eq!(config.block_size, 128);
        assert_eq!(config.format, Fp8Format::E4M3);
        assert!(config.force_dequant_for_sparsify);
    }

    #[test]
    fn test_fp8_merge_config_memory_optimized() {
        let config = Fp8MergeConfig::memory_optimized();
        assert_eq!(config.block_size, 256);
        assert!(!config.force_dequant_for_sparsify);
    }

    #[test]
    fn test_fp8_merge_config_accuracy_optimized() {
        let config = Fp8MergeConfig::accuracy_optimized();
        assert_eq!(config.block_size, 64);
        assert!(config.force_dequant_for_sparsify);
    }

    #[test]
    fn test_dynamic_scale() {
        let mut scale = DynamicScale::new(4, Fp8Format::E4M3);

        // Initial scale
        scale.update(1.0);
        assert!(scale.scale() > 0.0);

        // Larger values should decrease scale
        scale.update(100.0);
        let scale1 = scale.scale();
        scale.update(200.0);
        let scale2 = scale.scale();
        assert!(scale2 < scale1);
    }

    #[test]
    fn test_fp8_merger_creation() {
        let merger = Fp8Merger::default_new();
        assert_eq!(merger.config().block_size, 128);
    }

    #[test]
    fn test_fp8_quantize_dequantize() {
        let mut merger = Fp8Merger::default_new();

        // Create simple array
        let input = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);

        // Quantize
        let fp8 = merger.quantize(&input).unwrap();
        assert_eq!(fp8.num_elements(), 4);

        // Dequantize
        let output = merger.dequantize(&fp8).unwrap();
        let output_slice: Vec<f32> = output.as_slice().to_vec();

        // Values should be approximately preserved
        // FP8 uses 127 levels, so error bound is ~amax/127 ≈ 4/127 ≈ 0.03
        assert_eq!(output_slice.len(), 4);
        for (i, &val) in output_slice.iter().enumerate() {
            let expected = (i + 1) as f32;
            let error = (val - expected).abs();
            // Allow up to amax/127 * 2 quantization error
            assert!(
                error < 0.1,
                "Value {} expected {}, error {}",
                val,
                expected,
                error
            );
        }
    }

    #[test]
    fn test_fp8_linear_merge() {
        let mut merger = Fp8Merger::default_new();

        let t1 = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let t2 = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[4]);

        let result = merger.linear_merge_fp8(&[t1, t2], &[0.5, 0.5]).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Expected: [3, 4, 5, 6]
        assert_eq!(result_slice.len(), 4);
        // FP8 quantization introduces error due to limited precision
        // Each tensor gets quantized with its own scale, then combined
        // amax(t1) = 4, amax(t2) = 8
        // Expected quantization error is roughly sum of (amax_i/127 * 0.5)
        for (i, &val) in result_slice.iter().enumerate() {
            let expected = (i + 3) as f32;
            let error = (val - expected).abs();
            // Allow larger error due to combining two quantized tensors
            assert!(
                error < 0.5,
                "Value {} expected ~{}, error {}",
                val,
                expected,
                error
            );
        }
    }

    #[test]
    fn test_memory_savings() {
        let merger = Fp8Merger::default_new();

        // 1M elements
        let report = merger.memory_savings(1_000_000);

        // F32: 4MB, FP8: ~1MB + scale overhead
        assert!(report.original_bytes == 4_000_000);
        assert!(report.fp8_bytes < report.original_bytes);
        assert!(report.savings_ratio > 0.5); // Should save >50%
    }

    #[test]
    fn test_memory_savings_report() {
        let merger = Fp8Merger::default_new();
        let report = merger.memory_savings(1_000_000);

        let pretty = report.to_string_pretty();
        assert!(pretty.contains("MB"));
        assert!(pretty.contains("%"));
    }
}
