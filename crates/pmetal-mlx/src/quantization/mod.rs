//! Quantization implementations for memory-efficient training.
//!
//! This module provides implementations of various quantization schemes:
//! - NF4 (Normal Float 4-bit) - optimal for neural network weights
//! - FP4 (Floating Point 4-bit) - simpler alternative
//! - Int8 (8-bit integer) - for activations
//! - Group-wise quantization (GPTQ/AWQ format) - for loading pre-quantized models

pub mod fp4;
pub mod group;
pub mod int8;
pub mod nf4;

pub use fp4::*;
pub use group::*;
pub use int8::*;
pub use nf4::*;

use pmetal_core::Result;

/// Quantized tensor representation.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data (packed for 4-bit).
    pub data: Vec<u8>,
    /// Per-block absolute maximum values for dequantization.
    pub absmax: Vec<f32>,
    /// Quantized absmax for double quantization (optional).
    pub absmax_quant: Option<QuantizedAbsmax>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Block size used for quantization.
    pub block_size: usize,
    /// Quantization scheme used.
    pub scheme: QuantScheme,
}

/// Quantized absolute maximum values (for double quantization).
#[derive(Debug, Clone)]
pub struct QuantizedAbsmax {
    /// Quantized absmax data.
    pub data: Vec<u8>,
    /// Offset for dequantization.
    pub offset: f32,
    /// Scale for dequantization.
    pub scale: f32,
}

/// Quantization scheme identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// NF4 quantization.
    NF4,
    /// FP4 quantization.
    FP4,
    /// Int8 quantization.
    Int8,
    /// FP8 quantization.
    FP8,
    /// Group-wise 2-bit quantization (GPTQ/AWQ compatible).
    Group2,
    /// Group-wise 3-bit quantization (GPTQ/AWQ compatible).
    Group3,
    /// Group-wise 4-bit quantization (GPTQ/AWQ compatible).
    Group4,
    /// Group-wise 8-bit quantization (GPTQ/AWQ compatible).
    Group8,
}

/// Common interface for all quantizers.
pub trait QuantizerOps {
    /// Quantize a tensor.
    fn quantize(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedTensor>;

    /// Dequantize a tensor.
    fn dequantize(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>>;

    /// Get the quantization scheme.
    fn scheme(&self) -> QuantScheme;
}
