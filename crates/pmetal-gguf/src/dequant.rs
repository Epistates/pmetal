//! Dequantization routines for GGUF quantized tensors.
//!
//! This module provides functions to convert quantized tensor data from GGUF files
//! to floating-point representations.
//!
//! # Supported Quantization Types
//!
//! - **F32**: No conversion needed
//! - **F16**: Half-precision to f32
//! - **BF16**: Brain float to f32
//! - **Q4_0**: 4-bit quantization (32 values per block)
//! - **Q8_0**: 8-bit quantization (32 values per block)
//! - **Q2K-Q8K**: K-quant types (256 values per block, hierarchical scaling)
//! - **Q1_0/TQ/MXFP4/NVFP4**: current GGML low-bit and FP4 formats
//!
//! # Block Quantization
//!
//! Most GGUF quantization formats use block quantization where values are grouped
//! into blocks (typically 32 values) that share a single scale factor.
//!
//! For Q4_0:
//! - Block size: 32 values
//! - Block bytes: 18 (2 bytes scale + 16 bytes data)
//! - Dequantization: `value = (quant - 8) * scale`
//!
//! For Q8_0:
//! - Block size: 32 values
//! - Block bytes: 34 (2 bytes scale + 32 bytes data)
//! - Dequantization: `value = quant * scale`

use crate::GgmlType;
use crate::fp4::{KVALUES_MXFP4, e8m0_to_fp32_half, ue4m3_to_fp32};
use crate::iq_quants;
use crate::k_quants;

/// Error type for dequantization.
#[derive(Debug, thiserror::Error)]
pub enum DequantError {
    /// Unsupported quantization type.
    #[error("Unsupported quantization type: {0:?}")]
    UnsupportedType(GgmlType),
    /// Invalid data size for the given shape.
    #[error("Invalid data size: expected {expected} bytes, got {actual}")]
    InvalidSize {
        /// Expected size in bytes.
        expected: usize,
        /// Actual size in bytes.
        actual: usize,
    },
    /// Shape mismatch.
    #[error("Shape mismatch: cannot reshape {elements} elements to {shape:?}")]
    ShapeMismatch {
        /// Number of elements.
        elements: usize,
        /// Target shape.
        shape: Vec<i32>,
    },
    /// Invalid shape dimension.
    #[error("Invalid shape {shape:?}: dimensions must be non-negative")]
    InvalidShape {
        /// Target shape.
        shape: Vec<i32>,
    },
    /// Shape element count overflowed usize.
    #[error("Shape {shape:?} is too large")]
    ShapeTooLarge {
        /// Target shape.
        shape: Vec<i32>,
    },
}

/// Dequantize raw tensor data to f32 values.
///
/// # Arguments
///
/// * `data` - Raw quantized bytes from GGUF file
/// * `dtype` - The quantization type
/// * `shape` - Target tensor shape
///
/// # Returns
///
/// Vector of f32 values in row-major order
///
/// # Example
///
/// ```ignore
/// let data = content.read_tensor_data(&mut file, "blk.0.attn_q.weight")?;
/// let info = content.get_tensor_info("blk.0.attn_q.weight").unwrap();
/// let shape: Vec<i32> = info.dimensions.iter().map(|&d| d as i32).collect();
/// let floats = dequantize(&data, info.dtype, &shape)?;
/// ```
pub fn dequantize(data: &[u8], dtype: GgmlType, shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    match dtype {
        GgmlType::F32 => dequantize_f32(data, shape),
        GgmlType::F16 => dequantize_f16(data, shape),
        GgmlType::Bf16 => dequantize_bf16(data, shape),
        GgmlType::Q4_0 => dequantize_q4_0(data, shape),
        GgmlType::Q4_1 => dequantize_q4_1(data, shape),
        GgmlType::Q5_0 => dequantize_q5_0(data, shape),
        GgmlType::Q5_1 => dequantize_q5_1(data, shape),
        GgmlType::Q8_0 => dequantize_q8_0(data, shape),
        GgmlType::Q8_1 => dequantize_q8_1(data, shape),
        GgmlType::Q1_0 => dequantize_q1_0(data, shape),
        GgmlType::Tq1_0 => dequantize_tq1_0(data, shape),
        GgmlType::Tq2_0 => dequantize_tq2_0(data, shape),
        GgmlType::Mxfp4 => dequantize_mxfp4(data, shape),
        GgmlType::Nvfp4 => dequantize_nvfp4(data, shape),
        GgmlType::I8 => dequantize_i8(data, shape),
        GgmlType::I16 => dequantize_i16(data, shape),
        GgmlType::I32 => dequantize_i32(data, shape),
        GgmlType::I64 => dequantize_i64(data, shape),
        GgmlType::F64 => dequantize_f64(data, shape),
        // K-quant types
        GgmlType::Q2K => dequantize_q2k(data, shape),
        GgmlType::Q3K => dequantize_q3k(data, shape),
        GgmlType::Q4K => dequantize_q4k(data, shape),
        GgmlType::Q5K => dequantize_q5k(data, shape),
        GgmlType::Q6K => dequantize_q6k(data, shape),
        GgmlType::Q8K => dequantize_q8k(data, shape),
        // IQ types (importance-weighted quantization)
        GgmlType::Iq4Nl => dequantize_iq4nl(data, shape),
        GgmlType::Iq4Xs => dequantize_iq4xs(data, shape),
        GgmlType::Iq2Xxs => dequantize_iq2xxs(data, shape),
        GgmlType::Iq2Xs => dequantize_iq2xs(data, shape),
        GgmlType::Iq2S => dequantize_iq2s(data, shape),
        GgmlType::Iq3Xxs => dequantize_iq3xxs(data, shape),
        GgmlType::Iq3S => dequantize_iq3s(data, shape),
        GgmlType::Iq1S => dequantize_iq1s(data, shape),
        GgmlType::Iq1M => dequantize_iq1m(data, shape),
        other => Err(DequantError::UnsupportedType(other)),
    }
}

/// Check if a quantization type is supported for dequantization.
pub fn is_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Bf16
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8_1
            | GgmlType::Q1_0
            | GgmlType::Tq1_0
            | GgmlType::Tq2_0
            | GgmlType::Mxfp4
            | GgmlType::Nvfp4
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64
            | GgmlType::F64
            // K-quant types
            | GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Q8K
            // IQ types
            | GgmlType::Iq4Nl
            | GgmlType::Iq4Xs
            | GgmlType::Iq2Xxs
            | GgmlType::Iq2Xs
            | GgmlType::Iq2S
            | GgmlType::Iq3Xxs
            | GgmlType::Iq3S
            | GgmlType::Iq1S
            | GgmlType::Iq1M
    )
}

/// Get the expected byte size for a tensor with given dtype and shape.
pub fn expected_byte_size(dtype: GgmlType, shape: &[i32]) -> usize {
    expected_byte_size_checked(dtype, shape)
        .expect("invalid GGUF tensor shape for expected_byte_size")
}

/// Get the expected byte size for a tensor with checked shape arithmetic.
pub fn expected_byte_size_checked(dtype: GgmlType, shape: &[i32]) -> Result<usize, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    dtype
        .tensor_size_checked(n_elements)
        .ok_or_else(|| DequantError::ShapeTooLarge {
            shape: shape.to_vec(),
        })
}

fn shape_n_elements(shape: &[i32]) -> Result<usize, DequantError> {
    if shape.iter().any(|&d| d < 0) {
        return Err(DequantError::InvalidShape {
            shape: shape.to_vec(),
        });
    }

    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim as usize)
            .ok_or_else(|| DequantError::ShapeTooLarge {
                shape: shape.to_vec(),
            })
    })
}

fn validate_data_size(data: &[u8], expected: usize) -> Result<(), DequantError> {
    if data.len() != expected {
        return Err(DequantError::InvalidSize {
            expected,
            actual: data.len(),
        });
    }
    Ok(())
}

/// F32 - direct conversion (no dequantization needed).
fn dequantize_f32(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = n_elements * 4;
    validate_data_size(data, expected_size)?;

    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(floats)
}

/// F16 - half-precision to f32.
fn dequantize_f16(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = n_elements * 2;
    validate_data_size(data, expected_size)?;

    let floats: Vec<f32> = data
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();

    Ok(floats)
}

/// BF16 - brain float to f32.
fn dequantize_bf16(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = n_elements * 2;
    validate_data_size(data, expected_size)?;

    let floats: Vec<f32> = data
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();

    Ok(floats)
}

/// Q1_0 dequantization.
///
/// Block structure (18 bytes per 128 values):
/// - bytes 0-1: scale `d` (f16)
/// - bytes 2-17: 128 sign bits packed LSB-first
///
/// Dequantization: `value = bit ? d : -d`
fn dequantize_q1_0(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    const BLOCK_SIZE: usize = 128;
    const BLOCK_BYTES: usize = 18;

    let n_elements = shape_n_elements(shape)?;
    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    let expected_size = n_blocks * BLOCK_BYTES;
    validate_data_size(data, expected_size)?;

    let mut output = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();

        for i in 0..BLOCK_SIZE {
            let byte = block[2 + i / 8];
            let bit = (byte >> (i % 8)) & 1;
            output.push(if bit == 1 { scale } else { -scale });
        }
    }

    output.truncate(n_elements);
    Ok(output)
}

/// Q4_0 dequantization.
///
/// Block structure (18 bytes per 32 values):
/// - bytes 0-1: scale (f16)
/// - bytes 2-17: 32 4-bit quantized values packed in 16 bytes
///
/// Dequantization: `value = (quant - 8) * scale`
fn dequantize_q4_0(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let n_elements = shape_n_elements(shape)?;
    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    let expected_size = n_blocks * BLOCK_BYTES;
    validate_data_size(data, expected_size)?;

    let mut output = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];
        let mut block_values = [0.0f32; BLOCK_SIZE];

        // Scale is f16 at start of block
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();

        // Low nibbles encode elements 0..16, high nibbles encode 16..32.
        for i in 0..16 {
            let byte = block[2 + i];
            let low = (byte & 0x0F) as i8 - 8;
            let high = ((byte >> 4) & 0x0F) as i8 - 8;
            block_values[i] = low as f32 * scale;
            block_values[i + 16] = high as f32 * scale;
        }
        output.extend_from_slice(&block_values);
    }

    // Truncate to exact element count (last block may be partial)
    output.truncate(n_elements);

    Ok(output)
}

/// Q4_1 dequantization.
///
/// Block structure (20 bytes per 32 values):
/// - bytes 0-1: scale `d` (f16)
/// - bytes 2-3: minimum `m` (f16)
/// - bytes 4-19: 32 unsigned 4-bit values packed in 16 bytes
///
/// Dequantization: `value = quant * d + m`
fn dequantize_q4_1(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 20;

    let n_elements = shape_n_elements(shape)?;
    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    let expected_size = n_blocks * BLOCK_BYTES;
    validate_data_size(data, expected_size)?;

    let mut output = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];
        let mut block_values = [0.0f32; BLOCK_SIZE];
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let min = half::f16::from_le_bytes([block[2], block[3]]).to_f32();

        for i in 0..16 {
            let byte = block[4 + i];
            block_values[i] = (byte & 0x0F) as f32 * scale + min;
            block_values[i + 16] = (byte >> 4) as f32 * scale + min;
        }
        output.extend_from_slice(&block_values);
    }

    output.truncate(n_elements);
    Ok(output)
}

/// Q5_0 dequantization.
///
/// Block structure (22 bytes per 32 values):
/// - bytes 0-1: scale `d` (f16)
/// - bytes 2-5: high fifth bits for all 32 values
/// - bytes 6-21: low 4-bit nibbles
///
/// Dequantization: `value = (quant - 16) * d`
fn dequantize_q5_0(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 22;

    let n_elements = shape_n_elements(shape)?;
    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    let expected_size = n_blocks * BLOCK_BYTES;
    validate_data_size(data, expected_size)?;

    let mut output = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];
        let mut block_values = [0.0f32; BLOCK_SIZE];
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];

        for i in 0..16 {
            let byte = qs[i];
            let high_lo = ((qh >> i) & 1) as i32;
            let high_hi = ((qh >> (i + 16)) & 1) as i32;
            let q_lo = (byte & 0x0F) as i32 | (high_lo << 4);
            let q_hi = (byte >> 4) as i32 | (high_hi << 4);
            block_values[i] = (q_lo - 16) as f32 * scale;
            block_values[i + 16] = (q_hi - 16) as f32 * scale;
        }
        output.extend_from_slice(&block_values);
    }

    output.truncate(n_elements);
    Ok(output)
}

/// Q5_1 dequantization.
///
/// Block structure (24 bytes per 32 values):
/// - bytes 0-1: scale `d` (f16)
/// - bytes 2-3: minimum `m` (f16)
/// - bytes 4-7: high fifth bits for all 32 values
/// - bytes 8-23: low 4-bit nibbles
///
/// Dequantization: `value = quant * d + m`
fn dequantize_q5_1(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 24;

    let n_elements = shape_n_elements(shape)?;
    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    let expected_size = n_blocks * BLOCK_BYTES;
    validate_data_size(data, expected_size)?;

    let mut output = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];
        let mut block_values = [0.0f32; BLOCK_SIZE];
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let min = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];

        for i in 0..16 {
            let byte = qs[i];
            let high_lo = ((qh >> i) & 1) as i32;
            let high_hi = ((qh >> (i + 16)) & 1) as i32;
            let q_lo = (byte & 0x0F) as i32 | (high_lo << 4);
            let q_hi = (byte >> 4) as i32 | (high_hi << 4);
            block_values[i] = q_lo as f32 * scale + min;
            block_values[i + 16] = q_hi as f32 * scale + min;
        }
        output.extend_from_slice(&block_values);
    }

    output.truncate(n_elements);
    Ok(output)
}

/// Q8_0 dequantization.
///
/// Block structure (34 bytes per 32 values):
/// - bytes 0-1: scale (f16)
/// - bytes 2-33: 32 8-bit quantized values
///
/// Dequantization: `value = quant * scale`
fn dequantize_q8_0(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;

    let n_elements = shape_n_elements(shape)?;
    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    let expected_size = n_blocks * BLOCK_BYTES;
    validate_data_size(data, expected_size)?;

    let mut output = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];

        // Scale is f16 at start of block
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();

        // 32 i8 values
        for i in 0..32 {
            let quant = block[2 + i] as i8;
            output.push(quant as f32 * scale);
        }
    }

    // Truncate to exact element count (last block may be partial)
    output.truncate(n_elements);

    Ok(output)
}

/// Q8_1 dequantization.
///
/// Block structure (36 bytes per 32 values):
/// - bytes 0-1: scale `d` (f16)
/// - bytes 2-3: sum helper `s` (f16), used by dot kernels but not needed here
/// - bytes 4-35: 32 i8 quantized values
///
/// Dequantization: `value = quant * d`
fn dequantize_q8_1(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 36;

    let n_elements = shape_n_elements(shape)?;
    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    let expected_size = n_blocks * BLOCK_BYTES;
    validate_data_size(data, expected_size)?;

    let mut output = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();

        for i in 0..32 {
            let quant = block[4 + i] as i8;
            output.push(quant as f32 * scale);
        }
    }

    output.truncate(n_elements);
    Ok(output)
}

/// MXFP4 dequantization.
///
/// Block structure (17 bytes per 32 values):
/// - byte 0: E8M0 shared scale
/// - bytes 1-16: packed 4-bit E2M1 values
fn dequantize_mxfp4(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 17;

    let n_elements = shape_n_elements(shape)?;
    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    let expected_size = n_blocks * BLOCK_BYTES;
    validate_data_size(data, expected_size)?;

    let mut output = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];
        let scale = e8m0_to_fp32_half(block[0]);
        let mut block_values = [0.0f32; BLOCK_SIZE];

        for i in 0..16 {
            let byte = block[1 + i];
            let low = KVALUES_MXFP4[(byte & 0x0f) as usize];
            let high = KVALUES_MXFP4[(byte >> 4) as usize];
            block_values[i] = low as f32 * scale;
            block_values[i + 16] = high as f32 * scale;
        }
        output.extend_from_slice(&block_values);
    }

    output.truncate(n_elements);
    Ok(output)
}

/// NVFP4 dequantization.
///
/// Block structure (36 bytes per 64 values):
/// - bytes 0-3: UE4M3 scales, one per 16-value sub-block
/// - bytes 4-35: packed 4-bit E2M1 values
fn dequantize_nvfp4(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    const BLOCK_SIZE: usize = 64;
    const SUB_BLOCK_SIZE: usize = 16;
    const N_SUB_BLOCKS: usize = BLOCK_SIZE / SUB_BLOCK_SIZE;
    const BLOCK_BYTES: usize = 36;

    let n_elements = shape_n_elements(shape)?;
    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    let expected_size = n_blocks * BLOCK_BYTES;
    validate_data_size(data, expected_size)?;

    let mut output = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];
        let mut block_values = [0.0f32; BLOCK_SIZE];

        for sub in 0..N_SUB_BLOCKS {
            let scale = ue4m3_to_fp32(block[sub]);
            let qs_offset = 4 + sub * (SUB_BLOCK_SIZE / 2);
            let value_offset = sub * SUB_BLOCK_SIZE;

            for i in 0..SUB_BLOCK_SIZE / 2 {
                let byte = block[qs_offset + i];
                let low = KVALUES_MXFP4[(byte & 0x0f) as usize];
                let high = KVALUES_MXFP4[(byte >> 4) as usize];
                block_values[value_offset + i] = low as f32 * scale;
                block_values[value_offset + i + SUB_BLOCK_SIZE / 2] = high as f32 * scale;
            }
        }
        output.extend_from_slice(&block_values);
    }

    output.truncate(n_elements);
    Ok(output)
}

/// TQ1_0 dequantization.
///
/// Block structure (54 bytes per 256 values):
/// - bytes 0-47: 5 ternary values per byte
/// - bytes 48-51: 4 ternary values per byte
/// - bytes 52-53: scale `d` (f16)
fn dequantize_tq1_0(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    const BLOCK_SIZE: usize = 256;
    const QS_BYTES: usize = 48;
    const QH_BYTES: usize = 4;
    const BLOCK_BYTES: usize = 54;
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

    let n_elements = shape_n_elements(shape)?;
    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    let expected_size = n_blocks * BLOCK_BYTES;
    validate_data_size(data, expected_size)?;

    let mut output = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];
        let qs = &block[..QS_BYTES];
        let qh = &block[QS_BYTES..QS_BYTES + QH_BYTES];
        let scale = half::f16::from_le_bytes([block[52], block[53]]).to_f32();

        for j in (0..QS_BYTES - QS_BYTES % 32).step_by(32) {
            for pow in POW3.iter().take(5) {
                for m in 0..32 {
                    let q = qs[j + m].wrapping_mul(*pow);
                    let xi = ((u16::from(q) * 3) >> 8) as i16;
                    output.push((xi - 1) as f32 * scale);
                }
            }
        }
        for j in (QS_BYTES - QS_BYTES % 32..QS_BYTES).step_by(16) {
            for pow in POW3.iter().take(5) {
                for m in 0..16 {
                    let q = qs[j + m].wrapping_mul(*pow);
                    let xi = ((u16::from(q) * 3) >> 8) as i16;
                    output.push((xi - 1) as f32 * scale);
                }
            }
        }
        for pow in POW3.iter().take(4) {
            for qh_byte in qh {
                let q = qh_byte.wrapping_mul(*pow);
                let xi = ((u16::from(q) * 3) >> 8) as i16;
                output.push((xi - 1) as f32 * scale);
            }
        }
    }

    output.truncate(n_elements);
    Ok(output)
}

/// TQ2_0 dequantization.
///
/// Block structure (66 bytes per 256 values):
/// - bytes 0-63: four 2-bit ternary values per byte
/// - bytes 64-65: scale `d` (f16)
fn dequantize_tq2_0(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    const BLOCK_SIZE: usize = 256;
    const QS_BYTES: usize = 64;
    const BLOCK_BYTES: usize = 66;

    let n_elements = shape_n_elements(shape)?;
    let n_blocks = n_elements.div_ceil(BLOCK_SIZE);
    let expected_size = n_blocks * BLOCK_BYTES;
    validate_data_size(data, expected_size)?;

    let mut output = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];
        let scale = half::f16::from_le_bytes([block[64], block[65]]).to_f32();

        for j in (0..QS_BYTES).step_by(32) {
            for lane in 0..4 {
                for m in 0..32 {
                    let q = ((block[j + m] >> (lane * 2)) & 3) as i8;
                    output.push((q - 1) as f32 * scale);
                }
            }
        }
    }

    output.truncate(n_elements);
    Ok(output)
}

fn dequantize_i8(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    validate_data_size(data, n_elements)?;
    Ok(data.iter().map(|&value| (value as i8) as f32).collect())
}

fn dequantize_i16(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    validate_data_size(data, n_elements * 2)?;
    Ok(data
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32)
        .collect())
}

fn dequantize_i32(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    validate_data_size(data, n_elements * 4)?;
    Ok(data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
        .collect())
}

fn dequantize_i64(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    validate_data_size(data, n_elements * 8)?;
    Ok(data
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
        .collect())
}

fn dequantize_f64(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    validate_data_size(data, n_elements * 8)?;
    Ok(data
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
        .collect())
}

// =============================================================================
// K-Quant Dequantization Wrappers (Q2K-Q8K)
// =============================================================================

/// Q2K dequantization (2-bit K-quant, 256 values per block).
fn dequantize_q2k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = k_quants::BlockQ2K::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(k_quants::dequantize_q2k_bytes(data, n_elements))
}

/// Q3K dequantization (3-bit K-quant, 256 values per block).
fn dequantize_q3k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = k_quants::BlockQ3K::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(k_quants::dequantize_q3k_bytes(data, n_elements))
}

/// Q4K dequantization (4-bit K-quant, 256 values per block).
fn dequantize_q4k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = k_quants::BlockQ4K::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(k_quants::dequantize_q4k_bytes(data, n_elements))
}

/// Q5K dequantization (5-bit K-quant, 256 values per block).
fn dequantize_q5k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = k_quants::BlockQ5K::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(k_quants::dequantize_q5k_bytes(data, n_elements))
}

/// Q6K dequantization (6-bit K-quant, 256 values per block).
fn dequantize_q6k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = k_quants::BlockQ6K::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(k_quants::dequantize_q6k_bytes(data, n_elements))
}

/// Q8K dequantization (8-bit K-quant, 256 values per block).
fn dequantize_q8k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = k_quants::BlockQ8K::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(k_quants::dequantize_q8k_bytes(data, n_elements))
}

// =============================================================================
// IQ Dequantization Wrappers (Importance-weighted Quantization)
// =============================================================================

/// IQ4_NL dequantization (4-bit non-linear, 32 values per block).
fn dequantize_iq4nl(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = iq_quants::BlockIq4Nl::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(iq_quants::dequantize_iq4nl_bytes(data, n_elements))
}

/// IQ4_XS dequantization (4-bit non-linear with scales, 256 values per block).
fn dequantize_iq4xs(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = iq_quants::BlockIq4Xs::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(iq_quants::dequantize_iq4xs_bytes(data, n_elements))
}

/// IQ2_XXS dequantization (2-bit extra-extra-small, 256 values per block).
fn dequantize_iq2xxs(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = iq_quants::BlockIq2Xxs::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(iq_quants::dequantize_iq2xxs_bytes(data, n_elements))
}

/// IQ2_XS dequantization (2-bit extra-small, 256 values per block).
fn dequantize_iq2xs(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = iq_quants::BlockIq2Xs::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(iq_quants::dequantize_iq2xs_bytes(data, n_elements))
}

/// IQ2_S dequantization (2-bit small, 256 values per block).
fn dequantize_iq2s(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = iq_quants::BlockIq2S::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(iq_quants::dequantize_iq2s_bytes(data, n_elements))
}

/// IQ3_XXS dequantization (3-bit extra-extra-small, 256 values per block).
fn dequantize_iq3xxs(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = iq_quants::BlockIq3Xxs::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(iq_quants::dequantize_iq3xxs_bytes(data, n_elements))
}

/// IQ3_S dequantization (3-bit small, 256 values per block).
fn dequantize_iq3s(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = iq_quants::BlockIq3S::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(iq_quants::dequantize_iq3s_bytes(data, n_elements))
}

/// IQ1_S dequantization (1-bit small, 256 values per block).
fn dequantize_iq1s(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = iq_quants::BlockIq1S::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(iq_quants::dequantize_iq1s_bytes(data, n_elements))
}

/// IQ1_M dequantization (1-bit medium, 256 values per block).
fn dequantize_iq1m(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements = shape_n_elements(shape)?;
    let expected_size = iq_quants::BlockIq1M::byte_size(n_elements);
    validate_data_size(data, expected_size)?;

    Ok(iq_quants::dequantize_iq1m_bytes(data, n_elements))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_f32() {
        let data: Vec<u8> = vec![
            0x00, 0x00, 0x80, 0x3f, // 1.0f
            0x00, 0x00, 0x00, 0x40, // 2.0f
            0x00, 0x00, 0x40, 0x40, // 3.0f
        ];
        let shape = [3];
        let result = dequantize_f32(&data, &shape).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dequantize_f16() {
        // f16 representations
        let one_f16 = half::f16::from_f32(1.0).to_le_bytes();
        let two_f16 = half::f16::from_f32(2.0).to_le_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&one_f16);
        data.extend_from_slice(&two_f16);

        let shape = [2];
        let result = dequantize_f16(&data, &shape).unwrap();

        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_bf16() {
        // bf16 representations
        let one_bf16 = half::bf16::from_f32(1.0).to_le_bytes();
        let two_bf16 = half::bf16::from_f32(2.0).to_le_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&one_bf16);
        data.extend_from_slice(&two_bf16);

        let shape = [2];
        let result = dequantize_bf16(&data, &shape).unwrap();

        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_q4_0_block_structure() {
        // Create a Q4_0 block with scale=1.0 and values 0-7 repeated
        let scale = half::f16::from_f32(1.0).to_le_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&scale);

        // 16 bytes of packed 4-bit values
        // Each byte contains two 4-bit values: low nibble first, high nibble second
        // For value i, the quantized value is i + 8 (since we subtract 8 during dequant)
        for i in 0..16u8 {
            // Pack two values per byte
            let low = (i % 16 + 8) & 0x0F; // Will be (i+8)-8 = i after dequant
            let high = ((i + 1) % 16 + 8) & 0x0F;
            data.push(low | (high << 4));
        }

        let shape = [32];
        let result = dequantize_q4_0(&data, &shape).unwrap();

        assert_eq!(result.len(), 32);
        // Verify first few values
        assert!((result[0] - 0.0).abs() < 0.01); // (8 - 8) * 1.0 = 0
        assert!((result[1] - 1.0).abs() < 0.01); // (9 - 8) * 1.0 = 1
        assert!((result[16] - 1.0).abs() < 0.01); // high nibble from first byte
    }

    #[test]
    fn test_q1_0_block_structure() {
        let scale = half::f16::from_f32(0.5).to_le_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&scale);
        data.extend(std::iter::repeat(0b1010_1010).take(16));

        let result = dequantize_q1_0(&data, &[128]).unwrap();
        assert_eq!(result.len(), 128);
        assert!((result[0] + 0.5).abs() < 0.01);
        assert!((result[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_q4_1_block_structure() {
        let scale = half::f16::from_f32(0.5).to_le_bytes();
        let min = half::f16::from_f32(1.0).to_le_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&scale);
        data.extend_from_slice(&min);
        data.push(2 | (4 << 4));
        data.extend(std::iter::repeat(0).take(15));

        let result = dequantize_q4_1(&data, &[32]).unwrap();
        assert!((result[0] - 2.0).abs() < 0.01);
        assert!((result[16] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_q5_legacy_block_structures() {
        let scale = half::f16::from_f32(1.0).to_le_bytes();
        let min = half::f16::from_f32(0.25).to_le_bytes();

        let mut q5_0 = Vec::new();
        q5_0.extend_from_slice(&scale);
        q5_0.extend_from_slice(&1u32.to_le_bytes()); // high bit for element 0
        q5_0.push(1 | (15 << 4));
        q5_0.extend(std::iter::repeat(0).take(15));
        let result = dequantize_q5_0(&q5_0, &[32]).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01); // (17 - 16) * 1
        assert!((result[16] + 1.0).abs() < 0.01); // (15 - 16) * 1

        let mut q5_1 = Vec::new();
        q5_1.extend_from_slice(&scale);
        q5_1.extend_from_slice(&min);
        q5_1.extend_from_slice(&1u32.to_le_bytes());
        q5_1.push(1 | (15 << 4));
        q5_1.extend(std::iter::repeat(0).take(15));
        let result = dequantize_q5_1(&q5_1, &[32]).unwrap();
        assert!((result[0] - 17.25).abs() < 0.01);
        assert!((result[16] - 15.25).abs() < 0.01);
    }

    #[test]
    fn test_q8_0_block_structure() {
        // Create a Q8_0 block with scale=0.5 and values
        let scale = half::f16::from_f32(0.5).to_le_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&scale);

        // 32 i8 values
        for i in 0..32i8 {
            data.push(i as u8);
        }

        let shape = [32];
        let result = dequantize_q8_0(&data, &shape).unwrap();

        assert_eq!(result.len(), 32);
        // Verify: value = quant * 0.5
        assert!((result[0] - 0.0).abs() < 0.01); // 0 * 0.5
        assert!((result[2] - 1.0).abs() < 0.01); // 2 * 0.5
        assert!((result[4] - 2.0).abs() < 0.01); // 4 * 0.5
    }

    #[test]
    fn test_q8_1_block_structure() {
        let scale = half::f16::from_f32(0.25).to_le_bytes();
        let sum = half::f16::from_f32(0.0).to_le_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&scale);
        data.extend_from_slice(&sum);
        for i in 0..32i8 {
            data.push(i as u8);
        }

        let result = dequantize_q8_1(&data, &[32]).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[4] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mxfp4_block_structure() {
        let mut data = Vec::new();
        data.push(128); // d = 1.0
        data.push(1 | (9 << 4)); // +1, -1
        data.extend(std::iter::repeat(0).take(15));

        let result = dequantize_mxfp4(&data, &[32]).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[16] + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_nvfp4_block_structure() {
        let mut data = vec![64, 0, 0, 0]; // first sub-block d = 1.0
        data.push(5 | (13 << 4)); // +6, -6
        data.extend(std::iter::repeat(0).take(31));

        let result = dequantize_nvfp4(&data, &[64]).unwrap();
        assert!((result[0] - 6.0).abs() < 0.01);
        assert!((result[8] + 6.0).abs() < 0.01);
    }

    #[test]
    fn test_tq2_0_block_structure() {
        let mut data = vec![0u8; 66];
        data[0] = 0 | (1 << 2) | (2 << 4) | (0 << 6);
        data[64..66].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());

        let result = dequantize_tq2_0(&data, &[256]).unwrap();
        assert!((result[0] + 2.0).abs() < 0.01);
        assert!(result[32].abs() < 0.01);
        assert!((result[64] - 2.0).abs() < 0.01);
        assert!((result[96] + 2.0).abs() < 0.01);
    }

    #[test]
    fn test_is_supported() {
        // Basic types
        assert!(is_supported(GgmlType::F32));
        assert!(is_supported(GgmlType::F16));
        assert!(is_supported(GgmlType::Bf16));
        assert!(is_supported(GgmlType::Q4_0));
        assert!(is_supported(GgmlType::Q4_1));
        assert!(is_supported(GgmlType::Q5_0));
        assert!(is_supported(GgmlType::Q5_1));
        assert!(is_supported(GgmlType::Q8_0));
        assert!(is_supported(GgmlType::Q8_1));
        assert!(is_supported(GgmlType::Q1_0));
        assert!(is_supported(GgmlType::Tq1_0));
        assert!(is_supported(GgmlType::Tq2_0));
        assert!(is_supported(GgmlType::Mxfp4));
        assert!(is_supported(GgmlType::Nvfp4));
        assert!(is_supported(GgmlType::I8));
        assert!(is_supported(GgmlType::I16));
        assert!(is_supported(GgmlType::I32));
        assert!(is_supported(GgmlType::I64));
        assert!(is_supported(GgmlType::F64));

        // K-quant types
        assert!(is_supported(GgmlType::Q2K));
        assert!(is_supported(GgmlType::Q3K));
        assert!(is_supported(GgmlType::Q4K));
        assert!(is_supported(GgmlType::Q5K));
        assert!(is_supported(GgmlType::Q6K));
        assert!(is_supported(GgmlType::Q8K));

        // IQ types
        assert!(is_supported(GgmlType::Iq4Nl));
        assert!(is_supported(GgmlType::Iq4Xs));
        assert!(is_supported(GgmlType::Iq2Xxs));
        assert!(is_supported(GgmlType::Iq2Xs));
        assert!(is_supported(GgmlType::Iq2S));
        assert!(is_supported(GgmlType::Iq3Xxs));
        assert!(is_supported(GgmlType::Iq3S));
        assert!(is_supported(GgmlType::Iq1S));
        assert!(is_supported(GgmlType::Iq1M));

        // Repacked kernel-specific layouts are not decoded as portable tensors.
        assert!(!is_supported(GgmlType::Q4_0_4_4));
        assert!(!is_supported(GgmlType::Iq4Nl_4_4));
    }

    #[test]
    fn test_invalid_size() {
        let data = vec![0u8; 10]; // Wrong size for 4 f32 values
        let shape = [4];
        let result = dequantize_f32(&data, &shape);
        assert!(matches!(result, Err(DequantError::InvalidSize { .. })));
    }

    #[test]
    fn test_invalid_shape() {
        let result = dequantize_f32(&[], &[-1]);
        assert!(matches!(result, Err(DequantError::InvalidShape { .. })));
    }
}
