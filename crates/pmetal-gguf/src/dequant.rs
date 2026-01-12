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

use crate::iq_quants;
use crate::k_quants;
use crate::GgmlType;

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
        GgmlType::Q8_0 => dequantize_q8_0(data, shape),
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
            | GgmlType::Q8_0
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
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    dtype.tensor_size(n_elements)
}

/// F32 - direct conversion (no dequantization needed).
fn dequantize_f32(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = n_elements * 4;

    if data.len() != expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(floats)
}

/// F16 - half-precision to f32.
fn dequantize_f16(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = n_elements * 2;

    if data.len() != expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    let floats: Vec<f32> = data
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();

    Ok(floats)
}

/// BF16 - brain float to f32.
fn dequantize_bf16(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = n_elements * 2;

    if data.len() != expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    let floats: Vec<f32> = data
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();

    Ok(floats)
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

    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let n_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let expected_size = n_blocks * BLOCK_BYTES;

    if data.len() != expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    let mut output = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];

        // Scale is f16 at start of block
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();

        // 32 4-bit values packed in 16 bytes
        for i in 0..16 {
            let byte = block[2 + i];
            // Low nibble first, then high nibble
            let low = (byte & 0x0F) as i8 - 8;
            let high = ((byte >> 4) & 0x0F) as i8 - 8;
            output.push(low as f32 * scale);
            output.push(high as f32 * scale);
        }
    }

    // Truncate to exact element count (last block may be partial)
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

    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let n_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let expected_size = n_blocks * BLOCK_BYTES;

    if data.len() != expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

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

// =============================================================================
// K-Quant Dequantization Wrappers (Q2K-Q8K)
// =============================================================================

/// Q2K dequantization (2-bit K-quant, 256 values per block).
fn dequantize_q2k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = k_quants::BlockQ2K::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(k_quants::dequantize_q2k_bytes(data, n_elements))
}

/// Q3K dequantization (3-bit K-quant, 256 values per block).
fn dequantize_q3k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = k_quants::BlockQ3K::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(k_quants::dequantize_q3k_bytes(data, n_elements))
}

/// Q4K dequantization (4-bit K-quant, 256 values per block).
fn dequantize_q4k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = k_quants::BlockQ4K::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(k_quants::dequantize_q4k_bytes(data, n_elements))
}

/// Q5K dequantization (5-bit K-quant, 256 values per block).
fn dequantize_q5k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = k_quants::BlockQ5K::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(k_quants::dequantize_q5k_bytes(data, n_elements))
}

/// Q6K dequantization (6-bit K-quant, 256 values per block).
fn dequantize_q6k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = k_quants::BlockQ6K::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(k_quants::dequantize_q6k_bytes(data, n_elements))
}

/// Q8K dequantization (8-bit K-quant, 256 values per block).
fn dequantize_q8k(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = k_quants::BlockQ8K::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(k_quants::dequantize_q8k_bytes(data, n_elements))
}

// =============================================================================
// IQ Dequantization Wrappers (Importance-weighted Quantization)
// =============================================================================

/// IQ4_NL dequantization (4-bit non-linear, 32 values per block).
fn dequantize_iq4nl(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = iq_quants::BlockIq4Nl::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(iq_quants::dequantize_iq4nl_bytes(data, n_elements))
}

/// IQ4_XS dequantization (4-bit non-linear with scales, 256 values per block).
fn dequantize_iq4xs(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = iq_quants::BlockIq4Xs::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(iq_quants::dequantize_iq4xs_bytes(data, n_elements))
}

/// IQ2_XXS dequantization (2-bit extra-extra-small, 256 values per block).
fn dequantize_iq2xxs(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = iq_quants::BlockIq2Xxs::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(iq_quants::dequantize_iq2xxs_bytes(data, n_elements))
}

/// IQ2_XS dequantization (2-bit extra-small, 256 values per block).
fn dequantize_iq2xs(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = iq_quants::BlockIq2Xs::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(iq_quants::dequantize_iq2xs_bytes(data, n_elements))
}

/// IQ2_S dequantization (2-bit small, 256 values per block).
fn dequantize_iq2s(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = iq_quants::BlockIq2S::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(iq_quants::dequantize_iq2s_bytes(data, n_elements))
}

/// IQ3_XXS dequantization (3-bit extra-extra-small, 256 values per block).
fn dequantize_iq3xxs(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = iq_quants::BlockIq3Xxs::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(iq_quants::dequantize_iq3xxs_bytes(data, n_elements))
}

/// IQ3_S dequantization (3-bit small, 256 values per block).
fn dequantize_iq3s(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = iq_quants::BlockIq3S::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(iq_quants::dequantize_iq3s_bytes(data, n_elements))
}

/// IQ1_S dequantization (1-bit small, 256 values per block).
fn dequantize_iq1s(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = iq_quants::BlockIq1S::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

    Ok(iq_quants::dequantize_iq1s_bytes(data, n_elements))
}

/// IQ1_M dequantization (1-bit medium, 256 values per block).
fn dequantize_iq1m(data: &[u8], shape: &[i32]) -> Result<Vec<f32>, DequantError> {
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let expected_size = iq_quants::BlockIq1M::byte_size(n_elements);

    if data.len() < expected_size {
        return Err(DequantError::InvalidSize {
            expected: expected_size,
            actual: data.len(),
        });
    }

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
    fn test_is_supported() {
        // Basic types
        assert!(is_supported(GgmlType::F32));
        assert!(is_supported(GgmlType::F16));
        assert!(is_supported(GgmlType::Bf16));
        assert!(is_supported(GgmlType::Q4_0));
        assert!(is_supported(GgmlType::Q8_0));

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

        // Unsupported types
        assert!(!is_supported(GgmlType::Q4_1));
    }

    #[test]
    fn test_invalid_size() {
        let data = vec![0u8; 10]; // Wrong size for 4 f32 values
        let shape = [4];
        let result = dequantize_f32(&data, &shape);
        assert!(matches!(result, Err(DequantError::InvalidSize { .. })));
    }
}
