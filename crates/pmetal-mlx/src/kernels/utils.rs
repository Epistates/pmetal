//! Shared utilities for Metal kernel integration.
//!
//! This module provides common functions for converting between MLX arrays
//! and Metal buffers, used by training_attention and differentiable_attention.

use crate::error::MlxError;
use half::f16;
use mlx_rs::{Array, Dtype};
use pmetal_metal::{
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
};

/// Result type for kernel utilities.
pub type Result<T> = std::result::Result<T, MlxError>;

/// Convert MLX Array to f16 MetalBuffer.
///
/// This copies data to ensure compatibility with Metal kernels that expect f16.
/// For zero-copy f32 operations, use `MlxMetalBridge::view_f32` instead.
///
/// # Arguments
/// * `ctx` - Metal context
/// * `array` - MLX array to convert (will be cast to f16 if needed)
///
/// # Returns
/// A MetalBuffer containing the array data in f16 format.
pub fn array_to_metal_buffer_f16(ctx: &MetalContext, array: &Array) -> Result<MetalBuffer<f16>> {
    // Ensure array is evaluated and in f16
    let array = if array.dtype() != Dtype::Float16 {
        array.as_dtype(Dtype::Float16)?
    } else {
        array.clone()
    };
    array.eval()?;

    // Get data as slice
    let data: &[f16] = array.as_slice();

    // Create Metal buffer
    MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)
        .map_err(|e| MlxError::Metal(e.to_string()))
}

/// Convert f16 MetalBuffer back to MLX Array.
///
/// # Arguments
/// * `buffer` - MetalBuffer to convert
/// * `shape` - Desired shape for the output array
///
/// # Returns
/// An MLX Array with the buffer data.
pub fn metal_buffer_to_array_f16(buffer: &MetalBuffer<f16>, shape: &[i32]) -> Result<Array> {
    let data = buffer
        .to_vec()
        .map_err(|e| MlxError::Metal(e.to_string()))?;

    Ok(Array::from_slice(&data, shape))
}

/// Convert MLX Array to f32 MetalBuffer.
///
/// For zero-copy operations, prefer `MlxMetalBridge::view_f32` when possible.
pub fn array_to_metal_buffer_f32(ctx: &MetalContext, array: &Array) -> Result<MetalBuffer<f32>> {
    // Ensure array is evaluated and in f32
    let array = if array.dtype() != Dtype::Float32 {
        array.as_dtype(Dtype::Float32)?
    } else {
        array.clone()
    };
    array.eval()?;

    let data: &[f32] = array.as_slice();
    MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)
        .map_err(|e| MlxError::Metal(e.to_string()))
}

/// Convert f32 MetalBuffer back to MLX Array.
pub fn metal_buffer_to_array_f32(buffer: &MetalBuffer<f32>, shape: &[i32]) -> Result<Array> {
    let data = buffer
        .to_vec()
        .map_err(|e| MlxError::Metal(e.to_string()))?;

    Ok(Array::from_slice(&data, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_to_metal_buffer_f16() {
        let ctx = MetalContext::new().unwrap();
        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);

        let buffer = array_to_metal_buffer_f16(&ctx, &array).unwrap();
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.size_bytes(), 8); // 4 * 2 bytes for f16
    }

    #[test]
    fn test_roundtrip_f16() {
        let ctx = MetalContext::new().unwrap();
        let original = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);

        let buffer = array_to_metal_buffer_f16(&ctx, &original).unwrap();
        let recovered = metal_buffer_to_array_f16(&buffer, &[2, 2]).unwrap();

        // Should match after f16 conversion (with potential precision loss)
        let orig_data: Vec<f32> = original.as_slice().to_vec();
        let recv_f32 = recovered.as_dtype(Dtype::Float32).unwrap();
        let recv_data: Vec<f32> = recv_f32.as_slice().to_vec();

        for (o, r) in orig_data.iter().zip(recv_data.iter()) {
            assert!((o - r).abs() < 0.01, "Values differ: {} vs {}", o, r);
        }
    }

    #[test]
    fn test_array_to_metal_buffer_f32() {
        let ctx = MetalContext::new().unwrap();
        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);

        let buffer = array_to_metal_buffer_f32(&ctx, &array).unwrap();
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.size_bytes(), 16); // 4 * 4 bytes for f32
    }
}
