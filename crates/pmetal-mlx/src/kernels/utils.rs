//! Shared utilities for Metal kernel integration.
//!
//! This module provides common functions for converting between MLX arrays
//! and Metal buffers.

use crate::{array_ext::ArrayDtypeExt, bridge::MlxMetalBridge, error::MlxError};
use half::f16;
use pmetal_bridge::compat::{Array, Dtype};
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
    // Convert to f32, then cast each element to f16.
    // This avoids unsafe raw-pointer access; the copy is acceptable for Metal buffer creation.
    let mut f32_arr = array.as_dtype(Dtype::Float32.as_i32());
    f32_arr
        .try_eval()
        .map_err(|e| MlxError::Metal(format!("failed to evaluate f32 array: {e}")))?;
    let n = f32_arr.size();
    let f32_data = f32_arr
        .to_f32_vec(n)
        .ok_or_else(|| MlxError::Metal("failed to read array data as f32".to_string()))?;
    let f16_data: Vec<f16> = f32_data.iter().map(|&x| f16::from_f32(x)).collect();

    // Create Metal buffer
    MetalBuffer::from_slice(ctx, &f16_data, BufferUsage::Shared)
        .map_err(|e| MlxError::Metal(e.to_string()))
}

/// Convert f16 MetalBuffer back to MLX Array (Zero-Copy).
///
/// # Arguments
/// * `buffer` - MetalBuffer to consume
/// * `shape` - Desired shape for the output array
///
/// # Returns
/// An MLX Array referencing the buffer data directly.
pub fn metal_buffer_into_array_f16(buffer: MetalBuffer<f16>, shape: &[i32]) -> Result<Array> {
    MlxMetalBridge::buffer_into_array_f16(buffer, shape).map_err(|e| MlxError::Metal(e.to_string()))
}

/// Convert MLX Array to f32 MetalBuffer.
///
/// For zero-copy operations, prefer `MlxMetalBridge::view_f32` when possible.
pub fn array_to_metal_buffer_f32(ctx: &MetalContext, array: &Array) -> Result<MetalBuffer<f32>> {
    // Ensure array is evaluated and in f32
    let mut converted = if array.dtype() != Dtype::Float32 {
        array.as_dtype(Dtype::Float32.as_i32())
    } else {
        array.clone()
    };
    converted
        .try_eval()
        .map_err(|e| MlxError::Metal(format!("failed to evaluate f32 array: {e}")))?;

    let n = converted.size();
    let data = converted
        .to_f32_vec(n)
        .ok_or_else(|| MlxError::Metal("failed to read array data as f32".to_string()))?;
    MetalBuffer::from_slice(ctx, &data, BufferUsage::Shared)
        .map_err(|e| MlxError::Metal(e.to_string()))
}

/// Convert f32 MetalBuffer back to MLX Array (Zero-Copy).
pub fn metal_buffer_into_array_f32(buffer: MetalBuffer<f32>, shape: &[i32]) -> Result<Array> {
    MlxMetalBridge::buffer_into_array_f32(buffer, shape).map_err(|e| MlxError::Metal(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_to_metal_buffer_f16() {
        let ctx = MetalContext::new().unwrap();
        let array = Array::from_f32_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);

        let buffer = array_to_metal_buffer_f16(&ctx, &array).unwrap();
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.size_bytes(), 8); // 4 * 2 bytes for f16
    }

    #[test]
    fn test_array_to_metal_buffer_f32() {
        let ctx = MetalContext::new().unwrap();
        let array = Array::from_f32_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);

        let buffer = array_to_metal_buffer_f32(&ctx, &array).unwrap();
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.size_bytes(), 16); // 4 * 4 bytes for f32
    }
}
