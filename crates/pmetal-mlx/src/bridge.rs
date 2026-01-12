//! Zero-copy bridging between MLX arrays and Metal buffers.
//!
//! On Apple Silicon, MLX and Metal share unified memory. This module provides
//! high-level utilities to pass MLX array data directly to Metal kernels without
//! copying, providing significant performance improvements.
//!
//! # Zero-Copy vs Copy
//!
//! - **Zero-copy** (`view_*` methods): Creates a Metal buffer view that wraps the MLX
//!   array's memory. No data is copied. The MLX array must outlive the view.
//!
//! - **Copy** (`copy_*` methods): Creates a new Metal buffer and copies data into it.
//!   Required when type conversion is needed (e.g., f32 → f16) or when the source
//!   array may be modified/deallocated.
//!
//! # Example
//!
//! ```ignore
//! use pmetal_mlx::bridge::MlxMetalBridge;
//! use mlx_rs::Array;
//!
//! let array = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
//! array.eval()?;
//!
//! // Zero-copy view (preferred for f32)
//! let view = MlxMetalBridge::view_f32(&ctx, &array)?;
//!
//! // Use view with Metal kernel...
//! kernel.forward(&view)?;
//! ```

use pmetal_metal::{
    bridge::{metal_buffer_from_ptr, MetalBufferView},
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result as MetalResult},
};
use half::f16;
use mlx_rs::{Array, Dtype};

/// High-level bridge for MLX ↔ Metal data transfer.
///
/// Provides both zero-copy views (when possible) and copy-based transfers
/// (when type conversion is needed).
pub struct MlxMetalBridge;

impl MlxMetalBridge {
    /// Create a zero-copy buffer view from an f32 MLX array.
    ///
    /// This is the preferred method for f32 data as it avoids all copying.
    /// The returned view wraps the MLX array's memory directly.
    ///
    /// # Safety Requirements
    ///
    /// - The array must be evaluated before calling this method
    /// - The array must remain valid for the lifetime of the returned view
    /// - The array must not be modified while the view is in use
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The array is not f32 dtype
    /// - The array's data pointer is null
    /// - Metal buffer creation fails
    pub fn view_f32(ctx: &MetalContext, array: &Array) -> MetalResult<MetalBufferView<f32>> {
        // Validate dtype
        if array.dtype() != Dtype::Float32 {
            return Err(MetalError::InvalidConfig(format!(
                "Expected f32 array, got {:?}",
                array.dtype()
            )));
        }

        // Ensure array is evaluated
        array.eval().map_err(|e| MetalError::InvalidConfig(format!(
            "Failed to evaluate array: {}",
            e
        )))?;

        // Get raw data pointer using mlx-rs safe API
        // Using as_slice() is safe - it returns a slice backed by the array's data
        let slice = array.as_slice::<f32>();
        let ptr = slice.as_ptr() as *mut f32;

        // Create zero-copy Metal buffer view
        // SAFETY:
        // 1. ptr is from a valid slice (as_slice ensures array is evaluated)
        // 2. Array remains in scope - slice borrows from it
        // 3. Apple Silicon unified memory allows GPU access to CPU memory
        // 4. array.size() correctly represents the number of f32 elements
        // 5. The caller must ensure the array outlives the returned view
        unsafe { metal_buffer_from_ptr(ctx, ptr, array.size()) }
    }

    /// Create a zero-copy buffer view from an f16 MLX array.
    ///
    /// Similar to `view_f32` but for half-precision arrays.
    ///
    /// # Note
    ///
    /// If the input array is f32, consider using `copy_as_f16` instead,
    /// as type conversion requires copying anyway.
    pub fn view_f16(ctx: &MetalContext, array: &Array) -> MetalResult<MetalBufferView<f16>> {
        // Validate dtype
        if array.dtype() != Dtype::Float16 {
            return Err(MetalError::InvalidConfig(format!(
                "Expected f16 array, got {:?}. Use copy_as_f16() for type conversion.",
                array.dtype()
            )));
        }

        // Ensure array is evaluated
        array.eval().map_err(|e| MetalError::InvalidConfig(format!(
            "Failed to evaluate array: {}",
            e
        )))?;

        // Get raw data pointer using mlx-rs safe API
        // Using as_slice() is safe - it returns a slice backed by the array's data
        let slice = array.as_slice::<f16>();
        let ptr = slice.as_ptr() as *mut f16;

        // Create zero-copy Metal buffer view
        // SAFETY:
        // 1. ptr is from a valid slice (as_slice ensures array is evaluated)
        // 2. Array remains in scope - slice borrows from it
        // 3. Apple Silicon unified memory allows GPU access to CPU memory
        // 4. array.size() correctly represents the number of f16 elements
        // 5. The caller must ensure the array outlives the returned view
        unsafe { metal_buffer_from_ptr(ctx, ptr, array.size()) }
    }

    /// Copy an MLX array to a new Metal buffer, converting to f16.
    ///
    /// Use this when:
    /// - The source array is f32 but you need f16 for the kernel
    /// - The source array may be modified or deallocated
    /// - You need an owned buffer rather than a view
    pub fn copy_as_f16(ctx: &MetalContext, array: &Array) -> MetalResult<MetalBuffer<f16>> {
        // Convert to f16 if needed
        let array = if array.dtype() != Dtype::Float16 {
            array.as_dtype(Dtype::Float16).map_err(|e| {
                MetalError::InvalidConfig(format!("Failed to convert to f16: {}", e))
            })?
        } else {
            array.clone()
        };

        array.eval().map_err(|e| MetalError::InvalidConfig(format!(
            "Failed to evaluate array: {}",
            e
        )))?;

        let data: &[f16] = array.as_slice();
        MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)
    }

    /// Copy an MLX array to a new Metal buffer as f32.
    ///
    /// Use this when you need an owned buffer rather than a view,
    /// or when the source array may be modified.
    pub fn copy_as_f32(ctx: &MetalContext, array: &Array) -> MetalResult<MetalBuffer<f32>> {
        // Convert to f32 if needed
        let array = if array.dtype() != Dtype::Float32 {
            array.as_dtype(Dtype::Float32).map_err(|e| {
                MetalError::InvalidConfig(format!("Failed to convert to f32: {}", e))
            })?
        } else {
            array.clone()
        };

        array.eval().map_err(|e| MetalError::InvalidConfig(format!(
            "Failed to evaluate array: {}",
            e
        )))?;

        let data: &[f32] = array.as_slice();
        MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)
    }

    /// Convert a Metal buffer back to an MLX Array.
    ///
    /// This always copies data from the Metal buffer into a new MLX array.
    pub fn buffer_to_array_f32(buffer: &MetalBuffer<f32>, shape: &[i32]) -> MetalResult<Array> {
        let data = buffer.to_vec()?;
        Ok(Array::from_slice(&data, shape))
    }

    /// Convert an f16 Metal buffer back to an MLX Array.
    pub fn buffer_to_array_f16(buffer: &MetalBuffer<f16>, shape: &[i32]) -> MetalResult<Array> {
        let data = buffer.to_vec()?;
        Ok(Array::from_slice(&data, shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_f32() {
        let ctx = MetalContext::new().unwrap();
        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        array.eval().unwrap();

        let view = MlxMetalBridge::view_f32(&ctx, &array).unwrap();
        assert_eq!(view.len(), 4);
        assert_eq!(view.size_bytes(), 16);
    }

    #[test]
    fn test_view_f32_rejects_f16() {
        let ctx = MetalContext::new().unwrap();
        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4])
            .as_dtype(Dtype::Float16)
            .unwrap();
        array.eval().unwrap();

        let result = MlxMetalBridge::view_f32(&ctx, &array);
        assert!(result.is_err());
    }

    #[test]
    fn test_copy_as_f16() {
        let ctx = MetalContext::new().unwrap();
        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);

        let buffer = MlxMetalBridge::copy_as_f16(&ctx, &array).unwrap();
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.size_bytes(), 8); // f16 = 2 bytes
    }

    #[test]
    fn test_buffer_to_array_f32() {
        let ctx = MetalContext::new().unwrap();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buffer = MetalBuffer::from_slice(&ctx, &data, BufferUsage::Shared).unwrap();

        let array = MlxMetalBridge::buffer_to_array_f32(&buffer, &[2, 2]).unwrap();
        assert_eq!(array.shape(), &[2, 2]);
    }
}
