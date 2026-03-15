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

#![allow(unsafe_code)]

use half::f16;
use mlx_rs::{Array, Dtype};
use pmetal_metal::{
    bridge::{MetalBufferView, metal_buffer_from_ptr},
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
    error::{MetalError, Result as MetalResult},
    kernels::dequant::DequantKernels,
};

/// High-level bridge for MLX ↔ Metal data transfer.
///
/// Provides both zero-copy views (when possible) and copy-based transfers
/// (when type conversion or dequantization is needed).
pub struct MlxMetalBridge;

impl MlxMetalBridge {
    /// Dequantize Q4_0 data directly into an MLX Array (Metal-accelerated).
    pub fn dequantize_q4_0(
        ctx: &MetalContext,
        data: &[u8],
        n_elements: usize,
        shape: &[i32],
    ) -> MetalResult<Array> {
        let dequant = DequantKernels::new(ctx)?;
        let in_buf = MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)?;
        let out_buf = MetalBuffer::<f32>::new(ctx, n_elements, BufferUsage::Shared)?;

        dequant.dequantize_q4_0(
            ctx,
            &in_buf.as_retained(),
            &out_buf.as_retained(),
            n_elements,
        )?;

        Self::buffer_into_array_f32(out_buf, shape)
    }

    /// Dequantize IQ4_XS data directly into an MLX Array (Metal-accelerated).
    pub fn dequantize_iq4_xs(
        ctx: &MetalContext,
        data: &[u8],
        n_elements: usize,
        shape: &[i32],
    ) -> MetalResult<Array> {
        let dequant = DequantKernels::new(ctx)?;
        let in_buf = MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)?;
        let out_buf = MetalBuffer::<f32>::new(ctx, n_elements, BufferUsage::Shared)?;

        dequant.dequantize_iq4_xs(
            ctx,
            &in_buf.as_retained(),
            &out_buf.as_retained(),
            n_elements,
        )?;

        Self::buffer_into_array_f32(out_buf, shape)
    }

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
                "view_f32 requires Float32 array, got {:?}",
                array.dtype()
            )));
        }

        // Ensure array is evaluated before accessing data pointer
        array
            .eval()
            .map_err(|e| MetalError::InvalidConfig(format!("Failed to eval array: {e}")))?;

        // Get pointer to array data via sys call
        let ptr = unsafe { mlx_sys::mlx_array_data_float32(array.as_ptr()) };
        if ptr.is_null() {
            return Err(MetalError::InvalidConfig(
                "MLX array data pointer is null".into(),
            ));
        }

        // Create a view (unsafe - we assume the array outlives the view)
        unsafe { metal_buffer_from_ptr(ctx, ptr as *mut f32, array.size()) }
    }

    /// Create a zero-copy buffer view from an f16 MLX array.
    pub fn view_f16(ctx: &MetalContext, array: &Array) -> MetalResult<MetalBufferView<f16>> {
        // Validate dtype
        if array.dtype() != Dtype::Float16 {
            return Err(MetalError::InvalidConfig(format!(
                "view_f16 requires Float16 array, got {:?}",
                array.dtype()
            )));
        }

        // Ensure array is evaluated before accessing data pointer
        array
            .eval()
            .map_err(|e| MetalError::InvalidConfig(format!("Failed to eval array: {e}")))?;

        // Get pointer to array data via sys call
        let ptr = unsafe { mlx_sys::mlx_array_data_float16(array.as_ptr()) };
        if ptr.is_null() {
            return Err(MetalError::InvalidConfig(
                "MLX array data pointer is null".into(),
            ));
        }

        // Create a view (unsafe - we assume the array outlives the view)
        unsafe { metal_buffer_from_ptr(ctx, ptr as *mut f16, array.size()) }
    }

    /// Copy MLX array data to a new f32 Metal buffer, converting dtype if needed.
    ///
    /// This method is safer than zero-copy as it owns its data, but slower
    /// due to the copy operation. Auto-converts non-f32 arrays.
    pub fn copy_as_f32(ctx: &MetalContext, array: &Array) -> MetalResult<MetalBuffer<f32>> {
        let converted = if array.dtype() != Dtype::Float32 {
            array
                .as_dtype(Dtype::Float32)
                .map_err(|e| MetalError::InvalidConfig(format!("dtype conversion failed: {e}")))?
        } else {
            array.clone()
        };
        let data = converted.as_slice::<f32>();
        MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)
    }

    /// Copy MLX array data to a new f16 Metal buffer, converting dtype if needed.
    pub fn copy_as_f16(ctx: &MetalContext, array: &Array) -> MetalResult<MetalBuffer<f16>> {
        let converted = if array.dtype() != Dtype::Float16 {
            array
                .as_dtype(Dtype::Float16)
                .map_err(|e| MetalError::InvalidConfig(format!("dtype conversion failed: {e}")))?
        } else {
            array.clone()
        };
        let data = converted.as_slice::<f16>();
        MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)
    }

    /// Convert an f32 Metal buffer back to an MLX Array (Zero-Copy).
    ///
    /// This uses `mlx_sys` to wrap the existing Metal buffer's memory in
    /// an MLX array with a custom deleter that keeps the buffer alive.
    pub fn buffer_into_array_f32(buffer: MetalBuffer<f32>, shape: &[i32]) -> MetalResult<Array> {
        let ptr = buffer.contents_ptr() as *mut std::ffi::c_void;

        // Wrap the buffer in a Box to pass as payload to the deleter
        let payload = Box::into_raw(Box::new(buffer)) as *mut std::ffi::c_void;

        // Custom deleter that re-claims the Box and drops it, freeing the buffer
        unsafe extern "C" fn deleter(payload: *mut std::ffi::c_void) {
            unsafe {
                let _ = Box::from_raw(payload as *mut MetalBuffer<f32>);
            }
        }

        let dim = shape.len() as i32;
        let c_array = unsafe {
            mlx_sys::mlx_array_new_data_managed_payload(
                ptr,
                shape.as_ptr(),
                dim,
                mlx_sys::mlx_dtype__MLX_FLOAT32,
                payload,
                Some(deleter),
            )
        };

        // Create Array wrapper using official constructor
        let array = unsafe { Array::from_ptr(c_array) };
        Ok(array)
    }

    /// Convert an f16 Metal buffer back to an MLX Array (Zero-Copy).
    pub fn buffer_into_array_f16(buffer: MetalBuffer<f16>, shape: &[i32]) -> MetalResult<Array> {
        let ptr = buffer.contents_ptr() as *mut std::ffi::c_void;
        let payload = Box::into_raw(Box::new(buffer)) as *mut std::ffi::c_void;

        unsafe extern "C" fn deleter(payload: *mut std::ffi::c_void) {
            unsafe {
                let _ = Box::from_raw(payload as *mut MetalBuffer<f16>);
            }
        }

        let dim = shape.len() as i32;
        let c_array = unsafe {
            mlx_sys::mlx_array_new_data_managed_payload(
                ptr,
                shape.as_ptr(),
                dim,
                mlx_sys::mlx_dtype__MLX_FLOAT16,
                payload,
                Some(deleter),
            )
        };

        let array = unsafe { Array::from_ptr(c_array) };
        Ok(array)
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
}
