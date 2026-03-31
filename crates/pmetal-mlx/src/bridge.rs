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

#![allow(unsafe_code)]

use crate::array_ext::ArrayDtypeExt;
use half::f16;
use pmetal_bridge::compat::{Array, Dtype};
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
    pub fn view_f32<'a>(
        ctx: &'a MetalContext,
        array: &'a Array,
    ) -> MetalResult<MetalBufferView<'a, f32>> {
        // Validate dtype
        if array.dtype() != Dtype::Float32 {
            return Err(MetalError::InvalidConfig(format!(
                "view_f32 requires Float32 array, got {:?}",
                array.dtype()
            )));
        }

        // Ensure array is evaluated before accessing data pointer
        let evaled = array.clone();
        evaled.eval();

        // Get pointer to array data
        let ptr = evaled.data_ptr() as *mut f32;
        if ptr.is_null() {
            return Err(MetalError::InvalidConfig(
                "MLX array data pointer is null".into(),
            ));
        }

        // Create a view — the pointer remains valid while `evaled` is live,
        // but view lifetime is tied to `array` parameter.
        unsafe { metal_buffer_from_ptr(ctx, ptr, array.size()) }
    }

    /// Create a zero-copy buffer view from an f16 MLX array.
    pub fn view_f16<'a>(
        ctx: &'a MetalContext,
        array: &'a Array,
    ) -> MetalResult<MetalBufferView<'a, f16>> {
        // Validate dtype
        if array.dtype() != Dtype::Float16 {
            return Err(MetalError::InvalidConfig(format!(
                "view_f16 requires Float16 array, got {:?}",
                array.dtype()
            )));
        }

        let evaled = array.clone();
        evaled.eval();

        let ptr = evaled.data_ptr() as *mut f16;
        if ptr.is_null() {
            return Err(MetalError::InvalidConfig(
                "MLX array data pointer is null".into(),
            ));
        }

        unsafe { metal_buffer_from_ptr(ctx, ptr, array.size()) }
    }

    /// Create a zero-copy buffer view from a packed `u32` MLX array.
    pub fn view_u32<'a>(
        ctx: &'a MetalContext,
        array: &'a Array,
    ) -> MetalResult<MetalBufferView<'a, u32>> {
        if array.dtype() != Dtype::Uint32 {
            return Err(MetalError::InvalidConfig(format!(
                "view_u32 requires Uint32 array, got {:?}",
                array.dtype()
            )));
        }

        let evaled = array.clone();
        evaled.eval();

        let ptr = evaled.data_ptr() as *mut u32;
        if ptr.is_null() {
            return Err(MetalError::InvalidConfig(
                "MLX array data pointer is null".into(),
            ));
        }

        unsafe { metal_buffer_from_ptr(ctx, ptr as *mut u32, array.size()) }
    }

    /// Copy MLX array data to a new f32 Metal buffer, converting dtype if needed.
    ///
    /// This method is safer than zero-copy as it owns its data, but slower
    /// due to the copy operation. Auto-converts non-f32 arrays.
    pub fn copy_as_f32(ctx: &MetalContext, array: &Array) -> MetalResult<MetalBuffer<f32>> {
        let converted = if array.dtype() != Dtype::Float32 {
            array.as_dtype(Dtype::Float32.as_i32())
        } else {
            array.clone()
        };
        converted.eval();
        let n = converted.size();
        let ptr = converted.data_ptr() as *const f32;
        let data: &[f32] = unsafe { std::slice::from_raw_parts(ptr, n) };
        MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)
    }

    /// Copy MLX array data to a new f16 Metal buffer, converting dtype if needed.
    pub fn copy_as_f16(ctx: &MetalContext, array: &Array) -> MetalResult<MetalBuffer<f16>> {
        let converted = if array.dtype() != Dtype::Float16 {
            array.as_dtype(Dtype::Float16.as_i32())
        } else {
            array.clone()
        };
        converted.eval();
        let n = converted.size();
        let ptr = converted.data_ptr() as *const f16;
        let data: &[f16] = unsafe { std::slice::from_raw_parts(ptr, n) };
        MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)
    }

    /// Convert an f32 Metal buffer back to an MLX Array.
    ///
    /// Copies data from the Metal buffer into a new MLX array.
    pub fn buffer_into_array_f32(buffer: MetalBuffer<f32>, shape: &[i32]) -> MetalResult<Array> {
        let n = buffer.len();
        let ptr = buffer.contents_ptr() as *const f32;
        let data: &[f32] = unsafe { std::slice::from_raw_parts(ptr, n) };
        Ok(Array::from_f32_slice(data, shape))
    }

    /// Convert an f16 Metal buffer back to an MLX Array.
    ///
    /// Converts f16 data to f32, then creates an MLX array.
    pub fn buffer_into_array_f16(buffer: MetalBuffer<f16>, shape: &[i32]) -> MetalResult<Array> {
        let n = buffer.len();
        let ptr = buffer.contents_ptr() as *const f16;
        let data_f16: &[f16] = unsafe { std::slice::from_raw_parts(ptr, n) };
        // Convert f16 → f32 then create array and cast back to f16
        let data_f32: Vec<f32> = data_f16.iter().map(|x| x.to_f32()).collect();
        let arr_f32 = Array::from_f32_slice(&data_f32, shape);
        Ok(arr_f32.as_dtype(Dtype::Float16.as_i32()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_f32() {
        let ctx = MetalContext::new().unwrap();
        let array = Array::from_f32_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let mut evaled = array.clone();
        evaled.eval();

        let view = MlxMetalBridge::view_f32(&ctx, &array).unwrap();
        assert_eq!(view.len(), 4);
        assert_eq!(view.size_bytes(), 16);
    }

    #[test]
    fn test_view_f32_rejects_f16() {
        let ctx = MetalContext::new().unwrap();
        let array =
            Array::from_f32_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).as_dtype(Dtype::Float16.as_i32());

        let result = MlxMetalBridge::view_f32(&ctx, &array);
        assert!(result.is_err());
    }
}
