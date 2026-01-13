//! Bridge for zero-copy interop between MLX arrays and Metal buffers.
//!
//! On Apple Silicon, MLX and Metal share unified memory. This module provides
//! utilities to pass data between them without copying.

use bytemuck::{Pod, Zeroable};
use half::f16;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ptr::NonNull;

use crate::buffer::AsMetalBuffer;
use crate::context::MetalContext;
use crate::error::{MetalError, Result};

/// Create a Metal buffer from a raw pointer (zero-copy).
///
/// This creates a Metal buffer that wraps existing memory WITHOUT copying.
/// The memory must remain valid for the lifetime of the buffer.
///
/// # Safety
///
/// The caller must guarantee:
/// 1. `ptr` is valid and properly aligned for `T`
/// 2. `ptr` points to at least `len * size_of::<T>()` bytes of valid memory
/// 3. The memory remains valid and unchanged for the lifetime of the returned view
/// 4. The memory is accessible by the GPU (unified memory on Apple Silicon)
/// 5. No mutable references exist to the memory while the view is in use
/// 6. Proper synchronization is used between CPU and GPU access
///
/// # Arguments
///
/// * `ctx` - Metal context
/// * `ptr` - Pointer to the data (must be from unified memory, e.g., MLX arrays)
/// * `len` - Number of elements of type T
///
/// # Common Use Case
///
/// This function is designed for zero-copy bridging from MLX arrays to Metal:
/// ```ignore
/// let mlx_ptr = mlx_sys::mlx_array_data_float32(array.as_ptr());
/// let view = unsafe { metal_buffer_from_ptr(&ctx, mlx_ptr, array.size())? };
/// ```
pub unsafe fn metal_buffer_from_ptr<T: Pod + Zeroable>(
    ctx: &MetalContext,
    ptr: *mut T,
    len: usize,
) -> Result<MetalBufferView<T>> {
    let size = len * std::mem::size_of::<T>();

    let ptr_void =
        NonNull::new(ptr as *mut std::ffi::c_void).ok_or_else(|| MetalError::BufferCreation {
            size,
            reason: "Null pointer".to_string(),
        })?;

    // Create buffer without copy - wraps existing memory
    // StorageModeShared: Required for unified memory (CPU + GPU access)
    // HazardTrackingModeTracked: Metal tracks read/write hazards automatically
    let options =
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked;

    // SAFETY (internal):
    // 1. ptr_void is non-null (checked above)
    // 2. size is correctly computed from len and size_of::<T>()
    // 3. newBufferWithBytesNoCopy creates a view without copying data
    // 4. deallocator is None because we don't own the memory - the caller
    //    (typically an MLX Array) is responsible for deallocation
    // 5. The Pod + Zeroable bounds ensure T has no padding or invariants
    let buffer = ctx
        .device()
        .newBufferWithBytesNoCopy_length_options_deallocator(ptr_void, size, options, None)
        .ok_or_else(|| MetalError::BufferCreation {
            size,
            reason: "Failed to create buffer view".to_string(),
        })?;

    Ok(MetalBufferView {
        buffer,
        len,
        _phantom: std::marker::PhantomData,
    })
}

/// A view into existing memory as a Metal buffer.
///
/// Unlike `MetalBuffer`, this does not own its memory - it's a view into
/// memory owned elsewhere (e.g., an MLX array).
pub struct MetalBufferView<T: Pod + Zeroable> {
    buffer: objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Pod + Zeroable> AsMetalBuffer for MetalBufferView<T> {
    fn as_metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

impl<T: Pod + Zeroable> MetalBufferView<T> {
    /// Get the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Get a reference to the underlying Metal buffer.
    #[inline]
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

// SAFETY: MetalBufferView can be sent between threads
//
// MetalBufferView wraps an MTLBuffer which is a thread-safe Objective-C object.
// The view does not own the underlying memory - it just provides a Metal interface
// to memory owned elsewhere (typically an MLX array).
//
// Thread safety requirements:
// 1. The source memory (e.g., MLX array) must remain valid across all threads
// 2. The source memory must not be modified while any thread holds this view
// 3. GPU operations using this buffer must be properly synchronized
unsafe impl<T: Pod + Zeroable> Send for MetalBufferView<T> {}

// SAFETY: MetalBufferView can be shared between threads via &reference
//
// The view provides read-only access to the underlying Metal buffer.
// Multiple threads can safely share a reference since:
// 1. MTLBuffer is thread-safe for concurrent reads
// 2. The view's methods only provide read access to the buffer metadata
// 3. The underlying memory is immutable from this view's perspective
unsafe impl<T: Pod + Zeroable> Sync for MetalBufferView<T> {}

/// Type alias for f16 buffer views (common for attention).
pub type MetalBufferViewF16 = MetalBufferView<f16>;

/// Type alias for f32 buffer views.
pub type MetalBufferViewF32 = MetalBufferView<f32>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_view_from_ptr() {
        let ctx = MetalContext::new().unwrap();

        // Create some data
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];

        // Create a view (unsafe - we know the data is valid)
        let view = unsafe { metal_buffer_from_ptr(&ctx, data.as_mut_ptr(), data.len()) }.unwrap();

        assert_eq!(view.len(), 4);
        assert_eq!(view.size_bytes(), 16);
    }
}
