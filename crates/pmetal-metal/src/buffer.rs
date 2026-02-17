//! GPU buffer management with type safety.
//!
//! This module provides typed GPU buffers that work with Metal's unified memory,
//! enabling zero-copy interop with MLX and CPU code.

use bytemuck::{Pod, Zeroable};
use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::marker::PhantomData;
use std::mem;
use std::ptr::NonNull;
use std::slice;

use crate::context::MetalContext;
use crate::error::{MetalError, Result};

/// Trait for types that provide access to a Metal buffer.
pub trait AsMetalBuffer {
    /// Get the underlying Metal buffer.
    fn as_metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer>;

    /// Get the number of elements in the buffer.
    fn len(&self) -> usize;

    /// Check if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Buffer usage flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    /// Read-only from GPU, written by CPU.
    GpuReadOnly,
    /// Write-only from GPU, read by CPU.
    GpuWriteOnly,
    /// Read-write from both CPU and GPU (default for unified memory).
    Shared,
    /// GPU private storage (fastest, but no CPU access).
    Private,
}

impl BufferUsage {
    /// Convert to Metal resource options.
    fn to_metal_options(self) -> MTLResourceOptions {
        match self {
            BufferUsage::GpuReadOnly | BufferUsage::Shared | BufferUsage::GpuWriteOnly => {
                // Shared mode for unified memory - both CPU and GPU can access
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeTracked
            }
            BufferUsage::Private => {
                // Private mode - GPU only, fastest
                MTLResourceOptions::StorageModePrivate
                    | MTLResourceOptions::HazardTrackingModeTracked
            }
        }
    }
}

/// A typed GPU buffer.
///
/// This wrapper provides type-safe access to Metal buffers with unified memory support.
/// On Apple Silicon, the buffer can be accessed from both CPU and GPU without explicit
/// copies.
pub struct MetalBuffer<T: Pod + Zeroable> {
    /// The underlying Metal buffer.
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Number of elements in the buffer.
    len: usize,
    /// Buffer usage mode.
    usage: BufferUsage,
    /// Phantom data for type safety.
    _phantom: PhantomData<T>,
}

impl<T: Pod + Zeroable> AsMetalBuffer for MetalBuffer<T> {
    fn as_metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl<T: Pod + Zeroable> MetalBuffer<T> {
    /// Create a new buffer with uninitialized contents.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Metal context
    /// * `len` - Number of elements
    /// * `usage` - Buffer usage mode
    ///
    /// # Errors
    ///
    /// Returns an error if buffer creation fails (e.g., out of memory).
    pub fn new(ctx: &MetalContext, len: usize, usage: BufferUsage) -> Result<Self> {
        let size = len.checked_mul(mem::size_of::<T>()).ok_or_else(|| {
            MetalError::BufferCreation {
                size: usize::MAX,
                reason: format!(
                    "Buffer size overflow: {} elements * {} bytes/element",
                    len,
                    mem::size_of::<T>()
                ),
            }
        })?;
        let options = usage.to_metal_options();

        let buffer = ctx
            .device()
            .newBufferWithLength_options(size, options)
            .ok_or_else(|| MetalError::BufferCreation {
                size,
                reason: "Device returned null".to_string(),
            })?;

        Ok(Self {
            buffer,
            len,
            usage,
            _phantom: PhantomData,
        })
    }

    /// Create a new buffer initialized with zeros.
    pub fn zeros(ctx: &MetalContext, len: usize, usage: BufferUsage) -> Result<Self> {
        let buffer = Self::new(ctx, len, usage)?;

        if usage != BufferUsage::Private {
            // Zero the buffer contents
            let slice = buffer.as_mut_slice_unchecked();
            slice.fill(T::zeroed());
        }

        Ok(buffer)
    }

    /// Create a new buffer from existing data.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Metal context
    /// * `data` - Source data to copy
    /// * `usage` - Buffer usage mode
    pub fn from_slice(ctx: &MetalContext, data: &[T], usage: BufferUsage) -> Result<Self> {
        if usage == BufferUsage::Private {
            return Err(MetalError::InvalidConfig(
                "Cannot create Private buffer from slice (no CPU access)".to_string(),
            ));
        }

        let size = std::mem::size_of_val(data);
        let options = usage.to_metal_options();

        // Create buffer with data
        let data_ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void).ok_or_else(|| {
            MetalError::BufferCreation {
                size,
                reason: "Null data pointer".to_string(),
            }
        })?;

        // SAFETY:
        // 1. data_ptr is a NonNull pointer obtained from a valid slice, so it's non-null
        //    and properly aligned for T
        // 2. The size is computed from data.len() * size_of::<T>(), matching the slice's memory
        // 3. Metal's newBufferWithBytes_length_options copies the data into the buffer,
        //    so the source slice can be safely dropped after this call
        // 4. The Pod + Zeroable bounds on T ensure the data is safely transmutable
        let buffer = unsafe {
            ctx.device()
                .newBufferWithBytes_length_options(data_ptr, size, options)
        }
        .ok_or_else(|| MetalError::BufferCreation {
            size,
            reason: "Device returned null".to_string(),
        })?;

        Ok(Self {
            buffer,
            len: data.len(),
            usage,
            _phantom: PhantomData,
        })
    }

    /// Get the number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        // Safe: len was validated in new(), but use saturating_mul defensively
        self.len.saturating_mul(mem::size_of::<T>())
    }

    /// Get the buffer usage mode.
    #[inline]
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Get a reference to the underlying Metal buffer.
    #[inline]
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// Get a raw pointer to the buffer contents.
    ///
    /// Returns `None` for private buffers (no CPU access).
    pub fn as_ptr(&self) -> Option<*const T> {
        if self.usage == BufferUsage::Private {
            return None;
        }

        let contents = self.buffer.contents();
        Some(contents.as_ptr() as *const T)
    }

    /// Get a mutable raw pointer to the buffer contents.
    ///
    /// Returns `None` for private buffers (no CPU access).
    pub fn as_mut_ptr(&self) -> Option<*mut T> {
        if self.usage == BufferUsage::Private {
            return None;
        }

        let contents = self.buffer.contents();
        Some(contents.as_ptr() as *mut T)
    }

    /// Get the buffer contents as a slice.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is private (no CPU access).
    pub fn as_slice(&self) -> &[T] {
        assert!(
            self.usage != BufferUsage::Private,
            "Cannot get slice of private buffer"
        );
        self.as_slice_unchecked()
    }

    /// Get the buffer contents as a mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is private (no CPU access).
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(
            self.usage != BufferUsage::Private,
            "Cannot get mutable slice of private buffer"
        );
        self.as_mut_slice_unchecked()
    }

    /// Get the buffer contents as a slice without checking usage.
    ///
    /// # Safety Contract (not marked unsafe but internal-only)
    ///
    /// Caller must ensure:
    /// - The buffer is not private (no CPU access for private buffers)
    /// - The GPU is not concurrently writing to this data (data races)
    /// - The buffer has been properly initialized
    #[inline]
    fn as_slice_unchecked(&self) -> &[T] {
        // SAFETY:
        // 1. buffer.contents() returns a NonNull pointer to the buffer's memory
        // 2. The memory is valid for self.len elements of type T
        // 3. T: Pod + Zeroable ensures safe transmutation from raw bytes
        // 4. The buffer's StorageModeShared ensures CPU-accessible unified memory
        // 5. Caller is responsible for ensuring no concurrent GPU writes
        unsafe {
            let ptr = self.buffer.contents().as_ptr() as *const T;
            slice::from_raw_parts(ptr, self.len)
        }
    }

    /// Get the buffer contents as a mutable slice without checking usage.
    ///
    /// # Safety Contract (not marked unsafe but internal-only)
    ///
    /// Caller must ensure:
    /// - The buffer is not private (no CPU access for private buffers)
    /// - The GPU is not concurrently accessing this data (data races)
    /// - The buffer has been properly initialized
    #[inline]
    #[allow(clippy::mut_from_ref)] // Metal buffers have interior mutability via unified memory
    fn as_mut_slice_unchecked(&self) -> &mut [T] {
        // SAFETY:
        // 1. buffer.contents() returns a NonNull pointer to the buffer's memory
        // 2. The memory is valid for self.len elements of type T
        // 3. T: Pod + Zeroable ensures safe transmutation from raw bytes
        // 4. The buffer's StorageModeShared ensures CPU-accessible unified memory
        // 5. Caller is responsible for ensuring no concurrent GPU access
        // 6. The &self receiver is intentional - Metal buffers allow interior
        //    mutability through their unified memory model
        unsafe {
            let ptr = self.buffer.contents().as_ptr() as *mut T;
            slice::from_raw_parts_mut(ptr, self.len)
        }
    }

    /// Copy data from a slice into the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The buffer is private
    /// - The slice length doesn't match the buffer length
    pub fn copy_from_slice(&mut self, data: &[T]) -> Result<()> {
        if self.usage == BufferUsage::Private {
            return Err(MetalError::InvalidConfig(
                "Cannot copy to private buffer".to_string(),
            ));
        }

        if data.len() != self.len {
            return Err(MetalError::BufferSizeMismatch {
                expected: self.len,
                actual: data.len(),
            });
        }

        let dst = self.as_mut_slice_unchecked();
        dst.copy_from_slice(data);

        Ok(())
    }

    /// Copy data from the buffer to a slice.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The buffer is private
    /// - The slice length doesn't match the buffer length
    pub fn copy_to_slice(&self, data: &mut [T]) -> Result<()> {
        if self.usage == BufferUsage::Private {
            return Err(MetalError::InvalidConfig(
                "Cannot copy from private buffer".to_string(),
            ));
        }

        if data.len() != self.len {
            return Err(MetalError::BufferSizeMismatch {
                expected: self.len,
                actual: data.len(),
            });
        }

        let src = self.as_slice_unchecked();
        data.copy_from_slice(src);

        Ok(())
    }

    /// Convert to a Vec, consuming the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is private.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        if self.usage == BufferUsage::Private {
            return Err(MetalError::InvalidConfig(
                "Cannot convert private buffer to vec".to_string(),
            ));
        }

        Ok(self.as_slice_unchecked().to_vec())
    }
}

// SAFETY: MetalBuffer can be sent between threads
//
// Metal's buffer objects are reference-counted and thread-safe at the API level.
// The underlying MTLBuffer is an Objective-C object with its own reference counting.
// Apple's Metal documentation states that MTLBuffer objects can be safely accessed
// from multiple threads, provided proper synchronization is used for the data contents.
//
// Our implementation ensures:
// 1. The Retained<ProtocolObject<dyn MTLBuffer>> provides thread-safe reference counting
// 2. Shared storage mode uses unified memory accessible from any thread
// 3. PhantomData<T> ensures T's Send/Sync requirements are respected
// 4. Data access methods require proper synchronization (GPU work must complete
//    before CPU access)
unsafe impl<T: Pod + Zeroable> Send for MetalBuffer<T> {}

// SAFETY: MetalBuffer can be shared between threads via &reference
//
// The buffer's contents are only modified through:
// 1. as_mut_slice() which requires &mut self (exclusive access)
// 2. copy_from_slice() which requires &mut self (exclusive access)
// 3. GPU compute/render operations (external synchronization required)
//
// Read-only access via as_slice() is safe to share between threads as long as
// no concurrent writes are happening (standard Rust borrowing rules).
unsafe impl<T: Pod + Zeroable> Sync for MetalBuffer<T> {}

impl<T: Pod + Zeroable> std::fmt::Debug for MetalBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("len", &self.len)
            .field("size_bytes", &self.size_bytes())
            .field("usage", &self.usage)
            .field("type", &std::any::type_name::<T>())
            .finish()
    }
}

/// Type alias for half-precision float buffers.
pub type MetalBufferF16 = MetalBuffer<f16>;

/// Type alias for single-precision float buffers.
pub type MetalBufferF32 = MetalBuffer<f32>;

/// Type alias for 32-bit integer buffers.
pub type MetalBufferI32 = MetalBuffer<i32>;

/// Type alias for 32-bit unsigned integer buffers.
pub type MetalBufferU32 = MetalBuffer<u32>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let ctx = MetalContext::new().unwrap();

        let buffer: MetalBuffer<f32> = MetalBuffer::new(&ctx, 1024, BufferUsage::Shared).unwrap();
        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.size_bytes(), 1024 * 4);
    }

    #[test]
    fn test_buffer_from_slice() {
        let ctx = MetalContext::new().unwrap();

        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let buffer = MetalBuffer::from_slice(&ctx, &data, BufferUsage::Shared).unwrap();

        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.as_slice(), &data);
    }

    #[test]
    fn test_buffer_copy() {
        let ctx = MetalContext::new().unwrap();

        let mut buffer: MetalBuffer<f32> =
            MetalBuffer::zeros(&ctx, 1024, BufferUsage::Shared).unwrap();

        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        buffer.copy_from_slice(&data).unwrap();

        let mut output = vec![0.0f32; 1024];
        buffer.copy_to_slice(&mut output).unwrap();

        assert_eq!(output, data);
    }

    #[test]
    fn test_buffer_f16() {
        let ctx = MetalContext::new().unwrap();

        let data: Vec<f16> = (0..1024).map(|i| f16::from_f32(i as f32)).collect();
        let buffer = MetalBuffer::from_slice(&ctx, &data, BufferUsage::Shared).unwrap();

        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.size_bytes(), 1024 * 2);
    }
}
