//! Array extension utilities for MLX.

use mlx_rs::{Array, Dtype};

/// Extension trait for MLX arrays with additional utilities.
pub trait ArrayExt {
    /// Get the total number of elements.
    fn numel(&self) -> usize;

    /// Check if the array is contiguous in memory.
    fn is_contiguous(&self) -> bool;

    /// Get the size of each element in bytes.
    fn element_size(&self) -> usize;

    /// Get total memory in bytes.
    fn nbytes(&self) -> usize;
}

impl ArrayExt for Array {
    fn numel(&self) -> usize {
        self.size()
    }

    fn is_contiguous(&self) -> bool {
        // MLX arrays are always contiguous in the current implementation
        true
    }

    fn element_size(&self) -> usize {
        match self.dtype() {
            Dtype::Bool | Dtype::Int8 | Dtype::Uint8 => 1,
            Dtype::Int16 | Dtype::Uint16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Int32 | Dtype::Uint32 | Dtype::Float32 => 4,
            Dtype::Int64 | Dtype::Uint64 | Dtype::Float64 | Dtype::Complex64 => 8,
        }
    }

    fn nbytes(&self) -> usize {
        self.numel() * self.element_size()
    }
}

/// Create a zeros array with the given shape and dtype.
pub fn zeros(shape: &[i32], dtype: Dtype) -> mlx_rs::error::Result<Array> {
    mlx_rs::ops::zeros::<f32>(shape).map(|a| a.as_dtype(dtype).unwrap())
}

/// Create a ones array with the given shape and dtype.
pub fn ones(shape: &[i32], dtype: Dtype) -> mlx_rs::error::Result<Array> {
    mlx_rs::ops::ones::<f32>(shape).map(|a| a.as_dtype(dtype).unwrap())
}

/// Create a random normal array with the given shape and dtype.
pub fn randn(shape: &[i32], dtype: Dtype) -> mlx_rs::error::Result<Array> {
    mlx_rs::random::normal::<f32>(shape, None, None, None).map(|a| a.as_dtype(dtype).unwrap())
}

/// Create a random uniform array with the given shape, range, and dtype.
pub fn rand(shape: &[i32], low: f32, high: f32, dtype: Dtype) -> mlx_rs::error::Result<Array> {
    mlx_rs::random::uniform::<_, f32>(low, high, shape, None).map(|a| a.as_dtype(dtype).unwrap())
}
