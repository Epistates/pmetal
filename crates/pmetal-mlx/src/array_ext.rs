//! Array extension utilities for MLX.

use mlx_rs::error::{Exception, Result};
use mlx_rs::{Array, Dtype, Stream};

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

/// Matrix multiplication with gathered indices.
///
/// Performs `a @ b` where either `a` or `b` (or both) can be indexed using
/// provided indices. This is useful for MoE (Mixture of Experts) where different
/// expert weights need to be selected per token.
///
/// # Arguments
/// * `a` - First matrix operand
/// * `b` - Second matrix operand
/// * `lhs_indices` - Optional indices for selecting from `a` (batched by first dim)
/// * `rhs_indices` - Optional indices for selecting from `b` (batched by first dim)
/// * `sorted_indices` - If true, indices are pre-sorted for better memory access
///
/// # Returns
/// Result of gathered matrix multiplication
///
/// # Example
/// ```ignore
/// // For MoE: x @ expert_weights[expert_indices]
/// // x: [num_tokens, hidden]
/// // expert_weights: [num_experts, hidden, intermediate]
/// // expert_indices: [num_tokens, top_k]
/// let result = gather_mm(&x, &expert_weights, None, Some(&expert_indices), false)?;
/// ```
pub fn gather_mm(
    a: &Array,
    b: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    sorted_indices: bool,
) -> Result<Array> {
    let stream = Stream::default();
    gather_mm_device(a, b, lhs_indices, rhs_indices, sorted_indices, &stream)
}

/// Matrix multiplication with gathered indices (device-specific).
pub fn gather_mm_device(
    a: &Array,
    b: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    sorted_indices: bool,
    stream: &Stream,
) -> Result<Array> {
    unsafe {
        let mut result = mlx_sys::mlx_array { ctx: std::ptr::null_mut() };

        let lhs_ptr = lhs_indices
            .map(|arr| arr.as_ptr())
            .unwrap_or(mlx_sys::mlx_array { ctx: std::ptr::null_mut() });

        let rhs_ptr = rhs_indices
            .map(|arr| arr.as_ptr())
            .unwrap_or(mlx_sys::mlx_array { ctx: std::ptr::null_mut() });

        let status = mlx_sys::mlx_gather_mm(
            &mut result,
            a.as_ptr(),
            b.as_ptr(),
            lhs_ptr,
            rhs_ptr,
            sorted_indices,
            stream.as_ptr(),
        );

        if status != 0 {
            return Err(Exception::from("gather_mm failed"));
        }

        Ok(Array::from_ptr(result))
    }
}
