//! Array extension utilities for MLX.

use pmetal_bridge::compat::{Array, Dtype, Exception};

type Result<T> = std::result::Result<T, Exception>;

/// Convert a raw dtype i32 to a `Dtype` enum value.
///
/// This maps MLX's internal dtype codes (from `dtype_raw()`) back to the `Dtype` enum.
/// Unknown codes fall back to `Dtype::Float32`.
pub fn dtype_from_raw(raw: i32) -> Dtype {
    match raw {
        0 => Dtype::Bool,
        1 => Dtype::Uint8,
        2 => Dtype::Uint16,
        3 => Dtype::Uint32,
        4 => Dtype::Uint64,
        5 => Dtype::Int8,
        6 => Dtype::Int16,
        7 => Dtype::Int32,
        8 => Dtype::Int64,
        9 => Dtype::Float16,
        10 => Dtype::Float32,
        11 => Dtype::Bfloat16,
        12 => Dtype::Complex64,
        _ => Dtype::Float32,
    }
}

/// Extension trait providing a typed `dtype()` method and additional utilities.
pub trait ArrayDtypeExt {
    /// Return the array dtype as a typed `Dtype` value.
    fn dtype(&self) -> Dtype;
}

impl ArrayDtypeExt for Array {
    fn dtype(&self) -> Dtype {
        dtype_from_raw(self.dtype_raw())
    }
}

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
            Dtype::Int64 | Dtype::Uint64 | Dtype::Complex64 => 8,
        }
    }

    fn nbytes(&self) -> usize {
        self.numel() * self.element_size()
    }
}

/// Create a zeros array with the given shape and dtype.
pub fn zeros(shape: &[i32], dtype: Dtype) -> Result<Array> {
    Ok(pmetal_bridge::compat::ops::zeros(shape, dtype))
}

/// Create a ones array with the given shape and dtype.
pub fn ones(shape: &[i32], dtype: Dtype) -> Result<Array> {
    Ok(pmetal_bridge::compat::ops::ones(shape, dtype))
}

/// Create a random normal array with the given shape and dtype.
pub fn randn(shape: &[i32], dtype: Dtype) -> Result<Array> {
    let a = pmetal_bridge::compat::random::normal(shape, Dtype::Float32);
    Ok(a.as_dtype(dtype.as_i32()))
}

/// Create a random uniform array with the given shape, range, and dtype.
pub fn rand(shape: &[i32], low: f32, high: f32, dtype: Dtype) -> Result<Array> {
    let base = pmetal_bridge::compat::random::uniform(shape, Dtype::Float32);
    // Scale from [0,1) to [low, high)
    let range = Array::from_f32(high - low);
    let offset = Array::from_f32(low);
    let a = base.multiply(&range).add(&offset);
    Ok(a.as_dtype(dtype.as_i32()))
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
    Ok(a.gather_mm(b, lhs_indices, rhs_indices, sorted_indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_uses_requested_dtype() {
        let array = zeros(&[2, 3], Dtype::Int32).unwrap();
        array.eval();

        assert_eq!(array.dtype(), Dtype::Int32);
        assert_eq!(array.shape(), &[2, 3]);
    }

    #[test]
    fn test_ones_uses_requested_dtype() {
        let array = ones(&[2, 2], Dtype::Int32).unwrap();
        array.eval();

        assert_eq!(array.dtype(), Dtype::Int32);
        assert_eq!(array.shape(), &[2, 2]);
    }

    #[test]
    fn test_randn_uses_requested_dtype() {
        let array = randn(&[4, 5], Dtype::Float16).unwrap();
        array.eval();

        assert_eq!(array.dtype(), Dtype::Float16);
        assert_eq!(array.shape(), &[4, 5]);
    }

    #[test]
    fn test_rand_uses_requested_dtype() {
        let array = rand(&[3, 2], -1.0, 1.0, Dtype::Float16).unwrap();
        array.eval();

        assert_eq!(array.dtype(), Dtype::Float16);
        assert_eq!(array.shape(), &[3, 2]);
    }
}
