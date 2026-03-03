//! MLX Array ↔ numpy conversion bridge.
//!
//! Copy-based bridge — zero-copy adds lifetime complexity not worth the boundary cost.
//! Apple Silicon unified memory makes copies fast.
//!
//! Note: These functions are internal utilities for future use by model.rs and trainer.rs.
//! They are not directly exposed to Python since mlx_rs::Array is not a pyclass.

#![allow(dead_code)]

use mlx_rs::Array;

/// Convert an MLX Array to a Vec<f32>.
///
/// Float16 arrays are widened to f32. This is the internal helper used by
/// Python-facing code that then wraps the result in numpy.
pub fn mlx_to_f32_vec(array: &Array) -> Result<Vec<f32>, String> {
    // Force evaluation of lazy array
    array.eval().map_err(|e| format!("MLX eval failed: {e}"))?;

    match array.dtype() {
        mlx_rs::Dtype::Float32 => Ok(array.as_slice::<f32>().to_vec()),
        mlx_rs::Dtype::Int32 => Ok(array.as_slice::<i32>().iter().map(|&v| v as f32).collect()),
        _ => {
            // Convert to f32 first
            let f32_array = array
                .as_dtype(mlx_rs::Dtype::Float32)
                .map_err(|e| format!("Dtype conversion failed: {e}"))?;
            f32_array
                .eval()
                .map_err(|e| format!("MLX eval failed: {e}"))?;
            Ok(f32_array.as_slice::<f32>().to_vec())
        }
    }
}

/// Convert a f32 slice to an MLX Array with the given shape.
pub fn f32_slice_to_mlx(data: &[f32], shape: &[i32]) -> Array {
    Array::from_slice(data, shape)
}
