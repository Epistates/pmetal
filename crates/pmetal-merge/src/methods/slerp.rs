//! SLERP merge method - Spherical Linear Interpolation.
//!
//! SLERP interpolates between two points on a hypersphere, maintaining
//! a constant "distance" from the origin during the interpolation.
//! This is geometrically superior to linear interpolation for normalized
//! vectors and can produce smoother blending for model weights.
//!
//! Formula:
//! ```text
//! slerp(A, B, t) = sin((1-t)θ)/sin(θ) * A + sin(tθ)/sin(θ) * B
//! where θ = arccos(A·B / |A||B|)
//! ```
//!
//! Best for:
//! - Smooth interpolation between two models
//! - When you want to preserve the "magnitude structure" of weights
//! - Blending models with similar architectures

use super::MergeMethod;
use crate::{MergeError, MergeParameters, Result};
use mlx_rs::Array;

/// SLERP merge implementation.
#[derive(Debug, Clone, Default)]
pub struct SlerpMerge;

impl SlerpMerge {
    /// Create a new SLERP merge method.
    pub fn new() -> Self {
        Self
    }

    /// Compute SLERP between two tensors.
    ///
    /// # Arguments
    /// * `a` - First tensor (t=0)
    /// * `b` - Second tensor (t=1)
    /// * `t` - Interpolation factor (0.0 to 1.0)
    pub fn slerp(a: &Array, b: &Array, t: f32) -> Result<Array> {
        // Handle edge cases
        if t <= 0.0 {
            return Ok(a.clone());
        }
        if t >= 1.0 {
            return Ok(b.clone());
        }

        // Flatten tensors for computation
        let original_shape = a.shape().to_vec();
        let a_flat = a.reshape(&[-1])?;
        let b_flat = b.reshape(&[-1])?;

        // Compute norms (sum over all dimensions)
        let a_norm = a_flat.multiply(&a_flat)?.sum(None)?.sqrt()?;
        let b_norm = b_flat.multiply(&b_flat)?.sum(None)?.sqrt()?;

        // Get scalar values
        let a_norm_val: f32 = a_norm.item();
        let b_norm_val: f32 = b_norm.item();

        // Handle degenerate cases
        if a_norm_val < 1e-8 {
            return Ok(b.multiply(&Array::from_f32(t))?);
        }
        if b_norm_val < 1e-8 {
            return Ok(a.multiply(&Array::from_f32(1.0 - t))?);
        }

        // Normalize
        let a_unit = a_flat.divide(&a_norm)?;
        let b_unit = b_flat.divide(&b_norm)?;

        // Compute dot product (cosine of angle)
        let dot = a_unit.multiply(&b_unit)?.sum(None)?;
        let mut cos_theta: f32 = dot.item();

        // Clamp to valid range for arccos
        cos_theta = cos_theta.clamp(-1.0, 1.0);

        // If vectors are very close, use linear interpolation
        if cos_theta.abs() > 0.9999 {
            let result_flat = a_flat
                .multiply(&Array::from_f32(1.0 - t))?
                .add(&b_flat.multiply(&Array::from_f32(t))?)?;
            return Ok(result_flat.reshape(&original_shape)?);
        }

        // If vectors are nearly opposite, add small perturbation
        let (a_unit, b_unit, cos_theta) = if cos_theta < 0.0 {
            // Negate b to take the shorter path
            let b_neg = b_unit.negative()?;
            let new_dot = a_unit.multiply(&b_neg)?.sum(None)?;
            let _new_cos: f32 = new_dot.item();
            (a_unit, b_neg, -cos_theta)
        } else {
            (a_unit, b_unit, cos_theta)
        };

        // Compute SLERP
        let theta = cos_theta.acos();
        let sin_theta = theta.sin();

        if sin_theta.abs() < 1e-8 {
            // Fallback to linear interpolation
            let result_flat = a_flat
                .multiply(&Array::from_f32(1.0 - t))?
                .add(&b_flat.multiply(&Array::from_f32(t))?)?;
            return Ok(result_flat.reshape(&original_shape)?);
        }

        let s0 = ((1.0 - t) * theta).sin() / sin_theta;
        let s1 = (t * theta).sin() / sin_theta;

        // Interpolate unit vectors
        let result_unit = a_unit
            .multiply(&Array::from_f32(s0))?
            .add(&b_unit.multiply(&Array::from_f32(s1))?)?;

        // Interpolate norms linearly
        let result_norm = a_norm_val * (1.0 - t) + b_norm_val * t;

        // Scale result
        let result_flat = result_unit.multiply(&Array::from_f32(result_norm))?;
        Ok(result_flat.reshape(&original_shape)?)
    }
}

impl MergeMethod for SlerpMerge {
    fn name(&self) -> &'static str {
        "slerp"
    }

    fn description(&self) -> &'static str {
        "Spherical linear interpolation between two models"
    }

    fn requires_base_model(&self) -> bool {
        false
    }

    fn merge(
        &self,
        tensors: &[Array],
        _base_tensor: Option<&Array>,
        params: &[MergeParameters],
        global_params: &MergeParameters,
    ) -> Result<Array> {
        if tensors.len() != 2 {
            return Err(MergeError::InvalidConfig(format!(
                "SLERP requires exactly 2 models, got {}",
                tensors.len()
            )));
        }

        // Get t from first model's params or global
        let t = params
            .first()
            .and_then(|p| p.t)
            .or(global_params.t)
            .unwrap_or(0.5);

        Self::slerp(&tensors[0], &tensors[1], t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slerp_endpoints() {
        // t=0 should return first tensor
        let a = Array::from_slice(&[1.0_f32, 0.0, 0.0], &[3]);
        let b = Array::from_slice(&[0.0_f32, 1.0, 0.0], &[3]);

        let result = SlerpMerge::slerp(&a, &b, 0.0).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();
        assert!((result_slice[0] - 1.0).abs() < 1e-5);
        assert!((result_slice[1] - 0.0).abs() < 1e-5);

        // t=1 should return second tensor
        let result = SlerpMerge::slerp(&a, &b, 1.0).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();
        assert!((result_slice[0] - 0.0).abs() < 1e-5);
        assert!((result_slice[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_slerp_midpoint() {
        // Orthogonal unit vectors
        let a = Array::from_slice(&[1.0_f32, 0.0], &[2]);
        let b = Array::from_slice(&[0.0_f32, 1.0], &[2]);

        let result = SlerpMerge::slerp(&a, &b, 0.5).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Should be at 45 degrees: [sqrt(2)/2, sqrt(2)/2]
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!((result_slice[0] - expected).abs() < 1e-4);
        assert!((result_slice[1] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_slerp_parallel_vectors() {
        // Parallel vectors (same direction)
        let a = Array::from_slice(&[1.0_f32, 0.0], &[2]);
        let b = Array::from_slice(&[2.0_f32, 0.0], &[2]);

        let result = SlerpMerge::slerp(&a, &b, 0.5).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Should interpolate magnitude: (1 + 2) / 2 = 1.5
        assert!((result_slice[0] - 1.5).abs() < 1e-4);
        assert!((result_slice[1] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_slerp_preserves_shape() {
        let a = Array::from_slice(&[1.0_f32; 12], &[3, 4]);
        let b = Array::from_slice(&[2.0_f32; 12], &[3, 4]);

        let result = SlerpMerge::slerp(&a, &b, 0.5).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
    }
}
