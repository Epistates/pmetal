//! Multi-SLERP – barycentric interpolation on a hypersphere for N models.
//!
//! Standard SLERP only works for two models.  Multi-SLERP extends it to an
//! arbitrary number of models by working in tangent space:
//!
//! 1. Stack and (optionally) subtract the base tensor from each model.
//! 2. Project each tensor to the unit hypersphere.
//! 3. Compute the **weighted Euclidean mean** of the unit vectors and normalize
//!    it — this becomes the tangent-space origin `mean`.
//! 4. For each unit vector, compute its **tangent vector** relative to `mean`:
//!    `v_i = unit_i − (unit_i · mean) * mean`
//! 5. Compute the weighted sum of tangent vectors: `T = Σ w_i * v_i`
//! 6. Map back to the sphere using the **exponential map**:
//!    `result = cos(‖T‖) * mean + sin(‖T‖) * T / ‖T‖`
//! 7. Scale by the weighted average of original norms.
//! 8. If a base was subtracted, add it back.
//!
//! Fallback: when `‖mean‖ < eps` (antipodal cancellation) and there are exactly
//! two models the method falls back to linear interpolation.
//!
//! Reference: <https://goddard.blog/posts/multislerp-wow-what-a-cool-idea>

use super::MergeMethod;
use crate::{MergeError, MergeParameters, Result};
use mlx_rs::Array;

// =============================================================================
// Core algorithm (CPU, f32 arithmetic)
// =============================================================================

/// Compute the multi-SLERP of `tensors` with given `weights`.
///
/// All heavy lifting is done on the CPU to avoid MLX graph-building overhead
/// for the many per-element scalar operations involved.
///
/// # Arguments
/// * `tensors`           – Model tensors to interpolate.
/// * `weights`           – Per-model weights (will be normalized if `normalize_weights=true`).
/// * `base_tensor`       – Optional origin; subtracted before and added back after.
/// * `normalize_weights` – If true, normalize weights to sum to 1.0.
/// * `eps`               – Numerical stability epsilon.
pub fn multislerp(
    tensors: &[Array],
    weights: &[f32],
    base_tensor: Option<&Array>,
    normalize_weights: bool,
    eps: f32,
) -> Result<Array> {
    assert_eq!(tensors.len(), weights.len());

    if tensors.is_empty() {
        return Err(MergeError::NotEnoughModels {
            expected: 1,
            actual: 0,
        });
    }

    if tensors.len() == 1 {
        return Ok(tensors[0].clone());
    }

    let original_shape = tensors[0].shape().to_vec();
    let n_params: usize = original_shape.iter().map(|&d| d as usize).product();

    // Optionally normalize weights
    let weights: Vec<f32> = if normalize_weights {
        let s: f32 = weights.iter().sum();
        if s <= 0.0 {
            return Err(MergeError::InvalidConfig(
                "Multi-SLERP: weight sum is zero".to_string(),
            ));
        }
        weights.iter().map(|w| w / s).collect()
    } else {
        weights.to_vec()
    };

    // Pull all tensors onto CPU as flat f32
    let flat_tensors: Vec<Vec<f32>> = tensors
        .iter()
        .map(|t| t.reshape(&[-1]).unwrap().as_slice().to_vec())
        .collect();

    let base_flat: Option<Vec<f32>> =
        base_tensor.map(|b| b.reshape(&[-1]).unwrap().as_slice::<f32>().to_vec());

    // Subtract base if provided
    let flat_vecs: Vec<Vec<f32>> = if let Some(ref bf) = base_flat {
        flat_tensors
            .iter()
            .map(|t| t.iter().zip(bf.iter()).map(|(a, b)| a - b).collect())
            .collect()
    } else {
        flat_tensors.clone()
    };

    // Compute per-tensor norms
    let norms: Vec<f32> = flat_vecs
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();

    // Compute unit vectors
    let unit_vecs: Vec<Vec<f32>> = flat_vecs
        .iter()
        .zip(norms.iter())
        .map(|(v, &n)| {
            let denom = n + eps;
            v.iter().map(|x| x / denom).collect()
        })
        .collect();

    // Weighted mean of unit vectors
    let mut mean = vec![0.0f32; n_params];
    for (unit, &w) in unit_vecs.iter().zip(weights.iter()) {
        for (m, &u) in mean.iter_mut().zip(unit.iter()) {
            *m += w * u;
        }
    }

    let mean_norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mean_norm < eps {
        // Antipodal cancellation fallback
        if tensors.len() == 2 {
            // Linear interpolation fallback
            let w0 = weights[0];
            let w1 = weights[1];
            let result: Vec<f32> = flat_vecs[0]
                .iter()
                .zip(flat_vecs[1].iter())
                .map(|(a, b)| w0 * a + w1 * b)
                .collect();

            let result_with_base: Vec<f32> = if let Some(ref bf) = base_flat {
                result.iter().zip(bf.iter()).map(|(r, b)| r + b).collect()
            } else {
                result
            };

            let arr = Array::from_slice(&result_with_base, &[n_params as i32]);
            return Ok(arr.reshape(&original_shape)?);
        }
        return Err(MergeError::InvalidConfig(
            "Multi-SLERP: weighted sum of unit tensors is zero (antipodal cancellation). \
             This happens when tensors with equal weights point in opposite directions. \
             Try using different weights."
                .to_string(),
        ));
    }

    // Normalize mean to unit length
    let mean_unit: Vec<f32> = mean.iter().map(|x| x / mean_norm).collect();

    // Project each unit vector into the tangent plane at mean_unit
    // tangent_i = unit_i − (unit_i · mean_unit) * mean_unit
    let tangent_vecs: Vec<Vec<f32>> = unit_vecs
        .iter()
        .map(|unit| {
            let dot: f32 = unit.iter().zip(mean_unit.iter()).map(|(u, m)| u * m).sum();
            unit.iter()
                .zip(mean_unit.iter())
                .map(|(u, m)| u - dot * m)
                .collect()
        })
        .collect();

    // Weighted sum of tangent vectors
    let mut tangent_result = vec![0.0f32; n_params];
    for (tv, &w) in tangent_vecs.iter().zip(weights.iter()) {
        for (tr, &t) in tangent_result.iter_mut().zip(tv.iter()) {
            *tr += w * t;
        }
    }

    // Exponential map: result = cos(‖T‖) * mean_unit + sin(‖T‖) * T / ‖T‖
    let tangent_norm: f32 = tangent_result.iter().map(|x| x * x).sum::<f32>().sqrt() + eps;
    let cos_t = tangent_norm.cos();
    let sin_t = tangent_norm.sin();

    let result_unit: Vec<f32> = mean_unit
        .iter()
        .zip(tangent_result.iter())
        .map(|(&m, &t)| cos_t * m + sin_t * t / tangent_norm)
        .collect();

    // Scale by weighted average of original norms
    let avg_norm: f32 = weights
        .iter()
        .zip(norms.iter())
        .map(|(w, n)| w * n)
        .sum::<f32>();

    let result_scaled: Vec<f32> = result_unit.iter().map(|x| x * avg_norm).collect();

    // Add base back if it was subtracted
    let final_result: Vec<f32> = if let Some(ref bf) = base_flat {
        result_scaled
            .iter()
            .zip(bf.iter())
            .map(|(r, b)| r + b)
            .collect()
    } else {
        result_scaled
    };

    let arr = Array::from_slice(&final_result, &[n_params as i32]);
    Ok(arr.reshape(&original_shape)?)
}

// =============================================================================
// MergeMethod impl
// =============================================================================

/// Multi-SLERP merge implementation.
#[derive(Debug, Clone)]
pub struct MultiSlerpMerge {
    /// Numerical stability epsilon (default 1e-8).
    eps: f32,
}

impl Default for MultiSlerpMerge {
    fn default() -> Self {
        Self { eps: 1e-8 }
    }
}

impl MultiSlerpMerge {
    /// Create a new Multi-SLERP merge method with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the numerical stability epsilon.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl MergeMethod for MultiSlerpMerge {
    fn name(&self) -> &'static str {
        "multislerp"
    }

    fn description(&self) -> &'static str {
        "Multi-model SLERP via barycentric tangent-space interpolation"
    }

    fn requires_base_model(&self) -> bool {
        false
    }

    fn merge(
        &self,
        tensors: &[Array],
        base_tensor: Option<&Array>,
        params: &[MergeParameters],
        global_params: &MergeParameters,
    ) -> Result<Array> {
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        let weights: Vec<f32> = params
            .iter()
            .map(|p| global_params.merge_with(p).weight())
            .collect();

        // normalize_weights follows the global normalize flag (default true)
        let normalize = global_params.normalize();

        multislerp(tensors, &weights, base_tensor, normalize, self.eps)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multislerp_single_tensor() {
        let ms = MultiSlerpMerge::new();
        let t = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let params = vec![MergeParameters::default()];
        let global = MergeParameters::default();

        let result = ms
            .merge(std::slice::from_ref(&t), None, &params, &global)
            .unwrap();
        let r: Vec<f32> = result.as_slice().to_vec();
        let t_vals: Vec<f32> = t.as_slice().to_vec();
        for (rv, tv) in r.iter().zip(t_vals.iter()) {
            assert!((rv - tv).abs() < 1e-5);
        }
    }

    #[test]
    fn test_multislerp_two_tensors_equal_weights() {
        // Two orthogonal unit vectors; result should be at 45° with the same norm.
        let ms = MultiSlerpMerge::new();
        let a = Array::from_slice(&[1.0_f32, 0.0], &[2]);
        let b = Array::from_slice(&[0.0_f32, 1.0], &[2]);
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let result = ms.merge(&[a, b], None, &params, &global).unwrap();
        let r: Vec<f32> = result.as_slice().to_vec();

        // Result should have equal x and y components (45°)
        assert!((r[0] - r[1]).abs() < 1e-4, "expected x≈y, got {:?}", r);
        // Length should ≈ 1.0 (avg_norm of two unit vectors = 1)
        let len = (r[0] * r[0] + r[1] * r[1]).sqrt();
        assert!((len - 1.0).abs() < 1e-4, "expected len≈1, got {}", len);
    }

    #[test]
    fn test_multislerp_preserves_shape() {
        let ms = MultiSlerpMerge::new();
        let a = Array::from_slice(&[1.0_f32; 12], &[3, 4]);
        let b = Array::from_slice(&[2.0_f32; 12], &[3, 4]);
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let result = ms.merge(&[a, b], None, &params, &global).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
    }

    #[test]
    fn test_multislerp_identical_tensors() {
        // Merging identical tensors should return the same tensor.
        let ms = MultiSlerpMerge::new();
        let t = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let params = vec![MergeParameters::default(); 3];
        let global = MergeParameters::default();

        let result = ms
            .merge(&[t.clone(), t.clone(), t.clone()], None, &params, &global)
            .unwrap();
        let r: Vec<f32> = result.as_slice().to_vec();
        let t_vals: Vec<f32> = t.as_slice().to_vec();
        for (rv, tv) in r.iter().zip(t_vals.iter()) {
            assert!((rv - tv).abs() < 1e-3, "expected {} got {}", tv, rv);
        }
    }

    #[test]
    fn test_multislerp_with_base_tensor() {
        // With a base tensor, the result should be the mean in delta-space, then add base.
        let ms = MultiSlerpMerge::new();
        let base = Array::from_slice(&[1.0_f32, 0.0], &[2]);
        // Both deltas are the same → result delta = same, result = base + delta
        let a = Array::from_slice(&[2.0_f32, 0.0], &[2]); // delta [1, 0]
        let b = Array::from_slice(&[2.0_f32, 0.0], &[2]); // delta [1, 0]
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let result = ms.merge(&[a, b], Some(&base), &params, &global).unwrap();
        let r: Vec<f32> = result.as_slice().to_vec();
        // result ≈ [2, 0]
        assert!((r[0] - 2.0).abs() < 1e-4, "r[0] = {}", r[0]);
        assert!(r[1].abs() < 1e-4, "r[1] = {}", r[1]);
    }

    #[test]
    fn test_multislerp_three_models_symmetric_cancellation() {
        // Three unit vectors at 0°, 120°, 240° with equal weights sum to zero
        // (antipodal cancellation).  The method should return an error in this case
        // since it cannot determine a meaningful mean direction.
        let ms = MultiSlerpMerge::new();
        let cos0 = 1.0_f32;
        let sin0 = 0.0_f32;
        let cos120 = -0.5_f32;
        let sin120 = 3.0_f32.sqrt() / 2.0;
        let cos240 = -0.5_f32;
        let sin240 = -(3.0_f32.sqrt() / 2.0);

        let a = Array::from_slice(&[cos0, sin0], &[2]);
        let b = Array::from_slice(&[cos120, sin120], &[2]);
        let c = Array::from_slice(&[cos240, sin240], &[2]);
        let params = vec![MergeParameters::default(); 3];
        let global = MergeParameters::default();

        // Equal-weighted vectors at 120° intervals cancel → should error
        let result = ms.merge(&[a, b, c], None, &params, &global);
        assert!(
            result.is_err(),
            "expected antipodal cancellation error for symmetric 3-vector case"
        );
    }

    #[test]
    fn test_multislerp_three_models_asymmetric() {
        // Three vectors that do NOT cancel: all in the positive x half-plane.
        let ms = MultiSlerpMerge::new();
        let a = Array::from_slice(&[1.0_f32, 0.0], &[2]);
        let b = Array::from_slice(&[0.8_f32, 0.6], &[2]); // ~37°
        let c = Array::from_slice(&[0.6_f32, 0.8], &[2]); // ~53°
        let params = vec![MergeParameters::default(); 3];
        let global = MergeParameters::default();

        let result = ms.merge(&[a, b, c], None, &params, &global).unwrap();
        let r: Vec<f32> = result.as_slice().to_vec();
        // All inputs have positive x and y → result should too
        assert!(r[0] > 0.0, "expected x>0, got {:?}", r);
        assert!(r[1] > 0.0, "expected y>0, got {:?}", r);
        assert!(r.iter().all(|x| x.is_finite()), "non-finite: {:?}", r);
    }

    #[test]
    fn test_multislerp_weighted_bias() {
        // Heavy weight on tensor A → result closer to A.
        let ms = MultiSlerpMerge::new();
        let a = Array::from_slice(&[1.0_f32, 0.0], &[2]);
        let b = Array::from_slice(&[0.0_f32, 1.0], &[2]);
        let params = vec![
            MergeParameters {
                weight: Some(crate::config::ParameterSetting::Scalar(9.0)),
                ..Default::default()
            },
            MergeParameters {
                weight: Some(crate::config::ParameterSetting::Scalar(1.0)),
                ..Default::default()
            },
        ];
        let global = MergeParameters::default();

        let result = ms.merge(&[a, b], None, &params, &global).unwrap();
        let r: Vec<f32> = result.as_slice().to_vec();
        // Should be strongly biased toward [1, 0]: r[0] >> r[1]
        assert!(r[0] > r[1], "expected x>y for 9:1 weighting, got {:?}", r);
    }

    #[test]
    fn test_multislerp_not_requires_base() {
        assert!(!MultiSlerpMerge::new().requires_base_model());
    }
}
