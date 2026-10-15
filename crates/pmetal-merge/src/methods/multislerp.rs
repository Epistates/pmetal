//! Multi-SLERP - the weighted spherical mean of N models.
//!
//! Plain SLERP interpolates between two points on a hypersphere. Multi-SLERP
//! generalises it to N points with weights by taking their weighted
//! Fréchet (Karcher) mean on the unit sphere: the point `μ` minimising
//! `Σ w_i · d(μ, u_i)²` for the geodesic distance `d`. For two models with
//! weights `1 - t` and `t` that point is exactly the SLERP point at `t`.
//!
//! # Algorithm
//!
//! 1. Optionally subtract the base tensor, so the mean is taken over task
//!    vectors.
//! 2. Split each tensor into its norm `‖x_i‖` and unit direction `u_i`.
//! 3. Start from the normalised weighted Euclidean mean of the directions and
//!    refine it by the Riemannian gradient step
//!    `v = Σ w_i · Log_μ(u_i)`, `μ ← Exp_μ(v)`, with
//!    `Log_μ(u) = θ / sin θ · (u - cos θ · μ)` and `θ = arccos⟨μ, u⟩`,
//!    until `‖v‖` is negligible.
//! 4. Scale the result by the weighted mean of the norms, `Σ w_i ‖x_i‖`, and
//!    add the base back.
//!
//! Models with a zero task vector have no direction and take no part in the
//! direction mean (their weight is spread over the others), but they still
//! pull the norm down. When the directions cancel exactly (an antipodal pair
//! with equal weights) the mean is undefined: two models fall back to linear
//! interpolation, more are an error.

use super::MergeMethod;
use crate::{MergeError, MergeParameters, Result};
use pmetal_runtime::compat::Array;

/// Gradient steps of the Karcher iteration; it converges in a handful for
/// models that share a base, so this is a safety net, not a tuning knob.
const MAX_ITERATIONS: usize = 50;

/// The iteration stops once the Riemannian gradient is shorter than this
/// many radians.
const TOLERANCE: f32 = 1e-6;

/// `Σ_i weights[i] · terms[i]` for arrays; `weights.len() == terms.len() >= 1`.
fn weighted_sum(terms: &[Array], weights: &[f32]) -> Array {
    let mut sum = terms[0].multiply(&Array::from_f32(weights[0]));
    for (term, &w) in terms[1..].iter().zip(&weights[1..]) {
        sum = sum.add(&term.multiply(&Array::from_f32(w)));
    }
    sum
}

fn dot(a: &Array, b: &Array) -> f32 {
    a.multiply(b).sum_all().item_f32()
}

/// Compute the multi-SLERP of `tensors` with given `weights`.
///
/// # Arguments
/// * `tensors`           - Model tensors to interpolate.
/// * `weights`           - Per-model weights (will be normalized if `normalize_weights=true`).
/// * `base_tensor`       - Optional origin; subtracted before and added back after.
/// * `normalize_weights` - If true, normalize weights to sum to 1.0.
/// * `eps`               - Norm below which a vector counts as zero and a
///   weighted direction sum below which directions count as cancelling.
pub fn multislerp(
    tensors: &[Array],
    weights: &[f32],
    base_tensor: Option<&Array>,
    normalize_weights: bool,
    eps: f32,
) -> Result<Array> {
    if tensors.is_empty() {
        return Err(MergeError::NotEnoughModels {
            expected: 1,
            actual: 0,
        });
    }
    if weights.len() != tensors.len() {
        return Err(MergeError::InvalidConfig(format!(
            "Multi-SLERP: {} tensors but {} weights",
            tensors.len(),
            weights.len()
        )));
    }
    if tensors.len() == 1 {
        return Ok(tensors[0].clone());
    }

    let weights: Vec<f32> = if normalize_weights {
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return Err(MergeError::InvalidConfig(
                "Multi-SLERP: weight sum is zero".to_string(),
            ));
        }
        weights.iter().map(|w| w / total).collect()
    } else {
        weights.to_vec()
    };

    let shape = tensors[0].shape().to_vec();
    let vectors: Vec<Array> = tensors
        .iter()
        .map(|t| {
            let flat = t.reshape(&[-1]);
            match base_tensor {
                Some(base) => flat.subtract(&base.reshape(&[-1])),
                None => flat,
            }
        })
        .collect();
    let finish = |result: Array| -> Array {
        let result = match base_tensor {
            Some(base) => result.add(&base.reshape(&[-1])),
            None => result,
        };
        result.reshape(&shape)
    };

    let norms: Vec<f32> = vectors.iter().map(|v| dot(v, v).max(0.0).sqrt()).collect();
    let mean_norm: f32 = weights.iter().zip(&norms).map(|(w, n)| w * n).sum();

    // Directions exist only for non-zero vectors; their weights are
    // renormalised to sum to the same total.
    let directional: Vec<usize> = (0..vectors.len()).filter(|&i| norms[i] > eps).collect();
    if directional.is_empty() {
        return Ok(finish(vectors[0].multiply(&Array::from_f32(0.0))));
    }
    let direction_total: f32 = directional.iter().map(|&i| weights[i]).sum();
    if direction_total.abs() < f32::EPSILON {
        return Err(MergeError::InvalidConfig(
            "Multi-SLERP: the weights of the non-zero vectors sum to zero".to_string(),
        ));
    }
    let units: Vec<Array> = directional
        .iter()
        .map(|&i| vectors[i].multiply(&Array::from_f32(1.0 / norms[i])))
        .collect();
    let dir_weights: Vec<f32> = directional
        .iter()
        .map(|&i| weights[i] / direction_total)
        .collect();

    // Start from the normalised Euclidean mean of the directions.
    let start = weighted_sum(&units, &dir_weights);
    let start_norm = dot(&start, &start).max(0.0).sqrt();
    if start_norm < eps {
        if tensors.len() == 2 {
            // Antipodal directions with equal weights: no geodesic is
            // preferred, interpolate linearly.
            return Ok(finish(weighted_sum(&vectors, &weights)));
        }
        return Err(MergeError::InvalidConfig(
            "Multi-SLERP: weighted sum of unit tensors is zero (antipodal cancellation). \
             This happens when tensors with equal weights point in opposite directions. \
             Try using different weights."
                .to_string(),
        ));
    }
    let mut mu = start.multiply(&Array::from_f32(1.0 / start_norm));

    // Karcher iteration: step along the weighted mean of the log maps.
    for _ in 0..MAX_ITERATIONS {
        let mut gradient: Option<Array> = None;
        for (unit, &w) in units.iter().zip(&dir_weights) {
            let cos = dot(&mu, unit).clamp(-1.0, 1.0);
            let theta = cos.acos();
            if theta > std::f32::consts::PI - 1e-3 {
                // Antipodal to the current mean: Log is undefined there, and
                // the other directions decide the step.
                continue;
            }
            // Log_μ(u) = θ / sin θ · (u - cos θ · μ); θ / sin θ -> 1 as θ -> 0.
            let stretch = if theta < 1e-4 {
                1.0
            } else {
                theta / theta.sin()
            };
            let log = unit
                .subtract(&mu.multiply(&Array::from_f32(cos)))
                .multiply(&Array::from_f32(w * stretch));
            gradient = Some(match gradient {
                Some(g) => g.add(&log),
                None => log,
            });
        }
        let gradient = gradient.expect("at least one direction");
        let step = dot(&gradient, &gradient).max(0.0).sqrt();
        if step < TOLERANCE {
            break;
        }
        // Exp_μ(v) = cos‖v‖ · μ + sin‖v‖ · v / ‖v‖
        mu = mu
            .multiply(&Array::from_f32(step.cos()))
            .add(&gradient.multiply(&Array::from_f32(step.sin() / step)));
    }

    Ok(finish(mu.multiply(&Array::from_f32(mean_norm))))
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
    use crate::test_util::{arr, assert_close, values};

    #[test]
    fn test_multislerp_single_tensor() {
        let ms = MultiSlerpMerge::new();
        let t = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let params = vec![MergeParameters::default()];
        let global = MergeParameters::default();

        let mut result = ms
            .merge(std::slice::from_ref(&t), None, &params, &global)
            .unwrap();
        let r = result.to_f32_vec(3).unwrap();
        let mut t_clone = t.clone();
        let t_vals = t_clone.to_f32_vec(3).unwrap();
        for (rv, tv) in r.iter().zip(t_vals.iter()) {
            assert!((rv - tv).abs() < 1e-5);
        }
    }

    #[test]
    fn test_multislerp_two_tensors_equal_weights() {
        // Two orthogonal unit vectors; result should be at 45° with the same norm.
        let ms = MultiSlerpMerge::new();
        let a = Array::from_f32_slice(&[1.0_f32, 0.0], &[2]);
        let b = Array::from_f32_slice(&[0.0_f32, 1.0], &[2]);
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let mut result = ms.merge(&[a, b], None, &params, &global).unwrap();
        let r = result.to_f32_vec(2).unwrap();

        // Result should have equal x and y components (45°)
        assert!((r[0] - r[1]).abs() < 1e-4, "expected x≈y, got {:?}", r);
        // Length should ≈ 1.0 (avg_norm of two unit vectors = 1)
        let len = (r[0] * r[0] + r[1] * r[1]).sqrt();
        assert!((len - 1.0).abs() < 1e-4, "expected len≈1, got {}", len);
    }

    #[test]
    fn test_multislerp_preserves_shape() {
        let ms = MultiSlerpMerge::new();
        let a = Array::from_f32_slice(&[1.0_f32; 12], &[3, 4]);
        let b = Array::from_f32_slice(&[2.0_f32; 12], &[3, 4]);
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let result = ms.merge(&[a, b], None, &params, &global).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
    }

    #[test]
    fn test_multislerp_identical_tensors() {
        // Merging identical tensors should return the same tensor.
        let ms = MultiSlerpMerge::new();
        let t = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let params = vec![MergeParameters::default(); 3];
        let global = MergeParameters::default();

        let mut result = ms
            .merge(&[t.clone(), t.clone(), t.clone()], None, &params, &global)
            .unwrap();
        let r = result.to_f32_vec(3).unwrap();
        let mut t_clone = t.clone();
        let t_vals = t_clone.to_f32_vec(3).unwrap();
        for (rv, tv) in r.iter().zip(t_vals.iter()) {
            assert!((rv - tv).abs() < 1e-3, "expected {} got {}", tv, rv);
        }
    }

    #[test]
    fn test_multislerp_with_base_tensor() {
        // With a base tensor, the result should be the mean in delta-space, then add base.
        let ms = MultiSlerpMerge::new();
        let base = Array::from_f32_slice(&[1.0_f32, 0.0], &[2]);
        // Both deltas are the same → result delta = same, result = base + delta
        let a = Array::from_f32_slice(&[2.0_f32, 0.0], &[2]); // delta [1, 0]
        let b = Array::from_f32_slice(&[2.0_f32, 0.0], &[2]); // delta [1, 0]
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let mut result = ms.merge(&[a, b], Some(&base), &params, &global).unwrap();
        let r = result.to_f32_vec(2).unwrap();
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

        let a = Array::from_f32_slice(&[cos0, sin0], &[2]);
        let b = Array::from_f32_slice(&[cos120, sin120], &[2]);
        let c = Array::from_f32_slice(&[cos240, sin240], &[2]);
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
        let a = Array::from_f32_slice(&[1.0_f32, 0.0], &[2]);
        let b = Array::from_f32_slice(&[0.8_f32, 0.6], &[2]); // ~37°
        let c = Array::from_f32_slice(&[0.6_f32, 0.8], &[2]); // ~53°
        let params = vec![MergeParameters::default(); 3];
        let global = MergeParameters::default();

        let mut result = ms.merge(&[a, b, c], None, &params, &global).unwrap();
        let r = result.to_f32_vec(2).unwrap();
        // All inputs have positive x and y → result should too
        assert!(r[0] > 0.0, "expected x>0, got {:?}", r);
        assert!(r[1] > 0.0, "expected y>0, got {:?}", r);
        assert!(r.iter().all(|x| x.is_finite()), "non-finite: {:?}", r);
    }

    #[test]
    fn test_multislerp_weighted_bias() {
        // Heavy weight on tensor A → result closer to A.
        let ms = MultiSlerpMerge::new();
        let a = Array::from_f32_slice(&[1.0_f32, 0.0], &[2]);
        let b = Array::from_f32_slice(&[0.0_f32, 1.0], &[2]);
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

        let mut result = ms.merge(&[a, b], None, &params, &global).unwrap();
        let r = result.to_f32_vec(2).unwrap();
        // Should be strongly biased toward [1, 0]: r[0] >> r[1]
        assert!(r[0] > r[1], "expected x>y for 9:1 weighting, got {:?}", r);
    }

    #[test]
    fn two_models_are_the_slerp_point_not_the_normalized_lerp() {
        // Orthogonal unit vectors with weights 0.75 / 0.25 are the SLERP
        // point at t = 0.25: angle 22.5 degrees. A normalized weighted mean
        // would sit at atan(1/3) = 18.4 degrees instead.
        let a = arr(&[1.0, 0.0], &[2]);
        let b = arr(&[0.0, 1.0], &[2]);
        let out = multislerp(&[a, b], &[0.75, 0.25], None, true, 1e-8).unwrap();
        let angle = std::f32::consts::FRAC_PI_8;
        assert_close(&values(&out), &[angle.cos(), angle.sin()], 1e-5);
    }

    #[test]
    fn two_models_agree_with_slerp_on_arbitrary_vectors() {
        use super::super::SlerpMerge;
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) as f32 * 0.37).sin()).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| ((i * 5 + 1) as f32 * 0.91).cos() * 2.0)
            .collect();
        let (a, b) = (arr(&x, &[n]), arr(&y, &[n]));
        let t = 0.3;
        let want = SlerpMerge::slerp(&a, &b, t).unwrap();
        let got = multislerp(&[a, b], &[1.0 - t, t], None, true, 1e-8).unwrap();
        assert_close(&values(&got), &values(&want), 1e-4);
    }

    #[test]
    fn unit_axes_meet_at_the_diagonal_and_norms_are_averaged() {
        // Three orthonormal axes: by symmetry the weighted mean is
        // (1, 1, 1) / sqrt 3, with norm 1.
        let e = |i: usize| {
            let mut v = vec![0.0; 3];
            v[i] = 1.0;
            arr(&v, &[3])
        };
        let out = multislerp(&[e(0), e(1), e(2)], &[1.0; 3], None, true, 1e-8).unwrap();
        let c = 1.0 / 3.0_f32.sqrt();
        assert_close(&values(&out), &[c, c, c], 1e-5);

        // Norms 2 and 4 on orthogonal axes: direction at 45 degrees, norm 3.
        let out = multislerp(
            &[arr(&[2.0, 0.0], &[2]), arr(&[0.0, 4.0], &[2])],
            &[1.0, 1.0],
            None,
            true,
            1e-8,
        )
        .unwrap();
        let v = 3.0 * std::f32::consts::FRAC_1_SQRT_2;
        assert_close(&values(&out), &[v, v], 1e-5);
    }

    #[test]
    fn a_zero_task_vector_only_pulls_the_norm_down() {
        let base = arr(&[10.0, 10.0], &[2]);
        let out = multislerp(
            &[arr(&[12.0, 10.0], &[2]), arr(&[10.0, 10.0], &[2])],
            &[1.0, 1.0],
            Some(&base),
            true,
            1e-8,
        )
        .unwrap();
        // Direction (1, 0) from the only non-zero task vector, norm 0.5 * 2.
        assert_close(&values(&out), &[11.0, 10.0], 1e-5);
    }

    #[test]
    fn test_multislerp_not_requires_base() {
        assert!(!MultiSlerpMerge::new().requires_base_model());
    }
}
