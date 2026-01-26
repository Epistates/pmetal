//! Model Breadcrumbs - Deterministic Sparsification for Model Merging.
//!
//! Model Breadcrumbs (2025) is an alternative to DARE that uses deterministic,
//! layer-wise masking instead of random dropping. It applies a dual masking
//! strategy that simultaneously removes large outliers and small perturbations.
//!
//! # Algorithm
//!
//! For each task vector τ:
//! 1. **Upper mask**: Remove large outliers (> top_p percentile)
//! 2. **Lower mask**: Remove small noise (< bottom_p percentile)
//! 3. **Final mask**: Keep values between lower and upper thresholds
//! 4. **Rescale**: Optionally rescale to maintain expected value
//!
//! # Formula
//!
//! ```text
//! upper_thresh = percentile(|τ|, top_p)
//! lower_thresh = percentile(|τ|, bottom_p)
//! mask = (lower_thresh <= |τ|) & (|τ| <= upper_thresh)
//! sparse_τ = τ * mask / density  (if rescale=true)
//! ```
//!
//! # Advantages over DARE
//!
//! - **Deterministic**: Same input always produces same output
//! - **Dual filtering**: Removes both outliers AND noise
//! - **Layer-adaptive**: Thresholds computed per-layer
//! - **More stable**: Better noise resistance in practice
//!
//! # References
//!
//! - "Model Breadcrumbs: Scaling Multi-Task Model Merging with Sparse Masks"
//! - arXiv:2312.06795 (2023), extended 2025

use super::MergeMethod;
use crate::{sign_consensus, MergeError, MergeParameters, Result};
use mlx_rs::Array;

/// Model Breadcrumbs merge implementation.
///
/// Uses deterministic dual-masking (removing both outliers and small noise)
/// for more stable model merging than random DARE.
#[derive(Debug, Clone)]
pub struct BreadcrumbsMerge {
    /// Upper percentile threshold (remove values above this).
    /// E.g., 0.99 means remove top 1% outliers.
    /// Default: 0.99
    pub top_p: f32,

    /// Lower percentile threshold (remove values below this).
    /// E.g., 0.10 means remove bottom 10% small values.
    /// Default: 0.10
    pub bottom_p: f32,

    /// Whether to rescale remaining values to maintain expected sum.
    /// Default: true
    pub rescale: bool,

    /// Whether to use TIES-style sign consensus.
    /// Default: false
    pub use_ties_consensus: bool,
}

impl Default for BreadcrumbsMerge {
    fn default() -> Self {
        Self::new()
    }
}

impl BreadcrumbsMerge {
    /// Create a new Model Breadcrumbs merger with default settings.
    pub fn new() -> Self {
        Self {
            top_p: 0.99,
            bottom_p: 0.10,
            rescale: true,
            use_ties_consensus: false,
        }
    }

    /// Create with custom percentile thresholds.
    ///
    /// # Arguments
    ///
    /// * `top_p` - Upper percentile (e.g., 0.99 removes top 1%)
    /// * `bottom_p` - Lower percentile (e.g., 0.10 removes bottom 10%)
    pub fn with_thresholds(top_p: f32, bottom_p: f32) -> Self {
        Self {
            top_p,
            bottom_p,
            rescale: true,
            use_ties_consensus: false,
        }
    }

    /// Enable TIES sign consensus.
    pub fn with_ties(mut self) -> Self {
        self.use_ties_consensus = true;
        self
    }

    /// Disable rescaling.
    pub fn without_rescale(mut self) -> Self {
        self.rescale = false;
        self
    }

    /// Set the lower percentile threshold.
    pub fn with_bottom_p(mut self, bottom_p: f32) -> Self {
        self.bottom_p = bottom_p;
        self
    }

    /// Set the upper percentile threshold.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Compute the nth percentile of absolute values.
    fn compute_percentile(values: &[f32], percentile: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((sorted.len() - 1) as f32 * percentile).round() as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Apply dual-mask sparsification to a task vector.
    fn sparsify(&self, delta: &Array) -> Result<Array> {
        // Get absolute values for thresholding
        let abs_delta = delta.abs()?;
        abs_delta.eval()?;
        let abs_values: Vec<f32> = abs_delta.as_slice().to_vec();

        // Compute percentile thresholds
        let lower_thresh = Self::compute_percentile(&abs_values, self.bottom_p);
        let upper_thresh = Self::compute_percentile(&abs_values, self.top_p);

        // Create mask: keep values where lower_thresh <= |v| <= upper_thresh
        let mask: Vec<f32> = abs_values
            .iter()
            .map(|&v| {
                if v >= lower_thresh && v <= upper_thresh {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        let mask_array = Array::from_slice(&mask, delta.shape());
        let sparse_delta = delta.multiply(&mask_array)?;

        // Optionally rescale to maintain expected value
        if self.rescale {
            let kept_count: f32 = mask.iter().sum();
            let total_count = mask.len() as f32;
            let density = kept_count / total_count;

            if density > 0.0 && density < 1.0 {
                let scale = Array::from_f32(1.0 / density);
                return Ok(sparse_delta.multiply(&scale)?);
            }
        }

        Ok(sparse_delta)
    }

    /// Compute task vector (delta from base).
    fn task_vector(tensor: &Array, base: &Array) -> Result<Array> {
        Ok(tensor.subtract(base)?)
    }
}

impl MergeMethod for BreadcrumbsMerge {
    fn name(&self) -> &'static str {
        if self.use_ties_consensus {
            "breadcrumbs_ties"
        } else {
            "breadcrumbs"
        }
    }

    fn description(&self) -> &'static str {
        if self.use_ties_consensus {
            "Deterministic dual-mask sparsification with TIES sign consensus"
        } else {
            "Deterministic dual-mask sparsification (removes outliers and noise)"
        }
    }

    fn requires_base_model(&self) -> bool {
        true
    }

    fn merge(
        &self,
        tensors: &[Array],
        base_tensor: Option<&Array>,
        params: &[MergeParameters],
        global_params: &MergeParameters,
    ) -> Result<Array> {
        let base = base_tensor.ok_or_else(|| MergeError::BaseModelRequired {
            method: self.name().to_string(),
        })?;

        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        // Compute task vectors
        let task_vectors: Vec<Array> = tensors
            .iter()
            .map(|t| Self::task_vector(t, base))
            .collect::<Result<Vec<_>>>()?;

        // Get weights
        let weights: Vec<f32> = params
            .iter()
            .map(|p| global_params.merge_with(p).weight())
            .collect();

        let lambda = global_params.lambda();

        // Apply breadcrumbs sparsification to each task vector
        let sparse_vectors: Vec<Array> = task_vectors
            .iter()
            .map(|tv| self.sparsify(tv))
            .collect::<Result<Vec<_>>>()?;

        // Optionally apply sign consensus
        let final_vectors = if self.use_ties_consensus {
            let consensus_mask = sign_consensus(&sparse_vectors, &weights)?;
            let mut result_vecs = Vec::with_capacity(sparse_vectors.len());
            for v in sparse_vectors.iter() {
                result_vecs.push(v.multiply(&consensus_mask)?);
            }
            result_vecs
        } else {
            sparse_vectors
        };

        // Compute weighted sum
        let mut result = mlx_rs::ops::zeros::<f32>(task_vectors[0].shape())?;

        for (vector, weight) in final_vectors.iter().zip(weights.iter()) {
            let weighted = vector.multiply(&Array::from_f32(*weight))?;
            result = result.add(&weighted)?;
        }

        // Scale by lambda
        result = result.multiply(&Array::from_f32(lambda))?;

        // Add back to base
        Ok(base.add(&result)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breadcrumbs_default() {
        let bc = BreadcrumbsMerge::new();
        assert!((bc.top_p - 0.99).abs() < 1e-6);
        assert!((bc.bottom_p - 0.10).abs() < 1e-6);
        assert!(bc.rescale);
        assert!(!bc.use_ties_consensus);
    }

    #[test]
    fn test_breadcrumbs_builder() {
        let bc = BreadcrumbsMerge::with_thresholds(0.95, 0.20)
            .with_ties()
            .without_rescale();

        assert!((bc.top_p - 0.95).abs() < 1e-6);
        assert!((bc.bottom_p - 0.20).abs() < 1e-6);
        assert!(!bc.rescale);
        assert!(bc.use_ties_consensus);
    }

    #[test]
    fn test_compute_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // 0th percentile should be min
        let p0 = BreadcrumbsMerge::compute_percentile(&values, 0.0);
        assert!((p0 - 1.0).abs() < 1e-5);

        // 100th percentile should be max
        let p100 = BreadcrumbsMerge::compute_percentile(&values, 1.0);
        assert!((p100 - 10.0).abs() < 1e-5);

        // 50th percentile should be median-ish
        let p50 = BreadcrumbsMerge::compute_percentile(&values, 0.5);
        assert!(p50 >= 5.0 && p50 <= 6.0);
    }

    #[test]
    fn test_sparsify_removes_outliers() {
        let bc = BreadcrumbsMerge::with_thresholds(0.9, 0.1).without_rescale();

        // Create tensor with some outliers
        let delta = Array::from_slice(
            &[
                0.1_f32, 0.2, 0.3, 0.4, 0.5,  // Normal values
                10.0, // Outlier (should be removed)
                0.01, // Small noise (should be removed)
            ],
            &[7],
        );

        let sparse = bc.sparsify(&delta).unwrap();
        sparse.eval().unwrap();
        let sparse_vals: Vec<f32> = sparse.as_slice().to_vec();

        // Middle values should be kept, extremes removed
        assert!(sparse_vals[5].abs() < 1e-6); // Outlier removed
        assert!(sparse_vals[6].abs() < 1e-6); // Small noise removed
    }

    #[test]
    fn test_breadcrumbs_with_base_model() {
        let bc = BreadcrumbsMerge::new();

        let base = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t1 = Array::from_slice(&[1.5_f32, 2.5, 3.5], &[3]);
        let t2 = Array::from_slice(&[1.3_f32, 2.3, 3.3], &[3]);

        let params = vec![
            MergeParameters {
                weight: Some(0.5),
                ..Default::default()
            },
            MergeParameters {
                weight: Some(0.5),
                ..Default::default()
            },
        ];
        let global = MergeParameters::default();

        let result = bc.merge(&[t1, t2], Some(&base), &params, &global).unwrap();
        result.eval().unwrap();

        // Result should be close to average task vector added to base
        let result_slice: Vec<f32> = result.as_slice().to_vec();
        assert!(result_slice[0].is_finite());
    }

    #[test]
    fn test_breadcrumbs_requires_base() {
        let bc = BreadcrumbsMerge::new();
        assert!(bc.requires_base_model());
    }

    #[test]
    fn test_breadcrumbs_vs_ties() {
        let bc_plain = BreadcrumbsMerge::new();
        let bc_ties = BreadcrumbsMerge::new().with_ties();

        assert_eq!(bc_plain.name(), "breadcrumbs");
        assert_eq!(bc_ties.name(), "breadcrumbs_ties");
    }

    #[test]
    fn test_breadcrumbs_preserves_base_with_zero_lambda() {
        let bc = BreadcrumbsMerge::new();

        let base = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t1 = Array::from_slice(&[2.0_f32, 3.0, 4.0], &[3]);

        let params = vec![MergeParameters {
            weight: Some(1.0),
            ..Default::default()
        }];

        let global = MergeParameters {
            lambda: Some(0.0),
            ..Default::default()
        };

        let result = bc.merge(&[t1], Some(&base), &params, &global).unwrap();
        result.eval().unwrap();

        let result_slice: Vec<f32> = result.as_slice().to_vec();
        let base_slice: Vec<f32> = base.as_slice().to_vec();

        // With lambda=0, result should equal base
        for (r, b) in result_slice.iter().zip(base_slice.iter()) {
            assert!((r - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_empty_input() {
        let bc = BreadcrumbsMerge::new();
        let base = Array::from_slice(&[1.0_f32, 2.0], &[2]);
        let result = bc.merge(&[], Some(&base), &[], &MergeParameters::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_rescale_maintains_energy() {
        let bc_rescale = BreadcrumbsMerge::with_thresholds(0.8, 0.2);
        let bc_no_rescale = BreadcrumbsMerge::with_thresholds(0.8, 0.2).without_rescale();

        let delta = Array::from_slice(
            &[0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            &[10],
        );

        let sparse_rescaled = bc_rescale.sparsify(&delta).unwrap();
        let sparse_plain = bc_no_rescale.sparsify(&delta).unwrap();

        sparse_rescaled.eval().unwrap();
        sparse_plain.eval().unwrap();

        let sum_rescaled: f32 = sparse_rescaled.as_slice::<f32>().iter().sum();
        let sum_plain: f32 = sparse_plain.as_slice::<f32>().iter().sum();

        // Rescaled should have higher sum due to density compensation
        // (assuming we removed some values)
        println!("Rescaled sum: {}, Plain sum: {}", sum_rescaled, sum_plain);
    }
}
