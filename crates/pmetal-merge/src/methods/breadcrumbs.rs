//! Model Breadcrumbs - sparse masks that drop both noise and outliers.
//!
//! Model Breadcrumbs (Davari and Belilovsky, "Model Breadcrumbs: Scaling
//! Multi-Task Model Merging with Sparse Masks", ECCV 2024, arXiv:2312.06795)
//! thins each fine-tuned task vector layer by layer, masking out two tails
//! of its magnitude distribution: the smallest entries, which are mostly
//! fine-tuning noise, and the largest ones, which are outliers that dominate
//! a sum and interfere across tasks. What remains is the middle band:
//!
//! ```text
//! mask_i = 1  where  rank-from-top(|τ_i|) is in (gamma·n, density·n]
//! θ = θ_base + λ · combine(mask_1 ⊙ τ_1, ..., mask_N ⊙ τ_N)
//! ```
//!
//! `density` (per model, default 1) is the fraction of entries kept before
//! the outliers go, and `gamma` (global, default 0.01) the fraction of the
//! largest entries removed, so a fraction `density - gamma` survives. The
//! masks are deterministic. Unlike DARE there is no rescaling: the paper's
//! sparse vectors are added as they are.
//!
//! `combine` is the weighted sum (divided by the total weight under
//! `normalize`, the default), or with TIES consensus the sign election and
//! disjoint merge.

use super::MergeMethod;
use crate::consensus::combine_task_vectors;
use crate::{MergeError, MergeParameters, Result, sparsify_breadcrumbs};
use pmetal_runtime::compat::Array;

/// Model Breadcrumbs merge implementation.
#[derive(Debug, Clone, Default)]
pub struct BreadcrumbsMerge {
    /// Whether to use TIES-style sign consensus.
    use_ties_consensus: bool,
}

impl BreadcrumbsMerge {
    /// Model Breadcrumbs with a weighted-sum combination.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable TIES sign consensus.
    pub fn with_ties(mut self) -> Self {
        self.use_ties_consensus = true;
        self
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
        if params.len() != tensors.len() {
            return Err(MergeError::InvalidConfig(format!(
                "{} tensors but {} per-model parameter sets",
                tensors.len(),
                params.len()
            )));
        }

        let gamma = global_params.gamma();
        let mut sparse = Vec::with_capacity(tensors.len());
        let mut weights = Vec::with_capacity(tensors.len());
        for (tensor, per_model) in tensors.iter().zip(params) {
            let effective = global_params.merge_with(per_model);
            let task_vector = tensor.subtract(base);
            sparse.push(sparsify_breadcrumbs(
                &task_vector,
                effective.density(),
                gamma,
            )?);
            weights.push(effective.weight());
        }

        let combined = combine_task_vectors(
            &sparse,
            &weights,
            self.use_ties_consensus,
            global_params.normalize(),
        )?;
        Ok(base.add(&combined.multiply(&Array::from_f32(global_params.lambda()))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{arr, assert_close, values, weight_density};

    fn global(gamma: f32, normalize: bool) -> MergeParameters {
        MergeParameters {
            gamma: Some(gamma),
            normalize: Some(normalize),
            ..Default::default()
        }
    }

    #[test]
    fn names_and_variants() {
        assert_eq!(BreadcrumbsMerge::new().name(), "breadcrumbs");
        assert_eq!(
            BreadcrumbsMerge::new().with_ties().name(),
            "breadcrumbs_ties"
        );
        assert!(BreadcrumbsMerge::new().requires_base_model());
    }

    #[test]
    fn masks_the_smallest_and_the_largest_entries_of_each_task_vector() {
        // Task vector magnitudes 1..=10 (signs mixed). density 0.8 drops the
        // two smallest (1, 2); gamma 0.1 drops the largest (10).
        let task = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0];
        let base = arr(&[100.0; 10], &[10]);
        let model: Vec<f32> = task.iter().map(|t| 100.0 + t).collect();
        let out = BreadcrumbsMerge::new()
            .merge(
                &[arr(&model, &[10])],
                Some(&base),
                &[weight_density(1.0, 0.8)],
                &global(0.1, true),
            )
            .unwrap();
        let want: Vec<f32> = [0.0, 0.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, 0.0]
            .iter()
            .map(|d| 100.0 + d)
            .collect();
        assert_close(&values(&out), &want, 1e-5);
    }

    #[test]
    fn survivors_are_not_rescaled() {
        // Dropping entries must not inflate the others (DARE's 1/density).
        let base = arr(&[0.0; 4], &[4]);
        let out = BreadcrumbsMerge::new()
            .merge(
                &[arr(&[1.0, 2.0, 3.0, 4.0], &[4])],
                Some(&base),
                &[weight_density(1.0, 0.5)],
                &global(0.0, true),
            )
            .unwrap();
        assert_close(&values(&out), &[0.0, 0.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn combines_models_by_weighted_sum_and_normalize() {
        // Full density, no outlier removal: two task vectors [2] and [6]
        // with weights 1 and 3.
        let base = arr(&[0.0], &[1]);
        let models = [arr(&[2.0], &[1]), arr(&[6.0], &[1])];
        let params = [weight_density(1.0, 1.0), weight_density(3.0, 1.0)];
        let run = |normalize| {
            values(
                &BreadcrumbsMerge::new()
                    .merge(&models, Some(&base), &params, &global(0.0, normalize))
                    .unwrap(),
            )
        };
        // (1*2 + 3*6) / 4 = 5, or 20 summed.
        assert_close(&run(true), &[5.0], 1e-6);
        assert_close(&run(false), &[20.0], 1e-6);
    }

    #[test]
    fn ties_variant_drops_the_minority_sign() {
        let base = arr(&[0.0], &[1]);
        let models = [arr(&[10.0], &[1]), arr(&[-1.0], &[1]), arr(&[-1.0], &[1])];
        let params = vec![weight_density(1.0, 1.0); 3];
        let out = BreadcrumbsMerge::new()
            .with_ties()
            .merge(&models, Some(&base), &params, &global(0.0, true))
            .unwrap();
        // Mass +8 elects positive; only +10 agrees.
        assert_close(&values(&out), &[10.0], 1e-6);
    }

    #[test]
    fn gamma_larger_than_density_is_rejected() {
        let base = arr(&[0.0, 0.0], &[2]);
        let err = BreadcrumbsMerge::new().merge(
            &[arr(&[1.0, 2.0], &[2])],
            Some(&base),
            &[weight_density(1.0, 0.5)],
            &global(0.75, true),
        );
        assert!(err.is_err());
    }
}
