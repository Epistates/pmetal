//! DARE merge method - Drop And REscale.
//!
//! DARE (Yu et al., "Language Models are Super Mario", 2024) observes that
//! most of a fine-tuned model's task vector is redundant: dropping a random
//! fraction `1 - density` of its entries and multiplying the survivors by
//! `1 / density` preserves the model's behaviour in expectation, and the
//! thinned vectors of several models interfere less when combined.
//!
//! ```text
//! m ~ Bernoulli(density)                 (independent per entry)
//! τ̂ = m ⊙ τ / density                    (rescale = true, the default)
//! θ = θ_base + λ · combine(τ̂_1, ..., τ̂_N)
//! ```
//!
//! `combine` is the weighted sum (`dare_linear`) or, for `dare_ties`, the
//! TIES sign election and disjoint merge. With `normalize` (the default) the
//! weighted sum is divided by the total weight and the disjoint merge is a
//! mean over the agreeing models.
//!
//! Each (tensor, model) pair draws its own mask from the `seed` parameter, so
//! a merge is reproducible and no two models share a drop pattern.
//!
//! Best for:
//! - Alternative to TIES when you want stochastic sparsification
//! - When magnitude-based pruning removes important small-magnitude weights

use super::MergeMethod;
use crate::consensus::combine_task_vectors;
use crate::seed::mask_seed;
use crate::{MergeError, MergeParameters, Result, sparsify_dare};
use pmetal_runtime::compat::Array;

/// DARE merge implementation.
#[derive(Debug, Clone)]
pub struct DareMerge {
    /// Whether to use TIES-style sign consensus.
    use_ties_consensus: bool,
    /// Seed overriding the `seed` merge parameter.
    seed: Option<u64>,
}

impl Default for DareMerge {
    fn default() -> Self {
        Self::new(false)
    }
}

impl DareMerge {
    /// Create a new DARE merge method.
    ///
    /// # Arguments
    /// * `use_ties_consensus` - If true, apply TIES sign consensus after random pruning
    pub fn new(use_ties_consensus: bool) -> Self {
        Self {
            use_ties_consensus,
            seed: None,
        }
    }

    /// Create DARE with TIES consensus (dare_ties).
    pub fn with_ties() -> Self {
        Self::new(true)
    }

    /// Create DARE with linear combination (dare_linear).
    pub fn linear() -> Self {
        Self::new(false)
    }

    /// Fix the seed, taking precedence over the `seed` merge parameter.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn merge_tensor(
        &self,
        name: &str,
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

        let rescale = global_params.rescale();
        let seed = self.seed.unwrap_or_else(|| global_params.seed());

        let mut sparse = Vec::with_capacity(tensors.len());
        let mut weights = Vec::with_capacity(tensors.len());
        for (slot, (tensor, per_model)) in tensors.iter().zip(params).enumerate() {
            let effective = global_params.merge_with(per_model);
            let task_vector = tensor.subtract(base);
            sparse.push(sparsify_dare(
                &task_vector,
                effective.density(),
                mask_seed(seed, name, slot),
                rescale,
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

impl MergeMethod for DareMerge {
    fn name(&self) -> &'static str {
        if self.use_ties_consensus {
            "dare_ties"
        } else {
            "dare_linear"
        }
    }

    fn description(&self) -> &'static str {
        if self.use_ties_consensus {
            "Random pruning with rescaling and TIES sign consensus"
        } else {
            "Random pruning with rescaling and linear combination"
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
        self.merge_tensor("", tensors, base_tensor, params, global_params)
    }

    fn merge_named(
        &self,
        name: &str,
        tensors: &[Array],
        base_tensor: Option<&Array>,
        params: &[MergeParameters],
        global_params: &MergeParameters,
    ) -> Result<Array> {
        let params: Vec<MergeParameters> = params.iter().map(|p| p.resolved(name)).collect();
        self.merge_tensor(
            name,
            tensors,
            base_tensor,
            &params,
            &global_params.resolved(name),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{arr, assert_close, values, weight_density};

    fn run(
        method: &DareMerge,
        base: &Array,
        models: &[Array],
        params: &[MergeParameters],
        global: &MergeParameters,
    ) -> Vec<f32> {
        values(&method.merge(models, Some(base), params, global).unwrap())
    }

    #[test]
    fn full_density_is_a_weighted_task_arithmetic() {
        // delta1 = 4, delta2 = 8 with weights 1 and 3.
        let base = arr(&[10.0], &[1]);
        let models = [arr(&[14.0], &[1]), arr(&[18.0], &[1])];
        let params = [weight_density(1.0, 1.0), weight_density(3.0, 1.0)];

        // normalize (default): 10 + (1*4 + 3*8) / 4 = 17
        let out = run(
            &DareMerge::linear(),
            &base,
            &models,
            &params,
            &MergeParameters::default(),
        );
        assert_close(&out, &[17.0], 1e-6);

        // No normalization: 10 + 28 = 38; lambda scales the sum.
        let global = MergeParameters {
            normalize: Some(false),
            lambda: Some(0.5_f32.into()),
            ..Default::default()
        };
        let out = run(&DareMerge::linear(), &base, &models, &params, &global);
        assert_close(&out, &[24.0], 1e-6);
    }

    #[test]
    fn zero_lambda_returns_the_base() {
        let base = arr(&[1.0, 2.0, 3.0], &[3]);
        let models = [arr(&[2.0, 3.0, 4.0], &[3])];
        let global = MergeParameters {
            lambda: Some(0.0_f32.into()),
            ..Default::default()
        };
        let out = run(
            &DareMerge::linear(),
            &base,
            &models,
            &[weight_density(1.0, 0.5)],
            &global,
        );
        assert_close(&out, &[1.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn survivors_are_rescaled_by_the_inverse_density() {
        let n = 4096;
        let base = arr(&vec![0.0; n], &[n as i32]);
        let models = [arr(&vec![1.0; n], &[n as i32])];
        let out = run(
            &DareMerge::linear(),
            &base,
            &models,
            &[weight_density(1.0, 0.25)],
            &MergeParameters::default(),
        );
        assert!(out.iter().all(|&v| v == 0.0 || v == 4.0));
        assert!(out.iter().any(|&v| v == 4.0) && out.iter().any(|&v| v == 0.0));
    }

    #[test]
    fn merges_are_reproducible_by_default_and_follow_the_seed() {
        let n = 2048;
        let base = arr(&vec![0.0; n], &[n as i32]);
        let models = [arr(&vec![1.0; n], &[n as i32])];
        let params = [weight_density(1.0, 0.5)];
        let merge = |seed: Option<u64>| {
            let global = MergeParameters {
                seed,
                ..Default::default()
            };
            run(&DareMerge::linear(), &base, &models, &params, &global)
        };
        assert_eq!(merge(None), merge(None));
        assert_eq!(merge(Some(3)), merge(Some(3)));
        assert_ne!(merge(Some(3)), merge(Some(4)));
    }

    #[test]
    fn models_and_tensors_draw_independent_masks() {
        let n = 8192;
        let base = arr(&vec![0.0; n], &[n as i32]);
        // Two models with the same all-ones task vector, no normalization:
        // independent masks give entries of 0, 2 (one survivor) and 4 (both).
        let models = [arr(&vec![1.0; n], &[n as i32]), arr(&vec![1.0; n], &[n as i32])];
        let params = [weight_density(1.0, 0.5), weight_density(1.0, 0.5)];
        let global = MergeParameters {
            normalize: Some(false),
            ..Default::default()
        };
        let out = run(&DareMerge::linear(), &base, &models, &params, &global);
        let count = |v: f32| out.iter().filter(|&&x| x == v).count() as f32 / n as f32;
        assert!(out.iter().all(|&x| x == 0.0 || x == 2.0 || x == 4.0));
        // Independent Bernoulli(0.5): P(0) = P(4) = 0.25, P(2) = 0.5. Shared
        // masks would give P(2) = 0.
        assert!((count(2.0) - 0.5).abs() < 0.05, "P(one survivor) {}", count(2.0));
        assert!((count(0.0) - 0.25).abs() < 0.05, "P(none) {}", count(0.0));

        // The same model slot in two differently named tensors differs too.
        let merge_named = |name: &str| {
            values(
                &DareMerge::linear()
                    .merge_named(name, &models[..1], Some(&base), &params[..1], &global)
                    .unwrap(),
            )
        };
        assert_ne!(merge_named("a.weight"), merge_named("b.weight"));
        assert_eq!(merge_named("a.weight"), merge_named("a.weight"));
    }

    #[test]
    fn dare_ties_elects_a_sign_among_the_survivors() {
        // Full density turns DARE-TIES into plain TIES: mass +8 at p0 keeps
        // only +10; both negative at p1 average to -3.
        let base = arr(&[0.0, 0.0], &[2]);
        let models = [
            arr(&[10.0, -4.0], &[2]),
            arr(&[-1.0, -2.0], &[2]),
            arr(&[-1.0, 0.0], &[2]),
        ];
        let params = vec![weight_density(1.0, 1.0); 3];
        let out = run(
            &DareMerge::with_ties(),
            &base,
            &models,
            &params,
            &MergeParameters::default(),
        );
        assert_close(&out, &[10.0, -3.0], 1e-6);
    }

    #[test]
    fn rejects_an_out_of_range_density() {
        let base = arr(&[0.0], &[1]);
        let models = [arr(&[1.0], &[1])];
        let err = DareMerge::linear().merge(
            &models,
            Some(&base),
            &[weight_density(1.0, 1.5)],
            &MergeParameters::default(),
        );
        assert!(err.is_err());
    }
}
