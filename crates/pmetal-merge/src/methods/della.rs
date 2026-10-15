//! DELLA-Merging - magnitude-aware dropout for task vectors.
//!
//! DELLA (Deep et al., "DELLA-Merging: Reducing Interference in Model
//! Merging through Magnitude-Based Sampling", 2024, arXiv:2406.11617) refines
//! DARE's uniform dropout. Its MAGPRUNE step ranks the entries of each row of
//! a task vector by magnitude and gives each a keep probability that grows
//! linearly with its rank, from `density - epsilon` for the smallest entry to
//! `density + epsilon` for the largest. Small updates, which carry the least
//! signal, are dropped most often, yet the average keep rate stays `density`.
//! Every survivor is divided by its own keep probability, so each entry's
//! expectation is unchanged:
//!
//! ```text
//! p_i  = (density - epsilon) + 2 * epsilon * rank_i / (n - 1)     (per row)
//! τ̂_i  = τ_i / p_i   with probability p_i, else 0
//! ```
//!
//! The thinned vectors are then combined as in TIES (elect a sign by total
//! mass, take the disjoint mean) for `della`, or as a weighted sum for
//! `della_linear`:
//!
//! ```text
//! θ = θ_base + λ · combine(τ̂_1, ..., τ̂_N)
//! ```
//!
//! `density` is per model, `epsilon` is the global `epsilon` parameter
//! (default 0.1), and each (tensor, model) pair draws its own mask from the
//! `seed` parameter.

use super::MergeMethod;
use crate::consensus::combine_task_vectors;
use crate::seed::mask_seed;
use crate::{MergeError, MergeParameters, Result, sparsify_della};
use pmetal_runtime::compat::Array;

/// DELLA merge implementation.
#[derive(Debug, Clone)]
pub struct DellaMerge {
    /// Whether to elect a sign and take the disjoint mean (`della`) instead
    /// of a plain weighted sum (`della_linear`).
    use_ties_consensus: bool,

    /// Seed overriding the `seed` merge parameter.
    seed: Option<u64>,
}

impl Default for DellaMerge {
    fn default() -> Self {
        Self::new()
    }
}

impl DellaMerge {
    /// DELLA with TIES sign election (`della`).
    pub fn new() -> Self {
        Self {
            use_ties_consensus: true,
            seed: None,
        }
    }

    /// DELLA with a weighted-sum combination (`della_linear`).
    pub fn linear() -> Self {
        Self {
            use_ties_consensus: false,
            seed: None,
        }
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
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }
        let base = base_tensor.ok_or_else(|| MergeError::BaseModelRequired {
            method: self.name().to_string(),
        })?;
        if params.len() != tensors.len() {
            return Err(MergeError::InvalidConfig(format!(
                "{} tensors but {} per-model parameter sets",
                tensors.len(),
                params.len()
            )));
        }

        let epsilon = global_params.epsilon();
        let seed = self.seed.unwrap_or_else(|| global_params.seed());

        let mut sparse = Vec::with_capacity(tensors.len());
        let mut weights = Vec::with_capacity(tensors.len());
        for (slot, (tensor, per_model)) in tensors.iter().zip(params).enumerate() {
            let effective = global_params.merge_with(per_model);
            let task_vector = tensor.subtract(base);
            sparse.push(sparsify_della(
                &task_vector,
                effective.density(),
                epsilon,
                mask_seed(seed, name, slot),
                true,
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

impl MergeMethod for DellaMerge {
    fn name(&self) -> &'static str {
        if self.use_ties_consensus {
            "della"
        } else {
            "della_linear"
        }
    }

    fn description(&self) -> &'static str {
        if self.use_ties_consensus {
            "Magnitude-ranked dropout (MAGPRUNE) with TIES sign election"
        } else {
            "Magnitude-ranked dropout (MAGPRUNE) with linear combination"
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
        let params: Vec<MergeParameters> = params
            .iter()
            .map(|p| p.resolved_in(name, global_params.layer_count))
            .collect();
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

    #[test]
    fn names_and_variants() {
        assert_eq!(DellaMerge::new().name(), "della");
        assert_eq!(DellaMerge::linear().name(), "della_linear");
        assert!(DellaMerge::new().requires_base_model());
    }

    #[test]
    fn full_density_reduces_to_the_combination_step() {
        // Nothing is dropped at density 1, so della_linear is the weighted
        // sum: base 1 + (1*2 + 1*4) / 2 = 4.
        let base = arr(&[1.0], &[1]);
        let models = [arr(&[3.0], &[1]), arr(&[5.0], &[1])];
        let params = [weight_density(1.0, 1.0), weight_density(1.0, 1.0)];
        let out = DellaMerge::linear()
            .merge(&models, Some(&base), &params, &MergeParameters::default())
            .unwrap();
        assert_close(&values(&out), &[4.0], 1e-6);
    }

    #[test]
    fn survivors_are_divided_by_their_rank_dependent_keep_probability() {
        // One model, one row of 1000 entries with magnitudes 1..=1000 and
        // density 0.5, epsilon 0.4: the entry of rank r keeps with
        // p = 0.1 + 0.8 r / 999 and, if kept, is 1/p times its value.
        let n = 1000;
        let task: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let base = arr(&vec![0.0; n], &[1, n as i32]);
        let models = [arr(&task, &[1, n as i32])];
        let global = MergeParameters {
            epsilon: Some(0.4),
            normalize: Some(false),
            ..Default::default()
        };
        let out = values(
            &DellaMerge::linear()
                .merge(&models, Some(&base), &[weight_density(1.0, 0.5)], &global)
                .unwrap(),
        );
        let mut kept = 0_i32;
        for (r, (&x, &y)) in task.iter().zip(&out).enumerate() {
            if y != 0.0 {
                kept += 1;
                let p = 0.1 + 0.8 * r as f32 / (n as f32 - 1.0);
                assert!(
                    (y - x / p).abs() <= 1e-3 * (x / p),
                    "rank {r}: {y} vs {}",
                    x / p
                );
            }
        }
        // E[kept] = n * density = 500; binomial sigma is about 16.
        assert!((kept - 500).abs() < 100, "kept {kept}");
    }

    #[test]
    fn merges_are_reproducible_and_seeded_per_model() {
        let n = 2048;
        let base = arr(&vec![0.0; n], &[1, n as i32]);
        let task: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let models = [arr(&task, &[1, n as i32]), arr(&task, &[1, n as i32])];
        let params = [weight_density(1.0, 0.5), weight_density(1.0, 0.5)];
        let run = |seed| {
            let global = MergeParameters {
                seed,
                normalize: Some(false),
                ..Default::default()
            };
            values(
                &DellaMerge::linear()
                    .merge(&models, Some(&base), &params, &global)
                    .unwrap(),
            )
        };
        assert_eq!(run(None), run(None));
        assert_ne!(run(Some(1)), run(Some(2)));

        // Two identical models still drop different entries: at many
        // positions exactly one of them survives, so the value is x/p where
        // both surviving would give 2x/p (default epsilon 0.1, so p runs
        // from 0.4 to 0.6 with rank).
        let out = run(Some(1));
        let one_of_two = out
            .iter()
            .zip(&task)
            .enumerate()
            .filter(|(r, (y, x))| {
                let p = 0.4 + 0.2 * *r as f32 / (n as f32 - 1.0);
                (**y - **x / p).abs() <= 1e-3 * (**x / p)
            })
            .count();
        // P(exactly one) = 2p(1-p), about 0.48 here.
        assert!(
            one_of_two > n / 3,
            "{one_of_two} positions kept by one model only"
        );
    }

    #[test]
    fn rejects_a_missing_base_and_bad_parameters() {
        let models = [arr(&[1.0], &[1])];
        let params = [weight_density(1.0, 0.5)];
        let global = MergeParameters::default();
        assert!(
            DellaMerge::new()
                .merge(&models, None, &params, &global)
                .is_err()
        );
        let base = arr(&[0.0], &[1]);
        assert!(
            DellaMerge::new()
                .merge(&models, Some(&base), &[weight_density(1.0, 1.5)], &global)
                .is_err()
        );
        assert!(
            DellaMerge::new()
                .merge(&models, Some(&base), &[], &global)
                .is_err()
        );
    }
}
