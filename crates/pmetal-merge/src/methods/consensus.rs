//! Consensus merging with TALL masks.
//!
//! "Localizing Task Information for Improved Model Merging and Compression"
//! (Wang et al., ICML 2024, arXiv:2405.07813) observes that a multi-task
//! vector `τ_MTL = Σ_t τ_t` carries two kinds of coordinates the individual
//! tasks do not need: ones no task relies on ("catastrophic weights", which
//! only add noise) and ones a single task relies on ("selfish weights",
//! which interfere with the rest). Task-specific information is localized by
//! the TALL mask of each task,
//!
//! ```text
//! m_t = 1[ |τ_t| >= λ · |τ_MTL - τ_t| ]
//! ```
//!
//! which keeps the coordinates where task `t`'s own update outweighs (by the
//! factor `1/λ`) everything the other tasks add there. **Consensus
//! merging** keeps only the coordinates that at least `k` tasks claim,
//!
//! ```text
//! m_cons = 1[ Σ_t m_t >= k ],      θ = θ_base + α · (τ_MTL ⊙ m_cons)
//! ```
//!
//! dropping both the selfish (`k = 2`, the default) and the catastrophic
//! (nobody claims them) weights. `λ` is the `tall_lambda` parameter
//! (default 0.4), `k` is `min_tasks` (default 2, capped at the number of
//! models), `α` is `lambda`, and the per-model `weight` scales each task
//! vector in the final sum (the masks are computed from the unweighted
//! vectors, as in the paper). `normalize: true` divides by the total weight.

use super::MergeMethod;
use crate::{MergeError, MergeParameters, Result};
use pmetal_runtime::compat::{Array, Dtype};

/// Default TALL-mask threshold `λ`.
const DEFAULT_TALL_LAMBDA: f32 = 0.4;
/// Default number of tasks that must claim a coordinate.
const DEFAULT_MIN_TASKS: u32 = 2;

/// Consensus merge (TALL masks) implementation.
#[derive(Debug, Clone, Default)]
pub struct ConsensusMerge;

impl ConsensusMerge {
    /// Create a new consensus merge.
    pub fn new() -> Self {
        Self
    }
}

impl MergeMethod for ConsensusMerge {
    fn name(&self) -> &'static str {
        "consensus"
    }

    fn description(&self) -> &'static str {
        "Consensus merging with TALL masks (Wang et al., 2024)"
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
        let tall_lambda = global_params.tall_lambda.unwrap_or(DEFAULT_TALL_LAMBDA);
        if !tall_lambda.is_finite() || tall_lambda < 0.0 {
            return Err(MergeError::InvalidConfig(format!(
                "tall_lambda must be finite and non-negative, got {tall_lambda}"
            )));
        }
        let min_tasks = global_params.min_tasks.unwrap_or(DEFAULT_MIN_TASKS).max(1);
        let min_tasks = (min_tasks as usize).min(tensors.len()) as f32;

        let task_vectors: Vec<Array> = tensors.iter().map(|t| t.subtract(base)).collect();
        let weights: Vec<f32> = params
            .iter()
            .map(|p| global_params.merge_with(p).weight())
            .collect();

        // τ_MTL = Σ τ_t, and the weighted sum that is actually added.
        let mut multi_task = task_vectors[0].clone();
        for tv in &task_vectors[1..] {
            multi_task = multi_task.add(tv);
        }

        // How many tasks claim each coordinate: |τ_t| >= λ |τ_MTL - τ_t|.
        let lambda = Array::from_f32(tall_lambda);
        let mut claims = Array::zeros(base.shape(), Dtype::Float32.as_i32());
        for tv in &task_vectors {
            let others = multi_task.subtract(tv).abs_val().multiply(&lambda);
            let mask = tv
                .abs_val()
                .greater_equal(&others)
                .as_dtype(Dtype::Float32.as_i32());
            claims = claims.add(&mask);
        }
        let consensus = claims
            .greater_equal(&Array::from_f32(min_tasks))
            .as_dtype(Dtype::Float32.as_i32());

        let divisor = if global_params.normalize.unwrap_or(false) {
            let total: f32 = weights.iter().sum();
            if total.abs() < 1e-12 {
                return Err(MergeError::InvalidConfig(
                    "Cannot normalize weights: they sum to zero".to_string(),
                ));
            }
            total
        } else {
            1.0
        };
        let mut combined = task_vectors[0].multiply(&Array::from_f32(weights[0] / divisor));
        for (tv, w) in task_vectors[1..].iter().zip(&weights[1..]) {
            combined = combined.add(&tv.multiply(&Array::from_f32(w / divisor)));
        }

        let delta = combined
            .multiply(&consensus)
            .multiply(&Array::from_f32(global_params.lambda()));
        Ok(base.add(&delta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{arr, assert_close, values};

    fn run(base: &[f32], taus: &[&[f32]], global: &MergeParameters) -> Vec<f32> {
        let shape = [base.len() as i32];
        let models: Vec<Array> = taus
            .iter()
            .map(|tau| {
                let m: Vec<f32> = tau.iter().zip(base).map(|(t, b)| t + b).collect();
                arr(&m, &shape)
            })
            .collect();
        let params = vec![MergeParameters::default(); models.len()];
        values(
            &ConsensusMerge::new()
                .merge(&models, Some(&arr(base, &shape)), &params, global)
                .unwrap(),
        )
    }

    #[test]
    fn drops_selfish_and_catastrophic_coordinates() {
        // Three tasks, lambda 0.5, k 2. Per coordinate (tau1, tau2, tau3):
        //  p0 (4, 4, 0):   MTL 8; tasks 1, 2 claim it (4 >= 0.5 * 4) -> keep 8
        //  p1 (1, 0, 1):   MTL 2; tasks 1, 3 claim it -> keep 2
        //  p2 (0, 1, 1):   MTL 2; tasks 2, 3 claim it -> keep 2
        //  p3 (5, 0, 0):   only task 1 claims it (selfish) -> 0
        //  p4 (1, -1, 0):  MTL 0; all three claim it -> keep MTL = 0
        //  p5 (2, 2, -3):  MTL 1; all three claim it -> keep 1
        let global = MergeParameters {
            tall_lambda: Some(0.5),
            ..Default::default()
        };
        let out = run(
            &[0.0; 6],
            &[
                &[4.0, 1.0, 0.0, 5.0, 1.0, 2.0],
                &[4.0, 0.0, 1.0, 0.0, -1.0, 2.0],
                &[0.0, 1.0, 1.0, 0.0, 0.0, -3.0],
            ],
            &global,
        );
        assert_close(&out, &[8.0, 2.0, 2.0, 0.0, 0.0, 1.0], 1e-6);
    }

    #[test]
    fn a_large_lambda_keeps_only_coordinates_no_other_task_touches() {
        // With lambda 10 a task claims a coordinate only when the other tasks
        // add (almost) nothing there; with k = 1 that is every coordinate
        // owned by exactly one task.
        let global = MergeParameters {
            tall_lambda: Some(10.0),
            min_tasks: Some(1),
            ..Default::default()
        };
        let out = run(&[1.0; 3], &[&[2.0, 3.0, 0.0], &[0.0, 3.0, 0.0]], &global);
        // p0: only task 1 (owner) -> +2; p1: both tasks, neither dominates
        // 10x -> dropped; p2: nothing there -> unchanged.
        assert_close(&out, &[3.0, 1.0, 1.0], 1e-6);
    }

    #[test]
    fn alpha_and_weights_scale_the_surviving_sum() {
        let global = MergeParameters {
            tall_lambda: Some(1.0),
            lambda: Some(0.5_f32.into()),
            ..Default::default()
        };
        // Both tasks claim p0 (equal updates), so MTL * m = 2 + 2 weighted
        // 1 and 3: (1 * 2 + 3 * 2) * 0.5 = 4.
        let base = arr(&[0.0], &[1]);
        let models = [arr(&[2.0], &[1]), arr(&[2.0], &[1])];
        let params = [
            MergeParameters {
                weight: Some(1.0_f32.into()),
                ..Default::default()
            },
            MergeParameters {
                weight: Some(3.0_f32.into()),
                ..Default::default()
            },
        ];
        let out = ConsensusMerge::new()
            .merge(&models, Some(&base), &params, &global)
            .unwrap();
        assert_close(&values(&out), &[4.0], 1e-6);
    }

    #[test]
    fn needs_a_base_model_and_valid_parameters() {
        let models = [arr(&[1.0], &[1])];
        let params = [MergeParameters::default()];
        assert!(
            ConsensusMerge::new()
                .merge(&models, None, &params, &MergeParameters::default())
                .is_err()
        );
        let bad = MergeParameters {
            tall_lambda: Some(-1.0),
            ..Default::default()
        };
        let base = arr(&[0.0], &[1]);
        assert!(
            ConsensusMerge::new()
                .merge(&models, Some(&base), &params, &bad)
                .is_err()
        );
    }
}
