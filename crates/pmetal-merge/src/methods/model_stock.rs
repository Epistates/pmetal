//! Model Stock merge method.
//!
//! Implements "Model Stock: All we need is just a few fine-tuned models"
//! (Jang, Yun and Han, ECCV 2024, arXiv:2403.19522).
//!
//! The paper observes that fine-tuned weights `w_i` sit on a thin shell
//! around a centre `w_H` that has the best performance, and that the angle
//! `θ` between the task vectors `w_i - w_0` (from the pre-trained weights
//! `w_0`) is nearly constant across fine-tuning runs. The centre is the
//! foot of the perpendicular from `w_0` to the line through the fine-tuned
//! weights' average `w_avg`, which has the closed form
//!
//! ```text
//! w_H = t · w_avg + (1 - t) · w_0,      t = N·cosθ / (1 + (N - 1)·cosθ)
//! ```
//!
//! with `N` fine-tuned models and `cosθ` estimated per tensor as the mean
//! cosine similarity over all pairs of task vectors. Two fine-tuned models
//! are enough for an estimate. Orthogonal task vectors give `t = 0` (stay at
//! the pre-trained weights); identical ones give `t = 1` (the average).
//!
//! A single fine-tuned model has no angle to measure, so it is returned
//! unchanged. `t` is clamped to `[0, 1]`: task vectors that point in
//! opposing directions (`cosθ <= 0`) fall back to the pre-trained weights
//! instead of extrapolating away from them.

use crate::{MergeError, MergeMethod, MergeParameters, Result};
use pmetal_runtime::compat::Array;

/// Model Stock merge method.
#[derive(Debug, Clone, Default)]
pub struct ModelStockMerge;

impl ModelStockMerge {
    /// Create a new Model Stock merger.
    pub fn new() -> Self {
        Self
    }

    /// The interpolation ratio `t` for `n` task vectors whose pairwise
    /// cosine similarity averages `cos_theta`.
    pub fn interpolation_ratio(n: usize, cos_theta: f32) -> f32 {
        let n = n as f32;
        let denominator = 1.0 + (n - 1.0) * cos_theta;
        if denominator <= f32::EPSILON {
            return 0.0;
        }
        (n * cos_theta / denominator).clamp(0.0, 1.0)
    }

    /// Mean cosine similarity over all pairs of `task_vectors`. A pair that
    /// includes an all-zero vector has no direction and counts as 0.
    fn mean_pairwise_cosine(task_vectors: &[Array]) -> f32 {
        let norms: Vec<f32> = task_vectors
            .iter()
            .map(|tv| tv.square().sum_all().sqrt().item_f32())
            .collect();
        let mut total = 0.0_f64;
        let mut pairs = 0_u32;
        for i in 0..task_vectors.len() {
            for j in (i + 1)..task_vectors.len() {
                pairs += 1;
                let denominator = f64::from(norms[i]) * f64::from(norms[j]);
                if denominator > 0.0 {
                    let dot = task_vectors[i]
                        .multiply(&task_vectors[j])
                        .sum_all()
                        .item_f32();
                    total += f64::from(dot) / denominator;
                }
            }
        }
        (total / f64::from(pairs)) as f32
    }
}

impl MergeMethod for ModelStockMerge {
    fn name(&self) -> &'static str {
        "model_stock"
    }

    fn description(&self) -> &'static str {
        "Geometric interpolation using Model Stock algorithm (ECCV 2024)"
    }

    fn requires_base_model(&self) -> bool {
        true
    }

    fn merge(
        &self,
        tensors: &[Array],
        base_tensor: Option<&Array>,
        _params: &[MergeParameters],
        _global_params: &MergeParameters,
    ) -> Result<Array> {
        let base = base_tensor.ok_or(MergeError::BaseModelRequired {
            method: "model_stock".to_string(),
        })?;
        match tensors {
            [] => Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            }),
            [only] => Ok(only.clone()),
            _ => {
                let task_vectors: Vec<Array> =
                    tensors.iter().map(|w| w.subtract(base)).collect();
                let t = Self::interpolation_ratio(
                    tensors.len(),
                    Self::mean_pairwise_cosine(&task_vectors),
                );
                let mut mean = task_vectors[0].clone();
                for tv in &task_vectors[1..] {
                    mean = mean.add(tv);
                }
                let mean = mean.multiply(&Array::from_f32(1.0 / tensors.len() as f32));
                // w_H = w_0 + t · (w_avg - w_0)
                Ok(base.add(&mean.multiply(&Array::from_f32(t))))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{arr, assert_close, values};

    fn merge(base: &[f32], models: &[&[f32]]) -> Vec<f32> {
        let shape = [base.len() as i32];
        let tensors: Vec<Array> = models.iter().map(|m| arr(m, &shape)).collect();
        values(
            &ModelStockMerge::new()
                .merge(
                    &tensors,
                    Some(&arr(base, &shape)),
                    &[],
                    &MergeParameters::default(),
                )
                .unwrap(),
        )
    }

    #[test]
    fn ratio_matches_the_closed_form() {
        // t = N cos / (1 + (N - 1) cos)
        assert!((ModelStockMerge::interpolation_ratio(2, 0.5) - 2.0 / 3.0).abs() < 1e-6);
        assert!((ModelStockMerge::interpolation_ratio(3, 0.5) - 1.5 / 2.0).abs() < 1e-6);
        assert!((ModelStockMerge::interpolation_ratio(4, 1.0) - 1.0).abs() < 1e-6);
        assert_eq!(ModelStockMerge::interpolation_ratio(5, 0.0), 0.0);
        // Opposing task vectors stay at the pre-trained weights.
        assert_eq!(ModelStockMerge::interpolation_ratio(2, -0.5), 0.0);
        assert_eq!(ModelStockMerge::interpolation_ratio(3, -0.9), 0.0);
    }

    #[test]
    fn orthogonal_task_vectors_stay_at_the_pretrained_weights() {
        // cos = 0, so t = 0 although the plain average would be [0.5, 0.5].
        let out = merge(&[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        assert_close(&out, &[0.0, 0.0], 1e-6);
    }

    #[test]
    fn identical_task_vectors_give_their_average() {
        let out = merge(&[1.0, 1.0], &[&[3.0, 5.0], &[3.0, 5.0]]);
        assert_close(&out, &[3.0, 5.0], 1e-6);
    }

    #[test]
    fn two_models_interpolate_toward_the_average_by_t() {
        // tau1 = [2, 0], tau2 = [1, 1]: cos = 2 / (2 * sqrt 2) = 1/sqrt 2,
        // t = 2c / (1 + c) with c = 0.70710678 is 0.82842712, and the
        // average task vector is [1.5, 0.5], so the result is t * [1.5, 0.5].
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let t = 2.0 * c / (1.0 + c);
        let out = merge(&[0.0, 0.0], &[&[2.0, 0.0], &[1.0, 1.0]]);
        assert_close(&out, &[(t * 1.5) as f32, (t * 0.5) as f32], 1e-5);
    }

    #[test]
    fn three_models_use_the_mean_pairwise_cosine() {
        // tau = [1,0,0], [1,1,0], [1,0,1] (base at the origin): the pair
        // cosines are 1/sqrt 2, 1/sqrt 2 and 1/2, mean (sqrt 2 + 1) / 3 ...
        // computed in f64 below. t = 3c / (1 + 2c); average = [1, 1/3, 1/3].
        let r2 = std::f64::consts::FRAC_1_SQRT_2;
        let c = (r2 + r2 + 0.5) / 3.0;
        let t = 3.0 * c / (1.0 + 2.0 * c);
        let out = merge(
            &[0.0, 0.0, 0.0],
            &[&[1.0, 0.0, 0.0], &[1.0, 1.0, 0.0], &[1.0, 0.0, 1.0]],
        );
        assert_close(
            &out,
            &[t as f32, (t / 3.0) as f32, (t / 3.0) as f32],
            1e-5,
        );
    }

    #[test]
    fn the_base_offset_is_removed_before_measuring_angles() {
        // The same geometry as the two-model test, shifted by a base of 10.
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let t = 2.0 * c / (1.0 + c);
        let out = merge(&[10.0, 10.0], &[&[12.0, 10.0], &[11.0, 11.0]]);
        assert_close(&out, &[10.0 + (t * 1.5) as f32, 10.0 + (t * 0.5) as f32], 1e-4);
    }

    #[test]
    fn single_model_is_returned_and_missing_inputs_are_errors() {
        let out = merge(&[1.0, 2.0], &[&[1.5, 2.5]]);
        assert_close(&out, &[1.5, 2.5], 1e-6);

        let base = arr(&[0.0], &[1]);
        let none: [Array; 0] = [];
        let global = MergeParameters::default();
        assert!(ModelStockMerge::new().merge(&none, Some(&base), &[], &global).is_err());
        assert!(
            ModelStockMerge::new()
                .merge(&[arr(&[1.0], &[1])], None, &[], &global)
                .is_err()
        );
    }
}
