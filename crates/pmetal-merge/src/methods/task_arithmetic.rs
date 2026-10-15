use pmetal_runtime::compat::{Array, Dtype};

use crate::{MergeMethod, MergeParameters, Result, error::MergeError};

/// Task Arithmetic Merging.
///
/// As described in "Editing Models with Task Arithmetic" (Ilharco et al., 2022).
/// Computes task vectors by subtracting the base model from each fine-tuned model,
/// then sums these task vectors (weighted) and adds them back to the base.
///
/// Formula: `W_new = W_base + lambda * sum(w_i * (W_i - W_base))`
///
/// Best for:
/// - Combining multiple fine-tuned models from the same base
/// - Arithmetic operations on model capabilities (add/subtract skills)
/// - Transfer learning scenarios
#[derive(Debug, Clone, Default)]
pub struct TaskArithmeticMerge;

impl TaskArithmeticMerge {
    /// Create a new Task Arithmetic merge method.
    pub fn new() -> Self {
        Self
    }
}

impl MergeMethod for TaskArithmeticMerge {
    fn name(&self) -> &'static str {
        "task_arithmetic"
    }

    fn description(&self) -> &'static str {
        "Task Arithmetic (Ilharco et al., 2022)"
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
        // Tensors list contains [model1, model2, ...]. Base is separate.
        let base = base_tensor.ok_or_else(|| {
            MergeError::InvalidConfig("Task Arithmetic requires a base model".into())
        })?;

        if tensors.is_empty() {
            return Err(MergeError::InvalidConfig(
                "Task Arithmetic requires at least one fine-tuned model".into(),
            ));
        }

        // Helper to get weight for model i
        let get_weight = |i: usize| -> f32 {
            // Priority: per-model weight -> global weight -> 1.0
            if let Some(p) = params.get(i) {
                if p.weight.is_some() {
                    return p.weight();
                }
            }
            global_params.weight()
        };

        // Helper to get lambda (scaling factor)
        // Priority: global lambda -> 1.0
        let lambda = global_params.lambda();

        // The paper sums the task vectors; `normalize: true` asks for their
        // weighted mean instead (off unless set, unlike the sparsifying methods).
        let weights: Vec<f32> = (0..tensors.len()).map(get_weight).collect();
        let divisor = if global_params.normalize == Some(true) {
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

        // Accumulate weighted task vectors
        let mut accumulated_task_vector = Array::zeros(base.shape(), Dtype::Float32.as_i32());

        for (model_tensor, weight) in tensors.iter().zip(weights.iter().map(|w| w / divisor)) {
            // Task vector = Model - Base
            let task_vector = model_tensor.subtract(base);

            // Weighted task vector
            let weight_arr = Array::from_f32(weight);
            let weighted_task = task_vector.multiply(&weight_arr);

            // Accumulate
            accumulated_task_vector = accumulated_task_vector.add(&weighted_task);
        }

        // Apply global scaling factor (lambda)
        let lambda_arr = Array::from_f32(lambda);
        let final_delta = accumulated_task_vector.multiply(&lambda_arr);

        // Add back to base
        let merged = base.add(&final_delta);

        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{arr, assert_close, values};

    fn weight(w: f32) -> MergeParameters {
        MergeParameters {
            weight: Some(w.into()),
            ..Default::default()
        }
    }

    /// theta = theta_0 + lambda * sum_i w_i (theta_i - theta_0)
    #[test]
    fn sums_the_weighted_task_vectors_and_scales_by_lambda() {
        let base = arr(&[1.0, 1.0], &[2]);
        let models = [arr(&[2.0, 3.0], &[2]), arr(&[4.0, 0.0], &[2])];
        let global = MergeParameters {
            lambda: Some(0.5_f32.into()),
            ..Default::default()
        };
        let out = TaskArithmeticMerge::new()
            .merge(&models, Some(&base), &[weight(0.5), weight(2.0)], &global)
            .unwrap();
        // tau = [1, 2], [3, -1]: 0.5 * (0.5 * [1, 2] + 2 * [3, -1]) = [3.25, -0.5]
        assert_close(&values(&out), &[4.25, 0.5], 1e-6);
    }

    #[test]
    fn explicit_normalize_takes_the_weighted_mean_and_a_negative_weight_subtracts() {
        let base = arr(&[0.0], &[1]);
        let models = [arr(&[4.0], &[1]), arr(&[8.0], &[1])];
        let global = MergeParameters {
            normalize: Some(true),
            ..Default::default()
        };
        let out = TaskArithmeticMerge::new()
            .merge(&models, Some(&base), &[weight(1.0), weight(3.0)], &global)
            .unwrap();
        assert_close(&values(&out), &[7.0], 1e-6); // (4 + 24) / 4

        // Forgetting a skill: negating a task vector (Ilharco et al., 2023).
        let out = TaskArithmeticMerge::new()
            .merge(
                &models[..1],
                Some(&base),
                &[weight(-1.0)],
                &MergeParameters::default(),
            )
            .unwrap();
        assert_close(&values(&out), &[-4.0], 1e-6);
    }
}
