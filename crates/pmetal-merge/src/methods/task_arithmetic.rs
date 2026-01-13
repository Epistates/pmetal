use mlx_rs::Array;

use crate::{error::MergeError, MergeMethod, MergeParameters, Result};

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
            params
                .get(i)
                .and_then(|p| p.weight)
                .or(global_params.weight)
                .unwrap_or(1.0)
        };

        // Helper to get lambda (scaling factor)
        // Priority: global lambda -> 1.0
        let lambda = global_params.lambda.unwrap_or(1.0);

        // Accumulate weighted task vectors
        let mut accumulated_task_vector = mlx_rs::ops::zeros_like(base)?;

        for (i, model_tensor) in tensors.iter().enumerate() {
            let weight = get_weight(i);

            // Task vector = Model - Base
            let task_vector = model_tensor.subtract(base)?;

            // Weighted task vector
            let weight_arr = Array::from_f32(weight);
            let weighted_task = task_vector.multiply(&weight_arr)?;

            // Accumulate
            accumulated_task_vector = accumulated_task_vector.add(&weighted_task)?;
        }

        // Apply global scaling factor (lambda)
        let lambda_arr = Array::from_f32(lambda);
        let final_delta = accumulated_task_vector.multiply(&lambda_arr)?;

        // Add back to base
        let merged = base.add(&final_delta)?;

        Ok(merged)
    }
}
