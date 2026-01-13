//! Linear merge method - simple weighted averaging.
//!
//! The simplest merge method: compute a weighted average of model parameters.
//!
//! Formula: merged[i] = Σ(weight[j] * model[j][i]) / Σ(weight[j])
//!
//! Best for:
//! - Model soups (averaging checkpoints from the same training run)
//! - Simple model combinations where interference is minimal

use super::MergeMethod;
use crate::{MergeError, MergeParameters, Result};
use mlx_rs::Array;

/// Linear merge implementation.
#[derive(Debug, Clone, Default)]
pub struct LinearMerge;

impl LinearMerge {
    /// Create a new linear merge method.
    pub fn new() -> Self {
        Self
    }
}

impl MergeMethod for LinearMerge {
    fn name(&self) -> &'static str {
        "linear"
    }

    fn description(&self) -> &'static str {
        "Simple weighted averaging of parameters"
    }

    fn requires_base_model(&self) -> bool {
        false
    }

    fn merge(
        &self,
        tensors: &[Array],
        _base_tensor: Option<&Array>,
        params: &[MergeParameters],
        global_params: &MergeParameters,
    ) -> Result<Array> {
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        // Get weights for each tensor
        let weights: Vec<f32> = params
            .iter()
            .map(|p| global_params.merge_with(p).weight())
            .collect();

        // Optionally normalize weights
        let weights = if global_params.normalize() {
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                weights.iter().map(|w| w / sum).collect()
            } else {
                weights
            }
        } else {
            weights
        };

        // Compute weighted sum
        let mut result = tensors[0].multiply(Array::from_f32(weights[0]))?;

        for (tensor, weight) in tensors[1..].iter().zip(&weights[1..]) {
            let weighted = tensor.multiply(Array::from_f32(*weight))?;
            result = result.add(&weighted)?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_merge_equal_weights() {
        let merge = LinearMerge::new();

        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t2 = Array::from_slice(&[4.0_f32, 5.0, 6.0], &[3]);

        let params = vec![
            MergeParameters {
                weight: Some(1.0),
                ..Default::default()
            },
            MergeParameters {
                weight: Some(1.0),
                ..Default::default()
            },
        ];

        let global = MergeParameters {
            normalize: Some(true),
            ..Default::default()
        };

        let result = merge.merge(&[t1, t2], None, &params, &global).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Average of [1,2,3] and [4,5,6] = [2.5, 3.5, 4.5]
        assert!((result_slice[0] - 2.5).abs() < 1e-5);
        assert!((result_slice[1] - 3.5).abs() < 1e-5);
        assert!((result_slice[2] - 4.5).abs() < 1e-5);
    }

    #[test]
    fn test_linear_merge_unequal_weights() {
        let merge = LinearMerge::new();

        let t1 = Array::from_slice(&[1.0_f32, 0.0], &[2]);
        let t2 = Array::from_slice(&[0.0_f32, 1.0], &[2]);

        let params = vec![
            MergeParameters {
                weight: Some(0.75),
                ..Default::default()
            },
            MergeParameters {
                weight: Some(0.25),
                ..Default::default()
            },
        ];

        let global = MergeParameters {
            normalize: Some(true),
            ..Default::default()
        };

        let result = merge.merge(&[t1, t2], None, &params, &global).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Weighted average: 0.75 * [1,0] + 0.25 * [0,1] = [0.75, 0.25]
        assert!((result_slice[0] - 0.75).abs() < 1e-5);
        assert!((result_slice[1] - 0.25).abs() < 1e-5);
    }
}
