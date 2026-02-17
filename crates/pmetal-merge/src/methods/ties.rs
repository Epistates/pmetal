//! TIES-Merging - Task arithmetic with sparsification and sign consensus.
//!
//! TIES (TrIm, Elect Sign & merge) improves on basic task arithmetic by:
//! 1. Computing task vectors (delta from base model)
//! 2. Sparsifying by keeping only top `density` parameters by magnitude
//! 3. Applying sign consensus - only keep parameters where models agree on direction
//! 4. Merging the sparsified, sign-agreed task vectors
//!
//! Reference: Yadav et al., "TIES-Merging: Resolving Interference When Merging Models" (2023)
//!
//! Best for:
//! - Combining multiple fine-tuned models from the same base
//! - Reducing interference when merging many specialized models
//! - When you have models with conflicting capabilities

use super::MergeMethod;
use crate::{sign_consensus, sparsify_by_magnitude, MergeError, MergeParameters, Result};
use mlx_rs::Array;

/// TIES merge implementation.
#[derive(Debug, Clone, Default)]
pub struct TiesMerge;

impl TiesMerge {
    /// Create a new TIES merge method.
    pub fn new() -> Self {
        Self
    }

    /// Compute task vector (delta from base).
    fn task_vector(tensor: &Array, base: &Array) -> Result<Array> {
        Ok(tensor.subtract(base)?)
    }

    /// Apply TIES-Merging to task vectors.
    ///
    /// # Arguments
    /// * `task_vectors` - Task vectors (model - base) for each model
    /// * `densities` - Sparsification density for each model
    /// * `weights` - Weight for each model
    /// * `lambda` - Global scaling factor
    pub fn merge_task_vectors(
        task_vectors: &[Array],
        densities: &[f32],
        weights: &[f32],
        lambda: f32,
    ) -> Result<Array> {
        if task_vectors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        if densities.len() != task_vectors.len() {
            return Err(MergeError::InvalidConfig(format!(
                "densities length ({}) must match task_vectors length ({})",
                densities.len(),
                task_vectors.len()
            )));
        }

        for (i, &d) in densities.iter().enumerate() {
            if !(0.0..=1.0).contains(&d) {
                return Err(MergeError::InvalidConfig(format!(
                    "density[{}] = {} is out of valid range [0.0, 1.0]",
                    i, d
                )));
            }
        }

        // Step 1: Sparsify each task vector by magnitude
        let sparse_vectors: Vec<Array> = task_vectors
            .iter()
            .zip(densities.iter())
            .map(|(tv, &density)| sparsify_by_magnitude(tv, density))
            .collect::<Result<Vec<_>>>()?;

        // Step 2: Apply sign consensus
        // Only keep parameters where the weighted majority agrees on the sign
        let consensus_mask = sign_consensus(&sparse_vectors, weights)?;

        // Step 3: Apply consensus mask and compute weighted sum
        let mut result = Array::zeros::<f32>(task_vectors[0].shape())?;

        for (sparse, weight) in sparse_vectors.iter().zip(weights.iter()) {
            let masked = sparse.multiply(&consensus_mask)?;
            let weighted = masked.multiply(Array::from_f32(*weight))?;
            result = result.add(&weighted)?;
        }

        // Step 4: Scale by lambda
        result = result.multiply(Array::from_f32(lambda))?;

        Ok(result)
    }
}

impl MergeMethod for TiesMerge {
    fn name(&self) -> &'static str {
        "ties"
    }

    fn description(&self) -> &'static str {
        "Task arithmetic with sparsification and sign consensus"
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
            method: "TIES".to_string(),
        })?;

        // Compute task vectors
        let task_vectors: Vec<Array> = tensors
            .iter()
            .map(|t| Self::task_vector(t, base))
            .collect::<Result<Vec<_>>>()?;

        // Get parameters for each model
        let densities: Vec<f32> = params
            .iter()
            .map(|p| global_params.merge_with(p).density())
            .collect();

        let weights: Vec<f32> = params
            .iter()
            .map(|p| global_params.merge_with(p).weight())
            .collect();

        let lambda = global_params.lambda();

        // Merge task vectors
        let merged_delta = Self::merge_task_vectors(&task_vectors, &densities, &weights, lambda)?;

        // Add back to base
        Ok(base.add(&merged_delta)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_vector() {
        let tensor = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let base = Array::from_slice(&[0.5_f32, 1.0, 1.5], &[3]);

        let tv = TiesMerge::task_vector(&tensor, &base).unwrap();
        let tv_slice: Vec<f32> = tv.as_slice().to_vec();

        assert!((tv_slice[0] - 0.5).abs() < 1e-5);
        assert!((tv_slice[1] - 1.0).abs() < 1e-5);
        assert!((tv_slice[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_ties_preserves_base_with_zero_lambda() {
        let merge = TiesMerge::new();

        let base = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t1 = Array::from_slice(&[2.0_f32, 3.0, 4.0], &[3]);
        let t2 = Array::from_slice(&[3.0_f32, 4.0, 5.0], &[3]);

        let params = vec![
            MergeParameters {
                weight: Some(1.0),
                density: Some(1.0),
                ..Default::default()
            },
            MergeParameters {
                weight: Some(1.0),
                density: Some(1.0),
                ..Default::default()
            },
        ];

        let global = MergeParameters {
            lambda: Some(0.0),
            ..Default::default()
        };

        let result = merge
            .merge(&[t1, t2], Some(&base), &params, &global)
            .unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();
        let base_slice: Vec<f32> = base.as_slice().to_vec();

        // With lambda=0, result should equal base
        for (r, b) in result_slice.iter().zip(base_slice.iter()) {
            assert!((r - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ties_with_full_density() {
        let merge = TiesMerge::new();

        let base = Array::from_slice(&[0.0_f32, 0.0, 0.0], &[3]);
        let t1 = Array::from_slice(&[1.0_f32, 1.0, 1.0], &[3]);

        let params = vec![MergeParameters {
            weight: Some(1.0),
            density: Some(1.0),
            ..Default::default()
        }];

        let global = MergeParameters {
            lambda: Some(1.0),
            ..Default::default()
        };

        let result = merge
            .merge(std::slice::from_ref(&t1), Some(&base), &params, &global)
            .unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // With single model, full density, lambda=1: should equal t1
        for (i, r) in result_slice.iter().enumerate() {
            assert!((r - 1.0).abs() < 1e-5, "index {}: {} != 1.0", i, r);
        }
    }
}
