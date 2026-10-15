//! TIES-Merging - Task arithmetic with sparsification and sign consensus.
//!
//! TIES (TrIm, Elect Sign & merge) improves on basic task arithmetic by:
//! 1. Computing task vectors (delta from base model)
//! 2. Trimming each to its top `density` entries by magnitude
//! 3. Electing, per position, the sign with the larger total mass
//! 4. Averaging, per position, only the entries that carry the elected sign
//!    (the disjoint mean), then adding `lambda` times that to the base
//!
//! Reference: Yadav et al., "TIES-Merging: Resolving Interference When Merging Models" (2023)
//!
//! Best for:
//! - Combining multiple fine-tuned models from the same base
//! - Reducing interference when merging many specialized models
//! - When you have models with conflicting capabilities

use super::MergeMethod;
use crate::{
    MergeError, MergeParameters, Result, SignElection, disjoint_merge, sparsify_by_magnitude,
};
use pmetal_runtime::compat::Array;

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
        Ok(tensor.subtract(base))
    }

    /// Apply TIES-Merging to task vectors.
    ///
    /// # Arguments
    /// * `task_vectors` - Task vectors (model - base) for each model
    /// * `densities` - Sparsification density for each model
    /// * `weights` - Weight for each model
    /// * `lambda` - Global scaling factor
    /// * `normalize` - Divide by the agreeing weights (the paper's disjoint
    ///   mean); otherwise sum the agreeing weighted vectors
    pub fn merge_task_vectors(
        task_vectors: &[Array],
        densities: &[f32],
        weights: &[f32],
        lambda: f32,
        normalize: bool,
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

        // Steps 2-3: elect the sign by total mass and take the disjoint mean
        // of the entries that agree with it (TIES paper, Algorithm 1).
        let result = disjoint_merge(&sparse_vectors, weights, SignElection::Mass, normalize)?;

        // Step 3: Scale by lambda
        let result = result.multiply(&Array::from_f32(lambda));

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
        let merged_delta = Self::merge_task_vectors(
            &task_vectors,
            &densities,
            &weights,
            lambda,
            global_params.normalize(),
        )?;

        // Add back to base
        Ok(base.add(&merged_delta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_vector() {
        let tensor = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let base = Array::from_f32_slice(&[0.5_f32, 1.0, 1.5], &[3]);

        let tv = TiesMerge::task_vector(&tensor, &base).unwrap();
        let mut tv = tv;
        let tv_slice = tv.to_f32_vec(3).unwrap();

        assert!((tv_slice[0] - 0.5).abs() < 1e-5);
        assert!((tv_slice[1] - 1.0).abs() < 1e-5);
        assert!((tv_slice[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_ties_preserves_base_with_zero_lambda() {
        let merge = TiesMerge::new();

        let base = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t1 = Array::from_f32_slice(&[2.0_f32, 3.0, 4.0], &[3]);
        let t2 = Array::from_f32_slice(&[3.0_f32, 4.0, 5.0], &[3]);

        let params = vec![
            MergeParameters {
                weight: Some(1.0_f32.into()),
                density: Some(1.0_f32.into()),
                ..Default::default()
            },
            MergeParameters {
                weight: Some(1.0_f32.into()),
                density: Some(1.0_f32.into()),
                ..Default::default()
            },
        ];

        let global = MergeParameters {
            lambda: Some(0.0_f32.into()),
            ..Default::default()
        };

        let result = merge
            .merge(&[t1, t2], Some(&base), &params, &global)
            .unwrap();
        let mut result = result;
        let result_slice = result.to_f32_vec(3).unwrap();
        let mut base_clone = base.clone();
        let base_slice = base_clone.to_f32_vec(3).unwrap();

        // With lambda=0, result should equal base
        for (r, b) in result_slice.iter().zip(base_slice.iter()) {
            assert!((r - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ties_with_full_density() {
        let merge = TiesMerge::new();

        let base = Array::from_f32_slice(&[0.0_f32, 0.0, 0.0], &[3]);
        let t1 = Array::from_f32_slice(&[1.0_f32, 1.0, 1.0], &[3]);

        let params = vec![MergeParameters {
            weight: Some(1.0_f32.into()),
            density: Some(1.0_f32.into()),
            ..Default::default()
        }];

        let global = MergeParameters {
            lambda: Some(1.0_f32.into()),
            ..Default::default()
        };

        let result = merge
            .merge(std::slice::from_ref(&t1), Some(&base), &params, &global)
            .unwrap();
        let mut result = result;
        let result_slice = result.to_f32_vec(3).unwrap();

        // With single model, full density, lambda=1: should equal t1
        for (i, r) in result_slice.iter().enumerate() {
            assert!((r - 1.0).abs() < 1e-5, "index {}: {} != 1.0", i, r);
        }
    }

    // ---- Reference values from Yadav et al. (2023), Algorithm 1 ----------
    //
    // trim: keep the top-`density` fraction of each task vector by magnitude;
    // elect: gamma_p = sgn(sum_t tau_t[p]) over the trimmed vectors;
    // disjoint merge: tau_m[p] = mean of the tau_t[p] whose sign is gamma_p.

    use crate::test_util::{arr, assert_close, values, weight_density};

    fn merge_with(
        base: &[f32],
        models: &[&[f32]],
        params: Vec<MergeParameters>,
        global: MergeParameters,
    ) -> Vec<f32> {
        let shape = [base.len() as i32];
        let tensors: Vec<Array> = models.iter().map(|m| arr(m, &shape)).collect();
        let out = TiesMerge::new()
            .merge(&tensors, Some(&arr(base, &shape)), &params, &global)
            .unwrap();
        values(&out)
    }

    #[test]
    fn elects_the_sign_by_mass_not_by_vote_count() {
        // Position 4 separates the two readings: the models give +10, -1,
        // -1. The signed mass is +8, so the elected sign is positive and
        // only +10 is averaged; a count of signs would elect negative.
        let tau1 = [3.0, -1.0, 2.0, 0.0, 10.0];
        let tau2 = [1.0, -4.0, -3.0, 0.0, -1.0];
        let tau3 = [2.0, 2.0, -1.0, 5.0, -1.0];
        let out = merge_with(
            &[0.0; 5],
            &[&tau1, &tau2, &tau3],
            vec![weight_density(1.0, 1.0); 3],
            MergeParameters::default(),
        );
        // p0: all agree, (3+1+2)/3 = 2;      p1: mass -3, mean(-1,-4) = -2.5
        // p2: mass -2, mean(-3,-1) = -2;     p3: only tau3, 5
        // p4: mass +8, only +10 agrees, 10
        assert_close(&out, &[2.0, -2.5, -2.0, 5.0, 10.0], 1e-6);
    }

    #[test]
    fn trims_each_task_vector_before_electing_and_averaging() {
        // density 0.5 keeps the two largest magnitudes of each vector:
        // tau1 -> [4, -3, 0, 0], tau2 -> [3, 0, -4, 0].
        let out = merge_with(
            &[1.0, 1.0, 1.0, 1.0],
            // base + tau1, base + tau2 with tau1 = [4, -3, 2, 1], tau2 = [3, 2, -4, 1]
            &[&[5.0, -2.0, 3.0, 2.0], &[4.0, 3.0, -3.0, 2.0]],
            vec![weight_density(1.0, 0.5); 2],
            MergeParameters::default(),
        );
        // p0 both agree: (4+3)/2 = 3.5, p1 only tau1: -3, p2 only tau2: -4.
        assert_close(&out, &[4.5, -2.0, -3.0, 1.0], 1e-6);
    }

    #[test]
    fn lambda_scales_the_merged_task_vector() {
        let global = MergeParameters {
            lambda: Some(0.5_f32.into()),
            ..Default::default()
        };
        let out = merge_with(
            &[10.0, 10.0],
            &[&[12.0, 8.0], &[14.0, 8.0]],
            vec![weight_density(1.0, 1.0); 2],
            global,
        );
        // tau = [2, -2] and [4, -2]; merged tau = [3, -2]; 10 + 0.5 * tau.
        assert_close(&out, &[11.5, 9.0], 1e-6);
    }

    #[test]
    fn weights_scale_votes_and_normalize_divides_by_the_agreeing_weight() {
        // Model 1 (weight 2) says +1, model 2 (weight 1) says +1 and -1 on
        // the second element, where the weighted mass 2*1 - 1*3 = -1 elects
        // negative, so only model 2 survives there.
        let params = vec![weight_density(2.0, 1.0), weight_density(1.0, 1.0)];
        let out = merge_with(
            &[0.0, 0.0],
            &[&[1.0, 1.0], &[1.0, -3.0]],
            params.clone(),
            MergeParameters::default(),
        );
        // p0: (2*1 + 1*1) / (2 + 1) = 1;  p1: (1 * -3) / 1 = -3
        assert_close(&out, &[1.0, -3.0], 1e-6);

        let unnormalized = MergeParameters {
            normalize: Some(false),
            ..Default::default()
        };
        let out = merge_with(&[0.0, 0.0], &[&[1.0, 1.0], &[1.0, -3.0]], params, unnormalized);
        // The same election, summed: p0 = 2*1 + 1*1 = 3, p1 = -3.
        assert_close(&out, &[3.0, -3.0], 1e-6);
    }
}
