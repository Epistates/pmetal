//! DARE merge method - Drop And REscale.
//!
//! DARE is an alternative to TIES that uses random pruning instead of
//! magnitude-based pruning. It randomly drops a fraction of parameters
//! and rescales the remaining ones to maintain the expected value.
//!
//! Formula:
//! ```text
//! mask = random_uniform() < density
//! sparse_delta = mask * delta / density  (if rescale=true)
//! sparse_delta = mask * delta            (if rescale=false)
//! ```
//!
//! Reference: Yu et al., "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch" (2023)
//!
//! Best for:
//! - Alternative to TIES when you want stochastic sparsification
//! - When magnitude-based pruning removes important small-magnitude weights

use super::MergeMethod;
use crate::{MergeError, MergeParameters, Result, sign_consensus};
use pmetal_runtime::compat::{Array, Dtype};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// DARE merge implementation.
#[derive(Debug, Clone)]
pub struct DareMerge {
    /// Whether to use TIES-style sign consensus.
    use_ties_consensus: bool,
    /// Random seed for reproducibility.
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

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Apply random mask with rescaling.
    ///
    /// # Arguments
    /// * `shape` - Shape of the mask to generate.
    /// * `density` - Fraction of parameters to keep (0.0 - 1.0).
    /// * `rescale` - If true, divide kept values by `density` to preserve expected magnitude.
    /// * `seed` - Optional RNG seed for reproducibility.
    fn random_mask(shape: &[i32], density: f32, rescale: bool, seed: Option<u64>) -> Result<Array> {
        let size: usize = shape.iter().map(|&s| s as usize).product();

        // Seed-controlled RNG: use StdRng when a seed is provided so results
        // are reproducible, fall back to the thread-local RNG otherwise.
        let mask: Vec<f32> = match seed {
            Some(s) => {
                let mut rng = StdRng::seed_from_u64(s);
                (0..size)
                    .map(|_| {
                        if rng.random::<f32>() < density {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }
            None => {
                let mut rng = rand::rng();
                (0..size)
                    .map(|_| {
                        if rng.random::<f32>() < density {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }
        };

        let mut mask_array = Array::from_f32_slice(&mask, shape);

        // Rescale to maintain expected value
        if rescale && density > 0.0 && density < 1.0 {
            mask_array = mask_array.divide(&Array::from_f32(density));
        }

        Ok(mask_array)
    }

    /// Apply DARE to a single task vector.
    fn dare_sparsify(&self, delta: &Array, density: f32, rescale: bool) -> Result<Array> {
        if density >= 1.0 {
            return Ok(delta.clone());
        }
        if density <= 0.0 {
            return Ok(Array::zeros(delta.shape(), Dtype::Float32.as_i32()));
        }

        let mask = Self::random_mask(delta.shape(), density, rescale, self.seed)?;
        Ok(delta.multiply(&mask))
    }

    /// Compute task vector (delta from base).
    fn task_vector(tensor: &Array, base: &Array) -> Result<Array> {
        Ok(tensor.subtract(base))
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
        let base = base_tensor.ok_or_else(|| MergeError::BaseModelRequired {
            method: self.name().to_string(),
        })?;

        // Compute task vectors
        let task_vectors: Vec<Array> = tensors
            .iter()
            .map(|t| Self::task_vector(t, base))
            .collect::<Result<Vec<_>>>()?;

        // Get parameters
        let densities: Vec<f32> = params
            .iter()
            .map(|p| global_params.merge_with(p).density())
            .collect();

        let weights: Vec<f32> = params
            .iter()
            .map(|p| global_params.merge_with(p).weight())
            .collect();

        let rescale = global_params.rescale();
        let lambda = global_params.lambda();

        // Apply DARE sparsification to each task vector
        let sparse_vectors: Vec<Array> = task_vectors
            .iter()
            .zip(densities.iter())
            .map(|(tv, &density)| self.dare_sparsify(tv, density, rescale))
            .collect::<Result<Vec<_>>>()?;

        // Optionally apply sign consensus and compute the weighted sum.
        // sign_consensus returns the weighted sum of agreeing contributions directly.
        let weighted_sum = if self.use_ties_consensus {
            sign_consensus(&sparse_vectors, &weights)?
        } else {
            // Compute weighted sum without sign filtering.
            let mut acc = Array::zeros(task_vectors[0].shape(), Dtype::Float32.as_i32());
            for (vector, weight) in sparse_vectors.iter().zip(weights.iter()) {
                let weighted = vector.multiply(&Array::from_f32(*weight));
                acc = acc.add(&weighted);
            }
            acc
        };

        // Scale by lambda and add back to base
        let result = weighted_sum.multiply(&Array::from_f32(lambda));
        Ok(base.add(&result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dare_random_mask() {
        // With density=1.0, all elements should be kept
        let mut mask = DareMerge::random_mask(&[100], 1.0, false, Some(42)).unwrap();
        let mask_slice = mask.to_f32_vec(100).unwrap();
        assert!(mask_slice.iter().all(|&x| x == 1.0));

        // With density=0.0, all elements should be dropped
        let mut mask = DareMerge::random_mask(&[100], 0.0, false, Some(42)).unwrap();
        let mask_slice = mask.to_f32_vec(100).unwrap();
        assert!(mask_slice.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dare_random_mask_seeded_reproducible() {
        // Same seed must produce identical masks.
        let mut m1 = DareMerge::random_mask(&[1000], 0.5, false, Some(99)).unwrap();
        let mut m2 = DareMerge::random_mask(&[1000], 0.5, false, Some(99)).unwrap();
        let s1 = m1.to_f32_vec(1000).unwrap();
        let s2 = m2.to_f32_vec(1000).unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_dare_rescaling() {
        // With rescale=true, kept values should be divided by density
        let mut mask = DareMerge::random_mask(&[1000], 0.5, true, Some(42)).unwrap();
        let mask_slice = mask.to_f32_vec(1000).unwrap();

        // Non-zero values should be 2.0 (1.0 / 0.5)
        for &v in &mask_slice {
            assert!(v == 0.0 || (v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dare_preserves_base_with_zero_lambda() {
        let merge = DareMerge::linear();

        let base = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t1 = Array::from_f32_slice(&[2.0_f32, 3.0, 4.0], &[3]);

        let params = vec![MergeParameters {
            weight: Some(1.0_f32.into()),
            density: Some(0.5_f32.into()),
            ..Default::default()
        }];

        let global = MergeParameters {
            lambda: Some(0.0_f32.into()),
            ..Default::default()
        };

        let mut result = merge.merge(&[t1], Some(&base), &params, &global).unwrap();
        let result_slice = result.to_f32_vec(3).unwrap();
        let mut base_clone = base.clone();
        let base_slice = base_clone.to_f32_vec(3).unwrap();

        // With lambda=0, result should equal base
        for (r, b) in result_slice.iter().zip(base_slice.iter()) {
            assert!((r - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dare_ties_vs_linear() {
        // Just verify both variants can be created and run
        let dare_ties = DareMerge::with_ties();
        let dare_linear = DareMerge::linear();

        assert_eq!(dare_ties.name(), "dare_ties");
        assert_eq!(dare_linear.name(), "dare_linear");
        assert!(dare_ties.use_ties_consensus);
        assert!(!dare_linear.use_ties_consensus);
    }
}
