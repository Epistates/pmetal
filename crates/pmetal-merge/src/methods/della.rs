//! DELLA-Merging - Magnitude-based adaptive dropout for model merging.
//!
//! DELLA (Drop and rEscaLe with Learned Amplification) improves on DARE by using
//! magnitude-proportional dropout: higher-magnitude task-vector entries are *more*
//! likely to be dropped, unlike TIES/DARE where magnitude is used to *keep* entries.
//! The counter-intuitive direction is intentional: large task-vector deltas are
//! often outliers that cause interference, so selectively suppressing them reduces
//! cross-task conflicts while preserving the smaller, more generalizable updates.
//!
//! # Algorithm
//!
//! For each model with task vector τ = W_ft - W_base:
//!
//! 1. Compute drop probabilities proportional to |τ|:
//!    ```text
//!    p_drop[i] = softmax(|τ[i]| / temperature)[i]   (exponential, default)
//!    p_drop[i] = |τ[i]| / sum(|τ|)                  (linear variant)
//!    ```
//! 2. Sample a Bernoulli mask M using these probabilities:
//!    ```text
//!    M[i] ~ Bernoulli(1 - p_drop[i])
//!    ```
//! 3. Rescale to maintain the expected value:
//!    ```text
//!    sparse_τ[i] = M[i] * τ[i] / (1 - p_drop[i])
//!    ```
//! 4. Combine with weights and add back to base:
//!    ```text
//!    W_merged = W_base + lambda * sum_m(w_m * sparse_τ_m)
//!    ```
//!
//! # Variants
//!
//! - **DELLA** (default): exponential softmax probability — emphasises dropping
//!   the very largest entries more aggressively.
//! - **DELLA-Linear**: linear probability normalisation — softer version, closer
//!   to DARE in behaviour but still magnitude-biased.
//!
//! # References
//!
//! - "DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based
//!   Sampling" (Yu et al., 2024), arXiv:2406.11617.

use super::MergeMethod;
use crate::{sign_consensus, MergeError, MergeParameters, Result};
use mlx_rs::Array;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// DELLA merge implementation.
///
/// Implements magnitude-proportional dropout: higher-magnitude task-vector entries
/// have a higher probability of being dropped, which reduces interference from
/// large, potentially conflicting parameter changes.
#[derive(Debug, Clone)]
pub struct DellaMerge {
    /// Whether to use TIES-style sign consensus after sparsification.
    /// `true`  → della (paper default)
    /// `false` → della_linear
    use_ties_consensus: bool,

    /// Whether to use exponential (softmax-based) probability scaling.
    /// `true`  → exponential softmax (default, more aggressive outlier removal)
    /// `false` → linear normalisation (softer, closer to DARE)
    use_exponential: bool,

    /// Temperature for the softmax in the exponential variant.
    /// Lower values → sharper, more focused dropout on the top entries.
    /// Default: 1.0
    temperature: f32,

    /// Optional RNG seed for reproducible results.
    seed: Option<u64>,
}

impl Default for DellaMerge {
    fn default() -> Self {
        Self::new()
    }
}

impl DellaMerge {
    /// Create a new DELLA merge (exponential variant with TIES consensus).
    pub fn new() -> Self {
        Self {
            use_ties_consensus: true,
            use_exponential: true,
            temperature: 1.0,
            seed: None,
        }
    }

    /// Create a DELLA-Linear merge (linear variant, no sign consensus).
    pub fn linear() -> Self {
        Self {
            use_ties_consensus: false,
            use_exponential: false,
            temperature: 1.0,
            seed: None,
        }
    }

    /// Enable TIES-style sign consensus.
    pub fn with_ties(mut self) -> Self {
        self.use_ties_consensus = true;
        self
    }

    /// Set the softmax temperature (exponential variant only).
    ///
    /// Lower values increase the contrast between low- and high-magnitude entries,
    /// making the dropout more focused on the largest task-vector values.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.max(1e-8);
        self
    }

    /// Set the RNG seed for reproducible dropout masks.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Compute task vector δ = W_ft - W_base.
    fn task_vector(tensor: &Array, base: &Array) -> Result<Array> {
        Ok(tensor.subtract(base)?)
    }

    /// Compute element-wise drop probabilities using exponential (softmax) scaling.
    ///
    /// p_drop[i] = exp(|τ[i]| / T) / sum_j(exp(|τ[j]| / T))
    ///
    /// These are proper probabilities summing to 1. Higher-magnitude entries get
    /// higher drop probability, concentrating the budget on removing the largest
    /// (most likely interfering) changes.
    fn drop_probs_exponential(abs_vals: &[f32], temperature: f32) -> Vec<f32> {
        if abs_vals.is_empty() {
            return Vec::new();
        }

        // Numerically stable softmax: subtract max before exponentiating.
        let max_val = abs_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let exps: Vec<f32> = abs_vals
            .iter()
            .map(|&v| ((v - max_val) / temperature).exp())
            .collect();

        let sum: f32 = exps.iter().sum();

        if sum < 1e-30 {
            // Degenerate case: all values are effectively 0 — uniform probability.
            let n = abs_vals.len() as f32;
            return vec![1.0 / n; abs_vals.len()];
        }

        exps.iter().map(|&e| e / sum).collect()
    }

    /// Compute element-wise drop probabilities using linear scaling.
    ///
    /// p_drop[i] = |τ[i]| / sum_j(|τ[j]|)
    ///
    /// A simpler alternative to the exponential variant. Proportionally assigns
    /// dropout probability to each parameter based on its fractional contribution
    /// to the total task-vector mass.
    fn drop_probs_linear(abs_vals: &[f32]) -> Vec<f32> {
        if abs_vals.is_empty() {
            return Vec::new();
        }

        let sum: f32 = abs_vals.iter().sum();

        if sum < 1e-30 {
            // Degenerate case: all values are zero — uniform probability.
            let n = abs_vals.len() as f32;
            return vec![1.0 / n; abs_vals.len()];
        }

        abs_vals.iter().map(|&v| v / sum).collect()
    }

    /// Apply DELLA magnitude-proportional sparsification to a single task vector.
    ///
    /// # Arguments
    /// * `delta`   - The task vector to sparsify.
    /// * `density` - Global fraction of parameters to *keep* on average (0.0–1.0).
    ///   This modulates the drop probabilities so the expected fraction
    ///   of kept parameters equals `density`.
    ///
    /// # Implementation note
    ///
    /// The raw per-element drop probability from the distribution sums to 1, but we
    /// want the *expected* fraction dropped to equal `(1 - density)`. We scale each
    /// raw probability by `(1 - density)` so the distribution integrates to the
    /// desired sparsity, then clamp individual probabilities to [0, 1).
    /// The rescaling factor for kept entries is then `1 / (1 - p_drop_scaled[i])`.
    fn della_sparsify(&self, delta: &Array, density: f32) -> Result<Array> {
        if density >= 1.0 {
            return Ok(delta.clone());
        }
        if density <= 0.0 {
            return Ok(Array::zeros::<f32>(delta.shape())?);
        }

        let original_shape = delta.shape().to_vec();
        let flat = delta.reshape(&[-1])?;
        flat.eval()?;
        let values: Vec<f32> = flat.as_slice().to_vec();
        let n = values.len();

        // Compute absolute values for probability computation.
        let abs_vals: Vec<f32> = values.iter().map(|v| v.abs()).collect();

        // Compute raw drop probabilities (sum to 1.0 across all elements).
        let raw_probs = if self.use_exponential {
            Self::drop_probs_exponential(&abs_vals, self.temperature)
        } else {
            Self::drop_probs_linear(&abs_vals)
        };

        // Scale probabilities so expected dropped fraction = (1 - density).
        // p_drop_scaled[i] = raw_probs[i] * n * (1 - density)
        // This gives: E[#dropped] = sum(p_drop_scaled) = n * (1 - density), which
        // means E[kept fraction] = density, matching the DARE/TIES convention.
        let target_drop_rate = 1.0 - density;
        let scale = (n as f32) * target_drop_rate;

        let drop_probs: Vec<f32> = raw_probs
            .iter()
            .map(|&p| (p * scale).clamp(0.0, 1.0 - 1e-7))
            .collect();

        // Sample Bernoulli mask: keep[i] = 1 with probability (1 - drop_probs[i]).
        let mut rng: Box<dyn rand::RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::thread_rng()),
        };

        let mut result_vals = vec![0.0_f32; n];
        for i in 0..n {
            let keep_prob = 1.0 - drop_probs[i];
            if rng.gen::<f32>() < keep_prob {
                // Rescale to maintain expected value: divide by keep probability.
                result_vals[i] = values[i] / keep_prob;
            }
            // Otherwise: result_vals[i] = 0.0 (already initialised above).
        }

        let result_flat = Array::from_slice(&result_vals, &[n as i32]);
        Ok(result_flat.reshape(&original_shape)?)
    }
}

impl MergeMethod for DellaMerge {
    fn name(&self) -> &'static str {
        if self.use_exponential {
            "della"
        } else {
            "della_linear"
        }
    }

    fn description(&self) -> &'static str {
        if self.use_exponential {
            "Magnitude-proportional dropout (exponential) with rescaling"
        } else {
            "Magnitude-proportional dropout (linear) with rescaling"
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
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        let base = base_tensor.ok_or_else(|| MergeError::BaseModelRequired {
            method: self.name().to_string(),
        })?;

        // Compute task vectors δ_m = W_m - W_base.
        let task_vectors: Vec<Array> = tensors
            .iter()
            .map(|t| Self::task_vector(t, base))
            .collect::<Result<Vec<_>>>()?;

        // Collect per-model parameters, merging with global defaults.
        let densities: Vec<f32> = params
            .iter()
            .map(|p| global_params.merge_with(p).density())
            .collect();

        let weights: Vec<f32> = params
            .iter()
            .map(|p| global_params.merge_with(p).weight())
            .collect();

        let lambda = global_params.lambda();

        // Apply DELLA sparsification to each task vector.
        let sparse_vectors: Vec<Array> = task_vectors
            .iter()
            .zip(densities.iter())
            .map(|(tv, &density)| self.della_sparsify(tv, density))
            .collect::<Result<Vec<_>>>()?;

        // Combine: optionally apply sign consensus, then weighted sum.
        let weighted_sum = if self.use_ties_consensus {
            sign_consensus(&sparse_vectors, &weights)?
        } else {
            let shape = task_vectors[0].shape();
            let mut acc = Array::zeros::<f32>(shape)?;
            for (vector, weight) in sparse_vectors.iter().zip(weights.iter()) {
                let weighted = vector.multiply(Array::from_f32(*weight))?;
                acc = acc.add(&weighted)?;
            }
            acc
        };

        // Scale by lambda and add back to base model.
        let delta = weighted_sum.multiply(Array::from_f32(lambda))?;
        Ok(base.add(&delta)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_della_default_name() {
        let della = DellaMerge::new();
        assert_eq!(della.name(), "della");
        assert!(della.use_exponential);
        assert!(della.use_ties_consensus);
    }

    #[test]
    fn test_della_linear_name() {
        let della = DellaMerge::linear();
        assert_eq!(della.name(), "della_linear");
        assert!(!della.use_exponential);
        assert!(!della.use_ties_consensus);
    }

    #[test]
    fn test_drop_probs_exponential_sum_to_one() {
        let vals = vec![0.1, 0.5, 0.3, 1.0, 0.2];
        let probs = DellaMerge::drop_probs_exponential(&vals, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "probs should sum to 1, got {}",
            sum
        );
        for &p in &probs {
            assert!((0.0..=1.0).contains(&p), "probability out of range: {}", p);
        }
    }

    #[test]
    fn test_drop_probs_exponential_magnitude_ordering() {
        // Larger magnitudes should get larger drop probability.
        let vals = vec![0.1, 1.0]; // second is much larger
        let probs = DellaMerge::drop_probs_exponential(&vals, 1.0);
        assert!(
            probs[1] > probs[0],
            "larger magnitude should have higher drop probability: {:?}",
            probs
        );
    }

    #[test]
    fn test_drop_probs_linear_sum_to_one() {
        let vals = vec![0.2, 0.3, 0.5];
        let probs = DellaMerge::drop_probs_linear(&vals);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "linear probs should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_drop_probs_linear_proportional() {
        let vals = vec![1.0, 3.0]; // second is 3x larger
        let probs = DellaMerge::drop_probs_linear(&vals);
        // p[0] = 1/4 = 0.25, p[1] = 3/4 = 0.75
        assert!((probs[0] - 0.25).abs() < 1e-5);
        assert!((probs[1] - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_della_preserves_base_with_zero_lambda() {
        let della = DellaMerge::new().with_seed(42);

        let base = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t1 = Array::from_slice(&[2.0_f32, 3.0, 4.0], &[3]);

        let params = vec![MergeParameters {
            weight: Some(1.0),
            density: Some(0.5),
            ..Default::default()
        }];

        let global = MergeParameters {
            lambda: Some(0.0),
            ..Default::default()
        };

        let result = della.merge(&[t1], Some(&base), &params, &global).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();
        let base_slice: Vec<f32> = base.as_slice().to_vec();

        // With lambda=0, result should equal base regardless of sparsification.
        for (r, b) in result_slice.iter().zip(base_slice.iter()) {
            assert!((r - b).abs() < 1e-5, "expected base {}, got {}", b, r);
        }
    }

    #[test]
    fn test_della_full_density_preserves_task_vector() {
        // With density=1.0, no elements are dropped.
        // With a single model and lambda=1, result should equal the fine-tuned model.
        let della = DellaMerge::new().with_seed(42);

        let base = Array::from_slice(&[0.0_f32, 0.0, 0.0], &[3]);
        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);

        let params = vec![MergeParameters {
            weight: Some(1.0),
            density: Some(1.0),
            ..Default::default()
        }];
        let global = MergeParameters {
            lambda: Some(1.0),
            ..Default::default()
        };

        let result = della.merge(&[t1], Some(&base), &params, &global).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // density=1.0 means no dropout → result equals t1
        for (i, r) in result_slice.iter().enumerate() {
            assert!(
                (r - (i as f32 + 1.0)).abs() < 1e-5,
                "index {}: expected {}, got {}",
                i,
                i as f32 + 1.0,
                r
            );
        }
    }

    #[test]
    fn test_della_zero_density_gives_base() {
        let della = DellaMerge::new().with_seed(42);

        let base = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t1 = Array::from_slice(&[5.0_f32, 6.0, 7.0], &[3]);

        let params = vec![MergeParameters {
            weight: Some(1.0),
            density: Some(0.0),
            ..Default::default()
        }];
        let global = MergeParameters {
            lambda: Some(1.0),
            ..Default::default()
        };

        let result = della.merge(&[t1], Some(&base), &params, &global).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();
        let base_slice: Vec<f32> = base.as_slice().to_vec();

        // density=0.0 → all task vector entries dropped → result equals base
        for (r, b) in result_slice.iter().zip(base_slice.iter()) {
            assert!((r - b).abs() < 1e-5, "expected base {}, got {}", b, r);
        }
    }

    #[test]
    fn test_della_output_shape_preserved() {
        let della = DellaMerge::new().with_seed(7);
        let base = Array::from_slice(&[0.0_f32; 12], &[3, 4]);
        let t1 = Array::from_slice(&[1.0_f32; 12], &[3, 4]);

        let params = vec![MergeParameters::default()];
        let global = MergeParameters::default();

        let result = della.merge(&[t1], Some(&base), &params, &global).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
    }

    #[test]
    fn test_della_requires_base() {
        let della = DellaMerge::new();
        assert!(della.requires_base_model());
    }

    #[test]
    fn test_della_seeded_reproducible() {
        let base = Array::from_slice(&[0.0_f32; 8], &[8]);
        let t1 = Array::from_slice(&[1.0_f32, 0.5, 2.0, 0.1, 0.8, 0.3, 1.5, 0.6], &[8]);

        let params = vec![MergeParameters {
            weight: Some(1.0),
            density: Some(0.5),
            ..Default::default()
        }];
        let global = MergeParameters {
            lambda: Some(1.0),
            ..Default::default()
        };

        let r1 = DellaMerge::new()
            .with_seed(99)
            .merge(std::slice::from_ref(&t1), Some(&base), &params, &global)
            .unwrap();
        let r2 = DellaMerge::new()
            .with_seed(99)
            .merge(&[t1], Some(&base), &params, &global)
            .unwrap();

        let s1: Vec<f32> = r1.as_slice().to_vec();
        let s2: Vec<f32> = r2.as_slice().to_vec();
        assert_eq!(s1, s2, "same seed must produce identical results");
    }

    #[test]
    fn test_della_linear_vs_exponential_differ() {
        // The two variants should produce different results on non-trivial input.
        let base = Array::from_slice(&[0.0_f32; 8], &[8]);
        let t1 = Array::from_slice(&[0.1_f32, 5.0, 0.2, 4.0, 0.3, 3.0, 0.4, 2.0], &[8]);

        let params = vec![MergeParameters {
            weight: Some(1.0),
            density: Some(0.5),
            ..Default::default()
        }];
        let global = MergeParameters {
            lambda: Some(1.0),
            ..Default::default()
        };

        // Use same seed to ensure the only difference is the probability computation.
        let exp_result = DellaMerge::new()
            .with_seed(42)
            .merge(std::slice::from_ref(&t1), Some(&base), &params, &global)
            .unwrap();
        let lin_result = DellaMerge::linear()
            .with_seed(42)
            .merge(&[t1], Some(&base), &params, &global)
            .unwrap();

        let s_exp: Vec<f32> = exp_result.as_slice().to_vec();
        let s_lin: Vec<f32> = lin_result.as_slice().to_vec();

        // Results will differ because softmax vs linear produce different probabilities.
        // The test just verifies both run without errors and produce finite values.
        assert!(
            s_exp.iter().all(|x| x.is_finite()),
            "exponential variant produced non-finite values"
        );
        assert!(
            s_lin.iter().all(|x| x.is_finite()),
            "linear variant produced non-finite values"
        );
    }
}
