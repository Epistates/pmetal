//! Model Stock merge method.
//!
//! Implements the Model Stock algorithm from "Model Stock: All we need is just
//! a few fine-tuned models" (Jang et al., ECCV 2024).
//!
//! Key insight: Fine-tuned weights lie on a thin shell (sphere) centered around
//! a "central" point µ. The algorithm finds the perpendicular foot from µ to
//! the plane defined by the pre-trained weights and fine-tuned models.
//!
//! For 2 fine-tuned models:
//! 1. Define plane using w0 (pretrained), w1, w2
//! 2. Estimate center µ as average of fine-tuned weights
//! 3. Find perpendicular foot wH from µ to the plane
//!
//! For N > 2 models, we use iterative geometric averaging with cosine
//! similarity weighting to approximate the center.

use crate::{MergeError, MergeMethod, MergeParameters, Result};
use pmetal_runtime::compat::Array;

/// Model Stock merge method.
///
/// Implements geometric interpolation of task vectors using the Model Stock
/// algorithm. Achieves model soup performance with just 2-3 fine-tuned models.
#[derive(Debug, Clone)]
pub struct ModelStockMerge {
    /// Whether to use cosine similarity weighting for N > 2 models.
    pub use_cosine_weighting: bool,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

impl Default for ModelStockMerge {
    fn default() -> Self {
        Self {
            use_cosine_weighting: true,
            eps: 1e-8,
        }
    }
}

impl ModelStockMerge {
    /// Create a new Model Stock merger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom settings.
    pub fn with_cosine_weighting(mut self, enabled: bool) -> Self {
        self.use_cosine_weighting = enabled;
        self
    }

    /// Compute cosine similarity between two arrays (CPU path).
    fn cosine_similarity(&self, a: &Array, b: &Array) -> Result<f32> {
        let n = a.shape().iter().map(|&s| s as usize).product::<usize>();
        let mut a_clone = a.clone();
        let mut b_clone = b.clone();
        let a_flat = a_clone.to_f32_vec(n).unwrap_or_default();
        let b_flat = b_clone.to_f32_vec(n).unwrap_or_default();

        let dot: f32 = a_flat.iter().zip(b_flat.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a_flat.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b_flat.iter().map(|x| x * x).sum::<f32>().sqrt();

        let denom = norm_a * norm_b + self.eps;
        Ok(dot / denom)
    }

    /// Compute the L2 norm scalar of an array (CPU path).
    fn l2_norm_scalar(&self, a: &Array) -> f32 {
        let n = a.shape().iter().map(|&s| s as usize).product::<usize>();
        let mut a_clone = a.clone();
        let vals = a_clone.to_f32_vec(n).unwrap_or_default();
        vals.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Model Stock for exactly 2 fine-tuned models.
    ///
    /// Finds the perpendicular foot from the estimated center to the plane
    /// defined by the pretrained weights and two fine-tuned weights.
    fn merge_two_models(&self, base: &Array, w1: &Array, w2: &Array) -> Result<Array> {
        let n = base.shape().iter().map(|&s| s as usize).product::<usize>();

        // Task vectors
        let tau1 = w1.subtract(base);
        let tau2 = w2.subtract(base);

        // Estimate center as average of fine-tuned weights
        // µ = (w1 + w2) / 2
        let mu = w1.add(w2).multiply(&Array::from_f32(0.5));

        // Vector from base to center: d = µ - w0
        let d = mu.subtract(base);

        // Gram-Schmidt orthogonalization to get basis vectors for the plane
        // v1 = tau1 (normalized)
        let norm_tau1 = self.l2_norm_scalar(&tau1);
        let v1 = tau1.divide(&Array::from_f32(norm_tau1 + self.eps));

        // Project tau2 onto v1 (CPU dot product)
        let mut tau2_clone = tau2.clone();
        let tau2_vals = tau2_clone.to_f32_vec(n).unwrap_or_default();
        let mut v1_clone = v1.clone();
        let v1_vals = v1_clone.to_f32_vec(n).unwrap_or_default();
        let proj_coeff: f32 = tau2_vals
            .iter()
            .zip(v1_vals.iter())
            .map(|(a, b)| a * b)
            .sum();

        let proj = v1.multiply(&Array::from_f32(proj_coeff));
        let v2_unnorm = tau2.subtract(&proj);
        let norm_v2 = self.l2_norm_scalar(&v2_unnorm);
        let v2 = v2_unnorm.divide(&Array::from_f32(norm_v2 + self.eps));

        // Project d onto the plane spanned by v1 and v2
        // wH = w0 + proj(d onto plane)
        // proj(d onto plane) = (d·v1)*v1 + (d·v2)*v2
        let mut d_clone = d.clone();
        let d_vals = d_clone.to_f32_vec(n).unwrap_or_default();
        let mut v1_clone2 = v1.clone();
        let v1_vals2 = v1_clone2.to_f32_vec(n).unwrap_or_default();
        let mut v2_clone = v2.clone();
        let v2_vals = v2_clone.to_f32_vec(n).unwrap_or_default();

        let coeff1: f32 = d_vals.iter().zip(v1_vals2.iter()).map(|(a, b)| a * b).sum();
        let coeff2: f32 = d_vals.iter().zip(v2_vals.iter()).map(|(a, b)| a * b).sum();

        let proj_d = v1
            .multiply(&Array::from_f32(coeff1))
            .add(&v2.multiply(&Array::from_f32(coeff2)));

        // Result: wH = base + proj_d
        Ok(base.add(&proj_d))
    }

    /// Model Stock for N > 2 fine-tuned models using cosine similarity weighting.
    ///
    /// Uses an iterative approach that weights each model's contribution
    /// by its cosine similarity to the estimated center direction.
    fn merge_n_models(&self, base: &Array, tensors: &[Array]) -> Result<Array> {
        let n = tensors.len();

        // Compute task vectors
        let mut task_vectors: Vec<Array> = Vec::with_capacity(n);
        for t in tensors {
            task_vectors.push(t.subtract(base));
        }

        // Estimate center direction as average of task vectors
        let mut avg_tau = task_vectors[0].clone();
        for tau in task_vectors.iter().skip(1) {
            avg_tau = avg_tau.add(tau);
        }
        avg_tau = avg_tau.multiply(&Array::from_f32(1.0 / n as f32));

        if !self.use_cosine_weighting {
            // Simple averaging (Task Arithmetic)
            return Ok(base.add(&avg_tau));
        }

        // Compute softmax cosine similarity weights per the Model Stock paper.
        // w_i = exp(sim_i) / sum_j( exp(sim_j) )
        // This is strictly positive for all inputs and does not discard negative
        // cosine similarities (which can arise for out-of-distribution models).
        let mut raw_sims: Vec<f32> = Vec::with_capacity(n);
        for tau in &task_vectors {
            let sim = self.cosine_similarity(tau, &avg_tau)?;
            raw_sims.push(sim);
        }

        // Numerically stable softmax: subtract max before exponentiation.
        let max_sim = raw_sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sims: Vec<f32> = raw_sims.iter().map(|&s| (s - max_sim).exp()).collect();
        let sum_exp: f32 = exp_sims.iter().sum();

        let weights: Vec<f32> = if sum_exp < self.eps {
            // Degenerate case: fall back to uniform weights.
            vec![1.0 / n as f32; n]
        } else {
            exp_sims.iter().map(|&e| e / sum_exp).collect()
        };

        // Weighted average of task vectors
        let mut weighted_tau = task_vectors[0].multiply(&Array::from_f32(weights[0]));
        for (tau, &w) in task_vectors.iter().skip(1).zip(weights.iter().skip(1)) {
            weighted_tau = weighted_tau.add(&tau.multiply(&Array::from_f32(w)));
        }

        // Result: base + weighted_tau
        Ok(base.add(&weighted_tau))
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

        match tensors.len() {
            0 => Ok(base.clone()),
            1 => {
                // Single model: simple task vector addition
                let tau = tensors[0].subtract(base);
                Ok(base.add(&tau))
            }
            2 => {
                // Optimal case: use perpendicular foot projection
                self.merge_two_models(base, &tensors[0], &tensors[1])
            }
            _ => {
                // N > 2: use cosine similarity weighted averaging
                self.merge_n_models(base, tensors)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_stock_two_models() {
        let merger = ModelStockMerge::new();

        // Base model weights
        let base = Array::from_f32_slice(&[1.0f32, 0.0, 0.0, 0.0], &[4]);

        // Fine-tuned models (slight variations)
        let w1 = Array::from_f32_slice(&[1.1f32, 0.2, 0.0, 0.0], &[4]);
        let w2 = Array::from_f32_slice(&[1.0f32, 0.0, 0.3, 0.0], &[4]);

        let result = merger
            .merge(&[w1, w2], Some(&base), &[], &MergeParameters::default())
            .unwrap();

        // Result should be between base and the fine-tuned models
        let shape = result.shape();
        assert_eq!(shape, &[4]);
    }

    #[test]
    fn test_model_stock_cosine_weighting() {
        let merger = ModelStockMerge::new().with_cosine_weighting(true);

        let base = Array::from_f32_slice(&[0.0f32, 0.0, 0.0], &[3]);

        // Three fine-tuned models
        let w1 = Array::from_f32_slice(&[1.0f32, 0.0, 0.0], &[3]);
        let w2 = Array::from_f32_slice(&[0.9f32, 0.1, 0.0], &[3]); // Similar to w1
        let w3 = Array::from_f32_slice(&[0.0f32, 0.0, 1.0], &[3]); // Different direction

        let mut result = merger
            .merge(&[w1, w2, w3], Some(&base), &[], &MergeParameters::default())
            .unwrap();

        // w1 and w2 should have higher weights due to similarity
        // Result should lean towards their direction
        let vals = result.to_f32_vec(3).unwrap();

        // X component should be higher than Z since w1, w2 point that way
        assert!(vals[0] > vals[2]);
    }

    #[test]
    fn test_model_stock_single_model() {
        let merger = ModelStockMerge::new();

        let base = Array::from_f32_slice(&[1.0f32, 2.0], &[2]);
        let w1 = Array::from_f32_slice(&[1.5f32, 2.5], &[2]);

        let mut result = merger
            .merge(
                std::slice::from_ref(&w1),
                Some(&base),
                &[],
                &MergeParameters::default(),
            )
            .unwrap();

        // Single model should just return w1
        let mut w1_clone = Array::from_f32_slice(&[1.5f32, 2.5], &[2]);

        let result_vals = result.to_f32_vec(2).unwrap();
        let expected_vals = w1_clone.to_f32_vec(2).unwrap();

        for (r, e) in result_vals.iter().zip(expected_vals.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let merger = ModelStockMerge::new();

        let a = Array::from_f32_slice(&[1.0f32, 0.0, 0.0], &[3]);
        let b = Array::from_f32_slice(&[1.0f32, 0.0, 0.0], &[3]);

        let sim = merger.cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-5); // Identical vectors = 1.0

        let c = Array::from_f32_slice(&[0.0f32, 1.0, 0.0], &[3]);
        let sim_orthogonal = merger.cosine_similarity(&a, &c).unwrap();
        assert!(sim_orthogonal.abs() < 1e-5); // Orthogonal vectors = 0.0
    }
}
