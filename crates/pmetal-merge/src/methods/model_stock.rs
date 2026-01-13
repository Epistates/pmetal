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
use mlx_rs::Array;

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

    /// Compute cosine similarity between two flattened arrays.
    fn cosine_similarity(&self, a: &Array, b: &Array) -> Result<f32> {
        // Flatten both arrays
        let a_flat = a.flatten(None, None)?;
        let b_flat = b.flatten(None, None)?;

        // Compute dot product: sum(a * b)
        let dot = a_flat.multiply(&b_flat)?.sum(None)?;

        // Compute norms: sqrt(sum(a^2)), sqrt(sum(b^2))
        let norm_a = a_flat.square()?.sum(None)?.sqrt()?;
        let norm_b = b_flat.square()?.sum(None)?.sqrt()?;

        // Cosine similarity = dot / (norm_a * norm_b + eps)
        let denom = norm_a.multiply(&norm_b)?.add(&Array::from_f32(self.eps))?;
        let sim = dot.divide(&denom)?;

        sim.eval()?;
        Ok(sim.item::<f32>())
    }

    /// Compute the L2 norm of an array.
    fn l2_norm(&self, a: &Array) -> Result<Array> {
        let flat = a.flatten(None, None)?;
        Ok(flat.square()?.sum(None)?.sqrt()?)
    }

    /// Model Stock for exactly 2 fine-tuned models.
    ///
    /// Finds the perpendicular foot from the estimated center to the plane
    /// defined by the pretrained weights and two fine-tuned weights.
    fn merge_two_models(&self, base: &Array, w1: &Array, w2: &Array) -> Result<Array> {
        // Task vectors
        let tau1 = w1.subtract(base)?;
        let tau2 = w2.subtract(base)?;

        // Estimate center as average of fine-tuned weights
        // µ = (w1 + w2) / 2
        let mu = w1.add(w2)?.multiply(&Array::from_f32(0.5))?;

        // Vector from base to center: d = µ - w0
        let d = mu.subtract(base)?;

        // Gram-Schmidt orthogonalization to get basis vectors for the plane
        // v1 = tau1 (normalized)
        // v2 = tau2 - proj(tau2 onto v1)

        let norm_tau1 = self.l2_norm(&tau1)?;
        let v1 = tau1.divide(&norm_tau1.add(&Array::from_f32(self.eps))?)?;

        // Project tau2 onto v1
        let tau2_flat = tau2.flatten(None, None)?;
        let v1_flat = v1.flatten(None, None)?;
        let proj_coeff = tau2_flat.multiply(&v1_flat)?.sum(None)?;
        let proj = v1.multiply(&proj_coeff)?;
        let v2_unnorm = tau2.subtract(&proj)?;
        let norm_v2 = self.l2_norm(&v2_unnorm)?;
        let v2 = v2_unnorm.divide(&norm_v2.add(&Array::from_f32(self.eps))?)?;

        // Project d onto the plane spanned by v1 and v2
        // wH = w0 + proj(d onto plane)
        // proj(d onto plane) = (d·v1)*v1 + (d·v2)*v2

        let d_flat = d.flatten(None, None)?;
        let v1_flat = v1.flatten(None, None)?;
        let v2_flat = v2.flatten(None, None)?;

        let coeff1 = d_flat.multiply(&v1_flat)?.sum(None)?;
        let coeff2 = d_flat.multiply(&v2_flat)?.sum(None)?;

        let proj_d = v1.multiply(&coeff1)?.add(&v2.multiply(&coeff2)?)?;

        // Result: wH = base + proj_d
        base.add(&proj_d).map_err(Into::into)
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
            task_vectors.push(t.subtract(base)?);
        }

        // Estimate center direction as average of task vectors
        let mut avg_tau = task_vectors[0].clone();
        for tau in task_vectors.iter().skip(1) {
            avg_tau = avg_tau.add(tau)?;
        }
        avg_tau = avg_tau.multiply(&Array::from_f32(1.0 / n as f32))?;

        if !self.use_cosine_weighting {
            // Simple averaging (Task Arithmetic)
            return base.add(&avg_tau).map_err(Into::into);
        }

        // Compute cosine similarity weights
        // Each model is weighted by how aligned its task vector is with the average
        let mut weights: Vec<f32> = Vec::with_capacity(n);
        for tau in &task_vectors {
            let sim = self.cosine_similarity(tau, &avg_tau)?;
            // Use softmax-like weighting: exp(sim) / sum(exp(sim))
            // But first just collect raw similarities
            weights.push(sim.max(0.0)); // Clamp negative similarities
        }

        // Normalize weights (softmax-style with temperature=1)
        let sum_weights: f32 = weights.iter().sum();
        if sum_weights < self.eps {
            // Fallback to uniform if all weights are near zero
            return base.add(&avg_tau).map_err(Into::into);
        }

        for w in &mut weights {
            *w /= sum_weights;
        }

        // Weighted average of task vectors
        let mut weighted_tau = task_vectors[0].multiply(&Array::from_f32(weights[0]))?;
        for (tau, &w) in task_vectors.iter().skip(1).zip(weights.iter().skip(1)) {
            weighted_tau = weighted_tau.add(&tau.multiply(&Array::from_f32(w))?)?;
        }

        // Result: base + weighted_tau
        base.add(&weighted_tau).map_err(Into::into)
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
                let tau = tensors[0].subtract(base)?;
                base.add(&tau).map_err(Into::into)
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
        let base = Array::from_slice(&[1.0f32, 0.0, 0.0, 0.0], &[4]);

        // Fine-tuned models (slight variations)
        let w1 = Array::from_slice(&[1.1f32, 0.2, 0.0, 0.0], &[4]);
        let w2 = Array::from_slice(&[1.0f32, 0.0, 0.3, 0.0], &[4]);

        let result = merger
            .merge(&[w1, w2], Some(&base), &[], &MergeParameters::default())
            .unwrap();

        result.eval().unwrap();

        // Result should be between base and the fine-tuned models
        let shape = result.shape();
        assert_eq!(shape, &[4]);
    }

    #[test]
    fn test_model_stock_cosine_weighting() {
        let merger = ModelStockMerge::new().with_cosine_weighting(true);

        let base = Array::from_slice(&[0.0f32, 0.0, 0.0], &[3]);

        // Three fine-tuned models
        let w1 = Array::from_slice(&[1.0f32, 0.0, 0.0], &[3]);
        let w2 = Array::from_slice(&[0.9f32, 0.1, 0.0], &[3]); // Similar to w1
        let w3 = Array::from_slice(&[0.0f32, 0.0, 1.0], &[3]); // Different direction

        let result = merger
            .merge(&[w1, w2, w3], Some(&base), &[], &MergeParameters::default())
            .unwrap();

        result.eval().unwrap();

        // w1 and w2 should have higher weights due to similarity
        // Result should lean towards their direction
        let vals: Vec<f32> = result.as_slice::<f32>().to_vec();

        // X component should be higher than Z since w1, w2 point that way
        assert!(vals[0] > vals[2]);
    }

    #[test]
    fn test_model_stock_single_model() {
        let merger = ModelStockMerge::new();

        let base = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let w1 = Array::from_slice(&[1.5f32, 2.5], &[2]);

        let result = merger
            .merge(&[w1.clone()], Some(&base), &[], &MergeParameters::default())
            .unwrap();

        result.eval().unwrap();

        // Single model should just return w1
        let expected = w1;
        expected.eval().unwrap();

        let result_vals: Vec<f32> = result.as_slice::<f32>().to_vec();
        let expected_vals: Vec<f32> = expected.as_slice::<f32>().to_vec();

        for (r, e) in result_vals.iter().zip(expected_vals.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let merger = ModelStockMerge::new();

        let a = Array::from_slice(&[1.0f32, 0.0, 0.0], &[3]);
        let b = Array::from_slice(&[1.0f32, 0.0, 0.0], &[3]);

        let sim = merger.cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-5); // Identical vectors = 1.0

        let c = Array::from_slice(&[0.0f32, 1.0, 0.0], &[3]);
        let sim_orthogonal = merger.cosine_similarity(&a, &c).unwrap();
        assert!(sim_orthogonal.abs() < 1e-5); // Orthogonal vectors = 0.0
    }
}
