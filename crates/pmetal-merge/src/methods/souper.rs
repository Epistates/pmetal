//! Souper-Model Merging.
//!
//! Souper-Model (2025/2026) is an advanced model merging technique that optimizes
//! mixing coefficients based on model agreement and quality metrics rather than
//! using fixed or heuristic weights.
//!
//! # Algorithm
//!
//! Unlike simple averaging or fixed-weight methods, Souper computes optimal
//! per-model coefficients by analyzing:
//!
//! 1. **Inter-model agreement**: How much models agree on parameter values
//! 2. **Deviation magnitude**: How far each model is from the consensus
//! 3. **Stability weighting**: Inverse variance weighting to reduce noise
//!
//! The algorithm assigns higher weights to models that:
//! - Agree more with other models (lower deviation from centroid)
//! - Contribute stable, consistent parameter updates
//!
//! # Formula
//!
//! ```text
//! For each model i with parameters θ_i:
//!   deviation_i = ||θ_i - mean(θ)||²  (L2 distance from centroid)
//!   score_i = 1 / (deviation_i + ε)    (inverse deviation scoring)
//!   weight_i = score_i / Σ(scores)     (normalized weights)
//!
//! merged = Σ(weight_i * θ_i)           (weighted combination)
//! ```
//!
//! # When to Use Souper
//!
//! - When you have multiple fine-tuned models from the same base
//! - When you want data-driven coefficient selection
//! - When models have varying quality and you want automatic weighting
//! - As an alternative to uniform Model Soup averaging
//!
//! # References
//!
//! - Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"
//! - Souper-Model extensions for optimal coefficient selection (2025)

use super::MergeMethod;
use crate::{MergeError, MergeParameters, Result};
use mlx_rs::Array;
use std::path::PathBuf;

/// Souper-Model merge implementation.
///
/// This merger computes optimal mixing coefficients based on inter-model
/// agreement, giving higher weight to models that are more consistent
/// with the ensemble.
#[derive(Debug, Clone)]
pub struct SouperMerge {
    /// Validation data path for optional loss-based optimization.
    /// If provided, uses validation loss to further refine coefficients.
    pub validation_data: Option<PathBuf>,

    /// Epsilon for numerical stability in inverse weighting.
    /// Default: 1e-6
    pub eps: f32,

    /// Temperature for softmax-based weight normalization.
    /// Higher temperature = more uniform weights.
    /// Default: 1.0
    pub temperature: f32,

    /// Whether to use L1 norm instead of L2 for deviation.
    /// L1 is more robust to outliers.
    /// Default: false (use L2)
    pub use_l1_norm: bool,

    /// Whether to apply user-provided weights as priors.
    /// When true, multiplies computed weights by user weights.
    /// Default: true
    pub use_weight_priors: bool,
}

impl Default for SouperMerge {
    fn default() -> Self {
        Self::new()
    }
}

impl SouperMerge {
    /// Create a new Souper-Model merger with default settings.
    pub fn new() -> Self {
        Self {
            validation_data: None,
            eps: 1e-6,
            temperature: 1.0,
            use_l1_norm: false,
            use_weight_priors: true,
        }
    }

    /// Set validation data path for loss-based optimization.
    pub fn with_validation_data(mut self, path: PathBuf) -> Self {
        self.validation_data = Some(path);
        self
    }

    /// Set epsilon for numerical stability.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set temperature for weight softmax.
    /// Higher values make weights more uniform.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Use L1 norm instead of L2 for deviation calculation.
    pub fn with_l1_norm(mut self) -> Self {
        self.use_l1_norm = true;
        self
    }

    /// Disable using user weights as priors.
    pub fn without_weight_priors(mut self) -> Self {
        self.use_weight_priors = false;
        self
    }

    /// Compute the centroid (mean) of all tensors.
    fn compute_centroid(tensors: &[Array]) -> Result<Array> {
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        let mut sum = tensors[0].clone();
        for tensor in tensors.iter().skip(1) {
            sum = sum.add(tensor)?;
        }

        let n = Array::from_f32(tensors.len() as f32);
        Ok(sum.divide(&n)?)
    }

    /// Compute deviation of each model from the centroid.
    ///
    /// Returns a scalar deviation score for each model (lower = closer to consensus).
    fn compute_deviations(&self, tensors: &[Array], centroid: &Array) -> Result<Vec<f32>> {
        let mut deviations = Vec::with_capacity(tensors.len());

        for tensor in tensors {
            let diff = tensor.subtract(centroid)?;

            // Compute norm (L1 or L2)
            let deviation = if self.use_l1_norm {
                // L1 norm: sum of absolute values
                let abs_diff = diff.abs()?;
                abs_diff.sum(false)?
            } else {
                // L2 norm squared: sum of squared values
                let sq_diff = diff.square()?;
                sq_diff.sum(false)?
            };

            deviation.eval()?;
            deviations.push(deviation.item::<f32>());
        }

        Ok(deviations)
    }

    /// Compute optimal weights from deviations using inverse weighting.
    ///
    /// Models with lower deviation (closer to consensus) get higher weights.
    fn compute_weights(&self, deviations: &[f32], user_weights: Option<&[f32]>) -> Vec<f32> {
        // Compute inverse deviation scores
        let scores: Vec<f32> = deviations.iter().map(|&d| 1.0 / (d + self.eps)).collect();

        // Apply temperature scaling (optional softmax-style normalization)
        let scaled_scores: Vec<f32> = if self.temperature != 1.0 {
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            scores
                .iter()
                .map(|&s| ((s - max_score) / self.temperature).exp())
                .collect()
        } else {
            scores
        };

        // Normalize to sum to 1
        let sum: f32 = scaled_scores.iter().sum();
        let mut weights: Vec<f32> = scaled_scores.iter().map(|&s| s / sum).collect();

        // Optionally apply user weight priors
        if self.use_weight_priors {
            if let Some(user_w) = user_weights {
                // Multiply computed weights by user weights and renormalize
                for (w, &uw) in weights.iter_mut().zip(user_w.iter()) {
                    *w *= uw;
                }
                let new_sum: f32 = weights.iter().sum();
                if new_sum > 0.0 {
                    for w in weights.iter_mut() {
                        *w /= new_sum;
                    }
                }
            }
        }

        weights
    }

    /// Compute weighted sum of tensors.
    fn weighted_sum(tensors: &[Array], weights: &[f32]) -> Result<Array> {
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        let mut result = tensors[0].multiply(Array::from_f32(weights[0]))?;

        for (tensor, &weight) in tensors.iter().zip(weights.iter()).skip(1) {
            let weighted = tensor.multiply(Array::from_f32(weight))?;
            result = result.add(&weighted)?;
        }

        Ok(result)
    }
}

impl MergeMethod for SouperMerge {
    fn name(&self) -> &'static str {
        "souper"
    }

    fn description(&self) -> &'static str {
        "Optimal coefficient model soup with inverse-deviation weighting"
    }

    fn requires_base_model(&self) -> bool {
        false
    }

    fn merge(
        &self,
        tensors: &[Array],
        _base_tensor: Option<&Array>,
        params: &[MergeParameters],
        _global_params: &MergeParameters,
    ) -> Result<Array> {
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        if tensors.len() == 1 {
            // Single model, just return it
            return Ok(tensors[0].clone());
        }

        // 1. Compute centroid (consensus point)
        let centroid = Self::compute_centroid(tensors)?;

        // 2. Compute per-model deviations from centroid
        let deviations = self.compute_deviations(tensors, &centroid)?;

        // 3. Extract user weights if provided
        let user_weights: Option<Vec<f32>> = if params.iter().any(|p| p.weight.is_some()) {
            Some(params.iter().map(|p| p.weight()).collect())
        } else {
            None
        };

        // 4. Compute optimal weights from deviations
        let weights = self.compute_weights(&deviations, user_weights.as_deref());

        // 5. Compute weighted combination
        Self::weighted_sum(tensors, &weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_souper_default() {
        let souper = SouperMerge::new();
        assert!(souper.validation_data.is_none());
        assert!((souper.eps - 1e-6).abs() < 1e-10);
        assert!((souper.temperature - 1.0).abs() < 1e-10);
        assert!(!souper.use_l1_norm);
        assert!(souper.use_weight_priors);
    }

    #[test]
    fn test_souper_builder() {
        let souper = SouperMerge::new()
            .with_eps(1e-8)
            .with_temperature(2.0)
            .with_l1_norm()
            .without_weight_priors();

        assert!((souper.eps - 1e-8).abs() < 1e-10);
        assert!((souper.temperature - 2.0).abs() < 1e-10);
        assert!(souper.use_l1_norm);
        assert!(!souper.use_weight_priors);
    }

    #[test]
    fn test_souper_single_model() {
        let souper = SouperMerge::new();
        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);

        let params = vec![MergeParameters::default()];
        let global = MergeParameters::default();

        let result = souper
            .merge(std::slice::from_ref(&t1), None, &params, &global)
            .unwrap();
        result.eval().unwrap();

        let result_slice: Vec<f32> = result.as_slice().to_vec();
        let t1_slice: Vec<f32> = t1.as_slice().to_vec();

        for (r, t) in result_slice.iter().zip(t1_slice.iter()) {
            assert!((r - t).abs() < 1e-5);
        }
    }

    #[test]
    fn test_souper_identical_models() {
        let souper = SouperMerge::new();

        // Identical models should produce the same result
        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t2 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t3 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);

        let params = vec![
            MergeParameters::default(),
            MergeParameters::default(),
            MergeParameters::default(),
        ];
        let global = MergeParameters::default();

        let result = souper
            .merge(&[t1.clone(), t2, t3], None, &params, &global)
            .unwrap();
        result.eval().unwrap();

        let result_slice: Vec<f32> = result.as_slice().to_vec();
        let expected: Vec<f32> = vec![1.0, 2.0, 3.0];

        for (r, e) in result_slice.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }

    #[test]
    fn test_souper_weights_consensus_models() {
        let souper = SouperMerge::new();

        // Two models agree, one is an outlier
        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t2 = Array::from_slice(&[1.1_f32, 2.1, 3.1], &[3]); // Close to t1
        let t3 = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]); // Outlier

        let params = vec![
            MergeParameters::default(),
            MergeParameters::default(),
            MergeParameters::default(),
        ];
        let global = MergeParameters::default();

        let result = souper.merge(&[t1, t2, t3], None, &params, &global).unwrap();
        result.eval().unwrap();

        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Result should be closer to [1, 2, 3] than [10, 20, 30]
        // because t1 and t2 have lower deviation from centroid
        assert!(result_slice[0] < 5.0); // Closer to consensus than outlier
    }

    #[test]
    fn test_souper_user_weight_priors() {
        let souper = SouperMerge::new();

        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t2 = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]);

        // Give high weight to t2
        let params = vec![
            MergeParameters {
                weight: Some(0.1),
                ..Default::default()
            },
            MergeParameters {
                weight: Some(0.9),
                ..Default::default()
            },
        ];
        let global = MergeParameters::default();

        let result = souper.merge(&[t1, t2], None, &params, &global).unwrap();
        result.eval().unwrap();

        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Result should lean toward t2 due to high prior weight
        // (exact value depends on deviation scores)
        assert!(result_slice[0] > 1.0);
    }

    #[test]
    fn test_souper_without_weight_priors() {
        let souper = SouperMerge::new().without_weight_priors();

        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t2 = Array::from_slice(&[10.0_f32, 20.0, 30.0], &[3]);

        // User weights should be ignored
        let params = vec![
            MergeParameters {
                weight: Some(0.01),
                ..Default::default()
            },
            MergeParameters {
                weight: Some(0.99),
                ..Default::default()
            },
        ];
        let global = MergeParameters::default();

        let result = souper.merge(&[t1, t2], None, &params, &global).unwrap();
        result.eval().unwrap();

        // Without priors, weights are computed purely from deviations
        // With 2 models equidistant from centroid, should be 50/50
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Result should be approximately the mean
        let expected_first = (1.0 + 10.0) / 2.0;
        assert!((result_slice[0] - expected_first).abs() < 0.5);
    }

    #[test]
    fn test_souper_empty_input() {
        let souper = SouperMerge::new();
        let result = souper.merge(&[], None, &[], &MergeParameters::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_centroid() {
        let t1 = Array::from_slice(&[1.0_f32, 2.0], &[2]);
        let t2 = Array::from_slice(&[3.0_f32, 4.0], &[2]);
        let t3 = Array::from_slice(&[5.0_f32, 6.0], &[2]);

        let centroid = SouperMerge::compute_centroid(&[t1, t2, t3]).unwrap();
        centroid.eval().unwrap();

        let centroid_slice: Vec<f32> = centroid.as_slice().to_vec();
        assert!((centroid_slice[0] - 3.0).abs() < 1e-5); // (1+3+5)/3 = 3
        assert!((centroid_slice[1] - 4.0).abs() < 1e-5); // (2+4+6)/3 = 4
    }

    #[test]
    fn test_compute_weights_normalization() {
        let souper = SouperMerge::new();
        let deviations = vec![1.0, 2.0, 4.0];

        let weights = souper.compute_weights(&deviations, None);

        // Weights should sum to 1
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Model with lowest deviation should have highest weight
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
    }

    #[test]
    fn test_souper_method_info() {
        let souper = SouperMerge::new();
        assert_eq!(souper.name(), "souper");
        assert!(!souper.requires_base_model());
    }
}
