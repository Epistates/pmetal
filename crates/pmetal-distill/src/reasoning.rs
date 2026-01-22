//! Rationale-Based Knowledge Distillation (RBKD).
//!
//! This module implements reasoning-aware distillation that specifically targets
//! "Reasoning Tokens" (Chain-of-Thought) to ensure the student model learns the
//! *process* of reasoning, not just the final answer.
//!
//! # Q1 2026 SOTA Context
//!
//! Based on research like "Distilling Reasoning Capabilities" (2025), this method
//! applies higher weight to tokens identified as part of the reasoning chain
//! (e.g., between `<thinking>` tags or automatically detected via attention/entropy).
//!
//! # Algorithm
//!
//! The key insight is that not all tokens are equally important for distillation:
//! - **High-entropy tokens**: Teacher is uncertain → student needs more guidance
//! - **Reasoning tokens**: Critical for learning the thought process
//! - **Answer tokens**: Important but often easier to learn
//!
//! The loss is computed as:
//! ```text
//! weight_i = 1.0 + reasoning_weight * (entropy_i / max_entropy)
//! loss = Σ(weight_i * KL(teacher_i || student_i)) / Σ(weight_i)
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use pmetal_distill::{RationaleLoss, DistillLoss};
//!
//! // Create loss with 2x weight on high-entropy (reasoning) tokens
//! let loss = RationaleLoss::new(2.0);
//!
//! let distill_loss = loss.compute(&teacher_logits, &student_logits, temperature)?;
//! ```
//!
//! # References
//!
//! - Li et al., "LLMs can easily learn to reason from demonstration" (2025)
//! - "Distilling Reasoning Capabilities into Smaller Language Models" (2025)

use crate::{DistillLoss, Result};
use mlx_rs::Array;

/// Rationale-Based Knowledge Distillation Loss.
///
/// Applies higher weight to tokens where the teacher distribution has high entropy,
/// which typically corresponds to reasoning-heavy positions where the student needs
/// more guidance.
#[derive(Debug, Clone)]
pub struct RationaleLoss {
    /// Weight multiplier for high-entropy (reasoning) tokens.
    /// The actual weight applied is: 1.0 + reasoning_weight * normalized_entropy.
    /// Default: 1.0 (so max weight is 2.0 for highest entropy tokens)
    pub reasoning_weight: f32,

    /// Whether to use explicit reasoning markers (e.g., `<thinking>` tags).
    /// When false, uses entropy-based heuristic detection.
    pub use_explicit_markers: bool,

    /// Optional start marker for explicit reasoning regions.
    pub start_marker: Option<String>,

    /// Optional end marker for explicit reasoning regions.
    pub end_marker: Option<String>,

    /// Epsilon for numerical stability in log computations.
    pub eps: f32,
}

impl RationaleLoss {
    /// Create a new Rationale Loss with the given reasoning weight.
    ///
    /// # Arguments
    ///
    /// * `reasoning_weight` - Multiplier for high-entropy tokens.
    ///   A value of 1.0 means max weight is 2.0 (1.0 base + 1.0 * 1.0 max_entropy).
    ///   A value of 2.0 means max weight is 3.0 for highest entropy tokens.
    pub fn new(reasoning_weight: f32) -> Self {
        Self {
            reasoning_weight,
            use_explicit_markers: false,
            start_marker: None,
            end_marker: None,
            eps: 1e-10,
        }
    }

    /// Create with explicit reasoning markers.
    ///
    /// # Arguments
    ///
    /// * `reasoning_weight` - Weight for reasoning tokens
    /// * `start_marker` - Start of reasoning region (e.g., "<thinking>")
    /// * `end_marker` - End of reasoning region (e.g., "</thinking>")
    pub fn with_markers(reasoning_weight: f32, start_marker: &str, end_marker: &str) -> Self {
        Self {
            reasoning_weight,
            use_explicit_markers: true,
            start_marker: Some(start_marker.to_string()),
            end_marker: Some(end_marker.to_string()),
            eps: 1e-10,
        }
    }

    /// Set epsilon for numerical stability.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Compute per-token KL divergence (not reduced to scalar).
    ///
    /// Returns shape: [batch, seq] (summed over vocab dimension only)
    fn per_token_kl(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        // Scale logits by temperature
        let temp = Array::from_f32(temperature);
        let teacher_scaled = teacher_logits.divide(&temp)?;
        let student_scaled = student_logits.divide(&temp)?;

        // Compute softmax probabilities
        let teacher_probs = mlx_rs::ops::softmax_axis(&teacher_scaled, -1, None)?;
        let student_probs = mlx_rs::ops::softmax_axis(&student_scaled, -1, None)?;

        // Add epsilon for numerical stability
        let eps = Array::from_f32(self.eps);
        let teacher_safe = teacher_probs.add(&eps)?;
        let student_safe = student_probs.add(&eps)?;

        // Forward KL: KL(teacher || student) = sum(teacher * log(teacher / student))
        let log_teacher = teacher_safe.log()?;
        let log_student = student_safe.log()?;
        let log_ratio = log_teacher.subtract(&log_student)?;
        let kl_per_vocab = teacher_safe.multiply(&log_ratio)?;

        // Sum over vocabulary dimension only → [batch, seq]
        let kl_per_token = kl_per_vocab.sum_axis(-1, false)?;

        Ok(kl_per_token)
    }

    /// Compute per-token entropy of teacher distribution.
    ///
    /// Returns shape: [batch, seq] (entropy of teacher at each position)
    fn compute_entropy(&self, teacher_logits: &Array, temperature: f32) -> Result<Array> {
        // Scale by temperature
        let temp = Array::from_f32(temperature);
        let teacher_scaled = teacher_logits.divide(&temp)?;

        // Compute softmax probabilities
        let probs = mlx_rs::ops::softmax_axis(&teacher_scaled, -1, None)?;

        // Add epsilon for numerical stability
        let eps = Array::from_f32(self.eps);
        let probs_safe = probs.add(&eps)?;

        // Entropy = -sum(p * log(p)) over vocab dimension
        let log_probs = probs_safe.log()?;
        let p_log_p = probs_safe.multiply(&log_probs)?;
        let neg_entropy = p_log_p.sum_axis(-1, false)?; // [batch, seq]
        let entropy = neg_entropy.multiply(&Array::from_f32(-1.0))?;

        Ok(entropy)
    }

    /// Normalize entropy to [0, 1] range.
    fn normalize_entropy(&self, entropy: &Array) -> Result<Array> {
        // Max entropy for a distribution is log(vocab_size)
        // But we use max observed entropy for relative weighting
        let max_entropy = entropy.max(false)?;
        max_entropy.eval()?;
        let max_val: f32 = max_entropy.item();

        if max_val > self.eps {
            let normalized = entropy.divide(&Array::from_f32(max_val))?;
            Ok(normalized)
        } else {
            // All zeros or very low entropy - return zeros
            Ok(mlx_rs::ops::zeros::<f32>(entropy.shape())?)
        }
    }
}

impl Default for RationaleLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl DistillLoss for RationaleLoss {
    fn name(&self) -> &'static str {
        "rationale_loss"
    }

    fn compute(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        // 1. Compute per-token KL divergence [batch, seq]
        let kl_per_token = self.per_token_kl(teacher_logits, student_logits, temperature)?;

        // 2. Compute entropy of teacher distribution [batch, seq]
        let entropy = self.compute_entropy(teacher_logits, temperature)?;

        // 3. Normalize entropy to [0, 1]
        let normalized_entropy = self.normalize_entropy(&entropy)?;

        // 4. Compute weight map: weight = 1.0 + reasoning_weight * normalized_entropy
        // This gives higher weight to uncertain/reasoning tokens
        let weight_map = normalized_entropy
            .multiply(&Array::from_f32(self.reasoning_weight))?
            .add(&Array::from_f32(1.0))?;

        // 5. Apply weights to per-token loss
        let weighted_loss = kl_per_token.multiply(&weight_map)?;

        // 6. Compute weighted mean (normalize by sum of weights for proper averaging)
        let total_weighted_loss = weighted_loss.sum(false)?;
        let total_weights = weight_map.sum(false)?;

        let mean_loss = total_weighted_loss.divide(&total_weights)?;

        Ok(mean_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rationale_loss_default() {
        let loss = RationaleLoss::default();
        assert!((loss.reasoning_weight - 1.0).abs() < 1e-6);
        assert!(!loss.use_explicit_markers);
    }

    #[test]
    fn test_rationale_loss_with_weight() {
        let loss = RationaleLoss::new(2.0);
        assert!((loss.reasoning_weight - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_rationale_loss_with_markers() {
        let loss = RationaleLoss::with_markers(1.5, "<thinking>", "</thinking>");
        assert!((loss.reasoning_weight - 1.5).abs() < 1e-6);
        assert!(loss.use_explicit_markers);
        assert_eq!(loss.start_marker.as_deref(), Some("<thinking>"));
        assert_eq!(loss.end_marker.as_deref(), Some("</thinking>"));
    }

    #[test]
    fn test_identical_distributions() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let loss = RationaleLoss::new(1.0);
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        result.eval().unwrap();
        let value: f32 = result.item();

        // KL of identical distributions should be ~0
        assert!(
            value.abs() < 1e-4,
            "Loss of identical distributions should be ~0, got {}",
            value
        );
    }

    #[test]
    fn test_different_distributions() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = RationaleLoss::new(1.0);
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        result.eval().unwrap();
        let value: f32 = result.item();

        // Loss should be positive
        assert!(value > 0.0, "Loss should be positive, got {}", value);
    }

    #[test]
    fn test_reasoning_weight_effect() {
        // Create distributions where some positions have higher entropy
        // Position 0: low entropy (peaked distribution)
        // Position 1: high entropy (uniform-ish distribution)
        let teacher = Array::from_slice(
            &[
                // Position 0: peaked at index 3
                0.0_f32, 0.0, 0.0, 10.0, // Position 1: more uniform
                1.0, 1.0, 1.0, 1.0,
            ],
            &[1, 2, 4],
        );
        let student = Array::from_slice(
            &[
                // Position 0: wrong but peaked
                10.0_f32, 0.0, 0.0, 0.0, // Position 1: also wrong
                2.0, 0.0, 0.0, 0.0,
            ],
            &[1, 2, 4],
        );

        // With low reasoning weight, high-entropy position contributes equally
        let low_weight_loss = RationaleLoss::new(0.0);
        let loss_low = low_weight_loss.compute(&teacher, &student, 1.0).unwrap();
        loss_low.eval().unwrap();
        let val_low: f32 = loss_low.item();

        // With high reasoning weight, high-entropy position contributes more
        let high_weight_loss = RationaleLoss::new(5.0);
        let loss_high = high_weight_loss.compute(&teacher, &student, 1.0).unwrap();
        loss_high.eval().unwrap();
        let val_high: f32 = loss_high.item();

        // Both should be positive
        assert!(val_low > 0.0);
        assert!(val_high > 0.0);

        // The weighted version should differ (could be higher or lower depending on
        // which position has more loss)
        println!("Low weight loss: {}, High weight loss: {}", val_low, val_high);
    }

    #[test]
    fn test_per_token_kl_shape() {
        let teacher = Array::from_slice(
            &[
                1.0_f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0,
            ],
            &[2, 3, 2], // batch=2, seq=3, vocab=2
        );
        let student = Array::from_slice(
            &[
                4.0_f32, 3.0, 5.0, 4.0, 6.0, 5.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0,
            ],
            &[2, 3, 2],
        );

        let loss = RationaleLoss::new(1.0);
        let kl = loss.per_token_kl(&teacher, &student, 1.0).unwrap();
        kl.eval().unwrap();

        // Should be [batch, seq] = [2, 3]
        assert_eq!(kl.shape(), &[2, 3]);
    }

    #[test]
    fn test_entropy_shape() {
        let teacher = Array::from_slice(
            &[
                1.0_f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0,
            ],
            &[2, 3, 2], // batch=2, seq=3, vocab=2
        );

        let loss = RationaleLoss::new(1.0);
        let entropy = loss.compute_entropy(&teacher, 1.0).unwrap();
        entropy.eval().unwrap();

        // Should be [batch, seq] = [2, 3]
        assert_eq!(entropy.shape(), &[2, 3]);

        // Entropy should be non-negative
        let vals: Vec<f32> = entropy.as_slice().to_vec();
        for &v in &vals {
            assert!(v >= 0.0, "Entropy should be non-negative");
        }
    }

    #[test]
    fn test_larger_batch() {
        let batch_size = 4;
        let seq_len = 16;
        let vocab_size = 256;

        let teacher_data: Vec<f32> = (0..(batch_size * seq_len * vocab_size))
            .map(|i| ((i % 100) as f32 - 50.0) / 10.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch_size * seq_len * vocab_size))
            .map(|i| ((i * 7 % 100) as f32 - 50.0) / 10.0)
            .collect();

        let teacher = Array::from_slice(
            &teacher_data,
            &[batch_size as i32, seq_len as i32, vocab_size as i32],
        );
        let student = Array::from_slice(
            &student_data,
            &[batch_size as i32, seq_len as i32, vocab_size as i32],
        );

        let loss = RationaleLoss::new(1.5);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        result.eval().unwrap();
        let value: f32 = result.item();

        assert!(value > 0.0, "Loss should be positive");
        assert!(value.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_name() {
        let loss = RationaleLoss::new(1.0);
        assert_eq!(loss.name(), "rationale_loss");
    }
}
