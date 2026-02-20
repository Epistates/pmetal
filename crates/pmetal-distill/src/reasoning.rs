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

    /// Cached Metal context for GPU acceleration.
    #[cfg(feature = "metal")]
    ctx: Option<std::sync::Arc<pmetal_metal::context::MetalContext>>,
}

impl RationaleLoss {
    /// Create a new Rationale Loss with the given reasoning weight.
    pub fn new(reasoning_weight: f32) -> Self {
        Self {
            reasoning_weight,
            use_explicit_markers: false,
            start_marker: None,
            end_marker: None,
            eps: 1e-6,
            #[cfg(feature = "metal")]
            ctx: pmetal_metal::context::MetalContext::global().ok(),
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
            eps: 1e-6,
            #[cfg(feature = "metal")]
            ctx: pmetal_metal::context::MetalContext::global().ok(),
        }
    }

    /// Set epsilon for numerical stability.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl Default for RationaleLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl RationaleLoss {
    /// Compute per-token KL divergence between teacher and student distributions.
    ///
    /// Returns a `[batch, seq]` array containing KL(teacher_i || student_i) per token.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher logits `[batch, seq, vocab]`
    /// * `student_logits` - Student logits `[batch, seq, vocab]`
    /// * `temperature` - Softmax temperature
    pub fn per_token_kl(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        let temp = Array::from_f32(temperature);
        let teacher_scaled = teacher_logits.divide(&temp)?;
        let student_scaled = student_logits.divide(&temp)?;

        let teacher_logprobs = mlx_rs::nn::log_softmax(&teacher_scaled, -1)?;
        let student_logprobs = mlx_rs::nn::log_softmax(&student_scaled, -1)?;

        let teacher_probs = teacher_logprobs.exp()?;
        let log_ratio = teacher_logprobs.subtract(&student_logprobs)?;
        let kl_per_vocab = teacher_probs.multiply(&log_ratio)?;

        // Sum over vocab dimension -> [batch, seq]
        Ok(kl_per_vocab.sum_axis(-1, false)?)
    }

    /// Compute per-token entropy of the teacher distribution.
    ///
    /// Returns a `[batch, seq]` array containing H(teacher_i) per token.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher logits `[batch, seq, vocab]`
    /// * `temperature` - Softmax temperature
    pub fn compute_entropy(&self, teacher_logits: &Array, temperature: f32) -> Result<Array> {
        let temp = Array::from_f32(temperature);
        let teacher_scaled = teacher_logits.divide(&temp)?;

        let teacher_logprobs = mlx_rs::nn::log_softmax(&teacher_scaled, -1)?;
        let teacher_probs = teacher_logprobs.exp()?;

        // H = -sum(p * log(p)) over vocab -> [batch, seq]
        let p_log_p = teacher_probs.multiply(&teacher_logprobs)?;
        Ok(p_log_p
            .sum_axis(-1, false)?
            .multiply(&Array::from_f32(-1.0))?)
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
        let temp = Array::from_f32(temperature);
        let teacher_scaled = teacher_logits.divide(&temp)?;
        let student_scaled = student_logits.divide(&temp)?;

        // 1. Compute per-token KL divergence stably
        let teacher_logprobs = mlx_rs::nn::log_softmax(&teacher_scaled, -1)?;
        let student_logprobs = mlx_rs::nn::log_softmax(&student_scaled, -1)?;

        let teacher_probs = teacher_logprobs.exp()?;
        let log_ratio = teacher_logprobs.subtract(&student_logprobs)?;
        let kl_per_vocab = teacher_probs.multiply(&log_ratio)?;
        let kl_per_token = kl_per_vocab.sum_axis(-1, false)?;

        // 2. Compute per-token entropy stably
        let p_log_p = teacher_probs.multiply(&teacher_logprobs)?;
        let entropy = p_log_p
            .sum_axis(-1, false)?
            .multiply(&Array::from_f32(-1.0))?;

        // 3. Normalize entropy and compute weight map
        let max_entropy = entropy.max(false)?;
        max_entropy.eval()?;
        let max_val: f32 = max_entropy.item();

        let normalized_entropy = if max_val > 1e-6 {
            entropy.divide(&Array::from_f32(max_val))?
        } else {
            mlx_rs::ops::zeros::<f32>(entropy.shape())?
        };

        let weight_map = normalized_entropy
            .multiply(&Array::from_f32(self.reasoning_weight))?
            .add(&Array::from_f32(1.0))?;

        // 4. Apply weights and compute mean
        let weighted_loss = kl_per_token.multiply(&weight_map)?;

        let total_weighted_loss = weighted_loss.sum(false)?;
        let total_weights = weight_map.sum(false)?;

        total_weighted_loss.eval()?;
        total_weights.eval()?;

        let total_weights_val: f32 = total_weights.item();
        if total_weights_val > 1e-6 {
            Ok(total_weighted_loss.divide(&total_weights)?)
        } else {
            Ok(weighted_loss.mean(None)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

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
        println!(
            "Low weight loss: {}, High weight loss: {}",
            val_low, val_high
        );
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
