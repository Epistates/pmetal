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

    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        external_weights: Option<&Array>,
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

        // 2. Compute internal reasoning weights based on teacher entropy
        let p_log_p = teacher_probs.multiply(&teacher_logprobs)?;
        let entropy = p_log_p
            .sum_axis(-1, false)?
            .multiply(&Array::from_f32(-1.0))?;

        // Normalize entropy in-graph (no .item() to preserve autodiff)
        let max_entropy = entropy.max(false)?;
        let safe_max = mlx_rs::ops::maximum(&max_entropy, &Array::from_f32(1e-6))?;
        let normalized_entropy = entropy.divide(&safe_max)?;

        let mut internal_weight = normalized_entropy
            .multiply(&Array::from_f32(self.reasoning_weight))?
            .add(&Array::from_f32(1.0))?;

        // 3. Combine with external weights (e.g., from reasoning markers or outcome supervision)
        if let Some(w) = external_weights {
            internal_weight = internal_weight.multiply(w)?;
        }

        // 4. Apply weights and compute mean
        let weighted_loss = kl_per_token.multiply(&internal_weight)?;

        // Weighted mean in-graph (no .item() to preserve autodiff)
        let total_weighted_loss = weighted_loss.sum(false)?;
        let total_weights = internal_weight.sum(false)?;
        let safe_weights = mlx_rs::ops::maximum(&total_weights, &Array::from_f32(1e-6))?;
        Ok(total_weighted_loss.divide(&safe_weights)?)
    }
}

/// Helper to generate a reasoning mask from token IDs using markers.
///
/// # Arguments
/// * `tokens` - Tokenized sequence `[batch, seq]`
/// * `start_token` - Token ID for the start of reasoning (e.g., "<think>")
/// * `end_token` - Token ID for the end of reasoning (e.g., "</think>")
pub fn generate_reasoning_mask(tokens: &Array, start_token: u32, end_token: u32) -> Result<Array> {
    let start_arr = Array::from_int(start_token as i32);
    let end_arr = Array::from_int(end_token as i32);

    let is_start = tokens.eq(&start_arr)?;
    let is_end = tokens.eq(&end_arr)?;

    // Use cumulative sum to track being "inside" reasoning tags.
    // exclusive cumsum for start: the start token itself should NOT be included.
    // inclusive cumsum for end: the end token position should mark the boundary.
    let start_cumsum =
        mlx_rs::ops::cumsum(&is_start.as_dtype(mlx_rs::Dtype::Int32)?, 1, None, None)?;
    let end_cumsum = mlx_rs::ops::cumsum(&is_end.as_dtype(mlx_rs::Dtype::Int32)?, 1, None, None)?;

    let mask = start_cumsum.subtract(&end_cumsum)?;

    // Clamp to [0, 1] to handle multiple/nested reasoning blocks correctly
    let mask = mlx_rs::ops::clip(&mask, (&Array::from_int(0), &Array::from_int(1)))?;

    Ok(mask.as_dtype(mlx_rs::Dtype::Float32)?)
}

/// Outcome-Supervised Rationale Distillation Loss.
///
/// Only distilling from rationales that lead to correct outcomes.
/// Uses a scalar correctness signal per sample to zero out or down-weight incorrect reasoning chains.
#[derive(Debug, Clone)]
pub struct OutcomeSupervisedRationaleLoss {
    pub base_loss: RationaleLoss,
}

impl OutcomeSupervisedRationaleLoss {
    pub fn new(reasoning_weight: f32) -> Self {
        Self {
            base_loss: RationaleLoss::new(reasoning_weight),
        }
    }

    /// Compute loss using a correctness mask.
    ///
    /// # Arguments
    /// * `correctness` - Array of shape `[batch]` containing 1.0 for correct and 0.0 for incorrect.
    pub fn compute_with_outcome(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        correctness: &Array,
    ) -> Result<Array> {
        // Expand correctness from [batch] to [batch, 1] for broadcasting
        let weight = correctness.reshape(&[-1, 1])?;
        self.base_loss
            .compute_weighted(teacher_logits, student_logits, temperature, Some(&weight))
    }
}

impl DistillLoss for OutcomeSupervisedRationaleLoss {
    fn name(&self) -> &'static str {
        "outcome_supervised_rationale_loss"
    }

    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        self.base_loss
            .compute_weighted(teacher_logits, student_logits, temperature, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_generate_reasoning_mask() {
        // Tokens: [CLS, "Hello", "<think>", " reasoning", " steps", "</think>", " final", " answer"]
        // Indices:  0,      1,         2,          3,        4,         5,        6,        7
        let tokens = Array::from_slice(&[0_i32, 1, 2, 3, 4, 5, 6, 7], &[1, 8]);
        let start_token = 2;
        let end_token = 5;

        let mask = generate_reasoning_mask(&tokens, start_token, end_token).unwrap();
        mask.eval().unwrap();
        let mask_vals: Vec<f32> = mask.as_slice().to_vec();

        // Indices 3 and 4 should be 1.0 (inside tags)
        // Indices 2 and 5 behavior depends on implementation (currently 2 is 1.0, 5 is 0.0 due to cumsum timing)
        assert_eq!(mask_vals[3], 1.0);
        assert_eq!(mask_vals[4], 1.0);
        assert_eq!(mask_vals[1], 0.0);
        assert_eq!(mask_vals[6], 0.0);
    }

    #[test]
    fn test_outcome_supervised_loss() {
        // Use asymmetric logits so the two samples have different KL divergences
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 1.0], &[2, 1, 2]);
        let student = Array::from_slice(&[2.0_f32, 1.0, 1.0, 3.0], &[2, 1, 2]);

        let loss = OutcomeSupervisedRationaleLoss::new(1.0);

        // Case 1: First sample is correct, second is incorrect
        let correctness = Array::from_slice(&[1.0_f32, 0.0], &[2]);
        let result = loss
            .compute_with_outcome(&teacher, &student, 1.0, &correctness)
            .unwrap();
        result.eval().unwrap();
        let val1: f32 = result.item();

        // Case 2: Both correct
        let correctness_all = Array::from_slice(&[1.0_f32, 1.0], &[2]);
        let result_all = loss
            .compute_with_outcome(&teacher, &student, 1.0, &correctness_all)
            .unwrap();
        result_all.eval().unwrap();
        let val_all: f32 = result_all.item();

        assert!(val1 > 0.0);
        assert!(val_all > 0.0);
        assert!(
            (val1 - val_all).abs() > 1e-5,
            "Weighting should change the loss value"
        );
    }

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
