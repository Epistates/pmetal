//! Rationale-weighted distillation for reasoning traces.
//!
//! Distilling a reasoning model spends most of its tokens on formulaic text
//! (restating the question, connectives, boilerplate) and a few on the steps
//! where the teacher actually decides something. [`RationaleLoss`] puts more
//! weight where the teacher is uncertain, a token-level reweighting of the
//! forward KL:
//!
//! ```text
//! w_i  = 1 + reasoning_weight · H(teacher_i) / ln V
//! loss = Σ_i w_i · KL(teacher_i || student_i) / Σ_i w_i
//! ```
//!
//! where `H` is the entropy of the teacher's softened distribution and `ln V`
//! the entropy of a uniform one, so `H / ln V ∈ [0, 1]` and a token's weight
//! depends only on that token, not on what else is in the batch. This is a
//! heuristic, not a published estimator.
//!
//! Which tokens are *reasoning* tokens is a separate signal. An explicit span
//! (`<think> ... </think>`) is turned into a 0/1 mask over token ids by
//! [`generate_reasoning_mask`]; pass it, scaled as you like, as the per-token
//! `weights` of any [`DistillLoss`]. Resolving marker *strings* to ids needs
//! a tokenizer and is the caller's: [`LossConfig`](crate::LossConfig) carries
//! the strings so the training loop can.
//!
//! [`OutcomeSupervisedRationaleLoss`] adds a per-sample correctness weight so
//! only traces that reached a correct answer are distilled.

use crate::{DistillLoss, Result};
use pmetal_runtime::compat::{Array, Dtype, ops};

/// Entropy-weighted forward-KL loss for reasoning distillation.
///
/// Applies higher weight to tokens where the teacher distribution has high
/// entropy, which typically corresponds to the steps of a trace where the
/// student needs more guidance.
#[derive(Debug, Clone)]
pub struct RationaleLoss {
    /// Weight multiplier for high-entropy tokens: a token's weight is
    /// `1 + reasoning_weight · H / ln V`. Default 1.0 (so the highest-entropy
    /// tokens weigh twice the lowest).
    pub reasoning_weight: f32,
}

impl RationaleLoss {
    /// Create a new Rationale Loss with the given reasoning weight.
    pub fn new(reasoning_weight: f32) -> Self {
        Self { reasoning_weight }
    }
}

impl Default for RationaleLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// `log_softmax(logits / T)` in f32.
fn softened_log_probs(logits: &Array, temperature: f32) -> Array {
    logits
        .as_dtype(Dtype::Float32.as_i32())
        .divide(&Array::from_f32(temperature))
        .log_softmax(-1)
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
        let teacher_logprobs = softened_log_probs(&teacher_logits.stop_gradient(), temperature);
        let student_logprobs = softened_log_probs(student_logits, temperature);
        Ok(teacher_logprobs
            .exp()
            .multiply(&teacher_logprobs.subtract(&student_logprobs))
            .sum_axes(&[-1], false))
    }

    /// Compute per-token entropy of the teacher distribution.
    ///
    /// Returns a `[batch, seq]` array containing H(teacher_i) per token.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher logits `[batch, seq, vocab]`
    /// * `temperature` - Softmax temperature
    pub fn compute_entropy(&self, teacher_logits: &Array, temperature: f32) -> Result<Array> {
        let teacher_logprobs = softened_log_probs(&teacher_logits.stop_gradient(), temperature);
        // H = -sum(p * log(p)) over vocab -> [batch, seq]
        Ok(teacher_logprobs
            .exp()
            .multiply(&teacher_logprobs)
            .sum_axes(&[-1], false)
            .negative())
    }

    /// The weight of every token, `1 + reasoning_weight · H / ln V`, a
    /// constant of the teacher.
    fn token_weights(&self, teacher_logits: &Array, temperature: f32) -> Result<Array> {
        let entropy = self.compute_entropy(teacher_logits, temperature)?;
        let max_entropy = (teacher_logits.dim(-1).max(2) as f32).ln();
        Ok(entropy
            .divide(&Array::from_f32(max_entropy))
            .multiply(&Array::from_f32(self.reasoning_weight))
            .add(&Array::from_f32(1.0))
            .stop_gradient())
    }
}

impl DistillLoss for RationaleLoss {
    fn name(&self) -> &'static str {
        "rationale_loss"
    }

    fn weighted_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        external_weights: Option<&Array>,
    ) -> Result<Array> {
        let kl_per_token = self.per_token_kl(teacher_logits, student_logits, temperature)?;
        let mut weight = self.token_weights(teacher_logits, temperature)?;

        // Combine with external weights (a padding or reasoning-span mask,
        // outcome correctness). They may broadcast, e.g. `[batch, 1]`.
        if let Some(w) = external_weights {
            weight = weight.multiply(&w.as_dtype(Dtype::Float32.as_i32()));
        }

        // Weighted mean in-graph.
        let total = kl_per_token.multiply(&weight).sum_all();
        let total_weights = weight.sum_all();
        Ok(total.divide(&ops::maximum(&total_weights, &Array::from_f32(1e-6))))
    }
}

/// A `[batch, seq]` float mask that is 1 on the tokens inside a reasoning
/// span and 0 elsewhere.
///
/// A span opens at `start_token` (which is inside it) and closes at the next
/// `end_token` (which is not). The state is whichever marker came last: a
/// closing marker with nothing open (a window that began mid-span), repeated
/// markers and several spans per row all behave, and a span still open at the
/// end of the row runs to the end.
///
/// # Arguments
/// * `tokens` - Tokenized sequence `[batch, seq]`
/// * `start_token` - Token ID for the start of reasoning (e.g., `<think>`)
/// * `end_token` - Token ID for the end of reasoning (e.g., `</think>`)
pub fn generate_reasoning_mask(tokens: &Array, start_token: u32, end_token: u32) -> Result<Array> {
    let i32_ = Dtype::Int32.as_i32();
    let seq = tokens.dim(-1);
    let is_start = tokens.equal(&Array::from_i32(start_token as i32));
    let is_end = tokens.equal(&Array::from_i32(end_token as i32));
    let start = is_start.as_dtype(i32_);
    let end = is_end.as_dtype(i32_);

    // +1 at a start marker, -1 at an end marker, 0 elsewhere.
    let sign = start.subtract(&end);
    let marker = start.add(&end);
    // How many markers up to and including each position.
    let seen = marker.cumsum(-1);

    // Compact the markers' signs in position order: sorting by
    // `(not a marker, position)` puts them first, so the n-th marker's sign
    // sits at rank n-1.
    let position = Array::arange(seq, i32_);
    let key = Array::from_i32(1)
        .subtract(&marker)
        .multiply(&Array::from_i32(seq))
        .add(&position);
    let by_rank = sign.take_along_axis(&key.argsort(-1), -1);

    // The sign of the latest marker at or before each position.
    let rank = ops::maximum(&seen.subtract(&Array::from_i32(1)), &Array::from_i32(0));
    let latest = by_rank.take_along_axis(&rank, -1);

    let open = latest
        .greater(&Array::from_i32(0))
        .as_dtype(Dtype::Float32.as_i32());
    let any_marker = seen
        .greater(&Array::from_i32(0))
        .as_dtype(Dtype::Float32.as_i32());
    Ok(open.multiply(&any_marker))
}

/// Outcome-Supervised Rationale Distillation Loss.
///
/// Only distilling from rationales that lead to correct outcomes.
/// Uses a scalar correctness signal per sample to zero out or down-weight incorrect reasoning chains.
#[derive(Debug, Clone)]
pub struct OutcomeSupervisedRationaleLoss {
    /// The entropy-weighted loss the correctness weights are applied to.
    pub base_loss: RationaleLoss,
}

impl OutcomeSupervisedRationaleLoss {
    /// Create the loss with the given entropy weight.
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
        let weight = correctness.reshape(&[-1, 1]);
        self.base_loss
            .compute_weighted(teacher_logits, student_logits, temperature, Some(&weight))
    }
}

impl DistillLoss for OutcomeSupervisedRationaleLoss {
    fn name(&self) -> &'static str {
        "outcome_supervised_rationale_loss"
    }

    /// Without a correctness signal this is the plain entropy-weighted loss;
    /// pass per-sample correctness as `weights` (`[batch, 1]`) or call
    /// [`compute_with_outcome`](Self::compute_with_outcome).
    fn weighted_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        self.base_loss
            .weighted_loss(teacher_logits, student_logits, temperature, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_outcome_supervised_loss() {
        // Use asymmetric logits so the two samples have different KL divergences
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 1.0], &[2, 1, 2]);
        let student = Array::from_f32_slice(&[2.0_f32, 1.0, 1.0, 3.0], &[2, 1, 2]);

        let loss = OutcomeSupervisedRationaleLoss::new(1.0);

        // Case 1: First sample is correct, second is incorrect
        let correctness = Array::from_f32_slice(&[1.0_f32, 0.0], &[2]);
        let result = loss
            .compute_with_outcome(&teacher, &student, 1.0, &correctness)
            .unwrap();
        result.eval();
        let val1: f32 = result.item();

        // Case 2: Both correct
        let correctness_all = Array::from_f32_slice(&[1.0_f32, 1.0], &[2]);
        let result_all = loss
            .compute_with_outcome(&teacher, &student, 1.0, &correctness_all)
            .unwrap();
        result_all.eval();
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
    }

    #[test]
    #[serial]
    fn test_identical_distributions() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let loss = RationaleLoss::new(1.0);
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        result.eval();
        let value: f32 = result.item();

        // KL of identical distributions should be ~0
        assert!(
            value.abs() < 1e-4,
            "Loss of identical distributions should be ~0, got {}",
            value
        );
    }

    #[test]
    #[serial]
    fn test_different_distributions() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = RationaleLoss::new(1.0);
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        result.eval();
        let value: f32 = result.item();

        // Loss should be positive
        assert!(value > 0.0, "Loss should be positive, got {}", value);
    }

    #[test]
    #[serial]
    fn test_per_token_kl_shape() {
        let teacher = Array::from_f32_slice(
            &[
                1.0_f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0,
            ],
            &[2, 3, 2], // batch=2, seq=3, vocab=2
        );
        let student = Array::from_f32_slice(
            &[
                4.0_f32, 3.0, 5.0, 4.0, 6.0, 5.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0,
            ],
            &[2, 3, 2],
        );

        let loss = RationaleLoss::new(1.0);
        let kl = loss.per_token_kl(&teacher, &student, 1.0).unwrap();
        kl.eval();

        // Should be [batch, seq] = [2, 3]
        assert_eq!(kl.shape(), &[2, 3]);
    }

    #[test]
    #[serial]
    fn test_entropy_shape() {
        let teacher = Array::from_f32_slice(
            &[
                1.0_f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0,
            ],
            &[2, 3, 2], // batch=2, seq=3, vocab=2
        );

        let loss = RationaleLoss::new(1.0);
        let entropy = loss.compute_entropy(&teacher, 1.0).unwrap();
        entropy.eval();

        // Should be [batch, seq] = [2, 3]
        assert_eq!(entropy.shape(), &[2, 3]);

        // Entropy should be non-negative
        let vals: Vec<f32> = entropy.clone().to_f32_vec(6).unwrap();
        for &v in &vals {
            assert!(v >= 0.0, "Entropy should be non-negative");
        }
    }

    #[test]
    #[serial]
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

        let teacher = Array::from_f32_slice(&teacher_data, &[batch_size, seq_len, vocab_size]);
        let student = Array::from_f32_slice(&student_data, &[batch_size, seq_len, vocab_size]);

        let loss = RationaleLoss::new(1.5);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        result.eval();
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
