//! Total Variation Distance loss for knowledge distillation.
//!
//! TVD = 0.5 * Σ|P_teacher - P_student| over the vocabulary dimension.
//!
//! TVD is a proper distance metric (symmetric, satisfies triangle inequality)
//! bounded in [0, 1], making it easy to interpret and compare across runs.

use crate::Result;
use pmetal_runtime::compat::Array;

use super::{DistillLoss, SPARSE_TOPK_DEFAULT, aligned_log_probs, reduce_per_token};

/// Total Variation Distance loss.
///
/// Computes `0.5 * Σ_i |P_teacher_i - P_student_i|` per token,
/// then averages over all tokens.
pub struct TvdLoss {
    /// Number of top-k teacher tokens to retain when vocab sizes differ.
    sparse_top_k: i32,
}

impl TvdLoss {
    /// Create a new TVD loss.
    pub fn new() -> Self {
        Self {
            sparse_top_k: SPARSE_TOPK_DEFAULT,
        }
    }

    /// Set the number of top-k teacher tokens used in cross-vocab distillation.
    pub fn with_sparse_top_k(mut self, k: i32) -> Self {
        self.sparse_top_k = k.max(1);
        self
    }
}

impl Default for TvdLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for TvdLoss {
    fn weighted_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        let (teacher_log_probs, student_log_probs) = aligned_log_probs(
            teacher_logits,
            student_logits,
            temperature,
            self.sparse_top_k,
        )?;

        // TVD = 0.5 * sum(|P - Q|, axis=-1)
        let tvd_per_token = teacher_log_probs
            .exp()
            .subtract(&student_log_probs.exp())
            .abs_val()
            .sum_axes(&[-1], false)
            .multiply(&Array::from_f32(0.5));

        reduce_per_token(&tvd_per_token, weights)
    }

    fn name(&self) -> &'static str {
        "tvd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_distributions_zero_loss() {
        let logits = Array::from_f32_slice(&[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let loss = TvdLoss::new();
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        result.eval();
        let val: f32 = result.item();
        assert!(val.abs() < 1e-5, "TVD of identical should be 0, got {val}");
    }

    #[test]
    fn different_distributions_positive_loss() {
        let teacher = Array::from_f32_slice(&[10.0f32, 0.0, 0.0], &[1, 3]);
        let student = Array::from_f32_slice(&[0.0f32, 0.0, 10.0], &[1, 3]);
        let loss = TvdLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        result.eval();
        let val: f32 = result.item();
        assert!(val > 0.9, "TVD of disjoint should be ~1.0, got {val}");
    }

    #[test]
    fn tvd_bounded_zero_to_one() {
        let teacher = Array::from_f32_slice(&[5.0f32, -5.0, 0.0], &[1, 3]);
        let student = Array::from_f32_slice(&[-5.0f32, 5.0, 0.0], &[1, 3]);
        let loss = TvdLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        result.eval();
        let val: f32 = result.item();
        assert!(
            (0.0..=1.001).contains(&val),
            "TVD must be in [0,1], got {val}"
        );
    }
}
