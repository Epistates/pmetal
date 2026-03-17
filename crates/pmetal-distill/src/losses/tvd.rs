//! Total Variation Distance loss for knowledge distillation.
//!
//! TVD = 0.5 * Σ|P_teacher - P_student| over the vocabulary dimension.
//!
//! TVD is a proper distance metric (symmetric, satisfies triangle inequality)
//! bounded in [0, 1], making it easy to interpret and compare across runs.

use crate::Result;
use mlx_rs::Array;

use super::DistillLoss;

/// Total Variation Distance loss.
///
/// Computes `0.5 * Σ_i |P_teacher_i - P_student_i|` per token,
/// then averages over all tokens.
pub struct TvdLoss;

impl TvdLoss {
    /// Create a new TVD loss.
    pub fn new() -> Self {
        Self
    }
}

impl Default for TvdLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for TvdLoss {
    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        let temp = Array::from_f32(temperature);

        // Temperature-scaled softmax
        let teacher_scaled = teacher_logits.divide(&temp)?;
        let student_scaled = student_logits.divide(&temp)?;

        let teacher_probs = super::softmax(&teacher_scaled, -1)?;
        let student_probs = super::softmax(&student_scaled, -1)?;

        // TVD = 0.5 * sum(|P - Q|, axis=-1)
        let diff = teacher_probs.subtract(&student_probs)?;
        let abs_diff = mlx_rs::ops::abs(&diff)?;
        let tvd_per_token = abs_diff.sum_axes(&[-1], None)?;
        let tvd_per_token = tvd_per_token.multiply(&Array::from_f32(0.5))?;

        if let Some(w) = weights {
            let weighted = tvd_per_token.multiply(w)?;
            let sum = weighted.sum(false)?;
            let w_sum = w.sum(false)?;
            let safe_sum = mlx_rs::ops::maximum(&w_sum, &Array::from_f32(1.0))?;
            Ok(sum.divide(&safe_sum)?)
        } else {
            Ok(tvd_per_token.mean(None)?)
        }
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
        let logits = Array::from_slice(&[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let loss = TvdLoss::new();
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        result.eval().unwrap();
        let val: f32 = result.item();
        assert!(val.abs() < 1e-5, "TVD of identical should be 0, got {val}");
    }

    #[test]
    fn different_distributions_positive_loss() {
        let teacher = Array::from_slice(&[10.0f32, 0.0, 0.0], &[1, 3]);
        let student = Array::from_slice(&[0.0f32, 0.0, 10.0], &[1, 3]);
        let loss = TvdLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        result.eval().unwrap();
        let val: f32 = result.item();
        assert!(val > 0.9, "TVD of disjoint should be ~1.0, got {val}");
    }

    #[test]
    fn tvd_bounded_zero_to_one() {
        let teacher = Array::from_slice(&[5.0f32, -5.0, 0.0], &[1, 3]);
        let student = Array::from_slice(&[-5.0f32, 5.0, 0.0], &[1, 3]);
        let loss = TvdLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        result.eval().unwrap();
        let val: f32 = result.item();
        assert!(
            (0.0..=1.001).contains(&val),
            "TVD must be in [0,1], got {val}"
        );
    }
}
