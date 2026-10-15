//! Skew KL divergences from DistiLLM.
//!
//! Ko, Kim, Chen, Yun, *DistiLLM: Towards Streamlined Distillation for Large
//! Language Models* (ICML 2024). Plain KL between a small student and a large
//! teacher has two failure modes: forward KL `KL(T || S)` blows up wherever
//! the student has not yet put mass, and reverse KL `KL(S || T)` blows up
//! wherever the teacher has none. Mixing the other distribution into the
//! second argument bounds both, at a small cost in bias (eq. 5 and its
//! reverse):
//!
//! ```text
//! SKL(α)  = KL( T || α·T + (1-α)·S )
//! SRKL(α) = KL( S || (1-α)·T + α·S )
//! ```
//!
//! `α = 0` is the plain forward / reverse KL; the gradient stays finite for
//! any `α > 0` even where the student assigns a token probability zero. The
//! paper's best value is `α = 0.1`.
//!
//! Everything is computed in log space from f32 log-probabilities.

use super::{
    DistillLoss, SPARSE_TOPK_DEFAULT, aligned_log_probs, kl_from_log_probs, log_mixture,
    reduce_per_token,
};
use crate::Result;
use pmetal_runtime::compat::Array;

/// Which skew divergence to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// `KL(T || α·T + (1-α)·S)`.
    Forward,
    /// `KL(S || (1-α)·T + α·S)`.
    Reverse,
}

/// Skew KL (`SKL`) or skew reverse KL (`SRKL`) with mixing weight `α`.
pub struct SkewKlLoss {
    alpha: f32,
    direction: Direction,
    sparse_top_k: i32,
}

impl SkewKlLoss {
    /// `SKL(α) = KL(T || α·T + (1-α)·S)`, `α` clamped to `[0, 1)`.
    pub fn forward(alpha: f32) -> Self {
        Self::new(alpha, Direction::Forward)
    }

    /// `SRKL(α) = KL(S || (1-α)·T + α·S)`, `α` clamped to `[0, 1)`.
    pub fn reverse(alpha: f32) -> Self {
        Self::new(alpha, Direction::Reverse)
    }

    fn new(alpha: f32, direction: Direction) -> Self {
        // α = 1 makes SKL compare the teacher with itself and SRKL the student
        // with itself: a loss that is zero for every student.
        Self {
            alpha: alpha.clamp(0.0, 1.0 - 1e-6),
            direction,
            sparse_top_k: SPARSE_TOPK_DEFAULT,
        }
    }

    /// Override the sparse top-k used for cross-vocab alignment.
    pub fn with_sparse_top_k(mut self, k: i32) -> Self {
        self.sparse_top_k = k.max(1);
        self
    }

    /// The mixing weight in use.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Whether this is the reverse (`SRKL`) direction.
    pub fn is_reverse(&self) -> bool {
        self.direction == Direction::Reverse
    }
}

impl DistillLoss for SkewKlLoss {
    fn weighted_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        let (log_t, log_s) = aligned_log_probs(
            teacher_logits,
            student_logits,
            temperature,
            self.sparse_top_k,
        )?;
        let per_token = match (self.direction, self.alpha) {
            (Direction::Forward, a) if a == 0.0 => kl_from_log_probs(&log_t, &log_s),
            (Direction::Reverse, a) if a == 0.0 => kl_from_log_probs(&log_s, &log_t),
            (Direction::Forward, a) => kl_from_log_probs(&log_t, &log_mixture(&log_t, &log_s, a)),
            (Direction::Reverse, a) => {
                kl_from_log_probs(&log_s, &log_mixture(&log_t, &log_s, 1.0 - a))
            }
        };
        reduce_per_token(&per_token, weights)
    }

    fn name(&self) -> &'static str {
        match self.direction {
            Direction::Forward => "skew_kl",
            Direction::Reverse => "skew_reverse_kl",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// Identical teacher and student give zero for every α.
    #[test]
    #[serial]
    fn identical_inputs_zero_loss() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 0.5, 1.5, 2.5], &[1, 2, 3]);
        for alpha in [0.0_f32, 0.1, 0.5] {
            for loss in [SkewKlLoss::forward(alpha), SkewKlLoss::reverse(alpha)] {
                let v: f32 = loss.compute(&logits, &logits, 1.0).unwrap().item();
                assert!(v.abs() < 1e-5, "alpha={alpha} got {v}");
            }
        }
    }

    /// The skew divergences stay finite where a plain KL does not: a student
    /// that puts (numerically) no mass on the teacher's favourite token.
    #[test]
    #[serial]
    fn stays_finite_when_the_student_has_no_mass() {
        let teacher = Array::from_f32_slice(&[20.0_f32, 0.0, 0.0], &[1, 1, 3]);
        let student = Array::from_f32_slice(&[-80.0_f32, 5.0, 0.0], &[1, 1, 3]);
        let v: f32 = SkewKlLoss::forward(0.1)
            .compute(&teacher, &student, 1.0)
            .unwrap()
            .item();
        assert!(v.is_finite(), "got {v}");
        // Bounded by -ln(α) for SKL: KL(T || αT + (1-α)S) <= ln(1/α).
        assert!(v <= (1.0_f32 / 0.1).ln() + 1e-4, "got {v}");
    }
}
