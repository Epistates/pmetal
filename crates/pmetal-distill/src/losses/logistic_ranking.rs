//! Logistic ranking loss for knowledge distillation.
//!
//! A softer alternative to hinge ranking loss using softplus:
//!
//! For each pair (i, j) where P_teacher[i] > P_teacher[j]:
//!   loss += softplus(-(logit_student[i] - logit_student[j]))
//!         = log(1 + exp(-(logit_student[i] - logit_student[j])))
//!
//! Provides smoother gradients than hinge for more stable training.

use crate::Result;
use pmetal_runtime::compat::{Array, Dtype, ops};

use super::{DistillLoss, reduce_per_token};

/// Logistic ranking loss for smooth pairwise token ordering.
///
/// Uses softplus (log(1 + exp(-x))) instead of relu for smoother gradients.
/// Operates on student logits directly (not probabilities) for stability.
pub struct LogisticRankingLoss {
    /// Number of top teacher tokens to consider for pairwise ranking.
    top_k: i32,
    /// Minimum teacher probability difference to form a valid pair.
    eps: f32,
}

impl LogisticRankingLoss {
    /// Create a new logistic ranking loss with default top_k=32.
    pub fn new() -> Self {
        Self {
            top_k: 32,
            eps: 1e-6,
        }
    }

    /// Set the number of top-k teacher tokens for pairwise ranking.
    pub fn with_top_k(mut self, k: i32) -> Self {
        self.top_k = k;
        self
    }
}

impl Default for LogisticRankingLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for LogisticRankingLoss {
    fn weighted_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        let temp = Array::from_f32(temperature);
        let k = self.top_k;

        // Teacher probabilities for ranking
        let teacher_scaled = teacher_logits.divide(&temp);
        let teacher_probs = super::softmax(&teacher_scaled, -1)?;

        // Top-k indices
        let vocab = teacher_logits.dim(-1);
        let k = k.min((vocab - 1).max(0)); // argpartition requires kth < axis size
        let neg_teacher = teacher_probs.negative();
        let partitioned = ops::argpartition_axis(&neg_teacher, k, -1);
        // Slice first k along the last axis: reshape to 2D, slice, reshape back
        let part_shape = partitioned.shape().to_vec();
        let part_ndim = part_shape.len();
        let part_n: i32 = part_shape[..part_ndim - 1].iter().product();
        let part_2d = partitioned.reshape(&[part_n, vocab]);
        let top_k_2d = part_2d.slice(&[0, 0], &[part_n, k]);
        let mut top_k_shape: Vec<i32> = part_shape[..part_ndim - 1].to_vec();
        top_k_shape.push(k);
        let top_k_idx = top_k_2d.reshape(&top_k_shape);

        // Gather teacher probs at top-k
        let t_gathered = teacher_probs.take_along_axis(&top_k_idx, -1);

        // Student LOGITS (not probs) at same positions — for softplus stability
        let student_scaled = student_logits.divide(&temp);
        let s_gathered = student_scaled.take_along_axis(&top_k_idx, -1);

        // Flatten to [N, K]
        let shape = teacher_logits.shape();
        let ndim = shape.len();
        let n: i32 = shape[..ndim - 1].iter().product();

        let t_flat = t_gathered.reshape(&[n, k]);
        let s_flat = s_gathered.reshape(&[n, k]);

        // Pairwise: [N, K, 1] vs [N, 1, K]
        let t_i = t_flat.reshape(&[n, k, 1]);
        let t_j = t_flat.reshape(&[n, 1, k]);
        let s_i = s_flat.reshape(&[n, k, 1]);
        let s_j = s_flat.reshape(&[n, 1, k]);

        let teacher_margin = t_i.subtract(&t_j);
        let student_diff = s_i.subtract(&s_j); // logit difference

        // Valid pairs
        let eps_arr = Array::from_f32(self.eps);
        let valid_mask = teacher_margin
            .greater(&eps_arr)
            .as_dtype(Dtype::Float32.as_i32());

        // Softplus(-diff) = log(1 + exp(-diff))
        // Numerically stable: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
        let neg_diff = student_diff.negative();
        let abs_diff = ops::abs(&student_diff);
        let zero = Array::from_f32(0.0);
        // log1p replacement: log(1 + exp(-|diff|)) = abs_diff.negative().exp().add(1.0).log()
        let softplus = ops::maximum(&neg_diff, &zero)
            .add(&abs_diff.negative().exp().add(&Array::from_f32(1.0)).log());

        // Masked mean
        let masked = softplus.multiply(&valid_mask);
        let pair_sum = masked.sum_axes(&[-1, -2], false);
        let pair_count = valid_mask.sum_axes(&[-1, -2], false);
        let safe_count = ops::maximum(&pair_count, &Array::from_f32(1.0));
        let per_token = pair_sum.divide(&safe_count);

        reduce_per_token(&per_token, weights)
    }

    fn name(&self) -> &'static str {
        "logistic_ranking"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reversed_ranking_higher_loss() {
        let teacher = Array::from_f32_slice(&[10.0f32, 0.0, -10.0], &[1, 1, 3]);
        let student_good = Array::from_f32_slice(&[8.0f32, 0.0, -8.0], &[1, 1, 3]);
        let student_bad = Array::from_f32_slice(&[-10.0f32, 0.0, 10.0], &[1, 1, 3]);

        let loss = LogisticRankingLoss::new().with_top_k(3);
        let good = loss.compute(&teacher, &student_good, 1.0).unwrap();
        let bad = loss.compute(&teacher, &student_bad, 1.0).unwrap();
        good.eval();
        bad.eval();

        let good_val: f32 = good.item();
        let bad_val: f32 = bad.item();
        assert!(
            bad_val > good_val,
            "Reversed ranking → higher loss: good={good_val}, bad={bad_val}"
        );
    }
}
