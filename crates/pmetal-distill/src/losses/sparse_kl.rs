//! Forward KL against a top-k teacher, without materializing the teacher.
//!
//! An offline cache stores only the teacher's `k` largest logits per token
//! ([`SparseLogits`]). The teacher distribution it stands for is the softmax
//! of those `k` logits among themselves (every other token has probability
//! zero), at the loss's temperature `T`:
//!
//! ```text
//! p_i = softmax(v / T)_i,    i over the k kept tokens
//! KL(p || q) = Σ_i p_i · (log p_i - log q_{idx_i})
//! ```
//!
//! with `q = softmax(z / T)` the student over its **full** vocabulary. The
//! student's mass outside the teacher's top-k is therefore penalized through
//! the normalizer, and nothing outside it enters the sum. It is the same value
//! (and gradient) as running [`KlDivergenceLoss`](super::KlDivergenceLoss)
//! on the dense `[tokens, vocab]` tensor with the dropped logits at `-1e10`,
//! but the teacher side costs `k` values per token instead of `vocab`, and
//! the student side one log-sum-exp and a gather of `k` entries: the
//! `[tokens, vocab]` log-probability tensors are never formed.
//!
//! Truncating the teacher to its top-k is a biased estimate of the full
//! teacher; at `k = 128` it keeps the mass that matters for peaked LM
//! distributions.

use crate::offline::SparseLogits;
use crate::{DistillError, Result};
use pmetal_runtime::compat::{Array, Dtype};

use super::reduce_per_token;

/// Forward KL `KL(teacher_topk || student)` against a [`SparseLogits`] teacher.
#[derive(Debug, Clone, Copy, Default)]
pub struct SparseKlLoss;

impl SparseKlLoss {
    /// A sparse forward-KL loss.
    pub fn new() -> Self {
        Self
    }

    /// The mean per-token loss.
    ///
    /// * `teacher` - `[.., k]` kept logits and ids
    /// * `student_logits` - `[.., vocab]`, the same leading axes and the
    ///   teacher's vocabulary
    /// * `temperature` - softens both distributions
    /// * `weights` - optional per-token weights (a 0/1 mask), one per token
    pub fn compute(
        &self,
        teacher: &SparseLogits,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        let t_shape = teacher.shape();
        let s_shape = student_logits.shape();
        let leading = t_shape.len() - 1;
        if s_shape.len() != t_shape.len() || s_shape[..leading] != t_shape[..leading] {
            return Err(DistillError::ShapeMismatch {
                teacher: t_shape.to_vec(),
                student: s_shape.to_vec(),
            });
        }
        let vocab = s_shape[leading] as usize;
        if vocab != teacher.vocab_size() {
            return Err(DistillError::VocabMismatch {
                teacher: teacher.vocab_size(),
                student: vocab,
            });
        }

        let f32_ = Dtype::Float32.as_i32();
        let inv_t = Array::from_f32(1.0 / temperature.max(1e-6));

        // log p over the kept tokens; the teacher is a constant.
        let log_p = teacher
            .values()
            .as_dtype(f32_)
            .multiply(&inv_t)
            .stop_gradient()
            .log_softmax(-1);

        // log q at the kept tokens: gather the scaled logits, subtract the
        // full-vocabulary log-sum-exp.
        let scaled = student_logits.as_dtype(f32_).multiply(&inv_t);
        let lse = scaled.logsumexp(-1, true);
        let log_q = scaled
            .take_along_axis(&teacher.indices().stop_gradient(), -1)
            .subtract(&lse);

        let per_token = log_p
            .exp()
            .multiply(&log_p.subtract(&log_q))
            .sum_axes(&[-1], false);
        reduce_per_token(&per_token, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::{DistillLoss, KlDivergenceLoss};
    use serial_test::serial;

    fn sparse(values: &[f32], indices: &[i32], tokens: i32, k: i32, vocab: usize) -> SparseLogits {
        SparseLogits::new(
            Array::from_f32_slice(values, &[1, tokens, k]),
            Array::from_i32_slice_shaped(indices, &[1, tokens, k]),
            vocab,
        )
        .unwrap()
    }

    /// The sparse loss is the dense loss on the zero-filled teacher.
    #[test]
    #[serial]
    fn matches_the_dense_loss_on_the_densified_teacher() {
        let teacher = sparse(
            &[3.0, 1.0, -0.5, 2.0, 2.0, 0.0],
            &[4, 0, 2, 1, 3, 5],
            2,
            3,
            6,
        );
        let student = Array::from_f32_slice(
            &[
                0.3, 0.9, -0.7, 1.4, 0.2, 1.1, -0.5, 0.6, 0.2, -1.3, 0.4, 0.8,
            ],
            &[1, 2, 6],
        );
        for temperature in [1.0_f32, 2.0] {
            let sparse_loss: f32 = SparseKlLoss::new()
                .compute(&teacher, &student, temperature, None)
                .unwrap()
                .item();
            let dense_loss: f32 = KlDivergenceLoss::new()
                .compute(&teacher.to_dense(), &student, temperature)
                .unwrap()
                .item();
            assert!(
                (sparse_loss - dense_loss).abs() <= 1e-5 * dense_loss.abs().max(1e-3),
                "T={temperature}: sparse {sparse_loss} vs dense {dense_loss}"
            );
        }
    }

    #[test]
    #[serial]
    fn rejects_a_student_with_another_vocabulary() {
        let teacher = sparse(&[1.0, 0.0], &[0, 1], 1, 2, 6);
        let student = Array::from_f32_slice(&[0.0; 5], &[1, 1, 5]);
        assert!(
            SparseKlLoss::new()
                .compute(&teacher, &student, 1.0, None)
                .is_err()
        );
    }
}
