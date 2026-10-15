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

use crate::offline::{SparseLogits, SparseTargets};
use crate::{DistillError, Result};
use pmetal_runtime::compat::{Array, Dtype, ops};

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
        check_pair(
            teacher.shape(),
            teacher.vocab_size(),
            student_logits.shape(),
        )?;
        let f32_ = Dtype::Float32.as_i32();
        // log p over the kept tokens; the teacher is a constant.
        let log_p = teacher
            .values()
            .as_dtype(f32_)
            .multiply(&Array::from_f32(1.0 / temperature.max(1e-6)))
            .stop_gradient()
            .log_softmax(-1);
        let p = log_p.exp();
        kl_to_sparse(
            &p,
            &log_p,
            teacher.indices(),
            student_logits,
            temperature,
            weights,
        )
    }

    /// The same loss against sampled draws ([`SparseTargets`], Random-Sampling
    /// KD): `KL(empirical || student)` where the empirical distribution is the
    /// frequencies of the drawn tokens.
    ///
    /// The draws were taken from `softmax(logits / T_cache)`, so `temperature`
    /// must be the temperature they were drawn at. Its expected gradient is the
    /// full distillation gradient `(q − p)/T`: an unbiased estimate where the
    /// top-k's is biased. (The value differs from the full KL by the entropy
    /// of the empirical distribution, which does not move the student.)
    pub fn compute_sampled(
        &self,
        teacher: &SparseTargets,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        check_pair(
            teacher.shape(),
            teacher.vocab_size(),
            student_logits.shape(),
        )?;
        if (temperature - teacher.temperature()).abs() > 1e-6 * temperature.abs().max(1.0) {
            return Err(DistillError::InvalidConfig(format!(
                "the sampled teacher was drawn at temperature {} but the loss uses {temperature}; \
                 the draws estimate the teacher at their own temperature only",
                teacher.temperature()
            )));
        }
        let p = teacher.probs().as_dtype(Dtype::Float32.as_i32());
        // Unused slots have frequency 0: their log is clamped so 0 * log 0 is
        // exactly 0 rather than NaN.
        let log_p = ops::maximum(&p, &Array::from_f32(f32::MIN_POSITIVE)).log();
        kl_to_sparse(
            &p,
            &log_p,
            teacher.indices(),
            student_logits,
            temperature,
            weights,
        )
    }
}

/// The teacher's `[.., k]` layout must sit on the student's leading axes and
/// vocabulary.
fn check_pair(teacher_shape: &[i32], teacher_vocab: usize, student_shape: &[i32]) -> Result<()> {
    let leading = teacher_shape.len() - 1;
    if student_shape.len() != teacher_shape.len()
        || student_shape[..leading] != teacher_shape[..leading]
    {
        return Err(DistillError::ShapeMismatch {
            teacher: teacher_shape.to_vec(),
            student: student_shape.to_vec(),
        });
    }
    let vocab = student_shape[leading] as usize;
    if vocab != teacher_vocab {
        return Err(DistillError::VocabMismatch {
            teacher: teacher_vocab,
            student: vocab,
        });
    }
    Ok(())
}

/// `Σ_i p_i · (log p_i − log q_{idx_i})` per token, averaged by `weights`:
/// `log q` at the kept tokens is the student's scaled logit there minus its
/// full-vocabulary log-sum-exp.
fn kl_to_sparse(
    p: &Array,
    log_p: &Array,
    indices: &Array,
    student_logits: &Array,
    temperature: f32,
    weights: Option<&Array>,
) -> Result<Array> {
    let scaled = student_logits
        .as_dtype(Dtype::Float32.as_i32())
        .multiply(&Array::from_f32(1.0 / temperature.max(1e-6)));
    let lse = scaled.logsumexp(-1, true);
    let log_q = scaled
        .take_along_axis(&indices.stop_gradient(), -1)
        .subtract(&lse);
    let per_token = p.multiply(&log_p.subtract(&log_q)).sum_axes(&[-1], false);
    reduce_per_token(&per_token, weights)
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
