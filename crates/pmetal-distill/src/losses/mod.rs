//! Loss functions for knowledge distillation.
//!
//! This module provides various loss functions used in knowledge distillation:
//! - KL Divergence (forward and reverse)
//! - Jensen-Shannon Divergence
//! - Soft Cross-Entropy
//! - MSE on logits
//! - Hidden state alignment losses
//!
//! # Differentiability
//!
//! Every loss here is built from MLX operations, so the value it returns is a
//! node in the autodiff graph: the student is trained by differentiating it.
//! A loss must never evaluate its inputs and hand back a scalar rebuilt from
//! host memory (or from a Metal buffer), because that scalar is a constant to
//! autodiff and the student would receive no gradient from it.
//!
//! # Example
//!
//! ```rust,ignore
//! use pmetal_distill::losses::{KlDivergenceLoss, DistillLoss};
//!
//! let loss = KlDivergenceLoss::new();
//! let result = loss.compute(&teacher_logits, &student_logits, 2.0)?;
//! ```

mod attention_transfer;
mod gkd;
pub mod hidden_state;
mod hinge_ranking;
mod jensen_shannon;
mod jsd_skewed;
mod kl_divergence;
mod logistic_ranking;
mod minillm;
mod mse;
mod skew_kl;
mod soft_cross_entropy;
mod sparse_kl;
mod tvd;
mod universal_logit;

pub use attention_transfer::AttentionTransferLoss;
pub use gkd::{GkdLoss, GreedySampler, OnPolicySampler, TemperatureSampler};
pub use hidden_state::HiddenStateLoss;
pub use hinge_ranking::HingeRankingLoss;
pub use jensen_shannon::JensenShannonLoss;
pub use jsd_skewed::JsdSkewedLoss;
pub use kl_divergence::KlDivergenceLoss;
pub use logistic_ranking::LogisticRankingLoss;
pub use minillm::{MiniLlmLoss, MiniLlmOutput};
pub use mse::MseLoss;
pub use skew_kl::SkewKlLoss;
pub use soft_cross_entropy::SoftCrossEntropyLoss;
pub use sparse_kl::SparseKlLoss;
pub use tvd::TvdLoss;
pub use universal_logit::UniversalLogitLoss;

use crate::Result;
use pmetal_runtime::compat::{Array, Dtype, ops};

/// Default number of top-k teacher tokens to use when vocab sizes differ.
///
/// 128 covers the vast majority of teacher probability mass in practice
/// (top-128 tokens typically account for >99.9% of mass in peaked distributions).
pub const SPARSE_TOPK_DEFAULT: i32 = 128;

/// Trait for distillation loss functions.
///
/// An implementation provides [`weighted_loss`](Self::weighted_loss); the
/// public entry points ([`compute`](Self::compute),
/// [`compute_weighted`](Self::compute_weighted),
/// [`compute_masked`](Self::compute_masked)) cut the teacher out of the
/// autodiff graph before calling it, so a teacher forward that ran inside the
/// differentiated function still contributes no gradient.
pub trait DistillLoss: Send + Sync {
    /// The loss between a teacher and a student, averaged over tokens.
    ///
    /// The teacher logits arrive already detached from the autodiff graph:
    /// treat them as constants. `weights` is one entry per token; the result
    /// is `sum(loss * w) / sum(w)`, which [`reduce_per_token`] computes.
    fn weighted_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array>;

    /// Compute the distillation loss between teacher and student outputs.
    ///
    /// # Arguments
    /// * `teacher_logits` - Logits from the teacher model
    /// * `student_logits` - Logits from the student model
    /// * `temperature` - Temperature for softmax scaling
    ///
    /// # Returns
    /// The computed loss value as a scalar array.
    fn compute(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        self.compute_weighted(teacher_logits, student_logits, temperature, None)
    }

    /// Compute weighted distillation loss.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher logits; detached here
    /// * `student_logits` - Student logits
    /// * `temperature` - Softmax temperature
    /// * `weights` - Per-token weights `[batch, seq]`
    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        self.weighted_loss(
            &teacher_logits.stop_gradient(),
            student_logits,
            temperature,
            weights,
        )
    }

    /// Compute token-level masked distillation loss.
    ///
    /// Tokens where `mask == 0` are excluded from the mean so that padding
    /// positions and special tokens do not dilute the gradient signal.
    ///
    /// The mask is the per-token weight of [`compute_weighted`](Self::compute_weighted),
    /// so the result is the mean over the unmasked tokens only.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher logits `[batch, seq, vocab]`
    /// * `student_logits` - Student logits `[batch, seq, vocab]`
    /// * `temperature` - Softmax temperature
    /// * `mask` - Binary mask `[batch, seq]` where 1 = include, 0 = exclude
    fn compute_masked(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        mask: &Array,
    ) -> Result<Array> {
        self.compute_weighted(teacher_logits, student_logits, temperature, Some(mask))
    }

    /// The factor [`Distiller`](crate::Distiller) multiplies this loss by.
    ///
    /// Softening both distributions by `T` shrinks a divergence's gradient
    /// with respect to the student logits by `1/T²` (Hinton et al., 2015,
    /// §2.1), and `T²` restores it. That holds for losses on the softened
    /// distributions, the default. A loss that does not depend on the
    /// temperature (a logit regression) has nothing to restore and returns
    /// `1.0`.
    fn temperature_scale(&self, temperature: f32) -> f32 {
        temperature * temperature
    }

    /// Get the name of this loss function.
    fn name(&self) -> &'static str;
}

/// Temperature-softened log-probabilities, `log_softmax(logits / T)`, in f32.
///
/// Logits are upcast before scaling so a bf16 student is normalized at full
/// precision; `log_softmax` subtracts the row's log-sum-exp rather than taking
/// the log of a softmax, so no probability underflows to `log(0)`.
pub(crate) fn tempered_log_probs(logits: &Array, temperature: f32) -> Array {
    logits
        .as_dtype(Dtype::Float32.as_i32())
        .divide(&Array::from_f32(temperature))
        .log_softmax(-1)
}

/// Temperature-softened f32 log-probabilities of a teacher/student pair, after
/// [`align_vocab_with_k`] has given them a common vocabulary axis.
pub(crate) fn aligned_log_probs(
    teacher_logits: &Array,
    student_logits: &Array,
    temperature: f32,
    sparse_top_k: i32,
) -> Result<(Array, Array)> {
    let (teacher, student, _) = align_vocab_with_k(teacher_logits, student_logits, sparse_top_k)?;
    Ok((
        tempered_log_probs(&teacher, temperature),
        tempered_log_probs(&student, temperature),
    ))
}

/// Numerically stable `log(exp(a) + exp(b))` = `m + log(exp(a - m) + exp(b - m))`
/// with `m = max(a, b)`.
///
/// `m` appears both outside and inside the log, so its own derivative cancels
/// and the gradient splits evenly between `a` and `b` where they tie, as the
/// derivative of `log(exp(a) + exp(b))` does. The shortcut
/// `m + log(1 + exp(-|a - b|))` has the same value but not that property: the
/// gradient of `maximum` sends a tie entirely to one argument and `|x|`
/// contributes nothing there, so wherever teacher and student gave a token the
/// same log-probability the student's gradient through the mixture came out
/// wrong (tenfold on a permuted-row fixture).
pub(crate) fn log_add_exp(a: &Array, b: &Array) -> Array {
    let m = ops::maximum(a, b);
    m.add(&a.subtract(&m).exp().add(&b.subtract(&m).exp()).log())
}

/// `log(w·P + (1-w)·Q)` from `log P` and `log Q`, for `0 < w < 1`.
pub(crate) fn log_mixture(log_p: &Array, log_q: &Array, weight_p: f32) -> Array {
    debug_assert!(weight_p > 0.0 && weight_p < 1.0);
    log_add_exp(
        &log_p.add(&Array::from_f32(weight_p.ln())),
        &log_q.add(&Array::from_f32((1.0 - weight_p).ln())),
    )
}

/// `KL(P || Q)` per row from `log P` and `log Q`, summed over the last axis.
pub(crate) fn kl_from_log_probs(log_p: &Array, log_q: &Array) -> Array {
    log_p
        .exp()
        .multiply(&log_p.subtract(log_q))
        .sum_axes(&[-1], false)
}

/// Reduce per-token losses to a scalar, in the autodiff graph.
///
/// Without `weights` this is the mean. With them it is
/// `sum(loss * w) / max(sum(w), 1e-8)`, where `weights` holds one entry per
/// token (typically `[batch, seq]`), so a 0/1 mask averages over the kept
/// tokens only.
pub(crate) fn reduce_per_token(per_token: &Array, weights: Option<&Array>) -> Result<Array> {
    let Some(w) = weights else {
        return Ok(per_token.mean_all());
    };
    if w.size() != per_token.size() {
        return Err(crate::DistillError::Other(format!(
            "weight tensor has {} elements but expected {} (one per token)",
            w.size(),
            per_token.size()
        )));
    }
    let w = w
        .reshape(per_token.shape())
        .as_dtype(Dtype::Float32.as_i32());
    let safe_weight = ops::maximum(&w.sum_all(), &Array::from_f32(1e-8));
    Ok(per_token.multiply(&w).sum_all().divide(&safe_weight))
}

/// Align teacher and student logits when their vocab sizes differ.
///
/// Uses the default top-k of [`SPARSE_TOPK_DEFAULT`] (128).  For a custom `k`,
/// call [`align_vocab_with_k`] directly.
///
/// When teacher and student have different vocabulary sizes, the losses compare
/// them on the teacher's `k` most likely token ids, renormalized on each side.
/// Ids are matched **by value**, so this is meaningful when the two models share
/// a tokenizer and differ only in how far the embedding matrix is padded, or in
/// a few added special tokens. Models with different tokenizers have unrelated
/// ids: use [`UniversalLogitLoss`], which compares order statistics instead.
/// For a pure padding difference [`align_vocab_truncate`] keeps the whole
/// distribution.
///
/// # Returns
///
/// `(teacher_aligned, student_aligned, mismatched)`.  `mismatched` is `true`
/// when vocab sizes differed and the sparse path was taken.
pub fn align_vocab(teacher_logits: &Array, student_logits: &Array) -> Result<(Array, Array, bool)> {
    align_vocab_with_k(teacher_logits, student_logits, SPARSE_TOPK_DEFAULT)
}

/// Align teacher and student logits when their vocab sizes differ, with a configurable top-k.
///
/// When teacher and student have different vocabulary sizes, the losses compare
/// them on the teacher's `k` most likely token ids, renormalized on each side.
/// Ids are matched **by value**, so this is meaningful when the two models share
/// a tokenizer and differ only in how far the embedding matrix is padded, or in
/// a few added special tokens. Models with different tokenizers have unrelated
/// ids: use [`UniversalLogitLoss`], which compares order statistics instead.
/// For a pure padding difference [`align_vocab_truncate`] keeps the whole
/// distribution.
///
/// # Algorithm
///
/// 1. Identify the teacher's top-k token indices along the last axis.
/// 2. Gather teacher logits at those k positions → shape `[..., k]`.
/// 3. Gather student logits at those same k positions → shape `[..., k]`.
///    Teacher indices that exceed the student vocab are masked to `-1e9` so
///    they contribute negligible softmax probability.
///
/// The returned pair has matching last dimensions so any loss function can be
/// applied without modification.
///
/// # Parameters
///
/// * `top_k` – number of teacher tokens to retain per position.  Clamped to
///   `teacher_vocab` if larger.  Must be ≥ 1.
///
/// # Returns
///
/// `(teacher_aligned, student_aligned, mismatched)`.  `mismatched` is `true`
/// when vocab sizes differed and the sparse path was taken.
pub fn align_vocab_with_k(
    teacher_logits: &Array,
    student_logits: &Array,
    top_k: i32,
) -> Result<(Array, Array, bool)> {
    // The top-k ids are a function of the teacher alone and must stay out of
    // the autodiff graph (MLX cannot differentiate a gather's indices).
    let teacher_logits = &teacher_logits.stop_gradient();
    let teacher_vocab = teacher_logits.dim(-1) as usize;
    let student_vocab = student_logits.dim(-1) as usize;

    if teacher_vocab == student_vocab {
        return Ok((teacher_logits.clone(), student_logits.clone(), false));
    }

    // argpartition requires kth < size; cap at teacher_vocab - 1 so we never
    // request kth == vocab_size (which would be out-of-range).
    let k = top_k
        .max(1)
        .min((teacher_vocab as i32).saturating_sub(1).max(1));

    tracing::debug!(
        teacher_vocab,
        student_vocab,
        k,
        "vocab mismatch: using sparse top-k distillation"
    );

    // Argpartition descending: negate teacher logits then argpartition so the
    // first k indices correspond to the k largest teacher logit positions.
    // O(V) vs O(V log V) for argsort — significant win at vocab size ~150k.
    let neg_teacher = teacher_logits.negative();
    let partitioned_indices = ops::argpartition_axis(&neg_teacher, k, -1);

    // Slice first k positions along the last axis.
    //
    // Reshape to 2D [N, vocab], slice to [N, k], reshape back to leading_dims + [k].
    // This is rank-agnostic and works correctly for tensors of any rank.
    let shape = partitioned_indices.shape().to_vec();
    let ndim = shape.len();
    let vocab_dim = shape[ndim - 1] as usize;
    let n: i32 = shape[..ndim - 1].iter().product();

    let part_2d = partitioned_indices.reshape(&[n, vocab_dim as i32]);
    let top_k_2d = part_2d.slice(&[0, 0], &[n, k]);
    // Rebuild leading shape + [k]
    let mut out_shape: Vec<i32> = shape[..ndim - 1].to_vec();
    out_shape.push(k);
    let top_k_indices = top_k_2d.reshape(&out_shape);

    // Gather teacher logits at the top-k token positions.
    let teacher_aligned = teacher_logits.take_along_axis(&top_k_indices, -1);

    // Clamp teacher indices to the student vocab range for gathering.
    // Out-of-range positions are masked below so clamping to 0 is safe here.
    let student_vocab_minus1 = Array::from_i32((student_vocab as i32) - 1);
    let zero = Array::from_i32(0);
    let clamped =
        ops::minimum(&top_k_indices, &student_vocab_minus1).as_dtype(Dtype::Int32.as_i32());
    let clamped = ops::maximum(&clamped, &zero);

    // Gather student logits at clamped positions.
    let student_gathered = student_logits.take_along_axis(&clamped, -1);

    // Mask out positions where the teacher index fell outside the student vocab.
    // Those tokens leave the support on BOTH sides (-1e9 gives them softmax
    // weight ~0): the teacher is renormalized over the ids the student can
    // score. Masking only the student left the teacher's probability on a
    // token the student gives exactly zero, a KL term of ~1e9 per unit of mass.
    let student_vocab_arr = Array::from_i32(student_vocab as i32);
    let out_of_range = top_k_indices
        .as_dtype(Dtype::Int32.as_i32())
        .greater_equal(&student_vocab_arr);
    let neg_large = Array::from_f32(-1e9_f32);
    let student_aligned = ops::where_fn(&out_of_range, &neg_large, &student_gathered);
    let teacher_aligned = ops::where_fn(&out_of_range, &neg_large, &teacher_aligned);

    Ok((teacher_aligned, student_aligned, true))
}

/// Cut both logit tensors to their common vocabulary, `[.., min(V_T, V_S)]`.
///
/// For models that share a tokenizer but pad the embedding matrix to
/// different sizes (the extra ids are never real tokens), this is exact, and
/// keeps the whole distribution where [`align_vocab`]'s top-k keeps only the
/// teacher's head. Do not use it for models with different tokenizers: the
/// ids mean different tokens there and nothing about their positions lines up
/// (see [`UniversalLogitLoss`] for that case).
pub fn align_vocab_truncate(teacher_logits: &Array, student_logits: &Array) -> (Array, Array) {
    let common = teacher_logits.dim(-1).min(student_logits.dim(-1));
    let cut = |logits: &Array| {
        let mut stop = logits.shape().to_vec();
        let last = stop.len() - 1;
        stop[last] = common;
        logits.slice(&vec![0; stop.len()], &stop)
    };
    (cut(teacher_logits), cut(student_logits))
}

/// Softmax along specified axis.
pub fn softmax(x: &Array, axis: i32) -> Result<Array> {
    Ok(x.softmax(axis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_softmax() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[1, 3]);
        let probs = softmax(&logits, -1).unwrap();
        let probs_data: Vec<f32> = probs.clone().to_f32_vec(3).unwrap();

        // Check probabilities sum to 1
        let sum: f32 = probs_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check relative ordering
        assert!(probs_data[2] > probs_data[1]);
        assert!(probs_data[1] > probs_data[0]);
    }

    #[test]
    #[serial]
    fn test_log_softmax() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[1, 3]);
        let log_probs = logits.log_softmax(-1);
        let log_probs_data: Vec<f32> = log_probs.clone().to_f32_vec(3).unwrap();

        // All log probs should be <= 0
        for lp in &log_probs_data {
            assert!(*lp <= 0.0);
        }

        // exp(log_softmax) should equal softmax
        let probs = log_probs.exp();
        let probs_data: Vec<f32> = probs.clone().to_f32_vec(3).unwrap();
        let sum: f32 = probs_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------
    // align_vocab / cross-vocab tests
    // -----------------------------------------------------------------

    /// Same vocab size → passthrough, no copy.
    #[test]
    #[serial]
    fn test_align_vocab_same_size_passthrough() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let (t, s, mismatched) = align_vocab(&logits, &logits).unwrap();
        assert!(!mismatched, "same vocab should not be mismatched");
        // Shapes preserved
        assert_eq!(t.shape(), logits.shape());
        assert_eq!(s.shape(), logits.shape());
    }

    /// Teacher vocab > student vocab (the Qwen3-4B → Qwen3.5-0.8B scenario).
    /// After alignment both tensors must have the same last dimension (k).
    #[test]
    #[serial]
    fn test_align_vocab_teacher_larger_2d() {
        // teacher: [1, 8], student: [1, 6]  (teacher larger)
        let teacher_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let student_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[1, 8]);
        let student = Array::from_f32_slice(&student_data, &[1, 6]);

        let k = 4_i32;
        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval();
        sa.eval();

        assert!(mismatched);
        assert_eq!(ta.dim(-1), k);
        assert_eq!(sa.dim(-1), k);
        // Teacher leading dims preserved
        assert_eq!(ta.shape()[..ta.shape().len() - 1], [1]);
    }

    /// Student vocab > teacher vocab.
    #[test]
    #[serial]
    fn test_align_vocab_student_larger_2d() {
        // teacher: [1, 6], student: [1, 8]  (student larger)
        let teacher_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let student_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[1, 6]);
        let student = Array::from_f32_slice(&student_data, &[1, 8]);

        let k = 4_i32;
        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval();
        sa.eval();

        assert!(mismatched);
        assert_eq!(ta.dim(-1), k);
        assert_eq!(sa.dim(-1), k);
    }

    /// 3-D tensors [batch, seq, vocab] — the common real-world shape.
    /// This is the critical regression test: the previous `(.., ..k)` indexing
    /// would slice the sequence dimension, producing wrong shapes.
    #[test]
    #[serial]
    fn test_align_vocab_3d_batch_seq_vocab() {
        let batch = 2_i32;
        let seq = 3_i32;
        let teacher_vocab = 10_i32;
        let student_vocab = 8_i32;
        let k = 4_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| i as f32)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| i as f32)
            .collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_f32_slice(&student_data, &[batch, seq, student_vocab]);

        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval();
        sa.eval();

        assert!(mismatched);
        // Shape must be [batch, seq, k] — NOT [batch, k, teacher_vocab]
        assert_eq!(
            ta.shape(),
            &[batch, seq, k],
            "teacher_aligned shape wrong: {:?}",
            ta.shape()
        );
        assert_eq!(
            sa.shape(),
            &[batch, seq, k],
            "student_aligned shape wrong: {:?}",
            sa.shape()
        );
    }

    /// Verify that top-k indices actually correspond to the largest teacher logits.
    #[test]
    #[serial]
    fn test_align_vocab_selects_top_logits() {
        // teacher: [1, 6] with clear ordering
        // token 5 has highest logit (10.0), token 4 second (9.0), etc.
        let teacher_data = [1.0_f32, 2.0, 3.0, 4.0, 9.0, 10.0];
        let student_data = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // 8 tokens
        let teacher = Array::from_f32_slice(&teacher_data, &[1, 6]);
        let student = Array::from_f32_slice(&student_data, &[1, 8]);

        let (ta, sa, _) = align_vocab_with_k(&teacher, &student, 3).unwrap();
        ta.eval();

        let mut ta_vals: Vec<f32> = ta.clone().to_f32_vec(3).unwrap();
        ta_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(ta_vals, vec![10.0, 9.0, 4.0]);

        // The student is read at the same three ids (5, 4, 3): 0.6, 0.5, 0.4.
        let mut sa_vals: Vec<f32> = sa.clone().to_f32_vec(3).unwrap();
        sa_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for (got, want) in sa_vals.iter().zip([0.6, 0.5, 0.4]) {
            assert!((got - want).abs() < 1e-6, "{sa_vals:?}");
        }
    }

    /// Positions where the teacher index falls outside the student vocab must be
    /// masked to -1e9 in the student-aligned tensor.
    #[test]
    #[serial]
    fn test_align_vocab_out_of_range_masked() {
        // teacher has 8 tokens; student has only 4.
        // The highest teacher logits (indices 7, 6, 5, 4) all exceed student_vocab=4,
        // so all 4 selected student positions must be masked.
        let teacher_data = [1.0_f32, 1.0, 1.0, 1.0, 5.0, 6.0, 7.0, 8.0]; // indices 4-7 top
        let student_data = [0.1_f32, 0.2, 0.3, 0.4];
        let teacher = Array::from_f32_slice(&teacher_data, &[1, 8]);
        let student = Array::from_f32_slice(&student_data, &[1, 4]);

        let (_ta, sa, _) = align_vocab_with_k(&teacher, &student, 4).unwrap();
        sa.eval();

        let sa_vals: Vec<f32> = sa.clone().to_f32_vec(4).unwrap();
        // Every entry must be ≤ -1e8 (masked)
        for v in &sa_vals {
            assert!(*v <= -1e8, "expected masked value (-1e9), got {}", v);
        }
    }

    /// Simulate the real cross-architecture case: teacher_vocab=151936, student_vocab=152080.
    /// Here student_vocab > teacher_vocab so no masking needed.
    /// Verifies the function completes without panic and returns sensible shapes.
    #[test]
    #[serial]
    fn test_align_vocab_qwen3_to_qwen35_shapes() {
        // Use small stand-ins for the real vocab sizes but with the same relationship
        // (student > teacher), scaled down for test speed.
        let teacher_vocab = 120_i32;
        let student_vocab = 130_i32;
        let batch = 1_i32;
        let seq = 2_i32;
        let k = 32_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| (i % 50) as f32 - 25.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| (i % 50) as f32 - 25.0)
            .collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_f32_slice(&student_data, &[batch, seq, student_vocab]);

        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval();
        sa.eval();

        assert!(mismatched);
        assert_eq!(ta.shape(), &[batch, seq, k]);
        assert_eq!(sa.shape(), &[batch, seq, k]);

        // All teacher-aligned values must be finite
        let ta_vals: Vec<f32> = ta.clone().to_f32_vec(64).unwrap();
        assert!(ta_vals.iter().all(|v| v.is_finite()));

        // student-aligned: indices < student_vocab so no masking expected
        let sa_vals: Vec<f32> = sa.clone().to_f32_vec(64).unwrap();
        assert!(sa_vals.iter().all(|v| v.is_finite()));
    }
}
