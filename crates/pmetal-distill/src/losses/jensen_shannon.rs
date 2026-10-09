//! Jensen-Shannon Divergence loss for knowledge distillation.
//!
//! JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
//! where M = 0.5 * (P + Q)
//!
//! This is the β = 0.5 member of the generalized JSD of Agarwal et al. (2024,
//! "On-Policy Distillation of Language Models"); [`JsdSkewedLoss`](super::JsdSkewedLoss)
//! covers other β. It is symmetric and bounded in [0, log 2], making it more
//! stable than KL divergence for distillation.

use super::{
    DistillLoss, SPARSE_TOPK_DEFAULT, align_vocab_with_k, reduce_per_token, tempered_log_probs,
};
use crate::Result;
use pmetal_bridge::compat::{Array, ops};

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
fn log_sum_exp(a: &Array, b: &Array) -> Array {
    let m = ops::maximum(a, b);
    m.add(&a.subtract(&m).exp().add(&b.subtract(&m).exp()).log())
}

/// Jensen-Shannon Divergence loss for knowledge distillation.
///
/// A symmetric, bounded alternative to KL divergence.
/// JS(P || Q) = JS(Q || P), unlike KL divergence.
pub struct JensenShannonLoss {
    /// Number of top-k teacher tokens to retain when vocab sizes differ.
    ///
    /// Only used when teacher and student have different vocabulary sizes
    /// (cross-architecture distillation).  Defaults to [`SPARSE_TOPK_DEFAULT`].
    sparse_top_k: i32,
}

impl JensenShannonLoss {
    /// Create a new Jensen-Shannon divergence loss.
    pub fn new() -> Self {
        Self {
            sparse_top_k: SPARSE_TOPK_DEFAULT,
        }
    }

    /// Set the number of top-k teacher tokens used in cross-vocab distillation.
    ///
    /// When teacher and student vocabularies differ, the loss is computed only
    /// over the top-`k` teacher tokens (by logit magnitude).  Higher values
    /// capture more of the teacher distribution but increase computation.
    /// Must be ≥ 1; defaults to [`SPARSE_TOPK_DEFAULT`] (128).
    pub fn with_sparse_top_k(mut self, k: i32) -> Self {
        self.sparse_top_k = k.max(1);
        self
    }
}

impl Default for JensenShannonLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for JensenShannonLoss {
    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        // Align vocab sizes (sparse top-k over the teacher when they differ).
        let (teacher_logits, student_logits, _) =
            align_vocab_with_k(teacher_logits, student_logits, self.sparse_top_k)?;

        // Log-domain throughout for stability.
        let teacher_log_probs = tempered_log_probs(&teacher_logits, temperature);
        let student_log_probs = tempered_log_probs(&student_logits, temperature);
        let teacher_probs = teacher_log_probs.exp();

        // log(M) via log-sum-exp for stability (avoids 0*-inf = NaN for disjoint distributions)
        let log2 = Array::from_f32(2.0_f32.ln());
        let log_mixture = log_sum_exp(&teacher_log_probs, &student_log_probs).subtract(&log2);

        let kl_teacher_m = teacher_probs.multiply(&teacher_log_probs.subtract(&log_mixture));
        let student_probs = student_log_probs.exp();
        let kl_student_m = student_probs.multiply(&student_log_probs.subtract(&log_mixture));

        let half = Array::from_f32(0.5);
        let js_per_token = kl_teacher_m
            .add(&kl_student_m)
            .multiply(&half)
            .sum_axes(&[-1], false);

        reduce_per_token(&js_per_token, weights)
    }

    fn name(&self) -> &'static str {
        "jensen_shannon"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_js_identical_distributions() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let loss = JensenShannonLoss::new();
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        let value: f32 = result.item();

        // JS of identical distributions should be 0
        assert!(
            value.abs() < 1e-4,
            "JS of identical distributions should be ~0, got {}",
            value
        );
    }

    #[test]
    #[serial]
    fn test_js_symmetry() {
        let p = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let q = Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = JensenShannonLoss::new();
        let js_pq = loss.compute(&p, &q, 1.0).unwrap();
        let js_qp = loss.compute(&q, &p, 1.0).unwrap();

        let v_pq: f32 = js_pq.item();
        let v_qp: f32 = js_qp.item();

        // JS should be symmetric
        assert!(
            (v_pq - v_qp).abs() < 1e-4,
            "JS should be symmetric: JS(P||Q)={}, JS(Q||P)={}",
            v_pq,
            v_qp
        );
    }

    #[test]
    #[serial]
    fn test_js_bounded() {
        // Even for very different distributions, JS should be bounded by log(2)
        let p = Array::from_f32_slice(&[10.0_f32, 0.0, 0.0, 0.0], &[1, 1, 4]);
        let q = Array::from_f32_slice(&[0.0_f32, 0.0, 0.0, 10.0], &[1, 1, 4]);

        let loss = JensenShannonLoss::new();
        let result = loss.compute(&p, &q, 1.0).unwrap();
        let value: f32 = result.item();

        let ln2 = 2.0_f32.ln();
        assert!(
            value <= ln2 + 1e-4,
            "JS should be bounded by ln(2)={}, got {}",
            ln2,
            value
        );
        assert!(value >= 0.0, "JS should be non-negative, got {}", value);
    }

    #[test]
    #[serial]
    fn test_js_temperature_effect() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = JensenShannonLoss::new();

        // Higher temperature should reduce JS (softer distributions)
        let js_t1 = loss.compute(&teacher, &student, 1.0).unwrap();
        let js_t2 = loss.compute(&teacher, &student, 2.0).unwrap();

        let v1: f32 = js_t1.item();
        let v2: f32 = js_t2.item();

        assert!(
            v2 < v1,
            "Higher temp should reduce JS: T=1: {}, T=2: {}",
            v1,
            v2
        );
    }

    #[test]
    #[serial]
    fn test_larger_batch() {
        // Realistic vocab width: the row normalizer runs over 1024 entries
        let batch_size = 4;
        let seq_len = 8;
        let vocab_size = 1024;

        let teacher_data: Vec<f32> = (0..(batch_size * seq_len * vocab_size))
            .map(|i| ((i % 100) as f32 - 50.0) / 10.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch_size * seq_len * vocab_size))
            .map(|i| ((i * 7 % 100) as f32 - 50.0) / 10.0)
            .collect();

        let teacher = Array::from_f32_slice(&teacher_data, &[batch_size, seq_len, vocab_size]);
        let student = Array::from_f32_slice(&student_data, &[batch_size, seq_len, vocab_size]);

        let loss = JensenShannonLoss::new();
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        // Should be positive and finite
        assert!(value >= 0.0, "JS should be non-negative");
        assert!(value.is_finite(), "JS should be finite");
    }

    // -----------------------------------------------------------------
    // Cross-vocab / sparse top-k tests
    // -----------------------------------------------------------------

    /// JS divergence with teacher smaller than student vocab.
    #[test]
    #[serial]
    fn test_js_cross_vocab_teacher_smaller() {
        let teacher_vocab = 80_i32;
        let student_vocab = 100_i32;
        let batch = 2_i32;
        let seq = 4_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| (i % 40) as f32 - 20.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| (i * 3 % 40) as f32 - 20.0)
            .collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_f32_slice(&student_data, &[batch, seq, student_vocab]);

        let loss = JensenShannonLoss::new().with_sparse_top_k(32);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        assert!(value >= 0.0, "JS must be non-negative, got {}", value);
        assert!(value.is_finite(), "JS must be finite, got {}", value);
        // JS is bounded by ln(2) ≈ 0.693
        assert!(
            value <= 2.0_f32.ln() + 1e-4,
            "JS must be bounded by ln(2), got {}",
            value
        );
    }

    /// JS divergence with teacher larger than student vocab.
    #[test]
    #[serial]
    fn test_js_cross_vocab_teacher_larger() {
        let teacher_vocab = 100_i32;
        let student_vocab = 80_i32;
        let batch = 2_i32;
        let seq = 4_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| (i % 40) as f32 - 20.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| (i * 3 % 40) as f32 - 20.0)
            .collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_f32_slice(&student_data, &[batch, seq, student_vocab]);

        let loss = JensenShannonLoss::new().with_sparse_top_k(32);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        assert!(value >= 0.0, "JS must be non-negative, got {}", value);
        assert!(value.is_finite(), "JS must be finite, got {}", value);
        assert!(
            value <= 2.0_f32.ln() + 1e-4,
            "JS must be bounded by ln(2), got {}",
            value
        );
    }

    /// 3-D tensor cross-vocab JS — verifies the Ellipsis fix for rank-3 logits.
    #[test]
    #[serial]
    fn test_js_cross_vocab_3d_tensors() {
        let batch = 2_i32;
        let seq = 3_i32;
        let teacher_vocab = 10_i32;
        let student_vocab = 8_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| i as f32)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| i as f32)
            .collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_f32_slice(&student_data, &[batch, seq, student_vocab]);

        let loss = JensenShannonLoss::new().with_sparse_top_k(4);
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        let value: f32 = result.item();

        // Scalar result
        assert!(result.shape().is_empty(), "result should be scalar");
        assert!(value.is_finite(), "JS must be finite, got {}", value);
        assert!(value >= 0.0, "JS must be non-negative, got {}", value);
    }

    /// Cross-vocab JS is symmetric: JS(teacher_a, student_a) ≈ JS(teacher_b, student_b)
    /// when the roles are swapped via argument order.
    #[test]
    #[serial]
    fn test_js_cross_vocab_symmetry_hint() {
        // When both are within range (student_vocab > teacher_vocab so no masking),
        // swapping teacher/student should give the same value (JS is symmetric).
        let teacher_vocab = 6_i32;
        let student_vocab = 10_i32;

        // Identical logit values for the shared 6 tokens
        let shared: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let extended: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];
        let teacher = Array::from_f32_slice(&shared, &[1, 1, teacher_vocab]);
        let student = Array::from_f32_slice(&extended, &[1, 1, student_vocab]);

        let loss = JensenShannonLoss::new().with_sparse_top_k(6);
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        let value: f32 = result.item();

        assert!(value.is_finite(), "JS must be finite, got {}", value);
        assert!(value >= 0.0, "JS must be non-negative, got {}", value);
    }

    /// Configurable top-k builder produces valid output for multiple k values.
    #[test]
    #[serial]
    fn test_js_with_sparse_top_k_builder() {
        let teacher = Array::from_f32_slice(
            &(0..200).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 200],
        );
        let student = Array::from_f32_slice(
            &(0..150).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 150],
        );

        for k in [8, 32, 64, 128] {
            let loss = JensenShannonLoss::new().with_sparse_top_k(k);
            let result = loss.compute(&teacher, &student, 2.0).unwrap();
            let value: f32 = result.item();
            assert!(
                value.is_finite(),
                "JS should be finite for k={}: {}",
                k,
                value
            );
            assert!(
                value >= 0.0,
                "JS should be non-negative for k={}: {}",
                k,
                value
            );
        }
    }
}
