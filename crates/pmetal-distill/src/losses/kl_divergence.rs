//! KL Divergence loss for knowledge distillation.
//!
//! KL(P || Q) = sum(P * log(P / Q))
//!
//! In distillation:
//! - Forward KL: KL(teacher || student) - mode-covering (Hinton et al., 2015)
//! - Reverse KL: KL(student || teacher) - mode-seeking (Gu et al., 2023)
//!
//! Both are taken between the temperature-softened distributions
//! `softmax(logits / T)`, computed in log space. The `T²` factor of Hinton et
//! al. §2 is applied once, by [`Distiller::compute_loss`](crate::Distiller::compute_loss),
//! not here, so the value is the divergence itself.

use super::{
    DistillLoss, SPARSE_TOPK_DEFAULT, align_vocab_with_k, reduce_per_token, tempered_log_probs,
};
use crate::Result;
use pmetal_bridge::compat::Array;

/// KL Divergence loss for knowledge distillation.
///
/// Computes either forward KL (teacher || student) or reverse KL (student || teacher).
/// Forward KL encourages the student to cover all modes of the teacher distribution.
/// Reverse KL encourages the student to match the dominant modes.
pub struct KlDivergenceLoss {
    /// Whether to use reverse KL (student || teacher).
    reverse: bool,

    /// Number of top-k teacher tokens to retain when vocab sizes differ.
    ///
    /// Only used when teacher and student have different vocabulary sizes
    /// (cross-architecture distillation).  Defaults to [`SPARSE_TOPK_DEFAULT`].
    sparse_top_k: i32,
}

impl KlDivergenceLoss {
    /// Create a new KL divergence loss (forward by default).
    pub fn new() -> Self {
        Self {
            reverse: false,
            sparse_top_k: SPARSE_TOPK_DEFAULT,
        }
    }

    /// Create a reverse KL divergence loss.
    pub fn reverse() -> Self {
        Self {
            reverse: true,
            sparse_top_k: SPARSE_TOPK_DEFAULT,
        }
    }

    /// Set whether to use reverse KL.
    pub fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
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

impl Default for KlDivergenceLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for KlDivergenceLoss {
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

        let teacher_log_probs = tempered_log_probs(&teacher_logits, temperature);
        let student_log_probs = tempered_log_probs(&student_logits, temperature);

        let kl_per_token = if self.reverse {
            let log_ratio = student_log_probs.subtract(&teacher_log_probs);
            student_log_probs
                .exp()
                .multiply(&log_ratio)
                .sum_axes(&[-1], false)
        } else {
            let log_ratio = teacher_log_probs.subtract(&student_log_probs);
            teacher_log_probs
                .exp()
                .multiply(&log_ratio)
                .sum_axes(&[-1], false)
        };

        reduce_per_token(&kl_per_token, weights)
    }

    fn name(&self) -> &'static str {
        if self.reverse {
            "reverse_kl_divergence"
        } else {
            "kl_divergence"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_kl_identical_distributions() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let loss = KlDivergenceLoss::new();
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        let value: f32 = result.item();

        // KL divergence of identical distributions should be 0
        assert!(
            value.abs() < 1e-4,
            "KL of identical distributions should be ~0, got {}",
            value
        );
    }

    #[test]
    #[serial]
    fn test_kl_different_distributions() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = KlDivergenceLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        let value: f32 = result.item();

        // KL divergence should be positive
        assert!(
            value > 0.0,
            "KL divergence should be positive, got {}",
            value
        );
    }

    #[test]
    #[serial]
    fn test_kl_temperature_effect() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = KlDivergenceLoss::new();

        // Higher temperature should reduce KL (softer distributions are more similar)
        let kl_t1 = loss.compute(&teacher, &student, 1.0).unwrap();
        let kl_t2 = loss.compute(&teacher, &student, 2.0).unwrap();
        let kl_t4 = loss.compute(&teacher, &student, 4.0).unwrap();

        let v1: f32 = kl_t1.item();
        let v2: f32 = kl_t2.item();
        let v4: f32 = kl_t4.item();

        assert!(
            v2 < v1,
            "Higher temp should reduce KL: T=1: {}, T=2: {}",
            v1,
            v2
        );
        assert!(
            v4 < v2,
            "Higher temp should reduce KL: T=2: {}, T=4: {}",
            v2,
            v4
        );
    }

    #[test]
    #[serial]
    fn test_forward_vs_reverse_kl() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let forward = KlDivergenceLoss::new();
        let reverse = KlDivergenceLoss::reverse();

        let fwd_loss = forward.compute(&teacher, &student, 1.0).unwrap();
        let rev_loss = reverse.compute(&teacher, &student, 1.0).unwrap();

        let fwd_val: f32 = fwd_loss.item();
        let rev_val: f32 = rev_loss.item();

        // Both should be positive
        assert!(fwd_val > 0.0);
        assert!(rev_val > 0.0);
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

        let teacher = Array::from_f32_slice(
            &teacher_data,
            &[batch_size as i32, seq_len as i32, vocab_size as i32],
        );
        let student = Array::from_f32_slice(
            &student_data,
            &[batch_size as i32, seq_len as i32, vocab_size as i32],
        );

        let loss = KlDivergenceLoss::new();
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        // Should be positive and finite
        assert!(value > 0.0, "KL should be positive");
        assert!(value.is_finite(), "KL should be finite");
    }

    // -----------------------------------------------------------------
    // Cross-vocab / sparse top-k tests
    // -----------------------------------------------------------------

    /// KL divergence with mismatched vocab (teacher smaller than student).
    /// Mirrors Qwen3-4B (151936) → Qwen3.5-0.8B (152080).
    #[test]
    #[serial]
    fn test_kl_cross_vocab_teacher_smaller() {
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

        let loss = KlDivergenceLoss::new().with_sparse_top_k(32);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        assert!(value >= 0.0, "KL must be non-negative, got {}", value);
        assert!(value.is_finite(), "KL must be finite, got {}", value);
    }

    /// KL divergence with mismatched vocab (teacher larger than student).
    #[test]
    #[serial]
    fn test_kl_cross_vocab_teacher_larger() {
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

        let loss = KlDivergenceLoss::new().with_sparse_top_k(32);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        assert!(value >= 0.0, "KL must be non-negative, got {}", value);
        assert!(value.is_finite(), "KL must be finite, got {}", value);
    }

    /// Cross-vocab KL should return higher loss when distributions differ.
    #[test]
    #[serial]
    fn test_kl_cross_vocab_ordered_distributions() {
        // teacher: vocab=6, student: vocab=4
        // Same ordering: teacher top tokens overlap with student → lower KL
        // vs. completely inverted: → higher KL
        let teacher_same = Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0, 0.5, 0.1], &[1, 1, 6]);
        let teacher_inv = Array::from_f32_slice(&[0.1_f32, 0.5, 1.0, 2.0, 3.0, 4.0], &[1, 1, 6]);
        let student = Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = KlDivergenceLoss::new().with_sparse_top_k(4);
        let kl_same = loss
            .compute(&teacher_same, &student, 1.0)
            .unwrap()
            .item::<f32>();
        let kl_inv = loss
            .compute(&teacher_inv, &student, 1.0)
            .unwrap()
            .item::<f32>();

        assert!(kl_same.is_finite());
        assert!(kl_inv.is_finite());
        // The inverted teacher puts all mass on tokens 4-5 which are out-of-range
        // for the 4-token student, so the student cannot match it → higher KL.
        assert!(
            kl_inv >= kl_same,
            "inverted teacher should give >= KL vs aligned teacher: inv={}, same={}",
            kl_inv,
            kl_same
        );
    }

    /// Configurable top-k builder compiles and produces valid output.
    #[test]
    #[serial]
    fn test_kl_with_sparse_top_k_builder() {
        let teacher = Array::from_f32_slice(
            &(0..200).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 200],
        );
        let student = Array::from_f32_slice(
            &(0..150).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 150],
        );

        for k in [8, 32, 64, 128] {
            let loss = KlDivergenceLoss::new().with_sparse_top_k(k);
            let result = loss.compute(&teacher, &student, 2.0).unwrap();
            let value: f32 = result.item();
            assert!(
                value.is_finite(),
                "KL should be finite for k={}: {}",
                k,
                value
            );
        }
    }

    /// Reverse KL also works across mismatched vocabs.
    #[test]
    #[serial]
    fn test_reverse_kl_cross_vocab() {
        let teacher_vocab = 90_i32;
        let student_vocab = 70_i32;
        let batch = 1_i32;
        let seq = 3_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| (i % 30) as f32 - 15.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| (i % 30) as f32 - 15.0)
            .collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_f32_slice(&student_data, &[batch, seq, student_vocab]);

        let loss = KlDivergenceLoss::reverse().with_sparse_top_k(16);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();
        assert!(value.is_finite(), "reverse KL cross-vocab must be finite");
    }

    // -----------------------------------------------------------------
    // Weighted reduction tests
    // -----------------------------------------------------------------

    /// Weighted reduction with uniform weights must match the unweighted mean
    /// within floating-point tolerance.
    #[test]
    #[serial]
    fn weighted_uniform_matches_unweighted_mean() {
        let batch = 2_i32;
        let seq = 4_i32;
        let vocab = 64_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * vocab))
            .map(|i| ((i % 50) as f32 - 25.0) / 5.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * vocab))
            .map(|i| ((i * 7 % 50) as f32 - 25.0) / 5.0)
            .collect();
        let teacher = Array::from_f32_slice(&teacher_data, &[batch, seq, vocab]);
        let student = Array::from_f32_slice(&student_data, &[batch, seq, vocab]);

        let loss = KlDivergenceLoss::new();
        let unweighted: f32 = loss.compute(&teacher, &student, 2.0).unwrap().item();

        let num_tokens = (batch * seq) as usize;
        let ones = Array::from_f32_slice(&vec![1.0_f32; num_tokens], &[batch, seq]);
        let weighted: f32 = loss
            .compute_weighted(&teacher, &student, 2.0, Some(&ones))
            .unwrap()
            .item();

        assert!(
            (weighted - unweighted).abs() < 1e-4,
            "weighted(uniform)={} should equal unweighted={}",
            weighted,
            unweighted
        );
    }

    /// Weighting with a 0/1 mask must match the mean over the selected
    /// positions only — i.e. padding tokens (weight=0) should not dilute the
    /// loss signal.
    #[test]
    #[serial]
    fn weighted_mask_ignores_zero_positions() {
        // Small deterministic input.
        let teacher = Array::from_f32_slice(
            &[
                1.0, 2.0, 3.0, // token 0
                5.0, 4.0, 3.0, // token 1 (padded)
                0.5, 1.5, 2.5, // token 2
                7.0, 8.0, 9.0, // token 3 (padded)
            ],
            &[1, 4, 3],
        );
        let student = Array::from_f32_slice(
            &[3.0, 2.0, 1.0, 4.0, 3.0, 5.0, 2.0, 2.5, 1.5, 9.0, 7.0, 8.0],
            &[1, 4, 3],
        );

        let loss = KlDivergenceLoss::new();

        // Ground truth: mean over tokens 0 and 2 only. Build a teacher/student
        // containing just those rows and compare to the masked result.
        let teacher_kept = Array::from_f32_slice(&[1.0, 2.0, 3.0, 0.5, 1.5, 2.5], &[1, 2, 3]);
        let student_kept = Array::from_f32_slice(&[3.0, 2.0, 1.0, 2.0, 2.5, 1.5], &[1, 2, 3]);
        let kept_mean: f32 = loss
            .compute(&teacher_kept, &student_kept, 2.0)
            .unwrap()
            .item();

        let mask = Array::from_f32_slice(&[1.0_f32, 0.0, 1.0, 0.0], &[1, 4]);
        let masked: f32 = loss
            .compute_weighted(&teacher, &student, 2.0, Some(&mask))
            .unwrap()
            .item();

        assert!(
            (masked - kept_mean).abs() < 1e-4,
            "masked weighted loss={} should equal mean over unmasked tokens={}",
            masked,
            kept_mean
        );
    }

    /// Mismatched weight shape must produce an error rather than silently
    /// corrupting the loss.
    #[test]
    #[serial]
    fn weighted_size_mismatch_is_error() {
        let teacher = Array::from_f32_slice(&[1.0_f32; 12], &[1, 4, 3]);
        let student = Array::from_f32_slice(&[2.0_f32; 12], &[1, 4, 3]);
        let bad_weights = Array::from_f32_slice(&[1.0_f32, 1.0], &[2]); // expected 4

        let loss = KlDivergenceLoss::new();
        let result = loss.compute_weighted(&teacher, &student, 1.0, Some(&bad_weights));
        assert!(result.is_err(), "expected error on weight-size mismatch");
    }
}
