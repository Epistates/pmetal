//! Soft Cross-Entropy loss for knowledge distillation.
//!
//! Uses the teacher's temperature-softened distribution as soft targets
//! instead of one-hot labels (Hinton et al., 2015):
//! CE(teacher_soft, student_logits) = -sum(softmax(t / T) * log_softmax(s / T))

use super::{
    DistillLoss, SPARSE_TOPK_DEFAULT, align_vocab_with_k, reduce_per_token, tempered_log_probs,
};
use crate::Result;
use pmetal_bridge::compat::Array;

/// Soft Cross-Entropy loss for knowledge distillation.
///
/// Computes cross-entropy between teacher's soft targets and student's predictions.
/// It differs from forward KL only by the teacher's entropy, a constant to the
/// student, so the two have the same gradient.
pub struct SoftCrossEntropyLoss {
    /// Number of top-k teacher tokens to retain when vocab sizes differ.
    ///
    /// Only used when teacher and student have different vocabulary sizes
    /// (cross-architecture distillation).  Defaults to [`SPARSE_TOPK_DEFAULT`].
    sparse_top_k: i32,
}

impl SoftCrossEntropyLoss {
    /// Create a new soft cross-entropy loss.
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

impl Default for SoftCrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for SoftCrossEntropyLoss {
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

        let teacher_probs = tempered_log_probs(&teacher_logits, temperature).exp();
        let student_log_probs = tempered_log_probs(&student_logits, temperature);

        let ce_per_token = teacher_probs
            .multiply(&student_log_probs)
            .sum_axes(&[-1], false)
            .negative();

        reduce_per_token(&ce_per_token, weights)
    }

    fn name(&self) -> &'static str {
        "soft_cross_entropy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_soft_ce_identical_distributions() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let loss = SoftCrossEntropyLoss::new();
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        let value: f32 = result.item();

        // Soft CE of a distribution with itself equals its entropy
        // This should be positive and bounded
        assert!(value > 0.0, "Soft CE should be positive, got {}", value);
        assert!(value < 10.0, "Soft CE should be reasonable, got {}", value);
    }

    #[test]
    #[serial]
    fn test_soft_ce_different_distributions() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = SoftCrossEntropyLoss::new();

        // CE with itself (entropy)
        let self_ce = loss.compute(&teacher, &teacher, 1.0).unwrap();
        // CE with different distribution
        let cross_ce = loss.compute(&teacher, &student, 1.0).unwrap();

        let self_val: f32 = self_ce.item();
        let cross_val: f32 = cross_ce.item();

        // Cross-entropy should be >= entropy (Gibbs' inequality)
        assert!(
            cross_val >= self_val - 1e-4,
            "CE(P, Q) should be >= H(P): CE={}, H={}",
            cross_val,
            self_val
        );
    }

    #[test]
    #[serial]
    fn test_soft_ce_temperature_effect() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = SoftCrossEntropyLoss::new();

        // Higher temperature makes distributions more uniform
        let ce_t1 = loss.compute(&teacher, &student, 1.0).unwrap();
        let ce_t4 = loss.compute(&teacher, &student, 4.0).unwrap();

        let v1: f32 = ce_t1.item();
        let v4: f32 = ce_t4.item();

        // At higher temperature, distributions are more similar
        // so cross-entropy approaches entropy
        assert!(
            v4 < v1,
            "Higher temp should reduce soft CE: T=1: {}, T=4: {}",
            v1,
            v4
        );
    }

    #[test]
    #[serial]
    fn test_soft_ce_batch_processing() {
        // Test with batch of sequences
        let teacher =
            Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], &[2, 1, 4]);
        let student =
            Array::from_f32_slice(&[4.0_f32, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0], &[2, 1, 4]);

        let loss = SoftCrossEntropyLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();

        // Result should be a scalar
        assert!(result.shape().is_empty());
        let value: f32 = result.item();
        assert!(value > 0.0);
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

        let loss = SoftCrossEntropyLoss::new();
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        // Should be positive and finite
        assert!(value > 0.0, "Soft CE should be positive");
        assert!(value.is_finite(), "Soft CE should be finite");
    }

    // -----------------------------------------------------------------
    // Cross-vocab / sparse top-k tests
    // -----------------------------------------------------------------

    /// Soft CE with teacher smaller than student vocab.
    #[test]
    #[serial]
    fn test_soft_ce_cross_vocab_teacher_smaller() {
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

        let loss = SoftCrossEntropyLoss::new().with_sparse_top_k(32);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        assert!(value > 0.0, "soft CE must be positive, got {}", value);
        assert!(value.is_finite(), "soft CE must be finite, got {}", value);
    }

    /// Soft CE with teacher larger than student vocab.
    #[test]
    #[serial]
    fn test_soft_ce_cross_vocab_teacher_larger() {
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

        let loss = SoftCrossEntropyLoss::new().with_sparse_top_k(32);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        assert!(value > 0.0, "soft CE must be positive, got {}", value);
        assert!(value.is_finite(), "soft CE must be finite, got {}", value);
    }

    /// 3-D tensor cross-vocab CE — verifies the Ellipsis fix for rank-3 logits.
    #[test]
    #[serial]
    fn test_soft_ce_cross_vocab_3d_tensors() {
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

        let loss = SoftCrossEntropyLoss::new().with_sparse_top_k(4);
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        let value: f32 = result.item();

        // Scalar result
        assert!(result.shape().is_empty(), "result should be scalar");
        assert!(value.is_finite(), "soft CE must be finite, got {}", value);
    }

    /// Configurable top-k builder produces consistent results across k values.
    #[test]
    #[serial]
    fn test_soft_ce_with_sparse_top_k_builder() {
        let teacher = Array::from_f32_slice(
            &(0..200).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 200],
        );
        let student = Array::from_f32_slice(
            &(0..150).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 150],
        );

        for k in [8, 32, 64, 128] {
            let loss = SoftCrossEntropyLoss::new().with_sparse_top_k(k);
            let result = loss.compute(&teacher, &student, 2.0).unwrap();
            let value: f32 = result.item();
            assert!(
                value.is_finite(),
                "soft CE should be finite for k={}: {}",
                k,
                value
            );
            // CE = -sum(p * log(q)).  When the top-k teacher and student logits share the
            // same relative ordering (both are monotone ascending slices), the distributions
            // become nearly identical after softmax, making CE ≈ entropy ≈ a small positive
            // or effectively 0.  The important invariant is that it is non-negative and finite.
            assert!(
                value >= -1e-5,
                "soft CE must be >= 0 for k={}: {}",
                k,
                value
            );
        }
    }
}
