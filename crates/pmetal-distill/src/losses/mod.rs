//! Loss functions for knowledge distillation.
//!
//! GPU-first implementations using Metal kernels for optimal Apple Silicon performance.
//! Falls back to MLX operations when Metal is unavailable.
//!
//! This module provides various loss functions used in knowledge distillation:
//! - KL Divergence (forward and reverse)
//! - Jensen-Shannon Divergence
//! - Soft Cross-Entropy
//! - MSE on logits
//! - Hidden state alignment losses
//!
//! # GPU Acceleration
//!
//! When the `metal` feature is enabled (default), all loss implementations
//! automatically use custom Metal kernels with these optimizations:
//! - Online softmax: O(1) memory per token instead of materializing full probability tensors
//! - Fused operations: temperature scaling + softmax + loss in single kernel pass
//! - SIMD parallelization: Optimized for large vocabularies (>1024 tokens)
//!
//! # Example
//!
//! ```rust,ignore
//! use pmetal_distill::losses::{KlDivergenceLoss, DistillLoss};
//!
//! let loss = KlDivergenceLoss::new();
//!
//! // GPU acceleration is automatic - no API changes needed
//! let result = loss.compute(&teacher_logits, &student_logits, 2.0)?;
//! ```

use std::ops::Neg;

pub mod hidden_state;
mod jensen_shannon;
mod kl_divergence;
mod mse;
mod soft_cross_entropy;

pub use hidden_state::HiddenStateLoss;
pub use jensen_shannon::JensenShannonLoss;
pub use kl_divergence::KlDivergenceLoss;
pub use mse::MseLoss;
pub use soft_cross_entropy::SoftCrossEntropyLoss;

use crate::Result;
use mlx_rs::Array;

/// Default number of top-k teacher tokens to use when vocab sizes differ.
///
/// 128 covers the vast majority of teacher probability mass in practice
/// (top-128 tokens typically account for >99.9% of mass in peaked distributions).
pub const SPARSE_TOPK_DEFAULT: i32 = 128;

/// Trait for distillation loss functions.
pub trait DistillLoss: Send + Sync {
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
    /// * `teacher_logits` - Teacher logits
    /// * `student_logits` - Student logits
    /// * `temperature` - Softmax temperature
    /// * `weights` - Per-token weights `[batch, seq]`
    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array>;

    /// Get the name of this loss function.
    fn name(&self) -> &'static str;
}

/// Check if GPU acceleration is available for distillation losses.
///
/// Returns true if Metal is available and the device supports GPU-accelerated
/// distillation loss computation.
#[cfg(feature = "metal")]
pub fn is_gpu_available() -> bool {
    pmetal_metal::context::MetalContext::global().is_ok()
}

#[cfg(not(feature = "metal"))]
pub fn is_gpu_available() -> bool {
    false
}

/// Combined distillation loss with hard and soft targets.
pub struct CombinedLoss {
    /// Soft target loss function.
    soft_loss: Box<dyn DistillLoss>,
    /// Alpha for blending (final = alpha * soft + (1-alpha) * hard).
    alpha: f32,
    /// Temperature for soft targets.
    temperature: f32,
}

impl CombinedLoss {
    /// Create a new combined loss.
    pub fn new(soft_loss: Box<dyn DistillLoss>, alpha: f32, temperature: f32) -> Self {
        Self {
            soft_loss,
            alpha,
            temperature,
        }
    }

    /// Compute combined loss with hard labels.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher model outputs
    /// * `student_logits` - Student model outputs
    /// * `labels` - Ground truth labels for hard loss
    pub fn compute_with_labels(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        labels: &Array,
    ) -> Result<Array> {
        // Soft loss (temperature-scaled KL/CE between teacher and student)
        let soft = self
            .soft_loss
            .compute(teacher_logits, student_logits, self.temperature)?;

        // Hard loss (cross-entropy with ground truth)
        let hard = cross_entropy_with_logits(student_logits, labels)?;

        // Weighted combination
        // Note: soft loss is scaled by T^2 to maintain gradient magnitude
        let t_squared = self.temperature * self.temperature;
        let soft_scaled = soft.multiply(&Array::from_f32(self.alpha * t_squared))?;
        let hard_scaled = hard.multiply(&Array::from_f32(1.0 - self.alpha))?;

        Ok(soft_scaled.add(&hard_scaled)?)
    }
}

/// Standard cross-entropy loss with logits and integer labels.
fn cross_entropy_with_logits(logits: &Array, labels: &Array) -> Result<Array> {
    // Log-softmax for numerical stability
    let log_probs = mlx_rs::nn::log_softmax(logits, -1)?;

    // Gather the log probabilities at label positions
    // Shape: logits [batch, seq, vocab], labels [batch, seq]
    let vocab_size = logits.dim(2);

    // Flatten for gather operation
    let log_probs_flat = log_probs.reshape(&[-1, vocab_size])?;
    let labels_flat = labels.reshape(&[-1])?;

    // Get log prob at each label position using MLX take_along_axis
    let gathered = gather_at_indices(&log_probs_flat, &labels_flat)?;

    // Mean negative log probability
    let neg_log_probs = gathered.neg();
    let loss = neg_log_probs.mean(None)?;

    Ok(loss)
}

/// Align teacher and student logits when their vocab sizes differ.
///
/// Uses the default top-k of [`SPARSE_TOPK_DEFAULT`] (128).  For a custom `k`,
/// call [`align_vocab_with_k`] directly.
///
/// When teacher and student have different vocabulary sizes (cross-architecture
/// distillation), aligning via sparse top-k is more principled than truncation
/// or padding: it preserves the distribution information where it matters most
/// and works regardless of which model has the larger vocabulary.
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
/// When teacher and student have different vocabulary sizes (cross-architecture
/// distillation), aligning via sparse top-k is more principled than truncation
/// or padding: it preserves the distribution information where it matters most
/// and works regardless of which model has the larger vocabulary.
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
/// applied without modification.  The Metal GPU path must be bypassed for
/// mismatched vocab because the fused kernels require equal vocab sizes.
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
    let teacher_vocab = teacher_logits.dim(-1) as usize;
    let student_vocab = student_logits.dim(-1) as usize;

    if teacher_vocab == student_vocab {
        return Ok((teacher_logits.clone(), student_logits.clone(), false));
    }

    let k = top_k.max(1).min(teacher_vocab as i32);

    tracing::debug!(
        teacher_vocab,
        student_vocab,
        k,
        "vocab mismatch: using sparse top-k distillation"
    );

    // Argsort descending: negate teacher logits then argsort ascending so the
    // first k indices correspond to the k largest teacher logit positions.
    let neg_teacher = teacher_logits.negative()?;
    let sorted_indices = mlx_rs::ops::argsort_axis(&neg_teacher, -1)?;

    // Slice first k positions along the **last** axis.
    //
    // `Ellipsis` consumes all leading dimensions so this correctly handles
    // tensors of any rank: [tokens, vocab], [batch, seq, vocab], etc.
    // Using `(.., ..k)` instead would be wrong for rank > 2 — it would slice
    // the second-to-last dimension rather than the last one.
    use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
    let top_k_indices = sorted_indices.index((Ellipsis, ..k));

    // Gather teacher logits at the top-k token positions.
    let teacher_aligned = teacher_logits.take_along_axis(&top_k_indices, -1)?;

    // Clamp teacher indices to the student vocab range for gathering.
    // Out-of-range positions are masked below so clamping to 0 is safe here.
    let student_vocab_minus1 = Array::from_int((student_vocab as i32) - 1);
    let zero = Array::from_int(0);
    let clamped = mlx_rs::ops::minimum(&top_k_indices, &student_vocab_minus1)?
        .as_dtype(mlx_rs::Dtype::Int32)?;
    let clamped = mlx_rs::ops::maximum(&clamped, &zero)?;

    // Gather student logits at clamped positions.
    let student_gathered = student_logits.take_along_axis(&clamped, -1)?;

    // Mask out positions where the teacher index fell outside the student vocab.
    // Those positions receive -1e9 so their softmax weight is negligible (~0).
    let student_vocab_arr = Array::from_int(student_vocab as i32);
    let out_of_range = top_k_indices
        .as_dtype(mlx_rs::Dtype::Int32)?
        .ge(&student_vocab_arr)?;
    let neg_large = Array::from_f32(-1e9_f32);
    let student_aligned = mlx_rs::ops::r#where(&out_of_range, &neg_large, &student_gathered)?;

    Ok((teacher_aligned, student_aligned, true))
}

/// Softmax along specified axis.
pub fn softmax(x: &Array, axis: i32) -> Result<Array> {
    let max_x = x.max_axes(&[axis], Some(true))?;
    let shifted = x.subtract(&max_x)?;
    let exp_shifted = shifted.exp()?;
    let sum_exp = exp_shifted.sum_axes(&[axis], Some(true))?;
    Ok(exp_shifted.divide(&sum_exp)?)
}

/// Gather values from a 2D array at specified column indices.
///
/// Uses MLX take operation for GPU-accelerated gather.
fn gather_at_indices(values: &Array, indices: &Array) -> Result<Array> {
    // values: [N, V], indices: [N] -> output: [N]
    let n = values.dim(0);
    let v = values.dim(1);

    // Create row indices [0, 1, 2, ..., N-1]
    let row_indices: Vec<i32> = (0..n).collect();
    let row_indices_arr = Array::from_slice(&row_indices, &[n]);

    // Compute flat indices: row * V + col
    let v_arr = Array::from_int(v);
    let flat_indices = row_indices_arr.multiply(&v_arr)?.add(indices)?;

    // Flatten values and gather
    let values_flat = values.reshape(&[-1])?;
    let gathered = values_flat.take(&flat_indices)?;

    Ok(gathered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_softmax() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[1, 3]);
        let probs = softmax(&logits, -1).unwrap();
        let probs_data: Vec<f32> = probs.as_slice().to_vec();

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
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[1, 3]);
        let log_probs = mlx_rs::nn::log_softmax(&logits, -1).unwrap();
        let log_probs_data: Vec<f32> = log_probs.as_slice().to_vec();

        // All log probs should be <= 0
        for lp in &log_probs_data {
            assert!(*lp <= 0.0);
        }

        // exp(log_softmax) should equal softmax
        let probs = log_probs.exp().unwrap();
        let probs_data: Vec<f32> = probs.as_slice().to_vec();
        let sum: f32 = probs_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    #[serial]
    fn test_gather_at_indices() {
        let values = Array::from_slice(&[0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
        let indices = Array::from_slice(&[1_i32, 2], &[2]);
        let result = gather_at_indices(&values, &indices).unwrap();
        let result_data: Vec<f32> = result.as_slice().to_vec();

        assert_eq!(result_data.len(), 2);
        assert!((result_data[0] - 0.2).abs() < 1e-5); // row 0, col 1
        assert!((result_data[1] - 0.6).abs() < 1e-5); // row 1, col 2
    }

    #[test]
    #[serial]
    fn test_gpu_availability_check() {
        // Should return true on Apple Silicon with Metal feature
        let available = is_gpu_available();
        println!("GPU acceleration available: {}", available);
    }

    // -----------------------------------------------------------------
    // align_vocab / cross-vocab tests
    // -----------------------------------------------------------------

    /// Same vocab size → passthrough, no copy.
    #[test]
    #[serial]
    fn test_align_vocab_same_size_passthrough() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
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
        let teacher = Array::from_slice(&teacher_data, &[1, 8]);
        let student = Array::from_slice(&student_data, &[1, 6]);

        let k = 4_i32;
        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval().unwrap();
        sa.eval().unwrap();

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
        let teacher = Array::from_slice(&teacher_data, &[1, 6]);
        let student = Array::from_slice(&student_data, &[1, 8]);

        let k = 4_i32;
        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval().unwrap();
        sa.eval().unwrap();

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
        let teacher = Array::from_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_slice(&student_data, &[batch, seq, student_vocab]);

        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval().unwrap();
        sa.eval().unwrap();

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
        let student_data = [0.1_f32, 0.2, 0.3, 0.4]; // only 4 tokens
        let teacher = Array::from_slice(&teacher_data, &[1, 6]);
        let student = Array::from_slice(&student_data, &[1, 4]);

        let (ta, _sa, _) = align_vocab_with_k(&teacher, &student, 3).unwrap();
        ta.eval().unwrap();

        let ta_vals: Vec<f32> = ta.as_slice().to_vec();
        // All three selected values must be from {4.0, 9.0, 10.0}
        // i.e. all >= 4.0 (the 4th-highest)
        for v in &ta_vals {
            assert!(*v >= 4.0 - 1e-5, "unexpected value in top-3: {}", v);
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
        let teacher = Array::from_slice(&teacher_data, &[1, 8]);
        let student = Array::from_slice(&student_data, &[1, 4]);

        let (_ta, sa, _) = align_vocab_with_k(&teacher, &student, 4).unwrap();
        sa.eval().unwrap();

        let sa_vals: Vec<f32> = sa.as_slice().to_vec();
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
        let teacher = Array::from_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_slice(&student_data, &[batch, seq, student_vocab]);

        let (ta, sa, mismatched) = align_vocab_with_k(&teacher, &student, k).unwrap();
        ta.eval().unwrap();
        sa.eval().unwrap();

        assert!(mismatched);
        assert_eq!(ta.shape(), &[batch, seq, k]);
        assert_eq!(sa.shape(), &[batch, seq, k]);

        // All teacher-aligned values must be finite
        let ta_vals: Vec<f32> = ta.as_slice().to_vec();
        assert!(ta_vals.iter().all(|v| v.is_finite()));

        // student-aligned: indices < student_vocab so no masking expected
        let sa_vals: Vec<f32> = sa.as_slice().to_vec();
        assert!(sa_vals.iter().all(|v| v.is_finite()));
    }
}
