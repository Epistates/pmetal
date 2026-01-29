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
    let log_probs = log_softmax(logits, -1)?;

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

/// Log-softmax along specified axis.
fn log_softmax(x: &Array, axis: i32) -> Result<Array> {
    // log_softmax(x) = x - log(sum(exp(x)))
    // For numerical stability: x - max(x) - log(sum(exp(x - max(x))))
    let max_x = x.max_axes(&[axis], Some(true))?;
    let shifted = x.subtract(&max_x)?;
    let exp_shifted = shifted.exp()?;
    let sum_exp = exp_shifted.sum_axes(&[axis], Some(true))?;
    let log_sum_exp = sum_exp.log()?;
    Ok(shifted.subtract(&log_sum_exp)?)
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
    fn test_log_softmax() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[1, 3]);
        let log_probs = log_softmax(&logits, -1).unwrap();
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
    fn test_gpu_availability_check() {
        // Should return true on Apple Silicon with Metal feature
        let available = is_gpu_available();
        println!("GPU acceleration available: {}", available);
    }
}
