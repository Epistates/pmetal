//! Mean Squared Error loss on logits for knowledge distillation.
//!
//! A simple alternative to KL-based losses that directly matches logit values.
//! MSE = mean((teacher_logits - student_logits)^2)

use mlx_rs::Array;
use crate::Result;
use super::DistillLoss;

/// Mean Squared Error loss on logits.
///
/// Directly minimizes the squared difference between teacher and student logits.
/// This is simpler than probability-based losses and can be effective for
/// models with similar architectures.
pub struct MseLoss;

impl MseLoss {
    /// Create a new MSE loss.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MseLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for MseLoss {
    fn compute(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        // Scale logits by temperature for consistency with other losses
        let temp = Array::from_f32(temperature);
        let teacher_scaled = teacher_logits.divide(&temp)?;
        let student_scaled = student_logits.divide(&temp)?;

        // Compute squared difference
        let diff = student_scaled.subtract(&teacher_scaled)?;
        let squared = diff.multiply(&diff)?;

        // Mean over all dimensions
        let loss = squared.mean(None)?;

        Ok(loss)
    }

    fn name(&self) -> &'static str {
        "mse"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_identical_logits() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let loss = MseLoss::new();
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        let value: f32 = result.item();

        // MSE of identical logits should be 0
        assert!(value.abs() < 1e-6, "MSE of identical logits should be 0, got {}", value);
    }

    #[test]
    fn test_mse_different_logits() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_slice(&[2.0_f32, 3.0, 4.0, 5.0], &[1, 1, 4]);

        let loss = MseLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        let value: f32 = result.item();

        // Each element differs by 1, so MSE = mean(1^2) = 1
        assert!((value - 1.0).abs() < 1e-5, "MSE should be 1.0, got {}", value);
    }

    #[test]
    fn test_mse_temperature_scaling() {
        let teacher = Array::from_slice(&[2.0_f32, 4.0, 6.0, 8.0], &[1, 1, 4]);
        let student = Array::from_slice(&[4.0_f32, 6.0, 8.0, 10.0], &[1, 1, 4]);

        let loss = MseLoss::new();

        // At T=1: diff = 2, MSE = 4
        let mse_t1 = loss.compute(&teacher, &student, 1.0).unwrap();
        // At T=2: scaled_diff = 1, MSE = 1
        let mse_t2 = loss.compute(&teacher, &student, 2.0).unwrap();

        let v1: f32 = mse_t1.item();
        let v2: f32 = mse_t2.item();

        assert!((v1 - 4.0).abs() < 1e-5, "MSE at T=1 should be 4, got {}", v1);
        assert!((v2 - 1.0).abs() < 1e-5, "MSE at T=2 should be 1, got {}", v2);
    }

    #[test]
    fn test_mse_symmetry() {
        let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let b = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = MseLoss::new();
        let mse_ab = loss.compute(&a, &b, 1.0).unwrap();
        let mse_ba = loss.compute(&b, &a, 1.0).unwrap();

        let v_ab: f32 = mse_ab.item();
        let v_ba: f32 = mse_ba.item();

        // MSE should be symmetric
        assert!((v_ab - v_ba).abs() < 1e-6, "MSE should be symmetric: {}, {}", v_ab, v_ba);
    }

    #[test]
    fn test_mse_batch_processing() {
        let teacher = Array::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0,
              5.0, 6.0, 7.0, 8.0],
            &[2, 1, 4]
        );
        let student = Array::from_slice(
            &[2.0_f32, 3.0, 4.0, 5.0,
              6.0, 7.0, 8.0, 9.0],
            &[2, 1, 4]
        );

        let loss = MseLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();

        // Result should be a scalar
        assert!(result.shape().is_empty());
        let value: f32 = result.item();
        // All elements differ by 1, so MSE = 1
        assert!((value - 1.0).abs() < 1e-5, "MSE should be 1.0, got {}", value);
    }
}
