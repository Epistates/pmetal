//! Mean Squared Error loss on logits for knowledge distillation.
//!
//! A simple alternative to KL-based losses that directly matches logit values.
//! MSE = mean((teacher_logits - student_logits)^2)

use super::DistillLoss;
use crate::Result;
use mlx_rs::Array;

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
    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        // MSE on raw logits. Note: temperature scaling is intentionally NOT applied
        // because MSE(x/T, y/T) * T^2 = MSE(x, y), making temperature a no-op after
        // the T^2 re-scaling in Distiller::compute_loss. This is a fundamental property
        // of MSE — use KL/JS/soft-CE losses when temperature softening is desired.
        let _ = temperature; // explicitly unused
        let diff = student_logits.subtract(teacher_logits)?;
        let mse_per_token = diff.multiply(&diff)?.mean_axes(&[-1], false)?;

        if let Some(w) = weights {
            let weighted = mse_per_token.multiply(w)?;
            let total_weight = w.sum(None)?;
            let safe_weight = mlx_rs::ops::maximum(&total_weight, &Array::from_f32(1e-8))?;
            Ok(weighted.sum(None)?.divide(&safe_weight)?)
        } else {
            Ok(mse_per_token.mean(None)?)
        }
    }

    fn name(&self) -> &'static str {
        "mse"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_mse_identical_logits() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let loss = MseLoss::new();
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        let value: f32 = result.item();

        // MSE of identical logits should be 0
        assert!(
            value.abs() < 1e-6,
            "MSE of identical logits should be 0, got {}",
            value
        );
    }

    #[test]
    #[serial]
    fn test_mse_different_logits() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_slice(&[2.0_f32, 3.0, 4.0, 5.0], &[1, 1, 4]);

        let loss = MseLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        let value: f32 = result.item();

        // Each element differs by 1, so MSE = mean(1^2) = 1
        assert!(
            (value - 1.0).abs() < 1e-5,
            "MSE should be 1.0, got {}",
            value
        );
    }

    #[test]
    #[serial]
    fn test_mse_temperature_ignored() {
        let teacher = Array::from_slice(&[2.0_f32, 4.0, 6.0, 8.0], &[1, 1, 4]);
        let student = Array::from_slice(&[4.0_f32, 6.0, 8.0, 10.0], &[1, 1, 4]);

        let loss = MseLoss::new();

        // Temperature should not affect raw logit MSE
        let mse_t1 = loss.compute(&teacher, &student, 1.0).unwrap();
        let mse_t2 = loss.compute(&teacher, &student, 2.0).unwrap();

        let v1: f32 = mse_t1.item();
        let v2: f32 = mse_t2.item();

        assert!(
            (v1 - v2).abs() < 1e-5,
            "MSE should be temperature-invariant on raw logits: T=1 ({}) vs T=2 ({})",
            v1,
            v2
        );
    }

    #[test]
    #[serial]
    fn test_mse_symmetry() {
        let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let b = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = MseLoss::new();
        let mse_ab = loss.compute(&a, &b, 1.0).unwrap();
        let mse_ba = loss.compute(&b, &a, 1.0).unwrap();

        let v_ab: f32 = mse_ab.item();
        let v_ba: f32 = mse_ba.item();

        // MSE should be symmetric
        assert!(
            (v_ab - v_ba).abs() < 1e-6,
            "MSE should be symmetric: {}, {}",
            v_ab,
            v_ba
        );
    }

    /// Verify gradients flow through MSE loss (finite + non-zero).
    #[test]
    #[serial]
    fn test_mse_gradient_flow() {
        use mlx_rs::transforms::value_and_grad;

        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);

        let loss_fn = |inputs: &[Array]| -> Vec<Array> {
            let student = &inputs[0];
            let diff = student.subtract(&teacher).unwrap();
            let mse = diff.multiply(&diff).unwrap().mean(None).unwrap();
            vec![mse]
        };

        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);
        let (values, grads) = value_and_grad(loss_fn)(&[student]).unwrap();

        values[0].eval().unwrap();
        grads[0].eval().unwrap();

        let loss_val: f32 = values[0].item();
        assert!(
            loss_val.is_finite(),
            "MSE loss must be finite, got {}",
            loss_val
        );
        assert!(
            loss_val > 0.0,
            "MSE loss must be positive, got {}",
            loss_val
        );

        let grad_data: Vec<f32> = grads[0].as_slice().to_vec();
        let grad_norm: f32 = grad_data.iter().map(|&g| g * g).sum::<f32>().sqrt();
        assert!(
            grad_norm.is_finite(),
            "gradient must be finite, got norm={}",
            grad_norm
        );
        assert!(
            grad_norm > 1e-10,
            "gradient must be non-zero, got norm={}",
            grad_norm
        );
    }

    #[test]
    #[serial]
    fn test_mse_batch_processing() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 1, 4]);
        let student = Array::from_slice(&[2.0_f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[2, 1, 4]);

        let loss = MseLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();

        // Result should be a scalar
        assert!(result.shape().is_empty());
        let value: f32 = result.item();
        // Each element differs by 1, so per-token MSE = mean(1^2) = 1.0 for both batches
        assert!(
            (value - 1.0).abs() < 1e-5,
            "Batch MSE should be 1.0, got {}",
            value
        );
    }
}
