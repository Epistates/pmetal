//! Attention-map transfer losses for knowledge distillation.
//!
//! These losses distill attention from the teacher's self-attention into the
//! student's. They consume attention probability tensors of shape
//! `[batch, heads, q_len, k_len]` — i.e. the post-softmax attention weights —
//! and return a scalar loss.
//!
//! Three formulations are supported:
//!
//! * [`AttentionLossType::Mse`] — elementwise MSE on attention probabilities.
//! * [`AttentionLossType::Kl`]  — forward KL per query row, `KL(teacher || student)`.
//!   Interprets each row as a distribution over keys; more statistically principled
//!   than MSE but sensitive to numerical stability.
//! * [`AttentionLossType::AttentionTransfer`] — Zagoruyko & Komodakis 2017:
//!   sum over heads → L2-normalise per (batch, query) → MSE.
//!
//! Head mismatch between teacher and student is handled by
//! [`AttentionHeadReduction::MeanOverHeads`] (mean across head axis before
//! comparison). The default [`AttentionHeadReduction::Exact`] requires matching
//! head counts and surfaces a shape error otherwise.
//!
//! This module is a *primitive*: the caller is responsible for extracting
//! attention weights from both models and aligning layer indices. It mirrors
//! [`super::HiddenStateLoss`] in that regard.

use crate::{AttentionHeadReduction, AttentionLossType, Result};
use pmetal_runtime::compat::Array;

/// Attention-map transfer loss.
pub struct AttentionTransferLoss {
    loss_type: AttentionLossType,
    head_reduction: AttentionHeadReduction,
    /// Small epsilon used in log/normalisation paths to avoid NaNs on zero rows.
    eps: f32,
}

impl AttentionTransferLoss {
    /// Create a new attention-transfer loss with MSE and exact head matching.
    pub fn new() -> Self {
        Self {
            loss_type: AttentionLossType::Mse,
            head_reduction: AttentionHeadReduction::Exact,
            eps: 1e-8,
        }
    }

    /// MSE on attention probability matrices.
    pub fn mse() -> Self {
        Self::new().with_loss_type(AttentionLossType::Mse)
    }

    /// Forward KL between teacher/student attention rows.
    pub fn kl() -> Self {
        Self::new().with_loss_type(AttentionLossType::Kl)
    }

    /// Zagoruyko & Komodakis attention-map distillation.
    pub fn attention_transfer() -> Self {
        Self::new().with_loss_type(AttentionLossType::AttentionTransfer)
    }

    pub fn with_loss_type(mut self, loss_type: AttentionLossType) -> Self {
        self.loss_type = loss_type;
        self
    }

    pub fn with_head_reduction(mut self, reduction: AttentionHeadReduction) -> Self {
        self.head_reduction = reduction;
        self
    }

    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps.max(0.0);
        self
    }

    pub fn name(&self) -> &'static str {
        match self.loss_type {
            AttentionLossType::Mse => "attn_mse",
            AttentionLossType::Kl => "attn_kl",
            AttentionLossType::AttentionTransfer => "attn_transfer",
        }
    }

    /// Compute attention-transfer loss.
    ///
    /// # Arguments
    /// * `teacher_attn` – attention probabilities `[batch, heads_t, q, k]`.
    /// * `student_attn` – attention probabilities `[batch, heads_s, q, k]`.
    ///
    /// `q` and `k` must match. `heads_t == heads_s` is required unless
    /// `MeanOverHeads` reduction is selected.
    pub fn compute(&self, teacher_attn: &Array, student_attn: &Array) -> Result<Array> {
        let t_shape = teacher_attn.shape().to_vec();
        let s_shape = student_attn.shape().to_vec();

        if t_shape.len() != 4 || s_shape.len() != 4 {
            return Err(crate::DistillError::Other(format!(
                "attention tensors must be 4D [batch, heads, q, k]; got teacher {:?}, student {:?}",
                t_shape, s_shape
            )));
        }

        // Validate batch / q / k match.
        if t_shape[0] != s_shape[0] || t_shape[2] != s_shape[2] || t_shape[3] != s_shape[3] {
            return Err(crate::DistillError::Other(format!(
                "attention batch/q/k must match: teacher {:?}, student {:?}",
                t_shape, s_shape
            )));
        }

        // Reconcile head count.
        let (teacher, student) = match self.head_reduction {
            AttentionHeadReduction::Exact => {
                if t_shape[1] != s_shape[1] {
                    return Err(crate::DistillError::Other(format!(
                        "head count mismatch: teacher={} student={} — use MeanOverHeads to reconcile",
                        t_shape[1], s_shape[1]
                    )));
                }
                (teacher_attn.clone(), student_attn.clone())
            }
            AttentionHeadReduction::MeanOverHeads => (
                teacher_attn.mean_axes(&[1], true),
                student_attn.mean_axes(&[1], true),
            ),
        };

        match self.loss_type {
            AttentionLossType::Mse => self.mse_loss(&teacher, &student),
            AttentionLossType::Kl => self.kl_loss(&teacher, &student),
            AttentionLossType::AttentionTransfer => {
                self.attention_transfer_loss(&teacher, &student)
            }
        }
    }

    fn mse_loss(&self, teacher: &Array, student: &Array) -> Result<Array> {
        let diff = student.subtract(teacher);
        let squared = diff.multiply(&diff);
        Ok(squared.mean_all())
    }

    /// Forward KL per query row: `sum_k p_t log(p_t / p_s)`, averaged over
    /// (batch, heads, q). Assumes inputs are already softmaxed probabilities.
    fn kl_loss(&self, teacher: &Array, student: &Array) -> Result<Array> {
        let eps_arr = Array::from_f32(self.eps);
        let t_safe = teacher.add(&eps_arr);
        let s_safe = student.add(&eps_arr);
        // KL(p||q) = sum p * (log p - log q). Using log after epsilon clamp.
        let log_ratio = t_safe.log().subtract(&s_safe.log());
        let per_row = teacher.multiply(&log_ratio).sum_axes(&[-1], false);
        Ok(per_row.mean_all())
    }

    /// Zagoruyko & Komodakis 2017 attention-map transfer.
    ///
    /// 1. Aggregate across heads: `A = sum_h attn_h` → `[batch, q, k]`.
    /// 2. L2-normalise each `[k]` vector (per batch, per query).
    /// 3. MSE between teacher and student normalised maps.
    fn attention_transfer_loss(&self, teacher: &Array, student: &Array) -> Result<Array> {
        let t_sum = teacher.sum_axes(&[1], false); // [B, Q, K]
        let s_sum = student.sum_axes(&[1], false);

        let eps_arr = Array::from_f32(self.eps);

        let t_norm = t_sum.multiply(&t_sum).sum_axes(&[-1], true).sqrt();
        let s_norm = s_sum.multiply(&s_sum).sum_axes(&[-1], true).sqrt();

        let t_normed = t_sum.divide(&t_norm.add(&eps_arr));
        let s_normed = s_sum.divide(&s_norm.add(&eps_arr));

        let diff = s_normed.subtract(&t_normed);
        Ok(diff.multiply(&diff).mean_all())
    }
}

impl Default for AttentionTransferLoss {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// Build a valid attention tensor with shape [B, H, Q, K] whose last-axis
    /// rows sum to 1 (softmax-style).
    fn make_attn(b: i32, h: i32, q: i32, k: i32, seed: f32) -> Array {
        // Fill with a simple pattern then softmax along the last axis.
        let n = (b * h * q * k) as usize;
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.137 + seed).sin() + 1.5)
            .collect();
        Array::from_f32_slice(&data, &[b, h, q, k]).softmax(-1)
    }

    #[test]
    #[serial]
    fn mse_identical_is_zero() {
        let attn = make_attn(2, 4, 5, 5, 0.1);
        let loss = AttentionTransferLoss::mse();
        let out = loss.compute(&attn, &attn).unwrap();
        let v: f32 = out.item();
        assert!(v.abs() < 1e-6, "identical MSE expected ~0, got {}", v);
    }

    #[test]
    #[serial]
    fn kl_identical_is_zero() {
        let attn = make_attn(2, 4, 5, 5, 0.2);
        let loss = AttentionTransferLoss::kl();
        let out = loss.compute(&attn, &attn).unwrap();
        let v: f32 = out.item();
        assert!(v.abs() < 1e-4, "identical KL expected ~0, got {}", v);
    }

    #[test]
    #[serial]
    fn attention_transfer_identical_is_zero() {
        let attn = make_attn(2, 4, 5, 5, 0.3);
        let loss = AttentionTransferLoss::attention_transfer();
        let out = loss.compute(&attn, &attn).unwrap();
        let v: f32 = out.item();
        assert!(
            v.abs() < 1e-6,
            "identical attention-transfer expected ~0, got {}",
            v
        );
    }

    #[test]
    #[serial]
    fn different_attn_gives_positive_loss() {
        let teacher = make_attn(1, 2, 4, 4, 0.0);
        let student = make_attn(1, 2, 4, 4, 10.0);

        for (name, loss) in [
            ("mse", AttentionTransferLoss::mse()),
            ("kl", AttentionTransferLoss::kl()),
            ("at", AttentionTransferLoss::attention_transfer()),
        ] {
            let v: f32 = loss.compute(&teacher, &student).unwrap().item();
            assert!(v > 0.0, "{} expected > 0, got {}", name, v);
        }
    }

    #[test]
    #[serial]
    fn mean_over_heads_reconciles_head_mismatch() {
        let teacher = make_attn(1, 8, 4, 4, 0.5);
        let student = make_attn(1, 2, 4, 4, 0.5);

        // Default Exact reduction fails.
        let exact = AttentionTransferLoss::mse().compute(&teacher, &student);
        assert!(exact.is_err(), "exact must fail on head mismatch");

        // MeanOverHeads succeeds.
        let loss =
            AttentionTransferLoss::mse().with_head_reduction(AttentionHeadReduction::MeanOverHeads);
        let v: f32 = loss.compute(&teacher, &student).unwrap().item();
        assert!(v.is_finite());
    }

    #[test]
    #[serial]
    fn bad_rank_is_rejected() {
        let attn3d = Array::from_f32_slice(&[1.0, 0.0, 0.5, 0.5], &[2, 1, 2]);
        let loss = AttentionTransferLoss::mse();
        assert!(loss.compute(&attn3d, &attn3d).is_err());
    }

    #[test]
    #[serial]
    fn batch_q_k_mismatch_is_rejected() {
        let teacher = make_attn(1, 2, 4, 4, 0.1);
        let student = make_attn(1, 2, 4, 8, 0.1);
        let loss = AttentionTransferLoss::mse();
        assert!(loss.compute(&teacher, &student).is_err());
    }
}
