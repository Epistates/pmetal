//! Generalized Jensen-Shannon divergence, `JSD(β)`.
//!
//! From Agarwal et al. (ICLR 2024, *On-Policy Distillation of Language
//! Models: Learning from Self-Generated Mistakes*), eq. 2. With teacher `T`,
//! student `S` and the skewed mixture `M = β·T + (1-β)·S`:
//!
//! ```text
//! JSD(β)(T || S) = β · KL(T || M) + (1-β) · KL(S || M)
//! ```
//!
//! `β = 0.5` is the ordinary JSD ([`JensenShannonLoss`](super::JensenShannonLoss)).
//! The two limits are the classical divergences, each scaled by the weight of
//! its own term: `JSD(β) / β → KL(T || S)` as `β → 0` (forward KL,
//! mode-covering) and `JSD(β) / (1-β) → KL(S || T)` as `β → 1` (reverse KL,
//! mode-seeking). Taking the limit literally would make the loss, and so the
//! gradient, vanish, so `β = 0` and `β = 1` are defined here as the divergences
//! themselves.
//!
//! All computation happens in log-space using the log-sum-exp trick to avoid
//! catastrophic cancellation when `β` or `1-β` is close to zero.

use super::{
    DistillLoss, SPARSE_TOPK_DEFAULT, aligned_log_probs, kl_from_log_probs, log_mixture,
    reduce_per_token,
};
use crate::Result;
use pmetal_runtime::compat::Array;

/// Generalized JSD loss with mixing weight `β` (called `alpha` here).
///
/// Runs on the MLX graph. For `β = 0.5` prefer [`super::JensenShannonLoss`],
/// the same value with fewer operations.
pub struct JsdSkewedLoss {
    alpha: f32,
    sparse_top_k: i32,
}

impl JsdSkewedLoss {
    /// Construct with the given mixing weight, clamped to `[0, 1]`.
    ///
    /// `0` is forward KL `KL(T || S)`, `1` is reverse KL `KL(S || T)`, and an
    /// interior weight is clamped to `[1e-6, 1-1e-6]` to keep `log(β)` and
    /// `log(1-β)` finite.
    pub fn new(alpha: f32) -> Self {
        const EPS: f32 = 1e-6;
        let alpha = alpha.clamp(0.0, 1.0);
        let alpha = if alpha == 0.0 || alpha == 1.0 {
            alpha
        } else {
            alpha.clamp(EPS, 1.0 - EPS)
        };
        Self {
            alpha,
            sparse_top_k: SPARSE_TOPK_DEFAULT,
        }
    }

    /// Override the sparse top-k used for cross-vocab alignment.
    pub fn with_sparse_top_k(mut self, k: i32) -> Self {
        self.sparse_top_k = k.max(1);
        self
    }

    /// The mixing weight `β` actually in use after clamping.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// `JSD(β)` per token from the two log-probability tensors.
    fn per_token(&self, log_t: &Array, log_s: &Array) -> Array {
        if self.alpha == 0.0 {
            return kl_from_log_probs(log_t, log_s);
        }
        if self.alpha == 1.0 {
            return kl_from_log_probs(log_s, log_t);
        }
        let log_m = log_mixture(log_t, log_s, self.alpha);
        let kl_tm = kl_from_log_probs(log_t, &log_m);
        let kl_sm = kl_from_log_probs(log_s, &log_m);
        kl_tm
            .multiply(&Array::from_f32(self.alpha))
            .add(&kl_sm.multiply(&Array::from_f32(1.0 - self.alpha)))
    }
}

impl DistillLoss for JsdSkewedLoss {
    fn weighted_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        let (log_t, log_s) = aligned_log_probs(
            teacher_logits,
            student_logits,
            temperature,
            self.sparse_top_k,
        )?;
        reduce_per_token(&self.per_token(&log_t, &log_s), weights)
    }

    fn name(&self) -> &'static str {
        "jsd_skewed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// At α = 0.5 the skewed JSD must coincide with the symmetric JSD.
    #[test]
    #[serial]
    fn alpha_half_matches_symmetric_jsd() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 0.5, 1.5, 2.5], &[1, 2, 3]);
        let student = Array::from_f32_slice(&[2.0_f32, 1.0, 0.5, 1.0, 2.0, 1.0], &[1, 2, 3]);

        let skew = JsdSkewedLoss::new(0.5)
            .compute(&teacher, &student, 1.0)
            .unwrap();
        let sym = super::super::JensenShannonLoss::new()
            .compute(&teacher, &student, 1.0)
            .unwrap();
        let a: f32 = skew.item();
        let b: f32 = sym.item();
        assert!(
            (a - b).abs() < 1e-3,
            "skewed JSD at α=0.5 should match symmetric JSD: {} vs {}",
            a,
            b
        );
    }

    /// Identical teacher/student → loss = 0 regardless of α.
    #[test]
    #[serial]
    fn identical_distributions_yield_zero() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[1, 1, 3]);
        for &alpha in &[0.0_f32, 0.1, 0.5, 0.9, 1.0] {
            let loss = JsdSkewedLoss::new(alpha)
                .compute(&logits, &logits, 1.0)
                .unwrap();
            let v: f32 = loss.item();
            assert!(
                v.abs() < 1e-5,
                "JSD_α with α={} on identical inputs must be 0, got {}",
                alpha,
                v
            );
        }
    }

    /// Loss is always finite and non-negative for non-degenerate inputs.
    #[test]
    #[serial]
    fn loss_is_finite_and_nonnegative() {
        let teacher = Array::from_f32_slice(&[5.0_f32, -3.0, 2.0, 0.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[-2.0_f32, 4.0, -1.0, 1.0], &[1, 1, 4]);
        for &alpha in &[0.05_f32, 0.3, 0.7, 0.95] {
            let loss = JsdSkewedLoss::new(alpha)
                .compute(&teacher, &student, 2.0)
                .unwrap();
            let v: f32 = loss.item();
            assert!(v.is_finite() && v >= -1e-6, "α={} got {}", alpha, v);
        }
    }
}
