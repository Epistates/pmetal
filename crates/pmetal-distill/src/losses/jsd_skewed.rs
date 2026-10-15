//! Skewed Jensen-Shannon Divergence (α-JSD).
//!
//! From DistiLLM-2 (Ko et al., 2024 — *Distillation with Language Models for
//! Efficient Inference*): the symmetric JSD averages KL(T||M) and KL(S||M)
//! with M = 0.5·T + 0.5·S, which suppresses gradient signal when the student
//! deviates strongly. The skewed variant lets the mixture weight follow the
//! schedule's idea of "how much teacher to trust":
//!
//! ```text
//! JS_α(T || S) = α · KL(T || M_α) + (1-α) · KL(S || M_α)
//! where M_α    = α·T + (1-α)·S
//! ```
//!
//! Edge cases:
//!   * α = 0.5 — standard JSD (this module is bit-equivalent).
//!   * α → 1   — forward-KL(T || S) (mode-covering, like classical Hinton).
//!   * α → 0   — reverse-KL(S || T) (mode-seeking).
//!
//! All computation happens in log-space using the log-sum-exp trick to avoid
//! catastrophic cancellation when α or (1-α) is close to zero. The
//! implementation is a clean-room derivation from the equation above.

#![allow(unsafe_code)]

use super::{DistillLoss, SPARSE_TOPK_DEFAULT, align_vocab_with_k};
use crate::Result;
use pmetal_runtime::compat::{Array, ops};

/// Skewed JSD loss with mixing weight `alpha ∈ (0, 1)`.
///
/// The Metal-fused JSD kernel only handles the symmetric case (α = 0.5), so
/// this loss runs on the MLX graph path. For α in `(0.4, 0.6)` callers should
/// generally prefer [`super::JensenShannonLoss`] for the GPU-accelerated
/// equivalent.
pub struct JsdSkewedLoss {
    alpha: f32,
    sparse_top_k: i32,
}

impl JsdSkewedLoss {
    /// Construct with the given mixing weight; clamps to `[ε, 1-ε]` to keep
    /// `log(α)` and `log(1-α)` finite.
    pub fn new(alpha: f32) -> Self {
        const EPS: f32 = 1e-6;
        let alpha = alpha.clamp(EPS, 1.0 - EPS);
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

    /// The mixing weight `α` actually in use after clamping.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    fn compute_inner(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        // Cross-vocab path (top-k align). Returns matching last-dim arrays.
        let (t_aligned, s_aligned, _mismatched) =
            align_vocab_with_k(teacher_logits, student_logits, self.sparse_top_k)?;

        // Temperature scaling, then log-softmax for stability.
        let inv_t = 1.0_f32 / temperature.max(1e-6);
        let t_scaled = t_aligned.multiply(&Array::from_f32(inv_t));
        let s_scaled = s_aligned.multiply(&Array::from_f32(inv_t));
        let log_t = t_scaled.log_softmax(-1);
        let log_s = s_scaled.log_softmax(-1);

        // log(M_α) = log_sum_exp(log_t + log(α), log_s + log(1-α))
        let log_a = Array::from_f32(self.alpha.ln());
        let log_1ma = Array::from_f32((1.0 - self.alpha).ln());
        let a = log_t.add(&log_a);
        let b = log_s.add(&log_1ma);
        let max_ab = ops::maximum(&a, &b);
        let log_m = max_ab.add(
            &a.subtract(&max_ab)
                .exp()
                .add(&b.subtract(&max_ab).exp())
                .log(),
        );

        // KL(T || M) = Σ exp(log_t) · (log_t - log_m)
        // KL(S || M) = Σ exp(log_s) · (log_s - log_m)
        let p_t = log_t.exp();
        let p_s = log_s.exp();
        let kl_tm = p_t.multiply(&log_t.subtract(&log_m)).sum_axis(-1, false);
        let kl_sm = p_s.multiply(&log_s.subtract(&log_m)).sum_axis(-1, false);

        // JS_α = α·KL(T||M) + (1-α)·KL(S||M)
        let weighted_t = kl_tm.multiply(&Array::from_f32(self.alpha));
        let weighted_s = kl_sm.multiply(&Array::from_f32(1.0 - self.alpha));
        Ok(weighted_t.add(&weighted_s))
    }
}

impl DistillLoss for JsdSkewedLoss {
    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        let per_token = self.compute_inner(teacher_logits, student_logits, temperature)?;
        match weights {
            None => Ok(per_token.mean_all()),
            Some(w) => {
                let weighted = per_token.multiply(w);
                let sum = weighted.sum_all();
                let denom = ops::maximum(&w.sum_all(), &Array::from_f32(1.0));
                Ok(sum.divide(&denom))
            }
        }
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
        for &alpha in &[0.1_f32, 0.5, 0.9] {
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
