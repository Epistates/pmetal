//! Generalized Knowledge Distillation (GKD) — on-policy distillation.
//!
//! From Agarwal et al., 2024 (*GKD: Generalized Knowledge Distillation*).
//! Standard sequence-level distillation evaluates losses on either a fixed
//! offline corpus (off-policy) or fully on-policy student samples; GKD
//! interpolates: the dataloader supplies an off-policy corpus, the trainer
//! periodically samples from the student to produce on-policy sequences,
//! and the loss is a `λ`-weighted blend of off-policy and on-policy KL.
//!
//! ```text
//! L_GKD(T, S) = (1 - λ) · L_off(T, S) + λ · L_on(T, S)
//! ```
//!
//! Both `L_off` and `L_on` are forward-KL(T || S) by default — only the
//! source of the *input sequences* differs. This module provides the
//! mixing math: the trainer is responsible for producing the off-policy
//! and on-policy logit pairs.
//!
//! The on-policy sampler trait below is the seam the trainer plugs into.
//! It accepts the current student logits and returns one per-sequence
//! sampled token id (typically a top-`p` or top-`k` truncated multinomial
//! draw at the configured temperature). For unit-test paths a synthetic
//! greedy sampler that returns argmax is also provided.

#![allow(unsafe_code)]

use super::{DistillLoss, KlDivergenceLoss, SPARSE_TOPK_DEFAULT};
use crate::Result;
use pmetal_runtime::compat::{Array, ops};

/// Trait for an on-policy sampler. The trainer implements this to produce
/// student-generated sequences that the teacher then scores. Stays simple
/// on purpose — a richer interface (KV-cache reuse, batched draws) belongs
/// in `pmetal-trainer`.
pub trait OnPolicySampler: Send + Sync {
    /// Sample a single token id from the student's per-position logits.
    ///
    /// `student_logits` has shape `[batch, seq, vocab]`. Returns an `[batch, seq]`
    /// array of int32 token ids — one per position.
    fn sample(&self, student_logits: &Array) -> Result<Array>;
}

/// Greedy "sampler" that always picks the argmax. Useful for tests and as a
/// reference: GKD with greedy sampling reduces to off-policy KD with
/// teacher-forced sequences passed through the student's argmax map.
pub struct GreedySampler;

impl OnPolicySampler for GreedySampler {
    fn sample(&self, student_logits: &Array) -> Result<Array> {
        Ok(student_logits.argmax(-1))
    }
}

/// GKD loss with a configurable on-policy weight `lambda`.
pub struct GkdLoss {
    lambda: f32,
    sampler_temperature: f32,
    sparse_top_k: i32,
    /// Underlying soft-target loss reused for both the on- and off-policy
    /// passes. Defaults to forward-KL.
    inner: KlDivergenceLoss,
}

impl GkdLoss {
    /// `lambda ∈ [0, 1]` controls the on-policy weight. `λ = 0` is plain
    /// off-policy KD; `λ = 1` is fully on-policy; the paper recommends
    /// `λ = 0.5` as a balanced default.
    pub fn new(lambda: f32, sampler_temperature: f32) -> Self {
        Self {
            lambda: lambda.clamp(0.0, 1.0),
            sampler_temperature: sampler_temperature.max(1e-6),
            sparse_top_k: SPARSE_TOPK_DEFAULT,
            inner: KlDivergenceLoss::new(),
        }
    }

    /// Set the cross-vocab top-k for the inner KL.
    pub fn with_sparse_top_k(mut self, k: i32) -> Self {
        self.sparse_top_k = k.max(1);
        self.inner = self.inner.with_sparse_top_k(k);
        self
    }

    /// `lambda` actually in use after clamping.
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Sampler temperature (only used when the trainer constructs an
    /// auxiliary `OnPolicySampler` from this configuration).
    pub fn sampler_temperature(&self) -> f32 {
        self.sampler_temperature
    }

    /// Compute the full GKD loss given pre-evaluated logits for both the
    /// off-policy (corpus) sequence and the on-policy (student-sampled)
    /// sequence. The trainer supplies all four tensors: the same `lambda`
    /// is applied to both KL terms and they're summed in log-space.
    ///
    /// `temperature` is the temperature for both KLs; the sampler can use
    /// a different temperature via `sampler_temperature()` when it draws
    /// the on-policy sequences earlier in the step.
    pub fn compute_full(
        &self,
        teacher_off_logits: &Array,
        student_off_logits: &Array,
        teacher_on_logits: &Array,
        student_on_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        let off_loss = self
            .inner
            .compute(teacher_off_logits, student_off_logits, temperature)?;
        let on_loss = self
            .inner
            .compute(teacher_on_logits, student_on_logits, temperature)?;
        let off_w = Array::from_f32(1.0 - self.lambda);
        let on_w = Array::from_f32(self.lambda);
        Ok(off_loss.multiply(&off_w).add(&on_loss.multiply(&on_w)))
    }
}

impl DistillLoss for GkdLoss {
    /// Loss-only path: with no on-policy sequence available we cannot
    /// compute the GKD blend, so we fall through to plain forward-KL on
    /// the off-policy pair the caller did supply. The trainer should
    /// call [`GkdLoss::compute_full`] directly to get the actual
    /// `λ`-weighted blend.
    ///
    /// We deliberately do **not** scale by `(1 - λ)` here: at `λ = 1.0`
    /// that would zero out the gradient and silently kill training,
    /// which is a worse failure mode than "loss magnitude doesn't track
    /// the eventual blend". Callers that want the blended magnitude
    /// must use `compute_full`.
    fn weighted_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        self.inner
            .weighted_loss(teacher_logits, student_logits, temperature, weights)
    }

    fn name(&self) -> &'static str {
        "gkd"
    }
}

// Suppress dead-code warning when `ops` isn't used (older Rust versions).
const _: fn() = || {
    let _ = ops::maximum;
};

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// `λ = 0` reduces to off-policy KL exactly.
    #[test]
    #[serial]
    fn lambda_zero_matches_off_policy_kl() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[3.0_f32, 1.0, 4.0, 1.0], &[1, 1, 4]);
        let on_t = teacher.clone();
        let on_s = student.clone();

        let gkd = GkdLoss::new(0.0, 1.0);
        let blend: f32 = gkd
            .compute_full(&teacher, &student, &on_t, &on_s, 1.0)
            .unwrap()
            .item();
        let kl: f32 = KlDivergenceLoss::new()
            .compute(&teacher, &student, 1.0)
            .unwrap()
            .item();
        assert!(
            (blend - kl).abs() < 1e-3,
            "λ=0 must reduce to KL: {} vs {}",
            blend,
            kl
        );
    }

    /// Identical off and on logit pairs: blended loss equals KL regardless
    /// of `λ`.
    #[test]
    #[serial]
    fn identical_streams_collapse_to_kl() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[3.0_f32, 1.0, 4.0, 1.0], &[1, 1, 4]);
        let kl: f32 = KlDivergenceLoss::new()
            .compute(&teacher, &student, 1.0)
            .unwrap()
            .item();
        for &lambda in &[0.1_f32, 0.5, 0.9] {
            let v: f32 = GkdLoss::new(lambda, 1.0)
                .compute_full(&teacher, &student, &teacher, &student, 1.0)
                .unwrap()
                .item();
            assert!(
                (v - kl).abs() < 1e-3,
                "λ={} should collapse: {} vs {}",
                lambda,
                v,
                kl
            );
        }
    }

    /// Greedy sampler returns argmax shape `[batch, seq]`. Materializes
    /// through f32 since the bridge does not currently expose a typed
    /// int-extract helper for i32 arrays.
    #[test]
    #[serial]
    fn greedy_sampler_returns_argmax() {
        let logits =
            Array::from_f32_slice(&[0.0_f32, 1.0, 5.0, 2.0, 3.0, 4.0, 0.0, 0.0], &[1, 2, 4]);
        let sampled = GreedySampler.sample(&logits).unwrap();
        assert_eq!(sampled.shape(), &[1, 2]);
        let mut as_f = sampled.as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32());
        let ids: Vec<f32> = as_f.to_f32_vec(2).unwrap();
        assert_eq!(ids, vec![2.0_f32, 1.0]);
    }

    /// `DistillLoss` path: when no on-policy sequence is provided, the
    /// off-only loss is plain off-policy KL — independent of `λ`. This
    /// is a deliberate contract: scaling by `(1-λ)` would silently zero
    /// out training at `λ=1`, which is a far worse failure mode than
    /// "loss magnitude doesn't track the eventual blend".
    #[test]
    #[serial]
    fn off_only_compute_is_plain_kl_regardless_of_lambda() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[3.0_f32, 1.0, 4.0, 1.0], &[1, 1, 4]);
        let kl: f32 = KlDivergenceLoss::new()
            .compute(&teacher, &student, 1.0)
            .unwrap()
            .item();
        for &lambda in &[0.0_f32, 0.5, 1.0] {
            let v: f32 = GkdLoss::new(lambda, 1.0)
                .compute(&teacher, &student, 1.0)
                .unwrap()
                .item();
            assert!(
                (v - kl).abs() < 1e-3,
                "off-only path must equal plain KL at λ={}, got {} vs {}",
                lambda,
                v,
                kl
            );
        }
    }
}
