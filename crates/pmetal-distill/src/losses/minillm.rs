//! MiniLLM-style reverse-KL distillation loss.
//!
//! From Gu et al., 2024 (*MiniLLM: Knowledge Distillation of Large Language
//! Models*) and the closely-related Speculative-KD line (Xu et al., 2024).
//! The standard Hinton/KL formulation uses forward-KL(T || S), which is
//! mode-covering: the student spreads probability mass across all teacher
//! modes. For autoregressive language models this overspreads and produces
//! low-quality samples. Reverse-KL(S || T) is mode-seeking: the student
//! commits to high-probability teacher modes and ignores the tail.
//!
//! This module's loss is reverse-KL with two practical refinements:
//!   * **Length normalization** — divide the per-sequence loss by the
//!     number of valid tokens (already handled by `mean_axis`/`mean_all`,
//!     but exposed via the per-token return so masked sequences aren't
//!     length-biased).
//!   * **Teacher-mix prob `mix`** — the loss is computed against the
//!     mixture `M = mix · T + (1-mix) · S` instead of the bare teacher.
//!     `mix = 1.0` recovers vanilla reverse-KL; smaller values stabilize
//!     early training when the student is far from the teacher (the
//!     "speculative" KD insight).
//!
//! Clean-room derivation from the paper's equation set; no third-party
//! source consulted.

#![allow(unsafe_code)]

use super::{
    DistillLoss, SPARSE_TOPK_DEFAULT, aligned_log_probs, kl_from_log_probs, log_mixture,
    reduce_per_token,
};
use crate::Result;
use pmetal_runtime::compat::Array;

/// Reverse-KL distillation with optional teacher mixing.
pub struct MiniLlmLoss {
    /// Mixing weight `mix ∈ [0, 1]`: target = mix · T + (1-mix) · S.
    /// `mix = 1.0` is plain reverse-KL.
    mix: f32,
    sparse_top_k: i32,
}

impl MiniLlmLoss {
    /// Construct with the given teacher mixing weight (clamped to [0, 1]).
    pub fn new(mix: f32) -> Self {
        Self {
            mix: mix.clamp(0.0, 1.0),
            sparse_top_k: SPARSE_TOPK_DEFAULT,
        }
    }

    /// Plain reverse-KL (no mixing).
    pub fn reverse_only() -> Self {
        Self::new(1.0)
    }

    /// Override the cross-vocab top-k.
    pub fn with_sparse_top_k(mut self, k: i32) -> Self {
        self.sparse_top_k = k.max(1);
        self
    }

    /// The mixing weight in use.
    pub fn mix(&self) -> f32 {
        self.mix
    }

    /// Per-token reverse-KL(S || M). Returned with the leading `[batch, seq]`
    /// axes so callers can do their own masking / weighting.
    fn compute_inner(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        let (log_t, log_s) = aligned_log_probs(
            teacher_logits,
            student_logits,
            temperature,
            self.sparse_top_k,
        )?;

        // log_target = log(mix·T + (1-mix)·S) when mix is strictly inside
        // (0, 1); the two boundary cases shortcut to log_t / log_s directly to
        // avoid log(0).
        let log_target = if self.mix >= 1.0 - 1e-6 {
            log_t.clone()
        } else if self.mix <= 1e-6 {
            log_s.clone()
        } else {
            log_mixture(&log_t, &log_s, self.mix)
        };

        // Reverse-KL(S || target)
        Ok(kl_from_log_probs(&log_s, &log_target))
    }
}

impl DistillLoss for MiniLlmLoss {
    fn weighted_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        let per_token = self.compute_inner(teacher_logits, student_logits, temperature)?;
        reduce_per_token(&per_token, weights)
    }

    fn name(&self) -> &'static str {
        "minillm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// Identical inputs → reverse-KL = 0.
    #[test]
    #[serial]
    fn identical_inputs_zero_loss() {
        let logits = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 0.5, 1.5, 2.5], &[1, 2, 3]);
        let v: f32 = MiniLlmLoss::reverse_only()
            .compute(&logits, &logits, 1.0)
            .unwrap()
            .item();
        assert!(v.abs() < 1e-5, "got {}", v);
    }

    /// `mix = 1.0` must agree with the existing reverse-KL implementation
    /// in `KlDivergenceLoss::reverse()`.
    #[test]
    #[serial]
    fn mix_1_matches_reverse_kl() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[3.0_f32, 1.0, 4.0, 1.0], &[1, 1, 4]);
        let mini: f32 = MiniLlmLoss::reverse_only()
            .compute(&teacher, &student, 1.0)
            .unwrap()
            .item();
        let kl: f32 = super::super::KlDivergenceLoss::reverse()
            .compute(&teacher, &student, 1.0)
            .unwrap()
            .item();
        assert!(
            (mini - kl).abs() < 1e-3,
            "mix=1 should match reverse-KL: minillm={} vs reverse_kl={}",
            mini,
            kl
        );
    }

    /// `mix = 0.0` must reduce the target to the student itself, making the
    /// loss exactly zero (reverse-KL of any distribution with itself).
    #[test]
    #[serial]
    fn mix_0_yields_zero_loss() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[3.0_f32, 1.0, 4.0, 1.0], &[1, 1, 4]);
        let v: f32 = MiniLlmLoss::new(0.0)
            .compute(&teacher, &student, 1.0)
            .unwrap()
            .item();
        assert!(v.abs() < 1e-4, "mix=0 must yield ≈0, got {}", v);
    }

    /// Loss is finite and non-negative for every interior `mix`.
    #[test]
    #[serial]
    fn finite_and_nonnegative() {
        let teacher = Array::from_f32_slice(&[5.0_f32, -3.0, 2.0, 0.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[-2.0_f32, 4.0, -1.0, 1.0], &[1, 1, 4]);
        for &mix in &[0.05_f32, 0.3, 0.7, 0.95] {
            let v: f32 = MiniLlmLoss::new(mix)
                .compute(&teacher, &student, 2.0)
                .unwrap()
                .item();
            assert!(v.is_finite() && v >= -1e-6, "mix={} got {}", mix, v);
        }
    }
}
