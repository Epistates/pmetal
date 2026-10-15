//! Universal Logit Distillation (ULD): cross-tokenizer KD without a shared
//! vocabulary.
//!
//! From Boizard et al., 2024 (*Towards Cross-Tokenizer Distillation: the
//! Universal Logit Distillation Loss for LLMs*). Distillation by KL needs the
//! teacher's and the student's logit axes to mean the same tokens. ULD drops
//! that requirement: it compares the two distributions by their *order
//! statistics*, which do not depend on which token is which. With both
//! probability vectors sorted in decreasing order (`σ_S`, `σ_T`), the
//! Wasserstein distance with a 0-1 ground cost has the closed form
//!
//! ```text
//! W₁ = Σ_i | p_S(σ_S(i)) - p_T(σ_T(i)) |        (eq. 5)
//! ```
//!
//! where the shorter vector is padded with zeros to the longer's length. This
//! is a sum over the sorted entries, bounded by 2, and it costs one sort per
//! side. The paper adds it to the student's ordinary cross-entropy with a
//! weight λ (1.5 by default); this loss is the `W₁` term only, and the
//! [`Distiller`](crate::Distiller) combines it with the hard-label loss.
//!
//! Optional `top_k` truncates both sorted vectors to their K largest entries,
//! which bounds the cost on very large vocabularies at the price of ignoring
//! the tail mass (at most `2 · tail` of the distance).
//!
//! A permutation of the vocabulary leaves the loss unchanged. That is the
//! property that makes the loss tokenizer-invariant, and it is also why it is
//! a weak signal when a shared vocabulary exists: prefer KL or JSD then.
//!
//! The paper aligns the two models' sequences by truncating to the shorter
//! tokenization; position `t` of the teacher is compared with position `t` of
//! the student. Aligning positions across tokenizations is the caller's job.

use super::{DistillLoss, reduce_per_token, tempered_log_probs};
use crate::Result;
use pmetal_runtime::compat::Array;

/// Universal Logit Distillation loss.
///
/// Runs on the MLX graph. The student's sorting permutation is cut out of the
/// autodiff graph (the gradient flows through the sorted *values*, not the
/// permutation), which MLX requires of a gather's indices.
pub struct UniversalLogitLoss {
    temperature_override: Option<f32>,
    top_k: Option<usize>,
}

impl UniversalLogitLoss {
    /// New ULD loss using the temperature passed at compute time.
    pub fn new() -> Self {
        Self {
            temperature_override: None,
            top_k: None,
        }
    }

    /// Truncate both distributions to their top-K entries before comparing.
    /// Significantly faster on large vocabularies (e.g. 256K).
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k.max(1));
        self
    }

    /// Override the temperature parameter. Mostly useful for testing — the
    /// `compute_weighted` API already accepts a temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature_override = Some(t.max(1e-6));
        self
    }

    fn per_token(&self, teacher_logits: &Array, student_logits: &Array, temperature: f32) -> Array {
        let temp = self.temperature_override.unwrap_or(temperature).max(1e-6);
        let p_t = tempered_log_probs(teacher_logits, temp).exp();
        let p_s = tempered_log_probs(student_logits, temp).exp();

        let sorted_t = sort_desc(&p_t, self.top_k);
        let sorted_s = sort_desc(&p_s, self.top_k);

        // Pad the shorter with zeros: "no mass beyond this many tokens".
        let n = sorted_t.dim(-1).max(sorted_s.dim(-1));
        let padded_t = pad_last_axis_to(&sorted_t, n);
        let padded_s = pad_last_axis_to(&sorted_s, n);

        padded_s.subtract(&padded_t).abs_val().sum_axis(-1, false)
    }
}

impl Default for UniversalLogitLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for UniversalLogitLoss {
    fn weighted_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        let per_token = self.per_token(teacher_logits, student_logits, temperature);
        reduce_per_token(&per_token, weights)
    }

    fn name(&self) -> &'static str {
        "universal_logit"
    }
}

/// Sort each row in descending order, optionally keeping only the largest `k`.
///
/// There is no sort op, so this argsorts the negated values and gathers. The
/// permutation is an integer index set; it must not be in the autodiff graph
/// (MLX cannot differentiate a gather with respect to its indices), so it is
/// detached and the gradient reaches `p` through the gathered values.
fn sort_desc(p: &Array, top_k: Option<usize>) -> Array {
    let order = p.negative().argsort(-1).stop_gradient();
    let sorted = p.take_along_axis(&order, -1);
    match top_k {
        Some(k) if (k as i32) < sorted.dim(-1) => {
            let mut stop: Vec<i32> = sorted.shape().to_vec();
            let last = stop.len() - 1;
            stop[last] = k as i32;
            sorted.slice(&vec![0; stop.len()], &stop)
        }
        _ => sorted,
    }
}

/// Right-pad a tensor's last axis with zeros to length `n` (no-op when it is
/// already that long).
fn pad_last_axis_to(arr: &Array, n: i32) -> Array {
    let cur = arr.dim(-1);
    if cur >= n {
        return arr.clone();
    }
    let mut widths = vec![0; 2 * arr.shape().len()];
    let last = widths.len() - 1;
    widths[last] = n - cur;
    arr.pad_constant(&widths, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// Identical inputs (same vocab) must yield exactly zero loss. This is
    /// the strictest sanity check on the Wasserstein-1 derivation.
    #[test]
    #[serial]
    fn identical_inputs_zero_loss() {
        let logits =
            Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5, 4.5], &[1, 2, 4]);
        let loss = UniversalLogitLoss::new()
            .compute(&logits, &logits, 1.0)
            .unwrap();
        let v: f32 = loss.item();
        assert!(v.abs() < 1e-5, "expected ~0, got {}", v);
    }

    /// Permutation invariance: two logit tensors that are permutations of
    /// each other along the vocab axis must produce zero ULD loss. This is
    /// the *purpose* of ULD — to be tokenizer-permutation-invariant.
    #[test]
    #[serial]
    fn permutation_invariant() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        // Student has the same set of values but in a different order.
        let student = Array::from_f32_slice(&[3.0_f32, 1.0, 4.0, 2.0], &[1, 1, 4]);
        let loss = UniversalLogitLoss::new()
            .compute(&teacher, &student, 1.0)
            .unwrap();
        let v: f32 = loss.item();
        assert!(v.abs() < 1e-4, "permutation should yield ~0, got {}", v);
    }

    /// Different shapes with the same total mass should still produce a
    /// finite, non-negative loss.
    #[test]
    #[serial]
    fn cross_vocab_finite() {
        // Teacher vocab = 4, student vocab = 6.
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0], &[1, 1, 6]);
        let loss = UniversalLogitLoss::new()
            .compute(&teacher, &student, 1.0)
            .unwrap();
        let v: f32 = loss.item();
        assert!(v.is_finite() && v >= 0.0, "got {}", v);
    }

    /// `with_top_k` truncates without changing the relative ordering. For
    /// distributions that are already concentrated on the top-K entries,
    /// the loss should change by at most the tail mass.
    #[test]
    #[serial]
    fn top_k_truncation_is_finite() {
        let teacher = Array::from_f32_slice(&[5.0_f32, 4.0, 3.0, 2.0, 1.0, 0.0], &[1, 1, 6]);
        let student = Array::from_f32_slice(&[4.0_f32, 5.0, 2.0, 3.0, 0.0, 1.0], &[1, 1, 6]);
        let loss = UniversalLogitLoss::new()
            .with_top_k(3)
            .compute(&teacher, &student, 1.0)
            .unwrap();
        let v: f32 = loss.item();
        assert!(v.is_finite() && v >= 0.0);
    }
}
