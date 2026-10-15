//! Universal Logit Distillation (ULD) — cross-tokenizer KD via Wasserstein-1.
//!
//! From Boizard et al., 2024 (*Towards Cross-Tokenizer Distillation: the
//! Universal Logit Distillation Loss*). The original setup of knowledge
//! distillation assumes teacher and student share a tokenizer, so the logit
//! axis aligns position-by-position. ULD removes that assumption: it sorts
//! both distributions in descending order and compares the *order statistics*,
//! which are tokenizer-invariant. The Wasserstein-1 distance over sorted
//! probability vectors then quantifies the disagreement.
//!
//! Algorithm (clean-room, derived from the paper's Algorithm 1):
//!
//!   1. Apply temperature scaling and softmax to teacher and student logits.
//!   2. Sort each per-token distribution in descending order: `p_sorted`,
//!      `q_sorted` ∈ ℝ^V (the same V for both — see padding below).
//!   3. If the two distributions have unequal vocab sizes V_T and V_S, pad
//!      the shorter one on the right with zeros to length max(V_T, V_S).
//!      (Sorted distributions are heavy-tailed; padding zeros at the tail
//!      reflects "no mass beyond top-K" in the original tokenizer.)
//!   4. Loss = mean over tokens of `mean_v(|F_p(v) - F_q(v)|)`, where
//!      F is the cumulative distribution of the sorted vector — i.e.
//!      `mean(|cumsum(p_sorted) - cumsum(q_sorted)|)`. The classical
//!      Wasserstein-1 distance uses a *sum* over `v`; we use the mean so
//!      the loss magnitude is comparable across vocab sizes (Boizard et
//!      al. take the same approach for cross-tokenizer training stability).
//!      The optimum is unaffected — only the gradient scale changes by
//!      a constant `1/V`.
//!
//! Optional `top_k` truncates both distributions to their top-K entries
//! before sorting; this is exactly the truncation the paper uses for
//! efficiency when V is large (≫ 32K).
//!
//! Same-tokenizer parity: when V_T == V_S and both distributions are
//! actually identical, the loss is exactly zero. When V_T == V_S and the
//! distributions differ only by permutation, ULD also returns zero — that
//! is the *intentional* tokenizer-invariance property, distinct from
//! standard KL which would penalize the permutation. Tests cover both.

#![allow(unsafe_code)]

use super::{DistillLoss, reduce_per_token};
use crate::Result;
use pmetal_runtime::compat::Array;

/// Universal Logit Distillation loss.
///
/// Runs on the MLX graph path (no Metal kernel — sort dispatch is non-trivial
/// in the current MLX bindings). Suitable for use cases where teacher and
/// student have *different tokenizers* and standard KL is therefore
/// ill-defined; for shared-tokenizer flows prefer KL or JSD which preserve
/// gradient signal at the per-vocab-position level.
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

    /// Truncate both distributions to their top-K entries before sorting.
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

    fn compute_inner(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        let temp = self.temperature_override.unwrap_or(temperature).max(1e-6);
        let inv_t = 1.0_f32 / temp;
        let t_scaled = teacher_logits.multiply(&Array::from_f32(inv_t));
        let s_scaled = student_logits.multiply(&Array::from_f32(inv_t));
        let p_t = t_scaled.softmax(-1);
        let p_s = s_scaled.softmax(-1);

        // Optional top-K truncation. We sort first and slice; for the pure
        // ULD path this is exactly what the paper's Algorithm 1 step does
        // when it sets K ≪ V to bound runtime.
        let (sorted_t, sorted_s) = sort_desc_with_optional_topk(&p_t, &p_s, self.top_k);

        // Pad the shorter to match the longer along the last axis. After
        // sort_desc_with_optional_topk both are already truncated, so the
        // only remaining mismatch is when teacher/student vocab sizes
        // genuinely differ.
        let len_t = sorted_t.dim(-1) as usize;
        let len_s = sorted_s.dim(-1) as usize;
        let n = len_t.max(len_s);
        let padded_t = pad_last_axis_to(&sorted_t, n);
        let padded_s = pad_last_axis_to(&sorted_s, n);

        // Wasserstein-1 over sorted probability vectors:
        //   W1 = Σ_v |F_p(v) - F_q(v)| = Σ_v |cumsum(p)[v] - cumsum(q)[v]|
        // Average over the V axis, then mean over tokens.
        let cum_t = padded_t.cumsum(-1);
        let cum_s = padded_s.cumsum(-1);
        let abs_diff = cum_t.subtract(&cum_s).abs_val();
        Ok(abs_diff.mean_axis(-1, false))
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
        let per_token = self.compute_inner(teacher_logits, student_logits, temperature)?;
        reduce_per_token(&per_token, weights)
    }

    fn name(&self) -> &'static str {
        "universal_logit"
    }
}

/// Sort each row in descending order, optionally truncating to top-K.
///
/// The bridge exposes `argsort` (ascending) and `take_along_axis` but no
/// direct sort. We argsort the negated array — equivalent to descending
/// argsort of the original — then gather along the last axis.
fn sort_desc_with_optional_topk(p_t: &Array, p_s: &Array, top_k: Option<usize>) -> (Array, Array) {
    let neg = Array::from_f32(-1.0);
    let idx_t = p_t.multiply(&neg).argsort(-1);
    let idx_s = p_s.multiply(&neg).argsort(-1);
    let mut sorted_t = p_t.take_along_axis(&idx_t, -1);
    let mut sorted_s = p_s.take_along_axis(&idx_s, -1);
    if let Some(k) = top_k {
        sorted_t = slice_first_k_last_axis(&sorted_t, k);
        sorted_s = slice_first_k_last_axis(&sorted_s, k);
    }
    (sorted_t, sorted_s)
}

/// Slice the leading `k` entries along the last axis.
fn slice_first_k_last_axis(arr: &Array, k: usize) -> Array {
    let v = arr.dim(-1) as usize;
    if k >= v {
        return arr.clone();
    }
    let mut start: Vec<i32> = vec![0; arr.shape().len()];
    let mut stop: Vec<i32> = arr.shape().to_vec();
    let last = arr.shape().len() - 1;
    start[last] = 0;
    stop[last] = k as i32;
    arr.slice(&start, &stop)
}

/// Right-pad a tensor's last axis to length `n` with zeros (no-op when
/// already long enough). Implemented via `concatenate` because the bridge
/// does not expose a generic `pad` op.
fn pad_last_axis_to(arr: &Array, n: usize) -> Array {
    let cur = arr.dim(-1) as usize;
    if cur >= n {
        return arr.clone();
    }
    let extra = n - cur;
    let mut pad_shape: Vec<i32> = arr.shape().to_vec();
    let last = pad_shape.len() - 1;
    pad_shape[last] = extra as i32;
    let total: usize = pad_shape.iter().map(|&s| s as usize).product();
    let zeros = Array::from_f32_slice(&vec![0.0_f32; total], &pad_shape);
    pmetal_runtime::compat::ops::concatenate_axis(&[arr, &zeros], -1)
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
