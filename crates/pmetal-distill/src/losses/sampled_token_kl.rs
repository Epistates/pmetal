//! Single-sample KL estimators for on-policy distillation.
//!
//! On-policy distillation samples responses from the student and scores each
//! sampled token with the teacher. When the teacher can only report the
//! log-probability of the token that was sampled (a remote or very large
//! teacher, an inference server) the full-vocabulary losses are out of reach,
//! and the divergence has to be estimated from that one number per token.
//!
//! With `y ~ q_θ` the student's sample, `Δ = log q_θ(y) - log p(y)` and
//! `r = p(y) / q_θ(y)`, three estimators of the KL (Schulman, *Approximating
//! KL Divergence*, 2020) differ in what their **gradient** with respect to
//! `θ` is, in expectation over `y`:
//!
//! | estimator | value | expected gradient |
//! |---|---|---|
//! | `k1 = Δ` | unbiased `KL(q ‖ p)` | zero |
//! | `k2 = ½Δ²` | biased | `∇ KL(q ‖ p)` (reverse KL) |
//! | `k3 = r - 1 - log r` | unbiased `KL(q ‖ p)`, low variance | `∇ KL(p ‖ q)` (forward KL) |
//!
//! so to *descend the reverse KL* back-propagate `k2`, to descend the forward
//! KL back-propagate `k3`, and report `k1` as the (unbiased) value. This loss
//! returns the surrogate to differentiate and the `k1` estimate to log. The
//! expectations are checked exactly, by enumerating a small vocabulary.
//!
//! The samples must come from the *current* student (one gradient step per
//! rollout); for older rollouts reweight with an importance ratio.

use super::reduce_per_token;
use crate::Result;
use pmetal_runtime::compat::{Array, Dtype, ops};

/// Which divergence's gradient the surrogate follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampledTokenKlTarget {
    /// Descend `KL(student ‖ teacher)`: the surrogate is `k2 = ½Δ²`.
    Reverse,
    /// Descend `KL(teacher ‖ student)`: the surrogate is `k3 = r - 1 - log r`
    /// (with `log r` clamped to `[-10, 10]` so one token the student finds
    /// very unlikely does not dominate).
    Forward,
}

/// Estimate a KL from the teacher's log-probabilities of sampled tokens.
#[derive(Debug, Clone, Copy)]
pub struct SampledTokenKl {
    target: SampledTokenKlTarget,
}

/// The terms of [`SampledTokenKl::compute`].
#[derive(Debug)]
pub struct SampledTokenKlOutput {
    /// The surrogate to differentiate, averaged over the masked-in tokens.
    pub loss: Array,
    /// `k1 = log q - log p`, averaged: the unbiased estimate of
    /// `KL(student ‖ teacher)` to log. Carries no gradient.
    pub kl_estimate: Array,
}

impl SampledTokenKl {
    /// A surrogate that descends the given divergence.
    pub fn new(target: SampledTokenKlTarget) -> Self {
        Self { target }
    }

    /// The surrogate and the KL estimate for one batch of sampled tokens.
    ///
    /// * `student_logits` - `[batch, seq, vocab]`, the distributions the
    ///   tokens were sampled from
    /// * `tokens` - `[batch, seq]` sampled ids
    /// * `teacher_log_probs` - `[batch, seq]`, the teacher's `log p(token)`
    /// * `mask` - optional `[batch, seq]`, 1 on response tokens
    pub fn compute(
        &self,
        student_logits: &Array,
        tokens: &Array,
        teacher_log_probs: &Array,
        mask: Option<&Array>,
    ) -> Result<SampledTokenKlOutput> {
        let f32_ = Dtype::Float32.as_i32();
        let index = tokens
            .as_dtype(Dtype::Int32.as_i32())
            .stop_gradient()
            .expand_dims(-1);
        let log_q = student_logits
            .as_dtype(f32_)
            .log_softmax(-1)
            .take_along_axis(&index, -1)
            .squeeze(-1);
        let delta = log_q.subtract(&teacher_log_probs.as_dtype(f32_).stop_gradient());

        let per_token = match self.target {
            SampledTokenKlTarget::Reverse => delta.square().multiply(&Array::from_f32(0.5)),
            SampledTokenKlTarget::Forward => {
                let log_r = ops::minimum(
                    &ops::maximum(&delta.negative(), &Array::from_f32(-10.0)),
                    &Array::from_f32(10.0),
                );
                log_r.exp().subtract(&Array::from_f32(1.0)).subtract(&log_r)
            }
        };

        Ok(SampledTokenKlOutput {
            loss: reduce_per_token(&per_token, mask)?,
            kl_estimate: reduce_per_token(&delta.stop_gradient(), mask)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_runtime::compat::nn::value_and_grad_explicit;
    use serial_test::serial;

    fn log_softmax(z: &[f64]) -> Vec<f64> {
        let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + z.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
        z.iter().map(|x| x - lse).collect()
    }

    /// The expected gradient over `y ~ q` of each estimator, summed exactly
    /// over a five-token vocabulary, against the gradient of the divergence it
    /// is meant to follow (computed in float64).
    #[test]
    #[serial]
    fn expected_gradients_are_the_reverse_and_forward_kl_gradients() {
        let v = 5;
        let teacher = [1.0_f32, -0.5, 0.3, 2.0, -1.2];
        let student = [0.4_f32, 0.9, -0.7, 0.1, 1.5];
        let lp = log_softmax(&teacher.map(f64::from));
        let lq = log_softmax(&student.map(f64::from));
        let q: Vec<f64> = lq.iter().map(|x| x.exp()).collect();
        let p: Vec<f64> = lp.iter().map(|x| x.exp()).collect();
        let kl_qp: f64 = (0..v).map(|i| q[i] * (lq[i] - lp[i])).sum();

        for (target, want) in [
            // ∇_z KL(q‖p) = q ⊙ (log q - log p - KL(q‖p))
            (
                SampledTokenKlTarget::Reverse,
                (0..v)
                    .map(|i| q[i] * (lq[i] - lp[i] - kl_qp))
                    .collect::<Vec<_>>(),
            ),
            // ∇_z KL(p‖q) = q - p
            (
                SampledTokenKlTarget::Forward,
                (0..v).map(|i| q[i] - p[i]).collect(),
            ),
        ] {
            let estimator = SampledTokenKl::new(target);
            let mut expected = vec![0.0_f64; v];
            let mut expected_kl = 0.0_f64;
            for y in 0..v {
                let tokens = Array::from_i32_slice_shaped(&[y as i32], &[1, 1]);
                let teacher_lp = Array::from_f32_slice(&[lp[y] as f32], &[1, 1]);
                let (value, mut grads) = value_and_grad_explicit(
                    |a: &[Array]| {
                        estimator
                            .compute(&a[0], &tokens, &teacher_lp, None)
                            .unwrap()
                            .loss
                    },
                    &[Array::from_f32_slice(&student, &[1, 1, v as i32])],
                    &[],
                )
                .unwrap();
                value.eval();
                grads[0].eval();
                let g = grads[0].to_f32_vec(v).unwrap();
                for i in 0..v {
                    expected[i] += q[y] * g[i] as f64;
                }
                let out = estimator
                    .compute(
                        &Array::from_f32_slice(&student, &[1, 1, v as i32]),
                        &tokens,
                        &teacher_lp,
                        None,
                    )
                    .unwrap();
                expected_kl += q[y] * out.kl_estimate.item::<f32>() as f64;
            }
            pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
            for i in 0..v {
                assert!(
                    (expected[i] - want[i]).abs() <= 2e-5 + 1e-4 * want[i].abs(),
                    "{target:?} entry {i}: expected gradient {} vs {}",
                    expected[i],
                    want[i]
                );
            }
            // k1 is an unbiased estimate of KL(q‖p).
            assert!(
                (expected_kl - kl_qp).abs() <= 1e-5 * kl_qp,
                "{target:?}: E[k1] = {expected_kl}, KL = {kl_qp}"
            );
        }
    }

    /// Masked tokens carry neither loss nor gradient.
    #[test]
    #[serial]
    fn masked_tokens_are_excluded() {
        let student = Array::from_f32_slice(&[0.1, 0.9, 0.0, 1.0, -1.0, 0.5], &[1, 2, 3]);
        let tokens = Array::from_i32_slice_shaped(&[1, 2], &[1, 2]);
        let teacher_lp = Array::from_f32_slice(&[-1.2, -0.7], &[1, 2]);
        let mask = Array::from_f32_slice(&[1.0, 0.0], &[1, 2]);
        let (_, mut grads) = value_and_grad_explicit(
            |a: &[Array]| {
                SampledTokenKl::new(SampledTokenKlTarget::Reverse)
                    .compute(&a[0], &tokens, &teacher_lp, Some(&mask))
                    .unwrap()
                    .loss
            },
            std::slice::from_ref(&student),
            &[],
        )
        .unwrap();
        grads[0].eval();
        let g = grads[0].to_f32_vec(6).unwrap();
        assert!(g[3..].iter().all(|x| *x == 0.0), "{g:?}");
        assert!(g[..3].iter().any(|x| x.abs() > 1e-6), "{g:?}");
    }
}
