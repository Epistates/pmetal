//! MiniLLM: on-policy reverse-KL distillation by policy gradient.
//!
//! Gu, Dong, Wei and Huang, *MiniLLM: Knowledge Distillation of Large
//! Language Models* (ICLR 2024). Forward KL `KL(p || q_θ)` makes a student
//! that is too small to represent the teacher spread its mass over regions
//! the teacher never visits, and it samples poorly. The reverse KL
//! `KL(q_θ || p) = E_{y~q_θ}[log q_θ(y) - log p(y)]` is mode-seeking, but its
//! expectation is over the *student's* samples, so it is optimized with a
//! policy gradient. With `r_t = log p(y_t | y_<t) - log q_θ(y_t | y_<t)`:
//!
//! * **Single-step decomposition** (eq. 3). The reward of the current token
//!   is not sampled: it is summed exactly over the vocabulary, which is the
//!   per-position reverse KL `Σ_v q(v)·(log q(v) - log p(v))`. This removes
//!   most of the variance.
//! * **Long-term term.** The rest of the sequence is a REINFORCE term
//!   `R_{t+1} · ∇ log q_θ(y_t)` with `R_{t+1}` the sum of later rewards.
//! * **Length normalization** (eq. 5). `R^Norm_{t+1}` divides that sum by the
//!   number of later tokens, so short responses are not favoured.
//! * **Teacher-mixed sampling.** Responses are drawn from
//!   `p~ = α·p + (1-α)·q_θ` (α = 0.2 in the paper) so a poor early student
//!   does not wander; the importance ratio `ρ_t = q_θ(y_t) / p~(y_t)` corrects
//!   for it, and is clipped to `[1-ε, 1+ε]` as in Algorithm 1.
//!
//! [`MiniLlmLoss::compute`] is a surrogate whose gradient is
//! `(∇L)_Single + (∇L)_Long^Norm`: the single-step term is weighted by the
//! (clipped, constant) importance ratio, and the long-term term is the PPO
//! clipped surrogate `-min(ρ·R, clip(ρ, 1-ε, 1+ε)·R)` whose gradient is
//! `ρ·R·∇ log q_θ(y_t)` inside the clip range. Add the language-modelling loss
//! on pre-training data (`L_PT`) in the caller if desired.
//!
//! This loss needs sampled responses, so it is not a [`DistillLoss`](super::DistillLoss):
//! the rollout (sampling from [`MiniLlmLoss::mixed_log_probs`] position by
//! position) belongs to the training loop.
//!
//! # Alignment
//!
//! `tokens[b, t]` is the token sampled at position `t`, and
//! `logits[b, t]` the distribution it was sampled from; a causal LM's logits
//! at position `t` predict token `t + 1`, so shift before calling. `mask` is
//! 1 on response tokens and 0 on the prompt and on padding.

use super::{kl_from_log_probs, log_mixture, reduce_per_token};
use crate::Result;
use pmetal_runtime::compat::{Array, Dtype, ops};

/// The MiniLLM surrogate loss.
#[derive(Debug, Clone, Copy)]
pub struct MiniLlmLoss {
    teacher_mix: f32,
    clip_epsilon: f32,
    length_normalize: bool,
}

/// The terms of [`MiniLlmLoss::compute`].
#[derive(Debug)]
pub struct MiniLlmOutput {
    /// The surrogate to differentiate: `single_step + long_term`.
    pub total: Array,
    /// Importance-weighted per-position reverse KL, `(∇L)_Single`.
    pub single_step: Array,
    /// The clipped policy-gradient surrogate, `(∇L)_Long^Norm`.
    pub long_term: Array,
}

impl Default for MiniLlmLoss {
    /// The paper's setting: `α = 0.2`, `ε = 0.2`, length-normalized.
    fn default() -> Self {
        Self::new(0.2)
    }
}

impl MiniLlmLoss {
    /// A loss with teacher-mixing weight `α` (clamped to `[0, 1]`),
    /// `ε = 0.2` and length normalization.
    pub fn new(teacher_mix: f32) -> Self {
        Self {
            teacher_mix: teacher_mix.clamp(0.0, 1.0),
            clip_epsilon: 0.2,
            length_normalize: true,
        }
    }

    /// Set the importance-ratio clip range `[1-ε, 1+ε]`.
    pub fn with_clip_epsilon(mut self, epsilon: f32) -> Self {
        self.clip_epsilon = epsilon.max(0.0);
        self
    }

    /// Turn the length normalization of the long-term reward on or off.
    pub fn with_length_normalization(mut self, on: bool) -> Self {
        self.length_normalize = on;
        self
    }

    /// The teacher-mixing weight `α`.
    pub fn teacher_mix(&self) -> f32 {
        self.teacher_mix
    }

    /// `log p~ = log(α·p + (1-α)·q_θ)` over the vocabulary, f32: the
    /// distribution responses are sampled from. Neither input carries
    /// gradient out of it.
    pub fn mixed_log_probs(&self, teacher_logits: &Array, student_logits: &Array) -> Array {
        let lp = log_softmax_f32(&teacher_logits.stop_gradient());
        let lq = log_softmax_f32(&student_logits.stop_gradient());
        self.mix(&lp, &lq)
    }

    fn mix(&self, log_p: &Array, log_q: &Array) -> Array {
        if self.teacher_mix <= 0.0 {
            log_q.clone()
        } else if self.teacher_mix >= 1.0 {
            log_p.clone()
        } else {
            log_mixture(log_p, log_q, self.teacher_mix)
        }
    }

    /// The surrogate loss on one batch of sampled responses.
    ///
    /// * `teacher_logits`, `student_logits` - `[batch, seq, vocab]` on the
    ///   sampled sequences, teacher-forced.
    /// * `behavior_logits` - the student's logits when the responses were
    ///   sampled, if the student has been updated since; `None` means the
    ///   current student (one gradient step per batch).
    /// * `tokens` - `[batch, seq]` sampled ids.
    /// * `mask` - `[batch, seq]`, 1 on response tokens.
    ///
    /// The result is averaged over the masked-in tokens.
    pub fn compute(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        behavior_logits: Option<&Array>,
        tokens: &Array,
        mask: Option<&Array>,
    ) -> Result<MiniLlmOutput> {
        let f32_ = Dtype::Float32.as_i32();
        let log_p = log_softmax_f32(&teacher_logits.stop_gradient());
        let log_q = log_softmax_f32(student_logits);
        let log_q_behavior = match behavior_logits {
            Some(b) => log_softmax_f32(&b.stop_gradient()),
            None => log_q.stop_gradient(),
        };

        // Token ids are integers: keep them out of the gradient trace.
        let index = tokens
            .as_dtype(Dtype::Int32.as_i32())
            .stop_gradient()
            .expand_dims(-1);
        let gather = |lp: &Array| lp.take_along_axis(&index, -1).squeeze(-1);
        let log_p_y = gather(&log_p);
        let log_q_y = gather(&log_q);
        let log_q0_y = gather(&log_q_behavior);

        // Rewards r_t = log p - log q, constants.
        let ones = Array::ones(log_p_y.shape(), f32_);
        let m = match mask {
            Some(m) => m.reshape(log_p_y.shape()).as_dtype(f32_),
            None => ones,
        };
        let reward = log_p_y.subtract(&log_q0_y).multiply(&m).stop_gradient();

        // R_{t+1}: the rewards after position t, per remaining token if
        // length-normalized. Constants.
        let after = reward
            .sum_axes(&[-1], true)
            .subtract(&reward.cumsum(-1))
            .stop_gradient();
        let advantage = if self.length_normalize {
            let remaining = m.sum_axes(&[-1], true).subtract(&m.cumsum(-1));
            after.divide(&ops::maximum(&remaining, &Array::from_f32(1.0)))
        } else {
            after
        };

        // rho = q_θ(y_t) / p~(y_t), differentiable through q_θ only.
        let log_mixed_y = if self.teacher_mix <= 0.0 {
            log_q0_y.clone()
        } else if self.teacher_mix >= 1.0 {
            log_p_y.clone()
        } else {
            log_mixture(&log_p_y, &log_q0_y, self.teacher_mix)
        };
        let rho = log_q_y.subtract(&log_mixed_y).exp();
        let lo = Array::from_f32(1.0 - self.clip_epsilon);
        let hi = Array::from_f32(1.0 + self.clip_epsilon);
        let clipped = ops::minimum(&ops::maximum(&rho, &lo), &hi);

        // (∇L)_Single: the per-position reverse KL, weighted by the clipped
        // ratio taken as a constant.
        let kl = kl_from_log_probs(&log_q, &log_p);
        let single = reduce_per_token(&kl.multiply(&clipped.stop_gradient()), mask)?;

        // (∇L)_Long: PPO-clipped REINFORCE on the normalized later rewards.
        let long_per_token =
            ops::minimum(&rho.multiply(&advantage), &clipped.multiply(&advantage)).negative();
        let long = reduce_per_token(&long_per_token, mask)?;

        Ok(MiniLlmOutput {
            total: single.add(&long),
            single_step: single,
            long_term: long,
        })
    }
}

fn log_softmax_f32(logits: &Array) -> Array {
    logits.as_dtype(Dtype::Float32.as_i32()).log_softmax(-1)
}
