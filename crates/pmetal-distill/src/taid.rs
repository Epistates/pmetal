//! TAID: Temporally Adaptive Interpolated Distillation.
//!
//! Shing, Misaki, Bao, Yokoi and Akiba, *TAID: Temporally Adaptive
//! Interpolated Distillation for Efficient Knowledge Transfer in Language
//! Models* (ICLR 2025). A student far smaller than its teacher cannot match
//! the teacher's distribution directly: forward KL makes it spread mass over
//! modes it cannot represent, reverse KL collapses it onto one. TAID gives it
//! an intermediate target that starts at the student itself and moves toward
//! the teacher as the student improves.
//!
//! # The method
//!
//! With teacher logits `z_T`, the student's logits `z_S` held constant, and an
//! interpolation parameter `t ∈ [0, 1]`, the intermediate distribution is a
//! mixture **in logit space** (Definition 3.1):
//!
//! ```text
//! p_t = softmax( (1 - t) · sg(z_S) + t · z_T )
//! ```
//!
//! `sg` is a stop-gradient: the target does not move with the student's
//! gradient step, so the objective is plain forward KL against a fixed
//! target (Definition 3.2):
//!
//! ```text
//! J_t = KL( p_t || softmax(z_S) )         averaged over tokens
//! ```
//!
//! `t = 0` is the student itself (zero loss, zero gradient); `t = 1` is the
//! teacher. `t` does not follow a fixed schedule alone. After each step the
//! relative improvement of the objective drives a momentum that speeds `t` up
//! while the student keeps up and slows it while it does not (Algorithm 1):
//!
//! ```text
//! δ_n = (J_{n-1} - J_n) / (J_{n-1} + ε)
//! m_n = β · m_{n-1} + (1 - β) · δ_n
//! t_n = min(t_end, max(t_linear(n), t_{n-1} + α · sigmoid(m_n) · (t_end - t_{n-1})))
//! ```
//!
//! where `t_linear` is the straight line from `t_start` to `t_end` over the
//! run, a floor that guarantees the teacher is reached.
//!
//! # Usage
//!
//! [`TaidDistiller`] is stateful: it carries `t` and the momentum.
//! [`compute_loss`](TaidDistiller::compute_loss) builds the lazy loss at the
//! current `t`; after the optimizer step, hand the objective's value to
//! [`observe`](TaidDistiller::observe) to advance `t`.
//!
//! ```rust,ignore
//! use pmetal_distill::{TaidConfig, TaidDistiller};
//!
//! let mut taid = TaidDistiller::new(TaidConfig::default())?;
//! for step in 1..=total_steps {
//!     let out = taid.compute_loss(&teacher_logits, &student_logits, Some(&mask), None)?;
//!     // ... differentiate out.total, update the student ...
//!     taid.observe(out.distillation.item::<f32>() as f64, step, total_steps);
//! }
//! ```
//!
//! The teacher and the student must share a vocabulary (the mixture is taken
//! per token id).

use crate::losses::{kl_from_log_probs, reduce_per_token, tempered_log_probs};
use pmetal_runtime::compat::{Array, Exception};
use pmetal_runtime::training::CrossEntropy;

/// Error type for TAID operations.
#[derive(Debug, thiserror::Error)]
pub enum TaidError {
    /// MLX computation error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
    /// Distillation error.
    #[error("Distillation error: {0}")]
    Distill(#[from] crate::DistillError),
}

/// Result type for TAID operations.
pub type TaidResult<T> = std::result::Result<T, TaidError>;

/// TAID hyperparameters. Defaults are the paper's instruction-tuning values.
#[derive(Debug, Clone)]
pub struct TaidConfig {
    /// `t` at the first step (`t_start`). The paper uses 0.2 for most
    /// teacher/student pairs and 0.4 for Phi-3-mini and the vision-language
    /// pair. Must be in `(0, 1]` for the student to receive any gradient.
    pub t_start: f64,

    /// `t` the schedule reaches at the last step (`t_end`); 1.0 is the
    /// teacher itself.
    pub t_end: f64,

    /// Step size of the adaptive update (`α`): 5e-4 for instruction tuning
    /// and vision-language models, 5e-5 for pre-training.
    pub alpha: f64,

    /// Momentum of the improvement signal (`β`), 0.99 in every experiment.
    pub beta: f64,

    /// Guard on the denominator of the relative improvement (`ε`).
    pub epsilon: f64,

    /// Weight of the hard-label cross-entropy in the total loss:
    /// `(1 - w) · J + w · CE`. Zero is pure distillation.
    pub hard_target_weight: f64,

    /// Label value marking positions the hard-label loss ignores.
    pub ignore_index: i32,
}

impl Default for TaidConfig {
    fn default() -> Self {
        Self {
            t_start: 0.2,
            t_end: 1.0,
            alpha: 5e-4,
            beta: 0.99,
            epsilon: 1e-8,
            hard_target_weight: 0.0,
            ignore_index: -100,
        }
    }
}

impl TaidConfig {
    /// Create a new TAID config with the paper's defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `t_start`.
    pub fn with_t_start(mut self, t: f64) -> Self {
        self.t_start = t;
        self
    }

    /// Set `t_end`.
    pub fn with_t_end(mut self, t: f64) -> Self {
        self.t_end = t;
        self
    }

    /// Set the adaptive step size `α`.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the momentum `β`.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set the hard-label weight.
    pub fn with_hard_target_weight(mut self, weight: f64) -> Self {
        self.hard_target_weight = weight;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> TaidResult<()> {
        let in_unit = |x: f64| x.is_finite() && (0.0..=1.0).contains(&x);
        if !(in_unit(self.t_start) && self.t_start > 0.0) {
            return Err(TaidError::Config(
                "t_start must be in (0, 1]: at 0 the target is the student itself and there is no gradient"
                    .into(),
            ));
        }
        if !in_unit(self.t_end) || self.t_end < self.t_start {
            return Err(TaidError::Config("t_end must be in [t_start, 1]".into()));
        }
        if !(self.alpha.is_finite() && self.alpha >= 0.0) {
            return Err(TaidError::Config("alpha must be non-negative".into()));
        }
        if !(self.beta.is_finite() && (0.0..1.0).contains(&self.beta)) {
            return Err(TaidError::Config("beta must be in [0, 1)".into()));
        }
        if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            return Err(TaidError::Config("epsilon must be positive".into()));
        }
        if !in_unit(self.hard_target_weight) {
            return Err(TaidError::Config(
                "hard_target_weight must be in [0, 1]".into(),
            ));
        }
        Ok(())
    }
}

/// TAID Distiller: the intermediate distribution and the adaptive `t`.
pub struct TaidDistiller {
    config: TaidConfig,
    /// The interpolation parameter the next loss is built at.
    t: f64,
    /// Momentum of the relative improvement, `m_n`.
    momentum: f64,
    /// The objective the previous step reported, `J_{n-1}`.
    previous_objective: Option<f64>,
}

impl TaidDistiller {
    /// Create a new TAID distiller, with `t = t_start`.
    pub fn new(config: TaidConfig) -> TaidResult<Self> {
        config.validate()?;
        Ok(Self {
            t: config.t_start,
            momentum: 0.0,
            previous_objective: None,
            config,
        })
    }

    /// The interpolation parameter the next loss is built at.
    pub fn t(&self) -> f64 {
        self.t
    }

    /// The momentum `m_n` of the improvement signal.
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// The linear floor `t_linear` after `step` of `total_steps` steps.
    pub fn linear_t(&self, step: usize, total_steps: usize) -> f64 {
        let progress = if total_steps == 0 {
            1.0
        } else {
            (step as f64 / total_steps as f64).min(1.0)
        };
        self.config.t_start + (self.config.t_end - self.config.t_start) * progress
    }

    /// Advance `t` after a training step.
    ///
    /// `objective` is the value of the TAID objective `J` that step computed
    /// (the `distillation` term of its [`TaidLossOutput`]) and `step` the
    /// number of steps completed including this one. The first call has no
    /// earlier objective to compare with and contributes no improvement.
    /// Non-finite objectives are ignored: `t` still follows its linear floor.
    pub fn observe(&mut self, objective: f64, step: usize, total_steps: usize) {
        let delta = match self.previous_objective {
            Some(previous) if objective.is_finite() && previous.is_finite() => {
                (previous - objective) / (previous + self.config.epsilon)
            }
            _ => 0.0,
        };
        if objective.is_finite() {
            self.previous_objective = Some(objective);
        }
        self.momentum = self.config.beta * self.momentum + (1.0 - self.config.beta) * delta;

        let sigmoid = 1.0 / (1.0 + (-self.momentum).exp());
        let advanced = self.t + self.config.alpha * sigmoid * (self.config.t_end - self.t);
        let floor = self.linear_t(step, total_steps);
        self.t = advanced.max(floor).min(self.config.t_end);
    }

    /// Log-probabilities of the intermediate distribution
    /// `softmax((1 - t) · sg(z_S) + t · z_T)`, in f32.
    ///
    /// Neither the teacher nor the student logits carry gradient out of this.
    pub fn intermediate_log_probs(&self, teacher_logits: &Array, student_logits: &Array) -> Array {
        let f32_ = pmetal_runtime::compat::Dtype::Float32.as_i32();
        let t = self.t as f32;
        let teacher = teacher_logits.stop_gradient().as_dtype(f32_);
        let student = student_logits.stop_gradient().as_dtype(f32_);
        let mixed = student
            .multiply(&Array::from_f32(1.0 - t))
            .add(&teacher.multiply(&Array::from_f32(t)));
        tempered_log_probs(&mixed, 1.0)
    }

    /// Compute the TAID loss at the current `t`.
    ///
    /// # Arguments
    /// * `teacher_logits` - `[batch, seq, vocab]`
    /// * `student_logits` - `[batch, seq, vocab]`, the same vocabulary
    /// * `weights` - optional per-token weights `[batch, seq]` (a 0/1 mask of
    ///   the positions that count); the loss is `sum(J·w) / sum(w)`
    /// * `labels` - optional ground-truth ids `[batch, seq]` for the
    ///   hard-label term when `hard_target_weight > 0`
    pub fn compute_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        weights: Option<&Array>,
        labels: Option<&Array>,
    ) -> TaidResult<TaidLossOutput> {
        if teacher_logits.shape() != student_logits.shape() {
            return Err(crate::DistillError::ShapeMismatch {
                teacher: teacher_logits.shape().to_vec(),
                student: student_logits.shape().to_vec(),
            }
            .into());
        }

        let log_p_t = self.intermediate_log_probs(teacher_logits, student_logits);
        let log_q = tempered_log_probs(student_logits, 1.0);
        let per_token = kl_from_log_probs(&log_p_t, &log_q);
        let distillation = reduce_per_token(&per_token, weights)?;

        let hard_target = match labels {
            Some(labels) if self.config.hard_target_weight > 0.0 => Some(
                CrossEntropy::new()
                    .with_ignore_index(self.config.ignore_index)
                    .mean(student_logits, labels),
            ),
            _ => None,
        };

        let total = match &hard_target {
            Some(hard) => {
                let w = self.config.hard_target_weight as f32;
                distillation
                    .multiply(&Array::from_f32(1.0 - w))
                    .add(&hard.multiply(&Array::from_f32(w)))
            }
            None => distillation.clone(),
        };

        Ok(TaidLossOutput {
            total,
            distillation,
            hard_target,
            t: self.t,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &TaidConfig {
        &self.config
    }
}

/// Output from TAID loss computation.
#[derive(Debug)]
pub struct TaidLossOutput {
    /// Total combined loss.
    pub total: Array,
    /// The TAID objective `J_t`, a lazy scalar. Evaluate it after the step
    /// and pass it to [`TaidDistiller::observe`].
    pub distillation: Array,
    /// Hard-label cross-entropy (if used).
    pub hard_target: Option<Array>,
    /// The interpolation parameter this loss was built at.
    pub t: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid_and_starts_at_t_start() {
        let config = TaidConfig::default();
        assert!(config.validate().is_ok());
        let distiller = TaidDistiller::new(config).unwrap();
        assert!((distiller.t() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn config_validation_rejects_nonsense() {
        for bad in [
            TaidConfig::default().with_t_start(0.0),
            TaidConfig::default().with_t_start(1.5),
            TaidConfig::default().with_t_end(0.1),
            TaidConfig::default().with_beta(1.0),
            TaidConfig::default().with_alpha(-1.0),
            TaidConfig::default().with_hard_target_weight(2.0),
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
