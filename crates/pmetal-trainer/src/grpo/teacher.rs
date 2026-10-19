//! The teacher term of RLKD: distillation from a frozen teacher, blended into
//! the GRPO loss.

use pmetal_distill::losses::{DistillLoss, KlDivergenceLoss};
use pmetal_runtime::compat::{Array, Exception};

use crate::rlkd::TeacherModel;

/// A frozen teacher whose next-token distribution the policy is pulled
/// toward, on the policy's own samples, with weight `α(step)`:
/// `L = (1 − α)·L_grpo + α·T²·KL(teacher ‖ policy)`.
///
/// The KL is a mean over the completion tokens (the prompt and padding are
/// masked out), taken between the temperature-softened distributions, with the
/// `T²` factor that keeps its gradient's size under the temperature (Hinton et
/// al., 2015, §2.1). Because the tokens are the policy's own samples, this is
/// the on-policy distillation of Agarwal et al. (*On-Policy Distillation of
/// Language Models*, ICLR 2024).
pub(crate) struct TeacherTerm<'t> {
    pub(crate) model: &'t mut dyn TeacherModel,
    schedule: Box<dyn Fn(usize, usize) -> f32 + 't>,
    temperature: f32,
    reverse_kl: bool,
}

impl<'t> TeacherTerm<'t> {
    /// A term with weight `schedule(step, total_steps)`.
    pub(crate) fn new(
        model: &'t mut dyn TeacherModel,
        schedule: impl Fn(usize, usize) -> f32 + 't,
        temperature: f32,
        reverse_kl: bool,
    ) -> Self {
        Self {
            model,
            schedule: Box::new(schedule),
            temperature,
            reverse_kl,
        }
    }

    /// The weight of the distillation term at optimizer step `step` of `total_steps`.
    pub(crate) fn alpha_at(&self, step: usize, total_steps: usize) -> f32 {
        (self.schedule)(step, total_steps)
    }

    /// The distillation loss of a batch. `teacher_logits` and `policy_logits`
    /// are `[B, S, V]`; `mask` is the `[B, S-1]` completion mask of the
    /// shifted log-probs: position `t` predicts token `t + 1`, so the last
    /// position predicts nothing and is dropped.
    pub(crate) fn distillation(
        &self,
        teacher_logits: &Array,
        policy_logits: &Array,
        mask: &Array,
    ) -> Result<Array, Exception> {
        let seq = policy_logits.dim(1);
        let teacher = teacher_logits.slice(
            &[0, 0, 0],
            &[teacher_logits.dim(0), seq - 1, teacher_logits.dim(2)],
        );
        let student = policy_logits.slice(
            &[0, 0, 0],
            &[policy_logits.dim(0), seq - 1, policy_logits.dim(2)],
        );
        let kl = KlDivergenceLoss::new()
            .with_reverse(self.reverse_kl)
            .compute_weighted(&teacher, &student, self.temperature, Some(mask))
            .map_err(|e| Exception::custom(e.to_string()))?;
        Ok(kl.multiply(&Array::from_f32(self.temperature * self.temperature)))
    }
}
