//! RLKD: Reinforcement Learning with Knowledge Distillation.
//!
//! GRPO policy-gradient optimization and distillation from a frozen teacher
//! in one loop: the teacher scores the *student's own samples* token by
//! token, so the student is pulled toward the teacher's distribution exactly
//! where it generates (on-policy distillation, Agarwal et al., ICLR 2024),
//! while the rewards shape what it generates.
//!
//! # Loss
//!
//! ```text
//! L = (1 − α) · L_grpo + α · T² · KL(teacher ‖ policy)
//! ```
//!
//! - `L_grpo` is the [GRPO](crate::grpo) objective, with every option of
//!   [`GrpoConfig`] (loss aggregation, clip-higher, dynamic sampling, several
//!   updates per batch, a KL penalty to a reference model);
//! - the KL is a mean over the completion tokens, between the
//!   temperature-softened distributions, with Hinton's `T²` factor;
//! - `α` is annealed linearly from `distill_alpha` to `final_alpha` over the
//!   run (distillation-heavy first, RL-heavy at the end), or held constant.
//!
//! RLKD is [`GrpoTrainer`]'s loop with one more term in the loss, so it
//! samples, scores, schedules the learning rate, clips gradients, rolls back
//! and checkpoints exactly as GRPO does.

use pmetal_core::TrainingConfig;
use pmetal_lora::TrainableModel;
use pmetal_runtime::compat::{
    Array, Exception,
    module::{Module, ModuleParameters},
    optimizers::Optimizer,
};

use crate::{
    adaptive_lr::AdaptiveLrConfig,
    grpo::{
        CombinedReward, CompletionGroup, GrpoConfig, GrpoError, GrpoPrompt, GrpoResult,
        GrpoTrainer, TeacherTerm,
    },
};

// ---------------------------------------------------------------------------
// Teacher forward trait
// ---------------------------------------------------------------------------

/// Minimal interface required of a frozen teacher model.
///
/// Both `DynamicModel` (raw inference model) and `DynamicLoraModel` (LoRA
/// model used as teacher) satisfy this bound.  Using a dedicated trait keeps
/// `RlkdTrainer` independent of any concrete model type.
pub trait TeacherModel {
    /// Run a forward pass and return logits `[batch, seq, vocab]`.
    fn forward_teacher(&mut self, input_ids: &Array) -> Result<Array, Exception>;
}

/// Impl for `DynamicModel` — the canonical frozen inference model type.
///
/// `DynamicModel` is the standard way to load a teacher model (no LoRA, no grad).
impl TeacherModel for pmetal_models::DynamicModel {
    fn forward_teacher(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.forward(input_ids, None)
    }
}

/// Impl for `DynamicLoraModel` — allows using a fine-tuned LoRA model as teacher.
impl TeacherModel for pmetal_lora::DynamicLoraModel {
    fn forward_teacher(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.forward(input_ids, None)
            .map_err(|e| Exception::custom(e.to_string()))
    }
}

/// A helper wrapper for anything implementing the [`TrainableModel`] trait that
/// isn't covered by the concrete impls above.  Use this when the teacher is an
/// architecture-specific LoRA model (e.g. `LlamaLoraForCausalLM`).
///
/// ```rust,ignore
/// let teacher_wrapped = TrainableTeacher(my_lora_model);
/// trainer.run(&mut policy, &mut teacher_wrapped, ...);
/// ```
pub struct TrainableTeacher<T: TrainableModel>(pub T);

impl<T: TrainableModel> TeacherModel for TrainableTeacher<T> {
    fn forward_teacher(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.0
            .forward(input_ids, None)
            .map_err(|e| Exception::custom(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// RLKD training configuration.
#[derive(Debug, Clone)]
pub struct RlkdConfig {
    /// GRPO configuration (generation parameters, rewards, KL penalty, etc.).
    pub grpo: GrpoConfig,
    /// General training hyperparameters (LR, epochs, output dir, …).
    pub training: TrainingConfig,
    /// Distillation blend factor: 0.0 = pure RL, 1.0 = pure distillation.
    ///
    /// This is the starting value when `anneal_alpha` is true.
    pub distill_alpha: f32,
    /// Temperature for distillation soft targets.
    ///
    /// Higher temperatures soften the teacher distribution, transferring more
    /// information about non-peak token probabilities.  Typical values: 1.5–4.0.
    pub distill_temperature: f32,
    /// Distill with the reverse KL `KL(policy ‖ teacher)` (mode-seeking)
    /// instead of the forward KL `KL(teacher ‖ policy)` (mode-covering).
    /// Agarwal et al. find the better divergence depends on the task.
    pub reverse_kl: bool,
    /// Whether to linearly anneal alpha from `distill_alpha` toward `final_alpha`
    /// over the course of training (starting distillation-heavy, ending RL-heavy).
    pub anneal_alpha: bool,
    /// Target alpha value at the end of training when `anneal_alpha` is true.
    ///
    /// Defaults to 0.05 — almost pure RL by the end, but retaining a small
    /// regularisation signal from the teacher to prevent distribution collapse.
    pub final_alpha: f32,
}

impl Default for RlkdConfig {
    fn default() -> Self {
        Self {
            grpo: GrpoConfig::default(),
            training: TrainingConfig::default(),
            distill_alpha: 0.3,
            distill_temperature: 2.0,
            reverse_kl: false,
            anneal_alpha: true,
            final_alpha: 0.05,
        }
    }
}

impl RlkdConfig {
    /// Construct with a specific starting alpha.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.distill_alpha = alpha;
        self
    }

    /// Construct with a specific distillation temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.distill_temperature = temperature;
        self
    }

    /// Disable alpha annealing (keep `distill_alpha` constant throughout).
    pub fn without_annealing(mut self) -> Self {
        self.anneal_alpha = false;
        self
    }

    /// The blend factor `α` at optimizer step `step` of `total_steps`: linear
    /// from `distill_alpha` to `final_alpha` when annealing, else constant.
    pub fn alpha_at(&self, step: usize, total_steps: usize) -> f32 {
        if !self.anneal_alpha || total_steps == 0 {
            return self.distill_alpha;
        }
        let progress = (step as f32 / total_steps as f32).clamp(0.0, 1.0);
        self.distill_alpha + (self.final_alpha - self.distill_alpha) * progress
    }

    /// Check the RLKD options (the GRPO ones are checked by the GRPO trainer).
    pub fn validate(&self) -> GrpoResult<()> {
        let in_unit = |a: f32| a.is_finite() && (0.0..=1.0).contains(&a);
        if !in_unit(self.distill_alpha) || !in_unit(self.final_alpha) {
            return Err(GrpoError::Config(format!(
                "the distillation weights must be in [0, 1], got {} and {}",
                self.distill_alpha, self.final_alpha
            )));
        }
        if !(self.distill_temperature.is_finite() && self.distill_temperature > 0.0) {
            return Err(GrpoError::Config(format!(
                "the distillation temperature must be positive, got {}",
                self.distill_temperature
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Per-step statistics for RLKD training.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RlkdStepStats {
    /// Global step number.
    pub step: usize,
    /// Combined loss: (1 - alpha) * grpo_loss + alpha * distill_loss.
    pub total_loss: f32,
    /// The GRPO objective alone (policy loss plus any reference KL penalty).
    pub grpo_loss: f32,
    /// The distillation term alone, `T²·KL(teacher ‖ policy)`.
    pub distill_loss: f32,
    /// Mean reward over the completion group.
    pub reward: f32,
    /// The alpha blend factor in effect.
    pub alpha: f32,
    /// KL divergence from the reference model (0 when none is in use).
    pub kl: f32,
    /// Mean advantage over the completion group.
    pub advantage: f32,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// RLKD Trainer combining GRPO policy gradient with knowledge distillation.
pub struct RlkdTrainer {
    /// RLKD-specific configuration.
    pub config: RlkdConfig,
    /// The GRPO trainer whose loop this is.
    grpo: GrpoTrainer,
}

impl RlkdTrainer {
    /// Create a new RLKD trainer. Both the RLKD and the GRPO options are
    /// validated.
    pub fn new(config: RlkdConfig) -> GrpoResult<Self> {
        config.validate()?;
        let grpo = GrpoTrainer::new(config.grpo.clone(), config.training.clone())?;
        Ok(Self { config, grpo })
    }

    /// Add a training callback for metrics logging or dashboard integration.
    pub fn add_callback(&mut self, cb: Box<dyn pmetal_core::TrainingCallback>) {
        self.grpo.add_callback(cb);
    }

    /// Enable adaptive LR with a control file for TUI communication.
    pub fn enable_adaptive_lr_with_control(
        &mut self,
        config: AdaptiveLrConfig,
        control_file: std::path::PathBuf,
    ) {
        self.grpo
            .enable_adaptive_lr_with_control(config, control_file);
    }

    /// Save checkpoints through `manager`.
    pub fn set_checkpoint_manager(&mut self, manager: crate::CheckpointManager) {
        self.grpo.set_checkpoint_manager(manager);
    }

    /// Resume at optimizer step `step`.
    pub fn set_step(&mut self, step: usize) {
        self.grpo.set_step(step);
    }

    fn teacher_term<'t>(&self, teacher: &'t mut dyn TeacherModel) -> TeacherTerm<'t> {
        let config = self.config.clone();
        TeacherTerm::new(
            teacher,
            move |step, total| config.alpha_at(step, total),
            self.config.distill_temperature,
            self.config.reverse_kl,
        )
    }

    /// Run a single RLKD step on scored completion groups: the teacher's
    /// logits once, then `num_iterations` updates of the combined loss.
    pub fn train_step<M, T, R, O>(
        &mut self,
        policy: &mut M,
        teacher: &mut T,
        ref_model: Option<&mut R>,
        groups: &[CompletionGroup],
        optimizer: &mut O,
        set_lr: &mut impl FnMut(&mut O, f32),
    ) -> GrpoResult<RlkdStepStats>
    where
        M: TrainableModel,
        T: TeacherModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
    {
        let mut term = self.teacher_term(teacher);
        let stats = self
            .grpo
            .train_step_with_teacher(policy, ref_model, groups, optimizer, set_lr, &mut term)?;
        Ok(RlkdStepStats {
            step: stats.step,
            total_loss: stats.loss,
            grpo_loss: stats.rl_loss,
            distill_loss: stats.distill_loss,
            reward: stats.reward,
            alpha: stats.alpha,
            kl: stats.kl,
            advantage: stats.advantage,
        })
    }

    /// Run the full RLKD training loop over `prompts`.
    ///
    /// # Arguments
    /// * `policy_model` - Trainable LoRA student model.
    /// * `teacher_model` - Frozen teacher model (same or different architecture,
    ///   but the same tokenizer).
    /// * `ref_model` - The reference model of the GRPO KL penalty, needed only
    ///   when `grpo.beta > 0`.
    /// * `tokenizer` - Tokenizer for prompt/completion encoding.
    /// * `prompts` - Prompts to roll out from (see [`GrpoPrompt`]).
    /// * `reward_fn` - Combined reward function.
    /// * `optimizer` - Optimizer for the student.
    /// * `set_optimizer_lr` - Closure that updates the optimizer learning rate.
    #[expect(
        clippy::too_many_arguments,
        reason = "public API: the models, data and optimizer"
    )]
    pub fn run<M, T, R, O, F>(
        &mut self,
        policy_model: &mut M,
        teacher_model: &mut T,
        ref_model: Option<&mut R>,
        tokenizer: &pmetal_data::Tokenizer,
        prompts: &[GrpoPrompt],
        reward_fn: &CombinedReward,
        optimizer: &mut O,
        set_optimizer_lr: F,
    ) -> GrpoResult<()>
    where
        M: TrainableModel,
        T: TeacherModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
        F: FnMut(&mut O, f32),
    {
        tracing::info!(
            "Starting RLKD training: {} prompts | alpha={:.2} → {:.2} | T={:.1}",
            prompts.len(),
            self.config.distill_alpha,
            if self.config.anneal_alpha {
                self.config.final_alpha
            } else {
                self.config.distill_alpha
            },
            self.config.distill_temperature,
        );
        let mut term = self.teacher_term(teacher_model);
        self.grpo.run_with_teacher(
            policy_model,
            ref_model,
            tokenizer,
            prompts,
            reward_fn,
            &mut term,
            optimizer,
            set_optimizer_lr,
        )
    }
}

// ---------------------------------------------------------------------------
// CombinedReward re-export (for callers who only import from this module)
// ---------------------------------------------------------------------------

pub use crate::grpo::CombinedReward as RlkdCombinedReward;

#[cfg(test)]
mod tests {
    use super::*;

    fn annealing(start: f32, end: f32) -> RlkdConfig {
        RlkdConfig {
            anneal_alpha: true,
            distill_alpha: start,
            final_alpha: end,
            ..Default::default()
        }
    }

    #[test]
    fn alpha_is_constant_without_annealing() {
        let config = RlkdConfig::default().without_annealing();
        assert!((config.alpha_at(50, 100) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn alpha_anneals_linearly_from_start_to_end() {
        let config = annealing(0.3, 0.05);
        assert!((config.alpha_at(0, 100) - 0.3).abs() < 1e-6);
        assert!((config.alpha_at(50, 100) - 0.175).abs() < 1e-5);
        assert!((config.alpha_at(100, 100) - 0.05).abs() < 1e-6);
        // Past the end it holds, and before a schedule exists it starts.
        assert!((config.alpha_at(500, 100) - 0.05).abs() < 1e-6);
        assert!((config.alpha_at(7, 0) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn the_config_builder_sets_what_it_names() {
        let config = RlkdConfig::default()
            .with_alpha(0.5)
            .with_temperature(3.0)
            .without_annealing();
        assert!((config.distill_alpha - 0.5).abs() < 1e-6);
        assert!((config.distill_temperature - 3.0).abs() < 1e-6);
        assert!(!config.anneal_alpha);
    }

    fn llama(seed: u64) -> pmetal_models::DynamicModel {
        use pmetal_models::architectures::llama::LlamaConfig;
        pmetal_runtime::compat::random::seed(seed);
        let config = LlamaConfig {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 128,
            ..Default::default()
        };
        pmetal_models::DynamicModel::from_config(&serde_json::to_string(&config).unwrap()).unwrap()
    }

    fn scalar(a: &Array) -> f32 {
        a.eval();
        pmetal_runtime::check_unobserved_error().expect("bridge op failed");
        a.item_f32()
    }

    /// `T²·KL(teacher ‖ policy)` over the completion tokens, from the
    /// temperature-softened distributions, against float64. The last
    /// position predicts nothing and the mask drops the prompt.
    #[test]
    fn the_distillation_term_matches_the_definition() {
        struct Never;
        impl TeacherModel for Never {
            fn forward_teacher(&mut self, _: &Array) -> Result<Array, Exception> {
                unreachable!("the term only reads logits it is given")
            }
        }
        let teacher = [
            [2.0f64, 0.5, -1.0, 0.0],
            [0.1, 0.2, 0.3, 9.0],
            [5.0, 5.0, 5.0, 5.0],
        ];
        let student = [
            [0.3f64, 1.5, -0.2, 0.8],
            [1.0, 0.0, -1.0, 2.0],
            [0.0, 1.0, 2.0, 3.0],
        ];
        let flat = |rows: &[[f64; 4]; 3]| {
            Array::from_slice(
                &rows.iter().flatten().map(|&x| x as f32).collect::<Vec<_>>(),
                &[1, 3, 4],
            )
        };
        let t = 2.0f64;
        let mut model = Never;
        let term = TeacherTerm::new(&mut model, |_, _| 0.5, t as f32, false);
        // Position 0 is a completion target; position 1 is not; position 2
        // (the last) is dropped by the shift.
        let mask = Array::from_slice(&[1.0f32, 0.0], &[1, 2]);
        let got = scalar(
            &term
                .distillation(&flat(&teacher), &flat(&student), &mask)
                .unwrap(),
        );

        let softmax = |row: &[f64; 4]| {
            let z: Vec<f64> = row.iter().map(|x| (x / t).exp()).collect();
            let sum: f64 = z.iter().sum();
            z.iter().map(|x| x / sum).collect::<Vec<_>>()
        };
        let (p, q) = (softmax(&teacher[0]), softmax(&student[0]));
        let want = t * t * p.iter().zip(&q).map(|(p, q)| p * (p / q).ln()).sum::<f64>();
        assert!((got as f64 - want).abs() < 1e-5, "{got} vs {want}");
        assert_eq!(term.alpha_at(3, 10), 0.5);

        // The reverse KL is the other direction.
        let mut model = Never;
        let term = TeacherTerm::new(&mut model, |_, _| 0.5, t as f32, true);
        let got = scalar(
            &term
                .distillation(&flat(&teacher), &flat(&student), &mask)
                .unwrap(),
        );
        let want = t * t * q.iter().zip(&p).map(|(q, p)| q * (q / p).ln()).sum::<f64>();
        assert!((got as f64 - want).abs() < 1e-5, "{got} vs {want}");
    }

    /// With the whole loss on the teacher term (α = 1) a step moves the
    /// student toward the teacher: the KL on the same samples falls.
    #[test]
    fn distillation_pulls_the_student_toward_the_teacher() {
        let mut teacher = llama(1);
        let mut student = pmetal_lora::AdaptedModel::attach(
            llama(2),
            pmetal_core::LoraConfig {
                r: 4,
                alpha: 8.0,
                dropout: 0.0,
                target_modules: vec!["q_proj".into(), "v_proj".into()],
                ..Default::default()
            },
        )
        .unwrap();
        let config = RlkdConfig {
            distill_alpha: 1.0,
            distill_temperature: 1.0,
            anneal_alpha: false,
            training: TrainingConfig {
                learning_rate: 0.05,
                warmup_steps: 0,
                lr_scheduler: pmetal_core::LrSchedulerType::Constant,
                weight_decay: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut trainer = RlkdTrainer::new(config.clone()).unwrap();
        let mut optimizer = crate::TrainOptimizer::from_config(&config.training);
        let mut group = CompletionGroup::new(vec![1, 2, 3], 4);
        group.add_completion(vec![4, 5, 6], 1.0, false);
        group.add_completion(vec![7, 8], 0.0, false);
        group.add_completion(vec![9, 10, 11, 12], 1.0, false);
        group.add_completion(vec![13, 14, 15], 0.0, false);

        let mut distill = Vec::new();
        for _ in 0..30 {
            let stats = trainer
                .train_step(
                    &mut student,
                    &mut teacher,
                    None::<&mut pmetal_models::DynamicModel>,
                    std::slice::from_ref(&group),
                    &mut optimizer,
                    &mut |opt: &mut crate::TrainOptimizer, lr| opt.set_lr(lr),
                )
                .unwrap();
            pmetal_runtime::check_unobserved_error().unwrap();
            assert_eq!(stats.alpha, 1.0);
            distill.push(stats.distill_loss);
        }
        let (first, last) = (distill[0], *distill.last().unwrap());
        assert!(first > 0.0 && last.is_finite());
        assert!(last < 0.5 * first, "KL {first} -> {last}");
    }

    #[test]
    fn the_options_are_validated_at_construction() {
        assert!(RlkdTrainer::new(RlkdConfig::default()).is_ok());
        assert!(RlkdTrainer::new(RlkdConfig::default().with_alpha(1.5)).is_err());
        assert!(RlkdTrainer::new(RlkdConfig::default().with_temperature(0.0)).is_err());
        // The GRPO options are checked too.
        let bad = RlkdConfig {
            grpo: GrpoConfig {
                num_generations: 1,
                ..GrpoConfig::default()
            },
            ..Default::default()
        };
        assert!(RlkdTrainer::new(bad).is_err());
    }
}
