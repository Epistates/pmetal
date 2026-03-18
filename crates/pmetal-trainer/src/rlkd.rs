//! RLKD: Reinforcement Learning with Knowledge Distillation.
//!
//! Combines GRPO policy gradient optimization with knowledge distillation
//! from a teacher model in a single training loop. The teacher guides the
//! student's policy while RL rewards shape behavior.
//!
//! # Loss Formula
//!
//! ```text
//! L_total = (1 - alpha) * L_grpo + alpha * L_distill
//! ```
//!
//! Where:
//! - L_grpo: GRPO policy gradient loss (PPO-clip on advantages weighted by old log-probs)
//! - L_distill: KL divergence between teacher and student logits
//! - alpha: Blend factor (can be annealed from distillation-heavy to RL-heavy)
//!
//! # Architecture
//!
//! ```text
//! Teacher Model (frozen) ──→ soft targets ──→ distill_loss ──┐
//!                                                              ├──→ total_loss ──→ backward
//! Policy Model (trainable) ──→ log_probs ──→ grpo_loss  ─────┘
//!         ↑
//! Reward Function ──→ advantages
//! ```
//!
//! # Critical Implementation Notes
//!
//! - Teacher logits MUST be computed and `.eval()`'d outside the gradient closure to
//!   prevent them from being recomputed as part of the backward pass.
//! - Old log-probs (generation-time) MUST also be computed and `.eval()`'d before
//!   the `value_and_grad` closure, matching the GRPO training loop.
//! - The distillation loss uses the SAME student logits computed inside the closure,
//!   so both objectives share one forward pass per step.

use mlx_rs::{
    Array,
    error::Exception,
    nn,
    ops::indexing::IndexOp,
    optimizers::Optimizer,
};
use pmetal_core::{EvalMetrics, TrainingConfig};
use pmetal_lora::TrainableModel;
use std::time::Instant;
use tracing::info;

use crate::{
    adaptive_lr::{AdaptiveLrConfig, AdaptiveLrController},
    grpo::{CompletionGroup, CombinedReward, GrpoConfig, GrpoError, GrpoResult, GrpoTrainer},
    training_loop::AdaptiveAction,
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
    fn forward_teacher(
        &mut self,
        input_ids: &Array,
    ) -> Result<Array, Exception>;
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
    /// Whether to linearly anneal alpha from `distill_alpha` toward `final_alpha`
    /// over the course of training (starting distillation-heavy, ending RL-heavy).
    pub anneal_alpha: bool,
    /// Target alpha value at the end of training when `anneal_alpha` is true.
    ///
    /// Defaults to 0.05 — almost pure RL by the end, but retaining a small
    /// regularisation signal from the teacher to prevent distribution collapse.
    pub final_alpha: f32,
    /// Log metrics every N steps.
    pub log_every: usize,
    /// Enable adaptive LR (spike/plateau/divergence detection + rollback).
    pub adaptive_lr: bool,
}

impl Default for RlkdConfig {
    fn default() -> Self {
        Self {
            grpo: GrpoConfig::default(),
            training: TrainingConfig::default(),
            distill_alpha: 0.3,
            distill_temperature: 2.0,
            anneal_alpha: true,
            final_alpha: 0.05,
            log_every: 10,
            adaptive_lr: true,
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
    /// GRPO policy gradient component of the loss.
    pub grpo_loss: f32,
    /// Distillation KL component of the loss.
    pub distill_loss: f32,
    /// Mean reward over the completion group.
    pub reward: f32,
    /// Current alpha blend factor.
    pub alpha: f32,
    /// KL divergence from the reference model (0 when no reference provided).
    pub kl: f32,
    /// Mean advantage over the completion group.
    pub advantage: f32,
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// RLKD Trainer combining GRPO policy gradient with knowledge distillation.
///
/// Uses the existing [`GrpoTrainer`] for generation, advantage computation, and
/// per-token log-prob utilities, then adds a distillation loss component against
/// a frozen teacher model in the same backward pass.
pub struct RlkdTrainer {
    /// RLKD-specific configuration.
    pub config: RlkdConfig,
    /// Underlying GRPO trainer used for generation and GRPO loss utilities.
    grpo_trainer: GrpoTrainer,
    /// Current global step.
    step: usize,
    /// Total steps for the run (used for alpha annealing).
    total_steps: usize,
    /// Adaptive LR controller.
    adaptive_lr: Option<AdaptiveLrController>,
    /// Cached adaptive LR override.
    adaptive_lr_override: Option<f32>,
    /// In-memory snapshot of the best LoRA weights for rollback.
    best_lora_snapshot: Option<std::collections::HashMap<std::rc::Rc<str>, Array>>,
    /// Training callbacks.
    callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>>,
}

impl RlkdTrainer {
    /// Create a new RLKD trainer.
    pub fn new(config: RlkdConfig) -> GrpoResult<Self> {
        let grpo_trainer = GrpoTrainer::new(config.grpo.clone(), config.training.clone())?;
        Ok(Self {
            config,
            grpo_trainer,
            step: 0,
            total_steps: 0,
            adaptive_lr: None,
            adaptive_lr_override: None,
            best_lora_snapshot: None,
            callbacks: Vec::new(),
        })
    }

    /// Add a training callback for metrics logging or dashboard integration.
    pub fn add_callback(&mut self, cb: Box<dyn pmetal_core::TrainingCallback>) {
        self.callbacks.push(cb);
    }

    /// Enable adaptive LR with a control file for TUI communication.
    pub fn enable_adaptive_lr_with_control(
        &mut self,
        config: AdaptiveLrConfig,
        control_file: std::path::PathBuf,
    ) {
        self.adaptive_lr = Some(
            AdaptiveLrController::new(config).with_control_file(control_file),
        );
    }

    // -----------------------------------------------------------------------
    // Alpha scheduling
    // -----------------------------------------------------------------------

    /// Current distillation blend alpha, with optional linear annealing.
    ///
    /// Anneals linearly from `distill_alpha` → `final_alpha` over `total_steps`.
    fn current_alpha(&self) -> f32 {
        if !self.config.anneal_alpha || self.total_steps == 0 {
            return self.config.distill_alpha;
        }
        let progress = (self.step as f32 / self.total_steps as f32).clamp(0.0, 1.0);
        let start = self.config.distill_alpha;
        let end = self.config.final_alpha;
        start + (end - start) * progress
    }

    // -----------------------------------------------------------------------
    // Adaptive LR helpers (mirrors GrpoTrainer)
    // -----------------------------------------------------------------------

    fn get_learning_rate(&self) -> f32 {
        self.adaptive_lr_override
            .unwrap_or(self.config.training.learning_rate as f32)
    }

    fn apply_adaptive_lr_action(&mut self, loss: f64) -> AdaptiveAction {
        let scheduled = self.config.training.learning_rate;
        let step = self.step;
        if let Some(ref mut ctrl) = self.adaptive_lr {
            let (adjusted, event) = ctrl.step(step, loss, scheduled);
            self.adaptive_lr_override = Some(adjusted as f32);

            let action = match &event {
                crate::adaptive_lr::LrEvent::RollbackTriggered { new_lr, .. } => {
                    self.adaptive_lr_override = Some(*new_lr as f32);
                    AdaptiveAction::Rollback
                }
                crate::adaptive_lr::LrEvent::EarlyStop { .. } => AdaptiveAction::EarlyStop,
                _ => AdaptiveAction::Continue,
            };

            if !matches!(event, crate::adaptive_lr::LrEvent::Scheduled) {
                for cb in &mut self.callbacks {
                    cb.on_lr_event(&format!("{event}"));
                }
            }
            action
        } else {
            AdaptiveAction::Continue
        }
    }

    fn should_snapshot_best(&mut self) -> bool {
        if let Some(ref mut ctrl) = self.adaptive_lr {
            ctrl.should_snapshot_best(self.step)
        } else {
            false
        }
    }

    fn snapshot_best_weights<M: TrainableModel>(&mut self, model: &M) {
        let params = model.lora_parameters();
        tracing::debug!(
            "RLKD snapshot: saved best LoRA weights at step {} ({} params)",
            self.step,
            params.len(),
        );
        self.best_lora_snapshot = Some(params);
    }

    fn restore_best_weights<M: TrainableModel>(&mut self, model: &mut M) -> bool {
        if let Some(ref snapshot) = self.best_lora_snapshot {
            model.set_lora_parameters(snapshot);
            if let Some(ref mut ctrl) = self.adaptive_lr {
                ctrl.on_rollback_complete();
            }
            tracing::info!("RLKD rollback: restored best weights at step {}", self.step);
            true
        } else {
            tracing::warn!("RLKD rollback requested but no snapshot available");
            false
        }
    }

    // -----------------------------------------------------------------------
    // Distillation loss
    // -----------------------------------------------------------------------

    /// Compute forward KL distillation loss between teacher and student logits.
    ///
    /// Uses temperature-scaled KL divergence:
    ///
    /// ```text
    /// L_distill = T^2 * KL(softmax(teacher/T) || softmax(student/T))
    ///           = T^2 * sum_v [ p_t(v) * (log p_t(v) - log p_s(v)) ]
    /// ```
    ///
    /// The T^2 factor maintains gradient magnitude under temperature scaling,
    /// following Hinton et al. (2015).
    ///
    /// Only completion tokens contribute to the loss: the `completion_mask` zeroes
    /// out prompt and padding positions so the teacher's signal is applied only
    /// where the policy is generating new content.
    ///
    /// # Arguments
    /// * `teacher_logits` - Logits from the frozen teacher, `[batch, seq, vocab]`
    /// * `student_logits` - Logits from the policy model, `[batch, seq, vocab]`
    /// * `completion_mask` - Binary mask `[batch, seq-1]` (1 = completion token)
    /// * `temperature` - Softmax temperature T
    fn compute_distill_loss(
        teacher_logits: &Array,
        student_logits: &Array,
        completion_mask: &Array,
        temperature: f32,
    ) -> Result<Array, Exception> {
        let t = Array::from_f32(temperature);
        let t_sq = Array::from_f32(temperature * temperature);

        // Shift logits to align with the completion mask (which is already seq-1).
        // Both teacher and student have shape [batch, seq, vocab].
        // The mask is [batch, seq-1] — drop the last position of the logits.
        let seq = teacher_logits.dim(1);
        let teacher_shifted = teacher_logits.index((.., ..seq - 1, ..));
        let student_shifted = student_logits.index((.., ..seq - 1, ..));

        // Scale by temperature
        let teacher_scaled = teacher_shifted.divide(&t)?;
        let student_scaled = student_shifted.divide(&t)?;

        // log-softmax for numerical stability
        let teacher_log_probs = mlx_rs::nn::log_softmax(&teacher_scaled, -1)?;
        let student_log_probs = mlx_rs::nn::log_softmax(&student_scaled, -1)?;
        let teacher_probs = teacher_log_probs.exp()?;

        // Forward KL: sum_v [ p_t * (log p_t - log p_s) ]
        let kl_per_vocab = teacher_probs.multiply(&teacher_log_probs.subtract(&student_log_probs)?)?;
        // Sum over vocab dimension → [batch, seq-1]
        let kl_per_token = kl_per_vocab.sum_axes(&[-1], Some(false))?;

        // Apply completion mask and take mean over valid tokens
        let masked_kl = kl_per_token.multiply(completion_mask)?;
        let total_tokens = completion_mask.sum(None)?;
        let safe_tokens = mlx_rs::ops::maximum(&total_tokens, &Array::from_f32(1.0))?;
        let mean_kl = masked_kl.sum(None)?.divide(&safe_tokens)?;

        // Scale by T^2 to preserve gradient magnitude
        mean_kl.multiply(&t_sq)
    }

    // -----------------------------------------------------------------------
    // Core training step
    // -----------------------------------------------------------------------

    /// Run a single RLKD training step.
    ///
    /// Steps:
    /// 1. Build padded input_ids / labels tensors from completion groups.
    /// 2. Compute `old_per_token_logps` from the current policy (generation snapshot).
    /// 3. Compute teacher logits (frozen, outside gradient tape).
    /// 4. Run `value_and_grad` combining GRPO loss and distillation loss.
    /// 5. Update optimizer.
    pub fn train_step<M, T, O>(
        &mut self,
        policy: &mut M,
        teacher: &mut T,
        groups: &[CompletionGroup],
        optimizer: &mut O,
        alpha: f32,
    ) -> GrpoResult<RlkdStepStats>
    where
        M: TrainableModel,
        T: TeacherModel,
        O: Optimizer,
    {
        let start_time = Instant::now();

        // --- 1. Build batch tensors from completion groups ---
        let (all_prompts, all_completions, advantages, _all_masks, _) =
            self.grpo_trainer.prepare_batch(groups)?;

        let raw_rewards: Vec<f64> = groups
            .iter()
            .flat_map(|g| g.rewards.iter().copied())
            .collect();

        let n_completions = all_completions.len();
        let adv_array = Array::from_slice(
            &advantages.iter().map(|&a| a as f32).collect::<Vec<_>>(),
            &[n_completions as i32],
        );

        let max_len = all_prompts
            .iter()
            .zip(all_completions.iter())
            .map(|(p, c)| p.len() + c.len())
            .max()
            .unwrap_or(0);

        let mut input_ids_vec = Vec::with_capacity(n_completions * max_len);
        let mut labels_vec = Vec::with_capacity(n_completions * max_len);

        for (p, c) in all_prompts.iter().zip(all_completions.iter()) {
            let mut ids = p.clone();
            ids.extend(c);

            let mut labels = vec![-100i32; p.len()];
            labels.extend(c.iter().map(|&id| id as i32));

            let pad_len = max_len - ids.len();
            ids.extend(vec![0u32; pad_len]);
            labels.extend(vec![-100i32; pad_len]);

            input_ids_vec.extend(ids.iter().map(|&id| id as i32));
            labels_vec.extend(labels);
        }

        let input_ids = Array::from_slice(&input_ids_vec, &[n_completions as i32, max_len as i32]);
        let labels = Array::from_slice(&labels_vec, &[n_completions as i32, max_len as i32]);

        // Temperature for log-prob computation (None = 1.0)
        let temperature = if (self.config.grpo.temperature - 1.0).abs() > 1e-8 {
            Some(self.config.grpo.temperature as f32)
        } else {
            None
        };

        // --- 2. Compute old_per_token_logps from current policy ---
        // These represent the generation-time policy and are detached from the graph.
        let old_logits = policy
            .forward(&input_ids, None)
            .map_err(|e| GrpoError::Mlx(Exception::custom(e.to_string())))?;
        let (old_per_token_logps, completion_mask) = self
            .grpo_trainer
            .compute_per_token_logps(&old_logits, &labels, temperature)?;
        old_per_token_logps.eval()?;
        completion_mask.eval()?;

        // --- 3. Compute teacher logits (frozen, outside gradient tape) ---
        // We materialize these before entering value_and_grad so they become
        // constants in the backward pass — no teacher gradients are computed.
        let teacher_logits = teacher
            .forward_teacher(&input_ids)
            .map_err(|e| GrpoError::Mlx(e))?;
        teacher_logits.eval()?;

        let distill_temp = self.config.distill_temperature;

        // --- 4. Combined RLKD loss function ---
        let loss_fn = |model: &mut M,
                       (input_ids, labels, adv_array, old_logps, comp_mask): (
            &Array,
            &Array,
            &Array,
            &Array,
            &Array,
        )|
         -> Result<Array, Exception> {
            // Single forward pass — logits used by both GRPO and distillation
            let policy_logits = model
                .forward(input_ids, None)
                .map_err(|e| Exception::custom(e.to_string()))?;

            // GRPO: per-token log-probs from current policy
            let (per_token_logps, _) = self
                .grpo_trainer
                .compute_per_token_logps(&policy_logits, labels, temperature)
                .map_err(|e| Exception::custom(e.to_string()))?;

            // GRPO loss: PPO-clip + KL penalty (ref model path not wired here;
            // the KL toward ref is handled separately via grpo_trainer.train_step
            // when a reference model is provided — RLKD uses the teacher as the
            // knowledge signal instead of a separate reference model).
            let (grpo_loss, _kl, _policy_loss) = self
                .grpo_trainer
                .compute_grpo_loss(
                    &per_token_logps,
                    old_logps,
                    None, // No separate reference model; teacher serves as regularizer
                    adv_array,
                    comp_mask,
                    None,
                )
                .map_err(|e| Exception::custom(e.to_string()))?;

            // Distillation loss: KL(teacher || student) with temperature scaling
            let distill_loss = Self::compute_distill_loss(
                &teacher_logits,
                &policy_logits,
                comp_mask,
                distill_temp,
            )?;

            // Combined: (1 - alpha) * L_grpo + alpha * L_distill
            let alpha_arr = Array::from_f32(alpha);
            let one_minus_alpha = Array::from_f32(1.0 - alpha);
            let total = one_minus_alpha
                .multiply(&grpo_loss)?
                .add(&alpha_arr.multiply(&distill_loss)?)?;

            Ok(total)
        };

        let (total_loss_arr, grads) = {
            let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
            loss_and_grad_fn(
                policy,
                (
                    &input_ids,
                    &labels,
                    &adv_array,
                    &old_per_token_logps,
                    &completion_mask,
                ),
            )?
        };

        // --- 5. Update optimizer ---
        optimizer.update(policy, grads)?;
        mlx_rs::transforms::eval_params(policy.parameters())?;

        let total_loss = total_loss_arr.item::<f32>();

        // NOTE: These are proportional approximations, not actual component values.
        // Actual decomposition would require separate forward passes or returning
        // both losses from the closure.  For monitoring, the total_loss and alpha
        // are the authoritative metrics.
        let grpo_component = total_loss * (1.0 - alpha);
        let distill_component = total_loss * alpha;

        let mean_reward = if raw_rewards.is_empty() {
            0.0
        } else {
            raw_rewards.iter().sum::<f64>() / raw_rewards.len() as f64
        } as f32;

        let mean_adv = if advantages.is_empty() {
            0.0
        } else {
            advantages.iter().sum::<f64>() / advantages.len() as f64
        } as f32;

        self.step += 1;

        Ok(RlkdStepStats {
            step: self.step,
            total_loss,
            grpo_loss: grpo_component,
            distill_loss: distill_component,
            reward: mean_reward,
            alpha,
            kl: 0.0, // KL vs reference not computed in RLKD (teacher acts as regularizer)
            advantage: mean_adv,
        })
    }

    // -----------------------------------------------------------------------
    // Full training loop
    // -----------------------------------------------------------------------

    /// Run the full RLKD training loop.
    ///
    /// # Arguments
    /// * `policy_model` - Trainable LoRA student model.
    /// * `teacher_model` - Frozen teacher model (same or different architecture).
    /// * `tokenizer` - Tokenizer for prompt/completion encoding.
    /// * `dataset` - Training dataset (prompt samples).
    /// * `reward_fn` - Combined reward function.
    /// * `optimizer` - Optimizer for the student.
    /// * `set_optimizer_lr` - Closure that updates the optimizer learning rate.
    pub fn run<M, T, O, F>(
        &mut self,
        policy_model: &mut M,
        teacher_model: &mut T,
        tokenizer: &pmetal_data::Tokenizer,
        dataset: &pmetal_data::TrainingDataset,
        reward_fn: &CombinedReward,
        optimizer: &mut O,
        mut set_optimizer_lr: F,
    ) -> GrpoResult<()>
    where
        M: TrainableModel,
        T: TeacherModel,
        O: Optimizer,
        F: FnMut(&mut O, f32),
    {
        let n_epochs = self.config.training.num_epochs;
        let n_samples = dataset.samples().len();
        self.total_steps = n_samples * n_epochs;

        if let Some(ref mut ctrl) = self.adaptive_lr {
            ctrl.set_total_steps(self.total_steps);
        }

        info!(
            "Starting RLKD training: {} samples × {} epochs = {} steps | alpha={:.2} → {:.2} | T={:.1}",
            n_samples,
            n_epochs,
            self.total_steps,
            self.config.distill_alpha,
            if self.config.anneal_alpha { self.config.final_alpha } else { self.config.distill_alpha },
            self.config.distill_temperature,
        );

        for cb in &mut self.callbacks {
            cb.on_train_start();
        }

        'outer: for epoch in 0..n_epochs {
            info!("RLKD Epoch {}/{}", epoch + 1, n_epochs);

            for (i, sample) in dataset.samples().iter().enumerate() {
                let step_start = Instant::now();

                // --- Generation phase ---
                let gen_output = self
                    .grpo_trainer
                    .generate_completions(policy_model, &sample.input_ids, tokenizer)?;

                let prompt_text = tokenizer
                    .decode(&sample.input_ids)
                    .map_err(|e| GrpoError::Tokenizer(e.to_string()))?;

                let mut completions_text = Vec::new();
                for ids in &gen_output.token_ids {
                    let new_ids = &ids[sample.input_ids.len()..];
                    completions_text.push(
                        tokenizer
                            .decode(new_ids)
                            .map_err(|e| GrpoError::Tokenizer(e.to_string()))?,
                    );
                }

                // --- Reward computation ---
                let rewards = reward_fn.compute(
                    &vec![prompt_text; gen_output.token_ids.len()],
                    &completions_text,
                    None,
                )?;

                // --- Build completion group ---
                let mut group = CompletionGroup::new(
                    sample.input_ids.clone(),
                    self.config.grpo.num_generations,
                );
                for (j, ids) in gen_output.token_ids.iter().enumerate() {
                    let new_ids = ids[sample.input_ids.len()..].to_vec();
                    group.add_completion(new_ids, rewards[j], gen_output.stopped_by_length[j]);
                }

                // --- Apply adaptive LR override ---
                let current_lr = self.get_learning_rate();
                set_optimizer_lr(optimizer, current_lr);

                // --- RLKD training step ---
                let alpha = self.current_alpha();
                let stats = self.train_step(
                    policy_model,
                    teacher_model,
                    &[group],
                    optimizer,
                    alpha,
                )?;

                // --- Adaptive LR / rollback / early stop ---
                let action = self.apply_adaptive_lr_action(stats.total_loss as f64);

                if action == AdaptiveAction::Continue && self.should_snapshot_best() {
                    self.snapshot_best_weights(policy_model);
                }

                if action == AdaptiveAction::Rollback {
                    self.restore_best_weights(policy_model);
                    let rollback_lr = self
                        .adaptive_lr_override
                        .unwrap_or(self.config.training.learning_rate as f32);
                    set_optimizer_lr(optimizer, rollback_lr);
                    tracing::info!(
                        "RLKD rollback at step {}: new lr={:.2e}",
                        self.step,
                        rollback_lr
                    );
                }

                if action == AdaptiveAction::EarlyStop {
                    self.restore_best_weights(policy_model);
                    tracing::info!(
                        "RLKD early stop at step {} — adaptive LR exhausted rollbacks.",
                        self.step
                    );
                    break 'outer;
                }

                // --- Logging ---
                let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                if self.step % self.config.log_every == 0 {
                    info!(
                        "RLKD step {}: loss={:.4} grpo={:.4} distill={:.4} reward={:.3} alpha={:.3} lr={:.2e}",
                        stats.step,
                        stats.total_loss,
                        stats.grpo_loss,
                        stats.distill_loss,
                        stats.reward,
                        stats.alpha,
                        self.get_learning_rate(),
                    );
                }

                // --- Callbacks ---
                if !self.callbacks.is_empty() {
                    let lr = self.get_learning_rate();
                    let metrics = pmetal_core::StepMetrics {
                        step: self.step,
                        epoch,
                        total_epochs: n_epochs,
                        total_steps: self.total_steps,
                        loss: stats.total_loss as f64,
                        lr: lr as f64,
                        tok_sec: 0.0,
                        total_ms: step_ms,
                        tokens: 0,
                        ..Default::default()
                    };
                    for cb in &mut self.callbacks {
                        cb.on_step_end_with_metrics(&metrics);
                    }
                    if self.callbacks.iter().any(|cb| cb.should_stop()) {
                        return Err(GrpoError::Cancelled);
                    }
                }

                let _ = i; // suppress unused-variable lint for loop index
            }

            let epoch_metrics = EvalMetrics::default();
            for cb in &mut self.callbacks {
                cb.on_epoch_end(epoch, &epoch_metrics);
            }
        }

        for cb in &mut self.callbacks {
            cb.on_train_end();
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CombinedReward re-export (for callers who only import from this module)
// ---------------------------------------------------------------------------

pub use crate::grpo::CombinedReward as RlkdCombinedReward;

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn make_config() -> RlkdConfig {
        RlkdConfig {
            anneal_alpha: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_current_alpha_no_anneal() {
        let mut trainer = RlkdTrainer::new(make_config()).unwrap();
        trainer.total_steps = 100;
        trainer.step = 50;
        assert!((trainer.current_alpha() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_current_alpha_anneal_start() {
        let mut trainer = RlkdTrainer::new(RlkdConfig {
            anneal_alpha: true,
            distill_alpha: 0.3,
            final_alpha: 0.05,
            ..Default::default()
        })
        .unwrap();
        trainer.total_steps = 100;
        trainer.step = 0;
        assert!((trainer.current_alpha() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_current_alpha_anneal_end() {
        let mut trainer = RlkdTrainer::new(RlkdConfig {
            anneal_alpha: true,
            distill_alpha: 0.3,
            final_alpha: 0.05,
            ..Default::default()
        })
        .unwrap();
        trainer.total_steps = 100;
        trainer.step = 100;
        assert!((trainer.current_alpha() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_current_alpha_anneal_midpoint() {
        let mut trainer = RlkdTrainer::new(RlkdConfig {
            anneal_alpha: true,
            distill_alpha: 0.3,
            final_alpha: 0.05,
            ..Default::default()
        })
        .unwrap();
        trainer.total_steps = 100;
        trainer.step = 50;
        let expected = 0.3 + (0.05 - 0.3) * 0.5; // 0.175
        assert!((trainer.current_alpha() - expected).abs() < 1e-5);
    }

    #[test]
    #[serial]
    fn test_distill_loss_identical_distributions() {
        // KL(p || p) = 0 regardless of temperature
        let logits = Array::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            &[1, 2, 4], // [batch=1, seq=2, vocab=4]
        );
        // Completion mask covers the first (and only) shifted token
        let mask = Array::from_slice(&[1.0_f32], &[1, 1]);

        let loss =
            RlkdTrainer::compute_distill_loss(&logits, &logits, &mask, 2.0).unwrap();
        loss.eval().unwrap();
        let value: f32 = loss.item();

        assert!(
            value.abs() < 1e-4,
            "KL(p||p) should be ~0, got {}",
            value
        );
    }

    #[test]
    #[serial]
    fn test_distill_loss_different_distributions() {
        // KL(p || q) should be positive when distributions differ
        let teacher = Array::from_slice(
            &[4.0_f32, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
            &[1, 2, 4],
        );
        let student = Array::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            &[1, 2, 4],
        );
        let mask = Array::from_slice(&[1.0_f32], &[1, 1]);

        let loss = RlkdTrainer::compute_distill_loss(&teacher, &student, &mask, 2.0).unwrap();
        loss.eval().unwrap();
        let value: f32 = loss.item();

        assert!(value > 0.0, "KL must be positive, got {}", value);
        assert!(value.is_finite(), "KL must be finite, got {}", value);
    }

    #[test]
    #[serial]
    fn test_distill_loss_mask_zeroes_prompt_tokens() {
        // When mask is all-zero, the loss should be 0 (no completion tokens to learn from)
        let teacher = Array::from_slice(
            &[4.0_f32, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
            &[1, 2, 4],
        );
        let student = Array::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            &[1, 2, 4],
        );
        // Mask is all zeros — clamp denominator floors at 1.0 so result = 0 / 1 = 0
        let mask = Array::from_slice(&[0.0_f32], &[1, 1]);

        let loss = RlkdTrainer::compute_distill_loss(&teacher, &student, &mask, 2.0).unwrap();
        loss.eval().unwrap();
        let value: f32 = loss.item();

        assert!(value.abs() < 1e-6, "masked loss should be 0, got {}", value);
    }

    #[test]
    #[serial]
    fn test_distill_loss_temperature_effect() {
        // Higher temperature softens distributions → lower KL divergence
        let teacher = Array::from_slice(
            &[4.0_f32, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
            &[1, 2, 4],
        );
        let student = Array::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            &[1, 2, 4],
        );
        let mask = Array::from_slice(&[1.0_f32], &[1, 1]);

        let loss_t2 = RlkdTrainer::compute_distill_loss(&teacher, &student, &mask, 2.0)
            .unwrap()
            .item::<f32>();
        let loss_t4 = RlkdTrainer::compute_distill_loss(&teacher, &student, &mask, 4.0)
            .unwrap()
            .item::<f32>();

        // Note: T^2 scaling means the loss can increase at higher T — this is expected
        // and correct (it preserves gradient magnitude).  Both values must be finite and positive.
        assert!(loss_t2.is_finite(), "T=2 loss must be finite: {}", loss_t2);
        assert!(loss_t4.is_finite(), "T=4 loss must be finite: {}", loss_t4);
        assert!(loss_t2 > 0.0, "T=2 loss must be positive: {}", loss_t2);
        assert!(loss_t4 > 0.0, "T=4 loss must be positive: {}", loss_t4);
    }

    #[test]
    fn test_rlkd_config_builder() {
        let config = RlkdConfig::default()
            .with_alpha(0.5)
            .with_temperature(3.0)
            .without_annealing();

        assert!((config.distill_alpha - 0.5).abs() < 1e-6);
        assert!((config.distill_temperature - 3.0).abs() < 1e-6);
        assert!(!config.anneal_alpha);
    }
}
