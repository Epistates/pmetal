//! Group Relative Policy Optimization (GRPO).
//!
//! GRPO is a reinforcement learning algorithm that optimizes a policy by
//! comparing the completions it samples for the same prompt: a completion's
//! advantage is its reward relative to its group's, so no value network is
//! needed. It is effective for reasoning models (DeepSeek-R1).
//!
//! # Objective
//!
//! A step samples a group of completions per prompt (one batched decode, the
//! prompt prefilled once), scores them, turns the rewards into group-relative
//! advantages ([`group_advantages`]) and minimizes PPO's clipped surrogate
//! ([`grpo_loss`]). Every variant of that objective is an explicit option of
//! [`GrpoConfig`], with the presets of the papers that proposed them:
//!
//! - how losses are aggregated ([`GrpoLossType`]): per sequence (GRPO,
//!   arXiv 2402.03300), per token (DAPO, arXiv 2503.14476) or by a constant
//!   (Dr. GRPO, arXiv 2503.20783);
//! - the level of the importance ratio ([`ImportanceSampling`]): per token or
//!   per sequence (GSPO, arXiv 2507.18071);
//! - clip-higher, dynamic sampling and the overlong penalty (DAPO);
//! - the KL penalty's estimator ([`KlEstimator`]);
//! - whether advantages are divided by the group's standard deviation.
//!
//! The policy's log-probs are those of the model's own tempered
//! distribution, the one the rollouts sample from (no top-k, top-p or min-p
//! truncation by default), so the ratios describe the samples.

mod config;
mod job;
mod loss;
mod prompts;
mod rewards;
mod teacher;

pub(crate) use teacher::TeacherTerm;

pub use config::*;
pub use job::{GrpoJobOptions, GrpoJobResult, rewards_from_spec, run_grpo_job};
pub use loss::{GrpoLoss, group_advantages, grpo_loss};
pub use prompts::{GrpoPrompt, PromptColumns, load_prompts, load_prompts_with};
pub use rewards::*;

use pmetal_core::{ConfigValidator, TrainingConfig};
use pmetal_lora::TrainableModel;
use pmetal_runtime::compat::FlattenedModuleParam;
use pmetal_runtime::compat::{
    Array, Exception,
    module::{Module, ModuleParameters},
    nn, ops,
    optimizers::Optimizer,
    transforms,
};
use std::collections::VecDeque;
use std::time::Instant;
use tracing::info;

/// Iteration statistics for GRPO training.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GrpoIterationStats {
    /// Training step.
    pub step: usize,
    /// Total loss.
    pub loss: f32,
    /// KL divergence between policy and reference.
    pub kl: f32,
    /// Policy gradient loss.
    pub policy_loss: f32,
    /// Share of completion tokens whose objective clipping changed, averaged
    /// over the batch's updates. Always 0 with one update per batch.
    pub clip_fraction: f32,
    /// Optimizer updates taken on this generation batch.
    pub iterations: usize,
    /// Mean reward for this batch.
    pub reward: f32,
    /// Mean advantage for this batch.
    pub advantage: f32,
    /// The GRPO objective alone (policy loss plus the KL penalty), before any
    /// teacher term is blended in; equal to `loss` without a teacher.
    pub rl_loss: f32,
    /// The teacher distillation term, `T²·KL(teacher ‖ policy)` over the
    /// completion tokens, averaged over the batch's updates (0 without a
    /// teacher; see RLKD).
    pub distill_loss: f32,
    /// The weight `α` of the distillation term in the loss (0 without a teacher).
    pub alpha: f32,
    /// Mean completion length of this batch, in tokens.
    pub completion_length: f32,
    /// Generation throughput (completions/sec).
    pub completions_per_second: f32,
}

/// Error type for GRPO training.
#[derive(Debug, thiserror::Error)]
pub enum GrpoError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Generation error.
    #[error("Generation error: {0}")]
    Generation(String),
    /// Reward computation error.
    #[error("Reward error: {0}")]
    Reward(String),
    /// Tokenizer error.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    /// Training was cancelled by a callback.
    #[error("Training cancelled")]
    Cancelled,
}

/// Result type for GRPO operations.
pub type GrpoResult<T> = std::result::Result<T, GrpoError>;

/// A batch flattened from completion groups: prompt ids, completion ids and
/// advantage, one entry per completion.
pub type PreparedBatch = (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<f64>);

/// Completion group for a single prompt.
#[derive(Debug, Clone)]
pub struct CompletionGroup {
    /// The prompt's tokens.
    pub prompt_ids: Vec<u32>,
    /// Each completion's tokens (a stop token included).
    pub completion_ids: Vec<Vec<u32>>,
    /// Each completion's reward.
    pub rewards: Vec<f64>,
    /// Whether each completion ended at the length limit rather than a stop token.
    pub stopped_by_length: Vec<bool>,
    /// Optional preprocessed pixel values for VLM training.
    ///
    /// When VLM mode is active this holds the images loaded from the corresponding
    /// prompt.  Each element is one image as an MLX array of shape `[1, C, H, W]`
    /// (NCHW float32, model-specific normalization). All completions in this
    /// group share the same images — they all come from the same prompt.
    pub pixel_values: Option<Vec<Array>>,
}

impl CompletionGroup {
    /// An empty group for `prompt_ids`, with room for `num_generations`.
    pub fn new(prompt_ids: Vec<u32>, num_generations: usize) -> Self {
        Self {
            prompt_ids,
            completion_ids: Vec::with_capacity(num_generations),
            rewards: Vec::with_capacity(num_generations),
            stopped_by_length: Vec::with_capacity(num_generations),
            pixel_values: None,
        }
    }

    /// Add a completion with its reward.
    pub fn add_completion(&mut self, ids: Vec<u32>, reward: f64, stopped_by_length: bool) {
        self.completion_ids.push(ids);
        self.rewards.push(reward);
        self.stopped_by_length.push(stopped_by_length);
    }
}

/// Load and preprocess images from file paths into MLX arrays.
///
/// Each returned array has shape `[1, C, H, W]` (NCHW float32) with CLIP-style
/// normalization (mean/std from `FixedSizeImageProcessorConfig::default()`).
/// The image is resized to fit within `max_size × max_size` preserving aspect ratio.
fn load_images(image_paths: &[std::path::PathBuf], max_size: usize) -> GrpoResult<Vec<Array>> {
    use pmetal_data::image_processing::{FixedSizeImageProcessor, FixedSizeImageProcessorConfig};

    // Use CLIP-canonical normalization; the size will be overridden below.
    let config = FixedSizeImageProcessorConfig {
        size: (max_size as u32, max_size as u32),
        ..Default::default()
    };
    let processor = FixedSizeImageProcessor::new(config);

    let mut images = Vec::with_capacity(image_paths.len());
    for path in image_paths {
        let img = image::open(path).map_err(|e| {
            GrpoError::Generation(format!("Failed to open image {}: {}", path.display(), e))
        })?;

        // Resize preserving aspect ratio so neither dimension exceeds max_size.
        let (orig_w, orig_h) = (img.width(), img.height());
        let scale = (max_size as f32 / orig_w.max(orig_h) as f32).min(1.0);
        let new_w = ((orig_w as f32 * scale).round() as u32).max(1);
        let new_h = ((orig_h as f32 * scale).round() as u32).max(1);
        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3);

        // Delegate normalization to the existing processor (rescale + CLIP stats).
        let arr = processor.process_image(&resized).map_err(GrpoError::Mlx)?;

        images.push(arr);
    }
    Ok(images)
}

/// Stack a slice of per-image arrays into a single batched pixel-values tensor.
///
/// Each input array has shape `[1, C, H, W]`.  The output is `[N, C, H, W]`
/// where N = number of images.  Returns `None` for an empty slice.
fn stack_pixel_values(images: &[Array]) -> Option<Array> {
    if images.is_empty() {
        return None;
    }
    let refs: Vec<&Array> = images.iter().collect();
    Some(ops::concatenate_axis(&refs, 0))
}

/// What the loop does after a batch has been learned from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Continue,
    /// The adaptive-LR controller ran out of rollbacks.
    EarlyStop,
    /// A callback asked to stop.
    Cancelled,
}

/// A group that has been sampled and handed to the scorer.
struct PendingBatch {
    prompt_ids: Vec<u32>,
    /// (completion tokens, stopped by length)
    completions: Vec<(Vec<u32>, bool)>,
    pixel_values: Option<Vec<Array>>,
    started: Instant,
}

/// What a batch is scored from.
struct ScoreRequest {
    prompts: Vec<String>,
    completions: Vec<String>,
    references: Vec<Option<String>>,
    columns: Vec<DatasetRow>,
    images: Option<Vec<Vec<Array>>>,
}

/// Where rewards come from: scored on the spot, or by a background thread
/// that finishes a batch while the next is generated.
trait Scorer {
    /// Submit a request; the rewards of the oldest unscored batch, if any are
    /// ready (the synchronous scorer returns this request's own).
    fn submit(&mut self, request: ScoreRequest) -> GrpoResult<Option<Vec<f64>>>;
    /// The rewards of a batch still in flight.
    fn flush(&mut self) -> GrpoResult<Option<Vec<f64>>>;
}

struct SyncScorer<'a>(&'a CombinedReward);

impl Scorer for SyncScorer<'_> {
    fn submit(&mut self, request: ScoreRequest) -> GrpoResult<Option<Vec<f64>>> {
        self.0
            .score(&RewardBatch {
                prompts: &request.prompts,
                completions: &request.completions,
                references: &request.references,
                columns: &request.columns,
                images: request.images.as_deref(),
            })
            .map(Some)
    }

    fn flush(&mut self) -> GrpoResult<Option<Vec<f64>>> {
        Ok(None)
    }
}

struct AsyncScorer<'a>(crate::ane_reward::PipelinedGrpoSession<'a>);

impl Scorer for AsyncScorer<'_> {
    fn submit(&mut self, request: ScoreRequest) -> GrpoResult<Option<Vec<f64>>> {
        // The worker thread scores text; images stay with the batch.
        self.0.begin_step_with_columns(
            request.prompts,
            request.completions,
            request.references,
            request.columns,
        )
    }

    fn flush(&mut self) -> GrpoResult<Option<Vec<f64>>> {
        self.0.flush()
    }
}

/// `[B, T]` f32 mask of the completion tokens in `labels` `[B, S]`: the
/// positions the shifted log-probs score, 1 where the next token is a
/// completion token and 0 over the prompt and padding.
fn completion_mask(labels: &Array) -> Array {
    let next = labels.index((.., 1..));
    next.ne(&Array::from_int(-100).as_dtype(next.dtype_raw()))
        .as_dtype(pmetal_runtime::dtype::F32)
}

/// GRPO Trainer.
pub struct GrpoTrainer {
    /// The objective and rollout configuration.
    pub config: GrpoConfig,
    /// Learning rate, schedule, clipping and the rest of the optimization.
    pub training_config: TrainingConfig,
    /// Optimizer steps taken.
    pub step: usize,
    epoch: usize,
    /// Rollouts sampled so far: the index each rollout's seed is drawn from.
    rollouts: u64,
    /// Generation batches trained on.
    learned: usize,
    /// Adaptive LR controller (spike/plateau/divergence detection + manual override).
    adaptive_lr: Option<crate::adaptive_lr::AdaptiveLrController>,
    /// Cached adaptive LR override value.
    adaptive_lr_override: Option<f32>,
    /// Training callbacks for metrics/dashboard integration.
    callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>>,
    /// In-memory snapshot of the best LoRA weights for rollback.
    ///
    /// LoRA parameters are typically a few MB so this is cheap to hold in memory.
    /// Populated whenever `should_snapshot_best()` returns true; consumed on rollback.
    best_lora_snapshot: Option<FlattenedModuleParam>,
    checkpoint_manager: Option<crate::CheckpointManager>,
    best_checkpoint_loss: f64,
    last_loss: Option<f64>,
    /// The optimizer step of the last periodic checkpoint.
    last_checkpoint_step: usize,
    /// Optimizer steps the LR schedule spans, set by [`Self::plan_schedule`].
    schedule_total_steps: Option<usize>,
}

impl GrpoTrainer {
    /// A trainer for `config` and `training_config`, both validated.
    pub fn new(config: GrpoConfig, training_config: TrainingConfig) -> GrpoResult<Self> {
        config.validate()?;
        training_config
            .validate()
            .map_err(|e| GrpoError::Config(e.to_string()))?;
        Ok(Self {
            config,
            training_config,
            step: 0,
            epoch: 0,
            rollouts: 0,
            learned: 0,
            adaptive_lr: None,
            adaptive_lr_override: None,
            callbacks: Vec::new(),
            best_lora_snapshot: None,
            checkpoint_manager: None,
            best_checkpoint_loss: f64::MAX,
            last_loss: None,
            last_checkpoint_step: 0,
            schedule_total_steps: None,
        })
    }

    /// Add a training callback for metrics logging or dashboard integration.
    pub fn add_callback(&mut self, callback: Box<dyn pmetal_core::TrainingCallback>) {
        self.callbacks.push(callback);
    }

    /// Save checkpoints through `manager`, every `save_steps` optimizer steps
    /// and at the end.
    pub fn set_checkpoint_manager(&mut self, manager: crate::CheckpointManager) {
        self.checkpoint_manager = Some(manager);
    }

    /// Resume at optimizer step `step`.
    pub fn set_step(&mut self, step: usize) {
        self.step = step;
        self.last_checkpoint_step = step;
    }

    /// Enable adaptive LR with control file for TUI communication.
    pub fn enable_adaptive_lr_with_control(
        &mut self,
        config: crate::adaptive_lr::AdaptiveLrConfig,
        control_file: std::path::PathBuf,
    ) {
        self.adaptive_lr = Some(
            crate::adaptive_lr::AdaptiveLrController::new(config).with_control_file(control_file),
        );
    }

    /// Get the current learning rate, respecting adaptive override.
    fn get_learning_rate(&self) -> f32 {
        if let Some(lr) = self.adaptive_lr_override {
            return lr;
        }
        self.scheduled_lr()
    }

    /// The training config's schedule at the current optimizer step, over
    /// [`Self::plan_schedule`]'s total.
    fn scheduled_lr(&self) -> f32 {
        let total = self
            .schedule_total_steps
            .or(self.training_config.max_steps)
            .unwrap_or(0);
        pmetal_core::LearningRateScheduler::for_training(&self.training_config, total)
            .get_lr(self.step) as f32
    }

    /// Size the schedule for `generation_batches` batches an epoch, each
    /// taking `num_iterations` optimizer steps (or `max_steps`), and return
    /// the total.
    pub fn plan_schedule(&mut self, generation_batches: usize) -> usize {
        let iterations = self.config.num_iterations.max(1);
        let total = pmetal_core::total_training_steps(
            &self.training_config,
            generation_batches * iterations,
            1,
            true,
        );
        let warmup = pmetal_core::warmup_steps_for(&self.training_config, total);
        info!(
            "LR schedule: {:?} over {total} optimizer steps ({} epoch(s) of {generation_batches} \
             generation batches, {iterations} update(s) each{}), {warmup} warmup steps, peak lr {:.2e}",
            self.training_config.lr_scheduler,
            self.training_config.num_epochs.max(1),
            if self.training_config.max_steps.is_some() {
                ", capped by max_steps"
            } else {
                ""
            },
            self.training_config.learning_rate,
        );
        self.schedule_total_steps = Some(total);
        if let Some(ref mut ctrl) = self.adaptive_lr {
            ctrl.set_total_steps(total);
            ctrl.set_warmup_steps(warmup);
        }
        total
    }

    /// Whether `max_steps` optimizer steps have been taken.
    fn reached_max_steps(&self) -> bool {
        self.training_config
            .max_steps
            .is_some_and(|max| self.step >= max)
    }

    /// Take a snapshot of the model's LoRA weights as the current best.
    ///
    /// Called when the adaptive LR controller indicates the EMA loss has reached a new
    /// minimum.  The snapshot is held in memory for fast rollback (LoRA params are small).
    fn snapshot_best_weights<M: pmetal_lora::TrainableModel>(&mut self, model: &M) {
        let params = model.lora_parameters();
        tracing::debug!(
            "GRPO snapshot: saved best LoRA weights at step {} ({} params, ~{:.1} MB)",
            self.step,
            params.len(),
            params.values().map(|a| a.nbytes()).sum::<usize>() as f64 / 1_048_576.0,
        );
        self.best_lora_snapshot = Some(params);
    }

    /// Restore model weights from the best in-memory snapshot.
    ///
    /// Returns `true` if weights were successfully restored.
    fn restore_best_weights<M: pmetal_lora::TrainableModel>(&mut self, model: &mut M) -> bool {
        if let Some(ref snapshot) = self.best_lora_snapshot {
            model.set_lora_parameters(snapshot);

            if let Some(ref mut ctrl) = self.adaptive_lr {
                ctrl.on_rollback_complete();
            }

            tracing::info!(
                "GRPO rollback: restored best LoRA weights at step {}",
                self.step
            );
            true
        } else {
            tracing::warn!("GRPO rollback requested but no best snapshot available");
            false
        }
    }

    /// Save a checkpoint when `save_steps` optimizer steps have passed since
    /// the last (or `force`). A batch takes `num_iterations` steps, so the
    /// step count need not land on a multiple of `save_steps`.
    fn maybe_save_checkpoint<M: TrainableModel>(
        &mut self,
        model: &M,
        loss: f64,
        force: bool,
    ) -> GrpoResult<()> {
        self.last_loss = Some(loss);
        let every = self.training_config.save_steps.unwrap_or(0);
        let due = every > 0 && self.step >= self.last_checkpoint_step + every;
        if !force && !due {
            return Ok(());
        }

        let Some(manager) = self.checkpoint_manager.as_ref() else {
            return Ok(());
        };

        let is_best = loss < self.best_checkpoint_loss;
        if is_best {
            self.best_checkpoint_loss = loss;
        }

        let params = model.lora_parameters();
        let metadata = crate::CheckpointMetadata::new(
            self.step,
            self.epoch,
            loss,
            self.get_learning_rate() as f64,
        );
        manager
            .save_checkpoint(&params, &metadata, is_best)
            .map_err(|e| GrpoError::Io(std::io::Error::other(e.to_string())))?;
        self.last_checkpoint_step = self.step;
        Ok(())
    }

    /// Feed loss to the adaptive LR controller and update the override.
    ///
    /// Returns an `AdaptiveAction` indicating how the training loop should proceed.
    fn apply_adaptive_lr_action(&mut self, loss: f64) -> crate::training_loop::AdaptiveAction {
        let scheduled = self.scheduled_lr() as f64;
        let step = self.step;
        if let Some(ref mut ctrl) = self.adaptive_lr {
            let (adjusted, event) = ctrl.step(step, loss, scheduled);
            self.adaptive_lr_override = Some(adjusted as f32);

            let action = match &event {
                crate::adaptive_lr::LrEvent::RollbackTriggered { new_lr, .. } => {
                    // Reduce the adaptive LR override to the rollback-reduced value
                    self.adaptive_lr_override = Some(*new_lr as f32);
                    crate::training_loop::AdaptiveAction::Rollback
                }
                crate::adaptive_lr::LrEvent::EarlyStop { .. } => {
                    crate::training_loop::AdaptiveAction::EarlyStop
                }
                _ => crate::training_loop::AdaptiveAction::Continue,
            };

            if !matches!(event, crate::adaptive_lr::LrEvent::Scheduled) {
                for cb in &mut self.callbacks {
                    cb.on_lr_event(&format!("{event}"));
                }
            }

            action
        } else {
            crate::training_loop::AdaptiveAction::Continue
        }
    }

    /// Check if the adaptive LR controller recommends snapshotting the current weights.
    /// Must call `ctrl.should_snapshot_best(step)` to update `best_ema_step` — without
    /// this call, `best_ema_step` is never set and snapshots never trigger.
    fn should_snapshot_best(&mut self) -> bool {
        if let Some(ref mut ctrl) = self.adaptive_lr {
            ctrl.should_snapshot_best(self.step)
        } else {
            false
        }
    }

    /// Compute per-token log probabilities for a sequence.
    ///
    /// Gathers each token's log-prob without materializing the full
    /// `[B, S, V]` log_softmax tensor (~4 GB for 128K-vocab models at typical
    /// batch sizes), in f32 whatever the logits' dtype.
    ///
    /// Returns `(per_token_logps, completion_mask)` both `[B, T]` where `T = seq_len - 1`
    /// (shifted for next-token prediction). The mask is 1.0 for valid completion
    /// tokens and 0.0 for prompt/padding tokens.
    pub fn compute_per_token_logps(
        &self,
        logits: &Array,
        labels: &Array,
        temperature: Option<f32>,
    ) -> GrpoResult<(Array, Array)> {
        Ok(crate::logprob_utils::shifted_selective_log_softmax(
            logits,
            labels,
            temperature,
        )?)
    }

    /// Compute advantages using group-relative normalization
    /// ([`group_advantages`], whitened as the config says).
    pub fn compute_advantages(&self, rewards: &[f64], num_prompts: usize) -> GrpoResult<Vec<f64>> {
        group_advantages(rewards, num_prompts, self.config.whiten_advantages)
    }

    /// Apply the overlong penalty and dynamic sampling to a scored group;
    /// `false` when the group should not be trained on.
    fn prepare_group_for_loss(&self, group: &mut CompletionGroup) -> bool {
        if let Some(penalty) = self.config.dapo_overlong_penalty {
            for (reward, stopped_by_length) in
                group.rewards.iter_mut().zip(group.stopped_by_length.iter())
            {
                if *stopped_by_length {
                    *reward += penalty;
                }
            }
        }

        if group.rewards.len() < self.config.dapo_min_group_size {
            tracing::debug!(
                "DAPO: skipping group with {} completions (< {})",
                group.rewards.len(),
                self.config.dapo_min_group_size
            );
            return false;
        }

        if self.config.dapo_dynamic_sampling {
            let correct = group
                .rewards
                .iter()
                .filter(|&&reward| reward > self.config.dapo_accuracy_reward_threshold)
                .count();
            let accuracy = correct as f64 / group.rewards.len() as f64;
            let min_acc = self.config.dapo_dynamic_sampling_min_accuracy;
            let max_acc = 1.0 - min_acc;
            if accuracy <= min_acc || accuracy >= max_acc {
                tracing::debug!(
                    "DAPO: dynamic sampling skipped group with accuracy={:.3}",
                    accuracy
                );
                return false;
            }
        }

        true
    }

    /// The GRPO loss for one batch of completions: [`grpo_loss`] under this
    /// trainer's configuration.
    pub fn compute_grpo_loss(
        &self,
        per_token_logps: &Array,
        old_per_token_logps: &Array,
        ref_per_token_logps: Option<&Array>,
        advantages: &Array,
        completion_mask: &Array,
    ) -> GrpoLoss {
        grpo_loss(
            &self.config,
            per_token_logps,
            old_per_token_logps,
            ref_per_token_logps,
            advantages,
            completion_mask,
        )
    }

    /// Prepare a training batch from completion groups.
    pub fn prepare_batch(&mut self, groups: &[CompletionGroup]) -> GrpoResult<PreparedBatch> {
        let mut all_prompts = Vec::new();
        let mut all_completions = Vec::new();
        let mut all_rewards = Vec::new();

        for group in groups {
            for completion in &group.completion_ids {
                all_prompts.push(group.prompt_ids.clone());
                all_completions.push(completion.clone());
            }
            all_rewards.extend(&group.rewards);
        }

        let advantages = self.compute_advantages(&all_rewards, groups.len())?;

        Ok((all_prompts, all_completions, advantages))
    }

    /// Train on one generation batch of completion groups.
    ///
    /// 1. Build padded input_ids / labels / completion_mask tensors.
    /// 2. With `beta > 0`, compute the reference model's log-probs once.
    /// 3. `num_iterations` times: set the scheduled learning rate through
    ///    `set_lr`, take `value_and_grad` of [`grpo_loss`], clip the gradient
    ///    to `max_grad_norm` and update.
    ///
    /// The policy that generated the batch is the policy before the first
    /// update, so its log-probs are the first update's own forward, detached:
    /// the ratio of that update is exactly 1 and no separate forward is spent
    /// on it. The later updates take their ratios against those log-probs,
    /// which is what clipping bounds.
    ///
    /// When `vlm_mode` is enabled and the groups contain `pixel_values`, the policy
    /// forward passes use `forward_with_images` to condition on visual inputs.
    pub fn train_step<M, R, O>(
        &mut self,
        policy_model: &mut M,
        ref_model: Option<&mut R>,
        groups: &[CompletionGroup],
        optimizer: &mut O,
        set_lr: &mut impl FnMut(&mut O, f32),
    ) -> GrpoResult<GrpoIterationStats>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
    {
        self.train_step_inner(policy_model, ref_model, groups, optimizer, set_lr, None)
    }

    /// [`train_step`](Self::train_step) with a teacher term blended into the
    /// loss (RLKD).
    pub(crate) fn train_step_with_teacher<M, R, O>(
        &mut self,
        policy_model: &mut M,
        ref_model: Option<&mut R>,
        groups: &[CompletionGroup],
        optimizer: &mut O,
        set_lr: &mut impl FnMut(&mut O, f32),
        teacher: &mut TeacherTerm<'_>,
    ) -> GrpoResult<GrpoIterationStats>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
    {
        self.train_step_inner(
            policy_model,
            ref_model,
            groups,
            optimizer,
            set_lr,
            Some(teacher),
        )
    }

    /// [`run`](Self::run) with a teacher term blended into the loss (RLKD).
    #[expect(
        clippy::too_many_arguments,
        reason = "the models, data, teacher and optimizer of one run"
    )]
    pub(crate) fn run_with_teacher<M, R, O, F>(
        &mut self,
        policy_model: &mut M,
        ref_model: Option<&mut R>,
        tokenizer: &pmetal_data::Tokenizer,
        prompts: &[GrpoPrompt],
        reward_fn: &CombinedReward,
        teacher: &mut TeacherTerm<'_>,
        optimizer: &mut O,
        set_optimizer_lr: F,
    ) -> GrpoResult<()>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
        F: FnMut(&mut O, f32),
    {
        self.drive(
            policy_model,
            ref_model,
            tokenizer,
            prompts,
            &mut SyncScorer(reward_fn),
            Some(teacher),
            optimizer,
            set_optimizer_lr,
        )
    }

    pub(crate) fn train_step_inner<M, R, O>(
        &mut self,
        policy_model: &mut M,
        ref_model: Option<&mut R>,
        groups: &[CompletionGroup],
        optimizer: &mut O,
        set_lr: &mut impl FnMut(&mut O, f32),
        mut teacher: Option<&mut TeacherTerm<'_>>,
    ) -> GrpoResult<GrpoIterationStats>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
    {
        let start_time = Instant::now();
        if self.config.beta > 0.0 && ref_model.is_none() {
            return Err(GrpoError::Config(format!(
                "beta is {} but no reference model was given: the KL penalty needs one \
                 (or set beta to 0)",
                self.config.beta
            )));
        }
        let (all_prompts, all_completions, advantages) = self.prepare_batch(groups)?;

        // Collect raw rewards for logging before they're normalized into advantages
        let raw_rewards: Vec<f64> = groups
            .iter()
            .flat_map(|g| g.rewards.iter().copied())
            .collect();

        let n_completions = all_completions.len();
        if n_completions == 0 {
            return Err(GrpoError::Config("no completions to train on".into()));
        }
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
            ids.extend(vec![0; pad_len]);
            labels.extend(vec![-100; pad_len]);

            input_ids_vec.extend(ids.iter().map(|&id| id as i32));
            labels_vec.extend(labels);
        }

        let input_ids = Array::from_slice(&input_ids_vec, &[n_completions as i32, max_len as i32]);
        let labels = Array::from_slice(&labels_vec, &[n_completions as i32, max_len as i32]);
        let mask = completion_mask(&labels);

        // Collect pixel values from groups for VLM forward passes. Each
        // group's images are replicated once per completion so the pixel
        // tensor's batch matches `input_ids`' `[n_completions, seq_len]`.
        let pixel_values: Option<Array> = if self.config.vlm_mode {
            let all_images: Vec<Array> = groups
                .iter()
                .filter_map(|g| {
                    g.pixel_values
                        .as_ref()
                        .map(|imgs| (imgs, g.completion_ids.len()))
                })
                .flat_map(|(imgs, n_completions)| {
                    std::iter::repeat_n(imgs.iter().cloned(), n_completions).flatten()
                })
                .collect();
            stack_pixel_values(&all_images)
        } else {
            None
        };

        // Temperature for log-prob computation (None = 1.0, no scaling)
        let temperature = if (self.config.temperature - 1.0).abs() > 1e-8 {
            Some(self.config.temperature as f32)
        } else {
            None
        };

        // The reference model's log-probs, computed once for the batch. It is
        // text-only (the original weights), so it takes the plain forward.
        let ref_per_token_logps = match ref_model {
            Some(ref_m) if self.config.beta > 0.0 => {
                let ref_logits = ref_m.forward(input_ids.clone())?;
                let (ref_logps, _) =
                    self.compute_per_token_logps(&ref_logits, &labels, temperature)?;
                ref_logps.eval();
                Some(ref_logps)
            }
            _ => None,
        };

        // The teacher's logits, once for the batch and outside the gradient:
        // a constant of every update.
        let teacher_logits = match teacher.as_deref_mut() {
            Some(term) => {
                let logits = term.model.forward_teacher(&input_ids)?;
                logits.eval();
                Some(logits)
            }
            None => None,
        };

        let pixel_values_ref = pixel_values.as_ref();
        let iterations = self.config.num_iterations.max(1);
        let max_grad_norm = self.training_config.max_grad_norm as f32;
        let mut losses = Vec::with_capacity(iterations);
        let mut policy_losses = Vec::with_capacity(iterations);
        let mut kls = Vec::with_capacity(iterations);
        let mut clip_fractions = Vec::with_capacity(iterations);
        let mut distill_losses = Vec::with_capacity(iterations);
        let mut rl_losses = Vec::with_capacity(iterations);
        let mut alphas = Vec::with_capacity(iterations);
        // The generating policy's log-probs: the first update's, detached.
        let mut old_per_token_logps: Option<Array> = None;

        for _ in 0..iterations {
            let stash: std::cell::RefCell<Option<(GrpoLoss, Array, Option<Array>)>> =
                std::cell::RefCell::new(None);
            let old_ref = old_per_token_logps.as_ref();
            let total_steps = self.schedule_total_steps.unwrap_or(0);
            let blend = teacher
                .as_deref()
                .map(|term| (term.alpha_at(self.step, total_steps), term));
            if let Some((alpha, _)) = &blend {
                alphas.push(*alpha);
            }
            let teacher_logits_ref = teacher_logits.as_ref();
            let loss_fn =
                |model: &mut M,
                 (input_ids, labels, adv_array, mask): (&Array, &Array, &Array, &Array)|
                 -> std::result::Result<Array, Exception> {
                    let logits = model
                        .forward_with_images(input_ids, None, pixel_values_ref)
                        .map_err(|e| Exception::custom(e.to_string()))?;
                    let (per_token_logps, _) = self
                        .compute_per_token_logps(&logits, labels, temperature)
                        .map_err(|e| Exception::custom(e.to_string()))?;
                    let detached = per_token_logps.stop_gradient();
                    let parts = self.compute_grpo_loss(
                        &per_token_logps,
                        old_ref.unwrap_or(&detached),
                        ref_per_token_logps.as_ref(),
                        adv_array,
                        mask,
                    );
                    // (1 − α) · RL + α · distillation, over the completion tokens.
                    let (total, distill) = match (&blend, teacher_logits_ref) {
                        (Some((alpha, term)), Some(teacher_logits)) => {
                            let distill = term.distillation(teacher_logits, &logits, mask)?;
                            let total = parts
                                .total
                                .multiply(&Array::from_f32(1.0 - alpha))
                                .add(&distill.multiply(&Array::from_f32(*alpha)));
                            (total, Some(distill))
                        }
                        _ => (parts.total.clone(), None),
                    };
                    *stash.borrow_mut() = Some((parts, detached, distill));
                    Ok(total)
                };

            set_lr(optimizer, self.get_learning_rate());
            let (loss, mut grads) = {
                let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
                loss_and_grad_fn(policy_model, (&input_ids, &labels, &adv_array, &mask))?
            };
            let (parts, detached, distill) = stash
                .into_inner()
                .expect("value_and_grad calls the loss closure once");
            // Materialize before the update: these are values of the policy
            // that is about to change.
            rl_losses.push({
                parts.total.eval();
                parts.total.item_f32()
            });
            transforms::eval([
                &loss,
                &detached,
                &parts.policy,
                &parts.kl,
                &parts.clip_fraction,
            ])?;
            if let Some(distill) = &distill {
                distill.eval();
                distill_losses.push(distill.item_f32());
            }
            if old_per_token_logps.is_none() {
                old_per_token_logps = Some(detached);
            }
            if max_grad_norm > 0.0 {
                pmetal_runtime::training::clip_grad_norm_map(&mut grads, max_grad_norm);
            }
            optimizer.update(policy_model, grads)?;
            let params = policy_model.lora_parameters();
            transforms::eval(params.values())?;
            losses.push(crate::step_check::check_step(
                self.step + 1,
                loss.item_f32(),
            )?);
            policy_losses.push(parts.policy.item_f32());
            kls.push(parts.kl.item_f32());
            clip_fractions.push(parts.clip_fraction.item_f32());
            self.step += 1;
        }

        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
        let mean_reward = raw_rewards.iter().sum::<f64>() / raw_rewards.len().max(1) as f64;
        let mean_adv = advantages.iter().sum::<f64>() / advantages.len().max(1) as f64;
        let tokens: usize = all_completions.iter().map(Vec::len).sum();

        Ok(GrpoIterationStats {
            step: self.step,
            loss: mean(&losses),
            kl: mean(&kls),
            policy_loss: mean(&policy_losses),
            clip_fraction: mean(&clip_fractions),
            iterations,
            reward: mean_reward as f32,
            advantage: mean_adv as f32,
            rl_loss: mean(&rl_losses),
            distill_loss: mean(&distill_losses),
            alpha: mean(&alphas),
            completion_length: tokens as f32 / n_completions as f32,
            completions_per_second: n_completions as f32 / start_time.elapsed().as_secs_f32(),
        })
    }

    /// Generate the group of completions for a prompt: the prompt prefilled
    /// once (from the prefix cache where a repeat allows) and every
    /// completion decoded together, one batched forward per step.
    ///
    /// The completions are sampled from the policy itself, which GRPO's
    /// importance ratios assume, so no draft model is involved. A configured
    /// seed gives each rollout its own stream.
    pub fn generate_completions<M>(
        &mut self,
        model: &mut M,
        prompt_tokens: &[u32],
        tokenizer: &pmetal_data::Tokenizer,
    ) -> GrpoResult<pmetal_models::rl_generation::BatchedGenerationOutput>
    where
        M: TrainableModel,
    {
        let stop_tokens = if self.config.stop_tokens.is_empty() {
            tokenizer.eos_token_id().into_iter().collect()
        } else {
            self.config.stop_tokens.clone()
        };
        let seed = self
            .config
            .seed
            .map(|seed| pmetal_runtime::decode::draw_bits(seed, self.rollouts));
        self.rollouts += 1;
        crate::rollout::sample_completions(
            model,
            prompt_tokens,
            &crate::rollout::RolloutConfig {
                num_generations: self.config.num_generations,
                max_new_tokens: self.config.max_completion_length,
                temperature: self.config.temperature as f32,
                top_p: self.config.top_p as f32,
                top_k: self.config.top_k,
                min_p: self.config.min_p as f32,
                stop_tokens,
                seed,
                max_len: self.config.max_prompt_length + self.config.max_completion_length,
                kv_cache_bits: self.config.kv_cache_bits,
            },
        )
        .map_err(|e| GrpoError::Generation(e.to_string()))
    }

    /// Run the full GRPO training loop: every prompt in turn, for the
    /// configured epochs, sample, score, learn.
    #[expect(
        clippy::too_many_arguments,
        reason = "public API: the models, data and optimizer"
    )]
    pub fn run<M, R, O, F>(
        &mut self,
        policy_model: &mut M,
        ref_model: Option<&mut R>,
        tokenizer: &pmetal_data::Tokenizer,
        prompts: &[GrpoPrompt],
        reward_fn: &CombinedReward,
        optimizer: &mut O,
        set_optimizer_lr: F,
    ) -> GrpoResult<()>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
        F: FnMut(&mut O, f32),
    {
        info!("Starting GRPO training loop...");
        self.drive(
            policy_model,
            ref_model,
            tokenizer,
            prompts,
            &mut SyncScorer(reward_fn),
            None,
            optimizer,
            set_optimizer_lr,
        )
    }

    /// Run GRPO training with pipelined (asynchronous) reward scoring.
    ///
    /// Identical to [`run`](Self::run) except that `reward_fn` is scored on a
    /// background thread, so the scoring of one batch overlaps the generation
    /// of the next:
    ///
    /// ```text
    /// Step N:   Generate → Submit score (bg thread)
    /// Step N+1: Generate → Collect N, Submit N+1 → Train on N
    /// ```
    ///
    /// The batch trained on was generated one step earlier (see
    /// [`GrpoConfig::async_rewards`]), and the thread sees prompts, texts and
    /// reference answers but not images.
    ///
    /// # Errors
    ///
    /// Same as `run`, plus [`GrpoError::Reward`] if the background scorer
    /// thread terminates unexpectedly.
    #[expect(
        clippy::too_many_arguments,
        reason = "public API: the models, data and optimizer"
    )]
    pub fn run_async<M, R, O, F>(
        &mut self,
        policy_model: &mut M,
        ref_model: Option<&mut R>,
        tokenizer: &pmetal_data::Tokenizer,
        prompts: &[GrpoPrompt],
        reward_fn: Box<dyn RewardFunction>,
        optimizer: &mut O,
        set_optimizer_lr: F,
    ) -> GrpoResult<()>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
        F: FnMut(&mut O, f32),
    {
        use crate::ane_reward::{AsyncRewardModel, PipelinedGrpoSession};

        info!("Starting GRPO training loop (pipelined reward scoring via AsyncRewardModel)...");
        let async_reward = AsyncRewardModel::new(reward_fn);
        let mut scorer = AsyncScorer(PipelinedGrpoSession::new(&async_reward));
        self.drive(
            policy_model,
            ref_model,
            tokenizer,
            prompts,
            &mut scorer,
            None,
            optimizer,
            set_optimizer_lr,
        )
    }

    /// The loop both entry points share. The scorer decides when a batch's
    /// rewards are ready: at once, or a step later.
    #[expect(
        clippy::too_many_arguments,
        reason = "the models, data, scorer and optimizer of one run"
    )]
    fn drive<M, R, O, F>(
        &mut self,
        policy_model: &mut M,
        mut ref_model: Option<&mut R>,
        tokenizer: &pmetal_data::Tokenizer,
        prompts: &[GrpoPrompt],
        scorer: &mut dyn Scorer,
        mut teacher: Option<&mut TeacherTerm<'_>>,
        optimizer: &mut O,
        mut set_optimizer_lr: F,
    ) -> GrpoResult<()>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
        F: FnMut(&mut O, f32),
    {
        if prompts.is_empty() {
            return Err(GrpoError::Config("no prompts to train on".into()));
        }
        let n_epochs = self.training_config.num_epochs;
        let total_steps = self.plan_schedule(prompts.len());

        for cb in &mut self.callbacks {
            cb.on_train_start();
        }

        let mut flow = Flow::Continue;
        let outcome = (|| -> GrpoResult<()> {
            let mut pending: VecDeque<PendingBatch> = VecDeque::new();
            'epochs: for epoch in 0..n_epochs {
                self.epoch = epoch;
                info!("Epoch {}/{}", epoch + 1, n_epochs);

                for prompt in prompts {
                    let started = Instant::now();
                    let generated =
                        self.generate_completions(policy_model, &prompt.input_ids, tokenizer)?;

                    let prompt_text = tokenizer
                        .decode(&prompt.input_ids)
                        .map_err(|e| GrpoError::Tokenizer(e.to_string()))?;
                    let n = generated.token_ids.len();
                    let mut completions_text = Vec::with_capacity(n);
                    for ids in &generated.token_ids {
                        completions_text.push(
                            tokenizer
                                .decode(&ids[prompt.input_ids.len()..])
                                .map_err(|e| GrpoError::Tokenizer(e.to_string()))?,
                        );
                    }

                    // Images load once per prompt; a failure falls back to text.
                    let sample_images: Option<Vec<Array>> = if self.config.vlm_mode {
                        match &prompt.images {
                            Some(paths) if !paths.is_empty() => {
                                match load_images(paths, self.config.max_image_size) {
                                    Ok(imgs) => Some(imgs),
                                    Err(e) => {
                                        tracing::warn!("VLM: failed to load images: {e}");
                                        None
                                    }
                                }
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let images = sample_images.as_ref().map(|imgs| vec![imgs.clone(); n]);

                    let ready = scorer.submit(ScoreRequest {
                        prompts: vec![prompt_text; n],
                        completions: completions_text,
                        references: vec![prompt.reference.clone(); n],
                        columns: vec![prompt.columns.clone(); n],
                        images,
                    })?;
                    pending.push_back(PendingBatch {
                        prompt_ids: prompt.input_ids.clone(),
                        completions: generated
                            .token_ids
                            .iter()
                            .zip(&generated.stopped_by_length)
                            .map(|(ids, &by_length)| {
                                (ids[prompt.input_ids.len()..].to_vec(), by_length)
                            })
                            .collect(),
                        pixel_values: sample_images,
                        started,
                    });

                    if let Some(rewards) = ready {
                        let batch = pending
                            .pop_front()
                            .expect("a batch is pending for every scored request");
                        flow = self.learn(
                            batch,
                            rewards,
                            policy_model,
                            ref_model.as_deref_mut(),
                            optimizer,
                            &mut set_optimizer_lr,
                            teacher.as_deref_mut(),
                            total_steps,
                        )?;
                        if flow != Flow::Continue {
                            break 'epochs;
                        }
                    }
                    if self.reached_max_steps() {
                        info!("Reached max_steps={} optimizer steps, stopping", self.step);
                        // Let the scorer finish what is in flight, untrained.
                        let _ = scorer.flush();
                        return Ok(());
                    }
                }
            }

            // The batch still being scored when the data ran out.
            while flow == Flow::Continue {
                let Some(rewards) = scorer.flush()? else {
                    break;
                };
                let batch = pending
                    .pop_front()
                    .expect("a batch is pending for every scored request");
                flow = self.learn(
                    batch,
                    rewards,
                    policy_model,
                    ref_model.as_deref_mut(),
                    optimizer,
                    &mut set_optimizer_lr,
                    teacher.as_deref_mut(),
                    total_steps,
                )?;
                if self.reached_max_steps() {
                    break;
                }
            }
            if flow != Flow::Continue {
                // Drain the worker so it is not left blocked on a response.
                let _ = scorer.flush();
            }
            Ok(())
        })();

        for cb in &mut self.callbacks {
            cb.on_train_end();
        }
        outcome?;
        match flow {
            Flow::Cancelled => Err(GrpoError::Cancelled),
            Flow::EarlyStop => Ok(()),
            Flow::Continue => {
                self.maybe_save_checkpoint(policy_model, self.last_loss.unwrap_or(0.0), true)
            }
        }
    }

    /// Learn from a scored batch: the optimizer steps, then the adaptive-LR
    /// action, a checkpoint, the log line and the callbacks.
    #[expect(
        clippy::too_many_arguments,
        reason = "the batch, the models and the optimizer of one step"
    )]
    fn learn<M, R, O>(
        &mut self,
        batch: PendingBatch,
        rewards: Vec<f64>,
        policy_model: &mut M,
        ref_model: Option<&mut R>,
        optimizer: &mut O,
        set_optimizer_lr: &mut impl FnMut(&mut O, f32),
        teacher: Option<&mut TeacherTerm<'_>>,
        total_steps: usize,
    ) -> GrpoResult<Flow>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
    {
        let mut group = CompletionGroup::new(batch.prompt_ids, self.config.num_generations);
        for ((ids, by_length), reward) in batch.completions.into_iter().zip(rewards) {
            group.add_completion(ids, reward, by_length);
        }
        group.pixel_values = batch.pixel_values;
        if !self.prepare_group_for_loss(&mut group) {
            return Ok(Flow::Continue);
        }

        let stats = self.train_step_inner(
            policy_model,
            ref_model,
            std::slice::from_ref(&group),
            optimizer,
            set_optimizer_lr,
            teacher,
        )?;
        self.learned += 1;

        let action = self.apply_adaptive_lr_action(stats.loss as f64);

        // Snapshot best weights when loss reaches a new minimum
        if action == crate::training_loop::AdaptiveAction::Continue && self.should_snapshot_best() {
            self.snapshot_best_weights(policy_model);
        }

        // Rollback: restore best weights and reduce LR
        if action == crate::training_loop::AdaptiveAction::Rollback {
            self.restore_best_weights(policy_model);
            let rollback_lr = self
                .adaptive_lr_override
                .unwrap_or(self.training_config.learning_rate as f32);
            set_optimizer_lr(optimizer, rollback_lr);
            tracing::info!(
                "GRPO rollback at step {}: new lr={:.2e}",
                self.step,
                rollback_lr
            );
        }

        // Early stop: restore best weights and finish
        if action == crate::training_loop::AdaptiveAction::EarlyStop {
            self.restore_best_weights(policy_model);
            tracing::info!("Early stopping GRPO training — adaptive LR exhausted rollbacks.");
            self.maybe_save_checkpoint(policy_model, stats.loss as f64, true)?;
            return Ok(Flow::EarlyStop);
        }

        self.maybe_save_checkpoint(policy_model, stats.loss as f64, false)?;

        let step_ms = batch.started.elapsed().as_secs_f64() * 1000.0;
        let lr = self.get_learning_rate();
        if (self.learned - 1) % 10 == 0 {
            info!(
                "Step {}: loss={:.4}, kl={:.4}, reward={:.4}, clip_fraction={:.3} over {} update(s), \
                 lr={:.2e}, completion_len={:.1}",
                stats.step,
                stats.loss,
                stats.kl,
                stats.reward,
                stats.clip_fraction,
                stats.iterations,
                lr,
                stats.completion_length,
            );
        }

        if !self.callbacks.is_empty() {
            let metrics = pmetal_core::StepMetrics {
                step: self.step,
                epoch: self.epoch,
                total_epochs: self.training_config.num_epochs,
                total_steps,
                loss: stats.loss as f64,
                lr: lr as f64,
                tok_sec: 0.0, // GRPO doesn't track tokens the same way
                total_ms: step_ms,
                tokens: 0,
                ..Default::default()
            };
            for cb in &mut self.callbacks {
                cb.on_step_end_with_metrics(&metrics);
            }
            if self.callbacks.iter().any(|cb| cb.should_stop()) {
                return Ok(Flow::Cancelled);
            }
        }
        Ok(Flow::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_runtime::compat::Array;
    use serial_test::serial;

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /// Build a GrpoTrainer with a minimal config, overriding fields as needed.
    fn make_trainer(config: GrpoConfig) -> GrpoTrainer {
        GrpoTrainer::new(config, pmetal_core::TrainingConfig::default()).unwrap()
    }

    // ---------------------------------------------------------------------------
    // 1. compute_advantages — whitened
    // ---------------------------------------------------------------------------

    /// Group 1: rewards=[1,2,3,4], mean=2.5, std=sqrt(5/3)≈1.291
    /// Group 2: rewards=[10,10,10,10], mean=10.0, std≈0 → all advantages ≈ 0
    #[test]
    fn test_compute_advantages_whitened() {
        let trainer = make_trainer(GrpoConfig {
            whiten_advantages: true,
            ..GrpoConfig::default()
        });

        let rewards = [1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0];
        let advantages = trainer.compute_advantages(&rewards, 2).unwrap();

        assert_eq!(advantages.len(), 8);

        // Group 1 — values should be non-zero (distinct rewards)
        let g1 = &advantages[0..4];
        let g1_max_abs = g1.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            g1_max_abs > 0.1,
            "group 1 advantages should be non-zero; got {g1:?}"
        );

        // The whitened advantages for group 1 should sum to ~0
        let g1_sum: f64 = g1.iter().sum();
        assert!(
            g1_sum.abs() < 1e-10,
            "whitened advantages should sum to 0; got {g1_sum}"
        );

        // Verify ordering preserved: reward 4 should yield the highest advantage
        assert!(
            g1[3] > g1[2] && g1[2] > g1[1] && g1[1] > g1[0],
            "advantages should be monotone with rewards; got {g1:?}"
        );

        // Group 2 — all same reward → variance ≈ 0 → clamped by 1e-4 floor
        // The advantages will be (10 - 10) / std ≈ 0 / 1e-4 = 0
        let g2 = &advantages[4..8];
        for (i, &adv) in g2.iter().enumerate() {
            assert!(
                adv.abs() < 1e-9,
                "group 2 advantage[{i}] should be ~0; got {adv}"
            );
        }
    }

    // ---------------------------------------------------------------------------
    // 2. compute_advantages — unwhitened
    // ---------------------------------------------------------------------------

    #[test]
    fn test_compute_advantages_unwhitened() {
        let trainer = make_trainer(GrpoConfig {
            whiten_advantages: false,
            ..GrpoConfig::default()
        });

        let rewards = [1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0];
        let advantages = trainer.compute_advantages(&rewards, 2).unwrap();

        assert_eq!(advantages.len(), 8);

        // Group 1: mean = 2.5 → advantages = [-1.5, -0.5, 0.5, 1.5]
        let g1 = &advantages[0..4];
        let expected_g1 = [-1.5_f64, -0.5, 0.5, 1.5];
        for (i, (&got, &exp)) in g1.iter().zip(expected_g1.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "g1[{i}]: expected {exp}, got {got}"
            );
        }

        // Group 2: mean = 10.0 → all advantages = 0.0
        let g2 = &advantages[4..8];
        for (i, &adv) in g2.iter().enumerate() {
            assert!(adv.abs() < 1e-12, "g2[{i}]: expected 0.0, got {adv}");
        }
    }

    // ---------------------------------------------------------------------------
    // 3. compute_advantages — error cases
    // ---------------------------------------------------------------------------

    #[test]
    fn test_compute_advantages_errors() {
        let trainer = make_trainer(GrpoConfig::default());

        // num_prompts = 0 → Config error
        let err = trainer.compute_advantages(&[1.0, 2.0], 0).unwrap_err();
        assert!(
            matches!(err, GrpoError::Config(_)),
            "expected Config error for num_prompts=0, got {err:?}"
        );

        // rewards.len() not divisible by num_prompts
        let err = trainer.compute_advantages(&[1.0, 2.0, 3.0], 2).unwrap_err();
        assert!(
            matches!(err, GrpoError::Config(_)),
            "expected Config error for indivisible len, got {err:?}"
        );

        // Empty rewards with num_prompts=1 → group size 0 → Config error
        let err = trainer.compute_advantages(&[], 1).unwrap_err();
        assert!(
            matches!(err, GrpoError::Config(_)),
            "expected Config error for empty rewards, got {err:?}"
        );
    }

    // ---------------------------------------------------------------------------
    // 4. compute_grpo_loss
    // ---------------------------------------------------------------------------

    fn arr(v: &[f32], shape: &[i32]) -> Array {
        Array::from_slice(v, shape)
    }

    fn scalar(a: &Array) -> f32 {
        a.eval();
        pmetal_runtime::check_unobserved_error().expect("bridge op failed");
        a.item_f32()
    }

    fn values(a: &Array) -> Vec<f32> {
        a.eval();
        pmetal_runtime::check_unobserved_error().expect("bridge op failed");
        a.as_slice::<f32>().to_vec()
    }

    /// With the current and old log-probs equal the ratio is 1 everywhere,
    /// nothing is clipped, and the loss is −A averaged as the loss type says.
    #[test]
    #[serial]
    fn on_policy_the_ratio_is_one_and_nothing_clips() {
        let trainer = make_trainer(GrpoConfig {
            beta: 0.0,
            ..GrpoConfig::default()
        });
        let logps = arr(&[-0.5, -0.5, -0.5, -0.5, -1.0, -1.0, -1.0, -1.0], &[2, 4]);
        let parts = trainer.compute_grpo_loss(
            &logps,
            &logps,
            None,
            &arr(&[1.0, -1.0], &[2]),
            &arr(&[1.0; 8], &[2, 4]),
        );
        assert!(scalar(&parts.total).abs() < 1e-6);
        assert_eq!(scalar(&parts.kl), 0.0);
        assert_eq!(scalar(&parts.clip_fraction), 0.0);
        assert!(values(&parts.ratio).iter().all(|&r| r == 1.0));
    }

    /// A ratio of e⁵ on a positive advantage is clipped to 1 + ε.
    #[test]
    #[serial]
    fn a_large_token_ratio_is_clipped_at_one_plus_epsilon() {
        let trainer = make_trainer(GrpoConfig {
            beta: 0.0,
            epsilon_low: 0.2,
            epsilon_high: 0.2,
            ..GrpoConfig::default()
        });
        let parts = trainer.compute_grpo_loss(
            &arr(&[-0.1; 8], &[2, 4]),
            &arr(&[-5.1; 8], &[2, 4]),
            None,
            &arr(&[1.0, 1.0], &[2]),
            &arr(&[1.0; 8], &[2, 4]),
        );
        assert!((scalar(&parts.total) + 1.2).abs() < 1e-4);
        assert_eq!(scalar(&parts.clip_fraction), 1.0);
    }

    /// The reported KL is k3 = exp(ref − π) − (ref − π) − 1 ≥ 0, 0 when they
    /// agree; the penalty is that estimator's (K3) or ½(π − ref)² (K2, the
    /// default), whose gradient is the reverse KL's, times β.
    #[test]
    #[serial]
    fn the_kl_estimate_is_non_negative_and_zero_at_the_reference() {
        let policy = arr(&[-0.5; 8], &[2, 4]);
        let reference = arr(&[-0.1; 8], &[2, 4]);
        let adv = arr(&[0.0, 0.0], &[2]);
        let mask = arr(&[1.0; 8], &[2, 4]);
        let k3 = 0.4f32.exp() - 0.4 - 1.0;
        let k2 = 0.5 * 0.4f32 * 0.4;
        for (estimator, penalty) in [(KlEstimator::K3, k3), (KlEstimator::K2, k2)] {
            let trainer = make_trainer(GrpoConfig {
                beta: 0.1,
                kl_estimator: estimator,
                ..GrpoConfig::default()
            });
            let parts = trainer.compute_grpo_loss(&policy, &policy, Some(&reference), &adv, &mask);
            assert!((scalar(&parts.kl) - k3).abs() < 1e-6, "{estimator:?}");
            // With zero advantages the loss is β times the estimator.
            assert!(
                (scalar(&parts.total) - 0.1 * penalty).abs() < 1e-6,
                "{estimator:?}"
            );
            let parts = trainer.compute_grpo_loss(&policy, &policy, Some(&policy), &adv, &mask);
            assert!(scalar(&parts.kl).abs() < 1e-6);
            assert!(scalar(&parts.total).abs() < 1e-6);
        }
    }

    /// Differentiated as a loss, k2 follows the reverse KL's gradient and k3
    /// the forward KL's (Shah et al., arXiv 2512.21852). With π a softmax over
    /// three tokens and samples drawn from it, the expected gradient of each
    /// estimator is checked against the closed-form KL gradients in f64.
    #[test]
    #[serial]
    fn k2_is_the_reverse_kl_gradient_and_k3_the_forward_kl_gradient() {
        let logits = [0.3f64, -0.4, 0.9];
        let ref_logits = [-0.2f64, 0.5, 0.1];
        let log_softmax = |z: &[f64; 3]| {
            let lse = z.iter().map(|x| x.exp()).sum::<f64>().ln();
            z.map(|x| x - lse)
        };
        let (lp, lr) = (log_softmax(&logits), log_softmax(&ref_logits));
        let (p, q) = (lp.map(f64::exp), lr.map(f64::exp));
        // d/dz_j of KL(p‖q) and KL(q‖p), p = softmax(z).
        let kl_rev: f64 = (0..3).map(|i| p[i] * (lp[i] - lr[i])).sum();
        let grad_reverse: Vec<f64> = (0..3).map(|j| p[j] * (lp[j] - lr[j] - kl_rev)).collect();
        let grad_forward: Vec<f64> = (0..3).map(|j| p[j] - q[j]).collect();

        // Expected loss over x ~ p of the estimator at x: one "sequence" of
        // one token per vocabulary entry, weighted by p. Gradient of the
        // weighted sum with the weights held fixed is the on-policy gradient.
        for (estimator, want) in [
            (KlEstimator::K2, &grad_reverse),
            (KlEstimator::K3, &grad_forward),
        ] {
            let trainer = make_trainer(GrpoConfig {
                beta: 1.0,
                kl_estimator: estimator,
                loss_type: GrpoLossType::Dapo,
                ..GrpoConfig::default()
            });
            let z = Array::from_slice(&logits.map(|x| x as f32), &[3]);
            let ref_lp = Array::from_slice(&lr.map(|x| x as f32), &[1, 3]);
            // The mask carries p, which sums to 1: the token-level mean is
            // the expectation over the sampling distribution.
            let weights = Array::from_slice(&p.map(|x| x as f32), &[1, 3]);
            let (_, grads) = nn::value_and_grad_explicit(
                |params: &[Array]| {
                    let logps = nn::log_softmax(&params[0].reshape(&[1, 3]), -1);
                    trainer
                        .compute_grpo_loss(
                            &logps,
                            &logps.stop_gradient(),
                            Some(&ref_lp),
                            &Array::from_slice(&[0.0f32], &[1]),
                            &weights,
                        )
                        .total
                },
                &[z],
                &[],
            )
            .unwrap();
            let got = values(&grads[0]);
            for j in 0..3 {
                assert!(
                    (got[j] as f64 - want[j]).abs() < 1e-5,
                    "{estimator:?} d/dz{j}: {} vs {}",
                    got[j],
                    want[j]
                );
            }
        }
    }

    /// Two completions of 2 and 4 tokens with advantages +1 and −1, on
    /// policy, so each token's loss is −A: −1 −1 for the first, +1 four times
    /// for the second.
    #[test]
    #[serial]
    fn each_loss_type_aggregates_as_its_paper_does() {
        let logps = arr(&[-0.5; 8], &[2, 4]);
        let adv = arr(&[1.0, -1.0], &[2]);
        let mask = arr(&[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[2, 4]);
        let cases = [
            // GRPO: each sequence over its own length, (−1 + 1) / 2.
            (GrpoLossType::Grpo, 0.0),
            // DAPO: every token alike, (−2 + 4) / 6.
            (GrpoLossType::Dapo, 2.0 / 6.0),
            // Dr. GRPO: over 2 sequences × max_completion_length 4.
            (GrpoLossType::DrGrpo, 2.0 / 8.0),
        ];
        for (loss_type, want) in cases {
            let trainer = make_trainer(GrpoConfig {
                beta: 0.0,
                loss_type,
                max_completion_length: 4,
                ..GrpoConfig::default()
            });
            let parts = trainer.compute_grpo_loss(&logps, &logps, None, &adv, &mask);
            let got = scalar(&parts.total);
            assert!(
                (got - want).abs() < 1e-6,
                "{loss_type:?}: got {got}, want {want}"
            );
        }
    }

    /// GSPO's ratio is the length-normalized sequence likelihood ratio,
    /// exp((log π(o|q) − log π_old(o|q)) / |o|), over completion tokens only.
    #[test]
    #[serial]
    fn the_gspo_ratio_is_the_length_normalized_sequence_ratio() {
        let trainer = make_trainer(GrpoConfig {
            beta: 0.0,
            ..GrpoConfig::default().for_gspo()
        });
        let old = [-1.0f32, -2.0, -0.5, -0.5, -1.5, -1.0, -2.0, -0.7];
        // Per-token log ratios: [0.1, 0.3, 5, 5] (the 5s are masked out) and
        // [0.5, −0.1, 0.2, 0.0].
        let deltas = [0.1f32, 0.3, 5.0, 5.0, 0.5, -0.1, 0.2, 0.0];
        let current: Vec<f32> = old.iter().zip(deltas).map(|(o, d)| o + d).collect();
        let mask = [1.0f32, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let parts = trainer.compute_grpo_loss(
            &arr(&current, &[2, 4]),
            &arr(&old, &[2, 4]),
            None,
            &arr(&[1.0, -1.0], &[2]),
            &arr(&mask, &[2, 4]),
        );

        // By hand, from the sequences' log-likelihoods.
        let seq_ratio = |cur: &[f32], old: &[f32], mask: &[f32]| {
            let n: f32 = mask.iter().sum();
            let lp: f32 = cur.iter().zip(mask).map(|(x, m)| x * m).sum();
            let lp_old: f32 = old.iter().zip(mask).map(|(x, m)| x * m).sum();
            ((lp - lp_old) / n).exp()
        };
        let s0 = seq_ratio(&current[..4], &old[..4], &mask[..4]);
        let s1 = seq_ratio(&current[4..], &old[4..], &mask[4..]);
        assert!((s0 - 0.2f32.exp()).abs() < 1e-6 && (s1 - 0.15f32.exp()).abs() < 1e-6);
        assert_eq!(parts.ratio.shape(), &[2, 1]);
        let ratio = values(&parts.ratio);
        assert!(
            (ratio[0] - s0).abs() < 1e-5 && (ratio[1] - s1).abs() < 1e-5,
            "{ratio:?}"
        );

        // Clip range [1 − 3e-4, 1 + 4e-4]. s₀ > 1 + ε with A > 0 is clipped to
        // 1.0004; s₁ > 1 + ε with A < 0 keeps its (worse) unclipped value.
        // Sequences average as in GRPO: (−1.0004 + s₁) / 2.
        let want = (-(1.0 + 4e-4) + s1) / 2.0;
        assert!((scalar(&parts.total) - want).abs() < 1e-5);
        // The clipped sequence's 2 tokens of the 6.
        assert!((scalar(&parts.clip_fraction) - 2.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn loss_presets_set_what_their_papers_describe() {
        let base = GrpoConfig::default();
        assert_eq!(base.loss_type, GrpoLossType::Dapo);
        let dr = base.clone().with_loss_preset("dr_grpo").unwrap();
        assert_eq!(dr.loss_type, GrpoLossType::DrGrpo);
        assert!(!dr.whiten_advantages);
        let gspo = base.clone().with_loss_preset("gspo").unwrap();
        assert_eq!(gspo.importance_sampling, ImportanceSampling::Sequence);
        assert_eq!(gspo.loss_type, GrpoLossType::Grpo);
        assert_eq!((gspo.epsilon_low, gspo.epsilon_high), (3e-4, 4e-4));
        let grpo = base.clone().with_loss_preset("GRPO").unwrap();
        assert_eq!(grpo.loss_type, GrpoLossType::Grpo);
        assert!(base.clone().with_loss_preset("bnpo").is_err());
        let dapo = base.for_dapo();
        assert_eq!((dapo.epsilon_low, dapo.epsilon_high), (0.2, 0.28));
        assert_eq!(dapo.dapo_overlong_penalty, Some(-1.0));
        assert_eq!(GrpoConfig::default().dapo_overlong_penalty, None);
    }

    // ---------------------------------------------------------------------------
    // 7. num_iterations on a model
    // ---------------------------------------------------------------------------

    fn small_policy() -> pmetal_lora::AdaptedModel {
        use pmetal_models::architectures::llama::LlamaConfig;
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
        let base =
            pmetal_models::DynamicModel::from_config(&serde_json::to_string(&config).unwrap())
                .unwrap();
        let lora = pmetal_core::LoraConfig {
            r: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            ..Default::default()
        };
        pmetal_lora::AdaptedModel::attach(base, lora).unwrap()
    }

    fn one_group() -> CompletionGroup {
        let mut group = CompletionGroup::new(vec![1, 2, 3], 4);
        group.add_completion(vec![4, 5, 6], 1.0, false);
        group.add_completion(vec![7, 8], 0.0, false);
        group.add_completion(vec![9, 10, 11, 12], 1.0, false);
        group.add_completion(vec![13, 14, 15], 0.0, false);
        group
    }

    type Weights = FlattenedModuleParam;

    /// Train one generation batch with `config` from LoRA weights `init`.
    fn train_batch(
        model: &mut pmetal_lora::AdaptedModel,
        init: &Weights,
        config: GrpoConfig,
    ) -> (GrpoIterationStats, Weights) {
        model.set_lora_parameters(init);
        let mut trainer = make_trainer(GrpoConfig {
            beta: 0.0,
            ..config
        });
        trainer.training_config.learning_rate = 0.05;
        trainer.training_config.warmup_steps = 0;
        trainer.training_config.lr_scheduler = pmetal_core::LrSchedulerType::Constant;
        trainer.training_config.weight_decay = 0.0;
        let mut optimizer = crate::TrainOptimizer::from_config(&trainer.training_config);
        let stats = trainer
            .train_step(
                model,
                None::<&mut pmetal_models::DynamicModel>,
                &[one_group()],
                &mut optimizer,
                &mut |opt: &mut crate::TrainOptimizer, lr| opt.set_lr(lr),
            )
            .unwrap();
        assert_eq!(trainer.step, stats.iterations);
        (stats, model.lora_parameters())
    }

    fn max_abs_diff(a: &Weights, b: &Weights) -> f32 {
        a.iter()
            .map(|(k, x)| scalar(&x.subtract(&b[k]).abs().max(None)))
            .fold(0.0, f32::max)
    }

    #[test]
    #[serial]
    fn later_iterations_on_a_batch_are_clipped_and_clipping_changes_the_update() {
        let mut model = small_policy();
        let init = model.lora_parameters();
        let tight = GrpoConfig {
            epsilon_low: 0.01,
            epsilon_high: 0.01,
            ..GrpoConfig::default()
        };

        // One update per batch is on policy: the ratio is 1, nothing clips.
        let (stats, _) = train_batch(&mut model, &init, tight.clone());
        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.clip_fraction, 0.0);

        // A second update on the same batch sees the moved policy.
        let two = GrpoConfig {
            num_iterations: 2,
            ..tight
        };
        let (stats, clipped) = train_batch(&mut model, &init, two.clone());
        assert_eq!(stats.iterations, 2);
        assert!(stats.loss.is_finite());
        assert!(
            stats.clip_fraction > 0.0,
            "no token clipped on the second update"
        );

        // The same two updates with a clip range nothing reaches.
        let unclipped_config = GrpoConfig {
            epsilon_low: 1e3,
            epsilon_high: 1e3,
            ..two
        };
        let (stats, unclipped) = train_batch(&mut model, &init, unclipped_config);
        assert_eq!(stats.clip_fraction, 0.0);
        let diff = max_abs_diff(&clipped, &unclipped);
        assert!(
            diff > 1e-6,
            "clipping did not change the update (diff {diff})"
        );

        // GSPO clips whole sequences once the policy moves.
        let gspo = GrpoConfig {
            num_iterations: 2,
            ..GrpoConfig::default().for_gspo()
        };
        let (stats, _) = train_batch(&mut model, &init, gspo);
        assert!(stats.clip_fraction > 0.0, "GSPO never clipped");
    }

    // ---------------------------------------------------------------------------
    // 8. The loop learns
    // ---------------------------------------------------------------------------

    /// A word-level tokenizer over `w0`..`w62` and `</s>` (id 63), the
    /// vocabulary of [`small_policy`].
    fn word_tokenizer() -> pmetal_data::Tokenizer {
        let mut vocab = serde_json::Map::new();
        for i in 0..63 {
            vocab.insert(format!("w{i}"), serde_json::json!(i));
        }
        vocab.insert("</s>".into(), serde_json::json!(63));
        let tokenizer = serde_json::json!({
            "version": "1.0",
            "pre_tokenizer": {"type": "Whitespace"},
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "w0"}
        });
        pmetal_data::Tokenizer::from_bytes(tokenizer.to_string().as_bytes()).unwrap()
    }

    /// Reward 1 for a completion that is one of the words `w0`..`w7`.
    struct InTargetSet;
    impl RewardFunction for InTargetSet {
        fn compute(
            &self,
            _: &[String],
            completions: &[String],
            _: Option<&[Vec<Array>]>,
        ) -> GrpoResult<Vec<f64>> {
            Ok(completions
                .iter()
                .map(|c| {
                    let hit = c
                        .split_whitespace()
                        .next()
                        .and_then(|w| w.strip_prefix('w'))
                        .and_then(|n| n.parse::<u32>().ok())
                        .is_some_and(|n| n < 8);
                    f64::from(hit)
                })
                .collect())
        }
        fn name(&self) -> &str {
            "in_target_set"
        }
    }

    /// The policy's probability that its first completion token is one of
    /// `w0`..`w7` after `prompt`: the expected reward of [`InTargetSet`] for
    /// one-token completions, exactly.
    fn expected_reward(model: &mut pmetal_lora::AdaptedModel, prompt: &[u32]) -> f64 {
        let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let input = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let logits = model.forward(&input, None).unwrap();
        let last = logits
            .slice(&[0, ids.len() as i32 - 1, 0], &[1, ids.len() as i32, 64])
            .as_dtype(pmetal_runtime::dtype::F32);
        let row = values(&last);
        let max = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let z: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
        row[..8]
            .iter()
            .map(|&x| (x as f64 - max).exp())
            .sum::<f64>()
            / z
    }

    /// One `run` over a single prompt, `epochs` times, rewarded for sampling
    /// a word of the target set. Returns the expected reward before and after.
    fn learn_the_target_set(
        config: GrpoConfig,
        epochs: usize,
        reward: Box<dyn RewardFunction>,
    ) -> (f64, f64) {
        pmetal_runtime::compat::random::seed(11);
        let mut model = small_policy();
        let prompt = vec![1u32, 2, 3];
        let before = expected_reward(&mut model, &prompt);

        let training = pmetal_core::TrainingConfig {
            learning_rate: 0.05,
            num_epochs: epochs,
            warmup_steps: 0,
            lr_scheduler: pmetal_core::LrSchedulerType::Constant,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut trainer = GrpoTrainer::new(config, training).unwrap();
        let mut optimizer = crate::TrainOptimizer::from_config(&trainer.training_config);
        let mut combined = CombinedReward::new();
        combined.functions.push((reward, 1.0));
        trainer
            .run(
                &mut model,
                None::<&mut pmetal_models::DynamicModel>,
                &word_tokenizer(),
                &[GrpoPrompt::new(prompt.clone())],
                &combined,
                &mut optimizer,
                |opt: &mut crate::TrainOptimizer, lr| opt.set_lr(lr),
            )
            .unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();
        (before, expected_reward(&mut model, &prompt))
    }

    fn one_token_config() -> GrpoConfig {
        GrpoConfig {
            num_generations: 16,
            max_completion_length: 1,
            max_prompt_length: 8,
            seed: Some(5),
            ..GrpoConfig::default()
        }
    }

    /// The point of the whole loop: sampling, scoring, advantages and the
    /// update move the policy toward what the reward pays for. Each variant of
    /// the objective raises the expected reward of a toy task, from 1/8 to well
    /// above it; a constant reward (zero advantages) moves nothing.
    #[test]
    #[serial]
    fn every_loss_variant_raises_the_expected_reward() {
        for preset in ["dapo", "grpo", "dr_grpo", "gspo"] {
            let config = GrpoConfig {
                num_iterations: if preset == "gspo" { 2 } else { 1 },
                ..one_token_config().with_loss_preset(preset).unwrap()
            };
            let (before, after) = learn_the_target_set(config, 40, Box::new(InTargetSet));
            println!("{preset}: expected reward {before:.3} -> {after:.3}");
            assert!(
                after > before + 0.15,
                "{preset}: expected reward {before:.3} -> {after:.3}"
            );
        }
    }

    /// A batch takes `num_iterations` optimizer steps, so the step count
    /// steps over multiples of `save_steps`; a checkpoint is due once that many
    /// steps have passed since the last one.
    #[test]
    #[serial]
    fn periodic_checkpoints_follow_the_steps_taken_not_their_multiples() {
        let dir = tempfile::tempdir().unwrap();
        let model = small_policy();
        let training = pmetal_core::TrainingConfig {
            save_steps: Some(5),
            ..Default::default()
        };
        let mut trainer = GrpoTrainer::new(GrpoConfig::default(), training).unwrap();
        trainer.set_checkpoint_manager(
            crate::CheckpointManager::new(dir.path())
                .unwrap()
                .with_max_checkpoints(10),
        );
        // Three steps a batch: 3, 6, 9, 12, 15, 18.
        for batch in 1..=6 {
            trainer.step = 3 * batch;
            trainer.maybe_save_checkpoint(&model, 1.0, false).unwrap();
        }
        let saved: Vec<usize> = trainer
            .checkpoint_manager
            .as_ref()
            .unwrap()
            .list_checkpoints()
            .unwrap()
            .into_iter()
            .map(|(step, _)| step)
            .collect();
        // Every 5 steps once passed: at 6 (≥5), 12 (≥11), 18 (≥17). The
        // old `step % 5 == 0` test fired only at 15.
        assert_eq!(saved, vec![6, 12, 18]);
    }

    #[test]
    #[serial]
    fn a_constant_reward_trains_nothing() {
        struct Constant;
        impl RewardFunction for Constant {
            fn compute(
                &self,
                _: &[String],
                completions: &[String],
                _: Option<&[Vec<Array>]>,
            ) -> GrpoResult<Vec<f64>> {
                Ok(vec![0.5; completions.len()])
            }
            fn name(&self) -> &str {
                "constant"
            }
        }
        let (before, after) = learn_the_target_set(one_token_config(), 5, Box::new(Constant));
        assert!((after - before).abs() < 1e-6, "{before} -> {after}");
    }
}
