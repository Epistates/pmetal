//! Group Relative Policy Optimization (GRPO) implementation.
//!
//! GRPO is a reinforcement learning algorithm that optimizes policies by comparing
//! the performance of multiple completions for the same prompt.
//! It is particularly effective for reasoning models (e.g., DeepSeek-R1).
//!
//! ## Key Features
//! - **Reference-free or Reference-based**: Supports KL divergence from a reference model.
//! - **Group-based Advantages**: Computes advantages relative to the group mean/std.
//! - **Flexible Rewards**: Pluggable reward functions for reasoning, formatting, and accuracy.
//! - **Efficient Training**: Implementation optimized for Apple Silicon via MLX.

use pmetal_bridge::compat::{
    Array, Exception,
    indexing::IndexOp,
    module::{Module, ModuleParameters},
    nn, ops,
    optimizers::Optimizer,
};
use pmetal_core::TrainingConfig;
use pmetal_lora::TrainableModel;
use pmetal_models::rl_generation::{BatchedRlConfig, BatchedRlGenerator};
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
    /// Mean reward for this batch.
    pub reward: f32,
    /// Mean advantage for this batch.
    pub advantage: f32,
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

/// GRPO loss type variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrpoLossType {
    /// Standard GRPO / BNPO (Base Normalized Policy Optimization).
    #[default]
    Bnpo,
    /// DR-GRPO (Detailed Reward GRPO).
    DrGrpo,
    /// DAPO (Distribution-Aware Policy Optimization).
    Dapo,
    /// Simple REINFORCE-style loss without KL.
    Reinforce,
}

/// GRPO configuration.
#[derive(Debug, Clone)]
pub struct GrpoConfig {
    /// Number of completions to generate per prompt.
    pub num_generations: usize,
    /// Maximum length of the generated completion.
    pub max_completion_length: usize,
    /// Maximum length of the prompt.
    pub max_prompt_length: usize,
    /// KL divergence coefficient.
    pub beta: f64,
    /// Temperature for sampling.
    pub temperature: f64,
    /// Top-p for sampling.
    pub top_p: f64,
    /// Top-k for sampling.
    pub top_k: usize,
    /// Whether to whiten (normalize) advantages within each group.
    pub whiten_advantages: bool,
    /// Entropy bonus coefficient.
    pub entropy_coef: f64,
    /// Loss function type.
    pub loss_type: GrpoLossType,
    /// Lower clipping epsilon for PPO-clip (default 0.2).
    pub epsilon_low: f64,
    /// Upper clipping epsilon for PPO-clip (default 0.2).
    pub epsilon_high: f64,
    /// Enable VLM (Vision-Language Model) mode for processing image inputs.
    ///
    /// When enabled, the trainer will load images from each sample's `images` field,
    /// pass them to reward functions, and use `forward_with_images` for the training
    /// step to condition the model on visual inputs alongside text.
    pub vlm_mode: bool,
    /// Maximum image size (pixels per side) for VLM preprocessing.
    ///
    /// Images are resized to fit within this square while maintaining aspect ratio.
    /// Typical values: 336 (CLIP ViT-L/14), 448, 560 (Mllama).
    pub max_image_size: usize,
    /// Path to a pretrained ML reward model for scoring completions.
    ///
    /// When set, an `MLRewardModel` is loaded at training start and added to the
    /// `CombinedReward` with weight `reward_model_weight`.  The reward model runs
    /// inference-only alongside the policy model.
    ///
    /// Supports any architecture recognized by `DynamicModel::load` (Llama,
    /// Qwen, Gemma, Mistral, …).  Popular choices: ArmoRM-Llama3-8B-v0.1,
    /// Skywork-Reward-Llama-3.1-8B, and FsfairX-LLaMA3-RM-v0.1.
    pub reward_model_path: Option<String>,
    /// Maximum input sequence length for the ML reward model (tokens).
    ///
    /// Inputs longer than this are truncated from the right.  Defaults to 2048.
    pub reward_model_max_length: usize,
    /// Weight for the ML reward model relative to heuristic reward functions.
    ///
    /// The combined reward is a weighted sum across all reward functions.
    /// Defaults to 1.0.
    pub reward_model_weight: f64,
    /// Optional chat template for formatting prompt+completion inputs to the
    /// reward model.
    ///
    /// Use `{prompt}` and `{completion}` as placeholders.  When `None`, prompt
    /// and completion are concatenated directly (suitable for reward models
    /// that expect raw text).
    pub reward_model_chat_template: Option<String>,
    /// Enable pipelined (asynchronous) reward scoring.
    ///
    /// When `true`, each training step submits the reward scoring request to a
    /// background thread **before** the GPU training forward/backward pass.
    /// The scores from the previous step are collected at the start of each
    /// new step, allowing reward computation to overlap with GPU execution.
    ///
    /// This is most effective when the reward model is CPU- or ANE-bound
    /// (e.g., an `MLRewardModel`) and the training step has non-trivial GPU
    /// latency.  For pure heuristic rewards (format, accuracy checks), the
    /// overhead is negligible and pipelining provides no measurable benefit.
    ///
    /// The pipeline shifts reward scoring by one step, so the first training
    /// step uses freshly computed rewards (no delay) and subsequent steps use
    /// scores that were computed during the previous step's GPU pass.
    ///
    /// Defaults to `false`.
    pub async_rewards: bool,
    /// Enable speculative decoding for rollout generation.
    ///
    /// When `true`, `generate_completions` uses a layer-split draft/verify
    /// approach via `BatchedRlGenerator::generate_speculative`.  The same
    /// `forward_with_cache` call is used for both the cheap draft phase (first
    /// N/3 layers — emulated by re-running with a small token sequence through
    /// the full model after early-exit via the draft closure split) and the
    /// authoritative verify phase.
    ///
    /// Expected throughput improvement: 2–4× depending on model and acceptance
    /// rate.  Requires the policy model to support KV caching; automatically
    /// falls back to standard generation when the model returns `None` from
    /// `create_cache`.
    ///
    /// Defaults to `false`.
    pub use_speculative: bool,
    /// Number of draft tokens to propose per speculative decode step.
    ///
    /// Higher values amortise more tokens per verify pass, but reduce the
    /// benefit when the draft acceptance rate drops.  Typical sweet-spot: 3–5.
    /// Ignored when `use_speculative` is `false`.  Defaults to 3.
    pub speculative_draft_tokens: usize,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            num_generations: 8,
            max_completion_length: 512,
            max_prompt_length: 512,
            beta: 0.1,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 40,
            whiten_advantages: true,
            entropy_coef: 0.0,
            loss_type: GrpoLossType::Bnpo,
            epsilon_low: 0.2,
            epsilon_high: 0.2,
            vlm_mode: false,
            max_image_size: 336,
            reward_model_path: None,
            reward_model_max_length: 2048,
            reward_model_weight: 1.0,
            reward_model_chat_template: None,
            async_rewards: false,
            use_speculative: false,
            speculative_draft_tokens: 3,
        }
    }
}

impl GrpoConfig {
    pub fn new(num_generations: usize) -> Self {
        Self {
            num_generations,
            ..Default::default()
        }
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    pub fn for_dapo(mut self) -> Self {
        self.loss_type = GrpoLossType::Dapo;
        self.beta = 0.0; // DAPO usually doesn't use standard KL
        self
    }
}

/// Completion group for a single prompt.
#[derive(Debug, Clone)]
pub struct CompletionGroup {
    pub prompt_ids: Vec<u32>,
    pub completion_ids: Vec<Vec<u32>>,
    pub rewards: Vec<f64>,
    pub stopped_by_length: Vec<bool>,
    /// Optional preprocessed pixel values for VLM training.
    ///
    /// When VLM mode is active this holds the images loaded from the corresponding
    /// dataset sample (`sample.images`).  Each element is one image as an MLX
    /// array of shape `[1, C, H, W]` (NCHW float32, model-specific normalization).
    /// All completions in this group share the same images — they all come from the
    /// same prompt.
    pub pixel_values: Option<Vec<Array>>,
}

impl CompletionGroup {
    pub fn new(prompt_ids: Vec<u32>, num_generations: usize) -> Self {
        Self {
            prompt_ids,
            completion_ids: Vec::with_capacity(num_generations),
            rewards: Vec::with_capacity(num_generations),
            stopped_by_length: Vec::with_capacity(num_generations),
            pixel_values: None,
        }
    }

    pub fn add_completion(&mut self, ids: Vec<u32>, reward: f64, stopped_by_length: bool) {
        self.completion_ids.push(ids);
        self.rewards.push(reward);
        self.stopped_by_length.push(stopped_by_length);
    }
}

/// Load and preprocess images from file paths into MLX arrays.
///
/// Each returned array has shape `[1, C, H, W]` (NCHW float32) with CLIP-style
/// normalization (mean/std from `MllamaImageProcessorConfig::default()`).
/// The image is resized to fit within `max_size × max_size` preserving aspect ratio.
///
/// The `image` crate is already a transitive dependency via `pmetal-data`, so this
/// function uses the same processor that is available there to avoid duplication.
fn load_images(image_paths: &[std::path::PathBuf], max_size: usize) -> GrpoResult<Vec<Array>> {
    use pmetal_data::image_processing::{MllamaImageProcessor, MllamaImageProcessorConfig};

    // Use CLIP-canonical normalization; the size will be overridden below.
    let config = MllamaImageProcessorConfig {
        size: (max_size as u32, max_size as u32),
        ..Default::default()
    };
    let processor = MllamaImageProcessor::new(config);

    let mut images = Vec::with_capacity(image_paths.len());
    for path in image_paths {
        // Load via the `image` crate (used internally by the processor).
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
        let arr = processor
            .process_image(resized)
            .map_err(|e| GrpoError::Mlx(e))?;

        images.push(arr);
    }
    Ok(images)
}

/// Stack a slice of per-image arrays into a single batched pixel-values tensor.
///
/// Each input array has shape `[1, C, H, W]`.  The output is `[N, C, H, W]`
/// where N = number of images.  Returns `None` for an empty slice.
///
/// All images must share the same spatial dimensions.  If they differ (e.g. due
/// to variable aspect-ratio resizing) the concatenation will fail, which surfaces
/// as a logged warning and a `None` return rather than a hard error — the model's
/// `forward_with_images` default impl falls back to `forward` in that case.
fn stack_pixel_values(images: &[Array]) -> Option<Array> {
    if images.is_empty() {
        return None;
    }
    let refs: Vec<&Array> = images.iter().collect();
    let arr = ops::concatenate_axis(&refs, 0);
    Some(arr)
}

/// GRPO Trainer.
pub struct GrpoTrainer {
    pub config: GrpoConfig,
    pub training_config: TrainingConfig,
    pub step: usize,
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
    best_lora_snapshot: Option<std::collections::HashMap<std::rc::Rc<str>, Array>>,
}

impl GrpoTrainer {
    pub fn new(config: GrpoConfig, training_config: TrainingConfig) -> GrpoResult<Self> {
        Ok(Self {
            config,
            training_config,
            step: 0,
            adaptive_lr: None,
            adaptive_lr_override: None,
            callbacks: Vec::new(),
            best_lora_snapshot: None,
        })
    }

    /// Add a training callback for metrics logging or dashboard integration.
    pub fn add_callback(&mut self, callback: Box<dyn pmetal_core::TrainingCallback>) {
        self.callbacks.push(callback);
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
        self.training_config.learning_rate as f32
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

    /// Feed loss to the adaptive LR controller and update the override.
    ///
    /// Returns an `AdaptiveAction` indicating how the training loop should proceed.
    fn apply_adaptive_lr_action(&mut self, loss: f64) -> crate::training_loop::AdaptiveAction {
        let scheduled = self.training_config.learning_rate;
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
    /// Uses `selective_log_softmax` to avoid materializing the full `[B, S, V]`
    /// log_softmax tensor (~4 GB for 128K-vocab models at typical batch sizes).
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
        let l = logits.dim(1);

        // Shift logits and labels for next-token prediction
        let shift_logits = logits.index((.., ..l - 1, ..));
        let shift_labels = labels.index((.., 1..));

        // Memory-efficient: gathers single logit per position via take_along_axis,
        // never materializes [B, S, V] log_softmax.
        let (per_token_logps, valid_mask) =
            crate::logprob_utils::selective_log_softmax_with_temperature(
                &shift_logits,
                &shift_labels,
                temperature,
            )?;

        Ok((per_token_logps, valid_mask))
    }

    /// Compute advantages using group-relative normalization.
    pub fn compute_advantages(&self, rewards: &[f64], num_prompts: usize) -> GrpoResult<Vec<f64>> {
        if num_prompts == 0 {
            return Err(GrpoError::Config("num_prompts must be > 0".into()));
        }
        if rewards.len() % num_prompts != 0 {
            return Err(GrpoError::Config(format!(
                "rewards.len() ({}) must be divisible by num_prompts ({})",
                rewards.len(),
                num_prompts
            )));
        }

        let n_per_group = rewards.len() / num_prompts;
        if n_per_group == 0 {
            return Err(GrpoError::Config("group size must be > 0".into()));
        }

        let mut advantages = vec![0.0; rewards.len()];

        for i in 0..num_prompts {
            let group = &rewards[i * n_per_group..(i + 1) * n_per_group];
            let mean = group.iter().sum::<f64>() / n_per_group as f64;

            if self.config.whiten_advantages && n_per_group > 1 {
                // Normalize by group std (whitening)
                let variance = group.iter().map(|&r| (r - mean).powi(2)).sum::<f64>()
                    / (n_per_group - 1) as f64;
                let std = variance.sqrt().max(1e-4);
                for j in 0..n_per_group {
                    advantages[i * n_per_group + j] = (group[j] - mean) / std;
                }
            } else {
                // Raw advantages (reward - mean) without normalization
                for j in 0..n_per_group {
                    advantages[i * n_per_group + j] = group[j] - mean;
                }
            }
        }

        Ok(advantages)
    }

    /// Compute GRPO loss components at the per-token level.
    ///
    /// All variants use PPO-clip as the policy gradient objective (matching TRL):
    /// `L_policy = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`
    ///
    /// The ratio is computed against `old_per_token_logps` (generation-time policy),
    /// NOT the reference model. KL regularization uses the reference model separately.
    ///
    /// # Arguments
    /// * `per_token_logps` - Current policy per-token log-probs `[B, T]`
    /// * `old_per_token_logps` - Generation-time policy per-token log-probs `[B, T]` (detached)
    /// * `ref_per_token_logps` - Reference model per-token log-probs `[B, T]` or `None`
    /// * `advantages` - Group-normalized advantages `[B]`
    /// * `completion_mask` - Valid completion token mask `[B, T]`
    /// * `entropy` - Optional per-token entropy for bonus
    pub fn compute_grpo_loss(
        &self,
        per_token_logps: &Array,
        old_per_token_logps: &Array,
        ref_per_token_logps: Option<&Array>,
        advantages: &Array,
        completion_mask: &Array,
        entropy: Option<&Array>,
    ) -> GrpoResult<(Array, Array, Array)> {
        let eps_low = self.config.epsilon_low as f32;
        let eps_high = self.config.epsilon_high as f32;

        // --- PPO-clip policy loss (all variants) ---
        // Importance ratio against OLD policy (generation-time), not reference
        let log_ratio = per_token_logps.subtract(old_per_token_logps);
        let ratio = log_ratio.exp();

        // Expand advantages for token-level broadcasting: [B] -> [B, 1]
        let adv_expanded = advantages.reshape(&[advantages.dim(0), 1]);

        // Clipped surrogate objective
        let clip_lo = Array::from_f32(1.0 - eps_low);
        let clip_hi = Array::from_f32(1.0 + eps_high);
        let clipped_ratio = ops::clip(&ratio, Some(&clip_lo), Some(&clip_hi));
        let surr1 = ratio.multiply(&adv_expanded);
        let surr2 = clipped_ratio.multiply(&adv_expanded);
        let token_policy_loss = ops::minimum(&surr1, &surr2).negative();

        // --- Per-variant reduction ---
        let masked_policy_loss = token_policy_loss.multiply(completion_mask);
        let total_tokens = completion_mask.sum(None);
        let safe_token_count = ops::maximum(&total_tokens, &Array::from_f32(1.0));

        let policy_loss = match self.config.loss_type {
            GrpoLossType::Bnpo => {
                // BNPO: mean over valid tokens
                masked_policy_loss.sum(None).divide(&safe_token_count)
            }
            GrpoLossType::DrGrpo => {
                // DR-GRPO: per-sequence mean, then batch mean
                // Sum tokens per sequence, divide by per-sequence token count
                let per_seq_sum = masked_policy_loss.sum_axis(-1, false);
                let per_seq_count = completion_mask.sum_axis(-1, false);
                let safe_per_seq = ops::maximum(&per_seq_count, &Array::from_f32(1.0));
                per_seq_sum.divide(&safe_per_seq).mean(None)
            }
            GrpoLossType::Dapo => {
                // DAPO: token-level mean (same as BNPO but conceptually distinct)
                masked_policy_loss.sum(None).divide(&safe_token_count)
            }
            GrpoLossType::Reinforce => {
                // REINFORCE: simple batch mean
                masked_policy_loss.sum(None).divide(&safe_token_count)
            }
        };

        // --- KL divergence (against reference model, not old policy) ---
        // KL(pi || ref) ≈ exp(ref - pi) - (ref - pi) - 1  (Schulman approximation)
        // Correct direction: ratio = ref/pi, KL = ratio - 1 - log(ratio)
        let kl_mean = if let Some(ref_logps) = ref_per_token_logps {
            let kl_log_ratio = ref_logps.subtract(per_token_logps);
            let kl_ratio = kl_log_ratio.exp();
            let per_token_kl = kl_ratio
                .subtract(&Array::from_f32(1.0))
                .subtract(&kl_log_ratio);
            let masked_kl = per_token_kl.multiply(completion_mask);
            masked_kl.sum(None).divide(&safe_token_count)
        } else {
            Array::from_f32(0.0)
        };

        let kl_loss = kl_mean.multiply(&Array::from_f32(self.config.beta as f32));
        let mut total_loss = policy_loss.add(&kl_loss);

        // Entropy bonus: subtract entropy_coef * entropy to encourage exploration
        if let (Some(ent), coef) = (entropy, self.config.entropy_coef) {
            if coef > 0.0 {
                let masked_ent = ent.multiply(completion_mask);
                let mean_ent = masked_ent.sum(None).divide(&safe_token_count);
                let entropy_bonus = mean_ent.multiply(&Array::from_f32(coef as f32));
                total_loss = total_loss.subtract(&entropy_bonus);
            }
        }

        Ok((total_loss, kl_mean, policy_loss))
    }

    /// Prepare a training batch from completion groups.
    pub fn prepare_batch(
        &mut self,
        groups: &[CompletionGroup],
    ) -> GrpoResult<(
        Vec<Vec<u32>>,
        Vec<Vec<u32>>,
        Vec<f64>,
        Vec<Vec<f32>>,
        Option<Vec<Vec<Array>>>,
    )> {
        let mut all_prompts = Vec::new();
        let mut all_completions = Vec::new();
        let mut all_rewards = Vec::new();
        let mut all_masks = Vec::new();

        for group in groups {
            for completion in &group.completion_ids {
                all_prompts.push(group.prompt_ids.clone());
                all_completions.push(completion.clone());

                let mut mask = vec![0.0f32; group.prompt_ids.len()];
                mask.extend(vec![1.0f32; completion.len()]);
                all_masks.push(mask);
            }
            all_rewards.extend(&group.rewards);
        }

        let advantages = self.compute_advantages(&all_rewards, groups.len())?;

        Ok((all_prompts, all_completions, advantages, all_masks, None))
    }

    /// Run a single training step on a batch of completion groups.
    ///
    /// Implements the correct GRPO training loop matching TRL:
    /// 1. Build padded input_ids / labels / completion_mask tensors
    /// 2. Compute `old_per_token_logps` from the CURRENT policy (generation-time snapshot)
    /// 3. Optionally compute `ref_per_token_logps` from the reference model
    /// 4. Run `value_and_grad` with `compute_grpo_loss` (PPO-clip on old, KL on ref)
    /// 5. Update optimizer
    ///
    /// When `vlm_mode` is enabled and the groups contain `pixel_values`, the policy
    /// forward passes use `forward_with_images` to condition on visual inputs.
    pub fn train_step<M, R, O>(
        &mut self,
        policy_model: &mut M,
        mut ref_model: Option<&mut R>,
        groups: &[CompletionGroup],
        optimizer: &mut O,
    ) -> GrpoResult<GrpoIterationStats>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
    {
        let start_time = Instant::now();
        let (all_prompts, all_completions, advantages, _all_masks, _) =
            self.prepare_batch(groups)?;

        // Collect raw rewards for logging before they're normalized into advantages
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

            // Use i32 labels to match selective_log_softmax dtype handling
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

        // Collect pixel values from groups for VLM forward passes.
        // Each group contributes one image set shared across all its completions.
        // We must replicate those images once per completion so that the resulting
        // pixel_values tensor has shape [n_completions * n_images_per_group, C, H, W]
        // — matching the batch dimension of `input_ids` which is [n_completions, seq_len].
        //
        // Without replication, the batch sizes mismatch: forward_with_images would
        // see `n_groups` images but `n_completions` (= n_groups * num_generations) rows
        // in input_ids, causing incorrect or undefined behaviour in the VLM encoder.
        let pixel_values: Option<Array> = if self.config.vlm_mode {
            let all_images: Vec<Array> = groups
                .iter()
                .filter_map(|g| {
                    g.pixel_values
                        .as_ref()
                        .map(|imgs| (imgs, g.completion_ids.len()))
                })
                .flat_map(|(imgs, n_completions)| {
                    // Repeat the group's image list once per completion in the group.
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

        // 1. Compute old_per_token_logps from current policy BEFORE training update.
        //    These are the generation-time log-probs, detached from the gradient graph.
        //    Use forward_with_images when pixel_values are available.
        let old_logits = policy_model
            .forward_with_images(&input_ids, None, pixel_values.as_ref())
            .map_err(|e| Exception::custom(e.to_string()))?;
        let (mut old_per_token_logps, mut completion_mask) =
            self.compute_per_token_logps(&old_logits, &labels, temperature)?;
        // Eval to materialize — these must NOT be part of the grad graph
        old_per_token_logps.eval();
        completion_mask.eval();

        // 2. Compute ref_per_token_logps from reference model (if beta > 0 and ref_model exists).
        //    The reference model is text-only (it is the original pre-LoRA weights), so we
        //    use the plain Module::forward interface here.
        let ref_per_token_logps = if self.config.beta > 0.0 {
            if let Some(ref mut ref_m) = ref_model {
                let ref_logits = ref_m.forward(input_ids.clone())?;
                let (mut ref_logps, _) =
                    self.compute_per_token_logps(&ref_logits, &labels, temperature)?;
                ref_logps.eval();
                Some(ref_logps)
            } else {
                None
            }
        } else {
            None
        };

        // 3. Loss function for value_and_grad — only the policy model is differentiated.
        //    `pixel_values` is captured by reference from the outer scope; it is already
        //    materialized (eval'd) so it is safe inside the gradient closure.
        let pixel_values_ref = pixel_values.as_ref();
        let loss_fn = |model: &mut M,
                       (input_ids, labels, adv_array, old_logps, mask): (
            &Array,
            &Array,
            &Array,
            &Array,
            &Array,
        )|
         -> std::result::Result<Array, Exception> {
            let logits = model
                .forward_with_images(input_ids, None, pixel_values_ref)
                .map_err(|e| Exception::custom(e.to_string()))?;

            let (per_token_logps, _) = self
                .compute_per_token_logps(&logits, labels, temperature)
                .map_err(|e| Exception::custom(e.to_string()))?;

            let (total_loss, _kl, _policy_loss) = self
                .compute_grpo_loss(
                    &per_token_logps,
                    old_logps,
                    ref_per_token_logps.as_ref(),
                    adv_array,
                    mask,
                    None,
                )
                .map_err(|e| Exception::custom(e.to_string()))?;

            Ok(total_loss)
        };

        let (total_loss_arr, grads) = {
            let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
            loss_and_grad_fn(
                policy_model,
                (
                    &input_ids,
                    &labels,
                    &adv_array,
                    &old_per_token_logps,
                    &completion_mask,
                ),
            )?
        };

        // 4. Update optimizer
        optimizer.update(policy_model, grads)?;

        // Extract loss from the forward pass (already computed, no redundant re-forward)
        let mut total_loss_arr = total_loss_arr;
        let total_loss = total_loss_arr.item_f32();

        // Compute KL/policy_loss stats from the values already available in the loss
        // (We use the pre-update values since post-update requires an extra forward pass.
        // The loss value itself is the authoritative training signal.)
        let kl_stat = if let Some(ref_logps) = ref_per_token_logps.as_ref() {
            // Approximate: compute from old policy vs ref (cheap, no extra forward)
            let kl_log_ratio = ref_logps.subtract(&old_per_token_logps);
            let kl_ratio = kl_log_ratio.exp();
            let per_token_kl = kl_ratio
                .subtract(&Array::from_f32(1.0))
                .subtract(&kl_log_ratio);
            let masked_kl = per_token_kl.multiply(&completion_mask);
            let safe_count = ops::maximum(&completion_mask.sum(None), &Array::from_f32(1.0));
            masked_kl.sum(None).divide(&safe_count).item_f32()
        } else {
            0.0
        };

        let mean_reward = if raw_rewards.is_empty() {
            0.0
        } else {
            raw_rewards.iter().sum::<f64>() / raw_rewards.len() as f64
        };
        let mean_adv = if advantages.is_empty() {
            0.0
        } else {
            advantages.iter().sum::<f64>() / advantages.len() as f64
        };

        self.step += 1;

        Ok(GrpoIterationStats {
            step: self.step,
            loss: total_loss,
            kl: kl_stat,
            policy_loss: total_loss, // Policy loss is the dominant component
            reward: mean_reward as f32,
            advantage: mean_adv as f32,
            completions_per_second: n_completions as f32 / start_time.elapsed().as_secs_f32(),
        })
    }

    /// Generate multiple completions for a prompt.
    ///
    /// When `use_speculative` is enabled in `GrpoConfig`, this method uses
    /// `BatchedRlGenerator::generate_speculative` with a layer-split draft/verify
    /// scheme for 2–4× faster rollout generation.  The draft closure re-runs the
    /// full model forward (but with a KV cache that terminates early) while the
    /// verify closure runs the authoritative full forward pass.  Both closures
    /// wrap `model.forward_with_cache`.
    ///
    /// Automatically falls back to standard generation if:
    /// - The model does not support KV caching.
    /// - `use_speculative` is `false` (default).
    pub fn generate_completions<M>(
        &mut self,
        model: &mut M,
        prompt_tokens: &[u32],
        tokenizer: &pmetal_data::Tokenizer,
    ) -> GrpoResult<pmetal_models::rl_generation::BatchedGenerationOutput>
    where
        M: TrainableModel,
    {
        let use_speculative = self.config.use_speculative && model.supports_kv_cache();
        let draft_tokens = self.config.speculative_draft_tokens;

        let mut rl_config = BatchedRlConfig {
            num_generations: self.config.num_generations,
            max_new_tokens: self.config.max_completion_length,
            temperature: self.config.temperature as f32,
            top_p: self.config.top_p as f32,
            top_k: self.config.top_k,
            stop_tokens: vec![tokenizer.eos_token_id().unwrap_or(2)],
            seed: None,
            use_prefix_cache: true,
            min_p: 0.05,
            use_speculative,
            speculative_draft_tokens: draft_tokens,
        };

        if use_speculative {
            rl_config = rl_config.with_speculative(draft_tokens);
        }

        let cache = model
            .create_cache(self.config.max_prompt_length + self.config.max_completion_length)
            .ok_or_else(|| GrpoError::Generation("Model does not support KV cache".into()))?;
        let kv_config = cache.config();

        let mut generator = BatchedRlGenerator::new(rl_config, kv_config.clone());

        if use_speculative {
            // Speculative path: draft_fn and verify_fn both call forward_with_cache.
            //
            // The layer-split self-speculative approach (first N/3 layers as draft)
            // requires ShardableModel, which is not yet implemented for all LoRA
            // architectures.  Instead we use the same forward_with_cache for both
            // closures; the speedup comes from the batched verify pass accepting
            // multiple draft tokens simultaneously.
            //
            // This is equivalent to "parallel verification" speculative decoding:
            // the draft phase runs the full model one token at a time (baseline cost),
            // while the verify phase processes k+1 tokens in a single forward pass.
            // The average throughput gain equals the mean accepted tokens per verify
            // call, which is bounded by k+1 at 100% acceptance.
            //
            // Rust ownership: generate_speculative takes two separate closure
            // parameters.  Both need to call forward_with_cache on the same model
            // reference.  We wrap the model in a RefCell to allow the two closures
            // to share a borrow without unsafe code.  This is sound because both
            // closures are invoked sequentially inside generate_speculative — never
            // concurrently — so the dynamic borrow check never fails.
            let model_cell = std::cell::RefCell::new(model);

            let result = generator
                .generate_speculative(
                    |input, cache| {
                        model_cell
                            .borrow_mut()
                            .forward_with_cache(input, None, Some(cache))
                            .map_err(|e| Exception::custom(e.to_string()))
                    },
                    |input, cache| {
                        model_cell
                            .borrow_mut()
                            .forward_with_cache(input, None, Some(cache))
                            .map_err(|e| Exception::custom(e.to_string()))
                    },
                    prompt_tokens,
                )
                .map_err(|e| GrpoError::Generation(e.to_string()));

            // Log speculative stats at debug level
            if let Some(stats) = generator.last_speculative_stats() {
                tracing::debug!(
                    "Speculative rollout: acceptance={:.1}%, tokens/step={:.2}, proposed={}, accepted={}",
                    stats.acceptance_rate() * 100.0,
                    stats.tokens_per_step(),
                    stats.total_draft_proposed,
                    stats.total_draft_accepted,
                );
            }

            result
        } else {
            generator
                .generate(
                    |input, cache| {
                        model
                            .forward_with_cache(input, None, Some(cache))
                            .map_err(|e| Exception::custom(e.to_string()))
                    },
                    prompt_tokens,
                )
                .map_err(|e| GrpoError::Generation(e.to_string()))
        }
    }

    /// Run full GRPO training loop.
    pub fn run<M, R, O, F>(
        &mut self,
        policy_model: &mut M,
        mut ref_model: Option<&mut R>,
        tokenizer: &pmetal_data::Tokenizer,
        dataset: &pmetal_data::TrainingDataset,
        reward_fn: &CombinedReward,
        optimizer: &mut O,
        mut set_optimizer_lr: F,
    ) -> GrpoResult<()>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
        F: FnMut(&mut O, f32),
    {
        info!("Starting GRPO training loop...");
        let n_epochs = self.training_config.num_epochs;
        let n_samples = dataset.samples().len();
        let total_steps = n_samples * n_epochs;
        if let Some(ref mut ctrl) = self.adaptive_lr {
            ctrl.set_total_steps(total_steps);
        }

        for cb in &mut self.callbacks {
            cb.on_train_start();
        }

        for epoch in 0..n_epochs {
            info!("Epoch {}/{}", epoch + 1, n_epochs);

            for (i, sample) in dataset.samples().iter().enumerate() {
                let step_start = std::time::Instant::now();

                let gen_output =
                    self.generate_completions(policy_model, &sample.input_ids, tokenizer)?;

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

                // Load images for VLM mode.  Each completion in this group shares
                // the same prompt images.  We load them once and replicate the
                // reference for the reward function.  Image loading failures are
                // soft-logged rather than hard-erroring so text-fallback still works.
                let sample_images: Option<Vec<Array>> = if self.config.vlm_mode {
                    match &sample.images {
                        Some(paths) if !paths.is_empty() => {
                            match load_images(paths, self.config.max_image_size) {
                                Ok(imgs) => {
                                    tracing::debug!(
                                        "VLM: loaded {} image(s) for sample {}",
                                        imgs.len(),
                                        i
                                    );
                                    Some(imgs)
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "VLM: failed to load images for sample {}: {}",
                                        i,
                                        e
                                    );
                                    None
                                }
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                // Build per-completion image vectors for the reward function.
                // Each completion gets the same set of images (same prompt).
                let images_for_reward: Option<Vec<Vec<Array>>> = sample_images
                    .as_ref()
                    .map(|imgs| vec![imgs.clone(); gen_output.token_ids.len()]);

                let rewards = reward_fn.compute(
                    &vec![prompt_text; gen_output.token_ids.len()],
                    &completions_text,
                    images_for_reward.as_deref(),
                )?;

                let mut group =
                    CompletionGroup::new(sample.input_ids.clone(), self.config.num_generations);
                for (j, ids) in gen_output.token_ids.iter().enumerate() {
                    let new_ids = ids[sample.input_ids.len()..].to_vec();
                    group.add_completion(new_ids, rewards[j], gen_output.stopped_by_length[j]);
                }
                // Attach pixel values so train_step can use forward_with_images.
                group.pixel_values = sample_images;

                // Apply adaptive LR override to optimizer before step
                let current_lr = self.get_learning_rate();
                set_optimizer_lr(optimizer, current_lr);

                let stats =
                    self.train_step(policy_model, ref_model.as_deref_mut(), &[group], optimizer)?;

                // Feed loss to adaptive LR controller and handle the resulting action
                let action = self.apply_adaptive_lr_action(stats.loss as f64);

                // Snapshot best weights when loss reaches a new minimum
                if action == crate::training_loop::AdaptiveAction::Continue
                    && self.should_snapshot_best()
                {
                    self.snapshot_best_weights(policy_model);
                }

                // Rollback: restore best weights and reduce LR (already done in controller)
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

                // Early stop: restore best weights and exit
                if action == crate::training_loop::AdaptiveAction::EarlyStop {
                    self.restore_best_weights(policy_model);
                    tracing::info!(
                        "Early stopping GRPO training — adaptive LR exhausted rollbacks."
                    );
                    return Ok(());
                }

                let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                if i % 10 == 0 {
                    let adjusted_lr = self.get_learning_rate();
                    info!(
                        "Step {}: loss={:.4}, kl={:.4}, reward={:.4}, lr={:.2e}, completion_len={:.1}",
                        stats.step,
                        stats.loss,
                        stats.kl,
                        stats.reward,
                        adjusted_lr,
                        gen_output.num_generated.iter().sum::<usize>() as f32
                            / gen_output.num_generated.len() as f32
                    );
                }

                // Emit metrics to callbacks
                if !self.callbacks.is_empty() {
                    let adjusted_lr = self.get_learning_rate();
                    let metrics = pmetal_core::StepMetrics {
                        step: self.step,
                        epoch,
                        total_epochs: n_epochs,
                        total_steps,
                        loss: stats.loss as f64,
                        lr: adjusted_lr as f64,
                        tok_sec: 0.0, // GRPO doesn't track tokens the same way
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
            }
        }

        for cb in &mut self.callbacks {
            cb.on_train_end();
        }

        Ok(())
    }

    /// Run GRPO training with pipelined (asynchronous) reward scoring.
    ///
    /// Identical to [`run`] but wraps `reward_fn` in an [`AsyncRewardModel`]
    /// so reward scoring for step N overlaps with GPU training for step N+1.
    ///
    /// # Pipeline
    ///
    /// ```text
    /// Step N:   Generate → Submit score (bg thread) → GPU Train
    /// Step N+1: Generate → Collect N + Submit N+1   → GPU Train
    /// ```
    ///
    /// The very first step submits scores immediately after generation (no
    /// overlap for step 0).  From step 1 onwards, each step collects the
    /// previous step's scores at the start of the reward-building phase,
    /// which has already completed (or is very close to completing) by the
    /// time the GPU training step finishes.
    ///
    /// # Arguments
    ///
    /// Identical to [`run`] except `reward_fn` is taken by value as
    /// `Box<dyn RewardFunction>` to allow ownership transfer to the
    /// background thread.  Pass `Box::new(combined_reward)` when wrapping a
    /// `CombinedReward` (which implements `RewardFunction`).
    ///
    /// # Errors
    ///
    /// Same as [`run`], plus [`GrpoError::Reward`] if the background scorer
    /// thread terminates unexpectedly.
    pub fn run_async<M, R, O, F>(
        &mut self,
        policy_model: &mut M,
        mut ref_model: Option<&mut R>,
        tokenizer: &pmetal_data::Tokenizer,
        dataset: &pmetal_data::TrainingDataset,
        reward_fn: Box<dyn RewardFunction>,
        optimizer: &mut O,
        mut set_optimizer_lr: F,
    ) -> GrpoResult<()>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
        F: FnMut(&mut O, f32),
    {
        use crate::ane_reward::{AsyncRewardModel, PipelinedGrpoSession};

        info!("Starting GRPO training loop (pipelined reward scoring via AsyncRewardModel)...");

        let n_epochs = self.training_config.num_epochs;
        let n_samples = dataset.samples().len();
        let total_steps = n_samples * n_epochs;
        if let Some(ref mut ctrl) = self.adaptive_lr {
            ctrl.set_total_steps(total_steps);
        }

        for cb in &mut self.callbacks {
            cb.on_train_start();
        }

        // Wrap the reward function in the async executor and create a session
        // that manages the one-step lookahead pipeline.
        let async_reward = AsyncRewardModel::new(reward_fn);
        let mut session = PipelinedGrpoSession::new(&async_reward);

        // Per-step context deferred until its rewards arrive next iteration.
        struct DeferredStep {
            prompt_ids: Vec<u32>,
            /// (completion token ids, stopped_by_length)
            completions: Vec<(Vec<u32>, bool)>,
            pixel_values: Option<Vec<Array>>,
        }

        let mut deferred: Option<DeferredStep> = None;

        for epoch in 0..n_epochs {
            info!("Epoch {}/{}", epoch + 1, n_epochs);

            for (i, sample) in dataset.samples().iter().enumerate() {
                let step_start = std::time::Instant::now();

                // 1. Generate completions on GPU.
                let gen_output =
                    self.generate_completions(policy_model, &sample.input_ids, tokenizer)?;

                // 2. Decode to text (cheap CPU work).
                let prompt_text = tokenizer
                    .decode(&sample.input_ids)
                    .map_err(|e| GrpoError::Tokenizer(e.to_string()))?;

                let mut completions_text: Vec<String> =
                    Vec::with_capacity(gen_output.token_ids.len());
                for ids in &gen_output.token_ids {
                    let new_ids = &ids[sample.input_ids.len()..];
                    completions_text.push(
                        tokenizer
                            .decode(new_ids)
                            .map_err(|e| GrpoError::Tokenizer(e.to_string()))?,
                    );
                }

                // 3. Load VLM images if needed (CPU/IO, overlaps nicely with scoring).
                let sample_images: Option<Vec<Array>> = if self.config.vlm_mode {
                    match &sample.images {
                        Some(paths) if !paths.is_empty() => {
                            match load_images(paths, self.config.max_image_size) {
                                Ok(imgs) => {
                                    tracing::debug!(
                                        "VLM: loaded {} image(s) for sample {}",
                                        imgs.len(),
                                        i
                                    );
                                    Some(imgs)
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "VLM: failed to load images for sample {}: {}",
                                        i,
                                        e
                                    );
                                    None
                                }
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                // 4. Submit scoring for the *current* step to the background thread.
                //    `begin_step` also returns any rewards from the *previous* step
                //    that finished during our generation + text-decode work above.
                let prompt_repeated = vec![prompt_text; gen_output.token_ids.len()];
                let prev_rewards = session.begin_step(prompt_repeated, completions_text.clone())?;

                // 5. Stash the current step's context; swap out the previous one.
                let prev_deferred = deferred.replace(DeferredStep {
                    prompt_ids: sample.input_ids.clone(),
                    completions: gen_output
                        .token_ids
                        .iter()
                        .zip(gen_output.stopped_by_length.iter())
                        .map(|(ids, &sbl)| {
                            let new_ids = ids[sample.input_ids.len()..].to_vec();
                            (new_ids, sbl)
                        })
                        .collect(),
                    pixel_values: sample_images,
                });

                // 6–7. Run the GPU training step for the *previous* batch
                //      using its now-ready rewards.
                if let (Some(prev_ctx), Some(rewards)) = (prev_deferred, prev_rewards) {
                    let mut group =
                        CompletionGroup::new(prev_ctx.prompt_ids, self.config.num_generations);
                    for ((new_ids, sbl), reward) in prev_ctx.completions.iter().zip(rewards.iter())
                    {
                        group.add_completion(new_ids.clone(), *reward, *sbl);
                    }
                    group.pixel_values = prev_ctx.pixel_values;

                    let current_lr = self.get_learning_rate();
                    set_optimizer_lr(optimizer, current_lr);

                    let stats = self.train_step(
                        policy_model,
                        ref_model.as_deref_mut(),
                        &[group],
                        optimizer,
                    )?;

                    // Adaptive LR + rollback (mirrors the synchronous run() path).
                    let action = self.apply_adaptive_lr_action(stats.loss as f64);

                    if action == crate::training_loop::AdaptiveAction::Continue
                        && self.should_snapshot_best()
                    {
                        self.snapshot_best_weights(policy_model);
                    }

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

                    if action == crate::training_loop::AdaptiveAction::EarlyStop {
                        // Drain the in-flight request to prevent the worker from
                        // blocking on a full channel response slot.
                        let _ = session.flush();
                        self.restore_best_weights(policy_model);
                        tracing::info!(
                            "Early stopping GRPO training — adaptive LR exhausted rollbacks."
                        );
                        for cb in &mut self.callbacks {
                            cb.on_train_end();
                        }
                        return Ok(());
                    }

                    let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                    if i % 10 == 0 {
                        let adjusted_lr = self.get_learning_rate();
                        info!(
                            "Step {}: loss={:.4}, kl={:.4}, reward={:.4}, lr={:.2e}, completion_len={:.1}",
                            stats.step,
                            stats.loss,
                            stats.kl,
                            stats.reward,
                            adjusted_lr,
                            gen_output.num_generated.iter().sum::<usize>() as f32
                                / gen_output.num_generated.len() as f32
                        );
                    }

                    if !self.callbacks.is_empty() {
                        let adjusted_lr = self.get_learning_rate();
                        let metrics = pmetal_core::StepMetrics {
                            step: self.step,
                            epoch,
                            total_epochs: n_epochs,
                            total_steps,
                            loss: stats.loss as f64,
                            lr: adjusted_lr as f64,
                            tok_sec: 0.0,
                            total_ms: step_ms,
                            tokens: 0,
                            ..Default::default()
                        };
                        for cb in &mut self.callbacks {
                            cb.on_step_end_with_metrics(&metrics);
                        }
                        if self.callbacks.iter().any(|cb| cb.should_stop()) {
                            let _ = session.flush();
                            for cb in &mut self.callbacks {
                                cb.on_train_end();
                            }
                            return Err(GrpoError::Cancelled);
                        }
                    }
                }
                // On the very first iteration (i == 0), prev_deferred is None and we
                // skip training — the first GPU step happens at i == 1 using step 0's
                // rewards, which were scored during step 1's generation.
            }
        }

        // 8. Flush the final pending step.
        //    After the epoch loop, `deferred` holds the last sample's context
        //    and `session` has its scoring request in flight.  Collect and train.
        if let Some(last_ctx) = deferred.take() {
            if let Some(rewards) = session.flush()? {
                let mut group =
                    CompletionGroup::new(last_ctx.prompt_ids, self.config.num_generations);
                for ((new_ids, sbl), reward) in last_ctx.completions.iter().zip(rewards.iter()) {
                    group.add_completion(new_ids.clone(), *reward, *sbl);
                }
                group.pixel_values = last_ctx.pixel_values;

                let current_lr = self.get_learning_rate();
                set_optimizer_lr(optimizer, current_lr);

                let flush_step_start = std::time::Instant::now();
                let stats = self.train_step(policy_model, ref_model, &[group], optimizer)?;

                // Apply the same adaptive LR / rollback / callback logic as the
                // main loop so the final step participates in divergence detection
                // and best-weight snapshotting.
                let action = self.apply_adaptive_lr_action(stats.loss as f64);

                if action == crate::training_loop::AdaptiveAction::Continue
                    && self.should_snapshot_best()
                {
                    self.snapshot_best_weights(policy_model);
                }

                if action == crate::training_loop::AdaptiveAction::Rollback {
                    self.restore_best_weights(policy_model);
                    let rollback_lr = self
                        .adaptive_lr_override
                        .unwrap_or(self.training_config.learning_rate as f32);
                    set_optimizer_lr(optimizer, rollback_lr);
                    tracing::info!(
                        "GRPO rollback at flush step {}: new lr={:.2e}",
                        self.step,
                        rollback_lr
                    );
                }

                if action == crate::training_loop::AdaptiveAction::EarlyStop {
                    self.restore_best_weights(policy_model);
                    tracing::info!(
                        "Early stopping GRPO training at flush step — adaptive LR exhausted rollbacks."
                    );
                    for cb in &mut self.callbacks {
                        cb.on_train_end();
                    }
                    return Ok(());
                }

                // Fire step callbacks for the flush step.
                if !self.callbacks.is_empty() {
                    let adjusted_lr = self.get_learning_rate();
                    let step_ms = flush_step_start.elapsed().as_secs_f64() * 1000.0;
                    let metrics = pmetal_core::StepMetrics {
                        step: self.step,
                        epoch: n_epochs.saturating_sub(1),
                        total_epochs: n_epochs,
                        total_steps,
                        loss: stats.loss as f64,
                        lr: adjusted_lr as f64,
                        tok_sec: 0.0,
                        total_ms: step_ms,
                        tokens: 0,
                        ..Default::default()
                    };
                    for cb in &mut self.callbacks {
                        cb.on_step_end_with_metrics(&metrics);
                    }
                    // Honour cancellation from callbacks even at the flush step.
                    if self.callbacks.iter().any(|cb| cb.should_stop()) {
                        for cb in &mut self.callbacks {
                            cb.on_train_end();
                        }
                        return Err(GrpoError::Cancelled);
                    }
                }
            }
        }

        for cb in &mut self.callbacks {
            cb.on_train_end();
        }

        Ok(())
    }
}

/// Trait for GRPO Reward functions.
pub trait RewardFunction: Send + Sync {
    fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>>;
    fn name(&self) -> &str;
}

/// Reward function that checks for proper XML tags (e.g., <thought> and <answer>).
pub struct XmlFormatReward {
    pub tags: Vec<(String, String)>,
}

impl XmlFormatReward {
    pub fn new(tags: Vec<(String, String)>) -> Self {
        Self { tags }
    }

    pub fn default_reasoning() -> Self {
        Self::new(vec![
            ("<thought>".into(), "</thought>".into()),
            ("<answer>".into(), "</answer>".into()),
        ])
    }
}

impl RewardFunction for XmlFormatReward {
    fn compute(
        &self,
        _: &[String],
        completions: &[String],
        _: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        let mut rewards = vec![0.0; completions.len()];
        for (i, completion) in completions.iter().enumerate() {
            let mut score = 0.0;
            for (start_tag, end_tag) in &self.tags {
                if let (Some(start_idx), Some(end_idx)) =
                    (completion.find(start_tag), completion.find(end_tag))
                {
                    if start_idx < end_idx {
                        score += 0.5;
                    }
                }
            }
            rewards[i] = score;
        }
        Ok(rewards)
    }

    fn name(&self) -> &str {
        "xml_format"
    }
}

/// Reward function that checks for exact matches with ground truth answers.
///
/// Extracts the model's answer using multiple strategies (in priority order):
/// 1. Last `<answer>...</answer>` tag pair (handles retries within CoT)
/// 2. Last `\boxed{...}` expression (common in math)
/// 3. Last non-empty line of the completion (best-effort fallback)
///
/// Comparison normalizes internal whitespace (collapses runs to single space)
/// so that formatting differences don't cause false negatives.
pub struct AccuracyReward {
    pub answers: Vec<String>,
}

impl AccuracyReward {
    pub fn new(answers: Vec<String>) -> Self {
        Self { answers }
    }
}

/// Normalize whitespace for answer comparison: trim + collapse internal runs to single space.
fn normalize_answer(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Extract the model's answer from a completion string.
///
/// Tries (in order):
/// 1. Last `<answer>...</answer>` tag pair
/// 2. Last `\boxed{...}` expression (with brace-depth tracking)
/// 3. Last non-empty line
fn extract_answer(completion: &str) -> &str {
    // Strategy 1: Last <answer>...</answer> pair
    if let Some(end_pos) = completion.rfind("</answer>") {
        let search_region = &completion[..end_pos];
        if let Some(start_pos) = search_region.rfind("<answer>") {
            let content_start = start_pos + "<answer>".len();
            if content_start <= end_pos {
                return completion[content_start..end_pos].trim();
            }
        }
    }

    // Strategy 2: Last \boxed{...} with brace-depth tracking
    if let Some(boxed_pos) = completion.rfind("\\boxed{") {
        let brace_start = boxed_pos + "\\boxed{".len();
        let mut depth = 1i32;
        let mut end = brace_start;
        for (i, ch) in completion[brace_start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = brace_start + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 {
            return completion[brace_start..end].trim();
        }
    }

    // Strategy 3: Last non-empty line
    for line in completion.lines().rev() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }

    completion.trim()
}

impl RewardFunction for AccuracyReward {
    fn compute(
        &self,
        _: &[String],
        completions: &[String],
        _: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        let num_generations = completions.len() / self.answers.len();
        let mut rewards = vec![0.0; completions.len()];

        for (prompt_idx, answer) in self.answers.iter().enumerate() {
            let norm_answer = normalize_answer(answer);
            for gen_idx in 0..num_generations {
                let comp_idx = prompt_idx * num_generations + gen_idx;
                let completion = &completions[comp_idx];

                let extracted = extract_answer(completion);
                let norm_extracted = normalize_answer(extracted);

                if norm_extracted == norm_answer {
                    rewards[comp_idx] = 1.0;
                }
            }
        }
        Ok(rewards)
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}

/// Reward function that evaluates VLM completions for image-understanding quality.
///
/// Scores each completion by checking how many of the expected answer patterns
/// appear in the model's response.  The score is normalized to `[0.0, 1.0]`:
/// `0.0` means no expected pattern was found; `1.0` means all of them were.
///
/// This is designed for visual QA tasks where the dataset supplies a reference
/// answer list.  Images are accepted via the `images` parameter but are not
/// directly inspected by this reward — they were already used during generation.
///
/// # Example
/// ```no_run
/// use pmetal_trainer::{VlmAccuracyReward, RewardFunction};
/// let reward = VlmAccuracyReward::new(vec!["cat".into(), "orange".into()]);
/// let scores = reward.compute(&[], &["I see an orange cat.".into()], None).unwrap();
/// assert!((scores[0] - 1.0).abs() < 1e-6); // both patterns found
/// ```
pub struct VlmAccuracyReward {
    /// Expected answer patterns (case-insensitive substring matches).
    pub expected_answers: Vec<String>,
}

impl VlmAccuracyReward {
    /// Create a new `VlmAccuracyReward` from a list of expected answer strings.
    pub fn new(expected_answers: Vec<String>) -> Self {
        Self { expected_answers }
    }
}

impl RewardFunction for VlmAccuracyReward {
    /// Score completions against expected answer patterns.
    ///
    /// For each completion, counts how many `expected_answers` appear as
    /// case-insensitive substrings and divides by the total count.
    ///
    /// The `images` parameter is accepted for API compatibility but is not
    /// inspected here — visual context is already baked into the completions
    /// via the model's multimodal forward pass.
    fn compute(
        &self,
        _prompts: &[String],
        completions: &[String],
        _images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        let n = self.expected_answers.len();
        completions
            .iter()
            .map(|completion| {
                if n == 0 {
                    return Ok(0.0);
                }
                let lower = completion.to_lowercase();
                let hits = self
                    .expected_answers
                    .iter()
                    .filter(|ans| lower.contains(ans.to_lowercase().as_str()))
                    .count();
                Ok(hits as f64 / n as f64)
            })
            .collect()
    }

    fn name(&self) -> &str {
        "vlm_accuracy"
    }
}

/// Combined reward function with weights.
pub struct CombinedReward {
    pub functions: Vec<(Box<dyn RewardFunction>, f64)>,
}

impl CombinedReward {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    pub fn add(mut self, function: Box<dyn RewardFunction>, weight: f64) -> Self {
        self.functions.push((function, weight));
        self
    }

    pub fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        if self.functions.is_empty() {
            return Err(GrpoError::Reward("No reward functions configured".into()));
        }

        let mut total_rewards = vec![0.0; completions.len()];
        for (func, weight) in &self.functions {
            let rewards = func.compute(prompts, completions, images)?;
            for (i, r) in rewards.iter().enumerate() {
                total_rewards[i] += r * weight;
            }
        }
        Ok(total_rewards)
    }
}

impl Default for CombinedReward {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardFunction for CombinedReward {
    fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        if self.functions.is_empty() {
            return Err(GrpoError::Reward("No reward functions configured".into()));
        }
        let mut total_rewards = vec![0.0f64; completions.len()];
        for (func, weight) in &self.functions {
            let rewards = func.compute(prompts, completions, images)?;
            for (i, r) in rewards.iter().enumerate() {
                total_rewards[i] += r * weight;
            }
        }
        Ok(total_rewards)
    }

    fn name(&self) -> &str {
        "combined_reward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::Array;
    use serial_test::serial;

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /// Build a GrpoTrainer with a minimal config, overriding fields as needed.
    fn make_trainer(config: GrpoConfig) -> GrpoTrainer {
        GrpoTrainer {
            config,
            training_config: pmetal_core::TrainingConfig::default(),
            step: 0,
            adaptive_lr: None,
            adaptive_lr_override: None,
            callbacks: Vec::new(),
            best_lora_snapshot: None,
        }
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
    // 4. compute_grpo_loss — ratio = 1 (per_token_logps == old_per_token_logps)
    // ---------------------------------------------------------------------------

    /// When the current and old log-probs are identical the importance ratio is
    /// exactly 1, so the PPO-clip surrogate is:
    ///   L = -min(1 * A, clip(1, 1-ε, 1+ε) * A) = -A
    /// The final loss (averaged over tokens) should equal -mean(advantages).
    #[test]
    #[serial]
    fn test_compute_grpo_loss_ratio_one() {
        let trainer = make_trainer(GrpoConfig {
            beta: 0.0, // no KL penalty
            ..GrpoConfig::default()
        });

        // [2 sequences, 4 tokens]
        let logps_data: Vec<f32> = vec![
            -0.5, -0.5, -0.5, -0.5, // seq 0
            -1.0, -1.0, -1.0, -1.0, // seq 1
        ];
        let per_token_logps = Array::from_slice(&logps_data, &[2, 4]);
        let old_per_token_logps = Array::from_slice(&logps_data, &[2, 4]);

        // advantages: [2] — one per sequence
        let advantages_data = [1.0f32, -1.0];
        let advantages = Array::from_slice(&advantages_data, &[2]);

        // completion_mask: all valid tokens
        let mask_data = [1.0f32; 8];
        let completion_mask = Array::from_slice(&mask_data, &[2, 4]);

        let (total_loss, kl_mean, policy_loss) = trainer
            .compute_grpo_loss(
                &per_token_logps,
                &old_per_token_logps,
                None,
                &advantages,
                &completion_mask,
                None,
            )
            .unwrap();

        total_loss.eval().unwrap();
        kl_mean.eval().unwrap();
        policy_loss.eval().unwrap();

        let loss_val: f32 = total_loss.item();
        let kl_val: f32 = kl_mean.item();

        assert!(
            loss_val.is_finite(),
            "total loss must be finite, got {loss_val}"
        );

        // KL is 0 when beta=0 and no ref model
        assert!(
            kl_val.abs() < 1e-6,
            "KL should be ~0 with no ref model; got {kl_val}"
        );

        // When ratio=1 and clipping is inactive, L = -mean(A).
        // mean(A) for the whole batch over equal token counts =
        //   (-A[0] * 4 tokens + -A[1] * 4 tokens) / 8  = -(1.0 - 1.0)/2 = 0.0
        // The per-token loss is -A (broadcast), so the mean should be ~0
        // (advantages cancel: one positive, one negative with equal weight).
        assert!(
            loss_val.abs() < 1e-5,
            "loss should be ~0 when advantages cancel; got {loss_val}"
        );
    }

    // ---------------------------------------------------------------------------
    // 5. compute_grpo_loss — PPO clipping
    // ---------------------------------------------------------------------------

    /// Drive the ratio far outside [0.8, 1.2] by using very different log-probs.
    /// The loss should still be finite (clipping prevents exploding gradients).
    #[test]
    #[serial]
    fn test_compute_grpo_loss_clipping() {
        let trainer = make_trainer(GrpoConfig {
            beta: 0.0,
            epsilon_low: 0.2,
            epsilon_high: 0.2,
            ..GrpoConfig::default()
        });

        // Current policy: very different from old (log ratio ~ +5)
        let per_token_logps = Array::from_slice(
            &[-0.1f32, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
            &[2, 4],
        );
        // Old policy: much lower log-probs → log_ratio = (-0.1) - (-5.1) = 5.0
        let old_per_token_logps = Array::from_slice(
            &[-5.1f32, -5.1, -5.1, -5.1, -5.1, -5.1, -5.1, -5.1],
            &[2, 4],
        );

        let advantages = Array::from_slice(&[1.0f32, 1.0], &[2]);
        let completion_mask = Array::from_slice(&[1.0f32; 8], &[2, 4]);

        let (total_loss, _kl, _policy_loss) = trainer
            .compute_grpo_loss(
                &per_token_logps,
                &old_per_token_logps,
                None,
                &advantages,
                &completion_mask,
                None,
            )
            .unwrap();

        total_loss.eval().unwrap();

        let loss_val: f32 = total_loss.item();

        assert!(!loss_val.is_nan(), "loss must not be NaN; got {loss_val}");
        assert!(loss_val.is_finite(), "loss must be finite; got {loss_val}");

        // With positive advantages and clipping at 1.2, the clipped surrogate yields:
        //   surr2 = clip(ratio, 0.8, 1.2) * A = 1.2 * 1.0 = 1.2  (ratio >> 1.2)
        //   surr1 = ratio * A >> 1.2
        //   min(surr1, surr2) = 1.2
        //   token_policy_loss = -min(...) = -1.2
        // Averaged across all tokens and both sequences → loss = -1.2.
        // This is correct: gradient descent on a negative loss ascends reward.
        let expected = -1.2f32;
        assert!(
            (loss_val - expected).abs() < 1e-4,
            "clipped loss should be ~{expected}, got {loss_val}"
        );
    }

    // ---------------------------------------------------------------------------
    // 6. KL penalty is non-negative
    // ---------------------------------------------------------------------------

    /// KL(pi || ref) estimated via the Schulman approximation:
    ///   kl ≈ exp(ref - pi) - 1 - (ref - pi)
    /// This is always >= 0. Verify that:
    ///   a) kl_mean >= 0 when policy ≠ reference
    ///   b) kl_mean ≈ 0 when policy == reference
    #[test]
    #[serial]
    fn test_kl_penalty_non_negative() {
        let trainer = make_trainer(GrpoConfig {
            beta: 0.1,
            ..GrpoConfig::default()
        });

        let per_token_logps = Array::from_slice(
            &[-0.5f32, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
            &[2, 4],
        );
        // Reference is much more certain (higher log-probs)
        let ref_per_token_logps = Array::from_slice(
            &[-0.1f32, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
            &[2, 4],
        );

        let advantages = Array::from_slice(&[0.0f32, 0.0], &[2]);
        let completion_mask = Array::from_slice(&[1.0f32; 8], &[2, 4]);

        // Case A: different ref → KL should be > 0
        let (_total, kl_mean_a, _policy) = trainer
            .compute_grpo_loss(
                &per_token_logps,
                &per_token_logps, // old == current (ratio=1)
                Some(&ref_per_token_logps),
                &advantages,
                &completion_mask,
                None,
            )
            .unwrap();

        kl_mean_a.eval().unwrap();
        let kl_val_a: f32 = kl_mean_a.item();
        assert!(kl_val_a >= 0.0, "KL must be >= 0; got {kl_val_a}");
        assert!(kl_val_a.is_finite(), "KL must be finite; got {kl_val_a}");
        // Expectation: ref is closer to 0 than policy, so KL > 0 is expected.
        // The Schulman approx: exp(ref-pi) - 1 - (ref-pi) = exp(0.4) - 1 - 0.4 ≈ 0.092
        assert!(
            kl_val_a > 0.01,
            "KL should be noticeably positive; got {kl_val_a}"
        );

        // Case B: ref == policy → KL should be ~0
        let (_total, kl_mean_b, _policy) = trainer
            .compute_grpo_loss(
                &per_token_logps,
                &per_token_logps,
                Some(&per_token_logps), // ref == policy
                &advantages,
                &completion_mask,
                None,
            )
            .unwrap();

        kl_mean_b.eval().unwrap();
        let kl_val_b: f32 = kl_mean_b.item();
        assert!(
            kl_val_b.abs() < 1e-5,
            "KL should be ~0 when ref==policy; got {kl_val_b}"
        );
    }

    // ---------------------------------------------------------------------------
    // 7. All four loss types produce finite scalars
    // ---------------------------------------------------------------------------

    #[test]
    #[serial]
    fn test_grpo_loss_reduction_variants() {
        let loss_types = [
            GrpoLossType::Bnpo,
            GrpoLossType::DrGrpo,
            GrpoLossType::Dapo,
            GrpoLossType::Reinforce,
        ];

        let per_token_logps = Array::from_slice(
            &[-0.5f32, -0.6, -0.7, -0.8, -0.4, -0.5, -0.6, -0.7],
            &[2, 4],
        );
        let old_per_token_logps = Array::from_slice(
            &[-0.5f32, -0.6, -0.7, -0.8, -0.4, -0.5, -0.6, -0.7],
            &[2, 4],
        );
        let advantages = Array::from_slice(&[0.5f32, -0.5], &[2]);
        let completion_mask = Array::from_slice(&[1.0f32; 8], &[2, 4]);

        for loss_type in loss_types {
            let trainer = make_trainer(GrpoConfig {
                beta: 0.0,
                loss_type,
                ..GrpoConfig::default()
            });

            let (total_loss, _kl, _policy_loss) = trainer
                .compute_grpo_loss(
                    &per_token_logps,
                    &old_per_token_logps,
                    None,
                    &advantages,
                    &completion_mask,
                    None,
                )
                .unwrap();

            total_loss.eval().unwrap();
            let loss_val: f32 = total_loss.item();

            assert!(
                !loss_val.is_nan(),
                "loss_type={loss_type:?}: loss must not be NaN, got {loss_val}"
            );
            assert!(
                loss_val.is_finite(),
                "loss_type={loss_type:?}: loss must be finite, got {loss_val}"
            );
        }
    }

    // ---------------------------------------------------------------------------
    // 8. XmlFormatReward
    // ---------------------------------------------------------------------------

    #[test]
    fn test_xml_format_reward() {
        let reward = XmlFormatReward::default_reasoning();

        // Proper XML with correct tag ordering → 2 tag-pairs × 0.5 = 1.0
        let good = "<thought>I need to think.</thought><answer>42</answer>".to_string();
        let rewards = reward
            .compute(&["prompt".to_string()], &[good], None)
            .unwrap();
        assert_eq!(rewards.len(), 1, "should return one reward per completion");
        assert!(
            (rewards[0] - 1.0).abs() < 1e-12,
            "proper XML should score 1.0, got {}",
            rewards[0]
        );

        // Missing both thought tags entirely → only answer pair can score → 0.5
        let missing_thought = "<answer>42</answer>".to_string();
        let rewards = reward
            .compute(&["prompt".to_string()], &[missing_thought], None)
            .unwrap();
        assert!(
            (rewards[0] - 0.5).abs() < 1e-12,
            "missing thought tags should score 0.5 (only answer pair valid), got {}",
            rewards[0]
        );

        // Missing ALL closing tags → score 0.0
        let missing_close = "<thought>no close tag <answer>42".to_string();
        let rewards = reward
            .compute(&["prompt".to_string()], &[missing_close], None)
            .unwrap();
        assert!(
            rewards[0].abs() < 1e-12,
            "missing all closing tags should score 0, got {}",
            rewards[0]
        );

        // Reversed tags (end before start) → the start_idx < end_idx check fails → 0.0
        let reversed = "</thought>content<thought><answer>42</answer>".to_string();
        let rewards = reward
            .compute(&["prompt".to_string()], &[reversed], None)
            .unwrap();
        // <thought> score: </thought> appears first → start_idx > end_idx → 0
        // <answer> score: correct → 0.5
        // Total: 0.5 (only the answer pair is valid)
        assert!(
            rewards[0] < 1.0,
            "reversed <thought> tags should not score full 1.0, got {}",
            rewards[0]
        );

        // Completely empty completion → 0.0
        let empty = "".to_string();
        let rewards = reward
            .compute(&["prompt".to_string()], &[empty], None)
            .unwrap();
        assert!(
            rewards[0].abs() < 1e-12,
            "empty completion should score 0.0, got {}",
            rewards[0]
        );
    }

    // ---------------------------------------------------------------------------
    // 9. AccuracyReward
    // ---------------------------------------------------------------------------

    #[test]
    fn test_accuracy_reward() {
        // 1 prompt, 2 generations → answers has 1 entry, completions has 2
        let reward = AccuracyReward::new(vec!["42".to_string()]);

        // Exact match inside <answer> tags
        let exact = "<thought>some thought</thought><answer>42</answer>".to_string();
        // Non-matching completion
        let wrong = "<answer>99</answer>".to_string();

        let rewards = reward
            .compute(
                &["what is 6*7?".to_string(), "what is 6*7?".to_string()],
                &[exact, wrong],
                None,
            )
            .unwrap();

        assert_eq!(rewards.len(), 2);
        assert!(
            (rewards[0] - 1.0).abs() < 1e-12,
            "exact match should score 1.0, got {}",
            rewards[0]
        );
        assert!(
            rewards[1].abs() < 1e-12,
            "wrong answer should score 0.0, got {}",
            rewards[1]
        );

        // No <answer> tags: raw completion compared directly to ground truth
        let raw_exact = AccuracyReward::new(vec!["hello".to_string()]);
        let completions = vec!["  hello  ".to_string(), "goodbye".to_string()];
        let rewards = raw_exact
            .compute(
                &["prompt".to_string(), "prompt".to_string()],
                &completions,
                None,
            )
            .unwrap();

        assert!(
            (rewards[0] - 1.0).abs() < 1e-12,
            "trimmed raw match should score 1.0, got {}",
            rewards[0]
        );
        assert!(
            rewards[1].abs() < 1e-12,
            "non-matching raw completion should score 0.0, got {}",
            rewards[1]
        );
    }

    // ---------------------------------------------------------------------------
    // 10. CombinedReward
    // ---------------------------------------------------------------------------

    #[test]
    fn test_combined_reward() {
        // Build a combined reward: 0.5 * xml_format + 0.5 * accuracy
        let combined = CombinedReward::new()
            .add(Box::new(XmlFormatReward::default_reasoning()), 0.5)
            .add(Box::new(AccuracyReward::new(vec!["42".to_string()])), 0.5);

        // Perfect completion: correct XML AND correct answer
        let perfect = "<thought>some reasoning</thought><answer>42</answer>".to_string();
        // Xml only: correct formatting but wrong answer
        let xml_only = "<thought>some reasoning</thought><answer>99</answer>".to_string();
        // Neither: no tags, wrong answer
        let neither = "the answer is probably 7".to_string();

        let completions = vec![perfect, xml_only, neither];
        let prompts = vec!["what is 6*7?".to_string(); 3];

        let rewards = combined.compute(&prompts, &completions, None).unwrap();

        assert_eq!(rewards.len(), 3);

        // Perfect: xml=1.0*0.5 + accuracy=1.0*0.5 = 1.0
        assert!(
            (rewards[0] - 1.0).abs() < 1e-12,
            "perfect completion should score 1.0, got {}",
            rewards[0]
        );

        // Xml only: xml=1.0*0.5 + accuracy=0.0*0.5 = 0.5
        assert!(
            (rewards[1] - 0.5).abs() < 1e-12,
            "xml-only should score 0.5, got {}",
            rewards[1]
        );

        // Neither: 0.0
        assert!(
            rewards[2].abs() < 1e-12,
            "neither should score 0.0, got {}",
            rewards[2]
        );

        // Empty functions list → error
        let empty = CombinedReward::new();
        let err = empty.compute(&[], &[], None).unwrap_err();
        assert!(
            matches!(err, GrpoError::Reward(_)),
            "empty CombinedReward should error, got {err:?}"
        );
    }
}
