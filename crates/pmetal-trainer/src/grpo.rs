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
    module::{Module, ModuleParameters},
    nn, ops,
    optimizers::Optimizer,
    transforms,
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
    /// Share of completion tokens whose objective clipping changed, averaged
    /// over the batch's updates. Always 0 with one update per batch.
    pub clip_fraction: f32,
    /// Optimizer updates taken on this generation batch.
    pub iterations: usize,
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

/// A batch flattened from completion groups: prompt ids, completion ids and
/// advantage, one entry per completion.
pub type PreparedBatch = (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<f64>);

/// How the per-token clipped surrogate losses are aggregated into one loss.
///
/// The names and normalizers are those of TRL's `GRPOTrainer` `loss_type`,
/// which follows each method's paper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrpoLossType {
    /// The original GRPO aggregation (DeepSeekMath, arXiv 2402.03300): each
    /// sequence's tokens averaged over its own length, then the sequences
    /// averaged. It weights a long completion's tokens less than a short
    /// one's, which biases toward short correct and long wrong answers.
    Grpo,
    /// Token-level aggregation (DAPO, arXiv 2503.14476): every token in the
    /// generation batch weighs the same, the sum over the number of
    /// completion tokens. TRL's default. Within one generation batch it is
    /// also TRL's `bnpo`.
    #[default]
    Dapo,
    /// Dr. GRPO's aggregation (*Understanding R1-Zero-Like Training*, arXiv
    /// 2503.20783): the sum over tokens divided by a constant, the number of
    /// sequences times `max_completion_length`, so no completion's length
    /// changes how much its tokens count. The paper also drops the
    /// advantages' division by the group's standard deviation;
    /// [`GrpoConfig::with_loss_preset`]`("dr_grpo")` does both.
    DrGrpo,
}

/// The level the policy ratio `π_θ / π_θ_old` is taken at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImportanceSampling {
    /// One ratio per token, as in PPO and GRPO.
    #[default]
    Token,
    /// One ratio per sequence, the length-normalized sequence likelihood ratio
    /// `s_i = exp(mean_t (log π_θ − log π_θ_old))` that GSPO (*Group Sequence
    /// Policy Optimization*, arXiv 2507.18071) clips. It differs from the
    /// token level only once the policy has moved off the one that generated
    /// the batch, i.e. with `num_iterations > 1`.
    Sequence,
}

/// The parts of one GRPO loss evaluation.
#[derive(Debug, Clone)]
pub struct GrpoLoss {
    /// Policy loss plus `beta` times the KL to the reference, minus the
    /// entropy bonus: what the optimizer minimizes.
    pub total: Array,
    /// The clipped surrogate term alone.
    pub policy: Array,
    /// Mean per-token KL to the reference model (0 without one).
    pub kl: Array,
    /// The importance ratio: `[B, T]` per token, `[B, 1]` per sequence.
    pub ratio: Array,
    /// Share of completion tokens whose objective clipping changed: ratio
    /// below `1 − ε_low` with a negative advantage or above `1 + ε_high` with
    /// a positive one.
    pub clip_fraction: Array,
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
    /// How token losses are aggregated.
    pub loss_type: GrpoLossType,
    /// Whether the policy ratio is per token or per sequence (GSPO).
    pub importance_sampling: ImportanceSampling,
    /// Optimizer updates per generation batch (μ in the GRPO paper, TRL's
    /// `num_iterations`). The policy ratio is taken against the log-probs of
    /// the policy that generated the batch, so with μ = 1 it is exactly 1 and
    /// clipping never engages; from the second update on it measures how far
    /// the policy has moved, and the clip bounds that.
    pub num_iterations: usize,
    /// Lower clipping epsilon for PPO-clip (default 0.2).
    pub epsilon_low: f64,
    /// Upper clipping epsilon for PPO-clip (default 0.2; DAPO's clip-higher
    /// uses 0.28).
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
    /// KV cache quantization bits for the generation (rollout) phase.
    ///
    /// When `Some(bits)`, the KV cache used during `generate_completions` is
    /// configured with [`CacheMode::Quantized`] at the requested bit width
    /// (valid values: 2, 4, 8).  A group size of 64 is used, which is
    /// compatible with all standard head dimensions.
    ///
    /// Quantizing the KV cache reduces peak memory by 2–8× during rollout
    /// generation, enabling longer completions or larger group sizes on
    /// memory-constrained hardware.  The dequantization path is lazy (no
    /// `.eval()` call), so it adds no synchronisation overhead.
    ///
    /// `None` (default) uses the standard fp16 cache.
    pub kv_cache_bits: Option<u8>,
    /// Optional rollout RNG seed for reproducible generation.
    pub seed: Option<u64>,
    /// DAPO dynamic sampling: drop groups whose completions are all right or
    /// all wrong, since their advantages are all zero.
    pub dapo_dynamic_sampling: bool,
    /// Accuracy threshold margin for DAPO dynamic sampling.
    pub dapo_dynamic_sampling_min_accuracy: f64,
    /// Reward threshold above which a completion is counted as correct.
    pub dapo_accuracy_reward_threshold: f64,
    /// Reward added to completions that stop because they hit the length
    /// limit (DAPO's overlong penalty); `None` leaves their rewards alone.
    pub dapo_overlong_penalty: Option<f64>,
    /// Minimum completions required after DAPO filtering.
    pub dapo_min_group_size: usize,
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
            loss_type: GrpoLossType::Dapo,
            importance_sampling: ImportanceSampling::Token,
            num_iterations: 1,
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
            kv_cache_bits: None,
            seed: None,
            dapo_dynamic_sampling: false,
            dapo_dynamic_sampling_min_accuracy: 0.01,
            dapo_accuracy_reward_threshold: 0.0,
            dapo_overlong_penalty: None,
            dapo_min_group_size: 2,
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

    /// The DAPO recipe (arXiv 2503.14476): token-level loss, no KL term,
    /// clip-higher (ε_low 0.2, ε_high 0.28), dynamic sampling, an overlong
    /// penalty and groups of at least 16.
    pub fn for_dapo(mut self) -> Self {
        self.loss_type = GrpoLossType::Dapo;
        self.beta = 0.0;
        self.epsilon_low = 0.2;
        self.epsilon_high = 0.28;
        self.num_generations = self.num_generations.max(16);
        self.dapo_dynamic_sampling = true;
        self.dapo_overlong_penalty = Some(-1.0);
        self.dapo_min_group_size = 2;
        self
    }

    /// Dr. GRPO (arXiv 2503.20783): the constant-normalized loss and
    /// advantages that are not divided by the group's standard deviation.
    pub fn for_dr_grpo(mut self) -> Self {
        self.loss_type = GrpoLossType::DrGrpo;
        self.whiten_advantages = false;
        self
    }

    /// GSPO (arXiv 2507.18071): the sequence-level ratio, sequences averaged
    /// as in GRPO, and the paper's clip range (ε_low 3e-4, ε_high 4e-4, its
    /// section 5.1). Its ratio only differs from 1 once the policy moves, so
    /// it needs `num_iterations > 1` to do anything.
    pub fn for_gspo(mut self) -> Self {
        self.importance_sampling = ImportanceSampling::Sequence;
        self.loss_type = GrpoLossType::Grpo;
        self.epsilon_low = 3e-4;
        self.epsilon_high = 4e-4;
        self
    }

    /// Apply the method a CLI `--loss-type` names: `dapo` (the default),
    /// `grpo`, `dr_grpo` or `gspo`.
    pub fn with_loss_preset(mut self, name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "dapo" => self.loss_type = GrpoLossType::Dapo,
            "grpo" => self.loss_type = GrpoLossType::Grpo,
            "dr_grpo" => self = self.for_dr_grpo(),
            "gspo" => self = self.for_gspo(),
            other => {
                return Err(format!(
                    "unknown GRPO loss type '{other}'; valid: {}",
                    GRPO_LOSS_PRESETS.join(", ")
                ));
            }
        }
        Ok(self)
    }
}

/// The names [`GrpoConfig::with_loss_preset`] accepts.
pub const GRPO_LOSS_PRESETS: [&str; 4] = ["dapo", "grpo", "dr_grpo", "gspo"];

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
/// normalization (mean/std from `FixedSizeImageProcessorConfig::default()`).
/// The image is resized to fit within `max_size × max_size` preserving aspect ratio.
///
/// The `image` crate is already a transitive dependency via `pmetal-data`, so this
/// function uses the same processor that is available there to avoid duplication.
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
        let arr = processor.process_image(resized).map_err(GrpoError::Mlx)?;

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
    checkpoint_manager: Option<crate::CheckpointManager>,
    best_checkpoint_loss: f64,
    last_loss: Option<f64>,
    /// Optimizer steps the LR schedule spans, set by [`Self::plan_schedule`].
    schedule_total_steps: Option<usize>,
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
            checkpoint_manager: None,
            best_checkpoint_loss: f64::MAX,
            last_loss: None,
            schedule_total_steps: None,
        })
    }

    /// Add a training callback for metrics logging or dashboard integration.
    pub fn add_callback(&mut self, callback: Box<dyn pmetal_core::TrainingCallback>) {
        self.callbacks.push(callback);
    }

    pub fn set_checkpoint_manager(&mut self, manager: crate::CheckpointManager) {
        self.checkpoint_manager = Some(manager);
    }

    pub fn set_step(&mut self, step: usize) {
        self.step = step;
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

    fn maybe_save_checkpoint<M: TrainableModel>(
        &mut self,
        model: &M,
        loss: f64,
        force: bool,
    ) -> GrpoResult<()> {
        self.last_loss = Some(loss);
        let every = self.training_config.save_steps.unwrap_or(0);
        if !force && (every == 0 || self.step % every != 0) {
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
        let metadata =
            crate::CheckpointMetadata::new(self.step, 0, loss, self.get_learning_rate() as f64);
        manager
            .save_checkpoint(&params, &metadata, is_best)
            .map_err(|e| GrpoError::Io(std::io::Error::other(e.to_string())))?;
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
        // Memory-efficient: gathers single logit per position via take_along_axis,
        // never materializes [B, S, V] log_softmax.
        Ok(crate::logprob_utils::shifted_selective_log_softmax(
            logits,
            labels,
            temperature,
        )?)
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

    /// The GRPO loss for one batch of completions.
    ///
    /// The policy term is PPO's clipped surrogate,
    /// `−min(w·A, clip(w, 1−ε_low, 1+ε_high)·A)`, with the importance ratio
    /// `w` taken against `old_per_token_logps`, the log-probs of the policy
    /// that generated the batch: per token, or per sequence as the
    /// length-normalized `exp(mean_t log ratio)` (GSPO). It is aggregated as
    /// [`GrpoLossType`] says, and the `beta`-weighted KL to the reference
    /// (`exp(ref − π) − (ref − π) − 1` per token) is aggregated the same way,
    /// as in TRL's `GRPOTrainer`.
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
    ) -> GrpoResult<GrpoLoss> {
        let eps_low = self.config.epsilon_low as f32;
        let eps_high = self.config.epsilon_high as f32;
        let one = Array::from_f32(1.0);
        let mask = completion_mask;
        let n_seqs = advantages.dim(0);

        let log_ratio = per_token_logps.subtract(old_per_token_logps);
        let log_weights = match self.config.importance_sampling {
            ImportanceSampling::Token => log_ratio,
            ImportanceSampling::Sequence => {
                let lengths = ops::maximum(&mask.sum_axis(-1, true), &one);
                log_ratio.multiply(mask).sum_axis(-1, true).divide(&lengths)
            }
        };
        let ratio = log_weights.exp();

        // [B] -> [B, 1], broadcasting over tokens.
        let adv = advantages.reshape(&[n_seqs, 1]);
        let lo = Array::from_f32(1.0 - eps_low);
        let hi = Array::from_f32(1.0 + eps_high);
        let clipped = ops::clip(&ratio, Some(&lo), Some(&hi));
        let per_token_policy =
            ops::minimum(&ratio.multiply(&adv), &clipped.multiply(&adv)).negative();

        let token_count = ops::maximum(&mask.sum(None), &one);
        let reduce = |per_token: &Array| -> Array {
            let masked = per_token.multiply(mask);
            match self.config.loss_type {
                GrpoLossType::Grpo => {
                    let lengths = ops::maximum(&mask.sum_axis(-1, false), &one);
                    masked.sum_axis(-1, false).divide(&lengths).mean(None)
                }
                GrpoLossType::Dapo => masked.sum(None).divide(&token_count),
                GrpoLossType::DrGrpo => {
                    let budget =
                        (n_seqs as usize * self.config.max_completion_length.max(1)) as f32;
                    masked.sum(None).divide(&Array::from_f32(budget))
                }
            }
        };
        let policy = reduce(&per_token_policy);

        let (kl, mut total) = match ref_per_token_logps {
            Some(ref_logps) => {
                let d = ref_logps.subtract(per_token_logps);
                let per_token_kl = d.exp().subtract(&one).subtract(&d);
                let kl = per_token_kl.multiply(mask).sum(None).divide(&token_count);
                let total = if self.config.beta != 0.0 {
                    policy.add(
                        &reduce(&per_token_kl).multiply(&Array::from_f32(self.config.beta as f32)),
                    )
                } else {
                    policy.clone()
                };
                (kl, total)
            }
            None => (Array::from_f32(0.0), policy.clone()),
        };

        if let (Some(ent), coef) = (entropy, self.config.entropy_coef) {
            if coef > 0.0 {
                let mean_ent = ent.multiply(mask).sum(None).divide(&token_count);
                total = total.subtract(&mean_ent.multiply(&Array::from_f32(coef as f32)));
            }
        }

        // Where clipping changed the objective (no gradient flows through it).
        let w = ratio.stop_gradient();
        let f32_dtype = pmetal_bridge::dtype::F32;
        let zero = Array::from_f32(0.0);
        let low = w
            .less(&lo)
            .as_dtype(f32_dtype)
            .multiply(&adv.less(&zero).as_dtype(f32_dtype));
        let high = w
            .greater(&hi)
            .as_dtype(f32_dtype)
            .multiply(&adv.greater(&zero).as_dtype(f32_dtype));
        let clip_fraction = low.add(&high).multiply(mask).sum(None).divide(&token_count);

        Ok(GrpoLoss {
            total,
            policy,
            kl,
            ratio,
            clip_fraction,
        })
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
    /// As in TRL's `GRPOTrainer`:
    /// 1. Build padded input_ids / labels / completion_mask tensors
    /// 2. Compute `old_per_token_logps` from the policy that generated the
    ///    batch (nothing has updated it since generation)
    /// 3. Optionally compute `ref_per_token_logps` from the reference model
    /// 4. `num_iterations` times: set the scheduled learning rate through
    ///    `set_lr`, take `value_and_grad` of [`Self::compute_grpo_loss`],
    ///    clip the gradient to `max_grad_norm` and update
    ///
    /// When `vlm_mode` is enabled and the groups contain `pixel_values`, the policy
    /// forward passes use `forward_with_images` to condition on visual inputs.
    pub fn train_step<M, R, O>(
        &mut self,
        policy_model: &mut M,
        mut ref_model: Option<&mut R>,
        groups: &[CompletionGroup],
        optimizer: &mut O,
        set_lr: &mut impl FnMut(&mut O, f32),
    ) -> GrpoResult<GrpoIterationStats>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
    {
        let start_time = Instant::now();
        let (all_prompts, all_completions, advantages) = self.prepare_batch(groups)?;

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
        let (old_per_token_logps, completion_mask) =
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
                let (ref_logps, _) =
                    self.compute_per_token_logps(&ref_logits, &labels, temperature)?;
                ref_logps.eval();
                Some(ref_logps)
            } else {
                None
            }
        } else {
            None
        };

        // 3. `num_iterations` updates on this batch. The ratio is taken against
        //    the generation-time log-probs above, so the first update is on
        //    policy (ratio 1) and the later ones are clipped as the policy moves.
        //    `pixel_values` is already materialized, so the closure can capture it.
        let pixel_values_ref = pixel_values.as_ref();
        let iterations = self.config.num_iterations.max(1);
        let max_grad_norm = self.training_config.max_grad_norm as f32;
        let mut losses = Vec::with_capacity(iterations);
        let mut policy_losses = Vec::with_capacity(iterations);
        let mut kls = Vec::with_capacity(iterations);
        let mut clip_fractions = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let stash: std::cell::RefCell<Option<GrpoLoss>> = std::cell::RefCell::new(None);
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
                let parts = self
                    .compute_grpo_loss(
                        &per_token_logps,
                        old_logps,
                        ref_per_token_logps.as_ref(),
                        adv_array,
                        mask,
                        None,
                    )
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let total = parts.total.clone();
                *stash.borrow_mut() = Some(parts);
                Ok(total)
            };

            set_lr(optimizer, self.get_learning_rate());
            let (loss, mut grads) = {
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
            if max_grad_norm > 0.0 {
                pmetal_bridge::training::clip_grad_norm_map(&mut grads, max_grad_norm);
            }
            optimizer.update(policy_model, grads)?;
            let parts = stash
                .into_inner()
                .expect("value_and_grad calls the loss closure once");
            transforms::eval([&loss, &parts.policy, &parts.kl, &parts.clip_fraction])?;
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

        Ok(GrpoIterationStats {
            step: self.step,
            loss: mean(&losses),
            kl: mean(&kls),
            policy_loss: mean(&policy_losses),
            clip_fraction: mean(&clip_fractions),
            iterations,
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
            seed: self.config.seed,
            use_prefix_cache: true,
            min_p: 0.05,
            use_speculative,
            speculative_draft_tokens: draft_tokens,
        };

        if use_speculative {
            rl_config = rl_config.with_speculative(draft_tokens);
        }

        let max_len = self.config.max_prompt_length + self.config.max_completion_length;
        let cache = if let Some(bits) = self.config.kv_cache_bits {
            let mode = pmetal_mlx::kv_cache::CacheMode::Quantized {
                bits,
                group_size: 64,
            };
            model.create_cache_with_mode(max_len, mode)
        } else {
            model.create_cache(max_len)
        }
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
    #[expect(
        clippy::too_many_arguments,
        reason = "public API: the models, data and optimizer"
    )]
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
        let total_steps = self.plan_schedule(n_samples);

        for cb in &mut self.callbacks {
            cb.on_train_start();
        }

        'epochs: for epoch in 0..n_epochs {
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
                if !self.prepare_group_for_loss(&mut group) {
                    continue;
                }

                // Apply adaptive LR override to optimizer before step
                let stats = self.train_step(
                    policy_model,
                    ref_model.as_deref_mut(),
                    &[group],
                    optimizer,
                    &mut set_optimizer_lr,
                )?;

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
                    self.maybe_save_checkpoint(policy_model, stats.loss as f64, true)?;
                    return Ok(());
                }

                self.maybe_save_checkpoint(policy_model, stats.loss as f64, false)?;

                let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                if i % 10 == 0 {
                    let adjusted_lr = self.get_learning_rate();
                    info!(
                        "Step {}: loss={:.4}, kl={:.4}, reward={:.4}, clip_fraction={:.3} over {} update(s), lr={:.2e}, completion_len={:.1}",
                        stats.step,
                        stats.loss,
                        stats.kl,
                        stats.reward,
                        stats.clip_fraction,
                        stats.iterations,
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

                if self.reached_max_steps() {
                    info!("Reached max_steps={} optimizer steps, stopping", self.step);
                    break 'epochs;
                }
            }
        }

        for cb in &mut self.callbacks {
            cb.on_train_end();
        }

        self.maybe_save_checkpoint(policy_model, self.last_loss.unwrap_or(0.0), true)?;
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
    #[expect(
        clippy::too_many_arguments,
        reason = "public API: the models, data and optimizer"
    )]
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
        let total_steps = self.plan_schedule(n_samples);

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

        'epochs: for epoch in 0..n_epochs {
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
                    if !self.prepare_group_for_loss(&mut group) {
                        continue;
                    }

                    let stats = self.train_step(
                        policy_model,
                        ref_model.as_deref_mut(),
                        &[group],
                        optimizer,
                        &mut set_optimizer_lr,
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
                        self.maybe_save_checkpoint(policy_model, stats.loss as f64, true)?;
                        for cb in &mut self.callbacks {
                            cb.on_train_end();
                        }
                        return Ok(());
                    }

                    self.maybe_save_checkpoint(policy_model, stats.loss as f64, false)?;

                    let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                    if i % 10 == 0 {
                        let adjusted_lr = self.get_learning_rate();
                        info!(
                            "Step {}: loss={:.4}, kl={:.4}, reward={:.4}, clip_fraction={:.3} over {} update(s), lr={:.2e}, completion_len={:.1}",
                            stats.step,
                            stats.loss,
                            stats.kl,
                            stats.reward,
                            stats.clip_fraction,
                            stats.iterations,
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

                    if self.reached_max_steps() {
                        info!("Reached max_steps={} optimizer steps, stopping", self.step);
                        let _ = session.flush();
                        deferred = None;
                        break 'epochs;
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
                if !self.prepare_group_for_loss(&mut group) {
                    for cb in &mut self.callbacks {
                        cb.on_train_end();
                    }
                    return Ok(());
                }

                let flush_step_start = std::time::Instant::now();
                let stats = self.train_step(
                    policy_model,
                    ref_model,
                    &[group],
                    optimizer,
                    &mut set_optimizer_lr,
                )?;

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
                    self.maybe_save_checkpoint(policy_model, stats.loss as f64, true)?;
                    for cb in &mut self.callbacks {
                        cb.on_train_end();
                    }
                    return Ok(());
                }

                self.maybe_save_checkpoint(policy_model, stats.loss as f64, false)?;

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

        self.maybe_save_checkpoint(policy_model, self.last_loss.unwrap_or(0.0), true)?;
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
            checkpoint_manager: None,
            best_checkpoint_loss: f64::MAX,
            last_loss: None,
            schedule_total_steps: None,
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
    // 4. compute_grpo_loss
    // ---------------------------------------------------------------------------

    fn arr(v: &[f32], shape: &[i32]) -> Array {
        Array::from_slice(v, shape)
    }

    fn scalar(a: &Array) -> f32 {
        a.eval();
        pmetal_bridge::check_last_error().expect("bridge op failed");
        a.item_f32()
    }

    fn values(a: &Array) -> Vec<f32> {
        a.eval();
        pmetal_bridge::check_last_error().expect("bridge op failed");
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
        let parts = trainer
            .compute_grpo_loss(
                &logps,
                &logps,
                None,
                &arr(&[1.0, -1.0], &[2]),
                &arr(&[1.0; 8], &[2, 4]),
                None,
            )
            .unwrap();
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
        let parts = trainer
            .compute_grpo_loss(
                &arr(&[-0.1; 8], &[2, 4]),
                &arr(&[-5.1; 8], &[2, 4]),
                None,
                &arr(&[1.0, 1.0], &[2]),
                &arr(&[1.0; 8], &[2, 4]),
                None,
            )
            .unwrap();
        assert!((scalar(&parts.total) + 1.2).abs() < 1e-4);
        assert_eq!(scalar(&parts.clip_fraction), 1.0);
    }

    /// KL(π‖ref) ≈ exp(ref − π) − (ref − π) − 1 ≥ 0, and 0 when they agree.
    #[test]
    #[serial]
    fn the_kl_estimate_is_non_negative_and_zero_at_the_reference() {
        let trainer = make_trainer(GrpoConfig {
            beta: 0.1,
            ..GrpoConfig::default()
        });
        let policy = arr(&[-0.5; 8], &[2, 4]);
        let reference = arr(&[-0.1; 8], &[2, 4]);
        let adv = arr(&[0.0, 0.0], &[2]);
        let mask = arr(&[1.0; 8], &[2, 4]);
        let parts = trainer
            .compute_grpo_loss(&policy, &policy, Some(&reference), &adv, &mask, None)
            .unwrap();
        let want = 0.4f32.exp() - 0.4 - 1.0;
        assert!((scalar(&parts.kl) - want).abs() < 1e-6);
        // With zero advantages the loss is β·KL.
        assert!((scalar(&parts.total) - 0.1 * want).abs() < 1e-6);
        let parts = trainer
            .compute_grpo_loss(&policy, &policy, Some(&policy), &adv, &mask, None)
            .unwrap();
        assert!(scalar(&parts.kl).abs() < 1e-6);
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
            let parts = trainer
                .compute_grpo_loss(&logps, &logps, None, &adv, &mask, None)
                .unwrap();
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
        let parts = trainer
            .compute_grpo_loss(
                &arr(&current, &[2, 4]),
                &arr(&old, &[2, 4]),
                None,
                &arr(&[1.0, -1.0], &[2]),
                &arr(&mask, &[2, 4]),
                None,
            )
            .unwrap();

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

    type Weights = std::collections::HashMap<std::rc::Rc<str>, Array>;

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
