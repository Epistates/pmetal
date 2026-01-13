//! Group Relative Policy Optimization (GRPO) trainer.
//!
//! GRPO is a reinforcement learning algorithm that trains language models by:
//! 1. Generating multiple completions (a "group") per prompt
//! 2. Computing rewards for each completion via reward functions
//! 3. Calculating group-relative advantages (reward - group mean)
//! 4. Updating the policy to favor above-average completions
//!
//! Based on: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
//! in Open Language Models" and TRL/Unsloth implementations.
//!
//! The GRPO loss is:
//! ```text
//! L_GRPO = -E[log_pi(y|x) * A(y)] + beta * KL(pi || pi_ref)
//! ```
//!
//! Where:
//! - `A(y)` is the group-relative advantage (reward - baseline)
//! - `beta` is the KL penalty coefficient
//! - `pi` is the policy model (trainable)
//! - `pi_ref` is the reference model (frozen)

use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use pmetal_core::TrainingConfig;

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
}

/// Result type for GRPO operations.
pub type GrpoResult<T> = std::result::Result<T, GrpoError>;

/// GRPO loss type variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrpoLossType {
    /// Standard GRPO / BNPO (Base Normalized Policy Optimization).
    /// Uses group mean as baseline for advantage calculation.
    #[default]
    Bnpo,
    /// DR-GRPO (Detailed Reward GRPO).
    /// Normalizes with global constant instead of group mean.
    /// Recommended: scale_rewards=false, use_kl_loss=false
    DrGrpo,
    /// DAPO (Distribution-Aware Policy Optimization).
    /// Per-token loss aggregation to address length bias.
    /// Recommended: mask_truncated=true, epsilon_high=0.28, beta=0.0
    Dapo,
    /// Simple REINFORCE-style loss without KL.
    Reinforce,
}

/// Advantage normalization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdvantageNormalization {
    /// Normalize by group (per-prompt).
    #[default]
    Group,
    /// Normalize by batch (global).
    Batch,
    /// No normalization.
    None,
}

/// GRPO configuration.
#[derive(Debug, Clone)]
pub struct GrpoConfig {
    /// Number of completions to generate per prompt (group size).
    /// Default: 8 (from unsloth/TRL defaults)
    pub num_generations: usize,

    /// KL penalty coefficient (beta).
    /// Higher values keep policy closer to reference.
    /// Default: 0.001 (modern default, was 0.04 in older TRL)
    pub beta: f64,

    /// Entropy bonus coefficient.
    /// Encourages exploration by rewarding high-entropy outputs.
    /// Default: 0.0 (disabled)
    pub entropy_coef: f64,

    /// Loss function type.
    pub loss_type: GrpoLossType,

    /// Advantage normalization strategy.
    pub advantage_norm: AdvantageNormalization,

    /// Whether to scale rewards by their standard deviation.
    /// Default: true
    pub scale_rewards: bool,

    /// Clip ratio lower bound for importance sampling.
    /// Default: f64::NEG_INFINITY (no lower clipping)
    pub epsilon_low: f64,

    /// Clip ratio upper bound for importance sampling.
    /// Default: f64::INFINITY (no upper clipping)
    /// For DAPO, recommended: 0.28
    pub epsilon_high: f64,

    /// Whether to use KL loss term.
    /// Default: true
    pub use_kl_loss: bool,

    /// Whether to mask truncated completions in loss.
    /// Recommended for DAPO.
    /// Default: false
    pub mask_truncated_completions: bool,

    /// Temperature for generation.
    /// Default: 1.0
    pub temperature: f64,

    /// Top-p (nucleus) sampling threshold.
    /// Default: 1.0 (disabled)
    pub top_p: f64,

    /// Top-k sampling threshold.
    /// Default: 0 (disabled)
    pub top_k: usize,

    /// Maximum length for prompt tokens.
    pub max_prompt_length: usize,

    /// Maximum length for completion tokens.
    pub max_completion_length: usize,

    /// Whether to use reference-free mode (no frozen reference model).
    /// Faster but may be less stable.
    pub reference_free: bool,

    /// Minimum reward value for clipping outliers.
    pub reward_clip_min: Option<f64>,

    /// Maximum reward value for clipping outliers.
    pub reward_clip_max: Option<f64>,

    /// Whether to whiten advantages (zero mean, unit variance).
    pub whiten_advantages: bool,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            num_generations: 8,
            beta: 0.001,
            entropy_coef: 0.0,
            loss_type: GrpoLossType::Bnpo,
            advantage_norm: AdvantageNormalization::Group,
            scale_rewards: true,
            epsilon_low: f64::NEG_INFINITY,
            epsilon_high: f64::INFINITY,
            use_kl_loss: true,
            mask_truncated_completions: false,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            max_prompt_length: 512,
            max_completion_length: 512,
            reference_free: false,
            reward_clip_min: None,
            reward_clip_max: None,
            whiten_advantages: false,
        }
    }
}

impl GrpoConfig {
    /// Create a new GRPO config with the given group size.
    pub fn new(num_generations: usize) -> Self {
        Self {
            num_generations,
            ..Default::default()
        }
    }

    /// Set the KL penalty coefficient.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set the loss type.
    pub fn with_loss_type(mut self, loss_type: GrpoLossType) -> Self {
        self.loss_type = loss_type;
        self
    }

    /// Set entropy coefficient.
    pub fn with_entropy_coef(mut self, coef: f64) -> Self {
        self.entropy_coef = coef;
        self
    }

    /// Enable reference-free mode.
    pub fn reference_free(mut self) -> Self {
        self.reference_free = true;
        self
    }

    /// Configure for DAPO loss.
    pub fn for_dapo(mut self) -> Self {
        self.loss_type = GrpoLossType::Dapo;
        self.mask_truncated_completions = true;
        self.epsilon_high = 0.28;
        self.beta = 0.0;
        self
    }

    /// Configure for DR-GRPO loss.
    pub fn for_dr_grpo(mut self) -> Self {
        self.loss_type = GrpoLossType::DrGrpo;
        self.scale_rewards = false;
        self.use_kl_loss = false;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> GrpoResult<()> {
        if self.num_generations == 0 {
            return Err(GrpoError::Config(
                "num_generations must be at least 1".into(),
            ));
        }

        if self.beta < 0.0 {
            return Err(GrpoError::Config("beta must be non-negative".into()));
        }

        if self.temperature <= 0.0 {
            return Err(GrpoError::Config("temperature must be positive".into()));
        }

        if self.epsilon_low > self.epsilon_high {
            return Err(GrpoError::Config(
                "epsilon_low must be <= epsilon_high".into(),
            ));
        }

        Ok(())
    }
}

/// A single group of completions for a prompt.
#[derive(Debug, Clone)]
pub struct CompletionGroup {
    /// Prompt token IDs.
    pub prompt_ids: Vec<u32>,

    /// Optional prompt images (for multimodal models).
    /// Each group has one set of images associated with the prompt.
    pub prompt_images: Option<Vec<Array>>,

    /// Completion token IDs for each sample in the group.
    /// Shape: [num_generations, completion_length]
    pub completion_ids: Vec<Vec<u32>>,

    /// Attention mask for each completion.
    pub completion_masks: Vec<Vec<u32>>,

    /// Rewards for each completion.
    pub rewards: Vec<f64>,

    /// Whether each completion was truncated.
    pub truncated: Vec<bool>,
}

impl CompletionGroup {
    /// Create a new completion group.
    pub fn new(prompt_ids: Vec<u32>, num_generations: usize) -> Self {
        Self {
            prompt_ids,
            prompt_images: None,
            completion_ids: Vec::with_capacity(num_generations),
            completion_masks: Vec::with_capacity(num_generations),
            rewards: Vec::with_capacity(num_generations),
            truncated: Vec::with_capacity(num_generations),
        }
    }

    /// Attach images to the prompt.
    pub fn with_images(mut self, images: Vec<Array>) -> Self {
        self.prompt_images = Some(images);
        self
    }

    /// Add a completion to the group.
    pub fn add_completion(&mut self, ids: Vec<u32>, reward: f64, truncated: bool) {
        let mask = vec![1u32; ids.len()];
        self.completion_masks.push(mask);
        self.completion_ids.push(ids);
        self.rewards.push(reward);
        self.truncated.push(truncated);
    }

    /// Get the number of completions in this group.
    pub fn len(&self) -> usize {
        self.completion_ids.len()
    }

    /// Check if the group is empty.
    pub fn is_empty(&self) -> bool {
        self.completion_ids.is_empty()
    }

    /// Compute the baseline (mean reward) for this group.
    pub fn baseline(&self) -> f64 {
        if self.rewards.is_empty() {
            return 0.0;
        }
        self.rewards.iter().sum::<f64>() / self.rewards.len() as f64
    }

    /// Compute group-relative advantages.
    pub fn advantages(&self) -> Vec<f64> {
        let baseline = self.baseline();
        self.rewards.iter().map(|r| r - baseline).collect()
    }
}

/// GRPO trainer for reinforcement learning.
pub struct GrpoTrainer {
    /// GRPO configuration.
    pub config: GrpoConfig,
    /// Training configuration.
    pub training_config: TrainingConfig,
    /// Current training step.
    step: usize,
    /// Running statistics for reward normalization.
    reward_running_mean: f64,
    reward_running_var: f64,
    reward_count: usize,
}

impl GrpoTrainer {
    /// Create a new GRPO trainer.
    pub fn new(config: GrpoConfig, training_config: TrainingConfig) -> GrpoResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            training_config,
            step: 0,
            reward_running_mean: 0.0,
            reward_running_var: 1.0,
            reward_count: 0,
        })
    }

    /// Compute per-token log probabilities for a sequence.
    ///
    /// # Arguments
    /// * `logits` - Model output logits [batch, seq_len, vocab_size]
    /// * `labels` - Target labels [batch, seq_len] (-100 for ignored positions)
    ///
    /// # Returns
    /// Per-token log probabilities [batch, seq_len-1]
    pub fn compute_per_token_logps(
        &self,
        logits: &Array,
        labels: &Array,
    ) -> GrpoResult<(Array, Array)> {
        let seq_len = logits.dim(1);

        // Shift logits and labels for next-token prediction
        let pred_logits = logits.index((.., ..seq_len - 1, ..));
        let target_labels = labels.index((.., 1..));

        // Compute log softmax
        let log_probs = mlx_rs::nn::log_softmax(&pred_logits, -1)?;

        // Create mask for valid (non -100) labels
        let ignore_index = Array::from_int(-100);
        let valid_mask = target_labels.ne(&ignore_index)?;

        // Replace -100 with 0 for gathering
        let gather_labels = mlx_rs::ops::maximum(&target_labels, &Array::from_int(0))?;

        // Gather log probabilities at target positions
        let batch_size = logits.dim(0);
        let pred_len = seq_len - 1;

        let mut token_logps = Vec::with_capacity(batch_size as usize);

        for b in 0..batch_size {
            let batch_log_probs = log_probs.index(b);
            let batch_labels = gather_labels.index(b);

            let mut seq_logps = Vec::with_capacity(pred_len as usize);
            batch_log_probs.eval()?;
            batch_labels.eval()?;

            for t in 0..pred_len {
                let label = batch_labels.index(t as i32);
                label.eval()?;
                let label_idx = label.item::<i32>();
                let log_prob = batch_log_probs.index((t as i32, label_idx));
                log_prob.eval()?;
                seq_logps.push(log_prob.item::<f32>());
            }
            token_logps.push(seq_logps);
        }

        // Convert to arrays
        let flat_logps: Vec<f32> = token_logps.iter().flatten().copied().collect();
        let logps_array = Array::from_slice(&flat_logps, &[batch_size, pred_len]);

        Ok((logps_array, valid_mask.as_dtype(mlx_rs::Dtype::Float32)?))
    }

    /// Compute summed log probabilities for each sequence.
    ///
    /// # Arguments
    /// * `per_token_logps` - Per-token log probs [batch, seq_len]
    /// * `mask` - Valid token mask [batch, seq_len]
    ///
    /// # Returns
    /// Summed log probabilities [batch]
    pub fn sum_log_probs(&self, per_token_logps: &Array, mask: &Array) -> GrpoResult<Array> {
        // Mask out invalid positions and sum
        let masked_logps = per_token_logps.multiply(mask)?;
        Ok(masked_logps.sum_axis(-1, None)?)
    }

    /// Compute advantages from rewards.
    ///
    /// # Arguments
    /// * `rewards` - Rewards for each completion [batch]
    /// * `group_size` - Number of completions per prompt
    ///
    /// # Returns
    /// Advantages [batch]
    pub fn compute_advantages(&self, rewards: &[f64], group_size: usize) -> GrpoResult<Vec<f64>> {
        let num_groups = rewards.len() / group_size;
        let mut advantages = vec![0.0; rewards.len()];

        match self.config.advantage_norm {
            AdvantageNormalization::Group => {
                // Group-relative advantages
                for g in 0..num_groups {
                    let start = g * group_size;
                    let end = start + group_size;
                    let group_rewards = &rewards[start..end];

                    // Compute group baseline (mean)
                    let baseline: f64 = group_rewards.iter().sum::<f64>() / group_size as f64;

                    // Compute group std for optional whitening
                    let group_std = if self.config.whiten_advantages {
                        let var: f64 = group_rewards
                            .iter()
                            .map(|r| (r - baseline).powi(2))
                            .sum::<f64>()
                            / group_size as f64;
                        var.sqrt().max(1e-8)
                    } else {
                        1.0
                    };

                    // Compute advantages
                    for (i, &reward) in group_rewards.iter().enumerate() {
                        advantages[start + i] = (reward - baseline) / group_std;
                    }
                }
            }
            AdvantageNormalization::Batch => {
                // Batch-level normalization
                let mean: f64 = rewards.iter().sum::<f64>() / rewards.len() as f64;
                let std = if self.config.whiten_advantages {
                    let var: f64 = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                        / rewards.len() as f64;
                    var.sqrt().max(1e-8)
                } else {
                    1.0
                };

                for (i, &reward) in rewards.iter().enumerate() {
                    advantages[i] = (reward - mean) / std;
                }
            }
            AdvantageNormalization::None => {
                // No normalization - use rewards directly
                advantages.copy_from_slice(rewards);
            }
        }

        Ok(advantages)
    }

    /// Compute GRPO loss for a batch.
    ///
    /// # Arguments
    /// * `policy_logps` - Log probs from policy model [batch]
    /// * `ref_logps` - Log probs from reference model [batch]
    /// * `advantages` - Computed advantages [batch]
    /// * `old_logps` - Optional old policy log probs for importance sampling [batch]
    ///
    /// # Returns
    /// (loss, kl_divergence, policy_loss)
    pub fn compute_grpo_loss(
        &self,
        policy_logps: &Array,
        ref_logps: &Array,
        advantages: &Array,
        old_logps: Option<&Array>,
    ) -> GrpoResult<(Array, Array, Array)> {
        // Compute log ratios for KL
        let log_ratio = if self.config.reference_free {
            policy_logps.clone()
        } else {
            policy_logps.subtract(ref_logps)?
        };

        // KL divergence approximation: (ratio - 1) - log_ratio
        // where ratio = exp(log_ratio)
        let ratio = log_ratio.exp()?;
        let one = Array::from_f32(1.0);
        let kl = ratio.subtract(&one)?.subtract(&log_ratio)?;
        let mean_kl = kl.mean(None)?;

        // Compute policy loss based on loss type
        let policy_loss = match self.config.loss_type {
            GrpoLossType::Bnpo | GrpoLossType::DrGrpo => {
                // Standard policy gradient: -log_prob * advantage
                let neg_logps = policy_logps.negative()?;
                neg_logps.multiply(advantages)?
            }
            GrpoLossType::Dapo => {
                // DAPO with importance sampling clipping
                if let Some(old) = old_logps {
                    let importance_ratio = policy_logps.subtract(old)?.exp()?;

                    // Clip importance ratio using min/max
                    let eps_low = Array::from_f32(self.config.epsilon_low as f32);
                    let eps_high = Array::from_f32(self.config.epsilon_high as f32);
                    let clipped_low = mlx_rs::ops::maximum(&importance_ratio, &eps_low)?;
                    let clipped_ratio = mlx_rs::ops::minimum(&clipped_low, &eps_high)?;

                    // Clipped objective
                    let obj1 = importance_ratio.multiply(advantages)?;
                    let obj2 = clipped_ratio.multiply(advantages)?;
                    let min_obj = mlx_rs::ops::minimum(&obj1, &obj2)?;
                    min_obj.negative()?
                } else {
                    // Fall back to standard if no old logps
                    let neg_logps = policy_logps.negative()?;
                    neg_logps.multiply(advantages)?
                }
            }
            GrpoLossType::Reinforce => {
                // Simple REINFORCE without KL
                let neg_logps = policy_logps.negative()?;
                neg_logps.multiply(advantages)?
            }
        };

        // Compute total loss
        let mean_policy_loss = policy_loss.mean(None)?;
        let total_loss = if self.config.use_kl_loss && self.config.beta > 0.0 {
            let beta = Array::from_f32(self.config.beta as f32);
            mean_policy_loss.add(&mean_kl.multiply(&beta)?)?
        } else {
            mean_policy_loss.clone()
        };

        Ok((total_loss, mean_kl, mean_policy_loss))
    }

    /// Compute per-token GRPO loss (for DAPO-style aggregation).
    ///
    /// # Arguments
    /// * `per_token_policy_logps` - Per-token log probs from policy [batch, seq]
    /// * `per_token_ref_logps` - Per-token log probs from reference [batch, seq]
    /// * `advantages` - Advantages per sequence [batch]
    /// * `mask` - Valid token mask [batch, seq]
    ///
    /// # Returns
    /// (loss, kl_divergence)
    pub fn compute_per_token_loss(
        &self,
        per_token_policy_logps: &Array,
        per_token_ref_logps: &Array,
        advantages: &Array,
        mask: &Array,
    ) -> GrpoResult<(Array, Array)> {
        // Per-token KL
        let log_ratio = per_token_policy_logps.subtract(per_token_ref_logps)?;
        let ratio = log_ratio.exp()?;
        let one = Array::from_f32(1.0);
        let per_token_kl = ratio.subtract(&one)?.subtract(&log_ratio)?;

        // Mask and average KL
        let masked_kl = per_token_kl.multiply(mask)?;
        let token_count = mask.sum(None)?;
        let mean_kl = masked_kl.sum(None)?.divide(&token_count)?;

        // Per-token policy loss with advantage broadcasting
        // advantages [batch] -> [batch, 1] for broadcasting
        let advantages_expanded = advantages.reshape(&[advantages.dim(0), 1])?;
        let neg_logps = per_token_policy_logps.negative()?;
        let per_token_loss = neg_logps.multiply(&advantages_expanded)?;

        // Mask and average
        let masked_loss = per_token_loss.multiply(mask)?;
        let mean_loss = masked_loss.sum(None)?.divide(&token_count)?;

        // Total loss with KL penalty
        let total_loss = if self.config.use_kl_loss && self.config.beta > 0.0 {
            let beta = Array::from_f32(self.config.beta as f32);
            mean_loss.add(&mean_kl.multiply(&beta)?)?
        } else {
            mean_loss
        };

        Ok((total_loss, mean_kl))
    }

    /// Compute entropy bonus from log probabilities.
    ///
    /// # Arguments
    /// * `logits` - Model output logits [batch, seq, vocab]
    /// * `mask` - Valid token mask [batch, seq]
    ///
    /// # Returns
    /// Mean entropy
    pub fn compute_entropy(&self, logits: &Array, mask: &Array) -> GrpoResult<Array> {
        // entropy = -sum(p * log_p)
        let log_probs = mlx_rs::nn::log_softmax(logits, -1)?;
        let probs = log_probs.exp()?; // softmax = exp(log_softmax)
        let neg_entropy = probs.multiply(&log_probs)?.sum_axis(-1, None)?;
        let entropy = neg_entropy.negative()?;

        // Mask and average
        let masked_entropy = entropy.multiply(mask)?;
        let token_count = mask.sum(None)?;
        Ok(masked_entropy.sum(None)?.divide(&token_count)?)
    }

    /// Update running reward statistics for normalization.
    pub fn update_reward_stats(&mut self, rewards: &[f64]) {
        for &reward in rewards {
            self.reward_count += 1;
            let delta = reward - self.reward_running_mean;
            self.reward_running_mean += delta / self.reward_count as f64;
            let delta2 = reward - self.reward_running_mean;
            self.reward_running_var +=
                (delta * delta2 - self.reward_running_var) / self.reward_count as f64;
        }
    }

    /// Normalize rewards using running statistics.
    pub fn normalize_rewards(&self, rewards: &mut [f64]) {
        if !self.config.scale_rewards || self.reward_count < 2 {
            return;
        }

        let std = self.reward_running_var.sqrt().max(1e-8);
        for reward in rewards.iter_mut() {
            *reward = (*reward - self.reward_running_mean) / std;
        }
    }

    /// Clip rewards to configured bounds.
    pub fn clip_rewards(&self, rewards: &mut [f64]) {
        for reward in rewards.iter_mut() {
            if let Some(min) = self.config.reward_clip_min {
                *reward = reward.max(min);
            }
            if let Some(max) = self.config.reward_clip_max {
                *reward = reward.min(max);
            }
        }
    }

    /// Get current training step.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Increment step counter.
    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    /// Process a batch of completion groups for training.
    ///
    /// # Arguments
    /// * `groups` - Completion groups with rewards
    ///
    /// # Returns
    /// Flattened batch data (prompt_ids, completion_ids, advantages, masks, images)
    pub fn prepare_batch(
        &mut self,
        groups: &[CompletionGroup],
    ) -> GrpoResult<(
        Vec<Vec<u32>>,
        Vec<Vec<u32>>,
        Vec<f64>,
        Vec<Vec<u32>>,
        Option<Vec<Vec<Array>>>, // Added images
    )> {
        // Collect all rewards and compute advantages
        let mut all_rewards: Vec<f64> = Vec::new();
        for group in groups {
            all_rewards.extend(&group.rewards);
        }

        // Clip and normalize rewards
        self.clip_rewards(&mut all_rewards);
        self.update_reward_stats(&all_rewards);
        self.normalize_rewards(&mut all_rewards);

        // Compute advantages
        let advantages = self.compute_advantages(&all_rewards, self.config.num_generations)?;

        // Flatten completion data
        let mut all_prompts = Vec::new();
        let mut all_completions = Vec::new();
        let mut all_masks = Vec::new();
        let mut all_images = Vec::new();
        let mut has_images = false;

        for group in groups {
            // Check if this group has images
            if let Some(imgs) = &group.prompt_images {
                has_images = true;
                // Repeat images for each completion in the group?
                // Usually training batch is flattened as [batch_size * num_generations].
                // So yes, we need to duplicate the images reference or structure for each sample.
                // However, images are large. It's better to keep them per prompt and handle broadcasting in the model forward.
                // But standard trainer expects flattened inputs.
                // Let's store them per completion row to be safe for now, referencing the group's images.
                // Cloning Array is cheap (shared reference).
                for _ in &group.completion_ids {
                    all_images.push(imgs.clone());
                }
            } else if has_images {
                // specific case where some have images and some don't?
                // For now assume all or nothing for simplicity, or handle empty vec
                for _ in &group.completion_ids {
                    all_images.push(Vec::new());
                }
            }

            for completion_ids in &group.completion_ids {
                all_prompts.push(group.prompt_ids.clone());
                all_completions.push(completion_ids.clone());
            }
            all_masks.extend(group.completion_masks.clone());
        }

        let images_out = if has_images { Some(all_images) } else { None };

        Ok((
            all_prompts,
            all_completions,
            advantages,
            all_masks,
            images_out,
        ))
    }
}

/// GRPO training metrics for logging.
#[derive(Debug, Clone, Default)]
pub struct GrpoMetrics {
    /// Total loss value.
    pub loss: f32,
    /// Policy gradient loss component.
    pub policy_loss: f32,
    /// KL divergence from reference.
    pub kl_divergence: f32,
    /// Entropy bonus (if used).
    pub entropy: f32,
    /// Mean reward across batch.
    pub mean_reward: f32,
    /// Reward standard deviation.
    pub reward_std: f32,
    /// Mean advantage.
    pub mean_advantage: f32,
    /// Mean completion length.
    pub completion_length: f32,
    /// Fraction of positive advantages.
    pub advantage_positive_frac: f32,
}

impl GrpoMetrics {
    /// Compute metrics from training data.
    pub fn compute(
        loss: f32,
        policy_loss: f32,
        kl: f32,
        entropy: f32,
        rewards: &[f64],
        advantages: &[f64],
        completion_lengths: &[usize],
    ) -> Self {
        let n = rewards.len() as f32;

        let mean_reward = rewards.iter().sum::<f64>() as f32 / n;
        let reward_var: f32 = rewards
            .iter()
            .map(|&r| (r as f32 - mean_reward).powi(2))
            .sum::<f32>()
            / n;
        let reward_std = reward_var.sqrt();

        let mean_advantage = advantages.iter().sum::<f64>() as f32 / n;
        let positive_count = advantages.iter().filter(|&&a| a > 0.0).count();
        let advantage_positive_frac = positive_count as f32 / n;

        let completion_length =
            completion_lengths.iter().sum::<usize>() as f32 / completion_lengths.len() as f32;

        Self {
            loss,
            policy_loss,
            kl_divergence: kl,
            entropy,
            mean_reward,
            reward_std,
            mean_advantage,
            completion_length,
            advantage_positive_frac,
        }
    }
}

/// Reward function trait for GRPO.
pub trait RewardFunction: Send + Sync {
    /// Compute rewards for a batch of completions.
    ///
    /// # Arguments
    /// * `prompts` - Prompt texts
    /// * `completions` - Completion texts
    /// * `images` - Optional images for each prompt [batch_size, num_images_per_prompt]
    ///
    /// # Returns
    /// Rewards for each completion
    fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>>;

    /// Name of this reward function for logging.
    fn name(&self) -> &str;
}

/// Combined reward from multiple reward functions.
pub struct CombinedReward {
    /// Individual reward functions with weights.
    pub functions: Vec<(Box<dyn RewardFunction>, f64)>,
}

impl CombinedReward {
    /// Create a new combined reward.
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    /// Add a reward function with weight.
    pub fn add(mut self, func: Box<dyn RewardFunction>, weight: f64) -> Self {
        self.functions.push((func, weight));
        self
    }

    /// Compute combined rewards.
    pub fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        if self.functions.is_empty() {
            return Err(GrpoError::Reward("No reward functions configured".into()));
        }

        let n = completions.len();
        let mut combined = vec![0.0; n];

        for (func, weight) in &self.functions {
            let rewards = func.compute(prompts, completions, images)?;
            for (i, &r) in rewards.iter().enumerate() {
                combined[i] += r * weight;
            }
        }

        Ok(combined)
    }
}

impl Default for CombinedReward {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpo_config_default() {
        let config = GrpoConfig::default();
        assert_eq!(config.num_generations, 8);
        assert_eq!(config.beta, 0.001);
        assert_eq!(config.loss_type, GrpoLossType::Bnpo);
        assert!(!config.reference_free);
    }

    #[test]
    fn test_grpo_config_validation() {
        let config = GrpoConfig::new(4);
        assert!(config.validate().is_ok());

        let invalid = GrpoConfig {
            num_generations: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid_beta = GrpoConfig {
            beta: -1.0,
            ..Default::default()
        };
        assert!(invalid_beta.validate().is_err());
    }

    #[test]
    fn test_grpo_config_dapo() {
        let config = GrpoConfig::new(8).for_dapo();
        assert_eq!(config.loss_type, GrpoLossType::Dapo);
        assert!(config.mask_truncated_completions);
        assert!((config.epsilon_high - 0.28).abs() < 0.01);
        assert_eq!(config.beta, 0.0);
    }

    #[test]
    fn test_completion_group() {
        let mut group = CompletionGroup::new(vec![1, 2, 3], 4);
        group.add_completion(vec![4, 5], 1.0, false);
        group.add_completion(vec![6, 7, 8], 2.0, false);
        group.add_completion(vec![9], 0.5, true);
        group.add_completion(vec![10, 11], 1.5, false);

        assert_eq!(group.len(), 4);
        assert!(!group.is_empty());
        assert!((group.baseline() - 1.25).abs() < 0.01);

        let advantages = group.advantages();
        assert_eq!(advantages.len(), 4);
        assert!((advantages[0] - (-0.25)).abs() < 0.01); // 1.0 - 1.25
        assert!((advantages[1] - 0.75).abs() < 0.01); // 2.0 - 1.25
    }

    #[test]
    fn test_compute_advantages_group() {
        let config = GrpoConfig::new(2);
        let training_config = TrainingConfig::default();
        let trainer = GrpoTrainer::new(config, training_config).unwrap();

        // Two groups of 2
        let rewards = vec![1.0, 3.0, 2.0, 4.0];
        let advantages = trainer.compute_advantages(&rewards, 2).unwrap();

        // Group 1: baseline = 2.0, advantages = [-1, 1]
        // Group 2: baseline = 3.0, advantages = [-1, 1]
        assert!((advantages[0] - (-1.0)).abs() < 0.01);
        assert!((advantages[1] - 1.0).abs() < 0.01);
        assert!((advantages[2] - (-1.0)).abs() < 0.01);
        assert!((advantages[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_advantages_batch() {
        let config = GrpoConfig {
            advantage_norm: AdvantageNormalization::Batch,
            ..GrpoConfig::new(2)
        };
        let training_config = TrainingConfig::default();
        let trainer = GrpoTrainer::new(config, training_config).unwrap();

        let rewards = vec![1.0, 2.0, 3.0, 4.0];
        let advantages = trainer.compute_advantages(&rewards, 2).unwrap();

        // Batch mean = 2.5
        assert!((advantages[0] - (-1.5)).abs() < 0.01);
        assert!((advantages[1] - (-0.5)).abs() < 0.01);
        assert!((advantages[2] - 0.5).abs() < 0.01);
        assert!((advantages[3] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_grpo_loss_computation() {
        let config = GrpoConfig::new(4).with_beta(0.1);
        let training_config = TrainingConfig::default();
        let trainer = GrpoTrainer::new(config, training_config).unwrap();

        // Mock log probabilities
        let policy_logps = Array::from_slice(&[-1.0f32, -2.0, -1.5, -1.8], &[4]);
        let ref_logps = Array::from_slice(&[-1.1f32, -2.1, -1.6, -1.9], &[4]);
        let advantages = Array::from_slice(&[-1.0f32, 1.0, 0.5, -0.5], &[4]);

        let (loss, kl, policy_loss) = trainer
            .compute_grpo_loss(&policy_logps, &ref_logps, &advantages, None)
            .unwrap();

        loss.eval().unwrap();
        kl.eval().unwrap();
        policy_loss.eval().unwrap();

        // Loss should be finite
        assert!(loss.item::<f32>().is_finite());
        assert!(kl.item::<f32>() >= 0.0); // KL is non-negative
    }

    #[test]
    fn test_reward_stats() {
        let config = GrpoConfig::new(4);
        let training_config = TrainingConfig::default();
        let mut trainer = GrpoTrainer::new(config, training_config).unwrap();

        trainer.update_reward_stats(&[1.0, 2.0, 3.0, 4.0]);
        assert!((trainer.reward_running_mean - 2.5).abs() < 0.01);

        trainer.update_reward_stats(&[5.0, 6.0, 7.0, 8.0]);
        assert!((trainer.reward_running_mean - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_clip_rewards() {
        let config = GrpoConfig {
            reward_clip_min: Some(-1.0),
            reward_clip_max: Some(1.0),
            ..GrpoConfig::new(4)
        };
        let training_config = TrainingConfig::default();
        let trainer = GrpoTrainer::new(config, training_config).unwrap();

        let mut rewards = vec![-2.0, -0.5, 0.5, 2.0];
        trainer.clip_rewards(&mut rewards);

        assert_eq!(rewards, vec![-1.0, -0.5, 0.5, 1.0]);
    }

    #[test]
    fn test_grpo_metrics() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0];
        let advantages = vec![-1.5, -0.5, 0.5, 1.5];
        let lengths = vec![10, 15, 12, 8];

        let metrics = GrpoMetrics::compute(0.5, 0.4, 0.01, 0.1, &rewards, &advantages, &lengths);

        assert!((metrics.mean_reward - 2.5).abs() < 0.01);
        assert!((metrics.mean_advantage - 0.0).abs() < 0.01);
        assert!((metrics.advantage_positive_frac - 0.5).abs() < 0.01);
        assert!((metrics.completion_length - 11.25).abs() < 0.01);
    }

    #[test]
    fn test_prepare_batch() {
        // Disable reward scaling to test raw advantages
        let config = GrpoConfig {
            scale_rewards: false,
            ..GrpoConfig::new(2)
        };
        let training_config = TrainingConfig::default();
        let mut trainer = GrpoTrainer::new(config, training_config).unwrap();

        let mut group1 = CompletionGroup::new(vec![1, 2], 2);
        group1.add_completion(vec![3, 4], 1.0, false);
        group1.add_completion(vec![5, 6], 2.0, false);

        let mut group2 = CompletionGroup::new(vec![7, 8], 2);
        group2.add_completion(vec![9, 10], 1.5, false);
        group2.add_completion(vec![11, 12], 0.5, false);

        let (prompts, completions, advantages, masks, _) =
            trainer.prepare_batch(&[group1, group2]).unwrap();

        assert_eq!(prompts.len(), 4);
        assert_eq!(completions.len(), 4);
        assert_eq!(advantages.len(), 4);
        assert_eq!(masks.len(), 4);

        // Advantages should be group-normalized
        // Group 1: mean=1.5, adv=[-0.5, 0.5]
        // Group 2: mean=1.0, adv=[0.5, -0.5]
        assert!((advantages[0] - (-0.5)).abs() < 0.1);
        assert!((advantages[1] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_combined_reward() {
        struct ConstantReward(f64);
        impl RewardFunction for ConstantReward {
            fn compute(
                &self,
                _: &[String],
                completions: &[String],
                _: Option<&[Vec<Array>]>,
            ) -> GrpoResult<Vec<f64>> {
                Ok(vec![self.0; completions.len()])
            }
            fn name(&self) -> &str {
                "constant"
            }
        }

        let combined = CombinedReward::new()
            .add(Box::new(ConstantReward(1.0)), 0.5)
            .add(Box::new(ConstantReward(2.0)), 0.5);

        let rewards = combined
            .compute(&["p".into()], &["c".into()], None)
            .unwrap();

        assert_eq!(rewards.len(), 1);
        assert!((rewards[0] - 1.5).abs() < 0.01); // 0.5*1 + 0.5*2
    }
}
