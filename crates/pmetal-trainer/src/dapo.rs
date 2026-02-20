//! DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization.
//!
//! Full implementation of ByteDance's DAPO algorithm from "DAPO: An Open-Source LLM
//! Reinforcement Learning System at Scale" (2025).
//!
//! DAPO addresses key instabilities in GRPO through four innovations:
//!
//! 1. **Clip-Higher**: Only clips the upper bound of importance ratio, not lower.
//!    This prevents the "entropy collapse" problem where diversity disappears.
//!
//! 2. **Dynamic Sampling**: Filters prompts based on accuracy. If all completions
//!    for a prompt have the same reward (all correct or all wrong), skip it.
//!
//! 3. **Token-Level Policy Gradient**: Aggregates loss at token level rather than
//!    sequence level to address length bias.
//!
//! 4. **Overlong Reward Penalty**: Penalizes sequences that hit max length without
//!    completing properly (truncated completions).
//!
//! # Usage
//!
//! ```rust,ignore
//! use pmetal_trainer::{DapoConfig, DapoTrainer};
//!
//! let config = DapoConfig::default()
//!     .with_clip_eps_high(0.28)
//!     .with_dynamic_sampling(true)
//!     .with_overlong_penalty(-1.0);
//!
//! let trainer = DapoTrainer::new(config)?;
//! ```

use mlx_rs::{error::Exception, Array};
use std::collections::HashMap;

/// Error type for DAPO training.
#[derive(Debug, thiserror::Error)]
pub enum DapoError {
    /// MLX computation error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
    /// Generation error.
    #[error("Generation error: {0}")]
    Generation(String),
}

/// Result type for DAPO operations.
pub type DapoResult<T> = std::result::Result<T, DapoError>;

/// DAPO configuration.
///
/// Implements all hyperparameters from the DAPO paper with recommended defaults.
#[derive(Debug, Clone)]
pub struct DapoConfig {
    /// Number of completions per prompt (group size).
    /// Default: 16 (DAPO paper recommends 16-64)
    pub num_generations: usize,

    /// KL penalty coefficient (beta).
    /// DAPO paper recommends beta=0 (no KL penalty) for maximum exploration.
    /// Default: 0.0
    pub beta: f64,

    /// Upper clip bound for importance ratio (epsilon_high).
    /// Only upper clipping is applied (Clip-Higher).
    /// Default: 0.28 (DAPO paper default)
    pub clip_eps_high: f64,

    /// Lower clip bound is effectively disabled in DAPO (Clip-Higher strategy).
    /// True DAPO semantics require 0.0 here so only the upper bound is enforced.
    /// Default: 0.0
    pub clip_eps_low: f64,

    /// Enable dynamic sampling - skip prompts where all completions have same reward.
    /// This is a key DAPO innovation that prevents gradient noise.
    /// Default: true
    pub dynamic_sampling: bool,

    /// Minimum accuracy threshold for prompt filtering in dynamic sampling.
    /// Prompts with accuracy < min_accuracy or > (1 - min_accuracy) are skipped.
    /// Default: 0.01 (skip if 0% or 100% accuracy)
    pub dynamic_sampling_min_accuracy: f64,

    /// Penalty added to rewards for truncated (overlong) completions.
    /// Negative values penalize hitting max length without proper completion.
    /// Default: -1.0
    pub overlong_penalty: f64,

    /// Whether to apply overlong penalty.
    /// Default: true
    pub use_overlong_penalty: bool,

    /// Temperature for generation.
    /// Default: 1.0
    pub temperature: f64,

    /// Top-p (nucleus) sampling.
    /// Default: 0.95
    pub top_p: f64,

    /// Maximum prompt length.
    pub max_prompt_length: usize,

    /// Maximum completion length.
    pub max_completion_length: usize,

    /// Whether to use token-level loss aggregation (DAPO default).
    /// If false, uses sequence-level aggregation like standard GRPO.
    /// Default: true
    pub token_level_loss: bool,

    /// Entropy bonus coefficient for exploration.
    /// Default: 0.001
    pub entropy_coef: f64,

    /// Whether to normalize advantages within each group.
    /// Default: true (standard practice)
    pub normalize_advantages: bool,

    /// Minimum group size after dynamic sampling to proceed with update.
    /// Default: 2
    pub min_group_size: usize,

    /// Reward threshold used when computing group accuracy.
    ///
    /// A completion is counted as "correct" when its reward strictly exceeds
    /// this value.  Set to `0.0` (default) to count any positive reward as
    /// correct, or to a higher value for stricter binary scoring.
    /// Default: 0.0
    pub accuracy_reward_threshold: f64,
}

impl Default for DapoConfig {
    fn default() -> Self {
        Self {
            num_generations: 16,
            beta: 0.0, // DAPO disables KL
            clip_eps_high: 0.28,
            clip_eps_low: 0.0, // True DAPO: Clip-Higher only (no lower clipping)
            dynamic_sampling: true,
            dynamic_sampling_min_accuracy: 0.01,
            overlong_penalty: -1.0,
            use_overlong_penalty: true,
            temperature: 1.0,
            top_p: 0.95,
            max_prompt_length: 512,
            max_completion_length: 512,
            token_level_loss: true, // DAPO innovation
            entropy_coef: 0.001,
            normalize_advantages: true,
            min_group_size: 2,
            accuracy_reward_threshold: 0.0,
        }
    }
}

impl DapoConfig {
    /// Create a new DAPO config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the clip epsilon high bound.
    pub fn with_clip_eps_high(mut self, eps: f64) -> Self {
        self.clip_eps_high = eps;
        self
    }

    /// Set the clip epsilon low bound.
    pub fn with_clip_eps_low(mut self, eps: f64) -> Self {
        self.clip_eps_low = eps;
        self
    }

    /// Enable/disable dynamic sampling.
    pub fn with_dynamic_sampling(mut self, enabled: bool) -> Self {
        self.dynamic_sampling = enabled;
        self
    }

    /// Set overlong penalty.
    pub fn with_overlong_penalty(mut self, penalty: f64) -> Self {
        self.overlong_penalty = penalty;
        self.use_overlong_penalty = true;
        self
    }

    /// Set number of generations per prompt.
    pub fn with_num_generations(mut self, n: usize) -> Self {
        self.num_generations = n;
        self
    }

    /// Set beta (KL coefficient).
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> DapoResult<()> {
        if self.num_generations == 0 {
            return Err(DapoError::Config(
                "num_generations must be at least 1".into(),
            ));
        }
        if self.clip_eps_high <= 0.0 {
            return Err(DapoError::Config("clip_eps_high must be positive".into()));
        }
        if self.temperature <= 0.0 {
            return Err(DapoError::Config("temperature must be positive".into()));
        }
        if self.min_group_size == 0 {
            return Err(DapoError::Config(
                "min_group_size must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

/// A single prompt with its completions for DAPO training.
#[derive(Debug, Clone)]
pub struct DapoPromptGroup {
    /// Prompt token IDs.
    pub prompt_ids: Vec<u32>,

    /// Completions for this prompt.
    pub completions: Vec<DapoCompletion>,

    /// Computed group accuracy (fraction of correct completions).
    pub accuracy: f64,
}

/// A single completion within a DAPO group.
#[derive(Debug, Clone)]
pub struct DapoCompletion {
    /// Completion token IDs.
    pub token_ids: Vec<u32>,

    /// Raw reward (before overlong penalty).
    pub raw_reward: f64,

    /// Final reward (after overlong penalty if applicable).
    pub reward: f64,

    /// Whether this completion was truncated (hit max length).
    pub truncated: bool,

    /// Per-token log probabilities from current policy.
    pub policy_logps: Option<Vec<f32>>,

    /// Per-token log probabilities from old policy (for importance sampling).
    pub old_policy_logps: Option<Vec<f32>>,

    /// Computed advantage for this completion.
    pub advantage: f64,
}

impl DapoCompletion {
    /// Create a new completion.
    pub fn new(token_ids: Vec<u32>, reward: f64, truncated: bool) -> Self {
        Self {
            token_ids,
            raw_reward: reward,
            reward,
            truncated,
            policy_logps: None,
            old_policy_logps: None,
            advantage: 0.0,
        }
    }

    /// Get completion length.
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

impl DapoPromptGroup {
    /// Create a new prompt group.
    pub fn new(prompt_ids: Vec<u32>) -> Self {
        Self {
            prompt_ids,
            completions: Vec::new(),
            accuracy: 0.0,
        }
    }

    /// Add a completion to this group.
    ///
    /// `accuracy_threshold` should come from `DapoConfig::accuracy_reward_threshold`.
    pub fn add_completion(&mut self, completion: DapoCompletion, accuracy_threshold: f64) {
        self.completions.push(completion);
        self.update_accuracy(accuracy_threshold);
    }

    /// Update computed accuracy using the given reward threshold.
    ///
    /// A completion is counted as correct when `reward > threshold`.
    /// Use `DapoConfig::accuracy_reward_threshold` as the threshold value.
    pub fn update_accuracy(&mut self, threshold: f64) {
        if self.completions.is_empty() {
            self.accuracy = 0.0;
            return;
        }
        let correct = self
            .completions
            .iter()
            .filter(|c| c.reward > threshold)
            .count();
        self.accuracy = correct as f64 / self.completions.len() as f64;
    }

    /// Get group size.
    pub fn len(&self) -> usize {
        self.completions.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.completions.is_empty()
    }

    /// Get mean reward for this group.
    pub fn mean_reward(&self) -> f64 {
        if self.completions.is_empty() {
            return 0.0;
        }
        self.completions.iter().map(|c| c.reward).sum::<f64>() / self.completions.len() as f64
    }

    /// Get reward standard deviation.
    pub fn reward_std(&self) -> f64 {
        if self.completions.len() < 2 {
            return 1.0;
        }
        let mean = self.mean_reward();
        let var: f64 = self
            .completions
            .iter()
            .map(|c| (c.reward - mean).powi(2))
            .sum::<f64>()
            / self.completions.len() as f64;
        var.sqrt().max(1e-8)
    }
}

/// DAPO Trainer.
///
/// Implements the full DAPO algorithm with all four innovations.
pub struct DapoTrainer {
    /// Configuration.
    pub config: DapoConfig,
    /// Current training step.
    step: usize,
    /// Statistics tracking.
    stats: DapoStats,
}

/// Statistics for DAPO training.
#[derive(Debug, Clone, Default)]
pub struct DapoStats {
    /// Total prompts processed.
    pub total_prompts: usize,
    /// Prompts skipped by dynamic sampling.
    pub skipped_prompts: usize,
    /// Total completions processed.
    pub total_completions: usize,
    /// Truncated completions count.
    pub truncated_completions: usize,
    /// Mean reward across all completions.
    pub mean_reward: f64,
    /// Mean advantage.
    pub mean_advantage: f64,
    /// Mean policy loss.
    pub mean_policy_loss: f64,
    /// Mean KL divergence.
    pub mean_kl: f64,
    /// Mean entropy.
    pub mean_entropy: f64,
    /// Mean clip fraction (how often clipping was applied).
    pub clip_fraction: f64,
}

impl DapoTrainer {
    /// Create a new DAPO trainer.
    pub fn new(config: DapoConfig) -> DapoResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            step: 0,
            stats: DapoStats::default(),
        })
    }

    /// Apply overlong penalty to truncated completions.
    pub fn apply_overlong_penalty(&self, groups: &mut [DapoPromptGroup]) {
        if !self.config.use_overlong_penalty {
            return;
        }

        for group in groups {
            for completion in &mut group.completions {
                if completion.truncated {
                    completion.reward = completion.raw_reward + self.config.overlong_penalty;
                }
            }
            // Update group accuracy after penalty
            group.update_accuracy(self.config.accuracy_reward_threshold);
        }
    }

    /// Filter groups using dynamic sampling.
    ///
    /// Returns indices of groups to keep (those with mixed accuracy).
    pub fn dynamic_sample(&self, groups: &[DapoPromptGroup]) -> Vec<usize> {
        if !self.config.dynamic_sampling {
            return (0..groups.len()).collect();
        }

        let min_acc = self.config.dynamic_sampling_min_accuracy;
        let max_acc = 1.0 - min_acc;

        groups
            .iter()
            .enumerate()
            .filter(|(_, g)| {
                // Keep groups with mixed accuracy (not all correct or all wrong)
                g.accuracy > min_acc && g.accuracy < max_acc
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute advantages for all completions using group-relative baseline.
    pub fn compute_advantages(&self, groups: &mut [DapoPromptGroup]) {
        for group in groups {
            let baseline = group.mean_reward();
            let std = if self.config.normalize_advantages {
                group.reward_std()
            } else {
                1.0
            };

            for completion in &mut group.completions {
                completion.advantage = (completion.reward - baseline) / std;
            }
        }
    }

    /// Compute token-level DAPO loss with Clip-Higher.
    ///
    /// # Arguments
    /// * `policy_logps` - Per-token log probs from current policy [batch, seq]
    /// * `old_policy_logps` - Per-token log probs from old policy [batch, seq]
    /// * `advantages` - Per-sequence advantages [batch]
    /// * `mask` - Valid token mask [batch, seq]
    ///
    /// # Returns
    /// (loss, clip_fraction)
    pub fn compute_token_level_loss(
        &self,
        policy_logps: &Array,
        old_policy_logps: &Array,
        advantages: &Array,
        mask: &Array,
    ) -> DapoResult<(Array, f32)> {
        // Importance ratio: exp(policy_logps - old_policy_logps)
        let log_ratio = policy_logps.subtract(old_policy_logps)?;
        let ratio = log_ratio.exp()?;

        // Clip-Higher: only clip upper bound
        // clipped_ratio = min(ratio, 1 + eps_high)
        // For lower bound, we use max(ratio, eps_low) only for numerical stability
        let eps_high = 1.0 + self.config.clip_eps_high;
        let eps_low = self.config.clip_eps_low;

        let ratio_clipped_high = mlx_rs::ops::minimum(&ratio, &Array::from_f32(eps_high as f32))?;
        let ratio_clipped =
            mlx_rs::ops::maximum(&ratio_clipped_high, &Array::from_f32(eps_low as f32))?;

        // Compute clip fraction for logging
        let is_clipped = ratio
            .gt(&Array::from_f32(eps_high as f32))?
            .as_dtype(mlx_rs::Dtype::Float32)?;
        let clip_fraction = is_clipped
            .multiply(mask)?
            .sum(None)?
            .divide(&mask.sum(None)?)?;
        clip_fraction.eval()?;
        let clip_frac = clip_fraction.item::<f32>();

        // Expand advantages for broadcasting: [batch] -> [batch, 1]
        let adv_expanded = advantages.reshape(&[advantages.dim(0), 1])?;

        // Token-level loss: -clipped_ratio * advantage
        let token_loss = ratio_clipped.multiply(&adv_expanded)?.negative()?;

        // Masked mean over all tokens, guarded against division by zero
        let masked_loss = token_loss.multiply(mask)?;
        let total_tokens = mask.sum(None)?;
        let safe_count = mlx_rs::ops::maximum(&total_tokens, &Array::from_f32(1.0))?;
        let mean_loss = masked_loss.sum(None)?.divide(&safe_count)?;

        Ok((mean_loss, clip_frac))
    }

    /// Compute sequence-level DAPO loss (alternative to token-level).
    pub fn compute_sequence_level_loss(
        &self,
        policy_logps: &Array,     // [batch]
        old_policy_logps: &Array, // [batch]
        advantages: &Array,       // [batch]
    ) -> DapoResult<(Array, f32)> {
        // Importance ratio
        let log_ratio = policy_logps.subtract(old_policy_logps)?;
        let ratio = log_ratio.exp()?;

        // Clip-Higher
        let eps_high = 1.0 + self.config.clip_eps_high;
        let eps_low = self.config.clip_eps_low;

        let ratio_clipped_high = mlx_rs::ops::minimum(&ratio, &Array::from_f32(eps_high as f32))?;
        let ratio_clipped =
            mlx_rs::ops::maximum(&ratio_clipped_high, &Array::from_f32(eps_low as f32))?;

        // Clip fraction
        let is_clipped = ratio.gt(&Array::from_f32(eps_high as f32))?;
        let clip_count = is_clipped.as_dtype(mlx_rs::Dtype::Float32)?.sum(None)?;
        let total_count = Array::from_int(policy_logps.dim(0));
        let clip_fraction = clip_count.divide(&total_count)?;
        clip_fraction.eval()?;
        let clip_frac = clip_fraction.item::<f32>();

        // Sequence-level loss
        let loss = ratio_clipped.multiply(advantages)?.negative()?.mean(None)?;

        Ok((loss, clip_frac))
    }

    /// Compute KL divergence from reference model.
    pub fn compute_kl(
        &self,
        policy_logps: &Array,
        ref_logps: &Array,
        mask: Option<&Array>,
    ) -> DapoResult<Array> {
        // KL approximation: (ratio - 1) - log_ratio
        let log_ratio = policy_logps.subtract(ref_logps)?;
        let ratio = log_ratio.exp()?;
        let one = Array::from_f32(1.0);
        let kl = ratio.subtract(&one)?.subtract(&log_ratio)?;

        if let Some(m) = mask {
            let masked_kl = kl.multiply(m)?;
            let total = m.sum(None)?;
            Ok(masked_kl.sum(None)?.divide(&total)?)
        } else {
            Ok(kl.mean(None)?)
        }
    }

    /// Compute entropy bonus.
    pub fn compute_entropy(&self, logits: &Array, mask: Option<&Array>) -> DapoResult<Array> {
        // Efficient entropy: H = logsumexp(x) - sum(softmax(x) * x)
        // Only materializes softmax once instead of both softmax + log_softmax
        let entropy = crate::logprob_utils::efficient_entropy(logits)?;

        if let Some(m) = mask {
            let masked_entropy = entropy.multiply(m)?;
            let total = m.sum(None)?;
            Ok(masked_entropy.sum(None)?.divide(&total)?)
        } else {
            Ok(entropy.mean(None)?)
        }
    }

    /// Prepare a training batch from prompt groups.
    ///
    /// Applies dynamic sampling and computes advantages.
    pub fn prepare_batch(&mut self, groups: &mut [DapoPromptGroup]) -> DapoResult<DapoBatch> {
        // Apply overlong penalty
        self.apply_overlong_penalty(groups);

        // Dynamic sampling
        let keep_indices = self.dynamic_sample(groups);
        let skipped = groups.len() - keep_indices.len();

        self.stats.total_prompts += groups.len();
        self.stats.skipped_prompts += skipped;

        if keep_indices.len() < self.config.min_group_size {
            return Err(DapoError::Generation(format!(
                "Too few groups after dynamic sampling: {} (need {})",
                keep_indices.len(),
                self.config.min_group_size
            )));
        }

        // Compute advantages for all kept groups
        for &i in &keep_indices {
            if let Some(group) = groups.get_mut(i) {
                let baseline = group.mean_reward();
                let std = if self.config.normalize_advantages {
                    group.reward_std()
                } else {
                    1.0
                };
                for completion in &mut group.completions {
                    completion.advantage = (completion.reward - baseline) / std;
                }
            }
        }

        // Flatten into batch (using immutable references now)
        let mut prompt_ids = Vec::new();
        let mut completion_ids = Vec::new();
        let mut advantages = Vec::new();
        let mut truncated = Vec::new();
        let mut num_groups = 0;

        for &i in &keep_indices {
            if let Some(group) = groups.get(i) {
                num_groups += 1;
                for completion in &group.completions {
                    prompt_ids.push(group.prompt_ids.clone());
                    completion_ids.push(completion.token_ids.clone());
                    advantages.push(completion.advantage);
                    truncated.push(completion.truncated);

                    self.stats.total_completions += 1;
                    if completion.truncated {
                        self.stats.truncated_completions += 1;
                    }
                }
            }
        }

        // Update stats
        let total_reward: f64 = keep_indices
            .iter()
            .filter_map(|&i| groups.get(i))
            .flat_map(|g| g.completions.iter().map(|c| c.reward))
            .sum();
        let n_completions = advantages.len();
        if n_completions > 0 {
            self.stats.mean_reward = total_reward / n_completions as f64;
            self.stats.mean_advantage = advantages.iter().sum::<f64>() / n_completions as f64;
        }

        Ok(DapoBatch {
            prompt_ids,
            completion_ids,
            advantages,
            truncated,
            num_groups,
            num_completions: n_completions,
        })
    }

    /// Get current training step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Increment step.
    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    /// Get current stats.
    pub fn stats(&self) -> &DapoStats {
        &self.stats
    }

    /// Reset stats.
    pub fn reset_stats(&mut self) {
        self.stats = DapoStats::default();
    }
}

/// A prepared batch for DAPO training.
#[derive(Debug, Clone)]
pub struct DapoBatch {
    /// Prompt token IDs for each completion.
    pub prompt_ids: Vec<Vec<u32>>,
    /// Completion token IDs.
    pub completion_ids: Vec<Vec<u32>>,
    /// Computed advantages.
    pub advantages: Vec<f64>,
    /// Whether each completion was truncated.
    pub truncated: Vec<bool>,
    /// Number of groups (prompts) in this batch.
    pub num_groups: usize,
    /// Total number of completions.
    pub num_completions: usize,
}

/// Metrics for a single DAPO training step.
#[derive(Debug, Clone, Default)]
pub struct DapoStepMetrics {
    /// Policy gradient loss.
    pub policy_loss: f32,
    /// KL divergence from reference.
    pub kl_divergence: f32,
    /// Entropy bonus.
    pub entropy: f32,
    /// Total loss.
    pub total_loss: f32,
    /// Clip fraction (how often importance ratio was clipped).
    pub clip_fraction: f32,
    /// Mean reward.
    pub mean_reward: f32,
    /// Mean advantage.
    pub mean_advantage: f32,
    /// Number of prompts processed.
    pub num_prompts: usize,
    /// Number of prompts skipped by dynamic sampling.
    pub num_skipped: usize,
    /// Number of completions.
    pub num_completions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dapo_config_default() {
        let config = DapoConfig::default();
        assert_eq!(config.num_generations, 16);
        assert!((config.beta - 0.0).abs() < 1e-10);
        assert!((config.clip_eps_high - 0.28).abs() < 0.01);
        assert!(config.dynamic_sampling);
        assert!(config.token_level_loss);
    }

    #[test]
    fn test_dapo_config_validation() {
        let config = DapoConfig::default();
        assert!(config.validate().is_ok());

        let invalid = DapoConfig {
            num_generations: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid_clip = DapoConfig {
            clip_eps_high: -1.0,
            ..Default::default()
        };
        assert!(invalid_clip.validate().is_err());
    }

    #[test]
    fn test_dapo_completion() {
        let completion = DapoCompletion::new(vec![1, 2, 3], 1.0, false);
        assert_eq!(completion.len(), 3);
        assert!(!completion.truncated);
        assert!((completion.reward - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dapo_prompt_group() {
        let mut group = DapoPromptGroup::new(vec![1, 2]);

        group.add_completion(DapoCompletion::new(vec![3, 4], 1.0, false), 0.0);
        group.add_completion(DapoCompletion::new(vec![5, 6], 0.0, false), 0.0);
        group.add_completion(DapoCompletion::new(vec![7, 8], 1.0, true), 0.0);
        group.add_completion(DapoCompletion::new(vec![9, 10], -1.0, false), 0.0);

        assert_eq!(group.len(), 4);
        // 2 positive, 2 non-positive -> accuracy = 0.5
        assert!((group.accuracy - 0.5).abs() < 1e-10);
        // Mean reward = (1 + 0 + 1 - 1) / 4 = 0.25
        assert!((group.mean_reward() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_overlong_penalty() {
        let config = DapoConfig::default().with_overlong_penalty(-2.0);
        let trainer = DapoTrainer::new(config).unwrap();

        let mut groups = vec![DapoPromptGroup::new(vec![1, 2])];
        groups[0].add_completion(DapoCompletion::new(vec![3, 4], 1.0, false), 0.0);
        groups[0].add_completion(DapoCompletion::new(vec![5, 6], 1.0, true), 0.0); // truncated

        trainer.apply_overlong_penalty(&mut groups);

        assert!((groups[0].completions[0].reward - 1.0).abs() < 1e-10); // No penalty
        assert!((groups[0].completions[1].reward - (-1.0)).abs() < 1e-10); // 1.0 - 2.0
    }

    #[test]
    fn test_dynamic_sampling() {
        let config = DapoConfig::default();
        let trainer = DapoTrainer::new(config).unwrap();

        // Group 1: All correct (accuracy = 1.0) -> skip
        let mut g1 = DapoPromptGroup::new(vec![1]);
        g1.add_completion(DapoCompletion::new(vec![2], 1.0, false), 0.0);
        g1.add_completion(DapoCompletion::new(vec![3], 1.0, false), 0.0);

        // Group 2: All wrong (accuracy = 0.0) -> skip
        let mut g2 = DapoPromptGroup::new(vec![4]);
        g2.add_completion(DapoCompletion::new(vec![5], 0.0, false), 0.0);
        g2.add_completion(DapoCompletion::new(vec![6], 0.0, false), 0.0);

        // Group 3: Mixed (accuracy = 0.5) -> keep
        let mut g3 = DapoPromptGroup::new(vec![7]);
        g3.add_completion(DapoCompletion::new(vec![8], 1.0, false), 0.0);
        g3.add_completion(DapoCompletion::new(vec![9], 0.0, false), 0.0);

        let groups = vec![g1, g2, g3];
        let keep = trainer.dynamic_sample(&groups);

        assert_eq!(keep.len(), 1);
        assert_eq!(keep[0], 2); // Only group 3 kept
    }

    #[test]
    fn test_compute_advantages() {
        let config = DapoConfig::default();
        let trainer = DapoTrainer::new(config).unwrap();

        let mut group = DapoPromptGroup::new(vec![1]);
        group.add_completion(DapoCompletion::new(vec![2], 1.0, false), 0.0);
        group.add_completion(DapoCompletion::new(vec![3], 3.0, false), 0.0);
        group.add_completion(DapoCompletion::new(vec![4], 2.0, false), 0.0);
        group.add_completion(DapoCompletion::new(vec![5], 2.0, false), 0.0);

        let mut groups = vec![group];
        trainer.compute_advantages(&mut groups);

        // Mean = 2.0, normalized advantages should sum to ~0
        let adv_sum: f64 = groups[0].completions.iter().map(|c| c.advantage).sum();
        assert!(adv_sum.abs() < 0.01);
    }

    #[test]
    fn test_clip_higher_loss() {
        let config = DapoConfig::default();
        let trainer = DapoTrainer::new(config).unwrap();

        // Policy improved significantly (high ratio)
        let policy_logps = Array::from_slice(&[-1.0f32, -1.0], &[2, 1]);
        let old_logps = Array::from_slice(&[-3.0f32, -3.0], &[2, 1]); // much lower
        let advantages = Array::from_slice(&[1.0f32, 1.0], &[2]);
        let mask = Array::from_slice(&[1.0f32, 1.0], &[2, 1]);

        let (loss, clip_frac) = trainer
            .compute_token_level_loss(&policy_logps, &old_logps, &advantages, &mask)
            .unwrap();

        loss.eval().unwrap();
        assert!(loss.item::<f32>().is_finite());
        // High ratio (exp(2) ≈ 7.4) should be clipped to 1.28
        assert!(clip_frac > 0.0);
    }

    #[test]
    fn test_prepare_batch() {
        let config = DapoConfig {
            dynamic_sampling: false, // Disable for predictable test
            ..Default::default()
        };
        let mut trainer = DapoTrainer::new(config).unwrap();

        let mut groups = vec![
            DapoPromptGroup::new(vec![1, 2]),
            DapoPromptGroup::new(vec![3, 4]),
        ];

        groups[0].add_completion(DapoCompletion::new(vec![5, 6], 1.0, false), 0.0);
        groups[0].add_completion(DapoCompletion::new(vec![7, 8], 0.0, false), 0.0);
        groups[1].add_completion(DapoCompletion::new(vec![9, 10], 2.0, false), 0.0);
        groups[1].add_completion(DapoCompletion::new(vec![11, 12], 1.0, true), 0.0);

        let batch = trainer.prepare_batch(&mut groups).unwrap();

        assert_eq!(batch.num_groups, 2);
        assert_eq!(batch.num_completions, 4);
        assert_eq!(batch.advantages.len(), 4);
        assert_eq!(batch.truncated.len(), 4);
    }

    #[test]
    fn test_dapo_stats() {
        let config = DapoConfig::default();
        let mut trainer = DapoTrainer::new(config).unwrap();

        assert_eq!(trainer.step(), 0);
        trainer.increment_step();
        assert_eq!(trainer.step(), 1);

        let stats = trainer.stats();
        assert_eq!(stats.total_prompts, 0);

        trainer.reset_stats();
        assert_eq!(trainer.stats().total_completions, 0);
    }
}
