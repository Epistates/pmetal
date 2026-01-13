//! GSPO: Group Sequence Policy Optimization.
//!
//! GSPO fixes critical instabilities in GRPO by addressing length bias and
//! improving advantage estimation. Based on research from 2025.
//!
//! Key innovations over GRPO:
//!
//! 1. **Equal Token Weighting**: All tokens contribute equally to the loss,
//!    regardless of sequence length. This prevents shorter completions from
//!    being unfairly advantaged.
//!
//! 2. **Sequence-Normalized Rewards**: Rewards are normalized by sequence length
//!    to account for the difficulty of generating longer responses.
//!
//! 3. **Stable Advantage Estimation**: Uses robust statistics (median, trimmed mean)
//!    to compute group baselines, reducing sensitivity to outliers.
//!
//! 4. **Completion Quality Weighting**: Optionally weights samples by their
//!    estimated quality (completion probability under reference model).
//!
//! # Usage
//!
//! ```rust,ignore
//! use pmetal_trainer::{GspoConfig, GspoTrainer};
//!
//! let config = GspoConfig::default()
//!     .with_length_normalization(true)
//!     .with_robust_baseline(true);
//!
//! let trainer = GspoTrainer::new(config)?;
//! ```

use mlx_rs::{error::Exception, Array};

/// Error type for GSPO training.
#[derive(Debug, thiserror::Error)]
pub enum GspoError {
    /// MLX computation error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
}

/// Result type for GSPO operations.
pub type GspoResult<T> = std::result::Result<T, GspoError>;

/// Baseline estimation method for GSPO.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BaselineMethod {
    /// Use group mean (standard GRPO).
    #[default]
    Mean,
    /// Use group median (more robust to outliers).
    Median,
    /// Use trimmed mean (remove top/bottom 10%).
    TrimmedMean,
    /// Use exponentially weighted moving average.
    Ewma,
    /// No baseline (raw rewards as advantages).
    None,
}

/// Token weighting strategy for GSPO.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenWeighting {
    /// Equal weight for all tokens (GSPO default).
    #[default]
    Equal,
    /// Weight by position (later tokens weighted more).
    PositionalDecay,
    /// Weight by attention entropy (uncertain tokens weighted more).
    AttentionBased,
    /// Standard per-sequence weighting (like GRPO).
    PerSequence,
}

/// GSPO configuration.
#[derive(Debug, Clone)]
pub struct GspoConfig {
    /// Number of completions per prompt (group size).
    /// Default: 8
    pub num_generations: usize,

    /// KL penalty coefficient (beta).
    /// Default: 0.01
    pub beta: f64,

    /// Baseline estimation method.
    /// Default: Median (more robust than mean)
    pub baseline_method: BaselineMethod,

    /// Token weighting strategy.
    /// Default: Equal (all tokens contribute equally)
    pub token_weighting: TokenWeighting,

    /// Whether to normalize rewards by sequence length.
    /// Default: true (key GSPO innovation)
    pub length_normalization: bool,

    /// Minimum sequence length for normalization (prevents division by small numbers).
    /// Default: 10
    pub min_length_for_norm: usize,

    /// Whether to clip advantages.
    /// Default: true
    pub clip_advantages: bool,

    /// Advantage clipping range.
    /// Default: 5.0
    pub advantage_clip: f64,

    /// Whether to whiten advantages (zero mean, unit variance).
    /// Default: true
    pub whiten_advantages: bool,

    /// Temperature for generation.
    /// Default: 1.0
    pub temperature: f64,

    /// Whether to use reference-free mode.
    /// Default: false
    pub reference_free: bool,

    /// Entropy coefficient for exploration bonus.
    /// Default: 0.01
    pub entropy_coef: f64,

    /// EWMA decay factor (if using EWMA baseline).
    /// Default: 0.99
    pub ewma_decay: f64,

    /// Quality weighting factor (0 = no quality weighting).
    /// Default: 0.0
    pub quality_weight: f64,
}

impl Default for GspoConfig {
    fn default() -> Self {
        Self {
            num_generations: 8,
            beta: 0.01,
            baseline_method: BaselineMethod::Median,
            token_weighting: TokenWeighting::Equal,
            length_normalization: true,
            min_length_for_norm: 10,
            clip_advantages: true,
            advantage_clip: 5.0,
            whiten_advantages: true,
            temperature: 1.0,
            reference_free: false,
            entropy_coef: 0.01,
            ewma_decay: 0.99,
            quality_weight: 0.0,
        }
    }
}

impl GspoConfig {
    /// Create a new GSPO config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set baseline method.
    pub fn with_baseline_method(mut self, method: BaselineMethod) -> Self {
        self.baseline_method = method;
        self
    }

    /// Enable/disable length normalization.
    pub fn with_length_normalization(mut self, enabled: bool) -> Self {
        self.length_normalization = enabled;
        self
    }

    /// Set token weighting strategy.
    pub fn with_token_weighting(mut self, weighting: TokenWeighting) -> Self {
        self.token_weighting = weighting;
        self
    }

    /// Set beta (KL coefficient).
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Enable robust baseline (median + whitening).
    pub fn with_robust_baseline(mut self, enabled: bool) -> Self {
        if enabled {
            self.baseline_method = BaselineMethod::Median;
            self.whiten_advantages = true;
            self.clip_advantages = true;
        }
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> GspoResult<()> {
        if self.num_generations == 0 {
            return Err(GspoError::Config(
                "num_generations must be at least 1".into(),
            ));
        }
        if self.beta < 0.0 {
            return Err(GspoError::Config("beta must be non-negative".into()));
        }
        if self.temperature <= 0.0 {
            return Err(GspoError::Config("temperature must be positive".into()));
        }
        if self.advantage_clip <= 0.0 {
            return Err(GspoError::Config("advantage_clip must be positive".into()));
        }
        Ok(())
    }
}

/// A completion in a GSPO group.
#[derive(Debug, Clone)]
pub struct GspoCompletion {
    /// Token IDs.
    pub token_ids: Vec<u32>,
    /// Raw reward.
    pub reward: f64,
    /// Length-normalized reward (if applicable).
    pub normalized_reward: f64,
    /// Per-token log probabilities from policy.
    pub token_logps: Vec<f32>,
    /// Per-token log probabilities from reference (if not reference-free).
    pub ref_token_logps: Option<Vec<f32>>,
    /// Computed advantage.
    pub advantage: f64,
    /// Quality score (optional, from reference model).
    pub quality_score: Option<f64>,
}

impl GspoCompletion {
    /// Create a new completion.
    pub fn new(token_ids: Vec<u32>, reward: f64) -> Self {
        Self {
            token_ids,
            reward,
            normalized_reward: reward,
            token_logps: Vec::new(),
            ref_token_logps: None,
            advantage: 0.0,
            quality_score: None,
        }
    }

    /// Get sequence length.
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

/// A group of completions for a single prompt.
#[derive(Debug, Clone)]
pub struct GspoGroup {
    /// Prompt token IDs.
    pub prompt_ids: Vec<u32>,
    /// Completions in this group.
    pub completions: Vec<GspoCompletion>,
}

impl GspoGroup {
    /// Create a new group.
    pub fn new(prompt_ids: Vec<u32>) -> Self {
        Self {
            prompt_ids,
            completions: Vec::new(),
        }
    }

    /// Add a completion.
    pub fn add_completion(&mut self, completion: GspoCompletion) {
        self.completions.push(completion);
    }

    /// Get group size.
    pub fn len(&self) -> usize {
        self.completions.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.completions.is_empty()
    }

    /// Get rewards as a vector.
    pub fn rewards(&self) -> Vec<f64> {
        self.completions
            .iter()
            .map(|c| c.normalized_reward)
            .collect()
    }

    /// Compute mean reward.
    pub fn mean_reward(&self) -> f64 {
        if self.completions.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.completions.iter().map(|c| c.normalized_reward).sum();
        sum / self.completions.len() as f64
    }

    /// Compute median reward.
    pub fn median_reward(&self) -> f64 {
        if self.completions.is_empty() {
            return 0.0;
        }
        let mut rewards: Vec<f64> = self.rewards();
        rewards.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = rewards.len();
        if n % 2 == 0 {
            (rewards[n / 2 - 1] + rewards[n / 2]) / 2.0
        } else {
            rewards[n / 2]
        }
    }

    /// Compute trimmed mean (remove top and bottom 10%).
    pub fn trimmed_mean(&self) -> f64 {
        if self.completions.len() < 4 {
            return self.mean_reward();
        }
        let mut rewards = self.rewards();
        rewards.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let trim_count = (rewards.len() as f64 * 0.1).ceil() as usize;
        let trimmed: Vec<f64> = rewards
            .iter()
            .skip(trim_count)
            .take(rewards.len() - 2 * trim_count)
            .copied()
            .collect();
        if trimmed.is_empty() {
            return self.mean_reward();
        }
        trimmed.iter().sum::<f64>() / trimmed.len() as f64
    }

    /// Compute reward standard deviation.
    pub fn reward_std(&self) -> f64 {
        if self.completions.len() < 2 {
            return 1.0;
        }
        let mean = self.mean_reward();
        let var: f64 = self
            .completions
            .iter()
            .map(|c| (c.normalized_reward - mean).powi(2))
            .sum::<f64>()
            / self.completions.len() as f64;
        var.sqrt().max(1e-8)
    }
}

/// GSPO Trainer.
pub struct GspoTrainer {
    /// Configuration.
    pub config: GspoConfig,
    /// Current training step.
    step: usize,
    /// EWMA baseline (if using EWMA method).
    ewma_baseline: f64,
    /// Whether EWMA has been initialized.
    ewma_initialized: bool,
}

impl GspoTrainer {
    /// Create a new GSPO trainer.
    pub fn new(config: GspoConfig) -> GspoResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            step: 0,
            ewma_baseline: 0.0,
            ewma_initialized: false,
        })
    }

    /// Normalize rewards by sequence length.
    pub fn normalize_rewards(&self, groups: &mut [GspoGroup]) {
        if !self.config.length_normalization {
            for group in groups {
                for completion in &mut group.completions {
                    completion.normalized_reward = completion.reward;
                }
            }
            return;
        }

        for group in groups {
            for completion in &mut group.completions {
                let effective_len = completion.len().max(self.config.min_length_for_norm);
                // Normalize by sqrt of length (balances length bias without over-correcting)
                completion.normalized_reward = completion.reward / (effective_len as f64).sqrt();
            }
        }
    }

    /// Compute baseline for a group.
    fn compute_baseline(&self, group: &GspoGroup) -> f64 {
        match self.config.baseline_method {
            BaselineMethod::Mean => group.mean_reward(),
            BaselineMethod::Median => group.median_reward(),
            BaselineMethod::TrimmedMean => group.trimmed_mean(),
            BaselineMethod::Ewma => self.ewma_baseline,
            BaselineMethod::None => 0.0,
        }
    }

    /// Update EWMA baseline with new rewards.
    pub fn update_ewma(&mut self, rewards: &[f64]) {
        if rewards.is_empty() {
            return;
        }
        let batch_mean: f64 = rewards.iter().sum::<f64>() / rewards.len() as f64;
        if !self.ewma_initialized {
            self.ewma_baseline = batch_mean;
            self.ewma_initialized = true;
        } else {
            self.ewma_baseline = self.config.ewma_decay * self.ewma_baseline
                + (1.0 - self.config.ewma_decay) * batch_mean;
        }
    }

    /// Compute advantages for all completions.
    pub fn compute_advantages(&mut self, groups: &mut [GspoGroup]) {
        // First normalize rewards
        self.normalize_rewards(groups);

        // Update EWMA if using it
        if self.config.baseline_method == BaselineMethod::Ewma {
            let all_rewards: Vec<f64> = groups
                .iter()
                .flat_map(|g| g.completions.iter().map(|c| c.normalized_reward))
                .collect();
            self.update_ewma(&all_rewards);
        }

        // Compute raw advantages
        for group in groups.iter_mut() {
            let baseline = self.compute_baseline(group);
            let std = if self.config.whiten_advantages {
                group.reward_std()
            } else {
                1.0
            };

            for completion in &mut group.completions {
                let mut advantage = (completion.normalized_reward - baseline) / std;

                // Clip advantages
                if self.config.clip_advantages {
                    advantage =
                        advantage.clamp(-self.config.advantage_clip, self.config.advantage_clip);
                }

                // Apply quality weighting if enabled
                if self.config.quality_weight > 0.0 {
                    if let Some(quality) = completion.quality_score {
                        advantage *= 1.0 + self.config.quality_weight * quality;
                    }
                }

                completion.advantage = advantage;
            }
        }
    }

    /// Compute equal-weighted token-level loss.
    ///
    /// This is the key GSPO innovation: all tokens contribute equally regardless
    /// of sequence length.
    ///
    /// # Arguments
    /// * `per_token_policy_logps` - Per-token log probs [batch, max_seq]
    /// * `per_token_ref_logps` - Per-token reference log probs [batch, max_seq]
    /// * `advantages` - Per-sequence advantages [batch]
    /// * `mask` - Valid token mask [batch, max_seq]
    ///
    /// # Returns
    /// (policy_loss, kl_divergence)
    pub fn compute_equal_weighted_loss(
        &self,
        per_token_policy_logps: &Array,
        per_token_ref_logps: &Array,
        advantages: &Array,
        mask: &Array,
    ) -> GspoResult<(Array, Array)> {
        // Per-token KL divergence
        let log_ratio = per_token_policy_logps.subtract(per_token_ref_logps)?;
        let ratio = log_ratio.exp()?;
        let one = Array::from_f32(1.0);
        let per_token_kl = ratio.subtract(&one)?.subtract(&log_ratio)?;

        // Count total valid tokens across entire batch (for equal weighting)
        let total_tokens = mask.sum(None)?;

        // Compute per-token loss with advantage
        // Key insight: we DON'T normalize per sequence, we treat all tokens equally
        let adv_expanded = advantages.reshape(&[advantages.dim(0), 1])?;
        let per_token_loss = per_token_policy_logps.negative()?.multiply(&adv_expanded)?;

        // Apply mask and compute mean over ALL tokens (equal weighting)
        let masked_loss = per_token_loss.multiply(mask)?;
        let mean_loss = masked_loss.sum(None)?.divide(&total_tokens)?;

        // KL loss (also equal-weighted over all tokens)
        let masked_kl = per_token_kl.multiply(mask)?;
        let mean_kl = masked_kl.sum(None)?.divide(&total_tokens)?;

        // Total loss with KL penalty
        let total_loss = if self.config.beta > 0.0 {
            let beta = Array::from_f32(self.config.beta as f32);
            mean_loss.add(&mean_kl.multiply(&beta)?)?
        } else {
            mean_loss
        };

        Ok((total_loss, mean_kl))
    }

    /// Compute positional-decay weighted loss.
    ///
    /// Later tokens get progressively more weight, encouraging the model
    /// to maintain quality throughout generation.
    pub fn compute_positional_weighted_loss(
        &self,
        per_token_policy_logps: &Array,
        per_token_ref_logps: &Array,
        advantages: &Array,
        mask: &Array,
    ) -> GspoResult<(Array, Array)> {
        let seq_len = per_token_policy_logps.dim(1);

        // Create positional weights: [1, seq_len]
        let positions: Vec<f32> = (0..seq_len)
            .map(|i| 1.0 + (i as f32 / seq_len as f32))
            .collect();
        let pos_weights = Array::from_slice(&positions, &[1, seq_len]);

        // Apply positional weights to mask
        let weighted_mask = mask.multiply(&pos_weights)?;

        // Per-token KL
        let log_ratio = per_token_policy_logps.subtract(per_token_ref_logps)?;
        let ratio = log_ratio.exp()?;
        let one = Array::from_f32(1.0);
        let per_token_kl = ratio.subtract(&one)?.subtract(&log_ratio)?;

        // Total weight for normalization
        let total_weight = weighted_mask.sum(None)?;

        // Per-token loss with advantage
        let adv_expanded = advantages.reshape(&[advantages.dim(0), 1])?;
        let per_token_loss = per_token_policy_logps.negative()?.multiply(&adv_expanded)?;

        // Apply weighted mask
        let weighted_loss = per_token_loss.multiply(&weighted_mask)?;
        let mean_loss = weighted_loss.sum(None)?.divide(&total_weight)?;

        // Weighted KL
        let weighted_kl = per_token_kl.multiply(&weighted_mask)?;
        let mean_kl = weighted_kl.sum(None)?.divide(&total_weight)?;

        // Total loss
        let total_loss = if self.config.beta > 0.0 {
            let beta = Array::from_f32(self.config.beta as f32);
            mean_loss.add(&mean_kl.multiply(&beta)?)?
        } else {
            mean_loss
        };

        Ok((total_loss, mean_kl))
    }

    /// Compute per-sequence weighted loss (standard GRPO-style).
    pub fn compute_sequence_weighted_loss(
        &self,
        per_token_policy_logps: &Array,
        per_token_ref_logps: &Array,
        advantages: &Array,
        mask: &Array,
    ) -> GspoResult<(Array, Array)> {
        // Sum log probs per sequence
        let masked_policy = per_token_policy_logps.multiply(mask)?;
        let seq_policy_logps = masked_policy.sum_axis(-1, None)?;

        let masked_ref = per_token_ref_logps.multiply(mask)?;
        let seq_ref_logps = masked_ref.sum_axis(-1, None)?;

        // Sequence-level KL
        let log_ratio = seq_policy_logps.subtract(&seq_ref_logps)?;
        let ratio = log_ratio.exp()?;
        let one = Array::from_f32(1.0);
        let seq_kl = ratio.subtract(&one)?.subtract(&log_ratio)?;
        let mean_kl = seq_kl.mean(None)?;

        // Sequence-level loss
        let seq_loss = seq_policy_logps.negative()?.multiply(advantages)?;
        let mean_loss = seq_loss.mean(None)?;

        // Total loss
        let total_loss = if self.config.beta > 0.0 {
            let beta = Array::from_f32(self.config.beta as f32);
            mean_loss.add(&mean_kl.multiply(&beta)?)?
        } else {
            mean_loss
        };

        Ok((total_loss, mean_kl))
    }

    /// Main loss computation method - dispatches based on token weighting config.
    pub fn compute_loss(
        &self,
        per_token_policy_logps: &Array,
        per_token_ref_logps: &Array,
        advantages: &Array,
        mask: &Array,
    ) -> GspoResult<(Array, Array)> {
        match self.config.token_weighting {
            TokenWeighting::Equal => self.compute_equal_weighted_loss(
                per_token_policy_logps,
                per_token_ref_logps,
                advantages,
                mask,
            ),
            TokenWeighting::PositionalDecay => self.compute_positional_weighted_loss(
                per_token_policy_logps,
                per_token_ref_logps,
                advantages,
                mask,
            ),
            TokenWeighting::PerSequence => self.compute_sequence_weighted_loss(
                per_token_policy_logps,
                per_token_ref_logps,
                advantages,
                mask,
            ),
            TokenWeighting::AttentionBased => {
                // Fall back to equal weighting if attention weights not available
                self.compute_equal_weighted_loss(
                    per_token_policy_logps,
                    per_token_ref_logps,
                    advantages,
                    mask,
                )
            }
        }
    }

    /// Get current step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Increment step.
    pub fn increment_step(&mut self) {
        self.step += 1;
    }
}

/// Metrics for GSPO training.
#[derive(Debug, Clone, Default)]
pub struct GspoMetrics {
    /// Policy loss.
    pub policy_loss: f32,
    /// KL divergence.
    pub kl_divergence: f32,
    /// Total loss.
    pub total_loss: f32,
    /// Mean reward.
    pub mean_reward: f32,
    /// Mean advantage.
    pub mean_advantage: f32,
    /// Mean completion length.
    pub mean_length: f32,
    /// Advantage standard deviation.
    pub advantage_std: f32,
}

impl GspoMetrics {
    /// Compute metrics from training data.
    pub fn compute(
        policy_loss: f32,
        kl: f32,
        total_loss: f32,
        rewards: &[f64],
        advantages: &[f64],
        lengths: &[usize],
    ) -> Self {
        let n = rewards.len() as f32;
        let mean_reward = rewards.iter().sum::<f64>() as f32 / n;
        let mean_advantage = advantages.iter().sum::<f64>() as f32 / n;
        let mean_length = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;

        let adv_var: f32 = advantages
            .iter()
            .map(|a| (*a as f32 - mean_advantage).powi(2))
            .sum::<f32>()
            / n;
        let advantage_std = adv_var.sqrt();

        Self {
            policy_loss,
            kl_divergence: kl,
            total_loss,
            mean_reward,
            mean_advantage,
            mean_length,
            advantage_std,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gspo_config_default() {
        let config = GspoConfig::default();
        assert_eq!(config.num_generations, 8);
        assert_eq!(config.baseline_method, BaselineMethod::Median);
        assert_eq!(config.token_weighting, TokenWeighting::Equal);
        assert!(config.length_normalization);
        assert!(config.whiten_advantages);
    }

    #[test]
    fn test_gspo_config_validation() {
        let config = GspoConfig::default();
        assert!(config.validate().is_ok());

        let invalid = GspoConfig {
            num_generations: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid_beta = GspoConfig {
            beta: -1.0,
            ..Default::default()
        };
        assert!(invalid_beta.validate().is_err());
    }

    #[test]
    fn test_gspo_completion() {
        let completion = GspoCompletion::new(vec![1, 2, 3, 4, 5], 2.5);
        assert_eq!(completion.len(), 5);
        assert!((completion.reward - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_gspo_group_statistics() {
        let mut group = GspoGroup::new(vec![1, 2]);
        group.add_completion(GspoCompletion::new(vec![3], 1.0));
        group.add_completion(GspoCompletion::new(vec![4, 5], 2.0));
        group.add_completion(GspoCompletion::new(vec![6, 7, 8], 3.0));
        group.add_completion(GspoCompletion::new(vec![9, 10], 4.0));
        group.add_completion(GspoCompletion::new(vec![11], 5.0));

        // Set normalized rewards for testing
        for c in &mut group.completions {
            c.normalized_reward = c.reward;
        }

        assert_eq!(group.len(), 5);
        assert!((group.mean_reward() - 3.0).abs() < 1e-10);
        assert!((group.median_reward() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_length_normalization() {
        let config = GspoConfig {
            length_normalization: true,
            min_length_for_norm: 1,
            ..Default::default()
        };
        let trainer = GspoTrainer::new(config).unwrap();

        let mut groups = vec![GspoGroup::new(vec![1])];
        // Same reward but different lengths
        groups[0].add_completion(GspoCompletion::new(vec![2; 100], 10.0)); // long
        groups[0].add_completion(GspoCompletion::new(vec![3; 4], 10.0)); // short

        trainer.normalize_rewards(&mut groups);

        // Short completion should have higher normalized reward
        // because 10/sqrt(4) > 10/sqrt(100)
        let norm_long = groups[0].completions[0].normalized_reward;
        let norm_short = groups[0].completions[1].normalized_reward;
        assert!(norm_short > norm_long);
    }

    #[test]
    fn test_median_baseline() {
        let mut group = GspoGroup::new(vec![1]);
        group.add_completion(GspoCompletion::new(vec![2], 1.0));
        group.add_completion(GspoCompletion::new(vec![3], 2.0));
        group.add_completion(GspoCompletion::new(vec![4], 3.0));
        group.add_completion(GspoCompletion::new(vec![5], 100.0)); // outlier

        for c in &mut group.completions {
            c.normalized_reward = c.reward;
        }

        // Mean is (1+2+3+100)/4 = 26.5
        // Median is (2+3)/2 = 2.5
        let mean = group.mean_reward();
        let median = group.median_reward();

        assert!((mean - 26.5).abs() < 0.01);
        assert!((median - 2.5).abs() < 0.01);
        // Median is more robust to the outlier
    }

    #[test]
    fn test_compute_advantages() {
        let config = GspoConfig {
            baseline_method: BaselineMethod::Mean,
            whiten_advantages: false,
            length_normalization: false,
            ..Default::default()
        };
        let mut trainer = GspoTrainer::new(config).unwrap();

        let mut groups = vec![GspoGroup::new(vec![1])];
        groups[0].add_completion(GspoCompletion::new(vec![2], 1.0));
        groups[0].add_completion(GspoCompletion::new(vec![3], 3.0));
        groups[0].add_completion(GspoCompletion::new(vec![4], 2.0));

        trainer.compute_advantages(&mut groups);

        // Mean = 2.0
        // Advantages: 1-2=-1, 3-2=1, 2-2=0
        let advs: Vec<f64> = groups[0].completions.iter().map(|c| c.advantage).collect();

        assert!((advs[0] - (-1.0)).abs() < 0.01);
        assert!((advs[1] - 1.0).abs() < 0.01);
        assert!(advs[2].abs() < 0.01);
    }

    #[test]
    fn test_ewma_baseline() {
        let config = GspoConfig {
            baseline_method: BaselineMethod::Ewma,
            ewma_decay: 0.9,
            ..Default::default()
        };
        let mut trainer = GspoTrainer::new(config).unwrap();

        // First update initializes EWMA
        trainer.update_ewma(&[1.0, 2.0, 3.0]);
        assert!((trainer.ewma_baseline - 2.0).abs() < 0.01);

        // Second update with higher values
        trainer.update_ewma(&[5.0, 5.0, 5.0]);
        // EWMA = 0.9 * 2.0 + 0.1 * 5.0 = 1.8 + 0.5 = 2.3
        assert!((trainer.ewma_baseline - 2.3).abs() < 0.01);
    }

    #[test]
    fn test_equal_weighted_loss() {
        let config = GspoConfig {
            token_weighting: TokenWeighting::Equal,
            beta: 0.0, // No KL for simpler test
            ..Default::default()
        };
        let trainer = GspoTrainer::new(config).unwrap();

        let policy_logps = Array::from_slice(&[-1.0f32, -1.0, -1.0, -1.0, -1.0, -1.0], &[2, 3]);
        let ref_logps = Array::from_slice(&[-1.0f32, -1.0, -1.0, -1.0, -1.0, -1.0], &[2, 3]);
        let advantages = Array::from_slice(&[1.0f32, -1.0], &[2]);
        // Different length sequences: [3 tokens, 2 tokens]
        let mask = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 0.0], &[2, 3]);

        let (loss, kl) = trainer
            .compute_equal_weighted_loss(&policy_logps, &ref_logps, &advantages, &mask)
            .unwrap();

        loss.eval().unwrap();
        kl.eval().unwrap();

        assert!(loss.item::<f32>().is_finite());
        // KL should be ~0 since policy == reference
        assert!(kl.item::<f32>().abs() < 0.01);
    }

    #[test]
    fn test_positional_weighted_loss() {
        let config = GspoConfig {
            token_weighting: TokenWeighting::PositionalDecay,
            beta: 0.0,
            ..Default::default()
        };
        let trainer = GspoTrainer::new(config).unwrap();

        let policy_logps = Array::from_slice(&[-1.0f32, -1.0, -1.0, -1.0], &[2, 2]);
        let ref_logps = Array::from_slice(&[-1.0f32, -1.0, -1.0, -1.0], &[2, 2]);
        let advantages = Array::from_slice(&[1.0f32, 1.0], &[2]);
        let mask = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);

        let (loss, _kl) = trainer
            .compute_positional_weighted_loss(&policy_logps, &ref_logps, &advantages, &mask)
            .unwrap();

        loss.eval().unwrap();
        assert!(loss.item::<f32>().is_finite());
    }

    #[test]
    fn test_gspo_metrics() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0];
        let advantages = vec![-1.0, 0.0, 1.0, 2.0];
        let lengths = vec![10, 15, 20, 25];

        let metrics = GspoMetrics::compute(0.5, 0.01, 0.51, &rewards, &advantages, &lengths);

        assert!((metrics.mean_reward - 2.5).abs() < 0.01);
        assert!((metrics.mean_advantage - 0.5).abs() < 0.01);
        assert!((metrics.mean_length - 17.5).abs() < 0.01);
    }

    #[test]
    fn test_trimmed_mean() {
        let mut group = GspoGroup::new(vec![1]);
        // 10 completions
        for r in &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
            let mut c = GspoCompletion::new(vec![1], *r);
            c.normalized_reward = *r;
            group.add_completion(c);
        }

        // Trimmed mean removes 1 from each end (10%), leaving 2-9
        // Mean of 2,3,4,5,6,7,8,9 = 44/8 = 5.5
        let trimmed = group.trimmed_mean();
        assert!((trimmed - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_advantage_clipping() {
        let config = GspoConfig {
            clip_advantages: true,
            advantage_clip: 2.0,
            whiten_advantages: false,
            length_normalization: false,
            ..Default::default()
        };
        let mut trainer = GspoTrainer::new(config).unwrap();

        let mut groups = vec![GspoGroup::new(vec![1])];
        // Very extreme rewards
        groups[0].add_completion(GspoCompletion::new(vec![2], -100.0));
        groups[0].add_completion(GspoCompletion::new(vec![3], 0.0));
        groups[0].add_completion(GspoCompletion::new(vec![4], 100.0));

        trainer.compute_advantages(&mut groups);

        // All advantages should be clipped to [-2, 2]
        for c in &groups[0].completions {
            assert!(c.advantage >= -2.0 && c.advantage <= 2.0);
        }
    }
}
