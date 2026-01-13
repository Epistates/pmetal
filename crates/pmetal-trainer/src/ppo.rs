//! Proximal Policy Optimization (PPO) trainer.
//!
//! PPO is a popular reinforcement learning algorithm that uses clipped surrogate
//! objectives to prevent large policy updates, providing stable training.
//!
//! # Algorithm
//!
//! PPO optimizes a clipped objective:
//! ```text
//! L_CLIP = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
//! ```
//!
//! Where:
//! - `r(θ) = π_θ(a|s) / π_old(a|s)` is the probability ratio
//! - `A` is the advantage estimate (GAE or simple)
//! - `ε` is the clipping parameter (default: 0.2)
//!
//! # Key Features
//!
//! - **Clipped objective**: Prevents destructive policy updates
//! - **Value function**: Learns state-value baseline for advantage estimation
//! - **GAE**: Generalized Advantage Estimation for variance reduction
//! - **Multiple epochs**: Reuses collected data for multiple gradient updates
//!
//! # References
//!
//! - "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
//! - TRL/Unsloth implementations

use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use pmetal_core::TrainingConfig;

/// Error type for PPO training.
#[derive(Debug, thiserror::Error)]
pub enum PpoError {
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
    /// Value function error.
    #[error("Value error: {0}")]
    Value(String),
}

/// Result type for PPO operations.
pub type PpoResult<T> = std::result::Result<T, PpoError>;

/// GAE (Generalized Advantage Estimation) configuration.
#[derive(Debug, Clone)]
pub struct GaeConfig {
    /// GAE lambda (bias-variance tradeoff). Default: 0.95.
    pub lambda: f64,
    /// Discount factor gamma. Default: 1.0 for language models.
    pub gamma: f64,
    /// Normalize advantages. Default: true.
    pub normalize: bool,
}

impl Default for GaeConfig {
    fn default() -> Self {
        Self {
            lambda: 0.95,
            gamma: 1.0,
            normalize: true,
        }
    }
}

/// PPO configuration.
#[derive(Debug, Clone)]
pub struct PpoConfig {
    /// Clipping parameter epsilon. Default: 0.2.
    pub clip_eps: f64,

    /// Value function clipping. Default: None (no clipping).
    pub vf_clip: Option<f64>,

    /// Value function loss coefficient. Default: 0.5.
    pub vf_coef: f64,

    /// Entropy bonus coefficient. Default: 0.01.
    pub entropy_coef: f64,

    /// KL penalty coefficient (adaptive). Default: 0.0.
    pub kl_coef: f64,

    /// Target KL for adaptive penalty. Default: None.
    pub target_kl: Option<f64>,

    /// Number of PPO epochs per batch. Default: 4.
    pub ppo_epochs: usize,

    /// Mini-batch size for PPO updates. Default: 64.
    pub mini_batch_size: usize,

    /// GAE configuration.
    pub gae: GaeConfig,

    /// Whether to use a value head (critic). Default: true.
    pub use_value_head: bool,

    /// Whether to use reference model for KL. Default: true.
    pub use_reference_model: bool,

    /// Maximum gradient norm for clipping. Default: 0.5.
    pub max_grad_norm: f64,

    /// Maximum sequence length for rollouts.
    pub max_seq_length: usize,

    /// Number of rollouts per batch.
    pub num_rollouts: usize,

    /// Temperature for sampling. Default: 1.0.
    pub temperature: f64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_eps: 0.2,
            vf_clip: None,
            vf_coef: 0.5,
            entropy_coef: 0.01,
            kl_coef: 0.0,
            target_kl: None,
            ppo_epochs: 4,
            mini_batch_size: 64,
            gae: GaeConfig::default(),
            use_value_head: true,
            use_reference_model: true,
            max_grad_norm: 0.5,
            max_seq_length: 512,
            num_rollouts: 8,
            temperature: 1.0,
        }
    }
}

impl PpoConfig {
    /// Create a new PPO config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set clipping epsilon.
    pub fn with_clip_eps(mut self, eps: f64) -> Self {
        self.clip_eps = eps;
        self
    }

    /// Set value function coefficient.
    pub fn with_vf_coef(mut self, coef: f64) -> Self {
        self.vf_coef = coef;
        self
    }

    /// Set entropy coefficient.
    pub fn with_entropy_coef(mut self, coef: f64) -> Self {
        self.entropy_coef = coef;
        self
    }

    /// Set PPO epochs.
    pub fn with_ppo_epochs(mut self, epochs: usize) -> Self {
        self.ppo_epochs = epochs;
        self
    }

    /// Set mini-batch size.
    pub fn with_mini_batch_size(mut self, size: usize) -> Self {
        self.mini_batch_size = size;
        self
    }

    /// Enable adaptive KL penalty.
    pub fn with_adaptive_kl(mut self, init_coef: f64, target: f64) -> Self {
        self.kl_coef = init_coef;
        self.target_kl = Some(target);
        self
    }

    /// Disable value head.
    pub fn without_value_head(mut self) -> Self {
        self.use_value_head = false;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> PpoResult<()> {
        if self.clip_eps <= 0.0 {
            return Err(PpoError::Config("clip_eps must be positive".into()));
        }
        if self.ppo_epochs == 0 {
            return Err(PpoError::Config("ppo_epochs must be at least 1".into()));
        }
        if self.mini_batch_size == 0 {
            return Err(PpoError::Config(
                "mini_batch_size must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

/// A single rollout (trajectory) for PPO.
#[derive(Debug, Clone)]
pub struct Rollout {
    /// Prompt token IDs.
    pub prompt_ids: Vec<u32>,
    /// Response token IDs.
    pub response_ids: Vec<u32>,
    /// Log probabilities from the policy at collection time.
    pub old_log_probs: Vec<f32>,
    /// Values from the value head (if used).
    pub values: Vec<f32>,
    /// Rewards for each token (usually 0 except final).
    pub rewards: Vec<f32>,
    /// Attention mask.
    pub attention_mask: Vec<u32>,
    /// Whether the response was truncated.
    pub truncated: bool,
}

impl Rollout {
    /// Create a new rollout.
    pub fn new(prompt_ids: Vec<u32>) -> Self {
        Self {
            prompt_ids,
            response_ids: Vec::new(),
            old_log_probs: Vec::new(),
            values: Vec::new(),
            rewards: Vec::new(),
            attention_mask: Vec::new(),
            truncated: false,
        }
    }

    /// Total sequence length (prompt + response).
    pub fn total_length(&self) -> usize {
        self.prompt_ids.len() + self.response_ids.len()
    }

    /// Response length.
    pub fn response_length(&self) -> usize {
        self.response_ids.len()
    }
}

/// Batch of rollouts for PPO training.
#[derive(Debug)]
pub struct RolloutBatch {
    /// Input token IDs [batch, seq_len]
    pub input_ids: Array,
    /// Attention mask [batch, seq_len]
    pub attention_mask: Array,
    /// Old log probabilities [batch, response_len]
    pub old_log_probs: Array,
    /// Old values [batch, response_len]
    pub old_values: Option<Array>,
    /// Rewards [batch, response_len]
    pub rewards: Array,
    /// Advantages [batch, response_len]
    pub advantages: Array,
    /// Returns (value targets) [batch, response_len]
    pub returns: Option<Array>,
    /// Response mask [batch, response_len]
    pub response_mask: Array,
}

/// PPO trainer for reinforcement learning.
pub struct PpoTrainer {
    /// PPO configuration.
    pub config: PpoConfig,
    /// Training configuration.
    pub training_config: TrainingConfig,
    /// Current training step.
    step: usize,
    /// Adaptive KL coefficient.
    kl_coef: f64,
}

impl PpoTrainer {
    /// Create a new PPO trainer.
    pub fn new(config: PpoConfig, training_config: TrainingConfig) -> PpoResult<Self> {
        config.validate()?;
        let kl_coef = config.kl_coef;
        Ok(Self {
            config,
            training_config,
            step: 0,
            kl_coef,
        })
    }

    /// Compute per-token log probabilities.
    pub fn compute_log_probs(&self, logits: &Array, labels: &Array) -> PpoResult<Array> {
        let seq_len = logits.dim(1);

        // Shift for next-token prediction
        let pred_logits = logits.index((.., ..seq_len - 1, ..));
        let target_labels = labels.index((.., 1..));

        // Log softmax
        let log_probs = mlx_rs::nn::log_softmax(&pred_logits, -1)?;

        // Gather at target positions
        let batch_size = logits.dim(0) as usize;
        let pred_len = (seq_len - 1) as usize;

        let mut token_logps = Vec::with_capacity(batch_size * pred_len);

        log_probs.eval()?;
        target_labels.eval()?;

        for b in 0..batch_size {
            let batch_log_probs = log_probs.index(b as i32);
            let batch_labels = target_labels.index(b as i32);

            batch_log_probs.eval()?;
            batch_labels.eval()?;

            for t in 0..pred_len {
                let label = batch_labels.index(t as i32);
                label.eval()?;
                let label_idx = label.item::<i32>().max(0);
                let log_prob = batch_log_probs.index((t as i32, label_idx));
                log_prob.eval()?;
                token_logps.push(log_prob.item::<f32>());
            }
        }

        Ok(Array::from_slice(
            &token_logps,
            &[batch_size as i32, pred_len as i32],
        ))
    }

    /// Compute advantages using GAE.
    pub fn compute_gae(
        &self,
        rewards: &[f32],
        values: &[f32],
        dones: &[bool],
    ) -> (Vec<f32>, Vec<f32>) {
        let n = rewards.len();
        let mut advantages = vec![0.0f32; n];
        let mut returns = vec![0.0f32; n];

        let gamma = self.config.gae.gamma as f32;
        let lambda = self.config.gae.lambda as f32;

        // Compute GAE backwards
        let mut gae = 0.0f32;
        for t in (0..n).rev() {
            let next_value = if t == n - 1 { 0.0 } else { values[t + 1] };

            let delta = rewards[t] + gamma * next_value * (!dones[t] as i32 as f32) - values[t];
            gae = delta + gamma * lambda * (!dones[t] as i32 as f32) * gae;
            advantages[t] = gae;
            returns[t] = gae + values[t];
        }

        // Normalize advantages if configured
        if self.config.gae.normalize && n > 1 {
            let mean: f32 = advantages.iter().sum::<f32>() / n as f32;
            let var: f32 = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n as f32;
            let std = var.sqrt().max(1e-8);
            for a in &mut advantages {
                *a = (*a - mean) / std;
            }
        }

        (advantages, returns)
    }

    /// Compute PPO loss.
    ///
    /// Returns (total_loss, policy_loss, value_loss, entropy, kl)
    pub fn compute_ppo_loss(
        &self,
        new_log_probs: &Array,
        old_log_probs: &Array,
        advantages: &Array,
        new_values: Option<&Array>,
        old_values: Option<&Array>,
        returns: Option<&Array>,
        mask: &Array,
        logits: Option<&Array>,
    ) -> PpoResult<(Array, Array, Option<Array>, Option<Array>, Array)> {
        // Compute probability ratio: r = exp(new_logp - old_logp)
        let log_ratio = new_log_probs.subtract(old_log_probs)?;
        let ratio = log_ratio.exp()?;

        // Clipped objective
        let eps = Array::from_f32(self.config.clip_eps as f32);
        let one = Array::from_f32(1.0);
        let clip_low = one.subtract(&eps)?;
        let clip_high = one.add(&eps)?;
        // Use maximum/minimum to clip ratio to [1-eps, 1+eps]
        let clipped_ratio = mlx_rs::ops::maximum(&ratio, &clip_low)?;
        let clipped_ratio = mlx_rs::ops::minimum(&clipped_ratio, &clip_high)?;

        // Surrogate objectives
        let surr1 = ratio.multiply(advantages)?;
        let surr2 = clipped_ratio.multiply(advantages)?;
        let policy_loss = mlx_rs::ops::minimum(&surr1, &surr2)?;
        let masked_policy_loss = policy_loss.multiply(mask)?;
        let mean_policy_loss = masked_policy_loss
            .sum(None)?
            .divide(&mask.sum(None)?)?
            .negative()?;

        // Value loss (optional)
        let value_loss = if self.config.use_value_head {
            if let (Some(new_v), Some(old_v), Some(ret)) = (new_values, old_values, returns) {
                let vf_loss = if let Some(vf_clip) = self.config.vf_clip {
                    // Clipped value loss
                    let clip = Array::from_f32(vf_clip as f32);
                    let v_diff = new_v.subtract(old_v)?;
                    let clipped_diff_low = mlx_rs::ops::maximum(&v_diff, &clip.negative()?)?;
                    let clipped_diff = mlx_rs::ops::minimum(&clipped_diff_low, &clip)?;
                    let v_clipped = old_v.add(&clipped_diff)?;

                    let loss1 = new_v.subtract(ret)?.square()?;
                    let loss2 = v_clipped.subtract(ret)?.square()?;
                    mlx_rs::ops::maximum(&loss1, &loss2)?
                } else {
                    // Simple MSE loss
                    new_v.subtract(ret)?.square()?
                };

                let masked_vf = vf_loss.multiply(mask)?;
                let vf_coef = Array::from_f32(self.config.vf_coef as f32);
                let half = Array::from_f32(0.5);
                Some(
                    masked_vf
                        .sum(None)?
                        .divide(&mask.sum(None)?)?
                        .multiply(&vf_coef)?
                        .multiply(&half)?,
                )
            } else {
                None
            }
        } else {
            None
        };

        // Entropy bonus (optional)
        let entropy = if let Some(logits) = logits {
            if self.config.entropy_coef > 0.0 {
                let log_probs = mlx_rs::nn::log_softmax(logits, -1)?;
                let probs = log_probs.exp()?;
                let entropy_per_token =
                    probs.multiply(&log_probs)?.sum_axis(-1, None)?.negative()?;
                let entropy_coef = Array::from_f32(self.config.entropy_coef as f32);
                Some(
                    entropy_per_token
                        .multiply(mask)?
                        .sum(None)?
                        .divide(&mask.sum(None)?)?
                        .multiply(&entropy_coef)?,
                )
            } else {
                None
            }
        } else {
            None
        };

        // KL divergence (approximate)
        let kl = ratio.subtract(&one)?.subtract(&log_ratio)?;
        let mean_kl = kl.multiply(mask)?.sum(None)?.divide(&mask.sum(None)?)?;

        // Total loss
        let mut total_loss = mean_policy_loss.clone();
        if let Some(ref vl) = value_loss {
            total_loss = total_loss.add(vl)?;
        }
        if let Some(ref ent) = entropy {
            total_loss = total_loss.subtract(ent)?; // Subtract because we want to maximize entropy
        }
        if self.kl_coef > 0.0 {
            let kl_penalty = mean_kl.multiply(&Array::from_f32(self.kl_coef as f32))?;
            total_loss = total_loss.add(&kl_penalty)?;
        }

        Ok((total_loss, mean_policy_loss, value_loss, entropy, mean_kl))
    }

    /// Update adaptive KL coefficient based on measured KL.
    pub fn update_kl_coef(&mut self, kl: f32) {
        if let Some(target) = self.config.target_kl {
            // Adaptive KL penalty (from PPO paper)
            if kl > target as f32 * 1.5 {
                self.kl_coef *= 2.0;
            } else if kl < target as f32 / 1.5 {
                self.kl_coef /= 2.0;
            }
            // Clamp to reasonable range
            self.kl_coef = self.kl_coef.clamp(1e-5, 100.0);
        }
    }

    /// Prepare a batch of rollouts for training.
    pub fn prepare_batch(&self, rollouts: &[Rollout]) -> PpoResult<RolloutBatch> {
        let batch_size = rollouts.len();
        if batch_size == 0 {
            return Err(PpoError::Config("Empty rollout batch".into()));
        }

        // Find max lengths
        let max_total = rollouts.iter().map(|r| r.total_length()).max().unwrap();
        let max_response = rollouts.iter().map(|r| r.response_length()).max().unwrap();

        // Pad and collect input IDs
        let mut input_ids_data = vec![0i32; batch_size * max_total];
        let mut attn_mask_data = vec![0.0f32; batch_size * max_total];

        for (i, rollout) in rollouts.iter().enumerate() {
            let offset = i * max_total;
            for (j, &id) in rollout.prompt_ids.iter().enumerate() {
                input_ids_data[offset + j] = id as i32;
                attn_mask_data[offset + j] = 1.0;
            }
            for (j, &id) in rollout.response_ids.iter().enumerate() {
                input_ids_data[offset + rollout.prompt_ids.len() + j] = id as i32;
                attn_mask_data[offset + rollout.prompt_ids.len() + j] = 1.0;
            }
        }

        // Pad and collect response-level data
        let mut old_logps_data = vec![0.0f32; batch_size * max_response];
        let mut values_data = vec![0.0f32; batch_size * max_response];
        let mut rewards_data = vec![0.0f32; batch_size * max_response];
        let mut response_mask_data = vec![0.0f32; batch_size * max_response];

        for (i, rollout) in rollouts.iter().enumerate() {
            let offset = i * max_response;
            for (j, &lp) in rollout.old_log_probs.iter().enumerate() {
                old_logps_data[offset + j] = lp;
                response_mask_data[offset + j] = 1.0;
            }
            for (j, &v) in rollout.values.iter().enumerate() {
                values_data[offset + j] = v;
            }
            for (j, &r) in rollout.rewards.iter().enumerate() {
                rewards_data[offset + j] = r;
            }
        }

        // Compute advantages for each rollout
        let mut advantages_data = vec![0.0f32; batch_size * max_response];
        let mut returns_data = vec![0.0f32; batch_size * max_response];

        for (i, rollout) in rollouts.iter().enumerate() {
            let dones: Vec<bool> = (0..rollout.rewards.len())
                .map(|j| j == rollout.rewards.len() - 1)
                .collect();
            let (advs, rets) = self.compute_gae(&rollout.rewards, &rollout.values, &dones);

            let offset = i * max_response;
            for (j, &adv) in advs.iter().enumerate() {
                advantages_data[offset + j] = adv;
            }
            for (j, &ret) in rets.iter().enumerate() {
                returns_data[offset + j] = ret;
            }
        }

        Ok(RolloutBatch {
            input_ids: Array::from_slice(&input_ids_data, &[batch_size as i32, max_total as i32]),
            attention_mask: Array::from_slice(
                &attn_mask_data,
                &[batch_size as i32, max_total as i32],
            ),
            old_log_probs: Array::from_slice(
                &old_logps_data,
                &[batch_size as i32, max_response as i32],
            ),
            old_values: if self.config.use_value_head {
                Some(Array::from_slice(
                    &values_data,
                    &[batch_size as i32, max_response as i32],
                ))
            } else {
                None
            },
            rewards: Array::from_slice(&rewards_data, &[batch_size as i32, max_response as i32]),
            advantages: Array::from_slice(
                &advantages_data,
                &[batch_size as i32, max_response as i32],
            ),
            returns: if self.config.use_value_head {
                Some(Array::from_slice(
                    &returns_data,
                    &[batch_size as i32, max_response as i32],
                ))
            } else {
                None
            },
            response_mask: Array::from_slice(
                &response_mask_data,
                &[batch_size as i32, max_response as i32],
            ),
        })
    }

    /// Get current step.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Increment step counter.
    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    /// Get current KL coefficient.
    pub fn current_kl_coef(&self) -> f64 {
        self.kl_coef
    }
}

/// PPO training metrics.
#[derive(Debug, Clone, Default)]
pub struct PpoMetrics {
    /// Total loss.
    pub loss: f32,
    /// Policy loss.
    pub policy_loss: f32,
    /// Value loss.
    pub value_loss: f32,
    /// Entropy.
    pub entropy: f32,
    /// KL divergence.
    pub kl: f32,
    /// Mean reward.
    pub mean_reward: f32,
    /// Mean advantage.
    pub mean_advantage: f32,
    /// Clip fraction (how often ratio was clipped).
    pub clip_fraction: f32,
    /// Current KL coefficient.
    pub kl_coef: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_config_default() {
        let config = PpoConfig::default();
        assert!((config.clip_eps - 0.2).abs() < 0.01);
        assert_eq!(config.ppo_epochs, 4);
        assert!(config.use_value_head);
    }

    #[test]
    fn test_ppo_config_validation() {
        let valid = PpoConfig::default();
        assert!(valid.validate().is_ok());

        let invalid_eps = PpoConfig {
            clip_eps: 0.0,
            ..Default::default()
        };
        assert!(invalid_eps.validate().is_err());

        let invalid_epochs = PpoConfig {
            ppo_epochs: 0,
            ..Default::default()
        };
        assert!(invalid_epochs.validate().is_err());
    }

    #[test]
    fn test_gae_computation() {
        let config = PpoConfig::default();
        let training_config = TrainingConfig::default();
        let trainer = PpoTrainer::new(config, training_config).unwrap();

        let rewards = vec![0.0f32, 0.0, 0.0, 1.0];
        let values = vec![0.1, 0.2, 0.3, 0.4];
        let dones = vec![false, false, false, true];

        let (advantages, returns) = trainer.compute_gae(&rewards, &values, &dones);

        assert_eq!(advantages.len(), 4);
        assert_eq!(returns.len(), 4);

        // Last advantage should be reward - value = 1.0 - 0.4 = 0.6 (pre-normalization)
        // With normalization, values will shift
    }

    #[test]
    fn test_rollout_creation() {
        let mut rollout = Rollout::new(vec![1, 2, 3]);
        rollout.response_ids = vec![4, 5, 6, 7];
        rollout.old_log_probs = vec![-1.0, -1.5, -2.0, -1.0];
        rollout.values = vec![0.5, 0.6, 0.7, 0.8];
        rollout.rewards = vec![0.0, 0.0, 0.0, 1.0];

        assert_eq!(rollout.total_length(), 7);
        assert_eq!(rollout.response_length(), 4);
    }

    #[test]
    fn test_prepare_batch() {
        let config = PpoConfig::default();
        let training_config = TrainingConfig::default();
        let trainer = PpoTrainer::new(config, training_config).unwrap();

        let mut r1 = Rollout::new(vec![1, 2]);
        r1.response_ids = vec![3, 4];
        r1.old_log_probs = vec![-1.0, -1.5];
        r1.values = vec![0.5, 0.6];
        r1.rewards = vec![0.0, 1.0];

        let mut r2 = Rollout::new(vec![5, 6, 7]);
        r2.response_ids = vec![8];
        r2.old_log_probs = vec![-2.0];
        r2.values = vec![0.3];
        r2.rewards = vec![0.5];

        let batch = trainer.prepare_batch(&[r1, r2]).unwrap();

        // Check shapes
        assert_eq!(batch.input_ids.shape()[0], 2); // batch size
        assert_eq!(batch.old_log_probs.shape()[0], 2);
        assert_eq!(batch.advantages.shape()[0], 2);
    }

    #[test]
    fn test_ppo_loss_computation() {
        let config = PpoConfig::default();
        let training_config = TrainingConfig::default();
        let trainer = PpoTrainer::new(config, training_config).unwrap();

        let new_logps = Array::from_slice(&[-1.0f32, -1.5, -2.0, -1.8], &[2, 2]);
        let old_logps = Array::from_slice(&[-1.1f32, -1.6, -2.1, -1.9], &[2, 2]);
        let advantages = Array::from_slice(&[1.0f32, -0.5, 0.8, -0.3], &[2, 2]);
        let mask = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]);

        let (loss, policy_loss, _, _, kl) = trainer
            .compute_ppo_loss(
                &new_logps,
                &old_logps,
                &advantages,
                None,
                None,
                None,
                &mask,
                None,
            )
            .unwrap();

        loss.eval().unwrap();
        policy_loss.eval().unwrap();
        kl.eval().unwrap();

        assert!(loss.item::<f32>().is_finite());
        assert!(kl.item::<f32>() >= 0.0);
    }

    #[test]
    fn test_adaptive_kl_update() {
        let config = PpoConfig::new().with_adaptive_kl(0.1, 0.01);
        let training_config = TrainingConfig::default();
        let mut trainer = PpoTrainer::new(config, training_config).unwrap();

        let initial_coef = trainer.current_kl_coef();

        // KL too high -> increase coefficient
        trainer.update_kl_coef(0.02); // > 0.01 * 1.5
        assert!(trainer.current_kl_coef() > initial_coef);

        // Reset
        trainer.kl_coef = 0.1;

        // KL too low -> decrease coefficient
        trainer.update_kl_coef(0.005); // < 0.01 / 1.5
        assert!(trainer.current_kl_coef() < 0.1);
    }
}
