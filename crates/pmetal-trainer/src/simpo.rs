//! Simple Preference Optimization (SimPO) trainer.
//!
//! SimPO is a simpler alternative to DPO that doesn't require a reference model,
//! making it more memory-efficient and easier to implement.
//!
//! # Algorithm
//!
//! SimPO uses a length-normalized reward margin:
//! ```text
//! L_SimPO = -log(σ(β/|y_w| * log π(y_w|x) - β/|y_l| * log π(y_l|x) - γ))
//! ```
//!
//! Where:
//! - `y_w` and `y_l` are chosen and rejected responses
//! - `β` is the temperature/margin parameter
//! - `γ` is the target margin
//! - Length normalization prevents length bias
//!
//! # Key Advantages over DPO
//!
//! - **No reference model**: 50% less memory usage
//! - **Length normalization**: Prevents length exploitation
//! - **Simpler implementation**: No need to manage frozen model
//!
//! # References
//!
//! - "SimPO: Simple Preference Optimization with a Reference-Free Reward"
//!   (Meng et al., 2024)

use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use pmetal_core::TrainingConfig;

/// Error type for SimPO training.
#[derive(Debug, thiserror::Error)]
pub enum SimpoError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
}

/// Result type for SimPO operations.
pub type SimpoResult<T> = std::result::Result<T, SimpoError>;

/// SimPO loss type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimpoLossType {
    /// Standard SimPO with sigmoid.
    #[default]
    Sigmoid,
    /// Hinge-style loss.
    Hinge,
    /// IPO-style loss (squared).
    Ipo,
}

/// SimPO configuration.
#[derive(Debug, Clone)]
pub struct SimpoConfig {
    /// Temperature parameter beta. Higher = stronger preference signal.
    /// Default: 2.5 (from paper)
    pub beta: f64,

    /// Target margin gamma. Default: 0.5 (from paper)
    pub gamma: f64,

    /// Loss type. Default: Sigmoid
    pub loss_type: SimpoLossType,

    /// Whether to use length normalization. Default: true
    pub length_norm: bool,

    /// Label smoothing for soft labels. Default: 0.0
    pub label_smoothing: f64,

    /// CPO alpha for conservative update. Default: 0.0 (disabled)
    /// When > 0, adds behavior cloning loss to prevent forgetting.
    pub cpo_alpha: f64,

    /// SFT loss weight for multi-task training. Default: 0.0
    pub sft_weight: f64,

    /// Maximum sequence length.
    pub max_seq_length: usize,

    /// Prompt length (for masking).
    pub max_prompt_length: usize,
}

impl Default for SimpoConfig {
    fn default() -> Self {
        Self {
            beta: 2.5,
            gamma: 0.5,
            loss_type: SimpoLossType::Sigmoid,
            length_norm: true,
            label_smoothing: 0.0,
            cpo_alpha: 0.0,
            sft_weight: 0.0,
            max_seq_length: 1024,
            max_prompt_length: 512,
        }
    }
}

impl SimpoConfig {
    /// Create a new SimPO config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set beta (temperature).
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set gamma (margin).
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set loss type.
    pub fn with_loss_type(mut self, loss_type: SimpoLossType) -> Self {
        self.loss_type = loss_type;
        self
    }

    /// Disable length normalization.
    pub fn without_length_norm(mut self) -> Self {
        self.length_norm = false;
        self
    }

    /// Enable CPO (Conservative Policy Optimization) mode.
    pub fn with_cpo(mut self, alpha: f64) -> Self {
        self.cpo_alpha = alpha;
        self
    }

    /// Enable SFT auxiliary loss.
    pub fn with_sft_loss(mut self, weight: f64) -> Self {
        self.sft_weight = weight;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> SimpoResult<()> {
        if self.beta <= 0.0 {
            return Err(SimpoError::Config("beta must be positive".into()));
        }
        if self.label_smoothing < 0.0 || self.label_smoothing > 0.5 {
            return Err(SimpoError::Config(
                "label_smoothing must be in [0, 0.5]".into(),
            ));
        }
        Ok(())
    }
}

/// A preference pair for SimPO training.
#[derive(Debug, Clone)]
pub struct PreferencePair {
    /// Prompt token IDs.
    pub prompt_ids: Vec<u32>,
    /// Chosen (preferred) response token IDs.
    pub chosen_ids: Vec<u32>,
    /// Rejected response token IDs.
    pub rejected_ids: Vec<u32>,
    /// Attention mask for chosen.
    pub chosen_mask: Vec<u32>,
    /// Attention mask for rejected.
    pub rejected_mask: Vec<u32>,
}

impl PreferencePair {
    /// Create a new preference pair.
    pub fn new(prompt_ids: Vec<u32>, chosen_ids: Vec<u32>, rejected_ids: Vec<u32>) -> Self {
        let chosen_mask = vec![1u32; chosen_ids.len()];
        let rejected_mask = vec![1u32; rejected_ids.len()];
        Self {
            prompt_ids,
            chosen_ids,
            rejected_ids,
            chosen_mask,
            rejected_mask,
        }
    }

    /// Get chosen response length.
    pub fn chosen_length(&self) -> usize {
        self.chosen_ids.len()
    }

    /// Get rejected response length.
    pub fn rejected_length(&self) -> usize {
        self.rejected_ids.len()
    }
}

/// SimPO trainer.
pub struct SimpoTrainer {
    /// Configuration.
    pub config: SimpoConfig,
    /// Training configuration.
    pub training_config: TrainingConfig,
    /// Current step.
    step: usize,
}

impl SimpoTrainer {
    /// Create a new SimPO trainer.
    pub fn new(config: SimpoConfig, training_config: TrainingConfig) -> SimpoResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            training_config,
            step: 0,
        })
    }

    /// Compute per-token log probabilities.
    pub fn compute_log_probs(
        &self,
        logits: &Array,
        labels: &Array,
        mask: &Array,
    ) -> SimpoResult<Array> {
        let seq_len = logits.dim(1);

        // Shift for next-token prediction
        let pred_logits = logits.index((.., ..seq_len - 1, ..));
        let target_labels = labels.index((.., 1..));
        let target_mask = mask.index((.., 1..));

        // Selective log softmax: gather logit first, subtract logsumexp
        // Never materializes full [B, S, V] log_softmax tensor
        let (logps_array, _valid_mask) =
            crate::logprob_utils::selective_log_softmax(&pred_logits, &target_labels)?;

        // Apply the caller-provided mask (may differ from label-based mask)
        let target_mask_f32 = target_mask.as_dtype(mlx_rs::Dtype::Float32)?;
        let masked_logps = logps_array.multiply(&target_mask_f32)?;

        Ok(masked_logps)
    }

    /// Compute sequence-level reward (length-normalized or summed).
    pub fn compute_rewards(&self, per_token_logps: &Array, mask: &Array) -> SimpoResult<Array> {
        let masked_logps = per_token_logps.multiply(mask)?;
        let sum_logps = masked_logps.sum_axis(-1, None)?;

        if self.config.length_norm {
            // Length-normalized: avg log prob
            let lengths = mask.sum_axis(-1, None)?;
            let eps = Array::from_f32(1e-8);
            let lengths_safe = lengths.add(&eps)?;
            Ok(sum_logps.divide(&lengths_safe)?)
        } else {
            // Simple sum
            Ok(sum_logps)
        }
    }

    /// Compute SimPO loss.
    ///
    /// Returns (loss, chosen_rewards, rejected_rewards, margin)
    pub fn compute_simpo_loss(
        &self,
        chosen_logps: &Array,
        rejected_logps: &Array,
        chosen_mask: &Array,
        rejected_mask: &Array,
    ) -> SimpoResult<(Array, Array, Array, Array)> {
        // Compute length-normalized rewards
        let chosen_rewards = self.compute_rewards(chosen_logps, chosen_mask)?;
        let rejected_rewards = self.compute_rewards(rejected_logps, rejected_mask)?;

        // Scale by beta
        let beta = Array::from_f32(self.config.beta as f32);
        let chosen_scaled = chosen_rewards.multiply(&beta)?;
        let rejected_scaled = rejected_rewards.multiply(&beta)?;

        // Compute margin: chosen - rejected - gamma
        let gamma = Array::from_f32(self.config.gamma as f32);
        let margin = chosen_scaled.subtract(&rejected_scaled)?.subtract(&gamma)?;

        // Compute loss based on type
        let loss = match self.config.loss_type {
            SimpoLossType::Sigmoid => {
                // -log(sigmoid(margin))
                let logsigmoid = mlx_rs::nn::log_sigmoid(&margin)?;
                logsigmoid.negative()?
            }
            SimpoLossType::Hinge => {
                // max(0, 1 - margin)
                let one = Array::from_f32(1.0);
                let hinge = one.subtract(&margin)?;
                let zero = Array::from_f32(0.0);
                mlx_rs::ops::maximum(&hinge, &zero)?
            }
            SimpoLossType::Ipo => {
                // (margin - 1)^2
                let one = Array::from_f32(1.0);
                margin.subtract(&one)?.square()?
            }
        };

        // Apply label smoothing if configured
        let loss = if self.config.label_smoothing > 0.0 {
            let smooth = Array::from_f32(self.config.label_smoothing as f32);
            let one_minus_smooth = Array::from_f32(1.0 - self.config.label_smoothing as f32);

            // Smoothed loss: (1-ε)*loss + ε*flipped_loss
            let flipped_margin = rejected_scaled.subtract(&chosen_scaled)?.subtract(&gamma)?;
            let flipped_loss = match self.config.loss_type {
                SimpoLossType::Sigmoid => mlx_rs::nn::log_sigmoid(&flipped_margin)?.negative()?,
                SimpoLossType::Hinge => {
                    let one = Array::from_f32(1.0);
                    let hinge = one.subtract(&flipped_margin)?;
                    let zero = Array::from_f32(0.0);
                    mlx_rs::ops::maximum(&hinge, &zero)?
                }
                SimpoLossType::Ipo => {
                    let one = Array::from_f32(1.0);
                    flipped_margin.subtract(&one)?.square()?
                }
            };

            loss.multiply(&one_minus_smooth)?
                .add(&flipped_loss.multiply(&smooth)?)?
        } else {
            loss
        };

        // Mean loss
        let mean_loss = loss.mean(None)?;

        Ok((mean_loss, chosen_rewards, rejected_rewards, margin))
    }

    /// Compute SimPO loss with optional CPO regularization.
    pub fn compute_loss_with_cpo(
        &self,
        chosen_logps: &Array,
        rejected_logps: &Array,
        chosen_mask: &Array,
        rejected_mask: &Array,
        ref_chosen_logps: Option<&Array>,
    ) -> SimpoResult<(Array, SimpoMetrics)> {
        // Base SimPO loss
        let (simpo_loss, chosen_rewards, rejected_rewards, margin) =
            self.compute_simpo_loss(chosen_logps, rejected_logps, chosen_mask, rejected_mask)?;

        let mut total_loss = simpo_loss.clone();

        // CPO regularization: KL divergence penalty to stay close to reference
        let cpo_loss = if self.config.cpo_alpha > 0.0 {
            if let Some(ref_logps) = ref_chosen_logps {
                // KL(π || π_ref) = E_π[log π - log π_ref]
                // = chosen_logps - ref_logps (when sampling from π)
                // Minimizing this encourages policy to stay close to reference
                let kl = chosen_logps.subtract(ref_logps)?;
                let masked_kl = kl.multiply(chosen_mask)?;
                let mean_kl = masked_kl.sum(None)?.divide(&chosen_mask.sum(None)?)?;
                let alpha = Array::from_f32(self.config.cpo_alpha as f32);
                let cpo = mean_kl.multiply(&alpha)?;
                total_loss = total_loss.add(&cpo)?;
                Some(cpo)
            } else {
                None
            }
        } else {
            None
        };

        // SFT auxiliary loss
        let sft_loss = if self.config.sft_weight > 0.0 {
            // Negative log likelihood on chosen
            let masked_nll = chosen_logps.negative()?.multiply(chosen_mask)?;
            let mean_nll = masked_nll.sum(None)?.divide(&chosen_mask.sum(None)?)?;
            let weight = Array::from_f32(self.config.sft_weight as f32);
            let sft = mean_nll.multiply(&weight)?;
            total_loss = total_loss.add(&sft)?;
            Some(sft)
        } else {
            None
        };

        // Evaluate for metrics
        total_loss.eval()?;
        simpo_loss.eval()?;
        chosen_rewards.eval()?;
        rejected_rewards.eval()?;
        margin.eval()?;

        let metrics = SimpoMetrics {
            loss: total_loss.item::<f32>(),
            simpo_loss: simpo_loss.item::<f32>(),
            cpo_loss: cpo_loss
                .map(|l| {
                    l.eval().ok();
                    l.item::<f32>()
                })
                .unwrap_or(0.0),
            sft_loss: sft_loss
                .map(|l| {
                    l.eval().ok();
                    l.item::<f32>()
                })
                .unwrap_or(0.0),
            chosen_reward: chosen_rewards
                .mean(None)
                .ok()
                .map(|m| {
                    m.eval().ok();
                    m.item::<f32>()
                })
                .unwrap_or(0.0),
            rejected_reward: rejected_rewards
                .mean(None)
                .ok()
                .map(|m| {
                    m.eval().ok();
                    m.item::<f32>()
                })
                .unwrap_or(0.0),
            margin: margin
                .mean(None)
                .ok()
                .map(|m| {
                    m.eval().ok();
                    m.item::<f32>()
                })
                .unwrap_or(0.0),
            accuracy: (margin
                .gt(&Array::from_f32(0.0))
                .ok()
                .map(|m| {
                    m.as_dtype(mlx_rs::Dtype::Float32)
                        .ok()
                        .and_then(|m| m.mean(None).ok())
                        .map(|m| {
                            m.eval().ok();
                            m.item::<f32>()
                        })
                        .unwrap_or(0.0)
                })
                .unwrap_or(0.0)),
        };

        Ok((total_loss, metrics))
    }

    /// Get current step.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Increment step.
    pub fn increment_step(&mut self) {
        self.step += 1;
    }
}

/// SimPO training metrics.
#[derive(Debug, Clone, Default)]
pub struct SimpoMetrics {
    /// Total loss.
    pub loss: f32,
    /// SimPO preference loss.
    pub simpo_loss: f32,
    /// CPO regularization loss.
    pub cpo_loss: f32,
    /// SFT auxiliary loss.
    pub sft_loss: f32,
    /// Mean chosen reward.
    pub chosen_reward: f32,
    /// Mean rejected reward.
    pub rejected_reward: f32,
    /// Mean margin (chosen - rejected - gamma).
    pub margin: f32,
    /// Accuracy (fraction where margin > 0).
    pub accuracy: f32,
}

impl SimpoMetrics {
    /// Reward margin (chosen - rejected).
    pub fn reward_margin(&self) -> f32 {
        self.chosen_reward - self.rejected_reward
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simpo_config_default() {
        let config = SimpoConfig::default();
        assert!((config.beta - 2.5).abs() < 0.01);
        assert!((config.gamma - 0.5).abs() < 0.01);
        assert!(config.length_norm);
    }

    #[test]
    fn test_simpo_config_validation() {
        let valid = SimpoConfig::default();
        assert!(valid.validate().is_ok());

        let invalid_beta = SimpoConfig {
            beta: 0.0,
            ..Default::default()
        };
        assert!(invalid_beta.validate().is_err());

        let invalid_smooth = SimpoConfig {
            label_smoothing: 0.6,
            ..Default::default()
        };
        assert!(invalid_smooth.validate().is_err());
    }

    #[test]
    fn test_preference_pair() {
        let pair = PreferencePair::new(vec![1, 2, 3], vec![4, 5, 6], vec![7, 8]);

        assert_eq!(pair.chosen_length(), 3);
        assert_eq!(pair.rejected_length(), 2);
    }

    #[test]
    fn test_simpo_loss_basic() {
        let config = SimpoConfig::default();
        let training_config = TrainingConfig::default();
        let trainer = SimpoTrainer::new(config, training_config).unwrap();

        // Mock log probabilities [batch=2, seq=3]
        let chosen_logps = Array::from_slice(&[-1.0f32, -1.5, -2.0, -0.5, -1.0, -1.5], &[2, 3]);
        let rejected_logps = Array::from_slice(&[-2.0f32, -2.5, -3.0, -1.5, -2.0, -2.5], &[2, 3]);
        let chosen_mask = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]);
        let rejected_mask = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]);

        let (loss, chosen_rewards, rejected_rewards, margin) = trainer
            .compute_simpo_loss(&chosen_logps, &rejected_logps, &chosen_mask, &rejected_mask)
            .unwrap();

        loss.eval().unwrap();
        chosen_rewards.eval().unwrap();
        rejected_rewards.eval().unwrap();
        margin.eval().unwrap();

        // Loss should be finite
        assert!(loss.item::<f32>().is_finite());

        // Chosen should have higher rewards than rejected
        let chosen_mean = chosen_rewards.mean(None).unwrap();
        let rejected_mean = rejected_rewards.mean(None).unwrap();
        chosen_mean.eval().unwrap();
        rejected_mean.eval().unwrap();
        assert!(chosen_mean.item::<f32>() > rejected_mean.item::<f32>());
    }

    #[test]
    fn test_simpo_loss_hinge() {
        let config = SimpoConfig::new().with_loss_type(SimpoLossType::Hinge);
        let training_config = TrainingConfig::default();
        let trainer = SimpoTrainer::new(config, training_config).unwrap();

        let chosen_logps = Array::from_slice(&[-1.0f32, -1.0], &[1, 2]);
        let rejected_logps = Array::from_slice(&[-2.0f32, -2.0], &[1, 2]);
        let mask = Array::from_slice(&[1.0f32, 1.0], &[1, 2]);

        let (loss, _, _, _) = trainer
            .compute_simpo_loss(&chosen_logps, &rejected_logps, &mask, &mask)
            .unwrap();

        loss.eval().unwrap();
        assert!(loss.item::<f32>() >= 0.0); // Hinge loss is non-negative
    }

    #[test]
    fn test_simpo_with_cpo() {
        let config = SimpoConfig::new().with_cpo(0.1);
        let training_config = TrainingConfig::default();
        let trainer = SimpoTrainer::new(config, training_config).unwrap();

        let chosen_logps = Array::from_slice(&[-1.0f32, -1.5], &[1, 2]);
        let rejected_logps = Array::from_slice(&[-2.0f32, -2.5], &[1, 2]);
        let ref_logps = Array::from_slice(&[-1.1f32, -1.6], &[1, 2]);
        let mask = Array::from_slice(&[1.0f32, 1.0], &[1, 2]);

        let (loss, metrics) = trainer
            .compute_loss_with_cpo(
                &chosen_logps,
                &rejected_logps,
                &mask,
                &mask,
                Some(&ref_logps),
            )
            .unwrap();

        loss.eval().unwrap();
        assert!(loss.item::<f32>().is_finite());
        assert!(metrics.cpo_loss.is_finite());
    }

    #[test]
    fn test_simpo_with_sft() {
        let config = SimpoConfig::new().with_sft_loss(0.1);
        let training_config = TrainingConfig::default();
        let trainer = SimpoTrainer::new(config, training_config).unwrap();

        let chosen_logps = Array::from_slice(&[-1.0f32, -1.5], &[1, 2]);
        let rejected_logps = Array::from_slice(&[-2.0f32, -2.5], &[1, 2]);
        let mask = Array::from_slice(&[1.0f32, 1.0], &[1, 2]);

        let (loss, metrics) = trainer
            .compute_loss_with_cpo(&chosen_logps, &rejected_logps, &mask, &mask, None)
            .unwrap();

        loss.eval().unwrap();
        assert!(loss.item::<f32>().is_finite());
        assert!(metrics.sft_loss.is_finite());
    }

    #[test]
    fn test_length_normalization() {
        let config = SimpoConfig::default();
        let training_config = TrainingConfig::default();
        let trainer = SimpoTrainer::new(config, training_config).unwrap();

        // Same total log prob, different lengths
        let logps1 = Array::from_slice(&[-1.0f32, -1.0, -1.0, -1.0], &[1, 4]); // sum = -4, len = 4
        let logps2 = Array::from_slice(&[-2.0f32, -2.0], &[1, 2]); // sum = -4, len = 2

        let mask1 = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 4]);
        let mask2 = Array::from_slice(&[1.0f32, 1.0], &[1, 2]);

        let reward1 = trainer.compute_rewards(&logps1, &mask1).unwrap();
        let reward2 = trainer.compute_rewards(&logps2, &mask2).unwrap();

        reward1.eval().unwrap();
        reward2.eval().unwrap();

        // With length norm: -4/4 = -1 vs -4/2 = -2
        // So reward1 > reward2
        assert!(reward1.item::<f32>() > reward2.item::<f32>());
    }

    #[test]
    fn test_simpo_metrics() {
        let config = SimpoConfig::default();
        let training_config = TrainingConfig::default();
        let trainer = SimpoTrainer::new(config, training_config).unwrap();

        let chosen_logps = Array::from_slice(&[-0.5f32, -0.5], &[1, 2]);
        let rejected_logps = Array::from_slice(&[-2.0f32, -2.0], &[1, 2]);
        let mask = Array::from_slice(&[1.0f32, 1.0], &[1, 2]);

        let (_, metrics) = trainer
            .compute_loss_with_cpo(&chosen_logps, &rejected_logps, &mask, &mask, None)
            .unwrap();

        // With such a clear preference, accuracy should be high
        assert!(metrics.accuracy >= 0.5);
        assert!(metrics.reward_margin() > 0.0);
    }
}
