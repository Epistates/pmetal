//! Kahneman-Tversky Optimization (KTO) trainer.
//!
//! KTO is a preference learning algorithm based on Kahneman & Tversky's prospect
//! theory. Unlike DPO which requires paired preferences, KTO only needs binary
//! desirable/undesirable labels for each response.
//!
//! Based on: "KTO: Model Alignment as Prospect Theoretic Optimization"
//! by Ethayarajh et al. (arXiv:2402.01306)
//!
//! # Key Advantages
//!
//! - **No paired data required**: Each example is independently labeled as desirable
//!   or undesirable, making data collection easier.
//! - **Loss aversion**: Based on prospect theory, KTO naturally handles the asymmetry
//!   between gains (desirable) and losses (undesirable).
//! - **Competitive performance**: Matches or exceeds DPO from 1B to 30B scale.
//!
//! # Loss Function
//!
//! For desirable examples (y_d):
//! ```text
//! L_desirable = λ_D * (1 - σ(β * (log π(y_d|x) - log π_ref(y_d|x)) - z_ref))
//! ```
//!
//! For undesirable examples (y_u):
//! ```text
//! L_undesirable = λ_U * (1 - σ(z_ref - β * (log π(y_u|x) - log π_ref(y_u|x))))
//! ```
//!
//! Where:
//! - `σ` is the sigmoid function
//! - `β` is the temperature parameter (controls sensitivity to preferences)
//! - `z_ref` is the KL divergence baseline (typically 0 or estimated from data)
//! - `λ_D`, `λ_U` are weights for desirable/undesirable (default both 1.0)
//!
//! # Example
//!
//! ```ignore
//! use pmetal_trainer::{KtoConfig, KtoTrainer, KtoSample};
//!
//! let config = KtoConfig::new(0.1);
//! let trainer = KtoTrainer::new(config, training_config)?;
//!
//! // Create samples (no pairing needed!)
//! let desirable = KtoSample::desirable(prompt_ids, response_ids);
//! let undesirable = KtoSample::undesirable(prompt_ids, response_ids);
//! ```

use pmetal_core::TrainingConfig;
use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

/// Error type for KTO training.
#[derive(Debug, thiserror::Error)]
pub enum KtoError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Data error.
    #[error("Data error: {0}")]
    Data(String),
}

/// Result type for KTO operations.
pub type KtoResult<T> = std::result::Result<T, KtoError>;

/// KTO loss variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KtoLossType {
    /// Standard KTO loss (default).
    #[default]
    Standard,
    /// KTO with BCO (Binary Classifier Optimization) baseline.
    /// Estimates z_ref from the batch rather than using a fixed value.
    Bco,
    /// Asymmetric KTO with different beta for desirable/undesirable.
    Asymmetric,
}

/// KTO configuration.
#[derive(Debug, Clone)]
pub struct KtoConfig {
    /// Beta parameter controlling preference strength.
    /// Higher values make the model more sensitive to preferences.
    /// Typical range: 0.1 to 0.5. Default: 0.1
    pub beta: f64,

    /// Beta for desirable examples (only used with Asymmetric loss type).
    /// If None, uses `beta` for both.
    pub beta_desirable: Option<f64>,

    /// Beta for undesirable examples (only used with Asymmetric loss type).
    /// If None, uses `beta` for both.
    pub beta_undesirable: Option<f64>,

    /// Loss function type.
    pub loss_type: KtoLossType,

    /// Weight for desirable examples (λ_D).
    /// Default: 1.0
    pub desirable_weight: f64,

    /// Weight for undesirable examples (λ_U).
    /// Prospect theory suggests losses are felt ~2x as strongly as gains.
    /// Default: 1.0 (set to 2.0 for loss aversion)
    pub undesirable_weight: f64,

    /// Reference point z_ref for the KL baseline.
    /// Default: 0.0 (no baseline)
    pub z_ref: f64,

    /// If true, estimate z_ref from the batch (BCO-style).
    /// Overrides the fixed z_ref value.
    pub estimate_z_ref: bool,

    /// If true, don't use a reference model (faster but may be less stable).
    /// Sets reference log probs to zero.
    pub reference_free: bool,

    /// Maximum length for prompt tokens.
    pub max_prompt_length: usize,

    /// Maximum length for response tokens.
    pub max_completion_length: usize,

    /// Whether to truncate prompts from the left (keeps recent context).
    pub truncate_prompt_left: bool,
}

impl Default for KtoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            beta_desirable: None,
            beta_undesirable: None,
            loss_type: KtoLossType::Standard,
            desirable_weight: 1.0,
            undesirable_weight: 1.0,
            z_ref: 0.0,
            estimate_z_ref: false,
            reference_free: false,
            max_prompt_length: 512,
            max_completion_length: 512,
            truncate_prompt_left: true,
        }
    }
}

impl KtoConfig {
    /// Create a new KTO config with the given beta.
    pub fn new(beta: f64) -> Self {
        Self {
            beta,
            ..Default::default()
        }
    }

    /// Set the loss type.
    pub fn with_loss_type(mut self, loss_type: KtoLossType) -> Self {
        self.loss_type = loss_type;
        self
    }

    /// Set asymmetric betas.
    pub fn with_asymmetric_beta(mut self, desirable: f64, undesirable: f64) -> Self {
        self.loss_type = KtoLossType::Asymmetric;
        self.beta_desirable = Some(desirable);
        self.beta_undesirable = Some(undesirable);
        self
    }

    /// Set the weight for desirable examples.
    pub fn with_desirable_weight(mut self, weight: f64) -> Self {
        self.desirable_weight = weight;
        self
    }

    /// Set the weight for undesirable examples.
    pub fn with_undesirable_weight(mut self, weight: f64) -> Self {
        self.undesirable_weight = weight;
        self
    }

    /// Enable loss aversion weighting (undesirable_weight = 2.0).
    /// Based on prospect theory: losses are felt ~2x as strongly as gains.
    pub fn with_loss_aversion(mut self) -> Self {
        self.undesirable_weight = 2.0;
        self
    }

    /// Set the KL baseline (z_ref).
    pub fn with_z_ref(mut self, z_ref: f64) -> Self {
        self.z_ref = z_ref;
        self
    }

    /// Enable BCO-style z_ref estimation from batch.
    pub fn with_bco_estimation(mut self) -> Self {
        self.loss_type = KtoLossType::Bco;
        self.estimate_z_ref = true;
        self
    }

    /// Set reference-free mode.
    pub fn reference_free(mut self) -> Self {
        self.reference_free = true;
        self
    }

    /// Get the effective beta for desirable examples.
    pub fn effective_beta_desirable(&self) -> f64 {
        self.beta_desirable.unwrap_or(self.beta)
    }

    /// Get the effective beta for undesirable examples.
    pub fn effective_beta_undesirable(&self) -> f64 {
        self.beta_undesirable.unwrap_or(self.beta)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> KtoResult<()> {
        if self.beta <= 0.0 {
            return Err(KtoError::Config("KTO beta must be positive".into()));
        }

        if let Some(beta_d) = self.beta_desirable {
            if beta_d <= 0.0 {
                return Err(KtoError::Config(
                    "KTO beta_desirable must be positive".into(),
                ));
            }
        }

        if let Some(beta_u) = self.beta_undesirable {
            if beta_u <= 0.0 {
                return Err(KtoError::Config(
                    "KTO beta_undesirable must be positive".into(),
                ));
            }
        }

        if self.desirable_weight < 0.0 {
            return Err(KtoError::Config(
                "KTO desirable_weight must be non-negative".into(),
            ));
        }

        if self.undesirable_weight < 0.0 {
            return Err(KtoError::Config(
                "KTO undesirable_weight must be non-negative".into(),
            ));
        }

        Ok(())
    }
}

/// A single sample for KTO training.
/// Unlike DPO, each sample is independently labeled - no pairing required.
#[derive(Debug, Clone)]
pub struct KtoSample {
    /// Prompt/context tokens.
    pub prompt_ids: Vec<u32>,
    /// Response tokens (full sequence: prompt + response).
    pub response_ids: Vec<u32>,
    /// Attention mask for the full sequence.
    pub attention_mask: Vec<u32>,
    /// Labels (masked prompt with -100, then response tokens).
    pub labels: Vec<i64>,
    /// Whether this is a desirable (true) or undesirable (false) response.
    pub is_desirable: bool,
}

impl KtoSample {
    /// Create a KTO sample from prompt and response with desirability label.
    ///
    /// Labels are constructed with -100 for prompt tokens (ignored in loss)
    /// and actual token IDs for response tokens.
    pub fn new(prompt_ids: Vec<u32>, response_ids: Vec<u32>, is_desirable: bool) -> Self {
        let prompt_len = prompt_ids.len();

        // Build full sequence: prompt + response
        let mut full_seq: Vec<u32> = prompt_ids.clone();
        full_seq.extend(&response_ids);
        let attention_mask = vec![1u32; full_seq.len()];

        // Labels: -100 for prompt, actual IDs for response
        let mut labels: Vec<i64> = vec![-100i64; prompt_len];
        labels.extend(response_ids.iter().map(|&id| id as i64));

        Self {
            prompt_ids,
            response_ids: full_seq,
            attention_mask,
            labels,
            is_desirable,
        }
    }

    /// Create a desirable sample.
    pub fn desirable(prompt_ids: Vec<u32>, response_ids: Vec<u32>) -> Self {
        Self::new(prompt_ids, response_ids, true)
    }

    /// Create an undesirable sample.
    pub fn undesirable(prompt_ids: Vec<u32>, response_ids: Vec<u32>) -> Self {
        Self::new(prompt_ids, response_ids, false)
    }
}

/// KTO trainer for preference learning.
pub struct KtoTrainer {
    /// KTO configuration.
    pub config: KtoConfig,
    /// Training configuration.
    pub training_config: TrainingConfig,
    /// Current training step.
    step: usize,
    /// Running estimate of z_ref (for BCO mode).
    z_ref_estimate: f64,
    /// Number of samples used in z_ref estimation.
    z_ref_count: usize,
}

impl KtoTrainer {
    /// Create a new KTO trainer.
    pub fn new(config: KtoConfig, training_config: TrainingConfig) -> KtoResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            training_config,
            step: 0,
            z_ref_estimate: 0.0,
            z_ref_count: 0,
        })
    }

    /// Compute log probabilities for a sequence.
    ///
    /// # Arguments
    /// * `logits` - Model output logits [batch, seq_len, vocab_size]
    /// * `labels` - Target labels [batch, seq_len] (-100 for ignored positions)
    ///
    /// # Returns
    /// Sum of log probabilities for non-ignored tokens [batch]
    ///
    /// Note: Uses optimized batched operations where possible, falling back to
    /// per-batch processing for the gather operation.
    pub fn compute_log_probs(&self, logits: &Array, labels: &Array) -> KtoResult<Array> {
        // Validate inputs
        let logits_shape = logits.shape();
        let labels_shape = labels.shape();

        if logits_shape.len() != 3 {
            return Err(KtoError::Data(format!(
                "Expected 3D logits [batch, seq, vocab], got shape {:?}",
                logits_shape
            )));
        }

        if labels_shape.len() != 2 {
            return Err(KtoError::Data(format!(
                "Expected 2D labels [batch, seq], got shape {:?}",
                labels_shape
            )));
        }

        let batch_size = logits_shape[0];
        let seq_len = logits_shape[1];
        let vocab_size = logits_shape[2];

        if seq_len <= 1 {
            return Err(KtoError::Data(
                "Sequence length must be > 1 for next-token prediction".into(),
            ));
        }

        // Shift logits and labels for next-token prediction
        // logits[:, :-1, :] -> predict next token
        let pred_logits = logits.index((.., ..seq_len - 1, ..));

        // labels[:, 1:] -> target is next token
        let target_labels = labels.index((.., 1..));

        // Compute log softmax (vectorized across entire batch)
        let log_probs = mlx_rs::nn::log_softmax(&pred_logits, -1)?;
        log_probs.eval()?;

        let pred_len = (seq_len - 1) as usize;

        // Create mask for valid (non -100) labels
        let ignore_index = Array::from_int(-100);
        let valid_mask = target_labels.ne(&ignore_index)?;
        valid_mask.eval()?;

        // Replace -100 with 0 for gathering (will be masked out anyway)
        let gather_labels = mlx_rs::ops::maximum(&target_labels, &Array::from_int(0))?;
        gather_labels.eval()?;

        // Process each batch - gather operation still requires per-element access
        // but we batch the eval calls for better GPU utilization
        let mut total_log_probs = Vec::with_capacity(batch_size as usize);

        for b in 0..batch_size {
            let batch_log_probs = log_probs.index(b);
            let batch_labels = gather_labels.index(b);
            let batch_mask = valid_mask.index(b);

            // Batch eval for this sample
            batch_log_probs.eval()?;
            batch_labels.eval()?;
            batch_mask.eval()?;

            let mut sum = 0.0f32;

            for t in 0..pred_len {
                let mask_val = batch_mask.index(t as i32);
                mask_val.eval()?;

                if mask_val.item::<bool>() {
                    let label = batch_labels.index(t as i32);
                    label.eval()?;
                    let label_idx = label.item::<i32>();

                    // Bounds check for vocab index
                    if label_idx < 0 || label_idx >= vocab_size {
                        return Err(KtoError::Data(format!(
                            "Invalid token ID {} at batch {}, position {}. Vocab size is {}",
                            label_idx, b, t, vocab_size
                        )));
                    }

                    let log_prob = batch_log_probs.index((t as i32, label_idx));
                    log_prob.eval()?;
                    sum += log_prob.item::<f32>();
                }
            }
            total_log_probs.push(sum);
        }

        Ok(Array::from_slice(&total_log_probs, &[batch_size]))
    }

    /// Compute KTO loss for a batch of samples.
    ///
    /// # Arguments
    /// * `policy_logps` - Log probs from policy model [batch]
    /// * `ref_logps` - Log probs from reference model [batch]
    /// * `is_desirable` - Boolean mask indicating desirable samples [batch]
    ///
    /// # Returns
    /// (loss, rewards, desirable_loss, undesirable_loss)
    pub fn compute_kto_loss(
        &mut self,
        policy_logps: &Array,
        ref_logps: &Array,
        is_desirable: &[bool],
    ) -> KtoResult<KtoLossOutput> {
        let batch_size = policy_logps.dim(0) as usize;

        // Compute implicit rewards: r = log π(y|x) - log π_ref(y|x)
        let rewards = if self.config.reference_free {
            policy_logps.clone()
        } else {
            policy_logps.subtract(ref_logps)?
        };

        rewards.eval()?;

        // Get z_ref (either fixed or estimated)
        let z_ref = if self.config.estimate_z_ref {
            // BCO-style: estimate from batch
            self.update_z_ref_estimate(&rewards)?;
            self.z_ref_estimate
        } else {
            self.config.z_ref
        };

        // Compute loss for each sample based on desirability
        let mut losses = Vec::with_capacity(batch_size);
        let mut desirable_losses = Vec::new();
        let mut undesirable_losses = Vec::new();

        let beta_d = self.config.effective_beta_desirable() as f32;
        let beta_u = self.config.effective_beta_undesirable() as f32;
        let lambda_d = self.config.desirable_weight as f32;
        let lambda_u = self.config.undesirable_weight as f32;

        for i in 0..batch_size {
            let reward = rewards.index(i as i32);
            reward.eval()?;
            let r = reward.item::<f32>();

            let loss = if is_desirable[i] {
                // Desirable: λ_D * (1 - σ(β_d * r - z_ref))
                // = λ_D * σ(z_ref - β_d * r)
                let logit = (z_ref as f32) - beta_d * r;
                let sigmoid_val = 1.0 / (1.0 + (-logit).exp());
                let l = lambda_d * sigmoid_val;
                desirable_losses.push(l);
                l
            } else {
                // Undesirable: λ_U * (1 - σ(z_ref - β_u * r))
                // = λ_U * σ(β_u * r - z_ref)
                let logit = beta_u * r - (z_ref as f32);
                let sigmoid_val = 1.0 / (1.0 + (-logit).exp());
                let l = lambda_u * sigmoid_val;
                undesirable_losses.push(l);
                l
            };

            losses.push(loss);
        }

        // Create loss array
        let loss_array = Array::from_slice(&losses, &[batch_size as i32]);
        let mean_loss = loss_array.mean(None)?;
        mean_loss.eval()?;

        // Compute scaled rewards for logging
        let beta = Array::from_f32(self.config.beta as f32);
        let scaled_rewards = rewards.multiply(&beta)?;

        Ok(KtoLossOutput {
            loss: mean_loss,
            rewards: scaled_rewards,
            z_ref,
            desirable_loss: if desirable_losses.is_empty() {
                0.0
            } else {
                desirable_losses.iter().sum::<f32>() / desirable_losses.len() as f32
            },
            undesirable_loss: if undesirable_losses.is_empty() {
                0.0
            } else {
                undesirable_losses.iter().sum::<f32>() / undesirable_losses.len() as f32
            },
        })
    }

    /// Update the running estimate of z_ref (for BCO mode).
    fn update_z_ref_estimate(&mut self, rewards: &Array) -> KtoResult<()> {
        let batch_size = rewards.dim(0) as usize;

        // Compute mean KL divergence (reward) for this batch
        let mean_reward = rewards.mean(None)?;
        mean_reward.eval()?;
        let batch_mean = mean_reward.item::<f32>() as f64;

        // Exponential moving average
        let alpha = 0.1; // smoothing factor
        if self.z_ref_count == 0 {
            self.z_ref_estimate = batch_mean;
        } else {
            self.z_ref_estimate = alpha * batch_mean + (1.0 - alpha) * self.z_ref_estimate;
        }
        self.z_ref_count += batch_size;

        Ok(())
    }

    /// Get current z_ref value (estimated or fixed).
    pub fn z_ref(&self) -> f64 {
        if self.config.estimate_z_ref {
            self.z_ref_estimate
        } else {
            self.config.z_ref
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
}

/// Output from KTO loss computation.
#[derive(Debug)]
pub struct KtoLossOutput {
    /// The computed loss.
    pub loss: Array,
    /// Scaled rewards (β * log_ratio).
    pub rewards: Array,
    /// The z_ref value used.
    pub z_ref: f64,
    /// Average loss for desirable samples.
    pub desirable_loss: f32,
    /// Average loss for undesirable samples.
    pub undesirable_loss: f32,
}

/// Compute KTO metrics for logging.
#[derive(Debug, Clone)]
pub struct KtoMetrics {
    /// KTO loss value.
    pub loss: f32,
    /// Average reward across all samples.
    pub mean_reward: f32,
    /// Average reward for desirable samples.
    pub desirable_reward: f32,
    /// Average reward for undesirable samples.
    pub undesirable_reward: f32,
    /// z_ref baseline value.
    pub z_ref: f32,
    /// Loss for desirable samples.
    pub desirable_loss: f32,
    /// Loss for undesirable samples.
    pub undesirable_loss: f32,
    /// Number of desirable samples in batch.
    pub num_desirable: usize,
    /// Number of undesirable samples in batch.
    pub num_undesirable: usize,
}

impl KtoMetrics {
    /// Compute metrics from rewards and desirability labels.
    pub fn compute(
        loss: f32,
        rewards: &[f32],
        is_desirable: &[bool],
        z_ref: f32,
        desirable_loss: f32,
        undesirable_loss: f32,
    ) -> Self {
        let mut desirable_rewards = Vec::new();
        let mut undesirable_rewards = Vec::new();

        for (r, &d) in rewards.iter().zip(is_desirable.iter()) {
            if d {
                desirable_rewards.push(*r);
            } else {
                undesirable_rewards.push(*r);
            }
        }

        let mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let desirable_reward = if desirable_rewards.is_empty() {
            0.0
        } else {
            desirable_rewards.iter().sum::<f32>() / desirable_rewards.len() as f32
        };
        let undesirable_reward = if undesirable_rewards.is_empty() {
            0.0
        } else {
            undesirable_rewards.iter().sum::<f32>() / undesirable_rewards.len() as f32
        };

        Self {
            loss,
            mean_reward,
            desirable_reward,
            undesirable_reward,
            z_ref,
            desirable_loss,
            undesirable_loss,
            num_desirable: desirable_rewards.len(),
            num_undesirable: undesirable_rewards.len(),
        }
    }
}

/// KTO dataset format specification.
/// Describes the expected format for KTO training data.
#[derive(Debug, Clone)]
pub struct KtoDatasetFormat {
    /// Field name for the prompt.
    pub prompt_field: String,
    /// Field name for the response/completion.
    pub response_field: String,
    /// Field name for the desirability label.
    pub label_field: String,
    /// Value indicating desirable (e.g., "good", "preferred", "true", "1").
    pub desirable_value: String,
}

impl Default for KtoDatasetFormat {
    fn default() -> Self {
        Self {
            prompt_field: "prompt".to_string(),
            response_field: "completion".to_string(),
            label_field: "label".to_string(),
            desirable_value: "desirable".to_string(),
        }
    }
}

impl KtoDatasetFormat {
    /// Create a format specification for HuggingFace UltraFeedback-style datasets.
    pub fn ultrafeedback() -> Self {
        Self {
            prompt_field: "instruction".to_string(),
            response_field: "output".to_string(),
            label_field: "rating".to_string(),
            desirable_value: "good".to_string(),
        }
    }

    /// Create a format specification for Anthropic HH-style datasets.
    pub fn anthropic_hh() -> Self {
        Self {
            prompt_field: "prompt".to_string(),
            response_field: "response".to_string(),
            label_field: "chosen".to_string(),
            desirable_value: "true".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kto_config_default() {
        let config = KtoConfig::default();
        assert_eq!(config.beta, 0.1);
        assert_eq!(config.loss_type, KtoLossType::Standard);
        assert_eq!(config.desirable_weight, 1.0);
        assert_eq!(config.undesirable_weight, 1.0);
        assert_eq!(config.z_ref, 0.0);
        assert!(!config.reference_free);
        assert!(!config.estimate_z_ref);
    }

    #[test]
    fn test_kto_config_validation() {
        let config = KtoConfig::new(0.1);
        assert!(config.validate().is_ok());

        let invalid = KtoConfig::new(-0.1);
        assert!(invalid.validate().is_err());

        let mut invalid_weight = KtoConfig::new(0.1);
        invalid_weight.desirable_weight = -1.0;
        assert!(invalid_weight.validate().is_err());
    }

    #[test]
    fn test_kto_config_loss_aversion() {
        let config = KtoConfig::new(0.1).with_loss_aversion();
        assert_eq!(config.desirable_weight, 1.0);
        assert_eq!(config.undesirable_weight, 2.0);
    }

    #[test]
    fn test_kto_config_asymmetric() {
        let config = KtoConfig::new(0.1).with_asymmetric_beta(0.05, 0.2);
        assert_eq!(config.loss_type, KtoLossType::Asymmetric);
        assert_eq!(config.effective_beta_desirable(), 0.05);
        assert_eq!(config.effective_beta_undesirable(), 0.2);
    }

    #[test]
    fn test_kto_sample_creation() {
        let prompt = vec![1, 2, 3];
        let response = vec![4, 5];

        let desirable = KtoSample::desirable(prompt.clone(), response.clone());
        assert!(desirable.is_desirable);
        assert_eq!(desirable.prompt_ids, vec![1, 2, 3]);
        assert_eq!(desirable.response_ids, vec![1, 2, 3, 4, 5]); // prompt + response
        assert_eq!(desirable.labels, vec![-100, -100, -100, 4, 5]);

        let undesirable = KtoSample::undesirable(prompt.clone(), response.clone());
        assert!(!undesirable.is_desirable);
    }

    #[test]
    fn test_kto_trainer_creation() {
        let config = KtoConfig::new(0.1);
        let training_config = TrainingConfig::default();
        let trainer = KtoTrainer::new(config, training_config);
        assert!(trainer.is_ok());

        let trainer = trainer.unwrap();
        assert_eq!(trainer.current_step(), 0);
        assert_eq!(trainer.z_ref(), 0.0);
    }

    #[test]
    fn test_kto_loss_computation() {
        let config = KtoConfig::new(0.1);
        let training_config = TrainingConfig::default();
        let mut trainer = KtoTrainer::new(config, training_config).unwrap();

        // Test loss computation with mock log probs
        let policy_logps = Array::from_slice(&[-1.0f32, -2.0, -1.5, -2.5], &[4]);
        let ref_logps = Array::from_slice(&[-1.5f32, -1.5, -2.0, -2.0], &[4]);
        let is_desirable = vec![true, true, false, false];

        let output = trainer
            .compute_kto_loss(&policy_logps, &ref_logps, &is_desirable)
            .unwrap();

        output.loss.eval().unwrap();

        // Loss should be positive
        assert!(output.loss.item::<f32>() > 0.0);
        assert_eq!(output.z_ref, 0.0);
    }

    #[test]
    fn test_kto_loss_reference_free() {
        let config = KtoConfig::new(0.1).reference_free();
        let training_config = TrainingConfig::default();
        let mut trainer = KtoTrainer::new(config, training_config).unwrap();

        let policy_logps = Array::from_slice(&[-1.0f32, -2.0], &[2]);
        let ref_logps = Array::from_slice(&[-1.5f32, -1.5], &[2]); // Should be ignored
        let is_desirable = vec![true, false];

        let output = trainer
            .compute_kto_loss(&policy_logps, &ref_logps, &is_desirable)
            .unwrap();

        output.loss.eval().unwrap();
        assert!(output.loss.item::<f32>() > 0.0);
    }

    #[test]
    fn test_kto_bco_estimation() {
        let config = KtoConfig::new(0.1).with_bco_estimation();
        let training_config = TrainingConfig::default();
        let mut trainer = KtoTrainer::new(config, training_config).unwrap();

        assert!(trainer.config.estimate_z_ref);
        assert_eq!(trainer.z_ref_estimate, 0.0);

        let policy_logps = Array::from_slice(&[-1.0f32, -2.0, -1.5], &[3]);
        let ref_logps = Array::from_slice(&[-1.5f32, -1.5, -2.0], &[3]);
        let is_desirable = vec![true, true, false];

        let _ = trainer
            .compute_kto_loss(&policy_logps, &ref_logps, &is_desirable)
            .unwrap();

        // z_ref should now be estimated (non-zero after update)
        // The exact value depends on the rewards
    }

    #[test]
    fn test_kto_metrics() {
        let rewards = vec![1.0f32, 0.5, -0.5, -1.0];
        let is_desirable = vec![true, true, false, false];

        let metrics = KtoMetrics::compute(0.1, &rewards, &is_desirable, 0.0, 0.05, 0.15);

        assert_eq!(metrics.loss, 0.1);
        assert_eq!(metrics.num_desirable, 2);
        assert_eq!(metrics.num_undesirable, 2);
        assert!((metrics.desirable_reward - 0.75).abs() < 0.01); // (1.0 + 0.5) / 2
        assert!((metrics.undesirable_reward - (-0.75)).abs() < 0.01); // (-0.5 + -1.0) / 2
    }

    #[test]
    fn test_kto_dataset_formats() {
        let default = KtoDatasetFormat::default();
        assert_eq!(default.prompt_field, "prompt");
        assert_eq!(default.label_field, "label");

        let uf = KtoDatasetFormat::ultrafeedback();
        assert_eq!(uf.prompt_field, "instruction");
        assert_eq!(uf.response_field, "output");

        let hh = KtoDatasetFormat::anthropic_hh();
        assert_eq!(hh.prompt_field, "prompt");
        assert_eq!(hh.label_field, "chosen");
    }

    #[test]
    fn test_kto_step_counter() {
        let config = KtoConfig::new(0.1);
        let training_config = TrainingConfig::default();
        let mut trainer = KtoTrainer::new(config, training_config).unwrap();

        assert_eq!(trainer.current_step(), 0);
        trainer.increment_step();
        assert_eq!(trainer.current_step(), 1);
        trainer.increment_step();
        assert_eq!(trainer.current_step(), 2);
    }
}
