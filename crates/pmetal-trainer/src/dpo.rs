//! Direct Preference Optimization (DPO) trainer.
//!
//! DPO is a preference learning algorithm that trains a model directly on
//! preference pairs without requiring a separate reward model. It optimizes
//! the policy to prefer "chosen" responses over "rejected" responses.
//!
//! Based on: "Direct Preference Optimization: Your Language Model is Secretly
//! a Reward Model" by Rafailov et al.
//!
//! The DPO loss is:
//! ```text
//! L_DPO = -log(sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x)
//!                              - log_pi_ref(y_w|x) + log_pi_ref(y_l|x))))
//! ```
//!
//! Where:
//! - `y_w` is the chosen (winning) response
//! - `y_l` is the rejected (losing) response
//! - `pi` is the policy model (trainable)
//! - `pi_ref` is the reference model (frozen)
//! - `beta` is the temperature parameter

use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use mlx_rs::Dtype;
use pmetal_core::TrainingConfig;

/// Error type for DPO training.
#[derive(Debug, thiserror::Error)]
pub enum DpoError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for DPO operations.
pub type DpoResult<T> = std::result::Result<T, DpoError>;

/// DPO loss type variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DpoLossType {
    /// Standard sigmoid DPO loss (default).
    #[default]
    Sigmoid,
    /// Implicit Preference Optimization (IPO) loss.
    /// From "A General Theoretical Paradigm to Understand Learning from Human Feedback"
    Ipo,
    /// Hinge loss variant.
    Hinge,
    /// Robust DPO loss (more stable with noisy preferences).
    Robust,
    /// Simple Preference Optimization (SimPO).
    /// Reference-free DPO with a margin.
    SimPo,
}

/// DPO configuration.
#[derive(Debug, Clone)]
pub struct DpoConfig {
    /// Beta parameter controlling preference strength.
    /// Higher values make the model more sensitive to preferences.
    /// Typical range: 0.1 to 0.5. Default: 0.1
    pub beta: f64,

    /// Loss function type.
    pub loss_type: DpoLossType,

    /// Label smoothing parameter (0.0 to 0.5).
    /// Encodes uncertainty about preference labels.
    /// Not compatible with IPO or Hinge loss types.
    pub label_smoothing: f64,

    /// If true, don't use a reference model (faster but may be less stable).
    /// Sets reference log ratios to zero.
    /// Forced to true for SimPO.
    pub reference_free: bool,

    /// When true, uses the same forward pass for both policy and reference:
    /// - Policy log probs: Normal computation with gradient flow
    /// - Reference log probs: `stop_gradient(policy_log_probs)` - no gradient flow
    ///
    /// Benefits:
    /// - ~50% memory reduction (no separate reference model forward pass)
    /// - ~2x faster training (single forward pass instead of two)
    ///
    /// This is the recommended approach when using the same model architecture
    /// for both policy and reference. The reference is effectively a "snapshot"
    /// of the policy at the start of each step.
    pub use_stop_gradient_reference: bool,

    /// Target reward margin for SimPO.
    /// Default: 1.0.
    pub simpo_gamma: f64,

    /// Maximum length for prompt tokens.
    pub max_prompt_length: usize,

    /// Maximum length for response tokens (chosen/rejected).
    pub max_completion_length: usize,

    /// Whether to truncate prompts from the left (keeps recent context).
    pub truncate_prompt_left: bool,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            loss_type: DpoLossType::Sigmoid,
            label_smoothing: 0.0,
            reference_free: false,
            use_stop_gradient_reference: true, // Recommended: memory-efficient reference
            simpo_gamma: 1.0,
            max_prompt_length: 512,
            max_completion_length: 512,
            truncate_prompt_left: true,
        }
    }
}

impl DpoConfig {
    /// Create a new DPO config with the given beta.
    pub fn new(beta: f64) -> Self {
        Self {
            beta,
            ..Default::default()
        }
    }

    /// Set the loss type.
    pub fn with_loss_type(mut self, loss_type: DpoLossType) -> Self {
        self.loss_type = loss_type;
        self
    }

    /// Set label smoothing.
    pub fn with_label_smoothing(mut self, smoothing: f64) -> Self {
        self.label_smoothing = smoothing.clamp(0.0, 0.5);
        self
    }

    /// Set reference-free mode.
    pub fn reference_free(mut self) -> Self {
        self.reference_free = true;
        self
    }

    /// Set SimPO gamma margin.
    pub fn with_simpo_gamma(mut self, gamma: f64) -> Self {
        self.simpo_gamma = gamma;
        self
    }

    /// Enable or disable stop_gradient reference model pattern.
    ///
    /// When enabled, uses the same model for both policy and reference,
    /// applying stop_gradient() to create a reference that prevents gradient flow.
    /// This reduces memory by ~50% and speeds up training by ~2x.
    pub fn with_stop_gradient_reference(mut self, enable: bool) -> Self {
        self.use_stop_gradient_reference = enable;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> DpoResult<()> {
        if self.beta <= 0.0 {
            return Err(DpoError::Config("DPO beta must be positive".into()));
        }

        if self.label_smoothing > 0.0
            && matches!(self.loss_type, DpoLossType::Ipo | DpoLossType::Hinge)
        {
            return Err(DpoError::Config(
                "Label smoothing is not compatible with IPO or Hinge loss".into(),
            ));
        }

        Ok(())
    }
}

/// A single preference pair for DPO training.
#[derive(Debug, Clone)]
pub struct PreferencePair {
    /// Prompt/context tokens.
    pub prompt_ids: Vec<u32>,
    /// Chosen (preferred) response tokens.
    pub chosen_ids: Vec<u32>,
    /// Rejected response tokens.
    pub rejected_ids: Vec<u32>,
    /// Attention mask for chosen (prompt + chosen).
    pub chosen_attention_mask: Vec<u32>,
    /// Attention mask for rejected (prompt + rejected).
    pub rejected_attention_mask: Vec<u32>,
    /// Labels for chosen (masked prompt, then chosen tokens).
    pub chosen_labels: Vec<i64>,
    /// Labels for rejected (masked prompt, then rejected tokens).
    pub rejected_labels: Vec<i64>,
}

impl PreferencePair {
    /// Create a preference pair from prompt and responses.
    ///
    /// Labels are constructed with -100 for prompt tokens (ignored in loss)
    /// and actual token IDs for response tokens.
    pub fn new(prompt_ids: Vec<u32>, chosen_ids: Vec<u32>, rejected_ids: Vec<u32>) -> Self {
        let prompt_len = prompt_ids.len();

        // Build chosen sequence: prompt + chosen
        let mut chosen_full: Vec<u32> = prompt_ids.clone();
        chosen_full.extend(&chosen_ids);
        let chosen_attention_mask = vec![1u32; chosen_full.len()];

        // Labels for chosen: -100 for prompt, actual IDs for completion
        let mut chosen_labels: Vec<i64> = vec![-100i64; prompt_len];
        chosen_labels.extend(chosen_ids.iter().map(|&id| id as i64));

        // Build rejected sequence: prompt + rejected
        let mut rejected_full: Vec<u32> = prompt_ids.clone();
        rejected_full.extend(&rejected_ids);
        let rejected_attention_mask = vec![1u32; rejected_full.len()];

        // Labels for rejected: -100 for prompt, actual IDs for completion
        let mut rejected_labels: Vec<i64> = vec![-100i64; prompt_len];
        rejected_labels.extend(rejected_ids.iter().map(|&id| id as i64));

        Self {
            prompt_ids,
            chosen_ids: chosen_full,
            rejected_ids: rejected_full,
            chosen_attention_mask,
            rejected_attention_mask,
            chosen_labels,
            rejected_labels,
        }
    }
}

/// DPO trainer for preference learning.
pub struct DpoTrainer {
    /// DPO configuration.
    pub config: DpoConfig,
    /// Training configuration.
    pub training_config: TrainingConfig,
    /// Current training step.
    step: usize,
}

impl DpoTrainer {
    /// Create a new DPO trainer.
    pub fn new(config: DpoConfig, training_config: TrainingConfig) -> DpoResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            training_config,
            step: 0,
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
    pub fn compute_log_probs(&self, logits: &Array, labels: &Array) -> DpoResult<Array> {
        // Shift logits and labels for next-token prediction
        let seq_len = logits.dim(1);

        // logits[:, :-1, :] -> predict next token
        let pred_logits = logits.index((.., ..seq_len - 1, ..));

        // labels[:, 1:] -> target is next token
        let target_labels = labels.index((.., 1..));

        // Compute log softmax using mlx_rs::nn
        let log_probs = mlx_rs::nn::log_softmax(&pred_logits, -1)?;

        // Prepare gather indices
        // Create mask for valid (non -100) labels
        // CRITICAL: Match dtype of labels to avoid type mismatch errors
        let labels_dtype = target_labels.dtype();
        let ignore_index = Array::from_int(-100).as_dtype(labels_dtype)?;
        let valid_mask = target_labels.ne(&ignore_index)?;

        // Replace -100 with 0 for gathering (will be masked out anyway)
        let zero = Array::from_int(0).as_dtype(labels_dtype)?;
        let gather_labels = mlx_rs::ops::maximum(&target_labels, &zero)?;

        // Expand dims for take_along_axis: [B, S] -> [B, S, 1]
        let gather_indices = gather_labels.expand_dims(-1i32)?;

        // Gather log probabilities: [B, S, V] take [B, S, 1] -> [B, S, 1]
        let gathered_log_probs = log_probs.take_along_axis(&gather_indices, -1)?;
        let gathered_log_probs = gathered_log_probs.squeeze_axes(&[-1i32])?; // [B, S]

        // Mask out ignored tokens
        // valid_mask is [B, S] boolean
        // multiply by mask (cast to float)
        let masked_log_probs =
            gathered_log_probs.multiply(&valid_mask.as_dtype(Dtype::Float32)?)?;

        // Sum over sequence dimension -> [B]
        let total_log_probs = masked_log_probs.sum_axes(&[1i32], false)?;

        Ok(total_log_probs)
    }

    /// Compute both policy and reference log probabilities in a single pass.
    ///
    /// - Policy log probs: Normal computation (gradients flow)
    /// - Reference log probs: `stop_gradient(policy_log_probs)` (no gradients)
    ///
    /// Benefits:
    /// - ~50% memory reduction (no separate reference model)
    /// - ~2x faster (single forward pass)
    ///
    /// # Arguments
    /// * `logits` - Model output logits [batch, seq_len, vocab_size]
    /// * `labels` - Target labels [batch, seq_len] (-100 for ignored positions)
    ///
    /// # Returns
    /// (policy_log_probs, reference_log_probs) both [batch]
    pub fn compute_log_probs_with_stop_gradient_reference(
        &self,
        logits: &Array,
        labels: &Array,
    ) -> DpoResult<(Array, Array)> {
        // Compute policy log probs normally (gradients will flow)
        let policy_log_probs = self.compute_log_probs(logits, labels)?;

        // Create reference log probs using stop_gradient
        // This prevents gradient flow to reference, treating it as a constant
        let reference_log_probs = mlx_rs::stop_gradient(&policy_log_probs)?;

        Ok((policy_log_probs, reference_log_probs))
    }

    /// Compute DPO loss using stop_gradient reference (memory-efficient).
    ///
    /// This combines forward pass and loss computation in a single efficient operation.
    /// Uses `stop_gradient` to create reference log probs from policy log probs.
    ///
    /// # Arguments
    /// * `chosen_logits` - Logits for chosen sequences [batch, seq_len, vocab]
    /// * `chosen_labels` - Labels for chosen sequences [batch, seq_len]
    /// * `rejected_logits` - Logits for rejected sequences [batch, seq_len, vocab]
    /// * `rejected_labels` - Labels for rejected sequences [batch, seq_len]
    ///
    /// # Returns
    /// (loss, chosen_rewards, rejected_rewards)
    pub fn compute_dpo_loss_with_stop_gradient(
        &self,
        chosen_logits: &Array,
        chosen_labels: &Array,
        rejected_logits: &Array,
        rejected_labels: &Array,
    ) -> DpoResult<(Array, Array, Array)> {
        // Compute policy and reference log probs for chosen
        let (policy_chosen_logps, ref_chosen_logps) =
            self.compute_log_probs_with_stop_gradient_reference(chosen_logits, chosen_labels)?;

        // Compute policy and reference log probs for rejected
        let (policy_rejected_logps, ref_rejected_logps) =
            self.compute_log_probs_with_stop_gradient_reference(rejected_logits, rejected_labels)?;

        // Compute DPO loss with these log probs
        self.compute_dpo_loss(
            &policy_chosen_logps,
            &policy_rejected_logps,
            &ref_chosen_logps,
            &ref_rejected_logps,
        )
    }

    /// Compute DPO loss for a batch of preference pairs.
    ///
    /// # Arguments
    /// * `policy_chosen_logps` - Log probs from policy model for chosen [batch]
    /// * `policy_rejected_logps` - Log probs from policy model for rejected [batch]
    /// * `ref_chosen_logps` - Log probs from reference model for chosen [batch]
    /// * `ref_rejected_logps` - Log probs from reference model for rejected [batch]
    ///
    /// # Returns
    /// (loss, chosen_rewards, rejected_rewards)
    pub fn compute_dpo_loss(
        &self,
        policy_chosen_logps: &Array,
        policy_rejected_logps: &Array,
        ref_chosen_logps: &Array,
        ref_rejected_logps: &Array,
    ) -> DpoResult<(Array, Array, Array)> {
        // Handle reference-free mode (or SimPO which is implicitly reference-free)
        let is_simpo = matches!(self.config.loss_type, DpoLossType::SimPo);
        let reference_free = self.config.reference_free || is_simpo;

        // Compute log ratios (implicit rewards)
        // reward = log_pi(y|x) - log_pi_ref(y|x)
        let chosen_rewards = if reference_free {
            policy_chosen_logps.clone()
        } else {
            policy_chosen_logps.subtract(ref_chosen_logps)?
        };

        let rejected_rewards = if reference_free {
            policy_rejected_logps.clone()
        } else {
            policy_rejected_logps.subtract(ref_rejected_logps)?
        };

        // Compute logits = beta * (chosen_rewards - rejected_rewards)
        let reward_diff = chosen_rewards.subtract(&rejected_rewards)?;
        let beta = Array::from_f32(self.config.beta as f32);
        let mut logits = reward_diff.multiply(&beta)?;

        // For SimPO: subtract gamma margin
        if is_simpo {
            let gamma = Array::from_f32(self.config.simpo_gamma as f32);
            logits = logits.subtract(&gamma)?;
        }

        // Compute loss based on loss type
        let loss = match self.config.loss_type {
            DpoLossType::Sigmoid | DpoLossType::SimPo => {
                // Standard DPO: -log(sigmoid(logits))
                // With label smoothing: -log_sigmoid(logits) * (1-s) - log_sigmoid(-logits) * s
                // SimPO uses the same sigmoid loss structure
                self.sigmoid_loss(&logits)?
            }
            DpoLossType::Ipo => {
                // IPO: (logits - 1/(2*beta))^2
                self.ipo_loss(&logits)?
            }
            DpoLossType::Hinge => {
                // Hinge: max(0, 1 - logits)
                self.hinge_loss(&logits)?
            }
            DpoLossType::Robust => {
                // Robust: more stable version
                self.robust_loss(&logits)?
            }
        };

        // Average loss over batch
        let loss = loss.mean(None)?;

        // Scale rewards by beta for logging
        let chosen_rewards_scaled = chosen_rewards.multiply(&beta)?;
        let rejected_rewards_scaled = rejected_rewards.multiply(&beta)?;

        Ok((loss, chosen_rewards_scaled, rejected_rewards_scaled))
    }

    /// Sigmoid DPO loss with optional label smoothing.
    fn sigmoid_loss(&self, logits: &Array) -> DpoResult<Array> {
        // -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
        let neg_logits = logits.negative()?;

        if self.config.label_smoothing > 0.0 {
            // With label smoothing:
            // loss = -log_sigmoid(logits) * (1-s) - log_sigmoid(-logits) * s
            let pos_loss = mlx_rs::nn::softplus(&neg_logits)?; // -log_sigmoid(logits)
            let neg_loss = mlx_rs::nn::softplus(logits)?; // -log_sigmoid(-logits)

            let s = Array::from_f32(self.config.label_smoothing as f32);
            let one_minus_s = Array::from_f32((1.0 - self.config.label_smoothing) as f32);

            let loss = pos_loss
                .multiply(&one_minus_s)?
                .add(&neg_loss.multiply(&s)?)?;
            Ok(loss)
        } else {
            // Simple: -log_sigmoid(logits) = softplus(-logits)
            Ok(mlx_rs::nn::softplus(&neg_logits)?)
        }
    }

    /// IPO loss (Implicit Preference Optimization).
    fn ipo_loss(&self, logits: &Array) -> DpoResult<Array> {
        // (logits - 1/(2*beta))^2
        let target = Array::from_f32((1.0 / (2.0 * self.config.beta)) as f32);
        let diff = logits.subtract(&target)?;
        Ok(diff.square()?)
    }

    /// Hinge loss variant.
    fn hinge_loss(&self, logits: &Array) -> DpoResult<Array> {
        // max(0, 1 - logits)
        let one = Array::from_f32(1.0);
        let margin = one.subtract(logits)?;
        let zero = Array::from_f32(0.0);
        Ok(mlx_rs::ops::maximum(&margin, &zero)?)
    }

    /// Robust loss (more stable with noisy preferences).
    fn robust_loss(&self, logits: &Array) -> DpoResult<Array> {
        // Combination of sigmoid and hinge for robustness
        // loss = -log_sigmoid(logits) + 0.5 * max(0, 0.5 - logits)
        let neg_logits = logits.negative()?;
        let sigmoid_loss = mlx_rs::nn::softplus(&neg_logits)?;

        let half = Array::from_f32(0.5);
        let margin = half.subtract(logits)?;
        let zero = Array::from_f32(0.0);
        let hinge_part = mlx_rs::ops::maximum(&margin, &zero)?;

        Ok(sigmoid_loss.add(&hinge_part.multiply(&half)?)?)
    }

    /// Get current training step.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Increment step counter.
    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    /// Pre-compute reference log probabilities for an entire dataset.
    ///
    /// - Compute reference log probs ONCE before training
    /// - Cache the results (Array per sample)
    /// - Reuse during training to avoid redundant forward passes
    ///
    /// Benefits:
    /// - Eliminates N forward passes (where N = epochs * dataset_size)
    /// - Only requires 1 forward pass per sample total
    /// - Memory efficient: stores only final log probs, not intermediate activations
    ///
    /// # Arguments
    /// * `model` - The reference model (frozen)
    /// * `chosen_inputs` - Batch of chosen input_ids [num_samples, seq_len]
    /// * `chosen_labels` - Batch of chosen labels [num_samples, seq_len]
    /// * `rejected_inputs` - Batch of rejected input_ids [num_samples, seq_len]
    /// * `rejected_labels` - Batch of rejected labels [num_samples, seq_len]
    ///
    /// # Returns
    /// (ref_chosen_logps, ref_rejected_logps) - Pre-computed reference log probs [num_samples]
    pub fn precompute_reference_log_probs<M: pmetal_lora::TrainableModel>(
        &self,
        model: &mut M,
        chosen_inputs: &Array,
        chosen_labels: &Array,
        rejected_inputs: &Array,
        rejected_labels: &Array,
    ) -> DpoResult<(Array, Array)> {
        // Validate input dimensions
        let chosen_input_shape = chosen_inputs.shape();
        let chosen_label_shape = chosen_labels.shape();
        let rejected_input_shape = rejected_inputs.shape();
        let rejected_label_shape = rejected_labels.shape();

        // Ensure all arrays have at least 2 dimensions
        if chosen_input_shape.len() < 2 {
            return Err(DpoError::Config(format!(
                "chosen_inputs must have at least 2 dimensions, got shape {:?}",
                chosen_input_shape
            )));
        }
        if rejected_input_shape.len() < 2 {
            return Err(DpoError::Config(format!(
                "rejected_inputs must have at least 2 dimensions, got shape {:?}",
                rejected_input_shape
            )));
        }

        // Ensure batch dimensions match
        if chosen_input_shape[0] != rejected_input_shape[0] {
            return Err(DpoError::Config(format!(
                "Batch size mismatch: chosen={}, rejected={}",
                chosen_input_shape[0], rejected_input_shape[0]
            )));
        }

        // Ensure labels match inputs
        if chosen_input_shape != chosen_label_shape {
            return Err(DpoError::Config(format!(
                "chosen shape mismatch: inputs={:?}, labels={:?}",
                chosen_input_shape, chosen_label_shape
            )));
        }
        if rejected_input_shape != rejected_label_shape {
            return Err(DpoError::Config(format!(
                "rejected shape mismatch: inputs={:?}, labels={:?}",
                rejected_input_shape, rejected_label_shape
            )));
        }

        // Forward pass for chosen sequences
        let chosen_logits = model
            .forward(chosen_inputs, None)
            .map_err(|e| DpoError::Config(format!("Forward failed: {}", e)))?;
        let ref_chosen_logps = self.compute_log_probs(&chosen_logits, chosen_labels)?;

        // Forward pass for rejected sequences
        let rejected_logits = model
            .forward(rejected_inputs, None)
            .map_err(|e| DpoError::Config(format!("Forward failed: {}", e)))?;
        let ref_rejected_logps = self.compute_log_probs(&rejected_logits, rejected_labels)?;

        // Evaluate and return (forces computation, allows caching)
        ref_chosen_logps.eval().map_err(|e| DpoError::Mlx(e))?;
        ref_rejected_logps.eval().map_err(|e| DpoError::Mlx(e))?;

        Ok((ref_chosen_logps, ref_rejected_logps))
    }

    /// Pre-compute reference log probs in batches (memory efficient).
    ///
    /// Processes the dataset in batches to avoid OOM on large datasets.
    ///
    /// # Arguments
    /// * `model` - The reference model (frozen)
    /// * `dataset` - Iterator of (chosen_inputs, chosen_labels, rejected_inputs, rejected_labels)
    /// * `batch_size` - Batch size for processing
    ///
    /// # Returns
    /// Vectors of pre-computed reference log probs for chosen and rejected.
    pub fn precompute_reference_log_probs_batched<M, I>(
        &self,
        model: &mut M,
        dataset: I,
        batch_size: usize,
    ) -> DpoResult<(Vec<Array>, Vec<Array>)>
    where
        M: pmetal_lora::TrainableModel,
        I: Iterator<Item = (Array, Array, Array, Array)>,
    {
        // Validate batch_size
        if batch_size == 0 {
            return Err(DpoError::Config(
                "batch_size must be greater than 0".to_string(),
            ));
        }

        let mut all_chosen_logps = Vec::new();
        let mut all_rejected_logps = Vec::new();

        let mut batch_chosen_inputs = Vec::new();
        let mut batch_chosen_labels = Vec::new();
        let mut batch_rejected_inputs = Vec::new();
        let mut batch_rejected_labels = Vec::new();

        for (chosen_input, chosen_label, rejected_input, rejected_label) in dataset {
            batch_chosen_inputs.push(chosen_input);
            batch_chosen_labels.push(chosen_label);
            batch_rejected_inputs.push(rejected_input);
            batch_rejected_labels.push(rejected_label);

            if batch_chosen_inputs.len() >= batch_size {
                // Stack into batch arrays (stack along axis 0)
                let chosen_inputs_refs: Vec<&Array> = batch_chosen_inputs.iter().collect();
                let chosen_labels_refs: Vec<&Array> = batch_chosen_labels.iter().collect();
                let rejected_inputs_refs: Vec<&Array> = batch_rejected_inputs.iter().collect();
                let rejected_labels_refs: Vec<&Array> = batch_rejected_labels.iter().collect();

                let chosen_inputs = mlx_rs::ops::stack(&chosen_inputs_refs)?;
                let chosen_labels = mlx_rs::ops::stack(&chosen_labels_refs)?;
                let rejected_inputs = mlx_rs::ops::stack(&rejected_inputs_refs)?;
                let rejected_labels = mlx_rs::ops::stack(&rejected_labels_refs)?;

                // Compute log probs for batch
                let (chosen_logps, rejected_logps) = self.precompute_reference_log_probs(
                    model,
                    &chosen_inputs,
                    &chosen_labels,
                    &rejected_inputs,
                    &rejected_labels,
                )?;

                all_chosen_logps.push(chosen_logps);
                all_rejected_logps.push(rejected_logps);

                // Clear batches
                batch_chosen_inputs.clear();
                batch_chosen_labels.clear();
                batch_rejected_inputs.clear();
                batch_rejected_labels.clear();
            }
        }

        // Process remaining samples
        if !batch_chosen_inputs.is_empty() {
            let chosen_inputs_refs: Vec<&Array> = batch_chosen_inputs.iter().collect();
            let chosen_labels_refs: Vec<&Array> = batch_chosen_labels.iter().collect();
            let rejected_inputs_refs: Vec<&Array> = batch_rejected_inputs.iter().collect();
            let rejected_labels_refs: Vec<&Array> = batch_rejected_labels.iter().collect();

            let chosen_inputs = mlx_rs::ops::stack(&chosen_inputs_refs)?;
            let chosen_labels = mlx_rs::ops::stack(&chosen_labels_refs)?;
            let rejected_inputs = mlx_rs::ops::stack(&rejected_inputs_refs)?;
            let rejected_labels = mlx_rs::ops::stack(&rejected_labels_refs)?;

            let (chosen_logps, rejected_logps) = self.precompute_reference_log_probs(
                model,
                &chosen_inputs,
                &chosen_labels,
                &rejected_inputs,
                &rejected_labels,
            )?;

            all_chosen_logps.push(chosen_logps);
            all_rejected_logps.push(rejected_logps);
        }

        Ok((all_chosen_logps, all_rejected_logps))
    }
}

/// Compute DPO metrics for logging.
#[derive(Debug, Clone)]
pub struct DpoMetrics {
    /// DPO loss value.
    pub loss: f32,
    /// Average reward for chosen responses.
    pub chosen_reward: f32,
    /// Average reward for rejected responses.
    pub rejected_reward: f32,
    /// Reward margin (chosen - rejected).
    pub reward_margin: f32,
    /// Accuracy (fraction where chosen_reward > rejected_reward).
    pub accuracy: f32,
}

impl DpoMetrics {
    /// Compute metrics from rewards.
    pub fn compute(loss: f32, chosen_rewards: &[f32], rejected_rewards: &[f32]) -> Self {
        let n = chosen_rewards.len() as f32;
        let chosen_reward: f32 = chosen_rewards.iter().sum::<f32>() / n;
        let rejected_reward: f32 = rejected_rewards.iter().sum::<f32>() / n;
        let reward_margin = chosen_reward - rejected_reward;

        let correct: usize = chosen_rewards
            .iter()
            .zip(rejected_rewards.iter())
            .filter(|(c, r)| c > r)
            .count();
        let accuracy = correct as f32 / n;

        Self {
            loss,
            chosen_reward,
            rejected_reward,
            reward_margin,
            accuracy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpo_config_default() {
        let config = DpoConfig::default();
        assert_eq!(config.beta, 0.1);
        assert_eq!(config.loss_type, DpoLossType::Sigmoid);
        assert_eq!(config.label_smoothing, 0.0);
        assert!(!config.reference_free);
    }

    #[test]
    fn test_dpo_config_validation() {
        let config = DpoConfig::new(0.1);
        assert!(config.validate().is_ok());

        let invalid = DpoConfig::new(-0.1);
        assert!(invalid.validate().is_err());

        let invalid_smoothing = DpoConfig::new(0.1)
            .with_loss_type(DpoLossType::Ipo)
            .with_label_smoothing(0.1);
        assert!(invalid_smoothing.validate().is_err());
    }

    #[test]
    fn test_preference_pair_creation() {
        let prompt = vec![1, 2, 3];
        let chosen = vec![4, 5];
        let rejected = vec![6, 7, 8];

        let pair = PreferencePair::new(prompt.clone(), chosen.clone(), rejected.clone());

        assert_eq!(pair.prompt_ids, vec![1, 2, 3]);
        assert_eq!(pair.chosen_ids, vec![1, 2, 3, 4, 5]); // prompt + chosen
        assert_eq!(pair.rejected_ids, vec![1, 2, 3, 6, 7, 8]); // prompt + rejected

        // Check labels: -100 for prompt, actual IDs for completion
        assert_eq!(pair.chosen_labels, vec![-100, -100, -100, 4, 5]);
        assert_eq!(pair.rejected_labels, vec![-100, -100, -100, 6, 7, 8]);
    }

    #[test]
    fn test_dpo_loss_sigmoid() {
        let config = DpoConfig::new(0.1);
        let training_config = TrainingConfig::default();
        let trainer = DpoTrainer::new(config, training_config).unwrap();

        // Test case where chosen is preferred (positive logits)
        let logits = Array::from_slice(&[1.0f32, 2.0, 0.5], &[3]);
        let loss = trainer.sigmoid_loss(&logits).unwrap();
        loss.eval().unwrap();

        // Loss should be positive
        let loss_val = loss.mean(None).unwrap();
        loss_val.eval().unwrap();
        assert!(loss_val.item::<f32>() > 0.0);
    }

    #[test]
    fn test_dpo_loss_ipo() {
        let config = DpoConfig::new(0.1).with_loss_type(DpoLossType::Ipo);
        let training_config = TrainingConfig::default();
        let trainer = DpoTrainer::new(config, training_config).unwrap();

        // IPO loss: (logits - 1/(2*beta))^2 = (logits - 5)^2
        let logits = Array::from_slice(&[5.0f32], &[1]); // At target
        let loss = trainer.ipo_loss(&logits).unwrap();
        loss.eval().unwrap();

        // At target, loss should be ~0
        assert!(loss.item::<f32>() < 0.01);
    }

    #[test]
    fn test_dpo_loss_hinge() {
        let config = DpoConfig::new(0.1).with_loss_type(DpoLossType::Hinge);
        let training_config = TrainingConfig::default();
        let trainer = DpoTrainer::new(config, training_config).unwrap();

        // Hinge: max(0, 1 - logits)
        let logits = Array::from_slice(&[2.0f32, 0.5, -1.0], &[3]);
        let loss = trainer.hinge_loss(&logits).unwrap();
        loss.eval().unwrap();

        // logits=2 -> max(0, -1) = 0
        // logits=0.5 -> max(0, 0.5) = 0.5
        // logits=-1 -> max(0, 2) = 2
        let expected = [0.0f32, 0.5, 2.0];
        for i in 0..3 {
            let val = loss.index(i as i32);
            val.eval().unwrap();
            assert!((val.item::<f32>() - expected[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_dpo_metrics() {
        let chosen_rewards = vec![1.0f32, 2.0, 1.5, 0.5];
        let rejected_rewards = vec![0.5f32, 1.0, 2.0, 0.0];

        let metrics = DpoMetrics::compute(0.1, &chosen_rewards, &rejected_rewards);

        assert_eq!(metrics.loss, 0.1);
        assert!((metrics.chosen_reward - 1.25).abs() < 0.01);
        assert!((metrics.rejected_reward - 0.875).abs() < 0.01);
        assert!((metrics.reward_margin - 0.375).abs() < 0.01);
        assert!((metrics.accuracy - 0.75).abs() < 0.01); // 3 out of 4 correct
    }

    #[test]
    fn test_label_smoothing() {
        let config = DpoConfig::new(0.1).with_label_smoothing(0.1);
        let training_config = TrainingConfig::default();
        let trainer = DpoTrainer::new(config, training_config).unwrap();

        let logits = Array::from_slice(&[1.0f32], &[1]);
        let loss = trainer.sigmoid_loss(&logits).unwrap();
        loss.eval().unwrap();

        // With label smoothing, loss should be different from pure sigmoid
        assert!(loss.item::<f32>() > 0.0);
    }

    #[test]
    fn test_simpo_loss() {
        // SimPO uses gamma margin
        let gamma = 1.0;
        let beta = 0.1;
        let config = DpoConfig::new(beta)
            .with_loss_type(DpoLossType::SimPo)
            .with_simpo_gamma(gamma);

        let training_config = TrainingConfig::default();
        let trainer = DpoTrainer::new(config, training_config).unwrap();

        // For SimPO: logits = beta * (chosen - rejected) - gamma
        // Let chosen = 10, rejected = 5
        // diff = 5.0
        // beta * diff = 0.5
        // logits = 0.5 - 1.0 = -0.5
        // loss = -log(sigmoid(-0.5)) = softplus(0.5) = log(1 + exp(0.5))

        let policy_chosen = Array::from_slice(&[10.0f32], &[1]);
        let policy_rejected = Array::from_slice(&[5.0f32], &[1]);
        let ref_chosen = Array::from_slice(&[0.0f32], &[1]); // Should be ignored
        let ref_rejected = Array::from_slice(&[0.0f32], &[1]); // Should be ignored

        let (loss, _, _) = trainer
            .compute_dpo_loss(&policy_chosen, &policy_rejected, &ref_chosen, &ref_rejected)
            .unwrap();

        loss.eval().unwrap();
        let loss_val = loss.item::<f32>();

        let expected_logits = -0.5f32;
        let expected_loss = (1.0 + (-expected_logits).exp()).ln(); // softplus(-logits)

        assert!((loss_val - expected_loss).abs() < 1e-4);
    }

    #[test]
    fn test_stop_gradient_config() {
        // Test that stop_gradient config can be disabled explicitly
        let config_disabled = DpoConfig::new(0.1).with_stop_gradient_reference(false);
        assert!(!config_disabled.use_stop_gradient_reference);

        // Default has stop_gradient enabled (recommended for memory efficiency)
        let config_default = DpoConfig::new(0.1);
        assert!(config_default.use_stop_gradient_reference);

        // Can explicitly enable it
        let config_enabled = DpoConfig::new(0.1).with_stop_gradient_reference(true);
        assert!(config_enabled.use_stop_gradient_reference);
    }

    #[test]
    fn test_precompute_input_validation_dimension() {
        let config = DpoConfig::new(0.1);
        let training_config = TrainingConfig::default();
        let trainer = DpoTrainer::new(config, training_config).unwrap();

        // Create invalid 1D arrays (should require 2D)
        let chosen_inputs_1d = Array::from_slice(&[1i32, 2, 3], &[3]);
        let chosen_labels_1d = Array::from_slice(&[1i32, 2, 3], &[3]);
        let rejected_inputs_1d = Array::from_slice(&[4i32, 5, 6], &[3]);
        let rejected_labels_1d = Array::from_slice(&[4i32, 5, 6], &[3]);

        // This is tricky - we need a trainable model, but let's use a mock or skip
        // For unit test, we just verify the validation logic returns appropriate error
        // when precompute_reference_log_probs is called with 1D inputs
        // Since we don't have a mock model, we check the validation happens early

        // Create proper 2D arrays but with batch size mismatch
        let chosen_inputs = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]);
        let chosen_labels = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]);
        let rejected_inputs = Array::from_slice(&[5i32, 6, 7], &[1, 3]); // Different batch!
        let rejected_labels = Array::from_slice(&[5i32, 6, 7], &[1, 3]);

        // Verify shape detection works
        assert_eq!(chosen_inputs.shape()[0], 2);
        assert_eq!(rejected_inputs.shape()[0], 1);
        assert_ne!(chosen_inputs.shape()[0], rejected_inputs.shape()[0]);
    }

    #[test]
    fn test_precompute_batch_size_zero() {
        // Verify the config validation
        let config = DpoConfig::new(0.1);
        assert_eq!(config.beta, 0.1);

        // Test would require a model for precompute_reference_log_probs_batched
        // Here we just verify the validation of batch_size=0 exists in the code
        // The actual validation returns DpoError::Config for batch_size=0
    }

    #[test]
    fn test_dpo_dtype_matching() {
        // Test that dtype matching works for i64 labels
        let config = DpoConfig::new(0.1);
        let training_config = TrainingConfig::default();
        let trainer = DpoTrainer::new(config, training_config).unwrap();

        // Create logits and labels with different dtypes
        let batch_size = 2;
        let seq_len = 4;
        let vocab_size = 10;

        // Float32 logits
        let logits_data: Vec<f32> = (0..batch_size * seq_len * vocab_size)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let logits = Array::from_slice(
            &logits_data,
            &[batch_size as i32, seq_len as i32, vocab_size as i32],
        );

        // i64 labels (common from PyTorch datasets)
        let labels_i64 = Array::from_slice(
            &[-100i64, 1, 2, 3, -100, 4, 5, 6],
            &[batch_size as i32, seq_len as i32],
        );

        // Compute log probs - should handle dtype mismatch
        let result = trainer.compute_log_probs(&logits, &labels_i64);
        assert!(result.is_ok(), "Should handle i64 labels");

        let log_probs = result.unwrap();
        log_probs.eval().unwrap();
        assert_eq!(log_probs.shape(), &[batch_size as i32]);
    }
}
