//! Odds Ratio Preference Optimization (ORPO) trainer.
//!
//! ORPO is a reference-free, single-stage preference optimization method that
//! combines SFT loss with an odds ratio penalty to penalize rejected responses.
//!
//! Based on: "ORPO: Monolithic Preference Optimization without Reference Model"
//! by Hong and Lee (2024).
//!
//! The ORPO loss is:
//! ```text
//! L_ORPO = L_SFT + lambda * L_OR
//! L_OR = -log(sigmoid(log_odds_chosen - log_odds_rejected))
//! log_odds = log(probs / (1 - probs))
//! ```

use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};
use pmetal_core::TrainingConfig;

/// Error type for ORPO training.
#[derive(Debug, thiserror::Error)]
pub enum OrpoError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
}

/// Result type for ORPO operations.
pub type OrpoResult<T> = std::result::Result<T, OrpoError>;

/// ORPO configuration.
#[derive(Debug, Clone)]
pub struct OrpoConfig {
    /// Beta/Lambda parameter controlling the strength of the odds ratio penalty.
    /// Default: 0.1
    pub beta: f64,

    /// Maximum length for prompt tokens.
    pub max_prompt_length: usize,

    /// Maximum length for response tokens (chosen/rejected).
    pub max_completion_length: usize,

    /// Whether to truncate prompts from the left.
    pub truncate_prompt_left: bool,
}

impl Default for OrpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            max_prompt_length: 512,
            max_completion_length: 512,
            truncate_prompt_left: true,
        }
    }
}

impl OrpoConfig {
    /// Create a new ORPO config with the given beta.
    pub fn new(beta: f64) -> Self {
        Self {
            beta,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> OrpoResult<()> {
        if self.beta < 0.0 {
            return Err(OrpoError::Config("ORPO beta must be non-negative".into()));
        }
        Ok(())
    }
}

/// ORPO trainer for preference learning.
pub struct OrpoTrainer {
    /// ORPO configuration.
    pub config: OrpoConfig,
    /// Training configuration.
    pub training_config: TrainingConfig,
}

impl OrpoTrainer {
    /// Create a new ORPO trainer.
    pub fn new(config: OrpoConfig, training_config: TrainingConfig) -> OrpoResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            training_config,
        })
    }

    /// Compute log probabilities and average log probabilities for a sequence.
    ///
    /// # Arguments
    /// * `logits` - Model output logits [batch, seq_len, vocab_size]
    /// * `labels` - Target labels [batch, seq_len] (-100 for ignored positions)
    ///
    /// # Returns
    /// (total_log_probs, average_log_probs)
    /// - total_log_probs: Sum of log probs [batch] (for NLL/SFT loss)
    /// - average_log_probs: Mean of log probs [batch] (for Odds Ratio)
    pub fn compute_log_probs(&self, logits: &Array, labels: &Array) -> OrpoResult<(Array, Array)> {
        // Shift logits and labels for next-token prediction
        let seq_len = logits.dim(1);

        // logits[:, :-1, :] -> predict next token
        let pred_logits = logits.index((.., ..seq_len - 1, ..));

        // labels[:, 1:] -> target is next token
        let target_labels = labels.index((.., 1..));

        // Compute log softmax
        let log_probs = mlx_rs::nn::log_softmax(&pred_logits, -1)?;

        // Create mask for valid labels
        let ignore_index = Array::from_int(-100);
        let valid_mask = target_labels.ne(&ignore_index)?;

        // Replace -100 with 0 for gathering
        let gather_labels = mlx_rs::ops::maximum(&target_labels, &Array::from_int(0))?;

        // Expand dims for gather: [B, S] -> [B, S, 1]
        let gather_indices = gather_labels.expand_dims(-1i32)?;

        // Gather log probs: [B, S, V] take [B, S, 1] -> [B, S, 1]
        let gathered_log_probs = log_probs.take_along_axis(&gather_indices, -1)?;
        let gathered_log_probs = gathered_log_probs.squeeze_axes(&[-1i32])?;

        // Mask out ignored tokens
        let valid_mask_f32 = valid_mask.as_dtype(Dtype::Float32)?;
        let masked_log_probs = gathered_log_probs.multiply(&valid_mask_f32)?;

        // Sum over sequence dimension -> [B]
        let total_log_probs = masked_log_probs.sum_axes(&[1i32], false)?;

        // Count valid tokens per sequence
        let valid_counts = valid_mask_f32.sum_axes(&[1i32], false)?;

        // Compute average log probs (handle division by zero if needed, though unlikely with valid masks)
        let average_log_probs = total_log_probs.divide(&valid_counts)?;

        Ok((total_log_probs, average_log_probs))
    }

    /// Compute ORPO loss for a batch.
    ///
    /// # Arguments
    /// * `chosen_log_probs` - Total log probs for chosen [batch]
    /// * `chosen_avg_log_probs` - Average log probs for chosen [batch]
    /// * `rejected_avg_log_probs` - Average log probs for rejected [batch]
    ///
    /// # Returns
    /// (total_loss, sft_loss, or_loss, log_odds_chosen, log_odds_rejected)
    pub fn compute_orpo_loss(
        &self,
        chosen_log_probs: &Array,
        chosen_avg_log_probs: &Array,
        rejected_avg_log_probs: &Array,
    ) -> OrpoResult<(Array, Array, Array, Array, Array)> {
        // 1. SFT Loss: Negative Log Likelihood of chosen response
        // chosen_log_probs is sum(log P(y_w|x))
        // We want -mean(chosen_log_probs) / mean(sequence_length) usually,
        // but here we return per-batch losses to be averaged later if needed.
        // Standard PyTorch CrossEntropy is -log_prob.
        let sft_loss = chosen_log_probs.negative()?;

        // 2. Odds Ratio Loss
        // log_odds = log(P / (1 - P))
        // Since we have log_P (average per token), we can approximate P per token as exp(avg_log_P)
        // log_odds = avg_log_P - log(1 - exp(avg_log_P))

        let compute_log_odds = |avg_log_p: &Array| -> OrpoResult<Array> {
            let p = avg_log_p.exp()?;
            let one = Array::from_f32(1.0);
            let one_minus_p = one.subtract(&p)?;

            // Numerical stability: clip 1-p to avoid log(0)
            let epsilon = Array::from_f32(1e-10);
            let one_minus_p_safe = mlx_rs::ops::maximum(&one_minus_p, &epsilon)?;

            let log_one_minus_p = one_minus_p_safe.log()?;
            Ok(avg_log_p.subtract(&log_one_minus_p)?)
        };

        let log_odds_chosen = compute_log_odds(chosen_avg_log_probs)?;
        let log_odds_rejected = compute_log_odds(rejected_avg_log_probs)?;

        // ratio = log_odds_chosen - log_odds_rejected
        // loss = -log(sigmoid(ratio)) = softplus(-ratio)
        let ratio = log_odds_chosen.subtract(&log_odds_rejected)?;
        let neg_ratio = ratio.negative()?;
        let or_loss = mlx_rs::nn::softplus(&neg_ratio)?;

        // Total loss = SFT_loss + beta * OR_loss
        let beta = Array::from_f32(self.config.beta as f32);
        let weighted_or_loss = or_loss.multiply(&beta)?;
        let total_loss = sft_loss.add(&weighted_or_loss)?;

        // Return mean losses over batch
        Ok((
            total_loss.mean(None)?,
            sft_loss.mean(None)?,
            or_loss.mean(None)?,
            log_odds_chosen,
            log_odds_rejected,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orpo_log_odds() {
        // Test log odds calculation
        // If p = 0.5, log_odds = log(0.5/0.5) = 0
        // avg_log_p = log(0.5) = -0.6931

        let config = OrpoConfig::default();
        let _trainer = OrpoTrainer::new(config, TrainingConfig::default()).unwrap();

        let avg_log_p = Array::from_f32(-0.693147);
        // Manually invoke logic (can't access closure directly)
        // Replicating closure logic for test:
        let p = avg_log_p.exp().unwrap();
        let one = Array::from_f32(1.0);
        let one_minus_p = one.subtract(&p).unwrap();
        let log_one_minus_p = one_minus_p.log().unwrap();
        let log_odds = avg_log_p.subtract(&log_one_minus_p).unwrap();

        log_odds.eval().unwrap();
        assert!(log_odds.item::<f32>().abs() < 1e-4);
    }

    #[test]
    fn test_orpo_loss_calculation() {
        let config = OrpoConfig::new(1.0); // beta = 1.0 for simple math
        let trainer = OrpoTrainer::new(config, TrainingConfig::default()).unwrap();

        // Case: Chosen is better
        // Chosen avg log prob = log(0.9) approx -0.105
        // Rejected avg log prob = log(0.1) approx -2.302

        let chosen_log_p = Array::from_slice(&[-0.105f32], &[1]);
        let chosen_sum = Array::from_slice(&[-1.05f32], &[1]); // assuming 10 tokens

        let rejected_log_p = Array::from_slice(&[-2.302f32], &[1]);

        // SFT loss = -sum = 1.05

        let (total, sft, or, _, _) = trainer
            .compute_orpo_loss(&chosen_sum, &chosen_log_p, &rejected_log_p)
            .unwrap();

        total.eval().unwrap();
        sft.eval().unwrap();
        or.eval().unwrap();

        // chosen odds = 0.9/0.1 = 9, log = 2.19
        // rejected odds = 0.1/0.9 = 0.11, log = -2.19
        // diff = 4.38
        // or_loss = -log(sigmoid(4.38)) approx 0.012

        assert!((sft.item::<f32>() - 1.05).abs() < 1e-3);
        assert!(or.item::<f32>() < 0.1); // Loss should be small as chosen >> rejected
    }
}
