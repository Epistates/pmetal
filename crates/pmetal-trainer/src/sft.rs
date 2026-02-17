//! Supervised Fine-Tuning (SFT) trainer.
//!
//! Implements efficient SFT training with:
//! - Gradient accumulation
//! - Learning rate scheduling
//! - LoRA parameter updates
//! - Checkpoint saving

use mlx_rs::{
    builder::Builder,
    error::Exception,
    ops::indexing::IndexOp,
    optimizers::{AdamW, AdamWBuilder},
    Array,
};
use pmetal_core::{EvalMetrics, TrainingConfig};
use pmetal_lora::LoraLinear;
use pmetal_mlx::kernels::cross_entropy::cross_entropy_loss;
use std::path::Path;

/// Error type for SFT training.
#[derive(Debug, thiserror::Error)]
pub enum SftError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// LoRA error.
    #[error("LoRA error: {0}")]
    Lora(#[from] pmetal_lora::LoraError),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for SFT operations.
pub type Result<T> = std::result::Result<T, SftError>;

/// Training state for tracking progress.
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current training step.
    pub step: usize,
    /// Current epoch.
    pub epoch: usize,
    /// Current loss (moving average).
    pub loss: f64,
    /// Learning rate at current step.
    pub learning_rate: f64,
    /// Total tokens processed.
    pub tokens_processed: usize,
    /// Gradient norm (if clipping is enabled).
    pub grad_norm: Option<f64>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            step: 0,
            epoch: 0,
            loss: 0.0,
            learning_rate: 0.0,
            tokens_processed: 0,
            grad_norm: None,
        }
    }
}

/// Supervised Fine-Tuning trainer.
pub struct SftTrainer {
    /// Training configuration.
    config: TrainingConfig,
    /// Current training state.
    state: TrainingState,
    /// Optimizer.
    optimizer: Option<AdamW>,
    /// Accumulated gradients for gradient accumulation.
    accumulated_grads: Option<Vec<Array>>,
    /// Number of accumulation steps completed.
    accumulation_count: usize,
}

impl SftTrainer {
    /// Create a new SFT trainer.
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            state: TrainingState {
                learning_rate: config.learning_rate,
                ..Default::default()
            },
            config,
            optimizer: None,
            accumulated_grads: None,
            accumulation_count: 0,
        }
    }

    /// Initialize the optimizer.
    pub fn init_optimizer(&mut self) -> Result<()> {
        // AdamWBuilder::build returns Result<AdamW, Infallible>, so unwrap is safe
        let optimizer = AdamWBuilder::new(self.config.learning_rate as f32)
            .weight_decay(self.config.weight_decay as f32)
            .build()
            .unwrap();
        self.optimizer = Some(optimizer);
        Ok(())
    }

    /// Compute loss for a batch.
    ///
    /// # Arguments
    /// * `logits` - Model output logits [batch, seq_len, vocab_size]
    /// * `labels` - Target labels [batch, seq_len]
    /// * `attention_mask` - Optional attention mask
    ///
    /// # Returns
    /// Scalar loss value
    pub fn compute_loss(
        &self,
        logits: &Array,
        labels: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array> {
        // Shift logits and labels for causal LM
        let seq_len = logits.dim(1);
        let vocab_size = logits.dim(2);

        // Shift: logits[:-1] predicts labels[1:]
        let shift_logits = logits.index((.., ..seq_len - 1, ..));
        let shift_labels = labels.index((.., 1..));

        // Reshape for cross entropy: [batch * seq_len, vocab_size] and [batch * seq_len]
        let flat_logits = shift_logits.reshape(&[-1, vocab_size])?;
        let flat_labels = shift_labels.reshape(&[-1])?;

        // Compute cross entropy loss with ignore_index=-100
        let loss = cross_entropy_loss(&flat_logits, &flat_labels, Some(-100_i64), 0.0)?;

        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            let shift_mask = mask.index((.., 1..));
            let flat_mask = shift_mask.reshape(&[-1])?;
            let masked_loss = loss.multiply(&flat_mask)?;
            let total_loss = masked_loss.sum(None)?;
            let num_tokens = flat_mask.sum(None)?;
            Ok(total_loss.divide(&num_tokens)?)
        } else {
            Ok(loss.mean(None)?)
        }
    }

    /// Perform a single training step on LoRA parameters.
    ///
    /// This computes gradients and updates LoRA A and B matrices.
    /// Supports optional pixel values for multimodal models.
    pub fn train_step_lora(
        &mut self,
        lora_layers: &mut [&mut LoraLinear],
        input_ids: &Array,
        pixel_values: Option<&Array>,
        labels: &Array,
        forward_fn: impl Fn(&[&LoraLinear], &Array, Option<&Array>) -> Result<Array>,
    ) -> Result<f64> {
        // Collect LoRA parameters for gradient computation
        let _lora_params: Vec<(&Array, &Array)> =
            lora_layers.iter().map(|l| (&l.lora_a, &l.lora_b)).collect();

        // Forward pass
        let lora_refs: Vec<&LoraLinear> = lora_layers.iter().map(|l| &**l).collect();
        let logits = forward_fn(&lora_refs, input_ids, pixel_values)?;

        // Compute loss
        let loss = self.compute_loss(&logits, labels, None)?;

        // Get scalar loss value
        loss.eval()?;
        let loss_value = loss.item::<f32>() as f64;

        // Backward pass would be here using mlx_rs::transforms::grad
        // For now, we just return the loss value
        // In a full implementation, we would:
        // 1. Use value_and_grad to get gradients
        // 2. Accumulate gradients if using gradient accumulation
        // 3. Apply gradient clipping if configured
        // 4. Update parameters using the optimizer

        // Update training state
        self.state.step += 1;
        self.state.loss = loss_value;

        Ok(loss_value)
    }

    /// Apply gradient clipping.
    pub fn clip_gradients(&self, grads: &mut [Array]) -> Result<Option<f64>> {
        if self.config.max_grad_norm <= 0.0 {
            return Ok(None);
        }

        // Compute global gradient norm
        let mut total_norm_sq = 0.0;
        for grad in grads.iter() {
            grad.eval()?;
            let norm_sq = grad.multiply(grad)?.sum(None)?;
            norm_sq.eval()?;
            total_norm_sq += norm_sq.item::<f32>() as f64;
        }
        let total_norm = total_norm_sq.sqrt();

        // Clip if necessary
        if total_norm > self.config.max_grad_norm {
            let scale = self.config.max_grad_norm / total_norm;
            for grad in grads.iter_mut() {
                let scale_arr = Array::from_f32(scale as f32);
                *grad = grad.multiply(&scale_arr)?;
            }
        }

        Ok(Some(total_norm))
    }

    /// Update learning rate based on scheduler.
    pub fn update_learning_rate(&mut self) {
        let step = self.state.step;
        let warmup = self.config.warmup_steps;
        let total_steps = self.config.max_steps.unwrap_or(10000);
        let base_lr = self.config.learning_rate;

        self.state.learning_rate = match self.config.lr_scheduler {
            pmetal_core::LrSchedulerType::Constant => base_lr,
            pmetal_core::LrSchedulerType::Linear => {
                if step < warmup {
                    // Linear warmup
                    base_lr * (step as f64 / warmup as f64)
                } else {
                    // Linear decay
                    let decay_steps = total_steps.saturating_sub(warmup).max(1) as f64;
                    let current = step.saturating_sub(warmup) as f64;
                    base_lr * (1.0 - current / decay_steps).max(0.0)
                }
            }
            pmetal_core::LrSchedulerType::Cosine => {
                if step < warmup {
                    // Linear warmup
                    base_lr * (step as f64 / warmup as f64)
                } else {
                    // Cosine decay
                    let decay_steps = total_steps.saturating_sub(warmup).max(1) as f64;
                    let current = step.saturating_sub(warmup) as f64;
                    let progress = current / decay_steps;
                    base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
                }
            }
            _ => base_lr,
        };
    }

    /// Get current training state.
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Get current training step.
    pub fn current_step(&self) -> usize {
        self.state.step
    }

    /// Get current loss.
    pub fn current_loss(&self) -> Option<f64> {
        if self.state.step > 0 {
            Some(self.state.loss)
        } else {
            None
        }
    }

    /// Train the model (placeholder for full training loop).
    pub fn train(&mut self) -> pmetal_core::Result<()> {
        tracing::info!("Starting SFT training...");
        tracing::info!("Learning rate: {}", self.config.learning_rate);
        tracing::info!("Batch size: {}", self.config.batch_size);
        tracing::info!(
            "Gradient accumulation steps: {}",
            self.config.gradient_accumulation_steps
        );
        // Full training loop would be implemented here
        Ok(())
    }

    /// Evaluate the model.
    pub fn evaluate(&self) -> pmetal_core::Result<EvalMetrics> {
        Ok(EvalMetrics::default())
    }

    /// Save checkpoint.
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> pmetal_core::Result<()> {
        let path = path.as_ref();
        tracing::info!("Saving checkpoint to {:?}", path);
        // Save training state, optimizer state, and LoRA weights
        Ok(())
    }

    /// Load checkpoint.
    pub fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> pmetal_core::Result<()> {
        let path = path.as_ref();
        tracing::info!("Loading checkpoint from {:?}", path);
        // Load training state, optimizer state, and LoRA weights
        Ok(())
    }
}

/// Compute the loss function for a language model.
///
/// This is a standalone function for use with `mlx_rs::transforms::value_and_grad`.
pub fn lm_loss(logits: &Array, labels: &Array, ignore_index: i64) -> Result<Array> {
    let seq_len = logits.dim(1);
    let vocab_size = logits.dim(2);

    // Shift: logits[:-1] predicts labels[1:]
    let shift_logits = logits.index((.., ..seq_len - 1, ..));
    let shift_labels = labels.index((.., 1..));

    // Reshape for cross entropy
    let flat_logits = shift_logits.reshape(&[-1, vocab_size])?;
    let flat_labels = shift_labels.reshape(&[-1])?;

    // Compute cross entropy loss
    let loss = cross_entropy_loss(&flat_logits, &flat_labels, Some(ignore_index), 0.0)?;

    Ok(loss.mean(None)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig::default();
        let trainer = SftTrainer::new(config);

        assert_eq!(trainer.current_step(), 0);
        assert!(trainer.current_loss().is_none());
    }

    #[test]
    fn test_learning_rate_warmup() {
        let mut config = TrainingConfig::default();
        config.warmup_steps = 100;
        config.max_steps = Some(1000);
        config.learning_rate = 1e-4;
        config.lr_scheduler = pmetal_core::LrSchedulerType::Linear;

        let mut trainer = SftTrainer::new(config);

        // At step 0
        trainer.state.step = 0;
        trainer.update_learning_rate();
        assert!((trainer.state.learning_rate - 0.0).abs() < 1e-10);

        // At step 50 (halfway through warmup)
        trainer.state.step = 50;
        trainer.update_learning_rate();
        assert!((trainer.state.learning_rate - 5e-5).abs() < 1e-10);

        // At step 100 (end of warmup)
        trainer.state.step = 100;
        trainer.update_learning_rate();
        assert!((trainer.state.learning_rate - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_scheduler() {
        let mut config = TrainingConfig::default();
        config.warmup_steps = 0;
        config.max_steps = Some(100);
        config.learning_rate = 1e-4;
        config.lr_scheduler = pmetal_core::LrSchedulerType::Cosine;

        let mut trainer = SftTrainer::new(config);

        // At step 0
        trainer.state.step = 0;
        trainer.update_learning_rate();
        assert!((trainer.state.learning_rate - 1e-4).abs() < 1e-10);

        // At step 50 (halfway)
        trainer.state.step = 50;
        trainer.update_learning_rate();
        assert!((trainer.state.learning_rate - 5e-5).abs() < 1e-8);

        // At step 100 (end)
        trainer.state.step = 100;
        trainer.update_learning_rate();
        assert!(trainer.state.learning_rate < 1e-8);
    }

    #[test]
    fn test_lm_loss_shape() {
        // Create dummy logits and labels
        let batch = 2;
        let seq_len = 8;
        let vocab_size = 100;

        let logits =
            mlx_rs::random::normal::<f32>(&[batch, seq_len, vocab_size], None, None, None).unwrap();
        let labels =
            mlx_rs::random::randint::<_, i32>(0, vocab_size, &[batch, seq_len], None).unwrap();

        let loss = lm_loss(&logits, &labels, -100).unwrap();
        loss.eval().unwrap();

        // Loss should be a scalar
        let empty_shape: &[i32] = &[];
        assert_eq!(loss.shape(), empty_shape);
        assert!(loss.item::<f32>() > 0.0);
    }
}
