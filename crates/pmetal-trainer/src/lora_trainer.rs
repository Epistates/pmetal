//! LoRA-specific training loop with gradient computation.
//!
//! This module provides a complete training loop for LoRA fine-tuning using mlx-rs transforms.

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::{
    error::Exception, module::ModuleParameters, nn, ops::indexing::IndexOp, optimizers::Optimizer,
    transforms::eval_params, Array,
};
use pmetal_core::{LoraConfig, TrainingConfig};
use pmetal_lora::LlamaLoraForCausalLM;
use pmetal_mlx::kernels::cross_entropy::cross_entropy_loss;
use pmetal_models::architectures::llama::LlamaConfig;

use crate::{CheckpointManager, CheckpointMetadata, Result, SftError};

/// Training statistics for a single step.
#[derive(Debug, Clone)]
pub struct TrainStepStats {
    /// Loss value.
    pub loss: f32,
    /// Learning rate used.
    pub learning_rate: f32,
    /// Number of tokens processed.
    pub num_tokens: usize,
    /// Gradient norm (if computed).
    pub grad_norm: Option<f32>,
}

/// LoRA trainer for fine-tuning Llama models.
pub struct LoraTrainer {
    /// Model being trained.
    pub model: LlamaLoraForCausalLM,
    /// Training configuration.
    pub config: TrainingConfig,
    /// Current training step.
    pub step: usize,
    /// Current epoch.
    pub epoch: usize,
    /// Running loss average.
    pub running_loss: f64,
}

impl LoraTrainer {
    /// Create a new LoRA trainer.
    pub fn new(
        model_config: LlamaConfig,
        lora_config: LoraConfig,
        training_config: TrainingConfig,
    ) -> Result<Self> {
        let model = LlamaLoraForCausalLM::new(model_config, lora_config)
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

        Ok(Self {
            model,
            config: training_config,
            step: 0,
            epoch: 0,
            running_loss: 0.0,
        })
    }

    /// Create from an existing model.
    pub fn from_model(model: LlamaLoraForCausalLM, training_config: TrainingConfig) -> Self {
        Self {
            model,
            config: training_config,
            step: 0,
            epoch: 0,
            running_loss: 0.0,
        }
    }

    /// Compute loss for a batch.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [batch, seq_len]
    /// * `labels` - Target labels [batch, seq_len] (use -100 for ignored positions)
    ///
    /// # Returns
    /// Loss value as a scalar Array
    pub fn compute_loss(&mut self, input_ids: &Array, labels: &Array) -> Result<Array> {
        // Forward pass
        let logits = self
            .model
            .forward(input_ids, None)
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

        // Compute causal LM loss
        compute_lm_loss(&logits, labels)
    }

    /// Perform a single training step with manual gradient computation.
    ///
    /// This uses finite differences for gradient estimation - suitable for small models
    /// or when automatic differentiation isn't available.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [batch, seq_len]
    /// * `labels` - Target labels [batch, seq_len]
    ///
    /// # Returns
    /// Training statistics for this step
    pub fn train_step_finite_diff(
        &mut self,
        input_ids: &Array,
        labels: &Array,
    ) -> Result<TrainStepStats> {
        // Compute current loss
        let loss = self.compute_loss(input_ids, labels)?;
        loss.eval()?;
        let loss_val = loss.item::<f32>();

        // Get current learning rate
        let lr = self.get_learning_rate();

        // Epsilon for finite differences
        let epsilon = 1e-4_f32;

        // Note: Finite differences is impractical for real training.
        // This is a placeholder demonstrating the API structure.
        // For actual training, use train_step_sgd with computed gradients
        // or implement proper autodiff with mlx-rs transforms.
        let _ = (epsilon, lr); // Suppress unused warnings

        self.step += 1;
        self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;

        Ok(TrainStepStats {
            loss: loss_val,
            learning_rate: lr,
            num_tokens: (input_ids.dim(0) * input_ids.dim(1)) as usize,
            grad_norm: None,
        })
    }

    /// Perform a training step with SGD using numerical gradients.
    ///
    /// This is a simplified training step that uses the LoRA parameter update directly.
    /// For production, implement proper autodiff.
    pub fn train_step_sgd(
        &mut self,
        input_ids: &Array,
        labels: &Array,
        gradients: &HashMap<Rc<str>, Array>,
    ) -> Result<TrainStepStats> {
        // Compute loss first
        let loss = self.compute_loss(input_ids, labels)?;
        loss.eval()?;
        let loss_val = loss.item::<f32>();

        // Get learning rate
        let lr = self.get_learning_rate();

        // Apply gradients
        self.model
            .apply_gradients(gradients, lr)
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

        // Evaluate updated params
        self.model
            .eval_lora_params()
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

        self.step += 1;
        self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;

        Ok(TrainStepStats {
            loss: loss_val,
            learning_rate: lr,
            num_tokens: (input_ids.dim(0) * input_ids.dim(1)) as usize,
            grad_norm: None,
        })
    }

    /// Perform a training step using automatic differentiation.
    ///
    /// This uses mlx-rs `nn::value_and_grad` for proper gradient computation
    /// with respect to LoRA parameters.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [batch, seq_len]
    /// * `labels` - Target labels [batch, seq_len]
    /// * `optimizer` - Optimizer to use for parameter updates
    ///
    /// # Returns
    /// Training statistics for this step
    pub fn train_step<O: Optimizer>(
        &mut self,
        input_ids: &Array,
        labels: &Array,
        optimizer: &mut O,
    ) -> Result<TrainStepStats> {
        // Define loss function for value_and_grad
        let loss_fn = |model: &mut LlamaLoraForCausalLM,
                       (ids, lbls): (&Array, &Array)|
         -> std::result::Result<Array, Exception> {
            let logits = model
                .forward(ids, None)
                .map_err(|e| Exception::custom(e.to_string()))?;

            let seq_len = logits.dim(1);
            let vocab_size = logits.dim(2);

            let shift_logits = logits.index((.., ..seq_len - 1, ..));
            let shift_labels = lbls.index((.., 1..));

            let flat_logits = shift_logits.reshape(&[-1, vocab_size])?;
            let flat_labels = shift_labels.reshape(&[-1])?;

            let loss = cross_entropy_loss(&flat_logits, &flat_labels, Some(-100_i64), 0.0)?;
            loss.mean(None)
        };

        // Create value_and_grad function
        let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);

        // Compute loss and gradients
        let (loss, gradients) = loss_and_grad_fn(&mut self.model, (input_ids, labels))?;
        loss.eval()?;
        let loss_val = loss.item::<f32>();

        // Update parameters with optimizer
        optimizer.update(&mut self.model, gradients)?;

        // Evaluate updated parameters
        eval_params(self.model.parameters())?;

        self.step += 1;
        self.running_loss = 0.99 * self.running_loss + 0.01 * loss_val as f64;

        Ok(TrainStepStats {
            loss: loss_val,
            learning_rate: self.get_learning_rate(),
            num_tokens: (input_ids.dim(0) * input_ids.dim(1)) as usize,
            grad_norm: None,
        })
    }

    /// Get the current learning rate based on scheduler.
    pub fn get_learning_rate(&self) -> f32 {
        let warmup = self.config.warmup_steps;
        let total_steps = self.config.max_steps.unwrap_or(10000);
        let base_lr = self.config.learning_rate as f32;

        if self.step < warmup {
            // Linear warmup
            base_lr * (self.step as f32 / warmup as f32)
        } else {
            match self.config.lr_scheduler {
                pmetal_core::LrSchedulerType::Constant => base_lr,
                pmetal_core::LrSchedulerType::Linear => {
                    let decay_steps = (total_steps - warmup) as f32;
                    let current = (self.step - warmup) as f32;
                    base_lr * (1.0 - current / decay_steps).max(0.0)
                }
                pmetal_core::LrSchedulerType::Cosine => {
                    let decay_steps = (total_steps - warmup) as f32;
                    let current = (self.step - warmup) as f32;
                    let progress = current / decay_steps;
                    base_lr * 0.5 * (1.0 + (std::f64::consts::PI as f32 * progress).cos())
                }
                _ => base_lr,
            }
        }
    }

    /// Get current training step.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Get current running loss.
    pub fn current_loss(&self) -> f64 {
        self.running_loss
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    /// Save LoRA weights to a file.
    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.model
            .save_lora_weights(path)
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
        Ok(())
    }

    /// Load LoRA weights from a file.
    pub fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.model
            .load_lora_weights(path)
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
        Ok(())
    }

    /// Merge LoRA weights into base model.
    pub fn merge_lora(&mut self) -> Result<()> {
        self.model
            .merge_lora()
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
        Ok(())
    }

    /// Save a training checkpoint.
    ///
    /// # Arguments
    /// * `checkpoint_manager` - Checkpoint manager to use
    /// * `is_best` - Whether this is the best checkpoint so far
    /// * `best_val_loss` - Best validation loss seen (optional)
    pub fn save_checkpoint(
        &self,
        checkpoint_manager: &CheckpointManager,
        is_best: bool,
        best_val_loss: Option<f64>,
    ) -> Result<std::path::PathBuf> {
        // Collect LoRA parameters
        let lora_params = self.model.lora_parameters();

        // Create metadata
        let mut metadata = CheckpointMetadata::new(
            self.step,
            self.epoch,
            self.running_loss,
            self.get_learning_rate() as f64,
        );

        if let Some(loss) = best_val_loss {
            metadata = metadata.with_best_val_loss(loss);
        }

        checkpoint_manager.save_checkpoint(&lora_params, &metadata, is_best)
    }

    /// Restore training state from a checkpoint.
    ///
    /// # Arguments
    /// * `checkpoint_path` - Path to checkpoint directory
    pub fn restore_checkpoint(
        &mut self,
        checkpoint_path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        let (params, metadata) = CheckpointManager::load_checkpoint(checkpoint_path)?;

        // Restore parameters
        self.model.set_lora_parameters(&params);

        // Restore training state
        self.step = metadata.step;
        self.epoch = metadata.epoch;
        self.running_loss = metadata.running_loss;

        tracing::info!(
            "Restored checkpoint from step {}, epoch {}, loss {:.4}",
            self.step,
            self.epoch,
            self.running_loss
        );

        Ok(())
    }

    /// Resume training from the latest checkpoint.
    ///
    /// # Arguments
    /// * `checkpoint_manager` - Checkpoint manager to use
    ///
    /// # Returns
    /// Whether a checkpoint was found and restored
    pub fn resume_from_latest(&mut self, checkpoint_manager: &CheckpointManager) -> Result<bool> {
        match checkpoint_manager.load_latest()? {
            Some((params, metadata)) => {
                self.model.set_lora_parameters(&params);
                self.step = metadata.step;
                self.epoch = metadata.epoch;
                self.running_loss = metadata.running_loss;
                tracing::info!(
                    "Resumed from checkpoint at step {}, epoch {}",
                    self.step,
                    self.epoch
                );
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

/// Compute language model loss (cross-entropy with shifting).
pub fn compute_lm_loss(logits: &Array, labels: &Array) -> Result<Array> {
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

    Ok(loss.mean(None)?)
}

/// Create a simple training loop for testing.
///
/// # Arguments
/// * `model_config` - Llama model configuration
/// * `lora_config` - LoRA configuration
/// * `training_config` - Training hyperparameters
/// * `data` - Iterator of (input_ids, labels) batches
///
/// # Returns
/// Final loss value
pub fn simple_training_loop<I>(
    model_config: LlamaConfig,
    lora_config: LoraConfig,
    training_config: TrainingConfig,
    data: I,
) -> Result<f64>
where
    I: Iterator<Item = (Array, Array)>,
{
    let mut trainer = LoraTrainer::new(model_config, lora_config, training_config)?;

    tracing::info!(
        "Starting LoRA training with {} trainable parameters",
        trainer.num_trainable_params()
    );

    for (step, (input_ids, labels)) in data.enumerate() {
        // Compute loss (no gradient update in this simple version)
        let loss = trainer.compute_loss(&input_ids, &labels)?;
        loss.eval()?;
        let loss_val = loss.item::<f32>() as f64;

        trainer.running_loss = if step == 0 {
            loss_val
        } else {
            0.99 * trainer.running_loss + 0.01 * loss_val
        };

        if step % 10 == 0 {
            tracing::info!(
                "Step {}: loss = {:.4}, lr = {:.6}",
                step,
                trainer.running_loss,
                trainer.get_learning_rate()
            );
        }

        trainer.step = step + 1;
    }

    Ok(trainer.running_loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: None,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            ..Default::default()
        }
    }

    fn small_lora_config() -> LoraConfig {
        LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
            use_rslora: false,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            bias: "none".to_string(),
            init_lora_weights: true,
            use_dora: false,
        }
    }

    fn small_training_config() -> TrainingConfig {
        TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 2,
            max_steps: Some(100),
            warmup_steps: 10,
            ..Default::default()
        }
    }

    #[test]
    fn test_lora_trainer_creation() {
        let trainer =
            LoraTrainer::new(small_config(), small_lora_config(), small_training_config()).unwrap();

        assert!(trainer.num_trainable_params() > 0);
        assert_eq!(trainer.current_step(), 0);
    }

    #[test]
    fn test_lora_trainer_loss_computation() {
        let mut trainer =
            LoraTrainer::new(small_config(), small_lora_config(), small_training_config()).unwrap();

        // Create dummy data
        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let labels = Array::from_slice(&[2_i32, 3, 4, 5], &[1, 4]);

        let loss = trainer.compute_loss(&input_ids, &labels).unwrap();
        loss.eval().unwrap();

        // Loss should be positive
        assert!(loss.item::<f32>() > 0.0);
    }

    #[test]
    fn test_learning_rate_schedule() {
        let mut trainer =
            LoraTrainer::new(small_config(), small_lora_config(), small_training_config()).unwrap();

        // At step 0, should be in warmup
        trainer.step = 0;
        let lr0 = trainer.get_learning_rate();
        assert!(lr0 < 1e-4); // Should be less than base LR during warmup

        // At step 10 (end of warmup), should be at base LR
        trainer.step = 10;
        let lr10 = trainer.get_learning_rate();
        assert!((lr10 - 1e-4).abs() < 1e-8);
    }

    #[test]
    fn test_train_step_autodiff() {
        use mlx_rs::optimizers::Sgd;

        let mut trainer =
            LoraTrainer::new(small_config(), small_lora_config(), small_training_config()).unwrap();

        // Create dummy data
        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let labels = Array::from_slice(&[2_i32, 3, 4, 5], &[1, 4]);

        // Create optimizer
        let mut optimizer = Sgd::new(1e-4);

        // Get initial loss
        let initial_loss = trainer.compute_loss(&input_ids, &labels).unwrap();
        initial_loss.eval().unwrap();
        let initial_loss_val = initial_loss.item::<f32>();

        // Perform training step with autodiff
        let stats = trainer
            .train_step(&input_ids, &labels, &mut optimizer)
            .unwrap();

        // Verify stats are reasonable
        assert!(stats.loss > 0.0);
        assert_eq!(trainer.current_step(), 1);

        // Loss should have been computed
        assert!((stats.loss - initial_loss_val).abs() < 0.1);
    }
}
