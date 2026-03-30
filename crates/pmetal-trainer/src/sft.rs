//! Supervised Fine-Tuning (SFT) trainer.
//!
//! Implements efficient SFT training with:
//! - Gradient accumulation
//! - Learning rate scheduling
//! - LoRA parameter updates
//! - Checkpoint saving

use pmetal_bridge::compat::{
    Array, Exception,
    indexing::IndexOp,
    ops,
    optimizers::{AdamW, AdamWBuilder},
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
    /// Training was cancelled by a callback.
    #[error("Training cancelled")]
    Cancelled,
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
    /// Accumulated gradients for gradient accumulation (handled in training loop).
    _accumulated_grads: Option<Vec<Array>>,
    /// Number of accumulation steps completed (handled in training loop).
    _accumulation_count: usize,
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
            _accumulated_grads: None,
            _accumulation_count: 0,
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
        if let Some(mask) = attention_mask {
            let mask_dtype = mask.dtype_raw();
            let zero = Array::from_int(0).as_dtype(mask_dtype);
            let cond = mask.ne(&zero);
            let ignore = Array::from_int(-100).as_dtype(labels.dtype_raw());
            let ignore_full = ops::broadcast_to(&ignore, &labels.shape());
            let masked_labels = ops::where_fn(&cond, labels, &ignore_full);
            Ok(pmetal_bridge::training::causal_lm_loss(
                logits,
                &masked_labels,
                -100,
            ))
        } else {
            Ok(pmetal_bridge::training::causal_lm_loss(
                logits, labels, -100,
            ))
        }
    }

    /// Perform a single training step on LoRA parameters.
    ///
    /// # Deprecated
    ///
    /// This method only performs a forward pass and does **not** compute
    /// gradients or update any parameters.  It exists for legacy compatibility
    /// only and will be removed in a future release.
    ///
    /// The real training path lives in
    /// `training_loop::TrainingLoop::compute_text_loss_and_grads`, which uses
    /// `nn::value_and_grad` to compute gradients and applies them via
    /// the optimizer.  Use `TrainingLoop` directly for functional LoRA training.
    #[deprecated(
        since = "0.1.0",
        note = "No backward pass is performed. Use `TrainingLoop::train` which calls \
                `compute_text_loss_and_grads` for a complete forward+backward+update cycle."
    )]
    pub fn train_step_lora(
        &mut self,
        lora_layers: &mut [&mut LoraLinear],
        input_ids: &Array,
        pixel_values: Option<&Array>,
        labels: &Array,
        forward_fn: impl Fn(&[&LoraLinear], &Array, Option<&Array>) -> Result<Array>,
    ) -> Result<f64> {
        // Forward pass only - no backward pass or parameter updates are
        // performed here.  See the deprecation notice above.
        let lora_refs: Vec<&LoraLinear> = lora_layers.iter().map(|l| &**l).collect();
        let logits = forward_fn(&lora_refs, input_ids, pixel_values)?;
        let loss = self.compute_loss(&logits, labels, None)?;

        let mut loss = loss;
        loss.eval();
        let loss_value = loss.item_f32() as f64;

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
            let mut grad_eval = grad.clone();
            grad_eval.eval();
            let mut norm_sq = grad.multiply(grad).sum(None);
            norm_sq.eval();
            total_norm_sq += norm_sq.item_f32() as f64;
        }
        let total_norm = total_norm_sq.sqrt();

        // Clip if necessary
        if total_norm > self.config.max_grad_norm {
            let scale = self.config.max_grad_norm / total_norm;
            for grad in grads.iter_mut() {
                let scale_arr = Array::from_f32(scale as f32);
                *grad = grad.multiply(&scale_arr);
            }
        }

        Ok(Some(total_norm))
    }

    /// Compute MTP (Multi-Token Prediction) loss for a batch.
    ///
    /// # Arguments
    /// * `all_logits` - List of logits for each prediction depth [D+1, batch, seq, vocab]
    /// * `labels` - Target labels [batch, seq]
    /// * `loss_weights` - Weights for each MTP depth (e.g. [1.0, 0.3, 0.3])
    ///
    /// # Returns
    /// Weighted scalar loss value
    pub fn compute_mtp_loss(
        &self,
        all_logits: &[Array],
        labels: &Array,
        loss_weights: &[f32],
    ) -> Result<Array> {
        if all_logits.is_empty() {
            return Err(SftError::Mlx(Exception::custom(
                "No logits provided for MTP loss",
            )));
        }

        let mut total_loss = Array::from_f32(0.0);
        let seq_len = all_logits[0].dim(1);
        let vocab_size = all_logits[0].dim(2);

        for (depth, logits) in all_logits.iter().enumerate() {
            let weight = if depth < loss_weights.len() {
                loss_weights[depth]
            } else {
                *loss_weights.last().unwrap_or(&0.1)
            };

            // Shift labels based on depth
            // Depth 0: predicts labels[1:]
            // Depth 1: predicts labels[2:]
            // ...
            let shift = depth + 1;
            if shift >= seq_len as usize {
                continue;
            }

            let d_logits = logits.index((.., ..seq_len - shift as i32, ..));
            let d_labels = labels.index((.., shift as i32..));

            let flat_logits = d_logits.reshape(&[-1, vocab_size]);
            let flat_labels = d_labels.reshape(&[-1]);

            let loss = cross_entropy_loss(&flat_logits, &flat_labels, Some(-100_i64), 0.0)?;
            let mean_loss = loss.mean(None);

            let weighted_loss = mean_loss.multiply(&Array::from_f32(weight));
            total_loss = total_loss.add(&weighted_loss);
        }

        Ok(total_loss)
    }

    /// Update learning rate based on scheduler.
    ///
    /// Delegates to the canonical `pmetal_core::LearningRateScheduler`.
    pub fn update_learning_rate(&mut self) {
        use pmetal_core::LearningRateScheduler;

        let step = self.state.step;
        let total_steps = self.config.max_steps.unwrap_or(10000);

        let scheduler = LearningRateScheduler::new(
            self.config.learning_rate,
            total_steps,
            self.config.warmup_steps,
            self.config.lr_scheduler,
        );

        self.state.learning_rate = scheduler.get_lr(step);
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

    /// Train the model.
    ///
    /// This legacy trainer no longer pretends to implement a full end-to-end
    /// loop. Use [`crate::TrainingLoop`] for production training.
    pub fn train(&mut self) -> pmetal_core::Result<()> {
        Err(pmetal_core::PMetalError::NotImplemented(
            "SftTrainer::train is a legacy surface. Use pmetal_trainer::TrainingLoop for a complete training loop with optimizer updates, evaluation, and checkpointing.".to_string(),
        ))
    }

    /// Evaluate the model.
    ///
    /// Use [`crate::TrainingLoop`] for production evaluation.
    pub fn evaluate(&self) -> pmetal_core::Result<EvalMetrics> {
        Err(pmetal_core::PMetalError::NotImplemented(
            "SftTrainer::evaluate is not implemented on the legacy trainer surface. Use TrainingLoop evaluation instead.".to_string(),
        ))
    }

    /// Save checkpoint.
    ///
    /// Use [`crate::CheckpointManager`] with [`crate::TrainingLoop`] for
    /// production checkpoint persistence.
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> pmetal_core::Result<()> {
        let path = path.as_ref();
        Err(pmetal_core::PMetalError::NotImplemented(format!(
            "SftTrainer::save_checkpoint does not persist state for {:?}. Use CheckpointManager via TrainingLoop instead.",
            path
        )))
    }

    /// Load checkpoint.
    ///
    /// Use [`crate::CheckpointManager`] with [`crate::TrainingLoop`] for
    /// production checkpoint restore.
    pub fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> pmetal_core::Result<()> {
        let path = path.as_ref();
        Err(pmetal_core::PMetalError::NotImplemented(format!(
            "SftTrainer::load_checkpoint does not restore state from {:?}. Use CheckpointManager via TrainingLoop instead.",
            path
        )))
    }
}

/// Compute the loss function for a language model.
///
/// This is a standalone function for use with `nn::value_and_grad`.
pub fn lm_loss(logits: &Array, labels: &Array, ignore_index: i64) -> Result<Array> {
    let seq_len = logits.dim(1);
    let vocab_size = logits.dim(2);

    // Shift: logits[:-1] predicts labels[1:]
    let shift_logits = logits.index((.., ..seq_len - 1, ..));
    let shift_labels = labels.index((.., 1..));

    // Reshape for cross entropy
    let flat_logits = shift_logits.reshape(&[-1, vocab_size]);
    let flat_labels = shift_labels.reshape(&[-1]);

    // Compute cross entropy loss
    let loss = cross_entropy_loss(&flat_logits, &flat_labels, Some(ignore_index), 0.0)?;

    Ok(loss.mean(None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::{ops::zeros, random};

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

        use pmetal_bridge::compat::Dtype;
        let logits = random::normal(&[batch, seq_len, vocab_size], Dtype::Float32);
        let labels = random::randint(0, vocab_size, &[batch, seq_len], Dtype::Int32);

        let loss = lm_loss(&logits, &labels, -100).unwrap();
        loss.eval();

        // Loss should be a scalar
        let empty_shape: &[i32] = &[];
        assert_eq!(loss.shape(), empty_shape);
        assert!(loss.item_f32() > 0.0);
    }

    #[test]
    fn test_compute_loss_with_fully_masked_batch_returns_zero() {
        let trainer = SftTrainer::new(TrainingConfig::default());
        use pmetal_bridge::compat::Dtype;
        let logits = zeros(&[1, 3, 8], Dtype::Float32);
        let labels = zeros(&[1, 3], Dtype::Int32);
        let mask = zeros(&[1, 3], Dtype::Float32);

        let loss = trainer.compute_loss(&logits, &labels, Some(&mask)).unwrap();
        loss.eval();

        assert_eq!(loss.item_f32(), 0.0);
    }

    #[test]
    fn test_legacy_train_surface_fails_explicitly() {
        let mut trainer = SftTrainer::new(TrainingConfig::default());
        let err = trainer.train().unwrap_err();

        match err {
            pmetal_core::PMetalError::NotImplemented(message) => {
                assert!(message.contains("TrainingLoop"));
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }
}
