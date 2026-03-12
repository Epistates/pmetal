//! Knowledge Distillation Trainer.
//!
//! Implements the training loop for distilling knowledge from a teacher model
//! to a student model. Supports Online, Offline, and Progressive distillation.

use mlx_rs::{Array, error::Exception, nn, optimizers::Optimizer};
use pmetal_data::{DataLoader, TrainingDataset};
use pmetal_distill::{DistillLossOutput, Distiller};
use pmetal_lora::TrainableModel;
use pmetal_mlx::kernels::with_training_mode;

use crate::{
    AdamWGroups, AdaptiveAction, CheckpointManager, CheckpointMetadata, Result, SftError,
    StepStats, TrainingLoop, TrainingLoopConfig,
};

/// Trainer for Knowledge Distillation.
pub struct DistillationTrainer {
    /// The distillation engine (holds config and loss logic).
    distiller: Distiller,
    /// Underlying training loop state.
    loop_state: TrainingLoop,
}

impl DistillationTrainer {
    /// Create a new DistillationTrainer.
    pub fn new(distiller: Distiller, config: TrainingLoopConfig) -> Self {
        Self {
            distiller,
            loop_state: TrainingLoop::new(config),
        }
    }

    /// Add a training callback for metrics logging or dashboard integration.
    pub fn add_callback(&mut self, callback: Box<dyn pmetal_core::TrainingCallback>) {
        self.loop_state.add_callback(callback);
    }

    /// Enable adaptive LR with control file for TUI communication.
    pub fn enable_adaptive_lr_with_control(
        &mut self,
        config: crate::adaptive_lr::AdaptiveLrConfig,
        control_file: std::path::PathBuf,
    ) {
        self.loop_state
            .enable_adaptive_lr_with_control(config, control_file);
    }

    /// Perform a single distillation step.
    ///
    /// # Arguments
    /// * `student` - The student model (trainable).
    /// * `teacher` - The teacher model (frozen/inference).
    /// * `batch` - Training batch.
    /// * `optimizer` - Optimizer for student.
    pub fn train_step<S, T, O>(
        &mut self,
        student: &mut S,
        teacher: &mut T,
        batch: &pmetal_data::TrainingBatch,
        optimizer: &mut O,
    ) -> Result<StepStats>
    where
        S: TrainableModel,
        T: TrainableModel, // Teacher must be forward-able
        O: Optimizer,
    {
        let start_time = std::time::Instant::now();
        let batch_tokens = batch
            .batch_size
            .checked_mul(batch.seq_len)
            .unwrap_or(usize::MAX);

        // 1. Teacher Forward Pass (No Grad)
        // We run this outside the autodiff scope to save memory/compute
        let teacher_logits = teacher
            .forward(&batch.input_ids, None)
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

        // Note: No explicit stop_gradient needed here as these logits enter the loss function
        // as a constant input (not the first argument to value_and_grad).

        // 2. Define Loss Function for Student
        let loss_fn = |student: &mut S,
                       (input_ids, labels, teacher_logits): (&Array, &Array, &Array)|
         -> std::result::Result<Array, Exception> {
            // Student Forward
            let student_logits = student
                .forward(input_ids, None)
                .map_err(|e| Exception::custom(e.to_string()))?;

            // Compute Distillation Loss
            // We can optionally pass labels for "hard" loss component
            let labels_opt = if labels.size() > 0 {
                Some(labels)
            } else {
                None
            };

            let output: DistillLossOutput = self
                .distiller
                .compute_loss(teacher_logits, &student_logits, labels_opt, None)
                .map_err(|e| Exception::custom(e.to_string()))?;

            Ok(output.total)
        };

        // 3. Student Backward Pass & Update
        let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);

        // Use Metal FlashAttention if available
        let (loss, grads) = if self.loop_state.metal_fa_available {
            let result = with_training_mode(|| {
                loss_and_grad_fn(student, (&batch.input_ids, &batch.labels, &teacher_logits))
                    .map_err(|e| pmetal_mlx::error::MlxError::from(e))
            });
            result.map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?
        } else {
            loss_and_grad_fn(student, (&batch.input_ids, &batch.labels, &teacher_logits))?
        };

        let loss_val = loss.item::<f32>();

        // Apply gradients (gradient accumulation logic handles the actual update)
        // Re-using logic from TrainingLoop would be ideal, but here we manually do it
        // or we expose accumulator. For now, simple update:
        optimizer.update(student, grads)?;

        let step_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(StepStats {
            step: self.loop_state.step,
            loss: loss_val,
            learning_rate: self.loop_state.get_learning_rate(),
            tokens: batch_tokens,
            grad_norm: None,
            step_time_ms,
        })
    }

    /// Run the distillation loop.
    pub fn run<S, T>(
        &mut self,
        student: &mut S,
        teacher: &mut T,
        train_dataset: TrainingDataset,
        eval_dataset: Option<TrainingDataset>,
        checkpoint_manager: Option<&CheckpointManager>,
    ) -> Result<()>
    where
        S: TrainableModel,
        T: TrainableModel,
    {
        // Setup optimizer with AdamWGroups to support a separate embedding learning rate.
        let base_lr = self.loop_state.config.training.learning_rate as f32;
        let weight_decay = self.loop_state.config.training.weight_decay as f32;
        let embedding_lr = self.loop_state.config.embedding_lr;

        if let Some(emb_lr) = embedding_lr {
            tracing::info!(
                "Using separate embedding LR: {:.2e} (base: {:.2e})",
                emb_lr,
                base_lr
            );
        }

        let mut optimizer =
            AdamWGroups::new(base_lr, embedding_lr, weight_decay).map_err(SftError::Mlx)?;

        let num_epochs = self.loop_state.config.training.num_epochs;
        let checkpoint_every = self.loop_state.config.checkpoint_every;

        // Track the best eval (or train) loss seen so far for "is_best" tagging.
        let mut best_loss = f64::MAX;

        // Estimate total steps for progress reporting
        let steps_per_epoch = {
            let dl = DataLoader::new(
                train_dataset.clone(),
                self.loop_state.config.dataloader.clone(),
                None,
            );
            dl.num_batches()
        };
        let total_steps = steps_per_epoch * num_epochs;

        // Notify callbacks
        for cb in &mut self.loop_state.callbacks {
            cb.on_train_start();
        }

        tracing::info!("Starting distillation ({total_steps} steps)...");

        for epoch in 0..num_epochs {
            self.loop_state.epoch = epoch;

            let mut dataloader = DataLoader::new(
                train_dataset.clone(),
                self.loop_state.config.dataloader.clone(),
                None,
            );

            if epoch > 0 {
                dataloader.reset(Some(self.loop_state.config.dataloader.seed + epoch as u64));
            }

            while let Some(batch) = dataloader.next_batch() {
                let step_start = std::time::Instant::now();
                let stats = self.train_step(student, teacher, &batch, &mut optimizer)?;
                let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                self.loop_state.step += 1;

                // Feed loss to adaptive LR controller for next step
                let action = self.loop_state.apply_adaptive_lr(stats.loss as f64);

                if action == AdaptiveAction::Continue && self.loop_state.should_snapshot_best() {
                    self.loop_state.snapshot_best_weights(student);
                }
                if action == AdaptiveAction::Rollback {
                    self.loop_state.restore_best_weights(student);
                }
                if action == AdaptiveAction::EarlyStop {
                    self.loop_state.restore_best_weights(student);
                    tracing::info!("Early stopping distillation — best checkpoint restored.");
                    return Ok(());
                }

                // Apply adjusted LR to optimizer
                let next_lr = self.loop_state.get_learning_rate();
                optimizer.set_learning_rate(next_lr);

                // Use the adjusted LR (post-adaptive) for logging and metrics
                let adjusted_lr = next_lr;

                if self.loop_state.step % self.loop_state.config.log_every == 0 {
                    tracing::info!(
                        "Step {}: loss={:.4}, lr={:.2e}",
                        stats.step,
                        stats.loss,
                        adjusted_lr
                    );
                }

                // Emit metrics to callbacks
                let tok_sec = if step_ms > 0.0 {
                    stats.tokens as f64 / (step_ms / 1000.0)
                } else {
                    0.0
                };
                let metrics = pmetal_core::StepMetrics {
                    step: self.loop_state.step,
                    epoch,
                    total_epochs: num_epochs,
                    total_steps,
                    loss: stats.loss as f64,
                    lr: adjusted_lr as f64,
                    tok_sec,
                    total_ms: step_ms,
                    tokens: stats.tokens,
                    ..Default::default()
                };
                for cb in &mut self.loop_state.callbacks {
                    cb.on_step_end_with_metrics(&metrics);
                }

                // Checkpointing and Evaluation logic
                if checkpoint_every > 0 && self.loop_state.step % checkpoint_every == 0 {
                    // Run evaluation if an eval dataset is provided.
                    let eval_loss: Option<f64> = if let Some(ref eval_ds) = eval_dataset {
                        let loss = self.run_eval(student, teacher, eval_ds)?;
                        tracing::info!("Step {}: eval_loss={:.4}", self.loop_state.step, loss);
                        Some(loss)
                    } else {
                        None
                    };

                    // Use eval loss when available, otherwise fall back to train loss.
                    let reference_loss = eval_loss.unwrap_or(stats.loss as f64);
                    let is_best = reference_loss < best_loss;
                    if is_best {
                        best_loss = reference_loss;
                    }

                    if let Some(manager) = checkpoint_manager {
                        let lora_params = student.lora_parameters();
                        let mut metadata = CheckpointMetadata::new(
                            self.loop_state.step,
                            self.loop_state.epoch,
                            stats.loss as f64,
                            stats.learning_rate as f64,
                        );
                        if let Some(loss) = eval_loss {
                            metadata = metadata.with_best_val_loss(loss);
                        }
                        manager.save_checkpoint(&lora_params, &metadata, is_best)?;
                    }
                }
            }
        }

        for cb in &mut self.loop_state.callbacks {
            cb.on_train_end();
        }

        Ok(())
    }

    /// Evaluate the student against the teacher on a held-out dataset.
    ///
    /// Runs forward passes for both models without gradient tracking and returns
    /// the average distillation loss across all eval batches.
    fn run_eval<S, T>(
        &mut self,
        student: &mut S,
        teacher: &mut T,
        eval_dataset: &TrainingDataset,
    ) -> Result<f64>
    where
        S: TrainableModel,
        T: TrainableModel,
    {
        let mut eval_config = self.loop_state.config.dataloader.clone();
        eval_config.shuffle = false;
        eval_config.drop_last = false;

        let mut dataloader = DataLoader::new(eval_dataset.clone(), eval_config, None);

        let mut total_loss = 0.0_f64;
        let mut num_batches = 0_usize;

        while let Some(batch) = dataloader.next_batch() {
            // Teacher forward (no gradient needed; teacher is outside autodiff scope).
            let teacher_logits = teacher
                .forward(&batch.input_ids, None)
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

            // Student forward (no gradient needed during eval).
            let student_logits = student
                .forward(&batch.input_ids, None)
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

            let labels_opt = if batch.labels.size() > 0 {
                Some(&batch.labels)
            } else {
                None
            };

            let output: DistillLossOutput = self
                .distiller
                .compute_loss(&teacher_logits, &student_logits, labels_opt, None)
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;

            total_loss += output.total.item::<f32>() as f64;
            num_batches += 1;
        }

        let avg_loss = if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        };

        Ok(avg_loss)
    }
}
