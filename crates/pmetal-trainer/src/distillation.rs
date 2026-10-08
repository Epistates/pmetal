//! Knowledge Distillation Trainer.
//!
//! Implements the training loop for distilling knowledge from a teacher model
//! to a student model. Supports Online, Offline, and Progressive distillation.

use pmetal_bridge::compat::{Array, Dtype, Exception, nn, optimizers::Optimizer};
use pmetal_data::{DataLoader, TrainingBatch, TrainingDataset};
use pmetal_distill::{DistillLossOutput, Distiller, LogitCache};
use pmetal_lora::TrainableModel;
use pmetal_mlx::kernels::with_training_mode;

use crate::{
    AdaptiveAction, CheckpointManager, CheckpointMetadata, Result, SftError, StepStats,
    TrainingLoop, TrainingLoopConfig,
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

    pub fn set_progress(&mut self, step: usize, epoch: usize) {
        self.loop_state.set_step(step);
        self.loop_state.set_epoch(epoch);
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

    fn total_steps_for_dataset(&self, train_dataset: &TrainingDataset) -> usize {
        let dl = DataLoader::new(
            train_dataset.clone(),
            self.loop_state.config.dataloader.clone(),
            None,
        );
        dl.num_batches() * self.loop_state.config.training.num_epochs
    }

    fn distillation_weights(batch: &TrainingBatch) -> Array {
        batch
            .labels
            .greater_equal(&Array::from_i32(0))
            .as_dtype(Dtype::Float32.as_i32())
    }

    fn compute_distillation_output(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        labels: &Array,
        weights: &Array,
        current_step: usize,
        total_steps: usize,
    ) -> Result<DistillLossOutput> {
        let labels_opt = if labels.size() > 0 {
            Some(labels)
        } else {
            None
        };
        self.distiller
            .compute_loss(
                teacher_logits,
                student_logits,
                labels_opt,
                Some(weights),
                current_step,
                total_steps.max(1),
            )
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))
    }

    fn load_teacher_logits_from_cache(
        &self,
        cache: &LogitCache,
        batch: &TrainingBatch,
    ) -> Result<Array> {
        let vocab_size = cache.metadata().vocab_size;
        if vocab_size == 0 {
            return Err(SftError::Mlx(Exception::custom(
                "offline distillation cache metadata is missing vocab_size",
            )));
        }

        let mut batch_logits = vec![-1.0e10_f32; batch.batch_size * batch.seq_len * vocab_size];

        for (batch_row, &sample_index) in batch.sample_indices.iter().enumerate() {
            let sequence = cache
                .load_sequence(sample_index)
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
            let shape = sequence.shape().to_vec();
            let (seq_len, cached_vocab) = match shape.as_slice() {
                [1, seq_len, vocab] => (*seq_len as usize, *vocab as usize),
                [seq_len, vocab] => (*seq_len as usize, *vocab as usize),
                other => {
                    return Err(SftError::Mlx(Exception::custom(format!(
                        "cached teacher logits for sample {sample_index} have unsupported shape {other:?}"
                    ))));
                }
            };

            if cached_vocab != vocab_size {
                return Err(SftError::Mlx(Exception::custom(format!(
                    "cached teacher logits for sample {sample_index} use vocab {cached_vocab}, expected {vocab_size}"
                ))));
            }

            let numel: usize = shape.iter().map(|&dim| dim as usize).product();
            let flat = sequence.clone().to_f32_vec(numel).ok_or_else(|| {
                SftError::Mlx(Exception::custom(format!(
                    "failed to materialize cached teacher logits for sample {sample_index}"
                )))
            })?;

            let copy_seq_len = seq_len.min(batch.seq_len);
            for token_idx in 0..copy_seq_len {
                let src_offset = token_idx * vocab_size;
                let dst_offset = (batch_row * batch.seq_len + token_idx) * vocab_size;
                batch_logits[dst_offset..dst_offset + vocab_size]
                    .copy_from_slice(&flat[src_offset..src_offset + vocab_size]);
            }
        }

        Ok(Array::from_f32_slice(
            &batch_logits,
            &[
                batch.batch_size as i32,
                batch.seq_len as i32,
                vocab_size as i32,
            ],
        ))
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
        total_steps: usize,
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
        self.train_step_with_teacher_logits(
            student,
            batch,
            &teacher_logits,
            optimizer,
            batch_tokens,
            start_time,
            total_steps,
        )
    }

    fn train_step_with_teacher_logits<S, O>(
        &mut self,
        student: &mut S,
        batch: &TrainingBatch,
        teacher_logits: &Array,
        optimizer: &mut O,
        batch_tokens: usize,
        start_time: std::time::Instant,
        total_steps: usize,
    ) -> Result<StepStats>
    where
        S: TrainableModel,
        O: Optimizer,
    {
        let current_step = self.loop_state.step;
        let weights = Self::distillation_weights(batch);
        let loss_fn =
            |student: &mut S,
             (input_ids, labels, teacher_logits, weights): (&Array, &Array, &Array, &Array)|
             -> std::result::Result<Array, Exception> {
                let student_logits = student
                    .forward(input_ids, None)
                    .map_err(|e| Exception::custom(e.to_string()))?;

                let output = self
                    .compute_distillation_output(
                        teacher_logits,
                        &student_logits,
                        labels,
                        weights,
                        current_step,
                        total_steps,
                    )
                    .map_err(|e| Exception::custom(e.to_string()))?;

                Ok(output.total)
            };

        let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);

        let (loss, grads) = if self.loop_state.metal_fa_available {
            let result = with_training_mode(|| {
                loss_and_grad_fn(
                    student,
                    (&batch.input_ids, &batch.labels, teacher_logits, &weights),
                )
                .map_err(|e| pmetal_mlx::error::MlxError::from(e))
            });
            result.map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?
        } else {
            loss_and_grad_fn(
                student,
                (&batch.input_ids, &batch.labels, teacher_logits, &weights),
            )?
        };

        let loss_val = loss.item_f32();
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

    fn run_offline_eval<S>(
        &mut self,
        student: &mut S,
        cache: &LogitCache,
        eval_dataset: &TrainingDataset,
        total_steps: usize,
    ) -> Result<f64>
    where
        S: TrainableModel,
    {
        let mut eval_config = self.loop_state.config.dataloader.clone();
        eval_config.shuffle = false;
        eval_config.drop_last = false;

        let mut dataloader = DataLoader::new(eval_dataset.clone(), eval_config, None);
        let mut total_loss = 0.0_f64;
        let mut num_batches = 0_usize;

        while let Some(batch) = dataloader
            .try_next_batch()
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?
        {
            let teacher_logits = self.load_teacher_logits_from_cache(cache, &batch)?;
            let student_logits = student
                .forward(&batch.input_ids, None)
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
            let weights = Self::distillation_weights(&batch);
            let output = self.compute_distillation_output(
                &teacher_logits,
                &student_logits,
                &batch.labels,
                &weights,
                self.loop_state.step.min(total_steps.max(1)),
                total_steps,
            )?;

            total_loss += output.total.item_f32() as f64;
            num_batches += 1;
        }

        Ok(if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
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
        let mut optimizer = self.loop_state.build_optimizer();

        let num_epochs = self.loop_state.config.training.num_epochs;
        let checkpoint_every = self.loop_state.config.checkpoint_every;

        // Track the best eval (or train) loss seen so far for "is_best" tagging.
        let mut best_loss = f64::MAX;

        // Estimate total steps for progress reporting
        let total_steps = self.total_steps_for_dataset(&train_dataset);
        if let Some(ref mut ctrl) = self.loop_state.adaptive_lr {
            ctrl.set_total_steps(total_steps);
        }

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

            while let Some(batch) = dataloader
                .try_next_batch()
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?
            {
                let step_start = std::time::Instant::now();
                let stats =
                    self.train_step(student, teacher, &batch, &mut optimizer, total_steps)?;
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
                        let loss = self.run_eval(student, teacher, eval_ds, total_steps)?;
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

    /// Run offline distillation using a precomputed teacher-logit cache.
    pub fn run_offline<S>(
        &mut self,
        student: &mut S,
        cache: &LogitCache,
        train_dataset: TrainingDataset,
        eval_dataset: Option<TrainingDataset>,
        checkpoint_manager: Option<&CheckpointManager>,
    ) -> Result<()>
    where
        S: TrainableModel,
    {
        let mut optimizer = self.loop_state.build_optimizer();

        let num_epochs = self.loop_state.config.training.num_epochs;
        let checkpoint_every = self.loop_state.config.checkpoint_every;
        let mut best_loss = f64::MAX;
        let total_steps = self.total_steps_for_dataset(&train_dataset);

        if cache.metadata().num_sequences < train_dataset.len() {
            return Err(SftError::Mlx(Exception::custom(format!(
                "offline distillation cache is incomplete: expected at least {} sequences, found {}",
                train_dataset.len(),
                cache.metadata().num_sequences
            ))));
        }

        if let Some(ref mut ctrl) = self.loop_state.adaptive_lr {
            ctrl.set_total_steps(total_steps);
        }

        for cb in &mut self.loop_state.callbacks {
            cb.on_train_start();
        }

        tracing::info!("Starting offline distillation ({total_steps} steps)...");

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

            while let Some(batch) = dataloader
                .try_next_batch()
                .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?
            {
                let step_start = std::time::Instant::now();
                let batch_tokens = batch
                    .batch_size
                    .checked_mul(batch.seq_len)
                    .unwrap_or(usize::MAX);
                let teacher_logits = self.load_teacher_logits_from_cache(cache, &batch)?;
                let stats = self.train_step_with_teacher_logits(
                    student,
                    &batch,
                    &teacher_logits,
                    &mut optimizer,
                    batch_tokens,
                    step_start,
                    total_steps,
                )?;
                let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                self.loop_state.step += 1;

                let action = self.loop_state.apply_adaptive_lr(stats.loss as f64);

                if action == AdaptiveAction::Continue && self.loop_state.should_snapshot_best() {
                    self.loop_state.snapshot_best_weights(student);
                }
                if action == AdaptiveAction::Rollback {
                    self.loop_state.restore_best_weights(student);
                }
                if action == AdaptiveAction::EarlyStop {
                    self.loop_state.restore_best_weights(student);
                    tracing::info!(
                        "Early stopping offline distillation — best checkpoint restored."
                    );
                    return Ok(());
                }

                let next_lr = self.loop_state.get_learning_rate();
                optimizer.set_learning_rate(next_lr);
                let adjusted_lr = next_lr;

                if self.loop_state.step % self.loop_state.config.log_every == 0 {
                    tracing::info!(
                        "Step {}: loss={:.4}, lr={:.2e}",
                        stats.step,
                        stats.loss,
                        adjusted_lr
                    );
                }

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

                if checkpoint_every > 0 && self.loop_state.step % checkpoint_every == 0 {
                    let eval_loss: Option<f64> = if let Some(ref eval_ds) = eval_dataset {
                        let loss = self.run_offline_eval(student, cache, eval_ds, total_steps)?;
                        tracing::info!("Step {}: eval_loss={:.4}", self.loop_state.step, loss);
                        Some(loss)
                    } else {
                        None
                    };

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
        total_steps: usize,
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

        while let Some(batch) = dataloader
            .try_next_batch()
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?
        {
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

            let weights = Self::distillation_weights(&batch);
            let output = self.compute_distillation_output(
                &teacher_logits,
                &student_logits,
                &batch.labels,
                &weights,
                self.loop_state.step.min(total_steps.max(1)),
                total_steps,
            )?;

            total_loss += output.total.item_f32() as f64;
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

/// Generate or resume an offline teacher-logit cache for distillation.
pub fn generate_teacher_logit_cache<T>(
    teacher: &mut T,
    dataset: &TrainingDataset,
    cache: &mut LogitCache,
    teacher_id: &str,
    max_seq_len: usize,
) -> Result<()>
where
    T: TrainableModel,
{
    let mut vocab_size = cache.metadata().vocab_size;

    for (sample_index, sample) in dataset.samples().iter().enumerate() {
        if cache.has_sequence(sample_index) {
            continue;
        }

        let seq_len = sample.input_ids.len().min(max_seq_len);
        let input_ids: Vec<i32> = sample
            .input_ids
            .iter()
            .take(seq_len)
            .map(|&token| token as i32)
            .collect();
        let input = Array::from_i32_slice(&input_ids).reshape(&[1, seq_len as i32]);
        let logits = teacher
            .forward(&input, None)
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
        logits.eval();

        if vocab_size == 0 {
            vocab_size = logits.dim(-1) as usize;
        }

        cache
            .cache_sequence(sample_index, &logits)
            .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
    }

    if vocab_size == 0 {
        return Err(SftError::Mlx(Exception::custom(
            "cannot generate teacher-logit cache for an empty dataset",
        )));
    }

    cache.set_metadata(teacher_id.to_string(), vocab_size, max_seq_len);
    cache
        .save_metadata()
        .map_err(|e| SftError::Mlx(Exception::custom(e.to_string())))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_data::DataLoaderConfig;
    use pmetal_distill::{CompressionMethod, DistillConfig};
    use tempfile::tempdir;

    fn test_trainer() -> DistillationTrainer {
        let distiller = Distiller::new(DistillConfig {
            teacher: "teacher".to_string(),
            student: "student".to_string(),
            method: pmetal_distill::DistillMethod::Online,
            loss: pmetal_distill::LossConfig::default(),
            offline: None,
            output_path: None,
            training: pmetal_distill::TrainingConfig::default(),
        })
        .unwrap();

        DistillationTrainer::new(
            distiller,
            TrainingLoopConfig {
                dataloader: DataLoaderConfig {
                    batch_size: 2,
                    max_seq_len: 8,
                    shuffle: false,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
    }

    #[test]
    fn load_teacher_logits_from_cache_respects_batch_indices_and_padding() {
        let tempdir = tempdir().unwrap();
        let mut cache = LogitCache::new(tempdir.path(), CompressionMethod::None, 128).unwrap();
        cache.set_metadata("teacher".to_string(), 3, 8);
        cache
            .cache_sequence(
                0,
                &Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]),
            )
            .unwrap();
        cache
            .cache_sequence(
                1,
                &Array::from_f32_slice(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[1, 2, 3]),
            )
            .unwrap();
        cache.save_metadata().unwrap();

        let trainer = test_trainer();
        let batch = TrainingBatch {
            input_ids: Array::from_i32_slice(&[1, 2, 3, 4, 5, 0]).reshape(&[2, 3]),
            labels: Array::from_i32_slice(&[1, 2, 3, 4, 5, -100]).reshape(&[2, 3]),
            attention_mask: Array::from_i32_slice(&[1, 1, 1, 1, 1, 0]).reshape(&[2, 3]),
            pixel_values: None,
            batch_size: 2,
            seq_len: 3,
            sample_indices: vec![1, 0],
        };

        let mut logits = trainer
            .load_teacher_logits_from_cache(&cache, &batch)
            .unwrap();
        let values = logits.to_f32_vec(18).unwrap();

        assert_eq!(&values[0..6], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        assert_eq!(&values[9..15], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(values[6..9].iter().all(|v| *v < -1.0e9));
        assert!(values[15..18].iter().all(|v| *v < -1.0e9));
    }
}
