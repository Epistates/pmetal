//! Knowledge Distillation Trainer.
//!
//! Implements the training loop for distilling knowledge from a teacher model
//! to a student model. Supports Online, Offline, and Progressive distillation.

use pmetal_data::{DataLoader, TrainingDataset};
use pmetal_distill::{Distiller, DistillLossOutput};
use pmetal_lora::TrainableModel;
use pmetal_mlx::kernels::with_training_mode;
use mlx_rs::{
    error::Exception,
    nn,
    optimizers::Optimizer,
    Array,
};

use crate::{
    CheckpointManager, Result, SftError, StepStats, TrainingLoop, TrainingLoopConfig,
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
        let batch_tokens = batch.batch_size.checked_mul(batch.seq_len).unwrap_or(usize::MAX);

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
            let labels_opt = if labels.size() > 0 { Some(labels) } else { None };
            
            let output: DistillLossOutput = self.distiller
                .compute_loss(teacher_logits, &student_logits, labels_opt)
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

        loss.eval()?;
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
        _eval_dataset: Option<TrainingDataset>,
        _checkpoint_manager: Option<&CheckpointManager>,
    ) -> Result<()>
    where
        S: TrainableModel,
        T: TrainableModel,
    {
        // Setup optimizer
        let base_lr = self.loop_state.config.training.learning_rate as f32;
        // let _weight_decay = self.loop_state.config.training.weight_decay as f32;
        
        // TODO: Support embedding LR if needed
        let mut optimizer = mlx_rs::optimizers::AdamW::new(base_lr);
        // Set weight decay if exposed, mlx-rs AdamW might not expose builder easily here
        // assuming standard AdamW for now.

        let num_epochs = self.loop_state.config.training.num_epochs;

        tracing::info!("Starting distillation...");

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
                let stats = self.train_step(student, teacher, &batch, &mut optimizer)?;
                
                self.loop_state.step += 1;
                
                if self.loop_state.step % self.loop_state.config.log_every == 0 {
                    tracing::info!(
                        "Step {}: loss={:.4}, lr={:.2e}",
                        stats.step, stats.loss, stats.learning_rate
                    );
                }
                
                // TODO: Checkpointing and Evaluation logic
            }
        }

        Ok(())
    }
}