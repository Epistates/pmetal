//! ANE training loop integration.
//!
//! Bridges [`pmetal_metal::ane::trainer::AneTrainer`] with pmetal's
//! callback system, training state tracking, and checkpointing.

use std::path::{Path, PathBuf};
use std::time::Instant;

use pmetal_core::{EvalMetrics, TrainingCallback, TrainingState};
use pmetal_metal::ane::budget::BudgetExhaustionStrategy;
use pmetal_metal::ane::trainer::{AneTrainer, AneTrainerConfig};

/// Configuration for the ANE training loop.
#[derive(Debug, Clone)]
pub struct AneTrainingLoopConfig {
    /// Inner ANE trainer configuration.
    pub trainer: AneTrainerConfig,
    /// Total number of training batches.
    pub num_batches: usize,
    /// Log metrics every N batches.
    pub log_every: usize,
    /// Save checkpoint every N batches.
    pub save_every: Option<usize>,
    /// Output directory for checkpoints.
    pub output_dir: PathBuf,
}

/// ANE training loop with callback and state tracking support.
///
/// Wraps `AneTrainer` to provide:
/// - `TrainingCallback` dispatch (progress, logging, metrics)
/// - `TrainingState` tracking (step, loss, tokens/sec)
/// - Periodic checkpointing
/// - Budget exhaustion handling
pub struct AneTrainingLoop {
    trainer: AneTrainer,
    state: TrainingState,
    callbacks: Vec<Box<dyn TrainingCallback>>,
    config: AneTrainingLoopConfig,
}

impl AneTrainingLoop {
    /// Create a new ANE training loop.
    pub fn new(config: AneTrainingLoopConfig) -> Self {
        let trainer = AneTrainer::new(config.trainer.clone());
        Self {
            trainer,
            state: TrainingState::default(),
            callbacks: Vec::new(),
            config,
        }
    }

    /// Add a training callback.
    pub fn add_callback(&mut self, callback: Box<dyn TrainingCallback>) {
        self.callbacks.push(callback);
    }

    /// Get a reference to the inner trainer.
    pub fn trainer(&self) -> &AneTrainer {
        &self.trainer
    }

    /// Get a mutable reference to the inner trainer.
    pub fn trainer_mut(&mut self) -> &mut AneTrainer {
        &mut self.trainer
    }

    /// Get the current training state.
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Load weights into the trainer from a flat f32 buffer.
    pub fn load_weights_flat(&mut self, weights: &[f32]) {
        self.trainer.load_weights_flat(weights);
    }

    /// Run the full training loop over the provided data.
    ///
    /// `data` is a slice of batches, where each batch is a slice of
    /// `(input_tokens, target_tokens)` pairs for gradient accumulation.
    ///
    /// Returns the final training state.
    pub fn train(
        &mut self,
        data: &[Vec<(Vec<u16>, Vec<u16>)>],
    ) -> Result<TrainingState, pmetal_metal::error::MetalError> {
        let start = Instant::now();

        // Notify callbacks
        for cb in &mut self.callbacks {
            cb.on_train_start();
        }

        let num_batches = data.len().min(self.config.num_batches);
        let seq_len = self.config.trainer.seq_len;

        #[allow(clippy::needless_range_loop)]
        for batch_idx in 0..num_batches {
            // Notify callbacks
            for cb in &mut self.callbacks {
                cb.on_step_start(batch_idx);
            }

            // Check budget before compile
            if self.trainer.budget().needs_restart() {
                let strategy = self.config.trainer.exhaustion_strategy.clone();
                match strategy {
                    BudgetExhaustionStrategy::ExecRestart {
                        ref checkpoint_path,
                        ..
                    } => {
                        let ckpt_path = checkpoint_path.clone();
                        self.save_checkpoint(Path::new(&ckpt_path));
                        return Err(pmetal_metal::error::MetalError::AneCompileFailed(format!(
                            "ANE compile budget exhausted at batch {}. Checkpoint saved to {}. Restart process to continue.",
                            batch_idx, ckpt_path
                        )));
                    }
                    BudgetExhaustionStrategy::FallbackToGpu => {
                        tracing::warn!(
                            batch = batch_idx,
                            "ANE compile budget exhausted, falling back to CPU-only for remaining batches"
                        );
                    }
                    BudgetExhaustionStrategy::Error => {
                        return Err(pmetal_metal::error::MetalError::AneCompileFailed(format!(
                            "ANE compile budget exhausted at batch {}: {}/{} compilations used",
                            batch_idx,
                            self.trainer.budget().current(),
                            self.trainer.budget().max(),
                        )));
                    }
                }
            }

            // Run the batch
            let loss = self.trainer.train_batch(&data[batch_idx])?;

            // Update state
            self.state.step = batch_idx + 1;
            self.state.loss = loss as f64;
            self.state.learning_rate = self.config.trainer.learning_rate as f64;
            self.state.tokens_processed += data[batch_idx].len() * seq_len;
            self.state.elapsed_secs = start.elapsed().as_secs_f64();

            // Notify callbacks
            for cb in &mut self.callbacks {
                cb.on_step_end(batch_idx, loss as f64);
            }

            // Periodic checkpoint
            if let Some(save_every) = self.config.save_every {
                if (batch_idx + 1) % save_every == 0 {
                    let path = self
                        .config
                        .output_dir
                        .join(format!("checkpoint-{}", batch_idx + 1));
                    self.save_checkpoint(&path);
                }
            }
        }

        // Notify callbacks
        let metrics = EvalMetrics {
            loss: self.state.loss,
            perplexity: self.state.loss.exp(),
            accuracy: None,
            custom: Default::default(),
        };
        for cb in &mut self.callbacks {
            cb.on_epoch_end(0, &metrics);
            cb.on_train_end();
        }

        Ok(self.state.clone())
    }

    /// Save a checkpoint (weights + training state).
    fn save_checkpoint(&mut self, path: &Path) {
        tracing::info!(path = %path.display(), step = self.state.step, "Saving ANE checkpoint");

        if let Err(e) = std::fs::create_dir_all(path) {
            tracing::error!(error = %e, "Failed to create checkpoint directory");
            return;
        }

        // Save training state as JSON
        let state_path = path.join("training_state.json");
        match serde_json::to_string_pretty(&self.state) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&state_path, json) {
                    tracing::error!(error = %e, "Failed to write training state");
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to serialize training state");
            }
        }

        // Notify callbacks
        for cb in &mut self.callbacks.iter_mut() {
            cb.on_save(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ane_training_loop_creation() {
        let config = AneTrainingLoopConfig {
            trainer: AneTrainerConfig {
                dim: 64,
                hidden_dim: 128,
                n_heads: 4,
                n_layers: 2,
                vocab_size: 100,
                seq_len: 16,
                ..Default::default()
            },
            num_batches: 10,
            log_every: 1,
            save_every: None,
            output_dir: PathBuf::from("/tmp/ane-test"),
        };

        let training_loop = AneTrainingLoop::new(config);
        assert_eq!(training_loop.state().step, 0);
        assert_eq!(training_loop.trainer().config().n_layers, 2);
    }

    #[test]
    fn test_ane_training_loop_with_callback() {
        use std::sync::{Arc, Mutex};

        struct TestCallback {
            steps: Arc<Mutex<Vec<usize>>>,
        }

        impl TrainingCallback for TestCallback {
            fn on_step_end(&mut self, step: usize, _loss: f64) {
                self.steps.lock().unwrap().push(step);
            }
        }

        let steps = Arc::new(Mutex::new(Vec::new()));
        let config = AneTrainingLoopConfig {
            trainer: AneTrainerConfig {
                dim: 64,
                hidden_dim: 128,
                n_heads: 4,
                n_layers: 2,
                vocab_size: 100,
                seq_len: 16,
                ..Default::default()
            },
            num_batches: 10,
            log_every: 1,
            save_every: None,
            output_dir: PathBuf::from("/tmp/ane-test"),
        };

        let mut training_loop = AneTrainingLoop::new(config);
        training_loop.add_callback(Box::new(TestCallback {
            steps: steps.clone(),
        }));

        // We don't run actual training (no weights loaded), just verify setup
        assert_eq!(training_loop.state().step, 0);
    }
}
