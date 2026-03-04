//! ANE training loop integration (dynamic weight pipeline).
//!
//! Bridges [`pmetal_metal::ane::dynamic_trainer::DynamicAneTrainer`] with
//! pmetal's callback system, training state tracking, and checkpointing.
//!
//! The dynamic pipeline compiles 9 ANE kernels once at startup and never
//! recompiles. Weight updates are injected via IOSurface memcpy.

use std::path::{Path, PathBuf};
use std::time::Instant;

use pmetal_core::{EvalMetrics, StepMetrics, TrainingCallback, TrainingState};
use pmetal_metal::ane::dynamic_trainer::{DynamicAneTrainer, DynamicAneTrainerConfig};

/// Configuration for the ANE training loop.
#[derive(Debug, Clone)]
pub struct AneTrainingLoopConfig {
    /// Inner dynamic ANE trainer configuration.
    pub trainer: DynamicAneTrainerConfig,
    /// Total number of training batches.
    pub num_batches: usize,
    /// Total max steps for LR schedule.
    pub max_steps: usize,
    /// Log metrics every N batches.
    pub log_every: usize,
    /// Save checkpoint every N batches.
    pub save_every: Option<usize>,
    /// Output directory for checkpoints.
    pub output_dir: PathBuf,
}

/// ANE training loop with callback and state tracking support.
///
/// Wraps `DynamicAneTrainer` to provide:
/// - `TrainingCallback` dispatch (progress, logging, metrics)
/// - `TrainingState` tracking (step, loss, tokens/sec)
/// - Periodic checkpointing
/// - No budget exhaustion handling needed (9 compiles total)
pub struct AneTrainingLoop {
    trainer: DynamicAneTrainer,
    state: TrainingState,
    callbacks: Vec<Box<dyn TrainingCallback>>,
    config: AneTrainingLoopConfig,
}

impl AneTrainingLoop {
    /// Create a new ANE training loop.
    pub fn new(config: AneTrainingLoopConfig) -> Self {
        let trainer = DynamicAneTrainer::new(config.trainer.clone());
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
    pub fn trainer(&self) -> &DynamicAneTrainer {
        &self.trainer
    }

    /// Get a mutable reference to the inner trainer.
    pub fn trainer_mut(&mut self) -> &mut DynamicAneTrainer {
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

    /// Load weights from SafeTensors files on disk.
    pub fn load_weights_safetensors(
        &mut self,
        path: &Path,
    ) -> Result<(), pmetal_metal::error::MetalError> {
        self.trainer.load_weights_safetensors(path)
    }

    /// Install vocab compaction for faster classifier operations.
    ///
    /// Must be called after `load_weights_*` and before `compile_kernels()`.
    pub fn install_vocab_map(&mut self, vocab_map: pmetal_metal::ane::dynamic_trainer::VocabMap) {
        self.trainer.install_vocab_map(vocab_map);
    }

    /// Compile dynamic ANE kernels (one-time operation).
    pub fn compile_kernels(&mut self) -> Result<(), pmetal_metal::error::MetalError> {
        self.trainer.compile_kernels()
    }

    /// Run the full training loop over the provided data.
    ///
    /// `data` is a slice of batches, where each batch is a slice of
    /// `(input_tokens, target_tokens)` pairs for gradient accumulation.
    ///
    /// Kernels must be compiled before calling this method.
    pub fn train(
        &mut self,
        data: &[Vec<(Vec<u16>, Vec<u16>)>],
    ) -> Result<TrainingState, pmetal_metal::error::MetalError> {
        let start = Instant::now();

        for cb in &mut self.callbacks {
            cb.on_train_start();
        }

        let num_batches = data.len().min(self.config.num_batches);
        let seq_len = self.config.trainer.seq_len;
        let max_steps = self.config.max_steps;

        for batch_idx in 0..num_batches {
            for cb in &mut self.callbacks {
                cb.on_step_start(batch_idx);
            }

            // No budget check needed — dynamic pipeline never recompiles
            let loss = self.trainer.train_batch(&data[batch_idx], max_steps)?;

            // Update state
            self.state.step = batch_idx + 1;
            self.state.loss = loss as f64;
            self.state.learning_rate = self.config.trainer.learning_rate as f64;
            self.state.tokens_processed += data[batch_idx].len() * seq_len;
            self.state.elapsed_secs = start.elapsed().as_secs_f64();

            // Log
            if (batch_idx + 1) % self.config.log_every == 0 {
                let tok_sec = self.state.tokens_processed as f64 / self.state.elapsed_secs;
                tracing::info!(
                    batch = batch_idx + 1,
                    loss = format!("{:.4}", loss),
                    tok_sec = format!("{:.0}", tok_sec),
                    compiles = self.trainer.compile_count(),
                    "Training step"
                );
            }

            let timings = &self.trainer.last_timings;
            let tok_sec = self.state.tokens_processed as f64 / self.state.elapsed_secs.max(0.001);
            let metrics = StepMetrics {
                step: batch_idx,
                loss: loss as f64,
                lr: self.config.trainer.learning_rate as f64,
                tok_sec,
                ane_fwd_ms: timings.ane_fwd_us as f64 / 1000.0,
                ane_bwd_ms: timings.ane_bwd_us as f64 / 1000.0,
                rmsnorm_ms: timings.rmsnorm_us as f64 / 1000.0,
                cblas_ms: timings.cblas_dw_us as f64 / 1000.0,
                adam_ms: timings.adam_us as f64 / 1000.0,
                total_ms: timings.total_us as f64 / 1000.0,
                tokens: data[batch_idx].len() * seq_len,
                grad_norm: None,
            };
            for cb in &mut self.callbacks {
                cb.on_step_end_with_metrics(&metrics);
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

    /// Save a checkpoint.
    fn save_checkpoint(&mut self, path: &Path) {
        tracing::info!(path = %path.display(), step = self.state.step, "Saving ANE checkpoint");

        if let Err(e) = std::fs::create_dir_all(path) {
            tracing::error!(error = %e, "Failed to create checkpoint directory");
            return;
        }

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

        for cb in &mut self.callbacks {
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
            trainer: DynamicAneTrainerConfig {
                dim: 64,
                hidden_dim: 128,
                n_heads: 4,
                n_layers: 2,
                vocab_size: 100,
                seq_len: 16,
                ..Default::default()
            },
            num_batches: 10,
            max_steps: 100,
            log_every: 1,
            save_every: None,
            output_dir: PathBuf::from("/tmp/ane-test"),
        };

        let training_loop = AneTrainingLoop::new(config);
        assert_eq!(training_loop.state().step, 0);
        assert_eq!(training_loop.trainer().config().n_layers, 2);
        assert_eq!(training_loop.trainer().compile_count(), 0);
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
            trainer: DynamicAneTrainerConfig {
                dim: 64,
                hidden_dim: 128,
                n_heads: 4,
                n_layers: 2,
                vocab_size: 100,
                seq_len: 16,
                ..Default::default()
            },
            num_batches: 10,
            max_steps: 100,
            log_every: 1,
            save_every: None,
            output_dir: PathBuf::from("/tmp/ane-test"),
        };

        let mut training_loop = AneTrainingLoop::new(config);
        training_loop.add_callback(Box::new(TestCallback {
            steps: steps.clone(),
        }));

        assert_eq!(training_loop.state().step, 0);
    }
}
