//! Core trait definitions.

use crate::{EvalMetrics, Result, StepMetrics};
use std::path::Path;

/// Callback trait for training events.
pub trait TrainingCallback: Send + Sync {
    /// Called at the start of training.
    fn on_train_start(&mut self) {}

    /// Called at the end of training.
    fn on_train_end(&mut self) {}

    /// Called at the start of each epoch.
    fn on_epoch_start(&mut self, _epoch: usize) {}

    /// Called at the end of each epoch.
    fn on_epoch_end(&mut self, _epoch: usize, _metrics: &EvalMetrics) {}

    /// Called at the start of each step.
    fn on_step_start(&mut self, _step: usize) {}

    /// Called at the end of each step.
    fn on_step_end(&mut self, _step: usize, _loss: f64) {}

    /// Called at the end of each step with rich metrics (timing, throughput, lr).
    ///
    /// Default implementation delegates to [`on_step_end`](TrainingCallback::on_step_end).
    /// Override this method in callbacks that need timing breakdown or throughput data
    /// (e.g., dashboard, detailed JSONL logging).
    fn on_step_end_with_metrics(&mut self, metrics: &StepMetrics) {
        self.on_step_end(metrics.step, metrics.loss);
    }

    /// Called when a checkpoint is saved.
    fn on_save(&mut self, _path: &Path) {}

    /// Called when the adaptive LR controller triggers an event.
    fn on_lr_event(&mut self, _event: &str) {}

    /// Return `true` to request a clean early stop of the training loop.
    ///
    /// Checked after each step. When any callback returns `true`, the loop
    /// finishes the current step, saves the best weights, and returns cleanly.
    fn should_stop(&self) -> bool {
        false
    }
}

// ============================================================================
// Configuration Traits
// ============================================================================

/// Trait for configuration validation.
///
/// Implemented for every configuration type in this crate (see
/// `validation.rs`): a config that cannot run, or would run as something other
/// than what it says, is an error naming every offending key, not a value
/// clamped somewhere downstream.
pub trait ConfigValidator {
    /// Validate the configuration.
    ///
    /// Returns `Ok(())` if valid, or a [`PMetalError::Config`](crate::PMetalError::Config)
    /// listing every problem found.
    fn validate(&self) -> Result<()>;
}

// ============================================================================
// Default Value Helpers
// ============================================================================

/// Default training hyperparameters, one definition for every consumer.
///
/// `TrainingConfig::default()`, `LoraConfig::default()`, the job specs and the
/// CLI's `default_value` attributes all read these, so a default is changed in
/// one place. The `CLI_` constants are the command line's choice where it
/// differs from the library default of the same quantity.
pub mod defaults {
    /// Default learning rate for LoRA training.
    pub const LEARNING_RATE: f64 = 2e-4;
    /// Default batch size. Matches `pmetal train --batch-size`.
    pub const CLI_BATCH_SIZE: usize = 1;
    /// Default number of epochs: for the command line, a config file that does
    /// not say, and the library alike.
    pub const EPOCHS: usize = 1;
    /// Default weight decay.
    pub const WEIGHT_DECAY: f64 = 0.01;
    /// Default gradient clipping norm.
    pub const MAX_GRAD_NORM: f64 = 1.0;
    /// Default gradient accumulation steps.
    pub const GRADIENT_ACCUMULATION_STEPS: usize = 4;
    /// Default loss scale (1.0 = no scaling).
    pub const LOSS_SCALE: f64 = 1.0;
    /// Default random seed.
    pub const SEED: u64 = 42;
    /// Default interval, in optimizer steps, between logged losses.
    pub const LOGGING_STEPS: usize = 10;
    /// Default LoRA rank.
    pub const LORA_R: usize = 16;
    /// Default LoRA alpha.
    pub const LORA_ALPHA: f32 = 32.0;

    // -----------------------------------------------------------------------
    // Output-directory defaults (used by job specs to derive `output_dir`)
    // -----------------------------------------------------------------------

    /// Default output directory for `pmetal train`.
    pub const TRAIN_OUTPUT_DIR: &str = "./output";
    /// Default output directory for `pmetal distill`.
    pub const DISTILL_OUTPUT_DIR: &str = "./output/distilled";
    /// Default output directory for `pmetal grpo`.
    pub const GRPO_OUTPUT_DIR: &str = "./output/grpo";
    /// Default output directory for `pmetal rlkd`.
    pub const RLKD_OUTPUT_DIR: &str = "./output/rlkd";
    /// Default output directory for `pmetal preference`.
    pub const PREFERENCE_OUTPUT_DIR: &str = "./output/preference";
    /// Default output directory for `pmetal embed-train`.
    pub const EMBED_OUTPUT_DIR: &str = "./output-embed";
    /// Default output directory for `pmetal pretrain`.
    pub const PRETRAIN_OUTPUT_DIR: &str = "./pretrain-output";
    /// Default output directory for `pmetal merge`.
    pub const MERGE_OUTPUT_DIR: &str = "./merged";
    /// Default output directory for `pmetal fuse`.
    pub const FUSE_OUTPUT_DIR: &str = "./fused";
    /// Default output directory for `pmetal quantize` (GGUF/MLX).
    pub const QUANTIZE_OUTPUT_DIR: &str = "./quantized";
}
