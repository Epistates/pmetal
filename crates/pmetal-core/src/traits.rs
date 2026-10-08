//! Core trait definitions.

use crate::{EvalMetrics, Result, StepMetrics};
use std::path::Path;

/// Dataset trait for training data.
pub trait Dataset: Send + Sync {
    /// The item type yielded by this dataset.
    type Item;

    /// Get the number of samples in the dataset.
    fn len(&self) -> usize;

    /// Check if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a sample by index.
    fn get(&self, index: usize) -> Option<Self::Item>;

    /// Get an iterator over the dataset.
    fn iter(&self) -> impl Iterator<Item = Self::Item>;
}

/// Quantizer trait for different quantization schemes.
pub trait Quantizer {
    /// The tensor type used by this quantizer.
    type Tensor;

    /// The quantized tensor type.
    type QuantizedTensor;

    /// Quantize a tensor.
    ///
    /// # Arguments
    /// * `tensor` - The tensor to quantize
    /// * `block_size` - Block size for blockwise quantization
    fn quantize(&self, tensor: &Self::Tensor, block_size: usize) -> Result<Self::QuantizedTensor>;

    /// Dequantize a tensor back to full precision.
    fn dequantize(&self, quantized: &Self::QuantizedTensor) -> Result<Self::Tensor>;
}

/// Optimizer trait.
pub trait Optimizer {
    /// The tensor type used by this optimizer.
    type Tensor;

    /// Update parameters using gradients.
    fn step(&mut self, params: &mut [Self::Tensor], grads: &[Self::Tensor]) -> Result<()>;

    /// Zero all gradients.
    fn zero_grad(&mut self);

    /// Get current learning rate.
    fn learning_rate(&self) -> f64;

    /// Set learning rate.
    fn set_learning_rate(&mut self, lr: f64);
}

/// Learning rate scheduler trait.
pub trait LrScheduler {
    /// Get learning rate for the given step.
    fn get_lr(&self, step: usize) -> f64;

    /// Update scheduler state after a step.
    fn step(&mut self);
}

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
/// Implement this trait on configuration structs to provide consistent
/// validation across all crates.
pub trait ConfigValidator {
    /// Validate the configuration.
    ///
    /// Returns `Ok(())` if valid, or an error describing the validation failure.
    fn validate(&self) -> Result<()>;
}

/// Trait for loading configurations from files.
///
/// Provides default implementations for YAML and JSON loading.
pub trait ConfigLoader: Sized + serde::de::DeserializeOwned {
    /// Load configuration from a YAML file.
    fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content)
    }

    /// Load configuration from a YAML string.
    fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).map_err(|e| crate::PMetalError::Config(e.to_string()))
    }

    /// Load configuration from a JSON file.
    fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    /// Load configuration from a JSON string.
    fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| crate::PMetalError::Config(e.to_string()))
    }
}

// ============================================================================
// Default Value Helpers
// ============================================================================

/// Default training hyperparameters used across crates.
///
/// These constants provide consistent defaults and reduce duplication.
///
/// **Two contexts**: (1) library-default values (used when a non-CLI consumer
/// constructs a config without supplying a value); (2) CLI-default values
/// (the literal that appears in `default_value="..."` clap attributes). Most
/// constants below match both. The exceptions are flagged in the doc strings.
pub mod defaults {
    /// Default learning rate for LoRA training.
    pub const LEARNING_RATE: f64 = 2e-4;
    /// Default learning rate for embeddings (lower than base).
    pub const EMBEDDING_LR: f64 = 5e-5;
    /// Default batch size for non-CLI library consumers (CLI defaults to 1).
    pub const BATCH_SIZE: usize = 4;
    /// Default batch size for CLI invocation. Matches `pmetal train --batch-size`.
    pub const CLI_BATCH_SIZE: usize = 1;
    /// Default number of epochs for non-CLI library consumers (CLI defaults to 1).
    pub const EPOCHS: usize = 3;
    /// Default number of epochs for CLI invocation.
    pub const CLI_EPOCHS: usize = 1;
    /// Default warmup steps used by the scheduler when nothing is supplied.
    pub const WARMUP_STEPS: usize = 100;
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
    /// Default logging frequency.
    pub const LOGGING_STEPS: usize = 10;
    /// Default maximum sequence length for non-CLI library consumers.
    pub const MAX_SEQ_LEN: usize = 2048;
    /// Sentinel used by the CLI/TUI/GUI/MCP surfaces to mean "auto-detect from
    /// the model config". Resolved by the trainer/inference engine at load time.
    pub const MAX_SEQ_LEN_AUTO: usize = 0;
    /// Default LoRA rank.
    pub const LORA_R: usize = 16;
    /// Default LoRA alpha.
    pub const LORA_ALPHA: f32 = 32.0;
    /// Default temperature for preference learning.
    pub const BETA: f64 = 0.1;
    /// Default label smoothing.
    pub const LABEL_SMOOTHING: f64 = 0.0;
    /// Default RMS norm epsilon.
    pub const RMS_NORM_EPS: f32 = 1e-5;
    /// Default RoPE theta.
    pub const ROPE_THETA: f32 = 10000.0;

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
