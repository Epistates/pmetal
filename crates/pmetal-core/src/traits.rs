//! Core trait definitions.
//!
//! Note: Some traits in this module are deprecated. See individual trait documentation
//! for current recommended patterns.

use crate::{EvalMetrics, LoraConfig, MemoryStats, ModelConfig, Result, TrainingConfig};
use std::path::Path;

/// Core model trait for all LLM architectures.
///
/// # Deprecated
///
/// This trait is deprecated and not implemented by any models.
/// Use the following traits from `pmetal_models` instead:
/// - [`CausalLMModel`] - For forward pass and generation
/// - [`LoraCapable`] - For LoRA adapter management
/// - [`Quantizable`] - For model quantization
///
/// For trainable models, use [`TrainableModel`] from `pmetal_lora`.
///
/// [`CausalLMModel`]: pmetal_models::CausalLMModel
/// [`LoraCapable`]: pmetal_models::LoraCapable
/// [`Quantizable`]: pmetal_models::Quantizable
/// [`TrainableModel`]: pmetal_lora::TrainableModel
#[deprecated(
    since = "0.2.0",
    note = "Use CausalLMModel from pmetal_models instead. This trait will be removed in a future version."
)]
pub trait PMetalModel: Send + Sync {
    /// The tensor type used by this model.
    type Tensor;

    /// Run forward pass.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs tensor of shape `[batch_size, seq_len]`
    /// * `attention_mask` - Optional attention mask tensor
    ///
    /// # Returns
    /// Model output containing logits and optional hidden states.
    fn forward(
        &self,
        input_ids: &Self::Tensor,
        attention_mask: Option<&Self::Tensor>,
    ) -> Result<crate::ModelOutput<Self::Tensor>>;

    /// Get model configuration.
    fn config(&self) -> &ModelConfig;

    /// Get all trainable parameters with their names.
    fn trainable_parameters(&self) -> Vec<(&str, Self::Tensor)>;

    /// Get total number of parameters.
    fn num_parameters(&self) -> usize;

    /// Get number of trainable parameters.
    fn num_trainable_parameters(&self) -> usize;

    /// Apply LoRA adapters to the model.
    ///
    /// # Arguments
    /// * `lora_config` - Configuration for LoRA adaptation
    fn apply_lora(&mut self, lora_config: &LoraConfig) -> Result<()>;

    /// Merge LoRA weights into base model.
    fn merge_lora(&mut self) -> Result<()>;

    /// Get memory footprint statistics.
    fn memory_footprint(&self) -> MemoryStats;

    /// Save model to disk.
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()>;

    /// Load model from disk.
    fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<()>;
}

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

/// Trainer trait for different training methods.
///
/// # Deprecated
///
/// This trait is deprecated and not implemented by any trainers.
/// Use the concrete trainer implementations from `pmetal_trainer` instead:
/// - [`SftTrainer`] - For supervised fine-tuning
/// - [`LoraTrainer`] - For LoRA fine-tuning
/// - [`DpoTrainer`] - For Direct Preference Optimization
/// - [`GrpoTrainer`] - For Group Relative Policy Optimization
/// - [`TrainingLoop`] - For the main training loop abstraction
///
/// [`SftTrainer`]: pmetal_trainer::SftTrainer
/// [`LoraTrainer`]: pmetal_trainer::LoraTrainer
/// [`DpoTrainer`]: pmetal_trainer::DpoTrainer
/// [`GrpoTrainer`]: pmetal_trainer::GrpoTrainer
/// [`TrainingLoop`]: pmetal_trainer::TrainingLoop
#[deprecated(
    since = "0.2.0",
    note = "Use concrete trainers from pmetal_trainer instead. This trait will be removed in a future version."
)]
#[allow(deprecated)]
pub trait Trainer {
    /// The model type this trainer works with.
    type Model: PMetalModel;

    /// Training output/result type.
    type Output;

    /// Create a new trainer with the given model and configuration.
    fn new(model: Self::Model, config: TrainingConfig) -> Result<Self>
    where
        Self: Sized;

    /// Run training on the given dataset.
    fn train<D: Dataset>(&mut self, dataset: &D) -> Result<Self::Output>;

    /// Evaluate the model on a dataset.
    fn evaluate<D: Dataset>(&self, dataset: &D) -> Result<EvalMetrics>;

    /// Save a training checkpoint.
    fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()>;

    /// Load a training checkpoint.
    fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<()>;

    /// Get the current training step.
    fn current_step(&self) -> usize;

    /// Get the current loss value.
    fn current_loss(&self) -> Option<f64>;
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

    /// Called when a checkpoint is saved.
    fn on_save(&mut self, _path: &Path) {}
}
