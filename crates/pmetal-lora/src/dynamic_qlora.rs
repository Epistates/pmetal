//! Dynamic QLoRA model dispatch based on architecture.
//!
//! This module provides automatic architecture detection and model construction
//! for QLoRA training, eliminating the need for hardcoded model types in the
//! orchestrator.
//!
//! # Example
//!
//! ```ignore
//! use pmetal_lora::DynamicQloraModel;
//!
//! let mut model = DynamicQloraModel::from_model_dir("/path/to/model", qlora_config)?;
//! model.load_and_quantize_from_dir("/path/to/model")?;
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::module::{ModuleParamMut, ModuleParamRef, ModuleParameters};
use pmetal_models::ModelArchitecture;
use pmetal_models::architectures::{
    gemma::GemmaConfig, llama::LlamaConfig, mistral::MistralConfig, qwen3::Qwen3Config,
};

use crate::{
    LoraError, QLoraConfig, TrainableModel, gemma_qlora::GemmaQloraForCausalLM,
    llama_qlora::LlamaQloraForCausalLM, mistral_qlora::MistralQloraForCausalLM,
    qwen3_qlora::Qwen3QloraForCausalLM,
};

/// Dispatch a method call uniformly across all `DynamicQloraModel` variants.
macro_rules! dispatch_qlora {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            Self::Llama(m) => m.$method($($arg),*),
            Self::Mistral(m) => m.$method($($arg),*),
            Self::Qwen3(m) => m.$method($($arg),*),
            Self::Gemma(m) => m.$method($($arg),*),
        }
    };
}

/// Architecture-agnostic QLoRA model.
///
/// Wraps one of the four supported QLoRA architectures (Llama, Mistral, Qwen3,
/// Gemma) and exposes a unified interface including the QLoRA-specific methods
/// (`load_and_quantize_from_dir`, `memory_savings`, `memory_usage`) that are
/// not part of the `TrainableModel` trait.
pub enum DynamicQloraModel {
    Llama(LlamaQloraForCausalLM),
    Mistral(MistralQloraForCausalLM),
    Qwen3(Qwen3QloraForCausalLM),
    Gemma(GemmaQloraForCausalLM),
}

impl DynamicQloraModel {
    /// Detect the architecture from `config.json` in `model_dir` and construct
    /// the appropriate QLoRA model.
    ///
    /// Returns `Err` for architectures that do not have a QLoRA implementation
    /// (e.g., Phi, Qwen3Next / Qwen 3.5, DeepSeek, NemotronH …).
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        qlora_config: QLoraConfig,
    ) -> Result<Self, LoraError> {
        let model_dir = model_dir.as_ref();

        let arch = ModelArchitecture::detect(model_dir).map_err(|e| {
            LoraError::InvalidState(format!("Failed to detect model architecture: {}", e))
        })?;

        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| LoraError::InvalidState(format!("Failed to read config.json: {}", e)))?;

        match arch {
            ModelArchitecture::Llama | ModelArchitecture::Llama4 => {
                let cfg: LlamaConfig = serde_json::from_str(&config_content).map_err(|e| {
                    LoraError::InvalidState(format!("Failed to parse Llama config: {}", e))
                })?;
                let model = LlamaQloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Llama(model))
            }
            ModelArchitecture::Mistral | ModelArchitecture::Granite => {
                let cfg: MistralConfig = serde_json::from_str(&config_content).map_err(|e| {
                    LoraError::InvalidState(format!("Failed to parse Mistral config: {}", e))
                })?;
                let model = MistralQloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Mistral(model))
            }
            ModelArchitecture::Qwen3 | ModelArchitecture::Qwen2 => {
                let cfg: Qwen3Config = serde_json::from_str(&config_content).map_err(|e| {
                    LoraError::InvalidState(format!("Failed to parse Qwen3 config: {}", e))
                })?;
                let model = Qwen3QloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Qwen3(model))
            }
            ModelArchitecture::Gemma | ModelArchitecture::RecurrentGemma => {
                let cfg: GemmaConfig = serde_json::from_str(&config_content).map_err(|e| {
                    LoraError::InvalidState(format!("Failed to parse Gemma config: {}", e))
                })?;
                let model = GemmaQloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Gemma(model))
            }
            unsupported => Err(LoraError::InvalidState(format!(
                "QLoRA is not supported for {} models. \
                 QLoRA is available for: Llama, Mistral, Qwen3, Gemma. \
                 For other architectures use standard LoRA (`--lora`) instead.",
                unsupported
            ))),
        }
    }

    /// Load and quantize weights from the model directory.
    pub fn load_and_quantize_from_dir(
        &mut self,
        model_dir: impl AsRef<Path>,
    ) -> Result<(), LoraError> {
        let path = model_dir.as_ref();
        match self {
            Self::Llama(m) => m.load_and_quantize_from_dir(path),
            Self::Mistral(m) => m.load_and_quantize_from_dir(path),
            Self::Qwen3(m) => m.load_and_quantize_from_dir(path),
            Self::Gemma(m) => m.load_and_quantize_from_dir(path),
        }
    }

    /// Memory savings ratio vs. full-precision (ratio of QLoRA bytes to fp32 bytes).
    pub fn memory_savings(&self) -> f32 {
        dispatch_qlora!(self, memory_savings)
    }

    /// Memory usage in bytes: (quantized_bytes, lora_bytes, total_bytes).
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        dispatch_qlora!(self, memory_usage)
    }

    /// Architecture label for log messages.
    pub fn arch_name(&self) -> &'static str {
        match self {
            Self::Llama(_) => "Llama",
            Self::Mistral(_) => "Mistral",
            Self::Qwen3(_) => "Qwen3",
            Self::Gemma(_) => "Gemma",
        }
    }
}

// ---------------------------------------------------------------------------
// TrainableModel delegation
// ---------------------------------------------------------------------------

impl TrainableModel for DynamicQloraModel {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        // Use explicit trait dispatch to avoid ambiguity with inherent `forward` methods
        // that have different signatures (e.g., Qwen3 takes an extra position_ids arg).
        match self {
            Self::Llama(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Mistral(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Qwen3(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Gemma(m) => TrainableModel::forward(m, input_ids, mask),
        }
    }

    fn num_trainable_params(&self) -> usize {
        dispatch_qlora!(self, num_trainable_params)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        dispatch_qlora!(self, lora_parameters)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        dispatch_qlora!(self, set_lora_parameters, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        dispatch_qlora!(self, save_lora_weights, path.as_ref())
    }

    fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        dispatch_qlora!(self, load_lora_weights, path.as_ref())
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        dispatch_qlora!(self, enable_gradient_checkpointing, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        dispatch_qlora!(self, disable_gradient_checkpointing)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        dispatch_qlora!(self, supports_gradient_checkpointing)
    }
}

// ---------------------------------------------------------------------------
// ModuleParameters delegation (required by TrainingLoop::run)
// ---------------------------------------------------------------------------

impl ModuleParameters for DynamicQloraModel {
    fn num_parameters(&self) -> usize {
        dispatch_qlora!(self, num_parameters)
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        dispatch_qlora!(self, parameters)
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        dispatch_qlora!(self, parameters_mut)
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        dispatch_qlora!(self, trainable_parameters)
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        dispatch_qlora!(self, freeze_parameters, recursive)
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        dispatch_qlora!(self, unfreeze_parameters, recursive)
    }

    fn all_frozen(&self) -> Option<bool> {
        dispatch_qlora!(self, all_frozen)
    }

    fn any_frozen(&self) -> Option<bool> {
        dispatch_qlora!(self, any_frozen)
    }
}
