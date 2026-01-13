//! Dynamic LoRA model dispatch based on architecture.
//!
//! This module provides automatic architecture detection and model loading
//! for LoRA training, eliminating the need for hardcoded model types.
//!
//! # Example
//!
//! ```ignore
//! use pmetal_lora::DynamicLoraModel;
//! use pmetal_core::LoraConfig;
//!
//! // Automatically detect architecture and create LoRA model
//! let mut model = DynamicLoraModel::from_pretrained(
//!     "/path/to/model",
//!     LoraConfig::default(),
//! ).await?;
//!
//! // Training works regardless of architecture
//! let logits = model.forward(&input_ids, None)?;
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_core::LoraConfig;
use pmetal_mlx::kv_cache::KVCache;
use pmetal_models::{
    DispatchError, GgufModelConfig, ModelArchitecture, WeightFormat, WeightFormatError,
    WeightLoader,
};
use mlx_rs::Array;

use crate::{
    gemma_lora::GemmaLoraForCausalLM,
    llama_lora::LlamaLoraForCausalLM,
    mistral_lora::MistralLoraForCausalLM,
    phi_lora::PhiLoraForCausalLM,
    qwen3_lora::Qwen3LoraForCausalLM,
    LoraError, TrainableModel,
};

/// Dynamic LoRA model container using enum dispatch.
///
/// This approach uses static dispatch via enum variants rather than
/// trait objects, which is more efficient while still providing
/// runtime polymorphism based on detected architecture.
///
/// # Supported Architectures
///
/// - Llama (2, 3, 3.1, 3.2, 3.3, 4)
/// - Mistral (7B) - with sliding window attention
/// - Qwen3 (3, 3.5) - with gradient checkpointing support
/// - Gemma (2B, 7B) / Gemma2 (2B, 9B, 27B) - with GeGLU and special RMSNorm
/// - Phi (3, 3.5, 4) - with partial RoPE and fused gate_up
///
/// # Architecture-Specific Features
///
/// | Feature | Llama | Mistral | Qwen3 | Gemma | Phi |
/// |---------|-------|---------|-------|-------|-----|
/// | LoRA Training | Yes | Yes | Yes | Yes | Yes |
/// | QLoRA | Yes | Planned | Yes | Planned | Planned |
/// | Gradient Checkpointing | Yes | Yes | Yes | Yes | Yes |
/// | Packed Sequences | Yes | Yes | Yes | Yes | Yes |
pub enum DynamicLoraModel {
    /// Llama family with LoRA adapters (supports gradient checkpointing).
    Llama(LlamaLoraForCausalLM),
    /// Mistral family with LoRA adapters (supports gradient checkpointing and SWA).
    Mistral(MistralLoraForCausalLM),
    /// Qwen3 family with LoRA adapters (supports gradient checkpointing).
    Qwen3(Qwen3LoraForCausalLM),
    /// Gemma family with LoRA adapters (supports GeGLU and special RMSNorm).
    Gemma(GemmaLoraForCausalLM),
    /// Phi family with LoRA adapters (supports partial RoPE and fused gate_up).
    Phi(PhiLoraForCausalLM),
}

impl std::fmt::Debug for DynamicLoraModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama(_) => write!(f, "DynamicLoraModel::Llama"),
            Self::Mistral(_) => write!(f, "DynamicLoraModel::Mistral"),
            Self::Qwen3(_) => write!(f, "DynamicLoraModel::Qwen3"),
            Self::Gemma(_) => write!(f, "DynamicLoraModel::Gemma"),
            Self::Phi(_) => write!(f, "DynamicLoraModel::Phi"),
        }
    }
}

impl DynamicLoraModel {
    /// Create a LoRA model from a pretrained model directory.
    ///
    /// This function:
    /// 1. Reads config.json to detect the model architecture
    /// 2. Instantiates the correct LoRA model type
    /// 3. Loads base model weights from safetensors files
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to model directory containing config.json and weights
    /// * `lora_config` - LoRA configuration (rank, alpha, target modules)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = LoraConfig::default();
    /// let model = DynamicLoraModel::from_pretrained("/path/to/llama-3.2-1b", config)?;
    /// let model = DynamicLoraModel::from_pretrained("/path/to/qwen3-0.6b", config)?;
    /// ```
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        lora_config: LoraConfig,
    ) -> Result<Self, DynamicLoraError> {
        let model_dir = model_dir.as_ref();

        // Detect architecture
        let arch = ModelArchitecture::detect(model_dir)?;

        tracing::info!("Detected architecture: {}", arch);

        // Read config content
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)?;

        // Create and load the appropriate model
        match arch {
            ModelArchitecture::Llama => {
                let llama_config: pmetal_models::architectures::llama::LlamaConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = LlamaLoraForCausalLM::new(llama_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                Ok(DynamicLoraModel::Llama(model))
            }
            ModelArchitecture::Mistral => {
                let mistral_config: pmetal_models::architectures::mistral::MistralConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = MistralLoraForCausalLM::new(mistral_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded Mistral LoRA model");
                Ok(DynamicLoraModel::Mistral(model))
            }
            ModelArchitecture::Qwen3 => {
                let qwen_config: pmetal_models::architectures::qwen3::Qwen3Config =
                    serde_json::from_str(&config_content)?;

                let mut model = Qwen3LoraForCausalLM::new(qwen_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                Ok(DynamicLoraModel::Qwen3(model))
            }
            // Qwen2 uses Qwen3 implementation for now (similar architecture)
            ModelArchitecture::Qwen2 => {
                // Qwen2 and Qwen3 share the same base structure
                // We can treat Qwen2 as Qwen3 for LoRA training
                let qwen_config: pmetal_models::architectures::qwen3::Qwen3Config =
                    serde_json::from_str(&config_content)?;

                let mut model = Qwen3LoraForCausalLM::new(qwen_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loading Qwen2 model with Qwen3 LoRA implementation");
                Ok(DynamicLoraModel::Qwen3(model))
            }
            ModelArchitecture::Gemma => {
                let gemma_config: pmetal_models::architectures::gemma::GemmaConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = GemmaLoraForCausalLM::new(gemma_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded Gemma LoRA model");
                Ok(DynamicLoraModel::Gemma(model))
            }
            ModelArchitecture::Phi => {
                let phi_config: pmetal_models::architectures::phi::PhiConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = PhiLoraForCausalLM::new(phi_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded Phi LoRA model");
                Ok(DynamicLoraModel::Phi(model))
            }
            // Other architectures not yet supported for LoRA training
            arch => Err(DynamicLoraError::NotImplemented(arch)),
        }
    }

    /// Create a LoRA model from a GGUF file.
    ///
    /// This function:
    /// 1. Reads GGUF metadata to detect the model architecture
    /// 2. Extracts model configuration from GGUF metadata
    /// 3. Dequantizes weights to f32 for training
    /// 4. Loads weights into the appropriate LoRA model
    ///
    /// # Arguments
    ///
    /// * `gguf_path` - Path to .gguf file or directory containing .gguf
    /// * `lora_config` - LoRA configuration (rank, alpha, target modules)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = LoraConfig::default();
    /// let model = DynamicLoraModel::from_gguf("./model.gguf", config)?;
    /// ```
    pub fn from_gguf(
        gguf_path: impl AsRef<Path>,
        lora_config: LoraConfig,
    ) -> Result<Self, DynamicLoraError> {
        let gguf_path = gguf_path.as_ref();

        // Load weights from GGUF (auto-dequantizes and maps tensor names)
        tracing::info!("Loading weights from GGUF: {:?}", gguf_path);
        let weights = WeightLoader::load_gguf(gguf_path)?;

        // Find the actual GGUF file for metadata extraction
        let gguf_file = if gguf_path.is_file() {
            gguf_path.to_path_buf()
        } else {
            // Find .gguf file in directory
            std::fs::read_dir(gguf_path)?
                .filter_map(|e| e.ok())
                .find(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext.to_string_lossy().to_lowercase() == "gguf")
                        .unwrap_or(false)
                })
                .map(|e| e.path())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("No .gguf file found in {:?}", gguf_path),
                    )
                })?
        };

        // Read GGUF content for config extraction
        let content = pmetal_gguf::GgufContent::from_file(&gguf_file)
            .map_err(|e| WeightFormatError::Gguf(e.to_string()))?;

        // Extract model config from GGUF metadata
        let gguf_config = GgufModelConfig::from_gguf(&content)?;

        tracing::info!(
            "GGUF architecture: {}, hidden_size: {}, layers: {}",
            gguf_config.architecture,
            gguf_config.hidden_size,
            gguf_config.num_hidden_layers
        );

        // Map GGUF architecture name to ModelArchitecture
        let arch = match gguf_config.architecture.to_lowercase().as_str() {
            "llama" => ModelArchitecture::Llama,
            "qwen2" => ModelArchitecture::Qwen2,
            "qwen3" => ModelArchitecture::Qwen3,
            "mistral" => ModelArchitecture::Mistral,
            "gemma" => ModelArchitecture::Gemma,
            "phi" => ModelArchitecture::Phi,
            other => {
                return Err(DynamicLoraError::Dispatch(DispatchError::UnsupportedArchitecture(
                    other.to_string(),
                )));
            }
        };

        // Create model with extracted config and load weights
        Self::from_weights_with_arch(weights, arch, gguf_config, lora_config)
    }

    /// Create a LoRA model from pre-loaded weights with known architecture.
    ///
    /// This is the internal implementation used by both `from_pretrained` and `from_gguf`.
    fn from_weights_with_arch(
        weights: HashMap<String, Array>,
        arch: ModelArchitecture,
        gguf_config: GgufModelConfig,
        lora_config: LoraConfig,
    ) -> Result<Self, DynamicLoraError> {
        match arch {
            ModelArchitecture::Llama => {
                let config = gguf_config.to_llama_config();
                let mut model = LlamaLoraForCausalLM::new(config, lora_config)?;
                model.load_base_weights(&weights)?;
                model.eval_all()?;
                Ok(DynamicLoraModel::Llama(model))
            }
            ModelArchitecture::Mistral => {
                let config = gguf_config.to_mistral_config();
                let mut model = MistralLoraForCausalLM::new(config, lora_config)?;
                model.load_base_weights(&weights)?;
                model.eval_all()?;
                Ok(DynamicLoraModel::Mistral(model))
            }
            ModelArchitecture::Qwen3 | ModelArchitecture::Qwen2 => {
                let config = gguf_config.to_qwen3_config();
                let mut model = Qwen3LoraForCausalLM::new(config, lora_config)?;
                model.load_base_weights(&weights)?;
                model.eval_all()?;
                Ok(DynamicLoraModel::Qwen3(model))
            }
            ModelArchitecture::Gemma => {
                let config = gguf_config.to_gemma_config();
                let mut model = GemmaLoraForCausalLM::new(config, lora_config)?;
                model.load_base_weights(&weights)?;
                model.eval_all()?;
                Ok(DynamicLoraModel::Gemma(model))
            }
            ModelArchitecture::Phi => {
                let config = gguf_config.to_phi_config();
                let mut model = PhiLoraForCausalLM::new(config, lora_config)?;
                model.load_base_weights(&weights)?;
                model.eval_all()?;
                Ok(DynamicLoraModel::Phi(model))
            }
            arch => Err(DynamicLoraError::NotImplemented(arch)),
        }
    }

    /// Get the detected architecture.
    pub fn architecture(&self) -> ModelArchitecture {
        match self {
            Self::Llama(_) => ModelArchitecture::Llama,
            Self::Mistral(_) => ModelArchitecture::Mistral,
            Self::Qwen3(_) => ModelArchitecture::Qwen3,
            Self::Gemma(_) => ModelArchitecture::Gemma,
            Self::Phi(_) => ModelArchitecture::Phi,
        }
    }

    /// Get the architecture name as a string.
    pub fn architecture_name(&self) -> &'static str {
        match self {
            Self::Llama(_) => "Llama",
            Self::Mistral(_) => "Mistral",
            Self::Qwen3(_) => "Qwen3",
            Self::Gemma(_) => "Gemma",
            Self::Phi(_) => "Phi",
        }
    }
}

// Implement TrainableModel for DynamicLoraModel via dispatch
impl mlx_rs::module::ModuleParameters for DynamicLoraModel {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Llama(m) => m.num_trainable_params(),
            Self::Mistral(m) => m.num_trainable_params(),
            Self::Qwen3(m) => m.num_trainable_params(),
            Self::Gemma(m) => m.num_trainable_params(),
            Self::Phi(m) => m.num_trainable_params(),
        }
    }

    fn parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            Self::Llama(m) => m.parameters(),
            Self::Mistral(m) => m.parameters(),
            Self::Qwen3(m) => m.parameters(),
            Self::Gemma(m) => m.parameters(),
            Self::Phi(m) => m.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> mlx_rs::module::ModuleParamMut<'_> {
        match self {
            Self::Llama(m) => m.parameters_mut(),
            Self::Mistral(m) => m.parameters_mut(),
            Self::Qwen3(m) => m.parameters_mut(),
            Self::Gemma(m) => m.parameters_mut(),
            Self::Phi(m) => m.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            Self::Llama(m) => m.trainable_parameters(),
            Self::Mistral(m) => m.trainable_parameters(),
            Self::Qwen3(m) => m.trainable_parameters(),
            Self::Gemma(m) => m.trainable_parameters(),
            Self::Phi(m) => m.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Llama(m) => m.freeze_parameters(recursive),
            Self::Mistral(m) => m.freeze_parameters(recursive),
            Self::Qwen3(m) => m.freeze_parameters(recursive),
            Self::Gemma(m) => m.freeze_parameters(recursive),
            Self::Phi(m) => m.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Llama(m) => m.unfreeze_parameters(recursive),
            Self::Mistral(m) => m.unfreeze_parameters(recursive),
            Self::Qwen3(m) => m.unfreeze_parameters(recursive),
            Self::Gemma(m) => m.unfreeze_parameters(recursive),
            Self::Phi(m) => m.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Llama(m) => m.all_frozen(),
            Self::Mistral(m) => m.all_frozen(),
            Self::Qwen3(m) => m.all_frozen(),
            Self::Gemma(m) => m.all_frozen(),
            Self::Phi(m) => m.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Llama(m) => m.any_frozen(),
            Self::Mistral(m) => m.any_frozen(),
            Self::Qwen3(m) => m.any_frozen(),
            Self::Gemma(m) => m.any_frozen(),
            Self::Phi(m) => m.any_frozen(),
        }
    }
}

impl TrainableModel for DynamicLoraModel {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        match self {
            Self::Llama(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Mistral(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Qwen3(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Gemma(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Phi(m) => TrainableModel::forward(m, input_ids, mask),
        }
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        match self {
            Self::Llama(m) => m.forward_with_positions(input_ids, mask, position_ids),
            Self::Mistral(m) => m.forward_with_positions(input_ids, mask, position_ids),
            Self::Qwen3(m) => m.forward_with_positions(input_ids, mask, position_ids),
            // Gemma and Phi don't have forward_with_positions yet, fallback to standard forward
            Self::Gemma(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Phi(m) => TrainableModel::forward(m, input_ids, mask),
        }
    }

    fn num_trainable_params(&self) -> usize {
        match self {
            Self::Llama(m) => m.num_trainable_params(),
            Self::Mistral(m) => m.num_trainable_params(),
            Self::Qwen3(m) => m.num_trainable_params(),
            Self::Gemma(m) => m.num_trainable_params(),
            Self::Phi(m) => m.num_trainable_params(),
        }
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        match self {
            Self::Llama(m) => m.lora_parameters(),
            Self::Mistral(m) => m.lora_parameters(),
            Self::Qwen3(m) => m.lora_parameters(),
            Self::Gemma(m) => m.lora_parameters(),
            Self::Phi(m) => m.lora_parameters(),
        }
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        match self {
            Self::Llama(m) => m.set_lora_parameters(params),
            Self::Mistral(m) => m.set_lora_parameters(params),
            Self::Qwen3(m) => m.set_lora_parameters(params),
            Self::Gemma(m) => m.set_lora_parameters(params),
            Self::Phi(m) => m.set_lora_parameters(params),
        }
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        match self {
            Self::Llama(m) => m.save_lora_weights(path),
            Self::Mistral(m) => m.save_lora_weights(path),
            Self::Qwen3(m) => m.save_lora_weights(path),
            Self::Gemma(m) => m.save_lora_weights(path),
            Self::Phi(m) => m.save_lora_weights(path),
        }
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        match self {
            Self::Llama(m) => m.load_lora_weights(path),
            Self::Mistral(m) => m.load_lora_weights(path),
            Self::Qwen3(m) => m.load_lora_weights(path),
            Self::Gemma(m) => m.load_lora_weights(path),
            Self::Phi(m) => m.load_lora_weights(path),
        }
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        match self {
            Self::Llama(m) => m.enable_gradient_checkpointing(layers_per_block),
            Self::Mistral(m) => m.enable_gradient_checkpointing(layers_per_block),
            Self::Qwen3(m) => m.enable_gradient_checkpointing(layers_per_block),
            Self::Gemma(m) => m.enable_gradient_checkpointing(layers_per_block),
            Self::Phi(m) => m.enable_gradient_checkpointing(layers_per_block),
        }
    }

    fn disable_gradient_checkpointing(&mut self) {
        match self {
            Self::Llama(m) => m.disable_gradient_checkpointing(),
            Self::Mistral(m) => m.disable_gradient_checkpointing(),
            Self::Qwen3(m) => m.disable_gradient_checkpointing(),
            Self::Gemma(m) => m.disable_gradient_checkpointing(),
            Self::Phi(m) => m.disable_gradient_checkpointing(),
        }
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        match self {
            Self::Llama(_) => true,   // Implemented
            Self::Mistral(_) => true, // Implemented
            Self::Qwen3(_) => true,   // Implemented
            Self::Gemma(_) => true,   // Implemented
            Self::Phi(_) => true,     // Implemented
        }
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        match self {
            Self::Llama(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Mistral(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Qwen3(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Gemma(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Phi(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
        }
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        match self {
            Self::Llama(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Mistral(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Qwen3(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Gemma(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Phi(m) => TrainableModel::create_cache(m, max_seq_len),
        }
    }

    fn supports_kv_cache(&self) -> bool {
        match self {
            Self::Llama(_) => true,
            Self::Mistral(_) => true,
            Self::Qwen3(_) => true,
            Self::Gemma(_) => true,
            Self::Phi(_) => true,
        }
    }
}

/// Errors during dynamic LoRA model loading.
#[derive(Debug, thiserror::Error)]
pub enum DynamicLoraError {
    /// Architecture dispatch error.
    #[error("Dispatch error: {0}")]
    Dispatch(#[from] DispatchError),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON parsing error.
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
    /// LoRA error.
    #[error("LoRA error: {0}")]
    Lora(#[from] LoraError),
    /// Weight format error.
    #[error("Weight format error: {0}")]
    WeightFormat(#[from] WeightFormatError),
    /// Architecture not implemented for LoRA training.
    #[error("Architecture not implemented for LoRA training: {0}")]
    NotImplemented(ModelArchitecture),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_dispatch() {
        // Test that the dispatch logic is correct
        assert_eq!(ModelArchitecture::from_model_type("llama"), Some(ModelArchitecture::Llama));
        assert_eq!(ModelArchitecture::from_model_type("mistral"), Some(ModelArchitecture::Mistral));
        assert_eq!(ModelArchitecture::from_model_type("qwen3"), Some(ModelArchitecture::Qwen3));
        assert_eq!(ModelArchitecture::from_model_type("qwen2"), Some(ModelArchitecture::Qwen2));
    }
}
