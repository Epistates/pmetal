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

use pmetal_bridge::compat::Array;
use pmetal_bridge::compat::Exception;
use pmetal_bridge::compat::{ModuleParamMut, ModuleParamRef, ModuleParameters};
use pmetal_models::ModelArchitecture;
use pmetal_models::architectures::{
    gemma::GemmaConfig, llama::LlamaConfig, mistral::MistralConfig, qwen3::Qwen3Config,
};

use crate::{
    LoraError, QLoraConfig, TrainableModel, deepseek_qlora::DeepSeekQloraForCausalLM,
    gemma_qlora::GemmaQloraForCausalLM, gemma4_qlora::Gemma4QloraForCausalLM,
    gpt_oss_qlora::GptOssQloraForCausalLM, granite_qlora::GraniteQloraForCausalLM,
    llama_qlora::LlamaQloraForCausalLM, llama4_qlora::Llama4QloraForCausalLM,
    mistral_qlora::MistralQloraForCausalLM, nemotron_h_qlora::NemotronHQloraForCausalLM,
    qwen3_moe_qlora::Qwen3MoEQLoraForCausalLM, qwen3_next_qlora::Qwen3NextQloraForCausalLM,
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
            Self::Qwen3Next(m) => m.$method($($arg),*),
            Self::Qwen3MoE(m) => m.$method($($arg),*),
            Self::Gemma4(m) => m.$method($($arg),*),
            Self::GptOss(m) => m.$method($($arg),*),
            Self::Granite(m) => m.$method($($arg),*),
            Self::Llama4(m) => m.$method($($arg),*),
            Self::DeepSeek(m) => m.$method($($arg),*),
            Self::NemotronH(m) => m.$method($($arg),*),
        }
    };
}

/// Architecture-agnostic QLoRA model.
///
/// Wraps one of the supported QLoRA architectures and exposes a unified
/// interface including the QLoRA-specific methods (`load_and_quantize_from_dir`,
/// `memory_savings`, `memory_usage`) that are not part of the `TrainableModel`
/// trait.
// See `DynamicLoraModel` for the rationale behind allowing the size gap.
#[allow(clippy::large_enum_variant)]
pub enum DynamicQloraModel {
    Llama(LlamaQloraForCausalLM),
    Mistral(MistralQloraForCausalLM),
    Qwen3(Qwen3QloraForCausalLM),
    Gemma(GemmaQloraForCausalLM),
    Qwen3Next(Qwen3NextQloraForCausalLM),
    Qwen3MoE(Qwen3MoEQLoraForCausalLM),
    Gemma4(Gemma4QloraForCausalLM),
    GptOss(GptOssQloraForCausalLM),
    Granite(GraniteQloraForCausalLM),
    Llama4(Llama4QloraForCausalLM),
    DeepSeek(DeepSeekQloraForCausalLM),
    NemotronH(NemotronHQloraForCausalLM),
}

impl DynamicQloraModel {
    /// Detect the architecture from `config.json` in `model_dir` and construct
    /// the appropriate QLoRA model.
    ///
    /// Returns `Err` for architectures that do not have a QLoRA implementation
    /// (e.g., Llama4, Phi, DeepSeek, NemotronH …).
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
            ModelArchitecture::Llama => {
                let cfg: LlamaConfig = serde_json::from_str(&config_content).map_err(|e| {
                    LoraError::InvalidState(format!("Failed to parse Llama config: {}", e))
                })?;
                let model = LlamaQloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Llama(model))
            }
            ModelArchitecture::Mistral => {
                let cfg: MistralConfig = serde_json::from_str(&config_content).map_err(|e| {
                    LoraError::InvalidState(format!("Failed to parse Mistral config: {}", e))
                })?;
                let model = MistralQloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Mistral(model))
            }
            ModelArchitecture::Granite => {
                let cfg: pmetal_models::architectures::granite::GraniteConfig =
                    serde_json::from_str(&config_content).map_err(|e| {
                        LoraError::InvalidState(format!("Failed to parse Granite config: {}", e))
                    })?;
                let model = GraniteQloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Granite(model))
            }
            ModelArchitecture::Qwen3 | ModelArchitecture::Qwen2 => {
                let cfg: Qwen3Config = serde_json::from_str(&config_content).map_err(|e| {
                    LoraError::InvalidState(format!("Failed to parse Qwen3 config: {}", e))
                })?;
                let model = Qwen3QloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Qwen3(model))
            }
            ModelArchitecture::Gemma => {
                let cfg: GemmaConfig = serde_json::from_str(&config_content).map_err(|e| {
                    LoraError::InvalidState(format!("Failed to parse Gemma config: {}", e))
                })?;
                let model = GemmaQloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Gemma(model))
            }
            ModelArchitecture::Qwen3Next => {
                let config_json: serde_json::Value = serde_json::from_str(&config_content)
                    .map_err(|e| {
                        LoraError::InvalidState(format!("Failed to parse Qwen3Next JSON: {}", e))
                    })?;
                let text_config_str = if config_json.get("text_config").is_some()
                    && config_json.get("hidden_size").is_none()
                {
                    serde_json::to_string(&config_json["text_config"]).map_err(|e| {
                        LoraError::InvalidState(format!(
                            "Failed to serialize Qwen3Next text config: {}",
                            e
                        ))
                    })?
                } else {
                    config_content.clone()
                };
                let mut cfg: pmetal_models::architectures::qwen3_next::Qwen3NextConfig =
                    serde_json::from_str(&text_config_str).map_err(|e| {
                        LoraError::InvalidState(format!("Failed to parse Qwen3Next config: {}", e))
                    })?;
                cfg.apply_rope_parameters();
                let model = Qwen3NextQloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Qwen3Next(model))
            }
            ModelArchitecture::Qwen3MoE => {
                let cfg: pmetal_models::architectures::qwen3_moe::Qwen3MoEConfig =
                    serde_json::from_str(&config_content).map_err(|e| {
                        LoraError::InvalidState(format!("Failed to parse Qwen3MoE config: {}", e))
                    })?;
                let model = Qwen3MoEQLoraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Qwen3MoE(model))
            }
            ModelArchitecture::Gemma4 => {
                let config_json: serde_json::Value = serde_json::from_str(&config_content)
                    .map_err(|e| {
                        LoraError::InvalidState(format!("Failed to parse Gemma4 JSON: {}", e))
                    })?;
                let effective = if config_json.get("text_config").is_some()
                    && config_json.get("hidden_size").is_none()
                {
                    serde_json::to_string(&config_json["text_config"]).map_err(|e| {
                        LoraError::InvalidState(format!(
                            "Failed to serialize Gemma4 text config: {}",
                            e
                        ))
                    })?
                } else {
                    config_content.clone()
                };
                let cfg: pmetal_models::architectures::gemma4::Gemma4Config =
                    serde_json::from_str(&effective).map_err(|e| {
                        LoraError::InvalidState(format!("Failed to parse Gemma4 config: {}", e))
                    })?;
                let model = Gemma4QloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Gemma4(model))
            }
            ModelArchitecture::GptOss => {
                let cfg: pmetal_models::architectures::gpt_oss::GptOssConfig =
                    serde_json::from_str(&config_content).map_err(|e| {
                        LoraError::InvalidState(format!("Failed to parse GptOss config: {}", e))
                    })?;
                let model = GptOssQloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::GptOss(model))
            }
            ModelArchitecture::Llama4 => {
                let config_json: serde_json::Value = serde_json::from_str(&config_content)
                    .map_err(|e| {
                        LoraError::InvalidState(format!("Failed to parse Llama4 JSON: {}", e))
                    })?;
                let effective = if config_json.get("text_config").is_some()
                    && config_json.get("hidden_size").is_none()
                {
                    serde_json::to_string(&config_json["text_config"]).map_err(|e| {
                        LoraError::InvalidState(format!(
                            "Failed to serialize Llama4 text config: {}",
                            e
                        ))
                    })?
                } else {
                    config_content.clone()
                };
                let cfg: pmetal_models::architectures::llama4::Llama4TextConfig =
                    serde_json::from_str(&effective).map_err(|e| {
                        LoraError::InvalidState(format!("Failed to parse Llama4 config: {}", e))
                    })?;
                let model = Llama4QloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::Llama4(model))
            }
            ModelArchitecture::DeepSeek => {
                let cfg: pmetal_models::architectures::deepseek::DeepSeekConfig =
                    serde_json::from_str(&config_content).map_err(|e| {
                        LoraError::InvalidState(format!("Failed to parse DeepSeek config: {}", e))
                    })?;
                let model = DeepSeekQloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::DeepSeek(model))
            }
            ModelArchitecture::NemotronH => {
                let cfg: pmetal_models::architectures::nemotron_h::NemotronHConfig =
                    serde_json::from_str(&config_content).map_err(|e| {
                        LoraError::InvalidState(format!("Failed to parse NemotronH config: {}", e))
                    })?;
                let model = NemotronHQloraForCausalLM::with_qlora_config(cfg, qlora_config)?;
                Ok(Self::NemotronH(model))
            }
            unsupported => Err(LoraError::InvalidState(format!(
                "QLoRA is not supported for {} models. \
                 QLoRA is available for: Llama, Mistral, Qwen3, Gemma, Qwen3Next, Qwen3MoE, \
                 Gemma4, GptOss, Granite, Llama4, DeepSeek, NemotronH. \
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
            Self::Qwen3Next(m) => m.load_and_quantize_from_dir(path),
            Self::Qwen3MoE(m) => m.load_and_quantize_from_dir(path),
            Self::Gemma4(m) => m.load_and_quantize_from_dir(path),
            Self::GptOss(m) => m.load_and_quantize_from_dir(path),
            // These adapters quantize inline during weight loading.
            Self::Granite(m) => m.load_base_weights_from_dir(path),
            Self::Llama4(m) => m.load_base_weights_from_dir(path),
            Self::DeepSeek(m) => m.load_base_weights_from_dir(path),
            Self::NemotronH(m) => m.load_base_weights_from_dir(path),
        }
    }

    /// Memory savings ratio vs. full-precision (ratio of QLoRA bytes to fp32 bytes).
    pub fn memory_savings(&self) -> f32 {
        match self {
            Self::Llama(m) => m.memory_savings(),
            Self::Mistral(m) => m.memory_savings(),
            Self::Qwen3(m) => m.memory_savings(),
            Self::Gemma(m) => m.memory_savings(),
            Self::Qwen3Next(m) => m.memory_savings(),
            Self::Qwen3MoE(m) => m.memory_savings(),
            Self::Gemma4(m) => m.memory_savings(),
            Self::GptOss(m) => m.memory_savings(),
            Self::Granite(m) => m.memory_savings(),
            Self::Llama4(m) => m.memory_savings(),
            Self::NemotronH(m) => m.memory_savings(),
            // DeepSeek does not expose memory_savings — derive from memory_usage.
            Self::DeepSeek(m) => {
                let (quantized, lora, _) = m.memory_usage();
                let total = quantized + lora;
                if total == 0 { 1.0 } else { (quantized + lora) as f32 / (total * 2) as f32 }
            }
        }
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
            Self::Qwen3Next(_) => "Qwen3Next",
            Self::Qwen3MoE(_) => "Qwen3MoE",
            Self::Gemma4(_) => "Gemma4",
            Self::GptOss(_) => "GptOss",
            Self::Granite(_) => "Granite",
            Self::Llama4(_) => "Llama4",
            Self::DeepSeek(_) => "DeepSeek",
            Self::NemotronH(_) => "NemotronH",
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
            Self::Qwen3Next(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Qwen3MoE(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Gemma4(m) => TrainableModel::forward(m, input_ids, mask),
            Self::GptOss(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Granite(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Llama4(m) => TrainableModel::forward(m, input_ids, mask),
            Self::DeepSeek(m) => TrainableModel::forward(m, input_ids, mask),
            Self::NemotronH(m) => TrainableModel::forward(m, input_ids, mask),
        }
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        match self {
            Self::Llama(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Mistral(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Qwen3(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Gemma(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Qwen3Next(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Qwen3MoE(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Gemma4(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
            Self::GptOss(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Granite(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Llama4(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
            Self::DeepSeek(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
            Self::NemotronH(m) => {
                TrainableModel::forward_with_positions(m, input_ids, mask, position_ids)
            }
        }
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut pmetal_mlx::kv_cache::KVCache>,
    ) -> Result<Array, LoraError> {
        match self {
            Self::Llama(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Mistral(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Qwen3(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Gemma(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Qwen3Next(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Qwen3MoE(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Gemma4(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::GptOss(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Granite(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Llama4(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::DeepSeek(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::NemotronH(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
        }
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<pmetal_mlx::kv_cache::KVCache> {
        match self {
            Self::Llama(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Mistral(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Qwen3(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Gemma(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Qwen3Next(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Qwen3MoE(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Gemma4(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::GptOss(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Granite(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Llama4(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::DeepSeek(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::NemotronH(_) => None,
        }
    }

    fn supports_kv_cache(&self) -> bool {
        match self {
            Self::Llama(m) => TrainableModel::supports_kv_cache(m),
            Self::Mistral(m) => TrainableModel::supports_kv_cache(m),
            Self::Qwen3(m) => TrainableModel::supports_kv_cache(m),
            Self::Gemma(m) => TrainableModel::supports_kv_cache(m),
            Self::Qwen3Next(m) => TrainableModel::supports_kv_cache(m),
            Self::Qwen3MoE(m) => TrainableModel::supports_kv_cache(m),
            Self::Gemma4(m) => TrainableModel::supports_kv_cache(m),
            Self::GptOss(m) => TrainableModel::supports_kv_cache(m),
            Self::Granite(_) => true,
            Self::Llama4(_) => true,
            Self::DeepSeek(_) => true,
            Self::NemotronH(_) => false,
        }
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        match self {
            Self::Llama(m) => TrainableModel::forward_hidden(m, input_ids, mask),
            Self::Mistral(m) => TrainableModel::forward_hidden(m, input_ids, mask),
            Self::Qwen3(m) => TrainableModel::forward_hidden(m, input_ids, mask),
            Self::Gemma(m) => TrainableModel::forward_hidden(m, input_ids, mask),
            Self::Qwen3Next(m) => TrainableModel::forward_hidden(m, input_ids, mask),
            Self::Qwen3MoE(m) => TrainableModel::forward_hidden(m, input_ids, mask),
            Self::Gemma4(m) => TrainableModel::forward_hidden(m, input_ids, mask),
            Self::GptOss(m) => TrainableModel::forward_hidden(m, input_ids, mask),
            Self::Granite(m) => TrainableModel::forward_hidden(m, input_ids, mask),
            Self::Llama4(m) => TrainableModel::forward_hidden(m, input_ids, mask),
            Self::DeepSeek(m) => TrainableModel::forward_hidden(m, input_ids, mask),
            Self::NemotronH(m) => TrainableModel::forward_hidden(m, input_ids, mask),
        }
    }

    fn forward_hidden_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Option<Result<Array, LoraError>> {
        match self {
            Self::Llama(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Mistral(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Qwen3(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Gemma(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Qwen3Next(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Qwen3MoE(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Gemma4(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
            Self::GptOss(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Granite(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
            Self::Llama4(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
            Self::DeepSeek(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
            Self::NemotronH(m) => {
                TrainableModel::forward_hidden_with_positions(m, input_ids, mask, position_ids)
            }
        }
    }

    fn lm_head_weight(&self) -> Option<Array> {
        match self {
            Self::Llama(m) => TrainableModel::lm_head_weight(m),
            Self::Mistral(m) => TrainableModel::lm_head_weight(m),
            Self::Qwen3(m) => TrainableModel::lm_head_weight(m),
            Self::Gemma(m) => TrainableModel::lm_head_weight(m),
            Self::Qwen3Next(m) => TrainableModel::lm_head_weight(m),
            Self::Qwen3MoE(m) => TrainableModel::lm_head_weight(m),
            Self::Gemma4(m) => TrainableModel::lm_head_weight(m),
            Self::GptOss(m) => TrainableModel::lm_head_weight(m),
            Self::Granite(m) => TrainableModel::lm_head_weight(m),
            Self::Llama4(m) => TrainableModel::lm_head_weight(m),
            Self::DeepSeek(m) => TrainableModel::lm_head_weight(m),
            Self::NemotronH(m) => TrainableModel::lm_head_weight(m),
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

#[cfg(test)]
mod tests {
    use super::DynamicQloraModel;
    use crate::{QLoraConfig, Qwen3QloraForCausalLM, TrainableModel};
    use pmetal_bridge::compat::Array;
    use pmetal_models::architectures::{
        gemma4::Gemma4Config, gpt_oss::GptOssConfig, qwen3::Qwen3Config, qwen3_moe::Qwen3MoEConfig,
        qwen3_next::Qwen3NextConfig,
    };
    use std::fs;

    fn write_config(model_type: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = serde_json::json!({ "model_type": model_type });
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config).expect("config string"),
        )
        .expect("write config");
        dir
    }

    fn write_json_config(config: serde_json::Value) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config).expect("config string"),
        )
        .expect("write config");
        dir
    }

    fn tiny_qwen3_next_config() -> Qwen3NextConfig {
        Qwen3NextConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: Some(16),
            vocab_size: 100,
            linear_num_value_heads: 2,
            linear_num_key_heads: 1,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 32,
            mlp_only_layers: vec![],
            norm_topk_prob: false,
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    fn tiny_qwen3_moe_config() -> Qwen3MoEConfig {
        Qwen3MoEConfig {
            vocab_size: 128,
            hidden_size: 32,
            intermediate_size: 64,
            moe_intermediate_size: Some(16),
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: 8,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            ..Default::default()
        }
    }

    fn tiny_qwen3_config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 128,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: 8,
            max_position_embeddings: 64,
            ..Default::default()
        }
    }

    fn tiny_gemma4_config() -> Gemma4Config {
        Gemma4Config {
            model_type: "gemma4_text".to_string(),
            vocab_size: 128,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            global_head_dim: None,
            num_global_key_value_heads: None,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            attention_k_eq_v: false,
            tie_word_embeddings: false,
            sliding_window: 16,
            final_logit_softcapping: None,
            layer_types: vec!["sliding_attention".to_string(); 2],
            rope_parameters: None,
            _raw_rope_parameters: None,
            hidden_size_per_layer_input: Some(8),
            vocab_size_per_layer_input: Some(128),
            hidden_activation: Some("gelu_pytorch_tanh".to_string()),
            num_kv_shared_layers: Some(0),
            use_double_wide_mlp: Some(false),
            enable_moe_block: Some(false),
        }
    }

    fn tiny_gpt_oss_config() -> GptOssConfig {
        GptOssConfig {
            hidden_size: 32,
            intermediate_size: 48,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 8,
            max_position_embeddings: 256,
            initial_context_length: 32,
            num_local_experts: 4,
            experts_per_token: 2,
            num_experts_per_tok: Some(2),
            vocab_size: 128,
            ..Default::default()
        }
    }

    #[test]
    fn llama4_dispatches_to_real_qlora_impl() {
        // Llama4 configs nest model params inside `text_config`.
        // A minimal flat config (no text_config wrapper) should also parse.
        use pmetal_models::architectures::llama4::Llama4TextConfig;
        let cfg = Llama4TextConfig {
            vocab_size: 512,
            hidden_size: 64,
            intermediate_size: 32,
            intermediate_size_mlp: 48,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 128,
            tie_word_embeddings: false,
            num_experts_per_tok: 1,
            num_local_experts: 2,
            interleave_moe_layer_step: 1,
            moe_layers: None,
            no_rope_layer_interval: 4,
            no_rope_layers: None,
            attention_chunk_size: 64,
            use_qk_norm: false,
            attn_temperature_tuning: false,
            floor_scale: 64,
            attn_scale: 0.1,
            router_aux_loss_coef: 0.001,
            use_mod: false,
            mod_capacity: 0.5,
            mod_layers: None,
            mod_layer_interval: 2,
        };
        let dir = write_json_config(serde_json::json!({
            "model_type": "llama4",
            "text_config": serde_json::to_value(cfg).unwrap(),
        }));
        let model = DynamicQloraModel::from_model_dir(dir.path(), QLoraConfig::default())
            .expect("llama4 qlora should construct");
        assert_eq!(model.arch_name(), "Llama4");
    }

    #[test]
    fn qwen3_next_nested_text_config_dispatches_to_real_qlora_impl() {
        let dir = write_json_config(serde_json::json!({
            "model_type": "qwen3_next",
            "text_config": tiny_qwen3_next_config(),
        }));
        let model = DynamicQloraModel::from_model_dir(dir.path(), QLoraConfig::default())
            .expect("qwen3_next qlora should construct");
        assert_eq!(model.arch_name(), "Qwen3Next");
    }

    #[test]
    fn qwen3_moe_dispatches_to_real_qlora_impl() {
        let dir = write_json_config(serde_json::to_value(tiny_qwen3_moe_config()).unwrap());
        let model = DynamicQloraModel::from_model_dir(dir.path(), QLoraConfig::default())
            .expect("qwen3_moe qlora should construct");
        assert_eq!(model.arch_name(), "Qwen3MoE");
    }

    #[test]
    fn gemma4_nested_text_config_dispatches_to_real_qlora_impl() {
        let dir = write_json_config(serde_json::json!({
            "model_type": "gemma4_text",
            "text_config": tiny_gemma4_config(),
        }));
        let model = DynamicQloraModel::from_model_dir(dir.path(), QLoraConfig::default())
            .expect("gemma4 qlora should construct");
        assert_eq!(model.arch_name(), "Gemma4");
    }

    #[test]
    fn gpt_oss_dispatches_to_real_qlora_impl() {
        let dir = write_json_config(serde_json::to_value(tiny_gpt_oss_config()).unwrap());
        let model = DynamicQloraModel::from_model_dir(dir.path(), QLoraConfig::default())
            .expect("gpt_oss qlora should construct");
        assert_eq!(model.arch_name(), "GptOss");
    }

    #[test]
    fn qwen3_forward_with_positions_is_delegated() {
        let mut model = DynamicQloraModel::Qwen3(
            Qwen3QloraForCausalLM::with_qlora_config(tiny_qwen3_config(), QLoraConfig::default())
                .expect("qwen3 qlora should construct"),
        );
        let input_ids = Array::from_i32_slice_shaped(&[1, 2, 3, 4], &[1, 4]);
        let zeros = Array::from_i32_slice_shaped(&[0, 0, 0, 0], &[4]);
        let seq = Array::from_i32_slice_shaped(&[0, 1, 2, 3], &[4]);

        let logits_zero = model
            .forward_with_positions(&input_ids, None, &zeros)
            .expect("forward with zero positions");
        let logits_seq = model
            .forward_with_positions(&input_ids, None, &seq)
            .expect("forward with sequential positions");
        let diff = logits_zero.subtract(&logits_seq);
        let sq_sum = diff.multiply(&diff).sum(None).item_f32();
        assert!(
            sq_sum > 0.0,
            "dynamic QLoRA wrapper should delegate explicit position ids instead of ignoring them"
        );
    }
}
