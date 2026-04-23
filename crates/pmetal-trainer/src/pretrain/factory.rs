//! Model factory for pretraining — creates a concrete model from an
//! architecture name + optional config.json, wrapped in an enum that
//! implements [`CausalLm`] via dispatch.

use std::path::Path;

use pmetal_bridge::compat::{
    Array, Exception,
    module::{ModuleParamMut, ModuleParamRef, ModuleParameters},
};
use pmetal_models::architectures;

use super::CausalLm;

/// Concrete model enum covering every architecture that has a [`CausalLm`]
/// implementation. Sized, so it works with the generic `run_pretrain<M>`.
pub enum PretrainModel {
    GptOss(architectures::GptOssForCausalLM),
    Llama(architectures::LlamaForCausalLM),
    Qwen2(architectures::Qwen2ForCausalLM),
    Qwen3(architectures::Qwen3ForCausalLM),
    Qwen3MoE(architectures::Qwen3MoE),
    Qwen3Next(Box<architectures::Qwen3NextForCausalLM>),
    Gemma(architectures::GemmaForCausalLM),
    Gemma4(Box<architectures::Gemma4ForCausalLM>),
    Mistral(architectures::MistralForCausalLM),
    Phi(architectures::PhiForCausalLM),
}

/// Macro to dispatch a method uniformly across all PretrainModel variants.
macro_rules! dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            PretrainModel::GptOss(m) => m.$method($($arg),*),
            PretrainModel::Llama(m) => m.$method($($arg),*),
            PretrainModel::Qwen2(m) => m.$method($($arg),*),
            PretrainModel::Qwen3(m) => m.$method($($arg),*),
            PretrainModel::Qwen3MoE(m) => m.$method($($arg),*),
            PretrainModel::Qwen3Next(m) => m.$method($($arg),*),
            PretrainModel::Gemma(m) => m.$method($($arg),*),
            PretrainModel::Gemma4(m) => m.$method($($arg),*),
            PretrainModel::Mistral(m) => m.$method($($arg),*),
            PretrainModel::Phi(m) => m.$method($($arg),*),
        }
    };
}

impl ModuleParameters for PretrainModel {
    fn num_parameters(&self) -> usize {
        dispatch!(self, num_parameters)
    }
    fn parameters(&self) -> ModuleParamRef<'_> {
        dispatch!(self, parameters)
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        dispatch!(self, parameters_mut)
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        dispatch!(self, trainable_parameters)
    }
}

impl CausalLm for PretrainModel {
    fn forward_logits(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        dispatch!(self, forward_logits, input_ids)
    }
    fn vocab_size(&self) -> i32 {
        dispatch!(self, vocab_size)
    }
}

/// Number of hidden layers for the model (used for depth-scaled init).
pub fn n_layers(model: &PretrainModel) -> usize {
    match model {
        PretrainModel::GptOss(m) => m.config().num_hidden_layers as usize,
        PretrainModel::Llama(m) => m.config().num_hidden_layers as usize,
        PretrainModel::Qwen2(m) => m.config().num_hidden_layers as usize,
        PretrainModel::Qwen3(m) => m.config.num_hidden_layers as usize,
        PretrainModel::Qwen3MoE(m) => m.config.num_hidden_layers as usize,
        PretrainModel::Qwen3Next(m) => m.config.num_hidden_layers as usize,
        PretrainModel::Gemma(m) => m.config().num_hidden_layers as usize,
        PretrainModel::Gemma4(m) => m.config.num_hidden_layers as usize,
        PretrainModel::Mistral(m) => m.config().num_hidden_layers as usize,
        PretrainModel::Phi(m) => m.config().num_hidden_layers as usize,
    }
}

/// Create a model from an architecture name and optional config.json path.
///
/// If `config_path` is provided, the JSON is deserialized into the
/// architecture-specific config struct. Otherwise, the architecture's
/// `Default` config is used.
pub fn create_model(arch: &str, config_path: Option<&Path>) -> Result<PretrainModel, Exception> {
    // Load JSON if provided
    let json: Option<serde_json::Value> = if let Some(path) = config_path {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Exception::custom(format!("read config: {e}")))?;
        Some(
            serde_json::from_str(&content)
                .map_err(|e| Exception::custom(format!("parse config: {e}")))?,
        )
    } else {
        None
    };

    match arch {
        "gpt-oss" | "gpt_oss" | "gptoss" => {
            let config: architectures::GptOssConfig = match &json {
                Some(v) => serde_json::from_value(v.clone())
                    .map_err(|e| Exception::custom(format!("gpt-oss config: {e}")))?,
                None => architectures::GptOssConfig::default(),
            };
            Ok(PretrainModel::GptOss(
                architectures::GptOssForCausalLM::new(config)?,
            ))
        }
        "llama" | "llama2" | "llama3" => {
            let config: architectures::LlamaConfig = match &json {
                Some(v) => serde_json::from_value(v.clone())
                    .map_err(|e| Exception::custom(format!("llama config: {e}")))?,
                None => architectures::LlamaConfig::default(),
            };
            Ok(PretrainModel::Llama(architectures::LlamaForCausalLM::new(
                config,
            )?))
        }
        "qwen2" | "qwen2.5" | "qwen2_5" => {
            let config: architectures::Qwen2Config = match &json {
                Some(v) => serde_json::from_value(v.clone())
                    .map_err(|e| Exception::custom(format!("qwen2 config: {e}")))?,
                None => architectures::Qwen2Config::default(),
            };
            Ok(PretrainModel::Qwen2(architectures::Qwen2ForCausalLM::new(
                config,
            )?))
        }
        "qwen3" => {
            let config: architectures::Qwen3Config = match &json {
                Some(v) => serde_json::from_value(v.clone())
                    .map_err(|e| Exception::custom(format!("qwen3 config: {e}")))?,
                None => architectures::Qwen3Config::default(),
            };
            Ok(PretrainModel::Qwen3(architectures::Qwen3ForCausalLM::new(
                config,
            )?))
        }
        "gemma" | "gemma2" | "gemma3" => {
            let config: architectures::GemmaConfig = match &json {
                Some(v) => serde_json::from_value(v.clone())
                    .map_err(|e| Exception::custom(format!("gemma config: {e}")))?,
                None => architectures::GemmaConfig::default(),
            };
            Ok(PretrainModel::Gemma(architectures::GemmaForCausalLM::new(
                config,
            )?))
        }
        "gemma4" | "gemma4_text" => {
            let json = json.as_ref().ok_or_else(|| {
                Exception::custom("gemma4 pretrain requires --model-config with a Gemma 4 config")
            })?;
            let effective =
                if json.get("text_config").is_some() && json.get("hidden_size").is_none() {
                    json["text_config"].clone()
                } else {
                    json.clone()
                };
            let config: architectures::Gemma4Config = serde_json::from_value(effective)
                .map_err(|e| Exception::custom(format!("gemma4 config: {e}")))?;
            Ok(PretrainModel::Gemma4(Box::new(
                architectures::Gemma4ForCausalLM::new(config)?,
            )))
        }
        "mistral" | "mixtral" => {
            let config: architectures::MistralConfig = match &json {
                Some(v) => serde_json::from_value(v.clone())
                    .map_err(|e| Exception::custom(format!("mistral config: {e}")))?,
                None => architectures::MistralConfig::default(),
            };
            Ok(PretrainModel::Mistral(
                architectures::MistralForCausalLM::new(config)?,
            ))
        }
        "qwen3_moe" | "qwen3-moe" => {
            let config: architectures::Qwen3MoEConfig = match &json {
                Some(v) => serde_json::from_value(v.clone())
                    .map_err(|e| Exception::custom(format!("qwen3-moe config: {e}")))?,
                None => architectures::Qwen3MoEConfig::default(),
            };
            Ok(PretrainModel::Qwen3MoE(architectures::Qwen3MoE::new(
                config,
            )?))
        }
        "qwen3.5" | "qwen3_5" | "qwen3_next" | "qwen3-next" | "qwen35" => {
            let mut config: architectures::Qwen3NextConfig = match &json {
                Some(v) => {
                    let effective =
                        if v.get("text_config").is_some() && v.get("hidden_size").is_none() {
                            v["text_config"].clone()
                        } else {
                            v.clone()
                        };
                    serde_json::from_value(effective)
                        .map_err(|e| Exception::custom(format!("qwen3.5 config: {e}")))?
                }
                None => architectures::Qwen3NextConfig::default(),
            };
            config.apply_rope_parameters();
            Ok(PretrainModel::Qwen3Next(Box::new(
                architectures::Qwen3NextForCausalLM::new(config)?,
            )))
        }
        "phi" | "phi3" | "phi4" => {
            let config: architectures::PhiConfig = match &json {
                Some(v) => serde_json::from_value(v.clone())
                    .map_err(|e| Exception::custom(format!("phi config: {e}")))?,
                None => architectures::PhiConfig::default(),
            };
            Ok(PretrainModel::Phi(architectures::PhiForCausalLM::new(
                config,
            )?))
        }
        other => Err(Exception::custom(format!(
            "unsupported pretrain architecture: {other}\n\
             supported: llama, qwen2, qwen3, qwen3.5, qwen3_moe, gemma, gemma4, mistral, phi, gpt-oss"
        ))),
    }
}
