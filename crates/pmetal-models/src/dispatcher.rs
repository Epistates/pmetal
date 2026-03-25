//! Dynamic model dispatch based on config.json model_type.
//!
//! This module provides automatic architecture detection and model loading,
//! eliminating the need for hardcoded model types in application code.

use crate::architectures::*;
use crate::loader::{
    load_bert_weights, load_falcon_h1_weights, load_generic_weights, load_nemotron_weights,
    load_qwen3_next_weights_with_options, load_weights, Qwen3NextLoadOptions,
};
use crate::traits::{CausalLMModel, ModelConfig};
use mlx_rs::{
    Array,
    error::Exception,
    module::{Module, ModuleParameters},
};
use pmetal_mlx::kv_cache::{CacheMode, KVCache, KVCacheConfig, MambaCache};
use std::path::Path;

/// Find the largest power-of-2 group size that divides `head_dim`,
/// preferring the requested `preferred` if it already works.
fn find_compatible_group_size(head_dim: usize, preferred: usize) -> usize {
    if head_dim % preferred == 0 {
        return preferred;
    }
    // Try common group sizes in decreasing order: 64, 32, 16, 8, 4, 2, 1
    for gs in [64, 32, 16, 8, 4, 2, 1] {
        if head_dim % gs == 0 {
            return gs;
        }
    }
    1 // Fallback: per-element quantization (always works)
}

const PARAM_EVAL_BATCH_SIZE: usize = 128;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DynamicModelLoadOptions {
    pub prefer_expert_offload: bool,
}

fn eval_module_parameters_batched(module: &impl ModuleParameters) -> Result<(), Exception> {
    let params = module.parameters().flatten();
    let arrays: Vec<&Array> = params.values().copied().collect();

    for chunk in arrays.chunks(PARAM_EVAL_BATCH_SIZE) {
        mlx_rs::transforms::eval(chunk.iter().copied())?;
    }

    Ok(())
}

/// Model architecture types supported by PMetal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelArchitecture {
    Llama,
    Llama4,
    Qwen2,
    Qwen3,
    Qwen3MoE,
    Gemma,
    Mistral,
    Phi,
    Phi4,
    DeepSeek,
    Cohere,
    Granite,
    NemotronH,
    Qwen3Next,
    StarCoder2,
    RecurrentGemma,
    Jamba,
    FalconH1,
    Flux,
    /// BERT / RoBERTa / DistilBERT encoder-only model.
    Bert,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama => write!(f, "Llama"),
            Self::Llama4 => write!(f, "Llama 4"),
            Self::Qwen2 => write!(f, "Qwen 2"),
            Self::Qwen3 => write!(f, "Qwen 3"),
            Self::Qwen3MoE => write!(f, "Qwen 3 MoE"),
            Self::Gemma => write!(f, "Gemma"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Phi => write!(f, "Phi"),
            Self::Phi4 => write!(f, "Phi 4"),
            Self::DeepSeek => write!(f, "DeepSeek"),
            Self::Cohere => write!(f, "Cohere"),
            Self::Granite => write!(f, "Granite"),
            Self::NemotronH => write!(f, "NemotronH"),
            Self::Qwen3Next => write!(f, "Qwen 3.5"),
            Self::StarCoder2 => write!(f, "StarCoder2"),
            Self::RecurrentGemma => write!(f, "RecurrentGemma"),
            Self::Jamba => write!(f, "Jamba"),
            Self::FalconH1 => write!(f, "Falcon H1"),
            Self::Flux => write!(f, "Flux"),
            Self::Bert => write!(f, "BERT"),
        }
    }
}

impl ModelArchitecture {
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        let lower = model_type.to_lowercase();
        match lower.as_str() {
            "llama4" => Some(Self::Llama4),
            "llama" | "llama3" => Some(Self::Llama),
            "qwen3_moe" => Some(Self::Qwen3MoE),
            "qwen3_next"
            | "qwen3_5"
            | "qwen3.5"
            | "qwen3_5_text"
            | "qwen3_5_moe"
            | "qwen3_5_moe_text" => Some(Self::Qwen3Next),
            "qwen3" => Some(Self::Qwen3),
            "qwen2" | "qwen2_5" => Some(Self::Qwen2),
            "gemma" | "gemma2" | "gemma3" => Some(Self::Gemma),
            "mistral" | "mixtral" => Some(Self::Mistral),
            "phi4" => Some(Self::Phi4),
            "phi" | "phi3" => Some(Self::Phi),
            "deepseek" | "deepseek2" | "deepseek_v2" | "deepseek_v3" => Some(Self::DeepSeek),
            "cohere" | "cohere2" | "command_r" | "command-r" => Some(Self::Cohere),
            "granite" | "granitehybrid" | "granite_moe" => Some(Self::Granite),
            "nemotron_h" | "nemotronh" | "nemotron-h" => Some(Self::NemotronH),
            "starcoder2" | "starcoder-2" => Some(Self::StarCoder2),
            "recurrentgemma" | "recurrent-gemma" | "griffin" => Some(Self::RecurrentGemma),
            "jamba" | "jamba-1.5" => Some(Self::Jamba),
            "falcon_h1" | "falconh1" | "falcon-h1" => Some(Self::FalconH1),
            "flux" | "flux-1" | "flux.1" => Some(Self::Flux),
            "bert" | "roberta" | "distilbert" | "xlm-roberta" | "xlm_roberta" => Some(Self::Bert),
            _ => None,
        }
    }

    pub fn from_architectures(archs: &[String]) -> Option<Self> {
        for arch in archs {
            let lower = arch.to_lowercase();
            if lower.contains("llama4") {
                return Some(Self::Llama4);
            }
            if lower.contains("llama") {
                return Some(Self::Llama);
            }
            if lower.contains("qwen3moe") || lower.contains("qwen3_moe") {
                return Some(Self::Qwen3MoE);
            }
            if lower.contains("qwen3next")
                || lower.contains("qwen3_next")
                || lower.contains("qwen35")
                || lower.contains("qwen3_5")
                || lower.contains("qwen3.5")
                || lower.contains("qwen35moe")
                || lower.contains("qwen3_5_moe")
            {
                return Some(Self::Qwen3Next);
            }
            if lower.contains("qwen3") {
                return Some(Self::Qwen3);
            }
            if lower.contains("qwen2") || lower.contains("qwen") {
                return Some(Self::Qwen2);
            }
            if lower.contains("gemma") {
                if lower.contains("recurrent") {
                    return Some(Self::RecurrentGemma);
                }
                return Some(Self::Gemma);
            }
            if lower.contains("mistral") || lower.contains("mixtral") {
                return Some(Self::Mistral);
            }
            if lower.contains("phi4") {
                return Some(Self::Phi4);
            }
            if lower.contains("phi") {
                return Some(Self::Phi);
            }
            if lower.contains("deepseek") {
                return Some(Self::DeepSeek);
            }
            if lower.contains("cohere") || lower.contains("commandr") || lower.contains("command_r")
            {
                return Some(Self::Cohere);
            }
            if lower.contains("granite") {
                return Some(Self::Granite);
            }
            if lower.contains("nemotronhforcausallm") || lower.contains("nemotron_h") {
                return Some(Self::NemotronH);
            }
            if lower.contains("starcoder2") {
                return Some(Self::StarCoder2);
            }
            if lower.contains("jamba") {
                return Some(Self::Jamba);
            }
            if lower.contains("falconh1")
                || lower.contains("falcon_h1")
                || lower.contains("falcon-h1")
            {
                return Some(Self::FalconH1);
            }
            if lower.contains("flux") {
                return Some(Self::Flux);
            }
            // Check BERT family after other checks to avoid false positives
            if lower.contains("bert") {
                return Some(Self::Bert);
            }
        }
        None
    }

    pub fn detect<P: AsRef<Path>>(model_dir: P) -> Result<Self, Exception> {
        let config_path = model_dir.as_ref().join("config.json");
        if !config_path.exists() {
            return Err(Exception::custom(format!(
                "Config file not found: {:?}",
                config_path
            )));
        }
        let config_content = std::fs::read_to_string(config_path)
            .map_err(|e| Exception::custom(format!("{}", e)))?;
        let config: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| Exception::custom(format!("{}", e)))?;

        let architectures = config["architectures"].as_array().map(|a| {
            a.iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect::<Vec<_>>()
        });
        let model_type = config["model_type"].as_str().unwrap_or("");

        Self::from_model_type(model_type)
            .or_else(|| {
                architectures
                    .as_ref()
                    .and_then(|a| Self::from_architectures(a))
            })
            .ok_or_else(|| Exception::custom(format!("Unsupported model type: {}", model_type)))
    }
}

/// Dispatch a method call uniformly across all `DynamicModel` variants.
///
/// Every arm expands to `m.$method($args...)` where `m` is the inner model.
/// Use this only for methods where ALL variants have identical call signatures.
macro_rules! dispatch_uniform {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            Self::Llama(m) => m.$method($($arg),*),
            Self::Llama4(m) => m.$method($($arg),*),
            Self::Qwen2(m) => m.$method($($arg),*),
            Self::Qwen3(m) => m.$method($($arg),*),
            Self::Qwen3MoE(m) => m.$method($($arg),*),
            Self::Gemma(m) => m.$method($($arg),*),
            Self::Mistral(m) => m.$method($($arg),*),
            Self::Phi(m) => m.$method($($arg),*),
            Self::Phi4(m) => m.$method($($arg),*),
            Self::DeepSeek(m) => m.$method($($arg),*),
            Self::Cohere(m) => m.$method($($arg),*),
            Self::Granite(m) => m.$method($($arg),*),
            Self::NemotronH(m) => m.$method($($arg),*),
            Self::Qwen3Next(m) => m.$method($($arg),*),
            Self::StarCoder2(m) => m.$method($($arg),*),
            Self::RecurrentGemma(m) => m.$method($($arg),*),
            Self::Jamba(m) => m.$method($($arg),*),
            Self::FalconH1(m) => m.$method($($arg),*),
            Self::Flux(m) => m.$method($($arg),*),
            Self::Bert(m) => m.$method($($arg),*),
        }
    };
}

/// Map each `DynamicModel` variant to its corresponding `ModelArchitecture` constant.
macro_rules! dispatch_architecture {
    ($self:expr) => {
        match $self {
            Self::Llama(_) => ModelArchitecture::Llama,
            Self::Llama4(_) => ModelArchitecture::Llama4,
            Self::Qwen2(_) => ModelArchitecture::Qwen2,
            Self::Qwen3(_) => ModelArchitecture::Qwen3,
            Self::Qwen3MoE(_) => ModelArchitecture::Qwen3MoE,
            Self::Gemma(_) => ModelArchitecture::Gemma,
            Self::Mistral(_) => ModelArchitecture::Mistral,
            Self::Phi(_) => ModelArchitecture::Phi,
            Self::Phi4(_) => ModelArchitecture::Phi4,
            Self::DeepSeek(_) => ModelArchitecture::DeepSeek,
            Self::Cohere(_) => ModelArchitecture::Cohere,
            Self::Granite(_) => ModelArchitecture::Granite,
            Self::NemotronH(_) => ModelArchitecture::NemotronH,
            Self::Qwen3Next(_) => ModelArchitecture::Qwen3Next,
            Self::StarCoder2(_) => ModelArchitecture::StarCoder2,
            Self::RecurrentGemma(_) => ModelArchitecture::RecurrentGemma,
            Self::Jamba(_) => ModelArchitecture::Jamba,
            Self::FalconH1(_) => ModelArchitecture::FalconH1,
            Self::Flux(_) => ModelArchitecture::Flux,
            Self::Bert(_) => ModelArchitecture::Bert,
        }
    };
}

/// A model whose architecture is dispatched at runtime.
pub enum DynamicModel {
    Llama(LlamaForCausalLM),
    Llama4(Llama4ForCausalLM),
    Qwen2(Qwen2ForCausalLM),
    Qwen3(Qwen3ForCausalLM),
    Qwen3MoE(Qwen3MoE),
    Gemma(GemmaForCausalLM),
    Mistral(MistralForCausalLM),
    Phi(PhiForCausalLM),
    Phi4(PhiForCausalLM),
    DeepSeek(DeepSeek),
    Cohere(CohereForCausalLM),
    Granite(GraniteForCausalLM),
    NemotronH(NemotronHForCausalLM),
    Qwen3Next(Qwen3NextForCausalLM),
    StarCoder2(StarCoder2Model),
    RecurrentGemma(RecurrentGemmaModel),
    Jamba(JambaModel),
    FalconH1(FalconH1ForCausalLM),
    Flux(FluxDiT),
    Bert(BertForEmbedding),
}

impl std::fmt::Debug for DynamicModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama(_) => write!(f, "DynamicModel::Llama"),
            Self::Llama4(_) => write!(f, "DynamicModel::Llama4"),
            Self::Qwen2(_) => write!(f, "DynamicModel::Qwen2"),
            Self::Qwen3(_) => write!(f, "DynamicModel::Qwen3"),
            Self::Qwen3MoE(_) => write!(f, "DynamicModel::Qwen3MoE"),
            Self::Gemma(_) => write!(f, "DynamicModel::Gemma"),
            Self::Mistral(_) => write!(f, "DynamicModel::Mistral"),
            Self::Phi(_) => write!(f, "DynamicModel::Phi"),
            Self::Phi4(_) => write!(f, "DynamicModel::Phi4"),
            Self::DeepSeek(_) => write!(f, "DynamicModel::DeepSeek"),
            Self::Cohere(_) => write!(f, "DynamicModel::Cohere"),
            Self::Granite(_) => write!(f, "DynamicModel::Granite"),
            Self::NemotronH(_) => write!(f, "DynamicModel::NemotronH"),
            Self::Qwen3Next(_) => write!(f, "DynamicModel::Qwen3Next"),
            Self::StarCoder2(_) => write!(f, "DynamicModel::StarCoder2"),
            Self::RecurrentGemma(_) => write!(f, "DynamicModel::RecurrentGemma"),
            Self::Jamba(_) => write!(f, "DynamicModel::Jamba"),
            Self::FalconH1(_) => write!(f, "DynamicModel::FalconH1"),
            Self::Flux(_) => write!(f, "DynamicModel::Flux"),
            Self::Bert(_) => write!(f, "DynamicModel::Bert"),
        }
    }
}

impl DynamicModel {
    fn init_post_load_fast_paths(&mut self) -> Result<(), Exception> {
        match self {
            Self::Qwen3MoE(model) => model.init_stacked_moe(),
            Self::DeepSeek(model) => model.init_stacked_moe(),
            Self::NemotronH(model) => model.init_stacked_moe(),
            _ => Ok(()),
        }
    }

    /// Load a model from a directory, automatically detecting its architecture.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, Exception> {
        Self::load_with_options(model_dir, DynamicModelLoadOptions::default())
    }

    /// Load a model from a directory with caller-controlled load behavior.
    pub fn load_with_options(
        model_dir: impl AsRef<Path>,
        options: DynamicModelLoadOptions,
    ) -> Result<Self, Exception> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        if !config_path.exists() {
            return Err(Exception::custom(format!(
                "Config file not found: {:?}",
                config_path
            )));
        }
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| Exception::custom(format!("{}", e)))?;
        let base_config: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| Exception::custom(format!("{}", e)))?;
        let architectures = base_config["architectures"].as_array().map(|a| {
            a.iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect::<Vec<_>>()
        });
        let model_type = base_config["model_type"].as_str().unwrap_or("");
        let arch = ModelArchitecture::from_model_type(model_type)
            .or_else(|| {
                architectures
                    .as_ref()
                    .and_then(|a| ModelArchitecture::from_architectures(a))
            })
            .ok_or_else(|| {
                Exception::custom(format!("Unsupported architecture: {}", model_type))
            })?;

        match arch {
            ModelArchitecture::Llama => {
                let config: LlamaConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = LlamaForCausalLM::new(config)?;
                crate::loader::load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Llama(model))
            }
            ModelArchitecture::Llama4 => {
                let config: Llama4TextConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = Llama4ForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Llama4(model))
            }
            ModelArchitecture::Qwen2 => {
                let config: Qwen2Config = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = Qwen2ForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Qwen2(model))
            }
            ModelArchitecture::Qwen3 => {
                let config: Qwen3Config = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = Qwen3ForCausalLM::new_for_loading(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Qwen3(model))
            }
            ModelArchitecture::Qwen3MoE => {
                let config: Qwen3MoEConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = Qwen3MoE::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                let mut model = Self::Qwen3MoE(model);
                model.init_post_load_fast_paths()?;
                Ok(model)
            }
            ModelArchitecture::Gemma => {
                let mut config: GemmaConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                // Set the Gemma3 flag based on model_type to enable the correct
                // sliding window pattern (every 6th layer global, rest local).
                if config.model_type == "gemma3" {
                    config.is_gemma3 = true;
                }
                let mut model = GemmaForCausalLM::new(config)?;
                let weights = crate::loader::load_weights(model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                crate::loader::load_gemma_weights(&mut model, &weights)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Gemma(model))
            }
            ModelArchitecture::Mistral => {
                let config: MistralConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = MistralForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Mistral(model))
            }
            ModelArchitecture::Phi => {
                let config: PhiConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = PhiForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Phi(model))
            }
            ModelArchitecture::Phi4 => {
                let config: PhiConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = PhiForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Phi4(model))
            }
            ModelArchitecture::DeepSeek => {
                let config: DeepSeekConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = DeepSeek::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                let mut model = Self::DeepSeek(model);
                model.init_post_load_fast_paths()?;
                Ok(model)
            }
            ModelArchitecture::Cohere => {
                let config: CohereConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = CohereForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Cohere(model))
            }
            ModelArchitecture::Granite => {
                let config: GraniteConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = GraniteForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Granite(model))
            }
            ModelArchitecture::NemotronH => {
                let config: NemotronHConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = NemotronHForCausalLM::new(config)?;
                crate::loader::load_nemotron_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                let mut model = Self::NemotronH(model);
                model.init_post_load_fast_paths()?;
                Ok(model)
            }
            ModelArchitecture::Qwen3Next => {
                // Qwen 3.5 configs may have text_config nesting (VLM wrapper format)
                let config_json: serde_json::Value = serde_json::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let text_config_str = if config_json.get("text_config").is_some()
                    && config_json.get("hidden_size").is_none()
                {
                    serde_json::to_string(&config_json["text_config"])
                        .map_err(|e| Exception::custom(e.to_string()))?
                } else {
                    config_content.clone()
                };
                let mut config: Qwen3NextConfig = serde_json::from_str(&text_config_str)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                config.apply_rope_parameters();
                let skip_routed_experts =
                    options.prefer_expert_offload && config.num_experts > 0;
                let routed_expert_mode = if skip_routed_experts {
                    Qwen3NextRoutedExpertMode::Placeholder
                } else {
                    Qwen3NextRoutedExpertMode::Resident
                };
                let mut model =
                    Qwen3NextForCausalLM::new_with_routed_expert_mode(
                        config.clone(),
                        routed_expert_mode,
                    )?;
                let load_options = if skip_routed_experts {
                    Qwen3NextLoadOptions {
                        skip_routed_experts: true,
                    }
                } else {
                    Qwen3NextLoadOptions::default()
                };
                load_qwen3_next_weights_with_options(&mut model, model_dir, &config, load_options)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Qwen3Next(model))
            }
            ModelArchitecture::StarCoder2 => {
                let config: StarCoder2Config = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = StarCoder2Model::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::StarCoder2(model))
            }
            ModelArchitecture::RecurrentGemma => {
                let config: RecurrentGemmaConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = RecurrentGemmaModel::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::RecurrentGemma(model))
            }
            ModelArchitecture::Jamba => {
                let config: JambaConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = JambaModel::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Jamba(model))
            }
            ModelArchitecture::FalconH1 => {
                let config: FalconH1Config = serde_json::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = FalconH1ForCausalLM::new(config)?;
                let weights = crate::loader::load_weights(model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                load_falcon_h1_weights(&mut model, &weights)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::FalconH1(model))
            }
            ModelArchitecture::Flux => Err(Exception::custom(
                "Flux models are diffusion pipelines, not causal language models. Load them via pmetal_models::pipelines::FluxPipeline instead of DynamicModel::load.",
            )),
            ModelArchitecture::Bert => {
                let config: BertConfig = serde_json::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = BertForEmbedding::new(config)?;
                // Load weights using the HF→PMetal name remapper.  HuggingFace BERT
                // checkpoints use paths like `bert.encoder.layer.0.attention.self.query.*`
                // which differ from PMetal's `model.layers.0.attention.query.*`.
                let weights =
                    load_weights(model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?;
                load_bert_weights(&mut model, &weights)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Bert(model))
            }
        }
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        match self {
            Self::Llama(m) => m.forward(input_ids, mask),
            Self::Llama4(m) => m.forward(input_ids, mask, None),
            Self::Qwen2(m) => m.forward(input_ids, mask),
            Self::Qwen3(m) => m.forward(input_ids, mask),
            Self::Qwen3MoE(m) => m.forward(input_ids, mask, None),
            Self::Gemma(m) => m.forward(input_ids, mask),
            Self::Mistral(m) => m.forward(input_ids, mask),
            Self::Phi(m) => m.forward(input_ids, mask),
            Self::Phi4(m) => m.forward(input_ids, mask),
            Self::DeepSeek(m) => m.forward(input_ids, mask, None),
            Self::Cohere(m) => m.forward(input_ids, mask, None),
            Self::Granite(m) => m.forward(input_ids, mask, None),
            Self::NemotronH(m) => m.forward(input_ids, None),
            Self::Qwen3Next(m) => m.forward(input_ids, mask),
            Self::StarCoder2(m) => m.forward(input_ids, mask, None),
            Self::RecurrentGemma(m) => m.forward(input_ids),
            Self::Jamba(m) => m.forward(input_ids),
            Self::FalconH1(m) => m.forward(input_ids, mask),
            Self::Flux(_) => Err(Exception::custom(
                "Flux is not a CausalLM and does not support standard forward(input_ids, mask)",
            )),
            // BERT encoder: forward returns hidden states [batch, seq, hidden], not logits.
            // Use EmbeddingTrainer::encode() / pmetal_models::pooling::pool() for embeddings.
            Self::Bert(m) => BertForEmbedding::forward(m, input_ids, mask),
        }
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Llama(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Qwen2(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Qwen3(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Qwen3MoE(m) => m.forward(input_ids, mask, cache),
            Self::DeepSeek(m) => m.forward(input_ids, mask, cache),
            Self::Cohere(m) => m.forward_with_cache(input_ids, mask, cache),
            // Granite forward takes position_ids (not KVCache); cache ignored for now
            Self::Granite(m) => m.forward(input_ids, mask, None),
            Self::StarCoder2(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Llama4(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Gemma(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Mistral(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Phi(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Phi4(m) => m.forward_with_cache(input_ids, mask, cache),
            // Hybrid recurrent+attention models require both a KV cache and a
            // Mamba/GDN state cache. Use `forward_with_hybrid_cache` instead.
            Self::NemotronH(_) | Self::Qwen3Next(_) | Self::FalconH1(_) => Err(Exception::custom(
                "Hybrid architecture requires both KV and Mamba caches. \
                 Use DynamicModel::forward_with_hybrid_cache with both \
                 create_cache() and create_mamba_cache().",
            )),
            // Purely recurrent models have no KV cache; use forward() directly.
            Self::RecurrentGemma(m) => m.forward(input_ids),
            Self::Jamba(m) => m.forward(input_ids),
            Self::Flux(_) => Err(Exception::custom(
                "Flux is not a CausalLM and does not support forward_with_cache.",
            )),
            // BERT is encoder-only (no autoregressive cache) — delegate to standard forward.
            Self::Bert(m) => BertForEmbedding::forward(m, input_ids, mask),
        }
    }

    pub fn quantize_fp8(&mut self) -> Result<(), Exception> {
        match self {
            // NemotronH has a bespoke implementation that operates on concrete
            // `nn::Linear` structs and handles Mamba/attention blocks explicitly.
            Self::NemotronH(model) => model.quantize_fp8_weights(),

            // Flux is a diffusion model whose weight graph is not a flat set of
            // `nn::Linear` layers reachable through a single `ModuleParameters`
            // root.  Callers should use `FluxPipeline` and quantize each
            // component (transformer, VAE, text encoders) individually.
            Self::Flux(_) => Err(Exception::custom(
                "Flux FP8 quantization is not exposed through DynamicModel. \
                 Load the diffusion pipeline via pmetal_models::pipelines::FluxPipeline \
                 and quantize its components explicitly.",
            )),

            // All other causal-LM architectures: traverse the flattened
            // parameter map and quantize every `.weight` tensor generically.
            Self::Llama(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Llama4(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Qwen2(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Qwen3(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Qwen3MoE(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Gemma(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Mistral(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Phi(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Phi4(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::DeepSeek(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Cohere(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Granite(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Qwen3Next(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::StarCoder2(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::RecurrentGemma(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Jamba(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::FalconH1(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Bert(m) => crate::fp8_utils::quantize_model_linears(m),
        }
    }

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        match self {
            Self::Llama(m) => m.create_cache(max_seq_len),
            Self::Llama4(m) => KVCache::new(KVCacheConfig::new(
                m.config.num_hidden_layers as usize,
                max_seq_len,
                m.config.num_key_value_heads as usize,
                (m.config.hidden_size / m.config.num_attention_heads) as usize,
            )),
            Self::Qwen2(m) => KVCache::new(KVCacheConfig::new(
                m.config().num_hidden_layers() as usize,
                max_seq_len,
                m.config().num_kv_heads() as usize,
                m.config().head_dim() as usize,
            )),
            Self::Qwen3(m) => KVCache::new(KVCacheConfig::new(
                m.config().num_hidden_layers() as usize,
                max_seq_len,
                m.config().num_kv_heads() as usize,
                m.config().head_dim() as usize,
            )),
            Self::Qwen3MoE(m) => KVCache::new(KVCacheConfig::new(
                m.config.num_hidden_layers as usize,
                max_seq_len,
                m.config.num_kv_heads() as usize,
                (m.config.hidden_size / m.config.num_attention_heads) as usize,
            )),
            Self::Gemma(m) => m.create_cache(max_seq_len),
            Self::Mistral(m) => m.create_cache(max_seq_len),
            Self::Phi(m) => m.create_cache(max_seq_len),
            Self::Phi4(m) => m.create_cache(max_seq_len),
            Self::DeepSeek(m) => m.create_cache(max_seq_len),
            Self::Cohere(m) => KVCache::new(KVCacheConfig::new(
                m.config.num_hidden_layers as usize,
                max_seq_len,
                m.config.num_key_value_heads as usize,
                (m.config.hidden_size / m.config.num_attention_heads) as usize,
            )),
            Self::Granite(m) => KVCache::new(KVCacheConfig::new(
                m.config.num_hidden_layers as usize,
                max_seq_len,
                m.config.num_key_value_heads as usize,
                (m.config.hidden_size / m.config.num_attention_heads) as usize,
            )),
            Self::NemotronH(m) => KVCache::new(KVCacheConfig::new(
                m.config().num_hidden_layers() as usize,
                max_seq_len,
                m.config().num_kv_heads() as usize,
                m.config().head_dim() as usize,
            )),
            Self::Qwen3Next(m) => KVCache::new(KVCacheConfig::new(
                m.config().num_hidden_layers() as usize,
                max_seq_len,
                m.config().num_kv_heads() as usize,
                m.config().head_dim() as usize,
            )),
            Self::StarCoder2(m) => KVCache::new(KVCacheConfig::new(
                m.config.num_hidden_layers as usize,
                max_seq_len,
                m.config.num_key_value_heads as usize,
                (m.config.hidden_size / m.config.num_attention_heads) as usize,
            )),
            Self::RecurrentGemma(_) => KVCache::new(KVCacheConfig::new(0, 0, 0, 0)),
            Self::Jamba(m) => KVCache::new(KVCacheConfig::new(
                m.config.num_hidden_layers as usize,
                max_seq_len,
                m.config.num_key_value_heads as usize,
                (m.config.hidden_size / m.config.num_attention_heads) as usize,
            )),
            Self::FalconH1(m) => KVCache::new(KVCacheConfig::new(
                m.config().num_hidden_layers() as usize,
                max_seq_len,
                m.config().num_kv_heads() as usize,
                m.config().head_dim() as usize,
            )),
            Self::Flux(_) => KVCache::new(KVCacheConfig::new(0, 0, 0, 0)),
            // BERT is encoder-only with no autoregressive KV cache.
            Self::Bert(_) => KVCache::new(KVCacheConfig::new(0, 0, 0, 0)),
        }
    }

    /// Create a KV cache with a specific cache mode (e.g., quantized).
    ///
    /// This builds the same cache configuration as `create_cache` but applies
    /// the specified mode. Use `CacheMode::Quantized { bits: 8, group_size: 64 }`
    /// for the recommended q8_0 "free lunch" (< 0.4% PPL loss, 12-38% throughput gain).
    pub fn create_cache_with_mode(&self, max_seq_len: usize, mode: CacheMode) -> KVCache {
        let base = self.create_cache(max_seq_len);
        let base_config = base.config();
        let head_dim = base_config.head_dim;

        // Ensure group_size is compatible with head_dim (MLX quantize requires divisibility).
        // Models like Phi-3 mini (head_dim=96) or NemotronH (head_dim=32) need adjustment.
        let safe_mode = match mode {
            CacheMode::Quantized { bits, group_size }
                if head_dim > 0 && head_dim % group_size != 0 =>
            {
                let safe_gs = find_compatible_group_size(head_dim, group_size);
                tracing::info!(
                    "KV cache: adjusted group_size {group_size} → {safe_gs} (head_dim={head_dim})"
                );
                CacheMode::Quantized {
                    bits,
                    group_size: safe_gs,
                }
            }
            CacheMode::AsymmetricQuantized {
                key_bits,
                value_bits,
                group_size,
            } if head_dim > 0 && head_dim % group_size != 0 => {
                let safe_gs = find_compatible_group_size(head_dim, group_size);
                tracing::info!(
                    "KV cache: adjusted group_size {group_size} → {safe_gs} (head_dim={head_dim})"
                );
                CacheMode::AsymmetricQuantized {
                    key_bits,
                    value_bits,
                    group_size: safe_gs,
                }
            }
            other => other,
        };

        let config = base_config.clone().with_mode(safe_mode);
        KVCache::new(config)
    }

    pub fn create_mamba_cache(&self) -> Option<MambaCache> {
        match self {
            Self::NemotronH(m) => Some(MambaCache::new(m.config().num_hidden_layers() as usize)),
            Self::Qwen3Next(m) => Some(MambaCache::new(m.config().num_hidden_layers() as usize)),
            // FalconH1: every layer has Mamba, so the cache covers all layers.
            Self::FalconH1(m) => Some(MambaCache::new(m.config().num_hidden_layers() as usize)),
            _ => None,
        }
    }

    pub fn forward_with_hybrid_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        match self {
            Self::NemotronH(m) => m.forward_with_cache(input_ids, mask, kv_cache, mamba_cache),
            Self::Qwen3Next(m) => m.forward_with_cache(input_ids, mask, kv_cache, mamba_cache),
            Self::FalconH1(m) => m.forward_with_cache(input_ids, mask, kv_cache, mamba_cache),
            _ => self.forward_with_cache(input_ids, mask, kv_cache),
        }
    }

    pub fn architecture(&self) -> ModelArchitecture {
        dispatch_architecture!(self)
    }

    pub fn vocab_size(&self) -> i32 {
        match self {
            Self::Llama(m) => m.model.config.vocab_size,
            Self::Llama4(m) => m.config.vocab_size,
            Self::Qwen2(m) => m.config().vocab_size(),
            Self::Qwen3(m) => m.config().vocab_size(),
            Self::Qwen3MoE(m) => m.config.vocab_size,
            Self::Gemma(m) => m.config().vocab_size(),
            Self::Mistral(m) => m.config().vocab_size(),
            Self::Phi(m) => m.config().vocab_size(),
            Self::Phi4(m) => m.config().vocab_size(),
            Self::DeepSeek(m) => m.config.vocab_size,
            Self::Cohere(m) => m.config.vocab_size,
            Self::Granite(m) => m.config.vocab_size,
            Self::NemotronH(m) => m.config().vocab_size(),
            Self::Qwen3Next(m) => m.config().vocab_size(),
            Self::StarCoder2(m) => m.config.vocab_size,
            Self::RecurrentGemma(m) => m.config.vocab_size,
            Self::Jamba(m) => m.config.vocab_size,
            Self::FalconH1(m) => m.config().vocab_size(),
            Self::Flux(_) => 0,
            Self::Bert(m) => m.config().vocab_size as i32,
        }
    }

    pub fn hidden_size(&self) -> i32 {
        match self {
            Self::Llama(m) => m.model.config.hidden_size,
            Self::Llama4(m) => m.config.hidden_size,
            Self::Qwen2(m) => m.config().hidden_size(),
            Self::Qwen3(m) => m.config().hidden_size(),
            Self::Qwen3MoE(m) => m.config.hidden_size,
            Self::Gemma(m) => m.config().hidden_size(),
            Self::Mistral(m) => m.config().hidden_size(),
            Self::Phi(m) => m.config().hidden_size(),
            Self::Phi4(m) => m.config().hidden_size(),
            Self::DeepSeek(m) => m.config.hidden_size,
            Self::Cohere(m) => m.config.hidden_size,
            Self::Granite(m) => m.config.hidden_size,
            Self::NemotronH(m) => m.config().hidden_size(),
            Self::Qwen3Next(m) => m.config().hidden_size(),
            Self::StarCoder2(m) => m.config.hidden_size,
            Self::RecurrentGemma(m) => m.config.hidden_size,
            Self::Jamba(m) => m.config.hidden_size,
            Self::FalconH1(m) => m.config().hidden_size(),
            Self::Flux(m) => m.pos_embedder.dim as i32,
            Self::Bert(m) => m.config().hidden_size as i32,
        }
    }

    pub fn eval(&self) -> Result<(), Exception> {
        match self {
            Self::Llama(m) => eval_module_parameters_batched(m),
            Self::Llama4(m) => eval_module_parameters_batched(m),
            Self::Qwen2(m) => eval_module_parameters_batched(m),
            Self::Qwen3(m) => eval_module_parameters_batched(m),
            Self::Qwen3MoE(m) => eval_module_parameters_batched(m),
            Self::Gemma(m) => eval_module_parameters_batched(m),
            Self::Mistral(m) => eval_module_parameters_batched(m),
            Self::Phi(m) => eval_module_parameters_batched(m),
            Self::Phi4(m) => eval_module_parameters_batched(m),
            Self::DeepSeek(m) => eval_module_parameters_batched(m),
            Self::Cohere(m) => eval_module_parameters_batched(m),
            Self::Granite(m) => eval_module_parameters_batched(m),
            Self::NemotronH(m) => eval_module_parameters_batched(m),
            Self::Qwen3Next(m) => eval_module_parameters_batched(m),
            Self::StarCoder2(m) => eval_module_parameters_batched(m),
            Self::RecurrentGemma(m) => eval_module_parameters_batched(m),
            Self::Jamba(m) => eval_module_parameters_batched(m),
            Self::FalconH1(m) => eval_module_parameters_batched(m),
            Self::Flux(m) => eval_module_parameters_batched(m),
            Self::Bert(m) => eval_module_parameters_batched(m),
        }
    }

    /// Enable SSD-offloaded MoE inference with expert prefetching.
    ///
    /// Only supported for architectures with MoE (currently Qwen3Next).
    /// The `experts_dir` should contain packed expert files from `pmetal pack-experts`.
    pub fn enable_expert_offloading(&mut self, experts_dir: &Path) -> Result<(), Exception> {
        match self {
            Self::Qwen3Next(m) => m.enable_expert_offloading(experts_dir),
            _ => Err(Exception::custom(
                "expert offloading is only supported for qwen3_next architecture",
            )),
        }
    }

    pub fn requires_expert_offloading(&self) -> bool {
        match self {
            Self::Qwen3Next(m) => m.requires_expert_offloading(),
            _ => false,
        }
    }

    /// Get prefetch hit/miss statistics (if expert offloading is enabled).
    pub fn prefetch_stats(&self) -> Option<crate::expert_prefetch::PrefetchStats> {
        match self {
            Self::Qwen3Next(m) => m.prefetch_stats(),
            _ => None,
        }
    }

    /// Reset prefetch hit/miss statistics (if expert offloading is enabled).
    pub fn reset_prefetch_stats(&self) {
        if let Self::Qwen3Next(m) = self {
            m.reset_prefetch_stats();
        }
    }
}

impl ModuleParameters for DynamicModel {
    fn parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        dispatch_uniform!(self, parameters)
    }

    fn trainable_parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        dispatch_uniform!(self, trainable_parameters)
    }

    fn parameters_mut(&mut self) -> mlx_rs::module::ModuleParamMut<'_> {
        dispatch_uniform!(self, parameters_mut)
    }

    fn num_parameters(&self) -> usize {
        dispatch_uniform!(self, num_parameters)
    }

    fn freeze_parameters(&mut self, recurse: bool) {
        dispatch_uniform!(self, freeze_parameters, recurse)
    }

    fn unfreeze_parameters(&mut self, recurse: bool) {
        dispatch_uniform!(self, unfreeze_parameters, recurse)
    }

    fn all_frozen(&self) -> Option<bool> {
        dispatch_uniform!(self, all_frozen)
    }

    fn any_frozen(&self) -> Option<bool> {
        dispatch_uniform!(self, any_frozen)
    }
}

impl Module<Array> for DynamicModel {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Array) -> Result<Self::Output, Self::Error> {
        self.forward(&input, None)
    }

    fn training_mode(&mut self, _mode: bool) {
        // No-op for now as most models don't implement Module trait yet
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn tiny_qwen3_moe_config() -> Qwen3MoEConfig {
        Qwen3MoEConfig {
            hidden_size: 32,
            intermediate_size: 64,
            moe_intermediate_size: Some(32),
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: 16,
            vocab_size: 100,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    fn tiny_deepseek_config() -> DeepSeekConfig {
        DeepSeekConfig {
            hidden_size: 16,
            intermediate_size: 32,
            moe_intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(4),
            n_shared_experts: Some(1),
            n_routed_experts: Some(4),
            num_experts_per_tok: 2,
            moe_layer_freq: 1,
            first_k_dense_replace: 0,
            ..DeepSeekConfig::default()
        }
    }

    fn tiny_nemotron_h_config() -> NemotronHConfig {
        NemotronHConfig {
            model_type: "nemotron_h".to_string(),
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 4,
            max_position_embeddings: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            attention_bias: false,
            head_dim: Some(32),
            mamba_num_heads: 4,
            mamba_head_dim: 32,
            mamba_proj_bias: false,
            ssm_state_size: 16,
            conv_kernel: 4,
            n_groups: 2,
            time_step_limit: (0.0, f32::INFINITY),
            time_step_min: None,
            time_step_max: None,
            mlp_bias: false,
            mlp_hidden_act: "relu2".to_string(),
            layer_norm_epsilon: 1e-5,
            use_bias: false,
            use_conv_bias: true,
            tie_word_embeddings: true,
            hybrid_override_pattern: Some("M*-E".to_string()),
            moe_intermediate_size: Some(64),
            moe_shared_expert_intermediate_size: None,
            n_group: None,
            n_routed_experts: Some(2),
            n_shared_experts: None,
            topk_group: None,
            num_experts_per_tok: Some(1),
            norm_topk_prob: None,
            routed_scaling_factor: None,
            rope_theta: 10000.0,
        }
    }

    fn tiny_qwen35_moe_text_config() -> Qwen3NextConfig {
        Qwen3NextConfig {
            model_type: "qwen3_5_moe_text".to_string(),
            vocab_size: 100,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: Some(16),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            linear_num_value_heads: 2,
            linear_num_key_heads: 1,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            moe_intermediate_size: 48,
            shared_expert_intermediate_size: 32,
            mlp_only_layers: Vec::new(),
            norm_topk_prob: true,
            partial_rotary_factor: 0.25,
            attention_bias: false,
            rope_scaling: None,
            rope_parameters: None,
            layer_types: Some(vec![
                "linear_attention".to_string(),
                "linear_attention".to_string(),
            ]),
        }
    }

    #[test]
    fn qwen35_moe_model_type_detects_as_qwen3_next() {
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3_5_moe"),
            Some(ModelArchitecture::Qwen3Next)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3_5_moe_text"),
            Some(ModelArchitecture::Qwen3Next)
        );
    }

    #[test]
    fn qwen35_moe_architecture_string_detects_as_qwen3_next() {
        let architectures = vec!["Qwen3_5_MoeForConditionalGeneration".to_string()];
        assert_eq!(
            ModelArchitecture::from_architectures(&architectures),
            Some(ModelArchitecture::Qwen3Next)
        );
    }

    #[test]
    #[serial]
    fn qwen35_placeholder_model_requires_expert_offloading() {
        let model = DynamicModel::Qwen3Next(
            Qwen3NextForCausalLM::new_with_routed_expert_mode(
                tiny_qwen35_moe_text_config(),
                Qwen3NextRoutedExpertMode::Placeholder,
            )
            .unwrap(),
        );
        assert!(model.requires_expert_offloading());
    }

    #[test]
    #[serial]
    fn qwen3_moe_post_load_fast_paths_initialize_stacked_experts() {
        let mut model = DynamicModel::Qwen3MoE(Qwen3MoE::new(tiny_qwen3_moe_config()).unwrap());
        model.init_post_load_fast_paths().unwrap();

        let DynamicModel::Qwen3MoE(model) = &model else {
            panic!("expected qwen3-moe model");
        };
        let Qwen3MoEFeedForward::MoE(moe) = &model.model.layers[0].ffn else {
            panic!("expected moe layer");
        };
        assert!(moe.has_stacked_moe());
    }

    #[test]
    #[serial]
    fn deepseek_post_load_fast_paths_initialize_stacked_experts() {
        let mut model = DynamicModel::DeepSeek(DeepSeek::new(tiny_deepseek_config()).unwrap());
        model.init_post_load_fast_paths().unwrap();

        let DynamicModel::DeepSeek(model) = &model else {
            panic!("expected deepseek model");
        };
        let DeepSeekMLPType::MoE(moe) = &model.model.layers[0].mlp else {
            panic!("expected moe layer");
        };
        assert!(moe.has_stacked_moe());
    }

    #[test]
    #[serial]
    fn nemotron_h_post_load_fast_paths_initialize_stacked_experts() {
        let mut model =
            DynamicModel::NemotronH(NemotronHForCausalLM::new(tiny_nemotron_h_config()).unwrap());
        model.init_post_load_fast_paths().unwrap();

        let DynamicModel::NemotronH(model) = &model else {
            panic!("expected nemotron-h model");
        };
        let moe_layer = &model.backbone.layers[3].mixer;
        assert!(moe_layer.stacked_moe_up.is_some());
        assert!(moe_layer.stacked_moe_down.is_some());
    }
}
