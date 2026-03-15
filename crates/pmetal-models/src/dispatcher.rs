//! Dynamic model dispatch based on config.json model_type.
//!
//! This module provides automatic architecture detection and model loading,
//! eliminating the need for hardcoded model types in application code.

use crate::architectures::*;
use crate::loader::{load_generic_weights, load_nemotron_weights, load_qwen3_next_weights};
use crate::traits::{CausalLMModel, ModelConfig};
use mlx_rs::{
    Array,
    error::Exception,
    module::{Module, ModuleParameters, ModuleParametersExt},
};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig, MambaCache};
use std::path::Path;

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
    Flux,
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
            Self::Flux => write!(f, "Flux"),
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
            "qwen3_next" | "qwen3_5" | "qwen3.5" | "qwen3_5_text" => Some(Self::Qwen3Next),
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
            "flux" | "flux-1" | "flux.1" => Some(Self::Flux),
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
            if lower.contains("flux") {
                return Some(Self::Flux);
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
    Flux(FluxDiT),
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
            Self::Flux(_) => write!(f, "DynamicModel::Flux"),
        }
    }
}

impl DynamicModel {
    /// Load a model from a directory, automatically detecting its architecture.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, Exception> {
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
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Llama(model))
            }
            ModelArchitecture::Llama4 => {
                let config: Llama4TextConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = Llama4ForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Llama4(model))
            }
            ModelArchitecture::Qwen2 => {
                let config: Qwen2Config = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = Qwen2ForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Qwen2(model))
            }
            ModelArchitecture::Qwen3 => {
                let config: Qwen3Config = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = Qwen3ForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Qwen3(model))
            }
            ModelArchitecture::Qwen3MoE => {
                let config: Qwen3MoEConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = Qwen3MoE::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Qwen3MoE(model))
            }
            ModelArchitecture::Gemma => {
                let config: GemmaConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = GemmaForCausalLM::new(config)?;
                let weights = crate::loader::load_weights(model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                crate::loader::load_gemma_weights(&mut model, &weights)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Gemma(model))
            }
            ModelArchitecture::Mistral => {
                let config: MistralConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = MistralForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Mistral(model))
            }
            ModelArchitecture::Phi => {
                let config: PhiConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = PhiForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Phi(model))
            }
            ModelArchitecture::Phi4 => {
                let config: PhiConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = PhiForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Phi4(model))
            }
            ModelArchitecture::DeepSeek => {
                let config: DeepSeekConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = DeepSeek::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::DeepSeek(model))
            }
            ModelArchitecture::Cohere => {
                let config: CohereConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = CohereForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Cohere(model))
            }
            ModelArchitecture::Granite => {
                let config: GraniteConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = GraniteForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Granite(model))
            }
            ModelArchitecture::NemotronH => {
                let config: NemotronHConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = NemotronHForCausalLM::new(config)?;
                crate::loader::load_nemotron_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::NemotronH(model))
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
                let mut model = Qwen3NextForCausalLM::new(config.clone())?;
                load_qwen3_next_weights(&mut model, model_dir, &config)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Qwen3Next(model))
            }
            ModelArchitecture::StarCoder2 => {
                let config: StarCoder2Config = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = StarCoder2Model::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::StarCoder2(model))
            }
            ModelArchitecture::RecurrentGemma => {
                let config: RecurrentGemmaConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = RecurrentGemmaModel::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::RecurrentGemma(model))
            }
            ModelArchitecture::Jamba => {
                let config: JambaConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = JambaModel::new(config)?;
                load_generic_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                ModuleParametersExt::eval(&model)?;
                Ok(Self::Jamba(model))
            }
            ModelArchitecture::Flux => {
                Err(Exception::custom(
                    "Flux models are diffusion pipelines, not causal language models. Load them via pmetal_models::pipelines::FluxPipeline instead of DynamicModel::load.",
                ))
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
            Self::Flux(_) => Err(Exception::custom(
                "Flux is not a CausalLM and does not support standard forward(input_ids, mask)",
            )),
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
            Self::DeepSeek(_)
            | Self::Cohere(_)
            | Self::Granite(_)
            | Self::NemotronH(_)
            | Self::Qwen3Next(_)
            | Self::StarCoder2(_)
            | Self::Llama4(_)
            | Self::RecurrentGemma(_)
            | Self::Jamba(_)
            | Self::Flux(_) => Err(Exception::custom(
                "Architecture does not support KV caching yet",
            )),
            Self::Gemma(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Mistral(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Phi(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Phi4(m) => m.forward_with_cache(input_ids, mask, cache),
        }
    }

    pub fn quantize_fp8(&mut self) -> Result<(), Exception> {
        match self {
            Self::NemotronH(model) => model.quantize_fp8_weights(),
            Self::Flux(_) => Err(Exception::custom(
                "Flux FP8 quantization is not exposed through DynamicModel. Load the diffusion pipeline via pmetal_models::pipelines::FluxPipeline and quantize its components explicitly.",
            )),
            _ => Err(Exception::custom(
                "Runtime FP8 quantization is currently implemented for NemotronH only. Other architectures require dedicated FP8-aware kernels or pre-quantized checkpoints.",
            )),
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
            Self::Flux(_) => KVCache::new(KVCacheConfig::new(0, 0, 0, 0)),
        }
    }

    pub fn create_mamba_cache(&self) -> Option<MambaCache> {
        match self {
            Self::NemotronH(m) => Some(MambaCache::new(m.config().num_hidden_layers() as usize)),
            Self::Qwen3Next(m) => Some(MambaCache::new(m.config().num_hidden_layers() as usize)),
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
            _ => self.forward_with_cache(input_ids, mask, kv_cache),
        }
    }

    pub fn architecture(&self) -> ModelArchitecture {
        match self {
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
            Self::Flux(_) => ModelArchitecture::Flux,
        }
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
            Self::Flux(_) => 0,
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
            Self::Flux(m) => m.pos_embedder.dim as i32,
        }
    }

    pub fn eval(&self) -> Result<(), Exception> {
        match self {
            Self::Llama(m) => ModuleParametersExt::eval(m),
            Self::Llama4(m) => ModuleParametersExt::eval(m),
            Self::Qwen2(m) => ModuleParametersExt::eval(m),
            Self::Qwen3(m) => ModuleParametersExt::eval(m),
            Self::Qwen3MoE(m) => ModuleParametersExt::eval(m),
            Self::Gemma(m) => ModuleParametersExt::eval(m),
            Self::Mistral(m) => ModuleParametersExt::eval(m),
            Self::Phi(m) => ModuleParametersExt::eval(m),
            Self::Phi4(m) => ModuleParametersExt::eval(m),
            Self::DeepSeek(m) => ModuleParametersExt::eval(m),
            Self::Cohere(m) => ModuleParametersExt::eval(m),
            Self::Granite(m) => ModuleParametersExt::eval(m),
            Self::NemotronH(m) => ModuleParametersExt::eval(m),
            Self::Qwen3Next(m) => ModuleParametersExt::eval(m),
            Self::StarCoder2(m) => ModuleParametersExt::eval(m),
            Self::RecurrentGemma(m) => ModuleParametersExt::eval(m),
            Self::Jamba(m) => ModuleParametersExt::eval(m),
            Self::Flux(m) => ModuleParametersExt::eval(m),
        }
    }
}

impl ModuleParameters for DynamicModel {
    fn parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            Self::Llama(m) => m.parameters(),
            Self::Llama4(m) => m.parameters(),
            Self::Qwen2(m) => m.parameters(),
            Self::Qwen3(m) => m.parameters(),
            Self::Qwen3MoE(m) => m.parameters(),
            Self::Gemma(m) => m.parameters(),
            Self::Mistral(m) => m.parameters(),
            Self::Phi(m) => m.parameters(),
            Self::Phi4(m) => m.parameters(),
            Self::DeepSeek(m) => m.parameters(),
            Self::Cohere(m) => m.parameters(),
            Self::Granite(m) => m.parameters(),
            Self::NemotronH(m) => m.parameters(),
            Self::Qwen3Next(m) => m.parameters(),
            Self::StarCoder2(m) => m.parameters(),
            Self::RecurrentGemma(m) => m.parameters(),
            Self::Jamba(m) => m.parameters(),
            Self::Flux(m) => m.parameters(),
        }
    }

    fn trainable_parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            Self::Llama(m) => m.trainable_parameters(),
            Self::Llama4(m) => m.trainable_parameters(),
            Self::Qwen2(m) => m.trainable_parameters(),
            Self::Qwen3(m) => m.trainable_parameters(),
            Self::Qwen3MoE(m) => m.trainable_parameters(),
            Self::Gemma(m) => m.trainable_parameters(),
            Self::Mistral(m) => m.trainable_parameters(),
            Self::Phi(m) => m.trainable_parameters(),
            Self::Phi4(m) => m.trainable_parameters(),
            Self::DeepSeek(m) => m.trainable_parameters(),
            Self::Cohere(m) => m.trainable_parameters(),
            Self::Granite(m) => m.trainable_parameters(),
            Self::NemotronH(m) => m.trainable_parameters(),
            Self::Qwen3Next(m) => m.trainable_parameters(),
            Self::StarCoder2(m) => m.trainable_parameters(),
            Self::RecurrentGemma(m) => m.trainable_parameters(),
            Self::Jamba(m) => m.trainable_parameters(),
            Self::Flux(m) => m.trainable_parameters(),
        }
    }

    fn parameters_mut(&mut self) -> mlx_rs::module::ModuleParamMut<'_> {
        match self {
            Self::Llama(m) => m.parameters_mut(),
            Self::Llama4(m) => m.parameters_mut(),
            Self::Qwen2(m) => m.parameters_mut(),
            Self::Qwen3(m) => m.parameters_mut(),
            Self::Qwen3MoE(m) => m.parameters_mut(),
            Self::Gemma(m) => m.parameters_mut(),
            Self::Mistral(m) => m.parameters_mut(),
            Self::Phi(m) => m.parameters_mut(),
            Self::Phi4(m) => m.parameters_mut(),
            Self::DeepSeek(m) => m.parameters_mut(),
            Self::Cohere(m) => m.parameters_mut(),
            Self::Granite(m) => m.parameters_mut(),
            Self::NemotronH(m) => m.parameters_mut(),
            Self::Qwen3Next(m) => m.parameters_mut(),
            Self::StarCoder2(m) => m.parameters_mut(),
            Self::RecurrentGemma(m) => m.parameters_mut(),
            Self::Jamba(m) => m.parameters_mut(),
            Self::Flux(m) => m.parameters_mut(),
        }
    }

    fn num_parameters(&self) -> usize {
        match self {
            Self::Llama(m) => m.num_parameters(),
            Self::Llama4(m) => m.num_parameters(),
            Self::Qwen2(m) => m.num_parameters(),
            Self::Qwen3(m) => m.num_parameters(),
            Self::Qwen3MoE(m) => m.num_parameters(),
            Self::Gemma(m) => m.num_parameters(),
            Self::Mistral(m) => m.num_parameters(),
            Self::Phi(m) => m.num_parameters(),
            Self::Phi4(m) => m.num_parameters(),
            Self::DeepSeek(m) => m.num_parameters(),
            Self::Cohere(m) => m.num_parameters(),
            Self::Granite(m) => m.num_parameters(),
            Self::NemotronH(m) => m.num_parameters(),
            Self::Qwen3Next(m) => m.num_parameters(),
            Self::StarCoder2(m) => m.num_parameters(),
            Self::RecurrentGemma(m) => m.num_parameters(),
            Self::Jamba(m) => m.num_parameters(),
            Self::Flux(m) => m.num_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::Llama(m) => m.freeze_parameters(recurse),
            Self::Llama4(m) => m.freeze_parameters(recurse),
            Self::Qwen2(m) => m.freeze_parameters(recurse),
            Self::Qwen3(m) => m.freeze_parameters(recurse),
            Self::Qwen3MoE(m) => m.freeze_parameters(recurse),
            Self::Gemma(m) => m.freeze_parameters(recurse),
            Self::Mistral(m) => m.freeze_parameters(recurse),
            Self::Phi(m) => m.freeze_parameters(recurse),
            Self::Phi4(m) => m.freeze_parameters(recurse),
            Self::DeepSeek(m) => m.freeze_parameters(recurse),
            Self::Cohere(m) => m.freeze_parameters(recurse),
            Self::Granite(m) => m.freeze_parameters(recurse),
            Self::NemotronH(m) => m.freeze_parameters(recurse),
            Self::Qwen3Next(m) => m.freeze_parameters(recurse),
            Self::StarCoder2(m) => m.freeze_parameters(recurse),
            Self::RecurrentGemma(m) => m.freeze_parameters(recurse),
            Self::Jamba(m) => m.freeze_parameters(recurse),
            Self::Flux(m) => m.freeze_parameters(recurse),
        }
    }

    fn unfreeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::Llama(m) => m.unfreeze_parameters(recurse),
            Self::Llama4(m) => m.unfreeze_parameters(recurse),
            Self::Qwen2(m) => m.unfreeze_parameters(recurse),
            Self::Qwen3(m) => m.unfreeze_parameters(recurse),
            Self::Qwen3MoE(m) => m.unfreeze_parameters(recurse),
            Self::Gemma(m) => m.unfreeze_parameters(recurse),
            Self::Mistral(m) => m.unfreeze_parameters(recurse),
            Self::Phi(m) => m.unfreeze_parameters(recurse),
            Self::Phi4(m) => m.unfreeze_parameters(recurse),
            Self::DeepSeek(m) => m.unfreeze_parameters(recurse),
            Self::Cohere(m) => m.unfreeze_parameters(recurse),
            Self::Granite(m) => m.unfreeze_parameters(recurse),
            Self::NemotronH(m) => m.unfreeze_parameters(recurse),
            Self::Qwen3Next(m) => m.unfreeze_parameters(recurse),
            Self::StarCoder2(m) => m.unfreeze_parameters(recurse),
            Self::RecurrentGemma(m) => m.unfreeze_parameters(recurse),
            Self::Jamba(m) => m.unfreeze_parameters(recurse),
            Self::Flux(m) => m.unfreeze_parameters(recurse),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Llama(m) => m.all_frozen(),
            Self::Llama4(m) => m.all_frozen(),
            Self::Qwen2(m) => m.all_frozen(),
            Self::Qwen3(m) => m.all_frozen(),
            Self::Qwen3MoE(m) => m.all_frozen(),
            Self::Gemma(m) => m.all_frozen(),
            Self::Mistral(m) => m.all_frozen(),
            Self::Phi(m) => m.all_frozen(),
            Self::Phi4(m) => m.all_frozen(),
            Self::DeepSeek(m) => m.all_frozen(),
            Self::Cohere(m) => m.all_frozen(),
            Self::Granite(m) => m.all_frozen(),
            Self::NemotronH(m) => m.all_frozen(),
            Self::Qwen3Next(m) => m.all_frozen(),
            Self::StarCoder2(m) => m.all_frozen(),
            Self::RecurrentGemma(m) => m.all_frozen(),
            Self::Jamba(m) => m.all_frozen(),
            Self::Flux(m) => m.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Llama(m) => m.any_frozen(),
            Self::Llama4(m) => m.any_frozen(),
            Self::Qwen2(m) => m.any_frozen(),
            Self::Qwen3(m) => m.any_frozen(),
            Self::Qwen3MoE(m) => m.any_frozen(),
            Self::Gemma(m) => m.any_frozen(),
            Self::Mistral(m) => m.any_frozen(),
            Self::Phi(m) => m.any_frozen(),
            Self::Phi4(m) => m.any_frozen(),
            Self::DeepSeek(m) => m.any_frozen(),
            Self::Cohere(m) => m.any_frozen(),
            Self::Granite(m) => m.any_frozen(),
            Self::NemotronH(m) => m.any_frozen(),
            Self::Qwen3Next(m) => m.any_frozen(),
            Self::StarCoder2(m) => m.any_frozen(),
            Self::RecurrentGemma(m) => m.any_frozen(),
            Self::Jamba(m) => m.any_frozen(),
            Self::Flux(m) => m.any_frozen(),
        }
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
