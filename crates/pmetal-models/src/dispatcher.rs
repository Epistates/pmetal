//! Dynamic model dispatch based on config.json model_type.
//!
//! This module provides automatic architecture detection and model loading,
//! eliminating the need for hardcoded model types in application code.

use std::path::Path;
use mlx_rs::{Array, module::{ModuleParameters, ModuleParametersExt}};
use mlx_rs::error::Exception;
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig, MambaCache};
use crate::architectures::*;
use crate::loader::{load_generic_weights, load_nemotron_weights};

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    StarCoder2,
    RecurrentGemma,
    Jamba,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama => write!(f, "Llama"),
            Self::Llama4 => write!(f, "Llama4"),
            Self::Qwen2 => write!(f, "Qwen2"),
            Self::Qwen3 => write!(f, "Qwen3"),
            Self::Qwen3MoE => write!(f, "Qwen3MoE"),
            Self::Gemma => write!(f, "Gemma"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Phi => write!(f, "Phi"),
            Self::Phi4 => write!(f, "Phi4"),
            Self::DeepSeek => write!(f, "DeepSeek"),
            Self::Cohere => write!(f, "Cohere"),
            Self::Granite => write!(f, "Granite"),
            Self::NemotronH => write!(f, "NemotronH"),
            Self::StarCoder2 => write!(f, "StarCoder2"),
            Self::RecurrentGemma => write!(f, "RecurrentGemma"),
            Self::Jamba => write!(f, "Jamba"),
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
            _ => None,
        }
    }

    pub fn from_architectures(archs: &[String]) -> Option<Self> {
        for arch in archs {
            let lower = arch.to_lowercase();
            if lower.contains("llama4") { return Some(Self::Llama4); }
            if lower.contains("llama") { return Some(Self::Llama); }
            if lower.contains("qwen3moe") || lower.contains("qwen3_moe") { return Some(Self::Qwen3MoE); }
            if lower.contains("qwen3") { return Some(Self::Qwen3); }
            if lower.contains("qwen2") || lower.contains("qwen") { return Some(Self::Qwen2); }
            if lower.contains("gemma") { if lower.contains("recurrent") { return Some(Self::RecurrentGemma); } return Some(Self::Gemma); }
            if lower.contains("mistral") || lower.contains("mixtral") { return Some(Self::Mistral); }
            if lower.contains("phi4") { return Some(Self::Phi4); }
            if lower.contains("phi") { return Some(Self::Phi); }
            if lower.contains("deepseek") { return Some(Self::DeepSeek); }
            if lower.contains("cohere") || lower.contains("commandr") || lower.contains("command_r") { return Some(Self::Cohere); }
            if lower.contains("granite") { return Some(Self::Granite); }
            if lower.contains("nemotron") && lower.contains("h") { return Some(Self::NemotronH); }
            if lower.contains("starcoder2") { return Some(Self::StarCoder2); }
            if lower.contains("jamba") { return Some(Self::Jamba); }
        }
        None
    }

    pub fn detect<P: AsRef<Path>>(model_dir: P) -> Result<Self, Exception> {
        let config_path = model_dir.as_ref().join("config.json");
        let config_content = std::fs::read_to_string(config_path).map_err(|e| Exception::custom(format!("{}", e)))?;
        let base_config: serde_json::Value = serde_json::from_str(&config_content).map_err(|e| Exception::custom(format!("{}", e)))?;
        let architectures = base_config["architectures"].as_array().map(|a| a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect::<Vec<_>>());
        let model_type = base_config["model_type"].as_str().unwrap_or("");
        Self::from_model_type(model_type).or_else(|| architectures.as_ref().and_then(|a| Self::from_architectures(a))).ok_or_else(|| Exception::custom(format!("Unsupported: {}", model_type)))
    }
}

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
    StarCoder2(StarCoder2Model),
    RecurrentGemma(RecurrentGemmaModel),
    Jamba(JambaModel),
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
            Self::StarCoder2(_) => write!(f, "DynamicModel::StarCoder2"),
            Self::RecurrentGemma(_) => write!(f, "DynamicModel::RecurrentGemma"),
            Self::Jamba(_) => write!(f, "DynamicModel::Jamba"),
        }
    }
}

impl DynamicModel {
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self, Exception> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(config_path).map_err(|e| Exception::custom(format!("Failed to read config: {}", e)))?;
        let base_config: serde_json::Value = serde_json::from_str(&config_content).map_err(|e| Exception::custom(format!("{}", e)))?;
        let architectures = base_config["architectures"].as_array().map(|a| a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect::<Vec<_>>());
        let model_type = base_config["model_type"].as_str().unwrap_or("");
        let arch = ModelArchitecture::from_model_type(model_type).or_else(|| architectures.as_ref().and_then(|a| ModelArchitecture::from_architectures(a))).ok_or_else(|| Exception::custom(format!("Unsupported architecture: {}", model_type)))?;
        
        match arch {
            ModelArchitecture::Llama => { let config: LlamaConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = LlamaForCausalLM::new(config)?; crate::loader::load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Llama(model)) }
            ModelArchitecture::Llama4 => { let config: Llama4TextConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = Llama4ForCausalLM::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Llama4(model)) }
            ModelArchitecture::Qwen2 => { let config: Qwen2Config = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = Qwen2ForCausalLM::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Qwen2(model)) }
            ModelArchitecture::Qwen3 => { let config: Qwen3Config = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = Qwen3ForCausalLM::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Qwen3(model)) }
            ModelArchitecture::Qwen3MoE => { let config: Qwen3MoEConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = Qwen3MoE::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Qwen3MoE(model)) }
            ModelArchitecture::Gemma => { let config: GemmaConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = GemmaForCausalLM::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Gemma(model)) }
            ModelArchitecture::Mistral => { let config: MistralConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = MistralForCausalLM::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Mistral(model)) }
            ModelArchitecture::Phi => { let config: PhiConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = PhiForCausalLM::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Phi(model)) }
            ModelArchitecture::Phi4 => { let config: PhiConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = PhiForCausalLM::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Phi4(model)) }
            ModelArchitecture::DeepSeek => { let config: DeepSeekConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = DeepSeek::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::DeepSeek(model)) }
            ModelArchitecture::Cohere => { let config: CohereConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = CohereForCausalLM::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Cohere(model)) }
            ModelArchitecture::Granite => { let config: GraniteConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = GraniteForCausalLM::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Granite(model)) }
            ModelArchitecture::NemotronH => { let config: NemotronHConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = NemotronHForCausalLM::new(config)?; load_nemotron_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::NemotronH(model)) }
            ModelArchitecture::StarCoder2 => { let config: StarCoder2Config = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = StarCoder2Model::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::StarCoder2(model)) }
            ModelArchitecture::RecurrentGemma => { let config: RecurrentGemmaConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = RecurrentGemmaModel::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::RecurrentGemma(model)) }
            ModelArchitecture::Jamba => { let config: JambaConfig = json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?; let mut model = JambaModel::new(config)?; load_generic_weights(&mut model, model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?; model.eval()?; Ok(Self::Jamba(model)) }
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
            Self::StarCoder2(m) => m.forward(input_ids, mask, None),
            Self::RecurrentGemma(m) => m.forward(input_ids),
            Self::Jamba(m) => m.forward(input_ids),
        }
    }

    pub fn forward_with_cache(&mut self, input_ids: &Array, mask: Option<&Array>, cache: Option<&mut KVCache>) -> Result<Array, Exception> {
        match self {
            Self::Llama(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Qwen2(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Qwen3(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Qwen3MoE(m) => m.forward(input_ids, mask, cache),
            Self::Gemma(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Mistral(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Phi(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Phi4(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::DeepSeek(m) => m.forward(input_ids, mask, cache),
            Self::StarCoder2(m) => m.forward_with_cache(input_ids, mask, cache),
            _ => self.forward(input_ids, mask),
        }
    }

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        match self {
            Self::Llama(m) => m.create_cache(max_seq_len),
            Self::Llama4(m) => KVCache::new(KVCacheConfig::new(m.config.num_hidden_layers as usize, max_seq_len, m.config.num_key_value_heads as usize, m.config.head_dim as usize)),
            Self::Qwen2(m) => KVCache::new(KVCacheConfig::new(m.config().num_hidden_layers as usize, max_seq_len, m.config().num_kv_heads() as usize, m.config().get_head_dim() as usize)),
            Self::Qwen3(m) => KVCache::new(KVCacheConfig::new(m.config().num_hidden_layers as usize, max_seq_len, m.config().num_kv_heads() as usize, m.config().get_head_dim() as usize)),
            Self::Qwen3MoE(m) => KVCache::new(KVCacheConfig::new(m.config.num_hidden_layers as usize, max_seq_len, m.config.num_kv_heads() as usize, m.config.head_dim as usize)),
            Self::Gemma(m) => m.create_cache(max_seq_len),
            Self::Mistral(m) => m.create_cache(max_seq_len),
            Self::Phi(m) => m.create_cache(max_seq_len),
            Self::Phi4(m) => m.create_cache(max_seq_len),
            Self::DeepSeek(m) => m.create_cache(max_seq_len),
            Self::Cohere(m) => KVCache::new(KVCacheConfig::new(m.config.num_hidden_layers as usize, max_seq_len, m.config.num_key_value_heads as usize, m.config.head_dim as usize)),
            Self::Granite(m) => KVCache::new(KVCacheConfig::new(m.config.num_hidden_layers as usize, max_seq_len, m.config.num_key_value_heads as usize, m.config.head_dim as usize)),
            Self::NemotronH(m) => KVCache::new(KVCacheConfig::new(m.config().num_hidden_layers as usize, max_seq_len, m.config().num_key_value_heads as usize, m.config().attention_head_dim() as usize)),
            Self::StarCoder2(m) => KVCache::new(KVCacheConfig::new(m.config.num_hidden_layers as usize, max_seq_len, m.config.num_key_value_heads as usize, (m.config.hidden_size / m.config.num_attention_heads) as usize)),
            _ => KVCache::new(KVCacheConfig::new(0, 0, 0, 0)),
        }
    }

    pub fn create_mamba_cache(&self) -> Option<MambaCache> {
        match self { Self::NemotronH(m) => Some(MambaCache::new(m.config().num_hidden_layers as usize)), _ => None }
    }

    pub fn forward_with_hybrid_cache(&mut self, input_ids: &Array, mask: Option<&Array>, kv_cache: Option<&mut KVCache>, mamba_cache: Option<&mut MambaCache>) -> Result<Array, Exception> {
        match self { Self::NemotronH(m) => m.forward_with_cache(input_ids, mask, kv_cache, mamba_cache), _ => self.forward_with_cache(input_ids, mask, kv_cache) }
    }

    pub fn quantize_fp8(&mut self) -> Result<(), Exception> {
        Ok(())
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
            Self::StarCoder2(_) => ModelArchitecture::StarCoder2,
            Self::RecurrentGemma(_) => ModelArchitecture::RecurrentGemma,
            Self::Jamba(_) => ModelArchitecture::Jamba,
        }
    }

    pub fn vocab_size(&self) -> i32 {
        match self {
            Self::Llama(m) => m.config().vocab_size,
            Self::Llama4(m) => m.config.vocab_size,
            Self::Qwen2(m) => m.config().vocab_size,
            Self::Qwen3(m) => m.config().vocab_size,
            Self::Qwen3MoE(m) => m.config.vocab_size,
            Self::Gemma(m) => m.config().vocab_size,
            Self::Mistral(m) => m.config().vocab_size,
            Self::Phi(m) => m.config().vocab_size,
            Self::Phi4(m) => m.config().vocab_size,
            Self::DeepSeek(m) => m.config.vocab_size,
            Self::Cohere(m) => m.config.vocab_size,
            Self::Granite(m) => m.config.vocab_size,
            Self::NemotronH(m) => m.config().vocab_size,
            Self::StarCoder2(m) => m.config.vocab_size,
            Self::RecurrentGemma(m) => m.config.vocab_size,
            Self::Jamba(m) => m.config.vocab_size,
        }
    }

    pub fn hidden_size(&self) -> i32 {
        match self {
            Self::Llama(m) => m.config().hidden_size,
            Self::Llama4(m) => m.config.hidden_size,
            Self::Qwen2(m) => m.config().hidden_size,
            Self::Qwen3(m) => m.config().hidden_size,
            Self::Qwen3MoE(m) => m.config.hidden_size,
            Self::Gemma(m) => m.config().hidden_size,
            Self::Mistral(m) => m.config().hidden_size,
            Self::Phi(m) => m.config().hidden_size,
            Self::Phi4(m) => m.config().hidden_size,
            Self::DeepSeek(m) => m.config.hidden_size,
            Self::Cohere(m) => m.config.hidden_size,
            Self::Granite(m) => m.config.hidden_size,
            Self::NemotronH(m) => m.config().hidden_size,
            Self::StarCoder2(m) => m.config.hidden_size,
            Self::RecurrentGemma(m) => m.config.hidden_size,
            Self::Jamba(m) => m.config.hidden_size,
        }
    }

    pub fn eval(&self) -> Result<(), Exception> {
        match self {
            Self::Llama(m) => m.eval(),
            Self::Llama4(m) => m.eval(),
            Self::Qwen2(m) => m.eval(),
            Self::Qwen3(m) => m.eval(),
            Self::Qwen3MoE(m) => m.eval(),
            Self::Gemma(m) => m.eval(),
            Self::Mistral(m) => m.eval(),
            Self::Phi(m) => m.eval(),
            Self::Phi4(m) => m.eval(),
            Self::DeepSeek(m) => m.eval(),
            Self::Cohere(m) => m.eval(),
            Self::Granite(m) => m.eval(),
            Self::NemotronH(m) => m.eval(),
            Self::StarCoder2(m) => m.eval(),
            Self::RecurrentGemma(m) => m.eval(),
            Self::Jamba(m) => m.eval(),
        }
    }
}
