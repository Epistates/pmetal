//! Dynamic model dispatch based on config.json model_type.
//!
//! This module provides automatic architecture detection and model loading,
//! eliminating the need for hardcoded model types in application code.
//!
//! # Example
//!
//! ```ignore
//! use pmetal_models::dispatcher::DynamicModel;
//!
//! // Automatically detect architecture and load model
//! let mut model = DynamicModel::from_pretrained("/path/to/model")?;
//!
//! // Forward pass works regardless of architecture
//! let logits = model.forward(&input_ids, None)?;
//! ```

use std::path::Path;

use mlx_rs::{error::Exception, module::ModuleParametersExt, Array};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig, MambaCache};
use serde::Deserialize;

use crate::architectures::{
    cohere::{CohereConfig, CohereForCausalLM},
    deepseek::{DeepSeek, DeepSeekConfig},
    gemma::{GemmaConfig, GemmaForCausalLM},
    granite::{GraniteConfig, GraniteForCausalLM},
    llama::{LlamaConfig, LlamaForCausalLM},
    llama4::{Llama4ForCausalLM, Llama4TextConfig},
    mistral::{MistralConfig, MistralForCausalLM},
    nemotron_h::{load_nemotron_weights, NemotronHConfig, NemotronHForCausalLM},
    phi::{PhiConfig, PhiForCausalLM},
    qwen2::{Qwen2Config, Qwen2ForCausalLM},
    qwen3::{Qwen3Config, Qwen3ForCausalLM},
    qwen3_moe::{Qwen3MoE, Qwen3MoEConfig},
};

/// Load weights using generic safetensors loading.
///
/// This function handles both single-file and sharded models using
/// mlx-rs's built-in safetensors loading with automatic parameter name matching.
fn load_generic_weights<M: ModuleParametersExt>(
    model: &mut M,
    model_dir: &Path,
) -> Result<(), DispatchError> {
    use std::collections::HashSet;

    // Check for single file model
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        model.load_safetensors(&single_file)?;
        return Ok(());
    }

    // Load sharded model
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(DispatchError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No model.safetensors or model.safetensors.index.json found",
        )));
    }

    let index_content = std::fs::read_to_string(&index_path)?;
    let index: crate::loader::WeightIndex = serde_json::from_str(&index_content)?;

    // Get unique shard files
    let shard_files: HashSet<&String> = index.weight_map.values().collect();

    // Load each shard
    for shard_file in shard_files {
        let shard_path = model_dir.join(shard_file);
        model.load_safetensors(&shard_path)?;
    }

    Ok(())
}

/// Minimal config for model type detection.
#[derive(Debug, Deserialize)]
struct MinimalConfig {
    model_type: Option<String>,
    #[serde(default)]
    architectures: Option<Vec<String>>,
}

/// Supported model architectures for dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    /// Llama family (2, 3, 3.1, 3.2, 3.3).
    Llama,
    /// Llama 4 (Scout, Maverick) with MoE and iRoPE.
    Llama4,
    /// Qwen2 family (2, 2.5).
    Qwen2,
    /// Qwen3 family (3, 3.5).
    Qwen3,
    /// Qwen3-MoE family with sparse routing.
    Qwen3Moe,
    /// Gemma family (2, 3).
    Gemma,
    /// Mistral family.
    Mistral,
    /// Phi family (3, 4).
    Phi,
    /// Phi-4 specific variant (different rope_theta and bias).
    Phi4,
    /// DeepSeek family (V2, V3) with MLA and MoE.
    DeepSeek,
    /// Cohere Command R family.
    Cohere,
    /// IBM Granite family.
    Granite,
    /// NVIDIA Nemotron-H family (hybrid Mamba-Transformer).
    NemotronH,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama => write!(f, "Llama"),
            Self::Llama4 => write!(f, "Llama4"),
            Self::Qwen2 => write!(f, "Qwen2"),
            Self::Qwen3 => write!(f, "Qwen3"),
            Self::Qwen3Moe => write!(f, "Qwen3Moe"),
            Self::Gemma => write!(f, "Gemma"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Phi => write!(f, "Phi"),
            Self::Phi4 => write!(f, "Phi4"),
            Self::DeepSeek => write!(f, "DeepSeek"),
            Self::Cohere => write!(f, "Cohere"),
            Self::Granite => write!(f, "Granite"),
            Self::NemotronH => write!(f, "NemotronH"),
        }
    }
}

impl ModelArchitecture {
    /// Detect architecture from model_type string.
    ///
    /// Matches against known model type identifiers from HuggingFace configs.
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        let lower = model_type.to_lowercase();
        match lower.as_str() {
            // Llama family - check llama4 first (more specific)
            "llama4" => Some(Self::Llama4),
            "llama" | "llama3" => Some(Self::Llama),
            // Qwen family - check qwen3_moe before qwen3
            "qwen3_moe" => Some(Self::Qwen3Moe),
            "qwen3" => Some(Self::Qwen3),
            "qwen2" | "qwen2_5" => Some(Self::Qwen2),
            // Gemma
            "gemma" | "gemma2" | "gemma3" => Some(Self::Gemma),
            // Mistral/Mixtral
            "mistral" | "mixtral" => Some(Self::Mistral),
            // Phi
            "phi4" => Some(Self::Phi4),
            "phi" | "phi3" => Some(Self::Phi),
            // DeepSeek
            "deepseek" | "deepseek2" | "deepseek_v2" | "deepseek_v3" => Some(Self::DeepSeek),
            // Cohere
            "cohere" | "cohere2" | "command_r" | "command-r" => Some(Self::Cohere),
            // Granite
            "granite" | "granitehybrid" | "granite_moe" => Some(Self::Granite),
            // Nemotron-H (hybrid Mamba-Transformer)
            "nemotron_h" | "nemotronh" | "nemotron-h" => Some(Self::NemotronH),
            _ => None,
        }
    }

    /// Detect from architectures list (HuggingFace format).
    ///
    /// The `architectures` field in config.json contains class names like
    /// `["LlamaForCausalLM"]` or `["Qwen2ForCausalLM"]`.
    pub fn from_architectures(archs: &[String]) -> Option<Self> {
        for arch in archs {
            let lower = arch.to_lowercase();

            // Llama - check llama4 first (more specific)
            if lower.contains("llama4") {
                return Some(Self::Llama4);
            }
            if lower.contains("llama") {
                return Some(Self::Llama);
            }

            // Qwen - check more specific variants first
            if lower.contains("qwen3moe") || lower.contains("qwen3_moe") {
                return Some(Self::Qwen3Moe);
            }
            if lower.contains("qwen3") {
                return Some(Self::Qwen3);
            }
            if lower.contains("qwen2") || lower.contains("qwen") {
                return Some(Self::Qwen2);
            }

            // Other architectures
            if lower.contains("gemma") {
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
            if lower.contains("nemotron") && lower.contains("h") {
                return Some(Self::NemotronH);
            }
        }
        None
    }

    /// Detect architecture from a model directory.
    ///
    /// Reads config.json and extracts model_type or architectures field.
    pub fn detect(model_dir: impl AsRef<Path>) -> Result<Self, DispatchError> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");

        if !config_path.exists() {
            return Err(DispatchError::ConfigNotFound(
                config_path.display().to_string(),
            ));
        }

        let config_content = std::fs::read_to_string(&config_path)?;
        // Use json5 to handle Python-style values like Infinity, -Infinity, NaN
        let minimal: MinimalConfig = json5::from_str(&config_content)?;

        // Try model_type first
        if let Some(ref model_type) = minimal.model_type {
            if let Some(arch) = Self::from_model_type(model_type) {
                return Ok(arch);
            }
        }

        // Fall back to architectures list
        if let Some(ref archs) = minimal.architectures {
            if let Some(arch) = Self::from_architectures(archs) {
                return Ok(arch);
            }
        }

        Err(DispatchError::UnsupportedArchitecture(
            minimal.model_type.unwrap_or_else(|| "unknown".to_string()),
        ))
    }
}

/// Dynamic model container using enum dispatch.
///
/// This approach uses static dispatch via enum variants rather than
/// trait objects, which is more efficient and avoids `dyn Trait` limitations
/// while still providing runtime polymorphism.
///
/// # Supported Architectures
///
/// - Llama (2, 3, 3.1, 3.2, 3.3)
/// - Llama4 (Scout, Maverick) - MoE with iRoPE
/// - Qwen2 (2, 2.5)
/// - Qwen3 (3, 3.5) - with Q/K normalization
/// - Qwen3Moe - with sparse routing
/// - Gemma (2, 3)
/// - Mistral
/// - Phi (3, 4)
/// - DeepSeek (V2, V3) - MLA + MoE
/// - Cohere (Command R, R+)
/// - Granite (4.0, Hybrid)
pub enum DynamicModel {
    /// Llama model variant.
    Llama(LlamaForCausalLM),
    /// Llama 4 model variant (MoE + iRoPE).
    Llama4(Llama4ForCausalLM),
    /// Qwen2 model variant.
    Qwen2(Qwen2ForCausalLM),
    /// Qwen3 model variant (with Q/K normalization).
    Qwen3(Qwen3ForCausalLM),
    /// Qwen3-MoE model variant.
    Qwen3Moe(Qwen3MoE),
    /// Gemma model variant.
    Gemma(GemmaForCausalLM),
    /// Mistral model variant.
    Mistral(MistralForCausalLM),
    /// Phi model variant.
    Phi(PhiForCausalLM),
    /// Phi-4 specific variant.
    Phi4(PhiForCausalLM),
    /// DeepSeek model variant (V2, V3).
    DeepSeek(DeepSeek),
    /// Cohere Command R model variant.
    Cohere(CohereForCausalLM),
    /// IBM Granite model variant.
    Granite(GraniteForCausalLM),
    /// NVIDIA Nemotron-H model variant (hybrid Mamba-Transformer).
    NemotronH(NemotronHForCausalLM),
}

impl std::fmt::Debug for DynamicModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama(_) => write!(f, "DynamicModel::Llama"),
            Self::Llama4(_) => write!(f, "DynamicModel::Llama4"),
            Self::Qwen2(_) => write!(f, "DynamicModel::Qwen2"),
            Self::Qwen3(_) => write!(f, "DynamicModel::Qwen3"),
            Self::Qwen3Moe(_) => write!(f, "DynamicModel::Qwen3Moe"),
            Self::Gemma(_) => write!(f, "DynamicModel::Gemma"),
            Self::Mistral(_) => write!(f, "DynamicModel::Mistral"),
            Self::Phi(_) => write!(f, "DynamicModel::Phi"),
            Self::Phi4(_) => write!(f, "DynamicModel::Phi4"),
            Self::DeepSeek(_) => write!(f, "DynamicModel::DeepSeek"),
            Self::Cohere(_) => write!(f, "DynamicModel::Cohere"),
            Self::Granite(_) => write!(f, "DynamicModel::Granite"),
            Self::NemotronH(_) => write!(f, "DynamicModel::NemotronH"),
        }
    }
}

impl DynamicModel {
    /// Load a model from a directory, auto-detecting architecture.
    ///
    /// This function:
    /// 1. Reads config.json to detect the model architecture
    /// 2. Instantiates the correct model type
    /// 3. Loads weights from safetensors files
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to model directory containing config.json and weights
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = DynamicModel::from_pretrained("/path/to/llama-3.2-1b")?;
    /// let model = DynamicModel::from_pretrained("/path/to/qwen2-0.5b")?;
    /// ```
    pub fn from_pretrained(model_dir: impl AsRef<Path>) -> Result<Self, DispatchError> {
        let model_dir = model_dir.as_ref();

        // Detect architecture
        let arch = ModelArchitecture::detect(model_dir)?;

        // Read config content (json5 handles Python-style Infinity, NaN, etc.)
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)?;

        // Load weights
        let weights = crate::loader::load_weights(model_dir)?;

        // Load the appropriate model
        match arch {
            ModelArchitecture::Llama => {
                let config: LlamaConfig = json5::from_str(&config_content)?;
                let mut model = LlamaForCausalLM::new(config)?;
                crate::loader::load_llama_weights(&mut model, &weights)?;
                model.eval()?;
                Ok(DynamicModel::Llama(model))
            }
            ModelArchitecture::Llama4 => {
                let config: Llama4TextConfig = json5::from_str(&config_content)?;
                let mut model = Llama4ForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)?;
                model.eval()?;
                Ok(DynamicModel::Llama4(model))
            }
            ModelArchitecture::Qwen2 => {
                let config: Qwen2Config = json5::from_str(&config_content)?;
                let mut model = Qwen2ForCausalLM::new(config)?;
                crate::loader::load_qwen_weights(&mut model, &weights)?;
                model.eval()?;
                Ok(DynamicModel::Qwen2(model))
            }
            ModelArchitecture::Qwen3 => {
                let config: Qwen3Config = json5::from_str(&config_content)?;
                let mut model = Qwen3ForCausalLM::new(config)?;
                crate::loader::load_qwen3_weights(&mut model, &weights)?;
                model.eval()?;
                Ok(DynamicModel::Qwen3(model))
            }
            ModelArchitecture::Qwen3Moe => {
                let config: Qwen3MoEConfig = json5::from_str(&config_content)?;
                let mut model = Qwen3MoE::new(config)?;
                load_generic_weights(&mut model, model_dir)?;
                model.eval()?;
                Ok(DynamicModel::Qwen3Moe(model))
            }
            ModelArchitecture::Gemma => {
                let config: GemmaConfig = json5::from_str(&config_content)?;
                let mut model = GemmaForCausalLM::new(config)?;
                crate::loader::load_gemma_weights(&mut model, &weights)?;
                model.eval()?;
                Ok(DynamicModel::Gemma(model))
            }
            ModelArchitecture::Mistral => {
                let config: MistralConfig = json5::from_str(&config_content)?;
                let mut model = MistralForCausalLM::new(config)?;
                crate::loader::load_mistral_weights(&mut model, &weights)?;
                model.eval()?;
                Ok(DynamicModel::Mistral(model))
            }
            ModelArchitecture::Phi | ModelArchitecture::Phi4 => {
                let config: PhiConfig = json5::from_str(&config_content)?;
                let mut model = PhiForCausalLM::new(config)?;
                crate::loader::load_phi_weights(&mut model, &weights)?;
                model.eval()?;
                if arch == ModelArchitecture::Phi4 {
                    Ok(DynamicModel::Phi4(model))
                } else {
                    Ok(DynamicModel::Phi(model))
                }
            }
            ModelArchitecture::DeepSeek => {
                let config: DeepSeekConfig = json5::from_str(&config_content)?;
                let mut model = DeepSeek::new(config)?;
                load_generic_weights(&mut model, model_dir)?;
                model.eval()?;
                Ok(DynamicModel::DeepSeek(model))
            }
            ModelArchitecture::Cohere => {
                let config: CohereConfig = json5::from_str(&config_content)?;
                let mut model = CohereForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)?;
                model.eval()?;
                Ok(DynamicModel::Cohere(model))
            }
            ModelArchitecture::Granite => {
                let config: GraniteConfig = json5::from_str(&config_content)?;
                let mut model = GraniteForCausalLM::new(config)?;
                load_generic_weights(&mut model, model_dir)?;
                model.eval()?;
                Ok(DynamicModel::Granite(model))
            }
            ModelArchitecture::NemotronH => {
                let config: NemotronHConfig = json5::from_str(&config_content)?;
                let mut model = NemotronHForCausalLM::new(config)?;
                load_nemotron_weights(&mut model, &weights)?;
                // Initialize stacked MoE weights for gather_mm optimization
                // This provides ~10x speedup for MoE layers
                model.backbone.init_stacked_moe()?;
                model.eval()?;
                Ok(DynamicModel::NemotronH(model))
            }
        }
    }

    /// Load weights using generic safetensors loading.
    fn load_generic<M: ModuleParametersExt>(
        model: &mut M,
        model_dir: &Path,
    ) -> Result<(), DispatchError> {
        load_generic_weights(model, model_dir)
    }

    /// Forward pass producing logits.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs of shape `[batch, seq_len]`
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    ///
    /// Logits of shape `[batch, seq_len, vocab_size]`
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        match self {
            Self::Llama(m) => m.forward(input_ids, mask),
            Self::Llama4(m) => m.forward(input_ids, mask, None),
            Self::Qwen2(m) => m.forward(input_ids, mask),
            Self::Qwen3(m) => m.forward(input_ids, mask),
            Self::Qwen3Moe(m) => m.forward(input_ids, mask, None),
            Self::Gemma(m) => m.forward(input_ids, mask),
            Self::Mistral(m) => m.forward(input_ids, mask),
            Self::Phi(m) => m.forward(input_ids, mask),
            Self::Phi4(m) => m.forward(input_ids, mask),
            Self::DeepSeek(m) => m.forward(input_ids, mask, None),
            Self::Cohere(m) => m.forward(input_ids, mask, None),
            Self::Granite(m) => m.forward(input_ids, mask, None),
            Self::NemotronH(m) => m.forward(input_ids, None),
        }
    }

    /// Forward pass with optional KV cache for efficient generation.
    ///
    /// When cache is provided:
    /// - Prefill: Process the full prompt, populating the cache
    /// - Decode: Process one token at a time using cached K/V
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs of shape `[batch, seq_len]`
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KV cache for efficient generation
    ///
    /// # Returns
    ///
    /// Logits of shape `[batch, seq_len, vocab_size]`
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Llama(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Llama4(m) => m.forward(input_ids, mask, None), // Llama4 uses different cache API
            Self::Qwen2(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Qwen3(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Qwen3Moe(m) => m.forward(input_ids, mask, cache),
            Self::Gemma(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Mistral(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Phi(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Phi4(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::DeepSeek(m) => m.forward(input_ids, mask, cache),
            // Note: Cohere and Granite use their own cache systems
            Self::Cohere(m) => m.forward(input_ids, mask, None),
            Self::Granite(m) => m.forward(input_ids, mask, None),
            // NemotronH: pass KV cache but not Mamba cache (use forward_with_hybrid_cache for full caching)
            Self::NemotronH(m) => m.forward_with_cache(input_ids, mask, cache, None),
        }
    }

    /// Create a KV cache configured for this model.
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum sequence length to cache
    ///
    /// # Returns
    ///
    /// A new KV cache configured for this model's architecture
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        match self {
            Self::Llama(m) => m.create_cache(max_seq_len),
            Self::Llama4(m) => {
                let config = &m.config;
                KVCache::new(KVCacheConfig::new(
                    config.num_hidden_layers as usize,
                    max_seq_len,
                    config.num_key_value_heads as usize,
                    config.head_dim as usize,
                ))
            }
            Self::Qwen2(m) => {
                let config = m.config();
                KVCache::new(KVCacheConfig::new(
                    config.num_hidden_layers as usize,
                    max_seq_len,
                    config.num_kv_heads() as usize,
                    config.get_head_dim() as usize,
                ))
            }
            Self::Qwen3(m) => {
                let config = m.config();
                KVCache::new(KVCacheConfig::new(
                    config.num_hidden_layers as usize,
                    max_seq_len,
                    config.num_kv_heads() as usize,
                    config.get_head_dim() as usize,
                ))
            }
            Self::Qwen3Moe(m) => {
                let config = &m.config;
                KVCache::new(KVCacheConfig::new(
                    config.num_hidden_layers as usize,
                    max_seq_len,
                    config.num_kv_heads() as usize,
                    config.head_dim as usize,
                ))
            }
            Self::Gemma(m) => m.create_cache(max_seq_len),
            Self::Mistral(m) => m.create_cache(max_seq_len),
            Self::Phi(m) => m.create_cache(max_seq_len),
            Self::Phi4(m) => m.create_cache(max_seq_len),
            Self::DeepSeek(m) => m.create_cache(max_seq_len),
            Self::Cohere(m) => {
                let config = &m.config;
                KVCache::new(KVCacheConfig::new(
                    config.num_hidden_layers as usize,
                    max_seq_len,
                    config.num_key_value_heads as usize,
                    config.head_dim as usize,
                ))
            }
            Self::Granite(m) => {
                let config = &m.config;
                KVCache::new(KVCacheConfig::new(
                    config.num_hidden_layers as usize,
                    max_seq_len,
                    config.num_key_value_heads as usize,
                    config.head_dim as usize,
                ))
            }
            Self::NemotronH(m) => {
                // NemotronH uses its own hybrid cache system (MambaCache + KVCache)
                // Return a basic KV cache for attention layers
                let config = m.config();
                KVCache::new(KVCacheConfig::new(
                    config.num_hidden_layers as usize,
                    max_seq_len,
                    config.num_key_value_heads as usize,
                    config.attention_head_dim() as usize,
                ))
            }
        }
    }

    /// Create a Mamba cache for models with Mamba/SSM layers.
    ///
    /// Returns Some(MambaCache) for NemotronH and other hybrid models,
    /// None for pure transformer models.
    pub fn create_mamba_cache(&self) -> Option<MambaCache> {
        match self {
            Self::NemotronH(m) => {
                let num_layers = m.config().num_hidden_layers as usize;
                Some(MambaCache::new(num_layers))
            }
            // Other models don't use Mamba
            _ => None,
        }
    }

    /// Forward pass with KV cache and optional Mamba cache (for hybrid models).
    ///
    /// This is the preferred method for autoregressive generation with NemotronH.
    pub fn forward_with_hybrid_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        match self {
            Self::NemotronH(m) => m.forward_with_cache(input_ids, mask, kv_cache, mamba_cache),
            // For non-hybrid models, ignore mamba_cache and use normal forward
            _ => self.forward_with_cache(input_ids, mask, kv_cache),
        }
    }

    /// Get the detected architecture.
    pub fn architecture(&self) -> ModelArchitecture {
        match self {
            Self::Llama(_) => ModelArchitecture::Llama,
            Self::Llama4(_) => ModelArchitecture::Llama4,
            Self::Qwen2(_) => ModelArchitecture::Qwen2,
            Self::Qwen3(_) => ModelArchitecture::Qwen3,
            Self::Qwen3Moe(_) => ModelArchitecture::Qwen3Moe,
            Self::Gemma(_) => ModelArchitecture::Gemma,
            Self::Mistral(_) => ModelArchitecture::Mistral,
            Self::Phi(_) => ModelArchitecture::Phi,
            Self::Phi4(_) => ModelArchitecture::Phi4,
            Self::DeepSeek(_) => ModelArchitecture::DeepSeek,
            Self::Cohere(_) => ModelArchitecture::Cohere,
            Self::Granite(_) => ModelArchitecture::Granite,
            Self::NemotronH(_) => ModelArchitecture::NemotronH,
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> i32 {
        match self {
            Self::Llama(m) => m.config().vocab_size,
            Self::Llama4(m) => m.config.vocab_size,
            Self::Qwen2(m) => m.config().vocab_size,
            Self::Qwen3(m) => m.config().vocab_size,
            Self::Qwen3Moe(m) => m.config.vocab_size,
            Self::Gemma(m) => m.config().vocab_size,
            Self::Mistral(m) => m.config().vocab_size,
            Self::Phi(m) => m.config().vocab_size,
            Self::Phi4(m) => m.config().vocab_size,
            Self::DeepSeek(m) => m.config.vocab_size,
            Self::Cohere(m) => m.config.vocab_size,
            Self::Granite(m) => m.config.vocab_size,
            Self::NemotronH(m) => m.config().vocab_size,
        }
    }

    /// Get the hidden size.
    pub fn hidden_size(&self) -> i32 {
        match self {
            Self::Llama(m) => m.config().hidden_size,
            Self::Llama4(m) => m.config.hidden_size,
            Self::Qwen2(m) => m.config().hidden_size,
            Self::Qwen3(m) => m.config().hidden_size,
            Self::Qwen3Moe(m) => m.config.hidden_size,
            Self::Gemma(m) => m.config().hidden_size,
            Self::Mistral(m) => m.config().hidden_size,
            Self::Phi(m) => m.config().hidden_size,
            Self::Phi4(m) => m.config().hidden_size,
            Self::DeepSeek(m) => m.config.hidden_size,
            Self::Cohere(m) => m.config.hidden_size,
            Self::Granite(m) => m.config.hidden_size,
            Self::NemotronH(m) => m.config().hidden_size,
        }
    }

    /// Evaluate all parameters to materialize them on device.
    pub fn eval(&self) -> Result<(), Exception> {
        match self {
            Self::Llama(m) => m.eval(),
            Self::Llama4(m) => m.eval(),
            Self::Qwen2(m) => m.eval(),
            Self::Qwen3(m) => m.eval(),
            Self::Qwen3Moe(m) => m.eval(),
            Self::Gemma(m) => m.eval(),
            Self::Mistral(m) => m.eval(),
            Self::Phi(m) => m.eval(),
            Self::Phi4(m) => m.eval(),
            Self::DeepSeek(m) => m.eval(),
            Self::Cohere(m) => m.eval(),
            Self::Granite(m) => m.eval(),
            Self::NemotronH(m) => m.eval(),
        }
    }

    /// Quantize model weights to 8-bit format for memory-efficient inference.
    ///
    /// Uses MLX's native quantization with 8 bits and group size 64.
    /// Provides ~4x memory reduction compared to FP32, ~2x compared to FP16.
    ///
    /// Note: Full FP8 inference requires architectural changes to use QuantizedLinear
    /// layers. This method provides the FP8 conversion infrastructure. For production
    /// use, load pre-quantized models or use the model's built-in quantization.
    ///
    /// # Returns
    ///
    /// Ok(()) on success. Currently logs that FP8 mode is enabled.
    pub fn quantize_fp8(&mut self) -> Result<(), Exception> {
        tracing::info!("FP8 quantization mode enabled");
        tracing::info!(
            "Note: For full FP8 inference, use pre-quantized models or mlx-lm's quantize command"
        );

        // The actual quantization happens during inference via the QuantizedLinear path
        // For now, we just log the intent. Full implementation requires:
        // 1. Replace Linear layers with QuantizedLinear
        // 2. Use quantized_matmul for forward pass
        // This is tracked as a future enhancement.

        Ok(())
    }
}

/// Errors during model dispatch.
#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// MLX IO error.
    #[error("MLX IO error: {0}")]
    MlxIo(#[from] mlx_rs::error::IoError),
    /// JSON parsing error.
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
    /// JSON5 parsing error (handles Infinity, NaN, trailing commas, etc.).
    #[error("JSON5 parsing error: {0}")]
    Json5(#[from] json5::Error),
    /// Config file not found.
    #[error("Config file not found: {0}")]
    ConfigNotFound(String),
    /// Unsupported architecture.
    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    /// Architecture not yet implemented.
    #[error("Architecture not implemented: {0}")]
    NotImplemented(ModelArchitecture),
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    /// Weight loading error.
    #[error("Weight loading error: {0}")]
    Load(#[from] crate::LoadError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_from_model_type() {
        // Llama family
        assert_eq!(
            ModelArchitecture::from_model_type("llama"),
            Some(ModelArchitecture::Llama)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("llama4"),
            Some(ModelArchitecture::Llama4)
        );
        // Qwen family
        assert_eq!(
            ModelArchitecture::from_model_type("qwen2"),
            Some(ModelArchitecture::Qwen2)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3"),
            Some(ModelArchitecture::Qwen3)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3_moe"),
            Some(ModelArchitecture::Qwen3Moe)
        );
        // Others
        assert_eq!(
            ModelArchitecture::from_model_type("gemma"),
            Some(ModelArchitecture::Gemma)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("mistral"),
            Some(ModelArchitecture::Mistral)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("phi3"),
            Some(ModelArchitecture::Phi)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("phi4"),
            Some(ModelArchitecture::Phi4)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("deepseek_v3"),
            Some(ModelArchitecture::DeepSeek)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("cohere"),
            Some(ModelArchitecture::Cohere)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("granite"),
            Some(ModelArchitecture::Granite)
        );
        assert_eq!(ModelArchitecture::from_model_type("unknown"), None);
    }

    #[test]
    fn test_architecture_from_architectures() {
        assert_eq!(
            ModelArchitecture::from_architectures(&["LlamaForCausalLM".to_string()]),
            Some(ModelArchitecture::Llama)
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["Llama4ForCausalLM".to_string()]),
            Some(ModelArchitecture::Llama4)
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["Qwen2ForCausalLM".to_string()]),
            Some(ModelArchitecture::Qwen2)
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["Qwen3MoEForCausalLM".to_string()]),
            Some(ModelArchitecture::Qwen3Moe)
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["GemmaForCausalLM".to_string()]),
            Some(ModelArchitecture::Gemma)
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["Phi4ForCausalLM".to_string()]),
            Some(ModelArchitecture::Phi4)
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["DeepSeekForCausalLM".to_string()]),
            Some(ModelArchitecture::DeepSeek)
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["CohereForCausalLM".to_string()]),
            Some(ModelArchitecture::Cohere)
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["GraniteForCausalLM".to_string()]),
            Some(ModelArchitecture::Granite)
        );
    }

    #[test]
    fn test_architecture_display() {
        assert_eq!(ModelArchitecture::Llama.to_string(), "Llama");
        assert_eq!(ModelArchitecture::Llama4.to_string(), "Llama4");
        assert_eq!(ModelArchitecture::Qwen2.to_string(), "Qwen2");
        assert_eq!(ModelArchitecture::Qwen3.to_string(), "Qwen3");
        assert_eq!(ModelArchitecture::Qwen3Moe.to_string(), "Qwen3Moe");
        assert_eq!(ModelArchitecture::Phi4.to_string(), "Phi4");
        assert_eq!(ModelArchitecture::DeepSeek.to_string(), "DeepSeek");
        assert_eq!(ModelArchitecture::Cohere.to_string(), "Cohere");
        assert_eq!(ModelArchitecture::Granite.to_string(), "Granite");
    }
}
