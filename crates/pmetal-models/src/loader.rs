//! Model loading utilities for PMetal.
//!
//! Provides functionality to load model weights from safetensor files,
//! with support for HuggingFace model formats and weight name mapping.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::Array;
pub use mlx_rs::module::ModuleParametersExt;

use crate::architectures::gemma::GemmaForCausalLM;
use crate::architectures::llama::{LlamaConfig, LlamaForCausalLM};
use crate::architectures::mistral::MistralForCausalLM;
use crate::architectures::mllama::MllamaForConditionalGeneration;
use crate::architectures::phi::PhiForCausalLM;
use crate::architectures::qwen2::Qwen2ForCausalLM;
use crate::architectures::qwen3::Qwen3ForCausalLM;
use crate::architectures::nemotron_h::{NemotronHForCausalLM, load_nemotron_weights as load_nemotron};

/// Error type for model loading.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("IO error: {0}")] Io(#[from] std::io::Error),
    #[error("Safetensors error: {0}")] Safetensors(String),
    #[error("Missing weight: {0}")] MissingWeight(String),
    #[error("Shape mismatch for {key}: expected {expected:?}, got {actual:?}")] ShapeMismatch { key: String, expected: Vec<i32>, actual: Vec<i32> },
    #[error("JSON error: {0}")] Json(#[from] serde_json::Error),
    #[error("MLX error: {0}")] Mlx(#[from] mlx_rs::error::Exception),
    #[error("MLX IO error: {0}")] MlxIo(#[from] mlx_rs::error::IoError),
}

/// Weight index for sharded models.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightIndex {
    #[serde(default)] pub metadata: HashMap<String, serde_json::Value>,
    pub weight_map: HashMap<String, String>,
}

/// Load model configuration from a directory.
pub fn load_config(model_dir: impl AsRef<Path>) -> Result<LlamaConfig, LoadError> {
    let config_path = model_dir.as_ref().join("config.json");
    let config_content = std::fs::read_to_string(&config_path)?;
    let config: LlamaConfig = serde_json::from_str(&config_content)?;
    Ok(config)
}

/// Load generic weights using safetensors.
pub fn load_generic_weights<M: ModuleParametersExt>(model: &mut M, model_dir: impl AsRef<Path>) -> Result<(), LoadError> {
    let model_dir = model_dir.as_ref();
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        model.load_safetensors(&single_file).map_err(|e| LoadError::Safetensors(e.to_string()))?;
        return Ok(());
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() { return Err(LoadError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "No weights found"))); }
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: WeightIndex = serde_json::from_str(&index_content)?;
    let shard_files: HashSet<&String> = index.weight_map.values().collect();
    for shard_file in shard_files {
        let shard_path = model_dir.join(shard_file);
        model.load_safetensors(&shard_path).map_err(|e| LoadError::Safetensors(e.to_string()))?;
    }
    Ok(())
}

pub fn load_nemotron_weights(model: &mut NemotronHForCausalLM, model_dir: impl AsRef<Path>) -> Result<(), LoadError> {
    let weights = load_weights(model_dir)?;
    load_nemotron(model, &weights).map_err(|e| LoadError::Safetensors(format!("{:?}", e)))
}

/// Load all safetensor weights from a model directory.
pub fn load_weights(model_dir: impl AsRef<Path>) -> Result<HashMap<String, Array>, LoadError> {
    let model_dir = model_dir.as_ref();
    let mut all_weights = HashMap::new();
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() { let weights = Array::load_safetensors(&single_file)?; return Ok(weights); }
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() { return Err(LoadError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "No weights found"))); }
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: WeightIndex = serde_json::from_str(&index_content)?;
    let shard_files: HashSet<&String> = index.weight_map.values().collect();
    for shard_file in shard_files { let shard_path = model_dir.join(shard_file); let shard_weights = Array::load_safetensors(&shard_path)?; all_weights.extend(shard_weights); }
    Ok(all_weights)
}

pub fn load_llama_weights(model: &mut LlamaForCausalLM, weights: &HashMap<String, Array>) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") { model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone()); } else { return Err(LoadError::MissingWeight("model.embed_tokens.weight".into())); }
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");
        load_linear_weight(&mut layer.self_attn.q_proj, weights, &format!("{prefix}.self_attn.q_proj"))?;
        load_linear_weight(&mut layer.self_attn.k_proj, weights, &format!("{prefix}.self_attn.k_proj"))?;
        load_linear_weight(&mut layer.self_attn.v_proj, weights, &format!("{prefix}.self_attn.v_proj"))?;
        load_linear_weight(&mut layer.self_attn.o_proj, weights, &format!("{prefix}.self_attn.o_proj"))?;
        load_linear_weight(&mut layer.mlp.gate_proj, weights, &format!("{prefix}.mlp.gate_proj"))?;
        load_linear_weight(&mut layer.mlp.up_proj, weights, &format!("{prefix}.mlp.up_proj"))?;
        load_linear_weight(&mut layer.mlp.down_proj, weights, &format!("{prefix}.mlp.down_proj"))?;
        load_rms_norm_weight(&mut layer.input_layernorm, weights, &format!("{prefix}.input_layernorm"))?;
        load_rms_norm_weight(&mut layer.post_attention_layernorm, weights, &format!("{prefix}.post_attention_layernorm"))?;
    }
    load_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    if let Some(ref mut lm_head) = model.lm_head { if let Some(w) = weights.get("lm_head.weight") { lm_head.weight = mlx_rs::module::Param::new(w.clone()); } }
    Ok(())
}

pub fn load_mistral_weights(model: &mut MistralForCausalLM, weights: &HashMap<String, Array>) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") { model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone()); } else { return Err(LoadError::MissingWeight("model.embed_tokens.weight".into())); }
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");
        load_linear_weight(&mut layer.self_attn.q_proj, weights, &format!("{prefix}.self_attn.q_proj"))?;
        load_linear_weight(&mut layer.self_attn.k_proj, weights, &format!("{prefix}.self_attn.k_proj"))?;
        load_linear_weight(&mut layer.self_attn.v_proj, weights, &format!("{prefix}.self_attn.v_proj"))?;
        load_linear_weight(&mut layer.self_attn.o_proj, weights, &format!("{prefix}.self_attn.o_proj"))?;
        load_linear_weight(&mut layer.mlp.gate_proj, weights, &format!("{prefix}.mlp.gate_proj"))?;
        load_linear_weight(&mut layer.mlp.up_proj, weights, &format!("{prefix}.mlp.up_proj"))?;
        load_linear_weight(&mut layer.mlp.down_proj, weights, &format!("{prefix}.mlp.down_proj"))?;
        load_rms_norm_weight(&mut layer.input_layernorm, weights, &format!("{prefix}.input_layernorm"))?;
        load_rms_norm_weight(&mut layer.post_attention_layernorm, weights, &format!("{prefix}.post_attention_layernorm"))?;
    }
    load_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    if let Some(ref mut lm_head) = model.lm_head { if let Some(w) = weights.get("lm_head.weight") { lm_head.weight = mlx_rs::module::Param::new(w.clone()); } }
    Ok(())
}

pub fn load_gemma_weights(model: &mut GemmaForCausalLM, weights: &HashMap<String, Array>) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") { model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone()); } else { return Err(LoadError::MissingWeight("model.embed_tokens.weight".into())); }
    if let Some(ref mut layers) = model.model.layers.gemma1 {
        for (i, layer) in layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{i}");
            load_linear_weight(&mut layer.self_attn.q_proj, weights, &format!("{prefix}.self_attn.q_proj"))?;
            load_linear_weight(&mut layer.self_attn.k_proj, weights, &format!("{prefix}.self_attn.k_proj"))?;
            load_linear_weight(&mut layer.self_attn.v_proj, weights, &format!("{prefix}.self_attn.v_proj"))?;
            load_linear_weight(&mut layer.self_attn.o_proj, weights, &format!("{prefix}.self_attn.o_proj"))?;
            load_linear_weight(&mut layer.mlp.gate_proj, weights, &format!("{prefix}.mlp.gate_proj"))?;
            load_linear_weight(&mut layer.mlp.up_proj, weights, &format!("{prefix}.mlp.up_proj"))?;
            load_linear_weight(&mut layer.mlp.down_proj, weights, &format!("{prefix}.mlp.down_proj"))?;
            load_gemma_rms_norm_weight(&mut layer.input_layernorm, weights, &format!("{prefix}.input_layernorm"))?;
            load_gemma_rms_norm_weight(&mut layer.post_attention_layernorm, weights, &format!("{prefix}.post_attention_layernorm"))?;
        }
    }
    load_gemma_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    Ok(())
}

pub fn load_phi_weights(model: &mut PhiForCausalLM, weights: &HashMap<String, Array>) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") { model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone()); } else { return Err(LoadError::MissingWeight("model.embed_tokens.weight".into())); }
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");
        load_linear_weight(&mut layer.self_attn.q_proj, weights, &format!("{prefix}.self_attn.q_proj"))?;
        load_linear_weight(&mut layer.self_attn.k_proj, weights, &format!("{prefix}.self_attn.k_proj"))?;
        load_linear_weight(&mut layer.self_attn.v_proj, weights, &format!("{prefix}.self_attn.v_proj"))?;
        load_linear_weight(&mut layer.self_attn.o_proj, weights, &format!("{prefix}.self_attn.o_proj"))?;
        load_linear_weight(&mut layer.mlp.gate_up_proj, weights, &format!("{prefix}.mlp.gate_up_proj"))?;
        load_linear_weight(&mut layer.mlp.down_proj, weights, &format!("{prefix}.mlp.down_proj"))?;
        load_phi_rms_norm_weight(&mut layer.input_layernorm, weights, &format!("{prefix}.input_layernorm"))?;
        load_phi_rms_norm_weight(&mut layer.post_attention_layernorm, weights, &format!("{prefix}.post_attention_layernorm"))?;
    }
    load_phi_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    load_linear_weight(&mut model.lm_head, weights, "lm_head")?;
    Ok(())
}

fn load_linear_weight(linear: &mut mlx_rs::nn::Linear, weights: &HashMap<String, Array>, prefix: &str) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) { linear.weight = mlx_rs::module::Param::new(w.clone()); } else { return Err(LoadError::MissingWeight(weight_key)); }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) { linear.bias = mlx_rs::module::Param::new(Some(b.clone())); }
    Ok(())
}

fn load_rms_norm_weight(norm: &mut mlx_rs::nn::RmsNorm, weights: &HashMap<String, Array>, prefix: &str) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) { norm.weight = mlx_rs::module::Param::new(w.clone()); } else { return Err(LoadError::MissingWeight(weight_key)); }
    Ok(())
}

fn load_layer_norm_weight(norm: &mut mlx_rs::nn::LayerNorm, weights: &HashMap<String, Array>, prefix: &str) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) { norm.weight = mlx_rs::module::Param::new(Some(w.clone())); } else { return Err(LoadError::MissingWeight(weight_key)); }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) { norm.bias = mlx_rs::module::Param::new(Some(b.clone())); }
    Ok(())
}

fn load_gemma_rms_norm_weight(norm: &mut crate::architectures::gemma::GemmaRmsNorm, weights: &HashMap<String, Array>, prefix: &str) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) { norm.weight = mlx_rs::module::Param::new(w.clone()); } else { return Err(LoadError::MissingWeight(weight_key)); }
    Ok(())
}

fn load_phi_rms_norm_weight(norm: &mut crate::architectures::phi::PhiRMSNorm, weights: &HashMap<String, Array>, prefix: &str) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) { norm.weight = mlx_rs::module::Param::new(w.clone()); } else { return Err(LoadError::MissingWeight(weight_key)); }
    Ok(())
}
