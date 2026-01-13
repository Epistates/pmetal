//! Model loading utilities for PMetal.
//!
//! Provides functionality to load model weights from safetensor files,
//! with support for HuggingFace model formats and weight name mapping.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::module::ModuleParametersExt;
use mlx_rs::Array;

use crate::architectures::gemma::GemmaForCausalLM;
use crate::architectures::llama::{LlamaConfig, LlamaForCausalLM};
use crate::architectures::mistral::MistralForCausalLM;
use crate::architectures::mllama::MllamaForConditionalGeneration;
use crate::architectures::phi::PhiForCausalLM;
use crate::architectures::qwen2::Qwen2ForCausalLM;
use crate::architectures::qwen3::Qwen3ForCausalLM;

/// Error type for model loading.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Safetensors loading error.
    #[error("Safetensors error: {0}")]
    Safetensors(String),
    /// Missing weight.
    #[error("Missing weight: {0}")]
    MissingWeight(String),
    /// Shape mismatch.
    #[error("Shape mismatch for {key}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        key: String,
        expected: Vec<i32>,
        actual: Vec<i32>,
    },
    /// JSON parsing error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    /// IO error from mlx.
    #[error("MLX IO error: {0}")]
    MlxIo(#[from] mlx_rs::error::IoError),
}

/// Weight index for sharded models.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightIndex {
    /// Metadata about the weights.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Mapping from weight name to shard filename.
    pub weight_map: HashMap<String, String>,
}

/// Load model configuration from a directory.
pub fn load_config(model_dir: impl AsRef<Path>) -> Result<LlamaConfig, LoadError> {
    let config_path = model_dir.as_ref().join("config.json");
    let config_content = std::fs::read_to_string(&config_path)?;
    let config: LlamaConfig = serde_json::from_str(&config_content)?;
    Ok(config)
}

/// Load all safetensor weights from a model directory.
///
/// Handles both single-file and sharded models.
pub fn load_weights(model_dir: impl AsRef<Path>) -> Result<HashMap<String, Array>, LoadError> {
    let model_dir = model_dir.as_ref();
    let mut all_weights = HashMap::new();

    // Check for single file model
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        let weights = Array::load_safetensors(&single_file)?;
        return Ok(weights);
    }

    // Load sharded model
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No model.safetensors or model.safetensors.index.json found",
        )));
    }

    let index_content = std::fs::read_to_string(&index_path)?;
    let index: WeightIndex = serde_json::from_str(&index_content)?;

    // Get unique shard files
    let shard_files: HashSet<&String> = index.weight_map.values().collect();

    // Load each shard
    for shard_file in shard_files {
        let shard_path = model_dir.join(shard_file);
        let shard_weights = Array::load_safetensors(&shard_path)?;
        all_weights.extend(shard_weights);
    }

    Ok(all_weights)
}

/// Load weights into a Llama model.
///
/// This function loads weights from safetensor files and assigns them
/// to the corresponding model parameters.
pub fn load_llama_weights(
    model: &mut LlamaForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    // Load embed_tokens
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }

    // Load transformer layers
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");

        // Self-attention projections
        load_linear_weight(
            &mut layer.self_attn.q_proj,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.k_proj,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.v_proj,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.o_proj,
            weights,
            &format!("{prefix}.self_attn.o_proj"),
        )?;

        // MLP projections
        load_linear_weight(
            &mut layer.mlp.gate_proj,
            weights,
            &format!("{prefix}.mlp.gate_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.up_proj,
            weights,
            &format!("{prefix}.mlp.up_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.down_proj,
            weights,
            &format!("{prefix}.mlp.down_proj"),
        )?;

        // Layer norms
        load_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }

    // Load final norm
    load_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;

    // Load lm_head if not tied
    if let Some(ref mut lm_head) = model.lm_head {
        if let Some(w) = weights.get("lm_head.weight") {
            lm_head.weight = mlx_rs::module::Param::new(w.clone());
        }
        // If lm_head.weight is missing, it might be tied to embed_tokens
    }

    Ok(())
}

/// Load weights into a Mllama model.
pub fn load_mllama_weights(
    model: &mut MllamaForConditionalGeneration,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    // 1. Vision Model
    let v_prefix = "vision_model";

    // Embeddings
    if let Some(w) = weights.get(&format!("{v_prefix}.embeddings.patch_embedding.weight")) {
        model.vision_model.embeddings.patch_embedding.weight =
            mlx_rs::module::Param::new(w.clone());
    }
    if let Some(w) = weights.get(&format!("{v_prefix}.embeddings.class_embedding")) {
        model.vision_model.embeddings.class_embedding.weight =
            mlx_rs::module::Param::new(w.clone());
    }
    if let Some(w) = weights.get(&format!("{v_prefix}.embeddings.position_embedding.weight")) {
        model.vision_model.embeddings.position_embedding.weight =
            mlx_rs::module::Param::new(w.clone());
    }
    // Note: Tile embeddings skipped for brevity, add if needed

    // Vision Layers
    for (i, layer) in model.vision_model.layers.iter_mut().enumerate() {
        let prefix = format!("{v_prefix}.encoder.layers.{i}");

        // Self Attention
        load_linear_weight(
            &mut layer.self_attn.q_proj,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.k_proj,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.v_proj,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.o_proj,
            weights,
            &format!("{prefix}.self_attn.o_proj"),
        )?;

        // Gate (Attn)
        if let Some(w) = weights.get(&format!("{prefix}.self_attn.gate")) {
            let gate = crate::architectures::mllama::Gate {
                weight: mlx_rs::module::Param::new(w.clone()),
            };
            layer.gate_attn = vec![gate];
        }

        // MLP
        load_linear_weight(&mut layer.mlp.fc1, weights, &format!("{prefix}.mlp.fc1"))?;
        load_linear_weight(&mut layer.mlp.fc2, weights, &format!("{prefix}.mlp.fc2"))?;

        // Gate (MLP)
        if let Some(w) = weights.get(&format!("{prefix}.mlp.gate")) {
            let gate = crate::architectures::mllama::Gate {
                weight: mlx_rs::module::Param::new(w.clone()),
            };
            layer.gate_mlp = vec![gate];
        }

        // Layer norms
        load_layer_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_layer_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }

    // Vision Final Norm
    load_layer_norm_weight(
        &mut model.vision_model.layernorm,
        weights,
        &format!("{v_prefix}.encoder.layernorm"),
    )?;

    // 2. Projector
    let p_prefix = "multi_modal_projector";
    load_linear_weight(
        &mut model.multi_modal_projector.linear_1,
        weights,
        &format!("{p_prefix}.linear_1"),
    )?;
    load_linear_weight(
        &mut model.multi_modal_projector.linear_2,
        weights,
        &format!("{p_prefix}.linear_2"),
    )?;

    // 3. Language Model
    let l_prefix = "language_model.model";
    // Embed tokens
    if let Some(w) = weights.get(&format!("{l_prefix}.embed_tokens.weight")) {
        model.language_model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone());
    }

    // Layers
    for (i, layer) in model.language_model.layers.iter_mut().enumerate() {
        let prefix = format!("{l_prefix}.layers.{i}");

        // Self Attn
        load_linear_weight(
            &mut layer.self_attn.q_proj,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.k_proj,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.v_proj,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.o_proj,
            weights,
            &format!("{prefix}.self_attn.o_proj"),
        )?;

        // Cross Attn
        if let Some(cross) = &mut layer.cross_attn {
            load_linear_weight(
                &mut cross.q_proj,
                weights,
                &format!("{prefix}.cross_attn.q_proj"),
            )?;
            load_linear_weight(
                &mut cross.k_proj,
                weights,
                &format!("{prefix}.cross_attn.k_proj"),
            )?;
            load_linear_weight(
                &mut cross.v_proj,
                weights,
                &format!("{prefix}.cross_attn.v_proj"),
            )?;
            load_linear_weight(
                &mut cross.o_proj,
                weights,
                &format!("{prefix}.cross_attn.o_proj"),
            )?;

            if let Some(w) = weights.get(&format!("{prefix}.cross_attn.gate")) {
                let gate = crate::architectures::mllama::Gate {
                    weight: mlx_rs::module::Param::new(w.clone()),
                };
                cross.gate = vec![gate];
            }

            // Cross norm
            if let Some(norm) = &mut layer.cross_attention_layernorm {
                load_rms_norm_weight(
                    norm,
                    weights,
                    &format!("{prefix}.cross_attention_layernorm"),
                )?;
            }
        }

        // MLP
        load_linear_weight(
            &mut layer.mlp.gate_proj,
            weights,
            &format!("{prefix}.mlp.gate_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.up_proj,
            weights,
            &format!("{prefix}.mlp.up_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.down_proj,
            weights,
            &format!("{prefix}.mlp.down_proj"),
        )?;

        // Norms
        load_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }

    // LM Norm
    load_rms_norm_weight(
        &mut model.language_model.norm,
        weights,
        &format!("{l_prefix}.norm"),
    )?;

    // LM Head
    // Note: Usually "language_model.lm_head"
    load_linear_weight(&mut model.lm_head, weights, "language_model.lm_head")?;

    Ok(())
}

/// Load weight into a Linear layer.
fn load_linear_weight(
    linear: &mut mlx_rs::nn::Linear,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        linear.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }

    // Check for bias (Llama typically doesn't have bias)
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        linear.bias = mlx_rs::module::Param::new(Some(b.clone()));
    }

    Ok(())
}

/// Load weight into an RmsNorm layer.
fn load_rms_norm_weight(
    norm: &mut mlx_rs::nn::RmsNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    Ok(())
}

/// Load weight into a LayerNorm layer.
fn load_layer_norm_weight(
    norm: &mut mlx_rs::nn::LayerNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = mlx_rs::module::Param::new(Some(w.clone()));
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }

    // LayerNorm usually has bias
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        norm.bias = mlx_rs::module::Param::new(Some(b.clone()));
    }

    Ok(())
}

/// Load a complete Llama model from a directory.
///
/// This is the main entry point for loading pre-trained models.
///
/// # Example
/// ```ignore
/// let model = load_llama_model("/path/to/model")?;
/// ```
pub fn load_llama_model(model_dir: impl AsRef<Path>) -> Result<LlamaForCausalLM, LoadError> {
    let model_dir = model_dir.as_ref();

    // Load configuration
    let config = load_config(model_dir)?;

    // Create model with random initialization
    let mut model = LlamaForCausalLM::new(config)?;

    // Load weights
    let weights = load_weights(model_dir)?;
    load_llama_weights(&mut model, &weights)?;

    // Evaluate to materialize weights on device
    model.eval()?;

    Ok(model)
}

/// Load a Llama model using the ModuleParametersExt trait.
///
/// This is an alternative loading method that uses mlx-rs's built-in
/// safetensors loading with automatic parameter name matching.
///
/// # Example
/// ```ignore
/// let model = load_llama_model_auto("/path/to/model")?;
/// ```
pub fn load_llama_model_auto(model_dir: impl AsRef<Path>) -> Result<LlamaForCausalLM, LoadError> {
    let model_dir = model_dir.as_ref();

    // Load configuration
    let config = load_config(model_dir)?;

    // Create model with random initialization
    let mut model = LlamaForCausalLM::new(config)?;

    // Check for single file model
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        model.load_safetensors(&single_file)?;
        return Ok(model);
    }

    // Load sharded model
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No model.safetensors or model.safetensors.index.json found",
        )));
    }

    let index_content = std::fs::read_to_string(&index_path)?;
    let index: WeightIndex = serde_json::from_str(&index_content)?;

    // Get unique shard files
    let shard_files: HashSet<&String> = index.weight_map.values().collect();

    // Load each shard
    for shard_file in shard_files {
        let shard_path = model_dir.join(shard_file);
        model.load_safetensors(&shard_path)?;
    }

    Ok(model)
}

/// Load weights into a Qwen2 model.
///
/// Qwen2 uses the same structure as Llama but with attention bias.
pub fn load_qwen_weights(
    model: &mut Qwen2ForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    // Load embed_tokens
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }

    // Load transformer layers
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");

        // Self-attention projections (Qwen2 has bias on Q/K/V)
        load_linear_weight(
            &mut layer.self_attn.q_proj,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.k_proj,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.v_proj,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.o_proj,
            weights,
            &format!("{prefix}.self_attn.o_proj"),
        )?;

        // MLP projections
        load_linear_weight(
            &mut layer.mlp.gate_proj,
            weights,
            &format!("{prefix}.mlp.gate_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.up_proj,
            weights,
            &format!("{prefix}.mlp.up_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.down_proj,
            weights,
            &format!("{prefix}.mlp.down_proj"),
        )?;

        // Layer norms
        load_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }

    // Load final norm
    load_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;

    // Load lm_head if not tied
    if let Some(ref mut lm_head) = model.lm_head {
        if let Some(w) = weights.get("lm_head.weight") {
            lm_head.weight = mlx_rs::module::Param::new(w.clone());
        }
    }

    Ok(())
}

/// Load weights into a Qwen3 model.
///
/// Qwen3 has additional q_norm and k_norm layers in attention.
pub fn load_qwen3_weights(
    model: &mut Qwen3ForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    // Load embed_tokens
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }

    // Load transformer layers
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");

        // Self-attention projections
        load_linear_weight(
            &mut layer.self_attn.q_proj,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.k_proj,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.v_proj,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.o_proj,
            weights,
            &format!("{prefix}.self_attn.o_proj"),
        )?;

        // Qwen3-specific: Q/K normalization layers
        load_rms_norm_weight(
            &mut layer.self_attn.q_norm,
            weights,
            &format!("{prefix}.self_attn.q_norm"),
        )?;
        load_rms_norm_weight(
            &mut layer.self_attn.k_norm,
            weights,
            &format!("{prefix}.self_attn.k_norm"),
        )?;

        // MLP projections
        load_linear_weight(
            &mut layer.mlp.gate_proj,
            weights,
            &format!("{prefix}.mlp.gate_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.up_proj,
            weights,
            &format!("{prefix}.mlp.up_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.down_proj,
            weights,
            &format!("{prefix}.mlp.down_proj"),
        )?;

        // Layer norms
        load_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }

    // Load final norm
    load_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;

    // Load lm_head if not tied
    if let Some(ref mut lm_head) = model.lm_head {
        if let Some(w) = weights.get("lm_head.weight") {
            lm_head.weight = mlx_rs::module::Param::new(w.clone());
        }
    }

    Ok(())
}

/// Load weights into a Gemma model.
pub fn load_gemma_weights(
    model: &mut GemmaForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    // Load embed_tokens
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }

    // Load transformer layers
    if let Some(ref mut layers) = model.model.layers.gemma1 {
        for (i, layer) in layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{i}");
            load_linear_weight(
                &mut layer.self_attn.q_proj,
                weights,
                &format!("{prefix}.self_attn.q_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.k_proj,
                weights,
                &format!("{prefix}.self_attn.k_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.v_proj,
                weights,
                &format!("{prefix}.self_attn.v_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.o_proj,
                weights,
                &format!("{prefix}.self_attn.o_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.gate_proj,
                weights,
                &format!("{prefix}.mlp.gate_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.up_proj,
                weights,
                &format!("{prefix}.mlp.up_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.down_proj,
                weights,
                &format!("{prefix}.mlp.down_proj"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.input_layernorm,
                weights,
                &format!("{prefix}.input_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.post_attention_layernorm,
                weights,
                &format!("{prefix}.post_attention_layernorm"),
            )?;
        }
    } else if let Some(ref mut layers) = model.model.layers.gemma2 {
        for (i, layer) in layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{i}");
            load_linear_weight(
                &mut layer.self_attn.q_proj,
                weights,
                &format!("{prefix}.self_attn.q_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.k_proj,
                weights,
                &format!("{prefix}.self_attn.k_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.v_proj,
                weights,
                &format!("{prefix}.self_attn.v_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.o_proj,
                weights,
                &format!("{prefix}.self_attn.o_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.gate_proj,
                weights,
                &format!("{prefix}.mlp.gate_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.up_proj,
                weights,
                &format!("{prefix}.mlp.up_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.down_proj,
                weights,
                &format!("{prefix}.mlp.down_proj"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.input_layernorm,
                weights,
                &format!("{prefix}.input_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.post_attention_layernorm,
                weights,
                &format!("{prefix}.post_attention_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.pre_feedforward_layernorm,
                weights,
                &format!("{prefix}.pre_feedforward_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.post_feedforward_layernorm,
                weights,
                &format!("{prefix}.post_feedforward_layernorm"),
            )?;
        }
    }

    // Load final norm
    load_gemma_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;

    Ok(())
}

/// Load weights into a Mistral model.
pub fn load_mistral_weights(
    model: &mut MistralForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    // Load embed_tokens
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }

    // Load transformer layers
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");

        // Self-attention projections
        load_linear_weight(
            &mut layer.self_attn.q_proj,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.k_proj,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.v_proj,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.o_proj,
            weights,
            &format!("{prefix}.self_attn.o_proj"),
        )?;

        // MLP projections
        load_linear_weight(
            &mut layer.mlp.gate_proj,
            weights,
            &format!("{prefix}.mlp.gate_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.up_proj,
            weights,
            &format!("{prefix}.mlp.up_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.down_proj,
            weights,
            &format!("{prefix}.mlp.down_proj"),
        )?;

        // Layer norms
        load_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }

    // Load final norm
    load_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;

    // Load lm_head if not tied
    if let Some(ref mut lm_head) = model.lm_head {
        if let Some(w) = weights.get("lm_head.weight") {
            lm_head.weight = mlx_rs::module::Param::new(w.clone());
        }
    }

    Ok(())
}

/// Load weights into a Phi model.
pub fn load_phi_weights(
    model: &mut PhiForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    // Load embed_tokens
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }

    // Load transformer layers
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{i}");

        // Self-attention projections
        load_linear_weight(
            &mut layer.self_attn.q_proj,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.k_proj,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.v_proj,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        )?;
        load_linear_weight(
            &mut layer.self_attn.o_proj,
            weights,
            &format!("{prefix}.self_attn.o_proj"),
        )?;

        // MLP projections (Phi uses gate_up_proj)
        load_linear_weight(
            &mut layer.mlp.gate_up_proj,
            weights,
            &format!("{prefix}.mlp.gate_up_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.down_proj,
            weights,
            &format!("{prefix}.mlp.down_proj"),
        )?;

        // Layer norms
        load_phi_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_phi_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }

    // Load final norm
    load_phi_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;

    // Load lm_head
    load_linear_weight(&mut model.lm_head, weights, "lm_head")?;

    Ok(())
}

fn load_gemma_rms_norm_weight(
    norm: &mut crate::architectures::gemma::GemmaRmsNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    Ok(())
}

fn load_phi_rms_norm_weight(
    norm: &mut crate::architectures::phi::PhiRMSNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = mlx_rs::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_index_parsing() {
        let json = r#"{
            "metadata": {"format": "pt"},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors"
            }
        }"#;

        let index: WeightIndex = serde_json::from_str(json).unwrap();
        assert_eq!(index.weight_map.len(), 2);
        assert_eq!(
            index.weight_map.get("model.embed_tokens.weight"),
            Some(&"model-00001-of-00002.safetensors".to_string())
        );
    }
}
