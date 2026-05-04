//! Unified weight loading for multiple formats.
//!
//! This module provides format-agnostic weight loading, supporting:
//! - **Safetensors** (HuggingFace format) with automatic BF16→F32 conversion
//! - **GGUF** (llama.cpp format) with dequantization and tensor name mapping
//!
//! # Usage
//!
//! ```ignore
//! use pmetal_models::weight_format::{WeightLoader, WeightFormat};
//!
//! // Auto-detect format and load
//! let weights = WeightLoader::load("./model");
//!
//! // Or load specific format
//! let weights = WeightLoader::load_gguf("./model.gguf");
//! ```

use pmetal_bridge::compat::{Array, Dtype, Exception};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Error type for weight loading.
#[derive(Debug, thiserror::Error)]
pub enum WeightFormatError {
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Safetensors loading error.
    #[error("Safetensors error: {0}")]
    Safetensors(String),
    /// GGUF loading error.
    #[error("GGUF error: {0}")]
    Gguf(String),
    /// Dequantization error.
    #[error("Dequantization error: {0}")]
    Dequant(String),
    /// Unsupported format.
    #[error("Unsupported weight format: {0}")]
    UnsupportedFormat(String),
    /// Missing weight.
    #[error("Missing weight: {0}")]
    MissingWeight(String),
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] pmetal_bridge::compat::Exception),
    /// MLX IO error.
    #[error("MLX IO error: {0}")]
    MlxIo(#[from] pmetal_bridge::compat::IoError),
    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Detected weight format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// Safetensors format (HuggingFace).
    Safetensors,
    /// GGUF format (llama.cpp).
    Gguf,
}

impl WeightFormat {
    /// Detect format from a path.
    ///
    /// Checks file extension and directory contents.
    pub fn detect(path: impl AsRef<Path>) -> Option<Self> {
        let path = path.as_ref();

        // Check if path is a file with .gguf extension
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext.to_string_lossy().to_lowercase() == "gguf" {
                    return Some(Self::Gguf);
                }
                if ext.to_string_lossy().to_lowercase() == "safetensors" {
                    return Some(Self::Safetensors);
                }
            }
        }

        // Check if path is a directory
        if path.is_dir() {
            // Prefer canonical HuggingFace safetensors directories when both
            // formats are present; auxiliary GGUF exports are common in model
            // workspaces and should not shadow the primary checkpoint.
            if has_safetensors_weights(path) {
                return Some(Self::Safetensors);
            }
            if find_gguf_file_in_dir(path).is_ok() {
                return Some(Self::Gguf);
            }
        }

        None
    }
}

fn has_safetensors_weights(path: &Path) -> bool {
    path.join("model.safetensors").exists() || path.join("model.safetensors.index.json").exists()
}

fn find_gguf_file_in_dir(path: &Path) -> Result<PathBuf, WeightFormatError> {
    let mut matches: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    matches.sort();

    if let Some(model_gguf) = matches
        .iter()
        .find(|path| path.file_name().and_then(|n| n.to_str()) == Some("model.gguf"))
    {
        return Ok(model_gguf.clone());
    }

    match matches.len() {
        0 => Err(WeightFormatError::UnsupportedFormat(format!(
            "No .gguf file found in: {}",
            path.display()
        ))),
        1 => Ok(matches.remove(0)),
        _ => Err(WeightFormatError::UnsupportedFormat(format!(
            "Multiple .gguf files found in {}; pass the file path explicitly",
            path.display()
        ))),
    }
}

fn resolve_gguf_path(path: &Path) -> Result<PathBuf, WeightFormatError> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    find_gguf_file_in_dir(path)
}

fn gguf_shape_to_i32(dimensions: &[u64], tensor_name: &str) -> Result<Vec<i32>, WeightFormatError> {
    dimensions
        .iter()
        .map(|&dim| {
            i32::try_from(dim).map_err(|_| {
                WeightFormatError::Gguf(format!(
                    "Tensor {tensor_name} dimension {dim} exceeds MLX i32 shape limits"
                ))
            })
        })
        .collect()
}

/// Unified weight loader supporting multiple formats.
pub struct WeightLoader;

impl WeightLoader {
    /// Load weights from a path, auto-detecting the format.
    ///
    /// For directories, looks for safetensors or GGUF files.
    /// For files, uses the file extension to determine format.
    ///
    /// # Arguments
    /// * `path` - Path to model directory or weight file
    ///
    /// # Returns
    /// HashMap of weight name (in HuggingFace format) to Array
    pub fn load(path: impl AsRef<Path>) -> Result<HashMap<String, Array>, WeightFormatError> {
        let path = path.as_ref();

        match WeightFormat::detect(path) {
            Some(WeightFormat::Safetensors) => Self::load_safetensors(path),
            Some(WeightFormat::Gguf) => Self::load_gguf(path),
            None => Err(WeightFormatError::UnsupportedFormat(format!(
                "Could not detect weight format for: {}",
                path.display()
            ))),
        }
    }

    /// Load weights from safetensors format.
    ///
    /// Handles both single-file and sharded models.
    /// Automatically converts BF16 weights to F32 for training compatibility.
    pub fn load_safetensors(
        path: impl AsRef<Path>,
    ) -> Result<HashMap<String, Array>, WeightFormatError> {
        let path = path.as_ref();
        let mut weights = HashMap::new();

        if path.is_file() {
            // Single file
            let path_str = path.to_str().unwrap_or_default();
            if let Some(pairs) = pmetal_bridge::inline_array::load_safetensors_shard(path_str) {
                for (name, array) in pairs {
                    weights.insert(name, Self::convert_dtype_for_training(array)?);
                }
            }
        } else if path.is_dir() {
            // Directory - check for single file or sharded
            let single_file = path.join("model.safetensors");
            if single_file.exists() {
                let path_str = single_file.to_str().unwrap_or_default();
                if let Some(pairs) = pmetal_bridge::inline_array::load_safetensors_shard(path_str) {
                    for (name, array) in pairs {
                        weights.insert(name, Self::convert_dtype_for_training(array)?);
                    }
                }
            } else {
                // Load sharded model
                let index_path = path.join("model.safetensors.index.json");
                if !index_path.exists() {
                    return Err(WeightFormatError::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "No model.safetensors or model.safetensors.index.json found",
                    )));
                }

                #[derive(serde::Deserialize)]
                struct WeightIndex {
                    weight_map: HashMap<String, String>,
                }

                let index_content = std::fs::read_to_string(&index_path)?;
                let index: WeightIndex = serde_json::from_str(&index_content)?;

                // Get unique shard files
                let shard_files: std::collections::HashSet<&String> =
                    index.weight_map.values().collect();

                // Load each shard
                for shard_file in shard_files {
                    let shard_path = path.join(shard_file);
                    let path_str = shard_path.to_str().unwrap_or_default();
                    if let Some(pairs) =
                        pmetal_bridge::inline_array::load_safetensors_shard(path_str)
                    {
                        for (name, array) in pairs {
                            weights.insert(name, Self::convert_dtype_for_training(array)?);
                        }
                    }
                }
            }
        } else {
            return Err(WeightFormatError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Path not found: {}", path.display()),
            )));
        }

        Ok(weights)
    }

    /// Load weights from GGUF format.
    ///
    /// Dequantizes all tensors to F32 and maps tensor names to HuggingFace format.
    pub fn load_gguf(path: impl AsRef<Path>) -> Result<HashMap<String, Array>, WeightFormatError> {
        let path = path.as_ref();

        let gguf_path = resolve_gguf_path(path)?;

        // Read GGUF content
        let content = pmetal_gguf::GgufContent::from_file(&gguf_path)
            .map_err(|e| WeightFormatError::Gguf(e.to_string()))?;

        // Open file for tensor data reading
        let mut file = std::fs::File::open(&gguf_path)?;

        let mut weights = HashMap::new();

        // Load and dequantize each tensor
        let tensor_names: Vec<String> = content.tensor_names().map(String::from).collect();

        for tensor_name in tensor_names {
            let tensor_info = content.get_tensor_info(&tensor_name).ok_or_else(|| {
                WeightFormatError::Gguf(format!("Tensor not found: {}", tensor_name))
            })?;

            // Check if dequantization is supported
            if !pmetal_gguf::dequant::is_supported(tensor_info.dtype) {
                tracing::warn!(
                    "Skipping tensor {} with unsupported dtype {:?}",
                    tensor_name,
                    tensor_info.dtype
                );
                continue;
            }

            // Get shape for dequantization
            let shape = gguf_shape_to_i32(&tensor_info.dimensions, &tensor_name)?;

            // Read tensor data
            let data = content
                .read_tensor_data(&mut file, &tensor_name)
                .map_err(|e| WeightFormatError::Gguf(e.to_string()))?;

            // Dequantize to f32
            let floats = pmetal_gguf::dequant::dequantize(&data, tensor_info.dtype, &shape)
                .map_err(|e| WeightFormatError::Dequant(e.to_string()))?;

            // Create MLX array
            let array = Array::from_slice(&floats, &shape);

            // Map GGUF tensor name to HuggingFace format
            let hf_name = Self::gguf_to_hf_name(&tensor_name);
            weights.insert(hf_name, array);
        }

        Ok(weights)
    }

    /// Convert array dtype for training compatibility.
    ///
    /// BF16 is converted to F32 since MLX training works better with F32.
    /// F16 is kept as-is (MLX handles F16 well).
    fn convert_dtype_for_training(array: Array) -> Result<Array, WeightFormatError> {
        match array.dtype() {
            Dtype::Bfloat16 => {
                // Convert BF16 to F32 for training compatibility
                Ok(array.as_dtype(Dtype::Float32.as_i32()))
            }
            // Keep other dtypes as-is
            _ => Ok(array),
        }
    }

    /// Map GGUF tensor name to HuggingFace format.
    ///
    /// GGUF uses names like `blk.0.attn_q.weight`
    /// HuggingFace uses names like `model.layers.0.self_attn.q_proj.weight`
    pub fn gguf_to_hf_name(gguf_name: &str) -> String {
        // Handle special cases first
        match gguf_name {
            "token_embd.weight" => return "model.embed_tokens.weight".to_string(),
            "output_norm.weight" => return "model.norm.weight".to_string(),
            "output.weight" => return "lm_head.weight".to_string(),
            _ => {}
        }

        // Handle block tensors: blk.N.xxx -> model.layers.N.xxx
        if gguf_name.starts_with("blk.") {
            // Parse block number
            let parts: Vec<&str> = gguf_name.splitn(3, '.').collect();
            if parts.len() >= 3 {
                let block_num = parts[1];
                let rest = parts[2];

                // Map tensor names
                let hf_rest = match rest {
                    // Attention
                    "attn_norm.weight" => "input_layernorm.weight".to_string(),
                    "attn_q.weight" => "self_attn.q_proj.weight".to_string(),
                    "attn_k.weight" => "self_attn.k_proj.weight".to_string(),
                    "attn_v.weight" => "self_attn.v_proj.weight".to_string(),
                    "attn_output.weight" => "self_attn.o_proj.weight".to_string(),
                    // FFN
                    "ffn_norm.weight" => "post_attention_layernorm.weight".to_string(),
                    "ffn_gate.weight" => "mlp.gate_proj.weight".to_string(),
                    "ffn_up.weight" => "mlp.up_proj.weight".to_string(),
                    "ffn_down.weight" => "mlp.down_proj.weight".to_string(),
                    "ffn_gate_inp.weight" => "mlp.gate.weight".to_string(),
                    "ffn_gate_exps.weight" => "mlp.experts.gate_proj.weight".to_string(),
                    "ffn_up_exps.weight" => "mlp.experts.up_proj.weight".to_string(),
                    "ffn_down_exps.weight" => "mlp.experts.down_proj.weight".to_string(),
                    "ffn_gate_shexp.weight" => "mlp.shared_expert.gate_proj.weight".to_string(),
                    "ffn_up_shexp.weight" => "mlp.shared_expert.up_proj.weight".to_string(),
                    "ffn_down_shexp.weight" => "mlp.shared_expert.down_proj.weight".to_string(),
                    "ffn_gate_inp_shexp.weight" => "mlp.shared_expert_gate.weight".to_string(),
                    // Qwen3-specific (Q/K norm)
                    "attn_q_norm.weight" => "self_attn.q_norm.weight".to_string(),
                    "attn_k_norm.weight" => "self_attn.k_norm.weight".to_string(),
                    // Pass through unknown
                    other => other.to_string(),
                };

                return format!("model.layers.{}.{}", block_num, hf_rest);
            }
        }

        // Pass through unknown names
        gguf_name.to_string()
    }

    /// Map HuggingFace tensor names to GGUF naming.
    ///
    /// HuggingFace uses names like `model.layers.0.self_attn.q_proj.weight`;
    /// GGUF uses names like `blk.0.attn_q.weight`.
    pub fn hf_to_gguf_name(hf_name: &str) -> String {
        match hf_name {
            "model.embed_tokens.weight" => return "token_embd.weight".to_string(),
            "model.norm.weight" => return "output_norm.weight".to_string(),
            "lm_head.weight" => return "output.weight".to_string(),
            _ => {}
        }

        let Some(rest) = hf_name.strip_prefix("model.layers.") else {
            return hf_name.to_string();
        };
        let Some((layer_idx, suffix)) = rest.split_once('.') else {
            return hf_name.to_string();
        };
        if layer_idx.parse::<usize>().is_err() {
            return hf_name.to_string();
        }

        let gguf_suffix = match suffix {
            "input_layernorm.weight" => "attn_norm.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "mlp.gate.weight" => "ffn_gate_inp.weight",
            "mlp.router.weight" => "ffn_gate_inp.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "mlp.experts.gate_proj.weight" => "ffn_gate_exps.weight",
            "mlp.experts.up_proj.weight" => "ffn_up_exps.weight",
            "mlp.experts.down_proj.weight" => "ffn_down_exps.weight",
            "mlp.shared_expert.gate_proj.weight" => "ffn_gate_shexp.weight",
            "mlp.shared_expert.up_proj.weight" => "ffn_up_shexp.weight",
            "mlp.shared_expert.down_proj.weight" => "ffn_down_shexp.weight",
            "mlp.shared_experts.gate_proj.weight" => "ffn_gate_shexp.weight",
            "mlp.shared_experts.up_proj.weight" => "ffn_up_shexp.weight",
            "mlp.shared_experts.down_proj.weight" => "ffn_down_shexp.weight",
            "mlp.shared_expert_gate.weight" => "ffn_gate_inp_shexp.weight",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            _ => suffix,
        };

        format!("blk.{layer_idx}.{gguf_suffix}")
    }
}

/// Configuration extracted from GGUF metadata.
///
/// GGUF stores model config in metadata with architecture prefix, e.g.,
/// `llama.embedding_length`, `llama.block_count`, etc.
#[derive(Debug, Clone)]
pub struct GgufModelConfig {
    /// Model architecture (e.g., "llama", "qwen2", "mistral")
    pub architecture: String,
    /// Hidden size / embedding dimension
    pub hidden_size: i32,
    /// Number of transformer layers
    pub num_hidden_layers: i32,
    /// Number of attention heads
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: Option<i32>,
    /// Intermediate (FFN) size
    pub intermediate_size: i32,
    /// Vocabulary size
    pub vocab_size: i32,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta (frequency base)
    pub rope_theta: f32,
    /// Maximum sequence length
    pub max_position_embeddings: i32,
    /// Head dimension (if specified)
    pub head_dim: Option<i32>,
}

impl GgufModelConfig {
    /// Extract config from GGUF metadata.
    pub fn from_gguf(content: &pmetal_gguf::GgufContent) -> Result<Self, WeightFormatError> {
        let arch = content
            .architecture()
            .ok_or_else(|| WeightFormatError::Gguf("Missing general.architecture".into()))?
            .to_string();

        // Helper to get metadata with architecture prefix
        let get_u32 = |key: &str| -> Option<u32> {
            let full_key = format!("{}.{}", arch, key);
            match content.get_metadata(&full_key) {
                Some(pmetal_gguf::MetadataValue::Uint32(v)) => Some(*v),
                Some(pmetal_gguf::MetadataValue::Int32(v)) => Some(*v as u32),
                Some(pmetal_gguf::MetadataValue::Uint64(v)) => Some(*v as u32),
                Some(pmetal_gguf::MetadataValue::Int64(v)) => Some(*v as u32),
                _ => None,
            }
        };

        let get_f32 = |key: &str| -> Option<f32> {
            let full_key = format!("{}.{}", arch, key);
            match content.get_metadata(&full_key) {
                Some(pmetal_gguf::MetadataValue::Float32(v)) => Some(*v),
                Some(pmetal_gguf::MetadataValue::Float64(v)) => Some(*v as f32),
                _ => None,
            }
        };

        // Extract required values
        let hidden_size = get_u32("embedding_length").ok_or_else(|| {
            WeightFormatError::Gguf(format!("{}.embedding_length not found", arch))
        })? as i32;

        let num_hidden_layers = get_u32("block_count")
            .ok_or_else(|| WeightFormatError::Gguf(format!("{}.block_count not found", arch)))?
            as i32;

        let num_attention_heads = get_u32("attention.head_count").ok_or_else(|| {
            WeightFormatError::Gguf(format!("{}.attention.head_count not found", arch))
        })? as i32;

        let intermediate_size = get_u32("feed_forward_length").ok_or_else(|| {
            WeightFormatError::Gguf(format!("{}.feed_forward_length not found", arch))
        })? as i32;

        // Get vocab size from tokenizer or estimate from token embedding
        let vocab_size = content
            .get_metadata("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                pmetal_gguf::MetadataValue::Array(arr) => Some(arr.len() as i32),
                _ => None,
            })
            .or_else(|| {
                // Fallback: estimate from token embedding tensor shape
                content
                    .get_tensor_info("token_embd.weight")
                    .map(|t| t.dimensions[0] as i32)
            })
            .unwrap_or(32000); // Default fallback

        // Optional values with defaults
        let num_kv_heads = get_u32("attention.head_count_kv").map(|v| v as i32);
        let rms_norm_eps = get_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let rope_theta = get_f32("rope.freq_base").unwrap_or(10000.0);
        let max_position_embeddings = get_u32("context_length").unwrap_or(4096) as i32;
        let head_dim = get_u32("attention.head_dim")
            .or_else(|| get_u32("attention.key_length"))
            .or_else(|| get_u32("rope.dimension_count"))
            .map(|v| v as i32);

        Ok(Self {
            architecture: arch,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_kv_heads,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
            head_dim,
        })
    }

    /// Convert to Llama config.
    pub fn to_llama_config(&self) -> crate::architectures::llama::LlamaConfig {
        crate::architectures::llama::LlamaConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            ..Default::default()
        }
    }

    /// Convert to Qwen3 config.
    pub fn to_qwen3_config(&self) -> crate::architectures::qwen3::Qwen3Config {
        // Qwen3 requires head_dim as i32, compute if not provided
        let head_dim = self
            .head_dim
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads);
        crate::architectures::qwen3::Qwen3Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_kv_heads,
            head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            ..Default::default()
        }
    }

    /// Convert to Mistral config.
    pub fn to_mistral_config(&self) -> crate::architectures::mistral::MistralConfig {
        crate::architectures::mistral::MistralConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            ..Default::default()
        }
    }

    /// Convert to Gemma config.
    pub fn to_gemma_config(&self) -> crate::architectures::gemma::GemmaConfig {
        crate::architectures::gemma::GemmaConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            ..Default::default()
        }
    }

    /// Convert to Phi config.
    pub fn to_phi_config(&self) -> crate::architectures::phi::PhiConfig {
        use crate::architectures::phi::{LayerNormType, PhiActivation, PhiConfig};

        PhiConfig {
            model_type: "phi3".to_string(),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_kv_heads.unwrap_or(self.num_attention_heads),
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
            partial_rotary_factor: 0.5, // Default for Phi-3
            rms_norm_eps: self.rms_norm_eps,
            qkv_bias: false, // Default, Phi-4 uses true
            hidden_act: PhiActivation::SwiGLU,
            sliding_window: None,
            layer_norm_type: LayerNormType::RmsNorm,
            original_max_position_embeddings: None,
            rope_scaling: None,
            tie_word_embeddings: false,
        }
    }
}

/// Get the detected model architecture from a GGUF file.
pub fn get_gguf_architecture(path: impl AsRef<Path>) -> Result<Option<String>, WeightFormatError> {
    let path = path.as_ref();

    let gguf_path = resolve_gguf_path(path)?;

    let content = pmetal_gguf::GgufContent::from_file(&gguf_path)
        .map_err(|e| WeightFormatError::Gguf(e.to_string()))?;

    Ok(content.architecture().map(String::from))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_to_hf_name_embeddings() {
        assert_eq!(
            WeightLoader::gguf_to_hf_name("token_embd.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name("output_norm.weight"),
            "model.norm.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name("output.weight"),
            "lm_head.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_attention() {
        assert_eq!(
            WeightLoader::gguf_to_hf_name("blk.0.attn_q.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name("blk.15.attn_k.weight"),
            "model.layers.15.self_attn.k_proj.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name("blk.7.attn_v.weight"),
            "model.layers.7.self_attn.v_proj.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name("blk.0.attn_output.weight"),
            "model.layers.0.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_ffn() {
        assert_eq!(
            WeightLoader::gguf_to_hf_name("blk.0.ffn_gate.weight"),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name("blk.0.ffn_up.weight"),
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name("blk.0.ffn_down.weight"),
            "model.layers.0.mlp.down_proj.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_norms() {
        assert_eq!(
            WeightLoader::gguf_to_hf_name("blk.0.attn_norm.weight"),
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name("blk.0.ffn_norm.weight"),
            "model.layers.0.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_qwen3() {
        // Qwen3-specific Q/K norms
        assert_eq!(
            WeightLoader::gguf_to_hf_name("blk.0.attn_q_norm.weight"),
            "model.layers.0.self_attn.q_norm.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name("blk.0.attn_k_norm.weight"),
            "model.layers.0.self_attn.k_norm.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_round_trip_core_tensors() {
        let names = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.2.self_attn.v_proj.weight",
            "model.layers.3.self_attn.o_proj.weight",
            "model.layers.4.input_layernorm.weight",
            "model.layers.5.post_attention_layernorm.weight",
            "model.layers.6.mlp.gate_proj.weight",
            "model.layers.7.mlp.up_proj.weight",
            "model.layers.8.mlp.down_proj.weight",
            "model.layers.9.self_attn.q_norm.weight",
            "model.layers.10.self_attn.k_norm.weight",
        ];

        for name in names {
            let gguf = WeightLoader::hf_to_gguf_name(name);
            assert_eq!(WeightLoader::gguf_to_hf_name(&gguf), name);
        }
    }

    #[test]
    fn test_hf_to_gguf_name_moe_tensors() {
        assert_eq!(
            WeightLoader::hf_to_gguf_name("model.layers.0.mlp.gate.weight"),
            "blk.0.ffn_gate_inp.weight"
        );
        assert_eq!(
            WeightLoader::hf_to_gguf_name("model.layers.0.mlp.experts.gate_proj.weight"),
            "blk.0.ffn_gate_exps.weight"
        );
        assert_eq!(
            WeightLoader::hf_to_gguf_name("model.layers.0.mlp.shared_expert.up_proj.weight"),
            "blk.0.ffn_up_shexp.weight"
        );
    }

    #[test]
    fn test_format_detection_safetensors() {
        // Test with existing test paths (non-existent paths return None)
        assert_eq!(WeightFormat::detect("model.gguf"), None); // File doesn't exist
        assert_eq!(WeightFormat::detect("model.safetensors"), None); // File doesn't exist
    }

    #[test]
    fn test_format_detection_prefers_safetensors_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::File::create(dir.path().join("model.safetensors")).unwrap();
        std::fs::File::create(dir.path().join("extra.gguf")).unwrap();

        assert_eq!(
            WeightFormat::detect(dir.path()),
            Some(WeightFormat::Safetensors)
        );
    }

    #[test]
    fn test_find_gguf_file_requires_explicit_path_for_ambiguous_dirs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::File::create(dir.path().join("a.gguf")).unwrap();
        std::fs::File::create(dir.path().join("b.gguf")).unwrap();

        assert!(find_gguf_file_in_dir(dir.path()).is_err());
    }

    #[test]
    fn test_find_gguf_file_prefers_model_gguf() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::File::create(dir.path().join("z.gguf")).unwrap();
        std::fs::File::create(dir.path().join("model.gguf")).unwrap();

        assert_eq!(
            find_gguf_file_in_dir(dir.path()).unwrap(),
            dir.path().join("model.gguf")
        );
    }
}
