//! GGUF weight loading and the format detection that routes to it.
//!
//! Safetensors checkpoints are read by [`crate::loader`]; this module owns the
//! GGUF side: tensor-name mapping, dequantization and the model config GGUF
//! carries in its metadata.

use pmetal_runtime::compat::Array;
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
    Mlx(#[from] pmetal_runtime::compat::Exception),
    /// MLX IO error.
    #[error("MLX IO error: {0}")]
    MlxIo(#[from] pmetal_runtime::compat::IoError),
    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<pmetal_gguf::config::GgufConfigError> for WeightFormatError {
    fn from(e: pmetal_gguf::config::GgufConfigError) -> Self {
        Self::Gguf(e.to_string())
    }
}

/// Detected weight format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// Safetensors format (HuggingFace).
    Safetensors,
    /// GGUF format.
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
    // Multimodal projector files sit next to the language model but are not
    // it; the shards of a split model are one model, named by its first shard.
    matches.retain(|path| {
        !path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("mmproj"))
    });
    for path in &mut matches {
        if let Some(first) = pmetal_gguf::first_shard_path(path) {
            *path = first;
        }
    }
    matches.sort();
    matches.dedup();

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

/// Which Gemma layout a GGUF `general.architecture` writes.
///
/// Both store every norm weight with the 1 that Gemma's `x * (1 + w)` adds
/// already applied, so the checkpoint's own `w` is the file's value less one.
/// They differ in the block: Gemma 1 has the Llama layout (two norms), Gemma 2
/// and 3 have four, named by what they wrap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmaGguf {
    /// `gemma`: the Llama tensor names.
    Gemma,
    /// `gemma2` and `gemma3`: [`WeightLoader::gguf_to_hf_name_gemma`].
    PrePostNorm,
}

impl GemmaGguf {
    fn of(architecture: Option<&str>) -> Option<Self> {
        match architecture? {
            "gemma" => Some(Self::Gemma),
            "gemma2" | "gemma3" => Some(Self::PrePostNorm),
            _ => None,
        }
    }
}

/// Unified weight loader supporting multiple formats.
pub struct WeightLoader;

impl WeightLoader {
    /// Load weights from GGUF format.
    ///
    /// Dequantizes all tensors to F32 and maps tensor names to HuggingFace format.
    ///
    /// The tensors come out as the Hugging Face checkpoint of the same model
    /// holds them, so the conventions the GGUF architecture writes differently
    /// are undone here: Gemma's tensor names for Gemma 2 and 3 and its norm
    /// weights, which a GGUF stores with the 1 the model adds already applied.
    pub fn load_gguf(path: impl AsRef<Path>) -> Result<HashMap<String, Array>, WeightFormatError> {
        let path = path.as_ref();

        let gguf_path = resolve_gguf_path(path)?;

        // Map the model (any shard of a split set opens the whole set); tensor
        // bytes are borrowed from the mapping rather than read into buffers.
        let model = pmetal_gguf::GgufModel::open(&gguf_path)
            .map_err(|e| WeightFormatError::Gguf(e.to_string()))?;
        let gemma = GemmaGguf::of(model.architecture());

        let mut weights = HashMap::new();

        // Load and dequantize each tensor, in on-disk order
        let tensor_names: Vec<String> = model
            .tensors_in_disk_order()
            .into_iter()
            .map(|info| info.name.clone())
            .collect();

        for tensor_name in tensor_names {
            let tensor_info = model.tensor_info(&tensor_name).ok_or_else(|| {
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

            let data = model
                .tensor_data(&tensor_name)
                .map_err(|e| WeightFormatError::Gguf(e.to_string()))?;

            // Dequantize to f32
            let mut floats = pmetal_gguf::dequant::dequantize(data, tensor_info.dtype, &shape)
                .map_err(|e| WeightFormatError::Dequant(e.to_string()))?;

            // Map GGUF tensor name to HuggingFace format
            let hf_name = match gemma {
                Some(GemmaGguf::PrePostNorm) => Self::gguf_to_hf_name_gemma(&tensor_name),
                _ => Self::gguf_to_hf_name(&tensor_name),
            };
            if gemma.is_some() && hf_name.ends_with("norm.weight") {
                for value in &mut floats {
                    *value -= 1.0;
                }
            }

            // Create MLX array
            let array = Array::from_slice(&floats, &shape);
            weights.insert(hf_name, array);
        }

        Ok(weights)
    }

    /// Map GGUF tensor name to HuggingFace format.
    ///
    /// GGUF uses names like `blk.0.attn_q.weight`
    /// HuggingFace uses names like `model.layers.0.self_attn.q_proj.weight`
    pub fn gguf_to_hf_name(gguf_name: &str) -> String {
        // A bias is named as the weight beside it: `attn_q.bias` is the bias of
        // `attn_q.weight`.
        if let Some(stem) = gguf_name.strip_suffix(".bias") {
            let weight = format!("{stem}.weight");
            let mapped = Self::gguf_to_hf_name(&weight);
            return match mapped.strip_suffix(".weight") {
                Some(hf_stem) if mapped != weight => format!("{hf_stem}.bias"),
                _ => gguf_name.to_string(),
            };
        }

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

    /// Map a Gemma 2 / Gemma 3 GGUF tensor name to HuggingFace format.
    ///
    /// Gemma's per-layer norm layout differs from the Llama family, so the
    /// generic [`Self::gguf_to_hf_name`] mis-maps it: in Gemma the GGUF
    /// `ffn_norm` is the *pre*-feedforward norm (Llama's `ffn_norm` is the
    /// post-attention norm), and Gemma carries a separate `post_attention_norm`
    /// and `post_ffw_norm`. Names are taken verbatim from the `gguf` Python package's
    /// `gguf/constants.py` `TENSOR_NAMES` for the Gemma architectures:
    ///
    /// | GGUF                        | HuggingFace                       |
    /// |-----------------------------|-----------------------------------|
    /// | `attn_norm`                 | `input_layernorm`                 |
    /// | `post_attention_norm`       | `post_attention_layernorm`        |
    /// | `attn_q/k/v/output`         | `self_attn.{q,k,v,o}_proj`        |
    /// | `attn_q_norm`/`attn_k_norm` | `self_attn.{q,k}_norm`            |
    /// | `ffn_norm`                  | `pre_feedforward_layernorm`       |
    /// | `post_ffw_norm`             | `post_feedforward_layernorm`      |
    /// | `ffn_gate/up/down`          | `mlp.{gate,up,down}_proj`         |
    pub fn gguf_to_hf_name_gemma(gguf_name: &str) -> String {
        match gguf_name {
            "token_embd.weight" => return "model.embed_tokens.weight".to_string(),
            "output_norm.weight" => return "model.norm.weight".to_string(),
            "output.weight" => return "lm_head.weight".to_string(),
            _ => {}
        }

        if gguf_name.starts_with("blk.") {
            let parts: Vec<&str> = gguf_name.splitn(3, '.').collect();
            if parts.len() >= 3 {
                let block_num = parts[1];
                let hf_rest = match parts[2] {
                    "attn_norm.weight" => "input_layernorm.weight",
                    "post_attention_norm.weight" => "post_attention_layernorm.weight",
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "attn_q_norm.weight" => "self_attn.q_norm.weight",
                    "attn_k_norm.weight" => "self_attn.k_norm.weight",
                    // Gemma: ffn_norm is the PRE-feedforward norm (not
                    // post-attention as in Llama).
                    "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
                    "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    other => other,
                };
                return format!("model.layers.{block_num}.{hf_rest}");
            }
        }

        gguf_name.to_string()
    }

    /// Map HuggingFace tensor names to GGUF naming.
    ///
    /// HuggingFace uses names like `model.layers.0.self_attn.q_proj.weight`;
    /// GGUF uses names like `blk.0.attn_q.weight`.
    pub fn hf_to_gguf_name(hf_name: &str) -> String {
        // A bias is named as the weight beside it.
        if let Some(stem) = hf_name.strip_suffix(".bias") {
            let weight = format!("{stem}.weight");
            let mapped = Self::hf_to_gguf_name(&weight);
            return match mapped.strip_suffix(".weight") {
                Some(gguf_stem) if mapped != weight => format!("{gguf_stem}.bias"),
                _ => hf_name.to_string(),
            };
        }

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
            "block_sparse_moe.gate.weight" => "ffn_gate_inp.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "mlp.experts.gate_proj.weight" => "ffn_gate_exps.weight",
            "mlp.experts.up_proj.weight" => "ffn_up_exps.weight",
            "mlp.experts.down_proj.weight" => "ffn_down_exps.weight",
            // The stacked layout an MLX conversion (and a loaded model's own
            // parameter tree) holds the routed experts in.
            "mlp.switch_mlp.gate_proj.weight" => "ffn_gate_exps.weight",
            "mlp.switch_mlp.up_proj.weight" => "ffn_up_exps.weight",
            "mlp.switch_mlp.down_proj.weight" => "ffn_down_exps.weight",
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

    /// [`hf_to_gguf_name`](Self::hf_to_gguf_name) for Gemma 2 and Gemma 3, the
    /// inverse of [`gguf_to_hf_name_gemma`](Self::gguf_to_hf_name_gemma).
    ///
    /// Their four norms per block are named by what they wrap: the Hugging
    /// Face `post_attention_layernorm` is the norm *after* attention
    /// (`post_attention_norm`), and the feed-forward block has a pre- and a
    /// post-norm of its own (`ffn_norm`, `post_ffw_norm`). Gemma 1 has the
    /// Llama layout and uses the generic map.
    pub fn hf_to_gguf_name_gemma(hf_name: &str) -> String {
        if let Some(rest) = hf_name.strip_prefix("model.layers.")
            && let Some((layer, suffix)) = rest.split_once('.')
            && layer.parse::<usize>().is_ok()
        {
            let gguf_suffix = match suffix {
                "post_attention_layernorm.weight" => Some("post_attention_norm.weight"),
                "pre_feedforward_layernorm.weight" => Some("ffn_norm.weight"),
                "post_feedforward_layernorm.weight" => Some("post_ffw_norm.weight"),
                _ => None,
            };
            if let Some(gguf_suffix) = gguf_suffix {
                return format!("blk.{layer}.{gguf_suffix}");
            }
        }
        Self::hf_to_gguf_name(hf_name)
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
    /// Channels the rotation spans (`rope.dimension_count`): the head for a
    /// full-rotary model, a fraction of it for Phi-4-mini's partial one.
    pub rope_dimension_count: Option<i32>,
    /// The pretraining context the scaled frequencies are defined against
    /// (`rope.scaling.original_context_length`).
    pub original_context_length: Option<i32>,
    /// RoPE scaling in the `rope_scaling` dict form a `config.json` states it:
    /// from the file's `rope.scaling.*` metadata, or the `rope_freqs` tensor
    /// ([`set_rope_frequency_factors`](Self::set_rope_frequency_factors)).
    pub rope_scaling: Option<serde_json::Value>,
    /// Sliding-window width of the local-attention layers (Gemma 2 and 3).
    pub sliding_window: Option<i32>,
    /// RoPE base of the sliding-window layers (Gemma 3).
    pub rope_local_base_freq: Option<f32>,
    /// `tanh` cap on the attention scores (Gemma 2).
    pub attn_logit_softcapping: Option<f32>,
    /// `tanh` cap on the output logits (Gemma 2).
    pub final_logit_softcapping: Option<f32>,
}

impl GgufModelConfig {
    /// Apply the GGUF `rope_freqs` tensor, the per-frequency divisors a
    /// converter precomputes for Llama 3.x's banded scaling.
    ///
    /// The tensor replaces what the file's scaling metadata says; a file that
    /// states a scaling of its own as well would have two to apply, and gets an
    /// error rather than one of them.
    pub fn set_rope_frequency_factors(&mut self, factors: &[f32]) -> Result<(), WeightFormatError> {
        self.rope_scaling = Some(pmetal_gguf::config::frequency_factors_scaling(
            self.rope_scaling.as_ref(),
            factors,
        )?);
        Ok(())
    }

    /// Resolve a GGUF path (file or directory) and extract its model config
    /// from metadata. Keeps GGUF path resolution encapsulated in this module
    /// so callers (e.g. the inference dispatcher) need only a path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, WeightFormatError> {
        let gguf_path = resolve_gguf_path(path.as_ref())?;
        let model = pmetal_gguf::GgufModel::open(&gguf_path)
            .map_err(|e| WeightFormatError::Gguf(e.to_string()))?;
        Self::from_gguf(model.metadata_content())
    }

    /// Extract config from GGUF metadata.
    ///
    /// The parameters are read by [`pmetal_gguf::config::GgufArchConfig`], which
    /// the generated `config.json` is written from as well; this adds what
    /// loading a model needs on top: the parameters every architecture has to
    /// state (an error when the file leaves one out), and a default for the
    /// ones with a conventional value. The vocabulary has none: a file that
    /// does not say how large it is cannot be loaded as a guess.
    pub fn from_gguf(content: &pmetal_gguf::GgufContent) -> Result<Self, WeightFormatError> {
        let arch = pmetal_gguf::config::GgufArchConfig::from_gguf(content)?;
        let required = |value: Option<u32>, key: &str| -> Result<i32, WeightFormatError> {
            value.and_then(|v| i32::try_from(v).ok()).ok_or_else(|| {
                WeightFormatError::Gguf(format!("{}.{key} not found", arch.architecture))
            })
        };
        let as_i32 = |value: Option<u32>| value.and_then(|v| i32::try_from(v).ok());

        Ok(Self {
            hidden_size: required(arch.hidden_size, "embedding_length")?,
            num_hidden_layers: required(arch.num_hidden_layers, "block_count")?,
            num_attention_heads: required(arch.num_attention_heads, "attention.head_count")?,
            intermediate_size: required(arch.intermediate_size, "feed_forward_length")?,
            // No tokenizer, no `vocab_size`, no embedding to count rows of: the
            // size is unknown, and 32000 would be somebody else's.
            vocab_size: required(arch.vocab_size, "vocab_size")?,
            num_kv_heads: as_i32(arch.num_key_value_heads),
            rms_norm_eps: arch.rms_norm_eps.map_or(1e-5, |v| v as f32),
            rope_theta: arch.rope_theta.map_or(10000.0, |v| v as f32),
            max_position_embeddings: as_i32(arch.max_position_embeddings).unwrap_or(4096),
            head_dim: as_i32(arch.head_dim),
            rope_dimension_count: as_i32(arch.rope_dimension_count),
            original_context_length: as_i32(arch.original_context_length),
            rope_scaling: arch.rope_scaling,
            sliding_window: as_i32(arch.sliding_window),
            rope_local_base_freq: arch.rope_theta_swa.map(|v| v as f32),
            attn_logit_softcapping: arch.attn_logit_softcapping.map(|v| v as f32),
            final_logit_softcapping: arch.final_logit_softcapping.map(|v| v as f32),
            architecture: arch.architecture,
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
            rope_scaling: self.rope_scaling.clone(),
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
            rope_scaling: self.rope_scaling.clone(),
            ..Default::default()
        }
    }

    /// Convert to Qwen2 config.
    pub fn to_qwen2_config(&self) -> crate::architectures::qwen2::Qwen2Config {
        crate::architectures::qwen2::Qwen2Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_kv_heads,
            // Passed through rather than derived here: `Qwen2Config` resolves
            // an absent `head_dim` the same way, and one owner of that rule is
            // enough.
            head_dim: self.head_dim,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            rope_scaling: self.rope_scaling.clone(),
            ..Default::default()
        }
    }

    /// Complete a Phi-3 LongRoPE scaling with the `rope_factors_short` and
    /// `rope_factors_long` tensors the converter writes beside the weights.
    ///
    /// A model with neither has no scaling to complete; one tensor without the
    /// other is an error.
    pub fn set_longrope_factors(
        &mut self,
        short: Option<&[f32]>,
        long: Option<&[f32]>,
    ) -> Result<(), WeightFormatError> {
        let original = self
            .original_context_length
            .and_then(|v| u32::try_from(v).ok());
        if let Some(scaling) = pmetal_gguf::config::longrope_scaling(short, long, original)? {
            self.rope_scaling = Some(scaling);
        }
        Ok(())
    }

    /// Convert to Phi-3 / Phi-4-mini config.
    ///
    /// `rope.dimension_count` is the rotated width, so the head is
    /// `hidden / heads` and the partial rotary factor their ratio.
    pub fn to_phi_config(&self) -> crate::architectures::phi::PhiConfig {
        let head_dim = (self.hidden_size / self.num_attention_heads.max(1)).max(1);
        let rope_dim = self.rope_dimension_count.unwrap_or(head_dim);
        crate::architectures::phi::PhiConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_kv_heads.unwrap_or(self.num_attention_heads),
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
            partial_rotary_factor: rope_dim as f32 / head_dim as f32,
            rms_norm_eps: self.rms_norm_eps,
            original_max_position_embeddings: self.original_context_length,
            rope_scaling: self.rope_scaling.clone(),
            ..crate::architectures::phi::PhiConfig::phi3_mini()
        }
    }

    /// Convert to a Gemma 1, 2 or 3 config.
    ///
    /// The generation is the file's `general.architecture`. The attention
    /// scale is the one thing a GGUF leaves out: `query_pre_attn_scalar` is
    /// the head width, except for the two 27B models, which use
    /// `hidden / heads` (the engines that read GGUF infer it from the layer
    /// count the same way, since the file does not say).
    pub fn to_gemma_config(&self) -> crate::architectures::gemma::GemmaConfig {
        let is_gemma2 = self.architecture == "gemma2";
        let is_gemma3 = self.architecture == "gemma3";
        let head_dim = self
            .head_dim
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads.max(1));
        let wide_attention = (is_gemma2 && self.num_hidden_layers == 46)
            || (is_gemma3 && self.num_hidden_layers == 62);
        let query_pre_attn_scalar = (is_gemma2 || is_gemma3).then(|| {
            if wide_attention {
                self.hidden_size / self.num_attention_heads.max(1)
            } else {
                head_dim
            }
        });
        crate::architectures::gemma::GemmaConfig {
            model_type: self.architecture.clone(),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_kv_heads,
            head_dim: Some(head_dim),
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            attn_logit_softcapping: self.attn_logit_softcapping,
            final_logit_softcapping: self.final_logit_softcapping,
            query_pre_attn_scalar,
            sliding_window: self.sliding_window,
            rope_local_base_freq: self.rope_local_base_freq,
            is_gemma2,
            is_gemma3,
            rope_scaling: self.rope_scaling.clone(),
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
            rope_scaling: self.rope_scaling.clone(),
            ..Default::default()
        }
    }
}

/// Name a Phi-3 GGUF's fused tensors as the model holds them.
///
/// Phi-3 keeps the attention projections in one `attn_qkv` (rows q, k, v) and
/// the SwiGLU gate and up projections in one `ffn_up` (rows gate, up), where
/// the generic name map treats `ffn_up` as the up projection alone. They become
/// `self_attn.qkv_proj` (split into three by
/// [`split_phi_fused_qkv`](crate::loader::split_phi_fused_qkv)) and
/// `mlp.gate_up_proj`, the names of the Hugging Face checkpoint.
pub fn name_phi3_fused_tensors(weights: &mut HashMap<String, Array>) {
    let renames: Vec<(String, String)> = weights
        .keys()
        .filter_map(|key| {
            if let Some(layer) = key.strip_suffix(".attn_qkv.weight") {
                return Some((key.clone(), format!("{layer}.self_attn.qkv_proj.weight")));
            }
            let layer = key.strip_suffix(".mlp.up_proj.weight")?;
            // Only a layer with no gate of its own fuses it into the up
            // projection.
            (!weights.contains_key(&format!("{layer}.mlp.gate_proj.weight")))
                .then(|| (key.clone(), format!("{layer}.mlp.gate_up_proj.weight")))
        })
        .collect();
    for (from, to) in renames {
        if let Some(tensor) = weights.remove(&from) {
            weights.insert(to, tensor);
        }
    }
}

/// Undo the GGUF converter's row permutation of a Llama-architecture
/// checkpoint's query and key projections, in place.
///
/// GGUF runtimes rotate the `llama` architecture's heads in *interleaved* pairs
/// `(2i, 2i+1)`, where Hugging Face rotates `(i, i + head_dim/2)`. The
/// converter reorders every head's rows so the two conventions
/// agree: `reshape(n_head, 2, head_dim/2, ..).swapaxes(1, 2)`, which sends row
/// `a·head_dim/2 + b` of a head to row `2b + a`. The permutation touches
/// `q_proj` and `k_proj`, weights and biases, and nothing else, and it is only
/// done for the `llama` architecture: the Qwen and Phi families rotate in the
/// Hugging Face convention already.
///
/// Loading such a file without undoing it feeds the HF-convention rotation rows
/// in the wrong order: finite, plausibly shaped logits from attention that
/// attends to the wrong positions.
///
/// `num_heads` is the query head count and `num_kv_heads` the key head count
/// (the same for MHA).
pub fn unpermute_llama_qk(
    weights: &mut HashMap<String, Array>,
    num_heads: i32,
    num_kv_heads: i32,
) -> Result<(), WeightFormatError> {
    let heads_of = |name: &str| -> Option<i32> {
        if name.ends_with(".self_attn.q_proj.weight") || name.ends_with(".self_attn.q_proj.bias") {
            Some(num_heads)
        } else if name.ends_with(".self_attn.k_proj.weight")
            || name.ends_with(".self_attn.k_proj.bias")
        {
            Some(num_kv_heads)
        } else {
            None
        }
    };
    for (name, tensor) in weights.iter_mut() {
        let Some(heads) = heads_of(name) else {
            continue;
        };
        let shape = tensor.shape().to_vec();
        let rows = shape[0];
        if heads <= 0 || rows % (2 * heads) != 0 {
            return Err(WeightFormatError::Gguf(format!(
                "{name} has {rows} rows, which {heads} heads cannot split into rotary pairs"
            )));
        }
        let rest: Vec<i32> = shape[1..].to_vec();
        let mut grouped = vec![heads, rows / heads / 2, 2];
        grouped.extend(&rest);
        // The inverse of `[heads, 2, d/2] -> [heads, d/2, 2]`.
        *tensor = tensor.reshape(&grouped).swap_axes(1, 2).reshape(&shape);
    }
    Ok(())
}

/// Get the detected model architecture from a GGUF file.
pub fn get_gguf_architecture(path: impl AsRef<Path>) -> Result<Option<String>, WeightFormatError> {
    let path = path.as_ref();

    let gguf_path = resolve_gguf_path(path)?;

    let model = pmetal_gguf::GgufModel::open(&gguf_path)
        .map_err(|e| WeightFormatError::Gguf(e.to_string()))?;

    Ok(model.architecture().map(String::from))
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

    /// A bias is named as the weight beside it, both ways: Qwen 2 has query,
    /// key and value biases, and a file that names them `blk.N.self_attn.q_proj.bias`
    /// has tensors no engine looks for.
    #[test]
    fn test_biases_follow_their_weights() {
        for (hf, gguf) in [
            ("model.layers.3.self_attn.q_proj.bias", "blk.3.attn_q.bias"),
            ("model.layers.3.self_attn.k_proj.bias", "blk.3.attn_k.bias"),
            ("model.layers.3.self_attn.v_proj.bias", "blk.3.attn_v.bias"),
        ] {
            assert_eq!(WeightLoader::hf_to_gguf_name(hf), gguf);
            assert_eq!(WeightLoader::gguf_to_hf_name(gguf), hf);
        }
        // A bias outside the layers, with no weight of that name, is left as it is.
        assert_eq!(
            WeightLoader::hf_to_gguf_name("model.mystery.bias"),
            "model.mystery.bias"
        );
    }

    /// Gemma 2 and 3 name their four norms by what they wrap, and the reader
    /// maps the names back.
    #[test]
    fn test_hf_to_gguf_name_gemma_round_trips_with_the_reader() {
        for (hf, gguf) in [
            (
                "model.layers.4.post_attention_layernorm.weight",
                "blk.4.post_attention_norm.weight",
            ),
            (
                "model.layers.4.pre_feedforward_layernorm.weight",
                "blk.4.ffn_norm.weight",
            ),
            (
                "model.layers.4.post_feedforward_layernorm.weight",
                "blk.4.post_ffw_norm.weight",
            ),
            (
                "model.layers.4.input_layernorm.weight",
                "blk.4.attn_norm.weight",
            ),
            (
                "model.layers.4.self_attn.q_norm.weight",
                "blk.4.attn_q_norm.weight",
            ),
            ("model.layers.4.mlp.up_proj.weight", "blk.4.ffn_up.weight"),
        ] {
            assert_eq!(WeightLoader::hf_to_gguf_name_gemma(hf), gguf);
            assert_eq!(WeightLoader::gguf_to_hf_name_gemma(gguf), hf);
        }
    }

    fn gemma_config(architecture: &str, layers: i32, hidden: i32, heads: i32) -> GgufModelConfig {
        GgufModelConfig {
            architecture: architecture.to_string(),
            hidden_size: hidden,
            num_hidden_layers: layers,
            num_attention_heads: heads,
            num_kv_heads: Some(heads / 2),
            intermediate_size: hidden * 4,
            vocab_size: 262_144,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32_768,
            head_dim: Some(128),
            rope_dimension_count: Some(128),
            original_context_length: None,
            rope_scaling: None,
            sliding_window: Some(1024),
            rope_local_base_freq: Some(10_000.0),
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
        }
    }

    /// A GGUF does not state `query_pre_attn_scalar`: it is the head width,
    /// except for the two 27B models, where it is `hidden / heads`.
    #[test]
    fn the_gemma_attention_scale_is_inferred_the_way_the_converter_does() {
        // Gemma 3 1B / 4B / 12B: the head width.
        let small = gemma_config("gemma3", 34, 2560, 8).to_gemma_config();
        assert_eq!(small.query_pre_attn_scalar, Some(128));
        assert!(small.is_gemma3 && !small.is_gemma2);
        // Gemma 3 27B: 5376 / 32.
        let large = gemma_config("gemma3", 62, 5376, 32).to_gemma_config();
        assert_eq!(large.query_pre_attn_scalar, Some(168));
        // Gemma 2 27B: 4608 / 32.
        let large = gemma_config("gemma2", 46, 4608, 32).to_gemma_config();
        assert_eq!(large.query_pre_attn_scalar, Some(144));
        assert!(large.is_gemma2 && !large.is_gemma3);
        // Gemma 1 has no such scalar: the scale is the head width's.
        let one = gemma_config("gemma", 28, 3072, 16).to_gemma_config();
        assert_eq!(one.query_pre_attn_scalar, None);
        assert!(!one.is_gemma2 && !one.is_gemma3);
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
    fn test_gguf_to_hf_name_gemma_norm_layout() {
        // The Gemma-specific norm layout is the whole reason this map exists:
        // ffn_norm is the PRE-feedforward norm, post_attention_norm and
        // post_ffw_norm are separate.
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("blk.0.attn_norm.weight"),
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("blk.3.post_attention_norm.weight"),
            "model.layers.3.post_attention_layernorm.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("blk.3.ffn_norm.weight"),
            "model.layers.3.pre_feedforward_layernorm.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("blk.3.post_ffw_norm.weight"),
            "model.layers.3.post_feedforward_layernorm.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_gemma_attention_and_qk_norm() {
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("blk.1.attn_q.weight"),
            "model.layers.1.self_attn.q_proj.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("blk.1.attn_output.weight"),
            "model.layers.1.self_attn.o_proj.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("blk.2.attn_q_norm.weight"),
            "model.layers.2.self_attn.q_norm.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("blk.2.attn_k_norm.weight"),
            "model.layers.2.self_attn.k_norm.weight"
        );
    }

    #[test]
    fn test_gguf_to_hf_name_gemma_embeddings_and_ffn() {
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("token_embd.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("output_norm.weight"),
            "model.norm.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("blk.0.ffn_gate.weight"),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            WeightLoader::gguf_to_hf_name_gemma("blk.0.ffn_down.weight"),
            "model.layers.0.mlp.down_proj.weight"
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

    #[test]
    fn a_directory_with_a_split_model_and_a_projector_is_one_model() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "Model-Q4-00001-of-00003.gguf",
            "Model-Q4-00002-of-00003.gguf",
            "Model-Q4-00003-of-00003.gguf",
            "mmproj-F16.gguf",
        ] {
            std::fs::write(dir.path().join(name), b"x").unwrap();
        }
        assert_eq!(
            find_gguf_file_in_dir(dir.path()).unwrap(),
            dir.path().join("Model-Q4-00001-of-00003.gguf")
        );

        // Two different models are still ambiguous.
        std::fs::write(dir.path().join("Other-Q8_0.gguf"), b"x").unwrap();
        assert!(find_gguf_file_in_dir(dir.path()).is_err());
    }

    /// the GGUF converter's `LlamaModel.permute`, written out row by row rather than as
    /// the reshape it is in the converter, so the loader cannot agree with it by
    /// construction: head row `a·d/2 + b` goes to row `2b + a`.
    fn gguf_permute(rows: &[Vec<f32>], heads: usize) -> Vec<Vec<f32>> {
        let d = rows.len() / heads;
        let mut out = vec![Vec::new(); rows.len()];
        for h in 0..heads {
            for a in 0..2 {
                for b in 0..d / 2 {
                    out[h * d + 2 * b + a] = rows[h * d + a * (d / 2) + b].clone();
                }
            }
        }
        out
    }

    fn matrix(rows: &[Vec<f32>]) -> Array {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Array::from_f32_slice(&flat, &[rows.len() as i32, rows[0].len() as i32])
    }

    fn rows_of(array: &Array) -> Vec<Vec<f32>> {
        let (rows, cols) = (array.shape()[0] as usize, array.shape()[1] as usize);
        let mut array = array.clone();
        let flat = array.to_f32_vec(rows * cols).unwrap();
        flat.chunks(cols).map(<[f32]>::to_vec).collect()
    }

    /// ⚠️ Every `llama`-architecture GGUF stores its q and k rows permuted, and
    /// the loader assigned them as they were.
    #[test]
    fn gguf_row_permutation_is_undone_for_q_and_k_but_not_v() {
        let (heads, kv_heads, head_dim, hidden) = (4usize, 2usize, 8usize, 3usize);
        let rows = |n: usize, seed: f32| -> Vec<Vec<f32>> {
            (0..n)
                .map(|r| {
                    (0..hidden)
                        .map(|c| seed + (r * hidden + c) as f32)
                        .collect()
                })
                .collect()
        };
        let q = rows(heads * head_dim, 0.0);
        let k = rows(kv_heads * head_dim, 1000.0);
        let v = rows(kv_heads * head_dim, 2000.0);

        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            matrix(&gguf_permute(&q, heads)),
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            matrix(&gguf_permute(&k, kv_heads)),
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            matrix(&v),
        );

        unpermute_llama_qk(&mut weights, heads as i32, kv_heads as i32).unwrap();

        assert_eq!(
            rows_of(&weights["model.layers.0.self_attn.q_proj.weight"]),
            q
        );
        assert_eq!(
            rows_of(&weights["model.layers.0.self_attn.k_proj.weight"]),
            k
        );
        assert_eq!(
            rows_of(&weights["model.layers.0.self_attn.v_proj.weight"]),
            v
        );
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    /// A query bias is permuted the same way as the rows it belongs to.
    #[test]
    fn a_permuted_bias_is_unpermuted_too() {
        let (heads, head_dim) = (2usize, 4usize);
        let rows: Vec<Vec<f32>> = (0..heads * head_dim).map(|r| vec![r as f32]).collect();
        let permuted = gguf_permute(&rows, heads);
        let flat: Vec<f32> = permuted.iter().flatten().copied().collect();
        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.self_attn.q_proj.bias".to_string(),
            Array::from_f32_slice(&flat, &[(heads * head_dim) as i32]),
        );
        unpermute_llama_qk(&mut weights, heads as i32, heads as i32).unwrap();
        let mut bias = weights["model.layers.0.self_attn.q_proj.bias"].clone();
        let want: Vec<f32> = rows.iter().flatten().copied().collect();
        assert_eq!(bias.to_f32_vec(want.len()).unwrap(), want);
    }

    #[test]
    fn rows_that_cannot_be_rotary_pairs_are_refused() {
        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Array::from_f32_slice(&[0.0; 9], &[3, 3]),
        );
        assert!(unpermute_llama_qk(&mut weights, 1, 1).is_err());
    }
}
