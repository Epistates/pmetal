//! Model configuration from GGUF metadata.
//!
//! GGUF files are self-contained: the architecture lives in the file's
//! metadata, and the RoPE scaling that metadata cannot hold lives in tensors
//! beside the weights. [`GgufArchConfig`] reads both into one typed value. It
//! is what loaders build their model config from, and what
//! [`config_from_gguf_model`] serialises as a HuggingFace-schema `config.json`
//! so a downloaded GGUF loads like any other model directory.
//!
//! Nothing here invents a value. A field the file does not state is `None`
//! (and absent from the JSON), and a RoPE scaling the loaders cannot apply is
//! an error, because the positions past the original context would be wrong
//! without a word.
//!
//! # RoPE scaling
//!
//! | The file carries | `rope_scaling` |
//! |---|---|
//! | `rope.scaling.type` = `linear` / `yarn` and a factor | `{rope_type, factor, original_max_position_embeddings?}` |
//! | a `rope_freqs` tensor (Llama 3.x's banded scaling, precomputed by the converter) | `{rope_type: "frequency_factors", factors}` |
//! | `rope.scaling.type` = `longrope` and the two `rope_factors_*` tensors | `{rope_type: "longrope", short_factor, long_factor, original_max_position_embeddings?}` |
//!
//! The `rope_freqs` tensor is the per-frequency divisor the Llama 3 bands
//! reduce to, so the factors state them exactly; the band parameters that
//! produced it are not stored in the file and are not guessed back.

use crate::model::GgufModel;
use crate::reader::GgufContent;
use crate::types::{GgmlType, MetadataValue};
use serde_json::{Value, json};

/// Why a configuration could not be read from a GGUF file.
#[derive(Debug, thiserror::Error)]
pub enum GgufConfigError {
    /// The file has no `general.architecture`.
    #[error("the file has no general.architecture")]
    MissingArchitecture,
    /// The RoPE scaling the file states cannot be applied as it stands.
    #[error("{0}")]
    RopeScaling(String),
    /// A tensor the configuration is read from is not usable.
    #[error("tensor {name}: {reason}")]
    Tensor {
        /// The tensor's name in the file.
        name: String,
        /// What is wrong with it.
        reason: String,
    },
    /// `config.json` could not be written.
    #[error("could not write config.json: {0}")]
    Io(#[from] std::io::Error),
}

/// The tensor holding Llama 3.x's precomputed per-frequency scaling.
pub const ROPE_FREQS_TENSOR: &str = "rope_freqs.weight";
/// The tensor holding LongRoPE's short-context factors.
pub const ROPE_FACTORS_SHORT_TENSOR: &str = "rope_factors_short.weight";
/// The tensor holding LongRoPE's long-context factors.
pub const ROPE_FACTORS_LONG_TENSOR: &str = "rope_factors_long.weight";

/// The architecture parameters a GGUF file states.
///
/// Every parameter is optional: this reports what the file says, and the
/// caller decides which absences are fatal and which have a default.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufArchConfig {
    /// `general.architecture` (`llama`, `qwen2`, ...).
    pub architecture: String,
    /// `general.name`.
    pub name: Option<String>,
    /// Embedding width (`embedding_length`).
    pub hidden_size: Option<u32>,
    /// Transformer layers (`block_count`).
    pub num_hidden_layers: Option<u32>,
    /// Attention heads (`attention.head_count`).
    pub num_attention_heads: Option<u32>,
    /// Key/value heads (`attention.head_count_kv`).
    pub num_key_value_heads: Option<u32>,
    /// Feed-forward width (`feed_forward_length`).
    pub intermediate_size: Option<u32>,
    /// Context length (`context_length`).
    pub max_position_embeddings: Option<u32>,
    /// Vocabulary size: `vocab_size`, else the token list's length, else the
    /// token embedding's row count.
    pub vocab_size: Option<u32>,
    /// RMS-norm epsilon (`attention.layer_norm_rms_epsilon`).
    pub rms_norm_eps: Option<f64>,
    /// RoPE base frequency (`rope.freq_base`).
    pub rope_theta: Option<f64>,
    /// Head width: `attention.key_length`, `attention.head_dim` or
    /// `rope.dimension_count`, in that order.
    pub head_dim: Option<u32>,
    /// Channels the rotation spans (`rope.dimension_count`): the head for a
    /// full-rotary model, a fraction of it for a partial one.
    pub rope_dimension_count: Option<u32>,
    /// The context the scaled frequencies are defined against
    /// (`rope.scaling.original_context_length`).
    pub original_context_length: Option<u32>,
    /// Experts per MoE layer (`expert_count`).
    pub num_experts: Option<u32>,
    /// Experts routed per token (`expert_used_count`).
    pub num_experts_per_tok: Option<u32>,
    /// `general.file_type`.
    pub file_type: Option<u32>,
    /// RoPE scaling in the `rope_scaling` dict form a `config.json` states it.
    /// See the [module documentation](self).
    pub rope_scaling: Option<Value>,
}

impl GgufArchConfig {
    /// Read the configuration the file's metadata states.
    ///
    /// Scaling that lives in tensors is not visible here; [`Self::from_model`]
    /// adds it. A `longrope` type is left as a bare `{rope_type: "longrope"}`
    /// until [`Self::set_longrope_factors`] completes it.
    ///
    /// # Errors
    ///
    /// No architecture, or a `rope.scaling.type` that is not understood or
    /// lacks its factor.
    pub fn from_gguf(content: &GgufContent) -> Result<Self, GgufConfigError> {
        let arch = content
            .architecture()
            .ok_or(GgufConfigError::MissingArchitecture)?
            .to_string();
        let u32_of = |suffix: &str| {
            content
                .get_u64(&format!("{arch}.{suffix}"))
                .and_then(|v| u32::try_from(v).ok())
        };
        let f64_of = |suffix: &str| content.get_f64(&format!("{arch}.{suffix}"));

        let original_context_length = u32_of("rope.scaling.original_context_length");
        let rope_scaling = match content.get_str(&format!("{arch}.rope.scaling.type")) {
            None | Some("none") | Some("") => None,
            // LongRoPE's factors are tensors, not metadata: the dict stays
            // incomplete here and a family that cannot read it is refused by
            // the missing factors.
            Some("longrope") => Some(json!({ "rope_type": "longrope" })),
            Some(kind @ ("linear" | "yarn")) => {
                let factor = f64_of("rope.scaling.factor").ok_or_else(|| {
                    GgufConfigError::RopeScaling(format!(
                        "{arch}.rope.scaling.type is {kind:?} but {arch}.rope.scaling.factor is \
                         missing"
                    ))
                })?;
                let mut dict = json!({ "rope_type": kind, "factor": factor });
                if let Some(original) = original_context_length {
                    dict["original_max_position_embeddings"] = original.into();
                }
                Some(dict)
            }
            Some(other) => {
                return Err(GgufConfigError::RopeScaling(format!(
                    "{arch}.rope.scaling.type {other:?} is not supported (linear, yarn, longrope \
                     and none are)"
                )));
            }
        };

        let vocab_size = u32_of("vocab_size")
            .or_else(|| match content.get_metadata("tokenizer.ggml.tokens") {
                Some(MetadataValue::Array(tokens)) => u32::try_from(tokens.len()).ok(),
                _ => None,
            })
            .or_else(|| {
                // Row-major `[vocab, hidden]`.
                let info = content.get_tensor_info("token_embd.weight")?;
                u32::try_from(*info.dimensions.first()?).ok()
            });
        let rope_dimension_count = u32_of("rope.dimension_count");

        Ok(Self {
            name: content.get_str("general.name").map(str::to_string),
            hidden_size: u32_of("embedding_length"),
            num_hidden_layers: u32_of("block_count"),
            num_attention_heads: u32_of("attention.head_count"),
            num_key_value_heads: u32_of("attention.head_count_kv"),
            intermediate_size: u32_of("feed_forward_length"),
            max_position_embeddings: u32_of("context_length"),
            vocab_size,
            rms_norm_eps: f64_of("attention.layer_norm_rms_epsilon"),
            rope_theta: f64_of("rope.freq_base"),
            head_dim: u32_of("attention.key_length")
                .or_else(|| u32_of("attention.head_dim"))
                .or(rope_dimension_count),
            rope_dimension_count,
            original_context_length,
            num_experts: u32_of("expert_count"),
            num_experts_per_tok: u32_of("expert_used_count"),
            file_type: content
                .get_u64("general.file_type")
                .and_then(|v| u32::try_from(v).ok()),
            rope_scaling,
            architecture: arch,
        })
    }

    /// Read the configuration of an open model: its metadata, and the RoPE
    /// scaling its `rope_freqs` / `rope_factors_*` tensors carry.
    ///
    /// # Errors
    ///
    /// As [`Self::from_gguf`], and when a scaling tensor is unreadable or
    /// contradicts the metadata ([`Self::set_rope_frequency_factors`],
    /// [`Self::set_longrope_factors`]).
    pub fn from_model(model: &GgufModel) -> Result<Self, GgufConfigError> {
        let mut config = Self::from_gguf(model.metadata_content())?;
        if let Some(factors) = read_f32_tensor(model, ROPE_FREQS_TENSOR)? {
            config.set_rope_frequency_factors(&factors)?;
        }
        let short = read_f32_tensor(model, ROPE_FACTORS_SHORT_TENSOR)?;
        let long = read_f32_tensor(model, ROPE_FACTORS_LONG_TENSOR)?;
        config.set_longrope_factors(short.as_deref(), long.as_deref())?;
        Ok(config)
    }

    /// Apply the `rope_freqs` tensor ([`frequency_factors_scaling`]).
    ///
    /// # Errors
    ///
    /// The metadata already states a scaling.
    pub fn set_rope_frequency_factors(&mut self, factors: &[f32]) -> Result<(), GgufConfigError> {
        self.rope_scaling = Some(frequency_factors_scaling(
            self.rope_scaling.as_ref(),
            factors,
        )?);
        Ok(())
    }

    /// Complete a LongRoPE scaling with its two factor tensors
    /// ([`longrope_scaling`]). A model with neither has nothing to complete.
    ///
    /// # Errors
    ///
    /// Only one of the two tensors is present.
    pub fn set_longrope_factors(
        &mut self,
        short: Option<&[f32]>,
        long: Option<&[f32]>,
    ) -> Result<(), GgufConfigError> {
        if let Some(scaling) = longrope_scaling(short, long, self.original_context_length)? {
            self.rope_scaling = Some(scaling);
        }
        Ok(())
    }

    /// The parameters as a HuggingFace-schema `config.json` object.
    pub fn to_config_json(&self) -> Value {
        let mut config = serde_json::Map::new();
        let mut put = |key: &str, value: Value| {
            config.insert(key.to_string(), value);
        };
        put("model_type", json!(self.architecture));
        put(
            "architectures",
            json!([map_architecture_class(&self.architecture)]),
        );
        let optional: [(&str, Option<Value>); 15] = [
            ("_name_or_path", self.name.as_ref().map(|v| json!(v))),
            ("hidden_size", self.hidden_size.map(|v| json!(v))),
            (
                "num_hidden_layers",
                self.num_hidden_layers.map(|v| json!(v)),
            ),
            (
                "num_attention_heads",
                self.num_attention_heads.map(|v| json!(v)),
            ),
            (
                "num_key_value_heads",
                self.num_key_value_heads.map(|v| json!(v)),
            ),
            (
                "intermediate_size",
                self.intermediate_size.map(|v| json!(v)),
            ),
            (
                "max_position_embeddings",
                self.max_position_embeddings.map(|v| json!(v)),
            ),
            ("vocab_size", self.vocab_size.map(|v| json!(v))),
            ("rope_theta", self.rope_theta.map(|v| json!(v))),
            ("head_dim", self.head_dim.map(|v| json!(v))),
            ("rms_norm_eps", self.rms_norm_eps.map(|v| json!(v))),
            ("num_local_experts", self.num_experts.map(|v| json!(v))),
            (
                "num_experts_per_tok",
                self.num_experts_per_tok.map(|v| json!(v)),
            ),
            ("gguf_file_type", self.file_type.map(|v| json!(v))),
            ("rope_scaling", self.rope_scaling.clone()),
        ];
        for (key, value) in optional {
            if let Some(value) = value {
                put(key, value);
            }
        }
        Value::Object(config)
    }
}

/// `rope_scaling` for a `rope_freqs` tensor: the per-frequency divisors the
/// converter precomputed for Llama 3.x's banded scaling.
///
/// `existing` is what the file's metadata already states. A file that states a
/// scaling of its own as well has two to apply, and gets an error rather than
/// one of them.
///
/// # Errors
///
/// `existing` is `Some`.
pub fn frequency_factors_scaling(
    existing: Option<&Value>,
    factors: &[f32],
) -> Result<Value, GgufConfigError> {
    if existing.is_some() {
        return Err(GgufConfigError::RopeScaling(
            "the file carries both a rope_freqs tensor and rope.scaling metadata; \
             applying one of them would be a guess"
                .into(),
        ));
    }
    Ok(json!({ "rope_type": "frequency_factors", "factors": factors }))
}

/// `rope_scaling` for a Phi-3 style LongRoPE: the `rope_factors_short` and
/// `rope_factors_long` tensors the converter writes beside the weights.
///
/// `None` when the file has neither (no scaling to state).
///
/// # Errors
///
/// Only one of the two is present.
pub fn longrope_scaling(
    short: Option<&[f32]>,
    long: Option<&[f32]>,
    original_context_length: Option<u32>,
) -> Result<Option<Value>, GgufConfigError> {
    let (short, long) = match (short, long) {
        (None, None) => return Ok(None),
        (Some(short), Some(long)) => (short, long),
        _ => {
            return Err(GgufConfigError::RopeScaling(
                "the file has only one of rope_factors_short and rope_factors_long".into(),
            ));
        }
    };
    let mut dict = json!({
        "rope_type": "longrope",
        "short_factor": short,
        "long_factor": long,
    });
    if let Some(original) = original_context_length {
        dict["original_max_position_embeddings"] = original.into();
    }
    Ok(Some(dict))
}

/// A one-dimensional `F32` tensor's values, or `None` when the file has no
/// such tensor.
fn read_f32_tensor(model: &GgufModel, name: &str) -> Result<Option<Vec<f32>>, GgufConfigError> {
    let Some(info) = model.tensor_info(name) else {
        return Ok(None);
    };
    let tensor_error = |reason: String| GgufConfigError::Tensor {
        name: name.to_string(),
        reason,
    };
    if info.dtype != GgmlType::F32 {
        return Err(tensor_error(format!("is {:?}, expected F32", info.dtype)));
    }
    let bytes = model
        .tensor_data(name)
        .map_err(|e| tensor_error(e.to_string()))?;
    Ok(Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
    ))
}

/// The `config.json` object for a GGUF file's metadata alone.
///
/// It cannot see tensors, so a scaling carried by `rope_freqs` or the
/// `rope_factors_*` tensors is missing; use [`config_from_gguf_model`] when
/// the tensors are at hand.
///
/// # Errors
///
/// As [`GgufArchConfig::from_gguf`].
pub fn config_from_gguf(content: &GgufContent) -> Result<Value, GgufConfigError> {
    Ok(GgufArchConfig::from_gguf(content)?.to_config_json())
}

/// The `config.json` object for an open GGUF model, RoPE scaling included.
///
/// # Errors
///
/// As [`GgufArchConfig::from_model`].
pub fn config_from_gguf_model(model: &GgufModel) -> Result<Value, GgufConfigError> {
    Ok(GgufArchConfig::from_model(model)?.to_config_json())
}

/// Write a `config.json` into `output_dir` from a GGUF model.
///
/// Useful when downloading GGUF-only models that lack one. An existing
/// `config.json` is left alone and its path returned.
///
/// # Errors
///
/// As [`config_from_gguf_model`], or the file cannot be written.
pub fn write_config_from_gguf(
    model: &GgufModel,
    output_dir: &std::path::Path,
) -> Result<std::path::PathBuf, GgufConfigError> {
    let path = output_dir.join("config.json");
    if path.exists() {
        return Ok(path);
    }
    let config = config_from_gguf_model(model)?;
    let json = serde_json::to_string_pretty(&config).map_err(std::io::Error::other)?;
    // Through a temporary file, so an interrupted write never leaves a
    // truncated config.json that the "do not overwrite" check above would
    // then preserve.
    let mut temp = tempfile::NamedTempFile::new_in(output_dir)?;
    std::io::Write::write_all(&mut temp, json.as_bytes())?;
    temp.persist(&path).map_err(|e| e.error)?;
    Ok(path)
}

/// Map GGUF architecture name to HuggingFace model class name.
fn map_architecture_class(arch: &str) -> String {
    match arch {
        "llama" => "LlamaForCausalLM".to_string(),
        "mistral" => "MistralForCausalLM".to_string(),
        "gemma" | "gemma2" | "gemma3" => "GemmaForCausalLM".to_string(),
        "gemma4" => "Gemma4ForCausalLM".to_string(),
        "qwen" | "qwen2" | "qwen3" => "Qwen2ForCausalLM".to_string(),
        "phi" | "phi2" | "phi3" => "PhiForCausalLM".to_string(),
        "gpt2" => "GPT2LMHeadModel".to_string(),
        "falcon" => "FalconForCausalLM".to_string(),
        "mamba" => "MambaForCausalLM".to_string(),
        "cohere" | "command-r" => "CohereForCausalLM".to_string(),
        "deepseek" | "deepseek2" => "DeepseekV2ForCausalLM".to_string(),
        other => {
            // CamelCase fallback: "my_arch" → "MyArchForCausalLM"
            let capitalized: String = other
                .split('_')
                .map(|w| {
                    let mut c = w.chars();
                    match c.next() {
                        Some(f) => f.to_uppercase().to_string() + c.as_str(),
                        None => String::new(),
                    }
                })
                .collect();
            format!("{capitalized}ForCausalLM")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_content(arch: &str, metadata: Vec<(&str, MetadataValue)>) -> GgufContent {
        let mut map: HashMap<String, MetadataValue> = HashMap::new();
        map.insert(
            "general.architecture".to_string(),
            MetadataValue::String(arch.to_string()),
        );
        for (k, v) in metadata {
            map.insert(k.to_string(), v);
        }
        GgufContent {
            metadata: map,
            ..GgufContent::default()
        }
    }

    #[test]
    fn test_config_from_gguf_llama() {
        let content = make_content(
            "llama",
            vec![
                ("llama.block_count", MetadataValue::Uint32(32)),
                ("llama.embedding_length", MetadataValue::Uint32(4096)),
                ("llama.attention.head_count", MetadataValue::Uint32(32)),
                ("llama.attention.head_count_kv", MetadataValue::Uint32(8)),
                ("llama.context_length", MetadataValue::Uint32(8192)),
                ("llama.feed_forward_length", MetadataValue::Uint32(11008)),
            ],
        );

        let config = config_from_gguf(&content).unwrap();
        assert_eq!(config["model_type"], "llama");
        assert_eq!(config["num_hidden_layers"], 32);
        assert_eq!(config["hidden_size"], 4096);
        assert_eq!(config["num_attention_heads"], 32);
        assert_eq!(config["num_key_value_heads"], 8);
        assert_eq!(config["max_position_embeddings"], 8192);
        assert_eq!(config["intermediate_size"], 11008);
        assert_eq!(config["architectures"][0], "LlamaForCausalLM");
    }

    #[test]
    fn test_config_from_gguf_no_architecture() {
        let content = GgufContent::default();
        assert!(matches!(
            config_from_gguf(&content),
            Err(GgufConfigError::MissingArchitecture)
        ));
    }

    #[test]
    fn test_vocab_size_from_tokens() {
        let tokens: Vec<MetadataValue> = (0..32000)
            .map(|i| MetadataValue::String(format!("tok_{i}")))
            .collect();
        let content = make_content(
            "llama",
            vec![("tokenizer.ggml.tokens", MetadataValue::Array(tokens))],
        );

        let config = config_from_gguf(&content).unwrap();
        assert_eq!(config["vocab_size"], 32000);
    }
}

#[cfg(test)]
mod rope_tests {
    use super::*;
    use crate::{GgufBuilder, GgufModel};

    const HEAD_DIM: u32 = 8;

    /// A Llama-shaped GGUF on disk with whatever else `extra` adds.
    fn llama_model(dir: &std::path::Path, extra: impl FnOnce(&mut GgufBuilder)) -> GgufModel {
        let mut b = GgufBuilder::with_model("llama", "t");
        b.add_u32("llama.block_count", 2)
            .add_u32("llama.embedding_length", 32)
            .add_u32("llama.attention.head_count", 4)
            .add_u32("llama.attention.head_count_kv", 2)
            .add_u32("llama.feed_forward_length", 64)
            .add_u32("llama.context_length", 131_072)
            .add_f32("llama.rope.freq_base", 500_000.0)
            .add_u32("llama.rope.dimension_count", HEAD_DIM);
        b.add_f32_tensor("token_embd.weight", vec![100, 32], vec![0.0; 3200]);
        extra(&mut b);
        let path = dir.join("m.gguf");
        b.write_file(&path).unwrap();
        GgufModel::open(&path).unwrap()
    }

    /// Llama 3.x's converter writes its banded scaling as a `rope_freqs` tensor.
    /// The config.json written from the file has to carry it, or a downloaded
    /// Llama-3.1 GGUF claims it has no scaling.
    #[test]
    fn a_rope_freqs_tensor_becomes_the_configs_rope_scaling() {
        let dir = tempfile::tempdir().unwrap();
        let model = llama_model(dir.path(), |b| {
            b.add_f32_tensor("rope_freqs.weight", vec![4], vec![1.0, 1.0, 8.0, 32.0]);
        });
        let config = config_from_gguf_model(&model).unwrap();
        assert_eq!(config["rope_scaling"]["rope_type"], "frequency_factors");
        assert_eq!(
            config["rope_scaling"]["factors"],
            serde_json::json!([1.0, 1.0, 8.0, 32.0])
        );
        // The metadata-only reader cannot see tensors, so it says nothing.
        assert!(
            config_from_gguf(model.metadata_content())
                .unwrap()
                .get("rope_scaling")
                .is_none()
        );
    }

    #[test]
    fn yarn_and_linear_scaling_come_from_the_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let model = llama_model(dir.path(), |b| {
            b.add_string("llama.rope.scaling.type", "yarn")
                .add_f32("llama.rope.scaling.factor", 4.0)
                .add_u32("llama.rope.scaling.original_context_length", 32_768);
        });
        let config = config_from_gguf_model(&model).unwrap();
        assert_eq!(config["rope_scaling"]["rope_type"], "yarn");
        assert_eq!(config["rope_scaling"]["factor"], 4.0);
        assert_eq!(
            config["rope_scaling"]["original_max_position_embeddings"],
            32_768
        );

        let dir = tempfile::tempdir().unwrap();
        let model = llama_model(dir.path(), |b| {
            b.add_string("llama.rope.scaling.type", "linear")
                .add_f32("llama.rope.scaling.factor", 2.0);
        });
        let config = config_from_gguf_model(&model).unwrap();
        assert_eq!(config["rope_scaling"]["rope_type"], "linear");
        assert!(
            config["rope_scaling"]
                .get("original_max_position_embeddings")
                .is_none()
        );
    }

    #[test]
    fn longrope_is_completed_from_its_factor_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let model = llama_model(dir.path(), |b| {
            b.add_string("llama.rope.scaling.type", "longrope")
                .add_u32("llama.rope.scaling.original_context_length", 4096);
            b.add_f32_tensor(
                "rope_factors_short.weight",
                vec![4],
                vec![1.0, 1.5, 2.0, 2.5],
            );
            b.add_f32_tensor(
                "rope_factors_long.weight",
                vec![4],
                vec![2.0, 3.0, 4.0, 5.0],
            );
        });
        let config = config_from_gguf_model(&model).unwrap();
        let scaling = &config["rope_scaling"];
        assert_eq!(scaling["rope_type"], "longrope");
        assert_eq!(
            scaling["short_factor"],
            serde_json::json!([1.0, 1.5, 2.0, 2.5])
        );
        assert_eq!(
            scaling["long_factor"],
            serde_json::json!([2.0, 3.0, 4.0, 5.0])
        );
        assert_eq!(scaling["original_max_position_embeddings"], 4096);
    }

    #[test]
    fn scaling_that_cannot_be_applied_is_an_error_not_silence() {
        let dir = tempfile::tempdir().unwrap();
        let model = llama_model(dir.path(), |b| {
            b.add_string("llama.rope.scaling.type", "ntk-by-parts");
        });
        let err = config_from_gguf_model(&model).unwrap_err().to_string();
        assert!(err.contains("ntk-by-parts"), "{err}");

        // Two scalings for one model is a guess either way.
        let dir = tempfile::tempdir().unwrap();
        let model = llama_model(dir.path(), |b| {
            b.add_string("llama.rope.scaling.type", "linear")
                .add_f32("llama.rope.scaling.factor", 2.0);
            b.add_f32_tensor("rope_freqs.weight", vec![4], vec![1.0; 4]);
        });
        let err = config_from_gguf_model(&model).unwrap_err().to_string();
        assert!(err.contains("rope_freqs"), "{err}");

        // One of the two LongRoPE tables is not a table.
        let dir = tempfile::tempdir().unwrap();
        let model = llama_model(dir.path(), |b| {
            b.add_f32_tensor("rope_factors_short.weight", vec![4], vec![1.0; 4]);
        });
        let err = config_from_gguf_model(&model).unwrap_err().to_string();
        assert!(err.contains("rope_factors_long"), "{err}");
    }

    #[test]
    fn the_vocabulary_is_never_invented() {
        // No `vocab_size`, no token list: the embedding's rows say.
        let dir = tempfile::tempdir().unwrap();
        let model = llama_model(dir.path(), |_| {});
        let config = config_from_gguf_model(&model).unwrap();
        assert_eq!(config["vocab_size"], 100);

        // With nothing to say it, the field is absent, not 32000.
        let bare = GgufContent {
            metadata: [(
                "general.architecture".to_string(),
                crate::MetadataValue::String("llama".into()),
            )]
            .into(),
            ..GgufContent::default()
        };
        let mut cfg = GgufArchConfig::from_gguf(&bare).unwrap();
        assert_eq!(cfg.vocab_size, None);
        assert!(cfg.to_config_json().get("vocab_size").is_none());
        cfg.vocab_size = Some(7);
        assert_eq!(cfg.to_config_json()["vocab_size"], 7);
    }

    #[test]
    fn write_config_does_not_clobber_one_that_exists() {
        let dir = tempfile::tempdir().unwrap();
        let model = llama_model(dir.path(), |_| {});
        let out = tempfile::tempdir().unwrap();
        std::fs::write(out.path().join("config.json"), "{\"kept\": true}").unwrap();
        let path = write_config_from_gguf(&model, out.path()).unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), "{\"kept\": true}");

        let out = tempfile::tempdir().unwrap();
        let path = write_config_from_gguf(&model, out.path()).unwrap();
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(written["hidden_size"], 32);
    }
}
