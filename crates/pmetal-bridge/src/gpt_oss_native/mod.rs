//! Standalone GPT-OSS inference engine — zero dependency on mlx-rs or pmetal-models.
//!
//! GPT-OSS is OpenAI's first Apache-2.0 open-weight model (Aug 2025).  Available
//! in 20B and 120B variants, it uses:
//!   - Mixture of Experts (MoE) with top-k sigmoid routing and per-expert bias
//!   - Alternating sliding window (128 tok) and full-context attention patterns
//!   - GPT-OSS SwiGLU: `x_glu * sigmoid(α * x_glu) * (x_linear + 1)` with clamping
//!   - Grouped Multi-Query Attention (GQA), bias on q/k/v/o projections
//!   - Standard full-head RoPE (head_dim = 64)
//!   - No Q/K norm (unlike Qwen3.5)
//!
//! Every op on the hot path uses [`InlineArray`] (stack-allocated `mlx::core::array`,
//! direct C++ bridge). This eliminates ALL per-op heap allocation, matching
//! Python/nanobind's direct C++ binding performance.
//!
//! The stack is split across focused submodules:
//!   * [`weights`] — layer weight struct, safetensors loading, MXFP4 sanitization
//!   * [`cache`] — per-layer KV caches (sliding / full / quantized-full)
//!   * [`attention`] — attention forward step (three cache paths)
//!   * [`moe`] — sigmoid-routed MoE + clamped SwiGLU activation
//!   * [`forward`] — full-model forward + prefill/prime/generate wrappers

use serde::Deserialize;

mod attention;
mod cache;
mod forward;
mod moe;
mod weights;

pub use cache::{KvLayerCache, NativeCache};
pub use forward::{benchmark_trial, forward_step, generate, prefill_first_token};
pub use weights::{NativeWeights, load_model};

// ============================================================================
// Config
// ============================================================================

fn default_model_type() -> String {
    "gpt_oss".to_string()
}
fn default_vocab_size() -> i32 {
    201088
}
fn default_hidden_size() -> i32 {
    2880
}
fn default_intermediate_size() -> i32 {
    2880
}
fn default_num_hidden_layers() -> i32 {
    24
}
fn default_num_attention_heads() -> i32 {
    64
}
fn default_num_key_value_heads() -> i32 {
    8
}
fn default_head_dim() -> i32 {
    64
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    150000.0
}
fn default_num_local_experts() -> i32 {
    32
}
fn default_experts_per_token() -> i32 {
    4
}
fn default_sliding_window() -> i32 {
    128
}
fn default_true() -> bool {
    true
}
fn default_swiglu_alpha() -> f32 {
    1.702
}
fn default_swiglu_limit() -> f32 {
    7.0
}

/// Attention layer type for GPT-OSS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AttentionLayerType {
    SlidingAttention,
    #[default]
    FullAttention,
}

/// The rope dict transformers' `GptOssConfig` fills in when a config carries
/// none: YaRN, factor 32 over 4096, `truncate: false`, as every release ships.
pub fn default_rope_scaling() -> serde_json::Value {
    serde_json::json!({
        "rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
        "truncate": false, "original_max_position_embeddings": 4096
    })
}

/// Minimal, serde-deserializable GPT-OSS config.
///
/// Only the fields required for inference are included; unknown keys are
/// silently ignored by serde.
#[derive(Debug, Clone, Deserialize)]
pub struct GptOssConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// YaRN on every release; read by [`crate::rope`].
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub max_position_embeddings: Option<i32>,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_num_local_experts")]
    pub num_local_experts: i32,
    /// Primary field for experts-per-token.
    #[serde(default = "default_experts_per_token")]
    pub experts_per_token: i32,
    /// Alternate field name used in some checkpoints.
    #[serde(default)]
    pub num_experts_per_tok: Option<i32>,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    /// Explicit per-layer type list; if empty, alternates sliding/full.
    #[serde(default)]
    pub layer_types: Vec<AttentionLayerType>,
    /// SwiGLU alpha (scaling factor, default 1.702).
    #[serde(default = "default_swiglu_alpha")]
    pub swiglu_alpha: f32,
    /// SwiGLU clamp limit (default 7.0).
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
}

impl GptOssConfig {
    /// The rotary embedding: YaRN with gpt-oss's `truncate: false` and its
    /// attention factor on q and k (transformers' [`default_rope_scaling`]
    /// when the config has no rope dict); an unknown `rope_type` is an error
    /// naming it.
    pub fn rotary(&self) -> Result<crate::rope::RotaryEmbedding, String> {
        let default = default_rope_scaling();
        crate::rope::RotaryEmbedding::from_config(
            self.head_dim,
            crate::rope::RopeConfig {
                rope_scaling: Some(self.rope_scaling.as_ref().unwrap_or(&default)),
                rope_theta: Some(self.rope_theta as f64),
                max_position_embeddings: self.max_position_embeddings.map(f64::from),
                ..crate::rope::RopeConfig::default()
            },
            self.rope_theta as f64,
            1.0,
            false,
        )
        .map_err(|e| format!("gpt_oss config: {e}"))
    }

    /// Effective experts per token.
    pub fn experts_per_tok(&self) -> i32 {
        self.num_experts_per_tok.unwrap_or(self.experts_per_token)
    }

    /// Attention type at layer index `i`.
    pub fn layer_type(&self, i: usize) -> AttentionLayerType {
        if !self.layer_types.is_empty() && i < self.layer_types.len() {
            self.layer_types[i]
        } else {
            // Default: even indices are sliding, odd are full.
            if i % 2 == 0 {
                AttentionLayerType::SlidingAttention
            } else {
                AttentionLayerType::FullAttention
            }
        }
    }

    /// Index of the first full-attention layer (used for causal-mask build).
    pub fn first_full_attn_layer(&self) -> usize {
        for i in 0..self.num_hidden_layers as usize {
            if self.layer_type(i) == AttentionLayerType::FullAttention {
                return i;
            }
        }
        0
    }

    /// Index of the first sliding-attention layer.
    pub fn first_sliding_attn_layer(&self) -> usize {
        for i in 0..self.num_hidden_layers as usize {
            if self.layer_type(i) == AttentionLayerType::SlidingAttention {
                return i;
            }
        }
        0
    }
}

/// Parse `config.json` from a model directory.
pub fn load_config(model_dir: &std::path::Path) -> Result<GptOssConfig, String> {
    let text = crate::native_loader::read_config_json(model_dir)?;
    // Some checkpoints nest config under "text_config"
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("failed to parse config.json: {e}"))?;
    let config_str = if json.get("text_config").is_some() {
        serde_json::to_string(&json["text_config"]).map_err(|e| e.to_string())?
    } else {
        text
    };
    serde_json::from_str(&config_str).map_err(|e| format!("failed to parse config: {e}"))
}

#[cfg(test)]
mod tests {
    use super::GptOssConfig;
    use crate::rope::{RopeConfig, Rotary};

    /// The native engine rotated every gpt-oss with plain RoPE at
    /// `rope_theta`. It now runs the release's YaRN (`truncate: false`) and
    /// the attention factor, read the way every other engine reads it; a
    /// config with no rope dict gets transformers' gpt-oss default, and an
    /// unknown type is refused by name.
    #[test]
    fn native_gpt_oss_rotates_with_the_released_yarn() {
        let released = serde_json::json!({
            "head_dim": 64, "rope_theta": 150000.0, "max_position_embeddings": 131072,
            "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0,
                "original_max_position_embeddings": 4096, "rope_type": "yarn", "truncate": false}
        });
        let want = Rotary::from_config(64, RopeConfig::from_json(&released), 1e4, 1.0).unwrap();
        let config: GptOssConfig = serde_json::from_value(released.clone()).unwrap();
        let rotary = config.rotary().unwrap();
        assert_eq!(rotary.rotary(), &want);
        assert!(
            rotary.scalar().is_none(),
            "the compiled scalar-base path must step aside"
        );

        let mut bare = config.clone();
        bare.rope_scaling = None;
        assert_eq!(bare.rotary().unwrap().rotary(), &want);

        bare.rope_scaling = Some(serde_json::json!({"rope_type": "xpos"}));
        assert!(bare.rotary().unwrap_err().contains("xpos"));
    }
}
