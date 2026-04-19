//! Standalone Llama 4 inference engine — zero dependency on mlx-rs or pmetal-models.
//!
//! Every op on the hot path uses [`InlineArray`] (stack-allocated `mlx::core::array`,
//! direct C++ bridge). This eliminates ALL per-op heap allocation, matching
//! Python/nanobind's direct C++ binding performance.
//!
//! Architecture details (from mlx-lm llama4.py):
//!
//! - **iRoPE**: interleaved positional encoding.
//!   - Layers where `(layer_idx + 1) % 4 != 0` are "local" — use chunked attention
//!     with full RoPE (traditional=true) and optional QK-norm.
//!   - Layers where `(layer_idx + 1) % 4 == 0` are "global" — use full causal attention
//!     with NoPE (no positional encoding) and attention temperature tuning.
//!
//! - **MoE**: `interleave_moe_layer_step` controls which layers are MoE.
//!   `(layer_idx % step) == (step - 1)` → MoE layer; others are dense MLP.
//!   Top-1 routing with sigmoid scores, shared expert added after routed output.
//!   Expert weights stored as `[num_experts, out, in]` (SwitchLinear convention);
//!   sanitization splits and transposes the gate_up_proj block from safetensors.
//!
//! The stack is split across focused submodules:
//!   * [`weights`] — layer weight struct, safetensors loading, expert sanitization
//!   * [`cache`] — per-layer KV caches (bf16 + zero-overhead-quantized)
//!   * [`attention`] — iRoPE attention + chunk mask + temperature tuning
//!   * [`moe`] — dense SwiGLU MLP + top-1 SwitchGLU with shared expert
//!   * [`forward`] — full-model forward + prefill/prime/generate wrappers

use serde::Deserialize;

mod attention;
mod cache;
mod forward;
mod moe;
mod weights;

pub use cache::{KvLayerCache, NativeCache};
pub use forward::{benchmark_mlx_lm_trial, forward_step, generate, prefill_first_token};
pub use weights::{NativeWeights, load_model};

// ============================================================================
// Config
// ============================================================================

fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f64 {
    500_000.0
}
fn default_attn_temperature_tuning() -> i32 {
    4
}
fn default_floor_scale() -> i32 {
    8192
}
fn default_attn_scale() -> f32 {
    0.1
}
fn default_interleave_moe_layer_step() -> i32 {
    1
}
fn default_true() -> bool {
    true
}

/// Nested rope_scaling config — only a handful of fields matter for iRoPE.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeScalingConfig {
    #[serde(rename = "type", default)]
    pub scaling_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// Text-level config (lives under `text_config` in the outer JSON, or at the
/// top level for text-only models).
#[derive(Debug, Clone, Deserialize)]
pub struct Llama4TextConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,

    #[serde(default)]
    pub num_key_value_heads: Option<i32>,

    #[serde(default)]
    pub head_dim: Option<i32>,

    /// Dense MLP intermediate size (used by non-MoE layers).
    pub intermediate_size_mlp: i32,

    /// MoE expert hidden size.
    pub intermediate_size: i32,

    pub vocab_size: i32,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    /// Chunk size for local (chunked) attention — matches Python's `attention_chunk_size`.
    pub attention_chunk_size: i32,

    /// Step between MoE layers: `(layer_idx % step) == (step - 1)` is a MoE layer.
    #[serde(default = "default_interleave_moe_layer_step")]
    pub interleave_moe_layer_step: i32,

    pub num_local_experts: i32,
    pub num_experts_per_tok: i32,

    /// Whether QK-norm is applied (only on RoPE/local layers).
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,

    /// Whether attention projection biases are used.
    #[serde(default)]
    pub attention_bias: bool,

    pub max_position_embeddings: i32,

    /// Attention temperature tuning for NoPE (global) layers.
    #[serde(default = "default_attn_temperature_tuning")]
    pub attn_temperature_tuning: i32,

    #[serde(default = "default_floor_scale")]
    pub floor_scale: i32,

    #[serde(default = "default_attn_scale")]
    pub attn_scale: f32,

    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// Outer model config — Llama 4 nests text config under `text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct Llama4Config {
    pub model_type: String,

    /// Present in multi-modal checkpoint JSON.
    #[serde(default)]
    pub text_config: Option<Llama4TextConfig>,

    /// Flattened text config fields — present in text-only configs.
    #[serde(flatten)]
    pub text: Llama4TextConfig,
}

impl Llama4Config {
    /// Resolve the effective text config regardless of nesting.
    pub fn text(&self) -> &Llama4TextConfig {
        self.text_config.as_ref().unwrap_or(&self.text)
    }

    /// Returns `true` when layer `li` (0-indexed) is a MoE layer.
    ///
    /// Python: `(layer_idx % interleave_moe_layer_step) == (interleave_moe_layer_step - 1)`
    pub fn is_moe_layer(&self, li: usize) -> bool {
        let step = self.text().interleave_moe_layer_step;
        (li as i32 % step) == (step - 1)
    }

    /// Returns `true` when layer `li` (0-indexed) uses RoPE (local / chunked attention).
    ///
    /// Python: `use_rope = int((layer_idx + 1) % 4 != 0)`
    pub fn use_rope(&self, li: usize) -> bool {
        ((li as i32) + 1) % 4 != 0
    }

    /// Head dimension.
    pub fn head_dim(&self) -> i32 {
        let t = self.text();
        t.head_dim.unwrap_or(t.hidden_size / t.num_attention_heads)
    }

    /// Number of KV heads.
    pub fn num_kv_heads(&self) -> i32 {
        let t = self.text();
        t.num_key_value_heads.unwrap_or(t.num_attention_heads)
    }
}

/// Parse `config.json` from a model directory.
///
/// Handles both the flat text-only layout and the multi-modal layout where
/// `text_config` is nested.
pub fn load_config(model_dir: &std::path::Path) -> Result<Llama4Config, String> {
    let text = crate::native_loader::read_config_json(model_dir)?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("failed to parse config.json: {e}"))?;

    // If the outer config has `text_config`, try deserializing that inner object
    // as the canonical text config and wrap it.  Otherwise deserialize the
    // whole JSON as a flat Llama4Config.
    if json.get("text_config").is_some() {
        let cfg: Llama4Config = serde_json::from_str(&text)
            .map_err(|e| format!("failed to parse Llama4Config: {e}"))?;
        Ok(cfg)
    } else {
        // Flat layout: embed a synthetic `model_type` field if absent, then
        // deserialize the whole object as both outer + inner config.
        let mut obj = json.clone();
        if !obj
            .as_object()
            .map(|o| o.contains_key("model_type"))
            .unwrap_or(false)
        {
            obj["model_type"] = serde_json::Value::String("llama4".to_string());
        }
        let config_str = serde_json::to_string(&obj).map_err(|e| e.to_string())?;
        let cfg: Llama4Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("failed to parse Llama4Config (flat): {e}"))?;
        Ok(cfg)
    }
}
