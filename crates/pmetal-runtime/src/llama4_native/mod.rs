//! Standalone Llama 4 inference engine, independent of pmetal-models.
//!
//! Every op on the hot path uses [`InlineArray`](crate::InlineArray) (stack-allocated `mlx::core::array`,
//! direct C++ bridge). This eliminates ALL per-op heap allocation, matching
//! Python/nanobind's direct C++ binding performance.
//!
//! Architecture details (from the reference implementation):
//!
//! - **iRoPE**: interleaved positional encoding.
//!   - Layers where `(layer_idx + 1) % 4 != 0` are "local" — use chunked attention
//!     with full RoPE (traditional=true) and optional QK-norm.
//!   - Layers where `(layer_idx + 1) % 4 == 0` are "global" — use full causal attention
//!     with NoPE (no positional encoding) and attention temperature tuning.
//!
//! - **MoE**: `interleave_moe_layer_step` controls which layers are MoE.
//!   `(layer_idx % step) == (step - 1)` → MoE layer; others are dense MLP.
//!   Top-1 routing with sigmoid scores on the expert's input, shared expert
//!   added after routed output. Expert weights are held `[num_experts, in, out]`
//!   for `gather_mm`, which is how transformers' fused `gate_up_proj` /
//!   `down_proj` already store them (the former split into gate and up).
//!
//! The stack is split across focused submodules:
//!   * `weights` — layer weight struct, safetensors loading, expert sanitization
//!   * `cache` — per-layer KV caches (bf16 + zero-overhead-quantized)
//!   * `attention` — iRoPE attention + chunk mask + temperature tuning
//!   * `moe` — dense SwiGLU MLP + top-1 SwitchGLU with shared expert
//!   * `forward` — full-model forward + prefill/prime/generate wrappers

use serde::Deserialize;

mod attention;
mod cache;
mod forward;
mod moe;
mod weights;

pub use cache::{KvLayerCache, NativeCache};
pub use forward::{
    benchmark_trial, forward_step, forward_step_last, generate, prefill_first_token,
};
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

/// A JSON bool (`true` → 1) or integer.
fn bool_or_int<'de, D: serde::Deserializer<'de>>(d: D) -> Result<i32, D::Error> {
    match serde_json::Value::deserialize(d)? {
        serde_json::Value::Bool(b) => Ok(i32::from(b)),
        serde_json::Value::Number(n) => n
            .as_i64()
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| serde::de::Error::custom(format!("expected an integer, got {n}"))),
        other => Err(serde::de::Error::custom(format!(
            "expected a bool or an integer, got {other}"
        ))),
    }
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

    /// Llama 4 Scout ships `rope_type: "llama3"` (factor 16), read by
    /// [`crate::rope`].
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

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

    /// Attention temperature tuning for NoPE (global) layers: on when
    /// positive. transformers' configs (Scout's included) say `true`; older
    /// ones gave a number.
    #[serde(
        default = "default_attn_temperature_tuning",
        deserialize_with = "bool_or_int"
    )]
    pub attn_temperature_tuning: i32,

    /// The MoE layers, when listed (Scout lists every layer). Without it
    /// `interleave_moe_layer_step` decides.
    #[serde(default)]
    pub moe_layers: Option<Vec<i32>>,

    /// Per layer, 1 for RoPE and 0 for NoPE, when listed. Without it every
    /// fourth layer is NoPE.
    #[serde(default)]
    pub no_rope_layers: Option<Vec<i32>>,

    #[serde(default = "default_floor_scale")]
    pub floor_scale: i32,

    #[serde(default = "default_attn_scale")]
    pub attn_scale: f32,

    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// Model config: the text model's [`Llama4TextConfig`] and the outer
/// `model_type`.
///
/// A multimodal checkpoint nests the text model under `text_config` and keeps
/// `tie_word_embeddings` (and the vision settings) at the top; a text-only
/// one is flat.
#[derive(Debug, Clone)]
pub struct Llama4Config {
    pub model_type: String,
    text: Llama4TextConfig,
}

impl Llama4Config {
    /// Read a `config.json` value, nested or flat.
    ///
    /// `tie_word_embeddings` is the one text-level setting a nested config
    /// keeps outside `text_config`; it applies when the text config does not
    /// state it.
    pub fn from_json(json: &serde_json::Value) -> Result<Self, String> {
        let mut text = json.get("text_config").unwrap_or(json).clone();
        if let (Some(obj), Some(tie)) = (text.as_object_mut(), json.get("tie_word_embeddings")) {
            obj.entry("tie_word_embeddings")
                .or_insert_with(|| tie.clone());
        }
        let text: Llama4TextConfig = serde_json::from_value(text)
            .map_err(|e| format!("failed to parse Llama4Config: {e}"))?;
        if text.num_experts_per_tok < 1 || text.num_experts_per_tok > text.num_local_experts {
            return Err(format!(
                "llama4 config: num_experts_per_tok {} does not fit num_local_experts {}",
                text.num_experts_per_tok, text.num_local_experts
            ));
        }
        if text.attention_chunk_size < 1 {
            return Err(format!(
                "llama4 config: attention_chunk_size {} is not positive",
                text.attention_chunk_size
            ));
        }
        let model_type = json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("llama4")
            .to_string();
        Ok(Self { model_type, text })
    }

    /// The text model's config.
    pub fn text(&self) -> &Llama4TextConfig {
        &self.text
    }

    /// Returns `true` when layer `li` (0-indexed) is a MoE layer: listed in
    /// `moe_layers`, or else
    /// `(layer_idx % interleave_moe_layer_step) == (interleave_moe_layer_step - 1)`.
    pub fn is_moe_layer(&self, li: usize) -> bool {
        let t = self.text();
        match &t.moe_layers {
            Some(layers) => layers.contains(&(li as i32)),
            None => {
                let step = t.interleave_moe_layer_step;
                (li as i32 % step) == (step - 1)
            }
        }
    }

    /// Returns `true` when layer `li` (0-indexed) uses RoPE (local / chunked
    /// attention): its `no_rope_layers` entry, or else
    /// `(layer_idx + 1) % 4 != 0`.
    pub fn use_rope(&self, li: usize) -> bool {
        match self.text().no_rope_layers.as_ref().and_then(|l| l.get(li)) {
            Some(&flag) => flag != 0,
            None => ((li as i32) + 1) % 4 != 0,
        }
    }

    /// Head dimension.
    pub fn head_dim(&self) -> i32 {
        let t = self.text();
        t.head_dim.unwrap_or(t.hidden_size / t.num_attention_heads)
    }

    /// The interleaved rotary embedding of the RoPE layers, with the Llama 3
    /// frequency bands `rope_scaling` gives (Scout: factor 16); an unknown
    /// `rope_type` is an error naming it.
    pub fn rotary(&self) -> Result<crate::rope::RotaryEmbedding, String> {
        let t = self.text();
        crate::rope::RotaryEmbedding::from_config(
            self.head_dim(),
            crate::rope::RopeConfig {
                rope_scaling: t.rope_scaling.as_ref(),
                rope_theta: Some(t.rope_theta),
                max_position_embeddings: Some(t.max_position_embeddings as f64),
                ..crate::rope::RopeConfig::default()
            },
            t.rope_theta,
            1.0,
            true,
        )
        .map_err(|e| format!("llama4 config: {e}"))
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
    Llama4Config::from_json(&json)
}

#[cfg(test)]
mod tests {
    use super::Llama4Config;

    /// transformers' configs, Scout's included, say
    /// `"attn_temperature_tuning": true` and list `moe_layers` and
    /// `no_rope_layers`; the engine refused the bool and ignored the lists.
    #[test]
    fn released_config_keys_are_read() {
        let config = Llama4Config::from_json(&serde_json::json!({
            "model_type": "llama4_text", "hidden_size": 32, "num_hidden_layers": 4,
            "num_attention_heads": 4, "head_dim": 8, "intermediate_size_mlp": 48,
            "intermediate_size": 48, "vocab_size": 64, "attention_chunk_size": 64,
            "num_local_experts": 2, "num_experts_per_tok": 1, "max_position_embeddings": 64,
            "attn_temperature_tuning": true, "moe_layers": [], "no_rope_layers": [1, 0, 1, 1]
        }))
        .unwrap();
        assert_eq!(config.text().attn_temperature_tuning, 1);
        assert!((0..4).all(|l| !config.is_moe_layer(l)));
        assert!(!config.use_rope(1) && config.use_rope(3));

        let numeric = Llama4Config::from_json(&serde_json::json!({
            "model_type": "llama4_text", "hidden_size": 32, "num_hidden_layers": 4,
            "num_attention_heads": 4, "intermediate_size_mlp": 48, "intermediate_size": 48,
            "vocab_size": 64, "attention_chunk_size": 64, "num_local_experts": 2,
            "num_experts_per_tok": 1, "max_position_embeddings": 64,
            "attn_temperature_tuning": 0
        }))
        .unwrap();
        assert_eq!(numeric.text().attn_temperature_tuning, 0);
        assert!(numeric.is_moe_layer(0) && !numeric.use_rope(3));
    }

    /// The released Scout `config.json` (multimodal: the text model nests under
    /// `text_config`, `tie_word_embeddings` and nothing else text-level sits
    /// at the top) loads into the engine's text config.
    #[test]
    fn released_scout_config_loads() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            include_str!("../../tests/fixtures/llama4_scout_config.json"),
        )
        .unwrap();
        let config = super::load_config(dir.path()).expect("released Scout config");
        let t = config.text();
        assert_eq!(
            (t.hidden_size, t.num_hidden_layers, t.num_attention_heads),
            (5120, 48, 40)
        );
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.head_dim(), 128);
        assert!((0..48).all(|l| config.is_moe_layer(l)));
        assert!((0..48).all(|l| config.use_rope(l) == (l % 4 != 3)));
        assert_eq!(t.attn_temperature_tuning, 1);
        assert!(!t.tie_word_embeddings);
    }

    /// A router that picks more experts than exist, or a chunk of no tokens,
    /// is refused by name rather than run.
    #[test]
    fn impossible_routing_and_chunks_are_refused() {
        let with = |k: i32, chunk: i32| {
            super::Llama4Config::from_json(&serde_json::json!({
                "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4,
                "intermediate_size_mlp": 48, "intermediate_size": 48, "vocab_size": 64,
                "attention_chunk_size": chunk, "num_local_experts": 4, "num_experts_per_tok": k,
                "max_position_embeddings": 64
            }))
        };
        assert!(with(2, 8).is_ok());
        assert!(with(0, 8).unwrap_err().contains("num_experts_per_tok"));
        assert!(with(5, 8).unwrap_err().contains("num_experts_per_tok"));
        assert!(with(1, 0).unwrap_err().contains("attention_chunk_size"));
    }

    /// `tie_word_embeddings` sits outside `text_config` in a nested config and
    /// applies to the text model unless it states its own.
    #[test]
    fn outer_tie_word_embeddings_reaches_the_text_config() {
        let text = serde_json::json!({
            "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4,
            "intermediate_size_mlp": 48, "intermediate_size": 48, "vocab_size": 64,
            "attention_chunk_size": 64, "num_local_experts": 2, "num_experts_per_tok": 1,
            "max_position_embeddings": 64
        });
        let nested = |outer: bool, inner: Option<bool>| {
            let mut t = text.clone();
            if let Some(inner) = inner {
                t["tie_word_embeddings"] = inner.into();
            }
            super::Llama4Config::from_json(&serde_json::json!({
                "model_type": "llama4", "tie_word_embeddings": outer, "text_config": t
            }))
            .unwrap()
            .text()
            .tie_word_embeddings
        };
        assert!(nested(true, None));
        assert!(!nested(false, None));
        assert!(!nested(true, Some(false)));
        assert!(nested(false, Some(true)));
    }

    /// Llama 4 Scout ships `rope_type: "llama3"` (factor 16, low and high
    /// band factors both 1). The native engine ran it as plain RoPE; it now
    /// rotates with transformers' bands (the shared fixture's
    /// `llama4_scout` case), interleaved, and hands the compiled decode
    /// graph the bands' period table.
    #[test]
    fn scout_rotates_with_the_llama3_bands() {
        let config = Llama4Config::from_json(&serde_json::json!({
            "model_type": "llama4_text", "hidden_size": 256, "num_hidden_layers": 4,
            "num_attention_heads": 2, "head_dim": 128, "intermediate_size_mlp": 64,
            "intermediate_size": 64, "vocab_size": 32, "attention_chunk_size": 8192,
            "num_local_experts": 2, "num_experts_per_tok": 1,
            "max_position_embeddings": 10485760, "rope_theta": 500000.0,
            "rope_scaling": {"factor": 16.0, "high_freq_factor": 1.0, "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192, "rope_type": "llama3"}
        }))
        .unwrap();
        let rotary = config.rotary().unwrap();
        assert!(rotary.traditional());
        assert!(rotary.scalar().is_none());
        let kernel = rotary.fixed_kernel(1).expect("the bands are static");
        assert!(kernel.periods.is_some() && kernel.gain.is_none());
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/rope_scaling_reference.json"
        ))
        .unwrap();
        let case = reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "llama4_scout")
            .unwrap();
        let got = rotary.inverse_frequencies(0);
        for (g, w) in got.iter().zip(case["inv_freq"].as_array().unwrap()) {
            let w = w.as_f64().unwrap();
            assert!(((*g as f64) - w).abs() <= 2e-6 * w, "{g} vs {w}");
        }
    }
}
