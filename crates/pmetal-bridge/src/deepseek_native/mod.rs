//! Standalone DeepSeek V3/R1 inference engine, independent of pmetal-models.
//!
//! Implements Multi-head Latent Attention (MLA), the defining innovation of
//! DeepSeek V3. Instead of caching full K,V tensors, MLA caches a compressed
//! latent vector `c_kv` (shape `[B, 1, T, kv_lora_rank]`) and `k_pe` (shape
//! `[B, 1, T, qk_rope_head_dim]`). K and V are reconstructed on the fly
//! during each attention step via per-head linear projections.
//!
//! MoE routing uses the `noaux_tc` group-aware top-k method with sigmoid
//! scoring and auxiliary-loss-free load balancing (e_score_correction_bias).
//!
//! Every op on the hot path uses [`InlineArray`] — no per-op heap allocation.
//!
//! The stack is split across focused submodules:
//!   * [`weights`] — layer weight struct, safetensors loading, FP8 dequant, expert stacking
//!   * [`cache`] — MLA latent + k_pe cache (bf16 + affine-quantized variants)
//!   * [`attention`] — MLA forward (decode absorbs embed_q; prefill expands K/V)
//!   * [`moe`] — dense SwiGLU MLP + group-aware noaux_tc top-k MoE
//!   * [`forward`] — full-model forward + prefill/prime/generate wrappers

use serde::Deserialize;

mod attention;
mod cache;
mod forward;
mod moe;
mod weights;

pub use cache::{MlaLayerCache, NativeCache};
pub use forward::{benchmark_trial, forward_step, generate, prefill_first_token};
pub use weights::{NativeWeights, load_model};

// ============================================================================
// Config
// ============================================================================

fn default_vocab_size() -> i32 {
    102400
}
fn default_hidden_size() -> i32 {
    7168
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}
fn default_norm_topk_prob() -> bool {
    true
}
fn default_n_group() -> i32 {
    1
}
fn default_topk_group() -> i32 {
    1
}
fn default_false() -> bool {
    false
}
fn default_moe_layer_freq() -> i32 {
    1
}
fn default_first_k_dense_replace() -> i32 {
    0
}
fn default_model_type() -> String {
    "deepseek_v3".to_string()
}

/// Minimal, serde-deserializable DeepSeek V3/R1 config.
///
/// Only the fields required for inference are included; unknown keys are
/// silently ignored by serde.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,

    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,

    pub intermediate_size: i32,

    // MoE
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default)]
    pub n_routed_experts: Option<i32>,
    #[serde(default)]
    pub n_shared_experts: Option<i32>,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_n_group")]
    pub n_group: i32,
    #[serde(default = "default_topk_group")]
    pub topk_group: i32,
    pub num_experts_per_tok: i32,
    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: i32,
    #[serde(default = "default_first_k_dense_replace")]
    pub first_k_dense_replace: i32,

    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,

    #[serde(default)]
    pub num_key_value_heads: Option<i32>,

    // MLA dimensions
    pub kv_lora_rank: i32,
    #[serde(default)]
    pub q_lora_rank: Option<i32>,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,
    pub qk_nope_head_dim: i32,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    #[serde(default)]
    pub max_position_embeddings: Option<i32>,

    #[serde(default = "default_false")]
    pub attention_bias: bool,

    #[serde(default = "default_false")]
    pub tie_word_embeddings: bool,
}

impl DeepSeekConfig {
    /// Total Q head dimension = nope + rope.
    pub fn q_head_dim(&self) -> i32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// The interleaved rotary embedding of the `qk_rope_head_dim` channels,
    /// YaRN as `rope_scaling` gives it (frequencies and attention factor); an
    /// unknown `rope_type` is an error naming it.
    pub fn rotary(&self) -> Result<crate::rope::RotaryEmbedding, String> {
        crate::rope::RotaryEmbedding::from_config(
            self.qk_rope_head_dim,
            crate::rope::RopeConfig {
                rope_scaling: self.rope_scaling.as_ref(),
                rope_theta: Some(self.rope_theta),
                max_position_embeddings: self.max_position_embeddings.map(f64::from),
                ..crate::rope::RopeConfig::default()
            },
            self.rope_theta,
            1.0,
            true,
        )
        .map_err(|e| format!("deepseek config: {e}"))
    }

    /// Softmax scale — applied to Q before computing scores: `q_head_dim^-½`
    /// times YaRN's `get_mscale(factor, mscale_all_dim)²` when configured
    /// (see [`crate::rope::RopeScaling::mla_softmax_mscale`]).
    pub fn attention_scale(&self, rotary: &crate::rope::RotaryEmbedding) -> f32 {
        (self.q_head_dim() as f32).powf(-0.5) * rotary.rotary().scaling.mla_softmax_mscale() as f32
    }

    /// Returns true when layer `layer_id` uses MoE instead of dense MLP.
    pub fn is_moe_layer(&self, layer_id: usize) -> bool {
        if self.n_routed_experts.is_none() {
            return false;
        }
        let li = layer_id as i32;
        li >= self.first_k_dense_replace && li % self.moe_layer_freq == 0
    }
}

/// Parse `config.json` from a model directory.
pub fn load_config(model_dir: &std::path::Path) -> Result<DeepSeekConfig, String> {
    let text = crate::native_loader::read_config_json(model_dir)?;
    let cfg: DeepSeekConfig =
        serde_json::from_str(&text).map_err(|e| format!("failed to parse config.json: {e}"))?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::DeepSeekConfig;

    /// DeepSeek-V3's released attention config: YaRN factor 40 over 4096
    /// with `mscale = mscale_all_dim = 1`. The native engine used to rotate
    /// with plain RoPE and apply only the softmax mscale²; it now rotates
    /// with transformers' YaRN frequencies (the shared fixture's
    /// `yarn_deepseek_v3` case) and keeps the softmax term.
    #[test]
    fn deepseek_v3_native_rotates_with_yarn() {
        let config: DeepSeekConfig = serde_json::from_value(serde_json::json!({
            "intermediate_size": 64, "num_experts_per_tok": 2, "num_hidden_layers": 1,
            "num_attention_heads": 2, "kv_lora_rank": 16, "qk_rope_head_dim": 64,
            "v_head_dim": 32, "qk_nope_head_dim": 128, "rope_theta": 10000,
            "max_position_embeddings": 163840,
            "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 1.0,
                "mscale_all_dim": 1.0, "original_max_position_embeddings": 4096, "type": "yarn"}
        }))
        .unwrap();
        let rotary = config.rotary().unwrap();
        assert!(rotary.traditional());
        assert_eq!(rotary.attention_factor(), 1.0);
        let mscale = 0.1 * 40f32.ln() + 1.0;
        let want_scale = 192f32.powf(-0.5) * mscale * mscale;
        assert!((config.attention_scale(&rotary) - want_scale).abs() < 1e-6);

        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/rope_scaling_reference.json"
        ))
        .unwrap();
        let case = reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "yarn_deepseek_v3")
            .unwrap();
        let got = rotary.inverse_frequencies(0);
        for (g, w) in got.iter().zip(case["inv_freq"].as_array().unwrap()) {
            let w = w.as_f64().unwrap();
            assert!(((*g as f64) - w).abs() <= 2e-6 * w, "{g} vs {w}");
        }
    }
}
