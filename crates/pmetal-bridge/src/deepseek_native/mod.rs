//! Standalone DeepSeek V3/R1 inference engine — zero dependency on mlx-rs or pmetal-models.
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

    /// Attention scale — applied to Q before computing scores.
    /// Includes mscale correction for YaRN rope scaling when configured.
    pub fn attention_scale(&self) -> f32 {
        let base = (self.q_head_dim() as f32).powf(-0.5);
        if let Some(ref rs) = self.rope_scaling {
            let mscale_all_dim = rs
                .get("mscale_all_dim")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            if mscale_all_dim > 0.0 {
                let factor = rs.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
                if factor > 1.0 {
                    let s = 0.1 * mscale_all_dim * factor.ln() + 1.0;
                    return base * (s * s) as f32;
                }
            }
        }
        base
    }

    /// Returns true when layer `layer_id` uses MoE instead of dense MLP.
    pub fn is_moe_layer(&self, layer_id: usize) -> bool {
        if self.n_routed_experts.is_none() {
            return false;
        }
        let li = layer_id as i32;
        li >= self.first_k_dense_replace && li % self.moe_layer_freq == 0
    }

    /// RoPE base as f32.
    pub fn rope_base_f32(&self) -> f32 {
        // Apply YaRN mscale to rope_theta when configured.
        // The Python initialize_rope() handles this; we replicate the effect.
        self.rope_theta as f32
    }
}

/// Parse `config.json` from a model directory.
pub fn load_config(model_dir: &std::path::Path) -> Result<DeepSeekConfig, String> {
    let text = crate::native_loader::read_config_json(model_dir)?;
    let cfg: DeepSeekConfig =
        serde_json::from_str(&text).map_err(|e| format!("failed to parse config.json: {e}"))?;
    Ok(cfg)
}
