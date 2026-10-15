//! Standalone DeepSeek V2 / V3 / R1 inference engine, independent of pmetal-models.
//!
//! Implements Multi-head Latent Attention (MLA), the defining innovation of
//! DeepSeek V2. Instead of caching full K,V tensors, MLA caches a compressed
//! latent vector `c_kv` (shape `[B, 1, T, kv_lora_rank]`) and `k_pe` (shape
//! `[B, 1, T, qk_rope_head_dim]`). K and V are reconstructed on the fly
//! during each attention step via per-head linear projections.
//!
//! MoE routing follows the config's `scoring_func` and `topk_method`: V3's
//! `noaux_tc` (sigmoid scores, an auxiliary-loss-free correction bias, group-
//! limited top-k, normalised and scaled weights) and V2's softmax with the
//! plain (`greedy`) or group-limited (`group_limited_greedy`) top-k. The
//! routed experts and every projection run packed when the checkpoint is an
//! MLX quantization.
//!
//! Every op on the hot path uses [`InlineArray`](crate::InlineArray) — no per-op heap allocation.
//!
//! The stack is split across focused submodules:
//!   * `weights` — layer weight structs and the checkpoint loader (dense or packed)
//!   * `cache` — MLA latent + k_pe cache (bf16 + affine-quantized variants)
//!   * `attention` — MLA forward (decode absorbs embed_q; prefill expands K/V)
//!   * `moe` — dense SwiGLU MLP + the router and routed experts
//!   * `forward` — full-model forward + prefill/prime/generate wrappers

use serde::Deserialize;

mod attention;
mod cache;
mod forward;
mod moe;
mod weights;

pub use cache::{MlaLayerCache, NativeCache};
pub use forward::{benchmark_trial, forward_step, generate, prefill_first_token};
pub use moe::{ScoringFunc, TopkMethod};
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
fn default_true() -> bool {
    true
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

/// Minimal, serde-deserializable DeepSeek V2 / V3 / R1 config.
///
/// Only the fields required for inference are included; unknown keys are
/// silently ignored by serde, and [`load_config`] refuses the ones that name a
/// feature this engine does not implement.
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
    /// Expert groups the group-limited top-k picks among; absent or `null`
    /// is 1.
    #[serde(default)]
    pub n_group: Option<i32>,
    /// Groups kept; absent or `null` is 1.
    #[serde(default)]
    pub topk_group: Option<i32>,
    /// `sigmoid` (V3) or `softmax` (V2). Absent: the model type's.
    #[serde(default)]
    pub scoring_func: Option<ScoringFunc>,
    /// `noaux_tc` (V3), `greedy` or `group_limited_greedy` (V2). Absent: the
    /// model type's.
    #[serde(default)]
    pub topk_method: Option<TopkMethod>,
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

    /// Rotate adjacent channel pairs (DeepSeek's layout, and every release's)
    /// rather than the two halves; transformers' `rope_interleave`.
    #[serde(default = "default_true")]
    pub rope_interleave: bool,

    #[serde(default)]
    pub max_position_embeddings: Option<i32>,

    #[serde(default)]
    pub hidden_act: Option<String>,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default)]
    pub mlp_bias: bool,

    /// DeepSeek-V3.2's sparse-attention indexer; present means a model this
    /// engine would run as dense MLA, which is a different model.
    #[serde(default)]
    pub index_topk: Option<i32>,

    #[serde(default)]
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
            self.rope_interleave,
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

    /// Whether this is a V2-family config (softmax routing unless it says
    /// otherwise).
    fn is_v2(&self) -> bool {
        self.model_type == "deepseek_v2"
    }

    /// How expert scores are computed.
    pub fn scoring(&self) -> ScoringFunc {
        self.scoring_func.unwrap_or(if self.is_v2() {
            ScoringFunc::Softmax
        } else {
            ScoringFunc::Sigmoid
        })
    }

    /// How the top-k experts are chosen.
    pub fn topk(&self) -> TopkMethod {
        self.topk_method.unwrap_or(if self.is_v2() {
            TopkMethod::Greedy
        } else {
            TopkMethod::NoauxTc
        })
    }

    /// Expert groups (1 when the config has none).
    pub fn n_group(&self) -> i32 {
        self.n_group.unwrap_or(1)
    }

    /// Groups kept by the group-limited top-k (1 when the config has none).
    pub fn topk_group(&self) -> i32 {
        self.topk_group.unwrap_or(1)
    }

    /// Refuse a config this engine would run as a different model.
    pub fn validate(&self) -> Result<(), String> {
        let bad = |what: String| Err(format!("deepseek config: {what}"));
        if self.index_topk.is_some() {
            return bad(
                "`index_topk` marks DeepSeek-V3.2's sparse-attention indexer, which \
                        the native engine does not implement"
                    .into(),
            );
        }
        if let Some(act) = &self.hidden_act
            && act != "silu"
        {
            return bad(format!("hidden_act `{act}` is not supported (silu only)"));
        }
        if self.attention_bias {
            return bad("`attention_bias: true` is not supported".into());
        }
        if self.mlp_bias {
            return bad("`mlp_bias: true` is not supported".into());
        }
        let Some(experts) = self.n_routed_experts else {
            return Ok(());
        };
        let (groups, kept, top_k) = (self.n_group(), self.topk_group(), self.num_experts_per_tok);
        if experts < 1 || top_k < 1 || top_k > experts {
            return bad(format!(
                "num_experts_per_tok {top_k} does not fit n_routed_experts {experts}"
            ));
        }
        if self.topk() != TopkMethod::Greedy {
            if groups < 1 || experts % groups != 0 {
                return bad(format!(
                    "n_routed_experts {experts} does not divide into n_group {groups}"
                ));
            }
            if kept < 1 || kept > groups {
                return bad(format!("topk_group {kept} is not within n_group {groups}"));
            }
            if top_k > kept * (experts / groups) {
                return bad(format!(
                    "num_experts_per_tok {top_k} exceeds the {} experts of the {kept} kept groups",
                    kept * (experts / groups)
                ));
            }
        }
        Ok(())
    }
}

/// Parse and check `config.json` from a model directory.
pub fn load_config(model_dir: &std::path::Path) -> Result<DeepSeekConfig, String> {
    let text = crate::native_loader::read_config_json(model_dir)?;
    let cfg: DeepSeekConfig =
        serde_json::from_str(&text).map_err(|e| format!("failed to parse config.json: {e}"))?;
    cfg.validate()?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::{DeepSeekConfig, ScoringFunc, TopkMethod};

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

    fn parse(json: &str) -> DeepSeekConfig {
        let config: DeepSeekConfig = serde_json::from_str(json).expect("released config parses");
        config.validate().expect("released config is supported");
        config
    }

    /// The released V3 / R1 `config.json` routes by sigmoid scores and the
    /// group-limited `noaux_tc` top-k: 256 experts in 8 groups, 4 kept.
    #[test]
    fn released_v3_config_routes_by_noaux_tc() {
        let c = parse(include_str!("../../tests/fixtures/deepseek_v3_config.json"));
        assert_eq!(c.scoring(), ScoringFunc::Sigmoid);
        assert_eq!(c.topk(), TopkMethod::NoauxTc);
        assert_eq!((c.n_group(), c.topk_group()), (8, 4));
        assert_eq!(c.q_lora_rank, Some(1536));
        assert!(c.is_moe_layer(3) && !c.is_moe_layer(2));
    }

    /// V2-Lite's `config.json` has `"q_lora_rank": null` (a direct Q
    /// projection), softmax scoring and the plain top-k over one group.
    #[test]
    fn released_v2_lite_config_routes_by_softmax_greedy() {
        let c = parse(include_str!(
            "../../tests/fixtures/deepseek_v2_lite_config.json"
        ));
        assert_eq!(c.scoring(), ScoringFunc::Softmax);
        assert_eq!(c.topk(), TopkMethod::Greedy);
        assert_eq!(c.q_lora_rank, None);
        assert!(!c.is_moe_layer(0) && c.is_moe_layer(1));
    }

    /// A config that omits `scoring_func` and `topk_method` gets its model
    /// type's: V2 softmax and greedy, anything else V3's.
    #[test]
    fn routing_defaults_follow_the_model_type() {
        let base = |model_type: &str| {
            serde_json::from_value::<DeepSeekConfig>(serde_json::json!({
                "model_type": model_type, "intermediate_size": 8, "num_experts_per_tok": 2,
                "num_hidden_layers": 1, "num_attention_heads": 1, "kv_lora_rank": 4,
                "qk_rope_head_dim": 4, "v_head_dim": 4, "qk_nope_head_dim": 4
            }))
            .unwrap()
        };
        let v2 = base("deepseek_v2");
        assert_eq!(
            (v2.scoring(), v2.topk()),
            (ScoringFunc::Softmax, TopkMethod::Greedy)
        );
        let v3 = base("deepseek_v3");
        assert_eq!(
            (v3.scoring(), v3.topk()),
            (ScoringFunc::Sigmoid, TopkMethod::NoauxTc)
        );
    }

    /// Configs for models this engine would run as something else are
    /// refused by name, not run.
    #[test]
    fn unsupported_configs_are_refused() {
        let with = |extra: serde_json::Value| {
            let mut json = serde_json::json!({
                "intermediate_size": 8, "num_experts_per_tok": 2, "num_hidden_layers": 1,
                "num_attention_heads": 1, "kv_lora_rank": 4, "qk_rope_head_dim": 4,
                "v_head_dim": 4, "qk_nope_head_dim": 4, "n_routed_experts": 8,
                "n_group": 4, "topk_group": 2
            });
            json.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            serde_json::from_value::<DeepSeekConfig>(json)
                .unwrap()
                .validate()
        };
        assert!(with(serde_json::json!({})).is_ok());
        assert!(
            with(serde_json::json!({"index_topk": 2048}))
                .unwrap_err()
                .contains("index_topk")
        );
        assert!(
            with(serde_json::json!({"attention_bias": true}))
                .unwrap_err()
                .contains("attention_bias")
        );
        assert!(
            with(serde_json::json!({"hidden_act": "gelu"}))
                .unwrap_err()
                .contains("gelu")
        );
        assert!(
            with(serde_json::json!({"n_group": 3}))
                .unwrap_err()
                .contains("n_group")
        );
        assert!(
            with(serde_json::json!({"topk_group": 5}))
                .unwrap_err()
                .contains("topk_group")
        );
        assert!(
            with(serde_json::json!({"num_experts_per_tok": 5}))
                .unwrap_err()
                .contains("kept groups")
        );
    }
}
