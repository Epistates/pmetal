//! Standalone Qwen3/Qwen3.5 inference engine — zero dependency on mlx-rs or pmetal-models.
//!
//! Supports three variants:
//!   - **Qwen3 dense** (`model_type = "qwen3"`): standard attention (no gate), full RoPE.
//!   - **Qwen3.5 dense** (`model_type = "qwen3_5"` / `"qwen3_5_text"`): gated attention,
//!     partial RoPE, GDN layers.
//!   - **Qwen3.5 MoE** (same type fields but `num_experts > 0`): like Qwen3.5 dense but
//!     MLP replaced with routed expert dispatch (SwitchGLU) plus a shared expert.
//!
//! Every op on the hot path uses [`InlineArray`] (stack-allocated `mlx::core::array`,
//! direct C++ bridge). This eliminates ALL per-op heap allocation, matching
//! Python/nanobind's direct C++ binding performance.
//!
//! The stack is split across focused submodules:
//!   * [`weights`] — layer + model weight bundles + Hadamard/QJL/outlier preconditioning
//!   * [`cache`] — GDN state + KV caches (bf16 / affine-quantized / TurboQuant)
//!   * [`load`] — safetensors loading + sanitization + expert stacking
//!   * [`attention`] — per-position RoPE + attn_forward + tree-verify variant
//!   * [`mlp_moe`] — dense SwiGLU + SwitchGLU MoE + GDN step
//!   * [`forward`] — forward_step + capture + tree-verify + compact/rollback
//!   * [`generate`] — prefill/prime/generate loops + benchmarks + C++ decode session

use serde::Deserialize;

mod attention;
mod cache;
mod forward;
mod generate;
mod load;
mod mlp_moe;
mod weights;

pub use cache::{GdnCache, KvLayerCache, MixedBitConfig, NativeCache, QuantCacheConfig, QuantizedTuple};
pub use forward::{
    compact_tree_cache, forward_step, forward_step_tree_verify, forward_step_with_capture,
    rollback_cache,
};
pub use generate::{
    CppDecodeSession, CppForwardState, QwenDecodeBackend, benchmark_mlx_lm_trial,
    benchmark_mlx_lm_trial_canonical, build_cpp_forward_state, canonical_decode_backend, generate,
    generate_canonical, generate_from_primed_sample, generate_from_primed_sample_silent,
    generate_preserve_peak, prefill_first_token, prime_generation_preserve_peak,
    prime_generation_preserve_peak_silent,
};
pub use load::load_model;
pub use weights::{
    LayerWeight, NativeWeights, apply_kv_preconditioning, apply_outlier_permutation,
    apply_qjl_matrix,
};

// ============================================================================
// Config
// ============================================================================

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_quantization_bits() -> i32 {
    4
}
fn default_quantization_group_size() -> i32 {
    64
}
fn default_intermediate_size() -> i32 {
    14_336
}
fn default_rope_theta() -> f64 {
    100_000.0
}
fn default_true() -> bool {
    true
}
fn default_full_attn_interval() -> i32 {
    4
}
fn default_decoder_sparse_step() -> i32 {
    1
}
fn default_conv_kernel() -> i32 {
    4
}
fn default_model_type() -> String {
    "qwen3_5".to_string()
}

fn bundled_mlx_supports_quant_bits(bits: i32) -> bool {
    matches!(bits, 2 | 3 | 4 | 5 | 6 | 8)
}

pub(super) fn validate_quantization_runtime_support_for(
    bits: i32,
    mlx_kind: &str,
    mlx_git_tag: &str,
) -> Result<(), String> {
    if mlx_kind == "bundled-upstream" && !bundled_mlx_supports_quant_bits(bits) {
        return Err(format!(
            "This pmetal build uses bundled upstream MLX {mlx_git_tag}, whose Metal affine quantized kernels only support bits in {{2, 3, 4, 5, 6, 8}}. The requested model uses {bits}-bit quantization. Rebuild pmetal against a compatible libmlx by setting PMETAL_MLX_LIB_DIR to the directory containing libmlx.dylib before running `cargo build --release -p pmetal`. 1-bit checkpoints such as PrismML Bonsai require a 1-bit-capable MLX fork."
        ));
    }
    Ok(())
}

pub(super) fn validate_quantization_runtime_support(bits: i32) -> Result<(), String> {
    validate_quantization_runtime_support_for(
        bits,
        option_env!("PMETAL_BRIDGE_MLX_KIND").unwrap_or("bundled-upstream"),
        option_env!("PMETAL_BRIDGE_MLX_GIT_TAG").unwrap_or("unknown"),
    )
}

/// Minimal, serde-deserializable Qwen3/Qwen3.5 config.
///
/// Only the fields required for inference are included; unknown keys are
/// silently ignored by serde.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    /// Distinguishes model families: `"qwen3"` = dense Qwen3; `"qwen3_5"` /
    /// `"qwen3_5_text"` = Qwen3.5 hybrid (dense or MoE).
    #[serde(default = "default_model_type")]
    pub model_type: String,

    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,

    #[serde(default)]
    pub num_key_value_heads: Option<i32>,

    #[serde(default)]
    pub head_dim: Option<i32>,

    /// Dense MLP intermediate size.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// Fraction of head_dim rotated. Stored as Option so we can fall back to
    /// nested `rope_parameters.partial_rotary_factor` at parse time.
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,

    /// Nested rope config — only used during `finalize()` to promote
    /// `partial_rotary_factor` when the top-level field is absent.
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,

    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: i32,

    // GDN / linear-attention params (Qwen3.5 only)
    #[serde(default)]
    pub linear_num_key_heads: Option<i32>,
    #[serde(default)]
    pub linear_num_value_heads: Option<i32>,
    #[serde(default)]
    pub linear_key_head_dim: Option<i32>,
    #[serde(default)]
    pub linear_value_head_dim: Option<i32>,

    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: i32,

    // MoE fields (Qwen3.5 MoE only)
    #[serde(default)]
    pub num_experts: i32,
    #[serde(default)]
    pub num_experts_per_tok: i32,
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: i32,
    #[serde(default)]
    pub shared_expert_intermediate_size: i32,
    #[serde(default)]
    pub moe_intermediate_size: i32,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    /// Layer indices that use dense MLP even when MoE is active.
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,

    /// Optional quantization config.
    ///
    /// MLX checkpoints commonly use `quantization_config`, but newer Bonsai /
    /// MLX-LM exports may spell the same object as `quantization`.
    #[serde(default, alias = "quantization")]
    pub quantization_config: Option<QuantizationConfig>,
}

/// Weight quantization parameters (from `quantization_config` in config.json).
///
/// Present in models quantized with `mlx_lm.convert --q-bits 4` or similar.
/// The `mode` field is informational only — we always use affine dequant.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    #[serde(default = "default_quantization_bits")]
    pub bits: i32,
    #[serde(default = "default_quantization_group_size")]
    pub group_size: i32,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RopeParameters {
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    rope_theta: Option<f64>,
}

impl Qwen3Config {
    /// Promote nested `rope_parameters.partial_rotary_factor` when the
    /// top-level field is absent. Call once after deserializing.
    pub fn finalize(&mut self) {
        if let Some(ref rp) = self.rope_parameters.clone() {
            if self.partial_rotary_factor.is_none() {
                if let Some(prf) = rp.partial_rotary_factor {
                    self.partial_rotary_factor = Some(prf);
                }
            }
            if self.rope_theta == default_rope_theta() {
                if let Some(theta) = rp.rope_theta {
                    self.rope_theta = theta;
                }
            }
        }
    }

    /// Returns `true` for dense Qwen3 (standard attention, no GDN).
    pub fn is_qwen3_dense(&self) -> bool {
        let mt = self.model_type.to_ascii_lowercase();
        mt == "qwen3" || mt == "qwen3dense"
    }

    /// Returns `true` when this is a MoE model (routed experts + shared expert).
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// Returns `true` when the MLP at the given layer index should be dense.
    ///
    /// For Qwen3.5 MoE the MLP is dense when:
    ///   - the layer is listed in `mlp_only_layers`, OR
    ///   - `decoder_sparse_step > 0` and `(layer_idx + 1) % decoder_sparse_step != 0`
    pub fn is_dense_mlp_layer(&self, layer_idx: usize) -> bool {
        if !self.is_moe() {
            return true;
        }
        if self.mlp_only_layers.contains(&layer_idx) {
            return true;
        }
        if self.decoder_sparse_step > 1 {
            return ((layer_idx as i32) + 1) % self.decoder_sparse_step != 0;
        }
        false
    }

    /// Effective partial rotary factor.
    ///
    /// - Qwen3 dense: always 1.0 (full RoPE).
    /// - Qwen3.5: stored value, defaulting to 0.25.
    pub fn effective_partial_rotary_factor(&self) -> f64 {
        if self.is_qwen3_dense() {
            1.0
        } else {
            self.partial_rotary_factor.unwrap_or(0.25)
        }
    }

    /// Effective head dimension.
    pub fn get_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Effective number of KV heads.
    pub fn get_num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// RoPE dimensions for partial rotary.
    pub fn rope_dims(&self) -> i32 {
        (self.get_head_dim() as f64 * self.effective_partial_rotary_factor()) as i32
    }

    /// Returns `true` when layer `i` is a GDN (linear-attention) layer.
    ///
    /// For Qwen3 dense, all layers are attention — never GDN.
    /// For Qwen3.5: every `full_attention_interval`-th layer (1-indexed) is
    /// a full-attention layer; all others are GDN.
    pub fn is_linear_layer(&self, i: usize) -> bool {
        if self.is_qwen3_dense() {
            return false;
        }
        ((i as i32) + 1) % self.full_attention_interval != 0
    }

    // GDN dimension accessors (with Qwen3.5 defaults)
    pub fn gdn_nk(&self) -> i32 {
        self.linear_num_key_heads.unwrap_or(4)
    }
    pub fn gdn_nv(&self) -> i32 {
        self.linear_num_value_heads.unwrap_or(8)
    }
    pub fn gdn_dk(&self) -> i32 {
        self.linear_key_head_dim.unwrap_or(128)
    }
    pub fn gdn_dv(&self) -> i32 {
        self.linear_value_head_dim.unwrap_or(128)
    }
}

/// Parse `config.json` from a model directory.
pub fn load_config(model_dir: &std::path::Path) -> Result<Qwen3Config, String> {
    let text = crate::native_loader::read_config_json(model_dir)?;
    parse_config_text(&text)
}

pub(super) fn parse_config_text(text: &str) -> Result<Qwen3Config, String> {
    let json: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("failed to parse config.json: {e}"))?;

    // Qwen3.5 nests the LM config under `text_config`.
    // For plain Qwen3, the config.json is flat.
    let config_str = if json.get("text_config").is_some() {
        // Inject `model_type` from the outer object if absent from text_config,
        // so `is_qwen3_dense()` can distinguish the families.
        let mut tc = json["text_config"].clone();
        if tc.get("model_type").is_none() {
            if let Some(mt) = json.get("model_type") {
                tc["model_type"] = mt.clone();
            }
        }
        // Promote quantization metadata from the outer JSON into text_config when
        // present at the top level but absent from the nested config. MLX-LM
        // uses `quantization_config`, while newer Bonsai exports may use
        // `quantization` for the same object.
        if tc.get("quantization_config").is_none() && tc.get("quantization").is_none() {
            if let Some(qc) = json
                .get("quantization_config")
                .or_else(|| json.get("quantization"))
            {
                tc["quantization_config"] = qc.clone();
            }
        }
        serde_json::to_string(&tc).map_err(|e| e.to_string())?
    } else {
        text.to_owned()
    };

    let mut cfg: Qwen3Config =
        serde_json::from_str(&config_str).map_err(|e| format!("failed to parse config: {e}"))?;
    cfg.finalize();
    Ok(cfg)
}

/// Trace helper — toggled by the `PMETAL_TRACE_TURBOQUANT` env var. Used by
/// the TurboQuant code paths scattered across submodules.
pub(super) fn turboquant_trace_enabled() -> bool {
    std::env::var_os("PMETAL_TRACE_TURBOQUANT").is_some()
}

pub(super) fn trace_turboquant_qwen(message: &str) {
    if turboquant_trace_enabled() {
        eprintln!("[TURBOQUANT TRACE][QWEN] {message}");
    }
}

#[cfg(test)]
mod tests {
    use super::mlp_moe::moe_switch_glu_input;
    use super::{
        QwenDecodeBackend, canonical_decode_backend, parse_config_text,
        validate_quantization_runtime_support_for,
    };
    use crate::{compat::Dtype, inline_array::InlineArray};

    #[test]
    fn parse_nested_qwen35_promotes_rope_parameters() {
        let config = parse_config_text(
            r#"{
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 1536,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 12,
                    "num_key_value_heads": 2,
                    "head_dim": 128,
                    "intermediate_size": 3584,
                    "rope_parameters": {
                        "rope_theta": 10000000.0,
                        "partial_rotary_factor": 0.25,
                        "rope_type": "default"
                    }
                }
            }"#,
        )
        .expect("config parses");

        assert_eq!(config.model_type, "qwen3_5_text");
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert_eq!(config.partial_rotary_factor, Some(0.25));
    }

    #[test]
    fn parse_qwen35_moe_uses_mlx_reference_defaults_for_missing_fields() {
        let config = parse_config_text(
            r#"{
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "moe_intermediate_size": 512,
                    "shared_expert_intermediate_size": 512,
                    "rope_parameters": {
                        "rope_theta": 10000000.0,
                        "partial_rotary_factor": 0.25,
                        "rope_type": "default"
                    }
                }
            }"#,
        )
        .expect("moe config parses");

        assert!(config.is_moe());
        assert_eq!(config.intermediate_size, 14_336);
        assert_eq!(config.decoder_sparse_step, 1);
        assert!(config.norm_topk_prob);
        assert_eq!(config.rope_theta, 10_000_000.0);
    }

    #[test]
    fn parse_qwen3_accepts_mlx_quantization_alias() {
        let config = parse_config_text(
            r#"{
                "model_type": "qwen3",
                "hidden_size": 4096,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "quantization": {
                    "group_size": 128,
                    "bits": 1
                }
            }"#,
        )
        .expect("quantized qwen3 config parses");

        let qc = config
            .quantization_config
            .as_ref()
            .expect("quantization config present");
        assert_eq!(qc.group_size, 128);
        assert_eq!(qc.bits, 1);
    }

    #[test]
    fn parse_nested_qwen35_promotes_outer_quantization_alias() {
        let config = parse_config_text(
            r#"{
                "model_type": "qwen3_5",
                "quantization": {
                    "group_size": 64,
                    "bits": 4
                },
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 1536,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 12,
                    "num_key_value_heads": 2,
                    "head_dim": 128
                }
            }"#,
        )
        .expect("nested quantized qwen3.5 config parses");

        let qc = config
            .quantization_config
            .as_ref()
            .expect("quantization config promoted");
        assert_eq!(qc.group_size, 64);
        assert_eq!(qc.bits, 4);
    }

    #[test]
    fn bundled_upstream_mlx_rejects_one_bit_quantization() {
        let err = validate_quantization_runtime_support_for(1, "bundled-upstream", "v0.31.1")
            .expect_err("1-bit should be rejected");
        assert!(err.contains("PMETAL_MLX_LIB_DIR"));
        assert!(err.contains("1-bit"));
    }

    #[test]
    fn bundled_upstream_mlx_accepts_four_bit_quantization() {
        validate_quantization_runtime_support_for(4, "bundled-upstream", "v0.31.1")
            .expect("4-bit should be supported");
    }

    #[test]
    fn external_mlx_build_can_attempt_one_bit_quantization() {
        validate_quantization_runtime_support_for(1, "external", "v0.31.1")
            .expect("external libmlx should not be pre-rejected");
    }

    #[test]
    fn moe_switch_glu_input_matches_mlx_rank_contract() {
        let dt = Dtype::Bfloat16.as_i32();
        let x_flat = InlineArray::ones(&[3, 4], dt);
        let switch_in = moe_switch_glu_input(&x_flat);
        assert_eq!(switch_in.shape(), &[3, 1, 1, 4]);
    }

    #[test]
    fn canonical_decode_backend_prefers_rust_bridge_for_qwen35_moe() {
        let config = parse_config_text(
            r#"{
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "moe_intermediate_size": 512,
                    "shared_expert_intermediate_size": 512
                }
            }"#,
        )
        .expect("moe config parses");

        assert_eq!(
            canonical_decode_backend(&config, None),
            QwenDecodeBackend::RustBridge
        );
    }

    #[test]
    fn reserve_decode_inputs_grows_dense_kv_to_exact_target() {
        let dt = Dtype::Bfloat16.as_i32();
        let mut cache = super::NativeCache {
            gdn_caches: Vec::new(),
            kv_caches: vec![super::KvLayerCache {
                keys: Some(InlineArray::zeros(&[1, 2, 16, 8], dt)),
                values: Some(InlineArray::zeros(&[1, 2, 16, 8], dt)),
                offset: 12,
                turboquant: None,
                quantized_keys: None,
                quantized_values: None,
                quantized_keys_hi: None,
                quantized_values_hi: None,
                quant_config: None,
                qjl_signs: None,
                qjl_residual_norms: None,
            }],
            rope_offset: 0,
            turboquant_state: None,
        };

        cache.reserve_decode_inputs(5, dt);

        assert_eq!(cache.kv_caches[0].keys.as_ref().unwrap().dim(2), 17);
        assert_eq!(cache.kv_caches[0].values.as_ref().unwrap().dim(2), 17);
    }
}
