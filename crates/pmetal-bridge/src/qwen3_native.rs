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
//! The entire stack — config, weights, caches, forward pass, generation loop —
//! lives in this single module. The only external dependencies are
//! `serde`/`serde_json` (for config parsing) and `crate::InlineArray`.

use serde::Deserialize;

use crate::InlineArray;
use crate::inline_array as bridge;
use crate::inline_array::RawBuf;

fn turboquant_trace_enabled() -> bool {
    std::env::var_os("PMETAL_TRACE_TURBOQUANT").is_some()
}

fn trace_turboquant_qwen(message: &str) {
    if turboquant_trace_enabled() {
        eprintln!("[TURBOQUANT TRACE][QWEN] {message}");
    }
}

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

fn validate_quantization_runtime_support_for(
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

fn validate_quantization_runtime_support(bits: i32) -> Result<(), String> {
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
    let path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    parse_config_text(&text)
}

fn parse_config_text(text: &str) -> Result<Qwen3Config, String> {
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

#[cfg(test)]
mod tests {
    use super::{
        QwenDecodeBackend, canonical_decode_backend, moe_switch_glu_input, parse_config_text,
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

// ============================================================================
// Quantized / dense weight discriminant
// ============================================================================

/// A projection weight that is either a dense bf16 tensor or a 4-bit
/// quantized triple (`weight`, `scales`, `biases`).
///
/// - **Dense**: single `InlineArray`, used with `x.matmul(w)`.
/// - **Quantized**: three `InlineArray`s loaded from `{key}.weight`,
///   `{key}.scales`, `{key}.biases`; used with
///   `x.quantized_matmul(w, scales, biases, transpose=true, group_size, bits)`.
///
/// The `transpose=true` flag is standard: MLX stores quantized weights in
/// row-major `[out, in/group_size]` layout and expects the caller to signal
/// that the weight logically needs transposing for the matmul (i.e. the same
/// semantic as storing the dense weight transposed as `[in, out]`).
///
/// Dense weights are pre-transposed at load time (`w.t()`); quantized weights
/// are stored as-is from the checkpoint and the `transpose=true` flag handles
/// the layout internally inside `mx.quantized_matmul`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum LayerWeight {
    Dense(InlineArray),
    Quantized {
        weight: InlineArray, // packed uint32: shape [out, in/(32/bits)]
        scales: InlineArray, // per-group scale: shape [out, in/group_size]
        biases: InlineArray, // per-group bias:  shape [out, in/group_size]
        group_size: i32,
        bits: i32,
    },
}

impl LayerWeight {
    /// Get the underlying weight tensor (for use in copy_fresh or pointer export).
    pub fn weight_arr(&self) -> &InlineArray {
        match self {
            LayerWeight::Dense(w) => w,
            LayerWeight::Quantized { weight, .. } => weight,
        }
    }

    /// `x @ self` — dispatches to `quantized_matmul` or `matmul` as appropriate.
    ///
    /// For dense weights: `x.matmul(w)` — weight is pre-transposed `[in, out]`.
    /// For quantized:    `x.quantized_matmul(w, scales, biases, true, gs, bits)`.
    #[inline(always)]
    pub fn matmul_from(&self, x: &InlineArray) -> InlineArray {
        match self {
            LayerWeight::Dense(w) => x.matmul(w),
            LayerWeight::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => x.quantized_matmul(weight, scales, Some(biases), true, *group_size, *bits),
        }
    }

    /// Gather-matmul: `gather_mm` for dense, `gather_qmm` for quantized.
    ///
    /// Used by MoE expert dispatch.  All expert tensors in a layer must have
    /// the same variant — mixing dense and quantized experts is not supported.
    #[inline(always)]
    pub fn gather_mm_from(
        &self,
        x: &InlineArray,
        lhs_indices: Option<&InlineArray>,
        rhs_indices: Option<&InlineArray>,
        sorted: bool,
    ) -> InlineArray {
        match self {
            LayerWeight::Dense(w) => x.gather_mm(w, lhs_indices, rhs_indices, sorted),
            LayerWeight::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => x.gather_qmm(
                weight,
                scales,
                Some(biases),
                lhs_indices,
                rhs_indices,
                true,
                *group_size,
                *bits,
                sorted,
            ),
        }
    }

    /// Apply `copy_fresh` to all arrays in this weight (add zero + eval + detach).
    pub fn copy_fresh(&self, zero: &InlineArray) -> Self {
        match self {
            LayerWeight::Dense(w) => LayerWeight::Dense(copy_fresh_arr(w, zero)),
            LayerWeight::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => {
                // For quantized weights the zero must be int32 (weight dtype)
                // for the weight tensor and float for scales/biases.
                // Use add-zero on each independently via eval+detach.
                let w2 = copy_fresh_arr(weight, zero);
                let s2 = copy_fresh_arr(scales, zero);
                let b2 = copy_fresh_arr(biases, zero);
                LayerWeight::Quantized {
                    weight: w2,
                    scales: s2,
                    biases: b2,
                    group_size: *group_size,
                    bits: *bits,
                }
            }
        }
    }
}

/// Force a single array into a fresh Metal buffer (add zero + eval + detach).
/// This is the implementation detail shared by `LayerWeight::copy_fresh`.
fn copy_fresh_arr(w: &InlineArray, _hint_zero: &InlineArray) -> InlineArray {
    // For quantized weights (uint32 packed), skip copy_fresh — they're already
    // loaded directly through pmetal-bridge's MLX and don't need data duplication.
    let dt = w.dtype_raw();
    if dt == 3
    /* uint32 */
    {
        // Just eval + detach without add
        let mut fresh = w.clone();
        fresh.eval();
        fresh.detach();
        return fresh;
    }
    let own_zero = InlineArray::zeros(&[1], dt);
    let mut fresh = w.add(&own_zero);
    fresh.eval();
    fresh.detach();
    fresh
}

// ============================================================================
// Per-layer weights
// ============================================================================

pub(crate) struct LayerWeights {
    pub(crate) is_linear: bool,

    // Shared: layer norms (never quantized — 1D tensors)
    pub(crate) input_ln_w: InlineArray,
    pub(crate) input_ln_eps: f32,
    pub(crate) post_ln_w: InlineArray,
    pub(crate) post_ln_eps: f32,

    // Dense MLP (when !is_moe_layer)
    pub(crate) mlp_gate_w: Option<LayerWeight>,
    pub(crate) mlp_up_w: Option<LayerWeight>,
    pub(crate) mlp_down_w: Option<LayerWeight>,

    // MoE (when is_moe_layer)
    pub(crate) moe_router_w: Option<InlineArray>,
    pub(crate) moe_gate_w: Option<LayerWeight>,
    pub(crate) moe_up_w: Option<LayerWeight>,
    pub(crate) moe_down_w: Option<LayerWeight>,
    pub(crate) shared_gate_w: Option<LayerWeight>,
    pub(crate) shared_up_w: Option<LayerWeight>,
    pub(crate) shared_down_w: Option<LayerWeight>,
    pub(crate) shared_expert_gate_w: Option<InlineArray>,

    pub(crate) moe_top_k: i32,
    pub(crate) moe_norm_topk_prob: bool,
    pub(crate) is_moe_layer: bool,

    // Attention-specific (only when !is_linear)
    pub(crate) attn_q_w: Option<LayerWeight>,
    pub(crate) attn_k_w: Option<LayerWeight>,
    pub(crate) attn_v_w: Option<LayerWeight>,
    pub(crate) attn_o_w: Option<LayerWeight>,
    pub(crate) attn_q_norm_w: Option<InlineArray>,
    pub(crate) attn_q_norm_eps: f32,
    pub(crate) attn_k_norm_w: Option<InlineArray>,
    pub(crate) attn_k_norm_eps: f32,
    pub(crate) attn_n_heads: i32,
    pub(crate) attn_n_kv_heads: i32,
    pub(crate) attn_head_dim: i32,
    pub(crate) attn_scale: f32,
    pub(crate) attn_rope_dims: i32,
    pub(crate) attn_rope_base: f32,
    pub(crate) attn_rope_scale: f32,
    pub(crate) attn_gated: bool,

    // GDN-specific (only when is_linear)
    pub(crate) gdn_qkv_w: Option<LayerWeight>,
    pub(crate) gdn_z_w: Option<LayerWeight>,
    pub(crate) gdn_b_w: Option<LayerWeight>,
    pub(crate) gdn_a_w: Option<LayerWeight>,
    pub(crate) gdn_conv_w: Option<InlineArray>,
    pub(crate) gdn_q_nw: Option<InlineArray>,
    pub(crate) gdn_k_nw: Option<InlineArray>,
    pub(crate) gdn_a_log: Option<InlineArray>,
    pub(crate) gdn_dt_bias: Option<InlineArray>,
    pub(crate) gdn_norm_w: Option<InlineArray>,
    pub(crate) gdn_norm_eps: f32,
    pub(crate) gdn_out_w: Option<LayerWeight>,
    pub(crate) gdn_nv: i32,
    pub(crate) gdn_nk: i32,
    pub(crate) gdn_dk: i32,
    pub(crate) gdn_dv: i32,
    pub(crate) gdn_kd: i32,
    pub(crate) gdn_cd: i32,
    pub(crate) gdn_ck: i32,
}

// ============================================================================
// Full model weights
// ============================================================================

/// All model weights as InlineArray. Zero dependency on mlx-rs.
pub struct NativeWeights {
    pub embed_w: InlineArray,
    /// Quantized embedding: scales + biases (None for dense models)
    pub embed_scales: Option<InlineArray>,
    pub embed_biases: Option<InlineArray>,
    pub final_norm_w: InlineArray,
    pub final_norm_eps: f32,
    /// None when `tie_word_embeddings = true`.
    ///
    /// Untied heads may be dense or quantized, just like the projection
    /// matrices inside the decoder blocks.
    pub lm_head_w: Option<LayerWeight>,
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizationConfig>,
    /// Per-layer weights — opaque to callers; only accessed via [`forward_step`].
    pub(crate) layers: Vec<LayerWeights>,
    /// Model activation dtype (e.g., 11 = bfloat16).
    pub model_dtype: i32,
    /// QJL projection matrix S [head_dim, head_dim] for key residual correction.
    ///
    /// Generated at load time when `apply_qjl_matrix` is called. Used during
    /// attention to compute sign(S · residual) for unbiased inner product
    /// estimation at Q2-Q3 bits. `None` when QJL is disabled.
    pub qjl_matrix: Option<InlineArray>,
}

impl std::fmt::Debug for NativeWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeWeights")
            .field("layers", &self.layers.len())
            .field("tie_word_embeddings", &self.tie_word_embeddings)
            .field("model_dtype", &self.model_dtype)
            .finish()
    }
}

// ============================================================================
// Caches
// ============================================================================

/// GDN layer cache state (conv + SSM).
pub struct GdnCache {
    pub conv_state: Option<InlineArray>,
    pub ssm_state: Option<InlineArray>,
}

/// Affine-quantized KV cache tuple: (packed_uint32, scales, biases).
///
/// Matches mlx-lm's `QuantizedKVCache` storage format. The packed data, scales,
/// and biases are passed directly to `quantized_matmul` which dequantizes inside
/// the Metal kernel — zero-overhead vs bf16 SDPA.
#[derive(Clone)]
pub struct QuantizedTuple {
    pub packed: InlineArray, // [B, H, T, D_packed] uint32
    pub scales: InlineArray, // [B, H, T, D/group_size]
    pub biases: InlineArray, // [B, H, T, D/group_size]
}

/// Mixed-bit configuration for TurboQuant v2 presets (Q2.5, Q3.5).
///
/// Splits head dimensions into outlier channels (top 25% by magnitude) and
/// regular channels, quantizing each at different bit widths. The channel
/// permutation is absorbed into projection weights at load time for zero
/// runtime overhead.
#[derive(Clone, Copy, Debug)]
pub struct MixedBitConfig {
    /// Number of outlier channels per head (quantized at higher bits).
    pub outlier_count: i32,
    /// Bit width for outlier channels (e.g., 3 for Q2.5, 4 for Q3.5).
    pub outlier_bits: u8,
    /// Bit width for regular channels (e.g., 2 for Q2.5, 3 for Q3.5).
    pub regular_bits: u8,
}

/// Configuration for zero-overhead affine KV cache quantization.
#[derive(Clone, Copy, Debug)]
pub struct QuantCacheConfig {
    pub bits: u8,
    pub group_size: i32,
    /// Mixed-bit mode (TurboQuant v2). When set, `bits` is ignored and the
    /// outlier/regular split is used instead. Channel permutation must be
    /// applied to projection weights at load time via [`apply_outlier_permutation`].
    pub mixed_bit: Option<MixedBitConfig>,
    /// QJL residual correction for keys at Q2-Q3.
    ///
    /// When true, the uniform quantized path computes a 1-bit sign vector on
    /// the quantization residual and stores it in `KvLayerCache::qjl_signs`.
    /// At SDPA time, a correction term is added to attention scores to make
    /// the inner product estimate unbiased. Only active for bits <= 3 and
    /// uniform quantization (not mixed-bit).
    pub qjl: bool,
}

/// Per-layer KV cache using pre-allocated buffers with O(1) slice_set updates.
pub struct KvLayerCache {
    pub keys: Option<InlineArray>,   // [B, H, MAX_T, D] pre-allocated
    pub values: Option<InlineArray>, // [B, H, MAX_T, D] pre-allocated
    pub offset: i32,                 // number of valid tokens
    /// TurboQuant compressed cache (replaces keys/values when enabled)
    pub turboquant: Option<crate::turboquant::QuantizedKvCache>,
    /// Zero-overhead affine-quantized cache — regular channels (lower bits)
    pub quantized_keys: Option<QuantizedTuple>,
    pub quantized_values: Option<QuantizedTuple>,
    /// Mixed-bit outlier channels (higher bits). `None` in uniform-quantization mode.
    pub quantized_keys_hi: Option<QuantizedTuple>,
    pub quantized_values_hi: Option<QuantizedTuple>,
    pub quant_config: Option<QuantCacheConfig>,
    /// QJL residual correction for key inner products (uniform Q2-Q3 only).
    ///
    /// `qjl_signs`: `[B, H, MAX_T, D]` bf16 ±1.0 — sign(S · residual).
    /// `qjl_residual_norms`: `[B, H, MAX_T, 1]` f32 — L2 norm of residual.
    ///
    /// Both `None` when QJL is disabled or cache is empty.
    pub qjl_signs: Option<InlineArray>, // [B, H, MAX_T, D] model_dtype ±1.0
    pub qjl_residual_norms: Option<InlineArray>, // [B, H, MAX_T, 1] f32
}

/// Full model cache — both GDN and KV layers.
pub struct NativeCache {
    pub gdn_caches: Vec<GdnCache>,
    pub kv_caches: Vec<KvLayerCache>,
    pub rope_offset: i32,
    /// Shared TurboQuant state (rotation matrices, codebooks) — None = bf16 cache
    pub turboquant_state: Option<std::sync::Arc<crate::turboquant::TurboQuantState>>,
}

impl NativeCache {
    /// Evaluate all cache states in-place and detach them from their computation
    /// graph.  Must be called after the prefill forward pass and before decode.
    ///
    /// Python's `generate_step` does `mx.eval([c.state for c in prompt_cache])`
    /// at this point.  Without this, the unevaluated prefill SSM states have the
    /// entire prefill graph attached; when decode builds its graph those prefill
    /// nodes are included, adding hundreds of extra AsType/Matmul/etc. nodes.
    pub fn eval_and_detach_states(&mut self) {
        // Collect all non-None state arrays into a temporary vec for batch eval.
        let mut to_eval: Vec<&mut InlineArray> = Vec::new();
        for c in &mut self.gdn_caches {
            if let Some(ref mut s) = c.ssm_state {
                to_eval.push(s);
            }
            if let Some(ref mut s) = c.conv_state {
                to_eval.push(s);
            }
        }
        for c in &mut self.kv_caches {
            if let Some(k) = c.keys.take() {
                let trimmed = if c.offset > 0 && c.offset < k.dim(2) {
                    k.slice(&[0, 0, 0, 0], &[k.dim(0), k.dim(1), c.offset, k.dim(3)])
                } else {
                    k
                };
                c.keys = Some(trimmed);
            }
            if let Some(v) = c.values.take() {
                let trimmed = if c.offset > 0 && c.offset < v.dim(2) {
                    v.slice(&[0, 0, 0, 0], &[v.dim(0), v.dim(1), c.offset, v.dim(3)])
                } else {
                    v
                };
                c.values = Some(trimmed);
            }
            if let Some(ref mut k) = c.keys {
                to_eval.push(k);
            }
            if let Some(ref mut v) = c.values {
                to_eval.push(v);
            }
            if let Some(ref mut tq) = c.turboquant {
                tq.eval_and_detach_gpu_state();
            }
        }
        // Batch eval then detach each.
        bridge::eval_and_detach_many(&mut to_eval);
    }

    /// Create a fresh, empty cache for the given weight set.
    pub fn new_empty(weights: &NativeWeights) -> Self {
        Self::new_with_turboquant(weights, None)
    }

    /// Create cache with optional TurboQuant KV compression.
    pub fn new_with_turboquant(
        weights: &NativeWeights,
        tq_config: Option<crate::turboquant::TurboQuantConfig>,
    ) -> Self {
        let mut gdn_caches = Vec::new();
        let mut kv_caches = Vec::new();

        // Build shared TurboQuant state if enabled
        let tq_state = tq_config.map(|cfg| {
            // Use the first attention layer's head_dim for key/value dims
            let head_dim = weights
                .layers
                .iter()
                .find(|lw| !lw.is_linear)
                .map(|lw| lw.attn_head_dim as usize)
                .unwrap_or(128);
            crate::turboquant::build_state(head_dim, head_dim, cfg)
        });

        for lw in &weights.layers {
            if lw.is_linear {
                gdn_caches.push(GdnCache {
                    conv_state: None,
                    ssm_state: None,
                });
            } else {
                let tq_cache = tq_state.as_ref().map(|state| {
                    crate::turboquant::new_cache_with_state(tq_config.unwrap(), state.clone())
                });
                kv_caches.push(KvLayerCache {
                    keys: None,
                    values: None,
                    offset: 0,
                    turboquant: tq_cache,
                    quantized_keys: None,
                    quantized_values: None,
                    quantized_keys_hi: None,
                    quantized_values_hi: None,
                    quant_config: None,
                    qjl_signs: None,
                    qjl_residual_norms: None,
                });
            }
        }

        NativeCache {
            gdn_caches,
            kv_caches,
            rope_offset: 0,
            turboquant_state: tq_state,
        }
    }

    fn reserve_decode_inputs(&mut self, additional_tokens: i32, dtype: i32) {
        if additional_tokens <= 0 {
            return;
        }

        let mut changed_indices = Vec::new();
        for (idx, cache) in self.kv_caches.iter_mut().enumerate() {
            if cache.turboquant.is_some() {
                continue;
            }

            let Some(keys) = cache.keys.take() else {
                continue;
            };
            let Some(values) = cache.values.take() else {
                cache.keys = Some(keys);
                continue;
            };

            let current_capacity = keys.dim(2);
            let target_capacity = cache.offset + additional_tokens;
            if target_capacity <= current_capacity {
                cache.keys = Some(keys);
                cache.values = Some(values);
                continue;
            }

            let extend = target_capacity - current_capacity;
            let ext_keys =
                InlineArray::zeros(&[keys.dim(0), keys.dim(1), extend, keys.dim(3)], dtype);
            let ext_values = InlineArray::zeros(
                &[values.dim(0), values.dim(1), extend, values.dim(3)],
                dtype,
            );
            cache.keys = Some(keys.kv_cache_append(&ext_keys, 2));
            cache.values = Some(values.kv_cache_append(&ext_values, 2));
            changed_indices.push(idx);
        }

        if changed_indices.is_empty() {
            return;
        }

        let mut changed_ptrs: Vec<*mut InlineArray> = Vec::new();
        for idx in changed_indices {
            let cache = &mut self.kv_caches[idx];
            if let Some(ref mut keys) = cache.keys {
                changed_ptrs.push(keys as *mut InlineArray);
            }
            if let Some(ref mut values) = cache.values {
                changed_ptrs.push(values as *mut InlineArray);
            }
        }

        let mut to_eval: Vec<&mut InlineArray> = changed_ptrs
            .into_iter()
            .map(|ptr| unsafe { &mut *ptr })
            .collect();
        bridge::eval_and_detach_many(&mut to_eval);
    }
}

impl std::fmt::Debug for NativeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCache")
            .field("gdn_layers", &self.gdn_caches.len())
            .field("attn_layers", &self.kv_caches.len())
            .field("rope_offset", &self.rope_offset)
            .finish()
    }
}

// ============================================================================
// Weight loading
// ============================================================================

// ============================================================================
// Diagnostics helpers
// ============================================================================

/// Returns `true` if any projection weight in the loaded layers is `Quantized`.
/// Used to confirm at load time that quantized models were loaded correctly.
fn weights_are_quantized(layers: &[LayerWeights]) -> bool {
    for lw in layers {
        if let Some(LayerWeight::Quantized { .. }) = lw.mlp_gate_w {
            return true;
        }
        if let Some(LayerWeight::Quantized { .. }) = lw.attn_q_w {
            return true;
        }
        if let Some(LayerWeight::Quantized { .. }) = lw.gdn_qkv_w {
            return true;
        }
    }
    false
}

// ============================================================================
// MoE quantized expert stacking helper
// ============================================================================

/// Load pre-stacked MoE expert weights as a `LayerWeight`.
///
/// After Step 3e, the stacked weight tensor lives at `{base_key}.weight`.
/// For quantized models the stacking loop also stored:
///   `{base_key}.scales`  — [E, out, in/group_size] stacked scales
///   `{base_key}.biases`  — [E, out, in/group_size] stacked biases
///
/// If both scales and biases are present → `LayerWeight::Quantized`.
/// Otherwise → `LayerWeight::Dense` (the weight is already in [E, in, out] form
/// from the stacking step, so no additional transpose is needed).
fn get_stacked_expert_weight(
    raw: &std::collections::HashMap<String, InlineArray>,
    base_key: &str,
    group_size: i32,
    bits: i32,
) -> Result<LayerWeight, String> {
    let w_key = format!("{base_key}.weight");
    let s_key = format!("{base_key}.scales");
    let b_key = format!("{base_key}.biases");

    let w = raw
        .get(&w_key)
        .cloned()
        .ok_or_else(|| format!("missing stacked expert weight: {w_key}"))?;

    match (raw.get(&s_key), raw.get(&b_key)) {
        (Some(scales), Some(biases)) => Ok(LayerWeight::Quantized {
            weight: w,
            scales: scales.clone(),
            biases: biases.clone(),
            group_size,
            bits,
        }),
        _ => {
            // Dense — already [E, in, out]; no transpose needed.
            Ok(LayerWeight::Dense(w))
        }
    }
}

// Load model weights from a directory containing safetensors shards.
//
// Supports all three Qwen variants:
//   - Qwen3 dense (model_type = "qwen3"): standard attention, full RoPE.
//   - Qwen3.5 dense (model_type = "qwen3_5" / "qwen3_5_text", num_experts = 0).
//   - Qwen3.5 MoE (same type but num_experts > 0): routes through SwitchGLU +
//     shared expert. Per-expert weights are stacked into [E, in, out] tensors at
//     load time (matches Python's sanitize() stacking).
//
// Applies the same sanitization as the mlx-rs loader:
// - VLM prefix stripping (model.language_model. -> model.)
// - A_log -> a_log rename
// - mtp.* key drop
// - conv1d weight transpose (when shape is [out, k, in] not [out, k, 1])
// ============================================================================
// Hadamard preconditioning — absorb random rotation into Q/K/V/O weights
// ============================================================================

/// Seed for KV cache preconditioning rotation (distinct from TURBOQUANT_SEED).
const KV_PRECONDITION_SEED: u64 = 0x4b56_5052_4543_4f4e; // "KVPRECON"

/// Apply per-head random orthogonal rotation to attention projection weights.
///
/// Absorbs a random rotation R into Q/K/V/O weights at model load time so that
/// K/V vectors in the cache have more uniform coordinate distributions, improving
/// affine quantization quality at the same bit-width (TurboQuant/PolarQuant/QuIP# insight).
///
/// - Q: `W_q' = W_q @ R_block^T`  (queries in rotated space)
/// - K: `W_k' = W_k @ R_block^T`  (keys in rotated space → better quantization)
/// - V: `W_v' = W_v @ R_block^T`  (values in rotated space → better quantization)
/// - O: `W_o' = R_block @ W_o`    (undo rotation on output)
///
/// Since R is orthogonal (R^T R = I), attention scores Q@K^T are unchanged.
/// Zero runtime cost — rotation is in the weights, not the inference path.
pub fn apply_kv_preconditioning(weights: &mut NativeWeights) {
    use rand::SeedableRng;

    // Find head_dim from first attention layer
    let head_dim = weights
        .layers
        .iter()
        .find(|lw| !lw.is_linear)
        .map(|lw| lw.attn_head_dim)
        .unwrap_or(0);
    if head_dim == 0 {
        return;
    }

    // Generate deterministic orthogonal rotation R [head_dim, head_dim]
    let mut rng =
        rand::rngs::StdRng::seed_from_u64(KV_PRECONDITION_SEED ^ ((head_dim as u64) << 32));
    let r_data = crate::turboquant::generate_random_orthogonal(head_dim as usize, &mut rng);
    let r_f32 = InlineArray::from_f32_slice(&r_data, &[head_dim, head_dim]);
    // Cast rotation to model dtype to avoid f32 promotion cascade through weights
    let r = r_f32.as_dtype(weights.model_dtype);
    let r_t = r.transpose_axes(&[1, 0]); // R^T [head_dim, head_dim]

    eprintln!(
        "[PRECONDITION] Applying Hadamard preconditioning to attention weights (head_dim={})",
        head_dim
    );

    for lw in &mut weights.layers {
        if lw.is_linear {
            continue; // GDN layers — no Q/K/V/O
        }
        let n_heads = lw.attn_n_heads;
        let n_kv = lw.attn_n_kv_heads;

        if lw.attn_gated {
            // Gated attention (Qwen3.5): element-wise gate*sigmoid doesn't commute
            // with rotation, so we can only rotate Q (queries portion) and K.
            // V stays in original space → gate multiplication is correct.
            // O projection is unchanged (output stays in original space).
            // Key benefit: K preconditioning improves key quantization quality.
            rotate_projection_weight(&mut lw.attn_q_w, &r_t, n_heads, head_dim, true);
            rotate_projection_weight(&mut lw.attn_k_w, &r_t, n_kv, head_dim, false);
            // V and O untouched
        } else {
            // Non-gated attention (Qwen3): full Q/K/V/O rotation is exact.
            rotate_projection_weight(&mut lw.attn_q_w, &r_t, n_heads, head_dim, false);
            rotate_projection_weight(&mut lw.attn_k_w, &r_t, n_kv, head_dim, false);
            rotate_projection_weight(&mut lw.attn_v_w, &r_t, n_kv, head_dim, false);
            rotate_output_weight(&mut lw.attn_o_w, &r, n_heads, head_dim);
        }
    }
}

/// Generate and store the QJL projection matrix S on `weights`.
///
/// S is a [head_dim × head_dim] Gaussian random matrix used to project key
/// quantization residuals before taking their sign. Stored as the model dtype.
/// Must be called after `apply_kv_preconditioning` if both are used.
///
/// Only the uniform Q2-Q3 path uses QJL; mixed-bit paths are unaffected.
pub fn apply_qjl_matrix(weights: &mut NativeWeights) {
    use rand::SeedableRng;

    let head_dim = weights
        .layers
        .iter()
        .find(|lw| !lw.is_linear)
        .map(|lw| lw.attn_head_dim)
        .unwrap_or(0);
    if head_dim == 0 {
        return;
    }

    // Use a seed distinct from the Hadamard rotation seed so S is independent of R.
    let qjl_seed = KV_PRECONDITION_SEED ^ 0x514a_4c00; // "QJL\0"
    let mut rng = rand::rngs::StdRng::seed_from_u64(qjl_seed ^ ((head_dim as u64) << 32));
    let s_data = crate::turboquant::generate_random_projection(head_dim as usize, &mut rng);
    let s_f32 = InlineArray::from_f32_slice(&s_data, &[head_dim, head_dim]);
    let s = s_f32.as_dtype(weights.model_dtype);
    // Eval and detach into a fresh Metal buffer.
    let zero = InlineArray::from_f32(0.0).as_dtype(weights.model_dtype);
    let mut s = s.add(&zero);
    s.eval();
    s.detach();
    weights.qjl_matrix = Some(s);

    eprintln!(
        "[QJL] QJL projection matrix generated (head_dim={})",
        head_dim
    );
}

/// Reorder head dimensions so outlier channels come first, enabling mixed-bit
/// quantization where outlier channels get higher precision.
///
/// Computes per-channel L2 norms from K projection weights, finds the top
/// `outlier_fraction` channels, and builds a permutation that moves them to
/// the front of each head. The permutation is then absorbed into Q/K/V/O
/// projection weights — zero runtime cost.
///
/// Permutation matrices commute with element-wise nonlinearities (sigmoid),
/// so this is safe for gated attention unlike Hadamard rotation.
pub fn apply_outlier_permutation(weights: &mut NativeWeights, outlier_fraction: f32) -> i32 {
    // Find head_dim from first attention layer
    let (head_dim, _) = weights
        .layers
        .iter()
        .find(|lw| !lw.is_linear)
        .map(|lw| (lw.attn_head_dim, lw.attn_n_kv_heads))
        .unwrap_or((0, 0));
    if head_dim == 0 {
        return 0;
    }

    let outlier_count = (head_dim as f32 * outlier_fraction).round() as i32;
    // Round to group_size boundary for clean quantization
    let outlier_count = (outlier_count / 64) * 64;
    if outlier_count == 0 || outlier_count >= head_dim {
        return 0;
    }

    // Compute per-channel importance from K projection weights (first attn layer).
    // Use L2 norm of each output channel across the hidden dimension.
    let k_w = weights
        .layers
        .iter()
        .find(|lw| !lw.is_linear)
        .and_then(|lw| match &lw.attn_k_w {
            Some(LayerWeight::Dense(w)) => Some(w.clone()),
            _ => None,
        });
    let Some(k_weight) = k_w else { return 0 };

    // k_weight shape: [hidden, n_kv * head_dim] (pre-transposed)
    // Compute L2 norm per output channel (columns)
    let col_norms = k_weight.square().sum_axis(0, false); // [n_kv * head_dim]
    col_norms.eval();

    // Build per-head permutation: sort channels by norm (descending), outliers first.
    // The permutation is the same for all heads (same relative ordering).
    let first_head_norms = col_norms.slice(&[0], &[head_dim]);
    first_head_norms.eval();
    let norms_slice: &[f32] = first_head_norms.as_slice();
    let mut indices: Vec<usize> = (0..head_dim as usize).collect();
    indices.sort_by(|&a, &b| {
        norms_slice[b]
            .partial_cmp(&norms_slice[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build permutation matrix P [head_dim, head_dim]
    let mut p_data = vec![0.0f32; (head_dim * head_dim) as usize];
    for (new_pos, &old_pos) in indices.iter().enumerate() {
        p_data[new_pos * head_dim as usize + old_pos] = 1.0;
    }
    let p = InlineArray::from_f32_slice(&p_data, &[head_dim, head_dim]);
    let p = p.as_dtype(weights.model_dtype);
    let p_t = p.transpose_axes(&[1, 0]);

    eprintln!(
        "[PRECONDITION] Applying outlier channel permutation (outlier_count={}, head_dim={})",
        outlier_count, head_dim
    );

    // Apply P to all attention projection weights.
    // W' = W @ P^T moves outlier channels to front of each head.
    // Permutation commutes with sigmoid, so safe for gated attention.
    for lw in &mut weights.layers {
        if lw.is_linear {
            continue;
        }
        let n_heads = lw.attn_n_heads;
        let n_kv = lw.attn_n_kv_heads;

        // Q (including gate for Qwen3.5): permute query channels, gate channels
        rotate_projection_weight(&mut lw.attn_q_w, &p_t, n_heads, head_dim, lw.attn_gated);
        rotate_projection_weight(&mut lw.attn_k_w, &p_t, n_kv, head_dim, false);
        rotate_projection_weight(&mut lw.attn_v_w, &p_t, n_kv, head_dim, false);
        rotate_output_weight(&mut lw.attn_o_w, &p, n_heads, head_dim);
    }

    outlier_count
}

/// Rotate a Q/K/V projection weight in-place: `W' = W @ R_block^T`
/// For gated Q (Qwen3.5): treats both query and gate halves as heads to rotate.
fn rotate_projection_weight(
    w_opt: &mut Option<LayerWeight>,
    r_t: &InlineArray,
    n_heads: i32,
    head_dim: i32,
    gated: bool,
) {
    let w = match w_opt {
        Some(LayerWeight::Dense(w)) => w,
        _ => return,
    };
    let shape = w.shape();
    let hidden = shape[0];

    if gated {
        // Gated Q (Qwen3.5): W_q is [hidden, n_heads * head_dim * 2].
        // Per-head layout is interleaved: [head0_q(256), head0_gate(256), head1_q(256), ...].
        // After reshape to [hidden, n_heads, 2*head_dim], split at head_dim:
        //   queries = [:, :, :head_dim]  →  rotate by R^T
        //   gate    = [:, :, head_dim:]  →  leave alone (sigmoid doesn't commute)
        let w_3d = w.reshape(&[hidden, n_heads, head_dim * 2]);
        let w_queries = w_3d.slice(&[0, 0, 0], &[hidden, n_heads, head_dim]);
        let w_gate = w_3d.slice(&[0, 0, head_dim], &[hidden, n_heads, head_dim * 2]);

        let w_q_rot = w_queries.matmul(r_t); // [hidden, n_heads, head_dim] @ [head_dim, head_dim]

        // Concatenate rotated queries + original gate along last dim
        let w_combined = w_q_rot.kv_cache_append(&w_gate, 2); // [hidden, n_heads, 2*head_dim]
        *w_opt = Some(LayerWeight::Dense(
            w_combined.reshape(&[hidden, n_heads * head_dim * 2]),
        ));
    } else {
        // Standard Q/K/V: rotate all heads
        let w_3d = w.reshape(&[hidden, n_heads, head_dim]);
        let w_rot = w_3d.matmul(r_t);
        *w_opt = Some(LayerWeight::Dense(
            w_rot.reshape(&[hidden, n_heads * head_dim]),
        ));
    }
}

/// Rotate an O-projection weight in-place: `W_o' = R_block @ W_o`
fn rotate_output_weight(
    w_opt: &mut Option<LayerWeight>,
    r: &InlineArray,
    n_heads: i32,
    head_dim: i32,
) {
    let w = match w_opt {
        Some(LayerWeight::Dense(w)) => w,
        _ => return,
    };
    let shape = w.shape();
    let hidden = shape[1];
    let w_3d = w.reshape(&[n_heads, head_dim, hidden]);
    // Batched matmul: [head_dim, head_dim] @ [n_heads, head_dim, hidden] → [n_heads, head_dim, hidden]
    let w_rot = r.matmul(&w_3d);
    *w_opt = Some(LayerWeight::Dense(
        w_rot.reshape(&[n_heads * head_dim, hidden]),
    ));
}

/// - norm `(1+w)` offset when the model has `mtp.*` keys or unsanitized conv shapes
/// - Q/K scale synthesis for GDN (not stored in safetensors)
/// - MoE expert weight stacking into `[E, in, out]` for `gather_mm`
/// - `copy_fresh` on all weights (add zero + eval + detach) for fresh Metal buffers
pub fn load_model(
    model_dir: &std::path::Path,
    config: &Qwen3Config,
) -> Result<NativeWeights, String> {
    // ── Step 1: Shard discovery ─────────────────────────────────────────────
    let single_path = model_dir.join("model.safetensors");
    let index_path = model_dir.join("model.safetensors.index.json");

    let shard_paths: Vec<std::path::PathBuf> = if single_path.exists() {
        vec![single_path]
    } else if index_path.exists() {
        let content = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("failed to read index JSON: {e}"))?;
        let index: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("failed to parse index JSON: {e}"))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "index JSON missing weight_map".to_string())?;
        let mut seen = std::collections::HashSet::new();
        let mut paths = Vec::new();
        for shard_file in weight_map.values() {
            let name = shard_file
                .as_str()
                .ok_or_else(|| "shard filename is not a string".to_string())?;
            if seen.insert(name.to_string()) {
                if name.contains("..") || name.starts_with('/') {
                    return Err(format!("shard filename contains path traversal: {name}"));
                }
                paths.push(model_dir.join(name));
            }
        }
        paths
    } else {
        return Err(format!(
            "no model.safetensors or model.safetensors.index.json in {}",
            model_dir.display()
        ));
    };

    // ── Step 2: Load all tensors from all shards ────────────────────────────
    let mut raw: std::collections::HashMap<String, InlineArray> = std::collections::HashMap::new();

    for shard_path in &shard_paths {
        let path_str = shard_path
            .to_str()
            .ok_or_else(|| format!("non-UTF-8 shard path: {:?}", shard_path))?;
        let entries = bridge::load_safetensors_shard(path_str)
            .ok_or_else(|| format!("failed to load shard: {path_str}"))?;
        for (key, arr) in entries {
            raw.insert(key, arr);
        }
    }

    if raw.is_empty() {
        return Err(format!("no weights loaded from {}", model_dir.display()));
    }

    // ── Step 3: Sanitization ────────────────────────────────────────────────

    // Detect whether norm shift is needed before any renaming.
    let has_mtp = raw.keys().any(|k| k.contains("mtp."));
    let has_unsanitized_conv = raw
        .iter()
        .any(|(k, v)| k.contains("conv1d.weight") && v.ndim() == 3 && v.dim(2) != 1);
    let should_shift_norms = has_mtp || has_unsanitized_conv;

    // 3a. Key renaming: strip VLM prefix; A_log → a_log.
    // Qwen3.5-27B uses "language_model.model." prefix (no leading "model.").
    // Older checkpoints use "model.language_model." — strip both to "model.".
    // Qwen3 uses plain "model.layers.N..." with no prefix.
    let original_keys: Vec<String> = raw.keys().cloned().collect();
    for old_key in original_keys {
        let mut new_key = old_key.clone();
        if new_key.starts_with("language_model.model.") {
            // "language_model.model.X" → "model.X"
            new_key = new_key.replacen("language_model.", "", 1);
        } else if new_key.starts_with("language_model.") {
            // "language_model.lm_head.weight" → "lm_head.weight"
            new_key = new_key.replacen("language_model.", "", 1);
        } else if new_key.starts_with("model.language_model.") {
            new_key = new_key.replacen("model.language_model.", "model.", 1);
        }
        if new_key.contains(".A_log") {
            new_key = new_key.replace(".A_log", ".a_log");
        }
        if new_key != old_key {
            if let Some(v) = raw.remove(&old_key) {
                raw.insert(new_key, v);
            }
        }
    }

    // 3b. Drop mtp.* keys.
    raw.retain(|k, _| !k.contains("mtp."));

    // 3c. Drop lm_head.weight when embeddings are tied.
    if config.tie_word_embeddings {
        raw.remove("lm_head.weight");
    }

    // Norm suffixes that receive the (1+w) shift.
    // Excludes `.linear_attn.norm.weight` (the GDN sub-norm, which is NOT shifted).
    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];

    // 3d. Conv1d transpose + norm shift + f32→model_dtype casts.
    let detected_model_dtype = raw
        .get("model.embed_tokens.weight")
        .map(|w| w.dtype_raw())
        .unwrap_or(11); // 11 = bfloat16 fallback
    let one = InlineArray::from_f32(1.0).as_dtype(detected_model_dtype);

    // GDN-specific f32 weights that must be cast to model dtype to prevent f32
    // dtype propagation through the residual stream.
    let f32_gdn_suffixes = [
        "linear_attn.a_log",
        "linear_attn.norm.weight",
        "linear_attn.q_norm.weight",
        "linear_attn.k_norm.weight",
    ];

    let all_keys: Vec<String> = raw.keys().cloned().collect();
    for k in &all_keys {
        if k.contains("conv1d.weight") {
            if let Some(v) = raw.get(k) {
                if v.ndim() == 3 && v.dim(2) != 1 {
                    let transposed = v.transpose_axes(&[0, 2, 1]);
                    raw.insert(k.clone(), transposed);
                }
            }
        }
        if f32_gdn_suffixes.iter().any(|sfx| k.ends_with(sfx)) {
            if let Some(v) = raw.get(k) {
                if v.dtype_raw() != detected_model_dtype {
                    let cast = v.as_dtype(detected_model_dtype);
                    raw.insert(k.clone(), cast);
                }
            }
        }
        if should_shift_norms && norm_suffixes.iter().any(|sfx| k.ends_with(sfx)) {
            if let Some(v) = raw.get(k) {
                if v.ndim() == 1 {
                    let shifted = v.add(&one);
                    raw.insert(k.clone(), shifted);
                }
            }
        }
    }

    // 3e. MoE expert weight stacking / normalization.
    //
    // Python's sanitize() stacks per-expert weights into [E, out, in]:
    //   to_join = [weights[f"{prefix}.experts.{e}.{n}.weight"] for e in range(E)]
    //   weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)
    //
    // Newer Qwen3.5 MoE checkpoints already ship experts in a packed form:
    //   - `{prefix}.experts.gate_up_proj` : [E, 2H, in]
    //   - `{prefix}.experts.down_proj`    : [E, out, H]
    // We normalize both layouts into the same `switch_mlp.*` keys so the hot
    // path stays identical.
    //
    // For dense weights:
    //   `gather_mm` in SwitchLinear calls `weight.swapaxes(-1,-2)` at forward
    //   time (so it uses [E, in, out] at runtime).  We pre-transpose to
    //   [E, in, out] here so the forward pass can call gather_mm directly
    //   without the extra transpose node.
    //
    // For quantized weights:
    //   Each expert has `.weight`, `.scales`, `.biases`.
    //   We stack weight: [E, out, in_packed] — no transpose (gather_qmm handles layout).
    //   We stack scales: [E, out, in/group_size].
    //   We stack biases: [E, out, in/group_size].
    //   These are stored under "switch_mlp.{proj}.weight/.scales/.biases".
    //
    // We store the stacked tensors under "switch_mlp.{n}.weight" keys matching
    // the post-sanitize Python layout for clarity, then look them up by layer.
    if config.is_moe() {
        for li in 0..config.num_hidden_layers as usize {
            if config.is_dense_mlp_layer(li) {
                continue;
            }
            let prefix = format!("model.layers.{li}.mlp");
            let packed_gate_up_key = format!("{prefix}.experts.gate_up_proj");
            let packed_down_key = format!("{prefix}.experts.down_proj");
            if raw.contains_key(&packed_gate_up_key) && raw.contains_key(&packed_down_key) {
                let gate_up = raw.remove(&packed_gate_up_key).ok_or_else(|| {
                    format!("MoE: missing packed expert tensor {packed_gate_up_key}")
                })?;
                let down = raw.remove(&packed_down_key).ok_or_else(|| {
                    format!("MoE: missing packed expert tensor {packed_down_key}")
                })?;
                let hidden = config.moe_intermediate_size;
                if hidden <= 0 {
                    return Err(format!(
                        "MoE: invalid moe_intermediate_size={} for packed expert weights at layer {li}",
                        hidden
                    ));
                }

                // gate_up: [E, 2H, in] -> split into gate/up [E, H, in], then
                // transpose to [E, in, H] for gather_mm.
                let gate = gate_up
                    .index((.., 0..hidden, ..))
                    .transpose_axes(&[0, 2, 1]);
                let up = gate_up
                    .index((.., hidden..(hidden * 2), ..))
                    .transpose_axes(&[0, 2, 1]);
                // down: [E, out, H] -> [E, H, out] for gather_mm.
                let down = down.transpose_axes(&[0, 2, 1]);

                raw.insert(format!("{prefix}.switch_mlp.gate_proj.weight"), gate);
                raw.insert(format!("{prefix}.switch_mlp.up_proj.weight"), up);
                raw.insert(format!("{prefix}.switch_mlp.down_proj.weight"), down);
                continue;
            }

            for proj in &["gate_proj", "up_proj", "down_proj"] {
                // Detect whether first expert is quantized.
                let is_quantized = raw.contains_key(&format!("{prefix}.experts.0.{proj}.scales"));

                if is_quantized {
                    // Quantized: stack weight, scales, biases separately.
                    let mut w_shards: Vec<InlineArray> =
                        Vec::with_capacity(config.num_experts as usize);
                    let mut s_shards: Vec<InlineArray> =
                        Vec::with_capacity(config.num_experts as usize);
                    let mut b_shards: Vec<InlineArray> =
                        Vec::with_capacity(config.num_experts as usize);

                    for e in 0..config.num_experts as usize {
                        let wk = format!("{prefix}.experts.{e}.{proj}.weight");
                        let sk = format!("{prefix}.experts.{e}.{proj}.scales");
                        let bk = format!("{prefix}.experts.{e}.{proj}.biases");
                        w_shards.push(
                            raw.remove(&wk)
                                .ok_or_else(|| format!("MoE quant: missing {wk}"))?,
                        );
                        s_shards.push(
                            raw.remove(&sk)
                                .ok_or_else(|| format!("MoE quant: missing {sk}"))?,
                        );
                        b_shards.push(
                            raw.remove(&bk)
                                .ok_or_else(|| format!("MoE quant: missing {bk}"))?,
                        );
                    }

                    // Stack: weight [E, out, in_packed], scales [E, out, in/g], biases same.
                    // For quantized gather_qmm the weight layout is [E, out, in_packed]
                    // (no pre-transpose — gather_qmm handles the implicit transpose via
                    // the `transpose=true` flag we pass in LayerWeight::gather_mm_from).
                    let w_stacked = stack_arrays(w_shards, 0)?;
                    let s_stacked = stack_arrays(s_shards, 0)?;
                    let b_stacked = stack_arrays(b_shards, 0)?;

                    raw.insert(format!("{prefix}.switch_mlp.{proj}.weight"), w_stacked);
                    raw.insert(format!("{prefix}.switch_mlp.{proj}.scales"), s_stacked);
                    raw.insert(format!("{prefix}.switch_mlp.{proj}.biases"), b_stacked);
                } else {
                    // Dense: collect and pre-transpose to [E, in, out] for gather_mm.
                    let mut shards: Vec<InlineArray> =
                        Vec::with_capacity(config.num_experts as usize);
                    for e in 0..config.num_experts as usize {
                        let key = format!("{prefix}.experts.{e}.{proj}.weight");
                        let w = raw
                            .remove(&key)
                            .ok_or_else(|| format!("MoE: missing expert weight {key}"))?;
                        shards.push(w);
                    }
                    // Stack [E, out, in] → then transpose to [E, in, out] for gather_mm.
                    let stacked = stack_arrays(shards, 0)?;
                    let transposed = stacked.transpose_axes(&[0, 2, 1]);
                    raw.insert(format!("{prefix}.switch_mlp.{proj}.weight"), transposed);
                }
            }
        }
    }

    // ── Step 4: Build per-layer weight structs ──────────────────────────────
    // Quantization params — present only in quantized checkpoints.
    let (q_bits, q_group_size) = config
        .quantization_config
        .as_ref()
        .map(|qc| (qc.bits, qc.group_size))
        .unwrap_or((4, 64));
    validate_quantization_runtime_support(q_bits)?;

    let get = |key: &str| -> Result<InlineArray, String> {
        raw.get(key).cloned().ok_or_else(|| {
            let parts: Vec<&str> = key.rsplitn(2, '.').collect();
            let suffix = parts[0];
            let close: Vec<&String> = raw.keys().filter(|k| k.ends_with(suffix)).take(5).collect();
            format!("missing weight key: {key} (close matches: {close:?})")
        })
    };

    // Load a projection weight as `LayerWeight`, auto-detecting quantized format.
    //
    // For a dense weight stored at `{base_key}.weight`, looks up sibling keys
    // `{base_key}.scales` and `{base_key}.biases`.  When both are present the
    // weight is loaded as `LayerWeight::Quantized` (no transpose — the caller
    // passes `transpose=true` to `quantized_matmul` at runtime).  Otherwise
    // falls back to `LayerWeight::Dense` with the weight pre-transposed.
    //
    // `base_key` is the full key WITHOUT the trailing `.weight` suffix.
    let get_layer_weight = |base_key: &str| -> Result<LayerWeight, String> {
        let w_key = format!("{base_key}.weight");
        let s_key = format!("{base_key}.scales");
        let b_key = format!("{base_key}.biases");
        match (raw.get(&w_key), raw.get(&s_key), raw.get(&b_key)) {
            (Some(w), Some(s), Some(b)) => Ok(LayerWeight::Quantized {
                weight: w.clone(),
                scales: s.clone(),
                biases: b.clone(),
                group_size: q_group_size,
                bits: q_bits,
            }),
            _ => {
                // Dense path: load `.weight` and pre-transpose for matmul.
                let w = get(&w_key)?;
                Ok(LayerWeight::Dense(w.t()))
            }
        }
    };

    let embed_w = get("model.embed_tokens.weight")?;
    let embed_scales = get("model.embed_tokens.scales").ok();
    let embed_biases = get("model.embed_tokens.biases").ok();
    let final_norm_w = get("model.norm.weight")?;
    let final_norm_eps = config.rms_norm_eps;
    let lm_head_w = if config.tie_word_embeddings {
        None
    } else {
        Some(get_layer_weight("lm_head")?)
    };

    let model_dtype = embed_w.dtype_raw();

    // GDN dimensions (identical across all GDN layers; only meaningful for Qwen3.5).
    let nv = config.gdn_nv();
    let nk = config.gdn_nk();
    let dk = config.gdn_dk();
    let dv = config.gdn_dv();
    let ck = config.linear_conv_kernel_dim;
    let kd = nk * dk; // total key dim
    let cd = kd * 2 + nv * dv; // conv projection dim

    // Attention dimensions.
    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.get_num_kv_heads();
    let head_dim = config.get_head_dim();
    let attn_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let rope_dims = config.rope_dims();
    let rope_base = config.rope_theta as f32;
    let rope_scale = 1.0_f32;
    let attn_gated = !config.is_qwen3_dense();

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);

    for li in 0..config.num_hidden_layers as usize {
        let p = format!("model.layers.{li}");
        let is_linear = config.is_linear_layer(li);
        let is_moe_layer = !config.is_dense_mlp_layer(li);

        let input_ln_w = get(&format!("{p}.input_layernorm.weight"))?;
        let post_ln_w = get(&format!("{p}.post_attention_layernorm.weight"))?;

        // MLP weights — dense or MoE.
        let (
            mlp_gate_w,
            mlp_up_w,
            mlp_down_w,
            moe_router_w,
            moe_gate_w,
            moe_up_w,
            moe_down_w,
            shared_gate_w,
            shared_up_w,
            shared_down_w,
            shared_expert_gate_w,
        ) = if is_moe_layer {
            // MoE layer: router + stacked experts + shared expert.
            // router gate: stored as [num_experts, hidden]; transpose to [hidden, num_experts].
            // The router is a tiny matrix and is never quantized.
            let router = get(&format!("{p}.mlp.gate.weight"))?.t();

            // Stacked expert weights were placed under "switch_mlp.{proj}.weight" during
            // Step 3e above.  For quantized models the stacking loop also placed
            // "switch_mlp.{proj}.scales" and ".biases" — check for those here.
            let moe_gate = get_stacked_expert_weight(
                &raw,
                &format!("{p}.mlp.switch_mlp.gate_proj"),
                q_group_size,
                q_bits,
            )?;
            let moe_up = get_stacked_expert_weight(
                &raw,
                &format!("{p}.mlp.switch_mlp.up_proj"),
                q_group_size,
                q_bits,
            )?;
            let moe_down = get_stacked_expert_weight(
                &raw,
                &format!("{p}.mlp.switch_mlp.down_proj"),
                q_group_size,
                q_bits,
            )?;

            // Shared expert weights — may be quantized.
            let sh_gate = get_layer_weight(&format!("{p}.mlp.shared_expert.gate_proj"))?;
            let sh_up = get_layer_weight(&format!("{p}.mlp.shared_expert.up_proj"))?;
            let sh_down = get_layer_weight(&format!("{p}.mlp.shared_expert.down_proj"))?;
            // Shared expert gate: [1, hidden]; tiny matrix, never quantized.
            let sh_eg = get(&format!("{p}.mlp.shared_expert_gate.weight"))?.t();
            (
                None,
                None,
                None,
                Some(router),
                Some(moe_gate),
                Some(moe_up),
                Some(moe_down),
                Some(sh_gate),
                Some(sh_up),
                Some(sh_down),
                Some(sh_eg),
            )
        } else {
            // Dense MLP — may be quantized.
            let gate = get_layer_weight(&format!("{p}.mlp.gate_proj"))?;
            let up = get_layer_weight(&format!("{p}.mlp.up_proj"))?;
            let down = get_layer_weight(&format!("{p}.mlp.down_proj"))?;
            (
                Some(gate),
                Some(up),
                Some(down),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        };

        let mut lw = LayerWeights {
            is_linear,
            input_ln_w,
            input_ln_eps: config.rms_norm_eps,
            post_ln_w,
            post_ln_eps: config.rms_norm_eps,
            mlp_gate_w,
            mlp_up_w,
            mlp_down_w,
            moe_router_w,
            moe_gate_w,
            moe_up_w,
            moe_down_w,
            shared_gate_w,
            shared_up_w,
            shared_down_w,
            shared_expert_gate_w,
            moe_top_k: config.num_experts_per_tok,
            moe_norm_topk_prob: config.norm_topk_prob,
            is_moe_layer,
            // Attention — filled below when !is_linear
            attn_q_w: None,
            attn_k_w: None,
            attn_v_w: None,
            attn_o_w: None,
            attn_q_norm_w: None,
            attn_q_norm_eps: config.rms_norm_eps,
            attn_k_norm_w: None,
            attn_k_norm_eps: config.rms_norm_eps,
            attn_n_heads: 0,
            attn_n_kv_heads: 0,
            attn_head_dim: 0,
            attn_scale: 0.0,
            attn_rope_dims: 0,
            attn_rope_base: 0.0,
            attn_rope_scale: 0.0,
            attn_gated,
            // GDN — filled below when is_linear
            gdn_qkv_w: None,
            gdn_z_w: None,
            gdn_b_w: None,
            gdn_a_w: None,
            gdn_conv_w: None,
            gdn_q_nw: None,
            gdn_k_nw: None,
            gdn_a_log: None,
            gdn_dt_bias: None,
            gdn_norm_w: None,
            gdn_norm_eps: config.rms_norm_eps,
            gdn_out_w: None,
            gdn_nv: 0,
            gdn_nk: 0,
            gdn_dk: 0,
            gdn_dv: 0,
            gdn_kd: 0,
            gdn_cd: 0,
            gdn_ck: 0,
        };

        if is_linear {
            let la = format!("{p}.linear_attn");
            // in_proj and out_proj are large matrices — can be quantized.
            lw.gdn_qkv_w = Some(get_layer_weight(&format!("{la}.in_proj_qkv"))?);
            lw.gdn_z_w = Some(get_layer_weight(&format!("{la}.in_proj_z"))?);
            lw.gdn_b_w = Some(get_layer_weight(&format!("{la}.in_proj_b"))?);
            lw.gdn_a_w = Some(get_layer_weight(&format!("{la}.in_proj_a"))?);
            // conv1d weight is a small 3D tensor — never quantized.
            lw.gdn_conv_w = Some(get(&format!("{la}.conv1d.weight"))?);

            // Q/K scale weights are SYNTHETIC — not present in safetensors.
            // They encode the Q/K normalization scaling factor:
            //   q_norm_weight = ones(dk) * inv_scale²
            //   k_norm_weight = ones(dk) * inv_scale
            // where inv_scale = 1/sqrt(dk).
            // IMPORTANT: the scalar MUST be cast to model_dtype before multiply.
            let inv_scale = (dk as f32).sqrt().recip();
            let q_scale_arr = {
                let a = InlineArray::ones(&[dk], model_dtype);
                let scale = InlineArray::from_f32(inv_scale * inv_scale).as_dtype(model_dtype);
                a.multiply(&scale)
            };
            let k_scale_arr = {
                let a = InlineArray::ones(&[dk], model_dtype);
                let scale = InlineArray::from_f32(inv_scale).as_dtype(model_dtype);
                a.multiply(&scale)
            };
            lw.gdn_q_nw = Some(
                get(&format!("{la}.q_norm_weight"))
                    .or_else(|_| get(&format!("{la}.q_norm.weight")))
                    .unwrap_or(q_scale_arr),
            );
            lw.gdn_k_nw = Some(
                get(&format!("{la}.k_norm_weight"))
                    .or_else(|_| get(&format!("{la}.k_norm.weight")))
                    .unwrap_or(k_scale_arr),
            );
            lw.gdn_a_log = Some(get(&format!("{la}.a_log"))?);
            lw.gdn_dt_bias = Some(get(&format!("{la}.dt_bias"))?);
            lw.gdn_norm_w = Some(get(&format!("{la}.norm.weight"))?);
            // out_proj is a large matrix — can be quantized.
            lw.gdn_out_w = Some(get_layer_weight(&format!("{la}.out_proj"))?);
            lw.gdn_nv = nv;
            lw.gdn_nk = nk;
            lw.gdn_dk = dk;
            lw.gdn_dv = dv;
            lw.gdn_kd = kd;
            lw.gdn_cd = cd;
            lw.gdn_ck = ck;

            // Log GDN config once (first linear-attention layer only).
            // Uses stderr to avoid interleaving with streamed stdout output.
        } else {
            let sa = format!("{p}.self_attn");
            // Q projection width differs between Qwen3 and Qwen3.5:
            //   Qwen3:   [n_heads * head_dim, hidden]
            //   Qwen3.5: [n_heads * head_dim * 2, hidden]  (queries + gate concatenated)
            // We just load whatever is in the checkpoint; the forward pass
            // uses `attn_gated` to decide how to interpret the output.
            lw.attn_q_w = Some(get_layer_weight(&format!("{sa}.q_proj"))?);
            lw.attn_k_w = Some(get_layer_weight(&format!("{sa}.k_proj"))?);
            lw.attn_v_w = Some(get_layer_weight(&format!("{sa}.v_proj"))?);
            lw.attn_o_w = Some(get_layer_weight(&format!("{sa}.o_proj"))?);
            // Q/K norms are 1D scale vectors — never quantized.
            lw.attn_q_norm_w = Some(get(&format!("{sa}.q_norm.weight"))?);
            lw.attn_k_norm_w = Some(get(&format!("{sa}.k_norm.weight"))?);
            lw.attn_n_heads = n_heads;
            lw.attn_n_kv_heads = n_kv_heads;
            lw.attn_head_dim = head_dim;
            lw.attn_scale = attn_scale;
            lw.attn_rope_dims = rope_dims;
            lw.attn_rope_base = rope_base;
            lw.attn_rope_scale = rope_scale;
        }

        layers.push(lw);
    }

    // ── Step 5: copy_fresh — force all weights into fresh Metal buffers ─────
    //
    // For dense InlineArray weights: `w.add(zero).eval().detach()` creates a
    // new buffer with use_count=1.  Without this, weights share data with the
    // raw tensors (use_count=2), preventing optimal Metal buffer scheduling.
    //
    // For LayerWeight: `LayerWeight::copy_fresh` handles each tensor (weight,
    // scales, biases) independently using a same-dtype zero.
    //
    // Drop the raw safetensors handle map first. Large checkpoints like
    // Qwen3.5-35B-A3B otherwise keep both the raw handle set and the copied
    // weights live through the entire pass, needlessly spiking load-time peak
    // memory and causing benchmark/process kills under pressure.
    drop(raw);
    let zero = InlineArray::from_f32(0.0).as_dtype(model_dtype);
    let cf_arr = |w: &InlineArray| -> InlineArray { copy_fresh_arr(w, &zero) };
    let cf_lw = |w: &LayerWeight| -> LayerWeight { w.copy_fresh(&zero) };

    let embed_w = cf_arr(&embed_w);
    let final_norm_w = cf_arr(&final_norm_w);
    let lm_head_w = lm_head_w.map(|w| cf_lw(&w));

    for lw in &mut layers {
        lw.input_ln_w = cf_arr(&lw.input_ln_w);
        lw.post_ln_w = cf_arr(&lw.post_ln_w);
        // Dense MLP (LayerWeight)
        if let Some(ref w) = lw.mlp_gate_w {
            lw.mlp_gate_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.mlp_up_w {
            lw.mlp_up_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.mlp_down_w {
            lw.mlp_down_w = Some(cf_lw(w));
        }
        // MoE — router is InlineArray; stacked expert weights are LayerWeight
        if let Some(ref w) = lw.moe_router_w {
            lw.moe_router_w = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.moe_gate_w {
            lw.moe_gate_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.moe_up_w {
            lw.moe_up_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.moe_down_w {
            lw.moe_down_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.shared_gate_w {
            lw.shared_gate_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.shared_up_w {
            lw.shared_up_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.shared_down_w {
            lw.shared_down_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.shared_expert_gate_w {
            lw.shared_expert_gate_w = Some(cf_arr(w));
        }
        // Attention projections (LayerWeight); norms are InlineArray
        if let Some(ref w) = lw.attn_q_w {
            lw.attn_q_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.attn_k_w {
            lw.attn_k_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.attn_v_w {
            lw.attn_v_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.attn_o_w {
            lw.attn_o_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.attn_q_norm_w {
            lw.attn_q_norm_w = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.attn_k_norm_w {
            lw.attn_k_norm_w = Some(cf_arr(w));
        }
        // GDN projections (LayerWeight); small tensors are InlineArray
        if let Some(ref w) = lw.gdn_qkv_w {
            lw.gdn_qkv_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.gdn_z_w {
            lw.gdn_z_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.gdn_b_w {
            lw.gdn_b_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.gdn_a_w {
            lw.gdn_a_w = Some(cf_lw(w));
        }
        if let Some(ref w) = lw.gdn_conv_w {
            lw.gdn_conv_w = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_q_nw {
            lw.gdn_q_nw = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_k_nw {
            lw.gdn_k_nw = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_a_log {
            lw.gdn_a_log = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_dt_bias {
            lw.gdn_dt_bias = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_norm_w {
            lw.gdn_norm_w = Some(cf_arr(w));
        }
        if let Some(ref w) = lw.gdn_out_w {
            lw.gdn_out_w = Some(cf_lw(w));
        }
    }

    // Determine quantization mode for diagnostic output.
    let _quant_mode = if let Some(ref qc) = config.quantization_config {
        // Inspect the first projection weight to confirm quantized loading succeeded.
        let confirmed = weights_are_quantized(&layers);
        format!(
            "quantized {}-bit (group_size={}) confirmed={}",
            qc.bits, qc.group_size, confirmed,
        )
    } else {
        "dense bf16".to_string()
    };

    Ok(NativeWeights {
        embed_w,
        embed_scales,
        embed_biases,
        final_norm_w,
        final_norm_eps,
        lm_head_w,
        tie_word_embeddings: config.tie_word_embeddings,
        quantization_config: config.quantization_config.clone(),
        layers,
        model_dtype,
        qjl_matrix: None, // populated by apply_qjl_matrix() when --kv-qjl is set
    })
}

// ============================================================================
// Forward step
// ============================================================================

/// Run one forward step — works for both T=1 decode and T=N prefill.
///
/// `token_ids` must be shape `[B, T]` int32. Returns logits `[B, T, vocab]`.
///
/// For T=1 the GDN path uses `compiled_gdn_layer_fixed` (tape-replay, ~10 ms
/// cheaper per step). For T>1 it falls through to the direct-ops prefill path.
pub fn forward_step(
    weights: &NativeWeights,
    token_ids: &InlineArray, // [B, T]
    cache: &mut NativeCache,
) -> InlineArray {
    let b = token_ids.dim(0);
    let s = token_ids.dim(1);
    let dtype = weights.model_dtype;

    // Debug removed
    // Embedding lookup: [B, T, hidden]
    // For quantized models: index into weight/scales/biases rows, then dequantize.
    // Matches Python's QuantizedEmbedding: dequantize(weight[x], scales[x], biases[x])
    let mut hidden =
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            let w_rows = weights.embed_w.take_axis(token_ids, 0); // [B, T, hidden/pack]
            let s_rows = scales.take_axis(token_ids, 0); // [B, T, hidden/group_size]
            let b_rows = biases.take_axis(token_ids, 0); // [B, T, hidden/group_size]
            w_rows.dequantize(&s_rows, &b_rows, gs, bits) // [B, T, hidden] bf16
        } else {
            weights.embed_w.take_axis(token_ids, 0)
        };
    let trace_qwen35 = std::env::var_os("PMETAL_TRACE_QWEN35").is_some();

    let mut gdn_slot = 0usize;
    let mut attn_slot = 0usize;

    for (layer_idx, lw) in weights.layers.iter().enumerate() {
        if trace_qwen35 {
            eprintln!(
                "[QWEN35 TRACE] layer={layer_idx} start linear={} moe={} rope_offset={} seq={s}",
                lw.is_linear, lw.is_moe_layer, cache.rope_offset
            );
        }
        // Input LayerNorm
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.input_ln_eps);

        let r = if lw.is_linear {
            let result = gdn_forward(lw, &normed, b, s, &mut cache.gdn_caches[gdn_slot], dtype);
            gdn_slot += 1;
            result
        } else {
            let result = attn_forward(
                lw,
                &normed,
                b,
                s,
                &mut cache.kv_caches[attn_slot],
                cache.rope_offset,
                dtype,
                weights.qjl_matrix.as_ref(),
            );
            attn_slot += 1;
            result
        };
        if trace_qwen35 {
            eprintln!("[QWEN35 TRACE] layer={layer_idx} after_attention");
        }

        // Residual
        let h = hidden.add(&r);

        // Post-attention LayerNorm + MLP (dense or MoE)
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.post_ln_eps);
        let mlp_out = if lw.is_moe_layer {
            moe_forward(lw, &mlp_in)
        } else {
            dense_mlp_forward(lw, &mlp_in)
        };
        if trace_qwen35 {
            eprintln!("[QWEN35 TRACE] layer={layer_idx} after_mlp");
        }

        // Residual
        hidden = h.add(&mlp_out);
    }

    // Advance position counter
    cache.rope_offset += s;

    // Final norm + LM head
    let hidden = hidden.rms_norm(Some(&weights.final_norm_w), weights.final_norm_eps);
    if weights.tie_word_embeddings {
        // For quantized models: use quantized_matmul with the packed embedding weight
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            hidden.quantized_matmul(&weights.embed_w, scales, Some(biases), true, gs, bits)
        } else {
            hidden.matmul(&weights.embed_w.t())
        }
    } else {
        weights.lm_head_w.as_ref().unwrap().matmul_from(&hidden)
    }
}

/// Variant of [`forward_step`] that tees post-layer hidden states at the
/// indices listed in `tap_layers` into `captured`, in ascending-layer order.
///
/// The extra per-layer check is just a `contains` on a tiny vec and a
/// refcount bump (`InlineArray` is a thin handle around an `mx::array`), so
/// this path is within a few percent of the non-capture `forward_step`. It
/// is intentionally a separate entry point rather than a generic hook so the
/// hot path stays unbranched for non-DFlash callers.
///
/// `captured` is populated in the SAME ORDER as `tap_layers`. Callers that
/// want the concatenated target-hidden tensor for a DFlash draft should
/// concatenate along the last axis after this returns:
///
/// ```ignore
/// let refs: Vec<&InlineArray> = captured.iter().collect();
/// let target_hidden = ops::concatenate_axis(&refs, -1);
/// ```
pub fn forward_step_with_capture(
    weights: &NativeWeights,
    token_ids: &InlineArray,
    cache: &mut NativeCache,
    tap_layers: &[usize],
    captured: &mut Vec<InlineArray>,
) -> InlineArray {
    captured.clear();
    let b = token_ids.dim(0);
    let s = token_ids.dim(1);
    let dtype = weights.model_dtype;

    let mut hidden =
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            let w_rows = weights.embed_w.take_axis(token_ids, 0);
            let s_rows = scales.take_axis(token_ids, 0);
            let b_rows = biases.take_axis(token_ids, 0);
            w_rows.dequantize(&s_rows, &b_rows, gs, bits)
        } else {
            weights.embed_w.take_axis(token_ids, 0)
        };

    let mut gdn_slot = 0usize;
    let mut attn_slot = 0usize;

    for (layer_idx, lw) in weights.layers.iter().enumerate() {
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.input_ln_eps);
        let r = if lw.is_linear {
            let result = gdn_forward(lw, &normed, b, s, &mut cache.gdn_caches[gdn_slot], dtype);
            gdn_slot += 1;
            result
        } else {
            let result = attn_forward(
                lw,
                &normed,
                b,
                s,
                &mut cache.kv_caches[attn_slot],
                cache.rope_offset,
                dtype,
                weights.qjl_matrix.as_ref(),
            );
            attn_slot += 1;
            result
        };
        let h = hidden.add(&r);
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.post_ln_eps);
        let mlp_out = if lw.is_moe_layer {
            moe_forward(lw, &mlp_in)
        } else {
            dense_mlp_forward(lw, &mlp_in)
        };
        hidden = h.add(&mlp_out);

        // Tap hidden state POST-residual, matching mlx-lm's
        // `hidden_states = layer(...)` + `if idx in target_layer_ids:
        // selected_hidden_states.append(hidden_states)` pattern in
        // `dflash_mlx/adapters.py`.
        if tap_layers.contains(&layer_idx) {
            captured.push(hidden.clone());
        }
    }

    cache.rope_offset += s;

    let hidden = hidden.rms_norm(Some(&weights.final_norm_w), weights.final_norm_eps);
    if weights.tie_word_embeddings {
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            hidden.quantized_matmul(&weights.embed_w, scales, Some(biases), true, gs, bits)
        } else {
            hidden.matmul(&weights.embed_w.t())
        }
    } else {
        weights.lm_head_w.as_ref().unwrap().matmul_from(&hidden)
    }
}

/// Tree-verify variant of [`forward_step_with_capture`]. Each token
/// carries its own position id (from the DDTree tree compile) and the
/// target's attention is gated by a custom additive mask encoding the
/// tree visibility. All tapped layers are captured, same as the
/// linear variant, so the DFlash draft can condition its next round
/// on the accepted path's hidden states.
///
/// Only the plain-bf16 KV cache path is wired today. TurboQuant /
/// quant-config modes silently fall back to their existing causal
/// paths — the tree context is ignored for those (and the caller gets
/// a warning via the normal `cache.turboquant.is_some()` routing).
pub fn forward_step_tree_verify(
    weights: &NativeWeights,
    token_ids: &InlineArray,
    cache: &mut NativeCache,
    position_ids: &InlineArray,
    attention_mask: &InlineArray,
    tap_layers: &[usize],
    captured: &mut Vec<InlineArray>,
) -> InlineArray {
    captured.clear();
    let b = token_ids.dim(0);
    let s = token_ids.dim(1);
    let dtype = weights.model_dtype;
    // Compile_tree builds `position_ids` as [1, S] for consistency
    // with `token_ids`. Squeeze the batch axis to 1D — MLX's rope
    // array-offset API expects offset shape `[batch]` where batch is
    // the rope kernel's batch dim. After our [1,H,T,D]→[T,H,1,D]
    // transpose trick, the rope batch dim has T elements.
    let pos_ids_1d = if position_ids.ndim() == 2 && position_ids.dim(0) == 1 {
        position_ids.reshape(&[position_ids.dim(1)])
    } else {
        position_ids.clone()
    };
    let tree_ctx = TreeVerifyInputs {
        pos_ids: &pos_ids_1d,
        attention_mask,
    };

    let mut hidden =
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            let w_rows = weights.embed_w.take_axis(token_ids, 0);
            let s_rows = scales.take_axis(token_ids, 0);
            let b_rows = biases.take_axis(token_ids, 0);
            w_rows.dequantize(&s_rows, &b_rows, gs, bits)
        } else {
            weights.embed_w.take_axis(token_ids, 0)
        };

    let mut gdn_slot = 0usize;
    let mut attn_slot = 0usize;

    for (layer_idx, lw) in weights.layers.iter().enumerate() {
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.input_ln_eps);
        let r = if lw.is_linear {
            // Linear-attn layers in Qwen3.5 do not yet support tree
            // verify; they process the sequence recurrently with no
            // meaningful "mask" analogue. Fall back to scalar offset.
            let result = gdn_forward(lw, &normed, b, s, &mut cache.gdn_caches[gdn_slot], dtype);
            gdn_slot += 1;
            result
        } else {
            let result = attn_forward_with_tree_ctx(
                lw,
                &normed,
                b,
                s,
                &mut cache.kv_caches[attn_slot],
                cache.rope_offset,
                dtype,
                weights.qjl_matrix.as_ref(),
                Some(tree_ctx),
            );
            attn_slot += 1;
            result
        };
        let h = hidden.add(&r);
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.post_ln_eps);
        let mlp_out = if lw.is_moe_layer {
            moe_forward(lw, &mlp_in)
        } else {
            dense_mlp_forward(lw, &mlp_in)
        };
        hidden = h.add(&mlp_out);

        if tap_layers.contains(&layer_idx) {
            captured.push(hidden.clone());
        }
    }

    // Advance the global rope offset by the full tree length so that
    // `rollback_cache` (which subtracts from BOTH `rope_offset` and
    // each `kv.offset`) leaves the cache in a consistent state when
    // the caller wants to discard the tree write entirely (the
    // tree-pick + linear-commit path). Compact_tree_cache, used by
    // the tree-only commit path, also still works because it
    // unconditionally OVERWRITES `cache.rope_offset` to
    // `past_length + accepted_count`.
    cache.rope_offset += s;

    let hidden = hidden.rms_norm(Some(&weights.final_norm_w), weights.final_norm_eps);
    if weights.tie_word_embeddings {
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            hidden.quantized_matmul(&weights.embed_w, scales, Some(biases), true, gs, bits)
        } else {
            hidden.matmul(&weights.embed_w.t())
        }
    } else {
        weights.lm_head_w.as_ref().unwrap().matmul_from(&hidden)
    }
}

/// Compact the target KV cache after a tree-verify round: the verify
/// forward wrote `tree_length` (full tree) keys/values into the cache
/// starting at `past_length`; this function gathers the accepted
/// indices (relative to the tree's local index space 0..tree_length)
/// and rewrites them contiguously at the start of the appended
/// window. Advances `cache.offset` by `accepted_indices.len()` on
/// each layer and sets `cache.rope_offset` accordingly.
///
/// Matches DDTree's `compact_dynamic_cache` but operates on the
/// native `KvLayerCache` buffers directly: we use `take_axis` for the
/// row gather and `slice_set` to write the compacted rows back.
pub fn compact_tree_cache(
    cache: &mut NativeCache,
    past_length: i32,
    tree_length: i32,
    accepted_indices: &[usize],
) {
    if accepted_indices.is_empty() {
        return;
    }
    let keep_len = accepted_indices.len() as i32;
    let kept_indices: Vec<i32> = accepted_indices
        .iter()
        .map(|&i| past_length + i as i32)
        .collect();
    let idx_arr = InlineArray::from_i32_slice(&kept_indices);

    for kv in cache.kv_caches.iter_mut() {
        let Some(k_buf) = kv.keys.take() else { continue };
        let Some(v_buf) = kv.values.take() else { continue };
        let shape = k_buf.shape().to_vec();
        let n_kv = shape[1];
        let head_dim = shape[3];

        // Gather the accepted rows from along the seq axis using
        // `take_axis(axis=2)`. Result is [B, Hkv, keep_len, D].
        let kept_k = k_buf.take_axis(&idx_arr, 2);
        let kept_v = v_buf.take_axis(&idx_arr, 2);

        // Write the gathered rows back starting at `past_length` via
        // `slice_set`. This leaves any rows past `past_length +
        // keep_len` in the buffer untouched — the rejected
        // verify-block keys/values. They'll get overwritten by the
        // next round's writes since `offset` is now lower.
        let start = [0, 0, past_length, 0];
        let stop = [1, n_kv, past_length + keep_len, head_dim];
        let new_k = k_buf.slice_set(&kept_k, &start, &stop);
        let new_v = v_buf.slice_set(&kept_v, &start, &stop);

        kv.keys = Some(new_k);
        kv.values = Some(new_v);
        kv.offset = past_length + keep_len;
    }
    cache.rope_offset = past_length + keep_len;
    // `tree_length` is the total number of positions the verify forward
    // wrote beyond `past_length`. `accepted_indices.len()` is how many
    // we committed. The unused slots (tree_length - keep_len) stay
    // allocated but are now beyond `offset` — effectively garbage
    // that the next write will clobber.
    let _ = tree_length;
}

/// Rollback the target KV state by `n` positions. Mirrors
/// [`pmetal_mlx::kv_cache::KVCache::rollback`] but operates on the native
/// cache — decrements `rope_offset` on the cache and on each KV layer,
/// leaving the underlying buffer intact. Subsequent writes refill the
/// rejected slots.
///
/// Caller is responsible for ensuring the cache is in a quiescent state
/// (no pending ops referencing the rolled-back slots).
pub fn rollback_cache(cache: &mut NativeCache, n: i32) {
    if n <= 0 {
        return;
    }
    cache.rope_offset = cache.rope_offset.saturating_sub(n);
    for kv in cache.kv_caches.iter_mut() {
        kv.offset = kv.offset.saturating_sub(n);
    }
}

// ============================================================================
// Dense MLP forward
// ============================================================================

#[inline(always)]
fn dense_mlp_forward(lw: &LayerWeights, mlp_in: &InlineArray) -> InlineArray {
    let gate = lw.mlp_gate_w.as_ref().unwrap().matmul_from(mlp_in);
    let up = lw.mlp_up_w.as_ref().unwrap().matmul_from(mlp_in);
    let activated = InlineArray::fused_swiglu(&gate, &up);
    lw.mlp_down_w.as_ref().unwrap().matmul_from(&activated)
}


// ============================================================================
// MoE forward
// ============================================================================
//
// Mirrors Python's Qwen3NextSparseMoeBlock.__call__:
//
//   gates = softmax(gate(x), axis=-1, precise=True)
//   inds  = argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
//   scores = take_along_axis(gates, inds, axis=-1)
//   if norm_topk_prob: scores /= scores.sum(-1, keepdims=True)
//   y = switch_mlp(x, inds)                        # gather_mm
//   y = (y * scores[..., None]).sum(-2)
//   shared_y = shared_expert(x)
//   shared_y = sigmoid(shared_expert_gate(x)) * shared_y
//   return y + shared_y
//
// Input x: [B, T, hidden].  For decode T=1, B=1 → x is [1, 1, hidden].
// We work in [B*T, hidden] = [S, hidden] throughout, then reshape back.

#[inline]
fn moe_switch_glu_input(x_flat: &InlineArray) -> InlineArray {
    debug_assert_eq!(x_flat.ndim(), 2);
    // MLX SwitchGLU does `mx.expand_dims(x, (-2, -3))` before gather_mm. Use
    // positive axes here so insertion order is unambiguous and yields the same
    // `[S, 1, 1, hidden]` layout for flattened `[S, hidden]` inputs.
    x_flat.expand_dims(1).expand_dims(2)
}

#[inline]
fn moe_routed_forward(lw: &LayerWeights, x_flat: &InlineArray) -> InlineArray {
    let s = x_flat.dim(0);
    let top_k = lw.moe_top_k;

    // ── Router ──────────────────────────────────────────────────────────────
    // gates: [S, num_experts]
    let gates = x_flat
        .matmul(lw.moe_router_w.as_ref().unwrap())
        .softmax_precise(-1);

    // Top-k selection: argpartition returns full permutation, take last top_k.
    // inds: [S, num_experts] → slice to [S, top_k]
    let all_inds = gates.argpartition(-top_k, -1);
    let num_experts_dim = gates.dim(1);
    let inds = all_inds.slice(&[0, num_experts_dim - top_k], &[s, num_experts_dim]);

    // Gather expert scores: [S, top_k]
    let mut scores = gates.take_along_axis(&inds, -1);
    if lw.moe_norm_topk_prob {
        let score_sum = scores.sum_axis(-1, true);
        scores = scores.divide(&score_sum);
    }

    // ── Expert dispatch via gather_mm / gather_qmm ─────────────────────────
    //
    // Mirror MLX SwitchGLU rank semantics exactly:
    //   x: [S, hidden] -> [S, 1, 1, hidden]
    //   up/gate gather_mm -> [S, top_k, 1, moe_intermediate]
    //   down gather_mm -> [S, top_k, 1, hidden]
    //   squeeze(-2) -> [S, top_k, hidden]
    //
    // Without these singleton axes, the down projection can reinterpret the
    // sequence axis as an additional batch dimension and produce
    // `[S, top_k, S, hidden]`, which then breaks score broadcasting.
    let switch_in = moe_switch_glu_input(x_flat);
    let x_gate_exp =
        lw.moe_gate_w
            .as_ref()
            .unwrap()
            .gather_mm_from(&switch_in, None, Some(&inds), false);
    let x_up_exp =
        lw.moe_up_w
            .as_ref()
            .unwrap()
            .gather_mm_from(&switch_in, None, Some(&inds), false);

    // Fused swiglu: silu(gate) * up
    let x_act = InlineArray::fused_swiglu(&x_gate_exp, &x_up_exp);

    // gather_mm for down projection: [S, top_k, 1, moe_intermediate] ×
    // [E, moe_intermediate, hidden] → [S, top_k, 1, hidden]
    let y_exp = lw
        .moe_down_w
        .as_ref()
        .unwrap()
        .gather_mm_from(&x_act, None, Some(&inds), false)
        .squeeze(-2);

    // Weighted sum over top_k: [S, top_k, hidden] * [S, top_k, 1] →
    // sum(-2) → [S, hidden]
    let scores_exp = scores.reshape(&[s, top_k, 1]);
    y_exp.multiply(&scores_exp).sum_axis(-2, false)
}

fn moe_forward(lw: &LayerWeights, x: &InlineArray) -> InlineArray {
    let b = x.dim(0);
    let t = x.dim(1);
    let h = x.dim(2);
    let s = b * t; // flattened sequence length

    if s == 1 {
        if let (
            Some(router_w),
            Some(LayerWeight::Dense(moe_gate_w)),
            Some(LayerWeight::Dense(moe_up_w)),
            Some(LayerWeight::Dense(moe_down_w)),
            Some(LayerWeight::Dense(shared_gate_w)),
            Some(LayerWeight::Dense(shared_up_w)),
            Some(LayerWeight::Dense(shared_down_w)),
            Some(shared_expert_gate_w),
        ) = (
            lw.moe_router_w.as_ref(),
            lw.moe_gate_w.as_ref(),
            lw.moe_up_w.as_ref(),
            lw.moe_down_w.as_ref(),
            lw.shared_gate_w.as_ref(),
            lw.shared_up_w.as_ref(),
            lw.shared_down_w.as_ref(),
            lw.shared_expert_gate_w.as_ref(),
        ) {
            return InlineArray::compiled_moe_layer_fixed(
                x,
                router_w,
                moe_gate_w,
                moe_up_w,
                moe_down_w,
                shared_gate_w,
                shared_up_w,
                shared_down_w,
                shared_expert_gate_w,
                lw.moe_top_k,
                lw.moe_norm_topk_prob,
            );
        }
    }

    // Flatten to [S, hidden].
    let x_flat = x.reshape(&[s, h]);
    let y_routed = moe_routed_forward(lw, &x_flat);

    // ── Shared expert ────────────────────────────────────────────────────────
    //
    // shared_expert(x): standard SwiGLU MLP with its own gate/up/down weights.
    // shared_expert_gate: Linear(hidden, 1) → sigmoid → scales shared output.
    let sh_gate = lw.shared_gate_w.as_ref().unwrap().matmul_from(&x_flat);
    let sh_up = lw.shared_up_w.as_ref().unwrap().matmul_from(&x_flat);
    let sh_act = InlineArray::fused_swiglu(&sh_gate, &sh_up);
    let sh_out = lw.shared_down_w.as_ref().unwrap().matmul_from(&sh_act);

    // shared_expert_gate: [S, 1] sigmoid gate
    let sh_scale = x_flat
        .matmul(lw.shared_expert_gate_w.as_ref().unwrap())
        .sigmoid();
    let y_shared = sh_out.multiply(&sh_scale);

    // ── Combine ──────────────────────────────────────────────────────────────
    y_routed.add(&y_shared).reshape(&[b, t, h])
}

// ============================================================================
// GDN layer forward
// ============================================================================

fn gdn_forward(
    lw: &LayerWeights,
    normed: &InlineArray,
    _b: i32,
    _s: i32,
    cache: &mut GdnCache,
    dtype: i32,
) -> InlineArray {
    let nv = lw.gdn_nv;
    let nk = lw.gdn_nk;
    let dk = lw.gdn_dk;
    let dv = lw.gdn_dv;
    let kd = lw.gdn_kd;
    let cd = lw.gdn_cd;
    let ck = lw.gdn_ck;
    let b = normed.dim(0);
    let s = normed.dim(1);

    // For decode-time T=1 on dense checkpoints, replay the fixed-shape compiled
    // GDN tape instead of rebuilding the full op graph every step.
    if s == 1 {
        if let (
            Some(LayerWeight::Dense(qkv_w)),
            Some(LayerWeight::Dense(z_w)),
            Some(LayerWeight::Dense(b_w)),
            Some(LayerWeight::Dense(a_w)),
            Some(LayerWeight::Dense(out_w)),
        ) = (
            &lw.gdn_qkv_w,
            &lw.gdn_z_w,
            &lw.gdn_b_w,
            &lw.gdn_a_w,
            &lw.gdn_out_w,
        ) {
            let conv_state = cache
                .conv_state
                .take()
                .unwrap_or_else(|| InlineArray::zeros(&[b, ck - 1, cd], dtype));
            let ssm_state = cache
                .ssm_state
                .take()
                .unwrap_or_else(|| InlineArray::zeros(&[b, nv, dv, dk], 10));

            let (output, new_conv, new_state) = InlineArray::compiled_gdn_layer_fixed(
                normed,
                qkv_w,
                z_w,
                b_w,
                a_w,
                lw.gdn_conv_w.as_ref().unwrap(),
                lw.gdn_q_nw.as_ref().unwrap(),
                lw.gdn_k_nw.as_ref().unwrap(),
                lw.gdn_a_log.as_ref().unwrap(),
                lw.gdn_dt_bias.as_ref().unwrap(),
                lw.gdn_norm_w.as_ref().unwrap(),
                out_w,
                &conv_state,
                &ssm_state,
                nv,
                nk,
                dk,
                dv,
                cd,
                ck,
                kd,
                lw.gdn_norm_eps,
            );

            cache.conv_state = Some(new_conv);
            cache.ssm_state = Some(new_state);
            return output;
        }
    }

    // Unified path for all T (T=1 decode and T>1 prefill).
    // Structure mirrors Python's gated_delta_update exactly:
    //   1. 4 separate matmul projections (qkv, z, b, a)
    //   2. Conv1d with fused silu activation
    //   3. split → q/k/v + rms_norm on q/k
    //   4. fused_compute_g (shapeless=True compiled — opaque Compiled node)
    //   5. gdn_metal_step (CustomKernel, outside any compile boundary)
    //   6. fused_precise_swiglu (shapeless=True compiled — opaque Compiled node)
    //   7. out_proj matmul
    let qkv = lw.gdn_qkv_w.as_ref().unwrap().matmul_from(normed);
    let z = lw
        .gdn_z_w
        .as_ref()
        .unwrap()
        .matmul_from(normed)
        .reshape(&[b, s, nv, dv]);
    let b_val = lw.gdn_b_w.as_ref().unwrap().matmul_from(normed);
    let a_val = lw.gdn_a_w.as_ref().unwrap().matmul_from(normed);

    // Conv state: concat previous state + new QKV, take new state, apply conv1d + silu
    let conv_state = cache
        .conv_state
        .take()
        .unwrap_or_else(|| InlineArray::zeros(&[b, ck - 1, cd], dtype));
    let conv_in = conv_state.concatenate_2(&qkv, 1);

    let new_conv = conv_in.slice(&[0, 1, 0], &[b, ck, cd]);
    let conv_out = conv_in
        .conv1d(lw.gdn_conv_w.as_ref().unwrap(), 1, 0, 1, cd)
        .fused_silu();

    // Split conv_out → q [B,S,nk,dk], k [B,S,nk,dk], v [B,S,nv,dv]
    let mut conv_parts = conv_out.split(&[kd, kd * 2], -1);
    let v = conv_parts.pop().unwrap().reshape(&[b, s, nv, dv]);
    let k = conv_parts.pop().unwrap().reshape(&[b, s, nk, dk]);
    let q = conv_parts.pop().unwrap().reshape(&[b, s, nk, dk]);

    // Q/K normalization
    let q = q.rms_norm(lw.gdn_q_nw.as_ref(), 1e-6);
    let k = k.rms_norm(lw.gdn_k_nw.as_ref(), 1e-6);

    // Decay gate: fused compute_g
    let g = InlineArray::fused_compute_g(
        lw.gdn_a_log.as_ref().unwrap(),
        &a_val,
        lw.gdn_dt_bias.as_ref().unwrap(),
    );
    let beta = b_val.sigmoid();

    // GDN Metal kernel recurrence
    let ssm_state = cache
        .ssm_state
        .take()
        .unwrap_or_else(|| InlineArray::zeros(&[b, nv, dv, dk], 10));
    let (out, new_state) = InlineArray::gdn_metal_step(&q, &k, &v, &g, &beta, &ssm_state, s);

    cache.conv_state = Some(new_conv);
    cache.ssm_state = Some(new_state);

    // Output: rms_norm → precise_swiglu → reshape → out_proj
    let out_n = out.rms_norm(lw.gdn_norm_w.as_ref(), lw.gdn_norm_eps);
    let gated = InlineArray::fused_precise_swiglu(&out_n, &z);
    let flat = gated.reshape(&[b, s, -1]);
    lw.gdn_out_w.as_ref().unwrap().matmul_from(&flat)
}

// ============================================================================
// Attention layer forward
// ============================================================================

#[allow(clippy::too_many_arguments)]
/// Optional inputs for tree-verify attention: per-token absolute
/// position ids and a custom additive attention mask.
///
/// Routes the per-token RoPE through MLX's fused `fast::rope` (the
/// SAME kernel linear DFlash uses) via a reshape trick: the input
/// `[B, H, T, D]` tensor is transposed to `[T, H, 1, D]` so the
/// `T`-token sequence becomes the rope's batch axis. MLX's array-
/// offset overload accepts one offset per batch element, so passing
/// `pos_ids` of shape `[T]` rotates each token at its own absolute
/// position with the EXACT same numerics linear DFlash gets. No
/// hand-rolled cos/sin → no bf16 ULP drift compounding across
/// 100+ decode rounds.
#[derive(Clone, Copy)]
pub(crate) struct TreeVerifyInputs<'a> {
    /// `[T]` int32 — per-token absolute positions (round offset +
    /// tree depth). Used as the array-offset for MLX rope.
    pub pos_ids: &'a InlineArray,
    /// `[1, 1, seq_len, past_length + seq_len]` additive mask in the
    /// current dtype. `0` at visible positions, `-inf` elsewhere.
    pub attention_mask: &'a InlineArray,
}

/// Apply per-token RoPE to a `[1, H, T, D]` tensor by routing through
/// MLX's fused `fast::rope` array-offset overload.
///
/// The trick: MLX's `rope(x, array offset)` accepts one offset per
/// BATCH element. We transpose `[1, H, T, D]` → `[T, H, 1, D]` so the
/// `T`-token sequence becomes the rope's batch axis with one token
/// per batch slot. The `pos_ids` array (shape `[T]`) then provides
/// the per-token absolute positions. After the rope call, transpose
/// back to `[1, H, T, D]`.
///
/// This guarantees numerical equivalence with linear DFlash's RoPE
/// path because both use the SAME MLX kernel — no hand-rolled
/// cos/sin, no separate ops with intermediate fp32 stores. Each
/// position rotation is computed in fused fp32 registers exactly as
/// MLX does for sequential positions.
///
/// `rotated_dims` must equal the head's rotary width (== head_dim
/// for Qwen3 full RoPE).
fn apply_per_position_rope(
    x: &InlineArray,
    pos_ids: &InlineArray,
    rotated_dims: i32,
    base: f32,
    scale: f32,
) -> InlineArray {
    // x shape: [B=1, H, T, D]. Transpose to [T, H, 1, D] so the rope
    // batch axis carries one token per row. The fast::rope kernel
    // sees a "batch" of T elements, each with H heads × 1 token × D
    // dim, and applies offset[batch_idx] + 0 = pos_ids[batch_idx] to
    // each one. Bit-exact with the linear path because both go
    // through the SAME MLX kernel.
    let xt = x.transpose_axes(&[2, 1, 0, 3]);
    let rotated = xt.rope_with_pos_ids(rotated_dims, false, base, scale, pos_ids);
    rotated.transpose_axes(&[2, 1, 0, 3])
}

fn attn_forward(
    lw: &LayerWeights,
    normed: &InlineArray,
    b: i32,
    s: i32,
    cache: &mut KvLayerCache,
    rope_offset: i32,
    dtype: i32,
    qjl_matrix: Option<&InlineArray>,
) -> InlineArray {
    attn_forward_with_tree_ctx(lw, normed, b, s, cache, rope_offset, dtype, qjl_matrix, None)
}

fn attn_forward_with_tree_ctx(
    lw: &LayerWeights,
    normed: &InlineArray,
    b: i32,
    s: i32,
    cache: &mut KvLayerCache,
    rope_offset: i32,
    dtype: i32,
    qjl_matrix: Option<&InlineArray>,
    tree_ctx: Option<TreeVerifyInputs>,
) -> InlineArray {
    let n_heads = lw.attn_n_heads;
    let n_kv_heads = lw.attn_n_kv_heads;
    let head_dim = lw.attn_head_dim;
    let scale = lw.attn_scale;
    let prev = cache.offset;
    let next = prev + s;
    if cache.keys.is_none() {
        let alloc = ((next + 255) / 256) * 256;
        cache.keys = Some(InlineArray::zeros(&[b, n_kv_heads, alloc, head_dim], dtype));
        cache.values = Some(InlineArray::zeros(&[b, n_kv_heads, alloc, head_dim], dtype));
    } else {
        let allocated = cache.keys.as_ref().unwrap().dim(2);
        if next > allocated {
            let grow_to = ((next + 255) / 256) * 256;
            let extend = grow_to - allocated;
            let old_k = cache.keys.take().unwrap();
            let old_v = cache.values.take().unwrap();
            let ext_k = InlineArray::zeros(&[b, n_kv_heads, extend, head_dim], dtype);
            let ext_v = InlineArray::zeros(&[b, n_kv_heads, extend, head_dim], dtype);
            cache.keys = Some(old_k.kv_cache_append(&ext_k, 2));
            cache.values = Some(old_v.kv_cache_append(&ext_v, 2));
        }
    }

    if s == 1 && cache.turboquant.is_none() && cache.quant_config.is_none() {
        if let (
            Some(LayerWeight::Dense(q_w)),
            Some(LayerWeight::Dense(k_w)),
            Some(LayerWeight::Dense(v_w)),
            Some(LayerWeight::Dense(o_w)),
        ) = (&lw.attn_q_w, &lw.attn_k_w, &lw.attn_v_w, &lw.attn_o_w)
        {
            let cache_keys = cache.keys.take().unwrap();
            let cache_vals = cache.values.take().unwrap();
            let (output, new_cache_keys, new_cache_vals) = InlineArray::compiled_attn_layer_fixed(
                normed,
                q_w,
                k_w,
                v_w,
                o_w,
                lw.attn_q_norm_w.as_ref().unwrap(),
                lw.attn_k_norm_w.as_ref().unwrap(),
                &cache_keys,
                &cache_vals,
                prev,
                rope_offset,
                n_heads,
                n_kv_heads,
                head_dim,
                scale,
                lw.attn_rope_dims,
                lw.attn_rope_base,
                lw.attn_rope_scale,
                lw.attn_q_norm_eps,
                lw.attn_k_norm_eps,
                lw.attn_gated,
            );
            cache.keys = Some(new_cache_keys);
            cache.values = Some(new_cache_vals);
            cache.offset = next;
            return output;
        }
    }

    // Q projection.
    //
    // Qwen3.5 (gated): q_proj output width = n_heads * head_dim * 2.
    //   Reshape to [B,S,H,D*2], split at D → [queries [B,S,H,D], gate [B,S,H,D]].
    //   gate is later reshaped to [B,S,H*D] and used to sigmoid-scale the output.
    //
    // Qwen3 (standard): q_proj output width = n_heads * head_dim.
    //   Reshape to [B,S,H,D] directly, no gate split.
    let q_proj_out = lw.attn_q_w.as_ref().unwrap().matmul_from(normed);

    let (queries, gate_opt) = if lw.attn_gated {
        // Gated path (Qwen3.5)
        let q_gate = q_proj_out.reshape(&[b, s, n_heads, head_dim * 2]);
        let mut qg_parts = q_gate.split(&[head_dim], -1);
        let gate = qg_parts.pop().unwrap().reshape(&[b, s, n_heads * head_dim]);
        let queries = qg_parts.pop().unwrap(); // [B, S, n_heads, head_dim]
        (queries, Some(gate))
    } else {
        // Standard path (Qwen3)
        let queries = q_proj_out.reshape(&[b, s, n_heads, head_dim]);
        (queries, None)
    };

    // K, V projections
    let new_keys = lw.attn_k_w.as_ref().unwrap().matmul_from(normed);
    let new_values = lw.attn_v_w.as_ref().unwrap().matmul_from(normed);

    // Q/K norms
    let queries = queries.rms_norm(lw.attn_q_norm_w.as_ref(), lw.attn_q_norm_eps);
    let keys = new_keys
        .reshape(&[b, s, n_kv_heads, head_dim])
        .rms_norm(lw.attn_k_norm_w.as_ref(), lw.attn_k_norm_eps);
    let values = new_values.reshape(&[b, s, n_kv_heads, head_dim]);

    // Transpose to [B, H, S, D]
    let queries = queries.transpose_axes(&[0, 2, 1, 3]);
    let keys = keys.transpose_axes(&[0, 2, 1, 3]);
    let values = values.transpose_axes(&[0, 2, 1, 3]);

    // RoPE: default path uses `fast::rope` with a scalar offset.
    // Tree-verify mode threads each token's absolute position
    // through MLX's array-offset overload via the [1,H,T,D]→[T,H,1,D]
    // transpose trick. Both paths use the SAME MLX kernel — no
    // hand-rolled rope, no bf16 ULP drift between the two modes, so
    // tree DFlash output stays bit-exact with linear DFlash
    // (and therefore with greedy decode at temperature=0).
    let (queries, keys) = if let Some(ctx) = tree_ctx {
        (
            apply_per_position_rope(
                &queries,
                ctx.pos_ids,
                lw.attn_rope_dims,
                lw.attn_rope_base,
                lw.attn_rope_scale,
            ),
            apply_per_position_rope(
                &keys,
                ctx.pos_ids,
                lw.attn_rope_dims,
                lw.attn_rope_base,
                lw.attn_rope_scale,
            ),
        )
    } else {
        (
            queries.rope(
                lw.attn_rope_dims,
                false,
                lw.attn_rope_base,
                lw.attn_rope_scale,
                rope_offset,
            ),
            keys.rope(
                lw.attn_rope_dims,
                false,
                lw.attn_rope_base,
                lw.attn_rope_scale,
                rope_offset,
            ),
        )
    };

    // KV cache update + SDPA
    let output = if let Some(ref mut tq_cache) = cache.turboquant {
        if s == 1 {
            match tq_cache.append_and_compute_attention(&queries, &keys, &values, scale) {
                Ok(output) => {
                    cache.offset = next;
                    output
                }
                Err(err) => {
                    trace_turboquant_qwen(&format!(
                        "decode_fallback=append_and_compute_attention_err seq={} prev={} err={}",
                        next, prev, err
                    ));
                    tq_cache.append(&keys, &values).ok();
                    let full_keys = tq_cache.dequantize_keys().unwrap_or_else(|| keys.clone());
                    let full_values = tq_cache
                        .dequantize_values()
                        .unwrap_or_else(|| values.clone());
                    cache.offset = next;
                    crate::decode::sdpa_causal_like_mlx(
                        &queries,
                        &full_keys,
                        &full_values,
                        scale,
                        s,
                    )
                }
            }
        } else {
            tq_cache.append(&keys, &values).ok();
            cache.offset = next;
            if prev == 0 {
                trace_turboquant_qwen(&format!("prefill_path=dense_prompt_only seq={}", next));
                crate::decode::sdpa_causal_like_mlx(&queries, &keys, &values, scale, s)
            } else {
                trace_turboquant_qwen(&format!(
                    "prefill_fallback=full_dequantized seq={} prev={}",
                    next, prev
                ));
                let full_keys = tq_cache.dequantize_keys().unwrap_or_else(|| keys.clone());
                let full_values = tq_cache
                    .dequantize_values()
                    .unwrap_or_else(|| values.clone());
                cache.offset = next;
                crate::decode::sdpa_causal_like_mlx(&queries, &full_keys, &full_values, scale, s)
            }
        }
    } else if let Some(qcfg) = cache.quant_config {
        let group_size = qcfg.group_size;

        if let Some(mb) = qcfg.mixed_bit {
            // ---- MIXED-BIT PATH (TurboQuant v2: Q2.5 / Q3.5) ----
            // After outlier permutation, the first `oc` dims of each head are
            // outliers (quantized at higher bits); the remaining `rc` are regular
            // (quantized at lower bits).
            let oc = mb.outlier_count; // outlier channel count per head
            let rc = head_dim - oc; // regular channel count per head
            let bits_hi = mb.outlier_bits as i32;
            let bits_lo = mb.regular_bits as i32;

            // MLX packed-uint32 dims for each half
            let packed_dim_hi = (oc * bits_hi + 31) / 32;
            let packed_dim_lo = (rc * bits_lo + 31) / 32;
            let scales_dim_hi = oc / group_size;
            let scales_dim_lo = rc / group_size;

            // Split K/V along the head-dim axis: [B, Hkv, S, oc] and [B, Hkv, S, rc]
            let k_hi = keys.slice(&[0, 0, 0, 0], &[b, n_kv_heads, s, oc]);
            let k_lo = keys.slice(&[0, 0, 0, oc], &[b, n_kv_heads, s, head_dim]);
            let v_hi = values.slice(&[0, 0, 0, 0], &[b, n_kv_heads, s, oc]);
            let v_lo = values.slice(&[0, 0, 0, oc], &[b, n_kv_heads, s, head_dim]);

            // Quantize each half → (packed, scales, biases)
            let (kp_hi, ks_hi, kb_hi) = {
                let flat = k_hi.reshape(&[b * n_kv_heads * s, oc]);
                let (p, s_, bi) = flat.quantize_weights(group_size, bits_hi);
                (
                    p.reshape(&[b, n_kv_heads, s, packed_dim_hi]),
                    s_.reshape(&[b, n_kv_heads, s, scales_dim_hi]),
                    bi.reshape(&[b, n_kv_heads, s, scales_dim_hi]),
                )
            };
            let (kp_lo, ks_lo, kb_lo) = {
                let flat = k_lo.reshape(&[b * n_kv_heads * s, rc]);
                let (p, s_, bi) = flat.quantize_weights(group_size, bits_lo);
                (
                    p.reshape(&[b, n_kv_heads, s, packed_dim_lo]),
                    s_.reshape(&[b, n_kv_heads, s, scales_dim_lo]),
                    bi.reshape(&[b, n_kv_heads, s, scales_dim_lo]),
                )
            };
            let (vp_hi, vs_hi, vb_hi) = {
                let flat = v_hi.reshape(&[b * n_kv_heads * s, oc]);
                let (p, s_, bi) = flat.quantize_weights(group_size, bits_hi);
                (
                    p.reshape(&[b, n_kv_heads, s, packed_dim_hi]),
                    s_.reshape(&[b, n_kv_heads, s, scales_dim_hi]),
                    bi.reshape(&[b, n_kv_heads, s, scales_dim_hi]),
                )
            };
            let (vp_lo, vs_lo, vb_lo) = {
                let flat = v_lo.reshape(&[b * n_kv_heads * s, rc]);
                let (p, s_, bi) = flat.quantize_weights(group_size, bits_lo);
                (
                    p.reshape(&[b, n_kv_heads, s, packed_dim_lo]),
                    s_.reshape(&[b, n_kv_heads, s, scales_dim_lo]),
                    bi.reshape(&[b, n_kv_heads, s, scales_dim_lo]),
                )
            };

            // ---- Cache management: allocate or grow 4 quantized buffers ----
            let uint32_dt = crate::compat::Dtype::Uint32.as_i32();
            if cache.quantized_keys_hi.is_none() {
                let alloc = ((next + 255) / 256) * 256;
                cache.quantized_keys_hi = Some(QuantizedTuple {
                    packed: InlineArray::zeros(&[b, n_kv_heads, alloc, packed_dim_hi], uint32_dt),
                    scales: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim_hi], dtype),
                    biases: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim_hi], dtype),
                });
                cache.quantized_keys = Some(QuantizedTuple {
                    packed: InlineArray::zeros(&[b, n_kv_heads, alloc, packed_dim_lo], uint32_dt),
                    scales: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim_lo], dtype),
                    biases: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim_lo], dtype),
                });
                cache.quantized_values_hi = Some(QuantizedTuple {
                    packed: InlineArray::zeros(&[b, n_kv_heads, alloc, packed_dim_hi], uint32_dt),
                    scales: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim_hi], dtype),
                    biases: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim_hi], dtype),
                });
                cache.quantized_values = Some(QuantizedTuple {
                    packed: InlineArray::zeros(&[b, n_kv_heads, alloc, packed_dim_lo], uint32_dt),
                    scales: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim_lo], dtype),
                    biases: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim_lo], dtype),
                });
            } else {
                let allocated = cache.quantized_keys_hi.as_ref().unwrap().packed.dim(2);
                if next > allocated {
                    let grow_to = ((next + 255) / 256) * 256;
                    let extend = grow_to - allocated;
                    let qkh = cache.quantized_keys_hi.take().unwrap();
                    let qkl = cache.quantized_keys.take().unwrap();
                    let qvh = cache.quantized_values_hi.take().unwrap();
                    let qvl = cache.quantized_values.take().unwrap();
                    cache.quantized_keys_hi = Some(QuantizedTuple {
                        packed: qkh.packed.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, packed_dim_hi], uint32_dt),
                            2,
                        ),
                        scales: qkh.scales.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim_hi], dtype),
                            2,
                        ),
                        biases: qkh.biases.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim_hi], dtype),
                            2,
                        ),
                    });
                    cache.quantized_keys = Some(QuantizedTuple {
                        packed: qkl.packed.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, packed_dim_lo], uint32_dt),
                            2,
                        ),
                        scales: qkl.scales.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim_lo], dtype),
                            2,
                        ),
                        biases: qkl.biases.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim_lo], dtype),
                            2,
                        ),
                    });
                    cache.quantized_values_hi = Some(QuantizedTuple {
                        packed: qvh.packed.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, packed_dim_hi], uint32_dt),
                            2,
                        ),
                        scales: qvh.scales.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim_hi], dtype),
                            2,
                        ),
                        biases: qvh.biases.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim_hi], dtype),
                            2,
                        ),
                    });
                    cache.quantized_values = Some(QuantizedTuple {
                        packed: qvl.packed.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, packed_dim_lo], uint32_dt),
                            2,
                        ),
                        scales: qvl.scales.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim_lo], dtype),
                            2,
                        ),
                        biases: qvl.biases.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim_lo], dtype),
                            2,
                        ),
                    });
                }
            }

            // slice_set new tokens into all four cache buffers
            let start_q = [0, 0, prev, 0];

            let qkh_ref = cache.quantized_keys_hi.as_mut().unwrap();
            qkh_ref.packed =
                qkh_ref
                    .packed
                    .slice_set(&kp_hi, &start_q, &[b, n_kv_heads, next, packed_dim_hi]);
            qkh_ref.scales =
                qkh_ref
                    .scales
                    .slice_set(&ks_hi, &start_q, &[b, n_kv_heads, next, scales_dim_hi]);
            qkh_ref.biases =
                qkh_ref
                    .biases
                    .slice_set(&kb_hi, &start_q, &[b, n_kv_heads, next, scales_dim_hi]);

            let qkl_ref = cache.quantized_keys.as_mut().unwrap();
            qkl_ref.packed =
                qkl_ref
                    .packed
                    .slice_set(&kp_lo, &start_q, &[b, n_kv_heads, next, packed_dim_lo]);
            qkl_ref.scales =
                qkl_ref
                    .scales
                    .slice_set(&ks_lo, &start_q, &[b, n_kv_heads, next, scales_dim_lo]);
            qkl_ref.biases =
                qkl_ref
                    .biases
                    .slice_set(&kb_lo, &start_q, &[b, n_kv_heads, next, scales_dim_lo]);

            let qvh_ref = cache.quantized_values_hi.as_mut().unwrap();
            qvh_ref.packed =
                qvh_ref
                    .packed
                    .slice_set(&vp_hi, &start_q, &[b, n_kv_heads, next, packed_dim_hi]);
            qvh_ref.scales =
                qvh_ref
                    .scales
                    .slice_set(&vs_hi, &start_q, &[b, n_kv_heads, next, scales_dim_hi]);
            qvh_ref.biases =
                qvh_ref
                    .biases
                    .slice_set(&vb_hi, &start_q, &[b, n_kv_heads, next, scales_dim_hi]);

            let qvl_ref = cache.quantized_values.as_mut().unwrap();
            qvl_ref.packed =
                qvl_ref
                    .packed
                    .slice_set(&vp_lo, &start_q, &[b, n_kv_heads, next, packed_dim_lo]);
            qvl_ref.scales =
                qvl_ref
                    .scales
                    .slice_set(&vs_lo, &start_q, &[b, n_kv_heads, next, scales_dim_lo]);
            qvl_ref.biases =
                qvl_ref
                    .biases
                    .slice_set(&vb_lo, &start_q, &[b, n_kv_heads, next, scales_dim_lo]);

            cache.offset = next;

            // Slice valid portions from all four cache buffers
            let qkh = cache.quantized_keys_hi.as_ref().unwrap();
            let cached_kp_hi = qkh
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim_hi]);
            let cached_ks_hi = qkh
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_hi]);
            let cached_kb_hi = qkh
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_hi]);

            let qkl = cache.quantized_keys.as_ref().unwrap();
            let cached_kp_lo = qkl
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim_lo]);
            let cached_ks_lo = qkl
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_lo]);
            let cached_kb_lo = qkl
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_lo]);

            let qvh = cache.quantized_values_hi.as_ref().unwrap();
            let cached_vp_hi = qvh
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim_hi]);
            let cached_vs_hi = qvh
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_hi]);
            let cached_vb_hi = qvh
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_hi]);

            let qvl = cache.quantized_values.as_ref().unwrap();
            let cached_vp_lo = qvl
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim_lo]);
            let cached_vs_lo = qvl
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_lo]);
            let cached_vb_lo = qvl
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_lo]);

            // Mixed-bit SDPA: two quantized_matmul calls per score/value aggregation
            crate::decode::quantized_sdpa_mixed(
                &queries,
                (&cached_kp_lo, &cached_ks_lo, &cached_kb_lo),
                (&cached_vp_lo, &cached_vs_lo, &cached_vb_lo),
                (&cached_kp_hi, &cached_ks_hi, &cached_kb_hi),
                (&cached_vp_hi, &cached_vs_hi, &cached_vb_hi),
                scale,
                s,
                n_heads,
                n_kv_heads,
                oc,
                group_size,
                bits_lo,
                bits_hi,
            )
        } else {
            // ---- UNIFORM-BIT PATH (unchanged) ----
            // Zero-overhead quantized KV cache path using quantized_matmul.
            // Matches mlx-lm's QuantizedKVCache: quantize K/V immediately after RoPE,
            // store as (packed_uint32, scales, biases), pass to quantized_matmul
            // which dequantizes inside the Metal kernel. No separate dequant pass.
            let bits = qcfg.bits as i32;

            // MLX packs quantized values into uint32: packed_dim = ceil(head_dim * bits / 32).
            // This is NOT head_dim / (32/bits) which fails for non-power-of-2 bit widths (Q3, Q5, Q6).
            let packed_dim = (head_dim * bits + 31) / 32;
            let scales_dim = head_dim / group_size;

            // Quantize new K/V → (packed, scales, biases)
            let keys_2d = keys.reshape(&[b * n_kv_heads * s, head_dim]);
            let (kp, ks, kb) = keys_2d.quantize_weights(group_size, bits);
            let kp = kp.reshape(&[b, n_kv_heads, s, packed_dim]);
            let ks = ks.reshape(&[b, n_kv_heads, s, scales_dim]);
            let kb = kb.reshape(&[b, n_kv_heads, s, scales_dim]);

            // QJL residual computation (keys only, Q2-Q3, uniform path).
            //
            // After quantizing keys, reconstruct the approximate key, compute the
            // residual (original - reconstructed), and store:
            //   qjl_signs      = sign(S · residual)  [B, Hkv, s, D] dtype ±1.0
            //   residual_norms = ||residual||₂        [B, Hkv, s, 1] f32
            //
            // These are later used in quantized_sdpa_with_qjl to add an unbiased
            // correction to attention scores: E[⟨q, k̃⟩] = ⟨q, k⟩.
            let qjl_active = qcfg.qjl && bits <= 3 && qjl_matrix.is_some();
            let (new_qjl_signs, new_qjl_norms) = if qjl_active {
                let s_mat = qjl_matrix.unwrap();
                // Dequantize to get the affine reconstruction.
                // kp/ks/kb are [B,Hkv,s,*] — reshape back to 2D for dequantize.
                let kp_flat = kp.reshape(&[b * n_kv_heads * s, packed_dim]);
                let ks_flat = ks.reshape(&[b * n_kv_heads * s, scales_dim]);
                let kb_flat = kb.reshape(&[b * n_kv_heads * s, scales_dim]);
                let k_recon_2d = kp_flat.dequantize(&ks_flat, &kb_flat, group_size, bits);
                // Residual: original keys (2D) minus affine reconstruction.
                let residual = keys_2d.subtract(&k_recon_2d); // [N, D]
                // Per-row L2 norm: [N, 1]
                let norms_2d = residual.square().sum_axis(-1, true).sqrt(); // [N, 1]
                // Project residual through S: [N, D] @ [D, D]^T = [N, D]
                // S is [D, D], so S^T = S.transpose_axes([1, 0])
                let s_t = s_mat.transpose_axes(&[1, 0]);
                let projected = residual.matmul(&s_t); // [N, D]
                // sign: positive → 1.0, negative → -1.0, zero → 0.0
                let signs_2d = projected.sign(); // [N, D] dtype (same as keys)
                // Reshape back to [B, Hkv, s, D] and [B, Hkv, s, 1]
                let signs = signs_2d.reshape(&[b, n_kv_heads, s, head_dim]);
                let norms = norms_2d.reshape(&[b, n_kv_heads, s, 1]);
                // Cast norms to f32 for numerical stability in correction.
                let norms_f32 = norms.as_dtype(crate::compat::Dtype::Float32.as_i32());
                (Some(signs), Some(norms_f32))
            } else {
                (None, None)
            };

            let values_2d = values.reshape(&[b * n_kv_heads * s, head_dim]);
            let (vp, vs, vb) = values_2d.quantize_weights(group_size, bits);
            let vp = vp.reshape(&[b, n_kv_heads, s, packed_dim]);
            let vs = vs.reshape(&[b, n_kv_heads, s, scales_dim]);
            let vb = vb.reshape(&[b, n_kv_heads, s, scales_dim]);

            // Cache management: allocate or grow quantized + QJL buffers
            let uint32_dt = crate::compat::Dtype::Uint32.as_i32();
            let f32_dt = crate::compat::Dtype::Float32.as_i32();
            if cache.quantized_keys.is_none() {
                let alloc = ((next + 255) / 256) * 256;
                cache.quantized_keys = Some(QuantizedTuple {
                    packed: InlineArray::zeros(&[b, n_kv_heads, alloc, packed_dim], uint32_dt),
                    scales: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim], dtype),
                    biases: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim], dtype),
                });
                cache.quantized_values = Some(QuantizedTuple {
                    packed: InlineArray::zeros(&[b, n_kv_heads, alloc, packed_dim], uint32_dt),
                    scales: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim], dtype),
                    biases: InlineArray::zeros(&[b, n_kv_heads, alloc, scales_dim], dtype),
                });
                if qjl_active {
                    cache.qjl_signs =
                        Some(InlineArray::zeros(&[b, n_kv_heads, alloc, head_dim], dtype));
                    cache.qjl_residual_norms =
                        Some(InlineArray::zeros(&[b, n_kv_heads, alloc, 1], f32_dt));
                }
            } else {
                let allocated = cache.quantized_keys.as_ref().unwrap().packed.dim(2);
                if next > allocated {
                    let grow_to = ((next + 255) / 256) * 256;
                    let extend = grow_to - allocated;
                    let qk = cache.quantized_keys.take().unwrap();
                    let qv = cache.quantized_values.take().unwrap();
                    cache.quantized_keys = Some(QuantizedTuple {
                        packed: qk.packed.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, packed_dim], uint32_dt),
                            2,
                        ),
                        scales: qk.scales.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim], dtype),
                            2,
                        ),
                        biases: qk.biases.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim], dtype),
                            2,
                        ),
                    });
                    cache.quantized_values = Some(QuantizedTuple {
                        packed: qv.packed.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, packed_dim], uint32_dt),
                            2,
                        ),
                        scales: qv.scales.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim], dtype),
                            2,
                        ),
                        biases: qv.biases.kv_cache_append(
                            &InlineArray::zeros(&[b, n_kv_heads, extend, scales_dim], dtype),
                            2,
                        ),
                    });
                    if qjl_active {
                        if let Some(qs) = cache.qjl_signs.take() {
                            cache.qjl_signs = Some(qs.kv_cache_append(
                                &InlineArray::zeros(&[b, n_kv_heads, extend, head_dim], dtype),
                                2,
                            ));
                        }
                        if let Some(qn) = cache.qjl_residual_norms.take() {
                            cache.qjl_residual_norms = Some(qn.kv_cache_append(
                                &InlineArray::zeros(&[b, n_kv_heads, extend, 1], f32_dt),
                                2,
                            ));
                        }
                    }
                }
            }

            // slice_set quantized data into cache
            let start_q = [0, 0, prev, 0];
            let qk_ref = cache.quantized_keys.as_mut().unwrap();
            let stop_kp = [b, n_kv_heads, next, packed_dim];
            let stop_ks = [b, n_kv_heads, next, scales_dim];
            qk_ref.packed = qk_ref.packed.slice_set(&kp, &start_q, &stop_kp);
            qk_ref.scales = qk_ref.scales.slice_set(&ks, &start_q, &stop_ks);
            qk_ref.biases = qk_ref.biases.slice_set(&kb, &start_q, &stop_ks);

            let qv_ref = cache.quantized_values.as_mut().unwrap();
            qv_ref.packed = qv_ref.packed.slice_set(&vp, &start_q, &stop_kp);
            qv_ref.scales = qv_ref.scales.slice_set(&vs, &start_q, &stop_ks);
            qv_ref.biases = qv_ref.biases.slice_set(&vb, &start_q, &stop_ks);

            // slice_set QJL data into cache
            if let (Some(signs), Some(norms)) = (new_qjl_signs, new_qjl_norms) {
                let stop_signs = [b, n_kv_heads, next, head_dim];
                let stop_norms = [b, n_kv_heads, next, 1];
                if let Some(ref mut qs) = cache.qjl_signs {
                    *qs = qs.slice_set(&signs, &start_q, &stop_signs);
                }
                if let Some(ref mut qn) = cache.qjl_residual_norms {
                    *qn = qn.slice_set(&norms, &start_q, &stop_norms);
                }
            }
            cache.offset = next;

            // Slice valid portion
            let qk = cache.quantized_keys.as_ref().unwrap();
            let cached_kp = qk
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim]);
            let cached_ks = qk
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim]);
            let cached_kb = qk
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim]);
            let qv = cache.quantized_values.as_ref().unwrap();
            let cached_vp = qv
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim]);
            let cached_vs = qv
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim]);
            let cached_vb = qv
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim]);

            // SDPA — with optional QJL correction when enabled
            if qjl_active {
                // Slice valid QJL data and project queries through S^T for correction.
                let cached_signs = cache
                    .qjl_signs
                    .as_ref()
                    .unwrap()
                    .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, head_dim]);
                let cached_norms = cache
                    .qjl_residual_norms
                    .as_ref()
                    .unwrap()
                    .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, 1]);
                // Project queries through S^T: [B, Hq, L, D] @ [D, D] = [B, Hq, L, D]
                let s_mat = qjl_matrix.unwrap();
                crate::decode::quantized_sdpa_with_qjl(
                    &queries,
                    (&cached_kp, &cached_ks, &cached_kb),
                    (&cached_vp, &cached_vs, &cached_vb),
                    &cached_signs,
                    &cached_norms,
                    s_mat,
                    scale,
                    s,
                    n_heads,
                    n_kv_heads,
                    group_size,
                    bits,
                )
            } else {
                // Quantized SDPA — zero overhead, dequant fused into Metal kernel
                crate::decode::quantized_sdpa(
                    &queries,
                    (&cached_kp, &cached_ks, &cached_kb),
                    (&cached_vp, &cached_vs, &cached_vb),
                    scale,
                    s,
                    n_heads,
                    n_kv_heads,
                    group_size,
                    bits,
                )
            }
        }
    } else {
        // Standard bf16 path
        let start = [0, 0, prev, 0];
        let stop = [b, n_kv_heads, next, head_dim];
        let k_buf = cache.keys.take().unwrap();
        let v_buf = cache.values.take().unwrap();
        cache.keys = Some(k_buf.slice_set(&keys, &start, &stop));
        cache.values = Some(v_buf.slice_set(&values, &start, &stop));
        cache.offset = next;

        let valid_keys = cache
            .keys
            .as_ref()
            .unwrap()
            .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, head_dim]);
        let valid_values = cache
            .values
            .as_ref()
            .unwrap()
            .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, head_dim]);
        if let Some(ctx) = tree_ctx {
            // Tree verify: use the caller-supplied additive mask that
            // encodes tree visibility instead of "causal" semantics.
            queries.sdpa_with_mask(&valid_keys, &valid_values, scale, Some(ctx.attention_mask))
        } else {
            crate::decode::sdpa_causal_like_mlx(&queries, &valid_keys, &valid_values, scale, s)
        }
    };

    // Output projection
    let output = output
        .transpose_axes(&[0, 2, 1, 3])
        .reshape(&[b, s, n_heads * head_dim]);

    let o_proj = lw.attn_o_w.as_ref().unwrap();
    if let Some(gate) = gate_opt {
        // Qwen3.5 gated output: o_proj(attn_out * sigmoid(gate))
        let gated = output.multiply(&gate.sigmoid());
        o_proj.matmul_from(&gated)
    } else {
        // Qwen3 standard output: o_proj(attn_out)
        o_proj.matmul_from(&output)
    }
}

// ============================================================================
// Sampling
// ============================================================================

/// Sample one token from `logits_2d` of shape `[B, vocab]`.
///
/// `temperature <= 0.0` → greedy argmax. Otherwise categorical sampling.
pub fn sample_token(logits_2d: &InlineArray, temperature: f32) -> InlineArray {
    crate::decode::sample_token(logits_2d, temperature)
}

/// Run prompt prefill and return the first sampled token.
pub fn prefill_first_token(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    input_ids: &[u32],
    temperature: f32,
) -> u32 {
    crate::decode::prefill_first_token(weights, cache, input_ids, temperature, forward_step)
}

// ============================================================================
// Generation loop
// ============================================================================

/// Run the full generation loop with async GPU pipelining.
///
/// `first_token` is the token at the end of the prompt (already prefilled into
/// `cache`). Each call to `on_token` receives the sampled token ID and returns
/// `false` to stop early (e.g. on EOS).
///
/// Returns all generated token IDs (not including `first_token`).
fn prepare_generation_cache(cache: &mut NativeCache, reserve_decode_inputs: i32, model_dtype: i32) {
    let trace_qwen35 = std::env::var_os("PMETAL_TRACE_QWEN35").is_some();
    if trace_qwen35 {
        eprintln!("[QWEN35 TRACE] begin_generation_session before_eval_and_detach");
    }
    cache.eval_and_detach_states();
    if trace_qwen35 {
        eprintln!("[QWEN35 TRACE] begin_generation_session after_eval_and_detach");
    }
    cache.reserve_decode_inputs(reserve_decode_inputs, model_dtype);
    if trace_qwen35 {
        eprintln!(
            "[QWEN35 TRACE] begin_generation_session after_reserve decode_inputs={reserve_decode_inputs}"
        );
    }
    if std::env::var_os("PMETAL_SKIP_CLEAR_CACHE").is_none() {
        bridge::clear_cache();
        if trace_qwen35 {
            eprintln!("[QWEN35 TRACE] begin_generation_session after_clear_cache");
        }
    } else if trace_qwen35 {
        eprintln!("[QWEN35 TRACE] begin_generation_session skipped_clear_cache");
    }
}

fn prime_generation_impl(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    reserve_decode_inputs: usize,
    temperature: f32,
    reset_peak_memory: bool,
    log_session: bool,
) -> InlineArray {
    let reserve_decode_inputs = reserve_decode_inputs.min(i32::MAX as usize) as i32;
    crate::decode::prime_generation(
        "NATIVE",
        weights.model_dtype,
        weights,
        cache,
        first_token,
        temperature,
        reset_peak_memory,
        log_session,
        |cache| prepare_generation_cache(cache, reserve_decode_inputs, weights.model_dtype),
        forward_step,
    )
}

fn generate_from_primed_sample_impl(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    current_y: InlineArray,
    max_tokens: usize,
    params: crate::decode::SamplingParams,
    log_stats: bool,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    crate::decode::generate_from_primed_sample_with_params(
        "NATIVE",
        weights,
        cache,
        current_y,
        max_tokens,
        params,
        log_stats,
        on_token,
        forward_step,
    )
}

/// Prime the canonical decode loop without resetting peak memory.
///
/// This is used by the MLX-LM parity benchmark so the timing path shares the
/// same bridge decode implementation as live inference.
pub fn prime_generation_preserve_peak(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    reserve_decode_inputs: usize,
    temperature: f32,
) -> InlineArray {
    prime_generation_impl(
        weights,
        cache,
        first_token,
        reserve_decode_inputs,
        temperature,
        false,
        true,
    )
}

pub fn prime_generation_preserve_peak_silent(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    reserve_decode_inputs: usize,
    temperature: f32,
) -> InlineArray {
    prime_generation_impl(
        weights,
        cache,
        first_token,
        reserve_decode_inputs,
        temperature,
        false,
        false,
    )
}

/// Continue generation from an already-primed async sample.
///
/// `current_y` must come from [`prime_generation_preserve_peak`] or the
/// equivalent internal priming path.
pub fn generate_from_primed_sample(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    current_y: InlineArray,
    max_tokens: usize,
    temperature: f32,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    generate_from_primed_sample_impl(
        weights,
        cache,
        current_y,
        max_tokens,
        crate::decode::SamplingParams::new(temperature),
        true,
        on_token,
    )
}

/// Run one MLX-LM-style benchmark trial on the canonical Qwen native path.
///
/// The timing split matches `mlx_lm.benchmark`: prompt timing includes prefill,
/// first-token sampling, and priming the next decode step; generation timing
/// covers only the remaining decode loop.
pub fn benchmark_mlx_lm_trial(
    weights: &NativeWeights,
    prompt_ids: &[u32],
    generation_tokens: usize,
    turboquant: Option<crate::turboquant::TurboQuantConfig>,
) -> crate::decode::BenchmarkTrial {
    crate::inline_array::reset_peak_memory();
    let mut cache = NativeCache::new_with_turboquant(weights, turboquant);

    let prompt_tic = std::time::Instant::now();
    let first_tok = prefill_first_token(weights, &mut cache, prompt_ids, 0.0);
    let current_y = prime_generation_preserve_peak_silent(
        weights,
        &mut cache,
        first_tok,
        generation_tokens.saturating_sub(1),
        0.0,
    );
    let prompt_secs = prompt_tic.elapsed().as_secs_f64();

    let generation_secs = if generation_tokens > 1 {
        let generation_tic = std::time::Instant::now();
        let generated_tail = generate_from_primed_sample_silent(
            weights,
            &mut cache,
            current_y,
            generation_tokens - 1,
            0.0,
            |_| true,
        );
        debug_assert_eq!(generated_tail.len(), generation_tokens - 1);
        generation_tic.elapsed().as_secs_f64()
    } else {
        crate::inline_array::synchronize();
        f64::MIN_POSITIVE
    };

    let trial = crate::decode::BenchmarkTrial {
        prompt_secs,
        generation_secs,
        peak_memory_bytes: crate::inline_array::get_peak_memory(),
    };

    crate::inline_array::synchronize();
    crate::inline_array::clear_cache();
    trial
}

pub fn generate_from_primed_sample_silent(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    current_y: InlineArray,
    max_tokens: usize,
    temperature: f32,
    on_token: impl FnMut(u32) -> bool,
) -> Vec<u32> {
    generate_from_primed_sample_impl(
        weights,
        cache,
        current_y,
        max_tokens,
        crate::decode::SamplingParams::new(temperature),
        false,
        on_token,
    )
    .0
}

pub fn generate(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    params: crate::decode::SamplingParams,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    let current_y = prime_generation_impl(
        weights,
        cache,
        first_token,
        max_tokens,
        params.temperature,
        true,
        true,
    );
    generate_from_primed_sample_impl(
        weights, cache, current_y, max_tokens, params, true, on_token,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenDecodeBackend {
    RustBridge,
}

pub fn canonical_decode_backend(
    _config: &Qwen3Config,
    _turboquant: Option<crate::turboquant::TurboQuantConfig>,
) -> QwenDecodeBackend {
    QwenDecodeBackend::RustBridge
}

#[allow(clippy::too_many_arguments)]
pub fn generate_canonical(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    config: &Qwen3Config,
    first_token: u32,
    max_tokens: usize,
    params: crate::decode::SamplingParams,
    turboquant: Option<crate::turboquant::TurboQuantConfig>,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    match canonical_decode_backend(config, turboquant) {
        QwenDecodeBackend::RustBridge => {
            generate(weights, cache, first_token, max_tokens, params, on_token)
        }
    }
}

pub fn benchmark_mlx_lm_trial_canonical(
    weights: &NativeWeights,
    config: &Qwen3Config,
    prompt_ids: &[u32],
    generation_tokens: usize,
    turboquant: Option<crate::turboquant::TurboQuantConfig>,
) -> crate::decode::BenchmarkTrial {
    match canonical_decode_backend(config, turboquant) {
        QwenDecodeBackend::RustBridge => {
            benchmark_mlx_lm_trial(weights, prompt_ids, generation_tokens, turboquant)
        }
    }
}

pub fn generate_preserve_peak(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    temperature: f32,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    let current_y = prime_generation_impl(
        weights,
        cache,
        first_token,
        max_tokens,
        temperature,
        false,
        true,
    );
    generate_from_primed_sample_impl(
        weights,
        cache,
        current_y,
        max_tokens,
        crate::decode::SamplingParams::new(temperature),
        true,
        on_token,
    )
}

// ============================================================================
// C++ monolithic per-token generation loop
// ============================================================================

fn begin_cpp_generation_session<'a>(
    weights: &'a NativeWeights,
    cache: &'a mut NativeCache,
    config: &Qwen3Config,
    first_token: u32,
    temperature: f32,
    reset_peak_memory: bool,
    log_session: bool,
) -> (CppDecodeSession<'a>, InlineArray) {
    if reset_peak_memory {
        crate::decode::begin_generation_session("NATIVE-CPP", weights.model_dtype);
    } else if log_session {
        crate::decode::begin_generation_session_preserve_peak("NATIVE-CPP", weights.model_dtype);
    } else {
        crate::decode::begin_generation_session_preserve_peak_silent(
            "NATIVE-CPP",
            weights.model_dtype,
        );
    }

    cache.eval_and_detach_states();
    bridge::clear_cache();

    let mut session = start_cpp_decode_session(weights, cache, config);
    let logits = session.step(first_token);
    let logits_2d = logits.squeeze(1);
    let current_y = sample_token(&logits_2d, temperature);
    current_y.async_eval_ref();
    (session, current_y)
}

fn generate_from_primed_cpp_session(
    mut session: CppDecodeSession<'_>,
    mut current_y: InlineArray,
    max_tokens: usize,
    temperature: f32,
    log_stats: bool,
    mut on_token: impl FnMut(u32) -> bool,
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut step_times: Vec<f64> = Vec::new();

    for step in 0..max_tokens {
        if step == 0 {
            current_y.eval();
        }
        let token_val = current_y.item_u32();

        tokens.push(token_val);
        if !on_token(token_val) {
            break;
        }
        if step + 1 >= max_tokens {
            break;
        }

        let t_step = std::time::Instant::now();
        let next_logits = session.step(token_val);
        let next_logits_2d = next_logits.squeeze(1);
        current_y = sample_token(&next_logits_2d, temperature);
        current_y.async_eval_ref();

        step_times.push(t_step.elapsed().as_secs_f64() * 1000.0);

        if step % 256 == 255 {
            bridge::clear_cache();
        }
    }

    if log_stats && step_times.len() > 20 {
        step_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let skip = 10;
        let avg = step_times[skip..].iter().sum::<f64>() / (step_times.len() - skip) as f64;
        let p50 = step_times[step_times.len() / 2];
        eprintln!(
            "[NATIVE-CPP] per-step: avg={avg:.2}ms p50={p50:.2}ms = {:.0} tok/s",
            1000.0 / avg
        );
    }

    drop(session);
    bridge::synchronize();
    tokens
}

/// Generation loop using the C++ monolithic per-token forward path.
///
/// Equivalent to [`generate`] but each decode step executes all per-layer ops
/// inside a single C++ function call (`mlx_inline_qwen35_decode_step`), which
/// removes per-op FFI overhead while still using the same bridge-native MLX
/// tensors and cache ownership as the Rust path.
#[allow(dead_code)]
fn generate_cpp(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    config: &Qwen3Config,
    first_token: u32,
    max_tokens: usize,
    temperature: f32,
    on_token: impl FnMut(u32) -> bool,
) -> Vec<u32> {
    if !supports_cpp_decode(config) {
        return generate(
            weights,
            cache,
            first_token,
            max_tokens,
            crate::decode::SamplingParams::new(temperature),
            on_token,
        )
        .0;
    }

    let (session, current_y) =
        begin_cpp_generation_session(weights, cache, config, first_token, temperature, true, true);
    generate_from_primed_cpp_session(session, current_y, max_tokens, temperature, true, on_token)
}

#[allow(dead_code)]
fn benchmark_mlx_lm_trial_cpp(
    weights: &NativeWeights,
    config: &Qwen3Config,
    prompt_ids: &[u32],
    generation_tokens: usize,
) -> crate::decode::BenchmarkTrial {
    crate::inline_array::reset_peak_memory();
    let mut cache = NativeCache::new_empty(weights);

    let prompt_tic = std::time::Instant::now();
    let first_tok = prefill_first_token(weights, &mut cache, prompt_ids, 0.0);
    let (session, current_y) =
        begin_cpp_generation_session(weights, &mut cache, config, first_tok, 0.0, false, false);
    let prompt_secs = prompt_tic.elapsed().as_secs_f64();

    let generation_secs = if generation_tokens > 1 {
        let generation_tic = std::time::Instant::now();
        let generated_tail = generate_from_primed_cpp_session(
            session,
            current_y,
            generation_tokens - 1,
            0.0,
            false,
            |_| true,
        );
        debug_assert_eq!(generated_tail.len(), generation_tokens - 1);
        generation_tic.elapsed().as_secs_f64()
    } else {
        crate::inline_array::synchronize();
        f64::MIN_POSITIVE
    };

    let trial = crate::decode::BenchmarkTrial {
        prompt_secs,
        generation_secs,
        peak_memory_bytes: crate::inline_array::get_peak_memory(),
    };

    crate::inline_array::synchronize();
    crate::inline_array::clear_cache();
    trial
}

#[allow(dead_code)]
fn supports_cpp_decode(config: &Qwen3Config) -> bool {
    let is_quantized = config.quantization_config.is_some();
    !config.is_qwen3_dense() && !is_quantized
}

fn sync_cpp_state_back(cache: &mut NativeCache, state: &CppForwardState) {
    cache.rope_offset = state.rope_offset;
    for (layer_cache, offset) in cache.kv_caches.iter_mut().zip(state.attn_kv_offsets.iter()) {
        layer_cache.offset = *offset;
    }
}

// ============================================================================
// C++ monolithic per-token path
// ============================================================================
//
// `CppForwardState` packages the flat weight pointer arrays, config int/float
// arrays, and the mutable cache pointer arrays required by
// `mlx_inline_qwen35_decode_step`. It is built once from `NativeWeights` +
// `NativeCache` and then passed to `forward_step_cpp_with_token` on every
// decode step.
//
// Layout matches the documentation in `bridge.h`:
//
//   weight_ptrs:  [embed_w, final_norm_w, lm_head_w, layer_0_block, ..., layer_N-1_block]
//                 where each layer block is QWEN35_WEIGHTS_PER_LAYER pointers.
//
//   cache_ptrs:   [gdn_0_conv, gdn_0_ssm, ..., gdn_{n_gdn-1}_ssm,
//                  attn_0_keys, attn_0_vals, ..., attn_{n_attn-1}_vals]
//                 n_attn cache slots = n_attn * 4 (keys + vals + 2 reserved/future slots)
//                 Actually: n_gdn*2 + n_attn*4 — but we only use keys+vals (2 slots each).
//                 NOTE: the bridge contract uses n_attn*4; slots +2 and +3 are zero-init
//                 sentinels included so that the cache pointer array is uniformly spaced.
//
// IMPORTANT: `CppForwardState` stores RAW POINTERS into `NativeWeights` and
// `NativeCache` arrays.  The caller MUST ensure both outlive the state.

const WEIGHTS_PER_LAYER: usize = 21;
#[allow(dead_code)]
pub struct CppForwardState {
    // Flat weight pointer array (const *const RawBuf).
    // All None slots (attention layers' GDN slots, etc.) are filled with a
    // dummy sentinel InlineArray that the C++ side never dereferences.
    weight_storage: Vec<InlineArray>, // owns sentinel arrays (indices where weight is absent)
    weight_ptrs: Vec<*const RawBuf>, // flat pointer array, length = 3 + num_layers * WEIGHTS_PER_LAYER

    // Flat cache pointer array (mutable, in/out).
    // n_gdn*2 slots for GDN + n_attn*4 slots for attn (keys, vals, sent, sent).
    cache_ptrs: Vec<*mut RawBuf>,

    // Scalar cache — updated by C++ in-place.
    pub attn_kv_offsets: Vec<i32>, // [n_attn]
    pub rope_offset: i32,

    // Config arrays
    config_ints: Vec<i32>,
    config_floats: Vec<f32>,

    // Counts for bounds checking / documentation
    n_gdn: usize,
    n_attn: usize,
    num_layers: usize,
}

// SAFETY: CppForwardState is only used from a single thread per generation
// step (the Rust caller holds &mut NativeCache).  Raw pointers into
// NativeWeights/NativeCache are stable because those structures never
// reallocate their InlineArray storage once constructed.
unsafe impl Send for CppForwardState {}
unsafe impl Sync for CppForwardState {}

pub struct CppDecodeSession<'a> {
    state: CppForwardState,
    cache: &'a mut NativeCache,
    _weights: std::marker::PhantomData<&'a NativeWeights>,
}

impl CppDecodeSession<'_> {
    pub fn step(&mut self, token_id: u32) -> InlineArray {
        // SAFETY: `start_cpp_decode_session` ties the session lifetime to the
        // borrowed weights/cache, so the raw pointers captured in `state`
        // remain valid for the whole session.
        unsafe { forward_step_cpp_with_token(&mut self.state, token_id) }
    }
}

impl Drop for CppDecodeSession<'_> {
    fn drop(&mut self) {
        sync_cpp_state_back(self.cache, &self.state);
    }
}

#[allow(dead_code)]
fn start_cpp_decode_session<'a>(
    weights: &'a NativeWeights,
    cache: &'a mut NativeCache,
    config: &Qwen3Config,
) -> CppDecodeSession<'a> {
    // SAFETY: the returned session borrows both `weights` and `cache` for its
    // lifetime, so neither can be moved or dropped while the raw-pointer state
    // is in use.
    let state = unsafe { build_cpp_forward_state(weights, cache, config) };
    CppDecodeSession {
        state,
        cache,
        _weights: std::marker::PhantomData,
    }
}

// Sentinel zero array used to fill unused weight slots.
fn sentinel() -> InlineArray {
    InlineArray::from_f32(0.0)
}

/// Build a `CppForwardState` from weights + cache.
///
/// This is called ONCE after `load_model` + `NativeCache::new_empty`. The
/// returned state holds raw pointers into `weights` and `cache`; both must
/// remain live and un-moved for the state's lifetime.
///
/// # Safety
/// `weights` and `cache` must not be moved or dropped while `CppForwardState`
/// is alive. In practice both live in the same generation loop scope.
pub unsafe fn build_cpp_forward_state(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    config: &Qwen3Config,
) -> CppForwardState {
    let num_layers = weights.layers.len();
    let n_gdn = cache.gdn_caches.len();
    let n_attn = cache.kv_caches.len();

    // Compute counts for the config arrays.
    let n_config_floats = 4 + num_layers * 2 + n_gdn + n_attn * 2;
    let n_config_ints = 20 + num_layers;

    // ── Build config_ints ──────────────────────────────────────────────────
    let gdn_nv = config.gdn_nv();
    let gdn_nk = config.gdn_nk();
    let gdn_dk = config.gdn_dk();
    let gdn_dv = config.gdn_dv();
    let ck = config.linear_conv_kernel_dim;
    let kd = gdn_nk * gdn_dk;
    let cd = kd * 2 + gdn_nv * gdn_dv;
    let n_heads = config.num_attention_heads;
    let n_kv = config.get_num_kv_heads();
    let head_dim = config.get_head_dim();
    let rope_dims = config.rope_dims();

    let mut config_ints = Vec::with_capacity(n_config_ints);
    config_ints.extend_from_slice(&[
        num_layers as i32,                               // [0]
        config.hidden_size,                              // [1]
        weights.model_dtype,                             // [2]
        n_gdn as i32,                                    // [3]
        n_attn as i32,                                   // [4]
        gdn_nv,                                          // [5]
        gdn_nk,                                          // [6]
        gdn_dk,                                          // [7]
        gdn_dv,                                          // [8]
        cd,                                              // [9]  gdn_cd
        ck,                                              // [10] gdn_ck
        kd,                                              // [11] gdn_kd
        n_heads,                                         // [12]
        n_kv,                                            // [13]
        head_dim,                                        // [14]
        rope_dims,                                       // [15]
        config.full_attention_interval,                  // [16]
        if weights.tie_word_embeddings { 1 } else { 0 }, // [17]
        config.num_experts_per_tok,                      // [18]
        if config.norm_topk_prob { 1 } else { 0 },       // [19]
    ]);
    for lw in &weights.layers {
        config_ints.push(if lw.is_moe_layer { 1 } else { 0 });
    }

    // ── Build config_floats ────────────────────────────────────────────────
    let attn_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut config_floats = Vec::with_capacity(n_config_floats);
    config_floats.push(weights.final_norm_eps); // [0]
    config_floats.push(attn_scale); // [1]
    config_floats.push(config.rope_theta as f32); // [2]
    config_floats.push(1.0_f32); // [3] rope_scale

    // Per-layer norm eps (input + post)
    for lw in &weights.layers {
        config_floats.push(lw.input_ln_eps);
        config_floats.push(lw.post_ln_eps);
    }
    // GDN norm eps
    for lw in &weights.layers {
        if lw.is_linear {
            config_floats.push(lw.gdn_norm_eps);
        }
    }
    // Attention Q/K norm eps
    for lw in &weights.layers {
        if !lw.is_linear {
            config_floats.push(lw.attn_q_norm_eps);
            config_floats.push(lw.attn_k_norm_eps);
        }
    }

    // ── Build weight_ptrs ──────────────────────────────────────────────────
    let total_weight_slots = 3 + num_layers * WEIGHTS_PER_LAYER;
    let mut weight_storage: Vec<InlineArray> = Vec::new();
    let mut weight_ptrs: Vec<*const RawBuf> = Vec::with_capacity(total_weight_slots);

    let push_real = |ptrs: &mut Vec<*const RawBuf>, w: &InlineArray| {
        ptrs.push(w.as_raw_ptr());
    };
    let push_sent = |ptrs: &mut Vec<*const RawBuf>, storage: &mut Vec<InlineArray>| {
        storage.push(sentinel());
        ptrs.push(storage.last().unwrap().as_raw_ptr());
    };

    // Global weights [0..3)
    push_real(&mut weight_ptrs, &weights.embed_w);
    push_real(&mut weight_ptrs, &weights.final_norm_w);
    if let Some(ref lm) = weights.lm_head_w {
        push_real(&mut weight_ptrs, lm.weight_arr());
    } else {
        push_sent(&mut weight_ptrs, &mut weight_storage);
    }

    // Per-layer weight blocks [3 + li*WEIGHTS_PER_LAYER .. 3 + (li+1)*WEIGHTS_PER_LAYER)
    // Slot layout (21 per layer):
    //   0: input_ln_w
    //   1: post_ln_w
    //   2: dense mlp_gate_w / moe_router_w
    //   3: dense mlp_up_w   / moe_gate_w
    //   4: dense mlp_down_w / moe_up_w
    //   5: attn_q_w   / gdn_qkv_w
    //   6: attn_k_w   / gdn_z_w
    //   7: attn_v_w   / gdn_b_w
    //   8: attn_o_w   / gdn_a_w
    //   9: attn_q_norm_w / gdn_conv_w
    //  10: attn_k_norm_w / gdn_q_nw
    //  11: gdn_k_nw
    //  12: gdn_a_log
    //  13: gdn_dt_bias
    //  14: gdn_norm_w
    //  15: gdn_out_w
    //  16: moe_down_w
    //  17: shared_gate_w
    //  18: shared_up_w
    //  19: shared_down_w
    //  20: shared_expert_gate_w
    for lw in &weights.layers {
        push_real(&mut weight_ptrs, &lw.input_ln_w);
        push_real(&mut weight_ptrs, &lw.post_ln_w);

        // MLP prefix slots. Dense layers expose gate/up/down; MoE layers expose
        // router/gate/up so the C++ path can execute either post-attention block.
        if lw.is_moe_layer {
            if let Some(w) = &lw.moe_router_w {
                push_real(&mut weight_ptrs, w);
            } else {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
            for opt in [&lw.moe_gate_w, &lw.moe_up_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w.weight_arr());
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
        } else {
            for opt in [&lw.mlp_gate_w, &lw.mlp_up_w, &lw.mlp_down_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w.weight_arr());
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
        }

        if lw.is_linear {
            // GDN slots — mixed types: LayerWeight for projections, InlineArray for small tensors.
            // Projections (LayerWeight):
            for opt in [&lw.gdn_qkv_w, &lw.gdn_z_w, &lw.gdn_b_w, &lw.gdn_a_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w.weight_arr());
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
            // Small tensors (InlineArray):
            for opt in [
                &lw.gdn_conv_w,
                &lw.gdn_q_nw,
                &lw.gdn_k_nw,
                &lw.gdn_a_log,
                &lw.gdn_dt_bias,
                &lw.gdn_norm_w,
            ] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w);
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
            // out_proj (LayerWeight):
            if let Some(w) = &lw.gdn_out_w {
                push_real(&mut weight_ptrs, w.weight_arr());
            } else {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
        } else {
            // Attention projection slots (LayerWeight):
            for opt in [&lw.attn_q_w, &lw.attn_k_w, &lw.attn_v_w, &lw.attn_o_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w.weight_arr());
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
            // Norm slots (InlineArray):
            for opt in [&lw.attn_q_norm_w, &lw.attn_k_norm_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w);
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
            // Attention layers do not use the GDN-only slots.
            for _ in 0..5 {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
        }

        if lw.is_moe_layer {
            if let Some(w) = &lw.moe_down_w {
                push_real(&mut weight_ptrs, w.weight_arr());
            } else {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
            for opt in [&lw.shared_gate_w, &lw.shared_up_w, &lw.shared_down_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w.weight_arr());
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
            if let Some(w) = &lw.shared_expert_gate_w {
                push_real(&mut weight_ptrs, w);
            } else {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
        } else {
            for _ in 0..5 {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
        }
    }

    // ── Build cache_ptrs ───────────────────────────────────────────────────
    let total_cache_slots = n_gdn * 2 + n_attn * 4;
    let mut cache_ptrs: Vec<*mut RawBuf> = Vec::with_capacity(total_cache_slots);

    for gc in &mut cache.gdn_caches {
        if let Some(ref mut s) = gc.conv_state {
            cache_ptrs.push(s.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
        if let Some(ref mut s) = gc.ssm_state {
            cache_ptrs.push(s.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
    }

    let attn_kv_offsets: Vec<i32> = cache.kv_caches.iter().map(|c| c.offset).collect();
    for kvc in &mut cache.kv_caches {
        if let Some(ref mut k) = kvc.keys {
            cache_ptrs.push(k.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
        if let Some(ref mut v) = kvc.values {
            cache_ptrs.push(v.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
        // 2 sentinel padding slots (bridge contract: 4 slots per attn layer)
        for _ in 0..2 {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
    }

    CppForwardState {
        weight_storage,
        weight_ptrs,
        cache_ptrs,
        attn_kv_offsets,
        rope_offset: cache.rope_offset,
        config_ints,
        config_floats,
        n_gdn,
        n_attn,
        num_layers,
    }
}

/// Run one forward step using the C++ monolithic per-token path.
///
/// This still builds the MLX graph for the full model on each call; it only
/// avoids Rust-side per-op FFI traffic by doing the work inside one C++ entry
/// point.
///
/// # Safety
///
/// The `state` must have been created by `build_cpp_forward_state` with valid
/// weight and cache pointers that outlive this call.
#[allow(dead_code)]
pub unsafe fn forward_step_cpp_with_token(
    state: &mut CppForwardState,
    token_id: u32,
) -> InlineArray {
    let token_ids = InlineArray::from_i32(token_id as i32).reshape(&[1, 1]);
    // SAFETY: caller guarantees weight/cache pointers are valid (upheld by build_cpp_forward_state).
    unsafe {
        bridge::qwen35_decode_step(
            &token_ids,
            &state.weight_ptrs,
            &mut state.cache_ptrs,
            &mut state.attn_kv_offsets,
            &mut state.rope_offset,
            &state.config_ints,
            &state.config_floats,
        )
    }
}

// ============================================================================
// Stack helper — concatenates arrays along a new axis
// ============================================================================
//
// MLX does not expose a standalone `stack` in the bridge; we implement it as
// expand_dims(axis=0) on each shard + successive concatenate_2 calls.
// For small E (e.g. 512 experts) this is done ONCE at load time and is not
// on the hot path.

fn stack_arrays(arrays: Vec<InlineArray>, axis: i32) -> Result<InlineArray, String> {
    if arrays.is_empty() {
        return Err("stack_arrays: empty input".to_string());
    }
    // Expand each shard: [out, in] → [1, out, in]
    let mut expanded: Vec<InlineArray> = arrays.into_iter().map(|a| a.expand_dims(axis)).collect();

    // Concatenate along the new axis: [1, out, in] × E → [E, out, in]
    let mut acc = expanded.remove(0);
    for e in expanded {
        acc = acc.concatenate_2(&e, axis);
    }
    Ok(acc)
}
