//! Qwen 3.5 (qwen3_next) hybrid architecture.
//!
//! Implements the Qwen3.5 architecture which combines:
//! - **Gated Delta Net (GDN)** linear attention layers (75% of layers)
//! - **Full attention** layers with gated output (every 4th layer)
//! - **Sparse MoE** with 512 experts + shared expert
//! - **(1+w) RMSNorm** (Gemma-style, requires f32 upcast)
//! - **Partial rotary** (25% of head dimensions)
//!
//! Ported from the MLX reference implementation (Apple, 2025).

use pmetal_runtime::compat::ops::{slice_axis, slice_last_from, slice_last_to};
use pmetal_runtime::compat::{
    Array, Dtype, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, Param,
    VisitLinears, nn, ops,
};
use pmetal_runtime::impl_module_params;
use pmetal_runtime::qwen3_native::family::gdn_qk_rms_norm_eps;
use pmetal_runtime::qwen3_native::mrope::MropeTables;
use pmetal_runtime::rope::RotaryEmbedding;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::expert_offload::OffloadedExperts;
use crate::fp8_utils::dequantize_fp8_weight_for_compute;
use crate::traits::ModelConfig;
use pmetal_cache::kv_cache::{KVCache, MambaCache, MambaCacheEntry};
use pmetal_runtime::native_weight::LayerWeight;
use pmetal_runtime::nn::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    gated_delta::gated_delta_with_gates,
    rope::{RopePositions, rope_embedding},
};

// ============================================================================
// Configuration
// ============================================================================

/// Qwen 3.5 (qwen3_next) model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3NextConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Hidden dimension.
    pub hidden_size: i32,
    /// Intermediate size for dense MLP layers.
    #[serde(default)]
    pub intermediate_size: i32,
    /// Number of hidden layers.
    pub num_hidden_layers: i32,
    /// Number of attention heads (for full attention layers).
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA in full attention).
    #[serde(default)]
    pub num_key_value_heads: Option<i32>,
    /// Head dimension.
    #[serde(default = "default_head_dim")]
    pub head_dim: Option<i32>,
    /// Maximum position embeddings.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// RMS norm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE theta base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // --- GDN linear attention params ---
    /// Number of value heads for GDN layers.
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: i32,
    /// Number of key heads for GDN layers.
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: i32,
    /// Key head dimension for GDN.
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: i32,
    /// Value head dimension for GDN.
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: i32,
    /// Conv1d kernel size for GDN.
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: i32,

    // --- Hybrid layer control ---
    /// Every Nth layer is a full attention layer (default 4).
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: i32,

    // --- MoE params ---
    /// Number of routed experts.
    #[serde(default)]
    pub num_experts: i32,
    /// Number of experts per token.
    #[serde(default)]
    pub num_experts_per_tok: i32,
    /// MoE layer frequency (every Nth layer).
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: i32,
    /// MoE intermediate size.
    #[serde(default)]
    pub moe_intermediate_size: i32,
    /// Shared expert intermediate size.
    #[serde(default)]
    pub shared_expert_intermediate_size: i32,
    /// Layers forced to use dense MLP instead of MoE.
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    /// Normalize top-k routing probabilities.
    #[serde(default)]
    pub norm_topk_prob: bool,

    // --- Attention ---
    /// Fraction of head dimensions to apply RoPE to (default 0.25).
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    /// Whether attention uses bias.
    #[serde(default)]
    pub attention_bias: bool,
    /// RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,

    /// Nested rope_parameters (HF format). Fields extracted during post-processing.
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,

    /// Explicit layer types (e.g., ["linear_attention", "full_attention", ...]).
    /// When present, overrides full_attention_interval-based layer type detection.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,

    /// Number of bundled MTP predictor layers, when present in the checkpoint.
    #[serde(default)]
    pub mtp_num_hidden_layers: Option<i32>,
    /// Alternate field name for bundled next-N predictors.
    #[serde(default)]
    pub num_nextn_predict_layers: Option<i32>,
    /// Whether the bundled MTP predictor ships its own embedding table
    /// (`mtp.embed_tokens.weight`) instead of sharing the model's.
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: Option<bool>,

    /// MLP / convolution activation. Only SiLU exists in this family.
    #[serde(default)]
    pub hidden_act: Option<String>,
    /// Activation of the gated-delta-net output gate (`"silu"`/`"swish"`
    /// or `"sigmoid"`); `hidden_act` when absent.
    #[serde(default)]
    pub output_gate_type: Option<String>,
    /// Whether full-attention layers gate their output. Always `true` in
    /// this family; `false` is refused.
    #[serde(default)]
    pub attn_output_gate: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3NextRoutedExpertMode {
    Resident,
    Placeholder,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Qwen3NextSanitizeOptions {
    pub skip_routed_experts: bool,
}

/// Nested RoPE parameters from HuggingFace config format.
///
/// The YaRN keys arrive resolved by
/// [`family::normalize_text_config`](pmetal_runtime::qwen3_native::family::normalize_text_config)
/// (defaults filled in, `attention_factor` computed); see
/// [`Qwen3NextConfig::rotary`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
    #[serde(default)]
    pub mrope_section: Option<Vec<i32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factor: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_max_position_embeddings: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta_fast: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta_slow: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_factor: Option<f64>,
}

fn default_model_type() -> String {
    "qwen3_next".to_string()
}
fn default_vocab_size() -> i32 {
    151936
}
fn default_head_dim() -> Option<i32> {
    Some(128)
}
fn default_max_position_embeddings() -> i32 {
    131072
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_linear_num_value_heads() -> i32 {
    8
}
fn default_linear_num_key_heads() -> i32 {
    4
}
fn default_linear_key_head_dim() -> i32 {
    128
}
fn default_linear_value_head_dim() -> i32 {
    128
}
fn default_linear_conv_kernel_dim() -> i32 {
    4
}
fn default_full_attention_interval() -> i32 {
    4
}
fn default_decoder_sparse_step() -> i32 {
    1
}
fn default_partial_rotary_factor() -> f32 {
    0.25
}

impl Qwen3NextConfig {
    /// Parse a checkpoint's `config.json`, flat or nested under `text_config`.
    ///
    /// The text config is read through
    /// [`family::normalize_text_config`](pmetal_runtime::qwen3_native::family::normalize_text_config),
    /// the reading the native engine parses too: transformers' defaults for
    /// absent keys, the outer `tie_word_embeddings` for a nested config,
    /// `rope_parameters` over the legacy RoPE keys, `layer_types` expanded,
    /// and every value this implementation cannot run refused by name.
    pub fn from_config_json(config_json: &str) -> Result<Self, Exception> {
        // json5, like the rest of the dispatcher: some exporters write
        // `Infinity`/`NaN`, which strict JSON rejects.
        let value: serde_json::Value = json5::from_str(config_json)
            .map_err(|e| Exception::custom(format!("config.json: {e}")))?;
        Self::from_config_value(&value)
    }

    /// [`from_config_json`](Self::from_config_json) on an already-parsed value.
    pub fn from_config_value(config: &serde_json::Value) -> Result<Self, Exception> {
        let text = pmetal_runtime::qwen3_native::family::normalize_text_config(config)
            .map_err(|e| Exception::custom(format!("Qwen3.5 config: {e}")))?;
        let mut parsed: Self = serde_json::from_value(text)
            .map_err(|e| Exception::custom(format!("Qwen3.5 config: {e}")))?;
        // Qwen3.5 MoE configs carry no dense `intermediate_size`; the dense
        // MLP of an `mlp_only_layers` layer is sized like the shared expert.
        if parsed.intermediate_size <= 0 {
            parsed.intermediate_size = parsed
                .shared_expert_intermediate_size
                .max(parsed.moe_intermediate_size);
        }
        Ok(parsed)
    }

    /// Activation of the gated-delta-net output gate.
    pub fn gdn_gate(&self) -> Result<GateActivation, Exception> {
        GateActivation::resolve(self.output_gate_type.as_deref(), self.hidden_act.as_deref())
            .map_err(Exception::custom)
    }

    /// Get head dimension.
    pub fn get_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Get KV heads count.
    pub fn get_num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Check if layer at index is a linear (GDN) layer.
    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        if let Some(ref layer_types) = self.layer_types {
            if layer_idx < layer_types.len() {
                return layer_types[layer_idx] == "linear_attention";
            }
        }
        ((layer_idx as i32) + 1) % self.full_attention_interval != 0
    }

    /// Check if layer uses MoE.
    pub fn use_moe_at(&self, layer_idx: usize) -> bool {
        let idx = layer_idx as i32;
        if self.mlp_only_layers.contains(&idx) {
            return false;
        }
        let sparse_step = self.decoder_sparse_step.max(1);
        self.num_experts > 0 && ((idx + 1) % sparse_step == 0)
    }

    /// RoPE dimensions for partial rotary.
    pub fn rope_dims(&self) -> i32 {
        (self.get_head_dim() as f32 * self.partial_rotary_factor) as i32
    }

    /// The rotary embedding, YaRN included, read by the same
    /// [`pmetal_runtime::rope`] parser the native engine uses.
    ///
    /// A YaRN `rope_parameters` comes resolved from
    /// [`from_config_value`](Self::from_config_value); a config built in code
    /// without its resolved keys gets transformers' defaults.
    pub fn rotary(&self) -> Result<pmetal_runtime::rope::Rotary, Exception> {
        let to_json = |v: Option<serde_json::Result<serde_json::Value>>| {
            v.transpose()
                .map_err(|e| Exception::custom(format!("rope config: {e}")))
        };
        let rope_scaling = to_json(self.rope_scaling.as_ref().map(serde_json::to_value))?;
        let rope_parameters = to_json(self.rope_parameters.as_ref().map(serde_json::to_value))?;
        pmetal_runtime::rope::Rotary::from_config(
            self.get_head_dim(),
            pmetal_runtime::rope::RopeConfig {
                rope_scaling: rope_scaling.as_ref(),
                rope_parameters: rope_parameters.as_ref(),
                rope_theta: Some(self.rope_theta as f64),
                partial_rotary_factor: Some(self.partial_rotary_factor as f64),
                max_position_embeddings: Some(self.max_position_embeddings as f64),
                original_max_position_embeddings: None,
            },
            10_000.0,
            0.25,
        )
        .map_err(|e| Exception::custom(format!("Qwen3.5 config: {e}")))
    }

    /// How many rotary frequencies each position axis (temporal, row,
    /// column) of a multimodal prompt drives; transformers' default when the
    /// config names none.
    pub fn mrope_section(&self) -> [usize; 3] {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.mrope_section.as_deref())
            .and_then(|s| <[i32; 3]>::try_from(s).ok())
            .map_or([11, 11, 10], |s| s.map(|v| v.max(0) as usize))
    }

    /// Number of bundled MTP predictor layers advertised by the config.
    pub fn mtp_num_hidden_layers(&self) -> usize {
        self.mtp_num_hidden_layers
            .or(self.num_nextn_predict_layers)
            .unwrap_or(0)
            .max(0) as usize
    }
}

impl ModelConfig for Qwen3NextConfig {
    fn model_type(&self) -> &str {
        &self.model_type
    }
    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }
    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }
    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }
    fn num_kv_heads(&self) -> i32 {
        self.get_num_kv_heads()
    }
    fn head_dim(&self) -> i32 {
        self.get_head_dim()
    }
    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }
    fn max_position_embeddings(&self) -> i32 {
        self.max_position_embeddings
    }
    fn norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }
    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
}

impl Default for Qwen3NextConfig {
    fn default() -> Self {
        Self {
            model_type: "qwen3_next".to_string(),
            vocab_size: 151936,
            hidden_size: 1536,
            intermediate_size: 4096,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: Some(4),
            head_dim: Some(128),
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            linear_num_value_heads: 8,
            linear_num_key_heads: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            num_experts: 512,
            num_experts_per_tok: 8,
            decoder_sparse_step: 1,
            moe_intermediate_size: 256,
            shared_expert_intermediate_size: 4096,
            mlp_only_layers: vec![],
            norm_topk_prob: false,
            partial_rotary_factor: 0.25,
            attention_bias: false,
            rope_scaling: None,
            rope_parameters: None,
            layer_types: None,
            mtp_num_hidden_layers: None,
            num_nextn_predict_layers: None,
            mtp_use_dedicated_embeddings: None,
            hidden_act: None,
            output_gate_type: None,
            attn_output_gate: None,
        }
    }
}

// ============================================================================
// Profiling
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3NextProfileSection {
    pub name: String,
    pub elapsed_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3NextLayerProfile {
    pub layer_idx: usize,
    pub layer_kind: String,
    pub sections: Vec<Qwen3NextProfileSection>,
    pub total_us: u64,
}

impl Qwen3NextLayerProfile {
    fn new(layer_idx: usize, is_linear: bool) -> Self {
        Self {
            layer_idx,
            layer_kind: if is_linear {
                "linear_attention".to_string()
            } else {
                "full_attention".to_string()
            },
            sections: Vec::new(),
            total_us: 0,
        }
    }

    fn push_section(&mut self, name: &str, start: Instant) {
        self.sections.push(Qwen3NextProfileSection {
            name: name.to_string(),
            elapsed_us: start.elapsed().as_micros() as u64,
        });
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3NextForwardProfile {
    pub phase: String,
    pub input_shape: Vec<i32>,
    pub embedding_us: u64,
    pub layers: Vec<Qwen3NextLayerProfile>,
    pub final_norm_us: u64,
    pub lm_head_us: u64,
    pub total_us: u64,
}

impl Qwen3NextForwardProfile {
    fn new(phase: impl Into<String>, input_shape: Vec<i32>) -> Self {
        Self {
            phase: phase.into(),
            input_shape,
            embedding_us: 0,
            layers: Vec::new(),
            final_norm_us: 0,
            lm_head_us: 0,
            total_us: 0,
        }
    }
}

/// What a section of a forward pass produced, evaluated so that its timing
/// covers the work and not just the building of the graph.
pub(crate) trait EvalAll {
    fn eval_all(&self);
}

impl EvalAll for Array {
    fn eval_all(&self) {
        self.eval();
    }
}

impl EvalAll for (Array, Array) {
    fn eval_all(&self) {
        self.0.eval();
        self.1.eval();
    }
}

impl EvalAll for (Array, Array, Array, Array) {
    fn eval_all(&self) {
        self.0.eval();
        self.1.eval();
        self.2.eval();
        self.3.eval();
    }
}

/// Whether a forward pass times its sections.
///
/// A layer is written once, as a series of named sections, and runs the same
/// code with or without a profile: [`Probe::off`] runs each section's closure
/// as it stands (the graph stays lazy), [`Probe::timing`] evaluates what it
/// returns and records the elapsed time under the section's name.
pub(crate) struct Probe<'a> {
    layer: Option<&'a mut Qwen3NextLayerProfile>,
}

impl<'a> Probe<'a> {
    pub(crate) fn off() -> Self {
        Self { layer: None }
    }

    pub(crate) fn timing(layer: &'a mut Qwen3NextLayerProfile) -> Self {
        Self { layer: Some(layer) }
    }

    /// The profile being filled, if any, for sections timed by hand.
    pub(crate) fn layer_mut(&mut self) -> Option<&mut Qwen3NextLayerProfile> {
        self.layer.as_deref_mut()
    }

    pub(crate) fn section<T: EvalAll>(
        &mut self,
        name: &str,
        build: impl FnOnce() -> Result<T, Exception>,
    ) -> Result<T, Exception> {
        let Some(layer) = self.layer.as_deref_mut() else {
            return build();
        };
        let start = Instant::now();
        let out = build()?;
        out.eval_all();
        layer.push_section(name, start);
        Ok(out)
    }
}

// ============================================================================
// Gated RMSNorm (with optional silu gate)
// ============================================================================

/// RMSNorm with optional gating: `rms_norm(x, w, eps) * act(gate)`.
///
/// Used by GDN linear attention layers (`linear_attn.norm`). Unlike the other
/// RMSNorm layers in the model, this norm does NOT use the (1+w) convention —
/// its weights are initialized at 1.0 and used directly. The (1+w) offset in
/// `sanitize_weights` intentionally excludes `.linear_attn.norm.weight`.
///
/// The gate activation is the config's `output_gate_type`: SiLU (`"swish"`)
/// for Qwen 3.5 / 3.6 / 3.8, sigmoid for Qwen4-Exp.
#[derive(Debug)]
pub struct Qwen3NextRMSNormGated {
    pub weight: Param<Array>,
    pub eps: f32,
    pub gate_activation: GateActivation,
}
impl_module_params!(Qwen3NextRMSNormGated; weight);

/// Activation applied to the gate of a [`Qwen3NextRMSNormGated`]: the one
/// type the native engine uses too, so the two read `output_gate_type` alike.
pub use pmetal_runtime::qwen3_native::family::GdnGateActivation as GateActivation;

impl Qwen3NextRMSNormGated {
    pub fn new(hidden_size: i32, eps: f32) -> Result<Self, Exception> {
        let weight = Array::ones_f32(&[hidden_size]);
        Ok(Self {
            weight: Param::new(weight),
            eps,
            gate_activation: GateActivation::Silu,
        })
    }

    /// `act(gate) * rms_norm(x)`, the product taken in float32 and returned in
    /// `x`'s dtype: the runtime's fused `_precise_swiglu` (compiled, one
    /// kernel for the casts, the activation and the product), which the native
    /// engine runs too.
    pub fn forward(&self, x: &Array, gate: Option<&Array>) -> Result<Array, Exception> {
        let normed = pmetal_runtime::compat::fast::rms_norm(x, self.weight.as_ref(), self.eps);
        let Some(gate) = gate else {
            return Ok(normed);
        };
        let gated = match self.gate_activation {
            GateActivation::Silu => Array::fused_precise_swiglu(&normed, gate),
            GateActivation::Sigmoid => Array::precise_sigmoid_gate(&normed, gate),
        };
        // The norm promotes when its weight is wider than the activations.
        Ok(if gated.dtype() != x.dtype() {
            gated.as_dtype(x.dtype().as_i32())
        } else {
            gated
        })
    }
}

// ============================================================================
// Qwen3Next MLP (SwiGLU)
// ============================================================================

#[derive(Debug)]
pub struct Qwen3NextMLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(Qwen3NextMLP; gate_proj, up_proj, down_proj);

impl Qwen3NextMLP {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Through each `Linear`'s own forward, so that an adapter or a packed
    /// weight on a projection is honoured.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let hidden = nn::silu(&self.gate_proj.forward(x)).multiply(&self.up_proj.forward(x));
        Ok(self.down_proj.forward(&hidden))
    }
}

// ============================================================================
// Qwen3Next Attention (Full attention with gated output + partial RoPE)
// ============================================================================

#[derive(Debug)]
pub struct Qwen3NextAttention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub effective_base: f32,
    pub rope_scale: f32,
    /// Rotary frequencies per position axis for a prompt with media.
    pub mrope_section: [usize; 3],
    /// The rotary embedding every position path rotates with.
    pub rope: RotaryEmbedding,
    /// Set when the rotary embedding is scaled (YaRN), which the inline
    /// decode path's `effective_base` / `rope_scale` cannot express.
    pub scaled_rope: Option<RotaryEmbedding>,
}
impl_module_params!(Qwen3NextAttention; q_proj, k_proj, v_proj, o_proj, q_norm, k_norm);

impl Qwen3NextAttention {
    pub fn new(config: &Qwen3NextConfig) -> Result<Self, Exception> {
        if config.attn_output_gate == Some(false) {
            return Err(Exception::custom(
                "attn_output_gate = false is not supported: Qwen3.5-family attention always \
                 gates its output",
            ));
        }
        let head_dim = config.get_head_dim();
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.get_num_kv_heads();

        let rope = RotaryEmbedding::new(config.rotary()?, false);
        let (effective_base, rope_scale) =
            rope.scalar().unwrap_or((rope.rotary().theta as f32, 1.0));
        let scaled_rope = rope.scalar().is_none().then(|| rope.clone());

        // q_proj outputs 2x for gated output: n_heads * head_dim * 2
        let q_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim * 2)
            .bias(config.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, config.hidden_size)
            .bias(config.attention_bias)
            .build()?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_dims: rope.dims(),
            effective_base,
            rope_scale,
            mrope_section: config.mrope_section(),
            rope,
            scaled_rope,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.run(x, mask, cache, positions, &mut Probe::off())
    }

    pub fn forward_profiled(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
        layer_profile: &mut Qwen3NextLayerProfile,
    ) -> Result<Array, Exception> {
        self.run(x, mask, cache, positions, &mut Probe::timing(layer_profile))
    }

    fn run(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        mut cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
        probe: &mut Probe<'_>,
    ) -> Result<Array, Exception> {
        let (b, l) = (x.dim(0), x.dim(1));
        let width = self.n_heads * self.head_dim;

        // The query projection holds the queries and the output gate, per head.
        let (queries, keys, values, gate) = probe.section("attn_prepare_qkv", || {
            let q_gate = self
                .q_proj
                .forward(x)
                .reshape(&[b, l, self.n_heads, self.head_dim * 2]);
            let queries = slice_last_to(&q_gate, self.head_dim);
            let gate = slice_last_from(&q_gate, self.head_dim).reshape(&[b, l, width]);
            let keys = self.k_proj.forward(x);
            let values = self.v_proj.forward(x);

            let queries = self.q_norm.forward(&queries);
            let keys = self
                .k_norm
                .forward(&keys.reshape(&[b, l, self.n_kv_heads, self.head_dim]));
            let values = values.reshape(&[b, l, self.n_kv_heads, self.head_dim]);

            // To [B, heads, L, head_dim].
            Ok((
                queries.transpose_axes(&[0, 2, 1, 3]),
                keys.transpose_axes(&[0, 2, 1, 3]),
                values.transpose_axes(&[0, 2, 1, 3]),
                gate,
            ))
        })?;

        let offset = cache
            .as_ref()
            .map_or(0, |(c, layer)| c.rope_offset_for(*layer));
        let (queries, keys) = probe.section("attn_rope", || {
            self.rotate(&queries, &keys, positions, offset)
        })?;

        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(Self::mask_type(mask));
        let attended = probe.section("attn_sdpa", || {
            // A compressed (TurboQuant) cache attends over its own storage.
            if mask.is_none()
                && let Some((kv, layer)) = cache.as_mut()
                && let Some(output) =
                    kv.try_compressed_attention(*layer, &queries, &keys, &values, &attn_config)?
            {
                return Ok(output);
            }
            let (keys, values) = match cache {
                Some((kv, layer)) => kv.update_and_fetch(layer, &keys, &values)?,
                None => (keys, values),
            };
            fused_sdpa(&queries, &keys, &values, &attn_config, mask)
        })?;

        // Gated output: o_proj(attention * sigmoid(gate)).
        probe.section("attn_out_proj", || {
            let attended = attended
                .transpose_axes(&[0, 2, 1, 3])
                .reshape(&[b, l, width]);
            Ok(self.o_proj.forward(&attended.multiply(&nn::sigmoid(&gate))))
        })
    }

    /// Rotate queries and keys `[B, heads, L, head_dim]`: one position per
    /// token (`positions` `[L]`, or the contiguous run from `offset`), or
    /// three, for a prompt carrying images or videos (`positions` `[3, L]`,
    /// interleaved mRoPE; see `pmetal_runtime::qwen3_native::mrope`).
    fn rotate(
        &self,
        queries: &Array,
        keys: &Array,
        positions: Option<&Array>,
        offset: i32,
    ) -> Result<(Array, Array), Exception> {
        if let Some(positions) = positions.filter(|p| p.ndim() == 2) {
            let tables = MropeTables::new(positions, &self.rope, self.mrope_section);
            return Ok((tables.apply(queries), tables.apply(keys)));
        }
        let positions = RopePositions::resolve(positions, offset);
        Ok((
            rope_embedding(queries, positions, &self.rope),
            rope_embedding(keys, positions, &self.rope),
        ))
    }

    /// Causal unless the caller passes a mask of its own.
    fn mask_type(mask: Option<&Array>) -> AttentionMaskType {
        if mask.is_some() {
            AttentionMaskType::None
        } else {
            AttentionMaskType::Causal
        }
    }
}

// ============================================================================
// Qwen3Next Gated Delta Net (GDN) linear attention
// ============================================================================

/// The four input projections (`qkv`, `z`, `b`, `a`) as one `[total, hidden]`
/// matrix, for a decode step, which multiplies by all four at once.
#[derive(Debug)]
struct FusedInputProjection {
    weight: Array,
    /// The ids of the four weights it was built from, so that a weight
    /// replaced since (a merge, a reload, an optimizer step) rebuilds it.
    built_from: [usize; 4],
}

#[derive(Debug)]
pub struct Qwen3NextGatedDeltaNet {
    pub conv1d: nn::Conv1d,
    pub in_proj_qkv: nn::Linear,
    pub in_proj_z: nn::Linear,
    pub in_proj_b: nn::Linear,
    pub in_proj_a: nn::Linear,
    pub norm: Qwen3NextRMSNormGated,
    pub out_proj: nn::Linear,
    pub dt_bias: Param<Array>,
    pub a_log: Param<Array>,

    pub hidden_size: i32,
    pub num_v_heads: i32,
    pub num_k_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub key_dim: i32,
    pub value_dim: i32,
    pub conv_dim: i32,
    pub conv_kernel_size: i32,
    fused_input_projection: Option<FusedInputProjection>,
    /// Pre-computed rms_norm weight for Q normalization: ones * inv_scale²
    pub q_norm_weight: Array,
    /// Pre-computed rms_norm weight for K normalization: ones * inv_scale
    pub k_norm_weight: Array,
    /// Epsilon of the q/k `rms_norm`, which stands in for an L2 norm.
    ///
    /// `rms_norm` adds it to the *mean* of squares, so `1e-6` (the default)
    /// is `head_k_dim · 1e-6` on the sum of squares. transformers' `l2norm`
    /// adds `1e-6` to the sum; `1e-6 / head_k_dim` here reproduces it exactly.
    pub qk_norm_eps: f32,
}
impl_module_params!(Qwen3NextGatedDeltaNet; conv1d, in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, norm, out_proj, dt_bias, a_log);

impl Qwen3NextGatedDeltaNet {
    /// Widest model (hidden size) whose decode step multiplies by the fused
    /// projection: past it the concatenated copy of the weights costs more
    /// memory than the one fewer dispatch is worth.
    const FUSED_PROJECTION_MAX_HIDDEN: i32 = 2048;

    pub fn new(config: &Qwen3NextConfig) -> Result<Self, Exception> {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = config.linear_conv_kernel_dim;

        let conv_dim = key_dim * 2 + value_dim;
        // Depthwise conv1d: in_channels = out_channels = conv_dim, groups = conv_dim.
        // Weight shape: [conv_dim, kernel, in_channels/groups] = [conv_dim, kernel, 1].
        // Using in_channels=1 would give in_channels/groups = 1/conv_dim = 0 (integer
        // division), producing a zero-size weight and crashing during eval.
        let conv1d = nn::Conv1dBuilder::new(conv_dim, conv_dim, conv_kernel_size)
            .bias(false)
            .groups(conv_dim)
            .padding(0)
            .build()?;

        // 4 separate projections matching the checkpoint's tensors.
        let in_proj_qkv = nn::LinearBuilder::new(hidden_size, key_dim * 2 + value_dim)
            .bias(false)
            .build()?;
        let in_proj_z = nn::LinearBuilder::new(hidden_size, value_dim)
            .bias(false)
            .build()?;
        let in_proj_b = nn::LinearBuilder::new(hidden_size, num_v_heads)
            .bias(false)
            .build()?;
        let in_proj_a = nn::LinearBuilder::new(hidden_size, num_v_heads)
            .bias(false)
            .build()?;

        let dt_bias = Param::new(Array::ones_f32(&[num_v_heads]));
        let a_log = Param::new(
            pmetal_runtime::compat::random::uniform_range(
                0.0,
                16.0,
                &[num_v_heads],
                pmetal_runtime::compat::Dtype::Float32,
            )
            .log(),
        );

        let mut norm = Qwen3NextRMSNormGated::new(head_v_dim, config.rms_norm_eps)?;
        norm.gate_activation = config.gdn_gate()?;

        let out_proj = nn::LinearBuilder::new(value_dim, hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            conv1d,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            norm,
            out_proj,
            dt_bias,
            a_log,
            hidden_size,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size,
            fused_input_projection: None,
            q_norm_weight: {
                let inv = (head_k_dim as f32).sqrt().recip();
                Array::ones_f32(&[head_k_dim]).multiply(&Array::from_f32(inv * inv))
            },
            k_norm_weight: {
                let inv = (head_k_dim as f32).sqrt().recip();
                Array::ones_f32(&[head_k_dim]).multiply(&Array::from_f32(inv))
            },
            // transformers' `l2norm(eps=1e-6)` exactly; see
            // `gdn_qk_rms_norm_eps` for why it is not `1e-6` itself.
            qk_norm_eps: gdn_qk_rms_norm_eps(head_k_dim),
        })
    }

    /// Whether every projection is a plain dense weight. The fused decode
    /// projection multiplies by the weights directly, which would skip an
    /// adapter and misread a packed weight.
    fn projections_are_plain(&self) -> bool {
        [
            &self.in_proj_qkv,
            &self.in_proj_z,
            &self.in_proj_b,
            &self.in_proj_a,
        ]
        .iter()
        .all(|linear| linear.plain_weight().is_some())
    }

    /// The fused projection: the four input projections multiplied at once,
    /// for the decode step of a small model with plain weights.
    fn uses_fused_projection(&self, inputs: &Array, mask: Option<&Array>, cached: bool) -> bool {
        cached
            && mask.is_none()
            && inputs.dim(1) == 1
            && self.hidden_size <= Self::FUSED_PROJECTION_MAX_HIDDEN
            && self.projections_are_plain()
    }

    /// The concatenated `[qkv; z; b; a]` weight, rebuilt when any of the four
    /// has been replaced. An FP8-stored weight is dequantized into it once.
    fn fused_input_weight(&mut self) -> Result<Array, Exception> {
        // `Array::id()` is stable for an array's lifetime and safe on lazy
        // weights; `data_ptr()` would dereference an unmaterialised buffer.
        let ids = [
            self.in_proj_qkv.weight.as_ref().id(),
            self.in_proj_z.weight.as_ref().id(),
            self.in_proj_b.weight.as_ref().id(),
            self.in_proj_a.weight.as_ref().id(),
        ];
        if self
            .fused_input_projection
            .as_ref()
            .is_none_or(|fused| fused.built_from != ids)
        {
            let weights = [
                dequantize_fp8_weight_for_compute(self.in_proj_qkv.weight.as_ref())?,
                dequantize_fp8_weight_for_compute(self.in_proj_z.weight.as_ref())?,
                dequantize_fp8_weight_for_compute(self.in_proj_b.weight.as_ref())?,
                dequantize_fp8_weight_for_compute(self.in_proj_a.weight.as_ref())?,
            ];
            let refs: Vec<&Array> = weights.iter().collect();
            let weight = ops::concatenate_axis(&refs, 0);
            weight.eval();
            self.fused_input_projection = Some(FusedInputProjection {
                weight,
                built_from: ids,
            });
        }
        Ok(self
            .fused_input_projection
            .as_ref()
            .map(|fused| fused.weight.clone())
            .expect("the fused projection was just built"))
    }

    /// `(qkv, z, b, a)` for `inputs` `[B, S, hidden]`: `z` as `[B, S, Hv, Dv]`,
    /// the rest as projected.
    fn project(
        &mut self,
        inputs: &Array,
        fused: bool,
    ) -> Result<(Array, Array, Array, Array), Exception> {
        let (b, s) = (inputs.dim(0), inputs.dim(1));
        let z_shape = [b, s, self.num_v_heads, self.head_v_dim];
        if !fused {
            return Ok((
                self.in_proj_qkv.forward(inputs),
                self.in_proj_z.forward(inputs).reshape(&z_shape),
                self.in_proj_b.forward(inputs),
                self.in_proj_a.forward(inputs),
            ));
        }
        let weight = self.fused_input_weight()?;
        let projected = ops::matmul(inputs, &weight.t());
        let z_start = self.conv_dim;
        let b_start = z_start + self.value_dim;
        let a_start = b_start + self.num_v_heads;
        Ok((
            slice_last_to(&projected, z_start),
            slice_axis(&projected, -1, z_start, b_start).reshape(&z_shape),
            slice_axis(&projected, -1, b_start, a_start),
            slice_last_from(&projected, a_start),
        ))
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        self.run(inputs, mask, cache, &mut Probe::off(), None)
    }

    /// GDN forward that ALSO records the per-token inputs the GDN
    /// recurrence saw, so a speculative-decoding rollback can replay the
    /// state over a partially-accepted prefix.
    pub fn forward_with_capture(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: Option<&mut MambaCacheEntry>,
        layer_idx: usize,
        capture: &mut pmetal_cache::speculative::SpecCapture,
    ) -> Result<Array, Exception> {
        self.run(
            inputs,
            mask,
            cache,
            &mut Probe::off(),
            Some((layer_idx, capture)),
        )
    }

    pub fn forward_profiled(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: Option<&mut MambaCacheEntry>,
        layer_profile: &mut Qwen3NextLayerProfile,
    ) -> Result<Array, Exception> {
        self.run(inputs, mask, cache, &mut Probe::timing(layer_profile), None)
    }

    /// The layer: projections, causal convolution over the cached window, the
    /// q/k norm, the gated delta rule, the gated norm and the output
    /// projection. One path for training (no cache, a gradient trace), prefill
    /// and decode (a cache), a verify (`capture`) and a profile (`probe`).
    fn run(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut MambaCacheEntry>,
        probe: &mut Probe<'_>,
        capture: Option<(usize, &mut pmetal_cache::speculative::SpecCapture)>,
    ) -> Result<Array, Exception> {
        let (b, s) = (inputs.dim(0), inputs.dim(1));
        let fused = self.uses_fused_projection(inputs, mask, cache.is_some());
        let (qkv, z, b_val, a) = probe.section("gdn_input_proj", || self.project(inputs, fused))?;

        // Padding contributes nothing to the convolution or the recurrence.
        let qkv = match mask {
            Some(mask) => {
                let mask = mask.reshape(&[mask.dim(0), mask.dim(1), 1]);
                ops::r#where(&mask, &qkv, &Array::from_f32(0.0))
            }
            None => qkv,
        };

        // The last `kernel - 1` inputs of everything before this call (zeros
        // for the start of a sequence) in front of this call's.
        let conv_input = match cache.as_deref_mut() {
            Some(entry) => entry.update_conv_state(&qkv, self.conv_kernel_size)?,
            None => {
                let history =
                    ops::zeros_dtype(&[b, self.conv_kernel_size - 1, self.conv_dim], qkv.dtype());
                ops::concatenate_axis(&[&history, &qkv], 1)
            }
        };
        let conv_out = probe.section("gdn_conv", || {
            Ok(nn::silu(&Module::forward(&mut self.conv1d, &conv_input)?))
        })?;

        // Split to q, k, v by head, and normalize q and k: `rms_norm` with
        // the pre-baked scales is transformers' L2 norm (and the 1/sqrt(Dk)
        // query scale).
        let parts = ops::split_sections(&conv_out, &[self.key_dim, self.key_dim * 2], -1);
        let q = parts[0].reshape(&[b, s, self.num_k_heads, self.head_k_dim]);
        let k = parts[1].reshape(&[b, s, self.num_k_heads, self.head_k_dim]);
        let v = parts[2].reshape(&[b, s, self.num_v_heads, self.head_v_dim]);
        let q = pmetal_runtime::compat::fast::rms_norm(&q, &self.q_norm_weight, self.qk_norm_eps);
        let k = pmetal_runtime::compat::fast::rms_norm(&k, &self.k_norm_weight, self.qk_norm_eps);

        // The recurrence's inputs, which a rollback replays from.
        let beta = b_val.sigmoid();
        let g = pmetal_runtime::nn::compute_g(self.a_log.as_ref(), &a, self.dt_bias.as_ref())?;
        if let Some((layer_idx, capture)) = capture {
            capture.record_gdn(
                layer_idx,
                pmetal_cache::kv_cache::GdnVerifyInputs {
                    keys: k.clone(),
                    values: v.clone(),
                    g: g.clone(),
                    beta: beta.clone(),
                    conv_input: qkv.clone(),
                    conv_kernel_size: self.conv_kernel_size as usize,
                },
            );
        }

        // The state is float32 whatever the activations' dtype: it sums every
        // token's update, and rounding it to bf16 each step would compound.
        let state = match cache.as_deref().and_then(|entry| entry.ssm_state.clone()) {
            Some(state) => state,
            None => ops::zeros_dtype(
                &[b, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                Dtype::Float32,
            ),
        };
        let (out, new_state) = probe.section("gdn_recurrence", || {
            // The fused kernel has no VJP, so inputs under a gradient trace
            // take the differentiable chunked or sequential path, and
            // everything else keeps the kernel.
            gated_delta_with_gates(&q, &k, &v, &g, &beta, &state, mask)
        })?;
        if let Some(entry) = cache {
            entry.ssm_state = Some(new_state);
        }

        probe.section("gdn_out_proj", || {
            let out = self.norm.forward(&out, Some(&z))?;
            Ok(self.out_proj.forward(&out.reshape(&[b, s, -1])))
        })
    }
}

// ============================================================================
// Sparse MoE Block
// ============================================================================

/// Switch `blocks` (each with its decoder-layer index) to experts packed by
/// `pmetal pack-experts` in `experts_dir`: opens the packed files, hands every
/// block the shared reader and frees its stacked expert arrays, since the
/// experts are read from the files as each layer routes.
///
/// Shared by every architecture built on [`Qwen3NextSparseMoeBlock`].
pub(crate) fn attach_expert_offload(
    experts_dir: &Path,
    blocks: &mut [(usize, &mut Qwen3NextSparseMoeBlock)],
) -> Result<(), Exception> {
    if blocks.is_empty() {
        return Err(Exception::custom("no MoE layers found to offload"));
    }
    let experts = Arc::new(
        OffloadedExperts::open(experts_dir)
            .map_err(|e| Exception::custom(format!("expert offload init: {e}")))?,
    );
    for (layer_idx, block) in blocks.iter_mut() {
        if !experts.layout().moe_layer_indices.contains(layer_idx) {
            return Err(Exception::custom(format!(
                "layer {layer_idx} is a sparse layer, but the packed experts in {} have none for it",
                experts_dir.display()
            )));
        }
        if experts.layout().num_experts != block.num_experts as usize {
            return Err(Exception::custom(format!(
                "the packed experts hold {} experts per layer, the model has {}",
                experts.layout().num_experts,
                block.num_experts
            )));
        }
        block.layer_idx = *layer_idx;
        block.offloaded = Some(experts.clone());
        // The experts come from the files from here on: free the stacks.
        *block.switch_mlp_gate_proj = Array::zeros_f32(&[1]);
        *block.switch_mlp_up_proj = Array::zeros_f32(&[1]);
        *block.switch_mlp_down_proj = Array::zeros_f32(&[1]);
        block.routed_experts_loaded = false;
    }

    tracing::info!(
        moe_layers = blocks.len(),
        expert_size_mb = experts.layout().expert_size as f64 / 1e6,
        "Expert offloading enabled"
    );
    Ok(())
}

#[derive(Debug)]
pub struct Qwen3NextSparseMoeBlock {
    pub gate: nn::Linear,
    pub switch_mlp_gate_proj: Param<Array>,
    pub switch_mlp_up_proj: Param<Array>,
    pub switch_mlp_down_proj: Param<Array>,
    pub shared_expert: Qwen3NextMLP,
    pub shared_expert_gate: nn::Linear,
    pub num_experts: i32,
    pub top_k: i32,
    pub norm_topk_prob: bool,
    /// Whether routed expert weights are currently resident in MLX arrays.
    pub routed_experts_loaded: bool,
    /// Which transformer layer this block belongs to (used for expert file lookup).
    pub layer_idx: usize,
    /// If set, routed expert weights are read from SSD as the layer routes
    /// instead of residing in memory. `None` means the resident (or packed)
    /// experts run.
    pub offloaded: Option<Arc<OffloadedExperts>>,
    /// Cached concatenated [gate; up; shared_gate] projection weights for the shared expert path.
    pub shared_combined_in_proj_weight: Option<Array>,
    /// Weight pointer signature used to invalidate the shared projection cache.
    pub shared_combined_in_proj_signature: Option<Vec<usize>>,
    /// Routed experts kept packed. When set, forward runs them on MLX's
    /// quantized gather kernels and the `switch_mlp_*` arrays are placeholders.
    pub packed_experts: Option<PackedRoutedExperts>,
}
impl_module_params!(Qwen3NextSparseMoeBlock; gate, switch_mlp_gate_proj, switch_mlp_up_proj, switch_mlp_down_proj, shared_expert, shared_expert_gate);

/// A MoE layer's routed experts on MLX's packed kernels, each projection
/// stacked `[E, out, in]` (an NVIDIA ModelOpt NVFP4 checkpoint keeps its
/// bytes this way, with one tensor scale per expert).
///
/// Not parameters: a packed weight has nothing to train, and leaving them out
/// of the tree keeps a parameter-wide pass (fp8 quantization, adapters) from
/// touching the packed bytes.
#[derive(Clone)]
pub struct PackedRoutedExperts {
    pub gate: LayerWeight,
    pub up: LayerWeight,
    pub down: LayerWeight,
}

impl std::fmt::Debug for PackedRoutedExperts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedRoutedExperts")
            .field("params", &self.gate.params())
            .finish_non_exhaustive()
    }
}

impl Qwen3NextSparseMoeBlock {
    pub fn new(config: &Qwen3NextConfig) -> Result<Self, Exception> {
        Self::new_with_routed_expert_mode(config, Qwen3NextRoutedExpertMode::Resident)
    }

    pub fn new_with_routed_expert_mode(
        config: &Qwen3NextConfig,
        routed_expert_mode: Qwen3NextRoutedExpertMode,
    ) -> Result<Self, Exception> {
        let dim = config.hidden_size;
        let intermediate_size = config.moe_intermediate_size;
        let num_experts = config.num_experts;

        let gate = nn::LinearBuilder::new(dim, num_experts)
            .bias(false)
            .build()?;

        // SwitchGLU stacked weights: [num_experts, intermediate_size, dim] etc.
        let routed_experts_loaded = routed_expert_mode == Qwen3NextRoutedExpertMode::Resident;
        let gate_proj = if routed_experts_loaded {
            Array::zeros_f32(&[num_experts, intermediate_size, dim])
        } else {
            Array::zeros_f32(&[1])
        };
        let up_proj = if routed_experts_loaded {
            Array::zeros_f32(&[num_experts, intermediate_size, dim])
        } else {
            Array::zeros_f32(&[1])
        };
        let down_proj = if routed_experts_loaded {
            Array::zeros_f32(&[num_experts, dim, intermediate_size])
        } else {
            Array::zeros_f32(&[1])
        };

        let shared_expert = Qwen3NextMLP::new(dim, config.shared_expert_intermediate_size)?;

        let shared_expert_gate = nn::LinearBuilder::new(dim, 1).bias(false).build()?;

        Ok(Self {
            gate,
            switch_mlp_gate_proj: Param::new(gate_proj),
            switch_mlp_up_proj: Param::new(up_proj),
            switch_mlp_down_proj: Param::new(down_proj),
            shared_expert,
            shared_expert_gate,
            num_experts,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            routed_experts_loaded,
            layer_idx: 0,
            offloaded: None,
            shared_combined_in_proj_weight: None,
            shared_combined_in_proj_signature: None,
            packed_experts: None,
        })
    }

    fn current_shared_input_proj_signature(&self) -> Vec<usize> {
        // Use `Array::id()` — same rationale as the GDN layer's fused projection.
        vec![
            self.shared_expert.gate_proj.weight.as_ref().id(),
            self.shared_expert.up_proj.weight.as_ref().id(),
            self.shared_expert_gate.weight.as_ref().id(),
        ]
    }

    fn ensure_shared_combined_input_proj_weight(&mut self) -> Result<Array, Exception> {
        let signature = self.current_shared_input_proj_signature();
        let needs_refresh = self.shared_combined_in_proj_weight.is_none()
            || self.shared_combined_in_proj_signature.as_ref() != Some(&signature);

        if needs_refresh {
            let weights = [
                dequantize_fp8_weight_for_compute(self.shared_expert.gate_proj.weight.as_ref())?,
                dequantize_fp8_weight_for_compute(self.shared_expert.up_proj.weight.as_ref())?,
                dequantize_fp8_weight_for_compute(self.shared_expert_gate.weight.as_ref())?,
            ];
            let weight_refs: Vec<&Array> = weights.iter().collect();
            let combined = ops::concatenate_axis(&weight_refs, 0);
            combined.eval();
            self.shared_combined_in_proj_weight = Some(combined);
            self.shared_combined_in_proj_signature = Some(signature);
        }

        Ok(self
            .shared_combined_in_proj_weight
            .as_ref()
            .unwrap()
            .clone())
    }

    fn forward_shared_expert_and_gate(&mut self, x: &Array) -> Result<(Array, Array), Exception> {
        let shape = x.shape();
        let batch_seq: i32 = shape[..shape.len() - 1].iter().product();
        let hidden = shape[shape.len() - 1];
        let x_flat = if shape.len() == 2 {
            x.clone()
        } else {
            x.reshape(&[batch_seq, hidden])
        };

        // The concatenated weights below are a fast path for plain dense
        // projections; an adapter or a packed weight goes through its layer.
        let plain = [
            &self.shared_expert.gate_proj,
            &self.shared_expert.up_proj,
            &self.shared_expert.down_proj,
            &self.shared_expert_gate,
        ]
        .iter()
        .all(|linear| linear.plain_weight().is_some());
        if !plain {
            let shared_y = self.shared_expert.forward(&x_flat)?;
            let shared_gate_logit = self
                .shared_expert_gate
                .forward(&x_flat)
                .reshape(&[batch_seq, 1]);
            return Ok((shared_y, shared_gate_logit));
        }

        let combined_weight = self.ensure_shared_combined_input_proj_weight()?;
        let projected = ops::matmul(&x_flat, &combined_weight.t());
        let intermediate = self.shared_expert.gate_proj.weight.as_ref().dim(0);

        let gate = slice_last_to(&projected, intermediate);
        let up = slice_axis(&projected, -1, intermediate, intermediate * 2);
        let shared_gate_logit =
            slice_last_from(&projected, intermediate * 2).reshape(&[batch_seq, 1]);
        let hidden = nn::silu(&gate).multiply(&up);
        let down_weight =
            dequantize_fp8_weight_for_compute(self.shared_expert.down_proj.weight.as_ref())?;
        let shared_y = ops::matmul(&hidden, &down_weight.t());

        Ok((shared_y, shared_gate_logit))
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        self.forward_timed(x, None)
    }

    /// [`forward`](Self::forward) with the sections of an offloaded layer
    /// timed when `probe` times.
    fn run(&mut self, x: &Array, probe: &mut Probe<'_>) -> Result<Array, Exception> {
        if self.offloaded.is_some() {
            return self.forward_timed(x, probe.layer_mut());
        }
        probe.section("moe_resident", || self.forward(x))
    }

    /// The router: softmax over the experts in float32, the top `k`, and with
    /// `norm_topk_prob` their renormalization; then the weights in the
    /// activations' dtype, which is where transformers casts them. The top-k
    /// helper is the positive form of `argpartition`: the sign-flipped one
    /// selects the `k` smallest experts (see
    /// `moe_routing::tests::sign_flipped_argpartition_is_anti_topk`).
    ///
    /// Returns `(indices [N, k] as u32, weights [N, k])`.
    fn route(&self, x_flat: &Array) -> Result<(Array, Array), Exception> {
        let logits = self.gate.forward(x_flat).cast(Dtype::Float32);
        let gates = ops::softmax_axis(&logits, -1);
        let (indices, weights) =
            crate::moe_routing::topk_normalize(&gates, self.top_k, self.norm_topk_prob)?;
        Ok((
            indices.cast(Dtype::Uint32),
            weights.as_dtype(x_flat.dtype().as_i32()),
        ))
    }

    /// The block: route, the routed experts (by `routed`, which is given the
    /// tokens `[N, hidden]` and their routing), and the shared expert behind
    /// its sigmoid gate. The decoder layer adds the residual.
    fn combine(
        &mut self,
        x: &Array,
        routed: impl FnOnce(&Array, &Array, &Array) -> Result<Array, Exception>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch_seq: i32 = shape[..shape.len() - 1].iter().product();
        let hidden = shape[shape.len() - 1];
        let x_flat = x.reshape(&[batch_seq, hidden]);

        let (indices, weights) = self.route(&x_flat)?;
        let routed = routed(&x_flat, &indices, &weights)?;

        let (shared_y, shared_gate_logit) = self.forward_shared_expert_and_gate(&x_flat)?;
        let shared = nn::sigmoid(&shared_gate_logit).multiply(&shared_y);
        Ok(routed.add(&shared).reshape(shape))
    }

    fn forward_timed(
        &mut self,
        x: &Array,
        layer_profile: Option<&mut Qwen3NextLayerProfile>,
    ) -> Result<Array, Exception> {
        let experts = if let Some(offloaded) = self.offloaded.clone() {
            let layer_idx = self.layer_idx;
            return self.combine(x, |x_flat, indices, weights| {
                let (routed, timings) = offloaded.routed(layer_idx, x_flat, indices, weights)?;
                if let Some(profile) = layer_profile {
                    profile.sections.push(Qwen3NextProfileSection {
                        name: "moe_expert_io".into(),
                        elapsed_us: timings.read_us,
                    });
                }
                Ok(routed)
            });
        } else if let Some(packed) = &self.packed_experts {
            packed.clone()
        } else if self.routed_experts_loaded {
            self.resident_experts()?
        } else {
            return Err(Exception::custom(
                "routed expert weights are not resident; enable expert offloading with --experts-dir",
            ));
        };
        self.combine(x, |x_flat, indices, weights| {
            Ok(pmetal_runtime::native_moe::switch_glu(
                x_flat,
                &experts.gate,
                &experts.up,
                &experts.down,
                indices,
                weights,
                |gate, up| nn::silu(gate).multiply(up),
            ))
        })
    }

    /// The resident stacks `[E, out, in]` as the dense weights the gather
    /// kernels take, `[E, in, out]` (views: the transpose copies nothing).
    fn resident_experts(&self) -> Result<PackedRoutedExperts, Exception> {
        let dense = |stack: &Array| -> Result<LayerWeight, Exception> {
            Ok(LayerWeight::Dense(
                dequantize_fp8_weight_for_compute(stack)?.swap_axes(-1, -2),
            ))
        };
        Ok(PackedRoutedExperts {
            gate: dense(self.switch_mlp_gate_proj.as_ref())?,
            up: dense(self.switch_mlp_up_proj.as_ref())?,
            down: dense(self.switch_mlp_down_proj.as_ref())?,
        })
    }
}

// ============================================================================
// Feed-forward enum (Dense MLP or MoE)
// ============================================================================

#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per layer, never moved; boxing would add an indirection to every forward"
)]
pub enum Qwen3NextFeedForward {
    Dense(Qwen3NextMLP),
    MoE(Qwen3NextSparseMoeBlock),
}

impl VisitLinears for Qwen3NextFeedForward {
    fn visit_linears_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut nn::Linear)) {
        match self {
            Self::Dense(mlp) => mlp.visit_linears_mut(prefix, f),
            Self::MoE(moe) => moe.visit_linears_mut(prefix, f),
        }
    }
}

impl ModuleParameters for Qwen3NextFeedForward {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(m) => m.num_parameters(),
            Self::MoE(m) => m.num_parameters(),
        }
    }
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(m) => m.parameters(),
            Self::MoE(m) => m.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Dense(m) => m.parameters_mut(),
            Self::MoE(m) => m.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(m) => m.trainable_parameters(),
            Self::MoE(m) => m.trainable_parameters(),
        }
    }
    fn freeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::Dense(m) => m.freeze_parameters(recurse),
            Self::MoE(m) => m.freeze_parameters(recurse),
        }
    }
    fn unfreeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::Dense(m) => m.unfreeze_parameters(recurse),
            Self::MoE(m) => m.unfreeze_parameters(recurse),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(m) => m.all_frozen(),
            Self::MoE(m) => m.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(m) => m.any_frozen(),
            Self::MoE(m) => m.any_frozen(),
        }
    }
}

impl Qwen3NextFeedForward {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        self.run(x, &mut Probe::off())
    }

    fn run(&mut self, x: &Array, probe: &mut Probe<'_>) -> Result<Array, Exception> {
        match self {
            Self::Dense(m) => probe.section("mlp_dense", || m.forward(x)),
            Self::MoE(m) => m.run(x, probe),
        }
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

/// Hybrid decoder layer: uses `linear_attn` (GDN) OR `self_attn` (full attention)
/// based on layer index. Option fields produce correct HF weight key names
/// (e.g. `model.layers.0.linear_attn.in_proj_qkv.weight`).
#[derive(Debug)]
pub struct Qwen3NextDecoderLayer {
    pub is_linear: bool,
    pub linear_attn: Option<Qwen3NextGatedDeltaNet>,
    pub self_attn: Option<Qwen3NextAttention>,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
    pub mlp: Qwen3NextFeedForward,
}
impl_module_params!(Qwen3NextDecoderLayer; linear_attn, self_attn, input_layernorm, post_attention_layernorm, mlp);

impl Qwen3NextDecoderLayer {
    pub fn new(config: &Qwen3NextConfig, layer_idx: usize) -> Result<Self, Exception> {
        Self::new_with_routed_expert_mode(config, layer_idx, Qwen3NextRoutedExpertMode::Resident)
    }

    pub fn new_with_routed_expert_mode(
        config: &Qwen3NextConfig,
        layer_idx: usize,
        routed_expert_mode: Qwen3NextRoutedExpertMode,
    ) -> Result<Self, Exception> {
        let is_linear = config.is_linear_layer(layer_idx);

        let linear_attn = if is_linear {
            Some(Qwen3NextGatedDeltaNet::new(config)?)
        } else {
            None
        };
        let self_attn = if !is_linear {
            Some(Qwen3NextAttention::new(config)?)
        } else {
            None
        };

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        let mlp = if config.use_moe_at(layer_idx) {
            Qwen3NextFeedForward::MoE(Qwen3NextSparseMoeBlock::new_with_routed_expert_mode(
                config,
                routed_expert_mode,
            )?)
        } else {
            Qwen3NextFeedForward::Dense(Qwen3NextMLP::new(
                config.hidden_size,
                config.intermediate_size,
            )?)
        };

        Ok(Self {
            is_linear,
            linear_attn,
            self_attn,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn new_mtp(
        config: &Qwen3NextConfig,
        layer_idx: usize,
        routed_expert_mode: Qwen3NextRoutedExpertMode,
    ) -> Result<Self, Exception> {
        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        let mlp = if config.use_moe_at(layer_idx) {
            Qwen3NextFeedForward::MoE(Qwen3NextSparseMoeBlock::new_with_routed_expert_mode(
                config,
                routed_expert_mode,
            )?)
        } else {
            Qwen3NextFeedForward::Dense(Qwen3NextMLP::new(
                config.hidden_size,
                config.intermediate_size,
            )?)
        };

        Ok(Self {
            is_linear: false,
            linear_attn: None,
            self_attn: Some(Qwen3NextAttention::new(config)?),
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        self.run(
            x,
            mask,
            kv_cache,
            positions,
            mamba_cache,
            LayerExtras::none(),
        )
    }

    /// Speculative-verify forward: records GDN verify inputs into
    /// `capture` for every linear-attention layer. Full-attention layers
    /// (every 4th, see `Qwen3NextConfig::is_linear_layer`) run the plain
    /// `forward` path — their KV cache rollback is just `KVCache::rollback`
    /// and needs no extra per-token state.
    #[expect(
        clippy::too_many_arguments,
        reason = "the layer forward's inputs plus the capture"
    )]
    pub fn forward_with_capture(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
        mamba_cache: Option<&mut MambaCacheEntry>,
        layer_idx: usize,
        capture: &mut pmetal_cache::speculative::SpecCapture,
    ) -> Result<Array, Exception> {
        self.run(
            x,
            mask,
            kv_cache,
            positions,
            mamba_cache,
            LayerExtras {
                capture: Some((layer_idx, capture)),
                probe: Probe::off(),
            },
        )
    }

    pub fn forward_profiled(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
        mamba_cache: Option<&mut MambaCacheEntry>,
        layer_idx: usize,
    ) -> Result<(Array, Qwen3NextLayerProfile), Exception> {
        let layer_start = Instant::now();
        let mut profile = Qwen3NextLayerProfile::new(layer_idx, self.is_linear);
        let out = self.run(
            x,
            mask,
            kv_cache,
            positions,
            mamba_cache,
            LayerExtras {
                capture: None,
                probe: Probe::timing(&mut profile),
            },
        )?;
        profile.total_us = layer_start.elapsed().as_micros() as u64;
        Ok((out, profile))
    }

    /// The layer: the token mixer (GDN or full attention) and the feed-forward
    /// block, each behind its norm and around a residual.
    fn run(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
        mamba_cache: Option<&mut MambaCacheEntry>,
        extras: LayerExtras<'_, '_>,
    ) -> Result<Array, Exception> {
        let LayerExtras { capture, mut probe } = extras;
        let normed = probe.section("input_layernorm", || Ok(self.input_layernorm.forward(x)))?;
        let mixed = if self.is_linear {
            self.linear_attn
                .as_mut()
                .expect("linear_attn must be Some for linear layers")
                .run(&normed, mask, mamba_cache, &mut probe, capture)?
        } else {
            self.self_attn
                .as_mut()
                .expect("self_attn must be Some for attention layers")
                .run(&normed, mask, kv_cache, positions, &mut probe)?
        };
        let h = probe.section("attn_residual", || Ok(x.add(&mixed)))?;
        let mlp_in = probe.section("post_attention_layernorm", || {
            Ok(self.post_attention_layernorm.forward(&h))
        })?;

        let mlp_start = Instant::now();
        let mlp_out = self.mlp.run(&mlp_in, &mut probe)?;
        if let Some(layer) = probe.layer_mut() {
            mlp_out.eval();
            layer.push_section("mlp", mlp_start);
        }
        probe.section("mlp_residual", || Ok(h.add(&mlp_out)))
    }
}

/// What a layer's forward leaves behind besides its output: the GDN inputs of
/// a speculative verify, and the timing of each section.
struct LayerExtras<'a, 'p> {
    capture: Option<(usize, &'a mut pmetal_cache::speculative::SpecCapture)>,
    probe: Probe<'p>,
}

impl LayerExtras<'_, '_> {
    fn none() -> Self {
        Self {
            capture: None,
            probe: Probe::off(),
        }
    }
}

// ============================================================================
// Model
// ============================================================================

#[derive(Debug)]
pub struct Qwen3NextModel {
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<Qwen3NextDecoderLayer>,
    pub norm: nn::RmsNorm,
    pub full_attention_interval: i32,
}
impl_module_params!(Qwen3NextModel; embed_tokens, layers, norm);

impl Qwen3NextModel {
    pub fn new(config: &Qwen3NextConfig) -> Result<Self, Exception> {
        Self::new_with_routed_expert_mode(config, Qwen3NextRoutedExpertMode::Resident)
    }

    pub fn new_with_routed_expert_mode(
        config: &Qwen3NextConfig,
        routed_expert_mode: Qwen3NextRoutedExpertMode,
    ) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| {
                Qwen3NextDecoderLayer::new_with_routed_expert_mode(config, i, routed_expert_mode)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            full_attention_interval: config.full_attention_interval,
        })
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None, None)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        self.forward_with_cache_and_capture(input_ids, mask, kv_cache, None, mamba_cache, None)
    }

    /// Forward pass with one rotary position per token, `[seq_len]`.
    ///
    /// Packed training concatenates several sequences into one row, so the
    /// positions have to restart at each boundary instead of running through
    /// it. Only the full-attention layers rotate; the GDN layers carry their
    /// order in the recurrence and take no positions.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_cache_and_capture(input_ids, mask, None, positions, None, None)
    }

    /// Forward pass with optional capture for speculative decoding.
    ///
    /// Behaves identically to `forward_with_cache` when `capture` is `None`.
    /// With a capture buffer, the post-layer hidden state is cloned into
    /// `capture.hidden_states` for every layer index the buffer asks for, and
    /// each GDN layer records the per-token inputs of its recurrence
    /// (`capture.gdn_inputs`), from which a rollback replays the layer's state
    /// over the accepted prefix of a verify block.
    pub fn forward_with_cache_and_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        positions: Option<&Array>,
        mamba_cache: Option<&mut MambaCache>,
        capture: Option<&mut pmetal_cache::speculative::SpecCapture>,
    ) -> Result<Array, Exception> {
        let hidden = Module::forward(&mut self.embed_tokens, input_ids)?;
        self.forward_embeddings_with_cache_and_capture(
            hidden,
            mask,
            kv_cache,
            positions,
            mamba_cache,
            capture,
        )
    }

    /// [`forward_with_cache_and_capture`](Self::forward_with_cache_and_capture)
    /// from input embeddings `[B, L, hidden]` instead of token ids: a prompt
    /// whose image and video tokens carry vision features. Such a prompt also
    /// has three positions per token, passed as `[3, L]` `positions`.
    pub fn forward_embeddings_with_cache_and_capture(
        &mut self,
        hidden: Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        positions: Option<&Array>,
        mamba_cache: Option<&mut MambaCache>,
        capture: Option<&mut pmetal_cache::speculative::SpecCapture>,
    ) -> Result<Array, Exception> {
        self.run_layers(
            hidden,
            mask,
            kv_cache,
            positions,
            mamba_cache,
            capture,
            None,
        )
    }

    pub fn forward_with_cache_profiled(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
        phase: impl Into<String>,
    ) -> Result<(Array, Qwen3NextForwardProfile), Exception> {
        let model_start = Instant::now();
        let mut profile = Qwen3NextForwardProfile::new(phase, input_ids.shape().to_vec());

        let embed_start = Instant::now();
        let hidden = Module::forward(&mut self.embed_tokens, input_ids)?;
        hidden.eval();
        profile.embedding_us = embed_start.elapsed().as_micros() as u64;

        let hidden = self.run_layers(
            hidden,
            mask,
            kv_cache,
            None,
            mamba_cache,
            None,
            Some(&mut profile),
        )?;
        profile.total_us = model_start.elapsed().as_micros() as u64;
        Ok((hidden, profile))
    }

    /// The decoder stack and the final norm, over embeddings `hidden`.
    #[allow(clippy::too_many_arguments)]
    fn run_layers(
        &mut self,
        mut hidden: Array,
        mask: Option<&Array>,
        mut kv_cache: Option<&mut KVCache>,
        positions: Option<&Array>,
        mut mamba_cache: Option<&mut MambaCache>,
        mut capture: Option<&mut pmetal_cache::speculative::SpecCapture>,
        mut profile: Option<&mut Qwen3NextForwardProfile>,
    ) -> Result<Array, Exception> {
        // The full-attention layers take the caller's mask (none for cached
        // decode, which is causal by construction). The GDN layers take none:
        // a `[1, 1, T, T]` attention mask would be read as a `[B, T]` validity
        // mask and zero the wrong positions, and there is no left padding in
        // this cache.
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let kv = if !layer.is_linear {
                kv_cache.as_deref_mut().map(|c| (c, layer_idx))
            } else {
                None
            };
            let mamba = if layer.is_linear {
                mamba_cache
                    .as_deref_mut()
                    .and_then(|c| c.get_mut(layer_idx))
            } else {
                None
            };
            let layer_mask = if layer.is_linear { None } else { mask };

            hidden = match profile.as_deref_mut() {
                Some(profile) => {
                    let (out, layer_profile) = layer
                        .forward_profiled(&hidden, layer_mask, kv, positions, mamba, layer_idx)?;
                    profile.layers.push(layer_profile);
                    out
                }
                None => match capture.as_deref_mut() {
                    // The GDN layers record their recurrence inputs; the
                    // attention layers' KV caches roll back by their offsets.
                    Some(buf) if layer.is_linear => layer.forward_with_capture(
                        &hidden, layer_mask, kv, positions, mamba, layer_idx, buf,
                    )?,
                    _ => layer.forward(&hidden, layer_mask, kv, positions, mamba)?,
                },
            };

            if let Some(buf) = capture.as_deref_mut()
                && buf.wants_hidden_for(layer_idx)
            {
                buf.record_hidden(layer_idx, hidden.clone());
            }
        }

        let norm_start = Instant::now();
        let hidden = self.norm.forward(&hidden);
        if let Some(profile) = profile {
            hidden.eval();
            profile.final_norm_us = norm_start.elapsed().as_micros() as u64;
        }
        Ok(hidden)
    }
}

// ============================================================================
// ForCausalLM
// ============================================================================

#[derive(Debug)]
pub struct Qwen3NextForCausalLM {
    pub model: Qwen3NextModel,
    pub lm_head: Option<nn::Linear>,
    pub config: Qwen3NextConfig,
}
impl_module_params!(Qwen3NextForCausalLM; model, lm_head);

impl Qwen3NextForCausalLM {
    pub fn new(config: Qwen3NextConfig) -> Result<Self, Exception> {
        Self::new_with_routed_expert_mode(config, Qwen3NextRoutedExpertMode::Resident)
    }

    pub fn new_with_routed_expert_mode(
        config: Qwen3NextConfig,
        routed_expert_mode: Qwen3NextRoutedExpertMode,
    ) -> Result<Self, Exception> {
        let model = Qwen3NextModel::new_with_routed_expert_mode(&config, routed_expert_mode)?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        };
        Ok(Self {
            model,
            lm_head,
            config,
        })
    }

    pub fn requires_expert_offloading(&self) -> bool {
        self.model.layers.iter().any(|layer| {
            matches!(
                &layer.mlp,
                Qwen3NextFeedForward::MoE(block)
                    if !block.routed_experts_loaded && block.offloaded.is_none()
            )
        })
    }

    fn lm_head_forward(&mut self, h: &Array) -> Result<Array, Exception> {
        if let Some(ref mut lm_head) = self.lm_head {
            lm_head.forward(h)
        } else {
            Ok(self.model.embed_tokens.as_linear(h))
        }
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let h = self.model.forward(input_ids, mask)?;
        self.lm_head_forward(&h)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let h = self
            .model
            .forward_with_cache(input_ids, mask, kv_cache, mamba_cache)?;
        self.lm_head_forward(&h)
    }

    /// Forward pass with one rotary position per token; see
    /// [`Qwen3NextModel::forward_with_positions`].
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let h = self
            .model
            .forward_with_positions(input_ids, mask, positions)?;
        self.lm_head_forward(&h)
    }

    /// Forward from input embeddings `[B, L, hidden]` at explicit positions,
    /// with optional caches: a prompt carrying images or videos, whose media
    /// tokens' embeddings are vision features and whose positions are `[3, L]`
    /// (see [`Qwen3NextModel::forward_embeddings_with_cache_and_capture`]).
    ///
    /// Returns the final normalized hidden states and the logits. Decoding
    /// after such a prompt passes each token's position explicitly
    /// ([`forward_with_cache_at`](Self::forward_with_cache_at)), since it
    /// continues from one past the prompt's largest position rather than from
    /// its length.
    pub fn forward_embeddings(
        &mut self,
        embeddings: &Array,
        positions: &Array,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<(Array, Array), Exception> {
        let h = self.model.forward_embeddings_with_cache_and_capture(
            embeddings.clone(),
            None,
            kv_cache,
            Some(positions),
            mamba_cache,
            None,
        )?;
        let logits = self.lm_head_forward(&h)?;
        Ok((h, logits))
    }

    /// Cached forward of token ids at explicit `[L]` positions: decoding after
    /// a prompt with media, where positions run from one past the prompt's
    /// largest position.
    pub fn forward_with_cache_at(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let h = self.model.forward_with_cache_and_capture(
            input_ids,
            None,
            kv_cache,
            Some(positions),
            mamba_cache,
            None,
        )?;
        self.lm_head_forward(&h)
    }

    /// Forward pass that returns logits AND pre-lm-head hidden states for
    /// the tapped layers — the target side of a DFlash speculative verify
    /// step for the hybrid-attention Qwen3.5 stack.
    ///
    /// Verify always runs with `T > 1` (multiple draft tokens), so this is the
    /// layer-by-layer forward with the GDN layers recording their rollback
    /// inputs.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
        capture: &mut pmetal_cache::speculative::SpecCapture,
    ) -> Result<Array, Exception> {
        let h = self.model.forward_with_cache_and_capture(
            input_ids,
            mask,
            kv_cache,
            None,
            mamba_cache,
            Some(capture),
        )?;
        self.lm_head_forward(&h)
    }

    /// Forward pass that returns both final normalized hidden states and logits.
    ///
    /// Qwen MTP consumes the target hidden state for committed tokens while the
    /// verifier still needs logits. Returning both from one trunk pass avoids
    /// rerunning the hybrid attention/GDN stack.
    pub fn forward_hidden_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
        capture: &mut pmetal_cache::speculative::SpecCapture,
    ) -> Result<(Array, Array), Exception> {
        let h = self.model.forward_with_cache_and_capture(
            input_ids,
            mask,
            kv_cache,
            None,
            mamba_cache,
            Some(capture),
        )?;
        let logits = self.lm_head_forward(&h)?;
        Ok((h, logits))
    }

    pub fn forward_with_cache_profiled(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
        phase: impl Into<String>,
    ) -> Result<(Array, Qwen3NextForwardProfile), Exception> {
        let total_start = Instant::now();
        let (h, mut profile) = self.model.forward_with_cache_profiled(
            input_ids,
            mask,
            kv_cache,
            mamba_cache,
            phase,
        )?;
        let lm_head_start = Instant::now();
        let logits = self.lm_head_forward(&h)?;
        logits.eval();
        profile.lm_head_us = lm_head_start.elapsed().as_micros() as u64;
        profile.total_us = total_start.elapsed().as_micros() as u64;
        Ok((logits, profile))
    }

    pub fn config(&self) -> &Qwen3NextConfig {
        &self.config
    }

    /// The reader of the offloaded experts, once [`enable_expert_offloading`]
    /// has run.
    ///
    /// [`enable_expert_offloading`]: Self::enable_expert_offloading
    pub fn offloaded_experts(&self) -> Option<&Arc<OffloadedExperts>> {
        self.model.layers.iter().find_map(|layer| match &layer.mlp {
            Qwen3NextFeedForward::MoE(block) => block.offloaded.as_ref(),
            Qwen3NextFeedForward::Dense(_) => None,
        })
    }

    /// Enable SSD-offloaded MoE inference: each sparse layer reads the experts
    /// its tokens route to from the packed files in `experts_dir` (written by
    /// `pmetal pack-experts`), and the stacked expert arrays are freed.
    pub fn enable_expert_offloading(&mut self, experts_dir: &Path) -> Result<(), Exception> {
        let mut blocks: Vec<(usize, &mut Qwen3NextSparseMoeBlock)> = self
            .model
            .layers
            .iter_mut()
            .enumerate()
            .filter_map(|(layer_idx, layer)| match &mut layer.mlp {
                Qwen3NextFeedForward::MoE(block) => Some((layer_idx, block)),
                Qwen3NextFeedForward::Dense(_) => None,
            })
            .collect();
        attach_expert_offload(experts_dir, &mut blocks)
    }
}

// ============================================================================
// Weight sanitization
// ============================================================================

/// Sanitize weights for Qwen3Next models.
///
/// Handles:
/// 1. Stripping HF prefix `model.language_model.` → `model.` (VLM wrapper format)
/// 2. Renaming `A_log` → `a_log` (Python uses `self.A_log`, Rust uses lowercase)
/// 3. Stacking per-expert weights into SwitchGLU format
/// 4. Conditional (1+w) RMSNorm offset (only when MTP or unsanitized conv detected)
/// 5. Transposing conv1d weights if needed
pub fn sanitize_weights(
    weights: &mut HashMap<String, Array>,
    config: &Qwen3NextConfig,
    options: Qwen3NextSanitizeOptions,
) -> Result<(), Exception> {
    // Detect shift condition BEFORE removing MTP (matching Python line 289-293)
    let has_mtp = weights.keys().any(|k| k.contains("mtp."));
    let has_unsanitized_conv = weights
        .iter()
        .any(|(k, v)| k.contains("conv1d.weight") && v.ndim() == 3 && v.dim(2) != 1);
    let should_shift_norms = has_mtp || has_unsanitized_conv;

    // Strip the vision-language wrapper prefixes and rename A_log → a_log,
    // the same canonicalization the native loader applies.
    let original_keys: Vec<String> = weights.keys().cloned().collect();
    for old_key in original_keys {
        let new_key = pmetal_runtime::qwen3_native::family::canonical_checkpoint_key(&old_key);
        if new_key != old_key {
            if let Some(v) = weights.remove(&old_key) {
                weights.insert(new_key, v);
            }
        }
    }

    let needs_fused_stacking = weights.contains_key("model.layers.0.mlp.experts.gate_up_proj");
    let needs_split_stacking = weights.contains_key("model.layers.0.mlp.experts.0.up_proj.weight");

    if needs_fused_stacking {
        for l in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{l}.mlp");
            let gate_up_key = format!("{prefix}.experts.gate_up_proj");
            let down_key = format!("{prefix}.experts.down_proj");

            if options.skip_routed_experts {
                weights.remove(&gate_up_key);
                weights.remove(&down_key);
                continue;
            }

            if let Some(gate_up) = weights.remove(&gate_up_key) {
                let inter = config.moe_intermediate_size;
                let gate = slice_axis(&gate_up, 1, 0, inter);
                let up = slice_axis(&gate_up, 1, inter, inter * 2);
                weights.insert(format!("{prefix}.switch_mlp_gate_proj"), gate);
                weights.insert(format!("{prefix}.switch_mlp_up_proj"), up);
            }
            if let Some(down) = weights.remove(&down_key) {
                weights.insert(format!("{prefix}.switch_mlp_down_proj"), down);
            }
        }
    } else if needs_split_stacking {
        for l in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{l}.mlp");
            for n in &["up_proj", "down_proj", "gate_proj"] {
                let mut expert_weights = Vec::new();
                for e in 0..config.num_experts {
                    let key = format!("{prefix}.experts.{e}.{n}.weight");
                    if let Some(w) = weights.remove(&key) {
                        if !options.skip_routed_experts {
                            expert_weights.push(w);
                        }
                    }
                }
                if !expert_weights.is_empty() {
                    let stacked = ops::stack_axis(&expert_weights, 0);
                    let dest = match *n {
                        "gate_proj" => format!("{prefix}.switch_mlp_gate_proj"),
                        "up_proj" => format!("{prefix}.switch_mlp_up_proj"),
                        "down_proj" => format!("{prefix}.switch_mlp_down_proj"),
                        _ => unreachable!(),
                    };
                    weights.insert(dest, stacked);
                }
            }
        }
    }

    // Remove MTP (multi-token prediction) weights
    let mtp_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.contains("mtp."))
        .cloned()
        .collect();
    for k in mtp_keys {
        weights.remove(&k);
    }

    // Apply conv1d transpose and conditional (1+w) norm offset
    // (1+w) RMSNorm offset applies to these norms only.
    // `.linear_attn.norm.weight` (Qwen3NextRMSNormGated) is intentionally absent —
    // the GDN norm uses standard weights (initialized at 1.0, no offset needed).
    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];

    let keys: Vec<String> = weights.keys().cloned().collect();
    for k in &keys {
        // Transpose conv1d weights if needed: [out, in, kernel] -> [out, kernel, in]
        if k.contains("conv1d.weight") {
            if let Some(v) = weights.get(k) {
                if v.ndim() == 3 && v.dim(2) != 1 {
                    let transposed = v.swap_axes(1, 2);
                    weights.insert(k.clone(), transposed);
                }
            }
        }

        // Add +1 to (1+w) norm weights — only when should_shift_norms
        if should_shift_norms && norm_suffixes.iter().any(|sfx| k.ends_with(sfx)) {
            if let Some(v) = weights.get(k) {
                if v.ndim() == 1 {
                    let offset = v.add(&Array::from_f32(1.0));
                    weights.insert(k.clone(), offset);
                }
            }
        }
    }

    // Remove lm_head.weight if tied
    if config.tie_word_embeddings {
        weights.remove("lm_head.weight");
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// The gated norm is `act(gate) * rms_norm(x)`, the product taken in float32
    /// and rounded once: for either activation, in bf16, at decode and prefill
    /// lengths.
    #[test]
    #[serial]
    fn the_gated_norm_is_its_formula_in_f32_rounded_once() {
        for activation in [GateActivation::Silu, GateActivation::Sigmoid] {
            for seq in [1, 7, 33] {
                let mut norm = Qwen3NextRMSNormGated::new(64, 1e-6).unwrap();
                norm.gate_activation = activation;
                let x = pmetal_runtime::compat::random::normal(&[1, seq, 64], Dtype::Bfloat16);
                let gate = pmetal_runtime::compat::random::normal(&[1, seq, 64], Dtype::Bfloat16);
                let got = norm.forward(&x, Some(&gate)).unwrap();
                assert_eq!(got.dtype(), Dtype::Bfloat16);

                let f32s = |a: &Array| a.as_dtype(Dtype::Float32.as_i32());
                let normed = f32s(&pmetal_runtime::compat::fast::rms_norm(
                    &x,
                    norm.weight.as_ref(),
                    norm.eps,
                ));
                let act = match activation {
                    GateActivation::Silu => nn::silu(&f32s(&gate)),
                    GateActivation::Sigmoid => nn::sigmoid(&f32s(&gate)),
                };
                let want = act.multiply(&normed).as_dtype(Dtype::Bfloat16.as_i32());
                let diff = f32s(&got).subtract(&f32s(&want)).abs().max(None).item_f32();
                assert!(
                    diff <= 1e-2,
                    "{activation:?} at {seq} tokens differs from its formula by {diff}"
                );
            }
        }
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    }

    fn tiny_config() -> Qwen3NextConfig {
        Qwen3NextConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: Some(16),
            vocab_size: 100,
            linear_num_value_heads: 2,
            linear_num_key_heads: 1,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 32,
            mlp_only_layers: vec![],
            norm_topk_prob: false,
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_config_layer_types() {
        let config = tiny_config();
        // With full_attention_interval=4, layers 0,1,2 are linear, layer 3 is full
        assert!(config.is_linear_layer(0), "Layer 0 should be linear");
        assert!(config.is_linear_layer(1), "Layer 1 should be linear");
        assert!(config.is_linear_layer(2), "Layer 2 should be linear");
        assert!(
            !config.is_linear_layer(3),
            "Layer 3 should be full attention"
        );
    }

    #[test]
    fn test_config_deserialization() {
        // Minimal config JSON similar to Qwen3.5-0.8B
        let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 1536,
            "intermediate_size": 4096,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 151936,
            "linear_num_value_heads": 8,
            "linear_num_key_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "num_experts": 512,
            "num_experts_per_tok": 8,
            "decoder_sparse_step": 1,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 4096,
            "mlp_only_layers": [0, 1, 2, 3],
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true
        }"#;

        let config: Qwen3NextConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.num_experts, 512);
        assert_eq!(config.full_attention_interval, 4);
        assert_eq!(config.partial_rotary_factor, 0.25);
        assert_eq!(config.mlp_only_layers, vec![0, 1, 2, 3]);
        assert!(config.is_linear_layer(0));
        assert!(!config.is_linear_layer(3));
    }

    #[test]
    fn test_model_config_trait() {
        let config = tiny_config();
        assert_eq!(config.model_type(), "qwen3_next");
        assert_eq!(config.vocab_size(), 100);
        assert_eq!(config.hidden_size(), 32);
        assert_eq!(config.num_hidden_layers(), 4);
        assert_eq!(ModelConfig::head_dim(&config), 16);
    }

    #[test]
    #[serial]
    fn test_mlp_forward_shape() {
        let mut mlp = Qwen3NextMLP::new(32, 64).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[1, 4, 32],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let output = mlp.forward(&x).unwrap();
        assert_eq!(output.shape(), &[1, 4, 32]);
    }

    #[test]
    #[serial]
    fn test_mlp_forward_matches_manual_swiglu() {
        let mut mlp = Qwen3NextMLP::new(32, 64).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[1, 4, 32],
            pmetal_runtime::compat::Dtype::Float32,
        );

        let gate = mlp.gate_proj.forward(&x).unwrap();
        let up = mlp.up_proj.forward(&x).unwrap();
        let expected = mlp
            .down_proj
            .forward(&nn::silu(&gate).unwrap().multiply(&up).unwrap())
            .unwrap();
        let actual = mlp.forward(&x).unwrap();
        let max_diff = actual
            .subtract(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(max_diff < 1e-5, "max diff too high: {max_diff}");
    }

    #[test]
    #[serial]
    fn test_gated_rms_norm() {
        let norm = Qwen3NextRMSNormGated::new(32, 1e-6).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[1, 4, 32],
            pmetal_runtime::compat::Dtype::Float32,
        );

        // Without gate
        let out1 = norm.forward(&x, None).unwrap();
        assert_eq!(out1.shape(), &[1, 4, 32]);

        // With gate
        let gate = pmetal_runtime::compat::random::normal(
            &[1, 4, 32],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let out2 = norm.forward(&x, Some(&gate)).unwrap();
        assert_eq!(out2.shape(), &[1, 4, 32]);
    }

    #[test]
    #[serial]
    fn test_attention_output_shape() {
        let config = tiny_config();
        let mut attn = Qwen3NextAttention::new(&config).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[1, 4, 32],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let output = attn.forward(&x, None, None, None).unwrap();
        assert_eq!(output.shape(), &[1, 4, 32]);
    }

    #[test]
    fn test_attention_mask_type_selection() {
        assert_eq!(
            Qwen3NextAttention::mask_type(None),
            AttentionMaskType::Causal
        );
        let custom_mask =
            pmetal_runtime::compat::ops::ones(&[1, 1, 1, 1], pmetal_runtime::compat::Dtype::Bool);
        assert_eq!(
            Qwen3NextAttention::mask_type(Some(&custom_mask)),
            AttentionMaskType::None
        );
    }

    #[test]
    #[serial]
    fn test_gdn_output_shape() {
        let config = tiny_config();
        let mut gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[1, 4, 32],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let output = gdn.forward(&x, None, None).unwrap();
        assert_eq!(output.shape(), &[1, 4, 32]);
    }

    /// ⚠️ An uncached forward is what training differentiates. It used the
    /// fused GDN Metal kernel, which has no VJP, so value_and_grad threw
    /// "[Primitive::vjp] Not implemented for CustomKernel" and LoRA training
    /// on Qwen3.5 reported NaN loss. (The bridge turns the throw into a
    /// placeholder array, so check its error channel, not just the shape.)
    #[test]
    #[serial]
    fn test_gdn_uncached_forward_is_differentiable() {
        // The fused kernel only takes key dims that are a multiple of 32, so
        // tiny_config's 16 would never reach it.
        let config = Qwen3NextConfig {
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            ..tiny_config()
        };
        let mut gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[1, 4, 32],
            pmetal_runtime::compat::Dtype::Float32,
        );
        pmetal_runtime::clear_last_error();
        let (loss, grads) = pmetal_runtime::compat::nn::value_and_grad_explicit(
            |arrays: &[Array]| gdn.forward(&arrays[0], None, None).unwrap().sum(None),
            std::slice::from_ref(&x),
            &[],
        )
        .unwrap();
        pmetal_runtime::check_unobserved_error().expect("no bridge op failed under value_and_grad");

        assert!(loss.item::<f32>().is_finite());
        // Evaluate before reading: as_slice on a lazy array reads a null pointer.
        grads[0].eval();
        pmetal_runtime::check_unobserved_error().expect("the gradient evaluates");
        let dx = grads[0].as_slice::<f32>();
        assert!(dx.iter().all(|g| g.is_finite()), "finite input gradient");
        assert!(dx.iter().any(|g| *g != 0.0), "non-zero input gradient");
    }

    #[test]
    fn test_sanitize_weights_keeps_legacy_norm_shift_when_needed() {
        let config = tiny_config();
        let mut weights = HashMap::from([
            (
                "model.language_model.layers.0.input_layernorm.weight".to_string(),
                Array::from_slice(&[0.25f32, -0.5, 1.0], &[3]),
            ),
            (
                "model.language_model.layers.0.linear_attn.conv1d.weight".to_string(),
                Array::zeros_f32(&[4, 1, 4]),
            ),
            ("mtp.fc.weight".to_string(), Array::from_f32(1.0)),
        ]);

        sanitize_weights(&mut weights, &config, Qwen3NextSanitizeOptions::default()).unwrap();

        let shifted = weights
            .get("model.layers.0.input_layernorm.weight")
            .unwrap()
            .cast(pmetal_runtime::compat::Dtype::Float32)
            .unwrap();
        // cast() is lazy; evaluate before reading via as_slice() to avoid a
        // null-dereference inside Buffer::raw_ptr on an unevaluated array.
        shifted.eval();
        let shifted_vec: Vec<f32> = shifted.as_slice().to_vec();
        assert_eq!(shifted_vec, vec![1.25, 0.5, 2.0]);
    }

    #[test]
    #[serial]
    fn test_gdn_combined_input_projection_matches_separate_linears() {
        let config = tiny_config();
        let mut gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[1, 1, 32],
            pmetal_runtime::compat::Dtype::Float32,
        );

        let separate = gdn.project(&x, false).unwrap();
        let fused = gdn.project(&x, true).unwrap();
        let max_diff = |a: &Array, b: &Array| {
            assert_eq!(a.shape(), b.shape());
            a.subtract(b).abs().max(None).item::<f32>()
        };
        for (name, fused, separate) in [
            ("qkv", &fused.0, &separate.0),
            ("z", &fused.1, &separate.1),
            ("b", &fused.2, &separate.2),
            ("a", &fused.3, &separate.3),
        ] {
            let diff = max_diff(fused, separate);
            assert!(diff < 1e-5, "fused {name} projection diverged: {diff}");
        }
    }

    #[test]
    fn test_gdn_combined_input_projection_is_gated_by_hidden_size_and_decode_shape() {
        let mut config = tiny_config();
        config.hidden_size = 2048;
        let gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let decode = Array::zeros_f32(&[1, 1, 2048]);
        let prefill = Array::zeros_f32(&[1, 2, 2048]);

        // Only a decode step that has a cache to decode against fuses.
        assert!(gdn.uses_fused_projection(&decode, None, true));
        assert!(!gdn.uses_fused_projection(&decode, None, false));
        assert!(!gdn.uses_fused_projection(&prefill, None, true));

        let mut large_config = config.clone();
        large_config.hidden_size = 4096;
        let large_gdn = Qwen3NextGatedDeltaNet::new(&large_config).unwrap();
        assert!(!large_gdn.uses_fused_projection(&decode, None, true));
    }

    #[test]
    fn test_sparse_moe_shared_forward_matches_separate_paths() {
        let config = tiny_config();
        let mut moe = Qwen3NextSparseMoeBlock::new(&config).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[5, config.hidden_size],
            pmetal_runtime::compat::Dtype::Float32,
        );

        let shared_ref = moe.shared_expert.forward(&x).unwrap();
        let gate_ref = moe.shared_expert_gate.forward(&x).unwrap();
        let (shared_y, shared_gate_logit) = moe.forward_shared_expert_and_gate(&x).unwrap();

        shared_ref.eval().unwrap();
        gate_ref.eval().unwrap();
        shared_y.eval().unwrap();
        shared_gate_logit.eval().unwrap();

        let max_shared_diff = shared_y
            .subtract(&shared_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        let max_gate_diff = shared_gate_logit
            .subtract(&gate_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();

        assert!(
            max_shared_diff < 1e-5,
            "combined shared expert diverged: {max_shared_diff}"
        );
        assert!(
            max_gate_diff < 1e-5,
            "combined shared gate diverged: {max_gate_diff}"
        );
    }

    #[test]
    fn test_sparse_moe_shared_combined_projection_refreshes_after_weight_replacement() {
        let config = tiny_config();
        let mut moe = Qwen3NextSparseMoeBlock::new(&config).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[3, config.hidden_size],
            pmetal_runtime::compat::Dtype::Float32,
        );

        let _ = moe.forward_shared_expert_and_gate(&x).unwrap();
        let initial_signature = moe.shared_combined_in_proj_signature.clone().unwrap();

        let replacement = Array::ones_f32(&[1, config.hidden_size]);
        moe.shared_expert_gate.weight = Param::new(replacement);

        let gate_ref = moe.shared_expert_gate.forward(&x).unwrap();
        let (_, shared_gate_logit) = moe.forward_shared_expert_and_gate(&x).unwrap();

        gate_ref.eval().unwrap();
        shared_gate_logit.eval().unwrap();

        let max_gate_diff = shared_gate_logit
            .subtract(&gate_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();

        assert!(
            max_gate_diff < 1e-5,
            "refreshed shared gate diverged: {max_gate_diff}"
        );
        assert_ne!(
            moe.shared_combined_in_proj_signature.as_ref().unwrap(),
            &initial_signature
        );
    }

    /// The sparse block returns its own output; the decoder layer adds the
    /// residual. With every weight zero nothing should come out. It used to
    /// return its input, which the decoder then added to the residual stream a
    /// second time.
    #[test]
    #[serial]
    fn test_sparse_moe_block_output_excludes_its_input() {
        use pmetal_runtime::compat::ModuleParametersExt;

        let config = Qwen3NextConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            ..tiny_config()
        };
        let mut moe = Qwen3NextSparseMoeBlock::new(&config).unwrap();
        for (_, p) in moe.flatten_params_mut() {
            let shape = p.shape().to_vec();
            *p = Array::zeros_f32(&shape);
        }
        let x = Array::ones_f32(&[1, 3, config.hidden_size]);
        let y = moe.forward(&x).unwrap();
        y.eval();
        let values: Vec<f32> = y.as_slice().to_vec();
        assert!(
            values.iter().all(|&v| v == 0.0),
            "zero-weight MoE block returned its input: {values:?}"
        );
    }

    /// A prefill sorts its (token, expert) pairs by expert before the gather
    /// matmuls and a decode step does not; the two orders compute the same
    /// block, values and gradients: tokens run alone (router order) against the
    /// whole sequence in one pass (grouped), the expert weights' gradient
    /// summed over the tokens against the whole sequence's.
    #[test]
    #[serial]
    fn a_sequence_through_a_sparse_block_is_its_tokens_through_it() {
        let config = Qwen3NextConfig {
            num_experts: 8,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            ..tiny_config()
        };
        let mut moe = Qwen3NextSparseMoeBlock::new(&config).unwrap();
        let randn = |shape: &[i32]| pmetal_runtime::compat::random::normal(shape, Dtype::Float32);
        *moe.gate.weight = randn(&[8, 32]);
        *moe.switch_mlp_gate_proj = randn(&[8, 16, 32]);
        *moe.switch_mlp_up_proj = randn(&[8, 16, 32]);
        *moe.switch_mlp_down_proj = randn(&[8, 32, 16]);
        *moe.shared_expert.gate_proj.weight = randn(&[32, 32]);
        *moe.shared_expert.up_proj.weight = randn(&[32, 32]);
        *moe.shared_expert.down_proj.weight = randn(&[32, 32]);
        *moe.shared_expert_gate.weight = randn(&[1, 32]);

        // 40 tokens x 2 experts: past the 64-pair threshold only with a
        // second row, so use two sequences' worth.
        let tokens = 40;
        let x = randn(&[2, tokens, 32]);
        let max_diff = |a: &Array, b: &Array| a.subtract(b).abs().max(None).item::<f32>();

        // Values: the whole batch against one token at a time.
        let whole = moe.forward(&x).unwrap();
        let mut rows = Vec::new();
        for b in 0..2 {
            for t in 0..tokens {
                let token = slice_axis(&slice_axis(&x, 0, b, b + 1), 1, t, t + 1);
                rows.push(moe.forward(&token).unwrap().reshape(&[1, 32]));
            }
        }
        let rows_ref: Vec<&Array> = rows.iter().collect();
        let alone = ops::concatenate_axis(&rows_ref, 0).reshape(&[2, tokens, 32]);
        let value_scale = whole.abs().max(None).item::<f32>().max(1.0);
        assert!(
            max_diff(&whole, &alone) / value_scale < 1e-5,
            "values differ by {} (scale {value_scale})",
            max_diff(&whole, &alone)
        );

        // Gradient of sum(output) with respect to the gate expert stack.
        let stack = moe.switch_mlp_gate_proj.as_ref().clone();
        let mut grad_of = |input: &Array| {
            let (_, grads) = pmetal_runtime::compat::nn::value_and_grad_explicit(
                |arrays: &[Array]| {
                    *moe.switch_mlp_gate_proj = arrays[0].clone();
                    moe.forward(&arrays[1]).unwrap().sum(None)
                },
                &[stack.clone(), input.clone()],
                &[],
            )
            .unwrap();
            pmetal_runtime::check_unobserved_error().expect("no op failed under the trace");
            grads
        };
        let whole_grads = grad_of(&x);
        let mut summed: Option<Array> = None;
        let mut dx_rows = Vec::new();
        for b in 0..2 {
            for t in 0..tokens {
                let token = slice_axis(&slice_axis(&x, 0, b, b + 1), 1, t, t + 1);
                let grads = grad_of(&token);
                summed = Some(match summed {
                    Some(acc) => acc.add(&grads[0]),
                    None => grads[0].clone(),
                });
                dx_rows.push(grads[1].reshape(&[1, 32]));
            }
        }
        let summed = summed.unwrap();
        let dx_refs: Vec<&Array> = dx_rows.iter().collect();
        let dx_alone = ops::concatenate_axis(&dx_refs, 0).reshape(&[2, tokens, 32]);
        assert!(
            whole_grads[0].abs().sum(None).item::<f32>() > 0.0,
            "the stack's gradient is not zero"
        );
        let scale = summed.abs().max(None).item::<f32>().max(1.0);
        assert!(
            max_diff(&whole_grads[0], &summed) / scale < 1e-4,
            "expert gradient differs by {} (scale {scale})",
            max_diff(&whole_grads[0], &summed)
        );
        let dx_scale = dx_alone.abs().max(None).item::<f32>().max(1.0);
        assert!(
            max_diff(&whole_grads[1], &dx_alone) / dx_scale < 1e-4,
            "input gradient differs by {} (scale {dx_scale})",
            max_diff(&whole_grads[1], &dx_alone)
        );
    }

    /// The routing weights are float32 (the softmax is), but the block
    /// computes in the model's dtype: the weights are cast to it before they
    /// scale the experts' outputs, as transformers does, so a bf16 model's
    /// hidden state stays bf16 through a sparse layer.
    #[test]
    #[serial]
    fn a_bf16_sparse_block_returns_bf16() {
        use pmetal_runtime::compat::ModuleParametersExt;

        let config = Qwen3NextConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            ..tiny_config()
        };
        let mut moe = Qwen3NextSparseMoeBlock::new(&config).unwrap();
        for (_, p) in moe.flatten_params_mut() {
            let shape = p.shape().to_vec();
            *p = pmetal_runtime::compat::random::normal(&shape, Dtype::Bfloat16);
        }
        let x =
            pmetal_runtime::compat::random::normal(&[1, 3, config.hidden_size], Dtype::Bfloat16);
        let y = moe.forward(&x).unwrap();
        y.eval();
        assert_eq!(y.dtype(), Dtype::Bfloat16);
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    /// A decode step through the cache is the last position of the whole
    /// sequence run in one pass without one: output, convolution window and
    /// recurrent state.
    #[test]
    #[serial]
    fn test_gdn_cached_decode_matches_the_whole_sequence() {
        let config = tiny_config();
        let mut gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let sequence = pmetal_runtime::compat::random::normal(
            &[1, 4, 32],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let max_diff = |a: &Array, b: &Array| a.subtract(b).abs().max(None).item::<f32>();

        let mut whole_cache = MambaCacheEntry::default();
        let whole = gdn
            .forward(&sequence, None, Some(&mut whole_cache))
            .unwrap();

        let mut cache = MambaCacheEntry::default();
        gdn.forward(&slice_axis(&sequence, 1, 0, 3), None, Some(&mut cache))
            .unwrap();
        let decoded = gdn
            .forward(&slice_axis(&sequence, 1, 3, 4), None, Some(&mut cache))
            .unwrap();

        let output_diff = max_diff(&decoded, &slice_axis(&whole, 1, 3, 4));
        assert!(
            output_diff < 1e-4,
            "cached decode output diverged from the whole sequence: {output_diff}"
        );
        let conv_diff = max_diff(
            cache.conv_state.as_ref().unwrap(),
            whole_cache.conv_state.as_ref().unwrap(),
        );
        assert!(
            conv_diff < 1e-6,
            "cached decode conv state diverged: {conv_diff}"
        );
        let ssm_diff = max_diff(
            cache.ssm_state.as_ref().unwrap(),
            whole_cache.ssm_state.as_ref().unwrap(),
        );
        assert!(
            ssm_diff < 1e-4,
            "cached decode SSM state diverged: {ssm_diff}"
        );
    }

    /// A GDN layer with its weights rounded to bf16, as a released checkpoint
    /// loads them.
    fn bf16_gdn(config: &Qwen3NextConfig) -> Qwen3NextGatedDeltaNet {
        let mut gdn = Qwen3NextGatedDeltaNet::new(config).unwrap();
        let to_bf16 = |w: &Array| w.as_dtype(Dtype::Bfloat16.as_i32());
        for linear in [
            &mut gdn.in_proj_qkv,
            &mut gdn.in_proj_z,
            &mut gdn.in_proj_b,
            &mut gdn.in_proj_a,
            &mut gdn.out_proj,
        ] {
            *linear.weight = to_bf16(linear.weight.as_ref());
        }
        *gdn.conv1d.weight = to_bf16(gdn.conv1d.weight.as_ref());
        *gdn.norm.weight = to_bf16(gdn.norm.weight.as_ref());
        *gdn.a_log = to_bf16(gdn.a_log.as_ref());
        *gdn.dt_bias = to_bf16(gdn.dt_bias.as_ref());
        gdn
    }

    /// The recurrent state is a running sum of rank-one updates over the whole
    /// context: rounding it to the model's bf16 after every token loses a
    /// little of every update, and the loss compounds. It is carried in
    /// float32 whatever the activations' dtype; the conv state, a window of
    /// the activations themselves, keeps theirs.
    #[test]
    #[serial]
    fn the_recurrent_state_is_carried_in_f32_for_a_bf16_layer() {
        let config = tiny_config();
        let mut gdn = bf16_gdn(&config);
        let bf16 = |shape: &[i32]| {
            pmetal_runtime::compat::random::normal(shape, Dtype::Float32)
                .as_dtype(Dtype::Bfloat16.as_i32())
        };
        let mut cache = MambaCacheEntry::default();

        // A cached prefill, then a decode step (the fused projection path).
        gdn.forward(&bf16(&[1, 5, 32]), None, Some(&mut cache))
            .unwrap();
        assert_eq!(
            cache.ssm_state.as_ref().unwrap().dtype(),
            Dtype::Float32,
            "state after a prefill"
        );
        assert_eq!(cache.conv_state.as_ref().unwrap().dtype(), Dtype::Bfloat16);
        gdn.forward(&bf16(&[1, 1, 32]), None, Some(&mut cache))
            .unwrap();
        assert_eq!(
            cache.ssm_state.as_ref().unwrap().dtype(),
            Dtype::Float32,
            "state after a decode step"
        );
        assert_eq!(cache.conv_state.as_ref().unwrap().dtype(), Dtype::Bfloat16);
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    /// Weights replaced after a decode step (a merge, a reload, an optimizer
    /// step) are the ones the next step multiplies by: the decode fast path
    /// keeps a concatenated copy of the four input projections.
    #[test]
    #[serial]
    fn a_decode_step_uses_the_weights_in_place_after_they_are_replaced() {
        let config = tiny_config();
        let mut gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let prefill = pmetal_runtime::compat::random::normal(&[1, 3, 32], Dtype::Float32);
        let decode = pmetal_runtime::compat::random::normal(&[1, 1, 32], Dtype::Float32);
        let mut cache = MambaCacheEntry::default();
        gdn.forward(&prefill, None, Some(&mut cache)).unwrap();
        // Populate the fast path's concatenated copy.
        let mut warm = cache.clone();
        gdn.forward(&decode, None, Some(&mut warm)).unwrap();

        for linear in [
            &mut gdn.in_proj_qkv,
            &mut gdn.in_proj_z,
            &mut gdn.in_proj_b,
            &mut gdn.in_proj_a,
        ] {
            let shape = linear.weight.as_ref().shape().to_vec();
            *linear.weight = pmetal_runtime::compat::random::normal(&shape, Dtype::Float32);
        }

        // The same layer, built fresh around the replaced weights.
        let mut fresh = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        *fresh.in_proj_qkv.weight = gdn.in_proj_qkv.weight.as_ref().clone();
        *fresh.in_proj_z.weight = gdn.in_proj_z.weight.as_ref().clone();
        *fresh.in_proj_b.weight = gdn.in_proj_b.weight.as_ref().clone();
        *fresh.in_proj_a.weight = gdn.in_proj_a.weight.as_ref().clone();
        *fresh.out_proj.weight = gdn.out_proj.weight.as_ref().clone();
        *fresh.conv1d.weight = gdn.conv1d.weight.as_ref().clone();
        *fresh.norm.weight = gdn.norm.weight.as_ref().clone();
        *fresh.a_log = gdn.a_log.as_ref().clone();
        *fresh.dt_bias = gdn.dt_bias.as_ref().clone();

        let mut fresh_cache = cache.clone();
        let want = fresh
            .forward(&decode, None, Some(&mut fresh_cache))
            .unwrap();
        let got = gdn.forward(&decode, None, Some(&mut cache)).unwrap();
        let max_diff = want.subtract(&got).abs().max(None).item::<f32>();
        assert!(
            max_diff < 1e-4,
            "a decode step after a weight replacement differs from a layer built on those weights by {max_diff}"
        );
    }

    /// A prompt prefilled in passes of any size leaves the layer's state, and
    /// produces the outputs, of one pass over the whole prompt: the passes cut
    /// the convolution window (kernel 4) at every offset.
    #[test]
    #[serial]
    fn passes_of_any_size_leave_the_state_of_one_pass() {
        let config = tiny_config();
        let mut gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let tokens = 11;
        let x = pmetal_runtime::compat::random::normal(&[1, tokens, 32], Dtype::Float32);
        let max_abs = |a: &Array, b: &Array| {
            let d = a.subtract(b).unwrap().abs().unwrap();
            d.max(None).unwrap().item::<f32>()
        };

        let mut whole_cache = MambaCacheEntry::default();
        let whole = gdn.forward(&x, None, Some(&mut whole_cache)).unwrap();
        for chunk in [1, 2, 3, 4, 5, 7] {
            let mut cache = MambaCacheEntry::default();
            let mut outputs = Vec::new();
            let mut start = 0;
            while start < tokens {
                let end = (start + chunk).min(tokens);
                outputs.push(
                    gdn.forward(&slice_axis(&x, 1, start, end), None, Some(&mut cache))
                        .unwrap(),
                );
                start = end;
            }
            let refs: Vec<&Array> = outputs.iter().collect();
            let joined = ops::concatenate_axis(&refs, 1);
            assert!(
                max_abs(&joined, &whole) < 1e-4,
                "chunk {chunk}: outputs differ by {}",
                max_abs(&joined, &whole)
            );
            assert!(
                max_abs(
                    cache.conv_state.as_ref().unwrap(),
                    whole_cache.conv_state.as_ref().unwrap()
                ) < 1e-6,
                "chunk {chunk}: conv state"
            );
            assert!(
                max_abs(
                    cache.ssm_state.as_ref().unwrap(),
                    whole_cache.ssm_state.as_ref().unwrap()
                ) < 1e-4,
                "chunk {chunk}: recurrent state"
            );
        }
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    #[test]
    #[serial]
    fn test_profiled_forward_collects_layer_sections() {
        let config = tiny_config();
        let mut model = Qwen3NextForCausalLM::new(config).unwrap();
        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);

        let (logits, profile) = model
            .forward_with_cache_profiled(&input_ids, None, None, None, "prefill")
            .unwrap();

        assert_eq!(logits.shape(), &[1, 4, 100]);
        assert_eq!(profile.phase, "prefill");
        assert_eq!(profile.layers.len(), 4);
        assert!(profile.embedding_us > 0);
        assert!(profile.final_norm_us > 0);
        assert!(profile.lm_head_us > 0);
        assert!(profile.total_us >= profile.embedding_us + profile.final_norm_us);

        let linear_layer = &profile.layers[0];
        assert_eq!(linear_layer.layer_kind, "linear_attention");
        assert!(
            linear_layer
                .sections
                .iter()
                .any(|section| section.name == "gdn_recurrence")
        );

        let attn_layer = &profile.layers[3];
        assert_eq!(attn_layer.layer_kind, "full_attention");
        assert!(
            attn_layer
                .sections
                .iter()
                .any(|section| section.name == "attn_sdpa")
        );
    }

    #[test]
    #[serial]
    fn test_decoder_layer_shapes() {
        let config = tiny_config();

        // Linear (GDN) layer
        let mut layer0 = Qwen3NextDecoderLayer::new(&config, 0).unwrap();
        assert!(layer0.is_linear);
        let x = pmetal_runtime::compat::random::normal(
            &[1, 4, 32],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let out0 = layer0.forward(&x, None, None, None, None).unwrap();
        assert_eq!(out0.shape(), &[1, 4, 32]);

        // Full attention layer
        let mut layer3 = Qwen3NextDecoderLayer::new(&config, 3).unwrap();
        assert!(!layer3.is_linear);
        let out3 = layer3.forward(&x, None, None, None, None).unwrap();
        assert_eq!(out3.shape(), &[1, 4, 32]);
    }

    #[test]
    #[serial]
    fn test_causal_lm_forward_shape() {
        let config = tiny_config();
        let vocab_size = config.vocab_size;
        let mut model = Qwen3NextForCausalLM::new(config).unwrap();

        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 4, vocab_size]);
    }

    #[test]
    #[serial]
    fn test_tie_word_embeddings() {
        let mut config = tiny_config();
        config.tie_word_embeddings = true;
        let model = Qwen3NextForCausalLM::new(config).unwrap();
        assert!(model.lm_head.is_none());

        let mut config = tiny_config();
        config.tie_word_embeddings = false;
        let model = Qwen3NextForCausalLM::new(config).unwrap();
        assert!(model.lm_head.is_some());
    }

    #[test]
    fn test_text_config_nesting_parse() {
        // Simulates the real Qwen 3.5 config.json from HuggingFace which wraps
        // model params inside `text_config` (VLM wrapper format).
        let nested_json = r#"{
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_next",
                "hidden_size": 1536,
                "intermediate_size": 8960,
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": true,
                "linear_num_value_heads": 8,
                "linear_num_key_heads": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "partial_rotary_factor": 0.25,
                "attention_bias": false,
                "num_experts": 0,
                "num_experts_per_tok": 0,
                "decoder_sparse_step": 1,
                "moe_intermediate_size": 0,
                "shared_expert_intermediate_size": 0,
                "norm_topk_prob": false,
                "rope_parameters": {
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25,
                    "rope_type": "default"
                },
                "layer_types": [
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention"
                ]
            }
        }"#;

        let config = Qwen3NextConfig::from_config_json(nested_json).unwrap();

        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert_eq!(config.partial_rotary_factor, 0.25);
        assert_eq!(config.layer_types.as_ref().unwrap().len(), 28);

        // Verify layer type detection using the explicit array
        assert!(config.is_linear_layer(0));
        assert!(config.is_linear_layer(1));
        assert!(config.is_linear_layer(2));
        assert!(!config.is_linear_layer(3)); // full_attention
        assert!(config.is_linear_layer(4));
        assert!(!config.is_linear_layer(7)); // full_attention
    }

    #[test]
    fn test_qwen35_moe_text_config_without_intermediate_size_parses() {
        let nested_json = r#"{
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 3072,
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": false,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 1024,
                "shared_expert_intermediate_size": 1024,
                "mlp_only_layers": []
            }
        }"#;

        let config = Qwen3NextConfig::from_config_json(nested_json).unwrap();

        assert_eq!(config.intermediate_size, 1024);
        assert!(config.use_moe_at(0));
        assert!(config.use_moe_at(1));
    }

    #[test]
    fn test_sanitize_weights_stacks_fused_qwen35_moe_experts() {
        let mut config = tiny_config();
        config.num_hidden_layers = 1;
        config.num_experts = 2;
        config.num_experts_per_tok = 2;
        config.moe_intermediate_size = 4;
        config.hidden_size = 8;

        let gate_up_data: Vec<f32> = (0..(2 * 8 * 8)).map(|i| i as f32).collect();
        let down_data: Vec<f32> = (0..(2 * 8 * 4)).map(|i| 1000.0 + i as f32).collect();
        let mut weights = HashMap::from([
            (
                "model.language_model.layers.0.mlp.experts.gate_up_proj".to_string(),
                Array::from_slice(&gate_up_data, &[2, 8, 8]),
            ),
            (
                "model.language_model.layers.0.mlp.experts.down_proj".to_string(),
                Array::from_slice(&down_data, &[2, 8, 4]),
            ),
        ]);

        sanitize_weights(&mut weights, &config, Qwen3NextSanitizeOptions::default()).unwrap();

        let gate = weights
            .get("model.layers.0.mlp.switch_mlp_gate_proj")
            .unwrap();
        let up = weights
            .get("model.layers.0.mlp.switch_mlp_up_proj")
            .unwrap();
        let down = weights
            .get("model.layers.0.mlp.switch_mlp_down_proj")
            .unwrap();
        assert_eq!(gate.shape(), &[2, 4, 8]);
        assert_eq!(up.shape(), &[2, 4, 8]);
        assert_eq!(down.shape(), &[2, 8, 4]);
        assert!(!weights.contains_key("model.language_model.layers.0.mlp.experts.gate_up_proj"));
    }

    #[test]
    fn test_sanitize_weights_can_skip_fused_routed_experts() {
        let mut config = tiny_config();
        config.num_hidden_layers = 1;
        config.num_experts = 2;
        config.num_experts_per_tok = 2;
        config.moe_intermediate_size = 4;
        config.hidden_size = 8;

        let mut weights = HashMap::from([
            (
                "model.language_model.layers.0.mlp.experts.gate_up_proj".to_string(),
                Array::zeros_f32(&[2, 8, 8]),
            ),
            (
                "model.language_model.layers.0.mlp.experts.down_proj".to_string(),
                Array::zeros_f32(&[2, 8, 4]),
            ),
        ]);

        sanitize_weights(
            &mut weights,
            &config,
            Qwen3NextSanitizeOptions {
                skip_routed_experts: true,
            },
        )
        .unwrap();

        assert!(weights.is_empty());
    }

    #[test]
    #[serial]
    fn test_placeholder_sparse_moe_block_requires_offload_before_forward() {
        let mut config = tiny_config();
        config.num_experts = 2;
        config.num_experts_per_tok = 2;
        config.moe_intermediate_size = 8;
        config.shared_expert_intermediate_size = 16;

        let mut block = Qwen3NextSparseMoeBlock::new_with_routed_expert_mode(
            &config,
            Qwen3NextRoutedExpertMode::Placeholder,
        )
        .unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[1, 2, config.hidden_size],
            pmetal_runtime::compat::Dtype::Float32,
        );
        let err = block.forward(&x).unwrap_err().to_string();
        assert!(err.contains("enable expert offloading"));
    }
}
