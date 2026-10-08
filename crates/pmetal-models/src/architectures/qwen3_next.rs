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

use pmetal_bridge::compat::ops::{
    select_axis, slice_axis, slice_axis_from, slice_last_from, slice_last_to,
};
use pmetal_bridge::compat::{
    Array, Dtype, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, Param,
    VisitLinears, fast, nn, ops, random,
};
use pmetal_bridge::impl_module_params;
use pmetal_bridge::qwen3_native::family::gdn_qk_rms_norm_eps;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::expert_io::ExpertOffloadContext;
use crate::expert_prefetch::{ExpertPrefetcher, PrefetchedExpert};
use crate::fp8_utils::dequantize_fp8_weight_for_compute;
use crate::traits::ModelConfig;
use pmetal_bridge::native_weight::LayerWeight;
use pmetal_metal::expert_buffer::{ExpertBufferPool, ExpertBufferPoolConfig};
use pmetal_mlx::kv_cache::{KVCache, MambaCache, MambaCacheEntry, SequenceKey};
use pmetal_mlx::{
    gather_mm,
    kernels::{
        AttentionMaskType, FusedAttentionConfig,
        fused_moe::moe_combine_mlx,
        fused_sdpa,
        gated_delta::{self, gated_delta_update},
        metal_swiglu::fused_swiglu_forward,
        rope::{RopePositions, RopeScaling, rope},
    },
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// [`family::normalize_text_config`](pmetal_bridge::qwen3_native::family::normalize_text_config),
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
        let text = pmetal_bridge::qwen3_native::family::normalize_text_config(config)
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

fn profile_array_section<F>(layer_profile: &mut Qwen3NextLayerProfile, name: &str, op: F) -> Array
where
    F: FnOnce() -> Array,
{
    let start = Instant::now();
    let output = op();
    output.eval();
    layer_profile.push_section(name, start);
    output
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
pub use pmetal_bridge::qwen3_native::family::GdnGateActivation as GateActivation;

/// Compiled _precise_swiglu: `(silu(gate.f32()) * norm_out.f32()).as(dtype)`.
///
/// Equivalent to a `@partial(mx.compile, shapeless=True)` `_precise_swiglu(h, gate, x)`.
/// Fuses 6 element-wise ops (2 casts, sigmoid, 2 multiplies, cast) into 1 Metal dispatch.
fn compiled_precise_swiglu(
    norm_out: &Array,
    gate: &Array,
    out_dtype: pmetal_bridge::compat::Dtype,
) -> Result<Array, Exception> {
    thread_local! {
        static COMPILED: std::cell::RefCell<Option<pmetal_bridge::compat::compile::Closure>> =
            const { std::cell::RefCell::new(None) };
    }

    // Initialize the compiled closure on first call
    COMPILED.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let closure = pmetal_bridge::compat::compile::Closure::new(|inputs: &[Array]| {
                let norm_out = &inputs[0];
                let gate = &inputs[1];
                let gate_f32 = nn::silu(&gate.cast(pmetal_bridge::compat::Dtype::Float32));
                let norm_f32 = norm_out.cast(pmetal_bridge::compat::Dtype::Float32);
                let result = gate_f32.multiply(&norm_f32);
                vec![result.as_dtype(norm_out.dtype_raw())]
            });
            *slot = pmetal_bridge::compat::compile::compile(closure, true).ok();
        }
    });

    // Try the compiled path; fall back to uncompiled if unavailable
    let result = COMPILED.with(|cell| -> Option<Result<Array, Exception>> {
        let slot = cell.borrow();
        slot.as_ref().map(|compiled_fn| {
            let outputs = compiled_fn.apply(&[norm_out.clone(), gate.clone()])?;
            let result = outputs.into_iter().next().unwrap();
            Ok(if result.dtype() != out_dtype {
                result.as_dtype(out_dtype.as_i32())
            } else {
                result
            })
        })
    });

    if let Some(r) = result {
        r
    } else {
        // Fallback: uncompiled
        let gate_f32 = nn::silu(&gate.cast(pmetal_bridge::compat::Dtype::Float32));
        let norm_f32 = norm_out.cast(pmetal_bridge::compat::Dtype::Float32);
        Ok(gate_f32.multiply(&norm_f32).as_dtype(out_dtype.as_i32()))
    }
}

impl Qwen3NextRMSNormGated {
    pub fn new(hidden_size: i32, eps: f32) -> Result<Self, Exception> {
        let weight = Array::ones_f32(&[hidden_size]);
        Ok(Self {
            weight: Param::new(weight),
            eps,
            gate_activation: GateActivation::Silu,
        })
    }

    pub fn forward(&self, x: &Array, gate: Option<&Array>) -> Result<Array, Exception> {
        let normed = pmetal_bridge::compat::fast::rms_norm(x, self.weight.as_ref(), self.eps);
        match (gate, self.gate_activation) {
            // Compiled _precise_swiglu (shapeless mx.compile).
            // Fuses silu(gate.f32()) * norm.f32() → cast_back into 1 Metal dispatch
            // instead of 6 separate dispatches (2 casts + silu(2 ops) + mul + cast).
            (Some(g), GateActivation::Silu) => compiled_precise_swiglu(&normed, g, x.dtype()),
            // Same f32 product, sigmoid in place of SiLU.
            (Some(g), GateActivation::Sigmoid) => {
                let gate_f32 = nn::sigmoid(&g.cast(Dtype::Float32));
                Ok(gate_f32
                    .multiply(&normed.cast(Dtype::Float32))
                    .as_dtype(x.dtype().as_i32()))
            }
            (None, _) => Ok(normed),
        }
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

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        fused_swiglu_forward(
            x,
            self.gate_proj.weight.as_ref(),
            self.up_proj.weight.as_ref(),
            self.down_proj.weight.as_ref(),
        )
        .map_err(|e| Exception::custom(e.to_string()))
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

        let rope_scaling = config
            .rope_scaling
            .as_ref()
            .map(RopeScaling::from_config_map)
            .unwrap_or(RopeScaling::None);
        let rope_scale = rope_scaling.scale();
        let effective_base = rope_scaling.effective_base(config.rope_theta, head_dim);

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
            rope_dims: config.rope_dims(),
            effective_base,
            rope_scale,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];
        let cache_active = cache.is_some();
        let mut cache = cache;

        // Project Q (with gate) and K, V
        let q_proj_out = self.q_proj.forward(x);
        // Reshape to [B, L, n_heads, head_dim * 2], split into queries and gate
        let q_gate = q_proj_out.reshape(&[b, l, self.n_heads, self.head_dim * 2]);
        let queries = slice_last_to(&q_gate, self.head_dim);
        let gate =
            slice_last_from(&q_gate, self.head_dim).reshape(&[b, l, self.n_heads * self.head_dim]);

        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);

        // Reshape and apply Q/K norm
        let mut queries = self.q_norm.forward(&queries);
        let mut keys = self
            .k_norm
            .forward(&keys.reshape(&[b, l, self.n_kv_heads, self.head_dim]));
        let values = values.reshape(&[b, l, self.n_kv_heads, self.head_dim]);

        // Transpose to [B, heads, L, head_dim]
        queries = queries.transpose_axes(&[0, 2, 1, 3]);
        keys = keys.transpose_axes(&[0, 2, 1, 3]);
        let values = values.transpose_axes(&[0, 2, 1, 3]);

        // Apply partial RoPE
        let rope_positions = RopePositions::resolve(
            positions,
            cache
                .as_ref()
                .map_or(0, |(c, layer)| c.rope_offset_for(*layer)),
        );
        let queries = rope(
            &queries,
            rope_positions,
            self.rope_dims,
            false,
            self.effective_base,
            self.rope_scale,
        )?;
        let keys = rope(
            &keys,
            rope_positions,
            self.rope_dims,
            false,
            self.effective_base,
            self.rope_scale,
        )?;

        // Fused SDPA with GQA
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(Self::mask_type_for_call(mask, cache_active, l));

        if mask.is_none() {
            if let Some((cache_ref, layer_idx)) = cache.as_mut() {
                if let Some(output) = (*cache_ref).try_turboquant_attention(
                    *layer_idx,
                    &queries,
                    &keys,
                    &values,
                    &attn_config,
                )? {
                    let output = output.transpose_axes(&[0, 2, 1, 3]).reshape(&[
                        b,
                        l,
                        self.n_heads * self.head_dim,
                    ]);
                    let gated = output.multiply(&nn::sigmoid(&gate));
                    return Ok(self.o_proj.forward(&gated));
                }
            }
        }

        // Update KV cache
        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &keys, &values)?
        } else {
            (keys, values)
        };

        let output = fused_sdpa(&queries, &keys, &values, &attn_config, mask)?;
        let output =
            output
                .transpose_axes(&[0, 2, 1, 3])
                .reshape(&[b, l, self.n_heads * self.head_dim]);

        // Gated output: o_proj(output * sigmoid(gate))
        let gated = output.multiply(&nn::sigmoid(&gate));
        Ok(self.o_proj.forward(&gated))
    }

    pub fn forward_profiled(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
        layer_profile: &mut Qwen3NextLayerProfile,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];
        let cache_active = cache.is_some();
        let mut cache = cache;

        let prep_start = Instant::now();
        let q_proj_out = self.q_proj.forward(x);
        let q_gate = q_proj_out.reshape(&[b, l, self.n_heads, self.head_dim * 2]);
        let queries = slice_last_to(&q_gate, self.head_dim);
        let gate =
            slice_last_from(&q_gate, self.head_dim).reshape(&[b, l, self.n_heads * self.head_dim]);
        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);
        let mut queries = self.q_norm.forward(&queries);
        let mut keys = self
            .k_norm
            .forward(&keys.reshape(&[b, l, self.n_kv_heads, self.head_dim]));
        let values = values.reshape(&[b, l, self.n_kv_heads, self.head_dim]);
        queries = queries.transpose_axes(&[0, 2, 1, 3]);
        keys = keys.transpose_axes(&[0, 2, 1, 3]);
        let values = values.transpose_axes(&[0, 2, 1, 3]);
        queries.eval();
        keys.eval();
        values.eval();
        layer_profile.push_section("attn_prepare_qkv", prep_start);

        let rope_cache_start = Instant::now();
        let rope_positions = RopePositions::resolve(
            positions,
            cache
                .as_ref()
                .map_or(0, |(c, layer)| c.rope_offset_for(*layer)),
        );
        let queries = rope(
            &queries,
            rope_positions,
            self.rope_dims,
            false,
            self.effective_base,
            self.rope_scale,
        )?;
        let keys = rope(
            &keys,
            rope_positions,
            self.rope_dims,
            false,
            self.effective_base,
            self.rope_scale,
        )?;
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(Self::mask_type_for_call(mask, cache_active, l));
        if mask.is_none() {
            let turbo_attn_start = Instant::now();
            if let Some((cache_ref, layer_idx)) = cache.as_mut() {
                if let Some(output) = (*cache_ref).try_turboquant_attention(
                    *layer_idx,
                    &queries,
                    &keys,
                    &values,
                    &attn_config,
                )? {
                    queries.eval();
                    output.eval();
                    layer_profile.push_section("attn_rope_cache", rope_cache_start);
                    layer_profile.push_section("attn_sdpa", turbo_attn_start);

                    let out_start = Instant::now();
                    let output = output.transpose_axes(&[0, 2, 1, 3]).reshape(&[
                        b,
                        l,
                        self.n_heads * self.head_dim,
                    ]);
                    let gated = output.multiply(&nn::sigmoid(&gate));
                    let projected = self.o_proj.forward(&gated);
                    projected.eval();
                    layer_profile.push_section("attn_out_proj", out_start);
                    return Ok(projected);
                }
            }
        }
        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &keys, &values)?
        } else {
            (keys, values)
        };
        queries.eval();
        keys.eval();
        values.eval();
        layer_profile.push_section("attn_rope_cache", rope_cache_start);

        let sdpa_start = Instant::now();
        let output = fused_sdpa(&queries, &keys, &values, &attn_config, mask)?;
        output.eval();
        layer_profile.push_section("attn_sdpa", sdpa_start);

        let out_start = Instant::now();
        let output =
            output
                .transpose_axes(&[0, 2, 1, 3])
                .reshape(&[b, l, self.n_heads * self.head_dim]);
        let gated = output.multiply(&nn::sigmoid(&gate));
        let projected = self.o_proj.forward(&gated);
        projected.eval();
        layer_profile.push_section("attn_out_proj", out_start);
        Ok(projected)
    }

    fn mask_type_for_call(
        mask: Option<&Array>,
        cache_active: bool,
        query_len: i32,
    ) -> AttentionMaskType {
        let _ = (cache_active, query_len);
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
    pub combined_in_proj_weight: Option<Array>,
    pub combined_in_proj_signature: Option<Vec<usize>>,
    /// Cached qkvz combined weight (qkv + z concatenated, matching Python's in_proj_qkvz)
    pub combined_qkvz_weight: Option<Array>,
    /// Cached ba combined weight (b + a concatenated, matching Python's in_proj_ba)
    pub combined_ba_weight: Option<Array>,
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
    const COMBINED_INPUT_PROJ_MAX_HIDDEN: i32 = 2048;
    const DECODE_FLATTEN_MAX_HIDDEN: i32 = 2048;

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

        // 4 separate projections matching HF weight format (qwen3_5.py)
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
            pmetal_bridge::compat::random::uniform_range(
                0.0,
                16.0,
                &[num_v_heads],
                pmetal_bridge::compat::Dtype::Float32,
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
            combined_in_proj_weight: None,
            combined_in_proj_signature: None,
            combined_qkvz_weight: None,
            combined_ba_weight: None,
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

    fn current_input_proj_signature(&self) -> Vec<usize> {
        // Use `Array::id()` — stable across the array's lifetime and safe on
        // lazy (unevaluated) weights. `data_ptr()` would dereference a null
        // buffer on an unmaterialised array.
        vec![
            self.in_proj_qkv.weight.as_ref().id(),
            self.in_proj_z.weight.as_ref().id(),
            self.in_proj_b.weight.as_ref().id(),
            self.in_proj_a.weight.as_ref().id(),
        ]
    }

    fn ensure_combined_input_proj_weight(&mut self) -> Result<Array, Exception> {
        let signature = self.current_input_proj_signature();
        let needs_refresh = self.combined_in_proj_weight.is_none()
            || self.combined_in_proj_signature.as_ref() != Some(&signature);

        if needs_refresh {
            let weights = [
                dequantize_fp8_weight_for_compute(self.in_proj_qkv.weight.as_ref())?,
                dequantize_fp8_weight_for_compute(self.in_proj_z.weight.as_ref())?,
                dequantize_fp8_weight_for_compute(self.in_proj_b.weight.as_ref())?,
                dequantize_fp8_weight_for_compute(self.in_proj_a.weight.as_ref())?,
            ];
            let weight_refs: Vec<&Array> = weights.iter().collect();
            let combined = ops::concatenate_axis(&weight_refs, 0);
            combined.eval();
            self.combined_in_proj_weight = Some(combined);
            self.combined_in_proj_signature = Some(signature);
        }

        Ok(self.combined_in_proj_weight.as_ref().unwrap().clone())
    }

    /// Cached qkvz weight = concat(qkv_w, z_w, axis=0).
    /// Matches Python's `in_proj_qkvz` (key_dim*2 + value_dim*2 output dim).
    pub fn ensure_combined_qkvz_weight(&mut self) -> Result<Array, Exception> {
        if self.combined_qkvz_weight.is_none() {
            let qkv = dequantize_fp8_weight_for_compute(self.in_proj_qkv.weight.as_ref())?;
            let z = dequantize_fp8_weight_for_compute(self.in_proj_z.weight.as_ref())?;
            let w = ops::concatenate_axis(&[&qkv, &z], 0);
            w.eval();
            self.combined_qkvz_weight = Some(w);
        }
        Ok(self.combined_qkvz_weight.as_ref().unwrap().clone())
    }

    /// Cached ba weight = concat(b_w, a_w, axis=0).
    /// Matches Python's `in_proj_ba` (num_v_heads*2 output dim).
    pub fn ensure_combined_ba_weight(&mut self) -> Result<Array, Exception> {
        if self.combined_ba_weight.is_none() {
            let b = dequantize_fp8_weight_for_compute(self.in_proj_b.weight.as_ref())?;
            let a = dequantize_fp8_weight_for_compute(self.in_proj_a.weight.as_ref())?;
            let w = ops::concatenate_axis(&[&b, &a], 0);
            w.eval();
            self.combined_ba_weight = Some(w);
        }
        Ok(self.combined_ba_weight.as_ref().unwrap().clone())
    }

    fn combined_input_projection(
        &mut self,
        inputs: &Array,
    ) -> Result<(Array, Array, Array, Array), Exception> {
        let shape = inputs.shape();
        let b = shape[0];
        let s = shape[1];
        let combined_weight = self.ensure_combined_input_proj_weight()?;
        let projected = if s == 1 {
            let flat = inputs.reshape(&[b, self.hidden_size]);
            ops::matmul(&flat, &combined_weight.t())
        } else {
            ops::matmul(inputs, &combined_weight.t())
        };

        let qkv_end = self.conv_dim;
        let z_end = qkv_end + self.value_dim;
        let b_end = z_end + self.num_v_heads;

        let qkv = if s == 1 {
            slice_last_to(&projected, qkv_end).reshape(&[b, s, self.conv_dim])
        } else {
            slice_last_to(&projected, qkv_end)
        };
        let z = slice_axis(&projected, -1, qkv_end, z_end).reshape(&[
            b,
            s,
            self.num_v_heads,
            self.head_v_dim,
        ]);
        let b_val = if s == 1 {
            slice_axis(&projected, -1, z_end, b_end).reshape(&[b, s, self.num_v_heads])
        } else {
            slice_axis(&projected, -1, z_end, b_end)
        };
        let a = if s == 1 {
            slice_last_from(&projected, b_end).reshape(&[b, s, self.num_v_heads])
        } else {
            slice_last_from(&projected, b_end)
        };
        Ok((qkv, z, b_val, a))
    }

    fn decode_linear_projection(
        inputs: &Array,
        weight: &Array,
        output_dim: i32,
    ) -> Result<Array, Exception> {
        let batch = inputs.dim(0);
        let hidden = inputs.dim(2);
        let flat = inputs.reshape(&[batch, hidden]);
        let weight = dequantize_fp8_weight_for_compute(weight)?;
        let projected = ops::matmul(&flat, &weight.t());
        Ok(projected.reshape(&[batch, 1, output_dim]))
    }

    fn decode_out_projection(&self, out: &Array, batch: i32) -> Result<Array, Exception> {
        let flat = out.reshape(&[batch, self.value_dim]);
        let weight = dequantize_fp8_weight_for_compute(self.out_proj.weight.as_ref())?;
        let projected = ops::matmul(&flat, &weight.t());
        Ok(projected.reshape(&[batch, 1, self.hidden_size]))
    }

    fn should_use_flattened_decode_proj(&self, inputs: &Array, mask: Option<&Array>) -> bool {
        mask.is_none() && inputs.dim(1) == 1 && self.hidden_size <= Self::DECODE_FLATTEN_MAX_HIDDEN
    }

    fn should_use_combined_input_proj(&self, inputs: &Array, mask: Option<&Array>) -> bool {
        mask.is_none()
            && inputs.dim(1) == 1
            && self.hidden_size <= Self::COMBINED_INPUT_PROJ_MAX_HIDDEN
    }

    /// Decode forward (T=1) — compiled via mx.compile for fused GPU evaluation.
    ///
    /// MLX traces the function on the first call and replays the cached trace on
    /// subsequent calls. This fuses intermediate ops, dramatically reducing
    /// Metal kernel dispatch count during eval (from ~30 kernels to ~5 per layer).
    fn forward_cached_decode(
        &mut self,
        inputs: &Array,
        cache: &mut MambaCacheEntry,
    ) -> Result<Array, Exception> {
        let b = inputs.dim(0);

        // Extract cache state as explicit arrays for the pure compiled closure
        let conv_state = cache.conv_state.as_ref().cloned().unwrap_or_else(|| {
            ops::zeros_dtype(
                &[b, self.conv_kernel_size - 1, self.conv_dim],
                inputs.dtype(),
            )
        });
        let ssm_state = cache.ssm_state.take().unwrap_or_else(|| {
            ops::zeros_dtype(
                &[b, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                inputs.dtype(),
            )
        });

        // Direct decode: uses Metal GDN kernel (1 dispatch for recurrence) +
        // individually dispatched ops for projections/conv/norm.
        // No per-layer mx.compile — relies on MLX global compile for element-wise fusion.
        let b_dim = inputs.dim(0);
        let s = 1i32;

        // 2 combined matmuls matching Python's in_proj_qkvz + in_proj_ba
        let qkvz_w = self.ensure_combined_qkvz_weight()?;
        let ba_w = self.ensure_combined_ba_weight()?;
        let qkvz = ops::matmul(inputs, &qkvz_w.t());
        let ba = ops::matmul(inputs, &ba_w.t());

        // Split qkvz → qkv (conv_dim) + z (value_dim)
        let qkv = slice_last_to(&qkvz, self.conv_dim);
        let z = slice_last_from(&qkvz, self.conv_dim).reshape(&[
            b_dim,
            s,
            self.num_v_heads,
            self.head_v_dim,
        ]);

        // Split ba → b_val + a
        let b_val = slice_last_to(&ba, self.num_v_heads);
        let a = slice_last_from(&ba, self.num_v_heads);

        // Conv: concat old state + new, apply conv1d + silu
        let conv_input_arr = ops::concatenate_axis(&[&conv_state, &qkv], 1);
        let new_conv = slice_axis_from(&conv_input_arr, 1, -(self.conv_kernel_size - 1));
        let conv_out = nn::silu(&self.conv1d.forward(&conv_input_arr));

        // Split conv output → q, k, v
        let parts = ops::split_sections(&conv_out, &[self.key_dim, self.key_dim * 2], -1);
        let q = parts[0].reshape(&[b_dim, s, self.num_k_heads, self.head_k_dim]);
        let k = parts[1].reshape(&[b_dim, s, self.num_k_heads, self.head_k_dim]);
        let v = parts[2].reshape(&[b_dim, s, self.num_v_heads, self.head_v_dim]);

        // Q/K normalization (fused fast:: ops)
        let q = pmetal_bridge::compat::fast::rms_norm(&q, &self.q_norm_weight, self.qk_norm_eps);
        let k = pmetal_bridge::compat::fast::rms_norm(&k, &self.k_norm_weight, self.qk_norm_eps);

        // GDN recurrence via Metal kernel (1 dispatch) or ops fallback
        let (out, new_ssm) = gated_delta_update(
            &q,
            &k,
            &v,
            &a,
            &b_val,
            self.a_log.as_ref(),
            self.dt_bias.as_ref(),
            Some(&ssm_state),
            None,
            false,
        )?;

        // Gated norm (f32 precision)
        let out_n = self.norm.forward(&out, Some(&z))?;
        let result = self.out_proj.forward(&out_n.reshape(&[b_dim, s, -1]));
        let outputs = [result, new_conv, new_ssm];

        cache.conv_state = Some(outputs[1].clone());
        cache.ssm_state = Some(outputs[2].clone());

        Ok(outputs[0].clone())
    }

    fn forward_cached_decode_profiled(
        &mut self,
        inputs: &Array,
        cache: &mut MambaCacheEntry,
        layer_profile: &mut Qwen3NextLayerProfile,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let b = shape[0];
        let s = shape[1];

        let input_proj_start = Instant::now();
        let (qkv, z, b_val, a) = self.combined_input_projection(inputs)?;
        qkv.eval();
        z.eval();
        b_val.eval();
        a.eval();
        layer_profile.push_section("gdn_input_proj", input_proj_start);

        let conv_start = Instant::now();
        let conv_input = cache.update_conv_state(&qkv, self.conv_kernel_size)?;
        let conv_out = nn::silu(&Module::forward(&mut self.conv1d, &conv_input)?);
        conv_out.eval();
        layer_profile.push_section("gdn_conv", conv_start);

        self.finish_forward_from_conv_out_profiled(
            &conv_out,
            b,
            s,
            &z,
            &a,
            &b_val,
            None,
            Some(cache),
            layer_profile,
        )
    }

    /// Same math as [`finish_forward_from_conv_out`] but writes
    /// `(keys, values, g, beta, conv_input)` into the supplied
    /// `SpecCapture` before running the recurrence, so a speculative
    /// rollback can replay the state over the accepted prefix.
    ///
    /// `conv_input_qkv` is the pre-conv projection (the `qkv` local in
    /// `forward_general_with_capture`) — that is the tensor
    /// `MambaCacheEntry::rewind` needs to reconstruct the rolled-back
    /// conv state.
    #[allow(clippy::too_many_arguments)]
    fn finish_forward_from_conv_out_capturing(
        &mut self,
        conv_out: &Array,
        batch: i32,
        seq_len: i32,
        z: &Array,
        a: &Array,
        b_val: &Array,
        mask: Option<&Array>,
        cache: Option<&mut MambaCacheEntry>,
        capture: Option<(usize, &mut pmetal_mlx::speculative::SpecCapture)>,
        conv_input_qkv: &Array,
        conv_kernel_size: usize,
    ) -> Result<Array, Exception> {
        // Split / reshape / Q-K norm exactly like the non-capturing variant.
        let splits = ops::split_sections(conv_out, &[self.key_dim, self.key_dim * 2], -1);
        let (q_conv, k_conv, v_conv) = (&splits[0], &splits[1], &splits[2]);
        let q_conv = q_conv.reshape(&[batch, seq_len, self.num_k_heads, self.head_k_dim]);
        let k_conv = k_conv.reshape(&[batch, seq_len, self.num_k_heads, self.head_k_dim]);
        let v_conv = v_conv.reshape(&[batch, seq_len, self.num_v_heads, self.head_v_dim]);
        let q_normed =
            pmetal_bridge::compat::fast::rms_norm(&q_conv, &self.q_norm_weight, self.qk_norm_eps);
        let k_normed =
            pmetal_bridge::compat::fast::rms_norm(&k_conv, &self.k_norm_weight, self.qk_norm_eps);

        // Compute g and beta OUTSIDE gated_delta_update so we can stash
        // them in the capture — the internal implementation derives them
        // from (a, b_val, a_log, dt_bias) on every call.
        let beta = b_val.sigmoid();
        let g = pmetal_mlx::kernels::compute_g(self.a_log.as_ref(), a, self.dt_bias.as_ref())?;

        // Record the rollback inputs before advancing the state.
        if let Some((layer_idx, buf)) = capture {
            let rec = pmetal_mlx::kv_cache::GdnVerifyInputs {
                keys: k_normed.clone(),
                values: v_conv.clone(),
                g: g.clone(),
                beta: beta.clone(),
                conv_input: conv_input_qkv.clone(),
                conv_kernel_size,
            };
            buf.record_gdn(layer_idx, rec);
        }

        let ssm_state_arr;
        let ssm_state_ref: Option<&Array> = match cache.as_ref().and_then(|c| c.ssm_state.as_ref())
        {
            Some(s) => Some(s),
            None => {
                ssm_state_arr = pmetal_bridge::compat::ops::zeros_dtype(
                    &[batch, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                    conv_out.dtype(),
                );
                Some(&ssm_state_arr)
            }
        };
        let state_ref = ssm_state_ref.expect("ssm state ref initialized above");

        // Use the inference dispatch with the pre-computed g/beta so the
        // capture and the recurrence share the exact same tensors.
        let (out, new_state) = pmetal_mlx::kernels::gated_delta_inference_dispatch(
            &q_normed, &k_normed, &v_conv, &g, &beta, state_ref, mask,
        )?;
        if let Some(cache) = cache {
            cache.ssm_state = Some(new_state);
        }
        let out = self.norm.forward(&out, Some(z))?;
        if mask.is_none() && seq_len == 1 && self.hidden_size <= Self::DECODE_FLATTEN_MAX_HIDDEN {
            self.decode_out_projection(&out, batch)
        } else {
            Ok(self.out_proj.forward(&out.reshape(&[batch, seq_len, -1])))
        }
    }

    fn finish_forward_from_conv_out(
        &mut self,
        conv_out: &Array,
        batch: i32,
        seq_len: i32,
        z: &Array,
        a: &Array,
        b_val: &Array,
        mask: Option<&Array>,
        cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        // Split conv output into q, k, v using mx.split (1 op vs 3 index ops).
        // Same as the reference: mx.split(conv_out, [...], -1)
        let splits = ops::split_sections(conv_out, &[self.key_dim, self.key_dim * 2], -1);
        let (q_conv, k_conv, v_conv) = (&splits[0], &splits[1], &splits[2]);

        // Reshape to head dimensions.
        // For T=1 decode, conv output seq_len already matches (no trim needed).
        let q_conv = q_conv.reshape(&[batch, seq_len, self.num_k_heads, self.head_k_dim]);
        let k_conv = k_conv.reshape(&[batch, seq_len, self.num_k_heads, self.head_k_dim]);
        let v_conv = v_conv.reshape(&[batch, seq_len, self.num_v_heads, self.head_v_dim]);

        // Q/K normalization: fast::rms_norm (fused Metal op) with pre-baked scale weights.
        let q_normed =
            pmetal_bridge::compat::fast::rms_norm(&q_conv, &self.q_norm_weight, self.qk_norm_eps);
        let k_normed =
            pmetal_bridge::compat::fast::rms_norm(&k_conv, &self.k_norm_weight, self.qk_norm_eps);

        let ssm_state = cache.as_ref().and_then(|c| c.ssm_state.as_ref());

        // An uncached forward is a training (or scoring) pass, which needs the
        // differentiable ops path: the fused GDN Metal kernel has no VJP, so
        // under value_and_grad it failed ("Primitive::vjp Not implemented
        // for CustomKernel") and the loss came out NaN. Same rule as
        // qwen3_next_qlora. Cached prefill and decode keep the kernel.
        let training = cache.is_none();
        let (out, new_state) = gated_delta_update(
            &q_normed,
            &k_normed,
            &v_conv,
            a,
            b_val,
            self.a_log.as_ref(),
            self.dt_bias.as_ref(),
            ssm_state,
            mask,
            training,
        )?;

        if let Some(cache) = cache {
            cache.ssm_state = Some(new_state);
        }

        // Apply gated norm and output projection
        let out = self.norm.forward(&out, Some(z))?;
        if mask.is_none() && seq_len == 1 && self.hidden_size <= Self::DECODE_FLATTEN_MAX_HIDDEN {
            self.decode_out_projection(&out, batch)
        } else {
            Ok(self.out_proj.forward(&out.reshape(&[batch, seq_len, -1])))
        }
    }

    fn finish_forward_from_conv_out_profiled(
        &mut self,
        conv_out: &Array,
        batch: i32,
        seq_len: i32,
        z: &Array,
        a: &Array,
        b_val: &Array,
        mask: Option<&Array>,
        cache: Option<&mut MambaCacheEntry>,
        layer_profile: &mut Qwen3NextLayerProfile,
    ) -> Result<Array, Exception> {
        let recurrence_start = Instant::now();
        let q_conv = slice_last_to(conv_out, self.key_dim);
        let k_conv = slice_axis(conv_out, -1, self.key_dim, self.key_dim * 2);
        let v_conv = slice_last_from(conv_out, self.key_dim * 2);
        let out_len = q_conv.dim(1);
        let q_conv = slice_axis_from(&q_conv, 1, out_len - seq_len).reshape(&[
            batch,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ]);
        let k_conv = slice_axis_from(&k_conv, 1, out_len - seq_len).reshape(&[
            batch,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ]);
        let v_conv = slice_axis_from(&v_conv, 1, out_len - seq_len).reshape(&[
            batch,
            seq_len,
            self.num_v_heads,
            self.head_v_dim,
        ]);
        // Q/K normalization: use fast::rms_norm (1 fused Metal op) instead of
        // l2norm_last_dim (5 separate ops). Matches the reference exactly.
        // Pass ones weight since mlx-rs binding requires a weight array.
        // Q/K normalization: fast::rms_norm (1 fused Metal op) with pre-baked
        // scale factors. inv_scale = 1/sqrt(dk).
        // q gets inv_scale² (because rms_norm divides by sqrt(mean) not sqrt(sum)),
        // k gets inv_scale.
        let q_normed =
            pmetal_bridge::compat::fast::rms_norm(&q_conv, &self.q_norm_weight, self.qk_norm_eps);
        let k_normed =
            pmetal_bridge::compat::fast::rms_norm(&k_conv, &self.k_norm_weight, self.qk_norm_eps);
        let ssm_state = cache.as_ref().and_then(|c| c.ssm_state.as_ref());
        let (out, new_state) = gated_delta_update(
            &q_normed,
            &k_normed,
            &v_conv,
            a,
            b_val,
            self.a_log.as_ref(),
            self.dt_bias.as_ref(),
            ssm_state,
            mask,
            false,
        )?;
        out.eval();
        layer_profile.push_section("gdn_recurrence", recurrence_start);

        if let Some(cache) = cache {
            cache.ssm_state = Some(new_state);
        }

        let out_proj_start = Instant::now();
        let out = self.norm.forward(&out, Some(z))?;
        let projected = if mask.is_none()
            && seq_len == 1
            && self.hidden_size <= Self::DECODE_FLATTEN_MAX_HIDDEN
        {
            self.decode_out_projection(&out, batch)?
        } else {
            self.out_proj.forward(&out.reshape(&[batch, seq_len, -1]))
        };
        projected.eval();
        layer_profile.push_section("gdn_out_proj", out_proj_start);
        Ok(projected)
    }

    fn forward_general(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let b = shape[0];
        let s = shape[1];
        let decode_linear = self.should_use_flattened_decode_proj(inputs, mask);

        // 4 separate projections matching HF weight format (qwen3_5.py:136-139)
        let qkv = if decode_linear {
            Self::decode_linear_projection(inputs, self.in_proj_qkv.weight.as_ref(), self.conv_dim)?
        } else {
            self.in_proj_qkv.forward(inputs)
        };
        let z = if decode_linear {
            Self::decode_linear_projection(inputs, self.in_proj_z.weight.as_ref(), self.value_dim)?
                .reshape(&[b, s, self.num_v_heads, self.head_v_dim])
        } else {
            self.in_proj_z
                .forward(inputs)
                .reshape(&[b, s, self.num_v_heads, self.head_v_dim])
        };
        let b_val = if decode_linear {
            Self::decode_linear_projection(
                inputs,
                self.in_proj_b.weight.as_ref(),
                self.num_v_heads,
            )?
        } else {
            self.in_proj_b.forward(inputs)
        };
        let a = if decode_linear {
            Self::decode_linear_projection(
                inputs,
                self.in_proj_a.weight.as_ref(),
                self.num_v_heads,
            )?
        } else {
            self.in_proj_a.forward(inputs)
        };
        let qkv = if let Some(mask) = mask {
            let mask_expanded = mask.reshape(&[mask.dim(0), mask.dim(1), 1]);
            ops::r#where(&mask_expanded, &qkv, &Array::from_f32(0.0))
        } else {
            qkv
        };

        let conv_out = if let Some(cache_entry) = cache.as_deref_mut() {
            let conv_input = cache_entry.update_conv_state(&qkv, self.conv_kernel_size)?;
            nn::silu(&Module::forward(&mut self.conv1d, &conv_input)?)
        } else {
            let conv_state = pmetal_bridge::compat::ops::zeros_dtype(
                &[b, self.conv_kernel_size - 1, self.conv_dim],
                qkv.dtype(),
            );
            let conv_input = ops::concatenate_axis(&[&conv_state, &qkv], 1);
            nn::silu(&Module::forward(&mut self.conv1d, &conv_input)?)
        };

        self.finish_forward_from_conv_out(&conv_out, b, s, &z, &a, &b_val, mask, cache)
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        if self.should_use_combined_input_proj(inputs, mask)
            && let Some(cache) = cache
        {
            return self.forward_cached_decode(inputs, cache);
        }
        self.forward_general(inputs, mask, cache)
    }

    /// GDN forward that ALSO records the per-token inputs the GDN
    /// recurrence saw, so a speculative-decoding rollback can replay the
    /// state over a partially-accepted prefix.
    ///
    /// Always routes through the general (ops + Metal kernel) path — the
    /// compiled decode fast-path (T=1) is not used during verify, which
    /// by construction runs with `T = block_size >= 2`.
    pub fn forward_with_capture(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: Option<&mut MambaCacheEntry>,
        layer_idx: usize,
        capture: &mut pmetal_mlx::speculative::SpecCapture,
    ) -> Result<Array, Exception> {
        self.forward_general_with_capture(inputs, mask, cache, Some((layer_idx, capture)))
    }

    /// Internal helper: same math as `forward_general` but with an optional
    /// capture slot. When `capture` is `None` this is a line-for-line copy
    /// of `forward_general`; when it is `Some`, the per-token
    /// `(keys, values, g, beta, conv_input)` tensors are stored into the
    /// capture buffer immediately before `gated_delta_update` runs.
    fn forward_general_with_capture(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut MambaCacheEntry>,
        capture: Option<(usize, &mut pmetal_mlx::speculative::SpecCapture)>,
    ) -> Result<Array, Exception> {
        if capture.is_none() {
            return self.forward_general(inputs, mask, cache);
        }
        let shape = inputs.shape();
        let b = shape[0];
        let s = shape[1];

        // Projection phase — same as `forward_general`, minus the
        // decode_linear fast-paths (capture only runs with T > 1).
        let qkv = self.in_proj_qkv.forward(inputs);
        let z = self
            .in_proj_z
            .forward(inputs)
            .reshape(&[b, s, self.num_v_heads, self.head_v_dim]);
        let b_val = self.in_proj_b.forward(inputs);
        let a = self.in_proj_a.forward(inputs);
        let qkv = if let Some(mask) = mask {
            let mask_expanded = mask.reshape(&[mask.dim(0), mask.dim(1), 1]);
            ops::r#where(&mask_expanded, &qkv, &Array::from_f32(0.0))
        } else {
            qkv
        };

        // `qkv_for_capture` is the pre-conv projection the caller will
        // need to rebuild conv_state on rollback.
        let qkv_for_capture = qkv.clone();
        let conv_kernel_size = self.conv_kernel_size as usize;

        let conv_out = if let Some(cache_entry) = cache.as_deref_mut() {
            let conv_input = cache_entry.update_conv_state(&qkv, self.conv_kernel_size)?;
            nn::silu(&Module::forward(&mut self.conv1d, &conv_input)?)
        } else {
            let conv_state = pmetal_bridge::compat::ops::zeros_dtype(
                &[b, self.conv_kernel_size - 1, self.conv_dim],
                qkv.dtype(),
            );
            let conv_input = ops::concatenate_axis(&[&conv_state, &qkv], 1);
            nn::silu(&Module::forward(&mut self.conv1d, &conv_input)?)
        };

        self.finish_forward_from_conv_out_capturing(
            &conv_out,
            b,
            s,
            &z,
            &a,
            &b_val,
            mask,
            cache,
            capture,
            &qkv_for_capture,
            conv_kernel_size,
        )
    }

    pub fn forward_profiled(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut MambaCacheEntry>,
        layer_profile: &mut Qwen3NextLayerProfile,
    ) -> Result<Array, Exception> {
        if self.should_use_combined_input_proj(inputs, mask)
            && let Some(cache) = cache
        {
            return self.forward_cached_decode_profiled(inputs, cache, layer_profile);
        }

        let shape = inputs.shape();
        let b = shape[0];
        let s = shape[1];
        let decode_linear = self.should_use_flattened_decode_proj(inputs, mask);

        let qkv_start = Instant::now();
        let qkv = if decode_linear {
            Self::decode_linear_projection(inputs, self.in_proj_qkv.weight.as_ref(), self.conv_dim)?
        } else {
            self.in_proj_qkv.forward(inputs)
        };
        qkv.eval();
        layer_profile.push_section("gdn_input_qkv", qkv_start);

        let z_start = Instant::now();
        let z = if decode_linear {
            Self::decode_linear_projection(inputs, self.in_proj_z.weight.as_ref(), self.value_dim)?
                .reshape(&[b, s, self.num_v_heads, self.head_v_dim])
        } else {
            self.in_proj_z
                .forward(inputs)
                .reshape(&[b, s, self.num_v_heads, self.head_v_dim])
        };
        z.eval();
        layer_profile.push_section("gdn_input_z", z_start);

        let b_start = Instant::now();
        let b_val = if decode_linear {
            Self::decode_linear_projection(
                inputs,
                self.in_proj_b.weight.as_ref(),
                self.num_v_heads,
            )?
        } else {
            self.in_proj_b.forward(inputs)
        };
        b_val.eval();
        layer_profile.push_section("gdn_input_b", b_start);

        let a_start = Instant::now();
        let a = if decode_linear {
            Self::decode_linear_projection(
                inputs,
                self.in_proj_a.weight.as_ref(),
                self.num_v_heads,
            )?
        } else {
            self.in_proj_a.forward(inputs)
        };
        a.eval();
        layer_profile.push_section("gdn_input_a", a_start);

        let qkv = if let Some(mask) = mask {
            let mask_start = Instant::now();
            let mask_expanded = mask.reshape(&[mask.dim(0), mask.dim(1), 1]);
            let masked = ops::r#where(&mask_expanded, &qkv, &Array::from_f32(0.0));
            masked.eval();
            layer_profile.push_section("gdn_input_mask", mask_start);
            masked
        } else {
            qkv
        };

        let conv_start = Instant::now();
        let conv_out = if let Some(cache_entry) = cache.as_deref_mut() {
            let conv_input = cache_entry.update_conv_state(&qkv, self.conv_kernel_size)?;
            nn::silu(&Module::forward(&mut self.conv1d, &conv_input)?)
        } else {
            let conv_state = pmetal_bridge::compat::ops::zeros_dtype(
                &[b, self.conv_kernel_size - 1, self.conv_dim],
                qkv.dtype(),
            );
            let conv_input = ops::concatenate_axis(&[&conv_state, &qkv], 1);
            nn::silu(&Module::forward(&mut self.conv1d, &conv_input)?)
        };
        conv_out.eval();
        layer_profile.push_section("gdn_conv", conv_start);

        self.finish_forward_from_conv_out_profiled(
            &conv_out,
            b,
            s,
            &z,
            &a,
            &b_val,
            mask,
            cache,
            layer_profile,
        )
    }
}

// ============================================================================
// Sparse MoE Block
// ============================================================================

/// What [`attach_expert_offload`] wired into the MoE blocks, for the model to
/// keep alive and to drive prefetching from its layer loop.
#[derive(Debug)]
pub(crate) struct ExpertOffloadAttachment {
    pub(crate) ctx: Arc<ExpertOffloadContext>,
    pub(crate) runtime: Arc<Qwen3NextOffloadRuntime>,
    pub(crate) prefetcher: Arc<ExpertPrefetcher>,
}

/// Switch `blocks` (each with its decoder-layer index) to experts packed by
/// `pmetal pack-experts` in `experts_dir`:
///
/// 1. Opens the packed expert files
/// 2. Extracts gate weights to build the prefetch predictor
/// 3. Wires offload context and prefetcher into each MoE block
/// 4. Zeros stacked expert weight arrays to reclaim GPU memory
///
/// Shared by every architecture built on [`Qwen3NextSparseMoeBlock`].
pub(crate) fn attach_expert_offload(
    experts_dir: &Path,
    blocks: &mut [(usize, &mut Qwen3NextSparseMoeBlock)],
    num_experts: usize,
    hidden_size: usize,
    top_k: usize,
) -> Result<ExpertOffloadAttachment, Exception> {
    if blocks.is_empty() {
        return Err(Exception::custom("no MoE layers found to offload"));
    }
    let ctx = Arc::new(
        ExpertOffloadContext::new(experts_dir)
            .map_err(|e| Exception::custom(format!("expert offload init: {e}")))?,
    );
    let runtime = Arc::new(Qwen3NextOffloadRuntime::new(
        &ctx,
        top_k,
        Qwen3NextSparseMoeBlock::configured_prefill_expert_window_tokens(),
    ));

    // Extract gate weights from each MoE layer for the prefetcher
    let mut gate_weights: HashMap<usize, Vec<f32>> = HashMap::new();
    for (layer_idx, block) in blocks.iter_mut() {
        // Enable offloading on the block (sets offload_ctx + layer_idx)
        block.enable_offloading(ctx.clone(), *layer_idx, runtime.clone());

        // Extract gate weight matrix for prefetch prediction
        let w = block
            .gate
            .weight
            .as_ref()
            .cast(pmetal_bridge::compat::Dtype::Float32);
        w.eval();
        gate_weights.insert(*layer_idx, w.as_slice().to_vec());
    }
    let num_moe_layers = gate_weights.len();

    let prefetcher = Arc::new(ExpertPrefetcher::new(
        gate_weights,
        num_experts,
        hidden_size,
        top_k,
        runtime.buffer_pool.clone(),
    ));

    // Wire the prefetcher in, and zero the stacked expert arrays to reclaim
    // GPU memory: the offloaded path loads weights from SSD.
    for (_, block) in blocks.iter_mut() {
        block.prefetcher = Some(prefetcher.clone());
        *block.switch_mlp_gate_proj = Array::zeros_f32(&[1]);
        *block.switch_mlp_up_proj = Array::zeros_f32(&[1]);
        *block.switch_mlp_down_proj = Array::zeros_f32(&[1]);
        block.routed_experts_loaded = false;
    }

    tracing::info!(
        moe_layers = num_moe_layers,
        expert_size_mb = ctx.layout.expert_size as f64 / 1e6,
        "Expert offloading enabled with prefetching"
    );
    Ok(ExpertOffloadAttachment {
        ctx,
        runtime,
        prefetcher,
    })
}

#[derive(Debug)]
pub(crate) struct Qwen3NextOffloadRuntime {
    buffer_pool: Option<Arc<ExpertBufferPool>>,
    fused_expert: Option<pmetal_metal::FusedMoeExpert>,
    expert_out_bufs: Vec<pmetal_metal::buffer::MetalBuffer<f32>>,
    expert_intermediate: Option<pmetal_metal::buffer::MetalBuffer<f32>>,
}

impl Qwen3NextOffloadRuntime {
    fn new(ctx: &Arc<ExpertOffloadContext>, top_k: usize, prefill_window_tokens: usize) -> Self {
        let mut runtime = Self {
            buffer_pool: None,
            fused_expert: None,
            expert_out_bufs: Vec::new(),
            expert_intermediate: None,
        };

        let Ok(metal_ctx) = pmetal_metal::context::MetalContext::global() else {
            return runtime;
        };

        let expert_size = ctx.layout.expert_size;
        let total_pool_buffers =
            Qwen3NextSparseMoeBlock::required_pool_buffers(top_k, prefill_window_tokens);
        let k = total_pool_buffers.div_ceil(2);

        match ExpertBufferPool::new(
            &metal_ctx,
            ExpertBufferPoolConfig {
                buffer_size: expert_size,
                k,
            },
        ) {
            Ok(pool) => runtime.buffer_pool = Some(Arc::new(pool)),
            Err(e) => {
                tracing::warn!("ExpertBufferPool allocation failed, using legacy copy path: {e}")
            }
        }

        let bits = match ctx.layout.bits {
            crate::expert_layout::PackedBits::Four => pmetal_metal::ExpertBits::Four,
            crate::expert_layout::PackedBits::Two => pmetal_metal::ExpertBits::Two,
        };
        match pmetal_metal::FusedMoeExpert::new(
            metal_ctx.clone(),
            pmetal_metal::FusedMoeExpertConfig {
                hidden_dim: ctx.layout.hidden_dim as u32,
                intermediate_dim: ctx.layout.intermediate_dim as u32,
                group_size: ctx.layout.group_size as u32,
                bits,
            },
        ) {
            Ok(expert) => runtime.fused_expert = Some(expert),
            Err(e) => tracing::warn!("FusedMoeExpert creation failed: {e}"),
        }

        let output_buffer_count =
            Qwen3NextSparseMoeBlock::required_output_buffers(top_k, prefill_window_tokens);
        let mut out_bufs = Vec::with_capacity(output_buffer_count);
        for _ in 0..output_buffer_count {
            if let Ok(buf) = pmetal_metal::buffer::MetalBuffer::<f32>::new(
                &metal_ctx,
                ctx.layout.hidden_dim,
                pmetal_metal::buffer::BufferUsage::Shared,
            ) {
                out_bufs.push(buf);
            }
        }
        if out_bufs.len() == output_buffer_count {
            runtime.expert_out_bufs = out_bufs;
        }

        if let Ok(scratch) = pmetal_metal::buffer::MetalBuffer::<f32>::new(
            &metal_ctx,
            ctx.layout.intermediate_dim,
            pmetal_metal::buffer::BufferUsage::Shared,
        ) {
            runtime.expert_intermediate = Some(scratch);
        }

        runtime
    }
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
    /// If set, routed expert weights are loaded on-demand from SSD instead of
    /// residing in GPU memory.  `None` means the standard resident path is used.
    pub offload_ctx: Option<Arc<ExpertOffloadContext>>,
    /// If set, pre-gated prediction engine that prefetches experts in the background.
    pub prefetcher: Option<Arc<ExpertPrefetcher>>,
    /// Shared offloaded-MoE runtime resources used by every sparse layer.
    offload_runtime: Option<Arc<Qwen3NextOffloadRuntime>>,
    /// Token window size used by prompt-time exact-routing expert reuse.
    pub prefill_expert_window_tokens: usize,
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
    const DEFAULT_PREFILL_EXPERT_WINDOW_TOKENS: usize = 8;
    const PREFILL_EXPERT_WINDOW_TOKENS_ENV_VAR: &str = "PMETAL_PREFILL_EXPERT_WINDOW_TOKENS";

    fn sanitize_prefill_expert_window_tokens(value: usize) -> usize {
        value.clamp(1, 32)
    }

    fn configured_prefill_expert_window_tokens() -> usize {
        std::env::var(Self::PREFILL_EXPERT_WINDOW_TOKENS_ENV_VAR)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map(Self::sanitize_prefill_expert_window_tokens)
            .unwrap_or(Self::DEFAULT_PREFILL_EXPERT_WINDOW_TOKENS)
    }

    fn required_pool_buffers(top_k: usize, prefill_window_tokens: usize) -> usize {
        top_k
            .max(prefill_window_tokens.saturating_mul(top_k))
            .max(1)
    }

    fn required_output_buffers(top_k: usize, prefill_window_tokens: usize) -> usize {
        prefill_window_tokens
            .saturating_mul(top_k)
            .max(top_k)
            .max(1)
    }

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
        let prefill_expert_window_tokens = Self::configured_prefill_expert_window_tokens();

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
            offload_ctx: None,
            prefetcher: None,
            offload_runtime: None,
            prefill_expert_window_tokens,
            shared_combined_in_proj_weight: None,
            shared_combined_in_proj_signature: None,
            packed_experts: None,
        })
    }

    fn current_shared_input_proj_signature(&self) -> Vec<usize> {
        // Use `Array::id()` — same rationale as current_input_proj_signature.
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

    /// Attach an offload context/runtime and record which layer this block lives in.
    ///
    /// After calling this the [`forward`] method will load routed-expert weights
    /// from SSD rather than from resident GPU arrays.
    fn enable_offloading(
        &mut self,
        ctx: Arc<ExpertOffloadContext>,
        layer_idx: usize,
        runtime: Arc<Qwen3NextOffloadRuntime>,
    ) {
        self.layer_idx = layer_idx;
        self.offload_runtime = Some(runtime);
        self.offload_ctx = Some(ctx);
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Dispatch to offloaded path when an ExpertOffloadContext is present.
        if self.offload_ctx.is_some() {
            return self.forward_offloaded(x, None);
        }
        if let Some(packed) = self.packed_experts.clone() {
            return self.forward_packed(x, &packed);
        }
        if !self.routed_experts_loaded {
            return Err(Exception::custom(
                "routed expert weights are not resident; enable expert offloading with --experts-dir",
            ));
        }

        let shape = x.shape();
        let batch_seq: i32 = shape[..shape.len() - 1].iter().product();
        let hidden = shape[shape.len() - 1];
        let x_flat = x.reshape(&[batch_seq, hidden]);

        // Compute routing probabilities
        let gate_logits = self.gate.forward(&x_flat);
        let gates = ops::softmax_axis(
            &if gate_logits.dtype() != pmetal_bridge::compat::Dtype::Float32 {
                gate_logits.cast(pmetal_bridge::compat::Dtype::Float32)
            } else {
                gate_logits
            },
            -1,
        );

        // Top-k selection + optional renormalisation via the shared helper.
        // The earlier hand-rolled version here used the sign-flipped form
        // `argpartition(-gates, -k, -1)[..., -k:]` which selects the k SMALLEST
        // experts — an anti-top-k bug (see moe_routing::tests::
        // sign_flipped_argpartition_is_anti_topk). Matches the
        // reference, which uses the positive form.
        let k = self.top_k;
        let (top_indices, top_weights) =
            crate::moe_routing::topk_normalize(&gates, k, self.norm_topk_prob)?;

        // SwitchGLU forward using gather_mm — same as the reference SwitchGLU
        // x_flat: [N, D], indices: [N, k]
        let top_indices_i32 = top_indices.cast(pmetal_bridge::compat::Dtype::Int32);

        // Reshape [N, D] → [N, 1, 1, D] — same as the reference expand_dims(-2, -3).
        // Critical for gather_mm batch dimension semantics — without this,
        // M=N gets preserved in output producing [N, k, N, out] instead of [N, k, 1, out].
        let x_expanded = x_flat.reshape(&[batch_seq, 1, 1, hidden]);

        // Weights are stored as [E, out, in] from checkpoint; gather_mm expects
        // A[..., M, K] @ B[E, K, N], so we transpose the last two dims (like the reference
        // SwitchLinear, which calls weight.swapaxes(-1, -2) at forward time).
        let gate_weight = dequantize_fp8_weight_for_compute(self.switch_mlp_gate_proj.as_ref())?;
        let up_weight = dequantize_fp8_weight_for_compute(self.switch_mlp_up_proj.as_ref())?;
        let gate_w = gate_weight.swap_axes(-1, -2);
        let up_w = up_weight.swap_axes(-1, -2);
        let gate_out = gather_mm(&x_expanded, &gate_w, None, Some(&top_indices_i32), false)?; // [N, k, 1, intermediate]
        let up_out = gather_mm(&x_expanded, &up_w, None, Some(&top_indices_i32), false)?; // [N, k, 1, intermediate]

        let activated = nn::silu(&gate_out).multiply(&up_out);

        // Down projection
        let down_weight = dequantize_fp8_weight_for_compute(self.switch_mlp_down_proj.as_ref())?;
        let down_w = down_weight.swap_axes(-1, -2);
        let down_out = gather_mm(&activated, &down_w, None, Some(&top_indices_i32), false)?; // [N, k, 1, D]

        // squeeze(-2) removes the size-1 dim: [N, k, 1, D] → [N, k, D]
        let down_out = down_out.squeeze_axes(&[-2]);

        // Shared expert forward + gate logit
        let (shared_y, shared_gate_logit) = self.forward_shared_expert_and_gate(&x_flat)?;

        // Combine: weighted expert sum + sigmoid-gated shared expert. The
        // decoder layer adds the residual.
        // Pure MLX ops — all async on GPU, no synchronization barriers
        let result = moe_combine_mlx(
            &down_out,
            &top_weights,
            &shared_y,
            &shared_gate_logit,
            k,
            batch_seq,
        );
        Ok(result?.reshape(shape))
    }

    /// [`forward`](Self::forward) with the routed experts on packed kernels:
    /// the same router and shared expert, the experts through
    /// `native_moe::switch_glu`, which applies each expert's tensor scale to
    /// its product.
    fn forward_packed(
        &mut self,
        x: &Array,
        packed: &PackedRoutedExperts,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch_seq: i32 = shape[..shape.len() - 1].iter().product();
        let hidden = shape[shape.len() - 1];
        let x_flat = x.reshape(&[batch_seq, hidden]);

        let gate_logits = self.gate.forward(&x_flat);
        let gates = ops::softmax_axis(&gate_logits.cast(Dtype::Float32), -1);
        let (top_indices, top_weights) =
            crate::moe_routing::topk_normalize(&gates, self.top_k, self.norm_topk_prob)?;
        let routed = pmetal_bridge::native_moe::switch_glu(
            &x_flat,
            &packed.gate,
            &packed.up,
            &packed.down,
            &top_indices.cast(Dtype::Uint32),
            &top_weights,
            |gate, up| nn::silu(gate).multiply(up),
        );

        let (shared_y, shared_gate_logit) = self.forward_shared_expert_and_gate(&x_flat)?;
        let shared = nn::sigmoid(&shared_gate_logit).multiply(&shared_y);
        Ok(routed.add(&shared).reshape(shape))
    }

    pub fn forward_profiled(
        &mut self,
        x: &Array,
        layer_profile: &mut Qwen3NextLayerProfile,
    ) -> Result<Array, Exception> {
        if self.offload_ctx.is_some() {
            return self.forward_offloaded(x, Some(layer_profile));
        }

        let start = Instant::now();
        let output = self.forward(x)?;
        output.eval();
        layer_profile.push_section("moe_resident", start);
        Ok(output)
    }

    fn route_experts(&mut self, x_flat: &Array) -> Result<(Array, Vec<usize>, i32), Exception> {
        let gate_logits = self.gate.forward(x_flat);
        let gates = ops::softmax_axis(
            &if gate_logits.dtype() != pmetal_bridge::compat::Dtype::Float32 {
                gate_logits.cast(pmetal_bridge::compat::Dtype::Float32)
            } else {
                gate_logits
            },
            -1,
        );

        // Shared top-k helper — see the note on the canonical `forward()` path
        // for why the previous `argpartition(-gates, -k, -1)` form was wrong.
        let k = self.top_k;
        let (top_indices, top_weights) =
            crate::moe_routing::topk_normalize(&gates, k, self.norm_topk_prob)?;

        let top_indices_u32 = top_indices.cast(pmetal_bridge::compat::Dtype::Uint32);
        top_indices_u32.eval();
        let flat_u32: Vec<u32> = top_indices_u32.as_slice().to_vec();
        let expert_indices: Vec<usize> = flat_u32.iter().map(|&i| i as usize).collect();

        Ok((top_weights, expert_indices, k))
    }

    fn ordered_unique_experts(expert_indices: &[usize]) -> (Vec<usize>, HashMap<usize, usize>) {
        let mut unique = Vec::new();
        let mut index_map = HashMap::new();
        for &expert_idx in expert_indices {
            if let std::collections::hash_map::Entry::Vacant(entry) = index_map.entry(expert_idx) {
                let slot = unique.len();
                unique.push(expert_idx);
                entry.insert(slot);
            }
        }
        (unique, index_map)
    }

    /// Offloaded forward pass: routes to top-k experts whose weights are
    /// read on-demand from packed SSD files and dispatched to Metal kernels.
    ///
    /// Two paths:
    /// - **Aligned fast path**: prefetched hits are copied into AlignedBuffers,
    ///   misses are pread into AlignedBuffers, then the Metal expert kernel reads
    ///   component byte offsets directly
    /// - **Legacy**: Vec<u8> buffers → parse_expert_weights → MetalBuffer copies
    fn forward_offloaded(
        &mut self,
        x: &Array,
        mut layer_profile: Option<&mut Qwen3NextLayerProfile>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch_seq: i32 = shape[..shape.len() - 1].iter().product();
        let hidden = shape[shape.len() - 1];
        let x_flat = x.reshape(&[batch_seq, hidden]);

        // The fused offloaded expert path is still single-token at the kernel
        // level. For prompt prefill, process small token windows with exact
        // routing, load each unique expert once per window, and then reuse
        // those loaded experts across the tokens in the window.
        if batch_seq > 1 {
            let prefill_start = Instant::now();
            let ctx = self.offload_ctx.as_ref().unwrap().clone();
            let layer_idx = self.layer_idx;
            let metal_ctx = pmetal_metal::context::MetalContext::global()
                .map_err(|e| Exception::custom(e.to_string()))?;
            let record = &ctx.layout.record;
            let runtime = self.offload_runtime.clone().ok_or_else(|| {
                Exception::custom("forward_offloaded: offload runtime not initialized")
            })?;
            let prefill_window_tokens = self.prefill_expert_window_tokens;

            let mut window_outputs =
                Vec::with_capacity(batch_seq as usize / prefill_window_tokens + 1);

            for window_start in (0..batch_seq as usize).step_by(prefill_window_tokens) {
                let window_end = (window_start + prefill_window_tokens).min(batch_seq as usize);
                let window_len = (window_end - window_start) as i32;
                let window_input = slice_axis(&x_flat, 0, window_start as i32, window_end as i32)
                    .reshape(&[window_len, hidden]);

                let route_start = Instant::now();
                let (window_top_weights, window_expert_indices, k) =
                    self.route_experts(&window_input)?;
                if let Some(profile) = layer_profile.as_deref_mut() {
                    profile.push_section("moe_prefill_route", route_start);
                }
                let k_usize = k as usize;

                let (unique_experts, unique_index_map) =
                    Self::ordered_unique_experts(&window_expert_indices);
                let prefetched = self
                    .prefetcher
                    .as_ref()
                    .map(|p| p.try_get(layer_idx, &unique_experts))
                    .unwrap_or_else(|| (0..unique_experts.len()).map(|_| None).collect());

                let miss_plan = prefetched_miss_plan(&prefetched, &unique_experts);
                let window_output_buffers = window_len as usize * k_usize;
                let use_aligned = runtime.buffer_pool.is_some()
                    && runtime.expert_out_bufs.len() >= window_output_buffers
                    && runtime
                        .buffer_pool
                        .as_ref()
                        .is_some_and(|pool| pool.total_buffers() >= unique_experts.len());
                let mut aligned_release_bufs: Option<
                    Vec<pmetal_metal::expert_buffer::AlignedBuffer>,
                > = None;
                let mut window_down_tokens = Vec::with_capacity(window_len as usize);

                let cmd_buf = if use_aligned {
                    let fused_expert = runtime.fused_expert.as_ref().ok_or_else(|| {
                        Exception::custom("forward_offloaded: fused_expert not initialized")
                    })?;
                    let intermediate = runtime.expert_intermediate.as_ref().ok_or_else(|| {
                        Exception::custom("forward_offloaded: expert_intermediate not initialized")
                    })?;
                    let pool = runtime.buffer_pool.as_ref().unwrap();
                    let mut unique_bufs = acquire_prefetched_aligned_buffers(pool, prefetched)?;

                    let io_start = Instant::now();
                    if let Err(error) = load_missing_experts_into_aligned_buffers(
                        &ctx,
                        layer_idx,
                        &miss_plan,
                        &mut unique_bufs,
                    ) {
                        for buf in unique_bufs.drain(..) {
                            pool.release(buf);
                        }
                        return Err(error);
                    }
                    if let Some(profile) = layer_profile.as_deref_mut() {
                        profile.push_section("moe_prefill_io", io_start);
                    }

                    let encode_submit_start = Instant::now();
                    let cmd_buf = fused_expert
                        .create_command_buffer()
                        .map_err(|e| Exception::custom(e.to_string()))?;
                    for token_offset in 0..window_len as usize {
                        let token_input = select_axis(&window_input, token_offset as i32, 0)
                            .reshape(&[1, hidden]);
                        let token_input_f32 =
                            token_input.cast(pmetal_bridge::compat::Dtype::Float32);
                        token_input_f32.eval();
                        let input_buf = pmetal_mlx::bridge::MlxMetalBridge::copy_as_f32(
                            &metal_ctx,
                            &token_input_f32,
                        )
                        .map_err(|e| Exception::custom(e.to_string()))?;
                        for slot in 0..k_usize {
                            let expert_idx = window_expert_indices[token_offset * k_usize + slot];
                            let unique_slot = *unique_index_map
                                .get(&expert_idx)
                                .expect("unique expert index must exist");
                            let abuf = &unique_bufs[unique_slot];
                            let output_slot = token_offset * k_usize + slot;
                            fused_expert
                                .encode_expert_aligned(
                                    &cmd_buf,
                                    &input_buf,
                                    abuf.metal_buffer(),
                                    record.gate_weight.offset,
                                    record.gate_scales.offset,
                                    record.gate_biases.offset,
                                    record.up_weight.offset,
                                    record.up_scales.offset,
                                    record.up_biases.offset,
                                    record.down_weight.offset,
                                    record.down_scales.offset,
                                    record.down_biases.offset,
                                    &runtime.expert_out_bufs[output_slot],
                                    intermediate,
                                )
                                .map_err(|e| Exception::custom(e.to_string()))?;
                        }
                    }

                    fused_expert
                        .submit(&cmd_buf)
                        .map_err(|e| Exception::custom(e.to_string()))?;
                    if let Some(profile) = layer_profile.as_deref_mut() {
                        profile
                            .push_section("moe_prefill_expert_encode_submit", encode_submit_start);
                    }
                    aligned_release_bufs = Some(unique_bufs);
                    cmd_buf
                } else {
                    let fused_expert = runtime.fused_expert.as_ref().ok_or_else(|| {
                        Exception::custom("forward_offloaded: fused_expert not initialized")
                    })?;
                    let intermediate = runtime.expert_intermediate.as_ref().ok_or_else(|| {
                        Exception::custom("forward_offloaded: expert_intermediate not initialized")
                    })?;
                    let miss_ids: Vec<usize> = miss_plan
                        .iter()
                        .map(|(_, expert_idx)| *expert_idx)
                        .collect();
                    let io_start = Instant::now();
                    let miss_bufs = if !miss_ids.is_empty() {
                        ctx.read_experts(layer_idx, &miss_ids).map_err(|e| {
                            Exception::custom(format!("expert pread layer {layer_idx}: {e}"))
                        })?
                    } else {
                        vec![]
                    };

                    let unique_bufs = materialize_prefetched_raw_bytes(
                        prefetched,
                        ctx.layout.expert_size,
                        miss_bufs,
                    );
                    if let Some(profile) = layer_profile.as_deref_mut() {
                        profile.push_section("moe_prefill_io", io_start);
                    }

                    let encode_submit_start = Instant::now();
                    let cmd_buf = fused_expert
                        .create_command_buffer()
                        .map_err(|e| Exception::custom(e.to_string()))?;
                    for token_offset in 0..window_len as usize {
                        let token_input = select_axis(&window_input, token_offset as i32, 0)
                            .reshape(&[1, hidden]);
                        let token_input_f32 =
                            token_input.cast(pmetal_bridge::compat::Dtype::Float32);
                        token_input_f32.eval();
                        let input_buf = pmetal_mlx::bridge::MlxMetalBridge::copy_as_f32(
                            &metal_ctx,
                            &token_input_f32,
                        )
                        .map_err(|e| Exception::custom(e.to_string()))?;
                        for slot in 0..k_usize {
                            let expert_idx = window_expert_indices[token_offset * k_usize + slot];
                            let unique_slot = *unique_index_map
                                .get(&expert_idx)
                                .expect("unique expert index must exist");
                            let output_slot = token_offset * k_usize + slot;
                            let weights = crate::expert_dequant::parse_expert_weights(
                                &unique_bufs[unique_slot],
                                record,
                                &metal_ctx,
                            )
                            .map_err(Exception::custom)?;

                            fused_expert
                                .encode_into(
                                    &cmd_buf,
                                    &input_buf,
                                    &weights,
                                    &runtime.expert_out_bufs[output_slot],
                                    intermediate,
                                )
                                .map_err(|e| Exception::custom(e.to_string()))?;
                        }
                    }

                    fused_expert
                        .submit(&cmd_buf)
                        .map_err(|e| Exception::custom(e.to_string()))?;
                    if let Some(profile) = layer_profile.as_deref_mut() {
                        profile
                            .push_section("moe_prefill_expert_encode_submit", encode_submit_start);
                    }
                    cmd_buf
                };

                let shared_start = Instant::now();
                let (shared_y, shared_gate_logit) =
                    self.forward_shared_expert_and_gate(&window_input)?;
                shared_y.eval();
                shared_gate_logit.eval();
                if let Some(profile) = layer_profile.as_deref_mut() {
                    profile.push_section("moe_prefill_shared", shared_start);
                }

                let wait_start = Instant::now();
                let fused_expert = runtime.fused_expert.as_ref().ok_or_else(|| {
                    Exception::custom("forward_offloaded: fused_expert not initialized")
                })?;
                fused_expert
                    .wait_for_completion(&cmd_buf)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                if let Some(profile) = layer_profile.as_deref_mut() {
                    profile.push_section("moe_prefill_expert_wait", wait_start);
                }

                if let Some(unique_bufs) = aligned_release_bufs {
                    let pool = runtime.buffer_pool.as_ref().unwrap();
                    for buf in unique_bufs {
                        pool.release(buf);
                    }
                }

                let output_wrap_start = Instant::now();
                for token_offset in 0..window_len as usize {
                    let mut expert_arrays: Vec<Array> = Vec::with_capacity(k_usize);
                    for slot in 0..k_usize {
                        let output_slot = token_offset * k_usize + slot;
                        let slice = runtime.expert_out_bufs[output_slot].as_slice();
                        expert_arrays.push(Array::from_slice(slice, &[1, hidden]));
                    }
                    let down_out = ops::stack_axis(&expert_arrays, 1).reshape(&[k, hidden]);
                    window_down_tokens.push(down_out);
                }
                if let Some(profile) = layer_profile.as_deref_mut() {
                    profile.push_section("moe_prefill_output_wrap", output_wrap_start);
                }

                let combine_start = Instant::now();
                let window_down_out =
                    ops::stack_axis(&window_down_tokens, 0).reshape(&[window_len, k, hidden]);
                let window_output = moe_combine_mlx(
                    &window_down_out,
                    &window_top_weights,
                    &shared_y,
                    &shared_gate_logit,
                    k,
                    window_len,
                )?
                .reshape(&[window_len, hidden]);
                if let Some(profile) = layer_profile.as_deref_mut() {
                    profile.push_section("moe_prefill_combine", combine_start);
                }
                window_outputs.push(window_output);
            }
            let window_refs: Vec<&Array> = window_outputs.iter().collect();
            let stacked = ops::concatenate_axis(&window_refs, 0).reshape(shape);
            stacked.eval();
            if let Some(profile) = layer_profile.as_deref_mut() {
                profile.push_section("moe_prefill_windowed", prefill_start);
            }
            return Ok(stacked);
        }

        // ---- Routing (identical to resident path) ----
        let routing_start = Instant::now();
        let (top_weights, expert_indices, k) = self.route_experts(&x_flat)?;
        if let Some(profile) = layer_profile.as_deref_mut() {
            profile.push_section("moe_route", routing_start);
        }

        // ---- Load expert weights from SSD ----
        let ctx = self.offload_ctx.as_ref().unwrap().clone();
        let layer_idx = self.layer_idx;

        #[cfg(not(unix))]
        return Err(Exception::custom(
            "expert offloading requires a Unix platform (pread is not available)",
        ));

        #[cfg(unix)]
        {
            // ---- Prefetch-aware expert loading ----
            let io_start = std::time::Instant::now();
            let runtime = self.offload_runtime.clone().ok_or_else(|| {
                Exception::custom("forward_offloaded: offload runtime not initialized")
            })?;

            let prefetched = self
                .prefetcher
                .as_ref()
                .map(|p| p.try_get(layer_idx, &expert_indices))
                .unwrap_or_else(|| (0..expert_indices.len()).map(|_| None).collect());

            let miss_plan = prefetched_miss_plan(&prefetched, &expert_indices);
            let miss_ids: Vec<usize> = miss_plan
                .iter()
                .map(|(_, expert_idx)| *expert_idx)
                .collect();

            let prefetch_hits = prefetched.iter().filter(|b| b.is_some()).count();

            // ---- Dispatch: zero-copy aligned path or legacy copy path ----
            let metal_ctx = pmetal_metal::context::MetalContext::global()
                .map_err(|e| Exception::custom(e.to_string()))?;
            let record = &ctx.layout.record;

            // x_flat to Metal buffer (tiny for T=1 decode)
            let x_flat_f32 = x_flat.cast(pmetal_bridge::compat::Dtype::Float32);
            x_flat_f32.eval();
            let input_buf =
                pmetal_mlx::bridge::MlxMetalBridge::copy_as_f32(&metal_ctx, &x_flat_f32)
                    .map_err(|e| Exception::custom(e.to_string()))?;

            let use_aligned = runtime.buffer_pool.is_some()
                && runtime.expert_out_bufs.len() >= expert_indices.len();
            let mut aligned_release_bufs: Option<Vec<pmetal_metal::expert_buffer::AlignedBuffer>> =
                None;
            let cmd_buf = if use_aligned {
                let fused_expert = runtime.fused_expert.as_ref().ok_or_else(|| {
                    Exception::custom("forward_offloaded: fused_expert not initialized")
                })?;
                let intermediate = runtime.expert_intermediate.as_ref().ok_or_else(|| {
                    Exception::custom("forward_offloaded: expert_intermediate not initialized")
                })?;
                // ---- ALIGNED FAST PATH ----
                // Prefetched aligned hits reuse their pooled buffers directly.
                let pool = runtime.buffer_pool.as_ref().unwrap();
                let mut aligned_bufs = acquire_prefetched_aligned_buffers(pool, prefetched)?;

                let aligned_load = load_missing_experts_into_aligned_buffers(
                    &ctx,
                    layer_idx,
                    &miss_plan,
                    &mut aligned_bufs,
                );

                if let Err(error) = aligned_load {
                    for buf in aligned_bufs.drain(..) {
                        pool.release(buf);
                    }
                    return Err(error);
                }
                if let Some(profile) = layer_profile.as_deref_mut() {
                    profile.push_section("moe_expert_io", io_start);
                }

                let io_us = io_start.elapsed().as_micros();
                tracing::trace!(
                    layer = layer_idx,
                    io_ms = io_us as f64 / 1000.0,
                    path = if prefetch_hits > 0 {
                        "aligned-mixed"
                    } else {
                        "aligned-direct"
                    },
                    experts = expert_indices.len(),
                    hits = prefetch_hits,
                    misses = miss_ids.len(),
                    "offloaded MoE I/O"
                );

                let encode_submit_start = Instant::now();
                let cmd_buf = fused_expert
                    .create_command_buffer()
                    .map_err(|e| Exception::custom(e.to_string()))?;
                for (i, abuf) in aligned_bufs.iter().enumerate() {
                    fused_expert
                        .encode_expert_aligned(
                            &cmd_buf,
                            &input_buf,
                            abuf.metal_buffer(),
                            record.gate_weight.offset,
                            record.gate_scales.offset,
                            record.gate_biases.offset,
                            record.up_weight.offset,
                            record.up_scales.offset,
                            record.up_biases.offset,
                            record.down_weight.offset,
                            record.down_scales.offset,
                            record.down_biases.offset,
                            &runtime.expert_out_bufs[i],
                            intermediate,
                        )
                        .map_err(|e| Exception::custom(e.to_string()))?;
                }
                fused_expert
                    .submit(&cmd_buf)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                if let Some(profile) = layer_profile.as_deref_mut() {
                    profile.push_section("moe_expert_encode_submit", encode_submit_start);
                }
                aligned_release_bufs = Some(aligned_bufs);
                cmd_buf
            } else {
                let fused_expert = runtime.fused_expert.as_ref().ok_or_else(|| {
                    Exception::custom("forward_offloaded: fused_expert not initialized")
                })?;
                let intermediate = runtime.expert_intermediate.as_ref().ok_or_else(|| {
                    Exception::custom("forward_offloaded: expert_intermediate not initialized")
                })?;
                // ---- LEGACY COPY PATH ----
                // Used when the aligned buffer pool is unavailable.
                let cmd_buf = fused_expert
                    .create_command_buffer()
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let miss_bufs = if !miss_ids.is_empty() {
                    ctx.read_experts(layer_idx, &miss_ids).map_err(|e| {
                        Exception::custom(format!("expert pread layer {layer_idx}: {e}"))
                    })?
                } else {
                    vec![]
                };

                let expert_bufs =
                    materialize_prefetched_raw_bytes(prefetched, ctx.layout.expert_size, miss_bufs);

                let io_us = io_start.elapsed().as_micros();
                tracing::trace!(
                    layer = layer_idx,
                    io_ms = io_us as f64 / 1000.0,
                    path = "legacy",
                    hits = prefetch_hits,
                    misses = miss_ids.len(),
                    "offloaded MoE I/O"
                );
                if let Some(profile) = layer_profile.as_deref_mut() {
                    profile.push_section("moe_expert_io", io_start);
                }

                let encode_submit_start = Instant::now();
                for (i, raw_bytes) in expert_bufs.iter().enumerate() {
                    let weights =
                        crate::expert_dequant::parse_expert_weights(raw_bytes, record, &metal_ctx)
                            .map_err(|e| Exception::custom(e))?;

                    fused_expert
                        .encode_into(
                            &cmd_buf,
                            &input_buf,
                            &weights,
                            &runtime.expert_out_bufs[i],
                            intermediate,
                        )
                        .map_err(|e| Exception::custom(e.to_string()))?;
                }

                fused_expert
                    .submit(&cmd_buf)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                if let Some(profile) = layer_profile.as_deref_mut() {
                    profile.push_section("moe_expert_encode_submit", encode_submit_start);
                }
                cmd_buf
            };
            let shared_start = Instant::now();
            let (shared_y, shared_gate_logit) = self.forward_shared_expert_and_gate(&x_flat)?;
            shared_y.eval();
            shared_gate_logit.eval();
            if let Some(profile) = layer_profile.as_deref_mut() {
                profile.push_section("moe_shared", shared_start);
            }

            let wait_start = Instant::now();
            let fused_expert = runtime.fused_expert.as_ref().ok_or_else(|| {
                Exception::custom("forward_offloaded: fused_expert not initialized")
            })?;
            fused_expert
                .wait_for_completion(&cmd_buf)
                .map_err(|e| Exception::custom(e.to_string()))?;
            if let Some(profile) = layer_profile.as_deref_mut() {
                profile.push_section("moe_expert_wait", wait_start);
                profile.push_section("moe_expert_gpu", wait_start);
            }

            if let Some(aligned_bufs) = aligned_release_bufs {
                let pool = runtime.buffer_pool.as_ref().unwrap();
                for buf in aligned_bufs {
                    pool.release(buf);
                }
            }

            // ---- Combine: weighted sum + shared expert ----
            let combine_start = Instant::now();
            // Convert Metal output buffers to MLX arrays (zero-copy)
            let output_wrap_start = Instant::now();
            let k_usize = k as usize;
            let mut expert_arrays: Vec<Array> = Vec::with_capacity(k_usize);
            for i in 0..expert_indices.len().min(runtime.expert_out_bufs.len()) {
                let slice = runtime.expert_out_bufs[i].as_slice();
                expert_arrays.push(Array::from_slice(slice, &[1, hidden]));
            }
            let down_out = ops::stack_axis(&expert_arrays, 1);
            let down_out = down_out.reshape(&[batch_seq, k, hidden]);
            if let Some(profile) = layer_profile.as_deref_mut() {
                profile.push_section("moe_output_wrap", output_wrap_start);
            }

            let result = moe_combine_mlx(
                &down_out,
                &top_weights,
                &shared_y,
                &shared_gate_logit,
                k,
                batch_seq,
            )?;
            let reshaped = result.reshape(shape);
            reshaped.eval();
            if let Some(profile) = layer_profile {
                profile.push_section("moe_combine", combine_start);
            }
            Ok(reshaped)
        }
    }
}

fn prefetched_miss_plan<T>(
    prefetched: &[Option<T>],
    expert_indices: &[usize],
) -> Vec<(usize, usize)> {
    prefetched
        .iter()
        .enumerate()
        .filter_map(|(slot_idx, hit)| {
            hit.is_none()
                .then_some((slot_idx, expert_indices[slot_idx]))
        })
        .collect()
}

fn acquire_prefetched_aligned_buffers(
    pool: &Arc<ExpertBufferPool>,
    prefetched: Vec<Option<PrefetchedExpert>>,
) -> Result<Vec<pmetal_metal::expert_buffer::AlignedBuffer>, Exception> {
    let mut aligned_bufs = Vec::with_capacity(prefetched.len());
    for hit in prefetched {
        let aligned = match hit {
            Some(PrefetchedExpert::Aligned(buf)) => buf.into_inner(),
            Some(PrefetchedExpert::Raw(raw_bytes)) => {
                let mut aligned = pool.acquire_blocking();
                copy_prefetched_expert_into_aligned(&mut aligned, &raw_bytes)?;
                aligned
            }
            None => pool.acquire_blocking(),
        };
        aligned_bufs.push(aligned);
    }
    Ok(aligned_bufs)
}

fn materialize_prefetched_raw_bytes(
    prefetched: Vec<Option<PrefetchedExpert>>,
    expert_size: usize,
    miss_bufs: Vec<Vec<u8>>,
) -> Vec<Arc<Vec<u8>>> {
    let mut expert_bufs = Vec::with_capacity(prefetched.len());
    let mut miss_iter = miss_bufs.into_iter().map(Arc::new);
    for hit in prefetched {
        let buf = match hit {
            Some(PrefetchedExpert::Raw(buf)) => Arc::new(buf),
            Some(PrefetchedExpert::Aligned(buf)) => Arc::new(buf.to_vec(expert_size)),
            None => miss_iter.next().unwrap(),
        };
        expert_bufs.push(buf);
    }
    expert_bufs
}

fn copy_prefetched_expert_into_aligned(
    aligned: &mut pmetal_metal::expert_buffer::AlignedBuffer,
    raw_bytes: &[u8],
) -> Result<(), Exception> {
    aligned
        .write_prefix(raw_bytes)
        .map_err(|e| Exception::custom(e.to_string()))?;
    Ok(())
}

#[cfg(unix)]
fn load_missing_experts_into_aligned_buffers(
    ctx: &ExpertOffloadContext,
    layer_idx: usize,
    miss_plan: &[(usize, usize)],
    aligned_bufs: &mut [pmetal_metal::expert_buffer::AlignedBuffer],
) -> Result<(), Exception> {
    ctx.read_experts_aligned_with_plan(layer_idx, miss_plan, aligned_bufs)
        .map_err(|e| Exception::custom(format!("expert aligned pread layer {layer_idx}: {e}")))
}

// ============================================================================
// Feed-forward enum (Dense MLP or MoE)
// ============================================================================

#[derive(Debug)]
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
        match self {
            Self::Dense(m) => m.forward(x),
            Self::MoE(m) => m.forward(x),
        }
    }

    pub fn forward_profiled(
        &mut self,
        x: &Array,
        layer_profile: &mut Qwen3NextLayerProfile,
    ) -> Result<Array, Exception> {
        match self {
            Self::Dense(m) => {
                let start = Instant::now();
                let output = m.forward(x)?;
                output.eval();
                layer_profile.push_section("mlp_dense", start);
                Ok(output)
            }
            Self::MoE(m) => m.forward_profiled(x, layer_profile),
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
        let normed = self.input_layernorm.forward(x);
        let r = if self.is_linear {
            self.linear_attn
                .as_mut()
                .expect("linear_attn must be Some for linear layers")
                .forward(&normed, mask, mamba_cache)?
        } else {
            self.self_attn
                .as_mut()
                .expect("self_attn must be Some for attention layers")
                .forward(&normed, mask, kv_cache, positions)?
        };
        let h = x.add(&r);
        let mlp_in = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&mlp_in)?;
        Ok(h.add(&mlp_out))
    }

    /// Speculative-verify forward: records GDN verify inputs into
    /// `capture` for every linear-attention layer. Full-attention layers
    /// (every 4th, see `Qwen3NextConfig::is_linear_layer`) run the plain
    /// [`forward`] path — their KV cache rollback is just `KVCache::rollback`
    /// and needs no extra per-token state.
    pub fn forward_with_capture(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
        mamba_cache: Option<&mut MambaCacheEntry>,
        layer_idx: usize,
        capture: &mut pmetal_mlx::speculative::SpecCapture,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x);
        let r = if self.is_linear {
            self.linear_attn
                .as_mut()
                .expect("linear_attn must be Some for linear layers")
                .forward_with_capture(&normed, mask, mamba_cache, layer_idx, capture)?
        } else {
            self.self_attn
                .as_mut()
                .expect("self_attn must be Some for attention layers")
                .forward(&normed, mask, kv_cache, positions)?
        };
        let h = x.add(&r);
        let mlp_in = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&mlp_in)?;
        Ok(h.add(&mlp_out))
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

        let normed = profile_array_section(&mut profile, "input_layernorm", || {
            self.input_layernorm.forward(x)
        });
        let r = if self.is_linear {
            self.linear_attn
                .as_mut()
                .expect("linear_attn must be Some for linear layers")
                .forward_profiled(&normed, mask, mamba_cache, &mut profile)?
        } else {
            self.self_attn
                .as_mut()
                .expect("self_attn must be Some for attention layers")
                .forward_profiled(&normed, mask, kv_cache, positions, &mut profile)?
        };
        let h = profile_array_section(&mut profile, "attn_residual", || x.add(&r));
        let mlp_in = profile_array_section(&mut profile, "post_attention_layernorm", || {
            self.post_attention_layernorm.forward(&h)
        });
        let mlp_start = Instant::now();
        let mlp_out = self.mlp.forward_profiled(&mlp_in, &mut profile)?;
        let mlp_out_eval = mlp_out.clone();
        mlp_out_eval.eval();
        profile.push_section("mlp", mlp_start);
        let out = profile_array_section(&mut profile, "mlp_residual", || h.add(&mlp_out));
        profile.total_us = layer_start.elapsed().as_micros() as u64;
        Ok((out, profile))
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
    /// Shared expert offload context (set via `enable_expert_offloading`).
    pub offload_ctx: Option<Arc<ExpertOffloadContext>>,
    /// Pre-gated expert prediction engine (set via `enable_expert_offloading`).
    pub prefetcher: Option<Arc<ExpertPrefetcher>>,
    /// Shared runtime for offloaded sparse experts.
    offload_runtime: Option<Arc<Qwen3NextOffloadRuntime>>,
    /// Pre-computed indices of MoE layers for fast next-MoE lookup during prefetch.
    pub moe_layer_indices: Vec<usize>,
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

        let moe_layer_indices: Vec<usize> = layers
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l.mlp, Qwen3NextFeedForward::MoE(_)))
            .map(|(i, _)| i)
            .collect();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            full_attention_interval: config.full_attention_interval,
            offload_ctx: None,
            prefetcher: None,
            offload_runtime: None,
            moe_layer_indices,
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

    /// Forward pass with optional hidden-state capture for speculative
    /// decoding.
    ///
    /// Behaves identically to [`forward_with_cache`] when `capture` is
    /// `None`. When a capture buffer is supplied, the post-layer hidden
    /// state is cloned into `capture.hidden_states` for every layer index in
    /// `capture.requested_hidden_layers`, without otherwise perturbing the
    /// loop.
    ///
    /// GDN verify-input capture is intentionally *not* populated here. The
    /// hybrid-attention rollback path needs the linear-layer `k,v,g,β` and
    /// `conv_input` captured from inside the mixer; wiring that would touch
    /// the compiled-decode hot path. Hidden-state capture alone is enough
    /// for the Qwen3 DFlash path, and the Qwen3.5 path will add GDN
    /// capture as a follow-up that plumbs through the mixer.
    pub fn forward_with_cache_and_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut kv_cache: Option<&mut KVCache>,
        positions: Option<&Array>,
        mut mamba_cache: Option<&mut MambaCache>,
        mut capture: Option<&mut pmetal_mlx::speculative::SpecCapture>,
    ) -> Result<Array, Exception> {
        let mut hidden = Module::forward(&mut self.embed_tokens, input_ids)?;

        // Create separate masks for full attention vs GDN layers (matching MLX reference):
        // - Full attention: uses the causal mask from the caller (or None for cached decode)
        // - GDN (linear attention): uses None (no left-padding in our MambaCache impl)
        // Passing a causal mask [1,1,T,T] to GDN layers causes garbage output because
        // GDN interprets the mask as a token-validity boolean [B,T], not attention weights.
        let fa_mask = mask;
        let ssm_mask: Option<&Array> = None;
        let prefill_like = hidden.dim(1) > 1;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Kick off prefetch for the next MoE layer using the current
            // layer input only for prompt prefill, where overlapping the next
            // sparse layer's reads is worth the extra predictor work. For
            // single-token decode, keep prefetch after the layer so we do not
            // serialize the hot path before the current layer runs.
            if prefill_like
                && let (Some(prefetcher), Some(ctx)) = (&self.prefetcher, &self.offload_ctx)
            {
                let search = self.moe_layer_indices.partition_point(|&i| i <= layer_idx);
                if search < self.moe_layer_indices.len() {
                    prefetcher.predict_and_prefetch(self.moe_layer_indices[search], &hidden, ctx);
                }
            }

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

            let layer_mask = if layer.is_linear { ssm_mask } else { fa_mask };

            // When a capture buffer is supplied we route the GDN layers
            // through `forward_with_capture` so their per-token verify
            // inputs end up in `capture.gdn_inputs`. Full-attention
            // layers do not need the capture and just reuse `forward`.
            hidden = if let Some(buf) = capture.as_deref_mut() {
                if layer.is_linear {
                    layer.forward_with_capture(
                        &hidden, layer_mask, kv, positions, mamba, layer_idx, buf,
                    )?
                } else {
                    layer.forward(&hidden, layer_mask, kv, positions, mamba)?
                }
            } else {
                layer.forward(&hidden, layer_mask, kv, positions, mamba)?
            };

            if let Some(buf) = capture.as_deref_mut()
                && buf.wants_hidden_for(layer_idx)
            {
                buf.record_hidden(layer_idx, hidden.clone());
            }

            if !prefill_like
                && let (Some(prefetcher), Some(ctx)) = (&self.prefetcher, &self.offload_ctx)
            {
                let search = self.moe_layer_indices.partition_point(|&i| i <= layer_idx);
                if search < self.moe_layer_indices.len() {
                    prefetcher.predict_and_prefetch(self.moe_layer_indices[search], &hidden, ctx);
                }
            }
        }

        let hidden = self.norm.forward(&hidden);
        if prefill_like && let Some(prefetcher) = &self.prefetcher {
            prefetcher.reset_pending();
        }
        Ok(hidden)
    }

    pub fn forward_with_cache_profiled(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut kv_cache: Option<&mut KVCache>,
        mut mamba_cache: Option<&mut MambaCache>,
        phase: impl Into<String>,
    ) -> Result<(Array, Qwen3NextForwardProfile), Exception> {
        let model_start = Instant::now();
        let mut profile = Qwen3NextForwardProfile::new(phase, input_ids.shape().to_vec());

        let embed_start = Instant::now();
        let mut hidden = Module::forward(&mut self.embed_tokens, input_ids)?;
        hidden.eval();
        profile.embedding_us = embed_start.elapsed().as_micros() as u64;

        let fa_mask = mask;
        let ssm_mask: Option<&Array> = None;
        let prefill_like = hidden.dim(1) > 1;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            if prefill_like
                && let (Some(prefetcher), Some(ctx)) = (&self.prefetcher, &self.offload_ctx)
            {
                let search = self.moe_layer_indices.partition_point(|&i| i <= layer_idx);
                if search < self.moe_layer_indices.len() {
                    prefetcher.predict_and_prefetch(self.moe_layer_indices[search], &hidden, ctx);
                }
            }

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
            let layer_mask = if layer.is_linear { ssm_mask } else { fa_mask };
            let (next_hidden, layer_profile) =
                layer.forward_profiled(&hidden, layer_mask, kv, None, mamba, layer_idx)?;
            hidden = next_hidden;
            profile.layers.push(layer_profile);

            if !prefill_like
                && let (Some(prefetcher), Some(ctx)) = (&self.prefetcher, &self.offload_ctx)
            {
                let search = self.moe_layer_indices.partition_point(|&i| i <= layer_idx);
                if search < self.moe_layer_indices.len() {
                    prefetcher.predict_and_prefetch(self.moe_layer_indices[search], &hidden, ctx);
                }
            }
        }

        let final_norm_start = Instant::now();
        let hidden = self.norm.forward(&hidden);
        hidden.eval();
        profile.final_norm_us = final_norm_start.elapsed().as_micros() as u64;
        if prefill_like && let Some(prefetcher) = &self.prefetcher {
            prefetcher.reset_pending();
        }
        profile.total_us = model_start.elapsed().as_micros() as u64;
        Ok((hidden, profile))
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
    /// InlineArray weights for zero-overhead decode. Built lazily on first decode call.
    pub inline_weights: Option<super::qwen3_next_inline::InlineModelWeights>,
    /// InlineArray decode state, one per sequence being decoded, keyed by the
    /// caller's `MambaCache`. It lives across decode steps (no conversion per
    /// step) and takes over from the caller's caches, which the inline path
    /// doesn't update. Keeping a single one made every sequence after the
    /// first decode against the first one's state.
    pub inline_caches: Vec<(SequenceKey, super::qwen3_next_inline::InlineCache)>,
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
            inline_weights: None,
            inline_caches: Vec::new(),
        })
    }

    pub fn requires_expert_offloading(&self) -> bool {
        self.model.layers.iter().any(|layer| {
            matches!(
                &layer.mlp,
                Qwen3NextFeedForward::MoE(block)
                    if !block.routed_experts_loaded && block.offload_ctx.is_none()
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
        // Decode (T=1): use InlineArray path for zero-overhead graph build
        if mask.is_none() && input_ids.dim(1) == 1 && kv_cache.is_some() && mamba_cache.is_some() {
            // Lazily build InlineArray weights on first decode
            if self.inline_weights.is_none() {
                eprintln!("[INLINE] Building InlineArray weights...");
                match super::qwen3_next_inline::InlineModelWeights::from_model(self) {
                    Ok(w) => {
                        eprintln!(
                            "[INLINE] InlineArray weights ready ({} layers)",
                            w.layers.len()
                        );
                        self.inline_weights = Some(w);
                    }
                    Err(e) => {
                        eprintln!(
                            "[INLINE] Failed to build InlineArray weights: {e}, falling back"
                        );
                    }
                }
            }
            if let (Some(weights), Some(kv), Some(mb)) = (
                self.inline_weights.as_ref(),
                kv_cache.as_ref(),
                mamba_cache.as_ref(),
            ) {
                // This sequence's inline state, bootstrapped from its caches
                // on its first decode step. Sequences that have finished
                // (their cache dropped or reset) give theirs up.
                self.inline_caches.retain(|(key, _)| key.is_live());
                let slot = match self
                    .inline_caches
                    .iter()
                    .position(|(key, _)| key.belongs_to(mb))
                {
                    Some(slot) => slot,
                    None => {
                        tracing::debug!("[INLINE] Bootstrapping InlineCache from mlx-rs caches");
                        self.inline_caches.push((
                            mb.sequence_key(),
                            super::qwen3_next_inline::InlineCache::from_caches(
                                kv,
                                mb,
                                &weights.layers,
                            ),
                        ));
                        self.inline_caches.len() - 1
                    }
                };

                // Pure InlineArray decode. From here on the InlineCache owns
                // the decode state; `kv_cache` / `mamba_cache` are not updated.
                let token = super::qwen3_next_inline::ia_from_array(input_ids);
                let logits = super::qwen3_next_inline::inline_decode_step_pure(
                    weights,
                    &token,
                    &mut self.inline_caches[slot].1,
                );
                return Ok(super::qwen3_next_inline::ia_to_array(&logits));
            }
        }
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

    /// Forward pass that returns logits AND pre-lm-head hidden states for
    /// the tapped layers — the target side of a DFlash speculative verify
    /// step for the hybrid-attention Qwen3.5 stack.
    ///
    /// This path intentionally bypasses the InlineArray decode closure used
    /// by [`forward_with_cache`] for `T = 1`: verify always runs with
    /// `T > 1` (multiple draft tokens), so the standard mlx-rs path is the
    /// correct target.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
        capture: &mut pmetal_mlx::speculative::SpecCapture,
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
        capture: &mut pmetal_mlx::speculative::SpecCapture,
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

    /// Enable SSD-offloaded MoE inference with expert prefetching.
    ///
    /// 1. Opens packed expert files from `experts_dir`
    /// 2. Extracts gate weights to build the prefetch predictor
    /// 3. Wires offload context and prefetcher into each MoE layer
    /// 4. Zeros stacked expert weight arrays to reclaim GPU memory
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
        let offload = attach_expert_offload(
            experts_dir,
            &mut blocks,
            self.config.num_experts as usize,
            self.config.hidden_size as usize,
            self.config.num_experts_per_tok as usize,
        )?;

        // Store on model for use in forward_with_cache
        self.model.offload_ctx = Some(offload.ctx);
        self.model.offload_runtime = Some(offload.runtime);
        self.model.prefetcher = Some(offload.prefetcher);
        Ok(())
    }

    /// Get prefetch hit/miss statistics (if offloading is enabled).
    pub fn prefetch_stats(&self) -> Option<crate::expert_prefetch::PrefetchStats> {
        self.model.prefetcher.as_ref().map(|p| p.stats())
    }

    /// Reset prefetch hit/miss statistics.
    pub fn reset_prefetch_stats(&self) {
        if let Some(prefetcher) = self.model.prefetcher.as_ref() {
            prefetcher.reset_stats();
        }
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
        let new_key = pmetal_bridge::qwen3_native::family::canonical_checkpoint_key(&old_key);
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
        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = mlp.forward(&x).unwrap();
        assert_eq!(output.shape(), &[1, 4, 32]);
    }

    #[test]
    #[serial]
    fn test_mlp_forward_matches_manual_swiglu() {
        let mut mlp = Qwen3NextMLP::new(32, 64).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 32],
            pmetal_bridge::compat::Dtype::Float32,
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
        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );

        // Without gate
        let out1 = norm.forward(&x, None).unwrap();
        assert_eq!(out1.shape(), &[1, 4, 32]);

        // With gate
        let gate = pmetal_bridge::compat::random::normal(
            &[1, 4, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let out2 = norm.forward(&x, Some(&gate)).unwrap();
        assert_eq!(out2.shape(), &[1, 4, 32]);
    }

    #[test]
    #[serial]
    fn test_attention_output_shape() {
        let config = tiny_config();
        let mut attn = Qwen3NextAttention::new(&config).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = attn.forward(&x, None, None, None).unwrap();
        assert_eq!(output.shape(), &[1, 4, 32]);
    }

    #[test]
    fn test_attention_mask_type_selection() {
        assert_eq!(
            Qwen3NextAttention::mask_type_for_call(None, false, 4),
            AttentionMaskType::Causal
        );
        assert_eq!(
            Qwen3NextAttention::mask_type_for_call(None, true, 1),
            AttentionMaskType::Causal
        );

        let custom_mask =
            pmetal_bridge::compat::ops::ones(&[1, 1, 1, 1], pmetal_bridge::compat::Dtype::Bool);
        assert_eq!(
            Qwen3NextAttention::mask_type_for_call(Some(&custom_mask), false, 4),
            AttentionMaskType::None
        );
        assert_eq!(
            Qwen3NextAttention::mask_type_for_call(Some(&custom_mask), true, 1),
            AttentionMaskType::None
        );
    }

    #[test]
    #[serial]
    fn test_gdn_output_shape() {
        let config = tiny_config();
        let mut gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 32],
            pmetal_bridge::compat::Dtype::Float32,
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
        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );
        pmetal_bridge::clear_last_error();
        let (loss, grads) = pmetal_bridge::compat::nn::value_and_grad_explicit(
            |arrays: &[Array]| gdn.forward(&arrays[0], None, None).unwrap().sum(None),
            std::slice::from_ref(&x),
            &[],
        )
        .unwrap();
        pmetal_bridge::check_last_error().expect("no bridge op failed under value_and_grad");

        assert!(loss.item::<f32>().is_finite());
        // Evaluate before reading: as_slice on a lazy array reads a null pointer.
        grads[0].eval();
        pmetal_bridge::check_last_error().expect("the gradient evaluates");
        let dx = grads[0].as_slice::<f32>();
        assert!(dx.iter().all(|g| g.is_finite()), "finite input gradient");
        assert!(dx.iter().any(|g| *g != 0.0), "non-zero input gradient");
    }

    #[test]
    fn test_prefetched_miss_plan_tracks_slots_and_experts() {
        let prefetched = vec![Some(vec![1u8; 4]), None, Some(vec![2u8; 4]), None];
        let expert_indices = vec![2usize, 3, 7, 11];
        let miss_plan = prefetched_miss_plan(&prefetched, &expert_indices);
        assert_eq!(miss_plan, vec![(1, 3), (3, 11)]);
    }

    #[test]
    fn test_sparse_moe_required_pool_buffers_covers_prefill_window() {
        assert_eq!(Qwen3NextSparseMoeBlock::required_pool_buffers(0, 8), 1);
        assert_eq!(Qwen3NextSparseMoeBlock::required_pool_buffers(4, 8), 32);
        assert_eq!(Qwen3NextSparseMoeBlock::required_pool_buffers(8, 8), 64);
        assert_eq!(Qwen3NextSparseMoeBlock::required_pool_buffers(4, 16), 64);
    }

    #[test]
    fn test_sparse_moe_required_output_buffers_covers_prefill_window() {
        assert_eq!(Qwen3NextSparseMoeBlock::required_output_buffers(0, 8), 1);
        assert_eq!(Qwen3NextSparseMoeBlock::required_output_buffers(4, 8), 32);
        assert_eq!(Qwen3NextSparseMoeBlock::required_output_buffers(8, 8), 64);
        assert_eq!(Qwen3NextSparseMoeBlock::required_output_buffers(4, 16), 64);
    }

    #[test]
    fn test_prefill_expert_window_tokens_is_sanitized() {
        assert_eq!(
            Qwen3NextSparseMoeBlock::sanitize_prefill_expert_window_tokens(0),
            1
        );
        assert_eq!(
            Qwen3NextSparseMoeBlock::sanitize_prefill_expert_window_tokens(8),
            8
        );
        assert_eq!(
            Qwen3NextSparseMoeBlock::sanitize_prefill_expert_window_tokens(64),
            32
        );
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
            .cast(pmetal_bridge::compat::Dtype::Float32)
            .unwrap();
        // cast() is lazy; evaluate before reading via as_slice() to avoid a
        // null-dereference inside Buffer::raw_ptr on an unevaluated array.
        shifted.eval();
        let shifted_vec: Vec<f32> = shifted.as_slice().to_vec();
        assert_eq!(shifted_vec, vec![1.25, 0.5, 2.0]);
    }

    #[test]
    #[serial]
    fn test_copy_prefetched_expert_into_aligned_buffer() {
        let ctx = pmetal_metal::context::MetalContext::global().unwrap();
        let mut aligned =
            pmetal_metal::expert_buffer::AlignedBuffer::new(ctx.device(), 16).unwrap();
        let payload = vec![9u8, 7, 5, 3, 1];
        copy_prefetched_expert_into_aligned(&mut aligned, &payload).unwrap();
        assert!(aligned.size() >= payload.len());
    }

    #[test]
    #[serial]
    fn test_gdn_combined_input_projection_matches_separate_linears() {
        let config = tiny_config();
        let mut gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[1, 1, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let qkv_ref = gdn.in_proj_qkv.forward(&x).unwrap();
        let z_ref = gdn
            .in_proj_z
            .forward(&x)
            .unwrap()
            .reshape(&[1, 1, gdn.num_v_heads, gdn.head_v_dim])
            .unwrap();
        let b_ref = gdn.in_proj_b.forward(&x).unwrap();
        let a_ref = gdn.in_proj_a.forward(&x).unwrap();

        let (qkv, z, b_val, a) = gdn.combined_input_projection(&x).unwrap();

        qkv.eval().unwrap();
        z.eval().unwrap();
        b_val.eval().unwrap();
        a.eval().unwrap();
        qkv_ref.eval().unwrap();
        z_ref.eval().unwrap();
        b_ref.eval().unwrap();
        a_ref.eval().unwrap();

        let max_qkv_diff = qkv
            .subtract(&qkv_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        let max_z_diff = z
            .subtract(&z_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        let max_b_diff = b_val
            .subtract(&b_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        let max_a_diff = a
            .subtract(&a_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();

        assert!(
            max_qkv_diff < 1e-5,
            "combined qkv projection diverged: {max_qkv_diff}"
        );
        assert!(
            max_z_diff < 1e-5,
            "combined z projection diverged: {max_z_diff}"
        );
        assert!(
            max_b_diff < 1e-5,
            "combined b projection diverged: {max_b_diff}"
        );
        assert!(
            max_a_diff < 1e-5,
            "combined a projection diverged: {max_a_diff}"
        );
    }

    #[test]
    fn test_gdn_combined_input_projection_is_gated_by_hidden_size_and_decode_shape() {
        let mut config = tiny_config();
        config.hidden_size = 2048;
        let gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let decode = Array::zeros_f32(&[1, 1, 2048]);
        let prefill = Array::zeros_f32(&[1, 2, 2048]);

        assert!(gdn.should_use_combined_input_proj(&decode, None));
        assert!(!gdn.should_use_combined_input_proj(&prefill, None));

        let mut large_config = config.clone();
        large_config.hidden_size = 4096;
        let large_gdn = Qwen3NextGatedDeltaNet::new(&large_config).unwrap();
        assert!(!large_gdn.should_use_combined_input_proj(&decode, None));
    }

    #[test]
    fn test_sparse_moe_shared_forward_matches_separate_paths() {
        let config = tiny_config();
        let mut moe = Qwen3NextSparseMoeBlock::new(&config).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[5, config.hidden_size],
            pmetal_bridge::compat::Dtype::Float32,
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
        let x = pmetal_bridge::compat::random::normal(
            &[3, config.hidden_size],
            pmetal_bridge::compat::Dtype::Float32,
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
        use pmetal_bridge::compat::ModuleParametersExt;

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

    #[test]
    fn test_gdn_decode_linear_projection_matches_linear_forward() {
        let mut config = tiny_config();
        config.hidden_size = 4096;
        let gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[1, 1, 4096],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let qkv_ref = gdn.in_proj_qkv.forward(&x).unwrap();
        let z_ref = gdn.in_proj_z.forward(&x).unwrap();
        let b_ref = gdn.in_proj_b.forward(&x).unwrap();
        let a_ref = gdn.in_proj_a.forward(&x).unwrap();

        let qkv = Qwen3NextGatedDeltaNet::decode_linear_projection(
            &x,
            gdn.in_proj_qkv.weight.as_ref(),
            gdn.conv_dim,
        )
        .unwrap();
        let z = Qwen3NextGatedDeltaNet::decode_linear_projection(
            &x,
            gdn.in_proj_z.weight.as_ref(),
            gdn.value_dim,
        )
        .unwrap();
        let b_val = Qwen3NextGatedDeltaNet::decode_linear_projection(
            &x,
            gdn.in_proj_b.weight.as_ref(),
            gdn.num_v_heads,
        )
        .unwrap();
        let a = Qwen3NextGatedDeltaNet::decode_linear_projection(
            &x,
            gdn.in_proj_a.weight.as_ref(),
            gdn.num_v_heads,
        )
        .unwrap();

        qkv.eval().unwrap();
        z.eval().unwrap();
        b_val.eval().unwrap();
        a.eval().unwrap();
        qkv_ref.eval().unwrap();
        z_ref.eval().unwrap();
        b_ref.eval().unwrap();
        a_ref.eval().unwrap();

        assert!(
            qkv.subtract(&qkv_ref)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>()
                < 1e-5
        );
        assert!(
            z.subtract(&z_ref)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>()
                < 1e-5
        );
        assert!(
            b_val
                .subtract(&b_ref)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>()
                < 1e-5
        );
        assert!(
            a.subtract(&a_ref)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>()
                < 1e-5
        );
    }

    #[test]
    fn test_gdn_decode_flatten_fastpath_is_gated_by_hidden_size_and_decode_shape() {
        let mut small_config = tiny_config();
        small_config.hidden_size = 2048;
        let small_gdn = Qwen3NextGatedDeltaNet::new(&small_config).unwrap();
        let decode = pmetal_bridge::compat::random::normal(
            &[1, 1, 2048],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let prefill = pmetal_bridge::compat::random::normal(
            &[1, 2, 2048],
            pmetal_bridge::compat::Dtype::Float32,
        );
        assert!(small_gdn.should_use_flattened_decode_proj(&decode, None));
        assert!(!small_gdn.should_use_flattened_decode_proj(&prefill, None));

        let mut large_config = tiny_config();
        large_config.hidden_size = 4096;
        let large_gdn = Qwen3NextGatedDeltaNet::new(&large_config).unwrap();
        let large_decode = pmetal_bridge::compat::random::normal(
            &[1, 1, 4096],
            pmetal_bridge::compat::Dtype::Float32,
        );
        assert!(!large_gdn.should_use_flattened_decode_proj(&large_decode, None));
    }

    #[test]
    fn test_gdn_decode_out_projection_matches_linear_forward() {
        let config = tiny_config();
        let gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let out = pmetal_bridge::compat::random::normal(
            &[1, 1, gdn.num_v_heads, gdn.head_v_dim],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let reference = gdn
            .out_proj
            .forward(&out.reshape(&[1, 1, gdn.value_dim]).unwrap())
            .unwrap();
        let projected = gdn.decode_out_projection(&out, 1).unwrap();

        reference.eval().unwrap();
        projected.eval().unwrap();
        let max_diff = reference
            .subtract(&projected)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            max_diff < 1e-5,
            "decode out projection diverged: {max_diff}"
        );
    }

    #[test]
    #[serial]
    fn test_gdn_cached_decode_matches_general_path() {
        let config = tiny_config();
        let mut gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let prefill = pmetal_bridge::compat::random::normal(
            &[1, 3, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let decode = pmetal_bridge::compat::random::normal(
            &[1, 1, 32],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let mut cache_general = MambaCacheEntry::default();
        let mut cache_fast = MambaCacheEntry::default();

        let _ = gdn
            .forward_general(&prefill, None, Some(&mut cache_general))
            .unwrap();
        let _ = gdn.forward(&prefill, None, Some(&mut cache_fast)).unwrap();

        let reference = gdn
            .forward_general(&decode, None, Some(&mut cache_general))
            .unwrap();
        let optimized = gdn.forward(&decode, None, Some(&mut cache_fast)).unwrap();

        let output_diff = reference.subtract(&optimized).unwrap().abs().unwrap();
        let max_output_diff = output_diff.max(None).unwrap().item::<f32>();
        assert!(
            max_output_diff < 1e-4,
            "cached decode output diverged from general path: {max_output_diff}"
        );

        let conv_diff = cache_general
            .conv_state
            .as_ref()
            .unwrap()
            .subtract(cache_fast.conv_state.as_ref().unwrap())
            .unwrap()
            .abs()
            .unwrap();
        let max_conv_diff = conv_diff.max(None).unwrap().item::<f32>();
        assert!(
            max_conv_diff < 1e-6,
            "cached decode conv state diverged from general path: {max_conv_diff}"
        );

        let ssm_diff = cache_general
            .ssm_state
            .as_ref()
            .unwrap()
            .subtract(cache_fast.ssm_state.as_ref().unwrap())
            .unwrap()
            .abs()
            .unwrap();
        let max_ssm_diff = ssm_diff.max(None).unwrap().item::<f32>();
        assert!(
            max_ssm_diff < 1e-4,
            "cached decode SSM state diverged from general path: {max_ssm_diff}"
        );
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
        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 32],
            pmetal_bridge::compat::Dtype::Float32,
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
        let x = pmetal_bridge::compat::random::normal(
            &[1, 2, config.hidden_size],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let err = block.forward(&x).unwrap_err().to_string();
        assert!(err.contains("enable expert offloading"));
    }
}
