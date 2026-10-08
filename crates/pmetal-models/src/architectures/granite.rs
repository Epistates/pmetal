//! IBM Granite: one implementation for the four text families that share a
//! decoder layer.
//!
//! | `model_type`        | Released as                         | Mixer            | FFN                    |
//! |---------------------|-------------------------------------|------------------|------------------------|
//! | `granite`           | Granite 3.x dense, 4.1, 4.2         | RoPE attention   | SwiGLU `mlp`           |
//! | `granitemoe`        | Granite 3.x `a400m` / `a800m`       | RoPE attention   | routed experts         |
//! | `granitemoeshared`  | (no IBM release)                    | RoPE attention   | experts + shared MLP   |
//! | `granitemoehybrid`  | Granite 4.0 (`-h-*` and dense)      | Mamba-2 or attention, RoPE or NoPE | shared MLP, plus experts when `num_local_experts > 0` |
//!
//! All four scale the same four places: embeddings by `embedding_multiplier`,
//! attention logits by `attention_multiplier` (instead of `1/sqrt(head_dim)`),
//! each residual branch by `residual_multiplier`, and the logits divided by
//! `logits_scaling`. The Mamba-2 mixer, the experts and the shared MLP live in
//! [`super::granite_hybrid`].
//!
//! Anything else that says "granite" (the sliding-window `granite_swa`
//! variants, `granite_switch`, the speech and vision wrappers) is refused by
//! name in [`granite_refusal`] rather than routed here, and a config of one of
//! the four families that sets something this does not compute is refused by
//! [`GraniteConfig::validate`].

use pmetal_bridge::compat::{Array, Exception, Module, ModuleParameters, ModuleParametersExt, nn};
use pmetal_bridge::impl_module_params;

use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopePositions, rope},
};
use pmetal_mlx::kv_cache::{KVCache, MambaCache};

use serde::{Deserialize, Serialize};

use super::granite_hybrid::{GraniteMamba, GraniteMoe, GraniteSharedMlp};
use crate::checkpointing::checkpointed_layer;
use crate::decoder_layer::{AttentionModule, DecoderLayer, MlpModule};
use crate::traits::ModelConfig;

/// Which Granite family a config describes, from its `model_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraniteFamily {
    /// `granite`: attention + SwiGLU `mlp`.
    Dense,
    /// `granitemoe`: attention + routed experts, no shared MLP.
    Moe,
    /// `granitemoeshared`: attention + routed experts + shared MLP.
    MoeShared,
    /// `granitemoehybrid`: per-layer Mamba-2 or attention, shared MLP, and
    /// routed experts when `num_local_experts > 0`.
    MoeHybrid,
}

impl GraniteFamily {
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "granite" => Some(Self::Dense),
            "granitemoe" => Some(Self::Moe),
            "granitemoeshared" => Some(Self::MoeShared),
            "granitemoehybrid" => Some(Self::MoeHybrid),
            _ => None,
        }
    }
}

/// Why a Granite `model_type` outside the four families pmetal computes is
/// refused, or `None` for one that is not Granite at all.
///
/// These all name their architectures `Granite*`, which an earlier substring
/// match routed to plain Granite: a sliding-window config ran with full
/// attention, and a speech or vision wrapper loaded its text weights under a
/// prefix nothing matched.
pub fn granite_refusal(model_type: &str) -> Option<String> {
    let reason = match model_type {
        "granite_swa" | "granitemoe_swa" => {
            "its sliding-window attention layers are not implemented"
        }
        "granite_switch" => "the granite_switch architecture is not implemented",
        "granite4_vision" | "granite_vision" => {
            "it is a vision-language wrapper; pmetal runs Granite text models only"
        }
        t if t.starts_with("granite_speech") => {
            "it is a speech model; pmetal runs Granite text models only"
        }
        t if t.starts_with("granite") && GraniteFamily::from_model_type(t).is_none() => {
            "it is not one of the Granite families pmetal implements \
             (granite, granitemoe, granitemoeshared, granitemoehybrid)"
        }
        _ => return None,
    };
    Some(format!(
        "Unsupported Granite model_type {model_type:?}: {reason}"
    ))
}

/// Layer type for Granite Hybrid models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraniteLayerType {
    /// Standard attention layer.
    #[default]
    Attention,
    /// Mamba2 state-space layer.
    Mamba2,
}

impl GraniteLayerType {
    /// Read a `layer_types` entry. Released configs say `"mamba"` and
    /// `"attention"`; transformers normalizes those to `"linear_attention"` and
    /// `"full_attention"`, and accepts either spelling.
    pub fn from_config_str(name: &str) -> Option<Self> {
        match name {
            "attention" | "full_attention" => Some(Self::Attention),
            "mamba" | "linear_attention" => Some(Self::Mamba2),
            _ => None,
        }
    }
}

/// Granite model configuration, for all four families.
///
/// Field names and defaults are transformers' (`GraniteConfig`,
/// `GraniteMoeConfig`, `GraniteMoeSharedConfig`, `GraniteMoeHybridConfig`).
/// Where the families default differently the field is an `Option` resolved
/// by `model_type`; read those through the accessor methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraniteConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    /// Head dimension, or `None` to derive it.
    ///
    /// Released Granite configs write `"head_dim": null` and expect
    /// `hidden_size / num_attention_heads`, exactly as `GraniteConfig` in
    /// `transformers` does. This used to be a required `i32`, so every real
    /// Granite checkpoint failed to deserialize on the `null`. Read it through
    /// [`GraniteConfig::resolved_head_dim`].
    #[serde(default)]
    pub head_dim: Option<i32>,
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// The transformers-5 spelling of the RoPE settings. Its `rope_theta`
    /// wins over the top-level one, as it does in the reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_parameters: Option<serde_json::Value>,
    /// Any value but `null` is refused: no released Granite scales RoPE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<serde_json::Value>,
    pub rms_norm_eps: f32,
    /// Defaults to `false`, as in every Granite family's reference config.
    /// Every release states it.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub attention_bias: bool,
    /// Plain Granite only.
    #[serde(default)]
    pub mlp_bias: bool,

    // ---- Granite's four scalar multipliers ----
    //
    // These are what distinguish Granite from Llama; `modeling_granite.py`
    // marks the logits one "main diff with Llama". All four were missing, so
    // pmetal ran Granite as plain Llama. The attention one is not a rounding
    // error: `granite-3.1-2b` sets `attention_multiplier` to 0.015625 against a
    // head_dim of 64, whose standard `1/sqrt(64)` scale is 0.125 — attention
    // logits eight times too large, before softmax.
    /// Attention logit scale, replacing `1/sqrt(head_dim)` outright.
    ///
    /// `None` falls back to the standard scale. The reference defaults this to
    /// `1.0`, which is right for a config that omits it *by construction* but
    /// catastrophic for a hand-built one; every released Granite config states
    /// it, so the fallback only ever applies to synthetic configs, where the
    /// standard scale is what a caller means.
    #[serde(default)]
    pub attention_multiplier: Option<f32>,
    /// Token embeddings are multiplied by this immediately after lookup.
    #[serde(default = "default_one")]
    pub embedding_multiplier: f32,
    /// Every residual branch is scaled by this before it is added back.
    #[serde(default = "default_one")]
    pub residual_multiplier: f32,
    /// Final logits are **divided** by this.
    #[serde(default = "default_one")]
    pub logits_scaling: f32,

    // ---- Routed experts (granitemoe*) ----
    /// Experts per MoE layer. Defaults to 8 for the MoE families, as in the
    /// reference; `granitemoehybrid` releases set 0 for their dense models.
    #[serde(default)]
    pub num_local_experts: Option<i32>,
    #[serde(default)]
    pub num_experts_per_tok: Option<i32>,
    /// Width of the shared MLP. Defaults to 0 (none) for `granitemoeshared`
    /// and 1024 for `granitemoehybrid`; `granitemoe` has none.
    #[serde(default)]
    pub shared_intermediate_size: Option<i32>,

    // ---- Hybrid (granitemoehybrid) ----
    /// `"rope"` rotates the attention layers; anything else (released hybrids
    /// say `"nope"`) leaves them without positional encoding. Only read for
    /// `granitemoehybrid`; the other families always rotate.
    #[serde(default)]
    pub position_embedding_type: Option<String>,
    /// Per-layer `"mamba"` / `"attention"`. A `granitemoehybrid` config that
    /// omits it is all Mamba, as in the reference.
    #[serde(default, alias = "layers_block_type")]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "default_mamba_n_heads")]
    pub mamba_n_heads: i32,
    #[serde(default = "default_one_i32")]
    pub mamba_n_groups: i32,
    #[serde(default = "default_mamba_d_state")]
    pub mamba_d_state: i32,
    /// An integer, or `"auto"` for `mamba_expand * hidden_size / mamba_n_heads`.
    #[serde(default)]
    pub mamba_d_head: Option<serde_json::Value>,
    #[serde(default = "default_mamba_d_conv")]
    pub mamba_d_conv: i32,
    #[serde(default = "default_mamba_expand")]
    pub mamba_expand: i32,
    #[serde(default = "default_mamba_chunk_size")]
    pub mamba_chunk_size: i32,
    #[serde(default = "default_true")]
    pub mamba_conv_bias: bool,
    #[serde(default)]
    pub mamba_proj_bias: bool,
    /// Only the reference default, `(0, inf)`, is accepted; see `validate`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_step_limit: Option<(f32, f32)>,
}

fn default_model_type() -> String {
    "granite".to_string()
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_hidden_act() -> String {
    "silu".to_string()
}
fn default_true() -> bool {
    true
}
fn default_one() -> f32 {
    1.0
}
fn default_one_i32() -> i32 {
    1
}
fn default_mamba_n_heads() -> i32 {
    128
}
fn default_mamba_d_state() -> i32 {
    256
}
fn default_mamba_d_conv() -> i32 {
    4
}
fn default_mamba_expand() -> i32 {
    2
}
fn default_mamba_chunk_size() -> i32 {
    256
}

impl Default for GraniteConfig {
    fn default() -> Self {
        Self {
            model_type: default_model_type(),
            vocab_size: 49152,
            hidden_size: 2048,
            intermediate_size: 5504,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: Some(128),
            max_position_embeddings: 8192,
            rope_theta: 10000.0,
            rope_parameters: None,
            rope_scaling: None,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            hidden_act: default_hidden_act(),
            attention_bias: false,
            mlp_bias: false,
            attention_multiplier: None,
            embedding_multiplier: 1.0,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
            num_local_experts: None,
            num_experts_per_tok: None,
            shared_intermediate_size: None,
            position_embedding_type: None,
            layer_types: None,
            mamba_n_heads: default_mamba_n_heads(),
            mamba_n_groups: 1,
            mamba_d_state: default_mamba_d_state(),
            mamba_d_head: None,
            mamba_d_conv: default_mamba_d_conv(),
            mamba_expand: default_mamba_expand(),
            mamba_chunk_size: default_mamba_chunk_size(),
            mamba_conv_bias: true,
            mamba_proj_bias: false,
            time_step_limit: None,
        }
    }
}

impl GraniteConfig {
    /// Parse a `config.json` and refuse anything [`validate`](Self::validate)
    /// does not accept. Every construction path from a checkpoint goes through
    /// here.
    pub fn from_config_json(config_content: &str) -> Result<Self, Exception> {
        let config: Self = json5::from_str(config_content)
            .map_err(|e| Exception::custom(format!("Granite config: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    /// The family `model_type` names.
    pub fn family(&self) -> Result<GraniteFamily, Exception> {
        GraniteFamily::from_model_type(&self.model_type).ok_or_else(|| {
            Exception::custom(granite_refusal(&self.model_type).unwrap_or_else(|| {
                format!(
                    "Granite config with unknown model_type {:?}",
                    self.model_type
                )
            }))
        })
    }

    /// Refuse a config whose model pmetal would compute differently from the
    /// reference, naming what it cannot do.
    pub fn validate(&self) -> Result<(), Exception> {
        let refuse = |what: String| {
            Err(Exception::custom(format!(
                "Unsupported {} config: {what}",
                self.model_type
            )))
        };
        let family = self.family()?;

        if self.hidden_act != "silu" {
            return refuse(format!(
                "hidden_act {:?} (every Granite family runs SiLU)",
                self.hidden_act
            ));
        }
        if self.uses_rope() {
            if self.rope_scaling.as_ref().is_some_and(|v| !v.is_null()) {
                return refuse("rope_scaling is set; scaled RoPE is not implemented".into());
            }
            if let Some(params) = &self.rope_parameters {
                let rope_type = params.get("rope_type").and_then(|v| v.as_str());
                if !matches!(rope_type, None | Some("default")) {
                    return refuse(format!(
                        "rope_parameters.rope_type {rope_type:?}; only default RoPE is implemented"
                    ));
                }
                let partial = params.get("partial_rotary_factor").and_then(|v| v.as_f64());
                if partial.is_some_and(|p| p != 1.0) {
                    return refuse("partial rotary embeddings are not implemented".into());
                }
            }
        }
        if family == GraniteFamily::MoeHybrid {
            match self.position_embedding_type.as_deref() {
                None | Some("rope") | Some("nope") => {}
                Some(other) => {
                    return refuse(format!("position_embedding_type {other:?}"));
                }
            }
        }
        if self.mlp_bias && family != GraniteFamily::Dense {
            return refuse("mlp_bias is only defined for plain Granite".into());
        }

        // Layer layout.
        if let Some(types) = &self.layer_types {
            if types.len() != self.num_hidden_layers as usize {
                return refuse(format!(
                    "layer_types has {} entries for {} layers",
                    types.len(),
                    self.num_hidden_layers
                ));
            }
            for name in types {
                match GraniteLayerType::from_config_str(name) {
                    Some(GraniteLayerType::Mamba2) if family != GraniteFamily::MoeHybrid => {
                        return refuse(format!(
                            "layer type {name:?} in a {:?} model",
                            self.model_type
                        ));
                    }
                    Some(_) => {}
                    None => return refuse(format!("unknown layer type {name:?}")),
                }
            }
        }

        // Experts.
        let experts = self.num_experts();
        if experts < 0 {
            return refuse(format!("num_local_experts {experts}"));
        }
        if matches!(family, GraniteFamily::Moe | GraniteFamily::MoeShared) && experts == 0 {
            return refuse("an MoE family with no experts".into());
        }
        if experts > 0 {
            let k = self.experts_per_token();
            if k < 1 || k > experts {
                return refuse(format!("num_experts_per_tok {k} with {experts} experts"));
            }
        }
        if family == GraniteFamily::MoeHybrid && self.shared_mlp_size() <= 0 {
            // The reference builds the shared MLP unconditionally, and for a
            // dense hybrid it is the only FFN there is.
            return refuse("granitemoehybrid with shared_intermediate_size <= 0".into());
        }

        // Mamba.
        if self.has_mamba_layers() {
            let intermediate = self.mamba_intermediate_size();
            if self.mamba_n_heads <= 0 || intermediate % self.mamba_n_heads != 0 {
                return refuse(format!(
                    "mamba_n_heads {} does not divide mamba_expand * hidden_size = {intermediate}",
                    self.mamba_n_heads
                ));
            }
            let head_dim = match &self.mamba_d_head {
                None => None,
                Some(v) if v.as_str() == Some("auto") => None,
                Some(v) => match v.as_i64() {
                    Some(d) => Some(d as i32),
                    None => return refuse(format!("mamba_d_head {v}")),
                },
            };
            if head_dim.is_some_and(|d| d * self.mamba_n_heads != intermediate) {
                return refuse(format!(
                    "mamba_d_head * mamba_n_heads != mamba_expand * hidden_size ({intermediate})"
                ));
            }
            if self.mamba_n_groups <= 0 || self.mamba_n_heads % self.mamba_n_groups != 0 {
                return refuse(format!(
                    "mamba_n_groups {} does not divide mamba_n_heads {}",
                    self.mamba_n_groups, self.mamba_n_heads
                ));
            }
            if self.mamba_d_conv < 1 {
                return refuse(format!("mamba_d_conv {}", self.mamba_d_conv));
            }
            // The reference clamps `dt` to this range in its chunked scan and
            // not at all in its single-token step, so any limit but the
            // default makes prefill and decode disagree with each other.
            if let Some((lo, hi)) = self.time_step_limit {
                if lo != 0.0 || hi != f32::INFINITY {
                    return refuse(format!(
                        "time_step_limit ({lo}, {hi}); only (0, inf) is computed consistently"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Head dimension, derived when the config leaves it `null`.
    ///
    /// Matches `GraniteConfig.head_dim = head_dim or hidden_size //
    /// num_attention_heads` in the reference.
    pub fn resolved_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// The attention logit scale this config asks for.
    ///
    /// Granite states `attention_multiplier` and the reference uses it *as*
    /// the scale — it does not combine with `1/sqrt(head_dim)`, it replaces it.
    /// See the field docs for why the fallback is the standard scale rather
    /// than the reference's `1.0`.
    pub fn attention_scale(&self) -> f32 {
        self.attention_multiplier
            .unwrap_or_else(|| (self.resolved_head_dim() as f32).sqrt().recip())
    }

    /// RoPE base: `rope_parameters.rope_theta` when present, else the
    /// top-level `rope_theta`.
    pub fn resolved_rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|p| p.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(self.rope_theta)
    }

    /// Whether attention layers rotate. Every family but the hybrid always
    /// does; a hybrid only with `position_embedding_type: "rope"`.
    pub fn uses_rope(&self) -> bool {
        match GraniteFamily::from_model_type(&self.model_type) {
            Some(GraniteFamily::MoeHybrid) => {
                self.position_embedding_type.as_deref() == Some("rope")
            }
            _ => true,
        }
    }

    /// Get the layer type for a given layer index.
    pub fn layer_type(&self, layer_idx: usize) -> GraniteLayerType {
        if GraniteFamily::from_model_type(&self.model_type) != Some(GraniteFamily::MoeHybrid) {
            return GraniteLayerType::Attention;
        }
        match &self.layer_types {
            Some(types) => types
                .get(layer_idx)
                .and_then(|name| GraniteLayerType::from_config_str(name))
                .unwrap_or_default(),
            // `GraniteMoeHybridConfig` fills an absent list with Mamba.
            None => GraniteLayerType::Mamba2,
        }
    }

    /// Whether any layer is a Mamba-2 mixer, i.e. the model carries recurrent
    /// state between decode steps.
    pub fn has_mamba_layers(&self) -> bool {
        (0..self.num_hidden_layers as usize).any(|i| self.layer_type(i) == GraniteLayerType::Mamba2)
    }

    /// Routed experts per layer, 0 for none.
    pub fn num_experts(&self) -> i32 {
        match GraniteFamily::from_model_type(&self.model_type) {
            Some(GraniteFamily::Dense) | None => 0,
            Some(_) => self.num_local_experts.unwrap_or(8),
        }
    }

    pub fn experts_per_token(&self) -> i32 {
        self.num_experts_per_tok.unwrap_or(2)
    }

    /// Shared-MLP width, 0 for none.
    pub fn shared_mlp_size(&self) -> i32 {
        match GraniteFamily::from_model_type(&self.model_type) {
            Some(GraniteFamily::MoeShared) => self.shared_intermediate_size.unwrap_or(0),
            Some(GraniteFamily::MoeHybrid) => self.shared_intermediate_size.unwrap_or(1024),
            _ => 0,
        }
    }

    pub fn mamba_intermediate_size(&self) -> i32 {
        self.mamba_expand * self.hidden_size
    }

    pub fn mamba_head_dim(&self) -> i32 {
        self.mamba_d_head
            .as_ref()
            .and_then(|v| v.as_i64())
            .map(|d| d as i32)
            .unwrap_or(self.mamba_intermediate_size() / self.mamba_n_heads)
    }

    pub fn mamba_conv_dim(&self) -> i32 {
        self.mamba_intermediate_size() + 2 * self.mamba_n_groups * self.mamba_d_state
    }
}

impl ModelConfig for GraniteConfig {
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
        self.num_key_value_heads
    }
    fn head_dim(&self) -> i32 {
        self.resolved_head_dim()
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
        self.resolved_rope_theta()
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
}

// =============================================================================
// Model Components
// =============================================================================

/// SwiGLU MLP for plain Granite.
#[derive(Debug)]
pub struct GraniteMLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(GraniteMLP; gate_proj, up_proj, down_proj);

impl GraniteMLP {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        Self::with_bias(hidden_size, intermediate_size, false)
    }

    /// `mlp_bias` puts a bias on all three projections.
    pub fn with_bias(
        hidden_size: i32,
        intermediate_size: i32,
        bias: bool,
    ) -> Result<Self, Exception> {
        let linear = |i, o| nn::LinearBuilder::new(i, o).bias(bias).build();
        Ok(Self {
            gate_proj: linear(hidden_size, intermediate_size)?,
            up_proj: linear(hidden_size, intermediate_size)?,
            down_proj: linear(intermediate_size, hidden_size)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // SwiGLU: silu(gate) * up
        let gate = Module::forward(&mut self.gate_proj, x)?;
        let gate = nn::silu(&gate);
        let up = Module::forward(&mut self.up_proj, x)?;
        let hidden = gate.multiply(&up);
        Module::forward(&mut self.down_proj, &hidden)
    }
}

/// Granite attention with GQA, and RoPE unless the config says NoPE.
#[derive(Debug)]
pub struct GraniteAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,
    /// `false` for a `granitemoehybrid` NoPE config: the Mamba layers carry
    /// position and the attention layers do not rotate at all.
    pub use_rope: bool,

    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
}
impl_module_params!(GraniteAttention; q_proj, k_proj, v_proj, o_proj);

impl GraniteAttention {
    pub fn new(config: &GraniteConfig) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.resolved_head_dim();
        let hidden_size = config.hidden_size;
        let linear = |i, o| {
            nn::LinearBuilder::new(i, o)
                .bias(config.attention_bias)
                .build()
        };

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: config.attention_scale(),
            rope_theta: config.resolved_rope_theta(),
            use_rope: config.uses_rope(),
            q_proj: linear(hidden_size, n_heads * head_dim)?,
            k_proj: linear(hidden_size, n_kv_heads * head_dim)?,
            v_proj: linear(hidden_size, n_kv_heads * head_dim)?,
            o_proj: linear(n_heads * head_dim, hidden_size)?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        _position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None, None)
    }

    /// Forward pass with optional KV cache: project, reshape, transpose to
    /// `[B, heads, seq, head_dim]` *before* RoPE so axis -2 is the sequence,
    /// rotate, write the cache, fused SDPA.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let q = Module::forward(&mut self.q_proj, x)?;
        let k = Module::forward(&mut self.k_proj, x)?;
        let v = Module::forward(&mut self.v_proj, x)?;

        let q = q
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let k = k
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let v = v
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        let (q, k) = if self.use_rope {
            let rope_positions = RopePositions::resolve(
                positions,
                cache
                    .as_ref()
                    .map_or(0, |(c, layer)| c.rope_offset_for(*layer)),
            );
            (
                rope(
                    &q,
                    rope_positions,
                    self.head_dim,
                    false,
                    self.rope_theta,
                    1.0,
                )?,
                rope(
                    &k,
                    rope_positions,
                    self.head_dim,
                    false,
                    self.rope_theta,
                    1.0,
                )?,
            )
        } else {
            (q, k)
        };

        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });
        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);
        Module::forward(&mut self.o_proj, &output)
    }
}

impl AttentionModule for GraniteAttention {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        GraniteAttention::forward_with_cache(self, x, mask, cache, positions)
    }
}

impl MlpModule for GraniteMLP {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        GraniteMLP::forward(self, x)
    }
}

/// The per-layer recurrent state a Mamba layer reads and advances, or none.
type MambaSlot<'a> = Option<&'a mut pmetal_mlx::kv_cache::MambaCacheEntry>;

/// Granite decoder layer.
///
/// Exactly one of `self_attn` / `mamba` is set (the mixer), and the FFN is
/// either the plain-Granite `mlp` or the MoE family's `block_sparse_moe`
/// and/or `shared_mlp`, summed when both are present. Field names are the
/// checkpoint's, which is what lets the loader match by exact path.
#[derive(Debug)]
pub struct GraniteDecoderLayer {
    pub layer_type: GraniteLayerType,
    /// `config.residual_multiplier`, applied to both residual branches.
    pub residual_multiplier: f32,

    /// Named `self_attn`, not `attention`, because the generic loader assigns
    /// weights by exact parameter path: released Granite checkpoints ship
    /// `model.layers.N.self_attn.q_proj.weight`, and a field called `attention`
    /// makes every one of those keys unmatched. An unmatched key used to be
    /// dropped silently, so the whole attention stack stayed at random init
    /// and Granite inference returned noise while parsing perfectly.
    pub self_attn: Option<GraniteAttention>,
    pub mamba: Option<GraniteMamba>,
    pub mlp: Option<GraniteMLP>,
    pub block_sparse_moe: Option<GraniteMoe>,
    pub shared_mlp: Option<GraniteSharedMlp>,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}
impl_module_params!(
    GraniteDecoderLayer;
    self_attn,
    mamba,
    mlp,
    block_sparse_moe,
    shared_mlp,
    input_layernorm,
    post_attention_layernorm
);

impl GraniteDecoderLayer {
    pub fn new(config: &GraniteConfig, layer_idx: usize) -> Result<Self, Exception> {
        let family = config.family()?;
        let layer_type = config.layer_type(layer_idx);
        let hidden = config.hidden_size;

        let (self_attn, mamba) = match layer_type {
            GraniteLayerType::Attention => (Some(GraniteAttention::new(config)?), None),
            GraniteLayerType::Mamba2 => (None, Some(GraniteMamba::new(config)?)),
        };

        let mlp = if family == GraniteFamily::Dense {
            Some(GraniteMLP::with_bias(
                hidden,
                config.intermediate_size,
                config.mlp_bias,
            )?)
        } else {
            None
        };
        let block_sparse_moe = if config.num_experts() > 0 {
            Some(GraniteMoe::new(
                hidden,
                config.intermediate_size,
                config.num_experts(),
                config.experts_per_token(),
            )?)
        } else {
            None
        };
        let shared_mlp = if config.shared_mlp_size() > 0 {
            Some(GraniteSharedMlp::new(hidden, config.shared_mlp_size())?)
        } else {
            None
        };

        let norm = || {
            nn::RmsNormBuilder::new(hidden)
                .eps(config.rms_norm_eps)
                .build()
        };
        Ok(Self {
            layer_type,
            residual_multiplier: config.residual_multiplier,
            self_attn,
            mamba,
            mlp,
            block_sparse_moe,
            shared_mlp,
            input_layernorm: norm()?,
            post_attention_layernorm: norm()?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let _ = position_ids;
        self.forward_with_cache(x, mask, None, None)
    }

    /// Forward with no recurrent state: a Mamba layer runs the sequence from
    /// zero state, which is right for training and an uncached forward.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_state(x, mask, cache, positions, None)
    }

    /// Forward with both caches: attention layers read and extend `cache`,
    /// Mamba layers read and advance `mamba`.
    ///
    /// `mask` only reaches attention; the Mamba mixer is causal by
    /// construction.
    pub fn forward_with_state(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
        mamba: MambaSlot<'_>,
    ) -> Result<Array, Exception> {
        let m = self.residual_multiplier;
        // `mul_scalar` casts the scalar to the branch's dtype.
        let scale = |branch: Array| {
            if m == 1.0 {
                branch
            } else {
                branch.mul_scalar(m)
            }
        };

        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let mixed = match (self.self_attn.as_mut(), self.mamba.as_mut()) {
            (Some(attn), _) => attn.forward_with_cache(&normed, mask, cache, positions)?,
            (None, Some(mixer)) => mixer.forward(&normed, mamba)?,
            (None, None) => unreachable!("a Granite layer always has a mixer"),
        };
        let h = x.add(&scale(mixed));

        let normed = Module::forward(&mut self.post_attention_layernorm, &h)?;
        let ffn = self.feed_forward(&normed)?;
        Ok(h.add(&scale(ffn)))
    }

    /// `mlp`, or `block_sparse_moe` plus `shared_mlp`.
    fn feed_forward(&mut self, x: &Array) -> Result<Array, Exception> {
        if let Some(mlp) = self.mlp.as_mut() {
            return mlp.forward(x);
        }
        let routed = match self.block_sparse_moe.as_mut() {
            Some(moe) => Some(moe.forward(x)?),
            None => None,
        };
        let shared = match self.shared_mlp.as_mut() {
            Some(shared) => Some(shared.forward(x)?),
            None => None,
        };
        match (routed, shared) {
            (Some(r), Some(s)) => Ok(r.add(&s)),
            (Some(r), None) => Ok(r),
            (None, Some(s)) => Ok(s),
            (None, None) => Err(Exception::custom("Granite layer has no feed-forward")),
        }
    }
}

impl DecoderLayer for GraniteDecoderLayer {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        GraniteDecoderLayer::forward_with_cache(self, x, mask, cache, positions)
    }
}

/// Granite model.
#[derive(Debug)]
pub struct GraniteModel {
    pub config: GraniteConfig,

    pub embed_tokens: nn::Embedding,
    pub layers: Vec<GraniteDecoderLayer>,
    pub norm: nn::RmsNorm,
    /// Recompute each layer's activations during the backward pass instead of
    /// holding them. Training only; see [`crate::checkpointing`].
    pub grad_checkpoint: bool,
}
impl_module_params!(GraniteModel; embed_tokens, layers, norm);

impl GraniteModel {
    pub fn new(config: GraniteConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| GraniteDecoderLayer::new(&config, i as usize))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            grad_checkpoint: false,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_inner(input_ids, mask, None, position_ids, None)
    }

    /// Cached forward for a model with no Mamba layers. A hybrid needs its
    /// recurrent state too; see [`forward_with_hybrid_cache`](Self::forward_with_hybrid_cache).
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        self.forward_inner(input_ids, mask, cache, None, None)
    }

    /// Cached forward with the recurrent state the Mamba layers need.
    pub fn forward_with_hybrid_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        self.forward_inner(input_ids, mask, cache, None, mamba_cache)
    }

    /// Forward pass with one rotary position per token, `[seq_len]`.
    ///
    /// Packed training concatenates several sequences into one row, so the
    /// positions have to restart at each boundary instead of running through
    /// it. The block-diagonal mask keeps the content apart; this keeps the
    /// positions apart.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_inner(input_ids, mask, None, positions, None)
    }

    /// The layer loop, with every per-call input the layers can take.
    fn forward_inner(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
        positions: Option<&Array>,
        mut mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        // A KV cache means a decode is being carried across calls. Running the
        // Mamba layers of that decode without their state would restart the
        // recurrence on every call: finite, plausible, and wrong.
        if cache.is_some() && mamba_cache.is_none() && self.config.has_mamba_layers() {
            return Err(Exception::custom(
                "Granite hybrid: a cached forward needs the Mamba state as well as the KV \
                 cache. Use forward_with_hybrid_cache with create_mamba_cache().",
            ));
        }

        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?
            .mul_scalar(self.config.embedding_multiplier);

        // Hoisted: the loop below borrows `self.layers` mutably.
        let grad_checkpoint = self.grad_checkpoint;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let c = cache.as_deref_mut().map(|c| (c, idx));
            let state = match layer.layer_type {
                GraniteLayerType::Mamba2 => mamba_cache.as_deref_mut().and_then(|m| m.get_mut(idx)),
                GraniteLayerType::Attention => None,
            };
            // A cache means generation, which has no backward pass for the
            // recompute to pay for.
            hidden_states = if grad_checkpoint && c.is_none() && state.is_none() {
                checkpointed_layer(
                    layer,
                    &hidden_states,
                    mask,
                    positions,
                    |layer, h, mask, positions| layer.forward_with_cache(h, mask, None, positions),
                )?
            } else {
                layer.forward_with_state(&hidden_states, mask, c, positions, state)?
            };
        }

        Module::forward(&mut self.norm, &hidden_states)
    }
}

/// Granite for causal language modeling.
#[derive(Debug)]
pub struct GraniteForCausalLM {
    pub config: GraniteConfig,

    pub model: GraniteModel,
    pub lm_head: Option<nn::Linear>,
}
impl_module_params!(GraniteForCausalLM; model, lm_head);

impl GraniteForCausalLM {
    pub fn new(config: GraniteConfig) -> Result<Self, Exception> {
        // Only create separate lm_head if not tied
        let lm_head = if !config.tie_word_embeddings {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        } else {
            None
        };

        let model = GraniteModel::new(config.clone())?;

        Ok(Self {
            config,
            model,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward(input_ids, mask, position_ids)?;
        self.project_logits(&hidden_states)
    }

    /// Forward pass with optional KV cache for incremental decoding. Errors
    /// for a hybrid given a cache; see [`Self::forward_with_hybrid_cache`].
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward_with_cache(input_ids, mask, cache)?;
        self.project_logits(&hidden_states)
    }

    /// Cached forward carrying both the KV cache and the Mamba state.
    pub fn forward_with_hybrid_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let hidden_states =
            self.model
                .forward_with_hybrid_cache(input_ids, mask, cache, mamba_cache)?;
        self.project_logits(&hidden_states)
    }

    /// Forward pass with one rotary position per token; see
    /// `forward_with_positions` on the inner model.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden_states = self
            .model
            .forward_with_positions(input_ids, mask, positions)?;
        self.project_logits(&hidden_states)
    }

    fn project_logits(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        let logits = if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, hidden_states)?
        } else {
            // Tied embeddings: logits = hidden @ embed.weight.T
            let embed_weight = self.model.embed_tokens.weight.as_ref();
            let embed_t = embed_weight.t();
            hidden_states.matmul(&embed_t)
        };
        // `logits = logits / self.config.logits_scaling` — the line
        // `modeling_granite.py` annotates "main diff with Llama".
        Ok(logits.div_scalar(self.config.logits_scaling))
    }

    /// Create a fresh KV cache sized for this model. Mamba layers leave their
    /// slots empty, which keeps slot indices equal to layer indices.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        use pmetal_mlx::kv_cache::KVCacheConfig;
        KVCache::new(KVCacheConfig::new(
            self.config.num_hidden_layers as usize,
            max_seq_len,
            self.config.num_key_value_heads as usize,
            self.config.resolved_head_dim() as usize,
        ))
    }

    /// Recurrent state for the Mamba layers, `None` for a model without any.
    pub fn create_mamba_cache(&self) -> Option<MambaCache> {
        self.config
            .has_mamba_layers()
            .then(|| MambaCache::new(self.config.num_hidden_layers as usize))
    }

    /// Whether the fused `[N, 1]` batched decode computes this model.
    ///
    /// Only plain Granite with a unit `residual_multiplier`:
    /// `batched_prenorm_layer` runs a SwiGLU `mlp` and attention with RoPE,
    /// carries no recurrent state, and adds its residuals unscaled. Every
    /// released Granite scales them, so in practice this is the serial path.
    pub fn supports_fused_batched(&self) -> bool {
        GraniteFamily::from_model_type(&self.config.model_type) == Some(GraniteFamily::Dense)
            && self.config.residual_multiplier == 1.0
    }

    /// Fused batched-decode forward for the configs
    /// [`supports_fused_batched`](Self::supports_fused_batched) admits.
    pub fn forward_batched_impl(
        &mut self,
        input_ids: &Array,
        active_indices: &[usize],
        cache: &mut pmetal_mlx::kv_cache::FusedBatchKVCache,
    ) -> Result<Array, Exception> {
        use crate::common::{BatchedGqaAttnCfg, batched_prenorm_layer};

        if !self.supports_fused_batched() {
            return Err(Exception::custom(
                "forward_batched_impl invoked on a Granite config the fused path does not \
                 compute; supports_fused_batched should have gated this off",
            ));
        }
        let cfg = &self.config;
        let attn_cfg = BatchedGqaAttnCfg::new(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.resolved_head_dim(),
            cfg.resolved_rope_theta(),
            1.0,
        )
        .with_scale(cfg.attention_scale());
        let embedding_multiplier = cfg.embedding_multiplier;

        let mut hidden = Module::forward(&mut self.model.embed_tokens, input_ids)?
            .mul_scalar(embedding_multiplier);
        for (layer_idx, layer) in self.model.layers.iter_mut().enumerate() {
            let (Some(attn), Some(mlp)) = (layer.self_attn.as_mut(), layer.mlp.as_mut()) else {
                return Err(Exception::custom(
                    "forward_batched_impl: plain Granite layer without attention or mlp",
                ));
            };
            hidden = batched_prenorm_layer(
                &hidden,
                &mut layer.input_layernorm,
                &mut attn.q_proj,
                &mut attn.k_proj,
                &mut attn.v_proj,
                &mut attn.o_proj,
                None,
                None,
                &mut layer.post_attention_layernorm,
                mlp,
                &attn_cfg,
                cache,
                active_indices,
                layer_idx,
            )?;
        }
        let hidden = Module::forward(&mut self.model.norm, &hidden)?;
        self.project_logits(&hidden)
    }

    /// Number of parameters, for diagnostics.
    pub fn parameter_count(&self) -> usize {
        self.num_parameters()
    }

    /// Materialise every parameter.
    pub fn eval(&self) -> Result<(), Exception> {
        ModuleParametersExt::eval(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::{ModuleParameters, random};
    use serial_test::serial;

    fn hybrid_config() -> GraniteConfig {
        GraniteConfig {
            model_type: "granitemoehybrid".into(),
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 16,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: None,
            layer_types: Some(
                ["mamba", "attention", "mamba", "mamba"]
                    .map(String::from)
                    .to_vec(),
            ),
            position_embedding_type: Some("nope".into()),
            num_local_experts: Some(4),
            num_experts_per_tok: Some(2),
            shared_intermediate_size: Some(24),
            mamba_n_heads: 8,
            mamba_d_state: 8,
            mamba_chunk_size: 4,
            ..Default::default()
        }
    }

    #[test]
    fn released_layer_type_spellings_resolve() {
        let config = hybrid_config();
        assert_eq!(config.layer_type(0), GraniteLayerType::Mamba2);
        assert_eq!(config.layer_type(1), GraniteLayerType::Attention);
        assert!(config.has_mamba_layers());
        assert!(!config.uses_rope());
        assert_eq!(config.mamba_head_dim(), 8);

        // transformers' normalized spelling reads the same.
        let normalized = GraniteConfig {
            layer_types: Some(
                [
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                ]
                .map(String::from)
                .to_vec(),
            ),
            ..hybrid_config()
        };
        assert_eq!(normalized.layer_type(0), GraniteLayerType::Mamba2);
        assert_eq!(normalized.layer_type(1), GraniteLayerType::Attention);

        // An absent list is all Mamba, as `GraniteMoeHybridConfig` fills it.
        let absent = GraniteConfig {
            layer_types: None,
            ..hybrid_config()
        };
        assert_eq!(absent.layer_type(1), GraniteLayerType::Mamba2);
    }

    #[test]
    fn family_defaults_follow_the_reference() {
        let moe = GraniteConfig {
            model_type: "granitemoe".into(),
            ..Default::default()
        };
        assert_eq!(moe.num_experts(), 8);
        assert_eq!(moe.shared_mlp_size(), 0);
        assert!(moe.uses_rope());

        let hybrid = GraniteConfig {
            model_type: "granitemoehybrid".into(),
            layer_types: Some(vec!["attention".into(); 24]),
            ..Default::default()
        };
        assert_eq!(hybrid.shared_mlp_size(), 1024);
        // No `position_embedding_type`: the reference builds no rotary
        // embedding at all.
        assert!(!hybrid.uses_rope());

        let dense = GraniteConfig::default();
        assert_eq!(dense.num_experts(), 0);
        assert!(dense.uses_rope());
    }

    #[test]
    fn configs_pmetal_would_compute_wrong_are_refused() {
        let cases: Vec<(&str, GraniteConfig)> = vec![
            (
                "time_step_limit",
                GraniteConfig {
                    time_step_limit: Some((0.001, 0.1)),
                    ..hybrid_config()
                },
            ),
            (
                "hidden_act",
                GraniteConfig {
                    hidden_act: "gelu".into(),
                    ..hybrid_config()
                },
            ),
            (
                "rope_scaling",
                GraniteConfig {
                    rope_scaling: Some(serde_json::json!({"type": "linear", "factor": 2.0})),
                    ..Default::default()
                },
            ),
            (
                "unknown layer type",
                GraniteConfig {
                    layer_types: Some(
                        ["mamba", "sliding_attention", "mamba", "mamba"]
                            .map(String::from)
                            .to_vec(),
                    ),
                    ..hybrid_config()
                },
            ),
            (
                "mamba layer outside the hybrid family",
                GraniteConfig {
                    model_type: "granitemoe".into(),
                    ..hybrid_config()
                },
            ),
            (
                "unknown model_type",
                GraniteConfig {
                    model_type: "granite_swa".into(),
                    ..Default::default()
                },
            ),
        ];
        for (what, config) in cases {
            assert!(config.validate().is_err(), "{what} should be refused");
        }
        hybrid_config()
            .validate()
            .expect("the hybrid fixture is valid");
    }

    #[test]
    fn granite_variants_outside_the_four_families_are_refused_by_name() {
        for model_type in [
            "granite_swa",
            "granitemoe_swa",
            "granite_switch",
            "granite4_vision",
            "granite_speech",
            "granite_speech_plus",
        ] {
            let reason = granite_refusal(model_type).expect("refused");
            assert!(reason.contains(model_type), "{reason}");
        }
        for model_type in [
            "granite",
            "granitemoe",
            "granitemoeshared",
            "granitemoehybrid",
            "llama",
        ] {
            assert!(granite_refusal(model_type).is_none(), "{model_type}");
        }
    }

    #[test]
    #[serial]
    fn hybrid_cached_decode_equals_the_full_forward() {
        random::seed(7);
        let mut model = GraniteForCausalLM::new(hybrid_config()).unwrap();
        // Ten tokens with a chunk size of 4: the uncached forward crosses two
        // chunk boundaries.
        let tokens = [3, 11, 7, 2, 9, 40, 5, 17, 23, 61];
        let full = model
            .forward(
                &Array::from_i32_slice(&tokens).reshape(&[1, 10]),
                None,
                None,
            )
            .unwrap();

        let mut kv = model.create_cache(16);
        let mut state = model.create_mamba_cache().expect("hybrid");
        let prefill = model
            .forward_with_hybrid_cache(
                &Array::from_i32_slice(&tokens[..6]).reshape(&[1, 6]),
                None,
                Some(&mut kv),
                Some(&mut state),
            )
            .unwrap();
        let mut rows = vec![prefill];
        for &t in &tokens[6..] {
            rows.push(
                model
                    .forward_with_hybrid_cache(
                        &Array::from_i32_slice(&[t]).reshape(&[1, 1]),
                        None,
                        Some(&mut kv),
                        Some(&mut state),
                    )
                    .unwrap(),
            );
        }
        let refs: Vec<&Array> = rows.iter().collect();
        let cached = pmetal_bridge::compat::ops::concatenate_axis(&refs, 1);
        let diff = cached.subtract(&full).abs().max(None).item::<f32>();
        pmetal_bridge::check_last_error().unwrap();
        assert!(
            diff < 1e-4,
            "cached decode drifted from the full forward by {diff}"
        );

        // A KV cache without the Mamba state is refused, not run statelessly.
        let mut kv = model.create_cache(16);
        assert!(
            model
                .forward_with_cache(
                    &Array::from_i32_slice(&[3]).reshape(&[1, 1]),
                    None,
                    Some(&mut kv)
                )
                .is_err()
        );
    }

    #[test]
    #[serial]
    fn test_granite_mlp() {
        let mlp = GraniteMLP::new(64, 256).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[1, 10, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let mut mlp = mlp;
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[1, 10, 64]);
    }

    /// A config shaped like `ibm-granite/granite-3.1-2b-instruct` — same
    /// derived head_dim and the same four multipliers, at toy width.
    fn multiplier_config() -> GraniteConfig {
        GraniteConfig {
            vocab_size: 128,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            // `null` in the released config; derived as 64/4 = 16.
            head_dim: None,
            attention_multiplier: Some(0.015625),
            embedding_multiplier: 12.0,
            residual_multiplier: 0.22,
            logits_scaling: 8.0,
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    #[test]
    fn granite_derives_head_dim_when_the_config_says_null() {
        let config = multiplier_config();
        assert_eq!(config.resolved_head_dim(), 16);
        // A stated value still wins.
        let stated = GraniteConfig {
            head_dim: Some(48),
            ..multiplier_config()
        };
        assert_eq!(stated.resolved_head_dim(), 48);
    }

    #[test]
    fn granite_attention_scale_is_the_multiplier_not_one_over_sqrt_head_dim() {
        let config = multiplier_config();
        // The distinction is not academic: these differ by 8x, and the scale
        // lands on attention logits before the softmax.
        let standard = (config.resolved_head_dim() as f32).sqrt().recip();
        assert_eq!(standard, 0.25);
        assert_eq!(config.attention_scale(), 0.015625);

        let attn = GraniteAttention::new(&config).unwrap();
        assert_eq!(attn.scale, 0.015625);

        // Omitted, as in a hand-built config: fall back to the standard scale.
        let bare = GraniteConfig {
            attention_multiplier: None,
            ..multiplier_config()
        };
        assert_eq!(bare.attention_scale(), standard);
    }

    #[test]
    #[serial]
    fn granite_applies_embedding_residual_and_logits_scalars() {
        use pmetal_bridge::compat::transforms;

        let scaled = multiplier_config();
        let neutral = GraniteConfig {
            attention_multiplier: Some(0.015625),
            embedding_multiplier: 1.0,
            residual_multiplier: 1.0,
            logits_scaling: 1.0,
            ..multiplier_config()
        };

        let input = Array::from_i32_slice(&[3_i32, 9, 14]).reshape(&[1, 3]);

        // Same weights in both models: seed the RNG identically.
        let logits_for = |config: GraniteConfig| {
            random::seed(1234);
            let mut model = GraniteForCausalLM::new(config).unwrap();
            let out = model.forward(&input, None, None).unwrap();
            transforms::eval([&out]).unwrap();
            out.as_slice::<f32>().to_vec()
        };

        let with_scalars = logits_for(scaled);
        let without = logits_for(neutral);

        assert_eq!(with_scalars.len(), without.len());
        assert!(
            with_scalars.iter().all(|v| v.is_finite()),
            "scaled forward produced non-finite logits"
        );
        // Three multipliers of 12.0, 0.22 and 8.0 cannot compose to the
        // identity, so the two forwards must differ. Before this fix they were
        // the same computation: pmetal ran Granite as plain Llama.
        let max_diff = with_scalars
            .iter()
            .zip(&without)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "multipliers had no effect on the forward (max diff {max_diff})"
        );
    }

    #[test]
    fn granite_lora_and_base_agree_on_the_scalars() {
        // The GELU audit's lesson: an adapter trained against different math
        // than inference runs is worse than one that fails to build. These are
        // read from one place so the three Granite implementations cannot
        // drift.
        let config = multiplier_config();
        let layer = GraniteDecoderLayer::new(&config, 0).unwrap();
        assert_eq!(layer.residual_multiplier, config.residual_multiplier);
        assert_eq!(
            GraniteAttention::new(&config).unwrap().scale,
            config.attention_scale()
        );
    }

    #[test]
    #[serial]
    fn test_granite_model_instantiation() {
        let mut config = GraniteConfig::default();
        config.hidden_size = 64;
        config.intermediate_size = 256;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.head_dim = Some(16);
        config.vocab_size = 1000;
        config.tie_word_embeddings = true;

        let model = GraniteForCausalLM::new(config).unwrap();

        let params = model.flatten_params();
        assert!(params.len() > 0);
    }
}
