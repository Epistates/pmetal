//! DFlash block-diffusion draft model.
//!
//! DFlash (Chen et al., 2026) is a block-diffusion drafter for flash
//! speculative decoding. Instead of autoregressively proposing tokens one at
//! a time, a small transformer takes a block of mask-token noise embeddings
//! plus intermediate hidden states tapped from the target model's previous
//! forward pass and proposes every token in the block in a single pass.
//!
//! The target model then verifies the entire block with one forward pass
//! (via [`crate::architectures::qwen3::Qwen3ForCausalLM::forward_with_capture`]
//! or the Qwen3.5 equivalent), accepts the longest matching prefix, and
//! feeds the next draft with the just-captured verifier hidden states.
//!
//! # Weights
//!
//! This model loads checkpoints from the `z-lab/*-DFlash*` family on Hugging
//! Face (currently `z-lab/Qwen3-4B-DFlash-b16` and
//! `z-lab/Qwen3.5-4B-DFlash`). Weight naming follows the upstream Python
//! implementation:
//! * `layers.{i}.self_attn.{q,k,v,o}_proj.weight`
//! * `layers.{i}.self_attn.{q,k}_norm.weight`
//! * `layers.{i}.{input_layernorm,post_attention_layernorm}.weight`
//! * `layers.{i}.mlp.{gate,up,down}_proj.weight`
//! * `fc.weight` (shape `[hidden, L * hidden]`)
//! * `hidden_norm.weight`, `norm.weight`
//!
//! The `dflash_config.target_layer_ids` list in the checkpoint's
//! `config.json` enumerates which target-model layers the drafter taps. The
//! verifier must request hidden-state capture at exactly those indices.

use std::collections::HashMap;

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParameters, ModuleParametersExt, Param, nn, ops,
};
use pmetal_bridge::impl_module_params;
use serde::{Deserialize, Serialize};

use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopePositions, RopeScaling, rope},
};
use pmetal_mlx::kv_cache::KVCache;

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_block_size() -> i32 {
    16
}

/// The RoPE base when a config names none, the reference implementation's.
const DEFAULT_ROPE_THETA: f32 = 10_000.0;

/// The architecture name a DFlash 2 checkpoint declares.
pub const DFLASH2_ARCHITECTURE: &str = "DFlash2DraftModel";

/// Extra DFlash-specific config section (stored under `dflash_config` in
/// the upstream config.json).
///
/// Everything after `mask_token_id` arrived with DFlash 2 and is absent from
/// the first generation's checkpoints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DFlashExtras {
    /// Target-model layer indices whose hidden states the drafter consumes.
    pub target_layer_ids: Vec<i32>,
    /// Token id of the `[MASK]` token used for block-diffusion noise.
    pub mask_token_id: i32,
    /// Block size. The reference reads it here first and falls back to the
    /// top-level `block_size`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<i32>,
    /// Taps of each dynamic convolution (DFlash 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conv_kernel_size: Option<i32>,
    /// Channels sharing one dynamic kernel weight (DFlash 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conv_group_size: Option<i32>,
    /// Rank of the candidate selector's codebooks (DFlash 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector_rank: Option<i32>,
    /// Candidates the selector chooses among at each position (DFlash 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector_top_k: Option<i32>,
    /// Multiplies the target's token embedding of the block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_embedding_scale: Option<f32>,
    /// Multiplies the draft logits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_multiplier: Option<f32>,
    /// `cap * tanh(logits / cap)` on the draft logits when positive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f32>,
}

/// Configuration for [`DFlashDraftModel`] and
/// [`DFlash2DraftModel`](super::dflash2_draft::DFlash2DraftModel).
///
/// Matches the reference implementation's `DFlashConfig`, which both
/// generations share. Fields that have sensible defaults in the upstream
/// implementation are given `#[serde(default)]` so a config.json that omits
/// them still loads.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DFlashDraftConfig {
    /// `["DFlash2DraftModel"]` for DFlash 2; the first generation says
    /// `DFlashDraftModel` or nothing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default)]
    pub max_position_embeddings: i32,
    /// The legacy top-level RoPE base; see [`Self::rope_theta`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<f32>,
    pub head_dim: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    /// The newer home of the RoPE base and scaling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_parameters: Option<HashMap<String, serde_json::Value>>,
    /// The legacy top-level block size; see [`Self::block_size`].
    #[serde(default = "default_block_size")]
    pub block_size: i32,
    /// `full_attention` or `sliding_attention` per layer; all full when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    /// Context window of the `sliding_attention` layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window: Option<i32>,
    /// Whether a block attends to itself causally. When absent, sliding layers
    /// are causal and full ones aren't, as in the reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_causal: Option<bool>,
    /// DFlash-specific hyperparameters.
    pub dflash_config: DFlashExtras,
}

impl DFlashDraftConfig {
    /// Target layer indices as `usize` — the form needed by
    /// [`pmetal_mlx::speculative::SpecCapture::with_layers`].
    pub fn target_layer_ids(&self) -> Vec<usize> {
        self.dflash_config
            .target_layer_ids
            .iter()
            .map(|&id| id as usize)
            .collect()
    }

    /// Number of layers the drafter conditions on.
    pub fn num_target_layers(&self) -> usize {
        self.dflash_config.target_layer_ids.len()
    }

    /// Whether the checkpoint is a DFlash 2 draft, by its declared
    /// architecture, as the reference implementation tells them apart.
    pub fn is_dflash2(&self) -> bool {
        self.architectures.iter().any(|a| a == DFLASH2_ARCHITECTURE)
    }

    /// Tokens per drafted block, the anchor included: `dflash_config`'s,
    /// else the top-level one.
    pub fn block_size(&self) -> i32 {
        self.dflash_config.block_size.unwrap_or(self.block_size)
    }

    /// RoPE base: the top-level `rope_theta`, else `rope_parameters`'.
    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|p| p.get("rope_theta"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
            })
            .unwrap_or(DEFAULT_ROPE_THETA)
    }

    /// RoPE scaling from `rope_scaling`, else `rope_parameters`.
    pub fn rope_scaling(&self) -> RopeScaling {
        self.rope_scaling
            .as_ref()
            .or(self.rope_parameters.as_ref())
            .map(RopeScaling::from_config_map)
            .unwrap_or(RopeScaling::None)
    }

    /// The sliding window of layer `layer`, `None` for a full-attention one.
    pub fn layer_window(&self, layer: usize) -> Option<i32> {
        let sliding = self
            .layer_types
            .as_ref()
            .and_then(|types| types.get(layer))
            .is_some_and(|t| t == "sliding_attention");
        if sliding { self.sliding_window } else { None }
    }

    /// Whether layer `layer`'s block attends to itself causally.
    pub fn layer_is_causal(&self, layer: usize) -> bool {
        self.is_causal
            .unwrap_or_else(|| self.layer_window(layer).is_some())
    }

    /// What the target's token embedding of the block is multiplied by.
    pub fn input_embedding_scale(&self) -> f32 {
        self.dflash_config.input_embedding_scale.unwrap_or(1.0)
    }

    /// The draft's logits from the LM head's: times `output_multiplier`,
    /// then soft-capped by `final_logit_softcapping`, as the reference does
    /// for both generations.
    pub fn scale_logits(&self, head_logits: Array) -> Array {
        let extras = &self.dflash_config;
        let dtype = head_logits.dtype().as_i32();
        let mut logits = head_logits;
        if let Some(m) = extras.output_multiplier.filter(|&m| m != 1.0) {
            logits = logits.multiply(&Array::from_f32(m).as_dtype(dtype));
        }
        if let Some(cap) = extras.final_logit_softcapping.filter(|&c| c > 0.0) {
            let cap = Array::from_f32(cap).as_dtype(dtype);
            logits = ops::tanh(&logits.divide(&cap)).multiply(&cap);
        }
        logits
    }

    /// Reject what the reference rejects: a layer type it doesn't know, a
    /// `layer_types` of the wrong length, or sliding layers with no window.
    pub fn validate(&self) -> Result<(), Exception> {
        if let Some(types) = &self.layer_types {
            if types.len() != self.num_hidden_layers as usize {
                return Err(Exception::custom(format!(
                    "DFlash draft: {} layer_types for {} layers",
                    types.len(),
                    self.num_hidden_layers
                )));
            }
            if let Some(t) = types
                .iter()
                .find(|t| *t != "full_attention" && *t != "sliding_attention")
            {
                return Err(Exception::custom(format!(
                    "DFlash draft: unsupported layer type {t:?}"
                )));
            }
            if types.iter().any(|t| t == "sliding_attention")
                && !self.sliding_window.is_some_and(|w| w > 0)
            {
                return Err(Exception::custom(
                    "DFlash draft: sliding_attention layers need a positive sliding_window",
                ));
            }
        }
        if self.block_size() < 1 {
            return Err(Exception::custom(format!(
                "DFlash draft: block_size {} is not positive",
                self.block_size()
            )));
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// MLP
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct DFlashMlp {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(DFlashMlp; gate_proj, up_proj, down_proj);

impl DFlashMlp {
    pub fn new(config: &DFlashDraftConfig) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        Ok(self.down_proj.forward(&nn::silu(&gate).multiply(&up)))
    }
}

// ----------------------------------------------------------------------------
// Attention
// ----------------------------------------------------------------------------

/// DFlash cross-attention.
///
/// Queries come from the block alone. Keys and values come from the context
/// (the target's hidden states, projected) followed by the block, so the
/// block attends to everything decided so far and to itself. With a cache,
/// only the context's keys and values are kept: each draft adds the context
/// rows decided since the last one, and the block's are dropped with the
/// draft, as in the reference implementation.
///
/// A sliding layer sees the context within `window` positions of each query;
/// a causal one sees the block only up to its own position.
#[derive(Debug)]
pub struct DFlashAttention {
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
    pub rope_scale: f32,
    pub effective_base: f32,
    /// Context window of a sliding layer, `None` for a full one.
    pub window: Option<i32>,
    /// Whether the block attends to itself causally.
    pub causal: bool,
}
impl_module_params!(DFlashAttention; q_proj, k_proj, v_proj, o_proj, q_norm, k_norm);

impl DFlashAttention {
    /// Attention for draft layer `layer`, whose window and causality the
    /// config's `layer_types` and `is_causal` decide.
    pub fn new(config: &DFlashDraftConfig, layer: usize) -> Result<Self, Exception> {
        let head_dim = config.head_dim;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;

        let rope_scaling = config.rope_scaling();
        let rope_scale = rope_scaling.scale();
        let effective_base = rope_scaling.effective_base(config.rope_theta(), head_dim);

        let q_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
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
            rope_scale,
            effective_base,
            window: config.layer_window(layer),
            causal: config.layer_is_causal(layer),
        })
    }

    /// Keys (normed) and values of `x`, `[B, kv_heads, T, head_dim]`.
    fn keys_values(&mut self, x: &Array) -> (Array, Array) {
        let (batch, len) = (x.dim(0), x.dim(1));
        let shape = [batch, len, self.n_kv_heads, self.head_dim];
        let keys = self.k_norm.forward(&self.k_proj.forward(x).reshape(&shape));
        let values = self.v_proj.forward(x).reshape(&shape);
        (
            keys.transpose_axes(&[0, 2, 1, 3]),
            values.transpose_axes(&[0, 2, 1, 3]),
        )
    }

    fn rope(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        rope(
            x,
            RopePositions::Offset(offset),
            self.head_dim,
            false,
            self.effective_base,
            self.rope_scale,
        )
    }

    /// Attend the block `hidden_states` `[B, L, hidden]` to the context and
    /// itself. `target_hidden` `[B, S, hidden]` is the projected context: with
    /// a `cache` (a single-layer [`KVCache`]) the rows new since the last
    /// call, which join it; without one, the whole context, from position 0.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        target_hidden: &Array,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let batch = hidden_states.dim(0);
        let query_len = hidden_states.dim(1);
        let new_context = target_hidden.dim(1);

        let queries = self
            .q_norm
            .forward(&self.q_proj.forward(hidden_states).reshape(&[
                batch,
                query_len,
                self.n_heads,
                self.head_dim,
            ]))
            .transpose_axes(&[0, 2, 1, 3]);
        let (context_keys, context_values) = self.keys_values(target_hidden);
        let (block_keys, block_values) = self.keys_values(hidden_states);

        // The context rows take the positions after those already cached and
        // the block the ones after them, as in the reference implementation.
        let start = cache.as_ref().map_or(0, |c| c.rope_offset_for(0));
        let block_start = start + new_context;
        let queries = self.rope(&queries, block_start)?;
        let context_keys = self.rope(&context_keys, start)?;
        let block_keys = self.rope(&block_keys, block_start)?;

        let (context_keys, context_values) = match cache {
            Some(cache) => cache.update_and_fetch(0, &context_keys, &context_values)?,
            None => (context_keys, context_values),
        };
        let context_len = context_keys.dim(2);
        let keys = ops::concatenate_axis(&[&context_keys, &block_keys], 2);
        let values = ops::concatenate_axis(&[&context_values, &block_values], 2);

        let mask = block_attention_mask(context_len, query_len, self.window, self.causal)
            .map(|m| m.as_dtype(queries.dtype().as_i32()));
        let output = pmetal_bridge::compat::fast::scaled_dot_product_attention_masked(
            &queries,
            &keys,
            &values,
            self.scale,
            mask.as_ref(),
        );

        let output = output.transpose_axes(&[0, 2, 1, 3]).reshape(&[
            batch,
            query_len,
            self.n_heads * self.head_dim,
        ]);
        Ok(self.o_proj.forward(&output))
    }
}

/// The additive mask `[1, 1, L, C + L]` for a block of `L` queries over `C`
/// context keys and the block's own, or `None` when every key is visible.
///
/// The context keys are the `C` positions just before the block, so query
/// `i` and context key `j` are `C + i - j` apart; a sliding layer hides those
/// `window` or more apart. A causal layer hides the block's later positions.
pub fn block_attention_mask(
    context_len: i32,
    block_len: i32,
    window: Option<i32>,
    causal: bool,
) -> Option<Array> {
    let windowed = window.is_some_and(|w| context_len + block_len > w);
    if !windowed && !causal {
        return None;
    }
    let keys = context_len + block_len;
    let mut mask = Vec::with_capacity((block_len * keys) as usize);
    for i in 0..block_len {
        for j in 0..keys {
            let visible = if j < context_len {
                window.is_none_or(|w| context_len + i - j < w)
            } else {
                !causal || j - context_len <= i
            };
            mask.push(if visible { 0.0f32 } else { f32::NEG_INFINITY });
        }
    }
    Some(Array::from_slice(&mask, &[1, 1, block_len, keys]))
}

// ----------------------------------------------------------------------------
// Decoder layer
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct DFlashDecoderLayer {
    pub input_layernorm: nn::RmsNorm,
    pub self_attn: DFlashAttention,
    pub post_attention_layernorm: nn::RmsNorm,
    pub mlp: DFlashMlp,
}
impl_module_params!(DFlashDecoderLayer; input_layernorm, self_attn, post_attention_layernorm, mlp);

impl DFlashDecoderLayer {
    pub fn new(config: &DFlashDraftConfig, layer: usize) -> Result<Self, Exception> {
        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let self_attn = DFlashAttention::new(config, layer)?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let mlp = DFlashMlp::new(config)?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        target_hidden: &Array,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let residual = hidden_states.clone();
        let normed = self.input_layernorm.forward(hidden_states);
        let attn = self.self_attn.forward(&normed, target_hidden, cache)?;
        let hidden_states = residual.add(&attn);

        let residual = hidden_states.clone();
        let normed = self.post_attention_layernorm.forward(&hidden_states);
        let mlp = self.mlp.forward(&normed)?;
        Ok(residual.add(&mlp))
    }
}

// ----------------------------------------------------------------------------
// Top-level draft model
// ----------------------------------------------------------------------------

/// DFlash draft model.
///
/// Unlike a standard causal LM the draft does not own token embeddings or
/// an lm_head — the DFlash pipeline shares both with the target model. The
/// draft is therefore a stack of decoder layers plus the `fc` + `hidden_norm`
/// projection that conditions on target hidden states.
#[derive(Debug)]
pub struct DFlashDraftModel {
    pub layers: Vec<DFlashDecoderLayer>,
    /// Projects `[B, T, L * hidden]` target hidden states down to `[B, T, hidden]`.
    pub fc: nn::Linear,
    /// RMSNorm applied to the projected target hidden states.
    pub hidden_norm: nn::RmsNorm,
    /// Final RMSNorm over the draft hidden states.
    pub norm: nn::RmsNorm,
    pub config: DFlashDraftConfig,
}
impl_module_params!(DFlashDraftModel; layers, fc, hidden_norm, norm);

impl DFlashDraftModel {
    pub fn new(config: DFlashDraftConfig) -> Result<Self, Exception> {
        let l = config.num_target_layers() as i32;
        if l == 0 {
            return Err(Exception::custom(
                "DFlashDraftModel requires at least one target_layer_id",
            ));
        }

        config.validate()?;
        let layers = (0..config.num_hidden_layers as usize)
            .map(|layer| DFlashDecoderLayer::new(&config, layer))
            .collect::<Result<Vec<_>, _>>()?;

        let fc = nn::LinearBuilder::new(l * config.hidden_size, config.hidden_size)
            .bias(false)
            .build()?;
        let hidden_norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            layers,
            fc,
            hidden_norm,
            norm,
            config,
        })
    }

    /// DFlash block size — how many tokens the drafter proposes per step.
    pub fn block_size(&self) -> usize {
        self.config.block_size() as usize
    }

    /// Token id used to fill proposal slots in the noise embedding.
    pub fn mask_token_id(&self) -> i32 {
        self.config.dflash_config.mask_token_id
    }

    /// Number of layers in the draft stack — handy when constructing
    /// per-layer `KVCache`s.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass.
    ///
    /// * `noise_embedding`: `[B, T_block, hidden]` — the target model's
    ///   token embedding of the mask-token block. The DFlash pipeline
    ///   computes this via `target.embed_tokens(block_input)` and passes
    ///   the result in directly, so the draft does not need its own token
    ///   embeddings.
    /// * `target_hidden`: `[B, T_ctx, L * hidden]` — the hidden states
    ///   captured from the target model's most recent forward pass,
    ///   concatenated along the hidden dimension in the order of
    ///   [`DFlashExtras::target_layer_ids`].
    /// * `cache`: optional per-layer KV cache from
    ///   [`make_cache`](Self::make_cache), which `target_hidden` joins; pass
    ///   `None` for cacheless operation over a whole context.
    pub fn forward(
        &mut self,
        noise_embedding: &Array,
        target_hidden: &Array,
        mut cache: Option<&mut [KVCache]>,
    ) -> Result<Array, Exception> {
        // Project target_hidden [B, T, L*hidden] → [B, T, hidden] and norm.
        let projected = self.fc.forward(target_hidden);
        let target_hidden = self.hidden_norm.forward(&projected);

        let mut hidden = noise_embedding.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = cache.as_deref_mut().and_then(|caches| caches.get_mut(i));
            hidden = layer.forward(&hidden, &target_hidden, layer_cache)?;
        }
        Ok(self.norm.forward(&hidden))
    }

    /// Pack every projection for MLX's quantized matmul. At 8 bits the
    /// drafts barely change (the same tokens per verify step on
    /// Qwen3-4B) and a draft reads half the bytes.
    pub fn quantize(
        &mut self,
        params: pmetal_bridge::native_weight::QuantParams,
    ) -> Result<(), Exception> {
        let mut result = Ok(());
        pmetal_bridge::compat::VisitLinears::visit_linears_mut(self, "", &mut |_, linear| {
            if result.is_ok() {
                result = linear.quantize(params);
            }
        });
        result
    }

    /// A KV cache per layer, with room for `context` positions of context,
    /// for [`draft_block`](Self::draft_block).
    pub fn make_cache(&self, context: usize) -> Vec<KVCache> {
        make_context_cache(&self.config, context)
    }

    /// Draft a block against everything drafted against so far:
    /// `target_hidden` holds the target's tapped states for
    /// the context positions since the last draft, which join `cache` for
    /// good, and `noise_embedding` the block, whose keys and values never
    /// enter it. Every draft attends to the whole context (a sliding layer
    /// to its window of it).
    pub fn draft_block(
        &mut self,
        noise_embedding: &Array,
        target_hidden: &Array,
        cache: &mut [KVCache],
    ) -> Result<Array, Exception> {
        self.forward(noise_embedding, target_hidden, Some(cache))
    }
}

/// One single-layer [`KVCache`] per draft layer, holding the context only.
/// A sliding layer keeps the `window - 1` positions its next block can see
/// (the reference's rotating cache); a full one room for `context`.
pub fn make_context_cache(config: &DFlashDraftConfig, context: usize) -> Vec<KVCache> {
    (0..config.num_hidden_layers as usize)
        .map(|layer| {
            let kv = pmetal_mlx::kv_cache::KVCacheConfig::new(
                1,
                context.max(1),
                config.num_key_value_heads as usize,
                config.head_dim as usize,
            );
            let kv = match config.layer_window(layer) {
                Some(window) => kv.with_sliding_window((window - 1).max(1) as usize),
                None => kv,
            };
            KVCache::new(kv)
        })
        .collect()
}

// ----------------------------------------------------------------------------
// Weight loading
// ----------------------------------------------------------------------------

impl DFlashDraftModel {
    /// Load weights from a flat `name → tensor` map.
    ///
    /// Accepts both the upstream DFlash naming (`layers.{i}.…`) and a
    /// `model.layers.{i}.…` prefixed variant so a safetensors file that
    /// follows either convention drops in.
    pub fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
    ) -> Result<LoadReport, Exception> {
        let mut report = LoadReport::default();
        for (name, weight) in weights {
            let stripped = name.strip_prefix("model.").unwrap_or(name.as_str());
            match stripped {
                "fc.weight" => {
                    self.fc.weight = Param::new(weight.clone());
                    report.loaded += 1;
                }
                "hidden_norm.weight" => {
                    self.hidden_norm.weight = Param::new(weight.clone());
                    report.loaded += 1;
                }
                "norm.weight" => {
                    self.norm.weight = Param::new(weight.clone());
                    report.loaded += 1;
                }
                other if other.starts_with("layers.") => {
                    // layers.{i}.{rest}
                    let parts: Vec<&str> = other.splitn(3, '.').collect();
                    if parts.len() < 3 {
                        report.skipped.push(name.clone());
                        continue;
                    }
                    let Ok(layer_idx) = parts[1].parse::<usize>() else {
                        report.skipped.push(name.clone());
                        continue;
                    };
                    if layer_idx >= self.layers.len() {
                        report.skipped.push(name.clone());
                        continue;
                    }
                    if assign_layer_weight(&mut self.layers[layer_idx], parts[2], weight.clone()) {
                        report.loaded += 1;
                    } else {
                        report.skipped.push(name.clone());
                    }
                }
                _ => report.skipped.push(name.clone()),
            }
        }
        Ok(report)
    }
}

fn assign_layer_weight(layer: &mut DFlashDecoderLayer, suffix: &str, weight: Array) -> bool {
    match suffix {
        "input_layernorm.weight" => {
            layer.input_layernorm.weight = Param::new(weight);
        }
        "post_attention_layernorm.weight" => {
            layer.post_attention_layernorm.weight = Param::new(weight);
        }
        "self_attn.q_proj.weight" => layer.self_attn.q_proj.weight = Param::new(weight),
        "self_attn.k_proj.weight" => layer.self_attn.k_proj.weight = Param::new(weight),
        "self_attn.v_proj.weight" => layer.self_attn.v_proj.weight = Param::new(weight),
        "self_attn.o_proj.weight" => layer.self_attn.o_proj.weight = Param::new(weight),
        "self_attn.q_norm.weight" => layer.self_attn.q_norm.weight = Param::new(weight),
        "self_attn.k_norm.weight" => layer.self_attn.k_norm.weight = Param::new(weight),
        "mlp.gate_proj.weight" => layer.mlp.gate_proj.weight = Param::new(weight),
        "mlp.up_proj.weight" => layer.mlp.up_proj.weight = Param::new(weight),
        "mlp.down_proj.weight" => layer.mlp.down_proj.weight = Param::new(weight),
        _ => return false,
    }
    true
}

/// Summary of a [`DFlashDraftModel::load_weights`] call — shared with every
/// other hand-written loader.
pub use crate::architectures::utils::LoadReport;

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn tiny_config() -> DFlashDraftConfig {
        DFlashDraftConfig {
            model_type: "dflash_qwen3".to_string(),
            hidden_size: 32,
            num_hidden_layers: 2,
            intermediate_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            vocab_size: 128,
            max_position_embeddings: 64,
            rope_theta: Some(10_000.0),
            head_dim: 8,
            block_size: 4,
            dflash_config: DFlashExtras {
                target_layer_ids: vec![1, 3],
                mask_token_id: 7,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    #[serial]
    fn test_dflash_draft_forward_shape() {
        let config = tiny_config();
        let hidden = config.hidden_size;
        let block = config.block_size;
        let num_target = config.num_target_layers() as i32;

        let mut model = DFlashDraftModel::new(config).unwrap();

        // Simulate the DFlash pipeline: noise embedding is [B, block, hidden]
        // and target_hidden is [B, ctx, num_target * hidden].
        let noise = pmetal_bridge::compat::random::normal(
            &[1, block, hidden],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let target_hidden = pmetal_bridge::compat::random::normal(
            &[1, 6, num_target * hidden],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let out = model.forward(&noise, &target_hidden, None).unwrap();
        assert_eq!(out.shape(), &[1, block, hidden]);
    }

    /// Drafting against context A, then against B through the cache, is
    /// drafting once against A and B together: the cache keeps the context
    /// and only the block's own keys and values are dropped. Drafting against
    /// B alone (what `DFlashDecoder` did, without a cache) is not.
    #[test]
    #[serial]
    fn test_dflash_draft_block_attends_to_the_whole_context() {
        use pmetal_bridge::compat::{Dtype, random::normal};
        let config = tiny_config();
        let (hidden, block) = (config.hidden_size, config.block_size);
        let row = config.num_target_layers() as i32 * hidden;
        let mut model = DFlashDraftModel::new(config).unwrap();
        let first = normal(&[1, block, hidden], Dtype::Float32);
        let second = normal(&[1, block, hidden], Dtype::Float32);
        let (a, b) = (
            normal(&[1, 3, row], Dtype::Float32),
            normal(&[1, 2, row], Dtype::Float32),
        );

        let mut cache = model.make_cache(16);
        model.draft_block(&first, &a, &mut cache).unwrap();
        let cached = model.draft_block(&second, &b, &mut cache).unwrap();
        let whole = model
            .forward(&second, &ops::concatenate_axis(&[&a, &b], 1), None)
            .unwrap();
        let last_only = model.forward(&second, &b, None).unwrap();

        let max_diff = |x: &Array, y: &Array| {
            let (x, y) = (x.clone(), y.clone());
            let _ = x.eval();
            let _ = y.eval();
            x.as_slice::<f32>()
                .iter()
                .zip(y.as_slice::<f32>())
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(
            max_diff(&cached, &whole) < 1e-4,
            "cached draft != whole-context draft"
        );
        assert!(
            max_diff(&last_only, &whole) > 1e-3,
            "the context made no difference"
        );
    }

    #[test]
    #[serial]
    fn test_dflash_draft_block_size_and_mask_token() {
        let model = DFlashDraftModel::new(tiny_config()).unwrap();
        assert_eq!(model.block_size(), 4);
        assert_eq!(model.mask_token_id(), 7);
        assert_eq!(model.num_layers(), 2);
    }

    #[test]
    fn test_dflash_draft_config_requires_target_layers() {
        let mut config = tiny_config();
        config.dflash_config.target_layer_ids.clear();
        let err = DFlashDraftModel::new(config).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("target_layer_id"),
            "expected target_layer_id error, got: {msg}"
        );
    }
}
