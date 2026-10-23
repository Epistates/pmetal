//! Standalone Gemma 4 inference engine — zero dependency on `pmetal-models`.
//!
//! This is the Gemma 4 counterpart of [`crate::qwen3_native`]: every hot-path
//! op goes directly through [`InlineArray`] (stack-allocated `mlx::core::array`
//! values routed through the `mlx_inline_*` C bridge), so there are no heap
//! allocations or Rust wrapper roundtrips per op.
//!
//! Gemma 4-specific features (text tower only):
//! * Per-layer-type `head_dim` (full=512, sliding=256).
//! * Per-layer-type `num_kv_heads` (`num_global_key_value_heads` for full
//!   layers, `num_key_value_heads` for sliding).
//! * `attention_k_eq_v`: full layers have NO `v_proj` — values come from the
//!   raw `k_proj` output BEFORE `k_norm` is applied.
//! * `v_norm`: RMSNorm **without** a learnable scale on the values.
//! * Per-layer-type partial RoPE. Full layers use a custom inverse-frequency
//!   array (the reference `ProportionalRoPE`) — `freqs[i] = base^(2i/head_dim)` for
//!   `i in 0..rotated_dims/2` with the remaining slots filled with `inf` so
//!   `fast::rope(..., freqs=)` leaves them untouched.
//! * Per-layer `layer_scalar` multiplier applied at the end of each decoder
//!   layer forward.
//! * Per-layer embeddings / gating (`hidden_size_per_layer_input`) used by
//!   the E2B/E4B checkpoints.
//! * KV-sharing (`num_kv_shared_layers`) used by the E2B/E4B checkpoints.
//! * Final logit softcap: `softcap * tanh(logits / softcap)`.
//! * Embedding scale by `sqrt(hidden_size)` (shared with Gemma 2/3).
//! * Tanh-approximation GELU in the MLP (matching MLX's `nn.gelu_approx`).
//!
//! Weights are pre-transposed to `[in, out]` at load time (`w.t()`) so the
//! per-decode matmuls are contiguous.

use crate::{RuntimeError, RuntimeResult};
use serde::Deserialize;

use crate::InlineArray;
use crate::compat::Dtype;
use crate::inline_array as bridge;
use crate::native_weight::{EmbeddingWeight, LayerWeight, QuantParams, detect_model_dtype};

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta_global() -> f32 {
    1_000_000.0
}
fn default_rope_theta_sliding() -> f32 {
    10_000.0
}
fn default_partial_rotary_factor() -> f32 {
    1.0
}
/// `Gemma4TextConfig.sliding_window`.
fn default_sliding_window() -> i32 {
    512
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Gemma4RopeLayerConfig {
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_type: Option<String>,
    /// Proportional RoPE's frequency divisor; Gemma 4 releases set none.
    #[serde(default)]
    pub factor: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Gemma4RopeConfig {
    #[serde(default)]
    pub full_attention: Gemma4RopeLayerConfig,
    #[serde(default)]
    pub sliding_attention: Gemma4RopeLayerConfig,
}

/// Minimal, serde-deserializable Gemma 4 text config. Unknown keys are
/// silently ignored so multimodal wrappers (`model.language_model.*`) can
/// share the same struct.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    #[serde(default)]
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub head_dim: i32,
    /// Per-layer-type `head_dim` used by full-attention layers. `None`
    /// reuses `head_dim`.
    #[serde(default)]
    pub global_head_dim: Option<i32>,
    /// Number of KV heads used by full-attention layers when
    /// `attention_k_eq_v` is set.
    #[serde(default)]
    pub num_global_key_value_heads: Option<i32>,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    /// `None` (the transformers default) applies no softcap.
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    /// Per-layer `"sliding_attention"` / `"full_attention"`. Parsing fills in
    /// `Gemma4TextConfig`'s default (every sixth layer full) when the config
    /// has none, and makes the last layer full as it does.
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// `"all"` makes every layer attend bidirectionally; `"vision"` only
    /// opens attention among a prompt's image tokens, which leaves a text
    /// prompt causal. Only the unset and `"vision"` cases run.
    #[serde(default)]
    pub use_bidirectional_attention: Option<String>,
    /// Biases on the attention projections, which the engine does not load.
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_parameters: Option<Gemma4RopeConfig>,

    // Unsupported-today blocks. Gemma 4 31B-it has `num_kv_shared_layers=0`,
    // no MoE — all remaining unsupported features are still strictly checked.
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<i32>,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<i32>,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub num_kv_shared_layers: Option<i32>,
    #[serde(default)]
    pub use_double_wide_mlp: Option<bool>,
    #[serde(default)]
    pub enable_moe_block: Option<bool>,
    /// Routed experts per layer, when `enable_moe_block` is set.
    #[serde(default)]
    pub num_experts: Option<i32>,
    /// Experts each token is routed to.
    #[serde(default)]
    pub top_k_experts: Option<i32>,
    /// Each expert's hidden width. Only the fused `gate_up_proj` layout needs
    /// it, to find where gate ends and up begins.
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
}

fn default_true() -> bool {
    true
}

impl Gemma4Config {
    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map(|s| s == "full_attention")
            .unwrap_or(false)
    }

    pub fn layer_head_dim(&self, layer_idx: usize) -> i32 {
        if self.is_full_attention(layer_idx) {
            self.global_head_dim.unwrap_or(self.head_dim)
        } else {
            self.head_dim
        }
    }

    pub fn layer_num_kv_heads(&self, layer_idx: usize) -> i32 {
        if self.is_full_attention(layer_idx) && self.attention_k_eq_v {
            if let Some(h) = self.num_global_key_value_heads {
                return h;
            }
        }
        self.num_key_value_heads
    }

    pub fn layer_uses_k_eq_v(&self, layer_idx: usize) -> bool {
        self.attention_k_eq_v && self.is_full_attention(layer_idx)
    }

    /// Layer `layer_idx`'s rotation; see [`gemma4_layer_rope`].
    pub fn layer_rope(&self, layer_idx: usize) -> RuntimeResult<Gemma4LayerRope> {
        let is_full = self.is_full_attention(layer_idx);
        let head_dim = self.layer_head_dim(layer_idx);
        let (default_base, default_frac) = if is_full {
            (default_rope_theta_global(), 0.25)
        } else {
            (default_rope_theta_sliding(), 1.0)
        };
        let cfg = self.rope_parameters.as_ref().map(|rp| {
            if is_full {
                &rp.full_attention
            } else {
                &rp.sliding_attention
            }
        });
        gemma4_layer_rope(
            head_dim,
            cfg.and_then(|c| c.rope_type.as_deref()),
            cfg.and_then(|c| c.rope_theta).unwrap_or(default_base),
            cfg.map_or(default_frac, |c| c.partial_rotary_factor),
            cfg.and_then(|c| c.factor),
            is_full,
        )
    }

    pub fn embed_scale(&self) -> f32 {
        (self.hidden_size as f32).sqrt()
    }

    pub fn uses_per_layer_inputs(&self) -> bool {
        self.hidden_size_per_layer_input.unwrap_or(0) > 0
    }

    pub fn per_layer_input_dim(&self) -> i32 {
        self.hidden_size_per_layer_input.unwrap_or(0)
    }

    pub fn per_layer_input_vocab_size(&self) -> i32 {
        self.vocab_size_per_layer_input.unwrap_or(self.vocab_size)
    }

    /// Whether every layer runs a routed-experts branch beside its dense MLP.
    pub fn uses_moe(&self) -> bool {
        self.enable_moe_block.unwrap_or(false)
    }

    pub fn num_kv_shared_layers(&self) -> usize {
        self.num_kv_shared_layers.unwrap_or(0).max(0) as usize
    }

    pub fn first_kv_shared_layer_idx(&self) -> usize {
        let total = self.num_hidden_layers.max(0) as usize;
        total.saturating_sub(self.num_kv_shared_layers())
    }

    pub fn kv_shared_source_layer(&self, layer_idx: usize) -> Option<usize> {
        let first_shared = self.first_kv_shared_layer_idx();
        if layer_idx < first_shared || first_shared == 0 {
            return None;
        }
        let is_full = self.is_full_attention(layer_idx);
        (0..first_shared)
            .rev()
            .find(|&src| self.is_full_attention(src) == is_full)
    }

    fn validate_supported(&self) -> RuntimeResult<()> {
        // Both the MLP's activation and the per-layer-input gate's.
        if let Some(ref act) = self.hidden_activation
            && !matches!(act.as_str(), "gelu" | "gelu_pytorch_tanh" | "gelu_tanh")
        {
            return Err(RuntimeError::Config(format!(
                "Gemma 4 native: unsupported hidden_activation {act:?}; the MLP and the \
             per-layer-input gate are tanh-approximate GELU"
            )));
        }
        match self.use_bidirectional_attention.as_deref() {
            None | Some("vision") => {}
            Some(other) => {
                return Err(RuntimeError::Config(format!(
                    "Gemma 4 native: use_bidirectional_attention {other:?} is not supported; \
                 the engine's attention is causal"
                )));
            }
        }
        if self.attention_bias {
            return Err(RuntimeError::Config(
                "Gemma 4 native: attention_bias is not supported; the projections load no biases"
                    .to_string(),
            ));
        }
        if self.uses_moe()
            && (self.num_experts.is_none()
                || self.top_k_experts.is_none()
                || self.moe_intermediate_size.is_none())
        {
            return Err(RuntimeError::Config("Gemma 4 native: MoE block needs num_experts, top_k_experts and moe_intermediate_size".to_string()));
        }
        if self.use_double_wide_mlp.unwrap_or(false) {
            return Err(RuntimeError::Config(
                "Gemma 4 native: double-wide MLP is not ported yet".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn load_config(model_dir: &std::path::Path) -> RuntimeResult<Gemma4Config> {
    let text = crate::native_loader::read_config_json(model_dir)?;
    parse_config_text(&text)
}

fn parse_config_text(text: &str) -> RuntimeResult<Gemma4Config> {
    let json: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| RuntimeError::Config(format!("failed to parse config.json: {e}")))?;

    // Multimodal Gemma 4 nests text params under `text_config`.
    let config_str = if json.get("text_config").is_some() {
        serde_json::to_string(&json["text_config"])
            .map_err(|e| RuntimeError::Config(e.to_string()))?
    } else {
        text.to_owned()
    };

    let mut cfg: Gemma4Config = serde_json::from_str(&config_str)
        .map_err(|e| RuntimeError::Config(format!("failed to parse gemma4 config: {e}")))?;
    cfg.validate_supported()?;
    // `Gemma4TextConfig.__post_init__`: every sixth layer is full attention
    // unless the config says otherwise, and the last one always is.
    if cfg.layer_types.is_empty() {
        cfg.layer_types = (0..cfg.num_hidden_layers)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .map(str::to_string)
            .collect();
    }
    if let Some(last) = cfg.layer_types.last_mut() {
        *last = "full_attention".to_string();
    }
    Ok(cfg)
}

// ----------------------------------------------------------------------------
// Per-layer weights + cache
// ----------------------------------------------------------------------------

/// Per-layer weight bundle. A dense linear is pre-transposed to `[in, out]`
/// form so the matmul hot path is contiguous; a packed one keeps the
/// checkpoint's `[out, in]` and transposes in-kernel. [`LayerWeight`] hides
/// the difference from every call site.
pub struct PerLayerInputWeights {
    pub embed_w: EmbeddingWeight,
    pub projection_w: LayerWeight,
    pub projection_norm_w: InlineArray,
    pub embed_scale_scalar: InlineArray,
    pub projection_scale_scalar: InlineArray,
    pub input_scale_scalar: InlineArray,
    pub vocab_size: i32,
    pub hidden_size: i32,
}

pub struct PerLayerGateWeights {
    pub gate_w: LayerWeight,
    pub projection_w: LayerWeight,
    pub post_norm_w: InlineArray,
}

/// Where a layer's keys and values come from.
///
/// A Gemma 4 layer either projects its own K/V or reuses an earlier layer's
/// cache. The E2B/E4B checkpoints ship *no* `k_proj`, `v_proj` or `k_norm` for
/// a shared layer, so the two cases carry different tensors and this keeps the
/// combination that cannot exist out of the type.
// One per layer and never moved after load, so the size difference between the
// variants costs nothing worth a `Box` indirection on the decode path. Same
// reasoning as `LayerWeight` and `EmbeddingWeight` in `native_weight`.
#[allow(clippy::large_enum_variant)]
pub enum AttentionKv {
    /// The layer projects its own keys and values.
    Projected {
        k_w: LayerWeight,
        k_norm_w: InlineArray,
        /// `None` for full-attention layers under `attention_k_eq_v`, where
        /// values are the raw k_proj output taken *before* `k_norm`, so there
        /// is no `v_proj` to load.
        v_w: Option<LayerWeight>,
    },
    /// The layer reuses the cache of the last non-shared layer with the same
    /// attention type, and projects queries only.
    Shared { source_layer: usize },
}

impl AttentionKv {
    /// The layer whose cache this one reads, when it does not fill its own.
    pub fn source_layer(&self) -> Option<usize> {
        match self {
            Self::Projected { .. } => None,
            Self::Shared { source_layer } => Some(*source_layer),
        }
    }
}

/// One layer's routed-experts branch (`Gemma4TextRouter` + `Gemma4TextExperts`).
///
/// It runs beside the dense MLP rather than in place of it. Each branch is
/// post-normed on its own, the two are summed, and the layer's shared
/// `post_feedforward_layernorm` applies to the sum.
pub struct MoeWeights {
    pub router_proj_w: LayerWeight,
    /// The router's per-channel `scale` times `hidden_size^-0.5`, folded once.
    pub router_scale: InlineArray,
    /// `[num_experts]`, multiplied into each routed weight after renormalising.
    pub per_expert_scale: InlineArray,
    /// Stacked over experts, `[E, ...]`, dense or packed.
    pub gate_w: LayerWeight,
    pub up_w: LayerWeight,
    pub down_w: LayerWeight,
    /// Applied to the expert branch's input, which is the raw residual.
    pub pre_ffn_norm_2_w: InlineArray,
    /// Post-norms for the dense branch (`_1`) and the expert branch (`_2`).
    pub post_ffn_norm_1_w: InlineArray,
    pub post_ffn_norm_2_w: InlineArray,
    pub top_k: i32,
}

pub struct LayerWeights {
    pub input_norm_w: InlineArray,
    pub q_w: LayerWeight,
    pub kv: AttentionKv,
    pub o_w: LayerWeight,
    pub q_norm_w: InlineArray,
    pub post_attn_norm_w: InlineArray,
    pub pre_ffn_norm_w: InlineArray,
    pub gate_w: LayerWeight,
    pub up_w: LayerWeight,
    pub down_w: LayerWeight,
    pub post_ffn_norm_w: InlineArray,
    pub layer_scalar: InlineArray,
    pub per_layer_gate: Option<PerLayerGateWeights>,
    /// The routed-experts branch, on `enable_moe_block` checkpoints.
    pub moe: Option<MoeWeights>,
    /// Precomputed inverse-frequency array for partial RoPE (full layers).
    /// `None` for full-rotation sliding layers.
    pub rope_freqs: Option<InlineArray>,

    // Per-layer attributes used by the forward step.
    pub is_full_attention: bool,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_base: f32,
    pub rope_dims: i32,
    pub sliding_window: Option<i32>,
}

/// Full model weight bundle. `lm_head_w` is `None` when the model ties
/// embeddings (the norm step uses `embed_w` directly).
pub struct NativeWeights {
    pub embed_w: EmbeddingWeight,
    pub per_layer_inputs: Option<PerLayerInputWeights>,
    pub final_norm_w: InlineArray,
    pub lm_head_w: Option<LayerWeight>,
    pub layers: Vec<LayerWeights>,
    pub model_dtype: i32,
    pub config: Gemma4Config,
    /// `sqrt(hidden_size)` pre-cast to `model_dtype`. Broadcasting against
    /// a bf16 embedding must NOT promote — an f32 scalar here would turn
    /// the entire hidden state f32 and force every downstream matmul to
    /// cast its weights per-op. Owned here so the forward step can reuse
    /// it without per-step construction.
    pub embed_scale_scalar: InlineArray,
}

/// Slots a sliding layer's decode buffer holds beyond the window it keeps, so
/// that it is moved forward once per this many tokens rather than every one.
pub const SLIDING_SLACK: i32 = 256;

/// Per-layer KV cache: keys and values `[B, n_kv, L, D]` where slot `j` holds
/// absolute position `base + j`.
///
/// A layer that attends to everything (`window` is `None`) keeps every key,
/// with `base` 0 and `L` growing in
/// [`CHUNK_TOKENS`](crate::native_common::kv_cache::CHUNK_TOKENS)-token
/// chunks. A sliding layer never reads further back than its window, so it
/// keeps the window and a little more, and moves `base` forward as the
/// prompt and the decode go on: the cache of a long context is the full
/// layers' keys plus `window` per sliding layer, not every key of every layer.
pub struct LayerCache {
    pub keys: Option<InlineArray>,
    pub values: Option<InlineArray>,
    /// Tokens this layer has been fed: the absolute position of the next one.
    pub offset: usize,
    /// Absolute position of slot 0.
    pub base: usize,
    /// How far back the layer attends, `None` for a full-attention layer.
    pub window: Option<i32>,
    /// Slots a sliding layer's decode buffer holds past its window; see
    /// [`SLIDING_SLACK`]. A field so a test can make the buffer move often.
    pub slack: i32,
}

impl LayerCache {
    fn new(window: Option<i32>) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            base: 0,
            window,
            slack: SLIDING_SLACK,
        }
    }

    /// Slots written: the tokens from `base` to `offset`.
    fn filled(&self) -> i32 {
        (self.offset - self.base) as i32
    }

    /// Make room for `tokens` more tokens in the buffer without changing what
    /// the layer attends to.
    ///
    /// A full layer grows its buffer. A sliding layer whose buffer is full
    /// drops what the window no longer reaches and moves `base` forward,
    /// keeping the last `window` keys (the next token needs `window - 1` of
    /// them) in a buffer with `max(tokens, slack)` free slots after them.
    fn make_room(&mut self, batch: i32, kv_heads: i32, head_dim: i32, tokens: i32, dtype: i32) {
        let Some(window) = self.window else {
            crate::native_common::kv_cache::alloc_or_grow_kv(
                crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
                &mut self.keys,
                &mut self.values,
                batch,
                kv_heads,
                self.offset as i32 + tokens,
                head_dim,
                dtype,
            );
            return;
        };
        let filled = self.filled();
        let capacity = self.keys.as_ref().map_or(0, |k| k.dim(2));
        if self.keys.is_some() && filled + tokens <= capacity {
            return;
        }
        let keep = filled.min(window);
        let capacity = keep + tokens.max(self.slack);
        let rebase = |old: Option<InlineArray>| -> InlineArray {
            let fresh = InlineArray::zeros(&[batch, kv_heads, capacity - keep, head_dim], dtype);
            match old {
                Some(old) if keep > 0 => old
                    .slice(
                        &[0, 0, filled - keep, 0],
                        &[batch, kv_heads, filled, head_dim],
                    )
                    .kv_cache_append(&fresh, 2),
                _ => InlineArray::zeros(&[batch, kv_heads, capacity, head_dim], dtype),
            }
        };
        self.keys = Some(rebase(self.keys.take()));
        self.values = Some(rebase(self.values.take()));
        self.base += (filled - keep) as usize;
    }
}

/// Full-model cache. `rope_offset` mirrors the layer offsets — for Gemma 4
/// all layers share the same position counter.
pub struct NativeCache {
    pub layers: Vec<LayerCache>,
    pub rope_offset: i32,
}

impl NativeCache {
    /// Evaluate every layer's KV buffer and **trim** it before decode starts:
    /// a full layer down to the tokens it holds, a sliding layer to its window.
    /// That drops the unused trailing slots of the prefill allocation, so
    /// decode-time SDPA doesn't process them under a validity mask every step,
    /// and the keys a window can no longer reach.
    /// Mirrors `qwen3_native::NativeCache::eval_and_detach_states`.
    pub fn eval_and_detach_states(&mut self) {
        for layer in self.layers.iter_mut() {
            let filled = layer.filled();
            if filled <= 0 {
                continue;
            }
            let keep = layer.window.map_or(filled, |w| filled.min(w));
            let dropped = filled - keep;
            let trim = |a: InlineArray| {
                let trimmed = if dropped > 0 || filled < a.dim(2) {
                    a.slice(&[0, 0, dropped, 0], &[a.dim(0), a.dim(1), filled, a.dim(3)])
                } else {
                    a
                };
                trimmed.async_eval_ref();
                trimmed
            };
            layer.keys = layer.keys.take().map(trim);
            layer.values = layer.values.take().map(trim);
            layer.base += dropped as usize;
        }
    }

    /// Prepare every layer's buffer for a decode run of `additional_tokens`
    /// tokens. A full layer grows to exactly `offset + additional_tokens`
    /// slots, so the decode loop never grows it and per-step SDPA only
    /// processes the tokens held instead of a buffer rounded up to the next
    /// 256. A sliding layer gets room for as many as its slack holds (the
    /// decode moves it forward when that runs out).
    pub fn reserve_decode_inputs(&mut self, additional_tokens: i32, dtype: i32) {
        if additional_tokens <= 0 {
            return;
        }
        for layer in self.layers.iter_mut() {
            let (Some(keys), Some(values)) = (layer.keys.as_ref(), layer.values.as_ref()) else {
                continue;
            };
            let (b, nkv, hd_k, hd_v) = (keys.dim(0), keys.dim(1), keys.dim(3), values.dim(3));
            if layer.window.is_some() {
                layer.make_room(b, nkv, hd_k, additional_tokens.min(layer.slack), dtype);
                continue;
            }
            let current = keys.dim(2);
            let target = layer.offset as i32 + additional_tokens;
            if target <= current {
                continue;
            }
            let extend = target - current;
            let ext_k = InlineArray::zeros(&[b, nkv, extend, hd_k], dtype);
            let ext_v = InlineArray::zeros(&[b, nkv, extend, hd_v], dtype);
            layer.keys = layer.keys.take().map(|k| k.kv_cache_append(&ext_k, 2));
            layer.values = layer.values.take().map(|v| v.kv_cache_append(&ext_v, 2));
        }
    }
}

pub fn build_cache(_weights: &NativeWeights, config: &Gemma4Config) -> NativeCache {
    NativeCache {
        layers: (0..config.num_hidden_layers as usize)
            .map(|i| {
                LayerCache::new((!config.is_full_attention(i)).then_some(config.sliding_window))
            })
            .collect(),
        rope_offset: 0,
    }
}

// ----------------------------------------------------------------------------
// Loader
// ----------------------------------------------------------------------------

/// One Gemma 4 layer's rotation, as both engines run it: the channels it
/// rotates, and for a partial (`proportional`) layer the `[head_dim / 2]`
/// period table `fast::rope` takes, `+inf` past the rotated pairs.
#[derive(Debug, Clone)]
pub struct Gemma4LayerRope {
    /// `rope_theta`.
    pub base: f32,
    /// `2 * int(partial_rotary_factor * head_dim / 2)`.
    pub rotated_dims: i32,
    /// The period table, when fewer than `head_dim` channels rotate.
    pub periods: Option<Vec<f32>>,
}

/// Resolve one layer's `rope_parameters` entry (`full_attention` or
/// `sliding_attention`) through [`crate::rope`].
///
/// Gemma 4 runs `default` (sliding layers) and `proportional` (global
/// layers: the exponent over the whole head, the tail at frequency 0).
/// Anything else, a `default` that is partial, or a proportional `factor`
/// is refused by name rather than rotated some other way.
pub fn gemma4_layer_rope(
    head_dim: i32,
    rope_type: Option<&str>,
    theta: f32,
    partial_rotary_factor: f32,
    factor: Option<f32>,
    is_full: bool,
) -> RuntimeResult<Gemma4LayerRope> {
    let rope_type = rope_type.unwrap_or(if is_full { "proportional" } else { "default" });
    let mut params = serde_json::json!({
        "rope_type": rope_type,
        "rope_theta": theta,
        "partial_rotary_factor": partial_rotary_factor,
    });
    if let Some(factor) = factor {
        params["factor"] = factor.into();
    }
    let rotary = crate::rope::Rotary::from_config(
        head_dim,
        crate::rope::RopeConfig {
            rope_parameters: Some(&params),
            ..crate::rope::RopeConfig::default()
        },
        theta as f64,
        1.0,
    )
    .map_err(|e| RuntimeError::Config(format!("Gemma 4 rope_parameters: {e}")))?;
    let rotated_dims =
        (2 * ((partial_rotary_factor * head_dim as f32) / 2.0) as i32).clamp(0, head_dim);
    match rotary.scaling {
        crate::rope::RopeScaling::Default if rotated_dims == head_dim => {}
        crate::rope::RopeScaling::Proportional { factor: 1.0 } => {}
        ref other => {
            return Err(RuntimeError::Config(format!(
                "Gemma 4 rope_parameters: rope_type {:?} with partial_rotary_factor \
             {partial_rotary_factor} and factor {factor:?} is not supported; Gemma 4 runs \
             \"default\" over the whole head and \"proportional\" without a factor",
                other.rope_type()
            )));
        }
    }
    let periods = (rotated_dims < head_dim).then(|| rotary.periods(0));
    Ok(Gemma4LayerRope {
        base: theta,
        rotated_dims,
        periods,
    })
}

fn apply_gemma4_partial_rope(
    x: &InlineArray,
    head_dim: i32,
    rotated_dims: i32,
    base: f32,
    offset: i32,
    partial_freqs: Option<&InlineArray>,
) -> InlineArray {
    if rotated_dims == 0 {
        return x.clone();
    }
    if rotated_dims == head_dim {
        return x.rope(head_dim, false, base, 1.0, offset);
    }
    if let Some(freqs) = partial_freqs {
        return x.rope_with_freqs(head_dim, false, 1.0, offset, freqs);
    }

    let shape = x.shape();
    debug_assert_eq!(shape.len(), 4);
    let b = shape[0];
    let h = shape[1];
    let l = shape[2];
    let half = head_dim / 2;
    let rot_half = rotated_dims / 2;

    let left = x.slice(&[0, 0, 0, 0], &[b, h, l, half]);
    let right = x.slice(&[0, 0, 0, half], &[b, h, l, head_dim]);
    let left_rot = left.slice(&[0, 0, 0, 0], &[b, h, l, rot_half]);
    let right_rot = right.slice(&[0, 0, 0, 0], &[b, h, l, rot_half]);
    let rotated_input = left_rot.concatenate_2(&right_rot, -1);
    let effective_base = base.powf(rotated_dims as f32 / head_dim as f32);
    let rotated = rotated_input.rope(rotated_dims, false, effective_base, 1.0, offset);
    let new_left_rot = rotated.slice(&[0, 0, 0, 0], &[b, h, l, rot_half]);
    let new_right_rot = rotated.slice(&[0, 0, 0, rot_half], &[b, h, l, rotated_dims]);
    let left_tail = left.slice(&[0, 0, 0, rot_half], &[b, h, l, half]);
    let right_tail = right.slice(&[0, 0, 0, rot_half], &[b, h, l, half]);
    let new_left = new_left_rot.concatenate_2(&left_tail, -1);
    let new_right = new_right_rot.concatenate_2(&right_tail, -1);
    new_left.concatenate_2(&new_right, -1)
}

/// Which keys each query of a sliding-window layer may attend to: a boolean
/// `[1, 1, query_len, key_len]` that is true where the key's absolute position
/// is no later than the query's and less than `window` behind it.
///
/// The queries sit at absolute positions `query_start..query_start + query_len`
/// and the keys at `key_start..key_start + key_len`, so a pass can hand
/// attention only the slice of the cache its window reaches.
fn sliding_window_mask(
    query_start: i32,
    query_len: i32,
    key_start: i32,
    key_len: i32,
    window: i32,
) -> InlineArray {
    let int32 = Dtype::Int32.as_i32();
    let queries = InlineArray::arange(query_len, int32)
        .add(&InlineArray::from_i32(query_start))
        .reshape(&[query_len, 1]);
    let keys = InlineArray::arange(key_len, int32)
        .add(&InlineArray::from_i32(key_start))
        .reshape(&[1, key_len]);
    let not_after = keys.less_equal(&queries);
    let within = keys.greater(&queries.subtract(&InlineArray::from_i32(window)));
    crate::compat::ops::logical_and(&not_after, &within).reshape(&[1, 1, query_len, key_len])
}

/// Causal attention of `queries` (`[B, H, S, D]`, at absolute positions
/// `prev..prev + S`) over `keys` and `values`, which hold the absolute
/// positions from `key_start` up to `prev + S`, with `window` limiting how far
/// back a sliding layer reaches.
///
/// A full layer takes MLX's causal mode over all of its keys (`key_start` 0).
/// A sliding layer slices to the keys its first query's window reaches and
/// masks the rest, so a pass over a long prompt attends `window + S` keys per
/// query instead of all `prev + S`; a mask handed to the fused kernel is
/// evaluated over every key it is given, not skipped by block as causal mode
/// is.
fn prefill_attention(
    queries: &InlineArray,
    keys: &InlineArray,
    values: &InlineArray,
    prev: i32,
    key_start: i32,
    window: Option<i32>,
) -> InlineArray {
    let (b, kv_heads, len, head_dim) = (keys.dim(0), keys.dim(1), keys.dim(2), keys.dim(3));
    let s = queries.dim(2);
    let next = key_start + len;
    debug_assert_eq!(next, prev + s, "the keys end where this pass ends");
    // Query `prev` reaches back `window - 1` keys; later queries reach later
    // keys.
    let first = window.map_or(key_start, |w| (prev - w + 1).max(0));
    debug_assert!(
        first >= key_start,
        "the keys start before the window's reach"
    );
    let slice = |a: &InlineArray| {
        if first == key_start {
            a.clone()
        } else {
            a.slice(&[0, 0, first - key_start, 0], &[b, kv_heads, len, head_dim])
        }
    };
    match window {
        // Some query's window leaves out a key the causal mask would allow.
        // (When the window covers everything up to the last query, it is the
        // causal mask.)
        Some(window) if prev + s > window => {
            let mask = sliding_window_mask(prev, s, first, next - first, window);
            queries.sdpa_with_mask(&slice(keys), &slice(values), 1.0, Some(&mask))
        }
        _ => crate::decode::sdpa_causal_like_mlx(queries, keys, values, 1.0, s),
    }
}

fn compute_per_layer_inputs(
    weights: &NativeWeights,
    input_ids: &InlineArray,
    inputs_embeds: &InlineArray,
) -> Option<InlineArray> {
    let ple = weights.per_layer_inputs.as_ref()?;
    let per_layer_dim = weights.config.per_layer_input_dim();
    let safe_token_ids = if ple.vocab_size == weights.config.vocab_size {
        input_ids.clone()
    } else {
        let ge_zero = input_ids.greater_equal(&InlineArray::from_i32(0));
        let lt_vocab = input_ids.less(&InlineArray::from_i32(ple.vocab_size));
        let mask = crate::compat::ops::logical_and(&ge_zero, &lt_vocab);
        mask.where_cond(input_ids, &input_ids.zeros_like())
    };
    let per_layer_embeds = ple
        .embed_w
        .lookup(&safe_token_ids)
        .multiply(&ple.embed_scale_scalar)
        .reshape(&[
            input_ids.dim(0),
            input_ids.dim(1),
            weights.config.num_hidden_layers,
            per_layer_dim,
        ]);
    let projection = ple
        .projection_w
        .matmul_from(inputs_embeds)
        .multiply(&ple.projection_scale_scalar)
        .reshape(&[
            input_ids.dim(0),
            input_ids.dim(1),
            weights.config.num_hidden_layers,
            per_layer_dim,
        ])
        .rms_norm(Some(&ple.projection_norm_w), weights.config.rms_norm_eps);
    Some(
        projection
            .add(&per_layer_embeds)
            .multiply(&ple.input_scale_scalar),
    )
}

fn layer_per_input(per_layer_inputs: &InlineArray, layer_idx: usize) -> InlineArray {
    let b = per_layer_inputs.dim(0);
    let s = per_layer_inputs.dim(1);
    let d = per_layer_inputs.dim(3);
    per_layer_inputs
        .slice(
            &[0, 0, layer_idx as i32, 0],
            &[b, s, layer_idx as i32 + 1, d],
        )
        .squeeze(2)
}

fn apply_per_layer_input_block(
    hidden: &InlineArray,
    layer: &LayerWeights,
    layer_per_input: &InlineArray,
    eps: f32,
) -> InlineArray {
    let Some(ple) = layer.per_layer_gate.as_ref() else {
        return hidden.clone();
    };
    if hidden.dim(1) == 1 {
        return InlineArray::compiled_gemma4_per_layer_input_block(
            hidden,
            layer_per_input,
            &ple.gate_w,
            &ple.projection_w,
            &ple.post_norm_w,
            eps,
        );
    }
    let residual = hidden.clone();
    let gate = ple.gate_w.matmul_from(hidden);
    // `gelu(gate) * layer_input`, in the gate's dtype: `gelu_approx` builds
    // its polynomial from f32 scalars, which promoted a bf16 residual stream
    // to f32 at the first layer with per-layer inputs and kept it there for
    // the rest of every multi-token pass.
    let projected = ple
        .projection_w
        .matmul_from(&InlineArray::fused_geglu_tanh(&gate, layer_per_input))
        .rms_norm(Some(&ple.post_norm_w), eps);
    residual.add(&projected)
}

fn shared_kv_attention_forward(
    hidden: &InlineArray,
    layer: &LayerWeights,
    source_cache: &LayerCache,
    rope_offset: i32,
    eps: f32,
) -> InlineArray {
    let b = hidden.dim(0);
    let s = hidden.dim(1);
    let cache_len = source_cache.offset as i32;
    if s == 1 {
        return InlineArray::compiled_gemma4_shared_attn_decode(
            hidden,
            &layer.input_norm_w,
            &layer.q_w,
            &layer.o_w,
            &layer.q_norm_w,
            &layer.post_attn_norm_w,
            layer.rope_freqs.as_ref(),
            source_cache
                .keys
                .as_ref()
                .expect("Gemma 4 native: missing shared KV source keys"),
            source_cache
                .values
                .as_ref()
                .expect("Gemma 4 native: missing shared KV source values"),
            cache_len,
            rope_offset,
            source_cache.base as i32,
            layer.n_heads,
            layer.n_kv_heads,
            layer.head_dim,
            eps,
            eps,
            eps,
            layer.sliding_window.unwrap_or(0),
            layer.rope_base,
            layer.rope_dims,
        );
    }
    // The source's keys from its `base` to this pass's end.
    let filled = source_cache.filled();
    let source_keys = source_cache
        .keys
        .as_ref()
        .expect("Gemma 4 native: missing shared KV source keys")
        .slice(
            &[0, 0, 0, 0],
            &[b, layer.n_kv_heads, filled, layer.head_dim],
        );
    let source_values = source_cache
        .values
        .as_ref()
        .expect("Gemma 4 native: missing shared KV source values")
        .slice(
            &[0, 0, 0, 0],
            &[b, layer.n_kv_heads, filled, layer.head_dim],
        );

    let normed = hidden.rms_norm(Some(&layer.input_norm_w), eps);
    let queries = layer
        .q_w
        .matmul_from(&normed)
        .reshape(&[b, s, layer.n_heads, layer.head_dim])
        .rms_norm(Some(&layer.q_norm_w), eps)
        .transpose_axes(&[0, 2, 1, 3]);
    let queries = apply_gemma4_partial_rope(
        &queries,
        layer.head_dim,
        layer.rope_dims,
        layer.rope_base,
        rope_offset,
        layer.rope_freqs.as_ref(),
    );

    let attn_out = prefill_attention(
        &queries,
        &source_keys,
        &source_values,
        rope_offset,
        source_cache.base as i32,
        layer.sliding_window,
    );

    layer
        .o_w
        .matmul_from(&attn_out.transpose_axes(&[0, 2, 1, 3]).reshape(&[
            b,
            s,
            layer.n_heads * layer.head_dim,
        ]))
        .rms_norm(Some(&layer.post_attn_norm_w), eps)
}

/// A layer that projects its own keys and values, attending over a pass of
/// more than one token (a prefill chunk, a verified draft block): input norm,
/// projections, per-head norms, RoPE, the cache write, attention, the output
/// projection and the post-attention norm. `rope_offset` is the number of
/// tokens the cache already holds.
///
/// This is what `compiled_gemma4_attn_block` computes, per op. That block is
/// traced per cache length and prefill shape, so every prompt of a new length
/// compiled and kept a graph for good, and it scatters the new rows with
/// `put_along_axis` and masks over the whole cache; here the write is a slice
/// update and attention sees only the keys in its reach ([`prefill_attention`]).
/// A single token goes through the compiled block, whose shape is fixed for the
/// length of a decode.
#[allow(clippy::too_many_arguments)]
fn projected_attention_multi(
    hidden: &InlineArray,
    layer: &LayerWeights,
    k_w: &LayerWeight,
    k_norm_w: &InlineArray,
    v_w: Option<&LayerWeight>,
    layer_cache: &mut LayerCache,
    rope_offset: i32,
    dtype: i32,
    eps: f32,
) -> InlineArray {
    let (b, s) = (hidden.dim(0), hidden.dim(1));
    let (n_heads, n_kv, head_dim) = (layer.n_heads, layer.n_kv_heads, layer.head_dim);
    let next = rope_offset + s;

    let normed = hidden.rms_norm(Some(&layer.input_norm_w), eps);
    let queries = layer
        .q_w
        .matmul_from(&normed)
        .reshape(&[b, s, n_heads, head_dim])
        .rms_norm(Some(&layer.q_norm_w), eps)
        .transpose_axes(&[0, 2, 1, 3]);
    let raw_keys = k_w.matmul_from(&normed);
    // `attention_k_eq_v`: the values are the raw key projection, before k_norm.
    let raw_values = match v_w {
        Some(v_w) => v_w.matmul_from(&normed),
        None => raw_keys.clone(),
    };
    let keys = raw_keys
        .reshape(&[b, s, n_kv, head_dim])
        .rms_norm(Some(k_norm_w), eps)
        .transpose_axes(&[0, 2, 1, 3]);
    let values = raw_values
        .reshape(&[b, s, n_kv, head_dim])
        .rms_norm(None, eps)
        .transpose_axes(&[0, 2, 1, 3]);

    let rotate = |x: &InlineArray| {
        apply_gemma4_partial_rope(
            x,
            head_dim,
            layer.rope_dims,
            layer.rope_base,
            rope_offset,
            layer.rope_freqs.as_ref(),
        )
    };
    let (queries, keys) = (rotate(&queries), rotate(&keys));
    // The cache holds the model's dtype whatever a pass computed in: keys and
    // values written in f32 would be read as f32 by every decode step.
    let (keys, values) = (keys.as_dtype(dtype), values.as_dtype(dtype));

    // The keys this pass attends over, and the absolute position of the first.
    let (attended_keys, attended_values, key_start) = match layer.sliding_window {
        // A sliding layer reaches back `window - 1` keys from this pass's
        // first query: the tail of what it holds, then the new keys. The
        // cache keeps exactly those, so it holds a window and a pass, not the
        // prompt.
        Some(window) => {
            let first = (rope_offset - window + 1).max(0);
            let base = layer_cache.base as i32;
            let reach = |old: Option<InlineArray>, new: &InlineArray| match old {
                Some(old) if rope_offset > first => old
                    .slice(
                        &[0, 0, first - base, 0],
                        &[b, n_kv, rope_offset - base, head_dim],
                    )
                    .kv_cache_append(new, 2),
                _ => new.clone(),
            };
            let all_keys = reach(layer_cache.keys.take(), &keys);
            let all_values = reach(layer_cache.values.take(), &values);
            layer_cache.keys = Some(all_keys.clone());
            layer_cache.values = Some(all_values.clone());
            layer_cache.base = first as usize;
            (all_keys, all_values, first)
        }
        // A full layer writes the pass after what it holds and attends over
        // all of it.
        None => {
            layer_cache.make_room(b, n_kv, head_dim, s, dtype);
            let (start, stop) = ([0, 0, rope_offset, 0], [b, n_kv, next, head_dim]);
            let all_keys = layer_cache
                .keys
                .take()
                .expect("make_room allocated the keys")
                .slice_set(&keys, &start, &stop);
            let all_values = layer_cache
                .values
                .take()
                .expect("make_room allocated the values")
                .slice_set(&values, &start, &stop);
            let valid = |a: &InlineArray| a.slice(&[0, 0, 0, 0], &stop);
            let attended = (valid(&all_keys), valid(&all_values), 0);
            layer_cache.keys = Some(all_keys);
            layer_cache.values = Some(all_values);
            attended
        }
    };
    layer_cache.offset = next as usize;

    let attended = prefill_attention(
        &queries,
        &attended_keys,
        &attended_values,
        rope_offset,
        key_start,
        layer.sliding_window,
    );

    layer
        .o_w
        .matmul_from(
            &attended
                .transpose_axes(&[0, 2, 1, 3])
                .reshape(&[b, s, n_heads * head_dim]),
        )
        .rms_norm(Some(&layer.post_attn_norm_w), eps)
}

/// Normalise a Gemma 4 checkpoint key onto the bare `model.*` text-tower
/// layout, returning `None` for vision / audio entries the text tower never
/// reads.
///
/// Three layouts ship in the wild:
///
/// * `model.*` — text-only export (`gemma4_text`).
/// * `model.language_model.*` — transformers multimodal export, e.g.
///   `google/gemma-4-31B-it`, with the towers under `model.vision_tower.*`.
/// * `language_model.model.*` — unified export (`gemma4_unified`), e.g.
///   `mlx-community/gemma-4-12B-it-bf16`, where the towers sit *beside* the
///   text backbone as `vision_embedder.*`, `embed_vision.*`, `embed_audio.*`
///   rather than under a shared `model.` root.
///
/// Shared with `pmetal_models::architectures::gemma4::load_gemma4_weights` so
/// the two loaders cannot drift on which checkpoints they accept.
pub fn normalize_checkpoint_key(key: &str) -> Option<String> {
    const NON_TEXT_TOWER: [&str; 6] = [
        "embed_vision",
        "embed_audio",
        "vision_tower",
        "vision_embedder",
        "audio_tower",
        "multi_modal_projector",
    ];

    let stripped = key
        .strip_prefix("model.language_model.")
        .map(|rest| format!("model.{rest}"))
        .or_else(|| key.strip_prefix("language_model.").map(str::to_string))
        .unwrap_or_else(|| key.to_string());

    if NON_TEXT_TOWER.iter().any(|tower| stripped.contains(tower)) {
        return None;
    }
    Some(stripped)
}

/// Load a Gemma 4 checkpoint into [`NativeWeights`]. Dense linears are
/// pre-transposed at load time so the per-step matmuls are contiguous.
pub fn load_model(
    model_dir: &std::path::Path,
    config: &Gemma4Config,
) -> RuntimeResult<NativeWeights> {
    // Shards come from the index rather than a directory glob. A glob picks up
    // whatever else ends in `.safetensors`, and on a non-APFS volume that
    // includes the `._model-0000N-of-…` AppleDouble sidecars macOS writes
    // beside every file, which fail with "Invalid json header length".
    let mut shard_files = crate::native_loader::discover_safetensors_shards(model_dir)?;
    shard_files.sort();

    // Collect every `(key, array)` pair across shards, normalised onto the
    // bare `model.*` text-tower layout so text-only, multimodal and unified
    // checkpoints all load through the same key set.
    let mut raw: std::collections::HashMap<String, InlineArray> = std::collections::HashMap::new();
    for shard in &shard_files {
        let shard_str = shard.to_str().ok_or_else(|| {
            RuntimeError::Config(format!("shard path is not valid UTF-8: {shard:?}"))
        })?;
        let Some(pairs) = bridge::load_safetensors_shard(shard_str) else {
            return Err(RuntimeError::Config(format!(
                "failed to load safetensors shard {shard_str}"
            )));
        };
        for (key, arr) in pairs {
            let Some(stripped) = normalize_checkpoint_key(&key) else {
                continue;
            };
            // Two layouts normalising onto one key would resolve silently in
            // whatever order the shards happened to be read.
            if raw.insert(stripped.clone(), arr).is_some() {
                return Err(RuntimeError::Config(format!(
                    "Gemma 4 native: two checkpoint keys both normalise to {stripped}"
                )));
            }
        }
    }

    // The `quantization` block, if any, read the way MLX-format loaders read it. Its
    // module paths get the same normalisation as the weight keys, or every
    // per-module override would miss and the 8-bit tensors in a QAT
    // checkpoint would be decoded as 4-bit.
    let quantization = crate::native_loader::load_mlx_quantization(model_dir, |path| {
        normalize_checkpoint_key(path)
    })?;
    // How to read one module's packed tensors: its own override, else the
    // file-level block, else MLX's affine defaults. Only ever consulted for a
    // module that has scales, so the last case decodes nothing.
    let params_for = |module: &str| -> QuantParams {
        quantization
            .as_ref()
            .and_then(|q| q.params_for(module).or_else(|| q.default_params()))
            .unwrap_or_else(|| QuantParams::defaults_for(crate::QuantizedMode::Affine))
    };
    // `params_for`, checked against the tensors the module stores when it is
    // packed: a block and weights that disagree about the scheme are an error,
    // not a model that decodes noise.
    let params_for_tensors = |module: &str,
                              scales: &Option<InlineArray>,
                              biases: &Option<InlineArray>|
     -> RuntimeResult<QuantParams> {
        let params = params_for(module);
        if scales.is_some() {
            params
                .for_tensors(module, biases.is_some())
                .map_err(|e| RuntimeError::Config(format!("Gemma 4 native: {e}")))
        } else {
            Ok(params)
        }
    };

    let take = |map: &mut std::collections::HashMap<String, InlineArray>,
                key: &str|
     -> RuntimeResult<InlineArray> {
        map.remove(key)
            .ok_or_else(|| RuntimeError::Config(format!("Gemma 4 native: missing weight {key}")))
    };

    // ⚠️ Not from `embed_tokens.weight`: on a quantized checkpoint that is the
    // packed `uint32` payload, and deriving the trunk dtype from it makes
    // every scalar and the whole KV cache integer. The model then loads, runs
    // at the right speed, and decodes noise.
    let model_dtype = detect_model_dtype(|key| raw.get(key).map(|arr| arr.dtype_raw()));

    // NVIDIA ModelOpt projections, built up front from their own sidecars.
    // Their mode comes from the tensors: the config's `targets` names layer
    // classes rather than module paths, so `params_for` cannot say.
    let mut modelopt: std::collections::HashMap<String, LayerWeight> =
        std::collections::HashMap::new();
    for (module, weight) in crate::native_loader::take_modelopt_weights(&mut raw)? {
        let native = weight
            .into_native(model_dtype)
            .map_err(|e| RuntimeError::Config(format!("Gemma 4 native: {module}.weight: {e}")))?;
        modelopt.insert(module, native.into_layer_weight());
    }

    // One module's three tensors. `scales` decides the variant, which is
    // the MLX checkpoint rule too: a module is quantized exactly when
    // `{path}.scales` is in the weights.
    let mut take_proj = |map: &mut std::collections::HashMap<String, InlineArray>,
                         module: &str|
     -> RuntimeResult<LayerWeight> {
        if let Some(weight) = modelopt.remove(module) {
            return Ok(weight);
        }
        let weight = map.remove(&format!("{module}.weight")).ok_or_else(|| {
            RuntimeError::Config(format!("Gemma 4 native: missing weight {module}.weight"))
        })?;
        let scales = map.remove(&format!("{module}.scales"));
        let biases = map.remove(&format!("{module}.biases"));
        let params = params_for_tensors(module, &scales, &biases)?;
        Ok(LayerWeight::new(weight, scales, biases, params))
    };

    let embed_w = {
        let (scales, biases) = (
            raw.remove("model.embed_tokens.scales"),
            raw.remove("model.embed_tokens.biases"),
        );
        let params = params_for_tensors("model.embed_tokens", &scales, &biases)?;
        EmbeddingWeight::new(
            take(&mut raw, "model.embed_tokens.weight")?,
            scales,
            biases,
            params,
        )
    };
    let final_norm_w = take(&mut raw, "model.norm.weight")?;
    let lm_head_w = if config.tie_word_embeddings {
        None
    } else {
        Some(take_proj(&mut raw, "lm_head")?)
    };
    let per_layer_inputs = if config.uses_per_layer_inputs() {
        let total_ple_dim = config.num_hidden_layers * config.per_layer_input_dim();
        Some(PerLayerInputWeights {
            embed_w: {
                let module = "model.embed_tokens_per_layer";
                let (scales, biases) = (
                    raw.remove(&format!("{module}.scales")),
                    raw.remove(&format!("{module}.biases")),
                );
                let params = params_for_tensors(module, &scales, &biases)?;
                EmbeddingWeight::new(
                    take(&mut raw, &format!("{module}.weight"))?,
                    scales,
                    biases,
                    params,
                )
            },
            projection_w: take_proj(&mut raw, "model.per_layer_model_projection")?,
            projection_norm_w: take(&mut raw, "model.per_layer_projection_norm.weight")?,
            embed_scale_scalar: InlineArray::scalar_with_dtype(
                (config.per_layer_input_dim() as f32).sqrt(),
                model_dtype,
            ),
            projection_scale_scalar: InlineArray::scalar_with_dtype(
                (config.hidden_size as f32).powf(-0.5),
                model_dtype,
            ),
            input_scale_scalar: InlineArray::scalar_with_dtype(2.0f32.powf(-0.5), model_dtype),
            vocab_size: config.per_layer_input_vocab_size(),
            hidden_size: total_ple_dim,
        })
    } else {
        None
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
    let first_kv_shared_layer_idx = config.first_kv_shared_layer_idx();
    for idx in 0..config.num_hidden_layers as usize {
        let p = format!("model.layers.{idx}");
        let is_full = config.is_full_attention(idx);
        let head_dim = config.layer_head_dim(idx);
        let n_kv_heads = config.layer_num_kv_heads(idx);
        let use_k_eq_v = config.layer_uses_k_eq_v(idx);
        let rope = config.layer_rope(idx)?;
        let (rope_base, rope_dims) = (rope.base, rope.rotated_dims);
        let sliding_window = if is_full {
            None
        } else {
            Some(config.sliding_window)
        };
        let rope_freqs = rope
            .periods
            .map(|p| InlineArray::from_f32_slice(&p, &[p.len() as i32]));
        let shared_source = if idx >= first_kv_shared_layer_idx && first_kv_shared_layer_idx > 0 {
            Some(config.kv_shared_source_layer(idx).ok_or_else(|| {
                RuntimeError::Config(format!(
                    "Gemma 4 native: no KV-sharing source layer found for layer {idx} ({})",
                    if is_full {
                        "full_attention"
                    } else {
                        "sliding_attention"
                    }
                ))
            })?)
        } else {
            None
        };

        let input_norm_w = take(&mut raw, &format!("{p}.input_layernorm.weight"))?;
        let post_attn_norm_w = take(&mut raw, &format!("{p}.post_attention_layernorm.weight"))?;
        let pre_ffn_norm_w = take(&mut raw, &format!("{p}.pre_feedforward_layernorm.weight"))?;
        let post_ffn_norm_w = take(&mut raw, &format!("{p}.post_feedforward_layernorm.weight"))?;
        let q_norm_w = take(&mut raw, &format!("{p}.self_attn.q_norm.weight"))?;

        let q_w = take_proj(&mut raw, &format!("{p}.self_attn.q_proj"))?;
        // A KV-shared layer reads an earlier layer's cache, so the checkpoint
        // ships no k_proj, v_proj or k_norm for it at all. Asking for them is
        // what kept every E2B/E4B checkpoint from loading (#31).
        let kv = match shared_source {
            Some(source_layer) => AttentionKv::Shared { source_layer },
            None => AttentionKv::Projected {
                k_w: take_proj(&mut raw, &format!("{p}.self_attn.k_proj"))?,
                k_norm_w: take(&mut raw, &format!("{p}.self_attn.k_norm.weight"))?,
                v_w: if use_k_eq_v {
                    None
                } else {
                    Some(take_proj(&mut raw, &format!("{p}.self_attn.v_proj"))?)
                },
            },
        };
        let o_w = take_proj(&mut raw, &format!("{p}.self_attn.o_proj"))?;

        let gate_w = take_proj(&mut raw, &format!("{p}.mlp.gate_proj"))?;
        let up_w = take_proj(&mut raw, &format!("{p}.mlp.up_proj"))?;
        let down_w = take_proj(&mut raw, &format!("{p}.mlp.down_proj"))?;

        let moe = if config.uses_moe() {
            let experts = format!("{p}.experts");
            let (gate, up, down) =
                if let Some(gate_up) = raw.remove(&format!("{experts}.gate_up_proj")) {
                    let down = take(&mut raw, &format!("{experts}.down_proj"))?;
                    experts_from_fused(
                        &gate_up,
                        down,
                        config.moe_intermediate_size.unwrap_or_default(),
                    )
                } else if raw.contains_key(&format!("{experts}.switch_glu.gate_proj.weight")) {
                    // MLX-format checkpoints stack experts as `switch_glu.*`, `[E, out, in]`.
                    let mut stacked = |name: &str| -> RuntimeResult<LayerWeight> {
                        let module = format!("{experts}.switch_glu.{name}");
                        if raw.contains_key(&format!("{module}.scales")) {
                            return take_proj(&mut raw, &module);
                        }
                        // ⚠️ Not `take_proj`: its dense arm is `.t()`, which
                        // reverses every axis and would make this `[in, out, E]`.
                        let w = take(&mut raw, &format!("{module}.weight"))?;
                        Ok(LayerWeight::Dense(w.transpose_axes(&[0, 2, 1])))
                    };
                    (
                        stacked("gate_proj")?,
                        stacked("up_proj")?,
                        stacked("down_proj")?,
                    )
                } else {
                    // One module per expert, as NVIDIA ModelOpt ships them.
                    let num_experts = config.num_experts.unwrap_or_default();
                    let mut per_expert = |name: &str| -> RuntimeResult<LayerWeight> {
                        let mut parts = Vec::with_capacity(num_experts as usize);
                        for e in 0..num_experts {
                            parts.push(take_proj(&mut raw, &format!("{experts}.{e}.{name}"))?);
                        }
                        LayerWeight::stack(parts)
                            .map_err(|e| RuntimeError::Config(format!("{experts}.*.{name}: {e}")))
                    };
                    (
                        per_expert("gate_proj")?,
                        per_expert("up_proj")?,
                        per_expert("down_proj")?,
                    )
                };

            // `norm(x) * scale * hidden^-0.5`: the two constants fold into one.
            let root = (config.hidden_size as f32).powf(-0.5);
            let router_scale = take(&mut raw, &format!("{p}.router.scale"))?
                .as_dtype(Dtype::Float32.as_i32())
                .multiply(&InlineArray::scalar_with_dtype(
                    root,
                    Dtype::Float32.as_i32(),
                ))
                .as_dtype(model_dtype);
            Some(MoeWeights {
                router_proj_w: take_proj(&mut raw, &format!("{p}.router.proj"))?,
                router_scale,
                // Routing weights are fp32 until the experts use them.
                per_expert_scale: take(&mut raw, &format!("{p}.router.per_expert_scale"))?
                    .as_dtype(Dtype::Float32.as_i32()),
                gate_w: gate,
                up_w: up,
                down_w: down,
                pre_ffn_norm_2_w: take(
                    &mut raw,
                    &format!("{p}.pre_feedforward_layernorm_2.weight"),
                )?,
                post_ffn_norm_1_w: take(
                    &mut raw,
                    &format!("{p}.post_feedforward_layernorm_1.weight"),
                )?,
                post_ffn_norm_2_w: take(
                    &mut raw,
                    &format!("{p}.post_feedforward_layernorm_2.weight"),
                )?,
                top_k: config.top_k_experts.unwrap_or_default(),
            })
        } else {
            None
        };

        let per_layer_gate = if config.uses_per_layer_inputs() {
            Some(PerLayerGateWeights {
                gate_w: take_proj(&mut raw, &format!("{p}.per_layer_input_gate"))?,
                projection_w: take_proj(&mut raw, &format!("{p}.per_layer_projection"))?,
                post_norm_w: take(&mut raw, &format!("{p}.post_per_layer_input_norm.weight"))?,
            })
        } else {
            None
        };

        // Gemma 4 stores `layer_scalar` as an f32 scalar (the reference uses
        // `mx.ones((1,))` which defaults to f32). The hidden state is
        // bf16, so multiplying by an f32 scalar would promote the whole
        // residual to f32 and force every downstream layer's matmul to
        // cast its weights. Pre-cast to `model_dtype` once at load time.
        let layer_scalar = raw
            .remove(&format!("{p}.layer_scalar"))
            .unwrap_or_else(|| InlineArray::from_f32_slice(&[1.0], &[1]))
            .as_dtype(model_dtype);

        layers.push(LayerWeights {
            input_norm_w,
            q_w,
            kv,
            o_w,
            q_norm_w,
            post_attn_norm_w,
            pre_ffn_norm_w,
            gate_w,
            up_w,
            down_w,
            post_ffn_norm_w,
            layer_scalar,
            per_layer_gate,
            moe,
            rope_freqs,
            is_full_attention: is_full,
            n_heads: config.num_attention_heads,
            n_kv_heads,
            head_dim,
            rope_base,
            rope_dims,
            sliding_window,
        });
    }

    // A quantized module nothing asked for is a key this loader maps wrongly,
    // and its layer would otherwise have loaded without it.
    if !modelopt.is_empty() {
        let mut unused: Vec<&String> = modelopt.keys().collect();
        unused.sort();
        return Err(RuntimeError::Config(format!(
            "Gemma 4 native: quantized modules with no place in the model: {unused:?}"
        )));
    }

    // Pre-cast scalars once so the forward step never constructs an
    // f32 broadcast and quietly promotes the bf16 residual stream.
    let embed_scale_scalar = InlineArray::scalar_with_dtype(config.embed_scale(), model_dtype);

    let weights = NativeWeights {
        embed_w,
        per_layer_inputs,
        final_norm_w,
        lm_head_w,
        layers,
        model_dtype,
        config: config.clone(),
        embed_scale_scalar,
    };

    // Eagerly evaluate so load-time cost is paid here instead of on first
    // decode step. Matches qwen3_native's pattern.
    for l in weights.layers.iter() {
        l.q_w.async_eval_ref();
        if let AttentionKv::Projected { k_w, v_w, .. } = &l.kv {
            k_w.async_eval_ref();
            if let Some(v) = v_w {
                v.async_eval_ref();
            }
        }
        l.o_w.async_eval_ref();
        l.gate_w.async_eval_ref();
        l.up_w.async_eval_ref();
        l.down_w.async_eval_ref();
        if let Some(ref ple) = l.per_layer_gate {
            ple.gate_w.async_eval_ref();
            ple.projection_w.async_eval_ref();
        }
        // The stacked experts are still lazy `stack` graphs until evaluated.
        if let Some(ref moe) = l.moe {
            moe.router_proj_w.async_eval_ref();
            moe.router_scale.async_eval_ref();
            moe.per_expert_scale.async_eval_ref();
            moe.gate_w.async_eval_ref();
            moe.up_w.async_eval_ref();
            moe.down_w.async_eval_ref();
        }
    }
    if let Some(ref ple) = weights.per_layer_inputs {
        ple.embed_w.async_eval_ref();
        ple.projection_w.async_eval_ref();
    }
    weights.embed_w.async_eval_ref();
    weights.final_norm_w.async_eval_ref();

    Ok(weights)
}

// ----------------------------------------------------------------------------
// MoE branch
// ----------------------------------------------------------------------------

/// Hugging Face's fused expert layout as `(gate, up, down)`, each `[E, in, out]`.
///
/// `gate_up` is `[E, 2I, H]` with gate first (`Gemma4TextExperts` chunks it in
/// that order) and `down` is `[E, H, I]`.
///
/// ⚠️ Swapping gate and up passes the synthetic oracle: its weights are small
/// enough that GELU is linear there and `gelu(g)·u ≈ gelu(u)·g`. The order is
/// pinned by its own test instead.
fn experts_from_fused(
    gate_up: &InlineArray,
    down: InlineArray,
    intermediate: i32,
) -> (LayerWeight, LayerWeight, LayerWeight) {
    let dense = |w: InlineArray| LayerWeight::Dense(w.transpose_axes(&[0, 2, 1]));
    (
        dense(gate_up.index((.., 0..intermediate, ..))),
        dense(gate_up.index((.., intermediate..2 * intermediate, ..))),
        dense(down),
    )
}

/// `Gemma4TextRouter`: each row's `top_k` experts and their weights, both
/// `[S, top_k]`, weights in fp32.
///
/// A weight-less RMSNorm, the folded `scale · hidden^-0.5`, the projection,
/// then a softmax in fp32. The top-k probabilities are renormalised to sum to
/// one and multiplied by each chosen expert's `per_expert_scale`.
fn route_experts(moe: &MoeWeights, h_flat: &InlineArray, eps: f32) -> (InlineArray, InlineArray) {
    let scaled = h_flat.rms_norm(None, eps).multiply(&moe.router_scale);
    let probs = moe
        .router_proj_w
        .matmul_from(&scaled)
        .as_dtype(Dtype::Float32.as_i32())
        .softmax(-1);
    let (s, num_experts, k) = (probs.dim(0), probs.dim(1), moe.top_k);

    // ⚠️ `argpartition(-k)` puts the k largest in the last k slots. Taking the
    // first k instead routes every token to its least likely experts.
    let inds = probs
        .argpartition(-k, -1)
        .slice(&[0, num_experts - k], &[s, num_experts]);
    let top = probs.take_along_axis(&inds, -1);
    let top = top.divide(&top.sum_axis(-1, true));
    let per_expert = moe
        .per_expert_scale
        .take_axis(&inds.reshape(&[s * k]), 0)
        .reshape(&[s, k]);
    (inds, top.multiply(&per_expert))
}

/// A MoE layer's feed-forward output: the dense branch and the routed experts,
/// each post-normed, summed, then through the shared post-norm.
///
/// `h` is the post-attention residual, which the router and the experts read
/// directly. `dense` is the dense MLP's output before any post-norm.
fn moe_feed_forward(
    moe: &MoeWeights,
    h: &InlineArray,
    dense: &InlineArray,
    post_ffn_norm_w: &InlineArray,
    eps: f32,
) -> InlineArray {
    let shape = h.shape();
    let hidden = shape[shape.len() - 1];
    let h_flat = h.reshape(&[-1, hidden]);

    let (inds, weights) = route_experts(moe, &h_flat, eps);
    let experts = crate::native_moe::switch_glu(
        &h_flat.rms_norm(Some(&moe.pre_ffn_norm_2_w), eps),
        &moe.gate_w,
        &moe.up_w,
        &moe.down_w,
        &inds,
        &weights.as_dtype(h.dtype_raw()),
        InlineArray::fused_geglu_tanh,
    )
    .reshape(shape);

    let dense = dense.rms_norm(Some(&moe.post_ffn_norm_1_w), eps);
    let experts = experts.rms_norm(Some(&moe.post_ffn_norm_2_w), eps);
    dense.add(&experts).rms_norm(Some(post_ffn_norm_w), eps)
}

// ----------------------------------------------------------------------------
// Forward step
// ----------------------------------------------------------------------------

// Tanh-approx GEGLU runs through the shapeless-compiled
// `InlineArray::fused_geglu_tanh` helper (see bridge_compiled.cpp).
// Its scalar constants are cast to `gate.dtype()` inside the compiled
// lambda, so bf16 inputs stay bf16 and we avoid the hidden-state
// promotion that would otherwise force every matmul's weights to be
// cast from bf16→f32 on the fly.

/// One full Gemma 4 forward pass: embedding scale → N decoder layers →
/// final norm → tied LM head → logit softcap. Signature matches the
/// `forward_step` contract `crate::decode::*` expects: takes `weights`,
/// `input_ids` as an `[B, T]` int array, and a mutable `cache`.
///
/// Returns the logits of every position, `[B, T, vocab]`; a prefill that only
/// wants the last one calls [`forward_step_last`].
pub fn forward_step(
    weights: &NativeWeights,
    input_ids: &InlineArray,
    cache: &mut NativeCache,
) -> InlineArray {
    logits_from_hidden(weights, &forward_hidden(weights, input_ids, cache))
}

/// [`forward_step`]'s logits for the last position alone, `[B, vocab]`.
///
/// The LM head and its softcap run over one row instead of `T`: over Gemma 4's
/// 262144-token vocabulary the logits of a long prompt are gigabytes, and all
/// but the last row are discarded by whoever samples.
pub fn forward_step_last(
    weights: &NativeWeights,
    input_ids: &InlineArray,
    cache: &mut NativeCache,
) -> InlineArray {
    let hidden = forward_hidden(weights, input_ids, cache);
    logits_from_hidden(weights, &crate::decode::last_token_hidden(&hidden)).squeeze(1)
}

/// The final norm's output for every position, `[B, T, hidden]`: everything
/// [`forward_step`] does before the LM head.
///
/// Structured after `qwen3_native::forward_step`: norms and residuals run
/// as plain per-op calls in Rust, attention goes through
/// `compiled_gemma4_attn_block` (the narrow qkv → sdpa → o_proj fusion),
/// and the MLP is per-op matmuls + a tanh-approx GELU helper. Wrapping
/// the norms and MLP inside bigger compiled lambdas made the graph
/// harder to fuse efficiently; matching Qwen3's layout recovers the
/// single-compile-per-layer pattern.
fn forward_hidden(
    weights: &NativeWeights,
    input_ids: &InlineArray,
    cache: &mut NativeCache,
) -> InlineArray {
    let seq_len = input_ids.dim(1);
    let profile = crate::env_flag::env_flag!("PMETAL_GEMMA4_TIMING");
    let t_start = if profile {
        Some(std::time::Instant::now())
    } else {
        None
    };

    // 1. Embedding + scale. Scale is pre-cast to model dtype at load
    // time so broadcasting doesn't promote the bf16 residual stream.
    let mut hidden = weights
        .embed_w
        .lookup(input_ids)
        .multiply(&weights.embed_scale_scalar);
    let per_layer_inputs = compute_per_layer_inputs(weights, input_ids, &hidden);

    let rope_offset = cache.rope_offset;
    let eps = weights.config.rms_norm_eps;

    let t_embed = if profile {
        // Force sync so timing reflects actual GPU work, not queue-only
        // dispatch. `eval()` returns an `EvalToken` that blocks until
        // the array is materialized.
        let _ = hidden.eval();
        Some(std::time::Instant::now())
    } else {
        None
    };
    let mut attn_elapsed_ns: u128 = 0;
    let mut mlp_elapsed_ns: u128 = 0;

    // 2. Decoder layer stack. Structured exactly like
    // `qwen3_native::forward_step`: per-op norms on the Rust side,
    // narrow compiled attention kernel, per-op MLP.
    for (i, layer) in weights.layers.iter().enumerate() {
        let t_attn_start = if profile {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let attn_out = match &layer.kv {
            AttentionKv::Shared { source_layer } => {
                let source_layer = *source_layer;
                let attn = {
                    let source_cache = &cache.layers[source_layer];
                    shared_kv_attention_forward(&hidden, layer, source_cache, rope_offset, eps)
                };
                let source_offset = cache.layers[source_layer].offset;
                cache.layers[i].offset = source_offset;
                attn
            }
            AttentionKv::Projected { k_w, k_norm_w, v_w } if seq_len > 1 => {
                projected_attention_multi(
                    &hidden,
                    layer,
                    k_w,
                    k_norm_w,
                    v_w.as_ref(),
                    &mut cache.layers[i],
                    rope_offset,
                    weights.model_dtype,
                    eps,
                )
            }
            AttentionKv::Projected { k_w, k_norm_w, v_w } => {
                let layer_cache = &mut cache.layers[i];
                // Room for the token: a full layer grows its buffer, a sliding
                // one moves its base forward once its slack is used up.
                layer_cache.make_room(
                    hidden.dim(0),
                    layer.n_kv_heads,
                    layer.head_dim,
                    seq_len,
                    weights.model_dtype,
                );

                let cache_k = layer_cache.keys.take().unwrap();
                let cache_v = layer_cache.values.take().unwrap();

                // Wide compiled attention block: includes input_layernorm at
                // the top and post_attention_layernorm at the bottom. Measured
                // to be ~15-20% faster on 31B than the narrower "norms-outside"
                // variant — probably because the extra per-op RMS-norm kernels
                // become their own dispatch-cost floor for large hidden sizes.
                let (attn_out, new_k, new_v) = InlineArray::compiled_gemma4_attn_block(
                    &hidden,
                    &layer.input_norm_w,
                    &layer.q_w,
                    k_w,
                    v_w.as_ref(),
                    &layer.o_w,
                    &layer.q_norm_w,
                    k_norm_w,
                    &layer.post_attn_norm_w,
                    layer.rope_freqs.as_ref(),
                    &cache_k,
                    &cache_v,
                    rope_offset,
                    layer_cache.base as i32,
                    layer.n_heads,
                    layer.n_kv_heads,
                    layer.head_dim,
                    eps,
                    eps,
                    eps,
                    layer.sliding_window.unwrap_or(0),
                    layer.rope_base,
                    layer.rope_dims,
                );
                layer_cache.keys = Some(new_k);
                layer_cache.values = Some(new_v);
                layer_cache.offset = (rope_offset + seq_len) as usize;
                attn_out
            }
        };

        if profile {
            let _ = attn_out.eval();
            attn_elapsed_ns += t_attn_start.unwrap().elapsed().as_nanos();
        }

        // attn_out is already post-attention-layernormed.
        let h = hidden.add(&attn_out);

        let t_mlp_start = if profile {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── MLP block ──────────────────────────────────────────────
        // Plain per-op path, following Qwen3's `dense_mlp_forward`:
        // pre_ffn_norm → gate / up matmuls → GELU·multiply → down
        // matmul → post_ffn_norm. mlx's per-op matmul is already highly
        // tuned for `[in, out]` pre-transposed weights.
        let mlp_out = if let Some(moe) = &layer.moe {
            // The compiled block ends in the post-FFN norm, and a MoE layer
            // needs the dense output before it.
            let mlp_in = h.rms_norm(Some(&layer.pre_ffn_norm_w), eps);
            let gate = layer.gate_w.matmul_from(&mlp_in);
            let up = layer.up_w.matmul_from(&mlp_in);
            let dense = layer
                .down_w
                .matmul_from(&InlineArray::fused_geglu_tanh(&gate, &up));
            moe_feed_forward(moe, &h, &dense, &layer.post_ffn_norm_w, eps)
        } else if seq_len == 1 {
            InlineArray::compiled_gemma4_mlp_block(
                &h,
                &layer.pre_ffn_norm_w,
                &layer.gate_w,
                &layer.up_w,
                &layer.down_w,
                &layer.post_ffn_norm_w,
                eps,
                eps,
            )
        } else {
            let mlp_in = h.rms_norm(Some(&layer.pre_ffn_norm_w), eps);
            let gate = layer.gate_w.matmul_from(&mlp_in);
            let up = layer.up_w.matmul_from(&mlp_in);
            let activated = InlineArray::fused_geglu_tanh(&gate, &up);
            let down = layer.down_w.matmul_from(&activated);
            down.rms_norm(Some(&layer.post_ffn_norm_w), eps)
        };

        hidden = h.add(&mlp_out);
        if let Some(ref per_layer_inputs) = per_layer_inputs
            && layer.per_layer_gate.is_some()
        {
            let layer_input = layer_per_input(per_layer_inputs, i);
            hidden = apply_per_layer_input_block(&hidden, layer, &layer_input, eps);
        }
        hidden = hidden.multiply(&layer.layer_scalar);

        if profile {
            let _ = hidden.eval();
            mlp_elapsed_ns += t_mlp_start.unwrap().elapsed().as_nanos();
        }
    }

    cache.rope_offset += seq_len;

    if profile {
        let _ = hidden.eval();
        let t_layers_end = std::time::Instant::now();
        let layers_elapsed = t_layers_end.duration_since(t_embed.unwrap()).as_secs_f64() * 1000.0;
        let embed_elapsed = t_embed
            .unwrap()
            .duration_since(t_start.unwrap())
            .as_secs_f64()
            * 1000.0;
        let attn_ms = (attn_elapsed_ns as f64) / 1_000_000.0;
        let mlp_ms = (mlp_elapsed_ns as f64) / 1_000_000.0;
        eprintln!(
            "[gemma4_native profile] total_layers={layers_elapsed:.2}ms embed={embed_elapsed:.2}ms attn={attn_ms:.2}ms mlp={mlp_ms:.2}ms (seq_len={seq_len} rope_offset={})",
            cache.rope_offset
        );
    }

    hidden.rms_norm(Some(&weights.final_norm_w), eps)
}

/// The LM head and the final logit softcap over normalized hidden states.
fn logits_from_hidden(weights: &NativeWeights, normed: &InlineArray) -> InlineArray {
    let raw_logits = match weights.lm_head_w.as_ref() {
        Some(w) => w.matmul_from(normed),
        // Tied embedding: `as_linear` is `mlx.nn.Embedding.as_linear`, which
        // takes the `[vocab, hidden]` table's transpose for a dense model and
        // runs a quantized matmul against the packed rows otherwise.
        None => weights.embed_w.as_linear(normed),
    };
    match weights.config.final_logit_softcapping {
        Some(cap) => crate::nn::softcap(&raw_logits, cap),
        None => raw_logits,
    }
}

// ----------------------------------------------------------------------------
// Public prefill / generate wrappers
// ----------------------------------------------------------------------------

/// Prefill `input_ids` in [`PREFILL_CHUNK`](crate::decode::PREFILL_CHUNK)
/// passes and sample the first token from the last position's logits.
pub fn prefill_first_token(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    input_ids: &[u32],
    params: crate::decode::SamplingParams,
) -> (u32, crate::decode::DecodeSampler) {
    crate::decode::prefill_first_token_chunked(
        weights,
        cache,
        input_ids,
        params,
        crate::decode::PREFILL_CHUNK,
        forward_step_last,
    )
}

pub fn generate(
    weights: &NativeWeights,
    _config: &Gemma4Config,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    mut sampler: crate::decode::DecodeSampler,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    let model_dtype = weights.model_dtype;
    let reserve = max_tokens.min(i32::MAX as usize) as i32;
    let current_y = crate::decode::prime_generation(
        "NATIVE-GEMMA4",
        model_dtype,
        weights,
        cache,
        first_token,
        &mut sampler,
        true,
        true,
        move |cache| {
            // Trim the prefill-allocated KV slack then grow to exactly
            // `prompt_len + max_tokens` so decode-time SDPA walks the
            // real cache length instead of the chunk-rounded allocation. Mirrors qwen3_native's decode priming.
            cache.eval_and_detach_states();
            cache.reserve_decode_inputs(reserve, model_dtype);
        },
        forward_step,
    );
    crate::decode::generate_from_primed_sample(
        "NATIVE-GEMMA4",
        weights,
        cache,
        current_y,
        max_tokens,
        sampler,
        true,
        on_token,
        forward_step,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma 4's global layers rotate with transformers' proportional RoPE
    /// (the shared fixture's `proportional` case: head 256, partial 0.25,
    /// theta 1e6), its sliding layers with plain RoPE over the whole head;
    /// any other rotation is refused by name rather than run as one of those.
    #[test]
    fn gemma4_layer_rope_is_proportional_or_refused() {
        let global = gemma4_layer_rope(256, Some("proportional"), 1e6, 0.25, None, true).unwrap();
        assert_eq!(global.rotated_dims, 64);
        let periods = global.periods.expect("a partial layer has a table");
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/rope_scaling_reference.json"
        ))
        .unwrap();
        let case = reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "proportional")
            .unwrap();
        for (p, w) in periods.iter().zip(case["inv_freq"].as_array().unwrap()) {
            let w = w.as_f64().unwrap();
            if w == 0.0 {
                assert!(p.is_infinite());
            } else {
                assert!(((1.0 / *p as f64) - w).abs() <= 2e-6 * w, "{p} vs 1/{w}");
            }
        }
        let sliding = gemma4_layer_rope(256, Some("default"), 1e4, 1.0, None, false).unwrap();
        assert_eq!((sliding.rotated_dims, sliding.periods), (256, None));

        for (rope_type, partial, factor, needle) in [
            (Some("yarn"), 1.0, None, "yarn"),
            (Some("default"), 0.25, None, "default"),
            (Some("proportional"), 0.25, Some(8.0), "proportional"),
        ] {
            let err = gemma4_layer_rope(256, rope_type, 1e6, partial, factor, true).unwrap_err();
            assert!(err.to_string().contains(needle), "{err}");
        }
    }

    /// Gate is the first half of `gate_up_proj`, checked where GELU is far
    /// from linear so that swapping the halves changes the answer. The
    /// synthetic oracle cannot see this: its weights keep GELU linear.
    #[test]
    fn fused_experts_split_gate_first() {
        let (experts, hidden, inter) = (2usize, 4usize, 3usize);
        let gate_up: Vec<f32> = (0..experts * 2 * inter * hidden)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) / 2.5)
            .collect();
        let down: Vec<f32> = (0..experts * hidden * inter)
            .map(|i| ((i * 5 % 9) as f32 - 4.0) / 3.0)
            .collect();
        let x: Vec<f32> = vec![0.9, -1.3, 1.7, 0.4];

        let (gate, up, down_w) = experts_from_fused(
            &InlineArray::from_f32_slice(
                &gate_up,
                &[experts as i32, 2 * inter as i32, hidden as i32],
            ),
            InlineArray::from_f32_slice(&down, &[experts as i32, hidden as i32, inter as i32]),
            inter as i32,
        );
        // Route the one token to expert 1 alone, at weight 1.
        let inds = InlineArray::from_u32_slice(&[1], &[1, 1]);
        let scores = InlineArray::from_f32_slice(&[1.0], &[1, 1]);
        let mut got = crate::native_moe::switch_glu(
            &InlineArray::from_f32_slice(&x, &[1, hidden as i32]),
            &gate,
            &up,
            &down_w,
            &inds,
            &scores,
            InlineArray::fused_geglu_tanh,
        );
        crate::check_unobserved_error().expect("switch_glu");
        let got = got.to_f32_vec(hidden).expect("read output");

        let gelu = |v: f64| {
            0.5 * v
                * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh())
        };
        let row = |m: &[f32], r: usize, cols: usize| -> Vec<f64> {
            m[r * cols..(r + 1) * cols]
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        };
        let dot =
            |a: &[f64], b: &[f32]| a.iter().zip(b).map(|(p, &q)| p * f64::from(q)).sum::<f64>();
        let e = 1usize;
        let activated: Vec<f64> = (0..inter)
            .map(|j| {
                let g = dot(&row(&gate_up, e * 2 * inter + j, hidden), &x);
                let u = dot(&row(&gate_up, e * 2 * inter + inter + j, hidden), &x);
                gelu(g) * u
            })
            .collect();
        for (h, got) in got.iter().enumerate() {
            let want: f64 = row(&down, e * hidden + h, inter)
                .iter()
                .zip(&activated)
                .map(|(d, a)| d * a)
                .sum();
            assert!(
                (f64::from(*got) - want).abs() < 1e-3,
                "output {h}: {got} against {want}; gate and up split the wrong way?"
            );
        }
    }

    fn toy_config() -> Gemma4Config {
        Gemma4Config {
            vocab_size: 8,
            hidden_size: 4,
            num_hidden_layers: 2,
            intermediate_size: 8,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            global_head_dim: None,
            num_global_key_value_heads: None,
            rms_norm_eps: 1e-6,
            attention_k_eq_v: false,
            tie_word_embeddings: true,
            sliding_window: 4,
            final_logit_softcapping: None,
            layer_types: vec![
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            use_bidirectional_attention: None,
            attention_bias: false,
            rope_parameters: None,
            hidden_size_per_layer_input: Some(2),
            vocab_size_per_layer_input: Some(8),
            hidden_activation: Some("gelu_pytorch_tanh".to_string()),
            num_kv_shared_layers: Some(0),
            use_double_wide_mlp: Some(false),
            enable_moe_block: Some(false),
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
        }
    }

    #[test]
    fn parse_config_text_accepts_e4b_like_small_model_fields() {
        let text = r#"{
            "text_config": {
                "vocab_size": 262144,
                "hidden_size": 2560,
                "num_hidden_layers": 42,
                "intermediate_size": 10240,
                "num_attention_heads": 10,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "global_head_dim": 512,
                "rms_norm_eps": 1e-6,
                "sliding_window": 512,
                "layer_types": [
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "full_attention"
                ],
                "hidden_size_per_layer_input": 256,
                "vocab_size_per_layer_input": 262144,
                "hidden_activation": "gelu_pytorch_tanh",
                "num_kv_shared_layers": 18,
                "use_double_wide_mlp": false,
                "enable_moe_block": false
            }
        }"#;
        let cfg = parse_config_text(text).expect("small-model config parses");
        assert!(cfg.uses_per_layer_inputs());
        assert_eq!(cfg.per_layer_input_dim(), 256);
        assert_eq!(cfg.per_layer_input_vocab_size(), 262144);
        assert_eq!(cfg.num_kv_shared_layers(), 18);
        assert_eq!(cfg.first_kv_shared_layer_idx(), 24);
    }

    fn minimal_config(extra: &str) -> String {
        format!(
            r#"{{"vocab_size": 16, "hidden_size": 8, "num_hidden_layers": 12,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                "hidden_size_per_layer_input": 0{extra}}}"#
        )
    }

    /// What `Gemma4TextConfig` does with keys a `config.json` leaves out: no
    /// final softcap, a window of 512, and layers in its default 5:1 pattern.
    #[test]
    fn absent_keys_take_transformers_defaults() {
        let cfg = parse_config_text(&minimal_config("")).expect("parses");
        assert_eq!(cfg.final_logit_softcapping, None);
        assert_eq!(cfg.sliding_window, 512);
        assert_eq!(cfg.layer_types.len(), 12);
        let full: Vec<usize> = (0..12).filter(|&i| cfg.is_full_attention(i)).collect();
        assert_eq!(full, [5, 11]);

        let capped = parse_config_text(&minimal_config(r#", "final_logit_softcapping": 30.0"#))
            .expect("parses");
        assert_eq!(capped.final_logit_softcapping, Some(30.0));
    }

    /// `Gemma4TextConfig.__post_init__` forces the last layer to full
    /// attention, whatever `layer_types` says.
    #[test]
    fn the_last_layer_is_full_attention() {
        let cfg = parse_config_text(
            r#"{"vocab_size": 16, "hidden_size": 8, "num_hidden_layers": 3,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                "hidden_size_per_layer_input": 0,
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention"]}"#,
        )
        .expect("parses");
        assert!(cfg.is_full_attention(1));
        assert!(cfg.is_full_attention(2), "the last layer is forced full");
    }

    /// A config asking for what the engine would silently drop is refused by
    /// name rather than run as something else.
    #[test]
    fn features_the_engine_does_not_run_are_refused() {
        for (extra, needle) in [
            (r#", "use_bidirectional_attention": "all""#, "bidirectional"),
            (r#", "attention_bias": true"#, "attention_bias"),
            (r#", "hidden_activation": "silu""#, "hidden_activation"),
        ] {
            let err = parse_config_text(&minimal_config(extra)).expect_err(extra);
            assert!(err.to_string().contains(needle), "{extra}: {err}");
        }
        // Vision-only bidirectionality leaves a text prompt causal.
        parse_config_text(&minimal_config(
            r#", "use_bidirectional_attention": "vision""#,
        ))
        .expect("vision-only bidirectional attention is a text-causal model");
        parse_config_text(&minimal_config(r#", "attention_bias": false"#)).expect("parses");
    }

    #[test]
    fn kv_shared_source_mapping_matches_upstream_rule() {
        let mut cfg = toy_config();
        cfg.num_hidden_layers = 6;
        cfg.layer_types = vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ];
        cfg.num_kv_shared_layers = Some(2);
        assert_eq!(cfg.first_kv_shared_layer_idx(), 4);
        assert_eq!(cfg.kv_shared_source_layer(3), None);
        assert_eq!(cfg.kv_shared_source_layer(4), Some(2));
        assert_eq!(cfg.kv_shared_source_layer(5), Some(3));
    }

    /// `mlx-community/gemma-4-e4b-it-4bit`: 42 layers, the last 18 shared, so
    /// every shared layer reads the last concrete layer of its own attention
    /// type (22 sliding, 23 full). This is `layer_idx_to_cache_idx` in
    /// the reference Gemma 3n model, which E4B inherits from.
    ///
    /// ⚠️ Those 18 layers ship no `k_proj`, `v_proj` or `k_norm` at all, which
    /// is why demanding them kept the whole E2B/E4B family from loading (#31).
    #[test]
    fn e4b_shared_layers_read_the_last_concrete_layer_of_their_type() {
        let mut cfg = toy_config();
        cfg.num_hidden_layers = 42;
        cfg.layer_types = (0..42)
            .map(|i| {
                if i % 2 == 0 {
                    "sliding_attention".to_string()
                } else {
                    "full_attention".to_string()
                }
            })
            .collect();
        cfg.num_kv_shared_layers = Some(18);

        assert_eq!(cfg.first_kv_shared_layer_idx(), 24);
        // The last concrete layer of each type, layers 0..24.
        assert_eq!(cfg.kv_shared_source_layer(24), Some(22), "sliding");
        assert_eq!(cfg.kv_shared_source_layer(25), Some(23), "full");
        assert_eq!(cfg.kv_shared_source_layer(41), Some(23), "last layer, full");
        // Everything below the boundary fills its own cache.
        assert_eq!(cfg.kv_shared_source_layer(23), None);
    }

    /// A layer either projects its own K/V or reuses a cache, and the type
    /// keeps the third combination from existing.
    #[test]
    fn attention_kv_reports_its_source_layer_only_when_shared() {
        assert_eq!(
            AttentionKv::Shared { source_layer: 22 }.source_layer(),
            Some(22)
        );
        assert_eq!(
            AttentionKv::Projected {
                k_w: LayerWeight::Dense(InlineArray::zeros(&[2, 2], Dtype::Float32.as_i32())),
                k_norm_w: InlineArray::ones(&[2], Dtype::Float32.as_i32()),
                v_w: None,
            }
            .source_layer(),
            None
        );
    }

    #[test]
    fn compute_per_layer_inputs_matches_scaled_embedding_path() {
        let config = toy_config();
        let weights = NativeWeights {
            embed_w: EmbeddingWeight::Dense(InlineArray::zeros(&[8, 4], Dtype::Float32.as_i32())),
            per_layer_inputs: Some(PerLayerInputWeights {
                embed_w: EmbeddingWeight::Dense(InlineArray::from_f32_slice(
                    &[
                        0.0, 0.0, 0.0, 0.0, //
                        1.0, 2.0, 3.0, 4.0, //
                        5.0, 6.0, 7.0, 8.0, //
                        0.0, 0.0, 0.0, 0.0, //
                        0.0, 0.0, 0.0, 0.0, //
                        0.0, 0.0, 0.0, 0.0, //
                        0.0, 0.0, 0.0, 0.0, //
                        0.0, 0.0, 0.0, 0.0,
                    ],
                    &[8, 4],
                )),
                projection_w: LayerWeight::Dense(InlineArray::zeros(
                    &[4, 4],
                    Dtype::Float32.as_i32(),
                )),
                projection_norm_w: InlineArray::ones(&[2], Dtype::Float32.as_i32()),
                embed_scale_scalar: InlineArray::scalar_with_dtype(
                    2.0f32.sqrt(),
                    Dtype::Float32.as_i32(),
                ),
                projection_scale_scalar: InlineArray::scalar_with_dtype(
                    0.5,
                    Dtype::Float32.as_i32(),
                ),
                input_scale_scalar: InlineArray::scalar_with_dtype(
                    2.0f32.powf(-0.5),
                    Dtype::Float32.as_i32(),
                ),
                vocab_size: 8,
                hidden_size: 4,
            }),
            final_norm_w: InlineArray::ones(&[4], Dtype::Float32.as_i32()),
            lm_head_w: None,
            layers: vec![],
            model_dtype: Dtype::Float32.as_i32(),
            config,
            embed_scale_scalar: InlineArray::scalar_with_dtype(2.0, Dtype::Float32.as_i32()),
        };
        let input_ids = InlineArray::from_i32_slice_shaped(&[1, 2], &[1, 2]);
        let inputs_embeds = InlineArray::zeros(&[1, 2, 4], Dtype::Float32.as_i32());
        let mut ple = compute_per_layer_inputs(&weights, &input_ids, &inputs_embeds)
            .expect("per-layer inputs computed");
        let got = ple.to_f32_vec(8).expect("to_f32_vec");
        let want = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for (actual, expected) in got.iter().zip(want.iter()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "PLE value mismatch: got {actual} expected {expected}"
            );
        }
    }

    fn mask_rows(mask: InlineArray, n: usize) -> Vec<f32> {
        let mut mask = mask.as_dtype(Dtype::Float32.as_i32()).reshape(&[-1]);
        mask.to_f32_vec(n).expect("to_f32_vec")
    }

    /// Queries at 4, 5 and 6 against keys 0 to 6 with a window of 3: each sees
    /// itself and the two before it.
    #[test]
    fn sliding_attention_mask_respects_offset_window() {
        let got = mask_rows(sliding_window_mask(4, 3, 0, 7, 3), 21);
        assert_eq!(
            got,
            vec![
                0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            ]
        );
        // The same queries against the keys from 2 on: the first two columns
        // gone, the rest unchanged.
        let from_two = mask_rows(sliding_window_mask(4, 3, 2, 5, 3), 15);
        assert_eq!(
            from_two,
            vec![
                1.0, 1.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, 1.0, 1.0,
            ]
        );
    }

    /// Attention over a pass of `s` tokens after `prev` cached ones, against
    /// softmax over the allowed keys worked out on the host: for a full layer,
    /// for a sliding layer whose window does and does not reach back to the
    /// start of the cache, and with a window longer than the whole sequence.
    #[test]
    fn prefill_attention_matches_a_host_softmax() {
        let (heads, kv_heads, head_dim) = (4usize, 2usize, 4usize);
        let fill = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 29 + salt * 13) % 31) as f32 - 15.0) / 9.0)
                .collect()
        };
        // (tokens before the pass, tokens in it, window, the position of the
        // first key handed over: a sliding layer's cache starts where its
        // window did)
        for (prev, s, window, key_start) in [
            (0usize, 6usize, None, 0usize),
            (7, 5, None, 0),
            (7, 5, Some(4usize), 0),
            (20, 3, Some(6), 0),
            (2, 4, Some(5), 0),
            (9, 1, Some(4), 0),
            (3, 4, Some(100), 0),
            (20, 3, Some(6), 14),
            (40, 5, Some(8), 33),
            (9, 1, Some(4), 6),
        ] {
            let next = prev + s;
            let (q, k, v) = (
                fill(heads * s * head_dim, 1),
                fill(kv_heads * next * head_dim, 2),
                fill(kv_heads * next * head_dim, 3),
            );
            // The keys from `key_start` on, per kv head.
            let from = |all: &[f32]| -> Vec<f32> {
                (0..kv_heads)
                    .flat_map(|kv| {
                        all[(kv * next + key_start) * head_dim..(kv + 1) * next * head_dim]
                            .iter()
                            .copied()
                    })
                    .collect()
            };
            let len = (next - key_start) as i32;
            let got = prefill_attention(
                &InlineArray::from_f32_slice(&q, &[1, heads as i32, s as i32, head_dim as i32]),
                &InlineArray::from_f32_slice(
                    &from(&k),
                    &[1, kv_heads as i32, len, head_dim as i32],
                ),
                &InlineArray::from_f32_slice(
                    &from(&v),
                    &[1, kv_heads as i32, len, head_dim as i32],
                ),
                prev as i32,
                key_start as i32,
                window.map(|w| w as i32),
            );
            crate::check_unobserved_error().expect("prefill attention");
            let mut got = got.reshape(&[-1]);
            let got = got.to_f32_vec(heads * s * head_dim).expect("read back");

            let mut want = vec![0f32; heads * s * head_dim];
            for h in 0..heads {
                let kv = h / (heads / kv_heads);
                for r in 0..s {
                    let pos = prev + r;
                    let allowed: Vec<usize> = (0..next)
                        .filter(|&j| j <= pos && window.is_none_or(|w| j + w > pos))
                        .collect();
                    let scores: Vec<f64> = allowed
                        .iter()
                        .map(|&j| {
                            (0..head_dim)
                                .map(|d| {
                                    f64::from(q[(h * s + r) * head_dim + d])
                                        * f64::from(k[(kv * next + j) * head_dim + d])
                                })
                                .sum()
                        })
                        .collect();
                    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|x| (x - top).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    for d in 0..head_dim {
                        want[(h * s + r) * head_dim + d] = allowed
                            .iter()
                            .zip(&weights)
                            .map(|(&j, w)| w / total * f64::from(v[(kv * next + j) * head_dim + d]))
                            .sum::<f64>()
                            as f32;
                    }
                }
            }
            let worst = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < 1e-4,
                "prev {prev}, {s} tokens, window {window:?}: off by {worst}"
            );
        }
    }

    /// `mlx-community/gemma-4-12B-it-bf16` and `-31b-it-bf16` publish the text
    /// tower under `language_model.model.*`, which the loader used to leave
    /// untouched and then reject with `missing weight model.embed_tokens.weight`
    /// (issue #26).
    #[test]
    fn unified_layout_normalises_onto_the_text_tower() {
        assert_eq!(
            normalize_checkpoint_key("language_model.model.embed_tokens.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            normalize_checkpoint_key("language_model.model.layers.7.self_attn.q_proj.weight")
                .as_deref(),
            Some("model.layers.7.self_attn.q_proj.weight")
        );
        assert_eq!(
            normalize_checkpoint_key("language_model.lm_head.weight").as_deref(),
            Some("lm_head.weight")
        );
    }

    #[test]
    fn transformers_and_text_only_layouts_normalise_onto_the_text_tower() {
        assert_eq!(
            normalize_checkpoint_key("model.language_model.embed_tokens.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            normalize_checkpoint_key("model.layers.0.layer_scalar").as_deref(),
            Some("model.layers.0.layer_scalar")
        );
        assert_eq!(
            normalize_checkpoint_key("lm_head.weight").as_deref(),
            Some("lm_head.weight")
        );
    }

    /// The unified export parks its towers *beside* the text backbone rather
    /// than under a shared `model.` root, so the skip list has to name those
    /// spellings too.
    #[test]
    fn tower_weights_are_skipped_in_every_layout() {
        for key in [
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
            "model.embed_vision.embedding_projection.weight",
            "vision_embedder.patch_dense.weight",
            "embed_vision.embedding_projection.weight",
            "embed_audio.embedding_projection.weight",
            "language_model.model.audio_tower.something.weight",
            "multi_modal_projector.weight",
        ] {
            assert_eq!(normalize_checkpoint_key(key), None, "should skip {key}");
        }
    }

    /// The compiled attention block is traced per cache capacity, and a server
    /// sees a new capacity per request. Past the number of traces a call site
    /// keeps, the oldest is dropped and made again on its next use, which has
    /// to give the answer it gave the first time.
    #[test]
    fn compiled_attention_block_is_unchanged_by_trace_eviction() {
        use crate::compat::random;
        let bf16 = Dtype::Bfloat16.as_i32();
        let (hidden_size, n_heads, n_kv, head_dim) = (64, 2, 1, 32);
        let rand = |shape: &[i32]| random::normal(shape, Dtype::Float32).as_dtype(bf16);
        let dense = |shape: &[i32]| LayerWeight::Dense(rand(shape));
        let (q_w, k_w, v_w) = (
            dense(&[hidden_size, n_heads * head_dim]),
            dense(&[hidden_size, n_kv * head_dim]),
            dense(&[hidden_size, n_kv * head_dim]),
        );
        let o_w = dense(&[n_heads * head_dim, hidden_size]);
        let ones = |n: i32| InlineArray::ones(&[n], bf16);
        let (in_norm, q_norm, k_norm, post_norm) = (
            ones(hidden_size),
            ones(head_dim),
            ones(head_dim),
            ones(hidden_size),
        );
        let x = rand(&[1, 1, hidden_size]);
        let run = |capacity: i32| -> Vec<f32> {
            let keys = rand(&[1, n_kv, capacity, head_dim]);
            let values = rand(&[1, n_kv, capacity, head_dim]);
            let (out, _, _) = InlineArray::compiled_gemma4_attn_block(
                &x,
                &in_norm,
                &q_w,
                &k_w,
                Some(&v_w),
                &o_w,
                &q_norm,
                &k_norm,
                &post_norm,
                None,
                &keys,
                &values,
                capacity / 2,
                0,
                n_heads,
                n_kv,
                head_dim,
                1e-6,
                1e-6,
                1e-6,
                0,
                10000.0,
                head_dim,
            );
            out.as_dtype(Dtype::Float32.as_i32())
                .to_f32_vec(hidden_size as usize)
                .unwrap()
        };
        // The same cache contents on every call at one capacity.
        random::seed(7);
        let first = run(16);
        for capacity in 17..17 + 2 * 64 {
            run(capacity);
        }
        random::seed(7);
        let again = run(16);
        crate::check_unobserved_error().expect("a runtime op threw");
        assert!(first.iter().all(|v| v.is_finite()));
        assert_eq!(first, again, "a re-traced block changed its answer");
    }
}
