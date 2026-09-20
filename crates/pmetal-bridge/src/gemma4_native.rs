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
//!   array (mlx-lm's `ProportionalRoPE`) — `freqs[i] = base^(2i/head_dim)` for
//!   `i in 0..rotated_dims/2` with the remaining slots filled with `inf` so
//!   `fast::rope(..., freqs=)` leaves them untouched.
//! * Per-layer `layer_scalar` multiplier applied at the end of each decoder
//!   layer forward.
//! * Per-layer embeddings / gating (`hidden_size_per_layer_input`) used by
//!   the E2B/E4B checkpoints.
//! * KV-sharing (`num_kv_shared_layers`) used by the E2B/E4B checkpoints.
//! * Final logit softcap: `softcap * tanh(logits / softcap)`.
//! * Embedding scale by `sqrt(hidden_size)` (shared with Gemma 2/3).
//! * Tanh-approximation GELU in the MLP (matching mlx-lm's `nn.gelu_approx`).
//!
//! Weights are pre-transposed to `[in, out]` at load time (`w.t()`) so the
//! per-decode matmuls are contiguous.

use serde::Deserialize;

use crate::InlineArray;
use crate::compat::{Dtype, nn};
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
fn default_sliding_window() -> i32 {
    1024
}
fn default_final_logit_softcapping() -> Option<f32> {
    Some(30.0)
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Gemma4RopeLayerConfig {
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_type: Option<String>,
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
    #[serde(default = "default_final_logit_softcapping")]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub layer_types: Vec<String>,
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

    pub fn layer_rope(&self, layer_idx: usize) -> (f32, f32) {
        let is_full = self.is_full_attention(layer_idx);
        let defaults = if is_full {
            (default_rope_theta_global(), 0.25)
        } else {
            (default_rope_theta_sliding(), 1.0)
        };
        if let Some(ref rp) = self.rope_parameters {
            let cfg = if is_full {
                &rp.full_attention
            } else {
                &rp.sliding_attention
            };
            let base = cfg.rope_theta.unwrap_or(defaults.0);
            let frac = cfg.partial_rotary_factor;
            return (base, frac);
        }
        defaults
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

    fn validate_supported(&self) -> Result<(), String> {
        if let Some(ref act) = self.hidden_activation
            && self.uses_per_layer_inputs()
            && !matches!(act.as_str(), "gelu" | "gelu_pytorch_tanh" | "gelu_tanh")
        {
            return Err(format!(
                "Gemma 4 native: unsupported per-layer-input activation {act:?}"
            ));
        }
        if self.enable_moe_block.unwrap_or(false) {
            return Err("Gemma 4 native: MoE block is not ported yet".to_string());
        }
        if self.use_double_wide_mlp.unwrap_or(false) {
            return Err("Gemma 4 native: double-wide MLP is not ported yet".to_string());
        }
        Ok(())
    }
}

pub fn load_config(model_dir: &std::path::Path) -> Result<Gemma4Config, String> {
    let text = crate::native_loader::read_config_json(model_dir)?;
    parse_config_text(&text)
}

fn parse_config_text(text: &str) -> Result<Gemma4Config, String> {
    let json: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("failed to parse config.json: {e}"))?;

    // Multimodal Gemma 4 nests text params under `text_config`.
    let config_str = if json.get("text_config").is_some() {
        serde_json::to_string(&json["text_config"]).map_err(|e| e.to_string())?
    } else {
        text.to_owned()
    };

    let cfg: Gemma4Config = serde_json::from_str(&config_str)
        .map_err(|e| format!("failed to parse gemma4 config: {e}"))?;
    cfg.validate_supported()?;
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
    /// `final_logit_softcapping` pre-cast to `model_dtype` (only set when
    /// the config opts in). Same reasoning as `embed_scale_scalar`.
    pub softcap_scalar: Option<InlineArray>,
}

/// Per-layer KV cache. Keys/values are allocated to `[B, n_kv, L, D]`
/// where `L` grows in `CACHE_STEP_SIZE`-token chunks.
pub struct LayerCache {
    pub keys: Option<InlineArray>,
    pub values: Option<InlineArray>,
    pub offset: usize,
}

/// Full-model cache. `rope_offset` mirrors the layer offsets — for Gemma 4
/// all layers share the same position counter.
pub struct NativeCache {
    pub layers: Vec<LayerCache>,
    pub rope_offset: i32,
}

impl NativeCache {
    /// Eval every layer's KV buffer and **trim** it down to the actual
    /// offset before decode starts. This drops the unused trailing
    /// slots from the prefill allocation so decode-time SDPA doesn't
    /// have to process them via validity-mask masking every step.
    /// Mirrors `qwen3_native::NativeCache::eval_and_detach_states`.
    pub fn eval_and_detach_states(&mut self) {
        for layer in self.layers.iter_mut() {
            if let Some(k) = layer.keys.take() {
                let trimmed = if layer.offset > 0 && (layer.offset as i32) < k.dim(2) {
                    k.slice(
                        &[0, 0, 0, 0],
                        &[k.dim(0), k.dim(1), layer.offset as i32, k.dim(3)],
                    )
                } else {
                    k
                };
                trimmed.async_eval_ref();
                layer.keys = Some(trimmed);
            }
            if let Some(v) = layer.values.take() {
                let trimmed = if layer.offset > 0 && (layer.offset as i32) < v.dim(2) {
                    v.slice(
                        &[0, 0, 0, 0],
                        &[v.dim(0), v.dim(1), layer.offset as i32, v.dim(3)],
                    )
                } else {
                    v
                };
                trimmed.async_eval_ref();
                layer.values = Some(trimmed);
            }
        }
    }

    /// Grow every layer's KV buffer to exactly `offset + additional_tokens`
    /// slots, preparing the cache for a decode run of known length. The
    /// resulting buffer is sized so the decode loop never hits the
    /// `ensure_cache_capacity` grow path — and, critically, so per-step
    /// SDPA only processes `offset + step` slots instead of the
    /// CACHE_STEP_SIZE-rounded-up 256-token buffer.
    pub fn reserve_decode_inputs(&mut self, additional_tokens: i32, dtype: i32) {
        if additional_tokens <= 0 {
            return;
        }
        for layer in self.layers.iter_mut() {
            let Some(keys) = layer.keys.take() else {
                continue;
            };
            let Some(values) = layer.values.take() else {
                layer.keys = Some(keys);
                continue;
            };
            let current = keys.dim(2);
            let target = layer.offset as i32 + additional_tokens;
            if target <= current {
                layer.keys = Some(keys);
                layer.values = Some(values);
                continue;
            }
            let extend = target - current;
            let b = keys.dim(0);
            let nkv = keys.dim(1);
            let hd_k = keys.dim(3);
            let hd_v = values.dim(3);
            let ext_k = InlineArray::zeros(&[b, nkv, extend, hd_k], dtype);
            let ext_v = InlineArray::zeros(&[b, nkv, extend, hd_v], dtype);
            layer.keys = Some(keys.kv_cache_append(&ext_k, 2));
            layer.values = Some(values.kv_cache_append(&ext_v, 2));
        }
    }
}

pub fn build_cache(_weights: &NativeWeights, config: &Gemma4Config) -> NativeCache {
    NativeCache {
        layers: (0..config.num_hidden_layers as usize)
            .map(|_| LayerCache {
                keys: None,
                values: None,
                offset: 0,
            })
            .collect(),
        rope_offset: 0,
    }
}

// ----------------------------------------------------------------------------
// Loader
// ----------------------------------------------------------------------------

const CACHE_STEP_SIZE: i32 = 256;

fn build_partial_rope_freqs(head_dim: i32, rotated_dims: i32, base: f32) -> Option<InlineArray> {
    if rotated_dims == 0 || rotated_dims == head_dim {
        return None;
    }
    if rotated_dims % 2 != 0 || head_dim % 2 != 0 {
        return None;
    }
    let half = (head_dim / 2) as usize;
    let rot_half = (rotated_dims / 2) as usize;
    let mut freqs = Vec::with_capacity(half);
    for i in 0..rot_half {
        let exponent = (2 * i) as f32 / head_dim as f32;
        freqs.push(base.powf(exponent));
    }
    for _ in rot_half..half {
        freqs.push(f32::INFINITY);
    }
    Some(InlineArray::from_f32_slice(&freqs, &[half as i32]))
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

fn build_sliding_attention_mask(
    offset: i32,
    query_len: i32,
    key_len: i32,
    window: i32,
) -> InlineArray {
    let total_kv = key_len as usize;
    let mut mask_data = vec![0i32; query_len as usize * total_kv];
    for qi in 0..query_len as usize {
        let q_pos = offset + qi as i32;
        let start = (q_pos - window + 1).max(0) as usize;
        let end = q_pos as usize;
        for ki in start..=end.min(total_kv.saturating_sub(1)) {
            mask_data[qi * total_kv + ki] = 1;
        }
    }
    InlineArray::from_i32_slice(&mask_data).reshape(&[query_len, key_len])
}

fn make_additive_mask(bool_mask: &InlineArray, dtype: i32) -> InlineArray {
    let neg_inf = InlineArray::scalar_with_dtype(-1e9, dtype);
    let zero = InlineArray::scalar_with_dtype(0.0, dtype);
    bool_mask
        .as_dtype(Dtype::Bool.as_i32())
        .where_cond(&zero, &neg_inf)
        .as_dtype(dtype)
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
    let activated = nn::gelu_tanh_approximate(&gate);
    let projected = ple
        .projection_w
        .matmul_from(&activated.multiply(layer_per_input))
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
    let source_keys = source_cache
        .keys
        .as_ref()
        .expect("Gemma 4 native: missing shared KV source keys")
        .slice(
            &[0, 0, 0, 0],
            &[b, layer.n_kv_heads, cache_len, layer.head_dim],
        );
    let source_values = source_cache
        .values
        .as_ref()
        .expect("Gemma 4 native: missing shared KV source values")
        .slice(
            &[0, 0, 0, 0],
            &[b, layer.n_kv_heads, cache_len, layer.head_dim],
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

    let attn_out = if let Some(window) = layer.sliding_window {
        if s == 1 {
            let start = (cache_len - window).max(0);
            let keys = source_keys.slice(
                &[0, 0, start, 0],
                &[b, layer.n_kv_heads, cache_len, layer.head_dim],
            );
            let values = source_values.slice(
                &[0, 0, start, 0],
                &[b, layer.n_kv_heads, cache_len, layer.head_dim],
            );
            queries.sdpa_with_mask(&keys, &values, 1.0, None)
        } else {
            let mask = build_sliding_attention_mask(rope_offset, s, cache_len, window);
            let mask = make_additive_mask(&mask, hidden.dtype_raw()).reshape(&[1, 1, s, cache_len]);
            queries.sdpa_with_mask(&source_keys, &source_values, 1.0, Some(&mask))
        }
    } else {
        crate::decode::sdpa_causal_like_mlx(&queries, &source_keys, &source_values, 1.0, s)
    };

    layer
        .o_w
        .matmul_from(&attn_out.transpose_axes(&[0, 2, 1, 3]).reshape(&[
            b,
            s,
            layer.n_heads * layer.head_dim,
        ]))
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
///   `unsloth/gemma-4-31B-it`, with the towers under `model.vision_tower.*`.
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
) -> Result<NativeWeights, String> {
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
        let shard_str = shard
            .to_str()
            .ok_or_else(|| format!("shard path is not valid UTF-8: {shard:?}"))?;
        let Some(pairs) = bridge::load_safetensors_shard(shard_str) else {
            return Err(format!("failed to load safetensors shard {shard_str}"));
        };
        for (key, arr) in pairs {
            let Some(stripped) = normalize_checkpoint_key(&key) else {
                continue;
            };
            // Two layouts normalising onto one key would resolve silently in
            // whatever order the shards happened to be read.
            if raw.insert(stripped.clone(), arr).is_some() {
                return Err(format!(
                    "Gemma 4 native: two checkpoint keys both normalise to {stripped}"
                ));
            }
        }
    }

    // The `quantization` block, if any, read the way mlx-lm reads it. Its
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

    let take = |map: &mut std::collections::HashMap<String, InlineArray>,
                key: &str|
     -> Result<InlineArray, String> {
        map.remove(key)
            .ok_or_else(|| format!("Gemma 4 native: missing weight {key}"))
    };

    // ⚠️ Not from `embed_tokens.weight`: on a quantized checkpoint that is the
    // packed `uint32` payload, and deriving the trunk dtype from it makes
    // every scalar and the whole KV cache integer. The model then loads, runs
    // at the right speed, and decodes noise.
    let model_dtype = detect_model_dtype(|key| raw.get(key).map(|arr| arr.dtype_raw()));

    // One module's three tensors. `scales` decides the variant, which is
    // upstream's rule too: mlx-lm quantizes a module exactly when
    // `{path}.scales` is in the weights.
    let take_proj = |map: &mut std::collections::HashMap<String, InlineArray>,
                     module: &str|
     -> Result<LayerWeight, String> {
        let weight = map
            .remove(&format!("{module}.weight"))
            .ok_or_else(|| format!("Gemma 4 native: missing weight {module}.weight"))?;
        let scales = map.remove(&format!("{module}.scales"));
        let biases = map.remove(&format!("{module}.biases"));
        Ok(LayerWeight::new(weight, scales, biases, params_for(module)))
    };

    let embed_w = EmbeddingWeight::new(
        take(&mut raw, "model.embed_tokens.weight")?,
        raw.remove("model.embed_tokens.scales"),
        raw.remove("model.embed_tokens.biases"),
        params_for("model.embed_tokens"),
    );
    let final_norm_w = take(&mut raw, "model.norm.weight")?;
    let lm_head_w = if config.tie_word_embeddings {
        None
    } else {
        Some(take_proj(&mut raw, "lm_head")?)
    };
    let per_layer_inputs = if config.uses_per_layer_inputs() {
        let total_ple_dim = config.num_hidden_layers * config.per_layer_input_dim();
        Some(PerLayerInputWeights {
            embed_w: EmbeddingWeight::new(
                take(&mut raw, "model.embed_tokens_per_layer.weight")?,
                raw.remove("model.embed_tokens_per_layer.scales"),
                raw.remove("model.embed_tokens_per_layer.biases"),
                params_for("model.embed_tokens_per_layer"),
            ),
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
        let (rope_base, rope_factor) = config.layer_rope(idx);
        let rope_dims = {
            let angles = ((rope_factor * head_dim as f32) / 2.0) as i32;
            (2 * angles).max(0).min(head_dim)
        };
        let sliding_window = if is_full {
            None
        } else {
            Some(config.sliding_window)
        };
        let rope_freqs = build_partial_rope_freqs(head_dim, rope_dims, rope_base);
        let shared_source = if idx >= first_kv_shared_layer_idx && first_kv_shared_layer_idx > 0 {
            Some(config.kv_shared_source_layer(idx).ok_or_else(|| {
                format!(
                    "Gemma 4 native: no KV-sharing source layer found for layer {idx} ({})",
                    if is_full {
                        "full_attention"
                    } else {
                        "sliding_attention"
                    }
                )
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
        let per_layer_gate = if config.uses_per_layer_inputs() {
            Some(PerLayerGateWeights {
                gate_w: take_proj(&mut raw, &format!("{p}.per_layer_input_gate"))?,
                projection_w: take_proj(&mut raw, &format!("{p}.per_layer_projection"))?,
                post_norm_w: take(&mut raw, &format!("{p}.post_per_layer_input_norm.weight"))?,
            })
        } else {
            None
        };

        // Gemma 4 stores `layer_scalar` as an f32 scalar (mlx-lm uses
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

    // Pre-cast scalars once so the forward step never constructs an
    // f32 broadcast and quietly promotes the bf16 residual stream.
    let embed_scale_scalar = InlineArray::scalar_with_dtype(config.embed_scale(), model_dtype);
    let softcap_scalar = config
        .final_logit_softcapping
        .map(|cap| InlineArray::scalar_with_dtype(cap, model_dtype));

    let weights = NativeWeights {
        embed_w,
        per_layer_inputs,
        final_norm_w,
        lm_head_w,
        layers,
        model_dtype,
        config: config.clone(),
        embed_scale_scalar,
        softcap_scalar,
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
// Forward step
// ----------------------------------------------------------------------------

fn ensure_cache_capacity(
    layer_cache: &mut LayerCache,
    needed: i32,
    n_kv: i32,
    head_dim: i32,
    dtype: Dtype,
) {
    let needs_grow = match layer_cache.keys.as_ref() {
        None => true,
        Some(k) => k.dim(2) < needed,
    };
    if !needs_grow {
        return;
    }
    let alloc = ((needed + CACHE_STEP_SIZE - 1) / CACHE_STEP_SIZE) * CACHE_STEP_SIZE;
    let shape = [1, n_kv, alloc, head_dim];
    let new_k = InlineArray::zeros(&shape, dtype.as_i32());
    let new_v = InlineArray::zeros(&shape, dtype.as_i32());
    if let (Some(existing_k), Some(existing_v)) =
        (layer_cache.keys.take(), layer_cache.values.take())
    {
        layer_cache.keys = Some(existing_k.kv_cache_append(&new_k, 2));
        layer_cache.values = Some(existing_v.kv_cache_append(&new_v, 2));
    } else {
        layer_cache.keys = Some(new_k);
        layer_cache.values = Some(new_v);
    }
}

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
/// Structured after `qwen3_native::forward_step`: norms and residuals run
/// as plain per-op calls in Rust, attention goes through
/// `compiled_gemma4_attn_block` (the narrow qkv → sdpa → o_proj fusion),
/// and the MLP is per-op matmuls + a tanh-approx GELU helper. Wrapping
/// the norms and MLP inside bigger compiled lambdas made the graph
/// harder to fuse efficiently; matching Qwen3's layout recovers the
/// single-compile-per-layer pattern that mlx-lm replays.
pub fn forward_step(
    weights: &NativeWeights,
    input_ids: &InlineArray,
    cache: &mut NativeCache,
) -> InlineArray {
    let dtype = Dtype::from_raw(weights.model_dtype);
    let seq_len = input_ids.dim(1);
    let profile = std::env::var_os("PMETAL_GEMMA4_TIMING").is_some();
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
            AttentionKv::Projected { k_w, k_norm_w, v_w } => {
                let layer_cache = &mut cache.layers[i];
                let needed = rope_offset + seq_len;
                ensure_cache_capacity(layer_cache, needed, layer.n_kv_heads, layer.head_dim, dtype);

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
        let mlp_out = if seq_len == 1 {
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

    // 3. Final norm + tied LM head + logit softcap.
    let normed = hidden.rms_norm(Some(&weights.final_norm_w), eps);
    let raw_logits = match weights.lm_head_w.as_ref() {
        Some(w) => w.matmul_from(&normed),
        // Tied embedding: `as_linear` is `mlx.nn.Embedding.as_linear`, which
        // takes the `[vocab, hidden]` table's transpose for a dense model and
        // runs a quantized matmul against the packed rows otherwise.
        None => weights.embed_w.as_linear(&normed),
    };
    match weights.softcap_scalar.as_ref() {
        Some(cap_arr) => {
            use crate::compat::ops;
            let scaled = raw_logits.divide(cap_arr);
            let t = ops::tanh(&scaled);
            t.multiply(cap_arr)
        }
        None => raw_logits,
    }
}

// ----------------------------------------------------------------------------
// Public prefill / generate wrappers
// ----------------------------------------------------------------------------

pub fn prefill_first_token(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    input_ids: &[u32],
    temperature: f32,
) -> u32 {
    crate::decode::prefill_first_token(weights, cache, input_ids, temperature, forward_step)
}

pub fn generate(
    weights: &NativeWeights,
    _config: &Gemma4Config,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    params: crate::decode::SamplingParams,
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
        params.temperature,
        true,
        true,
        move |cache| {
            // Trim the prefill-allocated KV slack then grow to exactly
            // `prompt_len + max_tokens` so decode-time SDPA walks the
            // real cache length instead of the CACHE_STEP_SIZE-rounded
            // allocation. Mirrors qwen3_native's decode priming.
            cache.eval_and_detach_states();
            cache.reserve_decode_inputs(reserve, model_dtype);
        },
        forward_step,
    );
    crate::decode::generate_from_primed_sample_with_params(
        "NATIVE-GEMMA4",
        weights,
        cache,
        current_y,
        max_tokens,
        params,
        true,
        on_token,
        forward_step,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
            rope_parameters: None,
            hidden_size_per_layer_input: Some(2),
            vocab_size_per_layer_input: Some(8),
            hidden_activation: Some("gelu_pytorch_tanh".to_string()),
            num_kv_shared_layers: Some(0),
            use_double_wide_mlp: Some(false),
            enable_moe_block: Some(false),
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
    /// mlx-lm's `gemma3n.py`, which E4B inherits from.
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
            softcap_scalar: None,
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

    #[test]
    fn sliding_attention_mask_respects_offset_window() {
        let mut mask = build_sliding_attention_mask(4, 3, 7, 3);
        let got = mask.to_f32_vec(21).expect("to_f32_vec");
        assert_eq!(
            got,
            vec![
                0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            ]
        );
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
}
