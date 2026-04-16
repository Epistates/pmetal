//! Standalone Gemma 4 inference engine — zero dependency on `pmetal-models`.
//!
//! This is the Gemma 4 counterpart of [`crate::qwen3_native`]: every hot-path
//! op goes directly through [`InlineArray`] (stack-allocated `mlx::core::array`
//! values routed through the `mlx_inline_*` C bridge), so there are no heap
//! allocations or Rust wrapper roundtrips per op.
//!
//! Gemma 4-specific features (26B/31B path — the other variants are blocked
//! on follow-up work):
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
//! * Final logit softcap: `softcap * tanh(logits / softcap)`.
//! * Embedding scale by `sqrt(hidden_size)` (shared with Gemma 2/3).
//! * Tanh-approximation GELU in the MLP (matching mlx-lm's `nn.gelu_approx`).
//!
//! Weights are pre-transposed to `[in, out]` at load time (`w.t()`) so the
//! per-decode matmuls are contiguous.

use serde::Deserialize;

use crate::InlineArray;
use crate::compat::Dtype;
use crate::inline_array as bridge;

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
    // `hidden_size_per_layer_input=0`, no MoE — all are strictly checked.
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<i32>,
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

    fn validate_supported(&self) -> Result<(), String> {
        if self.hidden_size_per_layer_input.unwrap_or(0) != 0 {
            return Err(
                "Gemma 4 native: per-layer-input gating (2B/4B models) is not ported yet"
                    .to_string(),
            );
        }
        if self.num_kv_shared_layers.unwrap_or(0) != 0 {
            return Err(
                "Gemma 4 native: KV sharing (num_kv_shared_layers != 0) is not ported yet"
                    .to_string(),
            );
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
    let path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
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

/// Per-layer weight bundle. All dense linears are pre-transposed to
/// `[in, out]` form so the matmul hot path is contiguous.
pub struct LayerWeights {
    pub input_norm_w: InlineArray,
    pub q_w: InlineArray,
    pub k_w: InlineArray,
    /// `None` for full-attention layers under `attention_k_eq_v` (values
    /// come from the raw k_proj output).
    pub v_w: Option<InlineArray>,
    pub o_w: InlineArray,
    pub q_norm_w: InlineArray,
    pub k_norm_w: InlineArray,
    pub post_attn_norm_w: InlineArray,
    pub pre_ffn_norm_w: InlineArray,
    pub gate_w: InlineArray,
    pub up_w: InlineArray,
    pub down_w: InlineArray,
    pub post_ffn_norm_w: InlineArray,
    pub layer_scalar: InlineArray,
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
    pub embed_w: InlineArray,
    pub final_norm_w: InlineArray,
    pub lm_head_w: Option<InlineArray>,
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

/// Load a Gemma 4 checkpoint into [`NativeWeights`]. Dense linears are
/// pre-transposed at load time so the per-step matmuls are contiguous.
pub fn load_model(
    model_dir: &std::path::Path,
    config: &Gemma4Config,
) -> Result<NativeWeights, String> {
    let model_path_str = model_dir
        .to_str()
        .ok_or_else(|| "model path is not valid UTF-8".to_string())?;

    // Gather all safetensors shards in the directory.
    let mut shard_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)
        .map_err(|e| format!("read_dir({model_path_str}): {e}"))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shard_files.sort();
    if shard_files.is_empty() {
        return Err(format!("no .safetensors shards found in {model_path_str}"));
    }

    // Collect every `(key, array)` pair across shards. Stripping the
    // `model.language_model.` prefix lets us consume both text-only and
    // multimodal checkpoints transparently.
    let mut raw: std::collections::HashMap<String, InlineArray> = std::collections::HashMap::new();
    for shard in &shard_files {
        let shard_str = shard
            .to_str()
            .ok_or_else(|| format!("shard path is not valid UTF-8: {shard:?}"))?;
        let Some(pairs) = bridge::load_safetensors_shard(shard_str) else {
            return Err(format!("failed to load safetensors shard {shard_str}"));
        };
        for (key, arr) in pairs {
            let stripped = key
                .strip_prefix("model.language_model.")
                .map(|rest| format!("model.{rest}"))
                .unwrap_or(key.clone());
            if stripped.contains("embed_vision")
                || stripped.contains("vision_tower")
                || stripped.contains("audio_tower")
                || stripped.contains("multi_modal_projector")
            {
                continue;
            }
            raw.insert(stripped, arr);
        }
    }

    let take = |map: &mut std::collections::HashMap<String, InlineArray>,
                key: &str|
     -> Result<InlineArray, String> {
        map.remove(key)
            .ok_or_else(|| format!("Gemma 4 native: missing weight {key}"))
    };

    let embed_w = take(&mut raw, "model.embed_tokens.weight")?;
    let final_norm_w = take(&mut raw, "model.norm.weight")?;
    let lm_head_w = if config.tie_word_embeddings {
        None
    } else {
        Some(take(&mut raw, "lm_head.weight")?.t())
    };
    let model_dtype = embed_w.dtype().as_i32();

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
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

        let input_norm_w = take(&mut raw, &format!("{p}.input_layernorm.weight"))?;
        let post_attn_norm_w = take(&mut raw, &format!("{p}.post_attention_layernorm.weight"))?;
        let pre_ffn_norm_w = take(&mut raw, &format!("{p}.pre_feedforward_layernorm.weight"))?;
        let post_ffn_norm_w = take(&mut raw, &format!("{p}.post_feedforward_layernorm.weight"))?;
        let q_norm_w = take(&mut raw, &format!("{p}.self_attn.q_norm.weight"))?;
        let k_norm_w = take(&mut raw, &format!("{p}.self_attn.k_norm.weight"))?;

        let q_w = take(&mut raw, &format!("{p}.self_attn.q_proj.weight"))?.t();
        let k_w = take(&mut raw, &format!("{p}.self_attn.k_proj.weight"))?.t();
        let v_w = if use_k_eq_v {
            None
        } else {
            Some(take(&mut raw, &format!("{p}.self_attn.v_proj.weight"))?.t())
        };
        let o_w = take(&mut raw, &format!("{p}.self_attn.o_proj.weight"))?.t();

        let gate_w = take(&mut raw, &format!("{p}.mlp.gate_proj.weight"))?.t();
        let up_w = take(&mut raw, &format!("{p}.mlp.up_proj.weight"))?.t();
        let down_w = take(&mut raw, &format!("{p}.mlp.down_proj.weight"))?.t();

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
            k_w,
            v_w,
            o_w,
            q_norm_w,
            k_norm_w,
            post_attn_norm_w,
            pre_ffn_norm_w,
            gate_w,
            up_w,
            down_w,
            post_ffn_norm_w,
            layer_scalar,
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
    let embed_scale_scalar = InlineArray::from_f32(config.embed_scale()).as_dtype(model_dtype);
    let softcap_scalar = config
        .final_logit_softcapping
        .map(|cap| InlineArray::from_f32(cap).as_dtype(model_dtype));

    let weights = NativeWeights {
        embed_w,
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
        l.k_w.async_eval_ref();
        l.o_w.async_eval_ref();
        l.gate_w.async_eval_ref();
        l.up_w.async_eval_ref();
        l.down_w.async_eval_ref();
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
    let dtype = weights.embed_w.dtype();
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
        .take_axis(input_ids, 0)
        .multiply(&weights.embed_scale_scalar);

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
        let layer_cache = &mut cache.layers[i];
        let needed = rope_offset + seq_len;
        ensure_cache_capacity(layer_cache, needed, layer.n_kv_heads, layer.head_dim, dtype);

        let cache_k = layer_cache.keys.take().unwrap();
        let cache_v = layer_cache.values.take().unwrap();

        let t_attn_start = if profile {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Wide compiled attention block: includes input_layernorm at
        // the top and post_attention_layernorm at the bottom. Measured
        // to be ~15-20% faster on 31B than the narrower "norms-outside"
        // variant — probably because the extra per-op RMS-norm kernels
        // become their own dispatch-cost floor for large hidden sizes.
        let (attn_out, new_k, new_v) = InlineArray::compiled_gemma4_attn_block(
            &hidden,
            &layer.input_norm_w,
            &layer.q_w,
            &layer.k_w,
            layer.v_w.as_ref(),
            &layer.o_w,
            &layer.q_norm_w,
            &layer.k_norm_w,
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
        let mlp_in = h.rms_norm(Some(&layer.pre_ffn_norm_w), eps);
        let gate = mlp_in.matmul(&layer.gate_w);
        let up = mlp_in.matmul(&layer.up_w);
        let activated = InlineArray::fused_geglu_tanh(&gate, &up);
        let down = activated.matmul(&layer.down_w);
        let mlp_out = down.rms_norm(Some(&layer.post_ffn_norm_w), eps);

        hidden = h.add(&mlp_out).multiply(&layer.layer_scalar);

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
        Some(w) => normed.matmul(w),
        // Tied embedding: `embed_w` is stored as `[vocab, hidden]`. Use
        // its `.t()` view so the matmul shape `[B, T, hidden] @
        // [hidden, vocab]` lines up. `.t()` is metadata-only in mlx.
        None => normed.matmul(&weights.embed_w.t()),
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
