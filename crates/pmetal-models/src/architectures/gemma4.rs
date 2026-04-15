//! Gemma 4 language model (text tower only).
//!
//! This is the text-only path of the Gemma 4 architecture from
//! `Gemma4ForConditionalGeneration`. Ported against the mlx-vlm reference
//! at `mlx_vlm/models/gemma4/language.py`.
//!
//! Supported features (sufficient for the gemma-4-31B checkpoint):
//! * Per-layer-type attention head_dim and num_kv_heads (full-attention
//!   layers use `global_head_dim` and `num_global_key_value_heads`).
//! * `attention_k_eq_v`: full-attention layers have NO `v_proj`; values
//!   are taken from the raw `k_proj` output BEFORE `k_norm` is applied.
//! * `v_norm`: RMSNorm without a learnable scale (applied to values). Uses
//!   the weight-less `fast::rms_norm_opt(x, None, eps)` path to match
//!   Python's `mx.fast.rms_norm(x, None, eps)` — passing an all-ones
//!   weight goes through a different kernel with subtly different rounding.
//! * Per-layer-type RoPE base frequency (full = 1e6, sliding = 1e4).
//! * Partial-rotary RoPE for full-attention layers
//!   (`partial_rotary_factor = 0.25`) via `apply_gemma4_partial_rope`,
//!   which translates mlx-lm's `ProportionalRoPE` — `theta_i = base^(-2i / head_dim)`
//!   — into the standard rope formula by passing
//!   `effective_base = base^(rotated_dims / head_dim)`.
//! * Per-layer `layer_scalar` multiplier applied to the layer output.
//! * Final logit softcap: `softcap * tanh(logits / softcap)`.
//! * Scale factor of `1.0` on SDPA (not `1/sqrt(head_dim)`).
//! * Embedding scale by `sqrt(hidden_size)` (shared with Gemma 2/3).
//! * RMSNorm with learnable scale and NO `+1` offset (a.k.a. `scale_shift=0`
//!   in the mlx-vlm reference).
//! * `gelu_tanh_approx`: the MLP gate activation uses the tanh-based GELU
//!   approximation (matching mlx-lm `nn.gelu_approx`), NOT pmetal's
//!   `nn::gelu_approximate` which maps to the sigmoid fast-approx variant.
//!
//! NOT supported (skipped for the 31B path):
//! * MoE block (`enable_moe_block`)
//! * Per-layer input gating (`hidden_size_per_layer_input`)
//! * KV sharing (`num_kv_shared_layers`)
//! * Double-wide MLP (`use_double_wide_mlp`)
//! * Vision / audio towers
//!
//! # Correctness status
//!
//! Numerically verified against mlx-lm's reference implementation via the
//! `gemma4_synthetic_parity` integration test in
//! `crates/pmetal-models/tests/gemma4_parity.rs`. All tapped checkpoints
//! (post-embedding, each per-layer hidden state, post-norm hidden, softcap
//! logits, argmax tokens) match the Python reference to single-precision
//! round-off (`max_abs_diff ≤ 1e-5` on the f32 synthetic config). The real
//! 31B path runs under the same test when `PMETAL_GEMMA4_REFERENCE` points
//! at a dump produced by `.strategy/parity/dump_gemma4_reference.py`.

use std::collections::HashMap;

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParameters, ModuleParametersExt, Param, nn, ops,
};
use pmetal_bridge::impl_module_params;
use serde::{Deserialize, Serialize};

use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa, rope::apply_rope,
};
use pmetal_mlx::kv_cache::KVCache;

/// Apply Gemma 4 partial rotary embedding to a `[B, H, L, head_dim]` tensor.
///
/// Gemma 4's `ProportionalRoPE` (mlx-lm `rope_utils.py::ProportionalRoPE`)
/// rotates only a fraction of each head dimension and **uses the full
/// `head_dim` as the freq denominator**, not the rotated subset:
///
/// ```text
///     theta_i = base^(-2i / head_dim)     for i in 0..rotated_dims/2
/// ```
///
/// Standard rope (`apply_rope(dims=N)`) computes
/// `theta_i = base^(-2i / N)`, so calling it with `dims = rotated_dims`
/// would use the wrong denominator. We translate by passing
/// `effective_base = base ^ (rotated_dims / head_dim)` so that the standard
/// formula collapses to Gemma 4's:
///
/// ```text
///     effective_base ^ (-2i / rotated_dims)
///         = base ^ ((rotated_dims/head_dim) * (-2i/rotated_dims))
///         = base ^ (-2i / head_dim)         ✓
/// ```
///
/// On top of that, Gemma 4 pairs `(x[i], x[head_dim/2 + i])` rather than
/// `(x[i], x[rotated_dims/2 + i])`, so we have to extract the rotated
/// subset by gathering the first `rotated_dims/2` entries of each half,
/// rotate that contiguous tensor, then scatter the result back into the
/// originally-untouched positions.
fn apply_gemma4_partial_rope(
    x: &Array,
    head_dim: i32,
    rotated_dims: i32,
    base: f32,
    offset: i32,
    partial_freqs: Option<&Array>,
) -> Result<Array, Exception> {
    if rotated_dims == 0 {
        return Ok(x.clone());
    }
    if rotated_dims == head_dim {
        // Full rotation — standard rope works directly.
        return apply_rope(x, head_dim, false, base, 1.0, offset);
    }
    // Fast path: a precomputed `[head_dim / 2]` inverse-frequency array
    // with `inf` in the non-rotated slots lets us call `fast::rope` once
    // over the whole head — no slicing, no concats. This matches
    // mlx-lm's `ProportionalRoPE` and is ~5-7x faster than the manual
    // slice/concat dance (the old fallback path) during decode.
    if let Some(freqs) = partial_freqs {
        return Ok(pmetal_bridge::compat::fast::rope_with_freqs(
            x, head_dim, false, 1.0, offset, freqs,
        ));
    }
    if rotated_dims % 2 != 0 || head_dim % 2 != 0 {
        return Err(Exception::custom(format!(
            "gemma4 partial rope requires even head_dim ({head_dim}) and rotated_dims ({rotated_dims})"
        )));
    }

    let shape = x.shape();
    if shape.len() != 4 {
        return Err(Exception::custom(format!(
            "gemma4 partial rope expects [B,H,L,D], got {shape:?}"
        )));
    }
    let b = shape[0];
    let h = shape[1];
    let l = shape[2];
    let d = shape[3];
    if d != head_dim {
        return Err(Exception::custom(format!(
            "gemma4 partial rope: last dim {d} != head_dim {head_dim}"
        )));
    }

    let half = head_dim / 2;
    let rot_half = rotated_dims / 2;

    // left  = x[..., :half]            (shape [B,H,L,half])
    // right = x[..., half:]            (shape [B,H,L,half])
    let left = x.slice(&[0, 0, 0, 0], &[b, h, l, half]);
    let right = x.slice(&[0, 0, 0, half], &[b, h, l, head_dim]);

    // left_rot  = left[..., :rot_half]   (shape [B,H,L,rot_half])
    // right_rot = right[..., :rot_half]  (shape [B,H,L,rot_half])
    let left_rot = left.slice(&[0, 0, 0, 0], &[b, h, l, rot_half]);
    let right_rot = right.slice(&[0, 0, 0, 0], &[b, h, l, rot_half]);

    // Concatenate the two rotated halves along the last dim and apply
    // standard MLX rope. The resulting tensor's pairs are
    //   (rotated[i], rotated[rot_half + i])  for i in 0..rot_half
    // which corresponds to the original
    //   (left_rot[i], right_rot[i]) = (x[i], x[half + i]).
    let rotated_input = ops::concatenate_axis(&[&left_rot, &right_rot], -1);
    let effective_base = base.powf(rotated_dims as f32 / head_dim as f32);
    let rotated = apply_rope(
        &rotated_input,
        rotated_dims,
        false,
        effective_base,
        1.0,
        offset,
    )?;

    // Split rotated back into its two halves.
    let new_left_rot = rotated.slice(&[0, 0, 0, 0], &[b, h, l, rot_half]);
    let new_right_rot = rotated.slice(&[0, 0, 0, rot_half], &[b, h, l, rotated_dims]);

    // Recombine: replace the first `rot_half` slots of each half with the
    // rotated values, leaving the trailing slots untouched.
    let left_tail = left.slice(&[0, 0, 0, rot_half], &[b, h, l, half]);
    let right_tail = right.slice(&[0, 0, 0, rot_half], &[b, h, l, half]);
    let new_left = ops::concatenate_axis(&[&new_left_rot, &left_tail], -1);
    let new_right = ops::concatenate_axis(&[&new_right_rot, &right_tail], -1);
    Ok(ops::concatenate_axis(&[&new_left, &new_right], -1))
}

/// Build the `[head_dim / 2]` inverse-frequency array used by the fast
/// `rope_with_freqs` path. Non-rotated slots are filled with `f32::INF`
/// so `mx.fast.rope` skips them. Matches mlx-lm's `ProportionalRoPE`:
///
/// ```text
///     freqs[i] = factor * base^(2i / head_dim)   for i in 0..rotated_dims/2
///     freqs[i] = +inf                             for i in rotated_dims/2..head_dim/2
/// ```
///
/// The full `head_dim / 2` length pads the array out to the shape
/// `mx.fast.rope` expects when `dims = head_dim`. Infinity as an inverse
/// frequency means `angle = pos * inf = inf`, which mlx's kernel special-
/// cases to the identity rotation (cos=1, sin=0) — leaving those
/// dimensions untouched.
fn build_gemma4_partial_rope_freqs(
    head_dim: i32,
    rotated_dims: i32,
    base: f32,
) -> Option<Array> {
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
        // Inverse frequency: base^(2i / head_dim). factor=1.0 here; the
        // rope scaling is applied via the `scale` argument to mlx_rope.
        let exponent = (2 * i) as f32 / head_dim as f32;
        freqs.push(base.powf(exponent));
    }
    for _ in rot_half..half {
        freqs.push(f32::INFINITY);
    }
    Some(Array::from_f32_slice(&freqs, &[half as i32]))
}

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_hidden_size() -> i32 {
    5376
}
fn default_num_hidden_layers() -> i32 {
    60
}
fn default_num_attention_heads() -> i32 {
    32
}
fn default_num_key_value_heads() -> i32 {
    16
}
fn default_head_dim() -> i32 {
    256
}
fn default_vocab_size() -> i32 {
    262144
}
fn default_max_position_embeddings() -> i32 {
    262_144
}
fn default_sliding_window() -> i32 {
    1024
}
fn default_final_logit_softcapping() -> Option<f32> {
    Some(30.0)
}
fn default_partial_rotary_factor() -> f32 {
    1.0
}
fn default_rope_theta_sliding() -> f32 {
    10_000.0
}
fn default_rope_theta_global() -> f32 {
    1_000_000.0
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Gemma4RopeLayerConfig {
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Gemma4RopeConfig {
    #[serde(default)]
    pub full_attention: Gemma4RopeLayerConfig,
    #[serde(default)]
    pub sliding_attention: Gemma4RopeLayerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gemma4Config {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default)]
    pub intermediate_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    /// Head dim used by full-attention layers. `None` reuses `head_dim`.
    #[serde(default)]
    pub global_head_dim: Option<i32>,
    /// Number of KV heads used by full-attention layers when
    /// `attention_k_eq_v` is set. `None` reuses `num_key_value_heads`.
    #[serde(default)]
    pub num_global_key_value_heads: Option<i32>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    #[serde(default = "default_final_logit_softcapping")]
    pub final_logit_softcapping: Option<f32>,
    /// Per-layer-type attention mode: `"full_attention"` or `"sliding_attention"`.
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Option<Gemma4RopeConfig>,
    /// Alternative per-layer rope stored as a free-form map (fallback when
    /// `rope_parameters` is not directly deserialisable).
    #[serde(default)]
    pub _raw_rope_parameters: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<i32>,
    #[serde(default)]
    pub num_kv_shared_layers: Option<i32>,
    #[serde(default)]
    pub use_double_wide_mlp: Option<bool>,
    #[serde(default)]
    pub enable_moe_block: Option<bool>,
}

fn default_model_type() -> String {
    "gemma4_text".to_string()
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
        if self.is_full_attention(layer_idx)
            && self.attention_k_eq_v
            && self.num_global_key_value_heads.is_some()
        {
            self.num_global_key_value_heads.unwrap()
        } else {
            self.num_key_value_heads
        }
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

    pub fn pruned_unsupported_blocks(&self) -> Result<(), Exception> {
        if self.hidden_size_per_layer_input.unwrap_or(0) != 0 {
            return Err(Exception::custom(
                "Gemma 4 per-layer input gating (hidden_size_per_layer_input != 0) \
                 is not ported yet — only the 26B/31B path is supported.",
            ));
        }
        if self.num_kv_shared_layers.unwrap_or(0) != 0 {
            return Err(Exception::custom(
                "Gemma 4 KV sharing (num_kv_shared_layers != 0) is not ported yet.",
            ));
        }
        if self.enable_moe_block.unwrap_or(false) {
            return Err(Exception::custom(
                "Gemma 4 MoE block is not ported yet.",
            ));
        }
        if self.use_double_wide_mlp.unwrap_or(false) {
            return Err(Exception::custom(
                "Gemma 4 double-wide MLP is not ported yet.",
            ));
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Building blocks
// ----------------------------------------------------------------------------

/// RMSNorm with a learnable scale but without the Gemma 2/3 `(1+w)` offset.
#[derive(Debug)]
pub struct Gemma4RmsNorm {
    pub weight: Param<Array>,
    pub eps: f32,
}
impl_module_params!(Gemma4RmsNorm; weight);

impl Gemma4RmsNorm {
    pub fn new(dim: i32, eps: f32) -> Self {
        Self {
            weight: Param::new(Array::ones_f32(&[dim])),
            eps,
        }
    }

    pub fn forward(&self, x: &Array) -> Array {
        pmetal_bridge::compat::fast::rms_norm(x, self.weight.as_ref(), self.eps)
    }
}

/// RMSNorm without a learnable scale (Gemma 4's `RMSNormNoScale`). Matches
/// the Python reference which calls `mx.fast.rms_norm(x, None, eps)` —
/// passing `None` lets MLX take the weight-less kernel path instead of
/// materialising an all-ones tensor and doing an identity multiply, which
/// avoids the tiny rounding drift that the ones path introduces.
fn rms_norm_noscale(x: &Array, eps: f32) -> Array {
    pmetal_bridge::compat::fast::rms_norm_opt(x, None, eps)
}


// ----------------------------------------------------------------------------
// MLP
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct Gemma4Mlp {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(Gemma4Mlp; gate_proj, up_proj, down_proj);

impl Gemma4Mlp {
    pub fn new(config: &Gemma4Config) -> Result<Self, Exception> {
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
        // Gemma 4 uses the tanh-approximation GELU as its gate activation.
        // `nn::gelu_approximate` maps to the sigmoid fast-approx variant,
        // which is NOT what mlx-lm's `gelu_approx` computes — see
        // `nn::gelu_tanh_approximate` in the bridge compat layer.
        let gelu_gate = nn::gelu_tanh_approximate(&gate);
        Ok(self.down_proj.forward(&gelu_gate.multiply(&up)))
    }
}

// ----------------------------------------------------------------------------
// Attention (per-layer)
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct Gemma4Attention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: Option<nn::Linear>,
    pub o_proj: nn::Linear,
    pub q_norm: Gemma4RmsNorm,
    pub k_norm: Gemma4RmsNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_base: f32,
    pub rope_partial_dims: i32,
    pub is_full_attention: bool,
    pub rms_norm_eps: f32,
    pub use_k_eq_v: bool,
    pub sliding_window: Option<i32>,
    /// Precomputed inverse-frequency array for the fast partial-rope
    /// path (`[head_dim / 2]`, non-rotated slots are `inf`). Built once
    /// per layer at construction time — `None` for full-rotation layers
    /// that already use the fused-kernel direct path.
    pub rope_partial_freqs: Option<Array>,
}
impl_module_params!(Gemma4Attention; q_proj, k_proj, v_proj, o_proj, q_norm, k_norm);

impl Gemma4Attention {
    pub fn new(config: &Gemma4Config, layer_idx: usize) -> Result<Self, Exception> {
        let head_dim = config.layer_head_dim(layer_idx);
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.layer_num_kv_heads(layer_idx);
        let use_k_eq_v = config.layer_uses_k_eq_v(layer_idx);
        let is_full = config.is_full_attention(layer_idx);
        let (rope_base, rope_factor) = config.layer_rope(layer_idx);
        let rope_partial_dims = {
            let angles = ((rope_factor * head_dim as f32) / 2.0) as i32;
            (2 * angles).max(0).min(head_dim)
        };
        let sliding_window = if is_full {
            None
        } else {
            Some(config.sliding_window)
        };

        let q_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = if use_k_eq_v {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
                    .bias(false)
                    .build()?,
            )
        };
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, config.hidden_size)
            .bias(false)
            .build()?;
        let q_norm = Gemma4RmsNorm::new(head_dim, config.rms_norm_eps);
        let k_norm = Gemma4RmsNorm::new(head_dim, config.rms_norm_eps);

        let rope_partial_freqs =
            build_gemma4_partial_rope_freqs(head_dim, rope_partial_dims, rope_base);
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
            rope_base,
            rope_partial_dims,
            is_full_attention: is_full,
            rms_norm_eps: config.rms_norm_eps,
            use_k_eq_v,
            sliding_window,
            rope_partial_freqs,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        mut cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        // Projections
        let q = self
            .q_proj
            .forward(x)
            .reshape(&[b, l, self.n_heads, self.head_dim]);
        let k = self
            .k_proj
            .forward(x)
            .reshape(&[b, l, self.n_kv_heads, self.head_dim]);

        // v comes from k_proj (pre-norm) when k_eq_v; otherwise from v_proj.
        // Bind through `as_ref` so we call the inherent `Linear::forward`
        // (&self → Array), not the trait method (&mut self → Result).
        let v_raw = match self.v_proj.as_ref() {
            Some(v_proj) => v_proj
                .forward(x)
                .reshape(&[b, l, self.n_kv_heads, self.head_dim]),
            None => k.clone(),
        };

        // Normalise: q_norm, k_norm are learnable RMSNorm; v is RMSNorm without scale.
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
        let v = rms_norm_noscale(&v_raw, self.rms_norm_eps);

        // Transpose to [B, H, L, D] for SDPA.
        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        // Partial / full RoPE via Gemma 4's ProportionalRoPE math (see
        // `apply_gemma4_partial_rope` for the freq derivation).
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let partial_freqs = self.rope_partial_freqs.as_ref();
        let q = apply_gemma4_partial_rope(
            &q,
            self.head_dim,
            self.rope_partial_dims,
            self.rope_base,
            offset,
            partial_freqs,
        )?;
        let k = apply_gemma4_partial_rope(
            &k,
            self.head_dim,
            self.rope_partial_dims,
            self.rope_base,
            offset,
            partial_freqs,
        )?;

        // Update KV cache.
        let (k, v) = if let Some((cache_ref, layer_idx)) = cache.as_mut() {
            (*cache_ref).update_and_fetch(*layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        // SDPA with scale=1.0 (Gemma 4 bakes the scale into the weights).
        // For decode (`query_len == 1`) where the entire KV cache fits
        // inside the sliding window, we can skip mask construction
        // entirely — mlx's fast SDPA handles "no mask" as the cheapest
        // path. mlx-lm relies on the same observation in
        // `create_attention_mask` (returns `None` for `N == 1`).
        let query_len = q.dim(2);
        let key_len = k.dim(2);
        let mask_type = if mask.is_some() {
            AttentionMaskType::None
        } else if let Some(w) = self.sliding_window {
            if query_len == 1 && key_len <= w {
                AttentionMaskType::None
            } else {
                AttentionMaskType::SlidingWindow(w)
            }
        } else if query_len == 1 {
            // Decode against an attention cache: the single query is by
            // construction at the last position, so causal masking is a
            // no-op. Skip the mask build to match mlx-lm's fast path.
            AttentionMaskType::None
        } else {
            AttentionMaskType::Causal
        };
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(1.0)
            .with_mask_type(mask_type);
        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[b, l, self.n_heads * self.head_dim]);
        Ok(self.o_proj.forward(&output))
    }
}

// ----------------------------------------------------------------------------
// Decoder layer
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct Gemma4DecoderLayer {
    pub input_layernorm: Gemma4RmsNorm,
    pub self_attn: Gemma4Attention,
    pub post_attention_layernorm: Gemma4RmsNorm,
    pub pre_feedforward_layernorm: Gemma4RmsNorm,
    pub mlp: Gemma4Mlp,
    pub post_feedforward_layernorm: Gemma4RmsNorm,
    /// Per-layer scalar multiplier. The reference stores it as a 1-element
    /// tensor initialised to 1.0; applied as `h = h * layer_scalar` at the
    /// end of the layer forward.
    pub layer_scalar: Param<Array>,
}
impl_module_params!(
    Gemma4DecoderLayer;
    input_layernorm,
    self_attn,
    post_attention_layernorm,
    pre_feedforward_layernorm,
    mlp,
    post_feedforward_layernorm,
    layer_scalar
);

impl Gemma4DecoderLayer {
    pub fn new(config: &Gemma4Config, layer_idx: usize) -> Result<Self, Exception> {
        Ok(Self {
            input_layernorm: Gemma4RmsNorm::new(config.hidden_size, config.rms_norm_eps),
            self_attn: Gemma4Attention::new(config, layer_idx)?,
            post_attention_layernorm: Gemma4RmsNorm::new(config.hidden_size, config.rms_norm_eps),
            pre_feedforward_layernorm: Gemma4RmsNorm::new(config.hidden_size, config.rms_norm_eps),
            mlp: Gemma4Mlp::new(config)?,
            post_feedforward_layernorm: Gemma4RmsNorm::new(config.hidden_size, config.rms_norm_eps),
            layer_scalar: Param::new(Array::ones_f32(&[1])),
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        // Dynamic-path decoder (used by training, parity tests, and
        // generation when the native bridge isn't available). The fused
        // compiled layer blocks live in `pmetal-bridge::gemma4_native`
        // and require pre-transposed weights, so we keep this side on
        // the plain per-op path — `cargo test gemma4_synthetic_parity`
        // exercises exactly what's below.
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let h = self.self_attn.forward(&h, mask, cache)?;
        let h = self.post_attention_layernorm.forward(&h);
        let h = residual.add(&h);

        let residual = h.clone();
        let h = self.pre_feedforward_layernorm.forward(&h);
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h);
        let h = residual.add(&h);

        Ok(h.multiply(self.layer_scalar.as_ref()))
    }
}

// ----------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct Gemma4Model {
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<Gemma4DecoderLayer>,
    pub norm: Gemma4RmsNorm,
    pub config: Gemma4Config,
    pub embed_scale: f32,
}
impl_module_params!(Gemma4Model; embed_tokens, layers, norm);

impl Gemma4Model {
    pub fn new(config: Gemma4Config) -> Result<Self, Exception> {
        config.pruned_unsupported_blocks()?;
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| Gemma4DecoderLayer::new(&config, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = Gemma4RmsNorm::new(config.hidden_size, config.rms_norm_eps);
        let embed_scale = (config.hidden_size as f32).sqrt();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            config,
            embed_scale,
        })
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        self.forward_with_capture(input_ids, mask, cache, None)
    }

    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
        mut capture: Option<&mut pmetal_mlx::speculative::SpecCapture>,
    ) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(input_ids);
        let scale = Array::from_f32(self.embed_scale);
        h = h.multiply(&scale);
        if let Some(buf) = capture.as_deref_mut()
            && buf.wants_embedding()
        {
            buf.record_embedding(h.clone());
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let c = cache.as_deref_mut().map(|c| (c, i));
            h = layer.forward(&h, mask, c)?;
            if let Some(buf) = capture.as_deref_mut()
                && buf.wants_hidden_for(i)
            {
                buf.record_hidden(i, h.clone());
            }
        }
        Ok(self.norm.forward(&h))
    }
}

// ----------------------------------------------------------------------------
// ForCausalLM
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct Gemma4ForCausalLM {
    pub model: Gemma4Model,
    pub config: Gemma4Config,
}
impl_module_params!(Gemma4ForCausalLM; model);

impl Gemma4ForCausalLM {
    pub fn new(config: Gemma4Config) -> Result<Self, Exception> {
        let model = Gemma4Model::new(config.clone())?;
        Ok(Self { model, config })
    }

    fn logit_softcap(&self, logits: &Array) -> Array {
        if let Some(cap) = self.config.final_logit_softcapping {
            let cap_arr = Array::from_f32(cap);
            let scaled = logits.divide(&cap_arr);
            let tanh = ops::tanh(&scaled);
            tanh.multiply(&cap_arr)
        } else {
            logits.clone()
        }
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward_with_cache(input_ids, mask, cache)?;
        // Gemma 4 ties embeddings; project via transposed embed table.
        let logits = self.model.embed_tokens.as_linear(&hidden);
        Ok(self.logit_softcap(&logits))
    }

    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        capture: &mut pmetal_mlx::speculative::SpecCapture,
    ) -> Result<Array, Exception> {
        let hidden = self
            .model
            .forward_with_capture(input_ids, mask, cache, Some(capture))?;
        let logits = self.model.embed_tokens.as_linear(&hidden);
        Ok(self.logit_softcap(&logits))
    }
}

// ----------------------------------------------------------------------------
// Weight loading
// ----------------------------------------------------------------------------

/// Load Gemma 4 weights into an existing [`Gemma4ForCausalLM`] instance.
///
/// The loader tolerates the Gemma 4 multimodal wrapper by first stripping
/// the `model.language_model.` prefix when present — multimodal checkpoints
/// also carry vision / audio weights which are simply skipped.
pub fn load_gemma4_weights(
    model: &mut Gemma4ForCausalLM,
    raw_weights: &HashMap<String, Array>,
) -> Result<LoadReport, Exception> {
    let mut report = LoadReport::default();

    // Strip multimodal prefix + skip vision / audio tower entries.
    let weights: HashMap<String, Array> = raw_weights
        .iter()
        .filter_map(|(key, value)| {
            let stripped = key
                .strip_prefix("model.language_model.")
                .map(|rest| format!("model.{rest}"))
                .unwrap_or_else(|| key.clone());
            if stripped.contains("embed_vision")
                || stripped.contains("vision_tower")
                || stripped.contains("audio_tower")
                || stripped.contains("multi_modal_projector")
            {
                None
            } else {
                Some((stripped, value.clone()))
            }
        })
        .collect();

    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = Param::new(w.clone());
        report.loaded += 1;
    } else {
        return Err(Exception::custom(
            "Gemma 4: missing model.embed_tokens.weight after prefix strip",
        ));
    }
    if let Some(w) = weights.get("model.norm.weight") {
        model.model.norm.weight = Param::new(w.clone());
        report.loaded += 1;
    }

    for (layer_idx, layer) in model.model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{layer_idx}");

        // Load each norm's learnable weight. Inlined to avoid a slice-of-
        // mut-references dance (which would require `**slot` deref through
        // the for-loop binding).
        load_norm(
            &mut layer.input_layernorm.weight,
            &weights,
            &format!("{prefix}.input_layernorm.weight"),
            &mut report,
        );
        load_norm(
            &mut layer.post_attention_layernorm.weight,
            &weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &mut report,
        );
        load_norm(
            &mut layer.pre_feedforward_layernorm.weight,
            &weights,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
            &mut report,
        );
        load_norm(
            &mut layer.post_feedforward_layernorm.weight,
            &weights,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
            &mut report,
        );
        load_norm(
            &mut layer.self_attn.q_norm.weight,
            &weights,
            &format!("{prefix}.self_attn.q_norm.weight"),
            &mut report,
        );
        load_norm(
            &mut layer.self_attn.k_norm.weight,
            &weights,
            &format!("{prefix}.self_attn.k_norm.weight"),
            &mut report,
        );

        load_linear(
            &mut layer.self_attn.q_proj,
            &weights,
            &format!("{prefix}.self_attn.q_proj"),
            &mut report,
        );
        load_linear(
            &mut layer.self_attn.k_proj,
            &weights,
            &format!("{prefix}.self_attn.k_proj"),
            &mut report,
        );
        if let Some(ref mut v) = layer.self_attn.v_proj {
            load_linear(
                v,
                &weights,
                &format!("{prefix}.self_attn.v_proj"),
                &mut report,
            );
        }
        load_linear(
            &mut layer.self_attn.o_proj,
            &weights,
            &format!("{prefix}.self_attn.o_proj"),
            &mut report,
        );

        load_linear(
            &mut layer.mlp.gate_proj,
            &weights,
            &format!("{prefix}.mlp.gate_proj"),
            &mut report,
        );
        load_linear(
            &mut layer.mlp.up_proj,
            &weights,
            &format!("{prefix}.mlp.up_proj"),
            &mut report,
        );
        load_linear(
            &mut layer.mlp.down_proj,
            &weights,
            &format!("{prefix}.mlp.down_proj"),
            &mut report,
        );

        if let Some(w) = weights.get(&format!("{prefix}.layer_scalar")) {
            layer.layer_scalar = Param::new(w.clone());
            report.loaded += 1;
        } else {
            report.skipped.push(format!("{prefix}.layer_scalar"));
        }
    }

    Ok(report)
}

fn load_linear(
    linear: &mut nn::Linear,
    weights: &HashMap<String, Array>,
    prefix: &str,
    report: &mut LoadReport,
) {
    if let Some(w) = weights.get(&format!("{prefix}.weight")) {
        linear.weight = Param::new(w.clone());
        report.loaded += 1;
    } else {
        report.skipped.push(format!("{prefix}.weight"));
    }
}

fn load_norm(
    slot: &mut Param<Array>,
    weights: &HashMap<String, Array>,
    key: &str,
    report: &mut LoadReport,
) {
    if let Some(w) = weights.get(key) {
        *slot = Param::new(w.clone());
        report.loaded += 1;
    } else {
        report.skipped.push(key.to_string());
    }
}

#[derive(Debug, Default, Clone)]
pub struct LoadReport {
    pub loaded: usize,
    pub skipped: Vec<String>,
}
