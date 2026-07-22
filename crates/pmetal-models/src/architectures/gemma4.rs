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
//! NOT supported:
//! * MoE block (`enable_moe_block`)
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

use pmetal_core::LoraConfig;
use pmetal_mlx::kernels::fast_lora::create_lora_params;
use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa, rope::apply_rope};
use pmetal_mlx::kv_cache::KVCache;

/// A single low-rank adapter `ΔW = scale · Bᵀ · Aᵀ` applied additively to a
/// base `nn::Linear` output (PEFT-style, baked into the module rather than a
/// parallel model). `a` is `[rank, in]`, `b` is `[out, rank]`; `create_lora_params`
/// zero-initialises `b`, so a freshly-attached adapter is a numerical no-op until
/// trained — attaching LoRA never perturbs inference parity.
#[derive(Debug, Clone)]
pub struct LoraDelta {
    pub a: Array,
    pub b: Array,
    pub scale: f32,
}

impl LoraDelta {
    fn new(in_features: i32, out_features: i32, rank: i32, alpha: f32) -> Result<Self, Exception> {
        let (a, b) = create_lora_params(in_features, out_features, rank)?;
        Ok(Self {
            a,
            b,
            scale: alpha / rank as f32,
        })
    }

    /// `scale · (x · Aᵀ) · Bᵀ` — the additive delta for input `x [.., in]`.
    fn delta(&self, x: &Array) -> Array {
        x.matmul(&self.a.t())
            .matmul(&self.b.t())
            .multiply(&Array::from_f32(self.scale))
    }
}

/// Optional per-projection LoRA adapters for [`Gemma4Attention`]. Populated by
/// `attach_lora`; `None` slots (and the whole `Option` on the attention) mean
/// the base projection is used unchanged. Trainable state is collected via
/// `lora_parameters` / `lora_parameters_mut`, deliberately *outside* the
/// `impl_module_params!` tree so base-weight loading never touches it.
#[derive(Debug, Default, Clone)]
pub struct Gemma4AttnLora {
    pub q: Option<LoraDelta>,
    pub k: Option<LoraDelta>,
    pub v: Option<LoraDelta>,
    pub o: Option<LoraDelta>,
}

/// A 4-/8-bit affine-quantized replacement for a `nn::Linear` weight, used by
/// the QLoRA base. Holds the packed weights + per-group `scales`/`biases`; the
/// forward is MLX's fused `quantized_matmul` (`x @ dequant(w)ᵀ`), so the base
/// weight is never materialised at full precision. `bias`-free, matching the
/// Gemma 4 attention projections.
#[derive(Debug, Clone)]
pub struct QuantLinear {
    pub w_q: Array,
    pub scales: Array,
    pub biases: Array,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantLinear {
    /// Quantize a dense `nn::Linear` weight `[out, in]`. `in` must be a multiple
    /// of `group_size`, and `group_size ∈ {32, 64, 128}` / `bits ∈ {2,3,4,5,6,8}`
    /// (MLX affine quantization). These are validated up front: MLX's `quantize`
    /// throws a *foreign* C++ exception that Rust cannot catch, so an unchecked
    /// bad argument would abort the process.
    pub fn from_linear(proj: &nn::Linear, group_size: i32, bits: i32) -> Result<Self, Exception> {
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(Exception::custom(format!(
                "QuantLinear: unsupported group_size {group_size} (expected 32, 64, or 128)"
            )));
        }
        if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(Exception::custom(format!(
                "QuantLinear: unsupported bits {bits} (expected one of 2,3,4,5,6,8)"
            )));
        }
        let in_features = proj.weight.as_ref().dim(1);
        if in_features % group_size != 0 {
            return Err(Exception::custom(format!(
                "QuantLinear: in_features {in_features} is not a multiple of group_size {group_size}"
            )));
        }
        let (w_q, scales, biases) = proj.weight.as_ref().quantize_weights(group_size, bits);
        Ok(Self {
            w_q,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    /// `x @ dequant(w)ᵀ` via the fused quantized matmul (transpose = weight is
    /// `[out, in]`, matching `nn::Linear`).
    pub fn forward(&self, x: &Array) -> Array {
        x.quantized_matmul(
            &self.w_q,
            &self.scales,
            Some(&self.biases),
            true,
            self.group_size,
            self.bits,
        )
    }
}

/// Optional per-projection quantized base for [`Gemma4Attention`] (QLoRA). When
/// present, the projection forward runs through [`QuantLinear`] instead of the
/// dense `nn::Linear`; `None` slots fall back to the dense weight. Populated by
/// `quantize_projections`; kept outside the module-parameter tree like the LoRA
/// bake-in, so it composes with [`Gemma4AttnLora`].
#[derive(Debug, Default, Clone)]
pub struct Gemma4AttnQuant {
    pub q: Option<QuantLinear>,
    pub k: Option<QuantLinear>,
    pub v: Option<QuantLinear>,
    pub o: Option<QuantLinear>,
}

/// Base projection (dense `nn::Linear` or an optional quantized `quant`) plus an
/// optional LoRA delta. Byte-identical to `proj.forward(x)` when both `quant`
/// and `lora` are `None`, preserving inference parity.
fn linear_lora(
    proj: &nn::Linear,
    quant: Option<&QuantLinear>,
    x: &Array,
    lora: Option<&LoraDelta>,
) -> Array {
    let base = match quant {
        Some(q) => q.forward(x),
        None => proj.forward(x),
    };
    match lora {
        Some(l) => base.add(&l.delta(x)),
        None => base,
    }
}

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
pub(crate) fn apply_gemma4_partial_rope(
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
pub(crate) fn build_gemma4_partial_rope_freqs(
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
    pub vocab_size_per_layer_input: Option<i32>,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub num_kv_shared_layers: Option<i32>,
    #[serde(default)]
    pub use_double_wide_mlp: Option<bool>,
    #[serde(default)]
    pub enable_moe_block: Option<bool>,
    /// MoE expert count (`enable_moe_block` layers only). `None` for dense.
    #[serde(default)]
    pub num_experts: Option<i32>,
    /// Experts activated per token (top-k routing).
    #[serde(default)]
    pub top_k_experts: Option<i32>,
    /// Per-expert FFN intermediate size.
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
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

    pub fn pruned_unsupported_blocks(&self) -> Result<(), Exception> {
        if let Some(ref act) = self.hidden_activation
            && self.uses_per_layer_inputs()
            && !matches!(act.as_str(), "gelu" | "gelu_pytorch_tanh" | "gelu_tanh")
        {
            return Err(Exception::custom(format!(
                "Gemma 4 unsupported per-layer-input activation {act:?}"
            )));
        }
        if self.enable_moe_block.unwrap_or(false)
            && (self.num_experts.is_none()
                || self.top_k_experts.is_none()
                || self.moe_intermediate_size.is_none())
        {
            return Err(Exception::custom(
                "Gemma 4 MoE block enabled but num_experts / top_k_experts / \
                 moe_intermediate_size are missing from the config.",
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
pub(crate) fn rms_norm_noscale(x: &Array, eps: f32) -> Array {
    pmetal_bridge::compat::fast::rms_norm_opt(x, None, eps)
}

fn layer_per_input(per_layer_inputs: &Array, layer_idx: usize) -> Array {
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

// ----------------------------------------------------------------------------
// Per-layer inputs
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct Gemma4PerLayerInputs {
    pub embed_tokens: nn::Embedding,
    pub model_projection: nn::Linear,
    pub projection_norm: Gemma4RmsNorm,
    pub embed_scale: f32,
    pub projection_scale: f32,
    pub input_scale: f32,
    pub num_layers: i32,
    pub per_layer_dim: i32,
    pub vocab_size: i32,
}
impl_module_params!(Gemma4PerLayerInputs; embed_tokens, model_projection, projection_norm);

impl Gemma4PerLayerInputs {
    pub fn new(config: &Gemma4Config) -> Result<Self, Exception> {
        let per_layer_dim = config.per_layer_input_dim();
        let total_ple_dim = config.num_hidden_layers * per_layer_dim;
        Ok(Self {
            embed_tokens: nn::Embedding::new(config.per_layer_input_vocab_size(), total_ple_dim)?,
            model_projection: nn::LinearBuilder::new(config.hidden_size, total_ple_dim)
                .bias(false)
                .build()?,
            projection_norm: Gemma4RmsNorm::new(per_layer_dim, config.rms_norm_eps),
            embed_scale: (per_layer_dim as f32).sqrt(),
            projection_scale: (config.hidden_size as f32).powf(-0.5),
            input_scale: 2.0f32.powf(-0.5),
            num_layers: config.num_hidden_layers,
            per_layer_dim,
            vocab_size: config.per_layer_input_vocab_size(),
        })
    }

    pub fn compute(&self, input_ids: &Array, inputs_embeds: &Array) -> Array {
        let ge_zero = ops::greater_equal(input_ids, &Array::from_i32(0));
        let lt_vocab = ops::less(input_ids, &Array::from_i32(self.vocab_size));
        let mask = ops::logical_and(&ge_zero, &lt_vocab);
        let safe_input_ids = mask.where_cond(input_ids, &ops::zeros_like(input_ids));

        let per_layer_embeds = self
            .embed_tokens
            .forward(&safe_input_ids)
            .multiply(&Array::from_f32(self.embed_scale))
            .reshape(&[
                input_ids.dim(0),
                input_ids.dim(1),
                self.num_layers,
                self.per_layer_dim,
            ]);
        let projection = self
            .model_projection
            .forward(inputs_embeds)
            .multiply(&Array::from_f32(self.projection_scale))
            .reshape(&[
                input_ids.dim(0),
                input_ids.dim(1),
                self.num_layers,
                self.per_layer_dim,
            ]);
        let projection = self.projection_norm.forward(&projection);
        projection
            .add(&per_layer_embeds)
            .multiply(&Array::from_f32(self.input_scale))
    }
}

#[derive(Debug)]
pub struct Gemma4PerLayerInputBlock {
    pub gate_proj: nn::Linear,
    pub projection: nn::Linear,
    pub post_norm: Gemma4RmsNorm,
}
impl_module_params!(Gemma4PerLayerInputBlock; gate_proj, projection, post_norm);

impl Gemma4PerLayerInputBlock {
    pub fn new(config: &Gemma4Config) -> Result<Self, Exception> {
        let per_layer_dim = config.per_layer_input_dim();
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(config.hidden_size, per_layer_dim)
                .bias(false)
                .build()?,
            projection: nn::LinearBuilder::new(per_layer_dim, config.hidden_size)
                .bias(false)
                .build()?,
            post_norm: Gemma4RmsNorm::new(config.hidden_size, config.rms_norm_eps),
        })
    }

    pub fn forward(&mut self, hidden: &Array, layer_input: &Array) -> Result<Array, Exception> {
        let residual = hidden.clone();
        let gate = self.gate_proj.forward(hidden);
        let activated = nn::gelu_tanh_approximate(&gate);
        let projected = self.projection.forward(&activated.multiply(layer_input));
        let projected = self.post_norm.forward(&projected);
        Ok(residual.add(&projected))
    }
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
// MoE block (parallel dense + routed experts)
// ----------------------------------------------------------------------------

/// Gemma 4 MoE router (`Gemma4TextRouter`).
///
/// `softmax(proj(norm(x) · scale · hidden^-0.5))` over experts, then top-k,
/// renormalise to sum 1, and multiply by the learned `per_expert_scale`. The
/// softmax runs in fp32; `norm` is weight-less RMSNorm. Also reused by
/// DiffusionGemma (its `DiffusionGemmaRouter` is an alias).
#[derive(Debug)]
pub struct Gemma4Router {
    pub proj: nn::Linear,
    /// Per-channel pre-projection scale `[hidden_size]`.
    pub scale: Param<Array>,
    /// Per-expert output scale `[num_experts]`.
    pub per_expert_scale: Param<Array>,
    pub top_k: i32,
    pub scalar_root_size: f32,
    pub eps: f32,
}
impl_module_params!(Gemma4Router; proj, scale, per_expert_scale);

impl Gemma4Router {
    pub fn new(
        hidden_size: i32,
        num_experts: i32,
        top_k: i32,
        eps: f32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            proj: nn::LinearBuilder::new(hidden_size, num_experts)
                .bias(false)
                .build()?,
            scale: Param::new(Array::ones_f32(&[hidden_size])),
            per_expert_scale: Param::new(Array::ones_f32(&[num_experts])),
            top_k,
            scalar_root_size: (hidden_size as f32).powf(-0.5),
            eps,
        })
    }

    /// Route a `[N, hidden]` tensor of (flattened) tokens. Returns
    /// `(top_indices, top_weights)`, both `[N, top_k]` — indices `i32`,
    /// weights already renormalised and per-expert-scaled.
    pub fn route(&mut self, hidden_flat: &Array) -> Result<(Array, Array), Exception> {
        let normed = rms_norm_noscale(hidden_flat, self.eps);
        let scaled = normed
            .multiply(self.scale.as_ref())
            .multiply(&Array::from_f32(self.scalar_root_size));
        let scores = self.proj.forward(&scaled);
        let scores_f32 = scores.as_type::<f32>();
        let probs = ops::softmax_axis(&scores_f32, -1);

        let (top_indices, top_weights) =
            crate::moe_routing::topk_normalize(&probs, self.top_k, true)?;

        let n = top_indices.dim(0);
        let k = top_indices.dim(1);
        let idx_flat = top_indices.reshape(&[n * k]);
        let gathered = self
            .per_expert_scale
            .as_ref()
            .take_axis(&idx_flat, 0)
            .reshape(&[n, k]);
        let top_weights = top_weights.multiply(&gathered);
        Ok((top_indices, top_weights))
    }
}

/// Quantized (QLoRA) base for [`Gemma4Experts`]: the fused expert weights in
/// 4-/8-bit affine form. `gate_up`/`down` are each `(packed_weights, scales,
/// biases)` from `quantize_weights`, quantized along the `in` axis (`H` for
/// `gate_up_proj [E, 2I, H]`, `I` for `down_proj [E, H, I]`). The forward routes
/// them through the fused `gather_qmm` (SwitchGLU rank), never materialising the
/// experts at full precision — this is the parameter-mass win for the 26B-A4B
/// model.
#[derive(Debug, Clone)]
pub struct Gemma4ExpertsQuant {
    pub gate_up: (Array, Array, Array),
    pub down: (Array, Array, Array),
    pub group_size: i32,
    pub bits: i32,
}

/// Gemma 4 grouped expert FFNs (`Gemma4TextExperts`).
///
/// Fused 3-D parameters `gate_up_proj [E, 2·I, H]` (gate and up concatenated
/// along the output axis) and `down_proj [E, H, I]`, matching `nn.Linear`
/// weight layout (`out_features` first). Activation is gelu-tanh. Reused by
/// DiffusionGemma (alias `DiffusionGemmaExperts`).
#[derive(Debug)]
pub struct Gemma4Experts {
    pub gate_up_proj: Param<Array>,
    pub down_proj: Param<Array>,
    pub num_experts: i32,
    pub moe_intermediate_size: i32,
    pub hidden_size: i32,
    /// Optional quantized base (QLoRA). `None` = dense path (the pre-transposed
    /// cache below). Populated by [`Gemma4Experts::quantize`]; kept outside the
    /// module-parameter tree.
    quant: Option<Gemma4ExpertsQuant>,
    /// Cached pre-transposed, materialised-contiguous expert weights:
    /// `gate_up_t [E, H, 2·I]` and `down_t [E, I, H]`. The per-token dispatch
    /// matmul then reads a contiguous gathered operand instead of transposing
    /// the fused `[E, 2·I, H]` / `[E, H, I]` params on every forward. Refreshed
    /// when the `Param` handles change (weight load / merge), keyed by
    /// `Array::id()` — the same change-detection gpt_oss's stacked cache uses.
    gate_up_t: Option<Array>,
    down_t: Option<Array>,
    transposed_sig: Option<Vec<usize>>,
}
impl_module_params!(Gemma4Experts; gate_up_proj, down_proj);

impl Gemma4Experts {
    pub fn new(
        num_experts: i32,
        moe_intermediate_size: i32,
        hidden_size: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_up_proj: Param::new(Array::zeros_f32(&[
                num_experts,
                2 * moe_intermediate_size,
                hidden_size,
            ])),
            down_proj: Param::new(Array::zeros_f32(&[
                num_experts,
                hidden_size,
                moe_intermediate_size,
            ])),
            num_experts,
            moe_intermediate_size,
            hidden_size,
            quant: None,
            gate_up_t: None,
            down_t: None,
            transposed_sig: None,
        })
    }

    /// Quantize the fused expert weights to `bits`-bit affine (group size
    /// `group_size`) for the QLoRA base. After this, `forward` routes through the
    /// fused `gather_qmm` instead of the dense pre-transposed cache. Both
    /// `hidden_size` and `moe_intermediate_size` must be a multiple of
    /// `group_size ∈ {32, 64, 128}`. Validated up front — MLX's `quantize`
    /// throws a foreign C++ exception Rust cannot catch.
    pub fn quantize(&mut self, group_size: i32, bits: i32) -> Result<(), Exception> {
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(Exception::custom(format!(
                "Gemma4Experts::quantize: unsupported group_size {group_size} (expected 32, 64, or 128)"
            )));
        }
        if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(Exception::custom(format!(
                "Gemma4Experts::quantize: unsupported bits {bits} (expected one of 2,3,4,5,6,8)"
            )));
        }
        if self.hidden_size % group_size != 0 {
            return Err(Exception::custom(format!(
                "Gemma4Experts::quantize: hidden_size {} not a multiple of group_size {group_size}",
                self.hidden_size
            )));
        }
        if self.moe_intermediate_size % group_size != 0 {
            return Err(Exception::custom(format!(
                "Gemma4Experts::quantize: moe_intermediate_size {} not a multiple of group_size {group_size}",
                self.moe_intermediate_size
            )));
        }
        let gate_up = self
            .gate_up_proj
            .as_ref()
            .quantize_weights(group_size, bits);
        let down = self.down_proj.as_ref().quantize_weights(group_size, bits);
        self.quant = Some(Gemma4ExpertsQuant {
            gate_up,
            down,
            group_size,
            bits,
        });
        Ok(())
    }

    /// Quantized expert dispatch via the fused `gather_qmm` (SwitchGLU rank),
    /// numerically equivalent to the dense [`Self::forward`] within quantization
    /// tolerance. `x [N, H] -> [N, 1, 1, H]`, gathered per `top_indices
    /// [N, top_k]` to `[N, top_k, 1, ·]`, then weighted-summed over `top_k`.
    fn forward_quantized(
        &self,
        hidden_flat: &Array,
        top_indices: &Array,
        top_weights: &Array,
    ) -> Result<Array, Exception> {
        let q = self
            .quant
            .as_ref()
            .expect("forward_quantized: quant present");
        let n = hidden_flat.dim(0);
        let i = self.moe_intermediate_size;
        let top_k = top_indices.dim(1);

        // x: [N, H] -> [N, 1, 1, H] (SwitchGLU rank).
        let switch_in = hidden_flat.expand_dims(1).expand_dims(2);
        // gate_up: [N, top_k, 1, 2I]
        let gate_up = switch_in.gather_qmm(
            &q.gate_up.0,
            &q.gate_up.1,
            Some(&q.gate_up.2),
            None,
            Some(top_indices),
            true,
            q.group_size,
            q.bits,
            false,
        );
        let gate = ops::slice_axis(&gate_up, -1, 0, i);
        let up = ops::slice_axis(&gate_up, -1, i, 2 * i);
        let activated = nn::gelu_tanh_approximate(&gate).multiply(&up); // [N, top_k, 1, I]

        // down: [N, top_k, 1, H] -> squeeze the singleton -> [N, top_k, H]
        let down = activated
            .gather_qmm(
                &q.down.0,
                &q.down.1,
                Some(&q.down.2),
                None,
                Some(top_indices),
                true,
                q.group_size,
                q.bits,
                false,
            )
            .squeeze_axes(&[2]);

        let weighted = down.multiply(&top_weights.reshape(&[n, top_k, 1]));
        Ok(weighted.sum_axis(1, false)) // [N, H]
    }

    /// Signature of the current fused-weight handles, used to invalidate the
    /// transposed cache when the weights are replaced (load / merge).
    fn current_signature(&self) -> Vec<usize> {
        vec![
            self.gate_up_proj.as_ref().id(),
            self.down_proj.as_ref().id(),
        ]
    }

    /// Build or refresh the pre-transposed, materialised expert weights.
    fn ensure_transposed(&mut self) -> Result<(), Exception> {
        let sig = self.current_signature();
        if self.gate_up_t.is_some()
            && self.down_t.is_some()
            && self.transposed_sig.as_ref() == Some(&sig)
        {
            return Ok(());
        }
        // [E, 2I, H] -> [E, H, 2I] and [E, H, I] -> [E, I, H]; eval to force a
        // contiguous materialisation so the per-forward gathered matmul reads a
        // contiguous operand.
        let gate_up_t = self.gate_up_proj.as_ref().transpose_axes(&[0, 2, 1]);
        let down_t = self.down_proj.as_ref().transpose_axes(&[0, 2, 1]);
        gate_up_t.eval();
        down_t.eval();
        self.gate_up_t = Some(gate_up_t);
        self.down_t = Some(down_t);
        self.transposed_sig = Some(sig);
        Ok(())
    }

    /// Eagerly build the transposed-weight cache (optional warm-up; the forward
    /// path builds it lazily on first use anyway).
    pub fn init_expert_cache(&mut self) -> Result<(), Exception> {
        self.ensure_transposed()
    }

    /// Apply the experts to a `[N, hidden]` tensor, dispatching each token to
    /// its `top_indices` experts and weighting by `top_weights`
    /// (both `[N, top_k]`). Returns `[N, hidden]`.
    ///
    /// Uses the pre-transposed weight cache (built lazily on first call and
    /// refreshed when the params change) so the per-slot gathered matmul reads a
    /// contiguous operand instead of transposing the fused params every forward.
    pub fn forward(
        &mut self,
        hidden_flat: &Array,
        top_indices: &Array,
        top_weights: &Array,
    ) -> Result<Array, Exception> {
        if self.quant.is_some() {
            return self.forward_quantized(hidden_flat, top_indices, top_weights);
        }
        self.ensure_transposed()?;
        let gate_up_t = self.gate_up_t.as_ref().expect("transposed cache built");
        let down_t = self.down_t.as_ref().expect("transposed cache built");

        let n = hidden_flat.dim(0);
        let h = self.hidden_size;
        let i = self.moe_intermediate_size;
        let k = top_indices.dim(1);

        let mut out = ops::zeros_dtype(&[n, h], hidden_flat.dtype());
        for slot in 0..k {
            let slot_experts = ops::slice_axis(top_indices, -1, slot, slot + 1).reshape(&[n]);
            let slot_weights = ops::slice_axis(top_weights, -1, slot, slot + 1); // [N, 1]

            let gate_up_w = gate_up_t.take_axis(&slot_experts, 0); // [N, H, 2I]
            let x_b = hidden_flat.reshape(&[n, 1, h]);
            let gate_up = ops::matmul(&x_b, &gate_up_w).squeeze_axes(&[1]); // [N, 2I]
            let gate = ops::slice_axis(&gate_up, -1, 0, i);
            let up = ops::slice_axis(&gate_up, -1, i, 2 * i);
            let activated = nn::gelu_tanh_approximate(&gate).multiply(&up); // [N, I]

            let down_w = down_t.take_axis(&slot_experts, 0); // [N, I, H]
            let act_b = activated.reshape(&[n, 1, i]);
            let down = ops::matmul(&act_b, &down_w).squeeze_axes(&[1]); // [N, H]

            out = out.add(&down.multiply(&slot_weights));
        }
        Ok(out)
    }

    /// Reference forward that transposes the fused params inline every call (no
    /// cache). Kept for the parity guard test that pins the cached path to this.
    #[cfg(test)]
    pub fn forward_reference(
        &self,
        hidden_flat: &Array,
        top_indices: &Array,
        top_weights: &Array,
    ) -> Result<Array, Exception> {
        let n = hidden_flat.dim(0);
        let h = self.hidden_size;
        let i = self.moe_intermediate_size;
        let k = top_indices.dim(1);

        let mut out = ops::zeros_dtype(&[n, h], hidden_flat.dtype());
        for slot in 0..k {
            let slot_experts = ops::slice_axis(top_indices, -1, slot, slot + 1).reshape(&[n]);
            let slot_weights = ops::slice_axis(top_weights, -1, slot, slot + 1);

            let gate_up_w = self
                .gate_up_proj
                .as_ref()
                .take_axis(&slot_experts, 0)
                .transpose_axes(&[0, 2, 1]);
            let x_b = hidden_flat.reshape(&[n, 1, h]);
            let gate_up = ops::matmul(&x_b, &gate_up_w).squeeze_axes(&[1]);
            let gate = ops::slice_axis(&gate_up, -1, 0, i);
            let up = ops::slice_axis(&gate_up, -1, i, 2 * i);
            let activated = nn::gelu_tanh_approximate(&gate).multiply(&up);

            let down_w = self
                .down_proj
                .as_ref()
                .take_axis(&slot_experts, 0)
                .transpose_axes(&[0, 2, 1]);
            let act_b = activated.reshape(&[n, 1, i]);
            let down = ops::matmul(&act_b, &down_w).squeeze_axes(&[1]);

            out = out.add(&down.multiply(&slot_weights));
        }
        Ok(out)
    }
}

/// Gemma 4 per-layer MoE block: the routed-experts branch that runs in
/// *parallel* with the dense MLP and is summed into the same residual. Present
/// only on `enable_moe_block` layers. Holds the router, the grouped experts,
/// and the three MoE-only RMSNorms (the dense branch's post-norm `_1`, and the
/// expert branch's pre/post norms `_2`).
#[derive(Debug)]
pub struct Gemma4MoeBlock {
    pub router: Gemma4Router,
    pub experts: Gemma4Experts,
    pub post_feedforward_layernorm_1: Gemma4RmsNorm,
    pub pre_feedforward_layernorm_2: Gemma4RmsNorm,
    pub post_feedforward_layernorm_2: Gemma4RmsNorm,
}
impl_module_params!(
    Gemma4MoeBlock;
    router,
    experts,
    post_feedforward_layernorm_1,
    pre_feedforward_layernorm_2,
    post_feedforward_layernorm_2
);

impl Gemma4MoeBlock {
    pub fn new(config: &Gemma4Config) -> Result<Self, Exception> {
        let num_experts = config
            .num_experts
            .ok_or_else(|| Exception::custom("Gemma4MoeBlock: num_experts missing from config"))?;
        let top_k = config.top_k_experts.ok_or_else(|| {
            Exception::custom("Gemma4MoeBlock: top_k_experts missing from config")
        })?;
        let moe_intermediate = config.moe_intermediate_size.ok_or_else(|| {
            Exception::custom("Gemma4MoeBlock: moe_intermediate_size missing from config")
        })?;
        let h = config.hidden_size;
        let eps = config.rms_norm_eps;
        Ok(Self {
            router: Gemma4Router::new(h, num_experts, top_k, eps)?,
            experts: Gemma4Experts::new(num_experts, moe_intermediate, h)?,
            post_feedforward_layernorm_1: Gemma4RmsNorm::new(h, eps),
            pre_feedforward_layernorm_2: Gemma4RmsNorm::new(h, eps),
            post_feedforward_layernorm_2: Gemma4RmsNorm::new(h, eps),
        })
    }

    /// Combine the dense-MLP output with the routed-experts output.
    ///
    /// `dense` is the dense branch's output (`mlp(pre_feedforward_layernorm(h))`,
    /// before any post-norm); `residual` is the raw post-attention residual
    /// `[B, S, H]` that the router reads. Returns `dense_1 + moe_2` (each
    /// post-normed), still pre the shared `post_feedforward_layernorm`.
    pub fn forward(&mut self, dense: &Array, residual: &Array) -> Result<Array, Exception> {
        let dense = self.post_feedforward_layernorm_1.forward(dense);

        let b = residual.dim(0);
        let s = residual.dim(1);
        let hidden = residual.dim(2);
        let flat = residual.reshape(&[b * s, hidden]);
        let (top_indices, top_weights) = self.router.route(&flat)?;
        let experts_in = self.pre_feedforward_layernorm_2.forward(&flat);
        let moe = self
            .experts
            .forward(&experts_in, &top_indices, &top_weights)?
            .reshape(&[b, s, hidden]);
        let moe = self.post_feedforward_layernorm_2.forward(&moe);

        Ok(dense.add(&moe))
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
    /// Optional LoRA adapters for the q/k/v/o projections (PEFT bake-in).
    /// `None` = plain attention. Attached via [`Gemma4Attention::attach_lora`];
    /// managed outside the module-parameter tree.
    pub lora: Option<Gemma4AttnLora>,
    /// Optional quantized (QLoRA) base for the q/k/v/o projections. `None` =
    /// dense `nn::Linear`. Populated by [`Gemma4Attention::quantize_projections`];
    /// managed outside the module-parameter tree and composes with `lora`.
    pub qbase: Option<Gemma4AttnQuant>,
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
            lora: None,
            qbase: None,
        })
    }

    /// Quantize the q/k/v/o projection weights to `bits`-bit affine (group size
    /// `group_size`) for the QLoRA base. After this the projection forward runs
    /// through [`QuantLinear`] (fused `quantized_matmul`) instead of the dense
    /// `nn::Linear`. `hidden_size` (and `n_heads·head_dim` for `o_proj`) must be
    /// a multiple of `group_size`. Composes with LoRA adapters attached before
    /// or after.
    pub fn quantize_projections(&mut self, group_size: i32, bits: i32) -> Result<(), Exception> {
        let v = match self.v_proj.as_ref() {
            Some(vp) => Some(QuantLinear::from_linear(vp, group_size, bits)?),
            None => None,
        };
        self.qbase = Some(Gemma4AttnQuant {
            q: Some(QuantLinear::from_linear(&self.q_proj, group_size, bits)?),
            k: Some(QuantLinear::from_linear(&self.k_proj, group_size, bits)?),
            v,
            o: Some(QuantLinear::from_linear(&self.o_proj, group_size, bits)?),
        });
        Ok(())
    }

    /// Attach LoRA adapters to the projections named in `config.target_modules`
    /// (`q_proj` / `k_proj` / `v_proj` / `o_proj`). `v_proj` is skipped on
    /// `k_eq_v` (full-attention) layers that have no value projection. Adapters
    /// initialise to a no-op (`B = 0`), so inference is unchanged until trained.
    pub fn attach_lora(&mut self, config: &LoraConfig) -> Result<(), Exception> {
        let rank = config.r as i32;
        let alpha = config.alpha;
        let has = |m: &str| config.target_modules.iter().any(|t| t == m);
        let shape = |proj: &nn::Linear| {
            let w = proj.weight.as_ref();
            (w.dim(1), w.dim(0)) // (in, out) from [out, in]
        };
        let mk = |proj: &nn::Linear| -> Result<LoraDelta, Exception> {
            let (i, o) = shape(proj);
            LoraDelta::new(i, o, rank, alpha)
        };
        self.lora = Some(Gemma4AttnLora {
            q: if has("q_proj") {
                Some(mk(&self.q_proj)?)
            } else {
                None
            },
            k: if has("k_proj") {
                Some(mk(&self.k_proj)?)
            } else {
                None
            },
            v: match (self.v_proj.as_ref(), has("v_proj")) {
                (Some(vp), true) => Some(mk(vp)?),
                _ => None,
            },
            o: if has("o_proj") {
                Some(mk(&self.o_proj)?)
            } else {
                None
            },
        });
        Ok(())
    }

    /// Named LoRA parameters (`{proj}.lora_{a,b}`) for inventory / counting.
    pub fn lora_parameters(&self) -> Vec<(String, &Array)> {
        let mut out = Vec::new();
        if let Some(l) = &self.lora {
            for (name, slot) in [
                ("q_proj", &l.q),
                ("k_proj", &l.k),
                ("v_proj", &l.v),
                ("o_proj", &l.o),
            ] {
                if let Some(d) = slot {
                    out.push((format!("{name}.lora_a"), &d.a));
                    out.push((format!("{name}.lora_b"), &d.b));
                }
            }
        }
        out
    }

    /// Mutable LoRA parameters for the optimiser (`{proj}.lora_{a,b}`).
    pub fn lora_parameters_mut(&mut self) -> Vec<(String, &mut Array)> {
        let mut out = Vec::new();
        if let Some(lora) = self.lora.as_mut() {
            let Gemma4AttnLora { q, k, v, o } = lora;
            for (name, slot) in [("q_proj", q), ("k_proj", k), ("v_proj", v), ("o_proj", o)] {
                if let Some(d) = slot.as_mut() {
                    out.push((format!("{name}.lora_a"), &mut d.a));
                    out.push((format!("{name}.lora_b"), &mut d.b));
                }
            }
        }
        out
    }

    fn attention_mask_type(
        &self,
        query_len: i32,
        key_len: i32,
        mask: Option<&Array>,
    ) -> AttentionMaskType {
        if mask.is_some() {
            AttentionMaskType::None
        } else if let Some(w) = self.sliding_window {
            if query_len == 1 && key_len <= w {
                AttentionMaskType::None
            } else {
                AttentionMaskType::SlidingWindow(w)
            }
        } else if query_len == 1 {
            AttentionMaskType::None
        } else {
            AttentionMaskType::Causal
        }
    }

    fn attend(
        &mut self,
        q: &Array,
        k: &Array,
        v: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let query_len = q.dim(2);
        let key_len = k.dim(2);
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(1.0)
            .with_mask_type(self.attention_mask_type(query_len, key_len, mask));
        let output = fused_sdpa(q, k, v, &attn_config, mask)?;
        let b = q.dim(0);
        let output = output.transpose_axes(&[0, 2, 1, 3]).reshape(&[
            b,
            query_len,
            self.n_heads * self.head_dim,
        ]);
        Ok(linear_lora(
            &self.o_proj,
            self.qbase.as_ref().and_then(|q| q.o.as_ref()),
            &output,
            self.lora.as_ref().and_then(|l| l.o.as_ref()),
        ))
    }

    fn project_queries(&mut self, x: &Array, offset: i32) -> Result<Array, Exception> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];
        let q = linear_lora(
            &self.q_proj,
            self.qbase.as_ref().and_then(|q| q.q.as_ref()),
            x,
            self.lora.as_ref().and_then(|l| l.q.as_ref()),
        )
        .reshape(&[b, l, self.n_heads, self.head_dim]);
        let q = self.q_norm.forward(&q).transpose_axes(&[0, 2, 1, 3]);
        apply_gemma4_partial_rope(
            &q,
            self.head_dim,
            self.rope_partial_dims,
            self.rope_base,
            offset,
            self.rope_partial_freqs.as_ref(),
        )
    }

    fn project_qkv(&mut self, x: &Array, offset: i32) -> Result<(Array, Array, Array), Exception> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        let lora = self.lora.as_ref();
        let qbase = self.qbase.as_ref();
        let q = linear_lora(
            &self.q_proj,
            qbase.and_then(|q| q.q.as_ref()),
            x,
            lora.and_then(|l| l.q.as_ref()),
        )
        .reshape(&[b, l, self.n_heads, self.head_dim]);
        let k = linear_lora(
            &self.k_proj,
            qbase.and_then(|q| q.k.as_ref()),
            x,
            lora.and_then(|l| l.k.as_ref()),
        )
        .reshape(&[b, l, self.n_kv_heads, self.head_dim]);
        let v_raw = match self.v_proj.as_ref() {
            Some(v_proj) => linear_lora(
                v_proj,
                qbase.and_then(|q| q.v.as_ref()),
                x,
                lora.and_then(|l| l.v.as_ref()),
            )
            .reshape(&[b, l, self.n_kv_heads, self.head_dim]),
            None => k.clone(),
        };

        let q = self.q_norm.forward(&q).transpose_axes(&[0, 2, 1, 3]);
        let k = self.k_norm.forward(&k).transpose_axes(&[0, 2, 1, 3]);
        let v = rms_norm_noscale(&v_raw, self.rms_norm_eps).transpose_axes(&[0, 2, 1, 3]);

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
        Ok((q, k, v))
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        mut cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let (q, k, v) = self.project_qkv(x, offset)?;

        // Update KV cache.
        let (k, v) = if let Some((cache_ref, layer_idx)) = cache.as_mut() {
            (*cache_ref).update_and_fetch(*layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        self.attend(&q, &k, &v, mask)
    }

    pub fn forward_collect_kv(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        offset: i32,
    ) -> Result<(Array, Array, Array), Exception> {
        let (q, k, v) = self.project_qkv(x, offset)?;
        let output = self.attend(&q, &k, &v, mask)?;
        Ok((output, k, v))
    }

    pub fn forward_with_shared_kv(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        source_keys: &Array,
        source_values: &Array,
        offset: i32,
    ) -> Result<Array, Exception> {
        let q = self.project_queries(x, offset)?;
        self.attend(&q, source_keys, source_values, mask)
    }

    /// Cross/self attention that prepends a read-only set of encoder keys and
    /// values to this layer's freshly-projected K/V and attends with an
    /// explicit (bidirectional) mask. Used by the DiffusionGemma decoder,
    /// where each layer reads the encoder KV cache (never writing to it) and
    /// the canvas attends bidirectionally over `[encoder_kv | canvas]`.
    ///
    /// `encoder_keys` / `encoder_values` are `[B, n_kv_heads, enc_len,
    /// head_dim]` (post-norm, post-rope), matching this layer's geometry.
    /// `offset` positions the canvas RoPE after the encoder sequence.
    pub fn forward_with_encoder_kv(
        &mut self,
        x: &Array,
        encoder_keys: &Array,
        encoder_values: &Array,
        mask: Option<&Array>,
        offset: i32,
    ) -> Result<Array, Exception> {
        let (q, k, v) = self.project_qkv(x, offset)?;
        let k = ops::concatenate_axis(&[encoder_keys, &k], 2);
        let v = ops::concatenate_axis(&[encoder_values, &v], 2);
        self.attend(&q, &k, &v, mask)
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
    pub per_layer_input_block: Option<Gemma4PerLayerInputBlock>,
    /// Parallel routed-experts branch, present only on `enable_moe_block`
    /// layers. When `Some`, the dense MLP and the MoE block both feed the
    /// same residual and are summed before `post_feedforward_layernorm`.
    pub moe: Option<Gemma4MoeBlock>,
    /// Per-layer scalar multiplier. The reference stores it as a 1-element
    /// tensor initialised to 1.0; applied as `h = h * layer_scalar` at the
    /// end of the layer forward.
    pub layer_scalar: Param<Array>,
    pub kv_shared_source_layer: Option<usize>,
}
impl_module_params!(
    Gemma4DecoderLayer;
    input_layernorm,
    self_attn,
    post_attention_layernorm,
    pre_feedforward_layernorm,
    mlp,
    post_feedforward_layernorm,
    per_layer_input_block,
    moe,
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
            per_layer_input_block: if config.uses_per_layer_inputs() {
                Some(Gemma4PerLayerInputBlock::new(config)?)
            } else {
                None
            },
            moe: if config.enable_moe_block.unwrap_or(false) {
                Some(Gemma4MoeBlock::new(config)?)
            } else {
                None
            },
            layer_scalar: Param::new(Array::ones_f32(&[1])),
            kv_shared_source_layer: config.kv_shared_source_layer(layer_idx),
        })
    }

    fn finish_forward(
        &mut self,
        residual_in: &Array,
        attn_out: &Array,
        layer_input: Option<&Array>,
    ) -> Result<Array, Exception> {
        let h = self.post_attention_layernorm.forward(attn_out);
        let h = residual_in.add(&h);

        let residual = h.clone();
        let dense = self
            .mlp
            .forward(&self.pre_feedforward_layernorm.forward(&h))?;
        // MoE layers add a routed-experts branch (reading the RAW residual)
        // in parallel with the dense MLP; both are summed before the shared
        // post-feedforward norm. Dense-only layers pass `dense` through.
        let ffn = if let Some(ref mut moe) = self.moe {
            moe.forward(&dense, &residual)?
        } else {
            dense
        };
        let h = self.post_feedforward_layernorm.forward(&ffn);
        let mut h = residual.add(&h);

        if let Some(layer_input) = layer_input
            && let Some(ref mut block) = self.per_layer_input_block
        {
            h = block.forward(&h, layer_input)?;
        }

        Ok(h.multiply(self.layer_scalar.as_ref()))
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        layer_input: Option<&Array>,
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
        self.finish_forward(&residual, &h, layer_input)
    }

    pub fn forward_collect_kv(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        offset: i32,
        layer_input: Option<&Array>,
    ) -> Result<(Array, Array, Array), Exception> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let (attn_out, keys, values) = self.self_attn.forward_collect_kv(&h, mask, offset)?;
        let hidden = self.finish_forward(&residual, &attn_out, layer_input)?;
        Ok((hidden, keys, values))
    }

    pub fn forward_with_shared_kv(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        source_keys: &Array,
        source_values: &Array,
        offset: i32,
        layer_input: Option<&Array>,
    ) -> Result<Array, Exception> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let attn_out =
            self.self_attn
                .forward_with_shared_kv(&h, mask, source_keys, source_values, offset)?;
        self.finish_forward(&residual, &attn_out, layer_input)
    }
}

// ----------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct Gemma4Model {
    pub embed_tokens: nn::Embedding,
    pub per_layer_inputs: Option<Gemma4PerLayerInputs>,
    pub layers: Vec<Gemma4DecoderLayer>,
    pub norm: Gemma4RmsNorm,
    pub config: Gemma4Config,
    pub embed_scale: f32,
}
impl_module_params!(Gemma4Model; embed_tokens, per_layer_inputs, layers, norm);

impl Gemma4Model {
    pub fn new(config: Gemma4Config) -> Result<Self, Exception> {
        config.pruned_unsupported_blocks()?;
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let per_layer_inputs = if config.uses_per_layer_inputs() {
            Some(Gemma4PerLayerInputs::new(&config)?)
        } else {
            None
        };
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| Gemma4DecoderLayer::new(&config, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = Gemma4RmsNorm::new(config.hidden_size, config.rms_norm_eps);
        let embed_scale = (config.hidden_size as f32).sqrt();
        Ok(Self {
            embed_tokens,
            per_layer_inputs,
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
        let per_layer_inputs = self
            .per_layer_inputs
            .as_ref()
            .map(|inputs| inputs.compute(input_ids, &h));
        let mut local_shared_kv = if cache.is_none() && self.config.num_kv_shared_layers() > 0 {
            Some((0..self.layers.len()).map(|_| None).collect::<Vec<_>>())
        } else {
            None
        };
        if let Some(buf) = capture.as_deref_mut()
            && buf.wants_embedding()
        {
            buf.record_embedding(h.clone());
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_input = per_layer_inputs
                .as_ref()
                .map(|inputs| layer_per_input(inputs, i));
            let layer_input_ref = layer_input.as_ref();
            if let Some(shared_source) = layer.kv_shared_source_layer {
                let rope_offset = cache.as_ref().map(|c| c.rope_offset()).unwrap_or(0);
                if let Some(cache_ref) = cache.as_ref() {
                    let (source_keys, source_values) = cache_ref.get(shared_source).ok_or_else(|| {
                        Exception::custom(format!(
                            "Gemma 4 shared-KV layer {i} missing source layer {shared_source} cache"
                        ))
                    })?;
                    h = layer.forward_with_shared_kv(
                        &h,
                        mask,
                        &source_keys,
                        &source_values,
                        rope_offset,
                        layer_input_ref,
                    )?;
                } else {
                    let (source_keys, source_values) = local_shared_kv
                        .as_ref()
                        .and_then(|entries| entries.get(shared_source))
                        .and_then(|entry| entry.as_ref())
                        .ok_or_else(|| {
                            Exception::custom(format!(
                                "Gemma 4 shared-KV layer {i} missing source layer {shared_source} activations"
                            ))
                        })?;
                    h = layer.forward_with_shared_kv(
                        &h,
                        mask,
                        source_keys,
                        source_values,
                        rope_offset,
                        layer_input_ref,
                    )?;
                }
            } else if let Some(ref mut shared_kv) = local_shared_kv {
                let (next_h, keys, values) =
                    layer.forward_collect_kv(&h, mask, 0, layer_input_ref)?;
                shared_kv[i] = Some((keys, values));
                h = next_h;
            } else {
                let c = cache.as_deref_mut().map(|c| (c, i));
                h = layer.forward(&h, mask, c, layer_input_ref)?;
            }
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
        let (_hidden, logits) =
            self.forward_hidden_with_capture(input_ids, mask, cache, capture)?;
        Ok(logits)
    }

    /// Forward pass that returns both final normalized hidden states and logits.
    ///
    /// Gemma 4 MTP assistants consume the target model's final hidden state for
    /// the last accepted token, while the verifier still needs logits. Keeping
    /// this as a single trunk pass avoids recomputing the target block stack.
    pub fn forward_hidden_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        capture: &mut pmetal_mlx::speculative::SpecCapture,
    ) -> Result<(Array, Array), Exception> {
        let hidden = self
            .model
            .forward_with_capture(input_ids, mask, cache, Some(capture))?;
        let logits = self.model.embed_tokens.as_linear(&hidden);
        Ok((hidden, self.logit_softcap(&logits)))
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
    if let Some(ref mut per_layer_inputs) = model.model.per_layer_inputs {
        if let Some(w) = weights.get("model.embed_tokens_per_layer.weight") {
            per_layer_inputs.embed_tokens.weight = Param::new(w.clone());
            report.loaded += 1;
        } else {
            report
                .skipped
                .push("model.embed_tokens_per_layer.weight".to_string());
        }
        load_linear(
            &mut per_layer_inputs.model_projection,
            &weights,
            "model.per_layer_model_projection",
            &mut report,
        );
        load_norm(
            &mut per_layer_inputs.projection_norm.weight,
            &weights,
            "model.per_layer_projection_norm.weight",
            &mut report,
        );
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
        if let Some(ref mut block) = layer.per_layer_input_block {
            load_linear(
                &mut block.gate_proj,
                &weights,
                &format!("{prefix}.per_layer_input_gate"),
                &mut report,
            );
            load_linear(
                &mut block.projection,
                &weights,
                &format!("{prefix}.per_layer_projection"),
                &mut report,
            );
            load_norm(
                &mut block.post_norm.weight,
                &weights,
                &format!("{prefix}.post_per_layer_input_norm.weight"),
                &mut report,
            );
        }

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

        if let Some(ref mut moe) = layer.moe {
            load_linear(
                &mut moe.router.proj,
                &weights,
                &format!("{prefix}.router.proj"),
                &mut report,
            );
            load_norm(
                &mut moe.router.scale,
                &weights,
                &format!("{prefix}.router.scale"),
                &mut report,
            );
            load_norm(
                &mut moe.router.per_expert_scale,
                &weights,
                &format!("{prefix}.router.per_expert_scale"),
                &mut report,
            );
            load_norm(
                &mut moe.experts.gate_up_proj,
                &weights,
                &format!("{prefix}.experts.gate_up_proj"),
                &mut report,
            );
            load_norm(
                &mut moe.experts.down_proj,
                &weights,
                &format!("{prefix}.experts.down_proj"),
                &mut report,
            );
            load_norm(
                &mut moe.post_feedforward_layernorm_1.weight,
                &weights,
                &format!("{prefix}.post_feedforward_layernorm_1.weight"),
                &mut report,
            );
            load_norm(
                &mut moe.pre_feedforward_layernorm_2.weight,
                &weights,
                &format!("{prefix}.pre_feedforward_layernorm_2.weight"),
                &mut report,
            );
            load_norm(
                &mut moe.post_feedforward_layernorm_2.weight,
                &weights,
                &format!("{prefix}.post_feedforward_layernorm_2.weight"),
                &mut report,
            );
        }

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

#[cfg(test)]
mod moe_tests {
    use super::*;
    use serial_test::serial;

    const VOCAB: i32 = 64;
    const HIDDEN: i32 = 32;
    const NUM_EXPERTS: i32 = 4;
    const TOP_K: i32 = 2;
    const MOE_INTER: i32 = 16;

    /// Tiny 2-layer (1 sliding, 1 full) Gemma 4 text config. `moe` toggles the
    /// always-on MoE block.
    fn tiny_config(moe: bool) -> Gemma4Config {
        Gemma4Config {
            model_type: "gemma4_text".to_string(),
            vocab_size: VOCAB,
            hidden_size: HIDDEN,
            intermediate_size: 48,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            global_head_dim: Some(16),
            num_global_key_value_heads: Some(1),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            attention_k_eq_v: true,
            tie_word_embeddings: true,
            sliding_window: 8,
            final_logit_softcapping: Some(30.0),
            layer_types: vec![
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            rope_parameters: None,
            _raw_rope_parameters: None,
            hidden_size_per_layer_input: None,
            vocab_size_per_layer_input: None,
            hidden_activation: Some("gelu_pytorch_tanh".to_string()),
            num_kv_shared_layers: None,
            use_double_wide_mlp: Some(false),
            enable_moe_block: Some(moe),
            num_experts: moe.then_some(NUM_EXPERTS),
            top_k_experts: moe.then_some(TOP_K),
            moe_intermediate_size: moe.then_some(MOE_INTER),
        }
    }

    fn max_abs_diff(a: &Array, b: &Array, len: usize) -> f32 {
        let mut a = a.clone();
        let mut b = b.clone();
        let av = a.to_f32_vec(len).expect("a vec");
        let bv = b.to_f32_vec(len).expect("b vec");
        av.iter()
            .zip(&bv)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// A quantized projection (`QuantLinear`) must track its dense `nn::Linear`
    /// within the expected quantization tolerance: tight at 8-bit, looser at
    /// 4-bit.
    #[test]
    #[serial]
    fn quant_linear_matches_dense_within_tolerance() {
        use pmetal_bridge::compat::{Dtype, random};
        let (in_f, out_f, gs) = (32, 24, 32);
        let w = random::uniform_range(-0.5, 0.5, &[out_f, in_f], Dtype::Float32);
        let mut lin = nn::LinearBuilder::new(in_f, out_f)
            .bias(false)
            .build()
            .unwrap();
        lin.weight = Param::new(w);
        let x = random::uniform_range(-1.0, 1.0, &[3, in_f], Dtype::Float32);
        let dense = lin.forward(&x);
        let len = (3 * out_f) as usize;

        let q8 = QuantLinear::from_linear(&lin, gs, 8).unwrap();
        let d8 = max_abs_diff(&dense, &q8.forward(&x), len);
        assert!(d8 < 0.1, "8-bit quant diff too large: {d8}");

        let q4 = QuantLinear::from_linear(&lin, gs, 4).unwrap();
        let d4 = max_abs_diff(&dense, &q4.forward(&x), len);
        assert!(d4 < 1.5, "4-bit quant diff implausibly large: {d4}");
        // 4-bit must still be coarser than 8-bit (sanity that bits matter).
        assert!(
            d4 >= d8,
            "4-bit ({d4}) should be no tighter than 8-bit ({d8})"
        );
    }

    /// Quantizing the attention projections keeps `forward` finite and close to
    /// the dense output; the default (`qbase = None`) path is unchanged.
    #[test]
    #[serial]
    fn quantize_projections_preserves_attention_within_tolerance() {
        use pmetal_bridge::compat::{Dtype, random};
        let rand = |shape: &[i32]| random::uniform_range(-0.5, 0.5, shape, Dtype::Float32);
        let cfg = tiny_config(false);
        let mut attn = Gemma4Attention::new(&cfg, 0).unwrap();
        let out_q = attn.n_heads * attn.head_dim;
        let out_kv = attn.n_kv_heads * attn.head_dim;
        attn.q_proj.weight = Param::new(rand(&[out_q, HIDDEN]));
        attn.k_proj.weight = Param::new(rand(&[out_kv, HIDDEN]));
        if let Some(v) = attn.v_proj.as_mut() {
            v.weight = Param::new(rand(&[out_kv, HIDDEN]));
        }
        attn.o_proj.weight = Param::new(rand(&[HIDDEN, out_q]));

        let x = rand(&[1, 5, HIDDEN]);
        let dense = attn.forward(&x, None, None).unwrap();
        assert!(attn.qbase.is_none(), "no quant base before quantize");

        attn.quantize_projections(32, 8).unwrap();
        assert!(attn.qbase.is_some(), "quant base present after quantize");
        let quant = attn.forward(&x, None, None).unwrap();

        let len = (5 * HIDDEN) as usize;
        let mut dense_c = dense.clone();
        let scale = dense_c
            .to_f32_vec(len)
            .unwrap()
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(1e-3);
        let d = max_abs_diff(&dense, &quant, len);
        assert!(
            d < 0.15 * scale,
            "8-bit quantized attention diverged: diff={d}, scale={scale}"
        );
    }

    /// The pre-transposed expert cache must produce output identical to the
    /// inline-transpose reference path, and must refresh when the underlying
    /// weights are replaced (load / merge).
    #[test]
    #[serial]
    fn expert_cache_matches_reference_and_refreshes() {
        use pmetal_bridge::compat::{Dtype, random};
        let n = 6;
        let len = (n * HIDDEN) as usize;
        let rand = |shape: &[i32]| random::uniform_range(-0.5, 0.5, shape, Dtype::Float32);

        let mut experts = Gemma4Experts::new(NUM_EXPERTS, MOE_INTER, HIDDEN).unwrap();
        experts.gate_up_proj = Param::new(rand(&[NUM_EXPERTS, 2 * MOE_INTER, HIDDEN]));
        experts.down_proj = Param::new(rand(&[NUM_EXPERTS, HIDDEN, MOE_INTER]));

        let mut router = Gemma4Router::new(HIDDEN, NUM_EXPERTS, TOP_K, 1e-6).unwrap();
        let x = random::uniform_range(-1.0, 1.0, &[n, HIDDEN], Dtype::Float32);
        let (idx, w) = router.route(&x).unwrap();

        let reference = experts.forward_reference(&x, &idx, &w).unwrap();
        let cached = experts.forward(&x, &idx, &w).unwrap();
        assert!(
            max_abs_diff(&cached, &reference, len) < 1e-6,
            "cached expert forward drifted from reference"
        );

        // Replace the weights: the Array::id() signature changes, so the cache
        // must rebuild rather than serve stale transposed tensors.
        experts.gate_up_proj = Param::new(rand(&[NUM_EXPERTS, 2 * MOE_INTER, HIDDEN]));
        let reference2 = experts.forward_reference(&x, &idx, &w).unwrap();
        let cached2 = experts.forward(&x, &idx, &w).unwrap();
        assert!(
            max_abs_diff(&cached2, &reference2, len) < 1e-6,
            "expert cache failed to refresh after weight change"
        );
    }

    /// Quantized (QLoRA) experts via `gather_qmm` must track the dense expert
    /// forward within 8-bit quantization tolerance, and `quantize` must reject an
    /// unsupported group size instead of aborting. Uses `H`/`I` divisible by the
    /// minimum group size (32).
    #[test]
    #[serial]
    fn quantized_experts_match_dense_within_tolerance() {
        use pmetal_bridge::compat::{Dtype, random};
        let (h, i, e, k, n) = (64, 32, NUM_EXPERTS, TOP_K, 6);
        let rand = |shape: &[i32]| random::uniform_range(-0.3, 0.3, shape, Dtype::Float32);

        let mut experts = Gemma4Experts::new(e, i, h).unwrap();
        experts.gate_up_proj = Param::new(rand(&[e, 2 * i, h]));
        experts.down_proj = Param::new(rand(&[e, h, i]));

        let mut router = Gemma4Router::new(h, e, k, 1e-6).unwrap();
        let x = random::uniform_range(-1.0, 1.0, &[n, h], Dtype::Float32);
        let (idx, w) = router.route(&x).unwrap();

        // A bad group size must error cleanly (not abort the process).
        assert!(
            experts.quantize(48, 8).is_err(),
            "48 is not a valid group size"
        );

        let dense = experts.forward(&x, &idx, &w).unwrap();
        experts.quantize(32, 8).unwrap();
        let quant = experts.forward(&x, &idx, &w).unwrap();

        let len = (n * h) as usize;
        let mut dense_c = dense.clone();
        let scale = dense_c
            .to_f32_vec(len)
            .unwrap()
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(1e-3);
        let d = max_abs_diff(&dense, &quant, len);
        assert!(
            d < 0.2 * scale,
            "8-bit quantized experts diverged: diff={d}, scale={scale}"
        );
    }

    /// The MoE block is built only when `enable_moe_block` is set, and is
    /// wired into the module-parameter tree (so load / train / eval traverse
    /// it). A dense config leaves every layer's `moe` as `None`.
    #[test]
    #[serial]
    fn moe_block_present_only_when_enabled_and_wired() {
        let dense = Gemma4ForCausalLM::new(tiny_config(false)).unwrap();
        assert!(dense.model.layers.iter().all(|l| l.moe.is_none()));

        let moe = Gemma4ForCausalLM::new(tiny_config(true)).unwrap();
        assert!(moe.model.layers.iter().all(|l| l.moe.is_some()));

        // The MoE tensors must appear in the flattened parameter map.
        let params = moe.flatten_params();
        assert!(
            params.keys().any(|k| k.ends_with("experts.gate_up_proj")),
            "expected fused expert weights in the param tree"
        );
        assert!(
            params
                .keys()
                .any(|k| k.ends_with("router.per_expert_scale")),
            "expected router per-expert scale in the param tree"
        );
    }

    /// An MoE model runs end-to-end through the parallel dense+MoE layer and
    /// produces finite, correctly-shaped logits.
    #[test]
    #[serial]
    fn moe_forward_runs_and_is_finite() {
        let mut model = Gemma4ForCausalLM::new(tiny_config(true)).unwrap();
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 4, VOCAB]);

        let host = {
            let l = logits.as_type::<f32>();
            l.eval();
            l.as_slice::<f32>().to_vec()
        };
        assert!(
            host.iter().all(|v| v.is_finite()),
            "MoE forward produced non-finite logits"
        );
    }

    /// The weight loader resolves the MoE tensor keys (router + fused experts +
    /// the three MoE-only norms) for every MoE layer.
    #[test]
    #[serial]
    fn load_gemma4_moe_weights_resolves_expert_keys() {
        let mut model = Gemma4ForCausalLM::new(tiny_config(true)).unwrap();

        // Minimal weight set: the required embedding + every layer's MoE keys,
        // each shaped as the loader expects. Non-MoE tensors are intentionally
        // omitted (they land in `skipped`); we only assert the MoE keys load.
        let mut weights: HashMap<String, Array> = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Array::zeros_f32(&[VOCAB, HIDDEN]),
        );
        let mut moe_keys: Vec<String> = Vec::new();
        for i in 0..2 {
            let p = format!("model.layers.{i}");
            let entries: Vec<(String, Vec<i32>)> = vec![
                (format!("{p}.router.proj.weight"), vec![NUM_EXPERTS, HIDDEN]),
                (format!("{p}.router.scale"), vec![HIDDEN]),
                (format!("{p}.router.per_expert_scale"), vec![NUM_EXPERTS]),
                (
                    format!("{p}.experts.gate_up_proj"),
                    vec![NUM_EXPERTS, 2 * MOE_INTER, HIDDEN],
                ),
                (
                    format!("{p}.experts.down_proj"),
                    vec![NUM_EXPERTS, HIDDEN, MOE_INTER],
                ),
                (
                    format!("{p}.post_feedforward_layernorm_1.weight"),
                    vec![HIDDEN],
                ),
                (
                    format!("{p}.pre_feedforward_layernorm_2.weight"),
                    vec![HIDDEN],
                ),
                (
                    format!("{p}.post_feedforward_layernorm_2.weight"),
                    vec![HIDDEN],
                ),
            ];
            for (key, shape) in entries {
                weights.insert(key.clone(), Array::zeros_f32(&shape));
                // The loader strips `.weight` off linear keys before lookup, so
                // track the key form it reports skipped (the full tensor name).
                moe_keys.push(key);
            }
        }

        let report = load_gemma4_weights(&mut model, &weights).unwrap();
        for key in &moe_keys {
            assert!(
                !report.skipped.contains(key),
                "MoE weight {key} was not resolved by the loader"
            );
        }
        // 8 MoE tensors per layer × 2 layers + embedding.
        assert!(report.loaded >= 17, "loaded only {}", report.loaded);
    }
}
