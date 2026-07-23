//! Gemma 4 vision tower (`gemma4_vision`).
//!
//! A bespoke Google vision transformer (NOT standard SigLIP / CLIP) used as the
//! image backbone by the Gemma 4 multimodal family and, wholesale, by
//! DiffusionGemma (`DiffusionGemma uses Gemma 4's vision block`). It differs
//! from a textbook ViT in almost every block:
//!
//! * **Patch embed** ([`Gemma4VisionPatchEmbedder`]): a *linear* projection of
//!   flattened `3·patch²` pixel patches (no conv), pixel-normalised as
//!   `2·(x − 0.5)`, plus a 2-axis learned `position_embedding_table` looked up
//!   by `(x, y)` patch coordinates (padding patches at `(-1, -1)` are zeroed).
//! * **Attention** ([`Gemma4VisionAttention`]): q/k/v/o projections with
//!   per-head `q_norm`/`k_norm` (scaled RMSNorm) and a weight-less `v_norm`,
//!   `scaling = 1.0`, GQA, and **multidimensional RoPE** (independent x/y
//!   frequency bands, θ = 100) applied to q and k. Fully bidirectional.
//! * **MLP** ([`Gemma4VisionMLP`]): gelu-tanh SwiGLU.
//! * **Encoder layer** ([`Gemma4VisionEncoderLayer`]): the standard Gemma
//!   4-norm sandwich (input / post-attention / pre-feedforward /
//!   post-feedforward), *without* the text tower's `layer_scalar`.
//! * **Pooler** ([`Gemma4VisionPooler`]): 2-D average pooling by patch position
//!   into `num_patches / pooling_kernel²` soft tokens, then a `√hidden` scaling
//!   computed in fp32.
//!
//! The RMSNorms are exactly [`Gemma4RmsNorm`] / [`rms_norm_noscale`] (plain-`w`,
//! eps-inside-sqrt), so those verified helpers are reused directly.
//!
//! # Scope
//!
//! This module ports the vision **tower** ([`Gemma4VisionModel`]): pixels →
//! pooled, `√hidden`-scaled soft-token features. The projection into the text
//! embedding space (`DiffusionGemmaMultimodalEmbedder`) and the merge into the
//! encoder's token embeddings live in [`super::diffusion_gemma`].

use std::collections::HashMap;

use pmetal_bridge::compat::{Array, Dtype, Exception, Param, nn, ops};
use pmetal_bridge::impl_module_params;
use serde::{Deserialize, Serialize};

use super::gemma4::{Gemma4RmsNorm, LoadReport, rms_norm_noscale};

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------

fn default_vision_model_type() -> String {
    "gemma4_vision".to_string()
}
fn default_vision_hidden_size() -> i32 {
    768
}
fn default_vision_intermediate_size() -> i32 {
    3072
}
fn default_vision_num_hidden_layers() -> i32 {
    16
}
fn default_vision_num_attention_heads() -> i32 {
    12
}
fn default_vision_num_key_value_heads() -> i32 {
    12
}
fn default_vision_head_dim() -> i32 {
    64
}
fn default_vision_hidden_activation() -> String {
    "gelu_pytorch_tanh".to_string()
}
fn default_vision_rms_norm_eps() -> f32 {
    1e-6
}
fn default_vision_max_position_embeddings() -> i32 {
    131_072
}
fn default_vision_pooling_kernel_size() -> i32 {
    3
}
fn default_vision_patch_size() -> i32 {
    16
}
fn default_vision_position_embedding_size() -> i32 {
    10 * 1024
}
fn default_vision_rope_theta() -> f32 {
    100.0
}

/// Gemma 4 vision-tower configuration (`Gemma4VisionConfig`, `model_type:
/// gemma4_vision`).
///
/// `use_clipped_linears` (activation clipping for the QAT path) and
/// `standardize` (post-pool whitening) default off — the released fp path
/// treats every `Gemma4ClippableLinear` as a plain bias-less `Linear`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gemma4VisionConfig {
    #[serde(default = "default_vision_model_type")]
    pub model_type: String,
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_vision_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_vision_num_key_value_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_vision_head_dim")]
    pub head_dim: i32,
    #[serde(default = "default_vision_hidden_activation")]
    pub hidden_activation: String,
    #[serde(default = "default_vision_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_vision_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default = "default_vision_pooling_kernel_size")]
    pub pooling_kernel_size: i32,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: i32,
    #[serde(default = "default_vision_position_embedding_size")]
    pub position_embedding_size: i32,
    /// RoPE base θ (the reference default RoPE uses `rope_theta = 100.0`). When
    /// the checkpoint expresses this as a `rope_parameters` dict, the loader
    /// folds `rope_theta` into this field.
    #[serde(default = "default_vision_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default)]
    pub standardize: bool,
}

impl Default for Gemma4VisionConfig {
    fn default() -> Self {
        Self {
            model_type: default_vision_model_type(),
            hidden_size: default_vision_hidden_size(),
            intermediate_size: default_vision_intermediate_size(),
            num_hidden_layers: default_vision_num_hidden_layers(),
            num_attention_heads: default_vision_num_attention_heads(),
            num_key_value_heads: default_vision_num_key_value_heads(),
            head_dim: default_vision_head_dim(),
            hidden_activation: default_vision_hidden_activation(),
            rms_norm_eps: default_vision_rms_norm_eps(),
            max_position_embeddings: default_vision_max_position_embeddings(),
            pooling_kernel_size: default_vision_pooling_kernel_size(),
            patch_size: default_vision_patch_size(),
            position_embedding_size: default_vision_position_embedding_size(),
            rope_theta: default_vision_rope_theta(),
            use_clipped_linears: false,
            standardize: false,
        }
    }
}

// ----------------------------------------------------------------------------
// RoPE helpers (multidimensional / 2-D)
// ----------------------------------------------------------------------------

/// Index of the last axis of `a`.
fn last_axis(a: &Array) -> i32 {
    a.shape().len() as i32 - 1
}

/// `rotate_half`: split the last axis in two and return `[-x2, x1]`.
fn rotate_half(x: &Array) -> Array {
    let ax = last_axis(x);
    let d = x.dim(ax);
    let half = d / 2;
    let x1 = ops::slice_axis(x, ax, 0, half);
    let x2 = ops::slice_axis(x, ax, half, d);
    ops::concatenate_axis(&[&x2.multiply(&Array::from_f32(-1.0)), &x1], ax)
}

/// Apply one rotary band to a `[B, N, heads, D]` tensor with `[B, N, D]`
/// cos/sin (unsqueezed on the head axis).
fn apply_rotary(x: &Array, cos: &Array, sin: &Array) -> Array {
    let cos_u = cos.expand_dims(2);
    let sin_u = sin.expand_dims(2);
    x.multiply(&cos_u).add(&rotate_half(x).multiply(&sin_u))
}

/// Multidimensional RoPE (`apply_multidimensional_rope`, `ndim = 2`): split the
/// per-head channels into two equal spatial bands and rotate each with its own
/// (x / y) cos/sin. `x` is `[B, N, heads, head_dim]`; `cos`/`sin` are
/// `[B, N, head_dim]`.
fn apply_multidim_rope(x: &Array, cos: &Array, sin: &Array) -> Array {
    let ndim = 2i32;
    let channels = x.dim(last_axis(x));
    // Channels rotated per spatial dimension: 2·(C / (2·ndim)).
    let per_dim = 2 * (channels / (2 * ndim));
    let xax = last_axis(x);
    let cax = last_axis(cos);
    let mut parts = Vec::with_capacity(ndim as usize);
    for k in 0..ndim {
        let (s, e) = (k * per_dim, (k + 1) * per_dim);
        let x_k = ops::slice_axis(x, xax, s, e);
        let cos_k = ops::slice_axis(cos, cax, s, e);
        let sin_k = ops::slice_axis(sin, cax, s, e);
        parts.push(apply_rotary(&x_k, &cos_k, &sin_k));
    }
    let refs: Vec<&Array> = parts.iter().collect();
    ops::concatenate_axis(&refs, xax)
}

/// Precomputes the RoPE inverse frequencies and produces `(cos, sin)` from 2-D
/// patch position ids. Carries no learnable parameters.
#[derive(Debug)]
pub struct Gemma4VisionRotaryEmbedding {
    /// `[count]` inverse frequencies, `count = head_dim / 4`.
    inv_freq: Array,
}

impl Gemma4VisionRotaryEmbedding {
    pub fn new(config: &Gemma4VisionConfig) -> Self {
        // Each spatial dim rotates `head_dim / 2` channels; its frequency band
        // is `arange(0, spatial_dim, 2) / spatial_dim`, i.e. `head_dim / 4`
        // distinct frequencies.
        let spatial_dim = config.head_dim / 2;
        let count = spatial_dim / 2;
        let theta = config.rope_theta;
        let inv_freq: Vec<f32> = (0..count)
            .map(|j| {
                let exponent = (2 * j) as f32 / spatial_dim as f32;
                1.0 / theta.powf(exponent)
            })
            .collect();
        Self {
            inv_freq: Array::from_slice(&inv_freq, &[count.max(1)]),
        }
    }

    /// `position_ids`: `[B, N, 2]` (x, y) integer patch coordinates. Returns
    /// `(cos, sin)` each `[B, N, head_dim]`.
    pub fn forward(&self, position_ids: &Array) -> (Array, Array) {
        let pos_f = position_ids.as_type::<f32>();
        let count = self.inv_freq.dim(0);
        let inv = self.inv_freq.reshape(&[1, 1, count]);
        let mut cos_parts = Vec::with_capacity(2);
        let mut sin_parts = Vec::with_capacity(2);
        for i in 0..2 {
            let dim_pos = ops::slice_axis(&pos_f, 2, i, i + 1).squeeze(2); // [B, N]
            let freqs = dim_pos.expand_dims(-1).multiply(&inv); // [B, N, count]
            let emb = ops::concatenate_axis(&[&freqs, &freqs], -1); // [B, N, spatial_dim]
            cos_parts.push(ops::cos(&emb));
            sin_parts.push(ops::sin(&emb));
        }
        let cos = ops::concatenate_axis(&[&cos_parts[0], &cos_parts[1]], -1);
        let sin = ops::concatenate_axis(&[&sin_parts[0], &sin_parts[1]], -1);
        (cos, sin)
    }
}

// ----------------------------------------------------------------------------
// MLP
// ----------------------------------------------------------------------------

/// Gelu-tanh SwiGLU MLP (`Gemma4VisionMLP`).
#[derive(Debug)]
pub struct Gemma4VisionMLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(Gemma4VisionMLP; gate_proj, up_proj, down_proj);

impl Gemma4VisionMLP {
    pub fn new(config: &Gemma4VisionConfig) -> Result<Self, Exception> {
        let h = config.hidden_size;
        let i = config.intermediate_size;
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(h, i).bias(false).build()?,
            up_proj: nn::LinearBuilder::new(h, i).bias(false).build()?,
            down_proj: nn::LinearBuilder::new(i, h).bias(false).build()?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = nn::gelu_tanh_approximate(&self.gate_proj.forward(x));
        Ok(self
            .down_proj
            .forward(&gate.multiply(&self.up_proj.forward(x))))
    }
}

// ----------------------------------------------------------------------------
// Attention
// ----------------------------------------------------------------------------

/// Multi-head (GQA-capable) vision attention (`Gemma4VisionAttention`).
///
/// `scaling = 1.0`; q/k get per-head scaled RMSNorm then multidimensional RoPE;
/// v gets a weight-less RMSNorm and no RoPE. Fully bidirectional (an optional
/// additive `mask` handles patch padding).
#[derive(Debug)]
pub struct Gemma4VisionAttention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub q_norm: Gemma4RmsNorm,
    pub k_norm: Gemma4RmsNorm,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
}
impl_module_params!(Gemma4VisionAttention; q_proj, k_proj, v_proj, o_proj, q_norm, k_norm);

impl Gemma4VisionAttention {
    pub fn new(config: &Gemma4VisionConfig) -> Result<Self, Exception> {
        let h = config.hidden_size;
        let hd = config.head_dim;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        Ok(Self {
            q_proj: nn::LinearBuilder::new(h, nh * hd).bias(false).build()?,
            k_proj: nn::LinearBuilder::new(h, nkv * hd).bias(false).build()?,
            v_proj: nn::LinearBuilder::new(h, nkv * hd).bias(false).build()?,
            o_proj: nn::LinearBuilder::new(nh * hd, h).bias(false).build()?,
            q_norm: Gemma4RmsNorm::new(hd, config.rms_norm_eps),
            k_norm: Gemma4RmsNorm::new(hd, config.rms_norm_eps),
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            rms_norm_eps: config.rms_norm_eps,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let b = x.dim(0);
        let n = x.dim(1);

        let q = self
            .q_proj
            .forward(x)
            .reshape(&[b, n, self.num_heads, self.head_dim]);
        let q = self.q_norm.forward(&q);
        let q = apply_multidim_rope(&q, cos, sin).transpose_axes(&[0, 2, 1, 3]);

        let k = self
            .k_proj
            .forward(x)
            .reshape(&[b, n, self.num_kv_heads, self.head_dim]);
        let k = self.k_norm.forward(&k);
        let k = apply_multidim_rope(&k, cos, sin).transpose_axes(&[0, 2, 1, 3]);

        let v = self
            .v_proj
            .forward(x)
            .reshape(&[b, n, self.num_kv_heads, self.head_dim]);
        let v = rms_norm_noscale(&v, self.rms_norm_eps).transpose_axes(&[0, 2, 1, 3]);

        // MLX SDPA broadcasts the KV heads for GQA. scaling = 1.0.
        let out = q.sdpa_with_mask(&k, &v, 1.0, mask);
        let out =
            out.transpose_axes(&[0, 2, 1, 3])
                .reshape(&[b, n, self.num_heads * self.head_dim]);
        Ok(self.o_proj.forward(&out))
    }
}

// ----------------------------------------------------------------------------
// Encoder layer + encoder
// ----------------------------------------------------------------------------

/// A vision encoder layer: the Gemma 4-norm sandwich (no `layer_scalar`).
#[derive(Debug)]
pub struct Gemma4VisionEncoderLayer {
    pub self_attn: Gemma4VisionAttention,
    pub mlp: Gemma4VisionMLP,
    pub input_layernorm: Gemma4RmsNorm,
    pub post_attention_layernorm: Gemma4RmsNorm,
    pub pre_feedforward_layernorm: Gemma4RmsNorm,
    pub post_feedforward_layernorm: Gemma4RmsNorm,
}
impl_module_params!(
    Gemma4VisionEncoderLayer;
    self_attn,
    mlp,
    input_layernorm,
    post_attention_layernorm,
    pre_feedforward_layernorm,
    post_feedforward_layernorm
);

impl Gemma4VisionEncoderLayer {
    pub fn new(config: &Gemma4VisionConfig) -> Result<Self, Exception> {
        let h = config.hidden_size;
        let eps = config.rms_norm_eps;
        Ok(Self {
            self_attn: Gemma4VisionAttention::new(config)?,
            mlp: Gemma4VisionMLP::new(config)?,
            input_layernorm: Gemma4RmsNorm::new(h, eps),
            post_attention_layernorm: Gemma4RmsNorm::new(h, eps),
            pre_feedforward_layernorm: Gemma4RmsNorm::new(h, eps),
            post_feedforward_layernorm: Gemma4RmsNorm::new(h, eps),
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let h = self.self_attn.forward(&h, cos, sin, mask)?;
        let h = self.post_attention_layernorm.forward(&h);
        let h = residual.add(&h);

        let residual = h.clone();
        let ff = self.pre_feedforward_layernorm.forward(&h);
        let ff = self.mlp.forward(&ff)?;
        let ff = self.post_feedforward_layernorm.forward(&ff);
        Ok(residual.add(&ff))
    }
}

/// The vision encoder stack (`Gemma4VisionEncoder`).
#[derive(Debug)]
pub struct Gemma4VisionEncoder {
    pub layers: Vec<Gemma4VisionEncoderLayer>,
    rotary_emb: Gemma4VisionRotaryEmbedding,
}
impl_module_params!(Gemma4VisionEncoder; layers);

impl Gemma4VisionEncoder {
    pub fn new(config: &Gemma4VisionConfig) -> Result<Self, Exception> {
        let layers = (0..config.num_hidden_layers)
            .map(|_| Gemma4VisionEncoderLayer::new(config))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers,
            rotary_emb: Gemma4VisionRotaryEmbedding::new(config),
        })
    }

    /// `inputs_embeds`: `[B, N, hidden]`; `position_ids`: `[B, N, 2]`; `mask`:
    /// optional additive `[.., N, N]` padding mask (`None` = fully visible).
    pub fn forward(
        &mut self,
        inputs_embeds: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let (cos, sin) = self.rotary_emb.forward(position_ids);
        let mut h = inputs_embeds.clone();
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, &cos, &sin, mask)?;
        }
        Ok(h)
    }
}

// ----------------------------------------------------------------------------
// Patch embedder
// ----------------------------------------------------------------------------

/// Linear patch embedding + 2-D learned position table (`Gemma4VisionPatchEmbedder`).
#[derive(Debug)]
pub struct Gemma4VisionPatchEmbedder {
    pub input_proj: nn::Linear,
    /// `[2, position_embedding_size, hidden]` — row 0 = x, row 1 = y.
    pub position_embedding_table: Param<Array>,
}
impl_module_params!(Gemma4VisionPatchEmbedder; input_proj, position_embedding_table);

impl Gemma4VisionPatchEmbedder {
    pub fn new(config: &Gemma4VisionConfig) -> Result<Self, Exception> {
        let h = config.hidden_size;
        let patch_dim = 3 * config.patch_size * config.patch_size;
        Ok(Self {
            input_proj: nn::LinearBuilder::new(patch_dim, h).bias(false).build()?,
            position_embedding_table: Param::new(Array::ones_f32(&[
                2,
                config.position_embedding_size,
                h,
            ])),
        })
    }

    /// Sum of the x- and y-position embeddings for each patch; padding patches
    /// (`padding_positions` true) are zeroed.
    fn position_embeddings(&self, position_ids: &Array, padding_positions: &Array) -> Array {
        let table = self.position_embedding_table.as_ref();
        let table_x = ops::slice_axis(table, 0, 0, 1).squeeze(0); // [pos_size, hidden]
        let table_y = ops::slice_axis(table, 0, 1, 2).squeeze(0);

        // Clamp negatives (padding) to 0 before the lookup; the result is zeroed
        // below regardless.
        let clamped = ops::maximum(position_ids, &ops::zeros_like(position_ids));
        let x_idx = ops::slice_axis(&clamped, 2, 0, 1)
            .squeeze(2)
            .as_type::<i32>();
        let y_idx = ops::slice_axis(&clamped, 2, 1, 2)
            .squeeze(2)
            .as_type::<i32>();

        let x_emb = ops::take_axis(&table_x, &x_idx, 0); // [B, N, hidden]
        let y_emb = ops::take_axis(&table_y, &y_idx, 0);
        let pos = x_emb.add(&y_emb);
        ops::where_fn(
            &padding_positions.expand_dims(-1),
            &ops::zeros_like(&pos),
            &pos,
        )
    }

    /// `pixel_values`: `[B, N, 3·patch²]`; `position_ids`: `[B, N, 2]`;
    /// `padding_positions`: `[B, N]` bool. Returns `[B, N, hidden]`.
    pub fn forward(
        &mut self,
        pixel_values: &Array,
        position_ids: &Array,
        padding_positions: &Array,
    ) -> Array {
        // Gemma 4 scales pixels in model code instead of normalising the input.
        let px = pixel_values
            .multiply(&Array::from_f32(2.0))
            .subtract(&Array::from_f32(1.0));
        let hidden = self.input_proj.forward(&px);
        hidden.add(&self.position_embeddings(position_ids, padding_positions))
    }
}

// ----------------------------------------------------------------------------
// Pooler
// ----------------------------------------------------------------------------

/// 2-D spatial pooler with `√hidden` fp32 scaling (`Gemma4VisionPooler`).
/// Carries no learnable parameters.
#[derive(Debug)]
pub struct Gemma4VisionPooler {
    root_hidden_size: f32,
}

impl Gemma4VisionPooler {
    pub fn new(config: &Gemma4VisionConfig) -> Self {
        Self {
            root_hidden_size: (config.hidden_size as f32).sqrt(),
        }
    }

    /// Average patches within each `k×k` spatial block (`k` inferred from the
    /// input/output length ratio) using position ids, returning
    /// `([B, output_length, hidden], valid_mask [B, output_length])`.
    fn avg_pool_by_positions(
        &self,
        hidden_states: &Array,
        position_ids: &Array,
        output_length: i32,
    ) -> Result<(Array, Array), Exception> {
        let input_seq_len = hidden_states.dim(1);
        let k = ((input_seq_len / output_length) as f64).sqrt() as i32;
        let k_squared = k * k;
        if k_squared * output_length != input_seq_len {
            return Err(Exception::custom(format!(
                "cannot pool seq_len {input_seq_len} to {output_length}: k={k}^2 * length must equal seq_len"
            )));
        }

        let pos_f = position_ids.as_type::<f32>();
        let clamped = ops::maximum(&pos_f, &ops::zeros_like(&pos_f));
        let x_coord = ops::slice_axis(&clamped, 2, 0, 1).squeeze(2); // [B, N]
        let y_coord = ops::slice_axis(&clamped, 2, 1, 2).squeeze(2);

        let kf = Array::from_f32(k as f32);
        let kx = ops::floor(&x_coord.divide(&kf));
        let ky = ops::floor(&y_coord.divide(&kf));
        // max_x = max(x) + 1; number of pooled columns = floor(max_x / k).
        let max_x = ops::max_axis(&x_coord, 1, true).add(&Array::from_f32(1.0)); // [B, 1]
        let cols = ops::floor(&max_x.divide(&kf));
        let flat_idx = kx.add(&cols.multiply(&ky)); // [B, N]
        let flat_idx = flat_idx.as_type::<i32>();

        // one_hot(flat_idx, output_length) / k² → [B, N, output_length].
        let ar = ops::arange(output_length, Dtype::Int32).reshape(&[1, 1, output_length]);
        let onehot = ops::equal(&flat_idx.expand_dims(-1), &ar).as_type::<f32>();
        let weights = onehot.divide(&Array::from_f32(k_squared as f32));

        // output = weightsᵀ @ hidden → [B, output_length, hidden].
        let output = ops::matmul(
            &weights.transpose_axes(&[0, 2, 1]),
            &hidden_states.as_type::<f32>(),
        );
        // A block is valid if any patch maps to it (column not all-zero).
        let col_sum = weights.sum_axis(1, false); // [B, output_length]
        let mask = ops::greater(&col_sum, &Array::from_f32(0.0));
        Ok((output, mask))
    }

    /// `hidden_states`: `[B, N, hidden]`; `position_ids`: `[B, N, 2]`;
    /// `padding_positions`: `[B, N]` bool; `output_length`: soft-token count.
    /// Returns `([B, output_length, hidden] fp32, valid_mask [B, output_length])`.
    pub fn forward(
        &self,
        hidden_states: &Array,
        position_ids: &Array,
        padding_positions: &Array,
        output_length: i32,
    ) -> Result<(Array, Array), Exception> {
        // Zero padding patches so they contribute nothing to the average.
        let masked = ops::where_fn(
            &padding_positions.expand_dims(-1),
            &ops::zeros_like(hidden_states),
            hidden_states,
        );

        let (mut hidden, mask) = if hidden_states.dim(1) != output_length {
            self.avg_pool_by_positions(&masked, position_ids, output_length)?
        } else {
            // No pooling needed: every soft token is valid.
            let b = masked.dim(0);
            let z = ops::zeros(&[b, output_length], Dtype::Int32);
            (masked.as_type::<f32>(), ops::equal(&z, &z))
        };

        // √hidden scaling in fp32 (can exceed the fp16 range).
        hidden = hidden.multiply(&Array::from_f32(self.root_hidden_size));
        Ok((hidden, mask))
    }
}

// ----------------------------------------------------------------------------
// Vision model
// ----------------------------------------------------------------------------

/// The Gemma 4 vision encoder (`Gemma4VisionModel`): pixels → pooled,
/// `√hidden`-scaled soft-token features `[B, output_length, hidden]`.
///
/// The reference strips padding soft tokens (`hidden_states[pooler_mask]`,
/// flattening to `[valid, hidden]`). For a single un-padded image that is a
/// no-op reshape; ragged multi-image stripping happens where features are
/// merged into text (see [`super::diffusion_gemma`]).
#[derive(Debug)]
pub struct Gemma4VisionModel {
    pub patch_embedder: Gemma4VisionPatchEmbedder,
    pub encoder: Gemma4VisionEncoder,
    pooler: Gemma4VisionPooler,
    pooling_kernel_size: i32,
    /// Optional post-pool whitening buffers (`standardize`); `None` unless the
    /// checkpoint provides them.
    pub std_bias: Option<Array>,
    pub std_scale: Option<Array>,
}
impl_module_params!(Gemma4VisionModel; patch_embedder, encoder);

impl Gemma4VisionModel {
    pub fn new(config: &Gemma4VisionConfig) -> Result<Self, Exception> {
        Ok(Self {
            patch_embedder: Gemma4VisionPatchEmbedder::new(config)?,
            encoder: Gemma4VisionEncoder::new(config)?,
            pooler: Gemma4VisionPooler::new(config),
            pooling_kernel_size: config.pooling_kernel_size,
            std_bias: None,
            std_scale: None,
        })
    }

    /// Encode `pixel_values` `[B, N, 3·patch²]` at `position_ids` `[B, N, 2]`
    /// (padding patches marked `(-1, -1)`). Returns pooled features
    /// `[B, output_length, hidden]`.
    pub fn forward(
        &mut self,
        pixel_values: &Array,
        position_ids: &Array,
    ) -> Result<Array, Exception> {
        let n = pixel_values.dim(1);
        let k = self.pooling_kernel_size;
        let output_length = n / (k * k);

        // padding_positions = (position_ids == -1).all(axis=-1): a patch is
        // padding iff both (x, y) coordinates are -1.
        let neg_one = Array::from_f32(-1.0).as_type::<i32>();
        let is_pad = ops::equal(position_ids, &neg_one).as_type::<f32>();
        let padding_positions = ops::greater(&is_pad.sum_axis(2, false), &Array::from_f32(1.5)); // [B, N] bool

        let inputs_embeds =
            self.patch_embedder
                .forward(pixel_values, position_ids, &padding_positions);

        // No-padding fast path uses a fully-visible mask (None).
        let hidden = self.encoder.forward(&inputs_embeds, position_ids, None)?;

        let (mut pooled, _mask) =
            self.pooler
                .forward(&hidden, position_ids, &padding_positions, output_length)?;

        if let (Some(bias), Some(scale)) = (&self.std_bias, &self.std_scale) {
            pooled = pooled.subtract(bias).multiply(scale);
        }
        Ok(pooled)
    }
}

// ----------------------------------------------------------------------------
// Multimodal embedder (vision → text projection)
// ----------------------------------------------------------------------------

/// Projects pooled vision soft tokens into the text model's embedding space
/// (`Gemma4MultimodalEmbedder`): a weight-less pre-projection RMSNorm followed
/// by a bias-less linear from the vision hidden size to the text hidden size.
/// DiffusionGemma reuses this verbatim (`DiffusionGemmaMultimodalEmbedder`).
#[derive(Debug)]
pub struct Gemma4MultimodalEmbedder {
    pub embedding_projection: nn::Linear,
    eps: f32,
}
impl_module_params!(Gemma4MultimodalEmbedder; embedding_projection);

impl Gemma4MultimodalEmbedder {
    pub fn new(multimodal_hidden: i32, text_hidden: i32, eps: f32) -> Result<Self, Exception> {
        Ok(Self {
            embedding_projection: nn::LinearBuilder::new(multimodal_hidden, text_hidden)
                .bias(false)
                .build()?,
            eps,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Array {
        let normed = rms_norm_noscale(x, self.eps);
        self.embedding_projection.forward(&normed)
    }

    /// Load the projection weight (the pre-projection norm is weight-less).
    /// `weights` is rooted at the embedder, so the key is
    /// `embedding_projection.weight`.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, report: &mut LoadReport) {
        vload_linear(
            &mut self.embedding_projection,
            weights,
            "embedding_projection.weight",
            report,
        );
    }
}

// ----------------------------------------------------------------------------
// Weight loading
// ----------------------------------------------------------------------------

fn vload_linear(
    linear: &mut nn::Linear,
    weights: &HashMap<String, Array>,
    key: &str,
    report: &mut LoadReport,
) {
    if let Some(w) = weights.get(key) {
        linear.weight = Param::new(w.clone());
        report.loaded += 1;
    } else {
        report.skipped.push(key.to_string());
    }
}

fn vload_param(
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

/// Load HF `Gemma4VisionModel` weights, with keys rooted at the vision model
/// (`patch_embedder.*`, `encoder.layers.{i}.*`). The `Gemma4ClippableLinear`
/// projections wrap an inner `nn.Linear`, so their weights live under
/// `…{proj}.linear.weight`; `input_proj` is a plain `nn.Linear`, and
/// `position_embedding_table` is a raw parameter. `v_norm` is weight-less.
pub fn load_gemma4_vision_weights(
    model: &mut Gemma4VisionModel,
    weights: &HashMap<String, Array>,
) -> Result<LoadReport, Exception> {
    let mut report = LoadReport::default();

    vload_linear(
        &mut model.patch_embedder.input_proj,
        weights,
        "patch_embedder.input_proj.weight",
        &mut report,
    );
    vload_param(
        &mut model.patch_embedder.position_embedding_table,
        weights,
        "patch_embedder.position_embedding_table",
        &mut report,
    );

    for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
        let p = format!("encoder.layers.{i}");
        for (proj, key) in [
            (&mut layer.self_attn.q_proj, "q_proj"),
            (&mut layer.self_attn.k_proj, "k_proj"),
            (&mut layer.self_attn.v_proj, "v_proj"),
            (&mut layer.self_attn.o_proj, "o_proj"),
        ] {
            vload_linear(
                proj,
                weights,
                &format!("{p}.self_attn.{key}.linear.weight"),
                &mut report,
            );
        }
        vload_param(
            &mut layer.self_attn.q_norm.weight,
            weights,
            &format!("{p}.self_attn.q_norm.weight"),
            &mut report,
        );
        vload_param(
            &mut layer.self_attn.k_norm.weight,
            weights,
            &format!("{p}.self_attn.k_norm.weight"),
            &mut report,
        );
        for (proj, key) in [
            (&mut layer.mlp.gate_proj, "gate_proj"),
            (&mut layer.mlp.up_proj, "up_proj"),
            (&mut layer.mlp.down_proj, "down_proj"),
        ] {
            vload_linear(
                proj,
                weights,
                &format!("{p}.mlp.{key}.linear.weight"),
                &mut report,
            );
        }
        for (norm, key) in [
            (&mut layer.input_layernorm.weight, "input_layernorm"),
            (
                &mut layer.post_attention_layernorm.weight,
                "post_attention_layernorm",
            ),
            (
                &mut layer.pre_feedforward_layernorm.weight,
                "pre_feedforward_layernorm",
            ),
            (
                &mut layer.post_feedforward_layernorm.weight,
                "post_feedforward_layernorm",
            ),
        ] {
            vload_param(norm, weights, &format!("{p}.{key}.weight"), &mut report);
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny vision config with all axes divisible by the pooling/RoPE
    /// constraints (head_dim % 4 == 0; num_patches = k²·output_length).
    fn tiny_vision_config() -> Gemma4VisionConfig {
        Gemma4VisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 8,
            pooling_kernel_size: 2,
            patch_size: 4,
            position_embedding_size: 64,
            rope_theta: 100.0,
            ..Default::default()
        }
    }

    /// A regular `grid×grid` patch grid with row-major (x, y) coordinates.
    fn grid_positions(grid: i32) -> Array {
        let mut coords = Vec::with_capacity((grid * grid * 2) as usize);
        for y in 0..grid {
            for x in 0..grid {
                coords.push(x);
                coords.push(y);
            }
        }
        Array::from_slice(&coords, &[1, grid * grid, 2])
    }

    #[test]
    fn vision_tower_forward_shape_and_finite() {
        let config = tiny_vision_config();
        let mut model = Gemma4VisionModel::new(&config).unwrap();

        let grid = 4; // 16 patches → k²=4 → output_length = 4
        let n = grid * grid;
        let patch_dim = 3 * config.patch_size * config.patch_size;
        let pixel_values = pmetal_bridge::compat::random::uniform_f32(&[1, n, patch_dim]);
        let position_ids = grid_positions(grid);

        let out = model.forward(&pixel_values, &position_ids).unwrap();
        assert_eq!(out.shape(), &[1, 4, config.hidden_size]);

        let mut out = out;
        out.eval();
        let v = out.to_f32_vec((4 * config.hidden_size) as usize).unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "vision output has non-finite values"
        );
    }

    #[test]
    fn rotary_produces_head_dim_cos_sin() {
        let config = tiny_vision_config();
        let rope = Gemma4VisionRotaryEmbedding::new(&config);
        let position_ids = grid_positions(4);
        let (cos, sin) = rope.forward(&position_ids);
        assert_eq!(cos.shape(), &[1, 16, config.head_dim]);
        assert_eq!(sin.shape(), &[1, 16, config.head_dim]);
    }
}
