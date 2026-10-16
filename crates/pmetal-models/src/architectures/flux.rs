//! The FLUX.1 transformer.
//!
//! `FluxTransformer2DModel` from `diffusers` (Black Forest Labs, 2024): a
//! rectified-flow diffusion transformer over packed 2x2 latent patches. Text
//! (T5 states, plus a pooled CLIP vector folded into the conditioning) and
//! image tokens run through `num_layers` **double-stream** blocks, which keep
//! separate weights per modality and attend jointly, then through
//! `num_single_layers` **single-stream** blocks that treat the concatenated
//! sequence as one, with the attention and the MLP in parallel.
//!
//! The parameter tree uses `diffusers`' own names (`transformer_blocks.0.attn.
//! to_q`, `single_transformer_blocks.3.proj_mlp`, `time_text_embed.
//! timestep_embedder.linear_1`, ...), so a pipeline's `transformer/` directory
//! loads without translation, and adapters trained against `diffusers` name
//! the layers they were trained on. The two places `diffusers` spells a
//! parameter path with a list index that a field name cannot (`ff.net.0.proj`,
//! `ff.net.2`) are `ff.proj_in` and `ff.proj_out` here; the loader renames.
//!
//! Position is 3-axis rotary: a token's `(0, row, column)` id (text tokens
//! are all zeros) rotates each head's channels in interleaved pairs, the first
//! `axes_dims_rope[0]` channels by the first id, and so on.

use pmetal_runtime::compat::{Array, Dtype, Exception, nn, ops};
use pmetal_runtime::impl_module_params;
use pmetal_runtime::nn::{AttentionMaskType, FusedAttentionConfig, fused_sdpa};
use serde::{Deserialize, Serialize};

/// The rotary base, fixed by the architecture.
const ROPE_THETA: f64 = 10_000.0;
/// Width of the sinusoidal timestep (and guidance) embedding.
const TIMESTEP_DIM: usize = 256;
/// Epsilon of every LayerNorm and RMSNorm in the model.
const NORM_EPS: f32 = 1e-6;
/// The MLP's expansion over the hidden size.
const MLP_RATIO: usize = 4;

/// FLUX transformer configuration, in the field names of its `diffusers`
/// `config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxConfig {
    /// Channels of a packed latent patch (`4 * latent_channels`).
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    /// Channels of the prediction; the input's when absent.
    #[serde(default)]
    pub out_channels: Option<usize>,
    /// Double-stream blocks.
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    /// Single-stream blocks.
    #[serde(default = "default_num_single_layers")]
    pub num_single_layers: usize,
    #[serde(default = "default_attention_head_dim")]
    pub attention_head_dim: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    /// Width of the text encoder's states.
    #[serde(default = "default_joint_attention_dim")]
    pub joint_attention_dim: usize,
    /// Width of the pooled text vector.
    #[serde(default = "default_pooled_projection_dim")]
    pub pooled_projection_dim: usize,
    /// Whether the model is guidance-distilled (`dev`: yes, `schnell`: no) and
    /// so takes a guidance scale as an input.
    #[serde(default)]
    pub guidance_embeds: bool,
    /// How many of a head's channels each of the three position ids rotates.
    #[serde(default = "default_axes_dims_rope")]
    pub axes_dims_rope: Vec<usize>,
}

fn default_in_channels() -> usize {
    64
}
fn default_num_layers() -> usize {
    19
}
fn default_num_single_layers() -> usize {
    38
}
fn default_attention_head_dim() -> usize {
    128
}
fn default_num_attention_heads() -> usize {
    24
}
fn default_joint_attention_dim() -> usize {
    4096
}
fn default_pooled_projection_dim() -> usize {
    768
}
fn default_axes_dims_rope() -> Vec<usize> {
    vec![16, 56, 56]
}

impl Default for FluxConfig {
    /// `diffusers`' `FluxTransformer2DModel()` defaults: FLUX.1 without the
    /// guidance embedder.
    fn default() -> Self {
        Self {
            in_channels: default_in_channels(),
            out_channels: None,
            num_layers: default_num_layers(),
            num_single_layers: default_num_single_layers(),
            attention_head_dim: default_attention_head_dim(),
            num_attention_heads: default_num_attention_heads(),
            joint_attention_dim: default_joint_attention_dim(),
            pooled_projection_dim: default_pooled_projection_dim(),
            guidance_embeds: false,
            axes_dims_rope: default_axes_dims_rope(),
        }
    }
}

impl FluxConfig {
    /// The hidden size, `num_attention_heads * attention_head_dim`.
    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    fn validate(&self) -> Result<(), Exception> {
        let rotated: usize = self.axes_dims_rope.iter().sum();
        if rotated != self.attention_head_dim {
            return Err(Exception::custom(format!(
                "flux: axes_dims_rope {:?} rotates {rotated} channels, but a head has {}",
                self.axes_dims_rope, self.attention_head_dim
            )));
        }
        if self.axes_dims_rope.iter().any(|d| d % 2 != 0) {
            return Err(Exception::custom(format!(
                "flux: axes_dims_rope {:?} has an odd entry; each rotates channels in pairs",
                self.axes_dims_rope
            )));
        }
        Ok(())
    }
}

fn linear(inputs: usize, outputs: usize) -> Result<nn::Linear, Exception> {
    nn::LinearBuilder::new(inputs as i32, outputs as i32).build()
}

fn rms_norm(dim: usize) -> Result<nn::RmsNorm, Exception> {
    nn::RmsNormBuilder::new(dim as i32).eps(NORM_EPS).build()
}

fn layer_norm(dim: usize) -> Result<nn::LayerNorm, Exception> {
    nn::LayerNormBuilder::new(dim as i32)
        .affine(false)
        .eps(NORM_EPS)
        .build()
}

/// `x * (1 + scale) + shift`, with `scale` and `shift` `[batch, dim]`
/// broadcast over the sequence.
fn modulate(x: &Array, scale: &Array, shift: &Array) -> Array {
    let scale = scale.add_scalar(1.0).expand_dims_axes(&[1]);
    x.multiply(&scale).add(&shift.expand_dims_axes(&[1]))
}

/// Adaptive LayerNorm of a double-stream block: six modulation vectors from the
/// conditioning (shift, scale and gate for the attention, then the MLP).
#[derive(Debug)]
pub struct AdaLayerNormZero {
    pub linear: nn::Linear,
    norm: nn::LayerNorm,
}
impl_module_params!(AdaLayerNormZero; linear);

/// What [`AdaLayerNormZero::forward`] hands the block.
pub struct ZeroModulation {
    /// The modulated, normalised input of the attention.
    pub x: Array,
    pub gate_msa: Array,
    pub shift_mlp: Array,
    pub scale_mlp: Array,
    pub gate_mlp: Array,
}

impl AdaLayerNormZero {
    pub fn new(dim: usize) -> Result<Self, Exception> {
        Ok(Self {
            linear: linear(dim, 6 * dim)?,
            norm: layer_norm(dim)?,
        })
    }

    pub fn forward(&mut self, x: &Array, temb: &Array) -> Result<ZeroModulation, Exception> {
        let emb = self.linear.forward(&nn::silu(temb));
        let [
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ] = split_chunks::<6>(&emb)?;
        Ok(ZeroModulation {
            x: modulate(&self.norm.forward(x), &scale_msa, &shift_msa),
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        })
    }
}

/// Adaptive LayerNorm of a single-stream block: shift, scale and gate.
#[derive(Debug)]
pub struct AdaLayerNormZeroSingle {
    pub linear: nn::Linear,
    norm: nn::LayerNorm,
}
impl_module_params!(AdaLayerNormZeroSingle; linear);

impl AdaLayerNormZeroSingle {
    pub fn new(dim: usize) -> Result<Self, Exception> {
        Ok(Self {
            linear: linear(dim, 3 * dim)?,
            norm: layer_norm(dim)?,
        })
    }

    /// The modulated input and the gate.
    pub fn forward(&mut self, x: &Array, temb: &Array) -> Result<(Array, Array), Exception> {
        let emb = self.linear.forward(&nn::silu(temb));
        let [shift, scale, gate] = split_chunks::<3>(&emb)?;
        Ok((modulate(&self.norm.forward(x), &scale, &shift), gate))
    }
}

/// Adaptive LayerNorm of the output: scale, then shift (the opposite order to
/// the blocks').
#[derive(Debug)]
pub struct AdaLayerNormContinuous {
    pub linear: nn::Linear,
    norm: nn::LayerNorm,
}
impl_module_params!(AdaLayerNormContinuous; linear);

impl AdaLayerNormContinuous {
    pub fn new(dim: usize) -> Result<Self, Exception> {
        Ok(Self {
            linear: linear(dim, 2 * dim)?,
            norm: layer_norm(dim)?,
        })
    }

    pub fn forward(&mut self, x: &Array, temb: &Array) -> Result<Array, Exception> {
        let emb = self.linear.forward(&nn::silu(temb));
        let [scale, shift] = split_chunks::<2>(&emb)?;
        Ok(modulate(&self.norm.forward(x), &scale, &shift))
    }
}

/// Split `[batch, N * dim]` into its `N` equal chunks along the last axis.
fn split_chunks<const N: usize>(x: &Array) -> Result<[Array; N], Exception> {
    let chunks = ops::split(x, N as i32, -1);
    chunks.try_into().map_err(|chunks: Vec<Array>| {
        Exception::custom(format!(
            "flux: split into {} chunks, wanted {N}",
            chunks.len()
        ))
    })
}

/// Two linear layers with a SiLU between, the timestep, guidance and pooled
/// text embedders alike.
#[derive(Debug)]
pub struct TimestepEmbeddings {
    pub linear_1: nn::Linear,
    pub linear_2: nn::Linear,
}
impl_module_params!(TimestepEmbeddings; linear_1, linear_2);

impl TimestepEmbeddings {
    pub fn new(inputs: usize, outputs: usize) -> Result<Self, Exception> {
        Ok(Self {
            linear_1: linear(inputs, outputs)?,
            linear_2: linear(outputs, outputs)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        Ok(self.linear_2.forward(&nn::silu(&self.linear_1.forward(x))))
    }
}

/// The conditioning vector: timestep, optional guidance and pooled text, each
/// embedded and summed.
#[derive(Debug)]
pub struct CombinedTimestepTextProjEmbeddings {
    pub timestep_embedder: TimestepEmbeddings,
    pub guidance_embedder: Option<TimestepEmbeddings>,
    pub text_embedder: TimestepEmbeddings,
}
impl_module_params!(CombinedTimestepTextProjEmbeddings; timestep_embedder, guidance_embedder, text_embedder);

/// `[batch, dim]` sinusoidal embedding of `timesteps`, cosines first, as
/// `diffusers`' `Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)`.
pub fn get_timestep_embedding(
    timesteps: &Array,
    embedding_dim: usize,
    max_period: f32,
) -> Result<Array, Exception> {
    let half = embedding_dim / 2;
    let frequencies: Vec<f32> = (0..half)
        .map(|i| (-(f64::from(max_period).ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    let frequencies = Array::from_slice(&frequencies, &[1, half as i32]);
    let angles = timesteps
        .as_dtype(Dtype::Float32.as_i32())
        .expand_dims_axes(&[1])
        .multiply(&frequencies);
    Ok(ops::concatenate_axis(&[&angles.cos(), &angles.sin()], -1))
}

impl CombinedTimestepTextProjEmbeddings {
    pub fn new(config: &FluxConfig) -> Result<Self, Exception> {
        let dim = config.hidden_size();
        Ok(Self {
            timestep_embedder: TimestepEmbeddings::new(TIMESTEP_DIM, dim)?,
            guidance_embedder: config
                .guidance_embeds
                .then(|| TimestepEmbeddings::new(TIMESTEP_DIM, dim))
                .transpose()?,
            text_embedder: TimestepEmbeddings::new(config.pooled_projection_dim, dim)?,
        })
    }

    /// `timestep` and `guidance` are `[batch]`, already scaled by 1000.
    pub fn forward(
        &mut self,
        timestep: &Array,
        guidance: Option<&Array>,
        pooled: &Array,
        dtype: Dtype,
    ) -> Result<Array, Exception> {
        let embed = |t: &Array| -> Result<Array, Exception> {
            Ok(get_timestep_embedding(t, TIMESTEP_DIM, 10_000.0)?.as_dtype(dtype.as_i32()))
        };
        let mut temb = self.timestep_embedder.forward(&embed(timestep)?)?;
        match (self.guidance_embedder.as_mut(), guidance) {
            (Some(embedder), Some(guidance)) => {
                temb = temb.add(&embedder.forward(&embed(guidance)?)?);
            }
            (Some(_), None) => {
                return Err(Exception::custom(
                    "flux: this model is guidance-distilled and needs a guidance scale",
                ));
            }
            (None, Some(_)) => {
                return Err(Exception::custom(
                    "flux: this model has no guidance embedder (`guidance_embeds` is false); \
                     pass no guidance",
                ));
            }
            (None, None) => {}
        }
        Ok(temb.add(
            &self
                .text_embedder
                .forward(&pooled.as_dtype(dtype.as_i32()))?,
        ))
    }
}

/// The rotary tables of a token sequence: `cos` and `sin`, `[tokens,
/// head_dim / 2]`, one angle per channel pair.
#[derive(Debug, Clone)]
pub struct RotaryTables {
    pub cos: Array,
    pub sin: Array,
}

/// Three-axis rotary embedding.
#[derive(Debug, Clone)]
pub struct FluxRoPE {
    pub theta: f64,
    pub axes_dim: Vec<usize>,
}

impl FluxRoPE {
    pub fn new(axes_dim: Vec<usize>) -> Self {
        Self {
            theta: ROPE_THETA,
            axes_dim,
        }
    }

    /// Tables for `ids` (`[tokens, axes]`, integer or float): axis `i`'s id
    /// rotates the next `axes_dim[i]` channels, pair `k` of them by
    /// `id / theta^(2k / axes_dim[i])` radians.
    pub fn forward(&self, ids: &Array) -> Result<RotaryTables, Exception> {
        if ids.ndim() != 2 || ids.dim(1) as usize != self.axes_dim.len() {
            return Err(Exception::custom(format!(
                "flux: position ids must be [tokens, {}], got {:?}",
                self.axes_dim.len(),
                ids.shape()
            )));
        }
        let ids = ids.as_dtype(Dtype::Float32.as_i32());
        let mut cos = Vec::with_capacity(self.axes_dim.len());
        let mut sin = Vec::with_capacity(self.axes_dim.len());
        for (axis, &dim) in self.axes_dim.iter().enumerate() {
            // The frequencies in f64, the angles in f32: a position of a few
            // thousand times a frequency below one is exact to 1e-4 radians
            // that way, where an f32 power would drift.
            let omega: Vec<f32> = (0..dim / 2)
                .map(|k| (1.0 / self.theta.powf(2.0 * k as f64 / dim as f64)) as f32)
                .collect();
            let omega = Array::from_slice(&omega, &[1, (dim / 2) as i32]);
            let angles = ops::select_axis(&ids, axis as i32, 1)
                .expand_dims_axes(&[1])
                .multiply(&omega);
            cos.push(angles.cos());
            sin.push(angles.sin());
        }
        Ok(RotaryTables {
            cos: ops::concatenate_axis(&cos.iter().collect::<Vec<_>>(), -1),
            sin: ops::concatenate_axis(&sin.iter().collect::<Vec<_>>(), -1),
        })
    }
}

/// Rotate the channel pairs of `x` (`[batch, heads, tokens, head_dim]`) by the
/// tables, `(a, b) -> (a cos - b sin, b cos + a sin)`, in f32.
fn apply_rope(x: &Array, tables: &RotaryTables) -> Array {
    let shape = x.shape().to_vec();
    let (cos, sin) = (
        tables.cos.reshape(&[1, 1, shape[2], -1]),
        tables.sin.reshape(&[1, 1, shape[2], -1]),
    );
    let pairs = x
        .as_dtype(Dtype::Float32.as_i32())
        .reshape(&[shape[0], shape[1], shape[2], -1, 2]);
    let even = ops::select_axis(&pairs, 0, 4);
    let odd = ops::select_axis(&pairs, 1, 4);
    let rotated_even = even.multiply(&cos).subtract(&odd.multiply(&sin));
    let rotated_odd = odd.multiply(&cos).add(&even.multiply(&sin));
    ops::concatenate_axis(
        &[
            &rotated_even.expand_dims_axes(&[-1]),
            &rotated_odd.expand_dims_axes(&[-1]),
        ],
        -1,
    )
    .reshape(&shape)
    .as_dtype(x.dtype().as_i32())
}

/// Joint attention of the double-stream blocks, and the plain self-attention of
/// the single-stream ones (which has no added projections).
#[derive(Debug)]
pub struct FluxAttention {
    pub to_q: nn::Linear,
    pub to_k: nn::Linear,
    pub to_v: nn::Linear,
    pub norm_q: nn::RmsNorm,
    pub norm_k: nn::RmsNorm,
    /// The text stream's projections (double-stream blocks only).
    pub add_q_proj: Option<nn::Linear>,
    pub add_k_proj: Option<nn::Linear>,
    pub add_v_proj: Option<nn::Linear>,
    pub norm_added_q: Option<nn::RmsNorm>,
    pub norm_added_k: Option<nn::RmsNorm>,
    /// The image stream's output projection; empty in a single-stream block,
    /// whose output projection takes the MLP branch as well.
    pub to_out: Vec<nn::Linear>,
    pub to_add_out: Option<nn::Linear>,
    pub num_heads: usize,
    pub head_dim: usize,
}
impl_module_params!(FluxAttention; to_q, to_k, to_v, norm_q, norm_k, add_q_proj, add_k_proj, add_v_proj, norm_added_q, norm_added_k, to_out, to_add_out);

impl FluxAttention {
    fn new(config: &FluxConfig, joint: bool) -> Result<Self, Exception> {
        let dim = config.hidden_size();
        let head_dim = config.attention_head_dim;
        Ok(Self {
            to_q: linear(dim, dim)?,
            to_k: linear(dim, dim)?,
            to_v: linear(dim, dim)?,
            norm_q: rms_norm(head_dim)?,
            norm_k: rms_norm(head_dim)?,
            add_q_proj: joint.then(|| linear(dim, dim)).transpose()?,
            add_k_proj: joint.then(|| linear(dim, dim)).transpose()?,
            add_v_proj: joint.then(|| linear(dim, dim)).transpose()?,
            norm_added_q: joint.then(|| rms_norm(head_dim)).transpose()?,
            norm_added_k: joint.then(|| rms_norm(head_dim)).transpose()?,
            to_out: if joint {
                vec![linear(dim, dim)?]
            } else {
                Vec::new()
            },
            to_add_out: joint.then(|| linear(dim, dim)).transpose()?,
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    /// Attend over `[text; image]` (the text stream first), rotating by
    /// `tables`, and return the attention output `[batch, text + image, dim]`
    /// before any output projection.
    fn attend(
        &mut self,
        image: &Array,
        text: Option<&Array>,
        tables: &RotaryTables,
    ) -> Result<Array, Exception> {
        let (heads, head_dim) = (self.num_heads as i32, self.head_dim as i32);
        // `[batch, tokens, dim]` to `[batch, heads, tokens, head_dim]`.
        let split = |x: Array| {
            x.reshape(&[x.dim(0), x.dim(1), heads, head_dim])
                .transpose_axes(&[0, 2, 1, 3])
        };
        let mut q = self.norm_q.forward(&split(self.to_q.forward(image)));
        let mut k = self.norm_k.forward(&split(self.to_k.forward(image)));
        let mut v = split(self.to_v.forward(image));
        if let Some(text) = text {
            let missing = || Exception::custom("flux: a single-stream attention got a text stream");
            let (Some(add_q), Some(add_k), Some(add_v), Some(norm_q), Some(norm_k)) = (
                self.add_q_proj.as_mut(),
                self.add_k_proj.as_mut(),
                self.add_v_proj.as_mut(),
                self.norm_added_q.as_ref(),
                self.norm_added_k.as_ref(),
            ) else {
                return Err(missing());
            };
            let tq = norm_q.forward(&split(add_q.forward(text)));
            let tk = norm_k.forward(&split(add_k.forward(text)));
            let tv = split(add_v.forward(text));
            q = ops::concatenate_axis(&[&tq, &q], 2);
            k = ops::concatenate_axis(&[&tk, &k], 2);
            v = ops::concatenate_axis(&[&tv, &v], 2);
        }
        let (q, k) = (apply_rope(&q, tables), apply_rope(&k, tables));

        let config = FusedAttentionConfig::new(
            self.num_heads as i32,
            self.num_heads as i32,
            self.head_dim as i32,
        )
        .with_mask_type(AttentionMaskType::None);
        let out = fused_sdpa(&q, &k, &v, &config, None)?;
        Ok(out
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[out.dim(0), out.dim(2), -1]))
    }
}

/// FLUX's MLP: `Linear -> GELU(tanh) -> Linear`, `diffusers`' `FeedForward`
/// with `activation_fn="gelu-approximate"`.
#[derive(Debug)]
pub struct FluxFeedForward {
    /// `diffusers`: `net.0.proj`.
    pub proj_in: nn::Linear,
    /// `diffusers`: `net.2`.
    pub proj_out: nn::Linear,
}
impl_module_params!(FluxFeedForward; proj_in, proj_out);

impl FluxFeedForward {
    fn new(dim: usize) -> Result<Self, Exception> {
        Ok(Self {
            proj_in: linear(dim, MLP_RATIO * dim)?,
            proj_out: linear(MLP_RATIO * dim, dim)?,
        })
    }

    fn forward(&mut self, x: &Array) -> Array {
        let hidden = crate::architectures::utils::activate(
            nn::gelu_tanh_approximate,
            &self.proj_in.forward(x),
        );
        self.proj_out.forward(&hidden)
    }
}

/// A double-stream block: modulated joint attention, then a modulated MLP per
/// stream, each added to its stream.
#[derive(Debug)]
pub struct FluxTransformerBlock {
    pub norm1: AdaLayerNormZero,
    pub norm1_context: AdaLayerNormZero,
    pub attn: FluxAttention,
    pub ff: FluxFeedForward,
    pub ff_context: FluxFeedForward,
    norm2: nn::LayerNorm,
    norm2_context: nn::LayerNorm,
}
impl_module_params!(FluxTransformerBlock; norm1, norm1_context, attn, ff, ff_context);

impl FluxTransformerBlock {
    pub fn new(config: &FluxConfig) -> Result<Self, Exception> {
        let dim = config.hidden_size();
        Ok(Self {
            norm1: AdaLayerNormZero::new(dim)?,
            norm1_context: AdaLayerNormZero::new(dim)?,
            attn: FluxAttention::new(config, true)?,
            ff: FluxFeedForward::new(dim)?,
            ff_context: FluxFeedForward::new(dim)?,
            norm2: layer_norm(dim)?,
            norm2_context: layer_norm(dim)?,
        })
    }

    /// Returns the updated `(image, text)` streams.
    pub fn forward(
        &mut self,
        image: &Array,
        text: &Array,
        temb: &Array,
        tables: &RotaryTables,
    ) -> Result<(Array, Array), Exception> {
        let a = self.norm1.forward(image, temb)?;
        let t = self.norm1_context.forward(text, temb)?;

        let attended = self.attn.attend(&a.x, Some(&t.x), tables)?;
        let text_len = text.dim(1);
        let text_attended = ops::slice_axis(&attended, 1, 0, text_len);
        let image_attended = ops::slice_axis_from(&attended, 1, text_len);
        let missing =
            || Exception::custom("flux: a double-stream block lost its output projections");
        let image_attended = self
            .attn
            .to_out
            .first_mut()
            .ok_or_else(missing)?
            .forward(&image_attended);
        let text_attended = self
            .attn
            .to_add_out
            .as_mut()
            .ok_or_else(missing)?
            .forward(&text_attended);

        let gate = |g: &Array| g.expand_dims_axes(&[1]);
        let image = image.add(&image_attended.multiply(&gate(&a.gate_msa)));
        let mlp_in = modulate(&self.norm2.forward(&image), &a.scale_mlp, &a.shift_mlp);
        let image = image.add(&self.ff.forward(&mlp_in).multiply(&gate(&a.gate_mlp)));

        let text = text.add(&text_attended.multiply(&gate(&t.gate_msa)));
        let mlp_in = modulate(
            &self.norm2_context.forward(&text),
            &t.scale_mlp,
            &t.shift_mlp,
        );
        let text = text.add(
            &self
                .ff_context
                .forward(&mlp_in)
                .multiply(&gate(&t.gate_mlp)),
        );
        Ok((image, text))
    }
}

/// A single-stream block over the concatenated sequence: attention and MLP in
/// parallel from one modulated input, joined by one output projection.
#[derive(Debug)]
pub struct FluxSingleTransformerBlock {
    pub norm: AdaLayerNormZeroSingle,
    pub proj_mlp: nn::Linear,
    pub proj_out: nn::Linear,
    pub attn: FluxAttention,
}
impl_module_params!(FluxSingleTransformerBlock; norm, proj_mlp, proj_out, attn);

impl FluxSingleTransformerBlock {
    pub fn new(config: &FluxConfig) -> Result<Self, Exception> {
        let dim = config.hidden_size();
        Ok(Self {
            norm: AdaLayerNormZeroSingle::new(dim)?,
            proj_mlp: linear(dim, MLP_RATIO * dim)?,
            proj_out: linear(dim + MLP_RATIO * dim, dim)?,
            attn: FluxAttention::new(config, false)?,
        })
    }

    /// `x` is `[batch, text + image, dim]`.
    pub fn forward(
        &mut self,
        x: &Array,
        temb: &Array,
        tables: &RotaryTables,
    ) -> Result<Array, Exception> {
        let (normed, gate) = self.norm.forward(x, temb)?;
        let mlp = crate::architectures::utils::activate(
            nn::gelu_tanh_approximate,
            &self.proj_mlp.forward(&normed),
        );
        let attended = self.attn.attend(&normed, None, tables)?;
        let joined = ops::concatenate_axis(&[&attended, &mlp], 2);
        let out = self.proj_out.forward(&joined);
        Ok(x.add(&out.multiply(&gate.expand_dims_axes(&[1]))))
    }
}

/// The FLUX.1 transformer.
#[derive(Debug)]
pub struct FluxDiT {
    pub x_embedder: nn::Linear,
    pub context_embedder: nn::Linear,
    pub time_text_embed: CombinedTimestepTextProjEmbeddings,
    pub transformer_blocks: Vec<FluxTransformerBlock>,
    pub single_transformer_blocks: Vec<FluxSingleTransformerBlock>,
    pub norm_out: AdaLayerNormContinuous,
    pub proj_out: nn::Linear,
    pos_embed: FluxRoPE,
    /// Model configuration (not a parameter).
    pub config: FluxConfig,
}
impl_module_params!(FluxDiT; x_embedder, context_embedder, time_text_embed, transformer_blocks, single_transformer_blocks, norm_out, proj_out);

impl FluxDiT {
    pub fn new(config: FluxConfig) -> Result<Self, Exception> {
        config.validate()?;
        let dim = config.hidden_size();
        let out_channels = config.out_channels.unwrap_or(config.in_channels);
        Ok(Self {
            x_embedder: linear(config.in_channels, dim)?,
            context_embedder: linear(config.joint_attention_dim, dim)?,
            time_text_embed: CombinedTimestepTextProjEmbeddings::new(&config)?,
            transformer_blocks: (0..config.num_layers)
                .map(|_| FluxTransformerBlock::new(&config))
                .collect::<Result<_, _>>()?,
            single_transformer_blocks: (0..config.num_single_layers)
                .map(|_| FluxSingleTransformerBlock::new(&config))
                .collect::<Result<_, _>>()?,
            norm_out: AdaLayerNormContinuous::new(dim)?,
            proj_out: linear(dim, out_channels)?,
            pos_embed: FluxRoPE::new(config.axes_dims_rope.clone()),
            config,
        })
    }

    pub fn config(&self) -> &FluxConfig {
        &self.config
    }

    /// The rotary tables for `ids` (`[tokens, 3]`): the text tokens' ids, then
    /// the image tokens'.
    pub fn rope_tables(&self, ids: &Array) -> Result<RotaryTables, Exception> {
        self.pos_embed.forward(ids)
    }

    /// Predict the velocity of the packed latents `hidden_states`.
    ///
    /// * `hidden_states` - `[batch, image tokens, in_channels]`.
    /// * `timestep` - `[batch]`, the noise level in `[0, 1]` (the scheduler's
    ///   sigma; the model scales it by 1000 itself).
    /// * `encoder_hidden_states` - `[batch, text tokens, joint_attention_dim]`.
    /// * `pooled_projections` - `[batch, pooled_projection_dim]`.
    /// * `guidance` - `[batch]`, the distilled guidance scale (unscaled, as
    ///   `3.5`); required by a model with `guidance_embeds` and refused by one
    ///   without.
    /// * `txt_ids`, `img_ids` - `[tokens, 3]` position ids (text: zeros;
    ///   image: `(0, row, column)`), shared by the batch.
    ///
    /// Returns `[batch, image tokens, out_channels]`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the reference transformer's forward inputs"
    )]
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        timestep: &Array,
        encoder_hidden_states: &Array,
        pooled_projections: &Array,
        guidance: Option<&Array>,
        txt_ids: &Array,
        img_ids: &Array,
    ) -> Result<Array, Exception> {
        let mut image = self.x_embedder.forward(hidden_states);
        let dtype = image.dtype();
        // The sinusoids are formed in f32 from the f32 noise level; rounding it
        // to a bf16 model's dtype first would quantise a sigma of 0.5 to the
        // nearest of 256 steps before the model ever saw it.
        let scaled = |t: &Array| t.as_dtype(Dtype::Float32.as_i32()).mul_scalar(1000.0);
        let timestep = scaled(timestep);
        let guidance = guidance.map(scaled);
        let temb = self.time_text_embed.forward(
            &timestep,
            guidance.as_ref(),
            pooled_projections,
            dtype,
        )?;
        let mut text = self.context_embedder.forward(encoder_hidden_states);

        let tables = self.rope_tables(&ops::concatenate_axis(&[txt_ids, img_ids], 0))?;

        for block in &mut self.transformer_blocks {
            (image, text) = block.forward(&image, &text, &temb, &tables)?;
        }
        let text_len = text.dim(1);
        let mut joint = ops::concatenate_axis(&[&text, &image], 1);
        for block in &mut self.single_transformer_blocks {
            joint = block.forward(&joint, &temb, &tables)?;
        }
        let image = ops::slice_axis_from(&joint, 1, text_len);
        let image = self.norm_out.forward(&image, &temb)?;
        Ok(self.proj_out.forward(&image))
    }
}
