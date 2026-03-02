//! Flux model architecture.
//!
//! Implementation of Flux.1 DiT (Diffusion Transformer) optimized for Apple Silicon.
//! Based on the architecture from Black Forest Labs and DiffSynth-Studio.

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt},
    nn,
    ops::{concatenate_axis, indexing::IndexOp},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Flux model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxConfig {
    /// Input dimension (e.g., 64 for Flux.1).
    pub input_dim: usize,
    /// Hidden dimension (e.g., 3072).
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of joint transformer blocks (e.g., 19).
    pub num_blocks: usize,
    /// Number of single transformer blocks (e.g., 38).
    pub num_single_blocks: usize,
    /// Joint RoPE theta.
    pub rope_theta: f32,
    /// RoPE axes dimensions.
    pub axes_dim: Vec<usize>,
    /// Disable guidance embedder.
    pub disable_guidance_embedder: bool,
    /// Timestep embedding dimension (default 256 for Flux.1).
    #[serde(default = "default_timestep_dim")]
    pub timestep_dim: usize,
    /// Pooled text embedding input dimension (CLIP output dim, default 768).
    #[serde(default = "default_pooled_embed_dim")]
    pub pooled_embed_dim: usize,
    /// Context embedding input dimension (T5 output dim, default 4096).
    #[serde(default = "default_context_embed_dim")]
    pub context_embed_dim: usize,
    /// Normalization epsilon for all LayerNorm/RMSNorm layers.
    #[serde(default = "default_norm_epsilon")]
    pub norm_epsilon: f32,
}

fn default_timestep_dim() -> usize {
    256
}
fn default_pooled_embed_dim() -> usize {
    768
}
fn default_context_embed_dim() -> usize {
    4096
}
fn default_norm_epsilon() -> f32 {
    1e-6
}

impl Default for FluxConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_size: 3072,
            num_attention_heads: 24,
            num_blocks: 19,
            num_single_blocks: 38,
            rope_theta: 10000.0,
            axes_dim: vec![16, 56, 56],
            disable_guidance_embedder: false,
            timestep_dim: 256,
            pooled_embed_dim: 768,
            context_embed_dim: 4096,
            norm_epsilon: 1e-6,
        }
    }
}

/// Adaptive LayerNorm for Flux.
#[derive(Debug, ModuleParameters)]
pub struct AdaLayerNorm {
    #[param]
    pub linear: nn::Linear,
    #[param]
    pub norm: nn::LayerNorm,
}

impl AdaLayerNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        let linear = nn::LinearBuilder::new(dim as i32, (dim * 6) as i32)
            .build()
            .expect("Infallible");
        let norm = nn::LayerNormBuilder::new(dim as i32)
            .affine(false)
            .eps(eps)
            .build()
            .expect("Infallible");
        Self { linear, norm }
    }

    pub fn forward(
        &mut self,
        x: &Array,
        emb: &Array,
    ) -> Result<(Array, Array, Array, Array, Array), Exception> {
        let emb = self.linear.forward(&nn::silu(emb)?)?;
        // Reshape emb to [batch, 1, 6 * dim] for broadcasting
        let emb = emb.expand_dims_axes(&[1])?;
        let chunks = mlx_rs::ops::split(&emb, 6, -1)?;

        let shift_msa = &chunks[0];
        let scale_msa = &chunks[1];
        let gate_msa = &chunks[2];
        let shift_mlp = &chunks[3];
        let scale_mlp = &chunks[4];
        let gate_mlp = &chunks[5];

        let x = self.norm.forward(x)?;
        let x = x
            .multiply(&(scale_msa.add(&Array::from_f32(1.0))?))?
            .add(shift_msa)?;

        Ok((
            x,
            gate_msa.clone(),
            shift_mlp.clone(),
            scale_mlp.clone(),
            gate_mlp.clone(),
        ))
    }
}

/// Adaptive LayerNorm for Flux Single blocks.
#[derive(Debug, ModuleParameters)]
pub struct AdaLayerNormSingle {
    #[param]
    pub linear: nn::Linear,
    #[param]
    pub norm: nn::LayerNorm,
}

impl AdaLayerNormSingle {
    pub fn new(dim: usize, eps: f32) -> Self {
        let linear = nn::LinearBuilder::new(dim as i32, (dim * 3) as i32)
            .build()
            .expect("Infallible");
        let norm = nn::LayerNormBuilder::new(dim as i32)
            .affine(false)
            .eps(eps)
            .build()
            .expect("Infallible");
        Self { linear, norm }
    }

    pub fn forward(&mut self, x: &Array, emb: &Array) -> Result<(Array, Array), Exception> {
        let emb = self.linear.forward(&nn::silu(emb)?)?;
        let emb = emb.expand_dims_axes(&[1])?;
        let chunks = mlx_rs::ops::split(&emb, 3, -1)?;

        let shift_msa = &chunks[0];
        let scale_msa = &chunks[1];
        let gate_msa = &chunks[2];

        let x = self.norm.forward(x)?;
        let x = x
            .multiply(&(scale_msa.add(&Array::from_f32(1.0))?))?
            .add(shift_msa)?;

        Ok((x, gate_msa.clone()))
    }
}

/// Adaptive LayerNorm Continuous for Flux final output.
#[derive(Debug, ModuleParameters)]
pub struct AdaLayerNormContinuous {
    #[param]
    pub linear: nn::Linear,
    #[param]
    pub norm: nn::LayerNorm,
}

impl AdaLayerNormContinuous {
    pub fn new(dim: usize, eps: f32) -> Self {
        let linear = nn::LinearBuilder::new(dim as i32, (dim * 2) as i32)
            .build()
            .expect("Infallible");
        let norm = nn::LayerNormBuilder::new(dim as i32)
            .affine(false)
            .eps(eps)
            .build()
            .expect("Infallible");
        Self { linear, norm }
    }

    pub fn forward(&mut self, x: &Array, conditioning: &Array) -> Result<Array, Exception> {
        let emb = self.linear.forward(&nn::silu(conditioning)?)?;
        let emb = emb.expand_dims_axes(&[1])?;
        let chunks = mlx_rs::ops::split(&emb, 2, -1)?;

        let shift = &chunks[0];
        let scale = &chunks[1];

        let x = self.norm.forward(x)?;
        let x = x
            .multiply(&(scale.add(&Array::from_f32(1.0))?))?
            .add(shift)?;

        Ok(x)
    }
}

/// Timestep Embeddings for Flux.
#[derive(Debug, ModuleParameters)]
pub struct TimestepEmbeddings {
    #[param]
    pub linear_1: nn::Linear,
    #[param]
    pub linear_2: nn::Linear,
}

impl TimestepEmbeddings {
    pub fn new(dim_in: usize, dim_out: usize) -> Self {
        let linear_1 = nn::LinearBuilder::new(dim_in as i32, dim_out as i32)
            .build()
            .expect("Infallible");
        let linear_2 = nn::LinearBuilder::new(dim_out as i32, dim_out as i32)
            .build()
            .expect("Infallible");
        Self { linear_1, linear_2 }
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let x = self.linear_1.forward(x)?;
        let x = nn::silu(&x)?;
        let x = self.linear_2.forward(&x)?;
        Ok(x)
    }
}

pub fn get_timestep_embedding(
    timesteps: &Array,
    embedding_dim: usize,
    max_period: f32,
) -> Result<Array, Exception> {
    let half_dim = (embedding_dim / 2) as i32;
    let exponent = mlx_rs::ops::arange::<i32, f32>(0, half_dim, None)?
        .multiply(&Array::from_f32(-(max_period.ln()) / (half_dim as f32)))?;
    let emb = mlx_rs::ops::exp(&exponent)?;

    // timesteps: [batch]
    let timesteps = timesteps.as_dtype(mlx_rs::Dtype::Float32)?;
    let emb = timesteps
        .expand_dims_axes(&[1])?
        .matmul(&emb.expand_dims_axes(&[0])?)?;

    let sin_emb = mlx_rs::ops::sin(&emb)?;
    let cos_emb = mlx_rs::ops::cos(&emb)?;

    concatenate_axis(&[&cos_emb, &sin_emb], -1)
}

/// Flux Rotary Position Embedding (3D).
#[derive(Debug, ModuleParameters)]
pub struct FluxRoPE {
    pub dim: usize,
    pub theta: f32,
    pub axes_dim: Vec<usize>,
}

impl FluxRoPE {
    pub fn new(dim: usize, theta: f32, axes_dim: Vec<usize>) -> Self {
        Self {
            dim,
            theta,
            axes_dim,
        }
    }

    fn rope(&self, pos: &Array, dim: usize, theta: f32) -> Result<Array, Exception> {
        let half_dim = (dim / 2) as i32;
        let scale = mlx_rs::ops::arange::<i32, f32>(0, half_dim, None)?
            .divide(&Array::from_f32(half_dim as f32))?;
        let omega = Array::from_f32(1.0).divide(&Array::from_f32(theta).power(&scale)?)?;

        let out = pos
            .expand_dims_axes(&[-1])?
            .matmul(&omega.expand_dims_axes(&[0])?)?;

        let cos_out = mlx_rs::ops::cos(&out)?;
        let sin_out = mlx_rs::ops::sin(&out)?;

        let freqs_cis = concatenate_axis(
            &[
                &cos_out.expand_dims_axes(&[-1])?,
                &sin_out.expand_dims_axes(&[-1])?,
            ],
            -1,
        )?;
        Ok(freqs_cis)
    }

    pub fn forward(&self, ids: &Array) -> Result<Array, Exception> {
        let n_axes = ids.dim(-1);
        let mut embs = Vec::new();
        for i in 0..n_axes {
            let axis_ids = ids.index((.., .., i as i32));
            embs.push(self.rope(&axis_ids, self.axes_dim[i as usize], self.theta)?);
        }

        let embs_refs: Vec<&Array> = embs.iter().collect();
        let emb = concatenate_axis(&embs_refs, -2)?;
        emb.expand_dims_axes(&[1])
    }
}

/// Joint Attention for Flux.
#[derive(Debug, ModuleParameters)]
pub struct FluxJointAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    #[param]
    pub a_to_qkv: nn::Linear,
    #[param]
    pub b_to_qkv: nn::Linear,
    #[param]
    pub norm_q_a: nn::RmsNorm,
    #[param]
    pub norm_k_a: nn::RmsNorm,
    #[param]
    pub norm_q_b: nn::RmsNorm,
    #[param]
    pub norm_k_b: nn::RmsNorm,
    #[param]
    pub a_to_out: nn::Linear,
    #[param]
    pub b_to_out: nn::Linear,
}

impl FluxJointAttention {
    pub fn new(dim: usize, num_heads: usize, eps: f32) -> Self {
        let head_dim = dim / num_heads;
        Self {
            num_heads,
            head_dim,
            a_to_qkv: nn::LinearBuilder::new(dim as i32, (dim * 3) as i32)
                .build()
                .expect("Infallible"),
            b_to_qkv: nn::LinearBuilder::new(dim as i32, (dim * 3) as i32)
                .build()
                .expect("Infallible"),
            norm_q_a: nn::RmsNormBuilder::new(head_dim as i32)
                .eps(eps)
                .build()
                .expect("Infallible"),
            norm_k_a: nn::RmsNormBuilder::new(head_dim as i32)
                .eps(eps)
                .build()
                .expect("Infallible"),
            norm_q_b: nn::RmsNormBuilder::new(head_dim as i32)
                .eps(eps)
                .build()
                .expect("Infallible"),
            norm_k_b: nn::RmsNormBuilder::new(head_dim as i32)
                .eps(eps)
                .build()
                .expect("Infallible"),
            a_to_out: nn::LinearBuilder::new(dim as i32, dim as i32)
                .build()
                .expect("Infallible"),
            b_to_out: nn::LinearBuilder::new(dim as i32, dim as i32)
                .build()
                .expect("Infallible"),
        }
    }

    fn apply_rope(&self, x: &Array, freqs_cis: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let x_complex = x.reshape(&[shape[0], shape[1], shape[2], -1, 2])?;

        let cos = &x_complex.index((.., .., .., .., 0));
        let sin = &x_complex.index((.., .., .., .., 1));

        let freq_cos = &freqs_cis.index((.., .., .., .., 0));
        let freq_sin = &freqs_cis.index((.., .., .., .., 1));

        let out_real = cos.multiply(freq_cos)?.subtract(&sin.multiply(freq_sin)?)?;
        let out_imag = cos.multiply(freq_sin)?.add(&sin.multiply(freq_cos)?)?;

        concatenate_axis(
            &[
                &out_real.expand_dims_axes(&[-1])?,
                &out_imag.expand_dims_axes(&[-1])?,
            ],
            -1,
        )?
        .reshape(shape)
    }

    pub fn forward(
        &mut self,
        hidden_states_a: &Array,
        hidden_states_b: &Array,
        image_rotary_emb: &Array,
    ) -> Result<(Array, Array), Exception> {
        let batch_size = hidden_states_a.dim(0);

        let qkv_a = self.a_to_qkv.forward(hidden_states_a)?;
        let qkv_a = qkv_a
            .reshape(&[
                batch_size,
                -1,
                3,
                self.num_heads as i32,
                self.head_dim as i32,
            ])?
            .transpose_axes(&[0, 3, 1, 2, 4])?;
        let q_a = &qkv_a.index((.., .., .., 0, ..));
        let k_a = &qkv_a.index((.., .., .., 1, ..));
        let v_a = &qkv_a.index((.., .., .., 2, ..));
        let q_a = self.norm_q_a.forward(q_a)?;
        let k_a = self.norm_k_a.forward(k_a)?;

        let qkv_b = self.b_to_qkv.forward(hidden_states_b)?;
        let qkv_b = qkv_b
            .reshape(&[
                batch_size,
                -1,
                3,
                self.num_heads as i32,
                self.head_dim as i32,
            ])?
            .transpose_axes(&[0, 3, 1, 2, 4])?;
        let q_b = &qkv_b.index((.., .., .., 0, ..));
        let k_b = &qkv_b.index((.., .., .., 1, ..));
        let v_b = &qkv_b.index((.., .., .., 2, ..));
        let q_b = self.norm_q_b.forward(q_b)?;
        let k_b = self.norm_k_b.forward(k_b)?;

        let q = concatenate_axis(&[q_b, q_a], 2)?;
        let k = concatenate_axis(&[k_b, k_a], 2)?;
        let v = concatenate_axis(&[v_b, v_a], 2)?;

        let q = self.apply_rope(&q, image_rotary_emb)?;
        let k = self.apply_rope(&k, image_rotary_emb)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_out = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            Option::<mlx_rs::fast::ScaledDotProductAttentionMask>::None,
            Option::<&Array>::None,
        )?;

        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch_size,
            -1,
            (self.num_heads * self.head_dim) as i32,
        ])?;

        let split_idx = hidden_states_b.dim(1);
        let hidden_states_b_out = attn_out.index((.., ..split_idx as i32, ..));
        let hidden_states_a_out = attn_out.index((.., split_idx as i32.., ..));

        let hidden_states_a_out = self.a_to_out.forward(&hidden_states_a_out)?;
        let hidden_states_b_out = self.b_to_out.forward(&hidden_states_b_out)?;

        Ok((hidden_states_a_out, hidden_states_b_out))
    }
}

/// Joint Transformer Block for Flux.
#[derive(Debug, ModuleParameters)]
pub struct FluxJointTransformerBlock {
    #[param]
    pub norm1_a: AdaLayerNorm,
    #[param]
    pub norm1_b: AdaLayerNorm,
    #[param]
    pub attn: FluxJointAttention,
    #[param]
    pub norm2_a: nn::LayerNorm,
    #[param]
    pub ff_a: Vec<nn::Linear>,
    #[param]
    pub norm2_b: nn::LayerNorm,
    #[param]
    pub ff_b: Vec<nn::Linear>,
}

impl FluxJointTransformerBlock {
    pub fn new(dim: usize, num_heads: usize, eps: f32) -> Self {
        let norm1_a = AdaLayerNorm::new(dim, eps);
        let norm1_b = AdaLayerNorm::new(dim, eps);
        let attn = FluxJointAttention::new(dim, num_heads, eps);

        let norm2_a = nn::LayerNormBuilder::new(dim as i32)
            .affine(false)
            .eps(eps)
            .build()
            .expect("Infallible");
        let ff_a = vec![
            nn::LinearBuilder::new(dim as i32, (dim * 4) as i32)
                .build()
                .expect("Infallible"),
            nn::LinearBuilder::new((dim * 4) as i32, dim as i32)
                .build()
                .expect("Infallible"),
        ];

        let norm2_b = nn::LayerNormBuilder::new(dim as i32)
            .affine(false)
            .eps(eps)
            .build()
            .expect("Infallible");
        let ff_b = vec![
            nn::LinearBuilder::new(dim as i32, (dim * 4) as i32)
                .build()
                .expect("Infallible"),
            nn::LinearBuilder::new((dim * 4) as i32, dim as i32)
                .build()
                .expect("Infallible"),
        ];

        Self {
            norm1_a,
            norm1_b,
            attn,
            norm2_a,
            ff_a,
            norm2_b,
            ff_b,
        }
    }

    pub fn forward(
        &mut self,
        hidden_states_a: &Array,
        hidden_states_b: &Array,
        temb: &Array,
        image_rotary_emb: &Array,
    ) -> Result<(Array, Array), Exception> {
        let (norm_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a) =
            self.norm1_a.forward(hidden_states_a, temb)?;
        let (norm_b, gate_msa_b, shift_mlp_b, scale_mlp_b, gate_mlp_b) =
            self.norm1_b.forward(hidden_states_b, temb)?;

        let (attn_a, attn_b) = self.attn.forward(&norm_a, &norm_b, image_rotary_emb)?;

        let hidden_states_a = hidden_states_a.add(&attn_a.multiply(&gate_msa_a)?)?;
        let norm_hidden_a = self.norm2_a.forward(&hidden_states_a)?;
        let norm_hidden_a = norm_hidden_a
            .multiply(&(scale_mlp_a.add(&Array::from_f32(1.0))?))?
            .add(&shift_mlp_a)?;

        let ff_a_0_out = self.ff_a[0].forward(&norm_hidden_a)?;
        let ff_a_out = self.ff_a[1].forward(&nn::gelu_approximate(&ff_a_0_out)?)?;
        let hidden_states_a = hidden_states_a.add(&ff_a_out.multiply(&gate_mlp_a)?)?;

        let hidden_states_b = hidden_states_b.add(&attn_b.multiply(&gate_msa_b)?)?;
        let norm_hidden_b = self.norm2_b.forward(&hidden_states_b)?;
        let norm_hidden_b = norm_hidden_b
            .multiply(&(scale_mlp_b.add(&Array::from_f32(1.0))?))?
            .add(&shift_mlp_b)?;

        let ff_b_0_out = self.ff_b[0].forward(&norm_hidden_b)?;
        let ff_b_out = self.ff_b[1].forward(&nn::gelu_approximate(&ff_b_0_out)?)?;
        let hidden_states_b = hidden_states_b.add(&ff_b_out.multiply(&gate_mlp_b)?)?;

        Ok((hidden_states_a, hidden_states_b))
    }
}

/// Single Transformer Block for Flux.
#[derive(Debug, ModuleParameters)]
pub struct FluxSingleTransformerBlock {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    #[param]
    pub norm: AdaLayerNormSingle,
    #[param]
    pub to_qkv_mlp: nn::Linear,
    #[param]
    pub norm_q_a: nn::RmsNorm,
    #[param]
    pub norm_k_a: nn::RmsNorm,
    #[param]
    pub proj_out: nn::Linear,
}

impl FluxSingleTransformerBlock {
    pub fn new(dim: usize, num_heads: usize, eps: f32) -> Self {
        let head_dim = dim / num_heads;
        Self {
            dim,
            num_heads,
            head_dim,
            norm: AdaLayerNormSingle::new(dim, eps),
            to_qkv_mlp: nn::LinearBuilder::new(dim as i32, (dim * 7) as i32)
                .build()
                .expect("Infallible"),
            norm_q_a: nn::RmsNormBuilder::new(head_dim as i32)
                .eps(eps)
                .build()
                .expect("Infallible"),
            norm_k_a: nn::RmsNormBuilder::new(head_dim as i32)
                .eps(eps)
                .build()
                .expect("Infallible"),
            proj_out: nn::LinearBuilder::new((dim * 5) as i32, dim as i32)
                .build()
                .expect("Infallible"),
        }
    }

    fn apply_rope(&self, x: &Array, freqs_cis: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let x_complex = x.reshape(&[shape[0], shape[1], shape[2], -1, 2])?;

        let cos = &x_complex.index((.., .., .., .., 0));
        let sin = &x_complex.index((.., .., .., .., 1));

        let freq_cos = &freqs_cis.index((.., .., .., .., 0));
        let freq_sin = &freqs_cis.index((.., .., .., .., 1));

        let out_real = cos.multiply(freq_cos)?.subtract(&sin.multiply(freq_sin)?)?;
        let out_imag = cos.multiply(freq_sin)?.add(&sin.multiply(freq_cos)?)?;

        concatenate_axis(
            &[
                &out_real.expand_dims_axes(&[-1])?,
                &out_imag.expand_dims_axes(&[-1])?,
            ],
            -1,
        )?
        .reshape(shape)
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        temb: &Array,
        image_rotary_emb: &Array,
    ) -> Result<Array, Exception> {
        let residual = hidden_states;
        let (norm_hidden, gate) = self.norm.forward(hidden_states, temb)?;
        let qkv_mlp = self.to_qkv_mlp.forward(&norm_hidden)?;

        let batch_size = hidden_states.dim(0);
        let qkv = &qkv_mlp.index((.., .., ..(self.dim * 3) as i32));
        let mlp_hidden = &qkv_mlp.index((.., .., (self.dim * 3) as i32..));

        let qkv = qkv
            .reshape(&[
                batch_size,
                -1,
                3,
                self.num_heads as i32,
                self.head_dim as i32,
            ])?
            .transpose_axes(&[0, 3, 1, 2, 4])?;
        let q = &qkv.index((.., .., .., 0, ..));
        let k = &qkv.index((.., .., .., 1, ..));
        let v = &qkv.index((.., .., .., 2, ..));

        let q = self.norm_q_a.forward(q)?;
        let k = self.norm_k_a.forward(k)?;

        let q = self.apply_rope(&q, image_rotary_emb)?;
        let k = self.apply_rope(&k, image_rotary_emb)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_out = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            Option::<mlx_rs::fast::ScaledDotProductAttentionMask>::None,
            Option::<&Array>::None,
        )?;

        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch_size,
            -1,
            (self.num_heads * self.head_dim) as i32,
        ])?;
        let mlp_hidden = nn::gelu_approximate(&mlp_hidden)?;

        let combined = concatenate_axis(&[&attn_out, &mlp_hidden], -1)?;
        let out = self.proj_out.forward(&combined)?;
        let out = out.multiply(&gate)?;

        residual.add(&out)
    }
}

/// Flux DiT Model.
#[derive(Debug, ModuleParameters)]
pub struct FluxDiT {
    pub pos_embedder: FluxRoPE,
    pub timestep_dim: usize,
    #[param]
    pub time_embedder: TimestepEmbeddings,
    #[param]
    pub guidance_embedder: Option<TimestepEmbeddings>,
    #[param]
    pub pooled_text_embedder: Vec<nn::Linear>,
    #[param]
    pub context_embedder: nn::Linear,
    #[param]
    pub x_embedder: nn::Linear,
    #[param]
    pub blocks: Vec<FluxJointTransformerBlock>,
    #[param]
    pub single_blocks: Vec<FluxSingleTransformerBlock>,
    #[param]
    pub final_norm_out: AdaLayerNormContinuous,
    #[param]
    pub final_proj_out: nn::Linear,
}

impl FluxDiT {
    pub fn new(config: FluxConfig) -> Self {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let pos_embedder = FluxRoPE::new(dim, config.rope_theta, config.axes_dim);
        let time_embedder = TimestepEmbeddings::new(config.timestep_dim, dim);
        let guidance_embedder = if config.disable_guidance_embedder {
            None
        } else {
            Some(TimestepEmbeddings::new(config.timestep_dim, dim))
        };

        let pooled_text_embedder = vec![
            nn::LinearBuilder::new(config.pooled_embed_dim as i32, dim as i32)
                .build()
                .expect("Infallible"),
            nn::LinearBuilder::new(dim as i32, dim as i32)
                .build()
                .expect("Infallible"),
        ];

        let context_embedder = nn::LinearBuilder::new(config.context_embed_dim as i32, dim as i32)
            .build()
            .expect("Infallible");
        let x_embedder = nn::LinearBuilder::new(config.input_dim as i32, dim as i32)
            .build()
            .expect("Infallible");

        let eps = config.norm_epsilon;

        let blocks = (0..config.num_blocks)
            .map(|_| FluxJointTransformerBlock::new(dim, num_heads, eps))
            .collect();

        let single_blocks = (0..config.num_single_blocks)
            .map(|_| FluxSingleTransformerBlock::new(dim, num_heads, eps))
            .collect();

        let final_norm_out = AdaLayerNormContinuous::new(dim, eps);
        let final_proj_out = nn::LinearBuilder::new(dim as i32, config.input_dim as i32)
            .build()
            .expect("Infallible");

        Self {
            pos_embedder,
            timestep_dim: config.timestep_dim,
            time_embedder,
            guidance_embedder,
            pooled_text_embedder,
            context_embedder,
            x_embedder,
            blocks,
            single_blocks,
            final_norm_out,
            final_proj_out,
        }
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        timestep: &Array,
        prompt_emb: &Array,
        pooled_prompt_emb: &Array,
        guidance: Option<&Array>,
        text_ids: &Array,
        image_ids: &Array,
    ) -> Result<Array, Exception> {
        let mut temb = self.time_embedder.forward(&get_timestep_embedding(
            timestep,
            self.timestep_dim,
            10000.0,
        )?)?;
        if let (Some(g_emb), Some(g)) = (&mut self.guidance_embedder, guidance) {
            temb = temb.add(&g_emb.forward(&get_timestep_embedding(
                g,
                self.timestep_dim,
                10000.0,
            )?)?)?;
        }

        let pooled_emb_pre = self.pooled_text_embedder[0].forward(pooled_prompt_emb)?;
        let pooled_emb = self.pooled_text_embedder[1].forward(&nn::silu(&pooled_emb_pre)?)?;
        temb = temb.add(&pooled_emb)?;

        let mut hidden_states_a = self.x_embedder.forward(hidden_states)?;
        let mut hidden_states_b = self.context_embedder.forward(prompt_emb)?;

        // Rotary embeddings
        let ids = concatenate_axis(&[text_ids, image_ids], 1)?;
        let image_rotary_emb = self.pos_embedder.forward(&ids)?;

        // Joint blocks
        for block in &mut self.blocks {
            let (ha, hb) =
                block.forward(&hidden_states_a, &hidden_states_b, &temb, &image_rotary_emb)?;
            hidden_states_a = ha;
            hidden_states_b = hb;
        }

        // Single blocks
        let mut combined = concatenate_axis(&[&hidden_states_b, &hidden_states_a], 1)?;
        for block in &mut self.single_blocks {
            combined = block.forward(&combined, &temb, &image_rotary_emb)?;
        }

        // Final output
        let split_idx = hidden_states_b.dim(1);
        let hidden_states_a = combined.index((.., split_idx as i32.., ..));

        let out = self.final_norm_out.forward(&hidden_states_a, &temb)?;
        let out = self.final_proj_out.forward(&out)?;

        Ok(out)
    }
}

impl FluxDiT {
    /// Evaluate all parameters (force computation of any lazy arrays).
    pub fn eval_params(&self) -> Result<(), Exception> {
        // ModuleParametersExt::parameters() covers all #[param] fields.
        // pos_embedder has no trainable params (it's RoPE), eval manually if needed.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flux_dit_forward() {
        let config = FluxConfig {
            num_blocks: 1,
            num_single_blocks: 1,
            ..Default::default()
        };
        let mut model = FluxDiT::new(config);

        let batch = 1;
        let seq_len = 16 * 16; // 128x128 image patchified with 2x2 patches
        let hidden_states =
            mlx_rs::random::normal::<f32>(&[batch, seq_len as i32, 64], None, None, None).unwrap();
        let timestep = Array::from_slice(&[1.0f32], &[1]);
        let prompt_emb =
            mlx_rs::random::normal::<f32>(&[batch, 512, 4096], None, None, None).unwrap();
        let pooled_prompt_emb =
            mlx_rs::random::normal::<f32>(&[batch, 768], None, None, None).unwrap();
        let text_ids = mlx_rs::ops::zeros::<f32>(&[batch, 512, 3]).unwrap();
        let image_ids = mlx_rs::ops::zeros::<f32>(&[batch, seq_len as i32, 3]).unwrap();

        let out = model
            .forward(
                &hidden_states,
                &timestep,
                &prompt_emb,
                &pooled_prompt_emb,
                None,
                &text_ids,
                &image_ids,
            )
            .unwrap();

        assert_eq!(out.shape(), hidden_states.shape());
    }
}
