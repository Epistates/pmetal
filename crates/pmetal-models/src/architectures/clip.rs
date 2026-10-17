//! CLIP text encoder.
//!
//! `CLIPTextModel` (and `CLIPTextModelWithProjection`) from `transformers`,
//! after Radford et al., 2021: a pre-norm transformer with **causal**
//! self-attention over a 77-token context, whose sentence embedding is the
//! final-norm state at the end-of-text token. FLUX conditions on that pooled
//! vector (the `CLIP-L` text tower), and Stable Diffusion on the penultimate
//! layer's states, so both are returned.

use pmetal_runtime::compat::ops::{arange, argmax_axis, broadcast_to, take_along_axis};
use pmetal_runtime::compat::{Array, Dtype, Exception, nn};
use pmetal_runtime::impl_module_params;
use pmetal_runtime::nn::{AttentionMaskType, FusedAttentionConfig, fused_sdpa};
use serde::{Deserialize, Serialize};

use crate::architectures::utils::{Activation, resolve_activation};

/// CLIP text encoder configuration, in the field names of the text half of a
/// `transformers` CLIP `config.json` (`hidden_size`, `num_hidden_layers`,
/// `num_attention_heads` also read as `embed_dim`, `num_layers`, `num_heads`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CLIPConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_embed_dim", alias = "hidden_size")]
    pub embed_dim: usize,
    #[serde(default = "default_num_layers", alias = "num_hidden_layers")]
    pub num_layers: usize,
    #[serde(default = "default_num_heads", alias = "num_attention_heads")]
    pub num_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// `hidden_act`, verbatim from the reference config.
    ///
    /// This was a `use_quick_gelu: bool`, which cannot distinguish the three
    /// GELUs a CLIP checkpoint can name: OpenAI's CLIP is `"quick_gelu"`, but
    /// LAION re-trains ship `"gelu"` (exact erf) and some variants
    /// `"gelu_new"` (tanh). Both non-quick spellings collapsed to the same
    /// wrong function under the boolean.
    #[serde(default = "default_hidden_act", alias = "activation_function")]
    pub hidden_act: String,
    /// The id of the end-of-text token, whose position the sentence embedding
    /// is read from. `2` is the value of configs written before the field was
    /// meaningful; for those the reference takes the position of the largest id
    /// instead (end-of-text is the last id of the vocabulary), which is also
    /// what [`CLIPTextModel::forward`] does for `2`.
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: usize,
    /// Width of the projected sentence embedding of a
    /// `CLIPTextModelWithProjection` ([`CLIPTextModel::with_projection`]). A
    /// plain `CLIPTextModel` has no projection whatever its config says (FLUX's
    /// text encoder names `768` and pools without it).
    #[serde(default = "default_projection_dim")]
    pub projection_dim: usize,
}

fn default_vocab_size() -> usize {
    49408
}
fn default_embed_dim() -> usize {
    512
}
fn default_num_layers() -> usize {
    12
}
fn default_num_heads() -> usize {
    8
}
fn default_intermediate_size() -> usize {
    2048
}
fn default_max_position_embeddings() -> usize {
    77
}
fn default_layer_norm_eps() -> f32 {
    1e-5
}
fn default_hidden_act() -> String {
    "quick_gelu".to_string()
}
fn default_eos_token_id() -> usize {
    2
}
fn default_projection_dim() -> usize {
    512
}

impl Default for CLIPConfig {
    /// `transformers`' `CLIPTextConfig()`.
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            embed_dim: default_embed_dim(),
            num_layers: default_num_layers(),
            num_heads: default_num_heads(),
            intermediate_size: default_intermediate_size(),
            max_position_embeddings: default_max_position_embeddings(),
            layer_norm_eps: default_layer_norm_eps(),
            hidden_act: default_hidden_act(),
            eos_token_id: default_eos_token_id(),
            projection_dim: default_projection_dim(),
        }
    }
}

impl CLIPConfig {
    /// The text tower of `openai/clip-vit-large-patch14`, FLUX's `CLIP-L`.
    pub fn clip_vit_l_14() -> Self {
        Self {
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            ..Self::default()
        }
    }
}

/// CLIP causal self-attention.
#[derive(Debug)]
pub struct CLIPAttention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub out_proj: nn::Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}
impl_module_params!(CLIPAttention; q_proj, k_proj, v_proj, out_proj);

impl CLIPAttention {
    pub fn new(config: &CLIPConfig) -> Result<Self, Exception> {
        if config.embed_dim % config.num_heads != 0 {
            return Err(Exception::custom(format!(
                "clip: embed_dim {} is not a multiple of num_heads {}",
                config.embed_dim, config.num_heads
            )));
        }
        let dim = config.embed_dim as i32;
        let linear = || nn::LinearBuilder::new(dim, dim).build();
        Ok(Self {
            q_proj: linear()?,
            k_proj: linear()?,
            v_proj: linear()?,
            out_proj: linear()?,
            num_heads: config.num_heads,
            head_dim: config.embed_dim / config.num_heads,
        })
    }

    /// Causal self-attention over `x` (`[batch, length, embed_dim]`).
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let (b, l) = (x.dim(0), x.dim(1));
        let split = |t: Array| {
            t.reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])
                .transpose_axes(&[0, 2, 1, 3])
        };
        let q = split(self.q_proj.forward(x));
        let k = split(self.k_proj.forward(x));
        let v = split(self.v_proj.forward(x));

        let config = FusedAttentionConfig::new(
            self.num_heads as i32,
            self.num_heads as i32,
            self.head_dim as i32,
        )
        .with_mask_type(AttentionMaskType::Causal);
        let out = fused_sdpa(&q, &k, &v, &config, None)?;
        let out = out.transpose_axes(&[0, 2, 1, 3]).reshape(&[b, l, -1]);
        Ok(self.out_proj.forward(&out))
    }
}

/// CLIP MLP layer.
#[derive(Debug)]
pub struct CLIPMLP {
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    /// `ACT2FN[config.hidden_act]`, resolved once at construction.
    pub act: Activation,
}
impl_module_params!(CLIPMLP; fc1, fc2);

impl CLIPMLP {
    pub fn new(config: &CLIPConfig) -> Result<Self, Exception> {
        let act = resolve_activation(&config.hidden_act).ok_or_else(|| {
            Exception::custom(format!(
                "clip: unsupported hidden_act {:?}",
                config.hidden_act
            ))
        })?;
        Ok(Self {
            fc1: nn::LinearBuilder::new(config.embed_dim as i32, config.intermediate_size as i32)
                .build()?,
            fc2: nn::LinearBuilder::new(config.intermediate_size as i32, config.embed_dim as i32)
                .build()?,
            act,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let x = (self.act)(&self.fc1.forward(x));
        Ok(self.fc2.forward(&x))
    }
}

/// CLIP Encoder Layer.
#[derive(Debug)]
pub struct CLIPEncoderLayer {
    pub attn: CLIPAttention,
    pub mlp: CLIPMLP,
    pub norm1: nn::LayerNorm,
    pub norm2: nn::LayerNorm,
}
impl_module_params!(CLIPEncoderLayer; attn, mlp, norm1, norm2);

impl CLIPEncoderLayer {
    pub fn new(config: &CLIPConfig) -> Result<Self, Exception> {
        let norm = || {
            nn::LayerNormBuilder::new(config.embed_dim as i32)
                .eps(config.layer_norm_eps)
                .build()
        };
        Ok(Self {
            attn: CLIPAttention::new(config)?,
            mlp: CLIPMLP::new(config)?,
            norm1: norm()?,
            norm2: norm()?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let x = x.add(&self.attn.forward(&self.norm1.forward(x))?);
        let mlp = self.mlp.forward(&self.norm2.forward(&x))?;
        Ok(x.add(&mlp))
    }
}

/// What a CLIP text forward returns.
#[derive(Debug, Clone)]
pub struct CLIPTextOutput {
    /// The sentence embedding, `[batch, embed_dim]`: the final-norm state at the
    /// end-of-text token (through `text_projection`, if the model has one).
    pub pooled_output: Array,
    /// The final-norm states, `[batch, length, embed_dim]`.
    pub last_hidden_state: Array,
    /// The states entering the last layer, `[batch, length, embed_dim]`
    /// (`hidden_states[-2]`; before any final norm), which Stable Diffusion
    /// conditions on.
    pub penultimate_hidden_state: Array,
}

/// CLIP Text Model.
#[derive(Debug)]
pub struct CLIPTextModel {
    pub token_embedding: nn::Embedding,
    pub position_embedding: nn::Embedding,
    pub layers: Vec<CLIPEncoderLayer>,
    pub final_layer_norm: nn::LayerNorm,
    /// Present for `CLIPTextModelWithProjection` only.
    pub text_projection: Option<nn::Linear>,
    /// Model configuration (not a parameter).
    pub config: CLIPConfig,
}
impl_module_params!(CLIPTextModel; token_embedding, position_embedding, layers, final_layer_norm, text_projection);

impl CLIPTextModel {
    /// A `CLIPTextModel`: the pooled output is the final-norm state itself.
    pub fn new(config: CLIPConfig) -> Result<Self, Exception> {
        Self::build(config, false)
    }

    /// A `CLIPTextModelWithProjection`: the pooled output goes through a
    /// bias-free `text_projection` to `config.projection_dim`.
    pub fn with_projection(config: CLIPConfig) -> Result<Self, Exception> {
        Self::build(config, true)
    }

    fn build(config: CLIPConfig, projected: bool) -> Result<Self, Exception> {
        let layers = (0..config.num_layers)
            .map(|_| CLIPEncoderLayer::new(&config))
            .collect::<Result<Vec<_>, _>>()?;
        let text_projection = projected
            .then(|| {
                nn::LinearBuilder::new(config.embed_dim as i32, config.projection_dim as i32)
                    .bias(false)
                    .build()
            })
            .transpose()?;
        Ok(Self {
            token_embedding: nn::Embedding::new(config.vocab_size as i32, config.embed_dim as i32)?,
            position_embedding: nn::Embedding::new(
                config.max_position_embeddings as i32,
                config.embed_dim as i32,
            )?,
            layers,
            final_layer_norm: nn::LayerNormBuilder::new(config.embed_dim as i32)
                .eps(config.layer_norm_eps)
                .build()?,
            text_projection,
            config,
        })
    }

    pub fn config(&self) -> &CLIPConfig {
        &self.config
    }

    /// The position of each row's end-of-text token, `[batch]`.
    ///
    /// A config with `eos_token_id == 2` (the legacy default) takes the
    /// position of the largest id, the first one if it repeats, which is
    /// end-of-text because it ends the vocabulary: a prompt padded with that
    /// token reads its first occurrence. Any other id is looked for directly.
    fn eos_positions(&self, input_ids: &Array) -> Array {
        if self.config.eos_token_id == 2 {
            argmax_axis(input_ids, -1)
        } else {
            let is_eos = input_ids
                .equal(&Array::from_int(self.config.eos_token_id as i32))
                .as_dtype(Dtype::Int32.as_i32());
            argmax_axis(&is_eos, -1)
        }
    }

    /// Encode `input_ids` (`[batch, length]`, at most `max_position_embeddings`
    /// long).
    pub fn forward(&mut self, input_ids: &Array) -> Result<CLIPTextOutput, Exception> {
        let l = input_ids.dim(1);
        if l as usize > self.config.max_position_embeddings {
            return Err(Exception::custom(format!(
                "clip: {l} tokens exceed the {} positions the model has",
                self.config.max_position_embeddings
            )));
        }
        let positions = arange(l, Dtype::Int32);
        let mut x = self
            .token_embedding
            .forward(input_ids)
            .add(&self.position_embedding.forward(&positions));

        let mut penultimate = x.clone();
        let last = self.layers.len().saturating_sub(1);
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i == last {
                penultimate = x.clone();
            }
            x = layer.forward(&x)?;
        }
        let last_hidden_state = self.final_layer_norm.forward(&x);

        let b = last_hidden_state.dim(0);
        let eos = self.eos_positions(input_ids).reshape(&[b, 1, 1]);
        let eos = broadcast_to(&eos, &[b, 1, last_hidden_state.dim(2)]);
        let mut pooled_output = take_along_axis(&last_hidden_state, &eos, 1).squeeze(1);
        if let Some(projection) = self.text_projection.as_mut() {
            pooled_output = projection.forward(&pooled_output);
        }

        Ok(CLIPTextOutput {
            pooled_output,
            last_hidden_state,
            penultimate_hidden_state: penultimate,
        })
    }
}
