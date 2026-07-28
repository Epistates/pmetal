//! CLIP text encoder architecture.
//!
//! Implementation of CLIP (Contrastive Language-Image Pre-training) text encoder.
//! Based on the architecture from OpenAI and used in Flux.1.

use pmetal_bridge::compat::ops::{argmax_axis, slice_axis, take_along_axis, tri};
use pmetal_bridge::compat::{Array, Dtype, Exception, ModuleParameters, Param, fast, nn, ops};
use pmetal_bridge::impl_module_params;
use serde::{Deserialize, Serialize};

use crate::architectures::utils::{Activation, resolve_activation};

/// CLIP text encoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CLIPConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    /// `hidden_act`, verbatim from the reference config.
    ///
    /// This was a `use_quick_gelu: bool`, which cannot distinguish the three
    /// GELUs a CLIP checkpoint can name: OpenAI's CLIP is `"quick_gelu"`, but
    /// LAION re-trains ship `"gelu"` (exact erf) and some variants
    /// `"gelu_new"` (tanh). Both non-quick spellings collapsed to the same
    /// wrong function under the boolean.
    pub hidden_act: String,
}

impl Default for CLIPConfig {
    fn default() -> Self {
        // Defaults for CLIP-ViT-L/14 used in Flux.1
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            hidden_act: "quick_gelu".to_string(),
        }
    }
}

/// CLIP Attention layer.
#[derive(Debug)]
pub struct CLIPAttention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub out_proj: nn::Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}
impl_module_params!(CLIPAttention; q_proj, k_proj, v_proj, out_proj);

impl CLIPAttention {
    pub fn new(config: &CLIPConfig) -> Self {
        let dim = config.embed_dim as i32;
        let num_heads = config.num_heads;
        let head_dim = config.embed_dim / num_heads;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(dim, dim).build().unwrap();
        let k_proj = nn::LinearBuilder::new(dim, dim).build().unwrap();
        let v_proj = nn::LinearBuilder::new(dim, dim).build().unwrap();
        let out_proj = nn::LinearBuilder::new(dim, dim).build().unwrap();

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale,
        }
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let b = x.dim(0);
        let l = x.dim(1);

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = q
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])
            .transpose_axes(&[0, 2, 1, 3]);
        let k = k
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])
            .transpose_axes(&[0, 2, 1, 3]);
        let v = v
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])
            .transpose_axes(&[0, 2, 1, 3]);

        let out = pmetal_bridge::compat::fast::scaled_dot_product_attention_masked(
            &q, &k, &v, self.scale, mask,
        );
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
    pub fn new(config: &CLIPConfig) -> Self {
        let fc1 = nn::LinearBuilder::new(config.embed_dim as i32, config.intermediate_size as i32)
            .build()
            .unwrap();
        let fc2 = nn::LinearBuilder::new(config.intermediate_size as i32, config.embed_dim as i32)
            .build()
            .unwrap();
        let act = resolve_activation(&config.hidden_act)
            .unwrap_or_else(|| panic!("clip: unsupported hidden_act {:?}", config.hidden_act));
        Self { fc1, fc2, act }
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let x = self.fc1.forward(x);
        let x = (self.act)(&x);
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
    pub fn new(config: &CLIPConfig) -> Self {
        let attn = CLIPAttention::new(config);
        let mlp = CLIPMLP::new(config);
        let norm1 = nn::LayerNormBuilder::new(config.embed_dim as i32)
            .eps(config.layer_norm_eps)
            .build()
            .unwrap();
        let norm2 = nn::LayerNormBuilder::new(config.embed_dim as i32)
            .eps(config.layer_norm_eps)
            .build()
            .unwrap();
        Self {
            attn,
            mlp,
            norm1,
            norm2,
        }
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let residual = x;
        let x = self.norm1.forward(x);
        let x = self.attn.forward(&x, mask)?;
        let x = residual.add(&x);

        let residual = &x;
        let x = self.norm2.forward(&x);
        let x = self.mlp.forward(&x)?;
        Ok(residual.add(&x))
    }
}

/// CLIP Text Model.
#[derive(Debug)]
pub struct CLIPTextModel {
    pub token_embedding: nn::Embedding,
    pub position_embedding: Param<Array>,
    pub layers: Vec<CLIPEncoderLayer>,
    pub final_layer_norm: nn::LayerNorm,
}
impl_module_params!(CLIPTextModel; token_embedding, position_embedding, layers, final_layer_norm);

impl CLIPTextModel {
    pub fn new(config: CLIPConfig) -> Self {
        let token_embedding =
            nn::Embedding::new(config.vocab_size as i32, config.embed_dim as i32).unwrap();
        let position_embedding = Param::new(pmetal_bridge::compat::ops::zeros(
            &[
                1,
                config.max_position_embeddings as i32,
                config.embed_dim as i32,
            ],
            Dtype::Float32,
        ));
        let layers = (0..config.num_layers)
            .map(|_| CLIPEncoderLayer::new(&config))
            .collect();
        let final_layer_norm = nn::LayerNormBuilder::new(config.embed_dim as i32)
            .eps(config.layer_norm_eps)
            .build()
            .unwrap();

        Self {
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm,
        }
    }

    fn create_causal_mask(l: i32) -> Result<Array, Exception> {
        let mask = tri(l, l, 0, Dtype::Float32);
        let neg_inf = Array::from_f32(f32::NEG_INFINITY);
        let zero = Array::from_f32(0.0);
        Ok(pmetal_bridge::compat::ops::where_fn(
            &mask.equal(&zero),
            &neg_inf,
            &zero,
        ))
    }

    pub fn forward(&mut self, input_ids: &Array) -> Result<(Array, Array), Exception> {
        let l = input_ids.dim(1);
        let mut x = self.token_embedding.forward(input_ids);
        let pos_emb = self.position_embedding.as_ref();
        x = x.add(&slice_axis(pos_emb, 1, 0, l as i32));

        let mask = Self::create_causal_mask(l)?;

        let mut hidden_state = x.clone();
        let num_layers = self.layers.len();
        let clip_skip = 2;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x, Some(&mask))?;
            // CLIP often uses hidden states from the penultimate layer (clip_skip=2)
            if (num_layers >= clip_skip && i == num_layers - clip_skip)
                || (num_layers < clip_skip && i == num_layers - 1)
            {
                hidden_state = x.clone();
            }
        }

        let x = self.final_layer_norm.forward(&x);

        // Pooled output: HuggingFace CLIP pools at the EOS token position,
        // which is at argmax(input_ids) since EOS has the highest token ID.
        let b = x.dim(0);
        let eos_positions = argmax_axis(input_ids, -1);
        let eos_idx = eos_positions.reshape(&[b, 1, 1]);
        let eos_idx = pmetal_bridge::compat::ops::broadcast_to(&eos_idx, &[b, 1, x.dim(2)]);
        let pooled_output = take_along_axis(&x, &eos_idx, 1).squeeze(1);

        Ok((pooled_output, hidden_state))
    }
}
