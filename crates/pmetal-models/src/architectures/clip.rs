//! CLIP text encoder architecture.
//!
//! Implementation of CLIP (Contrastive Language-Image Pre-training) text encoder.
//! Based on the architecture from OpenAI and used in Flux.1.

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        indexing::{IndexOp, argmax_axis, take_along_axis},
        tri,
    },
};
use serde::{Deserialize, Serialize};

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
    pub use_quick_gelu: bool,
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
            use_quick_gelu: true,
        }
    }
}

/// CLIP Attention layer.
#[derive(Debug, ModuleParameters)]
pub struct CLIPAttention {
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub out_proj: nn::Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl CLIPAttention {
    pub fn new(config: &CLIPConfig) -> Self {
        let dim = config.embed_dim as i32;
        let num_heads = config.num_heads;
        let head_dim = config.embed_dim / num_heads;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(dim, dim)
            .build()
            .expect("Infallible");
        let k_proj = nn::LinearBuilder::new(dim, dim)
            .build()
            .expect("Infallible");
        let v_proj = nn::LinearBuilder::new(dim, dim)
            .build()
            .expect("Infallible");
        let out_proj = nn::LinearBuilder::new(dim, dim)
            .build()
            .expect("Infallible");

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

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let out = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            self.scale,
            mask.map(Into::into),
            None,
        )?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        self.out_proj.forward(&out)
    }
}

/// CLIP MLP layer.
#[derive(Debug, ModuleParameters)]
pub struct CLIPMLP {
    #[param]
    pub fc1: nn::Linear,
    #[param]
    pub fc2: nn::Linear,
    pub use_quick_gelu: bool,
}

impl CLIPMLP {
    pub fn new(config: &CLIPConfig) -> Self {
        let fc1 = nn::LinearBuilder::new(config.embed_dim as i32, config.intermediate_size as i32)
            .build()
            .expect("Infallible");
        let fc2 = nn::LinearBuilder::new(config.intermediate_size as i32, config.embed_dim as i32)
            .build()
            .expect("Infallible");
        Self {
            fc1,
            fc2,
            use_quick_gelu: config.use_quick_gelu,
        }
    }

    fn quick_gelu(x: &Array) -> Result<Array, Exception> {
        x.multiply(&mlx_rs::ops::sigmoid(
            &x.multiply(&Array::from_f32(1.702))?,
        )?)
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let x = self.fc1.forward(x)?;
        let x = if self.use_quick_gelu {
            Self::quick_gelu(&x)?
        } else {
            nn::gelu(&x)?
        };
        self.fc2.forward(&x)
    }
}

/// CLIP Encoder Layer.
#[derive(Debug, ModuleParameters)]
pub struct CLIPEncoderLayer {
    #[param]
    pub attn: CLIPAttention,
    #[param]
    pub mlp: CLIPMLP,
    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub norm2: nn::LayerNorm,
}

impl CLIPEncoderLayer {
    pub fn new(config: &CLIPConfig) -> Self {
        let attn = CLIPAttention::new(config);
        let mlp = CLIPMLP::new(config);
        let norm1 = nn::LayerNormBuilder::new(config.embed_dim as i32)
            .eps(config.layer_norm_eps)
            .build()
            .expect("Infallible");
        let norm2 = nn::LayerNormBuilder::new(config.embed_dim as i32)
            .eps(config.layer_norm_eps)
            .build()
            .expect("Infallible");
        Self {
            attn,
            mlp,
            norm1,
            norm2,
        }
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = self.attn.forward(&x, mask)?;
        let x = residual.add(&x)?;

        let residual = &x;
        let x = self.norm2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual.add(&x)
    }
}

/// CLIP Text Model.
#[derive(Debug, ModuleParameters)]
pub struct CLIPTextModel {
    #[param]
    pub token_embedding: nn::Embedding,
    #[param]
    pub position_embedding: Param<Array>,
    #[param]
    pub layers: Vec<CLIPEncoderLayer>,
    #[param]
    pub final_layer_norm: nn::LayerNorm,
}

impl CLIPTextModel {
    pub fn new(config: CLIPConfig) -> Self {
        let token_embedding = nn::Embedding::new(config.vocab_size as i32, config.embed_dim as i32)
            .expect("Infallible");
        let position_embedding = Param::new(
            mlx_rs::ops::zeros::<f32>(&[
                1,
                config.max_position_embeddings as i32,
                config.embed_dim as i32,
            ])
            .expect("Infallible"),
        );
        let layers = (0..config.num_layers)
            .map(|_| CLIPEncoderLayer::new(&config))
            .collect();
        let final_layer_norm = nn::LayerNormBuilder::new(config.embed_dim as i32)
            .eps(config.layer_norm_eps)
            .build()
            .expect("Infallible");

        Self {
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm,
        }
    }

    fn create_causal_mask(l: i32) -> Result<Array, Exception> {
        let mask = tri::<f32>(l, None, None)?;
        let neg_inf = Array::from_f32(f32::NEG_INFINITY);
        let zero = Array::from_f32(0.0);
        mlx_rs::ops::r#where(&mask.eq(&zero)?, &neg_inf, &zero)
    }

    pub fn forward(&mut self, input_ids: &Array) -> Result<(Array, Array), Exception> {
        let l = input_ids.dim(1);
        let mut x = self.token_embedding.forward(input_ids)?;
        x = x.add(&self.position_embedding.as_ref().index((.., ..l as i32, ..)))?;

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

        let x = self.final_layer_norm.forward(&x)?;

        // Pooled output: HuggingFace CLIP pools at the EOS token position,
        // which is at argmax(input_ids) since EOS has the highest token ID.
        let b = x.dim(0);
        let eos_positions = argmax_axis(input_ids, -1, None)?;
        let eos_idx = eos_positions.reshape(&[b, 1, 1])?;
        let eos_idx = mlx_rs::ops::broadcast_to(&eos_idx, &[b, 1, x.dim(2)])?;
        let pooled_output = take_along_axis(&x, &eos_idx, 1)?.squeeze_axes(&[1])?;

        Ok((pooled_output, hidden_state))
    }
}
