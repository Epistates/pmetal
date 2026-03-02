//! T5 Encoder architecture.
//!
//! Implementation of T5 (Text-to-Text Transfer Transformer) encoder.
//! Based on the architecture from Google and used in Flux.1 (T5-XXL).

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops::{indexing::IndexOp, tri},
};
use serde::{Deserialize, Serialize};

/// T5 encoder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub d_kv: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub dropout_rate: f32,
    pub layer_norm_epsilon: f32,
    pub feed_forward_proj: String,
    pub is_gated_act: bool,
}

impl Default for T5Config {
    fn default() -> Self {
        // Defaults for T5-XXL used in Flux.1
        Self {
            vocab_size: 32128,
            d_model: 4096,
            d_ff: 10240,
            d_kv: 64,
            num_layers: 24,
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            feed_forward_proj: "gated-gelu".to_string(),
            is_gated_act: true,
        }
    }
}

/// T5 Relative Position Bias.
#[derive(Debug, ModuleParameters)]
pub struct T5RelativePositionBias {
    #[param]
    pub embedding: nn::Embedding,
    pub num_buckets: usize,
    pub max_distance: usize,
    pub num_heads: usize,
}

impl T5RelativePositionBias {
    pub fn new(config: &T5Config) -> Self {
        let embedding = nn::Embedding::new(
            config.relative_attention_num_buckets as i32,
            config.num_heads as i32,
        )
        .expect("Infallible");
        Self {
            embedding,
            num_buckets: config.relative_attention_num_buckets,
            max_distance: config.relative_attention_max_distance,
            num_heads: config.num_heads,
        }
    }

    fn relative_position_bucket(
        relative_position: &Array,
        bidirectional: bool,
        num_buckets: i32,
        max_distance: i32,
    ) -> Result<Array, Exception> {
        let mut ret = mlx_rs::ops::zeros_like(relative_position)?;
        let mut num_buckets = num_buckets;
        let rel_pos_f = relative_position.as_dtype(mlx_rs::Dtype::Float32)?;

        let n = if bidirectional {
            num_buckets /= 2;
            // Offset positive relative positions by num_buckets
            let is_positive = relative_position
                .gt(&Array::from_int(0))?
                .as_dtype(mlx_rs::Dtype::Int32)?;
            ret = is_positive.multiply(&Array::from_int(num_buckets))?;
            rel_pos_f.abs()?
        } else {
            // Clamp to non-positive, then negate
            mlx_rs::ops::maximum(&rel_pos_f.negative()?, &Array::from_f32(0.0))?
        };

        // Half buckets for exact (linear) positions, half for log-spaced
        let max_exact = num_buckets / 2;
        let is_small = n.lt(&Array::from_f32(max_exact as f32))?;

        // Log-linear bucketing for large positions
        let log_ratio = n
            .divide(&Array::from_f32(max_exact as f32))?
            .log()?
            .divide(&Array::from_f32(
                (max_distance as f32 / max_exact as f32).ln(),
            ))?;
        let large_val = log_ratio
            .multiply(&Array::from_f32((num_buckets - max_exact) as f32))?
            .add(&Array::from_f32(max_exact as f32))?;
        let large_val =
            mlx_rs::ops::minimum(&large_val, &Array::from_f32((num_buckets - 1) as f32))?;

        let buckets = mlx_rs::ops::r#where(&is_small, &n, &large_val)?;
        let buckets = ret.add(&buckets.as_dtype(mlx_rs::Dtype::Int32)?)?;
        Ok(buckets)
    }

    pub fn forward(&mut self, query_length: usize, key_length: usize) -> Result<Array, Exception> {
        let context_position = mlx_rs::ops::arange::<i32, i32>(0, query_length as i32, None)?;
        let memory_position = mlx_rs::ops::arange::<i32, i32>(0, key_length as i32, None)?;

        let relative_position = memory_position
            .expand_dims_axes(&[0])?
            .subtract(&context_position.expand_dims_axes(&[1])?)?;

        let buckets = Self::relative_position_bucket(
            &relative_position,
            true,
            self.num_buckets as i32,
            self.max_distance as i32,
        )?;

        let values = self.embedding.forward(&buckets)?; // [Q, K, Heads]
        values.transpose_axes(&[2, 0, 1]) // [Heads, Q, K]
    }
}

/// T5 Layer Norm (RMSNorm style, but often called T5LayerNorm).
#[derive(Debug, ModuleParameters)]
pub struct T5LayerNorm {
    pub weight: mlx_rs::module::Param<Array>,
    pub variance_epsilon: f32,
}

impl T5LayerNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        let weight = mlx_rs::module::Param::new(
            mlx_rs::ops::ones::<f32>(&[dim as i32]).expect("Infallible"),
        );
        Self {
            weight,
            variance_epsilon: eps,
        }
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let variance = x.power(&Array::from_f32(2.0))?.mean_axes(&[-1], true)?;
        let x = x.multiply(&mlx_rs::ops::rsqrt(
            &variance.add(&Array::from_f32(self.variance_epsilon))?,
        )?)?;
        x.multiply(&self.weight)
    }
}

/// T5 Dense Gated Activation (for T5-v1.1 and later).
#[derive(Debug, ModuleParameters)]
pub struct T5DenseGatedActDense {
    #[param]
    pub wi_0: nn::Linear,
    #[param]
    pub wi_1: nn::Linear,
    #[param]
    pub wo: nn::Linear,
}

impl T5DenseGatedActDense {
    pub fn new(config: &T5Config) -> Self {
        let wi_0 = nn::LinearBuilder::new(config.d_model as i32, config.d_ff as i32)
            .bias(false)
            .build()
            .expect("Infallible");
        let wi_1 = nn::LinearBuilder::new(config.d_model as i32, config.d_ff as i32)
            .bias(false)
            .build()
            .expect("Infallible");
        let wo = nn::LinearBuilder::new(config.d_ff as i32, config.d_model as i32)
            .bias(false)
            .build()
            .expect("Infallible");
        Self { wi_0, wi_1, wo }
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let hidden_gelu = nn::gelu_approximate(&self.wi_0.forward(x)?)?;
        let hidden_linear = self.wi_1.forward(x)?;
        let x = hidden_gelu.multiply(&hidden_linear)?;
        self.wo.forward(&x)
    }
}

/// T5 Attention layer.
#[derive(Debug, ModuleParameters)]
pub struct T5Attention {
    #[param]
    pub q: nn::Linear,
    #[param]
    pub k: nn::Linear,
    #[param]
    pub v: nn::Linear,
    #[param]
    pub o: nn::Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    #[param]
    pub relative_attention_bias: Option<T5RelativePositionBias>,
}

impl T5Attention {
    pub fn new(config: &T5Config, has_relative_attention_bias: bool) -> Self {
        let dim = config.d_model as i32;
        let inner_dim = (config.num_heads * config.d_kv) as i32;

        let q = nn::LinearBuilder::new(dim, inner_dim)
            .bias(false)
            .build()
            .expect("Infallible");
        let k = nn::LinearBuilder::new(dim, inner_dim)
            .bias(false)
            .build()
            .expect("Infallible");
        let v = nn::LinearBuilder::new(dim, inner_dim)
            .bias(false)
            .build()
            .expect("Infallible");
        let o = nn::LinearBuilder::new(inner_dim, dim)
            .bias(false)
            .build()
            .expect("Infallible");

        let relative_attention_bias = if has_relative_attention_bias {
            Some(T5RelativePositionBias::new(config))
        } else {
            None
        };

        Self {
            q,
            k,
            v,
            o,
            num_heads: config.num_heads,
            head_dim: config.d_kv,
            scale: (config.d_kv as f32).sqrt().recip(),
            relative_attention_bias,
        }
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_bias: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let b = x.dim(0);
        let l = x.dim(1);

        let q = self.q.forward(x)?;
        let k = self.k.forward(x)?;
        let v = self.v.forward(x)?;

        let q = q
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let mut bias = position_bias.cloned();
        if bias.is_none() {
            if let Some(ref mut rel_bias) = self.relative_attention_bias {
                bias = Some(rel_bias.forward(l as usize, l as usize)?);
            }
        }

        // Add mask and bias to attention scores
        // Note: simplified SDPA might not support external bias easily if it's not a mask.
        // We'll use manual attention if bias is present or combine it into mask.

        let mut attn_mask = mask.cloned();
        if let Some(ref b) = bias {
            // Combine mask and bias
            // mask is usually additive (0 or -inf)
            // bias is additive scores
            if let Some(ref m) = attn_mask {
                attn_mask = Some(m.add(b)?);
            } else {
                attn_mask = Some(b.clone());
            }
        }

        let out = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            self.scale,
            attn_mask.as_ref().map(Into::into),
            None,
        )?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        let out = self.o.forward(&out)?;

        Ok((
            out,
            bias.unwrap_or_else(|| mlx_rs::ops::zeros::<f32>(&[1, 1, 1]).unwrap()),
        )) // Return bias for next layers
    }
}

/// T5 Block.
#[derive(Debug, ModuleParameters)]
pub struct T5Block {
    #[param]
    pub layer_0_norm: T5LayerNorm,
    #[param]
    pub layer_0_attn: T5Attention,
    #[param]
    pub layer_1_norm: T5LayerNorm,
    #[param]
    pub layer_1_mlp: T5DenseGatedActDense,
}

impl T5Block {
    pub fn new(config: &T5Config, has_relative_attention_bias: bool) -> Self {
        let layer_0_norm = T5LayerNorm::new(config.d_model, config.layer_norm_epsilon);
        let layer_0_attn = T5Attention::new(config, has_relative_attention_bias);
        let layer_1_norm = T5LayerNorm::new(config.d_model, config.layer_norm_epsilon);
        let layer_1_mlp = T5DenseGatedActDense::new(config);

        Self {
            layer_0_norm,
            layer_0_attn,
            layer_1_norm,
            layer_1_mlp,
        }
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_bias: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let residual = x;
        let x_norm = self.layer_0_norm.forward(x)?;
        let (attn_out, bias) = self.layer_0_attn.forward(&x_norm, mask, position_bias)?;
        let x = residual.add(&attn_out)?;

        let residual = &x;
        let x_norm = self.layer_1_norm.forward(&x)?;
        let mlp_out = self.layer_1_mlp.forward(&x_norm)?;
        let x = residual.add(&mlp_out)?;

        Ok((x, bias))
    }
}

/// T5 Encoder Model.
#[derive(Debug, ModuleParameters)]
pub struct T5EncoderModel {
    #[param]
    pub shared: nn::Embedding,
    #[param]
    pub blocks: Vec<T5Block>,
    #[param]
    pub final_layer_norm: T5LayerNorm,
}

impl T5EncoderModel {
    pub fn new(config: T5Config) -> Self {
        let shared = nn::Embedding::new(config.vocab_size as i32, config.d_model as i32)
            .expect("Infallible");
        let blocks = (0..config.num_layers)
            .map(|i| T5Block::new(&config, i == 0))
            .collect();
        let final_layer_norm = T5LayerNorm::new(config.d_model, config.layer_norm_epsilon);

        Self {
            shared,
            blocks,
            final_layer_norm,
        }
    }

    pub fn forward(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        let mut x = self.shared.forward(input_ids)?;

        let mut position_bias = None;
        for block in &mut self.blocks {
            let (out, bias) = block.forward(&x, None, position_bias.as_ref())?;
            x = out;
            position_bias = Some(bias);
        }

        self.final_layer_norm.forward(&x)
    }
}
