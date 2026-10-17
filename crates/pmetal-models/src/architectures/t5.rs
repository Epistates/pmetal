//! T5 encoder.
//!
//! The encoder half of T5 (Raffel et al., 2020; v1.1 and Flan follow Shazeer,
//! 2020 for the gated feed-forward), as `T5EncoderModel` in `transformers`: the
//! text encoder FLUX conditions on and the backbone of sentence-T5 style
//! embedders.
//!
//! What sets it apart from a BERT-style encoder:
//!
//! - **No score scaling.** `softmax(Q·Kᵀ + bias)`; the `1/√d` is folded into
//!   the weights at initialisation. Scaling here as well shrinks every
//!   attention logit by `√d_kv` and flattens the attention.
//! - **Relative position bias.** Only the first block owns the bucket
//!   embedding; its `[heads, query, key]` bias (plus the padding mask) is
//!   shared by every block.
//! - **A bias-free RMS norm** (`T5LayerNorm`) and bias-free projections.
//! - A **gated** (`gated-gelu`, `gated-silu`) or plain (`relu`) feed-forward.

use pmetal_runtime::compat::{Array, Exception, nn};
use pmetal_runtime::impl_module_params;
use pmetal_runtime::nn::{AttentionMaskType, FusedAttentionConfig, fused_sdpa};
use serde::{Deserialize, Serialize};

use crate::architectures::utils::{
    Activation, additive_padding_mask, resolve_activation,
};

/// T5 encoder configuration, with the defaults of `transformers`' `T5Config`
/// for the fields a `config.json` may leave out.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T5Config {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_d_ff")]
    pub d_ff: usize,
    /// Width of one attention head. Not `d_model / num_heads`: T5 sizes the
    /// attention's inner dimension independently of the model width.
    #[serde(default = "default_d_kv")]
    pub d_kv: usize,
    #[serde(default = "default_num_layers", alias = "num_hidden_layers")]
    pub num_layers: usize,
    #[serde(default = "default_num_heads", alias = "num_attention_heads")]
    pub num_heads: usize,
    #[serde(default = "default_buckets")]
    pub relative_attention_num_buckets: usize,
    #[serde(default = "default_max_distance")]
    pub relative_attention_max_distance: usize,
    #[serde(default = "default_epsilon")]
    pub layer_norm_epsilon: f32,
    /// `"relu"` (the original T5), or `"gated-gelu"` / `"gated-silu"` (v1.1,
    /// Flan).
    #[serde(default = "default_feed_forward_proj")]
    pub feed_forward_proj: String,
}

fn default_vocab_size() -> usize {
    32128
}
fn default_d_model() -> usize {
    512
}
fn default_d_ff() -> usize {
    2048
}
fn default_d_kv() -> usize {
    64
}
fn default_num_layers() -> usize {
    6
}
fn default_num_heads() -> usize {
    8
}
fn default_buckets() -> usize {
    32
}
fn default_max_distance() -> usize {
    128
}
fn default_epsilon() -> f32 {
    1e-6
}
fn default_feed_forward_proj() -> String {
    "relu".to_string()
}

impl T5Config {
    /// The activation name the reference derives from `feed_forward_proj`.
    ///
    /// `T5Config.__init__` splits on `-` and takes the last segment, with one
    /// special case: `"gated-gelu"` resolves to `"gelu_new"` — the **tanh**
    /// approximation, not the exact erf GELU that a bare `"gelu"` would mean.
    /// T5-XXL (FLUX's text encoder) is `gated-gelu`, so that special case is
    /// the one that actually runs.
    pub fn dense_act_fn(&self) -> &str {
        if self.feed_forward_proj == "gated-gelu" {
            "gelu_new"
        } else {
            self.feed_forward_proj
                .rsplit('-')
                .next()
                .unwrap_or(&self.feed_forward_proj)
        }
    }

    /// Whether the feed-forward is gated (`wi_0 · wi_1`), as against a single
    /// `wi`.
    pub fn is_gated_act(&self) -> bool {
        self.feed_forward_proj.starts_with("gated-")
    }

    /// The width of the attention's query, key and value projections.
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.d_kv
    }
}

impl Default for T5Config {
    /// `transformers`' `T5Config()`: the t5-small shape with a ReLU
    /// feed-forward.
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            d_model: default_d_model(),
            d_ff: default_d_ff(),
            d_kv: default_d_kv(),
            num_layers: default_num_layers(),
            num_heads: default_num_heads(),
            relative_attention_num_buckets: default_buckets(),
            relative_attention_max_distance: default_max_distance(),
            layer_norm_epsilon: default_epsilon(),
            feed_forward_proj: default_feed_forward_proj(),
        }
    }
}

/// The bucket of a signed relative position (`key - query`) in a bidirectional
/// encoder, as `T5Attention._relative_position_bucket`.
///
/// Half the buckets are for keys after the query. Within a direction, the
/// nearest `num_buckets / 4` offsets get one bucket each and the rest share
/// logarithmically spaced buckets up to `max_distance`, beyond which they
/// share the last. The logarithm runs in f32 and truncates toward zero, as the
/// reference's `.to(torch.long)` does, so bucket boundaries land on the same
/// offsets.
fn relative_position_bucket(
    relative_position: i64,
    num_buckets: usize,
    max_distance: usize,
) -> i32 {
    let half = (num_buckets / 2) as i64;
    let offset = if relative_position > 0 { half } else { 0 };
    let distance = relative_position.abs();
    let max_exact = half / 2;
    if distance < max_exact {
        return (offset + distance) as i32;
    }
    let log_range = ((max_distance as f64) / (max_exact as f64)).ln() as f32;
    let scaled = (distance as f32 / max_exact as f32).ln() / log_range * (half - max_exact) as f32;
    (offset + (max_exact + scaled as i64).min(half - 1)) as i32
}

/// T5's learned relative position bias.
#[derive(Debug)]
pub struct T5RelativePositionBias {
    pub embedding: nn::Embedding,
    pub num_buckets: usize,
    pub max_distance: usize,
    pub num_heads: usize,
}
impl_module_params!(T5RelativePositionBias; embedding);

impl T5RelativePositionBias {
    pub fn new(config: &T5Config) -> Result<Self, Exception> {
        Ok(Self {
            embedding: nn::Embedding::new(
                config.relative_attention_num_buckets as i32,
                config.num_heads as i32,
            )?,
            num_buckets: config.relative_attention_num_buckets,
            max_distance: config.relative_attention_max_distance,
            num_heads: config.num_heads,
        })
    }

    /// The bias for a `length x length` self-attention: `[1, heads, length,
    /// length]`, in the embedding's dtype.
    pub fn forward(&mut self, length: usize) -> Result<Array, Exception> {
        // The bucket depends only on `key - query`, so look each of the
        // `2·length - 1` offsets up once and gather the matrix from them.
        let offsets: Vec<i32> = (0..2 * length as i64 - 1)
            .map(|i| {
                relative_position_bucket(
                    i - (length as i64 - 1),
                    self.num_buckets,
                    self.max_distance,
                )
            })
            .collect();
        let buckets: Vec<i32> = (0..length)
            .flat_map(|query| {
                let offsets = &offsets;
                (0..length).map(move |key| offsets[key + length - 1 - query])
            })
            .collect();
        let buckets = Array::from_slice(&buckets, &[length as i32, length as i32]);

        let values = self.embedding.forward(&buckets); // [query, key, heads]
        Ok(values.transpose_axes(&[2, 0, 1]).expand_dims_axes(&[0]))
    }
}

/// T5's feed-forward: `wo(act(wi_0 x) · wi_1 x)` for a gated variant, `wo(act(wi
/// x))` for the original.
#[derive(Debug)]
pub struct T5FeedForward {
    /// The single input projection of the original, ungated T5.
    pub wi: Option<nn::Linear>,
    /// The activated half of a gated feed-forward.
    pub wi_0: Option<nn::Linear>,
    /// The linear half of a gated feed-forward.
    pub wi_1: Option<nn::Linear>,
    pub wo: nn::Linear,
    /// `ACT2FN[config.dense_act_fn]`, resolved once — see
    /// [`T5Config::dense_act_fn`].
    pub act: Activation,
}
impl_module_params!(T5FeedForward; wi, wi_0, wi_1, wo);

impl T5FeedForward {
    pub fn new(config: &T5Config) -> Result<Self, Exception> {
        let linear = |inputs: usize, outputs: usize| {
            nn::LinearBuilder::new(inputs as i32, outputs as i32)
                .bias(false)
                .build()
        };
        let act = resolve_activation(config.dense_act_fn()).ok_or_else(|| {
            Exception::custom(format!(
                "t5: unsupported feed_forward_proj {:?}",
                config.feed_forward_proj
            ))
        })?;
        let (wi, wi_0, wi_1) = if config.is_gated_act() {
            (
                None,
                Some(linear(config.d_model, config.d_ff)?),
                Some(linear(config.d_model, config.d_ff)?),
            )
        } else {
            (Some(linear(config.d_model, config.d_ff)?), None, None)
        };
        Ok(Self {
            wi,
            wi_0,
            wi_1,
            wo: linear(config.d_ff, config.d_model)?,
            act,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let hidden = match (&mut self.wi, &mut self.wi_0, &mut self.wi_1) {
            (_, Some(wi_0), Some(wi_1)) => {
                (self.act)(&wi_0.forward(x)).multiply(&wi_1.forward(x))
            }
            (Some(wi), _, _) => (self.act)(&wi.forward(x)),
            _ => {
                return Err(Exception::custom(
                    "t5: feed-forward has no input projection",
                ));
            }
        };
        Ok(self.wo.forward(&hidden))
    }
}

/// T5 self-attention.
#[derive(Debug)]
pub struct T5Attention {
    pub q: nn::Linear,
    pub k: nn::Linear,
    pub v: nn::Linear,
    pub o: nn::Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Present on the first block only.
    pub relative_attention_bias: Option<T5RelativePositionBias>,
}
impl_module_params!(T5Attention; q, k, v, o, relative_attention_bias);

impl T5Attention {
    pub fn new(config: &T5Config, has_relative_attention_bias: bool) -> Result<Self, Exception> {
        let linear = |inputs: usize, outputs: usize| {
            nn::LinearBuilder::new(inputs as i32, outputs as i32)
                .bias(false)
                .build()
        };
        let inner = config.inner_dim();
        Ok(Self {
            q: linear(config.d_model, inner)?,
            k: linear(config.d_model, inner)?,
            v: linear(config.d_model, inner)?,
            o: linear(inner, config.d_model)?,
            num_heads: config.num_heads,
            head_dim: config.d_kv,
            relative_attention_bias: has_relative_attention_bias
                .then(|| T5RelativePositionBias::new(config))
                .transpose()?,
        })
    }

    /// Attend over `x` with `bias`, the `[batch | 1, heads, query, key]`
    /// relative position bias plus padding mask, added to the *unscaled*
    /// scores.
    pub fn forward(&mut self, x: &Array, bias: &Array) -> Result<Array, Exception> {
        let (b, l) = (x.dim(0), x.dim(1));
        let split = |t: Array| {
            t.reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])
                .transpose_axes(&[0, 2, 1, 3])
        };
        let q = split(self.q.forward(x));
        let k = split(self.k.forward(x));
        let v = split(self.v.forward(x));

        let config = FusedAttentionConfig::new(
            self.num_heads as i32,
            self.num_heads as i32,
            self.head_dim as i32,
        )
        .with_scale(1.0)
        .with_mask_type(AttentionMaskType::None);
        let out = fused_sdpa(&q, &k, &v, &config, Some(bias))?;
        let out = out.transpose_axes(&[0, 2, 1, 3]).reshape(&[b, l, -1]);
        Ok(self.o.forward(&out))
    }
}

/// One encoder block: pre-norm self-attention, then a pre-norm feed-forward,
/// each added to the residual stream.
///
/// The fields keep the order of `T5Block.layer`, which holds the attention
/// layer at index 0 and the feed-forward at index 1.
#[derive(Debug)]
pub struct T5Block {
    pub layer_0_norm: nn::RmsNorm,
    pub layer_0_attn: T5Attention,
    pub layer_1_norm: nn::RmsNorm,
    pub layer_1_mlp: T5FeedForward,
}
impl_module_params!(T5Block; layer_0_norm, layer_0_attn, layer_1_norm, layer_1_mlp);

impl T5Block {
    pub fn new(config: &T5Config, has_relative_attention_bias: bool) -> Result<Self, Exception> {
        let norm = || {
            nn::RmsNormBuilder::new(config.d_model as i32)
                .eps(config.layer_norm_epsilon)
                .build()
        };
        Ok(Self {
            layer_0_norm: norm()?,
            layer_0_attn: T5Attention::new(config, has_relative_attention_bias)?,
            layer_1_norm: norm()?,
            layer_1_mlp: T5FeedForward::new(config)?,
        })
    }

    pub fn forward(&mut self, x: &Array, bias: &Array) -> Result<Array, Exception> {
        let attn = self
            .layer_0_attn
            .forward(&self.layer_0_norm.forward(x), bias)?;
        let x = x.add(&attn);
        let mlp = self.layer_1_mlp.forward(&self.layer_1_norm.forward(&x))?;
        Ok(x.add(&mlp))
    }
}

/// The T5 encoder: token embedding, blocks, final norm.
#[derive(Debug)]
pub struct T5EncoderModel {
    pub shared: nn::Embedding,
    pub blocks: Vec<T5Block>,
    pub final_layer_norm: nn::RmsNorm,
    /// Model configuration (not a parameter).
    pub config: T5Config,
}
impl_module_params!(T5EncoderModel; shared, blocks, final_layer_norm);

impl T5EncoderModel {
    pub fn new(config: T5Config) -> Result<Self, Exception> {
        let blocks = (0..config.num_layers)
            .map(|i| T5Block::new(&config, i == 0))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            shared: nn::Embedding::new(config.vocab_size as i32, config.d_model as i32)?,
            blocks,
            final_layer_norm: nn::RmsNormBuilder::new(config.d_model as i32)
                .eps(config.layer_norm_epsilon)
                .build()?,
            config,
        })
    }

    pub fn config(&self) -> &T5Config {
        &self.config
    }

    /// Encode `input_ids` (`[batch, length]`).
    ///
    /// `attention_mask` is `[batch, length]`, non-zero for real tokens; padded
    /// keys are hidden from every query. Without one every token attends to
    /// every other, which is what FLUX's pipeline does with its padded prompt.
    ///
    /// Returns `[batch, length, d_model]` in the dtype of the weights.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut x = self.shared.forward(input_ids);
        let Some(first) = self.blocks.first_mut() else {
            return Ok(self.final_layer_norm.forward(&x));
        };
        let relative = first
            .layer_0_attn
            .relative_attention_bias
            .as_mut()
            .ok_or_else(|| Exception::custom("t5: the first block has no relative bias"))?;
        let mut bias = relative.forward(input_ids.dim(1) as usize)?;
        if let Some(mask) = attention_mask {
            bias = bias.add(&additive_padding_mask(mask, bias.dtype()));
        }
        for block in &mut self.blocks {
            x = block.forward(&x, &bias)?;
        }
        Ok(self.final_layer_norm.forward(&x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every bucket of a 32-bucket, 128-distance bidirectional encoder for
    /// offsets -599 to 599, against `T5Attention._relative_position_bucket`
    /// run in torch. The logarithmic range truncates, so a one-ulp difference
    /// in the logarithm would move a boundary offset to its neighbour's bucket;
    /// only the whole table pins that down.
    #[test]
    fn buckets_match_the_reference_for_every_offset() {
        let want: Vec<i32> =
            serde_json::from_str(include_str!("../../tests/fixtures/t5_buckets_600.json"))
                .expect("fixture parses");
        assert_eq!(want.len(), 1199);
        for (i, &want) in want.iter().enumerate() {
            let relative = i as i64 - 599;
            assert_eq!(
                relative_position_bucket(relative, 32, 128),
                want,
                "relative position {relative}"
            );
        }
    }

    /// The first of each bucket's offsets, spot-checked by value so the
    /// fixture comparison above cannot pass on two tables that are both wrong
    /// the same way: exact buckets below 8, logarithmic ones above, mirrored.
    #[test]
    fn buckets_split_exact_and_logarithmic_ranges() {
        let bucket = |rp| relative_position_bucket(rp, 32, 128);
        assert_eq!(
            [bucket(0), bucket(-1), bucket(-7), bucket(-8)],
            [0, 1, 7, 8]
        );
        assert_eq!([bucket(1), bucket(7), bucket(8)], [17, 23, 24]);
        assert_eq!([bucket(-40), bucket(-90), bucket(-127)], [12, 14, 15]);
        assert_eq!([bucket(500), bucket(-500)], [31, 15]);
    }
}
