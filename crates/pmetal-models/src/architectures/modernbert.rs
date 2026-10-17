//! ModernBERT encoder (Warner et al., 2024).
//!
//! A modernised bidirectional encoder, as `ModernBertModel` in `transformers`:
//! pre-norm blocks, a GeGLU feed-forward (`Wi` produces the activated and the
//! gating halves together), rotary positions in place of a learned table, and
//! attention that alternates between **global** layers and **local** ones that
//! see only the tokens within `local_attention / 2` positions either side.
//! Global and local layers rotate with different bases (160000 and 10000 in the
//! released models), and every bias defaults to off.
//!
//! The parameter tree uses `transformers`' own names (`layers.3.attn.Wqkv`,
//! `layers.3.mlp.Wi`, `final_norm`), so a checkpoint loads by name. Layer 0 has
//! no attention norm: the embedding norm stands in for it.
//!
//! Like [`BertModel`](super::bert::BertModel) this returns hidden states
//! `[batch, seq, hidden]` for pooling; the masked-LM head, the decoder and any
//! classification head are not part of it.

use pmetal_runtime::compat::{Array, Dtype, Exception, nn, ops};
use pmetal_runtime::impl_module_params;
use pmetal_runtime::nn::{AttentionMaskType, FusedAttentionConfig, fused_sdpa};
use pmetal_runtime::rope::RotaryEmbedding;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::architectures::utils::{
    Activation, additive_mask, padding_keep, resolve_activation,
};

/// The released models' rotary bases.
const DEFAULT_GLOBAL_ROPE_THETA: f64 = 160_000.0;
const DEFAULT_LOCAL_ROPE_THETA: f64 = 10_000.0;

/// ModernBERT configuration, reading both spellings of the rotary settings:
/// the original `global_rope_theta` / `local_rope_theta` and the newer
/// `rope_parameters` per layer type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModernBertConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: String,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    /// Whether the LayerNorms have a bias (the released models: no).
    #[serde(default)]
    pub norm_bias: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    /// The full width of a local layer's window; a token sees `local_attention
    /// / 2` positions each way.
    #[serde(default = "default_local_attention")]
    pub local_attention: usize,
    /// Every n-th layer, from the first, attends globally.
    #[serde(default = "default_global_attn_every_n_layers")]
    pub global_attn_every_n_layers: usize,
    /// Explicit per-layer types (`full_attention` or `sliding_attention`), which
    /// replace the every-n-th rule when given.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub global_rope_theta: Option<f64>,
    #[serde(default)]
    pub local_rope_theta: Option<f64>,
    /// `{"full_attention": {"rope_theta": ..}, "sliding_attention": {..}}`.
    #[serde(default)]
    pub rope_parameters: Option<Value>,
    #[serde(default)]
    pub pad_token_id: Option<usize>,
    #[serde(default)]
    pub model_type: Option<String>,
}

fn default_vocab_size() -> usize {
    50368
}
fn default_hidden_size() -> usize {
    768
}
fn default_intermediate_size() -> usize {
    1152
}
fn default_num_hidden_layers() -> usize {
    22
}
fn default_num_attention_heads() -> usize {
    12
}
fn default_hidden_activation() -> String {
    "gelu".to_string()
}
fn default_max_position_embeddings() -> usize {
    8192
}
fn default_norm_eps() -> f32 {
    1e-5
}
fn default_local_attention() -> usize {
    128
}
fn default_global_attn_every_n_layers() -> usize {
    3
}

impl Default for ModernBertConfig {
    /// `answerdotai/ModernBERT-base`.
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            hidden_activation: default_hidden_activation(),
            max_position_embeddings: default_max_position_embeddings(),
            norm_eps: default_norm_eps(),
            norm_bias: false,
            attention_bias: false,
            mlp_bias: false,
            local_attention: default_local_attention(),
            global_attn_every_n_layers: default_global_attn_every_n_layers(),
            layer_types: None,
            global_rope_theta: None,
            local_rope_theta: None,
            rope_parameters: None,
            pad_token_id: None,
            model_type: Some("modernbert".to_string()),
        }
    }
}

/// Whether a layer attends over everything or over a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModernBertLayerType {
    Global,
    Local,
}

impl ModernBertConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// How many positions each side a local layer's token sees.
    pub fn window(&self) -> usize {
        self.local_attention / 2
    }

    /// The type of each layer.
    pub fn layer_types(&self) -> Result<Vec<ModernBertLayerType>, Exception> {
        match &self.layer_types {
            Some(types) => {
                if types.len() != self.num_hidden_layers {
                    return Err(Exception::custom(format!(
                        "modernbert: {} layer types for {} layers",
                        types.len(),
                        self.num_hidden_layers
                    )));
                }
                types
                    .iter()
                    .map(|t| match t.as_str() {
                        "full_attention" => Ok(ModernBertLayerType::Global),
                        "sliding_attention" => Ok(ModernBertLayerType::Local),
                        other => Err(Exception::custom(format!(
                            "modernbert: unknown layer type {other:?}"
                        ))),
                    })
                    .collect()
            }
            None => Ok((0..self.num_hidden_layers)
                .map(|i| {
                    if self.global_attn_every_n_layers != 0
                        && i % self.global_attn_every_n_layers == 0
                    {
                        ModernBertLayerType::Global
                    } else {
                        ModernBertLayerType::Local
                    }
                })
                .collect()),
        }
    }

    /// The rotary base of a layer type, from `rope_parameters` if given, else
    /// the original `*_rope_theta`, else the released models' value.
    pub fn rope_theta(&self, layer_type: ModernBertLayerType) -> Result<f64, Exception> {
        let (key, flat, default) = match layer_type {
            ModernBertLayerType::Global => (
                "full_attention",
                self.global_rope_theta,
                DEFAULT_GLOBAL_ROPE_THETA,
            ),
            ModernBertLayerType::Local => (
                "sliding_attention",
                self.local_rope_theta,
                DEFAULT_LOCAL_ROPE_THETA,
            ),
        };
        let params = self.rope_parameters.as_ref().and_then(|p| p.get(key));
        if let Some(rope_type) = params
            .and_then(|p| p.get("rope_type"))
            .and_then(Value::as_str)
            .filter(|t| *t != "default")
        {
            return Err(Exception::custom(format!(
                "modernbert: rope_type {rope_type:?} is not implemented; only \"default\" is"
            )));
        }
        Ok(params
            .and_then(|p| p.get("rope_theta"))
            .and_then(Value::as_f64)
            .or(flat)
            .unwrap_or(default))
    }

    fn validate(&self) -> Result<(), Exception> {
        if let Some(model_type) = self.model_type.as_deref()
            && model_type != "modernbert"
        {
            return Err(Exception::custom(format!(
                "modernbert: model_type {model_type:?} is not ModernBERT"
            )));
        }
        if self.num_attention_heads == 0 || self.hidden_size % self.num_attention_heads != 0 {
            return Err(Exception::custom(format!(
                "modernbert: hidden size {} is not a multiple of {} heads",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.head_dim() % 2 != 0 {
            return Err(Exception::custom(
                "modernbert: the head width must be even for rotary positions",
            ));
        }
        Ok(())
    }
}

fn linear(inputs: usize, outputs: usize, bias: bool) -> Result<nn::Linear, Exception> {
    nn::LinearBuilder::new(inputs as i32, outputs as i32)
        .bias(bias)
        .build()
}

fn layer_norm(config: &ModernBertConfig) -> Result<nn::LayerNorm, Exception> {
    let mut norm = nn::LayerNormBuilder::new(config.hidden_size as i32)
        .eps(config.norm_eps)
        .build()?;
    if !config.norm_bias {
        norm.bias = pmetal_runtime::compat::Param::new(None);
    }
    Ok(norm)
}

/// Token embeddings, then a LayerNorm (there is no position table).
#[derive(Debug)]
pub struct ModernBertEmbeddings {
    pub tok_embeddings: nn::Embedding,
    pub norm: nn::LayerNorm,
}
impl_module_params!(ModernBertEmbeddings; tok_embeddings, norm);

impl ModernBertEmbeddings {
    fn new(config: &ModernBertConfig) -> Result<Self, Exception> {
        Ok(Self {
            tok_embeddings: nn::Embedding::new(
                config.vocab_size as i32,
                config.hidden_size as i32,
            )?,
            norm: layer_norm(config)?,
        })
    }

    pub fn forward(&mut self, input_ids: &Array) -> Array {
        self.norm.forward(&self.tok_embeddings.forward(input_ids))
    }
}

/// Multi-head self-attention with a fused `Wqkv` and rotary positions.
#[allow(non_snake_case)]
#[derive(Debug)]
pub struct ModernBertAttention {
    pub Wqkv: nn::Linear,
    pub Wo: nn::Linear,
    num_heads: i32,
    head_dim: i32,
    rope: RotaryEmbedding,
}
impl_module_params!(ModernBertAttention; Wqkv, Wo);

impl ModernBertAttention {
    fn new(config: &ModernBertConfig, layer_type: ModernBertLayerType) -> Result<Self, Exception> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim() as i32;
        Ok(Self {
            Wqkv: linear(hidden, 3 * hidden, config.attention_bias)?,
            Wo: linear(hidden, hidden, config.attention_bias)?,
            num_heads: config.num_attention_heads as i32,
            head_dim,
            rope: RotaryEmbedding::plain(head_dim, head_dim, config.rope_theta(layer_type)?, false),
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let (batch, length) = (x.dim(0), x.dim(1));
        // [B, L, 3 * H * D] -> three of [B, H, L, D]
        let qkv = self
            .Wqkv
            .forward(x)
            .reshape(&[batch, length, 3, self.num_heads, self.head_dim]);
        let part = |i: i32| ops::select_axis(&qkv, i, 2).transpose_axes(&[0, 2, 1, 3]);
        let (q, k, v) = (part(0), part(1), part(2));
        let (q, k) = (self.rope.apply(&q, 0), self.rope.apply(&k, 0));

        let config = FusedAttentionConfig::new(self.num_heads, self.num_heads, self.head_dim)
            .with_mask_type(AttentionMaskType::None);
        let out = fused_sdpa(&q, &k, &v, &config, mask)?;
        Ok(self
            .Wo
            .forward(&out.transpose_axes(&[0, 2, 1, 3]).reshape(&[
                batch,
                length,
                self.num_heads * self.head_dim,
            ])))
    }
}

/// GeGLU feed-forward: `Wo(act(input) * gate)` with `input, gate = Wi(x)`.
#[allow(non_snake_case)]
#[derive(Debug)]
pub struct ModernBertMLP {
    pub Wi: nn::Linear,
    pub Wo: nn::Linear,
    pub act: Activation,
}
impl_module_params!(ModernBertMLP; Wi, Wo);

impl ModernBertMLP {
    fn new(config: &ModernBertConfig) -> Result<Self, Exception> {
        let act = resolve_activation(&config.hidden_activation).ok_or_else(|| {
            Exception::custom(format!(
                "modernbert: unsupported hidden_activation {:?}",
                config.hidden_activation
            ))
        })?;
        Ok(Self {
            Wi: linear(
                config.hidden_size,
                2 * config.intermediate_size,
                config.mlp_bias,
            )?,
            Wo: linear(
                config.intermediate_size,
                config.hidden_size,
                config.mlp_bias,
            )?,
            act,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Array {
        let halves = ops::split(&self.Wi.forward(x), 2, -1);
        let gated = (self.act)(&halves[0]).multiply(&halves[1]);
        self.Wo.forward(&gated)
    }
}

/// One pre-norm block. Layer 0 has no attention norm.
#[derive(Debug)]
pub struct ModernBertLayer {
    pub attn_norm: Option<nn::LayerNorm>,
    pub attn: ModernBertAttention,
    pub mlp_norm: nn::LayerNorm,
    pub mlp: ModernBertMLP,
    pub layer_type: ModernBertLayerType,
}
impl_module_params!(ModernBertLayer; attn_norm, attn, mlp_norm, mlp);

impl ModernBertLayer {
    fn new(
        config: &ModernBertConfig,
        index: usize,
        layer_type: ModernBertLayerType,
    ) -> Result<Self, Exception> {
        Ok(Self {
            attn_norm: (index != 0).then(|| layer_norm(config)).transpose()?,
            attn: ModernBertAttention::new(config, layer_type)?,
            mlp_norm: layer_norm(config)?,
            mlp: ModernBertMLP::new(config)?,
            layer_type,
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let normed = match &self.attn_norm {
            Some(norm) => norm.forward(x),
            None => x.clone(),
        };
        let x = x.add(&self.attn.forward(&normed, mask)?);
        let mlp = self.mlp.forward(&self.mlp_norm.forward(&x));
        Ok(x.add(&mlp))
    }
}

/// The ModernBERT encoder.
#[derive(Debug)]
pub struct ModernBertModel {
    pub embeddings: ModernBertEmbeddings,
    pub layers: Vec<ModernBertLayer>,
    pub final_norm: nn::LayerNorm,
    /// Model configuration (not a parameter).
    pub config: ModernBertConfig,
}
impl_module_params!(ModernBertModel; embeddings, layers, final_norm);

impl ModernBertModel {
    pub fn new(config: ModernBertConfig) -> Result<Self, Exception> {
        config.validate()?;
        let layer_types = config.layer_types()?;
        let layers = layer_types
            .iter()
            .enumerate()
            .map(|(i, &t)| ModernBertLayer::new(&config, i, t))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embeddings: ModernBertEmbeddings::new(&config)?,
            layers,
            final_norm: layer_norm(&config)?,
            config,
        })
    }

    pub fn from_config_str(config_str: &str) -> Result<Self, Exception> {
        let config: ModernBertConfig =
            serde_json::from_str(config_str).map_err(|e| Exception::custom(e.to_string()))?;
        Self::new(config)
    }

    pub fn config(&self) -> &ModernBertConfig {
        &self.config
    }

    /// `|query - key| <= window`, as a boolean `[1, 1, length, length]`.
    fn window_keep(length: i32, window: usize) -> Array {
        let positions = ops::arange(length, Dtype::Int32);
        positions
            .reshape(&[length, 1])
            .subtract(&positions.reshape(&[1, length]))
            .abs()
            .less_equal(&Array::from_int(window as i32))
            .reshape(&[1, 1, length, length])
    }

    /// Encode `input_ids` (`[batch, length]`); `attention_mask` is `[batch,
    /// length]`, non-zero for real tokens. Returns `[batch, length, hidden]`
    /// in the dtype of the weights.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let length = input_ids.dim(1);
        if let Some(mask) = attention_mask
            && (mask.ndim() != 2 || mask.dim(0) != input_ids.dim(0) || mask.dim(1) != length)
        {
            return Err(Exception::custom(format!(
                "modernbert: the attention mask must be [batch, seq] like the ids {:?}, got {:?}",
                input_ids.shape(),
                mask.shape()
            )));
        }
        let mut hidden = self.embeddings.forward(input_ids);
        let dtype = hidden.dtype();
        let keep = attention_mask.map(padding_keep);
        let padding = keep.as_ref().map(|keep| additive_mask(keep, dtype));
        // The local layers' mask: the window, narrowed by the padding when
        // there is any. The two are combined as booleans and converted once,
        // so that a padded query whose window holds only padding gets a mask
        // of finite values (a uniform row) rather than all `-inf` (NaN).
        let local = if self
            .layers
            .iter()
            .any(|l| l.layer_type == ModernBertLayerType::Local)
        {
            let window = Self::window_keep(length, self.config.window());
            let keep = match &keep {
                Some(keep) => ops::logical_and(&window, keep),
                None => window,
            };
            Some(additive_mask(&keep, dtype))
        } else {
            None
        };
        for layer in &mut self.layers {
            let mask = match layer.layer_type {
                ModernBertLayerType::Global => padding.as_ref(),
                ModernBertLayerType::Local => local.as_ref(),
            };
            hidden = layer.forward(&hidden, mask)?;
        }
        Ok(self.final_norm.forward(&hidden))
    }
}
