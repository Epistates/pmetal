//! BERT-family encoders for sentence embeddings and encoder fine-tuning.
//!
//! One bidirectional post-norm transformer encoder (Devlin et al., 2019) that
//! serves three `transformers` families, which differ only in how they name
//! things and in two details of the embeddings:
//!
//! - **BERT** (`model_type: "bert"`): learned absolute positions counted from
//!   0, and a token-type (segment) embedding.
//! - **RoBERTa** (`"roberta"`, `"xlm-roberta"`, `"camembert"`): positions
//!   counted from `pad_token_id + 1` over the non-padding tokens and pinned to
//!   `pad_token_id` at padding (Liu et al., 2019), so the table carries
//!   `pad_token_id + 1` extra rows; a one-row token-type table.
//! - **DistilBERT** (`"distilbert"`): BERT's encoder with no token types and a
//!   differently spelled config (`dim`, `n_layers`, `n_heads`, `hidden_dim`).
//!
//! Unlike causal LM architectures this model:
//! - uses **bidirectional** self-attention, hiding only padding;
//! - has **no KV cache** (full context every forward pass);
//! - returns **hidden states** `[batch, seq, hidden_dim]`, not logits.
//!
//! The output is intended for a pooling layer (`pmetal_models::pooling::pool`)
//! to produce fixed-size sentence embeddings. The pooler head of
//! `BertModel` and every language-model, classification or QA head are not
//! part of this model; the loader names them as skipped.

use pmetal_runtime::compat::ops::arange;
use pmetal_runtime::compat::{Array, Dtype, Exception, nn, ops};
use pmetal_runtime::impl_module_params;
use pmetal_runtime::nn::{AttentionMaskType, FusedAttentionConfig, fused_sdpa};
use serde::{Deserialize, Serialize};

use crate::architectures::utils::{Activation, additive_padding_mask, resolve_activation};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which `transformers` family a checkpoint is, from its `model_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderFamily {
    Bert,
    Roberta,
    DistilBert,
}

impl EncoderFamily {
    /// The family a `config.json`'s `model_type` names, if this model can run
    /// it.
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "bert" => Some(Self::Bert),
            "roberta" | "xlm-roberta" | "xlm_roberta" | "camembert" => Some(Self::Roberta),
            "distilbert" => Some(Self::DistilBert),
            _ => None,
        }
    }

    /// The prefix the family's task models put the encoder under
    /// (`BertForMaskedLM`: `bert.`); a bare `*Model` checkpoint has none.
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Bert => "bert.",
            Self::Roberta => "roberta.",
            Self::DistilBert => "distilbert.",
        }
    }
}

/// BERT / RoBERTa / DistilBERT configuration.
///
/// Fields correspond to the standard HuggingFace `config.json` keys;
/// DistilBERT's spellings are accepted as aliases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Dimensionality of the encoder layers.
    #[serde(alias = "dim")]
    pub hidden_size: usize,
    /// Number of transformer encoder layers.
    #[serde(alias = "n_layers")]
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    #[serde(alias = "n_heads")]
    pub num_attention_heads: usize,
    /// Dimensionality of the "intermediate" (feed-forward) layer.
    #[serde(alias = "hidden_dim")]
    pub intermediate_size: usize,
    /// Rows of the position table. RoBERTa's includes the `pad_token_id + 1`
    /// it offsets by (514 for a 512-token model).
    #[serde(default = "bert_default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// Number of token-type (segment) embedding types. Ignored (none) for
    /// DistilBERT.
    #[serde(default = "bert_default_type_vocab_size")]
    pub type_vocab_size: usize,
    /// Epsilon for LayerNorm layers. DistilBERT fixes it at 1e-12.
    #[serde(default = "bert_default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// Activation function for the feed-forward intermediate layer.
    #[serde(default = "bert_default_hidden_act", alias = "activation")]
    pub hidden_act: String,
    /// The padding token's id. RoBERTa counts positions from it; unset, it is
    /// 1 for RoBERTa (`<pad>`) and 0 otherwise.
    #[serde(default)]
    pub pad_token_id: Option<usize>,
    /// Model type string (used for architecture detection).
    #[serde(default)]
    pub model_type: Option<String>,
    /// DistilBERT's fixed sinusoidal positions in place of a learned table;
    /// not implemented.
    #[serde(default)]
    pub sinusoidal_pos_embds: bool,
    /// `"absolute"` (the only kind implemented) or one of the relative
    /// schemes (`"relative_key"`, `"relative_key_query"`).
    #[serde(default)]
    pub position_embedding_type: Option<String>,
}

fn bert_default_max_position_embeddings() -> usize {
    512
}
fn bert_default_type_vocab_size() -> usize {
    2
}
fn bert_default_layer_norm_eps() -> f32 {
    1e-12
}
fn bert_default_hidden_act() -> String {
    "gelu".to_string()
}

impl BertConfig {
    /// Compute head dimension from hidden size and number of heads.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// The family the config's `model_type` names; BERT when it names none.
    pub fn family(&self) -> Result<EncoderFamily, Exception> {
        match self.model_type.as_deref() {
            None => Ok(EncoderFamily::Bert),
            Some(model_type) => EncoderFamily::from_model_type(model_type).ok_or_else(|| {
                Exception::custom(format!(
                    "bert: model_type {model_type:?} is not a BERT, RoBERTa or DistilBERT encoder"
                ))
            }),
        }
    }

    /// The padding token's id.
    pub fn padding_idx(&self) -> usize {
        self.pad_token_id.unwrap_or(match self.family() {
            Ok(EncoderFamily::Roberta) => 1,
            _ => 0,
        })
    }

    /// Rows of the token-type table; DistilBERT has none.
    pub fn token_types(&self) -> usize {
        match self.family() {
            Ok(EncoderFamily::DistilBert) => 0,
            _ => self.type_vocab_size,
        }
    }

    /// The LayerNorm epsilon (DistilBERT hard-codes 1e-12).
    pub fn norm_eps(&self) -> f32 {
        match self.family() {
            Ok(EncoderFamily::DistilBert) => 1e-12,
            _ => self.layer_norm_eps,
        }
    }

    /// How many positions the model can read: the table's rows, less the
    /// `pad_token_id + 1` RoBERTa offsets by.
    pub fn max_sequence_length(&self) -> usize {
        match self.family() {
            Ok(EncoderFamily::Roberta) => self
                .max_position_embeddings
                .saturating_sub(self.padding_idx() + 1),
            _ => self.max_position_embeddings,
        }
    }

    fn validate(&self) -> Result<EncoderFamily, Exception> {
        let family = self.family()?;
        if self.sinusoidal_pos_embds {
            return Err(Exception::custom(
                "bert: sinusoidal position embeddings (DistilBERT `sinusoidal_pos_embds`) are not implemented",
            ));
        }
        if let Some(kind) = self
            .position_embedding_type
            .as_deref()
            .filter(|kind| *kind != "absolute")
        {
            return Err(Exception::custom(format!(
                "bert: position_embedding_type {kind:?} is not implemented; only \"absolute\" is"
            )));
        }
        if self.num_attention_heads == 0 || self.hidden_size % self.num_attention_heads != 0 {
            return Err(Exception::custom(format!(
                "bert: hidden size {} is not a multiple of {} heads",
                self.hidden_size, self.num_attention_heads
            )));
        }
        Ok(family)
    }
}

impl Default for BertConfig {
    /// BERT-base defaults (110M parameters).
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".to_string(),
            pad_token_id: None,
            model_type: Some("bert".to_string()),
            sinusoidal_pos_embds: false,
            position_embedding_type: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// BERT embedding layer (token + position + token-type).
///
/// The three embeddings are summed and passed through one LayerNorm.
#[derive(Debug)]
pub struct BertEmbeddings {
    pub word_embeddings: nn::Embedding,
    pub position_embeddings: nn::Embedding,
    /// Absent for DistilBERT.
    pub token_type_embeddings: Option<nn::Embedding>,
    pub layer_norm: nn::LayerNorm,
    /// RoBERTa's padding index, from which its positions are counted; `None`
    /// counts them from 0.
    position_padding_idx: Option<i32>,
    /// Rows of `position_embeddings`.
    max_positions: i32,
}
impl_module_params!(BertEmbeddings; word_embeddings, position_embeddings, token_type_embeddings, layer_norm);

impl BertEmbeddings {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let family = config.validate()?;
        let h = config.hidden_size as i32;
        let token_types = config.token_types();
        Ok(Self {
            word_embeddings: nn::Embedding::new(config.vocab_size as i32, h)?,
            position_embeddings: nn::Embedding::new(config.max_position_embeddings as i32, h)?,
            token_type_embeddings: (token_types > 0)
                .then(|| nn::Embedding::new(token_types as i32, h))
                .transpose()?,
            layer_norm: nn::LayerNormBuilder::new(h)
                .eps(config.norm_eps())
                .build()?,
            position_padding_idx: (family == EncoderFamily::Roberta)
                .then(|| config.padding_idx() as i32),
            max_positions: config.max_position_embeddings as i32,
        })
    }

    /// The position of each token, `[batch, seq_len]`.
    ///
    /// BERT counts `0, 1, 2, ...` from the start of the row, padding included.
    /// RoBERTa counts only the tokens that are not padding, from
    /// `padding_idx + 1`, and leaves padding at `padding_idx`
    /// (`create_position_ids_from_input_ids`): a left-padded row's first real
    /// token is position `padding_idx + 1` wherever it sits.
    fn position_ids(&self, input_ids: &Array) -> Array {
        let seq_len = input_ids.dim(1);
        match self.position_padding_idx {
            None => arange(seq_len, Dtype::Int32).reshape(&[1, seq_len]),
            Some(padding_idx) => {
                let real = input_ids
                    .as_dtype(Dtype::Int32.as_i32())
                    .not_equal(&Array::from_int(padding_idx))
                    .as_dtype(Dtype::Int32.as_i32());
                let counted = real.cumsum(1).multiply(&real);
                counted.add(&Array::from_int(padding_idx))
            }
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `input_ids` - `[batch, seq_len]` token IDs (i32)
    /// * `token_type_ids` - optional `[batch, seq_len]` segment IDs; defaults to all-zeros
    pub fn forward(
        &mut self,
        input_ids: &Array,
        token_type_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let (batch, seq_len) = (input_ids.dim(0), input_ids.dim(1));
        let reach = self.position_padding_idx.map_or(0, |p| p + 1) + seq_len;
        if reach > self.max_positions {
            return Err(Exception::custom(format!(
                "bert: {seq_len} tokens exceed the {} the position table holds",
                self.max_positions - self.position_padding_idx.map_or(0, |p| p + 1)
            )));
        }

        let mut embedded = self.word_embeddings.forward(input_ids);
        embedded = embedded.add(
            &self
                .position_embeddings
                .forward(&self.position_ids(input_ids)),
        );
        if let Some(table) = self.token_type_embeddings.as_mut() {
            let types = match token_type_ids {
                Some(types) => types.clone(),
                None => ops::zeros(&[batch, seq_len], Dtype::Int32),
            };
            embedded = embedded.add(&table.forward(&types));
        }
        Ok(self.layer_norm.forward(&embedded))
    }
}

// ---------------------------------------------------------------------------
// Self-attention
// ---------------------------------------------------------------------------

/// BERT multi-head self-attention.
#[derive(Debug)]
pub struct BertSelfAttention {
    pub query: nn::Linear,
    pub key: nn::Linear,
    pub value: nn::Linear,
    /// Number of attention heads.
    pub num_heads: i32,
    /// Dimension per head.
    pub head_dim: i32,
}
impl_module_params!(BertSelfAttention; query, key, value);

impl BertSelfAttention {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        config.validate()?;
        let h = config.hidden_size as i32;
        let linear = || nn::LinearBuilder::new(h, h).build();
        Ok(Self {
            query: linear()?,
            key: linear()?,
            value: linear()?,
            num_heads: config.num_attention_heads as i32,
            head_dim: config.head_dim() as i32,
        })
    }

    /// Forward pass (bidirectional — no causal mask).
    ///
    /// # Arguments
    /// * `x` - `[batch, seq_len, hidden_dim]`
    /// * `attention_mask` - optional additive mask broadcastable to
    ///   `[batch, heads, seq, seq]` (0 = attend, very negative = ignore), as
    ///   [`additive_padding_mask`] builds from a `[batch, seq]` padding mask
    pub fn forward(
        &mut self,
        x: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let (batch, seq_len) = (x.dim(0), x.dim(1));
        let (nh, hd) = (self.num_heads, self.head_dim);
        // Project and reshape to [batch, heads, seq, head_dim]
        let split = |t: Array| {
            t.reshape(&[batch, seq_len, nh, hd])
                .transpose_axes(&[0, 2, 1, 3])
        };
        let q = split(self.query.forward(x));
        let k = split(self.key.forward(x));
        let v = split(self.value.forward(x));

        let config = FusedAttentionConfig::new(nh, nh, hd).with_mask_type(AttentionMaskType::None);
        let out = fused_sdpa(&q, &k, &v, &config, attention_mask)?;
        Ok(out
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, nh * hd]))
    }
}

// ---------------------------------------------------------------------------
// Attention output + FFN sub-layers
// ---------------------------------------------------------------------------

/// BERT attention output: dense projection + residual + LayerNorm.
#[derive(Debug)]
pub struct BertSelfOutput {
    pub dense: nn::Linear,
    pub layer_norm: nn::LayerNorm,
}
impl_module_params!(BertSelfOutput; dense, layer_norm);

impl BertSelfOutput {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        Ok(Self {
            dense: nn::LinearBuilder::new(h, h).build()?,
            layer_norm: nn::LayerNormBuilder::new(h)
                .eps(config.norm_eps())
                .build()?,
        })
    }

    pub fn forward(&mut self, hidden: &Array, residual: &Array) -> Result<Array, Exception> {
        let out = self.dense.forward(hidden).add(residual);
        Ok(self.layer_norm.forward(&out))
    }
}

/// BERT intermediate (FFN first half): dense + activation.
#[derive(Debug)]
pub struct BertIntermediate {
    pub dense: nn::Linear,
    /// `config.hidden_act`, resolved once at construction. BERT's default is
    /// `"gelu"` — the *exact* erf definition, per `ACT2FN` — so this must not
    /// fall back to either fast approximation.
    pub act: Activation,
}
impl_module_params!(BertIntermediate; dense);

impl BertIntermediate {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let dense =
            nn::LinearBuilder::new(config.hidden_size as i32, config.intermediate_size as i32)
                .build()?;
        let act = resolve_activation(&config.hidden_act).ok_or_else(|| {
            Exception::custom(format!(
                "bert: unsupported hidden_act {:?}",
                config.hidden_act
            ))
        })?;
        Ok(Self { dense, act })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        Ok((self.act)(&self.dense.forward(x)))
    }
}

/// BERT FFN output: dense + residual + LayerNorm.
#[derive(Debug)]
pub struct BertOutput {
    pub dense: nn::Linear,
    pub layer_norm: nn::LayerNorm,
}
impl_module_params!(BertOutput; dense, layer_norm);

impl BertOutput {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let intermediate = config.intermediate_size as i32;
        Ok(Self {
            dense: nn::LinearBuilder::new(intermediate, h).build()?,
            layer_norm: nn::LayerNormBuilder::new(h)
                .eps(config.norm_eps())
                .build()?,
        })
    }

    pub fn forward(&mut self, hidden: &Array, residual: &Array) -> Result<Array, Exception> {
        let out = self.dense.forward(hidden).add(residual);
        Ok(self.layer_norm.forward(&out))
    }
}

// ---------------------------------------------------------------------------
// Encoder layer
// ---------------------------------------------------------------------------

/// Single BERT transformer encoder layer.
#[derive(Debug)]
pub struct BertLayer {
    pub attention: BertSelfAttention,
    pub attention_output: BertSelfOutput,
    pub intermediate: BertIntermediate,
    pub output: BertOutput,
}
impl_module_params!(BertLayer; attention, attention_output, intermediate, output);

impl BertLayer {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        Ok(Self {
            attention: BertSelfAttention::new(config)?,
            attention_output: BertSelfOutput::new(config)?,
            intermediate: BertIntermediate::new(config)?,
            output: BertOutput::new(config)?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Self-attention sub-layer
        let attn_out = self.attention.forward(x, attention_mask)?;
        let attn_out = self.attention_output.forward(&attn_out, x)?;

        // Feed-forward sub-layer
        let ff_out = self.intermediate.forward(&attn_out)?;
        self.output.forward(&ff_out, &attn_out)
    }
}

// ---------------------------------------------------------------------------
// Full encoder
// ---------------------------------------------------------------------------

/// A BERT, RoBERTa or DistilBERT encoder.
///
/// Returns the sequence of hidden states `[batch, seq_len, hidden_size]`.
/// No pooler or task head is included; use `pmetal_models::pooling` to
/// extract sentence embeddings.
#[derive(Debug)]
pub struct BertModel {
    pub embeddings: BertEmbeddings,
    pub layers: Vec<BertLayer>,
    /// Model configuration (non-trainable metadata).
    pub config: BertConfig,
}
impl_module_params!(BertModel; embeddings, layers);

impl BertModel {
    pub fn new(config: BertConfig) -> Result<Self, Exception> {
        let embeddings = BertEmbeddings::new(&config)?;
        let layers = (0..config.num_hidden_layers)
            .map(|_| BertLayer::new(&config))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embeddings,
            layers,
            config,
        })
    }

    /// Build from a `config.json` string.
    pub fn from_config_str(config_str: &str) -> Result<Self, Exception> {
        let config: BertConfig =
            serde_json::from_str(config_str).map_err(|e| Exception::custom(e.to_string()))?;
        Self::new(config)
    }

    /// Forward pass through the full encoder.
    ///
    /// # Arguments
    /// * `input_ids` - `[batch, seq_len]` i32 token IDs
    /// * `attention_mask` - optional `[batch, seq_len]` mask (non-zero = attend,
    ///   0 = pad); without one every token attends to every other
    /// * `token_type_ids` - optional `[batch, seq_len]` i32 segment IDs
    ///   (ignored by DistilBERT)
    ///
    /// # Returns
    /// `[batch, seq_len, hidden_size]` encoder hidden states, in the dtype of
    /// the weights.
    pub fn forward_with_token_types(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
        token_type_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        if let Some(mask) = attention_mask
            && (mask.ndim() != 2
                || mask.dim(0) != input_ids.dim(0)
                || mask.dim(1) != input_ids.dim(1))
        {
            return Err(Exception::custom(format!(
                "bert: the attention mask must be [batch, seq] like the ids {:?}, got {:?}",
                input_ids.shape(),
                mask.shape()
            )));
        }
        let mut hidden = self.embeddings.forward(input_ids, token_type_ids)?;
        let additive = attention_mask.map(|mask| additive_padding_mask(mask, hidden.dtype()));
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, additive.as_ref())?;
        }
        Ok(hidden)
    }

    /// [`forward_with_token_types`](Self::forward_with_token_types) with every
    /// token in segment 0.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_token_types(input_ids, attention_mask, None)
    }

    pub fn config(&self) -> &BertConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Module trait impl (for use as a standard MLX module)
// ---------------------------------------------------------------------------

impl pmetal_runtime::compat::Module<Array> for BertModel {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Array) -> Result<Self::Output, Self::Error> {
        BertModel::forward(self, &input, None)
    }

    fn training_mode(&mut self, _mode: bool) {}
}
