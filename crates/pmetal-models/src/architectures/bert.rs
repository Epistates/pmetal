//! BERT encoder architecture for embedding / sentence-transformer training.
//!
//! Supports BERT (`model_type: "bert"`) and checkpoints that share its config
//! and key layout.
//!
//! It does **not** support RoBERTa or DistilBERT, which this header used to
//! claim. DistilBERT's released config names none of the fields below — it
//! spells them `dim`, `hidden_dim`, `n_layers`, `n_heads` — and RoBERTa
//! prefixes its weights `roberta.` and offsets its position ids past the
//! padding index. Both were routed here and both failed; the dispatcher now
//! rejects them up front.
//!
//! Unlike causal LM architectures this model:
//! - Uses **bidirectional** self-attention (no causal mask).
//! - Has **no KV cache** (encoder-only, full context every forward pass).
//! - Returns **hidden states** `[batch, seq, hidden_dim]` instead of logits.
//! - Uses absolute position embeddings + token-type embeddings.
//!
//! The output is intended to be passed to a pooling layer
//! (`pmetal_models::pooling::pool`) to produce fixed-size sentence embeddings.

use pmetal_bridge::compat::{Array, Exception, Module, ModuleParameters, nn, ops};
use pmetal_bridge::impl_module_params;
use serde::{Deserialize, Serialize};

use crate::architectures::utils::{Activation, resolve_activation};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// BERT / RoBERTa model configuration.
///
/// Fields correspond to the standard HuggingFace `config.json` keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Dimensionality of the encoder layers and the pooler.
    pub hidden_size: usize,
    /// Number of transformer encoder layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Dimensionality of the "intermediate" (feed-forward) layer.
    pub intermediate_size: usize,
    /// Maximum sequence length supported by the position embeddings.
    #[serde(default = "bert_default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// Number of token-type (segment) embedding types.
    #[serde(default = "bert_default_type_vocab_size")]
    pub type_vocab_size: usize,
    /// Epsilon for LayerNorm layers.
    #[serde(default = "bert_default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// Activation function for the feed-forward intermediate layer.
    #[serde(default = "bert_default_hidden_act")]
    pub hidden_act: String,
    /// Model type string (used for architecture detection).
    #[serde(default)]
    pub model_type: Option<String>,
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
            model_type: Some("bert".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// BERT embedding layer (token + position + token-type).
///
/// Pre-LayerNorm variant used by BERT: the three embeddings are summed and
/// then passed through a single LayerNorm.
#[derive(Debug)]
pub struct BertEmbeddings {
    pub word_embeddings: nn::Embedding,
    pub position_embeddings: nn::Embedding,
    pub token_type_embeddings: nn::Embedding,
    pub layer_norm: nn::LayerNorm,
    /// Cached max position count (for position-ID generation).
    pub max_position_embeddings: i32,
}
impl_module_params!(BertEmbeddings; word_embeddings, position_embeddings, token_type_embeddings, layer_norm);

impl BertEmbeddings {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let word_embeddings = nn::Embedding::new(config.vocab_size as i32, h)?;
        let position_embeddings = nn::Embedding::new(config.max_position_embeddings as i32, h)?;
        let token_type_embeddings = nn::Embedding::new(config.type_vocab_size as i32, h)?;
        let layer_norm = nn::LayerNormBuilder::new(h)
            .eps(config.layer_norm_eps)
            .build()?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            max_position_embeddings: config.max_position_embeddings as i32,
        })
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
        let batch = input_ids.dim(0);
        let seq_len = input_ids.dim(1);

        let word_emb = Module::forward(&mut self.word_embeddings, input_ids)?;

        // Position IDs: [0, 1, ..., seq_len-1], broadcast over batch
        let pos_ids: Vec<i32> = (0..seq_len).collect();
        let pos_ids_arr = Array::from_slice(&pos_ids, &[1, seq_len]);
        let pos_emb = Module::forward(&mut self.position_embeddings, &pos_ids_arr)?;

        // Token type IDs (segment IDs) — default to zeros
        let type_emb = if let Some(tids) = token_type_ids {
            Module::forward(&mut self.token_type_embeddings, tids)?
        } else {
            let zeros =
                Array::from_slice(&vec![0i32; (batch * seq_len) as usize], &[batch, seq_len]);
            Module::forward(&mut self.token_type_embeddings, &zeros)?
        };

        let combined = word_emb.add(&pos_emb).add(&type_emb);
        Module::forward(&mut self.layer_norm, &combined)
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
    /// Softmax scale = 1/sqrt(head_dim).
    pub scale: f32,
}
impl_module_params!(BertSelfAttention; query, key, value);

impl BertSelfAttention {
    pub fn new(config: &BertConfig) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let num_heads = config.num_attention_heads as i32;
        let head_dim = (config.hidden_size / config.num_attention_heads) as i32;
        let scale = (head_dim as f32).sqrt().recip();

        // BERT uses bias in Q/K/V projections
        let query = nn::LinearBuilder::new(h, h).build()?;
        let key = nn::LinearBuilder::new(h, h).build()?;
        let value = nn::LinearBuilder::new(h, h).build()?;

        Ok(Self {
            query,
            key,
            value,
            num_heads,
            head_dim,
            scale,
        })
    }

    /// Forward pass (bidirectional — no causal mask).
    ///
    /// # Arguments
    /// * `x` - `[batch, seq_len, hidden_dim]`
    /// * `attention_mask` - optional `[batch, seq_len]` additive mask
    ///   (0 = attend, large negative = ignore)
    pub fn forward(
        &mut self,
        x: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let batch = x.dim(0);
        let seq_len = x.dim(1);
        let nh = self.num_heads;
        let hd = self.head_dim;

        // Project and reshape to [batch, heads, seq, head_dim]
        let q = Module::forward(&mut self.query, x)?
            .reshape(&[batch, seq_len, nh, hd])
            .transpose_axes(&[0, 2, 1, 3]);
        let k = Module::forward(&mut self.key, x)?
            .reshape(&[batch, seq_len, nh, hd])
            .transpose_axes(&[0, 2, 1, 3]);
        let v = Module::forward(&mut self.value, x)?
            .reshape(&[batch, seq_len, nh, hd])
            .transpose_axes(&[0, 2, 1, 3]);

        // Scaled dot-product: [batch, heads, seq, seq]
        let scores = q
            .matmul(&k.transpose_axes(&[0, 1, 3, 2]))
            .divide(&Array::from_f32(self.scale));

        // Apply attention mask: [batch, seq] → [batch, 1, 1, seq] additive bias
        let scores = if let Some(mask) = attention_mask {
            let mask_f = mask.as_dtype(scores.dtype().as_i32());
            // 1 = keep, 0 = ignore → convert to additive: (1 - mask) * -1e9
            let inv_mask = Array::from_f32(1.0).subtract(&mask_f);
            let additive = inv_mask
                .reshape(&[batch, 1, 1, seq_len])
                .multiply(&Array::from_f32(-1e9));
            scores.add(&additive)
        } else {
            scores
        };

        let attn_weights = ops::softmax_axis(&scores, -1);

        // Weighted sum: [batch, heads, seq, head_dim] → [batch, seq, hidden]
        let out = attn_weights
            .matmul(&v)
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, nh * hd]);

        Ok(out)
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
                .eps(config.layer_norm_eps)
                .build()?,
        })
    }

    pub fn forward(&mut self, hidden: &Array, residual: &Array) -> Result<Array, Exception> {
        let out = Module::forward(&mut self.dense, hidden)?;
        let out = out.add(residual);
        Module::forward(&mut self.layer_norm, &out)
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
        let h = Module::forward(&mut self.dense, x)?;
        Ok((self.act)(&h))
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
                .eps(config.layer_norm_eps)
                .build()?,
        })
    }

    pub fn forward(&mut self, hidden: &Array, residual: &Array) -> Result<Array, Exception> {
        let out = Module::forward(&mut self.dense, hidden)?;
        let out = out.add(residual);
        Module::forward(&mut self.layer_norm, &out)
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
// Full BERT encoder
// ---------------------------------------------------------------------------

/// Full BERT / RoBERTa encoder.
///
/// Returns the sequence of hidden states `[batch, seq_len, hidden_size]`.
/// No pooler or classification head is included; use `pmetal_models::pooling`
/// to extract sentence embeddings.
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
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(&config)?);
        }
        Ok(Self {
            embeddings: BertEmbeddings::new(&config)?,
            layers,
            config,
        })
    }

    /// Forward pass through the full encoder.
    ///
    /// # Arguments
    /// * `input_ids` - `[batch, seq_len]` i32 token IDs
    /// * `attention_mask` - optional `[batch, seq_len]` i32 mask (1=attend, 0=pad)
    /// * `token_type_ids` - optional `[batch, seq_len]` i32 segment IDs
    ///
    /// # Returns
    /// `[batch, seq_len, hidden_size]` encoder hidden states
    pub fn forward(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
        token_type_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut hidden = self.embeddings.forward(input_ids, token_type_ids)?;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, attention_mask)?;
        }
        Ok(hidden)
    }

    pub fn config(&self) -> &BertConfig {
        &self.config
    }
}

/// BERT configured for sentence embedding (no classification head).
///
/// Wraps `BertModel` and exposes a `forward` that returns encoder hidden
/// states `[batch, seq_len, hidden_size]` ready for pooling.
#[derive(Debug)]
pub struct BertForEmbedding {
    pub model: BertModel,
}
impl_module_params!(BertForEmbedding; model);

impl BertForEmbedding {
    pub fn new(config: BertConfig) -> Result<Self, Exception> {
        Ok(Self {
            model: BertModel::new(config)?,
        })
    }

    /// Load configuration from a `config.json` string.
    pub fn from_config_str(config_str: &str) -> Result<Self, Exception> {
        let config: BertConfig =
            serde_json::from_str(config_str).map_err(|e| Exception::custom(e.to_string()))?;
        Self::new(config)
    }

    /// Returns hidden states `[batch, seq_len, hidden_size]`.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.model.forward(input_ids, attention_mask, None)
    }

    pub fn config(&self) -> &BertConfig {
        self.model.config()
    }
}

// ---------------------------------------------------------------------------
// Module trait impl (for use as a standard MLX module)
// ---------------------------------------------------------------------------

impl Module<Array> for BertForEmbedding {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Array) -> Result<Self::Output, Self::Error> {
        BertForEmbedding::forward(self, &input, None)
    }

    fn training_mode(&mut self, _mode: bool) {}
}
