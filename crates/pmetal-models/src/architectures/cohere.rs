//! Cohere Command R architecture.
//!
//! Command R is a family of multilingual models designed for:
//! - Retrieval-Augmented Generation (RAG)
//! - Tool use and function calling
//! - Long context (128K tokens)
//!
//! Variants:
//! - **Command R**: 35B parameters
//! - **Command R+**: 104B parameters
//! - **Command A**: 111B parameters (2025)
use pmetal_runtime::compat::{Array, Exception, Module, nn};
use pmetal_runtime::decode::last_token_hidden;
use pmetal_runtime::impl_module_params;

use pmetal_cache::kv_cache::KVCache;
use pmetal_runtime::nn::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopePositions, rope_embedding},
};
use pmetal_runtime::rope::{RopeConfig, RotaryEmbedding};

use serde::{Deserialize, Serialize};

use crate::checkpointing::checkpointed_layer;
use crate::common::windowed_mask;
use crate::decoder_layer::{AttentionModule, MlpModule};
use crate::traits::ModelConfig;

/// Cohere model configuration: Command R / R+ (`model_type = "cohere"`) and
/// Command R7B / Command A (`"cohere2"`).
///
/// Read from a `config.json` through a file struct that fills what a
/// released config leaves out the way `transformers` does: `head_dim` is
/// `hidden_size / num_attention_heads`, `num_key_value_heads` defaults to the
/// attention heads, `rope_theta` comes from `rope_parameters` when that is how
/// the config spells it, and `tie_word_embeddings` is true.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "CohereConfigFile")]
pub struct CohereConfig {
    /// `"cohere"` or `"cohere2"`.
    pub model_type: String,
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    pub layer_norm_eps: f32,
    /// Output logit scaling factor. Cohere multiplies the LM-head logits by
    /// this (HF `CohereForCausalLM`:
    /// `out = out * logit_scale`). Command-R family ships `0.0625`.
    pub logit_scale: f32,
    pub tie_word_embeddings: bool,
    /// Command R+'s per-head LayerNorm on the queries and keys, after the
    /// projections and before RoPE (`use_qk_norm`).
    pub use_qk_norm: bool,
    /// Window of the sliding-attention layers of a Cohere2 model; `None` for
    /// Cohere, which attends globally everywhere.
    pub sliding_window: Option<i32>,
    /// Cohere2's attention type per layer, `"sliding_attention"` or
    /// `"full_attention"`; empty for Cohere.
    pub layer_types: Vec<String>,
    /// RoPE scaling, read by [`pmetal_runtime::rope`]; the Command-R
    /// releases ship none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<serde_json::Value>,
    /// transformers v5's spelling of the same, with `rope_theta` inside.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_parameters: Option<serde_json::Value>,
}

/// A Cohere `config.json` as released, before the defaults `transformers`
/// applies are resolved into a [`CohereConfig`].
#[derive(Debug, Deserialize)]
struct CohereConfigFile {
    #[serde(default = "default_model_type")]
    model_type: String,
    #[serde(default = "default_vocab_size")]
    vocab_size: i32,
    #[serde(default = "default_hidden_size")]
    hidden_size: i32,
    #[serde(default = "default_intermediate_size")]
    intermediate_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    num_hidden_layers: i32,
    #[serde(default = "default_num_attention_heads")]
    num_attention_heads: i32,
    #[serde(default)]
    num_key_value_heads: Option<i32>,
    #[serde(default)]
    head_dim: Option<i32>,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: i32,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default = "default_layer_norm_eps")]
    layer_norm_eps: f32,
    #[serde(default = "default_logit_scale")]
    logit_scale: f32,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    use_qk_norm: bool,
    #[serde(default)]
    sliding_window: Option<i32>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    /// Cohere2's older spelling of `layer_types`: every `n`-th layer attends
    /// globally.
    #[serde(default)]
    sliding_window_pattern: Option<i32>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
}

fn default_model_type() -> String {
    "cohere".to_string()
}
fn default_vocab_size() -> i32 {
    256_000
}
fn default_hidden_size() -> i32 {
    8192
}
fn default_intermediate_size() -> i32 {
    22_528
}
fn default_num_hidden_layers() -> i32 {
    40
}
fn default_num_attention_heads() -> i32 {
    64
}
fn default_max_position_embeddings() -> i32 {
    8192
}
fn default_layer_norm_eps() -> f32 {
    1e-5
}
fn default_logit_scale() -> f32 {
    0.0625
}
fn default_true() -> bool {
    true
}

/// `transformers`' `Cohere2Config` default window.
const COHERE2_SLIDING_WINDOW: i32 = 4096;
/// `transformers`' default of every how many Cohere2 layers attend globally.
const COHERE2_SLIDING_WINDOW_PATTERN: i32 = 4;

impl From<CohereConfigFile> for CohereConfig {
    fn from(file: CohereConfigFile) -> Self {
        let cohere2 = file.model_type == "cohere2";
        let rope_theta = file
            .rope_theta
            .or_else(|| {
                file.rope_parameters
                    .as_ref()
                    .and_then(|p| p.get("rope_theta"))
                    .and_then(serde_json::Value::as_f64)
                    .map(|t| t as f32)
            })
            .unwrap_or(10_000.0);
        let (sliding_window, layer_types) = if cohere2 {
            let pattern = file
                .sliding_window_pattern
                .unwrap_or(COHERE2_SLIDING_WINDOW_PATTERN);
            let layer_types = file.layer_types.unwrap_or_else(|| {
                (1..=file.num_hidden_layers)
                    .map(|i| {
                        if pattern > 0 && i % pattern == 0 {
                            "full_attention"
                        } else {
                            "sliding_attention"
                        }
                        .to_string()
                    })
                    .collect()
            });
            (
                Some(file.sliding_window.unwrap_or(COHERE2_SLIDING_WINDOW)),
                layer_types,
            )
        } else {
            (None, Vec::new())
        };
        Self {
            model_type: file.model_type,
            vocab_size: file.vocab_size,
            hidden_size: file.hidden_size,
            intermediate_size: file.intermediate_size,
            num_hidden_layers: file.num_hidden_layers,
            num_attention_heads: file.num_attention_heads,
            num_key_value_heads: file.num_key_value_heads.unwrap_or(file.num_attention_heads),
            head_dim: file
                .head_dim
                .unwrap_or(file.hidden_size / file.num_attention_heads.max(1)),
            max_position_embeddings: file.max_position_embeddings,
            rope_theta,
            layer_norm_eps: file.layer_norm_eps,
            logit_scale: file.logit_scale,
            tie_word_embeddings: file.tie_word_embeddings,
            use_qk_norm: file.use_qk_norm,
            sliding_window,
            layer_types,
            rope_scaling: file.rope_scaling,
            rope_parameters: file.rope_parameters,
        }
    }
}

impl CohereConfig {
    /// The interleaved rotary embedding, scaling included; an unknown
    /// `rope_type` is an error naming it.
    pub fn rotary(&self) -> Result<RotaryEmbedding, Exception> {
        crate::common::rotary_embedding(
            "cohere",
            self.head_dim,
            RopeConfig {
                rope_scaling: self.rope_scaling.as_ref(),
                rope_parameters: self.rope_parameters.as_ref(),
                rope_theta: Some(self.rope_theta as f64),
                max_position_embeddings: Some(self.max_position_embeddings as f64),
                ..RopeConfig::default()
            },
            1.0,
            true,
        )
    }

    /// Whether layer `layer_idx` attends within a sliding window (and rotates
    /// its queries and keys): Cohere2's `"sliding_attention"` layers. Its
    /// `"full_attention"` layers attend globally with no positional rotation
    /// at all; every layer of a Cohere model attends globally and rotates.
    pub fn is_local_layer(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .is_some_and(|t| t == "sliding_attention")
    }

    /// Whether layer `layer_idx` applies RoPE.
    pub fn uses_rope(&self, layer_idx: usize) -> bool {
        self.layer_types.is_empty() || self.is_local_layer(layer_idx)
    }

    /// Whether any layer attends within a window or skips RoPE, which the
    /// fused batched decode (one global-attention block shape) does not carry.
    pub fn has_cohere2_layers(&self) -> bool {
        !self.layer_types.is_empty()
    }
}

impl Default for CohereConfig {
    fn default() -> Self {
        // Default for Command R 35B
        Self {
            model_type: "cohere".to_string(),
            vocab_size: 256000,
            hidden_size: 8192,
            intermediate_size: 22528,
            num_hidden_layers: 40,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 131072,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            logit_scale: 0.0625,
            tie_word_embeddings: true,
            use_qk_norm: false,
            sliding_window: None,
            layer_types: Vec::new(),
            rope_scaling: None,
            rope_parameters: None,
        }
    }
}

impl ModelConfig for CohereConfig {
    fn model_type(&self) -> &str {
        &self.model_type
    }
    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }
    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }
    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }
    fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads
    }
    fn head_dim(&self) -> i32 {
        self.head_dim
    }
    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }
    fn max_position_embeddings(&self) -> i32 {
        self.max_position_embeddings
    }
    fn norm_eps(&self) -> f32 {
        self.layer_norm_eps
    }
    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
}

// =============================================================================
// Model Components
// =============================================================================

/// Cohere MLP layer.
#[derive(Debug)]
pub struct CohereMLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(CohereMLP; gate_proj, up_proj, down_proj);

impl CohereMLP {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(intermediate_size, hidden_size)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = Module::forward(&mut self.gate_proj, x)?;
        let gate = nn::silu(&gate);
        let up = Module::forward(&mut self.up_proj, x)?;
        let hidden = gate.multiply(&up);
        Module::forward(&mut self.down_proj, &hidden)
    }
}

/// Command R+'s query/key normalization: a LayerNorm over each head's
/// `head_dim` (mean subtracted, no bias) whose weight is per head,
/// `[heads, head_dim]`, as `transformers`' `CohereLayerNorm` with
/// `hidden_size = (heads, head_dim)` holds it.
#[derive(Debug)]
pub struct HeadLayerNorm {
    /// `[heads, head_dim]`.
    pub weight: Array,
    eps: f32,
}
impl_module_params!(HeadLayerNorm; weight);

impl HeadLayerNorm {
    fn new(heads: i32, head_dim: i32, eps: f32) -> Self {
        Self {
            weight: pmetal_runtime::compat::ops::ones(
                &[heads, head_dim],
                pmetal_runtime::compat::Dtype::Float32,
            ),
            eps,
        }
    }

    /// `[B, L, heads, head_dim]` normalized over the last axis, scaled by the
    /// head's weight, in the input's dtype. The scale is applied in `f32`
    /// before the cast back, as the reference does.
    fn forward(&self, x: &Array) -> Array {
        let f32 = pmetal_runtime::compat::Dtype::Float32.as_i32();
        let normed = x.layer_norm(None, None, self.eps).as_dtype(f32);
        normed
            .multiply(&self.weight.as_dtype(f32))
            .as_dtype(x.dtype().as_i32())
    }
}

/// Cohere attention: grouped-query, interleaved RoPE, optional per-head
/// query/key LayerNorm (Command R+), and for Cohere2 either a sliding window
/// with RoPE or global attention with none.
#[derive(Debug)]
pub struct CohereAttention {
    pub layer_idx: usize,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    /// Interleaved RoPE, scaling included.
    pub rotary: RotaryEmbedding,
    /// Whether queries and keys are rotated: every layer of a Cohere model,
    /// and the sliding layers of a Cohere2 one.
    pub use_rope: bool,
    /// The window of a sliding-attention layer; `None` attends globally.
    pub sliding_window: Option<i32>,

    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub q_norm: Option<HeadLayerNorm>,
    pub k_norm: Option<HeadLayerNorm>,
}
impl_module_params!(CohereAttention; q_proj, k_proj, v_proj, o_proj, q_norm, k_norm);

impl CohereAttention {
    pub fn new(config: &CohereConfig, layer_idx: usize) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;

        let q_proj = nn::LinearBuilder::new(hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, hidden_size)
            .bias(false)
            .build()?;
        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(HeadLayerNorm::new(n_heads, head_dim, config.layer_norm_eps)),
                Some(HeadLayerNorm::new(
                    n_kv_heads,
                    head_dim,
                    config.layer_norm_eps,
                )),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            rotary: config.rotary()?,
            use_rope: config.uses_rope(layer_idx),
            sliding_window: config
                .is_local_layer(layer_idx)
                .then_some(config.sliding_window)
                .flatten(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        _position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None, None)
    }

    /// Forward pass with optional KV cache.
    ///
    /// Mirrors the Llama-style serial cached path: project → reshape (→
    /// per-head norm) → **transpose** → `apply_rope` with axis -2 = seq →
    /// cache write → fused SDPA. A sliding layer restricts a caller's mask to
    /// its window.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let q = Module::forward(&mut self.q_proj, x)?;
        let k = Module::forward(&mut self.k_proj, x)?;
        let v = Module::forward(&mut self.v_proj, x)?;

        let mut q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        let mut k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            q = q_norm.forward(&q);
            k = k_norm.forward(&k);
        }

        // Transpose to [B, heads, seq, head_dim] BEFORE RoPE so axis -2 is
        // correctly the seq axis.
        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        // Cohere uses *traditional* (interleaved) RoPE — HF `CohereRotaryEmbedding`
        // rotates adjacent pairs, i.e. `nn.RoPE(..., traditional=True)`.
        // The split-half (`traditional=false`) form rotates the wrong element
        // pairs and silently corrupts every attention score.
        let (q, k) = if self.use_rope {
            let rope_positions = RopePositions::resolve(
                positions,
                cache
                    .as_ref()
                    .map_or(0, |(c, layer)| c.rope_offset_for(*layer)),
            );
            (
                rope_embedding(&q, rope_positions, &self.rotary),
                rope_embedding(&k, rope_positions, &self.rotary),
            )
        } else {
            (q, k)
        };

        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        // A caller's mask (packed sequences' block-diagonal one) says which
        // sequence a token belongs to, not how far back a sliding layer reaches.
        let windowed = windowed_mask(mask, self.sliding_window, seq_len, k.dim(2))?;
        let mask = windowed.as_ref().or(mask);

        let mask_type = match (mask.is_some(), self.sliding_window) {
            (true, _) => AttentionMaskType::None,
            (false, Some(window)) => AttentionMaskType::SlidingWindow(window),
            (false, None) => AttentionMaskType::Causal,
        };
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(mask_type);
        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);
        Module::forward(&mut self.o_proj, &output)
    }
}

impl AttentionModule for CohereAttention {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        CohereAttention::forward_with_cache(self, x, mask, cache, positions)
    }
}

impl MlpModule for CohereMLP {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        CohereMLP::forward(self, x)
    }
}

/// Cohere decoder layer.
#[derive(Debug)]
pub struct CohereDecoderLayer {
    pub layer_idx: usize,

    pub self_attn: CohereAttention,
    pub mlp: CohereMLP,
    pub input_layernorm: nn::LayerNorm,
}
impl_module_params!(CohereDecoderLayer; self_attn, mlp, input_layernorm);

impl CohereDecoderLayer {
    pub fn new(config: &CohereConfig, layer_idx: usize) -> Result<Self, Exception> {
        let self_attn = CohereAttention::new(config, layer_idx)?;
        let mlp = CohereMLP::new(config.hidden_size, config.intermediate_size)?;

        // Cohere uses LayerNorm (not RMSNorm)
        // Scale only: no Cohere release has a norm bias, so none exists to be
        // trained or saved.
        let input_layernorm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .bias(false)
            .build()?;

        Ok(Self {
            layer_idx,
            self_attn,
            mlp,
            input_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let _ = position_ids;
        self.forward_with_cache(x, mask, None, None)
    }

    /// Cohere parallel decoder block with optional KV cache.
    ///
    /// Layout: `x + attn(norm(x)) + ffn(norm(x))` — both branches consume
    /// the *same* normed input. There is no `post_attention_layernorm`.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self
            .self_attn
            .forward_with_cache(&normed, mask, cache, positions)?;
        let ffn_out = self.mlp.forward(&normed)?;
        Ok(x.add(&attn_out).add(&ffn_out))
    }
}

/// Cohere model.
#[derive(Debug)]
pub struct CohereModel {
    pub config: CohereConfig,

    pub embed_tokens: nn::Embedding,
    pub layers: Vec<CohereDecoderLayer>,
    pub norm: nn::LayerNorm,
    /// Recompute each layer's activations during the backward pass instead of
    /// holding them. Training only; see `crate::checkpointing`.
    pub grad_checkpoint: bool,
}
impl_module_params!(CohereModel; embed_tokens, layers, norm);

impl CohereModel {
    pub fn new(config: CohereConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| CohereDecoderLayer::new(&config, i as usize))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .bias(false)
            .build()?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            grad_checkpoint: false,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_inner(input_ids, mask, None, position_ids)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        self.forward_inner(input_ids, mask, cache, None)
    }

    /// Forward pass with one rotary position per token, `[seq_len]`.
    ///
    /// Packed training concatenates several sequences into one row, so the
    /// positions have to restart at each boundary instead of running through
    /// it. The block-diagonal mask keeps the content apart; this keeps the
    /// positions apart.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_inner(input_ids, mask, None, positions)
    }

    /// The layer loop, with every per-call input the layers can take.
    ///
    /// `positions` carries one rotary position per token, `[seq_len]`, which
    /// is what a packed batch needs so the second sequence in a row restarts
    /// at 0 rather than continuing the first.
    fn forward_inner(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        // Hoisted: the loop below borrows `self.layers` mutably.
        let grad_checkpoint = self.grad_checkpoint;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let c = cache.as_deref_mut().map(|c| (c, idx));
            // A cache means generation, which has no backward pass for the
            // recompute to pay for.
            hidden_states = if grad_checkpoint && c.is_none() {
                checkpointed_layer(
                    layer,
                    &hidden_states,
                    mask,
                    positions,
                    |layer, h, mask, positions| layer.forward_with_cache(h, mask, None, positions),
                )?
            } else {
                layer.forward_with_cache(&hidden_states, mask, c, positions)?
            };
        }

        Module::forward(&mut self.norm, &hidden_states)
    }
}

/// Cohere for causal language modeling.
#[derive(Debug)]
pub struct CohereForCausalLM {
    pub config: CohereConfig,

    pub model: CohereModel,
    /// None when `tie_word_embeddings` is true — Cohere ships tied weights,
    /// so the LM head reuses `embed_tokens` transposed. A separate `lm_head`
    /// would never load (no `lm_head.weight` in the checkpoint) and emit
    /// random logits.
    pub lm_head: Option<nn::Linear>,
}
impl_module_params!(CohereForCausalLM; model, lm_head);

impl CohereForCausalLM {
    /// Whether RoPE is a scalar base and position scale, which the fused
    /// batched decode path carries.
    pub fn has_scalar_rope(&self) -> bool {
        self.model
            .layers
            .first()
            .is_some_and(|layer| layer.self_attn.rotary.scalar().is_some())
    }

    pub fn new(config: CohereConfig) -> Result<Self, Exception> {
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        };

        let model = CohereModel::new(config.clone())?;

        Ok(Self {
            config,
            model,
            lm_head,
        })
    }

    /// Apply the LM head (tied or untied) and Cohere's output `logit_scale`.
    fn lm_head_forward(&mut self, h: &Array) -> Result<Array, Exception> {
        let logits = if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, h)?
        } else {
            self.model.embed_tokens.as_linear(h)
        };
        Ok(logits.mul_scalar(self.config.logit_scale))
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward(input_ids, mask, position_ids)?;
        self.lm_head_forward(&hidden_states)
    }

    /// Forward pass with optional KV cache for incremental decoding.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward_with_cache(input_ids, mask, cache)?;
        self.lm_head_forward(&hidden_states)
    }

    /// [`forward_with_cache`](Self::forward_with_cache) returning the last
    /// position's logits, `[B, 1, vocab]`: the LM head runs on one hidden state
    /// per sequence instead of every prompt token's, for a caller that samples
    /// from the last position alone (a prefill, a decode step).
    pub fn forward_with_cache_last(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward_with_cache(input_ids, mask, cache)?;
        self.lm_head_forward(&last_token_hidden(&hidden_states))
    }

    /// Forward pass with one rotary position per token; see
    /// `forward_with_positions` on the inner model.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden_states = self
            .model
            .forward_with_positions(input_ids, mask, positions)?;
        self.lm_head_forward(&hidden_states)
    }

    /// Create a fresh KV cache sized for this model.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        use pmetal_cache::kv_cache::KVCacheConfig;
        KVCache::new(KVCacheConfig::new(
            self.config.num_hidden_layers as usize,
            max_seq_len,
            self.config.num_key_value_heads as usize,
            self.config.head_dim as usize,
        ))
    }

    /// Fused batched-decode forward.
    ///
    /// Cohere's parallel decoder block — `x + attn(norm(x)) + ffn(norm(x))` —
    /// has no `post_attention_layernorm`, so it can't reuse
    /// `batched_prenorm_layer`. Routes through
    /// [`crate::common::batched_parallel_block`] instead.
    ///
    /// Sliding-window Cohere configs (`use_sliding_window = true` with
    /// non-global layers) take the serial fallback — gated by
    /// [`crate::dispatcher::DynamicModel::supports_fused_batched`].
    pub fn forward_batched_impl(
        &mut self,
        input_ids: &Array,
        active_indices: &[usize],
        cache: &mut pmetal_cache::kv_cache::FusedBatchKVCache,
    ) -> Result<Array, Exception> {
        use crate::common::{BatchedGqaAttnCfg, batched_parallel_block};

        let cfg = &self.config;
        let attn_cfg = BatchedGqaAttnCfg::for_rotary(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            &self.model.layers[0].self_attn.rotary,
        )?;

        let mut hidden = Module::forward(&mut self.model.embed_tokens, input_ids)?;
        for (layer_idx, layer) in self.model.layers.iter_mut().enumerate() {
            hidden = batched_parallel_block(
                &hidden,
                &mut layer.input_layernorm,
                &mut layer.self_attn.q_proj,
                &mut layer.self_attn.k_proj,
                &mut layer.self_attn.v_proj,
                &mut layer.self_attn.o_proj,
                None,
                None,
                &mut layer.mlp,
                &attn_cfg,
                cache,
                active_indices,
                layer_idx,
            )?;
        }
        let hidden = Module::forward(&mut self.model.norm, &hidden)?;
        self.lm_head_forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_runtime::compat::ModuleParametersExt;
    use serial_test::serial;

    /// A released Command-R config states no `head_dim` and, in transformers 5's
    /// spelling, no top-level `rope_theta`: the reference derives and defaults.
    #[test]
    fn a_released_command_r_config_parses_with_the_references_defaults() {
        let config: CohereConfig = json5::from_str(
            r#"{
                "model_type": "cohere",
                "vocab_size": 256000, "hidden_size": 8192, "intermediate_size": 22528,
                "num_hidden_layers": 40, "num_attention_heads": 64,
                "num_key_value_heads": 64, "max_position_embeddings": 8192,
                "rope_theta": 8000000.0, "layer_norm_eps": 1e-05, "logit_scale": 0.0625,
                "use_qk_norm": false, "tie_word_embeddings": true
            }"#,
        )
        .unwrap();
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.rope_theta, 8_000_000.0);
        assert!(config.layer_types.is_empty() && config.sliding_window.is_none());
        assert!(config.uses_rope(7) && !config.is_local_layer(7));

        let v5: CohereConfig = json5::from_str(
            r#"{"model_type": "cohere", "hidden_size": 64, "num_attention_heads": 4,
                "use_qk_norm": true,
                "rope_parameters": {"rope_type": "default", "rope_theta": 75000000.0}}"#,
        )
        .unwrap();
        assert_eq!((v5.head_dim, v5.num_key_value_heads), (16, 4));
        assert_eq!(v5.rope_theta, 75_000_000.0);
        assert!(v5.use_qk_norm && v5.tie_word_embeddings);
    }

    /// Cohere2 sliding-attention layers rotate and attend in a window; every
    /// fourth layer by default (or as `layer_types` says) attends globally
    /// with no RoPE.
    #[test]
    fn cohere2_layer_types_come_from_the_config_or_the_pattern() {
        let config: CohereConfig = json5::from_str(
            r#"{"model_type": "cohere2", "num_hidden_layers": 8, "hidden_size": 64,
                "num_attention_heads": 4}"#,
        )
        .unwrap();
        assert_eq!(config.sliding_window, Some(4096));
        let local: Vec<bool> = (0..8).map(|i| config.is_local_layer(i)).collect();
        assert_eq!(
            local,
            [true, true, true, false, true, true, true, false],
            "every 4th layer is global"
        );
        assert!(config.uses_rope(0) && !config.uses_rope(3));

        let listed: CohereConfig = json5::from_str(
            r#"{"model_type": "cohere2", "num_hidden_layers": 3, "hidden_size": 64,
                "num_attention_heads": 4, "sliding_window": 128,
                "layer_types": ["full_attention", "sliding_attention", "sliding_attention"]}"#,
        )
        .unwrap();
        assert_eq!(listed.sliding_window, Some(128));
        assert!(!listed.is_local_layer(0) && listed.is_local_layer(1));
    }

    #[test]
    #[serial]
    fn test_cohere_mlp() {
        let mlp = CohereMLP::new(64, 256).unwrap();
        let x = pmetal_runtime::compat::random::normal(
            &[1, 10, 64],
            pmetal_runtime::compat::Dtype::Float32,
        );

        let mut mlp = mlp;
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[1, 10, 64]);
    }

    #[test]
    #[serial]
    fn test_cohere_model_instantiation() {
        let config = CohereConfig {
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            vocab_size: 1000,
            ..Default::default()
        };

        let model = CohereForCausalLM::new(config).unwrap();

        let params = model.flatten_params();
        assert!(!params.is_empty());
    }
}
