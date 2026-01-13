//! Gemma and Gemma2 model architectures.
//!
//! Supports Gemma 2B, 7B and Gemma2 2B, 9B, 27B variants.
//! Key differences from Llama:
//! - GemmaRMSNorm: output = x * (1 + weight) instead of x * weight
//! - GeGLU instead of SwiGLU (uses GELU instead of SiLU)
//! - Embedding scaling by sqrt(hidden_size)
//! - Gemma2: Attention logit softcapping, sliding window, extra normalization

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn, Array,
};
use pmetal_mlx::kernels::{fused_sdpa, rope::apply_rope, AttentionMaskType, FusedAttentionConfig};
use pmetal_mlx::kv_cache::KVCache;
use serde::{Deserialize, Serialize};

use crate::traits::{CausalLMModel, ModelConfig};
use std::collections::HashMap;

/// Gemma model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmaConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Hidden dimension.
    pub hidden_size: i32,
    /// Intermediate size (MLP).
    pub intermediate_size: i32,
    /// Number of hidden layers.
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<i32>,
    /// Head dimension.
    #[serde(default)]
    pub head_dim: Option<i32>,
    /// Maximum sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// RMS normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Hidden activation function.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Attention logit softcapping (Gemma2 only).
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    /// Query pre-attention scalar (Gemma2 only).
    #[serde(default)]
    pub query_pre_attn_scalar: Option<i32>,
    /// Sliding window size for local attention (Gemma2 only).
    #[serde(default)]
    pub sliding_window: Option<i32>,
    /// Whether this is a Gemma2 model.
    #[serde(default)]
    pub is_gemma2: bool,
}

fn default_model_type() -> String {
    "gemma".to_string()
}
fn default_vocab_size() -> i32 {
    256000
}
fn default_max_position_embeddings() -> i32 {
    8192
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_hidden_act() -> String {
    "gelu".to_string()
}

impl GemmaConfig {
    /// Get the number of KV heads (defaults to num_attention_heads if not specified).
    pub fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the head dimension.
    pub fn get_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Get the embedding scaling factor.
    pub fn embedding_scale(&self) -> f32 {
        (self.hidden_size as f32).sqrt()
    }

    /// Get attention scale, considering query_pre_attn_scalar for Gemma2.
    pub fn attention_scale(&self) -> f32 {
        if let Some(scalar) = self.query_pre_attn_scalar {
            (scalar as f32).sqrt().recip()
        } else {
            (self.get_head_dim() as f32).sqrt().recip()
        }
    }
}

impl Default for GemmaConfig {
    fn default() -> Self {
        // Gemma 2B defaults
        Self {
            model_type: "gemma".to_string(),
            vocab_size: 256000,
            hidden_size: 2048,
            intermediate_size: 16384,
            num_hidden_layers: 18,
            num_attention_heads: 8,
            num_key_value_heads: Some(1),
            head_dim: Some(256),
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            hidden_act: "gelu".to_string(),
            attn_logit_softcapping: None,
            query_pre_attn_scalar: None,
            sliding_window: None,
            is_gemma2: false,
        }
    }
}

impl GemmaConfig {
    /// Create Gemma 2B configuration.
    pub fn gemma_2b() -> Self {
        Self::default()
    }

    /// Create Gemma 7B configuration.
    pub fn gemma_7b() -> Self {
        Self {
            hidden_size: 3072,
            intermediate_size: 24576,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: Some(16), // MHA for 7B
            head_dim: Some(256),
            ..Default::default()
        }
    }

    /// Create Gemma2 2B configuration.
    pub fn gemma2_2b() -> Self {
        Self {
            model_type: "gemma2".to_string(),
            hidden_size: 2304,
            intermediate_size: 9216,
            num_hidden_layers: 26,
            num_attention_heads: 8,
            num_key_value_heads: Some(4),
            head_dim: Some(256),
            attn_logit_softcapping: Some(50.0),
            query_pre_attn_scalar: Some(256),
            sliding_window: Some(4096),
            is_gemma2: true,
            ..Default::default()
        }
    }

    /// Create Gemma2 9B configuration.
    pub fn gemma2_9b() -> Self {
        Self {
            model_type: "gemma2".to_string(),
            hidden_size: 3584,
            intermediate_size: 14336,
            num_hidden_layers: 42,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            head_dim: Some(256),
            attn_logit_softcapping: Some(50.0),
            query_pre_attn_scalar: Some(256),
            sliding_window: Some(4096),
            is_gemma2: true,
            ..Default::default()
        }
    }

    /// Create Gemma2 27B configuration.
    pub fn gemma2_27b() -> Self {
        Self {
            model_type: "gemma2".to_string(),
            hidden_size: 4608,
            intermediate_size: 36864,
            num_hidden_layers: 46,
            num_attention_heads: 32,
            num_key_value_heads: Some(16),
            head_dim: Some(128),
            attn_logit_softcapping: Some(30.0),
            query_pre_attn_scalar: Some(256),
            sliding_window: Some(4096),
            is_gemma2: true,
            ..Default::default()
        }
    }
}

/// Gemma-style RMSNorm with +1 offset.
///
/// Unlike standard RMSNorm (x * weight), Gemma uses:
/// output = x * (1 + weight)
#[derive(Debug, ModuleParameters)]
pub struct GemmaRmsNorm {
    /// Weight parameter.
    #[param]
    pub weight: Param<Array>,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

impl GemmaRmsNorm {
    /// Create a new GemmaRmsNorm layer.
    pub fn new(hidden_size: i32, eps: f32) -> Result<Self, Exception> {
        // Initialize weights to zeros (will become 1 after +1 offset)
        let weight = mlx_rs::ops::zeros::<f32>(&[hidden_size])?;

        Ok(Self {
            weight: Param::new(weight),
            eps,
        })
    }

    /// Forward pass.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // Compute RMS
        let x_sq = x.multiply(x)?;
        let mean_sq = x_sq.mean_axis(-1, Some(true))?;
        let eps_arr = Array::from_f32(self.eps);
        let rms = mean_sq.add(&eps_arr)?.sqrt()?;

        // Normalize
        let normed = x.divide(&rms)?;

        // Apply weight with +1 offset: output = normed * (1 + weight)
        let one = Array::from_f32(1.0);
        let scale = self.weight.as_ref().add(&one)?;
        normed.multiply(&scale)
    }
}

/// GELU activation with tanh approximation.
fn gelu_tanh(x: &Array) -> Result<Array, Exception> {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Use tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    let coef = Array::from_f32(0.044715);
    let half = Array::from_f32(0.5);
    let one = Array::from_f32(1.0);
    let two = Array::from_f32(2.0);
    let sqrt_2_pi = Array::from_f32(sqrt_2_over_pi);

    let x_cubed = x.multiply(x)?.multiply(x)?;
    let inner = x.add(&x_cubed.multiply(&coef)?)?;
    let inner = inner.multiply(&sqrt_2_pi)?;

    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    let exp_2x = inner.multiply(&two)?.exp()?;
    let tanh_val = exp_2x.subtract(&one)?.divide(&exp_2x.add(&one)?)?;

    let gate = one.add(&tanh_val)?.multiply(&half)?;

    x.multiply(&gate)
}

/// Gemma attention layer.
#[derive(Debug, ModuleParameters)]
pub struct GemmaAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// Attention logit softcapping (Gemma2).
    pub logit_softcapping: Option<f32>,
    /// RoPE base frequency.
    pub rope_theta: f32,

    /// Query projection.
    #[param]
    pub q_proj: nn::Linear,
    /// Key projection.
    #[param]
    pub k_proj: nn::Linear,
    /// Value projection.
    #[param]
    pub v_proj: nn::Linear,
    /// Output projection.
    #[param]
    pub o_proj: nn::Linear,
    /// RoPE layer.
    #[param]
    pub rope: nn::Rope,
}

impl GemmaAttention {
    /// Create a new attention layer.
    pub fn new(config: &GemmaConfig) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.get_head_dim();
        let scale = config.attention_scale();
        let rope_theta = config.rope_theta;

        let q_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, config.hidden_size)
            .bias(false)
            .build()?;

        let rope = nn::RopeBuilder::new(head_dim)
            .base(rope_theta)
            .traditional(false)
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            logit_softcapping: config.attn_logit_softcapping,
            rope_theta,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
        })
    }

    /// Forward pass through attention using fused kernels.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None)
    }

    /// Forward pass with optional KV cache.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project to Q, K, V
        let queries = Module::forward(&mut self.q_proj, x)?;
        let keys = Module::forward(&mut self.k_proj, x)?;
        let values = Module::forward(&mut self.v_proj, x)?;

        // Reshape for multi-head attention: [B, L, heads, head_dim] -> [B, heads, L, head_dim]
        let queries = queries
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = keys
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let (queries, keys, values) = if let Some((ref cache_ref, _)) = cache {
            let offset = cache_ref.rope_offset();
            let queries = apply_rope(&queries, self.head_dim, false, self.rope_theta, 1.0, offset)?;
            let keys = apply_rope(&keys, self.head_dim, false, self.rope_theta, 1.0, offset)?;
            (queries, keys, values)
        } else {
            let queries = Module::forward(&mut self.rope, &queries)?;
            let keys = Module::forward(&mut self.rope, &keys)?;
            (queries, keys, values)
        };

        // Handle KV cache update
        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &keys, &values)?
        } else {
            (keys, values)
        };

        // Build fused attention config with optional softcapping
        let mut attn_config =
            FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
                .with_scale(self.scale)
                .with_mask_type(if mask.is_some() {
                    AttentionMaskType::None // Custom mask provided
                } else {
                    AttentionMaskType::Causal
                });

        // Add logit softcapping if configured (Gemma2)
        if let Some(cap) = self.logit_softcapping {
            attn_config = attn_config.with_logit_softcapping(cap);
        }

        // Fused SDPA with optional softcapping
        let output = fused_sdpa(&queries, &keys, &values, &attn_config, mask)?;

        // Reshape back: [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        Module::forward(&mut self.o_proj, &output)
    }
}

/// Gemma MLP layer (GeGLU).
#[derive(Debug, ModuleParameters)]
pub struct GemmaMLP {
    /// Gate projection.
    #[param]
    pub gate_proj: nn::Linear,
    /// Up projection.
    #[param]
    pub up_proj: nn::Linear,
    /// Down projection.
    #[param]
    pub down_proj: nn::Linear,
}

impl GemmaMLP {
    /// Create a new MLP layer.
    pub fn new(config: &GemmaConfig) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass (GeGLU activation).
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // GeGLU: down_proj(gelu(gate_proj(x)) * up_proj(x))
        let gate = Module::forward(&mut self.gate_proj, x)?;
        let gate = gelu_tanh(&gate)?;
        let up = Module::forward(&mut self.up_proj, x)?;
        let hidden = gate.multiply(&up)?;
        Module::forward(&mut self.down_proj, &hidden)
    }
}

/// Gemma decoder layer.
#[derive(Debug, ModuleParameters)]
pub struct GemmaDecoderLayer {
    /// Self-attention layer.
    #[param]
    pub self_attn: GemmaAttention,
    /// MLP layer.
    #[param]
    pub mlp: GemmaMLP,
    /// Input layer norm.
    #[param]
    pub input_layernorm: GemmaRmsNorm,
    /// Post-attention layer norm.
    #[param]
    pub post_attention_layernorm: GemmaRmsNorm,
}

impl GemmaDecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: &GemmaConfig) -> Result<Self, Exception> {
        let self_attn = GemmaAttention::new(config)?;
        let mlp = GemmaMLP::new(config)?;

        let input_layernorm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None)
    }

    /// Forward pass with optional KV cache.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        // Pre-norm + attention + residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_with_cache(&normed, mask, cache)?;
        let h = x.add(&attn_out)?;

        // Pre-norm + MLP + residual
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        h.add(&mlp_out)
    }
}

/// Gemma2 decoder layer with extra normalization.
#[derive(Debug, ModuleParameters)]
pub struct Gemma2DecoderLayer {
    /// Self-attention layer.
    #[param]
    pub self_attn: GemmaAttention,
    /// MLP layer.
    #[param]
    pub mlp: GemmaMLP,
    /// Input layer norm.
    #[param]
    pub input_layernorm: GemmaRmsNorm,
    /// Post-attention layer norm.
    #[param]
    pub post_attention_layernorm: GemmaRmsNorm,
    /// Pre-feedforward layer norm (Gemma2 specific).
    #[param]
    pub pre_feedforward_layernorm: GemmaRmsNorm,
    /// Post-feedforward layer norm (Gemma2 specific).
    #[param]
    pub post_feedforward_layernorm: GemmaRmsNorm,
}

impl Gemma2DecoderLayer {
    /// Create a new Gemma2 decoder layer.
    pub fn new(config: &GemmaConfig) -> Result<Self, Exception> {
        let self_attn = GemmaAttention::new(config)?;
        let mlp = GemmaMLP::new(config)?;

        let input_layernorm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let pre_feedforward_layernorm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_feedforward_layernorm =
            GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    /// Forward pass with extra normalization.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None)
    }

    /// Forward pass with optional KV cache.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        // Pre-norm + attention
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_with_cache(&normed, mask, cache)?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let h = x.add(&attn_out)?;

        // Pre-norm + MLP + post-norm + residual
        let normed = self.pre_feedforward_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        h.add(&mlp_out)
    }
}

/// Gemma transformer layers container.
#[derive(Debug, ModuleParameters)]
pub struct GemmaLayers {
    #[param]
    pub gemma1: Option<Vec<GemmaDecoderLayer>>,
    #[param]
    pub gemma2: Option<Vec<Gemma2DecoderLayer>>,
}

/// Gemma base model (without LM head).
#[derive(Debug, ModuleParameters)]
pub struct GemmaModel {
    /// Configuration.
    pub config: GemmaConfig,
    /// Token embeddings.
    #[param]
    pub embed_tokens: nn::Embedding,
    /// Transformer layers.
    #[param]
    pub layers: GemmaLayers,
    /// Final layer norm.
    #[param]
    pub norm: GemmaRmsNorm,
}

impl GemmaModel {
    /// Create a new Gemma model.
    pub fn new(config: GemmaConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = if config.is_gemma2 {
            GemmaLayers {
                gemma1: None,
                gemma2: Some(
                    (0..config.num_hidden_layers)
                        .map(|_| Gemma2DecoderLayer::new(&config))
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            }
        } else {
            GemmaLayers {
                gemma1: Some(
                    (0..config.num_hidden_layers)
                        .map(|_| GemmaDecoderLayer::new(&config))
                        .collect::<Result<Vec<_>, _>>()?,
                ),
                gemma2: None,
            }
        };

        let norm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None)
    }

    /// Forward pass with optional KV cache.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        // Get embeddings and scale
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;
        let scale = Array::from_f32(self.config.embedding_scale());
        hidden_states = hidden_states.multiply(&scale)?;

        // Create causal mask if not provided and not using cache
        let mask = if mask.is_none() && cache.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len)?)
        } else {
            mask.cloned()
        };

        // Pass through transformer layers
        if let Some(ref mut layers) = self.layers.gemma1 {
            for (idx, layer) in layers.iter_mut().enumerate() {
                let c = cache.as_deref_mut().map(|c| (c, idx));
                hidden_states = layer.forward_with_cache(&hidden_states, mask.as_ref(), c)?;
            }
        } else if let Some(ref mut layers) = self.layers.gemma2 {
            for (idx, layer) in layers.iter_mut().enumerate() {
                let c = cache.as_deref_mut().map(|c| (c, idx));
                hidden_states = layer.forward_with_cache(&hidden_states, mask.as_ref(), c)?;
            }
        }

        // Final norm
        self.norm.forward(&hidden_states)
    }
}

/// Gemma model with language modeling head.
#[derive(Debug, ModuleParameters)]
pub struct GemmaForCausalLM {
    /// Base model.
    #[param]
    pub model: GemmaModel,
    // Note: LM head is tied to embedding weights. Gemma always ties embeddings.
}

impl GemmaForCausalLM {
    /// Create a new Gemma model with LM head.
    pub fn new(config: GemmaConfig) -> Result<Self, Exception> {
        let model = GemmaModel::new(config)?;
        Ok(Self { model })
    }

    /// Forward pass producing logits.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None)
    }

    /// Forward pass with optional KV cache.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward_with_cache(input_ids, mask, cache)?;
        // Gemma always ties embeddings
        self.model.embed_tokens.as_linear(&hidden_states)
    }

    /// Create a KV cache for this model.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        use pmetal_mlx::kv_cache::KVCacheConfig;
        let config = &self.model.config;
        KVCache::new(KVCacheConfig::new(
            config.num_hidden_layers as usize,
            max_seq_len,
            config.num_kv_heads() as usize,
            config.get_head_dim() as usize,
        ))
    }

    /// Get configuration.
    pub fn config(&self) -> &GemmaConfig {
        &self.model.config
    }
}

// Trait implementations
impl ModelConfig for GemmaConfig {
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
        self.num_kv_heads()
    }
    fn head_dim(&self) -> i32 {
        self.get_head_dim()
    }
    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }
    fn max_position_embeddings(&self) -> i32 {
        self.max_position_embeddings
    }
    fn norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }
    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }
    fn tie_word_embeddings(&self) -> bool {
        true
    }
}

impl CausalLMModel for GemmaForCausalLM {
    type Config = GemmaConfig;

    fn new(config: Self::Config) -> Result<Self, Exception> {
        Self::new(config)
    }

    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        Self::forward(self, input_ids, mask)
    }

    fn config(&self) -> &Self::Config {
        Self::config(self)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), Exception> {
        crate::loader::load_gemma_weights(self, weights)
            .map_err(|e| Exception::custom(e.to_string()))
    }

    fn eval(&self) -> Result<(), Exception> {
        use mlx_rs::module::ModuleParameters;
        for (_, p) in self.parameters().flatten() {
            p.eval()?;
        }
        Ok(())
    }
}

/// Create a causal attention mask.
fn create_causal_mask(seq_len: i32) -> Result<Array, Exception> {
    let mask = mlx_rs::ops::tri::<f32>(seq_len, None, None)?;
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0);
    mlx_rs::ops::r#where(&mask.eq(&zero)?, &neg_inf, &zero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn small_config() -> GemmaConfig {
        GemmaConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(16),
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_gemma_config_defaults() {
        let config = GemmaConfig::default();
        assert_eq!(config.model_type, "gemma");
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.hidden_act, "gelu");
    }

    #[test]
    fn test_gemma_presets() {
        let config_7b = GemmaConfig::gemma_7b();
        assert_eq!(config_7b.hidden_size, 3072);
        assert_eq!(config_7b.num_hidden_layers, 28);

        let config_2_9b = GemmaConfig::gemma2_9b();
        assert!(config_2_9b.is_gemma2);
        assert!(config_2_9b.attn_logit_softcapping.is_some());
    }

    #[test]
    #[serial]
    fn test_gemma_rms_norm() {
        let norm = GemmaRmsNorm::new(64, 1e-6).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = norm.forward(&x).unwrap();
        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_gelu_tanh() {
        let x = mlx_rs::Array::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], &[4]);
        let output = gelu_tanh(&x).unwrap();
        output.eval().unwrap();
        // GELU(0) should be 0
        // GELU(-x) ≈ 0 for large negative x
        assert_eq!(output.shape(), &[4]);
    }

    #[test]
    #[serial]
    fn test_gemma_attention() {
        let config = small_config();
        let mut attn = GemmaAttention::new(&config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_gemma_mlp() {
        let config = small_config();
        let mut mlp = GemmaMLP::new(&config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_gemma_decoder_layer() {
        let config = small_config();
        let mut layer = GemmaDecoderLayer::new(&config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = layer.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_gemma_model() {
        let config = small_config();
        let mut model = GemmaModel::new(config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let output = model.forward(&input_ids, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_gemma_causal_lm() {
        let config = small_config();
        let mut model = GemmaForCausalLM::new(config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]); // [batch, seq, vocab]
    }

    #[test]
    #[serial]
    fn test_gemma2_decoder_layer() {
        let mut config = small_config();
        config.is_gemma2 = true;
        config.attn_logit_softcapping = Some(50.0);
        config.query_pre_attn_scalar = Some(256);

        let mut layer = Gemma2DecoderLayer::new(&config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = layer.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_attention_logit_softcapping() {
        let mut config = small_config();
        config.attn_logit_softcapping = Some(50.0);

        let mut attn = GemmaAttention::new(&config).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_gemma_kv_cache() {
        let config = small_config();
        let mut model = GemmaForCausalLM::new(config).unwrap();

        // Create cache
        let mut cache = model.create_cache(32);

        // First forward (prompt)
        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model
            .forward_with_cache(&input_ids, None, Some(&mut cache))
            .unwrap();
        logits.eval().unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]);

        // Second forward (incremental)
        let next_token = mlx_rs::Array::from_slice(&[5_i32], &[1, 1]);
        let logits = model
            .forward_with_cache(&next_token, None, Some(&mut cache))
            .unwrap();
        logits.eval().unwrap();

        assert_eq!(logits.shape(), &[1, 1, 1000]);
    }

    #[test]
    #[serial]
    fn test_gemma2_model() {
        let mut config = small_config();
        config.is_gemma2 = true;
        config.attn_logit_softcapping = Some(50.0);
        config.query_pre_attn_scalar = Some(256);

        let mut model = GemmaForCausalLM::new(config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]);
    }
}
