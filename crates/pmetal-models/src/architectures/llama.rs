//! Llama model architecture.
//!
//! Supports Llama 2, 3, 3.1, 3.2, 3.3, and 4 variants.

use std::collections::HashMap;

use pmetal_mlx::kernels::{
    differentiable_attention, fused_sdpa, get_training_context,
    rope::apply_rope, AttentionMaskType, FusedAttentionConfig,
};
use pmetal_mlx::kv_cache::KVCache;
use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    Array,
};
use serde::{Deserialize, Serialize};

/// Llama model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
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
    /// Head dimension (computed if not provided).
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
    /// RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, RopeScalingValue>>,
    /// Hidden activation function.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_model_type() -> String {
    "llama".to_string()
}
fn default_max_position_embeddings() -> i32 {
    4096
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_hidden_act() -> String {
    "silu".to_string()
}

/// RoPE scaling configuration value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RopeScalingValue {
    /// Floating point value.
    Float(f32),
    /// String value.
    String(String),
}

impl LlamaConfig {
    /// Get the number of KV heads (defaults to num_attention_heads if not specified).
    pub fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the head dimension.
    pub fn get_head_dim(&self) -> i32 {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

impl Default for LlamaConfig {
    fn default() -> Self {
        // Llama 3.2 1B defaults
        Self {
            model_type: "llama".to_string(),
            vocab_size: 128256,
            hidden_size: 2048,
            intermediate_size: 8192,
            num_hidden_layers: 16,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            head_dim: None,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            rope_scaling: None,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: true,
        }
    }
}

/// Llama attention layer.
#[derive(Debug, ModuleParameters)]
pub struct LlamaAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// RoPE base frequency (for apply_rope with offset).
    pub rope_theta: f32,
    /// Layer ID for training cache (set during model construction).
    pub layer_id: usize,

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
    /// RoPE layer (used for non-cached forward).
    #[param]
    pub rope: nn::Rope,
}

impl LlamaAttention {
    /// Create a new attention layer.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `layer_id` - Layer index (used for training cache)
    pub fn new(config: &LlamaConfig, layer_id: usize) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.get_head_dim();
        let scale = (head_dim as f32).sqrt().recip();
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

        // Initialize RoPE (for non-cached forward)
        let rope = nn::RopeBuilder::new(head_dim)
            .base(rope_theta)
            .traditional(false)
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            rope_theta,
            layer_id,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
        })
    }

    /// Forward pass through attention using fused Metal kernels.
    ///
    /// Uses MLX's optimized `scaled_dot_product_attention` which provides:
    /// - Metal kernel optimization for single-token generation
    /// - Native GQA support (no manual KV head expansion overhead)
    /// - Reduced memory bandwidth
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len, hidden_size]
    /// * `mask` - Optional attention mask (additive, -inf for masked positions)
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_size]
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        // Non-cached forward - delegate to cached version with None
        self.forward_with_cache(x, mask, None)
    }

    /// Forward pass with optional KV cache for efficient generation.
    ///
    /// When `cache` is provided:
    /// - RoPE is applied with position offset from cache
    /// - K/V tensors are concatenated with cached values
    /// - Only new tokens need to be processed (O(1) per token instead of O(n))
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len, hidden_size]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional (KVCache, layer_idx) tuple for cached generation
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_size]
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

        // Get RoPE offset and apply RoPE
        let (queries, keys, values) = if let Some((ref cache_ref, _layer_idx)) = cache {
            // Cached path: use apply_rope with offset
            let offset = cache_ref.rope_offset();
            let queries = apply_rope(&queries, self.head_dim, false, self.rope_theta, 1.0, offset)?;
            let keys = apply_rope(&keys, self.head_dim, false, self.rope_theta, 1.0, offset)?;
            (queries, keys, values)
        } else {
            // Non-cached path: use RoPE module (offset=0)
            let queries = Module::forward(&mut self.rope, &queries)?;
            let keys = Module::forward(&mut self.rope, &keys)?;
            (queries, keys, values)
        };

        // Handle KV cache update - keys/values are already in [B, heads, seq, head_dim] format
        // No transpose needed - cache uses attention format directly (SOTA performance matching mlx_lm)
        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &keys, &values)?
        } else {
            (keys, values)
        };

        // Use fused attention kernel - handles GQA natively (no KV head expansion needed)
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None // Custom mask provided
            } else {
                AttentionMaskType::Causal // Use built-in causal mask
            });

        // Check if training mode is enabled for efficient backward pass
        let is_training = get_training_context()
            .map(|ctx| ctx.lock().map(|c| c.is_training()).unwrap_or(false))
            .unwrap_or(false);

        // Use differentiable attention for training (O(n) backward pass via Metal FlashAttention)
        // or fused SDPA for inference
        let output = if is_training && mask.is_none() {
            // Training mode: use Metal FlashAttention with proper backward pass
            differentiable_attention(self.layer_id, &queries, &keys, &values, &attn_config)
                .map_err(|e| Exception::custom(e.to_string()))?
        } else {
            // Inference mode: use fused SDPA (faster for single token generation)
            fused_sdpa(&queries, &keys, &values, &attn_config, mask)?
        };

        // Reshape back: [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        // Output projection
        Module::forward(&mut self.o_proj, &output)
    }
}

/// Llama MLP layer (SwiGLU).
#[derive(Debug, ModuleParameters)]
pub struct LlamaMLP {
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

impl LlamaMLP {
    /// Create a new MLP layer.
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
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

    /// Forward pass (SwiGLU activation).
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = Module::forward(&mut self.gate_proj, x)?;
        let gate = nn::silu(gate)?;
        let up = Module::forward(&mut self.up_proj, x)?;
        let hidden = gate.multiply(&up)?;
        Module::forward(&mut self.down_proj, &hidden)
    }
}

/// Llama transformer block.
#[derive(Debug, ModuleParameters)]
pub struct LlamaDecoderLayer {
    /// Self-attention layer.
    #[param]
    pub self_attn: LlamaAttention,
    /// MLP layer.
    #[param]
    pub mlp: LlamaMLP,
    /// Input layer norm.
    #[param]
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl LlamaDecoderLayer {
    /// Create a new decoder layer.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `layer_id` - Layer index (used for training cache)
    pub fn new(config: &LlamaConfig, layer_id: usize) -> Result<Self, Exception> {
        let self_attn = LlamaAttention::new(config, layer_id)?;
        let mlp = LlamaMLP::new(config)?;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

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
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward_with_cache(&normed, mask, cache)?;
        let h = x.add(&attn_out)?;

        // Pre-norm + MLP + residual
        let normed = Module::forward(&mut self.post_attention_layernorm, &h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        h.add(&mlp_out)
    }
}

/// Llama base model (without LM head).
#[derive(Debug, ModuleParameters)]
pub struct LlamaModel {
    /// Configuration (not a parameter).
    pub config: LlamaConfig,
    /// Token embeddings.
    #[param]
    pub embed_tokens: nn::Embedding,
    /// Transformer layers.
    #[param]
    pub layers: Vec<LlamaDecoderLayer>,
    /// Final layer norm.
    #[param]
    pub norm: nn::RmsNorm,
}

impl LlamaModel {
    /// Create a new Llama model.
    pub fn new(config: LlamaConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|layer_id| LlamaDecoderLayer::new(&config, layer_id as usize))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs of shape [batch, seq_len]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Hidden states of shape [batch, seq_len, hidden_size]
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None)
    }

    /// Forward pass with optional KV cache.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs of shape [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KV cache for efficient generation
    ///
    /// # Returns
    /// Hidden states of shape [batch, seq_len, hidden_size]
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        // Get embeddings
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        // Create causal mask if not provided and not using cache
        // When using cache for decode step (seq_len=1), we don't need a mask
        let mask = if mask.is_none() && cache.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len)?)
        } else {
            mask.cloned()
        };

        // Pass through transformer layers
        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    hidden_states =
                        layer.forward_with_cache(&hidden_states, mask.as_ref(), Some((cache, layer_idx)))?;
                }
            }
            None => {
                for layer in &mut self.layers {
                    hidden_states = layer.forward(&hidden_states, mask.as_ref())?;
                }
            }
        }

        // Final norm
        Module::forward(&mut self.norm, &hidden_states)
    }
}

/// Llama model with language modeling head.
#[derive(Debug, ModuleParameters)]
pub struct LlamaForCausalLM {
    /// Base model.
    #[param]
    pub model: LlamaModel,
    /// LM head (optional, may share weights with embeddings).
    #[param]
    pub lm_head: Option<nn::Linear>,
}

impl LlamaForCausalLM {
    /// Create a new Llama model with LM head.
    pub fn new(config: LlamaConfig) -> Result<Self, Exception> {
        let tie_weights = config.tie_word_embeddings;
        let model = LlamaModel::new(config.clone())?;

        let lm_head = if !tie_weights {
            let head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                .bias(false)
                .build()?;
            Some(head)
        } else {
            None
        };

        Ok(Self { model, lm_head })
    }

    /// Forward pass producing logits.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs of shape [batch, seq_len]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Logits of shape [batch, seq_len, vocab_size]
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None)
    }

    /// Forward pass with optional KV cache for efficient generation.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs of shape [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KV cache
    ///
    /// # Returns
    /// Logits of shape [batch, seq_len, vocab_size]
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward_with_cache(input_ids, mask, cache)?;

        // Get logits from LM head or shared embeddings
        if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, &hidden_states)
        } else {
            // Tie weights: use embedding weight transposed
            self.model.embed_tokens.as_linear(&hidden_states)
        }
    }

    /// Create a KV cache for this model.
    ///
    /// # Arguments
    /// * `max_seq_len` - Maximum sequence length to cache
    ///
    /// # Returns
    /// A new KV cache configured for this model
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
    pub fn config(&self) -> &LlamaConfig {
        &self.model.config
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl crate::traits::ModelConfig for LlamaConfig {
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
        self.tie_word_embeddings
    }
}

impl crate::traits::CausalLMModel for LlamaForCausalLM {
    type Config = LlamaConfig;

    fn new(config: Self::Config) -> Result<Self, Exception> {
        LlamaForCausalLM::new(config)
    }

    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        LlamaForCausalLM::forward(self, input_ids, mask)
    }

    fn config(&self) -> &Self::Config {
        LlamaForCausalLM::config(self)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), Exception> {
        crate::loader::load_llama_weights(self, weights)
            .map_err(|e| Exception::custom(e.to_string()))
    }

    fn eval(&self) -> Result<(), Exception> {
        // Eval all parameters to materialize them on device
        use mlx_rs::module::ModuleParameters;
        let params = self.parameters().flatten();
        for (_, param) in params {
            param.eval()?;
        }
        Ok(())
    }
}

/// Create a causal attention mask.
fn create_causal_mask(seq_len: i32) -> Result<Array, Exception> {
    // Create lower triangular mask with 1s for valid positions
    let mask = mlx_rs::ops::tri::<f32>(seq_len, None, None)?;
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0);

    // Where mask is 0, put -inf; where mask is 1, put 0
    mlx_rs::ops::r#where(&mask.eq(&zero)?, &neg_inf, &zero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn small_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: None,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            ..Default::default()
        }
    }

    #[test]
    #[serial]
    fn test_llama_attention() {
        let config = small_config();
        let mut attn = LlamaAttention::new(&config, 0).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_llama_mlp() {
        let config = small_config();
        let mut mlp = LlamaMLP::new(&config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_llama_decoder_layer() {
        let config = small_config();
        let mut layer = LlamaDecoderLayer::new(&config, 0).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = layer.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_llama_model() {
        let config = small_config();
        let mut model = LlamaModel::new(config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let output = model.forward(&input_ids, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_llama_causal_lm() {
        let config = small_config();
        let mut model = LlamaForCausalLM::new(config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]); // [batch, seq, vocab]
    }
}
