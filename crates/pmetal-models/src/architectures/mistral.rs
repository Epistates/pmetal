//! Mistral model architecture.
//!
//! Supports Mistral 7B and Mixtral (MoE) variants.
//!
//! Key differences from Llama:
//! - Sliding Window Attention (SWA) for efficient long-context handling
//! - Different default configurations

use std::collections::HashMap;

use pmetal_mlx::kernels::{fused_sdpa, AttentionMaskType, FusedAttentionConfig, rope::apply_rope};
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

use crate::traits::{ModelConfig, CausalLMModel};

/// Mistral model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Hidden dimension.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    /// Intermediate size (MLP).
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    /// Number of hidden layers.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    #[serde(default = "default_num_attention_heads")]
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
    /// Sliding window size for attention.
    #[serde(default = "default_sliding_window")]
    pub sliding_window: Option<i32>,
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
    "mistral".to_string()
}
fn default_vocab_size() -> i32 {
    32000
}
fn default_hidden_size() -> i32 {
    4096
}
fn default_intermediate_size() -> i32 {
    14336
}
fn default_num_hidden_layers() -> i32 {
    32
}
fn default_num_attention_heads() -> i32 {
    32
}
fn default_max_position_embeddings() -> i32 {
    32768
}
fn default_sliding_window() -> Option<i32> {
    Some(4096) // Mistral uses 4096 token sliding window
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

// Re-use RopeScalingValue from llama module
pub use super::llama::RopeScalingValue;

impl MistralConfig {
    /// Get the number of KV heads (defaults to 8 for GQA if not specified).
    pub fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(8)
    }

    /// Get the head dimension.
    pub fn get_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

impl Default for MistralConfig {
    fn default() -> Self {
        // Mistral 7B defaults
        Self {
            model_type: "mistral".to_string(),
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8), // GQA with 8 KV heads
            head_dim: None,
            max_position_embeddings: 32768,
            sliding_window: Some(4096),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: false,
        }
    }
}

/// Mistral attention layer with sliding window support.
#[derive(Debug, ModuleParameters)]
pub struct MistralAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// Sliding window size (None for full attention).
    pub sliding_window: Option<i32>,
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

impl MistralAttention {
    /// Create a new attention layer.
    pub fn new(config: &MistralConfig) -> Result<Self, Exception> {
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

        // Initialize RoPE
        let rope = nn::RopeBuilder::new(head_dim)
            .base(rope_theta)
            .traditional(false)
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            sliding_window: config.sliding_window,
            rope_theta,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
        })
    }

    /// Forward pass through attention with optional sliding window using fused kernels.
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

        // Determine mask type for fused attention
        let mask_type = if mask.is_some() {
            AttentionMaskType::None // Custom mask provided
        } else if let Some(window) = self.sliding_window {
            AttentionMaskType::SlidingWindow(window)
        } else {
            AttentionMaskType::Causal
        };

        // Use fused attention kernel - handles GQA natively
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(mask_type);

        // Fused SDPA
        let output = fused_sdpa(&queries, &keys, &values, &attn_config, mask)?;

        // Reshape back: [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        // Output projection
        Module::forward(&mut self.o_proj, &output)
    }
}

/// Mistral MLP layer (SwiGLU).
#[derive(Debug, ModuleParameters)]
pub struct MistralMLP {
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

impl MistralMLP {
    /// Create a new MLP layer.
    pub fn new(config: &MistralConfig) -> Result<Self, Exception> {
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

/// Mistral transformer block.
#[derive(Debug, ModuleParameters)]
pub struct MistralDecoderLayer {
    /// Self-attention layer.
    #[param]
    pub self_attn: MistralAttention,
    /// MLP layer.
    #[param]
    pub mlp: MistralMLP,
    /// Input layer norm.
    #[param]
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl MistralDecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: &MistralConfig) -> Result<Self, Exception> {
        let self_attn = MistralAttention::new(config)?;
        let mlp = MistralMLP::new(config)?;

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

/// Mistral base model (without LM head).
#[derive(Debug, ModuleParameters)]
pub struct MistralModel {
    /// Configuration (not a parameter).
    pub config: MistralConfig,
    /// Token embeddings.
    #[param]
    pub embed_tokens: nn::Embedding,
    /// Transformer layers.
    #[param]
    pub layers: Vec<MistralDecoderLayer>,
    /// Final layer norm.
    #[param]
    pub norm: nn::RmsNorm,
}

impl MistralModel {
    /// Create a new Mistral model.
    pub fn new(config: MistralConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|_| MistralDecoderLayer::new(&config))
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
        // Get embeddings
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        // Create causal mask if not provided and not using cache
        let mask = if mask.is_none() && cache.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len)?)
        } else {
            mask.cloned()
        };

        // Pass through transformer layers
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let c = cache.as_deref_mut().map(|c| (c, idx));
            hidden_states = layer.forward_with_cache(
                &hidden_states,
                mask.as_ref(),
                c,
            )?;
        }

        // Final norm
        Module::forward(&mut self.norm, &hidden_states)
    }
}

/// Mistral model with language modeling head.
#[derive(Debug, ModuleParameters)]
pub struct MistralForCausalLM {
    /// Base model.
    #[param]
    pub model: MistralModel,
    /// LM head (optional, may share weights with embeddings).
    #[param]
    pub lm_head: Option<nn::Linear>,
}

impl MistralForCausalLM {
    /// Create a new Mistral model with LM head.
    pub fn new(config: MistralConfig) -> Result<Self, Exception> {
        let tie_weights = config.tie_word_embeddings;
        let model = MistralModel::new(config.clone())?;

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

        // Get logits from LM head or shared embeddings
        if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, &hidden_states)
        } else {
            // Tie weights: use embedding weight transposed
            self.model.embed_tokens.as_linear(&hidden_states)
        }
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
    pub fn config(&self) -> &MistralConfig {
        &self.model.config
    }
}

// Trait implementations
impl ModelConfig for MistralConfig {
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

impl CausalLMModel for MistralForCausalLM {
    type Config = MistralConfig;

    fn new(config: Self::Config) -> Result<Self, Exception> {
        Self::new(config)
    }

    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        MistralForCausalLM::forward(self, input_ids, mask)
    }

    fn config(&self) -> &Self::Config {
        Self::config(self)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), Exception> {
        crate::loader::load_mistral_weights(self, weights).map_err(|e| Exception::custom(e.to_string()))
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

    fn small_config() -> MistralConfig {
        MistralConfig {
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
            sliding_window: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_mistral_config_defaults() {
        let config = MistralConfig::default();
        assert_eq!(config.model_type, "mistral");
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_kv_heads(), 8); // GQA
    }

    #[test]
    fn test_mistral_config_with_sliding_window() {
        let mut config = MistralConfig::default();
        config.sliding_window = Some(4096);
        assert_eq!(config.sliding_window, Some(4096));
    }

    #[test]
    #[serial]
    fn test_mistral_rms_norm() {
        let mut norm = nn::RmsNorm::new(64).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = Module::forward(&mut norm, &x).unwrap();
        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_mistral_attention() {
        let config = small_config();
        let mut attn = MistralAttention::new(&config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_mistral_attention_with_sliding_window() {
        let mut config = small_config();
        config.sliding_window = Some(2);

        let mut attn = MistralAttention::new(&config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_mistral_mlp() {
        let config = small_config();
        let mut mlp = MistralMLP::new(&config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_mistral_decoder_layer() {
        let config = small_config();
        let mut layer = MistralDecoderLayer::new(&config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = layer.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_mistral_model() {
        let config = small_config();
        let mut model = MistralModel::new(config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let output = model.forward(&input_ids, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_mistral_causal_lm() {
        let config = small_config();
        let mut model = MistralForCausalLM::new(config.clone()).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, config.vocab_size]); // [batch, seq, vocab]
    }

    #[test]
    #[serial]
    fn test_mistral_kv_cache() {
        let config = small_config();
        let mut model = MistralForCausalLM::new(config).unwrap();

        // Create cache
        let mut cache = model.create_cache(32);

        // First forward (prompt)
        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward_with_cache(&input_ids, None, Some(&mut cache)).unwrap();
        logits.eval().unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]);

        // Second forward (incremental)
        let next_token = mlx_rs::Array::from_slice(&[5_i32], &[1, 1]);
        let logits = model.forward_with_cache(&next_token, None, Some(&mut cache)).unwrap();
        logits.eval().unwrap();

        assert_eq!(logits.shape(), &[1, 1, 1000]);
    }
}
