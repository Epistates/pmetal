//! Qwen3 model architecture.
//!
//! Supports Qwen3 0.6B, 1.7B, 4B, 8B, 14B, 32B, and larger variants.
//! Qwen3 differs from Qwen2 primarily in attention:
//! - RMSNorm applied to Q and K before RoPE (q_norm, k_norm)
//! - Support for layer_types configuration (full_attention vs sliding)
//! - Backwards compatible with Qwen2 weight files

use pmetal_mlx::kernels::{fused_sdpa, rope::apply_rope, AttentionMaskType, FusedAttentionConfig};
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

/// Qwen3 model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3Config {
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
    /// Head dimension (fixed at 128 for Qwen3).
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
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
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Sliding window attention size (None = disabled).
    #[serde(default)]
    pub sliding_window: Option<i32>,
    /// Use sliding window attention.
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Attention dropout (unused in inference).
    #[serde(default)]
    pub attention_dropout: f32,
    /// Layer types for each layer (full_attention or sliding_window).
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// Maximum layers using sliding window.
    #[serde(default)]
    pub max_window_layers: Option<i32>,
}

fn default_model_type() -> String {
    "qwen3".to_string()
}
fn default_vocab_size() -> i32 {
    151936
}
fn default_head_dim() -> i32 {
    128
}
fn default_max_position_embeddings() -> i32 {
    32768
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_hidden_act() -> String {
    "silu".to_string()
}

impl Qwen3Config {
    /// Get the number of KV heads (defaults to num_attention_heads if not specified).
    pub fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the head dimension.
    pub fn get_head_dim(&self) -> i32 {
        self.head_dim
    }

    /// Get GQA group count.
    pub fn num_groups(&self) -> i32 {
        self.num_attention_heads / self.num_kv_heads()
    }

    /// Check if layer at index should use sliding window.
    pub fn use_sliding_window_at(&self, layer_idx: usize) -> bool {
        if let Some(ref layer_types) = self.layer_types {
            if layer_idx < layer_types.len() {
                return layer_types[layer_idx] == "sliding_window";
            }
        }
        // Default: use sliding window only if configured and within max_window_layers
        if let Some(max_layers) = self.max_window_layers {
            return self.use_sliding_window && (layer_idx as i32) < max_layers;
        }
        false
    }
}

impl Default for Qwen3Config {
    fn default() -> Self {
        // Qwen3-0.6B defaults
        Self {
            model_type: "qwen3".to_string(),
            vocab_size: 151936,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            head_dim: 128,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: true,
            sliding_window: None,
            use_sliding_window: false,
            attention_dropout: 0.0,
            layer_types: None,
            max_window_layers: None,
        }
    }
}

impl Qwen3Config {
    /// Create Qwen3-0.6B configuration.
    pub fn qwen3_0_6b() -> Self {
        Self::default()
    }

    /// Create Qwen3-1.7B configuration.
    pub fn qwen3_1_7b() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            ..Default::default()
        }
    }

    /// Create Qwen3-4B configuration.
    pub fn qwen3_4b() -> Self {
        Self {
            hidden_size: 2560,
            intermediate_size: 9216,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            ..Default::default()
        }
    }

    /// Create Qwen3-8B configuration.
    pub fn qwen3_8b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            ..Default::default()
        }
    }
}

/// Qwen3 attention layer with Q/K normalization.
///
/// Key difference from Qwen2: applies RMSNorm to queries and keys before RoPE.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3Attention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension (always 128).
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Whether to use sliding window attention.
    pub use_sliding_window: bool,
    /// Sliding window size.
    pub sliding_window: Option<i32>,

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
    /// Query normalization (Qwen3 specific).
    #[param]
    pub q_norm: nn::RmsNorm,
    /// Key normalization (Qwen3 specific).
    #[param]
    pub k_norm: nn::RmsNorm,
    /// RoPE layer.
    #[param]
    pub rope: nn::Rope,
}

impl Qwen3Attention {
    /// Create a new attention layer.
    pub fn new(config: &Qwen3Config, use_sliding: bool) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.get_head_dim();
        let scale = (head_dim as f32).sqrt().recip();
        let rope_theta = config.rope_theta;

        let q_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
            .bias(false) // Qwen3 uses no bias in attention projections
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

        // Qwen3-specific: Q and K normalization before RoPE
        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
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
            rope_theta,
            use_sliding_window: use_sliding,
            sliding_window: config.sliding_window,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
        })
    }

    /// Forward pass through attention.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None)
    }

    /// Forward pass with optional KV cache.
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

        // Reshape for multi-head attention: [B, L, heads, head_dim]
        let queries = queries.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let keys = keys.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let values = values.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // Qwen3-specific: Apply Q/K normalization before RoPE
        let queries = Module::forward(&mut self.q_norm, &queries)?;
        let keys = Module::forward(&mut self.k_norm, &keys)?;

        // Transpose for attention: [B, heads, L, head_dim]
        let queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        let keys = keys.transpose_axes(&[0, 2, 1, 3])?;
        let values = values.transpose_axes(&[0, 2, 1, 3])?;

        // Get RoPE offset and apply RoPE (after Q/K norm)
        let (queries, keys, values) = if let Some((ref cache_ref, _layer_idx)) = cache {
            let offset = cache_ref.rope_offset();
            let queries = apply_rope(&queries, self.head_dim, false, self.rope_theta, 1.0, offset)?;
            let keys = apply_rope(&keys, self.head_dim, false, self.rope_theta, 1.0, offset)?;
            (queries, keys, values)
        } else {
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

        // Determine mask type for fused attention
        let mask_type = if mask.is_some() {
            AttentionMaskType::None // Custom mask provided
        } else if self.use_sliding_window {
            if let Some(window) = self.sliding_window {
                AttentionMaskType::SlidingWindow(window)
            } else {
                AttentionMaskType::Causal
            }
        } else {
            AttentionMaskType::Causal
        };

        // Use fused attention kernel
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(mask_type);

        let output = fused_sdpa(&queries, &keys, &values, &attn_config, mask)?;

        // Reshape back: [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        // Output projection
        Module::forward(&mut self.o_proj, &output)
    }
}

/// Qwen3 MLP layer (SwiGLU).
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MLP {
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

impl Qwen3MLP {
    /// Create a new MLP layer.
    pub fn new(config: &Qwen3Config) -> Result<Self, Exception> {
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
        let gate = Module::forward(&mut self.gate_proj, x)?;
        let gate = nn::silu(gate)?;
        let up = Module::forward(&mut self.up_proj, x)?;
        let hidden = gate.multiply(&up)?;
        Module::forward(&mut self.down_proj, &hidden)
    }
}

/// Qwen3 transformer block.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3DecoderLayer {
    /// Self-attention layer.
    #[param]
    pub self_attn: Qwen3Attention,
    /// MLP layer.
    #[param]
    pub mlp: Qwen3MLP,
    /// Input layer norm.
    #[param]
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl Qwen3DecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: &Qwen3Config, use_sliding: bool) -> Result<Self, Exception> {
        let self_attn = Qwen3Attention::new(config, use_sliding)?;
        let mlp = Qwen3MLP::new(config)?;

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

/// Qwen3 base model (without LM head).
#[derive(Debug, ModuleParameters)]
pub struct Qwen3Model {
    /// Configuration.
    pub config: Qwen3Config,
    /// Token embeddings.
    #[param]
    pub embed_tokens: nn::Embedding,
    /// Transformer layers.
    #[param]
    pub layers: Vec<Qwen3DecoderLayer>,
    /// Final layer norm.
    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen3Model {
    /// Create a new Qwen3 model.
    pub fn new(config: Qwen3Config) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| {
                let use_sliding = config.use_sliding_window_at(i);
                Qwen3DecoderLayer::new(&config, use_sliding)
            })
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
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        // Don't create explicit causal mask - fused SDPA handles it internally
        // with proper dtype handling. Only pass through user-provided masks.

        // Pass through transformer layers
        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    hidden_states =
                        layer.forward_with_cache(&hidden_states, mask, Some((cache, layer_idx)))?;
                }
            }
            None => {
                for layer in &mut self.layers {
                    hidden_states = layer.forward(&hidden_states, mask)?;
                }
            }
        }

        // Final norm
        Module::forward(&mut self.norm, &hidden_states)
    }
}

/// Qwen3 model with language modeling head.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3ForCausalLM {
    /// Base model.
    #[param]
    pub model: Qwen3Model,
    /// LM head (optional, may share weights with embeddings).
    #[param]
    pub lm_head: Option<nn::Linear>,
}

impl Qwen3ForCausalLM {
    /// Create a new Qwen3 model with LM head.
    pub fn new(config: Qwen3Config) -> Result<Self, Exception> {
        let tie_weights = config.tie_word_embeddings;
        let model = Qwen3Model::new(config.clone())?;

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

        if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, &hidden_states)
        } else {
            // Tie weights: use embedding weight transposed
            self.model.embed_tokens.as_linear(&hidden_states)
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &Qwen3Config {
        &self.model.config
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn small_config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: 64,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_qwen3_config_defaults() {
        let config = Qwen3Config::default();
        assert_eq!(config.model_type, "qwen3");
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.rope_theta, 1_000_000.0);
    }

    #[test]
    #[serial]
    fn test_qwen3_attention_has_q_k_norm() {
        // Just verify the attention module with q/k norm can be constructed and used
        let config = small_config();
        let mut attn = Qwen3Attention::new(&config, false).unwrap();

        // Verify a forward pass works (confirms q_norm and k_norm are functional)
        let x = mlx_rs::random::normal::<f32>(&[1, 4, 128], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();
        assert_eq!(output.shape(), &[1, 4, 128]);
    }

    #[test]
    #[serial]
    fn test_qwen3_attention() {
        let config = small_config();
        let mut attn = Qwen3Attention::new(&config, false).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 128], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 128]);
    }

    #[test]
    #[serial]
    fn test_qwen3_forward() {
        let config = small_config();
        let mut model = Qwen3ForCausalLM::new(config.clone()).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let output = model.forward(&input_ids, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, config.vocab_size]);
    }

    #[test]
    fn test_layer_types_config() {
        let config = Qwen3Config {
            layer_types: Some(vec![
                "full_attention".to_string(),
                "sliding_window".to_string(),
                "full_attention".to_string(),
            ]),
            ..Default::default()
        };

        assert!(!config.use_sliding_window_at(0));
        assert!(config.use_sliding_window_at(1));
        assert!(!config.use_sliding_window_at(2));
    }
}
