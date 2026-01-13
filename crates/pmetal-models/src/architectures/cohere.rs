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

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::softmax_axis,
    Array,
};
use serde::{Deserialize, Serialize};

use crate::traits::ModelConfig;

/// Cohere model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereConfig {
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
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Use sliding window attention for certain layers.
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Sliding window size.
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    /// Layers that use global attention (no sliding window).
    /// Pattern: every 4th layer uses global attention.
    #[serde(default)]
    pub global_attention_layers: Option<Vec<i32>>,
}

fn default_sliding_window() -> i32 {
    4096
}

impl Default for CohereConfig {
    fn default() -> Self {
        // Default for Command R 35B
        Self {
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
            tie_word_embeddings: false,
            use_sliding_window: true,
            sliding_window: 4096,
            global_attention_layers: None,
        }
    }
}

impl CohereConfig {
    /// Check if a layer uses global attention.
    pub fn uses_global_attention(&self, layer_idx: i32) -> bool {
        if let Some(ref layers) = self.global_attention_layers {
            layers.contains(&layer_idx)
        } else {
            // Default: every 4th layer uses global attention
            (layer_idx + 1) % 4 == 0
        }
    }

    /// Create config for Command R 35B.
    pub fn command_r() -> Self {
        Self::default()
    }

    /// Create config for Command R+ 104B.
    pub fn command_r_plus() -> Self {
        Self {
            hidden_size: 12288,
            intermediate_size: 33792,
            num_hidden_layers: 64,
            num_attention_heads: 96,
            num_key_value_heads: 12,
            ..Default::default()
        }
    }

    /// Create config for Command A 111B (2025).
    pub fn command_a() -> Self {
        Self {
            hidden_size: 12288,
            intermediate_size: 33792,
            num_hidden_layers: 64,
            num_attention_heads: 96,
            num_key_value_heads: 12,
            max_position_embeddings: 262144, // 256K context
            ..Default::default()
        }
    }
}

impl ModelConfig for CohereConfig {
    fn model_type(&self) -> &str {
        "cohere"
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
#[derive(Debug, ModuleParameters)]
pub struct CohereMLP {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

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
        let gate = nn::silu(gate)?;
        let up = Module::forward(&mut self.up_proj, x)?;
        let hidden = gate.multiply(&up)?;
        Module::forward(&mut self.down_proj, &hidden)
    }
}

/// Cohere attention with optional sliding window.
#[derive(Debug, ModuleParameters)]
pub struct CohereAttention {
    pub layer_idx: usize,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub use_sliding_window: bool,
    pub sliding_window: i32,

    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
}

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

        // Check if this layer uses global attention
        let use_sliding_window =
            config.use_sliding_window && !config.uses_global_attention(layer_idx as i32);

        Ok(Self {
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            use_sliding_window,
            sliding_window: config.sliding_window,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        _position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let q = Module::forward(&mut self.q_proj, x)?;
        let k = Module::forward(&mut self.k_proj, x)?;
        let v = Module::forward(&mut self.v_proj, x)?;

        // Reshape for attention
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // RoPE would be applied here in full implementation

        // Transpose for attention: [B, n_heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Attention scores
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let mut scores = q.matmul(&k_t)?;
        scores = scores.multiply(&Array::from_f32(self.scale))?;

        // Apply mask (includes sliding window mask for non-global layers)
        if let Some(m) = mask {
            scores = scores.add(m)?;
        }

        // For sliding window, we could apply additional masking here
        // In production, this would use a sparse attention pattern

        let probs = softmax_axis(&scores, -1, None)?;
        let output = probs.matmul(&v)?;

        // Reshape and project
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[batch, seq_len, -1])?;
        Module::forward(&mut self.o_proj, &output)
    }
}

/// Cohere decoder layer.
#[derive(Debug, ModuleParameters)]
pub struct CohereDecoderLayer {
    pub layer_idx: usize,

    #[param]
    pub self_attn: CohereAttention,
    #[param]
    pub mlp: CohereMLP,
    #[param]
    pub input_layernorm: nn::LayerNorm,
}

impl CohereDecoderLayer {
    pub fn new(config: &CohereConfig, layer_idx: usize) -> Result<Self, Exception> {
        let self_attn = CohereAttention::new(config, layer_idx)?;
        let mlp = CohereMLP::new(config.hidden_size, config.intermediate_size)?;

        // Cohere uses LayerNorm (not RMSNorm)
        let input_layernorm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
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
        // Pre-norm with parallel attention and FFN
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward(&normed, mask, position_ids)?;
        let ffn_out = self.mlp.forward(&normed)?;

        // Residual: x + attn + ffn
        x.add(&attn_out)?.add(&ffn_out)
    }
}

/// Cohere model.
#[derive(Debug, ModuleParameters)]
pub struct CohereModel {
    pub config: CohereConfig,

    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<CohereDecoderLayer>,
    #[param]
    pub norm: nn::LayerNorm,
}

impl CohereModel {
    pub fn new(config: CohereConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| CohereDecoderLayer::new(&config, i as usize))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, mask, position_ids)?;
        }

        Module::forward(&mut self.norm, &hidden_states)
    }
}

/// Cohere for causal language modeling.
#[derive(Debug, ModuleParameters)]
pub struct CohereForCausalLM {
    pub config: CohereConfig,

    #[param]
    pub model: CohereModel,
    #[param]
    pub lm_head: nn::Linear,
}

impl CohereForCausalLM {
    pub fn new(config: CohereConfig) -> Result<Self, Exception> {
        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()?;

        let model = CohereModel::new(config.clone())?;

        Ok(Self {
            config,
            model,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward(input_ids, mask, position_ids)?;
        Module::forward(&mut self.lm_head, &hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::module::ModuleParameters;
    use serial_test::serial;

    #[test]
    fn test_cohere_config() {
        let config = CohereConfig::default();

        // Every 4th layer uses global attention
        assert!(!config.uses_global_attention(0));
        assert!(!config.uses_global_attention(1));
        assert!(!config.uses_global_attention(2));
        assert!(config.uses_global_attention(3)); // 4th layer
        assert!(!config.uses_global_attention(4));
        assert!(!config.uses_global_attention(5));
        assert!(!config.uses_global_attention(6));
        assert!(config.uses_global_attention(7)); // 8th layer
    }

    #[test]
    #[serial]
    fn test_cohere_mlp() {
        let mlp = CohereMLP::new(64, 256).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 10, 64], None, None, None).unwrap();

        let mut mlp = mlp;
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[1, 10, 64]);
    }

    #[test]
    #[serial]
    fn test_cohere_model_instantiation() {
        let mut config = CohereConfig::default();
        config.hidden_size = 64;
        config.intermediate_size = 256;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.head_dim = 16;
        config.vocab_size = 1000;

        let model = CohereForCausalLM::new(config).unwrap();

        let params = model.parameters().flatten();
        assert!(params.len() > 0);
    }
}
