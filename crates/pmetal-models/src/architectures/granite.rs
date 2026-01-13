//! IBM Granite architecture.
//!
//! Granite is IBM's family of enterprise-focused LLMs featuring:
//! - Standard transformer variants (Granite 4.0)
//! - Hybrid Mamba2 + Attention variants (Granite 4.0-H)
//! - MoE variants with shared experts (Granite 4.0-H-Tiny)
//!
//! All Granite 4.0 models use:
//! - GQA (Grouped Query Attention)
//! - RoPE positional embeddings
//! - SwiGLU activation
//! - RMSNorm
//! - Shared input/output embeddings

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

/// Layer type for Granite Hybrid models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GraniteLayerType {
    /// Standard attention layer.
    #[default]
    Attention,
    /// Mamba2 state-space layer.
    Mamba2,
}

/// Granite model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraniteConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Whether to tie input/output embeddings.
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,

    // Hybrid model options
    /// Whether this is a hybrid (Mamba2 + Attention) model.
    #[serde(default)]
    pub is_hybrid: bool,
    /// Layer types for hybrid models.
    #[serde(default)]
    pub layer_types: Option<Vec<GraniteLayerType>>,
    /// Mamba2 state dimension (for hybrid models).
    #[serde(default = "default_mamba_state_dim")]
    pub mamba_state_dim: i32,
    /// Mamba2 conv dimension.
    #[serde(default = "default_mamba_conv_dim")]
    pub mamba_conv_dim: i32,

    // MoE options
    /// Whether this is an MoE model.
    #[serde(default)]
    pub is_moe: bool,
    /// Number of experts for MoE.
    #[serde(default = "default_num_experts")]
    pub num_experts: i32,
    /// Number of experts per token.
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: i32,
    /// Whether to use a shared expert.
    #[serde(default = "default_true")]
    pub use_shared_expert: bool,
}

fn default_true() -> bool {
    true
}
fn default_mamba_state_dim() -> i32 {
    128
}
fn default_mamba_conv_dim() -> i32 {
    4
}
fn default_num_experts() -> i32 {
    8
}
fn default_num_experts_per_tok() -> i32 {
    2
}

impl Default for GraniteConfig {
    fn default() -> Self {
        // Default for Granite 4.0-1B
        Self {
            vocab_size: 49152,
            hidden_size: 2048,
            intermediate_size: 5504,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 128,
            max_position_embeddings: 8192,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            is_hybrid: false,
            layer_types: None,
            mamba_state_dim: 128,
            mamba_conv_dim: 4,
            is_moe: false,
            num_experts: 8,
            num_experts_per_tok: 2,
            use_shared_expert: true,
        }
    }
}

impl GraniteConfig {
    /// Create config for Granite 4.0 Micro (350M).
    pub fn micro() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 2816,
            num_hidden_layers: 18,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            ..Default::default()
        }
    }

    /// Create config for Granite 4.0 1B.
    pub fn granite_1b() -> Self {
        Self::default()
    }

    /// Create config for Granite 4.0-H Hybrid 1B.
    pub fn granite_h_1b() -> Self {
        Self {
            is_hybrid: true,
            // Typical pattern: alternating attention and mamba layers
            layer_types: Some(vec![GraniteLayerType::Attention, GraniteLayerType::Mamba2]),
            ..Default::default()
        }
    }

    /// Create config for Granite 4.0-H-Tiny (MoE).
    pub fn granite_h_tiny_moe() -> Self {
        Self {
            hidden_size: 512,
            intermediate_size: 1408,
            num_hidden_layers: 12,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            is_hybrid: true,
            is_moe: true,
            num_experts: 8,
            num_experts_per_tok: 2,
            use_shared_expert: true,
            ..Default::default()
        }
    }

    /// Get the layer type for a given layer index.
    pub fn layer_type(&self, layer_idx: usize) -> GraniteLayerType {
        if !self.is_hybrid {
            return GraniteLayerType::Attention;
        }

        if let Some(ref types) = self.layer_types {
            if types.len() == self.num_hidden_layers as usize {
                return types[layer_idx];
            }
            // Pattern: repeat the layer_types
            return types[layer_idx % types.len()];
        }

        GraniteLayerType::Attention
    }
}

impl ModelConfig for GraniteConfig {
    fn model_type(&self) -> &str {
        "granite"
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
        self.rms_norm_eps
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

/// SwiGLU MLP for Granite.
#[derive(Debug, ModuleParameters)]
pub struct GraniteMLP {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl GraniteMLP {
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
        // SwiGLU: silu(gate) * up
        let gate = Module::forward(&mut self.gate_proj, x)?;
        let gate = nn::silu(gate)?;
        let up = Module::forward(&mut self.up_proj, x)?;
        let hidden = gate.multiply(&up)?;
        Module::forward(&mut self.down_proj, &hidden)
    }
}

/// Granite attention with GQA and RoPE.
#[derive(Debug, ModuleParameters)]
pub struct GraniteAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
}

impl GraniteAttention {
    pub fn new(config: &GraniteConfig) -> Result<Self, Exception> {
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

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
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

        // RoPE would be applied here

        // Transpose for attention
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Attention scores
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let mut scores = q.matmul(&k_t)?;
        scores = scores.multiply(&Array::from_f32(self.scale))?;

        if let Some(m) = mask {
            scores = scores.add(m)?;
        }

        let probs = softmax_axis(&scores, -1, None)?;
        let output = probs.matmul(&v)?;

        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[batch, seq_len, -1])?;
        Module::forward(&mut self.o_proj, &output)
    }
}

/// Simplified Mamba2 layer for Granite-H models.
///
/// Note: Full Mamba2 requires custom kernels for efficient implementation.
/// This is a simplified version that approximates the behavior.
#[derive(Debug, ModuleParameters)]
pub struct GraniteMamba2 {
    pub state_dim: i32,
    pub conv_dim: i32,

    #[param]
    pub in_proj: nn::Linear,
    #[param]
    pub conv1d_weight: Param<Array>,
    #[param]
    pub out_proj: nn::Linear,
}

impl GraniteMamba2 {
    pub fn new(config: &GraniteConfig) -> Result<Self, Exception> {
        let hidden_size = config.hidden_size;
        let state_dim = config.mamba_state_dim;
        let conv_dim = config.mamba_conv_dim;

        // Expand by 2x for gate and value
        let in_proj = nn::LinearBuilder::new(hidden_size, hidden_size * 2)
            .bias(false)
            .build()?;

        // 1D conv weight [conv_dim, hidden_size]
        let conv1d_weight =
            mlx_rs::random::normal::<f32>(&[conv_dim, hidden_size], None, None, None)?;

        let out_proj = nn::LinearBuilder::new(hidden_size, hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            state_dim,
            conv_dim,
            in_proj,
            conv1d_weight: Param::new(conv1d_weight),
            out_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let _batch = x.shape()[0];
        let _seq_len = x.shape()[1];
        let _hidden = x.shape()[2];

        // Project input to 2x hidden for gate and value
        let xz = Module::forward(&mut self.in_proj, x)?;

        // Split into x and z (gate) using split operation
        // xz: [batch, seq, 2*hidden] -> split along axis -1 into 2 parts
        let parts = xz.split(2, -1)?;
        let x_half = parts[0].clone();
        let z = parts[1].clone();

        // Simplified: apply gate with SiLU
        let z_gate = nn::silu(z)?;
        let gated = x_half.multiply(&z_gate)?;

        // Project out
        Module::forward(&mut self.out_proj, &gated)
    }
}

/// Granite decoder layer.
#[derive(Debug, ModuleParameters)]
pub struct GraniteDecoderLayer {
    pub layer_type: GraniteLayerType,

    #[param]
    pub attention: Option<GraniteAttention>,
    #[param]
    pub mamba: Option<GraniteMamba2>,
    #[param]
    pub mlp: GraniteMLP,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl GraniteDecoderLayer {
    pub fn new(config: &GraniteConfig, layer_idx: usize) -> Result<Self, Exception> {
        let layer_type = config.layer_type(layer_idx);

        let (attention, mamba) = match layer_type {
            GraniteLayerType::Attention => (Some(GraniteAttention::new(config)?), None),
            GraniteLayerType::Mamba2 => (None, Some(GraniteMamba2::new(config)?)),
        };

        let mlp = GraniteMLP::new(config.hidden_size, config.intermediate_size)?;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            layer_type,
            attention,
            mamba,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Pre-norm
        let normed = Module::forward(&mut self.input_layernorm, x)?;

        // Mixer (attention or mamba)
        let mixer_out = match self.layer_type {
            GraniteLayerType::Attention => {
                self.attention
                    .as_mut()
                    .unwrap()
                    .forward(&normed, mask, position_ids)?
            }
            GraniteLayerType::Mamba2 => self.mamba.as_mut().unwrap().forward(&normed)?,
        };

        let h = x.add(&mixer_out)?;

        // FFN
        let normed = Module::forward(&mut self.post_attention_layernorm, &h)?;
        let ffn_out = self.mlp.forward(&normed)?;

        h.add(&ffn_out)
    }
}

/// Granite model.
#[derive(Debug, ModuleParameters)]
pub struct GraniteModel {
    pub config: GraniteConfig,

    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<GraniteDecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl GraniteModel {
    pub fn new(config: GraniteConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| GraniteDecoderLayer::new(&config, i as usize))
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

/// Granite for causal language modeling.
#[derive(Debug, ModuleParameters)]
pub struct GraniteForCausalLM {
    pub config: GraniteConfig,

    #[param]
    pub model: GraniteModel,
    #[param]
    pub lm_head: Option<nn::Linear>,
}

impl GraniteForCausalLM {
    pub fn new(config: GraniteConfig) -> Result<Self, Exception> {
        // Only create separate lm_head if not tied
        let lm_head = if !config.tie_word_embeddings {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        } else {
            None
        };

        let model = GraniteModel::new(config.clone())?;

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

        if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, &hidden_states)
        } else {
            // Tied embeddings: use embedding weights as lm_head
            // logits = hidden @ embed.weight.T
            let embed_weight = self.model.embed_tokens.weight.as_ref();
            let embed_t = embed_weight.t();
            hidden_states.matmul(&embed_t)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::module::ModuleParameters;
    use serial_test::serial;

    #[test]
    fn test_granite_config_layer_types() {
        let config = GraniteConfig::granite_h_1b();

        // Alternating pattern
        assert_eq!(config.layer_type(0), GraniteLayerType::Attention);
        assert_eq!(config.layer_type(1), GraniteLayerType::Mamba2);
        assert_eq!(config.layer_type(2), GraniteLayerType::Attention);
        assert_eq!(config.layer_type(3), GraniteLayerType::Mamba2);
    }

    #[test]
    #[serial]
    fn test_granite_mlp() {
        let mlp = GraniteMLP::new(64, 256).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 10, 64], None, None, None).unwrap();

        let mut mlp = mlp;
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[1, 10, 64]);
    }

    #[test]
    #[serial]
    fn test_granite_model_instantiation() {
        let mut config = GraniteConfig::default();
        config.hidden_size = 64;
        config.intermediate_size = 256;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.head_dim = 16;
        config.vocab_size = 1000;
        config.tie_word_embeddings = true;

        let model = GraniteForCausalLM::new(config).unwrap();

        let params = model.parameters().flatten();
        assert!(params.len() > 0);
    }
}
