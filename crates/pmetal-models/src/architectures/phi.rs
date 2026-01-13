//! Phi model architecture (Phi-3, Phi-3.5, Phi-4).
//!
//! Phi models are compact, high-quality language models from Microsoft.
//! Key features:
//! - SuRoPE (Scaled Uniform RoPE) for extended context
//! - Partial RoPE (applied to subset of head dimensions)
//! - Uses SwiGLU or GELU activation
//! - QKV bias in attention
//!
//! ## Supported Models
//!
//! - `phi-3-mini-4k-instruct` (3.8B, 4K context)
//! - `phi-3-mini-128k-instruct` (3.8B, 128K context)
//! - `phi-3-small-8k-instruct` (7B, 8K context)
//! - `phi-3-medium-4k-instruct` (14B, 4K context)
//! - `phi-3.5-mini-instruct` (3.8B, 128K context)
//! - `phi-4` (14B, 16K context)

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn::{self, Embedding, Linear, RopeBuilder},
    ops::indexing::IndexOp,
    Array,
};
use pmetal_mlx::kernels::{fused_sdpa, rope::apply_rope, AttentionMaskType, FusedAttentionConfig};
use pmetal_mlx::kv_cache::KVCache;

use crate::traits::{CausalLMModel, ModelConfig};
use std::collections::HashMap;

/// Phi model configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PhiConfig {
    /// Model type identifier.
    pub model_type: String,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Hidden dimension.
    pub hidden_size: i32,
    /// Intermediate (FFN) dimension.
    pub intermediate_size: i32,
    /// Number of hidden layers.
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA).
    pub num_key_value_heads: i32,
    /// Maximum sequence length.
    pub max_position_embeddings: i32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Partial RoPE dimension (how much of head_dim uses RoPE).
    pub partial_rotary_factor: f32,
    /// RMS norm epsilon.
    pub rms_norm_eps: f32,
    /// Whether to use QKV bias.
    pub qkv_bias: bool,
    /// Activation function type.
    pub hidden_act: PhiActivation,
    /// Sliding window attention size (None for full attention).
    pub sliding_window: Option<i32>,
    /// Layer norm type.
    pub layer_norm_type: LayerNormType,
    /// Original max position embeddings (for RoPE scaling).
    pub original_max_position_embeddings: Option<i32>,
    /// RoPE scaling configuration.
    pub rope_scaling: Option<PhiRopeScaling>,
    /// Tie word embeddings.
    pub tie_word_embeddings: bool,
}

/// Activation function type for Phi models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PhiActivation {
    /// SwiGLU activation (Phi-3).
    #[default]
    #[serde(rename = "silu", alias = "swiglu")]
    SwiGLU,
    /// GELU approximation (Phi-2).
    #[serde(rename = "gelu_approx")]
    GeluApprox,
    /// GELU exact.
    #[serde(rename = "gelu")]
    GeluExact,
}

/// Layer normalization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum LayerNormType {
    /// RMS LayerNorm (default for Phi-3+).
    #[default]
    #[serde(rename = "rms_norm")]
    RmsNorm,
    /// Standard LayerNorm.
    #[serde(rename = "layer_norm")]
    LayerNorm,
}

/// RoPE scaling configuration for Phi.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhiRopeScaling {
    /// Scaling type.
    pub scaling_type: String,
    /// Short factor.
    pub short_factor: Vec<f32>,
    /// Long factor.
    pub long_factor: Vec<f32>,
}

impl Default for PhiConfig {
    fn default() -> Self {
        Self::phi3_mini()
    }
}

impl PhiConfig {
    /// Phi-3-mini configuration (3.8B, 4K context).
    pub fn phi3_mini() -> Self {
        Self {
            model_type: "phi3".to_string(),
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            rms_norm_eps: 1e-5,
            qkv_bias: false,
            hidden_act: PhiActivation::SwiGLU,
            sliding_window: None,
            layer_norm_type: LayerNormType::RmsNorm,
            original_max_position_embeddings: None,
            rope_scaling: None,
            tie_word_embeddings: false,
        }
    }

    /// Phi-3-mini-128k configuration (3.8B, 128K context).
    pub fn phi3_mini_128k() -> Self {
        Self {
            model_type: "phi3".to_string(),
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 131072,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            rms_norm_eps: 1e-5,
            qkv_bias: false,
            hidden_act: PhiActivation::SwiGLU,
            sliding_window: None,
            layer_norm_type: LayerNormType::RmsNorm,
            original_max_position_embeddings: Some(4096),
            rope_scaling: None, // SuRoPE handled separately
            tie_word_embeddings: false,
        }
    }

    /// Phi-3.5-mini configuration (3.8B, 128K context).
    pub fn phi35_mini() -> Self {
        Self {
            model_type: "phi3".to_string(),
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 131072,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            rms_norm_eps: 1e-5,
            qkv_bias: false,
            hidden_act: PhiActivation::SwiGLU,
            sliding_window: None,
            layer_norm_type: LayerNormType::RmsNorm,
            original_max_position_embeddings: Some(4096),
            rope_scaling: None,
            tie_word_embeddings: false,
        }
    }

    /// Phi-3-medium configuration (14B, 4K context).
    pub fn phi3_medium() -> Self {
        Self {
            model_type: "phi3".to_string(),
            vocab_size: 32064,
            hidden_size: 5120,
            intermediate_size: 17920,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 10,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.4,
            rms_norm_eps: 1e-5,
            qkv_bias: false,
            hidden_act: PhiActivation::SwiGLU,
            sliding_window: None,
            layer_norm_type: LayerNormType::RmsNorm,
            original_max_position_embeddings: None,
            rope_scaling: None,
            tie_word_embeddings: false,
        }
    }

    /// Phi-4 configuration (14B, 16K context).
    pub fn phi4() -> Self {
        Self {
            model_type: "phi3".to_string(),
            vocab_size: 100352,
            hidden_size: 5120,
            intermediate_size: 17920,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 10,
            max_position_embeddings: 16384,
            rope_theta: 250000.0,
            partial_rotary_factor: 0.4,
            rms_norm_eps: 1e-5,
            qkv_bias: true,
            hidden_act: PhiActivation::SwiGLU,
            sliding_window: None,
            layer_norm_type: LayerNormType::RmsNorm,
            original_max_position_embeddings: None,
            rope_scaling: None,
            tie_word_embeddings: false,
        }
    }

    /// Get head dimension.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    /// Get RoPE dimension (partial).
    pub fn rope_dim(&self) -> i32 {
        ((self.head_dim() as f32) * self.partial_rotary_factor) as i32
    }
}

/// RMS LayerNorm for Phi.
#[derive(Debug, ModuleParameters)]
pub struct PhiRMSNorm {
    #[param]
    pub weight: Param<Array>,
    pub eps: f32,
}

impl PhiRMSNorm {
    /// Create a new RMS LayerNorm.
    pub fn new(hidden_size: i32, eps: f32) -> Self {
        let weight = Param::new(Array::ones::<f32>(&[hidden_size]).unwrap());
        Self { weight, eps }
    }
}

impl PhiRMSNorm {
    /// Forward pass for RMS LayerNorm.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let variance = x.square()?.mean_axis(-1, Some(true))?;
        let eps = Array::from_f32(self.eps);
        let x_normed = x.divide(&variance.add(&eps)?.sqrt()?)?;
        x_normed.multiply(&*self.weight)
    }
}

/// Phi attention with partial RoPE.
#[derive(Debug, ModuleParameters)]
pub struct PhiAttention {
    #[param]
    pub q_proj: Linear,
    #[param]
    pub k_proj: Linear,
    #[param]
    pub v_proj: Linear,
    #[param]
    pub o_proj: Linear,
    pub rope: mlx_rs::nn::Rope,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,
}

impl PhiAttention {
    /// Create a new Phi attention layer.
    pub fn new(config: &PhiConfig) -> Self {
        let head_dim = config.head_dim();
        let rope_dim = config.rope_dim();
        let rope_theta = config.rope_theta;

        let q_proj =
            nn::LinearBuilder::new(config.hidden_size, config.num_attention_heads * head_dim)
                .bias(config.qkv_bias)
                .build()
                .unwrap();
        let k_proj =
            nn::LinearBuilder::new(config.hidden_size, config.num_key_value_heads * head_dim)
                .bias(config.qkv_bias)
                .build()
                .unwrap();
        let v_proj =
            nn::LinearBuilder::new(config.hidden_size, config.num_key_value_heads * head_dim)
                .bias(config.qkv_bias)
                .build()
                .unwrap();
        let o_proj =
            nn::LinearBuilder::new(config.num_attention_heads * head_dim, config.hidden_size)
                .bias(false)
                .build()
                .unwrap();

        let rope = RopeBuilder::new(rope_dim)
            .traditional(false)
            .base(rope_theta)
            .scale(1.0)
            .build()
            .unwrap();

        let scale = 1.0 / (head_dim as f32).sqrt();

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            head_dim,
            rope_dim,
            scale,
            rope_theta,
        }
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
        let (batch, seq_len, _) = (x.dim(0), x.dim(1), x.dim(2));

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, seq, n_heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // Apply partial RoPE
        let (q_rope, q_pass) = self.split_rotary(&q)?;
        let (k_rope, k_pass) = self.split_rotary(&k)?;

        let (q_rope, k_rope) = if let Some((ref cache_ref, _)) = cache {
            let offset = cache_ref.rope_offset();
            let qr = apply_rope(&q_rope, self.rope_dim, false, self.rope_theta, 1.0, offset)?;
            let kr = apply_rope(&k_rope, self.rope_dim, false, self.rope_theta, 1.0, offset)?;
            (qr, kr)
        } else {
            let qr = self.rope.forward(&q_rope)?;
            let kr = self.rope.forward(&k_rope)?;
            (qr, kr)
        };

        // Concatenate RoPE and pass-through parts
        let q = mlx_rs::ops::concatenate_axis(&[&q_rope, &q_pass], -1)?;
        let k = mlx_rs::ops::concatenate_axis(&[&k_rope, &k_pass], -1)?;

        // Transpose for attention: [batch, n_heads, seq, head_dim]
        let mut q = q.transpose_axes(&[0, 2, 1, 3])?;
        let mut k = k.transpose_axes(&[0, 2, 1, 3])?;
        let mut v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Update KV cache
        if let Some((cache, layer_idx)) = cache {
            let (nk, nv) = cache.update_and_fetch(layer_idx, &k, &v)?;
            k = nk;
            v = nv;
        }

        // Use fused attention
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });

        let attn_output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        // Transpose back and project
        let attn_output = attn_output.transpose_axes(&[0, 2, 1, 3])?;
        let attn_output = attn_output.reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

        self.o_proj.forward(&attn_output)
    }

    /// Split tensor into RoPE and pass-through parts.
    fn split_rotary(&self, x: &Array) -> Result<(Array, Array), Exception> {
        let rope_part = x.index((.., .., .., ..self.rope_dim));
        let pass_part = x.index((.., .., .., self.rope_dim..));
        Ok((rope_part, pass_part))
    }
}

/// Phi MLP with SwiGLU or GELU.
#[derive(Debug, ModuleParameters)]
pub struct PhiMLP {
    #[param]
    pub gate_up_proj: Linear,
    #[param]
    pub down_proj: Linear,
    pub activation: PhiActivation,
    pub intermediate_size: i32,
}

impl PhiMLP {
    /// Create a new Phi MLP.
    pub fn new(config: &PhiConfig) -> Self {
        // For SwiGLU, gate_up_proj projects to 2x intermediate_size (gate + up)
        let proj_size = match config.hidden_act {
            PhiActivation::SwiGLU => config.intermediate_size * 2,
            _ => config.intermediate_size,
        };

        let gate_up_proj = nn::LinearBuilder::new(config.hidden_size, proj_size)
            .bias(false)
            .build()
            .unwrap();
        let down_proj = nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
            .bias(false)
            .build()
            .unwrap();

        Self {
            gate_up_proj,
            down_proj,
            activation: config.hidden_act,
            intermediate_size: config.intermediate_size,
        }
    }
}

impl PhiMLP {
    /// Forward pass through MLP.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let hidden = self.gate_up_proj.forward(x)?;

        let activated = match self.activation {
            PhiActivation::SwiGLU => {
                // Split into gate and up projections
                let gate = hidden.index((.., .., ..self.intermediate_size));
                let up = hidden.index((.., .., self.intermediate_size..));
                // SwiGLU: silu(gate) * up
                let gate_activated = mlx_rs::ops::sigmoid(&gate)?.multiply(&gate)?;
                gate_activated.multiply(&up)?
            }
            PhiActivation::GeluApprox => mlx_rs::nn::gelu(&hidden)?, // gelu_approx not in mlx-rs
            PhiActivation::GeluExact => mlx_rs::nn::gelu(&hidden)?,
        };

        self.down_proj.forward(&activated)
    }
}

/// Phi decoder layer.
#[derive(Debug, ModuleParameters)]
pub struct PhiDecoderLayer {
    #[param]
    pub self_attn: PhiAttention,
    #[param]
    pub mlp: PhiMLP,
    #[param]
    pub input_layernorm: PhiRMSNorm,
    #[param]
    pub post_attention_layernorm: PhiRMSNorm,
}

impl PhiDecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: &PhiConfig) -> Self {
        Self {
            self_attn: PhiAttention::new(config),
            mlp: PhiMLP::new(config),
            input_layernorm: PhiRMSNorm::new(config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: PhiRMSNorm::new(config.hidden_size, config.rms_norm_eps),
        }
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
        // Pre-norm attention
        let residual = x.clone();
        let hidden = self.input_layernorm.forward(x)?;
        let hidden = self.self_attn.forward_with_cache(&hidden, mask, cache)?;
        let hidden = residual.add(&hidden)?;

        // Pre-norm MLP
        let residual = hidden.clone();
        let hidden = self.post_attention_layernorm.forward(&hidden)?;
        let hidden = self.mlp.forward(&hidden)?;
        residual.add(&hidden)
    }
}

/// Phi base model.
#[derive(Debug, ModuleParameters)]
pub struct PhiModel {
    #[param]
    pub embed_tokens: Embedding,
    #[param]
    pub layers: Vec<PhiDecoderLayer>,
    #[param]
    pub norm: PhiRMSNorm,
    pub config: PhiConfig,
}

impl PhiModel {
    /// Create a new Phi model.
    pub fn new(config: PhiConfig) -> Self {
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size).unwrap();
        let layers = (0..config.num_hidden_layers)
            .map(|_| PhiDecoderLayer::new(&config))
            .collect();
        let norm = PhiRMSNorm::new(config.hidden_size, config.rms_norm_eps);

        Self {
            embed_tokens,
            layers,
            norm,
            config,
        }
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
        let mut hidden = self.embed_tokens.forward(input_ids)?;

        // Create causal mask if not provided and not using cache
        let mask = if mask.is_none() && cache.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len)?)
        } else {
            mask.cloned()
        };

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let c = cache.as_deref_mut().map(|c| (c, idx));
            hidden = layer.forward_with_cache(&hidden, mask.as_ref(), c)?;
        }

        self.norm.forward(&hidden)
    }
}

/// Phi for causal language modeling.
#[derive(Debug, ModuleParameters)]
pub struct PhiForCausalLM {
    #[param]
    pub model: PhiModel,
    #[param]
    pub lm_head: Linear,
}

impl PhiForCausalLM {
    /// Create a new Phi causal LM.
    pub fn new(config: PhiConfig) -> Result<Self, Exception> {
        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()
            .unwrap();
        let model = PhiModel::new(config);
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
        let hidden = self.model.forward_with_cache(input_ids, mask, cache)?;
        self.lm_head.forward(&hidden)
    }

    /// Create a KV cache for this model.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        use pmetal_mlx::kv_cache::KVCacheConfig;
        let config = &self.model.config;
        KVCache::new(KVCacheConfig::new(
            config.num_hidden_layers as usize,
            max_seq_len,
            config.num_key_value_heads as usize,
            config.head_dim() as usize,
        ))
    }

    /// Get configuration.
    pub fn config(&self) -> &PhiConfig {
        &self.model.config
    }
}

// Trait implementations
impl ModelConfig for PhiConfig {
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
        self.head_dim()
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

impl CausalLMModel for PhiForCausalLM {
    type Config = PhiConfig;

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
        crate::loader::load_phi_weights(self, weights).map_err(|e| Exception::custom(e.to_string()))
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

    #[test]
    fn test_phi_config_presets() {
        let mini = PhiConfig::phi3_mini();
        assert_eq!(mini.hidden_size, 3072);
        assert_eq!(mini.num_hidden_layers, 32);
        assert_eq!(mini.head_dim(), 96);
        assert_eq!(mini.rope_dim(), 48); // 0.5 * 96

        let medium = PhiConfig::phi3_medium();
        assert_eq!(medium.hidden_size, 5120);
        assert_eq!(medium.num_key_value_heads, 10); // GQA

        let phi4 = PhiConfig::phi4();
        // assert_eq!(phi4.vocab_size, 100352); // Some versions might vary
        assert!(phi4.qkv_bias);
    }

    #[test]
    #[serial]
    fn test_phi_rms_norm() {
        let mut norm = PhiRMSNorm::new(64, 1e-5);
        let x = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();

        let out = norm.forward(&x).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    #[serial]
    fn test_phi_attention() {
        let config = PhiConfig {
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            qkv_bias: false,
            ..PhiConfig::phi3_mini()
        };

        let mut attn = PhiAttention::new(&config);
        let x = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();

        let out = attn.forward(&x, None).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[2, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_phi_mlp_swiglu() {
        let config = PhiConfig {
            hidden_size: 64,
            intermediate_size: 128,
            hidden_act: PhiActivation::SwiGLU,
            ..PhiConfig::phi3_mini()
        };

        let mut mlp = PhiMLP::new(&config);
        let x = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();

        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[2, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_phi_decoder_layer() {
        let config = PhiConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            partial_rotary_factor: 0.5,
            ..PhiConfig::phi3_mini()
        };

        let mut layer = PhiDecoderLayer::new(&config);
        let x = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();

        let out = layer.forward(&x, None).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[2, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_phi_model() {
        let config = PhiConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            partial_rotary_factor: 0.5,
            ..PhiConfig::phi3_mini()
        };

        let mut model = PhiModel::new(config);
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4, 5, 6, 7, 8], &[2, 4]);

        let out = model.forward(&input_ids, None).unwrap();
        out.eval().unwrap();

        assert_eq!(out.shape(), &[2, 4, 64]);
    }

    #[test]
    #[serial]
    fn test_phi_causal_lm() {
        let config = PhiConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            partial_rotary_factor: 0.5,
            ..PhiConfig::phi3_mini()
        };

        let mut model = PhiForCausalLM::new(config.clone()).unwrap();
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4, 5, 6, 7, 8], &[2, 4]);

        let logits = model.forward(&input_ids, None).unwrap();
        logits.eval().unwrap();

        assert_eq!(logits.shape(), &[2, 4, config.vocab_size]);
    }

    #[test]
    fn test_partial_rope() {
        let config = PhiConfig {
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            partial_rotary_factor: 0.5,
            ..PhiConfig::phi3_mini()
        };

        assert_eq!(config.head_dim(), 16);
        assert_eq!(config.rope_dim(), 8); // 50% of head_dim
    }

    #[test]
    #[serial]
    fn test_phi_kv_cache() {
        let config = PhiConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            partial_rotary_factor: 0.5,
            ..PhiConfig::phi3_mini()
        };

        let mut model = PhiForCausalLM::new(config).unwrap();

        // Create cache
        let mut cache = model.create_cache(32);

        // First forward (prompt)
        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model
            .forward_with_cache(&input_ids, None, Some(&mut cache))
            .unwrap();
        logits.eval().unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]);

        // Second forward (incremental)
        let next_token = Array::from_slice(&[5_i32], &[1, 1]);
        let logits = model
            .forward_with_cache(&next_token, None, Some(&mut cache))
            .unwrap();
        logits.eval().unwrap();

        assert_eq!(logits.shape(), &[1, 1, 1000]);
    }

    #[test]
    fn test_phi4_config() {
        let config = PhiConfig::phi4();
        assert_eq!(config.vocab_size, 100352);
        assert!(config.qkv_bias);
        assert_eq!(config.rope_theta, 250000.0);
        assert_eq!(config.partial_rotary_factor, 0.4);
    }
}
