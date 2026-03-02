//! Qwen3 model architecture.
//!
//! Implementation of the Qwen3 architecture (standard dense transformer)
//! optimized for Apple Silicon.
//!
//! Key differences from Qwen2:
//! - RMSNorm applied to Q and K before RoPE (q_norm, k_norm)
//! - No bias in attention projections
//! - Per-layer sliding window configuration via `layer_types`

use std::collections::HashMap;

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt},
    nn,
};
use serde::{Deserialize, Serialize};

use crate::traits::ModelConfig;
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopeScaling, apply_rope},
};
use pmetal_mlx::kv_cache::KVCache;

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
    /// Intermediate size for MLP.
    pub intermediate_size: i32,
    /// Number of hidden layers.
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA).
    pub num_key_value_heads: Option<i32>,
    /// Head dimension.
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    /// Maximum position embeddings.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// RMS norm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE theta base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Whether to use sliding window attention.
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Sliding window size.
    pub sliding_window: Option<i32>,
    /// Max number of layers using sliding window.
    pub max_window_layers: Option<i32>,
    /// Layer types (e.g., "dense", "sliding_window").
    pub layer_types: Option<Vec<String>>,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Hidden activation function.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<std::collections::HashMap<String, serde_json::Value>>,
}

fn default_model_type() -> String {
    "qwen3".to_string()
}
fn default_vocab_size() -> i32 {
    151936
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
fn default_head_dim() -> i32 {
    128
}
fn default_hidden_act() -> String {
    "silu".to_string()
}

impl ModelConfig for Qwen3Config {
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
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
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

impl Qwen3Config {
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
            if let Some(layer_type) = layer_types.get(layer_idx) {
                return layer_type == "sliding_window";
            }
        }

        // Default: layers at or above max_window_layers use sliding window (matches HF transformers)
        if let Some(max_layers) = self.max_window_layers {
            return self.use_sliding_window && (layer_idx as i32) >= max_layers;
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
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: None,
            layer_types: None,
            tie_word_embeddings: true,
            hidden_act: "silu".to_string(),
            rope_scaling: None,
        }
    }
}

/// Qwen3 MLP.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MLP {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl Qwen3MLP {
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
}

impl Module<Array> for Qwen3MLP {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: Array) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(&x)?;
        let up = self.up_proj.forward(&x)?;
        self.down_proj.forward(&nn::silu(&gate)?.multiply(&up)?)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// Qwen3 Attention with Q/K RMSNorm before RoPE.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3Attention {
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
    /// Q normalization (Qwen3-specific: RMSNorm over head_dim before RoPE).
    #[param]
    pub q_norm: nn::RmsNorm,
    /// K normalization (Qwen3-specific: RMSNorm over head_dim before RoPE).
    #[param]
    pub k_norm: nn::RmsNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,
    /// RoPE position scale (from rope_scaling config).
    pub rope_scale: f32,
    /// Effective RoPE base after scaling.
    pub effective_base: f32,
    pub use_sliding_window: bool,
    pub sliding_window: Option<i32>,
}

impl Qwen3Attention {
    pub fn new(config: &Qwen3Config, use_sliding_window: bool) -> Result<Self, Exception> {
        let head_dim = config.get_head_dim();
        let n_kv_heads = config.num_kv_heads();

        // Parse rope_scaling from config
        let rope_scaling = config
            .rope_scaling
            .as_ref()
            .map(|map| RopeScaling::from_config_map(map))
            .unwrap_or(RopeScaling::None);
        let rope_scale = rope_scaling.scale();
        let effective_base = rope_scaling.effective_base(config.rope_theta, head_dim);

        // Qwen3 uses no bias in attention projections
        let q_proj =
            nn::LinearBuilder::new(config.hidden_size, config.num_attention_heads * head_dim)
                .bias(false)
                .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj =
            nn::LinearBuilder::new(config.num_attention_heads * head_dim, config.hidden_size)
                .bias(false)
                .build()?;

        // Per-head RMSNorm over head_dim (Qwen3-specific)
        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_heads: config.num_attention_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_theta: config.rope_theta,
            rope_scale,
            effective_base,
            use_sliding_window,
            sliding_window: config.sliding_window,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [B, L, heads, head_dim] for per-head normalization
        let mut q = q.reshape(&[b, l, self.n_heads, self.head_dim])?;
        let mut k = k.reshape(&[b, l, self.n_kv_heads, self.head_dim])?;
        let v = v.reshape(&[b, l, self.n_kv_heads, self.head_dim])?;

        // Apply Q/K RMSNorm before RoPE (Qwen3-specific)
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;

        // Transpose to [B, heads, L, head_dim]
        q = q.transpose_axes(&[0, 2, 1, 3])?;
        k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE with cache offset
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q = apply_rope(
            &q,
            self.head_dim,
            false,
            self.effective_base,
            self.rope_scale,
            offset,
        )?;
        let k = apply_rope(
            &k,
            self.head_dim,
            false,
            self.effective_base,
            self.rope_scale,
            offset,
        )?;

        // Update KV cache with layer_idx (second element of tuple)
        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        // Fused SDPA with GQA-native path
        let mask_type = if mask.is_some() {
            AttentionMaskType::None
        } else if self.use_sliding_window {
            if let Some(window_size) = self.sliding_window {
                AttentionMaskType::SlidingWindow(window_size)
            } else {
                AttentionMaskType::Causal
            }
        } else {
            AttentionMaskType::Causal
        };
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(mask_type);

        let out = fused_sdpa(&q, &k, &v, &attn_config, mask)?;
        let out =
            out.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, l, self.n_heads * self.head_dim])?;
        self.o_proj.forward(&out)
    }
}

/// Qwen3 Layer.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3Layer {
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub self_attn: Qwen3Attention,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub mlp: Qwen3MLP,
}

impl Qwen3Layer {
    pub fn new(config: &Qwen3Config, use_sliding_window: bool) -> Result<Self, Exception> {
        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let self_attn = Qwen3Attention::new(config, use_sliding_window)?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let mlp = Qwen3MLP::new(config)?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let h = x.add(&self.self_attn.forward(
            &self.input_layernorm.forward(x)?,
            mask,
            cache,
        )?)?;
        let mlp_in = self.post_attention_layernorm.forward(&h)?;
        h.add(&self.mlp.forward(mlp_in)?)
    }
}

/// Qwen3 Model.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3Model {
    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<Qwen3Layer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen3Model {
    pub fn new(config: &Qwen3Config) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| {
                let use_sliding = config.use_sliding_window_at(i);
                Qwen3Layer::new(config, use_sliding)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_cache = cache.as_mut().map(|c| (&mut **c, i));
            h = layer.forward(&h, mask, layer_cache)?;
        }
        self.norm.forward(&h)
    }
}

/// Qwen3 for Causal LM.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3ForCausalLM {
    #[param]
    pub model: Qwen3Model,
    /// None when `tie_word_embeddings` is true (uses embed_tokens transposed).
    #[param]
    pub lm_head: Option<nn::Linear>,
    pub config: Qwen3Config,
}

impl Qwen3ForCausalLM {
    pub fn new(config: Qwen3Config) -> Result<Self, Exception> {
        let model = Qwen3Model::new(&config)?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        };
        Ok(Self {
            model,
            lm_head,
            config,
        })
    }

    fn lm_head_forward(&mut self, h: &Array) -> Result<Array, Exception> {
        if let Some(ref mut lm_head) = self.lm_head {
            lm_head.forward(h)
        } else {
            self.model.embed_tokens.as_linear(h)
        }
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let h = self.model.forward(input_ids, mask, None)?;
        self.lm_head_forward(&h)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let h = self.model.forward(input_ids, mask, cache)?;
        self.lm_head_forward(&h)
    }
}

impl crate::traits::CausalLMModel for Qwen3ForCausalLM {
    type Config = Qwen3Config;

    fn new(config: Self::Config) -> Result<Self, Exception> {
        Self::new(config)
    }

    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        Self::forward(self, input_ids, mask)
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), Exception> {
        for (name, weight) in weights {
            if name == "model.embed_tokens.weight" {
                self.model.embed_tokens.weight = mlx_rs::module::Param::new(weight.clone());
            } else if name == "lm_head.weight" {
                if let Some(ref mut lm_head) = self.lm_head {
                    lm_head.weight = mlx_rs::module::Param::new(weight.clone());
                }
            } else if name == "model.norm.weight" {
                self.model.norm.weight = mlx_rs::module::Param::new(weight.clone());
            } else if name.starts_with("model.layers.") {
                let parts: Vec<&str> = name.split('.').collect();
                let i: usize = parts[2].parse().map_err(|_| {
                    Exception::custom(format!("Invalid layer index in weight key: {}", name))
                })?;
                if i >= self.model.layers.len() {
                    continue;
                }
                let suffix = parts[3..].join(".");
                let layer = &mut self.model.layers[i];
                match suffix.as_str() {
                    "input_layernorm.weight" => {
                        layer.input_layernorm.weight = mlx_rs::module::Param::new(weight.clone())
                    }
                    "post_attention_layernorm.weight" => {
                        layer.post_attention_layernorm.weight =
                            mlx_rs::module::Param::new(weight.clone())
                    }
                    "self_attn.q_proj.weight" => {
                        layer.self_attn.q_proj.weight = mlx_rs::module::Param::new(weight.clone())
                    }
                    "self_attn.k_proj.weight" => {
                        layer.self_attn.k_proj.weight = mlx_rs::module::Param::new(weight.clone())
                    }
                    "self_attn.v_proj.weight" => {
                        layer.self_attn.v_proj.weight = mlx_rs::module::Param::new(weight.clone())
                    }
                    "self_attn.o_proj.weight" => {
                        layer.self_attn.o_proj.weight = mlx_rs::module::Param::new(weight.clone())
                    }
                    "self_attn.q_norm.weight" => {
                        layer.self_attn.q_norm.weight = mlx_rs::module::Param::new(weight.clone())
                    }
                    "self_attn.k_norm.weight" => {
                        layer.self_attn.k_norm.weight = mlx_rs::module::Param::new(weight.clone())
                    }
                    "mlp.gate_proj.weight" => {
                        layer.mlp.gate_proj.weight = mlx_rs::module::Param::new(weight.clone())
                    }
                    "mlp.up_proj.weight" => {
                        layer.mlp.up_proj.weight = mlx_rs::module::Param::new(weight.clone())
                    }
                    "mlp.down_proj.weight" => {
                        layer.mlp.down_proj.weight = mlx_rs::module::Param::new(weight.clone())
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn eval(&self) -> Result<(), Exception> {
        ModuleParametersExt::eval(self)
    }
}
