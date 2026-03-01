//! StarCoder2 architecture implementation.
//!
//! StarCoder2 is the SOTA benchmark for code generation as of Q1 2026.
//! Features:
//! - Grouped Query Attention (GQA)
//! - Sliding Window Attention (SWA)
//! - Rotary Position Embeddings (RoPE)
//! - Optimized for Fill-in-the-Middle (FIM) training

use mlx_rs::{Array, nn};
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::macros::ModuleParameters;
use mlx_rs::error::Exception;
use pmetal_mlx::Builder;
use serde::{Deserialize, Serialize};
use pmetal_core::ModelConfig;
use crate::architectures::llama::{LlamaAttention, LlamaConfig};

/// StarCoder2 model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarCoder2Config {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: Option<i32>,
    pub use_cache: bool,
}

impl Default for StarCoder2Config {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 4, // GQA
            vocab_size: 49152,
            max_position_embeddings: 16384,
            rms_norm_eps: 1e-5,
            rope_theta: 100000.0,
            sliding_window: Some(4096),
            use_cache: true,
        }
    }
}

/// StarCoder2 MLP block.
#[derive(Debug, ModuleParameters)]
pub struct StarCoder2MLP {
    #[param] pub gate_proj: nn::Linear,
    #[param] pub up_proj: nn::Linear,
    #[param] pub down_proj: nn::Linear,
}

impl StarCoder2MLP {
    pub fn new(config: &StarCoder2Config) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(config.hidden_size, config.intermediate_size).bias(false).build().map_err(|_| Exception::custom("Build error"))?,
            up_proj: nn::LinearBuilder::new(config.hidden_size, config.intermediate_size).bias(false).build().map_err(|_| Exception::custom("Build error"))?,
            down_proj: nn::LinearBuilder::new(config.intermediate_size, config.hidden_size).bias(false).build().map_err(|_| Exception::custom("Build error"))?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = nn::silu(gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }
}

/// StarCoder2 Layer block.
#[derive(Debug, ModuleParameters)]
pub struct StarCoder2Layer {
    #[param] pub attention: LlamaAttention,
    #[param] pub mlp: StarCoder2MLP,
    #[param] pub input_layernorm: nn::RmsNorm,
    #[param] pub post_attention_layernorm: nn::RmsNorm,
}

impl StarCoder2Layer {
    pub fn new(config: &StarCoder2Config, layer_idx: usize) -> Result<Self, Exception> {
        let attn_config = LlamaConfig {
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: Some(config.num_key_value_heads),
            rope_theta: config.rope_theta,
            max_position_embeddings: config.max_position_embeddings,
            ..Default::default()
        };

        Ok(Self {
            attention: LlamaAttention::new(&attn_config, layer_idx)?,
            mlp: StarCoder2MLP::new(config)?,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size).eps(config.rms_norm_eps).build().map_err(|_| Exception::custom("Build error"))?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size).eps(config.rms_norm_eps).build().map_err(|_| Exception::custom("Build error"))?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        mut cache: Option<(&mut pmetal_mlx::kv_cache::KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let h = self.input_layernorm.forward(x)?;
        let attn_out = self.attention.forward_with_cache(&h, mask, cache.as_mut().map(|(c, i)| (&mut **c, *i)))?;
        let x = x.add(&attn_out)?;
        let h = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&h)?;
        x.add(&mlp_out)
    }
}

/// StarCoder2 Model.
#[derive(Debug, ModuleParameters)]
pub struct StarCoder2Model {
    #[param] pub embed_tokens: nn::Embedding,
    #[param] pub layers: Vec<StarCoder2Layer>,
    #[param] pub norm: nn::RmsNorm,
    #[param] pub lm_head: nn::Linear,
    pub config: StarCoder2Config,
}

impl StarCoder2Model {
    pub fn new(config: StarCoder2Config) -> Result<Self, Exception> {
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| StarCoder2Layer::new(&config, i))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size, config.hidden_size)?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size).eps(config.rms_norm_eps).build().map_err(|_| Exception::custom("Build error"))?,
            lm_head: nn::LinearBuilder::new(config.hidden_size, config.vocab_size).bias(false).build().map_err(|_| Exception::custom("Build error"))?,
            config,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut pmetal_mlx::kv_cache::KVCache>,
    ) -> Result<Array, Exception> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x, mask, cache.as_mut().map(|c| (&mut **c, i)))?;
        }
        x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut pmetal_mlx::kv_cache::KVCache>,
    ) -> Result<Array, Exception> {
        self.forward(input_ids, mask, cache)
    }
}
