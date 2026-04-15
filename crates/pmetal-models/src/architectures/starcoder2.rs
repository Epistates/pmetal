//! StarCoder2 architecture implementation.
//!
//! StarCoder2 is the SOTA benchmark for code generation as of Q1 2026.
//! Features:
//! - Grouped Query Attention (GQA)
//! - Sliding Window Attention (SWA)
//! - Rotary Position Embeddings (RoPE)
//! - Optimized for Fill-in-the-Middle (FIM) training

use crate::architectures::llama::{LlamaAttention, LlamaConfig};
// ModuleParameters derive via impl_module_params!
use pmetal_bridge::compat::{Array, Exception, Module, ModuleParameters, nn};
use pmetal_bridge::impl_module_params;
use pmetal_core::ModelConfig;
use pmetal_mlx::Builder;
use serde::{Deserialize, Serialize};

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
#[derive(Debug)]
pub struct StarCoder2MLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}
impl_module_params!(StarCoder2MLP; gate_proj, up_proj, down_proj);

impl StarCoder2MLP {
    pub fn new(config: &StarCoder2Config) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
                .bias(false)
                .build()?,
            up_proj: nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
                .bias(false)
                .build()?,
            down_proj: nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
                .bias(false)
                .build()?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = nn::silu(&gate).multiply(&up);
        Ok(self.down_proj.forward(&activated))
    }
}

/// StarCoder2 Layer block.
#[derive(Debug)]
pub struct StarCoder2Layer {
    pub attention: LlamaAttention,
    pub mlp: StarCoder2MLP,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}
impl_module_params!(StarCoder2Layer; attention, mlp, input_layernorm, post_attention_layernorm);

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
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        mut cache: Option<(&mut pmetal_mlx::kv_cache::KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let h = self.input_layernorm.forward(x);
        let attn_out = self.attention.forward_with_cache(
            &h,
            mask,
            cache.as_mut().map(|(c, i)| (&mut **c, *i)),
        )?;
        let x = x.add(&attn_out);
        let h = self.post_attention_layernorm.forward(&x);
        let mlp_out = self.mlp.forward(&h)?;
        Ok(x.add(&mlp_out))
    }
}

/// StarCoder2 Model.
#[derive(Debug)]
pub struct StarCoder2Model {
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<StarCoder2Layer>,
    pub norm: nn::RmsNorm,
    pub lm_head: nn::Linear,
    pub config: StarCoder2Config,
}
impl_module_params!(StarCoder2Model; embed_tokens, layers, norm, lm_head);

impl StarCoder2Model {
    pub fn new(config: StarCoder2Config) -> Result<Self, Exception> {
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| StarCoder2Layer::new(&config, i))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size, config.hidden_size)?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            lm_head: nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                .bias(false)
                .build()?,
            config,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut pmetal_mlx::kv_cache::KVCache>,
    ) -> Result<Array, Exception> {
        self.forward_with_capture(input_ids, mask, cache, None)
    }

    /// Forward pass with optional DFlash hidden-state capture.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut pmetal_mlx::kv_cache::KVCache>,
        mut capture: Option<&mut pmetal_mlx::speculative::SpecCapture>,
    ) -> Result<Array, Exception> {
        let mut x = self.embed_tokens.forward(input_ids);
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x, mask, cache.as_mut().map(|c| (&mut **c, i)))?;
            if let Some(buf) = capture.as_deref_mut()
                && buf.wants_hidden_for(i)
            {
                buf.record_hidden(i, x.clone());
            }
        }
        x = self.norm.forward(&x);
        Ok(self.lm_head.forward(&x))
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
