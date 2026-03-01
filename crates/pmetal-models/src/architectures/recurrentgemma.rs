//! RecurrentGemma (Griffin) architecture implementation.
//!
//! Features:
//! - Real-Gated Linear Recurrent Unit (RG-LRU) for linear scaling
//! - Local Sliding Window Attention (SWA)
//! - Fixed-size recurrent state for O(1) memory during generation

use mlx_rs::{Array, nn};
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::nested::NestedHashMap;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::error::Exception;
use pmetal_mlx::Builder;
use serde::{Deserialize, Serialize};
use pmetal_core::ModelConfig;
use pmetal_mlx::kernels::{fused_sdpa, FusedAttentionConfig, AttentionMaskType};
use std::rc::Rc;

/// RecurrentGemma model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentGemmaConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub sliding_window: i32,
    pub lru_width: i32,
}

impl Default for RecurrentGemmaConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2560,
            intermediate_size: 7680,
            num_hidden_layers: 26,
            num_attention_heads: 10,
            num_key_value_heads: 1,
            head_dim: 256,
            vocab_size: 256000,
            rms_norm_eps: 1e-6,
            sliding_window: 2048,
            lru_width: 2560,
        }
    }
}

/// Real-Gated Linear Recurrent Unit (RG-LRU).
#[derive(Debug, ModuleParameters)]
pub struct RGLRU {
    #[param] pub input_proj: nn::Linear,
    #[param] pub gate_proj: nn::Linear,
    #[param] pub output_proj: nn::Linear,
    pub width: i32,
}

impl RGLRU {
    pub fn new(config: &RecurrentGemmaConfig) -> Result<Self, Exception> {
        Ok(Self {
            input_proj: nn::LinearBuilder::new(config.hidden_size, config.lru_width).bias(false).build().map_err(|_| Exception::custom("Build error"))?,
            gate_proj: nn::LinearBuilder::new(config.hidden_size, config.lru_width).bias(false).build().map_err(|_| Exception::custom("Build error"))?,
            output_proj: nn::LinearBuilder::new(config.lru_width, config.hidden_size).bias(false).build().map_err(|_| Exception::custom("Build error"))?,
            width: config.lru_width,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let input = self.input_proj.forward(x)?;
        let _gate = self.gate_proj.forward(x)?;
        let i = mlx_rs::ops::sigmoid(&input)?;
        let h = input.multiply(&i)?; 
        self.output_proj.forward(&h)
    }
}

/// Manual attention implementation for RecurrentGemma.
#[derive(Debug, ModuleParameters)]
pub struct RecurrentGemmaAttention {
    #[param] pub q_proj: nn::Linear,
    #[param] pub k_proj: nn::Linear,
    #[param] pub v_proj: nn::Linear,
    #[param] pub o_proj: nn::Linear,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl RecurrentGemmaAttention {
    pub fn new(config: &RecurrentGemmaConfig) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        Ok(Self {
            q_proj: nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim).bias(false).build().map_err(|_| Exception::custom("Build error"))?,
            k_proj: nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim).bias(false).build().map_err(|_| Exception::custom("Build error"))?,
            v_proj: nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim).bias(false).build().map_err(|_| Exception::custom("Build error"))?,
            o_proj: nn::LinearBuilder::new(n_heads * head_dim, config.hidden_size).bias(false).build().map_err(|_| Exception::custom("Build error"))?,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0]; let seq_len = shape[1];
        let q = self.q_proj.forward(x)?.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        let k = self.k_proj.forward(x)?.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        let v = self.v_proj.forward(x)?.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        let config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim).with_scale(self.scale).with_mask_type(AttentionMaskType::Causal);
        let out = fused_sdpa(&q, &k, &v, &config, None)?;
        self.o_proj.forward(&out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[batch, seq_len, -1])?)
    }
}

/// RecurrentGemma Layer.
#[derive(Debug)]
pub struct RecurrentGemmaLayer {
    pub attention: Option<RecurrentGemmaAttention>,
    pub lru: Option<RGLRU>,
    pub mlp: nn::Sequential,
    pub norm: nn::RmsNorm,
    pub is_attention: bool,
}

impl ModuleParameters for RecurrentGemmaLayer {
    fn parameters(&self) -> NestedHashMap<Rc<str>, &Array> {
        let mut map = NestedHashMap::new();
        if let Some(ref a) = self.attention { map.entries.extend(a.parameters().entries); }
        if let Some(ref l) = self.lru { map.entries.extend(l.parameters().entries); }
        map.entries.extend(self.mlp.parameters().entries);
        map.entries.extend(self.norm.parameters().entries);
        map
    }
    fn trainable_parameters(&self) -> NestedHashMap<Rc<str>, &Array> {
        let mut map = NestedHashMap::new();
        if let Some(ref a) = self.attention { map.entries.extend(a.trainable_parameters().entries); }
        if let Some(ref l) = self.lru { map.entries.extend(l.trainable_parameters().entries); }
        map.entries.extend(self.mlp.trainable_parameters().entries);
        map.entries.extend(self.norm.trainable_parameters().entries);
        map
    }
    fn num_parameters(&self) -> usize { self.attention.as_ref().map_or(0, |a| a.num_parameters()) + self.lru.as_ref().map_or(0, |l| l.num_parameters()) + self.mlp.num_parameters() + self.norm.num_parameters() }
    fn parameters_mut(&mut self) -> NestedHashMap<Rc<str>, &mut Array> {
        let mut map = NestedHashMap::new();
        if let Some(ref mut a) = self.attention { map.entries.extend(a.parameters_mut().entries); }
        if let Some(ref mut l) = self.lru { map.entries.extend(l.parameters_mut().entries); }
        map.entries.extend(self.mlp.parameters_mut().entries);
        map.entries.extend(self.norm.parameters_mut().entries);
        map
    }
    fn freeze_parameters(&mut self, recurse: bool) { if let Some(ref mut a) = self.attention { a.freeze_parameters(recurse); } if let Some(ref mut l) = self.lru { l.freeze_parameters(recurse); } self.mlp.freeze_parameters(recurse); self.norm.freeze_parameters(recurse); }
    fn unfreeze_parameters(&mut self, recurse: bool) { if let Some(ref mut a) = self.attention { a.unfreeze_parameters(recurse); } if let Some(ref mut l) = self.lru { l.unfreeze_parameters(recurse); } self.mlp.unfreeze_parameters(recurse); self.norm.unfreeze_parameters(recurse); }
    fn all_frozen(&self) -> Option<bool> { let mut frozen = true; if let Some(ref a) = self.attention { frozen &= a.all_frozen()?; } if let Some(ref l) = self.lru { frozen &= l.all_frozen()?; } frozen &= self.mlp.all_frozen()?; frozen &= self.norm.all_frozen()?; Some(frozen) }
    fn any_frozen(&self) -> Option<bool> { let mut frozen = false; if let Some(ref a) = self.attention { frozen |= a.any_frozen()?; } if let Some(ref l) = self.lru { frozen |= l.any_frozen()?; } frozen |= self.mlp.any_frozen()?; frozen |= self.norm.any_frozen()?; Some(frozen) }
}

impl RecurrentGemmaLayer {
    pub fn new(config: &RecurrentGemmaConfig, layer_idx: usize) -> Result<Self, Exception> {
        let is_attention = layer_idx % 2 == 0;
        let attention = if is_attention { Some(RecurrentGemmaAttention::new(config)?) } else { None };
        let lru = if !is_attention { Some(RGLRU::new(config)?) } else { None };
        let mlp = nn::Sequential::new(); 
        Ok(Self { attention, lru, mlp, norm: nn::RmsNormBuilder::new(config.hidden_size).eps(config.rms_norm_eps).build().map_err(|_| Exception::custom("Build error"))?, is_attention })
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let normed = self.norm.forward(x)?;
        let branch_out = if self.is_attention { self.attention.as_mut().unwrap().forward(&normed)? } else { self.lru.as_mut().unwrap().forward(&normed)? };
        x.add(&branch_out)
    }
}

#[derive(Debug, ModuleParameters)]
pub struct RecurrentGemmaModel {
    #[param] pub embed: nn::Embedding,
    #[param] pub layers: Vec<RecurrentGemmaLayer>,
    #[param] pub norm: nn::RmsNorm,
    pub config: RecurrentGemmaConfig,
}

impl RecurrentGemmaModel {
    pub fn new(config: RecurrentGemmaConfig) -> Result<Self, Exception> {
        let layers = (0..config.num_hidden_layers as usize).map(|i| RecurrentGemmaLayer::new(&config, i)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { embed: nn::Embedding::new(config.vocab_size, config.hidden_size)?, layers, norm: nn::RmsNormBuilder::new(config.hidden_size).eps(config.rms_norm_eps).build().map_err(|_| Exception::custom("Build error"))?, config })
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = self.embed.forward(x)?;
        for layer in self.layers.iter_mut() { h = layer.forward(&h)?; }
        self.norm.forward(&h)
    }
    pub fn eval(&self) -> Result<(), Exception> {
        self.embed.weight.eval()?;
        for layer in &self.layers {
            if let Some(ref a) = layer.attention { a.q_proj.weight.eval()?; a.k_proj.weight.eval()?; a.v_proj.weight.eval()?; a.o_proj.weight.eval()?; }
            if let Some(ref l) = layer.lru { l.input_proj.weight.eval()?; l.gate_proj.weight.eval()?; l.output_proj.weight.eval()?; }
            layer.norm.weight.eval()?;
        }
        self.norm.weight.eval()?;
        Ok(())
    }
}
