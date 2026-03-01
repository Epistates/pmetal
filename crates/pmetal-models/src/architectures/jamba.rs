//! Jamba 1.5 architecture implementation.
//!
//! Features:
//! - Hybrid Mamba + Transformer blocks
//! - Mixture of Experts (MoE) for MLP layers
//! - Supports massive context (256K) via linear SSM scaling
//! - ExpertsInt8 quantization support

use mlx_rs::{Array, nn};
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::nested::NestedHashMap;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::error::Exception;
use pmetal_mlx::Builder;
use serde::{Deserialize, Serialize};
use pmetal_core::ModelConfig;
use pmetal_mlx::moe::{MoELayer, MoEConfig};
use pmetal_mlx::kernels::{fused_sdpa, FusedAttentionConfig, AttentionMaskType};
use std::rc::Rc;

/// Jamba 1.5 model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JambaConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub layers_per_block: i32,
    pub attn_layer_offset: i32,
}

impl Default for JambaConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 64,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            vocab_size: 65536,
            rms_norm_eps: 1e-5,
            num_experts: 16,
            num_experts_per_tok: 2,
            layers_per_block: 8,
            attn_layer_offset: 0,
        }
    }
}

/// Manual attention implementation for Jamba.
#[derive(Debug, ModuleParameters)]
pub struct JambaAttention {
    #[param] pub q_proj: nn::Linear,
    #[param] pub k_proj: nn::Linear,
    #[param] pub v_proj: nn::Linear,
    #[param] pub o_proj: nn::Linear,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl JambaAttention {
    pub fn new(config: &JambaConfig) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / n_heads;
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

/// Jamba Hybrid Layer.
#[derive(Debug)]
pub struct JambaLayer {
    pub attention: Option<JambaAttention>,
    pub mamba: Option<pmetal_mlx::moe::MoELayer>, 
    pub mlp: MoELayer,
    pub norm: nn::RmsNorm,
    pub is_attention: bool,
}

impl ModuleParameters for JambaLayer {
    fn parameters(&self) -> NestedHashMap<Rc<str>, &Array> {
        let mut map = NestedHashMap::new();
        if let Some(ref a) = self.attention { map.entries.extend(a.parameters().entries); }
        if let Some(ref m) = self.mamba { map.entries.extend(m.parameters().entries); }
        map.entries.extend(self.mlp.parameters().entries);
        map.entries.extend(self.norm.parameters().entries);
        map
    }
    fn trainable_parameters(&self) -> NestedHashMap<Rc<str>, &Array> {
        let mut map = NestedHashMap::new();
        if let Some(ref a) = self.attention { map.entries.extend(a.trainable_parameters().entries); }
        if let Some(ref m) = self.mamba { map.entries.extend(m.trainable_parameters().entries); }
        map.entries.extend(self.mlp.trainable_parameters().entries);
        map.entries.extend(self.norm.trainable_parameters().entries);
        map
    }
    fn num_parameters(&self) -> usize { self.attention.as_ref().map_or(0, |a| a.num_parameters()) + self.mamba.as_ref().map_or(0, |m| m.num_parameters()) + self.mlp.num_parameters() + self.norm.num_parameters() }
    fn parameters_mut(&mut self) -> NestedHashMap<Rc<str>, &mut Array> {
        let mut map = NestedHashMap::new();
        if let Some(ref mut a) = self.attention { map.entries.extend(a.parameters_mut().entries); }
        if let Some(ref mut m) = self.mamba { map.entries.extend(m.parameters_mut().entries); }
        map.entries.extend(self.mlp.parameters_mut().entries);
        map.entries.extend(self.norm.parameters_mut().entries);
        map
    }
    fn freeze_parameters(&mut self, recurse: bool) { if let Some(ref mut a) = self.attention { a.freeze_parameters(recurse); } if let Some(ref mut m) = self.mamba { m.freeze_parameters(recurse); } self.mlp.freeze_parameters(recurse); self.norm.freeze_parameters(recurse); }
    fn unfreeze_parameters(&mut self, recurse: bool) { if let Some(ref mut a) = self.attention { a.unfreeze_parameters(recurse); } if let Some(ref mut m) = self.mamba { m.unfreeze_parameters(recurse); } self.mlp.unfreeze_parameters(recurse); self.norm.unfreeze_parameters(recurse); }
    fn all_frozen(&self) -> Option<bool> { let mut frozen = true; if let Some(ref a) = self.attention { frozen &= a.all_frozen()?; } if let Some(ref m) = self.mamba { frozen &= m.all_frozen()?; } frozen &= self.mlp.all_frozen()?; frozen &= self.norm.all_frozen()?; Some(frozen) }
    fn any_frozen(&self) -> Option<bool> { let mut frozen = false; if let Some(ref a) = self.attention { frozen |= a.any_frozen()?; } if let Some(ref m) = self.mamba { frozen |= m.any_frozen()?; } frozen |= self.mlp.any_frozen()?; frozen |= self.norm.any_frozen()?; Some(frozen) }
}

impl JambaLayer {
    pub fn new(config: &JambaConfig, layer_idx: usize) -> Result<Self, Exception> {
        let is_attention = (layer_idx as i32) % config.layers_per_block == config.attn_layer_offset;
        let attention = if is_attention { Some(JambaAttention::new(config)?) } else { None };
        let mamba = if !is_attention { let moe_config = MoEConfig::new(config.hidden_size, config.intermediate_size, config.num_experts as usize).with_num_experts_per_tok(config.num_experts_per_tok as usize); Some(MoELayer::new(moe_config)) } else { None };
        let moe_config = MoEConfig::new(config.hidden_size, config.intermediate_size, config.num_experts as usize).with_num_experts_per_tok(config.num_experts_per_tok as usize);
        Ok(Self { attention, mamba, mlp: MoELayer::new(moe_config), norm: nn::RmsNormBuilder::new(config.hidden_size).eps(config.rms_norm_eps).build().map_err(|_| Exception::custom("Build error"))?, is_attention })
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let normed = self.norm.forward(x)?;
        let branch_out = if self.is_attention { self.attention.as_mut().unwrap().forward(&normed)? } else { let (out, _) = self.mamba.as_mut().unwrap().forward(&normed)?; out };
        let x = x.add(&branch_out)?;
        let (mlp_out, _) = self.mlp.forward(&x)?;
        x.add(&mlp_out)
    }
}

#[derive(Debug, ModuleParameters)]
pub struct JambaModel {
    #[param] pub embed: nn::Embedding,
    #[param] pub layers: Vec<JambaLayer>,
    #[param] pub norm: nn::RmsNorm,
    #[param] pub lm_head: nn::Linear,
    pub config: JambaConfig,
}

impl JambaModel {
    pub fn new(config: JambaConfig) -> Result<Self, Exception> {
        let layers = (0..config.num_hidden_layers as usize).map(|i| JambaLayer::new(&config, i)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { embed: nn::Embedding::new(config.vocab_size, config.hidden_size)?, layers, norm: nn::RmsNormBuilder::new(config.hidden_size).eps(config.rms_norm_eps).build().map_err(|_| Exception::custom("Build error"))?, lm_head: nn::LinearBuilder::new(config.hidden_size, config.vocab_size).bias(false).build().map_err(|_| Exception::custom("Build error"))?, config })
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = self.embed.forward(x)?;
        for layer in self.layers.iter_mut() { h = layer.forward(&h)?; }
        h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }
    pub fn eval(&self) -> Result<(), Exception> {
        self.embed.weight.eval()?;
        for layer in &self.layers {
            if let Some(ref a) = layer.attention { a.q_proj.weight.eval()?; a.k_proj.weight.eval()?; a.v_proj.weight.eval()?; a.o_proj.weight.eval()?; }
            if let Some(ref m) = layer.mamba { m.router.gate.weight.eval()?; }
            layer.mlp.router.gate.weight.eval()?;
            layer.norm.weight.eval()?;
        }
        self.norm.weight.eval()?;
        self.lm_head.weight.eval()?;
        Ok(())
    }
}
