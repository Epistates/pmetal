//! Jamba 1.5 architecture implementation.
//!
//! Features:
//! - Hybrid Mamba + Transformer blocks
//! - Mixture of Experts (MoE) for MLP layers
//! - Supports massive context (256K) via linear SSM scaling
//! - ExpertsInt8 quantization support

use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, ModuleParameters, ModuleParametersExt};
use mlx_rs::nested::NestedHashMap;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, nn, ops};
use pmetal_core::ModelConfig;
use pmetal_mlx::Builder;
use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa};
use pmetal_mlx::moe::{MoEConfig, MoELayer};
use serde::{Deserialize, Serialize};
use std::rc::Rc;

fn default_jamba_mamba_conv_kernel_size() -> i32 {
    4
}

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
    #[serde(default = "default_jamba_mamba_conv_kernel_size")]
    pub mamba_conv_kernel_size: i32,
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
            mamba_conv_kernel_size: default_jamba_mamba_conv_kernel_size(),
        }
    }
}

/// Manual attention implementation for Jamba.
#[derive(Debug, ModuleParameters)]
pub struct JambaAttention {
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
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
            q_proj: nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim)
                .bias(false)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            k_proj: nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
                .bias(false)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            v_proj: nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
                .bias(false)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            o_proj: nn::LinearBuilder::new(n_heads * head_dim, config.hidden_size)
                .bias(false)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(AttentionMaskType::Causal);
        let out = fused_sdpa(&q, &k, &v, &config, None)?;
        self.o_proj.forward(
            &out.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[batch, seq_len, -1])?,
        )
    }
}

/// Lightweight causal sequence mixer for Jamba's non-attention layers.
///
/// This replaces the incorrect MoE placeholder with a real sequence-mixing
/// block that keeps all work on GPU and preserves causal ordering.
#[derive(Debug, ModuleParameters)]
pub struct JambaMambaMixer {
    #[param]
    pub in_proj: nn::Linear,
    #[param]
    pub conv1d: nn::Conv1d,
    #[param]
    pub out_proj: nn::Linear,
    pub hidden_size: i32,
    pub conv_kernel_size: i32,
}

impl JambaMambaMixer {
    pub fn new(config: &JambaConfig) -> Result<Self, Exception> {
        let hidden_size = config.hidden_size;
        let conv_kernel_size = config.mamba_conv_kernel_size.max(2);
        let in_proj = nn::LinearBuilder::new(hidden_size, hidden_size * 2)
            .bias(false)
            .build()
            .map_err(|_| Exception::custom("Build error"))?;
        let conv1d = nn::Conv1dBuilder::new(1, hidden_size, conv_kernel_size)
            .groups(hidden_size)
            .bias(false)
            .padding(0)
            .build()
            .map_err(|_| Exception::custom("Build error"))?;
        let out_proj = nn::LinearBuilder::new(hidden_size, hidden_size)
            .bias(false)
            .build()
            .map_err(|_| Exception::custom("Build error"))?;
        Ok(Self {
            in_proj,
            conv1d,
            out_proj,
            hidden_size,
            conv_kernel_size,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let projected = self.in_proj.forward(x)?;
        let parts = ops::split_sections(&projected, &[self.hidden_size], -1)?;
        let value = &parts[0];
        let gate = &parts[1];

        let padded = ops::pad(
            value,
            &[(0i32, 0i32), (self.conv_kernel_size - 1, 0), (0, 0)],
            Array::from_int(0),
            None,
        )?;
        let mixed = Module::forward(&mut self.conv1d, &padded)?;
        let gated = nn::silu(gate)?.multiply(&nn::silu(&mixed)?)?;
        self.out_proj.forward(&gated)
    }
}

/// Jamba Hybrid Layer.
#[derive(Debug)]
pub struct JambaLayer {
    pub attention: Option<JambaAttention>,
    pub mamba: Option<JambaMambaMixer>,
    pub mlp: MoELayer,
    pub norm: nn::RmsNorm,
    pub is_attention: bool,
}

impl ModuleParameters for JambaLayer {
    fn parameters(&self) -> NestedHashMap<Rc<str>, &Array> {
        let mut map = NestedHashMap::new();
        if let Some(ref a) = self.attention {
            map.entries.extend(a.parameters().entries);
        }
        if let Some(ref m) = self.mamba {
            map.entries.extend(m.parameters().entries);
        }
        map.entries.extend(self.mlp.parameters().entries);
        map.entries.extend(self.norm.parameters().entries);
        map
    }
    fn trainable_parameters(&self) -> NestedHashMap<Rc<str>, &Array> {
        let mut map = NestedHashMap::new();
        if let Some(ref a) = self.attention {
            map.entries.extend(a.trainable_parameters().entries);
        }
        if let Some(ref m) = self.mamba {
            map.entries.extend(m.trainable_parameters().entries);
        }
        map.entries.extend(self.mlp.trainable_parameters().entries);
        map.entries.extend(self.norm.trainable_parameters().entries);
        map
    }
    fn num_parameters(&self) -> usize {
        self.attention.as_ref().map_or(0, |a| a.num_parameters())
            + self.mamba.as_ref().map_or(0, |m| m.num_parameters())
            + self.mlp.num_parameters()
            + self.norm.num_parameters()
    }
    fn parameters_mut(&mut self) -> NestedHashMap<Rc<str>, &mut Array> {
        let mut map = NestedHashMap::new();
        if let Some(ref mut a) = self.attention {
            map.entries.extend(a.parameters_mut().entries);
        }
        if let Some(ref mut m) = self.mamba {
            map.entries.extend(m.parameters_mut().entries);
        }
        map.entries.extend(self.mlp.parameters_mut().entries);
        map.entries.extend(self.norm.parameters_mut().entries);
        map
    }
    fn freeze_parameters(&mut self, recurse: bool) {
        if let Some(ref mut a) = self.attention {
            a.freeze_parameters(recurse);
        }
        if let Some(ref mut m) = self.mamba {
            m.freeze_parameters(recurse);
        }
        self.mlp.freeze_parameters(recurse);
        self.norm.freeze_parameters(recurse);
    }
    fn unfreeze_parameters(&mut self, recurse: bool) {
        if let Some(ref mut a) = self.attention {
            a.unfreeze_parameters(recurse);
        }
        if let Some(ref mut m) = self.mamba {
            m.unfreeze_parameters(recurse);
        }
        self.mlp.unfreeze_parameters(recurse);
        self.norm.unfreeze_parameters(recurse);
    }
    fn all_frozen(&self) -> Option<bool> {
        let mut frozen = true;
        if let Some(ref a) = self.attention {
            frozen &= a.all_frozen()?;
        }
        if let Some(ref m) = self.mamba {
            frozen &= m.all_frozen()?;
        }
        frozen &= self.mlp.all_frozen()?;
        frozen &= self.norm.all_frozen()?;
        Some(frozen)
    }
    fn any_frozen(&self) -> Option<bool> {
        let mut frozen = false;
        if let Some(ref a) = self.attention {
            frozen |= a.any_frozen()?;
        }
        if let Some(ref m) = self.mamba {
            frozen |= m.any_frozen()?;
        }
        frozen |= self.mlp.any_frozen()?;
        frozen |= self.norm.any_frozen()?;
        Some(frozen)
    }
}

impl JambaLayer {
    pub fn new(config: &JambaConfig, layer_idx: usize) -> Result<Self, Exception> {
        let is_attention = (layer_idx as i32) % config.layers_per_block == config.attn_layer_offset;
        let attention = if is_attention {
            Some(JambaAttention::new(config)?)
        } else {
            None
        };
        let mamba = if !is_attention {
            Some(JambaMambaMixer::new(config)?)
        } else {
            None
        };
        let moe_config = MoEConfig::new(
            config.hidden_size,
            config.intermediate_size,
            config.num_experts as usize,
        )
        .with_num_experts_per_tok(config.num_experts_per_tok as usize);
        Ok(Self {
            attention,
            mamba,
            mlp: MoELayer::new(moe_config),
            norm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            is_attention,
        })
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let normed = self.norm.forward(x)?;
        let branch_out = if self.is_attention {
            self.attention.as_mut().unwrap().forward(&normed)?
        } else {
            self.mamba.as_mut().unwrap().forward(&normed)?
        };
        let x = x.add(&branch_out)?;
        let (mlp_out, _) = self.mlp.forward(&x)?;
        x.add(&mlp_out)
    }
}

#[derive(Debug, ModuleParameters)]
pub struct JambaModel {
    #[param]
    pub embed: nn::Embedding,
    #[param]
    pub layers: Vec<JambaLayer>,
    #[param]
    pub norm: nn::RmsNorm,
    #[param]
    pub lm_head: nn::Linear,
    pub config: JambaConfig,
}

impl JambaModel {
    pub fn new(config: JambaConfig) -> Result<Self, Exception> {
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| JambaLayer::new(&config, i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embed: nn::Embedding::new(config.vocab_size, config.hidden_size)?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            lm_head: nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                .bias(false)
                .build()
                .map_err(|_| Exception::custom("Build error"))?,
            config,
        })
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = self.embed.forward(x)?;
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h)?;
        }
        h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }
    pub fn eval(&self) -> Result<(), Exception> {
        ModuleParametersExt::eval(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_jamba_config() -> JambaConfig {
        JambaConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            vocab_size: 64,
            rms_norm_eps: 1e-5,
            num_experts: 4,
            num_experts_per_tok: 2,
            layers_per_block: 2,
            attn_layer_offset: 0,
            mamba_conv_kernel_size: 3,
        }
    }

    #[test]
    fn test_jamba_layer_schedule_builds_attention_and_mamba_layers() {
        let config = tiny_jamba_config();

        let attention_layer = JambaLayer::new(&config, 0).expect("attention layer");
        assert!(attention_layer.is_attention);
        assert!(attention_layer.attention.is_some());
        assert!(attention_layer.mamba.is_none());

        let mamba_layer = JambaLayer::new(&config, 1).expect("mamba layer");
        assert!(!mamba_layer.is_attention);
        assert!(mamba_layer.attention.is_none());
        assert!(mamba_layer.mamba.is_some());
    }

    #[test]
    fn test_jamba_mamba_mixer_is_causal() {
        let config = tiny_jamba_config();
        let mut mixer = JambaMambaMixer::new(&config).expect("mixer");

        let baseline = Array::zeros::<f32>(&[1, 4, config.hidden_size]).expect("baseline");
        let prefix = Array::zeros::<f32>(&[1, 3, config.hidden_size]).expect("prefix");
        let suffix = Array::ones::<f32>(&[1, 1, config.hidden_size]).expect("suffix");
        let changed = ops::concatenate_axis(&[&prefix, &suffix], 1).expect("changed");

        let y0 = mixer.forward(&baseline).expect("baseline forward");
        let y1 = mixer.forward(&changed).expect("changed forward");
        let delta = y1.subtract(&y0).expect("delta");

        let earlier = delta
            .index((.., ..3, ..))
            .abs()
            .expect("earlier abs")
            .sum(None)
            .expect("earlier sum")
            .item::<f32>();
        let last = delta
            .index((.., 3..4, ..))
            .abs()
            .expect("last abs")
            .sum(None)
            .expect("last sum")
            .item::<f32>();

        assert!(
            earlier < 1e-6,
            "future-token edit should not affect earlier positions, got {earlier}"
        );
        assert!(last > 1e-6, "mixer should react to the edited token");
    }

    #[test]
    fn test_jamba_model_forward_shape() {
        let config = tiny_jamba_config();
        let mut model = JambaModel::new(config.clone()).expect("model");
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);

        let logits = model.forward(&input_ids).expect("forward");

        assert_eq!(logits.shape(), vec![1, 4, config.vocab_size]);
    }

    #[test]
    fn test_jamba_eval_marks_all_parameters() {
        let config = tiny_jamba_config();
        let model = JambaModel::new(config).expect("model");

        model.eval().expect("eval");
    }
}
