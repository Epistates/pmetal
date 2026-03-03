//! Qwen 3.5 (qwen3_next) hybrid architecture.
//!
//! Implements the Qwen3.5 architecture which combines:
//! - **Gated Delta Net (GDN)** linear attention layers (75% of layers)
//! - **Full attention** layers with gated output (every 4th layer)
//! - **Sparse MoE** with 512 experts + shared expert
//! - **(1+w) RMSNorm** (Gemma-style, requires f32 upcast)
//! - **Partial rotary** (25% of head dimensions)
//!
//! Reference: `mlx-lm/models/qwen3_next.py` (Apple, 2025).

use std::collections::HashMap;

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters, ModuleParametersExt, Param},
    nn,
    ops::{self, indexing::IndexOp},
};
use serde::{Deserialize, Serialize};

use crate::traits::ModelConfig;
use pmetal_mlx::{
    gather_mm,
    kernels::{
        AttentionMaskType, FusedAttentionConfig, fused_sdpa,
        gated_delta::gated_delta_update,
        rope::{RopeScaling, apply_rope},
    },
};
use pmetal_mlx::kv_cache::{KVCache, MambaCache, MambaCacheEntry};

// ============================================================================
// Configuration
// ============================================================================

/// Qwen 3.5 (qwen3_next) model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3NextConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Hidden dimension.
    pub hidden_size: i32,
    /// Intermediate size for dense MLP layers.
    pub intermediate_size: i32,
    /// Number of hidden layers.
    pub num_hidden_layers: i32,
    /// Number of attention heads (for full attention layers).
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA in full attention).
    #[serde(default)]
    pub num_key_value_heads: Option<i32>,
    /// Head dimension.
    #[serde(default = "default_head_dim")]
    pub head_dim: Option<i32>,
    /// Maximum position embeddings.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// RMS norm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE theta base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // --- GDN linear attention params ---
    /// Number of value heads for GDN layers.
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: i32,
    /// Number of key heads for GDN layers.
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: i32,
    /// Key head dimension for GDN.
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: i32,
    /// Value head dimension for GDN.
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: i32,
    /// Conv1d kernel size for GDN.
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: i32,

    // --- Hybrid layer control ---
    /// Every Nth layer is a full attention layer (default 4).
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: i32,

    // --- MoE params ---
    /// Number of routed experts.
    #[serde(default)]
    pub num_experts: i32,
    /// Number of experts per token.
    #[serde(default)]
    pub num_experts_per_tok: i32,
    /// MoE layer frequency (every Nth layer).
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: i32,
    /// MoE intermediate size.
    #[serde(default)]
    pub moe_intermediate_size: i32,
    /// Shared expert intermediate size.
    #[serde(default)]
    pub shared_expert_intermediate_size: i32,
    /// Layers forced to use dense MLP instead of MoE.
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    /// Normalize top-k routing probabilities.
    #[serde(default)]
    pub norm_topk_prob: bool,

    // --- Attention ---
    /// Fraction of head dimensions to apply RoPE to (default 0.25).
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    /// Whether attention uses bias.
    #[serde(default)]
    pub attention_bias: bool,
    /// RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
}

fn default_model_type() -> String {
    "qwen3_next".to_string()
}
fn default_vocab_size() -> i32 {
    151936
}
fn default_head_dim() -> Option<i32> {
    Some(128)
}
fn default_max_position_embeddings() -> i32 {
    131072
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_linear_num_value_heads() -> i32 {
    8
}
fn default_linear_num_key_heads() -> i32 {
    4
}
fn default_linear_key_head_dim() -> i32 {
    128
}
fn default_linear_value_head_dim() -> i32 {
    128
}
fn default_linear_conv_kernel_dim() -> i32 {
    4
}
fn default_full_attention_interval() -> i32 {
    4
}
fn default_decoder_sparse_step() -> i32 {
    1
}
fn default_partial_rotary_factor() -> f32 {
    0.25
}

impl Qwen3NextConfig {
    /// Get head dimension.
    pub fn get_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Get KV heads count.
    pub fn get_num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Check if layer at index is a linear (GDN) layer.
    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        ((layer_idx as i32) + 1) % self.full_attention_interval != 0
    }

    /// Check if layer uses MoE.
    pub fn use_moe_at(&self, layer_idx: usize) -> bool {
        let idx = layer_idx as i32;
        if self.mlp_only_layers.contains(&idx) {
            return false;
        }
        self.num_experts > 0 && ((idx + 1) % self.decoder_sparse_step == 0)
    }

    /// RoPE dimensions for partial rotary.
    pub fn rope_dims(&self) -> i32 {
        (self.get_head_dim() as f32 * self.partial_rotary_factor) as i32
    }
}

impl ModelConfig for Qwen3NextConfig {
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
        self.get_num_kv_heads()
    }
    fn head_dim(&self) -> i32 {
        self.get_head_dim()
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

impl Default for Qwen3NextConfig {
    fn default() -> Self {
        Self {
            model_type: "qwen3_next".to_string(),
            vocab_size: 151936,
            hidden_size: 1536,
            intermediate_size: 4096,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: Some(4),
            head_dim: Some(128),
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            linear_num_value_heads: 8,
            linear_num_key_heads: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            num_experts: 512,
            num_experts_per_tok: 8,
            decoder_sparse_step: 1,
            moe_intermediate_size: 256,
            shared_expert_intermediate_size: 4096,
            mlp_only_layers: vec![],
            norm_topk_prob: false,
            partial_rotary_factor: 0.25,
            attention_bias: false,
            rope_scaling: None,
        }
    }
}

// ============================================================================
// Gated RMSNorm (with optional silu gate)
// ============================================================================

/// RMSNorm with optional gating: `rms_norm(x, w, eps) * silu(gate)`.
///
/// Note: Weights use (1+w) convention — the `+1` offset is applied during
/// weight sanitization, so at runtime we use standard RMSNorm.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3NextRMSNormGated {
    #[param]
    pub weight: Param<Array>,
    pub eps: f32,
}

impl Qwen3NextRMSNormGated {
    pub fn new(hidden_size: i32, eps: f32) -> Result<Self, Exception> {
        let weight = Array::ones::<f32>(&[hidden_size])?;
        Ok(Self {
            weight: Param::new(weight),
            eps,
        })
    }

    pub fn forward(&self, x: &Array, gate: Option<&Array>) -> Result<Array, Exception> {
        let normed = mlx_rs::fast::rms_norm(x, self.weight.as_ref(), self.eps)?;
        if let Some(g) = gate {
            normed.multiply(&nn::silu(g)?)
        } else {
            Ok(normed)
        }
    }
}

// ============================================================================
// Qwen3Next MLP (SwiGLU)
// ============================================================================

#[derive(Debug, ModuleParameters)]
pub struct Qwen3NextMLP {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl Qwen3NextMLP {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim).bias(false).build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim).bias(false).build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim).bias(false).build()?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&nn::silu(&gate)?.multiply(&up)?)
    }
}

// ============================================================================
// Qwen3Next Attention (Full attention with gated output + partial RoPE)
// ============================================================================

#[derive(Debug, ModuleParameters)]
pub struct Qwen3NextAttention {
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub effective_base: f32,
    pub rope_scale: f32,
}

impl Qwen3NextAttention {
    pub fn new(config: &Qwen3NextConfig) -> Result<Self, Exception> {
        let head_dim = config.get_head_dim();
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.get_num_kv_heads();

        let rope_scaling = config
            .rope_scaling
            .as_ref()
            .map(RopeScaling::from_config_map)
            .unwrap_or(RopeScaling::None);
        let rope_scale = rope_scaling.scale();
        let effective_base = rope_scaling.effective_base(config.rope_theta, head_dim);

        // q_proj outputs 2x for gated output: n_heads * head_dim * 2
        let q_proj = nn::LinearBuilder::new(config.hidden_size, n_heads * head_dim * 2)
            .bias(config.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(config.hidden_size, n_kv_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, config.hidden_size)
            .bias(config.attention_bias)
            .build()?;

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
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_dims: config.rope_dims(),
            effective_base,
            rope_scale,
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

        // Project Q (with gate) and K, V
        let q_proj_out = self.q_proj.forward(x)?;
        // Reshape to [B, L, n_heads, head_dim * 2], split into queries and gate
        let q_gate = q_proj_out.reshape(&[b, l, self.n_heads, self.head_dim * 2])?;
        let queries = q_gate.index((.., .., .., ..self.head_dim));
        let gate = q_gate
            .index((.., .., .., self.head_dim..))
            .reshape(&[b, l, self.n_heads * self.head_dim])?;

        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape and apply Q/K norm
        let mut queries = self.q_norm.forward(&queries)?;
        let mut keys = self.k_norm.forward(
            &keys.reshape(&[b, l, self.n_kv_heads, self.head_dim])?,
        )?;
        let values = values.reshape(&[b, l, self.n_kv_heads, self.head_dim])?;

        // Transpose to [B, heads, L, head_dim]
        queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        keys = keys.transpose_axes(&[0, 2, 1, 3])?;
        let values = values.transpose_axes(&[0, 2, 1, 3])?;

        // Apply partial RoPE
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let queries = apply_rope(
            &queries,
            self.rope_dims,
            false,
            self.effective_base,
            self.rope_scale,
            offset,
        )?;
        let keys = apply_rope(
            &keys,
            self.rope_dims,
            false,
            self.effective_base,
            self.rope_scale,
            offset,
        )?;

        // Update KV cache
        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &keys, &values)?
        } else {
            (keys, values)
        };

        // Fused SDPA with GQA
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });

        let output = fused_sdpa(&queries, &keys, &values, &attn_config, mask)?;
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, self.n_heads * self.head_dim])?;

        // Gated output: o_proj(output * sigmoid(gate))
        let gated = output.multiply(&nn::sigmoid(&gate)?)?;
        self.o_proj.forward(&gated)
    }
}

// ============================================================================
// Qwen3Next Gated Delta Net (GDN) linear attention
// ============================================================================

#[derive(Debug, ModuleParameters)]
pub struct Qwen3NextGatedDeltaNet {
    #[param]
    pub conv1d: nn::Conv1d,
    #[param]
    pub in_proj_qkvz: nn::Linear,
    #[param]
    pub in_proj_ba: nn::Linear,
    #[param]
    pub norm: Qwen3NextRMSNormGated,
    #[param]
    pub out_proj: nn::Linear,
    pub dt_bias: Param<Array>,
    pub a_log: Param<Array>,

    pub hidden_size: i32,
    pub num_v_heads: i32,
    pub num_k_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub key_dim: i32,
    pub value_dim: i32,
    pub conv_dim: i32,
    pub conv_kernel_size: i32,
}

impl Qwen3NextGatedDeltaNet {
    pub fn new(config: &Qwen3NextConfig) -> Result<Self, Exception> {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = config.linear_conv_kernel_dim;

        let conv_dim = key_dim * 2 + value_dim;
        // Depthwise conv1d: groups=conv_dim, so each group has in_channels/groups=1
        // Conv1dBuilder doesn't divide in_channels by groups for the weight shape,
        // so we pass in_channels=1 to get weight shape [conv_dim, kernel, 1].
        // When loading pretrained weights, sanitize_weights transposes as needed.
        let conv1d = nn::Conv1dBuilder::new(1, conv_dim, conv_kernel_size)
            .bias(false)
            .groups(conv_dim)
            .padding(0)
            .build()?;

        let in_proj_qkvz =
            nn::LinearBuilder::new(hidden_size, key_dim * 2 + value_dim * 2)
                .bias(false)
                .build()?;
        let in_proj_ba =
            nn::LinearBuilder::new(hidden_size, num_v_heads * 2)
                .bias(false)
                .build()?;

        let dt_bias = Param::new(Array::ones::<f32>(&[num_v_heads])?);
        let a_log = Param::new(
            mlx_rs::random::uniform::<_, f32>(0.0, 16.0, &[num_v_heads], None)?
                .log()?,
        );

        let norm = Qwen3NextRMSNormGated::new(head_v_dim, config.rms_norm_eps)?;
        let out_proj = nn::LinearBuilder::new(value_dim, hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            conv1d,
            in_proj_qkvz,
            in_proj_ba,
            norm,
            out_proj,
            dt_bias,
            a_log,
            hidden_size,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size,
        })
    }

    /// Reorder interleaved qkvz projection outputs into separate tensors.
    fn fix_query_key_value_ordering(
        &self,
        mixed_qkvz: &Array,
        mixed_ba: &Array,
    ) -> Result<(Array, Array, Array, Array, Array, Array), Exception> {
        let nk = self.num_k_heads;
        let dn = self.head_k_dim;
        let nv = self.num_v_heads;
        let dv = self.head_v_dim;
        let leading = &mixed_qkvz.shape()[..mixed_qkvz.ndim() - 1];

        // Reshape to [..., nk, -1]
        let mut qkvz_shape = leading.to_vec();
        qkvz_shape.push(nk);
        qkvz_shape.push(-1);
        let mixed = mixed_qkvz.reshape(&qkvz_shape)?;

        let mut ba_shape = mixed_ba.shape()[..mixed_ba.ndim() - 1].to_vec();
        ba_shape.push(nk);
        ba_shape.push(-1);
        let mixed_ba = mixed_ba.reshape(&ba_shape)?;

        // Split: q=[dn], k=[dn], v=[nv/nk*dv], z=[nv/nk*dv]
        let split1 = dn;
        let split2 = 2 * dn;
        let split3 = 2 * dn + (nv / nk) * dv;

        let q = mixed.index((.., .., .., ..split1));
        let k = mixed.index((.., .., .., split1..split2));
        let v_raw = mixed.index((.., .., .., split2..split3));
        let z_raw = mixed.index((.., .., .., split3..));

        // Split ba: b=[nv/nk], a=[nv/nk]
        let b_split = nv / nk;
        let b_raw = mixed_ba.index((.., .., .., ..b_split));
        let a_raw = mixed_ba.index((.., .., .., b_split..));

        // Reshape v, z to [..., nv, dv] and b, a to [..., nv]
        let batch_seq = &leading[..2]; // [B, S]
        let mut v_shape = batch_seq.to_vec();
        v_shape.extend_from_slice(&[nv, dv]);
        let mut ba_out_shape = batch_seq.to_vec();
        ba_out_shape.push(nv);

        let v = v_raw.reshape(&v_shape)?;
        let z = z_raw.reshape(&v_shape)?;
        let b = b_raw.reshape(&ba_out_shape)?;
        let a = a_raw.reshape(&ba_out_shape)?;

        Ok((q, k, v, z, b, a))
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let b = shape[0];
        let s = shape[1];

        // Project inputs (compute before passing to fix_query_key_value_ordering
        // to avoid simultaneous mutable/immutable borrow of self)
        let qkvz_proj = self.in_proj_qkvz.forward(inputs)?;
        let ba_proj = self.in_proj_ba.forward(inputs)?;
        let (q, k, v, z, b_val, a) = self.fix_query_key_value_ordering(
            &qkvz_proj,
            &ba_proj,
        )?;

        // Convolution state management
        let conv_state = if let Some(ref cache) = cache {
            cache.conv_state.clone()
        } else {
            None
        };
        let conv_state = conv_state.unwrap_or_else(|| {
            Array::zeros::<f32>(&[b, self.conv_kernel_size - 1, self.conv_dim]).unwrap()
        });

        // Concatenate q, k, v for conv input
        let mixed_qkv = ops::concatenate_axis(
            &[
                &q.reshape(&[b, s, -1])?,
                &k.reshape(&[b, s, -1])?,
                &v.reshape(&[b, s, -1])?,
            ],
            -1,
        )?;

        // Apply mask to conv input if present
        let mixed_qkv = if let Some(mask) = mask {
            // mask: [B, T] -> [B, T, 1]
            let mask_expanded = mask.reshape(&[mask.dim(0), mask.dim(1), 1])?;
            ops::r#where(&mask_expanded, &mixed_qkv, &Array::from_f32(0.0))?
        } else {
            mixed_qkv
        };

        // Prepend conv state and run conv1d
        let conv_input = ops::concatenate_axis(&[&conv_state, &mixed_qkv], 1)?;

        // Update conv state in cache
        if let Some(cache) = cache.as_deref_mut() {
            let keep = self.conv_kernel_size - 1;
            let total_len = conv_input.dim(1);
            cache.conv_state = Some(conv_input.index((.., (total_len - keep).., ..)));
        }

        let conv_out = nn::silu(&Module::forward(&mut self.conv1d, &conv_input)?)?;

        // Split conv output back into q, k, v
        let q_conv = conv_out.index((.., .., ..self.key_dim));
        let k_conv = conv_out.index((.., .., self.key_dim..self.key_dim * 2));
        let v_conv = conv_out.index((.., .., self.key_dim * 2..));

        // Take only the last S timesteps (conv adds padding)
        let out_len = q_conv.dim(1);
        let q_conv = q_conv
            .index((.., (out_len - s).., ..))
            .reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let k_conv = k_conv
            .index((.., (out_len - s).., ..))
            .reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let v_conv = v_conv
            .index((.., (out_len - s).., ..))
            .reshape(&[b, s, self.num_v_heads, self.head_v_dim])?;

        // Apply Q/K RMS normalization with scaling
        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let q_normed = mlx_rs::fast::rms_norm(
            &q_conv,
            &Array::ones::<f32>(&[self.head_k_dim])?,
            1e-6,
        )?
        .multiply(&Array::from_f32(inv_scale * inv_scale))?;
        let k_normed = mlx_rs::fast::rms_norm(
            &k_conv,
            &Array::ones::<f32>(&[self.head_k_dim])?,
            1e-6,
        )?
        .multiply(&Array::from_f32(inv_scale))?;

        // Get SSM state from cache
        let ssm_state = cache
            .as_ref()
            .and_then(|c| c.ssm_state.as_ref());

        // Run GDN recurrence
        let (out, new_state) = gated_delta_update(
            &q_normed,
            &k_normed,
            &v_conv,
            &a,
            &b_val,
            self.a_log.as_ref(),
            self.dt_bias.as_ref(),
            ssm_state,
            mask,
        )?;

        // Update SSM state in cache
        if let Some(cache) = cache {
            cache.ssm_state = Some(new_state);
        }

        // Apply gated norm and output projection
        let out = self.norm.forward(&out, Some(&z))?;
        self.out_proj.forward(&out.reshape(&[b, s, -1])?)
    }
}

// ============================================================================
// Sparse MoE Block
// ============================================================================

#[derive(Debug, ModuleParameters)]
pub struct Qwen3NextSparseMoeBlock {
    #[param]
    pub gate: nn::Linear,
    #[param]
    pub switch_mlp_gate_proj: Param<Array>,
    #[param]
    pub switch_mlp_up_proj: Param<Array>,
    #[param]
    pub switch_mlp_down_proj: Param<Array>,
    #[param]
    pub shared_expert: Qwen3NextMLP,
    #[param]
    pub shared_expert_gate: nn::Linear,
    pub num_experts: i32,
    pub top_k: i32,
    pub norm_topk_prob: bool,
}

impl Qwen3NextSparseMoeBlock {
    pub fn new(config: &Qwen3NextConfig) -> Result<Self, Exception> {
        let dim = config.hidden_size;
        let intermediate_size = config.moe_intermediate_size;
        let num_experts = config.num_experts;

        let gate = nn::LinearBuilder::new(dim, num_experts)
            .bias(false)
            .build()?;

        // SwitchGLU stacked weights: [num_experts, intermediate_size, dim] etc.
        let gate_proj = Array::zeros::<f32>(&[num_experts, intermediate_size, dim])?;
        let up_proj = Array::zeros::<f32>(&[num_experts, intermediate_size, dim])?;
        let down_proj = Array::zeros::<f32>(&[num_experts, dim, intermediate_size])?;

        let shared_expert = Qwen3NextMLP::new(dim, config.shared_expert_intermediate_size)?;
        let shared_expert_gate = nn::LinearBuilder::new(dim, 1).bias(false).build()?;

        Ok(Self {
            gate,
            switch_mlp_gate_proj: Param::new(gate_proj),
            switch_mlp_up_proj: Param::new(up_proj),
            switch_mlp_down_proj: Param::new(down_proj),
            shared_expert,
            shared_expert_gate,
            num_experts,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch_seq: i32 = shape[..shape.len() - 1].iter().product();
        let hidden = shape[shape.len() - 1];
        let x_flat = x.reshape(&[batch_seq, hidden])?;

        // Compute routing probabilities
        let gate_logits = self.gate.forward(&x_flat)?;
        let gates = ops::softmax_axis(
            &if gate_logits.dtype() != mlx_rs::Dtype::Float32 {
                gate_logits.as_type::<f32>()?
            } else {
                gate_logits
            },
            -1,
            None,
        )?;

        // Top-k selection
        let k = self.top_k;
        let neg_gates = gates.negative()?;
        let sorted_indices = ops::argsort_axis(&neg_gates, -1)?;
        let top_indices = sorted_indices.index((.., ..k));
        let top_weights = gates.take_along_axis(&top_indices, -1)?;

        let top_weights = if self.norm_topk_prob {
            let weight_sum = top_weights.sum_axis(-1, Some(true))?;
            let safe_sum = ops::maximum(&weight_sum, &Array::from_f32(1e-8))?;
            top_weights.divide(&safe_sum)?
        } else {
            top_weights
        };

        // SwitchGLU forward using gather_mm
        // x_flat: [N, D], indices: [N, k]
        let top_indices_i32 = top_indices.as_type::<i32>()?;

        // For each expert slot, gather expert weights and compute
        // gather_mm: batch matmul with expert selection
        let gate_out = gather_mm(
            &x_flat,
            self.switch_mlp_gate_proj.as_ref(),
            None,
            Some(&top_indices_i32),
            false,
        )?; // [N, k, intermediate]
        let up_out = gather_mm(
            &x_flat,
            self.switch_mlp_up_proj.as_ref(),
            None,
            Some(&top_indices_i32),
            false,
        )?; // [N, k, intermediate]

        let activated = nn::silu(&gate_out)?.multiply(&up_out)?;

        // Down projection
        let down_out = gather_mm(
            &activated.reshape(&[batch_seq * k, -1])?,
            self.switch_mlp_down_proj.as_ref(),
            None,
            Some(&top_indices_i32.reshape(&[batch_seq * k, 1])?),
            false,
        )?
        .reshape(&[batch_seq, k, hidden])?;

        // Weight and sum expert outputs
        let y = down_out
            .multiply(&top_weights.reshape(&[batch_seq, k, 1])?)?
            .sum_axis(-2, false)?;

        // Shared expert with gate
        let shared_y = self.shared_expert.forward(&x_flat)?;
        let shared_gate = nn::sigmoid(&self.shared_expert_gate.forward(&x_flat)?)?;
        let shared_y = shared_gate.multiply(&shared_y)?;

        let result = y.add(&shared_y)?;
        result.reshape(shape)
    }
}

// ============================================================================
// Feed-forward enum (Dense MLP or MoE)
// ============================================================================

#[derive(Debug)]
pub enum Qwen3NextFeedForward {
    Dense(Qwen3NextMLP),
    MoE(Qwen3NextSparseMoeBlock),
}

impl ModuleParameters for Qwen3NextFeedForward {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(m) => m.num_parameters(),
            Self::MoE(m) => m.num_parameters(),
        }
    }
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(m) => m.parameters(),
            Self::MoE(m) => m.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Dense(m) => m.parameters_mut(),
            Self::MoE(m) => m.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(m) => m.trainable_parameters(),
            Self::MoE(m) => m.trainable_parameters(),
        }
    }
    fn freeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::Dense(m) => m.freeze_parameters(recurse),
            Self::MoE(m) => m.freeze_parameters(recurse),
        }
    }
    fn unfreeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::Dense(m) => m.unfreeze_parameters(recurse),
            Self::MoE(m) => m.unfreeze_parameters(recurse),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(m) => m.all_frozen(),
            Self::MoE(m) => m.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(m) => m.any_frozen(),
            Self::MoE(m) => m.any_frozen(),
        }
    }
}

impl Qwen3NextFeedForward {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Dense(m) => m.forward(x),
            Self::MoE(m) => m.forward(x),
        }
    }
}

// ============================================================================
// Attention/GDN mixer enum
// ============================================================================

#[derive(Debug)]
pub enum Qwen3NextMixer {
    FullAttention(Qwen3NextAttention),
    LinearAttention(Qwen3NextGatedDeltaNet),
}

impl ModuleParameters for Qwen3NextMixer {
    fn num_parameters(&self) -> usize {
        match self {
            Self::FullAttention(m) => m.num_parameters(),
            Self::LinearAttention(m) => m.num_parameters(),
        }
    }
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::FullAttention(m) => m.parameters(),
            Self::LinearAttention(m) => m.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::FullAttention(m) => m.parameters_mut(),
            Self::LinearAttention(m) => m.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::FullAttention(m) => m.trainable_parameters(),
            Self::LinearAttention(m) => m.trainable_parameters(),
        }
    }
    fn freeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::FullAttention(m) => m.freeze_parameters(recurse),
            Self::LinearAttention(m) => m.freeze_parameters(recurse),
        }
    }
    fn unfreeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::FullAttention(m) => m.unfreeze_parameters(recurse),
            Self::LinearAttention(m) => m.unfreeze_parameters(recurse),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::FullAttention(m) => m.all_frozen(),
            Self::LinearAttention(m) => m.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::FullAttention(m) => m.any_frozen(),
            Self::LinearAttention(m) => m.any_frozen(),
        }
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

#[derive(Debug, ModuleParameters)]
pub struct Qwen3NextDecoderLayer {
    pub is_linear: bool,
    #[param]
    pub mixer: Qwen3NextMixer,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub mlp: Qwen3NextFeedForward,
}

impl Qwen3NextDecoderLayer {
    pub fn new(config: &Qwen3NextConfig, layer_idx: usize) -> Result<Self, Exception> {
        let is_linear = config.is_linear_layer(layer_idx);

        let mixer = if is_linear {
            Qwen3NextMixer::LinearAttention(Qwen3NextGatedDeltaNet::new(config)?)
        } else {
            Qwen3NextMixer::FullAttention(Qwen3NextAttention::new(config)?)
        };

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        let mlp = if config.use_moe_at(layer_idx) {
            Qwen3NextFeedForward::MoE(Qwen3NextSparseMoeBlock::new(config)?)
        } else {
            Qwen3NextFeedForward::Dense(Qwen3NextMLP::new(
                config.hidden_size,
                config.intermediate_size,
            )?)
        };

        Ok(Self {
            is_linear,
            mixer,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let r = match &mut self.mixer {
            Qwen3NextMixer::LinearAttention(gdn) => gdn.forward(&normed, mask, mamba_cache)?,
            Qwen3NextMixer::FullAttention(attn) => attn.forward(&normed, mask, kv_cache)?,
        };
        let h = x.add(&r)?;
        let mlp_in = self.post_attention_layernorm.forward(&h)?;
        h.add(&self.mlp.forward(&mlp_in)?)
    }
}

// ============================================================================
// Model
// ============================================================================

#[derive(Debug, ModuleParameters)]
pub struct Qwen3NextModel {
    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<Qwen3NextDecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
    pub full_attention_interval: i32,
}

impl Qwen3NextModel {
    pub fn new(config: &Qwen3NextConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| Qwen3NextDecoderLayer::new(config, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            full_attention_interval: config.full_attention_interval,
        })
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None, None)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut kv_cache: Option<&mut KVCache>,
        mut mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let mut hidden = Module::forward(&mut self.embed_tokens, input_ids)?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let kv = if !layer.is_linear {
                kv_cache.as_deref_mut().map(|c| (c, layer_idx))
            } else {
                None
            };
            let mamba = if layer.is_linear {
                mamba_cache
                    .as_deref_mut()
                    .and_then(|c| c.get_mut(layer_idx))
            } else {
                None
            };

            // Use mask for both layer types; the layer will use it appropriately
            hidden = layer.forward(&hidden, mask, kv, mamba)?;
        }

        self.norm.forward(&hidden)
    }
}

// ============================================================================
// ForCausalLM
// ============================================================================

#[derive(Debug, ModuleParameters)]
pub struct Qwen3NextForCausalLM {
    #[param]
    pub model: Qwen3NextModel,
    #[param]
    pub lm_head: Option<nn::Linear>,
    pub config: Qwen3NextConfig,
}

impl Qwen3NextForCausalLM {
    pub fn new(config: Qwen3NextConfig) -> Result<Self, Exception> {
        let model = Qwen3NextModel::new(&config)?;
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
        let h = self.model.forward(input_ids, mask)?;
        self.lm_head_forward(&h)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let h = self
            .model
            .forward_with_cache(input_ids, mask, kv_cache, mamba_cache)?;
        self.lm_head_forward(&h)
    }

    pub fn config(&self) -> &Qwen3NextConfig {
        &self.config
    }
}

// ============================================================================
// Weight sanitization
// ============================================================================

/// Sanitize weights for Qwen3Next models.
///
/// Handles:
/// 1. Stacking per-expert weights into SwitchGLU format
/// 2. Adding +1 offset to (1+w) RMSNorm weights
/// 3. Transposing conv1d weights if needed
pub fn sanitize_weights(
    weights: &mut HashMap<String, Array>,
    config: &Qwen3NextConfig,
) -> Result<(), Exception> {
    // Check if expert stacking is needed
    let needs_stacking =
        weights.contains_key("model.layers.0.mlp.experts.0.up_proj.weight");

    if needs_stacking {
        for l in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{l}.mlp");
            for n in &["up_proj", "down_proj", "gate_proj"] {
                let mut expert_weights = Vec::new();
                for e in 0..config.num_experts {
                    let key = format!("{prefix}.experts.{e}.{n}.weight");
                    if let Some(w) = weights.remove(&key) {
                        expert_weights.push(w);
                    }
                }
                if !expert_weights.is_empty() {
                    let refs: Vec<&Array> = expert_weights.iter().collect();
                    let stacked = ops::stack_axis(&refs, 0)?;
                    weights.insert(format!("{prefix}.switch_mlp.{n}.weight"), stacked);
                }
            }
        }
    }

    // Remove MTP (multi-token prediction) weights
    let mtp_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.contains("mtp."))
        .cloned()
        .collect();
    for k in mtp_keys {
        weights.remove(&k);
    }

    // Apply (1+w) offset to norm weights and transpose conv1d
    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];

    let keys: Vec<String> = weights.keys().cloned().collect();
    for k in &keys {
        // Transpose conv1d weights if needed: [out, in, kernel] -> [out, kernel, in]
        if k.contains("conv1d.weight") {
            if let Some(v) = weights.get(k) {
                if v.ndim() == 3 && v.dim(2) != 1 {
                    let transposed = v.swap_axes(1, 2)?;
                    weights.insert(k.clone(), transposed);
                }
            }
        }

        // Add +1 to (1+w) norm weights
        if norm_suffixes.iter().any(|sfx| k.ends_with(sfx)) {
            if let Some(v) = weights.get(k) {
                if v.ndim() == 1 {
                    let offset = v.add(&Array::from_f32(1.0))?;
                    weights.insert(k.clone(), offset);
                }
            }
        }
    }

    // Remove lm_head.weight if tied
    if config.tie_word_embeddings {
        weights.remove("lm_head.weight");
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn tiny_config() -> Qwen3NextConfig {
        Qwen3NextConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: Some(16),
            vocab_size: 100,
            linear_num_value_heads: 2,
            linear_num_key_heads: 1,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 32,
            mlp_only_layers: vec![],
            norm_topk_prob: false,
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_config_layer_types() {
        let config = tiny_config();
        // With full_attention_interval=4, layers 0,1,2 are linear, layer 3 is full
        assert!(config.is_linear_layer(0), "Layer 0 should be linear");
        assert!(config.is_linear_layer(1), "Layer 1 should be linear");
        assert!(config.is_linear_layer(2), "Layer 2 should be linear");
        assert!(!config.is_linear_layer(3), "Layer 3 should be full attention");
    }

    #[test]
    fn test_config_deserialization() {
        // Minimal config JSON similar to Qwen3.5-0.8B
        let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 1536,
            "intermediate_size": 4096,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 151936,
            "linear_num_value_heads": 8,
            "linear_num_key_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "num_experts": 512,
            "num_experts_per_tok": 8,
            "decoder_sparse_step": 1,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 4096,
            "mlp_only_layers": [0, 1, 2, 3],
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true
        }"#;

        let config: Qwen3NextConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.num_experts, 512);
        assert_eq!(config.full_attention_interval, 4);
        assert_eq!(config.partial_rotary_factor, 0.25);
        assert_eq!(config.mlp_only_layers, vec![0, 1, 2, 3]);
        assert!(config.is_linear_layer(0));
        assert!(!config.is_linear_layer(3));
    }

    #[test]
    fn test_model_config_trait() {
        let config = tiny_config();
        assert_eq!(config.model_type(), "qwen3_next");
        assert_eq!(config.vocab_size(), 100);
        assert_eq!(config.hidden_size(), 32);
        assert_eq!(config.num_hidden_layers(), 4);
        assert_eq!(ModelConfig::head_dim(&config), 16);
    }

    #[test]
    #[serial]
    fn test_mlp_forward_shape() {
        let mut mlp = Qwen3NextMLP::new(32, 64).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 4, 32], None, None, None).unwrap();
        let output = mlp.forward(&x).unwrap();
        assert_eq!(output.shape(), &[1, 4, 32]);
    }

    #[test]
    #[serial]
    fn test_gated_rms_norm() {
        let norm = Qwen3NextRMSNormGated::new(32, 1e-6).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 4, 32], None, None, None).unwrap();

        // Without gate
        let out1 = norm.forward(&x, None).unwrap();
        assert_eq!(out1.shape(), &[1, 4, 32]);

        // With gate
        let gate = mlx_rs::random::normal::<f32>(&[1, 4, 32], None, None, None).unwrap();
        let out2 = norm.forward(&x, Some(&gate)).unwrap();
        assert_eq!(out2.shape(), &[1, 4, 32]);
    }

    #[test]
    #[serial]
    fn test_attention_output_shape() {
        let config = tiny_config();
        let mut attn = Qwen3NextAttention::new(&config).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 4, 32], None, None, None).unwrap();
        let output = attn.forward(&x, None, None).unwrap();
        assert_eq!(output.shape(), &[1, 4, 32]);
    }

    #[test]
    #[serial]
    fn test_gdn_output_shape() {
        let config = tiny_config();
        let mut gdn = Qwen3NextGatedDeltaNet::new(&config).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 4, 32], None, None, None).unwrap();
        let output = gdn.forward(&x, None, None).unwrap();
        assert_eq!(output.shape(), &[1, 4, 32]);
    }

    #[test]
    #[serial]
    fn test_decoder_layer_shapes() {
        let config = tiny_config();

        // Linear (GDN) layer
        let mut layer0 = Qwen3NextDecoderLayer::new(&config, 0).unwrap();
        assert!(layer0.is_linear);
        let x = mlx_rs::random::normal::<f32>(&[1, 4, 32], None, None, None).unwrap();
        let out0 = layer0.forward(&x, None, None, None).unwrap();
        assert_eq!(out0.shape(), &[1, 4, 32]);

        // Full attention layer
        let mut layer3 = Qwen3NextDecoderLayer::new(&config, 3).unwrap();
        assert!(!layer3.is_linear);
        let out3 = layer3.forward(&x, None, None, None).unwrap();
        assert_eq!(out3.shape(), &[1, 4, 32]);
    }

    #[test]
    #[serial]
    fn test_causal_lm_forward_shape() {
        let config = tiny_config();
        let vocab_size = config.vocab_size;
        let mut model = Qwen3NextForCausalLM::new(config).unwrap();

        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 4, vocab_size]);
    }

    #[test]
    #[serial]
    fn test_tie_word_embeddings() {
        let mut config = tiny_config();
        config.tie_word_embeddings = true;
        let model = Qwen3NextForCausalLM::new(config).unwrap();
        assert!(model.lm_head.is_none());

        let mut config = tiny_config();
        config.tie_word_embeddings = false;
        let model = Qwen3NextForCausalLM::new(config).unwrap();
        assert!(model.lm_head.is_some());
    }
}
