//! GPT-OSS model architecture (OpenAI's first open-weight models).
//!
//! Implements GPT-OSS with:
//! - Mixture of Experts (MoE) with Top-4 sigmoid routing
//! - Alternating sliding window (128) and full attention patterns
//! - SwiGLU activation with clamping (limit=7.0)
//! - Grouped Multi-Query Attention (GQA, group size 8)
//! - YaRN RoPE scaling for 128K context
//! - MXFP4 quantization support for MoE weights
//! - Per-expert bias in MoE GEMMs
//!
//! ## Model Variants
//!
//! | Model | Total Params | Active Params | Layers | Experts |
//! |-------|-------------|---------------|--------|---------|
//! | gpt-oss-20b | 21B | 3.6B | 24 | 32 |
//! | gpt-oss-120b | 117B | 5.1B | 36 | 128 |
//!
//! ## Key Features
//!
//! - Configurable reasoning effort (low/medium/high)
//! - Tool use optimization
//! - Apache 2.0 license
use pmetal_bridge::compat::{Array, Exception, ModuleParameters, Param, indexing, nn, ops, random};
use pmetal_bridge::impl_module_params;

use crate::fp8_utils::dequantize_fp8_weight_for_compute;
use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa, rope::apply_rope};
use pmetal_mlx::kv_cache::KVCache;
use serde::{Deserialize, Serialize};

/// Attention layer type for GPT-OSS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AttentionType {
    /// Sliding window attention (128 tokens).
    SlidingAttention,
    /// Full context attention.
    #[default]
    FullAttention,
}

/// GPT-OSS model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptOssConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Hidden dimension.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    /// Intermediate size for MLP.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    /// Number of hidden layers.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA).
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    /// Head dimension.
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    /// Maximum position embeddings.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// Initial context length before extension.
    #[serde(default = "default_initial_context_length")]
    pub initial_context_length: i32,
    /// RMS normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    /// Whether to use attention bias.
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    /// Attention dropout rate.
    #[serde(default)]
    pub attention_dropout: f32,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Number of local experts.
    #[serde(default = "default_num_local_experts")]
    pub num_local_experts: i32,
    /// Number of experts per token (top-k).
    #[serde(default = "default_experts_per_token")]
    pub experts_per_token: i32,
    /// Also available as num_experts_per_tok for compatibility.
    #[serde(default)]
    pub num_experts_per_tok: Option<i32>,
    /// Router auxiliary loss coefficient.
    #[serde(default = "default_router_aux_loss_coef")]
    pub router_aux_loss_coef: f32,
    /// Output router logits.
    #[serde(default)]
    pub output_router_logits: bool,
    /// Sliding window size.
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    /// Layer types (alternating sliding_attention and full_attention).
    #[serde(default)]
    pub layer_types: Vec<AttentionType>,
    /// SwiGLU activation limit (clamp value).
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
    /// Hidden activation function.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// End of sequence token ID.
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: i32,
    /// Pad token ID.
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: i32,
}

/// RoPE scaling configuration (YaRN).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    /// Scaling type (typically "yarn").
    pub rope_type: String,
    /// Scaling factor.
    #[serde(default = "default_rope_factor")]
    pub factor: f32,
    /// Original max position embeddings.
    #[serde(default = "default_original_max_position")]
    pub original_max_position_embeddings: i32,
    /// Beta fast for YaRN.
    #[serde(default = "default_beta_fast")]
    pub beta_fast: f32,
    /// Beta slow for YaRN.
    #[serde(default = "default_beta_slow")]
    pub beta_slow: f32,
    /// Whether to truncate position embeddings.
    #[serde(default)]
    pub truncate: bool,
}

// Default functions
fn default_model_type() -> String {
    "gpt_oss".to_string()
}
fn default_vocab_size() -> i32 {
    201088
}
fn default_hidden_size() -> i32 {
    2880
}
fn default_intermediate_size() -> i32 {
    2880
}
fn default_num_hidden_layers() -> i32 {
    24
}
fn default_num_attention_heads() -> i32 {
    64
}
fn default_num_key_value_heads() -> i32 {
    8
}
fn default_head_dim() -> i32 {
    64
}
fn default_max_position_embeddings() -> i32 {
    131072
}
fn default_initial_context_length() -> i32 {
    4096
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    150000.0
}
fn default_true() -> bool {
    true
}
fn default_num_local_experts() -> i32 {
    32
}
fn default_experts_per_token() -> i32 {
    4
}
fn default_router_aux_loss_coef() -> f32 {
    0.9
}
fn default_sliding_window() -> i32 {
    128
}
fn default_swiglu_limit() -> f32 {
    7.0
}
fn default_hidden_act() -> String {
    "silu".to_string()
}
fn default_eos_token_id() -> i32 {
    200002
}
fn default_pad_token_id() -> i32 {
    199999
}
fn default_rope_factor() -> f32 {
    32.0
}
fn default_original_max_position() -> i32 {
    4096
}
fn default_beta_fast() -> f32 {
    32.0
}
fn default_beta_slow() -> f32 {
    1.0
}

impl GptOssConfig {
    /// Get the number of experts per token.
    pub fn num_experts_per_tok(&self) -> i32 {
        self.num_experts_per_tok.unwrap_or(self.experts_per_token)
    }

    /// Get the attention type for a given layer.
    pub fn attention_type_at(&self, layer_idx: usize) -> AttentionType {
        if !self.layer_types.is_empty() && layer_idx < self.layer_types.len() {
            self.layer_types[layer_idx]
        } else {
            // Default: alternate sliding and full
            if layer_idx % 2 == 0 {
                AttentionType::SlidingAttention
            } else {
                AttentionType::FullAttention
            }
        }
    }

    /// Get YaRN RoPE factor if configured.
    pub fn rope_factor(&self) -> f32 {
        self.rope_scaling.as_ref().map(|s| s.factor).unwrap_or(1.0)
    }

    /// Get the effective max position embeddings considering RoPE scaling.
    pub fn effective_max_position(&self) -> i32 {
        let base = self
            .rope_scaling
            .as_ref()
            .map(|s| s.original_max_position_embeddings)
            .unwrap_or(self.initial_context_length);
        (base as f32 * self.rope_factor()) as i32
    }

    /// Create config for GPT-OSS-20B.
    pub fn gpt_oss_20b() -> Self {
        Self {
            model_type: "gpt_oss".to_string(),
            vocab_size: 201088,
            hidden_size: 2880,
            intermediate_size: 2880,
            num_hidden_layers: 24,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 64,
            max_position_embeddings: 131072,
            initial_context_length: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 150000.0,
            rope_scaling: Some(RopeScalingConfig {
                rope_type: "yarn".to_string(),
                factor: 32.0,
                original_max_position_embeddings: 4096,
                beta_fast: 32.0,
                beta_slow: 1.0,
                truncate: false,
            }),
            attention_bias: true,
            attention_dropout: 0.0,
            tie_word_embeddings: false,
            num_local_experts: 32,
            experts_per_token: 4,
            num_experts_per_tok: Some(4),
            router_aux_loss_coef: 0.9,
            output_router_logits: false,
            sliding_window: 128,
            layer_types: vec![], // Will alternate by default
            swiglu_limit: 7.0,
            hidden_act: "silu".to_string(),
            eos_token_id: 200002,
            pad_token_id: 199999,
        }
    }

    /// Create config for GPT-OSS-120B.
    pub fn gpt_oss_120b() -> Self {
        let mut config = Self::gpt_oss_20b();
        config.num_hidden_layers = 36;
        config.num_local_experts = 128;
        config
    }
}

impl Default for GptOssConfig {
    fn default() -> Self {
        Self::gpt_oss_20b()
    }
}

/// GPT-OSS attention with alternating sliding/full patterns and GQA.
#[derive(Debug)]
pub struct GptOssAttention {
    /// Configuration.
    config: GptOssConfig,
    /// Layer index.
    layer_idx: usize,
    /// Number of attention heads.
    n_heads: i32,
    /// Number of KV heads.
    n_kv_heads: i32,
    /// Head dimension.
    head_dim: i32,
    /// Attention scale.
    scale: f32,
    /// RoPE theta.
    rope_theta: f32,
    /// Sliding window size (for sliding attention layers).
    sliding_window: i32,
    /// Attention type for this layer.
    attention_type: AttentionType,
    /// Query projection.
    pub q_proj: nn::Linear,
    /// Key projection.
    pub k_proj: nn::Linear,
    /// Value projection.
    pub v_proj: nn::Linear,
    /// Output projection.
    pub o_proj: nn::Linear,
}
impl_module_params!(GptOssAttention; q_proj, k_proj, v_proj, o_proj);

impl GptOssAttention {
    /// Create a new attention layer.
    pub fn new(config: GptOssConfig, layer_idx: usize) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;
        let scale = (head_dim as f32).powf(-0.5);
        let attention_type = config.attention_type_at(layer_idx);

        // GPT-OSS uses attention bias
        let use_bias = config.attention_bias;

        let q_proj = nn::LinearBuilder::new(hidden_size, n_heads * head_dim)
            .bias(use_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(use_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(use_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, hidden_size)
            .bias(use_bias)
            .build()?;

        Ok(Self {
            rope_theta: config.rope_theta,
            sliding_window: config.sliding_window,
            attention_type,
            layer_idx,
            config,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    /// Forward pass through attention.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let mut cache = cache;

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq, heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        // Apply RoPE
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q = apply_rope(&q, self.head_dim, false, self.rope_theta, 1.0, offset)?;
        let k = apply_rope(&k, self.head_dim, false, self.rope_theta, 1.0, offset)?;

        // Configure attention based on layer type
        let mask_type = match self.attention_type {
            AttentionType::SlidingAttention => {
                AttentionMaskType::SlidingWindow(self.sliding_window)
            }
            AttentionType::FullAttention => {
                if mask.is_some() {
                    AttentionMaskType::None
                } else {
                    AttentionMaskType::Causal
                }
            }
        };

        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(mask_type);

        if mask.is_none() {
            if let Some((cache_ref, layer_idx)) = cache.as_mut() {
                if let Some(output) =
                    (*cache_ref).try_turboquant_attention(*layer_idx, &q, &k, &v, &attn_config)?
                {
                    let output = output.transpose_axes(&[0, 2, 1, 3]);
                    let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim]);
                    return Ok(self.o_proj.forward(&output));
                }
            }
        }

        // Update cache if provided
        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        // Transpose back and project
        let output = output.transpose_axes(&[0, 2, 1, 3]);
        let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim]);

        Ok(self.o_proj.forward(&output))
    }
}

/// SwiGLU MLP with clamping for GPT-OSS.
#[derive(Debug)]
pub struct GptOssMLP {
    /// SwiGLU limit (clamp value).
    swiglu_limit: f32,
    /// Gate projection.
    pub gate_proj: nn::Linear,
    /// Up projection.
    pub up_proj: nn::Linear,
    /// Down projection.
    pub down_proj: nn::Linear,
}
impl_module_params!(GptOssMLP; gate_proj, up_proj, down_proj);

impl GptOssMLP {
    /// Create a new MLP.
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        swiglu_limit: f32,
    ) -> Result<Self, Exception> {
        // GPT-OSS uses bias in MLP
        let gate_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(true)
            .build()?;
        let up_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(true)
            .build()?;
        let down_proj = nn::LinearBuilder::new(intermediate_size, hidden_size)
            .bias(true)
            .build()?;

        Ok(Self {
            swiglu_limit,
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass through MLP with SwiGLU and clamping.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);

        let clamped = clamp_swiglu_hidden(&gate, &up, self.swiglu_limit)?;

        Ok(self.down_proj.forward(&clamped))
    }
}

fn clamp_swiglu_hidden(gate: &Array, up: &Array, limit: f32) -> Result<Array, Exception> {
    let activated = nn::silu(gate).multiply(up);
    let limit = Array::from_f32(limit);
    let neg_limit = Array::from_f32(-limit.item::<f32>());
    let clamped = pmetal_bridge::compat::ops::minimum(&activated, &limit);
    Ok(pmetal_bridge::compat::ops::maximum(&clamped, &neg_limit))
}

/// GPT-OSS expert MLP with bias and SwiGLU clamping.
#[derive(Debug)]
pub struct GptOssMoEExpert {
    /// SwiGLU limit (clamp value).
    swiglu_limit: f32,
    /// Gate projection.
    pub gate_proj: nn::Linear,
    /// Up projection.
    pub up_proj: nn::Linear,
    /// Down projection.
    pub down_proj: nn::Linear,
}
impl_module_params!(GptOssMoEExpert; gate_proj, up_proj, down_proj);

impl GptOssMoEExpert {
    /// Create a new expert.
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        swiglu_limit: f32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            swiglu_limit,
            gate_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(true)
                .build()?,
            up_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(true)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(true)
                .build()?,
        })
    }

    /// Forward pass through a single expert.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let clamped = clamp_swiglu_hidden(&gate, &up, self.swiglu_limit)?;
        Ok(self.down_proj.forward(&clamped))
    }
}

/// GPT-OSS MoE block with sigmoid routing.
#[derive(Debug)]
pub struct GptOssMoE {
    /// Top-k experts per token.
    top_k: usize,
    /// Router auxiliary loss coefficient.
    router_aux_loss_coef: f32,
    /// Gate projection (routes to experts).
    gate: nn::Linear,
    /// Expert MLPs.
    experts: Vec<GptOssMoEExpert>,
    /// SwiGLU limit for clamping.
    swiglu_limit: f32,
    /// Stacked gate projection weights `[num_experts, hidden, intermediate]`.
    stacked_gate_proj: Option<Array>,
    /// Stacked up projection weights `[num_experts, hidden, intermediate]`.
    stacked_up_proj: Option<Array>,
    /// Stacked down projection weights `[num_experts, intermediate, hidden]`.
    stacked_down_proj: Option<Array>,
    /// Stacked gate bias `[num_experts, intermediate]`.
    stacked_gate_bias: Option<Array>,
    /// Stacked up bias `[num_experts, intermediate]`.
    stacked_up_bias: Option<Array>,
    /// Stacked down bias `[num_experts, hidden]`.
    stacked_down_bias: Option<Array>,
    /// Signature of the current expert weight/bias handles.
    stacked_signature: Option<Vec<usize>>,
}
impl_module_params!(GptOssMoE; gate, experts);

impl GptOssMoE {
    /// Create a new MoE block.
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        num_experts: i32,
        experts_per_token: i32,
        swiglu_limit: f32,
        router_aux_loss_coef: f32,
    ) -> Result<Self, Exception> {
        let gate = nn::LinearBuilder::new(hidden_size, num_experts)
            .bias(false)
            .build()?;
        let experts = (0..num_experts as usize)
            .map(|_| GptOssMoEExpert::new(hidden_size, intermediate_size, swiglu_limit))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            top_k: experts_per_token as usize,
            router_aux_loss_coef,
            gate,
            experts,
            swiglu_limit,
            stacked_gate_proj: None,
            stacked_up_proj: None,
            stacked_down_proj: None,
            stacked_gate_bias: None,
            stacked_up_bias: None,
            stacked_down_bias: None,
            stacked_signature: None,
        })
    }

    fn current_signature(&self) -> Vec<usize> {
        // Use `Array::id()` (MLX array_desc pointer) as a stable change-
        // detection key. It is valid on unevaluated lazy arrays, unlike
        // `data_ptr()` which segfaults when the buffer has not been
        // materialised.
        let mut signature = Vec::with_capacity(self.experts.len() * 6);
        for expert in &self.experts {
            signature.push(expert.gate_proj.weight.as_ref().id());
            signature.push(
                expert
                    .gate_proj
                    .bias
                    .as_ref()
                    .as_ref()
                    .map(|bias| bias.id())
                    .unwrap_or(0),
            );
            signature.push(expert.up_proj.weight.as_ref().id());
            signature.push(
                expert
                    .up_proj
                    .bias
                    .as_ref()
                    .as_ref()
                    .map(|bias| bias.id())
                    .unwrap_or(0),
            );
            signature.push(expert.down_proj.weight.as_ref().id());
            signature.push(
                expert
                    .down_proj
                    .bias
                    .as_ref()
                    .as_ref()
                    .map(|bias| bias.id())
                    .unwrap_or(0),
            );
        }
        signature
    }

    fn stack_expert_weights(
        &self,
    ) -> Result<(Array, Array, Array, Array, Array, Array), Exception> {
        let gate_weights: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| {
                dequantize_fp8_weight_for_compute(expert.gate_proj.weight.as_ref()).map(|w| w.t())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let up_weights: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| {
                dequantize_fp8_weight_for_compute(expert.up_proj.weight.as_ref()).map(|w| w.t())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let down_weights: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| {
                dequantize_fp8_weight_for_compute(expert.down_proj.weight.as_ref()).map(|w| w.t())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let gate_biases: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| {
                expert
                    .gate_proj
                    .bias
                    .as_ref()
                    .map(|b| b.as_ref().clone())
                    .unwrap_or_else(|| Array::zeros_f32(&[expert.gate_proj.weight.as_ref().dim(0)]))
            })
            .collect();
        let up_biases: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| {
                expert
                    .up_proj
                    .bias
                    .as_ref()
                    .map(|b| b.as_ref().clone())
                    .unwrap_or_else(|| Array::zeros_f32(&[expert.up_proj.weight.as_ref().dim(0)]))
            })
            .collect();
        let down_biases: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| {
                expert
                    .down_proj
                    .bias
                    .as_ref()
                    .map(|b| b.as_ref().clone())
                    .unwrap_or_else(|| Array::zeros_f32(&[expert.down_proj.weight.as_ref().dim(0)]))
            })
            .collect();

        Ok((
            pmetal_bridge::compat::ops::stack_axis(&gate_weights, 0),
            pmetal_bridge::compat::ops::stack_axis(&up_weights, 0),
            pmetal_bridge::compat::ops::stack_axis(&down_weights, 0),
            pmetal_bridge::compat::ops::stack_axis(&gate_biases, 0),
            pmetal_bridge::compat::ops::stack_axis(&up_biases, 0),
            pmetal_bridge::compat::ops::stack_axis(&down_biases, 0),
        ))
    }

    fn ensure_stacked(&mut self) -> Result<(), Exception> {
        let signature = self.current_signature();
        let needs_refresh = self.stacked_gate_proj.is_none()
            || self.stacked_up_proj.is_none()
            || self.stacked_down_proj.is_none()
            || self.stacked_gate_bias.is_none()
            || self.stacked_up_bias.is_none()
            || self.stacked_down_bias.is_none()
            || self.stacked_signature.as_ref() != Some(&signature);

        if needs_refresh {
            let (
                stacked_gate_proj,
                stacked_up_proj,
                stacked_down_proj,
                stacked_gate_bias,
                stacked_up_bias,
                stacked_down_bias,
            ) = self.stack_expert_weights()?;
            stacked_gate_proj.eval();
            stacked_up_proj.eval();
            stacked_down_proj.eval();
            stacked_gate_bias.eval();
            stacked_up_bias.eval();
            stacked_down_bias.eval();
            self.stacked_gate_proj = Some(stacked_gate_proj);
            self.stacked_up_proj = Some(stacked_up_proj);
            self.stacked_down_proj = Some(stacked_down_proj);
            self.stacked_gate_bias = Some(stacked_gate_bias);
            self.stacked_up_bias = Some(stacked_up_bias);
            self.stacked_down_bias = Some(stacked_down_bias);
            self.stacked_signature = Some(signature);
        }

        Ok(())
    }

    /// Eagerly build or refresh the stacked expert cache.
    pub fn init_stacked_moe(&mut self) -> Result<(), Exception> {
        self.ensure_stacked()
    }

    /// Whether the stacked expert cache is populated.
    pub fn has_stacked_moe(&self) -> bool {
        self.stacked_gate_proj.is_some()
            && self.stacked_up_proj.is_some()
            && self.stacked_down_proj.is_some()
            && self.stacked_gate_bias.is_some()
            && self.stacked_up_bias.is_some()
            && self.stacked_down_bias.is_some()
    }

    fn route_topk(&mut self, hidden_flat: &Array) -> Result<(i32, i32, Array, Array), Exception> {
        let batch_seq = hidden_flat.dim(0);
        let hidden_size = hidden_flat.dim(1);

        // GPT-OSS uses sigmoid (not softmax) on router logits; weights are
        // always renormalised to sum to 1.
        let gate_logits = self.gate.forward(hidden_flat);
        let scores = pmetal_bridge::compat::ops::sigmoid(&gate_logits);

        let (top_indices, normalized_weights) = crate::moe_routing::topk_normalize(
            &scores,
            self.top_k as i32,
            /* norm_topk_prob */ true,
        )?;

        Ok((batch_seq, hidden_size, top_indices, normalized_weights))
    }

    fn batched_matmul(&self, x: &Array, w: &Array) -> Result<Array, Exception> {
        let x_expanded = x.reshape(&[x.dim(0), 1, x.dim(1)]);
        let result = pmetal_bridge::compat::ops::matmul(&x_expanded, w);
        Ok(result.squeeze_axes(&[1]))
    }

    #[cfg(test)]
    fn forward_reference(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch_seq: i32 = shape[..shape.len() - 1].iter().product();
        let hidden_size = shape[shape.len() - 1];
        let hidden_flat = x.reshape(&[batch_seq, hidden_size]);
        let (_batch_seq, _hidden_size, top_indices, normalized_weights) =
            self.route_topk(&hidden_flat)?;

        top_indices.eval();
        normalized_weights.eval();
        let expert_indices: Vec<i32> = top_indices.as_slice().to_vec();
        let expert_weights: Vec<f32> = normalized_weights.as_slice().to_vec();
        let mut assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.experts.len()];
        for token_idx in 0..batch_seq as usize {
            for slot in 0..self.top_k {
                let flat_idx = token_idx * self.top_k + slot;
                let expert_id = expert_indices[flat_idx] as usize;
                let weight = expert_weights[flat_idx];
                if expert_id < self.experts.len() {
                    assignments[expert_id].push((token_idx, weight));
                }
            }
        }

        let mut output =
            pmetal_bridge::compat::ops::zeros_dtype(&[batch_seq, hidden_size], hidden_flat.dtype());
        for (expert_idx, expert_assignments) in assignments.iter().enumerate() {
            if expert_assignments.is_empty() {
                continue;
            }
            let token_indices: Vec<i32> = expert_assignments
                .iter()
                .map(|&(idx, _)| idx as i32)
                .collect();
            let weights: Vec<f32> = expert_assignments
                .iter()
                .map(|&(_, weight)| weight)
                .collect();

            let idx_array = Array::from_slice(&token_indices, &[token_indices.len() as i32]);
            let weight_array = Array::from_slice(&weights, &[weights.len() as i32, 1]);
            let expert_input = hidden_flat.take_axis(&idx_array, 0);
            let expert_out = self.experts[expert_idx].forward(&expert_input)?;
            let weighted = expert_out.multiply(&weight_array);
            let updates = weighted.reshape(&[token_indices.len() as i32, 1, hidden_size]);
            output = pmetal_bridge::compat::indexing::scatter_add_single(
                &output, &idx_array, &updates, 0,
            );
        }

        let mut output_shape = shape.to_vec();
        output_shape[shape.len() - 1] = hidden_size;
        Ok(output.reshape(&output_shape))
    }

    fn forward_stacked(&mut self, x: &Array) -> Result<Array, Exception> {
        self.ensure_stacked()?;

        let shape = x.shape();
        let hidden_flat = x.reshape(&[
            shape[..shape.len() - 1].iter().product(),
            shape[shape.len() - 1],
        ]);
        let (batch_seq, hidden_size, top_indices, normalized_weights) =
            self.route_topk(&hidden_flat)?;
        let top_k = self.top_k as i32;
        let mut output =
            pmetal_bridge::compat::ops::zeros_dtype(&[batch_seq, hidden_size], hidden_flat.dtype());

        for slot in 0..top_k {
            let slot_experts =
                pmetal_bridge::compat::ops::slice_axis(&top_indices, -1, slot, slot + 1)
                    .reshape(&[top_indices.dim(0)]);
            let slot_weights =
                pmetal_bridge::compat::ops::slice_axis(&normalized_weights, -1, slot, slot + 1);
            let gate_weights = self
                .stacked_gate_proj
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0);
            let up_weights = self
                .stacked_up_proj
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0);
            let down_weights = self
                .stacked_down_proj
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0);
            let gate_bias = self
                .stacked_gate_bias
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0);
            let up_bias = self
                .stacked_up_bias
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0);
            let down_bias = self
                .stacked_down_bias
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0);

            let gate_out = self
                .batched_matmul(&hidden_flat, &gate_weights)?
                .add(&gate_bias);
            let up_out = self
                .batched_matmul(&hidden_flat, &up_weights)?
                .add(&up_bias);
            let clamped = clamp_swiglu_hidden(&gate_out, &up_out, self.swiglu_limit)?;
            let slot_out = self
                .batched_matmul(&clamped, &down_weights)?
                .add(&down_bias);
            output = output.add(&slot_out.multiply(&slot_weights));
        }

        let mut output_shape = shape.to_vec();
        output_shape[shape.len() - 1] = hidden_size;
        Ok(output.reshape(&output_shape))
    }

    /// Forward pass through MoE.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let _ = self.router_aux_loss_coef;
        self.forward_stacked(x)
    }
}

impl crate::decoder_layer::MlpModule for GptOssMoE {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        GptOssMoE::forward(self, x)
    }
}

/// GPT-OSS decoder layer.
#[derive(Debug)]
pub struct GptOssDecoderLayer {
    /// Self-attention.
    pub self_attn: GptOssAttention,
    /// MoE or dense MLP.
    pub mlp: GptOssMoE,
    /// Input layer norm.
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: nn::RmsNorm,
}
impl_module_params!(GptOssDecoderLayer; self_attn, mlp, input_layernorm, post_attention_layernorm);

impl GptOssDecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: GptOssConfig, layer_idx: usize) -> Result<Self, Exception> {
        let self_attn = GptOssAttention::new(config.clone(), layer_idx)?;

        let mlp = GptOssMoE::new(
            config.hidden_size,
            config.intermediate_size,
            config.num_local_experts,
            config.num_experts_per_tok(),
            config.swiglu_limit,
            config.router_aux_loss_coef,
        )?;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass through decoder layer.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        // Pre-norm attention
        let residual = x;
        let hidden = self.input_layernorm.forward(x);
        let hidden = self.self_attn.forward(&hidden, mask, cache)?;
        let hidden = residual.add(&hidden);

        // Pre-norm MLP
        let residual = &hidden;
        let hidden = self.post_attention_layernorm.forward(&hidden);
        let hidden = self.mlp.forward(&hidden)?;
        Ok(residual.add(&hidden))
    }

    /// Eagerly build the stacked expert cache for this layer's MoE block.
    pub fn init_stacked_moe(&mut self) -> Result<(), Exception> {
        self.mlp.init_stacked_moe()
    }
}

/// GPT-OSS model.
#[derive(Debug)]
pub struct GptOssModel {
    /// Configuration.
    config: GptOssConfig,
    /// Token embeddings.
    pub embed_tokens: nn::Embedding,
    /// Decoder layers.
    pub layers: Vec<GptOssDecoderLayer>,
    /// Final layer norm.
    pub norm: nn::RmsNorm,
}
impl_module_params!(GptOssModel; embed_tokens, layers, norm);

impl GptOssModel {
    /// Create a new GPT-OSS model.
    pub fn new(config: GptOssConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| GptOssDecoderLayer::new(config.clone(), i))
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

    /// Forward pass through the model.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        self.forward_with_capture(input_ids, mask, cache, None)
    }

    /// Forward pass with optional hidden-state capture for DFlash
    /// speculative decoding.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        mut capture: Option<&mut pmetal_mlx::speculative::SpecCapture>,
    ) -> Result<Array, Exception> {
        let mut hidden = self.embed_tokens.forward(input_ids);

        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    hidden = layer.forward(&hidden, mask, Some((cache, layer_idx)))?;
                    if let Some(buf) = capture.as_deref_mut()
                        && buf.wants_hidden_for(layer_idx)
                    {
                        buf.record_hidden(layer_idx, hidden.clone());
                    }
                }
            }
            None => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    hidden = layer.forward(&hidden, mask, None)?;
                    if let Some(buf) = capture.as_deref_mut()
                        && buf.wants_hidden_for(layer_idx)
                    {
                        buf.record_hidden(layer_idx, hidden.clone());
                    }
                }
            }
        }

        Ok(self.norm.forward(&hidden))
    }

    /// Eagerly build stacked expert caches for all decoder layers.
    pub fn init_stacked_moe(&mut self) -> Result<(), Exception> {
        for layer in &mut self.layers {
            layer.init_stacked_moe()?;
        }
        Ok(())
    }

    /// Get the configuration.
    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }
}

/// GPT-OSS for causal language modeling.
#[derive(Debug)]
pub struct GptOssForCausalLM {
    /// Base model.
    pub model: GptOssModel,
    /// Language model head.
    pub lm_head: nn::Linear,
}
impl_module_params!(GptOssForCausalLM; model, lm_head);

impl GptOssForCausalLM {
    /// Create a new GPT-OSS for causal LM.
    pub fn new(config: GptOssConfig) -> Result<Self, Exception> {
        let model = GptOssModel::new(config.clone())?;

        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()?;

        Ok(Self { model, lm_head })
    }

    /// Forward pass returning logits.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward(input_ids, mask, cache)?;
        Ok(self.lm_head.forward(&hidden))
    }

    /// Forward pass that records hidden states into a DFlash capture
    /// buffer at every tapped layer index.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        capture: &mut pmetal_mlx::speculative::SpecCapture,
    ) -> Result<Array, Exception> {
        let hidden = self
            .model
            .forward_with_capture(input_ids, mask, cache, Some(capture))?;
        Ok(self.lm_head.forward(&hidden))
    }

    /// Eagerly build stacked expert caches for all MoE layers.
    pub fn init_stacked_moe(&mut self) -> Result<(), Exception> {
        self.model.init_stacked_moe()
    }

    /// Get the configuration.
    pub fn config(&self) -> &GptOssConfig {
        self.model.config()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> i32 {
        self.model.config.vocab_size
    }

    /// Fused batched decode forward for continuous-batching.
    ///
    /// GPT-OSS interleaves sliding-window and full-attention layers
    /// (default: even layers slide with `config.sliding_window`, odd
    /// layers full-attend). Attention bias is carried by the Q/K/V/O
    /// projections themselves, so no helper changes are needed beyond
    /// the per-layer sliding-window overlay. The stacked MoE block
    /// handles `[N_active, 1, H]` natively via its 2-D flatten path.
    pub fn forward_batched_impl(
        &mut self,
        input_ids: &Array,
        active_indices: &[usize],
        cache: &mut pmetal_mlx::kv_cache::FusedBatchKVCache,
    ) -> Result<Array, Exception> {
        use crate::common::{BatchedGqaAttnCfg, batched_prenorm_layer};

        let cfg = &self.model.config;
        let base_cfg = BatchedGqaAttnCfg::new(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rope_theta,
            1.0,
        );
        let window = cfg.sliding_window;
        // Snapshot per-layer attention types so we don't borrow `self` twice
        // during the mutable iteration over `self.model.layers`.
        let attn_types: Vec<AttentionType> = (0..self.model.layers.len())
            .map(|i| cfg.attention_type_at(i))
            .collect();
        let mut hidden =
            pmetal_bridge::compat::Module::forward(&mut self.model.embed_tokens, input_ids)?;
        for (layer_idx, layer) in self.model.layers.iter_mut().enumerate() {
            let attn_cfg = match attn_types[layer_idx] {
                AttentionType::SlidingAttention => base_cfg.with_sliding_window(window),
                AttentionType::FullAttention => base_cfg,
            };
            hidden = batched_prenorm_layer(
                &hidden,
                &mut layer.input_layernorm,
                &mut layer.self_attn.q_proj,
                &mut layer.self_attn.k_proj,
                &mut layer.self_attn.v_proj,
                &mut layer.self_attn.o_proj,
                None,
                None,
                &mut layer.post_attention_layernorm,
                &mut layer.mlp,
                &attn_cfg,
                cache,
                active_indices,
                layer_idx,
            )?;
        }
        let hidden = pmetal_bridge::compat::Module::forward(&mut self.model.norm, &hidden)?;
        pmetal_bridge::compat::Module::forward(&mut self.lm_head, &hidden)
    }
}

// =============================================================================
// LoRA Support for GPT-OSS
// =============================================================================

use pmetal_core::LoraConfig;
use pmetal_mlx::kernels::fast_lora::{create_lora_params, fused_lora_forward};

/// LoRA-enabled linear layer for GPT-OSS.
#[derive(Debug)]
pub struct LoraLinear {
    /// Base weight [out_features, in_features].
    pub weight: Array,
    /// Optional bias [out_features].
    pub bias: Option<Array>,
    /// LoRA A [rank, in_features].
    pub lora_a: Array,
    /// LoRA B [out_features, rank].
    pub lora_b: Array,
    /// LoRA scale (alpha / rank).
    pub scale: f32,
    /// Whether LoRA is active.
    pub lora_active: bool,
}

impl LoraLinear {
    /// Create LoRA linear from base nn::Linear.
    pub fn from_linear(linear: &nn::Linear, rank: i32, alpha: f32) -> Result<Self, Exception> {
        // Access weight through the parameter
        use pmetal_bridge::compat::ModuleParametersExt;
        let flat = linear.flatten_params();
        let weight = flat
            .get("weight")
            .ok_or_else(|| Exception::from("Missing weight in Linear"))?
            .clone();

        let shape = weight.shape();
        let out_features = shape[0];
        let in_features = shape[1];

        let (lora_a, lora_b) = create_lora_params(in_features, out_features, rank)?;
        let scale = alpha / rank as f32;

        // Get bias if present (need to clone the inner Array)
        let bias = flat.get("bias").cloned();

        Ok(Self {
            weight,
            bias,
            lora_a,
            lora_b,
            scale,
            lora_active: true,
        })
    }

    /// Forward pass with or without LoRA.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let weight = dequantize_fp8_weight_for_compute(&self.weight)?;
        if self.lora_active {
            let out = fused_lora_forward(x, &weight, &self.lora_a, &self.lora_b, self.scale)?;
            if let Some(ref bias) = self.bias {
                Ok(out.add(bias))
            } else {
                Ok(out)
            }
        } else {
            let out = x.matmul(&weight.t());
            if let Some(ref bias) = self.bias {
                Ok(out.add(bias))
            } else {
                Ok(out)
            }
        }
    }

    /// Merge LoRA into base weight (for inference).
    pub fn merge(&mut self) -> Result<(), Exception> {
        if self.lora_active {
            // merged = W + scale * B @ A
            let lora_contrib = self.lora_b.matmul(&self.lora_a);
            let scale_arr = Array::from_f32(self.scale);
            let scaled = lora_contrib.multiply(&scale_arr);
            self.weight = self.weight.add(&scaled);
            self.lora_active = false;
        }
        Ok(())
    }

    /// Get trainable parameters (LoRA A and B).
    pub fn trainable_parameters(&self) -> Vec<(&str, &Array)> {
        if self.lora_active {
            vec![("lora_a", &self.lora_a), ("lora_b", &self.lora_b)]
        } else {
            vec![]
        }
    }

    /// Get mutable references to LoRA parameters.
    pub fn lora_parameters_mut(&mut self) -> (&mut Array, &mut Array) {
        (&mut self.lora_a, &mut self.lora_b)
    }
}

/// GPT-OSS attention with LoRA adapters.
#[derive(Debug)]
pub struct GptOssLoraAttention {
    /// Layer index.
    #[allow(dead_code)] // Retained for debugging and future layer-aware LoRA scheduling
    pub layer_idx: usize,
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale.
    pub scale: f32,
    /// RoPE theta.
    pub rope_theta: f32,
    /// Sliding window size.
    pub sliding_window: i32,
    /// Attention type.
    pub attention_type: AttentionType,
    /// Query projection with LoRA.
    pub q_proj: LoraLinear,
    /// Key projection with LoRA.
    pub k_proj: LoraLinear,
    /// Value projection with LoRA.
    pub v_proj: LoraLinear,
    /// Output projection with LoRA.
    pub o_proj: LoraLinear,
    /// Config reference.
    #[allow(dead_code)] // Retained for LoRA merge/export which needs full arch config
    config: GptOssConfig,
}

impl GptOssLoraAttention {
    /// Create LoRA attention from base attention.
    pub fn from_attention(
        attn: GptOssAttention,
        lora_config: &LoraConfig,
    ) -> Result<Self, Exception> {
        let rank = lora_config.r as i32;
        let alpha = lora_config.alpha;

        // Check which projections should have LoRA
        let has_q = lora_config.target_modules.iter().any(|m| m == "q_proj");
        let has_k = lora_config.target_modules.iter().any(|m| m == "k_proj");
        let has_v = lora_config.target_modules.iter().any(|m| m == "v_proj");
        let has_o = lora_config.target_modules.iter().any(|m| m == "o_proj");

        let mut q_proj = LoraLinear::from_linear(&attn.q_proj, rank, alpha)?;
        let mut k_proj = LoraLinear::from_linear(&attn.k_proj, rank, alpha)?;
        let mut v_proj = LoraLinear::from_linear(&attn.v_proj, rank, alpha)?;
        let mut o_proj = LoraLinear::from_linear(&attn.o_proj, rank, alpha)?;

        // Disable LoRA for non-target modules
        if !has_q {
            q_proj.lora_active = false;
        }
        if !has_k {
            k_proj.lora_active = false;
        }
        if !has_v {
            v_proj.lora_active = false;
        }
        if !has_o {
            o_proj.lora_active = false;
        }

        Ok(Self {
            layer_idx: attn.layer_idx,
            n_heads: attn.n_heads,
            n_kv_heads: attn.n_kv_heads,
            head_dim: attn.head_dim,
            scale: attn.scale,
            rope_theta: attn.rope_theta,
            sliding_window: attn.sliding_window,
            attention_type: attn.attention_type,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            config: attn.config,
        })
    }

    /// Forward pass through LoRA attention.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let mut cache = cache;

        // Project Q, K, V using LoRA layers
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, seq, heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        // Apply RoPE
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q = apply_rope(&q, self.head_dim, false, self.rope_theta, 1.0, offset)?;
        let k = apply_rope(&k, self.head_dim, false, self.rope_theta, 1.0, offset)?;

        // Configure attention based on layer type
        let mask_type = match self.attention_type {
            AttentionType::SlidingAttention => {
                AttentionMaskType::SlidingWindow(self.sliding_window)
            }
            AttentionType::FullAttention => {
                if mask.is_some() {
                    AttentionMaskType::None
                } else {
                    AttentionMaskType::Causal
                }
            }
        };

        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(mask_type);

        if mask.is_none() {
            if let Some((cache_ref, layer_idx)) = cache.as_mut() {
                if let Some(output) =
                    (*cache_ref).try_turboquant_attention(*layer_idx, &q, &k, &v, &attn_config)?
                {
                    let output = output.transpose_axes(&[0, 2, 1, 3]);
                    let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim]);
                    return self.o_proj.forward(&output);
                }
            }
        }

        // Update cache if provided
        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        // Transpose back and project
        let output = output.transpose_axes(&[0, 2, 1, 3]);
        let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim]);

        self.o_proj.forward(&output)
    }

    /// Get trainable parameters from all LoRA layers.
    pub fn trainable_parameters(&self) -> Vec<(String, &Array)> {
        let mut params = Vec::new();
        for (name, arr) in self.q_proj.trainable_parameters() {
            params.push((format!("q_proj.{}", name), arr));
        }
        for (name, arr) in self.k_proj.trainable_parameters() {
            params.push((format!("k_proj.{}", name), arr));
        }
        for (name, arr) in self.v_proj.trainable_parameters() {
            params.push((format!("v_proj.{}", name), arr));
        }
        for (name, arr) in self.o_proj.trainable_parameters() {
            params.push((format!("o_proj.{}", name), arr));
        }
        params
    }

    /// Merge all LoRA weights.
    pub fn merge(&mut self) -> Result<(), Exception> {
        self.q_proj.merge()?;
        self.k_proj.merge()?;
        self.v_proj.merge()?;
        self.o_proj.merge()?;
        Ok(())
    }
}

/// GPT-OSS LoRA-enabled decoder layer.
#[derive(Debug)]
pub struct GptOssLoraDecoderLayer {
    /// Self-attention with LoRA.
    pub self_attn: GptOssLoraAttention,
    /// MoE (no LoRA - too many experts).
    pub mlp: GptOssMoE,
    /// Input layer norm.
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: nn::RmsNorm,
}

impl GptOssLoraDecoderLayer {
    /// Create LoRA decoder layer from base layer.
    pub fn from_layer(
        layer: GptOssDecoderLayer,
        lora_config: &LoraConfig,
    ) -> Result<Self, Exception> {
        let self_attn = GptOssLoraAttention::from_attention(layer.self_attn, lora_config)?;

        Ok(Self {
            self_attn,
            mlp: layer.mlp,
            input_layernorm: layer.input_layernorm,
            post_attention_layernorm: layer.post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        // Pre-norm attention
        let residual = x;
        let hidden = self.input_layernorm.forward(x);
        let hidden = self.self_attn.forward(&hidden, mask, cache)?;
        let hidden = residual.add(&hidden);

        // Pre-norm MLP
        let residual = &hidden;
        let hidden = self.post_attention_layernorm.forward(&hidden);
        let hidden = self.mlp.forward(&hidden)?;
        Ok(residual.add(&hidden))
    }

    /// Get trainable parameters.
    pub fn trainable_parameters(&self) -> Vec<(String, &Array)> {
        self.self_attn.trainable_parameters()
    }

    /// Merge LoRA.
    pub fn merge(&mut self) -> Result<(), Exception> {
        self.self_attn.merge()
    }
}

/// GPT-OSS model with LoRA adapters.
#[derive(Debug)]
pub struct GptOssLoraModel {
    /// Configuration.
    config: GptOssConfig,
    /// Token embeddings.
    pub embed_tokens: nn::Embedding,
    /// LoRA decoder layers.
    pub layers: Vec<GptOssLoraDecoderLayer>,
    /// Final layer norm.
    pub norm: nn::RmsNorm,
}

impl GptOssLoraModel {
    /// Create LoRA model from base model.
    pub fn from_model(model: GptOssModel, lora_config: &LoraConfig) -> Result<Self, Exception> {
        let layers = model
            .layers
            .into_iter()
            .map(|layer| GptOssLoraDecoderLayer::from_layer(layer, lora_config))
            .collect::<Result<Vec<_>, _>>();

        Ok(Self {
            config: model.config,
            embed_tokens: model.embed_tokens,
            layers: layers?,
            norm: model.norm,
        })
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let mut hidden = self.embed_tokens.forward(input_ids);

        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    hidden = layer.forward(&hidden, mask, Some((cache, layer_idx)))?;
                }
            }
            None => {
                for layer in self.layers.iter_mut() {
                    hidden = layer.forward(&hidden, mask, None)?;
                }
            }
        }

        Ok(self.norm.forward(&hidden))
    }

    /// Get all trainable parameters.
    pub fn trainable_parameters(&self) -> Vec<(String, &Array)> {
        let mut params = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, arr) in layer.trainable_parameters() {
                params.push((format!("layers.{}.self_attn.{}", i, name), arr));
            }
        }
        params
    }

    /// Merge all LoRA weights.
    pub fn merge(&mut self) -> Result<(), Exception> {
        for layer in &mut self.layers {
            layer.merge()?;
        }
        Ok(())
    }

    /// Count trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.trainable_parameters()
            .iter()
            .map(|(_, arr)| arr.size())
            .sum()
    }

    /// Get configuration.
    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }
}

/// GPT-OSS for causal LM with LoRA.
#[derive(Debug)]
pub struct GptOssLoraForCausalLM {
    /// LoRA model.
    pub model: GptOssLoraModel,
    /// Language model head (frozen).
    pub lm_head: nn::Linear,
}

impl GptOssLoraForCausalLM {
    /// Create LoRA model from base model.
    pub fn from_model(
        base: GptOssForCausalLM,
        lora_config: &LoraConfig,
    ) -> Result<Self, Exception> {
        let model = GptOssLoraModel::from_model(base.model, lora_config)?;

        Ok(Self {
            model,
            lm_head: base.lm_head,
        })
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward(input_ids, mask, cache)?;
        Ok(self.lm_head.forward(&hidden))
    }

    /// Get trainable parameters.
    pub fn trainable_parameters(&self) -> Vec<(String, &Array)> {
        self.model.trainable_parameters()
    }

    /// Merge LoRA.
    pub fn merge(&mut self) -> Result<(), Exception> {
        self.model.merge()
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }
}

impl GptOssForCausalLM {
    /// Convert to LoRA-enabled model.
    pub fn into_lora(self, lora_config: &LoraConfig) -> Result<GptOssLoraForCausalLM, Exception> {
        GptOssLoraForCausalLM::from_model(self, lora_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_config_defaults() {
        let config = GptOssConfig::default();
        assert_eq!(config.hidden_size, 2880);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_local_experts, 32);
        assert_eq!(config.experts_per_token, 4);
        assert_eq!(config.sliding_window, 128);
        assert_eq!(config.swiglu_limit, 7.0);
    }

    #[test]
    fn test_attention_type_alternating() {
        let config = GptOssConfig::default();
        assert_eq!(config.attention_type_at(0), AttentionType::SlidingAttention);
        assert_eq!(config.attention_type_at(1), AttentionType::FullAttention);
        assert_eq!(config.attention_type_at(2), AttentionType::SlidingAttention);
        assert_eq!(config.attention_type_at(3), AttentionType::FullAttention);
    }

    #[test]
    fn test_gpt_oss_120b_config() {
        let config = GptOssConfig::gpt_oss_120b();
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.num_local_experts, 128);
    }

    fn tiny_gpt_oss_config() -> GptOssConfig {
        GptOssConfig {
            hidden_size: 32,
            intermediate_size: 48,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 8,
            vocab_size: 64,
            num_local_experts: 4,
            experts_per_token: 2,
            num_experts_per_tok: Some(2),
            sliding_window: 16,
            ..GptOssConfig::default()
        }
    }

    #[test]
    #[serial]
    fn test_gpt_oss_moe_forward_shape() {
        let config = tiny_gpt_oss_config();
        let mut moe = GptOssMoE::new(
            config.hidden_size,
            config.intermediate_size,
            config.num_local_experts,
            config.num_experts_per_tok(),
            config.swiglu_limit,
            config.router_aux_loss_coef,
        )
        .unwrap();
        let x = Array::zeros_f32(&[2, 5, config.hidden_size]);
        let out = moe.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 5, config.hidden_size]);
        assert!(moe.has_stacked_moe());
    }

    #[test]
    #[serial]
    fn test_gpt_oss_moe_stacked_matches_reference() {
        let config = tiny_gpt_oss_config();
        let mut moe = GptOssMoE::new(
            config.hidden_size,
            config.intermediate_size,
            config.num_local_experts,
            config.num_experts_per_tok(),
            config.swiglu_limit,
            config.router_aux_loss_coef,
        )
        .unwrap();
        let x = pmetal_bridge::compat::random::uniform_range(
            -1.0,
            1.0,
            &[2, 5, config.hidden_size],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let reference = moe.forward_reference(&x).unwrap();
        let fast = moe.forward(&x).unwrap();
        reference.eval().unwrap();
        fast.eval().unwrap();

        let max_diff = fast
            .subtract(&reference)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        max_diff.eval().unwrap();
        assert!(
            max_diff.item::<f32>() < 1e-4,
            "gpt-oss stacked MoE drifted from reference"
        );
    }

    #[test]
    #[serial]
    fn test_gpt_oss_moe_cache_refreshes_after_weight_change() {
        let config = tiny_gpt_oss_config();
        let mut moe = GptOssMoE::new(
            config.hidden_size,
            config.intermediate_size,
            config.num_local_experts,
            config.num_experts_per_tok(),
            config.swiglu_limit,
            config.router_aux_loss_coef,
        )
        .unwrap();
        let x = pmetal_bridge::compat::random::uniform_range(
            -1.0,
            1.0,
            &[1, 4, config.hidden_size],
            pmetal_bridge::compat::Dtype::Float32,
        );

        let _ = moe.forward(&x).unwrap();
        moe.experts[0].gate_proj.weight =
            pmetal_bridge::compat::module::Param::new(Array::zeros_f32(&[
                config.intermediate_size,
                config.hidden_size,
            ]));

        let reference = moe.forward_reference(&x).unwrap();
        let fast = moe.forward(&x).unwrap();
        reference.eval().unwrap();
        fast.eval().unwrap();

        let max_diff = fast
            .subtract(&reference)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        max_diff.eval().unwrap();
        assert!(
            max_diff.item::<f32>() < 1e-4,
            "gpt-oss stacked cache failed to refresh after weight change"
        );
    }

    #[test]
    #[serial]
    fn test_gpt_oss_full_model_init_stacked_moe_preserves_forward() {
        let config = tiny_gpt_oss_config();
        let mut model = GptOssForCausalLM::new(config.clone()).unwrap();
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);

        let reference = model.forward(&input_ids, None, None).unwrap();
        model.init_stacked_moe().unwrap();
        let fast = model.forward(&input_ids, None, None).unwrap();
        reference.eval().unwrap();
        fast.eval().unwrap();

        let max_diff = fast
            .subtract(&reference)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        max_diff.eval().unwrap();
        assert!(
            max_diff.item::<f32>() < 1e-4,
            "gpt-oss full model drifted after stacked init"
        );
    }

    // LoRA Tests
    #[test]
    fn test_lora_config_default_target_modules() {
        let lora_config = LoraConfig::default();
        // Default targets attention projections
        assert!(
            lora_config
                .target_modules
                .iter()
                .any(|m| m == "q_proj" || m.contains("q_proj"))
        );
    }

    #[test]
    fn test_lora_linear_creation() {
        let linear = nn::LinearBuilder::new(256, 512).bias(true).build().unwrap();

        let lora_linear = LoraLinear::from_linear(&linear, 8, 16.0).unwrap();

        assert_eq!(lora_linear.lora_a.shape(), &[8, 256]);
        assert_eq!(lora_linear.lora_b.shape(), &[512, 8]);
        assert!(lora_linear.lora_active);
        assert_eq!(lora_linear.scale, 16.0 / 8.0);
    }

    #[test]
    fn test_lora_linear_forward_shape() {
        let linear = nn::LinearBuilder::new(64, 128).bias(false).build().unwrap();

        let lora_linear = LoraLinear::from_linear(&linear, 4, 8.0).unwrap();
        let x = Array::zeros_f32(&[2, 8, 64]);
        let out = lora_linear.forward(&x).unwrap();

        assert_eq!(out.shape(), &[2, 8, 128]);
    }

    #[test]
    fn test_lora_linear_merge() {
        let linear = nn::LinearBuilder::new(32, 64).bias(false).build().unwrap();

        let mut lora_linear = LoraLinear::from_linear(&linear, 4, 8.0).unwrap();
        assert!(lora_linear.lora_active);

        lora_linear.merge().unwrap();
        assert!(!lora_linear.lora_active);

        // Forward should still work
        let x = Array::zeros_f32(&[1, 4, 32]);
        let out = lora_linear.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_lora_trainable_params() {
        let linear = nn::LinearBuilder::new(32, 64).bias(false).build().unwrap();

        let lora_linear = LoraLinear::from_linear(&linear, 4, 8.0).unwrap();
        let params = lora_linear.trainable_parameters();

        assert_eq!(params.len(), 2); // lora_a and lora_b
        assert_eq!(params[0].0, "lora_a");
        assert_eq!(params[1].0, "lora_b");
    }
}
