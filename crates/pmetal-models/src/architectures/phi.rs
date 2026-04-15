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
use pmetal_bridge::compat::nn::{Embedding, Linear, RmsNorm, RopeBuilder};
use pmetal_bridge::compat::{
    Array, Exception, ModuleParameters, ModuleParametersExt, Param, fast, nn, ops, random,
};
use pmetal_bridge::impl_module_params;

use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa, rope::apply_rope};
use pmetal_mlx::kv_cache::KVCache;

use crate::traits::{CausalLMModel, ModelConfig};
use std::collections::HashMap;

/// Compute SuRoPE precomputed frequencies for Phi-3 128K / Phi-3.5 models.
///
/// SuRoPE (Scaled Uniform RoPE) from the Phi-3 128K paper uses per-dimension
/// scaling factors. The frequencies passed to `pmetal_bridge::compat::fast::rope` are:
///   `freqs[i] = factor[i] * base^(2i / rope_dim)`
/// where `factor` is either `short_factor` or `long_factor`.
///
/// In practice mlx-lm always uses `long_factor` and applies a single mscale
/// attention scalar at the Q/K level. The `mscale` is:
///   `sqrt(1 + ln(max_pos / orig_max_pos) / ln(orig_max_pos))`
///
/// Returns `(freqs, mscale)`.
fn compute_su_rope_freqs(
    scaling: &PhiRopeScaling,
    rope_dim: i32,
    rope_theta: f32,
    max_position_embeddings: i32,
    original_max_position_embeddings: i32,
) -> Result<(Array, f32), Exception> {
    let half = (rope_dim / 2) as usize;
    let long_factor = &scaling.long_factor;

    // Compute INVERSE frequencies: 1 / (factor * base^(2i/rope_dim))
    // SuRoPE scales the inverse frequencies, NOT the periods.
    // inv_freq[i] = 1 / (long_factor[i] * theta^(2i/D))
    let mut freqs = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = (2 * i) as f32 / rope_dim as f32;
        let base_period = rope_theta.powf(exponent); // theta^(2i/D) = period
        let factor = long_factor.get(i).copied().unwrap_or(1.0);
        // Inverse frequency scaled by factor
        freqs.push(1.0 / (factor * base_period));
    }
    let freqs_arr = Array::from_slice(&freqs, &[half as i32]);

    // mscale = sqrt(1 + ln(factor) / ln(original_max_pos))
    // where factor = max_pos / original_max_pos
    let factor = max_position_embeddings as f32 / original_max_position_embeddings as f32;
    let mscale = if factor <= 1.0 {
        1.0_f32
    } else {
        (1.0 + factor.ln() / (original_max_position_embeddings as f32).ln()).sqrt()
    };

    Ok((freqs_arr, mscale))
}

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
#[derive(Debug)]
pub struct PhiRMSNorm {
    pub weight: Param<Array>,
    pub eps: f32,
}
impl_module_params!(PhiRMSNorm; weight);

impl PhiRMSNorm {
    /// Create a new RMS LayerNorm.
    pub fn new(hidden_size: i32, eps: f32) -> Self {
        let weight = Param::new(Array::ones_f32(&[hidden_size]));
        Self { weight, eps }
    }
}

impl PhiRMSNorm {
    /// Forward pass for RMS LayerNorm.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let variance = x.square().mean_axis(-1, true);
        let eps = Array::from_f32(self.eps);
        let x_normed = x.divide(&variance.add(&eps).sqrt());
        Ok(x_normed.multiply(&self.weight))
    }
}

/// Phi attention with partial RoPE (and optional SuRoPE for 128K context models).
#[derive(Debug)]
pub struct PhiAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    /// Standard RoPE module (used when `su_freqs` is None).
    pub rope: pmetal_bridge::compat::nn::Rope,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,
    /// SuRoPE precomputed per-dimension frequencies (shape [rope_dim/2]).
    /// Present only for Phi-3 128K / Phi-3.5 models with `rope_scaling` set.
    pub su_freqs: Option<Array>,
    /// Attention mscale applied to Q and K when using SuRoPE.
    pub su_mscale: f32,
}
impl_module_params!(PhiAttention; q_proj, k_proj, v_proj, o_proj);

impl PhiAttention {
    /// Create a new Phi attention layer.
    pub fn new(config: &PhiConfig) -> Result<Self, Exception> {
        let head_dim = config.head_dim();
        let rope_dim = config.rope_dim();
        let rope_theta = config.rope_theta;

        let q_proj =
            nn::LinearBuilder::new(config.hidden_size, config.num_attention_heads * head_dim)
                .bias(config.qkv_bias)
                .build()?;
        let k_proj =
            nn::LinearBuilder::new(config.hidden_size, config.num_key_value_heads * head_dim)
                .bias(config.qkv_bias)
                .build()?;
        let v_proj =
            nn::LinearBuilder::new(config.hidden_size, config.num_key_value_heads * head_dim)
                .bias(config.qkv_bias)
                .build()?;
        let o_proj =
            nn::LinearBuilder::new(config.num_attention_heads * head_dim, config.hidden_size)
                .bias(false)
                .build()?;

        let rope = RopeBuilder::new(rope_dim)
            .traditional(false)
            .base(rope_theta)
            .scale(1.0)
            .build()?;

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute SuRoPE frequencies if rope_scaling is provided (Phi-3 128K / Phi-3.5)
        let (su_freqs, su_mscale) = if let Some(ref rope_scaling) = config.rope_scaling {
            if rope_scaling.scaling_type == "su"
                || rope_scaling.scaling_type == "longrope"
                || rope_scaling.scaling_type == "linear"
            {
                let orig_max = config
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings);
                match compute_su_rope_freqs(
                    rope_scaling,
                    rope_dim,
                    rope_theta,
                    config.max_position_embeddings,
                    orig_max,
                ) {
                    Ok((freqs, mscale)) => (Some(freqs), mscale),
                    Err(_) => (None, 1.0),
                }
            } else {
                (None, 1.0)
            }
        } else {
            (None, 1.0)
        };

        Ok(Self {
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
            su_freqs,
            su_mscale,
        })
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
        let mut cache = cache;
        let (batch, seq_len, _) = (x.dim(0), x.dim(1), x.dim(2));

        // Project Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq, n_heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Apply partial RoPE split first
        let (q_rope_raw, q_pass) = self.split_rotary(&q);
        let (k_rope_raw, k_pass) = self.split_rotary(&k); // infallible

        // Apply SuRoPE mscale to the rotary portion only (matching the Python reference:
        // `x[..., :self.dim] = self._scale * x[..., :self.dim]` before rope call)
        let (q_rope_raw, k_rope_raw) = if self.su_freqs.is_some() && self.su_mscale != 1.0 {
            let mscale = Array::from_f32(self.su_mscale);
            (q_rope_raw.multiply(&mscale), k_rope_raw.multiply(&mscale))
        } else {
            (q_rope_raw, k_rope_raw)
        };

        let (q_rope, k_rope) = if let Some(ref _su_freqs) = self.su_freqs {
            // SuRoPE path: use standard rope with rope_theta (su_freqs baked into theta via scaling)
            let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
            let qr = apply_rope(
                &q_rope_raw,
                self.rope_dim,
                false,
                self.rope_theta,
                self.su_mscale,
                offset,
            )?;
            let kr = apply_rope(
                &k_rope_raw,
                self.rope_dim,
                false,
                self.rope_theta,
                self.su_mscale,
                offset,
            )?;
            (qr, kr)
        } else if let Some((ref cache_ref, _)) = cache {
            let offset = cache_ref.rope_offset();
            let qr = apply_rope(
                &q_rope_raw,
                self.rope_dim,
                false,
                self.rope_theta,
                1.0,
                offset,
            )?;
            let kr = apply_rope(
                &k_rope_raw,
                self.rope_dim,
                false,
                self.rope_theta,
                1.0,
                offset,
            )?;
            (qr, kr)
        } else {
            let qr = self.rope.forward(&q_rope_raw, 0);
            let kr = self.rope.forward(&k_rope_raw, 0);
            (qr, kr)
        };

        // Concatenate RoPE and pass-through parts
        let q = pmetal_bridge::compat::ops::concatenate_axis(&[&q_rope, &q_pass], -1);
        let k = pmetal_bridge::compat::ops::concatenate_axis(&[&k_rope, &k_pass], -1);

        // Transpose for attention: [batch, n_heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k_transposed = k.transpose_axes(&[0, 2, 1, 3]);
        let v_transposed = v.transpose_axes(&[0, 2, 1, 3]);

        // Use fused attention
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });

        if mask.is_none() {
            if let Some((cache_ref, layer_idx)) = cache.as_mut() {
                if let Some(attn_output) = (*cache_ref).try_turboquant_attention(
                    *layer_idx,
                    &q,
                    &k_transposed,
                    &v_transposed,
                    &attn_config,
                )? {
                    let attn_output = attn_output.transpose_axes(&[0, 2, 1, 3]);
                    let attn_output =
                        attn_output.reshape(&[batch, seq_len, self.n_heads * self.head_dim]);
                    return Ok(self.o_proj.forward(&attn_output));
                }
            }
        }

        // Update KV cache
        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k_transposed, &v_transposed)?
        } else {
            (k_transposed, v_transposed)
        };

        let attn_output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        // Transpose back and project
        let attn_output = attn_output.transpose_axes(&[0, 2, 1, 3]);
        let attn_output = attn_output.reshape(&[batch, seq_len, self.n_heads * self.head_dim]);

        Ok(self.o_proj.forward(&attn_output))
    }

    /// Split tensor into RoPE and pass-through parts.
    fn split_rotary(&self, x: &Array) -> (Array, Array) {
        let rope_part = pmetal_bridge::compat::ops::slice_last_to(x, self.rope_dim as i32);
        let pass_part = pmetal_bridge::compat::ops::slice_last_from(x, self.rope_dim as i32);
        (rope_part, pass_part)
    }
}

/// Phi MLP with SwiGLU or GELU.
#[derive(Debug)]
pub struct PhiMLP {
    pub gate_up_proj: Linear,
    pub down_proj: Linear,
    pub activation: PhiActivation,
    pub intermediate_size: i32,
}
impl_module_params!(PhiMLP; gate_up_proj, down_proj);

impl PhiMLP {
    /// Create a new Phi MLP.
    pub fn new(config: &PhiConfig) -> Result<Self, Exception> {
        // For SwiGLU, gate_up_proj projects to 2x intermediate_size (gate + up)
        let proj_size = match config.hidden_act {
            PhiActivation::SwiGLU => config.intermediate_size * 2,
            _ => config.intermediate_size,
        };

        let gate_up_proj = nn::LinearBuilder::new(config.hidden_size, proj_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            activation: config.hidden_act,
            intermediate_size: config.intermediate_size,
        })
    }
}

impl PhiMLP {
    /// Forward pass through MLP.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let hidden = self.gate_up_proj.forward(x);

        let activated = match self.activation {
            PhiActivation::SwiGLU => {
                // Split into gate and up projections
                let gate = pmetal_bridge::compat::ops::slice_last_to(
                    &hidden,
                    self.intermediate_size as i32,
                );
                let up = pmetal_bridge::compat::ops::slice_last_from(
                    &hidden,
                    self.intermediate_size as i32,
                );
                // SwiGLU: silu(gate) * up
                let gate_activated = pmetal_bridge::compat::ops::sigmoid(&gate).multiply(&gate);
                gate_activated.multiply(&up)
            }
            PhiActivation::GeluApprox => pmetal_bridge::compat::nn::gelu(&hidden), // gelu_approx not in mlx-rs
            PhiActivation::GeluExact => pmetal_bridge::compat::nn::gelu(&hidden),
        };

        Ok(self.down_proj.forward(&activated))
    }
}

/// Phi decoder layer.
#[derive(Debug)]
pub struct PhiDecoderLayer {
    pub self_attn: PhiAttention,
    pub mlp: PhiMLP,
    pub input_layernorm: PhiRMSNorm,
    pub post_attention_layernorm: PhiRMSNorm,
}
impl_module_params!(PhiDecoderLayer; self_attn, mlp, input_layernorm, post_attention_layernorm);

impl PhiDecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: &PhiConfig) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: PhiAttention::new(config)?,
            mlp: PhiMLP::new(config)?,
            input_layernorm: PhiRMSNorm::new(config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: PhiRMSNorm::new(config.hidden_size, config.rms_norm_eps),
        })
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
        let hidden = residual.add(&hidden);

        // Pre-norm MLP
        let residual = hidden.clone();
        let hidden = self.post_attention_layernorm.forward(&hidden)?;
        let hidden = self.mlp.forward(&hidden)?;
        Ok(residual.add(&hidden))
    }
}

/// Phi base model.
#[derive(Debug)]
pub struct PhiModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<PhiDecoderLayer>,
    pub norm: PhiRMSNorm,
    pub config: PhiConfig,
}
impl_module_params!(PhiModel; embed_tokens, layers, norm);

impl PhiModel {
    /// Create a new Phi model.
    pub fn new(config: PhiConfig) -> Result<Self, Exception> {
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size).unwrap();
        let layers = (0..config.num_hidden_layers)
            .map(|_| PhiDecoderLayer::new(&config))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = PhiRMSNorm::new(config.hidden_size, config.rms_norm_eps);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            config,
        })
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
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        self.forward_with_capture(input_ids, mask, cache, None)
    }

    /// Forward pass with optional hidden-state capture for DFlash
    /// speculative decoding. Identical to [`forward_with_cache`] when
    /// `capture` is `None`.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
        mut capture: Option<&mut pmetal_mlx::speculative::SpecCapture>,
    ) -> Result<Array, Exception> {
        let mut hidden = self.embed_tokens.forward(input_ids);

        // Create causal mask if not provided and not using cache
        let mask_owned;
        let mask = if mask.is_none() && cache.is_none() {
            let seq_len = input_ids.dim(1);
            mask_owned = create_causal_mask(seq_len)?;
            Some(&mask_owned)
        } else {
            mask
        };

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let c = cache.as_deref_mut().map(|c| (c, idx));
            hidden = layer.forward_with_cache(&hidden, mask, c)?;
            if let Some(buf) = capture.as_deref_mut()
                && buf.wants_hidden_for(idx)
            {
                buf.record_hidden(idx, hidden.clone());
            }
        }

        self.norm.forward(&hidden)
    }
}

/// Phi for causal language modeling.
#[derive(Debug)]
pub struct PhiForCausalLM {
    pub model: PhiModel,
    pub lm_head: Linear,
}
impl_module_params!(PhiForCausalLM; model, lm_head);

impl PhiForCausalLM {
    /// Create a new Phi causal LM.
    pub fn new(config: PhiConfig) -> Result<Self, Exception> {
        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()?;
        let model = PhiModel::new(config)?;
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
        Ok(self.lm_head.forward(&hidden))
    }

    /// Forward pass that records hidden states into a DFlash capture
    /// buffer at every requested layer index.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        capture: &mut pmetal_mlx::speculative::SpecCapture,
    ) -> Result<Array, Exception> {
        let hidden =
            self.model
                .forward_with_capture(input_ids, mask, cache, Some(capture))?;
        Ok(self.lm_head.forward(&hidden))
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
        crate::loader::load_phi_weights(self, weights)
            .map_err(|e: crate::loader::LoadError| Exception::custom(e.to_string()))
    }

    fn eval(&self) -> Result<(), Exception> {
        pmetal_bridge::compat::ModuleParametersExt::eval(self)
    }
}

/// Re-export the shared causal mask utility.
use super::utils::create_causal_mask;

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
        let x = pmetal_bridge::compat::random::normal(
            &[2, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );

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

        let mut attn = PhiAttention::new(&config).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[2, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );

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

        let mut mlp = PhiMLP::new(&config).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[2, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );

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

        let mut layer = PhiDecoderLayer::new(&config).unwrap();
        let x = pmetal_bridge::compat::random::normal(
            &[2, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );

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

        let mut model = PhiModel::new(config).unwrap();
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
