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
use crate::decoder_layer::{
    AttentionModule, DecoderLayer, MlpModule, NormModule, std_pre_norm_forward,
};
use pmetal_bridge::compat::nn::{Embedding, Linear, RmsNorm, RopeBuilder};
use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParameters, ModuleParametersExt, Param, fast, nn, ops, random,
};
use pmetal_bridge::impl_module_params;

use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{apply_rope, apply_rope_with_freqs},
};
use pmetal_mlx::kv_cache::KVCache;

use crate::architectures::utils::{Activation, resolve_activation};
use crate::traits::{CausalLMModel, ModelConfig};
use std::collections::HashMap;

/// LongRoPE / SuRoPE state for Phi-3 128K, Phi-3.5 and Phi-4 models.
///
/// LongRoPE ships **two** per-dimension scaling vectors and picks between them
/// by sequence length, so one precomputed table is not enough. Holding both and
/// choosing per forward is what makes short-context runs match the reference.
#[derive(Debug)]
pub struct LongRopeFreqs {
    /// Inverse frequencies from `short_factor`, for sequences that stay within
    /// the pretraining length.
    pub short: Array,
    /// Inverse frequencies from `long_factor`, for sequences that pass it.
    pub long: Array,
    /// The pretraining length the two tables switch at.
    pub original_max_position: i32,
    /// Attention mscale applied to Q and K. Independent of which table is in
    /// use — `transformers` computes it once at construction and never revises
    /// it when the branch flips.
    pub mscale: f32,
}

impl LongRopeFreqs {
    /// Pick the table for a forward whose highest absolute position is
    /// `max_position` (0-based), i.e. an effective length of `max_position + 1`.
    ///
    /// Mirrors `transformers`' `longrope_frequency_update`, which re-decides on
    /// **every** forward from `max(position_ids) + 1`. A decode that crosses the
    /// boundary therefore rotates later tokens with `long` while the cache still
    /// holds `short`-rotated keys; that is the reference's behaviour, quirk and
    /// all, and diverging from it is what this reproduces.
    pub fn table_for(&self, max_position: i32) -> &Array {
        if max_position + 1 > self.original_max_position {
            &self.long
        } else {
            &self.short
        }
    }
}

/// Compute both LongRoPE frequency tables and the attention mscale.
///
/// Each table holds INVERSE frequencies — LongRoPE scales those, not the
/// periods:
///   `inv_freq[i] = 1 / (factor[i] * theta^(2i / rope_dim))`
///
/// The mscale is `sqrt(1 + ln(factor) / ln(orig_max_pos))` for
/// `factor = max_pos / orig_max_pos`, matching `transformers`'
/// `_compute_longrope_parameters` when the config states no explicit
/// `attention_factor`.
///
/// Public so `pmetal-lora`'s Phi can build the same tables. A second
/// implementation of this is how a 128k Phi comes to be fine-tuned against
/// positions it will never be served with.
pub fn compute_longrope_freqs(
    scaling: &PhiRopeScaling,
    rope_dim: i32,
    rope_theta: f32,
    max_position_embeddings: i32,
    original_max_position_embeddings: i32,
) -> Result<LongRopeFreqs, Exception> {
    let half = (rope_dim / 2) as usize;
    let table = |factors: &[f32]| {
        let freqs: Vec<f32> = (0..half)
            .map(|i| {
                let exponent = (2 * i) as f32 / rope_dim as f32;
                let base_period = rope_theta.powf(exponent); // theta^(2i/D) = period
                let factor = factors.get(i).copied().unwrap_or(1.0);
                1.0 / (factor * base_period)
            })
            .collect();
        Array::from_slice(&freqs, &[half as i32])
    };

    let factor = max_position_embeddings as f32 / original_max_position_embeddings as f32;
    let mscale = if factor <= 1.0 {
        1.0_f32
    } else {
        (1.0 + factor.ln() / (original_max_position_embeddings as f32).ln()).sqrt()
    };

    Ok(LongRopeFreqs {
        short: table(&scaling.short_factor),
        long: table(&scaling.long_factor),
        original_max_position: original_max_position_embeddings,
        mscale,
    })
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
///
/// The GELU spellings map to HuggingFace's `ACT2FN` entries, which are three
/// distinct functions — see [`resolve_activation`]. Phi-2's released config
/// says `"gelu_new"` (the tanh approximation), so that spelling has to
/// deserialize; before this was aliased, loading a stock Phi-2 `config.json`
/// failed outright on an unknown variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PhiActivation {
    /// SwiGLU activation (Phi-3).
    #[default]
    #[serde(rename = "silu", alias = "swiglu")]
    SwiGLU,
    /// Tanh-approximated GELU (Phi-2's `"gelu_new"`).
    #[serde(
        rename = "gelu_new",
        alias = "gelu_approx",
        alias = "gelu_pytorch_tanh",
        alias = "gelu_fast"
    )]
    GeluApprox,
    /// Exact (erf) GELU.
    #[serde(rename = "gelu")]
    GeluExact,
}

impl PhiActivation {
    /// The `ACT2FN` name this variant corresponds to.
    pub fn act_name(self) -> &'static str {
        match self {
            Self::SwiGLU => "silu",
            Self::GeluApprox => "gelu_new",
            Self::GeluExact => "gelu",
        }
    }

    /// The pointwise function this variant applies.
    ///
    /// For [`PhiActivation::SwiGLU`] this is the gate activation, which the
    /// gated MLP applies to half the projection rather than all of it. Shared
    /// with the LoRA and QLoRA MLPs so all three agree on what a config's
    /// activation string means.
    pub fn act_fn(self) -> Activation {
        resolve_activation(self.act_name()).expect("act_name is always supported")
    }
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
    /// Scaling type. HF stores this under the JSON key `"type"`.
    #[serde(rename = "type")]
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
    ///
    /// Also the `Default`, and therefore the value every field a Phi config
    /// omits falls back to — `PhiConfig` is `#[serde(default)]`.
    /// `partial_rotary_factor` is the one that matters: Phi-3 is *full* rotary
    /// and `microsoft/Phi-3-mini-4k-instruct` ships `"partial_rotary_factor":
    /// null`, so this default is what gets used. `transformers` reads the same
    /// absent field as `1.0` (`configuration_phi3.py` setdefault), and 0.5 here
    /// rotated half of every 96-wide head. Configs that state a value still
    /// win: Phi-4-mini says 0.75, Phi-2 says 0.4.
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
            partial_rotary_factor: 1.0,
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

impl NormModule for PhiRMSNorm {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        PhiRMSNorm::forward(self, x)
    }
}

/// Phi attention with partial RoPE (and optional SuRoPE for 128K context models).
#[derive(Debug)]
pub struct PhiAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    /// Standard RoPE module (used when `long_rope` is None).
    pub rope: pmetal_bridge::compat::nn::Rope,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,
    /// LongRoPE / SuRoPE tables, present only for models with `rope_scaling`
    /// set (Phi-3 128K, Phi-3.5, Phi-4).
    pub long_rope: Option<LongRopeFreqs>,
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

        // Compute LongRoPE tables if rope_scaling is provided (Phi-3 128K / Phi-3.5 / Phi-4)
        let long_rope = config
            .rope_scaling
            .as_ref()
            .filter(|scaling| matches!(scaling.scaling_type.as_str(), "su" | "longrope" | "linear"))
            .and_then(|scaling| {
                let orig_max = config
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings);
                compute_longrope_freqs(
                    scaling,
                    rope_dim,
                    rope_theta,
                    config.max_position_embeddings,
                    orig_max,
                )
                .ok()
            });

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
            long_rope,
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

        // Reshape to [batch, seq, n_heads, head_dim] then transpose to
        // [batch, n_heads, seq, head_dim] BEFORE applying RoPE. MLX's
        // `fast::rope` treats axis -2 as the seq axis, so the transpose
        // must happen first; otherwise partial-RoPE on `[B, S, H, rope_dim]`
        // misreads the heads dim as seq for offset > 0 (silent off-by-head).
        let q = q
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let k = k
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let v_transposed = v
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        // Partial RoPE split on the last axis (head_dim → rope_dim + pass).
        let (q_rope_raw, q_pass) = self.split_rotary(&q);
        let (k_rope_raw, k_pass) = self.split_rotary(&k);

        // Apply SuRoPE mscale to the rotary portion only (matches the Python
        // reference: `x[..., :self.dim] = self._scale * x[..., :self.dim]`).
        let mscale = self.long_rope.as_ref().map_or(1.0, |lr| lr.mscale);
        let (q_rope_raw, k_rope_raw) = if mscale != 1.0 {
            let mscale = Array::from_f32(mscale);
            (q_rope_raw.multiply(&mscale), k_rope_raw.multiply(&mscale))
        } else {
            (q_rope_raw, k_rope_raw)
        };

        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        // SuRoPE/LongRoPE: rotate with the per-dimension factor-scaled inverse
        // frequencies, choosing the short or long table by how far this forward
        // actually reaches. The mscale above is the *value* scale; it must NOT
        // also be passed as a position `scale` to the rotation (that was a
        // double-application bug — and plain `apply_rope` ignored the tables
        // entirely, falling back to un-scaled base frequencies).
        let (q_rope, k_rope) = if let Some(ref long_rope) = self.long_rope {
            let freqs = long_rope.table_for(offset + seq_len - 1);
            (
                apply_rope_with_freqs(&q_rope_raw, freqs, self.rope_dim, false, offset)?,
                apply_rope_with_freqs(&k_rope_raw, freqs, self.rope_dim, false, offset)?,
            )
        } else {
            (
                apply_rope(
                    &q_rope_raw,
                    self.rope_dim,
                    false,
                    self.rope_theta,
                    1.0,
                    offset,
                )?,
                apply_rope(
                    &k_rope_raw,
                    self.rope_dim,
                    false,
                    self.rope_theta,
                    1.0,
                    offset,
                )?,
            )
        };

        // Concatenate RoPE and pass-through parts back into the head_dim axis.
        let q = pmetal_bridge::compat::ops::concatenate_axis(&[&q_rope, &q_pass], -1);
        let k_transposed = pmetal_bridge::compat::ops::concatenate_axis(&[&k_rope, &k_pass], -1);

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
            // `ACT2FN[hidden_act]` — `"gelu_new"` is the tanh approximation and
            // `"gelu"` the exact erf definition. They are not interchangeable.
            PhiActivation::GeluApprox | PhiActivation::GeluExact => {
                (self.activation.act_fn())(&hidden)
            }
        };

        Ok(self.down_proj.forward(&activated))
    }
}

impl MlpModule for PhiMLP {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        PhiMLP::forward(self, x)
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
    ///
    /// Delegates to the shared pre-norm skeleton —
    /// see `crate::decoder_layer::std_pre_norm_forward`.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        std_pre_norm_forward(
            &mut self.input_layernorm,
            &mut self.self_attn,
            &mut self.post_attention_layernorm,
            &mut self.mlp,
            x,
            mask,
            cache,
        )
    }
}

impl AttentionModule for PhiAttention {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        PhiAttention::forward_with_cache(self, x, mask, cache)
    }
}

impl DecoderLayer for PhiDecoderLayer {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        PhiDecoderLayer::forward_with_cache(self, x, mask, cache)
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
    /// `None` when the config ties word embeddings, because a tied checkpoint
    /// ships no `lm_head.weight` at all. Holding an unconditional `Linear` here
    /// left Phi-4-mini's head at its random init — the trunk was bit-identical
    /// across seeds while every one of its 3.2M logits moved.
    pub lm_head: Option<Linear>,
}
impl_module_params!(PhiForCausalLM; model, lm_head);

impl PhiForCausalLM {
    /// Create a new Phi causal LM.
    pub fn new(config: PhiConfig) -> Result<Self, Exception> {
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        };
        let model = PhiModel::new(config)?;
        Ok(Self { model, lm_head })
    }

    /// Project trunk hidden states to logits, through the tied embedding when
    /// the checkpoint carries no separate head.
    ///
    /// Public because the DFlash decoder projects its own draft hidden states
    /// and would otherwise repeat the tied-head branch — the omission this
    /// method exists to prevent.
    pub fn project_logits(&mut self, hidden: &Array) -> Result<Array, Exception> {
        match self.lm_head.as_mut() {
            Some(head) => Ok(Module::forward(head, hidden)?),
            None => Ok(self.model.embed_tokens.as_linear(hidden)),
        }
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
        self.project_logits(&hidden)
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
        let hidden = self
            .model
            .forward_with_capture(input_ids, mask, cache, Some(capture))?;
        self.project_logits(&hidden)
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

    /// Fused batched-decode forward.
    ///
    /// Runs one `[N_active, 1]` forward across every active slot against a
    /// shared [`pmetal_mlx::kv_cache::FusedBatchKVCache`]. Phi uses partial
    /// RoPE (`partial_rotary_factor < 1.0`); routed through
    /// [`crate::common::BatchedGqaAttnCfg::with_rope_dims`].
    ///
    /// SuRoPE configs (`rope_scaling = Some(...)`) take the serial fallback —
    /// the fused config carries a single `rope_base`, not a per-dim freq
    /// array. Gated by [`crate::dispatcher::DynamicModel::supports_fused_batched`].
    pub fn forward_batched_impl(
        &mut self,
        input_ids: &Array,
        active_indices: &[usize],
        cache: &mut pmetal_mlx::kv_cache::FusedBatchKVCache,
    ) -> Result<Array, Exception> {
        use crate::common::{BatchedGqaAttnCfg, batched_prenorm_layer};
        use pmetal_bridge::compat::Module;

        let cfg = &self.model.config;
        let attn_cfg = BatchedGqaAttnCfg::new(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            cfg.rope_theta,
            1.0,
        )
        .with_rope_dims(cfg.rope_dim());

        let mut hidden = Module::forward(&mut self.model.embed_tokens, input_ids)?;
        for (layer_idx, layer) in self.model.layers.iter_mut().enumerate() {
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
        let hidden = self.model.norm.forward(&hidden)?;
        self.project_logits(&hidden)
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

    /// Phi-3 LongRoPE parity against the authoritative HuggingFace
    /// `transformers` oracle (`ROPE_INIT_FUNCTIONS["longrope"]` +
    /// `models.phi3.apply_rotary_pos_emb`).
    ///
    /// Guards three fixes: (1) the per-dimension `long_factor`-scaled inverse
    /// frequencies are actually applied (plain base RoPE used to be used,
    /// ignoring the tables); (2) the mscale is applied once, as a *value*
    /// scale, not also as a position scale; (3) `compute_longrope_freqs`
    /// produces `inv_freq = 1/(long_factor · base^(2i/d))`.
    ///
    /// The fixture forces the long branch (`seq_len = orig_max + 1`), so this
    /// reads the long table explicitly rather than through `table_for`.
    ///
    /// transformers folds the mscale into `cos`/`sin` where pmetal value-scales
    /// the input; rotation is linear, so the two are algebraically identical
    /// and this fixture is what proves it stays that way.
    ///
    /// Fixture: `.strategy/parity/dump_phi_surope_reference.py`.
    #[test]
    #[serial]
    fn phi_longrope_matches_transformers_oracle() {
        use pmetal_mlx::kernels::rope::apply_rope_with_freqs;

        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/fixtures/phi_surope_reference.safetensors");
        let shard: std::collections::HashMap<String, Array> =
            pmetal_bridge::inline_array::load_safetensors_shard(path.to_str().unwrap())
                .expect("load surope fixture")
                .into_iter()
                .collect();
        let x = shard.get("x").expect("x").clone();
        let y_ref = shard.get("y").expect("y").clone();
        let mut long_factor_arr = shard.get("long_factor").expect("long_factor").clone();
        let long_factor = long_factor_arr.to_f32_vec(16).expect("long_factor vec");

        // Match the dumper's SuScaledRoPE params.
        let dims = 32;
        let base = 10000.0_f32;
        let max_pos = 512;
        let orig_max = 128;
        let scaling = PhiRopeScaling {
            scaling_type: "longrope".to_string(),
            short_factor: vec![1.0; 16],
            long_factor,
        };
        let long_rope =
            compute_longrope_freqs(&scaling, dims, base, max_pos, orig_max).expect("su freqs");
        assert!(
            (long_rope.mscale - 1.133_893).abs() < 1e-4,
            "mscale {} != mlx 1.133893",
            long_rope.mscale
        );

        // SuScaledRoPE value-scales x[..., :dims] (here dims == head_dim) then
        // rotates with the long-factor freqs at offset 0.
        let x_scaled = x.multiply(&Array::from_f32(long_rope.mscale));
        let mut y_rust = apply_rope_with_freqs(&x_scaled, &long_rope.long, dims, false, 0)
            .expect("rope with freqs");
        y_rust.eval().unwrap();

        let got = y_rust.to_f32_vec(384).expect("rust vec");
        let mut y_ref_eval = y_ref;
        y_ref_eval.eval().unwrap();
        let want = y_ref_eval.to_f32_vec(384).expect("ref vec");
        let max_abs = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 1e-4,
            "SuRoPE output diverges from mlx oracle by {max_abs}"
        );
    }

    /// LongRoPE picks its table by sequence length, not unconditionally.
    ///
    /// pmetal used to precompute only `long_factor` and use it for everything.
    /// That is wrong for every sequence shorter than the pretraining length,
    /// which is most of them: Phi-4-mini's `original_max_position_embeddings`
    /// is 4096, so an ordinary prompt should rotate with `short_factor`. It
    /// showed up as a real-weight divergence at position 22 of a 640-token run.
    #[test]
    fn longrope_switches_tables_at_the_pretraining_length() {
        let scaling = PhiRopeScaling {
            scaling_type: "longrope".to_string(),
            short_factor: vec![1.0; 8],
            long_factor: (0..8).map(|i| 1.0 + 0.5 * i as f32).collect(),
        };
        let long_rope = compute_longrope_freqs(&scaling, 16, 10000.0, 512, 128).unwrap();

        // A `short_factor` of all ones leaves the base frequencies alone, so
        // the two tables really are different and the choice is observable.
        let short = long_rope.short.clone().to_f32_vec(8).unwrap();
        let long = long_rope.long.clone().to_f32_vec(8).unwrap();
        assert!((short[4] - 10000.0_f32.powf(-8.0 / 16.0)).abs() < 1e-6);
        assert!((long[4] / short[4] - 1.0 / 3.0).abs() < 1e-5);

        // The boundary is on the effective length (`max_position + 1`), the
        // same quantity transformers derives as `max(position_ids) + 1`.
        let id_of = |arr: &Array| arr.clone().to_f32_vec(8).unwrap();
        assert_eq!(id_of(long_rope.table_for(0)), short, "single token");
        assert_eq!(id_of(long_rope.table_for(126)), short, "127 tokens");
        assert_eq!(id_of(long_rope.table_for(127)), short, "exactly 128 tokens");
        assert_eq!(id_of(long_rope.table_for(128)), long, "129 tokens");
        assert_eq!(id_of(long_rope.table_for(639)), long, "a long run");
    }

    #[test]
    fn test_phi_config_presets() {
        let mini = PhiConfig::phi3_mini();
        assert_eq!(mini.hidden_size, 3072);
        assert_eq!(mini.num_hidden_layers, 32);
        assert_eq!(mini.head_dim(), 96);
        // Full rotary. `microsoft/Phi-3-mini-4k-instruct` ships
        // `"partial_rotary_factor": null` and `transformers` reads that as
        // 1.0, so the whole 96-wide head rotates. This asserted 48 while the
        // preset claimed 0.5, which is what halved RoPE on every real Phi-3
        // checkpoint pmetal loaded.
        assert_eq!(mini.rope_dim(), 96);

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
