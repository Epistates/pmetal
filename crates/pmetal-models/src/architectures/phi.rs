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
use pmetal_bridge::compat::nn::{Embedding, Linear, RmsNorm};
use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParameters, ModuleParametersExt, Param, fast, nn, ops, random,
};
use pmetal_bridge::impl_module_params;

use pmetal_bridge::rope::{RopeConfig, RotaryEmbedding};
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopePositions, rope_embedding},
};
use pmetal_mlx::kv_cache::KVCache;

use crate::architectures::utils::{Activation, resolve_activation};
use crate::checkpointing::checkpointed_layer;
use crate::traits::{CausalLMModel, ModelConfig};
use std::collections::HashMap;

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
    /// Original max position embeddings (for RoPE scaling). As in
    /// transformers' `Phi3Config`, it wins over the one inside `rope_scaling`.
    pub original_max_position_embeddings: Option<i32>,
    /// RoPE scaling (`longrope`, legacy `su`, for the 128K releases), read
    /// by [`pmetal_bridge::rope`].
    pub rope_scaling: Option<serde_json::Value>,
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

    /// The partial rotary embedding `rope_scaling` describes (LongRoPE for
    /// the 128K releases); an unknown `rope_type` is an error naming it.
    pub fn rotary(&self) -> Result<RotaryEmbedding, Exception> {
        crate::common::rotary_embedding(
            &self.model_type,
            self.head_dim(),
            RopeConfig {
                rope_scaling: self.rope_scaling.as_ref(),
                rope_theta: Some(self.rope_theta as f64),
                partial_rotary_factor: Some(self.partial_rotary_factor as f64),
                max_position_embeddings: Some(self.max_position_embeddings as f64),
                original_max_position_embeddings: self
                    .original_max_position_embeddings
                    .map(f64::from),
                ..RopeConfig::default()
            },
            1.0,
            false,
        )
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
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    /// The partial rotary embedding, LongRoPE included (Phi-3 128K, Phi-3.5,
    /// Phi-4-mini): its short or long table by how far a forward reaches,
    /// and its attention factor on the rotated channels only.
    pub rotary: RotaryEmbedding,
}
impl_module_params!(PhiAttention; q_proj, k_proj, v_proj, o_proj);

impl PhiAttention {
    /// Create a new Phi attention layer.
    pub fn new(config: &PhiConfig) -> Result<Self, Exception> {
        let head_dim = config.head_dim();
        let rotary = config.rotary()?;

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

        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            head_dim,
            scale,
            rotary,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        self.forward_with_cache(x, mask, None, None)
    }

    /// Forward pass with optional KV cache.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
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

        // Partial RoPE over the first `rotary.dims()` channels. LongRoPE
        // picks its short or long table by how far this forward reaches
        // (transformers' `max(position_ids) + 1`), and its attention factor
        // scales the rotated channels only, as `cos * attention_scaling` does.
        let offset = cache
            .as_ref()
            .map_or(0, |(c, layer)| c.rope_offset_for(*layer));
        let rope_positions = RopePositions::resolve(positions, offset);
        let q = rope_embedding(&q, rope_positions, &self.rotary);
        let k_transposed = rope_embedding(&k, rope_positions, &self.rotary);

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
        self.forward_with_cache(x, mask, None, None)
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
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        std_pre_norm_forward(
            &mut self.input_layernorm,
            &mut self.self_attn,
            &mut self.post_attention_layernorm,
            &mut self.mlp,
            x,
            mask,
            cache,
            positions,
        )
    }
}

impl AttentionModule for PhiAttention {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        PhiAttention::forward_with_cache(self, x, mask, cache, positions)
    }
}

impl DecoderLayer for PhiDecoderLayer {
    fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        PhiDecoderLayer::forward_with_cache(self, x, mask, cache, positions)
    }
}

/// Phi base model.
#[derive(Debug)]
pub struct PhiModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<PhiDecoderLayer>,
    pub norm: PhiRMSNorm,
    pub config: PhiConfig,
    /// Recompute each layer's activations during the backward pass instead of
    /// holding them. Training only; see [`crate::checkpointing`].
    pub grad_checkpoint: bool,
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
            grad_checkpoint: false,
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
        self.forward_with_capture(input_ids, mask, cache, None, None)
    }

    /// Forward pass with one rotary position per token, `[seq_len]`.
    ///
    /// Packed training concatenates several sequences into one row, so the
    /// positions have to restart at each boundary instead of running through
    /// it. The block-diagonal mask keeps the content apart; this keeps the
    /// positions apart.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_capture(input_ids, mask, None, positions, None)
    }

    /// Forward pass with optional hidden-state capture for DFlash
    /// speculative decoding. Identical to [`forward_with_cache`] when
    /// `capture` is `None`.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut KVCache>,
        positions: Option<&Array>,
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

        // Hoisted: the loop below borrows `self.layers` mutably.
        let grad_checkpoint = self.grad_checkpoint;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let c = cache.as_deref_mut().map(|c| (c, idx));
            // A cache means generation, which has no backward pass for the
            // recompute to pay for.
            hidden = if grad_checkpoint && c.is_none() {
                checkpointed_layer(
                    layer,
                    &hidden,
                    mask,
                    positions,
                    |layer, h, mask, positions| layer.forward_with_cache(h, mask, None, positions),
                )?
            } else {
                layer.forward_with_cache(&hidden, mask, c, positions)?
            };
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

    /// Forward pass with one rotary position per token; see
    /// `forward_with_positions` on the inner model.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden = self
            .model
            .forward_with_positions(input_ids, mask, positions)?;
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
        let hidden =
            self.model
                .forward_with_capture(input_ids, mask, cache, None, Some(capture))?;
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

    /// Whether RoPE is a scalar base and position scale, which the fused
    /// batched decode path carries; LongRoPE is not.
    pub fn has_scalar_rope(&self) -> bool {
        self.model
            .layers
            .first()
            .is_some_and(|layer| layer.self_attn.rotary.scalar().is_some())
    }

    /// Fused batched-decode forward.
    ///
    /// Runs one `[N_active, 1]` forward across every active slot against a
    /// shared [`pmetal_mlx::kv_cache::FusedBatchKVCache`]. Phi uses partial
    /// RoPE (`partial_rotary_factor < 1.0`); routed through
    /// [`crate::common::BatchedGqaAttnCfg::with_rope_dims`].
    ///
    /// LongRoPE configs take the serial fallback: the fused config carries a
    /// single `rope_base`, not a per-dim freq array. Gated by
    /// [`crate::dispatcher::DynamicModel::supports_fused_batched`].
    pub fn forward_batched_impl(
        &mut self,
        input_ids: &Array,
        active_indices: &[usize],
        cache: &mut pmetal_mlx::kv_cache::FusedBatchKVCache,
    ) -> Result<Array, Exception> {
        use crate::common::{BatchedGqaAttnCfg, batched_prenorm_layer};
        use pmetal_bridge::compat::Module;

        let cfg = &self.model.config;
        let attn_cfg = BatchedGqaAttnCfg::for_rotary(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            &self.model.layers[0].self_attn.rotary,
        )?;

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

    /// A LongRoPE Phi config: `head_dim` 32, full rotary, pretraining length
    /// 128 stretched to 512, as the transformers dumper sets it up.
    fn longrope_config(short_factor: Vec<f32>, long_factor: Vec<f32>) -> PhiConfig {
        PhiConfig {
            hidden_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            max_position_embeddings: 512,
            original_max_position_embeddings: Some(128),
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            rope_scaling: Some(serde_json::json!({
                "type": "longrope", "short_factor": short_factor, "long_factor": long_factor
            })),
            ..PhiConfig::default()
        }
    }

    /// Phi-3 LongRoPE through `PhiConfig::rotary` against the authoritative
    /// HuggingFace `transformers` oracle (`ROPE_INIT_FUNCTIONS["longrope"]` +
    /// `models.phi3.apply_rotary_pos_emb`): the per-dimension
    /// `long_factor`-scaled frequencies, and the attention factor
    /// `sqrt(1 + ln(512/128) / ln(128))` on the rotated channels, once.
    ///
    /// The fixture rotates positions 0..6 with the long table (the dumper
    /// asks for `seq_len = 129`), so the config here carries the fixture's
    /// `long_factor` as both tables; which table a reach picks is pinned
    /// below and in `pmetal_bridge::rope`.
    ///
    /// Fixture: `.strategy/parity/dump_phi_surope_reference.py`.
    #[test]
    #[serial]
    fn phi_longrope_matches_transformers_oracle() {
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

        let rotary = longrope_config(long_factor.clone(), long_factor)
            .rotary()
            .expect("longrope parses");
        assert!(
            (rotary.attention_factor() - 1.133_893).abs() < 1e-5,
            "attention factor {}",
            rotary.attention_factor()
        );
        let mut y_rust = rotary.apply(&x, 0);
        y_rust.eval().unwrap();
        pmetal_bridge::check_last_error().expect("bridge");

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
            max_abs < 1e-5,
            "LongRoPE output diverges from transformers by {max_abs}"
        );
    }

    /// LongRoPE picks its table by how far a forward reaches, not
    /// unconditionally: Phi-4-mini's `original_max_position_embeddings` is
    /// 4096, so an ordinary prompt rotates with `short_factor`. pmetal once
    /// used `long_factor` for everything, a real-weight divergence at position
    /// 22 of a 640-token run. And a scaling Phi cannot run is refused by name.
    #[test]
    fn longrope_switches_tables_at_the_pretraining_length() {
        let long: Vec<f32> = (0..16).map(|i| 1.0 + 0.5 * i as f32).collect();
        let rotary = longrope_config(vec![1.0; 16], long).rotary().unwrap();
        let short = rotary.inverse_frequencies(0);
        assert!((short[4] - 10000.0_f32.powf(-8.0 / 32.0)).abs() < 1e-6);
        assert_eq!(rotary.inverse_frequencies(128), short, "exactly 128 tokens");
        let long = rotary.inverse_frequencies(129);
        assert!((long[4] / short[4] - 1.0 / 3.0).abs() < 1e-5, "129 tokens");

        let mut config = longrope_config(vec![1.0; 16], vec![1.0; 16]);
        config.rope_scaling = Some(serde_json::json!({"type": "xpos", "factor": 2.0}));
        let err = config.rotary().unwrap_err().to_string();
        assert!(err.contains("xpos"), "{err}");
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
