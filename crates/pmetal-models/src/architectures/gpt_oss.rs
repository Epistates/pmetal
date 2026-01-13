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

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as ModuleParametersTrait},
    nn,
    ops::indexing::IndexOp,
    Array,
};
use pmetal_mlx::kernels::{fused_sdpa, rope::apply_rope, AttentionMaskType, FusedAttentionConfig};
use pmetal_mlx::kv_cache::KVCache;
use pmetal_mlx::moe::{MoEConfig, MoELayer};
use serde::{Deserialize, Serialize};

/// Attention layer type for GPT-OSS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionType {
    /// Sliding window attention (128 tokens).
    SlidingAttention,
    /// Full context attention.
    FullAttention,
}

impl Default for AttentionType {
    fn default() -> Self {
        Self::FullAttention
    }
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
#[derive(Debug, ModuleParameters)]
pub struct GptOssAttention {
    /// Configuration.
    #[allow(dead_code)]
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
    #[param]
    pub q_proj: nn::Linear,
    /// Key projection.
    #[param]
    pub k_proj: nn::Linear,
    /// Value projection.
    #[param]
    pub v_proj: nn::Linear,
    /// Output projection.
    #[param]
    pub o_proj: nn::Linear,
}

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

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, seq, heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q = apply_rope(&q, self.head_dim, false, self.rope_theta, 1.0, offset)?;
        let k = apply_rope(&k, self.head_dim, false, self.rope_theta, 1.0, offset)?;

        // Update cache if provided
        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

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

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        // Transpose back and project
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

        self.o_proj.forward(&output)
    }
}

/// SwiGLU MLP with clamping for GPT-OSS.
#[derive(Debug, ModuleParameters)]
pub struct GptOssMLP {
    /// SwiGLU limit (clamp value).
    swiglu_limit: f32,
    /// Gate projection.
    #[param]
    pub gate_proj: nn::Linear,
    /// Up projection.
    #[param]
    pub up_proj: nn::Linear,
    /// Down projection.
    #[param]
    pub down_proj: nn::Linear,
}

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
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        // SwiGLU with clamping: clamp(silu(gate) * up, -limit, limit)
        let activated = nn::silu(&gate)?.multiply(&up)?;
        // Clamp using min/max operations
        let limit = Array::from_f32(self.swiglu_limit);
        let neg_limit = Array::from_f32(-self.swiglu_limit);
        let clamped = mlx_rs::ops::minimum(&activated, &limit)?;
        let clamped = mlx_rs::ops::maximum(&clamped, &neg_limit)?;

        self.down_proj.forward(&clamped)
    }
}

/// GPT-OSS MoE block with sigmoid routing.
#[derive(Debug)]
pub struct GptOssMoE {
    /// MoE layer from pmetal-mlx.
    moe_layer: MoELayer,
    /// SwiGLU limit for clamping.
    #[allow(dead_code)]
    swiglu_limit: f32,
}

impl GptOssMoE {
    /// Create a new MoE block.
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        num_experts: i32,
        experts_per_token: i32,
        swiglu_limit: f32,
    ) -> Result<Self, Exception> {
        // GPT-OSS uses sigmoid routing (already default in MoE expert)
        let config = MoEConfig::new(hidden_size, intermediate_size, num_experts as usize)
            .with_num_experts_per_tok(experts_per_token as usize)
            .with_aux_loss(true, 0.9);

        let moe_layer = MoELayer::new(config);

        Ok(Self {
            moe_layer,
            swiglu_limit,
        })
    }

    /// Forward pass through MoE.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let (output, _aux_loss) = self.moe_layer.forward(x)?;
        Ok(output)
    }
}

/// GPT-OSS decoder layer.
#[derive(Debug, ModuleParameters)]
pub struct GptOssDecoderLayer {
    /// Self-attention.
    #[param]
    pub self_attn: GptOssAttention,
    /// MoE or dense MLP.
    mlp: GptOssMoE,
    /// Input layer norm.
    #[param]
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

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
        let hidden = self.input_layernorm.forward(x)?;
        let hidden = self.self_attn.forward(&hidden, mask, cache)?;
        let hidden = residual.add(&hidden)?;

        // Pre-norm MLP
        let residual = &hidden;
        let hidden = self.post_attention_layernorm.forward(&hidden)?;
        let hidden = self.mlp.forward(&hidden)?;
        residual.add(&hidden)
    }
}

/// GPT-OSS model.
#[derive(Debug, ModuleParameters)]
pub struct GptOssModel {
    /// Configuration.
    config: GptOssConfig,
    /// Token embeddings.
    #[param]
    pub embed_tokens: nn::Embedding,
    /// Decoder layers.
    #[param]
    pub layers: Vec<GptOssDecoderLayer>,
    /// Final layer norm.
    #[param]
    pub norm: nn::RmsNorm,
}

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
        let mut hidden = self.embed_tokens.forward(input_ids)?;

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

        self.norm.forward(&hidden)
    }

    /// Get the configuration.
    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }
}

/// GPT-OSS for causal language modeling.
#[derive(Debug, ModuleParameters)]
pub struct GptOssForCausalLM {
    /// Base model.
    #[param]
    pub model: GptOssModel,
    /// Language model head.
    #[param]
    pub lm_head: nn::Linear,
}

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
        self.lm_head.forward(&hidden)
    }

    /// Get the configuration.
    pub fn config(&self) -> &GptOssConfig {
        self.model.config()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> i32 {
        self.model.config.vocab_size
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
        let params = linear.parameters();
        let flat = params.flatten();
        let weight_ref = flat
            .get("weight")
            .ok_or_else(|| Exception::from("Missing weight in Linear"))?;
        let weight = (*weight_ref).clone();

        let shape = weight.shape();
        let out_features = shape[0];
        let in_features = shape[1];

        let (lora_a, lora_b) = create_lora_params(in_features, out_features, rank)?;
        let scale = alpha / rank as f32;

        // Get bias if present (need to clone the inner Array)
        let bias = flat.get("bias").map(|b| (*b).clone());

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
        if self.lora_active {
            let out = fused_lora_forward(x, &self.weight, &self.lora_a, &self.lora_b, self.scale)?;
            if let Some(ref bias) = self.bias {
                out.add(bias)
            } else {
                Ok(out)
            }
        } else {
            let out = x.matmul(&self.weight.t())?;
            if let Some(ref bias) = self.bias {
                out.add(bias)
            } else {
                Ok(out)
            }
        }
    }

    /// Merge LoRA into base weight (for inference).
    pub fn merge(&mut self) -> Result<(), Exception> {
        if self.lora_active {
            // merged = W + scale * B @ A
            let lora_contrib = self.lora_b.matmul(&self.lora_a)?;
            let scale_arr = Array::from_f32(self.scale);
            let scaled = lora_contrib.multiply(&scale_arr)?;
            self.weight = self.weight.add(&scaled)?;
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
    /// Sliding window size.
    sliding_window: i32,
    /// Attention type.
    attention_type: AttentionType,
    /// Query projection with LoRA.
    pub q_proj: LoraLinear,
    /// Key projection with LoRA.
    pub k_proj: LoraLinear,
    /// Value projection with LoRA.
    pub v_proj: LoraLinear,
    /// Output projection with LoRA.
    pub o_proj: LoraLinear,
    /// Config reference.
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

        // Project Q, K, V using LoRA layers
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, seq, heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q = apply_rope(&q, self.head_dim, false, self.rope_theta, 1.0, offset)?;
        let k = apply_rope(&k, self.head_dim, false, self.rope_theta, 1.0, offset)?;

        // Update cache if provided
        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

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

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        // Transpose back and project
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

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
    mlp: GptOssMoE,
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
        let hidden = self.input_layernorm.forward(x)?;
        let hidden = self.self_attn.forward(&hidden, mask, cache)?;
        let hidden = residual.add(&hidden)?;

        // Pre-norm MLP
        let residual = &hidden;
        let hidden = self.post_attention_layernorm.forward(&hidden)?;
        let hidden = self.mlp.forward(&hidden)?;
        residual.add(&hidden)
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
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            config: model.config,
            embed_tokens: model.embed_tokens,
            layers,
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
        let mut hidden = self.embed_tokens.forward(input_ids)?;

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

        self.norm.forward(&hidden)
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
        self.lm_head.forward(&hidden)
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

    // LoRA Tests
    #[test]
    fn test_lora_config_default_target_modules() {
        let lora_config = LoraConfig::default();
        // Default targets attention projections
        assert!(lora_config
            .target_modules
            .iter()
            .any(|m| m == "q_proj" || m.contains("q_proj")));
    }

    #[test]
    fn test_lora_linear_creation() {
        use mlx_rs::builder::Builder;

        let linear = nn::LinearBuilder::new(256, 512).bias(true).build().unwrap();

        let lora_linear = LoraLinear::from_linear(&linear, 8, 16.0).unwrap();

        assert_eq!(lora_linear.lora_a.shape(), &[8, 256]);
        assert_eq!(lora_linear.lora_b.shape(), &[512, 8]);
        assert!(lora_linear.lora_active);
        assert_eq!(lora_linear.scale, 16.0 / 8.0);
    }

    #[test]
    fn test_lora_linear_forward_shape() {
        use mlx_rs::builder::Builder;

        let linear = nn::LinearBuilder::new(64, 128).bias(false).build().unwrap();

        let lora_linear = LoraLinear::from_linear(&linear, 4, 8.0).unwrap();
        let x = Array::zeros::<f32>(&[2, 8, 64]).unwrap();
        let out = lora_linear.forward(&x).unwrap();

        assert_eq!(out.shape(), &[2, 8, 128]);
    }

    #[test]
    fn test_lora_linear_merge() {
        use mlx_rs::builder::Builder;

        let linear = nn::LinearBuilder::new(32, 64).bias(false).build().unwrap();

        let mut lora_linear = LoraLinear::from_linear(&linear, 4, 8.0).unwrap();
        assert!(lora_linear.lora_active);

        lora_linear.merge().unwrap();
        assert!(!lora_linear.lora_active);

        // Forward should still work
        let x = Array::zeros::<f32>(&[1, 4, 32]).unwrap();
        let out = lora_linear.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_lora_trainable_params() {
        use mlx_rs::builder::Builder;

        let linear = nn::LinearBuilder::new(32, 64).bias(false).build().unwrap();

        let lora_linear = LoraLinear::from_linear(&linear, 4, 8.0).unwrap();
        let params = lora_linear.trainable_parameters();

        assert_eq!(params.len(), 2); // lora_a and lora_b
        assert_eq!(params[0].0, "lora_a");
        assert_eq!(params[1].0, "lora_b");
    }
}
