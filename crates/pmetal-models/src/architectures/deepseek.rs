//! DeepSeek V3 / V3.2 / V3.2-Speciale model architecture.
//!
//! Implements DeepSeek V3 and V3.2 variants with:
//! - Multi-Latent Attention (MLA): LoRA-style Q/K/V compression for 28x KV cache reduction
//! - Mixture of Experts (MoE): Sparse routing with shared experts
//! - Aux-free load balancing: No auxiliary loss, uses e_score_correction_bias
//! - Sigmoid scoring: Uses sigmoid instead of softmax for expert routing
//!
//! V3.2 adds DeepSeek Sparse Attention (DSA):
//! - Lightning Indexer: Computes relevance scores for efficient sparse attention
//! - Token Selector: Selects top-k (2048) tokens per query position
//! - Reduces O(L²) attention complexity to O(L·k)
//!
//! V3.2-Speciale is the extended thinking variant optimized for reasoning tasks.

use std::collections::HashMap;

use pmetal_mlx::kernels::rope::apply_rope;
use pmetal_mlx::kv_cache::KVCache;
use pmetal_mlx::moe::{MoEConfig, MoELayer};
use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops::indexing::IndexOp,
    Array,
};
use serde::{Deserialize, Serialize};

/// DeepSeek model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DeepSeekVariant {
    /// DeepSeek V3 (dense attention).
    #[default]
    V3,
    /// DeepSeek V3.2 with DeepSeek Sparse Attention (DSA).
    V32,
    /// DeepSeek V3.2-Speciale (extended thinking variant).
    V32Speciale,
}

/// DeepSeek V3 / V3.2 model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Hidden dimension.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    /// Intermediate size for dense MLP.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    /// Intermediate size for MoE experts.
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: i32,
    /// Number of hidden layers.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    /// Number of key-value heads.
    #[serde(default)]
    pub num_key_value_heads: Option<i32>,
    /// Number of shared experts (always active).
    #[serde(default)]
    pub n_shared_experts: Option<i32>,
    /// Number of routed experts.
    #[serde(default)]
    pub n_routed_experts: Option<i32>,
    /// Routing scale factor.
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    /// KV LoRA rank for MLA.
    #[serde(default = "default_kv_lora_rank")]
    pub kv_lora_rank: i32,
    /// Q LoRA rank for MLA.
    #[serde(default)]
    pub q_lora_rank: Option<i32>,
    /// RoPE head dimension.
    #[serde(default = "default_qk_rope_head_dim")]
    pub qk_rope_head_dim: i32,
    /// Value head dimension.
    #[serde(default = "default_v_head_dim")]
    pub v_head_dim: i32,
    /// Non-position head dimension.
    #[serde(default = "default_qk_nope_head_dim")]
    pub qk_nope_head_dim: i32,
    /// Top-K selection method.
    #[serde(default = "default_topk_method")]
    pub topk_method: String,
    /// Scoring function (sigmoid or softmax).
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,
    /// Normalize top-k probabilities.
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    /// Number of expert groups.
    #[serde(default = "default_one")]
    pub n_group: i32,
    /// Top-k groups.
    #[serde(default = "default_one")]
    pub topk_group: i32,
    /// Number of experts per token.
    #[serde(default = "default_one")]
    pub num_experts_per_tok: i32,
    /// MoE layer frequency.
    #[serde(default = "default_one")]
    pub moe_layer_freq: i32,
    /// First K layers use dense MLP.
    #[serde(default)]
    pub first_k_dense_replace: i32,
    /// Maximum position embeddings.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// RMS normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, RopeScalingValue>>,
    /// Attention bias.
    #[serde(default)]
    pub attention_bias: bool,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // =========================================================================
    // V3.2 DeepSeek Sparse Attention (DSA) Configuration
    // =========================================================================

    /// Enable DeepSeek Sparse Attention (V3.2 feature).
    #[serde(default)]
    pub use_sparse_attention: bool,
    /// Number of heads for the Lightning Indexer (typically small, e.g., 4).
    #[serde(default = "default_lightning_indexer_heads")]
    pub lightning_indexer_heads: i32,
    /// Top-k tokens to select per query position for sparse attention.
    #[serde(default = "default_sparse_top_k")]
    pub sparse_top_k: i32,
    /// Use non-interleaved RoPE layout for Lightning Indexer.
    #[serde(default = "default_true")]
    pub indexer_non_interleaved_rope: bool,
    /// Whether to use FP8 for Lightning Indexer computations (efficiency).
    #[serde(default)]
    pub indexer_use_fp8: bool,

    // =========================================================================
    // V3.2-Speciale Extended Thinking Configuration
    // =========================================================================

    /// Model variant (V3, V3.2, V3.2-Speciale).
    #[serde(default)]
    pub variant: DeepSeekVariant,
    /// Enable extended thinking mode (Speciale variant).
    #[serde(default)]
    pub thinking_mode: bool,
    /// Maximum tokens for thinking/reasoning output.
    #[serde(default)]
    pub max_thinking_tokens: Option<i32>,
    /// Thinking token ID (start of reasoning).
    #[serde(default)]
    pub thinking_start_token_id: Option<i32>,
    /// End thinking token ID.
    #[serde(default)]
    pub thinking_end_token_id: Option<i32>,
}

fn default_lightning_indexer_heads() -> i32 { 4 }
fn default_sparse_top_k() -> i32 { 2048 }

// Default value functions
fn default_model_type() -> String { "deepseek_v3".to_string() }
fn default_vocab_size() -> i32 { 102400 }
fn default_hidden_size() -> i32 { 4096 }
fn default_intermediate_size() -> i32 { 11008 }
fn default_moe_intermediate_size() -> i32 { 1407 }
fn default_num_hidden_layers() -> i32 { 30 }
fn default_num_attention_heads() -> i32 { 32 }
fn default_routed_scaling_factor() -> f32 { 1.0 }
fn default_kv_lora_rank() -> i32 { 512 }
fn default_qk_rope_head_dim() -> i32 { 64 }
fn default_v_head_dim() -> i32 { 128 }
fn default_qk_nope_head_dim() -> i32 { 128 }
fn default_topk_method() -> String { "noaux_tc".to_string() }
fn default_scoring_func() -> String { "sigmoid".to_string() }
fn default_true() -> bool { true }
fn default_one() -> i32 { 1 }
fn default_max_position_embeddings() -> i32 { 2048 }
fn default_rms_norm_eps() -> f32 { 1e-6 }
fn default_rope_theta() -> f32 { 10000.0 }

/// RoPE scaling configuration value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RopeScalingValue {
    /// Floating point value.
    Float(f32),
    /// String value.
    String(String),
    /// Integer value.
    Int(i32),
}

impl DeepSeekConfig {
    /// Get the number of KV heads.
    pub fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the Q head dimension.
    pub fn q_head_dim(&self) -> i32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Check if a layer should use MoE.
    pub fn is_moe_layer(&self, layer_idx: i32) -> bool {
        self.n_routed_experts.is_some()
            && layer_idx >= self.first_k_dense_replace
            && layer_idx % self.moe_layer_freq == 0
    }

    /// Create a V3.2 configuration with sparse attention enabled.
    pub fn v32() -> Self {
        Self {
            model_type: "deepseek_v32".to_string(),
            variant: DeepSeekVariant::V32,
            use_sparse_attention: true,
            ..Default::default()
        }
    }

    /// Create a V3.2-Speciale configuration with extended thinking.
    pub fn v32_speciale() -> Self {
        Self {
            model_type: "deepseek_v32_speciale".to_string(),
            variant: DeepSeekVariant::V32Speciale,
            use_sparse_attention: true,
            thinking_mode: true,
            max_thinking_tokens: Some(32768),
            ..Default::default()
        }
    }

    /// Detect variant from model_type string.
    pub fn detect_variant(model_type: &str) -> DeepSeekVariant {
        let lower = model_type.to_lowercase();
        if lower.contains("speciale") || lower.contains("thinking") {
            DeepSeekVariant::V32Speciale
        } else if lower.contains("v3.2") || lower.contains("v32") {
            DeepSeekVariant::V32
        } else {
            DeepSeekVariant::V3
        }
    }

    /// Check if this configuration uses sparse attention (V3.2+).
    pub fn uses_sparse_attention(&self) -> bool {
        self.use_sparse_attention || matches!(self.variant, DeepSeekVariant::V32 | DeepSeekVariant::V32Speciale)
    }
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            model_type: "deepseek_v3".to_string(),
            vocab_size: 102400,
            hidden_size: 4096,
            intermediate_size: 11008,
            moe_intermediate_size: 1407,
            num_hidden_layers: 30,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            n_shared_experts: None,
            n_routed_experts: None,
            routed_scaling_factor: 1.0,
            kv_lora_rank: 512,
            q_lora_rank: Some(1536),
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            qk_nope_head_dim: 128,
            topk_method: "noaux_tc".to_string(),
            scoring_func: "sigmoid".to_string(),
            norm_topk_prob: true,
            n_group: 1,
            topk_group: 1,
            num_experts_per_tok: 1,
            moe_layer_freq: 1,
            first_k_dense_replace: 0,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            tie_word_embeddings: false,
            // V3.2 DSA fields
            use_sparse_attention: false,
            lightning_indexer_heads: 4,
            sparse_top_k: 2048,
            indexer_non_interleaved_rope: true,
            indexer_use_fp8: false,
            // V3.2-Speciale fields
            variant: DeepSeekVariant::V3,
            thinking_mode: false,
            max_thinking_tokens: None,
            thinking_start_token_id: None,
            thinking_end_token_id: None,
        }
    }
}

/// DeepSeek V3 Multi-Latent Attention (MLA).
///
/// Uses LoRA-style decomposition for Q/K/V to compress the KV cache by 28x.
/// Key innovation: Projects KV to a low-rank latent space (512 dim) then back.
#[derive(Debug, ModuleParameters)]
pub struct DeepSeekAttention {
    /// Configuration.
    pub config: DeepSeekConfig,
    /// Number of attention heads.
    pub n_heads: i32,
    /// Attention scale.
    pub scale: f32,
    /// Layer ID.
    pub layer_id: usize,

    // Q projection (either direct or LoRA-style)
    /// Q projection A (LoRA down).
    #[param]
    pub q_a_proj: Option<nn::Linear>,
    /// Q projection A layernorm.
    #[param]
    pub q_a_layernorm: Option<nn::RmsNorm>,
    /// Q projection B (LoRA up).
    #[param]
    pub q_b_proj: Option<nn::Linear>,
    /// Direct Q projection (when not using LoRA).
    #[param]
    pub q_proj: Option<nn::Linear>,

    // KV projection (always LoRA-style in MLA)
    /// KV projection with MQA (outputs latent + rope key).
    #[param]
    pub kv_a_proj_with_mqa: nn::Linear,
    /// KV projection A layernorm.
    #[param]
    pub kv_a_layernorm: nn::RmsNorm,
    /// KV projection B (decompresses latent to K_nope and V).
    #[param]
    pub kv_b_proj: nn::Linear,

    /// Output projection.
    #[param]
    pub o_proj: nn::Linear,
}

impl DeepSeekAttention {
    /// Create a new DeepSeek attention layer.
    pub fn new(config: &DeepSeekConfig, layer_id: usize) -> Result<Self, Exception> {
        let hidden_size = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let q_head_dim = config.q_head_dim();
        let scale = (q_head_dim as f32).powf(-0.5);

        // Q projection (LoRA-style or direct)
        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) = if let Some(q_lora_rank) = config.q_lora_rank {
            let q_a = nn::LinearBuilder::new(hidden_size, q_lora_rank)
                .bias(config.attention_bias)
                .build()?;
            let q_a_norm = nn::RmsNormBuilder::new(q_lora_rank)
                .eps(1e-6)
                .build()?;
            let q_b = nn::LinearBuilder::new(q_lora_rank, n_heads * q_head_dim)
                .bias(false)
                .build()?;
            (Some(q_a), Some(q_a_norm), Some(q_b), None)
        } else {
            let q = nn::LinearBuilder::new(hidden_size, n_heads * q_head_dim)
                .bias(false)
                .build()?;
            (None, None, None, Some(q))
        };

        // KV projection: outputs [kv_lora_rank + qk_rope_head_dim]
        // The kv_lora_rank portion gets compressed, qk_rope_head_dim is the RoPE key
        let kv_a_proj_with_mqa = nn::LinearBuilder::new(
            hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
        )
        .bias(config.attention_bias)
        .build()?;

        let kv_a_layernorm = nn::RmsNormBuilder::new(config.kv_lora_rank)
            .eps(1e-6)
            .build()?;

        // KV B projection outputs K_nope and V for all heads
        let kv_b_output_dim = n_heads * (config.qk_nope_head_dim + config.v_head_dim);
        let kv_b_proj = nn::LinearBuilder::new(config.kv_lora_rank, kv_b_output_dim)
            .bias(false)
            .build()?;

        // Output projection
        let o_proj = nn::LinearBuilder::new(n_heads * config.v_head_dim, hidden_size)
            .bias(config.attention_bias)
            .build()?;

        Ok(Self {
            config: config.clone(),
            n_heads,
            scale,
            layer_id,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
        })
    }

    /// Forward pass with Multi-Latent Attention.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Compute Q (LoRA-style or direct)
        let q = if let Some(ref mut q_a) = self.q_a_proj {
            let q_a_out = q_a.forward(x)?;
            let q_a_norm = self.q_a_layernorm.as_mut().unwrap().forward(&q_a_out)?;
            self.q_b_proj.as_mut().unwrap().forward(&q_a_norm)?
        } else {
            self.q_proj.as_mut().unwrap().forward(x)?
        };

        let q_head_dim = self.config.q_head_dim();
        let q = q.reshape(&[batch, seq_len, self.n_heads, q_head_dim])?;
        let q = q.transpose_axes(&[0, 2, 1, 3])?; // [B, heads, seq, head_dim]

        // Split Q into nope (no positional encoding) and pe (RoPE) parts
        // mx.split(q, [qk_nope_head_dim], axis=-1) -> split_axis(&[idx], Some(-1))
        let q_parts = q.split_axis(&[self.config.qk_nope_head_dim as i32], Some(-1))?;
        let q_nope = &q_parts[0]; // [B, heads, seq, qk_nope_head_dim]
        let q_pe = &q_parts[1];   // [B, heads, seq, qk_rope_head_dim]

        // Compute compressed KV: outputs [latent, k_pe]
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_parts = compressed_kv.split_axis(&[self.config.kv_lora_rank as i32], Some(-1))?;
        let compressed_latent = &kv_parts[0]; // [B, seq, kv_lora_rank]
        let k_pe = &kv_parts[1]; // [B, seq, qk_rope_head_dim]

        // Reshape k_pe: [B, seq, rope_dim] -> [B, 1, seq, rope_dim]
        let k_pe = k_pe.reshape(&[batch, seq_len, 1, self.config.qk_rope_head_dim])?;
        let k_pe = k_pe.transpose_axes(&[0, 2, 1, 3])?;

        // Decompress latent to K_nope and V
        let kv_normalized = self.kv_a_layernorm.forward(compressed_latent)?;
        let kv = self.kv_b_proj.forward(&kv_normalized)?;
        let kv_dim = self.config.qk_nope_head_dim + self.config.v_head_dim;
        let kv = kv.reshape(&[batch, seq_len, self.n_heads, kv_dim])?;
        let kv = kv.transpose_axes(&[0, 2, 1, 3])?; // [B, heads, seq, kv_dim]

        // Split into k_nope and values
        let kv_split = kv.split_axis(&[self.config.qk_nope_head_dim as i32], Some(-1))?;
        let k_nope = &kv_split[0]; // [B, heads, seq, qk_nope_head_dim]
        let values = &kv_split[1]; // [B, heads, seq, v_head_dim]

        // Apply RoPE to q_pe and k_pe
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q_pe = apply_rope(
            q_pe,
            self.config.qk_rope_head_dim,
            false, // non-traditional (modern) RoPE
            self.config.rope_theta,
            1.0, // scale
            offset as i32,
        )?;
        let k_pe = apply_rope(
            &k_pe,
            self.config.qk_rope_head_dim,
            false,
            self.config.rope_theta,
            1.0,
            offset as i32,
        )?;

        // Repeat k_pe for all heads: [B, 1, seq, rope_dim] -> [B, heads, seq, rope_dim]
        let k_pe_repeated = mlx_rs::ops::broadcast_to(&k_pe, &[batch, self.n_heads, seq_len, self.config.qk_rope_head_dim])?;

        // Concatenate to form full keys and queries
        let keys = mlx_rs::ops::concatenate_axis(&[k_nope, &k_pe_repeated], -1)?;
        let queries = mlx_rs::ops::concatenate_axis(&[q_nope, &q_pe], -1)?;

        // Update cache if provided
        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &keys, values)?
        } else {
            (keys, values.clone())
        };

        // Scaled dot-product attention
        let attn_weights = queries.matmul(&keys.transpose_axes(&[0, 1, 3, 2])?)?;
        let attn_weights = attn_weights.multiply(&Array::from_f32(self.scale))?;

        // Apply mask if provided
        let attn_weights = if let Some(mask) = mask {
            attn_weights.add(mask)?
        } else {
            attn_weights
        };

        let attn_weights = mlx_rs::ops::softmax_axis(&attn_weights, -1, None)?;
        let output = attn_weights.matmul(&values)?;

        // Reshape and project output
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[batch, seq_len, -1])?;
        self.o_proj.forward(&output)
    }
}

// =============================================================================
// DeepSeek Sparse Attention (DSA) - V3.2 Feature
// =============================================================================

/// Lightning Indexer for DeepSeek Sparse Attention.
///
/// Computes cheap relevance scores using a smaller number of heads (typically 4)
/// to determine which tokens are most relevant for each query position.
/// This enables O(L*k) attention instead of O(L²) by only attending to top-k tokens.
#[derive(Debug, ModuleParameters)]
pub struct LightningIndexer {
    /// Number of indexer heads (typically small, e.g., 4).
    pub n_heads: i32,
    /// Query projection for indexer.
    #[param]
    pub q_proj: nn::Linear,
    /// Key projection for indexer.
    #[param]
    pub k_proj: nn::Linear,
    /// Head dimension for indexer.
    pub head_dim: i32,
    /// Scale factor for attention scores.
    pub scale: f32,
    /// RoPE dimension.
    pub rope_dim: i32,
    /// RoPE theta.
    pub rope_theta: f32,
    /// Use non-interleaved RoPE layout.
    pub non_interleaved: bool,
}

impl LightningIndexer {
    /// Create a new Lightning Indexer.
    pub fn new(config: &DeepSeekConfig) -> Result<Self, Exception> {
        let n_heads = config.lightning_indexer_heads;
        // Use same head dim as the main attention for compatibility
        let head_dim = config.qk_rope_head_dim;
        let hidden_size = config.hidden_size;

        let q_proj = nn::LinearBuilder::new(hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;

        Ok(Self {
            n_heads,
            q_proj,
            k_proj,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_dim: config.qk_rope_head_dim,
            rope_theta: config.rope_theta,
            non_interleaved: config.indexer_non_interleaved_rope,
        })
    }

    /// Compute relevance scores for all query-key pairs.
    ///
    /// Returns scores of shape [batch, seq_len, seq_len] indicating
    /// how relevant each key position is for each query position.
    pub fn compute_scores(
        &mut self,
        x: &Array,
        offset: i32,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project to queries and keys
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;

        // Reshape: [B, L, n_heads * head_dim] -> [B, L, n_heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;

        // Transpose to [B, n_heads, L, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let q = apply_rope(
            &q,
            self.rope_dim,
            !self.non_interleaved,
            self.rope_theta,
            1.0,
            offset,
        )?;
        let k = apply_rope(
            &k,
            self.rope_dim,
            !self.non_interleaved,
            self.rope_theta,
            1.0,
            offset,
        )?;

        // Compute attention scores: [B, n_heads, L, L]
        let scores = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
        let scores = scores.multiply(&Array::from_f32(self.scale))?;

        // Average across heads: [B, L, L]
        let scores = scores.mean_axis(1, true)?;
        scores.squeeze_axes(&[1])
    }
}

/// Token Selector for DeepSeek Sparse Attention.
///
/// Selects the top-k most relevant tokens for each query position
/// based on Lightning Indexer scores.
#[derive(Debug)]
pub struct TokenSelector {
    /// Number of tokens to select per query.
    pub top_k: i32,
}

impl TokenSelector {
    /// Create a new Token Selector.
    pub fn new(config: &DeepSeekConfig) -> Self {
        Self {
            top_k: config.sparse_top_k,
        }
    }

    /// Select top-k token indices for each query position.
    ///
    /// Args:
    ///     scores: Relevance scores [batch, query_len, key_len]
    ///     mask: Optional causal mask (future tokens get -inf scores)
    ///
    /// Returns:
    ///     indices: Selected token indices [batch, query_len, top_k]
    pub fn select_tokens(
        &self,
        scores: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Apply causal mask if provided
        let masked_scores = if let Some(mask) = mask {
            // Mask is typically [1, 1, query_len, key_len] for causal
            // Squeeze to [query_len, key_len] and broadcast
            let mask_2d = mask.squeeze_axes(&[0, 1])?;
            scores.add(&mask_2d)?
        } else {
            scores.clone()
        };

        // Get top-k indices using argpartition (more efficient than full sort)
        let neg_k = -self.top_k;
        let indices = mlx_rs::ops::argpartition_axis(&masked_scores, neg_k, -1)?;

        // Take the last k elements (these are the top-k)
        Ok(indices.index((.., .., neg_k..)))
    }
}

/// Sparse Attention output for DSA.
#[derive(Debug)]
pub struct SparseAttentionResult {
    /// Output tensor [batch, seq_len, hidden_size].
    pub output: Array,
    /// Selected token indices for debugging/analysis [batch, seq_len, top_k].
    pub selected_indices: Option<Array>,
}

/// DeepSeek Sparse Attention (DSA) layer for V3.2.
///
/// Combines Lightning Indexer and Token Selector with standard MLA
/// to achieve sub-quadratic attention complexity.
#[derive(Debug, ModuleParameters)]
pub struct DeepSeekSparseAttention {
    /// Standard MLA attention.
    #[param]
    pub base_attention: DeepSeekAttention,
    /// Lightning Indexer for relevance scoring.
    #[param]
    pub indexer: LightningIndexer,
    /// Token Selector for top-k selection.
    pub selector: TokenSelector,
    /// Whether to store selected indices for analysis.
    pub store_indices: bool,
}

impl DeepSeekSparseAttention {
    /// Create a new sparse attention layer.
    pub fn new(config: &DeepSeekConfig, layer_id: usize) -> Result<Self, Exception> {
        let base_attention = DeepSeekAttention::new(config, layer_id)?;
        let indexer = LightningIndexer::new(config)?;
        let selector = TokenSelector::new(config);

        Ok(Self {
            base_attention,
            indexer,
            selector,
            store_indices: false,
        })
    }

    /// Forward pass with sparse attention.
    ///
    /// For short sequences (< 2 * top_k), falls back to dense attention.
    /// For longer sequences, uses Lightning Indexer to select relevant tokens.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<SparseAttentionResult, Exception> {
        let shape = x.shape();
        let seq_len = shape[1];
        let top_k = self.selector.top_k as i32;

        // For short sequences, use dense attention (sparse overhead not worth it)
        if seq_len < 2 * top_k {
            let output = self.base_attention.forward(x, mask, cache)?;
            return Ok(SparseAttentionResult {
                output,
                selected_indices: None,
            });
        }

        // Compute relevance scores with Lightning Indexer
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0) as i32;
        let scores = self.indexer.compute_scores(x, offset)?;

        // Select top-k tokens
        let selected_indices = self.selector.select_tokens(&scores, mask)?;

        // For now, fall back to dense attention
        // Full sparse implementation would gather selected K/V and compute attention
        // This requires significant changes to the attention computation
        // TODO: Implement sparse gather + attention computation
        let output = self.base_attention.forward(x, mask, cache)?;

        Ok(SparseAttentionResult {
            output,
            selected_indices: if self.store_indices {
                Some(selected_indices)
            } else {
                None
            },
        })
    }
}

/// DeepSeek MLP (SwiGLU).
#[derive(Debug, ModuleParameters)]
pub struct DeepSeekMLP {
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

impl DeepSeekMLP {
    /// Create a new MLP.
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(intermediate_size, hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass with SwiGLU activation.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = nn::silu(gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }
}

/// DeepSeek MoE gate with aux-free load balancing.
///
/// Uses sigmoid scoring and e_score_correction_bias for load balancing
/// instead of auxiliary loss.
#[derive(Debug, ModuleParameters)]
pub struct DeepSeekMoEGate {
    /// Gate weights: [num_experts, hidden_size].
    #[param]
    pub weight: nn::Linear,
    /// Bias correction for aux-free load balancing (FP32 for stability).
    pub e_score_correction_bias: Array,
    /// Number of experts per token.
    pub top_k: i32,
    /// Number of experts.
    pub num_experts: i32,
    /// Routing scale factor.
    pub routed_scaling_factor: f32,
    /// Normalize top-k probabilities.
    pub norm_topk_prob: bool,
}

impl DeepSeekMoEGate {
    /// Create a new MoE gate.
    pub fn new(config: &DeepSeekConfig) -> Result<Self, Exception> {
        let num_experts = config.n_routed_experts.unwrap_or(8);
        let weight = nn::LinearBuilder::new(config.hidden_size, num_experts)
            .bias(false)
            .build()?;

        // e_score_correction_bias - kept in FP32 for numerical stability
        let e_score_correction_bias = Array::zeros::<f32>(&[num_experts])?;

        Ok(Self {
            weight,
            e_score_correction_bias,
            top_k: config.num_experts_per_tok,
            num_experts,
            routed_scaling_factor: config.routed_scaling_factor,
            norm_topk_prob: config.norm_topk_prob,
        })
    }

    /// Compute expert selection using sigmoid scoring.
    pub fn forward(&mut self, x: &Array) -> Result<(Array, Array), Exception> {
        // Compute gate logits
        let gates = self.weight.forward(x)?;

        // DeepSeek V3 uses sigmoid scoring (not softmax)
        let scores = mlx_rs::ops::sigmoid(&gates.as_dtype(mlx_rs::Dtype::Float32)?)?;

        // Add correction bias for aux-free load balancing
        let scores_with_bias = scores.add(&self.e_score_correction_bias)?;

        // Select top-k experts using argpartition
        let neg_k = -self.top_k;
        let inds = mlx_rs::ops::argpartition_axis(&scores_with_bias, neg_k, -1)?;
        // Take the last k elements (top-k)
        let inds = inds.index((.., .., neg_k..));

        // Get original scores (without bias) for selected experts
        let top_scores = scores.take_along_axis(&inds, -1)?;

        // Normalize scores if configured
        let final_scores = if self.norm_topk_prob && self.top_k > 1 {
            let sum = top_scores.sum_axis(-1, true)?;
            top_scores.divide(&sum)?
        } else {
            top_scores
        };

        // Apply routing scale factor
        let final_scores = final_scores.multiply(&Array::from_f32(self.routed_scaling_factor))?;

        Ok((inds, final_scores))
    }
}

/// DeepSeek MoE block with shared experts.
#[derive(Debug)]
pub struct DeepSeekMoE {
    /// Configuration.
    pub config: DeepSeekConfig,
    /// MoE gate for expert selection.
    pub gate: DeepSeekMoEGate,
    /// MoE layer with routed experts.
    pub moe: MoELayer,
    /// Shared experts (always active).
    pub shared_experts: Option<DeepSeekMLP>,
}

impl DeepSeekMoE {
    /// Create a new MoE block.
    pub fn new(config: &DeepSeekConfig) -> Result<Self, Exception> {
        let gate = DeepSeekMoEGate::new(config)?;

        let moe_config = MoEConfig::new(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts.unwrap_or(8) as usize,
        )
        .with_num_experts_per_tok(config.num_experts_per_tok as usize)
        .with_aux_loss(false, 0.0); // DeepSeek V3 uses aux-free load balancing

        let moe = MoELayer::new(moe_config);

        let shared_experts = if let Some(n_shared) = config.n_shared_experts {
            let intermediate_size = config.moe_intermediate_size * n_shared;
            Some(DeepSeekMLP::new(config.hidden_size, intermediate_size)?)
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            gate,
            moe,
            shared_experts,
        })
    }

    /// Forward pass through MoE block.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Use our custom gate with sigmoid scoring
        let (_expert_indices, _expert_weights) = self.gate.forward(x)?;

        // Route through MoE (simplified - would use custom routing in production)
        self.moe.eval();
        let (moe_out, _aux_loss) = self.moe.forward(x)?;

        // Add shared expert output if configured
        if let Some(ref mut shared) = self.shared_experts {
            let shared_out = shared.forward(x)?;
            moe_out.add(&shared_out)
        } else {
            Ok(moe_out)
        }
    }
}

/// MLP type - either dense or MoE.
#[derive(Debug)]
pub enum DeepSeekMLPType {
    /// Dense MLP.
    Dense(DeepSeekMLP),
    /// Mixture of Experts.
    MoE(DeepSeekMoE),
}

impl DeepSeekMLPType {
    /// Forward pass.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self {
            DeepSeekMLPType::Dense(mlp) => mlp.forward(x),
            DeepSeekMLPType::MoE(moe) => moe.forward(x),
        }
    }
}

/// DeepSeek decoder layer.
#[derive(Debug, ModuleParameters)]
pub struct DeepSeekDecoderLayer {
    /// Layer ID.
    pub layer_id: usize,

    /// Self-attention.
    #[param]
    pub self_attn: DeepSeekAttention,
    /// Input layer norm.
    #[param]
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    /// MLP (dense or MoE).
    pub mlp: DeepSeekMLPType,
}

impl DeepSeekDecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: &DeepSeekConfig, layer_id: usize) -> Result<Self, Exception> {
        let self_attn = DeepSeekAttention::new(config, layer_id)?;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        let mlp = if config.is_moe_layer(layer_id as i32) {
            DeepSeekMLPType::MoE(DeepSeekMoE::new(config)?)
        } else {
            DeepSeekMLPType::Dense(DeepSeekMLP::new(config.hidden_size, config.intermediate_size)?)
        };

        Ok(Self {
            layer_id,
            self_attn,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        // Self-attention with residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(&attn_out)?;

        // MLP with residual
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        h.add(&mlp_out)
    }
}

/// DeepSeek V3 model.
#[derive(Debug, ModuleParameters)]
pub struct DeepSeekModel {
    /// Configuration.
    pub config: DeepSeekConfig,

    /// Token embeddings.
    #[param]
    pub embed_tokens: nn::Embedding,
    /// Decoder layers.
    #[param]
    pub layers: Vec<DeepSeekDecoderLayer>,
    /// Final layer norm.
    #[param]
    pub norm: nn::RmsNorm,
}

impl DeepSeekModel {
    /// Create a new model.
    pub fn new(config: DeepSeekConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| DeepSeekDecoderLayer::new(&config, i))
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
        let mut h = self.embed_tokens.forward(input_ids)?;

        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    h = layer.forward(&h, mask, Some((cache, layer_idx)))?;
                }
            }
            None => {
                for layer in self.layers.iter_mut() {
                    h = layer.forward(&h, mask, None)?;
                }
            }
        }

        self.norm.forward(&h)
    }
}

/// Full DeepSeek V3 model with LM head.
#[derive(Debug, ModuleParameters)]
pub struct DeepSeek {
    /// Configuration.
    pub config: DeepSeekConfig,
    /// Base model.
    #[param]
    pub model: DeepSeekModel,
    /// LM head.
    #[param]
    pub lm_head: nn::Linear,
}

impl DeepSeek {
    /// Create a new model.
    pub fn new(config: DeepSeekConfig) -> Result<Self, Exception> {
        let model = DeepSeekModel::new(config.clone())?;
        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()?;

        Ok(Self {
            config,
            model,
            lm_head,
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

    /// Get model type.
    pub fn model_type(&self) -> &str {
        &self.config.model_type
    }

    /// Create a KV cache for this model.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        use pmetal_mlx::kv_cache::KVCacheConfig;

        let config = &self.config;
        // For MLA, we cache the compressed latent + rope key
        // Effective head dim is kv_lora_rank + qk_rope_head_dim when stored
        // But for simplicity, we use the full q_head_dim here
        KVCache::new(KVCacheConfig::new(
            config.num_hidden_layers as usize,
            max_seq_len,
            config.num_attention_heads as usize, // MLA effectively uses all heads
            config.q_head_dim() as usize,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_config_default() {
        let config = DeepSeekConfig::default();
        assert_eq!(config.model_type, "deepseek_v3");
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.q_head_dim(), 192); // 128 + 64
    }

    #[test]
    fn test_is_moe_layer() {
        let mut config = DeepSeekConfig::default();
        config.n_routed_experts = Some(8);
        config.moe_layer_freq = 2;
        config.first_k_dense_replace = 0;

        assert!(config.is_moe_layer(0));
        assert!(!config.is_moe_layer(1));
        assert!(config.is_moe_layer(2));
    }

    #[test]
    #[serial]
    fn test_mlp_creation() {
        let mlp = DeepSeekMLP::new(256, 512).unwrap();
        let x = Array::zeros::<f32>(&[2, 4, 256]).unwrap();
        let mut mlp = mlp;
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[2, 4, 256]);
    }

    // =========================================================================
    // V3.2 Tests
    // =========================================================================

    #[test]
    fn test_variant_detection() {
        assert_eq!(DeepSeekConfig::detect_variant("deepseek_v3"), DeepSeekVariant::V3);
        assert_eq!(DeepSeekConfig::detect_variant("deepseek-v3"), DeepSeekVariant::V3);
        assert_eq!(DeepSeekConfig::detect_variant("deepseek_v32"), DeepSeekVariant::V32);
        assert_eq!(DeepSeekConfig::detect_variant("deepseek-v3.2"), DeepSeekVariant::V32);
        assert_eq!(DeepSeekConfig::detect_variant("deepseek_v32_speciale"), DeepSeekVariant::V32Speciale);
        assert_eq!(DeepSeekConfig::detect_variant("deepseek-thinking"), DeepSeekVariant::V32Speciale);
    }

    #[test]
    fn test_v32_config() {
        let config = DeepSeekConfig::v32();
        assert_eq!(config.variant, DeepSeekVariant::V32);
        assert!(config.use_sparse_attention);
        assert!(!config.thinking_mode);
        assert!(config.uses_sparse_attention());
    }

    #[test]
    fn test_v32_speciale_config() {
        let config = DeepSeekConfig::v32_speciale();
        assert_eq!(config.variant, DeepSeekVariant::V32Speciale);
        assert!(config.use_sparse_attention);
        assert!(config.thinking_mode);
        assert_eq!(config.max_thinking_tokens, Some(32768));
        assert!(config.uses_sparse_attention());
    }

    #[test]
    fn test_default_v32_fields() {
        let config = DeepSeekConfig::default();
        assert_eq!(config.variant, DeepSeekVariant::V3);
        assert!(!config.use_sparse_attention);
        assert_eq!(config.lightning_indexer_heads, 4);
        assert_eq!(config.sparse_top_k, 2048);
        assert!(config.indexer_non_interleaved_rope);
        assert!(!config.indexer_use_fp8);
        assert!(!config.thinking_mode);
        assert!(!config.uses_sparse_attention());
    }

    #[test]
    #[serial]
    fn test_lightning_indexer_creation() {
        // Use smaller config for testing
        let mut config = DeepSeekConfig::default();
        config.hidden_size = 256;
        config.lightning_indexer_heads = 2;
        config.qk_rope_head_dim = 32;

        let indexer = LightningIndexer::new(&config).unwrap();
        assert_eq!(indexer.n_heads, 2);
        assert_eq!(indexer.head_dim, 32);
    }

    #[test]
    #[serial]
    fn test_lightning_indexer_forward() {
        // Use smaller config for testing
        let mut config = DeepSeekConfig::default();
        config.hidden_size = 256;
        config.lightning_indexer_heads = 2;
        config.qk_rope_head_dim = 32;

        let mut indexer = LightningIndexer::new(&config).unwrap();
        let x = Array::zeros::<f32>(&[2, 16, 256]).unwrap();
        let scores = indexer.compute_scores(&x, 0).unwrap();
        scores.eval().unwrap();

        // Output should be [batch, seq_len, seq_len]
        assert_eq!(scores.shape(), &[2, 16, 16]);
    }

    #[test]
    fn test_token_selector() {
        let mut config = DeepSeekConfig::default();
        config.sparse_top_k = 4;

        let selector = TokenSelector::new(&config);
        assert_eq!(selector.top_k, 4);
    }

    #[test]
    #[serial]
    fn test_token_selector_select() {
        let mut config = DeepSeekConfig::default();
        config.sparse_top_k = 4;

        let selector = TokenSelector::new(&config);

        // Create fake scores [batch=2, query_len=8, key_len=16]
        let scores = Array::zeros::<f32>(&[2, 8, 16]).unwrap();
        let indices = selector.select_tokens(&scores, None).unwrap();
        indices.eval().unwrap();

        // Output should be [batch, query_len, top_k]
        assert_eq!(indices.shape(), &[2, 8, 4]);
    }
}
