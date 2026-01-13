//! Qwen3-MoE model architecture.
//!
//! Implements Qwen3-MoE with:
//! - Mixture of Experts with top-k routing (softmax-based)
//! - Configurable sparse step (decoder_sparse_step) to control MoE layer frequency
//! - RMSNorm applied to Q and K before RoPE (q_norm, k_norm)
//! - SwitchGLU-style expert MLP with gather_mm

use mlx_rs::{
    builder::Builder, error::Exception, macros::ModuleParameters, module::Module, nn,
    ops::indexing::IndexOp, Array,
};
use pmetal_mlx::kernels::{fused_sdpa, rope::apply_rope, AttentionMaskType, FusedAttentionConfig};
use pmetal_mlx::kv_cache::KVCache;
use pmetal_mlx::moe::{MoEConfig, MoELayer};
use serde::{Deserialize, Serialize};

/// Qwen3-MoE model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3MoEConfig {
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
    /// Intermediate size for MoE expert MLPs.
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    /// Number of hidden layers.
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<i32>,
    /// Head dimension (fixed at 128 for Qwen3).
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    /// Maximum sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// RMS normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Number of experts.
    #[serde(default = "default_num_experts")]
    pub num_experts: i32,
    /// Number of experts to route to per token.
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: i32,
    /// Sparse decoder step - MoE layers are used every N-th layer.
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: i32,
    /// Layers that should use dense MLP instead of MoE.
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    /// Whether to normalize top-k routing probabilities.
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
}

fn default_model_type() -> String {
    "qwen3_moe".to_string()
}
fn default_vocab_size() -> i32 {
    151936
}
fn default_head_dim() -> i32 {
    128
}
fn default_max_position_embeddings() -> i32 {
    32768
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_num_experts() -> i32 {
    64
}
fn default_num_experts_per_tok() -> i32 {
    8
}
fn default_decoder_sparse_step() -> i32 {
    1
}
fn default_norm_topk_prob() -> bool {
    true
}

impl Qwen3MoEConfig {
    /// Get the number of KV heads (defaults to num_attention_heads if not specified).
    pub fn num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the MoE intermediate size (defaults to intermediate_size if not specified).
    pub fn get_moe_intermediate_size(&self) -> i32 {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    /// Check if layer at index should use MoE.
    /// Returns true if:
    /// 1. The layer is not in mlp_only_layers
    /// 2. The layer index + 1 is divisible by decoder_sparse_step
    pub fn use_moe_at(&self, layer_idx: usize) -> bool {
        let layer_idx_i32 = layer_idx as i32;
        // Check if this layer is in the mlp_only list
        if self.mlp_only_layers.contains(&layer_idx_i32) {
            return false;
        }
        // Check if this is a sparse step layer
        self.num_experts > 0 && ((layer_idx_i32 + 1) % self.decoder_sparse_step == 0)
    }
}

impl Default for Qwen3MoEConfig {
    fn default() -> Self {
        Self {
            model_type: "qwen3_moe".to_string(),
            vocab_size: 151936,
            hidden_size: 2048,
            intermediate_size: 5632,
            moe_intermediate_size: Some(1408),
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            head_dim: 128,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            num_experts: 64,
            num_experts_per_tok: 8,
            decoder_sparse_step: 1,
            mlp_only_layers: vec![],
            norm_topk_prob: true,
        }
    }
}

/// Qwen3-MoE attention with Q/K normalization before RoPE.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MoEAttention {
    /// Configuration.
    #[allow(dead_code)]
    config: Qwen3MoEConfig,
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
    /// Query normalization.
    #[param]
    pub q_norm: nn::RmsNorm,
    /// Key normalization.
    #[param]
    pub k_norm: nn::RmsNorm,
}

impl Qwen3MoEAttention {
    /// Create a new attention layer.
    pub fn new(config: Qwen3MoEConfig) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;
        let scale = (head_dim as f32).powf(-0.5);

        let q_proj = nn::LinearBuilder::new(hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, hidden_size)
            .bias(false)
            .build()?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            rope_theta: config.rope_theta,
            config,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
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
        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape and apply per-head normalization
        q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // Apply Q/K normalization (Qwen3 specific)
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;

        // Transpose to [batch, heads, seq, head_dim]
        q = q.transpose_axes(&[0, 2, 1, 3])?;
        k = k.transpose_axes(&[0, 2, 1, 3])?;
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

        // Use fused SDPA
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        // Transpose back and project
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

        self.o_proj.forward(&output)
    }
}

/// Dense MLP for non-MoE layers.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MoEDenseMLP {
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

impl Qwen3MoEDenseMLP {
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

    /// Forward pass through MLP.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // SwiGLU: silu(gate) * up
        let activated = nn::silu(&gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }
}

/// MoE block with softmax routing for Qwen3-MoE.
#[derive(Debug)]
pub struct Qwen3MoEBlock {
    /// Number of experts.
    pub num_experts: usize,
    /// Top-k experts per token.
    pub top_k: i32,
    /// Whether to normalize top-k probabilities.
    pub norm_topk_prob: bool,
    /// Gate projection (routes to experts).
    pub gate: nn::Linear,
    /// MoE layer with all experts.
    pub moe: MoELayer,
}

impl Qwen3MoEBlock {
    /// Create a new MoE block.
    pub fn new(config: &Qwen3MoEConfig) -> Result<Self, Exception> {
        let num_experts = config.num_experts as usize;
        let moe_intermediate = config.get_moe_intermediate_size();

        let gate = nn::LinearBuilder::new(config.hidden_size, config.num_experts)
            .bias(false)
            .build()?;

        let moe_config = MoEConfig::new(config.hidden_size, moe_intermediate, num_experts)
            .with_num_experts_per_tok(config.num_experts_per_tok as usize)
            .with_aux_loss(false, 0.0);

        let moe = MoELayer::new(moe_config);

        Ok(Self {
            num_experts,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            gate,
            moe,
        })
    }

    /// Forward pass through MoE block.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Compute routing scores
        let gates = self.gate.forward(x)?;
        let gates = mlx_rs::ops::softmax_axis(&gates, -1, None)?;

        // Select top-k experts
        let neg_k = -self.top_k;
        let inds = mlx_rs::ops::argpartition_axis(&gates, neg_k, -1)?;
        // Get the last k elements (top-k)
        let k = self.top_k as usize;
        let inds_shape = inds.shape();
        let last_dim = inds_shape[inds_shape.len() - 1] as usize;
        let start = last_dim - k;

        // Slice to get top-k indices using index
        let inds = inds.index((.., .., start as i32..));

        // Get scores for selected experts
        let _scores = gates.take_along_axis(&inds, -1)?;

        // For now, use the MoE layer directly
        // TODO: Use custom routing with inds and scores
        self.moe.eval();
        let (output, _aux_loss) = self.moe.forward(x)?;

        Ok(output)
    }
}

/// Feed-forward for a decoder layer - either dense MLP or MoE.
#[derive(Debug)]
pub enum Qwen3MoEFeedForward {
    /// Dense MLP.
    Dense(Qwen3MoEDenseMLP),
    /// Mixture of Experts.
    MoE(Qwen3MoEBlock),
}

impl Qwen3MoEFeedForward {
    /// Forward pass.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::MoE(moe) => moe.forward(x),
        }
    }
}

/// Qwen3-MoE decoder layer.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MoEDecoderLayer {
    /// Configuration.
    #[allow(dead_code)]
    pub config: Qwen3MoEConfig,
    /// Self-attention.
    #[param]
    pub self_attn: Qwen3MoEAttention,
    /// Input layer norm.
    #[param]
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm.
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    /// Feed-forward (MLP or MoE) - not tracked as param since enum.
    pub ffn: Qwen3MoEFeedForward,
}

impl Qwen3MoEDecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: Qwen3MoEConfig, layer_idx: usize) -> Result<Self, Exception> {
        let self_attn = Qwen3MoEAttention::new(config.clone())?;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        // Determine if this layer uses MoE
        let ffn = if config.use_moe_at(layer_idx) {
            Qwen3MoEFeedForward::MoE(Qwen3MoEBlock::new(&config)?)
        } else {
            Qwen3MoEFeedForward::Dense(Qwen3MoEDenseMLP::new(
                config.hidden_size,
                config.intermediate_size,
            )?)
        };

        Ok(Self {
            config,
            self_attn,
            input_layernorm,
            post_attention_layernorm,
            ffn,
        })
    }

    /// Forward pass through decoder layer.
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

        // FFN with residual
        let normed = self.post_attention_layernorm.forward(&h)?;
        let ffn_out = self.ffn.forward(&normed)?;
        h.add(&ffn_out)
    }
}

/// Qwen3-MoE base model (without LM head).
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MoEModel {
    /// Configuration.
    pub config: Qwen3MoEConfig,
    /// Token embeddings.
    #[param]
    pub embed_tokens: nn::Embedding,
    /// Transformer layers.
    #[param]
    pub layers: Vec<Qwen3MoEDecoderLayer>,
    /// Final layer norm.
    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen3MoEModel {
    /// Create a new model.
    pub fn new(config: Qwen3MoEConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for layer_idx in 0..config.num_hidden_layers as usize {
            layers.push(Qwen3MoEDecoderLayer::new(config.clone(), layer_idx)?);
        }

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

/// Full Qwen3-MoE model with LM head.
#[derive(Debug, ModuleParameters)]
pub struct Qwen3MoE {
    /// Configuration.
    pub config: Qwen3MoEConfig,
    /// Base model.
    #[param]
    pub model: Qwen3MoEModel,
    /// LM head.
    #[param]
    pub lm_head: nn::Linear,
}

impl Qwen3MoE {
    /// Create a new model.
    pub fn new(config: Qwen3MoEConfig) -> Result<Self, Exception> {
        let model = Qwen3MoEModel::new(config.clone())?;

        let lm_head = if config.tie_word_embeddings {
            // When tying embeddings, lm_head uses embed_tokens weights
            // For now, create a separate linear layer
            nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                .bias(false)
                .build()?
        } else {
            nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                .bias(false)
                .build()?
        };

        Ok(Self {
            config,
            model,
            lm_head,
        })
    }

    /// Forward pass through full model.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.model.forward(input_ids, mask, cache)?;
        self.lm_head.forward(&hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_config_defaults() {
        let config = Qwen3MoEConfig::default();
        assert_eq!(config.num_experts, 64);
        assert_eq!(config.num_experts_per_tok, 8);
        assert!(config.use_moe_at(0)); // Layer 0 should use MoE with sparse_step=1
    }

    #[test]
    fn test_use_moe_at() {
        let mut config = Qwen3MoEConfig::default();

        // With sparse_step=2, only odd layers (0-indexed layer 1, 3, 5...) should be MoE
        config.decoder_sparse_step = 2;
        assert!(!config.use_moe_at(0)); // Layer 1: 0+1=1, 1%2=1 != 0
        assert!(config.use_moe_at(1)); // Layer 2: 1+1=2, 2%2=0
        assert!(!config.use_moe_at(2)); // Layer 3: 2+1=3, 3%2=1 != 0
        assert!(config.use_moe_at(3)); // Layer 4: 3+1=4, 4%2=0

        // Test mlp_only_layers
        config.mlp_only_layers = vec![1];
        assert!(!config.use_moe_at(1)); // Forced to dense
    }

    #[test]
    #[serial]
    fn test_create_attention() {
        let config = Qwen3MoEConfig::default();
        let attn = Qwen3MoEAttention::new(config);
        assert!(attn.is_ok());
    }

    #[test]
    #[serial]
    fn test_create_dense_mlp() {
        let mlp = Qwen3MoEDenseMLP::new(2048, 5632);
        assert!(mlp.is_ok());
    }
}
