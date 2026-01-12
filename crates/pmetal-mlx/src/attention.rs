//! Optimized attention implementations.
//!
//! This module provides attention implementations optimized for Apple Silicon:
//! - Standard scaled dot-product attention
//! - Flash attention (via MLX's mx.fast)
//! - Grouped-query attention (GQA)
//! - Multi-query attention (MQA)

use pmetal_core::Result;

/// Attention configuration.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads (for GQA/MQA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Dropout probability.
    pub dropout: f32,
    /// Use causal attention mask.
    pub is_causal: bool,
    /// Softmax scale (default: 1/sqrt(head_dim)).
    pub scale: Option<f32>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            dropout: 0.0,
            is_causal: true,
            scale: None,
        }
    }
}

impl AttentionConfig {
    /// Get the softmax scaling factor.
    #[must_use]
    pub fn scaling_factor(&self) -> f32 {
        self.scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt())
    }

    /// Check if this is grouped-query attention.
    #[must_use]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }

    /// Check if this is multi-query attention.
    #[must_use]
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }

    /// Get the number of query groups.
    #[must_use]
    pub fn num_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// Attention backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionBackend {
    /// Use MLX's mx.fast.scaled_dot_product_attention.
    #[default]
    MlxFast,
    /// Standard attention implementation.
    Standard,
    /// Variable-length attention for packed sequences.
    VarLen,
}

/// Multi-head attention layer.
pub struct MultiHeadAttention {
    /// Configuration.
    pub config: AttentionConfig,
    /// Backend to use.
    pub backend: AttentionBackend,
    // Projection weights will be added when integrating with mlx-rs
    // q_proj: Array,
    // k_proj: Array,
    // v_proj: Array,
    // o_proj: Array,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer.
    pub fn new(config: AttentionConfig, backend: AttentionBackend) -> Result<Self> {
        Ok(Self { config, backend })
    }

    // Placeholder for forward pass - will be implemented with mlx-rs
    // pub fn forward(
    //     &self,
    //     hidden_states: &Array,
    //     attention_mask: Option<&Array>,
    //     position_ids: Option<&Array>,
    //     past_key_value: Option<(&Array, &Array)>,
    // ) -> Result<(Array, Option<(Array, Array)>)> { ... }
}

/// Create a causal attention mask.
///
/// Returns a lower-triangular boolean mask of shape [seq_len, seq_len].
pub fn create_causal_mask(seq_len: usize) -> Vec<Vec<bool>> {
    (0..seq_len)
        .map(|i| (0..seq_len).map(|j| j <= i).collect())
        .collect()
}
