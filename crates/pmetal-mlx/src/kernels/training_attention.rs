//! Training-aware attention using Metal FlashAttention.
//!
//! This module provides attention computation with proper backward pass support
//! for training on Apple Silicon. It uses custom Metal FlashAttention kernels
//! that implement O(n) memory complexity for both forward and backward passes.
//!
//! # Why This Module?
//!
//! MLX's built-in `scaled_dot_product_attention` backward pass is NOT IMPLEMENTED
//! for Metal (it falls back to O(n²) naive computation). This module provides
//! the missing efficient backward pass needed for training.

use half::f16;
use mlx_rs::Array;
use std::sync::Arc;

use pmetal_metal::{
    FlashAttention, FlashAttentionConfig as MetalFAConfig, MetalBuffer, MetalContext,
};

use super::fused_attention::FusedAttentionConfig;
use super::utils::{array_to_metal_buffer_f16, metal_buffer_to_array_f16};
use crate::error::MlxError;

/// Result type for training attention.
pub type Result<T> = std::result::Result<T, MlxError>;

/// Context for training attention operations.
///
/// Manages Metal resources and caches for efficient training.
pub struct TrainingAttentionContext {
    /// Metal context (shared).
    metal_ctx: Arc<MetalContext>,
}

impl TrainingAttentionContext {
    /// Create a new training attention context.
    ///
    /// # Errors
    ///
    /// Returns an error if Metal initialization fails.
    pub fn new() -> Result<Self> {
        let metal_ctx = MetalContext::global()
            .map_err(|e| MlxError::Metal(e.to_string()))?;

        Ok(Self { metal_ctx })
    }

    /// Get the Metal context.
    pub fn metal_context(&self) -> &Arc<MetalContext> {
        &self.metal_ctx
    }
}

/// Saved tensors from forward pass needed for backward.
pub struct AttentionForwardCache {
    /// Query tensor [batch, n_heads, seq_len, head_dim] in f16.
    pub queries: MetalBuffer<f16>,
    /// Key tensor [batch, n_kv_heads, seq_len, head_dim] in f16.
    pub keys: MetalBuffer<f16>,
    /// Value tensor [batch, n_kv_heads, seq_len, head_dim] in f16.
    pub values: MetalBuffer<f16>,
    /// Output tensor [batch, n_heads, seq_len, head_dim] in f16.
    pub output: MetalBuffer<f16>,
    /// Log-sum-exp for backward [batch, n_heads, seq_len] in f32.
    pub logsumexp: MetalBuffer<f32>,
    /// The FlashAttention instance for backward.
    pub flash_attn: FlashAttention,
}

/// Output from training attention forward pass.
pub struct TrainingAttentionOutput {
    /// Attention output as MLX Array.
    pub output: Array,
    /// Cache for backward pass.
    pub cache: AttentionForwardCache,
}

/// Gradients from attention backward pass.
pub struct AttentionGradients {
    /// Gradient w.r.t queries [batch, n_heads, seq_len, head_dim].
    pub d_queries: Array,
    /// Gradient w.r.t keys [batch, n_kv_heads, seq_len, head_dim].
    pub d_keys: Array,
    /// Gradient w.r.t values [batch, n_kv_heads, seq_len, head_dim].
    pub d_values: Array,
}

/// Compute attention forward pass with cache for backward.
///
/// Uses Metal FlashAttention kernels for O(n) memory complexity.
///
/// # Arguments
///
/// * `ctx` - Training attention context
/// * `queries` - Query tensor [batch, n_heads, seq_len, head_dim]
/// * `keys` - Key tensor [batch, n_kv_heads, seq_len, head_dim]
/// * `values` - Value tensor [batch, n_kv_heads, seq_len, head_dim]
/// * `config` - Attention configuration
///
/// # Returns
///
/// Output tensor and cache for backward pass.
pub fn training_attention_forward(
    ctx: &TrainingAttentionContext,
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
) -> Result<TrainingAttentionOutput> {
    // Get shapes
    let q_shape = queries.shape();
    let batch_size = q_shape[0] as usize;
    let num_heads = q_shape[1] as usize;
    let query_seq_len = q_shape[2] as usize;
    let head_dim = q_shape[3] as usize;

    let k_shape = keys.shape();
    let num_kv_heads = k_shape[1] as usize;
    let kv_seq_len = k_shape[2] as usize;

    // Convert to Metal buffers
    let metal_ctx = ctx.metal_context();
    let q_buffer = array_to_metal_buffer_f16(metal_ctx, queries)?;
    let k_buffer = array_to_metal_buffer_f16(metal_ctx, keys)?;
    let v_buffer = array_to_metal_buffer_f16(metal_ctx, values)?;

    // Create FlashAttention config
    let is_causal = matches!(
        config.mask_type,
        super::fused_attention::AttentionMaskType::Causal
    );
    let sliding_window = match config.mask_type {
        super::fused_attention::AttentionMaskType::SlidingWindow(w) => Some(w as usize),
        _ => None,
    };

    let fa_config = MetalFAConfig {
        batch_size,
        num_heads,
        num_kv_heads,
        query_seq_len,
        kv_seq_len,
        head_dim,
        scale: Some(config.scale),
        is_causal,
        sliding_window,
        softcap: config.logit_softcapping,
        is_training: true,  // Store logsumexp for backward
    };

    // Create FlashAttention instance
    let flash_attn = FlashAttention::new(metal_ctx.clone(), fa_config)
        .map_err(|e| MlxError::Metal(e.to_string()))?;

    // Run forward pass
    let fa_output = flash_attn.forward(&q_buffer, &k_buffer, &v_buffer)
        .map_err(|e| MlxError::Metal(e.to_string()))?;

    // Convert output to MLX Array
    let output_array = metal_buffer_to_array_f16(
        &fa_output.output,
        &[q_shape[0], q_shape[1], q_shape[2], q_shape[3]],
    )?;

    // Build cache for backward
    let cache = AttentionForwardCache {
        queries: q_buffer,
        keys: k_buffer,
        values: v_buffer,
        output: fa_output.output,
        logsumexp: fa_output.logsumexp.ok_or_else(|| {
            MlxError::Metal("logsumexp not returned from forward pass".to_string())
        })?,
        flash_attn,
    };

    Ok(TrainingAttentionOutput {
        output: output_array,
        cache,
    })
}

/// Compute attention backward pass.
///
/// Uses Metal FlashAttention backward kernels for efficient gradient computation.
///
/// # Arguments
///
/// * `ctx` - Training attention context
/// * `d_output` - Gradient of loss w.r.t. attention output
/// * `cache` - Cache from forward pass
///
/// # Returns
///
/// Gradients w.r.t. queries, keys, and values.
pub fn training_attention_backward(
    ctx: &TrainingAttentionContext,
    d_output: &Array,
    cache: &AttentionForwardCache,
) -> Result<AttentionGradients> {
    let metal_ctx = ctx.metal_context();

    // Convert d_output to Metal buffer
    let d_out_buffer = array_to_metal_buffer_f16(metal_ctx, d_output)?;

    // Run backward pass
    let (d_q, d_k, d_v) = cache.flash_attn.backward(
        &cache.queries,
        &cache.keys,
        &cache.values,
        &cache.output,
        &d_out_buffer,
        &cache.logsumexp,
    ).map_err(|e| MlxError::Metal(e.to_string()))?;

    // Get shapes for conversion
    let q_shape = [
        cache.flash_attn.config().batch_size as i32,
        cache.flash_attn.config().num_heads as i32,
        cache.flash_attn.config().query_seq_len as i32,
        cache.flash_attn.config().head_dim as i32,
    ];
    let kv_shape = [
        cache.flash_attn.config().batch_size as i32,
        cache.flash_attn.config().num_kv_heads as i32,
        cache.flash_attn.config().kv_seq_len as i32,
        cache.flash_attn.config().head_dim as i32,
    ];

    // Convert gradients to MLX Arrays
    let d_queries = metal_buffer_to_array_f16(&d_q, &q_shape)?;
    let d_keys = metal_buffer_to_array_f16(&d_k, &kv_shape)?;
    let d_values = metal_buffer_to_array_f16(&d_v, &kv_shape)?;

    Ok(AttentionGradients {
        d_queries,
        d_keys,
        d_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::random::uniform;

    fn random_tensor(shape: &[i32]) -> Array {
        uniform::<_, f32>(0.0, 1.0, shape, None).unwrap()
    }

    #[test]
    fn test_training_context_creation() {
        let ctx = TrainingAttentionContext::new();
        assert!(ctx.is_ok(), "Should be able to create training context on macOS");
    }

    #[test]
    fn test_training_attention_forward() {
        let ctx = TrainingAttentionContext::new().unwrap();

        let batch = 1;
        let n_heads = 4;
        let n_kv_heads = 4;
        let seq_len = 32;
        let head_dim = 64;

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_kv_heads, head_dim);

        let result = training_attention_forward(&ctx, &queries, &keys, &values, &config);
        assert!(result.is_ok(), "Forward pass should succeed");

        let output = result.unwrap();
        assert_eq!(output.output.shape(), &[batch, n_heads, seq_len, head_dim]);
    }
}
