//! Differentiable attention with efficient Metal FlashAttention backend.
//!
//! This module provides a training-aware attention implementation that:
//! 1. Uses Metal FlashAttention for O(n) memory-efficient forward pass
//! 2. Caches activations for efficient backward pass
//! 3. Computes gradients using Metal FlashAttention backward kernels
//!
//! # Integration with MLX Autodiff
//!
//! MLX's autodiff works by building a computation graph and computing VJPs.
//! For attention, MLX's built-in backward pass is NOT IMPLEMENTED on Metal,
//! falling back to O(n²) naive computation.
//!
//! This module provides an alternative path:
//! 1. Use `stop_gradient` on the attention output for MLX's graph
//! 2. Manually compute attention gradients using Metal FlashAttention
//! 3. Inject gradients back into the optimizer step
//!
//! This "hybrid" approach lets us use MLX's autodiff for the rest of the model
//! while using our efficient Metal kernels for attention.

use half::f16;
use mlx_rs::Array;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pmetal_metal::{
    FlashAttention, FlashAttentionConfig as MetalFAConfig, MetalBuffer, MetalContext,
};

use super::fused_attention::{AttentionMaskType, FusedAttentionConfig};
use super::utils::{array_to_metal_buffer_f16, metal_buffer_to_array_f16};
use crate::error::MlxError;

/// Result type for differentiable attention operations.
pub type Result<T> = std::result::Result<T, MlxError>;

/// Activation cache for a single attention layer.
#[derive(Debug)]
pub struct AttentionCache {
    /// Queries in f16 [batch, n_heads, seq_len, head_dim].
    pub queries: MetalBuffer<f16>,
    /// Keys in f16 [batch, n_kv_heads, seq_len, head_dim].
    pub keys: MetalBuffer<f16>,
    /// Values in f16 [batch, n_kv_heads, seq_len, head_dim].
    pub values: MetalBuffer<f16>,
    /// Output in f16 [batch, n_heads, seq_len, head_dim].
    pub output: MetalBuffer<f16>,
    /// Log-sum-exp for numerical stability [batch, n_heads, seq_len].
    pub logsumexp: MetalBuffer<f32>,
    /// FlashAttention instance for backward pass.
    pub flash_attn: FlashAttention,
    /// Original query shape for gradient reconstruction.
    pub query_shape: Vec<i32>,
    /// Original KV shape for gradient reconstruction.
    pub kv_shape: Vec<i32>,
}

/// Global training context that manages attention caches across layers.
///
/// This allows the training loop to:
/// 1. Enable training mode before forward pass
/// 2. Collect attention caches during forward
/// 3. Compute attention gradients during backward
pub struct TrainingContext {
    /// Metal context (shared).
    metal_ctx: Arc<MetalContext>,
    /// Whether training mode is enabled.
    training: bool,
    /// Attention caches by layer ID.
    caches: Mutex<HashMap<usize, AttentionCache>>,
}

impl TrainingContext {
    /// Create a new training context.
    pub fn new() -> Result<Self> {
        let metal_ctx = MetalContext::global()
            .map_err(|e| MlxError::Metal(e.to_string()))?;

        Ok(Self {
            metal_ctx,
            training: false,
            caches: Mutex::new(HashMap::new()),
        })
    }

    /// Enable training mode.
    pub fn enable_training(&mut self) {
        self.training = true;
    }

    /// Disable training mode and clear caches.
    pub fn disable_training(&mut self) {
        self.training = false;
        self.clear_caches();
    }

    /// Check if training mode is enabled.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Clear all cached activations.
    pub fn clear_caches(&self) {
        if let Ok(mut caches) = self.caches.lock() {
            caches.clear();
        }
    }

    /// Get Metal context.
    pub fn metal_context(&self) -> &Arc<MetalContext> {
        &self.metal_ctx
    }

    /// Store attention cache for a layer.
    pub fn store_cache(&self, layer_id: usize, cache: AttentionCache) {
        if let Ok(mut caches) = self.caches.lock() {
            caches.insert(layer_id, cache);
        }
    }

    /// Take attention cache for a layer (removes from storage).
    pub fn take_cache(&self, layer_id: usize) -> Option<AttentionCache> {
        if let Ok(mut caches) = self.caches.lock() {
            caches.remove(&layer_id)
        } else {
            None
        }
    }

    /// Check if cache exists for a layer.
    pub fn has_cache(&self, layer_id: usize) -> bool {
        if let Ok(caches) = self.caches.lock() {
            caches.contains_key(&layer_id)
        } else {
            false
        }
    }
}

/// Thread-local training context for easy access.
thread_local! {
    static TRAINING_CONTEXT: std::cell::RefCell<Option<Arc<Mutex<TrainingContext>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Initialize the global training context.
pub fn init_training_context() -> Result<Arc<Mutex<TrainingContext>>> {
    let ctx = Arc::new(Mutex::new(TrainingContext::new()?));
    TRAINING_CONTEXT.with(|c| {
        *c.borrow_mut() = Some(ctx.clone());
    });
    Ok(ctx)
}

/// Get the global training context.
pub fn get_training_context() -> Option<Arc<Mutex<TrainingContext>>> {
    TRAINING_CONTEXT.with(|c| c.borrow().clone())
}

/// Differentiable attention forward pass.
///
/// When training mode is enabled via `TrainingContext`, this:
/// 1. Runs Metal FlashAttention forward
/// 2. Caches activations for backward pass
/// 3. Returns output wrapped with stop_gradient for MLX graph
///
/// When training mode is disabled, falls back to standard fused_sdpa.
///
/// # Arguments
///
/// * `layer_id` - Unique identifier for this attention layer
/// * `queries` - Query tensor [batch, n_heads, seq_len, head_dim]
/// * `keys` - Key tensor [batch, n_kv_heads, seq_len, head_dim]
/// * `values` - Value tensor [batch, n_kv_heads, seq_len, head_dim]
/// * `config` - Attention configuration
///
/// # Returns
///
/// Attention output tensor.
pub fn differentiable_attention(
    layer_id: usize,
    queries: &Array,
    keys: &Array,
    values: &Array,
    config: &FusedAttentionConfig,
) -> Result<Array> {
    // Check if training mode is enabled
    let training_ctx = get_training_context();
    let is_training = training_ctx
        .as_ref()
        .map(|ctx| ctx.lock().map(|c| c.is_training()).unwrap_or(false))
        .unwrap_or(false);

    if !is_training {
        // Inference mode: use standard fused_sdpa
        return super::fused_attention::fused_sdpa(queries, keys, values, config, None)
            .map_err(MlxError::from);
    }

    // Training mode: use Metal FlashAttention with caching
    let ctx = training_ctx.expect("Training context should be initialized");
    let ctx_guard = ctx.lock().map_err(|_| MlxError::Metal("Failed to lock training context".to_string()))?;
    let metal_ctx = ctx_guard.metal_context();

    // Get shapes
    let q_shape = queries.shape().to_vec();
    let k_shape = keys.shape().to_vec();
    let batch_size = q_shape[0] as usize;
    let num_heads = q_shape[1] as usize;
    let query_seq_len = q_shape[2] as usize;
    let head_dim = q_shape[3] as usize;
    let num_kv_heads = k_shape[1] as usize;
    let kv_seq_len = k_shape[2] as usize;

    // Convert to Metal buffers
    let q_buffer = array_to_metal_buffer_f16(metal_ctx, queries)?;
    let k_buffer = array_to_metal_buffer_f16(metal_ctx, keys)?;
    let v_buffer = array_to_metal_buffer_f16(metal_ctx, values)?;

    // Create FlashAttention config
    let is_causal = matches!(config.mask_type, AttentionMaskType::Causal);
    let sliding_window = match config.mask_type {
        AttentionMaskType::SlidingWindow(w) => Some(w as usize),
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
        is_training: true, // Store logsumexp for backward
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

    // Create cache for backward pass
    let cache = AttentionCache {
        queries: q_buffer,
        keys: k_buffer,
        values: v_buffer,
        output: fa_output.output,
        logsumexp: fa_output.logsumexp.ok_or_else(|| {
            MlxError::Metal("logsumexp not returned from forward pass".to_string())
        })?,
        flash_attn,
        query_shape: q_shape,
        kv_shape: k_shape,
    };

    // Store cache (need to drop ctx_guard first to release lock)
    drop(ctx_guard);
    let ctx_guard = ctx.lock().map_err(|_| MlxError::Metal("Failed to lock training context".to_string()))?;
    ctx_guard.store_cache(layer_id, cache);

    Ok(output_array)
}

/// Compute attention gradients for a layer.
///
/// Uses the cached activations from forward pass to compute gradients
/// via Metal FlashAttention backward kernels.
///
/// # Arguments
///
/// * `layer_id` - Layer ID matching the forward pass
/// * `d_output` - Gradient of loss w.r.t. attention output
///
/// # Returns
///
/// Tuple of (d_queries, d_keys, d_values) gradients.
pub fn compute_attention_gradients(
    layer_id: usize,
    d_output: &Array,
) -> Result<(Array, Array, Array)> {
    let training_ctx = get_training_context()
        .ok_or_else(|| MlxError::Metal("Training context not initialized".to_string()))?;

    let ctx_guard = training_ctx.lock()
        .map_err(|_| MlxError::Metal("Failed to lock training context".to_string()))?;

    // Take cache (removes it from storage)
    let cache = ctx_guard.take_cache(layer_id)
        .ok_or_else(|| MlxError::Metal(format!("No cache found for layer {}", layer_id)))?;

    let metal_ctx = ctx_guard.metal_context();

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

    // Convert gradients to MLX Arrays
    let d_queries = metal_buffer_to_array_f16(&d_q, &cache.query_shape)?;
    let d_keys = metal_buffer_to_array_f16(&d_k, &cache.kv_shape)?;
    let d_values = metal_buffer_to_array_f16(&d_v, &cache.kv_shape)?;

    Ok((d_queries, d_keys, d_values))
}

/// Training loop integration helper.
///
/// Wraps a training step with proper training context management:
/// 1. Enables training mode before forward pass
/// 2. Executes the training function
/// 3. Disables training mode after
///
/// # Example
///
/// ```ignore
/// with_training_mode(|| {
///     let logits = model.forward(&input_ids, None)?;
///     let loss = compute_loss(&logits, &labels)?;
///     // gradients are computed here
///     Ok(loss)
/// })
/// ```
pub fn with_training_mode<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    // Get or create training context
    let ctx = get_training_context()
        .map(Ok)
        .unwrap_or_else(init_training_context)?;

    // Enable training mode
    {
        let mut ctx_guard = ctx.lock()
            .map_err(|_| MlxError::Metal("Failed to lock training context".to_string()))?;
        ctx_guard.enable_training();
    }

    // Execute training function
    let result = f();

    // Disable training mode and clear caches
    {
        let mut ctx_guard = ctx.lock()
            .map_err(|_| MlxError::Metal("Failed to lock training context".to_string()))?;
        ctx_guard.disable_training();
    }

    result
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
        let ctx = TrainingContext::new();
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_training_context_mode() {
        let mut ctx = TrainingContext::new().unwrap();
        assert!(!ctx.is_training());

        ctx.enable_training();
        assert!(ctx.is_training());

        ctx.disable_training();
        assert!(!ctx.is_training());
    }

    #[test]
    fn test_differentiable_attention_inference() {
        // Without training context, should use fused_sdpa
        let batch = 1;
        let n_heads = 4;
        let n_kv_heads = 4;
        let seq_len = 32;
        let head_dim = 64;

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_kv_heads, head_dim);

        let result = differentiable_attention(0, &queries, &keys, &values, &config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().shape(), &[batch, n_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_differentiable_attention_training() {
        // Initialize training context
        let ctx = init_training_context().unwrap();

        // Enable training mode
        {
            let mut ctx_guard = ctx.lock().unwrap();
            ctx_guard.enable_training();
        }

        let batch = 1;
        let n_heads = 4;
        let n_kv_heads = 4;
        let seq_len = 32;
        let head_dim = 64;

        let queries = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let keys = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);
        let values = random_tensor(&[batch, n_kv_heads, seq_len, head_dim]);

        let config = FusedAttentionConfig::new(n_heads, n_kv_heads, head_dim);

        // Forward pass
        let result = differentiable_attention(0, &queries, &keys, &values, &config);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[batch, n_heads, seq_len, head_dim]);

        // Check cache was stored
        {
            let ctx_guard = ctx.lock().unwrap();
            assert!(ctx_guard.has_cache(0));
        }

        // Compute gradients
        let d_output = random_tensor(&[batch, n_heads, seq_len, head_dim]);
        let grads = compute_attention_gradients(0, &d_output);
        assert!(grads.is_ok(), "Gradient computation failed: {:?}", grads.err());

        let (d_q, d_k, d_v) = grads.unwrap();
        assert_eq!(d_q.shape(), &[batch, n_heads, seq_len, head_dim]);
        assert_eq!(d_k.shape(), &[batch, n_kv_heads, seq_len, head_dim]);
        assert_eq!(d_v.shape(), &[batch, n_kv_heads, seq_len, head_dim]);

        // Cache should be consumed
        {
            let ctx_guard = ctx.lock().unwrap();
            assert!(!ctx_guard.has_cache(0));
        }

        // Cleanup
        {
            let mut ctx_guard = ctx.lock().unwrap();
            ctx_guard.disable_training();
        }
    }

    #[test]
    fn test_with_training_mode() {
        let result = with_training_mode(|| {
            // Check training is enabled
            let ctx = get_training_context().unwrap();
            let ctx_guard = ctx.lock().unwrap();
            assert!(ctx_guard.is_training());
            Ok(42)
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        // Check training is disabled after
        if let Some(ctx) = get_training_context() {
            let ctx_guard = ctx.lock().unwrap();
            assert!(!ctx_guard.is_training());
        }
    }
}
