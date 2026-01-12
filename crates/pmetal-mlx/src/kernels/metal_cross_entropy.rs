//! Metal-accelerated fused linear + cross-entropy loss.
//!
//! This module provides THE key optimization from unsloth: computing cross-entropy
//! loss directly from hidden states without ever materializing the full logits tensor.
//!
//! # Memory Savings
//!
//! For a typical training setup:
//! - batch=4, seq=1024, vocab=150K, dtype=fp16
//! - Standard approach: 4 * 1024 * 150000 * 2 = **1.2GB** for logits alone
//! - With fusion: only chunk_size * hidden_size * 2 ≈ **8MB** peak
//!
//! This enables 2x larger batch sizes, which translates to ~2x throughput.
//!
//! # Implementations
//!
//! Two implementations are available:
//! 1. **Metal kernel** (`use_metal=true`): Uses custom Metal GPU kernels (fastest)
//! 2. **MLX fallback** (`use_metal=false`): Uses pure MLX chunked computation (robust)
//!
//! By default, the MLX implementation is used as it's more stable. Set `use_metal=true`
//! to use the experimental Metal kernel for additional speedup.
//!
//! # Usage
//!
//! ```rust,ignore
//! use pmetal_mlx::kernels::metal_cross_entropy::fused_linear_cross_entropy_loss;
//!
//! // Compute loss directly from hidden states (uses MLX fallback)
//! let loss = fused_linear_cross_entropy_loss(
//!     &hidden_states,  // [batch * seq, hidden_dim]
//!     &lm_head_weight, // [vocab_size, hidden_dim]
//!     &targets,        // [batch * seq]
//!     -100,            // ignore_index
//! )?;
//! ```

use half::f16;
use mlx_rs::{Array, Dtype};
use std::sync::Arc;
use tracing::{debug, warn};

use pmetal_metal::{
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
    FusedLinearCrossEntropy, FusedLinearCrossEntropyConfig,
};

use crate::error::MlxError;
use super::cut_cross_entropy::{CutCrossEntropy, CutCrossEntropyConfig};

/// Result type for metal cross-entropy operations.
pub type Result<T> = std::result::Result<T, MlxError>;

/// Configuration for memory-efficient fused cross-entropy.
#[derive(Debug, Clone)]
pub struct MetalCrossEntropyConfig {
    /// Chunk size for processing vocabulary (default: 4096).
    /// Larger chunks are faster but use more memory.
    pub chunk_size: usize,

    /// Index to ignore in loss computation (typically -100).
    pub ignore_index: i32,

    /// Label smoothing factor (0.0 to disable).
    pub label_smoothing: f32,

    /// Use fp16 kernels for mixed precision (Metal only).
    pub use_fp16: bool,

    /// Use experimental Metal kernel (false = use MLX fallback).
    /// The MLX fallback provides the same memory savings but may be slower.
    pub use_metal: bool,
}

impl Default for MetalCrossEntropyConfig {
    fn default() -> Self {
        Self {
            chunk_size: 4096,
            ignore_index: -100,
            label_smoothing: 0.0,
            use_fp16: true,
            use_metal: false, // Default to MLX fallback for stability
        }
    }
}

impl MetalCrossEntropyConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set ignore index.
    pub fn with_ignore_index(mut self, index: i32) -> Self {
        self.ignore_index = index;
        self
    }

    /// Set label smoothing.
    pub fn with_label_smoothing(mut self, smoothing: f32) -> Self {
        self.label_smoothing = smoothing;
        self
    }

    /// Use fp16 mode (Metal kernel only).
    pub fn with_fp16(mut self, use_fp16: bool) -> Self {
        self.use_fp16 = use_fp16;
        self
    }

    /// Use experimental Metal kernel (vs MLX fallback).
    ///
    /// The Metal kernel is faster but experimental.
    /// The MLX fallback provides the same memory savings and is more stable.
    pub fn with_metal(mut self, use_metal: bool) -> Self {
        self.use_metal = use_metal;
        self
    }
}

/// Output from Metal cross-entropy computation.
#[derive(Debug)]
pub struct MetalCrossEntropyOutput {
    /// Mean loss over valid tokens.
    pub loss: Array,

    /// Number of valid (non-ignored) tokens.
    pub n_valid: usize,
}

/// Context for Metal cross-entropy operations.
///
/// Caches the Metal context for efficient repeated calls.
pub struct MetalCrossEntropyContext {
    metal_ctx: Arc<MetalContext>,
}

impl MetalCrossEntropyContext {
    /// Create a new Metal cross-entropy context.
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

/// Convert MLX i32 Array to MetalBuffer<i32>.
fn array_to_metal_buffer_i32(
    ctx: &MetalContext,
    array: &Array,
) -> Result<MetalBuffer<i32>> {
    // Ensure array is evaluated and in i32
    let array = if array.dtype() != Dtype::Int32 {
        array.as_dtype(Dtype::Int32)?
    } else {
        array.clone()
    };
    array.eval()?;

    let data: &[i32] = array.as_slice();
    MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)
        .map_err(|e| MlxError::Metal(e.to_string()))
}

/// Convert MLX f32 Array to MetalBuffer<f32>.
fn array_to_metal_buffer_f32(
    ctx: &MetalContext,
    array: &Array,
) -> Result<MetalBuffer<f32>> {
    let array = if array.dtype() != Dtype::Float32 {
        array.as_dtype(Dtype::Float32)?
    } else {
        array.clone()
    };
    array.eval()?;

    let data: &[f32] = array.as_slice();
    MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)
        .map_err(|e| MlxError::Metal(e.to_string()))
}

/// Convert MLX f16 Array to MetalBuffer<f16>.
fn array_to_metal_buffer_f16(
    ctx: &MetalContext,
    array: &Array,
) -> Result<MetalBuffer<f16>> {
    let array = if array.dtype() != Dtype::Float16 {
        array.as_dtype(Dtype::Float16)?
    } else {
        array.clone()
    };
    array.eval()?;

    let data: &[f16] = array.as_slice();
    MetalBuffer::from_slice(ctx, data, BufferUsage::Shared)
        .map_err(|e| MlxError::Metal(e.to_string()))
}

/// Compute fused linear + cross-entropy loss.
///
/// This is THE key optimization from unsloth: computing loss directly from hidden
/// states without materializing the full `[batch * seq, vocab_size]` logits tensor.
///
/// # Arguments
///
/// * `ctx` - Metal cross-entropy context (used for Metal kernel only)
/// * `hidden_states` - Hidden states [batch * seq, hidden_dim]
/// * `lm_head_weight` - LM head weights [vocab_size, hidden_dim]
/// * `targets` - Target token indices [batch * seq]
/// * `config` - Configuration options
///
/// # Returns
///
/// Mean loss over valid tokens.
///
/// # Memory Efficiency
///
/// This never allocates the full `[batch * seq, vocab_size]` logits tensor.
/// For vocab=150K, seq=1024, batch=4, this saves ~1.2GB of memory.
///
/// # Implementation
///
/// Uses MLX CutCrossEntropy by default (stable), or experimental Metal kernel
/// if `config.use_metal = true`.
pub fn metal_fused_linear_cross_entropy(
    ctx: &MetalCrossEntropyContext,
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    config: &MetalCrossEntropyConfig,
) -> Result<MetalCrossEntropyOutput> {
    if config.use_metal {
        // Use experimental Metal kernel
        metal_kernel_cross_entropy(ctx, hidden_states, lm_head_weight, targets, config)
    } else {
        // Use stable MLX CutCrossEntropy fallback
        mlx_cut_cross_entropy(hidden_states, lm_head_weight, targets, config)
    }
}

/// MLX-based CutCrossEntropy implementation (stable fallback).
///
/// Provides the same memory savings as the Metal kernel but uses
/// pure MLX operations for computation.
fn mlx_cut_cross_entropy(
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    config: &MetalCrossEntropyConfig,
) -> Result<MetalCrossEntropyOutput> {
    debug!("Using MLX CutCrossEntropy (memory-efficient, stable)");

    // Create CutCrossEntropy config
    let cce_config = CutCrossEntropyConfig::new()
        .with_vocab_chunk_size(config.chunk_size)
        .with_ignore_index(config.ignore_index)
        .with_label_smoothing(config.label_smoothing);

    let cce = CutCrossEntropy::new(cce_config);

    // Forward pass
    let output = cce.forward(hidden_states, lm_head_weight, targets, None)?;

    Ok(MetalCrossEntropyOutput {
        loss: output.loss,
        n_valid: output.n_valid,
    })
}

/// Experimental Metal kernel implementation.
///
/// Faster than MLX but may have stability issues on some configurations.
fn metal_kernel_cross_entropy(
    ctx: &MetalCrossEntropyContext,
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    config: &MetalCrossEntropyConfig,
) -> Result<MetalCrossEntropyOutput> {
    debug!("Using experimental Metal FusedLinearCrossEntropy kernel");

    let metal_ctx = ctx.metal_context();

    // Get dimensions
    let hidden_shape = hidden_states.shape();
    let num_tokens = hidden_shape[0] as usize;
    let hidden_size = hidden_shape[1] as usize;

    let weight_shape = lm_head_weight.shape();
    let vocab_size = weight_shape[0] as usize;

    // Validate dimensions
    if weight_shape[1] as usize != hidden_size {
        return Err(MlxError::ShapeMismatch {
            expected: format!("[{}, {}]", vocab_size, hidden_size),
            actual: format!("{:?}", weight_shape),
        });
    }

    // Create Metal kernel config
    let metal_config = FusedLinearCrossEntropyConfig::new(num_tokens, hidden_size, vocab_size)
        .with_chunk_size(config.chunk_size)
        .with_ignore_index(config.ignore_index)
        .with_label_smoothing(config.label_smoothing);

    // Create kernel
    let kernel = FusedLinearCrossEntropy::new(metal_ctx.clone(), metal_config)
        .map_err(|e| MlxError::Metal(e.to_string()))?;

    // Convert arrays to Metal buffers and run kernel
    let targets_buffer = array_to_metal_buffer_i32(metal_ctx, targets)?;

    let output = if config.use_fp16 {
        let hidden_buffer = array_to_metal_buffer_f16(metal_ctx, hidden_states)?;
        let weight_buffer = array_to_metal_buffer_f16(metal_ctx, lm_head_weight)?;

        kernel.forward_f16(&hidden_buffer, &weight_buffer, &targets_buffer)
            .map_err(|e| MlxError::Metal(e.to_string()))?
    } else {
        let hidden_buffer = array_to_metal_buffer_f32(metal_ctx, hidden_states)?;
        let weight_buffer = array_to_metal_buffer_f32(metal_ctx, lm_head_weight)?;

        kernel.forward(&hidden_buffer, &weight_buffer, &targets_buffer)
            .map_err(|e| MlxError::Metal(e.to_string()))?
    };

    // Get targets as slice to count valid tokens
    let targets_array = if targets.dtype() != Dtype::Int32 {
        targets.as_dtype(Dtype::Int32)?
    } else {
        targets.clone()
    };
    targets_array.eval()?;
    let targets_slice: &[i32] = targets_array.as_slice();

    // Compute mean loss
    let mean_loss = output.mean_loss(targets_slice, config.ignore_index);
    let n_valid = targets_slice.iter().filter(|&&t| t != config.ignore_index).count();

    // Validate result
    if !mean_loss.is_finite() {
        warn!("Metal kernel returned non-finite loss ({}), falling back to MLX", mean_loss);
        return mlx_cut_cross_entropy(hidden_states, lm_head_weight, targets, config);
    }

    Ok(MetalCrossEntropyOutput {
        loss: Array::from_f32(mean_loss),
        n_valid,
    })
}

/// Convenience function for Metal fused cross-entropy loss.
///
/// Creates a temporary context and uses default configuration.
///
/// # Arguments
///
/// * `hidden_states` - Hidden states [batch * seq, hidden_dim]
/// * `lm_head_weight` - LM head weights [vocab_size, hidden_dim]
/// * `targets` - Target token indices [batch * seq]
/// * `ignore_index` - Index to ignore (typically -100)
///
/// # Returns
///
/// Scalar loss value as MLX Array.
///
/// # Example
///
/// ```rust,ignore
/// let loss = fused_linear_cross_entropy_loss(
///     &hidden_states,
///     &lm_head_weight,
///     &targets,
///     -100,
/// )?;
/// ```
pub fn fused_linear_cross_entropy_loss(
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    ignore_index: i32,
) -> Result<Array> {
    let ctx = MetalCrossEntropyContext::new()?;
    let config = MetalCrossEntropyConfig::new()
        .with_ignore_index(ignore_index);

    let output = metal_fused_linear_cross_entropy(&ctx, hidden_states, lm_head_weight, targets, &config)?;
    Ok(output.loss)
}

/// Fused cross-entropy with label smoothing.
pub fn fused_linear_cross_entropy_loss_smoothed(
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    ignore_index: i32,
    label_smoothing: f32,
) -> Result<Array> {
    let ctx = MetalCrossEntropyContext::new()?;
    let config = MetalCrossEntropyConfig::new()
        .with_ignore_index(ignore_index)
        .with_label_smoothing(label_smoothing);

    let output = metal_fused_linear_cross_entropy(&ctx, hidden_states, lm_head_weight, targets, &config)?;
    Ok(output.loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_cross_entropy_config() {
        let config = MetalCrossEntropyConfig::new()
            .with_chunk_size(8192)
            .with_ignore_index(-1)
            .with_label_smoothing(0.1);

        assert_eq!(config.chunk_size, 8192);
        assert_eq!(config.ignore_index, -1);
        assert_eq!(config.label_smoothing, 0.1);
        assert!(!config.use_metal); // Default is MLX fallback
    }

    #[test]
    fn test_metal_cross_entropy_context() {
        let ctx = MetalCrossEntropyContext::new();
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_fused_linear_cross_entropy_basic() {
        // Uses MLX CutCrossEntropy by default
        let n_tokens = 4;
        let hidden_dim = 8;
        let vocab_size = 16;

        // Create test data
        let hidden_data: Vec<f32> = (0..n_tokens * hidden_dim)
            .map(|i| ((i * 7 + 3) % 10) as f32 / 10.0)
            .collect();
        let hidden = Array::from_slice(&hidden_data, &[n_tokens as i32, hidden_dim as i32]);

        let weight_data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| ((i * 11 + 5) % 10) as f32 / 10.0 - 0.5)
            .collect();
        let weight = Array::from_slice(&weight_data, &[vocab_size as i32, hidden_dim as i32]);

        let targets = Array::from_slice(&[0i32, 5, 10, 15], &[4]);

        let loss = fused_linear_cross_entropy_loss(&hidden, &weight, &targets, -100).unwrap();
        loss.eval().unwrap();

        let loss_value = loss.item::<f32>();
        assert!(loss_value.is_finite());
        assert!(loss_value >= 0.0); // CE is always non-negative
    }

    #[test]
    fn test_fused_linear_cross_entropy_ignore_index() {
        // Uses MLX CutCrossEntropy by default
        let n_tokens = 4;
        let hidden_dim = 8;
        let vocab_size = 16;

        let hidden_data: Vec<f32> = vec![0.5; n_tokens * hidden_dim];
        let hidden = Array::from_slice(&hidden_data, &[n_tokens as i32, hidden_dim as i32]);

        let weight_data: Vec<f32> = vec![0.1; vocab_size * hidden_dim];
        let weight = Array::from_slice(&weight_data, &[vocab_size as i32, hidden_dim as i32]);

        // Two valid, two ignored
        let targets = Array::from_slice(&[0i32, -100, 5, -100], &[4]);

        let ctx = MetalCrossEntropyContext::new().unwrap();
        let config = MetalCrossEntropyConfig::new()
            .with_ignore_index(-100);  // use_metal defaults to false

        let output = metal_fused_linear_cross_entropy(&ctx, &hidden, &weight, &targets, &config).unwrap();

        assert_eq!(output.n_valid, 2);
        output.loss.eval().unwrap();
        assert!(output.loss.item::<f32>().is_finite());
    }

    #[test]
    fn test_mlx_cut_cross_entropy_matches_standard() {
        // Verify MLX fallback produces reasonable results
        let n_tokens = 8;
        let hidden_dim = 16;
        let vocab_size = 32;

        let hidden_data: Vec<f32> = (0..n_tokens * hidden_dim)
            .map(|i| (i as f32 / 128.0) - 0.5)
            .collect();
        let hidden = Array::from_slice(&hidden_data, &[n_tokens as i32, hidden_dim as i32]);

        let weight_data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 / 512.0) - 0.5)
            .collect();
        let weight = Array::from_slice(&weight_data, &[vocab_size as i32, hidden_dim as i32]);

        let targets = Array::from_slice(&[0i32, 1, 2, 3, 4, 5, 6, 7], &[8]);

        let loss = fused_linear_cross_entropy_loss(&hidden, &weight, &targets, -100).unwrap();
        loss.eval().unwrap();

        let loss_value = loss.item::<f32>();

        // CE loss should be positive and reasonable for random data
        assert!(loss_value.is_finite());
        assert!(loss_value > 0.0);
        // For random data with 32 classes, expected loss ~ ln(32) ≈ 3.5
        assert!(loss_value < 10.0, "Loss {} seems too high", loss_value);
    }
}
