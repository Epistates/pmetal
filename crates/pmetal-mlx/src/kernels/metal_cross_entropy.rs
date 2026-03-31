//! Metal-accelerated fused linear + cross-entropy loss.
//!
//! This module provides a key memory optimization: computing cross-entropy
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
//! 2. **MLX CutCrossEntropy**: Uses pure MLX chunked computation (robust)
//!
//! By default, PMetal benchmarks the MLX and Metal implementations for
//! benchmarkable shapes, validates the Metal result against the MLX reference,
//! and persists the winner per device/shape. Set `use_metal=true` to force the
//! Metal kernel, or disable `auto_select_backend` to force the MLX path.
//!
//! # Usage
//!
//! ```rust,ignore
//! use pmetal_mlx::kernels::metal_cross_entropy::fused_linear_cross_entropy_loss;
//!
//! // Compute loss directly from hidden states (auto-selects and caches backend)
//! let loss = fused_linear_cross_entropy_loss(
//!     &hidden_states,  // [batch * seq, hidden_dim]
//!     &lm_head_weight, // [vocab_size, hidden_dim]
//!     &targets,        // [batch * seq]
//!     -100,            // ignore_index
//! )?;
//! ```

use crate::ArrayDtypeExt;
use half::f16;
use pmetal_bridge::compat::{Array, Dtype};
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, OnceLock},
    time::Instant,
};
use tracing::{debug, warn};

use pmetal_metal::{
    FusedLinearCrossEntropy, FusedLinearCrossEntropyConfig,
    buffer::{BufferUsage, MetalBuffer},
    context::{DeviceProperties, DeviceTier, MetalContext},
};

use super::cut_cross_entropy::{CutCrossEntropy, CutCrossEntropyConfig};
use super::persistent_cache::PersistentChoiceCache;
use crate::error::MlxError;

/// Result type for metal cross-entropy operations.
pub type Result<T> = std::result::Result<T, MlxError>;

const DEFAULT_METAL_CROSS_ENTROPY_CHUNK_SIZE: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum CrossEntropyBackendChoice {
    MlxCut,
    MetalFused,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CrossEntropyDispatchKey {
    device_name: String,
    device_tier: &'static str,
    dtype: &'static str,
    num_tokens: i32,
    hidden_size: i32,
    vocab_size: i32,
    chunk_size: usize,
    ignore_index: i32,
    label_smoothing_bits: u32,
    use_fp16: bool,
}

static CROSS_ENTROPY_BACKEND_CACHE: OnceLock<PersistentChoiceCache<CrossEntropyBackendChoice>> =
    OnceLock::new();

fn cross_entropy_backend_cache() -> &'static PersistentChoiceCache<CrossEntropyBackendChoice> {
    CROSS_ENTROPY_BACKEND_CACHE
        .get_or_init(|| PersistentChoiceCache::new("cross_entropy_backends.json"))
}

fn device_tier_key(tier: DeviceTier) -> &'static str {
    match tier {
        DeviceTier::Base => "base",
        DeviceTier::Pro => "pro",
        DeviceTier::Max => "max",
        DeviceTier::Ultra => "ultra",
    }
}

fn dtype_key(dtype: Dtype) -> Option<&'static str> {
    match dtype {
        Dtype::Float16 => Some("f16"),
        Dtype::Float32 => Some("f32"),
        _ => None,
    }
}

impl CrossEntropyDispatchKey {
    fn new(
        props: &DeviceProperties,
        dtype: Dtype,
        num_tokens: i32,
        hidden_size: i32,
        vocab_size: i32,
        chunk_size: usize,
        config: &MetalCrossEntropyConfig,
    ) -> Option<Self> {
        Some(Self {
            device_name: props.name.clone(),
            device_tier: device_tier_key(props.device_tier),
            dtype: dtype_key(dtype)?,
            num_tokens,
            hidden_size,
            vocab_size,
            chunk_size,
            ignore_index: config.ignore_index,
            label_smoothing_bits: config.label_smoothing.to_bits(),
            use_fp16: config.use_fp16,
        })
    }

    fn cache_key(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
            self.device_name,
            self.device_tier,
            self.dtype,
            self.num_tokens,
            self.hidden_size,
            self.vocab_size,
            self.chunk_size,
            self.ignore_index,
            self.label_smoothing_bits,
            self.use_fp16
        )
    }
}

fn cached_cross_entropy_backend(
    key: &CrossEntropyDispatchKey,
) -> Option<CrossEntropyBackendChoice> {
    cross_entropy_backend_cache().get(&key.cache_key())
}

fn cache_cross_entropy_backend(key: CrossEntropyDispatchKey, backend: CrossEntropyBackendChoice) {
    cross_entropy_backend_cache().insert(key.cache_key(), backend);
}

#[cfg(test)]
fn clear_cached_cross_entropy_backends() {
    cross_entropy_backend_cache().clear();
}

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

    /// Automatically benchmark and persist the preferred backend.
    ///
    /// When enabled, PMetal compares the MLX chunked path against the Metal
    /// fused kernel for benchmarkable shapes, validates the Metal result
    /// against the MLX reference, and caches the winner per device/shape.
    pub auto_select_backend: bool,

    /// Force the experimental Metal kernel instead of auto-selection or MLX.
    pub use_metal: bool,
}

impl Default for MetalCrossEntropyConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_METAL_CROSS_ENTROPY_CHUNK_SIZE,
            ignore_index: -100,
            label_smoothing: 0.0,
            use_fp16: true,
            auto_select_backend: true,
            use_metal: false, // Default to auto-selection for stability + speed
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

    /// Enable or disable automatic backend benchmarking and cache selection.
    pub fn with_auto_backend(mut self, auto_select_backend: bool) -> Self {
        self.auto_select_backend = auto_select_backend;
        self
    }

    /// Force the experimental Metal kernel.
    ///
    /// The default path is automatic backend selection with persistent caching.
    pub fn with_metal(mut self, use_metal: bool) -> Self {
        self.use_metal = use_metal;
        self
    }
}

fn resolve_chunk_size(
    ctx: &MetalCrossEntropyContext,
    num_tokens: usize,
    hidden_size: usize,
    vocab_size: usize,
    use_fp16: bool,
    requested_chunk_size: usize,
) -> usize {
    if requested_chunk_size != DEFAULT_METAL_CROSS_ENTROPY_CHUNK_SIZE {
        return requested_chunk_size.max(1).min(vocab_size.max(1));
    }

    let tuning_config = if use_fp16 {
        FusedLinearCrossEntropyConfig::new(num_tokens, hidden_size, vocab_size).with_fp16()
    } else {
        FusedLinearCrossEntropyConfig::new(num_tokens, hidden_size, vocab_size)
    };

    match ctx
        .metal_context()
        .tuner()
        .tune_fused_linear_cross_entropy(ctx.metal_context(), &tuning_config)
    {
        Ok(tuned) => (tuned.chunk_size as usize).max(1).min(vocab_size.max(1)),
        Err(error) => {
            debug!("Cross-entropy tuning unavailable, using default chunk size: {error}");
            requested_chunk_size.max(1).min(vocab_size.max(1))
        }
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
        let metal_ctx = MetalContext::global().map_err(|e| MlxError::Metal(e.to_string()))?;
        Ok(Self { metal_ctx })
    }

    /// Get the Metal context.
    pub fn metal_context(&self) -> &Arc<MetalContext> {
        &self.metal_ctx
    }
}

/// Convert MLX i32 Array to MetalBuffer<i32>.
fn array_to_metal_buffer_i32(ctx: &MetalContext, array: &Array) -> Result<MetalBuffer<i32>> {
    // Cast to f32 for extraction (no i32 bulk-read in bridge), then convert
    let mut f32_arr = array.as_dtype(Dtype::Float32.as_i32());
    f32_arr.eval();
    let n = f32_arr.size();
    let data_f32 = f32_arr.to_f32_vec(n).unwrap_or_default();
    let data_i32: Vec<i32> = data_f32.into_iter().map(|v| v as i32).collect();
    MetalBuffer::from_slice(ctx, &data_i32, BufferUsage::Shared)
        .map_err(|e| MlxError::Metal(e.to_string()))
}

/// Convert MLX f32 Array to MetalBuffer<f32>.
fn array_to_metal_buffer_f32(ctx: &MetalContext, array: &Array) -> Result<MetalBuffer<f32>> {
    let mut f32_arr = if array.dtype() != Dtype::Float32 {
        array.as_dtype(Dtype::Float32.as_i32())
    } else {
        array.clone()
    };
    f32_arr.eval();
    let n = f32_arr.size();
    let data = f32_arr.to_f32_vec(n).unwrap_or_default();
    MetalBuffer::from_slice(ctx, &data, BufferUsage::Shared)
        .map_err(|e| MlxError::Metal(e.to_string()))
}

/// Convert MLX f16 Array to MetalBuffer<f16>.
fn array_to_metal_buffer_f16(ctx: &MetalContext, array: &Array) -> Result<MetalBuffer<f16>> {
    // Upcast to f32 for extraction, then downcast each element to f16
    let mut f32_arr = if array.dtype() != Dtype::Float32 {
        array.as_dtype(Dtype::Float32.as_i32())
    } else {
        array.clone()
    };
    f32_arr.eval();
    let n = f32_arr.size();
    let data_f32 = f32_arr.to_f32_vec(n).unwrap_or_default();
    let data_f16: Vec<f16> = data_f32.into_iter().map(f16::from_f32).collect();
    MetalBuffer::from_slice(ctx, &data_f16, BufferUsage::Shared)
        .map_err(|e| MlxError::Metal(e.to_string()))
}

/// Compute fused linear + cross-entropy loss.
///
/// This is a key memory optimization: computing loss directly from hidden
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
/// Uses automatic MLX-vs-Metal backend selection by default, or the forced
/// Metal path if `config.use_metal = true`.
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
    } else if config.auto_select_backend {
        auto_selected_cross_entropy(ctx, hidden_states, lm_head_weight, targets, config)
    } else {
        // Use stable MLX CutCrossEntropy fallback
        mlx_cut_cross_entropy(ctx, hidden_states, lm_head_weight, targets, config)
    }
}

fn benchmarkable_cross_entropy_dispatch_key(
    ctx: &MetalCrossEntropyContext,
    hidden_states: &Array,
    lm_head_weight: &Array,
    config: &MetalCrossEntropyConfig,
) -> Option<CrossEntropyDispatchKey> {
    let hidden_shape = hidden_states.shape();
    let weight_shape = lm_head_weight.shape();
    if hidden_shape.len() != 2 || weight_shape.len() != 2 {
        return None;
    }

    if hidden_shape[1] != weight_shape[1] {
        return None;
    }

    let hidden_dtype = hidden_states.dtype();
    if hidden_dtype != lm_head_weight.dtype() {
        return None;
    }

    CrossEntropyDispatchKey::new(
        ctx.metal_context().properties(),
        hidden_dtype,
        hidden_shape[0],
        hidden_shape[1],
        weight_shape[0],
        resolve_chunk_size(
            ctx,
            hidden_shape[0] as usize,
            hidden_shape[1] as usize,
            weight_shape[0] as usize,
            config.use_fp16,
            config.chunk_size,
        ),
        config,
    )
}

fn auto_selected_cross_entropy(
    ctx: &MetalCrossEntropyContext,
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    config: &MetalCrossEntropyConfig,
) -> Result<MetalCrossEntropyOutput> {
    let Some(dispatch_key) =
        benchmarkable_cross_entropy_dispatch_key(ctx, hidden_states, lm_head_weight, config)
    else {
        return mlx_cut_cross_entropy(ctx, hidden_states, lm_head_weight, targets, config);
    };

    if let Some(backend) = cached_cross_entropy_backend(&dispatch_key) {
        return execute_cross_entropy_backend(
            backend,
            ctx,
            hidden_states,
            lm_head_weight,
            targets,
            config,
        )
        .or_else(|error| {
            debug!(
                "Cached {:?} cross-entropy backend failed, falling back to MLX: {error}",
                backend
            );
            cache_cross_entropy_backend(dispatch_key.clone(), CrossEntropyBackendChoice::MlxCut);
            mlx_cut_cross_entropy(ctx, hidden_states, lm_head_weight, targets, config)
        });
    }

    let (backend, output) =
        benchmark_cross_entropy_backends(ctx, hidden_states, lm_head_weight, targets, config)?;
    cache_cross_entropy_backend(dispatch_key, backend);
    Ok(output)
}

fn execute_cross_entropy_backend(
    backend: CrossEntropyBackendChoice,
    ctx: &MetalCrossEntropyContext,
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    config: &MetalCrossEntropyConfig,
) -> Result<MetalCrossEntropyOutput> {
    match backend {
        CrossEntropyBackendChoice::MlxCut => {
            mlx_cut_cross_entropy(ctx, hidden_states, lm_head_weight, targets, config)
        }
        CrossEntropyBackendChoice::MetalFused => run_metal_kernel_cross_entropy_strict(
            ctx,
            hidden_states,
            lm_head_weight,
            targets,
            config,
        ),
    }
}

fn benchmark_cross_entropy_backends(
    ctx: &MetalCrossEntropyContext,
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    config: &MetalCrossEntropyConfig,
) -> Result<(CrossEntropyBackendChoice, MetalCrossEntropyOutput)> {
    let mlx_start = Instant::now();
    let mlx_output = mlx_cut_cross_entropy(ctx, hidden_states, lm_head_weight, targets, config)?;
    let mlx_loss_eval = mlx_output.loss.clone();
    mlx_loss_eval.eval();
    let mlx_loss = mlx_loss_eval.item_f32();
    let mlx_elapsed = mlx_start.elapsed();

    let metal_start = Instant::now();
    let metal_output = match run_metal_kernel_cross_entropy_strict(
        ctx,
        hidden_states,
        lm_head_weight,
        targets,
        config,
    ) {
        Ok(output) => {
            let loss_eval = output.loss.clone();
            loss_eval.eval();
            Some(output)
        }
        Err(error) => {
            debug!("Metal cross-entropy benchmark failed, using MLX CutCrossEntropy: {error}");
            None
        }
    };
    let metal_elapsed = metal_start.elapsed();

    if let Some(metal_output) = metal_output {
        let metal_loss_eval = metal_output.loss.clone();
        metal_loss_eval.eval();
        let metal_loss = metal_loss_eval.item_f32();
        if losses_match(mlx_loss, metal_loss, config.use_fp16)
            && metal_output.n_valid == mlx_output.n_valid
            && metal_elapsed < mlx_elapsed
        {
            debug!(
                "Selected Metal fused cross-entropy ({:?} vs {:?})",
                metal_elapsed, mlx_elapsed
            );
            return Ok((CrossEntropyBackendChoice::MetalFused, metal_output));
        }

        if metal_output.n_valid != mlx_output.n_valid {
            debug!(
                "Rejecting Metal fused cross-entropy due to n_valid mismatch (metal={}, mlx={})",
                metal_output.n_valid, mlx_output.n_valid
            );
        } else if !losses_match(mlx_loss, metal_loss, config.use_fp16) {
            debug!(
                "Rejecting Metal fused cross-entropy due to loss mismatch (metal={metal_loss:.6}, mlx={mlx_loss:.6})"
            );
        }
    }

    debug!(
        "Selected MLX CutCrossEntropy ({:?} vs {:?})",
        mlx_elapsed, metal_elapsed
    );
    Ok((CrossEntropyBackendChoice::MlxCut, mlx_output))
}

fn losses_match(reference: f32, candidate: f32, use_fp16: bool) -> bool {
    let abs_diff = (reference - candidate).abs();
    let abs_tol = if use_fp16 { 2e-2 } else { 5e-3 };
    let rel_tol = if use_fp16 { 5e-3 } else { 1e-3 };
    abs_diff <= abs_tol || abs_diff <= rel_tol * reference.abs().max(candidate.abs()).max(1.0)
}

/// MLX-based CutCrossEntropy implementation (stable fallback).
///
/// Provides the same memory savings as the Metal kernel but uses
/// pure MLX operations for computation.
fn mlx_cut_cross_entropy(
    ctx: &MetalCrossEntropyContext,
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    config: &MetalCrossEntropyConfig,
) -> Result<MetalCrossEntropyOutput> {
    debug!("Using MLX CutCrossEntropy (memory-efficient, stable)");

    let hidden_shape = hidden_states.shape();
    let weight_shape = lm_head_weight.shape();
    let chunk_size = resolve_chunk_size(
        ctx,
        hidden_shape[0] as usize,
        hidden_shape[1] as usize,
        weight_shape[0] as usize,
        config.use_fp16,
        config.chunk_size,
    );

    // Create CutCrossEntropy config
    let cce_config = CutCrossEntropyConfig::new()
        .with_vocab_chunk_size(chunk_size)
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

    match run_metal_kernel_cross_entropy_strict(ctx, hidden_states, lm_head_weight, targets, config)
    {
        Ok(output) => Ok(output),
        Err(error) => {
            warn!("Metal fused cross-entropy failed, falling back to MLX: {error}");
            mlx_cut_cross_entropy(ctx, hidden_states, lm_head_weight, targets, config)
        }
    }
}

fn run_metal_kernel_cross_entropy_strict(
    ctx: &MetalCrossEntropyContext,
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    config: &MetalCrossEntropyConfig,
) -> Result<MetalCrossEntropyOutput> {
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
    let effective_chunk_size = resolve_chunk_size(
        ctx,
        num_tokens,
        hidden_size,
        vocab_size,
        config.use_fp16,
        config.chunk_size,
    );
    let metal_config = FusedLinearCrossEntropyConfig::new(num_tokens, hidden_size, vocab_size)
        .with_chunk_size(effective_chunk_size)
        .with_ignore_index(config.ignore_index)
        .with_label_smoothing(config.label_smoothing);
    let metal_config = if config.use_fp16 {
        metal_config.with_fp16()
    } else {
        metal_config
    };

    // Create kernel
    let kernel = FusedLinearCrossEntropy::new(metal_ctx.clone(), metal_config)
        .map_err(|e| MlxError::Metal(e.to_string()))?;

    // Convert arrays to Metal buffers and run kernel
    let targets_buffer = array_to_metal_buffer_i32(metal_ctx, targets)?;

    let output = if config.use_fp16 {
        let hidden_buffer = array_to_metal_buffer_f16(metal_ctx, hidden_states)?;
        let weight_buffer = array_to_metal_buffer_f16(metal_ctx, lm_head_weight)?;

        kernel
            .forward_f16(&hidden_buffer, &weight_buffer, &targets_buffer)
            .map_err(|e| MlxError::Metal(e.to_string()))?
    } else {
        let hidden_buffer = array_to_metal_buffer_f32(metal_ctx, hidden_states)?;
        let weight_buffer = array_to_metal_buffer_f32(metal_ctx, lm_head_weight)?;

        kernel
            .forward(&hidden_buffer, &weight_buffer, &targets_buffer)
            .map_err(|e| MlxError::Metal(e.to_string()))?
    };

    // Get targets as Vec<i32> to count valid tokens and pass to mean_loss
    let mut targets_f32 = targets.as_dtype(Dtype::Float32.as_i32());
    targets_f32.eval();
    let n_targets = targets_f32.size();
    let targets_f32_vec = targets_f32.to_f32_vec(n_targets).unwrap_or_default();
    let targets_i32_vec: Vec<i32> = targets_f32_vec.into_iter().map(|v| v as i32).collect();
    let targets_slice: &[i32] = &targets_i32_vec;

    // Compute mean loss
    let mean_loss = output.mean_loss(targets_slice, config.ignore_index);
    let n_valid = targets_slice
        .iter()
        .filter(|&&t| t != config.ignore_index)
        .count();

    // Validate result
    if !mean_loss.is_finite() {
        warn!(
            "Metal kernel returned non-finite loss ({}), falling back to MLX",
            mean_loss
        );
        return Err(MlxError::Metal(format!(
            "Metal kernel returned non-finite loss ({mean_loss})"
        )));
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
    let config = MetalCrossEntropyConfig::new().with_ignore_index(ignore_index);

    let output =
        metal_fused_linear_cross_entropy(&ctx, hidden_states, lm_head_weight, targets, &config)?;
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

    let output =
        metal_fused_linear_cross_entropy(&ctx, hidden_states, lm_head_weight, targets, &config)?;
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
        assert!(config.auto_select_backend);
        assert!(!config.use_metal); // Default is auto-selection, not forced Metal
    }

    #[test]
    fn test_metal_cross_entropy_context() {
        let ctx = MetalCrossEntropyContext::new();
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_fused_linear_cross_entropy_basic() {
        // Uses MLX CutCrossEntropy by default
        let n_tokens: i32 = 4;
        let hidden_dim: i32 = 8;
        let vocab_size: i32 = 16;

        let hidden_data: Vec<f32> = (0..(n_tokens * hidden_dim) as usize)
            .map(|i| ((i * 7 + 3) % 10) as f32 / 10.0)
            .collect();
        let hidden = Array::from_f32_slice(&hidden_data, &[n_tokens, hidden_dim]);

        let weight_data: Vec<f32> = (0..(vocab_size * hidden_dim) as usize)
            .map(|i| ((i * 11 + 5) % 10) as f32 / 10.0 - 0.5)
            .collect();
        let weight = Array::from_f32_slice(&weight_data, &[vocab_size, hidden_dim]);

        let targets = Array::from_i32_slice(&[0i32, 5, 10, 15]).reshape(&[4]);

        let loss = fused_linear_cross_entropy_loss(&hidden, &weight, &targets, -100).unwrap();
        let mut loss_eval = loss.clone();
        loss_eval.eval();

        let loss_value = loss_eval.item_f32();
        assert!(loss_value.is_finite());
        assert!(loss_value >= 0.0);
    }

    #[test]
    fn test_fused_linear_cross_entropy_ignore_index() {
        let n_tokens: i32 = 4;
        let hidden_dim: i32 = 8;
        let vocab_size: i32 = 16;

        let hidden_data: Vec<f32> = vec![0.5; (n_tokens * hidden_dim) as usize];
        let hidden = Array::from_f32_slice(&hidden_data, &[n_tokens, hidden_dim]);

        let weight_data: Vec<f32> = vec![0.1; (vocab_size * hidden_dim) as usize];
        let weight = Array::from_f32_slice(&weight_data, &[vocab_size, hidden_dim]);

        let targets = Array::from_i32_slice(&[0i32, -100, 5, -100]).reshape(&[4]);

        let ctx = MetalCrossEntropyContext::new().unwrap();
        let config = MetalCrossEntropyConfig::new().with_ignore_index(-100);

        let output =
            metal_fused_linear_cross_entropy(&ctx, &hidden, &weight, &targets, &config).unwrap();

        assert_eq!(output.n_valid, 2);
        let mut loss_eval = output.loss.clone();
        loss_eval.eval();
        assert!(loss_eval.item_f32().is_finite());
    }

    #[test]
    fn test_cross_entropy_dispatch_key_roundtrip() {
        clear_cached_cross_entropy_backends();

        let key = CrossEntropyDispatchKey {
            device_name: "Apple M4 Max".to_string(),
            device_tier: "max",
            dtype: "f16",
            num_tokens: 128,
            hidden_size: 4096,
            vocab_size: 32000,
            chunk_size: 4096,
            ignore_index: -100,
            label_smoothing_bits: 0.0f32.to_bits(),
            use_fp16: true,
        };

        assert_eq!(cached_cross_entropy_backend(&key), None);
        cache_cross_entropy_backend(key.clone(), CrossEntropyBackendChoice::MetalFused);
        assert_eq!(
            cached_cross_entropy_backend(&key),
            Some(CrossEntropyBackendChoice::MetalFused)
        );

        clear_cached_cross_entropy_backends();
    }

    #[test]
    fn test_auto_backend_fused_linear_cross_entropy_basic() {
        clear_cached_cross_entropy_backends();

        let n_tokens = 4;
        let hidden_dim = 8;
        let vocab_size = 16;

        let hidden_data: Vec<f32> = (0..n_tokens * hidden_dim)
            .map(|i| ((i * 7 + 3) % 10) as f32 / 10.0)
            .collect();
        let hidden = Array::from_slice(&hidden_data, &[n_tokens as i32, hidden_dim as i32]);

        let weight_data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| ((i * 11 + 5) % 10) as f32 / 10.0 - 0.5)
            .collect();
        let weight = Array::from_slice(&weight_data, &[vocab_size as i32, hidden_dim as i32]);

        let targets = Array::from_slice(&[0i32, 5, 10, 15], &[4]);
        let ctx = MetalCrossEntropyContext::new().unwrap();
        let config = MetalCrossEntropyConfig::new().with_ignore_index(-100);

        let mut output =
            metal_fused_linear_cross_entropy(&ctx, &hidden, &weight, &targets, &config).unwrap();
        output.loss.eval();

        let loss_value = output.loss.item::<f32>();
        assert!(loss_value.is_finite());
        assert!(loss_value >= 0.0);
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

        let mut loss = fused_linear_cross_entropy_loss(&hidden, &weight, &targets, -100).unwrap();
        loss.eval();

        let loss_value = loss.item::<f32>();

        // CE loss should be positive and reasonable for random data
        assert!(loss_value.is_finite());
        assert!(loss_value > 0.0);
        // For random data with 32 classes, expected loss ~ ln(32) ≈ 3.5
        assert!(loss_value < 10.0, "Loss {} seems too high", loss_value);
    }
}
