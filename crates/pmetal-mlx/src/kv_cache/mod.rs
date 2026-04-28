//! Key-Value cache for efficient autoregressive inference.
//!
//! KV caching stores previously computed key and value tensors during generation,
//! avoiding redundant computation of attention for past tokens.
//!
//! ## Memory-Compute Tradeoff
//!
//! - **Without KV cache**: O(n²) attention computation per token
//! - **With KV cache**: O(n) attention computation per token
//!
//! For a sequence of length n, KV caching reduces total generation complexity
//! from O(n³) to O(n²), providing significant speedups for long sequences.
//!
//! ## Tensor Format (SOTA Performance)
//!
//! Keys/values are stored in **attention format** `[B, heads, seq, head_dim]`:
//! - Axis 0: Batch dimension
//! - Axis 1: Number of KV heads
//! - Axis 2: Sequence length (grows during generation)
//! - Axis 3: Head dimension
//!
//! This matches the mlx_lm implementation and eliminates transpose overhead
//! during cached generation. The sequence dimension is axis 2.
//!
//! ## Supported Modes
//!
//! - **Standard KV cache**: Stores all past keys/values (best for short sequences)
//! - **Sliding window cache**: Fixed-size window (constant memory, for long sequences)
//!
//! ## Usage
//!
//! ```ignore
//! let mut cache = KVCache::new(num_layers, max_len, num_kv_heads, head_dim);
//! for step in 0..generation_length {
//!     // Pass keys/values in [B, heads, seq, head_dim] format
//!     let (keys, values) = cache.update_and_fetch(layer_idx, new_keys, new_values)?;
//!     // Use keys, values directly in attention (no transpose needed)
//! }
//! ```

mod fused_batch;
mod mamba;
mod paged;
mod quantized;
mod rotating;
mod standard;
#[cfg(test)]
mod tests;
mod turboquant;

pub use fused_batch::*;
pub use mamba::*;
pub use paged::*;
pub use quantized::*;
pub use rotating::*;
pub use standard::*;
pub use turboquant::*;

use pmetal_bridge::compat::Dtype;

/// Configuration for KV cache.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Number of key-value heads (for GQA/MQA).
    pub num_kv_heads: usize,
    /// Key dimension per head.
    pub head_dim: usize,
    /// Value dimension per head. Defaults to `head_dim`.
    pub value_head_dim: usize,
    /// Data type for cached tensors.
    pub dtype: Dtype,
    /// Cache mode.
    pub mode: CacheMode,
    /// Whether to eagerly pre-allocate the full context window upfront.
    /// When true, allocates memory for max_seq_len tokens at creation time.
    /// This provides predictable memory usage but uses more memory initially.
    /// Default: false (lazy allocation in 256-token chunks).
    pub eager_allocate: bool,
    /// Batch size for eager allocation (only used when eager_allocate=true).
    /// Default: 1
    pub eager_batch_size: usize,
}

/// KV cache mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Standard cache - stores all past tokens.
    Standard,
    /// Sliding window cache with fixed size.
    SlidingWindow {
        /// Maximum number of past tokens to keep in cache.
        window_size: usize,
    },
    /// Rotating cache - circular buffer with fixed max size (MLX-LM parity).
    /// More memory-efficient than sliding window for long sequences.
    Rotating {
        /// Maximum number of tokens to keep.
        max_size: usize,
        /// Number of initial tokens to always keep (typically prompt tokens).
        keep: usize,
    },
    /// Quantized cache - stores K/V in lower precision (MLX-LM parity).
    /// Reduces memory by 2-8x depending on bits.
    Quantized {
        /// Number of bits for quantization (2, 4, or 8).
        bits: u8,
        /// Group size for quantization (default: 64).
        group_size: usize,
    },
    /// Asymmetric quantized cache - different precision for K and V.
    /// K is more sensitive to quantization than V (community benchmarks confirm
    /// <0.4% PPL diff at q8_0, V tolerates more aggressive quantization).
    /// Recommended: K@q8, V@q4 for best quality/memory tradeoff.
    AsymmetricQuantized {
        /// Number of bits for key quantization (2, 4, or 8).
        key_bits: u8,
        /// Number of bits for value quantization (2, 4, or 8).
        value_bits: u8,
        /// Group size for quantization (default: 64).
        group_size: usize,
    },
    /// TurboQuant cache - random rotation + Lloyd-Max + QJL residual keys.
    TurboQuant {
        /// TurboQuant K/V configuration.
        config: TurboQuantConfig,
    },
}

impl Default for CacheMode {
    fn default() -> Self {
        Self::Standard
    }
}

impl CacheMode {
    /// Human-readable description of the cache mode.
    pub fn describe(&self) -> String {
        match self {
            CacheMode::Standard => "fp16".to_string(),
            CacheMode::SlidingWindow { window_size } => format!("sliding-{window_size}"),
            CacheMode::Rotating { max_size, .. } => format!("rotating-{max_size}"),
            CacheMode::Quantized { bits, group_size } => format!("q{bits}_0 (group={group_size})"),
            CacheMode::AsymmetricQuantized {
                key_bits,
                value_bits,
                group_size,
            } => {
                format!("K@q{key_bits},V@q{value_bits} (group={group_size})")
            }
            CacheMode::TurboQuant { config } => format!(
                "turboquant K@{},V@{}",
                describe_turboquant_tensor(config.keys),
                describe_turboquant_tensor(config.values)
            ),
        }
    }
}

fn describe_turboquant_tensor(config: TurboQuantTensorConfig) -> String {
    match config {
        TurboQuantTensorConfig::Uniform { bits } => format!("{bits}b"),
        TurboQuantTensorConfig::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        } => format!("{regular_bits}/{outlier_bits}b (+{outlier_count} outliers)"),
    }
}

/// Returns whether a quantized KV cache `group_size` cleanly divides both key
/// and value head dimensions.
pub fn group_size_supported_for_dims(
    key_head_dim: usize,
    value_head_dim: usize,
    group_size: usize,
) -> bool {
    group_size > 0
        && (key_head_dim == 0 || key_head_dim % group_size == 0)
        && (value_head_dim == 0 || value_head_dim % group_size == 0)
}

/// Returns the best supported quantization group size for a K/V head-dimension
/// pair, preferring larger group sizes first.
pub fn compatible_group_size_for_dims(
    key_head_dim: usize,
    value_head_dim: usize,
    preferred: usize,
) -> usize {
    if group_size_supported_for_dims(key_head_dim, value_head_dim, preferred) {
        return preferred;
    }
    for candidate in [128, 64, 32, 16, 8, 4, 2, 1] {
        if group_size_supported_for_dims(key_head_dim, value_head_dim, candidate) {
            return candidate;
        }
    }
    1
}

/// Sanitizes a TurboQuant tensor config for a specific head dimension.
///
/// Mixed-bit configs clamp their outlier count into the valid per-head range.
/// Degenerate dimensions fall back to a uniform config using the higher bit
/// width to avoid invalid mixed schedules.
pub fn sanitize_turboquant_tensor_config(
    head_dim: usize,
    config: TurboQuantTensorConfig,
) -> TurboQuantTensorConfig {
    match config {
        TurboQuantTensorConfig::Uniform { .. } => config,
        TurboQuantTensorConfig::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        } => {
            if head_dim <= 1 {
                TurboQuantTensorConfig::uniform(outlier_bits.max(regular_bits))
            } else {
                TurboQuantTensorConfig::mixed(
                    regular_bits,
                    outlier_bits,
                    outlier_count.clamp(1, head_dim - 1),
                )
            }
        }
    }
}

/// Sanitizes a TurboQuant K/V config for concrete key and value head
/// dimensions.
pub fn sanitize_turboquant_config(
    key_head_dim: usize,
    value_head_dim: usize,
    config: TurboQuantConfig,
) -> TurboQuantConfig {
    TurboQuantConfig {
        keys: sanitize_turboquant_tensor_config(key_head_dim, config.keys),
        values: sanitize_turboquant_tensor_config(value_head_dim, config.values),
        recent_window: config.recent_window,
        qjl: config.qjl,
        skiplist_threshold: config.skiplist_threshold,
    }
}

/// Sanitizes a cache mode for a concrete pair of key/value head dimensions.
///
/// This normalizes quantized group sizes and TurboQuant mixed-bit schedules so
/// all cache construction sites use the same rules.
pub fn sanitize_cache_mode_for_dims(
    key_head_dim: usize,
    value_head_dim: usize,
    mode: CacheMode,
) -> CacheMode {
    match mode {
        CacheMode::Quantized { bits, group_size }
            if !group_size_supported_for_dims(key_head_dim, value_head_dim, group_size) =>
        {
            CacheMode::Quantized {
                bits,
                group_size: compatible_group_size_for_dims(
                    key_head_dim,
                    value_head_dim,
                    group_size,
                ),
            }
        }
        CacheMode::AsymmetricQuantized {
            key_bits,
            value_bits,
            group_size,
        } if !group_size_supported_for_dims(key_head_dim, value_head_dim, group_size) => {
            CacheMode::AsymmetricQuantized {
                key_bits,
                value_bits,
                group_size: compatible_group_size_for_dims(
                    key_head_dim,
                    value_head_dim,
                    group_size,
                ),
            }
        }
        CacheMode::TurboQuant { config } => CacheMode::TurboQuant {
            config: sanitize_turboquant_config(key_head_dim, value_head_dim, config),
        },
        other => other,
    }
}

/// Sanitizes a cache mode using a base KV cache configuration.
pub fn sanitize_cache_mode_for_config(config: &KVCacheConfig, mode: CacheMode) -> CacheMode {
    sanitize_cache_mode_for_dims(config.head_dim, config.value_head_dim, mode)
}

impl KVCacheConfig {
    /// Create a new KV cache configuration.
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            num_layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
            value_head_dim: head_dim,
            dtype: Dtype::Float32,
            mode: CacheMode::Standard,
            eager_allocate: false,
            eager_batch_size: 1,
        }
    }

    /// Set the data type for cached tensors.
    pub fn with_dtype(mut self, dtype: Dtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the cache mode.
    pub fn with_mode(mut self, mode: CacheMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set a distinct value head dimension.
    pub fn with_value_head_dim(mut self, value_head_dim: usize) -> Self {
        self.value_head_dim = value_head_dim;
        self
    }

    /// Enable sliding window mode.
    pub fn with_sliding_window(mut self, window_size: usize) -> Self {
        self.mode = CacheMode::SlidingWindow { window_size };
        self
    }

    /// Enable rotating cache mode (MLX-LM style).
    ///
    /// The rotating cache is a circular buffer that overwrites oldest entries
    /// when full, while optionally preserving `keep` initial tokens.
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of tokens to store
    /// * `keep` - Number of initial tokens to always preserve (0 for none)
    pub fn with_rotating(mut self, max_size: usize, keep: usize) -> Self {
        self.mode = CacheMode::Rotating { max_size, keep };
        self
    }

    /// Enable quantized cache mode (MLX-LM style).
    ///
    /// Stores keys/values in lower precision to reduce memory usage.
    /// - 8-bit: ~2x memory reduction
    /// - 4-bit: ~4x memory reduction
    /// - 2-bit: ~8x memory reduction
    ///
    /// # Arguments
    /// * `bits` - Number of bits (2, 4, or 8)
    /// * `group_size` - Group size for quantization (default: 64)
    pub fn with_quantized(mut self, bits: u8, group_size: usize) -> Self {
        self.mode = CacheMode::Quantized { bits, group_size };
        self
    }

    /// Enable asymmetric quantized cache mode.
    ///
    /// K is more sensitive to quantization than V. Using higher precision
    /// for keys and lower for values gives nearly the memory savings of
    /// aggressive quantization with quality closer to conservative quantization.
    ///
    /// # Arguments
    /// * `key_bits` - Number of bits for keys (2, 4, or 8)
    /// * `value_bits` - Number of bits for values (2, 4, or 8)
    /// * `group_size` - Group size for quantization (default: 64)
    pub fn with_asymmetric_quantized(
        mut self,
        key_bits: u8,
        value_bits: u8,
        group_size: usize,
    ) -> Self {
        self.mode = CacheMode::AsymmetricQuantized {
            key_bits,
            value_bits,
            group_size,
        };
        self
    }

    /// Enable TurboQuant KV cache mode.
    ///
    /// Keys use TurboQuant's unbiased inner-product quantizer and values use
    /// the MSE-optimized quantizer.
    pub fn with_turboquant(mut self, key_bits: u8, value_bits: u8) -> Self {
        self.mode = CacheMode::TurboQuant {
            config: TurboQuantConfig::uniform(key_bits, value_bits),
        };
        self
    }

    /// Enable TurboQuant with an explicit configuration.
    pub fn with_turboquant_config(mut self, config: TurboQuantConfig) -> Self {
        self.mode = CacheMode::TurboQuant { config };
        self
    }

    /// Enable mixed-bit TurboQuant KV cache mode.
    pub fn with_turboquant_mixed(
        mut self,
        key_regular_bits: u8,
        key_outlier_bits: u8,
        key_outlier_count: usize,
        value_regular_bits: u8,
        value_outlier_bits: u8,
        value_outlier_count: usize,
    ) -> Self {
        self.mode = CacheMode::TurboQuant {
            config: TurboQuantConfig::mixed(
                key_regular_bits,
                key_outlier_bits,
                key_outlier_count,
                value_regular_bits,
                value_outlier_bits,
                value_outlier_count,
            ),
        };
        self
    }

    /// Enable eager pre-allocation of the full context window.
    ///
    /// When enabled, the KV cache will allocate memory for the full `max_seq_len`
    /// at creation time rather than growing dynamically. This provides:
    /// - **Predictable memory usage**: Know exactly how much memory is needed upfront
    /// - **No allocation during generation**: Faster token generation
    /// - **Memory fragmentation prevention**: Single contiguous allocation
    ///
    /// Trade-off: Uses more memory initially even for short sequences.
    ///
    /// # Arguments
    /// * `batch_size` - Batch size to pre-allocate for (typically 1 for inference)
    ///
    /// # Example
    /// ```ignore
    /// let config = KVCacheConfig::new(32, 4096, 8, 128)
    ///     .with_eager_allocate(1);  // Pre-allocate for batch_size=1
    /// let cache = KVCache::new(config);  // ~1GB allocated immediately
    /// ```
    pub fn with_eager_allocate(mut self, batch_size: usize) -> Self {
        self.eager_allocate = true;
        self.eager_batch_size = batch_size;
        self
    }

    /// Calculate the memory footprint for this configuration in bytes.
    ///
    /// Useful for understanding memory requirements before allocation.
    pub fn memory_footprint(&self) -> usize {
        let key_dim = self.head_dim;
        let value_dim = self.value_head_dim;
        match self.mode {
            CacheMode::Quantized { bits, group_size } => {
                let el_per_int = 32 / bits as usize;
                let k_packed = (key_dim + el_per_int - 1) / el_per_int;
                let v_packed = (value_dim + el_per_int - 1) / el_per_int;
                let k_groups = (key_dim + group_size - 1) / group_size;
                let v_groups = (value_dim + group_size - 1) / group_size;
                // packed u32 data + f16 scale + f16 bias per group
                let k_per_token_bytes = (k_packed * 4 + k_groups * 4) * self.num_kv_heads;
                let v_per_token_bytes = (v_packed * 4 + v_groups * 4) * self.num_kv_heads;
                (k_per_token_bytes + v_per_token_bytes) * self.max_seq_len * self.num_layers
            }
            CacheMode::AsymmetricQuantized {
                key_bits,
                value_bits,
                group_size,
            } => {
                let k_el = 32 / key_bits as usize;
                let v_el = 32 / value_bits as usize;
                let k_packed = (key_dim + k_el - 1) / k_el;
                let v_packed = (value_dim + v_el - 1) / v_el;
                let k_groups = (key_dim + group_size - 1) / group_size;
                let v_groups = (value_dim + group_size - 1) / group_size;
                let k_per_token = (k_packed * 4 + k_groups * 4) * self.num_kv_heads;
                let v_per_token = (v_packed * 4 + v_groups * 4) * self.num_kv_heads;
                (k_per_token + v_per_token) * self.max_seq_len * self.num_layers
            }
            CacheMode::TurboQuant { config } => {
                let rows_per_token = self.num_kv_heads;
                let per_token_bytes = rows_per_token
                    * (turboquant_key_row_bytes(config.keys, key_dim)
                        + turboquant_value_row_bytes(config.values, value_dim));
                per_token_bytes * self.max_seq_len * self.num_layers
            }
            _ => {
                let bytes_per_element = match self.dtype {
                    Dtype::Float32 => 4,
                    Dtype::Float16 | Dtype::Bfloat16 => 2,
                    _ => 4,
                };
                self.eager_batch_size.max(1)
                    * self.num_kv_heads
                    * self.max_seq_len
                    * (key_dim + value_dim)
                    * bytes_per_element
                    * self.num_layers
            }
        }
    }

    /// Format the memory footprint as a human-readable string.
    pub fn memory_footprint_human(&self) -> String {
        let bytes = self.memory_footprint();
        if bytes >= 1024 * 1024 * 1024 {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        } else if bytes >= 1024 * 1024 {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        } else if bytes >= 1024 {
            format!("{:.2} KB", bytes as f64 / 1024.0)
        } else {
            format!("{} bytes", bytes)
        }
    }
}

fn turboquant_key_row_bytes(config: TurboQuantTensorConfig, head_dim: usize) -> usize {
    match config {
        TurboQuantTensorConfig::Uniform { bits } => {
            let mse_bits = usize::from(bits.saturating_sub(1));
            (head_dim * mse_bits).div_ceil(8)
                + head_dim.div_ceil(8)
                + (std::mem::size_of::<f32>() * 2)
        }
        TurboQuantTensorConfig::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        } => {
            let regular_dim = head_dim - outlier_count;
            (regular_dim * usize::from(regular_bits.saturating_sub(1))).div_ceil(8)
                + regular_dim.div_ceil(8)
                + (outlier_count * usize::from(outlier_bits.saturating_sub(1))).div_ceil(8)
                + outlier_count.div_ceil(8)
                + head_dim.div_ceil(8)
                + (std::mem::size_of::<f32>() * 4)
        }
    }
}

fn turboquant_value_row_bytes(config: TurboQuantTensorConfig, head_dim: usize) -> usize {
    match config {
        TurboQuantTensorConfig::Uniform { bits } => {
            (head_dim * usize::from(bits)).div_ceil(8) + std::mem::size_of::<f32>()
        }
        TurboQuantTensorConfig::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        } => {
            let regular_dim = head_dim - outlier_count;
            (regular_dim * usize::from(regular_bits)).div_ceil(8)
                + (outlier_count * usize::from(outlier_bits)).div_ceil(8)
                + head_dim.div_ceil(8)
                + (std::mem::size_of::<f32>() * 2)
        }
    }
}

/// Helper to get dtype size in bytes.
pub(crate) fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::Float32 => 4,
        Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Int32 => 4,
        Dtype::Int64 => 8,
        Dtype::Int16 => 2,
        Dtype::Int8 | Dtype::Uint8 => 1,
        Dtype::Uint16 => 2,
        Dtype::Uint32 => 4,
        Dtype::Uint64 => 8,
        Dtype::Bool => 1,
        Dtype::Complex64 => 8,
    }
}
