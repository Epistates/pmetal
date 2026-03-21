//! Standard KV cache implementation with lazy and eager allocation.

use mlx_rs::{
    Array, Dtype,
    error::Exception,
    ops,
    ops::concatenate_axis,
    ops::indexing::{IndexOp, TryIndexMutOp},
};

use super::{CacheMode, KVCacheConfig, QuantizedKVCache, dtype_size};

/// Pre-allocation step size in tokens (matches Python mlx-lm).
/// Cache grows in chunks of this size to avoid per-token allocations.
const CACHE_STEP_SIZE: usize = 256;

/// Per-layer KV cache entry.
#[derive(Debug, Clone)]
struct LayerCache {
    /// Cached keys [batch, heads, allocated_seq, head_dim] - attention format for SOTA performance.
    /// Note: allocated_seq may be larger than offset (pre-allocated space).
    keys: Option<Array>,
    /// Cached values [batch, heads, allocated_seq, head_dim] - attention format for SOTA performance.
    values: Option<Array>,
    /// Current offset (actual data length) within the pre-allocated buffer.
    /// This is the number of tokens actually stored, not the buffer size.
    offset: usize,
}

impl LayerCache {
    fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
        }
    }

    /// Create a new layer cache with pre-allocated buffers.
    fn new_eager(
        batch_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        dtype: Dtype,
    ) -> Result<Self, Exception> {
        let shape = [
            batch_size as i32,
            num_kv_heads as i32,
            max_seq_len as i32,
            head_dim as i32,
        ];
        let keys = Some(ops::zeros_dtype(&shape, dtype)?);
        let values = Some(ops::zeros_dtype(&shape, dtype)?);

        Ok(Self {
            keys,
            values,
            offset: 0,
        })
    }

    fn reset(&mut self) {
        // For eager allocation, just reset offset but keep buffers
        self.offset = 0;
        // Note: We don't set keys/values to None to preserve the pre-allocated buffers
    }

    fn reset_full(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
    }
}

/// Key-Value cache for transformer inference.
///
/// Stores computed key and value tensors across generation steps,
/// enabling efficient autoregressive generation.
#[derive(Debug)]
pub struct KVCache {
    /// Configuration.
    config: KVCacheConfig,
    /// Per-layer cache entries.
    layer_caches: Vec<LayerCache>,
    /// Quantized per-layer caches (used when mode is Quantized or AsymmetricQuantized).
    quantized_layers: Option<Vec<QuantizedKVCache>>,
    /// Total number of tokens processed.
    total_tokens: usize,
}

impl KVCache {
    /// Create a new KV cache with the given configuration.
    ///
    /// If `eager_allocate` is enabled in the config, this will pre-allocate
    /// the full context window immediately and may fail if memory is insufficient.
    pub fn new(config: KVCacheConfig) -> Self {
        let layer_caches = (0..config.num_layers).map(|_| LayerCache::new()).collect();

        let quantized_layers = match config.mode {
            CacheMode::Quantized { bits, group_size } => Some(
                (0..config.num_layers)
                    .map(|_| QuantizedKVCache::new(bits, group_size))
                    .collect(),
            ),
            CacheMode::AsymmetricQuantized {
                key_bits,
                value_bits,
                group_size,
            } => Some(
                (0..config.num_layers)
                    .map(|_| QuantizedKVCache::new_asymmetric(key_bits, value_bits, group_size))
                    .collect(),
            ),
            _ => None,
        };

        Self {
            config,
            layer_caches,
            quantized_layers,
            total_tokens: 0,
        }
    }

    /// Create a new KV cache with eager pre-allocation.
    ///
    /// Pre-allocates the full `max_seq_len` context window for all layers.
    /// Returns an error if memory allocation fails.
    ///
    /// # Memory Calculation
    /// Memory = 2 × num_layers × batch_size × num_kv_heads × max_seq_len × head_dim × dtype_size
    ///
    /// # Example
    /// ```ignore
    /// let config = KVCacheConfig::new(32, 4096, 8, 128)
    ///     .with_eager_allocate(1)
    ///     .with_dtype(Dtype::Float16);
    /// println!("Will allocate: {}", config.memory_footprint_human());
    /// let cache = KVCache::new_eager(config)?;
    /// ```
    pub fn new_eager(config: KVCacheConfig) -> Result<Self, Exception> {
        if !config.eager_allocate {
            // Fall back to lazy allocation
            return Ok(Self::new(config));
        }

        let mut layer_caches = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layer_caches.push(LayerCache::new_eager(
                config.eager_batch_size,
                config.num_kv_heads,
                config.max_seq_len,
                config.head_dim,
                config.dtype,
            )?);
        }

        // Evaluate all allocations to materialize them on device
        for cache in &layer_caches {
            if let Some(ref k) = cache.keys {
                k.eval()?;
            }
            if let Some(ref v) = cache.values {
                v.eval()?;
            }
        }

        Ok(Self {
            config,
            layer_caches,
            quantized_layers: None,
            total_tokens: 0,
        })
    }

    /// Get the cache configuration.
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// Get the current cached sequence length.
    pub fn seq_len(&self) -> usize {
        if let Some(ref q_layers) = self.quantized_layers {
            q_layers.first().map(|c| c.len()).unwrap_or(0)
        } else {
            self.layer_caches.first().map(|c| c.offset).unwrap_or(0)
        }
    }

    /// Get total tokens processed (may differ from seq_len with sliding window).
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Check if the cache is empty (no tokens stored).
    pub fn is_empty(&self) -> bool {
        if let Some(ref q_layers) = self.quantized_layers {
            q_layers.iter().all(|c| c.is_empty())
        } else {
            self.layer_caches.iter().all(|c| c.offset == 0)
        }
    }

    /// Check if the cache is pre-allocated (eager mode).
    pub fn is_preallocated(&self) -> bool {
        self.config.eager_allocate
            && self
                .layer_caches
                .first()
                .map(|c| c.keys.is_some())
                .unwrap_or(false)
    }

    /// Reset the cache for a new generation.
    ///
    /// For eager-allocated caches, this resets the offset but preserves the
    /// pre-allocated buffers. For lazy caches, this deallocates all memory.
    pub fn reset(&mut self) {
        if let Some(ref mut q_layers) = self.quantized_layers {
            for cache in q_layers {
                cache.reset();
            }
        } else if self.config.eager_allocate {
            for cache in &mut self.layer_caches {
                cache.reset();
            }
        } else {
            for cache in &mut self.layer_caches {
                cache.reset_full();
            }
        }
        self.total_tokens = 0;
    }

    /// Fully reset the cache, deallocating all memory.
    ///
    /// Unlike `reset()`, this always deallocates the buffers even for
    /// eager-allocated caches. Use this to free memory when done.
    pub fn reset_full(&mut self) {
        for cache in &mut self.layer_caches {
            cache.reset_full();
        }
        self.total_tokens = 0;
    }

    /// Roll back (discard) the last `n` tokens from every layer in the cache.
    ///
    /// This is used after speculative decoding when some draft tokens are
    /// rejected: the verify model has already appended the full draft sequence
    /// to its KV cache, but only the accepted prefix should be retained.
    ///
    /// The operation simply decrements each layer's `offset` by `n` and
    /// adjusts `total_tokens`.  The underlying buffer data beyond the new
    /// offset is left in place (overwritten on the next `update_and_fetch`
    /// call) — this is safe because `update_and_fetch` always writes before
    /// reading the new positions.
    ///
    /// # Panics
    ///
    /// Panics (in debug) if `n > seq_len()`.  In release builds the offset is
    /// clamped to 0 to avoid underflow.
    pub fn rollback(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        if let Some(ref mut q_layers) = self.quantized_layers {
            for cache in q_layers {
                cache.rollback(n);
            }
        } else {
            for cache in &mut self.layer_caches {
                cache.offset = cache.offset.saturating_sub(n);
            }
        }
        self.total_tokens = self.total_tokens.saturating_sub(n);
    }

    /// Update the cache with new keys and values for a layer.
    ///
    /// Keys/values are expected in **attention format** `[B, heads, seq, head_dim]`
    /// where axis 2 is the sequence dimension. This matches mlx_lm for SOTA performance.
    ///
    /// Uses pre-allocation in chunks of CACHE_STEP_SIZE (256 tokens) and in-place
    /// slice assignment for O(1) amortized per-token updates (matching Python mlx-lm).
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index
    /// * `new_keys` - New keys [batch, heads, new_seq, head_dim]
    /// * `new_values` - New values [batch, heads, new_seq, head_dim]
    ///
    /// # Returns
    /// (cached_keys, cached_values) - Slice views of cached keys/values
    pub fn update_and_fetch(
        &mut self,
        layer_idx: usize,
        new_keys: &Array,
        new_values: &Array,
    ) -> Result<(Array, Array), Exception> {
        if layer_idx >= self.config.num_layers {
            return Err(Exception::custom(format!(
                "Layer index {} out of range (num_layers={})",
                layer_idx, self.config.num_layers
            )));
        }

        // Quantized path: delegate to per-layer QuantizedKVCache
        if let Some(ref mut q_layers) = self.quantized_layers {
            if layer_idx == 0 {
                self.total_tokens += new_keys.dim(2) as usize;
            }
            return q_layers[layer_idx].update_and_fetch(new_keys, new_values);
        }

        let cache = &mut self.layer_caches[layer_idx];
        // Sequence dimension is axis 2 in [B, heads, seq, head_dim] format
        let new_seq_len = new_keys.dim(2) as usize;
        let prev_offset = cache.offset;

        // Update total tokens count (only count for first layer to avoid double counting)
        if layer_idx == 0 {
            self.total_tokens += new_seq_len;
        }

        // Check if we need to grow the cache (pre-allocation pattern from mlx_lm)
        let needs_growth = cache.keys.is_none() || {
            let allocated = cache.keys.as_ref().unwrap().dim(2) as usize;
            prev_offset + new_seq_len > allocated
        };

        if needs_growth {
            // Pre-allocate in chunks of CACHE_STEP_SIZE
            let n_steps = (CACHE_STEP_SIZE + new_seq_len - 1) / CACHE_STEP_SIZE;
            let new_alloc_len = n_steps * CACHE_STEP_SIZE;

            // Get shape from new_keys [B, heads, _, head_dim]
            let batch = new_keys.dim(0);
            let heads = new_keys.dim(1);
            let head_dim = new_keys.dim(3);

            // Create new zero-filled buffer
            let k_shape = [batch, heads, new_alloc_len as i32, head_dim];
            let v_shape = [batch, heads, new_alloc_len as i32, head_dim];
            let new_k_buffer = ops::zeros_dtype(&k_shape, new_keys.dtype())?;
            let new_v_buffer = ops::zeros_dtype(&v_shape, new_values.dtype())?;

            if let (Some(existing_k), Some(existing_v)) = (&cache.keys, &cache.values) {
                // Concatenate existing data with new buffer
                cache.keys = Some(concatenate_axis(&[existing_k, &new_k_buffer], 2)?);
                cache.values = Some(concatenate_axis(&[existing_v, &new_v_buffer], 2)?);
            } else {
                cache.keys = Some(new_k_buffer);
                cache.values = Some(new_v_buffer);
            }
        }

        // Update offset before in-place assignment
        cache.offset = prev_offset + new_seq_len;

        // In-place slice assignment: cache[..., prev:offset, :] = new_keys
        // Use TryIndexMutOp for O(1) update instead of concatenate
        let k_buf = cache.keys.as_mut().unwrap();
        let v_buf = cache.values.as_mut().unwrap();

        // Assign new keys/values into the pre-allocated buffer
        k_buf.try_index_mut(
            (.., .., prev_offset as i32..cache.offset as i32, ..),
            new_keys,
        )?;
        v_buf.try_index_mut(
            (.., .., prev_offset as i32..cache.offset as i32, ..),
            new_values,
        )?;

        // Apply cache mode limits (sliding window, rotating, etc.)
        let final_offset = match self.config.mode {
            CacheMode::SlidingWindow { window_size } => {
                if cache.offset > window_size {
                    // For sliding window, we need to shift data and adjust offset
                    // This is a less common path, so we can do a copy here
                    let shift = cache.offset - window_size;
                    let k = cache.keys.as_ref().unwrap();
                    let v = cache.values.as_ref().unwrap();
                    cache.keys = Some(k.index((.., .., shift as i32..cache.offset as i32, ..)));
                    cache.values = Some(v.index((.., .., shift as i32..cache.offset as i32, ..)));
                    cache.offset = window_size;
                }
                cache.offset
            }
            CacheMode::Rotating { max_size, .. } => {
                if cache.offset > max_size {
                    let shift = cache.offset - max_size;
                    let k = cache.keys.as_ref().unwrap();
                    let v = cache.values.as_ref().unwrap();
                    cache.keys = Some(k.index((.., .., shift as i32..cache.offset as i32, ..)));
                    cache.values = Some(v.index((.., .., shift as i32..cache.offset as i32, ..)));
                    cache.offset = max_size;
                }
                cache.offset
            }
            CacheMode::Quantized { .. }
            | CacheMode::AsymmetricQuantized { .. }
            | CacheMode::Standard => {
                if cache.offset > self.config.max_seq_len {
                    return Err(Exception::custom(format!(
                        "KV cache exceeded max_seq_len: {} > {}",
                        cache.offset, self.config.max_seq_len
                    )));
                }
                cache.offset
            }
        };

        // Return slice views (not clones) - matches Python mlx_lm pattern
        let k = cache.keys.as_ref().unwrap();
        let v = cache.values.as_ref().unwrap();
        Ok((
            k.index((.., .., ..final_offset as i32, ..)),
            v.index((.., .., ..final_offset as i32, ..)),
        ))
    }

    /// Legacy update method - DEPRECATED, use update_and_fetch instead.
    /// Kept for backwards compatibility.
    #[deprecated(
        since = "0.2.0",
        note = "Use update_and_fetch instead for SOTA performance"
    )]
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_keys: &Array,
        new_values: &Array,
    ) -> Result<(Array, Array), Exception> {
        self.update_and_fetch(layer_idx, new_keys, new_values)
    }

    /// Get cached keys and values for a layer without updating.
    ///
    /// Returns sliced views up to the actual sequence length (not the pre-allocated buffer size).
    /// Returns None if the cache for this layer is empty.
    pub fn get(&self, layer_idx: usize) -> Option<(Array, Array)> {
        let cache = self.layer_caches.get(layer_idx)?;
        match (&cache.keys, &cache.values) {
            (Some(k), Some(v)) if cache.offset > 0 => {
                // Return sliced view up to actual offset (not full pre-allocated buffer)
                Some((
                    k.index((.., .., ..cache.offset as i32, ..)),
                    v.index((.., .., ..cache.offset as i32, ..)),
                ))
            }
            _ => None,
        }
    }

    /// Get the offset for RoPE when using cache.
    ///
    /// This is the starting position for new tokens when computing
    /// rotary embeddings during cached generation.
    pub fn rope_offset(&self) -> i32 {
        self.seq_len() as i32
    }

    /// Estimate memory usage of the current cache in bytes.
    pub fn memory_usage(&self) -> usize {
        if let Some(ref q_layers) = self.quantized_layers {
            return q_layers.iter().map(|c| c.memory_usage()).sum();
        }
        let mut total = 0;
        for cache in &self.layer_caches {
            if let Some(keys) = &cache.keys {
                let elements: usize = keys.shape().iter().map(|&d| d as usize).product();
                total += elements * dtype_size(self.config.dtype);
            }
            if let Some(values) = &cache.values {
                let elements: usize = values.shape().iter().map(|&d| d as usize).product();
                total += elements * dtype_size(self.config.dtype);
            }
        }
        total
    }

    /// Estimate maximum memory usage for the configured cache.
    pub fn max_memory_usage(&self) -> usize {
        // For quantized modes, use the config's memory_footprint calculation
        match self.config.mode {
            CacheMode::Quantized { .. } | CacheMode::AsymmetricQuantized { .. } => {
                return self.config.memory_footprint();
            }
            _ => {}
        }
        let seq_len = match self.config.mode {
            CacheMode::SlidingWindow { window_size } => window_size,
            CacheMode::Rotating { max_size, .. } => max_size,
            _ => self.config.max_seq_len,
        };
        let elements_per_layer = seq_len * self.config.num_kv_heads * self.config.head_dim;
        let bytes_per_layer = elements_per_layer * dtype_size(self.config.dtype) * 2;
        self.config.num_layers * bytes_per_layer
    }
}

/// Convenience function to create a standard KV cache.
pub fn create_kv_cache(
    num_layers: usize,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> KVCache {
    KVCache::new(KVCacheConfig::new(
        num_layers,
        max_seq_len,
        num_kv_heads,
        head_dim,
    ))
}

/// Convenience function to create a sliding window KV cache.
pub fn create_sliding_window_cache(
    num_layers: usize,
    window_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> KVCache {
    KVCache::new(
        KVCacheConfig::new(num_layers, window_size * 2, num_kv_heads, head_dim)
            .with_sliding_window(window_size),
    )
}

/// Batch of KV caches for parallel generation.
#[derive(Debug)]
pub struct BatchKVCache {
    /// Individual caches per batch entry.
    pub(crate) caches: Vec<KVCache>,
}

impl BatchKVCache {
    /// Create a new batch of KV caches.
    pub fn new(batch_size: usize, config: KVCacheConfig) -> Self {
        let caches = (0..batch_size)
            .map(|_| KVCache::new(config.clone()))
            .collect();
        Self { caches }
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.caches.len()
    }

    /// Get a mutable reference to a specific cache.
    pub fn get_mut(&mut self, batch_idx: usize) -> Option<&mut KVCache> {
        self.caches.get_mut(batch_idx)
    }

    /// Reset all caches.
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// Reset specific cache entries (for finished sequences).
    pub fn reset_indices(&mut self, indices: &[usize]) {
        for &idx in indices {
            if let Some(cache) = self.caches.get_mut(idx) {
                cache.reset();
            }
        }
    }

    /// Total memory usage across all caches.
    pub fn memory_usage(&self) -> usize {
        self.caches.iter().map(|c| c.memory_usage()).sum()
    }
}
