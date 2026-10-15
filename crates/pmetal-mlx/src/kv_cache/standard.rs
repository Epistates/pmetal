//! The dense KV cache and the cache every model's layers share.

use pmetal_runtime::compat::{Array, Exception, ops};

use crate::kernels::FusedAttentionConfig;

use super::{
    CacheMode, KVCacheConfig, QuantizedKVCache, TurboQuantKvCache, create_turboquant_runtime,
    dtype_size,
};

/// Pre-allocation step size in tokens.
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
    /// Tokens this layer has ever been given, which `offset` stops tracking
    /// once a sliding or rotating mode starts dropping the oldest ones. RoPE
    /// needs the absolute position, so it reads this rather than `offset`.
    tokens_seen: usize,
}

impl LayerCache {
    fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            tokens_seen: 0,
        }
    }

    fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self.tokens_seen = 0;
    }
}

/// Key-Value cache for transformer inference.
///
/// Stores computed key and value tensors across generation steps,
/// enabling efficient autoregressive generation.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Configuration.
    config: KVCacheConfig,
    /// Per-layer cache entries.
    layer_caches: Vec<LayerCache>,
    /// Quantized per-layer caches (used when mode is Quantized or AsymmetricQuantized).
    quantized_layers: Option<Vec<QuantizedKVCache>>,
    /// TurboQuant per-layer caches.
    turboquant_layers: Option<Vec<TurboQuantKvCache>>,
    /// Total number of tokens processed.
    total_tokens: usize,
}

impl KVCache {
    /// Create a new KV cache with the given configuration. Buffers are
    /// allocated as tokens arrive.
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
        let turboquant_layers = match config.mode {
            CacheMode::TurboQuant { config: turboquant } => {
                let shared_runtime =
                    create_turboquant_runtime(config.head_dim, config.value_head_dim, turboquant);
                Some(
                    (0..config.num_layers)
                        .map(|_| {
                            TurboQuantKvCache::new_with_runtime(turboquant, shared_runtime.clone())
                        })
                        .collect(),
                )
            }
            _ => None,
        };

        Self {
            config,
            layer_caches,
            quantized_layers,
            turboquant_layers,
            total_tokens: 0,
        }
    }

    /// Get the cache configuration.
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// Returns `true` when the cache is in TurboQuant mode and at least one
    /// layer has actually compressed history.
    pub fn turboquant_compression_active(&self) -> bool {
        if !matches!(self.config.mode, CacheMode::TurboQuant { .. }) {
            return false;
        }
        match self.turboquant_layers.as_ref() {
            Some(layers) => layers.iter().any(|layer| !layer.is_empty()),
            None => false,
        }
    }

    /// Cheaply fork this cache for serving prefix reuse.
    ///
    /// `Array` clones are MLX handle clones, so dense K/V tensors are shared
    /// until the fork diverges. Quantized and TurboQuant layers clone their
    /// compressed metadata directly, preserving the active cache mode instead
    /// of round-tripping through dense fp16 snapshots.
    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Get the current cached sequence length.
    pub fn seq_len(&self) -> usize {
        if let Some(ref q_layers) = self.quantized_layers {
            q_layers.first().map(|c| c.len()).unwrap_or(0)
        } else if let Some(ref tq_layers) = self.turboquant_layers {
            tq_layers.first().map(|c| c.len()).unwrap_or(0)
        } else {
            self.layer_caches
                .iter()
                .map(|cache| cache.offset)
                .max()
                .unwrap_or(0)
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
        } else if let Some(ref tq_layers) = self.turboquant_layers {
            tq_layers.iter().all(|c| c.is_empty())
        } else {
            self.layer_caches.iter().all(|c| c.offset == 0)
        }
    }

    /// Reset the cache for a new generation.
    ///
    /// Every layer's buffers are released.
    pub fn reset(&mut self) {
        if let Some(ref mut q_layers) = self.quantized_layers {
            for cache in q_layers {
                cache.reset();
            }
        } else if let Some(ref mut tq_layers) = self.turboquant_layers {
            for cache in tq_layers {
                cache.reset();
            }
        } else {
            for cache in &mut self.layer_caches {
                cache.reset();
            }
        }
        self.total_tokens = 0;
    }

    /// Roll back (discard) the last `n` tokens from every layer in the cache
    /// and return the count actually removed (clamped to the current length).
    ///
    /// This is the speculative-decoding primitive: after a verify step that
    /// accepted only some of the drafted tokens, call `rollback(rejected)` to
    /// reclaim the trailing positions. The return value is the same contract
    /// as the reference `trim()` — callers can assert it matches the
    /// expected discard count.
    ///
    /// For the standard path this is an O(1) offset decrement; the buffer is
    /// left in place and the orphaned region is overwritten on the next
    /// `update_and_fetch`. Quantized / TurboQuant variants physically reclaim
    /// storage via sliced re-assignment.
    pub fn rollback(&mut self, n: usize) -> usize {
        let trimmed = n.min(self.seq_len());
        if trimmed == 0 {
            return 0;
        }
        if let Some(ref mut q_layers) = self.quantized_layers {
            for cache in q_layers {
                cache.rollback(trimmed);
            }
        } else if let Some(ref mut tq_layers) = self.turboquant_layers {
            for cache in tq_layers {
                cache.rollback(trimmed);
            }
        } else {
            for cache in &mut self.layer_caches {
                cache.offset = cache.offset.saturating_sub(trimmed);
            }
        }
        self.total_tokens = self.total_tokens.saturating_sub(trimmed);
        trimmed
    }

    /// Update the cache with new keys and values for a layer.
    ///
    /// Keys/values are expected in **attention format** `[B, heads, seq, head_dim]`
    /// where axis 2 is the sequence dimension.
    ///
    /// Uses pre-allocation in chunks of CACHE_STEP_SIZE (256 tokens) and in-place
    /// slice assignment for O(1) amortized per-token updates.
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
        if let Some(ref mut tq_layers) = self.turboquant_layers {
            if layer_idx == 0 {
                self.total_tokens += new_keys.dim(2) as usize;
            }
            return tq_layers[layer_idx].update_and_fetch(new_keys, new_values);
        }

        let cache = &mut self.layer_caches[layer_idx];
        // Sequence dimension is axis 2 in [B, heads, seq, head_dim] format
        let new_seq_len = new_keys.dim(2) as usize;
        let prev_offset = cache.offset;
        cache.tokens_seen += new_seq_len;

        // Update total tokens count (only count for first layer to avoid double counting)
        if layer_idx == 0 {
            self.total_tokens += new_seq_len;
        }

        // Check if we need to grow the cache (chunked pre-allocation)
        let needs_growth = cache.keys.is_none() || {
            let allocated = cache.keys.as_ref().unwrap().dim(2) as usize;
            prev_offset + new_seq_len > allocated
        };

        if needs_growth {
            // Pre-allocate in chunks of CACHE_STEP_SIZE
            let n_steps = (CACHE_STEP_SIZE + new_seq_len - 1) / CACHE_STEP_SIZE;
            let new_alloc_len = n_steps * CACHE_STEP_SIZE;

            // Get shapes from new keys/values [B, heads, _, head_dim]
            let batch = new_keys.dim(0);
            let heads = new_keys.dim(1);
            let key_head_dim = new_keys.dim(3);
            let value_head_dim = new_values.dim(3);

            // Create new zero-filled buffer with matching dtype
            let k_shape = [batch, heads, new_alloc_len as i32, key_head_dim];
            let v_shape = [batch, heads, new_alloc_len as i32, value_head_dim];
            let new_k_buffer = ops::zeros(&k_shape, new_keys.dtype());
            let new_v_buffer = ops::zeros(&v_shape, new_values.dtype());

            if let (Some(existing_k), Some(existing_v)) = (&cache.keys, &cache.values) {
                // Concatenate existing data with new buffer
                cache.keys = Some(ops::concatenate_axis(&[existing_k, &new_k_buffer], 2));
                cache.values = Some(ops::concatenate_axis(&[existing_v, &new_v_buffer], 2));
            } else {
                cache.keys = Some(new_k_buffer);
                cache.values = Some(new_v_buffer);
            }
        }

        // Update offset before in-place assignment
        cache.offset = prev_offset + new_seq_len;

        // In-place slice assignment: cache[..., prev:offset, :] = new_keys
        // Use slice_set for O(1) update instead of concatenate.
        // slice_set takes explicit start/stop for every dimension.
        let k_buf = cache.keys.take().unwrap();
        let v_buf = cache.values.take().unwrap();

        let batch = k_buf.dim(0) as usize;
        let heads = k_buf.dim(1) as usize;
        let _alloc = k_buf.dim(2) as usize;
        let key_hdim = k_buf.dim(3) as usize;
        let val_hdim = v_buf.dim(3) as usize;

        let k_start = [0i32, 0, prev_offset as i32, 0];
        let k_stop = [
            batch as i32,
            heads as i32,
            cache.offset as i32,
            key_hdim as i32,
        ];
        let v_start = [0i32, 0, prev_offset as i32, 0];
        let v_stop = [
            batch as i32,
            heads as i32,
            cache.offset as i32,
            val_hdim as i32,
        ];

        cache.keys = Some(k_buf.slice_set(new_keys, &k_start, &k_stop));
        cache.values = Some(v_buf.slice_set(new_values, &v_start, &v_stop));

        // Apply cache mode limits (sliding window, rotating, etc.)
        let final_offset = match self.config.mode {
            CacheMode::SlidingWindow { window_size } => {
                if cache.offset > window_size {
                    // For sliding window, shift data and adjust offset
                    let shift = cache.offset - window_size;
                    let k = cache.keys.as_ref().unwrap();
                    let v = cache.values.as_ref().unwrap();
                    let kb = k.dim(0) as usize;
                    let kh = k.dim(1) as usize;
                    let kd = k.dim(3) as usize;
                    let vd = v.dim(3) as usize;
                    cache.keys = Some(k.slice(
                        &[0, 0, shift as i32, 0],
                        &[kb as i32, kh as i32, cache.offset as i32, kd as i32],
                    ));
                    cache.values = Some(v.slice(
                        &[0, 0, shift as i32, 0],
                        &[kb as i32, kh as i32, cache.offset as i32, vd as i32],
                    ));
                    cache.offset = window_size;
                }
                cache.offset
            }
            CacheMode::Rotating { max_size, keep } => {
                if cache.offset > max_size {
                    let keep = keep.min(max_size);
                    let tail_len = max_size.saturating_sub(keep);
                    let k = cache.keys.as_ref().unwrap();
                    let v = cache.values.as_ref().unwrap();
                    let kb = k.dim(0) as usize;
                    let kh = k.dim(1) as usize;
                    let kd = k.dim(3) as usize;
                    let vd = v.dim(3) as usize;

                    let rotated_keys = if keep == 0 {
                        let tail_start = cache.offset - max_size;
                        k.slice(
                            &[0, 0, tail_start as i32, 0],
                            &[kb as i32, kh as i32, cache.offset as i32, kd as i32],
                        )
                    } else if tail_len == 0 {
                        k.slice(
                            &[0, 0, 0, 0],
                            &[kb as i32, kh as i32, keep as i32, kd as i32],
                        )
                    } else {
                        let kept = k.slice(
                            &[0, 0, 0, 0],
                            &[kb as i32, kh as i32, keep as i32, kd as i32],
                        );
                        let tail_start = cache.offset - tail_len;
                        let tail = k.slice(
                            &[0, 0, tail_start as i32, 0],
                            &[kb as i32, kh as i32, cache.offset as i32, kd as i32],
                        );
                        ops::concatenate_axis(&[&kept, &tail], 2)
                    };
                    let rotated_values = if keep == 0 {
                        let tail_start = cache.offset - max_size;
                        v.slice(
                            &[0, 0, tail_start as i32, 0],
                            &[kb as i32, kh as i32, cache.offset as i32, vd as i32],
                        )
                    } else if tail_len == 0 {
                        v.slice(
                            &[0, 0, 0, 0],
                            &[kb as i32, kh as i32, keep as i32, vd as i32],
                        )
                    } else {
                        let kept = v.slice(
                            &[0, 0, 0, 0],
                            &[kb as i32, kh as i32, keep as i32, vd as i32],
                        );
                        let tail_start = cache.offset - tail_len;
                        let tail = v.slice(
                            &[0, 0, tail_start as i32, 0],
                            &[kb as i32, kh as i32, cache.offset as i32, vd as i32],
                        );
                        ops::concatenate_axis(&[&kept, &tail], 2)
                    };
                    cache.keys = Some(rotated_keys);
                    cache.values = Some(rotated_values);
                    cache.offset = max_size;
                }
                cache.offset
            }
            CacheMode::Quantized { .. }
            | CacheMode::AsymmetricQuantized { .. }
            | CacheMode::TurboQuant { .. }
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

        // Return slice views up to final_offset
        let k = cache.keys.as_ref().unwrap();
        let v = cache.values.as_ref().unwrap();
        let kb = k.dim(0) as usize;
        let kh = k.dim(1) as usize;
        let kd = k.dim(3) as usize;
        let vd = v.dim(3) as usize;
        Ok((
            k.slice(
                &[0, 0, 0, 0],
                &[kb as i32, kh as i32, final_offset as i32, kd as i32],
            ),
            v.slice(
                &[0, 0, 0, 0],
                &[kb as i32, kh as i32, final_offset as i32, vd as i32],
            ),
        ))
    }

    /// Try the TurboQuant direct-attention path for single-token decode.
    ///
    /// Returns `Ok(Some(output))` when the active cache mode is TurboQuant and
    /// the inputs satisfy the decode-only fast-path constraints.
    pub fn try_turboquant_attention(
        &mut self,
        layer_idx: usize,
        queries: &Array,
        new_keys: &Array,
        new_values: &Array,
        attn_config: &FusedAttentionConfig,
    ) -> Result<Option<Array>, Exception> {
        if layer_idx >= self.config.num_layers {
            return Err(Exception::custom(format!(
                "Layer index {} out of range (num_layers={})",
                layer_idx, self.config.num_layers
            )));
        }

        let Some(tq_layers) = self.turboquant_layers.as_mut() else {
            return Ok(None);
        };
        if !tq_layers[layer_idx].can_direct_attention(queries, new_keys, new_values, attn_config) {
            return Ok(None);
        }

        if layer_idx == 0 {
            self.total_tokens += new_keys.dim(2) as usize;
        }
        tq_layers[layer_idx]
            .append_and_compute_attention(queries, new_keys, new_values, attn_config)
            .map(Some)
    }

    /// Get cached keys and values for a layer without updating.
    ///
    /// Returns sliced views up to the actual sequence length (not the pre-allocated buffer size).
    /// Returns None if the cache for this layer is empty.
    pub fn get(&self, layer_idx: usize) -> Option<(Array, Array)> {
        let cache = self.layer_caches.get(layer_idx)?;
        match (&cache.keys, &cache.values) {
            (Some(k), Some(v)) if cache.offset > 0 => {
                let kb = k.dim(0) as usize;
                let kh = k.dim(1) as usize;
                let kd = k.dim(3) as usize;
                let vd = v.dim(3) as usize;
                Some((
                    k.slice(
                        &[0, 0, 0, 0],
                        &[kb as i32, kh as i32, cache.offset as i32, kd as i32],
                    ),
                    v.slice(
                        &[0, 0, 0, 0],
                        &[kb as i32, kh as i32, cache.offset as i32, vd as i32],
                    ),
                ))
            }
            _ => None,
        }
    }

    /// The absolute position of this layer's next token, for RoPE.
    ///
    /// **Per layer, and read before that layer's own
    /// [`update_and_fetch`](Self::update_and_fetch).** Every layer advances its
    /// own offset during the same forward pass, so a whole-cache reading is
    /// already one chunk ahead by the time layer 1 asks: layer 0 would rotate
    /// its new token at the right position and every layer after it one step
    /// too far. Prefill hides that — shifting a whole layer uniformly is
    /// invisible to relative RoPE — but the first decode step lands the new
    /// token inside the range its own cached keys already occupy, and the error
    /// compounds with depth.
    ///
    /// Sliding and rotating modes drop the oldest entries, so `offset` stops
    /// being the absolute position and `tokens_seen` is what RoPE wants.
    pub fn rope_offset_for(&self, layer_idx: usize) -> i32 {
        if let Some(ref q_layers) = self.quantized_layers {
            return q_layers.get(layer_idx).map_or(0, |c| c.len()) as i32;
        }
        if let Some(ref tq_layers) = self.turboquant_layers {
            return tq_layers.get(layer_idx).map_or(0, |c| c.len()) as i32;
        }
        let Some(cache) = self.layer_caches.get(layer_idx) else {
            return 0;
        };
        match self.config.mode {
            CacheMode::SlidingWindow { .. } | CacheMode::Rotating { .. } => {
                cache.tokens_seen as i32
            }
            _ => cache.offset as i32,
        }
    }

    /// Estimate memory usage of the current cache in bytes.
    pub fn memory_usage(&self) -> usize {
        if let Some(ref q_layers) = self.quantized_layers {
            return q_layers.iter().map(|c| c.memory_usage()).sum();
        }
        if let Some(ref tq_layers) = self.turboquant_layers {
            return tq_layers.iter().map(|c| c.memory_usage()).sum();
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
            CacheMode::Quantized { .. }
            | CacheMode::AsymmetricQuantized { .. }
            | CacheMode::TurboQuant { .. } => {
                return self.config.memory_footprint();
            }
            _ => {}
        }
        let seq_len = match self.config.mode {
            CacheMode::SlidingWindow { window_size } => window_size,
            CacheMode::Rotating { max_size, .. } => max_size,
            _ => self.config.max_seq_len,
        };
        let key_elements = seq_len * self.config.num_kv_heads * self.config.head_dim;
        let value_elements = seq_len * self.config.num_kv_heads * self.config.value_head_dim;
        let bytes_per_layer = (key_elements + value_elements) * dtype_size(self.config.dtype);
        self.config.num_layers * bytes_per_layer
    }

    /// Fetch cached keys/values for a layer, for a decode path that keeps
    /// its own copy of the cache.
    ///
    /// Returns the current cached keys/values up to the current offset.
    pub fn fetch_for_compiled_decode(&self, layer_idx: usize) -> Result<(Array, Array), Exception> {
        let cache = &self.layer_caches[layer_idx];
        if let (Some(k), Some(v)) = (&cache.keys, &cache.values) {
            let kb = k.dim(0) as usize;
            let kh = k.dim(1) as usize;
            let kd = k.dim(3) as usize;
            let vd = v.dim(3) as usize;
            let offset = cache.offset;
            Ok((
                k.slice(
                    &[0, 0, 0, 0],
                    &[kb as i32, kh as i32, offset as i32, kd as i32],
                ),
                v.slice(
                    &[0, 0, 0, 0],
                    &[kb as i32, kh as i32, offset as i32, vd as i32],
                ),
            ))
        } else {
            // No cache yet — return empty arrays with correct shape
            Err(Exception::custom(format!(
                "KV cache for layer {layer_idx} not initialized — run prefill first"
            )))
        }
    }
}

/// Batch of KV caches for parallel generation.
#[derive(Debug, Clone)]
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

    /// Replace one batch row with a forked/restored cache.
    pub fn replace(&mut self, batch_idx: usize, cache: KVCache) -> Result<(), Exception> {
        let batch_size = self.caches.len();
        let Some(existing) = self.caches.get_mut(batch_idx) else {
            return Err(Exception::custom(format!(
                "batch index {batch_idx} out of range (batch_size={batch_size})",
            )));
        };
        if existing.config() != cache.config() {
            return Err(Exception::custom(
                "cannot replace batch KV cache row with mismatched KVCacheConfig",
            ));
        }
        *existing = cache;
        Ok(())
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
