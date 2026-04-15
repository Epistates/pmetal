//! Standard KV cache implementation with lazy and eager allocation.

use pmetal_bridge::compat::{Array, Dtype, Exception, ops};

use crate::array_ext::ArrayDtypeExt;
use crate::kernels::FusedAttentionConfig;

use super::{
    CacheMode, KVCacheConfig, QuantizedKVCache, TurboQuantKvCache, create_turboquant_runtime,
    dtype_size,
};

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
        key_head_dim: usize,
        value_head_dim: usize,
        dtype: Dtype,
    ) -> Self {
        let k_shape = [
            batch_size as i32,
            num_kv_heads as i32,
            max_seq_len as i32,
            key_head_dim as i32,
        ];
        let v_shape = [
            batch_size as i32,
            num_kv_heads as i32,
            max_seq_len as i32,
            value_head_dim as i32,
        ];
        let keys = Some(ops::zeros(&k_shape, dtype));
        let values = Some(ops::zeros(&v_shape, dtype));

        Self {
            keys,
            values,
            offset: 0,
        }
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
    /// TurboQuant per-layer caches.
    turboquant_layers: Option<Vec<TurboQuantKvCache>>,
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
                config.value_head_dim,
                config.dtype,
            ));
        }

        // Evaluate all allocations to materialize them on device
        for cache in &mut layer_caches {
            if let Some(ref mut k) = cache.keys {
                k.eval();
            }
            if let Some(ref mut v) = cache.values {
                v.eval();
            }
        }

        Ok(Self {
            config,
            layer_caches,
            quantized_layers: None,
            turboquant_layers: None,
            total_tokens: 0,
        })
    }

    /// Get the cache configuration.
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// Take the layer's KV buffers and current offset for a fused
    /// compiled update step (used by `compiled_*_attn_block` callers).
    /// The buffer is grown to fit the next `prev_offset + new_seq_len`
    /// tokens before being handed out, so the compiled call can write
    /// `new_seq_len` rows at `prev_offset` via `put_along_axis`.
    ///
    /// The caller is responsible for invoking
    /// [`KVCache::commit_compiled_layer_buffers`] with the *new* keys
    /// and values returned by the compiled function — otherwise the
    /// layer cache is left in an inconsistent state.
    pub fn take_compiled_layer_buffers(
        &mut self,
        layer_idx: usize,
        new_seq_len: usize,
        n_kv_heads: i32,
        key_head_dim: i32,
        value_head_dim: i32,
        dtype: Dtype,
    ) -> Result<(Array, Array, usize), Exception> {
        if layer_idx >= self.config.num_layers {
            return Err(Exception::custom(format!(
                "Layer index {} out of range (num_layers={})",
                layer_idx, self.config.num_layers
            )));
        }
        if self.quantized_layers.is_some() || self.turboquant_layers.is_some() {
            return Err(Exception::custom(
                "compiled cache buffer access is unsupported for quantized caches",
            ));
        }
        let cache = &mut self.layer_caches[layer_idx];
        let prev_offset = cache.offset;
        let needed = prev_offset + new_seq_len;
        let needs_grow = match cache.keys.as_ref() {
            None => true,
            Some(k) => (k.dim(2) as usize) < needed,
        };
        if needs_grow {
            // Pre-allocate in chunks of CACHE_STEP_SIZE.
            let n_steps = needed.div_ceil(CACHE_STEP_SIZE);
            let new_alloc_len = n_steps * CACHE_STEP_SIZE;
            let batch = 1; // current callers all use batch=1
            let k_shape = [batch, n_kv_heads, new_alloc_len as i32, key_head_dim];
            let v_shape = [batch, n_kv_heads, new_alloc_len as i32, value_head_dim];
            let new_k = ops::zeros(&k_shape, dtype);
            let new_v = ops::zeros(&v_shape, dtype);
            if let (Some(existing_k), Some(existing_v)) = (&cache.keys, &cache.values) {
                cache.keys = Some(ops::concatenate_axis(&[existing_k, &new_k], 2));
                cache.values = Some(ops::concatenate_axis(&[existing_v, &new_v], 2));
            } else {
                cache.keys = Some(new_k);
                cache.values = Some(new_v);
            }
        }
        let keys = cache.keys.take().unwrap();
        let values = cache.values.take().unwrap();
        Ok((keys, values, prev_offset))
    }

    /// Commit the new KV buffers returned by a compiled fused step back
    /// to the layer cache. `new_offset` is the post-update offset
    /// (typically `prev_offset + new_seq_len`).
    pub fn commit_compiled_layer_buffers(
        &mut self,
        layer_idx: usize,
        new_keys: Array,
        new_values: Array,
        new_offset: usize,
    ) {
        let cache = &mut self.layer_caches[layer_idx];
        cache.keys = Some(new_keys);
        cache.values = Some(new_values);
        cache.offset = new_offset;
        if layer_idx == 0 {
            self.total_tokens = new_offset;
        }
    }

    /// Get the current cached sequence length.
    pub fn seq_len(&self) -> usize {
        if let Some(ref q_layers) = self.quantized_layers {
            q_layers.first().map(|c| c.len()).unwrap_or(0)
        } else if let Some(ref tq_layers) = self.turboquant_layers {
            tq_layers.first().map(|c| c.len()).unwrap_or(0)
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
        } else if let Some(ref tq_layers) = self.turboquant_layers {
            tq_layers.iter().all(|c| c.is_empty())
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
        } else if let Some(ref mut tq_layers) = self.turboquant_layers {
            for cache in tq_layers {
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
        if let Some(ref mut q_layers) = self.quantized_layers {
            for cache in q_layers {
                cache.reset();
            }
        }
        if let Some(ref mut tq_layers) = self.turboquant_layers {
            for cache in tq_layers {
                cache.reset();
            }
        }
        for cache in &mut self.layer_caches {
            cache.reset_full();
        }
        self.total_tokens = 0;
    }

    /// Roll back (discard) the last `n` tokens from every layer in the cache
    /// and return the count actually removed (clamped to the current length).
    ///
    /// This is the speculative-decoding primitive: after a verify step that
    /// accepted only some of the drafted tokens, call `rollback(rejected)` to
    /// reclaim the trailing positions. The return value is the same contract
    /// as MLX-LM / dflash-mlx `trim()` — callers can assert it matches the
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

        // Return slice views up to final_offset — matches Python mlx_lm pattern
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

    /// Get the offset for RoPE when using cache.
    ///
    /// This is the starting position for new tokens when computing
    /// rotary embeddings during cached generation.
    pub fn rope_offset(&self) -> i32 {
        match self.config.mode {
            CacheMode::SlidingWindow { .. } | CacheMode::Rotating { .. } => {
                self.total_tokens as i32
            }
            _ => self.seq_len() as i32,
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

    /// Fetch cached keys/values for a layer (for compiled decode).
    ///
    /// Returns the current cached keys/values up to the current offset.
    /// The compiled closure will concatenate new K/V and return the full result.
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

    /// Update cache from compiled decode outputs.
    ///
    /// The compiled closure returns full_keys/full_values (old + new concatenated).
    /// We just replace the cache contents with these.
    pub fn update_from_compiled_decode(
        &mut self,
        layer_idx: usize,
        full_keys: &Array,
        full_values: &Array,
    ) -> Result<(), Exception> {
        let cache = &mut self.layer_caches[layer_idx];
        let new_seq_len = full_keys.dim(2) as usize;

        // Ensure buffer is large enough
        let needs_growth = cache.keys.is_none() || {
            let allocated = cache.keys.as_ref().unwrap().dim(2) as usize;
            new_seq_len > allocated
        };
        if needs_growth {
            // Just store the full arrays directly
            cache.keys = Some(full_keys.clone());
            cache.values = Some(full_values.clone());
        } else {
            // In-place update into pre-allocated buffer
            let k_buf = cache.keys.take().unwrap();
            let v_buf = cache.values.take().unwrap();
            let kb = k_buf.dim(0) as usize;
            let kh = k_buf.dim(1) as usize;
            let kd = k_buf.dim(3) as usize;
            let vd = v_buf.dim(3) as usize;
            cache.keys = Some(k_buf.slice_set(
                full_keys,
                &[0, 0, 0, 0],
                &[kb as i32, kh as i32, new_seq_len as i32, kd as i32],
            ));
            cache.values = Some(v_buf.slice_set(
                full_values,
                &[0, 0, 0, 0],
                &[kb as i32, kh as i32, new_seq_len as i32, vd as i32],
            ));
        }
        cache.offset = new_seq_len;
        if layer_idx == 0 {
            self.total_tokens = new_seq_len;
        }
        Ok(())
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
