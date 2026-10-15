//! The KV cache every model's layers share.

use pmetal_runtime::compat::{Array, Exception};

use crate::kernels::FusedAttentionConfig;

use super::dense::{DenseLayer, Window};
use super::{
    CacheMode, KVCacheConfig, QuantizedKVCache, TurboQuantKvCache, create_turboquant_runtime,
    dtype_size,
};

/// Key-Value cache for transformer inference.
///
/// Stores computed key and value tensors across generation steps,
/// enabling efficient autoregressive generation. The mode in the
/// [`KVCacheConfig`] picks how: every token in the model's dtype, a sliding
/// window with attention sinks (see [`DenseLayer`]), or a quantized store.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Configuration.
    config: KVCacheConfig,
    /// The window of a sliding or rotating mode.
    window: Option<Window>,
    /// Per-layer dense entries.
    layer_caches: Vec<DenseLayer>,
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
        let layer_caches = (0..config.num_layers)
            .map(|_| DenseLayer::default())
            .collect();

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
            window: Window::of(config.mode),
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

    /// Tokens the cache can serve a single-token step from: all of them, or
    /// at most the window of a sliding or rotating mode.
    pub fn seq_len(&self) -> usize {
        if let Some(ref q_layers) = self.quantized_layers {
            q_layers.first().map(|c| c.len()).unwrap_or(0)
        } else if let Some(ref tq_layers) = self.turboquant_layers {
            tq_layers.first().map(|c| c.len()).unwrap_or(0)
        } else {
            self.layer_caches
                .iter()
                .map(|layer| layer.retained(self.window))
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
            self.layer_caches.iter().all(DenseLayer::is_empty)
        }
    }

    /// Reset the cache for a new generation. Every layer's buffers are
    /// released.
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
    /// For the dense path this is an O(1) offset decrement; the buffer is
    /// left in place and the orphaned region is overwritten on the next
    /// `update_and_fetch`. A sliding or rotating mode can only give back
    /// tokens its window still covers: once it has dropped old tokens it
    /// refuses a rollback that would leave the next token short of its
    /// window, and returns the smaller count it did remove. Quantized /
    /// TurboQuant variants physically reclaim storage via sliced
    /// re-assignment.
    pub fn rollback(&mut self, n: usize) -> usize {
        let mut trimmed = n.min(self.seq_len());
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
            if let Some(window) = self.window {
                // Every layer must give back the same count.
                trimmed = self
                    .layer_caches
                    .iter()
                    .filter(|layer| !layer.is_empty())
                    .map(|layer| layer.max_rollback(Some(window)))
                    .fold(trimmed, usize::min);
            }
            for layer in &mut self.layer_caches {
                layer.rollback(trimmed, self.window);
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
    /// A token is one in-place slice write into a buffer that grows by
    /// doubling, so the cost per token does not grow with the context.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index
    /// * `new_keys` - New keys [batch, heads, new_seq, head_dim]
    /// * `new_values` - New values [batch, heads, new_seq, head_dim]
    ///
    /// # Returns
    /// The keys and values the new tokens attend to. With every token kept
    /// that is the whole cache. A sliding or rotating mode returns the sinks
    /// and the `window - 1` tokens before the first new one, then the new
    /// ones: for a single token exactly the `window` it attends to, for a
    /// longer chunk some keys an early query is not meant to see, which
    /// [`Self::window_mask`] hides.
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

        let fetched = self.layer_caches[layer_idx].update(
            new_keys,
            new_values,
            self.window,
            self.config.max_seq_len,
        )?;
        if layer_idx == 0 {
            self.total_tokens += new_keys.dim(2) as usize;
        }
        Ok(fetched)
    }

    /// The additive attention mask for the keys the last
    /// [`update_and_fetch`](Self::update_and_fetch) of `layer_idx` returned,
    /// which added `new_len` tokens: `[1, 1, new_len, keys]`, 0 where a query
    /// sees a key and -inf where its window does not reach it.
    ///
    /// `None` when no mask is needed: a mode that keeps every token (causal
    /// attention does the rest), or a single new token, which sees every key
    /// returned.
    pub fn window_mask(&self, layer_idx: usize, new_len: usize) -> Option<Array> {
        self.layer_caches
            .get(layer_idx)?
            .window_mask(self.window, new_len)
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

    /// The cached keys and values of a layer as a single-token step would
    /// see them, or `None` while the layer is empty.
    pub fn get(&self, layer_idx: usize) -> Option<(Array, Array)> {
        self.layer_caches.get(layer_idx)?.current(self.window)
    }

    /// The dense key buffer of a layer, for tests that watch where it lives.
    #[cfg(test)]
    pub(super) fn layer_key_buffer(&self, layer_idx: usize) -> Option<&Array> {
        self.layer_caches.get(layer_idx)?.key_buffer()
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
    /// Sliding and rotating modes drop the oldest entries, so the number of
    /// tokens held stops being the absolute position; the tokens the layer has
    /// been given is what RoPE wants, and it is what this returns.
    pub fn rope_offset_for(&self, layer_idx: usize) -> i32 {
        if let Some(ref q_layers) = self.quantized_layers {
            return q_layers.get(layer_idx).map_or(0, |c| c.len()) as i32;
        }
        if let Some(ref tq_layers) = self.turboquant_layers {
            return tq_layers.get(layer_idx).map_or(0, |c| c.len()) as i32;
        }
        self.layer_caches
            .get(layer_idx)
            .map_or(0, |layer| layer.seen() as i32)
    }

    /// Bytes the cache holds now: the arrays it keeps, in the dtype they are
    /// in, buffer slack included.
    pub fn memory_usage(&self) -> usize {
        if let Some(ref q_layers) = self.quantized_layers {
            return q_layers.iter().map(|c| c.memory_usage()).sum();
        }
        if let Some(ref tq_layers) = self.turboquant_layers {
            return tq_layers.iter().map(|c| c.memory_usage()).sum();
        }
        self.layer_caches.iter().map(DenseLayer::nbytes).sum()
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
    pub fn fetch_for_compiled_decode(&self, layer_idx: usize) -> Result<(Array, Array), Exception> {
        self.get(layer_idx).ok_or_else(|| {
            Exception::custom(format!(
                "KV cache for layer {layer_idx} not initialized — run prefill first"
            ))
        })
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
