//! Per-layer + full-model KV caches. Supports bf16 pre-allocated buffers and
//! the zero-overhead affine-quantized variant shared with `qwen3_native`.

use crate::InlineArray;
use crate::inline_array as bridge;

use super::weights::NativeWeights;

/// Per-layer KV cache using pre-allocated buffers with O(1) slice_set updates.
pub struct KvLayerCache {
    pub keys: Option<InlineArray>,   // [B, H, MAX_T, D]
    pub values: Option<InlineArray>, // [B, H, MAX_T, D]
    pub offset: i32,                 // valid tokens in the cache
    /// For chunked attention: how many tokens are in the "front" (trimmed portion).
    /// Python's ChunkedKVCache trims the front when the chunk fills.
    /// For decode we keep the entire sequence visible (attention_chunk_size is just
    /// a mask constraint, not an eviction policy in the native path).
    pub start_position: i32,
    /// Zero-overhead affine-quantized cache (uniform bit width).
    pub quantized_keys: Option<crate::qwen3_native::QuantizedTuple>,
    pub quantized_values: Option<crate::qwen3_native::QuantizedTuple>,
    pub quant_config: Option<crate::qwen3_native::QuantCacheConfig>,
}

/// Full model cache.
pub struct NativeCache {
    /// One entry per layer (both local and global — all are KV caches for Llama 4).
    pub kv_caches: Vec<KvLayerCache>,
    /// Global position offset (number of tokens processed so far).
    pub rope_offset: i32,
}

impl NativeCache {
    /// Evaluate and detach all cache arrays. Must be called after prefill and
    /// before decode to sever the prefill computation graph.
    pub fn eval_and_detach_states(&mut self) {
        let mut to_eval: Vec<&mut InlineArray> = Vec::new();
        for c in &mut self.kv_caches {
            if let Some(k) = c.keys.take() {
                let trimmed = if c.offset > 0 && c.offset < k.dim(2) {
                    k.slice(&[0, 0, 0, 0], &[k.dim(0), k.dim(1), c.offset, k.dim(3)])
                } else {
                    k
                };
                c.keys = Some(trimmed);
            }
            if let Some(v) = c.values.take() {
                let trimmed = if c.offset > 0 && c.offset < v.dim(2) {
                    v.slice(&[0, 0, 0, 0], &[v.dim(0), v.dim(1), c.offset, v.dim(3)])
                } else {
                    v
                };
                c.values = Some(trimmed);
            }
            if let Some(ref mut k) = c.keys {
                to_eval.push(k);
            }
            if let Some(ref mut v) = c.values {
                to_eval.push(v);
            }
        }
        bridge::eval_and_detach_many(&mut to_eval);
    }

    /// Create a fresh, empty cache for the given weight set.
    pub fn new_empty(weights: &NativeWeights) -> Self {
        Self::new_with_quant(weights, None)
    }

    /// Create a cache with optional zero-overhead affine KV quantization.
    pub fn new_with_quant(
        weights: &NativeWeights,
        quant_config: Option<crate::qwen3_native::QuantCacheConfig>,
    ) -> Self {
        let kv_caches = weights
            .layers
            .iter()
            .map(|_| KvLayerCache {
                keys: None,
                values: None,
                offset: 0,
                start_position: 0,
                quantized_keys: None,
                quantized_values: None,
                quant_config,
            })
            .collect();

        NativeCache {
            kv_caches,
            rope_offset: 0,
        }
    }
}

impl std::fmt::Debug for NativeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCache")
            .field("layers", &self.kv_caches.len())
            .field("rope_offset", &self.rope_offset)
            .finish()
    }
}
