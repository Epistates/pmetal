//! Per-layer + full-model KV caches. Supports bf16 pre-allocated buffers,
//! the zero-overhead affine-quantized variant shared with `qwen3_native`, and
//! TurboQuant compression on every layer.

use crate::InlineArray;
use crate::inline_array as bridge;

use super::weights::NativeWeights;

/// Per-layer KV cache using pre-allocated buffers with O(1) slice_set updates.
///
/// iRoPE invariant: this cache stores the *full* KV history regardless of
/// `lw.use_rope`. Chunked attention is enforced as a *mask* during prefill
/// (see `super::attention::build_chunk_mask`), not as a cache eviction
/// policy — so global (NoPE) layers still see every prior token, which is
/// the intended Llama 4 behaviour.
#[derive(Clone)]
pub struct KvLayerCache {
    pub keys: Option<InlineArray>,   // [B, H, MAX_T, D]
    pub values: Option<InlineArray>, // [B, H, MAX_T, D]
    pub offset: i32,                 // valid tokens in the cache
    /// TurboQuant compressed cache. When set, takes precedence over the bf16 /
    /// affine paths. Chunk-mask prefill paths still dequantize+SDPA inline.
    pub turboquant: Option<crate::turboquant::QuantizedKvCache>,
    /// Zero-overhead affine-quantized cache (uniform bit width).
    pub quantized_keys: Option<crate::qwen3_native::QuantizedTuple>,
    pub quantized_values: Option<crate::qwen3_native::QuantizedTuple>,
    pub quant_config: Option<crate::qwen3_native::QuantCacheConfig>,
}

/// Full model cache.
#[derive(Clone)]
pub struct NativeCache {
    /// One entry per layer (both local and global — all are KV caches for Llama 4).
    pub kv_caches: Vec<KvLayerCache>,
    /// Global position offset (number of tokens processed so far).
    pub rope_offset: i32,
    /// Shared TurboQuant runtime state. Built once when TurboQuant is enabled.
    pub turboquant_state: Option<std::sync::Arc<crate::turboquant::TurboQuantState>>,
}

impl NativeCache {
    /// Return a cheap branch of this cache for prefix reuse.
    ///
    /// `InlineArray` and TurboQuant cache clones are MLX reference bumps, not
    /// deep copies. Mutating the fork appends new cache buffers through normal
    /// slice_set/append paths, so stored prefixes can be reused across requests
    /// without duplicating the prefill tensors up front.
    pub fn fork(&self) -> Self {
        self.clone()
    }

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
            if let Some(ref mut tq) = c.turboquant {
                tq.eval_and_detach_gpu_state();
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
        Self::new_with_caches(weights, quant_config, None)
    }

    /// Create a cache with TurboQuant compression on every layer.
    pub fn new_with_turboquant(
        weights: &NativeWeights,
        tq_config: Option<crate::turboquant::TurboQuantConfig>,
    ) -> Self {
        Self::new_with_caches(weights, None, tq_config)
    }

    fn new_with_caches(
        weights: &NativeWeights,
        quant_config: Option<crate::qwen3_native::QuantCacheConfig>,
        tq_config: Option<crate::turboquant::TurboQuantConfig>,
    ) -> Self {
        let tq_state = tq_config.map(|cfg| {
            // Llama 4 uses a uniform head_dim across all attention layers.
            let head_dim = weights
                .layers
                .first()
                .map(|lw| lw.head_dim as usize)
                .unwrap_or(128);
            crate::turboquant::build_state(head_dim, head_dim, cfg)
        });

        let kv_caches = weights
            .layers
            .iter()
            .map(|_| KvLayerCache {
                keys: None,
                values: None,
                offset: 0,
                turboquant: tq_state.as_ref().map(|state| {
                    crate::turboquant::new_cache_with_state(
                        tq_config.expect("tq_config is Some when tq_state is Some"),
                        state.clone(),
                    )
                }),
                quantized_keys: None,
                quantized_values: None,
                quant_config,
            })
            .collect();

        NativeCache {
            kv_caches,
            rope_offset: 0,
            turboquant_state: tq_state,
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
