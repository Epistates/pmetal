//! KV caches — a ring of `sliding_window` slots for sliding-window layers,
//! unbounded growth for full-attention layers (with optional zero-overhead
//! affine quantization), plus the shared `NativeCache` wrapper.

use crate::InlineArray;
use crate::inline_array as bridge;

use super::weights::NativeWeights;

/// KV cache for one attention layer.
///
/// GPT-OSS uses two kinds:
///   - Full attention: unbounded growth (chunked reallocation).
///   - Sliding attention: a ring of `window` slots, position `p` in slot
///     `p % window`, so the ring always holds the last `window` positions and
///     a token's write never moves the others. Attention over the ring needs
///     no order (keys are rotated before they are stored), only the count of
///     filled slots.
///
/// Zero-overhead affine quantization is supported for full-attention layers
/// only. TurboQuant is not: its attention kernels have no sink term.
#[derive(Clone)]
pub struct KvLayerCache {
    pub keys: Option<InlineArray>, // [B, H, MAX_T, D] (or [B, H, window, D] for sliding)
    pub values: Option<InlineArray>, // [B, H, MAX_T, D]
    pub offset: i32,               // total tokens written
    pub is_sliding: bool,
    pub window: i32, // sliding window size (ignored when is_sliding=false)
    /// Zero-overhead affine-quantized cache (full-attention layers only).
    pub quantized_keys: Option<crate::qwen3_native::QuantizedTuple>,
    pub quantized_values: Option<crate::qwen3_native::QuantizedTuple>,
    /// None on sliding-window layers or when bf16 cache is used.
    pub quant_config: Option<crate::qwen3_native::QuantCacheConfig>,
}

/// Full model cache — one KV entry per layer.
#[derive(Clone)]
pub struct NativeCache {
    pub kv_caches: Vec<KvLayerCache>,
    pub rope_offset: i32,
}

impl NativeCache {
    /// Return a cheap branch of this cache for prefix reuse.
    ///
    /// `InlineArray` clones are MLX reference bumps, not deep copies. Mutating
    /// the fork writes new cache buffers through the normal slice_set paths,
    /// so stored prefixes can be reused across requests without duplicating
    /// the prefill tensors up front.
    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Evaluate and detach all cache state arrays in one GPU submission.
    ///
    /// Must be called after the prefill forward pass and before decode.
    /// Full-attention buffers are trimmed to the tokens written; a sliding
    /// ring keeps all its slots, since the next writes land in them.
    pub fn eval_and_detach_states(&mut self) {
        let mut to_eval: Vec<&mut InlineArray> = Vec::new();
        for c in &mut self.kv_caches {
            if !c.is_sliding {
                let offset = c.offset;
                let trim = |a: InlineArray| {
                    if offset > 0 && offset < a.dim(2) {
                        a.slice(&[0, 0, 0, 0], &[a.dim(0), a.dim(1), offset, a.dim(3)])
                    } else {
                        a
                    }
                };
                c.keys = c.keys.take().map(trim);
                c.values = c.values.take().map(trim);
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

    /// Create a cache with optional affine KV quantization.
    ///
    /// Quantization applies to full-attention layers only; sliding-window
    /// layers keep their ring in the model dtype.
    pub fn new_with_quant(
        weights: &NativeWeights,
        quant_config: Option<crate::qwen3_native::QuantCacheConfig>,
    ) -> Self {
        let kv_caches = weights
            .layers
            .iter()
            .map(|lw| KvLayerCache {
                keys: None,
                values: None,
                offset: 0,
                is_sliding: lw.attn_is_sliding,
                window: lw.attn_sliding_window,
                quantized_keys: None,
                quantized_values: None,
                quant_config: if lw.attn_is_sliding {
                    None
                } else {
                    quant_config
                },
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
