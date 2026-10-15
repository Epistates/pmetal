//! MLA KV cache — stores compressed latent + positional key instead of full K/V.

use crate::InlineArray;
use crate::inline_array as bridge;

use super::weights::NativeWeights;

/// Per-layer MLA cache.
///
/// Instead of caching full K and V tensors, MLA caches:
/// - `kv_latent`: the RMS-normed compressed latent `[B, 1, T, kv_lora_rank]`
/// - `k_pe`: the RoPE-encoded positional component `[B, 1, T, qk_rope_head_dim]`
///
/// This is the key compression: for DeepSeek V3 with kv_lora_rank=512 and
/// qk_rope_head_dim=64, the cache is 576 values per token vs
/// n_heads*(qk_nope_head_dim+v_head_dim) = 128*(128+128) = 32768 values for
/// standard MHA — a 56x reduction.
///
/// When `quant_config` is set, `kv_latent` and `k_pe` are stored in quantized
/// form using affine group quantization, halving (or further reducing) the cache
/// memory. Dequantization happens at attention time before SDPA.
#[derive(Clone)]
pub struct MlaLayerCache {
    /// Cached KV latent: [B, 1, T, kv_lora_rank]. Initialized lazily.
    pub kv_latent: Option<InlineArray>,
    /// Cached positional K: [B, 1, T, qk_rope_head_dim]. Initialized lazily.
    pub k_pe: Option<InlineArray>,
    /// Number of valid tokens stored.
    pub offset: i32,
    /// Quantized form of kv_latent (used when quant_config is set).
    pub quantized_latent: Option<crate::qwen3_native::QuantizedTuple>,
    /// Quantized form of k_pe (used when quant_config is set).
    pub quantized_k_pe: Option<crate::qwen3_native::QuantizedTuple>,
    /// Affine quantization config; None = bf16 (standard) path.
    pub quant_config: Option<crate::qwen3_native::QuantCacheConfig>,
}

/// Full model MLA cache — one entry per layer.
#[derive(Clone)]
pub struct NativeCache {
    pub mla_caches: Vec<MlaLayerCache>,
    /// Global position counter for RoPE offset.
    pub rope_offset: i32,
}

impl NativeCache {
    /// Return a cheap branch of this cache for prefix reuse.
    ///
    /// `InlineArray` clones are MLX reference bumps, not deep copies. Mutating
    /// the fork appends new cache buffers through normal slice_set paths, so
    /// stored prefixes can be reused across requests without duplicating the
    /// prefill tensors up front.
    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Create a fresh empty cache for the given weight set.
    pub fn new_empty(weights: &NativeWeights) -> Self {
        Self::new_with_quant(weights, None)
    }

    /// Create a cache with optional affine KV quantization.
    ///
    /// MLA already achieves a 56x compression ratio vs standard MHA by caching
    /// a latent vector rather than full K/V tensors. Quantization provides
    /// further memory reduction at the cost of a dequantization step before SDPA.
    pub fn new_with_quant(
        weights: &NativeWeights,
        quant_config: Option<crate::qwen3_native::QuantCacheConfig>,
    ) -> Self {
        let mla_caches = (0..weights.layers.len())
            .map(|_| MlaLayerCache {
                kv_latent: None,
                k_pe: None,
                offset: 0,
                quantized_latent: None,
                quantized_k_pe: None,
                quant_config,
            })
            .collect();
        NativeCache {
            mla_caches,
            rope_offset: 0,
        }
    }

    /// Evaluate and detach all cache states. Call after prefill before decode.
    pub fn eval_and_detach_states(&mut self) {
        let mut to_eval: Vec<&mut InlineArray> = Vec::new();
        for c in &mut self.mla_caches {
            if let Some(kv) = c.kv_latent.take() {
                let trimmed = if c.offset > 0 && c.offset < kv.dim(2) {
                    kv.slice(&[0, 0, 0, 0], &[kv.dim(0), kv.dim(1), c.offset, kv.dim(3)])
                } else {
                    kv
                };
                c.kv_latent = Some(trimmed);
            }
            if let Some(kp) = c.k_pe.take() {
                let trimmed = if c.offset > 0 && c.offset < kp.dim(2) {
                    kp.slice(&[0, 0, 0, 0], &[kp.dim(0), kp.dim(1), c.offset, kp.dim(3)])
                } else {
                    kp
                };
                c.k_pe = Some(trimmed);
            }
            if let Some(ref mut kv) = c.kv_latent {
                to_eval.push(kv);
            }
            if let Some(ref mut kp) = c.k_pe {
                to_eval.push(kp);
            }
        }
        bridge::eval_and_detach_many(&mut to_eval);
    }
}

impl std::fmt::Debug for NativeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCache")
            .field("layers", &self.mla_caches.len())
            .field("rope_offset", &self.rope_offset)
            .finish()
    }
}
