//! GDN + attention KV caches. Includes TurboQuant integration plus the
//! zero-overhead affine-quantized layout shared with `gpt_oss_native`,
//! `llama4_native`, and `deepseek_native`.

use crate::InlineArray;
use crate::inline_array as bridge;

use super::weights::NativeWeights;

// ============================================================================
// Caches
// ============================================================================

/// GDN layer cache state (conv + SSM).
pub struct GdnCache {
    pub conv_state: Option<InlineArray>,
    pub ssm_state: Option<InlineArray>,
}

/// Affine-quantized KV cache tuple: (packed_uint32, scales, biases).
///
/// Matches mlx-lm's `QuantizedKVCache` storage format. The packed data, scales,
/// and biases are passed directly to `quantized_matmul` which dequantizes inside
/// the Metal kernel — zero-overhead vs bf16 SDPA.
#[derive(Clone)]
pub struct QuantizedTuple {
    pub packed: InlineArray, // [B, H, T, D_packed] uint32
    pub scales: InlineArray, // [B, H, T, D/group_size]
    pub biases: InlineArray, // [B, H, T, D/group_size]
}

/// Mixed-bit configuration for TurboQuant v2 presets (Q2.5, Q3.5).
///
/// Splits head dimensions into outlier channels (top 25% by magnitude) and
/// regular channels, quantizing each at different bit widths. The channel
/// permutation is absorbed into projection weights at load time for zero
/// runtime overhead.
#[derive(Clone, Copy, Debug)]
pub struct MixedBitConfig {
    /// Number of outlier channels per head (quantized at higher bits).
    pub outlier_count: i32,
    /// Bit width for outlier channels (e.g., 3 for Q2.5, 4 for Q3.5).
    pub outlier_bits: u8,
    /// Bit width for regular channels (e.g., 2 for Q2.5, 3 for Q3.5).
    pub regular_bits: u8,
}

/// Configuration for zero-overhead affine KV cache quantization.
#[derive(Clone, Copy, Debug)]
pub struct QuantCacheConfig {
    pub bits: u8,
    pub group_size: i32,
    /// Mixed-bit mode (TurboQuant v2). When set, `bits` is ignored and the
    /// outlier/regular split is used instead. Channel permutation must be
    /// applied to projection weights at load time via [`apply_outlier_permutation`].
    pub mixed_bit: Option<MixedBitConfig>,
    /// QJL residual correction for keys at Q2-Q3.
    ///
    /// When true, the uniform quantized path computes a 1-bit sign vector on
    /// the quantization residual and stores it in `KvLayerCache::qjl_signs`.
    /// At SDPA time, a correction term is added to attention scores to make
    /// the inner product estimate unbiased. Only active for bits <= 3 and
    /// uniform quantization (not mixed-bit).
    pub qjl: bool,
}

/// Per-layer KV cache using pre-allocated buffers with O(1) slice_set updates.
pub struct KvLayerCache {
    pub keys: Option<InlineArray>,   // [B, H, MAX_T, D] pre-allocated
    pub values: Option<InlineArray>, // [B, H, MAX_T, D] pre-allocated
    pub offset: i32,                 // number of valid tokens
    /// TurboQuant compressed cache (replaces keys/values when enabled)
    pub turboquant: Option<crate::turboquant::QuantizedKvCache>,
    /// Zero-overhead affine-quantized cache — regular channels (lower bits)
    pub quantized_keys: Option<QuantizedTuple>,
    pub quantized_values: Option<QuantizedTuple>,
    /// Mixed-bit outlier channels (higher bits). `None` in uniform-quantization mode.
    pub quantized_keys_hi: Option<QuantizedTuple>,
    pub quantized_values_hi: Option<QuantizedTuple>,
    pub quant_config: Option<QuantCacheConfig>,
    /// QJL residual correction for key inner products (uniform Q2-Q3 only).
    ///
    /// `qjl_signs`: `[B, H, MAX_T, D]` bf16 ±1.0 — sign(S · residual).
    /// `qjl_residual_norms`: `[B, H, MAX_T, 1]` f32 — L2 norm of residual.
    ///
    /// Both `None` when QJL is disabled or cache is empty.
    pub qjl_signs: Option<InlineArray>, // [B, H, MAX_T, D] model_dtype ±1.0
    pub qjl_residual_norms: Option<InlineArray>, // [B, H, MAX_T, 1] f32
}

/// Full model cache — both GDN and KV layers.
pub struct NativeCache {
    pub gdn_caches: Vec<GdnCache>,
    pub kv_caches: Vec<KvLayerCache>,
    pub rope_offset: i32,
    /// Shared TurboQuant state (rotation matrices, codebooks) — None = bf16 cache
    pub turboquant_state: Option<std::sync::Arc<crate::turboquant::TurboQuantState>>,
}

impl NativeCache {
    /// Evaluate all cache states in-place and detach them from their computation
    /// graph.  Must be called after the prefill forward pass and before decode.
    ///
    /// Python's `generate_step` does `mx.eval([c.state for c in prompt_cache])`
    /// at this point.  Without this, the unevaluated prefill SSM states have the
    /// entire prefill graph attached; when decode builds its graph those prefill
    /// nodes are included, adding hundreds of extra AsType/Matmul/etc. nodes.
    pub fn eval_and_detach_states(&mut self) {
        // Collect all non-None state arrays into a temporary vec for batch eval.
        let mut to_eval: Vec<&mut InlineArray> = Vec::new();
        for c in &mut self.gdn_caches {
            if let Some(ref mut s) = c.ssm_state {
                to_eval.push(s);
            }
            if let Some(ref mut s) = c.conv_state {
                to_eval.push(s);
            }
        }
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
        // Batch eval then detach each.
        bridge::eval_and_detach_many(&mut to_eval);
    }

    /// Create a fresh, empty cache for the given weight set.
    pub fn new_empty(weights: &NativeWeights) -> Self {
        Self::new_with_turboquant(weights, None)
    }

    /// Create cache with optional TurboQuant KV compression.
    pub fn new_with_turboquant(
        weights: &NativeWeights,
        tq_config: Option<crate::turboquant::TurboQuantConfig>,
    ) -> Self {
        let mut gdn_caches = Vec::new();
        let mut kv_caches = Vec::new();

        // Build shared TurboQuant state if enabled
        let tq_state = tq_config.map(|cfg| {
            // Use the first attention layer's head_dim for key/value dims
            let head_dim = weights
                .layers
                .iter()
                .find(|lw| !lw.is_linear)
                .map(|lw| lw.attn_head_dim as usize)
                .unwrap_or(128);
            crate::turboquant::build_state(head_dim, head_dim, cfg)
        });

        for lw in &weights.layers {
            if lw.is_linear {
                gdn_caches.push(GdnCache {
                    conv_state: None,
                    ssm_state: None,
                });
            } else {
                let tq_cache = tq_state.as_ref().map(|state| {
                    crate::turboquant::new_cache_with_state(tq_config.unwrap(), state.clone())
                });
                kv_caches.push(KvLayerCache {
                    keys: None,
                    values: None,
                    offset: 0,
                    turboquant: tq_cache,
                    quantized_keys: None,
                    quantized_values: None,
                    quantized_keys_hi: None,
                    quantized_values_hi: None,
                    quant_config: None,
                    qjl_signs: None,
                    qjl_residual_norms: None,
                });
            }
        }

        NativeCache {
            gdn_caches,
            kv_caches,
            rope_offset: 0,
            turboquant_state: tq_state,
        }
    }

    pub(super) fn reserve_decode_inputs(&mut self, additional_tokens: i32, dtype: i32) {
        if additional_tokens <= 0 {
            return;
        }

        let mut changed_indices = Vec::new();
        for (idx, cache) in self.kv_caches.iter_mut().enumerate() {
            if cache.turboquant.is_some() {
                continue;
            }

            let Some(keys) = cache.keys.take() else {
                continue;
            };
            let Some(values) = cache.values.take() else {
                cache.keys = Some(keys);
                continue;
            };

            let current_capacity = keys.dim(2);
            let target_capacity = cache.offset + additional_tokens;
            if target_capacity <= current_capacity {
                cache.keys = Some(keys);
                cache.values = Some(values);
                continue;
            }

            let extend = target_capacity - current_capacity;
            let ext_keys =
                InlineArray::zeros(&[keys.dim(0), keys.dim(1), extend, keys.dim(3)], dtype);
            let ext_values = InlineArray::zeros(
                &[values.dim(0), values.dim(1), extend, values.dim(3)],
                dtype,
            );
            cache.keys = Some(keys.kv_cache_append(&ext_keys, 2));
            cache.values = Some(values.kv_cache_append(&ext_values, 2));
            changed_indices.push(idx);
        }

        if changed_indices.is_empty() {
            return;
        }

        let mut changed_ptrs: Vec<*mut InlineArray> = Vec::new();
        for idx in changed_indices {
            let cache = &mut self.kv_caches[idx];
            if let Some(ref mut keys) = cache.keys {
                changed_ptrs.push(keys as *mut InlineArray);
            }
            if let Some(ref mut values) = cache.values {
                changed_ptrs.push(values as *mut InlineArray);
            }
        }

        let mut to_eval: Vec<&mut InlineArray> = changed_ptrs
            .into_iter()
            .map(|ptr| unsafe { &mut *ptr })
            .collect();
        bridge::eval_and_detach_many(&mut to_eval);
    }
}

impl std::fmt::Debug for NativeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCache")
            .field("gdn_layers", &self.gdn_caches.len())
            .field("attn_layers", &self.kv_caches.len())
            .field("rope_offset", &self.rope_offset)
            .finish()
    }
}
