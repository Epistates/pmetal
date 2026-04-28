//! [`QuantizedKvCache`] — the per-layer state machine.
//!
//! Owns the hot-fp16 ring + cold-compressed stores and orchestrates encode,
//! eviction, dequantize, and the fused-attention dispatch decision tree. The
//! GPU score and dequantize kernels live in [`super::dispatch`]; the host
//! encode/decode primitives live in [`super::encode`].

use std::sync::Arc;

use crate::InlineArray;

use super::config::{HOT_EVICTION_CHUNK, TurboQuantConfig, TurboQuantTensorConfig};
use super::dispatch::{
    gpu_dequantize_keys, gpu_dequantize_keys_mixed, gpu_dequantize_values,
    gpu_dequantize_values_mixed, gpu_quantize_kv, gpu_quantize_kv_mixed,
};
use super::encode::{decode_key_rows, decode_value_rows, encode_key_rows, encode_value_rows};
use super::host_keystore::{QuantizedKeyStore, QuantizedValueStore};
use super::math::{f32_rows_to_bhsd_array, inline_array_to_bshd_rows};
use super::state::{TensorRuntime, TurboQuantState};
use super::{
    eval_stage_micros, trace_turboquant_bridge, turboquant_q8_fullbyte_enabled,
    turboquant_trace_enabled,
};


/// Compressed KV cache for one attention layer.
///
/// Stores all cached positions as bit-packed indices + f32 metadata.
/// Backed by [`TurboQuantState`] for dequantisation.
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    /// Compressed keys — inner-product optimised (MSE + QJL).
    pub keys: Option<QuantizedKeyStore>,
    /// Compressed values — MSE optimised.
    pub values: Option<QuantizedValueStore>,
    /// Layout from the first append (batch, heads, key_dim, value_dim).
    layout: Option<CacheLayout>,
    /// Total cached positions (cold + hot). Public for compatibility with
    /// callers that still read it directly. Maintained as the invariant
    /// `offset == cold_offset + hot_offset` after every append/rollback.
    pub offset: usize,
    /// Tokens currently sitting compressed in the cold stores. The GPU
    /// uniform attention kernels score against `cold_offset` slots.
    cold_offset: usize,
    /// Tokens currently held uncompressed in the fp16/bf16 hot ring.
    hot_offset: usize,
    /// Hot-ring keys, shape `[B, H_kv, hot_capacity, D_k]`. Native dtype
    /// (whatever was passed in on first append). `None` when the recent
    /// window is disabled or no tokens are currently uncompressed.
    pub(super) hot_keys: Option<InlineArray>,
    /// Hot-ring values, same shape semantics as `hot_keys`.
    pub(super) hot_values: Option<InlineArray>,
    /// Native dtype of the hot ring (taken from the first append). When the
    /// cold side is dequantized for the mixed-attention path, the cold
    /// f32/bf16 output is cast to this dtype before concatenation.
    hot_dtype: Option<i32>,
    /// Config used to build this cache.
    pub config: TurboQuantConfig,
    /// Shared pre-computed matrices and codebooks.
    pub state: Option<Arc<TurboQuantState>>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CacheLayout {
    batch: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum UniformAttentionBenchMode {
    Split,
    SpecializedQ8D128TwoPass,
    SpecializedQ8D256TwoPass,
    SpecializedQ8D256FullbytePass1,
    SpecializedQ8D256FullbytePass2,
    SpecializedQ8D256FullbyteSplitDenseV,
    SpecializedQ8D256FullbyteLocalSoftmax,
}

impl QuantizedKvCache {
    /// Create an empty cache.  `state` should be `None` on first use; call
    /// `append` to populate.
    pub fn new(config: TurboQuantConfig) -> Self {
        Self {
            keys: None,
            values: None,
            layout: None,
            offset: 0,
            cold_offset: 0,
            hot_offset: 0,
            hot_keys: None,
            hot_values: None,
            hot_dtype: None,
            config,
            state: None,
        }
    }

    /// Create with a pre-built shared state (avoids re-building QR/Lloyd-Max).
    pub fn with_state(config: TurboQuantConfig, state: Arc<TurboQuantState>) -> Self {
        let mut cache = Self::new(config);
        cache.state = Some(state);
        cache
    }

    /// Current number of cached sequence positions.
    pub fn len(&self) -> usize {
        self.offset
    }

    /// True when no positions have been cached yet.
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Reset to empty (retains pre-built state and config).
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.layout = None;
        self.hot_keys = None;
        self.hot_values = None;
        self.hot_dtype = None;
        self.offset = 0;
        self.cold_offset = 0;
        self.hot_offset = 0;
    }

    /// Number of tokens currently held uncompressed in the hot ring.
    pub fn hot_len(&self) -> usize {
        self.hot_offset
    }

    /// Number of tokens that have been compressed into the cold stores.
    pub fn cold_len(&self) -> usize {
        self.cold_offset
    }

    /// Hot-ring capacity = `recent_window + HOT_EVICTION_CHUNK` when the
    /// window is enabled, `0` when disabled (legacy compress-everything mode).
    fn hot_capacity(&self) -> usize {
        self.config
            .recent_window
            .map(|w| w + HOT_EVICTION_CHUNK)
            .unwrap_or(0)
    }

    /// Append new keys and values.
    ///
    /// `keys` and `values` must have shape `[B, H, S, D]` as f32 or bf16.
    ///
    /// For the Uniform quantisation config the entire pipeline runs on-GPU:
    /// normalise → rotate → argmin codebook → QJL projection → sign.
    /// No GPU→CPU transfer happens.  Results are stored as `InlineArray`s and
    /// concatenated along the T axis on subsequent calls.
    ///
    /// For the Mixed (outlier-aware) config the CPU path is used (outlier mask
    /// selection requires a per-row top-k sort that is not trivially vectorisable).
    ///
    /// Returns an error string on shape mismatch.
    ///
    /// Dispatches to one of two paths based on `config.recent_window`:
    /// - `None` (legacy): every appended token is compressed immediately.
    /// - `Some(N)`: the newest `N` tokens stay in an uncompressed fp16/bf16
    ///   hot ring; older history is evicted into the cold compressed stores
    ///   in `HOT_EVICTION_CHUNK` batches. Short-context callers therefore
    ///   pay zero compression overhead.
    pub fn append(&mut self, keys: &InlineArray, values: &InlineArray) -> Result<(), String> {
        let layout = self.ensure_layout(keys, values)?;
        let seq_len = keys.dim(2) as usize;
        if seq_len == 0 {
            return Ok(());
        }

        match self.config.recent_window {
            None => {
                self.compress_into_cold(keys, values, layout, seq_len)?;
                self.offset = self.cold_offset + self.hot_offset;
                Ok(())
            }
            Some(window) => self.append_with_recent_window(keys, values, layout, seq_len, window),
        }
    }

    /// Compress and accumulate `keys`/`values` (shape `[B, H, S, D]`) directly
    /// into the cold stores, advancing `self.cold_offset` by `seq_len`.
    /// Mirrors the legacy compress-immediately path.
    fn compress_into_cold(
        &mut self,
        keys: &InlineArray,
        values: &InlineArray,
        layout: CacheLayout,
        seq_len: usize,
    ) -> Result<(), String> {
        let config = self.config;
        let state = self.state.get_or_insert_with(|| {
            Arc::new(TurboQuantState::new(
                layout.key_dim,
                layout.value_dim,
                config,
            ))
        });
        let state = Arc::clone(state);

        // Cast to f32 once — needed for both GPU and CPU paths.
        let keys_f32 = keys.as_dtype(10 /* float32 */);
        let values_f32 = values.as_dtype(10 /* float32 */);

        let ks = self
            .keys
            .get_or_insert_with(|| QuantizedKeyStore::new(config.keys, config.qjl));
        let vs = self
            .values
            .get_or_insert_with(|| QuantizedValueStore::new(config.values));

        // ── GPU path (Uniform only) ───────────────────────────────────────
        let gpu_keys_ok = matches!(config.keys, TurboQuantTensorConfig::Uniform { .. });
        let gpu_vals_ok = matches!(config.values, TurboQuantTensorConfig::Uniform { .. });

        if gpu_keys_ok && gpu_vals_ok {
            if let Some((new_ks_gpu, new_vs_gpu)) =
                gpu_quantize_kv(&state, &keys_f32, &values_f32, config)
            {
                match ks.gpu.as_mut() {
                    None => ks.gpu = Some(new_ks_gpu),
                    Some(g) => g.append(new_ks_gpu),
                }
                match vs.gpu.as_mut() {
                    None => vs.gpu = Some(new_vs_gpu),
                    Some(g) => g.append(new_vs_gpu),
                }
                self.cold_offset += seq_len;
                return Ok(());
            }
            // GPU path failed — fall through to CPU. But CPU encode does
            // NOT mirror the Phase E.3 outlier override (the host
            // encode_key_component_rows has no notion of per-block
            // outliers). Silently falling through would produce broken
            // cache state — error explicitly so the caller can either
            // disable outliers or pin a GPU-supported config.
            if config.outliers.is_enabled() {
                return Err(
                    "TurboQuant outliers (PerBlock) require GPU encode; \
                     CPU fallback does not mirror the override. Disable \
                     outliers or use a GPU-supported config (Uniform + \
                     codebook available)."
                        .to_string(),
                );
            }
        }

        // ── CPU fallback path ─────────────────────────────────────────────
        let key_rows = inline_array_to_bshd_rows(&keys_f32)?;
        let value_rows = inline_array_to_bshd_rows(&values_f32)?;

        let rows_per_seq = layout.batch * layout.heads;
        debug_assert_eq!(key_rows.len(), rows_per_seq * seq_len * layout.key_dim);

        let encoded_keys = encode_key_rows(&state.keys, layout.key_dim, &key_rows, state.qjl);
        let encoded_values = encode_value_rows(&state.values, layout.value_dim, &value_rows);

        ks.extend(
            &encoded_keys.regular,
            encoded_keys.outlier.as_ref(),
            encoded_keys.outlier_mask.as_ref(),
        );
        vs.extend(
            &encoded_values.regular,
            encoded_values.outlier.as_ref(),
            encoded_values.outlier_mask.as_ref(),
        );

        // ── Mixed GPU path ────────────────────────────────────────────────
        // Mirrors the Uniform GPU path above but populates gpu_mixed instead
        // of gpu, alongside (not in lieu of) the CPU PackedBits stores. The
        // CPU stores remain the authoritative source for the score kernel
        // (which still reads PackedBits); the GPU store is consumed by the
        // cold dequantize path so reconstruction never re-uploads via CPU.
        let mixed_keys = matches!(config.keys, TurboQuantTensorConfig::Mixed { .. });
        let mixed_vals = matches!(config.values, TurboQuantTensorConfig::Mixed { .. });
        if mixed_keys && mixed_vals {
            if let Some((new_ks_gpu, new_vs_gpu)) =
                gpu_quantize_kv_mixed(&state, &keys_f32, &values_f32, config)
            {
                match ks.gpu_mixed.as_mut() {
                    None => ks.gpu_mixed = Some(new_ks_gpu),
                    Some(g) => g.append(new_ks_gpu),
                }
                match vs.gpu_mixed.as_mut() {
                    None => vs.gpu_mixed = Some(new_vs_gpu),
                    Some(g) => g.append(new_vs_gpu),
                }
            }
        }

        self.cold_offset += seq_len;
        Ok(())
    }

    /// Append into the hot fp16/bf16 ring, evicting oldest tokens to the cold
    /// stores once the ring exceeds `window + HOT_EVICTION_CHUNK`.
    fn append_with_recent_window(
        &mut self,
        keys: &InlineArray,
        values: &InlineArray,
        layout: CacheLayout,
        seq_len: usize,
        window: usize,
    ) -> Result<(), String> {
        // Lock in the hot dtype on first append. All subsequent writes go
        // through `as_dtype` so the ring always carries a single dtype.
        if self.hot_dtype.is_none() {
            self.hot_dtype = Some(keys.dtype_raw());
        }
        let hot_dtype = self.hot_dtype.unwrap();
        let capacity = self.hot_capacity().max(seq_len);

        // Lazy-allocate (or grow) the hot ring.
        if self.hot_keys.is_none() {
            self.hot_keys = Some(InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    capacity as i32,
                    layout.key_dim as i32,
                ],
                hot_dtype,
            ));
            self.hot_values = Some(InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    capacity as i32,
                    layout.value_dim as i32,
                ],
                hot_dtype,
            ));
        } else if self.hot_offset + seq_len > capacity {
            // One-shot prefill larger than the current ring — grow to fit.
            let need = self.hot_offset + seq_len;
            let new_cap = need.max(capacity);
            let prev_keys = self.hot_keys.take().unwrap();
            let prev_values = self.hot_values.take().unwrap();
            let mut new_keys = InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    new_cap as i32,
                    layout.key_dim as i32,
                ],
                hot_dtype,
            );
            let mut new_values = InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    new_cap as i32,
                    layout.value_dim as i32,
                ],
                hot_dtype,
            );
            if self.hot_offset > 0 {
                let kept_keys = prev_keys.slice(
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.key_dim as i32,
                    ],
                );
                let kept_values = prev_values.slice(
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.value_dim as i32,
                    ],
                );
                new_keys = new_keys.slice_set(
                    &kept_keys,
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.key_dim as i32,
                    ],
                );
                new_values = new_values.slice_set(
                    &kept_values,
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.value_dim as i32,
                    ],
                );
            }
            self.hot_keys = Some(new_keys);
            self.hot_values = Some(new_values);
        }

        // Write the new chunk into the ring at `hot_offset..hot_offset+seq_len`.
        let start = self.hot_offset;
        let stop = start + seq_len;
        let keys_typed = keys.as_dtype(hot_dtype);
        let values_typed = values.as_dtype(hot_dtype);
        {
            let hot_keys = self.hot_keys.as_mut().expect("hot_keys allocated above");
            *hot_keys = hot_keys.slice_set(
                &keys_typed,
                &[0, 0, start as i32, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    stop as i32,
                    layout.key_dim as i32,
                ],
            );
        }
        {
            let hot_values = self.hot_values.as_mut().expect("hot_values allocated above");
            *hot_values = hot_values.slice_set(
                &values_typed,
                &[0, 0, start as i32, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    stop as i32,
                    layout.value_dim as i32,
                ],
            );
        }
        self.hot_offset += seq_len;
        self.offset = self.cold_offset + self.hot_offset;

        // Drain the oldest tokens once the ring fills past `window + chunk`.
        // Eviction is amortized: each call moves `min(overflow, chunk)` tokens
        // instead of one-token-at-a-time shuffles.
        while self.hot_offset > window + HOT_EVICTION_CHUNK {
            let evict_seq = self
                .hot_offset
                .saturating_sub(window)
                .min(HOT_EVICTION_CHUNK);
            self.evict_oldest_to_cold(layout, evict_seq)?;
        }
        Ok(())
    }

    /// Move the leading `evict_seq` tokens out of the hot ring into cold,
    /// then shuffle the remaining tail back to position 0 of the buffer.
    /// Caller must guarantee `evict_seq <= self.hot_offset`.
    fn evict_oldest_to_cold(
        &mut self,
        layout: CacheLayout,
        evict_seq: usize,
    ) -> Result<(), String> {
        if evict_seq == 0 {
            return Ok(());
        }
        let remain = self.hot_offset - evict_seq;
        let hot_dtype = self
            .hot_dtype
            .expect("hot_dtype set when hot ring is non-empty");

        // Phase 1: extract the leading slice we want to compress and the
        // trailing slice we want to keep, into owned values. The immutable
        // borrows of `hot_keys`/`hot_values` are dropped at the end of this
        // block before any `&mut self` call below.
        let (evict_keys, evict_values, kept) = {
            let hot_keys = self.hot_keys.as_ref().ok_or_else(|| {
                "TurboQuant hot keys missing during evict".to_string()
            })?;
            let hot_values = self.hot_values.as_ref().ok_or_else(|| {
                "TurboQuant hot values missing during evict".to_string()
            })?;
            let evict_keys = hot_keys.slice(
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    evict_seq as i32,
                    layout.key_dim as i32,
                ],
            );
            let evict_values = hot_values.slice(
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    evict_seq as i32,
                    layout.value_dim as i32,
                ],
            );
            let kept = if remain > 0 {
                let kept_keys = hot_keys.slice(
                    &[0, 0, evict_seq as i32, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.key_dim as i32,
                    ],
                );
                let kept_values = hot_values.slice(
                    &[0, 0, evict_seq as i32, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.value_dim as i32,
                    ],
                );
                Some((kept_keys, kept_values))
            } else {
                None
            };
            (evict_keys, evict_values, kept)
        };

        // Phase 2: mutate. The borrows above are dropped.
        self.compress_into_cold(&evict_keys, &evict_values, layout, evict_seq)?;
        if let Some((kept_keys, kept_values)) = kept {
            let capacity = self.hot_capacity().max(remain);
            let mut new_keys = InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    capacity as i32,
                    layout.key_dim as i32,
                ],
                hot_dtype,
            );
            let mut new_values = InlineArray::zeros(
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    capacity as i32,
                    layout.value_dim as i32,
                ],
                hot_dtype,
            );
            new_keys = new_keys.slice_set(
                &kept_keys,
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    remain as i32,
                    layout.key_dim as i32,
                ],
            );
            new_values = new_values.slice_set(
                &kept_values,
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    remain as i32,
                    layout.value_dim as i32,
                ],
            );
            self.hot_keys = Some(new_keys);
            self.hot_values = Some(new_values);
        } else {
            // Hot ring fully drained — drop the buffers until the next append.
            self.hot_keys = None;
            self.hot_values = None;
        }
        self.hot_offset = remain;
        self.offset = self.cold_offset + self.hot_offset;
        Ok(())
    }

    /// Dequantise and return all cached keys as an `InlineArray` of shape
    /// `[B, H, T, D]` (f32). The returned tensor includes both the compressed
    /// cold history and the uncompressed hot tail (the latter cast to f32).
    ///
    /// Uses the GPU path when keys were quantised on-GPU; otherwise falls back
    /// to the CPU decode path.
    pub fn dequantize_keys(&self) -> Option<InlineArray> {
        let layout = self.layout?;

        let cold = if self.cold_offset > 0 {
            let ks = self.keys.as_ref()?;
            let state = self.state.as_ref()?;
            if let Some(ref g) = ks.gpu {
                let TurboQuantTensorConfig::Uniform { bits } = self.config.keys else {
                    unreachable!("Uniform GpuKeyStore only exists for Uniform config")
                };
                Some(gpu_dequantize_keys(g, &state.keys, bits, state.qjl)?)
            } else if let Some(ref g) = ks.gpu_mixed {
                // Phase 3c: GPU-side Mixed dequantize avoids the CPU→GPU
                // upload that dominated the dequantize+SDPA fallback. The
                // result is `[B, H, T, D]` f32 already on device, ready for
                // SDPA. Parity is gated by the round-trip test (see
                // `turboquant_gpu_mixed_storage_round_trip_matches_cpu_dequantize`).
                Some(gpu_dequantize_keys_mixed(g, &state.keys, &self.config)?)
            } else {
                let rows = decode_key_rows(&state.keys, layout.key_dim, ks, state.qjl);
                Some(f32_rows_to_bhsd_array(
                    &rows,
                    layout.batch,
                    layout.heads,
                    self.cold_offset,
                    layout.key_dim,
                ))
            }
        } else {
            None
        };

        let hot = if self.hot_offset > 0 {
            let hot_keys = self.hot_keys.as_ref()?;
            let active = hot_keys.slice(
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    self.hot_offset as i32,
                    layout.key_dim as i32,
                ],
            );
            // Cold is f32; cast hot to f32 so concat dtypes match.
            Some(active.as_dtype(10))
        } else {
            None
        };

        let mut combined = match (cold, hot) {
            (Some(c), Some(h)) => c.concatenate_2(&h, 2),
            (Some(c), None) => c,
            (None, Some(h)) => h,
            (None, None) => return None,
        };
        let mut to_eval = vec![&mut combined];
        crate::inline_array::eval_and_detach_many(&mut to_eval);
        Some(combined)
    }

    /// Dequantise and return all cached values as an `InlineArray` of shape
    /// `[B, H, T, D]` (f32). Includes both compressed cold history and the
    /// uncompressed hot tail (cast to f32).
    pub fn dequantize_values(&self) -> Option<InlineArray> {
        let layout = self.layout?;

        let cold = if self.cold_offset > 0 {
            let vs = self.values.as_ref()?;
            let state = self.state.as_ref()?;
            if let Some(ref g) = vs.gpu {
                let TurboQuantTensorConfig::Uniform { bits } = self.config.values else {
                    unreachable!("Uniform GpuValueStore only exists for Uniform config")
                };
                Some(gpu_dequantize_values(g, &state.values, bits)?)
            } else if let Some(ref g) = vs.gpu_mixed {
                // Phase 3c: see `dequantize_keys` for the rationale.
                Some(gpu_dequantize_values_mixed(g, &state.values, &self.config)?)
            } else {
                let rows = decode_value_rows(&state.values, layout.value_dim, vs);
                Some(f32_rows_to_bhsd_array(
                    &rows,
                    layout.batch,
                    layout.heads,
                    self.cold_offset,
                    layout.value_dim,
                ))
            }
        } else {
            None
        };

        let hot = if self.hot_offset > 0 {
            let hot_values = self.hot_values.as_ref()?;
            let active = hot_values.slice(
                &[0, 0, 0, 0],
                &[
                    layout.batch as i32,
                    layout.heads as i32,
                    self.hot_offset as i32,
                    layout.value_dim as i32,
                ],
            );
            Some(active.as_dtype(10))
        } else {
            None
        };

        let mut combined = match (cold, hot) {
            (Some(c), Some(h)) => c.concatenate_2(&h, 2),
            (Some(c), None) => c,
            (None, Some(h)) => h,
            (None, None) => return None,
        };
        let mut to_eval = vec![&mut combined];
        crate::inline_array::eval_and_detach_many(&mut to_eval);
        Some(combined)
    }

    /// Evaluate and detach GPU-resident cache arrays to keep graph chains short.
    pub fn eval_and_detach_gpu_state(&mut self) {
        let mut to_eval = Vec::new();
        if let Some(keys) = &mut self.keys {
            if let Some(gpu) = &mut keys.gpu {
                gpu.collect_for_detach(&mut to_eval);
            }
            if let Some(gpu) = &mut keys.gpu_mixed {
                gpu.collect_for_detach(&mut to_eval);
            }
        }
        if let Some(values) = &mut self.values {
            if let Some(gpu) = &mut values.gpu {
                gpu.collect_for_detach(&mut to_eval);
            }
            if let Some(gpu) = &mut values.gpu_mixed {
                gpu.collect_for_detach(&mut to_eval);
            }
        }
        if let Some(hot) = self.hot_keys.as_mut() {
            to_eval.push(hot);
        }
        if let Some(hot) = self.hot_values.as_mut() {
            to_eval.push(hot);
        }
        if !to_eval.is_empty() {
            crate::inline_array::eval_and_detach_many(&mut to_eval);
        }
    }

    /// Append a single-token KV chunk and compute attention output.
    ///
    /// Dispatch:
    /// - **Hot-only** (`cold_offset == 0`, common for short prompts when the
    ///   recent fp16 window is enabled): run standard SDPA against the active
    ///   prefix of the hot ring. No quantization round-trip on the decode path.
    /// - **Cold-only** (`hot_offset == 0`, recent window disabled or fully
    ///   evicted): try the optimized GPU TurboQuant attention kernels.
    /// - **Mixed**: dequantize the cold cache, concat with the active hot
    ///   suffix, run standard SDPA. Correctness-first; a future v2 may
    ///   score hot directly + cold compressedly in one kernel.
    pub fn append_and_compute_attention(
        &mut self,
        queries: &InlineArray,
        keys: &InlineArray,
        values: &InlineArray,
        scale: f32,
    ) -> Result<InlineArray, String> {
        if queries.ndim() != 4
            || keys.ndim() != 4
            || values.ndim() != 4
            || queries.dim(2) != 1
            || keys.dim(2) != 1
            || values.dim(2) != 1
        {
            return Err(
                "TurboQuant direct attention requires [B, H, 1, D] single-token decode inputs"
                    .to_string(),
            );
        }

        let layout = self.ensure_layout(keys, values)?;
        self.append(keys, values)?;
        let query_dtype = queries.dtype_raw();
        let queries_f32 = if query_dtype == 10 {
            queries.clone()
        } else {
            queries.as_dtype(10)
        };

        // Hot-only: standard fused SDPA against the fp16/bf16 hot ring.
        // Skips both the GPU TurboQuant kernels and any dequantize allocation.
        if self.cold_offset == 0 && self.hot_offset > 0 {
            let hot_keys = self
                .hot_keys
                .as_ref()
                .ok_or_else(|| "TurboQuant hot keys missing".to_string())?;
            let hot_values = self
                .hot_values
                .as_ref()
                .ok_or_else(|| "TurboQuant hot values missing".to_string())?;
            let active_keys = hot_keys
                .slice(
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.key_dim as i32,
                    ],
                )
                .as_dtype(10);
            let active_values = hot_values
                .slice(
                    &[0, 0, 0, 0],
                    &[
                        layout.batch as i32,
                        layout.heads as i32,
                        self.hot_offset as i32,
                        layout.value_dim as i32,
                    ],
                )
                .as_dtype(10);

            let q_heads = queries.dim(1) as usize;
            let kv_heads = layout.heads;
            let (keys_for_attn, values_for_attn) = if q_heads == kv_heads {
                (active_keys, active_values)
            } else {
                let groups = q_heads / kv_heads;
                if groups * kv_heads != q_heads {
                    return Err(format!(
                        "TurboQuant GQA mismatch: query heads {q_heads} not divisible by kv heads {kv_heads}"
                    ));
                }
                (
                    active_keys.repeat(groups as i32, 1),
                    active_values.repeat(groups as i32, 1),
                )
            };
            let output = crate::decode::sdpa_causal_like_mlx(
                &queries_f32,
                &keys_for_attn,
                &values_for_attn,
                scale,
                queries.dim(2),
            );
            return Ok(if query_dtype == 10 {
                output
            } else {
                output.as_dtype(query_dtype)
            });
        }

        // Phase F: Hamming skip-list pre-filter for very long contexts.
        // Gated on `config.skiplist_threshold`; the encode path only allocates
        // the sign-hash buffer when the threshold is set, so a missing buffer
        // here means the user opted out and we fall through to the dense path.
        // V1 supports cold-only Variant F (NoQjl) without GQA expansion;
        // mixed/standard/GQA cases fall through to the existing dispatch.
        if self.hot_offset == 0 {
            if let Some(output) =
                self.try_hamming_skiplist_attention(&queries_f32, layout, scale, query_dtype)
            {
                return Ok(if query_dtype == 10 || output.dtype_raw() == query_dtype {
                    output
                } else {
                    output.as_dtype(query_dtype)
                });
            }
        }

        // Cold-only: optimized GPU TurboQuant kernels. The score-against-cold
        // assumption inside `try_gpu_uniform_attention` is only correct when
        // the hot ring is empty.
        if self.hot_offset == 0 {
            if let Some(output) =
                self.try_gpu_uniform_attention(&queries_f32, layout, scale, query_dtype)
            {
                return Ok(if query_dtype == 10 || output.dtype_raw() == query_dtype {
                    output
                } else {
                    output.as_dtype(query_dtype)
                });
            }
        }

        // Mixed (or cold-only fallback): dequantize the full cache (cold +
        // hot tail) and run standard SDPA. dequantize_keys/values returns
        // concat(cold_f32, hot_f32) automatically.
        let full_keys = self
            .dequantize_keys()
            .ok_or_else(|| "TurboQuant failed to dequantize keys".to_string())?;
        let full_values = self
            .dequantize_values()
            .ok_or_else(|| "TurboQuant failed to dequantize values".to_string())?;

        let q_heads = queries.dim(1) as usize;
        let kv_heads = layout.heads;
        let (keys_for_attn, values_for_attn) = if q_heads == kv_heads {
            (full_keys, full_values)
        } else {
            let groups = q_heads / kv_heads;
            if groups * kv_heads != q_heads {
                return Err(format!(
                    "TurboQuant GQA mismatch: query heads {q_heads} not divisible by kv heads {kv_heads}"
                ));
            }
            (
                full_keys.repeat(groups as i32, 1),
                full_values.repeat(groups as i32, 1),
            )
        };

        let queries_f32 = queries.as_dtype(10);
        let output = crate::decode::sdpa_causal_like_mlx(
            &queries_f32,
            &keys_for_attn,
            &values_for_attn,
            scale,
            queries.dim(2),
        );
        Ok(if queries.dtype_raw() == 10 {
            output
        } else {
            output.as_dtype(queries.dtype_raw())
        })
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_attention_core_precomputed(
        &self,
        query_rot: &InlineArray,
        query_proj: &InlineArray,
        q_heads: i32,
        scale: f32,
        mode: UniformAttentionBenchMode,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;

        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };

        let key_dim = layout.key_dim as i32;
        let value_dim = layout.value_dim as i32;
        let kv_heads_i32 = layout.heads as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if q_rows <= 0 || n_seq <= 0 || cache_seq_capacity < n_seq || kv_heads_i32 <= 0 {
            return None;
        }

        let kv_rows = (layout.batch * layout.heads) as i32;
        let key_norms = ks
            .key_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_residual_norms = ks
            .residual_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_slot_scale = ks
            .key_slot_scale_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let qjl_words = ks.qjl_words();

        match mode {
            UniformAttentionBenchMode::SpecializedQ8D128TwoPass => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 128
                    || value_dim != 128
                    || qjl_words != 4
                {
                    return None;
                }
                let key_indices =
                    ks.indices_t_array()
                        .reshape(&[kv_rows, key_dim, cache_seq_capacity]);
                let key_qjl_signs = ks
                    .qjl_signs_t_array()?
                    .reshape(&[kv_rows, qjl_words, cache_seq_capacity]);
                let value_indices =
                    vs.indices_t_array()?
                        .reshape(&[kv_rows, value_dim, cache_seq_capacity]);
                InlineArray::turboquant_attention_q8_d128_2pass(
                    query_rot,
                    query_proj,
                    &key_indices,
                    &key_qjl_signs,
                    &key_norms,
                    &key_residual_norms,
                    &key_slot_scale,
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &value_indices,
                    &vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]),
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            }
            UniformAttentionBenchMode::SpecializedQ8D256TwoPass => self
                .try_gpu_uniform_attention_q8_d256_precomputed(
                    query_rot,
                    Some(query_proj),
                    q_heads,
                    scale,
                ),
            UniformAttentionBenchMode::SpecializedQ8D256FullbytePass1 => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
                    ks.q8_fullbyte_seq.as_ref(),
                    ks.q8_slot_scales_seq.as_ref(),
                    vs.d256_rot_values_seq.as_ref(),
                ) {
                    InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass_pass1(
                        query_rot,
                        &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 4]),
                        key_core.codebook_arr(key_bits)?,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                } else {
                    None
                }
            }
            UniformAttentionBenchMode::SpecializedQ8D256FullbytePass2 => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                let (partials, sums, maxs) = self
                    .bench_gpu_uniform_attention_state_precomputed_fullbyte(
                        query_rot, q_heads, scale,
                    )?;
                InlineArray::turboquant_attention_q8_d256_pass2_merge(
                    &partials,
                    &sums,
                    &maxs,
                    q_rows as u32,
                    sums.dim(1) as u32,
                )
            }
            UniformAttentionBenchMode::SpecializedQ8D256FullbyteSplitDenseV => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
                    ks.q8_fullbyte_seq.as_ref(),
                    ks.q8_slot_scales_seq.as_ref(),
                    vs.d256_rot_values_seq.as_ref(),
                ) {
                    let scores = InlineArray::turboquant_score_q8_d256_fullbyte(
                        query_rot,
                        &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 4]),
                        key_core.codebook_arr(key_bits)?,
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )?;
                    let weights = scores.softmax(-1);
                    InlineArray::turboquant_weighted_sum_d256_dense_values(
                        &weights,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                    )
                } else {
                    None
                }
            }
            UniformAttentionBenchMode::SpecializedQ8D256FullbyteLocalSoftmax => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
                    ks.q8_fullbyte_seq.as_ref(),
                    ks.q8_slot_scales_seq.as_ref(),
                    vs.d256_rot_values_seq.as_ref(),
                ) {
                    InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass_localsoftmax(
                        query_rot,
                        &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 4]),
                        key_core.codebook_arr(key_bits)?,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                } else {
                    None
                }
            }
            UniformAttentionBenchMode::Split => {
                let scores = self
                    .bench_gpu_uniform_scores_precomputed(query_rot, query_proj, q_heads, scale)?;
                let weights = scores.softmax(-1);
                InlineArray::turboquant_weighted_decode(
                    &weights,
                    &vs.indices_t_array()?,
                    &vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]),
                    value_core.codebook_arr(value_bits)?,
                    value_dim as u32,
                    1u32 << value_bits,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                )
            }
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_query_transforms(
        &self,
        queries_f32: &InlineArray,
    ) -> Option<(InlineArray, InlineArray)> {
        let state = self.state.as_ref()?;
        let key_core = match &state.keys {
            TensorRuntime::Uniform { core, .. } => core,
            _ => return None,
        };
        let key_rot = key_core.inverse_rotation_arr.as_ref()?;
        let key_proj = key_core.inverse_qjl_arr.as_ref()?;
        let batch = queries_f32.dim(0);
        let q_heads = queries_f32.dim(1);
        let key_dim = queries_f32.dim(3);
        let q_rows = batch * q_heads;
        let query_rows = queries_f32.reshape(&[q_rows, key_dim]);
        Some((query_rows.matmul(key_rot), query_rows.matmul(key_proj)))
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_query_transforms_wht(
        &self,
        queries_f32: &InlineArray,
    ) -> Option<(InlineArray, InlineArray)> {
        let state = self.state.as_ref()?;
        let key_core = match &state.keys {
            TensorRuntime::Uniform { core, .. } => core,
            _ => return None,
        };
        let batch = queries_f32.dim(0);
        let q_heads = queries_f32.dim(1);
        let key_dim = queries_f32.dim(3);
        if key_dim != 256 {
            return None;
        }
        let q_rows = batch * q_heads;
        let query_rows = queries_f32.reshape(&[q_rows, key_dim]);
        Some((
            key_core.rotate_rows_wht(&query_rows)?,
            key_core.project_rows_wht(&query_rows)?,
        ))
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_output_inverse_rotate_wht(
        &self,
        decoded_rot: &InlineArray,
    ) -> Option<InlineArray> {
        let state = self.state.as_ref()?;
        let value_core = match &state.values {
            TensorRuntime::Uniform { core, .. } => core,
            _ => return None,
        };
        let dim = decoded_rot.dim(1);
        if dim != 256 {
            return None;
        }
        value_core.inverse_rotate_rows_wht(decoded_rot)
    }
    fn try_gpu_uniform_attention_q8_d256_precomputed(
        &self,
        query_rot: &InlineArray,
        query_proj: Option<&InlineArray>,
        q_heads: i32,
        scale: f32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };

        let key_dim = layout.key_dim as i32;
        let value_dim = layout.value_dim as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if key_bits != 8
            || value_bits != 8
            || key_dim != 256
            || value_dim != 256
            || n_seq < 1024
            || q_rows <= 0
            || q_heads <= 0
            || (q_heads % layout.heads as i32) != 0
            || (q_heads / layout.heads as i32) > 8
            || cache_seq_capacity < n_seq
        {
            return None;
        }

        let kv_rows = (layout.batch * layout.heads) as i32;
        if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
            ks.q8_fullbyte_seq.as_ref(),
            ks.q8_slot_scales_seq.as_ref(),
            vs.d256_rot_values_seq.as_ref(),
        ) {
            InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
                query_rot,
                &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 4]),
                key_core.codebook_arr(key_bits)?,
                &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            )
        } else {
            let qjl_words = ks.qjl_words();
            if qjl_words != 8 {
                return None;
            }
            let query_proj = query_proj?;
            if let (Some(key_bytes), Some(slot_scales), Some(value_rot_dense)) = (
                ks.q8_keybytes_seq.as_ref(),
                ks.q8_slot_scales_seq.as_ref(),
                vs.d256_rot_values_seq.as_ref(),
            ) {
                InlineArray::turboquant_attention_q8_d256_packed_keys_dense_values_2pass(
                    query_rot,
                    query_proj,
                    &key_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                    &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 4]),
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            } else if let (Some(kv_bytes), Some(slot_scales)) =
                (ks.q8_kvbytes_seq.as_ref(), ks.q8_slot_scales_seq.as_ref())
            {
                if let Some(value_rot_dense) = vs.d256_rot_values_seq.as_ref() {
                    InlineArray::turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
                        query_rot,
                        query_proj,
                        &kv_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 4]),
                        key_core.codebook_arr(key_bits.saturating_sub(1))?,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                } else {
                    InlineArray::turboquant_attention_q8_d256_packed_kv_2pass(
                        query_rot,
                        query_proj,
                        &kv_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 4]),
                        key_core.codebook_arr(key_bits.saturating_sub(1))?,
                        value_core.codebook_arr(value_bits)?,
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                }
            } else if let (Some(key_bytes), Some(slot_scales)) =
                (ks.q8_keybytes_seq.as_ref(), ks.q8_slot_scales_seq.as_ref())
            {
                InlineArray::turboquant_attention_q8_d256_packed_keys_2pass(
                    query_rot,
                    query_proj,
                    &key_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                    &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 4]),
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &vs.indices
                        .as_ref()?
                        .reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            } else {
                InlineArray::turboquant_attention_q8_d256_2pass(
                    query_rot,
                    query_proj,
                    &ks.indices_t_array()
                        .reshape(&[kv_rows, key_dim, cache_seq_capacity]),
                    &ks.qjl_signs_t_array()?
                        .reshape(&[kv_rows, qjl_words, cache_seq_capacity]),
                    &ks.key_norms_array()?
                        .reshape(&[kv_rows, cache_seq_capacity]),
                    &ks.residual_norms_array()?
                        .reshape(&[kv_rows, cache_seq_capacity]),
                    &ks.key_slot_scale_array()?
                        .reshape(&[kv_rows, cache_seq_capacity]),
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &vs.indices_t_array()?
                        .reshape(&[kv_rows, value_dim, cache_seq_capacity]),
                    &vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]),
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            }
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_scores_precomputed(
        &self,
        query_rot: &InlineArray,
        query_proj: &InlineArray,
        q_heads: i32,
        scale: f32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let key_dim = layout.key_dim as i32;
        let kv_rows = (layout.batch * layout.heads) as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        let qjl_words = ks.qjl_words();
        let key_norms = ks
            .key_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_residual_norms = ks
            .residual_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_slot_scale = ks
            .key_slot_scale_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        if key_bits == 8
            && key_dim == 256
            && qjl_words == 8
            && q_heads > 0
            && (q_heads % layout.heads as i32) == 0
            && (q_heads / layout.heads as i32) <= 8
        {
            if let Some(scores) = InlineArray::turboquant_score_q8_d256(
                query_rot,
                query_proj,
                &ks.indices_t_array(),
                &ks.qjl_signs_t_array()?,
                &key_norms,
                &key_residual_norms,
                &key_slot_scale,
                key_core.codebook_arr(key_bits.saturating_sub(1))?,
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            ) {
                return Some(scores);
            }
        }
        InlineArray::turboquant_score(
            query_rot,
            query_proj,
            &ks.indices_t_array(),
            &ks.qjl_signs_t_array()?,
            &key_norms,
            &key_residual_norms,
            &key_slot_scale,
            key_core.codebook_arr(key_bits.saturating_sub(1))?,
            key_dim as u32,
            qjl_words as u32,
            key_core.codebook_arr(key_bits.saturating_sub(1))?.dim(0) as u32,
            q_rows as u32,
            n_seq as u32,
            cache_seq_capacity as u32,
            q_heads as u32,
            layout.heads as u32,
            scale,
        )
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_scores_precomputed_fullbyte(
        &self,
        query_rot: &InlineArray,
        q_heads: i32,
        scale: f32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let key_dim = layout.key_dim as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if key_bits != 8
            || key_dim != 256
            || q_heads <= 0
            || q_rows <= 0
            || n_seq <= 0
            || cache_seq_capacity < n_seq
        {
            return None;
        }
        let kv_rows = (layout.batch * layout.heads) as i32;
        if let (Some(key_indices), Some(slot_scales)) =
            (ks.q8_fullbyte_seq.as_ref(), ks.q8_slot_scales_seq.as_ref())
        {
            InlineArray::turboquant_score_q8_d256_fullbyte(
                query_rot,
                &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 4]),
                key_core.codebook_arr(key_bits)?,
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            )
        } else {
            None
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_attention_state_precomputed_fullbyte(
        &self,
        query_rot: &InlineArray,
        q_heads: i32,
        scale: f32,
    ) -> Option<(InlineArray, InlineArray, InlineArray)> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let key_dim = layout.key_dim as i32;
        let value_dim = layout.value_dim as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if key_bits != 8
            || key_dim != 256
            || value_dim != 256
            || q_heads <= 0
            || q_rows <= 0
            || n_seq < 1024
            || cache_seq_capacity < n_seq
        {
            return None;
        }
        let kv_rows = (layout.batch * layout.heads) as i32;
        if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
            ks.q8_fullbyte_seq.as_ref(),
            ks.q8_slot_scales_seq.as_ref(),
            vs.d256_rot_values_seq.as_ref(),
        ) {
            InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass_state(
                query_rot,
                &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 4]),
                key_core.codebook_arr(key_bits)?,
                &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            )
        } else {
            None
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_weighted_decode(
        &self,
        weights: &InlineArray,
        q_heads: i32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let value_dim = layout.value_dim as i32;
        let kv_rows = (layout.batch * layout.heads) as i32;
        let q_rows = weights.dim(0);
        let n_seq = self.cold_offset as i32;
        let indices_t = vs.indices_t_array()?;
        InlineArray::turboquant_weighted_decode(
            weights,
            &indices_t,
            &vs.norms_array()?.reshape(&[kv_rows, indices_t.dim(3)]),
            value_core.codebook_arr(value_bits)?,
            value_dim as u32,
            1u32 << value_bits,
            q_rows as u32,
            n_seq as u32,
            indices_t.dim(3) as u32,
            q_heads as u32,
            layout.heads as u32,
        )
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_weighted_sum_dense_values(
        &self,
        weights: &InlineArray,
        q_heads: i32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let value_dim = layout.value_dim as i32;
        let q_rows = weights.dim(0);
        let n_seq = self.cold_offset as i32;
        let value_rot_dense = vs.d256_rot_values_seq.as_ref()?;
        let cache_seq_capacity = value_rot_dense.dim(2);
        if value_dim != 256 || q_rows <= 0 || n_seq <= 0 || cache_seq_capacity < n_seq {
            return None;
        }
        let kv_rows = (layout.batch * layout.heads) as i32;
        InlineArray::turboquant_weighted_sum_d256_dense_values(
            weights,
            &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
            q_rows as u32,
            n_seq as u32,
            cache_seq_capacity as u32,
            q_heads as u32,
            layout.heads as u32,
        )
    }

    /// Variant F (NoQjl) fast path. Mirrors `try_gpu_uniform_attention` but
    /// dispatches to the `no_qjl` kernel families that don't read
    /// `qjl_signs` / `residual_norms`. Currently wired:
    ///   - d128/8b/8b uniform (`turboquant_attention_q8_d128_no_qjl_2pass`)
    ///   - d256/8b/8b uniform (`turboquant_attention_q8_d256_no_qjl_2pass`)
    ///
    /// Other configs (mixed, packed_keys variants) return None and the
    /// outer caller falls through to dequantize + SDPA. New no_qjl kernels
    /// can be wired here as they're added without touching the dispatch.
    fn try_gpu_uniform_attention_no_qjl(
        &self,
        queries_f32: &InlineArray,
        layout: CacheLayout,
        scale: f32,
        output_dtype: i32,
    ) -> Option<InlineArray> {
        // Phase E.3: outlier override is applied during gpu_dequantize_keys
        // but the fused score kernels read codebook + slot_scale only — they
        // would silently miss the override. Force the dequantize+SDPA path
        // when PerBlock outliers are enabled so attention sees the full
        // reconstruction (including outlier scatter). E.4 follow-up wires the
        // override into the score kernels themselves.
        if self.config.outliers.is_enabled() {
            return None;
        }
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;

        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };

        let batch = queries_f32.dim(0);
        let q_heads = queries_f32.dim(1);
        let key_dim = queries_f32.dim(3);
        let value_dim = layout.value_dim as i32;
        let q_rows = batch * q_heads;
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        // Currently d128/8b/8b and d256/8b/8b are supported. n_seq < 1024
        // falls back to dequantize+SDPA which is fine — the kernel is the
        // long-context win.
        let dim_supported = (key_dim == 128 && value_dim == 128) || (key_dim == 256 && value_dim == 256);
        if key_bits != 8
            || value_bits != 8
            || !dim_supported
            || n_seq < 1024
            || cache_seq_capacity < n_seq
            || q_heads <= 0
            || (q_heads % layout.heads as i32) != 0
        {
            return None;
        }

        let query_rows = queries_f32.reshape(&[q_rows, key_dim]);
        let query_rot = key_core.rotate_array(&query_rows)?;

        let kv_rows = (layout.batch * layout.heads) as i32;

        // Variant F uses the full key_bits codebook for keys; values use
        // their own codebook at value_bits resolution.
        let key_codebook = key_core.codebook_arr(key_bits)?;
        let value_codebook = value_core.codebook_arr(value_bits)?;

        // d256 fast path: when the cache populated the seq-major shadows
        // (`q8_slot_scales_seq` + `d256_rot_values_seq`), reuse the existing
        // `fullbyte_dense_values_2pass` kernel. It's structurally NoQjl-shaped
        // already (full 8-bit centroid index, no `query_proj`, no QJL term),
        // and NoQjl's `k_indices` were quantised from `k_rot / slot_scale` so
        // the kernel's `score_part *= key_slot_scale` recovers the original
        // magnitude exactly. Limited to `groups <= 8` (kernel's threadgroup
        // limit). Falls through to the base `no_qjl_2pass` kernel otherwise.
        if key_dim == 256 && (q_heads / layout.heads as i32) <= 8 {
            if let (Some(slot_scales), Some(value_rot_dense)) = (
                ks.q8_slot_scales_seq.as_ref(),
                vs.d256_rot_values_seq.as_ref(),
            ) {
                let key_indices_seq = ks
                    .indices
                    .reshape(&[kv_rows, cache_seq_capacity, key_dim]);
                if let Some(aggregated_rot) =
                    InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
                        &query_rot,
                        &key_indices_seq,
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 4]),
                        key_codebook,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                {
                    let output_rows =
                        value_core.inverse_rotate_output_array(&aggregated_rot, output_dtype)?;
                    return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
                }
            }
        }

        let key_norms = ks
            .key_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_slot_scale = ks
            .key_slot_scale_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        // Indices are scattered as [B, H, T, D]; the kernel reads them in the
        // [N, D, S] layout it expects.
        let key_indices_t = ks
            .indices_t_array()
            .reshape(&[kv_rows, key_dim, cache_seq_capacity]);
        let value_indices_t = vs
            .indices_t_array()?
            .reshape(&[kv_rows, value_dim, cache_seq_capacity]);
        let value_norms = vs
            .norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);

        let aggregated_rot = if key_dim == 128 {
            InlineArray::turboquant_attention_q8_d128_no_qjl_2pass(
                &query_rot,
                &key_indices_t,
                &key_norms,
                &key_slot_scale,
                key_codebook,
                &value_indices_t,
                &value_norms,
                value_codebook,
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            )?
        } else {
            InlineArray::turboquant_attention_q8_d256_no_qjl_2pass(
                &query_rot,
                &key_indices_t,
                &key_norms,
                &key_slot_scale,
                key_codebook,
                &value_indices_t,
                &value_norms,
                value_codebook,
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            )?
        };

        // Output is in the value's rotated frame; inverse-rotate via value_core.
        let output_rows = value_core.inverse_rotate_output_array(&aggregated_rot, output_dtype)?;
        Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]))
    }

    /// Phase F: Hamming skip-list pre-filter dispatch.
    ///
    /// Computes a per-query 1-bit sign hash, runs the XOR/popcount Hamming
    /// distance kernel against the cached sign hashes, picks the M slots with
    /// the smallest distances, gathers those slots' packed key/value cache
    /// rows into a contiguous M-length scratch view, and runs the existing
    /// Variant F (NoQjl) score kernel over that subset. The kernel's
    /// `cache_seq_capacity` is set to M so its strided indexing addresses the
    /// gathered scratch buffers correctly.
    ///
    /// V1 gates: skiplist_threshold set + cold_offset > threshold + sign_hash
    /// populated + Variant F (NoQjl) Uniform key/value config + d128 or d256 +
    /// non-GQA (q_heads == kv_heads). GQA + Standard QJL + Mixed all fall
    /// through to the dense kernel path.
    fn try_hamming_skiplist_attention(
        &self,
        queries_f32: &InlineArray,
        layout: CacheLayout,
        scale: f32,
        output_dtype: i32,
    ) -> Option<InlineArray> {
        let threshold = self.config.skiplist_threshold?;
        if self.cold_offset <= threshold {
            return None;
        }
        // Phase E.3: see try_gpu_uniform_attention_no_qjl — score kernels
        // would miss the outlier override. Fall through to dequantize+SDPA.
        if self.config.outliers.is_enabled() {
            return None;
        }
        let state = self.state.as_ref()?;
        if !matches!(state.qjl, super::TurboQuantQjlMode::NoQjl) {
            return None;
        }
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let sign_hash = ks.sign_hash.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        if key_bits != 8 || value_bits != 8 {
            return None;
        }

        let batch = queries_f32.dim(0);
        let q_heads = queries_f32.dim(1);
        let kv_heads = layout.heads as i32;
        let key_dim = queries_f32.dim(3);
        let value_dim = layout.value_dim as i32;
        // GQA: q_heads must be a positive multiple of kv_heads. Hamming
        // pre-filter runs per-q-row via row → kv_row = row / groups inside
        // the kernel; gather uses flat-index take_axis to pick per-q-row
        // slot subsets from kv-row-indexed cache buffers without expanding.
        if kv_heads <= 0 || q_heads <= 0 || (q_heads % kv_heads) != 0 {
            return None;
        }
        let groups = q_heads / kv_heads;
        if !((key_dim == 128 && value_dim == 128) || (key_dim == 256 && value_dim == 256)) {
            return None;
        }

        let q_rows = batch * q_heads;
        let kv_rows = (layout.batch * layout.heads) as i32;
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if cache_seq_capacity < n_seq {
            return None;
        }
        let packed_dim = (key_dim + 31) / 32;
        // Top-M defaults to min(cold_offset, 2048). top_m ≤ n_seq by
        // construction; the equal case still exercises the gather path
        // (no-op selection) and is allowed so smaller-than-2048 caches
        // near the threshold still go through the dispatch.
        let top_m = (self.cold_offset.min(2048)) as i32;
        if top_m <= 0 {
            return None;
        }

        // 1. Rotate query into the same frame as the encoded keys.
        let query_rows = queries_f32.reshape(&[q_rows, key_dim]);
        let query_rot = key_core.rotate_array(&query_rows)?;

        // 2. Pack query sign bits → [q_rows, packed_dim] u32.
        let query_signs = InlineArray::turboquant_pack_sign_bits(
            &query_rot,
            key_dim as u32,
            packed_dim as u32,
            q_rows as u32,
        )?;

        // 3. Slice cold sign-hash. Storage is [B, kv_heads, S_cap, packed_dim].
        // Reshape to [kv_rows, n_seq, packed_dim] — kernel maps q_row →
        // kv_row = q_row / groups internally.
        let sign_hash_active = sign_hash
            .slice(
                &[0, 0, 0, 0],
                &[layout.batch as i32, kv_heads, n_seq, packed_dim],
            )
            .reshape(&[kv_rows, n_seq, packed_dim]);

        // 4. Hamming distances [q_rows, n_seq] u32.
        let distances = InlineArray::turboquant_hamming_distances(
            &query_signs,
            &sign_hash_active,
            packed_dim as u32,
            q_rows as u32,
            n_seq as u32,
            groups as u32,
        )?;

        // 5. argpartition for the M smallest distances. argpartition pivots
        // such that result[..kth] are <= result[kth]; pivot at top_m - 1 to
        // place the M smallest in [0..top_m].
        let part = distances.argpartition(top_m - 1, -1);
        let top_indices = part.slice(&[0, 0], &[q_rows, top_m]);

        // 6. Per-q-row gather from kv-row-indexed cache buffers using flat
        // take_axis: flat_idx[q, m] = (q_row / groups) * S_cap + top[q, m].
        // For non-GQA (groups=1, q_rows=kv_rows) this is equivalent to the
        // V1 broadcast take_along_axis. For GQA (groups>1), it correctly
        // routes each q_row to its kv_row's cache without expanding the
        // source (which would be O(groups · kv_rows · S_cap · D)).
        let kv_row_for_q: Vec<i32> = (0..q_rows)
            .map(|q| (q / groups) * cache_seq_capacity)
            .collect();
        let kv_offset = InlineArray::from_i32_slice_shaped(&kv_row_for_q, &[q_rows, 1]);
        let top_indices_i32 = top_indices.as_dtype(crate::compat::Dtype::Int32.as_i32());
        let flat_idx_2d = top_indices_i32.add(&kv_offset); // [q_rows, top_m] i32

        // For [kv_rows, key_dim, S_cap]: transpose to [kv_rows, S_cap, key_dim],
        // flatten to [kv_rows*S_cap, key_dim], take_axis with flat_idx, then
        // transpose to [q_rows, key_dim, top_m].
        let key_indices_t = ks.indices_t_array();
        let key_indices_kvs_d = key_indices_t
            .reshape(&[kv_rows, key_dim, cache_seq_capacity])
            .transpose_axes(&[0, 2, 1])
            .reshape(&[kv_rows * cache_seq_capacity, key_dim]);
        let gathered_key_indices_t = key_indices_kvs_d
            .take_axis(&flat_idx_2d, 0) // [q_rows, top_m, key_dim]
            .transpose_axes(&[0, 2, 1]); // [q_rows, key_dim, top_m]

        let value_indices_kvs_d = vs
            .indices_t_array()?
            .reshape(&[kv_rows, value_dim, cache_seq_capacity])
            .transpose_axes(&[0, 2, 1])
            .reshape(&[kv_rows * cache_seq_capacity, value_dim]);
        let gathered_value_indices_t = value_indices_kvs_d
            .take_axis(&flat_idx_2d, 0)
            .transpose_axes(&[0, 2, 1]);

        // For [kv_rows, S_cap]: flatten to [kv_rows*S_cap], take_axis with
        // flat_idx → [q_rows, top_m].
        let key_norms_flat = ks
            .key_norms_array()?
            .reshape(&[kv_rows * cache_seq_capacity]);
        let gathered_key_norms = key_norms_flat.take_axis(&flat_idx_2d, 0);

        let key_slot_scale_flat = ks
            .key_slot_scale_array()?
            .reshape(&[kv_rows * cache_seq_capacity]);
        let gathered_key_slot_scale = key_slot_scale_flat.take_axis(&flat_idx_2d, 0);

        let value_norms_flat = vs
            .norms_array()?
            .reshape(&[kv_rows * cache_seq_capacity]);
        let gathered_value_norms = value_norms_flat.take_axis(&flat_idx_2d, 0);

        let key_codebook = key_core.codebook_arr(key_bits)?;
        let value_codebook = value_core.codebook_arr(value_bits)?;

        // 7. Run the existing Variant F kernel on the gathered subset.
        // Each q_row now has its own slice; pass q_heads = kv_heads = q_heads
        // so the kernel treats the gathered cache as 1:1 (groups=1 inside
        // the score kernel — GQA was already resolved during gather).
        let aggregated_rot = if key_dim == 128 {
            InlineArray::turboquant_attention_q8_d128_no_qjl_2pass(
                &query_rot,
                &gathered_key_indices_t,
                &gathered_key_norms,
                &gathered_key_slot_scale,
                key_codebook,
                &gathered_value_indices_t,
                &gathered_value_norms,
                value_codebook,
                q_rows as u32,
                top_m as u32,
                top_m as u32,
                q_heads as u32,
                q_heads as u32,
                scale,
            )?
        } else {
            InlineArray::turboquant_attention_q8_d256_no_qjl_2pass(
                &query_rot,
                &gathered_key_indices_t,
                &gathered_key_norms,
                &gathered_key_slot_scale,
                key_codebook,
                &gathered_value_indices_t,
                &gathered_value_norms,
                value_codebook,
                q_rows as u32,
                top_m as u32,
                top_m as u32,
                q_heads as u32,
                q_heads as u32,
                scale,
            )?
        };

        let output_rows = value_core.inverse_rotate_output_array(&aggregated_rot, output_dtype)?;
        Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]))
    }

    fn try_gpu_uniform_attention(
        &self,
        queries_f32: &InlineArray,
        layout: CacheLayout,
        scale: f32,
        output_dtype: i32,
    ) -> Option<InlineArray> {
        // Phase E.3: outlier override lives in gpu_dequantize_keys; the fused
        // score kernels read codebook + slot_scale only and would miss it.
        // Force dequantize+SDPA so attention sees the full reconstruction.
        if self.config.outliers.is_enabled() {
            return None;
        }
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        // Variant F (NoQjl): dispatch to the no_qjl fast path. Currently
        // wired for d128/8b/8b uniform; other configs fall back to
        // dequantize+SDPA inside the no_qjl helper.
        if matches!(state.qjl, super::TurboQuantQjlMode::NoQjl) {
            return self.try_gpu_uniform_attention_no_qjl(
                queries_f32,
                layout,
                scale,
                output_dtype,
            );
        }

        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            TensorRuntime::Uniform { .. } | TensorRuntime::Mixed { .. } => return None,
        };
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            TensorRuntime::Uniform { .. } | TensorRuntime::Mixed { .. } => return None,
        };

        let batch = queries_f32.dim(0);
        let q_heads = queries_f32.dim(1);
        let key_dim = queries_f32.dim(3);
        let value_dim = layout.value_dim as i32;
        let q_rows = batch * q_heads;
        let n_seq = self.cold_offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if q_rows <= 0 || n_seq <= 0 || cache_seq_capacity < n_seq {
            return None;
        }

        let trace_timing = turboquant_trace_enabled();
        let query_ready_us = if trace_timing {
            eval_stage_micros(queries_f32)
        } else {
            0
        };
        let query_rows = queries_f32.reshape(&[q_rows, key_dim]);
        // Phase D.2: pack_mode = Fullbyte enables the long-context fullbyte
        // path the same way the env-var debug override does — both are valid
        // ways to opt in.
        let pack_mode_fullbyte =
            matches!(self.config.pack_mode, super::TurboQuantPackMode::Fullbyte);
        let can_try_q8_fullbyte = (turboquant_q8_fullbyte_enabled() || pack_mode_fullbyte)
            && key_bits == 8
            && value_bits == 8
            && key_dim == 256
            && value_dim == 256
            && n_seq >= 1024
            && ks.q8_fullbyte_seq.is_some()
            && ks.q8_slot_scales_seq.is_some()
            && vs.d256_rot_values_seq.is_some();
        let mut project_us = 0;
        // Fused rotate+project: saves one dispatch per layer by doing
        // `input @ [inv_rot | inv_qjl]` as a single [N, 2*dim] matmul
        // instead of two separate [N, dim] matmuls. Only applied when
        // both outputs are needed (i.e., the q8 fullbyte fast path is
        // not taken). Falls back to sequential calls if the stacked
        // matrix wasn't built.
        let (query_rot, mut query_proj) = if !can_try_q8_fullbyte {
            if let Some((rot, proj)) = key_core.rotate_and_project_array(&query_rows) {
                (rot, Some(proj))
            } else {
                let rot = key_core.rotate_array(&query_rows)?;
                let proj = key_core.project_array(&query_rows)?;
                (rot, Some(proj))
            }
        } else {
            (key_core.rotate_array(&query_rows)?, None)
        };
        let rotate_us = if trace_timing {
            eval_stage_micros(&query_rot)
        } else {
            0
        };
        if let Some(proj) = query_proj.as_ref() {
            if trace_timing {
                project_us = eval_stage_micros(proj);
            }
        }

        let kv_rows = (layout.batch * layout.heads) as i32;
        let key_norms = ks
            .key_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_residual_norms = ks
            .residual_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_slot_scale = ks
            .key_slot_scale_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let qjl_words = ks.qjl_words();
        if can_try_q8_fullbyte {
            if let Some(decoded_rot) =
                self.try_gpu_uniform_attention_q8_d256_precomputed(&query_rot, None, q_heads, scale)
            {
                let decode_us = if trace_timing {
                    eval_stage_micros(&decoded_rot)
                } else {
                    0
                };
                let output_rows =
                    value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
                let inverse_rotate_us = if trace_timing {
                    eval_stage_micros(&output_rows)
                } else {
                    0
                };
                if trace_timing {
                    trace_turboquant_bridge(&format!(
                        "gpu_uniform_q8_d256_fullbyte_densev_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                        n_seq,
                        q_rows,
                        query_ready_us,
                        rotate_us,
                        project_us,
                        decode_us,
                        inverse_rotate_us,
                    ));
                }
                return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
            }
        }
        if query_proj.is_none() {
            let projected = key_core.project_array(&query_rows)?;
            if trace_timing {
                project_us = eval_stage_micros(&projected);
            }
            query_proj = Some(projected);
        }
        let query_proj = query_proj.as_ref()?;
        let key_codebook = key_core.codebook_arr(key_bits.saturating_sub(1))?;
        if let Some(decoded_rot) = self.try_gpu_uniform_attention_q8_d256_precomputed(
            &query_rot,
            Some(query_proj),
            q_heads,
            scale,
        ) {
            let decode_us = if trace_timing {
                eval_stage_micros(&decoded_rot)
            } else {
                0
            };
            let output_rows = value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
            let inverse_rotate_us = if trace_timing {
                eval_stage_micros(&output_rows)
            } else {
                0
            };
            if trace_timing {
                trace_turboquant_bridge(&format!(
                    "gpu_uniform_q8_d256_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                    n_seq,
                    q_rows,
                    query_ready_us,
                    rotate_us,
                    project_us,
                    decode_us,
                    inverse_rotate_us,
                ));
            }
            return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
        }
        if key_bits == 8 && value_bits == 8 && key_dim == 128 && value_dim == 128 && n_seq >= 1024 {
            let key_indices = ks
                .indices_t_array()
                .reshape(&[kv_rows, key_dim, cache_seq_capacity]);
            let value_indices =
                vs.indices_t_array()?
                    .reshape(&[kv_rows, value_dim, cache_seq_capacity]);
            let value_norms = vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]);

            if q_heads > 8 {
                if let Some(key_bytes) = ks.q8_keybytes_t.as_ref() {
                    let key_bytes = key_bytes.reshape(&[kv_rows, key_dim, cache_seq_capacity]);
                    if let Some(decoded_rot) =
                        InlineArray::turboquant_attention_q8_d128_packed_keys_2pass(
                            &query_rot,
                            query_proj,
                            &key_bytes,
                            &key_norms,
                            &key_residual_norms,
                            &key_slot_scale,
                            key_codebook,
                            &value_indices,
                            &value_norms,
                            value_core.codebook_arr(value_bits)?,
                            q_rows as u32,
                            n_seq as u32,
                            cache_seq_capacity as u32,
                            q_heads as u32,
                            layout.heads as u32,
                            scale,
                        )
                    {
                        let decode_us = if trace_timing {
                            eval_stage_micros(&decoded_rot)
                        } else {
                            0
                        };
                        let output_rows =
                            value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
                        let inverse_rotate_us = if trace_timing {
                            eval_stage_micros(&output_rows)
                        } else {
                            0
                        };
                        if trace_timing {
                            trace_turboquant_bridge(&format!(
                                "gpu_uniform_q8_d128_packed_keys_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                                n_seq,
                                q_rows,
                                query_ready_us,
                                rotate_us,
                                project_us,
                                decode_us,
                                inverse_rotate_us,
                            ));
                        }
                        return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
                    }
                }
            } else if qjl_words == 4 {
                let key_qjl_signs = ks
                    .qjl_signs_t_array()?
                    .reshape(&[kv_rows, qjl_words, cache_seq_capacity]);
                if let Some(decoded_rot) = InlineArray::turboquant_attention_q8_d128_2pass(
                    &query_rot,
                    query_proj,
                    &key_indices,
                    &key_qjl_signs,
                    &key_norms,
                    &key_residual_norms,
                    &key_slot_scale,
                    key_codebook,
                    &value_indices,
                    &value_norms,
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                ) {
                    let decode_us = if trace_timing {
                        eval_stage_micros(&decoded_rot)
                    } else {
                        0
                    };
                    let output_rows =
                        value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
                    let inverse_rotate_us = if trace_timing {
                        eval_stage_micros(&output_rows)
                    } else {
                        0
                    };
                    if trace_timing {
                        trace_turboquant_bridge(&format!(
                            "gpu_uniform_q8_d128_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                            n_seq,
                            q_rows,
                            query_ready_us,
                            rotate_us,
                            project_us,
                            decode_us,
                            inverse_rotate_us,
                        ));
                    }
                    return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
                }
            }
        }

        let scores =
            self.bench_gpu_uniform_scores_precomputed(&query_rot, query_proj, q_heads, scale)?;
        let score_us = if trace_timing {
            eval_stage_micros(&scores)
        } else {
            0
        };
        let weights = scores.softmax(-1);
        let softmax_us = if trace_timing {
            eval_stage_micros(&weights)
        } else {
            0
        };
        let value_norms = vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]);
        let decoded_rot = InlineArray::turboquant_weighted_decode(
            &weights,
            &vs.indices_t_array()?,
            &value_norms,
            value_core.codebook_arr(value_bits)?,
            value_dim as u32,
            1u32 << value_bits,
            q_rows as u32,
            n_seq as u32,
            cache_seq_capacity as u32,
            q_heads as u32,
            layout.heads as u32,
        )?;
        let decode_us = if trace_timing {
            eval_stage_micros(&decoded_rot)
        } else {
            0
        };
        let output_rows = value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
        let inverse_rotate_us = if trace_timing {
            eval_stage_micros(&output_rows)
        } else {
            0
        };
        if trace_timing {
            trace_turboquant_bridge(&format!(
                "gpu_uniform_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score={} softmax={} decode={} inverse_rotate={}",
                n_seq,
                q_rows,
                query_ready_us,
                rotate_us,
                project_us,
                score_us,
                softmax_us,
                decode_us,
                inverse_rotate_us
            ));
        }
        Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]))
    }

    fn ensure_layout(
        &mut self,
        keys: &InlineArray,
        values: &InlineArray,
    ) -> Result<CacheLayout, String> {
        // Validate shape: [B, H, S, D]
        if keys.ndim() != 4 || values.ndim() != 4 {
            return Err(format!(
                "TurboQuant: expected 4-D keys/values, got ndim {} / {}",
                keys.ndim(),
                values.ndim()
            ));
        }

        let b = keys.dim(0) as usize;
        let h = keys.dim(1) as usize;
        let kd = keys.dim(3) as usize;
        let vd = values.dim(3) as usize;

        if let Some(existing) = self.layout {
            if existing.batch != b
                || existing.heads != h
                || existing.key_dim != kd
                || existing.value_dim != vd
            {
                return Err(format!(
                    "TurboQuant: layout mismatch — expected [{b},{h},*,{kd}] / [{b},{h},*,{vd}]"
                ));
            }
            return Ok(existing);
        }

        let layout = CacheLayout {
            batch: b,
            heads: h,
            key_dim: kd,
            value_dim: vd,
        };
        self.layout = Some(layout);
        Ok(layout)
    }
}

