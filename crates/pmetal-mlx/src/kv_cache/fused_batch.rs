//! Fused batched KV cache for continuous-batching decode.
//!
//! # Why this exists
//!
//! The per-slot `BatchKVCache` (`Vec<KVCache>`) runs one forward per slot
//! per decode step. On Apple Silicon, decode matmuls are small
//! (`[1, hidden]`) and kernel-dispatch overhead dominates wall time.
//! `FusedBatchKVCache` owns a single `[max_slots, heads, max_seq, D]` K/V
//! tensor per layer so the model can run one fused `[N, 1]` forward per
//! tick, folding N kernel launches into one per op.
//!
//! # Layout
//!
//! Per layer:
//! - `layer_keys[i]: [max_slots, num_kv_heads, max_seq_len, head_dim]`
//! - `layer_values[i]: [max_slots, num_kv_heads, max_seq_len, value_head_dim]`
//!
//! Per slot:
//! - `offsets[slot]: usize` — number of valid tokens written so far.
//!
//! All rows are eager-allocated at construction. Total memory matches the
//! existing per-slot `Vec<KVCache>` sized with `max_seq_len`, so this is a
//! compute-side optimization only, not a memory regression.
//!
//! # Update protocol
//!
//! Each decode step the caller passes:
//! - `active_indices: &[usize]` — batch rows currently being decoded.
//! - `new_k, new_v: [N_active, heads, 1, D]` — freshly projected K/V.
//!
//! `update_and_fetch_batched` returns:
//! - `K: [N_active, heads, T_max, D]`
//! - `V: [N_active, heads, T_max, D_v]`
//! - `mask: [N_active, 1, 1, T_max]` — additive mask, `-inf` at positions
//!   `t >= new_offset[slot]` (per-slot left-padding against the shared
//!   `T_max`).
//!
//! Internally:
//! 1. `K_active = K_full.take(active_indices, axis=0)` — gather rows.
//! 2. `K_active = put_along_axis(K_active, write_idx, new_k, axis=2)` —
//!    write at each slot's current offset.
//! 3. `K_full = put_along_axis(K_full, scatter_idx, K_active, axis=0)` —
//!    scatter rows back into the persistent buffer.
//! 4. Advance `offsets[slot] += 1` for each active slot.
//!
//! # Scope
//!
//! Standard (non-quantized, non-sliding) caches only, which covers every
//! arch slated for the fused-batch rollout. TurboQuant/Quantized/Sliding
//! variants stay on the per-slot fallback.

use pmetal_bridge::compat::{Array, Dtype, Exception, ops};

use super::{CacheMode, KVCacheConfig};

/// Fused per-layer batched KV cache.
///
/// Offsets are `[num_layers][max_slots]` — each layer tracks its own
/// per-slot token count. This matches the serial `KVCache` semantics
/// (every layer advances its own offset independently during a forward)
/// so fused RoPE / update-and-fetch behave identically to the per-slot
/// path.
#[derive(Debug)]
pub struct FusedBatchKVCache {
    config: KVCacheConfig,
    max_slots: usize,
    /// Per-layer K buffer: `[max_slots, num_kv_heads, max_seq_len, head_dim]`.
    layer_keys: Vec<Array>,
    /// Per-layer V buffer: `[max_slots, num_kv_heads, max_seq_len, value_head_dim]`.
    layer_values: Vec<Array>,
    /// Per-layer per-slot offsets: `offsets[layer_idx][batch_idx]`.
    offsets: Vec<Vec<usize>>,
}

impl FusedBatchKVCache {
    /// Create a new fused batched cache with eager per-layer allocations.
    ///
    /// Only `CacheMode::Standard` is supported; other modes return an
    /// `Exception`. Callers that need quantized or sliding caches must
    /// fall back to the per-slot path.
    pub fn new(config: KVCacheConfig, max_slots: usize) -> Result<Self, Exception> {
        if !matches!(config.mode, CacheMode::Standard) {
            return Err(Exception::custom(format!(
                "FusedBatchKVCache only supports CacheMode::Standard, got {:?}",
                config.mode,
            )));
        }
        if max_slots == 0 {
            return Err(Exception::custom(
                "FusedBatchKVCache::new: max_slots must be >= 1",
            ));
        }

        let k_shape = [
            max_slots as i32,
            config.num_kv_heads as i32,
            config.max_seq_len as i32,
            config.head_dim as i32,
        ];
        let v_shape = [
            max_slots as i32,
            config.num_kv_heads as i32,
            config.max_seq_len as i32,
            config.value_head_dim as i32,
        ];
        let layer_keys = (0..config.num_layers)
            .map(|_| ops::zeros(&k_shape, config.dtype))
            .collect();
        let layer_values = (0..config.num_layers)
            .map(|_| ops::zeros(&v_shape, config.dtype))
            .collect();

        let offsets = (0..config.num_layers)
            .map(|_| vec![0; max_slots])
            .collect();
        Ok(Self {
            config,
            max_slots,
            layer_keys,
            layer_values,
            offsets,
        })
    }

    /// Configuration used to construct this cache.
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// Number of batch rows pre-allocated.
    pub fn max_slots(&self) -> usize {
        self.max_slots
    }

    /// Layer-0 token offset for `batch_idx`. This is the "canonical"
    /// per-slot position used for RoPE position ids — every layer's
    /// offset advances in lockstep during a single decode forward, so
    /// layer 0's offset at the *start* of the forward is the authoritative
    /// position of the token being written.
    pub fn offset(&self, batch_idx: usize) -> usize {
        self.offset_for(0, batch_idx)
    }

    /// Per-layer per-slot offset. Returns 0 for out-of-range indices.
    pub fn offset_for(&self, layer_idx: usize, batch_idx: usize) -> usize {
        self.offsets
            .get(layer_idx)
            .and_then(|layer| layer.get(batch_idx).copied())
            .unwrap_or(0)
    }

    /// Layer-0 offsets for all slots, indexed by batch row.
    pub fn offsets(&self) -> &[usize] {
        self.offsets.first().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Largest layer-0 offset among `active_indices`. Returns 0 on empty input.
    pub fn max_offset_for(&self, active_indices: &[usize]) -> usize {
        active_indices
            .iter()
            .map(|&i| self.offset_for(0, i))
            .max()
            .unwrap_or(0)
    }

    /// Reset a slot's offset back to 0 across all layers. No buffer
    /// zeroing — the next `update_and_fetch_batched` overwrites
    /// position 0 directly.
    pub fn admit(&mut self, batch_idx: usize) -> Result<(), Exception> {
        if batch_idx >= self.max_slots {
            return Err(Exception::custom(format!(
                "FusedBatchKVCache::admit: batch_idx {} out of range (max_slots {})",
                batch_idx, self.max_slots,
            )));
        }
        for layer in self.offsets.iter_mut() {
            layer[batch_idx] = 0;
        }
        Ok(())
    }

    /// Release a slot; offset reset to 0 across all layers so the next admit starts clean.
    pub fn release(&mut self, batch_idx: usize) {
        for layer in self.offsets.iter_mut() {
            if let Some(off) = layer.get_mut(batch_idx) {
                *off = 0;
            }
        }
    }

    /// Write `new_k`/`new_v` at each active slot's current offset,
    /// advance offsets, and return compact K/V + padding mask.
    ///
    /// # Shapes
    /// - `new_k: [N_active, num_kv_heads, 1, head_dim]`
    /// - `new_v: [N_active, num_kv_heads, 1, value_head_dim]`
    /// - `active_indices.len() == N_active`, each in `0..max_slots`.
    ///
    /// Returns `(K, V, mask)` where
    /// - `K: [N_active, num_kv_heads, T_max, head_dim]`
    /// - `V: [N_active, num_kv_heads, T_max, value_head_dim]`
    /// - `mask: [N_active, 1, 1, T_max]` additive `-inf`/`0.0`.
    pub fn update_and_fetch_batched(
        &mut self,
        layer_idx: usize,
        active_indices: &[usize],
        new_k: &Array,
        new_v: &Array,
    ) -> Result<(Array, Array, Array), Exception> {
        if layer_idx >= self.config.num_layers {
            return Err(Exception::custom(format!(
                "FusedBatchKVCache: layer_idx {} out of range ({})",
                layer_idx, self.config.num_layers,
            )));
        }
        let n_active = active_indices.len();
        if n_active == 0 {
            return Err(Exception::custom(
                "FusedBatchKVCache::update_and_fetch_batched: empty active_indices",
            ));
        }
        let heads = self.config.num_kv_heads as i32;
        let d_k = self.config.head_dim as i32;
        let d_v = self.config.value_head_dim as i32;

        if new_k.shape() != [n_active as i32, heads, 1, d_k]
            || new_v.shape() != [n_active as i32, heads, 1, d_v]
        {
            return Err(Exception::custom(format!(
                "FusedBatchKVCache: new_k/new_v shape mismatch (expected [{}, {}, 1, {}]/[.., {}]; got {:?}/{:?})",
                n_active,
                heads,
                d_k,
                d_v,
                new_k.shape(),
                new_v.shape(),
            )));
        }

        // Validate and collect current offsets; bounds-check against max_seq_len.
        let mut pre_offsets = Vec::with_capacity(n_active);
        for &idx in active_indices {
            if idx >= self.max_slots {
                return Err(Exception::custom(format!(
                    "FusedBatchKVCache: active batch_idx {} out of range (max_slots {})",
                    idx, self.max_slots,
                )));
            }
            let off = self.offsets[layer_idx][idx];
            if off + 1 > self.config.max_seq_len {
                return Err(Exception::custom(format!(
                    "FusedBatchKVCache: slot {} exceeds max_seq_len {}",
                    idx, self.config.max_seq_len,
                )));
            }
            pre_offsets.push(off);
        }

        // Gather active rows: [N_active, heads, max_seq, D]
        let active_i32: Vec<i32> = active_indices.iter().map(|&i| i as i32).collect();
        let gather_idx = Array::from_i32_slice_shaped(&active_i32, &[n_active as i32]);
        let k_full = &self.layer_keys[layer_idx];
        let v_full = &self.layer_values[layer_idx];
        let k_active = k_full.take_axis(&gather_idx, 0);
        let v_active = v_full.take_axis(&gather_idx, 0);

        // Write indices along axis 2: one position per slot, broadcast
        // across heads / head_dim.
        let pre_off_i32: Vec<i32> = pre_offsets.iter().map(|&o| o as i32).collect();
        let write_pos = Array::from_i32_slice_shaped(&pre_off_i32, &[n_active as i32, 1, 1, 1]);
        let write_idx_k = write_pos.broadcast_to(&[n_active as i32, heads, 1, d_k]);
        let write_idx_v = write_pos.broadcast_to(&[n_active as i32, heads, 1, d_v]);

        let k_written = ops::put_along_axis(&k_active, &write_idx_k, new_k, 2);
        let v_written = ops::put_along_axis(&v_active, &write_idx_v, new_v, 2);

        // Scatter back into persistent buffers along axis 0.
        let scatter_pos = Array::from_i32_slice_shaped(&active_i32, &[n_active as i32, 1, 1, 1]);
        let max_seq = self.config.max_seq_len as i32;
        let scatter_idx_k = scatter_pos.broadcast_to(&[n_active as i32, heads, max_seq, d_k]);
        let scatter_idx_v = scatter_pos.broadcast_to(&[n_active as i32, heads, max_seq, d_v]);

        // k_written has axis-2 dim = max_seq (we gathered full rows and
        // wrote one position), so the shapes align with scatter_idx.
        self.layer_keys[layer_idx] = ops::put_along_axis(k_full, &scatter_idx_k, &k_written, 0);
        self.layer_values[layer_idx] = ops::put_along_axis(v_full, &scatter_idx_v, &v_written, 0);

        // Advance offsets for this layer only.
        for &idx in active_indices {
            self.offsets[layer_idx][idx] += 1;
        }

        // Slice K/V to T_max, where T_max = max(new_offset) among active slots.
        let new_offsets: Vec<usize> = active_indices
            .iter()
            .map(|&i| self.offsets[layer_idx][i])
            .collect();
        let t_max = new_offsets.iter().copied().max().unwrap_or(0);
        let t_max_i32 = t_max as i32;
        let k_out = k_written.slice(
            &[0, 0, 0, 0],
            &[n_active as i32, heads, t_max_i32, d_k],
        );
        let v_out = v_written.slice(
            &[0, 0, 0, 0],
            &[n_active as i32, heads, t_max_i32, d_v],
        );

        // Build additive padding mask: mask[n, 0, 0, t] = 0 if t < new_offset[n] else -inf.
        let new_off_i32: Vec<i32> = new_offsets.iter().map(|&o| o as i32).collect();
        let offs_arr = Array::from_i32_slice_shaped(&new_off_i32, &[n_active as i32, 1, 1, 1]);
        let t_range = ops::arange_range(0, t_max_i32).reshape(&[1, 1, 1, t_max_i32]);
        let offs_f = offs_arr.as_dtype(Dtype::Float32.as_i32());
        let valid = ops::less(&t_range, &offs_f);
        let zero = Array::from_f32(0.0);
        let neg_inf = Array::from_f32(f32::NEG_INFINITY);
        let mask = ops::r#where(&valid, &zero, &neg_inf);

        Ok((k_out, v_out, mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(num_layers: usize, max_seq: usize, heads: usize, d: usize) -> KVCacheConfig {
        KVCacheConfig::new(num_layers, max_seq, heads, d)
    }

    #[test]
    fn fused_cache_admit_release_resets_offset() {
        let mut c = FusedBatchKVCache::new(cfg(2, 8, 2, 4), 3).unwrap();
        // Simulate two writes for slot 0 on layer 0, then release.
        c.offsets[0][0] = 2;
        c.release(0);
        assert_eq!(c.offset(0), 0);
        c.admit(0).unwrap();
        assert_eq!(c.offset(0), 0);
    }

    #[test]
    fn fused_cache_rejects_non_standard_mode() {
        let bad = cfg(1, 8, 1, 4).with_quantized(8, 64);
        assert!(FusedBatchKVCache::new(bad, 2).is_err());
    }

    #[test]
    fn fused_cache_update_advances_offset_and_shape() {
        let mut c = FusedBatchKVCache::new(cfg(1, 4, 2, 3), 4).unwrap();
        c.admit(0).unwrap();
        c.admit(2).unwrap();
        let new_k = Array::zeros_f32(&[2, 2, 1, 3]);
        let new_v = Array::zeros_f32(&[2, 2, 1, 3]);
        let (k, v, mask) = c
            .update_and_fetch_batched(0, &[0, 2], &new_k, &new_v)
            .unwrap();
        assert_eq!(k.shape(), &[2, 2, 1, 3]);
        assert_eq!(v.shape(), &[2, 2, 1, 3]);
        assert_eq!(mask.shape(), &[2, 1, 1, 1]);
        assert_eq!(c.offset(0), 1);
        assert_eq!(c.offset(2), 1);
        // Slot 1 stays untouched.
        assert_eq!(c.offset(1), 0);
    }

    #[test]
    fn fused_cache_update_grows_t_max_across_steps() {
        let mut c = FusedBatchKVCache::new(cfg(1, 4, 1, 2), 2).unwrap();
        c.admit(0).unwrap();
        c.admit(1).unwrap();

        // Step 1: both active, one token each → T_max=1 on layer 0.
        let nk1 = Array::zeros_f32(&[2, 1, 1, 2]);
        let nv1 = Array::zeros_f32(&[2, 1, 1, 2]);
        let (k1, _, m1) = c.update_and_fetch_batched(0, &[0, 1], &nk1, &nv1).unwrap();
        assert_eq!(k1.shape(), &[2, 1, 1, 2]);
        assert_eq!(m1.shape(), &[2, 1, 1, 1]);

        // Step 2: only slot 0 active for layer 0. Slot 0 offset becomes 2,
        // slot 1 stays at 1 → T_max on layer 0 = 2.
        let nk2 = Array::zeros_f32(&[1, 1, 1, 2]);
        let nv2 = Array::zeros_f32(&[1, 1, 1, 2]);
        let (k2, _, m2) = c.update_and_fetch_batched(0, &[0], &nk2, &nv2).unwrap();
        assert_eq!(k2.shape(), &[1, 1, 2, 2]);
        assert_eq!(m2.shape(), &[1, 1, 1, 2]);
        assert_eq!(c.offset(0), 2);
        assert_eq!(c.offset(1), 1);
    }

    #[test]
    fn fused_cache_offsets_are_per_layer() {
        let mut c = FusedBatchKVCache::new(cfg(2, 4, 1, 2), 1).unwrap();
        c.admit(0).unwrap();
        let nk = Array::zeros_f32(&[1, 1, 1, 2]);
        let nv = Array::zeros_f32(&[1, 1, 1, 2]);
        // Update layer 0 only — layer 1 offset must stay at 0.
        let _ = c.update_and_fetch_batched(0, &[0], &nk, &nv).unwrap();
        assert_eq!(c.offset_for(0, 0), 1);
        assert_eq!(c.offset_for(1, 0), 0);
        // Then layer 1 — independent advance.
        let _ = c.update_and_fetch_batched(1, &[0], &nk, &nv).unwrap();
        assert_eq!(c.offset_for(0, 0), 1);
        assert_eq!(c.offset_for(1, 0), 1);
    }

    #[test]
    fn fused_cache_rejects_shape_mismatch() {
        let mut c = FusedBatchKVCache::new(cfg(1, 4, 2, 3), 2).unwrap();
        c.admit(0).unwrap();
        let bad_k = Array::zeros_f32(&[1, 2, 1, 4]); // wrong head_dim
        let good_v = Array::zeros_f32(&[1, 2, 1, 3]);
        assert!(
            c.update_and_fetch_batched(0, &[0], &bad_k, &good_v)
                .is_err()
        );
    }
}
