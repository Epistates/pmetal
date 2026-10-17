//! Fused batched KV cache for continuous-batching decode.
//!
//! # Why this exists
//!
//! The per-slot `BatchKVCache` (`Vec<KVCache>`) runs one forward per slot
//! per decode step. On Apple Silicon a decode step is bound by reading the
//! weights, so `N` forwards read them `N` times. `FusedBatchKVCache` keeps
//! the K/V of every active request in one `[rows, heads, capacity, D]` buffer
//! per layer so the model can run one `[N, 1]` forward per step: the weights
//! are read once for all `N` requests (the iteration-level batching of Orca,
//! Yu et al., OSDI 2022, with the requests' K/V padded to a common length and
//! masked, as a contiguous batch needs).
//!
//! # Layout
//!
//! Per layer:
//! - `layer_keys[i]: [rows, num_kv_heads, capacity, head_dim]`
//! - `layer_values[i]: [rows, num_kv_heads, capacity, value_head_dim]`
//!
//! Per row (one request):
//! - `offsets[layer][row]: usize` — number of valid tokens written so far.
//!
//! The buffers start empty and grow as requests arrive and lengthen: `rows`
//! in steps of [`ROW_STEP`] up to `max_slots`, `capacity` in steps of
//! [`CAPACITY_STEP`] up to the configuration's `max_seq_len`. Memory is
//! therefore `rows * capacity` positions, set by the longest request and the
//! most requests that have been in flight together, not by `max_slots *
//! max_seq_len`. [`release_if_idle`](FusedBatchKVCache::release_if_idle)
//! gives all of it back once no request is running.
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
//! - `mask: Option<[N_active, 1, 1, T_max]>` — additive mask, `-inf` at
//!   positions `t >= new_offset[slot]` (per-slot padding against the shared
//!   `T_max`); `None` when no row is padded, which is always the case for a
//!   lone request.
//!
//! The new tokens go in with one scatter per buffer, in place (the buffer has
//! a single owner while it is written), so a step costs the tokens it adds
//! and not the cache, whatever the number of requests. Rows that are not in
//! `active_indices` get a zero written at their own next position, which no
//! attention reads and a later token overwrites.
//!
//! A request enters the cache with [`load_slot`](FusedBatchKVCache::load_slot),
//! which copies what a single-request prefill left in an ordinary
//! [`KVCache`].
//!
//! # Scope
//!
//! Standard (non-quantized, non-sliding) caches only. TurboQuant, quantized
//! and sliding-window requests stay on the per-slot path.

use pmetal_runtime::compat::{Array, Dtype, Exception, ops};

use super::{CacheMode, KVCache, KVCacheConfig};

/// Rows are allocated this many at a time.
pub const ROW_STEP: usize = 4;

/// Positions per row are allocated this many at a time.
pub const CAPACITY_STEP: usize = 256;

/// An array built from the rows of a step and their offsets, kept for the
/// layers after the first, which ask for the same one: every layer of a step
/// has the same rows at the same offsets.
#[derive(Debug)]
struct PerStep {
    active: Vec<usize>,
    offsets: Vec<usize>,
    array: Array,
}

impl PerStep {
    fn get(memo: &Option<Self>, active: &[usize], offsets: &[usize]) -> Option<Array> {
        memo.as_ref()
            .filter(|m| m.active == active && m.offsets == offsets)
            .map(|m| m.array.clone())
    }
}

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
    /// Rows every layer's buffers have, or will have when next touched.
    rows: usize,
    /// Positions per row every layer's buffers have, or will have when next
    /// touched.
    capacity: usize,
    /// Per-layer K buffer: `[rows, num_kv_heads, capacity, head_dim]`, once
    /// the layer is first written.
    layer_keys: Vec<Option<Array>>,
    /// Per-layer V buffer: `[rows, num_kv_heads, capacity, value_head_dim]`.
    layer_values: Vec<Option<Array>>,
    /// Per-layer per-slot offsets: `offsets[layer_idx][batch_idx]`.
    offsets: Vec<Vec<usize>>,
    padding_mask: Option<PerStep>,
    positions: Option<PerStep>,
}

fn round_up(n: usize, step: usize) -> usize {
    n.div_ceil(step) * step
}

impl FusedBatchKVCache {
    /// Create an empty fused batched cache for up to `max_slots` requests.
    /// Nothing is allocated until requests arrive.
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
        let offsets = (0..config.num_layers).map(|_| vec![0; max_slots]).collect();
        Ok(Self {
            layer_keys: (0..config.num_layers).map(|_| None).collect(),
            layer_values: (0..config.num_layers).map(|_| None).collect(),
            config,
            max_slots,
            rows: 0,
            capacity: 0,
            offsets,
            padding_mask: None,
            positions: None,
        })
    }

    /// Configuration used to construct this cache.
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// Number of batch rows the cache can hold.
    pub fn max_slots(&self) -> usize {
        self.max_slots
    }

    /// Rows currently allocated.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Positions per row currently allocated.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes of K/V the cache holds now (or will, once every layer has been
    /// touched at the current size).
    pub fn memory_usage(&self) -> usize {
        Self::bytes_for(&self.config, self.rows, self.capacity)
    }

    /// Bytes `rows` rows of `capacity` positions take for `config`'s model.
    pub fn bytes_for(config: &KVCacheConfig, rows: usize, capacity: usize) -> usize {
        let per_position = config.clone().memory_footprint() / config.max_seq_len.max(1);
        rows * capacity * per_position
    }

    /// Layer-0 token offset for `batch_idx`.
    ///
    /// Every layer's offset advances in lockstep during a forward, so this is
    /// the position of the token being written only before layer 0's update;
    /// a layer reads its own with [`offset_for`](Self::offset_for) before its
    /// update.
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

    /// Make room for row `batch_idx`.
    fn reserve_row(&mut self, batch_idx: usize) -> Result<(), Exception> {
        if batch_idx >= self.max_slots {
            return Err(Exception::custom(format!(
                "FusedBatchKVCache: batch_idx {} out of range (max_slots {})",
                batch_idx, self.max_slots,
            )));
        }
        self.rows = self
            .rows
            .max(round_up(batch_idx + 1, ROW_STEP).min(self.max_slots));
        Ok(())
    }

    /// Make room for `tokens` positions in every row.
    fn reserve_positions(&mut self, tokens: usize) -> Result<(), Exception> {
        if tokens > self.config.max_seq_len {
            return Err(Exception::custom(format!(
                "FusedBatchKVCache: {tokens} positions exceed max_seq_len {}",
                self.config.max_seq_len,
            )));
        }
        self.capacity = self
            .capacity
            .max(round_up(tokens, CAPACITY_STEP).min(self.config.max_seq_len));
        Ok(())
    }

    /// Reset a slot's offset back to 0 across all layers and make sure it has
    /// a row. No buffer zeroing — the next write overwrites position 0.
    pub fn admit(&mut self, batch_idx: usize) -> Result<(), Exception> {
        self.reserve_row(batch_idx)?;
        self.release(batch_idx);
        Ok(())
    }

    /// Release a slot; offset reset to 0 across all layers so the next admit starts clean.
    pub fn release(&mut self, batch_idx: usize) {
        for layer in self.offsets.iter_mut() {
            if let Some(off) = layer.get_mut(batch_idx) {
                *off = 0;
            }
        }
        self.padding_mask = None;
        self.positions = None;
    }

    /// Free every buffer when no slot holds a token, so a server with no
    /// request in flight holds no K/V. Returns whether it did.
    pub fn release_if_idle(&mut self) -> bool {
        if self.offsets.iter().flatten().any(|&o| o > 0) {
            return false;
        }
        self.layer_keys.iter_mut().for_each(|b| *b = None);
        self.layer_values.iter_mut().for_each(|b| *b = None);
        self.rows = 0;
        self.capacity = 0;
        self.padding_mask = None;
        self.positions = None;
        true
    }

    /// Bring layer `layer_idx`'s buffers to the cache's current `rows` and
    /// `capacity`, allocating them on first use and growing them since.
    fn fit_layer(&mut self, layer_idx: usize) {
        let heads = self.config.num_kv_heads as i32;
        let (rows, capacity) = (self.rows as i32, self.capacity as i32);
        let dtype = self.config.dtype;
        for (slot, dim) in [
            (&mut self.layer_keys[layer_idx], self.config.head_dim as i32),
            (
                &mut self.layer_values[layer_idx],
                self.config.value_head_dim as i32,
            ),
        ] {
            let want = [rows, heads, capacity, dim];
            let fitted = match slot.take() {
                None => ops::zeros(&want, dtype),
                Some(buffer) if buffer.shape() == want => buffer,
                Some(mut buffer) => {
                    let have = buffer.shape().to_vec();
                    if have[2] < capacity {
                        let pad = ops::zeros(&[have[0], heads, capacity - have[2], dim], dtype);
                        buffer = ops::concatenate_axis(&[&buffer, &pad], 2);
                    }
                    if have[0] < rows {
                        let pad = ops::zeros(&[rows - have[0], heads, capacity, dim], dtype);
                        buffer = ops::concatenate_axis(&[&buffer, &pad], 0);
                    }
                    buffer
                }
            };
            *slot = Some(fitted);
        }
    }

    /// Copy what an ordinary single-request [`KVCache`] holds (a prefill, or a
    /// prefix restored from the prefix cache) into row `batch_idx`, which
    /// then decodes on from it.
    ///
    /// # Errors
    /// When the cache is not a plain dense one, holds no tokens, or holds no
    /// room for the next token; or `batch_idx` is out of range.
    pub fn load_slot(&mut self, batch_idx: usize, source: &KVCache) -> Result<(), Exception> {
        if !matches!(source.config().mode, CacheMode::Standard) {
            return Err(Exception::custom(format!(
                "FusedBatchKVCache::load_slot: a {:?} cache cannot be loaded into the fused one",
                source.config().mode,
            )));
        }
        self.reserve_row(batch_idx)?;
        let heads = self.config.num_kv_heads as i32;
        let mut layers = Vec::with_capacity(self.config.num_layers);
        for layer in 0..self.config.num_layers {
            let (k, v) = source.get(layer).ok_or_else(|| {
                Exception::custom(format!(
                    "FusedBatchKVCache::load_slot: layer {layer} of the source holds no tokens"
                ))
            })?;
            let have = k.dim(2);
            if k.shape() != [1, heads, have, self.config.head_dim as i32] || v.dim(2) != have {
                return Err(Exception::custom(format!(
                    "FusedBatchKVCache::load_slot: layer {layer} holds {:?} / {:?}, \
                     not one row of {heads} heads",
                    k.shape(),
                    v.shape(),
                )));
            }
            layers.push((k, v, have as usize));
        }
        // Room for every layer's tokens and for the next one.
        let longest = layers.iter().map(|&(.., n)| n).max().unwrap_or(0);
        self.reserve_positions(longest + 1)?;
        let slot = batch_idx as i32;
        for (layer, (k, v, have)) in layers.into_iter().enumerate() {
            self.fit_layer(layer);
            let have_i = have as i32;
            let (dk, dv) = (
                self.config.head_dim as i32,
                self.config.value_head_dim as i32,
            );
            let dtype_k = self.layer_keys[layer].as_ref().map(Array::dtype_raw);
            let dtype_v = self.layer_values[layer].as_ref().map(Array::dtype_raw);
            let (k, v) = (
                k.as_dtype(dtype_k.expect("fitted")),
                v.as_dtype(dtype_v.expect("fitted")),
            );
            let keys = self.layer_keys[layer].take().expect("fitted");
            let values = self.layer_values[layer].take().expect("fitted");
            self.layer_keys[layer] =
                Some(keys.slice_set(&k, &[slot, 0, 0, 0], &[slot + 1, heads, have_i, dk]));
            self.layer_values[layer] =
                Some(values.slice_set(&v, &[slot, 0, 0, 0], &[slot + 1, heads, have_i, dv]));
            self.offsets[layer][batch_idx] = have;
        }
        self.padding_mask = None;
        self.positions = None;
        Ok(())
    }

    /// The position each of rows `active` is about to write in layer
    /// `layer_idx`, `[N]` int32: what the layer's RoPE rotates the new token
    /// by. Read it before that layer's update.
    pub fn positions(&mut self, layer_idx: usize, active: &[usize]) -> Array {
        let offsets: Vec<usize> = active
            .iter()
            .map(|&row| self.offset_for(layer_idx, row))
            .collect();
        if let Some(array) = PerStep::get(&self.positions, active, &offsets) {
            return array;
        }
        let as_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
        let array = Array::from_i32_slice(&as_i32);
        self.positions = Some(PerStep {
            active: active.to_vec(),
            offsets,
            array: array.clone(),
        });
        array
    }

    /// The additive padding mask `[N, 1, 1, t_max]` for rows `active` whose
    /// offsets are `offsets` (after the step's write): `0` where a position
    /// is in the row, `-inf` past it.
    fn padding_mask(&mut self, active: &[usize], offsets: &[usize], t_max: i32) -> Array {
        if let Some(mask) = PerStep::get(&self.padding_mask, active, offsets) {
            return mask;
        }
        let n_active = active.len() as i32;
        let off_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
        let offs = Array::from_i32_slice_shaped(&off_i32, &[n_active, 1, 1, 1]);
        let positions = ops::arange_range(0, t_max).reshape(&[1, 1, 1, t_max]);
        let valid = ops::less(&positions, &offs);
        // In the cache's dtype, which is the queries' for a model that keeps
        // its cache in its own: attention takes it as it is.
        let zero = Array::from_f32(0.0).as_dtype(self.config.dtype.as_i32());
        let neg_inf = Array::from_f32(f32::NEG_INFINITY).as_dtype(self.config.dtype.as_i32());
        let mask = ops::r#where(&valid, &zero, &neg_inf);
        self.padding_mask = Some(PerStep {
            active: active.to_vec(),
            offsets: offsets.to_vec(),
            array: mask.clone(),
        });
        mask
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
    /// - `mask: [N_active, 1, 1, T_max]` additive `-inf`/`0.0`, or `None`
    ///   when every active row reaches `T_max` and nothing is padded.
    pub fn update_and_fetch_batched(
        &mut self,
        layer_idx: usize,
        active_indices: &[usize],
        new_k: &Array,
        new_v: &Array,
    ) -> Result<(Array, Array, Option<Array>), Exception> {
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

        let mut seen = vec![false; self.max_slots];
        for &idx in active_indices {
            if idx >= self.max_slots {
                return Err(Exception::custom(format!(
                    "FusedBatchKVCache: active batch_idx {} out of range (max_slots {})",
                    idx, self.max_slots,
                )));
            }
            if std::mem::replace(&mut seen[idx], true) {
                return Err(Exception::custom(format!(
                    "FusedBatchKVCache: slot {idx} appears twice in active_indices",
                )));
            }
            if self.offsets[layer_idx][idx] + 1 > self.config.max_seq_len {
                return Err(Exception::custom(format!(
                    "FusedBatchKVCache: slot {} exceeds max_seq_len {}",
                    idx, self.config.max_seq_len,
                )));
            }
        }
        for &idx in active_indices {
            self.reserve_row(idx)?;
        }
        // Room for the next token of every row, the ones not in this step
        // too: they get a placeholder written at their own next position.
        let deepest = self.offsets[layer_idx].iter().copied().max().unwrap_or(0);
        self.reserve_positions(deepest + 1)?;
        self.fit_layer(layer_idx);
        let rows = self.rows;

        // Where each row writes, and which of the step's tokens it writes.
        // A row that is not in the step takes the zero row appended to the
        // new tokens.
        let positions: Vec<i32> = (0..rows)
            .map(|r| self.offsets[layer_idx][r] as i32)
            .collect();
        let mut source = vec![n_active as i32; rows];
        for (n, &slot) in active_indices.iter().enumerate() {
            source[slot] = n as i32;
        }
        let in_order = n_active == rows && active_indices.iter().enumerate().all(|(n, &s)| n == s);

        let mut k_full = self.layer_keys[layer_idx].take().expect("fitted");
        let mut v_full = self.layer_values[layer_idx].take().expect("fitted");
        let spread = |new: &Array, dim: i32, dtype: i32| -> Array {
            let new = new.as_dtype(dtype);
            if in_order {
                return new;
            }
            let zero_row = ops::zeros(&[1, heads, 1, dim], Dtype::Float32).as_dtype(dtype);
            let with_zero = ops::concatenate_axis(&[&new, &zero_row], 0);
            with_zero.take_axis(&Array::from_i32_slice(&source), 0)
        };
        if n_active == 1 {
            // One row: a slice update in place, as a single request's cache
            // does, instead of a scatter over every row.
            let (slot, at) = (active_indices[0] as i32, positions[active_indices[0]]);
            let k_new = new_k.as_dtype(k_full.dtype_raw());
            let v_new = new_v.as_dtype(v_full.dtype_raw());
            k_full = k_full.slice_set(&k_new, &[slot, 0, at, 0], &[slot + 1, heads, at + 1, d_k]);
            v_full = v_full.slice_set(&v_new, &[slot, 0, at, 0], &[slot + 1, heads, at + 1, d_v]);
        } else {
            let at = Array::from_i32_slice_shaped(&positions, &[rows as i32, 1, 1, 1])
                .broadcast_to(&[rows as i32, heads, 1, 1]);
            let k_rows = spread(new_k, d_k, k_full.dtype_raw());
            let v_rows = spread(new_v, d_v, v_full.dtype_raw());
            k_full = k_full.put_along_axis_op(
                &at.broadcast_to(&[rows as i32, heads, 1, d_k]),
                &k_rows,
                2,
            );
            v_full = v_full.put_along_axis_op(
                &at.broadcast_to(&[rows as i32, heads, 1, d_v]),
                &v_rows,
                2,
            );
        }

        // Advance offsets for this layer only.
        for &idx in active_indices {
            self.offsets[layer_idx][idx] += 1;
        }

        // The rows to attend over, as far as the longest of them reaches
        // (T_max = max(new_offset) among active slots). Slots that are
        // consecutive and in order are a view of the buffer; any other
        // choice is gathered into a batch.
        let new_offsets: Vec<usize> = active_indices
            .iter()
            .map(|&i| self.offsets[layer_idx][i])
            .collect();
        let t_max = new_offsets.iter().copied().max().unwrap_or(0);
        let t_max_i32 = t_max as i32;
        let first = active_indices[0];
        let consecutive = active_indices
            .iter()
            .enumerate()
            .all(|(row, &slot)| slot == first + row);
        let rows_of = |buf: &Array, dim: i32| -> Array {
            if consecutive {
                buf.slice(
                    &[first as i32, 0, 0, 0],
                    &[(first + n_active) as i32, heads, t_max_i32, dim],
                )
            } else {
                let gather = Array::from_i32_slice_shaped(
                    &active_indices.iter().map(|&i| i as i32).collect::<Vec<_>>(),
                    &[n_active as i32],
                );
                buf.slice(&[0, 0, 0, 0], &[rows as i32, heads, t_max_i32, dim])
                    .take_axis(&gather, 0)
            }
        };
        let k_out = rows_of(&k_full, d_k);
        let v_out = rows_of(&v_full, d_v);
        self.layer_keys[layer_idx] = Some(k_full);
        self.layer_values[layer_idx] = Some(v_full);

        // Nothing is padded when every row reaches as far as the longest.
        let mask = new_offsets
            .iter()
            .any(|&o| o != t_max)
            .then(|| self.padding_mask(active_indices, &new_offsets, t_max_i32));
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
        assert!(mask.is_none(), "two rows of one token each need no padding");
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
        assert!(m1.is_none());

        // Step 2: only slot 0 active for layer 0. Slot 0 offset becomes 2,
        // slot 1 stays at 1 → T_max on layer 0 = 2.
        let nk2 = Array::zeros_f32(&[1, 1, 1, 2]);
        let nv2 = Array::zeros_f32(&[1, 1, 1, 2]);
        let (k2, _, m2) = c.update_and_fetch_batched(0, &[0], &nk2, &nv2).unwrap();
        assert_eq!(k2.shape(), &[1, 1, 2, 2]);
        assert!(m2.is_none(), "a lone row has nothing to pad against");
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

    use crate::kv_cache::reference_tests::{Rng, array};
    use crate::test_utils::to_f32_vec_eval;

    const H: usize = 2;
    const D: usize = 4;
    const LAYERS: usize = 3;

    /// `[layer][token][H * D]`.
    type LayerTokens = Vec<Vec<Vec<f32>>>;

    /// A dense one-row cache holding `tokens` tokens of random keys and
    /// values in every layer; also returns them as `[layer][token][H * D]`.
    fn prefilled(rng: &mut Rng, tokens: usize) -> (KVCache, LayerTokens, LayerTokens) {
        let mut cache = KVCache::new(KVCacheConfig::new(LAYERS, 2048, H, D));
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for layer in 0..LAYERS {
            let k = rng.normals(tokens * H * D, 1.0);
            let v = rng.normals(tokens * H * D, 1.0);
            // The cache's layout is [1, H, T, D]; ours is token-major.
            let to_cache = |flat: &[f32]| -> Array {
                let mut laid = vec![0.0f32; flat.len()];
                for t in 0..tokens {
                    for h in 0..H {
                        for d in 0..D {
                            laid[(h * tokens + t) * D + d] = flat[(t * H + h) * D + d];
                        }
                    }
                }
                array(&laid, &[1, H, tokens, D])
            };
            cache
                .update_and_fetch(layer, &to_cache(&k), &to_cache(&v))
                .unwrap();
            keys.push(k.chunks(H * D).map(<[f32]>::to_vec).collect());
            values.push(v.chunks(H * D).map(<[f32]>::to_vec).collect());
        }
        (cache, keys, values)
    }

    /// A row's keys as `[token][H * D]` out of a fetched `[N, H, T, D]`.
    fn row_tokens(fetched: &Array, row: usize, tokens: usize) -> Vec<Vec<f32>> {
        let t_max = fetched.dim(2) as usize;
        let flat = to_f32_vec_eval(fetched);
        (0..tokens)
            .map(|t| {
                (0..H)
                    .flat_map(|h| {
                        let base = ((row * H + h) * t_max + t) * D;
                        flat[base..base + D].to_vec()
                    })
                    .collect()
            })
            .collect()
    }

    /// Requests prefilled elsewhere and loaded into rows keep every token
    /// through decode steps that run for some rows and not others, and a row
    /// that was idle for a while picks up where it left off.
    #[test]
    fn loaded_rows_keep_their_tokens_through_uneven_decoding() {
        let mut rng = Rng::new(7);
        let lengths = [5usize, 1, 9];
        let rows = [2usize, 0, 5];
        let mut cache = FusedBatchKVCache::new(KVCacheConfig::new(LAYERS, 2048, H, D), 6).unwrap();
        let mut keys: Vec<LayerTokens> = Vec::new();
        let mut values: Vec<LayerTokens> = Vec::new();
        for (&len, &row) in lengths.iter().zip(&rows) {
            let (source, k, v) = prefilled(&mut rng, len);
            cache.admit(row).unwrap();
            cache.load_slot(row, &source).unwrap();
            assert_eq!(cache.offset(row), len);
            keys.push(k);
            values.push(v);
        }
        assert_eq!(cache.rows(), 6, "rows grow in steps, up to max_slots");

        // Steps over different subsets of the three requests.
        let schedule: [&[usize]; 5] = [&[0, 1, 2], &[2], &[1, 2], &[0], &[0, 1, 2]];
        for (step, subset) in schedule.iter().enumerate() {
            let active: Vec<usize> = subset.iter().map(|&i| rows[i]).collect();
            for layer in 0..LAYERS {
                let new: Vec<Vec<f32>> = subset.iter().map(|_| rng.normals(H * D, 1.0)).collect();
                let flat: Vec<f32> = new.iter().flatten().copied().collect();
                // [N, H, 1, D]: the token-major data is already [H][D] per row.
                let nk = array(&flat, &[subset.len(), H, 1, D]);
                let nv = nk.clone();
                let (k, v, _mask) = cache
                    .update_and_fetch_batched(layer, &active, &nk, &nv)
                    .unwrap();
                for (n, &i) in subset.iter().enumerate() {
                    keys[i][layer].push(new[n].clone());
                    values[i][layer].push(new[n].clone());
                    let have = keys[i][layer].len();
                    assert_eq!(
                        row_tokens(&k, n, have),
                        keys[i][layer],
                        "step {step}, layer {layer}: keys of request {i}"
                    );
                    assert_eq!(
                        row_tokens(&v, n, have),
                        values[i][layer],
                        "step {step}, layer {layer}: values of request {i}"
                    );
                }
            }
        }
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    #[test]
    fn buffers_grow_with_the_requests_and_are_given_back_when_idle() {
        let mut cache = FusedBatchKVCache::new(KVCacheConfig::new(LAYERS, 4096, H, D), 16).unwrap();
        assert_eq!(
            (cache.rows(), cache.capacity(), cache.memory_usage()),
            (0, 0, 0)
        );

        let mut rng = Rng::new(3);
        let (short, ..) = prefilled(&mut rng, 10);
        cache.admit(0).unwrap();
        cache.load_slot(0, &short).unwrap();
        assert_eq!((cache.rows(), cache.capacity()), (ROW_STEP, CAPACITY_STEP));
        let small = cache.memory_usage();

        // A longer request, in a later row.
        let (long, ..) = prefilled(&mut rng, 600);
        cache.admit(5).unwrap();
        cache.load_slot(5, &long).unwrap();
        assert_eq!(
            (cache.rows(), cache.capacity()),
            (2 * ROW_STEP, 3 * CAPACITY_STEP)
        );
        assert!(cache.memory_usage() > small);
        assert_eq!(
            cache.memory_usage(),
            FusedBatchKVCache::bytes_for(cache.config(), cache.rows(), cache.capacity())
        );

        assert!(!cache.release_if_idle(), "requests still hold tokens");
        cache.release(0);
        assert!(!cache.release_if_idle());
        cache.release(5);
        assert!(cache.release_if_idle());
        assert_eq!((cache.rows(), cache.capacity()), (0, 0));
    }

    #[test]
    fn a_cache_that_is_not_plain_dense_cannot_be_loaded() {
        let mut cache = FusedBatchKVCache::new(KVCacheConfig::new(1, 64, H, D), 2).unwrap();
        let quantized = KVCache::new(KVCacheConfig::new(1, 64, H, D).with_quantized(8, 64));
        assert!(cache.load_slot(0, &quantized).is_err());
        let empty = KVCache::new(KVCacheConfig::new(1, 64, H, D));
        assert!(cache.load_slot(0, &empty).is_err());
    }

    /// Growing the cache does not hold two of it: MLX frees each layer's old
    /// buffer as its replacement is built, so a step that grows every layer
    /// peaks a layer or two above the grown cache, not a whole cache.
    #[test]
    fn growing_does_not_hold_the_cache_twice() {
        use pmetal_runtime::inline_array::{get_active_memory, get_peak_memory, reset_peak_memory};

        let (layers, heads, dim, tokens) = (12usize, 8usize, 64usize, 2048usize);
        let config = KVCacheConfig::new(layers, 4096, heads, dim).with_dtype(Dtype::Bfloat16);
        let mut cache = FusedBatchKVCache::new(config.clone(), 4).unwrap();
        let mut source = KVCache::new(config.clone());
        let ctx = ops::zeros(
            &[1, heads as i32, tokens as i32, dim as i32],
            Dtype::Bfloat16,
        );
        for layer in 0..layers {
            source.update_and_fetch(layer, &ctx, &ctx).unwrap();
        }
        cache.admit(0).unwrap();
        // 2048 tokens and room for one more: a whole step of capacity.
        cache.load_slot(0, &source).unwrap();
        let new = ops::zeros(&[1, heads as i32, 1, dim as i32], Dtype::Bfloat16);
        let step = |cache: &mut FusedBatchKVCache| {
            let mut outs = Vec::new();
            for layer in 0..layers {
                let (k, v, _) = cache
                    .update_and_fetch_batched(layer, &[0], &new, &new)
                    .unwrap();
                outs.push(k);
                outs.push(v);
            }
            for out in &outs {
                out.eval();
            }
        };
        // Fill the capacity there is, then take the step that grows it.
        while cache.offset(0) < cache.capacity() {
            step(&mut cache);
        }
        let cache_bytes = cache.memory_usage();
        assert!(cache_bytes > 20 << 20, "the cache is big enough to measure");
        reset_peak_memory();
        let before = get_active_memory();
        let capacity = cache.capacity();
        step(&mut cache);
        assert!(cache.capacity() > capacity, "the step grew the cache");
        let extra_peak = get_peak_memory().saturating_sub(before);
        let grown_by = cache.memory_usage() - cache_bytes;
        // The growth itself, and one layer's buffers twice over; not the
        // whole old cache beside the new.
        assert!(
            extra_peak < grown_by + cache_bytes / 4,
            "growing a {} MiB cache peaked {} MiB above it (it grew by {} MiB)",
            cache_bytes >> 20,
            extra_peak >> 20,
            grown_by >> 20
        );
        pmetal_runtime::check_unobserved_error().unwrap();
    }

    /// A decode step writes into the buffer it already has: the K/V of every
    /// request is not copied per token.
    #[test]
    fn a_decode_step_writes_in_place() {
        let mut rng = Rng::new(11);
        let mut cache = FusedBatchKVCache::new(KVCacheConfig::new(1, 2048, H, D), 4).unwrap();
        for row in 0..3 {
            let (source, ..) = prefilled(&mut rng, 20);
            cache.admit(row).unwrap();
            cache.load_slot(row, &source).unwrap();
        }
        let active = [0usize, 1, 2];
        let address = |cache: &FusedBatchKVCache| cache.layer_keys[0].as_ref().unwrap().data_ptr();
        let mut moves = 0;
        for _ in 0..24 {
            let before = address(&cache);
            let nk = array(&rng.normals(3 * H * D, 1.0), &[3, H, 1, D]);
            let (k, v, m) = cache
                .update_and_fetch_batched(0, &active, &nk, &nk)
                .unwrap();
            k.eval();
            v.eval();
            assert!(m.is_none());
            drop((k, v, m));
            if address(&cache) != before {
                moves += 1;
            }
        }
        // MLX hands a buffer over once the previous command buffer has let
        // go of it, which on a loaded machine is sometimes later than the next
        // write; a cache that copies every step moves at all of them.
        assert!(moves <= 6, "{moves} of 24 steps moved the buffer");
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
