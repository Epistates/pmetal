//! Storage behind a [`QuantizedKvCache`](super::QuantizedKvCache): the packed
//! cold store and the dense store that holds the attention sinks and the
//! recent-token ring.
//!
//! Every array has the KV heads of all batch entries flattened into its first
//! axis (`KV = batch * kv_heads`, entry-major), the token axis second, so a
//! `[B, H, S, D]` append reshapes into place with no transposition.

use crate::InlineArray;
use crate::compat::Dtype;

use super::quantizer::{Encoded, Quantizer};

/// Capacities grow in steps of this many tokens.
const GROWTH_STEP: usize = 256;

fn round_up(n: usize, step: usize) -> usize {
    n.div_ceil(step) * step
}

/// Packed rows of one tensor: `[KV, capacity, words]` u32 and `[KV, capacity]`
/// f32 scales.
#[derive(Debug, Clone)]
pub(super) struct PackedStore {
    pub(super) packed: InlineArray,
    pub(super) scales: InlineArray,
}

impl PackedStore {
    fn zeros(kv_rows: usize, capacity: usize, words: usize) -> Self {
        Self {
            packed: InlineArray::zeros(
                &[kv_rows as i32, capacity as i32, words as i32],
                Dtype::Uint32.as_i32(),
            ),
            scales: InlineArray::zeros(&[kv_rows as i32, capacity as i32], Dtype::Float32.as_i32()),
        }
    }

    fn grown(&self, kv_rows: usize, extra: usize, words: usize) -> Self {
        let pad = Self::zeros(kv_rows, extra, words);
        Self {
            packed: self.packed.concatenate_2(&pad.packed, 1),
            scales: self.scales.concatenate_2(&pad.scales, 1),
        }
    }

    fn write(&mut self, at: usize, rows: &Encoded, tokens: usize, kv_rows: usize, words: usize) {
        let packed = rows
            .packed
            .reshape(&[kv_rows as i32, tokens as i32, words as i32]);
        let scales = rows.scales.reshape(&[kv_rows as i32, tokens as i32]);
        self.packed = self.packed.slice_set(
            &packed,
            &[0, at as i32, 0],
            &[kv_rows as i32, (at + tokens) as i32, words as i32],
        );
        self.scales = self.scales.slice_set(
            &scales,
            &[0, at as i32],
            &[kv_rows as i32, (at + tokens) as i32],
        );
    }

    fn view(&self, kv_rows: usize, tokens: usize, words: usize) -> Encoded {
        Encoded {
            packed: self
                .packed
                .slice(&[0, 0, 0], &[kv_rows as i32, tokens as i32, words as i32])
                .reshape(&[(kv_rows * tokens) as i32, words as i32]),
            scales: self
                .scales
                .slice(&[0, 0], &[kv_rows as i32, tokens as i32])
                .reshape(&[(kv_rows * tokens) as i32]),
        }
    }
}

/// The compressed history: packed keys and values for tokens `0..len`.
#[derive(Debug, Clone)]
pub(super) struct ColdStore {
    pub(super) k: PackedStore,
    pub(super) v: PackedStore,
    /// Tokens stored.
    pub(super) len: usize,
    /// Tokens the arrays have room for.
    pub(super) capacity: usize,
    kv_rows: usize,
}

impl ColdStore {
    /// An empty store with room for one token, enough to hand the kernels
    /// valid (never read) operands.
    pub(super) fn empty(kv_rows: usize, keys: &Quantizer, values: &Quantizer) -> Self {
        Self {
            k: PackedStore::zeros(kv_rows, 1, keys.words()),
            v: PackedStore::zeros(kv_rows, 1, values.words()),
            len: 0,
            capacity: 1,
            kv_rows,
        }
    }

    /// Make room for `tokens` more, in one allocation.
    pub(super) fn reserve(&mut self, keys: &Quantizer, values: &Quantizer, tokens: usize) {
        let needed = self.len + tokens;
        if needed > self.capacity {
            let target = round_up(needed, GROWTH_STEP);
            let extra = target - self.capacity;
            self.k = self.k.grown(self.kv_rows, extra, keys.words());
            self.v = self.v.grown(self.kv_rows, extra, values.words());
            self.capacity = target;
        }
    }

    /// Append `tokens` quantized tokens per KV head (`rows` are ordered
    /// KV-head major, token minor).
    pub(super) fn append(
        &mut self,
        keys: &Quantizer,
        values: &Quantizer,
        k: &Encoded,
        v: &Encoded,
        tokens: usize,
    ) {
        let needed = self.len + tokens;
        if needed > self.capacity {
            // Geometric growth, so a long decode copies the store O(log n)
            // times, not once per eviction.
            self.reserve(
                keys,
                values,
                needed.max(self.capacity + self.capacity / 2) - self.len,
            );
        }
        self.k
            .write(self.len, k, tokens, self.kv_rows, keys.words());
        self.v
            .write(self.len, v, tokens, self.kv_rows, values.words());
        self.len = needed;
    }

    /// Keys and values of the whole store as `[KV, len, D]` f32 in the
    /// original basis.
    pub(super) fn dequantize(
        &self,
        keys: &Quantizer,
        values: &Quantizer,
    ) -> Result<(InlineArray, InlineArray), String> {
        let decode = |store: &PackedStore, q: &Quantizer| -> Result<InlineArray, String> {
            let rows = store.view(self.kv_rows, self.len, q.words());
            Ok(q.decode(&rows)?
                .reshape(&[self.kv_rows as i32, self.len as i32, q.dim() as i32]))
        };
        Ok((decode(&self.k, keys)?, decode(&self.v, values)?))
    }

    /// Every array, for [`eval`](InlineArray::eval)-and-detach.
    pub(super) fn arrays_mut(&mut self) -> [&mut InlineArray; 4] {
        [
            &mut self.k.packed,
            &mut self.k.scales,
            &mut self.v.packed,
            &mut self.v.scales,
        ]
    }
}

/// Uncompressed tokens: the first `sink_capacity` tokens of the sequence
/// (attention sinks) and a ring over the most recent ones.
///
/// Slots `0..sink_capacity` hold the sinks in order; the ring occupies the
/// `ring_capacity` slots after them.
#[derive(Debug, Clone)]
pub(super) struct DenseStore {
    /// `[KV, capacity, D]`.
    pub(super) keys: InlineArray,
    /// `[KV, capacity, Dv]`.
    pub(super) values: InlineArray,
    pub(super) capacity: usize,
    pub(super) sink_capacity: usize,
    pub(super) sink_len: usize,
    pub(super) ring_capacity: usize,
    pub(super) ring_start: usize,
    pub(super) ring_len: usize,
    kv_rows: usize,
    dim: usize,
    value_dim: usize,
}

impl DenseStore {
    pub(super) fn new(
        kv_rows: usize,
        dim: usize,
        value_dim: usize,
        dtype: i32,
        sink_capacity: usize,
        ring_capacity: usize,
    ) -> Self {
        let capacity = (sink_capacity + ring_capacity).max(1);
        Self {
            keys: InlineArray::zeros(&[kv_rows as i32, capacity as i32, dim as i32], dtype),
            values: InlineArray::zeros(&[kv_rows as i32, capacity as i32, value_dim as i32], dtype),
            capacity,
            sink_capacity,
            sink_len: 0,
            ring_capacity,
            ring_start: 0,
            ring_len: 0,
            kv_rows,
            dim,
            value_dim,
        }
    }

    pub(super) fn tokens(&self) -> usize {
        self.sink_len + self.ring_len
    }

    /// Sink slots still empty.
    pub(super) fn sink_room(&self) -> usize {
        self.sink_capacity - self.sink_len
    }

    /// Ring slots still free.
    pub(super) fn ring_room(&self) -> usize {
        self.ring_capacity - self.ring_len
    }

    fn write_slots(&mut self, src_k: &InlineArray, src_v: &InlineArray, slot: usize, n: usize) {
        let kv = self.kv_rows as i32;
        self.keys = self.keys.slice_set(
            src_k,
            &[0, slot as i32, 0],
            &[kv, (slot + n) as i32, self.dim as i32],
        );
        self.values = self.values.slice_set(
            src_v,
            &[0, slot as i32, 0],
            &[kv, (slot + n) as i32, self.value_dim as i32],
        );
    }

    fn token_range(&self, src: &InlineArray, lo: usize, hi: usize, width: usize) -> InlineArray {
        src.slice(
            &[0, lo as i32, 0],
            &[self.kv_rows as i32, hi as i32, width as i32],
        )
    }

    /// Store `n` tokens (`[KV, n, D]`, in the store's dtype): the first ones
    /// into any empty sink slots, the rest at the tail of the ring. The
    /// caller has made room.
    pub(super) fn push(&mut self, keys: &InlineArray, values: &InlineArray, n: usize) {
        debug_assert!(n <= self.sink_room() + self.ring_room());
        let to_sink = n.min(self.sink_room());
        if to_sink > 0 {
            let k = self.token_range(keys, 0, to_sink, self.dim);
            let v = self.token_range(values, 0, to_sink, self.value_dim);
            self.write_slots(&k, &v, self.sink_len, to_sink);
            self.sink_len += to_sink;
        }
        let mut done = to_sink;
        while done < n {
            let at = (self.ring_start + self.ring_len) % self.ring_capacity;
            let run = (n - done).min(self.ring_capacity - at);
            let k = self.token_range(keys, done, done + run, self.dim);
            let v = self.token_range(values, done, done + run, self.value_dim);
            self.write_slots(&k, &v, self.sink_capacity + at, run);
            self.ring_len += run;
            done += run;
        }
    }

    /// The `n` oldest ring tokens as `[KV, n, D]` and `[KV, n, Dv]`, removed
    /// from the ring.
    pub(super) fn evict(&mut self, n: usize) -> (InlineArray, InlineArray) {
        let out = self.ring_range(0, n);
        self.ring_start = (self.ring_start + n) % self.ring_capacity.max(1);
        self.ring_len -= n;
        out
    }

    /// Ring tokens `lo..hi` in logical order, `[KV, hi - lo, ·]`.
    fn ring_range(&self, lo: usize, hi: usize) -> (InlineArray, InlineArray) {
        let start = (self.ring_start + lo) % self.ring_capacity.max(1);
        let n = hi - lo;
        let first = n.min(self.ring_capacity - start);
        let take = |src: &InlineArray, width: usize| -> InlineArray {
            let a = self.token_range(
                src,
                self.sink_capacity + start,
                self.sink_capacity + start + first,
                width,
            );
            if first == n {
                a
            } else {
                let b = self.token_range(
                    src,
                    self.sink_capacity,
                    self.sink_capacity + (n - first),
                    width,
                );
                a.concatenate_2(&b, 1)
            }
        };
        (
            take(&self.keys, self.dim),
            take(&self.values, self.value_dim),
        )
    }

    /// The sinks followed by the ring in logical order: `[KV, tokens, ·]`.
    /// `None` when the store holds nothing.
    pub(super) fn ordered(&self) -> Option<(InlineArray, InlineArray)> {
        let sink = (self.sink_len > 0).then(|| {
            (
                self.token_range(&self.keys, 0, self.sink_len, self.dim),
                self.token_range(&self.values, 0, self.sink_len, self.value_dim),
            )
        });
        let ring = (self.ring_len > 0).then(|| self.ring_range(0, self.ring_len));
        match (sink, ring) {
            (Some((sk, sv)), Some((rk, rv))) => {
                Some((sk.concatenate_2(&rk, 1), sv.concatenate_2(&rv, 1)))
            }
            (Some(s), None) => Some(s),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }

    pub(super) fn arrays_mut(&mut self) -> [&mut InlineArray; 2] {
        [&mut self.keys, &mut self.values]
    }
}
