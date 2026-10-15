//! [`QuantizedKvCache`]: one attention layer's TurboQuant KV cache.
//!
//! A sequence is stored in three parts, oldest first:
//!
//! ```text
//!  sinks        the first `sink_tokens` tokens, uncompressed
//!  cold         everything older than the recent window, quantized
//!  recent ring  the newest `recent_window` tokens (up to one eviction chunk
//!               more), uncompressed
//! ```
//!
//! The sinks are the tokens attention concentrates on whatever the
//! query is, so they are the ones whose quantization noise is multiplied by
//! the largest weights. The recent window is kept exact for the same reason
//! the first tokens are: locality puts most of the weight there. When the ring
//! fills, its oldest `eviction chunk` tokens are quantized into the cold
//! store in one batch.
//!
//! A decode step attends to all three parts in one pass (see
//! `cpp/bridge_turboquant_attention.cpp`), reading the cold part packed.

use std::sync::Arc;

use crate::InlineArray;
use crate::compat::Dtype;

use super::attention::{DecodeShape, Visibility, decode_attention};
use super::config::{TurboQuantConfig, eviction_chunk};
use super::state::TurboQuantState;
use super::store::{ColdStore, DenseStore, KeyFrame};

/// Tokens quantized per piece when a long run of them is (a prompt, a large
/// eviction): 4096 tokens of 8 KV heads at head dimension 128 rotate through
/// 17 MB of f32.
const ENCODE_PIECE_TOKENS: usize = 4096;

/// The fewest tokens the key frame is calibrated from.
const FRAME_MIN_TOKENS: usize = 16;

/// Shape of what the cache holds, fixed by the first append.
#[derive(Debug, Clone, Copy)]
struct Layout {
    batch: usize,
    heads: usize,
    dim: usize,
    value_dim: usize,
    /// dtype of the uncompressed tokens (that of the first keys appended).
    dtype: i32,
}

impl Layout {
    fn kv_rows(&self) -> usize {
        self.batch * self.heads
    }
}

/// Compressed KV cache of one attention layer.
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    /// Tokens cached: sinks, cold and recent together.
    pub offset: usize,
    /// The configuration the cache was built with.
    pub config: TurboQuantConfig,
    /// Rotations and codebooks, shared between the layers of a model.
    pub state: Option<Arc<TurboQuantState>>,
    layout: Option<Layout>,
    cold: Option<ColdStore>,
    dense: Option<DenseStore>,
}

impl QuantizedKvCache {
    /// An empty cache that builds its quantizers on the first append.
    pub fn new(config: TurboQuantConfig) -> Self {
        Self {
            offset: 0,
            config,
            state: None,
            layout: None,
            cold: None,
            dense: None,
        }
    }

    /// An empty cache that uses `state`, built once for the model and shared
    /// by every layer.
    pub fn with_state(config: TurboQuantConfig, state: Arc<TurboQuantState>) -> Self {
        let mut cache = Self::new(config);
        cache.state = Some(state);
        cache
    }

    /// Tokens cached.
    pub fn len(&self) -> usize {
        self.offset
    }

    /// True when nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Tokens held uncompressed: the sinks and the recent ring.
    pub fn hot_len(&self) -> usize {
        self.dense.as_ref().map_or(0, DenseStore::tokens)
    }

    /// Tokens held quantized.
    pub fn cold_len(&self) -> usize {
        self.cold.as_ref().map_or(0, |c| c.len)
    }

    /// Bytes of device memory the cached tokens occupy (capacity included).
    pub fn nbytes(&self) -> usize {
        let cold = self.cold.as_ref().map_or(0, |c| {
            c.k.packed.nbytes()
                + c.k.scales.nbytes()
                + c.v.packed.nbytes()
                + c.v.scales.nbytes()
                + c.frame.as_ref().map_or(0, |f| {
                    f.center.nbytes() + f.scale.as_ref().map_or(0, InlineArray::nbytes)
                })
        });
        let dense = self
            .dense
            .as_ref()
            .map_or(0, |d| d.keys.nbytes() + d.values.nbytes());
        cold + dense
    }

    /// Forget every cached token, keeping the configuration and the state.
    pub fn reset(&mut self) {
        self.offset = 0;
        self.layout = None;
        self.cold = None;
        self.dense = None;
    }

    /// Take back the newest `n` tokens (all of them if fewer are cached) and
    /// return how many went: the speculative-decoding rollback.
    ///
    /// The ring gives its tokens back exact; tokens already quantized stay
    /// quantized, so after a deep cut the newest tokens are no longer the
    /// exact ones. The sinks are only touched once everything after them is
    /// gone.
    pub fn truncate(&mut self, n: usize) -> usize {
        let n = n.min(self.offset);
        if n == self.offset {
            self.reset();
            return n;
        }
        let mut left = n;
        if let Some(dense) = self.dense.as_mut() {
            left -= dense.drop_newest_ring(left);
        }
        if let Some(cold) = self.cold.as_mut() {
            left -= cold.drop_newest(left);
        }
        if let Some(dense) = self.dense.as_mut() {
            left -= dense.drop_newest_sinks(left);
        }
        debug_assert_eq!(left, 0, "{n} tokens taken back from {} cached", self.offset);
        self.offset -= n;
        n
    }

    /// An independent copy that shares storage with this cache until either
    /// side appends. MLX arrays are immutable values and every write makes a
    /// new one, so this costs a reference count per array, however long the
    /// cached sequence: the cheap prefix share a serving prefix cache needs.
    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Cut the lazy graph behind the cached arrays by evaluating them.
    pub fn eval_and_detach_gpu_state(&mut self) {
        let mut pending: Vec<&mut InlineArray> = Vec::new();
        if let Some(cold) = self.cold.as_mut() {
            pending.extend(cold.arrays_mut());
        }
        if let Some(dense) = self.dense.as_mut() {
            pending.extend(dense.arrays_mut());
        }
        if !pending.is_empty() {
            crate::inline_array::eval_and_detach_many(&mut pending);
        }
    }

    fn ensure_layout(
        &mut self,
        keys: &InlineArray,
        values: &InlineArray,
    ) -> Result<Layout, String> {
        if keys.ndim() != 4 || values.ndim() != 4 {
            return Err(format!(
                "TurboQuant: expected [B, H, S, D] keys and values, got {} and {} dimensions",
                keys.ndim(),
                values.ndim()
            ));
        }
        if keys.dim(0) != values.dim(0)
            || keys.dim(1) != values.dim(1)
            || keys.dim(2) != values.dim(2)
        {
            return Err(format!(
                "TurboQuant: keys {:?} and values {:?} disagree on batch, heads or length",
                keys.shape(),
                values.shape()
            ));
        }
        let seen = (
            keys.dim(0) as usize,
            keys.dim(1) as usize,
            keys.dim(3) as usize,
            values.dim(3) as usize,
        );
        if seen.2 % 4 != 0 {
            return Err(format!(
                "TurboQuant: the key head dimension must be a multiple of 4, got {}",
                seen.2
            ));
        }
        if let Some(layout) = self.layout {
            if (layout.batch, layout.heads, layout.dim, layout.value_dim) != seen {
                return Err(format!(
                    "TurboQuant: the cache holds [{}, {}, *, {}] keys and [.., {}] values, \
                     cannot append [{}, {}, *, {}] keys and [.., {}] values",
                    layout.batch,
                    layout.heads,
                    layout.dim,
                    layout.value_dim,
                    seen.0,
                    seen.1,
                    seen.2,
                    seen.3
                ));
            }
            return Ok(layout);
        }
        let layout = Layout {
            batch: seen.0,
            heads: seen.1,
            dim: seen.2,
            value_dim: seen.3,
            dtype: keys.dtype_raw(),
        };
        let state = match &self.state {
            Some(state) => Arc::clone(state),
            None => {
                let state = Arc::new(TurboQuantState::new(
                    layout.dim,
                    layout.value_dim,
                    self.config,
                )?);
                self.state = Some(Arc::clone(&state));
                state
            }
        };
        if state.keys.dim() != layout.dim || state.values.dim() != layout.value_dim {
            return Err(format!(
                "TurboQuant: the shared state quantizes {}-wide keys and {}-wide values, \
                 the cache is being given {} and {}",
                state.keys.dim(),
                state.values.dim(),
                layout.dim,
                layout.value_dim
            ));
        }
        let ring_capacity = self
            .config
            .recent_window
            .map_or(0, |w| w + eviction_chunk(w));
        self.dense = Some(DenseStore::new(
            layout.kv_rows(),
            layout.dim,
            layout.value_dim,
            layout.dtype,
            self.config.sink_tokens,
            ring_capacity,
        ));
        self.cold = Some(ColdStore::empty(
            layout.kv_rows(),
            &state.keys,
            &state.values,
        ));
        // `empty` is a placeholder with room for one token; the first append
        // allocates for real.
        if let Some(cold) = self.cold.as_mut() {
            cold.len = 0;
        }
        self.layout = Some(layout);
        Ok(layout)
    }

    /// Append `[B, H, S, D]` keys and `[B, H, S, Dv]` values.
    pub fn append(&mut self, keys: &InlineArray, values: &InlineArray) -> Result<(), String> {
        let layout = self.ensure_layout(keys, values)?;
        let seq = keys.dim(2) as usize;
        if seq == 0 {
            return Ok(());
        }
        let kv_rows = layout.kv_rows() as i32;
        let keys = keys
            .as_dtype(layout.dtype)
            .reshape(&[kv_rows, seq as i32, layout.dim as i32]);
        let values =
            values
                .as_dtype(layout.dtype)
                .reshape(&[kv_rows, seq as i32, layout.value_dim as i32]);

        let (sink_room, ring_capacity, ring_len, window) = {
            let dense = self
                .dense
                .as_ref()
                .expect("dense store exists after layout");
            (
                dense.sink_room(),
                dense.ring_capacity,
                dense.ring_len,
                self.config.recent_window,
            )
        };
        let to_sink = seq.min(sink_room);
        if to_sink > 0 {
            let dense = self.dense.as_mut().expect("dense store");
            dense.push(
                &slice_tokens(&keys, layout, 0, to_sink, true),
                &slice_tokens(&values, layout, 0, to_sink, false),
                to_sink,
            );
        }
        let rest = seq - to_sink;
        if rest > 0 {
            let rest_k = slice_tokens(&keys, layout, to_sink, seq, true);
            let rest_v = slice_tokens(&values, layout, to_sink, seq, false);
            match window {
                None => self.compress(&rest_k, &rest_v, rest)?,
                Some(window) => {
                    if ring_len + rest > ring_capacity {
                        if rest >= window {
                            // The new tokens alone fill the window: everything
                            // already in the ring goes cold, then all of the new
                            // tokens but the last `window`.
                            self.evict_ring(ring_len)?;
                            let direct = rest - window;
                            if direct > 0 {
                                self.compress(
                                    &slice_tokens(&rest_k, layout, 0, direct, true),
                                    &slice_tokens(&rest_v, layout, 0, direct, false),
                                    direct,
                                )?;
                            }
                            self.push_ring(
                                &slice_tokens(&rest_k, layout, direct, rest, true),
                                &slice_tokens(&rest_v, layout, direct, rest, false),
                                window,
                            );
                        } else {
                            self.evict_ring(ring_len + rest - window)?;
                            self.push_ring(&rest_k, &rest_v, rest);
                        }
                    } else {
                        self.push_ring(&rest_k, &rest_v, rest);
                    }
                }
            }
        }
        self.offset += seq;
        Ok(())
    }

    fn push_ring(&mut self, keys: &InlineArray, values: &InlineArray, n: usize) {
        self.dense
            .as_mut()
            .expect("dense store")
            .push(keys, values, n);
    }

    /// Move the `n` oldest recent tokens into the cold store.
    fn evict_ring(&mut self, n: usize) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        let (keys, values) = self.dense.as_mut().expect("dense store").evict(n);
        self.compress(&keys, &values, n)
    }

    /// Quantize `n` tokens (`[KV, n, D]` and `[KV, n, Dv]`) onto the cold store.
    fn compress(
        &mut self,
        keys: &InlineArray,
        values: &InlineArray,
        n: usize,
    ) -> Result<(), String> {
        let layout = self.layout.expect("layout");
        let state = Arc::clone(self.state.as_ref().expect("state"));
        let cold = self.cold.as_mut().expect("cold store");
        // One allocation for the whole run, then fixed-size pieces: each is
        // rotated into f32 and packed, and its temporaries are released
        // before the next, so a long prompt never holds more than a piece's.
        cold.reserve(&state.keys, &state.values, n);
        let conditioned = self.config.center_keys || self.config.scale_key_channels;
        if conditioned && cold.frame.is_none() {
            // Calibrated on the first tokens that go cold. Any frame gives
            // exact attention; this one removes the component the keys share
            // and evens out the channels, which is what it is for. A frame
            // taken from a few tokens would be mostly those tokens' own
            // content, which it would then add to every later key, so a
            // first run that short is stored as it is.
            cold.frame = Some(if n < FRAME_MIN_TOKENS {
                KeyFrame::calibrate(
                    &InlineArray::zeros(
                        &[layout.kv_rows() as i32, 1, layout.dim as i32],
                        Dtype::Float32.as_i32(),
                    ),
                    false,
                    false,
                )
            } else {
                let first = n.min(ENCODE_PIECE_TOKENS);
                KeyFrame::calibrate(
                    &keys.slice(
                        &[0, 0, 0],
                        &[layout.kv_rows() as i32, first as i32, layout.dim as i32],
                    ),
                    self.config.center_keys,
                    self.config.scale_key_channels,
                )
            });
        }
        let mut done = 0;
        while done < n {
            let piece = (n - done).min(ENCODE_PIECE_TOKENS);
            let rows = (layout.kv_rows() * piece) as i32;
            let take = |a: &InlineArray, width: usize| {
                a.slice(
                    &[0, done as i32, 0],
                    &[layout.kv_rows() as i32, (done + piece) as i32, width as i32],
                )
            };
            let take_keys = {
                let k = take(keys, layout.dim);
                match &cold.frame {
                    Some(frame) => frame.apply(&k),
                    None => k,
                }
            }
            .reshape(&[rows, layout.dim as i32]);
            let take_values =
                take(values, layout.value_dim).reshape(&[rows, layout.value_dim as i32]);
            let k = state.keys.encode(&take_keys)?;
            let v = state.values.encode(&take_values)?;
            cold.append(&state.keys, &state.values, &k, &v, piece);
            if n > ENCODE_PIECE_TOKENS {
                let mut pending = cold.arrays_mut();
                crate::inline_array::eval_and_detach_many(&mut pending);
            }
            done += piece;
        }
        Ok(())
    }

    /// Append one token and return the attention output of `queries` over the
    /// whole cache, `[B, Hq, 1, Dv]`. `keys` and `values` are `[B, H, 1, D]`.
    pub fn append_and_compute_attention(
        &mut self,
        queries: &InlineArray,
        keys: &InlineArray,
        values: &InlineArray,
        scale: f32,
    ) -> Result<InlineArray, String> {
        if keys.ndim() != 4 || keys.dim(2) != 1 || values.dim(2) != 1 || queries.dim(2) != 1 {
            return Err(
                "TurboQuant direct attention requires [B, H, 1, D] single-token decode inputs"
                    .to_string(),
            );
        }
        self.append(keys, values)?;
        self.compute_attention(queries, scale, 0, None)
    }

    /// Attention of one query token per head, `queries` `[B, Hq, 1, D]`, over
    /// the cached tokens at absolute positions `first_token..`, without
    /// appending anything. `sink_logits` is one logit per query head that
    /// joins the softmax normaliser without contributing a value.
    pub fn compute_attention(
        &self,
        queries: &InlineArray,
        scale: f32,
        first_token: usize,
        sink_logits: Option<&InlineArray>,
    ) -> Result<InlineArray, String> {
        let layout = self
            .layout
            .ok_or("TurboQuant: attention over an empty cache")?;
        let state = self.state.as_ref().ok_or("TurboQuant: no state")?;
        if queries.ndim() != 4 || queries.dim(2) != 1 {
            return Err(format!(
                "TurboQuant decode attention takes [B, Hq, 1, D] queries, got {:?}",
                queries.shape()
            ));
        }
        if queries.dim(0) as usize != layout.batch || queries.dim(3) as usize != layout.dim {
            return Err(format!(
                "TurboQuant: queries {:?} do not fit a cache of [{}, {}, *, {}]",
                queries.shape(),
                layout.batch,
                layout.heads,
                layout.dim
            ));
        }
        if queries.dim(1) as usize % layout.heads != 0 {
            return Err(format!(
                "TurboQuant: {} query heads is not a multiple of {} KV heads",
                queries.dim(1),
                layout.heads
            ));
        }
        let dense = self.dense.as_ref().expect("dense store");
        // Nothing quantized yet, nothing masked: the tokens are dense and
        // contiguous (the sinks, then the ring, which has not been evicted
        // from), so MLX's own SDPA reads them in place.
        if self.cold_len() == 0 && first_token == 0 && dense.tokens() > 0 {
            return self.dense_only_attention(queries, scale, sink_logits, layout, dense);
        }
        let shape = DecodeShape {
            batch: layout.batch,
            q_heads: queries.dim(1) as usize,
            kv_heads: layout.heads,
            dim: layout.dim,
            value_dim: layout.value_dim,
        };
        decode_attention(
            queries,
            scale,
            Visibility {
                first_token,
                sink_logits,
            },
            &shape,
            &state.keys,
            &state.values,
            self.cold.as_ref().filter(|c| c.len > 0),
            self.dense.as_ref().expect("dense store"),
        )
    }

    /// Decode attention when every cached token is dense: MLX's fused SDPA
    /// over the first `tokens` slots of the dense store.
    fn dense_only_attention(
        &self,
        queries: &InlineArray,
        scale: f32,
        sink_logits: Option<&InlineArray>,
        layout: Layout,
        dense: &DenseStore,
    ) -> Result<InlineArray, String> {
        let tokens = dense.tokens() as i32;
        let view = |store: &InlineArray, width: usize| -> InlineArray {
            let used = store
                .slice(&[0, 0, 0], &[layout.kv_rows() as i32, tokens, width as i32])
                .reshape(&[
                    layout.batch as i32,
                    layout.heads as i32,
                    tokens,
                    width as i32,
                ]);
            if used.dtype_raw() == queries.dtype_raw() {
                used
            } else {
                used.as_dtype(queries.dtype_raw())
            }
        };
        let keys = view(&dense.keys, layout.dim);
        let values = view(&dense.values, layout.value_dim);
        let out = match sink_logits {
            None => queries.sdpa(&keys, &values, scale, ""),
            Some(sinks) => queries.sdpa_with_sinks(
                &keys,
                &values,
                scale,
                "",
                None,
                &sinks.as_dtype(Dtype::Float32.as_i32()),
            ),
        };
        crate::error::check_last_error().map_err(|err| err.to_string())?;
        Ok(out)
    }

    /// Every cached key as `[B, H, T, D]`, in the dtype the keys were appended
    /// in: the sinks and recent tokens exact, the rest dequantized.
    pub fn dequantize_keys(&self) -> Option<InlineArray> {
        self.dequantized(true)
    }

    /// Every cached value as `[B, H, T, Dv]`; see [`Self::dequantize_keys`].
    pub fn dequantize_values(&self) -> Option<InlineArray> {
        self.dequantized(false)
    }

    fn dequantized(&self, keys: bool) -> Option<InlineArray> {
        let layout = self.layout?;
        let state = self.state.as_ref()?;
        let dense = self.dense.as_ref()?;
        let cold = self.cold.as_ref().filter(|c| c.len > 0);
        let cold_part = match cold {
            Some(c) => {
                let (k, v) = c.dequantize(&state.keys, &state.values).ok()?;
                Some(if keys { k } else { v }.as_dtype(layout.dtype))
            }
            None => None,
        };
        let ordered = dense.ordered().map(|(k, v)| if keys { k } else { v });
        // Sinks first, then the cold store, then the ring.
        let sinks = dense.sink_len;
        let width = if keys { layout.dim } else { layout.value_dim };
        let parts: Vec<InlineArray> = match (ordered, cold_part) {
            (Some(all), Some(cold)) => {
                let kv = layout.kv_rows() as i32;
                let total = dense.tokens() as i32;
                let sink = all.slice(&[0, 0, 0], &[kv, sinks as i32, width as i32]);
                let ring = all.slice(&[0, sinks as i32, 0], &[kv, total, width as i32]);
                let mut parts = Vec::new();
                if sinks > 0 {
                    parts.push(sink);
                }
                parts.push(cold);
                if total as usize > sinks {
                    parts.push(ring);
                }
                parts
            }
            (Some(all), None) => vec![all],
            (None, Some(cold)) => vec![cold],
            (None, None) => return None,
        };
        let mut iter = parts.into_iter();
        let mut out = iter.next()?;
        for part in iter {
            out = out.concatenate_2(&part, 1);
        }
        Some(out.reshape(&[
            layout.batch as i32,
            layout.heads as i32,
            self.offset as i32,
            width as i32,
        ]))
    }

    /// Dtype of the uncompressed tokens, once something has been appended.
    pub fn dtype(&self) -> Option<Dtype> {
        self.layout.map(|l| Dtype::from_raw(l.dtype))
    }
}

/// Tokens `lo..hi` of a `[KV, S, width]` array.
fn slice_tokens(a: &InlineArray, layout: Layout, lo: usize, hi: usize, keys: bool) -> InlineArray {
    let width = if keys { layout.dim } else { layout.value_dim };
    a.slice(
        &[0, lo as i32, 0],
        &[layout.kv_rows() as i32, hi as i32, width as i32],
    )
}
