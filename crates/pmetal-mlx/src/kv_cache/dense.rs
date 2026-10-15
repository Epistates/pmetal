//! One layer of the dense KV cache: keys and values `[B, heads, T, dim]` kept
//! in the model's dtype, in a buffer written in place.
//!
//! # Layout
//!
//! A buffer holds `filled` tokens in its first slots and has room for more.
//! A layer that attends to everything (no [`Window`]) keeps every token. A
//! windowed layer keeps its first `sinks` tokens (the attention sinks) and
//! the newest ones, and drops what is between once the buffer is full:
//!
//! ```text
//! slots  [ sinks | oldest kept ........ newest | free ............ ]
//! tokens [ 0..K  | K+evicted ........... seen  |                   ]
//! ```
//!
//! # What a fetch returns
//!
//! A query at absolute position `p` attends to the sinks and to the
//! `recent` tokens ending at `p` (itself included); a [`Window`] of `size`
//! `W` and `sinks` `K` has `recent = W - K`. A call that adds `s` tokens
//! therefore needs the sinks, the `recent - 1` tokens before its first new
//! token and the new ones, and returns exactly those (everything it holds
//! when that is fewer). A single token (`s = 1`) gets `W` keys and every one
//! of them is visible to it. A longer chunk gets keys its earlier queries may
//! not see: the cache cannot know the mask the model wants, so
//! [`DenseLayer::window_mask`] gives the exact one.
//!
//! # Cost
//!
//! A token is one in-place `slice_set` into the buffer. A full buffer grows
//! by doubling, so appending costs `O(1)` amortised. A windowed buffer is
//! compacted (the sinks and the newest `recent - 1` tokens copied to a fresh
//! buffer) once per [`WINDOW_SLACK`] tokens, so a long generation holds
//! `window + slack` slots however long it runs and copies `window / slack`
//! rows per token.

use pmetal_runtime::compat::{Array, Exception, ops};

use super::CacheMode;

/// Tokens a windowed buffer holds beyond what its next call needs, so it is
/// compacted once per this many tokens instead of at every one.
const WINDOW_SLACK: usize = 256;

/// A dense buffer grows to a multiple of this many tokens.
const GROWTH_CHUNK: usize = 256;

/// Which tokens a windowed layer keeps; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Window {
    /// The first tokens, always kept.
    pub sinks: usize,
    /// The newest tokens kept, a query's own included. At least 1.
    pub recent: usize,
}

impl Window {
    /// The window of a cache mode, `None` for modes that keep everything.
    pub(super) fn of(mode: CacheMode) -> Option<Self> {
        match mode {
            CacheMode::SlidingWindow { window_size } => Some(Self {
                sinks: 0,
                recent: window_size.max(1),
            }),
            CacheMode::Rotating { max_size, keep } => {
                let size = max_size.max(1);
                let sinks = keep.min(size - 1);
                Some(Self {
                    sinks,
                    recent: size - sinks,
                })
            }
            _ => None,
        }
    }

    /// Tokens a single-token step attends to: the sinks and the recent ones.
    pub(super) fn size(self) -> usize {
        self.sinks + self.recent
    }
}

/// One layer's keys and values.
#[derive(Debug, Clone, Default)]
pub(super) struct DenseLayer {
    /// `[B, heads, capacity, key_dim]`.
    keys: Option<Array>,
    /// `[B, heads, capacity, value_dim]`.
    values: Option<Array>,
    /// Slots written, from the first.
    filled: usize,
    /// Tokens this layer has been given: the absolute position of the next
    /// one, which RoPE needs and `filled` stops tracking once tokens are
    /// dropped.
    seen: usize,
    /// Tokens dropped from between the sinks and the newest.
    evicted: usize,
}

impl DenseLayer {
    pub(super) fn seen(&self) -> usize {
        self.seen
    }

    pub(super) fn is_empty(&self) -> bool {
        self.seen == 0
    }

    /// Bytes the buffers occupy (their capacity, in the dtype they hold).
    pub(super) fn nbytes(&self) -> usize {
        self.keys.as_ref().map_or(0, Array::nbytes) + self.values.as_ref().map_or(0, Array::nbytes)
    }

    /// Tokens held that a single-token step could still attend to.
    pub(super) fn retained(&self, window: Option<Window>) -> usize {
        match window {
            Some(w) => self.filled.min(w.size()),
            None => self.filled,
        }
    }

    /// Append `new_keys` `[B, heads, s, key_dim]` and `new_values`, and
    /// return the keys and values the new tokens' queries attend to.
    pub(super) fn update(
        &mut self,
        new_keys: &Array,
        new_values: &Array,
        window: Option<Window>,
        max_seq_len: usize,
    ) -> Result<(Array, Array), Exception> {
        if new_keys.ndim() != 4 || new_values.ndim() != 4 {
            return Err(Exception::custom(format!(
                "KV cache: expected [B, heads, seq, dim] keys and values, got {} and {} dimensions",
                new_keys.ndim(),
                new_values.ndim()
            )));
        }
        let s = new_keys.dim(2) as usize;
        if new_values.dim(2) as usize != s
            || new_keys.dim(0) != new_values.dim(0)
            || new_keys.dim(1) != new_values.dim(1)
        {
            return Err(Exception::custom(format!(
                "KV cache: keys {:?} and values {:?} disagree on batch, heads or length",
                new_keys.shape(),
                new_values.shape()
            )));
        }
        if let Some(held) = &self.keys {
            let want = (held.dim(0), held.dim(1), held.dim(3));
            let got = (new_keys.dim(0), new_keys.dim(1), new_keys.dim(3));
            if want != got || self.values.as_ref().map(|v| v.dim(3)) != Some(new_values.dim(3)) {
                return Err(Exception::custom(format!(
                    "KV cache holds [{}, {}, *, {}] keys; cannot append {:?}",
                    want.0,
                    want.1,
                    want.2,
                    new_keys.shape()
                )));
            }
        }
        if window.is_none() && self.seen + s > max_seq_len {
            return Err(Exception::custom(format!(
                "KV cache exceeded max_seq_len: {} > {}",
                self.seen + s,
                max_seq_len
            )));
        }

        self.make_room(new_keys, new_values, s, window, max_seq_len);

        let (Some(key_buf), Some(value_buf)) = (self.keys.take(), self.values.take()) else {
            unreachable!("make_room allocated the buffers");
        };
        let (b, h) = (new_keys.dim(0), new_keys.dim(1));
        let (kd, vd) = (new_keys.dim(3), new_values.dim(3));
        let (start, stop) = (self.filled as i32, (self.filled + s) as i32);
        // The cache holds one dtype whatever a pass computed in.
        let new_keys = as_dtype_of(new_keys, &key_buf);
        let new_values = as_dtype_of(new_values, &value_buf);
        self.keys = Some(key_buf.slice_set(&new_keys, &[0, 0, start, 0], &[b, h, stop, kd]));
        self.values = Some(value_buf.slice_set(&new_values, &[0, 0, start, 0], &[b, h, stop, vd]));
        self.filled += s;
        self.seen += s;
        Ok(self.view(window, s))
    }

    /// Make room for `s` more tokens: grow the buffer, or for a windowed
    /// layer drop what the window no longer reaches.
    fn make_room(
        &mut self,
        new_keys: &Array,
        new_values: &Array,
        s: usize,
        window: Option<Window>,
        max_seq_len: usize,
    ) {
        let capacity = self.keys.as_ref().map_or(0, |k| k.dim(2) as usize);
        if self.keys.is_some() && self.filled + s <= capacity {
            return;
        }
        let (b, h) = (new_keys.dim(0), new_keys.dim(1));
        let (kd, vd) = (new_keys.dim(3), new_values.dim(3));
        let (kdt, vdt) = (new_keys.dtype_raw(), new_values.dtype_raw());
        let (old_keys, old_values) = (self.keys.take(), self.values.take());

        let Some(w) = window else {
            let needed = self.filled + s;
            let target = needed
                .div_ceil(GROWTH_CHUNK)
                .max(1)
                .saturating_mul(GROWTH_CHUNK)
                .max(capacity.saturating_mul(2))
                .min(max_seq_len.max(needed));
            self.keys = Some(grown(old_keys, [b, h, kd], kdt, self.filled, target));
            self.values = Some(grown(old_values, [b, h, vd], vdt, self.filled, target));
            return;
        };

        let sinks = self.filled.min(w.sinks);
        let tail = self.filled - sinks;
        // The first new token's query reaches back `recent - 1` tokens.
        let kept_tail = tail.min(w.recent - 1);
        let dropped = tail - kept_tail;
        let retained = sinks + kept_tail;
        let slack = WINDOW_SLACK.min(w.size()).max(s);
        let target = if dropped == 0 {
            // Nothing to drop yet: grow like a full layer, but never past
            // what a compacted buffer would hold.
            (self.filled + s)
                .div_ceil(GROWTH_CHUNK)
                .saturating_mul(GROWTH_CHUNK)
                .max(capacity.saturating_mul(2))
                .min(retained + slack.max(w.size() + s))
        } else {
            retained + slack
        };
        let rebuild = |old: Option<Array>, dim: i32, dt: i32| -> Array {
            let Some(old) = old else {
                return Array::zeros(&[b, h, target as i32, dim], dt);
            };
            if dropped == 0 {
                return grown(Some(old), [b, h, dim], dt, self.filled, target);
            }
            let rows = |from: usize, to: usize| {
                old.slice(&[0, 0, from as i32, 0], &[b, h, to as i32, dim])
            };
            let free = Array::zeros(&[b, h, (target - retained) as i32, dim], dt);
            let mut parts: Vec<Array> = Vec::with_capacity(3);
            if sinks > 0 {
                parts.push(rows(0, sinks));
            }
            if kept_tail > 0 {
                parts.push(rows(self.filled - kept_tail, self.filled));
            }
            parts.push(free);
            let refs: Vec<&Array> = parts.iter().collect();
            ops::concatenate_axis(&refs, 2)
        };
        self.keys = Some(rebuild(old_keys, kd, kdt));
        self.values = Some(rebuild(old_values, vd, vdt));
        self.filled = retained;
        self.evicted += dropped;
    }

    /// The keys and values a call that just added `s` tokens attends to.
    fn view(&self, window: Option<Window>, s: usize) -> (Array, Array) {
        let (Some(keys), Some(values)) = (self.keys.as_ref(), self.values.as_ref()) else {
            unreachable!("view of an empty layer");
        };
        let rows = |a: &Array, from: usize, to: usize| {
            a.slice(
                &[0, 0, from as i32, 0],
                &[a.dim(0), a.dim(1), to as i32, a.dim(3)],
            )
        };
        let Some(w) = window else {
            return (rows(keys, 0, self.filled), rows(values, 0, self.filled));
        };
        let sinks = self.filled.min(w.sinks);
        let wanted = w.recent - 1 + s;
        if self.filled - sinks <= wanted {
            return (rows(keys, 0, self.filled), rows(values, 0, self.filled));
        }
        let from = self.filled - wanted;
        if sinks == 0 {
            return (
                rows(keys, from, self.filled),
                rows(values, from, self.filled),
            );
        }
        let join = |a: &Array| {
            ops::concatenate_axis(&[&rows(a, 0, sinks), &rows(a, from, self.filled)], 2)
        };
        (join(keys), join(values))
    }

    /// What is cached now, as a single-token step would see it.
    pub(super) fn current(&self, window: Option<Window>) -> Option<(Array, Array)> {
        if self.filled == 0 {
            return None;
        }
        Some(self.view(window, 1))
    }

    /// Absolute positions of the keys [`Self::update`] just returned for a
    /// call that added `s` tokens.
    pub(super) fn fetched_positions(&self, window: Option<Window>, s: usize) -> Vec<usize> {
        let Some(w) = window else {
            return (0..self.seen).collect();
        };
        let sinks = self.filled.min(w.sinks);
        let tail = (self.filled - sinks).min(w.recent - 1 + s);
        let first_tail = self.seen - tail;
        (0..sinks).chain(first_tail..self.seen).collect()
    }

    /// The additive mask `[1, 1, s, keys]` (0 where visible, -inf where not)
    /// for the keys [`Self::update`] just returned, or `None` when every
    /// returned key is visible to every query: a single token, or a layer
    /// that keeps everything and is attended causally.
    pub(super) fn window_mask(&self, window: Option<Window>, s: usize) -> Option<Array> {
        let w = window?;
        if s <= 1 {
            return None;
        }
        let keys = self.fetched_positions(Some(w), s);
        let first_query = self.seen - s;
        let mut data = Vec::with_capacity(s * keys.len());
        for q in first_query..self.seen {
            for &k in &keys {
                let visible = k <= q && (k < w.sinks || q - k < w.recent);
                data.push(if visible { 0.0f32 } else { f32::NEG_INFINITY });
            }
        }
        Some(Array::from_f32_slice(
            &data,
            &[1, 1, s as i32, keys.len() as i32],
        ))
    }

    /// The most tokens [`Self::rollback`] can drop and still serve the next
    /// token its whole window.
    pub(super) fn max_rollback(&self, window: Option<Window>) -> usize {
        match window {
            Some(w) if self.evicted > 0 => self
                .seen
                .saturating_sub(w.sinks + self.evicted + (w.recent - 1)),
            _ => self.seen,
        }
    }

    /// Drop the newest `n` tokens (at most [`Self::max_rollback`]); the
    /// slots are free for the next call to overwrite.
    pub(super) fn rollback(&mut self, n: usize, window: Option<Window>) -> usize {
        let n = n.min(self.max_rollback(window));
        self.filled -= n;
        self.seen -= n;
        if self.seen == 0 {
            self.reset();
        }
        n
    }

    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }
}

/// `a` in the dtype of `like`.
fn as_dtype_of(a: &Array, like: &Array) -> Array {
    if a.dtype_raw() == like.dtype_raw() {
        a.clone()
    } else {
        a.as_dtype(like.dtype_raw())
    }
}

/// `old` (`[B, heads, cap, dim]`, `filled` slots in use) extended with zeros
/// to `target` slots; a fresh zero buffer when there is no `old`.
fn grown(
    old: Option<Array>,
    [b, h, dim]: [i32; 3],
    dtype: i32,
    filled: usize,
    target: usize,
) -> Array {
    match old {
        None => Array::zeros(&[b, h, target as i32, dim], dtype),
        Some(old) => {
            let have = old.dim(2) as usize;
            // Only the written slots are worth keeping.
            let live = if filled < have {
                old.slice(&[0, 0, 0, 0], &[b, h, filled as i32, dim])
            } else {
                old
            };
            let extra = Array::zeros(&[b, h, (target - filled) as i32, dim], dtype);
            ops::concatenate_axis(&[&live, &extra], 2)
        }
    }
}

#[cfg(test)]
impl DenseLayer {
    /// The key buffer, for tests that watch where it lives.
    pub(super) fn key_buffer(&self) -> Option<&Array> {
        self.keys.as_ref()
    }
}
