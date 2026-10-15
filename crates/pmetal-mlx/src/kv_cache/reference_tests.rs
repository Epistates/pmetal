//! Cached decoding against an uncached, full-precision reference.
//!
//! A small two-layer attention model (grouped-query attention, RoPE, a
//! residual stream, tied token embeddings) is run twice over the same tokens:
//! once whole, with no cache and an explicit mask, and once through a
//! [`KVCache`] in chunks. The logits must agree. This exercises what a shape
//! check cannot: that each layer rotates its new tokens at the right absolute
//! position, that eviction keeps the keys a query is allowed to see, that a
//! rolled-back cache decodes as if the rolled-back tokens never happened.

use super::*;
use crate::test_utils::to_f32_vec_eval;
use pmetal_runtime::compat::{Array, Dtype, ops};

pub(super) const HQ: usize = 4;
pub(super) const HKV: usize = 2;
pub(super) const DIM: usize = 32;
pub(super) const HIDDEN: usize = HQ * DIM;
pub(super) const VOCAB: usize = 61;
pub(super) const LAYERS: usize = 2;
const ROPE_BASE: f32 = 10_000.0;

/// SplitMix64: a deterministic stream independent of the global MLX key.
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal by Box-Muller.
    pub(super) fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    pub(super) fn normals(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * scale).collect()
    }
}

pub(super) fn array(data: &[f32], shape: &[usize]) -> Array {
    let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    Array::from_f32_slice(data, &shape)
}

struct ToyLayer {
    wq: Array,
    wk: Array,
    wv: Array,
    wo: Array,
}

/// Which keys a query may attend to: everything up to itself (`window` of
/// `None`), or the first `keep` tokens plus the newest ones, `window` tokens
/// in all counting the sinks.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Visibility {
    pub window: Option<usize>,
    pub keep: usize,
}

impl Visibility {
    pub(super) fn sees(self, query: usize, key: usize) -> bool {
        if key > query {
            return false;
        }
        match self.window {
            None => true,
            Some(window) => key < self.keep || query - key < window - self.keep.min(window),
        }
    }
}

pub(super) struct Toy {
    layers: Vec<ToyLayer>,
    embed: Array,
    lm_head: Array,
}

impl Toy {
    pub(super) fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let w = 1.0 / (HIDDEN as f32).sqrt();
        let layers = (0..LAYERS)
            .map(|_| ToyLayer {
                wq: array(
                    &rng.normals(HIDDEN * HQ * DIM, w * 2.0),
                    &[HIDDEN, HQ * DIM],
                ),
                wk: array(
                    &rng.normals(HIDDEN * HKV * DIM, w * 2.0),
                    &[HIDDEN, HKV * DIM],
                ),
                wv: array(&rng.normals(HIDDEN * HKV * DIM, w), &[HIDDEN, HKV * DIM]),
                wo: array(&rng.normals(HQ * DIM * HIDDEN, w), &[HQ * DIM, HIDDEN]),
            })
            .collect();
        let embed = array(&rng.normals(VOCAB * HIDDEN, 1.0), &[VOCAB, HIDDEN]);
        let lm_head = array(&rng.normals(HIDDEN * VOCAB, w), &[HIDDEN, VOCAB]);
        Self {
            layers,
            embed,
            lm_head,
        }
    }

    /// The projections of one layer for `x` `[1, S, HIDDEN]`, before RoPE:
    /// `(q [1, HQ, S, D], k [1, HKV, S, D], v [1, HKV, S, D])`.
    pub(super) fn project(&self, layer: usize, x: &Array) -> (Array, Array, Array) {
        let l = &self.layers[layer];
        let s = x.dim(1);
        let heads = |a: Array, h: usize| {
            a.reshape(&[1, s, h as i32, DIM as i32])
                .transpose_axes(&[0, 2, 1, 3])
        };
        (
            heads(x.matmul(&l.wq), HQ),
            heads(x.matmul(&l.wk), HKV),
            heads(x.matmul(&l.wv), HKV),
        )
    }

    pub(super) fn rope(&self, x: &Array, offset: i32) -> Array {
        x.rope(DIM as i32, false, ROPE_BASE, 1.0, offset)
    }

    pub(super) fn embed(&self, tokens: &[i32]) -> Array {
        let ids = Array::from_i32_slice_shaped(tokens, &[tokens.len() as i32]);
        self.embed
            .take_axis(&ids, 0)
            .reshape(&[1, tokens.len() as i32, HIDDEN as i32])
    }

    pub(super) fn out_proj(&self, layer: usize, x: &Array, attn: &Array) -> Array {
        let s = x.dim(1);
        let merged = attn
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[1, s, (HQ * DIM) as i32]);
        x.add(&merged.matmul(&self.layers[layer].wo))
    }

    pub(super) fn logits(&self, x: &Array) -> Array {
        x.matmul(&self.lm_head)
    }

    /// Whole-sequence forward with no cache, `[T, VOCAB]` logits flattened.
    pub(super) fn reference(&self, tokens: &[i32], vis: Visibility) -> Vec<f32> {
        let t = tokens.len();
        let mut x = self.embed(tokens);
        let mask = additive_mask(
            (0..t).collect::<Vec<_>>().as_slice(),
            &(0..t).collect::<Vec<_>>(),
            vis,
        );
        for layer in 0..LAYERS {
            let (q, k, v) = self.project(layer, &x);
            let (q, k) = (self.rope(&q, 0), self.rope(&k, 0));
            let attn = q.sdpa_with_mask(&k, &v, scale(), Some(&mask));
            x = self.out_proj(layer, &x, &attn);
        }
        to_f32_vec_eval(&self.logits(&x))
    }

    /// Feed `tokens` through `cache` in `chunks` of the given lengths and
    /// return the logits of every token, concatenated.
    ///
    /// `positions` says which absolute positions the keys the cache returned
    /// for a layer hold, given how many tokens it had seen and how many keys
    /// it returned.
    pub(super) fn cached(
        &self,
        cache: &mut KVCache,
        tokens: &[i32],
        chunks: &[usize],
        vis: Visibility,
    ) -> Vec<f32> {
        assert_eq!(chunks.iter().sum::<usize>(), tokens.len());
        let mut out = Vec::new();
        let mut start = 0;
        for &len in chunks {
            out.extend(self.cached_chunk(cache, &tokens[start..start + len], vis));
            start += len;
        }
        out
    }

    /// One chunk of tokens through `cache`; the logits of its tokens.
    pub(super) fn cached_chunk(
        &self,
        cache: &mut KVCache,
        chunk: &[i32],
        vis: Visibility,
    ) -> Vec<f32> {
        let mut x = self.embed(chunk);
        let len = chunk.len();
        for layer in 0..LAYERS {
            let offset = cache.rope_offset_for(layer);
            let (q, k, v) = self.project(layer, &x);
            let (q, k) = (self.rope(&q, offset), self.rope(&k, offset));
            let (all_k, all_v) = cache.update_and_fetch(layer, &k, &v).unwrap();
            let fetched = all_k.dim(2) as usize;
            let seen = offset as usize + len;
            let key_positions = fetched_positions(seen, fetched, vis);
            let query_positions: Vec<usize> = (offset as usize..seen).collect();
            let mask = additive_mask(&query_positions, &key_positions, vis);
            let attn = q.sdpa_with_mask(&all_k, &all_v, scale(), Some(&mask));
            x = self.out_proj(layer, &x, &attn);
        }
        to_f32_vec_eval(&self.logits(&x))
    }
}

pub(super) fn scale() -> f32 {
    1.0 / (DIM as f32).sqrt()
}

/// The absolute positions of the `fetched` keys a cache returned after
/// `seen` tokens: all of them while nothing was evicted, otherwise the first
/// `keep` and the newest.
pub(super) fn fetched_positions(seen: usize, fetched: usize, vis: Visibility) -> Vec<usize> {
    if fetched >= seen {
        return (0..seen).collect();
    }
    let sinks = vis.keep.min(fetched);
    (0..sinks).chain(seen - (fetched - sinks)..seen).collect()
}

/// Additive mask `[1, 1, Q, K]`: 0 where the query sees the key.
pub(super) fn additive_mask(queries: &[usize], keys: &[usize], vis: Visibility) -> Array {
    let mut data = Vec::with_capacity(queries.len() * keys.len());
    for &q in queries {
        for &k in keys {
            data.push(if vis.sees(q, k) { 0.0f32 } else { -1e30 });
        }
    }
    Array::from_f32_slice(&data, &[1, 1, queries.len() as i32, keys.len() as i32])
}

pub(super) fn tokens(n: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| (rng.next_u64() % VOCAB as u64) as i32)
        .collect()
}

pub(super) fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit count");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

pub(super) fn cache_config() -> KVCacheConfig {
    KVCacheConfig::new(LAYERS, 512, HKV, DIM)
}

fn rows(logits: &[f32], from: usize, to: usize) -> &[f32] {
    &logits[from * VOCAB..to * VOCAB]
}

// ---------------------------------------------------------------------------
// Standard cache
// ---------------------------------------------------------------------------

#[test]
fn standard_cache_chunked_prefill_and_decode_match_uncached() {
    let toy = Toy::new(1);
    let toks = tokens(70, 11);
    let reference = toy.reference(&toks, Visibility::default());

    // A prefill in uneven chunks, then token-by-token decode.
    let mut chunks = vec![17, 1, 30, 7];
    chunks.extend(std::iter::repeat_n(1, 15));
    let mut cache = KVCache::new(cache_config());
    let cached = toy.cached(&mut cache, &toks, &chunks, Visibility::default());

    let diff = max_diff(&cached, &reference);
    assert!(diff < 2e-4, "chunked cache vs uncached: max diff {diff}");
    assert_eq!(cache.seq_len(), 70);
    for layer in 0..LAYERS {
        assert_eq!(cache.rope_offset_for(layer), 70);
    }
}

#[test]
fn standard_cache_grows_past_one_allocation_step() {
    let toy = Toy::new(2);
    let toks = tokens(300, 12);
    let reference = toy.reference(&toks, Visibility::default());
    let mut chunks = vec![120];
    chunks.extend(std::iter::repeat_n(1, 180));
    let mut cache = KVCache::new(cache_config());
    let cached = toy.cached(&mut cache, &toks, &chunks, Visibility::default());
    let diff = max_diff(&cached, &reference);
    assert!(diff < 5e-4, "growth across steps: max diff {diff}");
}

#[test]
fn rollback_then_regenerate_matches_uncached_of_the_accepted_sequence() {
    let toy = Toy::new(3);
    let accepted = tokens(40, 13);
    let drafted = tokens(6, 14);
    let alternative = tokens(5, 15);

    // Cache: the prompt, a verified draft of 6 tokens of which 2 are kept,
    // then the continuation the target actually produced.
    let mut cache = KVCache::new(cache_config());
    toy.cached_chunk(&mut cache, &accepted, Visibility::default());
    toy.cached_chunk(&mut cache, &drafted, Visibility::default());
    assert_eq!(cache.rollback(4), 4);
    let after = toy.cached_chunk(&mut cache, &alternative, Visibility::default());

    let mut full = accepted.clone();
    full.extend(&drafted[..2]);
    full.extend(&alternative);
    let reference = toy.reference(&full, Visibility::default());
    let tail = rows(&reference, 42, 47);
    let diff = max_diff(&after, tail);
    assert!(diff < 2e-4, "after rollback: max diff {diff}");
    for layer in 0..LAYERS {
        assert_eq!(cache.rope_offset_for(layer), 47);
    }
}

// ---------------------------------------------------------------------------
// Sliding window and attention sinks
// ---------------------------------------------------------------------------

fn windowed(window: usize, keep: usize) -> (KVCacheConfig, Visibility) {
    let config = if keep == 0 {
        cache_config().with_sliding_window(window)
    } else {
        cache_config().with_rotating(window, keep)
    };
    (
        config,
        Visibility {
            window: Some(window),
            keep,
        },
    )
}

#[test]
fn sliding_window_decode_matches_windowed_reference() {
    let toy = Toy::new(4);
    let toks = tokens(90, 16);
    let (config, vis) = windowed(16, 0);
    let reference = toy.reference(&toks, vis);
    let mut chunks = vec![10];
    chunks.extend(std::iter::repeat_n(1, 80));
    let mut cache = KVCache::new(config);
    let cached = toy.cached(&mut cache, &toks, &chunks, vis);
    let diff = max_diff(&cached, &reference);
    assert!(diff < 2e-4, "sliding decode: max diff {diff}");
    for layer in 0..LAYERS {
        assert_eq!(cache.rope_offset_for(layer), 90);
    }
}

#[test]
fn sliding_window_chunked_prefill_matches_windowed_reference() {
    // Chunks longer than the window: every query still sees its own window.
    let toy = Toy::new(5);
    let toks = tokens(100, 17);
    let (config, vis) = windowed(12, 0);
    let reference = toy.reference(&toks, vis);
    let mut cache = KVCache::new(config);
    let cached = toy.cached(&mut cache, &toks, &[30, 5, 20, 1, 1, 24, 19], vis);
    let diff = max_diff(&cached, &reference);
    assert!(diff < 2e-4, "sliding chunked prefill: max diff {diff}");
}

#[test]
fn rotating_cache_keeps_attention_sinks() {
    let toy = Toy::new(6);
    let toks = tokens(120, 18);
    let (config, vis) = windowed(20, 4);
    let reference = toy.reference(&toks, vis);
    let mut chunks = vec![9, 8];
    chunks.extend(std::iter::repeat_n(1, 103));
    let mut cache = KVCache::new(config);
    let cached = toy.cached(&mut cache, &toks, &chunks, vis);
    let diff = max_diff(&cached, &reference);
    assert!(diff < 2e-4, "sinks + window decode: max diff {diff}");
}

#[test]
fn rotating_cache_chunked_prefill_with_sinks_matches_reference() {
    let toy = Toy::new(7);
    let toks = tokens(100, 19);
    let (config, vis) = windowed(24, 4);
    let reference = toy.reference(&toks, vis);
    let mut cache = KVCache::new(config);
    let cached = toy.cached(&mut cache, &toks, &[40, 8, 1, 1, 30, 20], vis);
    let diff = max_diff(&cached, &reference);
    assert!(
        diff < 2e-4,
        "sinks + window chunked prefill: max diff {diff}"
    );
}

#[test]
fn windowed_rollback_gives_back_only_what_the_window_still_covers() {
    let toy = Toy::new(10);
    let (config, vis) = windowed(24, 4);
    let accepted = tokens(400, 24);
    let alternative = tokens(6, 25);

    let mut cache = KVCache::new(config);
    let mut chunks = vec![30];
    chunks.extend(std::iter::repeat_n(1, 370));
    toy.cached(&mut cache, &accepted, &chunks, vis);

    // Far more than the window: the cache gives back what it can.
    let removed = cache.rollback(300);
    assert!(removed > 0 && removed < 300, "removed {removed}");
    assert_eq!(cache.rope_offset_for(0) as usize, 400 - removed);
    for layer in 1..LAYERS {
        assert_eq!(cache.rope_offset_for(layer) as usize, 400 - removed);
    }

    // What follows decodes as if the removed tokens never happened.
    let kept = 400 - removed;
    let after = toy.cached_chunk(&mut cache, &alternative, vis);
    let mut full = accepted[..kept].to_vec();
    full.extend(&alternative);
    let reference = toy.reference(&full, vis);
    let diff = max_diff(&after, rows(&reference, kept, kept + 6));
    assert!(diff < 2e-4, "after a windowed rollback: max diff {diff}");
}

#[test]
fn window_mask_hides_what_an_early_query_must_not_see() {
    let (config, vis) = windowed(6, 2);
    let mut cache = KVCache::new(config);
    let mut rng = Rng::new(7);
    let chunk =
        |n: usize, rng: &mut Rng| array(&rng.normals(HKV * n * DIM, 1.0), &[1, HKV, n, DIM]);
    let first = chunk(20, &mut rng);
    cache.update_and_fetch(0, &first, &first).unwrap();
    let second = chunk(5, &mut rng);
    let (keys, _) = cache.update_and_fetch(0, &second, &second).unwrap();
    let mask = cache.window_mask(0, 5).expect("a chunk needs a mask");
    assert_eq!(mask.shape(), &[1, 1, 5, keys.dim(2)]);

    let positions = fetched_positions(25, keys.dim(2) as usize, vis);
    let expected = additive_mask(&(20..25).collect::<Vec<_>>(), &positions, vis);
    let visible =
        |m: &Array| -> Vec<bool> { to_f32_vec_eval(m).into_iter().map(|x| x == 0.0).collect() };
    assert_eq!(visible(&mask), visible(&expected));
    // A single token sees everything it is given; so does a plain cache.
    assert!(cache.window_mask(0, 1).is_none());
    assert!(KVCache::new(cache_config()).window_mask(0, 5).is_none());
}

#[test]
fn windowed_cache_memory_stays_bounded() {
    let toy = Toy::new(8);
    let toks = tokens(900, 20);
    let (config, vis) = windowed(32, 4);
    let mut cache = KVCache::new(config);
    let mut chunks = vec![64];
    chunks.extend(std::iter::repeat_n(1, 836));
    toy.cached(&mut cache, &toks, &chunks, vis);
    // 2 layers x (K + V) x HKV x DIM x f32 per token.
    let per_token = LAYERS * 2 * HKV * DIM * 4;
    let held = cache.memory_usage() / per_token;
    assert!(
        held <= 32 + 256 + 64,
        "a 32-token window holds {held} token slots after 900 tokens"
    );
}

// ---------------------------------------------------------------------------
// Prefix reuse
// ---------------------------------------------------------------------------

#[test]
fn forked_prefix_continues_with_the_logits_of_an_uncached_run() {
    let toy = Toy::new(9);
    let prefix = tokens(48, 21);
    let suffix_a = tokens(13, 22);
    let suffix_b = tokens(9, 23);

    let mut base = KVCache::new(cache_config());
    toy.cached_chunk(&mut base, &prefix, Visibility::default());

    let mut fork_a = base.fork();
    let mut fork_b = base.fork();
    let a = toy.cached_chunk(&mut fork_a, &suffix_a, Visibility::default());
    let b = toy.cached_chunk(&mut fork_b, &suffix_b, Visibility::default());

    let mut full_a = prefix.clone();
    full_a.extend(&suffix_a);
    let mut full_b = prefix.clone();
    full_b.extend(&suffix_b);
    let ref_a = toy.reference(&full_a, Visibility::default());
    let ref_b = toy.reference(&full_b, Visibility::default());
    assert!(max_diff(&a, rows(&ref_a, 48, 61)) < 2e-4);
    assert!(max_diff(&b, rows(&ref_b, 48, 57)) < 2e-4);
    // The forks did not disturb the cache they came from.
    assert_eq!(base.seq_len(), 48);
}

#[test]
fn memory_accounting_follows_the_stored_dtype() {
    // The config says f32 (its default); the model feeds bf16. What the cache
    // reports is what it holds.
    let held = |dtype: Dtype| {
        let mut cache = KVCache::new(cache_config().with_dtype(Dtype::Float32));
        let k = ops::zeros(&[1, HKV as i32, 10, DIM as i32], dtype);
        cache.update_and_fetch(0, &k, &k).unwrap();
        cache.memory_usage()
    };
    let (bf16, f32) = (held(Dtype::Bfloat16), held(Dtype::Float32));
    assert!(bf16 > 0);
    assert_eq!(bf16 * 2, f32, "bf16 data counted as {bf16} vs f32 {f32}");
}

// ---------------------------------------------------------------------------
// Quantized cache
// ---------------------------------------------------------------------------

fn quantized_config(bits: u8) -> KVCacheConfig {
    cache_config().with_quantized(bits, 32)
}

/// The largest logit difference of a quantized cache from the uncached run,
/// over the tokens after the prompt.
fn quantized_decode_error(bits: u8) -> (f32, f32) {
    let toy = Toy::new(30);
    let toks = tokens(80, 31);
    let reference = toy.reference(&toks, Visibility::default());
    let mut chunks = vec![50];
    chunks.extend(std::iter::repeat_n(1, 30));
    let mut cache = KVCache::new(quantized_config(bits));
    let cached = toy.cached(&mut cache, &toks, &chunks, Visibility::default());
    (
        max_diff(&cached[..50 * VOCAB], &reference[..50 * VOCAB]),
        max_diff(&cached[50 * VOCAB..], &reference[50 * VOCAB..]),
    )
}

#[test]
fn quantized_prefill_attends_to_exact_keys() {
    // The prompt's own attention must not see its quantization error.
    for bits in [2u8, 4, 8] {
        let (prompt, _) = quantized_decode_error(bits);
        assert!(prompt < 2e-4, "{bits}-bit prompt logits differ by {prompt}");
    }
}

#[test]
fn quantized_decode_error_follows_the_bit_width() {
    let errors: Vec<(u8, f32)> = [8u8, 6, 5, 4, 3, 2]
        .into_iter()
        .map(|bits| (bits, quantized_decode_error(bits).1))
        .collect();
    // The model's random keys give logits a spread of 1.6; the worst of 30
    // decode steps over the whole vocabulary is off by this much at each
    // width (measured: 0.11, 0.50, 1.0, 1.8, 3.0, 4.6), with margin.
    for (bits, bound) in [
        (8u8, 0.16f32),
        (6, 0.75),
        (5, 1.5),
        (4, 2.7),
        (3, 4.5),
        (2, 7.0),
    ] {
        let got = errors.iter().find(|(b, _)| *b == bits).unwrap().1;
        assert!(
            got < bound,
            "{bits}-bit decode logits off by {got} (bound {bound})"
        );
    }
    assert!(
        errors.first().unwrap().1 < errors.last().unwrap().1,
        "8-bit should beat 2-bit: {errors:?}"
    );
}

#[test]
fn quantized_cache_elements_are_within_half_a_step() {
    for bits in [2u8, 3, 4, 8] {
        let group = 32usize;
        let mut rng = Rng::new(40 + u64::from(bits));
        let data = rng.normals(HKV * 12 * DIM, 2.0);
        let keys = array(&data, &[1, HKV, 12, DIM]);
        let mut cache = KVCache::new(quantized_config(bits));
        cache.update_and_fetch(0, &keys, &keys).unwrap();
        // The next call hands the first tokens back dequantized.
        let next = array(&rng.normals(HKV * DIM, 1.0), &[1, HKV, 1, DIM]);
        let (back, _) = cache.update_and_fetch(0, &next, &next).unwrap();
        let back = to_f32_vec_eval(&back);

        let levels = ((1u32 << bits) - 1) as f32;
        for head in 0..HKV {
            for token in 0..12 {
                let row = |buf: &[f32], stride: usize| {
                    let at = (head * stride + token) * DIM;
                    buf[at..at + DIM].to_vec()
                };
                let (original, restored) = (row(&data, 12), row(&back, 13));
                for g in 0..DIM / group {
                    let span = &original[g * group..(g + 1) * group];
                    let (lo, hi) = span
                        .iter()
                        .fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
                    // Half a step for a plain min-max map; MLX aligns the
                    // group's larger edge to a level, which moves the levels
                    // by up to half a step and clips what falls past the
                    // last one: within one step in all.
                    let step = (hi - lo) / levels;
                    for i in 0..group {
                        let err = (span[i] - restored[g * group + i]).abs();
                        assert!(
                            err <= step + 1e-5,
                            "{bits}-bit element off by {err}, step {step}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn quantized_decode_attention_matches_attention_over_the_dequantized_cache() {
    use crate::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa};
    for (key_bits, value_bits) in [(8u8, 8u8), (4, 4), (8, 4), (4, 2)] {
        for window in [None, Some(9)] {
            let config = cache_config().with_asymmetric_quantized(key_bits, value_bits, 32);
            let mut fast = KVCache::new(config.clone());
            let mut dense = KVCache::new(config);
            let mut rng = Rng::new(50);
            let prompt_k = array(&rng.normals(HKV * 30 * DIM, 1.0), &[1, HKV, 30, DIM]);
            let prompt_v = array(&rng.normals(HKV * 30 * DIM, 1.0), &[1, HKV, 30, DIM]);
            fast.update_and_fetch(0, &prompt_k, &prompt_v).unwrap();
            dense.update_and_fetch(0, &prompt_k, &prompt_v).unwrap();

            let queries = array(&rng.normals(HQ * DIM, 1.0), &[1, HQ, 1, DIM]);
            let (k, v) = one_token(60);
            let mask_type = window.map_or(AttentionMaskType::Causal, |w| {
                AttentionMaskType::SlidingWindow(w)
            });
            let attn = FusedAttentionConfig::new(HQ as i32, HKV as i32, DIM as i32)
                .with_scale(scale())
                .with_mask_type(mask_type);

            let out = fast
                .try_compressed_attention(0, &queries, &k, &v, &attn)
                .unwrap()
                .expect("a quantized cache attends to its packed rows");
            // The same cache read the long way: dequantize everything it
            // holds (the new token included), then attend.
            dense.update_and_fetch(0, &k, &v).unwrap();
            let (all_k, all_v) = dense.dequantized(0).expect("layer 0 is quantized");
            let expected = fused_sdpa(&queries, &all_k, &all_v, &attn, None).unwrap();
            let diff = crate::test_utils::max_abs_diff(&out, &expected);
            assert!(
                diff < 2e-3,
                "K{key_bits}/V{value_bits} window {window:?}: packed vs dequantized attention differ by {diff}"
            );
            assert_eq!(fast.rope_offset_for(0), 31);
        }
    }
}

#[test]
fn quantized_attention_declines_what_it_cannot_serve() {
    use crate::kernels::FusedAttentionConfig;
    let mut cache = KVCache::new(quantized_config(4));
    let mut rng = Rng::new(51);
    let prompt = array(&rng.normals(HKV * 8 * DIM, 1.0), &[1, HKV, 8, DIM]);
    cache.update_and_fetch(0, &prompt, &prompt).unwrap();
    let attn = FusedAttentionConfig::new(HQ as i32, HKV as i32, DIM as i32);

    // Several queries, or a softcap, leave the cache untouched.
    let queries = array(&rng.normals(HQ * 3 * DIM, 1.0), &[1, HQ, 3, DIM]);
    let chunk = array(&rng.normals(HKV * 3 * DIM, 1.0), &[1, HKV, 3, DIM]);
    assert!(
        cache
            .try_compressed_attention(0, &queries, &chunk, &chunk, &attn)
            .unwrap()
            .is_none()
    );
    let (k, v) = one_token(61);
    let q1 = array(&rng.normals(HQ * DIM, 1.0), &[1, HQ, 1, DIM]);
    let capped = attn.clone().with_logit_softcapping(30.0);
    assert!(
        cache
            .try_compressed_attention(0, &q1, &k, &v, &capped)
            .unwrap()
            .is_none()
    );
    assert_eq!(cache.rope_offset_for(0), 8);
}

#[test]
fn quantized_rollback_matches_the_accepted_sequence() {
    let toy = Toy::new(32);
    let accepted = tokens(40, 33);
    let drafted = tokens(6, 34);
    let alternative = tokens(4, 35);
    let mut cache = KVCache::new(quantized_config(8));
    toy.cached_chunk(&mut cache, &accepted, Visibility::default());
    toy.cached_chunk(&mut cache, &drafted, Visibility::default());
    assert_eq!(cache.rollback(4), 4);
    assert_eq!(cache.rope_offset_for(0), 42);
    let after = toy.cached_chunk(&mut cache, &alternative, Visibility::default());

    let mut full = accepted.clone();
    full.extend(&drafted[..2]);
    full.extend(&alternative);
    let reference = toy.reference(&full, Visibility::default());
    let diff = max_diff(&after, rows(&reference, 42, 46));
    assert!(diff < 0.08, "8-bit cache after rollback: max diff {diff}");
}

#[test]
fn quantized_cache_is_a_fraction_of_the_dense_one() {
    let held = |mode: Option<u8>| {
        let config = match mode {
            Some(bits) => quantized_config(bits),
            None => cache_config(),
        };
        let mut cache = KVCache::new(config);
        let mut rng = Rng::new(52);
        let k = array(&rng.normals(HKV * 300 * DIM, 1.0), &[1, HKV, 300, DIM]);
        let k = k.as_dtype(Dtype::Bfloat16.as_i32());
        cache.update_and_fetch(0, &k, &k).unwrap();
        cache.memory_usage()
    };
    let dense = held(None);
    assert!(
        held(Some(4)) * 3 < dense,
        "4-bit: {} of {dense}",
        held(Some(4))
    );
    assert!(
        held(Some(8)) * 5 < dense * 3,
        "8-bit: {} of {dense}",
        held(Some(8))
    );
}

#[test]
fn memory_footprint_matches_the_bytes_a_full_cache_holds() {
    // A cache filled to max_seq_len in one call holds exactly its footprint.
    let modes: [(&str, Option<(u8, u8)>); 6] = [
        ("dense", None),
        ("q8", Some((8, 8))),
        ("q6", Some((6, 6))),
        ("q4", Some((4, 4))),
        ("q3", Some((3, 3))),
        ("k8v4", Some((8, 4))),
    ];
    for dtype in [Dtype::Bfloat16, Dtype::Float32] {
        for (name, bits) in modes {
            let base = KVCacheConfig::new(1, 256, HKV, DIM).with_dtype(dtype);
            let config = match bits {
                None => base,
                Some((k, v)) => base.with_asymmetric_quantized(k, v, 32),
            };
            let footprint = config.memory_footprint();
            let mut cache = KVCache::new(config);
            let data = ops::zeros(&[1, HKV as i32, 256, DIM as i32], dtype);
            cache.update_and_fetch(0, &data, &data).unwrap();
            assert_eq!(
                cache.memory_usage(),
                footprint,
                "{name} {dtype:?}: the cache holds what the config says"
            );
        }
    }
}

#[test]
fn windowed_footprint_is_the_window_not_the_context() {
    let dense = KVCacheConfig::new(2, 8192, HKV, DIM).with_dtype(Dtype::Bfloat16);
    let windowed = dense.clone().with_rotating(512, 4);
    assert_eq!(windowed.memory_footprint() * 16, dense.memory_footprint());
}

// ---------------------------------------------------------------------------
// Cost: writes land in the existing buffer
// ---------------------------------------------------------------------------

/// Where layer 0's key buffer lives, after everything pending has run.
fn key_buffer(cache: &KVCache) -> *const std::ffi::c_void {
    cache
        .layer_key_buffer(0)
        .expect("layer 0 holds keys")
        .data_ptr()
}

fn one_token(seed: u64) -> (Array, Array) {
    let mut rng = Rng::new(seed);
    let shape = [1, HKV, 1, DIM];
    (
        array(&rng.normals(HKV * DIM, 1.0), &shape),
        array(&rng.normals(HKV * DIM, 1.0), &shape),
    )
}

/// Decode `steps` single tokens, evaluating each step as a model would, and
/// return the buffer's address before the first and after each.
fn decode_addresses(cache: &mut KVCache, steps: usize) -> Vec<*const std::ffi::c_void> {
    let mut addresses = vec![key_buffer(cache)];
    for step in 0..steps {
        let (k, v) = one_token(1000 + step as u64);
        let (all_k, all_v) = cache.update_and_fetch(0, &k, &v).unwrap();
        all_k.eval();
        all_v.eval();
        drop((all_k, all_v));
        addresses.push(key_buffer(cache));
    }
    addresses
}

#[test]
fn decode_appends_into_the_existing_buffer() {
    // Writing into an input something else still holds copies the whole
    // buffer; a decode step must not.
    let mut cache = KVCache::new(cache_config());
    let mut rng = Rng::new(5);
    let prompt = array(&rng.normals(HKV * 40 * DIM, 1.0), &[1, HKV, 40, DIM]);
    cache.update_and_fetch(0, &prompt, &prompt).unwrap();
    let addresses = decode_addresses(&mut cache, 24);
    // The prompt already allocated room: a step writes into the buffer. MLX
    // hands a buffer over only once the previous step's command buffer has
    // released its references, which on a loaded machine is occasionally
    // later than the next write, so a stray step may copy; a cache that
    // copies at every step moves at all of them.
    let moves = addresses
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count();
    assert!(
        moves <= 6,
        "{moves} of 24 decode steps moved the cache buffer"
    );
}

#[test]
fn windowed_decode_moves_the_buffer_only_to_compact() {
    let (config, _) = windowed(32, 4);
    let mut cache = KVCache::new(config);
    let mut rng = Rng::new(6);
    let prompt = array(&rng.normals(HKV * 20 * DIM, 1.0), &[1, HKV, 20, DIM]);
    cache.update_and_fetch(0, &prompt, &prompt).unwrap();
    let steps = 700;
    let addresses = decode_addresses(&mut cache, steps);
    let moves = addresses
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count();
    // One compaction per window's worth of tokens (the slack of a window
    // under 256), not one per token.
    assert!(moves <= steps / 16, "{moves} moves in {steps} steps");
}
