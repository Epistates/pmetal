//! End-to-end accuracy tests against an f64 reference.
//!
//! The older tests in `tests.rs` mostly compare the fused kernels with
//! "dequantize, then SDPA" on the same cache, which cannot see an error that
//! both paths share. These tests compare the cache's attention output with
//! softmax attention computed in f64 from the *original* keys and values.

#![cfg(test)]

use super::*;
use crate::InlineArray;

/// Deterministic xorshift Gaussian stream.
pub(super) struct Gauss(u64);

impl Gauss {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn uniform(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (((x >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    }

    pub(super) fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Standard deviation of the attention logits the workload's queries produce.
const LOGIT_STD: f64 = 3.0;

/// Keys with the structure real models show: a few high-variance channels and
/// token norms that vary by a factor of a few. Values are plain Gaussian with
/// per-token scale.
pub(super) struct Workload {
    pub(super) batch: usize,
    pub(super) kv_heads: usize,
    pub(super) q_heads: usize,
    pub(super) seq: usize,
    pub(super) dim: usize,
    pub(super) keys: Vec<f32>,
    pub(super) values: Vec<f32>,
    pub(super) queries: Vec<f32>,
}

/// Which channels of the keys vary most, and where the queries look.
#[derive(Clone, Copy)]
struct Spread {
    /// Share of channels that are loud.
    loud_share: f64,
    /// A loud channel's standard deviation over a quiet one's.
    loud_scale: f64,
    /// Queries put their energy where the keys vary (the loud channels) when
    /// true, in the quiet channels when false.
    queries_follow_keys: bool,
}

impl Workload {
    pub(super) fn new(
        batch: usize,
        q_heads: usize,
        kv_heads: usize,
        seq: usize,
        dim: usize,
        seed: u64,
    ) -> Self {
        let spread = Spread {
            loud_share: 0.03,
            loud_scale: 6.0,
            queries_follow_keys: true,
        };
        Self::generate(batch, q_heads, kv_heads, seq, dim, seed, spread)
    }

    /// A quarter of the channels vary eight times as much as the rest, and
    /// the queries look almost only at the quiet ones: what tells the tokens
    /// apart is in the channels that carry the least energy, as in real
    /// heads, where the loud channels are close to constant.
    pub(super) fn lopsided(
        batch: usize,
        q_heads: usize,
        kv_heads: usize,
        seq: usize,
        dim: usize,
        seed: u64,
    ) -> Self {
        let spread = Spread {
            loud_share: 0.25,
            loud_scale: 8.0,
            queries_follow_keys: false,
        };
        Self::generate(batch, q_heads, kv_heads, seq, dim, seed, spread)
    }

    fn generate(
        batch: usize,
        q_heads: usize,
        kv_heads: usize,
        seq: usize,
        dim: usize,
        seed: u64,
        spread: Spread,
    ) -> Self {
        let mut g = Gauss::new(seed);
        let channel_scale: Vec<f64> = (0..kv_heads * dim)
            .map(|_| {
                if g.uniform() < spread.loud_share {
                    spread.loud_scale
                } else {
                    1.0
                }
            })
            .collect();
        let mut keys = Vec::with_capacity(batch * kv_heads * seq * dim);
        let mut values = Vec::with_capacity(batch * kv_heads * seq * dim);
        for b in 0..batch {
            for h in 0..kv_heads {
                let _ = b;
                for _ in 0..seq {
                    let tok = (0.3 * g.normal()).exp();
                    for c in 0..dim {
                        keys.push((tok * channel_scale[h * dim + c] * g.normal()) as f32);
                    }
                    let vtok = (0.3 * g.normal()).exp();
                    for _ in 0..dim {
                        values.push((vtok * g.normal()) as f32);
                    }
                }
            }
        }
        // Queries put their energy in the same channels as the keys, scaled
        // so the logits (after the 1/sqrt(d) scale) have a standard
        // deviation of `LOGIT_STD`: a clearly peaked softmax, as real heads
        // have, not a one-hot one.
        let mut queries = Vec::with_capacity(batch * q_heads * dim);
        let groups = q_heads / kv_heads;
        for _ in 0..batch {
            for qh in 0..q_heads {
                let h = qh / groups;
                let channels = &channel_scale[h * dim..(h + 1) * dim];
                // How much of the query's energy each channel gets.
                let weight: Vec<f64> = channels
                    .iter()
                    .map(|s| match (spread.queries_follow_keys, *s > 1.0) {
                        (true, _) => *s,
                        (false, true) => 0.05,
                        (false, false) => 1.0,
                    })
                    .collect();
                // The logits' variance is `sum (w sigma_q s)^2 / dim`.
                let energy: f64 = weight
                    .iter()
                    .zip(channels)
                    .map(|(w, s)| (w * s).powi(2))
                    .sum();
                let sigma = LOGIT_STD * (dim as f64 / energy).sqrt();
                for w in &weight {
                    queries.push((sigma * w * g.normal()) as f32);
                }
            }
        }
        Self {
            batch,
            kv_heads,
            q_heads,
            seq,
            dim,
            keys,
            values,
            queries,
        }
    }

    /// Add to every key of a head the same per-channel offset, drawn from a
    /// Gaussian of standard deviation `magnitude`: the token-shared component
    /// real keys carry. It adds the same `q . bias` to every logit of a head,
    /// so the exact attention output does not change, but the keys' norms
    /// are dominated by a constant.
    pub(super) fn with_key_bias(mut self, magnitude: f64, seed: u64) -> Self {
        let mut g = Gauss::new(seed);
        let bias: Vec<f64> = (0..self.kv_heads * self.dim)
            .map(|_| magnitude * g.normal())
            .collect();
        let per_head = self.seq * self.dim;
        for (i, k) in self.keys.iter_mut().enumerate() {
            let head = (i / per_head) % self.kv_heads;
            let channel = i % self.dim;
            *k += bias[head * self.dim + channel] as f32;
        }
        self
    }

    pub(super) fn scale(&self) -> f32 {
        1.0 / (self.dim as f32).sqrt()
    }

    /// Slice `[B, H, lo..hi, D]` of the keys or values as an array.
    pub(super) fn kv_slice(&self, values: bool, lo: usize, hi: usize) -> InlineArray {
        let src = if values { &self.values } else { &self.keys };
        let n = hi - lo;
        let mut out = Vec::with_capacity(self.batch * self.kv_heads * n * self.dim);
        for bh in 0..self.batch * self.kv_heads {
            let base = (bh * self.seq + lo) * self.dim;
            out.extend_from_slice(&src[base..base + n * self.dim]);
        }
        InlineArray::from_f32_slice(
            &out,
            &[
                self.batch as i32,
                self.kv_heads as i32,
                n as i32,
                self.dim as i32,
            ],
        )
    }

    pub(super) fn queries_array(&self) -> InlineArray {
        InlineArray::from_f32_slice(
            &self.queries,
            &[self.batch as i32, self.q_heads as i32, 1, self.dim as i32],
        )
    }

    /// Softmax attention of every query head over the first `len` tokens, in f64.
    pub(super) fn reference(&self, len: usize) -> Vec<f64> {
        self.reference_over(&self.keys, &self.values, self.seq, len)
    }

    /// As [`Self::reference`] over caller-supplied `[B, H, stride, D]` keys and
    /// values (for instance a cache's dequantized contents).
    pub(super) fn reference_over(
        &self,
        keys: &[f32],
        values: &[f32],
        stride: usize,
        len: usize,
    ) -> Vec<f64> {
        let groups = self.q_heads / self.kv_heads;
        let d = self.dim;
        let mut out = vec![0.0f64; self.batch * self.q_heads * d];
        for b in 0..self.batch {
            for qh in 0..self.q_heads {
                let kvh = qh / groups;
                let q = &self.queries[(b * self.q_heads + qh) * d..][..d];
                let base = (b * self.kv_heads + kvh) * stride * d;
                let mut logits = Vec::with_capacity(len);
                for t in 0..len {
                    let k = &keys[base + t * d..][..d];
                    let dot: f64 = q
                        .iter()
                        .zip(k)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum();
                    logits.push(dot * f64::from(self.scale()));
                }
                let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut z = 0.0;
                for l in &mut logits {
                    *l = (*l - m).exp();
                    z += *l;
                }
                let o = &mut out[(b * self.q_heads + qh) * d..][..d];
                for t in 0..len {
                    let v = &values[base + t * d..][..d];
                    let w = logits[t] / z;
                    for (dst, x) in o.iter_mut().zip(v) {
                        *dst += w * f64::from(*x);
                    }
                }
            }
        }
        out
    }
}

/// Relative L2 error `||a - b|| / ||b||`.
pub(super) fn relative_error(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - y).powi(2))
        .sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

/// Run the cache over `w`: prefill all but the last token, then decode the
/// last token with `append_and_compute_attention`. Returns the attention
/// output as f32 and fails the test if any bridge op threw.
pub(super) fn run_cache(w: &Workload, config: TurboQuantConfig) -> (QuantizedKvCache, Vec<f32>) {
    let mut cache = new_cache(config);
    let n = w.seq;
    cache
        .append(&w.kv_slice(false, 0, n - 1), &w.kv_slice(true, 0, n - 1))
        .expect("prefill append");
    let out = cache
        .append_and_compute_attention(
            &w.queries_array(),
            &w.kv_slice(false, n - 1, n),
            &w.kv_slice(true, n - 1, n),
            w.scale(),
        )
        .expect("decode step");
    let len = w.batch * w.q_heads * w.dim;
    let host = out
        .reshape(&[len as i32])
        .to_f32_vec(len)
        .expect("read output");
    crate::check_unobserved_error().expect("bridge op threw");
    (cache, host)
}

/// Host copy of an array, in row order.
pub(super) fn to_host(a: &InlineArray) -> Vec<f32> {
    let n: i32 = a.shape().iter().product();
    a.reshape(&[n]).to_f32_vec(n as usize).expect("read array")
}
