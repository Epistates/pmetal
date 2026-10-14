//! End-to-end accuracy tests against an f64 reference.
//!
//! The older tests in `tests.rs` mostly compare the fused kernels with
//! "dequantize, then SDPA" on the same cache, which cannot see an error that
//! both paths share. These tests compare the cache's attention output with
//! softmax attention computed in f64 from the *original* keys and values.

#![cfg(test)]

use super::*;

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

impl Workload {
    pub(super) fn new(
        batch: usize,
        q_heads: usize,
        kv_heads: usize,
        seq: usize,
        dim: usize,
        seed: u64,
    ) -> Self {
        let mut g = Gauss::new(seed);
        let channel_scale: Vec<f64> = (0..kv_heads * dim)
            .map(|_| if g.uniform() < 0.03 { 6.0 } else { 1.0 })
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
        // Queries share the key channel scale and are large enough for a
        // peaked softmax (logit std about 2.5 after the 1/sqrt(d) scale).
        let mut queries = Vec::with_capacity(batch * q_heads * dim);
        let groups = q_heads / kv_heads;
        for _ in 0..batch {
            for qh in 0..q_heads {
                let h = qh / groups;
                for c in 0..dim {
                    queries.push((2.5 * channel_scale[h * dim + c] * g.normal()) as f32);
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

/// Per-token relative reconstruction error summary: (mean, p99, max).
pub(super) fn token_errors(orig: &[f32], got: &[f32], dim: usize) -> (f64, f64, f64) {
    let mut errs: Vec<f64> = orig
        .chunks(dim)
        .zip(got.chunks(dim))
        .map(|(a, b)| {
            let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
            let den: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum();
            (num / den.max(1e-30)).sqrt()
        })
        .collect();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = errs.iter().sum::<f64>() / errs.len() as f64;
    (mean, errs[errs.len() * 99 / 100], *errs.last().unwrap())
}

#[test]
#[ignore = "exploration table"]
fn explore_reconstruction_error() {
    for &dim in &[64usize, 128, 256] {
        for &(kb, vb) in &[(8u8, 8u8), (4, 4), (3, 3)] {
            for qjl in [TurboQuantQjlMode::Standard, TurboQuantQjlMode::NoQjl] {
                let w = Workload::new(1, 4, 4, 1500, dim, 11 + dim as u64);
                let cfg = TurboQuantConfig::uniform(kb, vb)
                    .with_recent_window(None)
                    .with_qjl_mode(qjl);
                let mut cache = new_cache(cfg);
                cache
                    .append(&w.kv_slice(false, 0, 1500), &w.kv_slice(true, 0, 1500))
                    .unwrap();
                let dk = to_host(&cache.dequantize_keys().unwrap());
                let dv = to_host(&cache.dequantize_values().unwrap());
                let (km, kp, kx) = token_errors(&w.keys, &dk, dim);
                let (vm, vp, vx) = token_errors(&w.values, &dv, dim);
                eprintln!(
                    "d={dim:3} k{kb}v{vb} {qjl:?}: K mean/p99/max={km:.4}/{kp:.4}/{kx:.4}  V mean/p99/max={vm:.4}/{vp:.4}/{vx:.4}"
                );
            }
        }
    }
    crate::check_unobserved_error().expect("bridge op threw");
}

#[test]
#[ignore = "exploration table"]
fn explore_attention_error_table() {
    let mut rows = Vec::new();
    for &dim in &[64usize, 96, 128, 192, 256] {
        for &(kb, vb) in &[(8u8, 8u8), (4, 4), (3, 3)] {
            for qjl in [TurboQuantQjlMode::Standard, TurboQuantQjlMode::NoQjl] {
                for &(qh, kvh) in &[(4usize, 4usize), (8, 2)] {
                    for &seq in &[1500usize, 1537] {
                        let w = Workload::new(1, qh, kvh, seq, dim, 7 + dim as u64);
                        let cfg = TurboQuantConfig::uniform(kb, vb)
                            .with_recent_window(None)
                            .with_qjl_mode(qjl);
                        let (cache, got) = run_cache(&w, cfg);
                        let reference = w.reference(seq);
                        let err = relative_error(&got, &reference);
                        let dk = to_host(&cache.dequantize_keys().unwrap());
                        let dv = to_host(&cache.dequantize_values().unwrap());
                        let deq = w.reference_over(&dk, &dv, seq, seq);
                        let deq32: Vec<f32> = deq.iter().map(|x| *x as f32).collect();
                        let quant_err = relative_error(&deq32, &reference);
                        let kernel_err = relative_error(&got, &deq);
                        rows.push(format!(
                            "d={dim:3} k{kb}v{vb} {qjl:?} q{qh}/kv{kvh} T={seq}: fused_vs_ref={err:.4} dequant_vs_ref={quant_err:.4} fused_vs_dequant={kernel_err:.4}"
                        ));
                    }
                }
            }
        }
    }
    for r in rows {
        eprintln!("{r}");
    }
}
