//! Shared TurboQuant attention dispatch for the bridge native paths.
//!
//! Every native architecture with a TurboQuant cache (`qwen3_native`,
//! `llama4_native`, …) needs the same decode-time decision tree once it has
//! a `QuantizedKvCache`:
//!
//! - **Decode (S=1)**: run `append_and_compute_attention`, which attends to
//!   the sinks, the packed cold store and the recent ring in one fused pass
//!   straight from the packed cache, and surface errors instead of silently
//!   switching cache semantics mid-layer.
//! - **Prefill (S>1)**: append into the cache, then run standard SDPA. When
//!   `prev == 0` (first prefill, no prior history) use the fresh keys/values
//!   directly — no dequantization needed. Otherwise dequantize the full
//!   cache and run SDPA over the concatenated history.
//!
//! Shape contract — queries/keys/values must be `[B, H, S, D]` with `H` the
//! query head count and `H_kv` the cache head count. The function does **not**
//! perform GQA expansion: callers that have `H != H_kv` must replicate keys
//! and values along the head axis before invoking the fallback path. The
//! cache's own direct-attention path handles GQA internally.
//!
//! The function does **not** update the per-layer offset counter — callers
//! own that state because the surrounding code paths (e.g. quantized affine,
//! sliding window) update it from a single point at the end of the layer.

use crate::InlineArray;
use crate::turboquant::QuantizedKvCache;

/// Run a TurboQuant attention layer step on the given cache.
///
/// Returns the layer output `[B, H, S, D_v]`. `prev` is the cache offset
/// **before** the new chunk is appended, used to pick the prefill fallback
/// path. `trace_label` is a short upper-case identifier (e.g. `"QWEN"`,
/// `"LLAMA4"`) prefixed to trace messages emitted via the
/// `PMETAL_TRACE_TURBOQUANT` env var.
pub fn turboquant_attention_step(
    tq_cache: &mut QuantizedKvCache,
    queries: &InlineArray,
    keys: &InlineArray,
    values: &InlineArray,
    scale: f32,
    prev: i32,
    trace_label: &'static str,
) -> Result<InlineArray, String> {
    let s = queries.dim(2);
    if s == 1 {
        tq_cache
            .append_and_compute_attention(queries, keys, values, scale)
            .map_err(|err| {
                trace(
                    trace_label,
                    &format!("decode_error=append_and_compute_attention prev={prev} err={err}"),
                );
                err
            })
    } else {
        tq_cache.append(keys, values).map_err(|err| {
            trace(
                trace_label,
                &format!("prefill_error=append seq={s} prev={prev} err={err}"),
            );
            err
        })?;
        if prev == 0 {
            trace(
                trace_label,
                &format!("prefill_path=dense_prompt_only seq={s}"),
            );
            crate::decode::try_sdpa_causal_like_mlx(queries, keys, values, scale, s)
                .map_err(|err| err.to_string())
        } else {
            trace(
                trace_label,
                &format!("prefill_fallback=full_dequantized seq={s} prev={prev}"),
            );
            let full_keys = tq_cache.dequantize_keys().ok_or_else(|| {
                format!(
                    "TurboQuant failed to dequantize keys for prefill fallback seq={s} prev={prev}"
                )
            })?;
            let full_values = tq_cache
                .dequantize_values()
                .ok_or_else(|| {
                    format!(
                        "TurboQuant failed to dequantize values for prefill fallback seq={s} prev={prev}"
                    )
                })?;
            crate::decode::try_sdpa_causal_like_mlx(queries, &full_keys, &full_values, scale, s)
                .map_err(|err| err.to_string())
        }
    }
}

/// True when `PMETAL_TRACE_TURBOQUANT` was set when this was first consulted.
/// The environment is read once.
#[inline]
fn trace_enabled() -> bool {
    crate::env_flag::env_flag!("PMETAL_TRACE_TURBOQUANT")
}

#[inline]
fn trace(label: &'static str, message: &str) {
    if trace_enabled() {
        eprintln!("[TURBOQUANT TRACE][{label}] {message}");
    }
}

#[cfg(test)]
mod tests {
    //! A layer's whole life through `turboquant_attention_step`: a prompt,
    //! a second chunk against the cache, then single-token decode, against
    //! f64 causal attention over the original keys and values.

    use super::*;
    use crate::turboquant::{TurboQuantConfig, new_cache};

    struct Rng(u64);

    impl Rng {
        fn normal(&mut self) -> f32 {
            let mut next = || {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                (((self.0 >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
            };
            let (u1, u2) = (next(), next());
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        }
    }

    fn host(a: &InlineArray) -> Vec<f32> {
        let n: i32 = a.shape().iter().product();
        a.reshape(&[n]).to_f32_vec(n as usize).expect("read")
    }

    /// Causal attention in f64 of `s` query tokens at positions
    /// `prev..prev + s` over the first `prev + s` of `[kv_heads, total, dim]`
    /// keys and values. Queries are `[q_heads, s, dim]`.
    struct Problem {
        q_heads: usize,
        kv_heads: usize,
        dim: usize,
        total: usize,
        scale: f32,
    }

    impl Problem {
        fn reference(
            &self,
            q: &[f32],
            keys: &[f32],
            values: &[f32],
            prev: usize,
            s: usize,
        ) -> Vec<f64> {
            let groups = self.q_heads / self.kv_heads;
            let dim = self.dim;
            let mut out = vec![0.0f64; self.q_heads * s * dim];
            for h in 0..self.q_heads {
                let kv_head = h / groups;
                for i in 0..s {
                    let visible = prev + i + 1;
                    let query = &q[(h * s + i) * dim..][..dim];
                    let mut weights: Vec<f64> = (0..visible)
                        .map(|t| {
                            let key = &keys[(kv_head * self.total + t) * dim..][..dim];
                            query
                                .iter()
                                .zip(key)
                                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                                .sum::<f64>()
                                * f64::from(self.scale)
                        })
                        .collect();
                    let max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut z = 0.0;
                    for w in &mut weights {
                        *w = (*w - max).exp();
                        z += *w;
                    }
                    for (t, w) in weights.iter().enumerate() {
                        let value = &values[(kv_head * self.total + t) * dim..][..dim];
                        for d in 0..dim {
                            out[(h * s + i) * dim + d] += w / z * f64::from(value[d]);
                        }
                    }
                }
            }
            out
        }

        /// Tokens `lo..hi` of `[heads, total, dim]` data as `[1, heads, hi - lo, dim]`.
        fn slice(&self, data: &[f32], heads: usize, lo: usize, hi: usize) -> InlineArray {
            let mut out = Vec::new();
            for h in 0..heads {
                out.extend_from_slice(
                    &data[(h * self.total + lo) * self.dim..][..(hi - lo) * self.dim],
                );
            }
            InlineArray::from_f32_slice(&out, &[1, heads as i32, (hi - lo) as i32, self.dim as i32])
        }
    }

    #[test]
    fn a_prompt_a_second_chunk_and_decode_steps_match_dense_attention() {
        let p = Problem {
            q_heads: 8,
            kv_heads: 2,
            dim: 64,
            total: 140,
            scale: 1.0 / 8.0,
        };
        let mut rng = Rng(77);
        let keys: Vec<f32> = (0..p.kv_heads * p.total * p.dim)
            .map(|_| rng.normal())
            .collect();
        let values: Vec<f32> = (0..p.kv_heads * p.total * p.dim)
            .map(|_| rng.normal())
            .collect();
        let queries: Vec<f32> = (0..p.q_heads * p.total * p.dim)
            .map(|_| 0.5 * rng.normal())
            .collect();

        // A window of 32 with 4 sinks: by the end the cache holds quantized tokens.
        let cfg = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(Some(32))
            .with_sink_tokens(4);
        let mut cache = new_cache(cfg);
        let mut prev = 0usize;
        for chunk in [60usize, 40, 1, 1, 1, 1, 1, 1, 1, 1, 32] {
            let hi = prev + chunk;
            let q = p.slice(&queries, p.q_heads, prev, hi);
            let k = p.slice(&keys, p.kv_heads, prev, hi);
            let v = p.slice(&values, p.kv_heads, prev, hi);
            let out =
                turboquant_attention_step(&mut cache, &q, &k, &v, p.scale, prev as i32, "TEST")
                    .expect("attention step");
            assert_eq!(
                out.shape(),
                &[1, p.q_heads as i32, chunk as i32, p.dim as i32]
            );
            let want = p.reference(&host(&q), &keys, &values, prev, chunk);
            let num: f64 = host(&out)
                .iter()
                .zip(&want)
                .map(|(x, y)| (f64::from(*x) - y).powi(2))
                .sum();
            let den: f64 = want.iter().map(|y| y * y).sum();
            let err = (num / den).sqrt();
            // 8-bit keys and values: within a percent or two of exact.
            assert!(err < 0.03, "tokens {prev}..{hi}: relative error {err:.4}");
            prev = hi;
        }
        assert_eq!(cache.len(), p.total);
        assert!(cache.cold_len() > 0, "the window should have been exceeded");
        crate::check_unobserved_error().expect("bridge op threw");
    }
}
