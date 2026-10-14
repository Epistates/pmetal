//! Decode-attention timings. Ignored by default: run them with `--ignored
//! --nocapture` on an otherwise idle machine.

#![cfg(test)]

use std::time::Instant;

use super::attention::{DecodeShape, Visibility, decode_attention};
use super::quantizer::{Quantizer, words_per_row};
use super::reference_tests::Gauss;
use super::store::{ColdStore, DenseStore};
use crate::InlineArray;
use crate::compat::Dtype;

fn synthetic_cold(kv: usize, n: usize, keys: &Quantizer, values: &Quantizer) -> ColdStore {
    // Random words: the kernels read any bit pattern as valid indices.
    let mut g = Gauss::new(3);
    let mut store = ColdStore::empty(kv, keys, values);
    let words_k = words_per_row(keys.dim(), keys.bits());
    let words_v = words_per_row(values.dim(), values.bits());
    let chunk = 8192usize.min(n);
    let mut done = 0;
    while done < n {
        let step = chunk.min(n - done);
        let rand_words = |len: usize, g: &mut Gauss| -> Vec<u32> {
            (0..len)
                .map(|_| (g.normal().to_bits() >> 3) as u32 ^ (g.normal().to_bits() as u32))
                .collect()
        };
        let enc = |words: usize, g: &mut Gauss| super::quantizer::Encoded {
            packed: InlineArray::from_u32_slice(
                &rand_words(kv * step * words, g),
                &[(kv * step) as i32, words as i32],
            ),
            scales: InlineArray::from_f32_slice(&vec![1.0f32; kv * step], &[(kv * step) as i32]),
        };
        let k = enc(words_k, &mut g);
        let v = enc(words_v, &mut g);
        store.append(keys, values, &k, &v, step);
        done += step;
    }
    store
}

fn time_ms(iterations: usize, mut f: impl FnMut() -> InlineArray) -> f64 {
    for _ in 0..5 {
        f().eval();
    }
    crate::inline_array::synchronize();
    let start = Instant::now();
    for _ in 0..iterations {
        f().eval();
    }
    crate::inline_array::synchronize();
    start.elapsed().as_secs_f64() * 1e3 / iterations as f64
}

/// Milliseconds of each of the first decode steps through the cache, after a
/// prefill of `prefill` tokens.
#[test]
#[ignore = "benchmark"]
fn bench_cache_decode_steps() {
    let (heads, dim, prefill) = (8usize, 128usize, 2048usize);
    let w = super::reference_tests::Workload::new(1, heads, heads, prefill + 12, dim, 1);
    for (label, bits, window) in [
        ("4-bit, every token quantized", 4u8, None),
        ("4-bit, 1024-token window", 4, Some(1024usize)),
        ("8-bit, every token quantized", 8, None),
    ] {
        let cfg = super::TurboQuantConfig::uniform(bits, bits).with_recent_window(window);
        let mut cache = super::new_cache(cfg);
        cache
            .append(
                &w.kv_slice(false, 0, prefill),
                &w.kv_slice(true, 0, prefill),
            )
            .unwrap();
        let mut times = Vec::new();
        for t in prefill..prefill + 12 {
            let start = Instant::now();
            let out = cache
                .append_and_compute_attention(
                    &w.queries_array(),
                    &w.kv_slice(false, t, t + 1),
                    &w.kv_slice(true, t, t + 1),
                    w.scale(),
                )
                .unwrap();
            out.eval();
            times.push((start.elapsed().as_secs_f64() * 1e4).round() / 10.0);
        }
        eprintln!("STEPS {label}: ms per step {times:?}");
    }
}

#[test]
#[ignore = "benchmark"]
fn bench_decode_attention() {
    let (batch, q_heads, kv_heads, dim) = (1usize, 32usize, 8usize, 128usize);
    let kv = batch * kv_heads;
    let mut g = Gauss::new(11);
    let q_data: Vec<f32> = (0..batch * q_heads * dim)
        .map(|_| g.normal() as f32)
        .collect();
    let queries =
        InlineArray::from_f32_slice(&q_data, &[batch as i32, q_heads as i32, 1, dim as i32])
            .as_dtype(Dtype::Bfloat16.as_i32());
    let scale = 1.0 / (dim as f32).sqrt();
    let shape = DecodeShape {
        batch,
        q_heads,
        kv_heads,
        dim,
        value_dim: dim,
    };
    for &n in &[4096usize, 32768, 131072] {
        // bf16 reference: MLX's own vector SDPA.
        let k_dense = InlineArray::random_normal(
            &[batch as i32, kv_heads as i32, n as i32, dim as i32],
            Dtype::Bfloat16.as_i32(),
        );
        let v_dense = InlineArray::random_normal(
            &[batch as i32, kv_heads as i32, n as i32, dim as i32],
            Dtype::Bfloat16.as_i32(),
        );
        k_dense.eval();
        v_dense.eval();
        let t_bf16 = time_ms(30, || queries.sdpa(&k_dense, &v_dense, scale, ""));
        let bytes_bf16 = (2 * kv * n * dim * 2) as f64;
        eprintln!(
            "n={n:6}  mlx bf16 sdpa: {t_bf16:7.3} ms  ({:6.1} GB/s)",
            bytes_bf16 / t_bf16 / 1e6
        );
        for &bits in &[8u8, 4, 3, 2] {
            let keys = Quantizer::new(dim, bits);
            let values = Quantizer::new(dim, bits);
            let cold = synthetic_cold(kv, n, &keys, &values);
            let dense = DenseStore::new(kv, dim, dim, Dtype::Bfloat16.as_i32(), 0, 0);
            let t = time_ms(30, || {
                let visibility = Visibility {
                    first_token: 0,
                    sink_logits: None,
                };
                decode_attention(
                    &queries,
                    scale,
                    visibility,
                    &shape,
                    &keys,
                    &values,
                    Some(&cold),
                    &dense,
                )
                .expect("attention")
            });
            let bytes = (kv * n * (2 * words_per_row(dim, bits) * 4 + 8)) as f64;
            eprintln!(
                "n={n:6}  turboquant {bits}-bit: {t:7.3} ms  ({:6.1} GB/s packed)  {:.2}x vs bf16 sdpa",
                bytes / t / 1e6,
                t_bf16 / t
            );
        }
    }
}
