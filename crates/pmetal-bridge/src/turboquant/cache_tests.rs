//! The cache's bookkeeping (sinks, cold store, recent ring, eviction, forks)
//! and its attention, against the unquantized tokens that went in.

#![cfg(test)]

use super::attention_tests::KERNEL_TOLERANCE;
use super::reference_tests::{Workload, relative_error, to_host};
use super::*;
use crate::InlineArray;
use crate::compat::Dtype;

/// Per-token relative error bound of a `bits`-wide code: the square root of
/// the Lloyd-Max distortion, with room for the spread between tokens: the
/// error is dominated by the few coordinates in the tails, where the cells
/// are wide, so the worst of a thousand tokens errs by about three times the
/// average (and the worst of twenty thousand by ten times at 8 bits). What the
/// bound has to tell apart is a token quantized from the wrong data, which errs
/// by about 1.4.
fn token_error_bound(bits: u8) -> f64 {
    let distortion: [f64; 8] = [
        0.3609, 0.116, 0.034, 0.0093, 0.00245, 0.00063, 0.00016, 0.00004,
    ];
    distortion[usize::from(bits) - 1].sqrt() * 4.5
}

/// `dequantize_*` returns exactly the tokens that were appended, in order:
/// the sinks and the ring bit for bit, the rest within the code's error.
fn assert_contents(cache: &QuantizedKvCache, w: &Workload, len: usize, config: TurboQuantConfig) {
    let TurboQuantTensorConfig::Uniform { bits: kb } = config.keys else {
        panic!()
    };
    let TurboQuantTensorConfig::Uniform { bits: vb } = config.values else {
        panic!()
    };
    assert_eq!(cache.len(), len);
    assert_eq!(cache.hot_len() + cache.cold_len(), len);
    let keys = cache.dequantize_keys().expect("keys");
    let values = cache.dequantize_values().expect("values");
    let dim = w.dim;
    assert_eq!(
        keys.shape(),
        &[w.batch as i32, w.kv_heads as i32, len as i32, dim as i32]
    );
    let (k, v) = (to_host(&keys), to_host(&values));
    let sinks = config.sink_tokens.min(len);
    let hot_tail = cache.hot_len() - sinks;
    for row in 0..w.batch * w.kv_heads {
        for t in 0..len {
            let src = (row * w.seq + t) * dim;
            let got = (row * len + t) * dim;
            let exact = t < sinks || t >= len - hot_tail;
            for (name, got_arr, want_arr, bits) in
                [("key", &k, &w.keys, kb), ("value", &v, &w.values, vb)]
            {
                let a = &got_arr[got..got + dim];
                let b = &want_arr[src..src + dim];
                let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
                let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
                let rel = (num / den).sqrt();
                if exact {
                    assert_eq!(rel, 0.0, "{name} of token {t} (row {row}) should be exact");
                } else {
                    assert!(
                        rel < token_error_bound(bits),
                        "{name} of token {t} (row {row}): relative error {rel:.4}"
                    );
                    assert!(rel > 0.0, "{name} of token {t} was not quantized");
                }
            }
        }
    }
}

fn config(kb: u8, vb: u8, window: Option<usize>, sinks: usize) -> TurboQuantConfig {
    TurboQuantConfig::uniform(kb, vb)
        .with_recent_window(window)
        .with_sink_tokens(sinks)
}

#[test]
fn every_append_pattern_keeps_the_sequence_intact_and_in_order() {
    // (chunk sizes appended in turn, window, sinks)
    let cases: Vec<(Vec<usize>, Option<usize>, usize)> = vec![
        (vec![1; 300], Some(32), 4),
        (vec![100, 1, 1, 1, 250, 7], Some(32), 4),
        (vec![3], Some(32), 4),
        (vec![2, 2, 2], Some(32), 4),
        (vec![500], Some(32), 4),
        (vec![32 + 16, 1], Some(32), 4),
        (vec![1; 120], None, 4),
        (vec![40, 40, 40], None, 0),
        (vec![200, 1, 1], Some(50), 0),
        (vec![17, 31, 64, 5, 90], Some(64), 8),
    ];
    for (chunks, window, sinks) in cases {
        let total: usize = chunks.iter().sum();
        let w = Workload::new(2, 4, 2, total, 64, 3);
        let cfg = config(8, 8, window, sinks);
        let mut cache = new_cache(cfg);
        let mut len = 0;
        for &n in &chunks {
            cache
                .append(
                    &w.kv_slice(false, len, len + n),
                    &w.kv_slice(true, len, len + n),
                )
                .unwrap();
            len += n;
            assert_contents(&cache, &w, len, cfg);
        }
        if let Some(window) = window {
            // The ring holds at most the window plus one eviction chunk.
            assert!(cache.hot_len() <= sinks + window + window.clamp(16, 1024));
            if total > sinks + window + window.clamp(16, 1024) {
                assert!(cache.cold_len() > 0, "{chunks:?}: nothing was evicted");
            }
        } else {
            assert_eq!(cache.hot_len(), sinks.min(total));
        }
        crate::check_unobserved_error().unwrap();
    }
}

#[test]
fn a_long_decode_attends_correctly_at_every_step_across_evictions() {
    // 400 single-token steps through a 32-token window: the ring wraps and
    // is evicted a dozen times. At selected steps the fused attention equals
    // f64 attention over what the cache itself reports it holds.
    let (q_heads, kv_heads, dim) = (4usize, 2usize, 64usize);
    let steps = 400usize;
    let w = Workload::new(1, q_heads, kv_heads, steps, dim, 9);
    let cfg = config(4, 4, Some(32), 4);
    let mut cache = new_cache(cfg);
    for t in 0..steps {
        let out = cache
            .append_and_compute_attention(
                &w.queries_array(),
                &w.kv_slice(false, t, t + 1),
                &w.kv_slice(true, t, t + 1),
                w.scale(),
            )
            .unwrap();
        if [0usize, 1, 3, 4, 5, 20, 36, 37, 70, 71, 72, 150, 399].contains(&t) {
            let len = t + 1;
            let k = to_host(&cache.dequantize_keys().unwrap());
            let v = to_host(&cache.dequantize_values().unwrap());
            let want = w.reference_over(&k, &v, len, len);
            let err = relative_error(&to_host(&out), &want);
            assert!(
                err < KERNEL_TOLERANCE,
                "step {t}: attention differs from the cache's contents by {err:.2e}"
            );
        }
    }
    crate::check_unobserved_error().unwrap();
}

#[test]
fn memory_per_token_is_what_the_bit_widths_say() {
    // With every token quantized (no window, no sinks) a token takes
    // `ceil(D * bits / 32)` words and a norm for the key, and the same for the
    // value, per KV head.
    for &(dim, kb, vb) in &[(128usize, 4u8, 4u8), (128, 3, 2), (256, 4, 3), (96, 8, 8)] {
        let n = 4096usize;
        let w = Workload::new(1, 8, 8, n, dim, 4);
        let mut cache = new_cache(config(kb, vb, None, 0));
        cache
            .append(&w.kv_slice(false, 0, n), &w.kv_slice(true, 0, n))
            .unwrap();
        cache.eval_and_detach_gpu_state();
        let per_token = (words_per_row(dim, kb) * 4 + 4 + words_per_row(dim, vb) * 4 + 4) * 8;
        let bf16_per_token = dim * 2 * 2 * 8;
        let used = cache.nbytes();
        // Capacity is rounded up to a whole growth step.
        assert!(
            used >= per_token * n && used <= per_token * (n + 256) + 4096,
            "d={dim} k{kb} v{vb}: {used} bytes for {n} tokens, expected about {}",
            per_token * n
        );
        let ratio = (bf16_per_token * n) as f64 / used as f64;
        eprintln!("d={dim} k{kb} v{vb}: {:.2}x smaller than bf16", ratio);
        assert!(
            ratio > 16.0 / (f64::from(kb + vb) / 2.0 + 1.1),
            "ratio {ratio}"
        );
    }
}

#[test]
fn bf16_input_is_kept_in_bf16_and_attended_in_bf16() {
    let w = Workload::new(1, 4, 4, 200, 64, 5);
    let bf16 = |a: InlineArray| a.as_dtype(Dtype::Bfloat16.as_i32());
    let mut cache = new_cache(config(4, 4, Some(32), 4));
    cache
        .append(
            &bf16(w.kv_slice(false, 0, 199)),
            &bf16(w.kv_slice(true, 0, 199)),
        )
        .unwrap();
    let out = cache
        .append_and_compute_attention(
            &bf16(w.queries_array()),
            &bf16(w.kv_slice(false, 199, 200)),
            &bf16(w.kv_slice(true, 199, 200)),
            w.scale(),
        )
        .unwrap();
    assert_eq!(out.dtype_raw(), Dtype::Bfloat16.as_i32());
    assert_eq!(
        cache.dequantize_keys().unwrap().dtype_raw(),
        Dtype::Bfloat16.as_i32()
    );
    let k = to_host(&cache.dequantize_keys().unwrap());
    let v = to_host(&cache.dequantize_values().unwrap());
    let want = w.reference_over(&k, &v, 200, 200);
    // bf16 queries are rounded before they reach the kernel; the output is
    // rounded to bf16 on the way out.
    let q_bf16 = to_host(&bf16(w.queries_array()));
    let mut w2 = Workload::new(1, 4, 4, 200, 64, 5);
    w2.queries = q_bf16;
    let want = w2
        .reference_over(&k, &v, 200, 200)
        .into_iter()
        .zip(want)
        .map(|(a, _)| a)
        .collect::<Vec<_>>();
    let err = relative_error(&to_host(&out), &want);
    assert!(err < 1.2e-2, "bf16 attention error {err}");
}

#[test]
fn a_fork_diverges_from_its_parent_without_copying_it() {
    let w = Workload::new(1, 2, 2, 300, 64, 6);
    let cfg = config(8, 8, Some(32), 4);
    let mut parent = new_cache(cfg);
    parent
        .append(&w.kv_slice(false, 0, 200), &w.kv_slice(true, 0, 200))
        .unwrap();
    let mut child = parent.fork();
    child
        .append(&w.kv_slice(false, 200, 300), &w.kv_slice(true, 200, 300))
        .unwrap();
    assert_eq!(parent.len(), 200);
    assert_eq!(child.len(), 300);
    assert_contents(&parent, &w, 200, cfg);
    assert_contents(&child, &w, 300, cfg);
    // The parent can still grow on its own, differently.
    parent
        .append(&w.kv_slice(false, 200, 210), &w.kv_slice(true, 200, 210))
        .unwrap();
    assert_contents(&parent, &w, 210, cfg);
    assert_contents(&child, &w, 300, cfg);
}

#[test]
fn reset_empties_the_cache_and_it_is_reusable_with_other_shapes() {
    let a = Workload::new(1, 2, 2, 100, 64, 7);
    let mut cache = new_cache(config(4, 4, Some(32), 4));
    cache
        .append(&a.kv_slice(false, 0, 100), &a.kv_slice(true, 0, 100))
        .unwrap();
    assert!(cache.nbytes() > 0);
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.nbytes(), 0);
    assert!(cache.dequantize_keys().is_none());
    // A different head count and width, with the state built for 64.
    let b = Workload::new(1, 3, 3, 10, 64, 8);
    cache
        .append(&b.kv_slice(false, 0, 10), &b.kv_slice(true, 0, 10))
        .unwrap();
    assert_eq!(cache.len(), 10);
}

#[test]
fn wrong_shapes_are_refused_not_misread() {
    let w = Workload::new(1, 2, 2, 50, 64, 1);
    let mut cache = new_cache(config(4, 4, Some(32), 4));
    cache
        .append(&w.kv_slice(false, 0, 10), &w.kv_slice(true, 0, 10))
        .unwrap();
    let wider = Workload::new(1, 2, 2, 10, 128, 1);
    assert!(
        cache
            .append(&wider.kv_slice(false, 0, 1), &wider.kv_slice(true, 0, 1))
            .is_err()
    );
    let more_heads = Workload::new(1, 4, 4, 10, 64, 1);
    assert!(
        cache
            .append(
                &more_heads.kv_slice(false, 0, 1),
                &more_heads.kv_slice(true, 0, 1)
            )
            .is_err()
    );
    // A key head dimension the kernels cannot read four coordinates at a time of.
    let odd = Workload::new(1, 2, 2, 5, 66, 1);
    assert!(
        new_cache(config(4, 4, Some(32), 4))
            .append(&odd.kv_slice(false, 0, 5), &odd.kv_slice(true, 0, 5))
            .is_err()
    );
    // Keys and values of different lengths.
    assert!(
        cache
            .append(&w.kv_slice(false, 10, 12), &w.kv_slice(true, 10, 11))
            .is_err()
    );
    // Queries of the wrong width, or more than one token.
    let bad = InlineArray::zeros(&[1, 2, 1, 32], Dtype::Float32.as_i32());
    assert!(cache.compute_attention(&bad, 1.0, 0, None).is_err());
    let long = InlineArray::zeros(&[1, 2, 2, 64], Dtype::Float32.as_i32());
    assert!(cache.compute_attention(&long, 1.0, 0, None).is_err());
    // Query heads that are not a multiple of the KV heads.
    let three = InlineArray::zeros(&[1, 3, 1, 64], Dtype::Float32.as_i32());
    assert!(cache.compute_attention(&three, 1.0, 0, None).is_err());
    // An empty cache has nothing to attend to.
    let empty = new_cache(config(4, 4, Some(32), 4));
    assert!(
        empty
            .compute_attention(&w.queries_array(), 1.0, 0, None)
            .is_err()
    );
    // The cache the failed appends were refused by is still intact.
    assert_eq!(cache.len(), 10);
    // Unsupported configurations are refused up front.
    assert!(TurboQuantConfig::uniform(9, 4).validate().is_err());
    assert!(TurboQuantConfig::uniform(0, 4).validate().is_err());
    assert!(TurboQuantConfig::preset_q3_5(128).validate().is_err());
    assert!(TurboQuantConfig::uniform(4, 4).validate().is_ok());
}

#[test]
fn attention_from_a_given_position_ignores_earlier_tokens() {
    let w = Workload::new(1, 4, 2, 200, 64, 12);
    let cfg = config(8, 8, Some(32), 4);
    let mut cache = new_cache(cfg);
    cache
        .append(&w.kv_slice(false, 0, 200), &w.kv_slice(true, 0, 200))
        .unwrap();
    let k = to_host(&cache.dequantize_keys().unwrap());
    let v = to_host(&cache.dequantize_values().unwrap());
    for first in [0usize, 3, 4, 50, 150, 199] {
        let out = cache
            .compute_attention(&w.queries_array(), w.scale(), first, None)
            .unwrap();
        let mut ks = vec![0.0f32; 0];
        let mut vs = vec![0.0f32; 0];
        // Reference over the suffix, per KV row.
        for row in 0..w.kv_heads {
            ks.extend_from_slice(&k[(row * 200 + first) * 64..(row * 200 + 200) * 64]);
            vs.extend_from_slice(&v[(row * 200 + first) * 64..(row * 200 + 200) * 64]);
        }
        let want = w.reference_over(&ks, &vs, 200 - first, 200 - first);
        let err = relative_error(&to_host(&out), &want);
        assert!(err < KERNEL_TOLERANCE, "first token {first}: {err:.2e}");
    }
}

#[test]
fn per_head_sink_logits_reach_the_cache_attention() {
    let w = Workload::new(1, 4, 4, 100, 64, 13);
    let mut cache = new_cache(config(8, 8, Some(32), 4));
    cache
        .append(&w.kv_slice(false, 0, 100), &w.kv_slice(true, 0, 100))
        .unwrap();
    let plain = to_host(
        &cache
            .compute_attention(&w.queries_array(), w.scale(), 0, None)
            .unwrap(),
    );
    let norm = |v: &[f32], h: usize| {
        v[h * 64..(h + 1) * 64]
            .iter()
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    // A logit far above every score takes all the softmax weight, and the
    // output (which has no value for it) vanishes; one far below changes
    // nothing. Which head's logit applies to which head is told by giving
    // each head a different one.
    let huge = InlineArray::from_f32_slice(&[200.0, -200.0, 200.0, -200.0], &[4]);
    let with = to_host(
        &cache
            .compute_attention(&w.queries_array(), w.scale(), 0, Some(&huge))
            .unwrap(),
    );
    for h in 0..4 {
        if h % 2 == 0 {
            assert!(
                norm(&with, h) < 1e-6 * norm(&plain, h),
                "head {h} should vanish"
            );
        } else {
            assert!(
                (norm(&with, h) - norm(&plain, h)).abs() < 1e-5 * norm(&plain, h),
                "head {h} should be untouched"
            );
        }
    }
}
