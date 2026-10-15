//! End-to-end accuracy: the cache's attention output against f64 attention
//! over the original, unquantized keys and values.
//!
//! The workload (see `reference_tests`) has keys with a few high-variance
//! channels, token norms that vary, and queries scaled to a logit standard
//! deviation of 3. Random errors of relative size `e` in the keys move a
//! logit by about `e` times its standard deviation, so the output errors below
//! are those of the quantizer, not of the kernels: `attention_tests` shows
//! the kernels agree with f64 attention over the dequantized cache to 3e-3 (the queries are rounded to halves).

#![cfg(test)]

use super::reference_tests::{Workload, relative_error, run_cache};
use super::*;

fn mean_attention_error(
    dim: usize,
    bits: u8,
    groups: (usize, usize),
    seeds: u64,
    window: Option<usize>,
    sinks: usize,
) -> f64 {
    let seq = 1537;
    let mut total = 0.0;
    for seed in 0..seeds {
        let w = Workload::new(1, groups.0, groups.1, seq, dim, 100 + seed);
        // The key frame is off here: these budgets are the quantizer's own,
        // on a workload whose queries sit where the keys vary most (the
        // frame's tests below use ones that do not).
        let cfg = TurboQuantConfig::uniform(bits, bits)
            .with_recent_window(window)
            .with_sink_tokens(sinks)
            .with_center_keys(false)
            .with_scale_key_channels(false);
        let (_, got) = run_cache(&w, cfg);
        total += relative_error(&got, &w.reference(seq));
    }
    total / seeds as f64
}

#[test]
fn attention_error_stays_within_budget_for_every_bit_width() {
    // Measured means on this workload run from 0.7% to 0.9% at 8 bits, 3 to
    // 4% at 6, 6 to 8% at 5, 11 to 14% at 4, 22 to 26% at 3 and 39 to 50% at
    // 2; the budgets leave a third of headroom for the seeds.
    let budget = [
        (8u8, 0.013),
        (6, 0.055),
        (5, 0.11),
        (4, 0.19),
        (3, 0.35),
        (2, 0.66),
    ];
    for &dim in &[64usize, 128, 256] {
        for &(bits, limit) in &budget {
            for groups in [(4usize, 4usize), (8, 2)] {
                let err = mean_attention_error(dim, bits, groups, 3, None, 0);
                assert!(
                    err < limit,
                    "d={dim} {bits}-bit q{}/kv{}: attention error {err:.4} over the budget {limit}",
                    groups.0,
                    groups.1
                );
            }
        }
    }
}

#[test]
fn every_extra_bit_roughly_halves_the_attention_error() {
    for &dim in &[64usize, 128, 256] {
        let errors: Vec<f64> = (2..=8u8)
            .map(|bits| mean_attention_error(dim, bits, (4, 4), 3, None, 0))
            .collect();
        for pair in errors.windows(2) {
            assert!(
                pair[1] < pair[0] * 0.8,
                "d={dim}: errors by bit width {errors:?}"
            );
        }
        // 2 to 8 bits spans 6 halvings: the error falls by at least 20x.
        assert!(errors[0] / errors[6] > 20.0, "d={dim}: {errors:?}");
    }
}

#[test]
fn the_recent_window_and_the_sinks_keep_their_tokens_exact() {
    // With the whole sequence inside the window nothing is quantized, so the
    // output is the exact attention up to f32 arithmetic.
    let err = mean_attention_error(128, 2, (4, 4), 2, Some(4096), 0);
    assert!(
        err < super::attention_tests::KERNEL_TOLERANCE,
        "everything in the window: {err:.2e}"
    );
    // With a window covering half of it the error is below that of
    // quantizing everything, and falls as the window grows.
    let none = mean_attention_error(128, 3, (4, 4), 3, None, 0);
    let small = mean_attention_error(128, 3, (4, 4), 3, Some(512), 0);
    let large = mean_attention_error(128, 3, (4, 4), 3, Some(1024), 0);
    assert!(
        small < none && large < small,
        "window 0/512/1024: {none:.4} {small:.4} {large:.4}"
    );
}

#[test]
fn an_attention_sink_kept_exact_removes_the_error_it_would_carry() {
    // Token 0's key is built to draw most of the softmax weight whatever the
    // query, as real first tokens do.
    let (q_heads, kv_heads, dim, seq) = (4usize, 4usize, 128usize, 1000usize);
    let mut w = Workload::new(1, q_heads, kv_heads, seq, dim, 7);
    for h in 0..kv_heads {
        let q = w.queries[h * dim..(h + 1) * dim].to_vec();
        let norm_sq: f32 = q.iter().map(|x| x * x).sum();
        for (slot, q_c) in w.keys[h * seq * dim..][..dim].iter_mut().zip(&q) {
            // q . k0 / sqrt(d) = +45, against the sequence's +-3 (and a few
            // channels' +-20): the sink takes the whole softmax.
            *slot = q_c * 45.0 * (dim as f32).sqrt() / norm_sq;
        }
    }
    let error = |sinks: usize| {
        let cfg = TurboQuantConfig::uniform(3, 3)
            .with_recent_window(None)
            .with_sink_tokens(sinks);
        let (_, got) = run_cache(&w, cfg);
        relative_error(&got, &w.reference(seq))
    };
    let without = error(0);
    let with = error(4);
    // Without the sink the output is the sink's value quantized to 3 bits
    // (relative error about 0.18); with it, the output is exact.
    assert!(without > 0.1, "quantized sink: {without:.4}");
    assert!(with < 0.01, "exact sink: {with:.4}");
}

#[test]
fn centring_the_keys_removes_the_error_a_shared_key_component_causes() {
    // Every key of a head carries the same large offset, as real keys do. It
    // adds one constant to each query's logits, so attention is unchanged,
    // but without centring it takes most of each key's norm and the codebook
    // is spent on it. The window and the sinks leave both a dense and a cold
    // part, so the shift has to be right on both sides.
    let (q_heads, kv_heads, dim, seq) = (8usize, 2usize, 128usize, 1537usize);
    let mean_error = |center: bool| {
        let mut total = 0.0;
        for seed in 0..3 {
            let w = Workload::new(1, q_heads, kv_heads, seq, dim, 300 + seed)
                .with_key_bias(24.0, 900 + seed);
            let cfg = TurboQuantConfig::uniform(4, 4)
                .with_recent_window(Some(64))
                .with_sink_tokens(4)
                .with_center_keys(center)
                .with_scale_key_channels(false);
            let (_, got) = run_cache(&w, cfg);
            total += relative_error(&got, &w.reference(seq));
        }
        total / 3.0
    };
    let plain = mean_error(false);
    let centred = mean_error(true);
    eprintln!("4-bit keys with a shared offset: plain {plain:.4}, centred {centred:.4}");
    // The budget of a 4-bit cache on keys without the offset.
    assert!(centred < 0.19, "centred: {centred:.4}");
    assert!(
        plain > 2.0 * centred,
        "plain {plain:.4} vs centred {centred:.4}"
    );
}

#[test]
fn scaling_the_key_channels_removes_the_error_loud_channels_cause() {
    // A quarter of the channels vary 8x as much as the rest and the queries
    // look at the quiet ones. The quantizer's error is as large in every
    // direction, so next to the logits' small spread it is far too large
    // unless the loud channels are brought down to the others' size (and
    // the query's up by the same factor).
    let (q_heads, kv_heads, dim, seq) = (8usize, 2usize, 128usize, 1537usize);
    let mean_error = |bits: u8, scale: bool| {
        let mut total = 0.0;
        for seed in 0..3 {
            let w = Workload::lopsided(1, q_heads, kv_heads, seq, dim, 500 + seed);
            let cfg = TurboQuantConfig::uniform(bits, bits)
                .with_recent_window(Some(64))
                .with_sink_tokens(4)
                .with_scale_key_channels(scale);
            let (_, got) = run_cache(&w, cfg);
            total += relative_error(&got, &w.reference(seq));
        }
        total / 3.0
    };
    for (bits, budget) in [(8u8, 0.025), (4, 0.19)] {
        let plain = mean_error(bits, false);
        let scaled = mean_error(bits, true);
        eprintln!("{bits}-bit lopsided keys: plain {plain:.4}, scaled {scaled:.4}");
        assert!(scaled < budget, "{bits}-bit scaled: {scaled:.4}");
        assert!(
            plain > 2.0 * scaled,
            "{bits}-bit: plain {plain:.4} vs scaled {scaled:.4}"
        );
    }
}
