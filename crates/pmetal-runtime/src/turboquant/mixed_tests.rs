//! The mixed-precision cache against the unquantized tokens that went in.

#![cfg(test)]

use super::reference_tests::{Workload, relative_error, to_host};
use super::*;
use crate::InlineArray;

fn mixed(
    key: (u8, u8, usize),
    value: (u8, u8, usize),
    window: Option<usize>,
    sinks: usize,
) -> TurboQuantConfig {
    TurboQuantConfig::mixed(key.0, key.1, key.2, value.0, value.1, value.2)
        .with_recent_window(window)
        .with_sink_tokens(sinks)
}

/// Attention of the workload's queries over what `keys` and `values` hold,
/// against attention over the original tokens, in relative L2 error.
fn attention_error(w: &Workload, keys: &InlineArray, values: &InlineArray, len: usize) -> f64 {
    let got = w.reference_over(&to_host(keys), &to_host(values), len, len);
    let want = w.reference(len);
    let got: Vec<f32> = got.into_iter().map(|x| x as f32).collect();
    relative_error(&got, &want)
}

#[test]
fn the_channels_come_back_in_their_own_places() {
    // Everything still in the sinks and the ring is exact, so a channel
    // that landed in the wrong place cannot hide behind quantization error.
    let w = Workload::lopsided(2, 4, 2, 80, 64, 31);
    let cfg = mixed((2, 4, 16), (3, 5, 16), Some(100), 4);
    let mut cache = MixedQuantizedKvCache::new(cfg);
    cache
        .append(&w.kv_slice(false, 0, 80), &w.kv_slice(true, 0, 80))
        .unwrap();
    assert_eq!(cache.len(), 80);
    assert_eq!(cache.cold_len(), 0);
    assert_eq!(
        to_host(&cache.dequantize_keys().unwrap()),
        to_host(&w.kv_slice(false, 0, 80))
    );
    assert_eq!(
        to_host(&cache.dequantize_values().unwrap()),
        to_host(&w.kv_slice(true, 0, 80))
    );
    crate::check_unobserved_error().unwrap();
}

#[test]
fn the_history_is_quantized_and_the_recent_tokens_are_not() {
    let w = Workload::lopsided(1, 4, 2, 400, 64, 32);
    let cfg = mixed((2, 4, 16), (2, 4, 16), Some(32), 4);
    let mut cache = MixedQuantizedKvCache::new(cfg);
    let mut at = 0;
    for n in [150usize, 1, 1, 100, 1, 147] {
        cache
            .append(
                &w.kv_slice(false, at, at + n),
                &w.kv_slice(true, at, at + n),
            )
            .unwrap();
        at += n;
    }
    assert_eq!(cache.len(), 400);
    assert!(cache.cold_len() > 0 && cache.hot_len() > 4);
    let k = to_host(&cache.dequantize_keys().unwrap());
    let original = to_host(&w.kv_slice(false, 0, 400));
    let dim = 64;
    let hot = cache.hot_len();
    for row in 0..2 {
        // Sinks and the newest tokens bit for bit.
        for t in (0..4).chain(400 - (hot - 4)..400) {
            let at = (row * 400 + t) * dim;
            assert_eq!(
                k[at..at + dim],
                original[at..at + dim],
                "row {row} token {t}"
            );
        }
        // The rest carries the error of a 2/4-bit code, not of a wrong channel.
        for t in 4..400 - (hot - 4) {
            let at = (row * 400 + t) * dim;
            let num: f32 = (0..dim)
                .map(|i| (k[at + i] - original[at + i]).powi(2))
                .sum();
            let den: f32 = (0..dim).map(|i| original[at + i].powi(2)).sum();
            let rel = (num / den).sqrt();
            assert!(rel > 0.0 && rel < 0.8, "row {row} token {t}: {rel}");
        }
    }
    crate::check_unobserved_error().unwrap();
}

#[test]
fn mixed_error_lies_between_its_two_widths() {
    // Keys with a quarter of their channels loud (as real heads have): giving
    // those channels more bits is worth more than giving every channel the
    // same average.
    let (q_heads, kv_heads, dim, seq) = (4usize, 2usize, 64usize, 600usize);
    let mut sums = [0.0f64; 4];
    let seeds = 6u64;
    for seed in 0..seeds {
        let w = Workload::lopsided(1, q_heads, kv_heads, seq, dim, 700 + seed);
        let (k, v) = (w.kv_slice(false, 0, seq), w.kv_slice(true, 0, seq));
        let configs = [
            TurboQuantConfig::uniform(2, 2),
            mixed((2, 4, 16), (2, 4, 16), None, 0),
            TurboQuantConfig::uniform(3, 3),
            TurboQuantConfig::uniform(4, 4),
        ];
        for (slot, cfg) in configs.into_iter().enumerate() {
            let cfg = cfg.with_recent_window(None).with_sink_tokens(0);
            let (keys, values) = if cfg.is_mixed() {
                let mut cache = MixedQuantizedKvCache::new(cfg);
                cache.append(&k, &v).unwrap();
                (
                    cache.dequantize_keys().unwrap(),
                    cache.dequantize_values().unwrap(),
                )
            } else {
                let mut cache = new_cache(cfg);
                cache.append(&k, &v).unwrap();
                (
                    cache.dequantize_keys().unwrap(),
                    cache.dequantize_values().unwrap(),
                )
            };
            sums[slot] += attention_error(&w, &keys, &values, seq);
        }
    }
    let [u2, m24, u3, u4] = sums.map(|s| s / seeds as f64);
    // Measured: uniform 2 0.63, mixed 2/4 0.53, uniform 3 0.43, uniform 4 0.19.
    let all =
        format!("uniform 2 {u2:.3}, mixed 2/4 {m24:.3}, uniform 3 {u3:.3}, uniform 4 {u4:.3}");
    assert!(m24 < u2 * 0.9, "mixed should beat uniform 2-bit: {all}");
    assert!(
        u3 < m24,
        "a quarter of the channels at 4 bits is not 3 bits: {all}"
    );
}

#[test]
fn a_mixed_cache_stores_less_than_its_wider_width() {
    let w = Workload::new(1, 2, 2, 300, 64, 33);
    let bytes = |cfg: TurboQuantConfig| {
        let cfg = cfg.with_recent_window(None).with_sink_tokens(0);
        if cfg.is_mixed() {
            let mut cache = MixedQuantizedKvCache::new(cfg);
            cache
                .append(&w.kv_slice(false, 0, 300), &w.kv_slice(true, 0, 300))
                .unwrap();
            cache.nbytes()
        } else {
            let mut cache = new_cache(cfg);
            cache
                .append(&w.kv_slice(false, 0, 300), &w.kv_slice(true, 0, 300))
                .unwrap();
            cache.nbytes()
        }
    };
    let (u2, m, u4) = (
        bytes(TurboQuantConfig::uniform(2, 2)),
        bytes(mixed((2, 4, 16), (2, 4, 16), None, 0)),
        bytes(TurboQuantConfig::uniform(4, 4)),
    );
    assert!(u2 < m && m < u4, "2-bit {u2}, mixed {m}, 4-bit {u4}");
}

#[test]
fn truncate_a_mixed_cache_and_continue() {
    let w = Workload::lopsided(1, 4, 2, 260, 64, 34);
    let cfg = mixed((2, 4, 16), (3, 5, 16), Some(32), 4);
    let mut cache = MixedQuantizedKvCache::new(cfg);
    cache
        .append(&w.kv_slice(false, 0, 260), &w.kv_slice(true, 0, 260))
        .unwrap();
    assert_eq!(cache.truncate(90), 90);
    assert_eq!(cache.len(), 170);
    let kept = to_host(&cache.dequantize_keys().unwrap());
    assert_eq!(kept.len(), 2 * 170 * 64);
    cache
        .append(&w.kv_slice(false, 170, 260), &w.kv_slice(true, 170, 260))
        .unwrap();
    assert_eq!(cache.len(), 260);
    assert_eq!(
        to_host(&cache.dequantize_keys().unwrap()).len(),
        2 * 260 * 64
    );
    assert_eq!(cache.truncate(1000), 260);
    assert!(cache.is_empty() && cache.nbytes() == 0);
    assert!(cache.dequantize_keys().is_none());
    crate::check_unobserved_error().unwrap();
}

#[test]
fn a_fork_and_a_shared_state_leave_the_parent_alone() {
    let w = Workload::new(1, 2, 2, 200, 64, 35);
    let cfg = mixed((2, 4, 16), (2, 4, 16), Some(32), 4);
    let state = std::sync::Arc::new(MixedTurboQuantState::new(64, 64, cfg).unwrap());
    let mut parent = MixedQuantizedKvCache::with_state(cfg, state.clone());
    let mut other = MixedQuantizedKvCache::with_state(cfg, state);
    parent
        .append(&w.kv_slice(false, 0, 120), &w.kv_slice(true, 0, 120))
        .unwrap();
    other
        .append(&w.kv_slice(false, 0, 30), &w.kv_slice(true, 0, 30))
        .unwrap();
    let mut child = parent.fork();
    child
        .append(&w.kv_slice(false, 120, 200), &w.kv_slice(true, 120, 200))
        .unwrap();
    assert_eq!((parent.len(), child.len(), other.len()), (120, 200, 30));
    assert_eq!(
        to_host(&parent.dequantize_keys().unwrap()).len(),
        2 * 120 * 64
    );
}

#[test]
fn what_a_mixed_cache_cannot_hold_is_refused() {
    let w = Workload::new(1, 2, 2, 20, 64, 36);
    let (k, v) = (w.kv_slice(false, 0, 20), w.kv_slice(true, 0, 20));
    // Nothing mixed in the configuration.
    let mut plain = MixedQuantizedKvCache::new(TurboQuantConfig::uniform(4, 4));
    assert!(plain.append(&k, &v).is_err());
    // A uniform cache refuses a mixed configuration.
    assert!(mixed((2, 4, 16), (2, 4, 16), None, 0).validate().is_err());
    // A key dimension the kernels cannot split into groups of four.
    let narrow = Workload::new(1, 2, 2, 20, 6, 37);
    let mut cache = MixedQuantizedKvCache::new(mixed((2, 4, 4), (2, 4, 4), None, 0));
    assert!(
        cache
            .append(
                &narrow.kv_slice(false, 0, 20),
                &narrow.kv_slice(true, 0, 20)
            )
            .is_err()
    );
    // Shapes.
    let mut cache = MixedQuantizedKvCache::new(mixed((2, 4, 16), (2, 4, 16), None, 0));
    assert!(cache.append(&k.reshape(&[2, 20, 64]), &v).is_err());
    cache.append(&k, &v).unwrap();
    let other = Workload::new(1, 3, 3, 20, 64, 38);
    assert!(
        cache
            .append(&other.kv_slice(false, 0, 20), &other.kv_slice(true, 0, 20))
            .is_err()
    );
}

#[test]
fn outlier_counts_are_rounded_to_what_the_kernels_read() {
    // 30 asked for, groups of four: 32 outliers. One mixed tensor with a
    // uniform partner is fine.
    let w = Workload::new(1, 2, 2, 40, 64, 39);
    let cfg = TurboQuantConfig {
        keys: TurboQuantTensorConfig::mixed(2, 4, 30),
        ..TurboQuantConfig::uniform(2, 4)
    }
    .with_recent_window(None)
    .with_sink_tokens(0);
    let mut cache = MixedQuantizedKvCache::new(cfg);
    cache
        .append(&w.kv_slice(false, 0, 40), &w.kv_slice(true, 0, 40))
        .unwrap();
    assert_eq!(cache.len(), 40);
    assert_eq!(
        to_host(&cache.dequantize_keys().unwrap()).len(),
        2 * 40 * 64
    );
    crate::check_unobserved_error().unwrap();
}
