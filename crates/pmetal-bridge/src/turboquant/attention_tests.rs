//! The decode attention kernels against f64 softmax attention over exactly
//! what the cache holds (the dequantized cold store and the dense tokens), so
//! the only difference left is the kernels' f32 arithmetic.

#![cfg(test)]

use super::attention::{DecodeShape, Visibility, decode_attention, max_groups, plan_blocks};
use super::quantizer::Quantizer;
use super::reference_tests::{Gauss, to_host};
use super::store::{ColdStore, DenseStore};
use crate::InlineArray;
use crate::compat::Dtype;

struct Case {
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    dim: usize,
    value_dim: usize,
    key_bits: u8,
    value_bits: u8,
    cold: usize,
    sinks: usize,
    ring_len: usize,
    ring_capacity: usize,
    /// Where the ring starts, to make it wrap.
    ring_start: usize,
    dense_dtype: Dtype,
    query_dtype: Dtype,
    sink_logits: bool,
    seed: u64,
    first_token: usize,
}

impl Case {
    fn new(dim: usize, key_bits: u8, value_bits: u8) -> Self {
        Self {
            batch: 1,
            q_heads: 4,
            kv_heads: 4,
            dim,
            value_dim: dim,
            key_bits,
            value_bits,
            cold: 300,
            sinks: 0,
            ring_len: 0,
            ring_capacity: 0,
            ring_start: 0,
            dense_dtype: Dtype::Float32,
            query_dtype: Dtype::Float32,
            sink_logits: false,
            seed: 1,
            first_token: 0,
        }
    }
}

fn gaussian(n: usize, g: &mut Gauss, scale: f64) -> Vec<f32> {
    (0..n).map(|_| (scale * g.normal()) as f32).collect()
}

/// Tokens of `[KV, n, dim]` with a norm that varies from token to token.
fn tokens(kv: usize, n: usize, dim: usize, g: &mut Gauss) -> Vec<f32> {
    let mut out = Vec::with_capacity(kv * n * dim);
    for _ in 0..kv * n {
        let s = (0.4 * g.normal()).exp();
        out.extend(gaussian(dim, g, s));
    }
    out
}

/// Run `case` and return `(kernel output, f64 reference)`, both
/// `[B * Hq, Dv]`.
fn run(case: &Case) -> (Vec<f32>, Vec<f64>) {
    let kv = case.batch * case.kv_heads;
    let rows = case.batch * case.q_heads;
    let groups = case.q_heads / case.kv_heads;
    let mut g = Gauss::new(case.seed * 7919);
    let keys_q = Quantizer::new(case.dim, case.key_bits);
    let values_q = Quantizer::new(case.value_dim, case.value_bits);

    // Token layout: sinks, cold, ring.
    let k_sink = tokens(kv, case.sinks, case.dim, &mut g);
    let v_sink = tokens(kv, case.sinks, case.value_dim, &mut g);
    let k_cold = tokens(kv, case.cold, case.dim, &mut g);
    let v_cold = tokens(kv, case.cold, case.value_dim, &mut g);
    let k_ring = tokens(kv, case.ring_len, case.dim, &mut g);
    let v_ring = tokens(kv, case.ring_len, case.value_dim, &mut g);

    let dense_dtype = case.dense_dtype.as_i32();
    let arr = |data: &[f32], n: usize, d: usize| {
        InlineArray::from_f32_slice(data, &[kv as i32, n as i32, d as i32]).as_dtype(dense_dtype)
    };

    let mut dense = DenseStore::new(
        kv,
        case.dim,
        case.value_dim,
        dense_dtype,
        case.sinks,
        case.ring_capacity,
    );
    if case.sinks > 0 {
        dense.push(
            &arr(&k_sink, case.sinks, case.dim),
            &arr(&v_sink, case.sinks, case.value_dim),
            case.sinks,
        );
    }
    if case.ring_capacity > 0 {
        if case.ring_start > 0 {
            let pad_k = arr(
                &tokens(kv, case.ring_start, case.dim, &mut g),
                case.ring_start,
                case.dim,
            );
            let pad_v = arr(
                &tokens(kv, case.ring_start, case.value_dim, &mut g),
                case.ring_start,
                case.value_dim,
            );
            dense.push(&pad_k, &pad_v, case.ring_start);
            let _ = dense.evict(case.ring_start);
        }
        if case.ring_len > 0 {
            dense.push(
                &arr(&k_ring, case.ring_len, case.dim),
                &arr(&v_ring, case.ring_len, case.value_dim),
                case.ring_len,
            );
        }
    }

    let cold = (case.cold > 0).then(|| {
        let mut store = ColdStore::empty(kv, &keys_q, &values_q);
        let flat_k =
            InlineArray::from_f32_slice(&k_cold, &[(kv * case.cold) as i32, case.dim as i32]);
        let flat_v =
            InlineArray::from_f32_slice(&v_cold, &[(kv * case.cold) as i32, case.value_dim as i32]);
        store.append(
            &keys_q,
            &values_q,
            &keys_q.encode(&flat_k).unwrap(),
            &values_q.encode(&flat_v).unwrap(),
            case.cold,
        );
        store
    });

    let q_data = gaussian(rows * case.dim, &mut g, 1.7);
    let queries = InlineArray::from_f32_slice(
        &q_data,
        &[case.batch as i32, case.q_heads as i32, 1, case.dim as i32],
    )
    .as_dtype(case.query_dtype.as_i32());
    let sink_values: Vec<f32> = (0..case.q_heads).map(|h| 0.3 * h as f32 - 0.5).collect();
    let sink_arr = case
        .sink_logits
        .then(|| InlineArray::from_f32_slice(&sink_values, &[case.q_heads as i32]));
    let scale = 1.0 / (case.dim as f32).sqrt();

    let shape = DecodeShape {
        batch: case.batch,
        q_heads: case.q_heads,
        kv_heads: case.kv_heads,
        dim: case.dim,
        value_dim: case.value_dim,
    };
    let out = decode_attention(
        &queries,
        scale,
        Visibility {
            first_token: case.first_token,
            sink_logits: sink_arr.as_ref(),
        },
        &shape,
        &keys_q,
        &values_q,
        cold.as_ref(),
        &dense,
    )
    .expect("decode_attention");
    let out_host = to_host(&out);

    // Reference: the sequence in order (sinks, cold as dequantized, ring as
    // stored), from the first visible position on.
    let mut all_k = vec![Vec::new(); kv];
    let mut all_v = vec![Vec::new(); kv];
    let n_dense = dense.tokens();
    let dense_host = dense.ordered().map(|(k, v)| (to_host(&k), to_host(&v)));
    let cold_host = cold.as_ref().map(|store| {
        let (k, v) = store.dequantize(&keys_q, &values_q).unwrap();
        (to_host(&k), to_host(&v))
    });
    for row in 0..kv {
        let part = |src: &[f32], n: usize, d: usize, lo: usize, hi: usize| -> Vec<f32> {
            src[(row * n + lo) * d..(row * n + hi) * d].to_vec()
        };
        for (is_value, dst) in [(false, &mut all_k[row]), (true, &mut all_v[row])] {
            let d = if is_value { case.value_dim } else { case.dim };
            let pick = |pair: &(Vec<f32>, Vec<f32>)| {
                if is_value {
                    pair.1.clone()
                } else {
                    pair.0.clone()
                }
            };
            if let Some(dh) = &dense_host {
                dst.extend(part(&pick(dh), n_dense, d, 0, case.sinks));
            }
            if let Some(ch) = &cold_host {
                dst.extend(part(&pick(ch), case.cold, d, 0, case.cold));
            }
            if let Some(dh) = &dense_host {
                dst.extend(part(&pick(dh), n_dense, d, case.sinks, n_dense));
            }
            dst.drain(..(case.first_token * d).min(dst.len()));
        }
    }
    let q_host = to_host(&queries);
    let mut want = vec![0.0f64; rows * case.value_dim];
    for r in 0..rows {
        let b = r / case.q_heads;
        let h = r % case.q_heads;
        let kv_row = b * case.kv_heads + h / groups;
        let ks = &all_k[kv_row];
        let vs = &all_v[kv_row];
        let n = ks.len() / case.dim;
        let q = &q_host[r * case.dim..(r + 1) * case.dim];
        let mut logits: Vec<f64> = (0..n)
            .map(|t| {
                q.iter()
                    .zip(&ks[t * case.dim..(t + 1) * case.dim])
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>()
                    * f64::from(scale)
            })
            .collect();
        let sink = sink_arr.as_ref().map(|_| f64::from(sink_values[h]));
        let m = logits
            .iter()
            .copied()
            .chain(sink)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut z = sink.map_or(0.0, |s| (s - m).exp());
        for l in &mut logits {
            *l = (*l - m).exp();
            z += *l;
        }
        for t in 0..n {
            for d in 0..case.value_dim {
                want[r * case.value_dim + d] +=
                    logits[t] / z * f64::from(vs[t * case.value_dim + d]);
            }
        }
    }
    (out_host, want)
}

fn assert_close(case: &Case, label: &str, tolerance: f64) {
    let (got, want) = run(case);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in got.iter().zip(&want) {
        num += (f64::from(*a) - b).powi(2);
        den += b * b;
    }
    let err = (num / den.max(1e-30)).sqrt();
    assert!(
        err < tolerance,
        "{label}: relative error {err:.3e} (limit {tolerance:.0e})"
    );
}

#[test]
fn matches_reference_across_head_dims_and_bit_widths() {
    for &dim in &[64usize, 80, 96, 128, 192, 256] {
        for &(kb, vb) in &[(4u8, 4u8), (3, 2), (8, 8), (2, 5), (1, 1)] {
            let case = Case::new(dim, kb, vb);
            assert_close(&case, &format!("d={dim} k{kb} v{vb}"), 2e-5);
        }
    }
}

#[test]
fn keys_and_values_may_differ_in_width() {
    // Multi-head latent attention style: 192-wide keys, 128-wide values.
    let mut case = Case::new(192, 4, 3);
    case.value_dim = 128;
    case.cold = 777;
    assert_close(&case, "mla 192/128", 2e-5);
    let mut case = Case::new(64, 4, 4);
    case.value_dim = 256;
    assert_close(&case, "64/256", 2e-5);
}

#[test]
fn matches_reference_across_sequence_lengths_around_tile_and_block_edges() {
    for &cold in &[
        1usize, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 4097,
    ] {
        let mut case = Case::new(128, 4, 4);
        case.cold = cold;
        assert_close(&case, &format!("cold={cold}"), 2e-5);
    }
}

#[test]
fn grouped_query_attention_maps_query_heads_to_their_kv_head() {
    for &(q_heads, kv_heads, batch) in &[
        (8usize, 2usize, 1usize),
        (8, 1, 2),
        (6, 3, 2),
        (16, 2, 1),
        (4, 4, 3),
    ] {
        let mut case = Case::new(128, 4, 4);
        case.q_heads = q_heads;
        case.kv_heads = kv_heads;
        case.batch = batch;
        case.cold = 500;
        assert_close(&case, &format!("q{q_heads}/kv{kv_heads} b{batch}"), 2e-5);
    }
}

#[test]
fn dense_tail_alone_and_beside_the_cold_store() {
    for &(cold, sinks, ring_len, ring_capacity, ring_start) in &[
        (0usize, 0usize, 50usize, 64usize, 0usize),
        (0, 4, 10, 16, 0),
        (0, 4, 0, 16, 0),
        (300, 4, 40, 64, 0),
        // The ring wraps.
        (300, 4, 40, 64, 50),
        (0, 0, 64, 64, 33),
        (1000, 0, 129, 160, 100),
    ] {
        let mut case = Case::new(128, 4, 3);
        case.cold = cold;
        case.sinks = sinks;
        case.ring_len = ring_len;
        case.ring_capacity = ring_capacity;
        case.ring_start = ring_start;
        assert_close(
            &case,
            &format!("cold {cold} sinks {sinks} ring {ring_len}/{ring_capacity}+{ring_start}"),
            2e-5,
        );
    }
}

#[test]
fn many_query_heads_per_kv_head_are_split_across_launches() {
    // More query heads per KV head than a launch serves: 32 at head dim 128,
    // 16 at 576 (the latent attention of DeepSeek models: 576-wide keys, 512
    // wide values), 24 (not a power of two) at 256.
    for &(q_heads, kv_heads, dim, value_dim, bits) in &[
        (32usize, 1usize, 128usize, 128usize, 4u8),
        (32, 2, 128, 128, 3),
        (16, 1, 576, 512, 4),
        (24, 2, 256, 256, 4),
        (8, 1, 576, 512, 8),
    ] {
        assert!(
            max_groups(dim, value_dim, bits, bits) <= 16,
            "launch cap for d={dim}"
        );
        let mut case = Case::new(dim, bits, bits);
        case.value_dim = value_dim;
        case.q_heads = q_heads;
        case.kv_heads = kv_heads;
        case.cold = 300;
        case.sinks = 2;
        case.ring_len = 20;
        case.ring_capacity = 32;
        case.sink_logits = true;
        assert_close(
            &case,
            &format!("q{q_heads}/kv{kv_heads} d={dim}/{value_dim} {bits}-bit"),
            2e-5,
        );
    }
}

#[test]
fn the_launch_cap_follows_threadgroup_memory_and_registers() {
    // 8 KV-head queries of 128 floats and two 16-entry codebooks fit easily.
    assert_eq!(max_groups(128, 128, 4, 4), 16);
    // Wide values leave a lane fewer registers for accumulators.
    assert!(max_groups(128, 512, 4, 4) < 8);
    // 576-wide queries at 14 per launch would use 32.3 KB of 32 KB.
    assert!(max_groups(576, 512, 4, 4) <= 14);
    assert!(max_groups(4096, 4096, 8, 8) >= 1);
}

#[test]
fn a_window_or_chunk_masks_every_token_before_it() {
    // 4 sinks, 300 cold, 40 in a wrapping ring: 344 tokens in all.
    for first in [0usize, 1, 3, 4, 5, 100, 303, 304, 305, 320, 343] {
        let mut case = Case::new(128, 4, 4);
        case.cold = 300;
        case.sinks = 4;
        case.ring_len = 40;
        case.ring_capacity = 64;
        case.ring_start = 50;
        case.first_token = first;
        assert_close(&case, &format!("first token {first}"), 2e-5);
    }
    // No cold store.
    for first in [0usize, 2, 4, 5, 23] {
        let mut case = Case::new(64, 3, 3);
        case.cold = 0;
        case.sinks = 4;
        case.ring_len = 20;
        case.ring_capacity = 32;
        case.first_token = first;
        assert_close(&case, &format!("no cold, first token {first}"), 2e-5);
    }
    // Everything masked is an error, not a NaN.
    let mut case = Case::new(64, 3, 3);
    case.first_token = 10_000;
    let result = std::panic::catch_unwind(|| run(&case));
    assert!(result.is_err());
}

#[test]
fn half_precision_dense_tokens_and_queries() {
    for dense in [Dtype::Bfloat16, Dtype::Float16] {
        for query in [Dtype::Bfloat16, Dtype::Float16, Dtype::Float32] {
            let mut case = Case::new(128, 4, 4);
            case.dense_dtype = dense;
            case.query_dtype = query;
            case.sinks = 4;
            case.ring_len = 30;
            case.ring_capacity = 64;
            // The output has the queries' precision.
            let tolerance = if query == Dtype::Float32 {
                2e-5
            } else {
                1.2e-2
            };
            assert_close(
                &case,
                &format!("dense {dense:?} query {query:?}"),
                tolerance,
            );
        }
    }
}

#[test]
fn per_head_sink_logits_join_the_normaliser() {
    let mut case = Case::new(128, 4, 4);
    case.sink_logits = true;
    case.cold = 200;
    case.sinks = 2;
    case.ring_len = 20;
    case.ring_capacity = 32;
    assert_close(&case, "with sink logits", 2e-5);
    case.cold = 0;
    case.sinks = 0;
    case.ring_len = 5;
    assert_close(&case, "with sink logits, 5 tokens", 2e-5);
}

#[test]
fn block_plan_covers_every_token_once() {
    for &(cold, dense, kv_rows) in &[
        (0usize, 1usize, 1usize),
        (1, 0, 8),
        (5000, 300, 8),
        (100_000, 9000, 2),
        (131, 77, 64),
    ] {
        let plan = plan_blocks(cold, dense, kv_rows);
        assert_eq!(plan.tokens_per_block % 32, 0);
        assert!(plan.cold_blocks * plan.tokens_per_block >= cold);
        assert!((plan.cold_blocks.saturating_sub(1)) * plan.tokens_per_block < cold.max(1));
        assert!(
            (plan.total_blocks - plan.cold_blocks) * plan.tokens_per_block >= dense,
            "{plan:?}"
        );
    }
}
