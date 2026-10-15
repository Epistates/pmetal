//! Ring attention against full attention on one device, by real ranks over
//! loopback. The ranks cut the sequence between them (contiguously or
//! zig-zag), exchange key/value (or query) blocks around the ring, and merge
//! partial attention with an online softmax; every rank's rows must equal the
//! corresponding rows of attention over the whole sequence.

#![cfg(feature = "context-parallel")]

mod common;

use pmetal_distributed::context_parallel::{
    CPMode, RingAttentionConfig, SequenceLayout, gather_sequence, ring_attention_forward,
    split_sequence_with_layout,
};
use pmetal_distributed::mlx_dist::DistributedGroup;
use pmetal_runtime::compat::{Array, Dtype, ops};
use std::time::Duration;

fn noise(seed: u64, shape: &[i32]) -> Array {
    let len: i32 = shape.iter().product();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let data: Vec<f32> = (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect();
    Array::from_f32_slice(&data, shape)
}

fn read(array: &Array) -> Vec<f32> {
    let mut array = array.as_dtype(Dtype::Float32.as_i32());
    array.try_eval().expect("eval");
    let n = array.size();
    array.to_f32_vec(n).expect("readback")
}

/// Attention over the whole sequence on one device, in f32. `q` is
/// `[B, H, S, D]`, `k` and `v` are `[B, Hkv, S, D]`.
fn full_attention(q: &Array, k: &Array, v: &Array, scale: f32, causal: bool) -> Array {
    let (b, h, s, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let hkv = k.shape()[1];
    let expand = |x: &Array| {
        // Repeat each KV head for the `h / hkv` query heads that share it.
        ops::reshape(
            &ops::broadcast_to(&x.expand_dims(2), &[b, hkv, h / hkv, s, d]),
            &[b, h, s, d],
        )
    };
    let (k, v) = (expand(k), expand(v));
    let mut scores =
        ops::matmul(q, &k.swap_axes(2, 3)).multiply(&Array::from_f32_slice(&[scale], &[1]));
    if causal {
        let positions = Array::from_i32_slice(&(0..s).collect::<Vec<_>>());
        let visible = ops::greater_equal(&positions.reshape(&[s, 1]), &positions.reshape(&[1, s]));
        let floor = Array::from_f32_slice(&[-1.0e30], &[1]);
        scores = ops::which(&visible, &scores, &floor);
    }
    ops::matmul(&ops::softmax_axis(&scores, -1), &v)
}

struct Case {
    total: usize,
    heads: i32,
    kv_heads: i32,
    dim: i32,
    layout: SequenceLayout,
    causal: bool,
    mode: CPMode,
}

fn run_case(group: &DistributedGroup, case: &Case) {
    let (rank, size) = (group.rank() as usize, group.size() as usize);
    let s = case.total as i32;
    let (q, k, v) = (
        noise(11, &[1, case.heads, s, case.dim]),
        noise(12, &[1, case.kv_heads, s, case.dim]),
        noise(13, &[1, case.kv_heads, s, case.dim]),
    );
    let scale = 1.0 / (case.dim as f32).sqrt();
    let want = full_attention(&q, &k, &v, scale, case.causal);

    let positions = case.layout.local_positions(case.total, rank, size);
    let rows = Array::from_i32_slice(&positions.iter().map(|&p| p as i32).collect::<Vec<_>>());
    let local = |x: &Array| x.take_axis(&rows, 2);

    let config = RingAttentionConfig {
        scale,
        causal: case.causal,
        mode: case.mode,
        layout: case.layout,
        total_len: case.total,
    };
    let got = ring_attention_forward(&local(&q), &local(&k), &local(&v), &config, group)
        .unwrap_or_else(|e| panic!("rank {rank}: ring attention failed: {e}"));
    let want_rows = local(&want);

    assert_eq!(got.shape(), want_rows.shape(), "output shape");
    let (got, want_rows) = (read(&got), read(&want_rows));
    let worst = got
        .iter()
        .zip(&want_rows)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 2e-5,
        "rank {rank}/{size}, {} tokens, {:?}/{:?}, causal={}: worst error {worst}",
        case.total,
        case.mode,
        case.layout,
        case.causal
    );
}

fn matrix(ranks: usize) -> Vec<Case> {
    let mut cases = Vec::new();
    for &(total, heads, kv_heads) in &[(16, 4, 4), (37, 4, 2), (ranks * 2, 2, 1)] {
        for layout in [SequenceLayout::Contiguous, SequenceLayout::ZigZag] {
            for causal in [true, false] {
                for mode in [CPMode::PassKV, CPMode::PassQ] {
                    cases.push(Case {
                        total,
                        heads,
                        kv_heads,
                        dim: 8,
                        layout,
                        causal,
                        mode,
                    });
                }
            }
        }
    }
    cases
}

fn matches_full_attention(ranks: usize, name: &str) {
    if !common::is_rank() {
        return common::launch(ranks, name, Duration::from_secs(300));
    }
    let group = DistributedGroup::init(true).expect("group");
    assert_eq!(group.size() as usize, ranks);
    for case in matrix(ranks) {
        run_case(&group, &case);
    }
}

#[test]
fn two_ranks_match_full_attention() {
    matches_full_attention(2, "two_ranks_match_full_attention");
}

#[test]
fn three_ranks_match_full_attention() {
    matches_full_attention(3, "three_ranks_match_full_attention");
}

#[test]
fn four_ranks_match_full_attention() {
    matches_full_attention(4, "four_ranks_match_full_attention");
}

/// Blocks far larger than a socket buffer, with every rank sending to its
/// neighbour at once, must not deadlock.
#[test]
fn large_blocks_do_not_deadlock() {
    if !common::is_rank() {
        return common::launch(3, "large_blocks_do_not_deadlock", Duration::from_secs(90));
    }
    let group = DistributedGroup::init(true).expect("group");
    // 3 x 1024 tokens x 4 heads x 128 dims: each rank's K block is 2 MiB and
    // a stacked K/V message 4 MiB.
    run_case(
        &group,
        &Case {
            total: 3072,
            heads: 4,
            kv_heads: 4,
            dim: 128,
            layout: SequenceLayout::ZigZag,
            causal: true,
            mode: CPMode::PassKV,
        },
    );
}

/// Cutting a tensor between ranks and gathering the pieces back, for every
/// layout and for lengths that do not divide evenly, restores it.
#[test]
fn gathering_the_shares_restores_the_sequence() {
    if !common::is_rank() {
        return common::launch(
            3,
            "gathering_the_shares_restores_the_sequence",
            Duration::from_secs(120),
        );
    }
    let group = DistributedGroup::init(true).expect("group");
    let (rank, size) = (group.rank() as usize, group.size() as usize);
    for layout in [SequenceLayout::Contiguous, SequenceLayout::ZigZag] {
        for total in [6usize, 7, 17, 40] {
            let full = noise(5, &[2, total as i32, 3]);
            let share = split_sequence_with_layout(&full, 1, layout, rank, size).expect("split");
            let back = gather_sequence(&share, 1, layout, total, &group).expect("gather");
            assert_eq!(back.shape(), full.shape());
            assert_eq!(read(&back), read(&full), "{layout:?}, {total} tokens");
        }
    }
}
