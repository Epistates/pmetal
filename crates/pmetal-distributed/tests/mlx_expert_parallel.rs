//! Expert-parallel mixture-of-experts against the single-device layer, by
//! real ranks over loopback. Each rank owns a shard of the experts; the
//! output for every token must equal the weighted sum of its experts'
//! outputs computed on one device with all the experts.

#![cfg(feature = "expert-parallel")]

mod common;

use pmetal_distributed::expert_parallel::{
    CapacityConfig, DropPolicy, ExpertDispatcher, ExpertPlacement, apply_capacity,
};
use pmetal_distributed::mlx_dist::DistributedGroup;
use pmetal_runtime::compat::{Array, Dtype, Exception, ops};
use std::time::Duration;

const DIM: i32 = 6;
const EXPERTS: usize = 8;
const TOP_K: usize = 2;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn read(array: &Array) -> Vec<f32> {
    let mut array = array.as_dtype(Dtype::Float32.as_i32());
    array.try_eval().expect("eval");
    let n = array.size();
    array.to_f32_vec(n).expect("readback")
}

/// Experts `0..EXPERTS`, each a `DIM x DIM` matrix.
fn expert_weights() -> Array {
    let mut rng = Rng(0xDEADBEEF);
    let data: Vec<f32> = (0..EXPERTS as i32 * DIM * DIM)
        .map(|_| rng.next() - 0.5)
        .collect();
    Array::from_f32_slice(&data, &[EXPERTS as i32, DIM, DIM])
}

/// `tokens` tokens, their routing weights and the `TOP_K` distinct experts
/// each picks (always from `allowed` when given).
fn routing(seed: u64, tokens: usize, allowed: Option<&[usize]>) -> (Array, Array, Array, Vec<i32>) {
    let mut rng = Rng(seed | 1);
    let data: Vec<f32> = (0..tokens as i32 * DIM).map(|_| rng.next() - 0.5).collect();
    let hidden = Array::from_f32_slice(&data, &[tokens as i32, DIM]);
    let pool: Vec<usize> = allowed.map_or_else(|| (0..EXPERTS).collect(), <[usize]>::to_vec);
    let (mut weights, mut indices, mut flat): (Vec<f32>, Vec<i32>, Vec<i32>) =
        (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..tokens {
        let mut chosen = Vec::new();
        while chosen.len() < TOP_K {
            let e = pool[(rng.next() * pool.len() as f32) as usize % pool.len()];
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        let w0 = 0.25 + 0.5 * rng.next();
        weights.extend([w0, 1.0 - w0]);
        indices.extend(chosen.iter().map(|&e| e as i32));
        flat.extend(chosen.iter().map(|&e| e as i32));
    }
    (
        hidden,
        Array::from_f32_slice(&weights, &[tokens as i32, TOP_K as i32]),
        Array::from_i32_slice_shaped(&indices, &[tokens as i32, TOP_K as i32]),
        flat,
    )
}

/// The layer on one device with every expert.
/// An expert index below zero is an assignment capacity control dropped.
fn reference(hidden: &Array, weights: &Array, experts_flat: &[i32], table: &Array) -> Vec<f32> {
    let tokens = hidden.shape()[0] as usize;
    let h = read(hidden);
    let w = read(weights);
    let t = read(table);
    let d = DIM as usize;
    let mut out = vec![0f32; tokens * d];
    for tok in 0..tokens {
        for k in 0..TOP_K {
            let Ok(e) = usize::try_from(experts_flat[tok * TOP_K + k]) else {
                continue;
            };
            for j in 0..d {
                let y: f32 = (0..d)
                    .map(|i| h[tok * d + i] * t[e * d * d + i * d + j])
                    .sum();
                out[tok * d + j] += w[tok * TOP_K + k] * y;
            }
        }
    }
    out
}

fn run(ranks: usize, name: &str) {
    if !common::is_rank() {
        return common::launch(ranks, name, Duration::from_secs(180));
    }
    let group = DistributedGroup::init(true).expect("group");
    let (rank, size) = (group.rank() as usize, group.size() as usize);
    assert_eq!(size, ranks);
    let table = expert_weights();
    let placement = ExpertPlacement::uniform(EXPERTS, size, TOP_K);
    let dispatcher = ExpertDispatcher::new(placement.clone(), rank);

    // Only this rank's experts are visible to the expert function.
    let owned = placement.local_expert_ids(rank).to_vec();
    let local_table = table.take_axis(
        &Array::from_i32_slice(&owned.iter().map(|&e| e as i32).collect::<Vec<_>>()),
        0,
    );
    let expert_fn = |rows: &Array, ids: &[usize]| -> Result<Array, Exception> {
        let local: Vec<i32> = ids
            .iter()
            .map(|&e| {
                owned.iter().position(|&o| o == e).unwrap_or_else(|| {
                    panic!("rank {rank} asked to run expert {e} it does not own")
                }) as i32
            })
            .collect();
        let w = local_table.take_axis(&Array::from_i32_slice(&local), 0); // [n, D, D]
        let n = rows.shape()[0];
        Ok(ops::matmul(&rows.reshape(&[n, 1, DIM]), &w).reshape(&[n, DIM]))
    };

    // Different tokens on each rank, different counts, random routing.
    for (case, allowed) in [
        ("random", None),
        ("hot rank 0", Some(&[0usize, 1][..])),
        ("last rank only", Some(&[EXPERTS - 2, EXPERTS - 1][..])),
    ] {
        let tokens = 5 + 2 * rank;
        let (hidden, weights, indices, flat) = routing(100 + rank as u64, tokens, allowed);
        let want = reference(&hidden, &weights, &flat, &table);
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let input = hidden.as_dtype(dtype.as_i32());
            let got = dispatcher
                .dispatch_and_compute(&input, &weights, &indices, &group, expert_fn)
                .unwrap_or_else(|e| panic!("rank {rank} {case} {dtype:?}: {e}"));
            assert_eq!(
                got.dtype(),
                dtype,
                "{case}: the output keeps the input dtype"
            );
            let tolerance = if dtype == Dtype::Float32 { 1e-5 } else { 5e-2 };
            let worst = read(&got)
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < tolerance,
                "rank {rank}/{size}, {case}, {dtype:?}: worst error {worst}"
            );
        }
    }

    // Capacity control drops assignments (index -1); the layer must treat them
    // as contributing nothing, including when a whole rank's slots are gone.
    for policy in [DropPolicy::Drop, DropPolicy::Redistribute] {
        let tokens = 6 + rank;
        let (hidden, weights, indices, _) = routing(500 + rank as u64, tokens, None);
        let config = CapacityConfig {
            capacity_factor: 0.5,
            drop_policy: policy,
        };
        let (capped, _) = apply_capacity(&indices, &config, EXPERTS).expect("capacity");
        let mut ints = capped.as_dtype(Dtype::Float32.as_i32());
        ints.try_eval().expect("eval");
        let n = ints.size();
        let flat: Vec<i32> = ints
            .to_f32_vec(n)
            .expect("readback")
            .into_iter()
            .map(|x| x as i32)
            .collect();
        assert!(
            flat.contains(&-1),
            "the capacity should have dropped something"
        );
        let want = reference(&hidden, &weights, &flat, &table);
        let got = dispatcher
            .dispatch_and_compute(&hidden, &weights, &capped, &group, expert_fn)
            .unwrap_or_else(|e| panic!("rank {rank} capped {policy:?}: {e}"));
        let worst = read(&got)
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-5,
            "rank {rank}/{size}, capped {policy:?}: worst error {worst}"
        );
    }

    // The same tokens on every rank.
    let (hidden, weights, indices, flat) = routing(7, 9, None);
    let want = reference(&hidden, &weights, &flat, &table);
    let got = dispatcher
        .compute_replicated(&hidden, &weights, &indices, &group, expert_fn)
        .expect("replicated");
    let worst = read(&got)
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-5,
        "rank {rank}/{size}, replicated: worst error {worst}"
    );
}

#[test]
fn two_ranks_match_the_single_device_layer() {
    run(2, "two_ranks_match_the_single_device_layer");
}

#[test]
fn three_ranks_match_the_single_device_layer() {
    run(3, "three_ranks_match_the_single_device_layer");
}

#[test]
fn four_ranks_match_the_single_device_layer() {
    run(4, "four_ranks_match_the_single_device_layer");
}
