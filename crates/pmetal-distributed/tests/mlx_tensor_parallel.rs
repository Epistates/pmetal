//! Tensor-parallel layers against the single-device computation, by real
//! ranks over loopback: the forward result and the gradients of a SwiGLU MLP
//! (column-parallel up/gate projections, row-parallel down projection) must
//! be what one device computes with the whole weights.

#![cfg(feature = "tensor-parallel")]

mod common;

use pmetal_distributed::mlx_dist::DistributedGroup;
use pmetal_distributed::tensor_parallel::{
    ShardingDirective, all_to_sharded_forward, shard_weight, sharded_to_all_forward,
};
use pmetal_runtime::compat::{Array, nn, ops};
use pmetal_runtime::inline_array::value_and_grad;
use std::time::Duration;

const TOKENS: i32 = 4;
const HIDDEN: i32 = 8;
const INTER: i32 = 12;

fn noise(seed: u64, shape: &[i32]) -> Array {
    let len: i32 = shape.iter().product();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let data: Vec<f32> = (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.8
        })
        .collect();
    Array::from_f32_slice(&data, shape)
}

fn read(array: &Array) -> Vec<f32> {
    let mut array = array.clone();
    array.try_eval().expect("eval");
    let n = array.size();
    array.to_f32_vec(n).expect("readback")
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
            "{what}[{i}]: {g} vs {w}"
        );
    }
}

/// The MLP on whole weights: returns the scalar loss `sum(y * probe)`.
fn dense_loss(a: &[Array]) -> Array {
    let (x, w_up, w_gate, w_down, probe) = (&a[0], &a[1], &a[2], &a[3], &a[4]);
    let h = ops::multiply(&nn::silu(&x.matmul(&w_gate.t())), &x.matmul(&w_up.t()));
    ops::sum_all(&ops::multiply(&h.matmul(&w_down.t()), probe))
}

/// The same MLP on this rank's shards, communicating as tensor parallelism does.
fn sharded_loss(group: &DistributedGroup, a: &[Array]) -> Array {
    let (x, w_up, w_gate, w_down, probe) = (&a[0], &a[1], &a[2], &a[3], &a[4]);
    let up = all_to_sharded_forward(x, w_up, None, group).expect("up");
    let gate = all_to_sharded_forward(x, w_gate, None, group).expect("gate");
    let h = ops::multiply(&nn::silu(&gate), &up);
    let y = sharded_to_all_forward(&h, w_down, None, group).expect("down");
    ops::sum_all(&ops::multiply(&y, probe))
}

fn tensor_parallel_mlp(ranks: usize, name: &str) {
    if !common::is_rank() {
        return common::launch(ranks, name, Duration::from_secs(180));
    }
    let group = DistributedGroup::init(true).expect("group");
    let (rank, size) = (group.rank() as usize, group.size() as usize);
    assert_eq!(size, ranks);

    let x = noise(1, &[TOKENS, HIDDEN]);
    let w_up = noise(2, &[INTER, HIDDEN]);
    let w_gate = noise(3, &[INTER, HIDDEN]);
    let w_down = noise(4, &[HIDDEN, INTER]);
    let probe = noise(5, &[TOKENS, HIDDEN]);

    // One device, whole weights.
    let (loss, grads) = value_and_grad(
        dense_loss,
        &[x.clone(), w_up.clone(), w_gate.clone(), w_down.clone()],
        std::slice::from_ref(&probe),
    );
    let want_loss = read(&loss);
    let want: Vec<Vec<f32>> = grads.iter().map(read).collect();

    // This rank's shards.
    let col = ShardingDirective::AllToSharded { axis: 0 };
    let row = ShardingDirective::ShardedToAll { axis: 1 };
    let shard = |w: &Array, d: &ShardingDirective| shard_weight(w, d, rank, size).expect("shard");
    let (up_r, gate_r, down_r) = (
        shard(&w_up, &col),
        shard(&w_gate, &col),
        shard(&w_down, &row),
    );

    let (loss, grads) = value_and_grad(
        |a| sharded_loss(&group, a),
        &[x.clone(), up_r, gate_r, down_r],
        &[probe],
    );
    assert_close(&read(&loss), &want_loss, "loss");

    // The gradient of the input is the sum of every rank's contribution and
    // identical everywhere; each weight gradient is the shard of the full one.
    assert_close(&read(&grads[0]), &want[0], "dx");
    let shard_of = |full: &[f32], d: &ShardingDirective, shape: &[i32]| {
        read(&shard_weight(&Array::from_f32_slice(full, shape), d, rank, size).expect("shard"))
    };
    assert_close(
        &read(&grads[1]),
        &shard_of(&want[1], &col, &[INTER, HIDDEN]),
        "d w_up shard",
    );
    assert_close(
        &read(&grads[2]),
        &shard_of(&want[2], &col, &[INTER, HIDDEN]),
        "d w_gate shard",
    );
    assert_close(
        &read(&grads[3]),
        &shard_of(&want[3], &row, &[HIDDEN, INTER]),
        "d w_down shard",
    );
}

#[test]
fn a_swiglu_mlp_matches_one_device_on_two_ranks() {
    tensor_parallel_mlp(2, "a_swiglu_mlp_matches_one_device_on_two_ranks");
}

#[test]
fn a_swiglu_mlp_matches_one_device_on_three_ranks() {
    tensor_parallel_mlp(3, "a_swiglu_mlp_matches_one_device_on_three_ranks");
}
