//! MLX's ring collectives, driven through this crate's wrappers by real
//! ranks (separate processes over loopback). No GPU work beyond a few tiny
//! arrays.

#![cfg(feature = "mlx-collectives")]

mod common;

use pmetal_distributed::mlx_dist::{DistributedGroup, ops};
use pmetal_runtime::compat::Array;
use std::time::Duration;

fn values(array: &Array, n: usize) -> Vec<f32> {
    let mut array = array.clone();
    array.try_eval().expect("eval");
    array.to_f32_vec(n).expect("readback")
}

fn group() -> DistributedGroup {
    DistributedGroup::init(true).expect("join the MLX group")
}

#[test]
fn all_sum_and_all_gather_across_three_ranks() {
    if !common::is_rank() {
        return common::launch(
            3,
            "all_sum_and_all_gather_across_three_ranks",
            Duration::from_secs(120),
        );
    }
    let g = group();
    let (rank, size) = (g.rank(), g.size());
    assert_eq!(size, 3);
    let local = Array::from_f32_slice(&[rank as f32 + 1.0, 10.0 * (rank as f32 + 1.0)], &[2]);

    let summed = ops::all_sum(&local, Some(&g)).expect("all_sum");
    assert_eq!(values(&summed, 2), vec![6.0, 60.0]);

    let gathered = ops::all_gather(&local, Some(&g)).expect("all_gather");
    assert_eq!(
        values(&gathered, 6),
        vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
        "rank 0's block first"
    );
}
