//! Collective and point-to-point operations using MLX distributed.
//!
//! These wrap [`pmetal_bridge::distributed`], which calls MLX's C++
//! collectives, providing interfaces that work with
//! `pmetal_bridge::compat::Array`. All operations are lazy — they build the
//! MLX computation graph and execute when `eval()` is called. Passing no
//! group uses MLX's default one.
//!
//! # Collective Operations
//!
//! - [`all_sum`] — Element-wise sum across all ranks (primary for TP all-reduce)
//! - [`all_gather`] — Concatenate tensors from all ranks
//! - [`all_max`] / [`all_min`] — Element-wise max/min reduction
//! - [`sum_scatter`] — Reduce-scatter (sum + split across ranks)
//!
//! # Point-to-Point Operations
//!
//! - [`send`] — Send tensor to a specific rank
//! - [`recv`] — Receive tensor of known shape from a specific rank
//! - [`recv_like`] — Receive tensor matching another tensor's shape/dtype

use super::group::DistributedGroup;
use pmetal_bridge::compat::{Array, Dtype, Exception};
use pmetal_bridge::distributed::{self, Group};

fn group(group: Option<&DistributedGroup>) -> Option<&Group> {
    group.map(|g| &g.inner)
}

fn exception(op: &str, e: pmetal_bridge::BridgeError) -> Exception {
    Exception::custom(format!("mlx distributed {op} failed: {e}"))
}

/// Element-wise sum across all ranks.
///
/// This is the primary collective for tensor parallelism: used in
/// `ShardedToAllLinear` to reduce partial matmul results.
///
/// Returns a new array where each element is the sum of corresponding
/// elements across all ranks in the group.
pub fn all_sum(x: &Array, g: Option<&DistributedGroup>) -> Result<Array, Exception> {
    distributed::all_sum(x, group(g)).map_err(|e| exception("all_sum", e))
}

/// Gather tensors from all ranks, concatenating along the first axis.
///
/// If each rank has a tensor of shape `[N, ...]`, the result has shape
/// `[N * world_size, ...]` with data from rank 0 first, then rank 1, etc.
pub fn all_gather(x: &Array, g: Option<&DistributedGroup>) -> Result<Array, Exception> {
    distributed::all_gather(x, group(g)).map_err(|e| exception("all_gather", e))
}

/// Element-wise maximum across all ranks.
pub fn all_max(x: &Array, g: Option<&DistributedGroup>) -> Result<Array, Exception> {
    distributed::all_max(x, group(g)).map_err(|e| exception("all_max", e))
}

/// Element-wise minimum across all ranks.
pub fn all_min(x: &Array, g: Option<&DistributedGroup>) -> Result<Array, Exception> {
    distributed::all_min(x, group(g)).map_err(|e| exception("all_min", e))
}

/// Reduce-scatter: sum across ranks, then split the result.
///
/// Each rank receives a different shard of the reduced result.
/// If each rank has a tensor of shape `[N, ...]`, each rank gets
/// a tensor of shape `[N / world_size, ...]` after reduction.
pub fn sum_scatter(x: &Array, g: Option<&DistributedGroup>) -> Result<Array, Exception> {
    distributed::sum_scatter(x, group(g)).map_err(|e| exception("sum_scatter", e))
}

/// Send a tensor to a destination rank.
///
/// Returns a sentinel array that must be evaluated to trigger the send.
/// The send is non-blocking in the MLX graph but synchronizes when evaluated.
pub fn send(x: &Array, dst: i32, g: Option<&DistributedGroup>) -> Result<Array, Exception> {
    distributed::send(x, dst, group(g)).map_err(|e| exception("send", e))
}

/// Receive a tensor of known shape and dtype from a source rank.
///
/// The shape and dtype must match what the sender is transmitting.
pub fn recv(
    shape: &[i32],
    dtype: Dtype,
    src: i32,
    g: Option<&DistributedGroup>,
) -> Result<Array, Exception> {
    distributed::recv(shape, dtype.as_i32(), src, group(g)).map_err(|e| exception("recv", e))
}

/// Receive a tensor matching another tensor's shape and dtype.
///
/// Convenience wrapper around [`recv`] that infers shape and dtype
/// from a reference array.
pub fn recv_like(x: &Array, src: i32, g: Option<&DistributedGroup>) -> Result<Array, Exception> {
    distributed::recv_like(x, src, group(g)).map_err(|e| exception("recv_like", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A group of one reduces to its own input. These used to declare the
    /// MLX C API's symbols, which no library in the build provides, so any
    /// binary using them failed to link.
    #[test]
    fn collectives_over_a_group_of_one_return_their_input() {
        let Some(group) = DistributedGroup::init(false) else {
            return;
        };
        let x = Array::from_f32_slice(&[1.5, -2.0], &[2]);
        for out in [
            all_sum(&x, Some(&group)),
            all_max(&x, Some(&group)),
            all_min(&x, None),
            all_gather(&x, None),
        ] {
            let mut out = out.expect("collective");
            out.eval();
            assert_eq!(out.to_f32_vec(2).expect("eval"), vec![1.5, -2.0]);
        }
    }
}
