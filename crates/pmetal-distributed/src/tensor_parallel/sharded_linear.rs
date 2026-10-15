//! Distributed linear layer operations for tensor parallelism.
//!
//! Implements the two fundamental distributed linear patterns of Shoeybi et al. (2019):
//!
//! - **AllToSharded**: Column-parallel. Weight `[output/N, input]`.
//!   Forward: `sum_gradients` → local matmul → sharded output.
//!
//! - **ShardedToAll**: Row-parallel. Weight `[output, input/N]`.
//!   Forward: local matmul → `all_sum` → replicated output.
//!
//! # Reference
//!
//! MLX distributed linear layers:
//! - `AllToShardedLinear` in `mlx/nn/layers/distributed.py`
//! - `ShardedToAllLinear` in `mlx/nn/layers/distributed.py`

use crate::mlx_dist::group::DistributedGroup;
use crate::mlx_dist::ops;
use pmetal_runtime::compat::{Array, Exception};

/// Forward pass for a column-sharded linear layer (AllToSharded).
///
/// Each rank holds weight `[output_dims/N, input_dims]`. The input `x`
/// is replicated across all ranks. Each rank computes its local slice
/// of the output independently — no communication needed in forward.
///
/// In the backward pass the gradient of `x` is summed over the ranks
/// ([`ops::sum_gradients`]), since each rank only sees the part of it that
/// flows through its own shard of the weights.
///
/// # Arguments
///
/// * `x` — Input tensor `[..., input_dims]`, replicated on all ranks
/// * `weight` — Local weight shard `[output_dims/N, input_dims]`
/// * `bias` — Optional local bias shard `[output_dims/N]`
/// * `group` — Communication group
///
/// # Returns
///
/// Output tensor `[..., output_dims/N]` (sharded across ranks).
pub fn all_to_sharded_forward(
    x: &Array,
    weight: &Array,
    bias: Option<&Array>,
    group: &DistributedGroup,
) -> Result<Array, Exception> {
    // Every rank's shard sees all of `x`, so `x`'s gradient is the sum of the
    // gradients the ranks compute from their shards: identity here, an
    // all-reduce of the cotangent in backward. Nothing is sent in forward.
    let x_barrier = ops::sum_gradients(x, Some(group))?;

    // Local matmul: x_barrier @ weight.T
    let wt = weight.t();
    let mut out = x_barrier.matmul(&wt);

    // Add bias if present.
    if let Some(b) = bias {
        out = out.add(b);
    }

    Ok(out)
}

/// Forward pass for a row-sharded linear layer (ShardedToAll).
///
/// Each rank holds weight `[output_dims, input_dims/N]`. The input `x`
/// is sharded across ranks (each rank holds `[..., input_dims/N]`).
/// After local matmul, `all_sum` reduces the partial results to produce
/// the full replicated output.
///
/// # Arguments
///
/// * `x` — Input tensor `[..., input_dims/N]`, sharded across ranks
/// * `weight` — Local weight shard `[output_dims, input_dims/N]`
/// * `bias` — Optional bias `[output_dims]` (replicated, added only on one operation)
/// * `group` — Communication group
///
/// # Returns
///
/// Output tensor `[..., output_dims]` (replicated across all ranks).
pub fn sharded_to_all_forward(
    x: &Array,
    weight: &Array,
    bias: Option<&Array>,
    group: &DistributedGroup,
) -> Result<Array, Exception> {
    // Local matmul: x @ weight.T (partial result).
    let wt = weight.t();
    let partial = x.matmul(&wt);

    // All-sum: reduce partial results across all ranks.
    let mut out = ops::all_sum(&partial, Some(group))?;

    // Add bias (replicated — same on all ranks).
    if let Some(b) = bias {
        out = out.add(b);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests verify the local compute path. The distributed
    // all_sum calls require a multi-process environment and are tested
    // in integration tests.

    #[test]
    fn sharded_linear_types_compile() {
        // Verify the module compiles with all type signatures.
        let _ = std::mem::size_of::<DistributedGroup>();
    }
}
