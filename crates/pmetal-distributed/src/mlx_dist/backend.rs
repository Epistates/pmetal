//! MLX-native distributed backend implementing [`DistributedBackend`].
//!
//! Bridges the existing `DistributedBackend` trait (which operates on raw
//! byte buffers) to MLX's native collective operations (which operate on
//! `pmetal_runtime::compat::Array`). This enables using JACCL/RDMA over
//! Thunderbolt 5 for gradient synchronization in the training loop.

use super::group::DistributedGroup;
use super::ops;
use crate::{DistributedBackend, ReduceOp};
use anyhow::Result;
use async_trait::async_trait;
use pmetal_runtime::compat::Array;
use tracing::debug;

/// Distributed backend using MLX native collectives (JACCL/Ring).
///
/// This backend leverages Apple's JACCL library for RDMA-based
/// communication over Thunderbolt 5 when available, falling back
/// to TCP ring when JACCL is not configured.
///
/// # Performance
///
/// - **JACCL/RDMA**: ~3µs latency, ~45 GB/s throughput
/// - **Ring/TCP**: ~150µs latency, ~3.8 GB/s throughput
///
/// # Usage
///
/// ```ignore
/// let group = DistributedGroup::init(true).expect("init");
/// let backend = MlxDistributedBackend::new(group);
/// let ctx = DistributedContext::new(Box::new(backend));
/// ctx.all_reduce(&mut gradient_buffer, ReduceOp::Mean).await?;
/// ```
pub struct MlxDistributedBackend {
    group: DistributedGroup,
}

impl MlxDistributedBackend {
    /// Create a new MLX distributed backend from an initialized group.
    pub fn new(group: DistributedGroup) -> Self {
        debug!(
            "MlxDistributedBackend created: rank={}, size={}",
            group.rank(),
            group.size()
        );
        Self { group }
    }

    /// Try to create a backend, returning None if distributed is unavailable.
    pub fn try_new(strict: bool) -> Option<Self> {
        DistributedGroup::init(strict).map(Self::new)
    }

    /// Get a reference to the underlying group.
    pub fn group(&self) -> &DistributedGroup {
        &self.group
    }
}

#[async_trait]
impl DistributedBackend for MlxDistributedBackend {
    fn rank(&self) -> usize {
        self.group.rank() as usize
    }

    fn world_size(&self) -> usize {
        self.group.size() as usize
    }

    async fn all_reduce_f32(&self, data: &mut [f32], op: ReduceOp) -> Result<()> {
        // MLX array shapes are `i32`, so a buffer past 2^31 elements goes in
        // pieces (each rank cuts at the same places: lengths agree).
        const MAX_ELEMS: usize = 1 << 30;
        let size = self.group.size() as f32;
        blocking(|| {
            for part in data.chunks_mut(MAX_ELEMS) {
                let arr = Array::from_f32_slice(part, &[part.len() as i32]);
                let reduced = ops::all_sum(&arr, Some(&self.group))
                    .map_err(|e| anyhow::anyhow!("mlx all_sum failed: {e}"))?;
                let mut result = match op {
                    ReduceOp::Sum => reduced,
                    ReduceOp::Mean => reduced.divide(&Array::from_f32_slice(&[size], &[1])),
                };
                result
                    .try_eval()
                    .map_err(|e| anyhow::anyhow!("mlx all_reduce failed: {e}"))?;
                let values = result
                    .to_f32_vec(part.len())
                    .ok_or_else(|| anyhow::anyhow!("mlx all_reduce: could not read the result"))?;
                part.copy_from_slice(&values);
            }
            Ok(())
        })
    }

    async fn barrier(&self) -> Result<()> {
        // MLX has no explicit barrier; an all_sum of a scalar needs every rank.
        blocking(|| {
            let zero = Array::from_f32_slice(&[0.0f32], &[1]);
            let result = ops::all_sum(&zero, Some(&self.group))
                .map_err(|e| anyhow::anyhow!("mlx barrier (all_sum) failed: {e}"))?;
            result
                .try_eval()
                .map_err(|e| anyhow::anyhow!("mlx barrier failed: {e}"))?;
            Ok(())
        })
    }
}

/// Run a blocking MLX collective without stalling the async executor's
/// other tasks: on a multi-threaded runtime the worker is handed off, on a
/// single-threaded one there is nothing to hand off to.
fn blocking<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    use tokio::runtime::{Handle, RuntimeFlavor};
    match Handle::try_current().map(|h| h.runtime_flavor()) {
        Ok(RuntimeFlavor::MultiThread) => tokio::task::block_in_place(f),
        _ => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_returns_none_when_unavailable() {
        if !DistributedGroup::is_available() {
            assert!(MlxDistributedBackend::try_new(false).is_none());
        }
    }
}
