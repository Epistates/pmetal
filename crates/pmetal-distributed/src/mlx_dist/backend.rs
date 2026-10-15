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

    async fn all_reduce(&self, buffer: &mut [u8], op: ReduceOp) -> Result<()> {
        // Reinterpret the byte buffer as f32 values (same alignment requirement
        // as the TCP ring backend).
        if !buffer.len().is_multiple_of(4) {
            anyhow::bail!(
                "MlxDistributedBackend::all_reduce: buffer length {} is not a multiple of 4",
                buffer.len()
            );
        }

        let num_elements = buffer.len() / 4;
        #[allow(unsafe_code)]
        let f32_slice = unsafe {
            // SAFETY: buffer is valid, properly aligned for f32 (caller
            // guarantees gradient buffers are f32-aligned), and length
            // was verified to be a multiple of 4 above.
            std::slice::from_raw_parts(buffer.as_ptr().cast::<f32>(), num_elements)
        };

        // Create an MLX array from the buffer.
        let shape = [num_elements as i32];
        let arr = Array::from_f32_slice(f32_slice, &shape);

        // Perform all_sum using MLX native collectives.
        let reduced = ops::all_sum(&arr, Some(&self.group))
            .map_err(|e| anyhow::anyhow!("mlx all_sum failed: {e}"))?;

        // If Mean reduction, divide by world_size.
        let result = match op {
            ReduceOp::Sum => reduced,
            ReduceOp::Mean => {
                let divisor = Array::from_f32_slice(&[self.group.size() as f32], &[1]);
                reduced.divide(&divisor)
            }
        };

        // Evaluate the computation graph.
        let mut result = result;
        result.eval();

        // Copy the result back into the caller's buffer.
        let result_data = result
            .to_f32_vec(num_elements)
            .ok_or_else(|| anyhow::anyhow!("mlx eval failed: could not extract f32 data"))?;

        #[allow(unsafe_code)]
        let out_f32 = unsafe {
            // SAFETY: buffer is valid for writes, properly aligned for f32,
            // and length was verified above.
            std::slice::from_raw_parts_mut(buffer.as_mut_ptr().cast::<f32>(), num_elements)
        };
        out_f32.copy_from_slice(&result_data[..num_elements]);

        Ok(())
    }

    async fn barrier(&self) -> Result<()> {
        // MLX doesn't have an explicit barrier. Use all_sum of a zero scalar
        // as a synchronization point — all ranks must participate.
        let zero = Array::from_f32_slice(&[0.0f32], &[1]);
        let result = ops::all_sum(&zero, Some(&self.group))
            .map_err(|e| anyhow::anyhow!("mlx barrier (all_sum) failed: {e}"))?;
        result.eval();
        Ok(())
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
