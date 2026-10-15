//! MLX-native distributed backend implementing [`DistributedBackend`].
//!
//! Bridges the existing `DistributedBackend` trait (which operates on raw
//! byte buffers) to MLX's native collective operations (which operate on
//! `pmetal_runtime::compat::Array`). This enables using JACCL/RDMA over
//! Thunderbolt 5 for gradient synchronization in the training loop.

use super::group::DistributedGroup;
use super::ops;
use crate::collective::chunk_range;
use crate::{DistributedBackend, ReduceOp};
use anyhow::Result;
use async_trait::async_trait;
use pmetal_runtime::compat::Array;
use std::ops::Range;
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

    async fn reduce_scatter_f32(&self, data: &mut [f32], op: ReduceOp) -> Result<Range<usize>> {
        // MLX's ring backend has no `sum_scatter`, so reduce everything and
        // keep this rank's chunk: twice the traffic of a true reduce-scatter,
        // but the same result as the TCP ring's.
        self.all_reduce_f32(data, op).await?;
        Ok(chunk_range(data.len(), self.world_size(), self.rank()))
    }

    async fn all_gather_f32_in_place(&self, data: &mut [f32]) -> Result<()> {
        let (n, rank) = (self.world_size(), self.rank());
        // `all_gather` joins equal-shaped arrays, and chunks differ by one
        // element, so every chunk is padded to the longest.
        let longest = chunk_range(data.len(), n, 0).len();
        if longest == 0 {
            return Ok(());
        }
        blocking(|| {
            let mut mine = data[chunk_range(data.len(), n, rank)].to_vec();
            mine.resize(longest, 0.0);
            let gathered = ops::all_gather(
                &Array::from_f32_slice(&mine, &[longest as i32]),
                Some(&self.group),
            )
            .map_err(|e| anyhow::anyhow!("mlx all_gather failed: {e}"))?;
            let mut gathered = gathered;
            gathered
                .try_eval()
                .map_err(|e| anyhow::anyhow!("mlx all_gather failed: {e}"))?;
            let all = gathered
                .to_f32_vec(n * longest)
                .ok_or_else(|| anyhow::anyhow!("mlx all_gather: could not read the result"))?;
            for r in 0..n {
                let range = chunk_range(data.len(), n, r);
                data[range.clone()].copy_from_slice(&all[r * longest..r * longest + range.len()]);
            }
            Ok(())
        })
    }

    async fn all_gather_bytes(&self, local: &[u8]) -> Result<Vec<Vec<u8>>> {
        let n = self.world_size();
        blocking(|| {
            // Lengths first (as i32, exact at any size MLX can address).
            let lengths = self.gather_i32(&[i32::try_from(local.len())?])?;
            let longest = lengths.iter().copied().max().unwrap_or(0) as usize;
            if longest == 0 {
                return Ok(vec![Vec::new(); n]);
            }
            // The bytes travel as u32 words, padded to the longest.
            let words = longest.div_ceil(4);
            let mut mine = vec![0u32; words];
            for (word, chunk) in mine.iter_mut().zip(local.chunks(4)) {
                let mut le = [0u8; 4];
                le[..chunk.len()].copy_from_slice(chunk);
                *word = u32::from_le_bytes(le);
            }
            let gathered = ops::all_gather(
                &Array::from_slice(&mine, &[words as i32]),
                Some(&self.group),
            )
            .map_err(|e| anyhow::anyhow!("mlx all_gather failed: {e}"))?;
            gathered
                .try_eval()
                .map_err(|e| anyhow::anyhow!("mlx all_gather failed: {e}"))?;
            let all = gathered.as_slice::<u32>();
            Ok((0..n)
                .map(|r| {
                    let block = &all[r * words..(r + 1) * words];
                    let mut bytes: Vec<u8> = block.iter().flat_map(|w| w.to_le_bytes()).collect();
                    bytes.truncate(lengths[r] as usize);
                    bytes
                })
                .collect())
        })
    }

    async fn broadcast(&self, buffer: &mut [u8], root: usize) -> Result<()> {
        if root >= self.world_size() {
            anyhow::bail!(
                "broadcast root {root} outside a world of {}",
                self.world_size()
            );
        }
        if buffer.is_empty() {
            return Ok(());
        }
        blocking(|| {
            // The root contributes its words and every other rank zeros, so a
            // sum is the root's data exactly.
            let words = buffer.len().div_ceil(4);
            let mut mine = vec![0u32; words];
            if self.rank() == root {
                for (word, chunk) in mine.iter_mut().zip(buffer.chunks(4)) {
                    let mut le = [0u8; 4];
                    le[..chunk.len()].copy_from_slice(chunk);
                    *word = u32::from_le_bytes(le);
                }
            }
            let summed = ops::all_sum(
                &Array::from_slice(&mine, &[words as i32]),
                Some(&self.group),
            )
            .map_err(|e| anyhow::anyhow!("mlx all_sum failed: {e}"))?;
            summed
                .try_eval()
                .map_err(|e| anyhow::anyhow!("mlx broadcast failed: {e}"))?;
            let bytes: Vec<u8> = summed
                .as_slice::<u32>()
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            buffer.copy_from_slice(&bytes[..buffer.len()]);
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

impl MlxDistributedBackend {
    /// Every rank's `values` (all the same length), concatenated.
    fn gather_i32(&self, values: &[i32]) -> Result<Vec<i32>> {
        let gathered = ops::all_gather(
            &Array::from_slice(values, &[values.len() as i32]),
            Some(&self.group),
        )
        .map_err(|e| anyhow::anyhow!("mlx all_gather failed: {e}"))?;
        gathered
            .try_eval()
            .map_err(|e| anyhow::anyhow!("mlx all_gather failed: {e}"))?;
        Ok(gathered.as_slice::<i32>().to_vec())
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
