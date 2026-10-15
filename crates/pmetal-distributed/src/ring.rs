//! Ring backend for an explicitly configured list of peers.
//!
//! The algorithms live in [`crate::collective`]; this type only forms the
//! ring (authenticated and encrypted, see [`crate::transport`]) and presents
//! it as a [`DistributedBackend`].

use crate::{
    DistributedBackend, ReduceOp,
    collective::RingCollective,
    config::DistributedConfig,
    transport::{Channel, LinkSecurity, RingMembership, TcpTransport},
};
use anyhow::Result;
use async_trait::async_trait;
use std::ops::Range;

/// All-reduce over a ring of manually configured peers.
pub struct RingBackend {
    ring: RingCollective,
}

impl RingBackend {
    /// Join the ring `config` describes: bind this rank's port, dial the next
    /// rank, accept the previous one, and authenticate both with the cluster
    /// secret (`config.secret`, else the machine's; see
    /// [`ClusterSecret::resolve`](crate::secret::ClusterSecret::resolve)).
    pub async fn new(config: DistributedConfig) -> Result<Self> {
        config.validate()?;
        let security = LinkSecurity::from_config(&config)?;
        let membership = RingMembership::from_config(&config);
        let (sender, receiver) =
            TcpTransport::connect_ring(&config, security, membership, None, Channel::Ring).await?;
        Ok(Self {
            ring: RingCollective::new(config.rank, config.nodes.len(), sender, receiver),
        })
    }

    /// The collectives, for operations the [`DistributedBackend`] trait does
    /// not carry.
    pub fn collective(&self) -> &RingCollective {
        &self.ring
    }

    #[cfg(test)]
    pub(crate) fn from_collective(ring: RingCollective) -> Self {
        Self { ring }
    }
}

#[async_trait]
impl DistributedBackend for RingBackend {
    fn rank(&self) -> usize {
        self.ring.rank()
    }

    fn world_size(&self) -> usize {
        self.ring.world_size()
    }

    async fn all_reduce_f32(&self, data: &mut [f32], op: ReduceOp) -> Result<()> {
        self.ring.all_reduce(data, op).await
    }

    async fn reduce_scatter_f32(&self, data: &mut [f32], op: ReduceOp) -> Result<Range<usize>> {
        self.ring.reduce_scatter(data, op).await
    }

    async fn all_gather_f32_in_place(&self, data: &mut [f32]) -> Result<()> {
        self.ring.all_gather_in_place(data).await
    }

    async fn all_gather_bytes(&self, local: &[u8]) -> Result<Vec<Vec<u8>>> {
        self.ring.all_gather_bytes(local).await
    }

    async fn broadcast(&self, buffer: &mut [u8], root: usize) -> Result<()> {
        self.ring.broadcast(buffer, root).await
    }

    async fn barrier(&self) -> Result<()> {
        self.ring.barrier().await
    }
}

#[cfg(kani)]
mod verification {
    use crate::collective::chunk_range;

    #[kani::proof]
    #[kani::unwind(17)] // Sufficient for world_size up to 16 for testing
    fn verify_chunk_range() {
        let len: usize = kani::any();
        let world_size: usize = kani::any();

        // Preconditions
        kani::assume(world_size > 0 && world_size <= 16);
        kani::assume(len < 1024);

        let mut total_elements = 0;
        let mut last_end = 0;

        for i in 0..world_size {
            let range = chunk_range(len, world_size, i);

            // Chunks must be valid ranges
            assert!(range.start <= range.end);
            // Chunks must be contiguous
            assert!(range.start == last_end);
            // Chunks must be within bounds
            assert!(range.end <= len);

            total_elements += range.end - range.start;
            last_end = range.end;
        }

        // Total elements must match original length
        assert!(total_elements == len);
        assert!(last_end == len);
    }
}
