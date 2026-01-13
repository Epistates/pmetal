//! Configurable collective operations with pluggable strategies.
//!
//! Provides multiple all-reduce, reduce, and broadcast algorithms:
//! - Ring: Bandwidth-optimal for large tensors
//! - Tree: Latency-optimal for small tensors
//! - Centralized: Simple, works for small collectives
//!
//! Based on Burn's collective framework for  operations.

use crate::error::{DistributedError, DistributedResult};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

/// All-reduce algorithm strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AllReduceStrategy {
    /// Ring all-reduce: O(n) latency, O(1) bandwidth per node.
    /// Best for large tensors on high-bandwidth networks.
    Ring,
    /// Tree all-reduce: O(log n) latency, O(log n) bandwidth per node.
    /// Best for small tensors or latency-sensitive operations.
    Tree { arity: usize },
    /// Centralized all-reduce: O(n) latency, O(n) bandwidth on root.
    /// Simple, works for small collectives.
    Centralized,
    /// Automatic selection based on tensor size and cluster topology.
    #[default]
    Auto,
}

/// Reduce algorithm strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReduceStrategy {
    /// Tree reduce with configurable arity.
    Tree { arity: usize },
    /// Direct reduce to root.
    Direct,
    /// Automatic selection.
    #[default]
    Auto,
}

/// Broadcast algorithm strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BroadcastStrategy {
    /// Tree broadcast with configurable arity.
    Tree { arity: usize },
    /// Direct broadcast from root.
    Direct,
    /// Automatic selection.
    #[default]
    Auto,
}

/// Configuration for collective operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveConfig {
    /// Number of local devices (GPUs).
    pub num_devices: usize,
    /// Local all-reduce strategy.
    pub local_all_reduce: AllReduceStrategy,
    /// Local reduce strategy.
    pub local_reduce: ReduceStrategy,
    /// Local broadcast strategy.
    pub local_broadcast: BroadcastStrategy,

    // Global (multi-node) settings
    /// Number of nodes (None = single node).
    pub num_nodes: Option<usize>,
    /// Global all-reduce strategy.
    pub global_all_reduce: Option<AllReduceStrategy>,
    /// Global reduce strategy.
    pub global_reduce: Option<ReduceStrategy>,
    /// Global broadcast strategy.
    pub global_broadcast: Option<BroadcastStrategy>,

    // Tuning parameters
    /// Threshold (bytes) below which tree is preferred over ring.
    pub tree_threshold_bytes: usize,
    /// Tree arity (branching factor).
    pub tree_arity: usize,
    /// Timeout for collective operations.
    pub timeout: Duration,
}

impl Default for CollectiveConfig {
    fn default() -> Self {
        Self {
            num_devices: 1,
            local_all_reduce: AllReduceStrategy::Auto,
            local_reduce: ReduceStrategy::Auto,
            local_broadcast: BroadcastStrategy::Auto,
            num_nodes: None,
            global_all_reduce: None,
            global_reduce: None,
            global_broadcast: None,
            tree_threshold_bytes: 1024 * 1024, // 1 MB
            tree_arity: 2,
            timeout: Duration::from_secs(60),
        }
    }
}

impl CollectiveConfig {
    /// Create a config for a single node with multiple devices.
    pub fn single_node(num_devices: usize) -> Self {
        Self {
            num_devices,
            local_all_reduce: AllReduceStrategy::Ring,
            ..Default::default()
        }
    }

    /// Create a config for multi-node training.
    pub fn multi_node(num_devices: usize, num_nodes: usize) -> Self {
        Self {
            num_devices,
            num_nodes: Some(num_nodes),
            local_all_reduce: AllReduceStrategy::Tree { arity: 2 },
            global_all_reduce: Some(AllReduceStrategy::Ring),
            global_reduce: Some(ReduceStrategy::Tree { arity: 2 }),
            global_broadcast: Some(BroadcastStrategy::Tree { arity: 2 }),
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> DistributedResult<()> {
        if self.num_devices == 0 {
            return Err(DistributedError::Config("num_devices must be > 0".into()));
        }

        if let Some(n) = self.num_nodes {
            if n == 0 {
                return Err(DistributedError::Config("num_nodes must be > 0".into()));
            }

            // All global settings must be set together
            if self.global_all_reduce.is_none()
                || self.global_reduce.is_none()
                || self.global_broadcast.is_none()
            {
                return Err(DistributedError::Config(
                    "All global strategies must be set for multi-node".into(),
                ));
            }
        }

        if self.tree_arity < 2 {
            return Err(DistributedError::Config("tree_arity must be >= 2".into()));
        }

        Ok(())
    }

    /// Select the best all-reduce strategy for a given buffer size.
    pub fn select_all_reduce(&self, buffer_size: usize, world_size: usize) -> AllReduceStrategy {
        match self.local_all_reduce {
            AllReduceStrategy::Auto => {
                if buffer_size < self.tree_threshold_bytes || world_size < 4 {
                    AllReduceStrategy::Tree {
                        arity: self.tree_arity,
                    }
                } else {
                    AllReduceStrategy::Ring
                }
            }
            other => other,
        }
    }
}

/// Trait for collective operation implementations.
pub trait CollectiveOps: Send + Sync {
    /// Perform all-reduce with the configured strategy.
    fn all_reduce(
        &self,
        buffer: &mut [f32],
        strategy: AllReduceStrategy,
    ) -> impl std::future::Future<Output = DistributedResult<()>> + Send;

    /// Perform reduce to root with the configured strategy.
    fn reduce(
        &self,
        buffer: &mut [f32],
        root: usize,
        strategy: ReduceStrategy,
    ) -> impl std::future::Future<Output = DistributedResult<()>> + Send;

    /// Perform broadcast from root with the configured strategy.
    fn broadcast(
        &self,
        buffer: &mut [f32],
        root: usize,
        strategy: BroadcastStrategy,
    ) -> impl std::future::Future<Output = DistributedResult<()>> + Send;
}

/// Ring all-reduce implementation.
pub mod ring {
    use super::*;

    /// Perform ring all-reduce (scatter-reduce + all-gather).
    ///
    /// This is bandwidth-optimal for large tensors:
    /// - Total data transferred per node: 2 * (n-1) / n * buffer_size
    /// - Number of steps: 2 * (n - 1)
    pub async fn all_reduce<S, R>(
        buffer: &mut [f32],
        rank: usize,
        world_size: usize,
        send: &S,
        recv: &R,
    ) -> DistributedResult<()>
    where
        S: Fn(
                &[u8],
            )
                -> std::pin::Pin<Box<dyn std::future::Future<Output = DistributedResult<()>> + Send>>
            + Send
            + Sync,
        R: Fn(
                &mut [u8],
            )
                -> std::pin::Pin<Box<dyn std::future::Future<Output = DistributedResult<()>> + Send>>
            + Send
            + Sync,
    {
        if world_size < 2 {
            return Ok(());
        }

        let len = buffer.len();
        let chunk_size = len / world_size;
        let remainder = len % world_size;

        // Helper to get chunk range
        let get_chunk_range = |idx: usize| -> (usize, usize) {
            let start = idx * chunk_size + idx.min(remainder);
            let end = start + chunk_size + if idx < remainder { 1 } else { 0 };
            (start, end)
        };

        // === SCATTER-REDUCE PHASE ===
        for step in 0..(world_size - 1) {
            let send_idx = (rank + world_size - step) % world_size;
            let recv_idx = (rank + world_size - step - 1) % world_size;

            let (send_start, send_end) = get_chunk_range(send_idx);
            let (recv_start, recv_end) = get_chunk_range(recv_idx);

            // Prepare send buffer
            let send_bytes: Vec<u8> = buffer[send_start..send_end]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();

            let recv_len = (recv_end - recv_start) * 4;
            let mut recv_bytes = vec![0u8; recv_len];

            // Send and receive concurrently
            tokio::try_join!(send(&send_bytes), recv(&mut recv_bytes))?;

            // Reduce received data
            for (i, chunk) in recv_bytes.chunks_exact(4).enumerate() {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                buffer[recv_start + i] += val;
            }
        }

        // === ALL-GATHER PHASE ===
        for step in 0..(world_size - 1) {
            let send_idx = (rank + world_size - step + 1) % world_size;
            let recv_idx = (rank + world_size - step) % world_size;

            let (send_start, send_end) = get_chunk_range(send_idx);
            let (recv_start, recv_end) = get_chunk_range(recv_idx);

            // Prepare send buffer
            let send_bytes: Vec<u8> = buffer[send_start..send_end]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();

            let recv_len = (recv_end - recv_start) * 4;
            let mut recv_bytes = vec![0u8; recv_len];

            // Send and receive concurrently
            tokio::try_join!(send(&send_bytes), recv(&mut recv_bytes))?;

            // Copy received data
            for (i, chunk) in recv_bytes.chunks_exact(4).enumerate() {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                buffer[recv_start + i] = val;
            }
        }

        debug!("Ring all-reduce complete: {} elements", len);
        Ok(())
    }
}

/// Tree all-reduce implementation.
pub mod tree {

    /// Tree node role for a given phase.
    #[derive(Debug, Clone, Copy)]
    pub enum TreeRole {
        /// Leaf node in this phase.
        Leaf,
        /// Internal node with children.
        Internal { num_children: usize },
        /// Root node.
        Root { num_children: usize },
    }

    /// Compute tree role for a node in a k-ary tree.
    pub fn compute_role(rank: usize, world_size: usize, arity: usize) -> TreeRole {
        if rank == 0 {
            // Root
            let num_children = arity.min(world_size - 1);
            TreeRole::Root { num_children }
        } else {
            // Check if this node has children
            let first_child = rank * arity + 1;
            if first_child < world_size {
                let num_children = (world_size - first_child).min(arity);
                TreeRole::Internal { num_children }
            } else {
                TreeRole::Leaf
            }
        }
    }

    /// Get parent rank in a k-ary tree.
    pub fn parent_rank(rank: usize, _arity: usize) -> Option<usize> {
        if rank == 0 {
            None
        } else {
            Some((rank - 1) / _arity)
        }
    }

    /// Get child ranks in a k-ary tree.
    pub fn child_ranks(rank: usize, world_size: usize, arity: usize) -> Vec<usize> {
        let first_child = rank * arity + 1;
        (first_child..first_child + arity)
            .filter(|&c| c < world_size)
            .collect()
    }
}

/// Centralized all-reduce implementation.
pub mod centralized {
    use super::*;

    /// Perform centralized all-reduce (reduce to root + broadcast).
    ///
    /// Simple but not bandwidth-optimal:
    /// - Root receives from all, reduces, broadcasts to all
    /// - O(n) messages, O(n) bandwidth on root
    #[allow(clippy::too_many_arguments)]
    pub async fn all_reduce<S, R>(
        buffer: &mut [f32],
        _rank: usize,
        world_size: usize,
        is_root: bool,
        send_to_root: &S,
        recv_from_root: &R,
        recv_from_peer: &R,
        send_to_peer: &S,
    ) -> DistributedResult<()>
    where
        S: Fn(
                &[u8],
            )
                -> std::pin::Pin<Box<dyn std::future::Future<Output = DistributedResult<()>> + Send>>
            + Send
            + Sync,
        R: Fn(
                &mut [u8],
            )
                -> std::pin::Pin<Box<dyn std::future::Future<Output = DistributedResult<()>> + Send>>
            + Send
            + Sync,
    {
        if world_size < 2 {
            return Ok(());
        }

        let len = buffer.len();
        let byte_len = len * 4;

        if is_root {
            // === REDUCE PHASE ===
            // Receive from all peers and accumulate
            let mut recv_buf = vec![0u8; byte_len];

            for _ in 1..world_size {
                recv_from_peer(&mut recv_buf).await?;

                // Accumulate
                for (i, chunk) in recv_buf.chunks_exact(4).enumerate() {
                    let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    buffer[i] += val;
                }
            }

            // === BROADCAST PHASE ===
            // Send result to all peers
            let send_bytes: Vec<u8> = buffer.iter().flat_map(|f| f.to_le_bytes()).collect();

            for _ in 1..world_size {
                send_to_peer(&send_bytes).await?;
            }
        } else {
            // === REDUCE PHASE ===
            // Send to root
            let send_bytes: Vec<u8> = buffer.iter().flat_map(|f| f.to_le_bytes()).collect();
            send_to_root(&send_bytes).await?;

            // === BROADCAST PHASE ===
            // Receive from root
            let mut recv_buf = vec![0u8; byte_len];
            recv_from_root(&mut recv_buf).await?;

            // Copy result
            for (i, chunk) in recv_buf.chunks_exact(4).enumerate() {
                buffer[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }

        debug!("Centralized all-reduce complete: {} elements", len);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let mut config = CollectiveConfig::default();
        assert!(config.validate().is_ok());

        config.num_devices = 0;
        assert!(config.validate().is_err());

        config.num_devices = 1;
        config.num_nodes = Some(2);
        assert!(config.validate().is_err()); // Missing global strategies

        config.global_all_reduce = Some(AllReduceStrategy::Ring);
        config.global_reduce = Some(ReduceStrategy::Tree { arity: 2 });
        config.global_broadcast = Some(BroadcastStrategy::Tree { arity: 2 });
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_strategy_selection() {
        let config = CollectiveConfig {
            tree_threshold_bytes: 1024,
            tree_arity: 2,
            local_all_reduce: AllReduceStrategy::Auto,
            ..Default::default()
        };

        // Small buffer -> tree
        let strategy = config.select_all_reduce(512, 4);
        assert!(matches!(strategy, AllReduceStrategy::Tree { .. }));

        // Large buffer -> ring
        let strategy = config.select_all_reduce(2048, 4);
        assert!(matches!(strategy, AllReduceStrategy::Ring));

        // Small world size -> tree
        let strategy = config.select_all_reduce(2048, 2);
        assert!(matches!(strategy, AllReduceStrategy::Tree { .. }));
    }

    #[test]
    fn test_tree_roles() {
        // 2-ary tree with 7 nodes
        //       0
        //      / \
        //     1   2
        //    / \ / \
        //   3  4 5  6

        let world_size = 7;
        let arity = 2;

        assert!(matches!(
            tree::compute_role(0, world_size, arity),
            tree::TreeRole::Root { num_children: 2 }
        ));
        assert!(matches!(
            tree::compute_role(1, world_size, arity),
            tree::TreeRole::Internal { num_children: 2 }
        ));
        assert!(matches!(
            tree::compute_role(3, world_size, arity),
            tree::TreeRole::Leaf
        ));

        assert_eq!(tree::parent_rank(3, arity), Some(1));
        assert_eq!(tree::parent_rank(1, arity), Some(0));
        assert_eq!(tree::parent_rank(0, arity), None);

        assert_eq!(tree::child_ranks(0, world_size, arity), vec![1, 2]);
        assert_eq!(tree::child_ranks(1, world_size, arity), vec![3, 4]);
    }
}
