//! Distributed training backend for PMetal.
//!
//! Enables "Home Clusters" by synchronizing gradients across multiple devices
//! (e.g., Mac Studio + MacBook Pro) over standard networks (TCP/IP, Wi-Fi).
//!
//! # Features
//!
//! - **Zero-Configuration Discovery**: Automatically finds peers using mDNS/Bonjour
//! - **Ring All-Reduce**: Bandwidth-optimal gradient synchronization
//! - **Persistent Identity**: Ed25519 keypairs stored at `~/.pmetal/node_keypair`
//! - **Topology Awareness**: Graph-based cluster management with petgraph
//! - **Master Election**: Distributed leader election for coordination
//! - **Health Monitoring**: Heartbeat-based peer health tracking
//! - **Gradient Compression**: TopK, quantization, and error feedback
//! - **Network Isolation**: PSK-based namespace isolation
//! - **Observability**: Comprehensive metrics and tracing
//!
//! # Quick Start (Auto-Discovery)
//!
//! ```ignore
//! use pmetal_distributed::{AutoDiscoveryBackend, DistributedContext};
//! use std::time::Duration;
//!
//! // Create backend with automatic peer discovery
//! let backend = AutoDiscoveryBackend::new().await?;
//!
//! // Wait for at least 1 peer to join
//! backend.wait_for_peers(1, Duration::from_secs(30)).await?;
//!
//! // Create context for distributed operations
//! let ctx = DistributedContext::new(Box::new(backend));
//!
//! // Synchronize gradients across cluster
//! ctx.all_reduce(&mut gradient_buffer).await?;
//! ```
//!
//! # Manual Configuration
//!
//! For advanced use cases, you can manually configure peers:
//!
//! ```ignore
//! use pmetal_distributed::{DistributedConfig, RingBackend, DistributedContext};
//!
//! let config = DistributedConfig::new(
//!     vec!["192.168.1.10:52416".parse()?, "192.168.1.11:52416".parse()?],
//!     0, // This node's rank
//! );
//!
//! let backend = RingBackend::new(config).await?;
//! let ctx = DistributedContext::new(Box::new(backend));
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     AutoDiscoveryBackend                         │
//! │                                                                  │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
//! │  │   Identity   │  │  Discovery   │  │  Topology    │           │
//! │  │  (Ed25519)   │  │   (mDNS)     │  │  (petgraph)  │           │
//! │  └──────────────┘  └──────────────┘  └──────────────┘           │
//! │          │                │                 │                    │
//! │          └────────────────┼─────────────────┘                    │
//! │                           ▼                                      │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
//! │  │  Election    │  │   Health     │  │  Collective  │           │
//! │  │  (Master)    │  │  (Heartbeat) │  │  (Strategies)│           │
//! │  └──────────────┘  └──────────────┘  └──────────────┘           │
//! │          │                │                 │                    │
//! │          └────────────────┼─────────────────┘                    │
//! │                           ▼                                      │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
//! │  │ Compression  │  │   Metrics    │  │  Namespace   │           │
//! │  │  (TopK/Quant)│  │ (Observ.)    │  │  (PSK)       │           │
//! │  └──────────────┘  └──────────────┘  └──────────────┘           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use anyhow::Result;
use async_trait::async_trait;

// Core modules
pub mod auto;
pub mod config;
pub mod discovery;
pub mod error;
pub mod identity;
pub mod ring;
pub mod topology;
pub mod transport;

// Advanced modules
pub mod collective;
pub mod compression;
pub mod election;
pub mod health;
pub mod metrics;
pub mod namespace;

// Re-exports for convenience
pub use auto::{AutoDiscoveryBackend, AutoDiscoveryConfig};
pub use collective::{AllReduceStrategy, BroadcastStrategy, CollectiveConfig, ReduceStrategy};
pub use compression::{CompressionStrategy, GradientCompressor, QuantizationType};
pub use config::DistributedConfig;
pub use election::{ElectionConfig, ElectionEvent, ElectionManager, ElectionState};
pub use error::{DistributedError, DistributedResult};
pub use health::{HealthConfig, HealthEvent, HealthMonitor, HealthStatus, HealthSummary};
pub use identity::NodeIdentity;
pub use metrics::{DistributedMetrics, MetricsSnapshot, SharedMetrics};
pub use namespace::NetworkNamespace;
pub use ring::RingBackend;
pub use topology::{ClusterTopology, ConnectionProfile, NodeProfile, SharedTopology};

/// Interface for distributed operations.
#[async_trait]
pub trait DistributedBackend: Send + Sync {
    /// Get the rank of this node (0 to world_size - 1).
    fn rank(&self) -> usize;

    /// Get the total number of nodes.
    fn world_size(&self) -> usize;

    /// Perform an all-reduce operation on a buffer (sum).
    ///
    /// The input buffer contains the local gradients.
    /// On return, it should contain the sum of gradients from all nodes.
    async fn all_reduce(&self, buffer: &mut [u8]) -> Result<()>;

    /// Barrier synchronization.
    async fn barrier(&self) -> Result<()>;
}

/// A handle to the distributed runtime.
pub struct DistributedContext {
    backend: Box<dyn DistributedBackend>,
    metrics: Option<SharedMetrics>,
}

impl DistributedContext {
    /// Create a new distributed context with the given backend.
    pub fn new(backend: Box<dyn DistributedBackend>) -> Self {
        Self {
            backend,
            metrics: None,
        }
    }

    /// Create a new distributed context with metrics enabled.
    pub fn with_metrics(backend: Box<dyn DistributedBackend>, metrics: SharedMetrics) -> Self {
        Self {
            backend,
            metrics: Some(metrics),
        }
    }

    /// Get the rank of this node.
    pub fn rank(&self) -> usize {
        self.backend.rank()
    }

    /// Get the total number of nodes in the cluster.
    pub fn world_size(&self) -> usize {
        self.backend.world_size()
    }

    /// Perform an all-reduce operation (sum) on the buffer.
    ///
    /// After this call, all nodes will have the same values in their buffers,
    /// which is the element-wise sum of all input buffers.
    pub async fn all_reduce(&self, buffer: &mut [u8]) -> Result<()> {
        let start = std::time::Instant::now();
        let result = self.backend.all_reduce(buffer).await;

        if let Some(ref metrics) = self.metrics {
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics.all_reduce.duration_ms.observe(duration_ms);
            metrics.all_reduce.bytes_processed.add(buffer.len() as u64);

            if result.is_ok() {
                metrics.all_reduce.completed.inc();
            } else {
                metrics.all_reduce.failed.inc();
            }
        }

        result
    }

    /// Synchronize all nodes at a barrier.
    ///
    /// All nodes must call this method, and none will proceed until all have.
    pub async fn barrier(&self) -> Result<()> {
        let start = std::time::Instant::now();
        let result = self.backend.barrier().await;

        if let Some(ref metrics) = self.metrics {
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics.barrier.duration_ms.observe(duration_ms);

            if result.is_ok() {
                metrics.barrier.completed.inc();
            } else {
                metrics.barrier.failed.inc();
            }
        }

        result
    }

    /// Check if this is the master node (rank 0).
    pub fn is_master(&self) -> bool {
        self.rank() == 0
    }

    /// Get metrics snapshot if enabled.
    pub fn metrics_snapshot(&self) -> Option<MetricsSnapshot> {
        self.metrics.as_ref().map(|m| m.snapshot())
    }
}

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::auto::{AutoDiscoveryBackend, AutoDiscoveryConfig};
    pub use crate::collective::{AllReduceStrategy, CollectiveConfig};
    pub use crate::compression::{CompressionStrategy, GradientCompressor};
    pub use crate::config::DistributedConfig;
    pub use crate::election::{ElectionConfig, ElectionManager};
    pub use crate::error::{DistributedError, DistributedResult};
    pub use crate::health::{HealthConfig, HealthMonitor, HealthStatus};
    pub use crate::identity::NodeIdentity;
    pub use crate::metrics::{DistributedMetrics, SharedMetrics};
    pub use crate::namespace::NetworkNamespace;
    pub use crate::ring::RingBackend;
    pub use crate::topology::{ClusterTopology, NodeProfile};
    pub use crate::DistributedBackend;
    pub use crate::DistributedContext;
}
