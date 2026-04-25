//! Distributed training and inference backend for PMetal.
//!
//! Enables "Home Clusters" by synchronizing gradients, sharding models,
//! and distributing inference across multiple Apple Silicon devices
//! over standard networks (TCP/IP, Thunderbolt/RDMA).
//!
//! # Core Features
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
//! # Advanced Features (feature-gated)
//!
//! - **`mlx-collectives`**: MLX distributed C API wrappers (JACCL/RDMA over Thunderbolt 5)
//! - **`tensor-parallel`**: Weight sharding, distributed linear layers (AllToSharded/ShardedToAll)
//! - **`expert-parallel`**: MoE expert dispatch across nodes (all-to-all communication)
//! - **`context-parallel`**: Ring attention, KV cache sharding for long-context inference
//! - **`zero`**: ZeRO optimizer state/gradient sharding for memory-efficient training
//! - **`pipeline-training`**: 1F1B and zero-bubble pipeline training schedules
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

/// Reduction operation for `all_reduce`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum all contributions across nodes.
    Sum,
    /// Average all contributions across nodes (sum divided by `world_size`).
    Mean,
}

// Core modules
pub mod auto;
pub mod cloud_bridge;
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

// Canonical expert-ID ↔ rank mapping (pure Rust, no MLX dependency).
// Shared by tensor_parallel weight sharding, expert_parallel placement,
// and ZeRO MoE partitioning to enforce a single invariant.
pub mod expert_shard;

// Pipeline inference modules
pub mod activation_codec;
pub mod activation_transport;
pub mod layer_assignment;
pub mod pipeline;
pub mod solver;
pub mod ultrafusion;

// ─── Feature-gated modules ──────────────────────────────────────────────────

/// MLX distributed C API wrappers (JACCL/Ring/RDMA backends).
#[cfg(feature = "mlx-collectives")]
pub mod mlx_dist;

/// Tensor parallelism: weight sharding and distributed linear layers.
#[cfg(feature = "tensor-parallel")]
pub mod tensor_parallel;

/// Expert parallelism: MoE expert dispatch across nodes.
#[cfg(feature = "expert-parallel")]
pub mod expert_parallel;

/// Context parallelism: ring attention and KV cache sharding.
#[cfg(feature = "context-parallel")]
pub mod context_parallel;

/// ZeRO optimizer state/gradient sharding (pure Rust, no MLX dependency).
#[cfg(feature = "zero")]
pub mod zero;

/// Pipeline training schedules: 1F1B and zero-bubble (pure Rust).
#[cfg(feature = "pipeline-training")]
pub mod pipeline_training;

// Re-exports for convenience
pub use activation_codec::ActivationCodec;
pub use activation_transport::{ActivationMessage, DtypeTag};
pub use auto::{AutoDiscoveryBackend, AutoDiscoveryConfig};
pub use collective::{AllReduceStrategy, BroadcastStrategy, CollectiveConfig, ReduceStrategy};
pub use compression::{CompressionStrategy, GradientCompressor, QuantizationType};
pub use config::DistributedConfig;
pub use election::{ElectionConfig, ElectionEvent, ElectionManager, ElectionState};
pub use error::{DistributedError, DistributedResult};
pub use health::{HealthConfig, HealthEvent, HealthMonitor, HealthStatus, HealthSummary};
pub use identity::NodeIdentity;
pub use layer_assignment::{assign_layers_bandwidth_aware, assign_layers_proportional};
pub use metrics::{DistributedMetrics, MetricsSnapshot, SharedMetrics};
pub use namespace::NetworkNamespace;
pub use pipeline::{
    PipelineGenerationLoop, PipelineStageConfig, PipelineStageRuntime, StreamMultiplexer,
};
pub use ring::RingBackend;
pub use topology::{ClusterTopology, ConnectionProfile, NodeProfile, SharedTopology};
pub use ultrafusion::{
    DEFAULT_LOCAL_CHANNEL_CAPACITY, DEFAULT_ULTRAFUSION_INTERCONNECT_BYTES_PER_SEC,
    UltraFusionDieProfile, UltraFusionExecutionConfig, UltraFusionPlan, UltraFusionStagePlan,
};
// ReduceOp is already public via `pub enum ReduceOp` at module level

// ─── Feature-gated re-exports ───────────────────────────────────────────────

#[cfg(feature = "mlx-collectives")]
pub use mlx_dist::{DistributedGroup, MlxDistributedBackend};

#[cfg(feature = "tensor-parallel")]
pub use tensor_parallel::{ShardingDirective, ShardingPlan, shard_weight};

#[cfg(feature = "expert-parallel")]
pub use expert_parallel::{CapacityConfig, ExpertDispatcher, ExpertPlacement};

#[cfg(feature = "context-parallel")]
pub use context_parallel::{CPMode, ring_attention_forward};

#[cfg(feature = "zero")]
pub use zero::{ZeROPartitioner, ZeROStage};

#[cfg(feature = "pipeline-training")]
pub use pipeline_training::{MicroBatchAction, ZBAction, schedule_1f1b, schedule_zero_bubble};

/// Interface for distributed operations.
#[async_trait]
pub trait DistributedBackend: Send + Sync {
    /// Get the rank of this node (0 to world_size - 1).
    fn rank(&self) -> usize;

    /// Get the total number of nodes.
    fn world_size(&self) -> usize;

    /// Perform an all-reduce operation on a buffer.
    ///
    /// The input buffer contains the local gradients encoded as little-endian
    /// `f32` values.  On return, all nodes hold the same result:
    /// - `ReduceOp::Sum`  – element-wise sum across all nodes.
    /// - `ReduceOp::Mean` – element-wise sum divided by `world_size`.
    async fn all_reduce(&self, buffer: &mut [u8], op: ReduceOp) -> Result<()>;

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

    /// Perform an all-reduce operation on the buffer.
    ///
    /// After this call, all nodes will have the same values in their buffers.
    /// `op` controls whether the result is a sum or mean across nodes.
    pub async fn all_reduce(&self, buffer: &mut [u8], op: ReduceOp) -> Result<()> {
        let start = std::time::Instant::now();
        let result = self.backend.all_reduce(buffer, op).await;

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
    pub use crate::DistributedBackend;
    pub use crate::DistributedContext;
    pub use crate::ReduceOp;
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
    pub use crate::ultrafusion::{
        UltraFusionDieProfile, UltraFusionExecutionConfig, UltraFusionPlan, UltraFusionStagePlan,
    };

    #[cfg(feature = "mlx-collectives")]
    pub use crate::mlx_dist::{DistributedGroup, MlxDistributedBackend};

    #[cfg(feature = "tensor-parallel")]
    pub use crate::tensor_parallel::{ShardingDirective, ShardingPlan};

    #[cfg(feature = "expert-parallel")]
    pub use crate::expert_parallel::{ExpertDispatcher, ExpertPlacement};

    #[cfg(feature = "context-parallel")]
    pub use crate::context_parallel::CPMode;

    #[cfg(feature = "zero")]
    pub use crate::zero::{ZeROPartitioner, ZeROStage};

    #[cfg(feature = "pipeline-training")]
    pub use crate::pipeline_training::{MicroBatchAction, ZBAction};
}
