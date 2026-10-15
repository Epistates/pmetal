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
//! │  │ Compression  │  │   Metrics    │  │  Namespace   │           │
//! │  │  (TopK/Quant)│  │ (Observ.)    │  │  (PSK)       │           │
//! │  └──────────────┘  └──────────────┘  └──────────────┘           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use anyhow::Result;
use async_trait::async_trait;
use std::ops::Range;

/// Reduction operation for `all_reduce`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum all contributions across nodes.
    Sum,
    /// Average all contributions across nodes (sum divided by `world_size`).
    Mean,
}

// Core modules
pub mod apportion;
pub mod auto;
pub mod cluster_runtime;
pub mod collective;
pub mod config;
pub mod discovery;
pub mod error;
pub mod fabric;
pub mod identity;
pub mod ring;
pub mod secret;
pub mod topology;
pub mod transport;

// Advanced modules
pub mod compression;
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
pub mod pipeline_harness;
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

#[cfg(test)]
mod testing;

// Re-exports for convenience
pub use activation_codec::ActivationCodec;
pub use activation_transport::{ActivationMessage, DtypeTag};
pub use auto::{AutoDiscoveryBackend, AutoDiscoveryConfig};
pub use compression::{CompressionStrategy, GradientCompressor, QuantizationType};
pub use config::DistributedConfig;
pub use error::{DistributedError, DistributedResult};
pub use fabric::{
    InterfaceInfo, InterfaceKind, LinkScore, LocalFabric, is_link_local_ipv4, nominal_score,
    probe_local_fabric, score_link,
};
pub use identity::NodeIdentity;
pub use layer_assignment::{
    LayerProblem, StageNode, assign_layers, assign_layers_proportional, best_ring_order,
};
pub use metrics::{DistributedMetrics, MetricsSnapshot, SharedMetrics};
pub use namespace::NetworkNamespace;
pub use pipeline::{
    PipelineGenerationLoop, PipelineStageConfig, PipelineStageRuntime, StreamMultiplexer,
};
pub use ring::RingBackend;
pub use secret::ClusterSecret;
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
pub use zero::{ShardOptimizer, ShardedAdamW, ZeROStage, ZeroLayout, ZeroOptimizer};

#[cfg(feature = "pipeline-training")]
pub use pipeline_training::{MicroBatchAction, ZBAction, schedule_1f1b, schedule_zero_bubble};

/// Interface for distributed operations.
#[async_trait]
pub trait DistributedBackend: Send + Sync {
    /// Get the rank of this node (0 to world_size - 1).
    fn rank(&self) -> usize;

    /// Get the total number of nodes.
    fn world_size(&self) -> usize;

    /// Reduce `data` across all nodes, in place.
    ///
    /// On return every node holds the same result, bit for bit:
    /// - `ReduceOp::Sum`  – element-wise sum across all nodes.
    /// - `ReduceOp::Mean` – element-wise sum divided by `world_size`.
    ///
    /// Every node must call it with a buffer of the same length.
    async fn all_reduce_f32(&self, data: &mut [f32], op: ReduceOp) -> Result<()>;

    /// [`all_reduce_f32`](Self::all_reduce_f32) on a buffer of little-endian
    /// `f32` bytes.
    ///
    /// A `Vec<u8>` carries no alignment guarantee, so a buffer that cannot be
    /// viewed as `f32` in place is decoded to a temporary and written back.
    /// Prefer `all_reduce_f32`.
    async fn all_reduce(&self, buffer: &mut [u8], op: ReduceOp) -> Result<()> {
        if !buffer.len().is_multiple_of(4) {
            return Err(DistributedError::Protocol(format!(
                "Buffer length {} is not a multiple of 4 (f32 size)",
                buffer.len()
            ))
            .into());
        }
        if let Some(data) = collective::bytes_as_f32_mut(buffer) {
            return self.all_reduce_f32(data, op).await;
        }
        let mut data: Vec<f32> = buffer
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        self.all_reduce_f32(&mut data, op).await?;
        for (dst, x) in buffer.as_chunks_mut::<4>().0.iter_mut().zip(&data) {
            *dst = x.to_le_bytes();
        }
        Ok(())
    }

    /// Sum (or average) `data` across nodes, leaving this node with only its
    /// own chunk (`collective::chunk_range(len, world_size, rank)`) reduced;
    /// the rest of `data` is then meaningless. Returns the reduced range.
    /// Half the traffic of an all-reduce, for sharded optimizers.
    async fn reduce_scatter_f32(&self, _data: &mut [f32], _op: ReduceOp) -> Result<Range<usize>> {
        Err(unsupported("reduce_scatter_f32"))
    }

    /// Complete `data` on every node, given that each node's own chunk is
    /// final (the other half of [`reduce_scatter_f32`](Self::reduce_scatter_f32)).
    async fn all_gather_f32_in_place(&self, _data: &mut [f32]) -> Result<()> {
        Err(unsupported("all_gather_f32_in_place"))
    }

    /// Collect every node's `local` bytes (lengths may differ): element `r` of
    /// the result is node `r`'s contribution.
    async fn all_gather_bytes(&self, _local: &[u8]) -> Result<Vec<Vec<u8>>> {
        Err(unsupported("all_gather_bytes"))
    }

    /// Copy `buffer` from node `root` to every node.
    async fn broadcast(&self, _buffer: &mut [u8], _root: usize) -> Result<()> {
        Err(unsupported("broadcast"))
    }

    /// Barrier synchronization.
    async fn barrier(&self) -> Result<()>;
}

fn unsupported(op: &str) -> anyhow::Error {
    DistributedError::Protocol(format!("this backend does not implement {op}")).into()
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

    /// Perform an all-reduce operation on a buffer of little-endian `f32`
    /// bytes.
    ///
    /// After this call, all nodes will have the same values in their buffers.
    /// `op` controls whether the result is a sum or mean across nodes. Prefer
    /// [`all_reduce_f32`](Self::all_reduce_f32), which needs no byte view.
    pub async fn all_reduce(&self, buffer: &mut [u8], op: ReduceOp) -> Result<()> {
        let bytes = buffer.len();
        let result = self.backend.all_reduce(buffer, op).await;
        self.observe(|m| &m.all_reduce, bytes, std::time::Instant::now(), &result);
        result
    }

    /// Reduce `data` (sum or mean) across all nodes, in place.
    pub async fn all_reduce_f32(&self, data: &mut [f32], op: ReduceOp) -> Result<()> {
        let start = std::time::Instant::now();
        let result = self.backend.all_reduce_f32(data, op).await;
        self.observe(|m| &m.all_reduce, data.len() * 4, start, &result);
        result
    }

    /// Reduce `data` across all nodes through `compressor`: the nodes
    /// exchange compressed gradients and every node sums what they decode to
    /// (see [`compression::all_reduce_compressed`]). The compressor's error
    /// feedback carries what was not sent into the next call, so give each
    /// buffer its own compressor and call it once per step.
    pub async fn all_reduce_compressed(
        &self,
        compressor: &mut GradientCompressor,
        data: &mut [f32],
        op: ReduceOp,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        let result =
            compression::all_reduce_compressed(self.backend.as_ref(), compressor, data, op).await;
        let sent = result.as_ref().map_or(0, |sent| *sent);
        if let Some(ref metrics) = self.metrics {
            metrics.compression.bytes_before.add(data.len() as u64 * 4);
            metrics.compression.bytes_after.add(sent as u64);
        }
        self.observe(|m| &m.all_reduce, sent, start, &result);
        result.map(|_| ())
    }

    /// Reduce `data` across all nodes, leaving this node with only its own
    /// chunk reduced; returns that chunk's range.
    pub async fn reduce_scatter_f32(&self, data: &mut [f32], op: ReduceOp) -> Result<Range<usize>> {
        let start = std::time::Instant::now();
        let result = self.backend.reduce_scatter_f32(data, op).await;
        self.observe(|m| &m.reduce, data.len() * 4, start, &result);
        result
    }

    /// Complete `data` on every node from each node's own final chunk.
    pub async fn all_gather_f32_in_place(&self, data: &mut [f32]) -> Result<()> {
        let start = std::time::Instant::now();
        let result = self.backend.all_gather_f32_in_place(data).await;
        self.observe(|m| &m.reduce, data.len() * 4, start, &result);
        result
    }

    /// Collect every node's `local` bytes; element `r` is node `r`'s.
    pub async fn all_gather_bytes(&self, local: &[u8]) -> Result<Vec<Vec<u8>>> {
        let start = std::time::Instant::now();
        let result = self.backend.all_gather_bytes(local).await;
        self.observe(|m| &m.reduce, local.len(), start, &result);
        result
    }

    /// Copy `buffer` from node `root` to every node.
    pub async fn broadcast(&self, buffer: &mut [u8], root: usize) -> Result<()> {
        let bytes = buffer.len();
        let start = std::time::Instant::now();
        let result = self.backend.broadcast(buffer, root).await;
        self.observe(|m| &m.broadcast, bytes, start, &result);
        result
    }

    /// Synchronize all nodes at a barrier.
    ///
    /// All nodes must call this method, and none will proceed until all have.
    pub async fn barrier(&self) -> Result<()> {
        let start = std::time::Instant::now();
        let result = self.backend.barrier().await;
        self.observe(|m| &m.barrier, 0, start, &result);
        result
    }

    fn observe<T>(
        &self,
        select: impl Fn(&DistributedMetrics) -> &metrics::OperationMetrics,
        bytes: usize,
        start: std::time::Instant,
        result: &Result<T>,
    ) {
        if let Some(ref metrics) = self.metrics {
            let op = select(metrics);
            op.duration_ms
                .observe(start.elapsed().as_secs_f64() * 1000.0);
            op.bytes_processed.add(bytes as u64);
            if result.is_ok() {
                op.completed.inc();
            } else {
                op.failed.inc();
            }
        }
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
    pub use crate::compression::{CompressionStrategy, GradientCompressor};
    pub use crate::config::DistributedConfig;
    pub use crate::error::{DistributedError, DistributedResult};
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
    pub use crate::zero::{ZeROStage, ZeroLayout, ZeroOptimizer};

    #[cfg(feature = "pipeline-training")]
    pub use crate::pipeline_training::{MicroBatchAction, ZBAction};
}

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
