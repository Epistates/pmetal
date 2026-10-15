//! Pipeline-parallel inference harness.
//!
//! Wires together `auto::AutoDiscoveryBackend` (for topology + identity) +
//! `transport::TcpTransport` (for fabric-aware ring connections) +
//! `pipeline::PipelineStageRuntime` (for activation send/recv) into a
//! single high-level handle that callers can drive a model through.
//!
//! Topology of TCP connections in pipeline mode:
//!
//! ```text
//!   ┌──── activations ───┐  ┌──── activations ────┐
//!   │ rank 0 ─→ rank 1 ─→ rank 2 ─→ … ─→ rank N-1 │
//!   └──── result tokens ─────────────────────────┘
//! ```
//!
//! Each rank opens **two** TCP listeners during ring formation:
//!   1. A "next" listener on `activation_port` — accepts the prev-rank's
//!      sender. The prev rank dials this to forward activations.
//!   2. (rank 0 only) A "result" listener on `result_port` — accepts the
//!      last rank's result sender, used to ship sampled tokens back.
//!
//! The harness owns the connection setup; callers receive a fully-wired
//! [`crate::pipeline::PipelineStageRuntime`] ready to use with
//! [`crate::pipeline::PipelineGenerationLoop`].

use crate::activation_codec::ActivationCodec;
use crate::activation_transport::DtypeTag;
use crate::auto::AutoDiscoveryBackend;
use crate::config::DistributedConfig;
use crate::error::{DistributedError, DistributedResult};
use crate::pipeline::{PipelineStageConfig, PipelineStageRuntime, solve_layer_assignment};
use crate::secret::ClusterSecret;
use crate::topology::NodeProfile;
use crate::transport::{
    Channel, LinkParams, LinkSecurity, RingMembership, TcpTransport, accept_authenticated,
};
use std::net::SocketAddr;
use std::ops::Range;
use std::sync::Arc;
use tracing::info;

/// Configuration for [`PipelineHarness::start`].
#[derive(Debug, Clone)]
pub struct PipelineHarnessConfig {
    /// Total number of layers in the model. Used by the layer-assignment
    /// solver to give each rank a contiguous chunk.
    pub num_layers: usize,
    /// TCP port for activation transport (next ↔ prev). Independent of the
    /// gradient port used by all-reduce — the two can run side by side.
    ///
    /// **Semantics:** consulted only by [`PipelineHarness::start`] when
    /// building peer addresses from the topology. [`PipelineHarness::start_explicit`]
    /// reads the bind port from `peer_addrs[local_rank][0].port()` directly —
    /// in that path this field is unused.
    pub activation_port: u16,
    /// TCP port the **first rank** listens on for sampled tokens coming
    /// back from the last rank. Ignored for non-rank-0 ranks.
    pub result_port: u16,
    /// Wire dtype for activation transfer (Float16 is the typical choice
    /// — halves bandwidth with negligible accuracy impact for inference).
    pub wire_dtype: DtypeTag,
    /// Activation compression codec.
    pub codec: ActivationCodec,
    /// TCP connection timeout (ms).
    pub connection_timeout_ms: u64,
    /// Maximum connection retries.
    pub max_retries: u32,
    /// The cluster secret that authenticates and encrypts the links. Left
    /// unset it is looked up as [`ClusterSecret::resolve`] does.
    pub secret: Option<ClusterSecret>,
    /// Run the links unauthenticated and unencrypted when no secret exists.
    /// For local experiments only.
    pub allow_plaintext: bool,
}

impl Default for PipelineHarnessConfig {
    fn default() -> Self {
        Self {
            num_layers: 0,
            activation_port: 52417,
            result_port: 52418,
            wire_dtype: DtypeTag::Float16,
            codec: ActivationCodec::Float16,
            connection_timeout_ms: 30_000,
            max_retries: 50,
            secret: None,
            allow_plaintext: false,
        }
    }
}

/// Layer-range plan returned to the caller after [`PipelineHarness::start`].
/// Use [`Self::layer_range`] to drive your model's per-layer execution.
#[derive(Debug, Clone)]
pub struct PipelinePlan {
    /// All ranks' assignments, in ring order.
    pub stages: Vec<PipelineStageConfig>,
    /// This node's rank within the pipeline.
    pub local_rank: usize,
}

impl PipelinePlan {
    pub fn local_stage(&self) -> &PipelineStageConfig {
        &self.stages[self.local_rank]
    }
    pub fn layer_range(&self) -> Range<usize> {
        self.local_stage().layer_range.clone()
    }
    pub fn world_size(&self) -> usize {
        self.stages.len()
    }
}

/// High-level pipeline-parallel runtime.
pub struct PipelineHarness {
    pub plan: PipelinePlan,
    pub stage: PipelineStageRuntime,
}

impl PipelineHarness {
    /// Stand up a pipeline-parallel ring backed by an existing
    /// [`AutoDiscoveryBackend`]. The backend must already have its peers
    /// discovered (call `wait_for_peers` first); this method does **not**
    /// reuse the backend's all-reduce ring — it stands up a separate
    /// activation TCP ring on `activation_port`.
    pub async fn start(
        backend: Arc<AutoDiscoveryBackend>,
        cfg: PipelineHarnessConfig,
    ) -> DistributedResult<Self> {
        let (peer_addrs, profiles, local_rank, membership) = {
            let topo = backend.topology();
            let topo = topo.read();
            let order = topo.ring_order();
            let world_size = order.len();
            if world_size < 2 {
                return Err(DistributedError::Protocol(
                    "pipeline harness requires at least 2 ranks".into(),
                ));
            }
            let local_rank = topo.local_rank();
            // For activations we want every rank's IP at the dedicated
            // activation_port — fabric ranking is honoured because each
            // peer's `addrs` list is best-first.
            let peer_addrs: Vec<Vec<SocketAddr>> = order
                .iter()
                .map(|n| {
                    n.addrs
                        .iter()
                        .map(|(a, _)| SocketAddr::new(a.ip(), cfg.activation_port))
                        .collect()
                })
                .collect();
            let profiles: Vec<NodeProfile> = order.iter().map(|n| n.profile.clone()).collect();
            // Each machine reaches its peers at different, fabric-specific
            // addresses, so the ring is identified by who is in it.
            let ids: Vec<Vec<u8>> = order.iter().map(|n| n.peer_id.to_bytes()).collect();
            let membership = RingMembership::from_ids(local_rank, &ids);
            (peer_addrs, profiles, local_rank, membership)
        };

        Self::start_with_membership(peer_addrs, profiles, local_rank, cfg, Some(membership)).await
    }

    /// Stand up the pipeline ring directly from explicit peer addresses
    /// (no mDNS). Each `peer_addrs[r]` is a fabric-ranked list of socket
    /// addrs for rank `r`. Used by tests and the explicit-config CLI path.
    pub async fn start_explicit(
        peer_addrs: Vec<Vec<SocketAddr>>,
        profiles: Vec<NodeProfile>,
        local_rank: usize,
        cfg: PipelineHarnessConfig,
    ) -> DistributedResult<Self> {
        Self::start_with_membership(peer_addrs, profiles, local_rank, cfg, None).await
    }

    async fn start_with_membership(
        peer_addrs: Vec<Vec<SocketAddr>>,
        profiles: Vec<NodeProfile>,
        local_rank: usize,
        cfg: PipelineHarnessConfig,
        membership: Option<RingMembership>,
    ) -> DistributedResult<Self> {
        let world_size = peer_addrs.len();
        if world_size < 2 {
            return Err(DistributedError::Protocol(
                "pipeline harness requires at least 2 ranks".into(),
            ));
        }
        if profiles.len() != world_size {
            return Err(DistributedError::Protocol(format!(
                "profiles.len() ({}) != world_size ({})",
                profiles.len(),
                world_size
            )));
        }

        // Solve layer assignment from RAM profiles.
        let mut stages = solve_layer_assignment(cfg.num_layers, &profiles)?;
        for s in &mut stages {
            s.wire_dtype = cfg.wire_dtype;
            s.codec = cfg.codec;
        }
        let plan = PipelinePlan {
            stages: stages.clone(),
            local_rank,
        };
        info!(
            "Pipeline plan: rank {}/{}, layers {:?}",
            local_rank,
            world_size,
            plan.layer_range()
        );

        // Stand up the activation ring as a separate `RingBackend`-style TCP
        // ring on `activation_port`. We reuse `TcpTransport::connect` which
        // gives us the standard next-sender + prev-receiver pair, with our
        // fabric fallback machinery.
        let dist_cfg = DistributedConfig {
            nodes: peer_addrs
                .iter()
                .map(|e| {
                    e.first().copied().unwrap_or_else(|| {
                        "0.0.0.0:0".parse().expect("placeholder addr always valid")
                    })
                })
                .collect(),
            fallback_addrs: peer_addrs
                .iter()
                .map(|e| e.iter().skip(1).copied().collect())
                .collect(),
            rank: local_rank,
            connection_timeout_ms: cfg.connection_timeout_ms,
            max_retries: cfg.max_retries,
            secret: cfg.secret.clone(),
            allow_plaintext: cfg.allow_plaintext,
        };
        let security = LinkSecurity::from_config(&dist_cfg)?;
        let membership = membership.unwrap_or_else(|| RingMembership::from_config(&dist_cfg));
        let (next_sender, prev_receiver) = TcpTransport::connect_ring(
            &dist_cfg,
            security.clone(),
            membership.clone(),
            None,
            Channel::Activations,
        )
        .await
        .map_err(|e| DistributedError::Protocol(format!("activation ring connect: {e}")))?;

        // Now stand up the result loopback channel: last → first only.
        // We piggy-back on a second TCP transport pair, but in degenerate
        // form: rank 0 listens on result_port and accepts a single inbound
        // connection from rank world_size-1.
        let (result_sender, result_receiver) = if world_size == 1 {
            (None, None)
        } else if local_rank == 0 {
            // First rank: receive results from last rank.
            let result_listener_addr: SocketAddr = ([0, 0, 0, 0], cfg.result_port).into();
            let listener = tokio::net::TcpListener::bind(result_listener_addr)
                .await
                .map_err(|e| DistributedError::Protocol(format!("bind result_port: {e}")))?;
            let timeout = std::time::Duration::from_millis(cfg.connection_timeout_ms);
            let params = LinkParams {
                security: security.clone(),
                channel: Channel::Results,
                membership: membership.clone(),
                peer_rank: world_size - 1,
            };
            let link = tokio::time::timeout(timeout, accept_authenticated(&listener, &params))
                .await
                .map_err(|_| {
                    DistributedError::Protocol(
                        "timeout waiting for last-rank result connection".into(),
                    )
                })??;
            (None, Some(link.receiver))
        } else if local_rank == world_size - 1 {
            // Last rank: connect to first rank's result_port.
            // Use the rank-0 peer's primary IP.
            let first_ip = peer_addrs[0]
                .first()
                .copied()
                .ok_or_else(|| DistributedError::Protocol("no rank-0 address known".into()))?
                .ip();
            let result_addr: SocketAddr = SocketAddr::new(first_ip, cfg.result_port);
            let params = LinkParams {
                security: security.clone(),
                channel: Channel::Results,
                membership: membership.clone(),
                peer_rank: 0,
            };
            // Retry until rank 0's listener is up.
            let link =
                TcpTransport::connect_with_fallback(0, &[result_addr], cfg.max_retries, &params)
                    .await?;
            (Some(link.sender), None)
        } else {
            (None, None)
        };

        // Wire transport halves into the stage runtime.
        // First rank doesn't receive from a "prev" — drop the receiver.
        // Last rank doesn't send to a "next" — drop the sender.
        let local_stage_cfg = plan.local_stage().clone();
        let next_sender = if local_stage_cfg.is_last {
            None
        } else {
            Some(next_sender)
        };
        let prev_receiver = if local_stage_cfg.is_first {
            None
        } else {
            Some(prev_receiver)
        };

        let stage = PipelineStageRuntime::new(
            local_stage_cfg,
            next_sender,
            prev_receiver,
            result_sender,
            result_receiver,
        );

        Ok(Self { plan, stage })
    }
}
