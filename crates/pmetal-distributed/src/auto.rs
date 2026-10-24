//! Auto-discovery distributed backend.
//!
//! This module provides a zero-configuration distributed training backend
//! that automatically discovers peers on the local network using mDNS/Bonjour.
//!
//! # Usage
//!
//! ```ignore
//! use pmetal_distributed::{AutoDiscoveryBackend, DistributedContext};
//!
//! // Create backend with automatic peer discovery
//! let backend = AutoDiscoveryBackend::new().await?;
//!
//! // Wait for peers to join
//! backend.wait_for_peers(2, Duration::from_secs(30)).await?;
//!
//! // Use for distributed training
//! let ctx = DistributedContext::new(Box::new(backend));
//! ctx.all_reduce(&mut gradients).await?;
//! ```

use crate::collective::RingCollective;
use crate::discovery::{DiscoveryEvent, DiscoveryService};
use crate::error::DistributedError;
use crate::fabric::{InterfaceKind, LocalFabric, probe_local_fabric};
use crate::identity::NodeIdentity;
use crate::namespace::NetworkNamespace;
use crate::secret::ClusterSecret;
use crate::topology::{NodeProfile, SharedTopology, new_shared_topology};
use crate::transport::{Channel, LinkSecurity, RingMembership, TcpTransport};
use crate::{DistributedBackend, ReduceOp};
use anyhow::Result;
use async_trait::async_trait;
use libp2p::PeerId;
use pmetal_core::sync::{read_locked, write_locked};
use std::net::SocketAddr;
use std::ops::Range;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::{Mutex, OnceCell, mpsc};
use tracing::{debug, error, info, warn};

/// Default port for gradient exchange.
const DEFAULT_GRADIENT_PORT: u16 = 52416;

/// Default port for discovery/libp2p.
const DEFAULT_DISCOVERY_PORT: u16 = 52415;

/// Configuration for auto-discovery backend.
#[derive(Debug, Clone)]
pub struct AutoDiscoveryConfig {
    /// Port for gradient exchange (default: 52416).
    pub gradient_port: u16,
    /// Port for libp2p discovery (default: 52415).
    pub discovery_port: u16,
    /// Minimum peers required before training can start.
    pub min_peers: usize,
    /// Maximum time to wait for peers.
    pub peer_timeout: Duration,
    /// Local node profile (for topology awareness).
    pub profile: NodeProfile,
    /// The cluster secret that admits machines to the ring and encrypts its
    /// traffic. Left unset it is looked up as [`ClusterSecret::resolve`]
    /// does (`PMETAL_CLUSTER_SECRET`, then `~/.pmetal/cluster_secret`).
    pub secret: Option<ClusterSecret>,
    /// Allow an unauthenticated, unencrypted ring when no secret exists.
    /// Anyone on the network can then read gradients and inject their own;
    /// for local experiments only.
    pub allow_plaintext: bool,
}

impl Default for AutoDiscoveryConfig {
    fn default() -> Self {
        Self {
            gradient_port: DEFAULT_GRADIENT_PORT,
            discovery_port: DEFAULT_DISCOVERY_PORT,
            min_peers: 1,
            peer_timeout: Duration::from_secs(60),
            profile: NodeProfile::default(),
            secret: None,
            allow_plaintext: false,
        }
    }
}

/// Auto-discovery distributed backend.
///
/// Automatically discovers peers on the local network using mDNS/Bonjour
/// and establishes connections for gradient synchronization.
pub struct AutoDiscoveryBackend {
    /// Our node identity.
    identity: NodeIdentity,
    /// Configuration.
    config: AutoDiscoveryConfig,
    /// Local network fabric snapshot (Thunderbolt / Ethernet / Wi-Fi NICs).
    /// Used to classify peer addresses and pick the best fabric per ring edge.
    fabric: Arc<LocalFabric>,
    /// Cluster topology.
    topology: SharedTopology,
    /// Discovery state.
    discovery_state: Arc<RwLock<crate::discovery::DiscoveryState>>,
    /// How the ring's links are protected.
    security: LinkSecurity,
    /// The formed ring. Set exactly once, even under concurrent
    /// `establish_ring` calls, and only on success so a failed attempt can be
    /// retried. Its rank and world size are fixed from here on, whatever
    /// discovery later reports.
    ring: OnceCell<RingCollective>,
    /// Event receiver from discovery service.
    event_rx: Mutex<mpsc::Receiver<DiscoveryEvent>>,
}

impl AutoDiscoveryBackend {
    /// Create a new auto-discovery backend with default configuration.
    pub async fn new() -> Result<Self> {
        Self::with_config(AutoDiscoveryConfig::default()).await
    }

    /// Create a new auto-discovery backend with custom configuration.
    pub async fn with_config(config: AutoDiscoveryConfig) -> Result<Self> {
        let security = LinkSecurity::from_config(&crate::config::DistributedConfig {
            secret: config.secret.clone(),
            allow_plaintext: config.allow_plaintext,
            ..crate::config::DistributedConfig::new(Vec::new(), 0)
        })?;
        let identity = NodeIdentity::load_or_generate()?;
        let namespace = NetworkNamespace::default();
        let fabric = Arc::new(probe_local_fabric());
        let topology = new_shared_topology(*identity.peer_id(), config.profile.clone());

        // Seed the local node's advertised addresses from the fabric probe so
        // that peer-side ring formation can prefer Thunderbolt right away.
        {
            let mut t = write_locked(&topology);
            let local_addrs: Vec<_> = fabric
                .advertised_addrs()
                .into_iter()
                .map(|(ip, kind)| (SocketAddr::new(ip, config.gradient_port), kind))
                .collect();
            t.set_local_addrs(local_addrs);
        }

        let (event_tx, event_rx) = mpsc::channel(256);

        let mut discovery = DiscoveryService::new(
            identity.clone(),
            namespace.clone(),
            config.discovery_port,
            event_tx,
        );
        // Peers prove they hold the cluster secret before they count as
        // members; without that, any machine on the LAN joins the topology.
        if let LinkSecurity::Authenticated(secret) = &security {
            discovery = discovery.with_secret(secret.clone());
        }
        let discovery_state = discovery.state();

        tokio::spawn(async move {
            if let Err(e) = discovery.run().await {
                error!("Discovery service error: {}", e);
            }
        });

        let tb_count = fabric
            .interfaces()
            .iter()
            .filter(|i| i.kind == InterfaceKind::Thunderbolt)
            .count();
        info!(
            "AutoDiscoveryBackend initialized: peer_id={}, gradient_port={}, discovery_port={}, tb_ifaces={}",
            identity.peer_id(),
            config.gradient_port,
            config.discovery_port,
            tb_count,
        );

        Ok(Self {
            identity,
            config,
            fabric,
            topology,
            discovery_state,
            security,
            ring: OnceCell::new(),
            event_rx: Mutex::new(event_rx),
        })
    }

    /// Snapshot of this node's network fabric (NICs and their kinds).
    /// Used by `pmetal cluster status` and for fabric-aware routing decisions.
    pub fn fabric(&self) -> Arc<LocalFabric> {
        Arc::clone(&self.fabric)
    }

    /// Get the local node's peer ID.
    pub fn peer_id(&self) -> &PeerId {
        self.identity.peer_id()
    }

    /// Get the local node's peer ID as a string.
    pub fn peer_id_string(&self) -> String {
        self.identity.peer_id_string()
    }

    /// Get the current cluster topology.
    pub fn topology(&self) -> SharedTopology {
        Arc::clone(&self.topology)
    }

    /// Get the number of discovered peers.
    pub fn peer_count(&self) -> usize {
        read_locked(&self.discovery_state).connected_count()
    }

    /// Wait for a minimum number of peers to be discovered.
    ///
    /// Returns the number of peers found, or an error if timeout occurs.
    pub async fn wait_for_peers(
        &self,
        min_peers: usize,
        timeout_duration: Duration,
    ) -> Result<usize> {
        info!(
            "Waiting for {} peers (timeout: {:?})",
            min_peers, timeout_duration
        );

        let start = std::time::Instant::now();

        while start.elapsed() < timeout_duration {
            // Process discovery events
            {
                let mut rx = self.event_rx.lock().await;
                while let Ok(event) = rx.try_recv() {
                    self.handle_discovery_event(event).await;
                }
            }

            let count = self.peer_count();
            if count >= min_peers {
                info!("Found {} peers, proceeding", count);
                return Ok(count);
            }

            // Brief sleep before checking again
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let count = self.peer_count();
        if count >= min_peers {
            Ok(count)
        } else {
            Err(DistributedError::Protocol(format!(
                "Timeout waiting for peers: found {} of {} required",
                count, min_peers
            ))
            .into())
        }
    }

    /// Handle a discovery event.
    async fn handle_discovery_event(&self, event: DiscoveryEvent) {
        match event {
            DiscoveryEvent::PeerDiscovered { peer_id, addresses } => {
                debug!("Discovered peer: {} at {:?}", peer_id, addresses);
            }
            DiscoveryEvent::PeerConnected { peer_id, address } => {
                let kind = self.fabric.classify_peer(&address.ip());
                info!(
                    "Connected to peer: {} at {} via {}",
                    peer_id,
                    address,
                    kind.tag()
                );

                // Pull every fabric path the peer advertised (one multiaddr
                // per local NIC the peer is willing to accept connections
                // on). Classify each against our local fabric so the topology
                // has the full set of fallback addresses, not just the one
                // libp2p happened to dial first.
                let extra = read_locked(&self.discovery_state)
                    .get_peer(&peer_id)
                    .map(|p| p.all_socket_addrs())
                    .unwrap_or_default();
                let all_addrs = compose_peer_addrs(&self.fabric, address, kind, &extra);

                let mut topology = write_locked(&self.topology);
                topology.add_node(peer_id, all_addrs);
            }
            DiscoveryEvent::PeerDisconnected { peer_id } => {
                warn!("Disconnected from peer: {}", peer_id);

                let mut topology = write_locked(&self.topology);
                topology.remove_node(&peer_id);

                // Note: ring_init is a OnceCell and cannot be reset.  A peer
                // disconnect means the ring is broken; callers must create a
                // new AutoDiscoveryBackend to reform the ring.
            }
            DiscoveryEvent::PeerExpired { peer_id } => {
                debug!("Peer expired: {}", peer_id);
            }
            DiscoveryEvent::Message { peer_id, data } => {
                debug!("Message from {}: {} bytes", peer_id, data.len());
            }
        }
    }

    /// Internal ring setup — performs the actual TCP connection work.
    ///
    /// Called at most once successfully, via `ring.get_or_try_init(...)`.
    async fn establish_ring_inner(&self) -> Result<RingCollective> {
        // Collect all needed data from topology while holding the lock
        let (local_rank, world_size, endpoints, peer_ids, peer_id_bytes) = {
            let topology = read_locked(&self.topology);

            if !topology.can_form_ring() {
                return Err(DistributedError::Protocol(
                    "Not enough peers to form ring (need at least 2 nodes)".into(),
                )
                .into());
            }

            let ring_order = topology.ring_order();
            let local_rank = topology.local_rank();
            let world_size = ring_order.len();

            // Collect every fabric path per peer in ring order, best first.
            // The transport will try them in order on connect failure, so a
            // disconnected Thunderbolt cable falls back to Ethernet without
            // reforming the ring.
            let mut endpoints: Vec<Vec<SocketAddr>> = Vec::with_capacity(ring_order.len());
            for n in &ring_order {
                let v: Vec<SocketAddr> = n
                    .addrs
                    .iter()
                    .map(|(a, _)| SocketAddr::new(a.ip(), self.config.gradient_port))
                    .collect();
                endpoints.push(v);
            }

            // Collect peer IDs for logging
            let peer_ids: Vec<String> = ring_order.iter().map(|n| n.peer_id.to_base58()).collect();
            let peer_id_bytes: Vec<Vec<u8>> =
                ring_order.iter().map(|n| n.peer_id.to_bytes()).collect();

            (local_rank, world_size, endpoints, peer_ids, peer_id_bytes)
        }; // topology lock released here

        info!(
            "Establishing ring: rank={}/{}, peers={:?}",
            local_rank, world_size, peer_ids
        );

        let reachable_count = endpoints.iter().filter(|e| !e.is_empty()).count();
        if reachable_count < 2 {
            return Err(DistributedError::Protocol(
                "Not enough peers with known addresses to form ring".into(),
            )
            .into());
        }

        let config = crate::config::DistributedConfig {
            nodes: endpoints
                .iter()
                .map(|e| {
                    e.first().copied().unwrap_or_else(|| {
                        "0.0.0.0:0".parse().expect("placeholder addr always valid")
                    })
                })
                .collect(),
            fallback_addrs: endpoints
                .iter()
                .map(|e| e.iter().skip(1).copied().collect())
                .collect(),
            rank: local_rank,
            connection_timeout_ms: 30000,
            max_retries: 50,
            secret: None,
            allow_plaintext: false,
        };

        // Every rank must agree on who is in the ring, and that is a fact
        // about peer identities, not about addresses (each machine sees its
        // peers at different, fabric-specific ones).
        let membership = RingMembership::from_ids(local_rank, &peer_id_bytes);
        let (sender, receiver) = TcpTransport::connect_ring(
            &config,
            self.security.clone(),
            membership,
            None,
            Channel::Ring,
        )
        .await?;

        info!("Ring established successfully");
        Ok(RingCollective::new(
            local_rank, world_size, sender, receiver,
        ))
    }

    /// Ensure the ring is established, initialising it exactly once.
    ///
    /// Uses `tokio::sync::OnceCell::get_or_try_init` so that exactly one
    /// concurrent caller performs the TCP connection and all others wait.
    /// If the connection attempt fails the cell remains unset, allowing the
    /// caller to retry on the next all-reduce.
    pub async fn establish_ring(&self) -> Result<()> {
        self.ring().await.map(|_| ())
    }

    async fn ring(&self) -> Result<&RingCollective> {
        self.ring
            .get_or_try_init(|| self.establish_ring_inner())
            .await
    }

    /// Check if the ring has been successfully established.
    ///
    /// Returns `true` iff `establish_ring` has completed successfully at least
    /// once.  This is a cheap non-blocking check suitable for logging.
    pub fn is_ring_ready(&self) -> bool {
        self.ring.initialized()
    }
}

#[async_trait]
impl DistributedBackend for AutoDiscoveryBackend {
    fn rank(&self) -> usize {
        match self.ring.get() {
            Some(ring) => ring.rank(),
            None => read_locked(&self.topology).local_rank(),
        }
    }

    fn world_size(&self) -> usize {
        match self.ring.get() {
            Some(ring) => ring.world_size(),
            None => read_locked(&self.topology).node_count(),
        }
    }

    async fn all_reduce_f32(&self, data: &mut [f32], op: ReduceOp) -> Result<()> {
        // `ring` is idempotent: it forms the ring on first use only.
        self.ring().await?.all_reduce(data, op).await
    }

    async fn reduce_scatter_f32(&self, data: &mut [f32], op: ReduceOp) -> Result<Range<usize>> {
        self.ring().await?.reduce_scatter(data, op).await
    }

    async fn all_gather_f32_in_place(&self, data: &mut [f32]) -> Result<()> {
        self.ring().await?.all_gather_in_place(data).await
    }

    async fn all_gather_bytes(&self, local: &[u8]) -> Result<Vec<Vec<u8>>> {
        self.ring().await?.all_gather_bytes(local).await
    }

    async fn broadcast(&self, buffer: &mut [u8], root: usize) -> Result<()> {
        self.ring().await?.broadcast(buffer, root).await
    }

    async fn barrier(&self) -> Result<()> {
        self.ring().await?.barrier().await
    }

    fn traffic(&self) -> Option<(u64, u64)> {
        self.ring.get().map(RingCollective::traffic)
    }
}

/// Build the deduplicated `(addr, kind)` list for a freshly-connected peer.
///
/// `primary` / `primary_kind` come from libp2p's actual dialed connection;
/// `extra` is every other multiaddr the peer advertised. We classify each
/// `extra` against our local fabric snapshot and append it if it's not
/// already present in the list. The output is suitable to pass straight to
/// [`crate::topology::ClusterTopology::add_node`].
fn compose_peer_addrs(
    fabric: &LocalFabric,
    primary: SocketAddr,
    primary_kind: InterfaceKind,
    extra: &[SocketAddr],
) -> Vec<(SocketAddr, InterfaceKind)> {
    let mut all_addrs: Vec<(SocketAddr, InterfaceKind)> = vec![(primary, primary_kind)];
    for sa in extra {
        if all_addrs.iter().any(|(existing, _)| *existing == *sa) {
            continue;
        }
        let k = fabric.classify_peer(&sa.ip());
        all_addrs.push((*sa, k));
    }
    all_addrs
}

impl std::fmt::Debug for AutoDiscoveryBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoDiscoveryBackend")
            .field("peer_id", &self.identity.peer_id_string())
            .field("peer_count", &self.peer_count())
            .field("ring_ready", &self.is_ring_ready())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::InterfaceInfo;
    use std::net::Ipv4Addr;

    fn fabric_with(ifaces: Vec<(&str, InterfaceKind, Vec<Ipv4Addr>)>) -> LocalFabric {
        let interfaces = ifaces
            .into_iter()
            .map(|(name, kind, ips)| InterfaceInfo {
                name: name.to_string(),
                addrs: ips.into_iter().map(std::net::IpAddr::V4).collect(),
                kind,
                link_speed: None,
            })
            .collect();
        LocalFabric::from_interfaces(interfaces)
    }

    #[test]
    fn compose_peer_addrs_classifies_extra_multiaddrs() {
        // Local fabric: TB on 169.254.1.10, Ethernet on 192.168.1.5.
        let fabric = fabric_with(vec![
            (
                "bridge0",
                InterfaceKind::Thunderbolt,
                vec![Ipv4Addr::new(169, 254, 1, 10)],
            ),
            (
                "en0",
                InterfaceKind::Ethernet,
                vec![Ipv4Addr::new(192, 168, 1, 5)],
            ),
        ]);

        // Peer dialed in over Ethernet first; their TB address is in `extra`.
        let primary: SocketAddr = "192.168.1.42:52415".parse().unwrap();
        let extra: Vec<SocketAddr> = vec![
            "169.254.1.42:52415".parse().unwrap(), // TB path
            "192.168.1.42:52415".parse().unwrap(), // duplicate of primary — dedup
        ];

        let out = compose_peer_addrs(&fabric, primary, InterfaceKind::Ethernet, &extra);

        // Primary first, then the TB addr; dedup drops the duplicate.
        assert_eq!(out.len(), 2, "got {:?}", out);
        assert_eq!(out[0], (primary, InterfaceKind::Ethernet));
        assert_eq!(out[1].1, InterfaceKind::Thunderbolt);
    }

    #[test]
    fn compose_peer_addrs_empty_extra() {
        let fabric = fabric_with(vec![]);
        let primary: SocketAddr = "10.0.0.1:5000".parse().unwrap();
        let out = compose_peer_addrs(&fabric, primary, InterfaceKind::Unknown, &[]);
        assert_eq!(out, vec![(primary, InterfaceKind::Unknown)]);
    }

    #[test]
    fn topology_set_local_addrs_seeds_local_node_for_advertise() {
        // This test exercises the topology integration path that
        // AutoDiscoveryBackend::with_config uses at startup: probe fabric →
        // build advertised addrs → set on local node. We verify that the
        // local node's primary fabric is the highest-ranked interface.
        let local_id = libp2p::PeerId::random();
        let mut topo = crate::topology::ClusterTopology::new(
            local_id,
            crate::topology::NodeProfile::default(),
        );

        let fabric = fabric_with(vec![
            (
                "bridge0",
                InterfaceKind::Thunderbolt,
                vec![Ipv4Addr::new(169, 254, 1, 1)],
            ),
            (
                "en0",
                InterfaceKind::Ethernet,
                vec![Ipv4Addr::new(192, 168, 1, 50)],
            ),
        ]);

        let local_addrs: Vec<_> = fabric
            .advertised_addrs()
            .into_iter()
            .map(|(ip, kind)| (SocketAddr::new(ip, 52416), kind))
            .collect();
        topo.set_local_addrs(local_addrs);

        let local = topo.get_node(&local_id).expect("local node present");
        assert_eq!(local.addrs.len(), 2);
        assert_eq!(local.best_addr().unwrap().1, InterfaceKind::Thunderbolt);
    }
}
