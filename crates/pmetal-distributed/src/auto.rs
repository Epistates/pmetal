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

use crate::discovery::{DiscoveryEvent, DiscoveryService};
use crate::error::DistributedError;
use crate::identity::NodeIdentity;
use crate::topology::{new_shared_topology, NodeProfile, SharedTopology};
use crate::transport::{TcpTransport, TransportReceiver, TransportSender};
use crate::DistributedBackend;
use anyhow::Result;
use async_trait::async_trait;
use bytemuck::cast_slice_mut;
use libp2p::PeerId;
use parking_lot::RwLock;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
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
}

impl Default for AutoDiscoveryConfig {
    fn default() -> Self {
        Self {
            gradient_port: DEFAULT_GRADIENT_PORT,
            discovery_port: DEFAULT_DISCOVERY_PORT,
            min_peers: 1,
            peer_timeout: Duration::from_secs(60),
            profile: NodeProfile::default(),
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
    /// Cluster topology.
    topology: SharedTopology,
    /// Discovery state.
    discovery_state: Arc<RwLock<crate::discovery::DiscoveryState>>,
    /// Ring connections (sender to next, receiver from prev).
    ring_connections: Mutex<Option<(TransportSender, TransportReceiver)>>,
    /// Event receiver from discovery service.
    event_rx: Mutex<mpsc::Receiver<DiscoveryEvent>>,
    /// Whether the ring is established.
    ring_ready: Arc<std::sync::atomic::AtomicBool>,
}

impl AutoDiscoveryBackend {
    /// Create a new auto-discovery backend with default configuration.
    pub async fn new() -> Result<Self> {
        Self::with_config(AutoDiscoveryConfig::default()).await
    }

    /// Create a new auto-discovery backend with custom configuration.
    pub async fn with_config(config: AutoDiscoveryConfig) -> Result<Self> {
        let identity = NodeIdentity::load_or_generate()?;
        let topology = new_shared_topology(*identity.peer_id(), config.profile.clone());

        // Create event channel
        let (event_tx, event_rx) = mpsc::channel(256);

        // Create and spawn discovery service
        let discovery = DiscoveryService::new(identity.clone(), config.discovery_port, event_tx);
        let discovery_state = discovery.state();

        // Spawn discovery in background
        tokio::spawn(async move {
            if let Err(e) = discovery.run().await {
                error!("Discovery service error: {}", e);
            }
        });

        info!(
            "AutoDiscoveryBackend initialized: peer_id={}, gradient_port={}, discovery_port={}",
            identity.peer_id(),
            config.gradient_port,
            config.discovery_port
        );

        Ok(Self {
            identity,
            config,
            topology,
            discovery_state,
            ring_connections: Mutex::new(None),
            event_rx: Mutex::new(event_rx),
            ring_ready: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
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
        self.discovery_state.read().connected_count()
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
                info!("Connected to peer: {} at {}", peer_id, address);

                let mut topology = self.topology.write();
                topology.add_node(peer_id, Some(address));
            }
            DiscoveryEvent::PeerDisconnected { peer_id } => {
                warn!("Disconnected from peer: {}", peer_id);

                let mut topology = self.topology.write();
                topology.remove_node(&peer_id);

                // Mark ring as not ready
                self.ring_ready
                    .store(false, std::sync::atomic::Ordering::SeqCst);
            }
            DiscoveryEvent::PeerExpired { peer_id } => {
                debug!("Peer expired: {}", peer_id);
            }
            DiscoveryEvent::Message { peer_id, data } => {
                debug!("Message from {}: {} bytes", peer_id, data.len());
            }
        }
    }

    /// Establish the ring topology for all-reduce operations.
    ///
    /// This must be called before performing all-reduce operations.
    pub async fn establish_ring(&self) -> Result<()> {
        // Collect all needed data from topology while holding the lock
        let (local_rank, world_size, node_addrs, peer_ids) = {
            let topology = self.topology.read();

            if !topology.can_form_ring() {
                return Err(DistributedError::Protocol(
                    "Not enough peers to form ring (need at least 2 nodes)".into(),
                )
                .into());
            }

            let ring_order = topology.ring_order();
            let local_rank = topology.local_rank();
            let world_size = ring_order.len();

            // Collect socket addresses in ring order
            let node_addrs: Vec<SocketAddr> = ring_order
                .iter()
                .filter_map(|n| n.socket_addr)
                .map(|a| SocketAddr::new(a.ip(), self.config.gradient_port))
                .collect();

            // Collect peer IDs for logging
            let peer_ids: Vec<String> = ring_order.iter().map(|n| n.peer_id.to_base58()).collect();

            (local_rank, world_size, node_addrs, peer_ids)
        }; // topology lock released here

        info!(
            "Establishing ring: rank={}/{}, peers={:?}",
            local_rank, world_size, peer_ids
        );

        if node_addrs.len() < 2 {
            return Err(DistributedError::Protocol(
                "Not enough peers with known addresses to form ring".into(),
            )
            .into());
        }

        // Create configuration for TCP transport
        let config = crate::config::DistributedConfig {
            nodes: node_addrs,
            rank: local_rank,
            connection_timeout_ms: 30000,
            max_retries: 50,
        };

        // Establish ring connections
        let (sender, receiver) = TcpTransport::connect(&config).await?;

        *self.ring_connections.lock().await = Some((sender, receiver));
        self.ring_ready
            .store(true, std::sync::atomic::Ordering::SeqCst);

        info!("Ring established successfully");
        Ok(())
    }

    /// Check if the ring is ready for all-reduce operations.
    pub fn is_ring_ready(&self) -> bool {
        self.ring_ready.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait]
impl DistributedBackend for AutoDiscoveryBackend {
    fn rank(&self) -> usize {
        self.topology.read().local_rank()
    }

    fn world_size(&self) -> usize {
        self.topology.read().node_count()
    }

    async fn all_reduce(&self, buffer: &mut [u8]) -> Result<()> {
        if !self.is_ring_ready() {
            self.establish_ring().await?;
        }

        // Validate buffer
        if !buffer.len().is_multiple_of(4) {
            return Err(DistributedError::Protocol(format!(
                "Buffer length {} is not a multiple of 4 (f32 size)",
                buffer.len()
            ))
            .into());
        }

        if !(buffer.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(DistributedError::Protocol(
                "Buffer is not properly aligned for f32 operations".into(),
            )
            .into());
        }

        let floats: &mut [f32] = cast_slice_mut(buffer);
        let len = floats.len();
        let world_size = self.world_size();
        let rank = self.rank();

        if world_size < 2 {
            return Ok(()); // Nothing to reduce
        }

        let chunk_size = len / world_size;
        let remainder = len % world_size;

        let get_chunk_range = |idx: usize| -> (usize, usize) {
            let start = idx * chunk_size + idx.min(remainder);
            let end = start + chunk_size + (if idx < remainder { 1 } else { 0 });
            (start, end)
        };

        let mut connections = self.ring_connections.lock().await;
        let (sender, receiver) = connections
            .as_mut()
            .ok_or_else(|| DistributedError::Protocol("Ring not established".into()))?;

        // === SCATTER-REDUCE PHASE ===
        for step in 0..(world_size - 1) {
            let send_idx = (rank + world_size - step) % world_size;
            let recv_idx = (rank + world_size - step - 1) % world_size;

            let (send_start, send_end) = get_chunk_range(send_idx);
            let (recv_start, recv_end) = get_chunk_range(recv_idx);

            let send_bytes_len = (send_end - send_start) * 4;
            let recv_bytes_len = (recv_end - recv_start) * 4;

            // Copy data to send buffer
            let mut send_buf = vec![0u8; send_bytes_len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    floats[send_start..send_end].as_ptr() as *const u8,
                    send_buf.as_mut_ptr(),
                    send_bytes_len,
                );
            }

            // Send and receive concurrently
            let mut recv_buf = vec![0u8; recv_bytes_len];
            tokio::try_join!(sender.send(&send_buf), receiver.recv(&mut recv_buf))?;

            // Reduce received data into local buffer
            let recv_floats: &[f32] = bytemuck::cast_slice(&recv_buf);
            for (i, &val) in recv_floats.iter().enumerate() {
                floats[recv_start + i] += val;
            }
        }

        // === ALL-GATHER PHASE ===
        for step in 0..(world_size - 1) {
            let send_idx = (rank + world_size - step + 1) % world_size;
            let recv_idx = (rank + world_size - step) % world_size;

            let (send_start, send_end) = get_chunk_range(send_idx);
            let (recv_start, recv_end) = get_chunk_range(recv_idx);

            let recv_bytes_len = (recv_end - recv_start) * 4;

            let send_buf: &[u8] = bytemuck::cast_slice(&floats[send_start..send_end]);

            let mut recv_buf = vec![0u8; recv_bytes_len];
            tokio::try_join!(sender.send(send_buf), receiver.recv(&mut recv_buf))?;

            // Copy received data to local buffer
            unsafe {
                std::ptr::copy_nonoverlapping(
                    recv_buf.as_ptr(),
                    floats[recv_start..recv_end].as_mut_ptr() as *mut u8,
                    recv_bytes_len,
                );
            }
        }

        Ok(())
    }

    async fn barrier(&self) -> Result<()> {
        if !self.is_ring_ready() {
            self.establish_ring().await?;
        }

        let world_size = self.world_size();
        if world_size < 2 {
            return Ok(());
        }

        let mut connections = self.ring_connections.lock().await;
        let (sender, receiver) = connections
            .as_mut()
            .ok_or_else(|| DistributedError::Protocol("Ring not established".into()))?;

        // Simple barrier: send a token around the ring
        let token = [0u8; 4];

        for _ in 0..(world_size - 1) {
            let mut recv_buf = [0u8; 4];
            tokio::try_join!(sender.send(&token), receiver.recv(&mut recv_buf))?;
        }

        Ok(())
    }
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
