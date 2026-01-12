//! Automatic peer discovery using mDNS (Bonjour/Avahi).
//!
//! This module provides zero-configuration peer discovery for local networks.
//! Nodes automatically find each other using multicast DNS without manual
//! configuration of IP addresses.
//!
//! # Architecture
//!
//! The discovery system uses libp2p's mDNS implementation which:
//! - Announces the node's presence on the local network
//! - Discovers other pmetal nodes automatically
//! - Handles peer expiration when nodes leave
//!
//! # Service Name
//!
//! Nodes advertise themselves as `_pmetal._tcp.local` to avoid conflicts
//! with other libp2p applications.

use crate::error::DistributedError;
use crate::identity::NodeIdentity;
use anyhow::Result;
use futures::StreamExt;
use libp2p::{
    gossipsub, identify, mdns, noise, ping, swarm::NetworkBehaviour, swarm::SwarmEvent, tcp, yamux,
    Multiaddr, PeerId, Swarm, SwarmBuilder,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info, trace, warn};

/// Version string for network namespace isolation.
const NETWORK_VERSION: &str = "pmetal/0.1.0";

/// Service discovery TTL in seconds.
const MDNS_TTL_SECS: u64 = 300;

/// mDNS query interval in seconds.
const MDNS_QUERY_INTERVAL_SECS: u64 = 60;

/// Ping interval for connection health.
const PING_INTERVAL_SECS: u64 = 15;

/// Events emitted by the discovery system.
#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    /// A new peer was discovered.
    PeerDiscovered {
        peer_id: PeerId,
        addresses: Vec<Multiaddr>,
    },
    /// A peer has expired/disconnected.
    PeerExpired { peer_id: PeerId },
    /// A peer connection was established.
    PeerConnected {
        peer_id: PeerId,
        address: SocketAddr,
    },
    /// A peer connection was lost.
    PeerDisconnected { peer_id: PeerId },
    /// Received a message from a peer.
    Message { peer_id: PeerId, data: Vec<u8> },
}

/// Combined network behaviour for discovery and communication.
#[derive(NetworkBehaviour)]
pub struct PMetalBehaviour {
    /// mDNS for local peer discovery.
    mdns: mdns::tokio::Behaviour,
    /// Gossipsub for pub/sub messaging.
    gossipsub: gossipsub::Behaviour,
    /// Identify protocol for peer information exchange.
    identify: identify::Behaviour,
    /// Ping for connection health monitoring.
    ping: ping::Behaviour,
}

/// Discovered peer information.
#[derive(Debug, Clone)]
pub struct PeerInfo {
    /// The peer's ID.
    pub peer_id: PeerId,
    /// Known addresses for this peer.
    pub addresses: Vec<Multiaddr>,
    /// Socket address for direct TCP communication.
    pub socket_addr: Option<SocketAddr>,
    /// Whether we have an active connection.
    pub connected: bool,
}

/// Shared state for discovered peers.
#[derive(Debug, Default)]
pub struct DiscoveryState {
    /// Known peers indexed by PeerId.
    peers: HashMap<PeerId, PeerInfo>,
    /// Connected peer IDs in order of connection.
    connected_order: Vec<PeerId>,
}

impl DiscoveryState {
    /// Get all connected peers in connection order.
    pub fn connected_peers(&self) -> Vec<&PeerInfo> {
        self.connected_order
            .iter()
            .filter_map(|id| self.peers.get(id))
            .filter(|p| p.connected)
            .collect()
    }

    /// Get the number of connected peers.
    pub fn connected_count(&self) -> usize {
        self.peers.values().filter(|p| p.connected).count()
    }

    /// Get peer info by ID.
    pub fn get_peer(&self, peer_id: &PeerId) -> Option<&PeerInfo> {
        self.peers.get(peer_id)
    }

    /// Get socket addresses of all connected peers.
    pub fn connected_socket_addrs(&self) -> Vec<SocketAddr> {
        self.connected_peers()
            .iter()
            .filter_map(|p| p.socket_addr)
            .collect()
    }
}

/// Discovery service for automatic peer finding.
pub struct DiscoveryService {
    /// Our node identity.
    identity: NodeIdentity,
    /// Shared discovery state.
    state: Arc<RwLock<DiscoveryState>>,
    /// Event sender.
    event_tx: mpsc::Sender<DiscoveryEvent>,
    /// Port for listening.
    listen_port: u16,
}

impl DiscoveryService {
    /// Create a new discovery service.
    ///
    /// # Arguments
    ///
    /// * `identity` - The node's identity
    /// * `listen_port` - Port to listen on (0 for random)
    /// * `event_tx` - Channel to send discovery events
    pub fn new(
        identity: NodeIdentity,
        listen_port: u16,
        event_tx: mpsc::Sender<DiscoveryEvent>,
    ) -> Self {
        Self {
            identity,
            state: Arc::new(RwLock::new(DiscoveryState::default())),
            event_tx,
            listen_port,
        }
    }

    /// Get a handle to the shared discovery state.
    pub fn state(&self) -> Arc<RwLock<DiscoveryState>> {
        Arc::clone(&self.state)
    }

    /// Run the discovery service.
    ///
    /// This spawns the libp2p swarm and handles events.
    pub async fn run(self) -> Result<()> {
        let mut swarm = self.build_swarm()?;

        // Listen on all interfaces
        let listen_addr: Multiaddr = format!("/ip4/0.0.0.0/tcp/{}", self.listen_port)
            .parse()
            .map_err(|e| DistributedError::Config(format!("Invalid listen address: {}", e)))?;

        swarm.listen_on(listen_addr)?;

        info!(
            "Discovery service started, peer_id={}",
            self.identity.peer_id()
        );

        // Subscribe to the gradient sync topic
        let topic = gossipsub::IdentTopic::new("pmetal/gradients");
        swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

        // Event loop
        loop {
            match swarm.select_next_some().await {
                SwarmEvent::Behaviour(PMetalBehaviourEvent::Mdns(event)) => {
                    self.handle_mdns_event(&mut swarm, event).await;
                }
                SwarmEvent::Behaviour(PMetalBehaviourEvent::Gossipsub(event)) => {
                    self.handle_gossipsub_event(event).await;
                }
                SwarmEvent::Behaviour(PMetalBehaviourEvent::Identify(event)) => {
                    self.handle_identify_event(event).await;
                }
                SwarmEvent::Behaviour(PMetalBehaviourEvent::Ping(event)) => {
                    trace!("Ping event: {:?}", event);
                }
                SwarmEvent::ConnectionEstablished {
                    peer_id, endpoint, ..
                } => {
                    let addr = endpoint.get_remote_address();
                    info!("Connection established with {} at {}", peer_id, addr);

                    // Extract socket address
                    let socket_addr = multiaddr_to_socket_addr(addr);

                    {
                        let mut state = self.state.write();
                        if let Some(peer) = state.peers.get_mut(&peer_id) {
                            peer.connected = true;
                            peer.socket_addr = socket_addr;
                        }
                        if !state.connected_order.contains(&peer_id) {
                            state.connected_order.push(peer_id);
                        }
                    }

                    if let Some(addr) = socket_addr {
                        let _ = self
                            .event_tx
                            .send(DiscoveryEvent::PeerConnected {
                                peer_id,
                                address: addr,
                            })
                            .await;
                    }
                }
                SwarmEvent::ConnectionClosed { peer_id, .. } => {
                    info!("Connection closed with {}", peer_id);

                    {
                        let mut state = self.state.write();
                        if let Some(peer) = state.peers.get_mut(&peer_id) {
                            peer.connected = false;
                        }
                        state.connected_order.retain(|id| id != &peer_id);
                    }

                    let _ = self
                        .event_tx
                        .send(DiscoveryEvent::PeerDisconnected { peer_id })
                        .await;
                }
                SwarmEvent::NewListenAddr { address, .. } => {
                    info!("Listening on {}", address);
                }
                _ => {}
            }
        }
    }

    /// Build the libp2p swarm.
    fn build_swarm(&self) -> Result<Swarm<PMetalBehaviour>> {
        // Create mDNS behaviour
        let mdns_config = mdns::Config {
            ttl: Duration::from_secs(MDNS_TTL_SECS),
            query_interval: Duration::from_secs(MDNS_QUERY_INTERVAL_SECS),
            enable_ipv6: false, // IPv6 mDNS can be problematic
        };

        // Create gossipsub behaviour
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(1))
            .validation_mode(gossipsub::ValidationMode::Strict)
            .build()
            .map_err(|e| DistributedError::Config(format!("Gossipsub config error: {}", e)))?;

        let gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(self.identity.keypair().clone()),
            gossipsub_config,
        )
        .map_err(|e| DistributedError::Config(format!("Gossipsub error: {}", e)))?;

        // Create identify behaviour
        let identify = identify::Behaviour::new(identify::Config::new(
            format!("/pmetal/{}", NETWORK_VERSION),
            self.identity.keypair().public(),
        ));

        // Create ping behaviour
        let ping = ping::Behaviour::new(ping::Config::new().with_interval(Duration::from_secs(PING_INTERVAL_SECS)));

        let swarm = SwarmBuilder::with_existing_identity(self.identity.keypair().clone())
            .with_tokio()
            .with_tcp(
                tcp::Config::default().nodelay(true),
                noise::Config::new,
                yamux::Config::default,
            )
            .map_err(|e| DistributedError::Config(format!("TCP config error: {}", e)))?
            .with_behaviour(|key| {
                let peer_id = PeerId::from(key.public());
                let mdns = mdns::tokio::Behaviour::new(mdns_config, peer_id)
                    .expect("mDNS behaviour creation failed");

                PMetalBehaviour {
                    mdns,
                    gossipsub,
                    identify,
                    ping,
                }
            })
            .map_err(|e| DistributedError::Config(format!("Behaviour error: {}", e)))?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
            .build();

        Ok(swarm)
    }

    /// Handle mDNS events.
    async fn handle_mdns_event(&self, swarm: &mut Swarm<PMetalBehaviour>, event: mdns::Event) {
        match event {
            mdns::Event::Discovered(peers) => {
                for (peer_id, addr) in peers {
                    if peer_id == *self.identity.peer_id() {
                        continue; // Skip ourselves
                    }

                    debug!("mDNS discovered peer {} at {}", peer_id, addr);

                    // Add to state
                    {
                        let mut state = self.state.write();
                        let peer = state.peers.entry(peer_id).or_insert_with(|| PeerInfo {
                            peer_id,
                            addresses: Vec::new(),
                            socket_addr: None,
                            connected: false,
                        });

                        if !peer.addresses.contains(&addr) {
                            peer.addresses.push(addr.clone());
                        }
                    }

                    // Dial the peer
                    if let Err(e) = swarm.dial(addr.clone()) {
                        debug!("Failed to dial {}: {}", peer_id, e);
                    }

                    let addresses = {
                        let state = self.state.read();
                        state
                            .peers
                            .get(&peer_id)
                            .map(|p| p.addresses.clone())
                            .unwrap_or_default()
                    };

                    let _ = self
                        .event_tx
                        .send(DiscoveryEvent::PeerDiscovered { peer_id, addresses })
                        .await;
                }
            }
            mdns::Event::Expired(peers) => {
                for (peer_id, _addr) in peers {
                    debug!("mDNS peer expired: {}", peer_id);

                    let _ = self
                        .event_tx
                        .send(DiscoveryEvent::PeerExpired { peer_id })
                        .await;
                }
            }
        }
    }

    /// Handle gossipsub events.
    async fn handle_gossipsub_event(&self, event: gossipsub::Event) {
        if let gossipsub::Event::Message {
            propagation_source,
            message,
            ..
        } = event
        {
            debug!(
                "Received gossipsub message from {} on topic {}",
                propagation_source, message.topic
            );

            let _ = self
                .event_tx
                .send(DiscoveryEvent::Message {
                    peer_id: propagation_source,
                    data: message.data,
                })
                .await;
        }
    }

    /// Handle identify events.
    async fn handle_identify_event(&self, event: identify::Event) {
        if let identify::Event::Received { peer_id, info, .. } = event {
            debug!(
                "Identified peer {}: {} ({})",
                peer_id, info.protocol_version, info.agent_version
            );

            // Verify it's an pmetal node
            if !info.protocol_version.starts_with("/pmetal/") {
                warn!("Peer {} is not an pmetal node, ignoring", peer_id);
            }
        }
    }
}

/// Extract socket address from multiaddr.
fn multiaddr_to_socket_addr(addr: &Multiaddr) -> Option<SocketAddr> {
    let mut ip = None;
    let mut port = None;

    for protocol in addr.iter() {
        match protocol {
            libp2p::multiaddr::Protocol::Ip4(ipv4) => {
                ip = Some(std::net::IpAddr::V4(ipv4));
            }
            libp2p::multiaddr::Protocol::Ip6(ipv6) => {
                ip = Some(std::net::IpAddr::V6(ipv6));
            }
            libp2p::multiaddr::Protocol::Tcp(p) => {
                port = Some(p);
            }
            _ => {}
        }
    }

    match (ip, port) {
        (Some(ip), Some(port)) => Some(SocketAddr::new(ip, port)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiaddr_to_socket_addr() {
        let addr: Multiaddr = "/ip4/192.168.1.100/tcp/5000".parse().unwrap();
        let socket = multiaddr_to_socket_addr(&addr);
        assert!(socket.is_some());
        assert_eq!(socket.unwrap().to_string(), "192.168.1.100:5000");
    }

    #[test]
    fn test_discovery_state() {
        let mut state = DiscoveryState::default();
        let peer_id = PeerId::random();

        state.peers.insert(
            peer_id,
            PeerInfo {
                peer_id,
                addresses: Vec::new(),
                socket_addr: Some("192.168.1.1:5000".parse().unwrap()),
                connected: true,
            },
        );
        state.connected_order.push(peer_id);

        assert_eq!(state.connected_count(), 1);
        assert_eq!(state.connected_socket_addrs().len(), 1);
    }
}
