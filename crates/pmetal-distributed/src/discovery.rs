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
use crate::namespace::{NetworkNamespace, validate_peer_namespace};
use crate::secret::ClusterSecret;
use anyhow::Result;
use futures::StreamExt;
use libp2p::{
    Multiaddr, PeerId, Swarm, SwarmBuilder, gossipsub, identify, mdns, noise, ping,
    swarm::NetworkBehaviour, swarm::SwarmEvent, tcp, yamux,
};
use pmetal_core::sync::{read_locked, write_locked};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info, trace, warn};

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
    /// Known multiaddrs for this peer (includes every interface the peer
    /// advertised via mDNS).
    pub addresses: Vec<Multiaddr>,
    /// Socket address libp2p actually used to reach this peer.
    pub socket_addr: Option<SocketAddr>,
    /// Whether we have an active connection.
    pub connected: bool,
}

impl PeerInfo {
    /// All advertised TCP socket addresses extracted from `addresses`.
    /// Used by [`crate::auto::AutoDiscoveryBackend`] to discover *all*
    /// fabric paths to a peer (not just the one libp2p picked to dial).
    pub fn all_socket_addrs(&self) -> Vec<SocketAddr> {
        self.addresses
            .iter()
            .filter_map(multiaddr_to_socket_addr)
            .collect()
    }
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
    /// Namespace used for libp2p identify and gossipsub isolation.
    namespace: NetworkNamespace,
    /// Shared discovery state.
    state: Arc<RwLock<DiscoveryState>>,
    /// Event sender.
    event_tx: mpsc::Sender<DiscoveryEvent>,
    /// Port for listening.
    listen_port: u16,
    /// When set, a peer counts as a cluster member only once it proves it
    /// holds this secret (see [`attestation_token`]).
    secret: Option<ClusterSecret>,
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
        namespace: NetworkNamespace,
        listen_port: u16,
        event_tx: mpsc::Sender<DiscoveryEvent>,
    ) -> Self {
        Self {
            identity,
            namespace,
            state: Arc::new(RwLock::new(DiscoveryState::default())),
            event_tx,
            listen_port,
            secret: None,
        }
    }

    /// Admit only peers that prove they hold `secret`.
    ///
    /// The namespace is public (it is derived from the version string), so
    /// without this any machine on the LAN that speaks the protocol joins the
    /// topology, and the ring formed from it. With it, a peer that does not
    /// carry a valid attestation in its identify message is disconnected,
    /// never reported as connected, and never counted as a member.
    pub fn with_secret(mut self, secret: ClusterSecret) -> Self {
        self.secret = Some(secret);
        self
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
        let topic = gossipsub::IdentTopic::new(self.namespace.gossipsub_topic("gradients"));
        swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

        // Connections whose identify message has not yet proved membership.
        let mut unattested: HashMap<PeerId, Option<SocketAddr>> = HashMap::new();

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
                    self.handle_identify_event(&mut swarm, event, &mut unattested)
                        .await;
                }
                SwarmEvent::Behaviour(PMetalBehaviourEvent::Ping(event)) => {
                    trace!("Ping event: {:?}", event);
                }
                SwarmEvent::ConnectionEstablished {
                    peer_id, endpoint, ..
                } => {
                    let addr = endpoint.get_remote_address();
                    info!("Connection established with {} at {}", peer_id, addr);
                    let socket_addr = multiaddr_to_socket_addr(addr);
                    if self.secret.is_some() {
                        // Not a member until its identify message attests.
                        unattested.insert(peer_id, socket_addr);
                    } else {
                        self.mark_connected(peer_id, socket_addr).await;
                    }
                }
                SwarmEvent::ConnectionClosed { peer_id, .. } => {
                    info!("Connection closed with {}", peer_id);
                    if unattested.remove(&peer_id).is_some() {
                        // Never admitted, so nothing to retract.
                        continue;
                    }

                    {
                        let mut state = write_locked(&self.state);
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
        let identify = identify::Behaviour::new(
            identify::Config::new(
                self.namespace.protocol_id(),
                self.identity.keypair().public(),
            )
            .with_agent_version(agent_version(self.secret.as_ref(), self.identity.peer_id())),
        );

        // Create ping behaviour
        let ping = ping::Behaviour::new(
            ping::Config::new().with_interval(Duration::from_secs(PING_INTERVAL_SECS)),
        );

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
                        let mut state = write_locked(&self.state);
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
                        let state = read_locked(&self.state);
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

    /// Record a peer as connected and tell the backend.
    async fn mark_connected(&self, peer_id: PeerId, socket_addr: Option<SocketAddr>) {
        {
            let mut state = write_locked(&self.state);
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

    /// Handle identify events.
    async fn handle_identify_event(
        &self,
        swarm: &mut Swarm<PMetalBehaviour>,
        event: identify::Event,
        unattested: &mut HashMap<PeerId, Option<SocketAddr>>,
    ) {
        if let identify::Event::Received { peer_id, info, .. } = event {
            debug!(
                "Identified peer {}: {} ({})",
                peer_id, info.protocol_version, info.agent_version
            );

            if !validate_peer_namespace(&self.namespace, &info.protocol_version) {
                warn!(
                    "Peer {} has incompatible namespace protocol {}, disconnecting",
                    peer_id, info.protocol_version
                );
                let _ = swarm.disconnect_peer_id(peer_id);
                return;
            }

            if let Some(secret) = &self.secret {
                if !attestation_is_valid(secret, &peer_id, &info.agent_version) {
                    warn!(
                        "Peer {} did not prove it holds this cluster's secret, disconnecting",
                        peer_id
                    );
                    unattested.remove(&peer_id);
                    let _ = swarm.disconnect_peer_id(peer_id);
                    return;
                }
                if let Some(socket_addr) = unattested.remove(&peer_id) {
                    self.mark_connected(peer_id, socket_addr).await;
                }
            }
        }
    }
}

/// Prefix of the identify agent version a peer advertises.
const AGENT_PREFIX: &str = "pmetal";

/// Key in the agent version carrying a peer's attestation.
const ATTEST_KEY: &str = "attest=";

/// The agent version a node advertises: its name and, when the cluster has a
/// secret, a keyed hash of its own peer id. Only a holder of the secret can
/// compute it, and it is bound to one peer id, so it cannot be lifted from
/// one machine and presented by another.
fn agent_version(secret: Option<&ClusterSecret>, peer_id: &PeerId) -> String {
    let mut out = format!("{AGENT_PREFIX}/{}", env!("CARGO_PKG_VERSION"));
    if let Some(secret) = secret {
        out.push(' ');
        out.push_str(ATTEST_KEY);
        out.push_str(&agent_version_token(secret, peer_id));
    }
    out
}

/// Whether `agent_version` carries a valid attestation for `peer_id`.
fn attestation_is_valid(secret: &ClusterSecret, peer_id: &PeerId, agent_version: &str) -> bool {
    let expected = agent_version_token(secret, peer_id);
    agent_version
        .split_whitespace()
        .filter_map(|word| word.strip_prefix(ATTEST_KEY))
        .any(|got| {
            got.len() == expected.len()
                && got
                    .bytes()
                    .zip(expected.bytes())
                    .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                    == 0
        })
}

fn agent_version_token(secret: &ClusterSecret, peer_id: &PeerId) -> String {
    use std::fmt::Write as _;
    secret
        .attest(&peer_id.to_bytes())
        .iter()
        .fold(String::new(), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
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

    /// A peer is admitted only with a proof made from the cluster secret and
    /// its own peer id: not one made from another secret, not one lifted from
    /// another peer, and not none at all.
    #[test]
    fn membership_attestation_binds_the_secret_and_the_peer() {
        let secret = ClusterSecret::generate();
        let stranger_secret = ClusterSecret::generate();
        let (alice, mallory) = (PeerId::random(), PeerId::random());

        let alices = agent_version(Some(&secret), &alice);
        assert!(attestation_is_valid(&secret, &alice, &alices));

        // Another cluster's secret.
        let wrong_secret = agent_version(Some(&stranger_secret), &alice);
        assert!(!attestation_is_valid(&secret, &alice, &wrong_secret));
        // Alice's attestation, replayed by a different peer id.
        assert!(!attestation_is_valid(&secret, &mallory, &alices));
        // No attestation, and a malformed one.
        assert!(!attestation_is_valid(
            &secret,
            &alice,
            &agent_version(None, &alice)
        ));
        let truncated = &alices[..alices.len() - 2];
        assert!(!attestation_is_valid(&secret, &alice, truncated));
        assert!(!attestation_is_valid(&secret, &alice, "pmetal/1.0 attest="));
    }

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
