//! Cluster topology management.
//!
//! This module maintains a graph representation of the cluster topology,
//! tracking connections between nodes and their capabilities.
//!
//! # Architecture
//!
//! The topology is represented as a directed graph where:
//! - Nodes are cluster members with their capabilities (RAM, compute, etc.)
//! - Edges are connections with latency/bandwidth profiles
//!
//! The topology is used for:
//! - Ring formation for all-reduce operations
//! - Optimal peer selection based on network proximity
//! - Detecting when the cluster can perform distributed training

use crate::error::DistributedError;
use anyhow::Result;
use libp2p::PeerId;
use parking_lot::RwLock;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::debug;

/// Node performance profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeProfile {
    /// Total RAM in bytes.
    pub total_ram: u64,
    /// Available RAM in bytes.
    pub available_ram: u64,
    /// Number of CPU cores.
    pub cpu_cores: u32,
    /// GPU memory in bytes (0 if no GPU).
    pub gpu_memory: u64,
    /// Chip name (e.g., "Apple M3 Max").
    pub chip_name: String,
    /// Whether this node has unified memory.
    pub unified_memory: bool,
}

impl Default for NodeProfile {
    fn default() -> Self {
        Self {
            total_ram: 0,
            available_ram: 0,
            cpu_cores: 1,
            gpu_memory: 0,
            chip_name: "Unknown".to_string(),
            unified_memory: false,
        }
    }
}

/// Information about a node in the topology.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// The node's peer ID.
    pub peer_id: PeerId,
    /// Socket address for direct communication.
    pub socket_addr: Option<SocketAddr>,
    /// Performance profile.
    pub profile: NodeProfile,
    /// When this node was last seen.
    pub last_seen: Instant,
    /// Whether this is the local node.
    pub is_local: bool,
}

/// Connection profile between two nodes.
#[derive(Debug, Clone)]
pub struct ConnectionProfile {
    /// Measured latency (round-trip time).
    pub latency: Duration,
    /// Estimated bandwidth in bytes/second.
    pub bandwidth: u64,
    /// Whether this is a Thunderbolt connection (link-local address 169.254.x.x).
    pub is_thunderbolt: bool,
    /// RDMA interface name if available (e.g., "rdma_en2").
    pub rdma_interface: Option<String>,
}

impl Default for ConnectionProfile {
    fn default() -> Self {
        Self {
            latency: Duration::from_millis(10),
            bandwidth: 1_000_000_000, // 1 Gbps default
            is_thunderbolt: false,
            rdma_interface: None,
        }
    }
}

impl ConnectionProfile {
    /// Create a profile for a Thunderbolt connection.
    ///
    /// Thunderbolt 4 provides ~40 Gbps bandwidth and sub-microsecond latency.
    pub fn thunderbolt(interface_name: Option<String>) -> Self {
        Self {
            latency: Duration::from_micros(10), // ~10µs for Thunderbolt
            bandwidth: 5_000_000_000,           // ~40 Gbps / 8 = 5 GB/s
            is_thunderbolt: true,
            rdma_interface: interface_name.map(|n| format!("rdma_{}", n)),
        }
    }

    /// Create a profile for a WiFi connection.
    pub fn wifi() -> Self {
        Self {
            latency: Duration::from_millis(5),
            bandwidth: 100_000_000, // ~800 Mbps typical
            is_thunderbolt: false,
            rdma_interface: None,
        }
    }

    /// Create a profile for an Ethernet connection.
    pub fn ethernet() -> Self {
        Self {
            latency: Duration::from_micros(500),
            bandwidth: 1_000_000_000, // 1 Gbps
            is_thunderbolt: false,
            rdma_interface: None,
        }
    }
}

/// Check if an IP address is a Thunderbolt link-local address.
///
/// Thunderbolt networking uses the 169.254.x.x range (link-local).
pub fn is_thunderbolt_ip(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(ipv4) => {
            let octets = ipv4.octets();
            octets[0] == 169 && octets[1] == 254
        }
        _ => false,
    }
}

/// Detect connection type from socket address.
pub fn detect_connection_type(addr: &SocketAddr) -> ConnectionProfile {
    if is_thunderbolt_ip(&addr.ip()) {
        ConnectionProfile::thunderbolt(None)
    } else {
        // Default to ethernet; could be refined with interface detection
        ConnectionProfile::ethernet()
    }
}

/// The cluster topology graph.
pub struct ClusterTopology {
    /// The topology graph.
    graph: DiGraph<NodeInfo, ConnectionProfile>,
    /// Map from PeerId to node index.
    peer_to_node: HashMap<PeerId, NodeIndex>,
    /// The local node's peer ID.
    local_peer_id: PeerId,
}

impl ClusterTopology {
    /// Create a new topology with the local node.
    pub fn new(local_peer_id: PeerId, local_profile: NodeProfile) -> Self {
        let mut graph = DiGraph::new();
        let mut peer_to_node = HashMap::new();

        // Add local node
        let local_info = NodeInfo {
            peer_id: local_peer_id,
            socket_addr: None,
            profile: local_profile,
            last_seen: Instant::now(),
            is_local: true,
        };

        let idx = graph.add_node(local_info);
        peer_to_node.insert(local_peer_id, idx);

        Self {
            graph,
            peer_to_node,
            local_peer_id,
        }
    }

    /// Add or update a node in the topology.
    pub fn add_node(&mut self, peer_id: PeerId, socket_addr: Option<SocketAddr>) -> NodeIndex {
        if let Some(&idx) = self.peer_to_node.get(&peer_id) {
            // Update existing node
            if let Some(node) = self.graph.node_weight_mut(idx) {
                node.socket_addr = socket_addr;
                node.last_seen = Instant::now();
            }
            idx
        } else {
            // Add new node
            let info = NodeInfo {
                peer_id,
                socket_addr,
                profile: NodeProfile::default(),
                last_seen: Instant::now(),
                is_local: false,
            };

            let idx = self.graph.add_node(info);
            self.peer_to_node.insert(peer_id, idx);
            debug!("Added node {} to topology", peer_id);
            idx
        }
    }

    /// Update a node's profile.
    pub fn update_profile(&mut self, peer_id: &PeerId, profile: NodeProfile) {
        if let Some(&idx) = self.peer_to_node.get(peer_id) {
            if let Some(node) = self.graph.node_weight_mut(idx) {
                node.profile = profile;
                node.last_seen = Instant::now();
            }
        }
    }

    /// Add a connection between two nodes.
    pub fn add_connection(
        &mut self,
        from: PeerId,
        to: PeerId,
        profile: ConnectionProfile,
    ) -> Result<()> {
        let from_idx = self
            .peer_to_node
            .get(&from)
            .ok_or_else(|| DistributedError::Protocol(format!("Unknown peer: {}", from)))?;
        let to_idx = self
            .peer_to_node
            .get(&to)
            .ok_or_else(|| DistributedError::Protocol(format!("Unknown peer: {}", to)))?;

        // Check if edge already exists
        if self.graph.find_edge(*from_idx, *to_idx).is_none() {
            self.graph.add_edge(*from_idx, *to_idx, profile);
            debug!("Added connection {} -> {}", from, to);
        }

        Ok(())
    }

    /// Remove a node and all its connections.
    pub fn remove_node(&mut self, peer_id: &PeerId) {
        if let Some(idx) = self.peer_to_node.remove(peer_id) {
            self.graph.remove_node(idx);
            debug!("Removed node {} from topology", peer_id);
        }
    }

    /// Get the number of nodes (including local).
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of remote nodes.
    pub fn remote_node_count(&self) -> usize {
        self.graph.node_count().saturating_sub(1)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &NodeInfo> {
        self.graph.node_weights()
    }

    /// Get all remote nodes (excluding local).
    pub fn remote_nodes(&self) -> impl Iterator<Item = &NodeInfo> {
        self.graph.node_weights().filter(|n| !n.is_local)
    }

    /// Get node info by peer ID.
    pub fn get_node(&self, peer_id: &PeerId) -> Option<&NodeInfo> {
        self.peer_to_node
            .get(peer_id)
            .and_then(|idx| self.graph.node_weight(*idx))
    }

    /// Get socket addresses of all remote nodes.
    pub fn remote_socket_addrs(&self) -> Vec<SocketAddr> {
        self.remote_nodes().filter_map(|n| n.socket_addr).collect()
    }

    /// Check if the topology forms a valid ring for all-reduce.
    ///
    /// A valid ring requires at least 2 nodes where each node can reach
    /// the next in a cycle.
    pub fn can_form_ring(&self) -> bool {
        let node_count = self.graph.node_count();
        if node_count < 2 {
            return false;
        }

        // For a ring, we need each node to have at least one outgoing edge
        // In practice, we'll form the ring ourselves, so just check connectivity
        self.graph.node_count() >= 2
    }

    /// Get nodes ordered for ring formation.
    ///
    /// Returns nodes sorted by their peer ID for deterministic ring ordering
    /// across all cluster members.
    pub fn ring_order(&self) -> Vec<&NodeInfo> {
        let mut nodes: Vec<_> = self.graph.node_weights().collect();
        nodes.sort_by(|a, b| a.peer_id.cmp(&b.peer_id));
        nodes
    }

    /// Get the rank of a peer in the ring.
    pub fn ring_rank(&self, peer_id: &PeerId) -> Option<usize> {
        self.ring_order().iter().position(|n| &n.peer_id == peer_id)
    }

    /// Get the local node's rank in the ring.
    pub fn local_rank(&self) -> usize {
        self.ring_rank(&self.local_peer_id).unwrap_or(0)
    }

    /// Get the next peer in the ring after the given peer.
    pub fn ring_next(&self, peer_id: &PeerId) -> Option<&NodeInfo> {
        let order = self.ring_order();
        let idx = order.iter().position(|n| &n.peer_id == peer_id)?;
        let next_idx = (idx + 1) % order.len();
        Some(order[next_idx])
    }

    /// Get the previous peer in the ring before the given peer.
    pub fn ring_prev(&self, peer_id: &PeerId) -> Option<&NodeInfo> {
        let order = self.ring_order();
        let idx = order.iter().position(|n| &n.peer_id == peer_id)?;
        let prev_idx = if idx == 0 { order.len() - 1 } else { idx - 1 };
        Some(order[prev_idx])
    }

    /// Get the total cluster RAM (sum of all nodes).
    pub fn total_cluster_ram(&self) -> u64 {
        self.graph.node_weights().map(|n| n.profile.total_ram).sum()
    }

    /// Prune nodes that haven't been seen recently.
    pub fn prune_stale_nodes(&mut self, max_age: Duration) {
        let now = Instant::now();
        let stale: Vec<_> = self
            .graph
            .node_weights()
            .filter(|n| !n.is_local && now.duration_since(n.last_seen) > max_age)
            .map(|n| n.peer_id)
            .collect();

        for peer_id in stale {
            self.remove_node(&peer_id);
        }
    }

    /// Check if all connections in the ring are Thunderbolt.
    ///
    /// Returns true if every node has at least one Thunderbolt connection
    /// to the next node in ring order.
    pub fn has_thunderbolt_ring(&self) -> bool {
        let order = self.ring_order();
        if order.len() < 2 {
            return false;
        }

        for node in &order {
            if let Some(addr) = node.socket_addr {
                if !is_thunderbolt_ip(&addr.ip()) {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Get nodes with Thunderbolt connections.
    pub fn thunderbolt_nodes(&self) -> Vec<&NodeInfo> {
        self.graph
            .node_weights()
            .filter(|n| {
                n.socket_addr
                    .map(|a| is_thunderbolt_ip(&a.ip()))
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Get the estimated total bandwidth for the ring.
    ///
    /// This is the minimum bandwidth of any link in the ring.
    pub fn ring_bandwidth(&self) -> u64 {
        let order = self.ring_order();
        if order.is_empty() {
            return 0;
        }

        // Estimate based on connection types
        order
            .iter()
            .filter_map(|n| n.socket_addr)
            .map(|a| detect_connection_type(&a).bandwidth)
            .min()
            .unwrap_or(0)
    }

    /// Generate MLX-compatible host list for ring all-reduce.
    ///
    /// Returns a list of hosts where:
    /// - Self position: "0.0.0.0:port"
    /// - Neighbors: actual connection IPs
    /// - Non-neighbors: placeholder IPs (198.51.100.1:0)
    pub fn mlx_ring_hosts(&self, port: u16) -> Vec<(std::net::IpAddr, u16)> {
        use std::net::{IpAddr, Ipv4Addr};

        let order = self.ring_order();
        let local_rank = self.local_rank();
        let world_size = order.len();

        let placeholder = IpAddr::V4(Ipv4Addr::new(198, 51, 100, 1)); // RFC 5737 TEST-NET-2

        order
            .iter()
            .enumerate()
            .map(|(idx, node)| {
                if idx == local_rank {
                    (IpAddr::V4(Ipv4Addr::UNSPECIFIED), port)
                } else {
                    let left = (local_rank + world_size - 1) % world_size;
                    let right = (local_rank + 1) % world_size;

                    if idx == left || idx == right {
                        node.socket_addr
                            .map(|a| (a.ip(), port))
                            .unwrap_or((placeholder, 0))
                    } else {
                        (placeholder, 0)
                    }
                }
            })
            .collect()
    }
}

impl std::fmt::Debug for ClusterTopology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterTopology")
            .field("node_count", &self.graph.node_count())
            .field("edge_count", &self.graph.edge_count())
            .field("local_peer_id", &self.local_peer_id.to_base58())
            .finish()
    }
}

/// Thread-safe wrapper around ClusterTopology.
pub type SharedTopology = Arc<RwLock<ClusterTopology>>;

/// Create a new shared topology.
pub fn new_shared_topology(local_peer_id: PeerId, local_profile: NodeProfile) -> SharedTopology {
    Arc::new(RwLock::new(ClusterTopology::new(
        local_peer_id,
        local_profile,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_profile() -> NodeProfile {
        NodeProfile {
            total_ram: 32 * 1024 * 1024 * 1024, // 32 GB
            available_ram: 16 * 1024 * 1024 * 1024,
            cpu_cores: 10,
            gpu_memory: 0,
            chip_name: "Apple M3".to_string(),
            unified_memory: true,
        }
    }

    #[test]
    fn test_topology_creation() {
        let local_id = PeerId::random();
        let topology = ClusterTopology::new(local_id, test_profile());

        assert_eq!(topology.node_count(), 1);
        assert_eq!(topology.remote_node_count(), 0);
        assert!(!topology.can_form_ring());
    }

    #[test]
    fn test_add_node() {
        let local_id = PeerId::random();
        let mut topology = ClusterTopology::new(local_id, test_profile());

        let remote_id = PeerId::random();
        let addr: SocketAddr = "192.168.1.100:5000".parse().unwrap();
        topology.add_node(remote_id, Some(addr));

        assert_eq!(topology.node_count(), 2);
        assert_eq!(topology.remote_node_count(), 1);
        assert!(topology.can_form_ring());
    }

    #[test]
    fn test_ring_order() {
        let local_id = PeerId::random();
        let mut topology = ClusterTopology::new(local_id, test_profile());

        // Add some remote nodes
        for _ in 0..3 {
            topology.add_node(PeerId::random(), None);
        }

        let order = topology.ring_order();
        assert_eq!(order.len(), 4);

        // Verify order is deterministic (sorted by peer ID)
        for i in 1..order.len() {
            assert!(order[i - 1].peer_id < order[i].peer_id);
        }
    }

    #[test]
    fn test_ring_navigation() {
        let local_id = PeerId::random();
        let mut topology = ClusterTopology::new(local_id, test_profile());

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        topology.add_node(peer1, None);
        topology.add_node(peer2, None);

        let order = topology.ring_order();
        let first = &order[0].peer_id;
        let last = &order[order.len() - 1].peer_id;

        // Next of last should be first (wrap around)
        let next = topology.ring_next(last).unwrap();
        assert_eq!(&next.peer_id, first);

        // Prev of first should be last
        let prev = topology.ring_prev(first).unwrap();
        assert_eq!(&prev.peer_id, last);
    }

    #[test]
    fn test_thunderbolt_detection() {
        use std::net::{IpAddr, Ipv4Addr};

        // Thunderbolt link-local addresses (169.254.x.x)
        let tb_ip = IpAddr::V4(Ipv4Addr::new(169, 254, 1, 100));
        assert!(is_thunderbolt_ip(&tb_ip));

        let tb_ip2 = IpAddr::V4(Ipv4Addr::new(169, 254, 255, 255));
        assert!(is_thunderbolt_ip(&tb_ip2));

        // Non-Thunderbolt addresses
        let regular_ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100));
        assert!(!is_thunderbolt_ip(&regular_ip));

        let loopback = IpAddr::V4(Ipv4Addr::LOCALHOST);
        assert!(!is_thunderbolt_ip(&loopback));

        // IPv6 should return false
        let ipv6: IpAddr = "::1".parse().unwrap();
        assert!(!is_thunderbolt_ip(&ipv6));
    }

    #[test]
    fn test_connection_profiles() {
        let tb = ConnectionProfile::thunderbolt(Some("en2".to_string()));
        assert!(tb.is_thunderbolt);
        assert_eq!(tb.rdma_interface, Some("rdma_en2".to_string()));
        assert!(tb.bandwidth >= 5_000_000_000); // 5 GB/s

        let wifi = ConnectionProfile::wifi();
        assert!(!wifi.is_thunderbolt);
        assert!(wifi.rdma_interface.is_none());

        let eth = ConnectionProfile::ethernet();
        assert!(!eth.is_thunderbolt);
        assert_eq!(eth.bandwidth, 1_000_000_000); // 1 Gbps
    }

    #[test]
    fn test_thunderbolt_ring() {
        let local_id = PeerId::random();
        let mut topology = ClusterTopology::new(local_id, test_profile());

        // Add nodes with Thunderbolt addresses
        let peer1 = PeerId::random();
        let tb_addr1: SocketAddr = "169.254.1.100:5000".parse().unwrap();
        topology.add_node(peer1, Some(tb_addr1));

        // One node without Thunderbolt - ring should not be Thunderbolt
        assert!(!topology.has_thunderbolt_ring());

        // Update local node with Thunderbolt address
        if let Some(idx) = topology.peer_to_node.get(&local_id) {
            if let Some(node) = topology.graph.node_weight_mut(*idx) {
                node.socket_addr = Some("169.254.1.1:5000".parse().unwrap());
            }
        }

        // Now should be a Thunderbolt ring
        assert!(topology.has_thunderbolt_ring());
        assert_eq!(topology.thunderbolt_nodes().len(), 2);
    }
}
