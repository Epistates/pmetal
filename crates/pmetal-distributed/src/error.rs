//! Comprehensive error types for distributed operations.
//!
//! Modeled after Burn's GlobalCollectiveError for
//! error classification and handling.

use libp2p::PeerId;
use std::net::SocketAddr;
use thiserror::Error;

/// Errors that can occur during distributed operations.
#[derive(Error, Debug)]
pub enum DistributedError {
    // === IO & Network Errors ===
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Connection failed to peer at {addr}: {reason}")]
    ConnectionFailed { addr: SocketAddr, reason: String },

    #[error("Connection timeout to peer at {0} after {1:?}")]
    ConnectionTimeout(SocketAddr, std::time::Duration),

    #[error("Connection refused by peer at {0}")]
    ConnectionRefused(SocketAddr),

    #[error("Max retries ({max_retries}) exceeded connecting to {addr}")]
    MaxRetriesExceeded { addr: SocketAddr, max_retries: u32 },

    // === Peer Management Errors ===
    #[error("Peer lost: {0}")]
    PeerLost(PeerId),

    #[error("Peer {peer} sent incoherent data: expected {expected} bytes, got {actual}")]
    PeerIncoherentData {
        peer: PeerId,
        expected: usize,
        actual: usize,
    },

    #[error("Unknown peer: {0}")]
    UnknownPeer(PeerId),

    #[error("Peer {0} is unreachable")]
    PeerUnreachable(PeerId),

    #[error("Peer {peer} timed out during {operation}")]
    PeerTimeout { peer: PeerId, operation: String },

    // === Collective Operation Errors ===
    #[error("All-reduce called before registration")]
    AllReduceBeforeRegister,

    #[error("Node not registered when finish called")]
    NotRegisteredOnFinish,

    #[error("Double registration attempted")]
    DoubleRegister,

    #[error("Registration parameters mismatch: {0}")]
    RegisterParamsMismatch(String),

    #[error("All-reduce parameters mismatch: local has {local} elements, expected {expected}")]
    AllReduceParamsMismatch { local: usize, expected: usize },

    #[error("Ring reduce impossible: need at least 2 nodes, have {0}")]
    RingReduceImpossible(usize),

    #[error("Buffer alignment error: expected {expected}-byte alignment, got {actual}")]
    BufferAlignment { expected: usize, actual: usize },

    #[error("Buffer size error: expected multiple of {expected}, got {actual}")]
    BufferSize { expected: usize, actual: usize },

    // === Topology Errors ===
    #[error("Cannot form ring: insufficient nodes ({have} < {need})")]
    InsufficientNodes { have: usize, need: usize },

    #[error("Ring not established")]
    RingNotEstablished,

    #[error("Topology changed during operation")]
    TopologyChanged,

    #[error("No route to peer {0}")]
    NoRoute(PeerId),

    // === Election Errors ===
    #[error("Election timeout after {0:?}")]
    ElectionTimeout(std::time::Duration),

    #[error("Split brain detected: multiple masters ({0:?})")]
    SplitBrain(Vec<PeerId>),

    #[error("No master elected")]
    NoMaster,

    #[error("Master unreachable: {0}")]
    MasterUnreachable(PeerId),

    // === Protocol Errors ===
    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Invalid message: expected {expected}, got {actual}")]
    InvalidMessage { expected: String, actual: String },

    #[error("First message was not init")]
    FirstMsgNotInit,

    #[error("Version mismatch: local {local}, remote {remote}")]
    VersionMismatch { local: String, remote: String },

    #[error("Namespace mismatch: expected {expected}, got {actual}")]
    NamespaceMismatch { expected: String, actual: String },

    // === Configuration Errors ===
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Invalid address: {0}")]
    InvalidAddress(String),

    #[error("Port {0} already in use")]
    PortInUse(u16),

    // === Serialization Errors ===
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    // === Health Check Errors ===
    #[error("Health check failed for peer {peer}: {reason}")]
    HealthCheckFailed { peer: PeerId, reason: String },

    #[error("Heartbeat timeout for peer {0} after {1:?}")]
    HeartbeatTimeout(PeerId, std::time::Duration),

    // === Shutdown Errors ===
    #[error("Shutdown in progress")]
    ShuttingDown,

    #[error("Operation cancelled")]
    Cancelled,
}

impl DistributedError {
    /// Check if this error is recoverable (can retry).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::ConnectionTimeout(_, _)
                | Self::PeerTimeout { .. }
                | Self::HeartbeatTimeout(_, _)
                | Self::ElectionTimeout(_)
                | Self::TopologyChanged
        )
    }

    /// Check if this error indicates a peer failure.
    pub fn is_peer_failure(&self) -> bool {
        matches!(
            self,
            Self::PeerLost(_)
                | Self::PeerUnreachable(_)
                | Self::PeerTimeout { .. }
                | Self::PeerIncoherentData { .. }
                | Self::HealthCheckFailed { .. }
                | Self::HeartbeatTimeout(_, _)
        )
    }

    /// Check if this error is fatal (cannot continue).
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::SplitBrain(_)
                | Self::ShuttingDown
                | Self::Cancelled
                | Self::VersionMismatch { .. }
                | Self::NamespaceMismatch { .. }
        )
    }

    /// Get the peer ID associated with this error, if any.
    pub fn peer_id(&self) -> Option<&PeerId> {
        match self {
            Self::PeerLost(p)
            | Self::UnknownPeer(p)
            | Self::PeerUnreachable(p)
            | Self::NoRoute(p)
            | Self::MasterUnreachable(p)
            | Self::HeartbeatTimeout(p, _) => Some(p),
            Self::PeerTimeout { peer, .. }
            | Self::PeerIncoherentData { peer, .. }
            | Self::HealthCheckFailed { peer, .. } => Some(peer),
            _ => None,
        }
    }
}

/// Result type alias for distributed operations.
pub type DistributedResult<T> = Result<T, DistributedError>;
