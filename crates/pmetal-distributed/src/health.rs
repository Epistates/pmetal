//! Heartbeat and health check system for peer monitoring.
//!
//! This module provides configurable health monitoring with:
//! - Periodic heartbeat probes
//! - Configurable timeouts and intervals
//! - Automatic peer removal on failure
//! - Health state tracking per peer

use libp2p::PeerId;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Default heartbeat interval.
pub const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// Default heartbeat timeout.
pub const DEFAULT_HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(15);

/// Maximum consecutive failures before marking peer as unhealthy.
pub const DEFAULT_MAX_FAILURES: u32 = 3;

/// Health check configuration.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Interval between heartbeat probes.
    pub heartbeat_interval: Duration,
    /// Timeout for a single heartbeat probe.
    pub heartbeat_timeout: Duration,
    /// Maximum consecutive failures before peer is considered unhealthy.
    pub max_consecutive_failures: u32,
    /// Whether to automatically remove unhealthy peers.
    pub auto_remove_unhealthy: bool,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            heartbeat_timeout: DEFAULT_HEARTBEAT_TIMEOUT,
            max_consecutive_failures: DEFAULT_MAX_FAILURES,
            auto_remove_unhealthy: true,
        }
    }
}

impl HealthConfig {
    /// Create a config optimized for low-latency networks (Thunderbolt).
    pub fn low_latency() -> Self {
        Self {
            heartbeat_interval: Duration::from_millis(500),
            heartbeat_timeout: Duration::from_secs(2),
            max_consecutive_failures: 5,
            auto_remove_unhealthy: true,
        }
    }

    /// Create a config for high-latency networks (WiFi).
    pub fn high_latency() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(10),
            heartbeat_timeout: Duration::from_secs(30),
            max_consecutive_failures: 3,
            auto_remove_unhealthy: true,
        }
    }
}

/// Health status of a peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Peer is healthy and responding.
    Healthy,
    /// Peer has missed some heartbeats but not yet unhealthy.
    Degraded,
    /// Peer is not responding.
    Unhealthy,
    /// Peer status is unknown (newly added).
    Unknown,
}

impl HealthStatus {
    /// Check if the peer is considered operational.
    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded | Self::Unknown)
    }
}

/// Health state for a single peer.
#[derive(Debug, Clone)]
pub struct PeerHealth {
    /// Current health status.
    pub status: HealthStatus,
    /// Last successful heartbeat time.
    pub last_heartbeat: Option<Instant>,
    /// Number of consecutive failures.
    pub consecutive_failures: u32,
    /// Round-trip latency of last successful probe.
    pub last_latency: Option<Duration>,
    /// Rolling average latency (exponential moving average).
    pub avg_latency: Option<Duration>,
    /// Total successful heartbeats.
    pub total_success: u64,
    /// Total failed heartbeats.
    pub total_failures: u64,
}

impl Default for PeerHealth {
    fn default() -> Self {
        Self {
            status: HealthStatus::Unknown,
            last_heartbeat: None,
            consecutive_failures: 0,
            last_latency: None,
            avg_latency: None,
            total_success: 0,
            total_failures: 0,
        }
    }
}

impl PeerHealth {
    /// Record a successful heartbeat.
    pub fn record_success(&mut self, latency: Duration, max_failures: u32) {
        self.last_heartbeat = Some(Instant::now());
        self.consecutive_failures = 0;
        self.last_latency = Some(latency);
        self.total_success += 1;

        // Update exponential moving average (alpha = 0.2)
        self.avg_latency = Some(match self.avg_latency {
            Some(avg) => {
                let alpha = 0.2;
                Duration::from_secs_f64(
                    alpha * latency.as_secs_f64() + (1.0 - alpha) * avg.as_secs_f64(),
                )
            }
            None => latency,
        });

        // Update status
        self.status = HealthStatus::Healthy;
    }

    /// Record a failed heartbeat.
    pub fn record_failure(&mut self, max_failures: u32) {
        self.consecutive_failures += 1;
        self.total_failures += 1;

        // Update status based on consecutive failures
        if self.consecutive_failures >= max_failures {
            self.status = HealthStatus::Unhealthy;
        } else if self.consecutive_failures > 0 {
            self.status = HealthStatus::Degraded;
        }
    }

    /// Get uptime ratio (success / total).
    pub fn uptime_ratio(&self) -> f64 {
        let total = self.total_success + self.total_failures;
        if total == 0 {
            1.0
        } else {
            self.total_success as f64 / total as f64
        }
    }
}

/// Events emitted by the health monitor.
#[derive(Debug, Clone)]
pub enum HealthEvent {
    /// Peer became healthy.
    PeerHealthy(PeerId),
    /// Peer became degraded.
    PeerDegraded(PeerId),
    /// Peer became unhealthy.
    PeerUnhealthy(PeerId),
    /// Peer was removed due to health failure.
    PeerRemoved(PeerId),
}

/// Shared health state for all peers.
#[derive(Debug, Default)]
pub struct HealthState {
    /// Health status per peer.
    peers: HashMap<PeerId, PeerHealth>,
}

impl HealthState {
    /// Add a new peer to track.
    pub fn add_peer(&mut self, peer_id: PeerId) {
        self.peers.entry(peer_id).or_default();
    }

    /// Remove a peer from tracking.
    pub fn remove_peer(&mut self, peer_id: &PeerId) -> Option<PeerHealth> {
        self.peers.remove(peer_id)
    }

    /// Get health status of a peer.
    pub fn get(&self, peer_id: &PeerId) -> Option<&PeerHealth> {
        self.peers.get(peer_id)
    }

    /// Get mutable health status of a peer.
    pub fn get_mut(&mut self, peer_id: &PeerId) -> Option<&mut PeerHealth> {
        self.peers.get_mut(peer_id)
    }

    /// Get all healthy peers.
    pub fn healthy_peers(&self) -> Vec<PeerId> {
        self.peers
            .iter()
            .filter(|(_, h)| h.status == HealthStatus::Healthy)
            .map(|(p, _)| *p)
            .collect()
    }

    /// Get all unhealthy peers.
    pub fn unhealthy_peers(&self) -> Vec<PeerId> {
        self.peers
            .iter()
            .filter(|(_, h)| h.status == HealthStatus::Unhealthy)
            .map(|(p, _)| *p)
            .collect()
    }

    /// Get all operational peers (healthy or degraded).
    pub fn operational_peers(&self) -> Vec<PeerId> {
        self.peers
            .iter()
            .filter(|(_, h)| h.status.is_operational())
            .map(|(p, _)| *p)
            .collect()
    }

    /// Get cluster health summary.
    pub fn summary(&self) -> HealthSummary {
        let mut healthy = 0;
        let mut degraded = 0;
        let mut unhealthy = 0;
        let mut unknown = 0;

        for health in self.peers.values() {
            match health.status {
                HealthStatus::Healthy => healthy += 1,
                HealthStatus::Degraded => degraded += 1,
                HealthStatus::Unhealthy => unhealthy += 1,
                HealthStatus::Unknown => unknown += 1,
            }
        }

        HealthSummary {
            total: self.peers.len(),
            healthy,
            degraded,
            unhealthy,
            unknown,
        }
    }
}

/// Summary of cluster health.
#[derive(Debug, Clone)]
pub struct HealthSummary {
    /// Total number of tracked peers.
    pub total: usize,
    /// Number of healthy peers.
    pub healthy: usize,
    /// Number of degraded peers.
    pub degraded: usize,
    /// Number of unhealthy peers.
    pub unhealthy: usize,
    /// Number of unknown status peers.
    pub unknown: usize,
}

impl HealthSummary {
    /// Check if cluster is healthy (all peers healthy or degraded).
    pub fn is_cluster_healthy(&self) -> bool {
        self.unhealthy == 0
    }

    /// Get the percentage of healthy peers.
    pub fn health_percentage(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            (self.healthy as f64 / self.total as f64) * 100.0
        }
    }
}

/// Thread-safe shared health state.
pub type SharedHealthState = Arc<RwLock<HealthState>>;

/// Create a new shared health state.
pub fn new_shared_health_state() -> SharedHealthState {
    Arc::new(RwLock::new(HealthState::default()))
}

/// Health monitor that runs periodic checks.
pub struct HealthMonitor {
    /// Configuration.
    config: HealthConfig,
    /// Shared health state.
    state: SharedHealthState,
    /// Event sender.
    event_tx: mpsc::Sender<HealthEvent>,
    /// Shutdown flag.
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl HealthMonitor {
    /// Create a new health monitor.
    pub fn new(
        config: HealthConfig,
        state: SharedHealthState,
        event_tx: mpsc::Sender<HealthEvent>,
    ) -> Self {
        Self {
            config,
            state,
            event_tx,
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Get the shared health state.
    pub fn state(&self) -> SharedHealthState {
        Arc::clone(&self.state)
    }

    /// Signal shutdown.
    pub fn shutdown(&self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Record a heartbeat response (called by the transport layer).
    pub async fn record_heartbeat(&self, peer_id: PeerId, latency: Duration) {
        let previous_status;
        let new_status;

        {
            let mut state = self.state.write();
            if let Some(health) = state.get_mut(&peer_id) {
                previous_status = health.status;
                health.record_success(latency, self.config.max_consecutive_failures);
                new_status = health.status;
            } else {
                return;
            }
        }

        // Emit event if status changed
        if previous_status != new_status && new_status == HealthStatus::Healthy {
            let _ = self.event_tx.send(HealthEvent::PeerHealthy(peer_id)).await;
        }

        debug!(
            "Heartbeat from {}: latency={:?}",
            peer_id.to_base58(),
            latency
        );
    }

    /// Record a heartbeat failure (called by the transport layer).
    pub async fn record_failure(&self, peer_id: PeerId) {
        let previous_status;
        let new_status;

        {
            let mut state = self.state.write();
            if let Some(health) = state.get_mut(&peer_id) {
                previous_status = health.status;
                health.record_failure(self.config.max_consecutive_failures);
                new_status = health.status;
            } else {
                return;
            }
        }

        // Emit events if status changed
        if previous_status != new_status {
            let event = match new_status {
                HealthStatus::Degraded => HealthEvent::PeerDegraded(peer_id),
                HealthStatus::Unhealthy => HealthEvent::PeerUnhealthy(peer_id),
                _ => return,
            };
            let _ = self.event_tx.send(event).await;
        }

        warn!(
            "Heartbeat failure for {}: consecutive_failures={}",
            peer_id.to_base58(),
            self.state
                .read()
                .get(&peer_id)
                .map(|h| h.consecutive_failures)
                .unwrap_or(0)
        );
    }

    /// Add a peer to monitor.
    pub fn add_peer(&self, peer_id: PeerId) {
        self.state.write().add_peer(peer_id);
        info!("Added peer {} to health monitoring", peer_id.to_base58());
    }

    /// Remove a peer from monitoring.
    pub async fn remove_peer(&self, peer_id: PeerId) {
        self.state.write().remove_peer(&peer_id);
        let _ = self.event_tx.send(HealthEvent::PeerRemoved(peer_id)).await;
        info!(
            "Removed peer {} from health monitoring",
            peer_id.to_base58()
        );
    }

    /// Get cluster health summary.
    pub fn summary(&self) -> HealthSummary {
        self.state.read().summary()
    }

    /// Check if a specific peer is healthy.
    pub fn is_peer_healthy(&self, peer_id: &PeerId) -> bool {
        self.state
            .read()
            .get(peer_id)
            .map(|h| h.status == HealthStatus::Healthy)
            .unwrap_or(false)
    }

    /// Get all healthy peer IDs.
    pub fn healthy_peers(&self) -> Vec<PeerId> {
        self.state.read().healthy_peers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_health_success() {
        let mut health = PeerHealth::default();
        assert_eq!(health.status, HealthStatus::Unknown);

        health.record_success(Duration::from_millis(10), 3);
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.consecutive_failures, 0);
        assert_eq!(health.total_success, 1);
    }

    #[test]
    fn test_peer_health_failure() {
        let mut health = PeerHealth::default();
        health.record_success(Duration::from_millis(10), 3);

        // First failure -> degraded
        health.record_failure(3);
        assert_eq!(health.status, HealthStatus::Degraded);
        assert_eq!(health.consecutive_failures, 1);

        // Second failure -> still degraded
        health.record_failure(3);
        assert_eq!(health.status, HealthStatus::Degraded);

        // Third failure -> unhealthy
        health.record_failure(3);
        assert_eq!(health.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_peer_health_recovery() {
        let mut health = PeerHealth::default();

        // Go to unhealthy
        for _ in 0..3 {
            health.record_failure(3);
        }
        assert_eq!(health.status, HealthStatus::Unhealthy);

        // One success should recover
        health.record_success(Duration::from_millis(10), 3);
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.consecutive_failures, 0);
    }

    #[test]
    fn test_health_state() {
        let mut state = HealthState::default();
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        state.add_peer(peer1);
        state.add_peer(peer2);

        // Mark peer1 as healthy
        if let Some(h) = state.get_mut(&peer1) {
            h.record_success(Duration::from_millis(10), 3);
        }

        // Mark peer2 as unhealthy
        if let Some(h) = state.get_mut(&peer2) {
            for _ in 0..3 {
                h.record_failure(3);
            }
        }

        let summary = state.summary();
        assert_eq!(summary.total, 2);
        assert_eq!(summary.healthy, 1);
        assert_eq!(summary.unhealthy, 1);
    }

    #[test]
    fn test_uptime_ratio() {
        let mut health = PeerHealth::default();

        // 3 successes, 1 failure = 75% uptime
        health.record_success(Duration::from_millis(10), 3);
        health.record_success(Duration::from_millis(10), 3);
        health.record_success(Duration::from_millis(10), 3);
        health.record_failure(3);

        assert!((health.uptime_ratio() - 0.75).abs() < 0.001);
    }
}
