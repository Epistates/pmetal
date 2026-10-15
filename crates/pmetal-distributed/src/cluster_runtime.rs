//! Shared helpers for the `pmetal cluster …` CLI surface.
//!
//! Every cluster subcommand follows the same pattern: spin up a short-lived
//! [`AutoDiscoveryBackend`], inspect or use it, then exit. This module
//! centralises the boilerplate so commands stay a few dozen lines each.

use crate::auto::{AutoDiscoveryBackend, AutoDiscoveryConfig};
use crate::fabric::InterfaceKind;
use crate::topology::ClusterTopology;
use anyhow::Result;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

/// One row of `pmetal cluster status`: a peer, its primary fabric, and
/// every additional fabric path we know about.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerRow {
    pub peer_id: String,
    pub is_local: bool,
    pub primary_addr: Option<SocketAddr>,
    pub primary_fabric: InterfaceKind,
    /// Every (addr, kind) pair we know for this peer, ranked best-first.
    pub all_paths: Vec<(SocketAddr, InterfaceKind)>,
}

/// Snapshot rendered by `pmetal cluster status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatus {
    pub local_peer_id: String,
    pub local_rank: usize,
    pub world_size: usize,
    pub peers: Vec<PeerRow>,
    /// Local NICs and their classified kinds.
    pub local_interfaces: Vec<(String, InterfaceKind, Vec<SocketAddr>)>,
    pub has_thunderbolt_ring: bool,
}

impl ClusterStatus {
    /// Render to a human-friendly table for terminal output.
    pub fn render_table(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Local peer: {}  (rank {}/{})\n",
            self.local_peer_id, self.local_rank, self.world_size
        ));
        s.push_str(&format!(
            "Thunderbolt ring: {}\n\n",
            if self.has_thunderbolt_ring {
                "yes"
            } else {
                "no"
            }
        ));

        s.push_str("Local interfaces:\n");
        for (name, kind, addrs) in &self.local_interfaces {
            let addrs_str = addrs
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            s.push_str(&format!(
                "  {:<10} {:<12}  {}\n",
                name,
                kind.tag(),
                addrs_str
            ));
        }

        s.push_str("\nCluster peers:\n");
        s.push_str(&format!(
            "  {:<46} {:<6} {:<22} {:<12} paths\n",
            "peer-id", "local", "primary-addr", "fabric"
        ));
        for p in &self.peers {
            s.push_str(&format!(
                "  {:<46} {:<6} {:<22} {:<12} {}\n",
                truncate(&p.peer_id, 46),
                if p.is_local { "yes" } else { "no" },
                p.primary_addr.map(|a| a.to_string()).unwrap_or_default(),
                p.primary_fabric.tag(),
                p.all_paths.len(),
            ));
        }
        s
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

/// Build a [`ClusterStatus`] snapshot from a backend's topology + fabric.
pub fn snapshot_status(backend: &AutoDiscoveryBackend) -> ClusterStatus {
    let topology = backend.topology();
    let fabric = backend.fabric();
    let topo: parking_lot::RwLockReadGuard<'_, ClusterTopology> = topology.read();

    let local_peer_id = topo
        .nodes()
        .find(|n| n.is_local)
        .map(|n| n.peer_id.to_base58())
        .unwrap_or_default();
    let local_rank = topo.local_rank();
    let world_size = topo.node_count();

    let mut peers = Vec::with_capacity(world_size);
    for n in topo.ring_order() {
        let primary = n.best_addr();
        peers.push(PeerRow {
            peer_id: n.peer_id.to_base58(),
            is_local: n.is_local,
            primary_addr: primary.map(|(a, _)| a),
            primary_fabric: primary.map(|(_, k)| k).unwrap_or(InterfaceKind::Unknown),
            all_paths: n.addrs.clone(),
        });
    }

    let local_interfaces: Vec<(String, InterfaceKind, Vec<SocketAddr>)> = fabric
        .interfaces()
        .iter()
        .map(|i| {
            let addrs: Vec<SocketAddr> = i.addrs.iter().map(|ip| SocketAddr::new(*ip, 0)).collect();
            (i.name.clone(), i.kind, addrs)
        })
        .collect();

    let has_thunderbolt_ring = topo.has_thunderbolt_ring();

    ClusterStatus {
        local_peer_id,
        local_rank,
        world_size,
        peers,
        local_interfaces,
        has_thunderbolt_ring,
    }
}

/// Result of a single all-reduce micro-benchmark step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchSample {
    pub bytes: usize,
    pub elapsed: Duration,
    /// Effective bandwidth = bytes / elapsed. The ring all-reduce algorithm
    /// touches each byte 2(N-1)/N times across the slowest link, so users
    /// who want a per-link rate (comparable to a TB/Ethernet spec) should
    /// use [`Self::link_gbps`] instead of [`Self::gbps`].
    pub raw_bytes_per_sec: f64,
    /// World size at the time of the sample — needed for `link_gbps`.
    pub world_size: usize,
}

impl BenchSample {
    /// Effective all-reduce bandwidth: raw bytes/sec × 8.
    pub fn gbps(&self) -> f64 {
        self.raw_bytes_per_sec * 8.0 / 1_000_000_000.0
    }

    /// Per-link bandwidth: scales raw bytes/sec by 2(N-1)/N to back out the
    /// ring's redundancy factor. For N=2 this is a no-op (factor = 1.0);
    /// for N=4 the factor is 1.5×, so a 30 Gbps all-reduce throughput
    /// implies ~45 Gbps per link.
    pub fn link_gbps(&self) -> f64 {
        if self.world_size < 2 {
            return self.gbps();
        }
        let n = self.world_size as f64;
        let factor = 2.0 * (n - 1.0) / n;
        self.gbps() * factor
    }
}

/// Run a small all-reduce benchmark on `backend`. `payload_mb` controls the
/// per-iteration buffer size in MiB; `iters` is how many runs to time.
///
/// Returns one [`BenchSample`] per iteration plus the median bandwidth.
pub async fn run_allreduce_bench(
    backend: &dyn crate::DistributedBackend,
    payload_mb: usize,
    iters: usize,
) -> Result<Vec<BenchSample>> {
    let bytes = payload_mb * 1024 * 1024;
    let mut buf = vec![1.0_f32; bytes / 4];
    let world_size = backend.world_size();

    let mut samples = Vec::with_capacity(iters);
    for i in 0..iters {
        // Reset to ones so each iteration measures the same workload.
        buf.fill(1.0);

        let start = std::time::Instant::now();
        backend
            .all_reduce_f32(&mut buf, crate::ReduceOp::Sum)
            .await?;
        let elapsed = start.elapsed();

        let raw_bytes_per_sec = bytes as f64 / elapsed.as_secs_f64();
        samples.push(BenchSample {
            bytes,
            elapsed,
            raw_bytes_per_sec,
            world_size,
        });

        tracing::debug!(
            "bench iter {}/{}: {} bytes in {:?} ⇒ {:.2} Gbps",
            i + 1,
            iters,
            bytes,
            elapsed,
            samples.last().unwrap().gbps()
        );
    }

    Ok(samples)
}

/// Median Gbps across a sample set.
pub fn median_gbps(samples: &[BenchSample]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = samples.iter().map(|s| s.gbps()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Spin up an [`AutoDiscoveryBackend`], wait for `min_peers` peers, then
/// establish the ring. Wrapper used by every cluster subcommand that
/// needs a connected backend.
pub async fn join_cluster(
    cfg: AutoDiscoveryConfig,
    min_peers: usize,
    timeout: Duration,
) -> Result<Arc<AutoDiscoveryBackend>> {
    let backend = Arc::new(AutoDiscoveryBackend::with_config(cfg).await?);

    if min_peers > 0 {
        backend.wait_for_peers(min_peers, timeout).await?;
        backend.establish_ring().await?;
    }
    Ok(backend)
}

/// Iterate every peer's preferred fabric and aggregate counts.
pub fn count_fabrics_in_ring(
    topo: &ClusterTopology,
) -> std::collections::BTreeMap<InterfaceKind, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for n in topo.ring_order() {
        let kind = n
            .best_addr()
            .map(|(_, k)| k)
            .unwrap_or(InterfaceKind::Unknown);
        *counts.entry(kind).or_insert(0) += 1;
    }
    counts
}

/// Resolve a peer's display name from a `PeerId`. Currently just a base58
/// truncation; reserved as a hook for future identity-system integration.
pub fn short_peer_name(peer_id: &PeerId) -> String {
    let s = peer_id.to_base58();
    if s.len() <= 12 {
        s
    } else {
        format!("{}…{}", &s[..6], &s[s.len() - 4..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_short() {
        assert_eq!(truncate("abc", 10), "abc");
    }

    #[test]
    fn truncate_long() {
        assert_eq!(truncate("abcdefghij", 5).chars().count(), 5);
    }

    #[test]
    fn median_gbps_picks_middle() {
        let s = |gbps: f64| BenchSample {
            bytes: 1,
            elapsed: Duration::from_secs(1),
            raw_bytes_per_sec: gbps * 1_000_000_000.0 / 8.0,
            world_size: 2,
        };
        let samples = vec![s(1.0), s(5.0), s(10.0), s(100.0), s(20.0)];
        let m = median_gbps(&samples);
        assert!((m - 10.0).abs() < 0.001, "median was {}", m);
    }

    #[test]
    fn link_gbps_normalizes_by_world_size() {
        let s = |gbps: f64, ws: usize| BenchSample {
            bytes: 1,
            elapsed: Duration::from_secs(1),
            raw_bytes_per_sec: gbps * 1_000_000_000.0 / 8.0,
            world_size: ws,
        };
        // N=2: factor = 2*1/2 = 1.0 — link_gbps == gbps
        assert!((s(10.0, 2).link_gbps() - 10.0).abs() < 1e-6);
        // N=4: factor = 2*3/4 = 1.5
        assert!((s(20.0, 4).link_gbps() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn local_fabric_snapshot_includes_loopback() {
        let f = crate::fabric::probe_local_fabric();
        assert!(
            f.interfaces()
                .iter()
                .any(|i| i.kind == InterfaceKind::Loopback)
        );
    }
}
