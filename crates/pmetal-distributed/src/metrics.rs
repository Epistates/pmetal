//! Metrics and observability for distributed operations.
//!
//! Provides comprehensive metrics tracking for:
//! - Collective operations (all-reduce, reduce, broadcast)
//! - Network performance (latency, bandwidth, throughput)
//! - Peer health and connectivity
//! - Compression efficiency
//!
//! Metrics can be exposed via callbacks or collected for monitoring systems.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Maximum number of samples to keep in rolling windows.
const MAX_SAMPLES: usize = 1000;

/// Metrics configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable detailed per-operation metrics.
    pub detailed_ops: bool,
    /// Enable network latency tracking.
    pub track_latency: bool,
    /// Enable bandwidth tracking.
    pub track_bandwidth: bool,
    /// Rolling window size for averages.
    pub window_size: usize,
    /// Callback interval for metric updates.
    pub callback_interval: Duration,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            detailed_ops: true,
            track_latency: true,
            track_bandwidth: true,
            window_size: 100,
            callback_interval: Duration::from_secs(10),
        }
    }
}

/// Counter metric (monotonically increasing).
#[derive(Debug, Default)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Increment the counter.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Add a value to the counter.
    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the counter (for testing).
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

/// Gauge metric (can go up or down).
#[derive(Debug, Default)]
pub struct Gauge {
    value: AtomicU64,
}

impl Gauge {
    /// Set the gauge value.
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Increment the gauge.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the gauge.
    pub fn dec(&self) {
        self.value.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// Histogram for tracking distributions.
#[derive(Debug)]
pub struct Histogram {
    samples: RwLock<VecDeque<f64>>,
    max_samples: usize,
    sum: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    /// Create a new histogram.
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: RwLock::new(VecDeque::with_capacity(max_samples)),
            max_samples,
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record a sample.
    pub fn observe(&self, value: f64) {
        let value_bits = value.to_bits();
        self.sum.fetch_add(value_bits, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        let mut samples = self.samples.write();
        if samples.len() >= self.max_samples {
            samples.pop_front();
        }
        samples.push_back(value);
    }

    /// Get the mean of recent samples.
    pub fn mean(&self) -> f64 {
        let samples = self.samples.read();
        if samples.is_empty() {
            return 0.0;
        }
        samples.iter().sum::<f64>() / samples.len() as f64
    }

    /// Get the p50 (median) of recent samples.
    pub fn p50(&self) -> f64 {
        self.percentile(0.50)
    }

    /// Get the p95 of recent samples.
    pub fn p95(&self) -> f64 {
        self.percentile(0.95)
    }

    /// Get the p99 of recent samples.
    pub fn p99(&self) -> f64 {
        self.percentile(0.99)
    }

    /// Get a percentile of recent samples.
    pub fn percentile(&self, p: f64) -> f64 {
        let samples = self.samples.read();
        if samples.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((sorted.len() as f64 * p) as usize).min(sorted.len() - 1);
        sorted[idx]
    }

    /// Get the total count.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new(MAX_SAMPLES)
    }
}

/// Operation metrics.
#[derive(Debug)]
pub struct OperationMetrics {
    /// Number of operations completed.
    pub completed: Counter,
    /// Number of operations failed.
    pub failed: Counter,
    /// Duration histogram (milliseconds).
    pub duration_ms: Histogram,
    /// Bytes processed.
    pub bytes_processed: Counter,
}

impl Default for OperationMetrics {
    fn default() -> Self {
        Self {
            completed: Counter::default(),
            failed: Counter::default(),
            duration_ms: Histogram::new(MAX_SAMPLES),
            bytes_processed: Counter::default(),
        }
    }
}

/// Network metrics.
#[derive(Debug, Default)]
pub struct NetworkMetrics {
    /// Bytes sent.
    pub bytes_sent: Counter,
    /// Bytes received.
    pub bytes_received: Counter,
    /// Messages sent.
    pub messages_sent: Counter,
    /// Messages received.
    pub messages_received: Counter,
    /// Send latency histogram (microseconds).
    pub send_latency_us: Histogram,
    /// Receive latency histogram (microseconds).
    pub recv_latency_us: Histogram,
    /// Connection errors.
    pub connection_errors: Counter,
    /// Reconnection attempts.
    pub reconnections: Counter,
}

/// Peer metrics.
#[derive(Debug, Default)]
pub struct PeerMetrics {
    /// Number of connected peers.
    pub connected_peers: Gauge,
    /// Number of healthy peers.
    pub healthy_peers: Gauge,
    /// Number of degraded peers.
    pub degraded_peers: Gauge,
    /// Number of unhealthy peers.
    pub unhealthy_peers: Gauge,
    /// Total peer connections ever.
    pub total_connections: Counter,
    /// Total peer disconnections.
    pub total_disconnections: Counter,
}

/// Compression metrics.
#[derive(Debug, Default)]
pub struct CompressionMetrics {
    /// Bytes before compression.
    pub bytes_before: Counter,
    /// Bytes after compression.
    pub bytes_after: Counter,
    /// Compression time (microseconds).
    pub compression_time_us: Histogram,
    /// Decompression time (microseconds).
    pub decompression_time_us: Histogram,
}

impl CompressionMetrics {
    /// Get the overall compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        let before = self.bytes_before.get() as f64;
        let after = self.bytes_after.get() as f64;
        if after == 0.0 {
            1.0
        } else {
            before / after
        }
    }
}

/// Election metrics.
#[derive(Debug, Default)]
pub struct ElectionMetrics {
    /// Number of elections started.
    pub elections_started: Counter,
    /// Number of elections completed successfully.
    pub elections_completed: Counter,
    /// Number of election timeouts.
    pub election_timeouts: Counter,
    /// Time as master (seconds).
    pub time_as_master_secs: Counter,
    /// Time as follower (seconds).
    pub time_as_follower_secs: Counter,
}

/// All distributed metrics.
#[derive(Debug, Default)]
pub struct DistributedMetrics {
    /// All-reduce operation metrics.
    pub all_reduce: OperationMetrics,
    /// Reduce operation metrics.
    pub reduce: OperationMetrics,
    /// Broadcast operation metrics.
    pub broadcast: OperationMetrics,
    /// Barrier operation metrics.
    pub barrier: OperationMetrics,
    /// Network metrics.
    pub network: NetworkMetrics,
    /// Peer metrics.
    pub peer: PeerMetrics,
    /// Compression metrics.
    pub compression: CompressionMetrics,
    /// Election metrics.
    pub election: ElectionMetrics,
    /// Start time for uptime calculation.
    start_time: RwLock<Option<Instant>>,
}

impl DistributedMetrics {
    /// Create a new metrics instance.
    pub fn new() -> Self {
        let mut metrics = Self::default();
        *metrics.start_time.write() = Some(Instant::now());
        metrics
    }

    /// Get uptime in seconds.
    pub fn uptime_secs(&self) -> u64 {
        self.start_time
            .read()
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0)
    }

    /// Get a snapshot of key metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            uptime_secs: self.uptime_secs(),
            all_reduce_completed: self.all_reduce.completed.get(),
            all_reduce_failed: self.all_reduce.failed.get(),
            all_reduce_avg_ms: self.all_reduce.duration_ms.mean(),
            all_reduce_p99_ms: self.all_reduce.duration_ms.p99(),
            bytes_sent: self.network.bytes_sent.get(),
            bytes_received: self.network.bytes_received.get(),
            connected_peers: self.peer.connected_peers.get(),
            healthy_peers: self.peer.healthy_peers.get(),
            compression_ratio: self.compression.compression_ratio(),
        }
    }

    /// Reset all metrics (for testing).
    pub fn reset(&self) {
        self.all_reduce.completed.reset();
        self.all_reduce.failed.reset();
        self.network.bytes_sent.reset();
        self.network.bytes_received.reset();
    }
}

/// Snapshot of key metrics for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Uptime in seconds.
    pub uptime_secs: u64,
    /// All-reduce operations completed.
    pub all_reduce_completed: u64,
    /// All-reduce operations failed.
    pub all_reduce_failed: u64,
    /// Average all-reduce duration (ms).
    pub all_reduce_avg_ms: f64,
    /// P99 all-reduce duration (ms).
    pub all_reduce_p99_ms: f64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Number of connected peers.
    pub connected_peers: u64,
    /// Number of healthy peers.
    pub healthy_peers: u64,
    /// Overall compression ratio.
    pub compression_ratio: f64,
}

/// Thread-safe shared metrics.
pub type SharedMetrics = Arc<DistributedMetrics>;

/// Create a new shared metrics instance.
pub fn new_shared_metrics() -> SharedMetrics {
    Arc::new(DistributedMetrics::new())
}

/// RAII guard for timing operations.
pub struct TimingGuard<'a> {
    histogram: &'a Histogram,
    start: Instant,
    multiplier: f64,
}

impl<'a> TimingGuard<'a> {
    /// Create a new timing guard that records in milliseconds.
    pub fn new_ms(histogram: &'a Histogram) -> Self {
        Self {
            histogram,
            start: Instant::now(),
            multiplier: 1.0,
        }
    }

    /// Create a new timing guard that records in microseconds.
    pub fn new_us(histogram: &'a Histogram) -> Self {
        Self {
            histogram,
            start: Instant::now(),
            multiplier: 1000.0,
        }
    }
}

impl<'a> Drop for TimingGuard<'a> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64() * 1000.0 * self.multiplier;
        self.histogram.observe(elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let counter = Counter::default();
        assert_eq!(counter.get(), 0);

        counter.inc();
        assert_eq!(counter.get(), 1);

        counter.add(5);
        assert_eq!(counter.get(), 6);
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::default();
        assert_eq!(gauge.get(), 0);

        gauge.set(10);
        assert_eq!(gauge.get(), 10);

        gauge.inc();
        assert_eq!(gauge.get(), 11);

        gauge.dec();
        assert_eq!(gauge.get(), 10);
    }

    #[test]
    fn test_histogram() {
        let hist = Histogram::new(100);

        for i in 1..=100 {
            hist.observe(i as f64);
        }

        assert_eq!(hist.count(), 100);
        assert!((hist.mean() - 50.5).abs() < 0.1);
        assert!((hist.p50() - 50.0).abs() < 2.0);
        assert!((hist.p95() - 95.0).abs() < 2.0);
        assert!((hist.p99() - 99.0).abs() < 2.0);
    }

    #[test]
    fn test_metrics_snapshot() {
        let metrics = DistributedMetrics::new();

        metrics.all_reduce.completed.add(100);
        metrics.all_reduce.failed.add(5);
        metrics.network.bytes_sent.add(1000000);
        metrics.peer.connected_peers.set(4);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.all_reduce_completed, 100);
        assert_eq!(snapshot.all_reduce_failed, 5);
        assert_eq!(snapshot.bytes_sent, 1000000);
        assert_eq!(snapshot.connected_peers, 4);
    }

    #[test]
    fn test_compression_ratio() {
        let metrics = CompressionMetrics::default();

        metrics.bytes_before.add(1000);
        metrics.bytes_after.add(250);

        assert!((metrics.compression_ratio() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_timing_guard() {
        let hist = Histogram::new(100);

        {
            let _guard = TimingGuard::new_ms(&hist);
            std::thread::sleep(Duration::from_millis(10));
        }

        assert!(hist.mean() >= 10.0);
        assert!(hist.count() == 1);
    }
}
