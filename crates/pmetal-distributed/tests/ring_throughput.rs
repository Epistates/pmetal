//! Ring all-reduce throughput over loopback, secure and plaintext.
//!
//! Not a correctness test: run it by hand to see what the ring and its
//! encryption cost on this machine.
//!
//! ```text
//! cargo test -p pmetal-distributed --release --test ring_throughput -- --ignored --nocapture
//! ```
//!
//! All ranks share this machine's cores and loopback, so the figures bound
//! what a real network could carry from above only when the cluster's links are
//! faster than the machine's memory copies. Read them as relative costs.

use pmetal_distributed::{
    ClusterSecret, DistributedBackend, DistributedConfig, ReduceOp, RingBackend,
};
use std::net::{SocketAddr, TcpListener};
use std::time::Instant;

async fn measure(ranks: usize, megabytes: usize, secure: bool) -> f64 {
    // The probes stay open until every port is chosen, so the ports differ.
    let probes: Vec<TcpListener> = (0..ranks)
        .map(|_| TcpListener::bind("127.0.0.1:0").expect("bind"))
        .collect();
    let nodes: Vec<SocketAddr> = probes
        .iter()
        .map(|probe| probe.local_addr().expect("addr"))
        .collect();
    drop(probes);
    let secret = ClusterSecret::generate();
    let len = megabytes * 1024 * 1024 / 4;
    let iterations = 4;
    let mut tasks = Vec::new();
    for rank in 0..ranks {
        let mut config = DistributedConfig::new(nodes.clone(), rank);
        config = if secure {
            config.with_secret(secret.clone())
        } else {
            config.with_plaintext_allowed()
        };
        tasks.push(tokio::spawn(async move {
            let ring = RingBackend::new(config).await.expect("ring");
            let mut data = vec![1.0f32; len];
            // One warm-up pass, then time the rest.
            ring.all_reduce_f32(&mut data, ReduceOp::Sum).await.unwrap();
            ring.barrier().await.unwrap();
            let start = Instant::now();
            for _ in 0..iterations {
                ring.all_reduce_f32(&mut data, ReduceOp::Sum).await.unwrap();
            }
            ring.barrier().await.unwrap();
            start.elapsed().as_secs_f64() / f64::from(iterations)
        }));
    }
    let mut slowest = 0.0f64;
    for task in tasks {
        slowest = slowest.max(task.await.unwrap());
    }
    // Bytes each rank sends in a ring all-reduce: 2 (N - 1) / N of the buffer.
    let wire_bytes = (megabytes * 1024 * 1024) as f64 * 2.0 * (ranks as f64 - 1.0) / ranks as f64;
    wire_bytes / slowest / 1e9
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "a measurement, not a test"]
async fn ring_all_reduce_throughput() {
    for ranks in [2usize, 3, 4] {
        let plain = measure(ranks, 128, false).await;
        let secure = measure(ranks, 128, true).await;
        println!(
            "{ranks} ranks, 128 MiB: plaintext {plain:.2} GB/s per rank, \
             encrypted {secure:.2} GB/s per rank ({:.0}% of plaintext)",
            100.0 * secure / plain
        );
    }
}
