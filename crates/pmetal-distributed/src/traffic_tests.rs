//! What the ring reports having moved, and how the context turns it into
//! metrics.

use super::*;
use crate::testing::{fresh_security, tcp_ring};

/// A ring all-reduce of `len` floats on `n` ranks moves, per rank, a header
/// and then every chunk but one in each of its two phases; the counters
/// report exactly that.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn traffic_counters_report_exactly_what_the_ring_moved() {
    let (n, len) = (4usize, 103usize);
    let ring = tcp_ring(n, fresh_security()).await;
    let mut tasks = Vec::new();
    for (rank, ring) in ring.into_iter().enumerate() {
        tasks.push(tokio::spawn(async move {
            assert_eq!(ring.traffic(), (0, 0));
            ring.all_reduce(&mut vec![1.0f32; len], ReduceOp::Sum)
                .await
                .expect("all_reduce");
            (rank, ring.traffic())
        }));
    }
    let chunk = |i: usize| chunk_range(len, n, i % n).len() as u64 * 4;
    let header = OpHeader::LEN as u64;
    // Rank r sends every chunk but r in the reduce-scatter and every chunk but
    // r + 1 in the all-gather.
    let sends = |r: usize| header + (len as u64 * 4 - chunk(r)) + (len as u64 * 4 - chunk(r + 1));
    for task in tasks {
        let (rank, (sent, received)) = task.await.expect("task");
        assert_eq!(sent, sends(rank), "rank {rank} sent");
        assert_eq!(received, sends(rank + n - 1), "rank {rank} received");
    }
}

/// The context folds the backend's traffic into the network metrics.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_context_reports_network_bytes_in_its_metrics() {
    use crate::DistributedContext;
    use crate::metrics::new_shared_metrics;
    use crate::ring::RingBackend;

    let ring = tcp_ring(2, fresh_security()).await;
    let mut tasks = Vec::new();
    for ring in ring {
        tasks.push(tokio::spawn(async move {
            let metrics = new_shared_metrics();
            let ctx = DistributedContext::with_metrics(
                Box::new(RingBackend::from_collective(ring)),
                metrics.clone(),
            );
            ctx.all_reduce_f32(&mut vec![1.0f32; 1000], ReduceOp::Sum)
                .await
                .expect("all_reduce");
            ctx.barrier().await.expect("barrier");
            (
                metrics.network.bytes_sent.get(),
                metrics.network.bytes_received.get(),
                ctx.traffic().expect("a ring reports its traffic"),
            )
        }));
    }
    for task in tasks {
        let (sent, received, backend) = task.await.expect("task");
        // Two ranks: each sends half the buffer in each phase.
        assert!(sent >= 4000 && received >= 4000, "{sent} / {received}");
        assert_eq!((sent, received), backend);
    }
}
