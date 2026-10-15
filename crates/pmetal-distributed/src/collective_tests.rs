//! Ring collectives against sequential references, over real loopback TCP.

use super::*;
use crate::testing::{fresh_security, tcp_ring};
use std::sync::Arc;

/// Deterministic noise with a wide dynamic range, so that the order in
/// which values are added changes the low bits of a float sum.
fn noise(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mantissa = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let exponent = ((state >> 8) % 12) as i32 - 6;
            mantissa * 2f32.powi(exponent)
        })
        .collect()
}

/// The sum the ring computes for chunk `c`: contributions are added in
/// the order the partial sum travels, starting at rank `c + 1`.
fn reference(inputs: &[Vec<f32>], op: ReduceOp) -> Vec<f32> {
    let n = inputs.len();
    let len = inputs[0].len();
    let mut out = vec![0f32; len];
    for c in 0..n {
        for i in chunk_range(len, n, c) {
            let mut acc = inputs[(c + 1) % n][i];
            for j in 2..=n {
                acc += inputs[(c + j) % n][i];
            }
            out[i] = if op == ReduceOp::Mean {
                acc / n as f32
            } else {
                acc
            };
        }
    }
    out
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

async fn run_all_reduce(n: usize, len: usize, op: ReduceOp) {
    let ring = tcp_ring(n, fresh_security()).await;
    let inputs: Vec<Vec<f32>> = (0..n).map(|r| noise(r as u64 + 1, len)).collect();
    let want = reference(&inputs, op);
    let mut tasks = Vec::new();
    for (rank, ring) in ring.into_iter().enumerate() {
        let mut data = inputs[rank].clone();
        tasks.push(tokio::spawn(async move {
            ring.all_reduce(&mut data, op).await.expect("all_reduce");
            data
        }));
    }
    for (rank, task) in tasks.into_iter().enumerate() {
        let got = task.await.expect("task");
        assert!(
            same_bits(&got, &want),
            "rank {rank} of {n}, len {len}, {op:?}: result differs from the reference"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn all_reduce_equals_the_sequential_reference_bit_for_bit() {
    for n in 1..=5 {
        for len in [0, 1, 2, 3, 4, 5, 7, 64, 1001] {
            for op in [ReduceOp::Sum, ReduceOp::Mean] {
                run_all_reduce(n, len, op).await;
            }
        }
    }
}

/// A chunk larger than one wire segment is sent as several messages and
/// reduced segment by segment; the result must not depend on that.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn all_reduce_across_several_segments() {
    run_all_reduce(3, SEGMENT_ELEMS * 3 + 17, ReduceOp::Mean).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reduce_scatter_then_all_gather_is_all_reduce() {
    let n = 4;
    let len = 103;
    let ring = tcp_ring(n, fresh_security()).await;
    let inputs: Vec<Vec<f32>> = (0..n).map(|r| noise(r as u64 + 9, len)).collect();
    let want = reference(&inputs, ReduceOp::Sum);
    let mut tasks = Vec::new();
    for (rank, ring) in ring.into_iter().enumerate() {
        let mut data = inputs[rank].clone();
        let want = want.clone();
        tasks.push(tokio::spawn(async move {
            let owned = ring
                .reduce_scatter(&mut data, ReduceOp::Sum)
                .await
                .expect("reduce_scatter");
            assert_eq!(owned, chunk_range(len, n, rank));
            assert!(same_bits(&data[owned.clone()], &want[owned]));
            ring.all_gather_in_place(&mut data)
                .await
                .expect("all_gather");
            data
        }));
    }
    for task in tasks {
        assert!(same_bits(&task.await.expect("task"), &want));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn all_gather_bytes_carries_every_ranks_own_length() {
    for n in 1..=4 {
        let ring = tcp_ring(n, fresh_security()).await;
        let payloads: Vec<Vec<u8>> = (0..n)
            .map(|r| (0..r * 37).map(|i| (i + r) as u8).collect())
            .collect();
        let mut tasks = Vec::new();
        for (rank, ring) in ring.into_iter().enumerate() {
            let local = payloads[rank].clone();
            tasks.push(tokio::spawn(async move {
                ring.all_gather_bytes(&local).await.expect("gather")
            }));
        }
        for task in tasks {
            assert_eq!(task.await.expect("task"), payloads);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn broadcast_from_every_root() {
    for n in 1..=4 {
        for root in 0..n {
            let ring = tcp_ring(n, fresh_security()).await;
            let message: Vec<u8> = (0..100_003).map(|i| (i * 7 + root) as u8).collect();
            let mut tasks = Vec::new();
            for (rank, ring) in ring.into_iter().enumerate() {
                let mut buf = if rank == root {
                    message.clone()
                } else {
                    vec![0u8; message.len()]
                };
                tasks.push(tokio::spawn(async move {
                    ring.broadcast(&mut buf, root).await.expect("broadcast");
                    buf
                }));
            }
            for task in tasks {
                assert_eq!(task.await.expect("task"), message);
            }
        }
    }
}

/// No rank leaves the barrier before every rank has entered it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn barrier_waits_for_the_slowest_rank() {
    for n in 2..=5 {
        let ring = tcp_ring(n, fresh_security()).await;
        let entered = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for (rank, ring) in ring.into_iter().enumerate() {
            let entered = Arc::clone(&entered);
            tasks.push(tokio::spawn(async move {
                // The last rank arrives late.
                if rank == n - 1 {
                    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                }
                entered.fetch_add(1, Ordering::SeqCst);
                ring.barrier().await.expect("barrier");
                entered.load(Ordering::SeqCst)
            }));
        }
        for task in tasks {
            assert_eq!(
                task.await.expect("task"),
                n,
                "a rank left the barrier early"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ranks_that_disagree_on_the_length_fail_loudly() {
    let ring = tcp_ring(2, fresh_security()).await;
    let mut tasks = Vec::new();
    for (rank, ring) in ring.into_iter().enumerate() {
        tasks.push(tokio::spawn(async move {
            // Lengths 10 and 11 cut into chunks of equal first-step size,
            // which is how an unchecked mismatch would slip through.
            let mut data = vec![1.0f32; 10 + rank];
            let first = ring.all_reduce(&mut data, ReduceOp::Sum).await;
            let second = ring.barrier().await;
            (first, second)
        }));
    }
    for task in tasks {
        let (first, second) = task.await.expect("task");
        let err = first.expect_err("mismatched lengths must be an error");
        assert!(err.to_string().contains("disagree"), "{err}");
        assert!(
            second
                .expect_err("ring is unusable")
                .to_string()
                .contains("poisoned"),
            "the ring must be poisoned after a failure"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ranks_in_different_collectives_fail_loudly() {
    let ring = tcp_ring(2, fresh_security()).await;
    let mut tasks = Vec::new();
    for (rank, ring) in ring.into_iter().enumerate() {
        tasks.push(tokio::spawn(async move {
            if rank == 0 {
                ring.barrier().await
            } else {
                ring.all_reduce(&mut [1.0f32; 4], ReduceOp::Sum).await
            }
        }));
    }
    for task in tasks {
        let err = task.await.expect("task").expect_err("must fail");
        assert!(err.to_string().contains("disagree"), "{err}");
    }
}

/// A collective whose future is dropped half way leaves the stream
/// mid-message; the ring must refuse further use rather than misread it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cancelled_collective_poisons_the_ring() {
    let mut ring = tcp_ring(2, fresh_security()).await;
    let lone = ring.remove(0);
    // Rank 1 never participates, so rank 0 would wait forever.
    let outcome = tokio::time::timeout(
        std::time::Duration::from_millis(200),
        lone.all_reduce(&mut [1.0f32; 8], ReduceOp::Sum),
    )
    .await;
    assert!(outcome.is_err(), "the collective should still be waiting");
    assert!(lone.is_poisoned());
    let err = lone.barrier().await.expect_err("poisoned");
    assert!(err.to_string().contains("poisoned"), "{err}");
}

/// When a rank dies the others get an error, not an eternal wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dead_peer_is_an_error_not_a_hang() {
    let mut ring = tcp_ring(3, fresh_security()).await;
    drop(ring.remove(1));
    let mut tasks = Vec::new();
    for ring in ring {
        tasks.push(tokio::spawn(async move {
            ring.all_reduce(&mut [1.0f32; 64], ReduceOp::Sum).await
        }));
    }
    for task in tasks {
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(30), task)
            .await
            .expect("must not hang")
            .expect("task");
        assert!(outcome.is_err());
    }
}
