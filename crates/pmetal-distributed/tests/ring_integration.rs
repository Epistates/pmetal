//! The ring backend through its public API only: two ranks over loopback.
//!
//! These tests say what a user of the crate can rely on. They live outside
//! the crate so they cannot reach into its internals.

use pmetal_distributed::{
    ClusterSecret, DistributedBackend, DistributedConfig, NetworkNamespace, ReduceOp, RingBackend,
};
use std::net::{SocketAddr, TcpListener as StdListener};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

fn free_port() -> u16 {
    StdListener::bind("127.0.0.1:0")
        .expect("bind")
        .local_addr()
        .expect("addr")
        .port()
}

fn addrs() -> Vec<SocketAddr> {
    vec![
        format!("127.0.0.1:{}", free_port()).parse().unwrap(),
        format!("127.0.0.1:{}", free_port()).parse().unwrap(),
    ]
}

fn config(nodes: &[SocketAddr], rank: usize, secret: &ClusterSecret) -> DistributedConfig {
    let mut config = DistributedConfig::new(nodes.to_vec(), rank).with_secret(secret.clone());
    config.connection_timeout_ms = 15_000;
    config.max_retries = 8;
    config
}

fn bytes_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The all-reduce an honest pair of ranks must compute.
async fn honest_pair(nodes: &[SocketAddr], secret: &ClusterSecret) -> (Vec<f32>, Vec<f32>) {
    let mut tasks = Vec::new();
    for rank in 0..2 {
        let config = config(nodes, rank, secret);
        tasks.push(tokio::spawn(async move {
            let ring = RingBackend::new(config).await.expect("ring");
            let mut data: Vec<f32> = if rank == 0 {
                vec![1.0, 2.0, 3.0, 4.0]
            } else {
                vec![10.0, 20.0, 30.0, 40.0]
            };
            ring.all_reduce_f32(&mut data, ReduceOp::Mean)
                .await
                .expect("all_reduce");
            data
        }));
    }
    (
        tasks.remove(0).await.unwrap(),
        tasks.remove(0).await.unwrap(),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_ranks_average_their_gradients() {
    let secret = ClusterSecret::generate();
    let (a, b) = honest_pair(&addrs(), &secret).await;
    assert_eq!(a, vec![5.5, 11.0, 16.5, 22.0]);
    assert_eq!(a, b);
}

/// Anyone on the LAN can open a connection to a rank. One who knows only
/// what is public (the protocol, the version string, so the default
/// namespace key) must not be admitted as the previous rank of the ring,
/// where it could feed its own numbers into every gradient average.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lan_peer_knowing_only_the_public_namespace_cannot_join_the_ring() {
    let nodes = addrs();
    let secret = ClusterSecret::generate();

    // Rank 1 comes up first and waits for its predecessor.
    let victim = tokio::spawn({
        let config = config(&nodes, 1, &secret);
        async move { RingBackend::new(config).await }
    });

    // The attacker dials rank 1 and presents the namespace key that the
    // protocol used to accept as proof of membership: magic plus
    // SHA3-256("pmetal/<version>"), which anyone can compute.
    let mut stream = loop {
        match TcpStream::connect(nodes[1]).await {
            Ok(stream) => break stream,
            Err(_) => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    };
    let mut hello = b"PMETALD1".to_vec();
    hello.extend_from_slice(NetworkNamespace::default().psk());
    stream.write_all(&hello).await.unwrap();
    let mut verdict = [0u8; 1];
    let admitted = matches!(
        tokio::time::timeout(Duration::from_secs(10), stream.read_exact(&mut verdict)).await,
        Ok(Ok(_)) if verdict[0] == 1
    );
    assert!(
        !admitted,
        "a peer that only knows the public namespace was admitted to the ring"
    );
    drop(stream);

    // The honest predecessor still gets in, and the average is untouched.
    let rank0 = tokio::spawn({
        let config = config(&nodes, 0, &secret);
        async move {
            let ring = RingBackend::new(config).await.expect("rank 0");
            let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
            ring.all_reduce_f32(&mut data, ReduceOp::Mean)
                .await
                .unwrap();
            data
        }
    });
    let ring1 = victim.await.unwrap().expect("rank 1 forms its ring");
    let mut data = vec![10.0f32, 20.0, 30.0, 40.0];
    ring1
        .all_reduce_f32(&mut data, ReduceOp::Mean)
        .await
        .unwrap();
    assert_eq!(data, vec![5.5, 11.0, 16.5, 22.0]);
    assert_eq!(rank0.await.unwrap(), data);
}

/// A byte buffer promises nothing about alignment. The all-reduce must give
/// the right answer for any placement of the bytes, not refuse the ones that
/// happen to be misaligned.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_misaligned_byte_buffer_is_reduced_correctly() {
    let nodes = addrs();
    let secret = ClusterSecret::generate();
    let mut tasks = Vec::new();
    for rank in 0..2 {
        let config = config(&nodes, rank, &secret);
        tasks.push(tokio::spawn(async move {
            let ring = RingBackend::new(config).await.expect("ring");
            let values: Vec<f32> = (0..9).map(|i| (i * (rank + 1)) as f32).collect();
            // Put the payload at every offset in turn, so some are surely
            // not 4-byte aligned.
            let mut out = Vec::new();
            for offset in 0..4 {
                let mut backing = vec![0u8; offset + 36];
                backing[offset..].copy_from_slice(&bytes_of(&values));
                ring.all_reduce(&mut backing[offset..], ReduceOp::Sum)
                    .await
                    .unwrap_or_else(|e| panic!("offset {offset}: {e}"));
                out.push(backing[offset..].to_vec());
            }
            out
        }));
    }
    let want = bytes_of(&(0..9).map(|i| (i * 3) as f32).collect::<Vec<_>>());
    for task in tasks {
        for (offset, got) in task.await.unwrap().into_iter().enumerate() {
            assert_eq!(got, want, "offset {offset}");
        }
    }
}

/// Peers whose node lists differ must be told so, not left to exchange
/// gradients between what are really two different rings.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ranks_with_different_node_lists_refuse_to_form_a_ring() {
    let nodes = addrs();
    // Rank 0 believes there is a third rank; rank 1 believes there are two.
    let mut three = nodes.clone();
    three.push("127.0.0.1:9".parse().unwrap());
    let secret = ClusterSecret::generate();

    let a = tokio::spawn({
        let config = config(&three, 0, &secret);
        async move { RingBackend::new(config).await }
    });
    let b = tokio::spawn({
        let config = config(&nodes, 1, &secret);
        async move { RingBackend::new(config).await }
    });
    let (a, b) = (a.await.unwrap(), b.await.unwrap());
    let (a, b) = (
        a.err().expect("rank 0 formed a ring"),
        b.err().expect("rank 1 formed a ring"),
    );
    assert!(a.to_string().contains("mismatch"), "{a}");
    assert!(b.to_string().contains("mismatch"), "{b}");
}

/// One all-reduce chunk bigger than the old 512 MiB framing cap. Each rank
/// holds 1 GiB and the test allocates several, so it is opt-in.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "allocates several GiB"]
async fn a_buffer_whose_chunks_exceed_half_a_gibibyte() {
    let nodes = addrs();
    let secret = ClusterSecret::generate();
    let len = 2 * (512 * 1024 * 1024 / 4) + 8; // two chunks just over 512 MiB
    let mut tasks = Vec::new();
    for rank in 0..2 {
        let config = config(&nodes, rank, &secret);
        tasks.push(tokio::spawn(async move {
            let ring = RingBackend::new(config).await.expect("ring");
            let mut data = vec![(rank + 1) as f32; len];
            ring.all_reduce_f32(&mut data, ReduceOp::Sum)
                .await
                .expect("all_reduce");
            data.iter().all(|&x| x == 3.0)
        }));
    }
    for task in tasks {
        assert!(task.await.unwrap());
    }
}
