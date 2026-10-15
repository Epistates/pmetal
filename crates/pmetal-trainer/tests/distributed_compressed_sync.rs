//! Gradient synchronization with compression, across two ranks in one
//! process over loopback.
//!
//! Compressed gradients cannot be added on the wire, so the ranks must
//! exchange them and each sum what they decode to. The mean every rank ends
//! up with must be the mean of the decoded gradients, not of their bytes.

#![cfg(feature = "distributed")]

use pmetal_core::{DistributedCompression, DistributedTrainingConfig};
use pmetal_distributed::compression::GradientCompressor;
use pmetal_distributed::{
    ClusterSecret, CompressionStrategy, DistributedConfig, DistributedContext, RingBackend,
};
use pmetal_runtime::compat::{Array, module::FlattenedModuleParam};
use pmetal_trainer::DistributedGradientSync;
use std::net::{SocketAddr, TcpListener};
use std::rc::Rc;
use std::sync::Arc;

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind")
        .local_addr()
        .expect("addr")
        .port()
}

fn gradients(values: &[f32]) -> FlattenedModuleParam {
    let mut grads = FlattenedModuleParam::new();
    grads.insert(
        Rc::from("layer.weight"),
        Array::from_f32_slice(values, &[values.len() as i32]),
    );
    grads
}

fn read(grads: &FlattenedModuleParam) -> Vec<f32> {
    let mut array = grads.get("layer.weight").expect("gradient").clone();
    let n = array.size();
    array.to_f32_vec(n).expect("readback")
}

/// One rank: join the ring, sync `values` through the bridge, return what
/// the bridge left in the gradient.
fn run_rank(
    nodes: Vec<SocketAddr>,
    rank: usize,
    secret: ClusterSecret,
    compression: DistributedCompression,
    values: Vec<f32>,
) -> Vec<f32> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async move {
        let backend = RingBackend::new(DistributedConfig::new(nodes, rank).with_secret(secret))
            .await
            .expect("ring");
        let ctx = Arc::new(DistributedContext::new(Box::new(backend)));
        let config = DistributedTrainingConfig {
            compression,
            topk_ratio: 0.5,
            error_feedback: true,
            ..Default::default()
        };
        let mut sync = DistributedGradientSync::new(ctx, &config);
        let mut grads = gradients(&values);
        sync.sync_gradients(&mut grads).await.expect("sync");
        read(&grads)
    })
}

fn two_ranks(compression: DistributedCompression, a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let nodes: Vec<SocketAddr> = (0..2)
        .map(|_| format!("127.0.0.1:{}", free_port()).parse().unwrap())
        .collect();
    let secret = ClusterSecret::generate();
    let spawn = |rank: usize, values: Vec<f32>| {
        let (nodes, secret) = (nodes.clone(), secret.clone());
        std::thread::spawn(move || run_rank(nodes, rank, secret, compression, values))
    };
    let (t0, t1) = (spawn(0, a.to_vec()), spawn(1, b.to_vec()));
    (t0.join().expect("rank 0"), t1.join().expect("rank 1"))
}

#[test]
fn uncompressed_sync_averages_gradients() {
    let (a, b) = two_ranks(
        DistributedCompression::None,
        &[1.0, 2.0, 3.0, 4.0],
        &[3.0, 2.0, 1.0, 0.0],
    );
    assert_eq!(a, vec![2.0, 2.0, 2.0, 2.0]);
    assert_eq!(a, b);
}

#[test]
fn topk_sync_averages_what_the_ranks_actually_sent() {
    let g0 = [4.0f32, -1.0, 0.5, 8.0, 0.1, -6.0, 0.2, 0.3];
    let g1 = [0.1f32, 5.0, -7.0, 0.2, 9.0, 0.3, 0.4, -2.0];

    // Ratio 0.5: each rank sends its four largest magnitudes.
    let mut want = vec![0f32; g0.len()];
    for (rank, g) in [g0, g1].iter().enumerate() {
        GradientCompressor::with_seed(CompressionStrategy::TopK { ratio: 0.5 }, true, rank as u64)
            .compress(g)
            .add_to(&mut want);
    }
    want.iter_mut().for_each(|x| *x /= 2.0);

    let (a, b) = two_ranks(DistributedCompression::TopK, &g0, &g1);
    assert_eq!(a, want, "rank 0 got a different mean than the ranks sent");
    assert_eq!(a, b, "ranks disagree");
}
