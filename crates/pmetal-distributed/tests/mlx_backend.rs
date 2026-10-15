//! `MlxDistributedBackend` through the `DistributedBackend` trait, by real
//! ranks over MLX's ring on loopback: every collective the TCP ring offers
//! must also work here, with the same answers.

#![cfg(feature = "mlx-collectives")]

mod common;

use pmetal_distributed::collective::chunk_range;
use pmetal_distributed::compression::{CompressionStrategy, GradientCompressor};
use pmetal_distributed::mlx_dist::{DistributedGroup, MlxDistributedBackend};
use pmetal_distributed::{DistributedBackend, ReduceOp};
use std::time::Duration;

fn input(rank: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((rank + 1) as f32) * 0.25 + i as f32 * 0.5)
        .collect()
}

#[test]
fn the_mlx_backend_offers_every_collective() {
    if !common::is_rank() {
        return common::launch(
            3,
            "the_mlx_backend_offers_every_collective",
            Duration::from_secs(180),
        );
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let backend = MlxDistributedBackend::new(DistributedGroup::init(true).expect("group"));
        let (rank, n) = (backend.rank(), backend.world_size());
        assert_eq!(n, 3);

        for len in [0usize, 1, 2, 7, 100] {
            let all: Vec<Vec<f32>> = (0..n).map(|r| input(r, len)).collect();
            let sum: Vec<f32> = (0..len).map(|i| all.iter().map(|v| v[i]).sum()).collect();

            let mut data = all[rank].clone();
            backend
                .all_reduce_f32(&mut data, ReduceOp::Sum)
                .await
                .unwrap();
            assert_eq!(data, sum, "all_reduce len {len}");

            // Reduce-scatter leaves this rank's chunk reduced; all-gather
            // completes the rest from every rank's chunk.
            let mut data = all[rank].clone();
            let owned = backend
                .reduce_scatter_f32(&mut data, ReduceOp::Sum)
                .await
                .unwrap();
            assert_eq!(owned, chunk_range(len, n, rank));
            assert_eq!(data[owned.clone()], sum[owned], "reduce_scatter len {len}");
            backend.all_gather_f32_in_place(&mut data).await.unwrap();
            assert_eq!(data, sum, "all_gather len {len}");
        }

        // Variable-length byte gather, including an empty contribution.
        let payloads: Vec<Vec<u8>> = (0..n)
            .map(|r| (0..r * 1001).map(|i| (i * 7 + r) as u8).collect())
            .collect();
        let got = backend.all_gather_bytes(&payloads[rank]).await.unwrap();
        assert_eq!(got, payloads);

        // Broadcast from a rank other than 0, with a length that is not a
        // whole number of words.
        for root in 0..n {
            let message: Vec<u8> = (0..4099).map(|i| (i * 13 + root) as u8).collect();
            let mut buf = if rank == root {
                message.clone()
            } else {
                vec![0u8; message.len()]
            };
            backend.broadcast(&mut buf, root).await.unwrap();
            assert_eq!(buf, message, "broadcast from {root}");
        }

        backend.barrier().await.unwrap();

        // Compressed all-reduce through this backend: every rank decodes the
        // same packets in rank order.
        let len = 50;
        let strategy = CompressionStrategy::TopK { ratio: 0.2 };
        let mut want = vec![0f32; len];
        for r in 0..n {
            GradientCompressor::with_seed(strategy.clone(), true, r as u64)
                .compress(&input(r, len))
                .add_to(&mut want);
        }
        want.iter_mut().for_each(|x| *x /= n as f32);
        let mut data = input(rank, len);
        let mut compressor = GradientCompressor::with_seed(strategy, true, rank as u64);
        pmetal_distributed::compression::all_reduce_compressed(
            &backend,
            &mut compressor,
            &mut data,
            ReduceOp::Mean,
        )
        .await
        .unwrap();
        assert_eq!(data, want);
    });
}
