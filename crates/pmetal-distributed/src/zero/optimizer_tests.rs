use super::*;
use crate::collective::chunk_range;
use crate::ring::RingBackend;
use crate::testing::{fresh_security, noise, ring_mean, tcp_ring};
use crate::zero::ZeroLayout;

fn named_params() -> Vec<(String, usize)> {
    vec![
        ("layer.0.weight".into(), 61),
        ("embed".into(), 40),
        ("norm.bias".into(), 2),
    ]
}

const TOTAL: usize = 103;
const STEPS: usize = 5;

fn grads_for(step: usize, rank: usize) -> Vec<f32> {
    noise((step * 100 + rank) as u64 + 1, TOTAL)
}

/// One device, the whole vector, the mean gradient of every step.
fn single_device(n: usize, max_norm: Option<f32>) -> Vec<f32> {
    let mut params = noise(999, TOTAL);
    let mut opt = ShardedAdamW::new(TOTAL, 0.05);
    for step in 0..STEPS {
        let inputs: Vec<Vec<f32>> = (0..n).map(|r| grads_for(step, r)).collect();
        let mut g = ring_mean(&inputs);
        if let Some(max) = max_norm {
            // Per-shard sums of squares added in rank order, as the ranks do.
            let norm = (0..n)
                .map(|r| {
                    g[chunk_range(TOTAL, n, r)]
                        .iter()
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum::<f64>()
                })
                .sum::<f64>()
                .sqrt();
            if norm > f64::from(max) {
                let scale = (f64::from(max) / (norm + 1e-6)) as f32;
                g.iter_mut().for_each(|x| *x *= scale);
            }
        }
        opt.step(&mut params, &g);
    }
    params
}

async fn sharded(
    n: usize,
    stage: ZeROStage,
    max_norm: Option<f32>,
) -> Vec<(Vec<f32>, usize, usize)> {
    let ring = tcp_ring(n, fresh_security()).await;
    let mut tasks = Vec::new();
    for (rank, collective) in ring.into_iter().enumerate() {
        tasks.push(tokio::spawn(async move {
            let backend = RingBackend::from_collective(collective);
            let layout = ZeroLayout::new(&named_params(), rank, n).unwrap();
            let shard_len = layout.shard_range().len();
            let mut zero = ZeroOptimizer::new(layout, ShardedAdamW::new(shard_len, 0.05), stage);
            if let Some(max) = max_norm {
                zero = zero.with_max_grad_norm(max);
            }
            let mut params = noise(999, TOTAL);
            let mut released = 0;
            for step in 0..STEPS {
                let mut grads = grads_for(step, rank);
                zero.step(&backend, &mut params, &mut grads).await.unwrap();
                released = grads.len();
            }
            (params, zero.state_bytes(), released)
        }));
    }
    let mut out = Vec::new();
    for task in tasks {
        out.push(task.await.unwrap());
    }
    out
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sharded_step_is_bit_identical_to_one_device() {
    for n in 1..=4 {
        for stage in [ZeROStage::Stage1, ZeROStage::Stage2] {
            let want = single_device(n, None);
            for (rank, (params, state_bytes, released)) in
                sharded(n, stage, None).await.into_iter().enumerate()
            {
                assert!(
                    same_bits(&params, &want),
                    "{n} ranks, {stage:?}: rank {rank} differs from one device"
                );
                // Adam keeps two f32 per element, for this rank's shard only.
                let shard = chunk_range(TOTAL, n, rank).len();
                assert_eq!(state_bytes, shard * 8);
                assert_eq!(released == 0, stage == ZeROStage::Stage2);
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn global_norm_clipping_is_the_same_on_every_rank_and_as_one_device() {
    for n in [2, 3] {
        let want = single_device(n, Some(0.5));
        for (rank, (params, ..)) in sharded(n, ZeROStage::Stage2, Some(0.5))
            .await
            .into_iter()
            .enumerate()
        {
            assert!(same_bits(&params, &want), "{n} ranks: rank {rank} differs");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn optimizer_state_per_rank_shrinks_with_the_world() {
    let sizes: Vec<usize> = (1..=4)
        .map(|n| {
            let layout = ZeroLayout::new(&named_params(), 0, n).unwrap();
            ShardedAdamW::new(layout.shard_range().len(), 0.1).state_bytes()
        })
        .collect();
    assert_eq!(sizes[0], TOTAL * 8);
    assert!(sizes[3] * 4 <= sizes[0] + 32);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mismatched_lengths_are_rejected() {
    let ring = tcp_ring(1, fresh_security()).await;
    let backend = RingBackend::from_collective(ring.into_iter().next().unwrap());
    let layout = ZeroLayout::new(&named_params(), 0, 1).unwrap();
    let mut zero = ZeroOptimizer::new(layout, ShardedAdamW::new(TOTAL, 0.1), ZeROStage::Stage1);
    let err = zero
        .step(&backend, &mut vec![0.0; TOTAL], &mut vec![0.0; TOTAL - 1])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("layout has"), "{err}");
}
