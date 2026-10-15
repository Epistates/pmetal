use super::*;
use crate::compression::{CompressionStrategy, GradientCompressor, QuantizationType};
use crate::ring::RingBackend;
use crate::testing::{fresh_security, noise, ring_mean, tcp_ring};

const LEN: usize = 37;

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// What a worker's inner steps did to the parameters in round `round`.
fn locals(start: &[f32], rank: usize, round: u64) -> Vec<f32> {
    let drift = noise(1000 + round * 10 + rank as u64, LEN);
    start.iter().zip(&drift).map(|(s, d)| s + 0.1 * d).collect()
}

/// The outer optimizer applied to the mean pseudo-gradient, written out.
fn reference_round(
    config: &DiLoCoConfig,
    global: &mut [f32],
    velocity: &mut [f32],
    mean_delta: &[f32],
) {
    for ((g, v), &d) in global.iter_mut().zip(velocity).zip(mean_delta) {
        *v = config.outer_momentum * *v + d;
        let update = if config.nesterov {
            d + config.outer_momentum * *v
        } else {
            *v
        };
        *g -= config.outer_lr * update;
    }
}

async fn run_workers(
    n: usize,
    config: DiLoCoConfig,
    rounds: u64,
    compress: bool,
) -> Vec<(Vec<f32>, Vec<SyncStats>)> {
    let ring = tcp_ring(n, fresh_security()).await;
    let mut tasks = Vec::new();
    for (rank, collective) in ring.into_iter().enumerate() {
        tasks.push(tokio::spawn(async move {
            let backend = RingBackend::from_collective(collective);
            let mut diloco = DiLoCo::new(&noise(7, LEN), config).unwrap();
            let mut params = noise(7, LEN);
            let mut compressor = compress.then(|| {
                GradientCompressor::with_seed(
                    CompressionStrategy::Quantize(QuantizationType::FP16),
                    true,
                    rank as u64,
                )
            });
            let mut stats = Vec::new();
            for round in 0..rounds {
                let after_inner_steps = locals(&params, rank, round);
                params.copy_from_slice(&after_inner_steps);
                for _ in 0..config.sync_every - 1 {
                    assert!(!diloco.step_done());
                }
                assert!(diloco.step_done());
                stats.push(
                    diloco
                        .sync(&backend, &mut params, compressor.as_mut())
                        .await
                        .unwrap(),
                );
            }
            (params, stats)
        }));
    }
    let mut out = Vec::new();
    for task in tasks {
        out.push(task.await.unwrap());
    }
    out
}

/// With a unit learning rate and no momentum the outer step is the
/// plain average of the workers' parameters: the global model after a round is
/// the mean of what the workers trained.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn plain_averaging_is_the_special_case_of_unit_lr_and_no_momentum() {
    let config = DiLoCoConfig {
        sync_every: 3,
        outer_lr: 1.0,
        outer_momentum: 0.0,
        nesterov: false,
    };
    let n = 3;
    let mut global = noise(7, LEN);
    let mut velocity = vec![0.0; LEN];
    let mut want = Vec::new();
    for round in 0..3 {
        let deltas: Vec<Vec<f32>> = (0..n)
            .map(|r| {
                let local = locals(&global, r, round);
                global.iter().zip(&local).map(|(g, l)| g - l).collect()
            })
            .collect();
        reference_round(&config, &mut global, &mut velocity, &ring_mean(&deltas));
        want.push(global.clone());
    }
    for (rank, (params, stats)) in run_workers(n, config, 3, false)
        .await
        .into_iter()
        .enumerate()
    {
        assert!(same_bits(&params, &want[2]), "rank {rank} differs");
        assert_eq!(
            stats.iter().map(|s| s.round).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }
    // And it really is the mean of the workers' parameters, up to rounding.
    let start = noise(7, LEN);
    let mean_of_locals: Vec<f32> = (0..LEN)
        .map(|i| (0..n).map(|r| locals(&start, r, 0)[i]).sum::<f32>() / n as f32)
        .collect();
    let first_round = {
        let mut g = noise(7, LEN);
        let deltas: Vec<Vec<f32>> = (0..n)
            .map(|r| {
                let local = locals(&g, r, 0);
                g.iter().zip(&local).map(|(g, l)| g - l).collect()
            })
            .collect();
        reference_round(&config, &mut g, &mut [0.0; LEN], &ring_mean(&deltas));
        g
    };
    for (a, b) in first_round.iter().zip(&mean_of_locals) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn nesterov_momentum_matches_the_written_out_outer_optimizer() {
    let config = DiLoCoConfig {
        sync_every: 2,
        ..DiLoCoConfig::default()
    };
    for n in [2usize, 3] {
        let mut global = noise(7, LEN);
        let mut velocity = vec![0.0; LEN];
        for round in 0..4 {
            let deltas: Vec<Vec<f32>> = (0..n)
                .map(|r| {
                    let local = locals(&global, r, round);
                    global.iter().zip(&local).map(|(g, l)| g - l).collect()
                })
                .collect();
            reference_round(&config, &mut global, &mut velocity, &ring_mean(&deltas));
        }
        for (rank, (params, _)) in run_workers(n, config, 4, false)
            .await
            .into_iter()
            .enumerate()
        {
            assert!(
                same_bits(&params, &global),
                "{n} workers: rank {rank} differs"
            );
        }
    }
}

/// Compressed pseudo-gradients agree across workers bit for bit, and send
/// fewer bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compressed_rounds_leave_every_worker_identical() {
    let config = DiLoCoConfig {
        sync_every: 2,
        ..DiLoCoConfig::default()
    };
    let results = run_workers(3, config, 3, true).await;
    for (rank, (params, stats)) in results.iter().enumerate() {
        assert!(same_bits(params, &results[0].0), "rank {rank} diverged");
        assert!(stats.iter().all(|s| s.bytes_sent < LEN * 4 + 64));
        assert!(
            stats.iter().all(|s| s.bytes_sent < LEN * 4),
            "fp16 halves the payload"
        );
    }
    // And it stays close to the uncompressed result.
    let exact = run_workers(3, config, 3, false).await;
    for (a, b) in results[0].0.iter().zip(&exact[0].0) {
        assert!((a - b).abs() < 5e-3, "{a} vs {b}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_first_round_can_start_from_rank_zeros_parameters() {
    let ring = tcp_ring(3, fresh_security()).await;
    let mut tasks = Vec::new();
    for (rank, collective) in ring.into_iter().enumerate() {
        tasks.push(tokio::spawn(async move {
            let backend = RingBackend::from_collective(collective);
            // Every worker initialised differently.
            let mut params = noise(100 + rank as u64, LEN);
            let mut diloco = DiLoCo::new(&params, DiLoCoConfig::default()).unwrap();
            diloco
                .synchronize_initial(&backend, &mut params)
                .await
                .unwrap();
            (params, diloco.global_parameters().to_vec())
        }));
    }
    let want = noise(100, LEN);
    for task in tasks {
        let (params, global) = task.await.unwrap();
        assert!(same_bits(&params, &want));
        assert!(same_bits(&global, &want));
    }
}

#[test]
fn the_round_counter_and_bad_inputs() {
    let mut d = DiLoCo::new(
        &[0.0; 4],
        DiLoCoConfig {
            sync_every: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!d.step_done());
    assert!(!d.step_done());
    assert!(d.step_done());
    assert!(
        DiLoCo::new(
            &[0.0],
            DiLoCoConfig {
                sync_every: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        DiLoCo::new(
            &[0.0],
            DiLoCoConfig {
                outer_lr: f32::NAN,
                ..Default::default()
            }
        )
        .is_err()
    );
}
