//! Data-parallel training through the real loop: two ranks in one process
//! over a loopback ring, each on its own half of the data, step for step.
//!
//! The ranks see different samples, so their gradients differ; they end on the
//! same weights only if every step averaged them.

#![cfg(feature = "distributed")]

use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};

use pmetal_core::{DistributedTrainingConfig, LoraConfig, TrainingConfig};
use pmetal_data::{DataLoaderConfig, Sample, TrainingDataset};
use pmetal_distributed::{ClusterSecret, DistributedConfig, DistributedContext, RingBackend};
use pmetal_lora::AdaptedModel;
use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::random;
use pmetal_trainer::{DistributedGradientSync, TrainingLoop, TrainingLoopConfig};

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind")
        .local_addr()
        .expect("addr")
        .port()
}

/// MLX's random key is process-wide: one rank builds its model at a time, from
/// the same seed, so both start from the same weights.
static BUILD: Mutex<()> = Mutex::new(());

fn model() -> AdaptedModel {
    let _building = BUILD.lock().unwrap();
    random::seed(21);
    let config = LlamaConfig {
        vocab_size: 128,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        max_position_embeddings: 64,
        ..Default::default()
    };
    let base = DynamicModel::from_config(&serde_json::to_string(&config).unwrap()).unwrap();
    let lora = LoraConfig {
        r: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        ..Default::default()
    };
    AdaptedModel::attach(base, lora).unwrap()
}

/// One rank: join the ring, train on this rank's share, return the weights.
fn run_rank(
    nodes: Vec<SocketAddr>,
    rank: usize,
    secret: ClusterSecret,
    packed: bool,
) -> RankOutcome {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    let ctx = runtime.block_on(async {
        let backend = RingBackend::new(DistributedConfig::new(nodes, rank).with_secret(secret))
            .await
            .expect("ring");
        Arc::new(DistributedContext::new(Box::new(backend)))
    });
    let sync = DistributedGradientSync::new(ctx, &DistributedTrainingConfig::default());
    // The loop blocks on the ring's futures from this plain thread.
    let _inside = runtime.enter();

    let mut config = TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-2,
            batch_size: 1,
            num_epochs: 1,
            max_grad_norm: 1.0,
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size: 1,
            max_seq_len: 16,
            shuffle: true,
            seed: 11,
            ..Default::default()
        },
        use_sequence_packing: packed,
        pack_max_seq_len: Some(16),
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        ..Default::default()
    };
    config.training.warmup_steps = 0;
    let mut training_loop = TrainingLoop::new(config).expect("a valid configuration");
    training_loop.set_distributed(sync);

    // Sample i is its own token repeated, so the ranks' gradients differ.
    let dataset = TrainingDataset::from_samples(
        (0..8)
            .map(|i| Sample::new((0..8).map(|j| ((i * 5 + j) % 100 + 1) as u32).collect()))
            .collect(),
    );
    let mut model = model();
    training_loop
        .run(&mut model, dataset, None, None)
        .expect("training");
    pmetal_runtime::check_unobserved_error().expect("no runtime op threw");

    let mut weights: Vec<(String, Vec<f32>)> = model
        .lora_parameters()
        .iter()
        .map(|(k, v)| {
            v.eval();
            (k.to_string(), v.as_slice::<f32>().to_vec())
        })
        .collect();
    weights.sort_by(|a, b| a.0.cmp(&b.0));
    (weights, training_loop.current_step())
}

/// A rank's adapter weights by name, and the steps it took.
type RankOutcome = (Vec<(String, Vec<f32>)>, usize);

fn two_ranks(packed: bool) -> [RankOutcome; 2] {
    let nodes: Vec<SocketAddr> = (0..2)
        .map(|_| format!("127.0.0.1:{}", free_port()).parse().unwrap())
        .collect();
    let secret = ClusterSecret::generate();
    let spawn = |rank: usize| {
        let (nodes, secret) = (nodes.clone(), secret.clone());
        std::thread::spawn(move || run_rank(nodes, rank, secret, packed))
    };
    let (t0, t1) = (spawn(0), spawn(1));
    [t0.join().expect("rank 0"), t1.join().expect("rank 1")]
}

#[test]
fn two_ranks_take_the_same_number_of_steps_and_end_on_the_same_weights() {
    for packed in [false, true] {
        let [(w0, steps0), (w1, steps1)] = two_ranks(packed);
        assert_eq!(
            steps0, steps1,
            "packed: {packed}: the ranks' step counts differ"
        );
        // Each rank trains half of the 8 samples (unpacked), or half of the
        // packed rows.
        assert!(steps0 >= 2, "packed: {packed}: {steps0} steps");
        let mut worst = 0.0f32;
        for ((name, a), (_, b)) in w0.iter().zip(&w1) {
            for (x, y) in a.iter().zip(b) {
                worst = worst.max((x - y).abs());
            }
            assert_eq!(a.len(), b.len(), "{name}");
        }
        assert!(
            worst < 1e-5,
            "packed: {packed}: the ranks' weights differ by {worst}: gradients were not averaged"
        );
        let moved = w0
            .iter()
            .any(|(name, v)| name.contains("lora_b") && v.iter().any(|x| *x != 0.0));
        assert!(moved, "packed: {packed}: nothing trained");
    }
}

/// A process outside an MLX distributed group gets no MLX transport.
#[test]
fn a_process_outside_an_mlx_group_has_no_mlx_transport() {
    let sync = DistributedGradientSync::from_mlx_group(&DistributedTrainingConfig::default())
        .expect("asking is not an error");
    assert!(sync.is_none());
}
