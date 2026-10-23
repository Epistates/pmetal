//! Training is deterministic: two runs of the same steps from the same seed end
//! on bit-identical weights.
//!
//! The parameter maps are ordered (`BTreeMap`), so every walk over parameters
//! (the global gradient norm, the order autograd's arguments go in, the
//! optimizer update) is in key order. A `HashMap` iterates in an order that
//! differs per map, which let two runs differ in the last bits.
//!
//! The ignored test at the end times a training step on a small model, for
//! comparing the cost of the ordered maps against the unordered ones.

use std::time::Instant;

use pmetal_core::{LoraConfig, TrainingConfig};
use pmetal_data::{DataLoaderConfig, Sample, TrainingDataset};
use pmetal_lora::{AdaptedModel, FullFineTune};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{Dtype, ModuleParametersExt, random};
use pmetal_trainer::{TrainingLoop, TrainingLoopConfig};

fn llama(hidden: usize, layers: usize, vocab: usize) -> String {
    format!(
        r#"{{
    "model_type": "llama",
    "vocab_size": {vocab},
    "hidden_size": {hidden},
    "intermediate_size": {inter},
    "num_hidden_layers": {layers},
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": false
}}"#,
        inter = hidden * 2
    )
}

/// Qwen3 with its head tied to the embedding.
fn qwen3_tied() -> String {
    r#"{
    "model_type": "qwen3",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": true
}"#
    .to_string()
}

fn dataset(samples: usize, vocab: u32) -> TrainingDataset {
    TrainingDataset::from_samples(
        (0..samples)
            .map(|i| {
                Sample::new(
                    (0..16)
                        .map(|j| ((i as u32 * 7 + j * 3) % (vocab - 1)) + 1)
                        .collect(),
                )
            })
            .collect(),
    )
}

fn loop_config(epochs: usize) -> TrainingLoopConfig {
    TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-2,
            batch_size: 2,
            num_epochs: epochs,
            gradient_accumulation_steps: 1,
            warmup_steps: 0,
            max_grad_norm: 1.0,
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 16,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        use_sequence_packing: false,
        ..Default::default()
    }
}

/// How the model under test is trained.
#[derive(Clone, Copy)]
enum Mode {
    Full,
    Lora,
}

fn train(config_json: &str, mode: Mode, epochs: usize, vocab: u32) -> AdaptedModel {
    random::seed(11);
    let base = DynamicModel::from_config(config_json).expect("builds");
    let mut model = match mode {
        Mode::Full => AdaptedModel::full(base, FullFineTune::default()).expect("full"),
        Mode::Lora => AdaptedModel::attach(
            base,
            LoraConfig {
                r: 4,
                alpha: 8.0,
                dropout: 0.0,
                ..Default::default()
            },
        )
        .expect("attach"),
    };
    let mut training_loop = TrainingLoop::new(loop_config(epochs)).unwrap();
    training_loop
        .run(&mut model, dataset(8, vocab), None, None)
        .expect("trains");
    pmetal_runtime::check_unobserved_error().expect("no runtime error");
    model
}

/// Every parameter's bits, by name.
fn weights(model: &AdaptedModel) -> Vec<(String, Vec<u32>)> {
    model
        .model()
        .flatten_params()
        .iter()
        .map(|(name, value)| {
            let f32_ = value.as_dtype(Dtype::Float32.as_i32());
            f32_.eval();
            let bits = f32_
                .as_slice::<f32>()
                .iter()
                .map(|v: &f32| v.to_bits())
                .collect();
            (name.to_string(), bits)
        })
        .collect()
}

fn assert_identical(config_json: &str, mode: Mode, vocab: u32, label: &str) {
    let first = weights(&train(config_json, mode, 3, vocab));
    let second = weights(&train(config_json, mode, 3, vocab));
    assert_eq!(first.len(), second.len(), "{label}: parameter count");
    assert!(first.len() > 4, "{label}: only {} parameters", first.len());
    for ((name, a), (_, b)) in first.iter().zip(&second) {
        assert!(a == b, "{label}: `{name}` differs between two runs");
    }
}

#[test]
fn full_fine_tuning_is_bit_identical_run_to_run() {
    assert_identical(&llama(64, 2, 256), Mode::Full, 256, "llama full");
    assert_identical(&qwen3_tied(), Mode::Full, 256, "qwen3 tied full");
}

#[test]
fn lora_training_is_bit_identical_run_to_run() {
    assert_identical(&llama(64, 2, 256), Mode::Lora, 256, "llama lora");
}

/// Step time of a small model, for before/after comparisons.
/// `cargo test --release -p pmetal-trainer --test determinism -- --ignored --nocapture`
#[test]
#[ignore = "timing report"]
fn step_time_report() {
    let config = llama(256, 8, 4096);
    for (label, mode) in [("full", Mode::Full), ("lora", Mode::Lora)] {
        let mut per_step = Vec::new();
        // The first run warms the kernel caches.
        for epochs in [1, 6, 6, 6] {
            random::seed(11);
            let base = DynamicModel::from_config(&config).expect("builds");
            let mut model = match mode {
                Mode::Full => AdaptedModel::full(base, FullFineTune::default()).unwrap(),
                Mode::Lora => AdaptedModel::attach(
                    base,
                    LoraConfig {
                        r: 8,
                        alpha: 16.0,
                        dropout: 0.0,
                        ..Default::default()
                    },
                )
                .unwrap(),
            };
            let mut training_loop = TrainingLoop::new(loop_config(epochs)).unwrap();
            let started = Instant::now();
            training_loop
                .run(&mut model, dataset(16, 4096), None, None)
                .unwrap();
            let steps = training_loop.current_step().max(1);
            per_step.push((started.elapsed().as_secs_f64() * 1e3 / steps as f64, epochs));
        }
        let timed: Vec<f64> = per_step.iter().skip(1).map(|(ms, _)| *ms).collect();
        let best = timed.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("step_time {label}: runs(ms/step)={timed:.2?} best={best:.2}");
    }
}
