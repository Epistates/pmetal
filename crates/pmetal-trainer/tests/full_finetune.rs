//! Full-parameter fine-tuning through the one training loop.
//!
//! A tiny model is written as a checkpoint, loaded the way `pmetal train --full`
//! loads it, trained through `TrainingLoop::run`, and saved. What this holds:
//! the loss falls, the saved directory is a complete model the loader reads
//! back to the same logits, every parameter moved (the embedding, the norms
//! and the head included), a frozen embedding did not, and a resumed run picks
//! up the checkpoint it left.

use std::path::Path;
use std::sync::{Arc, Mutex};

use pmetal_core::{StepMetrics, TrainingCallback, TrainingConfig};
use pmetal_data::{DataLoaderConfig, Sample, TrainingDataset};
use pmetal_lora::{AdaptedModel, FullFineTune, TrainableModel};
use pmetal_merge::TensorWriter;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{Array, Dtype, ModuleParametersExt, random};
use pmetal_trainer::full_finetune::{plan_for_model, save_full_model};
use pmetal_trainer::{CheckpointManager, TrainingLoop, TrainingLoopConfig};

const LLAMA: &str = r#"{
    "model_type": "llama",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": false
}"#;

/// Qwen3 with its head tied to the embedding.
const QWEN3_TIED: &str = r#"{
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
}"#;

/// Write a randomly initialised model of `config_json` as a checkpoint in
/// `dir`: its `config.json`, a tokenizer-free weights file, and a side file
/// the saved model must carry over.
fn write_checkpoint(dir: &Path, config_json: &str) {
    random::seed(11);
    let model = DynamicModel::from_config(config_json).expect("builds");
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json).unwrap();
    std::fs::write(dir.join("chat_template.jinja"), "{{ messages }}").unwrap();
    let mut writer = TensorWriter::with_dtype(dir, safetensors::Dtype::F32).unwrap();
    let params = model.flatten_params();
    let mut names: Vec<_> = params.keys().collect();
    names.sort();
    for name in names {
        writer.write_tensor(name, &params[name]).unwrap();
    }
    writer.finalize().unwrap();
}

fn dataset() -> TrainingDataset {
    // The same few sequences over and over: a model can learn them.
    let samples: Vec<Sample> = (0..8)
        .map(|i| {
            Sample::new(
                (0..16)
                    .map(|j| (((i % 2) * 5 + j * 3) % 200 + 1) as u32)
                    .collect(),
            )
        })
        .collect();
    TrainingDataset::from_samples(samples)
}

fn loop_config(epochs: usize, checkpoint_every: usize) -> TrainingLoopConfig {
    TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 2e-3,
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
        checkpoint_every,
        eval_every: 0,
        use_sequence_packing: false,
        ..Default::default()
    }
}

#[derive(Clone, Default)]
struct Losses(Arc<Mutex<Vec<f64>>>);

impl TrainingCallback for Losses {
    fn on_step_end_with_metrics(&mut self, metrics: &StepMetrics) {
        self.0.lock().unwrap().push(metrics.loss);
    }
}

fn full_model(dir: &Path, options: FullFineTune) -> AdaptedModel {
    AdaptedModel::full(DynamicModel::load(dir).expect("loads"), options).expect("full")
}

fn logits(model: &mut DynamicModel) -> Array {
    let ids = Array::from_slice(
        &(0..16).map(|i| (i * 11 + 2) % 256).collect::<Vec<i32>>(),
        &[1, 16],
    );
    let out = model.forward(&ids, None).expect("forward");
    pmetal_runtime::check_unobserved_error().expect("no runtime error");
    out.as_dtype(Dtype::Float32.as_i32())
}

fn mean_abs_diff(a: &Array, b: &Array) -> f32 {
    a.subtract(b).abs().sum_all().item_f32() / a.shape().iter().product::<i32>() as f32
}

fn param_diffs(a: &AdaptedModel, b: &AdaptedModel) -> Vec<(String, f32)> {
    let (pa, pb) = (a.model().flatten_params(), b.model().flatten_params());
    let mut out: Vec<(String, f32)> = pa
        .iter()
        .map(|(name, value)| {
            (
                name.to_string(),
                value
                    .as_dtype(Dtype::Float32.as_i32())
                    .subtract(&pb[name].as_dtype(Dtype::Float32.as_i32()))
                    .abs()
                    .sum_all()
                    .item_f32(),
            )
        })
        .collect();
    out.sort_by(|x, y| x.0.cmp(&y.0));
    out
}

/// Train `model` for a few epochs, returning the losses of its steps.
fn train(model: &mut AdaptedModel, epochs: usize) -> Vec<f64> {
    let losses = Losses::default();
    let mut training_loop = TrainingLoop::new(loop_config(epochs, 0)).unwrap();
    training_loop.add_callback(Box::new(losses.clone()));
    training_loop
        .run(model, dataset(), None, None)
        .expect("trains");
    pmetal_runtime::check_unobserved_error().expect("no runtime error");
    losses.0.lock().unwrap().clone()
}

#[test]
fn the_loss_falls_every_parameter_moves_and_the_saved_model_reloads_to_the_same_logits() {
    for (name, config) in [("untied", LLAMA), ("tied", QWEN3_TIED)] {
        let dir = tempfile::tempdir().unwrap();
        let (source, out) = (dir.path().join("source"), dir.path().join("out"));
        write_checkpoint(&source, config);

        let initial = full_model(&source, FullFineTune::default());
        let mut model = full_model(&source, FullFineTune::default());
        let losses = train(&mut model, 6);
        assert!(losses.len() >= 12, "{name}: {} steps", losses.len());
        let (first, last) = (losses[0], *losses.last().unwrap());
        assert!(
            last < 0.8 * first,
            "{name}: the loss did not fall: {first} -> {last} ({losses:?})"
        );

        // Every parameter moved: the embedding, the norms, the head (or the tied
        // embedding that is the head) and every projection.
        let moved = param_diffs(&initial, &model);
        let unmoved: Vec<_> = moved.iter().filter(|(_, d)| *d == 0.0).collect();
        assert!(
            unmoved.is_empty(),
            "{name}: parameters that never moved: {unmoved:?}"
        );
        assert!(
            moved
                .iter()
                .any(|(n, _)| n.ends_with("embed_tokens.weight"))
        );
        assert!(moved.iter().any(|(n, _)| n.ends_with("norm.weight")));

        // Save, and load what was saved.
        let params = model.model().flatten_params();
        let plan = plan_for_model(&source, &params).expect("the layout can be saved");
        let weights = save_full_model(&params, &plan, &source, &out).expect("saves");
        assert_eq!(weights, out.join("model.safetensors"));
        for file in ["config.json", "chat_template.jinja"] {
            assert!(
                out.join(file).exists(),
                "{name}: {file} was not carried over"
            );
        }
        let config_back: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap())
                .unwrap();
        assert_eq!(config_back["torch_dtype"], "float32");

        let mut reloaded = DynamicModel::load(&out).expect("the saved model loads");
        let diff = mean_abs_diff(&logits(model.model_mut()), &logits(&mut reloaded));
        assert_eq!(
            diff, 0.0,
            "{name}: the reloaded model's logits differ by {diff}"
        );
        // ...and it is not the model that was started from.
        let mut original = DynamicModel::load(&source).unwrap();
        assert!(mean_abs_diff(&logits(&mut original), &logits(&mut reloaded)) > 1e-6);
    }
}

#[test]
fn a_saved_model_is_never_written_over_the_one_it_came_from() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");
    write_checkpoint(&source, LLAMA);
    let model = full_model(&source, FullFineTune::default());
    let params = model.model().flatten_params();
    let plan = plan_for_model(&source, &params).unwrap();
    let err = save_full_model(&params, &plan, &source, &source).unwrap_err();
    assert!(err.to_string().contains("the model's own"), "{err}");
    // The weights are as they were.
    assert!(DynamicModel::load(&source).is_ok());
}

#[test]
fn a_frozen_embedding_is_saved_as_it_was() {
    let dir = tempfile::tempdir().unwrap();
    let (source, out) = (dir.path().join("source"), dir.path().join("out"));
    write_checkpoint(&source, LLAMA);
    let initial = full_model(&source, FullFineTune::default());
    let mut model = full_model(
        &source,
        FullFineTune {
            freeze_embeddings: true,
        },
    );
    train(&mut model, 3);

    let moved = param_diffs(&initial, &model);
    for (name, diff) in &moved {
        if name == "model.embed_tokens.weight" {
            assert_eq!(*diff, 0.0, "the frozen embedding moved");
        } else {
            assert!(*diff > 0.0, "{name} never moved");
        }
    }
    let params = model.model().flatten_params();
    let plan = plan_for_model(&source, &params).unwrap();
    save_full_model(&params, &plan, &source, &out).unwrap();
    let mut reloaded = DynamicModel::load(&out).unwrap();
    assert_eq!(
        mean_abs_diff(&logits(model.model_mut()), &logits(&mut reloaded)),
        0.0
    );
}

#[test]
fn a_resumed_run_continues_from_the_checkpoint_it_left() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");
    write_checkpoint(&source, LLAMA);
    let manager = CheckpointManager::new(dir.path().join("checkpoints"))
        .unwrap()
        .with_max_checkpoints(1);

    // Two epochs of 4 steps, a checkpoint every 4.
    let mut model = full_model(&source, FullFineTune::default());
    let mut training_loop = TrainingLoop::new(loop_config(2, 4)).unwrap();
    training_loop
        .run(&mut model, dataset(), None, Some(&manager))
        .unwrap();
    assert!(
        manager.latest_dir().unwrap().is_some(),
        "no checkpoint written"
    );

    // A fresh model resumes into the checkpoint's weights, not the source's.
    let mut resumed = full_model(&source, FullFineTune::default());
    let mut resuming = TrainingLoop::new(loop_config(2, 4)).unwrap();
    assert!(resuming.resume_from(&mut resumed, &manager).unwrap());
    let from_source = param_diffs(&full_model(&source, FullFineTune::default()), &resumed);
    assert!(
        from_source.iter().all(|(_, d)| *d > 0.0),
        "the resumed model is the source's: {from_source:?}"
    );
    // The checkpoint is of the model as it was when it was written; a model
    // trained on past it differs from it only by those steps.
    assert!(resuming.current_step() > 0);
    assert_eq!(
        TrainableModel::num_trainable_params(&resumed),
        TrainableModel::num_trainable_params(&model)
    );
}
