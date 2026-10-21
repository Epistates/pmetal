use super::source::{BatchSource, BatchView, MicroBatch};
use super::step::{GradAccumulator, LossSpec, loss_and_grads};
use super::*;
use pmetal_core::{
    LoraConfig, LrSchedulerType, OptimizerType, StepMetrics, TrainingCallback, TrainingConfig,
};
use pmetal_data::{DataLoaderConfig, PackedTrainingBatch, Sample, TrainingBatch, TrainingDataset};
use pmetal_lora::AdaptedModel;
use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::module::ModuleParameters;
use pmetal_runtime::compat::{ModuleParametersExt, random};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The trainable model these tests run against: the dispatcher's own Llama,
/// with adapters attached, which is what `pmetal train` builds.
fn small_model() -> AdaptedModel {
    let base = DynamicModel::from_config(&serde_json::to_string(&small_config()).unwrap())
        .expect("llama builds");
    AdaptedModel::attach(base, small_lora_config()).expect("attach adapters")
}

fn small_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 1000,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: None,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        ..Default::default()
    }
}

fn small_lora_config() -> LoraConfig {
    LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        use_rslora: false,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ],
        bias: pmetal_core::LoraBias::None,
        init_lora_weights: pmetal_core::LoraInit::Standard,
        use_dora: false,
        loraplus_lr_ratio: None,
    }
}

/// The small model with every adapter nonzero, so each has a gradient.
fn seeded_model(seed: u64) -> AdaptedModel {
    random::seed(seed);
    let mut model = small_model();
    let params = model
        .lora_parameters()
        .into_iter()
        .map(|(k, v)| (k, v.add(&Array::from_f32(0.01))))
        .collect();
    model.set_lora_parameters(&params);
    model
}

fn create_dummy_dataset(num_samples: usize, seq_len: usize) -> TrainingDataset {
    let samples: Vec<Sample> = (0..num_samples)
        .map(|i| {
            let input_ids: Vec<u32> = (0..seq_len).map(|j| ((i + j) % 256) as u32).collect();
            Sample::new(input_ids)
        })
        .collect();
    TrainingDataset::from_samples(samples)
}

fn single_sequence_packed_batch(tokens: &[i32], labels: &[i64]) -> PackedTrainingBatch {
    let len = tokens.len() as i32;
    let position_ids: Vec<i32> = (0..len).collect();
    let cu_seqlens_raw = vec![0_i32, len];
    PackedTrainingBatch {
        input_ids: Array::from_i32_slice_shaped(tokens, &[len]),
        position_ids: Array::from_i32_slice_shaped(&position_ids, &[len]),
        cu_seqlens: Array::from_i32_slice_shaped(&cu_seqlens_raw, &[2]),
        cu_seqlens_raw,
        labels: Array::from_i32_slice_shaped(
            &labels.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            &[len],
        ),
        total_tokens: tokens.len(),
        num_sequences: 1,
        max_seqlen: tokens.len(),
    }
}

/// Two sequences of four, packed into one row with restarting positions.
fn two_sequence_packed_batch() -> PackedTrainingBatch {
    let tokens = [1_i32, 2, 3, 4, 5, 6, 7, 8];
    let labels = [2_i32, 3, 4, -100, 6, 7, 8, -100];
    let positions = [0_i32, 1, 2, 3, 0, 1, 2, 3];
    let cu_seqlens_raw = vec![0_i32, 4, 8];
    PackedTrainingBatch {
        input_ids: Array::from_i32_slice_shaped(&tokens, &[8]),
        position_ids: Array::from_i32_slice_shaped(&positions, &[8]),
        cu_seqlens: Array::from_i32_slice_shaped(&cu_seqlens_raw, &[3]),
        cu_seqlens_raw,
        labels: Array::from_i32_slice_shaped(&labels, &[8]),
        total_tokens: 8,
        num_sequences: 2,
        max_seqlen: 4,
    }
}

/// A padded batch of one row, with `labels` (use -100 to mask a token).
fn row_batch(tokens: &[i32], labels: &[i32]) -> TrainingBatch {
    let len = tokens.len() as i32;
    TrainingBatch {
        input_ids: Array::from_i32_slice_shaped(tokens, &[1, len]),
        labels: Array::from_i32_slice_shaped(labels, &[1, len]),
        attention_mask: Array::from_i32_slice_shaped(&vec![1; tokens.len()], &[1, len]),
        pixel_values: None,
        batch_size: 1,
        seq_len: tokens.len(),
        sample_indices: vec![0],
    }
}

fn plain_spec() -> LossSpec {
    LossSpec {
        label_smoothing: 0.0,
        z_loss: 0.0,
        neftune_alpha: None,
        cut_cross_entropy: false,
    }
}

fn scalar(a: &Array) -> f32 {
    a.eval();
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
    a.item_f32()
}

/// A named tensor's largest absolute value.
fn max_abs(a: &Array) -> f32 {
    scalar(&a.abs().max(None))
}

#[derive(Clone, Default)]
struct MetricsCapture {
    steps: Arc<Mutex<Vec<StepMetrics>>>,
}

impl MetricsCapture {
    fn snapshot(&self) -> Vec<StepMetrics> {
        self.steps.lock().expect("metrics capture lock").clone()
    }
}

impl TrainingCallback for MetricsCapture {
    fn on_step_end_with_metrics(&mut self, metrics: &StepMetrics) {
        self.steps
            .lock()
            .expect("metrics capture lock")
            .push(metrics.clone());
    }
}

fn config_of(
    training: TrainingConfig,
    batch_size: usize,
    max_seq_len: usize,
) -> TrainingLoopConfig {
    TrainingLoopConfig {
        training,
        dataloader: DataLoaderConfig {
            batch_size,
            max_seq_len,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        ..Default::default()
    }
}

fn simple_training(max_steps: Option<usize>) -> TrainingConfig {
    TrainingConfig {
        learning_rate: 1e-4,
        batch_size: 1,
        gradient_accumulation_steps: 1,
        num_epochs: 1,
        max_steps,
        max_grad_norm: 1.0,
        ..Default::default()
    }
}

#[test]
fn test_training_loop_creation() {
    let training_loop = TrainingLoop::new(TrainingLoopConfig::default()).unwrap();

    assert_eq!(training_loop.current_step(), 0);
    assert_eq!(training_loop.current_epoch(), 0);
}

/// A loop is never built on a configuration that cannot work: the training
/// section's own checks run in the constructor, and so do the loop's.
#[test]
fn a_loop_refuses_a_configuration_that_cannot_work() {
    let refused = |config: TrainingLoopConfig| match TrainingLoop::new(config) {
        Ok(_) => panic!("the loop accepted a configuration that cannot work"),
        Err(e) => e.to_string(),
    };
    let mut config = TrainingLoopConfig::default();
    config.training.label_smoothing = 1.5;
    assert!(refused(config).contains("label_smoothing"));

    let mut config = TrainingLoopConfig::default();
    config.training.gradient_accumulation_steps = 0;
    assert!(refused(config).contains("gradient_accumulation_steps"));

    let mut config = TrainingLoopConfig::default();
    config.dataloader.batch_size = 0;
    assert!(refused(config).contains("batch_size"));

    let mut config = TrainingLoopConfig::default();
    config.dataloader.world_size = 2;
    config.dataloader.rank = 2;
    assert!(refused(config).contains("rank 2 is not in a world of 2"));

    let mut config = TrainingLoopConfig::default();
    config.training.z_loss = f64::NAN;
    assert!(refused(config).contains("z_loss"));
}

#[test]
fn test_learning_rate_warmup() {
    let mut config = TrainingLoopConfig::default();
    config.training.warmup_steps = 100;
    config.training.max_steps = Some(1000);
    config.training.learning_rate = 1e-4;
    config.training.lr_scheduler = LrSchedulerType::Cosine;

    let mut training_loop = TrainingLoop::new(config).unwrap();

    // At step 0
    training_loop.step = 0;
    let lr0 = training_loop.get_learning_rate();
    assert!(lr0 < 1e-4);

    // At step 50 (halfway through warmup)
    training_loop.step = 50;
    let lr50 = training_loop.get_learning_rate();
    assert!((lr50 - 5e-5).abs() < 1e-8);

    // At step 100 (end of warmup)
    training_loop.step = 100;
    let lr100 = training_loop.get_learning_rate();
    assert!((lr100 - 1e-4).abs() < 1e-8);
}

/// Schedule-free AdamW trains at a constant rate after warmup whatever
/// schedule the config names: a decaying one on top of iterate averaging
/// undoes what the averaging is for.
#[test]
fn schedule_free_adamw_trains_at_a_constant_rate_after_warmup() {
    let mut config = TrainingLoopConfig::default();
    config.training.optimizer = OptimizerType::AdamWScheduleFree;
    config.training.lr_scheduler = LrSchedulerType::Cosine;
    config.training.warmup_steps = 10;
    config.training.max_steps = Some(100);
    config.training.learning_rate = 1e-3;
    let mut training_loop = TrainingLoop::new(config).unwrap();
    for step in [10, 50, 99] {
        training_loop.step = step;
        assert!(
            (training_loop.get_learning_rate() - 1e-3).abs() < 1e-9,
            "step {step}"
        );
    }
}

#[test]
fn test_take_log_interval_metrics_uses_actual_logged_steps() {
    let config = TrainingLoopConfig {
        log_every: 10,
        ..Default::default()
    };
    let mut training_loop = TrainingLoop::new(config).unwrap();

    training_loop.step = 1;
    training_loop.reset_log_interval();
    training_loop.last_log_time = Some(Instant::now() - Duration::from_millis(25));
    training_loop.tokens_since_log = 450;
    training_loop.step = 10;

    let metrics = training_loop.take_log_interval_metrics(Instant::now());

    assert_eq!(metrics.steps, 9);
    assert_eq!(metrics.tokens, 450);
    assert!(metrics.total_ms > 0.0);
    assert!(metrics.tok_sec > 0.0);
    assert_eq!(training_loop.tokens_since_log, 0);
    assert_eq!(training_loop.last_log_step, Some(10));
}

#[test]
fn test_learning_rate_division_by_zero_protection() {
    // Test edge case where total_steps == warmup_steps
    let mut config = TrainingLoopConfig::default();
    config.training.warmup_steps = 100;
    config.training.max_steps = Some(100); // Same as warmup!
    config.training.learning_rate = 1e-4;
    config.training.lr_scheduler = LrSchedulerType::Linear;

    let mut training_loop = TrainingLoop::new(config).unwrap();

    // At step 100 (past warmup, at max_steps)
    training_loop.step = 100;
    let lr = training_loop.get_learning_rate();

    // Should not panic or return NaN/Inf
    assert!(lr.is_finite(), "Learning rate should be finite, got {}", lr);
    assert!(lr >= 0.0, "Learning rate should be non-negative");
}

#[test]
fn test_batch_token_overflow_protection() {
    // The tokens of a batch are counted with saturating arithmetic.
    let batch = TrainingBatch {
        batch_size: usize::MAX / 2,
        seq_len: 3,
        ..row_batch(&[1, 2], &[2, 3])
    };
    assert_eq!(BatchView::Padded(&batch).tokens(), usize::MAX);
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

/// A tiny Qwen 3.5 hybrid: three gated-delta-net layers and one attention layer.
fn small_hybrid_model() -> AdaptedModel {
    use pmetal_models::architectures::qwen3_next::Qwen3NextConfig;
    let config = Qwen3NextConfig {
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: Some(1),
        head_dim: Some(16),
        vocab_size: 100,
        linear_num_value_heads: 2,
        linear_num_key_heads: 1,
        linear_key_head_dim: 32,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 16,
        shared_expert_intermediate_size: 32,
        mlp_only_layers: vec![],
        norm_topk_prob: false,
        tie_word_embeddings: true,
        ..Default::default()
    };
    let json = serde_json::json!({ "model_type": "qwen3_next", "text_config": config });
    let base = DynamicModel::from_config(&json.to_string()).expect("qwen3_next builds");
    AdaptedModel::attach(base, small_lora_config()).expect("attach adapters")
}

fn packing_config(learning_rate: f64) -> TrainingLoopConfig {
    TrainingLoopConfig {
        use_sequence_packing: true,
        pack_max_seq_len: Some(64),
        ..config_of(
            TrainingConfig {
                learning_rate,
                ..simple_training(None)
            },
            1,
            64,
        )
    }
}

fn short_sequences() -> TrainingDataset {
    let samples: Vec<Sample> = (0..6)
        .map(|i| Sample::new((0..8).map(|j| ((i * 7 + j) % 90 + 1) as u32).collect()))
        .collect();
    TrainingDataset::from_samples(samples)
}

/// Packing puts several sequences in one row. A gated-delta-net layer would
/// carry the first one's state into the second, so a model with recurrent
/// layers trains on unpacked batches: one step per sequence here, where the
/// attention-only control packs them into fewer steps.
#[test]
fn a_run_does_not_pack_models_with_recurrent_layers() {
    let mut hybrid = small_hybrid_model();
    assert!(hybrid.has_recurrent_layers());
    let mut training_loop = TrainingLoop::new(packing_config(1e-4)).unwrap();
    training_loop
        .run(&mut hybrid, short_sequences(), None, None)
        .unwrap();
    pmetal_runtime::check_unobserved_error().unwrap();
    assert_eq!(
        training_loop.current_step(),
        6,
        "a model with recurrent layers must train on one unpacked sequence per step"
    );

    let mut dense = small_model();
    assert!(!dense.has_recurrent_layers());
    let mut training_loop = TrainingLoop::new(packing_config(1e-4)).unwrap();
    training_loop
        .run(&mut dense, short_sequences(), None, None)
        .unwrap();
    assert!(
        training_loop.current_step() < 6,
        "the attention-only control should pack, took {} steps",
        training_loop.current_step()
    );
}

/// A tiny Nemotron-H: two Mamba-2 layers, attention and an MLP, with adapters
/// on the Mamba projections as well as attention.
fn small_nemotron_model() -> AdaptedModel {
    let config = serde_json::json!({
        "model_type": "nemotron_h",
        "vocab_size": 100,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "mamba_num_heads": 4,
        "mamba_head_dim": 16,
        "ssm_state_size": 8,
        "n_groups": 2,
        "chunk_size": 4,
        "hybrid_override_pattern": "M*M-",
        "tie_word_embeddings": true,
    });
    let base = DynamicModel::from_config(&config.to_string()).expect("nemotron_h builds");
    let mut lora = small_lora_config();
    lora.target_modules
        .extend(["in_proj".to_string(), "out_proj".to_string()]);
    AdaptedModel::attach(base, lora).expect("attach adapters")
}

/// A Mamba-2 layer resets where the position ids restart, so the sequences
/// of a packed row cannot see each other and Nemotron-H packs, where a gated
/// delta net does not (above). The adapters on its Mamba projections train
/// through the packed scan.
#[test]
fn a_run_packs_models_whose_recurrent_layers_reset() {
    let mut model = small_nemotron_model();
    assert!(
        !model.has_recurrent_layers(),
        "a Mamba-2 layer resets at a packed boundary on its own"
    );
    let mut training_loop = TrainingLoop::new(packing_config(1e-2)).unwrap();
    training_loop
        .run(&mut model, short_sequences(), None, None)
        .unwrap();
    pmetal_runtime::check_unobserved_error().unwrap();
    assert!(
        training_loop.current_step() < 6,
        "Nemotron-H should pack, took {} steps",
        training_loop.current_step()
    );

    // The adapter on the Mamba input projection was trained: its `B`, which
    // starts at zero, moved.
    let params = model.flatten_params();
    let moved = params
        .iter()
        .filter(|(key, _)| key.contains("mixer.in_proj") && key.contains("lora_b"))
        .map(|(_, b)| b.abs().max(None).item::<f32>())
        .fold(0.0f32, f32::max);
    assert!(
        moved > 0.0,
        "no adapter on a Mamba projection moved ({} parameters)",
        params.len()
    );
}

#[test]
fn a_run_trains_unpacked_when_no_sequences_are_combined() {
    // Each sample fills its own row, so packing combines nothing.
    let config = TrainingLoopConfig {
        use_sequence_packing: true,
        ..config_of(simple_training(Some(1)), 1, 8)
    };
    let mut training_loop = TrainingLoop::new(config).unwrap();
    let mut model = small_model();
    training_loop
        .run(&mut model, create_dummy_dataset(3, 8), None, None)
        .unwrap();
    assert_eq!(training_loop.current_step(), 1);
}

/// Packed rows go through accumulation like padded batches do, and NEFTune
/// and the optimizer's step count see them: a packed run of 6 rows with 2
/// rows to a step takes 3 optimizer steps.
#[test]
fn a_packed_run_accumulates_gradients() {
    let mut config = packing_config(1e-3);
    config.training.gradient_accumulation_steps = 2;
    config.dataloader.shuffle = false;
    // Rows of one 8-token sample each: pack at 8 so they do not merge.
    config.pack_max_seq_len = Some(8);
    let mut training_loop = TrainingLoop::new(config).unwrap();
    let mut model = small_model();
    // 4 rows, 2 per optimizer step. (Each row is a single sequence, so the
    // loop notices nothing combines and trains the same rows unpacked.)
    training_loop
        .run(&mut model, short_sequences(), None, None)
        .unwrap();
    assert_eq!(training_loop.current_step(), 6);
    assert_eq!(training_loop.optimizer_steps_taken(), 3);
}

#[test]
fn a_run_enables_gradient_checkpointing_the_model_supports() {
    for packing in [false, true] {
        let config = TrainingLoopConfig {
            gradient_checkpointing: true,
            gradient_checkpointing_layers: 3,
            use_sequence_packing: packing,
            ..config_of(simple_training(Some(1)), 1, 8)
        };
        let mut training_loop = TrainingLoop::new(config).unwrap();
        let mut model = small_model();
        training_loop
            .run(&mut model, create_dummy_dataset(4, 2), None, None)
            .unwrap();
        assert!(
            model.model().is_gradient_checkpointing(),
            "a run (packing: {packing}) declined checkpointing on a model that implements it"
        );
    }
}

#[test]
fn a_packed_run_reports_non_zero_interval_metrics() {
    let mut config = packing_config(1e-4);
    config.training.max_steps = Some(3);
    config.pack_max_seq_len = None;
    config.dataloader.max_seq_len = 32;
    let mut training_loop = TrainingLoop::new(config).unwrap();
    let capture = MetricsCapture::default();
    let snapshot = capture.clone();
    training_loop.add_callback(Box::new(capture));

    let mut model = small_model();
    training_loop
        .run(&mut model, create_dummy_dataset(6, 24), None, None)
        .unwrap();

    let metrics = snapshot.snapshot();
    assert!(!metrics.is_empty());
    assert!(metrics.iter().all(|metric| metric.tokens > 0));
    assert!(metrics.iter().all(|metric| metric.total_ms > 0.0));
    assert!(metrics.iter().all(|metric| metric.tok_sec > 0.0));
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

#[test]
fn test_single_train_step() {
    use pmetal_runtime::compat::optimizers::Sgd;

    let mut training_loop = TrainingLoop::new(TrainingLoopConfig::default()).unwrap();
    let mut model = small_model();
    let mut optimizer = Sgd::new(1e-4);
    let batch = row_batch(&[1, 2, 3, 4], &[2, 3, 4, 5]);

    let stats = training_loop
        .train_step(&mut model, &batch, &mut optimizer)
        .unwrap();

    assert!(stats.loss > 0.0);
    assert_eq!(stats.step, 1);
    assert_eq!(training_loop.current_step(), 1);
}

/// With gradient accumulation the optimizer steps on the last micro-batch of
/// each group, and the step before it reports no gradient norm.
#[test]
fn a_train_step_applies_gradients_on_the_last_micro_batch() {
    use pmetal_runtime::compat::optimizers::Sgd;
    let mut config = TrainingLoopConfig::default();
    config.training.gradient_accumulation_steps = 2;
    let mut training_loop = TrainingLoop::new(config).unwrap();
    let mut model = small_model();
    let mut optimizer = Sgd::new(1e-4);
    let batch = row_batch(&[1, 2, 3, 4], &[2, 3, 4, 5]);

    let first = training_loop
        .train_step(&mut model, &batch, &mut optimizer)
        .unwrap();
    assert!(
        first.grad_norm.is_none(),
        "the first micro-batch only accumulates"
    );
    let second = training_loop
        .train_step(&mut model, &batch, &mut optimizer)
        .unwrap();
    assert!(
        second.grad_norm.is_some_and(f32::is_finite),
        "the second applies the accumulated gradients"
    );
    assert_eq!(training_loop.current_step(), 2);
}

/// The gradient of an optimizer step is the gradient of the mean over every
/// token in it, however its micro-batches divide them. Two micro-batches of 3
/// and 7 counted tokens, accumulated, step exactly as the one batch holding
/// both does; averaging their means (what scaling each by `1/n` computes)
/// would weight the short one's tokens more than twice the long one's.
#[test]
fn accumulation_weights_every_token_equally() {
    use pmetal_runtime::compat::optimizers::Sgd;

    // Two rows; the first masks all but its last 3 labels, so the rows count
    // 3 and 7 tokens. Padding follows the tokens, which a causal model
    // never lets reach them, so the unaccumulated batch is the reference.
    let (short_tokens, short_labels) = ([11, 12, 13, 14, 15], [-100, -100, 13, 14, 15]);
    let long_tokens = [21, 22, 23, 24, 25, 26, 27, 28];
    let long_labels = [-100, 22, 23, 24, 25, 26, 27, 28];
    let first = row_batch(&short_tokens, &short_labels);
    let second = row_batch(&long_tokens, &long_labels);

    let mut padded_tokens = short_tokens.to_vec();
    padded_tokens.extend([0, 0, 0]);
    let mut padded_labels = short_labels.to_vec();
    padded_labels.extend([-100, -100, -100]);
    let both = TrainingBatch {
        input_ids: Array::from_i32_slice_shaped(
            &[padded_tokens.clone(), long_tokens.to_vec()].concat(),
            &[2, 8],
        ),
        labels: Array::from_i32_slice_shaped(
            &[padded_labels.clone(), long_labels.to_vec()].concat(),
            &[2, 8],
        ),
        attention_mask: Array::from_i32_slice_shaped(&[1; 16], &[2, 8]),
        pixel_values: None,
        batch_size: 2,
        seq_len: 8,
        sample_indices: vec![0, 1],
    };

    let train = |accumulate: usize, batches: &[&TrainingBatch]| {
        let mut config = TrainingLoopConfig::default();
        config.training.gradient_accumulation_steps = accumulate;
        config.training.max_grad_norm = 0.0;
        let mut training_loop = TrainingLoop::new(config).unwrap();
        let mut model = seeded_model(77);
        let mut optimizer = Sgd::new(1.0);
        let mut last = None;
        for batch in batches {
            last = Some(
                training_loop
                    .train_step(&mut model, batch, &mut optimizer)
                    .unwrap(),
            );
        }
        (model.lora_parameters(), last.unwrap().loss)
    };
    let (accumulated, accumulated_loss) = train(2, &[&first, &second]);
    let (whole, whole_loss) = train(1, &[&both]);
    assert!(
        (accumulated_loss - whole_loss).abs() <= 1e-5 * whole_loss.abs(),
        "mean loss {accumulated_loss} vs {whole_loss}"
    );
    let mut worst = 0.0f32;
    for (key, value) in &whole {
        worst = worst.max(max_abs(&accumulated[key].subtract(value)));
    }
    assert!(
        worst < 1e-5,
        "accumulated step differs from the whole batch's by {worst}"
    );
    pmetal_runtime::check_unobserved_error().unwrap();
}

/// Clipping and the token mean in one multiplication: the norm reported is
/// the unclipped mean gradient's, and the clipped gradient has norm
/// `max_norm`.
#[test]
fn the_accumulator_divides_by_the_token_count_and_clips_once() {
    use pmetal_runtime::compat::Dtype;
    let micro = |g: [f32; 2], loss: f32, count: f32| super::step::MicroOutput {
        loss_sum: Array::from_f32(loss),
        count: Array::from_f32(count),
        grads: [(std::rc::Rc::from("w"), Array::from_f32_slice(&g, &[2]))].into(),
    };
    let mut acc = GradAccumulator::default();
    acc.add(micro([3.0, 4.0], 2.0, 2.0));
    acc.add(micro([3.0, 4.0], 6.0, 2.0));
    assert_eq!(acc.micro_batches(), 2);
    // Sum [6, 8] over 4 tokens: the mean gradient is [1.5, 2], norm 2.5.
    let finished = acc.finish(1.0).expect("something was accumulated");
    let norm = scalar(&finished.grad_norm.expect("clipping is on"));
    assert!((norm - 2.5).abs() < 1e-5, "norm {norm}");
    let g = &finished.grads["w"];
    g.eval();
    let got = g
        .as_dtype(Dtype::Float32.as_i32())
        .as_slice::<f32>()
        .to_vec();
    assert!(
        (got[0] - 0.6).abs() < 1e-5 && (got[1] - 0.8).abs() < 1e-5,
        "{got:?}"
    );
    assert!(
        (scalar(&finished.loss) - 2.0).abs() < 1e-6,
        "mean loss is 8/4"
    );
    assert_eq!(acc.micro_batches(), 0, "finishing resets it");

    // Below the clip the gradient is just the mean.
    let mut acc = GradAccumulator::default();
    acc.add(micro([3.0, 4.0], 2.0, 2.0));
    let finished = acc.finish(100.0).unwrap();
    let g = &finished.grads["w"];
    g.eval();
    let got = g.as_slice::<f32>().to_vec();
    assert!(
        (got[0] - 1.5).abs() < 1e-6 && (got[1] - 2.0).abs() < 1e-6,
        "{got:?}"
    );

    // No clipping when it is off, and no norm computed.
    let mut acc = GradAccumulator::default();
    acc.add(micro([3.0, 4.0], 2.0, 2.0));
    assert!(acc.finish(0.0).unwrap().grad_norm.is_none());
}

/// A single-sequence packed batch is the padded batch it holds.
#[test]
fn a_single_sequence_packed_step_matches_the_padded_step() {
    let tokens = [1_i32, 2, 3, 4, 5, 6];
    let labels = [2_i64, 3, 4, 5, 6, 7];
    let padded = row_batch(&tokens, &labels.map(|l| l as i32));
    let packed = single_sequence_packed_batch(&tokens, &labels);

    let mut a = seeded_model(1337);
    let mut b = seeded_model(1337);
    let from_padded = loss_and_grads(&mut a, BatchView::Padded(&padded), &plain_spec()).unwrap();
    let from_packed = loss_and_grads(&mut b, BatchView::Packed(&packed), &plain_spec()).unwrap();
    let (x, y) = (scalar(&from_padded.loss_sum), scalar(&from_packed.loss_sum));
    assert!((x - y).abs() < 1e-4, "padded {x} vs packed {y}");
    assert_eq!(scalar(&from_padded.count), scalar(&from_packed.count));
    for (key, g) in &from_padded.grads {
        let diff = max_abs(&g.subtract(&from_packed.grads[key]));
        assert!(diff < 1e-4, "{key}: gradients differ by {diff}");
    }
}

/// Label smoothing and z-loss reach the loss on every path: they change it,
/// the full-logits and cut cross-entropy paths agree on the changed value and
/// its gradients, and the loss without them is untouched.
#[test]
fn label_smoothing_and_z_loss_reach_every_loss_path() {
    let batch = two_sequence_packed_batch();
    let spec = |smoothing: f32, z: f32, cce: bool| LossSpec {
        label_smoothing: smoothing,
        z_loss: z,
        neftune_alpha: None,
        cut_cross_entropy: cce,
    };
    let run = |spec: LossSpec| {
        let mut model = seeded_model(9);
        loss_and_grads(&mut model, BatchView::Packed(&batch), &spec).unwrap()
    };
    let plain = run(spec(0.0, 0.0, false));
    let smoothed = run(spec(0.1, 0.0, false));
    let zloss = run(spec(0.0, 1e-2, false));
    let (p, s, z) = (
        scalar(&plain.loss_sum),
        scalar(&smoothed.loss_sum),
        scalar(&zloss.loss_sum),
    );
    assert!(
        (s - p).abs() > 1e-3 * p,
        "smoothing left the loss at {p} ({s})"
    );
    assert!(z > p, "z-loss adds a positive term: {p} -> {z}");

    for (smoothing, z_weight) in [(0.1_f32, 0.0_f32), (0.0, 1e-2), (0.1, 1e-2)] {
        let full = run(spec(smoothing, z_weight, false));
        let cut = run(spec(smoothing, z_weight, true));
        let (f, c) = (scalar(&full.loss_sum), scalar(&cut.loss_sum));
        assert!(
            (f - c).abs() <= 1e-4 * f.abs(),
            "smoothing {smoothing}, z {z_weight}: full {f} vs cut {c}"
        );
        for (key, g) in &full.grads {
            let diff = max_abs(&g.subtract(&cut.grads[key]));
            assert!(diff < 1e-4, "{key}: gradients differ by {diff}");
        }
    }
    pmetal_runtime::check_unobserved_error().unwrap();
}

/// NEFTune noises the forward being differentiated, whichever forward that
/// is, and leaves the model as it found it.
#[test]
fn neftune_noises_every_forward_and_cleans_up() {
    let padded = row_batch(&[1, 2, 3, 4, 5, 6], &[2, 3, 4, 5, 6, 7]);
    let packed = two_sequence_packed_batch();
    let spec = |alpha: Option<f32>, cce: bool| LossSpec {
        neftune_alpha: alpha,
        cut_cross_entropy: cce,
        ..plain_spec()
    };
    for (name, view, cce) in [
        ("padded", BatchView::Padded(&padded), false),
        ("packed", BatchView::Packed(&packed), false),
        (
            "packed, cut cross-entropy",
            BatchView::Packed(&packed),
            true,
        ),
        (
            "padded, cut cross-entropy",
            BatchView::Padded(&padded),
            true,
        ),
    ] {
        let mut model = seeded_model(21);
        let plain = scalar(
            &loss_and_grads(&mut model, view, &spec(None, cce))
                .unwrap()
                .loss_sum,
        );
        let noised = scalar(
            &loss_and_grads(&mut model, view, &spec(Some(15.0), cce))
                .unwrap()
                .loss_sum,
        );
        assert!(
            (noised - plain).abs() > 1e-4,
            "{name}: NEFTune changed nothing ({plain})"
        );
        // The noise is off again for whatever shares the model.
        let after = scalar(
            &loss_and_grads(&mut model, view, &spec(None, cce))
                .unwrap()
                .loss_sum,
        );
        assert!(
            (after - plain).abs() < 1e-6,
            "{name}: the noise outlived its step ({plain} -> {after})"
        );
    }
}

// ---------------------------------------------------------------------------
// Optimizers end to end
// ---------------------------------------------------------------------------

type LoraWeights = std::collections::HashMap<std::rc::Rc<str>, Array>;

fn lora_diff(a: &LoraWeights, b: &LoraWeights) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .map(|(k, x)| max_abs(&x.subtract(&b[k])))
        .fold(0.0, f32::max)
}

/// Every optimizer a config can name trains the model through the loop: the
/// adapters move (a silent zero gradient or an update that never reaches the
/// weights would leave them), the loss at the end is below the loss at the
/// start, and what the loop hands back is finite.
#[test]
fn every_optimizer_trains_through_the_loop() {
    let rate = |kind| match kind {
        OptimizerType::Sgd => 0.05,
        OptimizerType::Lion => 3e-3,
        OptimizerType::Adafactor => 1e-2,
        OptimizerType::Muon => 2e-2,
        OptimizerType::AdamWScheduleFree => 2e-2,
        _ => 5e-3,
    };
    let dataset = create_dummy_dataset(2, 16);
    let mut model = small_model();
    let init = model.lora_parameters();
    for kind in OptimizerType::ALL {
        let mut config = config_of(
            TrainingConfig {
                learning_rate: rate(kind),
                optimizer: kind,
                weight_decay: 0.0,
                lr_scheduler: LrSchedulerType::Constant,
                max_steps: Some(24),
                ..simple_training(None)
            },
            1,
            16,
        );
        config.training.num_epochs = 20;
        let capture = MetricsCapture::default();
        let seen = capture.clone();
        model.set_lora_parameters(&init);
        let mut training_loop = TrainingLoop::new(config).unwrap();
        training_loop.add_callback(Box::new(capture));
        training_loop
            .run(&mut model, dataset.clone(), None, None)
            .unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();

        let trained = model.lora_parameters();
        assert!(
            lora_diff(&trained, &init) > 1e-4,
            "{kind}: the loop did not move the weights"
        );
        assert!(
            trained.values().all(|a| scalar(&a.sum(None)).is_finite()),
            "{kind}: non-finite weights"
        );
        let losses: Vec<f64> = seen.snapshot().iter().map(|m| m.loss).collect();
        assert_eq!(losses.len(), 24, "{kind}");
        // The running loss is an EMA that starts at the first step's loss, and
        // plain SGD's rate here is small: a dead gradient leaves the ratio at
        // one, which is what this separates from a slow fall.
        assert!(
            losses.last().unwrap() < &(losses[0] * 0.99),
            "{kind}: the loss did not fall: {} -> {}",
            losses[0],
            losses.last().unwrap()
        );
    }
}

// ---------------------------------------------------------------------------
// Resume
// ---------------------------------------------------------------------------

fn resume_config(kind: OptimizerType, steps: usize) -> TrainingLoopConfig {
    let mut config = config_of(
        TrainingConfig {
            learning_rate: 1e-2,
            optimizer: kind,
            weight_decay: 0.01,
            lr_scheduler: LrSchedulerType::Constant,
            max_steps: Some(steps),
            num_epochs: 2,
            seed: 5,
            ..simple_training(None)
        },
        1,
        12,
    );
    // Shuffled, so the position in the epoch's order matters to resume.
    config.dataloader.shuffle = true;
    config.dataloader.seed = 5;
    config.checkpoint_every = 3;
    config
}

/// Train `steps` straight, versus `3`, a checkpoint, a new process (a new
/// model, loop and optimizer) resuming from it, and the rest: the same
/// weights to the bit (or as far as the GPU's own run-to-run noise). The data position is mid-epoch (3 of 5 batches), and
/// the optimizer's moments, counters and averaging weights all come from the
/// checkpoint.
#[test]
fn resuming_from_a_checkpoint_continues_the_run_bit_for_bit() {
    for kind in [
        OptimizerType::AdamW,
        OptimizerType::AdamWScheduleFree,
        OptimizerType::Muon,
    ] {
        let dataset = || create_dummy_dataset(5, 12);
        let mut straight = seeded_model(33);
        TrainingLoop::new(resume_config(kind, 7))
            .unwrap()
            .run(&mut straight, dataset(), None, None)
            .unwrap();
        let want = straight.lora_parameters();
        // GPU reductions are not bit-reproducible, so the floor is how far
        // apart two identical straight runs land (zero, when they agree).
        let mut again = seeded_model(33);
        TrainingLoop::new(resume_config(kind, 7))
            .unwrap()
            .run(&mut again, dataset(), None, None)
            .unwrap();
        let noise = lora_diff(&again.lora_parameters(), &want);

        let dir = tempfile::TempDir::new().unwrap();
        let manager = CheckpointManager::new(dir.path()).unwrap();
        {
            // The interrupted run: it stops after its checkpoint at step 3.
            let mut interrupted = seeded_model(33);
            let mut training_loop = TrainingLoop::new(resume_config(kind, 3)).unwrap();
            // max_steps 3 would reshape a decaying schedule; the constant
            // one here is the same either way.
            training_loop
                .run(&mut interrupted, dataset(), None, Some(&manager))
                .unwrap();
        }
        let resumed_dir = manager
            .latest_dir()
            .unwrap()
            .expect("a checkpoint was written");
        assert!(resumed_dir.join("optimizer.safetensors").exists());
        assert!(resumed_dir.join("train_state.json").exists());

        let mut resumed = seeded_model(33);
        let mut training_loop = TrainingLoop::new(resume_config(kind, 7)).unwrap();
        assert!(training_loop.resume_from(&mut resumed, &manager).unwrap());
        assert_eq!(training_loop.current_step(), 3);
        training_loop
            .run(&mut resumed, dataset(), None, Some(&manager))
            .unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();

        let got = resumed.lora_parameters();
        let diff = lora_diff(&got, &want);
        // The bound is the GPU's own run-to-run spread, with a floor of a few
        // f32 ulps of a weight: a resume that dropped the optimizer's state
        // lands orders of magnitude above it.
        let bound = (noise * 4.0).max(1e-6);
        assert!(
            diff <= bound,
            "{kind}: the resumed run ended {diff} away from the straight one \
             (two straight runs differ by {noise})"
        );
        println!("{kind}: resumed vs straight {diff:e}, run-to-run {noise:e}");
    }
}

/// A resume without the optimizer's state is the old behaviour, now said
/// out loud: the run goes on, from zero moments, and ends elsewhere.
#[test]
fn a_resume_without_the_optimizer_state_ends_somewhere_else() {
    let kind = OptimizerType::AdamW;
    let dataset = || create_dummy_dataset(5, 12);
    let mut straight = seeded_model(33);
    TrainingLoop::new(resume_config(kind, 7))
        .unwrap()
        .run(&mut straight, dataset(), None, None)
        .unwrap();

    let dir = tempfile::TempDir::new().unwrap();
    let manager = CheckpointManager::new(dir.path()).unwrap();
    let mut interrupted = seeded_model(33);
    TrainingLoop::new(resume_config(kind, 3))
        .unwrap()
        .run(&mut interrupted, dataset(), None, Some(&manager))
        .unwrap();
    // Take away what makes the resume exact.
    let latest = manager.latest_dir().unwrap().unwrap();
    std::fs::remove_file(latest.join("train_state.json")).unwrap();

    let mut resumed = seeded_model(33);
    let mut training_loop = TrainingLoop::new(resume_config(kind, 7)).unwrap();
    assert!(training_loop.resume_from(&mut resumed, &manager).unwrap());
    training_loop
        .run(&mut resumed, dataset(), None, Some(&manager))
        .unwrap();
    let diff = lora_diff(&resumed.lora_parameters(), &straight.lora_parameters());
    println!("without optimizer state: {diff:e} off");
    assert!(
        diff > 1e-5,
        "dropping the optimizer state changed nothing ({diff:e})"
    );
}

/// Schedule-free AdamW trains on `y` and saves and returns `x`: the weights a
/// run hands back are the ones its last checkpoint holds.
#[test]
fn schedule_free_hands_back_the_averaged_weights_it_checkpoints() {
    let dir = tempfile::TempDir::new().unwrap();
    let manager = CheckpointManager::new(dir.path()).unwrap();
    let mut config = resume_config(OptimizerType::AdamWScheduleFree, 6);
    config.checkpoint_every = 6;
    let mut model = seeded_model(41);
    let mut training_loop = TrainingLoop::new(config).unwrap();
    training_loop
        .run(
            &mut model,
            create_dummy_dataset(5, 12),
            None,
            Some(&manager),
        )
        .unwrap();
    drop(training_loop);

    let (saved, _) = manager.load_latest().unwrap().expect("a checkpoint");
    let returned = model.lora_parameters();
    assert_eq!(
        lora_diff(&saved, &returned),
        0.0,
        "the run's weights are not its checkpoint's"
    );
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

fn unique_token_dataset(n: usize, len: usize) -> TrainingDataset {
    // Sample i is `i + 1` repeated, so a row's tokens name its samples.
    TrainingDataset::from_samples(
        (0..n)
            .map(|i| Sample::new(vec![(i + 1) as u32; len]))
            .collect(),
    )
}

fn ids_of(micro: &MicroBatch) -> Vec<i32> {
    let array = match micro {
        MicroBatch::Padded(b) => &b.input_ids,
        MicroBatch::Packed(p) => &p.input_ids,
    };
    array.eval();
    array
        .as_dtype(pmetal_runtime::compat::Dtype::Int32.as_i32())
        .as_slice::<i32>()
        .to_vec()
}

fn source_for(rank: usize, world: usize, packed: bool) -> BatchSource {
    let dataset = Arc::new(unique_token_dataset(12, 4));
    let config = DataLoaderConfig {
        batch_size: 1,
        max_seq_len: 8,
        shuffle: true,
        seed: 3,
        rank,
        world_size: world,
        ..Default::default()
    };
    if packed {
        BatchSource::packed(&dataset, &config, Some(8)).unwrap().0
    } else {
        BatchSource::padded(dataset, &config)
    }
}

/// Two simulated ranks see disjoint rows and the same number of batches, in
/// the padded and the packed source, so they take the same number of steps.
#[test]
fn ranks_see_disjoint_rows_and_equal_step_counts() {
    for packed in [false, true] {
        let mut seen: Vec<std::collections::BTreeSet<i32>> = Vec::new();
        let mut counts = Vec::new();
        for rank in 0..2 {
            let mut source = source_for(rank, 2, packed);
            let mut tokens = std::collections::BTreeSet::new();
            let mut count = 0;
            while let Some(micro) = source.next().unwrap() {
                tokens.extend(ids_of(&micro));
                count += 1;
            }
            assert_eq!(count, source.num_batches(), "packed: {packed}");
            seen.push(tokens);
            counts.push(count);
        }
        assert_eq!(
            counts[0], counts[1],
            "packed: {packed}: unequal step counts"
        );
        assert!(
            seen[0].is_disjoint(&seen[1]),
            "packed: {packed}: a sample went to two ranks: {seen:?}"
        );
        assert_eq!(
            seen[0].len() + seen[1].len(),
            12,
            "packed: {packed}: samples were dropped"
        );
    }
}

/// Resuming a source at a saved position continues the epoch's exact batch
/// sequence.
#[test]
fn a_source_resumes_mid_epoch_on_the_exact_batch_sequence() {
    for packed in [false, true] {
        let mut whole = source_for(0, 1, packed);
        let mut all = Vec::new();
        while let Some(micro) = whole.next().unwrap() {
            all.push(ids_of(&micro));
        }
        assert!(all.len() > 3);

        let mut first = source_for(0, 1, packed);
        for want in &all[..3] {
            assert_eq!(&ids_of(&first.next().unwrap().unwrap()), want);
        }
        let state = first.state();
        let json = serde_json::to_string(&state).unwrap();

        let mut second = source_for(0, 1, packed);
        second
            .restore(serde_json::from_str(&json).unwrap())
            .unwrap();
        for want in &all[3..] {
            assert_eq!(
                &ids_of(&second.next().unwrap().unwrap()),
                want,
                "packed: {packed}"
            );
        }
        assert!(second.next().unwrap().is_none());
    }
}

/// A corrupt sample is an error the run reports, not the end of the epoch.
#[test]
fn a_corrupt_sample_fails_the_run_instead_of_ending_the_epoch() {
    let mut bad = Sample::new(vec![1, 2, 3, 4]);
    bad.labels = Some(vec![1, 2]);
    let dataset = TrainingDataset::from_samples(vec![Sample::new(vec![1, 2, 3, 4]), bad]);
    let mut training_loop = TrainingLoop::new(config_of(simple_training(None), 1, 8)).unwrap();
    let mut model = small_model();
    let err = training_loop
        .run(&mut model, dataset, None, None)
        .expect_err("the corrupt sample must stop the run");
    assert!(err.to_string().contains("labels"), "{err}");
}

// ---------------------------------------------------------------------------
// Schedules
// ---------------------------------------------------------------------------

/// An epoch-based run: 2 epochs of 10 one-sample batches, 2 per step.
fn epoch_based_config() -> TrainingLoopConfig {
    let mut config = TrainingLoopConfig {
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        ..Default::default()
    };
    config.training.learning_rate = 1e-3;
    config.training.warmup_ratio = Some(0.1);
    config.training.lr_scheduler = LrSchedulerType::Cosine;
    config.training.gradient_accumulation_steps = 2;
    config.training.num_epochs = 2;
    config.training.max_steps = None;
    config.dataloader.batch_size = 1;
    config.dataloader.shuffle = false;
    config
}

#[test]
fn an_epoch_based_schedule_spans_the_run_in_optimizer_steps() {
    let mut training_loop = TrainingLoop::new(epoch_based_config()).unwrap();
    let micro = training_loop.micro_batches_per_epoch(&create_dummy_dataset(10, 8));
    assert_eq!(micro, 10);
    // 20 micro-batches over the run, 2 per step: 10 optimizer steps, of
    // which ceil(0.1 * 10) = 1 warms up.
    assert_eq!(training_loop.plan_schedule(micro, 2), 20);
    assert_eq!(training_loop.schedule_total_steps, Some(10));
    let lr_at = |tl: &mut TrainingLoop, micro_step: usize| {
        tl.step = micro_step;
        tl.get_learning_rate()
    };
    // Micro-batches 0 and 1 make up step 0: warmup's first rate, zero.
    assert_eq!(lr_at(&mut training_loop, 0), 0.0);
    assert_eq!(lr_at(&mut training_loop, 1), 0.0);
    assert!((lr_at(&mut training_loop, 2) - 1e-3).abs() < 1e-9);
    // The last step is the cosine's last: 1e-3 * (1 + cos(pi * 8/9)) / 2.
    let want = 1e-3 * 0.5 * (1.0 + (std::f64::consts::PI * 8.0 / 9.0).cos());
    assert!((lr_at(&mut training_loop, 19) as f64 - want).abs() < 1e-9);
    assert!(want < 3.1e-5);
}

#[test]
fn an_epoch_based_run_decays_to_the_end_and_max_steps_counts_optimizer_steps() {
    let capture = MetricsCapture::default();
    let mut training_loop = TrainingLoop::new(epoch_based_config()).unwrap();
    training_loop.add_callback(Box::new(capture.clone()));
    let mut model = small_model();
    training_loop
        .run(&mut model, create_dummy_dataset(10, 8), None, None)
        .unwrap();
    let steps = capture.snapshot();
    assert_eq!(training_loop.current_step(), 20);
    assert_eq!(steps.last().unwrap().total_steps, 20);
    // The rate the last logged step reports is the end of the cosine.
    let last_lr = steps.last().unwrap().lr;
    assert!(
        last_lr < 2e-5,
        "the schedule never decayed: last lr {last_lr}"
    );

    let mut config = epoch_based_config();
    config.training.max_steps = Some(3);
    let mut training_loop = TrainingLoop::new(config).unwrap();
    training_loop
        .run(&mut model, create_dummy_dataset(10, 8), None, None)
        .unwrap();
    // 3 optimizer steps of 2 micro-batches each.
    assert_eq!(training_loop.optimizer_steps_taken(), 3);
    assert_eq!(training_loop.current_step(), 6);
}

/// Micro-batches left over when the data runs out are one more, smaller,
/// optimizer step, not lost: 5 batches at 2 per step is 3 steps.
#[test]
fn leftover_micro_batches_take_a_last_optimizer_step() {
    let mut config = epoch_based_config();
    config.training.num_epochs = 1;
    config.training.warmup_ratio = None;
    let mut training_loop = TrainingLoop::new(config).unwrap();
    let mut model = small_model();
    let before = model.lora_parameters();
    training_loop
        .run(&mut model, create_dummy_dataset(5, 8), None, None)
        .unwrap();
    assert_eq!(training_loop.current_step(), 5);
    assert_eq!(
        training_loop.accumulator.micro_batches(),
        0,
        "nothing was left pending"
    );
    assert!(lora_diff(&model.lora_parameters(), &before) > 0.0);
}

/// A callback that asks the run to stop once `should_stop` has been asked
/// `after` times.
#[derive(Clone)]
struct StopWhenAsked {
    asked: Arc<std::sync::atomic::AtomicUsize>,
    after: usize,
}

impl TrainingCallback for StopWhenAsked {
    fn should_stop(&self) -> bool {
        self.asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= self.after
    }
}

/// A callback's request to stop is honoured after the step it follows, whether
/// or not that step is one the run logs on (`log_every` defaults to 10), and
/// the run ends cleanly (`Ok`), so the caller still saves what was trained.
#[test]
fn a_callback_that_asks_to_stop_ends_the_run_after_that_step() {
    let mut config = epoch_based_config();
    config.log_every = 100;
    config.training.gradient_accumulation_steps = 1;
    config.training.num_epochs = 1;
    config.training.warmup_ratio = None;
    let stop = StopWhenAsked {
        asked: Arc::default(),
        after: 3,
    };
    let mut training_loop = TrainingLoop::new(config).unwrap();
    training_loop.add_callback(Box::new(stop.clone()));
    let mut model = small_model();
    let before = model.lora_parameters();
    training_loop
        .run(&mut model, create_dummy_dataset(20, 8), None, None)
        .expect("a requested stop is a clean end, not an error");
    // Asked after steps 1, 2, 3 and 4: it says yes the fourth time.
    assert_eq!(training_loop.current_step(), 4);
    assert!(
        lora_diff(&model.lora_parameters(), &before) > 0.0,
        "the steps taken before the stop trained the model"
    );
}

// ---------------------------------------------------------------------------
// Failures
// ---------------------------------------------------------------------------

/// How [`Breaking`] goes wrong on its chosen step.
#[derive(Clone, Copy)]
enum Fault {
    /// The loss comes out NaN, as an overflow would leave it.
    NanLoss,
    /// A runtime op throws inside the gradient trace. The runtime records it and
    /// hands back an empty array, and nothing unwinds.
    RuntimeError,
}

/// The small model, broken on the `on`-th forward pass.
struct Breaking {
    inner: AdaptedModel,
    forwards: usize,
    on: usize,
    fault: Fault,
}

impl Breaking {
    fn new(on: usize, fault: Fault) -> Self {
        Self {
            inner: small_model(),
            forwards: 0,
            on,
            fault,
        }
    }
}

impl ModuleParameters for Breaking {
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }
    fn parameters(&self) -> pmetal_runtime::compat::ModuleParamRef<'_> {
        self.inner.parameters()
    }
    fn parameters_mut(&mut self) -> pmetal_runtime::compat::ModuleParamMut<'_> {
        self.inner.parameters_mut()
    }
    fn trainable_parameters(&self) -> pmetal_runtime::compat::ModuleParamRef<'_> {
        self.inner.trainable_parameters()
    }
}

impl TrainableModel for Breaking {
    fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> std::result::Result<Array, pmetal_lora::LoraError> {
        self.forwards += 1;
        let logits = TrainableModel::forward(&mut self.inner, input_ids, mask)?;
        if self.forwards != self.on {
            return Ok(logits);
        }
        Ok(match self.fault {
            Fault::NanLoss => logits.multiply(&Array::from_f32(f32::NAN)),
            Fault::RuntimeError => {
                // Inner dimensions that don't match: MLX throws.
                let bad = pmetal_runtime::compat::ops::matmul(&logits, &Array::ones_f32(&[3, 5]));
                logits.add(&bad.sum(None))
            }
        })
    }
    fn num_trainable_params(&self) -> usize {
        self.inner.num_trainable_params()
    }
    fn lora_parameters(&self) -> std::collections::HashMap<std::rc::Rc<str>, Array> {
        self.inner.lora_parameters()
    }
    fn set_lora_parameters(&mut self, params: &std::collections::HashMap<std::rc::Rc<str>, Array>) {
        self.inner.set_lora_parameters(params)
    }
    fn save_lora_weights(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::result::Result<(), pmetal_lora::LoraError> {
        self.inner.save_lora_weights(path)
    }
    fn load_lora_weights(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> std::result::Result<(), pmetal_lora::LoraError> {
        self.inner.load_lora_weights(path)
    }
}

fn short_run_config() -> TrainingLoopConfig {
    config_of(simple_training(Some(6)), 1, 16)
}

/// A step that fails inside the gradient trace stops the run there, naming
/// the step, whether it shows as a NaN loss or as an op that threw. The run
/// used to log `loss=NaN`, carry on to the end, and save an adapter no step
/// had moved.
#[test]
fn a_failed_step_stops_the_run_and_names_the_step() {
    for (fault, expect) in [
        (Fault::NanLoss, "loss of NaN"),
        (Fault::RuntimeError, "matmul"),
    ] {
        for packing in [false, true] {
            let mut model = Breaking::new(3, fault);
            let mut training_loop = TrainingLoop::new(TrainingLoopConfig {
                use_sequence_packing: packing,
                ..short_run_config()
            })
            .unwrap();
            let err = pmetal_runtime::expect_runtime_errors(|| {
                training_loop.run(&mut model, create_dummy_dataset(8, 12), None, None)
            })
            .expect_err("the run should stop at the broken step");
            let message = err.to_string();
            assert!(message.contains("step 3") || packing, "{message}");
            assert!(message.contains(expect) || packing, "{message}");
            pmetal_runtime::check_unobserved_error().expect("the failure was drained");
        }
    }
}

// ---------------------------------------------------------------------------
// Cut cross-entropy
// ---------------------------------------------------------------------------

/// A step with cut cross-entropy moves an adapter on the LM head exactly as a
/// step with the full logits does. The step used to read the head's weight
/// before differentiating, so the adapter on it was a constant to autograd:
/// its update was zero, and the loss left its contribution out.
#[test]
fn cut_cross_entropy_trains_an_adapter_on_the_lm_head() {
    let build = || {
        let config = LlamaConfig {
            tie_word_embeddings: false,
            ..small_config()
        };
        let lora = LoraConfig {
            target_modules: vec!["q_proj".to_string(), "lm_head".to_string()],
            ..small_lora_config()
        };
        random::seed(5);
        let base = DynamicModel::from_config(&serde_json::to_string(&config).unwrap()).unwrap();
        AdaptedModel::attach(base, lora).unwrap()
    };
    let batch = row_batch(&[1, 2, 3, 4, 5, 6], &[1, 2, 3, 4, 5, 6]);
    let spec = |cce| LossSpec {
        cut_cross_entropy: cce,
        ..plain_spec()
    };
    let full = loss_and_grads(&mut build(), BatchView::Padded(&batch), &spec(false)).unwrap();
    let cut = loss_and_grads(&mut build(), BatchView::Padded(&batch), &spec(true)).unwrap();
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");

    let key_of = |name: &str| {
        full.grads
            .keys()
            .find(|k| k.ends_with(name) || k.as_ref() == name)
            .unwrap_or_else(|| panic!("no {name} among {:?}", full.grads.keys()))
            .clone()
    };
    for name in ["lm_head.lora_b", "layers.0.self_attn.q_proj.lora_b"] {
        let key = key_of(name);
        let size = max_abs(&full.grads[&key]);
        let diff = max_abs(&cut.grads[&key].subtract(&full.grads[&key]));
        assert!(size > 0.0, "{key}: the full-logits gradient is zero");
        assert!(
            diff <= 1e-4 * size,
            "{key}: the CCE gradient differs (off by {diff}, gradient {size})"
        );
    }
}

/// A packed step with cut cross-entropy is the packed step with the full
/// logits. It used to *be* that step: the adapted model had no hidden-state
/// forward with positions, so every packed batch quietly fell back to the
/// full logits.
#[test]
fn a_packed_cut_cross_entropy_step_matches_the_full_logits_step() {
    let batch = two_sequence_packed_batch();
    let spec = |cce| LossSpec {
        cut_cross_entropy: cce,
        ..plain_spec()
    };
    let mut full_model = seeded_model(9);
    let mut cut_model = seeded_model(9);
    assert!(
        full_model
            .forward_hidden_with_positions(
                &batch.input_ids.reshape(&[1, 8]),
                None,
                &batch.position_ids
            )
            .is_some_and(|h| h.is_ok())
    );
    let want = loss_and_grads(&mut full_model, BatchView::Packed(&batch), &spec(false)).unwrap();
    let got = loss_and_grads(&mut cut_model, BatchView::Packed(&batch), &spec(true)).unwrap();
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
    let (w, g) = (scalar(&want.loss_sum), scalar(&got.loss_sum));
    assert!(
        (w - g).abs() <= 1e-5 * w.abs(),
        "loss {g} vs the full logits' {w}"
    );
    for (key, value) in &want.grads {
        let diff = max_abs(&got.grads[key].subtract(value));
        assert!(diff <= 1e-5, "{key}: the CCE step differs by {diff}");
    }
}

/// The paths ask one place whether cut cross-entropy applies, and it says
/// no, once, when it can't, rather than each path quietly taking the full
/// logits. NEFTune does not stand in its way: the noise is on the embedding
/// whichever head computes the loss.
#[test]
fn cut_cross_entropy_says_when_it_does_not_apply() {
    let model = small_model();
    let config = |neftune: Option<f64>| {
        let mut config = TrainingLoopConfig {
            use_cut_cross_entropy: true,
            ..short_run_config()
        };
        config.training.neftune_noise_alpha = neftune;
        config
    };

    let plain = TrainingLoop::new(config(None)).unwrap();
    assert!(plain.cut_cross_entropy_applies(&model, false));
    assert!(!plain.cce_unavailable_reported.get());

    let noised = TrainingLoop::new(config(Some(5.0))).unwrap();
    assert!(noised.cut_cross_entropy_applies(&model, false));
    assert!(!noised.cce_unavailable_reported.get());

    // A vision input goes through the vision forward, which returns logits.
    assert!(!plain.cut_cross_entropy_applies(&model, true));
    assert!(plain.cce_unavailable_reported.get());

    // Not asked for: nothing to say.
    let off = TrainingLoop::new(TrainingLoopConfig {
        use_cut_cross_entropy: false,
        ..config(None)
    })
    .unwrap();
    assert!(!off.cut_cross_entropy_applies(&model, true));
    assert!(!off.cce_unavailable_reported.get());
}

/// A model that has an LM head but no hidden-state forward fails the CCE
/// step instead of training on the full logits behind its back.
#[test]
fn a_cut_cross_entropy_step_without_hidden_states_fails() {
    // `Breaking` delegates everything but the hidden-state forwards, which
    // it leaves at the trait's "not implemented".
    let mut model = Breaking::new(usize::MAX, Fault::NanLoss);
    let spec = LossSpec {
        cut_cross_entropy: true,
        ..plain_spec()
    };
    let batch = row_batch(&[1, 2, 3, 4], &[1, 2, 3, 4]);
    let err = loss_and_grads(&mut model, BatchView::Padded(&batch), &spec)
        .err()
        .expect("no fallback");
    assert!(err.to_string().contains("hidden-state forward"), "{err}");
    let packed = two_sequence_packed_batch();
    let err = loss_and_grads(&mut model, BatchView::Packed(&packed), &spec)
        .err()
        .expect("no fallback");
    assert!(err.to_string().contains("hidden-state forward"), "{err}");
    pmetal_runtime::clear_last_error();
}

// ---------------------------------------------------------------------------
// Parameter groups
// ---------------------------------------------------------------------------

/// The loop trains the groups its config describes: LoRA+ on the B
/// matrices, the embedding rate on the LM head's adapter (the head the model
/// reports, whatever it is called), weight decay on the rest, with the
/// frozen base left alone.
#[test]
fn the_loop_trains_the_parameter_groups_the_config_describes() {
    use std::collections::HashMap;
    let lora = LoraConfig {
        target_modules: ["q_proj", "v_proj", "lm_head"].map(String::from).to_vec(),
        ..small_lora_config()
    };
    random::seed(29);
    let untied = LlamaConfig {
        tie_word_embeddings: false,
        ..small_config()
    };
    let base =
        DynamicModel::from_config(&serde_json::to_string(&untied).unwrap()).expect("llama builds");
    let mut model = AdaptedModel::attach(base, lora).expect("attach adapters");

    let mut config = epoch_based_config();
    config.training.learning_rate = 2e-3;
    config.training.weight_decay = 0.1;
    config.training.gradient_accumulation_steps = 1;
    config.training.num_epochs = 1;
    config.training.max_steps = Some(4);
    config.embedding_lr = Some(5e-4);
    config.loraplus_lr_ratio = Some(16.0);

    // The head, as the model reports it, is the embedding group: its
    // adapter and nothing else.
    let head = model.output_projection_path();
    assert_eq!(head.as_deref(), Some("lm_head"));
    let groups = TrainingLoop::new(config.clone())
        .unwrap()
        .parameter_groups(head);
    for name in model.flatten_trainable_params().keys() {
        assert_eq!(
            groups.classify(name) == crate::ParamClass::Embedding,
            name.starts_with("lm_head."),
            "{name}"
        );
    }

    let snapshot = |model: &AdaptedModel| -> HashMap<String, Vec<f32>> {
        model
            .flatten_params()
            .iter()
            .map(|(k, v)| {
                let v = v.as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32());
                v.eval();
                (k.to_string(), v.as_slice::<f32>().to_vec())
            })
            .collect()
    };
    let start = snapshot(&model);
    let trainable: Vec<String> = model
        .flatten_trainable_params()
        .keys()
        .map(|k| k.to_string())
        .collect();
    TrainingLoop::new(config)
        .unwrap()
        .run(&mut model, create_dummy_dataset(4, 12), None, None)
        .expect("the loop trains");
    let end = snapshot(&model);
    for (name, was) in &start {
        let moved = end[name].iter().zip(was).any(|(a, b)| a != b);
        assert_eq!(
            moved,
            trainable.contains(name),
            "{name}: moved {moved}, trainable {}",
            trainable.contains(name)
        );
    }
    assert!(
        trainable.iter().any(|k| k.contains("lm_head.lora_")),
        "the LM head is adapted"
    );
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
}
