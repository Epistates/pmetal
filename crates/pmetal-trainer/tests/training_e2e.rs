//! End-to-end training integration tests.
//!
//! These tests verify the complete training pipeline works correctly:
//! - DataLoader → Model → Optimizer flow
//! - Gradient accumulation
//! - Optional Metal FlashAttention for forward pass
//! - Checkpoint saving/loading

use pmetal_bridge::compat::optimizers::Sgd;
use pmetal_core::{LoraConfig, TrainingConfig};
use pmetal_data::{DataLoaderConfig, Sample, TrainingDataset};
use pmetal_lora::AdaptedModel;
use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_trainer::{TrainingLoop, TrainingLoopConfig};

fn small_llama_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 256,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: None,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        ..Default::default()
    }
}

fn small_lora_config() -> LoraConfig {
    LoraConfig {
        r: 4,
        alpha: 8.0,
        dropout: 0.0,
        use_rslora: false,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ],
        bias: pmetal_core::LoraBias::None,
        init_lora_weights: true,
        loraplus_lr_ratio: None,
        use_dora: false,
    }
}

/// The trainable model these tests run against: the dispatcher's own Llama
/// with adapters attached, which is what `pmetal train` builds.
fn small_model() -> AdaptedModel {
    let base = DynamicModel::from_config(&serde_json::to_string(&small_llama_config()).unwrap())
        .expect("llama builds");
    AdaptedModel::attach(base, small_lora_config()).expect("attach adapters")
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

#[test]
fn test_training_loop_single_step() {
    // Create model
    let mut model = small_model();

    // Create training loop config
    let config = TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 2,
            gradient_accumulation_steps: 1,
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 16,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        use_metal_flash_attention: false, // Disable for simpler test
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        use_jit_compilation: false,
        ..Default::default()
    };

    let mut training_loop = TrainingLoop::new(config);

    // Create a minimal dataset
    let dataset = create_dummy_dataset(4, 16);
    let mut dataloader = pmetal_data::DataLoader::new(
        dataset,
        DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 16,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        None,
    );

    // Create optimizer
    let mut optimizer = Sgd::new(1e-4);

    // Get a batch
    let batch = dataloader.next_batch().expect("Should have a batch");

    // Perform training step
    let stats = training_loop
        .train_step(&mut model, &batch, &mut optimizer)
        .expect("Training step should succeed");

    // Verify stats
    assert!(stats.loss > 0.0, "Loss should be positive");
    assert_eq!(stats.step, 1, "Step should be 1");
    assert!(stats.tokens > 0, "Tokens processed should be positive");
    assert_eq!(training_loop.current_step(), 1);
}

#[test]
fn test_training_loop_gradient_accumulation() {
    let mut model = small_model();

    let config = TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 1,
            gradient_accumulation_steps: 2, // Accumulate over 2 steps
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size: 1,
            max_seq_len: 8,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        use_metal_flash_attention: false,
        log_every: 100,
        checkpoint_every: 0,
        eval_every: 0,
        use_jit_compilation: false,
        ..Default::default()
    };

    let mut training_loop = TrainingLoop::new(config);
    let dataset = create_dummy_dataset(4, 8);
    let mut dataloader = pmetal_data::DataLoader::new(
        dataset,
        DataLoaderConfig {
            batch_size: 1,
            max_seq_len: 8,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        None,
    );

    let mut optimizer = Sgd::new(1e-4);

    // First step - should accumulate, not apply gradients
    let batch1 = dataloader.next_batch().unwrap();
    let stats1 = training_loop
        .train_step(&mut model, &batch1, &mut optimizer)
        .unwrap();
    assert!(
        stats1.grad_norm.is_none(),
        "First step should not apply gradients"
    );

    // Second step - should apply accumulated gradients
    let batch2 = dataloader.next_batch().unwrap();
    let stats2 = training_loop
        .train_step(&mut model, &batch2, &mut optimizer)
        .unwrap();
    // The accumulated gradients are applied here, and the default
    // max_grad_norm > 0 makes the step compute their norm.
    assert!(
        stats2.grad_norm.is_some(),
        "Second step should apply the accumulated gradients"
    );

    assert_eq!(training_loop.current_step(), 2);
}

#[test]
fn test_training_loop_multiple_steps() {
    let mut model = small_model();

    let config = TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-3,
            batch_size: 2,
            gradient_accumulation_steps: 1,
            max_grad_norm: 1.0, // Enable gradient clipping
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 16,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        use_metal_flash_attention: false,
        log_every: 100,
        checkpoint_every: 0,
        eval_every: 0,
        use_jit_compilation: false,
        ..Default::default()
    };

    let mut training_loop = TrainingLoop::new(config);
    let dataset = create_dummy_dataset(8, 16);
    let mut dataloader = pmetal_data::DataLoader::new(
        dataset,
        DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 16,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        None,
    );

    let mut optimizer = Sgd::new(1e-3);

    let mut losses = Vec::new();

    // Train for 4 steps
    for _ in 0..4 {
        if let Some(batch) = dataloader.next_batch() {
            let stats = training_loop
                .train_step(&mut model, &batch, &mut optimizer)
                .unwrap();
            losses.push(stats.loss);

            // Gradient norm should be computed (since max_grad_norm > 0)
            assert!(
                stats.grad_norm.is_some(),
                "Gradient norm should be computed"
            );
        }
    }

    assert_eq!(training_loop.current_step(), 4);
    assert_eq!(losses.len(), 4);

    // All losses should be positive
    for loss in &losses {
        assert!(*loss > 0.0);
    }
}

#[test]
fn test_training_with_metal_flash_attention() {
    // This test verifies that Metal FlashAttention integration doesn't break training
    let mut model = small_model();

    let config = TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 2,
            gradient_accumulation_steps: 1,
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 16,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        use_metal_flash_attention: true, // Enable Metal FlashAttention
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        use_jit_compilation: false,
        ..Default::default()
    };

    let mut training_loop = TrainingLoop::new(config);
    let dataset = create_dummy_dataset(4, 16);
    let mut dataloader = pmetal_data::DataLoader::new(
        dataset,
        DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 16,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        None,
    );

    let mut optimizer = Sgd::new(1e-4);

    // Should work regardless of whether Metal FA is available
    let batch = dataloader.next_batch().expect("Should have a batch");
    let stats = training_loop
        .train_step(&mut model, &batch, &mut optimizer)
        .expect("Training step should succeed even with Metal FA");

    assert!(stats.loss > 0.0);
    assert_eq!(stats.step, 1);
}

#[test]
fn test_learning_rate_schedules() {
    // Test different LR schedules
    for scheduler in [
        pmetal_core::LrSchedulerType::Constant,
        pmetal_core::LrSchedulerType::Linear,
        pmetal_core::LrSchedulerType::Cosine,
    ] {
        let config = TrainingLoopConfig {
            training: TrainingConfig {
                learning_rate: 1e-4,
                warmup_steps: 10,
                max_steps: Some(100),
                lr_scheduler: scheduler,
                ..Default::default()
            },
            use_metal_flash_attention: false,
            use_jit_compilation: false,
            ..Default::default()
        };

        let mut training_loop = TrainingLoop::new(config);

        // Test warmup phase
        training_loop.set_step(0);
        let lr_start = training_loop.get_learning_rate();

        training_loop.set_step(5);
        let lr_mid_warmup = training_loop.get_learning_rate();

        training_loop.set_step(10);
        let lr_end_warmup = training_loop.get_learning_rate();

        // Warmup should increase LR
        assert!(
            lr_mid_warmup > lr_start,
            "LR should increase during warmup for {:?}",
            scheduler
        );
        assert!(
            lr_end_warmup >= lr_mid_warmup,
            "LR should continue increasing during warmup for {:?}",
            scheduler
        );

        // After warmup, behavior depends on scheduler
        training_loop.set_step(50);
        let lr_mid = training_loop.get_learning_rate();

        match scheduler {
            pmetal_core::LrSchedulerType::Constant => {
                assert!(
                    (lr_mid - 1e-4).abs() < 1e-8,
                    "Constant LR should stay at base"
                );
            }
            pmetal_core::LrSchedulerType::Linear | pmetal_core::LrSchedulerType::Cosine => {
                assert!(lr_mid < 1e-4, "LR should decay for {:?}", scheduler);
            }
            _ => {}
        }
    }
}

/// [`small_model`] with its base packed for QLoRA before the adapters go on,
/// which is what `pmetal train --quantization` builds.
fn small_qlora_model(scheme: pmetal_lora::QLoraScheme) -> (AdaptedModel, pmetal_lora::PackedBase) {
    let mut base =
        DynamicModel::from_config(&serde_json::to_string(&small_llama_config()).unwrap())
            .expect("llama builds");
    let config = pmetal_lora::QLoraConfig {
        scheme,
        ..Default::default()
    };
    let packed = pmetal_lora::quantize_base(&mut base, &config).expect("pack base");
    let model = AdaptedModel::attach(base, small_lora_config()).expect("attach adapters");
    (model, packed)
}

fn qlora_loop(learning_rate: f64) -> TrainingLoop {
    TrainingLoop::new(TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate,
            batch_size: 2,
            gradient_accumulation_steps: 1,
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
        use_metal_flash_attention: false,
        log_every: 100,
        checkpoint_every: 0,
        eval_every: 0,
        use_jit_compilation: false,
        ..Default::default()
    })
}

#[test]
fn test_qlora_training_step() {
    let (mut model, packed) = small_qlora_model(pmetal_lora::QLoraScheme::Nf4);

    // Every projection but the (absent, tied) LM head packs, to under a
    // third of its f32 size.
    assert_eq!(packed.packed, 2 * 7, "two layers of seven projections");
    assert!(
        packed.packed_bytes * 3 < packed.dense_bytes,
        "{} packed bytes for {} dense",
        packed.packed_bytes,
        packed.dense_bytes
    );

    let mut training_loop = qlora_loop(1e-4);
    let mut dataloader = pmetal_data::DataLoader::new(
        create_dummy_dataset(4, 16),
        DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 16,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        None,
    );
    let mut optimizer = Sgd::new(1e-4);
    let batch = dataloader.next_batch().expect("Should have a batch");

    let stats = training_loop
        .train_step(&mut model, &batch, &mut optimizer)
        .expect("QLoRA training step should succeed");
    pmetal_bridge::check_last_error().expect("no bridge error");

    assert!(stats.loss > 0.0, "QLoRA loss should be positive");
    assert_eq!(stats.step, 1, "Step should be 1");
    assert!(stats.tokens > 0, "Tokens processed should be positive");
}

/// Each scheme fits one batch: the adapters train through the packed base.
#[test]
fn test_qlora_loss_decreases_for_every_scheme() {
    for scheme in [
        pmetal_lora::QLoraScheme::Nf4,
        pmetal_lora::QLoraScheme::Fp4,
        pmetal_lora::QLoraScheme::Int8,
    ] {
        let (mut model, _) = small_qlora_model(scheme);
        let mut training_loop = qlora_loop(1e-2);
        let mut dataloader = pmetal_data::DataLoader::new(
            create_dummy_dataset(2, 16),
            DataLoaderConfig {
                batch_size: 2,
                max_seq_len: 16,
                shuffle: false,
                pad_token_id: 0,
                ..Default::default()
            },
            None,
        );
        let batch = dataloader.next_batch().expect("a batch");
        let mut optimizer = pmetal_bridge::compat::optimizers::AdamW::new(1e-2, 0.0);

        let mut losses = Vec::new();
        for _ in 0..10 {
            let stats = training_loop
                .train_step(&mut model, &batch, &mut optimizer)
                .expect("step");
            losses.push(stats.loss);
        }
        pmetal_bridge::check_last_error().expect("no bridge error");
        assert!(losses.iter().all(|l| l.is_finite()), "{scheme}: {losses:?}");
        assert!(
            losses[9] < losses[0] - 0.1,
            "{scheme}: loss did not fall: {losses:?}"
        );
    }
}

#[test]
fn test_evaluation_metrics() {
    // Test that evaluation returns comprehensive metrics (loss, perplexity, accuracy)
    let mut model = small_model();

    let config = TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 2,
            gradient_accumulation_steps: 1,
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 16,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        use_metal_flash_attention: false,
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        use_jit_compilation: false,
        ..Default::default()
    };

    let training_loop = TrainingLoop::new(config);
    let eval_dataset = create_dummy_dataset(4, 16);

    // Run evaluation
    let metrics = training_loop
        .evaluate(&mut model, &eval_dataset)
        .expect("Evaluation should succeed");

    // Verify loss is positive
    assert!(
        metrics.loss > 0.0,
        "Loss should be positive, got {}",
        metrics.loss
    );

    // Verify perplexity = exp(loss)
    let expected_ppl = metrics.loss.exp();
    assert!(
        (metrics.perplexity - expected_ppl).abs() < 0.01,
        "Perplexity should be exp(loss), got {} vs expected {}",
        metrics.perplexity,
        expected_ppl
    );

    // Verify accuracy is present and reasonable (0-100%)
    assert!(metrics.accuracy.is_some(), "Accuracy should be computed");
    let acc = metrics.accuracy.unwrap();
    assert!(
        (0.0..=100.0).contains(&acc),
        "Accuracy should be 0-100%, got {}",
        acc
    );

    // With random weights, accuracy should be roughly 1/vocab_size * 100 = 0.39%
    // But could be higher due to model structure, so just check it's not 100%
    assert!(
        acc < 50.0,
        "Accuracy with random weights should be low, got {}",
        acc
    );
}
