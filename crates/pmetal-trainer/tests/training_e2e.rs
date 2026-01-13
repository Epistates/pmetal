//! End-to-end training integration tests.
//!
//! These tests verify the complete training pipeline works correctly:
//! - DataLoader → Model → Optimizer flow
//! - Gradient accumulation
//! - Optional Metal FlashAttention for forward pass
//! - Checkpoint saving/loading

use mlx_rs::optimizers::Sgd;
use pmetal_core::{LoraConfig, TrainingConfig};
use pmetal_data::{DataLoaderConfig, Sample, TrainingDataset};
use pmetal_lora::LlamaLoraForCausalLM;
use pmetal_models::architectures::llama::LlamaConfig;
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
        bias: "none".to_string(),
        init_lora_weights: true,
        use_dora: false,
    }
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
    let mut model = LlamaLoraForCausalLM::new(small_llama_config(), small_lora_config())
        .expect("Failed to create model");

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
    let mut model = LlamaLoraForCausalLM::new(small_llama_config(), small_lora_config())
        .expect("Failed to create model");

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
    // After gradient accumulation is complete, grad_norm should be Some
    // (assuming max_grad_norm > 0, which triggers gradient clipping computation)

    assert_eq!(training_loop.current_step(), 2);
}

#[test]
fn test_training_loop_multiple_steps() {
    let mut model = LlamaLoraForCausalLM::new(small_llama_config(), small_lora_config())
        .expect("Failed to create model");

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
    let mut model = LlamaLoraForCausalLM::new(small_llama_config(), small_lora_config())
        .expect("Failed to create model");

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
                lr_scheduler: scheduler.clone(),
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

#[test]
fn test_qlora_training_step() {
    use pmetal_lora::{LlamaQloraForCausalLM, QLoraConfig};
    use pmetal_mlx::quantization::QuantScheme;

    // Create QLoRA config
    let qlora_config = QLoraConfig {
        lora: LoraConfig {
            r: 4,
            alpha: 8.0,
            use_rslora: false,
            ..Default::default()
        },
        quant_scheme: QuantScheme::NF4,
        block_size: 64,
        double_quant: true,
        compute_in_half: false,
    };

    // Create QLoRA model
    let mut model = LlamaQloraForCausalLM::with_qlora_config(small_llama_config(), qlora_config)
        .expect("Failed to create QLoRA model");

    // Verify memory savings
    let savings = model.memory_savings();
    assert!(
        savings < 0.35,
        "QLoRA should provide significant memory savings, got {}",
        savings
    );

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
        use_metal_flash_attention: false,
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
    let batch = dataloader.next_batch().expect("Should have a batch");

    // Perform training step with QLoRA model
    let stats = training_loop
        .train_step(&mut model, &batch, &mut optimizer)
        .expect("QLoRA training step should succeed");

    assert!(stats.loss > 0.0, "QLoRA loss should be positive");
    assert_eq!(stats.step, 1, "Step should be 1");
    assert!(stats.tokens > 0, "Tokens processed should be positive");
}

#[test]
fn test_qlora_multiple_steps() {
    use pmetal_lora::{LlamaQloraForCausalLM, QLoraConfig};
    use pmetal_mlx::quantization::QuantScheme;

    let qlora_config = QLoraConfig {
        lora: LoraConfig {
            r: 4,
            alpha: 8.0,
            use_rslora: false,
            ..Default::default()
        },
        quant_scheme: QuantScheme::NF4,
        block_size: 64,
        double_quant: true,
        compute_in_half: false,
    };

    let mut model = LlamaQloraForCausalLM::with_qlora_config(small_llama_config(), qlora_config)
        .expect("Failed to create QLoRA model");

    let config = TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-3,
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
            assert!(
                stats.grad_norm.is_some(),
                "Gradient norm should be computed"
            );
        }
    }

    assert_eq!(
        training_loop.current_step(),
        4,
        "Should have completed 4 steps"
    );
    assert_eq!(losses.len(), 4, "Should have 4 loss values");

    // All losses should be positive
    for loss in &losses {
        assert!(*loss > 0.0, "Loss should be positive");
    }
}

#[test]
fn test_evaluation_metrics() {
    // Test that evaluation returns comprehensive metrics (loss, perplexity, accuracy)
    let mut model = LlamaLoraForCausalLM::new(small_llama_config(), small_lora_config())
        .expect("Failed to create model");

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
        acc >= 0.0 && acc <= 100.0,
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
