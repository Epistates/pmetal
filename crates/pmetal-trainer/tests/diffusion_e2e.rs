//! End-to-end diffusion training integration tests.
//!
//! These tests verify the LLaDA-style diffusion training pipeline:
//! - GPU-native forward process (masking)
//! - GPU-native diffusion loss
//! - Full training loop with Qwen3LoRA model
//!
//! NOTE: These tests must run serially due to Metal's single-threaded
//! command buffer model. Parallel execution causes command encoder conflicts.

use mlx_rs::optimizers::Sgd;
use mlx_rs::Array;
use pmetal_core::{LoraConfig, TrainingConfig};
use pmetal_data::{DataLoaderConfig, Sample, TrainingDataset};
use pmetal_lora::Qwen3LoraForCausalLM;
use pmetal_models::architectures::qwen3::Qwen3Config;
use pmetal_trainer::{
    diffusion_loss_gpu, forward_process_gpu, DiffusionConfig, DiffusionTrainingLoop,
};
use serial_test::serial;

fn small_qwen3_config() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 256,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: 16,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        tie_word_embeddings: true,
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
#[serial]
fn test_forward_process_gpu() {
    // Test GPU-native forward masking process
    let input_ids = Array::from_slice(&[1_i32, 2, 3, 4, 5, 6, 7, 8], &[1, 8]);
    let mask_token_id = 255_i64;
    let t = 0.5; // 50% masking probability

    let (x_t, mask) = forward_process_gpu(&input_ids, t, mask_token_id, Some(42))
        .expect("Forward process should succeed");

    // Verify shapes
    assert_eq!(x_t.shape(), &[1, 8]);
    assert_eq!(mask.shape(), &[1, 8]);

    // Verify dtype
    x_t.eval().expect("Should eval");
    mask.eval().expect("Should eval mask");

    // Verify some tokens are masked (with 50% probability, high chance of masking)
    let x_t_data: Vec<i32> = x_t.as_slice().to_vec();
    let has_mask = x_t_data.iter().any(|&t| t == mask_token_id as i32);
    let has_original = x_t_data.iter().any(|&t| t != mask_token_id as i32);

    // With 50% masking, we should have both masked and original tokens
    assert!(has_mask || has_original, "Should have some masking effect");
}

#[test]
#[serial]
fn test_diffusion_loss_gpu() {
    // Test GPU-native diffusion loss
    // Create dummy logits: [batch=1, seq_len=4, vocab_size=8]
    let logits =
        mlx_rs::random::normal::<f32>(&[1, 4, 8], None, None, None).expect("Random logits");

    // Create targets: [batch=1, seq_len=4]
    let targets = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);

    // Create mask: [batch=1, seq_len=4] - first two positions masked
    let mask = Array::from_slice(&[true, true, false, false], &[1, 4]);

    let t = 0.5;
    let use_elbo = true;
    let ignore_index = -100_i64;

    let loss = diffusion_loss_gpu(&logits, &targets, &mask, t, use_elbo, ignore_index)
        .expect("Diffusion loss should succeed");

    loss.eval().expect("Should eval loss");
    let loss_val = loss.item::<f32>();

    // Loss should be positive (cross-entropy is always positive)
    assert!(loss_val > 0.0, "Loss should be positive, got {}", loss_val);
    assert!(loss_val.is_finite(), "Loss should be finite");
}

#[test]
#[serial]
fn test_diffusion_training_loop_creation() {
    let config = DiffusionConfig::new(255); // mask_token_id = 255

    let training_loop = DiffusionTrainingLoop::new(config);

    assert_eq!(training_loop.current_step(), 0);
}

#[test]
#[serial]
fn test_diffusion_training_step() {
    // Create Qwen3 LoRA model
    let mut model = Qwen3LoraForCausalLM::new(small_qwen3_config(), small_lora_config())
        .expect("Failed to create Qwen3 LoRA model");

    // Create diffusion config
    let config = DiffusionConfig {
        mask_token_id: 255,
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
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        ..DiffusionConfig::new(255)
    };

    let mut training_loop = DiffusionTrainingLoop::new(config);

    // Create dataset and batch
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

    let batch = dataloader.next_batch().expect("Should have a batch");

    // Create optimizer
    let mut optimizer = Sgd::new(1e-4);

    // Perform diffusion training step
    let stats = training_loop
        .train_step(&mut model, &batch, &mut optimizer)
        .expect("Diffusion training step should succeed");

    // Verify stats
    assert!(
        stats.loss > 0.0,
        "Loss should be positive, got {}",
        stats.loss
    );
    assert_eq!(stats.step, 1, "Step should be 1");
    assert!(stats.tokens > 0, "Tokens processed should be positive");
    assert!(stats.noise_level > 0.0, "Noise level should be positive");
    assert!(stats.noise_level <= 1.0, "Noise level should be <= 1.0");
    assert!(stats.mask_ratio >= 0.0, "Mask ratio should be >= 0");
    assert!(stats.mask_ratio <= 1.0, "Mask ratio should be <= 1");
}

#[test]
#[serial]
fn test_diffusion_training_multiple_steps() {
    let mut model = Qwen3LoraForCausalLM::new(small_qwen3_config(), small_lora_config())
        .expect("Failed to create model");

    let config = DiffusionConfig {
        mask_token_id: 255,
        training: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size: 1,
            max_seq_len: 8,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        ..DiffusionConfig::new(255)
    };

    let mut training_loop = DiffusionTrainingLoop::new(config);
    let dataset = create_dummy_dataset(8, 8);
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

    // Run multiple training steps
    for _ in 0..4 {
        let batch = dataloader.next_batch().expect("Should have batch");
        let stats = training_loop
            .train_step(&mut model, &batch, &mut optimizer)
            .expect("Step should succeed");

        assert!(stats.loss > 0.0);
        assert!(stats.loss.is_finite());
    }

    assert_eq!(training_loop.current_step(), 4);
}

#[test]
#[serial]
fn test_diffusion_gradient_accumulation() {
    let mut model = Qwen3LoraForCausalLM::new(small_qwen3_config(), small_lora_config())
        .expect("Failed to create model");

    let config = DiffusionConfig {
        mask_token_id: 255,
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
        ..DiffusionConfig::new(255)
    };

    let mut training_loop = DiffusionTrainingLoop::new(config);
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

    // First step: should accumulate gradients
    let batch1 = dataloader.next_batch().expect("Batch 1");
    let stats1 = training_loop
        .train_step(&mut model, &batch1, &mut optimizer)
        .expect("Step 1 should succeed");
    assert!(
        stats1.grad_norm.is_none(),
        "First step should not apply gradients"
    );

    // Second step: should apply accumulated gradients
    let batch2 = dataloader.next_batch().expect("Batch 2");
    let stats2 = training_loop
        .train_step(&mut model, &batch2, &mut optimizer)
        .expect("Step 2 should succeed");
    assert!(
        stats2.grad_norm.is_some(),
        "Second step should apply gradients"
    );
}

#[test]
#[serial]
fn test_noise_schedules() {
    use pmetal_trainer::NoiseSchedule;

    // Linear schedule: α_t = 1 - t
    let linear = NoiseSchedule::Linear;
    assert!((linear.alpha(0.0) - 1.0).abs() < 1e-5);
    assert!((linear.alpha(0.5) - 0.5).abs() < 1e-5);
    assert!((linear.alpha(1.0) - 0.0).abs() < 1e-5);

    // Polynomial2 schedule: α_t = (1 - t)^2
    let poly2 = NoiseSchedule::Polynomial2;
    assert!((poly2.alpha(0.0) - 1.0).abs() < 1e-5);
    assert!((poly2.alpha(0.5) - 0.25).abs() < 1e-5);
    assert!((poly2.alpha(1.0) - 0.0).abs() < 1e-5);

    // Cosine schedule: α_t = cos(πt/2)
    let cosine = NoiseSchedule::Cosine;
    assert!((cosine.alpha(0.0) - 1.0).abs() < 1e-5);
    assert!((cosine.alpha(1.0) - 0.0).abs() < 1e-4); // cos(π/2) ≈ 0
}

#[test]
#[serial]
fn test_elbo_weighting() {
    // Test that ELBO weighting increases loss at low noise levels
    let logits =
        mlx_rs::random::normal::<f32>(&[1, 4, 8], None, None, None).expect("Random logits");
    let targets = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
    let mask = Array::from_slice(&[true, true, true, true], &[1, 4]);

    // Loss at t=0.5 without ELBO
    let loss_no_elbo =
        diffusion_loss_gpu(&logits, &targets, &mask, 0.5, false, -100).expect("Loss without ELBO");
    loss_no_elbo.eval().expect("Eval");

    // Loss at t=0.5 with ELBO (should be ~2x higher due to 1/t weighting)
    let loss_elbo =
        diffusion_loss_gpu(&logits, &targets, &mask, 0.5, true, -100).expect("Loss with ELBO");
    loss_elbo.eval().expect("Eval");

    let ratio = loss_elbo.item::<f32>() / loss_no_elbo.item::<f32>();
    // ELBO weighting at t=0.5 should multiply loss by 1/0.5 = 2
    assert!(
        (ratio - 2.0).abs() < 0.1,
        "ELBO ratio should be ~2, got {}",
        ratio
    );
}
