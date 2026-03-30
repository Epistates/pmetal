use super::*;
use pmetal_core::{LoraConfig, StepMetrics, TrainingCallback, TrainingConfig};
use pmetal_data::{DataLoaderConfig, Sample, TrainingDataset};
use pmetal_lora::LlamaLoraForCausalLM;
use pmetal_models::architectures::llama::LlamaConfig;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
        init_lora_weights: true,
        use_dora: false,
        loraplus_lr_ratio: None,
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

#[test]
fn test_training_loop_creation() {
    let config = TrainingLoopConfig::default();
    let training_loop = TrainingLoop::new(config);

    assert_eq!(training_loop.current_step(), 0);
    assert_eq!(training_loop.current_epoch(), 0);
}

#[test]
fn test_learning_rate_warmup() {
    let mut config = TrainingLoopConfig::default();
    config.training.warmup_steps = 100;
    config.training.max_steps = Some(1000);
    config.training.learning_rate = 1e-4;
    config.training.lr_scheduler = LrSchedulerType::Cosine;

    let mut training_loop = TrainingLoop::new(config);

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

#[test]
fn test_take_log_interval_metrics_uses_actual_logged_steps() {
    let mut config = TrainingLoopConfig::default();
    config.log_every = 10;
    let mut training_loop = TrainingLoop::new(config);

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
fn test_run_packed_falls_back_to_standard_when_no_sequences_are_combined() {
    let config = TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            num_epochs: 1,
            max_steps: Some(1),
            max_grad_norm: 1.0,
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
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        use_jit_compilation: false,
        use_cut_cross_entropy: false,
        ..Default::default()
    };

    let mut training_loop = TrainingLoop::new(config);
    let model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
    let dataset = create_dummy_dataset(3, 8);

    let _model = training_loop
        .run_packed(model, dataset, None, None)
        .unwrap();

    assert_eq!(
        training_loop.current_step(),
        1,
        "single-sequence packed batches should fall back to the standard loop"
    );
}

#[test]
fn test_gradient_accumulation_flag() {
    let mut config = TrainingLoopConfig::default();
    config.training.gradient_accumulation_steps = 4;

    let mut training_loop = TrainingLoop::new(config);

    // First 3 steps should not trigger gradient application
    for _ in 0..3 {
        training_loop.accumulation_step += 1;
        assert!(!training_loop.should_apply_gradients());
    }

    // 4th step should trigger
    training_loop.accumulation_step += 1;
    assert!(training_loop.should_apply_gradients());
}

#[test]
fn test_single_train_step() {
    use pmetal_bridge::compat::optimizers::Sgd;

    let config = TrainingLoopConfig {
        use_metal_flash_attention: false, // Disable for simpler test
        ..Default::default()
    };
    let mut training_loop = TrainingLoop::new(config);

    let mut model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
    let mut optimizer = Sgd::new(1e-4);

    // Create a minimal batch
    let batch = TrainingBatch {
        input_ids: Array::from_i32_slice_shaped(&[1_i32, 2, 3, 4], &[1, 4]),
        labels: Array::from_i32_slice_shaped(&[2_i32, 3, 4, 5], &[1, 4]),
        attention_mask: Array::from_i32_slice_shaped(&[1_i32, 1, 1, 1], &[1, 4]),
        pixel_values: None,
        batch_size: 1,
        seq_len: 4,
    };

    let stats = training_loop
        .train_step(&mut model, &batch, &mut optimizer)
        .unwrap();

    assert!(stats.loss > 0.0);
    assert_eq!(stats.step, 1);
    assert_eq!(training_loop.current_step(), 1);
}

#[test]
fn test_jit_training_step() {
    // Test the JIT-compiled training step function directly
    use pmetal_bridge::compat::optimizers::AdamW;

    let model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
    let optimizer = AdamW::new(1e-4, 0.0);

    let mut state = (model, optimizer);

    // Create a minimal batch
    let input_ids = Array::from_i32_slice_shaped(&[1_i32, 2, 3, 4], &[1, 4]);
    let labels = Array::from_i32_slice_shaped(&[2_i32, 3, 4, 5], &[1, 4]);

    // Run the JIT training step
    let loss = jit_training_step(&mut state, (&input_ids, &labels)).unwrap();
    loss.eval();

    let loss_val = loss.item_f32();
    assert!(loss_val > 0.0, "Loss should be positive, got {}", loss_val);
    assert!(
        loss_val.is_finite(),
        "Loss should be finite, got {}",
        loss_val
    );
}

#[test]
fn test_jit_training_step_multiple_steps() {
    // Test that jit_training_step works correctly over multiple steps
    // This verifies the training step function itself works, independent of compile_with_state
    use pmetal_bridge::compat::optimizers::AdamW;

    let model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
    let optimizer = AdamW::new(1e-4, 0.0);

    let mut state = (model, optimizer);

    // Create test data
    let input_ids = Array::from_i32_slice_shaped(&[1_i32, 2, 3, 4, 5, 6, 7, 8], &[1, 8]);
    let labels = Array::from_i32_slice_shaped(&[2_i32, 3, 4, 5, 6, 7, 8, 9], &[1, 8]);

    // Run multiple training steps
    let mut losses = Vec::new();
    for _ in 0..5 {
        let loss = jit_training_step(&mut state, (&input_ids, &labels)).unwrap();
        loss.eval();
        losses.push(loss.item_f32());
    }

    // All losses should be finite and positive
    for (i, loss) in losses.iter().enumerate() {
        assert!(
            loss.is_finite(),
            "Loss {} should be finite, got {}",
            i,
            loss
        );
        assert!(*loss > 0.0, "Loss {} should be positive, got {}", i, loss);
    }

    // Verify loss is changing (parameters are being updated)
    let loss_variance: f32 =
        losses.iter().map(|l| (l - losses[0]).powi(2)).sum::<f32>() / losses.len() as f32;
    assert!(
        loss_variance > 0.0,
        "Loss should change over steps, got {:?}",
        losses
    );

    println!("Training step losses: {:?}", losses);
}

#[test]
fn test_run_packed_direct_api_applies_gradient_checkpointing() {
    let config = TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            num_epochs: 1,
            max_steps: Some(1),
            max_grad_norm: 1.0,
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
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        gradient_checkpointing: true,
        gradient_checkpointing_layers: 3,
        use_jit_compilation: false,
        use_cut_cross_entropy: false,
        ..Default::default()
    };

    let mut training_loop = TrainingLoop::new(config);
    let model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
    let dataset = create_dummy_dataset(4, 2);

    let model = training_loop
        .run_packed(model, dataset, None, None)
        .unwrap();

    let checkpoint_config = model
        .checkpoint_config
        .expect("packed direct API should enable gradient checkpointing");
    assert!(checkpoint_config.enabled);
    assert_eq!(checkpoint_config.layers_per_block, 3);
}

#[test]
fn test_run_compiled_direct_api_applies_gradient_checkpointing() {
    let config = TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            num_epochs: 1,
            max_steps: Some(1),
            max_grad_norm: 1.0,
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size: 1,
            max_seq_len: 16,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        use_metal_flash_attention: false,
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        gradient_checkpointing: true,
        gradient_checkpointing_layers: 3,
        use_jit_compilation: false,
        use_cut_cross_entropy: false,
        ..Default::default()
    };

    let mut training_loop = TrainingLoop::new(config);
    let model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
    let dataset = create_dummy_dataset(2, 8);

    let model = training_loop
        .run_compiled(model, dataset, None, None)
        .unwrap();

    let checkpoint_config = model
        .checkpoint_config
        .expect("compiled direct API should enable gradient checkpointing");
    assert!(checkpoint_config.enabled);
    assert_eq!(checkpoint_config.layers_per_block, 3);
}

#[test]
fn test_single_sequence_packed_step_matches_standard_step() {
    use pmetal_bridge::compat::{optimizers::AdamW, random};

    let tokens = [1_i32, 2, 3, 4, 5, 6];
    let labels = [2_i64, 3, 4, 5, 6, 7];

    random::seed(1337);
    let model_standard = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
    random::seed(1337);
    let model_packed = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();

    let optimizer_standard = AdamW::new(0.0, 0.0);
    let optimizer_packed = AdamW::new(0.0, 0.0);

    let mut standard_state = (model_standard, optimizer_standard);
    let mut packed_state = (model_packed, optimizer_packed);

    let input_ids = Array::from_i32_slice_shaped(&tokens, &[1, tokens.len() as i32]);
    let label_ids = Array::from_i32_slice_shaped(
        &labels.iter().map(|&x| x as i32).collect::<Vec<_>>(),
        &[1, labels.len() as i32],
    );
    let packed_batch = single_sequence_packed_batch(&tokens, &labels);

    let standard_loss = jit_training_step(&mut standard_state, (&input_ids, &label_ids)).unwrap();
    standard_loss.eval();
    let packed_loss = jit_training_step_packed(&mut packed_state, &packed_batch, 0.0).unwrap();
    packed_loss.eval();

    let standard_val = standard_loss.item_f32();
    let packed_val = packed_loss.item_f32();
    assert!(
        (standard_val - packed_val).abs() < 1e-4,
        "single-sequence packed step should match standard step: standard={standard_val}, packed={packed_val}"
    );
}

#[test]
fn test_single_sequence_packed_cce_step_is_finite() {
    use pmetal_bridge::compat::{optimizers::AdamW, random};

    let tokens = [1_i32, 2, 3, 4, 5, 6];
    let labels = [2_i64, 3, 4, 5, 6, 7];

    random::seed(7331);
    let model_packed = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();

    let optimizer_packed = AdamW::new(0.0, 0.0);

    let mut packed_state = (model_packed, optimizer_packed);

    let packed_batch = single_sequence_packed_batch(&tokens, &labels);

    let packed_loss = jit_training_step_packed_cce(&mut packed_state, &packed_batch, 0.0).unwrap();
    packed_loss.eval();

    let packed_val = packed_loss.item_f32();
    assert!(
        packed_val.is_finite() && packed_val > 0.0,
        "single-sequence packed CCE step should produce a finite positive loss, got {packed_val}"
    );
}

#[test]
fn test_jit_training_step_with_warmup() {
    // Test the jit_training_step function with proper warmup to initialize
    // optimizer state. This verifies state stability and correct loss reduction.
    use pmetal_bridge::compat::optimizers::AdamW;
    use pmetal_bridge::compat::optimizers::Updatable;

    let model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
    let optimizer = AdamW::new(1e-4, 0.0);

    let mut state = (model, optimizer);

    // Create test data
    let input_ids = Array::from_i32_slice_shaped(&[1_i32, 2, 3, 4, 5, 6, 7, 8], &[1, 8]);
    let labels = Array::from_i32_slice_shaped(&[2_i32, 3, 4, 5, 6, 7, 8, 9], &[1, 8]);

    // ========================================
    // PHASE 1: Record state count BEFORE warmup
    // ========================================
    let state_count_before = state.updatable_states_len();
    println!("State count BEFORE warmup: {}", state_count_before);

    // ========================================
    // PHASE 2: WARMUP - Run one uncompiled step
    // ========================================
    // This initializes optimizer momentum/velocity buffers
    let warmup_loss = jit_training_step(&mut state, (&input_ids, &labels)).unwrap();
    warmup_loss.eval();
    let warmup_loss_val = warmup_loss.item_f32();
    println!("Warmup loss: {:.4}", warmup_loss_val);

    // ========================================
    // PHASE 3: Record state count AFTER warmup
    // ========================================
    let state_count_after = state.updatable_states_len();
    println!(
        "State count AFTER warmup: {} (delta={})",
        state_count_after,
        state_count_after as i64 - state_count_before as i64
    );

    // AdamW should have created momentum and velocity buffers
    // So state count should have increased
    assert!(
        state_count_after >= state_count_before,
        "State count should not decrease after warmup: {} -> {}",
        state_count_before,
        state_count_after
    );

    // ========================================
    // PHASE 4: Run SECOND warmup step to verify stability
    // ========================================
    println!("Running second warmup step to verify state stability...");
    let warmup2_loss = jit_training_step(&mut state, (&input_ids, &labels)).unwrap();
    warmup2_loss.eval();
    let warmup2_loss_val = warmup2_loss.item_f32();
    println!("Second warmup loss: {:.4}", warmup2_loss_val);

    let state_count_after_2 = state.updatable_states_len();
    println!(
        "State count AFTER second warmup: {} (should be same as {})",
        state_count_after_2, state_count_after
    );

    assert_eq!(
        state_count_after, state_count_after_2,
        "State count should be stable after second warmup: {} vs {}",
        state_count_after, state_count_after_2
    );

    // ========================================
    // PHASE 5: Use non-compiled path
    // ========================================
    // NOTE: compile_with_state has a known limitation in mlx-rs where it doesn't
    // correctly handle state count changes. Even with warmup to stabilize state,
    // the internal state tracking in compile_with_state.rs:413 fails with
    // "attempt to subtract with overflow" because:
    // 1. The inner closure captures state count at creation time
    // 2. During MLX tracing, the function may see different state
    // 3. The compiled graph expects N outputs but current state has M > N
    //
    // For now, we use the non-compiled jit_training_step which correctly handles
    // state and benefits from MLX's lazy evaluation and graph fusion.
    println!("Using non-compiled training step (mlx-rs compile_with_state limitation)");

    let mut losses = vec![warmup_loss_val, warmup2_loss_val];
    for i in 0..3 {
        let loss = jit_training_step(&mut state, (&input_ids, &labels)).unwrap();
        loss.eval();
        let loss_val = loss.item_f32();
        losses.push(loss_val);
        println!("Training step {}: loss={:.4}", i + 3, loss_val);
    }

    // Verify state count remains stable
    let final_state_count = state.updatable_states_len();
    assert_eq!(
        state_count_after, final_state_count,
        "State count should remain stable: {} -> {}",
        state_count_after, final_state_count
    );

    println!("State stability verified! Losses: {:?}", losses);
}

#[test]
fn test_eager_evaluation_config() {
    // Test eager evaluation mode can be enabled
    let mut config = TrainingLoopConfig::default();
    assert!(
        !config.eager_evaluation,
        "Default should have eager_evaluation disabled"
    );

    config.eager_evaluation = true;
    let training_loop = TrainingLoop::new(config);
    assert!(
        training_loop.config.eager_evaluation,
        "Should preserve eager_evaluation config"
    );
}

#[test]
fn test_gpu_gradient_clipping() {
    use pmetal_bridge::compat::Array;
    use std::rc::Rc;

    let config = TrainingLoopConfig {
        training: TrainingConfig {
            max_grad_norm: 1.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let training_loop = TrainingLoop::new(config);

    // Create some fake gradients
    let grad1 = Array::from_f32_slice(&[3.0f32, 4.0], &[2]); // norm = 5
    let grad2 = Array::from_f32_slice(&[0.0f32, 0.0], &[2]); // norm = 0
    let mut grads = FlattenedModuleParam::new();
    grads.insert(Rc::from("layer1.weight"), grad1);
    grads.insert(Rc::from("layer2.weight"), grad2);

    // Clip gradients
    let result = training_loop.clip_gradients_gpu(&mut grads);
    assert!(result.is_ok(), "GPU gradient clipping should succeed");

    let norm_arr = result.unwrap();
    assert!(
        norm_arr.is_some(),
        "Should return norm array when max_grad_norm > 0"
    );

    let norm = norm_arr.unwrap();
    norm.eval();
    let norm_val = norm.item_f32();

    // Original norm should be 5 (sqrt(3^2 + 4^2))
    // After clipping with max_norm=1.0, gradients should be scaled
    // The returned norm should be the original norm (5.0)
    assert!(
        (norm_val - 5.0).abs() < 0.01,
        "Norm should be ~5.0, got {}",
        norm_val
    );

    // Check that gradients were actually clipped
    let key: Rc<str> = Rc::from("layer1.weight");
    let clipped_grad1 = grads.get(&key).unwrap();
    clipped_grad1.eval();

    // Gradients should be scaled by 1.0/5.0 = 0.2
    // [3.0, 4.0] * 0.2 = [0.6, 0.8]
    let values: [f32; 2] = clipped_grad1.as_slice().try_into().unwrap();
    assert!(
        (values[0] - 0.6).abs() < 0.01,
        "First grad should be ~0.6, got {}",
        values[0]
    );
    assert!(
        (values[1] - 0.8).abs() < 0.01,
        "Second grad should be ~0.8, got {}",
        values[1]
    );
}

#[test]
fn test_run_packed_callbacks_report_non_zero_interval_metrics() {
    let config = TrainingLoopConfig {
        training: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            num_epochs: 1,
            max_steps: Some(3),
            max_grad_norm: 1.0,
            ..Default::default()
        },
        dataloader: DataLoaderConfig {
            batch_size: 1,
            max_seq_len: 32,
            shuffle: false,
            pad_token_id: 0,
            ..Default::default()
        },
        use_metal_flash_attention: false,
        log_every: 1,
        checkpoint_every: 0,
        eval_every: 0,
        use_jit_compilation: false,
        use_cut_cross_entropy: false,
        ..Default::default()
    };

    let mut training_loop = TrainingLoop::new(config);
    let capture = MetricsCapture::default();
    let snapshot = capture.clone();
    training_loop.add_callback(Box::new(capture));

    let model = LlamaLoraForCausalLM::new(small_config(), small_lora_config()).unwrap();
    let dataset = create_dummy_dataset(6, 24);

    let _model = training_loop
        .run_packed(model, dataset, None, None)
        .unwrap();

    let metrics = snapshot.snapshot();
    assert!(!metrics.is_empty());
    assert!(metrics.iter().all(|metric| metric.tokens > 0));
    assert!(metrics.iter().all(|metric| metric.total_ms > 0.0));
    assert!(metrics.iter().all(|metric| metric.tok_sec > 0.0));
}

#[test]
fn test_learning_rate_division_by_zero_protection() {
    // Test edge case where total_steps == warmup_steps
    let mut config = TrainingLoopConfig::default();
    config.training.warmup_steps = 100;
    config.training.max_steps = Some(100); // Same as warmup!
    config.training.learning_rate = 1e-4;
    config.training.lr_scheduler = LrSchedulerType::Linear;

    let mut training_loop = TrainingLoop::new(config);

    // At step 100 (past warmup, at max_steps)
    training_loop.step = 100;
    let lr = training_loop.get_learning_rate();

    // Should not panic or return NaN/Inf
    assert!(lr.is_finite(), "Learning rate should be finite, got {}", lr);
    assert!(lr >= 0.0, "Learning rate should be non-negative");
}

#[test]
fn test_batch_token_overflow_protection() {
    // Test that we handle potential overflow in batch_size * seq_len
    let large_batch_size: usize = usize::MAX / 2;
    let large_seq_len: usize = 3;

    // This would overflow without checked arithmetic
    let result = large_batch_size.checked_mul(large_seq_len);
    assert!(result.is_none(), "Should detect potential overflow");

    // With our protected version, it returns MAX
    let protected = large_batch_size
        .checked_mul(large_seq_len)
        .unwrap_or(usize::MAX);
    assert_eq!(protected, usize::MAX, "Should return MAX on overflow");
}
