//! End-to-end smoke test for the Phase-1 pretraining loop.
//!
//! Builds a deliberately-tiny GPT-OSS model from scratch and trains it for
//! a handful of steps on a fixed batch of synthetic token ids. Asserts the
//! loss decreases meaningfully — this proves that forward, autograd, and
//! the AdamW optimizer step are correctly wired through the full-parameter
//! path without LoRA or any dataset plumbing.

use pmetal_models::architectures::{GptOssConfig, GptOssForCausalLM};
use pmetal_runtime::compat::{Dtype, random};
use pmetal_trainer::pretrain::{PretrainConfig, run_pretrain};
use serial_test::serial;

fn tiny_gpt_oss_config(vocab: i32) -> GptOssConfig {
    GptOssConfig {
        hidden_size: 32,
        intermediate_size: 48,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        head_dim: 8,
        vocab_size: vocab,
        num_local_experts: 4,
        experts_per_token: 2,
        num_experts_per_tok: Some(2),
        sliding_window: 16,
        ..GptOssConfig::default()
    }
}

#[test]
#[serial]
fn pretrain_smoke_loss_decreases() {
    let vocab: i32 = 64;
    let batch: i32 = 2;
    let seq: i32 = 8;

    random::seed(0xC0FFEE_u64);
    let model_config = tiny_gpt_oss_config(vocab);
    let mut model = GptOssForCausalLM::new(model_config).expect("model init");

    // Deterministic synthetic batch — the loop sees the same tokens every
    // step, so loss must decrease as the model memorises them.
    let fixed_batch =
        random::randint(0, vocab, &[batch, seq], Dtype::Int32, None).expect("random tokens");
    let batch_iter = std::iter::repeat_with(move || fixed_batch.clone());

    let config = PretrainConfig {
        num_steps: 20,
        learning_rate: 1e-2,
        weight_decay: 0.0,
        warmup_steps: 0,
        max_grad_norm: None,
        ..PretrainConfig::default()
    };

    let losses = run_pretrain(&mut model, &config, batch_iter).expect("run_pretrain");

    assert_eq!(losses.len(), config.num_steps, "collected every step");
    for (i, l) in losses.iter().enumerate() {
        assert!(l.is_finite(), "step {i} loss {l} not finite");
    }

    let first_window: f32 = losses[..4].iter().sum::<f32>() / 4.0;
    let last_window: f32 = losses[losses.len() - 4..].iter().sum::<f32>() / 4.0;
    assert!(
        last_window < first_window - 0.1,
        "loss did not decrease: first4_avg={first_window:.4} last4_avg={last_window:.4}; full={losses:?}"
    );
}

/// A stop request ends the run after the step in flight and still writes the
/// final checkpoint, which a supervisor's SIGTERM used to lose: the run died
/// with everything since the last periodic checkpoint.
#[test]
#[serial]
fn a_stop_request_ends_pretraining_after_its_step_and_saves_it() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let vocab: i32 = 64;
    random::seed(7);
    let mut model = GptOssForCausalLM::new(tiny_gpt_oss_config(vocab)).expect("model init");
    let fixed_batch = random::randint(0, vocab, &[2, 8], Dtype::Int32, None).expect("tokens");

    let stop = Arc::new(AtomicBool::new(false));
    let dir = tempfile::tempdir().unwrap();
    let config = PretrainConfig {
        num_steps: 50,
        learning_rate: 1e-2,
        weight_decay: 0.0,
        warmup_steps: 0,
        max_grad_norm: None,
        checkpoint_dir: Some(dir.path().to_path_buf()),
        // Neither lands on the step the stop does.
        checkpoint_every: Some(40),
        log_every: 0,
        stop: Some(Arc::clone(&stop)),
        ..PretrainConfig::default()
    };
    // The request arrives while the fourth step's batch is fetched.
    let batches = (0..).map({
        let stop = Arc::clone(&stop);
        move |i| {
            if i == 3 {
                stop.store(true, Ordering::SeqCst);
            }
            fixed_batch.clone()
        }
    });

    let losses = run_pretrain(&mut model, &config, batches).expect("run_pretrain");

    assert_eq!(losses.len(), 4, "the step in flight finished, no more ran");
    let saved = dir.path().join("final");
    assert!(saved.join("model.safetensors").exists(), "{saved:?}");
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(saved.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(meta["step"], 4);
}
