//! End-to-end smoke test for the Phase-1 pretraining loop.
//!
//! Builds a deliberately-tiny GPT-OSS model from scratch and trains it for
//! a handful of steps on a fixed batch of synthetic token ids. Asserts the
//! loss decreases meaningfully — this proves that forward, autograd, and
//! the AdamW optimizer step are correctly wired through the full-parameter
//! path without LoRA or any dataset plumbing.

use pmetal_bridge::compat::{Dtype, random};
use pmetal_models::architectures::{GptOssConfig, GptOssForCausalLM};
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
        random::randint(0, vocab, &[batch, seq], Dtype::Int32).expect("random tokens");
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
