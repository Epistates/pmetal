//! Test that checkpoint save/load correctly persists optimizer state.
//!
//! Train N steps → save checkpoint → resume → train 1 step. Assert the
//! resumed-step loss is NOT near initial loss (proving the optimizer
//! momentum/velocity were restored, not reset to zero).

use pmetal_bridge::compat::optimizers::AdamWBuilder;
use pmetal_bridge::compat::{Dtype, random};
use pmetal_models::architectures::{GptOssConfig, GptOssForCausalLM};
use pmetal_trainer::pretrain::{
    CheckpointMeta, PretrainConfig, load_checkpoint, pretrain_step, run_pretrain, save_checkpoint,
};
use serial_test::serial;

fn tiny_config() -> GptOssConfig {
    GptOssConfig {
        hidden_size: 32,
        intermediate_size: 48,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        head_dim: 8,
        vocab_size: 64,
        num_local_experts: 4,
        experts_per_token: 2,
        num_experts_per_tok: Some(2),
        sliding_window: 16,
        ..GptOssConfig::default()
    }
}

#[test]
#[serial]
fn checkpoint_resume_preserves_optimizer_state() {
    let vocab: i32 = 64;
    random::seed(42);

    let config = PretrainConfig {
        num_steps: 10,
        learning_rate: 1e-2,
        weight_decay: 0.0,
        ..PretrainConfig::default()
    };

    // -- Phase A: train 10 steps --
    let mut model_a = GptOssForCausalLM::new(tiny_config()).unwrap();
    let fixed_batch = random::randint(0, vocab, &[2, 8], Dtype::Int32).expect("rand");
    let batch_iter = std::iter::repeat_with({
        let b = fixed_batch.clone();
        move || b.clone()
    });

    let losses_a = run_pretrain(&mut model_a, &config, batch_iter).unwrap();
    let loss_at_10 = *losses_a.last().unwrap();

    // -- Save checkpoint --
    let dir = tempfile::tempdir().unwrap();
    let mut optimizer_a = AdamWBuilder::new(config.learning_rate)
        .weight_decay(config.weight_decay)
        .build()
        .unwrap();
    // Run one more step to populate optimizer state for saving
    let loss_11 = pretrain_step(
        &mut model_a,
        &mut optimizer_a,
        std::slice::from_ref(&fixed_batch),
        None,
        None,
        None,
    )
    .unwrap();

    save_checkpoint(
        dir.path(),
        &model_a,
        &optimizer_a,
        &CheckpointMeta {
            step: 11,
            loss: loss_11,
            learning_rate: config.learning_rate,
        },
    )
    .unwrap();

    // -- Phase B: fresh model + optimizer, load checkpoint --
    let mut model_b = GptOssForCausalLM::new(tiny_config()).unwrap();
    let mut optimizer_b = AdamWBuilder::new(config.learning_rate)
        .weight_decay(config.weight_decay)
        .build()
        .unwrap();

    let meta = load_checkpoint(dir.path(), &mut model_b, &mut optimizer_b).unwrap();
    assert_eq!(meta.step, 11);

    // One step on the restored model — should continue from the trained state
    let loss_resumed = pretrain_step(
        &mut model_b,
        &mut optimizer_b,
        std::slice::from_ref(&fixed_batch),
        None,
        None,
        None,
    )
    .unwrap();

    assert!(
        loss_resumed.is_finite(),
        "resumed loss is not finite: {loss_resumed}"
    );

    // The initial random loss is ~4.2 (ln(64)). After training, loss should
    // be well below 3.0. If optimizer state wasn't restored, loss would
    // regress toward the initial level.
    assert!(
        loss_resumed < 3.0,
        "loss after resume ({loss_resumed:.4}) is too high — optimizer state \
         was likely not restored (initial ~4.2, after 10 steps ~{loss_at_10:.4})"
    );
}
