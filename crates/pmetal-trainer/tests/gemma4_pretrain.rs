use std::fs;

use pmetal_bridge::compat::{Dtype, random};
use pmetal_models::architectures::{Gemma4Config, Gemma4ForCausalLM};
use pmetal_trainer::pretrain::{PretrainConfig, PretrainModel, create_model, run_pretrain};
use serial_test::serial;

fn tiny_gemma4_config() -> Gemma4Config {
    Gemma4Config {
        model_type: "gemma4_text".to_string(),
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 48,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        head_dim: 8,
        global_head_dim: Some(8),
        num_global_key_value_heads: Some(1),
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        attention_k_eq_v: false,
        tie_word_embeddings: true,
        sliding_window: 8,
        final_logit_softcapping: None,
        layer_types: vec![
            "full_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
            "sliding_attention".to_string(),
        ],
        rope_parameters: None,
        _raw_rope_parameters: None,
        hidden_size_per_layer_input: Some(4),
        vocab_size_per_layer_input: Some(64),
        hidden_activation: Some("gelu_tanh".to_string()),
        num_kv_shared_layers: Some(2),
        use_double_wide_mlp: Some(false),
        enable_moe_block: Some(false),
    }
}

#[test]
#[serial]
fn gemma4_pretrain_smoke_with_small_model_features() {
    let vocab: i32 = 64;
    let batch: i32 = 2;
    let seq: i32 = 8;

    random::seed(0xA11CE_u64);
    let mut model = Gemma4ForCausalLM::new(tiny_gemma4_config()).expect("gemma4 model init");

    let fixed_batch =
        random::randint(0, vocab, &[batch, seq], Dtype::Int32).expect("random tokens");
    let batch_iter = std::iter::repeat_with({
        let b = fixed_batch.clone();
        move || b.clone()
    });

    let config = PretrainConfig {
        num_steps: 16,
        learning_rate: 5e-3,
        weight_decay: 0.0,
        warmup_steps: 0,
        max_grad_norm: None,
        ..PretrainConfig::default()
    };

    let losses = run_pretrain(&mut model, &config, batch_iter).expect("run_pretrain");
    assert_eq!(losses.len(), config.num_steps);
    for (i, loss) in losses.iter().enumerate() {
        assert!(loss.is_finite(), "step {i} loss {loss} not finite");
    }

    let first_window: f32 = losses[..4].iter().sum::<f32>() / 4.0;
    let last_window: f32 = losses[losses.len() - 4..].iter().sum::<f32>() / 4.0;
    assert!(
        last_window < first_window - 0.05,
        "loss did not decrease: first4_avg={first_window:.4} last4_avg={last_window:.4}; full={losses:?}"
    );
}

#[test]
fn pretrain_factory_builds_gemma4_from_text_config_wrapper() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("config.json");
    let wrapped = serde_json::json!({
        "model_type": "gemma4",
        "text_config": serde_json::to_value(tiny_gemma4_config()).expect("config json"),
    });
    fs::write(
        &config_path,
        serde_json::to_string_pretty(&wrapped).expect("config string"),
    )
    .expect("write config");

    let model = create_model("gemma4", Some(&config_path)).expect("create_model");
    match model {
        PretrainModel::Gemma4(_) => {}
        other => panic!(
            "expected Gemma4 pretrain model, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
}
