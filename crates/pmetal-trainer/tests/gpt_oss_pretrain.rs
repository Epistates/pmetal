use std::fs;

use pmetal_models::architectures::GptOssConfig;
use pmetal_trainer::pretrain::{PretrainModel, create_model};

fn tiny_gpt_oss_config() -> GptOssConfig {
    let mut config = GptOssConfig {
        model_type: "gpt_oss".to_string(),
        vocab_size: 128,
        hidden_size: 64,
        intermediate_size: 96,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        max_position_embeddings: 256,
        initial_context_length: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 150_000.0,
        rope_scaling: None,
        attention_bias: true,
        attention_dropout: 0.0,
        tie_word_embeddings: false,
        num_local_experts: 4,
        experts_per_token: 2,
        num_experts_per_tok: None,
        router_aux_loss_coef: 0.0,
        output_router_logits: false,
        sliding_window: 32,
        layer_types: vec![],
        swiglu_limit: 7.0,
        hidden_act: "silu".to_string(),
        eos_token_id: 0,
        pad_token_id: 0,
    };
    // Ensure alternating sliding/full pattern uses the default derivation in attention_type_at.
    config.layer_types.clear();
    config
}

#[test]
fn pretrain_factory_builds_gpt_oss_from_config() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("config.json");
    fs::write(
        &config_path,
        serde_json::to_string_pretty(&tiny_gpt_oss_config()).expect("config string"),
    )
    .expect("write config");

    let model = create_model("gpt-oss", Some(&config_path)).expect("create_model");
    match model {
        PretrainModel::GptOss(m) => {
            assert_eq!(m.config().hidden_size, 64);
            assert_eq!(m.config().num_local_experts, 4);
            assert_eq!(m.config().experts_per_token, 2);
        }
        other => panic!(
            "expected GptOss pretrain model, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

#[test]
fn pretrain_factory_accepts_gpt_oss_underscore_alias() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("config.json");
    fs::write(
        &config_path,
        serde_json::to_string_pretty(&tiny_gpt_oss_config()).expect("config string"),
    )
    .expect("write config");

    let model = create_model("gpt_oss", Some(&config_path)).expect("create_model");
    assert!(matches!(model, PretrainModel::GptOss(_)));
}
