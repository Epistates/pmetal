use std::fs;

use pmetal_models::architectures::{Qwen3NextConfig, RopeParameters};
use pmetal_trainer::pretrain::{PretrainModel, create_model};

fn tiny_qwen3_next_config() -> Qwen3NextConfig {
    Qwen3NextConfig {
        model_type: "qwen3_5_moe_text".to_string(),
        vocab_size: 128,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(16),
        max_position_embeddings: 256,
        linear_num_value_heads: 2,
        linear_num_key_heads: 2,
        linear_key_head_dim: 32,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 2,
        full_attention_interval: 1,
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 128,
        rope_parameters: Some(RopeParameters {
            rope_theta: Some(12_345.0),
            partial_rotary_factor: Some(0.5),
            rope_type: Some("default".to_string()),
            mrope_interleaved: None,
            mrope_section: None,
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn pretrain_factory_builds_qwen35_from_text_config_wrapper_and_applies_rope_parameters() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("config.json");
    let wrapped = serde_json::json!({
        "model_type": "qwen3_5_moe",
        "text_config": serde_json::to_value(tiny_qwen3_next_config()).expect("config json"),
    });
    fs::write(
        &config_path,
        serde_json::to_string_pretty(&wrapped).expect("config string"),
    )
    .expect("write config");

    let model = create_model("qwen3.5", Some(&config_path)).expect("create_model");
    match model {
        PretrainModel::Qwen3Next(model) => {
            assert_eq!(model.config.hidden_size, 64);
            assert_eq!(model.config.rope_theta, 12_345.0);
            assert!((model.config.partial_rotary_factor - 0.5).abs() < 1e-6);
        }
        other => panic!(
            "expected Qwen3Next pretrain model, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
}
