use std::fs;

use pmetal_models::architectures::Qwen3MoEConfig;
use pmetal_trainer::pretrain::{PretrainModel, create_model};

fn tiny_qwen3_moe_config() -> Qwen3MoEConfig {
    Qwen3MoEConfig {
        model_type: "qwen3_moe".to_string(),
        vocab_size: 128,
        hidden_size: 64,
        intermediate_size: 96,
        moe_intermediate_size: Some(48),
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: 16,
        max_position_embeddings: 256,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        tie_word_embeddings: false,
        num_experts: 4,
        num_experts_per_tok: 2,
        decoder_sparse_step: 1,
        mlp_only_layers: vec![],
        norm_topk_prob: true,
        rope_scaling: None,
    }
}

#[test]
fn pretrain_factory_builds_qwen3_moe_from_config() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("config.json");
    fs::write(
        &config_path,
        serde_json::to_string_pretty(&tiny_qwen3_moe_config()).expect("config string"),
    )
    .expect("write config");

    let model = create_model("qwen3_moe", Some(&config_path)).expect("create_model");
    match model {
        PretrainModel::Qwen3MoE(m) => {
            assert_eq!(m.config.hidden_size, 64);
            assert_eq!(m.config.num_experts, 4);
            assert_eq!(m.config.num_experts_per_tok, 2);
        }
        other => panic!(
            "expected Qwen3MoE pretrain model, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

#[test]
fn pretrain_factory_accepts_qwen3_moe_dash_alias() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("config.json");
    fs::write(
        &config_path,
        serde_json::to_string_pretty(&tiny_qwen3_moe_config()).expect("config string"),
    )
    .expect("write config");

    let model = create_model("qwen3-moe", Some(&config_path)).expect("create_model");
    assert!(matches!(model, PretrainModel::Qwen3MoE(_)));
}
