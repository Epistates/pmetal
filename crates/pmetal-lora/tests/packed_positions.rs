//! Packed positions reaching the model an adapter is attached to.
//!
//! `pmetal-models`' `tests/packed_positions.rs` holds each architecture to the
//! numerical claim. This one holds the trainer's side of the contract: the
//! adapted model answers `supports_packed_positions` from the architecture it
//! wraps rather than from a constant, and `forward_with_positions` reaches the
//! rotation instead of accepting the positions and dropping them.
//!
//! Both failures are silent. A run with dropped positions finishes, reports
//! the same loss, and produces an adapter trained on positions the model will
//! never see at inference.

use pmetal_bridge::compat::Array;
use pmetal_core::LoraConfig;
use pmetal_lora::{AdaptedModel, TrainableModel};
use pmetal_models::dispatcher::DynamicModel;

const CASES: &[(&str, &str)] = &[
    (
        "llama",
        r#"{
            "model_type": "llama",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }"#,
    ),
    (
        "qwen3",
        r#"{
            "model_type": "qwen3",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }"#,
    ),
    (
        "mistral",
        r#"{
            "model_type": "mistral",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "sliding_window": null, "tie_word_embeddings": false
        }"#,
    ),
    (
        "cohere",
        r#"{
            "model_type": "cohere",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16,
            "max_position_embeddings": 256,
            "layer_norm_eps": 1e-5, "rope_theta": 10000.0,
            "logit_scale": 0.0625, "use_sliding_window": false
        }"#,
    ),
];

const SEQ_LEN: i32 = 16;

fn build(config_json: &str) -> AdaptedModel {
    let config = LoraConfig {
        r: 4,
        alpha: 8.0,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        ..Default::default()
    };
    let model = DynamicModel::from_config(config_json).expect("architecture should build");
    AdaptedModel::attach(model, config).expect("adapters should attach")
}

fn input_ids() -> Array {
    let tokens: Vec<i32> = (0..SEQ_LEN).map(|i| (i * 5 + 1) % 128).collect();
    Array::from_slice(&tokens, &[1, SEQ_LEN])
}

#[test]
fn every_case_here_claims_packed_positions() {
    for (name, config_json) in CASES {
        let model = build(config_json);
        assert!(
            TrainableModel::supports_packed_positions(&model),
            "{name}: the adapted model says it cannot apply packed positions, so the \
             assertion below would be checking nothing"
        );
    }
}

/// Stretching the positions has to change the logits. A uniform shift would
/// not: RoPE rotates by the difference between positions, so it cancels.
#[test]
fn the_positions_reach_the_rotation() {
    let ids = input_ids();
    let contiguous: Vec<i32> = (0..SEQ_LEN).collect();
    let stretched: Vec<i32> = (0..SEQ_LEN).map(|i| i * 2).collect();
    let contiguous = Array::from_slice(&contiguous, &[SEQ_LEN]);
    let stretched = Array::from_slice(&stretched, &[SEQ_LEN]);

    for (name, config_json) in CASES {
        let mut model = build(config_json);

        let plain = TrainableModel::forward_with_positions(&mut model, &ids, None, &contiguous)
            .expect("contiguous forward");
        let spread = TrainableModel::forward_with_positions(&mut model, &ids, None, &stretched)
            .expect("stretched forward");

        let difference = plain.subtract(&spread).abs().max(None).item_f32();
        assert!(
            difference > 1e-4,
            "{name}: stretching the positions changed the logits by {difference:e}, so \
             `forward_with_positions` is dropping them"
        );
        assert!(
            difference.is_finite(),
            "{name}: forward_with_positions produced a non-finite output"
        );
    }
}

/// The contiguous run has to match the plain forward, or the positions path is
/// computing something else entirely rather than the same thing with positions
/// stated explicitly.
#[test]
fn stating_the_default_positions_changes_nothing() {
    let ids = input_ids();
    let contiguous: Vec<i32> = (0..SEQ_LEN).collect();
    let contiguous = Array::from_slice(&contiguous, &[SEQ_LEN]);

    for (name, config_json) in CASES {
        let mut model = build(config_json);

        let implicit = TrainableModel::forward(&mut model, &ids, None).expect("plain forward");
        let explicit = TrainableModel::forward_with_positions(&mut model, &ids, None, &contiguous)
            .expect("positions forward");

        let difference = implicit.subtract(&explicit).abs().max(None).item_f32();
        let scale = implicit.abs().max(None).item_f32().max(1e-6);
        assert!(
            difference / scale < 1e-4,
            "{name}: passing 0, 1, 2, … explicitly gave different logits from the \
             default (relative {:e})",
            difference / scale
        );
    }
}
