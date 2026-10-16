//! A checkpoint has to account for the whole model, and the model for the whole
//! checkpoint.
//!
//! `DynamicModel::load` used to succeed when a tensor reached no parameter and
//! when a parameter was never filled. Either one leaves the model computing
//! something other than what was published, behind finite, plausibly shaped
//! logits. These tests hold every architecture that loads by parameter name to
//! the rule from both sides, on checkpoints built from the model's own
//! parameters so the only difference between the passing and the failing case is
//! the one tensor the test changes.

use std::collections::HashMap;
use std::path::Path;

use pmetal_models::DynamicModel;
use pmetal_runtime::compat::{Array, ModuleParametersExt};

/// `(name, tiny config.json)` for each architecture whose checkpoint keys are
/// its parameter names.
const NAME_MATCHED: &[(&str, &str)] = &[
    (
        "llama",
        r#"{"model_type": "llama", "vocab_size": 64, "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 128, "rope_theta": 10000.0, "layer_norm_eps": 1e-5, "tie_word_embeddings": false}"#,
    ),
    (
        "qwen2",
        r#"{"model_type": "qwen2", "vocab_size": 64, "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 128, "rope_theta": 10000.0, "layer_norm_eps": 1e-5, "tie_word_embeddings": false}"#,
    ),
    (
        "qwen3",
        r#"{"model_type": "qwen3", "vocab_size": 64, "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 128, "rope_theta": 10000.0, "layer_norm_eps": 1e-5, "tie_word_embeddings": false}"#,
    ),
    (
        "mistral",
        r#"{"model_type": "mistral", "vocab_size": 64, "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 128, "rope_theta": 10000.0, "layer_norm_eps": 1e-5, "tie_word_embeddings": false}"#,
    ),
    (
        "cohere",
        r#"{"model_type": "cohere", "vocab_size": 64, "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 128, "rope_theta": 10000.0, "layer_norm_eps": 1e-5, "tie_word_embeddings": true}"#,
    ),
];

fn save(path: &Path, weights: &HashMap<String, Array>) {
    let entries: Vec<_> = weights.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(path.to_string_lossy().as_ref(), &entries).unwrap();
}

/// A checkpoint directory holding exactly the parameters `config` describes.
fn checkpoint(config: &str) -> (tempfile::TempDir, HashMap<String, Array>) {
    let dir = tempfile::tempdir().unwrap();
    let model = DynamicModel::from_config(config).unwrap();
    let weights: HashMap<String, Array> = model
        .flatten_params()
        .into_iter()
        .map(|(name, array)| (name.to_string(), array))
        .collect();
    std::fs::write(dir.path().join("config.json"), config).unwrap();
    save(&dir.path().join("model.safetensors"), &weights);
    (dir, weights)
}

fn rewrite(dir: &Path, weights: &HashMap<String, Array>) {
    save(&dir.join("model.safetensors"), weights);
}

#[test]
fn an_exact_checkpoint_loads() {
    for (name, config) in NAME_MATCHED {
        let (dir, weights) = checkpoint(config);
        let model = DynamicModel::load(dir.path()).unwrap_or_else(|e| {
            panic!("{name}: a checkpoint of the model's own parameters should load: {e}")
        });
        assert_eq!(model.flatten_params().len(), weights.len(), "{name}");
        pmetal_runtime::check_unobserved_error().unwrap();
    }
}

#[test]
fn a_missing_tensor_is_an_error_naming_it() {
    for (name, config) in NAME_MATCHED {
        let (dir, mut weights) = checkpoint(config);
        let victim = "model.layers.1.mlp.down_proj.weight";
        assert!(weights.remove(victim).is_some(), "{name}: no {victim}");
        rewrite(dir.path(), &weights);

        let err = DynamicModel::load(dir.path())
            .expect_err(&format!(
                "{name}: a checkpoint missing {victim} must not load"
            ))
            .to_string();
        assert!(
            err.contains(victim),
            "{name}: the error should name {victim}: {err}"
        );
    }
}

#[test]
fn a_tensor_nothing_consumes_is_an_error_naming_it() {
    for (name, config) in NAME_MATCHED {
        let (dir, mut weights) = checkpoint(config);
        let stray = "model.layers.0.mlp.shared_expert.up_proj.weight";
        weights.insert(stray.into(), Array::from_f32_slice(&[1.0; 4], &[2, 2]));
        rewrite(dir.path(), &weights);

        let err = DynamicModel::load(dir.path())
            .expect_err(&format!("{name}: a checkpoint with {stray} must not load"))
            .to_string();
        assert!(
            err.contains(stray),
            "{name}: the error should name {stray}: {err}"
        );
    }
}
