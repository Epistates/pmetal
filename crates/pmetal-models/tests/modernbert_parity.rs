//! Numerical parity of the ModernBERT encoder against `transformers`.
//!
//! Two tiny seeded encoders with four layers (global, local, local, global)
//! and a local window of 4 tokens either side over 12-token rows, so the
//! window is narrower than the sequence: one with the released models' layout
//! (no biases anywhere) and one with every optional bias on. Each runs a batch
//! of a full, a right-padded and a left-padded row.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, max_abs_diff, print_report_table};
use pmetal_models::architectures::encoder::EncoderModel;
use pmetal_models::architectures::modernbert::{ModernBertConfig, ModernBertModel};
use pmetal_models::loader::{LoadError, load_encoder_weights, load_modernbert_weights};
use pmetal_runtime::compat::{Array, Dtype};

fn config(biased: bool) -> ModernBertConfig {
    ModernBertConfig {
        vocab_size: 120,
        hidden_size: 32,
        intermediate_size: 48,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        max_position_embeddings: 64,
        local_attention: 8,
        global_attn_every_n_layers: 3,
        global_rope_theta: Some(1000.0),
        local_rope_theta: Some(100.0),
        pad_token_id: Some(0),
        norm_bias: biased,
        attention_bias: biased,
        mlp_bias: biased,
        ..Default::default()
    }
}

fn weights(variant: &str) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!(
        "modernbert_{variant}_weights.safetensors"
    )))
}

fn reference(variant: &str) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!(
        "modernbert_{variant}_reference.safetensors"
    )))
}

fn model(variant: &str) -> ModernBertModel {
    let mut model = ModernBertModel::new(config(variant == "biased")).expect("builds");
    load_modernbert_weights(&mut model, weights(variant)).expect("loads");
    model
}

fn compare(name: &str, got: &Array, want: &Array) -> ParityReport {
    let report =
        ParityReport::compute_with_per_position(name, got, want, Tolerance::new(3e-5, 3e-6));
    print_report_table(std::slice::from_ref(&report));
    report
}

#[test]
fn both_variants_match_transformers() {
    for variant in ["plain", "biased"] {
        let reference = reference(variant);
        let mut model = model(variant);
        let ids = ref_tensor(&reference, "input_ids");
        let mask = ref_tensor(&reference, "attention_mask");

        let embeddings = model.embeddings.forward(ids);
        let masked = model.forward(ids, Some(mask)).expect("forward");
        let unmasked = model.forward(ids, None).expect("unmasked");
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

        for (name, got, want) in [
            ("embeddings", &embeddings, "embeddings"),
            ("last_hidden_state", &masked, "last_hidden_state"),
            ("unmasked", &unmasked, "last_hidden_state_unmasked"),
        ] {
            let report = compare(name, got, ref_tensor(&reference, want));
            assert!(report.passed(), "{variant} {name}: {report:?}");
        }
    }
}

/// The window, the mask and the two rotary bases each have to matter in the
/// fixture for the comparison to be able to fail on them.
#[test]
fn the_fixture_is_sensitive_to_window_mask_and_rotary_base() {
    let reference = reference("plain");
    let ids = ref_tensor(&reference, "input_ids");
    let mask = ref_tensor(&reference, "attention_mask");
    let want = ref_tensor(&reference, "last_hidden_state");
    let gap = |config: ModernBertConfig, masked: bool| {
        let mut model = ModernBertModel::new(config).expect("builds");
        load_modernbert_weights(&mut model, weights("plain")).expect("loads");
        max_abs_diff(&model.forward(ids, masked.then_some(mask)).unwrap(), want)
    };
    // The same weights with every layer global.
    assert!(
        gap(
            ModernBertConfig {
                global_attn_every_n_layers: 1,
                ..config(false)
            },
            true
        ) > 1e-2,
        "the window is not exercised"
    );
    assert!(
        gap(config(false), false) > 1e-2,
        "the padding mask is not exercised"
    );
    assert!(
        gap(
            ModernBertConfig {
                local_rope_theta: Some(1000.0),
                ..config(false)
            },
            true
        ) > 1e-3,
        "the local rotary base is not exercised"
    );
    assert!(
        gap(
            ModernBertConfig {
                global_rope_theta: Some(100.0),
                ..config(false)
            },
            true
        ) > 1e-3,
        "the global rotary base is not exercised"
    );
}

/// A padded row whose window holds only padding gets a uniform attention row,
/// not NaN: the left-padded row's first positions (window 4, padding 5).
#[test]
fn a_window_that_holds_only_padding_stays_finite() {
    let reference = reference("plain");
    let mut model = model("plain");
    let out = model
        .forward(
            ref_tensor(&reference, "input_ids"),
            Some(ref_tensor(&reference, "attention_mask")),
        )
        .expect("forward");
    let flat = pmetal_cache::test_utils::to_f32_vec_eval(&out);
    assert!(flat.iter().all(|v| v.is_finite()));
}

#[test]
fn a_bf16_encoder_stays_in_bf16() {
    let reference = reference("plain");
    let mut model = ModernBertModel::new(config(false)).expect("builds");
    let half = weights("plain")
        .into_iter()
        .map(|(k, v)| (k, v.as_dtype(Dtype::Bfloat16.as_i32())))
        .collect();
    load_modernbert_weights(&mut model, half).expect("loads");
    let out = model
        .forward(
            ref_tensor(&reference, "input_ids"),
            Some(ref_tensor(&reference, "attention_mask")),
        )
        .expect("forward");
    assert_eq!(out.dtype(), Dtype::Bfloat16);
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    let gap = max_abs_diff(&out, ref_tensor(&reference, "last_hidden_state"));
    assert!(gap < 0.5, "bf16 is {gap:.2e} from float32");
}

#[test]
fn a_task_model_checkpoint_loads_its_encoder_and_the_export_loads_back() {
    let mut task: HashMap<String, Array> = weights("plain")
        .into_iter()
        .map(|(k, v)| (format!("model.{k}"), v))
        .collect();
    for head in [
        "head.dense.weight",
        "head.norm.weight",
        "decoder.bias",
        "decoder.weight",
    ] {
        task.insert(head.into(), Array::from_slice(&[0.5f32], &[1]));
    }
    let mut encoder = EncoderModel::from_config_str(
        r#"{"model_type": "modernbert", "vocab_size": 120, "hidden_size": 32,
            "intermediate_size": 48, "num_hidden_layers": 4, "num_attention_heads": 4,
            "local_attention": 8}"#,
    )
    .expect("builds from a config");
    load_encoder_weights(&mut encoder, task).expect("the encoder of a task model loads");

    // Exported under the released names, which load back unchanged.
    let exported: HashMap<String, Array> =
        pmetal_models::loader::encoder_checkpoint_tensors(&encoder)
            .expect("export")
            .into_iter()
            .collect();
    let mut names: Vec<_> = exported.keys().cloned().collect();
    let mut released: Vec<_> = weights("plain").keys().cloned().collect();
    names.sort();
    released.sort();
    assert_eq!(names, released);
    let mut again = EncoderModel::from_config_str(
        r#"{"model_type": "modernbert", "vocab_size": 120, "hidden_size": 32,
            "intermediate_size": 48, "num_hidden_layers": 4, "num_attention_heads": 4,
            "local_attention": 8}"#,
    )
    .unwrap();
    load_encoder_weights(&mut again, exported).expect("the export loads back");
}

#[test]
fn loading_is_strict() {
    let mut extra = weights("plain");
    extra.insert(
        "layers.1.attn.mystery.weight".into(),
        Array::from_slice(&[1.0f32], &[1]),
    );
    let mut m = ModernBertModel::new(config(false)).unwrap();
    let err = load_modernbert_weights(&mut m, extra).expect_err("a stray tensor");
    assert!(err.to_string().contains("mystery"), "{err}");

    let mut short = weights("plain");
    short.remove("layers.2.mlp.Wi.weight");
    let mut m = ModernBertModel::new(config(false)).unwrap();
    let err = load_modernbert_weights(&mut m, short).expect_err("a missing tensor");
    assert!(err.to_string().contains("layers.2.mlp.Wi.weight"), "{err}");

    // A checkpoint with biases into a model without them: refused.
    let mut m = ModernBertModel::new(config(false)).unwrap();
    assert!(load_modernbert_weights(&mut m, weights("biased")).is_err());

    let mut wrong = weights("plain");
    wrong.insert(
        "layers.1.attn.Wqkv.weight".into(),
        Array::from_slice(&vec![0.0f32; 32 * 32], &[32, 32]),
    );
    let mut m = ModernBertModel::new(config(false)).unwrap();
    let err = load_modernbert_weights(&mut m, wrong).expect_err("a wrong shape");
    assert!(matches!(err, LoadError::ShapeMismatch { .. }), "{err}");
}

#[test]
fn rope_settings_read_either_spelling_and_odd_ones_are_refused() {
    // The original spelling, the newer `rope_parameters`, and the released defaults.
    let flat: ModernBertConfig =
        serde_json::from_str(r#"{"global_rope_theta": 5.0, "local_rope_theta": 7.0}"#).unwrap();
    let nested: ModernBertConfig = serde_json::from_str(
        r#"{"rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 5.0},
                                "sliding_attention": {"rope_type": "default", "rope_theta": 7.0}}}"#,
    )
    .unwrap();
    use pmetal_models::architectures::modernbert::ModernBertLayerType::{Global, Local};
    for config in [&flat, &nested] {
        assert_eq!(config.rope_theta(Global).unwrap(), 5.0);
        assert_eq!(config.rope_theta(Local).unwrap(), 7.0);
    }
    let released = ModernBertConfig::default();
    assert_eq!(released.rope_theta(Global).unwrap(), 160_000.0);
    assert_eq!(released.rope_theta(Local).unwrap(), 10_000.0);

    let scaled: ModernBertConfig = serde_json::from_str(
        r#"{"rope_parameters": {"full_attention": {"rope_type": "yarn", "rope_theta": 5.0}}}"#,
    )
    .unwrap();
    assert!(
        scaled
            .rope_theta(Global)
            .unwrap_err()
            .to_string()
            .contains("yarn")
    );

    // Layer types default to every third layer global, and can be given.
    let types = ModernBertConfig {
        num_hidden_layers: 7,
        ..Default::default()
    }
    .layer_types()
    .unwrap();
    assert_eq!(
        types.iter().map(|t| *t == Global).collect::<Vec<_>>(),
        [true, false, false, true, false, false, true]
    );
    let bad = ModernBertConfig {
        num_hidden_layers: 2,
        layer_types: Some(vec!["full_attention".into()]),
        ..Default::default()
    };
    assert!(bad.layer_types().is_err());
}

/// The released `answerdotai/ModernBERT-base` config parses.
#[test]
fn the_released_config_parses() {
    let config: ModernBertConfig = serde_json::from_str(
        r#"{"architectures": ["ModernBertForMaskedLM"], "attention_bias": false,
            "attention_dropout": 0.0, "bos_token_id": 50281, "classifier_activation": "gelu",
            "cls_token_id": 50281, "decoder_bias": true, "deterministic_flash_attn": false,
            "embedding_dropout": 0.0, "eos_token_id": 50282, "global_attn_every_n_layers": 3,
            "global_rope_theta": 160000.0, "gradient_checkpointing": false,
            "hidden_activation": "gelu", "hidden_size": 768, "initializer_cutoff_factor": 2.0,
            "initializer_range": 0.02, "intermediate_size": 1152, "layer_norm_eps": 1e-05,
            "local_attention": 128, "local_rope_theta": 10000.0, "max_position_embeddings": 8192,
            "mlp_bias": false, "mlp_dropout": 0.0, "model_type": "modernbert", "norm_bias": false,
            "norm_eps": 1e-05, "num_attention_heads": 12, "num_hidden_layers": 22,
            "pad_token_id": 50283, "position_embedding_type": "absolute", "sep_token_id": 50282,
            "tie_word_embeddings": true, "vocab_size": 50368}"#,
    )
    .unwrap();
    assert_eq!(config.head_dim(), 64);
    assert_eq!(config.window(), 64);
    assert_eq!(config.layer_types().unwrap().len(), 22);
    ModernBertModel::new(config).expect("a released-size model builds without evaluating");
}
