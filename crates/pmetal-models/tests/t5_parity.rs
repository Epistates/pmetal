//! Numerical parity of the T5 encoder against `transformers`' `T5EncoderModel`.
//!
//! The fixtures are two tiny seeded encoders, one per feed-forward family:
//! gated GELU (v1.1 / Flan, the encoder FLUX conditions on) and the original
//! ReLU. Sequences run to 40 tokens so the relative-position buckets cover the
//! logarithmic range, not just the exact one, and the second row of the batch
//! is right-padded.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, print_report_table};
use pmetal_models::architectures::t5::{T5Config, T5EncoderModel};
use pmetal_models::loader::{LoadError, load_t5_weights};
use pmetal_runtime::compat::{Array, Dtype};

fn config(feed_forward_proj: &str) -> T5Config {
    T5Config {
        vocab_size: 200,
        d_model: 64,
        d_kv: 16,
        d_ff: 128,
        num_layers: 3,
        num_heads: 4,
        relative_attention_num_buckets: 32,
        relative_attention_max_distance: 128,
        layer_norm_epsilon: 1e-6,
        feed_forward_proj: feed_forward_proj.to_string(),
    }
}

fn weights(variant: &str) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!("t5_{variant}_weights.safetensors")))
}

fn model(variant: &str, feed_forward_proj: &str) -> T5EncoderModel {
    let mut model = T5EncoderModel::new(config(feed_forward_proj)).expect("model builds");
    load_t5_weights(&mut model, weights(variant)).expect("weights load");
    model
}

fn reference(variant: &str) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!("t5_{variant}_reference.safetensors")))
}

fn compare(name: &str, got: &Array, want: &Array, tolerance: f32) -> ParityReport {
    let report = ParityReport::compute_with_per_position(
        name,
        got,
        want,
        Tolerance::new(tolerance, tolerance),
    );
    print_report_table(std::slice::from_ref(&report));
    report
}

fn check(variant: &str, feed_forward_proj: &str) {
    let reference = reference(variant);
    let mut model = model(variant, feed_forward_proj);
    let ids = ref_tensor(&reference, "input_ids");
    let mask = ref_tensor(&reference, "attention_mask");

    let unmasked = model.forward(ids, None).expect("forward");
    let masked = model.forward(ids, Some(mask)).expect("masked forward");
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

    let a = compare(
        "last_hidden_state_nomask",
        &unmasked,
        ref_tensor(&reference, "last_hidden_state_nomask"),
        2e-5,
    );
    let b = compare(
        "last_hidden_state",
        &masked,
        ref_tensor(&reference, "last_hidden_state"),
        2e-5,
    );
    assert!(a.passed(), "{variant} without a mask: {a:?}");
    assert!(b.passed(), "{variant} with a padding mask: {b:?}");
}

#[test]
fn gated_gelu_encoder_matches_transformers() {
    check("gated_gelu", "gated-gelu");
}

#[test]
fn relu_encoder_matches_transformers() {
    check("relu", "relu");
}

/// The fixture only proves anything if the padding mask changes the output and
/// the unscaled attention is not a no-op: a model that ignored the mask, or
/// one that scaled its scores, must land outside the tolerance.
#[test]
fn the_fixtures_are_sensitive_to_the_mask() {
    let reference = reference("gated_gelu");
    let masked = ref_tensor(&reference, "last_hidden_state");
    let unmasked = ref_tensor(&reference, "last_hidden_state_nomask");
    let gap = pmetal_cache::test_utils::max_abs_diff(masked, unmasked);
    assert!(gap > 1e-2, "the mask moves the output by only {gap:.2e}");
}

/// A bf16 checkpoint runs in bf16 and stays close to the f32 answer: nothing
/// in the encoder promotes the stream to f32 (a GELU, a scalar), which would
/// also double the memory traffic of the largest text encoder FLUX loads.
#[test]
fn a_bf16_encoder_stays_in_bf16() {
    let reference = reference("gated_gelu");
    let ids = ref_tensor(&reference, "input_ids");
    let mask = ref_tensor(&reference, "attention_mask");
    let want = ref_tensor(&reference, "last_hidden_state");

    let mut half = T5EncoderModel::new(config("gated-gelu")).expect("model builds");
    let to_bf16: HashMap<String, Array> = weights("gated_gelu")
        .into_iter()
        .map(|(k, v)| (k, v.as_dtype(Dtype::Bfloat16.as_i32())))
        .collect();
    load_t5_weights(&mut half, to_bf16).expect("weights load");

    let out = half.forward(ids, Some(mask)).expect("forward");
    assert_eq!(out.dtype(), Dtype::Bfloat16);
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    let report = compare("bf16", &out, want, 0.25);
    assert!(report.passed(), "{report:?}");
}

#[test]
fn a_tensor_the_encoder_has_no_parameter_for_is_an_error() {
    let mut weights = weights("gated_gelu");
    weights.insert(
        "encoder.block.0.layer.0.SelfAttention.mystery.weight".to_string(),
        Array::from_slice(&[1.0f32], &[1]),
    );
    let mut model = T5EncoderModel::new(config("gated-gelu")).expect("model builds");
    let err = load_t5_weights(&mut model, weights).expect_err("must fail");
    assert!(err.to_string().contains("mystery"), "{err}");
}

#[test]
fn a_parameter_the_checkpoint_never_fills_is_an_error() {
    let mut weights = weights("gated_gelu");
    weights.remove("encoder.block.2.layer.1.DenseReluDense.wo.weight");
    let mut model = T5EncoderModel::new(config("gated-gelu")).expect("model builds");
    let err = load_t5_weights(&mut model, weights).expect_err("must fail");
    assert!(
        err.to_string().contains("blocks.2.layer_1_mlp.wo.weight"),
        "{err}"
    );
}

/// A gated checkpoint loaded into a model built for a plain feed-forward (a
/// config that says `relu` for weights that are `gated-gelu`) is refused for
/// the tensors it has no home for, not run on a half-random model.
#[test]
fn a_checkpoint_of_the_wrong_feed_forward_family_is_refused() {
    let mut model = T5EncoderModel::new(config("relu")).expect("model builds");
    let err = load_t5_weights(&mut model, weights("gated_gelu")).expect_err("must fail");
    assert!(err.to_string().contains("wi_0"), "{err}");
}

#[test]
fn a_tensor_of_the_wrong_shape_is_an_error() {
    let mut weights = weights("gated_gelu");
    weights.insert(
        "encoder.block.1.layer.0.SelfAttention.q.weight".to_string(),
        Array::from_slice(&vec![0.0f32; 64 * 32], &[32, 64]),
    );
    let mut model = T5EncoderModel::new(config("gated-gelu")).expect("model builds");
    let err = load_t5_weights(&mut model, weights).expect_err("must fail");
    assert!(matches!(err, LoadError::ShapeMismatch { .. }), "{err}");
}

/// A full T5 checkpoint carries `encoder.embed_tokens` (tied to `shared`), a
/// decoder and a head; the encoder takes its own tensors and leaves the rest.
#[test]
fn a_full_t5_checkpoint_loads_its_encoder() {
    let mut weights = weights("gated_gelu");
    weights.insert(
        "encoder.embed_tokens.weight".to_string(),
        weights["shared.weight"].clone(),
    );
    weights.insert(
        "decoder.block.0.layer.0.SelfAttention.q.weight".to_string(),
        Array::from_slice(&[1.0f32], &[1]),
    );
    weights.insert(
        "lm_head.weight".to_string(),
        Array::from_slice(&[1.0f32], &[1]),
    );
    let mut model = T5EncoderModel::new(config("gated-gelu")).expect("model builds");
    load_t5_weights(&mut model, weights).expect("an encoder inside a full checkpoint loads");
}

/// A checkpoint that carries only `encoder.embed_tokens` still fills `shared`.
#[test]
fn the_encoder_embedding_alone_fills_the_shared_table() {
    let mut weights = weights("gated_gelu");
    let table = weights.remove("shared.weight").expect("fixture has shared");
    weights.insert("encoder.embed_tokens.weight".to_string(), table);
    let mut model = T5EncoderModel::new(config("gated-gelu")).expect("model builds");
    load_t5_weights(&mut model, weights).expect("loads");
}

#[test]
fn a_t5_config_parses_the_way_released_configs_spell_it() {
    // google/t5-v1_1-xxl as FLUX ships it; extra keys are ignored.
    let config: T5Config = serde_json::from_str(
        r#"{"d_ff": 10240, "d_kv": 64, "d_model": 4096, "dense_act_fn": "gelu_new",
            "feed_forward_proj": "gated-gelu", "is_gated_act": true,
            "layer_norm_epsilon": 1e-06, "num_heads": 64, "num_layers": 24,
            "relative_attention_max_distance": 128,
            "relative_attention_num_buckets": 32, "vocab_size": 32128}"#,
    )
    .unwrap();
    assert!(config.is_gated_act());
    assert_eq!(config.inner_dim(), 4096);
    assert_eq!(config.dense_act_fn(), "gelu_new");
    // The original T5 names neither the feed-forward nor the norm epsilon.
    let small: T5Config = serde_json::from_str(r#"{"num_hidden_layers": 6}"#).unwrap();
    assert!(!small.is_gated_act());
    assert_eq!(small.num_layers, 6);
    assert_eq!(small.dense_act_fn(), "relu");
}
