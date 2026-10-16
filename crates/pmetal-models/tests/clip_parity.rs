//! Numerical parity of the CLIP text encoder against `transformers`.
//!
//! Two tiny seeded towers: the OpenAI layout FLUX uses (`quick_gelu`, the
//! legacy `eos_token_id: 2`, prompts padded with the end-of-text token itself),
//! and a re-trained one (exact GELU, a real `eos_token_id` that smaller and
//! larger ids surround, and a text projection). The two read the sentence
//! embedding from different positions, which the fixtures tell apart.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, max_abs_diff, print_report_table};
use pmetal_models::architectures::clip::{CLIPConfig, CLIPTextModel};
use pmetal_models::loader::{LoadError, load_clip_weights};
use pmetal_runtime::compat::{Array, Dtype};

fn base() -> CLIPConfig {
    CLIPConfig {
        vocab_size: 100,
        embed_dim: 64,
        num_layers: 3,
        num_heads: 4,
        intermediate_size: 128,
        max_position_embeddings: 16,
        layer_norm_eps: 1e-5,
        ..Default::default()
    }
}

fn legacy_config() -> CLIPConfig {
    CLIPConfig {
        hidden_act: "quick_gelu".into(),
        eos_token_id: 2,
        ..base()
    }
}

fn projected_config() -> CLIPConfig {
    CLIPConfig {
        hidden_act: "gelu".into(),
        eos_token_id: 7,
        projection_dim: 24,
        ..base()
    }
}

fn weights(variant: &str) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!(
        "clip_{variant}_weights.safetensors"
    )))
}

fn reference(variant: &str) -> HashMap<String, Array> {
    load_shard(&fixture_path(&format!(
        "clip_{variant}_reference.safetensors"
    )))
}

fn model(variant: &str, config: CLIPConfig) -> CLIPTextModel {
    let mut model = if variant == "projected" {
        CLIPTextModel::with_projection(config)
    } else {
        CLIPTextModel::new(config)
    }
    .expect("model builds");
    load_clip_weights(&mut model, weights(variant)).expect("weights load");
    model
}

fn report(name: &str, got: &Array, want: &Array) -> ParityReport {
    let report =
        ParityReport::compute_with_per_position(name, got, want, Tolerance::new(2e-5, 2e-5));
    print_report_table(std::slice::from_ref(&report));
    report
}

fn check(variant: &str, config: CLIPConfig) {
    let reference = reference(variant);
    let mut model = model(variant, config);
    let out = model
        .forward(ref_tensor(&reference, "input_ids"))
        .expect("forward");
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

    for (name, got) in [
        ("last_hidden_state", &out.last_hidden_state),
        ("penultimate_hidden_state", &out.penultimate_hidden_state),
        ("pooled_output", &out.pooled_output),
    ] {
        let report = report(name, got, ref_tensor(&reference, name));
        assert!(report.passed(), "{variant} {name}: {report:?}");
    }
}

#[test]
fn the_openai_layout_matches_transformers() {
    check("legacy", legacy_config());
}

#[test]
fn a_projected_tower_with_a_real_eos_id_matches_transformers() {
    check("projected", projected_config());
}

/// The sentence embedding is read from the end-of-text position, and which
/// position that is depends on the config. Reading the projected tower's rows
/// the legacy way (position of the largest id) lands on a different token, so
/// the fixture can tell the two rules apart.
#[test]
fn the_two_eos_rules_pick_different_positions() {
    let reference = reference("projected");
    let ids = ref_tensor(&reference, "input_ids");
    let mut as_legacy = CLIPTextModel::with_projection(CLIPConfig {
        eos_token_id: 2,
        ..projected_config()
    })
    .expect("model builds");
    load_clip_weights(&mut as_legacy, weights("projected")).expect("weights load");
    let wrong = as_legacy.forward(ids).expect("forward").pooled_output;
    let gap = max_abs_diff(&wrong, ref_tensor(&reference, "pooled_output"));
    assert!(gap > 1e-2, "both rules give the same embedding ({gap:.2e})");
}

/// A bf16 tower stays bf16: the causal mask is built in the tower's dtype
/// (an f32 mask against bf16 attention is an error in the fused kernel) and
/// the GELU's f32 result is cast back.
#[test]
fn a_bf16_tower_stays_in_bf16() {
    let reference = reference("legacy");
    let mut half = CLIPTextModel::new(legacy_config()).expect("model builds");
    let weights: HashMap<String, Array> = weights("legacy")
        .into_iter()
        .map(|(k, v)| (k, v.as_dtype(Dtype::Bfloat16.as_i32())))
        .collect();
    load_clip_weights(&mut half, weights).expect("weights load");

    let out = half
        .forward(ref_tensor(&reference, "input_ids"))
        .expect("forward");
    assert_eq!(out.last_hidden_state.dtype(), Dtype::Bfloat16);
    assert_eq!(out.pooled_output.dtype(), Dtype::Bfloat16);
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    let report = ParityReport::compute(
        "bf16 pooled",
        &out.pooled_output,
        ref_tensor(&reference, "pooled_output"),
        Tolerance::new(0.1, 0.1),
    );
    assert!(report.passed(), "{report:?}");
}

#[test]
fn a_sequence_longer_than_the_positions_is_an_error() {
    let mut model = model("legacy", legacy_config());
    let ids = Array::from_slice(&[1i32; 17], &[1, 17]);
    let err = model.forward(&ids).expect_err("17 > 16 positions");
    assert!(err.to_string().contains("17"), "{err}");
}

#[test]
fn an_unsupported_activation_is_an_error() {
    let err = CLIPTextModel::new(CLIPConfig {
        hidden_act: "mish".into(),
        ..base()
    })
    .expect_err("mish is not implemented");
    assert!(err.to_string().contains("mish"), "{err}");
}

#[test]
fn a_tensor_no_parameter_takes_and_a_missing_tensor_are_errors() {
    let mut extra = weights("legacy");
    extra.insert(
        "encoder.layers.0.self_attn.mystery.weight".into(),
        Array::from_slice(&[1.0f32], &[1]),
    );
    let mut m = CLIPTextModel::new(legacy_config()).unwrap();
    let err = load_clip_weights(&mut m, extra).expect_err("must fail");
    assert!(err.to_string().contains("mystery"), "{err}");

    let mut short = weights("legacy");
    short.remove("encoder.layers.1.mlp.fc2.bias");
    let mut m = CLIPTextModel::new(legacy_config()).unwrap();
    let err = load_clip_weights(&mut m, short).expect_err("must fail");
    assert!(err.to_string().contains("layers.1.mlp.fc2.bias"), "{err}");
}

/// The position table is `[positions, embed_dim]` as the checkpoint stores it;
/// a table shaped for a different context is refused rather than broadcast.
#[test]
fn a_position_table_of_the_wrong_shape_is_an_error() {
    let mut weights = weights("legacy");
    weights.insert(
        "embeddings.position_embedding.weight".into(),
        Array::from_slice(&vec![0.0f32; 64 * 77], &[77, 64]),
    );
    let mut m = CLIPTextModel::new(legacy_config()).unwrap();
    let err = load_clip_weights(&mut m, weights).expect_err("must fail");
    assert!(matches!(err, LoadError::ShapeMismatch { .. }), "{err}");
}

/// The text half of a full CLIP checkpoint loads: the image tower, its
/// projection and the temperature are named exemptions.
#[test]
fn a_full_clip_checkpoint_loads_its_text_tower() {
    let mut full: HashMap<String, Array> = weights("projected");
    for name in [
        "vision_model.embeddings.class_embedding",
        "visual_projection.weight",
        "logit_scale",
    ] {
        full.insert(name.into(), Array::from_slice(&[1.0f32], &[1]));
    }
    let mut m = CLIPTextModel::with_projection(projected_config()).unwrap();
    load_clip_weights(&mut m, full).expect("the text tower of a full checkpoint loads");

    // ... and one whose projection a plain `CLIPTextModel` does not use.
    let mut plain = CLIPTextModel::new(projected_config()).unwrap();
    load_clip_weights(&mut plain, weights("projected")).expect("an unused projection is exempt");
}

#[test]
fn a_clip_config_parses_the_way_released_configs_spell_it() {
    // The FLUX text encoder's config.json, abridged.
    let config: CLIPConfig = serde_json::from_str(
        r#"{"architectures": ["CLIPTextModel"], "eos_token_id": 2, "hidden_act": "quick_gelu",
            "hidden_size": 768, "intermediate_size": 3072, "layer_norm_eps": 1e-05,
            "max_position_embeddings": 77, "num_attention_heads": 12,
            "num_hidden_layers": 12, "projection_dim": 768, "vocab_size": 49408}"#,
    )
    .unwrap();
    assert_eq!(config.embed_dim, 768);
    assert_eq!(config.num_layers, 12);
    assert_eq!(config.num_heads, 12);
    assert_eq!(config.eos_token_id, 2);
}
