//! Real-weight parity of a released T5 checkpoint against `transformers`.
//!
//! `flan_t5_small_reference.safetensors` holds `T5EncoderModel`'s states for two
//! padded token sequences, computed in float32 over the released weights of
//! `google/flan-t5-small` (gated GELU, a *full* `T5ForConditionalGeneration`
//! checkpoint: `encoder.embed_tokens`, a decoder and an `lm_head` sit beside
//! the encoder). This is the layout T5-XXL for FLUX is cut from.
//!
//! Needs the model (~300 MB): set `PMETAL_T5_DIR` to a `flan-t5-small`
//! snapshot. Skipped, loudly, without it.

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, print_report_table};
use pmetal_models::architectures::t5::{T5Config, T5EncoderModel};
use pmetal_models::loader::{load_t5_weights, load_weights};
use pmetal_runtime::compat::Dtype;

fn check(dtype: Dtype, tolerance: Tolerance, min_cosine: f32) {
    let Ok(dir) = std::env::var("PMETAL_T5_DIR") else {
        eprintln!("SKIPPED: set PMETAL_T5_DIR to a google/flan-t5-small snapshot");
        return;
    };
    let raw = std::fs::read_to_string(std::path::Path::new(&dir).join("config.json"))
        .expect("the model has a config.json");
    let config: T5Config = serde_json::from_str(&raw).expect("config.json parses");
    assert!(config.is_gated_act());
    let mut model = T5EncoderModel::new(config).expect("the encoder builds");
    let weights = load_weights(&dir)
        .expect("the weights read")
        .into_iter()
        .map(|(k, v)| (k, v.as_dtype(dtype.as_i32())))
        .collect();
    load_t5_weights(&mut model, weights).expect("the encoder of a full T5 checkpoint loads");

    let reference = load_shard(&fixture_path("flan_t5_small_reference.safetensors"));
    let out = model
        .forward(
            ref_tensor(&reference, "input_ids"),
            Some(ref_tensor(&reference, "attention_mask")),
        )
        .expect("forward");
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

    // Padded positions of the second row attend to real keys only and carry
    // meaningful states, so every position is compared.
    let report = ParityReport::compute_with_per_position(
        "last_hidden_state",
        &out,
        ref_tensor(&reference, "last_hidden_state"),
        tolerance,
    );
    print_report_table(std::slice::from_ref(&report));
    assert!(report.passed(), "{dtype:?}: {report:?}");
    assert!(
        report.cosine_similarity > min_cosine,
        "{dtype:?}: {report:?}"
    );
}

#[test]
fn the_released_flan_t5_small_matches_transformers_in_float32() {
    check(Dtype::Float32, Tolerance::new(1e-4, 1e-4), 0.99999);
}

/// Eight bf16 layers move the states by a few percent: the reference run in
/// bfloat16 lands 0.079 (cosine 0.9978) from its float32 one on these inputs,
/// pmetal 0.061 (cosine 0.9989).
#[test]
fn the_released_flan_t5_small_matches_transformers_in_bfloat16() {
    check(Dtype::Bfloat16, Tolerance::new(0.1, 0.0), 0.997);
}
