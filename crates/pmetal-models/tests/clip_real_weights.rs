//! Real-weight parity of FLUX.1's CLIP-L text encoder against `transformers`.
//!
//! `clip_l_real_reference.safetensors` holds `CLIPTextModel`'s states and pooled
//! vector for two short prompts (end-of-text padded), computed in float32 over
//! the released weights. The released tensors are bfloat16, so this loads them,
//! widens them to float32 and compares at float32 accumulation order; a
//! bfloat16 run is held to bf16 noise.
//!
//! Needs the encoder (~250 MB): set `PMETAL_FLUX_CLIP_DIR` to the
//! `text_encoder` directory of a FLUX.1 snapshot. Skipped, loudly, without it.

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, print_report_table};
use pmetal_models::architectures::clip::{CLIPConfig, CLIPTextModel};
use pmetal_models::loader::{load_clip_weights, load_weights};
use pmetal_runtime::compat::Dtype;

fn check(dtype: Dtype, tolerance: Tolerance) {
    let Ok(dir) = std::env::var("PMETAL_FLUX_CLIP_DIR") else {
        eprintln!("SKIPPED: set PMETAL_FLUX_CLIP_DIR to a FLUX.1 `text_encoder` directory");
        return;
    };
    let raw = std::fs::read_to_string(std::path::Path::new(&dir).join("config.json"))
        .expect("the encoder has a config.json");
    let config: CLIPConfig = serde_json::from_str(&raw).expect("config.json parses");
    let mut model = CLIPTextModel::new(config).expect("the encoder builds");
    let weights = load_weights(&dir)
        .expect("the encoder's weights read")
        .into_iter()
        .map(|(k, v)| (k, v.as_dtype(dtype.as_i32())))
        .collect();
    load_clip_weights(&mut model, weights).expect("the released encoder loads strictly");

    let reference = load_shard(&fixture_path("clip_l_real_reference.safetensors"));
    let out = model
        .forward(ref_tensor(&reference, "input_ids"))
        .expect("forward");
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

    let reports = [
        ParityReport::compute(
            "last_hidden_state",
            &out.last_hidden_state,
            ref_tensor(&reference, "last_hidden_state"),
            tolerance,
        ),
        ParityReport::compute(
            "penultimate_hidden_state",
            &out.penultimate_hidden_state,
            ref_tensor(&reference, "penultimate_hidden_state"),
            tolerance,
        ),
        ParityReport::compute(
            "pooled_output",
            &out.pooled_output,
            ref_tensor(&reference, "pooled_output"),
            tolerance,
        ),
    ];
    print_report_table(&reports);
    for report in &reports {
        assert!(report.passed(), "{dtype:?} {}: {report:?}", report.name);
        assert!(
            report.cosine_similarity > 0.999,
            "{dtype:?} {}",
            report.name
        );
    }
}

#[test]
fn the_released_clip_l_matches_transformers_in_float32() {
    check(Dtype::Float32, Tolerance::new(2e-4, 1e-5));
}

#[test]
fn the_released_clip_l_matches_transformers_in_bfloat16() {
    check(Dtype::Bfloat16, Tolerance::new(0.5, 0.0));
}
