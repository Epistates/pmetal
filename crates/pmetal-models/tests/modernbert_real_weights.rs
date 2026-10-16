//! Real-weight parity of the released ModernBERT-base against `transformers`.
//!
//! `modernbert_base_reference.safetensors` holds `ModernBertModel`'s states at
//! nine positions of two padded rows (262 and 190 tokens, so the 128-token
//! local window and the global layers both matter) and their mean-pooled
//! vectors, computed in float32 over the released weights.
//!
//! Needs the model (~600 MB): set `PMETAL_MODERNBERT_DIR` to an
//! `answerdotai/ModernBERT-base` snapshot. Skipped, loudly, without it.

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, print_report_table};
use pmetal_models::pooling::{PoolingMode, pool};
use pmetal_models::{DynamicModel, ModelArchitecture};
use pmetal_runtime::compat::{Array, Dtype};

fn check(tolerance: Tolerance, min_cosine: f32) {
    let Ok(dir) = std::env::var("PMETAL_MODERNBERT_DIR") else {
        eprintln!("SKIPPED: set PMETAL_MODERNBERT_DIR to an answerdotai/ModernBERT-base snapshot");
        return;
    };
    let dir = std::path::Path::new(&dir);
    assert_eq!(
        ModelArchitecture::detect(dir).expect("detects"),
        ModelArchitecture::Bert
    );
    let mut model = DynamicModel::load(dir).expect("the released encoder loads strictly");
    let reference = load_shard(&fixture_path("modernbert_base_reference.safetensors"));
    let ids = ref_tensor(&reference, "input_ids");
    let mask = ref_tensor(&reference, "attention_mask");

    let hidden = model.forward_hidden(ids, Some(mask)).expect("forward");
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

    let positions: Vec<i32> = pmetal_cache::test_utils::to_f32_vec_eval(
        &ref_tensor(&reference, "positions").as_dtype(Dtype::Float32.as_i32()),
    )
    .into_iter()
    .map(|p| p as i32)
    .collect();
    let rows: Vec<Array> = positions
        .iter()
        .map(|&p| pmetal_runtime::compat::ops::slice_axis(&hidden, 1, p, p + 1))
        .collect();
    let states = pmetal_runtime::compat::ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 1);
    let pooled = pool(&hidden, mask, PoolingMode::Mean).expect("pool");

    let reports = [
        ParityReport::compute(
            "states",
            &states,
            ref_tensor(&reference, "states"),
            tolerance,
        ),
        ParityReport::compute(
            "mean_pooled",
            &pooled,
            ref_tensor(&reference, "mean_pooled"),
            tolerance,
        ),
    ];
    print_report_table(&reports);
    for report in &reports {
        assert!(report.passed(), "{}: {report:?}", report.name);
        assert!(report.cosine_similarity > min_cosine, "{}", report.name);
    }
}

#[test]
fn the_released_modernbert_base_matches_transformers_in_float32() {
    check(Tolerance::new(2e-3, 2e-4), 0.99999);
}
