//! The Cohere variants `cohere_parity` does not cover, loaded through
//! `DynamicModel::load` from a `config.json` as `transformers` writes it, held
//! to `transformers`' logits.
//!
//! * `cohere_qknorm`: Command R+'s `use_qk_norm`, a per-head LayerNorm
//!   (`[heads, head_dim]` weights, mean subtracted, no bias) on the queries and
//!   keys before RoPE. A model without it loads every other tensor and
//!   ignores these.
//! * `cohere2_synth`: Command R7B / Command A (`model_type = "cohere2"`). Three
//!   of four layers attend within a 4-token window and rotate their queries and
//!   keys; the fourth attends globally with no RoPE at all. The prompt is
//!   longer than the window, so both matter.
//!
//! Each is checked uncached and as a cached prefill followed by one-token
//! steps (which run the sliding window against a cache), and the Cohere2
//! fixture is also run as a plain Cohere model to show the fixture can tell
//! the two apart.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, argmax_last_axis, print_report_table};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{Array, ops::slice_axis};
use serial_test::serial;

/// fp32 against fp32; the fixtures' own fp32-vs-fp64 noise is under 1e-7.
const TOL: Tolerance = Tolerance::new(1e-5, 1e-5);

fn drain(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: runtime error: {e}"));
}

struct Fixture {
    _dir: tempfile::TempDir,
    model: DynamicModel,
    reference: HashMap<String, Array>,
}

fn checkpoint(name: &str, config_edit: impl Fn(String) -> String) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let config =
        std::fs::read_to_string(fixture_path(&format!("{name}_config.json"))).expect("read config");
    std::fs::write(dir.path().join("config.json"), config_edit(config)).expect("write config");
    std::fs::copy(
        fixture_path(&format!("{name}_weights.safetensors")),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    let model = DynamicModel::load(dir.path()).expect("the checkpoint loads");
    drain("load");
    Fixture {
        _dir: dir,
        model,
        reference: load_shard(&fixture_path(&format!("{name}_reference.safetensors"))),
    }
}

fn assert_all_pass(title: &str, reports: &[ParityReport]) {
    println!("\n== {title} ==");
    print_report_table(reports);
    let failed: Vec<&str> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.as_str())
        .collect();
    assert!(failed.is_empty(), "{title}: {failed:?} out of tolerance");
}

fn check(name: &str) {
    let mut fx = checkpoint(name, |c| c);
    let ids = ref_tensor(&fx.reference, "input_ids").clone();
    let want = ref_tensor(&fx.reference, "logits").clone();
    let t = ids.dim(1);

    let logits = fx.model.forward(&ids, None).expect("forward");
    drain("forward");
    assert_all_pass(
        &format!("{name}: uncached"),
        &[ParityReport::compute_with_per_position(
            "logits", &logits, &want, TOL,
        )],
    );
    assert_eq!(argmax_last_axis(&logits), argmax_last_axis(&want));

    // A prefill, then one token at a time: the window against a cache.
    let mut cache = fx.model.create_cache(t as usize + 4);
    let mut steps = vec![6];
    steps.extend(std::iter::repeat_n(1, (t - 6) as usize));
    let mut start = 0;
    let mut reports = Vec::new();
    for len in steps {
        let chunk = slice_axis(&ids, 1, start, start + len);
        let logits = fx
            .model
            .forward_with_cache(&chunk, None, Some(&mut cache))
            .expect("cached forward");
        drain("cached forward");
        reports.push(ParityReport::compute(
            &format!("pos_{start}"),
            &logits,
            &slice_axis(&want, 1, start, start + len),
            TOL,
        ));
        start += len;
    }
    assert_all_pass(&format!("{name}: cached"), &reports);
}

#[test]
#[serial]
fn command_r_plus_query_key_norm_matches_transformers() {
    check("cohere_qknorm");
}

#[test]
#[serial]
fn cohere2_sliding_and_global_layers_match_transformers() {
    check("cohere2_synth");
}

/// Dropping what each fixture exists to test moves the logits well past the
/// tolerance, so a pass above is not a pass by accident.
#[test]
#[serial]
fn the_fixtures_can_tell_the_variants_apart() {
    // Cohere2 read as Cohere: global attention and RoPE on every layer.
    let mut fx = checkpoint("cohere2_synth", |c| c.replace("\"cohere2\"", "\"cohere\""));
    let ids = ref_tensor(&fx.reference, "input_ids").clone();
    let want = ref_tensor(&fx.reference, "logits").clone();
    let logits = fx.model.forward(&ids, None).expect("forward");
    drain("forward");
    assert!(
        !ParityReport::compute("logits", &logits, &want, TOL).passed(),
        "a Cohere2 checkpoint run as Cohere must not match"
    );
}

/// A Command R+ checkpoint read by a model without query/key norm has tensors
/// no parameter consumes, which the load refuses rather than ignoring.
#[test]
#[serial]
fn a_model_without_query_key_norm_refuses_its_tensors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = std::fs::read_to_string(fixture_path("cohere_qknorm_config.json")).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        config.replace("\"use_qk_norm\": true", "\"use_qk_norm\": false"),
    )
    .unwrap();
    std::fs::copy(
        fixture_path("cohere_qknorm_weights.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .unwrap();
    let err = DynamicModel::load(dir.path()).expect_err("the load is refused");
    assert!(err.to_string().contains("q_norm"), "{err}");
}
