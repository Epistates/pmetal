//! Qwen2 and Qwen3 with a sliding window, loaded through `DynamicModel::load`
//! from a `config.json` as `transformers` writes it, held to its logits.
//!
//! Both fixtures have three layers, the first attending globally and the other
//! two within a 4-token window, over a 12-token prompt. The reference decides
//! which layers are local from `layer_types` (`"sliding_attention"`), which
//! Qwen2's config derives from `max_window_layers`; and only when
//! `use_sliding_window` is set. The Qwen2 fixture's attention projections
//! carry random biases.
//!
//! Each is checked uncached and as a cached prefill followed by one-token
//! steps, and the same weights are run with the window turned off to show the
//! fixture can tell a window from none.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, argmax_last_axis, print_report_table};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{Array, ops::slice_axis};
use serial_test::serial;

/// fp32 against fp32; the fixtures' own fp32-vs-fp64 noise is under 1e-6.
const TOL: Tolerance = Tolerance::new(2e-5, 2e-5);

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
        std::fs::read_to_string(fixture_path(&format!("{name}_config.json"))).expect("config");
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

    // The window matters: without it the logits leave the tolerance.
    let mut open = checkpoint(name, |c| {
        c.replace(
            "\"use_sliding_window\": true",
            "\"use_sliding_window\": false",
        )
    });
    let logits = open.model.forward(&ids, None).expect("forward");
    drain("forward without a window");
    assert!(
        !ParityReport::compute("logits", &logits, &want, TOL).passed(),
        "{name}: the fixture cannot tell a window from none"
    );
}

#[test]
#[serial]
fn qwen2_sliding_window_layers_match_transformers() {
    check("qwen2_sliding");
}

#[test]
#[serial]
fn qwen3_sliding_window_layers_match_transformers() {
    check("qwen3_sliding");
}
