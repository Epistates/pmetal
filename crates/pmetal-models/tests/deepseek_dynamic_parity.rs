//! The shared-path DeepSeek (`DynamicModel::DeepSeek`, what training and the
//! adapters run) against the authoritative Hugging Face `transformers`
//! `DeepseekV3ForCausalLM` / `DeepseekV2ForCausalLM`.
//!
//! The fixtures are the three-layer checkpoints the native engine's parity test
//! uses, in the layout the releases ship (per-expert `experts.E.*_proj.weight`,
//! `mlp.gate.e_score_correction_bias`, a dense first layer, a trailing
//! multi-token-prediction layer). Loaded through `DynamicModel::load`, the
//! production path, so a router bias the loader drops is in scope.
//!
//! * `v3`: sigmoid scoring, bias-corrected group-limited top-k, normalised and
//!   scaled weights. The bias is random, so a model that runs with a zero bias
//!   picks different experts.
//! * `v2_greedy`: softmax scoring with `n_group` / `topk_group` set in the
//!   config but `topk_method = greedy`: the plain top-k, no group limit.
//! * `v2_group_limited_greedy`: the top-k among the groups whose best expert
//!   scores highest.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, argmax_last_axis, print_report_table};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{Array, ops::slice_axis};
use serial_test::serial;

/// fp32 against fp32; the reference's own fp32-vs-fp64 noise is under 1e-6.
/// A zero router bias, a missed group or a swapped score moves the logits by
/// 1e-2.
const TOL: Tolerance = Tolerance::new(5e-5, 0.0);

struct Fixture {
    _dir: tempfile::TempDir,
    model: DynamicModel,
    reference: HashMap<String, Array>,
}

fn checkpoint(variant: &str, weights_file: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path(&format!("deepseek_native_{variant}_config.json")),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path(weights_file),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    let model = DynamicModel::load(dir.path()).expect("the checkpoint loads");
    drain("load");
    Fixture {
        _dir: dir,
        model,
        reference: load_shard(&fixture_path(&format!(
            "deepseek_native_{variant}_reference.safetensors"
        ))),
    }
}

fn drain(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: runtime error: {e}"));
}

fn check(variant: &str, weights_file: &str) {
    let mut fx = checkpoint(variant, weights_file);
    let ids = ref_tensor(&fx.reference, "input_ids").clone();
    let want = ref_tensor(&fx.reference, "logits").clone();
    let t = ids.dim(1);

    let logits = fx.model.forward(&ids, None).expect("forward");
    drain("forward");
    let report = ParityReport::compute_with_per_position("logits", &logits, &want, TOL);
    println!("\n== {variant}: uncached prefill ==");
    print_report_table(std::slice::from_ref(&report));
    assert!(
        report.passed(),
        "{variant}: prefill logits out of tolerance"
    );
    assert_eq!(argmax_last_axis(&logits), argmax_last_axis(&want));

    // The same prompt through the KV cache, 3 tokens and then one at a time.
    let mut cache = fx.model.create_cache(t as usize + 4);
    let mut steps = vec![3];
    steps.extend(std::iter::repeat_n(1, (t - 3) as usize));
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
    println!("\n== {variant}: cached decode ==");
    print_report_table(&reports);
    let failed: Vec<&str> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.as_str())
        .collect();
    assert!(failed.is_empty(), "{variant}: cached {failed:?}");
}

#[test]
#[serial]
fn dynamic_deepseek_v3_matches_transformers() {
    check("v3", "deepseek_native_v3_weights.safetensors");
}

#[test]
#[serial]
fn dynamic_deepseek_v2_greedy_matches_transformers() {
    check("v2_greedy", "deepseek_native_v2_weights.safetensors");
}

#[test]
#[serial]
fn dynamic_deepseek_v2_group_limited_matches_transformers() {
    check(
        "v2_group_limited_greedy",
        "deepseek_native_v2_weights.safetensors",
    );
}

/// The parameter tree is the checkpoint's own layout with each layer's experts
/// stacked (what an MLX conversion stores), so a saved model is a checkpoint:
/// every parameter has a tensor in the stacked fixture, and every fixture
/// tensor of the trunk has a parameter. The router bias is a parameter
/// (loaded, saved) the optimizer is never handed.
#[test]
#[serial]
fn dynamic_deepseek_parameters_are_the_checkpoints_tensors() {
    use pmetal_runtime::compat::ModuleParametersExt;

    let fx = checkpoint("v3", "deepseek_native_v3_weights.safetensors");
    let shard = pmetal_models::loader::experts::stack_experts(
        "deepseek",
        load_shard(&fixture_path("deepseek_native_v3_weights.safetensors")),
    )
    .expect("the experts stack");
    let params = fx.model.flatten_params();
    let trainable = fx.model.flatten_trainable_params();

    let missing: Vec<_> = params
        .keys()
        .filter(|k| !shard.contains_key(&***k))
        .collect();
    assert!(missing.is_empty(), "parameters with no tensor: {missing:?}");

    // The fixture ends in a multi-token-prediction layer, past the trunk.
    let unread: Vec<_> = shard
        .keys()
        .filter(|k| !k.starts_with("model.layers.3.") && !params.contains_key(k.as_str()))
        .collect();
    assert!(unread.is_empty(), "tensors no parameter reads: {unread:?}");

    let bias = "model.layers.1.mlp.gate.e_score_correction_bias";
    assert!(params.contains_key(bias), "the router bias is a parameter");
    assert!(
        !trainable.contains_key(bias),
        "the router bias is not trainable"
    );
    // The router picks experts by argmax; a step on it reroutes tokens the
    // experts were not tuned for, so no optimizer is handed it.
    assert!(params.contains_key("model.layers.1.mlp.gate.weight"));
    assert!(!trainable.contains_key("model.layers.1.mlp.gate.weight"));
}

/// An MLX conversion stores each layer's experts stacked (`mlp.switch_mlp`):
/// the same weights written that way load into the same model and give the
/// reference's logits, with no stacking left to do at load.
#[test]
#[serial]
fn dynamic_deepseek_loads_an_mlx_stacked_checkpoint() {
    let stacked = pmetal_models::loader::experts::stack_experts(
        "deepseek",
        load_shard(&fixture_path("deepseek_native_v3_weights.safetensors")),
    )
    .expect("the experts stack");
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path("deepseek_native_v3_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    let entries: Vec<(&str, &Array)> = stacked.iter().map(|(k, v)| (k.as_str(), v)).collect();
    assert!(entries.iter().any(|(k, _)| k.contains(".switch_mlp.")));
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    )
    .expect("save the stacked checkpoint");
    drain("save");

    let mut model = DynamicModel::load(dir.path()).expect("the stacked checkpoint loads");
    drain("load");
    let reference = load_shard(&fixture_path("deepseek_native_v3_reference.safetensors"));
    let ids = ref_tensor(&reference, "input_ids").clone();
    let logits = model.forward(&ids, None).expect("forward");
    drain("forward");
    let report = ParityReport::compute_with_per_position(
        "logits",
        &logits,
        ref_tensor(&reference, "logits"),
        TOL,
    );
    print_report_table(std::slice::from_ref(&report));
    assert!(report.passed());
}
