//! Numerical parity of the native engine on Qwen3-MoE (`model_type =
//! "qwen3_moe"`, Qwen3-30B-A3B and its family) against the authoritative
//! Hugging Face `transformers` `Qwen3MoeForCausalLM`.
//!
//! Each fixture is a whole flat checkpoint from transformers 5.19 (torch
//! 2.14.1, seed 7): `config.json`, an fp32 `model.safetensors` in the layout
//! transformers saves (experts fused as `gate_up_proj` and `down_proj`), and
//! the reference activations over a 40-token prompt. Two profiles:
//!
//! * `qwen3_moe`: 3 sparse layers of 8 experts, 2 per token, top-k weights
//!   renormalized (`norm_topk_prob`), no shared expert.
//! * `qwen3_moe_unnormalized`: `norm_topk_prob` false, and layer 1 a dense MLP
//!   (`mlp_only_layers`).
//!
//! Every profile is checked on an uncached prefill, on a cached decode that
//! prefills part of the prompt and feeds the rest one token at a time, and on
//! the prompt prefilled in passes of several sizes.

mod common;

use std::collections::HashMap;
use std::path::Path;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_mlx::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, print_report_table, to_f32_vec_eval,
};
use pmetal_runtime::compat::{Array, ops::slice_axis};
use pmetal_runtime::decode::{prefill_last_logits, prompt_tokens_to_input};
use pmetal_runtime::qwen3_native::{
    NativeCache, forward_step, forward_step_last, load_config, load_model,
};
use serial_test::serial;

/// Tokens fed as one prefill before the cached decode takes over.
const PREFILL: i32 = 24;

/// fp32 against fp32; the fixtures' own fp32-vs-fp64 noise is 1.5e-6 to 1.7e-6
/// (recorded in their meta), and the gate is ~30x that. A router that renormalizes
/// when it should not, or the reverse, moves the logits by 1e-2.
const TOL: Tolerance = Tolerance::new(5e-5, 0.0);

struct Fixture {
    _dir: tempfile::TempDir,
    reference: HashMap<String, Array>,
}

impl Fixture {
    fn path(&self) -> &Path {
        self._dir.path()
    }

    fn tensor(&self, key: &str) -> Array {
        ref_tensor(&self.reference, key).clone()
    }

    fn ids(&self) -> Array {
        self.tensor("input_ids")
    }
}

fn checkpoint(profile: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path(&format!("{profile}_config.json")),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path(&format!("{profile}_weights.safetensors")),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    Fixture {
        _dir: dir,
        reference: load_shard(&fixture_path(&format!("{profile}_reference.safetensors"))),
    }
}

fn rows(a: &Array, start: i32, end: i32) -> Array {
    slice_axis(a, 1, start, end)
}

fn drain(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
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

fn native_matches_transformers(profile: &str) {
    let fx = checkpoint(profile);
    let config = load_config(fx.path()).expect("native config parses");
    assert!(config.is_moe(), "{profile} is a routed model");
    let weights = load_model(fx.path(), &config).expect("native weights load");
    drain("load");
    let ids = fx.ids();
    let t = ids.dim(1);
    let want = fx.tensor("logits");

    // Uncached prefill over the whole prompt.
    let mut cache = NativeCache::new_empty(&weights);
    let logits = forward_step(&weights, &ids, &mut cache);
    drain("prefill");
    assert_all_pass(
        &format!("{profile}: prefill"),
        &[ParityReport::compute_with_per_position(
            "logits", &logits, &want, TOL,
        )],
    );
    assert_eq!(argmax_last_axis(&logits), argmax_last_axis(&want));

    // Part of the prompt, then one token at a time (the single-token MoE layer).
    let mut cache = NativeCache::new_empty(&weights);
    let prefill = forward_step(&weights, &rows(&ids, 0, PREFILL), &mut cache);
    drain("cached prefill");
    let mut reports = vec![ParityReport::compute(
        "prefill_logits",
        &prefill,
        &rows(&want, 0, PREFILL),
        TOL,
    )];
    for pos in PREFILL..t {
        let step = forward_step(&weights, &rows(&ids, pos, pos + 1), &mut cache);
        drain("decode step");
        reports.push(ParityReport::compute(
            &format!("step_{pos}"),
            &step,
            &rows(&want, pos, pos + 1),
            TOL,
        ));
    }
    assert_all_pass(&format!("{profile}: cached decode"), &reports);

    // The prompt in passes, then the last token as a decode step.
    let ids_vec: Vec<u32> = to_f32_vec_eval(&ids)
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let last = rows(&want, t - 1, t);
    let mut reports = Vec::new();
    for chunk in [1, 5, 16, 39, 40, 1000] {
        let mut cache = NativeCache::new_empty(&weights);
        let logits = prefill_last_logits(&weights, &mut cache, &ids_vec, chunk, forward_step_last);
        drain("chunked prefill");
        reports.push(ParityReport::compute(
            &format!("chunk_{chunk}_last_logits"),
            &logits.reshape(&[1, 1, -1]),
            &last,
            TOL,
        ));
        let mut cache = NativeCache::new_empty(&weights);
        let head = &ids_vec[..ids_vec.len() - 1];
        prefill_last_logits(&weights, &mut cache, head, chunk, forward_step_last).eval();
        let step = forward_step_last(
            &weights,
            &prompt_tokens_to_input(&ids_vec[ids_vec.len() - 1..]),
            &mut cache,
        );
        drain("decode after chunked prefill");
        reports.push(ParityReport::compute(
            &format!("chunk_{chunk}_decode"),
            &step.reshape(&[1, 1, -1]),
            &last,
            TOL,
        ));
    }
    assert_all_pass(&format!("{profile}: chunked prefill"), &reports);
}

#[test]
#[serial]
fn qwen3_moe_native_matches_transformers() {
    native_matches_transformers("qwen3_moe");
}

#[test]
#[serial]
fn qwen3_moe_without_topk_renormalization_and_with_a_dense_layer_matches_transformers() {
    native_matches_transformers("qwen3_moe_unnormalized");
}
