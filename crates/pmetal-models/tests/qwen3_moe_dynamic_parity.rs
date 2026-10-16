//! The shared-path Qwen3-MoE (`DynamicModel::Qwen3MoE`, what training and the
//! adapters run) against the authoritative Hugging Face `transformers`
//! `Qwen3MoeForCausalLM`.
//!
//! The fixtures are the three-layer checkpoints the native engine's parity test
//! uses. A checkpoint reaches the loader in one of three layouts, and all three
//! must give the reference's logits:
//!
//! * `fused`: transformers 5's own save, `mlp.experts.gate_up_proj`
//!   (`[experts, 2 * intermediate, hidden]`, gate then up) and
//!   `mlp.experts.down_proj`.
//! * `per_expert`: the released Qwen3-30B-A3B layout,
//!   `mlp.experts.{e}.{gate,up,down}_proj.weight`.
//! * `stacked`: an MLX conversion, `mlp.switch_mlp.{gate,up,down}_proj.weight`
//!   with `[experts, out, in]`.
//!
//! Profiles: `qwen3_moe` (top-k weights renormalised) and
//! `qwen3_moe_unnormalized` (`norm_topk_prob` false, layer 1 a dense MLP).

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, print_report_table, to_f32_vec_eval,
};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{Array, ops::slice_axis};
use serial_test::serial;

/// fp32 against fp32; the fixtures' own fp32-vs-fp64 noise is 1.5e-6 to 1.7e-6,
/// and the gate is ~30x that. Experts that never run, or a gate/up swap, move
/// the logits by 1e-2.
const TOL: Tolerance = Tolerance::new(5e-5, 0.0);

#[derive(Clone, Copy, Debug)]
enum Layout {
    Fused,
    PerExpert,
    Stacked,
}

fn drain(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: runtime error: {e}"));
}

/// The fixture's tensors rewritten into `layout`.
fn relayout(weights: HashMap<String, Array>, layout: Layout) -> HashMap<String, Array> {
    if matches!(layout, Layout::Fused) {
        return weights;
    }
    let mut out = HashMap::new();
    for (key, value) in weights {
        let Some(layer) = key.strip_suffix(".mlp.experts.gate_up_proj") else {
            if let Some(layer) = key.strip_suffix(".mlp.experts.down_proj") {
                match layout {
                    Layout::PerExpert => {
                        for e in 0..value.dim(0) {
                            out.insert(
                                format!("{layer}.mlp.experts.{e}.down_proj.weight"),
                                slice_axis(&value, 0, e, e + 1).squeeze(0),
                            );
                        }
                    }
                    Layout::Stacked => {
                        out.insert(format!("{layer}.mlp.switch_mlp.down_proj.weight"), value);
                    }
                    Layout::Fused => unreachable!(),
                }
            } else {
                out.insert(key, value);
            }
            continue;
        };
        let half = value.dim(1) / 2;
        let gate = slice_axis(&value, 1, 0, half);
        let up = slice_axis(&value, 1, half, 2 * half);
        match layout {
            Layout::PerExpert => {
                for e in 0..value.dim(0) {
                    for (name, w) in [("gate_proj", &gate), ("up_proj", &up)] {
                        out.insert(
                            format!("{layer}.mlp.experts.{e}.{name}.weight"),
                            slice_axis(w, 0, e, e + 1).squeeze(0),
                        );
                    }
                }
            }
            Layout::Stacked => {
                out.insert(format!("{layer}.mlp.switch_mlp.gate_proj.weight"), gate);
                out.insert(format!("{layer}.mlp.switch_mlp.up_proj.weight"), up);
            }
            Layout::Fused => unreachable!(),
        }
    }
    out
}

struct Fixture {
    _dir: tempfile::TempDir,
    model: DynamicModel,
    reference: HashMap<String, Array>,
}

fn checkpoint(profile: &str, layout: Layout) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path(&format!("{profile}_config.json")),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    let weights = relayout(
        load_shard(&fixture_path(&format!("{profile}_weights.safetensors"))),
        layout,
    );
    let entries: Vec<(&str, &Array)> = weights.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    )
    .expect("save the checkpoint");
    drain("save");
    let model = DynamicModel::load(dir.path()).expect("the checkpoint loads");
    drain("load");
    Fixture {
        _dir: dir,
        model,
        reference: load_shard(&fixture_path(&format!("{profile}_reference.safetensors"))),
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

fn check(profile: &str, layout: Layout) {
    let mut fx = checkpoint(profile, layout);
    let ids = ref_tensor(&fx.reference, "input_ids").clone();
    let want = ref_tensor(&fx.reference, "logits").clone();
    let t = ids.dim(1);
    let title = format!("{profile} ({layout:?})");

    let logits = fx.model.forward(&ids, None).expect("forward");
    drain("forward");
    assert_all_pass(
        &format!("{title}: uncached prefill"),
        &[ParityReport::compute_with_per_position(
            "logits", &logits, &want, TOL,
        )],
    );
    assert_eq!(argmax_last_axis(&logits), argmax_last_axis(&want));

    // The same prompt through the KV cache: a prefill, then one token at a time.
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
    assert_all_pass(&format!("{title}: cached decode"), &reports);
}

#[test]
#[serial]
fn dynamic_qwen3_moe_fused_checkpoint_matches_transformers() {
    check("qwen3_moe", Layout::Fused);
}

#[test]
#[serial]
fn dynamic_qwen3_moe_released_per_expert_checkpoint_matches_transformers() {
    check("qwen3_moe", Layout::PerExpert);
}

#[test]
#[serial]
fn dynamic_qwen3_moe_mlx_stacked_checkpoint_matches_transformers() {
    check("qwen3_moe", Layout::Stacked);
}

#[test]
#[serial]
fn dynamic_qwen3_moe_without_renormalization_and_with_a_dense_layer_matches_transformers() {
    for layout in [Layout::Fused, Layout::PerExpert, Layout::Stacked] {
        check("qwen3_moe_unnormalized", layout);
    }
}

/// A fixture the experts barely move would pass whatever the loader did with
/// them. Zero the experts of the reference model's first layer and the logits
/// must leave the tolerance by a wide margin.
#[test]
#[serial]
fn qwen3_moe_fixture_is_sensitive_to_its_experts() {
    let reference = load_shard(&fixture_path("qwen3_moe_reference.safetensors"));
    let ids = ref_tensor(&reference, "input_ids").clone();
    let want = ref_tensor(&reference, "logits").clone();

    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path("qwen3_moe_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    let mut weights = load_shard(&fixture_path("qwen3_moe_weights.safetensors"));
    for (key, value) in weights.iter_mut() {
        if key.ends_with(".mlp.experts.down_proj") {
            *value = Array::zeros_f32(&value.shape().to_vec());
        }
    }
    let entries: Vec<(&str, &Array)> = weights.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    )
    .expect("save");
    let mut model = DynamicModel::load(dir.path()).expect("loads");
    let logits = model.forward(&ids, None).expect("forward");
    drain("forward");
    let report = ParityReport::compute("logits", &logits, &want, TOL);
    assert!(
        !report.passed(),
        "zeroed experts must be detectable by this fixture"
    );
    let got = to_f32_vec_eval(&logits);
    assert!(got.iter().all(|v| !v.is_nan()));
}
