//! Gemma 4 checkpoint layouts: the module-based port (`DynamicModel`, what
//! training and the generic engines run) and the native engine must read the
//! layouts the releases ship, and must agree with each other on what they read.
//!
//! The committed `transformers` fixtures are in Hugging Face's layout
//! (`model.*`, experts fused as `gate_up_proj`) and dense. The checkpoints
//! people actually download for Apple Silicon are MLX conversions: the towers
//! beside the text backbone under `language_model.model.*`, experts stacked as
//! `switch_glu.{gate,up,down}_proj`, and (for the 4-bit and QAT releases) every
//! projection and embedding packed as `weight` / `scales` / `biases` with a
//! `quantization` block that may override single modules. This file re-lays the
//! fixtures out that way, so a loader that skips what it does not recognise
//! shows up as a logits mismatch instead of a model that runs and decodes noise.
//!
//! The oracle for the layout is the fixture itself, which `gemma4_parity.rs`
//! pins to `transformers`; the oracle for packing is the same weights
//! dequantized to dense, which computes the same function.

mod common;

use std::collections::BTreeMap;

use common::gemma4::{
    moe_synthetic_config_json, shared_synthetic_config_json, synthetic_config_json,
};
use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, print_report_table};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::Array;
use pmetal_runtime::gemma4_native;
use serial_test::serial;

const GROUP: i32 = 32;

/// How closely two correct readings of one checkpoint agree, relative to the
/// largest logit: `gemma4_parity.rs` holds each engine to 1.5e-5 of a logit
/// magnitude of 4e-2 against `transformers`, about 4e-4 relative.
const AGREE: Tolerance = Tolerance::new(0.0, 5e-4);

/// One checkpoint laid out the way an MLX conversion lays it out, twice: with
/// its tensors packed, and with the same tensors dequantized to dense.
struct Checkpoints {
    packed: tempfile::TempDir,
    dense: tempfile::TempDir,
    config_json: String,
    dense_tensors: Vec<(String, Array)>,
    input_ids: Array,
}

/// `module`'s `weight` as `(weight, scales, biases)` packed at `bits`, and as
/// the dense tensor those decode to.
fn pack(w: &Array, bits: i32) -> ((Array, Array, Array), Array) {
    let (q, scales, biases) = w.quantize_weights(GROUP, bits);
    let dense = q.dequantize(&scales, &biases, GROUP, bits);
    ((q, scales, biases), dense)
}

/// Lay `source` (a Hugging Face layout fixture) out as an MLX checkpoint.
fn mlx_checkpoints(config_json: &str, weights: &str, reference: &str) -> Checkpoints {
    let source = load_shard(&fixture_path(weights));
    let reference = load_shard(&fixture_path(reference));

    // `language_model.model.*`, experts stacked as `switch_glu`.
    let mut tensors: BTreeMap<String, Array> = BTreeMap::new();
    for (key, w) in &source {
        // ⚠️ The fixtures are blind to the branches this file exists to check:
        // their norm weights are around 0.05, so zeroing every expert moves
        // the logits by 2e-5 against a magnitude of 4e-2, the size of the
        // difference between two correct engines. A loader that skipped the
        // experts passed. Scaling the norms that sit behind the experts and the
        // logits lifts them clear of the noise; `moe_fixture_is_sensitive_to_its_experts`
        // holds it there.
        let w = if key == "model.norm.weight"
            || key.ends_with(".post_feedforward_layernorm_2.weight")
        {
            w.multiply(&Array::from_f32(40.0))
        } else {
            w.clone()
        };
        let w = &w;
        let key = format!("language_model.{key}");
        if let Some(layer) = key.strip_suffix(".experts.gate_up_proj") {
            let half = w.dim(1) / 2;
            let split = |from, to| pmetal_runtime::compat::ops::slice_axis(w, 1, from, to);
            tensors.insert(
                format!("{layer}.experts.switch_glu.gate_proj.weight"),
                split(0, half),
            );
            tensors.insert(
                format!("{layer}.experts.switch_glu.up_proj.weight"),
                split(half, 2 * half),
            );
        } else if let Some(layer) = key.strip_suffix(".experts.down_proj") {
            tensors.insert(
                format!("{layer}.experts.switch_glu.down_proj.weight"),
                w.clone(),
            );
        } else {
            tensors.insert(key, w.clone());
        }
    }

    // Every projection and embedding that divides into groups is packed, with
    // one module at 8 bits the way a QAT release keeps its sensitive ones.
    let mut packed: Vec<(String, Array)> = Vec::new();
    let mut dense: Vec<(String, Array)> = Vec::new();
    let mut quantization = serde_json::json!({"group_size": GROUP, "bits": 4});
    for (key, w) in tensors {
        let module = key.strip_suffix(".weight");
        match module {
            Some(module) if w.ndim() >= 2 && w.dim(w.ndim() - 1) % GROUP == 0 => {
                let bits = if module.ends_with("layers.0.mlp.gate_proj") {
                    quantization[module] = serde_json::json!({"group_size": GROUP, "bits": 8});
                    8
                } else {
                    4
                };
                let ((q, scales, biases), deq) = pack(&w, bits);
                packed.push((format!("{module}.weight"), q));
                packed.push((format!("{module}.scales"), scales));
                packed.push((format!("{module}.biases"), biases));
                dense.push((key, deq));
            }
            _ => {
                packed.push((key.clone(), w.clone()));
                dense.push((key, w));
            }
        }
    }

    Checkpoints {
        packed: write_checkpoint(config_json, &packed, Some(&quantization)),
        dense: write_checkpoint(config_json, &dense, None),
        config_json: config_json.to_string(),
        dense_tensors: dense,
        input_ids: ref_tensor(&reference, "input_ids").clone(),
    }
}

/// `tensors` as a multimodal-wrapped checkpoint directory.
fn write_checkpoint(
    text_config: &str,
    tensors: &[(String, Array)],
    quantization: Option<&serde_json::Value>,
) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut config = serde_json::json!({
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": serde_json::from_str::<serde_json::Value>(text_config).unwrap(),
    });
    if let Some(q) = quantization {
        config["quantization"] = q.clone();
    }
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config");
    let entries: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    )
    .expect("save the weights");
    dir
}

fn dynamic_logits(dir: &tempfile::TempDir, ids: &Array) -> Array {
    let mut model = DynamicModel::load(dir.path()).expect("the module-based port loads it");
    let logits = model.forward(ids, None).expect("forward");
    logits.eval();
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
    logits
}

fn native_logits(dir: &tempfile::TempDir, ids: &Array) -> Array {
    let config = gemma4_native::load_config(dir.path()).expect("native config parses");
    let weights = gemma4_native::load_model(dir.path(), &config).expect("native load");
    let mut cache = gemma4_native::build_cache(&weights, &config);
    let logits = gemma4_native::forward_step(&weights, ids, &mut cache);
    logits.eval();
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
    logits
}

fn assert_agree(label: &str, got: &Array, want: &Array, tol: Tolerance) {
    let report = ParityReport::compute_with_per_position(label, got, want, tol);
    print_report_table(std::slice::from_ref(&report));
    assert!(report.passed(), "{label}: outside {tol:?}");
}

/// What each layout of one fixture must satisfy:
/// * the module-based port reads the MLX layout (the dense twin) as the
///   native engine does, which `gemma4_parity.rs` pins to `transformers`;
/// * each engine, given the packed checkpoint, gives its own dense twin's logits;
/// * so the two engines agree on the packed checkpoint too.
fn check(label: &str, config_json: &str, weights: &str, reference: &str) {
    let ck = mlx_checkpoints(config_json, weights, reference);

    let native_dense = native_logits(&ck.dense, &ck.input_ids);
    let dynamic_dense = dynamic_logits(&ck.dense, &ck.input_ids);
    // The same bound `gemma4_parity.rs` holds each engine to against
    // `transformers`.
    assert_agree(
        &format!("{label}: dynamic vs native, dense"),
        &dynamic_dense,
        &native_dense,
        AGREE,
    );

    // Packing changes the summation order of every projection, nothing else.
    let native_packed = native_logits(&ck.packed, &ck.input_ids);
    assert_agree(
        &format!("{label}: native packed vs dense"),
        &native_packed,
        &native_dense,
        AGREE,
    );
    let dynamic_packed = dynamic_logits(&ck.packed, &ck.input_ids);
    assert_agree(
        &format!("{label}: dynamic packed vs dense"),
        &dynamic_packed,
        &dynamic_dense,
        AGREE,
    );
    assert_agree(
        &format!("{label}: dynamic vs native, packed"),
        &dynamic_packed,
        &native_packed,
        AGREE,
    );
}

#[test]
#[serial]
fn dense_checkpoint_in_the_mlx_layout() {
    check(
        "dense",
        synthetic_config_json(),
        "gemma4_synth_weights.safetensors",
        "gemma4_synth_reference.safetensors",
    );
}

/// The 26B-A4B shape: experts stacked as `switch_glu.*`, which the module-based
/// port used to skip, running every MoE layer on zeroed experts.
#[test]
#[serial]
fn moe_checkpoint_in_the_mlx_layout() {
    check(
        "MoE",
        moe_synthetic_config_json(),
        "gemma4_moe_synth_weights.safetensors",
        "gemma4_moe_synth_reference.safetensors",
    );
}

/// The E2B/E4B shape: per-layer input embeddings and KV-shared layers, which
/// ship no `k_proj`, `v_proj` or `k_norm` of their own.
#[test]
#[serial]
fn shared_kv_checkpoint_in_the_mlx_layout() {
    check(
        "shared-KV",
        shared_synthetic_config_json(),
        "gemma4_shared_synth_weights.safetensors",
        "gemma4_shared_synth_reference.safetensors",
    );
}

/// A tensor the module-based port was built to read but the checkpoint lacks
/// is an error naming it, not a layer left at its initial value.
#[test]
#[serial]
fn a_missing_trunk_tensor_is_refused() {
    let ck = mlx_checkpoints(
        synthetic_config_json(),
        "gemma4_synth_weights.safetensors",
        "gemma4_synth_reference.safetensors",
    );
    let victim = "language_model.model.layers.1.self_attn.q_proj.weight";
    assert!(
        ck.dense_tensors.iter().any(|(k, _)| k == victim),
        "the fixture has {victim}"
    );
    let without: Vec<(String, Array)> = ck
        .dense_tensors
        .iter()
        .filter(|(k, _)| k != victim)
        .cloned()
        .collect();
    let dir = write_checkpoint(&ck.config_json, &without, None);

    let err = match DynamicModel::load(dir.path()) {
        Ok(_) => panic!("a checkpoint without {victim} loaded"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("layers.1.self_attn.q_proj"), "{err}");
}

/// The layout checks above only mean something if a loader that dropped the
/// thing under test would fail them. Zeroing each part of the expert branch
/// must move the logits by far more than `AGREE` allows.
#[test]
#[serial]
fn moe_fixture_is_sensitive_to_its_experts() {
    let ck = mlx_checkpoints(
        moe_synthetic_config_json(),
        "gemma4_moe_synth_weights.safetensors",
        "gemma4_moe_synth_reference.safetensors",
    );
    let base = native_logits(&ck.dense, &ck.input_ids);
    let budget = AGREE.rtol * ParityReport::compute("m", &base, &base, AGREE).max_abs_ref;
    for needle in [
        "switch_glu.gate_proj",
        "switch_glu.up_proj",
        "switch_glu.down_proj",
        "router.proj",
    ] {
        let zeroed: Vec<(String, Array)> = ck
            .dense_tensors
            .iter()
            .map(|(k, v)| match k.contains(needle) {
                true => (k.clone(), v.multiply(&Array::from_f32(0.0))),
                false => (k.clone(), v.clone()),
            })
            .collect();
        let dir = write_checkpoint(&ck.config_json, &zeroed, None);
        let got = native_logits(&dir, &ck.input_ids);
        let moved = ParityReport::compute("z", &got, &base, AGREE).max_abs_diff;
        assert!(
            moved > 20.0 * budget,
            "zeroing {needle} moves the logits by {moved:e}, inside the {budget:e} two engines may differ by"
        );
    }
}
