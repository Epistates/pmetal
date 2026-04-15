//! Numerical-parity test for the Rust Gemma 4 port.
//!
//! Runs two profiles:
//!
//! 1. **Synthetic** (always) — loads a tiny 2-layer seeded fixture checked
//!    into `tests/fixtures/` and compares every tapped checkpoint against
//!    the reference tensors dumped by
//!    `.strategy/parity/dump_gemma4_reference.py`. Catches architectural
//!    bugs and weight-loader bugs under tight tolerances.
//! 2. **Real 31B** (only when `PMETAL_GEMMA4_REFERENCE` is set) — loads a
//!    reference safetensors file dumped from a `gemma-4-31B` checkpoint and
//!    compares it against the Rust forward of the same model. Tolerances
//!    loosened for 60-layer bf16 cumulative drift.
//!
//! Both profiles use the same helper functions from `pmetal_mlx::test_utils`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pmetal_bridge::compat::{Array, ops};
use pmetal_mlx::speculative::SpecCapture;
use pmetal_mlx::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, print_report_table, to_f32_vec_eval,
};

use pmetal_models::architectures::gemma4::{
    Gemma4Config, Gemma4ForCausalLM, load_gemma4_weights,
};

/// Load a safetensors file into a `HashMap<String, Array>` using the same
/// bridge loader the production weight-loading path uses.
fn load_shard(path: &Path) -> HashMap<String, Array> {
    let path_str = path.to_str().expect("utf8 path");
    let pairs = pmetal_bridge::inline_array::load_safetensors_shard(path_str)
        .unwrap_or_else(|| panic!("failed to load safetensors shard at {path_str:?}"));
    pairs.into_iter().collect()
}

/// Fetch a single tensor from the reference shard, panicking with a helpful
/// message if it's missing.
fn ref_tensor<'a>(shard: &'a HashMap<String, Array>, key: &str) -> &'a Array {
    shard
        .get(key)
        .unwrap_or_else(|| panic!("reference shard missing key {key:?}"))
}

/// Synthetic config that mirrors SYNTHETIC_ARGS in the Python dumper.
/// Fields must stay in sync with `.strategy/parity/dump_gemma4_reference.py`.
fn synthetic_config_json() -> &'static str {
    r#"{
        "model_type": "gemma4_text",
        "vocab_size": 512,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "global_head_dim": 64,
        "num_global_key_value_heads": 1,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "attention_k_eq_v": true,
        "tie_word_embeddings": true,
        "sliding_window": 8,
        "final_logit_softcapping": 30.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional"
            },
            "sliding_attention": {
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000.0,
                "rope_type": "default"
            }
        },
        "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0,
        "use_double_wide_mlp": false,
        "enable_moe_block": false
    }"#
}

/// Tolerance table used by the synthetic test.
/// Values are tightened ~10x vs the 31B profile because the 2-layer config
/// has minimal cumulative bf16 drift. The `OR` in `ParityReport::passed()`
/// means small-magnitude tensors don't also have to satisfy `rtol`.
fn synthetic_tolerances() -> Vec<(&'static str, Tolerance)> {
    vec![
        ("post_embed", Tolerance::new(1e-4, 1e-4)),
        ("layer_0_hidden", Tolerance::new(5e-4, 1e-3)),
        ("layer_1_hidden", Tolerance::new(1e-3, 2e-3)),
        ("final_hidden", Tolerance::new(1.5e-3, 2e-3)),
        ("logits", Tolerance::new(5e-3, 5e-3)),
    ]
}

/// Loosened tolerances for the 60-layer 31B run. Cumulative bf16 drift
/// dominates by the time we hit the final layers.
fn gemma4_31b_tolerances(tap_layers: &[i32]) -> Vec<(String, Tolerance)> {
    let mut t = vec![
        ("post_embed".to_string(), Tolerance::new(5e-3, 5e-3)),
        ("final_hidden".to_string(), Tolerance::new(1e-1, 5e-2)),
        ("logits".to_string(), Tolerance::new(2e-1, 5e-2)),
    ];
    for &idx in tap_layers {
        // Tighter bound for early layers, looser for late layers.
        let (atol, rtol) = match idx {
            0..=3 => (5e-3, 1e-2),
            4..=12 => (1.5e-2, 1.5e-2),
            13..=30 => (5e-2, 4e-2),
            _ => (1e-1, 5e-2),
        };
        t.push((format!("layer_{idx}_hidden"), Tolerance::new(atol, rtol)));
    }
    t
}

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// Build the Rust Gemma4 model from the synthetic config + synthetic weight
/// fixture and return it ready for forward passes.
fn build_synthetic_model() -> Gemma4ForCausalLM {
    let config: Gemma4Config = json5::from_str(synthetic_config_json())
        .expect("synthetic config parses");
    let mut model = Gemma4ForCausalLM::new(config).expect("synthetic model builds");
    let weights = load_shard(&fixture_path("gemma4_synth_weights.safetensors"));
    let report = load_gemma4_weights(&mut model, &weights)
        .expect("synthetic weight loader runs");
    assert!(
        report.loaded > 0,
        "synthetic weight loader loaded 0 tensors (skipped={:?})",
        report.skipped
    );
    model
}

/// Read `input_ids` from a reference shard. The dumper saves them as int32
/// `[1, T]`; we rematerialise as an Array with the same dtype.
fn input_ids_from_shard(shard: &HashMap<String, Array>) -> Array {
    let stored = ref_tensor(shard, "input_ids");
    // Copy to ensure the array is owned by this scope (the shard HashMap
    // will live as long as the test function anyway, but this matches the
    // production loader path which owns every tensor).
    stored.clone()
}

/// Compute and return a parity report for a single `(name, rust, ref)` triple
/// using the tolerance from the `tol_table` lookup.
fn compare_checkpoint(
    name: &str,
    rust: &Array,
    reference: &Array,
    tol_table: &[(&str, Tolerance)],
) -> ParityReport {
    let tol = tol_table
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| *t)
        .unwrap_or(Tolerance::new(1e-3, 1e-3));
    // Full-attention layers get the per-position breakdown so partial-RoPE
    // drift can be localised. Cheap — axis-1 is the seq dim, typically
    // length 4-16.
    if name == "layer_1_hidden" || name == "logits" || name == "final_hidden" {
        ParityReport::compute_with_per_position(name, rust, reference, tol)
    } else {
        ParityReport::compute(name, rust, reference, tol)
    }
}

/// Same as above but with owned-string names (used by the 31B path which
/// builds tolerance entries at runtime).
fn compare_checkpoint_owned(
    name: &str,
    rust: &Array,
    reference: &Array,
    tol_table: &[(String, Tolerance)],
) -> ParityReport {
    let tol = tol_table
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, t)| *t)
        .unwrap_or(Tolerance::new(5e-2, 5e-2));
    if name.starts_with("layer_") && name.ends_with("_hidden") {
        ParityReport::compute_with_per_position(name, rust, reference, tol)
    } else if name == "logits" || name == "final_hidden" {
        ParityReport::compute_with_per_position(name, rust, reference, tol)
    } else {
        ParityReport::compute(name, rust, reference, tol)
    }
}

#[test]
fn gemma4_synthetic_parity() {
    let ref_shard = load_shard(&fixture_path("gemma4_synth_reference.safetensors"));
    let input_ids = input_ids_from_shard(&ref_shard);

    let mut model = build_synthetic_model();

    // Use the production `forward_with_capture` path. That goes through
    // `Gemma4Attention::forward` with the internal mask-building logic and
    // whatever attention backend `fused_sdpa` benchmark-selects. Using this
    // path (rather than a manual layer walk) ensures the parity test
    // exercises the same code production code takes.
    let mut capture = SpecCapture::with_layers_and_embedding(vec![0, 1], true);
    let final_hidden_rust = model
        .model
        .forward_with_capture(&input_ids, None, None, Some(&mut capture))
        .expect("forward_with_capture runs");

    // Gemma4Model::forward_with_capture returns the post-norm hidden state.
    // Apply the tied embed LM head + softcap here so we can diff the final
    // logits against the reference.
    let raw_logits = model.model.embed_tokens.as_linear(&final_hidden_rust);
    let cap = model.config.final_logit_softcapping.unwrap_or(30.0);
    let cap_arr = Array::from_f32(cap);
    let logits_rust = ops::tanh(&raw_logits.divide(&cap_arr)).multiply(&cap_arr);

    // Pull captured taps out of SpecCapture.
    let post_embed_rust = capture
        .embedding
        .as_ref()
        .expect("embedding tap captured")
        .clone();
    let layer0_rust = capture
        .hidden_states
        .get(&0)
        .expect("layer 0 tap captured")
        .clone();
    let layer1_rust = capture
        .hidden_states
        .get(&1)
        .expect("layer 1 tap captured")
        .clone();

    let tol = synthetic_tolerances();
    let mut reports = Vec::new();
    reports.push(compare_checkpoint(
        "post_embed",
        &post_embed_rust,
        ref_tensor(&ref_shard, "post_embed"),
        &tol,
    ));
    reports.push(compare_checkpoint(
        "layer_0_hidden",
        &layer0_rust,
        ref_tensor(&ref_shard, "layer_0_hidden"),
        &tol,
    ));
    reports.push(compare_checkpoint(
        "layer_1_hidden",
        &layer1_rust,
        ref_tensor(&ref_shard, "layer_1_hidden"),
        &tol,
    ));
    reports.push(compare_checkpoint(
        "final_hidden",
        &final_hidden_rust,
        ref_tensor(&ref_shard, "final_hidden"),
        &tol,
    ));
    reports.push(compare_checkpoint(
        "logits",
        &logits_rust,
        ref_tensor(&ref_shard, "logits"),
        &tol,
    ));

    println!("\n== Gemma 4 synthetic parity report ==");
    print_report_table(&reports);

    // Argmax-exact check (separate from the tolerance table — the argmax
    // count is a discrete metric, not an absolute diff).
    let argmax_rust = argmax_last_axis(&logits_rust);
    let argmax_ref = to_f32_vec_eval(ref_tensor(&ref_shard, "argmax_tokens"))
        .into_iter()
        .map(|v| v as i32)
        .collect::<Vec<i32>>();
    assert_eq!(
        argmax_rust.len(),
        argmax_ref.len(),
        "argmax length mismatch: rust={} ref={}",
        argmax_rust.len(),
        argmax_ref.len()
    );
    let argmax_matches = argmax_rust
        .iter()
        .zip(argmax_ref.iter())
        .filter(|(a, b)| a == b)
        .count();
    println!(
        "argmax exact matches: {} / {}",
        argmax_matches,
        argmax_rust.len()
    );

    let mut failures = Vec::new();
    for r in &reports {
        if !r.passed() {
            failures.push(r.name.clone());
        }
    }
    assert!(
        failures.is_empty(),
        "Gemma 4 synthetic parity failed at checkpoints: {failures:?}"
    );
    assert_eq!(
        argmax_matches,
        argmax_rust.len(),
        "Gemma 4 synthetic parity: argmax mismatch at some positions"
    );
}

/// Real 31B parity run, gated behind `PMETAL_GEMMA4_REFERENCE`.
///
/// Set the env var to the path of a reference safetensors file produced by
/// `.strategy/parity/dump_gemma4_reference.py --model <gemma-4-31B checkpoint>`.
/// The sidecar `.meta.json` next to it is read for `model_id` (used to
/// load the real weights) and `tap_layers`.
#[test]
fn gemma4_31b_parity() {
    let ref_path = match std::env::var("PMETAL_GEMMA4_REFERENCE") {
        Ok(v) => PathBuf::from(v),
        Err(_) => {
            eprintln!(
                "PMETAL_GEMMA4_REFERENCE not set — skipping gemma4_31b_parity. \
                 To run: set it to a safetensors file dumped by \
                 .strategy/parity/dump_gemma4_reference.py."
            );
            return;
        }
    };
    if !ref_path.exists() {
        panic!(
            "PMETAL_GEMMA4_REFERENCE points at {ref_path:?} but the file does not exist"
        );
    }
    let meta_path = {
        let mut p = ref_path.clone();
        let fname = p
            .file_name()
            .and_then(|s| s.to_str())
            .expect("ref_path has file name");
        p.set_file_name(format!("{fname}.meta.json"));
        p
    };
    let meta: serde_json::Value = {
        let text = std::fs::read_to_string(&meta_path)
            .unwrap_or_else(|e| panic!("failed to read sidecar meta {meta_path:?}: {e}"));
        serde_json::from_str(&text).expect("sidecar meta parses")
    };
    let model_id = meta["model_id"]
        .as_str()
        .expect("sidecar meta missing model_id")
        .to_string();
    let tap_layers: Vec<i32> = meta["tap_layers"]
        .as_array()
        .expect("sidecar meta missing tap_layers")
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();

    let ref_shard = load_shard(&ref_path);
    let input_ids = input_ids_from_shard(&ref_shard);

    // Load the real model via the dispatcher so we take the same load path
    // the dflash smoke test takes. This only works with a local directory —
    // if `model_id` is an HF id, the test expects it to be resolvable via
    // HF cache or that the user resolved it manually before setting the env
    // var.
    let model_path = PathBuf::from(&model_id);
    if !model_path.exists() {
        eprintln!(
            "Real-weights path {model_path:?} does not exist locally. \
             Pass an absolute path to a local clone, or adjust the dumper \
             to also store the resolved HF cache path. Skipping."
        );
        return;
    }
    let dyn_model = pmetal_models::dispatcher::DynamicModel::load(&model_path)
        .expect("dispatcher loads gemma4 model");
    let mut gemma4 = match dyn_model {
        pmetal_models::dispatcher::DynamicModel::Gemma4(m) => m,
        other => panic!("expected Gemma4, got {other:?}"),
    };

    // Build capture with the full tap-layer list + embedding tap.
    let tap_usize: Vec<usize> = tap_layers.iter().map(|&i| i as usize).collect();
    let mut capture = SpecCapture::with_layers_and_embedding(tap_usize.clone(), true);

    let hidden_rust = gemma4
        .model
        .forward_with_capture(&input_ids, None, None, Some(&mut capture))
        .expect("real forward_with_capture runs");
    let final_hidden_rust = hidden_rust.clone();
    let raw_logits = gemma4.model.embed_tokens.as_linear(&final_hidden_rust);
    let cap = gemma4.config.final_logit_softcapping.unwrap_or(30.0);
    let cap_arr = Array::from_f32(cap);
    let logits_rust = ops::tanh(&raw_logits.divide(&cap_arr)).multiply(&cap_arr);

    let tol = gemma4_31b_tolerances(&tap_layers);
    let mut reports = Vec::new();

    if let Some(arr) = capture.embedding.as_ref() {
        reports.push(compare_checkpoint_owned(
            "post_embed",
            arr,
            ref_tensor(&ref_shard, "post_embed"),
            &tol,
        ));
    }
    for &idx in &tap_layers {
        let key = format!("layer_{idx}_hidden");
        if let Some(arr) = capture.hidden_states.get(&(idx as usize)) {
            reports.push(compare_checkpoint_owned(
                &key,
                arr,
                ref_tensor(&ref_shard, &key),
                &tol,
            ));
        }
    }
    reports.push(compare_checkpoint_owned(
        "final_hidden",
        &final_hidden_rust,
        ref_tensor(&ref_shard, "final_hidden"),
        &tol,
    ));
    reports.push(compare_checkpoint_owned(
        "logits",
        &logits_rust,
        ref_tensor(&ref_shard, "logits"),
        &tol,
    ));

    println!("\n== Gemma 4 31B parity report ({model_id}) ==");
    print_report_table(&reports);

    // Argmax: allow one position off for bf16 precision near tied tokens.
    let argmax_rust = argmax_last_axis(&logits_rust);
    let argmax_ref = to_f32_vec_eval(ref_tensor(&ref_shard, "argmax_tokens"))
        .into_iter()
        .map(|v| v as i32)
        .collect::<Vec<i32>>();
    let argmax_matches = argmax_rust
        .iter()
        .zip(argmax_ref.iter())
        .filter(|(a, b)| a == b)
        .count();
    println!(
        "argmax exact matches: {} / {}",
        argmax_matches,
        argmax_rust.len()
    );

    let mut failures = Vec::new();
    for r in &reports {
        if !r.passed() {
            failures.push(r.name.clone());
        }
    }
    assert!(
        failures.is_empty(),
        "Gemma 4 31B parity failed at checkpoints: {failures:?}"
    );
    let allowed_mismatch = 1;
    assert!(
        argmax_rust.len() - argmax_matches <= allowed_mismatch,
        "Gemma 4 31B parity: argmax mismatch at >{allowed_mismatch} positions \
         (rust={argmax_rust:?}, ref={argmax_ref:?})"
    );
}

