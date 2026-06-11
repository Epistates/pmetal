//! Numerical-parity test for the Rust Cohere (Command-R) port.
//!
//! Loads a tiny 2-layer seeded fixture dumped from `mlx_lm.models.cohere`
//! (`.strategy/parity/dump_cohere_reference.py`) through the *production*
//! weight loader and compares every tapped checkpoint against the reference
//! tensors, plus an argmax-exact check on the final logits.
//!
//! This is the regression guard for three forward-pass fixes:
//!   1. output `logit_scale` (0.0625) — applied via the LM head path,
//!   2. tied LM head — Cohere ships no `lm_head.weight`, so a separate head
//!      would emit random logits,
//!   3. traditional (interleaved) RoPE.
//!
//! Any of these regressing flips the argmax and blows the logits tolerance.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pmetal_bridge::compat::{Array, Module};
use pmetal_mlx::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, print_report_table, to_f32_vec_eval,
};

use pmetal_models::architectures::cohere::{CohereConfig, CohereForCausalLM};
use pmetal_models::loader::load_generic_weights;

/// Synthetic config — must mirror SYNTHETIC_ARGS in the Python dumper.
/// `use_sliding_window` is omitted (defaults false) so the synthetic run is a
/// plain global-causal forward, matching mlx-lm's `create_attention_mask`.
fn synthetic_config_json() -> &'static str {
    r#"{
        "vocab_size": 512,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 8192,
        "rope_theta": 10000.0,
        "layer_norm_eps": 1e-5,
        "logit_scale": 0.0625,
        "tie_word_embeddings": true
    }"#
}

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

fn load_shard(path: &Path) -> HashMap<String, Array> {
    let path_str = path.to_str().expect("utf8 path");
    pmetal_bridge::inline_array::load_safetensors_shard(path_str)
        .unwrap_or_else(|| panic!("failed to load safetensors shard at {path_str:?}"))
        .into_iter()
        .collect()
}

fn ref_tensor<'a>(shard: &'a HashMap<String, Array>, key: &str) -> &'a Array {
    shard
        .get(key)
        .unwrap_or_else(|| panic!("reference shard missing key {key:?}"))
}

/// Copy the checked-in weight fixture into a fresh temp dir named
/// `model.safetensors` and load it via the production `load_generic_weights`
/// path — exercising the same silent-skip loader the real models use, which
/// is what makes the tied-head bug observable (a stray `lm_head` would just
/// stay random).
fn build_model_via_production_loader() -> CohereForCausalLM {
    let config: CohereConfig =
        json5::from_str(synthetic_config_json()).expect("synthetic config parses");
    let mut model = CohereForCausalLM::new(config).expect("cohere model builds");

    let tmp = std::env::temp_dir().join(format!("pmetal_cohere_parity_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("temp dir");
    let dst = tmp.join("model.safetensors");
    std::fs::copy(fixture_path("cohere_synth_weights.safetensors"), &dst).expect("copy weights");
    load_generic_weights(&mut model, &tmp).expect("production loader runs");
    let _ = std::fs::remove_dir_all(&tmp);
    model
}

#[test]
fn cohere_synthetic_parity() {
    let ref_shard = load_shard(&fixture_path("cohere_synth_reference.safetensors"));
    let input_ids = ref_tensor(&ref_shard, "input_ids").clone();

    let mut model = build_model_via_production_loader();

    // Manual layer walk for intermediate taps (CohereModel has no capture
    // hook). Stateless without a cache, so this is the same computation the
    // production forward performs.
    let post_embed = Module::forward(&mut model.model.embed_tokens, &input_ids).expect("embed");
    let mut h = post_embed.clone();
    let mut layer_taps: Vec<Array> = Vec::new();
    for layer in model.model.layers.iter_mut() {
        h = layer
            .forward_with_cache(&h, None, None)
            .expect("layer forward");
        layer_taps.push(h.clone());
    }
    let final_hidden = Module::forward(&mut model.model.norm, &h).expect("final norm");

    // Final logits through the *production* path (exercises tied-head +
    // logit_scale exactly as inference does).
    let logits = model
        .forward_with_cache(&input_ids, None, None)
        .expect("model forward");

    let tol = [
        ("post_embed", Tolerance::new(1e-4, 1e-4)),
        ("layer_0_hidden", Tolerance::new(5e-4, 1e-3)),
        ("layer_1_hidden", Tolerance::new(1e-3, 2e-3)),
        ("final_hidden", Tolerance::new(1.5e-3, 2e-3)),
        ("logits", Tolerance::new(5e-3, 5e-3)),
    ];
    let lookup = |name: &str| {
        tol.iter()
            .find(|(n, _)| *n == name)
            .map(|(_, t)| *t)
            .unwrap_or(Tolerance::new(1e-3, 1e-3))
    };

    let reports = vec![
        ParityReport::compute(
            "post_embed",
            &post_embed,
            ref_tensor(&ref_shard, "post_embed"),
            lookup("post_embed"),
        ),
        ParityReport::compute(
            "layer_0_hidden",
            &layer_taps[0],
            ref_tensor(&ref_shard, "layer_0_hidden"),
            lookup("layer_0_hidden"),
        ),
        ParityReport::compute_with_per_position(
            "layer_1_hidden",
            &layer_taps[1],
            ref_tensor(&ref_shard, "layer_1_hidden"),
            lookup("layer_1_hidden"),
        ),
        ParityReport::compute_with_per_position(
            "final_hidden",
            &final_hidden,
            ref_tensor(&ref_shard, "final_hidden"),
            lookup("final_hidden"),
        ),
        ParityReport::compute_with_per_position(
            "logits",
            &logits,
            ref_tensor(&ref_shard, "logits"),
            lookup("logits"),
        ),
    ];

    println!("\n== Cohere synthetic parity report ==");
    print_report_table(&reports);

    let argmax_rust = argmax_last_axis(&logits);
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

    let failures: Vec<_> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.clone())
        .collect();
    assert!(failures.is_empty(), "Cohere parity failed at: {failures:?}");
    assert_eq!(
        argmax_matches,
        argmax_rust.len(),
        "Cohere parity: argmax mismatch (rust={argmax_rust:?}, ref={argmax_ref:?})"
    );
}
